"""Same-file copies: ONE scene with the same download attached twice (Stash's
"File count > 1"). Only same size + same content hash counts; the extra FILES
go (never the scene, never its last file), the copy in the tier's folder /
with the clean name / primary / oldest stays. First run only lists."""

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402

GB = 1_000_000_000
LEG = (100, 18)


def _f(fid, path, size=GB, oshash="aa", **kw):
    return {"id": fid, "path": path, "size": size, "oshash": oshash, **kw}


@pytest.fixture
def stash():
    return FakeStash({
        # plain copy: Stash's collision rename "None_2" goes, the clean name stays
        "1": {"files": [_f("11", "/data/Downloads/None_2.mp4"), _f("12", "/data/Downloads/Clip.mp4")]},
        # Légendaire scene: the copy outside the Légendaire folder goes even though it's primary
        "2": {"rating100": LEG[0], "o_counter": LEG[1],
              "files": [_f("21", "/data/Downloads/Clip.mp4", oshash="bb"),
                        _f("22", "/data/Légendaire/Clip.mp4", oshash="bb")]},
        # same size, different hash → needs a look, nothing deleted
        "3": {"files": [_f("31", "/data/a/x.mp4", oshash="c1"), _f("32", "/data/a/x_1.mp4", oshash="c2")]},
        # same size, hash unknown → needs a look
        "4": {"files": [_f("41", "/data/b/y.mp4", oshash=None), _f("42", "/data/b/y (1).mp4", oshash=None)]},
        # different sizes → a different version, your call
        "5": {"files": [_f("51", "/data/c/z.mp4", size=GB), _f("52", "/data/c/z_1080.mp4", size=2 * GB)]},
        # one file: nothing to do
        "6": {"files": [_f("61", "/data/d/w.mp4")]},
        # three identical copies, md5 only: two go, the oldest primary-less clean one stays
        "7": {"files": [_f("71", "/data/e/v (1).mp4", oshash=None, md5="m"),
                        _f("72", "/data/e/v.mp4", oshash=None, md5="m", mod_time="2020"),
                        _f("73", "/data/e/copy of v.mp4", oshash=None, md5="m")]},
    })


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setenv("PEAKS_CLEANUP_ROOT", str(tmp_path / "data"))
    return svc_mod.Service(cfg)


def _plan(r, sid):
    return next(p for p in r["plans"] if p["scene_id"] == sid)


def test_finds_only_identical_copies_and_picks_the_right_keeper(svc):
    r = svc.find_file_copies()
    assert {p["scene_id"] for p in r["plans"]} == {"1", "2", "7"}
    p1 = _plan(r, "1")
    assert p1["keep"]["path"].endswith("Clip.mp4") and [d["id"] for d in p1["delete"]] == ["11"]
    assert "cleaner name" in p1["why"]
    p2 = _plan(r, "2")
    assert p2["keep"]["id"] == "22" and not p2["keep"]["primary"] and [d["id"] for d in p2["delete"]] == ["21"]
    assert "Légendaire folder" in p2["why"]
    p7 = _plan(r, "7")
    assert p7["keep"]["id"] == "72" and sorted(d["id"] for d in p7["delete"]) == ["71", "73"]
    assert {n["scene_id"] for n in r["needs_look"]} == {"3", "4"}
    assert [v["scene_id"] for v in r["versions"]] == ["5"]
    assert r["files"] == 4 and r["bytes"] == 4 * GB


def test_first_run_lists_only_then_approval_removes(svc, stash):
    step = svc.file_copies_step(log=lambda *_: None)
    assert step["applied"] is False and step["files"] == 4
    assert not [c for c in stash.calls if c[0] in ("delete_files", "primary")]

    svc.save_copies_settings(approved=True)
    step = svc.file_copies_step(log=lambda *_: None)
    assert step["applied"] and step["removed"] == 4 and step["freed"] == 4 * GB
    # the keeper became primary BEFORE its copy was deleted
    i_primary = stash.calls.index(("primary", "2", "22"))
    i_delete = stash.calls.index(("delete_files", ("21",)))
    assert i_primary < i_delete
    assert [f["id"] for f in stash.s["2"]["files"]] == ["22"]
    assert [f["id"] for f in stash.s["1"]["files"]] == ["12"]
    # untouched: the ones that need a look, other versions, single files, every scene
    assert len(stash.s["3"]["files"]) == 2 and len(stash.s["4"]["files"]) == 2 and len(stash.s["5"]["files"]) == 2
    assert set(stash.s) == {"1", "2", "3", "4", "5", "6", "7"}
    assert not [c for c in stash.calls if c[0] in ("destroy", "update")]
    assert stash.s["2"]["rating100"] == 100 and stash.s["2"]["o_counter"] == 18
    logged = [e for e in svc.history(50) if e.get("action") == "file-copy"]
    assert len(logged) == 4 and all(e.get("kept") for e in logged)
    # a second run finds nothing left
    assert svc.file_copies_step()["files"] == 0


def test_turned_off_does_nothing(svc, stash):
    svc.save_copies_settings(on=False, approved=True)
    assert svc.file_copies_step() is None
    assert not [c for c in stash.calls if c[0] == "delete_files"]


def test_fallback_without_deletefiles_removes_only_under_the_library_root(svc, stash, tmp_path):
    stash.caps["deleteFiles"] = False
    root = tmp_path / "data"
    (root / "Downloads").mkdir(parents=True)
    (root / "Downloads" / "None_2.mp4").write_bytes(b"x")
    (root / "Downloads" / "Clip.mp4").write_bytes(b"x")
    stash.s = {"1": {"rating100": None, "o_counter": 0, "files": [
        _f("11", str(root / "Downloads" / "None_2.mp4")), _f("12", str(root / "Downloads" / "Clip.mp4"))]},
        "2": {"rating100": None, "o_counter": 0, "files": [
            _f("21", "/elsewhere/a.mp4", oshash="q"), _f("22", "/elsewhere/a_1.mp4", oshash="q")]}}
    svc.capabilities(refresh=True)
    r = svc.remove_file_copies(apply=True)
    assert r["method"] == "direct" and r["removed"] == 1
    assert not (root / "Downloads" / "None_2.mp4").exists() and (root / "Downloads" / "Clip.mp4").exists()
    assert [e["scene_id"] for e in r["errors"]] == ["2"]
    assert not [c for c in stash.calls if c[0] == "delete_files"]


def test_api_preview_approve_and_toggle(svc, monkeypatch, stash):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    r = api.post("/api/library/copies", params={"action": "preview"}).json()
    assert r["last"]["files"] == 4 and not r["last"]["applied"]
    st = api.get("/api/library/copies").json()
    assert st["approved"] is False and st["last"]["files"] == 4
    api.post("/api/library/copies", params={"action": "off"})
    assert api.get("/api/library/copies").json()["on"] is False
    api.post("/api/library/copies", params={"action": "on"})
    r = api.post("/api/library/copies", params={"action": "approve"}).json()
    assert api.get("/api/library/copies").json()["approved"] is True
    assert stash.s["1"]["files"][0]["id"] == "12" and len(stash.s["1"]["files"]) == 1


def test_client_reads_multi_file_scenes_and_falls_back_on_old_stash():
    from peaks.stash_client import StashClient, StashError

    sc = {"id": 9, "title": "T", "rating100": 100, "o_counter": 18, "files": [
        {"id": 1, "path": "/data/a.mp4", "basename": "a.mp4", "size": "5", "mod_time": "x",
         "fingerprints": [{"type": "oshash", "value": "h"}, {"type": "phash", "value": "p"}]},
        {"id": 2, "path": "/data/None_2.mp4", "basename": "None_2.mp4", "size": 5, "mod_time": "y",
         "fingerprints": [{"type": "oshash", "value": "h"}]}]}
    single = {"id": 10, "files": [{"id": 3, "path": "/data/b.mp4", "size": 1}]}

    class C(StashClient):
        def __init__(self, old):
            super().__init__(url="http://stash.test")
            self.old, self.calls = old, []

        def execute(self, query, variables=None):
            self.calls.append(variables)
            if self.old and "scene_filter" in variables:
                raise StashError("Unknown field file_count")
            return {"findScenes": {"scenes": [sc] + ([single] if self.old else [])}}

    for old in (False, True):
        c = C(old)
        r = c.multi_file_scenes()
        assert [s["id"] for s in r] == ["9"]
        assert r[0]["files"][0] == {"id": "1", "path": "/data/a.mp4", "basename": "a.mp4", "size": 5,
                                    "mod_time": "x", "fingerprints": {"oshash": "h", "phash": "p"}}
    assert C(False).capabilities  # deleteFiles is checked like every other op
    assert "deleteFiles" in StashClient.CAPABILITY_FIELDS
