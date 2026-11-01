"""The logic audit: Stash writes only when you ask, your answers stick, inputs checked."""

import pytest

pytest.importorskip("fastapi")

import peaks.web.app as app_mod  # noqa: E402
import peaks.web.service as svc_mod  # noqa: E402


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from peaks.config import Config

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("PEAKS_SCHEDULER", "0")
    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    made = {}
    real = svc_mod.Service

    class S(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made["svc"] = self
    monkeypatch.setattr(app_mod, "Service", S)
    return TestClient(app_mod.create_app(cfg)), made["svc"]


def test_a_save_that_promotes_to_legendaire_hands_back_an_undo(tmp_path, monkeypatch):
    c, svc = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(svc_mod.Service, "create_apex", lambda self, sid, t, end=None, tag=None: {"id": "m1"})
    monkeypatch.setattr(svc_mod.Service, "curation_settings", lambda self: {"auto_legendaire_on_save": True})
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: type("C", (), {
        "scene_details": lambda _s, ids: {"7": {"rating100": 80, "o_counter": 17}}})())
    prev = {"rating100": 80, "o_counter": 17, "tag_ids": ["3"], "organized": True}
    monkeypatch.setattr(svc_mod.Service, "grade_scene", lambda self, sid, g, source="": {"previous": prev})
    r = c.post("/api/scene/7/apex", params={"t": 12.0}).json()
    assert r["promoted_from"] and r["previous"] == prev


def test_a_pick_you_answered_isnt_asked_again_after_it_moves(tmp_path, monkeypatch):
    c, svc = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(svc_mod.Service, "answered_picks", lambda self, sid: [312.0])
    monkeypatch.setattr(svc_mod.Service, "scene_picks", lambda self, sid: [
        {"t": 313.4}, {"t": 330.0, "start": 300.0, "end": 340.0}, {"t": 500.0}])
    got = [p["answered"] for p in c.get("/api/scene/7/picks").json()]
    assert got == [True, True, False]


# --- no lost updates ---------------------------------------------------------------------------

def test_two_settings_saves_at_once_both_land(tmp_path, monkeypatch):
    import threading

    c, svc = _client(tmp_path, monkeypatch)
    real = svc_mod.Service._settings
    gate = threading.Event()

    def slow_settings(self):
        d = real(self)
        if threading.current_thread().name == "slow":
            gate.wait(0.3)                      # the other save lands in this window
        return d
    monkeypatch.setattr(svc_mod.Service, "_settings", slow_settings)
    t = threading.Thread(target=lambda: svc.save_hear_settings(on=False), name="slow")
    t.start()
    svc.save_motion_settings(on=False)
    gate.set()
    t.join()
    svc._settings_cache = None                                # as read back from disk
    assert svc.motion_settings()["on"] is False and svc.hear_settings()["on"] is False


def test_sense_records_from_two_jobs_both_survive(tmp_path, monkeypatch):
    import threading

    c, svc = _client(tmp_path, monkeypatch)
    ts = [threading.Thread(target=svc._sense_record, args=(k, {"done": i})) for i in range(20) for k in ("hear", "motion")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert svc.sense_last_run("hear") and svc.sense_last_run("motion")


def test_every_index_build_gets_its_own_number():
    from peaks.search import SearchIndex

    a = SearchIndex(None, "m")
    gen = a.gen
    del a
    b = SearchIndex(None, "m")
    assert b.gen != gen                         # id() may repeat after a rebuild; this never does


# --- inputs mean what they say ---------------------------------------------------------------

def test_the_text_false_is_off_and_bad_numbers_are_400s(tmp_path, monkeypatch):
    from peaks.web.inputs import as_bool, as_int

    assert as_bool("false") is False and as_bool("on") is True and as_bool(0) is False
    with pytest.raises(ValueError):
        as_bool("maybe")
    with pytest.raises(ValueError):
        as_int("lots", "per_run")
    c, svc = _client(tmp_path, monkeypatch)
    assert c.post("/api/hear/settings", json={"on": "false"}).json()["on"] is False
    assert c.post("/api/motion/settings", json={"per_run": "lots"}).status_code == 400
    assert c.post("/api/prefs/pick-prompts", json={"on": "false"}).json()["on"] is False
    assert c.post("/api/label", params={"t": 1, "label": 2, "key": "k"}).status_code == 400
    assert c.get("/api/search/text", params={"q": "x", "top_k": 10_000_000}).status_code == 422
    r = c.post("/api/zoom/settings", json={"on": False}).json()
    assert r["on"] is False and svc.zoom_settings()["on"] is False


def test_seedbox_switches_parse_strictly_and_backup_times_are_bounded(tmp_path, monkeypatch):
    c, svc = _client(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        svc.save_seedjob_settings({"categories": ["oops"]})
    out = svc.save_seedjob_settings({"pull_on": "false", "cutover": "false"})
    s = out["settings"] if "settings" in out else svc._settings()["seedjobs"]
    assert s["pull_on"] is False and s["cutover"] is False
    with pytest.raises(ValueError):
        svc.save_backup_settings(backup_day=9)
