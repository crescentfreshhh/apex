"""Ingest: scan (phashes forced on) → identify → auto tag, each with Stash's
saved task defaults and waited on, then a Peaks embed of just the new scenes and
a duplicate check on them. The Stash client builds its queries/inputs from
schema introspection, tested against a canned GraphQL schema."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fakestash import FakeStash  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.stash_client import StashClient  # noqa: E402


@pytest.fixture
def stash():
    s = FakeStash({"1": {"rating100": 100, "o_counter": 18, "width": 3840, "height": 2160},
                   "2": {}})
    s.arriving = {"10": {"width": 3840, "height": 2160, "bit_rate": 50e6},
                  "11": {"width": 1920, "height": 1080}}
    s.defaults = {"scan": {"scanGenerateCovers": True, "scanGeneratePhashes": False},
                  "identify": {"sources": [{"source": {"stash_box_endpoint": "https://stashdb.org/graphql"}}],
                               "options": {"setOrganized": False}, "paths": ["/x"]},
                  "autoTag": None}
    s.dupes = [["1", "10"]]
    return s


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "INGEST_POLL", 0.0)
    embedded = []
    def fake_embed(self, job=None, **kw):
        embedded.append(kw.get("scene_ids"))
        stash.calls.append(("embed", sorted(kw.get("scene_ids") or [])))
        return {"embedded": len(kw.get("scene_ids") or []), "failed": 0}
    monkeypatch.setattr(svc_mod.Service, "run_embed", fake_embed)
    return svc_mod.Service(cfg)


def _api(svc, monkeypatch):
    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    return TestClient(app_mod.create_app(svc.cfg))


def _wait(client, job):
    for _ in range(300):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _kinds(stash):
    return [c[0] for c in stash.calls if c[0] in ("scan", "identify", "auto_tag", "embed", "dupes")]


def test_full_ingest_order_and_inputs(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    j = _wait(client, client.post("/api/ingest").json())
    assert j["status"] == "done", j
    assert _kinds(stash) == ["scan", "identify", "auto_tag", "embed", "dupes"]
    calls = {c[0]: c for c in stash.calls}
    # the user's scan settings (covers + video phashes, nothing else), not Stash's saved ones
    assert calls["scan"][1] == {**{k: False for k in svc.INGEST_SCAN_FIELDS},
                                "scanGenerateCovers": True, "scanGeneratePhashes": True}
    ident = calls["identify"][1]
    assert ident["sceneIDs"] == ["10", "11"] and "paths" not in ident                      # new scenes only
    assert ident["sources"][0]["source"]["stash_box_endpoint"].startswith("https://stashdb")
    at = calls["auto_tag"][1]
    assert at["performers"] == ["*"] and at["studios"] == ["*"] and at["tags"] == ["*"]    # nothing saved → all
    assert at["paths"] == ["/data/10.mp4", "/data/11.mp4"]
    assert calls["embed"][1] == ["10", "11"]
    r = j["result"]
    assert r["new"] == 2 and r["duplicates"] == 1

    # the catalogue's New filter shows exactly the ingested scenes, dupes flagged
    d = client.get("/api/catalogue", params={"new": True, "tier": ""}).json()
    assert sorted(i["scene_id"] for i in d["items"]) == ["10", "11"]
    assert {i["scene_id"]: i["dupe"] for i in d["items"]} == {"10": True, "11": False}
    assert client.get("/api/ingest").json()["last"]["new"] == ["10", "11"]
    assert any(e["action"] == "ingest" for e in svc.history())
    # the duplicate found during ingest is already in the Duplicates view
    assert [g["keep"] for g in client.get("/api/duplicates").json()["groups"]] == ["10"]


def test_saved_auto_tag_choice_is_respected_and_identify_skipped_without_sources(svc, stash):
    stash.defaults = {"scan": None, "identify": {"sources": []}, "autoTag": {"performers": ["*"], "studios": [], "tags": None}}
    out = svc.run_ingest()
    assert _kinds(stash) == ["scan", "auto_tag", "embed", "dupes"]
    at = next(c[1] for c in stash.calls if c[0] == "auto_tag")
    assert at["performers"] == ["*"] and "studios" not in at and "tags" not in at
    assert out["stages"]["identify"].startswith("skipped")


def test_a_failed_stage_stops_the_chain(svc, stash, monkeypatch):
    stash.job_outcome = {"identify": "FAILED"}
    client = _api(svc, monkeypatch)
    j = _wait(client, client.post("/api/ingest").json())
    assert j["status"] == "error" and "identify failed" in j["error"]
    assert _kinds(stash) == ["scan", "identify"]


def test_nothing_new_runs_only_the_scan(svc, stash):
    stash.arriving = {}
    out = svc.run_ingest()
    assert _kinds(stash) == ["scan"] and out["new"] == 0


def test_embed_skipped_while_an_embed_pass_runs(svc, stash):
    out = svc.run_ingest(embed_busy=lambda: True)
    assert "embed" not in _kinds(stash) and out["stages"]["embed"].startswith("skipped")


def test_ingest_refused_without_scan_support(svc, stash, monkeypatch):
    stash.caps["metadataScan"] = False
    assert _api(svc, monkeypatch).post("/api/ingest").status_code == 501


# --- the client's introspection-driven queries --------------------------------------

def _t(kind, name=None, of=None):
    return {"kind": kind, "name": name, "ofType": of}


SCHEMA = {
    "ConfigDefaultSettingsResult": {"fields": [
        {"name": "scan", "type": _t("OBJECT", "ScanMetadataOptions")},
        {"name": "autoTag", "type": _t("OBJECT", "AutoTagMetadataOptions")},
        {"name": "deleteFile", "type": _t("SCALAR", "Boolean")}]},
    "ScanMetadataOptions": {"fields": [
        {"name": "scanGenerateCovers", "type": _t("NON_NULL", None, _t("SCALAR", "Boolean"))},
        {"name": "scanGeneratePhashes", "type": _t("SCALAR", "Boolean")}]},
    "AutoTagMetadataOptions": {"fields": [
        {"name": "performers", "type": _t("LIST", None, _t("NON_NULL", None, _t("SCALAR", "String")))}]},
    "IdentifyMetadataInput": {"inputFields": [
        {"name": "sources", "type": _t("LIST", None, _t("INPUT_OBJECT", "IdentifySourceInput"))},
        {"name": "sceneIDs", "type": _t("LIST", None, _t("SCALAR", "ID"))}]},
    "IdentifySourceInput": {"inputFields": [
        {"name": "source", "type": _t("INPUT_OBJECT", "ScraperSourceInput")}]},
    "ScraperSourceInput": {"inputFields": [
        {"name": "stash_box_endpoint", "type": _t("SCALAR", "String")}]},
    "Job": {"fields": [{"name": n, "type": _t("SCALAR", "String")} for n in ("id", "status", "progress")]},
}


class _Schema(StashClient):
    def __init__(self):
        super().__init__(url="http://stash.test")
        self.queries = []

    def execute(self, query, variables=None):
        self.queries.append(query)
        if "__type" in query:
            return {"__type": SCHEMA.get(variables["name"])}
        if "configuration" in query:
            return {"configuration": {"defaults": {"scan": {"scanGenerateCovers": True},
                                                   "autoTag": {"performers": ["*"]}}}}
        if "findJob" in query:
            return {"findJob": {"id": "7", "status": "RUNNING", "progress": 0.3}}
        raise AssertionError(query)


def test_client_builds_queries_and_inputs_from_the_schema():
    c = _Schema()
    d = c.config_defaults()
    assert d == {"scan": {"scanGenerateCovers": True}, "identify": None, "autoTag": {"performers": ["*"]}}
    q = c.queries[-1]
    assert "scan { scanGenerateCovers scanGeneratePhashes }" in q and "deleteFile" not in q
    # output-only / renamed fields are dropped; nested inputs are reshaped
    fitted = c.fit_input({"sources": [{"source": {"stash_box_endpoint": "e", "name": "StashDB"},
                                       "extra": 1}], "paths": None, "bogus": 3}, "IdentifyMetadataInput")
    assert fitted == {"sources": [{"source": {"stash_box_endpoint": "e"}}]}
    j = c.find_job("7")
    assert j["status"] == "RUNNING" and "error" not in c.queries[-1]       # this Stash has no Job.error


def test_scan_options_saved_locked_and_fitted_to_the_schema(svc, stash, monkeypatch):
    client = _api(svc, monkeypatch)
    got = client.post("/api/ingest/scan-options", json={"options": {
        "scanGenerateSprites": True, "scanGeneratePhashes": False,      # phashes can't go off
        "scanGenerateImagePreviews": True, "bogus": True}}).json()["options"]
    assert got["scanGenerateSprites"] and got["scanGeneratePhashes"]
    assert got["scanGenerateImagePreviews"] is False                    # previews are off
    assert "bogus" not in got
    assert client.get("/api/ingest/scan-options").json()["labels"]["rescan"] == "rescan files"
    # an older Stash without image phashes: that field is left out of the input
    monkeypatch.setattr(stash, "input_has", lambda t, f: f != "scanGenerateImagePhashes", raising=False)
    svc.run_ingest()
    scan = next(c[1] for c in stash.calls if c[0] == "scan")
    assert "scanGenerateImagePhashes" not in scan
    assert scan["scanGenerateSprites"] and scan["scanGenerateCovers"] and scan["scanGeneratePhashes"]
