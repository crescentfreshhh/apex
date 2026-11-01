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
