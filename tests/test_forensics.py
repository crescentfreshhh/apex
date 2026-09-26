"""Crash forensics: a previous run that didn't end with the clean-exit marker is
reported (with its last health readings and any native trace); the health line
formats without GPU tools; the endpoints serve the report and the log."""

import faulthandler

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from peaks.web import forensics  # noqa: E402


def _log(tmp_path, lines):
    p = tmp_path / "crash.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_clean_exit_means_no_report(tmp_path):
    p = _log(tmp_path, ["2026-09-26T10:00:00 [start] peaks 1 pid=7",
                        "2026-09-26T10:00:30 [health] rss=1.0G",
                        "2026-09-26T10:01:00 [clean-exit]"])
    assert forensics.previous_run_report(p) is None


def test_abrupt_end_reports_last_health_and_trace(tmp_path):
    p = _log(tmp_path, ["2026-09-26T09:00:00 [start] peaks 1 pid=5", "2026-09-26T09:05:00 [clean-exit]",
                        "2026-09-26T10:00:00 [start] peaks 1 pid=7",
                        "2026-09-26T10:00:30 [health] rss=3.0G/20.0G limit · ffmpeg=2",
                        "2026-09-26T10:01:00 [health] rss=19.0G/20.0G limit · ffmpeg=38"])
    r = forensics.previous_run_report(p)
    assert r["abrupt"] and r["started"] == "2026-09-26T10:00:00"
    assert r["ended_after"] == "2026-09-26T10:01:00" and len(r["last_health"]) == 2
    assert "OOM" in r["kind"] and r["trace"] == ""
    p.write_text(p.read_text() + "Fatal Python error: Segmentation fault\n\nCurrent thread 0x1:\n  File \"x.py\"\n")
    r = forensics.previous_run_report(p)
    assert "Segmentation fault" in r["trace"] and r["kind"].startswith("native crash")


def test_health_line_without_gpu_tools(monkeypatch):
    from peaks.web import memwatch

    monkeypatch.setattr(memwatch, "rss_bytes", lambda: 5 * 1073741824)
    monkeypatch.setattr(memwatch, "soft_limit_bytes", lambda: 20 * 1073741824)
    monkeypatch.setattr(forensics, "_descendants", lambda pid: [(1, "ffmpeg"), (2, "ffmpeg"), (3, "python")])
    monkeypatch.setattr(forensics, "_gpu", lambda: None)
    from collections import Counter
    forensics._state["routes"] = Counter()          # (other tests' requests counted too)
    forensics.note_request("/api/frame")
    forensics.note_request("/api/frame")
    line = forensics.health_line()
    assert "[health] rss=5.0G/20.0G limit" in line and "ffmpeg=2" in line and "children=3" in line
    assert "req=/api/frame×2" in line and "gpu" not in line


def test_enable_and_endpoints(tmp_path, monkeypatch):
    import peaks.web.app as app_mod
    from peaks.config import Config

    # a fresh forensics state writing into this test's settings dir
    monkeypatch.setattr(forensics, "_state", {**forensics._state, "fh": None, "path": None, "report": None})
    _log(tmp_path, ["2026-09-26T10:00:00 [start] peaks 1 pid=7", "2026-09-26T10:00:30 [health] rss=9.9G"])
    monkeypatch.setenv("PEAKS_HEALTH_SEC", "0")
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    client = TestClient(app_mod.create_app(cfg))
    assert faulthandler.is_enabled()
    r = client.get("/api/crash-report").json()
    assert r["abrupt"] and r["last_health"] == ["2026-09-26T10:00:30 [health] rss=9.9G"]
    assert "[start]" in client.get("/api/crashlog").text
    client.post("/api/crash-report/dismiss")
    assert client.get("/api/crash-report").json()["abrupt"] is False
