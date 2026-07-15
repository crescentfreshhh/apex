from peaks.failures import FailureLog, failure_log_for


def test_record_entries_resolve_clear(tmp_path):
    log = FailureLog(tmp_path / "failures.json")
    assert len(log) == 0 and log.entries() == []

    log.record("fp1", "1", "/data/Rando/a.mp4", error="boom", mode="sparse",
               hwaccel="cuda", pipeline="raw", model="dinov2")
    log.record("fp2", "2", "/data/Rando/b.mp4", error="nope")
    assert len(log) == 2
    e = {x["key"]: x for x in log.entries()}
    assert e["fp1"]["error"] == "boom" and e["fp1"]["mode"] == "sparse"
    assert e["fp1"]["scene_id"] == "1"

    # re-recording the same key overwrites (no duplicate)
    log.record("fp1", "1", "/data/Rando/a.mp4", error="boom2")
    assert len(log) == 2
    assert {x["key"]: x for x in log.entries()}["fp1"]["error"] == "boom2"

    assert log.resolve("fp1") is True
    assert log.resolve("fp1") is False  # already gone
    assert log.keys() == {"fp2"}

    assert log.clear() == 1
    assert len(log) == 0


def test_corrupt_log_is_treated_as_empty(tmp_path):
    p = tmp_path / "failures.json"
    p.write_text("{ this is not valid json")
    log = FailureLog(p)
    assert log.entries() == [] and len(log) == 0
    log.record("k", "1", "/x.mp4", error="e")  # still writable
    assert len(log) == 1


def test_failure_log_for_sits_beside_cache(tmp_path):
    class _Cfg:
        class embedding:
            cache_dir = str(tmp_path / "cache" / "embeddings")

    log = failure_log_for(_Cfg())
    assert log.path == tmp_path / "cache" / "failures.json"
