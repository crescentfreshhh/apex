"""The Seedbox page: runs.log parsing, the GREEN / YELLOW / RED rules, pool and
throughput maths, unavailable sources, and a qBittorrent client that only ever
logs in and GETs."""

import pytest

from peaks.seedbox import (DAY, QbtClient, QbtError, parse_runs, pool_summary,
                           script_health, throughput, verdict)

NOW = 1_760_000_000.0
H = 3600.0


def _log(*rows):
    return "\n".join("|".join(str(x) for x in r) for r in rows) + "\n"


# --- runs.log ----------------------------------------------------------------------

def test_parse_runs_reads_pull_and_prune_and_skips_garbage():
    text = _log((NOW - 100, "pull", "ok", 3), (NOW - 50, "prune", "ok", ""), (NOW - 10, "pull", "fail", 0)) + \
        "garbage\n\n1|pull|ok\nabc|pull|ok|1\n5|push|ok|1\n6|pull|maybe|1\n7|pull|ok|x\n"
    runs, bad = parse_runs(text)
    assert [(r.kind, r.ok, r.files) for r in runs] == [("pull", True, 3), ("prune", True, None), ("pull", False, 0)]
    assert bad == 6
    assert parse_runs("")[0] == [] and parse_runs(None)[1] == 0


def test_runs_are_sorted_oldest_first():
    runs, _ = parse_runs(_log((NOW, "pull", "ok", 1), (NOW - H, "pull", "ok", 2)))
    assert [r.files for r in runs] == [2, 1]


def test_script_health_last_ok_counts_and_recovery():
    runs, _ = parse_runs(_log((NOW - 30 * H, "pull", "fail", 0), (NOW - 3 * H, "pull", "fail", 0),
                              (NOW - 2 * H, "pull", "ok", 4), (NOW - 20 * 60, "pull", "ok", 1)))
    h = script_health(runs, "pull", NOW, failing=False)
    assert h["last_ok"] == NOW - 20 * 60 and h["last_ok_ago_min"] == 20
    assert h["ok_24h"] == 2 and h["failed_24h"] == 1 and h["recovered_24h"] is True
    assert script_health(runs, "pull", NOW, failing=True)["recovered_24h"] is False
    assert script_health(runs, "prune", NOW, failing=False)["last_ok"] is None


def test_throughput_24h_7d_and_per_day():
    runs, _ = parse_runs(_log((NOW - 8 * DAY, "pull", "ok", 100),          # outside the week
                              (NOW - 3 * DAY, "pull", "ok", 5),
                              (NOW - 2 * H, "pull", "ok", 2), (NOW - H, "pull", "fail", 1),
                              (NOW - H, "prune", "ok", "")))
    t = throughput(runs, NOW, tz_offset=0)
    assert t["last_24h"] == 3 and t["last_7d"] == 8
    assert len(t["per_day"]) == 7 and sum(d["files"] for d in t["per_day"]) == 8
    assert t["per_day"][3]["files"] == 5


# --- the pool -------------------------------------------------------------------------

def _t(cat, state, gb, age_days, done=True, name="x"):
    at = NOW - age_days * DAY
    return {"name": name, "category": cat, "state": state, "size": gb * 1e9,
            "added_on": at - 600, "completion_on": at if done else -1}


def test_pool_summary_by_category_states_and_waiting_prune():
    p = pool_summary([_t("1", "pausedUP", 2, 1), _t("vr", "pausedUP", 10, 5, name="old vr"),
                      _t("1", "uploading", 3, 1), _t("other", "pausedUP", 1, 9),
                      _t("vr", "stalledUP", 4, 0.5), _t("1", "downloading", 1, 0, done=False)], NOW)
    assert p["total"] == {"count": 6, "gb": 21.0}
    assert p["by_category"]["1"] == {"count": 3, "gb": 6.0} and p["by_category"]["vr"]["count"] == 2
    assert p["seeding"] == 2 and p["paused"] == 3 and p["other"] == 1
    assert p["waiting_prune"] == {"count": 2, "gb": 12.0}                 # paused in 1/vr only
    assert [s["name"] for s in p["stale_paused"]] == ["old vr"]          # >4 days; "other" never counts


def test_pool_uses_added_on_when_never_completed_and_5x_states():
    p = pool_summary([_t("1", "stoppedUP", 1, 6, done=False)], NOW)
    assert p["paused"] == 1 and p["stale_paused"][0]["age_days"] == 6.0


# --- the verdict -----------------------------------------------------------------------

def _health(last_ok_ago=600, failing=False, failed=0, recovered=False):
    return {"failing": failing, "last_ok": NOW - last_ok_ago if last_ok_ago is not None else None,
            "failed_24h": failed, "recovered_24h": recovered}


def _v(pull=None, prune=None, scripts_ok=True, qbt_ok=True, pool=None, qbt_reason="down"):
    return verdict(scripts_ok=scripts_ok, scripts_reason="mount missing",
                   pull=pull or _health(), prune=prune or _health(),
                   qbt_ok=qbt_ok, qbt_reason=qbt_reason, pool=pool or {"stale_paused": []}, now=NOW)


def test_green_when_all_is_well():
    v = _v()
    assert v["status"] == "green" and v["reasons"] == []


@pytest.mark.parametrize("kw, words", [
    ({"pull": _health(failing=True)}, "seedbox-pull is failing"),
    ({"prune": _health(failing=True)}, "seedbox-prune is failing"),
    ({"pull": _health(last_ok_ago=2 * H + 60)}, "no successful pull for"),
    ({"prune": _health(last_ok_ago=3 * H + 60)}, "no successful prune for"),
    ({"pull": _health(last_ok_ago=None)}, "no successful pull on record"),
    ({"qbt_ok": False}, "qBittorrent unreachable"),
    ({"scripts_ok": False}, "can't read the pipeline's state files"),
])
def test_each_red_rule(kw, words):
    v = _v(**kw)
    assert v["status"] == "red" and any(words in r for r in v["reasons"])


def test_gap_limits_are_exclusive():
    assert _v(pull=_health(last_ok_ago=2 * H - 60), prune=_health(last_ok_ago=3 * H - 60))["status"] == "green"


def test_yellow_for_a_recovered_failure_and_for_old_paused_torrents():
    v = _v(pull=_health(failed=2, recovered=True))
    assert v["status"] == "yellow" and "failed 2×" in v["reasons"][0] and "recovered" in v["reasons"][0]
    v = _v(pool={"stale_paused": [{"age_days": 5.2}, {"age_days": 4.5}]})
    assert v["status"] == "yellow" and "2 paused torrents" in v["reasons"][0] and "5.2" in v["reasons"][0]


def test_red_beats_yellow_and_lists_both():
    v = _v(pull=_health(failing=True), prune=_health(failed=1, recovered=True))
    assert v["status"] == "red" and len(v["reasons"]) == 2 and "failing" in v["reasons"][0]


def test_unreadable_scripts_never_green():
    v = verdict(scripts_ok=False, scripts_reason="no runs.log", pull=None, prune=None,
                qbt_ok=True, qbt_reason=None, pool={"stale_paused": []}, now=NOW)
    assert v["status"] == "red"


# --- qBittorrent: login + GET only ------------------------------------------------------

class _Resp:
    def __init__(self, status=200, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        if self._data is None:
            raise ValueError("not json")
        return self._data


class _Session:
    """Records every call; has no put/delete — and the client must not need them."""

    def __init__(self, login=("Ok.", 200), info=None):
        self.headers, self.calls = {}, []
        self.login_resp = _Resp(login[1], login[0])
        self.info = list(info or [_Resp(200, data=[{"name": "a"}])])

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url.split("/api/v2/")[1]))
        return self.login_resp

    def get(self, url, timeout=None):
        self.calls.append(("GET", url.split("/api/v2/")[1]))
        return self.info.pop(0)


def test_client_logs_in_then_lists_with_referer():
    s = _Session()
    c = QbtClient("https://box.example/qbittorrent/", "u", "p", session=s)
    assert c.torrents() == [{"name": "a"}]
    assert s.calls == [("POST", "auth/login"), ("GET", "torrents/info")]
    assert s.headers["Referer"] == "https://box.example/qbittorrent/"
    assert all(m in ("POST", "GET") for m, _ in s.calls)
    assert [u for m, u in s.calls if m == "POST"] == ["auth/login"]       # the only POST is the login


def test_client_reports_bad_credentials_ban_and_relogins_once():
    with pytest.raises(QbtError, match="wrong username or password"):
        QbtClient("https://b", "u", "p", session=_Session(login=("Fails.", 200))).torrents()
    with pytest.raises(QbtError, match="403"):
        QbtClient("https://b", "u", "p", session=_Session(login=("", 403))).torrents()
    s = _Session(info=[_Resp(403), _Resp(200, data=[])])
    assert QbtClient("https://b", "u", "p", session=s).torrents() == []
    assert [u for _, u in s.calls] == ["auth/login", "torrents/info", "auth/login", "torrents/info"]
    with pytest.raises(QbtError, match="wasn't JSON"):
        QbtClient("https://b", "u", "p", session=_Session(info=[_Resp(200, text="<html>")])).torrents()


def test_client_has_no_write_methods():
    public = {n for n in dir(QbtClient) if not n.startswith("_")}
    assert public == {"login", "torrents"}


# --- the page: sources reported separately ------------------------------------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from peaks.config import Config
    from peaks.web.service import Service

    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.seedbox.dir = str(tmp_path / "seedbox")
    cfg.seedbox.inbox = str(tmp_path / "data" / "Rando" / "norating")
    return Service(cfg)


def test_missing_mount_and_unconfigured_api_are_unavailable_not_zero(svc):
    st = svc.seedbox_status(now=NOW)
    assert st["scripts"]["ok"] is False and "/seedbox" in st["scripts"]["reason"]
    assert st["pool"]["ok"] is False and "PEAKS_QBT_URL" in st["pool"]["reason"]
    assert st["inbox"]["ok"] is False and "can't see" in st["inbox"]["reason"]
    assert "throughput" not in st["scripts"] and "total" not in st["pool"]
    assert st["verdict"]["status"] == "red"


def test_reads_the_folder_without_writing_and_caches_the_pool(svc, tmp_path, monkeypatch):
    d = tmp_path / "seedbox"
    d.mkdir()
    (d / "runs.log").write_text(_log((NOW - 25 * 60, "pull", "ok", 2), (NOW - 40 * 60, "prune", "ok", "")))
    (d / "1.done").write_text("a.mp4\nb.mp4\n")
    (d / "pull.log").write_text("x\n2026/10/04 ERROR : boom\n")
    inbox = tmp_path / "data" / "Rando" / "norating"
    inbox.mkdir(parents=True)
    (inbox / "v.mp4").write_bytes(b"x" * 1000)
    (inbox / "v.funscript").write_text("{}")
    before = {p: p.stat().st_mtime_ns for p in d.iterdir()}
    svc.cfg.seedbox.qbt_url, svc.cfg.seedbox.qbt_user, svc.cfg.seedbox.qbt_password = "https://b", "u", "p"
    calls = []

    class FakeQbt:
        base = "https://b"

        def torrents(self):
            calls.append(1)
            return [_t("1", "uploading", 1, 1)]

    svc._sb_qbt = FakeQbt()
    st = svc.seedbox_status(now=NOW)
    assert st["verdict"]["status"] == "green", st["verdict"]
    assert st["scripts"]["pull"]["errors"] == []                      # only shown while failing
    assert st["scripts"]["ever_pulled"] == {"1": 2, "vr": None}
    assert st["inbox"]["files"] == 1
    svc.seedbox_status(now=NOW + 60)
    svc.seedbox_status(now=NOW + 120, refresh=True)                  # ↻ after 30 s: re-fetches
    svc.seedbox_status(now=NOW + 130, refresh=True)                  # …but not again within 30 s
    assert len(calls) == 2
    (d / "pull.failing").touch()
    before[d / "pull.failing"] = (d / "pull.failing").stat().st_mtime_ns
    st = svc.seedbox_status(now=NOW + 140)
    assert st["verdict"]["status"] == "red" and st["scripts"]["pull"]["errors"] == ["2026/10/04 ERROR : boom"]
    assert {p: p.stat().st_mtime_ns for p in d.iterdir()} == before   # nothing written there


def test_api_route(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    r = api.get("/api/seedbox").json()
    assert r["verdict"]["status"] == "red" and "password" not in str(r).lower().replace("peaks_qbt_password", "")
    assert api.post("/api/seedbox/refresh").status_code == 200
