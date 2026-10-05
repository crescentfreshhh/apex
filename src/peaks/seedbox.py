"""The seedbox pipeline, read-only: the seedbox-pull / seedbox-prune scripts'
state files and the seedbox's qBittorrent pool, boiled down to one verdict.

Pure functions (parsing, health, throughput, pool, verdict) so the rules are
testable without a seedbox; `QbtClient` only ever logs in and GETs.

runs.log: one line per script run, ``<epoch>|<pull|prune>|<ok|fail>|<files>``
(files empty for prune). Malformed lines are skipped and counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

DAY = 86400.0
PULL_MAX_GAP = 2 * 3600.0          # RED: no successful pull for longer than this
PRUNE_MAX_GAP = 3 * 3600.0         # RED: no successful prune for longer than this
PAUSED_MAX_AGE = 4 * DAY           # YELLOW: a paused 1/vr torrent older than this (prune removes at 3)
PRUNE_CATEGORIES = ("1", "vr")

# qBittorrent states (4.3.x names, plus 5.x's stopped* for paused)
PAUSED_STATES = {"pausedUP", "pausedDL", "stoppedUP", "stoppedDL"}
SEEDING_STATES = {"uploading", "stalledUP", "forcedUP", "queuedUP", "checkingUP"}


@dataclass(frozen=True)
class Run:
    at: float
    kind: str          # "pull" | "prune"
    ok: bool
    files: int | None  # pull: files pulled; prune: None


def parse_runs(text: str) -> tuple[list[Run], int]:
    """(runs oldest first, malformed line count)."""
    runs, bad = [], 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:
            bad += 1
            continue
        at, kind, status, files = (p.strip() for p in parts)
        try:
            at_f = float(at)
        except ValueError:
            bad += 1
            continue
        if kind not in ("pull", "prune") or status not in ("ok", "fail"):
            bad += 1
            continue
        n = None
        if files:
            try:
                n = int(files)
            except ValueError:
                bad += 1
                continue
        runs.append(Run(at_f, kind, status == "ok", n))
    runs.sort(key=lambda r: r.at)
    return runs, bad


def script_health(runs: list[Run], kind: str, now: float, failing: bool) -> dict:
    mine = [r for r in runs if r.kind == kind]
    last_ok = max((r.at for r in mine if r.ok), default=None)
    day = [r for r in mine if r.at >= now - DAY]
    last = mine[-1] if mine else None
    return {
        "failing": failing,
        "last_ok": last_ok,
        "last_ok_ago_min": None if last_ok is None else round((now - last_ok) / 60),
        "last_run": last.at if last else None,
        "last_run_ok": last.ok if last else None,
        # what the last run accomplished: files pulled / torrents pruned (None = not logged)
        "last_run_count": last.files if last else None,
        "count_24h": (sum(r.files for r in day if r.files is not None)
                      if any(r.files is not None for r in day) else None),
        "ok_24h": sum(1 for r in day if r.ok),
        "failed_24h": sum(1 for r in day if not r.ok),
        # a failure in the last day that has been followed by a success since
        "recovered_24h": any(not r.ok for r in day) and bool(last and last.ok) and not failing,
    }


def throughput(runs: list[Run], now: float, tz_offset: float | None = None) -> dict:
    """Files pulled (every pull run that reported a count) in the last 24 h /
    7 days, and per local day for the week (oldest first, today last)."""
    if tz_offset is None:                      # local days (seconds east of UTC)
        tz_offset = time.localtime(now).tm_gmtoff
    pulls = [r for r in runs if r.kind == "pull" and r.files is not None]
    today = int((now + tz_offset) // DAY)
    days = [{"day": time.strftime("%a %d", time.gmtime((today - i) * DAY)), "files": 0} for i in range(6, -1, -1)]
    for r in pulls:
        d = today - int((r.at + tz_offset) // DAY)
        if 0 <= d <= 6:
            days[6 - d]["files"] += r.files
    return {"last_24h": sum(r.files for r in pulls if r.at >= now - DAY),
            "last_7d": sum(r.files for r in pulls if r.at >= now - 7 * DAY),
            "per_day": days}


def _age_base(t: dict) -> float:
    c = t.get("completion_on") or 0
    return float(c if c and c > 0 else (t.get("added_on") or 0))


def pool_summary(torrents: list[dict], now: float) -> dict:
    gb = 1e9

    def bucket():
        return {"count": 0, "gb": 0.0}

    total, by_cat = bucket(), {}
    seeding = paused = 0
    waiting, stale = bucket(), []
    for t in torrents:
        size = float(t.get("size") or 0)
        cat = t.get("category") or "(none)"
        st = t.get("state") or ""
        total["count"] += 1
        total["gb"] += size / gb
        c = by_cat.setdefault(cat, bucket())
        c["count"] += 1
        c["gb"] += size / gb
        if st in SEEDING_STATES:
            seeding += 1
        if st in PAUSED_STATES:
            paused += 1
            if cat in PRUNE_CATEGORIES:
                waiting["count"] += 1
                waiting["gb"] += size / gb
                age = now - _age_base(t)
                if _age_base(t) and age > PAUSED_MAX_AGE:
                    stale.append({"name": t.get("name") or "?", "category": cat,
                                  "age_days": round(age / DAY, 1), "gb": round(size / gb, 2)})
    rnd = lambda b: {"count": b["count"], "gb": round(b["gb"], 2)}  # noqa: E731
    return {"total": rnd(total), "by_category": {k: rnd(v) for k, v in sorted(by_cat.items())},
            "seeding": seeding, "paused": paused, "other": total["count"] - seeding - paused,
            "waiting_prune": rnd(waiting),
            "stale_paused": sorted(stale, key=lambda s: -s["age_days"])}


def verdict(*, scripts_ok: bool, scripts_reason: str | None, pull: dict | None, prune: dict | None,
            qbt_ok: bool, qbt_reason: str | None, pool: dict | None, now: float) -> dict:
    """GREEN / YELLOW / RED with every reason in plain words (worst first).
    A source that can't be read is RED — never a silent GREEN."""
    red, yellow = [], []
    if not scripts_ok:
        red.append(f"can't read the pipeline's state files — {scripts_reason}")
    else:
        for name, h, gap in (("pull", pull, PULL_MAX_GAP), ("prune", prune, PRUNE_MAX_GAP)):
            if h["failing"]:
                red.append(f"seedbox-{name} is failing right now")
            if h["last_ok"] is None:
                red.append(f"no successful {name} on record")
            elif now - h["last_ok"] > gap:
                red.append(f"no successful {name} for {_ago(now - h['last_ok'])} "
                           f"(expected within {int(gap // 3600)} h)")
            if h["recovered_24h"]:
                yellow.append(f"seedbox-{name} failed {h['failed_24h']}× in the last 24 h, "
                              "but has recovered")
    if not qbt_ok:
        red.append(f"qBittorrent unreachable — {qbt_reason}")
    elif pool and pool["stale_paused"]:
        n = len(pool["stale_paused"])
        oldest = pool["stale_paused"][0]["age_days"]
        yellow.append(f"{n} paused torrent{'s' if n != 1 else ''} in 1/vr older than 4 days "
                      f"(oldest {oldest:g} d) — prune should have removed them at 3")
    status = "red" if red else "yellow" if yellow else "green"
    return {"status": status, "reasons": red + yellow,
            "headline": {"red": "Pipeline needs attention", "yellow": "Working, with a warning",
                         "green": "Pipeline healthy"}[status]}


def _ago(sec: float) -> str:
    m = int(sec // 60)
    if m < 60:
        return f"{m} min"
    h = m / 60
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} days"


# --- qBittorrent: login + GET only -------------------------------------------------

class QbtError(RuntimeError):
    pass


class QbtClient:
    """Read-only qBittorrent Web API (v2, as in 4.3.x): it can log in and list
    torrents — nothing else exists here, so it can't pause, delete or change
    anything."""

    def __init__(self, url: str, user: str, password: str, timeout: float = 15.0, session=None):
        import requests

        self.base = url.rstrip("/")
        self.user, self.password, self.timeout = user, password, timeout
        self.s = session if session is not None else requests.Session()
        # 4.3.x checks Referer/Origin against its host (CSRF protection)
        self.s.headers.update({"Referer": self.base + "/", "Origin": self.base})
        self._logged_in = False

    def login(self) -> None:
        try:
            r = self.s.post(f"{self.base}/api/v2/auth/login",
                            data={"username": self.user, "password": self.password}, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 — network: report, don't raise raw
            raise QbtError(f"can't reach {self.base} ({type(exc).__name__})") from exc
        if r.status_code == 403:
            raise QbtError("qBittorrent refused the login (403 — too many failed attempts, IP banned for a while?)")
        if r.status_code != 200:
            raise QbtError(f"login failed (HTTP {r.status_code})")
        if (r.text or "").strip() != "Ok.":
            raise QbtError("wrong username or password")
        self._logged_in = True

    def torrents(self) -> list[dict]:
        if not self._logged_in:
            self.login()
        for attempt in (0, 1):
            try:
                r = self.s.get(f"{self.base}/api/v2/torrents/info", timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                raise QbtError(f"can't reach {self.base} ({type(exc).__name__})") from exc
            if r.status_code == 403 and attempt == 0:     # session expired: log in again once
                self.login()
                continue
            if r.status_code != 200:
                raise QbtError(f"torrent list failed (HTTP {r.status_code})")
            try:
                data = r.json()
            except ValueError as exc:
                raise QbtError("torrent list wasn't JSON — is the URL the qBittorrent Web UI?") from exc
            if not isinstance(data, list):
                raise QbtError("unexpected torrent list format")
            return data
        raise QbtError("torrent list failed after logging in again")
