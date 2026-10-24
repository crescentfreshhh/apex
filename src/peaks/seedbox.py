"""The seedbox pipeline, read-only: the seedbox-pull script's state files and
the seedbox's qBittorrent pool, boiled down to one verdict.

Pure functions (parsing, health, throughput, pool, verdict) so the rules are
testable without a seedbox; `QbtClient` only ever logs in and GETs.

runs.log: one line per script run, ``<epoch>|<kind>|<ok|fail>|<files>``. Only
``pull`` runs are watched; other scripts' lines (the retired seedbox-prune, its
replacement) are skipped quietly. Malformed lines are skipped and counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

DAY = 86400.0
PULL_MAX_GAP = 2 * 3600.0          # RED: no successful pull for longer than this

# qBittorrent states (4.3.x names, plus 5.x's stopped* for paused)
PAUSED_STATES = {"pausedUP", "pausedDL", "stoppedUP", "stoppedDL"}
SEEDING_STATES = {"uploading", "stalledUP", "forcedUP", "queuedUP", "checkingUP"}


@dataclass(frozen=True)
class Run:
    at: float
    kind: str          # "pull"
    ok: bool
    files: int | None  # files pulled (None = not logged)


def parse_runs(text: str) -> tuple[list[Run], int]:
    """(runs oldest first, malformed line count)."""
    runs, bad = [], 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) not in (4, 5):            # Peaks' purge adds a 5th field (bytes freed)
            bad += 1
            continue
        at, kind, status, files = (p.strip() for p in parts[:4])
        try:
            at_f = float(at)
        except ValueError:
            bad += 1
            continue
        if not kind or status not in ("ok", "fail"):
            bad += 1
            continue
        if kind != "pull":                     # another script's run: not ours to judge
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


def parse_all_runs(text: str) -> list[dict]:
    """Every well-formed runs.log line, any job: [{at, kind, ok, count, bytes}]
    oldest first (for the daily report and the factory floor)."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) not in (4, 5) or parts[2] not in ("ok", "fail") or not parts[1]:
            continue
        try:
            at = float(parts[0])
            count = int(parts[3]) if parts[3] else None
            nbytes = int(parts[4]) if len(parts) == 5 and parts[4] else None
        except ValueError:
            continue
        out.append({"at": at, "kind": parts[1], "ok": parts[2] == "ok", "count": count, "bytes": nbytes})
    out.sort(key=lambda r: r["at"])
    return out


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
        # what the last run accomplished: files pulled (None = not logged)
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


def pool_summary(torrents: list[dict], now: float) -> dict:
    gb = 1e9

    def bucket():
        return {"count": 0, "gb": 0.0}

    total, by_cat = bucket(), {}
    seeding = paused = 0
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
    rnd = lambda b: {"count": b["count"], "gb": round(b["gb"], 2)}  # noqa: E731
    return {"total": rnd(total), "by_category": {k: rnd(v) for k, v in sorted(by_cat.items())},
            "seeding": seeding, "paused": paused, "other": total["count"] - seeding - paused}


def verdict(*, scripts_ok: bool, scripts_reason: str | None, pull: dict | None,
            qbt_ok: bool, qbt_reason: str | None, pool: dict | None, now: float) -> dict:
    """GREEN / YELLOW / RED with every reason in plain words (worst first).
    A source that can't be read is RED — never a silent GREEN."""
    red, yellow = [], []
    if not scripts_ok:
        red.append(f"can't read the pipeline's state files — {scripts_reason}")
    else:
        h, gap = pull, PULL_MAX_GAP
        if h["failing"]:
            red.append("seedbox-pull is failing right now")
        if h["last_ok"] is None:
            red.append("no successful pull on record")
        elif now - h["last_ok"] > gap:
            red.append(f"no successful pull for {_ago(now - h['last_ok'])} "
                       f"(expected within {int(gap // 3600)} h)")
        if h["recovered_24h"]:
            yellow.append(f"seedbox-pull failed {h['failed_24h']}× in the last 24 h, "
                          "but has recovered")
    if not qbt_ok:
        red.append(f"qBittorrent unreachable — {qbt_reason}")
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


# --- qBittorrent: login + reads, and the purge's one delete ------------------------

class QbtError(RuntimeError):
    pass


class QbtClient:
    """qBittorrent Web API (v2, as in 4.3.x): log in, list torrents and their
    files — and one write, `delete_torrents`, used only by the purge job for
    torrents it has confirmed are safely on Jarvis. Nothing else can pause,
    add or change anything."""

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

    def _get_json(self, path: str, what: str):
        if not self._logged_in:
            self.login()
        for attempt in (0, 1):
            try:
                r = self.s.get(f"{self.base}{path}", timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                raise QbtError(f"can't reach {self.base} ({type(exc).__name__})") from exc
            if r.status_code == 403 and attempt == 0:
                self.login()
                continue
            if r.status_code != 200:
                raise QbtError(f"{what} failed (HTTP {r.status_code})")
            try:
                return r.json()
            except ValueError as exc:
                raise QbtError(f"{what} wasn't JSON") from exc
        raise QbtError(f"{what} failed after logging in again")

    def files(self, torrent_hash: str) -> list[dict]:
        """A torrent's files (name relative to its save path, priority, size)."""
        from urllib.parse import quote

        data = self._get_json(f"/api/v2/torrents/files?hash={quote(torrent_hash)}", "file list")
        if not isinstance(data, list):
            raise QbtError("unexpected file list format")
        return data

    def delete_torrents(self, hashes: list[str], delete_files: bool = True) -> None:
        """The purge's delete — the only write Peaks makes to qBittorrent."""
        if not hashes:
            return
        if not self._logged_in:
            self.login()
        try:
            r = self.s.post(f"{self.base}/api/v2/torrents/delete",
                            data={"hashes": "|".join(hashes), "deleteFiles": "true" if delete_files else "false"},
                            timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise QbtError(f"can't reach {self.base} ({type(exc).__name__})") from exc
        if r.status_code != 200:
            raise QbtError(f"delete failed (HTTP {r.status_code})")

