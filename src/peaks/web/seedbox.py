"""The Seedbox page: is the seedbox → library pipeline healthy?

Three sources, each reported on its own ({ok, reason, data}) so one being
unavailable never blanks the others or shows zeros as if they were real:
- the scripts' state folder (cfg.seedbox.dir, mounted read-only — opened for
  reading only): runs.log, pull.failing / prune.failing, 1.done / vr.done,
  the tail of pull.log;
- the seedbox's qBittorrent (login + GET only), cached for POOL_TTL;
- the norating inbox under /data.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..seedbox import (QbtClient, QbtError, parse_runs, pool_summary, script_health,
                       throughput, verdict)
from .watch import VIDEO_EXT

POOL_TTL = 300.0          # qBittorrent results are reused for 5 minutes
POOL_ERROR_TTL = 60.0     # …a failure for 1 (don't hammer a seedbox that's down)
REFRESH_MIN = 30.0        # the ↻ button can't force a re-fetch more often than this
LOG_TAIL_BYTES = 256 * 1024
ERROR_LINES = 15


class SeedboxMixin:
    # --- the scripts' folder ------------------------------------------------------

    def _sb_dir(self) -> Path:
        return Path(self.cfg.seedbox.dir or "/seedbox")

    def _sb_scripts(self, now: float) -> dict:
        d = self._sb_dir()
        if not d.is_dir():
            return {"ok": False, "reason": f"{d} isn't there — add the read-only path "
                    "/mnt/user/appdata/seedbox-pull → /seedbox to the Peaks container"}
        runs_file = d / "runs.log"
        try:
            text = runs_file.read_text(errors="replace")
        except FileNotFoundError:
            return {"ok": False, "reason": f"no runs.log in {d} — is this the seedbox-pull folder?"}
        except OSError as exc:
            return {"ok": False, "reason": f"can't read {runs_file}: {exc.strerror or exc}"}
        runs, bad = parse_runs(text)
        failing = {k: (d / f"{k}.failing").exists() for k in ("pull", "prune")}
        pull = script_health(runs, "pull", now, failing["pull"])
        prune = script_health(runs, "prune", now, failing["prune"])
        pull["errors"] = self._sb_error_lines(d / "pull.log") if failing["pull"] else []
        done = {}
        for cat in ("1", "vr"):
            try:
                with open(d / f"{cat}.done", "rb") as fh:
                    done[cat] = sum(1 for line in fh if line.strip())
            except OSError:
                done[cat] = None
        return {"ok": True, "reason": None, "pull": pull, "prune": prune, "malformed": bad,
                "runs": len(runs), "first_run": runs[0].at if runs else None,
                "throughput": throughput(runs, now), "ever_pulled": done}

    @staticmethod
    def _sb_error_lines(path: Path) -> list[str]:
        """The last error lines of rclone's log — read from the end, never the
        whole file."""
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - LOG_TAIL_BYTES))
                tail = fh.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []
        hits = [ln.strip() for ln in tail if any(w in ln for w in ("ERROR", "CRITICAL", "Failed", "FATAL"))]
        return (hits or [ln.strip() for ln in tail if ln.strip()])[-ERROR_LINES:]

    # --- the norating inbox ----------------------------------------------------------

    def _sb_inbox_path(self) -> str | None:
        if self.cfg.seedbox.inbox:
            return self.cfg.seedbox.inbox
        try:
            paths = self.watch_settings()["watch_paths"]
        except Exception:  # noqa: BLE001
            paths = []
        return next((p for p in paths if "norating" in p.lower()), None)

    def _sb_inbox(self) -> dict:
        path = self._sb_inbox_path()
        if not path:
            return {"ok": False, "reason": "no inbox folder — add the norating folder in Settings → Ingest → "
                    "Watch for new files (or set PEAKS_SEEDBOX_INBOX)"}
        if not os.path.isdir(path):
            return {"ok": False, "reason": f"Peaks can't see {path} — it has to be inside the /data mount"}
        files, size = 0, 0
        for d, dirs, names in os.walk(path, onerror=lambda e: None):
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            for n in names:
                if n.startswith(".") or os.path.splitext(n.lower())[1] not in VIDEO_EXT:
                    continue
                try:
                    size += os.stat(os.path.join(d, n)).st_size
                except OSError:
                    continue
                files += 1
        return {"ok": True, "reason": None, "path": path, "files": files, "gb": round(size / 1e9, 2)}

    # --- qBittorrent, cached ---------------------------------------------------------------

    def _sb_pool(self, now: float, force: bool = False) -> dict:
        c = self.cfg.seedbox
        if not (c.qbt_url and c.qbt_user and c.qbt_password):
            return {"ok": False, "reason": "not set up — add PEAKS_QBT_URL, PEAKS_QBT_USER and "
                    "PEAKS_QBT_PASSWORD to the Peaks container", "fetched_at": None}
        cached = self.__dict__.get("_sb_pool_cache")
        if cached:
            age = now - cached["fetched_at"]
            ttl = POOL_TTL if cached["ok"] else POOL_ERROR_TTL
            if age < (REFRESH_MIN if force else ttl):
                return cached
        try:
            client = self.__dict__.get("_sb_qbt")
            if client is None or client.base != c.qbt_url.rstrip("/"):
                client = QbtClient(c.qbt_url, c.qbt_user, c.qbt_password)
                self._sb_qbt = client
            torrents = client.torrents()
            out = {"ok": True, "reason": None, "fetched_at": now, **pool_summary(torrents, now)}
        except QbtError as exc:
            out = {"ok": False, "reason": str(exc), "fetched_at": now}
        self._sb_pool_cache = out
        return out

    # --- the page --------------------------------------------------------------------------

    def seedbox_status(self, refresh: bool = False, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        scripts = self._sb_scripts(now)
        pool = self._sb_pool(now, force=refresh)
        inbox = self._sb_inbox()
        v = verdict(scripts_ok=scripts["ok"], scripts_reason=scripts.get("reason"),
                    pull=scripts.get("pull"), prune=scripts.get("prune"),
                    qbt_ok=pool["ok"], qbt_reason=pool.get("reason"),
                    pool=pool if pool["ok"] else None, now=now)
        return {"verdict": v, "scripts": scripts, "pool": pool, "inbox": inbox, "now": now,
                "pool_ttl": POOL_TTL}
