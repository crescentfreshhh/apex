"""The seedbox jobs, run by Peaks (they used to be Unraid User Scripts):

- pull   (every 30 min) — new video / .funscript files from the seedbox into the
  array, a few at a time, each one's permissions fixed and logged as done the
  moment it lands. Never deletes anything on Jarvis.
- purge  (05:00) — delete torrents (and their files) on the seedbox once
  they're 100 % downloaded and confirmed on Jarvis (or imported by Whisparr).
  The first days are a dry run that only says what it would delete.
- report (06:00) — what came in and how the machinery ran; counts only.

rclone runs inside the container with your Unraid rclone.conf mounted
read-only (the seedbox credentials stay there). qBittorrent's login comes
from PEAKS_QBT_*. The state folder keeps the scripts' formats (runs.log,
pull.log, <cat>.done, pull.failing), so history carries over and the
Seedbox page keeps reading it.

Peaks won't run these until you confirm the Unraid scripts are switched off
(two pullers at once would fight over the same files), and it warns if
runs.log shows a pull it didn't make.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import seedjobs as sj
from ..seedbox import QbtClient, QbtError, parse_all_runs

DEFAULTS = {
    "remote": "",                       # e.g. seedbox:/home/<you>/downloads/qbittorrent
    "categories": [{"cat": "1", "dest": "/data/Rando/norating"}, {"cat": "vr", "dest": "/data/#VR"}],
    "filter": "/rclone/filters.txt",
    "parallel": 3, "streams": 4,
    "owner": "99:100",
    "whisparr_folder": "arr/whisparr",
    "whisparr_url": "",
    "whisparr_config": "/whisparr/config.xml",
    "pull_on": True, "pull_every_min": 30,
    "purge_on": True, "purge_at": "05:00",
    "report_on": True, "report_at": "06:00",
    "cutover": False,                   # "I've disabled the Unraid scripts"
    "purge_mode": "auto",               # auto = dry run for DRY_DAYS, then live · dry · live
    "purge_dry_until": None,
}
DRY_DAYS = 3
JOBS = ("pull", "purge", "report")
_NAMEISH = re.compile(r"[^/\s]+\.(mkv|mp4|m4v|mov|wmv|avi|ts|webm|funscript)\b", re.I)


def scrub(text: str) -> str:
    """A log line without file names (for the factory floor's ticker)."""
    return _NAMEISH.sub("<file>", text or "")


class SeedJobsMixin:
    # --- settings -------------------------------------------------------------------

    def seedjob_settings(self) -> dict:
        s = {**DEFAULTS, **(self._settings().get("seedjobs") or {})}
        s["whisparr_url"] = os.environ.get("PEAKS_WHISPARR_URL") or s["whisparr_url"] or "http://192.168.1.2:7171"
        s["remote"] = os.environ.get("PEAKS_SEEDBOX_REMOTE") or s["remote"]
        return s

    def save_seedjob_settings(self, body: dict) -> dict:
        from .. import atomic

        settings = dict(self._settings())
        cur = {**DEFAULTS, **(settings.get("seedjobs") or {})}
        for k in DEFAULTS:
            if k in body and body[k] is not None and k != "purge_dry_until":
                cur[k] = body[k]
        cur["parallel"] = max(1, min(8, int(cur["parallel"])))
        cur["streams"] = max(1, min(16, int(cur["streams"])))
        cur["pull_every_min"] = max(5, int(cur["pull_every_min"]))
        for k in ("purge_at", "report_at"):
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(cur[k])):
                raise ValueError(f"{k} must be HH:MM")
        if cur["purge_mode"] not in ("auto", "dry", "live"):
            raise ValueError("purge_mode is auto, dry or live")
        cur["categories"] = [{"cat": str(c["cat"]).strip(), "dest": str(c["dest"]).strip()}
                             for c in cur["categories"] if str(c.get("cat", "")).strip() and str(c.get("dest", "")).strip()]
        if cur["cutover"] and not cur.get("purge_dry_until"):
            cur["purge_dry_until"] = time.time() + DRY_DAYS * 86400      # the dry run starts with the takeover
        settings["seedjobs"] = cur
        path = self._settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(path, settings, indent=2, trailing_newline=True)
        self._settings_cache = settings
        return self.seedjob_status()

    def _purge_dry(self, s: dict, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if s["purge_mode"] == "dry":
            return True
        if s["purge_mode"] == "live":
            return False
        until = s.get("purge_dry_until")
        return until is None or now < until

    # --- state ----------------------------------------------------------------------

    def _sj_state_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "seedjobs.json"

    def _sj_lock(self):
        return self.__dict__.setdefault("_sj_lock_obj", threading.RLock())

    def _sj_load(self) -> dict:
        from .. import atomic

        try:
            d = atomic.read_json(self._sj_state_path(), default=None)
        except atomic.DamagedFile:
            d = None
        d = d if isinstance(d, dict) else {}
        d.setdefault("last", {})
        d.setdefault("own_runs", [])
        d.setdefault("alerts", [])
        return d

    def _sj_save(self, d: dict) -> None:
        from .. import atomic

        d["own_runs"] = d["own_runs"][-500:]
        d["alerts"] = d["alerts"][-50:]
        atomic.write_json(self._sj_state_path(), d, indent=1)

    def _sj_update(self, fn) -> dict:
        with self._sj_lock():
            d = self._sj_load()
            fn(d)
            self._sj_save(d)
            return d

    def _sj_alert(self, level: str, text: str) -> None:
        self._sj_update(lambda d: d["alerts"].append({"at": time.time(), "level": level, "text": text}))
        try:
            self.action_log().append("seedbox", detail=text, level=level)
        except Exception:  # noqa: BLE001 — the log is best-effort
            pass

    def _sj_dir(self) -> Path:
        return Path(self.cfg.seedbox.dir or "/seedbox")

    def _sj_append(self, name: str, line: str) -> None:
        with self.__dict__.setdefault("_sj_file_lock", threading.Lock()):
            with open(self._sj_dir() / name, "a", encoding="utf-8") as fh:
                fh.write(line.rstrip("\n") + "\n")

    def _sj_runs_line(self, kind: str, ok: bool, count: int, nbytes: int | None = None) -> None:
        at = time.time()
        self._sj_append("runs.log", sj.run_line(kind, ok, count, nbytes, at=at))
        self._sj_update(lambda d: d["own_runs"].append(int(at)))

    # --- the live picture (for the factory floor) ------------------------------------

    def _sj_live(self) -> dict:
        return self.__dict__.setdefault("_sj_live_obj", {"pull": {}, "purge": {}, "ticker": []})

    def _sj_tick(self, text: str) -> None:
        t = self._sj_live()["ticker"]
        clean = scrub(text).replace("·", "-").replace("—", "-").replace("→", "->")
        t.append({"at": time.time(), "text": clean.encode("ascii", "replace").decode()})     # the floor is ASCII
        del t[:-60]

    # --- rclone ------------------------------------------------------------------------

    def _rclone_cmd(self) -> str:
        return os.environ.get("PEAKS_RCLONE", "rclone")

    def _sj_find(self, name: str, preferred: str | None) -> str | None:
        """A file the jobs need (rclone.conf, the filter), wherever it is: the
        configured / mounted path, else a copy in the state folder's rclone/
        (Unraid's container editor only offers /mnt/user paths, so copying the
        files into /mnt/user/appdata/seedbox-pull/rclone/ is the easy route)."""
        for p in (preferred, str(self._sj_dir() / "rclone" / name)):
            if p and Path(p).is_file():
                return p
        return None

    def _rclone_conf(self) -> str | None:
        return self._sj_find("rclone.conf", os.environ.get("RCLONE_CONFIG") or "/rclone/rclone.conf")

    def _rclone_filter(self, s: dict) -> str | None:
        return self._sj_find("filters.txt", s["filter"])

    def _rclone(self, args: list[str], on_line=None, timeout: float | None = None) -> tuple[int, list[str]]:
        """Run rclone; stderr lines go to `on_line` as they come (stats, log).
        Returns (exit code, stdout lines)."""
        env = dict(os.environ)
        conf = self._rclone_conf()
        if conf:
            env["RCLONE_CONFIG"] = conf
        try:
            p = subprocess.Popen([self._rclone_cmd(), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=env)
        except FileNotFoundError:
            return 127, []
        out: list[str] = []

        def drain_err():
            for line in p.stderr:
                if on_line:
                    on_line(line.rstrip("\n"))
        th = threading.Thread(target=drain_err, daemon=True)
        th.start()
        for line in p.stdout:
            out.append(line.rstrip("\n"))
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
        th.join(5)
        return rc, out

    # --- guards ----------------------------------------------------------------------

    def _sj_foreign_pull(self, now: float) -> float | None:
        """A pull in runs.log in the last 45 min that Peaks didn't make — an
        Unraid script still running. Returns its time."""
        try:
            runs = parse_all_runs((self._sj_dir() / "runs.log").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None
        own = set(self._sj_load()["own_runs"])
        for r in reversed(runs):
            if r["kind"] == "pull" and r["at"] >= now - 45 * 60 and int(r["at"]) not in own:
                return r["at"]
        return None

    def _sj_ready(self, s: dict, kind: str) -> str | None:
        """Why this job can't run now (None = it can)."""
        if not s["cutover"]:
            return "switch off the Unraid seedbox scripts first, then tick “I've disabled them” in Settings → Seedbox jobs"
        if not self._sj_dir().is_dir():
            return f"the seedbox state folder {self._sj_dir()} isn't mounted"
        if kind == "pull":
            if not s["remote"]:
                return "set the seedbox rclone remote in Settings → Seedbox jobs"
            if not self._rclone_conf() and not os.environ.get("PEAKS_RCLONE"):
                return ("no rclone.conf — copy /boot/config/rclone/rclone.conf to "
                        "/mnt/user/appdata/seedbox-pull/rclone/rclone.conf (or mount it at /rclone/rclone.conf)")
            if not self._rclone_filter(s):
                return ("no pull filter — copy /boot/config/rclone-filters.txt to "
                        "/mnt/user/appdata/seedbox-pull/rclone/filters.txt (or mount it at /rclone/filters.txt)")
            missing = [c["dest"] for c in s["categories"] if not Path(c["dest"]).is_dir()]
            if missing:
                return f"destination folder not found in the container: {', '.join(missing)}"
            foreign = self._sj_foreign_pull(time.time())
            if foreign:
                return ("runs.log shows a pull Peaks didn't make "
                        f"({time.strftime('%H:%M', time.localtime(foreign))}) — is the Unraid script still on?")
        if kind == "purge":
            c = self.cfg.seedbox
            if not (c.qbt_url and c.qbt_user and c.qbt_password):
                return "qBittorrent isn't set up (PEAKS_QBT_URL / _USER / _PASSWORD)"
        return None

    # --- pull ------------------------------------------------------------------------

    def _sj_fix_perms(self, f: Path, dest: Path, owner: str) -> None:
        try:
            uid, gid = (int(x) for x in owner.split(":"))
        except ValueError:
            uid = gid = None
        targets = [(f, 0o666)]
        d = f.parent
        while len(str(d)) > len(str(dest)):
            targets.append((d, 0o777))
            d = d.parent
        for path, mode in targets:
            try:
                if uid is not None:
                    os.chown(path, uid, gid)
            except OSError:
                pass
            try:
                os.chmod(path, mode)
            except OSError:
                pass

    def run_seed_pull(self, job=None) -> dict:
        s = self.seedjob_settings()
        why = self._sj_ready(s, "pull")
        if why:
            raise RuntimeError(why)
        lock = self.__dict__.setdefault("_sj_pull_lock", threading.Lock())
        if not lock.acquire(blocking=False):
            return {"skipped": "a pull is already running"}
        log = job.log if job is not None else (lambda *_: None)
        live = self._sj_live()
        try:
            errors, total, total_bytes = [], 0, 0
            landed_paths: set[str] = set()
            landed_files: list[str] = []
            remote = s["remote"].rstrip("/")
            for c in s["categories"]:
                cat, dest = c["cat"], Path(c["dest"])
                done_path = self._sj_dir() / f"{cat}.done"
                done_path.touch(exist_ok=True)
                live["pull"] = {"stage": "listing", "cat": cat, "lanes": {}}
                self._sj_tick(f"[{cat}] listing the seedbox")
                rc, listing = self._rclone(["lsf", "-R", "--files-only", "--filter-from", self._rclone_filter(s),
                                            "--ignore-case", f"{remote}/{cat}"], timeout=600)
                if rc != 0:
                    errors.append(f"[{cat}] could not list the seedbox (connection/login problem)")
                    log(f"[{cat}] couldn't list the box (rclone exit {rc}) — skipping this run")
                    self._sj_tick(f"[{cat}] listing failed")
                    continue
                new = sj.new_files(listing, sj.read_done(done_path.read_text(encoding="utf-8", errors="replace")))
                if not new:
                    log(f"[{cat}] nothing new")
                    self._sj_tick(f"[{cat}] nothing new")
                    continue
                log(f"[{cat}] pulling {len(new)} file(s), {s['parallel']} at a time")
                self._sj_tick(f"[{cat}] pulling {len(new)} file(s)")
                live["pull"] = {"stage": "pulling", "cat": cat, "lanes": {}, "queued": len(new), "done": 0}
                results = []

                def fetch(i_rel, cat=cat, dest=dest):
                    i, rel = i_rel
                    if job is not None and job.cancelled:
                        return (rel, False, 0)
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    lane = {"id": i, "cat": cat, "pct": 0.0, "bytes": 0, "total": 0}
                    live["pull"]["lanes"][i] = lane

                    def on_line(line, lane=lane):
                        st = sj.rclone_stats(line)
                        if st:
                            lane.update(bytes=st["bytes"], total=st["total"], speed=st["speed"],
                                        pct=(st["bytes"] / st["total"]) if st["total"] else lane["pct"])
                        elif line.strip():
                            try:
                                msg = json.loads(line)
                                text = f"{msg.get('time', '')} {msg.get('level', '').upper()}: {msg.get('msg', '')}"
                            except ValueError:
                                text = line
                            self._sj_append("pull.log", text)
                    rc, _ = self._rclone(["copyto", f"{remote}/{cat}/{rel}", str(target),
                                          "--multi-thread-streams", str(s["streams"]),
                                          "--use-json-log", "--stats", "1s", "--log-level", "INFO"], on_line=on_line)
                    ok = rc == 0 and target.exists()
                    size = 0
                    if ok:
                        self._sj_fix_perms(target, dest, s["owner"])
                        self._sj_append(f"{cat}.done", rel)
                        size = target.stat().st_size
                        self._sj_append_ledger({"t": time.time(), "cat": cat, "rel": rel, "path": str(target),
                                                "size": size, "ext": rel.rsplit(".", 1)[-1].lower()})
                        if sj.wanted(rel) and not rel.lower().endswith(".funscript"):
                            self._sj_landed(str(dest), str(target))     # survives a restart mid-pull
                    live["pull"]["lanes"].pop(i, None)
                    live["pull"]["done"] = live["pull"].get("done", 0) + 1
                    self._sj_tick(f"[{cat}] {'OK  ' if ok else 'FAIL'} file {i + 1}/{len(new)}"
                                  + (f" · {size / 1e9:.2f} GB" if ok else ""))
                    log(f"{'OK  ' if ok else 'FAIL'} [{cat}] {rel}")
                    if job is not None:
                        job.progress = {"cat": cat, "done": live["pull"]["done"], "total": len(new)}
                    return (rel, ok, size)

                with ThreadPoolExecutor(max_workers=s["parallel"]) as ex:
                    results = list(ex.map(fetch, enumerate(new)))
                landed_here = [str(dest / rel) for rel, ok, _ in results if ok and sj.wanted(rel)
                               and not rel.lower().endswith(".funscript")]
                if landed_here:
                    landed_paths.add(str(dest))
                    landed_files.extend(landed_here)
                ok_n = sum(1 for _, ok, _ in results if ok)
                fail_n = len(results) - ok_n
                total += ok_n
                total_bytes += sum(sz for _, ok, sz in results if ok)
                log(f"[{cat}] done: {ok_n} pulled, {fail_n} failed")
                if fail_n:
                    errors.append(f"[{cat}] {fail_n} of {len(new)} file(s) failed; will retry next run")
            failing = self._sj_dir() / "pull.failing"
            self._sj_runs_line("pull", not errors, total)
            if not errors:
                if failing.exists():
                    failing.unlink()
                    self._sj_alert("ok", "Seedbox pull recovered — running normally again")
            elif not failing.exists():
                failing.touch()
                self._sj_alert("alert", "Seedbox pull failed: " + " ".join(errors))
            self._sj_update(lambda d: d["last"].__setitem__("pull", time.time()))
            if landed_files:            # Stash scans exactly where they landed (norating, #VR…)
                self._sj_hand_to_ingest(sorted(landed_paths), landed_files)
                log("handed to the next ingest: " + ", ".join(sorted(landed_paths)))
            self._sj_update(lambda d: d.pop("landing", None))   # handed over (or nothing landed)
            if total:
                try:
                    self.action_log().append("seedbox-pull", detail=f"pulled {total} file(s) · {total_bytes / 1e9:.1f} GB")
                except Exception:  # noqa: BLE001
                    pass
            return {"pulled": total, "bytes": total_bytes, "errors": errors, "ingest": sorted(landed_paths)}
        finally:
            live["pull"] = {}
            lock.release()

    # --- the hand-off to ingest: scan exactly where the pull put files ---------------

    def _sj_hand_to_ingest(self, paths: list[str], files: list[str]) -> None:
        def add(d):
            p = d.get("ingest_pending") or {"paths": [], "files": []}
            d["ingest_pending"] = {"paths": sorted(set(p["paths"]) | set(paths)),
                                   "files": sorted(set(p["files"]) | set(files)), "at": time.time()}
        self._sj_update(add)
        try:                                   # the folder watch mustn't start a second ingest for them
            self.watch_started(files)
        except Exception:  # noqa: BLE001
            pass

    def _sj_landed(self, path: str, file: str) -> None:
        """Note a file the moment it lands, so a restart before the pull ends
        can still hand it to ingest (see resume_seed_landings)."""
        def add(d):
            ln = d.setdefault("landing", {"paths": [], "files": []})
            if path not in ln["paths"]:
                ln["paths"].append(path)
            ln["files"].append(file)
        self._sj_update(add)

    def resume_seed_landings(self) -> int:
        """At startup: files an interrupted pull landed but never handed to
        ingest go to the next ingest now. Returns how many."""
        if not self._sj_load().get("landing"):
            return 0                            # nothing interrupted (and nothing written)
        lock = self.__dict__.setdefault("_sj_pull_lock", threading.Lock())
        if not lock.acquire(blocking=False):
            return 0                            # a pull is running: it hands them over itself
        try:
            ln = {}
            self._sj_update(lambda d: ln.update(d.pop("landing", None) or {}))
            files = sorted(set(ln.get("files") or []))
            if not files:
                return 0
            self._sj_hand_to_ingest(sorted(set(ln.get("paths") or [])), files)
            msg = f"handed {len(files)} file(s) from an interrupted pull to the next ingest"
            self._sj_tick(msg)
            self._sj_alert("warning", "Seedbox pull was interrupted — " + msg)
            return len(files)
        finally:
            lock.release()

    def take_seed_ingest(self) -> dict | None:
        """The folders waiting for an ingest (and forget them — `requeue` if it fails)."""
        out = {}

        def take(d):
            out.update(d.pop("ingest_pending", None) or {})
        self._sj_update(take)
        return out or None

    def requeue_seed_ingest(self, pending: dict) -> None:
        self._sj_update(lambda d: d.__setitem__("ingest_pending", {
            "paths": sorted(set((d.get("ingest_pending") or {}).get("paths", [])) | set(pending["paths"])),
            "files": sorted(set((d.get("ingest_pending") or {}).get("files", [])) | set(pending["files"])),
            "at": time.time(), "retry_after": time.time() + 600}))

    def seed_ingest_waiting(self, now: float | None = None) -> bool:
        p = self._sj_load().get("ingest_pending")
        return bool(p and p.get("paths")) and (now or time.time()) >= (p.get("retry_after") or 0)

    def _sj_dest_warnings(self, s: dict) -> list[str]:
        """Pull destinations Stash won't scan, or Peaks won't show."""
        dests = [c["dest"] for c in s["categories"]]
        out = []
        try:
            for w in self.watch_warnings(dests):
                if "isn't inside any Stash library folder" in w:
                    d = w.split(" isn't", 1)[0]
                    w = (f"Stash doesn't scan {d} — nothing pulled there will appear. "
                         "Add it in Stash → Settings → Library.")
                out.append(w)
        except Exception:  # noqa: BLE001
            pass
        scope = (self.cfg.library.path or "").rstrip("/")
        for d in dests:
            if scope and not (d == scope or d.startswith(scope + "/")):
                out.append(f"Peaks only shows scenes under {scope} (PEAKS_LIBRARY_PATH) — scenes from {d} won't appear in Peaks")
        return out

    def _sj_ledger_path(self) -> Path:
        return Path(self.cfg.modeling.dir) / "seedbox_pulled.jsonl"

    def _sj_append_ledger(self, rec: dict) -> None:
        with self.__dict__.setdefault("_sj_ledger_lock", threading.Lock()):
            p = self._sj_ledger_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _sj_ledger(self, since: float) -> list[dict]:
        out = []
        try:
            with open(self._sj_ledger_path(), encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("t", 0) >= since:
                        out.append(r)
        except OSError:
            pass
        return out

    # --- purge -----------------------------------------------------------------------

    def _whisparr(self, s: dict) -> dict:
        import requests

        try:
            xml = Path(s["whisparr_config"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"ok": False}
        m = re.search(r"<ApiKey>(.*?)</ApiKey>", xml)
        key = m.group(1) if m else ""
        if not key:
            return {"ok": False}
        base = s["whisparr_url"].rstrip("/")
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 90 * 86400))
        try:
            h = requests.get(f"{base}/api/v3/history/since", params={"date": since, "eventType": "downloadFolderImported"},
                             headers={"X-Api-Key": key}, timeout=60)
            q = requests.get(f"{base}/api/v3/queue", params={"page": 1, "pageSize": 1000,
                                                              "includeUnknownSeriesItems": "true"},
                             headers={"X-Api-Key": key}, timeout=60)
            if h.status_code != 200 or q.status_code != 200:
                return {"ok": False}
            hist, queue = h.json(), q.json()
        except Exception:  # noqa: BLE001 — Whisparr down: its torrents are left alone
            return {"ok": False}
        return {"ok": True,
                "imported": {str(x.get("downloadId")).upper() for x in hist
                             if x.get("eventType") == "downloadFolderImported" and x.get("downloadId")},
                "queued": {str(x.get("downloadId")).upper() for x in (queue or {}).get("records", []) if x.get("downloadId")}}

    def _sj_qbt(self) -> QbtClient:
        c = self.cfg.seedbox
        return QbtClient(c.qbt_url, c.qbt_user, c.qbt_password)

    def run_seed_purge(self, job=None, client: QbtClient | None = None) -> dict:
        s = self.seedjob_settings()
        why = self._sj_ready(s, "purge")
        if why:
            raise RuntimeError(why)
        log = job.log if job is not None else (lambda *_: None)
        dry = self._purge_dry(s)
        live = self._sj_live()
        live["purge"] = {"stage": "checking", "dry": dry}
        try:
            qbt = client or self._sj_qbt()
            base = s["remote"].split(":", 1)[-1].rstrip("/") if s["remote"] else ""
            cats = tuple(c["cat"] for c in s["categories"])
            try:
                torrents = qbt.torrents()
            except QbtError as exc:
                self._sj_runs_line("purge", False, 0, 0)
                self._sj_update(lambda d: d["last"].__setitem__("purge", time.time()))   # tomorrow, not every minute
                self._sj_alert("alert", f"Seedbox purge couldn't read qBittorrent: {exc}")
                raise RuntimeError(str(exc)) from exc
            files: dict[str, list | None] = {}
            for t in torrents:
                save = (t.get("save_path") or "").rstrip("/")
                if (t.get("progress") or 0) >= 1 and any(save == f"{base}/{c}" for c in cats):
                    try:
                        files[t["hash"]] = qbt.files(t["hash"])
                    except QbtError:
                        files[t["hash"]] = None
            done = {c: sj.read_done(self._read_text(self._sj_dir() / f"{c}.done")) for c in cats}
            w = self._whisparr(s) if s["whisparr_folder"] else {"ok": False}
            if s["whisparr_folder"] and not w["ok"]:
                log(f"Whisparr not reachable at {s['whisparr_url']} (or no API key) — its torrents are left alone this run")
            plan = sj.purge_plan(torrents, files, done, base, cats, s["whisparr_folder"], w)
            for d in plan:
                log(f"{('WOULD DELETE' if dry else 'DELETE') if d['delete'] else 'KEEP'} ({d['why']}): {d['name']}")
            gone = [d for d in plan if d["delete"]]
            freed = int(sum(d["size"] for d in gone))
            live["purge"] = {"stage": "deleting" if gone and not dry else "done", "dry": dry, "n": len(gone)}
            if gone and not dry:
                qbt.delete_torrents([d["hash"] for d in gone], delete_files=True)
                self._sj_tick(f"purge: deleted {len(gone)} torrent(s), {freed / 1e9:.1f} GB")
            else:
                self._sj_tick(f"purge ({'dry run' if dry else 'live'}): {len(gone)} torrent(s) ready to go")
            self._sj_runs_line("purge", True, 0 if dry else len(gone), 0 if dry else freed)
            result = {"dry": dry, "delete": len(gone), "freed_bytes": freed, "kept": len(plan) - len(gone)}
            self._sj_update(lambda d: (d["last"].__setitem__("purge", time.time()), d.__setitem__("last_purge", result)))
            try:
                self.action_log().append("seedbox-purge", detail=(
                    f"dry run: would delete {len(gone)} torrent(s), {freed / 1e9:.1f} GB" if dry
                    else f"deleted {len(gone)} torrent(s), freed {freed / 1e9:.1f} GB"))
            except Exception:  # noqa: BLE001
                pass
            # the dry run ends on its own after DRY_DAYS: say so once
            if s["purge_mode"] == "auto" and not dry and not self._sj_load().get("live_announced"):
                self._sj_update(lambda d: d.__setitem__("live_announced", True))
                self._sj_alert("ok", "The seedbox purge's dry run is over — it now deletes confirmed torrents")
            return result
        finally:
            live["purge"] = {}

    @staticmethod
    def _read_text(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # --- report ----------------------------------------------------------------------

    def run_seed_report(self, job=None, now: float | None = None) -> dict:
        from .. import atomic

        s = self.seedjob_settings()
        now = time.time() if now is None else now
        runs_path = self._sj_dir() / "runs.log"
        runs = parse_all_runs(self._read_text(runs_path))
        pulled = self._sj_ledger(now - 86400)
        by_path = {}
        try:
            for r in self._catalogue_all():
                if r.get("path"):
                    by_path[r["path"]] = r
        except Exception:  # noqa: BLE001 — Stash down: counts without the library side
            pass
        scenes = {}
        for p in pulled:
            r = by_path.get(p.get("path"))
            if r:
                p["scene_id"] = r["scene_id"]
                scenes[r["scene_id"]] = {"tier": r["tier"],
                                         "identified": bool(r.get("studio") or r.get("performers"))}
        rep = sj.report(runs, pulled, scenes, now, "dry run" if self._purge_dry(s, now) else "live")
        self._sj_update(lambda d: (d.__setitem__("report", rep), d["last"].__setitem__("report", now)))
        ci = rep["came_in"]
        text = (f"{ci['files']} file(s) came in ({ci['bytes'] / 1e9:.1f} GB): {ci['graded']} graded, "
                f"{ci['unreviewed']} unreviewed, {ci['not_in_stash_yet']} not in Stash yet")
        self._sj_alert("ok" if rep["ok"] else "warning",
                       ("Daily report: " if rep["ok"] else "Daily report — needs attention: "
                        + "; ".join(rep["issues"]) + ". ") + text)
        # runs.log keeps the last 7 days, as before
        keep = [ln for ln in self._read_text(runs_path).splitlines()
                if ln.split("|", 1)[0].strip().isdigit() and float(ln.split("|", 1)[0]) >= now - 7 * 86400]
        if runs_path.exists():
            atomic.write_text(runs_path, "\n".join(keep) + ("\n" if keep else ""))
        return rep

    # --- schedule --------------------------------------------------------------------

    @staticmethod
    def _at_today(hhmm: str, now: float) -> float:
        h, m = (int(x) for x in hhmm.split(":"))
        lt = time.localtime(now)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))

    def seedjob_due(self, kind: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        s = self.seedjob_settings()
        if not s["cutover"] or not s[f"{kind}_on"]:
            return False
        if kind != "report" and self._sj_ready(s, kind):      # the page says why; don't retry every minute
            return False
        last = self._sj_load()["last"].get(kind) or 0
        if kind == "pull":
            return now - last >= s["pull_every_min"] * 60
        at = self._at_today(s[f"{kind}_at"], now)
        return now >= at and last < at

    def seedjob_status(self) -> dict:
        s = self.seedjob_settings()
        d = self._sj_load()
        now = time.time()
        until = s.get("purge_dry_until")
        return {"settings": s, "last": d["last"], "report": d.get("report"), "last_purge": d.get("last_purge"),
                "alerts": d["alerts"][-10:],
                "ready": {k: self._sj_ready(s, k) for k in JOBS},
                "purge": {"dry": self._purge_dry(s, now), "mode": s["purge_mode"],
                          "dry_until": until, "dry_days_left": (max(0.0, (until - now) / 86400) if until else None)},
                "foreign_pull": self._sj_foreign_pull(now),
                "ingest_pending": d.get("ingest_pending"),
                "warnings": self._sj_dest_warnings(s)}
