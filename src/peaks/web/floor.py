"""The factory floor's data: every station of the pipeline, from the real jobs
and logs — what's running, how far along, the files in flight (anonymous:
a category, a percentage and a speed, never a name), each machine's readout,
a 60-minute history for the sparklines, and a ticker with file names
stripped. /api/floor/day adds the 24-hour timeline, today's totals and disk
space. The page (static/floor/) draws it as an ASCII control room."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

_HIST: dict = {}               # per service: {"series": {metric: [(minute, value)]}, "embed": (t, done)}
_HIST_LOCK = threading.Lock()
_GPU: dict = {"at": 0.0, "value": None}


# station → the job kinds that make it work
STATION_JOBS = {
    "ingest": ("ingest",), "embed": ("embed", "sync", "warmup"), "picture": ("picture", "picture-dupes"),
    "dupes": ("dupes", "dupe-drain", "library"), "upgrades": ("upgrades",), "backup": ("backup", "restore"),
    "pull": ("seed-pull",), "purge": ("seed-purge",), "report": ("seed-report",),
    "taste": ("taste-measure", "train"),
}


def _pct(progress: dict) -> float | None:
    p = progress or {}
    if p.get("pct") is not None:
        return max(0.0, min(1.0, float(p["pct"])))
    done, total = p.get("done"), p.get("total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total:
        return max(0.0, min(1.0, done / total))
    return None


def floor_state(service, jobs) -> dict:
    now = time.time()
    running = {j.kind: j for j in jobs.list() if j.status == "running"}
    recent = sorted((j for j in jobs.list() if j.status != "running" and (j.finished or 0) > now - 90),
                    key=lambda j: j.finished or 0)
    stations = {}
    for st, kinds in STATION_JOBS.items():
        j = next((running[k] for k in kinds if k in running), None)
        last = next((x for x in reversed(recent) if x.kind in kinds), None)
        state = "running" if j else ("error" if last and last.status == "error" else
                                     "done" if last and last.status == "done" else "idle")
        stations[st] = {"state": state, "pct": _pct(j.progress) if j else None}
    live = service._sj_live()
    pull_live = live.get("pull") or {}
    lanes = [{"id": int(k), "cat": v.get("cat"), "pct": round(float(v.get("pct") or 0), 3),
              "mbps": round(float(v.get("speed") or 0) / 1e6, 2)}
             for k, v in list((pull_live.get("lanes") or {}).items())]
    stations["seedbox"] = {"state": "running" if pull_live.get("stage") == "listing" else "idle",
                           "stage": "listing" if pull_live.get("stage") == "listing" else ""}
    stations["pull"]["queued"] = max(0, int(pull_live.get("queued") or 0) - int(pull_live.get("done") or 0) - len(lanes))
    try:
        sjs = service.seedjob_status()
        stations["purge"]["dry"] = sjs["purge"]["dry"]
        stations["pull"]["taken_over"] = bool(sjs["settings"]["cutover"])
        lp = sjs.get("last_purge") or {}
        stations["purge"]["last"] = {"delete": lp.get("delete"), "dry": lp.get("dry")}
    except Exception:  # noqa: BLE001
        pass
    # library shelves: tier counts from the cached scene list (never a Stash call here)
    shelves = {}
    cached = getattr(service, "_cat_cache", None)
    if cached:
        for r in cached[1]:
            shelves[r["tier"]] = shelves.get(r["tier"], 0) + 1
    dupes = service.cached_duplicates() or {}
    try:
        upg = service.upgrade_counts()["found"]
    except Exception:  # noqa: BLE001
        upg = 0
    # the seedbox jobs' own ticker never holds a path; other jobs show only what
    # they are and how far along — never their log text (titles, paths)
    ticker = list(live.get("ticker") or [])
    for j in list(running.values()):
        if j.kind.startswith("seed-"):
            continue
        pc = _pct(j.progress)
        ticker.append({"at": now, "text": f"{j.kind}: working" + (f" {round(pc * 100)}%" if pc is not None else "")})
    for j in recent[-3:]:
        if not j.kind.startswith("seed-"):
            ticker.append({"at": j.finished or now, "text": f"{j.kind}: {j.status}"})
    ticker.sort(key=lambda t: t["at"])
    # readouts: real numbers per machine (never a name)
    pull_mbps = round(sum(ln.get("mbps", 0) for ln in lanes), 2)
    stations["pull"]["mbps"] = pull_mbps
    ej = next((running[k] for k in ("embed", "sync") if k in running), None)
    stations["embed"]["rate"] = _embed_rate(service, ej, now)
    stations["embed"]["done"] = (ej.progress or {}).get("done") if ej else None
    stations["embed"]["total"] = (ej.progress or {}).get("total") if ej else None
    stations["embed"]["gpu"] = gpu_load(now)
    pj = next((running[k] for k in STATION_JOBS["picture"] if k in running), None)
    stations["picture"]["done"] = (pj.progress or {}).get("done") if pj else None
    stations["picture"]["total"] = (pj.progress or {}).get("total") if pj else None
    try:
        q = service.dupe_queue_status()
        stations["dupes"]["queued"] = len(q["queued"]) + len(q["running"])
    except Exception:  # noqa: BLE001
        stations["dupes"]["queued"] = 0
    stations["dupes"]["groups"] = len(dupes.get("groups") or [])
    try:
        lp = service.seedjob_status().get("last_purge") or {}
        stations["purge"]["ready"] = lp.get("delete")
        stations["purge"]["gb"] = round((lp.get("freed_bytes") or 0) / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    stations["upgrades"]["found"] = upg
    try:
        stations["upgrades"]["last"] = service._upg_load().get("last_pass")
    except Exception:  # noqa: BLE001
        pass
    try:
        snaps = service.list_snapshots()
        stations["backup"]["last"] = float(snaps[0]["created"]) if snaps and snaps[0].get("created") else None
    except Exception:  # noqa: BLE001
        stations["backup"]["last"] = None
    series = _record(service, now, {"pull_mbps": pull_mbps, "embeds": stations["embed"]["done"] or 0})
    return {"now": now, "stations": stations, "lanes": lanes, "shelves": shelves,
            "counts": {"dupes": len(dupes.get("groups") or []), "upgrades": upg},
            "series": series, "ticker": ticker[-14:]}


def gpu_load(now: float | None = None) -> dict | None:
    """{util %, mem_mb} from nvidia-smi (cached 5 s), or None without a GPU."""
    now = time.time() if now is None else now
    if now - _GPU["at"] < 5:
        return _GPU["value"]
    _GPU["at"] = now
    exe = shutil.which("nvidia-smi")
    if not exe:
        _GPU["value"] = None
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        util, mem = (float(x) for x in out.split(","))
        _GPU["value"] = {"util": round(util), "mem_mb": round(mem)}
    except Exception:  # noqa: BLE001 — a GPU that can't answer just isn't shown
        _GPU["value"] = None
    return _GPU["value"]


def _hist(service) -> dict:
    with _HIST_LOCK:
        return _HIST.setdefault(id(service), {"series": {}, "embed": None})


def _embed_rate(service, job, now: float) -> float | None:
    """Scenes per minute for the running embed, from how its count moves."""
    h = _hist(service)
    if job is None:
        h["embed"] = None
        return None
    done = (job.progress or {}).get("done") or 0
    prev = h.get("embed")
    if not prev or prev[2] != job.id:
        h["embed"] = (now, done, job.id, None)
        return None
    t0, d0, _, rate = prev
    if now - t0 >= 20:
        rate = round((done - d0) / ((now - t0) / 60), 2)
        h["embed"] = (now, done, job.id, rate)
    return rate


def _record(service, now: float, values: dict) -> dict:
    """Per-minute history (last 60) of a few live numbers, for the sparklines."""
    h = _hist(service)["series"]
    minute = int(now // 60)
    with _HIST_LOCK:
        for k, v in values.items():
            s = h.setdefault(k, [])
            if s and s[-1][0] == minute:
                s[-1] = (minute, max(s[-1][1], v))
            else:
                s.append((minute, v))
            del s[:-60]
        return {k: [round(v, 2) for _, v in s] for k, s in h.items()}


def floor_day(service, now: float | None = None) -> dict:
    """The day: per station, 144 ten-minute buckets of activity; today's totals;
    disk space (library, config, seedbox). Counts and sizes only."""
    now = time.time() if now is None else now
    start = now - 24 * 3600
    stations = ("pull", "ingest", "embed", "picture", "dupes", "library", "purge", "upgrades", "backup")
    buckets = {s: [0] * 144 for s in stations}
    for e in floor_replay(service, 24):
        i = int((e["at"] - start) // 600)
        if 0 <= i < 144 and e["station"] in buckets:
            buckets[e["station"]][i] += max(1, int(e.get("n") or 1))
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    today = {"pulled": 0, "pulled_gb": 0.0, "graded": 0, "rejected": 0, "freed_gb": 0.0, "embedded": 0,
             "upgrades": 0}
    try:
        for p in service._sj_ledger(midnight):
            today["pulled"] += 1
            today["pulled_gb"] += (p.get("size") or 0) / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        for e in service.action_log().tail(20000):
            try:
                at = time.mktime(time.strptime(e.get("ts", "")[:19], "%Y-%m-%dT%H:%M:%S"))
            except (ValueError, TypeError):
                continue
            if at < midnight:
                continue
            a = e.get("action")
            if a == "grade":
                today["graded"] += 1
                if e.get("grade") == "reject":
                    today["rejected"] += 1
            elif a == "upgrade-grab":
                today["upgrades"] += 1
            elif a in ("delete", "duplicate"):
                today["freed_gb"] += float(e.get("freed_bytes") or e.get("bytes") or 0) / 1e9
            elif a == "ingest":
                st = (e.get("stages") or {}).get("embed") or ""
                try:
                    today["embedded"] += int(str(st).split(" ", 1)[0])
                except ValueError:
                    pass
    except Exception:  # noqa: BLE001
        pass
    today = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in today.items()}
    disks = {}
    for name, path in (("library", service.cfg.library.path or "/data"), ("config", str(service.cfg.modeling.dir))):
        try:
            p = Path(path)
            u = shutil.disk_usage(p if p.exists() else p.parent)
            disks[name] = {"total": u.total, "free": u.free}
        except OSError:
            pass
    sb = getattr(service, "_floor_sb_free", None)
    if not sb or now - sb[0] > 300:
        val = None
        c = service.cfg.seedbox
        if c.qbt_url and c.qbt_user and c.qbt_password:
            try:
                val = service._sj_qbt().free_space()
            except Exception:  # noqa: BLE001
                val = None
        service._floor_sb_free = sb = (now, val)
    if sb[1] is not None:
        disks["seedbox"] = {"free": sb[1]}
    return {"now": now, "start": start, "buckets": buckets, "today": today, "disks": disks}


def floor_replay(service, hours: float = 24) -> list[dict]:
    """The last day as events for the fast-forward replay: runs.log + the
    action log, oldest first, counts only."""
    from ..seedbox import parse_all_runs

    now = time.time()
    out = []
    try:
        runs = parse_all_runs((service._sj_dir() / "runs.log").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        runs = []
    for r in runs:
        if r["at"] >= now - hours * 3600:
            out.append({"at": r["at"], "station": "pull" if r["kind"] == "pull" else "purge" if r["kind"] == "purge"
                        else r["kind"], "ok": r["ok"], "n": r["count"] or 0})
    station_of = {"grade": "library", "duplicate": "dupes", "delete": "dupes", "upgrade-grab": "upgrades",
                  "ingest": "ingest", "seedbox-pull": "pull", "seedbox-purge": "purge", "tag-sync": "library"}
    try:
        for e in service.action_log().tail(5000):
            try:
                at = time.mktime(time.strptime(e.get("ts", "")[:19], "%Y-%m-%dT%H:%M:%S"))
            except (ValueError, TypeError):
                continue
            st = station_of.get(e.get("action"))
            if st and at >= now - hours * 3600:
                out.append({"at": at, "station": st, "ok": True, "n": 1})
    except Exception:  # noqa: BLE001
        pass
    out.sort(key=lambda x: x["at"])
    return out[-600:]
