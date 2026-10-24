"""The factory floor's data: every station of the pipeline, from the real jobs
and logs — what's running, how far along, the files in flight (anonymous:
a category and a percentage, never a name) and a ticker with file names
stripped. The page (static app.js, the "floor" view) draws it in ASCII."""

from __future__ import annotations

import time


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
    lanes = [{"id": int(k), "cat": v.get("cat"), "pct": round(float(v.get("pct") or 0), 3)}
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
    return {"now": now, "stations": stations, "lanes": lanes, "shelves": shelves,
            "counts": {"dupes": len(dupes.get("groups") or []), "upgrades": upg},
            "ticker": ticker[-14:]}


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
