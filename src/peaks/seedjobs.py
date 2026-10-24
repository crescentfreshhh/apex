"""The seedbox jobs Peaks runs (they used to be Unraid User Scripts): pull new
files from the seedbox, purge torrents once they're safely on Jarvis, and the
daily report. Pure rules here — what's new, what may be deleted, what the
report says — so they're testable without a seedbox; Service runs rclone and
talks to qBittorrent / Whisparr (web/seedjobs.py).

Same rules as the scripts they replace:
- pull: list each category with the filter file, skip what's in
  `<category>.done`, copy the rest; each finished file gets its permissions
  fixed and is logged as done at once. Never deletes anything on Jarvis.
- purge: a torrent may be deleted only when it's 100 % downloaded and every
  video / .funscript file in it is recorded as pulled (1 / vr) — or, for
  Whisparr's folder, Whisparr imported it and it's not in Whisparr's queue.
  Anything that can't be checked is kept.
- report: what came in and how the machinery ran — counts and sizes only,
  never a file name or a title.
"""

from __future__ import annotations

import json
import re
import time

DAY = 86400.0

# the purge's (and the pull filter's) idea of a file worth keeping
_SAMPLE = re.compile(r"(^|/)sample/|sample", re.I)
_JUNK = re.compile(r"\.(nfo|csv|!qb|partial)$", re.I)
_VIDEO = re.compile(r"\.(mkv|mp4|m4v|mov|wmv|avi|ts|webm|funscript)$", re.I)


def wanted(path: str) -> bool:
    if _SAMPLE.search(path or "") or _JUNK.search(path or ""):
        return False
    return bool(_VIDEO.search(path or ""))


# --- pull --------------------------------------------------------------------------

def read_done(text: str) -> set[str]:
    return {ln.strip() for ln in (text or "").splitlines() if ln.strip()}


def new_files(listing: list[str], done: set[str]) -> list[str]:
    """Files on the box not yet recorded as pulled, in a stable order."""
    return sorted({p.strip() for p in listing if p.strip()} - done)


def rclone_stats(line: str) -> dict | None:
    """One `--use-json-log` line → {bytes, total, speed, eta} when it carries
    transfer stats, else None."""
    try:
        d = json.loads(line)
    except (ValueError, TypeError):
        return None
    st = d.get("stats") if isinstance(d, dict) else None
    if not isinstance(st, dict):
        return None
    return {"bytes": int(st.get("bytes") or 0), "total": int(st.get("totalBytes") or 0),
            "speed": float(st.get("speed") or 0.0), "eta": st.get("eta")}


def run_line(kind: str, ok: bool, count: int, nbytes: int | None = None, at: float | None = None) -> str:
    """One runs.log line: `<epoch>|<kind>|<ok|fail>|<count>[|<bytes>]`."""
    at = int(at if at is not None else time.time())
    base = f"{at}|{kind}|{'ok' if ok else 'fail'}|{int(count)}"
    return base + (f"|{int(nbytes)}" if nbytes is not None else "")


# --- purge -------------------------------------------------------------------------

def purge_plan(torrents: list[dict], files: dict[str, list[dict] | None], done: dict[str, set[str]],
               base: str, folders: tuple[str, ...] = ("1", "vr"), whisparr_folder: str = "arr/whisparr",
               whisparr: dict | None = None) -> list[dict]:
    """Decide every torrent: [{hash, name, size, folder, delete: bool, why}].
    `files[hash]` is the torrent's file list (None = couldn't be read);
    `whisparr` = {ok, imported: set, queued: set} (hashes upper-case)."""
    base = base.rstrip("/")
    wf = whisparr_folder.strip("/")
    out = []
    for t in torrents:
        save = (t.get("save_path") or "").rstrip("/")
        folder = next((f for f in folders if save == f"{base}/{f}"), None)
        is_w = bool(wf) and save == f"{base}/{wf}"
        if folder is None and not is_w:
            continue                                     # other folders: never touched
        d = {"hash": t.get("hash"), "name": t.get("name") or "?", "size": float(t.get("size") or 0),
             "folder": folder or wf, "delete": False}
        if (t.get("progress") or 0) < 1:
            out.append({**d, "why": "still downloading"})
            continue
        if is_w:
            w = whisparr or {}
            h = (t.get("hash") or "").upper()
            if not w.get("ok"):
                out.append({**d, "why": "couldn't check Whisparr"})
            elif h in w.get("queued", set()):
                out.append({**d, "why": "Whisparr hasn't finished importing it"})
            elif h not in w.get("imported", set()):
                out.append({**d, "why": "no import recorded in Whisparr"})
            else:
                out.append({**d, "delete": True, "why": "Whisparr imported it"})
            continue
        fl = files.get(t.get("hash"))
        if fl is None:
            out.append({**d, "why": "couldn't list its files"})
            continue
        need = [f.get("name") or "" for f in fl if (f.get("priority", 1) != 0) and wanted(f.get("name") or "")]
        if not need:
            out.append({**d, "why": "no video files found to verify"})
            continue
        missing = [p for p in need if p not in done.get(folder, set())]
        if missing:
            out.append({**d, "why": f"{len(missing)} of {len(need)} files not pulled yet"})
            continue
        out.append({**d, "delete": True, "why": f"all {len(need)} files confirmed on Jarvis"})
    return out


# --- the daily report: counts only -------------------------------------------------

def report(runs: list[dict], pulled: list[dict], scenes: dict[str, dict], now: float,
           purge_mode: str) -> dict:
    """runs: [{at, kind, ok, count, bytes}]; pulled: the pull ledger
    [{t, cat, size, scene_id?}]; scenes: scene_id → {tier, identified}.
    Never includes a file name or a title — only counts and sizes."""
    day = [r for r in runs if r["at"] >= now - DAY]
    p_ok = [r for r in day if r["kind"] == "pull" and r["ok"]]
    p_fail = [r for r in day if r["kind"] == "pull" and not r["ok"]]
    purges = [r for r in day if r["kind"] == "purge"]
    last_pull = max((r["at"] for r in runs if r["kind"] == "pull" and r["ok"]), default=None)
    came = [p for p in pulled if p.get("t", 0) >= now - DAY]
    by_cat: dict[str, dict] = {}
    for p in came:
        c = by_cat.setdefault(p.get("cat") or "?", {"files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += int(p.get("size") or 0)
    videos = [p for p in came if not str(p.get("ext", "")).lower() == "funscript"]
    found = [scenes.get(str(p.get("scene_id"))) for p in videos if p.get("scene_id")]
    found = [s for s in found if s]
    tiers: dict[str, int] = {}
    for s in found:
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
    issues = []
    if last_pull is None or now - last_pull > 2 * 3600:
        issues.append("no successful pull in over 2 h")
    if p_fail:
        issues.append(f"{len(p_fail)} pull run(s) failed")
    if any(not r["ok"] for r in purges):
        issues.append("the purge failed")
    return {
        "at": now, "ok": not issues, "issues": issues,
        "came_in": {"files": len(came), "bytes": sum(int(p.get("size") or 0) for p in came),
                    "by_category": by_cat,
                    "in_stash": len(found), "identified": sum(1 for s in found if s.get("identified")),
                    "graded": sum(v for t, v in tiers.items() if t not in ("unreviewed", "anomaly")),
                    "unreviewed": tiers.get("unreviewed", 0), "by_tier": tiers,
                    "not_in_stash_yet": len(videos) - len(found)},
        "pull": {"ok": len(p_ok), "failed": len(p_fail), "last_ok": last_pull},
        "purge": {"runs": len(purges), "deleted": sum(r.get("count") or 0 for r in purges if r["ok"]),
                  "freed_bytes": sum(r.get("bytes") or 0 for r in purges if r["ok"]), "mode": purge_mode},
    }
