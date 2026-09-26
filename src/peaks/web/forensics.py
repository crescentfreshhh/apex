"""Crash forensics: make an unexplained container restart explain itself.

Diagnostics only — nothing here changes behaviour.

- faulthandler: a native crash (segfault/abort in libav, CUDA, torch…) writes
  every thread's Python stack to `crash.log` (next to settings.json).
- heartbeat: every 30 s one `[health]` line — memory vs limit, ffmpeg/child
  processes, threads, GPU memory + decoder load, running jobs, busiest routes —
  to stderr and `crash.log`, so the last lines before a death show what climbed.
- clean-exit marker on shutdown; on the next start, a log that doesn't end with
  it means the previous run died abruptly, and that report is surfaced.
"""

from __future__ import annotations

import faulthandler
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

CLEAN = "[clean-exit]"
KEEP_LINES = 2000
_state: dict = {"fh": None, "path": None, "report": None, "routes": Counter(),
                "lock": threading.Lock(), "gpu": True, "busy": Counter()}


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write(line: str, echo: bool = True) -> None:
    if echo:
        print(line, file=sys.stderr, flush=True)
    fh = _state["fh"]
    if fh is None:
        return
    with _state["lock"]:
        try:
            fh.write(line + "\n")
            fh.flush()
        except (OSError, ValueError):
            pass


def _trim(path: Path) -> None:
    """Keep crash.log to its last KEEP_LINES lines."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    if len(lines) > KEEP_LINES:
        path.write_text("\n".join(lines[-KEEP_LINES:]) + "\n")


def previous_run_report(path: Path) -> dict | None:
    """If the log's last run didn't end with the clean-exit marker, describe
    how it ended: when, the last heartbeats, and any native crash trace."""
    try:
        lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines or lines[-1].split(" ", 1)[-1].startswith(CLEAN):
        return None
    starts = [i for i, ln in enumerate(lines) if "[start]" in ln]
    run = lines[starts[-1]:] if starts else lines
    health = [ln for ln in run if "[health]" in ln]
    trace_at = next((i for i, ln in enumerate(run) if "Fatal Python error" in ln), None)
    trace = "\n".join(run[trace_at:trace_at + 60]) if trace_at is not None else ""
    last = health[-1] if health else run[-1]
    return {"abrupt": True, "ended_after": last[:19], "started": run[0][:19],
            "last_health": health[-20:], "trace": trace,
            "kind": "native crash (see trace)" if trace else "killed without a trace (OOM or external kill likely)"}


def enable(log_dir: Path) -> dict | None:
    """Start forensics once per process. Returns the previous run's report."""
    if _state["fh"] is not None:
        return _state["report"]
    path = Path(log_dir) / "crash.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = previous_run_report(path) if path.exists() else None
        _trim(path)
        fh = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 — lives for the process
    except OSError:
        return None
    _state.update(fh=fh, path=path, report=report)
    faulthandler.enable(file=fh, all_threads=True)
    try:
        from importlib.metadata import version
        ver = version("peaks")
    except Exception:  # noqa: BLE001
        ver = "?"
    _write(f"{_stamp()} [start] peaks {ver} pid={os.getpid()} — crash traces + health → {path}")
    if report:
        _write(f"{_stamp()} [previous-run] ended abruptly after {report['ended_after']} — {report['kind']}")
        if report["last_health"]:
            _write(f"{_stamp()} [previous-run] last health: {report['last_health'][-1]}")
    return report


def mark_clean_exit() -> None:
    _write(f"{_stamp()} {CLEAN}", echo=False)


def note_request(path: str) -> None:
    _state["routes"][path] += 1


class busy:
    """`with forensics.busy("tier-train"):` — heavy in-process work (not a
    job) that the heartbeat should name, so a crash log shows what ran."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        with _state["lock"]:
            _state["busy"][self.name] += 1
        return self

    def __exit__(self, *exc):
        with _state["lock"]:
            _state["busy"][self.name] -= 1
            if _state["busy"][self.name] <= 0:
                del _state["busy"][self.name]
        return False


# --- the heartbeat -------------------------------------------------------------------

def _descendants(pid: int) -> list[tuple[int, str]]:
    """(pid, command name) of every descendant process, from /proc."""
    parent: dict[int, int] = {}
    name: dict[int, str] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        m = re.match(r"(\d+) \((.*)\) \S (\d+)", stat)
        if m:
            parent[int(m.group(1))] = int(m.group(3))
            name[int(m.group(1))] = m.group(2)
    out, frontier = [], [pid]
    while frontier:
        p = frontier.pop()
        for c, pp in parent.items():
            if pp == p:
                out.append((c, name.get(c, "?")))
                frontier.append(c)
    return out


def _gpu() -> str | None:
    if not _state["gpu"]:
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.decoder",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        used, total, dec = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return f"gpu={float(used) / 1024:.1f}/{float(total) / 1024:.0f}G dec={dec}%"
    except FileNotFoundError:
        _state["gpu"] = False            # no nvidia-smi here: stop asking
        return None
    except Exception:  # noqa: BLE001
        return "gpu=?"


def health_line(jobs=None) -> str:
    from . import memwatch

    rss = memwatch.rss_bytes()
    limit = memwatch.soft_limit_bytes()
    kids = _descendants(os.getpid())
    ffmpeg = sum(1 for _, n in kids if "ffmpeg" in n or "ffprobe" in n)
    parts = [f"rss={rss / 1073741824:.1f}G" + (f"/{limit / 1073741824:.1f}G limit" if limit else ""),
             f"ffmpeg={ffmpeg}", f"children={len(kids)}", f"threads={threading.active_count()}"]
    g = _gpu()
    if g:
        parts.append(g)
    if jobs is not None:
        running = sorted({j.kind for j in jobs.list() if j.status == "running"})
        parts.append("jobs=" + (",".join(running) or "-"))
    work = sorted(k for k, n in _state["busy"].items() if n > 0)
    if work:
        parts.append("work=" + ",".join(work))
    routes, _state["routes"] = _state["routes"], Counter()
    if routes:
        parts.append("req=" + ",".join(f"{r}×{n}" for r, n in routes.most_common(3)))
    return f"{_stamp()} [health] " + " · ".join(parts)


def start_heartbeat(jobs=None, every: float = 30.0) -> threading.Event:
    stop = threading.Event()

    def _loop():
        beats = 0
        while not stop.wait(every):
            try:
                _write(health_line(jobs))
                beats += 1
                if beats % 200 == 0 and _state["path"]:
                    with _state["lock"]:
                        _trim(_state["path"])
            except Exception:  # noqa: BLE001 — a watchdog must never crash the app
                pass

    threading.Thread(target=_loop, daemon=True, name="peaks-health").start()
    return stop


def report() -> dict | None:
    return _state["report"]


def log_path() -> Path | None:
    return _state["path"]
