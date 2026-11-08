"""Background job manager for the web UI.

Long operations (embed, score, index rebuild) run in a worker thread so the
HTTP request returns immediately; the UI polls status. One job per `kind` may
run at a time. Log lines are captured (bounded) for a live console view.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RATE_WINDOW = 180.0          # seconds of progress samples the speed / time-left are read from
HISTORY_KEEP = 200           # finished jobs remembered across restarts


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | error | cancelled
    started: float = field(default_factory=time.time)
    finished: float | None = None
    progress: dict = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None
    _log: deque = field(default_factory=lambda: deque(maxlen=500))      # (time, line)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _samples: deque = field(default_factory=lambda: deque(maxlen=400))  # (time, done-or-pct)

    def log(self, msg: str) -> None:
        self._log.append((time.time(), str(msg)))

    def lines(self, since: float | None = None) -> list[tuple[float, str]]:
        return [x for x in list(self._log) if since is None or x[0] > since]

    def timing(self) -> dict:
        """What the job is doing and how fast: percent, time per item, items per
        minute, running time and time left. Read from the job's own progress —
        `done`/`total`, or a `pct` for staged jobs; its own `eta_s` wins — with
        speed measured over the last few minutes (sampled whenever it's read)."""
        now = (self.finished or time.time())
        p = dict(self.progress or {})
        done, total = p.get("done"), p.get("total")
        counted = isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0
        if counted:
            pct = min(1.0, max(0.0, done / total))
            x = float(done)
        elif isinstance(p.get("pct"), (int, float)):
            pct = min(1.0, max(0.0, float(p["pct"])))
            x = pct
        else:
            pct, x = None, None
        elapsed = max(0.0, now - self.started)
        if x is not None and self.status == "running":
            if not self._samples or self._samples[-1][1] != x or now - self._samples[-1][0] > 30:
                self._samples.append((now, x))
        rate = None                                         # units (items, or pct) per second
        win = [s for s in self._samples if now - s[0] <= RATE_WINDOW]
        if len(win) >= 2 and win[-1][0] - win[0][0] >= 5 and win[-1][1] > win[0][1]:
            rate = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0])
        elif x and elapsed >= 5:
            rate = x / elapsed                              # early on: the average so far
        out = {"pct": round(pct, 4) if pct is not None else None, "stage": p.get("stage"),
               "done": done if counted else None, "total": total if counted else None,
               "elapsed_s": round(elapsed), "per_item_s": None, "rate_per_min": None, "eta_s": None}
        if counted and rate:
            out["per_item_s"] = round(1.0 / rate, 2)
            out["rate_per_min"] = round(rate * 60, 1)
        if self.status == "running" and self.cancelled:          # asked to stop: it's winding down
            pf = getattr(self, "paused_for", None)
            out["stage"] = f"pausing for {pf}" if pf else "stopping"
            out["stopping"] = True
            return out
        if self.status == "running":
            if isinstance(p.get("eta_s"), (int, float)):
                out["eta_s"] = round(p["eta_s"])
            elif rate and pct is not None:
                left = (total - done) if counted else (1.0 - x)
                out["eta_s"] = round(max(0.0, left / rate))
        return out

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def as_dict(self, log_tail: int = 60) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "log": [m for _, m in list(self._log)[-log_tail:]],
            "timing": self.timing(),
        }

    def summary(self) -> dict:
        """One line for the history: what ran, how long, how far, how it ended."""
        t = self.timing()
        r = self.result or {}
        bits = [f"{k} {v}" for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)][:4]
        return {"id": self.id, "kind": self.kind, "status": self.status, "started": self.started,
                "finished": self.finished, "elapsed_s": t["elapsed_s"], "done": t["done"], "total": t["total"],
                "error": self.error, "result": " · ".join(bits), "paused_for": getattr(self, "paused_for", None)}


KEEP_JOBS = 200


class Busy(RuntimeError):
    """Another job that must not run alongside is running."""

    def __init__(self, kind: str):
        super().__init__(f"a '{kind}' job is running")
        self.kind = kind


class JobManager:
    def __init__(self, history_path: str | Path | None = None):
        self._history_path = Path(history_path) if history_path else None
        self.paused: dict[str, dict] = {}    # kind → {for, args, at}: background work waiting its turn
        self._jobs: dict[str, Job] = {}
        self._by_kind: dict[str, str] = {}  # kind -> current running job id
        self._lock = threading.Lock()
        self._plock = threading.Lock()       # the waiting list (request threads + the scheduler)
        self._hlock = threading.Lock()       # the history file — never held with _lock
        self._seq = 0
        self.on_finish: Callable[[Job], None] | None = None   # e.g. hand GPU memory back
        self.stopped_until: dict[str, float] = {}   # kind → no automatic start before this (you stopped it)
        # "Pause jobs for an hour": nothing starts on its own before this; the library
        # jobs the pause stopped (ingest / embed / fix) are started again after it
        self.hold_until = 0.0
        self.held_restart: list[str] = []

    def hold(self, seconds: float) -> None:
        self.hold_until = time.time() + max(0.0, float(seconds))

    def unhold(self) -> None:
        self.hold_until = 0.0

    def held(self) -> bool:
        return time.time() < self.hold_until

    def stop_by_you(self, kind: str) -> None:
        """You stopped `kind` (or its waiting turn): nothing starts it on its own
        again until tomorrow — starting it yourself lifts that."""
        import datetime as _dt

        tomorrow = _dt.datetime.combine(_dt.date.today() + _dt.timedelta(days=1), _dt.time(4, 0))
        self.stopped_until[kind] = tomorrow.timestamp()
        self.pause_pop(kind)

    def stopped_by_you(self, kind: str) -> bool:
        return time.time() < self.stopped_until.get(kind, 0.0)

    # --- the waiting list: read and changed only under its lock ---------------------------

    def pause_put(self, kind: str, entry: dict) -> None:
        with self._plock:
            self.paused[kind] = entry

    def pause_pop(self, kind: str) -> dict | None:
        with self._plock:
            return self.paused.pop(kind, None)

    def paused_snapshot(self) -> dict[str, dict]:
        with self._plock:
            return {k: dict(v) for k, v in self.paused.items()}

    def _running_locked(self, kind: str) -> Job | None:
        """Caller must hold self._lock (the lock is not reentrant)."""
        jid = self._by_kind.get(kind)
        job = self._jobs.get(jid) if jid else None
        return job if job and job.status == "running" else None

    def running(self, kind: str) -> Job | None:
        with self._lock:
            return self._running_locked(kind)

    def start(self, kind: str, target: Callable[[Job], dict | None], args: dict | None = None,
              blockers: tuple[str, ...] = ()) -> Job:
        """Start `target(job)` in a thread. Raises if a job of `kind` is live — or,
        atomically with that check, if any of `blockers` is (Busy, naming it)."""
        with self._lock:
            if self._running_locked(kind) is not None:
                raise RuntimeError(f"a '{kind}' job is already running")
            for b in blockers:
                if self._running_locked(b) is not None:
                    raise Busy(b)
            self._seq += 1
            job = Job(id=f"{kind}-{self._seq}", kind=kind)
            job.args = dict(args or {})
            self._jobs[job.id] = job
            self._by_kind[kind] = job.id
            self._prune_locked()

        def _run():
            status = "error"
            try:
                result = target(job)
                job.result = result if isinstance(result, dict) else None
                status = ("paused" if getattr(job, "paused_for", None) else "cancelled") if job.cancelled else "done"
            except InterruptedError:                 # stopped mid-stage: a stop, not an error
                status = ("paused" if getattr(job, "paused_for", None) else "cancelled") if job.cancelled else "error"
                if status == "error":
                    job.error = "interrupted"
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                job.error = f"{type(exc).__name__}: {exc}"
                job.log("ERROR: " + job.error)
                job.log(traceback.format_exc().splitlines()[-1])
            finally:
                # finished time and history first, the status last: anyone who sees
                # the job done also finds it in the history
                job.finished = time.time()
                self._remember(job, status)
                job.status = status
                if self.on_finish is not None:
                    try:
                        self.on_finish(job)
                    except Exception:  # noqa: BLE001 — a hook never fails a job
                        pass

        threading.Thread(target=_run, daemon=True, name=f"job-{job.id}").start()
        return job

    def _remember(self, job: Job, status: str | None = None) -> None:
        """Append a finished job to the history file (kept to the last HISTORY_KEEP)."""
        if not self._history_path:
            return
        try:
            with self._hlock:
                p = self._history_path
                p.parent.mkdir(parents=True, exist_ok=True)
                lines = p.read_text().splitlines() if p.exists() else []
                lines = (lines + [json.dumps({**job.summary(), "status": status or job.status})])[-HISTORY_KEEP:]
                tmp = p.with_suffix(".tmp")
                tmp.write_text("\n".join(lines) + "\n")
                tmp.replace(p)
        except Exception:  # noqa: BLE001 — history is a convenience
            pass

    def _prune_locked(self) -> None:
        """Keep the last KEEP_JOBS finished jobs in memory (the history file has the rest)."""
        done = [j for j in self._jobs.values() if j.status != "running"]
        if len(done) <= KEEP_JOBS:
            return
        for j in sorted(done, key=lambda j: j.started)[: len(done) - KEEP_JOBS]:
            self._jobs.pop(j.id, None)

    def _history_file(self) -> list[dict]:
        """The remembered jobs on disk, newest first — parsed once per change of the
        file (the Activity log asked every 2 s and re-read it each time)."""
        p = self._history_path
        if not p:
            return []
        try:
            st = p.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return []
        hit = getattr(self, "_hist_cache", None)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        out = []
        try:
            for ln in reversed(p.read_text().splitlines()):
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
        except OSError:
            pass
        self._hist_cache = (stamp, out)
        return out

    def history(self) -> list[dict]:
        """Finished jobs, newest first: this run's plus the remembered ones."""
        seen, out = set(), []
        for j in self.list():
            if j.status != "running":
                seen.add(j.id + str(j.started))
                out.append(j.summary())
        for d in self._history_file():
            k = d.get("id", "") + str(d.get("started"))
            if k not in seen:
                seen.add(k)
                out.append(d)
        return sorted(out, key=lambda d: d.get("finished") or 0, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.started, reverse=True)
