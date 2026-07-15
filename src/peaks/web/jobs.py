"""Background job manager for the web UI.

Long operations (embed, score, index rebuild) run in a worker thread so the
HTTP request returns immediately; the UI polls status. One job per `kind` may
run at a time. Log lines are captured (bounded) for a live console view.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


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
    _log: deque = field(default_factory=lambda: deque(maxlen=500))
    _cancel: threading.Event = field(default_factory=threading.Event)

    def log(self, msg: str) -> None:
        self._log.append(str(msg))

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
            "log": list(self._log)[-log_tail:],
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._by_kind: dict[str, str] = {}  # kind -> current running job id
        self._lock = threading.Lock()
        self._seq = 0

    def _running_locked(self, kind: str) -> Job | None:
        """Caller must hold self._lock (the lock is not reentrant)."""
        jid = self._by_kind.get(kind)
        job = self._jobs.get(jid) if jid else None
        return job if job and job.status == "running" else None

    def running(self, kind: str) -> Job | None:
        with self._lock:
            return self._running_locked(kind)

    def start(self, kind: str, target: Callable[[Job], dict | None]) -> Job:
        """Start `target(job)` in a thread. Raises if a job of `kind` is live."""
        with self._lock:
            if self._running_locked(kind) is not None:
                raise RuntimeError(f"a '{kind}' job is already running")
            self._seq += 1
            job = Job(id=f"{kind}-{self._seq}", kind=kind)
            self._jobs[job.id] = job
            self._by_kind[kind] = job.id

        def _run():
            try:
                result = target(job)
                job.result = result if isinstance(result, dict) else None
                job.status = "cancelled" if job.cancelled else "done"
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                job.error = f"{type(exc).__name__}: {exc}"
                job.log("ERROR: " + job.error)
                job.log(traceback.format_exc().splitlines()[-1])
                job.status = "error"
            finally:
                job.finished = time.time()

        threading.Thread(target=_run, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)
