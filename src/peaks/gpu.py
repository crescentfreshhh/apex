"""One owner for the GPU.

Before this, "one GPU lane" was a set of job-kind rules checked when a job
started — and text / sound search ran their CUDA kernels in request threads
right beside an embed's batches, a sense slow to stop could be overtaken after
three minutes, and nothing knew who had the card.

The lease:

* ``owner(kind, job)`` — a job that loads a model holds the GPU for as long as
  it holds the model (embed, fix, lab, the senses, score). A second one waits
  ("waiting for the GPU — Embedding has it"), never overlaps. Re-entrant per
  thread (an ingest's embed stage, one embed pass per model).
* ``batch()`` — one GPU batch. Every call on a guarded model goes through it.
* ``borrow(what)`` — a request's short use (text / sound search): it needs no
  owner lease, and jumps the queue at the next batch boundary, so a search
  waits at most one batch — never a whole job.
* ``resident(name, unload)`` — a model kept between uses (the text towers): it
  leaves the GPU after 10 idle minutes, or when a job takes the lease and it
  has been idle a minute.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Callable

JOB_UNLOADS_IDLE_S = 60.0       # a job taking the lease drops text models idle this long


class _Cancelled(Exception):
    pass


class GpuLease:
    def __init__(self, clock: Callable[[], float] | None = None, free: Callable[[], None] | None = None):
        self.clock = clock or (lambda: time.time())      # looked up each call (tests move the clock)
        self._free = free
        self._cv = threading.Condition()
        self._owner: dict | None = None          # {kind, thread, depth, since}
        self._dev_thread: int | None = None      # who is inside a batch / borrow
        self._dev_depth = 0
        self._borrowers = 0                      # requests waiting for the device
        self._residents: dict[str, dict] = {}    # name → {unload, used}
        self.waits = 0                           # owners that had to wait (for tests and the log)

    # --- the job lease ---------------------------------------------------------------------

    def holder(self) -> dict | None:
        with self._cv:
            o = self._owner
            return None if o is None else {"kind": o["kind"], "since": o["since"]}

    @contextlib.contextmanager
    def owner(self, kind: str, job=None, label: str | None = None):
        """Hold the GPU for a job's model. Waits (checking `job.cancelled`)
        while another job holds it; returns without running if cancelled."""
        me = threading.get_ident()
        logged = False
        with self._cv:
            while self._owner is not None and self._owner["thread"] != me:
                if job is not None and getattr(job, "cancelled", False):
                    raise _Cancelled()
                if not logged:
                    logged = True
                    self.waits += 1
                    if job is not None and hasattr(job, "log"):
                        job.log(f"waiting for the GPU — {self._owner.get('label') or self._owner['kind']} has it")
                self._cv.wait(0.5)
            if self._owner is None:
                self._owner = {"kind": kind, "label": label or kind, "thread": me, "depth": 0, "since": self.clock()}
                fresh = True
            else:
                fresh = False
            self._owner["depth"] += 1
        if fresh:
            self.unload_idle(JOB_UNLOADS_IDLE_S)          # idle text towers make room for the job's model
        try:
            yield self
        finally:
            with self._cv:
                self._owner["depth"] -= 1
                done = self._owner["depth"] == 0
                if done:
                    self._owner = None
                self._cv.notify_all()
            if done:
                self._release_memory()

    def run_owned(self, kind: str, job, fn, label: str | None = None):
        """`fn()` under the lease; a job stopped while it waited just returns None."""
        try:
            with self.owner(kind, job, label):
                return fn()
        except _Cancelled:
            if job is not None and hasattr(job, "log"):
                job.log("stopped while waiting for the GPU")
            return None

    # --- the device: batches and borrows -----------------------------------------------------

    @contextlib.contextmanager
    def _device(self, borrower: bool):
        me = threading.get_ident()
        with self._cv:
            if borrower:
                self._borrowers += 1
            try:
                while self._dev_thread is not None and self._dev_thread != me or (
                        not borrower and self._borrowers and self._dev_thread != me):
                    self._cv.wait(0.5)        # a job's batch waits while a request is queued
            finally:
                if borrower:
                    self._borrowers -= 1
            self._dev_thread = me
            self._dev_depth += 1
        try:
            yield
        finally:
            with self._cv:
                self._dev_depth -= 1
                if self._dev_depth == 0:
                    self._dev_thread = None
                self._cv.notify_all()

    def batch(self):
        return self._device(borrower=False)

    def borrow(self, what: str = ""):
        return self._device(borrower=True)

    def guard(self, model, borrow: bool = False):
        return model if model is None or isinstance(model, _Guarded) else _Guarded(model, self, borrow)

    # --- models kept between uses ---------------------------------------------------------

    def resident(self, name: str, unload: Callable[[], None]) -> None:
        with self._cv:
            self._residents[name] = {"unload": unload, "used": self.clock()}

    def forget(self, name: str) -> None:
        """A resident unloaded elsewhere (the scheduler's 10-minute rule)."""
        with self._cv:
            self._residents.pop(name, None)

    def used(self, name: str | None = None) -> None:
        with self._cv:
            for n, r in self._residents.items():
                if name is None or n == name:
                    r["used"] = self.clock()

    def residents(self) -> list[str]:
        with self._cv:
            return sorted(self._residents)

    def last_used(self) -> float | None:
        with self._cv:
            return max((r["used"] for r in self._residents.values()), default=None)

    def unload_idle(self, idle_s: float) -> list[str]:
        now = self.clock()
        with self._cv:
            gone = [(n, r["unload"]) for n, r in self._residents.items() if now - r["used"] >= idle_s]
            for n, _ in gone:
                self._residents.pop(n, None)
        for _, unload in gone:
            try:
                unload()
            except Exception:  # noqa: BLE001 — unloading is best-effort
                pass
        if gone:
            self._release_memory()
        return sorted(n for n, _ in gone)

    def _release_memory(self) -> None:
        free = self._free
        if free is None:
            from .embedding import free_gpu as free
        try:
            free()
        except Exception:  # noqa: BLE001
            pass


def gpu_owned(kind: str, label: str | None = None):
    """Decorate a Service job method that loads a model: it runs holding the
    GPU lease (`self.gpu`), waiting its turn — a stop while it waits returns
    {"stopped": True, "waited_for_gpu": True}. Without a lease (a bare test
    double), it just runs."""
    import functools
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(self, *a, **k):
            lease = getattr(self, "gpu", None)
            if not isinstance(lease, GpuLease):
                return fn(self, *a, **k)
            try:
                job = sig.bind_partial(self, *a, **k).arguments.get("job")
            except TypeError:
                job = None
            out = lease.run_owned(kind, job, lambda: fn(self, *a, **k), label=label)
            return out if out is not None or not getattr(job, "cancelled", False) else \
                {"stopped": True, "waited_for_gpu": True}
        return wrapper
    return deco


class _Guarded:
    """A model whose every method call runs as one GPU batch (or borrow);
    plain attributes pass straight through."""

    __slots__ = ("_m", "_lease", "_borrow")

    def __init__(self, model, lease: GpuLease, borrow: bool):
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_borrow", borrow)

    def __getattr__(self, name):
        val = getattr(self._m, name)
        if not callable(val) or name.startswith("__") or isinstance(val, type):
            return val
        lease, borrow = self._lease, self._borrow

        def call(*a, **k):
            with (lease.borrow(name) if borrow else lease.batch()):
                return val(*a, **k)
        return call

    def __setattr__(self, name, value):
        setattr(self._m, name, value)

    @property
    def unguarded(self):
        return self._m
