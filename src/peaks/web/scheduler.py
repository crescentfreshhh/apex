"""The background scheduler: asleep until something happens.

It used to wake every 60 s and ask every job "are you due?" — which, on a quiet
server, meant a stat of the media share, of five state files and more, every
minute, for nothing. Now each step answers two questions:

* `fn()` — today's step, unchanged in what it decides (is it due → start it);
* `deadline(now)` — when the *clock alone* could make it due next (the weekly
  backup's window, the overnight measure's hour, the seedbox times, the pause
  ending, a retry back-off…), or None if only a change can.

It wakes when an event says something changed — a job finished, a library write,
the pause, the index or the scene list changed — and then runs every step once
(each step's idle path reads memory only), or when a step's deadline arrives,
and then runs just that step. Nothing else wakes it.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass
class Step:
    name: str
    fn: Callable[[], object]
    deadline: Callable[[float], float | None] = lambda now: None
    on_event: bool = True          # False: runs only on its own deadline (the library check)
    stops_rest: bool = False       # True: a truthy result means "nothing else now" (the pause)
    every: float | None = None     # a repeat: due this long after it last ran (anchored, never pushed back)
    active: Callable[[], bool] = lambda: True    # a repeat that's switched off has no deadline


class Scheduler:
    MIN_GAP = 60.0                 # a step is never re-run by the clock sooner than this
    MAX_SLEEP = 1800.0             # a heartbeat, in case a deadline was missed
    SETTLE = 2.0                   # events arriving together are handled in one run
    EVENT_EVERY = 5.0              # at most one event run per this many seconds

    def __init__(self, steps: list[Step], clock: Callable[[], float] = time.time, first_delay: float = 30.0):
        self.steps = steps
        self.clock = clock
        self.first_delay = first_delay
        self.next: dict[str, float | None] = {}
        self.ran: dict[str, float] = {}
        self.reasons: Counter = Counter()
        self.runs = 0
        self._ev = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_event_run = -1e18
        self._born = clock()

    # --- events -----------------------------------------------------------------------

    def wake(self, reason: str = "") -> None:
        self.reasons[reason or "?"] += 1
        self._ev.set()

    def stop(self) -> None:
        self._stop.set()
        self._ev.set()

    # --- one pass -----------------------------------------------------------------------

    def run_due(self, event: bool = False) -> list[str]:
        """Run the steps an event or the clock calls for; returns their names."""
        with self._lock:
            now = self.clock()
            ran = []
            for st in self.steps:
                nx = self.next.get(st.name)
                by_clock = nx is not None and now >= nx
                if not ((event and st.on_event) or by_clock):
                    continue
                ran.append(st.name)
                self.ran[st.name] = now
                try:
                    out = st.fn()
                except Exception:  # noqa: BLE001 — never let the scheduler die
                    out = None
                if st.stops_rest and out:
                    break
            if event:
                self._last_event_run = now
            self.runs += 1
            self._plan(now)
            return ran

    def _plan(self, now: float) -> None:
        for st in self.steps:
            try:
                if st.every is not None:
                    d = self.ran.get(st.name, self._born) + st.every if st.active() else None
                else:
                    d = st.deadline(now)
            except Exception:  # noqa: BLE001
                d = None
            if d is not None:
                d = max(d, self.ran.get(st.name, -1e18) + self.MIN_GAP)
            self.next[st.name] = d

    def sleep_for(self) -> float:
        now = self.clock()
        soon = [d for d in self.next.values() if d is not None]
        return max(1.0, min([self.MAX_SLEEP] + [d - now for d in soon]))

    # --- the thread ---------------------------------------------------------------------

    def start(self) -> "Scheduler":
        threading.Thread(target=self._loop, daemon=True, name="peaks-scheduler").start()
        return self

    def _loop(self) -> None:
        if self._stop.wait(self.first_delay):    # start-up isn't slammed
            return
        self.run_due(event=True)
        while not self._stop.is_set():
            woke = self._ev.wait(self.sleep_for())
            if self._stop.is_set():
                return
            if woke:
                gap = self._last_event_run + self.EVENT_EVERY - self.clock()
                self._stop.wait(max(self.SETTLE, gap))      # let a burst (a run of grades) land first
                self._ev.clear()
            self.run_due(event=woke)
