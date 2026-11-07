"""How long Stash takes to answer — every GraphQL call is timed, so a slow
moment can be read off the Speed panel ("Stash answered in 2.4 s") instead of
guessed. Two views: the last few minutes of calls (median / slowest), and,
per web request, how much of its time was spent waiting on Stash."""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections import deque

WINDOW_S = 300.0                  # "the last 5 minutes"
_calls: deque = deque(maxlen=2000)            # (when, seconds)
_lock = threading.Lock()
# a per-request [seconds, calls] — a context variable, so the worker thread a request's
# handler runs on adds to the same list the request's middleware started
_acc: contextvars.ContextVar = contextvars.ContextVar("peaks_stash_acc", default=None)


@contextlib.contextmanager
def timed():
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        with _lock:
            _calls.append((time.time(), dt))
        acc = _acc.get()
        if acc is not None:
            with _lock:
                acc[0] += dt
                acc[1] += 1


def request_start():
    """Start counting Stash time for this request; returns the token for request_take."""
    acc = [0.0, 0]
    return _acc.set(acc), acc


def request_take(started) -> tuple[float, int]:
    """(seconds waiting on Stash, calls) since request_start — and stop counting."""
    token, acc = started
    try:
        _acc.reset(token)
    except ValueError:
        pass
    return acc[0], acc[1]


def summary(now: float | None = None) -> dict:
    """{calls, median_ms, p90_ms, slowest_ms} over the last WINDOW_S."""
    now = now or time.time()
    with _lock:
        xs = sorted(dt for t, dt in _calls if now - t <= WINDOW_S)
    if not xs:
        return {"calls": 0, "median_ms": None, "p90_ms": None, "slowest_ms": None}
    pick = lambda q: round(1000 * xs[min(len(xs) - 1, int(q * len(xs)))])  # noqa: E731
    return {"calls": len(xs), "median_ms": pick(0.5), "p90_ms": pick(0.9), "slowest_ms": round(1000 * xs[-1])}


def reset() -> None:
    with _lock:
        _calls.clear()
