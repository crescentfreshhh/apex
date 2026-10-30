"""Best-effort RAM self-policing for the web process.

The search index (`SearchIndex.matrix`, N frames x dim) is mapped from its file
on the cache SSD, so it is page cache, not process memory; what's policed here
is real (anonymous) memory. It used to be the dominant resident cost — one per
embedding model, so a DINO + a CLIP index on a
large library can be many GB apiece. On a memory-capped host (e.g. a 24 GB
Unraid container) that runs the box out of RAM and the process gets OOM-killed.

This module gives the Service a watchdog: sample RSS against a soft limit
(auto-detected from the container's cgroup, or `PEAKS_MEM_LIMIT_MB`), and when
it's exceeded, shed derived caches and idle model indexes and hand the freed
pages back to the OS. Pure /proc + libc — no third-party deps.

Env knobs:
  PEAKS_MEM_LIMIT_MB   hard soft-limit in MB (0/unset -> auto from cgroup)
  PEAKS_MEM_SOFT_PCT   % of the detected cgroup limit to target (default 85)
  PEAKS_MEM_CHECK_SEC  seconds between checks (default 20; 0 disables the watch)
"""

from __future__ import annotations

import ctypes
import os

try:
    _PAGE = os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    _PAGE = 4096


def _status_kb(field: str) -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except Exception:  # noqa: BLE001 — best-effort
        pass
    return None


def rss_bytes() -> int:
    """This process's *real* memory, in bytes: its anonymous resident pages
    (heap, models, arrays) — not pages it has mapped from files, like the search
    index on the cache SSD, which the kernel drops whenever it needs the RAM.
    Falls back to the whole resident set where /proc doesn't say (0 if unreadable)."""
    kb = _status_kb("RssAnon")
    if kb is not None:
        return kb * 1024
    try:
        with open("/proc/self/statm") as f:
            # fields are in pages: size, resident, shared, ...
            return int(f.read().split()[1]) * _PAGE
    except Exception:  # noqa: BLE001 — best-effort
        return 0


def file_rss_bytes() -> int:
    """Resident pages mapped from files (the on-disk index, libraries): cache,
    not a requirement — reclaimed by the kernel under pressure."""
    kb = _status_kb("RssFile")
    return (kb or 0) * 1024


def cgroup_limit_bytes() -> int | None:
    """The container's memory limit from cgroup v2 then v1, or None if
    unlimited / not in a limited cgroup."""
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except Exception:  # noqa: BLE001
            continue
        if raw == "max":
            return None
        try:
            n = int(raw)
        except ValueError:
            continue
        # v1 reports a giant sentinel when unlimited
        if n <= 0 or n >= (1 << 62):
            return None
        return n
    return None


def soft_limit_bytes() -> int | None:
    """The RSS ceiling the watchdog targets: `PEAKS_MEM_LIMIT_MB` if set, else a
    percentage (`PEAKS_MEM_SOFT_PCT`, default 85) of the cgroup limit. None means
    'no detectable limit' -> the watchdog stays off."""
    env = os.environ.get("PEAKS_MEM_LIMIT_MB")
    if env:
        try:
            mb = int(float(env))
            if mb > 0:
                return mb * 1024 * 1024
        except ValueError:
            pass
    lim = cgroup_limit_bytes()
    if lim is None:
        return None
    try:
        pct = float(os.environ.get("PEAKS_MEM_SOFT_PCT", "85"))
    except ValueError:
        pct = 85.0
    pct = min(max(pct, 10.0), 99.0)
    return int(lim * pct / 100.0)


def check_seconds() -> float:
    try:
        return float(os.environ.get("PEAKS_MEM_CHECK_SEC", "20"))
    except ValueError:
        return 20.0


def malloc_trim() -> None:
    """Return free glibc arena memory to the OS (drops RSS). No-op off glibc."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 — musl / non-Linux / no libc
        pass


# --- the watchdog's policy: tiered, and never a purge loop -------------------
#
# A purge empties caches pages read; with jobs holding memory near the line,
# purging every check (20 s) meant every page rebuilt them from Stash and disk.
# So: a soft purge keeps the page path warm (metadata, board/taste scores) and
# drops only what's big and off it; it waits a cooldown between runs, and a
# purge that freed little isn't repeated until memory has grown again. The hard
# limit still sheds everything — the OOM guard is unchanged.

import time as _time
from collections import deque

MB = 1024 * 1024
PURGES: deque = deque(maxlen=50)          # recent purges, newest last (for the Speed panel)


class PurgePolicy:
    def __init__(self, limit: int, soft_frac: float = 0.9, cooldown: float = 300.0,
                 hard_gap: float = 60.0, small: int = 100 * MB, regrow: int = 500 * MB,
                 clock=_time.monotonic):
        self.limit, self.soft = limit, int(limit * soft_frac)
        self.cooldown, self.hard_gap, self.small, self.regrow = cooldown, hard_gap, small, regrow
        self.clock = clock
        self.last_t: float | None = None
        self.last_after = 0
        self.last_freed = None

    def decide(self, rss: int) -> str | None:
        """'hard', 'soft' or None for this check."""
        if rss < self.soft:
            return None
        since = None if self.last_t is None else self.clock() - self.last_t
        # the last purge freed little: purging again frees little too, until memory grows
        futile = self.last_freed is not None and self.last_freed < self.small and rss < self.last_after + self.regrow
        if rss >= self.limit:
            if since is not None and (since < self.hard_gap or futile):
                return None
            return "hard"
        if since is not None and (since < self.cooldown or futile):
            return None
        return "soft"

    def record(self, level: str, before: int, after: int, dropped=None) -> dict:
        self.last_t, self.last_after = self.clock(), after
        self.last_freed = max(0, before - after)
        entry = {"t": _time.time(), "level": level, "rss_before_mb": round(before / MB),
                 "rss_after_mb": round(after / MB), "freed_mb": round(self.last_freed / MB),
                 "dropped": list(dropped or [])}
        PURGES.append(entry)
        return entry
