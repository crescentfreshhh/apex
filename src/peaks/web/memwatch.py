"""Best-effort RAM self-policing for the web process.

The dominant resident cost is the in-memory `SearchIndex.matrix` (N frames x dim
float32) — and there's one per embedding model, so a DINO + a CLIP index on a
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


def rss_bytes() -> int:
    """Resident set size of this process, in bytes (0 if unreadable)."""
    try:
        with open("/proc/self/statm") as f:
            # fields are in pages: size, resident, shared, ...
            return int(f.read().split()[1]) * _PAGE
    except Exception:  # noqa: BLE001 — best-effort
        return 0


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
