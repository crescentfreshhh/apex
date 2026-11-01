"""Checking what a request sends: on/off values that mean what they say, and
numbers kept inside their range. A bad value is a 400 with a reason, never a
silently-wrong setting (the text "false" is off, not on)."""

from __future__ import annotations

_TRUE = {"1", "true", "on", "yes", "y"}
_FALSE = {"0", "false", "off", "no", "n", ""}


def as_bool(v, name: str = "value") -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    raise ValueError(f"{name} must be on or off")


def as_int(v, name: str = "value", lo: int | None = None, hi: int | None = None) -> int:
    try:
        n = int(float(v)) if not isinstance(v, bool) else None
    except (TypeError, ValueError):
        n = None
    if n is None:
        raise ValueError(f"{name} must be a whole number")
    if lo is not None and n < lo or hi is not None and n > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return n
