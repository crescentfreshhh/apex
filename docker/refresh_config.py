#!/usr/bin/env python3
"""Keep /config/config.toml in step with the image's bundled defaults.

config.toml is seeded once and then lives in the persistent /config volume, so
new defaults (and fixes to old ones — e.g. a corrected max_duration) never reach
an existing install, and a stale value silently overrides the shipped default.

On every start this compares the bundled config.example.toml against a stored
fingerprint. When the defaults have changed it backs up the current file and
writes the fresh one — but carries the user's [stash] connection block across,
since that's the only part that holds real settings (and it's usually supplied
via env vars anyway). Everything else is tuning that the GUI now overrides
per-run, so refreshing it is safe.

Pure-Python and side-effect-free except for the given paths, so the core is
unit-tested; the container entrypoint calls main().
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
import tomllib
from pathlib import Path

EXAMPLE = Path("/opt/peaks/config.example.toml")
CONFIG = Path("/config/config.toml")
VERSION = Path("/config/.config_version")


def _carry_stash(example_text: str, old_stash: dict) -> str:
    """Return the example text with [stash] url/api_key/timeout replaced by the
    user's previous values (so a refresh never drops their connection)."""
    text = example_text
    for key in ("url", "api_key"):
        val = old_stash.get(key)
        if isinstance(val, str):
            text = re.sub(
                rf'(?m)^{key}\s*=\s*".*"', lambda m, v=val: f'{key} = "{v}"',
                text, count=1,
            )
    if isinstance(old_stash.get("timeout"), int):
        text = re.sub(
            r"(?m)^timeout\s*=\s*\d+",
            lambda m: f'timeout = {old_stash["timeout"]}', text, count=1,
        )
    return text


def refresh(
    example: Path, config: Path, version: Path, *, now: str | None = None, log=print
) -> str:
    """Reconcile `config` with `example`. Returns the action taken:
    'seeded' | 'current' | 'refreshed'."""
    if not example.exists():
        return "current"
    example_text = example.read_text()
    new_hash = hashlib.md5(example_text.encode()).hexdigest()

    if not config.exists():
        config.write_text(example_text)
        version.write_text(new_hash)
        log("config: seeded config.toml from defaults")
        return "seeded"

    if version.exists() and version.read_text().strip() == new_hash:
        return "current"

    # defaults changed (or this is the first versioned start): back up + refresh
    try:
        old = tomllib.loads(config.read_text())
    except Exception:
        old = {}
    stash = old.get("stash", {}) if isinstance(old, dict) else {}
    merged = _carry_stash(example_text, stash if isinstance(stash, dict) else {})

    ts = now or time.strftime("%Y%m%d-%H%M%S")
    backup = config.with_name(f"config.toml.bak.{ts}")
    shutil.copy2(config, backup)
    config.write_text(merged)
    version.write_text(new_hash)
    log(
        f"config: refreshed config.toml to new defaults "
        f"(kept your [stash]; previous file saved as {backup.name})"
    )
    return "refreshed"


def main() -> None:
    try:
        refresh(EXAMPLE, CONFIG, VERSION)
    except Exception as exc:  # never let a config hiccup stop the container
        print(f"config: refresh skipped ({exc})")


if __name__ == "__main__":
    main()
