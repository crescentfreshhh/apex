"""Command-line entry point.

Usage:
    peaks test            # verify connection + print Stash version
    peaks scenes          # list scenes (id, duration, marker count, path)
    peaks stats           # library summary: scene count, total duration, markers

Run `python -m peaks <cmd>` if you haven't installed the console script.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .stash_client import StashClient, StashError


def _client(args) -> StashClient:
    cfg = Config.load(args.config)
    return StashClient.from_config(cfg)


def cmd_test(args) -> int:
    client = _client(args)
    try:
        v = client.version()
    except StashError as exc:
        print(f"✗ Connection failed.\n  {exc}", file=sys.stderr)
        print(
            "\nHints:\n"
            "  - Is the server URL correct in config.toml (or $STASH_URL)?\n"
            "  - If auth is on, set api_key in config.toml (or $STASH_API_KEY).\n"
            "  - Is the Stash server actually running and reachable from here?",
            file=sys.stderr,
        )
        return 1
    print(f"✓ Connected to Stash {v.get('version')} (build {v.get('build_time')})")
    try:
        print(f"  Library: {client.scene_count()} scenes")
    except StashError as exc:
        print(f"  (could not count scenes: {exc})")
    return 0


def cmd_scenes(args) -> int:
    client = _client(args)
    shown = 0
    for scene in client.iter_scenes():
        dur = scene.duration
        dur_s = f"{dur/60:6.1f}m" if dur else "    ?  "
        title = (scene.title or scene.path or "<no title>")[:60]
        print(f"[{scene.id:>6}] {dur_s}  markers:{len(scene.markers):<3}  {title}")
        shown += 1
        if args.limit and shown >= args.limit:
            break
    print(f"\n{shown} scene(s) shown.")
    return 0


def cmd_stats(args) -> int:
    client = _client(args)
    n = 0
    total_dur = 0.0
    total_markers = 0
    no_file = 0
    for scene in client.iter_scenes():
        n += 1
        if scene.duration:
            total_dur += scene.duration
        else:
            no_file += 1
        total_markers += len(scene.markers)
    print(f"Scenes:          {n}")
    print(f"Total duration:  {total_dur/3600:.1f} hours")
    print(f"Existing markers:{total_markers}")
    print(f"Scenes w/o file: {no_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="peaks", description=__doc__)
    p.add_argument(
        "-c", "--config", default=None, help="Path to config.toml (default: ./config.toml)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Verify connection to Stash").set_defaults(func=cmd_test)

    sp = sub.add_parser("scenes", help="List scenes")
    sp.add_argument("--limit", type=int, default=0, help="Max scenes to show (0 = all)")
    sp.set_defaults(func=cmd_scenes)

    sub.add_parser("stats", help="Library summary").set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
