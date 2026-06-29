"""Command-line entry point.

Usage:
    peaks test            # verify connection + print Stash version
    peaks scenes          # list scenes (id, duration, marker count, path)
    peaks stats           # library summary: scene count, total duration, markers
    peaks embed           # sample + embed the library into the cache (GPU pass)
    peaks score           # cache -> apex segments; --write to push markers

Run `python -m peaks <cmd>` if you haven't installed the console script.
"""

from __future__ import annotations

import argparse
import itertools
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


def _build_embedder(cfg, **kwargs):
    """Instantiate the configured embedder, with a friendly hint if torch/the
    ML extra isn't installed."""
    from .embedding import get_embedder

    try:
        return get_embedder(cfg.embedding.model, **kwargs)
    except ImportError as exc:
        print(
            f"✗ The '{cfg.embedding.model}' embedder needs the ML dependencies.\n"
            f"  {exc}\n"
            '  Install them with:  pip install -e ".[ml]"\n'
            "  (or set embedding.model = \"fake\" in config.toml to test plumbing)",
            file=sys.stderr,
        )
        raise SystemExit(2)


def cmd_embed(args) -> int:
    cfg = Config.load(args.config)
    client = StashClient.from_config(cfg)
    # heavy imports kept local so `peaks test/scenes/stats` stay torch-free
    from .cache import EmbeddingCache
    from .pipeline import embed_library
    from .sampling import FrameSampler

    sampler = FrameSampler(interval_seconds=cfg.sampling.interval_seconds)
    embedder = _build_embedder(
        cfg, **({"device": cfg.embedding.device} if cfg.embedding.device else {})
    )
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    scenes = client.iter_scenes()
    if args.limit:
        scenes = itertools.islice(scenes, args.limit)
    print(f"Embedding with '{embedder.name}' (dim={embedder.dim}) -> {cfg.embedding.cache_dir}")
    stats = embed_library(
        scenes, sampler, embedder, cache, batch_size=cfg.embedding.batch_size
    )
    print(
        f"\nDone. embedded={stats['embedded']} skipped(cached)={stats['skipped']} "
        f"failed={stats['failed']} frames={stats['frames']}"
    )
    return 0


def cmd_score(args) -> int:
    cfg = Config.load(args.config)
    client = StashClient.from_config(cfg)
    from .cache import EmbeddingCache
    from .pipeline import load_references, score_library

    embedder = _build_embedder(cfg)
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    refs_dir = args.references or cfg.scoring.references_dir
    try:
        references = load_references(embedder, refs_dir)
    except FileNotFoundError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print(f"  Put example stills you love in: {refs_dir}/", file=sys.stderr)
        return 1
    print(f"Loaded {references.shape[0]} reference image(s) from {refs_dir}/")

    scenes = client.iter_scenes()
    if args.limit:
        scenes = itertools.islice(scenes, args.limit)
    tag = args.tag or cfg.markers.tag_name
    mode = "WRITING markers" if args.write else "dry run (no writes)"
    print(f"Scoring tag '{tag}' — {mode}\n")
    stats = score_library(
        scenes,
        cache,
        embedder.name,
        references,
        cfg.scoring,
        client=client,
        tag_name=tag,
        write=args.write,
    )
    verb = "created" if args.write else "found"
    print(
        f"\nDone. scenes_scored={stats['scenes']} segments_{verb}={stats['segments']} "
        f"skipped(no cache)={stats['skipped']}"
    )
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

    ep = sub.add_parser("embed", help="Sample + embed the library into the cache")
    ep.add_argument("--limit", type=int, default=0, help="Max scenes (0 = all)")
    ep.set_defaults(func=cmd_embed)

    scp = sub.add_parser("score", help="Score cached scenes into apex segments")
    scp.add_argument(
        "--write",
        action="store_true",
        help="Write markers to Stash (default: dry-run preview)",
    )
    scp.add_argument("--references", help="Dir of reference stills (overrides config)")
    scp.add_argument("--tag", help="Marker tag name (overrides config)")
    scp.add_argument("--limit", type=int, default=0, help="Max scenes (0 = all)")
    scp.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
