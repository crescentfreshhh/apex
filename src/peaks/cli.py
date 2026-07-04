"""Command-line entry point.

Usage:
    peaks test            # verify connection + print Stash version
    peaks scenes          # list scenes (id, duration, marker count, path)
    peaks stats           # library summary: scene count, total duration, markers
    peaks embed           # sample + embed the library into the cache (GPU pass)
    peaks score           # cache -> apex segments; --write to push markers
    peaks label           # rate candidate frames (Tier 2 labeling)
    peaks train           # train a taste classifier from your labels
    peaks clear           # delete generated markers for a tag (dry-run first)
    peaks playlist        # export apex markers -> webapp/playlist.json
    peaks serve           # serve the megaboard webapp

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

    sampler = FrameSampler(
        interval_seconds=cfg.sampling.interval_seconds,
        mode=cfg.sampling.mode,
        hwaccel=cfg.sampling.hwaccel,
    )
    embedder = _build_embedder(
        cfg, **({"device": cfg.embedding.device} if cfg.embedding.device else {})
    )
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    total = args.limit or client.scene_count()
    scenes = client.iter_scenes()
    if args.limit:
        scenes = itertools.islice(scenes, args.limit)
    extras = f", mode={cfg.sampling.mode}" if cfg.sampling.mode != "interval" else ""
    extras += f", hwaccel={cfg.sampling.hwaccel}" if cfg.sampling.hwaccel else ""
    print(
        f"Embedding {total} scene(s) with '{embedder.name}' "
        f"(dim={embedder.dim}{extras}) -> {cfg.embedding.cache_dir}"
    )
    stats = embed_library(
        scenes, sampler, embedder, cache,
        batch_size=cfg.embedding.batch_size, total=total,
    )
    rate = (
        f" avg {stats['seconds_per_scene']}s/scene"
        if "seconds_per_scene" in stats
        else ""
    )
    print(
        f"\nDone. embedded={stats['embedded']} skipped(cached)={stats['skipped']} "
        f"failed={stats['failed']} frames={stats['frames']}{rate}"
    )
    return 0


def _safe_tag(tag: str) -> str:
    """Filesystem-safe model filename stem for a tag (apex:heels -> apex_heels)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)


def _build_scorer(cfg, args, tag):
    """Return (score_frames, model_name, label) choosing Tier-2 model when a
    trained one exists (unless --references forces Tier-1 similarity)."""
    from pathlib import Path

    from .embedding import canonical_name

    model_path = (
        Path(args.model)
        if getattr(args, "model", None)
        else Path(cfg.modeling.dir) / f"{_safe_tag(tag)}.pkl"
    )
    if model_path.exists() and not args.references:
        from .classifier import TasteClassifier

        clf = TasteClassifier.load(model_path)
        model_name = clf.model_name or canonical_name(cfg.embedding.model)
        return clf.predict_proba, model_name, f"Tier-2 model {model_path}"

    # Tier-1 similarity: embed reference stills
    from .pipeline import load_references
    from .scoring import make_similarity_scorer

    embedder = _build_embedder(cfg)
    refs_dir = args.references or cfg.scoring.references_dir
    references = load_references(embedder, refs_dir)  # may raise FileNotFoundError
    label = f"Tier-1 similarity ({references.shape[0]} refs from {refs_dir}/)"
    return make_similarity_scorer(references, cfg.scoring.reduce), embedder.name, label


def cmd_score(args) -> int:
    cfg = Config.load(args.config)
    client = StashClient.from_config(cfg)
    from .cache import EmbeddingCache
    from .pipeline import score_library

    tag = args.tag or cfg.markers.tag_name
    try:
        score_frames, model_name, label = _build_scorer(cfg, args, tag)
    except FileNotFoundError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print(
            f"  Either put reference stills in {cfg.scoring.references_dir}/ (Tier 1),\n"
            f"  or train a model with `peaks train --tag {tag}` (Tier 2).",
            file=sys.stderr,
        )
        return 1

    cache = EmbeddingCache(cfg.embedding.cache_dir)
    scenes = client.iter_scenes()
    if args.limit:
        scenes = itertools.islice(scenes, args.limit)
    mode = "WRITING markers" if args.write else "dry run (no writes)"
    print(f"Scoring tag '{tag}' via {label} — {mode}\n")
    stats = score_library(
        scenes,
        cache,
        model_name,
        score_frames,
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


def cmd_train(args) -> int:
    cfg = Config.load(args.config)
    from pathlib import Path

    from .cache import EmbeddingCache
    from .embedding import canonical_name
    from .labels import LabelStore
    from .pipeline import train_profile

    profile = args.tag or cfg.markers.tag_name
    store = LabelStore(cfg.modeling.labels_path)
    pos, neg = store.counts(profile)
    print(f"Labels for '{profile}': {pos} positive / {neg} negative")
    if pos == 0 or neg == 0:
        print("✗ Need at least one positive AND one negative label. Run `peaks label`.")
        return 1

    cache = EmbeddingCache(cfg.embedding.cache_dir)
    model_name = canonical_name(cfg.embedding.model)
    try:
        clf, stats = train_profile(
            store, cache, model_name, profile, kind=cfg.modeling.classifier
        )
    except ImportError as exc:
        print(f"✗ Training needs scikit-learn: {exc}", file=sys.stderr)
        print('  Install with:  pip install -e ".[ml]"', file=sys.stderr)
        return 2
    out = Path(cfg.modeling.dir) / f"{_safe_tag(profile)}.pkl"
    clf.save(out)
    print(
        f"Trained {cfg.modeling.classifier} on {stats['samples']} frames "
        f"({stats['positives']} positive) -> {out}"
    )
    if "cv_auc" in stats:
        auc = stats["cv_auc"]
        verdict = (
            "excellent separation" if auc >= 0.9
            else "usable — more labels will sharpen it" if auc >= 0.75
            else "weak — label more (especially diverse negatives) before trusting it"
        )
        print(f"Cross-validated AUC: {auc} over {stats['cv_folds']} folds ({verdict})")
    else:
        print("(too few labels per class for cross-validation — label more for a quality read)")
    return 0


def cmd_label(args) -> int:
    cfg = Config.load(args.config)
    from .cache import EmbeddingCache
    from .embedding import canonical_name
    from .labeler import launch_labeler
    from .labels import LabelStore
    from .pipeline import gather_candidates
    from .sampling import FrameSampler

    profile = args.tag or cfg.markers.tag_name
    cache = EmbeddingCache(cfg.embedding.cache_dir)
    model_name = canonical_name(cfg.embedding.model)
    try:
        score_frames, _, label = _build_scorer(cfg, args, profile)
    except FileNotFoundError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print(
            f"  Add reference stills to {cfg.scoring.references_dir}/ to seed candidates.",
            file=sys.stderr,
        )
        return 1
    store = LabelStore(cfg.modeling.labels_path)
    print(f"Seeding candidates for '{profile}' via {label}")
    cands = gather_candidates(
        cache,
        model_name,
        score_frames,
        limit=args.limit or None,
        exclude=store.labeled_ids(profile),  # don't re-show rated frames
    )
    if not cands:
        print("✗ No candidates — is the cache populated? Run `peaks embed` first.")
        return 1
    print(f"Launching labeler on {len(cands)} candidates (port {args.port}) ...")
    sampler = FrameSampler(
        interval_seconds=cfg.sampling.interval_seconds,
        hwaccel=cfg.sampling.hwaccel,
    )
    launch_labeler(cands, store, profile, sampler.grab_frame, server_port=args.port)
    return 0


def cmd_clear(args) -> int:
    """Delete generated markers for a tag. Dry-run by default, like `score`."""
    cfg = Config.load(args.config)
    client = StashClient.from_config(cfg)
    tag = args.tag or cfg.markers.tag_name
    # strictly ours: only markers whose PRIMARY tag is the target
    ids = [
        m["marker_id"]
        for m in client.iter_markers_by_tag(tag)
        if m["primary_tag"] == tag
    ]
    if not ids:
        print(f"No markers with primary tag '{tag}'. Nothing to do.")
        return 0
    if not args.write:
        print(f"Would delete {len(ids)} marker(s) with primary tag '{tag}'.")
        print("Re-run with --write to actually delete them.")
        return 0
    n = client.destroy_scene_markers(ids)
    print(f"Deleted {n} marker(s) with primary tag '{tag}'.")
    return 0


def cmd_playlist(args) -> int:
    cfg = Config.load(args.config)
    client = StashClient.from_config(cfg)
    from .playlist import build_playlist, write_playlist

    tag = args.tag or cfg.markers.tag_name
    pl = build_playlist(client, tag, limit=args.limit or None)
    out = args.out or "webapp/playlist.json"
    write_playlist(pl, out)
    print(f"Wrote {pl['count']} apex(es) for tag '{tag}' -> {out}")
    if pl["count"] == 0:
        print("  (no markers with that tag yet — run `peaks score --write` first)")
    return 0


def cmd_serve(args) -> int:
    import functools
    import http.server
    import socketserver

    directory = args.directory
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=directory
    )
    # default to loopback: playlist.json carries the Stash API key in its URLs,
    # so don't expose it to the whole LAN unless explicitly asked (--host)
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {directory}/ at http://{args.host}:{args.port}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
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
    scp.add_argument(
        "--references",
        help="Dir of reference stills — forces Tier-1 similarity (overrides config)",
    )
    scp.add_argument("--model", help="Path to a trained model (.pkl) for Tier-2")
    scp.add_argument("--tag", help="Marker tag name (overrides config)")
    scp.add_argument("--limit", type=int, default=0, help="Max scenes (0 = all)")
    scp.set_defaults(func=cmd_score)

    tp = sub.add_parser("train", help="Train a Tier-2 taste classifier from labels")
    tp.add_argument("--tag", help="Taste profile / tag to train (overrides config)")
    tp.set_defaults(func=cmd_train)

    lp = sub.add_parser("label", help="Launch the rapid frame-labeler (Tier 2)")
    lp.add_argument("--tag", help="Taste profile / tag to label (overrides config)")
    lp.add_argument(
        "--references", help="Seed candidates via these reference stills (Tier 1)"
    )
    lp.add_argument("--model", help="Seed candidates via this trained model")
    lp.add_argument("--limit", type=int, default=200, help="Max candidates (default 200)")
    lp.add_argument("--port", type=int, default=7860, help="Gradio port (default 7860)")
    lp.set_defaults(func=cmd_label)

    cp = sub.add_parser("clear", help="Delete generated markers for a tag")
    cp.add_argument("--tag", help="Tag whose markers to delete (overrides config)")
    cp.add_argument(
        "--write", action="store_true", help="Actually delete (default: dry-run count)"
    )
    cp.set_defaults(func=cmd_clear)

    pp = sub.add_parser("playlist", help="Export marker apexes to webapp/playlist.json")
    pp.add_argument("--tag", help="Marker tag to export (overrides config)")
    pp.add_argument("--out", help="Output path (default: webapp/playlist.json)")
    pp.add_argument("--limit", type=int, default=0, help="Max apexes (0 = all)")
    pp.set_defaults(func=cmd_playlist)

    svp = sub.add_parser("serve", help="Serve the megaboard webapp locally")
    svp.add_argument("--port", type=int, default=8800, help="Port (default: 8800)")
    svp.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1 — playlist.json contains your "
        "API key; use 0.0.0.0 only if you accept LAN exposure)",
    )
    svp.add_argument("--directory", default="webapp", help="Dir to serve (default: webapp)")
    svp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
