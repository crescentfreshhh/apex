# Architecture & gameplan

This captures the design decisions so we can pick the project back up cold.

## The problem

A large, highly-curated Stash library where most scenes contain only a few
moments worth watching (specific positions, angles, framing). We want to:

1. Learn the user's visual taste from examples.
2. Locate the matching timestamp-segments ("peaks") across the whole library.
3. Surface them as Stash scene markers.
4. Play them back — a queue, and a "megaboard" grid of simultaneously looping
   clips that continuously cycles in new peaks.

## Why this shape

- **Stash is the source of truth + output surface, not the ML engine.** It
  exposes a GraphQL API for reads, and its **scene markers** (timestamped,
  tagged points) are the natural home for our output. No need to invent storage.
- **Heavy ML is a batch pipeline**, not a Stash plugin. A Stash plugin will only
  ever be a thin trigger ("Find my highlights" button) and/or a UI launcher.
- **Embed once, learn cheaply.** We never classify video directly. We sample
  frames, embed them into vectors a single time (GPU-heavy, cached to disk),
  then learn taste in vector space — fast and re-trainable as taste drifts.

## Key decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Output storage | **Stash scene markers** | Reuses Stash UI; native timestamp+tag primitive |
| Stash version | **Latest dev build** | Newest manifest + current GraphQL schema; no legacy quirks |
| Megaboard playback | **Live-stream** (`?start=` seeks) initially | Avoid disk usage; modest grid (e.g. 3×3), direct stream where codec allows |
| Language | **Python** for the brain | Where the ML lives |
| Accelerator | **Borrow an RTX 3080 Ti for a burst** | GPU only needed for the one-time embedding pass |

## Key decisions (open / deferred)

- **Embedding model**: leaning **DINOv2** over CLIP. CLIP optimizes image↔text
  matching (semantic), weaker on geometric/compositional similarity. DINOv2 is
  self-supervised and captures visual *structure* (poses, angles, framing) —
  better fit. May ensemble with CLIP, and optionally add pose keypoints later.
- **Tier 1 vs Tier 2**: build Tier 1 first (validate), then Tier 2.
- **Final term/taxonomy**: "peak" is the working unit name; the marker tag is
  the user's "taste profile" label (configurable; can have multiple).

## The two tiers of taste learning

**Tier 1 — similarity, no training.** Pick ~20 reference frames you love → embed
→ score every sampled frame by similarity to the references → contiguous high
runs become candidate segments. Instant, blunt; used to validate the pipeline.

**Tier 2 — trained classifier.** Rate a few hundred frames yes/no in a small
Gradio tool (seed it with Tier-1 candidates = active learning, fewer labels
needed), train a logistic-regression/MLP on the cached embeddings. Learns *your*
taste; CPU-cheap to retrain.

**Segment post-processing** (both tiers): frame scores → moving-average smooth →
hysteresis threshold → merge into segments with min/max length.

## Compute notes

- One-time embedding pass is the only GPU-heavy job. ~1000 × 30-min videos at
  1 frame / 2s ≈ ~1M frames → a few hours on the 3080 Ti.
- **Cache every embedding to disk** (e.g. one `.npy` per scene keyed by file
  hash). Make the pass **resumable** (checkpoint per scene) so it can run in
  chunks when the GPU is free, and so new videos only embed the delta.
- Training + scoring are CPU-cheap and run anytime without the GPU.
- CPU-only fallback works with sparser sampling (every 3–5s), just slower.

## Privacy

Everything is local. No cloud vision APIs — they'd refuse this content and it
shouldn't leave the box anyway. Stash URL + API key live only in the gitignored
`config.toml`.

## Future directions

- **Stash plugin**: thin task trigger to kick off scoring from the Tasks page.
- **Pre-cut clips / culling**: an exporter that turns peaks into an ffmpeg
  keep-list (EDL) — for smoother megaboards and, eventually, removing
  non-taste footage. Non-destructive until explicitly chosen.
- **Multiple taste profiles**: separate tags/classifiers for different moods.

## Repo layout

```
src/peaks/
  config.py        # TOML + env config loading
  models.py        # dataclasses mirroring the Stash schema slice we use
  stash_client.py  # GraphQL client: version, scene iteration, marker writes
  cli.py           # `peaks test | scenes | stats`
config.example.toml
docs/ARCHITECTURE.md
```
