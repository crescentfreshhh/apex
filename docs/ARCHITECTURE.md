# Architecture & gameplan

This captures the design decisions so we can pick the project back up cold.

## The problem

A large, highly-curated Stash library where most scenes contain only a few
moments worth watching (specific positions, angles, framing). We want to:

1. Learn the user's visual taste from examples.
2. Locate the matching timestamp-segments ("apexes") across the whole library.
3. Surface them as Stash scene markers.
4. Play them back — a queue, and a "megaboard" grid of simultaneously looping
   clips that continuously cycles in new apexes.

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
| Naming | **Opus** = platform, **apex** = unit | Locked |
| Embedding | **DINOv2 + CLIP ensemble** | Structure *and* nameable attributes — both wanted |

## Key decisions (open / deferred)

- **Tier 1 vs Tier 2**: build Tier 1 first (validate), then Tier 2.
- **Per-profile channel weighting**: how DINOv2 vs CLIP vs detector signals get
  combined per taste profile — tune once we see real scores.

## Naming

**Locked:** **Opus** = the platform (the curated collection + megaboard).
**apex** = the unit (one good timestamp-segment); default marker tag name.

Shortlist we considered, for the record:

- *Summit family:* apex ✅, crest, zenith, acme, pinnacle, summit, climax
- *Music family:* opus ✅, crescendo, coda, hook, the drop, refrain
- *Gold family:* gold, paydirt, gem, nugget, prime, choice cut
- *Heat family:* heat, fire, fuel, the sauce, spark

Per-profile tags extend the unit name: `apex:position`, `apex:heels`,
`apex:bodytype`, etc.

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

## Embedding channels: what each one captures

We embed every sampled frame through **two** channels and cache both:

- **DINOv2** (self-supervised, no text) — captures visual *structure*: poses,
  positions, camera angle, framing, body type. The model's strong suit and the
  primary channel for the user's #1 target (position/angle).
- **CLIP** (image↔text) — open-vocabulary, so it scores *nameable* concepts
  ("high heels", "latex", "three people"). The channel for outfits/accessories.

A frame embedding captures the overall **gestalt** — big dominant things come
through strongly, small details get drowned out. So attribute by attribute:

| Target attribute | Captured? | Best channel | Notes |
|---|---|---|---|
| Sex position / pose | ✅ strong | DINOv2 | large, structural |
| Camera angle / framing | ✅ strong | DINOv2 | it *is* the composition |
| Body type | ✅ good | DINOv2 | overall shape is a big signal |
| Number of performers | 🟡 rough | person detector | embeddings sense "how many" loosely; a YOLO-class detector *counts* reliably |
| Specific clothing (heels) | 🟡 weak alone | CLIP / clothing detector | tiny fraction of pixels; global embedding barely weights it |

**Handling fine detail / multiple attributes:**

1. **Ensemble** DINOv2 (structure) + CLIP (nameable) so each attribute rides the
   channel that sees it.
2. **Detectors** (person/footwear/clothing) for countable or small, locatable
   things the embeddings miss.
3. **Multiple taste profiles**, not one god-model: a `apex:position` profile, an
   `apex:heels` profile, an `apex:bodytype` profile — each its own
   tag/classifier — then combine at query time (e.g. position AND heels). More
   controllable, each dialed independently.

**Confound warning.** A classifier learns whatever *statistically separates*
the yes/no sets — possibly things you didn't intend (a studio's color grade,
resolution, lighting). Mitigation: **diverse negatives**, so the only consistent
thing about the "yes" pile is the actual target. Watch for this at Tier-1
validation.

**Reference-material guidance.** A "reference frame" is a single still. Tier 1
wants ~10–30 good ones; Tier 2 ideally a few hundred yes + a few hundred no
(start ~100 each, grow). Diversity/quality beats raw count. Start with **one
profile for the #1 thing** (position/angle), validate, then add CLIP/detector
profiles for outfits and counts.

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
- **Pre-cut clips / culling**: an exporter that turns apexes into an ffmpeg
  keep-list (EDL) — for smoother megaboards and, eventually, removing
  non-taste footage. Non-destructive until explicitly chosen.
- **Multiple taste profiles**: separate tags/classifiers for different moods.

## Repo layout

```
src/peaks/
  config.py        # TOML + env config (stash, sampling, embedding, scoring, markers)
  models.py        # dataclasses mirroring the Stash schema slice (+ fingerprints)
  stash_client.py  # GraphQL: version, scene iteration, marker writes, stream URLs
  sampling.py      # ffmpeg frame sampler (+ pure plan_timestamps)
  embedding.py     # Embedder ABC; DINOv2, CLIP (lazy torch), FakeEmbedder
  cache.py         # resumable on-disk embedding cache (.npz per scene, by fingerprint)
  scoring.py       # similarity scoring + hysteresis segment extraction (pure numpy)
  pipeline.py      # embed_library / score_library orchestration
  cli.py           # `peaks test | scenes | stats | embed | score`
tests/             # offline suite (fake embedder; no torch/ffmpeg/Stash needed)
config.example.toml
docs/ARCHITECTURE.md
```

Schema note: GraphQL queries were verified against the current `stashapp/stash`
`develop` schema (VideoFile, SceneMarker, SceneMarkerCreateInput all confirmed).
