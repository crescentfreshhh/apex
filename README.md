# Opus

Find — and play back — the moments you actually care about in a [Stash](https://stashapp.cc) library.

> **Opus** is the platform. An **apex** is a single good timestamp-segment — the
> unit it finds. (The Python package is `peaks` for now; the concepts are what matter.)

Most scenes only have a small amount of material worth watching. **Opus** learns
*your* taste from examples, scores every video frame-by-frame to locate the
apexes — the moments that match — writes those back into Stash as **scene markers**,
and (later) feeds them into a live "megaboard" — a grid of simultaneously
looping clips that continuously cycles in new highlights.

Everything runs **locally**. Your library never leaves the machine.

---

## How it works

```
┌─────────────┐   GraphQL    ┌──────────────────────┐
│   Stash     │◄────────────►│  "Brain" (Python)     │
│  (library + │  read scenes │  - frame sampler      │
│   markers)  │  write markers│  - embedder (cached) │
└─────┬───────┘              │  - taste classifier   │
      │                      │  - segment scorer     │
      │ thin plugin          └──────────┬───────────┘
      │ (Tasks button)                   │ segments → markers
      │                                  ▼
      └─────────────────────►┌──────────────────────┐
                             │  Megaboard (web app) │
                             │  grid of looping cuts │
                             └──────────────────────┘
```

The ML never classifies video directly. Instead it **embeds sampled frames into
vectors once** (the only GPU-heavy step, cached to disk forever), then learns
your taste cheaply in that vector space. Two embedding channels work together:
**DINOv2** for visual *structure* (positions, angles, body type) and **CLIP**
for *nameable* attributes (outfits, heels, etc.). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and rationale.

## Build roadmap

| Step | What | Status |
|-----:|------|--------|
| 1 | **Plumbing** — config + GraphQL client that reads scenes/markers | ✅ |
| 2 | Frame sampler + DINOv2 **and CLIP** embedders, resumable, on-disk cache | ✅ built, offline-tested* |
| 3 | Tier-1 similarity scorer → `apex` markers (with dry-run) | ✅ built, offline-tested* |
| 4 | Tier-2 rapid frame-labeler + trained taste classifier | ✅ built, tested* |
| 5 | Megaboard web app (live-stream grid) | ✅ built, browser-tested |
| — | *(later)* Stash plugin trigger; pre-cut/cull exporter | ⬜ |

\* Logic is unit-tested with a deterministic fake embedder (no torch/ffmpeg/Stash
needed). The real DINOv2/CLIP run and live marker-writing await first contact
with the server + a GPU burst — that's the validation milestone.

## Setup

**Running on Unraid / Docker?** That's the intended deployment: a GitHub
Action publishes the image to `ghcr.io/crescentfreshhh/apex:latest`, and
`unraid/peaks.xml` is a ready-made Unraid template (GPU via the Nvidia Driver
plugin + `--runtime=nvidia`). Follow **[docs/SIMPLE_GUIDE.md](docs/SIMPLE_GUIDE.md)**
— it covers everything including moving the GPU off a VM. The steps below are
for a bare-metal/manual install.

Requires Python 3.11+.

```bash
# 1. Install the plumbing (light — just `requests`)
pip install -e .

# 2. Point it at your Stash server
cp config.example.toml config.toml
$EDITOR config.toml        # set url + api_key (config.toml is gitignored)

# 3. Verify the connection
peaks test
```

> ML dependencies (torch, etc.) are installed separately when we reach step 2:
> `pip install -e ".[ml]"`

## Usage

**Connect & inspect (step 1 — no ML deps):**
```bash
peaks test              # verify connection + print Stash version & scene count
peaks scenes            # list scenes: id, duration, marker count, title/path
peaks scenes --limit 20 # ...just the first 20
peaks stats             # library summary (scenes, total hours, markers)
```

**Embed & score (steps 2–3 — needs `pip install -e ".[ml]"` + ffmpeg):**
```bash
# 1. The GPU pass: sample frames and embed the whole library into the cache.
#    Resumable — re-running skips anything already cached.
peaks embed                     # or --limit 20 to try a slice first

# 2. Drop ~10-30 stills you love into ./references/, then preview matches.
#    Dry-run prints the segments it WOULD create — tune thresholds in config.
peaks score                     # dry run, writes nothing
peaks score --write             # write the apex markers into Stash
peaks score --tag apex-heels --write   # a second profile: reads its stills
                                       # from references/apex-heels/ automatically
peaks clear --tag apex          # count our markers; --write deletes them
```
Re-running `score --write` is idempotent: segments overlapping an existing
marker of the same tag are skipped, so threshold tweaks add new finds instead
of duplicating old ones. Use `clear --write` for a clean slate.

**Train a taste model (step 4 — Tier 2):** teach it *your* taste beyond simple
similarity. Needs the `[label]` extra (`pip install -e ".[label]"`) for the UI.
```bash
peaks label                     # rate candidate frames yes/no in a browser UI
                                # (seeded by Tier-1, so you rate what matters)
peaks train                     # fit a classifier from your labels -> models/apex.pkl
peaks score                     # now auto-uses the model (Tier 2) when present;
                                # pass --references to force Tier-1 similarity
```
Each taste profile is its own tag + model, e.g. `peaks label --tag apex:heels`
then `peaks train --tag apex:heels`.

**Megaboard (step 5):**
```bash
peaks playlist          # export apex markers -> webapp/playlist.json
peaks serve             # serve the grid at http://localhost:8800
```
A grid of looping clips that cycles in new apexes continuously. See
[`webapp/README.md`](webapp/README.md).

Config resolves from environment variables first (`STASH_URL`, `STASH_API_KEY`),
then `config.toml`, then built-in defaults. Run the test suite with `pytest`.

## A note on the names

- **Opus** — the platform / the whole curated collection of best moments.
- **apex** — one good timestamp-segment (the unit Opus finds). Used as the
  default marker **tag name**, configurable in `config.toml` (`markers.tag_name`).

You can maintain any number of **taste profiles**, each with its own tag,
reference folder (`references/<tag>/`), and trained model — e.g. `apex-position`,
`apex-heels`, `apex-bodytype` — and mix them on one megaboard
(`peaks playlist --tag apex --tag apex-heels`). Hyphenated tag names are
recommended: they map 1:1 to folder names (Windows/SMB can't create folders
containing `:`). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the
naming shortlist and how attributes map to embedding channels.
