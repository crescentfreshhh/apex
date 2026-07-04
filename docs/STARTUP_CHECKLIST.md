# Server-up checklist

> **Unraid/Docker install?** Phases 0–2 are replaced by
> [`SIMPLE_GUIDE.md`](SIMPLE_GUIDE.md) Parts A–B (GPU handover + container
> install). Phases 3+ apply as-is, run from the container console.

Run top to bottom on the machine where Stash + the video files + the borrowed
GPU live. Each phase has a "✅ expect" so you know it worked before moving on.
The starred phase (6) is the real validation milestone — don't full-send before it.

---

## 0. Get the code on the box
- [ ] `git clone <repo>` (or `git pull` if already cloned) — everything is on `main`
- [ ] `cd stasssh`
- [ ] Confirm `ffmpeg` and `ffprobe` are installed: `ffmpeg -version`
- [ ] Confirm the GPU is visible: `nvidia-smi` shows the 3080 Ti

## 1. Install
- [ ] (recommended) `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -e ".[ml]"`  (torch, open_clip, torchvision, pillow, sklearn)
- [ ] `pip install -e ".[label]"`  (gradio — only needed for Tier-2 labeling)
- [ ] ✅ expect: `peaks --help` lists all commands

## 2. Configure
- [ ] `cp config.example.toml config.toml`
- [ ] Set `stash.url` (`http://192.168.1.2:6969`) and `stash.api_key` if auth is on
- [ ] Set `embedding.model = "dino"` and `embedding.device = "cuda"`
- [ ] ✅ expect: `config.toml` is gitignored (it holds your key) — `git status` shows it untracked

## 3. Verify connection
- [ ] `peaks test`  → ✅ expect: "Connected to Stash <version>" + scene count
- [ ] `peaks stats` → ✅ expect: sane total hours + existing marker count
- [ ] `peaks scenes --limit 5` → ✅ expect: real titles/paths/durations

> If `peaks test` fails: check the URL, the API key, and that the box can reach
> the Stash host. The schema is already verified, so it shouldn't be a query error.

## 4. Trial embed on a slice (proves ffmpeg + GPU + cache)
- [ ] `peaks embed --limit 20`
- [ ] ✅ expect: "scene N: M frames -> cache" lines, no failures
- [ ] ✅ expect: `.npz` files under `cache/embeddings/dinov2/`
- [ ] Watch `nvidia-smi` — GPU should be working; note the per-scene time to
      estimate the full run
- [ ] 📝 report back: rough seconds/scene, any errors

## 5. Collect Tier-1 references
- [ ] `mkdir references`
- [ ] Drop ~10–30 **stills** that purely nail your #1 thing (position/angle)
- [ ] Aim for variety in everything *except* the target trait (avoids the model
      latching onto a studio/lighting confound)

## 6. ⭐ THE MOMENT OF TRUTH — dry-run score on the slice
- [ ] `peaks score --limit 20`  (dry run — writes nothing)
- [ ] ✅ expect: printed `scene X: start-end peak=...` lines
- [ ] Open a few of those scenes in Stash at the printed timestamps. **Do they
      land on the moments you actually want?**
- [ ] Tune in `config.toml` `[scoring]` and re-run:
      - too few/short hits → lower `high`/`low`, lower `min_duration`
      - too much junk → raise `high`/`low`, raise `min_duration`
- [ ] 📝 report back: is DINOv2 finding your apexes? screenshots/timestamps help

### Decision point
- **Looks good** → continue to phase 7 (full run), optionally 9 (Tier-2 to sharpen)
- **Mediocre** → try `embedding.model = "clip"` (re-embed the slice), or jump to
  Tier-2 labeling (phase 9) which learns your taste directly. Tell me and we'll adjust.

## 7. Full embed (the GPU burst)
- [ ] `peaks embed`  (resumable — safe to Ctrl-C and resume; only embeds new scenes)
- [ ] ✅ expect: it skips the 20 already cached, processes the rest
- [ ] When done, the GPU is no longer needed — **you can return the 3080 Ti**
      (scoring + training are CPU-cheap)

## 8. Score the library → write markers
- [ ] `peaks score`  (full dry run) → final threshold tune
- [ ] `peaks score --write`  → ✅ expect: "segments_created=N"
- [ ] Verify in Stash UI: scenes now have `apex` markers at the right spots

## 9. (Optional, recommended) Tier-2 — teach it your taste
- [ ] `peaks label`  → rate candidate frames yes/no in the browser (~15 min)
- [ ] `peaks train`  → ✅ expect: "Trained logreg on N frames -> models/apex.pkl"
- [ ] `peaks score --write`  → now auto-uses the model (prints "via Tier-2 model")
- [ ] Compare against the Tier-1 markers; keep whichever you prefer

## 10. Megaboard
- [ ] `peaks playlist`  → ✅ expect: "Wrote N apex(es) -> webapp/playlist.json"
- [ ] `peaks serve`  → open `http://localhost:8800`
- [ ] ✅ expect: a 3×3 grid of looping apexes; click a tile to unmute
- [ ] If playback stutters, shrink the grid (fewer simultaneous streams)
- [ ] 📝 report back: smooth? this tells us if we need the pre-cut-clips path

## 11. (Later) Multiple taste profiles
- [ ] References or labels for a second facet, then:
      `peaks label --tag apex:heels` → `peaks train --tag apex:heels`
      → `peaks score --tag apex:heels --write`
- [ ] Each profile is its own tag + model; combine at query/playlist time

---

### Things to tell me when you hit them
- Per-scene embed time (phase 4) — sets expectations for the full run
- Whether DINOv2 lights up your taste (phase 6) — the make-or-break signal
- Megaboard smoothness with live streaming (phase 10) — decides pre-cut clips

### Quick reference — the whole flow
```
test → stats → embed --limit 20 → [add references] → score --limit 20 (TUNE)
     → embed (full) → score --write → [label → train → score --write]
     → playlist → serve
```
