# Day 1 plan — server's up

A time-blocked session plan for the first day back. The phase-by-phase command
reference lives in [`STARTUP_CHECKLIST.md`](STARTUP_CHECKLIST.md); this is the
*strategy* for the day: what order, what runs in parallel, where the decision
gates are, and what to do when something disappoints.

**Goal for the day:** by end of day, know whether Tier-1 similarity finds your
apexes (the make-or-break question), have the full-library embed running or
done, and ideally have seen the megaboard play real segments.

**Not goals for day 1:** perfect thresholds, Tier-2 training, multiple
profiles. Those are day 2+ — don't rabbit-hole.

---

## Night before (or first 15 min) — no GPU needed

```bash
git clone <repo> && cd stasssh          # or git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"                  # big download (torch) — start it early
pip install -e ".[label]"
cp config.example.toml config.toml      # set url, api_key, device = "cuda"
ffmpeg -version && nvidia-smi           # both must answer
```

Also: start thinking about your 10–30 reference stills (Block C needs them).
Screenshots from scenes you own are ideal — grab them via Stash's UI or mpv
screenshots as you browse. Variety in everything *except* the target trait.

---

## Block A — smoke test (10 min)

```bash
peaks test      # "Connected to Stash <version>" + scene count
peaks stats     # total hours + existing markers — sanity-check the numbers
peaks scenes --limit 5
```

**Gate:** all three answer sanely → continue. If not, it's config/network —
fix before anything else. (The GraphQL is schema-verified; don't debug queries.)

## Block B — trial embed + throughput math (30–45 min)

```bash
peaks embed --limit 20
```

While it runs, watch `nvidia-smi` and note wall-clock. When done:

> **seconds-per-scene × library size = full embed ETA.** Write this number down.

Expect decode (CPU/ffmpeg), not the GPU, to be the bottleneck — frames are
downscaled at extraction which helps, but a 30-min HD file still takes minutes
to decode. Rough planning bands for ~1000 scenes:

| Per-scene time | Full library | Plan |
|---|---|---|
| < 1 min | < ~17 h | start full embed now, runs into the evening |
| 1–3 min | ~17–50 h | start now, let it run overnight+; it's resumable |
| > 3 min | days | **stop and ping me** — we bump `interval_seconds` to 3–4, and/or I add hw-accelerated decode / keyframe-only sampling same-day |

**Kick off the full run in a way that survives your terminal closing:**

```bash
nohup peaks embed > embed.log 2>&1 &     # or run it inside tmux/screen
tail -f embed.log                        # check in whenever
```

Safe to Ctrl-C / reboot / resume anytime — it skips everything already cached.
**Everything after this block uses only the 20 cached scenes** — the full embed
just churns in the background all day.

## Block C — references (30 min, parallel with the embed)

Collect 10–30 stills into `references/`. Quality bar per image: "if the model
found 100 more moments exactly like this, I'd be thrilled." Prune anything
you're lukewarm on — 12 great beats 30 decent.

## Block D — ⭐ the moment of truth (1–2 h, the heart of day 1)

```bash
peaks score --limit 20          # dry run against the 20 cached scenes
```

Open the printed timestamps in Stash. Judge honestly:

- **Hit rate**: of the segments it flagged, how many are actually your thing?
- **Miss rate**: scan one scene you know well — did it find the moments you'd
  have marked yourself?

Tune → re-run (each iteration is seconds, no GPU):

| Symptom | Knob (config.toml `[scoring]`) |
|---|---|
| Flags everything / segments huge | raise `high` (and `low`) |
| Finds almost nothing | lower `high`/`low`, lower `min_duration` |
| Right area, sloppy edges | raise `smooth_window`, adjust `pad` |
| Thresholds feel arbitrary / unstable across scenes | set `normalize = "scene-z"`, then think in std-devs: `high = 2.0`, `low = 1.0` |

**Gate — pick the branch that matches reality:**

1. **Most flagged segments are right, most known moments found** → Tier-1
   validated. Continue to Block E. 🎉
2. **Signal but fuzzy** (right scenes, wrong moments; or hit rate ~50%) →
   good enough to proceed *and* the fix is known: Tier-2. Do Block E with what
   you have, start labeling tomorrow.
3. **Barely better than random** → stop tuning. Ping me with 2–3 examples of
   what it flagged vs what you wanted. Likely moves: switch the slice to
   `model = "clip"` and compare, or go straight to Tier-2 (a trained head can
   work even when raw similarity doesn't). This outcome is informative, not
   fatal — don't burn the day re-running thresholds.

## Block E — write + megaboard on the slice (45 min)

```bash
peaks score --limit 20 --write   # idempotent; peaks clear --tag apex --write = reset
peaks playlist
peaks serve                      # http://127.0.0.1:8800
```

Check markers look right in Stash's UI, then watch the board (start 2×2 or
3×3 — every tile is a live seek against Stash). Click tiles, let it cycle a
few minutes. Note stutter — that's the pre-cut-clips decision data.

## End of day — capture these while fresh

1. Per-scene embed time + how far the full embed got.
2. Block D verdict: branch 1, 2, or 3 — plus a couple of example timestamps.
3. Megaboard smoothness + grid size used.
4. Anything that errored (paste from `embed.log`).

Drop them to me in the session and we plan day 2 from data.

## Day 2 preview (don't start these on day 1)

- Full-library score + `--write` once the embed finishes.
- Tier-2: `peaks label` (~15–20 min of rating) → `peaks train` → compare
  against Tier-1 markers.
- Megaboard tuning or the pre-cut-clips exporter, per Block E's verdict.
- Second taste profile if the first one's humming.

---

### Contingency quick-reference

| Problem | Move |
|---|---|
| torch can't see the GPU | `python3 -c "import torch; print(torch.cuda.is_available())"` — if False, reinstall torch with the CUDA build for your driver |
| DINOv2 hub download fails | needs GitHub access once; retry or ping me for an offline path |
| A scene fails to decode | it's logged and skipped, the run continues — collect them for me |
| Embed ETA is absurd | bump `interval_seconds` to 3–4 (cache re-embeds at the new density), or ping me for hw-decode |
| Megaboard tiles all black | wrong `?start=` behavior is auto-detected, so check the browser console + that `playlist.json` URLs open directly in a tab |
