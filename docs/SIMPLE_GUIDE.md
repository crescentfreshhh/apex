# The no-BS guide

Do these steps in order, on the computer where Stash and your videos live.
Every command is copy-paste. After each one I tell you what you should see.

If anything errors or looks wrong: **copy the whole error message and paste it
to Claude.** Don't spend an hour fighting it.

---

## Step 1 — Open a terminal and get the code

```bash
git clone https://github.com/crescentfreshhh/stasssh.git
cd stasssh
```

(Already have it? `cd stasssh` then `git pull` instead.)

## Step 2 — Install

One-time setup. The middle command downloads a few GB (the AI models' library),
so let it run.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"
pip install -e ".[label]"
```

**You should see:** lots of download text, ending back at your prompt with no
red errors.

> ⚠️ The `source .venv/bin/activate` line puts `(.venv)` at the start of your
> prompt. **Every time you open a new terminal for this project, run that line
> again first.** If a `peaks` command ever says "command not found", you forgot.

## Step 3 — Point it at your Stash

```bash
cp config.example.toml config.toml
nano config.toml
```

In the editor, fix two lines near the top:

- `url` — should already say `http://192.168.1.2:6969`. Fix it if your address
  changed.
- `api_key` — in Stash, go to **Settings → Security**, copy your API key,
  paste it between the quotes. (No key set up in Stash? Leave it as `""`.)

Also find the line `device = ""` and change it to `device = "cuda"`.

Save and exit: **Ctrl-O, Enter, Ctrl-X**.

## Step 4 — Does it connect?

```bash
peaks test
```

**You should see:** `✓ Connected to Stash ...` and your scene count.
If you see `✗ Connection failed` — check the url and api_key from Step 3.

## Step 5 — Test run on 20 videos

```bash
peaks embed --limit 20
```

This "reads" 20 videos and remembers what every moment looks like. It prints a
line per video like:

```
+ [3/20] scene 123: 900 frames in 45.2s, eta ~0.2h
```

**Look at the seconds per video.** Under a minute or two each = fine.
Way slower = tell Claude the number, there are speed settings we can flip.

## Step 6 — Show it what you like

Make a folder and put **10–30 screenshots** in it — stills that show exactly
the kind of moment you want it to find. Grab them however you like (Stash's
screenshot button works).

```bash
mkdir references
```

Then copy your images into that `references` folder (drag-and-drop in your
file manager is fine — .jpg or .png).

**Rule of thumb:** every picture should make you think "yes, THIS, exactly."
Ten great pictures beat thirty okay ones.

## Step 7 — The big moment

```bash
peaks score --limit 20
```

This prints a list of timestamps it *thinks* you'll like in those 20 videos —
it doesn't change anything in Stash yet. Something like:

```
~ scene 123:   312.5 -  341.0s  peak=0.512
```

**Now go check:** open those scenes in Stash, jump to those times.
Is it finding your kind of moments?

- **Mostly yes** → 🎉 continue to Step 8.
- **Mostly no / nothing found** → tell Claude what it flagged vs. what you
  wanted. This is expected sometimes and there's a plan B — you did nothing wrong.

## Step 8 — Save the results into Stash

```bash
peaks score --limit 20 --write
```

Now open Stash — those 20 scenes have **markers** at the good moments (tagged
`apex`). Made a mess and want to start over? `peaks clear --tag apex --write`
deletes all of them.

## Step 9 — Do the whole library

This is the long one — hours, maybe overnight. It keeps going after you close
the terminal, and if it gets interrupted it picks up where it left off.

```bash
nohup peaks embed > embed.log 2>&1 &
```

To check on it anytime: `tail embed.log` (the `eta ~X.Xh` on the last line
tells you how long is left).

When it's done:

```bash
peaks score --write
```

## Step 10 — The megaboard 🎬

```bash
peaks playlist
peaks serve
```

Leave that running, open a browser, go to **http://localhost:8800**.

A grid of your best moments, playing at once, forever. Click a tile for sound.
Click it again for silence. **Ctrl-C** in the terminal to stop the server.

---

## That's it. Later, when you want more

- **Make it smarter:** `peaks label` opens a page that shows you frames — press
  **→** for "yes, want it," **←** for "no." Do a few hundred (takes ~15 min),
  then run `peaks train`, then `peaks score --write` again. It now knows *your*
  taste, not just "looks similar to my screenshots."
- **Start fresh:** `peaks clear --tag apex --write` then score again.

## Cheat sheet

| I want to... | Type |
|---|---|
| Check it's connected | `peaks test` |
| Read videos into memory | `peaks embed` |
| Find my moments (preview) | `peaks score` |
| Save moments into Stash | `peaks score --write` |
| Delete the saved moments | `peaks clear --tag apex --write` |
| Teach it my taste | `peaks label` then `peaks train` |
| Watch the megaboard | `peaks playlist` then `peaks serve` |

*(Every terminal: `cd stasssh` then `source .venv/bin/activate` first.)*
