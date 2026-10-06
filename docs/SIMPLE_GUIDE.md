# The no-BS guide (Unraid edition)

peaks runs as a **Docker container on your Unraid box**, right next to Stash.
You install it from the Unraid web GUI — no terminal on the server. Commands
run inside the container's built-in console (one click), and the megaboard is
just a web page.

Do the parts in order. **If anything errors: copy the whole message and paste
it to Claude.** Don't spend an hour fighting it.

---

# Part A — Move the GPU from the Windows VM to Unraid

Your 3080 Ti is currently passed through to a Windows VM. Unraid can't share a
GPU between a VM and Docker at the same time, so we borrow it. This takes two
reboots of the Unraid server, ~30 minutes.

> ⚠️ While the GPU is on loan, **do not start the Windows VM** — it will fail
> to boot (or worse). Part E gives it back.

1. **Shut down the Windows VM.** VMs tab → stop it. If it has "Autostart" on,
   click the VM → toggle Autostart **off** (so a server reboot doesn't relaunch it).
2. **Unbind the GPU.** Tools → **System Devices**. Find your
   `NVIDIA GeForce RTX 3080 Ti` — it appears as **two** entries (video +
   audio, e.g. `01:00.0` and `01:00.1`), both with a checked box binding them
   to VFIO. **Uncheck both** → click the bind/apply button at the bottom.
3. **Reboot Unraid** (top-right power icon → Reboot).
4. **Install the Nvidia driver.** Apps tab → search **"Nvidia Driver"**
   (by ich777) → Install. Let it finish downloading the driver, then
   **reboot once more** if the plugin page asks for it.
5. **Verify:** Settings → **Nvidia Driver** should now show
   `NVIDIA GeForce RTX 3080 Ti` with a driver version. That's the GPU ready
   for Docker.

# Part B — Install the peaks container

1. **Make the image reachable (one-time, on GitHub).** The container image is
   published automatically from this repo to GitHub's registry. If your GitHub
   repo is private, the image is too, and Unraid won't be able to pull it:
   go to **github.com → your profile → Packages → `apex` → Package
   settings → Change visibility → Public**. (If the Packages page is empty,
   the image hasn't built yet — repo → Actions tab → wait for
   "Build and publish Docker image" to go green, ~20 min.)

2. **Find your Stash media path.** Docker tab → click your **Stash** container
   → Edit. Look at its path mappings and write down the pair for your videos —
   for example host `/mnt/user/media/` ↔ container `/data`. peaks must use the
   **exact same pair**, or it can't find the files Stash points it at.

3. **Add the container.** Modern Unraid removed the old "Template
   Repositories" box, so use one of these:

   **Method A — drop in the template (recommended, pre-fills everything).**
   From your PC, open the Unraid **flash** share in your file explorer:

   ```
   \\<your-unraid-name>\flash\config\plugins\dockerMan\templates-user
   ```

   (Don't see a `flash` share? Enable it: Unraid → **Main** → click **Flash**
   → set **Export** to Yes → it appears.) Download `unraid/peaks.xml` from the
   repo (GitHub → the file → **Download raw file**) and copy it into that
   folder, renamed to **`my-apex.xml`**.

   Then: Docker tab → **Add Container** → the **Template** dropdown at the top
   → under *User templates* pick **peaks**. Every field fills in for you.

   **Method B — by hand (works everywhere, more typing).** Docker tab → **Add
   Container**, leave Template blank, switch the toggle top-right to
   **Advanced view**, and enter:
   - **Repository:** `ghcr.io/crescentfreshhh/apex:latest`
   - **Extra Parameters:** `--runtime=nvidia`
   - **Add Port** ×2: `8800` and `7860` (both TCP)
   - **Add Path** — Config: container `/config`, host
     `/mnt/user/appdata/peaks`
   - **Add Variable** ×5: `STASH_URL`, `STASH_API_KEY`, `PEAKS_DEVICE` (=`cuda`),
     `NVIDIA_VISIBLE_DEVICES` (=`all`), `NVIDIA_DRIVER_CAPABILITIES` (=`all`)
   - the **Media** path is step 4 below

4. **Fill in / check these fields** (both methods):
   - **Media path** — click **Add Path** (Method B) or edit the Media path
     (Method A): container path = whatever your **Stash** container uses from
     step 2, host path = your media share, access mode **Read Only**.
   - **STASH_URL** — `http://192.168.1.2:6969` (fix if your address differs).
   - **STASH_API_KEY** — Stash → Settings → Security → copy the API key in.
     (No auth on your Stash? Leave empty.)
   - Everything else can stay at its defaults.

5. Click **Apply**. Unraid downloads the image (it's big — several GB — let it
   run) and starts the container.

# Part C — First contact

All commands run in the container's console: **Docker tab → click the peaks
icon → >_ Console**. A black window opens — that's where you type.

1. Does it see Stash?

   ```
   peaks test
   ```

   **You should see:** `✓ Connected to Stash ...` and your scene count.
   `✗ Connection failed` → recheck STASH_URL / STASH_API_KEY in the template.

2. Does it see your files? First:

   ```
   peaks scenes --limit 3
   ```

   Copy one of the printed file paths, then:

   ```
   ls "<paste the path here>"
   ```

   **You should see:** the filename echoed back. `No such file or directory` →
   your Media mapping doesn't match Stash's (Part B step 2) — fix the
   container's path mapping and try again.

3. Test-read 20 videos with the GPU:

   ```
   peaks embed --limit 20
   ```

   It prints a line per video like `+ [3/20] scene 123: 900 frames in 45.2s`.
   **Note the seconds per video.** A minute or two each = fine. Way slower =
   tell Claude the number — there are speed settings we can flip.

# Part D — Teach it, run it, watch it

1. **Give it examples.** From your normal PC, open the appdata share in your
   file explorer:

   ```
   \\<your-unraid-name>\appdata\peaks\references
   ```

   Drop in **10–30 screenshots** of exactly the kind of moment you want it to
   find (.jpg/.png — Stash's screenshot button is an easy source). Every image
   should make you think "yes, THIS." Ten great beats thirty okay.

2. **The big moment.** Back in the container console:

   ```
   peaks score --limit 20
   ```

   It prints timestamps it *thinks* you'll like — nothing is saved yet. Open
   those scenes in Stash at those times and judge it:
   - **Mostly right** → 🎉 carry on.
   - **Mostly wrong** → tell Claude what it flagged vs. what you wanted.
     There's a plan B; you did nothing wrong.

3. **Save markers for those 20:**

   ```
   peaks score --limit 20 --write
   ```

   Check Stash — those scenes now have `apex` markers. (Undo everything:
   `peaks clear --tag apex --write`.)

4. **The whole library** (hours — likely overnight; safe to close the console,
   it keeps running and resumes if interrupted):

   ```
   nohup peaks embed > /config/embed.log 2>&1 &
   ```

   Check progress anytime: `tail /config/embed.log` — the `eta ~X.Xh` on the
   last line is your answer. When it's done:

   ```
   peaks score --write
   ```

5. **The megaboard** 🎬 — the web page is already running. Generate its
   playlist, then open it:

   ```
   peaks playlist
   ```

   Browser → `http://<your-unraid-ip>:8800` (or click the peaks container →
   **WebUI**). A grid of your best moments, cycling forever. Click a tile for
   sound. After any re-score, run `peaks playlist` again and refresh the page.

# Part E — Give the GPU back to Windows

The GPU is only needed for `peaks embed` runs. Everything else (scoring,
training, the megaboard) is CPU-cheap — so once your library is embedded:

1. Docker tab → stop the **peaks** container (or set PEAKS_DEVICE to `cpu`
   and restart it, if you want to keep using it without the GPU).
2. Tools → **System Devices** → **re-check both** 3080 Ti entries → apply.
3. **Reboot Unraid.**
4. Start the Windows VM — it's back to normal. (The Nvidia driver plugin can
   stay installed; it won't touch a VFIO-bound card.)

To borrow the GPU again later (new videos to embed): repeat Part A steps 1–3
(the driver plugin is already installed).

---

## Later, when you want more

- **Add text search (CLIP):** on the Dashboard, hit **Embed → Advanced**, set
  **Model = CLIP**, and run it — a second pass that builds the text-search
  index alongside your DINOv2 one. No restart, no template edits; the model and
  sampling knobs are all set per-run right there. When it finishes, the search
  box in Explore comes alive.
- **Make it smarter:** in the console run `peaks label --host 0.0.0.0`, then
  browse to `http://<unraid-ip>:7860`. It shows frames — press **→** for
  "want it," **←** for "no." A few hundred takes ~15 min. Then `peaks train`
  (it prints a quality score) and `peaks score --write` again.
- **Files moved or deleted?** The cache is keyed by a stable file fingerprint,
  so a scene shuffled between subfolders in `/data/Rando` keeps its embeddings —
  nothing to redo. Hit **Sync now** on the Dashboard (or run `peaks sync`) to
  refresh moved paths and drop cache entries for scenes you deleted from Stash.
  If you set `PEAKS_EMBED_EVERY_HOURS`, this reconcile runs automatically after
  each recurring pass (moves always; deletions only if you set `PEAKS_PRUNE=true`).
- **A few scenes failed to embed?** They're logged automatically (a "Failed
  scenes" panel appears on the Dashboard, and `peaks fix --list` shows them).
  A failure is usually a seek/decode quirk, not a bad file — the same clip
  plays fine in VLC because VLC decodes linearly while our fast path seeks.
  Hit **Retry failed** (or run `peaks fix`) to re-attempt them through a
  fallback ladder: sparse without NVDEC, then a full linear decode. Anything
  that embeds is cleared from the log; stubborn survivors stay listed with the
  error so you can dig in.
- **Start fresh:** `peaks clear --tag apex --write`, then score again.
- **Updates:** when Claude pushes changes, Unraid's Docker tab will show an
  update for peaks — click Update, done.

## Many apexes — build up profiles over time

You're not limited to one taste. Each **profile** is its own tag, its own
example pictures, and (if you train one) its own model. `apex` is just the
first. Add as many as you want, whenever you want — a positions one, a heels
one, an angles one...

**Name them with hyphens** (`apex-heels`, `apex-pov`, `apex-bodytype`) — then
the folder name matches the tag exactly.

To add a profile, say `apex-heels`:

1. **Make its picture folder** inside references, named after the tag:

   ```
   \\<unraid-name>\appdata\peaks\references\apex-heels
   ```

   Drop 10–30 stills of *that specific thing* in there. (Pictures loose in
   the main `references` folder stay with plain `apex` — subfolders don't mix.)

2. **Score with its tag** (it finds the folder by itself):

   ```
   peaks score --tag apex-heels            # preview
   peaks score --tag apex-heels --write    # save markers
   ```

   In Stash, these show up as markers tagged `apex-heels`, living happily
   alongside your `apex` ones.

3. **Optionally train it smarter**, same as before but with the tag:

   ```
   peaks label --tag apex-heels --host 0.0.0.0
   peaks train --tag apex-heels
   ```

4. **Mix profiles on the megaboard** — repeat `--tag` for every profile you
   want in the grid:

   ```
   peaks playlist --tag apex --tag apex-heels
   ```

   Or just one profile for a themed board: `peaks playlist --tag apex-heels`.

The one-time work (the big overnight `peaks embed`) is **shared by all
profiles** — new profiles only need pictures and a scoring pass, which takes
minutes. Delete one profile's markers without touching the others:
`peaks clear --tag apex-heels --write`.

## Backups (and moving to a new server)

Peaks backs up everything it knows once a week — your grades log, taste labels
and models, settings, playlists, performer pictures and every embedding — to
`/data/.peaks-backups`, i.e. inside your library share on the array (so the
Media path must be **read-write**). Each snapshot is complete, but files that
didn't change since the last one are shared, so a week usually costs only what's
new. The library cleanup never touches that folder.

- **Settings → Backup & restore**: schedule (day, time, how many to keep),
  **Back up now**, and each snapshot's **Restore…** (a safety snapshot of how
  things are now is taken first; no restart needed) and **⬇ Small** (everything
  except the embeddings, as one small file for an off-box copy).
- **New server**: map the same library share to `/data` (read-write), open
  Settings → Backup & restore and restore the newest snapshot — no re-embedding.

## Automatic ingest (watch your download folder)

**Settings → Ingest → Watch for new files**: add the folder your downloads land
in (e.g. `/data/sysbackup/norating` — it must be inside a Stash library folder),
switch it on, Save. Files already there count as seen.

Peaks looks at the folder once a minute. A new video counts once it has stopped
growing for 2 minutes; when the folder has then been quiet for 3 minutes, one
Ingest runs for the whole batch — scanning only that folder — and the new scenes
appear in Review. Partial downloads (`.part`, `.!qB`, rsync's hidden temp files)
are ignored. One heavy job runs at a time: the watch waits for a running embed,
Ingest, Sync or backup, and the every-X-hours embed waits for an Ingest — each
starts the minute the other ends. New scenes are embedded with the main model;
text search (CLIP) for them is a manual pass (Activity → Embedding → Options →
CLIP). If an auto-ingest fails (Stash down…), its files are
tried again 10 minutes later; scenes an Ingest couldn't embed are embedded as
soon as nothing else runs. **Check now** looks straight away. Activity → Ingest shows what it's watching
and the last auto-ingest.

## Seedbox pipeline page (System → Seedbox)

One page that says whether the seedbox → library pipeline is healthy: a
GREEN / YELLOW / RED verdict with the reasons, the pull and prune scripts' last
successful runs, files pulled per day, the seedbox's torrent pool, and how much
is waiting in norating. Everything is read-only. In the container settings:

- **Seedbox pipeline state (read-only)**: host `/mnt/user/appdata/seedbox-pull`
  → container `/seedbox`, access mode **Read Only**.
- **PEAKS_QBT_URL / PEAKS_QBT_USER / PEAKS_QBT_PASSWORD**: the seedbox's
  qBittorrent Web UI address and login. Peaks only logs in and reads the torrent
  list (cached 5 minutes) — it never pauses, deletes or changes a torrent.
- The inbox is the norating folder you watch in Settings → Ingest (or set
  `PEAKS_SEEDBOX_INBOX` to its container path under `/data`).

A source Peaks can't read shows as "unavailable" with the reason — never as
zeros — and makes the verdict RED until it's fixed.

## Same-file copies (Stash's "File count > 1")

When the same download lands twice, Stash hangs both files on one scene (and
may rename one `None_2`). At the end of every Sync and Ingest Peaks removes the
extra file — **only** when it has the exact same size *and* the same content
hash as the one it keeps. It keeps the copy in the scene's tier folder
(Légendaire, Exceptionnelle…), then the cleaner name, then Stash's primary, then
the oldest. The scene itself — grade, O-count, markers, tags — is never touched,
and never loses its last file. This is not the Duplicates tool (that one is for
*different* scenes that look alike).

The first time, it only lists what it would remove: **Activity → Maintenance →
Same-file copies → Approve & remove**. After that it's automatic (Turn off
there any time). Same-size files it can't prove identical, and different-size
versions, are only listed for you to decide.

## Cheat sheet (in the container console)

| I want to... | Type |
|---|---|
| Check it's connected | `peaks test` |
| Read videos into memory | `peaks embed` |
| Find my moments (preview) | `peaks score` |
| Save moments into Stash | `peaks score --write` |
| Delete the saved moments | `peaks clear --tag apex --write` |
| Teach it my taste | `peaks label --host 0.0.0.0` then `peaks train` |
| Refresh the megaboard | `peaks playlist`, then reload the page |
| Reconcile after moving/deleting files | `peaks sync` (add `--dry-run` to preview) |
| Retry scenes that failed to embed | `peaks fix` (`--list` to see them first) |
| Add a new profile | folder `references\apex-<thing>`, then `peaks score --tag apex-<thing> --write` |
| Mix profiles on the board | `peaks playlist --tag apex --tag apex-<thing>` |

Megaboard: `http://<unraid-ip>:8800` · Labeler: `http://<unraid-ip>:7860` ·
References folder: `\\<unraid-name>\appdata\peaks\references`
