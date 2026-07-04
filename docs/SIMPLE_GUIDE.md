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
   go to **github.com → your profile → Packages → `stasssh` → Package
   settings → Change visibility → Public**. (If the Packages page is empty,
   the image hasn't built yet — repo → Actions tab → wait for
   "Build and publish Docker image" to go green, ~20 min.)

2. **Find your Stash media path.** Docker tab → click your **Stash** container
   → Edit. Look at its path mappings and write down the pair for your videos —
   for example host `/mnt/user/media/` ↔ container `/data`. peaks must use the
   **exact same pair**, or it can't find the files Stash points it at.

3. **Add the template.** Docker tab → scroll to the bottom → **Template
   Repositories** → add this line → Save:

   ```
   https://github.com/crescentfreshhh/stasssh
   ```

   Then click **Add Container** and pick **peaks** from the Template dropdown.
   (If it doesn't appear there, choose any blank template and fill in the
   fields by hand from `unraid/peaks.xml` in the repo.)

4. **Fill in the fields:**
   - **Media** — the pair from step 2: host path = your media share,
     container path = whatever Stash uses (edit it if it isn't `/data`).
   - **STASH_URL** — pre-filled with `http://192.168.1.2:6969`; fix if needed.
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

- **Make it smarter:** in the console run `peaks label --host 0.0.0.0`, then
  browse to `http://<unraid-ip>:7860`. It shows frames — press **→** for
  "want it," **←** for "no." A few hundred takes ~15 min. Then `peaks train`
  (it prints a quality score) and `peaks score --write` again.
- **Start fresh:** `peaks clear --tag apex --write`, then score again.
- **Updates:** when Claude pushes changes, Unraid's Docker tab will show an
  update for peaks — click Update, done.

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

Megaboard: `http://<unraid-ip>:8800` · Labeler: `http://<unraid-ip>:7860` ·
References folder: `\\<unraid-name>\appdata\peaks\references`
