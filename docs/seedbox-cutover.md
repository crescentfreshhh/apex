# Moving the seedbox jobs from Unraid to Peaks

Peaks now runs **seedbox-pull**, **seedbox-purge-pulled** and **seedbox-report**
itself. Do this once, in this order. It takes about an hour, most of it waiting.

## Why nothing gets pulled twice or purged too early

- **No transfer of `.done` logs is needed.** Peaks reads and appends the *same*
  files your scripts use: `/mnt/user/appdata/seedbox-pull/1.done` and `vr.done`,
  mounted into Peaks at `/seedbox`. It uses the same `rclone lsf` command and filter
  file, so the entries match line for line. Anything already listed is skipped.
- **Purge only deletes what `.done` proves is on Jarvis.** It also starts with
  a **3-day dry run**: it lists what it *would* delete and deletes nothing.
- **Peaks refuses to pull while a script is still pulling.** If `runs.log` shows a
  pull Peaks didn't make in the last 45 minutes, Peaks waits and says why.

## Before you start

1. **Change your qBittorrent password** on the seedbox (it was in the purge
   script you shared). Then:
   - update `PEAKS_QBT_PASSWORD` in the Peaks container;
   - you can leave the old script as it is: it's being switched off.
2. **Update the Peaks image** so it includes rclone: pull or rebuild it the way you
   normally update Peaks.
3. **Check where `/data` points.** In the Peaks container settings, look at the host
   path of **"Media (match Stash's container path!)"**. The pull writes to
   `<that path>/Rando/norating` and `<that path>/#VR`.
   - If `/data` is `/mnt/user/sysbackup`, the default destinations
     (`/data/Rando/norating`, `/data/#VR`) are already right.
   - If it's another share, set the two destinations in Settings → Seedbox jobs
     (step 9) to the container paths of
     `/mnt/user/sysbackup/Rando/norating` and `/mnt/user/sysbackup/#VR`. If those
     aren't under `/data` at all, add a mount for them first.

## In the Peaks container settings (Unraid → Docker → Peaks → Edit)

4. **Seedbox pipeline state:** `/mnt/user/appdata/seedbox-pull` → `/seedbox`, now
   **Read/Write** (was read-only).
5. **rclone config and pull filter: copy them, don't mount them.** Unraid's
   container editor only offers `/mnt/user/...` paths, so copy both files into
   the seedbox-pull folder. Peaks already sees that folder (step 4) and looks for
   them in its `rclone/` subfolder. On the **Unraid terminal**:

   ```sh
   mkdir -p /mnt/user/appdata/seedbox-pull/rclone
   cp /boot/config/rclone/rclone.conf /mnt/user/appdata/seedbox-pull/rclone/rclone.conf
   cp /boot/config/rclone-filters.txt /mnt/user/appdata/seedbox-pull/rclone/filters.txt
   chmod 600 /mnt/user/appdata/seedbox-pull/rclone/rclone.conf
   ```

   - **The seedbox password stays in that rclone.conf copy**, as it does in the
     original (rclone stores it obscured, not encrypted). Keep the `appdata` share
     unexported over SMB, or at least private.
   - **If you change the rclone remote later** (`rclone config` on Unraid), run the
     first `cp` again. Same for the filter file.
   - **A missing file is reported plainly:** the Seedbox page shows "no rclone.conf"
     or "no pull filter", and the pull won't start until the file is there.
6. *(Nothing else for rclone: no extra mounts needed.)*
7. **Whisparr config (read-only, advanced):** `/mnt/user/appdata/whisparr/config.xml`
   → `/whisparr/config.xml`, and `PEAKS_WHISPARR_URL` = `http://192.168.1.2:7171`.
   - If your template doesn't show these new fields, add them by hand: Add another
     Path / Variable with exactly these values.
8. Apply. Peaks restarts.

## Switch over

9. In **Peaks → Settings → Seedbox jobs**, fill in:
   - **Seedbox:** `seedbox:/home/nucleargtz/downloads/qbittorrent`. That's the
     `REMOTE` from your pull script: `seedbox` is the remote's name in rclone.conf.
     The purge reads qBittorrent's save paths against this same folder.
   - **Where files go:** check both destinations (step 3).
   - Leave the schedules as they are: pull every 30 min, purge 05:00, report
     06:00. Leave the purge on **"Dry run for 3 days, then live"**.
   - **Don't** tick "I've disabled the Unraid scripts" yet. Save.
10. In **Unraid → Settings → User Scripts**, set **seedbox-pull**,
    **seedbox-purge-pulled** and **seedbox-report** to **Schedule Disabled**.
11. **Make sure no pull is still running.** Wait for the current one to finish:
    `pull.log` stops growing, or `ls /tmp/seedbox-pull.lock` plus
    `fuser /tmp/seedbox-pull.lock` on the Unraid terminal shows nothing.
12. **Wait 45 minutes after the last script run** (or check its time on the Peaks
    Seedbox page). Until then Peaks treats that run as "a script is still on" and
    won't pull. That's intended.
13. Back in **Settings → Seedbox jobs**, tick **"I've disabled the Unraid scripts"**
    and save. The 3-day purge dry run starts now.

## Check it worked

14. **Peaks → Seedbox → Jobs Peaks runs → Pull new files → Run now.**
    - Expected: only files that arrived on the seedbox since the last script run.
      Often "nothing new".
    - It must **not** re-download things you already have. If the log shows old
      files being pulled again, press Stop and check the remote path (step 9): the
      category folders must be `1` and `vr` directly under it.
15. Open **Floor** (or `http://<peaks>/floor` on a spare screen) and watch the pull.
16. **Purge → Run now** (dry run):
    - read the job log in Activity: each torrent is listed as **KEEP (why)** or
      **WOULD DELETE (why)**;
    - every WOULD DELETE should be something you have on Jarvis.
17. Over the next 3 days, glance at the 05:00 dry runs the same way. After 3 days the
    purge goes live on its own; Peaks tells you on the Seedbox page. To keep
    checking, set the purge to **"Dry run only"**; to start earlier, set **"Live"**.

## Good to know

- **The seedbox fills up a little during the dry run:** nothing is purged for 3 days,
  so torrents stay on the seedbox that long.
- **The first daily report counts only what Peaks pulled.** Files the scripts pulled
  earlier aren't in its ledger, so its "came in" numbers start from the switchover.
- **Notifications:** Peaks doesn't push to Pushbullet. Failures and the report show
  on the Seedbox page, in Activity and in the phone app. The Unraid "Seedbox pull
  FAILED / recovered" pushes stop with the scripts.
- **Going back:** untick "I've disabled the Unraid scripts" in Peaks, then re-enable
  the three User Scripts. The files are in the same formats both ways. The only
  change is that purge runs now also appear in `runs.log`, as
  `<time>|purge|ok|<n>|<bytes>`; the old scripts ignore those lines.
