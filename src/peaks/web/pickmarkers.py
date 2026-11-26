"""Peaks's picks as Stash markers: each scene's best moments on the timeline of
any player that reads Stash markers — HereSphere for VR, Stash's own for flat.

Opt-in (Settings → Playback & moments). Every embedded, non-rejected scene gets
its top PICKS moments as markers under their own tag ("Peaks pick"), apart from
your saved moments: Peaks learns your taste from those, never from these. A
store map remembers each scene's marker ids and the pick times they were made
from, so a scene is rewritten only when its picks move (more than MOVE s) and a
reject's markers are removed. Turning it off removes every marker with that tag
— and nothing else.
"""

from __future__ import annotations

import time

from ..atomic import settings_saver as _settings_saver

PICK_TAG = "Peaks pick"
PICKS = 3
MOVE = 10.0                 # seconds a pick may drift before its marker is rewritten
SPAN = 20.0                 # a marker's length when the pick has no span of its own
CHUNK = 200                 # scenes per look at the index


class PickMarkersMixin:
    def pick_markers_settings(self) -> dict:
        on = bool((self._settings().get("pick_markers") or {}).get("on", False))
        return {"on": on, "tag": PICK_TAG, "written": len(self._pm_state()) if on else 0}

    @_settings_saver
    def save_pick_markers(self, on) -> dict:
        from .inputs import as_bool

        s = dict(self._settings())
        s["pick_markers"] = {"on": as_bool(on, "on")}
        self._save_settings_dict(s)
        self._pm_dirty = True
        return self.pick_markers_settings()

    def _pm_state(self) -> dict[str, dict]:
        return self.store().map_all("pick_markers")

    def pick_markers_due(self) -> bool:
        """Something to write or remove — from memory only (an idle server touches nothing)."""
        on = self.pick_markers_settings()["on"]
        if not on:
            return bool(self.__dict__.get("_pm_dirty")) and bool(self._pm_state())   # switched off: remove
        return self.__dict__.get("_pm_dirty", True) and self.index_if_loaded() is not None

    def mark_pick_markers_dirty(self) -> None:
        """New embeds or a new taste model: the picks may have moved."""
        self._pm_dirty = True

    def run_pick_markers(self, job=None) -> dict:
        """One pass: write / move / remove markers so Stash shows each scene's
        current picks. Resumable; stops on cancel."""
        log = job.log if job else (lambda *_: None)
        if not self.pick_markers_settings()["on"]:
            n = self.remove_pick_markers(job)
            self._pm_dirty = False
            return {"removed": n}
        client, st = self.client(), self.store()
        tag_id = str(client.find_or_create_tag(PICK_TAG).id)
        state = self._pm_state()
        rows = self._catalogue_all()
        live = {r["scene_id"]: r for r in rows}
        # rejects, and scenes gone from the library, lose their markers
        drop = [sid for sid in state if sid not in live or live[sid]["tier"] == "rejected"]
        removed = written = moved = 0
        if drop:
            client.destroy_scene_markers([m for sid in drop for m in state[sid].get("ids") or []])
            for sid in drop:
                st.map_delete("pick_markers", sid)
            removed = len(drop)
        todo = [r["scene_id"] for r in rows if r["tier"] != "rejected"]
        if job is not None:
            job.progress = {"total": len(todo), "done": 0}
        for i in range(0, len(todo), CHUNK):
            if job is not None and job.cancelled:
                log("stopped — the rest waits for the next pass")
                return {"written": written, "moved": moved, "removed": removed, "stopped": True}
            part = todo[i:i + CHUNK]
            strips = self._scene_moment_strips(part, n=PICKS, min_gap=20.0)
            for sid in part:
                picks = [p for p in strips.get(sid) or [] if p.get("score") is not None]
                if not picks:
                    continue                    # no taste yet: evenly spread frames aren't picks
                picks = sorted(picks, key=lambda p: -p["score"])[:PICKS]
                ts = [round(float(p["t"]), 2) for p in picks]
                old = state.get(sid)
                if old and len(old.get("ts") or []) == len(ts) and all(
                        abs(a - b) <= MOVE for a, b in zip(old["ts"], ts)):
                    continue
                if old and old.get("ids"):
                    client.destroy_scene_markers(old["ids"])
                    moved += 1
                ids = []
                for k, (p, t) in enumerate(zip(picks, ts), 1):
                    end = float(p.get("end") or t + SPAN)
                    mk = client.create_scene_marker(scene_id=sid, seconds=t, primary_tag_id=tag_id,
                                                    title=f"Peaks pick {k}", end_seconds=end)
                    ids.append(str((mk or {}).get("id", "")))
                st.map_set("pick_markers", sid, {"ts": ts, "ids": ids, "at": time.time()})
                written += 0 if old else 1
            if job is not None:
                job.progress = {"total": len(todo), "done": min(len(todo), i + CHUNK)}
        self._pm_dirty = False
        log(f"pick markers: {written} scene(s) marked, {moved} moved, {removed} removed")
        return {"written": written, "moved": moved, "removed": removed}

    def remove_pick_markers(self, job=None) -> int:
        """Every marker under the "Peaks pick" tag — those Peaks remembers and any
        it doesn't (a lost state) — and only those."""
        client, st = self.client(), self.store()
        ids = {m for e in self._pm_state().values() for m in e.get("ids") or []}
        try:
            ids |= {str(m["marker_id"]) for m in client.iter_markers_by_tag(PICK_TAG)}
        except Exception:  # noqa: BLE001 — the remembered ones still go
            pass
        ids.discard("")
        if ids:
            client.destroy_scene_markers(sorted(ids))
        st.map_replace("pick_markers", {})
        if job is not None:
            job.log(f"removed {len(ids)} Peaks pick marker(s)")
        return len(ids)
