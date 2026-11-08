"""One scene's story: what Peaks suggested, what you decided, what changed in
Stash — and why — in one timeline, newest first.

Nothing new is recorded for it: it reads what Peaks already keeps — the action
log (grades with their before/after and source, tags, moves, performers,
upgrades, deletes), your rulings and "keep as is" answers, pick answers,
thumbs, Upgrades and Orphans decisions — and adds "now": Peaks' current verdict
and suggestion, whether Settle would ask about it, and who its faces are.
"""

from __future__ import annotations

import time


WHO = {"you": "You", "peaks": "Peaks", "stash": "Stash"}


def _epoch(ts: str | float | None) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


class TimelineMixin:
    def _tl_actions(self, sid: str, fp: str | None) -> list[dict]:
        """Action-log lines about this scene (by id, or by file fingerprint after a re-scan)."""
        return self.action_log().about(sid, fp)

    def _tl_event(self, e: dict, names: dict) -> dict | None:
        a, src = e.get("action"), str(e.get("source") or "")
        at = _epoch(e.get("ts"))
        auto = src.startswith("auto")

        def tier(x):
            return names.get(x, x) if x else "—"
        if a == "grade":
            b, af = e.get("before") or {}, e.get("after") or {}
            text = f"Graded {tier(b.get('tier'))} → {tier(af.get('tier'))}"
            why = []
            if src and not auto:
                why.append({"catalogue": "from the Catalogue", "bulk": "in a bulk grade", "review": "in Review",
                            "settle": "answering Settle"}.get(src, f"from {src}"))
            if auto:
                why.append(src.split(":", 1)[-1].strip())
            pk = e.get("peaks")
            if pk and pk.get("tier"):
                agree = pk["tier"] == (af.get("tier") or "").replace("rejected", "reject")
                why.append(f"Peaks thought {tier(pk['tier'])} ({round(100 * float(pk.get('conf') or 0))}% sure)"
                           + (" — you agreed" if agree else " — you overruled it"))
            return {"at": at, "who": "peaks" if auto else "you", "kind": "grade", "text": text, "why": "; ".join(why)}
        texts = {
            "keep": ("you", "Kept as is — answered the suggestion"),
            "tag-sync": ("peaks", "Tier tag repaired in Stash"),
            "moved": ("stash", "File moved (your renamer)"),
            "delete": ("you", "Deleted"),
            "duplicate": ("you", "Duplicate resolved"),
            "performers": ("you", "Performers changed"),
            "performer-created": ("you", "New performer created and added"),
            "upgrade-grab": ("you", "Upgrade grabbed"),
            "discover-grab": ("you", "Grabbed from Discover"),
            "restore": ("you", "Grade restored (undo / backup)"),
            "file-copy": ("peaks", "Same-file copy removed"),
            "criteria": ("you", "Told Peaks why"),
        }
        who, text = texts.get(a, ("peaks", a or "change"))
        return {"at": at, "who": who, "kind": a, "text": text, "why": e.get("detail") or ""}

    def scene_story(self, scene_id: str) -> dict:
        sid = str(scene_id)
        row = next((r for r in self._catalogue_all() if r["scene_id"] == sid), None)
        if row is None:
            raise LookupError(f"scene {sid} isn't in the library")
        names = {**self.tier_display_names(), "reject": self.tier_display_names().get("rejected", "Reject")}
        ev: list[dict] = []
        for e in self._tl_actions(sid, row.get("fingerprint")):
            x = self._tl_event(e, names)
            if x:
                ev.append(x)
        # your answers that don't change Stash
        try:
            ru = self.rulings().get(sid)
            if ru:
                ev.append({"at": ru.get("at"), "who": "you", "kind": "ruling",
                           "text": f"Keep mine — your {names.get(ru['grade'], ru['grade'])} stands",
                           "why": "Settle won't ask again unless you regrade it"})
        except Exception:  # noqa: BLE001
            pass
        try:
            for x in self._mc_read("pick_feedback.jsonl"):
                if x.get("scene_id") == sid:
                    ev.append({"at": x.get("at"), "who": "you", "kind": "pick",
                               "text": f"{'👍' if int(x.get('label', 0)) else '👎'} on Peaks' pick at {_mmss(x.get('t'))}",
                               "why": "teaches the moment model what you like"})
            for x in self._mc_read("moment_checks.jsonl"):
                if x.get("scene_id") == sid:
                    r = x.get("rank")
                    ev.append({"at": x.get("at"), "who": "you", "kind": "save",
                               "text": f"Saved a moment at {_mmss(x.get('t'))}",
                               "why": f"Peaks' pick #{r} had it" if r else "Peaks' picks had missed it"})
        except Exception:  # noqa: BLE001
            pass
        try:
            for lab in self._label_store().for_profile(self.cfg.markers.tag_name):
                if str(getattr(lab, "scene_id", "") or "") == sid and getattr(lab, "ts", 0):
                    ev.append({"at": lab.ts, "who": "you", "kind": "thumb",
                               "text": f"{'👍' if lab.label == 1 else '👎'} at {_mmss(lab.time)}",
                               "why": "part of your taste" if getattr(lab, "source", "explicit") == "explicit"
                               else "from watching (counts for less)"})
        except Exception:  # noqa: BLE001
            pass
        try:
            u = (self._upg_load().get("scenes") or {}).get(sid) or {}
            if u.get("searched_at"):
                n = len(u.get("found") or [])
                ev.append({"at": u["searched_at"], "who": "peaks", "kind": "upgrade",
                           "text": f"Searched for a better copy — {n} healthy upgrade{'s' if n != 1 else ''}" if n
                           else "Searched for a better copy — nothing better yet", "why": ""})
            if u.get("state") == "dismissed":
                ev.append({"at": u.get("dismissed_at") or u.get("searched_at"), "who": "you", "kind": "upgrade",
                           "text": "Upgrades: not interested", "why": ""})
        except Exception:  # noqa: BLE001
            pass
        try:
            no = self._orph_no().get(sid) or []
            if no:
                pn = {p["id"]: p["name"] for p in self._stash_performers()}
                ev.append({"at": None, "who": "you", "kind": "performers",
                           "text": "Not them: " + ", ".join(pn.get(p, p) for p in no),
                           "why": "Peaks won't suggest them for this scene again"})
        except Exception:  # noqa: BLE001
            pass
        ev.sort(key=lambda x: -(x.get("at") or 0))
        return {"scene_id": sid, "title": row.get("title") or f"scene {sid}", "tier": row["tier"],
                "now": self._tl_now(row, names), "events": ev}

    def _tl_now(self, row: dict, names: dict) -> dict:
        """What Peaks thinks of it today."""
        from ..tier_model import train_class

        sid = row["scene_id"]
        out: dict = {"tier": names.get(row["tier"], row["tier"])}
        try:
            p = self._tier_predictions(self._catalogue_all()).get(sid)
        except Exception:  # noqa: BLE001
            p = None
        if p:
            out["peaks"] = {"tier": names.get(p["tier"], p["tier"]), "conf": round(float(p.get("conf") or 0), 3),
                            "from": "performer and studio record only (not embedded yet)" if p.get("from") == "who"
                            else "what the scene looks like, plus its performers, studio and picture"}
            graded = train_class(row["tier"])
            if graded and p.get("from") != "who":
                ru = (self.rulings() or {}).get(sid)
                if p["tier"] == graded:
                    out["settle"] = "agrees with your grade"
                elif ru and ru.get("grade") == row["tier"]:
                    out["settle"] = "disagrees, but you ruled: your grade stands"
                else:
                    out["settle"] = "disagrees — Settle the Library asks about it"
        try:
            sig = (self._scene_signals([row]) or {}).get(sid)
            rec = self._recommend(row, p, sig, names)
            if rec:
                out["suggests"] = {"grade": names.get(rec["grade"], rec["grade"]), "why": rec.get("why"),
                                   "detail": rec.get("detail")}
        except Exception:  # noqa: BLE001
            pass
        try:
            fm = self.face_matches(sid)
            if fm.get("matches"):
                out["faces"] = [{"name": m["name"], "credited": m["credited"], "strength": m["strength"]}
                                for m in fm["matches"]]
        except Exception:  # noqa: BLE001
            pass
        return out


def _mmss(t) -> str:
    try:
        t = int(float(t))
    except (TypeError, ValueError):
        return "?"
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"

