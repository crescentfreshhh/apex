"""Answered suggestions stay answered — until there's new evidence.

The Catalogue's suggestion lists (promotion candidates, second look, saved →
Légendaire, trim / passed over / Reject) are recomputed from scratch, so a
scene you answered by leaving it where it was used to come straight back. A
verdict remembers your answer: every grade you make (and "Keep as is", which
writes nothing to Stash) records {tier, when}; the first time the lists are
computed afterwards, the evidence the scene had is stored with it.

An answered scene stays out of a list unless something genuinely new happens:
new saved moments (saved → Légendaire), the tier model's guess moving a further
tier up (promotion) or down (second look), or the megaboard showing it 8+ more
times without a reaction (trim / passed / Reject). Retraining alone isn't new
evidence. A tier changed outside Peaks voids the verdict.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

USER_SOURCES = ("catalogue", "bulk", "keep")      # automatic changes aren't your verdict
MORE_SHOWINGS = 8                                  # the megaboard's "passed" threshold
QUESTIONS = ("saved", "promote", "second", "trim", "passed", "reject")


class VerdictMixin:
    # --- state -------------------------------------------------------------------

    def _vd_lock(self) -> threading.RLock:
        return self.__dict__.setdefault("_vd_lk", threading.RLock())

    def _vd_path(self) -> Path:
        return self._state_dir() / "verdicts.json"

    def _vd(self) -> dict:
        d = self.__dict__.get("_vd_mem")
        if d is None:
            try:
                d = json.loads(self._vd_path().read_text())
            except (OSError, ValueError):
                d = {}
            self._vd_mem = d
        return d

    def _vd_save(self) -> None:
        p = self._vd_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._vd()))
        os.replace(tmp, p)

    # --- recording ------------------------------------------------------------------

    def record_verdict(self, sid: str, tier: str, source: str = "catalogue") -> None:
        if source not in USER_SOURCES:
            return
        sid = str(sid)
        with self._vd_lock():
            d = self._vd()
            prev = d.get(sid)
            if prev:
                prev = {k: v for k, v in prev.items() if k != "prev"}   # one level of undo
            d[sid] = {"tier": tier, "at": time.time(), "src": source, "ev": None, "prev": prev}
            self._vd_save()

    def undo_verdict(self, sid: str) -> None:
        """An undone grade isn't an answer: the previous verdict (if any) is back."""
        sid = str(sid)
        with self._vd_lock():
            d = self._vd()
            v = d.pop(sid, None)
            if v and v.get("prev"):
                d[sid] = v["prev"]
            self._vd_save()

    def keep_as_is(self, scene_id: str) -> dict:
        """'Keep as is': answer the suggestion at the current tier — nothing is
        written to Stash (no renamer run, nothing moves)."""
        sid = str(scene_id)
        fresh = self._meta_client().scene_details([sid]).get(sid)
        if not fresh:
            raise LookupError(f"scene {sid} not found in Stash")
        row = self._cat_put_row(sid, fresh)
        self.record_verdict(sid, row["tier"], source="keep")
        # your answer is newer than the megaboard's evidence (kept for an undo)
        self.__dict__.setdefault("_expo_undo", {})[sid] = self.exposure().reset(sid)
        self._log_scene("keep", row, detail=f"kept as {row['tier']} — suggestion answered")
        return {"scene": row}

    def undo_keep(self, scene_id: str) -> dict:
        sid = str(scene_id)
        self.undo_verdict(sid)
        old = self.__dict__.get("_expo_undo", {}).pop(sid, None)
        if old is not None:
            self.exposure().restore(sid, old)
        return {"scene": self._cat_update_row(sid)}

    # --- the lists ask: answered already? ----------------------------------------------

    def _answered(self, question: str, row: dict, pred: dict | None, sig: dict | None) -> bool:
        """Has `question` been answered for this scene, with nothing new since?"""
        from ..tier_model import ORDINAL

        v = self._vd().get(str(row["scene_id"]))
        if not v:
            return False
        if v["tier"] != row["tier"]:                   # changed since (e.g. in Stash): void
            return False
        sig = sig or {}
        guess = (pred or {}).get("tier")
        if v.get("ev") is None:                        # the evidence you decided on
            with self._vd_lock():
                v["ev"] = {"saves": int(sig.get("saves") or 0), "pred": guess,
                           "showings": float(sig.get("showings") or 0)}
                self._vd_save()
            return True
        ev = v["ev"]

        def rank(t):
            return ORDINAL.get(t, ORDINAL.get(row["tier"], 0)) if t else ORDINAL.get(row["tier"], 0)

        if question == "saved":
            return int(sig.get("saves") or 0) <= ev["saves"]
        if question == "promote":
            return not (guess and rank(guess) > rank(ev["pred"]))
        if question == "second":
            return not (guess and rank(guess) < rank(ev["pred"]))
        if question in ("trim", "passed", "reject"):
            return not (sig.get("passed") and float(sig.get("showings") or 0) >= ev["showings"] + MORE_SHOWINGS)
        return False

    # --- settings ------------------------------------------------------------------------

    def verdict_status(self) -> dict:
        return {"answered": len(self._vd())}

    def clear_verdicts(self) -> dict:
        with self._vd_lock():
            self._vd().clear()
            self._vd_save()
        return self.verdict_status()
