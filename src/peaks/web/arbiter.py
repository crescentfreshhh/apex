"""The Arbiter: where Peaks agrees and disagrees with your grades, and why.

Every graded scene has an honest verdict — from the fold model that never saw
its grade (see `tier_model.fit_folds`) — so a disagreement is a real second
opinion, not your grade echoed back. For one scene, the argument is the model's
log-odds gap between its verdict and your grade, split exactly into what it
looked at (look · file · performers/studio · picture), with each part made
concrete: the named file/picture facts against your tiers, the performers'
records, the graded scenes it looks most like, and the moments that pull each
way. CLIP words appear only when they genuinely separate the two tiers in your
library (an effect size gate) — otherwise they're noise and stay hidden.

Library-wide, the same verdicts give the agreement grid, calibration, where the
disagreements cluster (studio, performer, resolution, picture band), and what
separates your tiers at all.
"""

from __future__ import annotations

import numpy as np

WORD_GATE = 0.3          # |effect size| a term needs between two tiers to be shown
WORD_MIN_N = 5           # graded scenes per tier before words are tried at all
CLUSTER_MIN_N = 5        # scenes sharing a trait before its disagreement rate counts
GROUP_NAMES = {"look": "Look", "file": "File", "who": "Performers / studio", "picture": "Picture"}


def _tier(c: str) -> str:
    return "rejected" if c == "reject" else c


class ArbiterMixin:
    # --- the honest verdict for one scene --------------------------------------------

    def _arb_model_for(self, sid: str):
        """(model to explain with, honest?) — the fold model that never saw this
        scene's grade when there is one, else the full model."""
        m = self._tier_model_state()["model"]
        o = (getattr(m, "oof", None) or {}).get(sid)
        folds = getattr(m, "folds", None)
        if o is not None and folds:
            return folds[o["fold"]], True
        return m, False

    def tier_explain(self, scene_id: str) -> dict:
        from ..tier_model import live_ordinal, picture_features, train_class, who_of

        sid = str(scene_id)
        st = self._tier_model_state()
        if st["model"] is None:
            return {"scene_id": sid, "ok": False, "reason": "the tier model isn't trained yet"}
        rows = self._catalogue_all()
        by_id = {r["scene_id"]: r for r in rows}
        row = by_id.get(sid)
        if row is None:
            raise LookupError(f"scene {sid} not found")
        pred = self._tier_predictions(rows).get(sid)
        feats = self._scene_features([row]).get(sid)
        if not pred or feats is None:
            return {"scene_id": sid, "ok": False, "reason": "not embedded yet — nothing to argue from"}
        m, honest = self._arb_model_for(sid)
        names = self.tier_display_names()
        mine = train_class(row["tier"])
        probs = pred["probs"]
        order = sorted(probs, key=lambda c: -probs[c])
        a = pred["tier"]
        b = mine if mine and mine != a else (order[1] if len(order) > 1 else a)
        if b not in m.classes_ or a not in m.classes_ or a == b:
            return {"scene_id": sid, "ok": False, "reason": "only one tier to compare"}
        pics = self._picture_entries()
        ex = m.explain(a, b, feats[0], feats[1], who=who_of(row),
                       own=None if honest else mine, pic=picture_features(pics.get(sid)))
        facts = self._arb_facts(row, a, b, rows, pics, names)
        for g in ex["groups"]:
            g["label"] = GROUP_NAMES.get(g["group"], g["group"])
            g["facts"] = facts.get(g["group"], [])
        live = live_ordinal(m.classes_)
        agree = None
        if mine:
            agree = "agree" if a == mine else ("higher" if live.get(a, 0) > live.get(mine, 0) else "lower")
        return {
            "scene_id": sid, "ok": True, "honest": honest,
            "yours": mine, "verdict": a, "against": b, "agree": agree,
            "conf": pred["conf"], "probs": probs,
            "names": {c: names.get(_tier(c), c) for c in m.classes_},
            **ex,
            "lookalikes": self._arb_lookalikes(sid, (a, b), rows),
            "moments": self._arb_moments(m, sid, a, b),
            "words": self._arb_words(sid, a, b, rows),
            "summary": self._arb_line(ex, a, b, names),
        }

    @staticmethod
    def _arb_line(ex: dict, a: str, b: str, names: dict) -> str:
        """One line for Today: what carried the verdict, and what argued back."""
        gs = sorted(ex["groups"], key=lambda g: -g["push"])
        pro = [g for g in gs if g["push"] > 0.15]
        con = [g for g in gs if g["push"] < -0.15]
        if not pro:
            return "a close call — nothing pushes strongly either way"
        line = "mostly " + " + ".join(GROUP_NAMES[g["group"]].lower() for g in pro[:2])
        if con:
            line += f"; {GROUP_NAMES[con[-1]['group']].lower()} argues for {names.get(_tier(b), b)}"
        return line

    # --- the concrete facts behind the file / who / picture pushes --------------------

    def _arb_facts(self, row, a, b, rows, pics, names) -> dict:
        def tier_rows(c):
            return [r for r in rows if r["tier"] == _tier(c)]
        A, B = tier_rows(a), tier_rows(b)
        na, nb = names.get(_tier(a), a), names.get(_tier(b), b)
        out: dict[str, list[str]] = {"file": [], "who": [], "picture": []}

        def median(vals):
            v = sorted(x for x in vals if x is not None)
            return v[len(v) // 2] if v else None

        q = row["quality"]
        if q.get("mbps"):
            ma = median([r["quality"].get("mbps") for r in A])
            mb = median([r["quality"].get("mbps") for r in B])
            if ma is not None and mb is not None:
                out["file"].append(f"{q['mbps']:g} Mbps — your {na} median {ma:g}, your {nb} median {mb:g}")
        if q.get("res"):
            sa = sum(1 for r in A if r["quality"].get("res") == q["res"])
            sb = sum(1 for r in B if r["quality"].get("res") == q["res"])
            if A and B:
                out["file"].append(f"{q['res']} — {round(100 * sa / len(A))}% of your {na} are {q['res']}, "
                                   f"{round(100 * sb / len(B))}% of your {nb}")
        p = pics.get(row["scene_id"]) or {}
        if p.get("score") is not None:
            ma = median([(pics.get(r["scene_id"]) or {}).get("score") for r in A])
            mb = median([(pics.get(r["scene_id"]) or {}).get("score") for r in B])
            out["picture"].append(f"picture {p.get('band') or ''} {p['score']}".strip()
                                  + (f" — your {na} median {ma}, {nb} median {mb}" if ma is not None and mb is not None else ""))
        if p.get("upscaled"):
            out["picture"].append("looks upscaled")
        try:
            who = self._who_reasons(row, self._who_records(rows), names)
            out["who"] = [ln.get("text") or ln.get("why") or str(ln) for ln in (who.get("lines") or [])][:4] \
                if isinstance(who, dict) else []
        except Exception:  # noqa: BLE001 — wording is extra
            pass
        return out

    # --- the look, by example ----------------------------------------------------------

    def _arb_lookalikes(self, sid: str, tiers, rows, k: int = 3) -> dict:
        means = self._scene_means()
        me = means.get(sid)
        out = {}
        if me is None:
            return out
        for c in tiers:
            cand = []
            for r in rows:
                if r["tier"] != _tier(c) or r["scene_id"] == sid:
                    continue
                v = means.get(r["scene_id"])
                if v is not None:
                    cand.append((float(me @ v), r))
            cand.sort(key=lambda x: -x[0])
            out[c] = [{"scene_id": r["scene_id"], "title": r.get("title") or f"scene {r['scene_id']}",
                       "sim": round(s, 3), "cover": f"/api/scene/{r['scene_id']}/cover"} for s, r in cand[:k]]
        return out

    # --- the moments that pull each way --------------------------------------------------

    def _arb_moments(self, m, sid: str, a: str, b: str, k: int = 2) -> dict:
        idx = self.index(self._model_name())
        key = next((kk for kk, mm in idx.key_meta.items() if str(mm.get("scene_id")) == sid), None)
        span = idx._key_rows.get(key) if key else None
        if not span:
            return {}
        s, e = span
        push = m.frame_push(a, b, idx.rows(s, e))
        if not len(push):
            return {}
        times = idx.times[s:e]

        def pick(order):
            got = []
            for i in order:
                t = float(times[i])
                if all(abs(t - g["t"]) >= 15 for g in got):
                    got.append({"t": round(t, 2), "push": round(float(push[i]), 3),
                                "thumb": f"/api/frame?key={key}&t={t:g}"})
                if len(got) >= k:
                    break
            return got
        return {a: pick(np.argsort(-push)), b: pick(np.argsort(push))}

    # --- words, only when they mean something ---------------------------------------------

    def _arb_word_table(self, rows):
        """(terms, scene_id → row index, z-scored scene × term matrix) from CLIP, or None."""
        try:
            if not self.has_clip_index():
                return None
            idx = self.index(self._clip_name())
        except Exception:  # noqa: BLE001
            return None
        cached = self.__dict__.get("_arb_words_cache")
        if cached and cached[0] == id(idx):
            return cached[1]
        labels, mat = self._vocab_matrix()
        sids, vecs = [], []
        for key, (s, e) in idx._key_rows.items():
            sid = str((idx.key_meta.get(key) or {}).get("scene_id") or "")
            if sid and e > s:
                sids.append(sid)
                vecs.append(self._unit(idx.rows(s, e).mean(axis=0)))
        if not vecs:
            return None
        S = np.stack(vecs) @ mat.T
        Z = (S - S.mean(axis=0)) / np.maximum(S.std(axis=0), 1e-6)
        out = (labels, {s: i for i, s in enumerate(sids)}, Z.astype(np.float32))
        self._arb_words_cache = (id(idx), out)
        return out

    def _arb_separating(self, a: str, b: str, rows):
        """[(term, effect size d, col)] terms that separate tier a from tier b, |d| ≥ gate."""
        t = self._arb_word_table(rows)
        if t is None:
            return [], None
        labels, pos, Z = t
        ia = [pos[r["scene_id"]] for r in rows if r["tier"] == _tier(a) and r["scene_id"] in pos]
        ib = [pos[r["scene_id"]] for r in rows if r["tier"] == _tier(b) and r["scene_id"] in pos]
        if len(ia) < WORD_MIN_N or len(ib) < WORD_MIN_N:
            return [], t
        A, B = Z[ia], Z[ib]
        pooled = np.sqrt((A.var(axis=0) + B.var(axis=0)) / 2) + 1e-6
        d = (A.mean(axis=0) - B.mean(axis=0)) / pooled
        keep = [(labels[j], float(d[j]), j) for j in np.argsort(-np.abs(d)) if abs(d[j]) >= WORD_GATE]
        return keep, t

    def _arb_words(self, sid: str, a: str, b: str, rows, k: int = 3) -> dict:
        sep, t = self._arb_separating(a, b, rows)
        if not sep or t is None or sid not in t[1]:
            return {}
        z = t[2][t[1][sid]]
        lean_a = [(w, d, float(z[j])) for w, d, j in sep if d > 0 and z[j] > 0.5]
        lean_b = [(w, d, float(z[j])) for w, d, j in sep if d < 0 and z[j] > 0.5]
        fmt = lambda xs: [{"term": w, "d": round(abs(d), 2), "z": round(zz, 2)}  # noqa: E731
                          for w, d, zz in sorted(xs, key=lambda x: -abs(x[1]) * x[2])[:k]]
        out = {a: fmt(lean_a), b: fmt(lean_b)}
        return out if out[a] or out[b] else {}

    # --- the library-wide picture --------------------------------------------------------

    def arbiter_summary(self) -> dict:
        from ..tier_model import live_ordinal, train_class

        st = self._tier_model_state()
        m = st["model"]
        oof = getattr(m, "oof", None) or {}
        if m is None or not oof:
            return {"ok": False, "reason": "train the tier model (Catalogue → ⋯ → Train) to get honest verdicts"}
        rows = self._catalogue_all()
        by_id = {r["scene_id"]: r for r in rows}
        names = self.tier_display_names()
        classes = list(m.classes_)
        live = live_ordinal(classes)
        judged = []
        for sid, o in oof.items():
            r = by_id.get(sid)
            g = train_class(r["tier"]) if r else None
            if r is None or g not in live or o["tier"] not in live:
                continue
            judged.append((r, g, o))
        if not judged:
            return {"ok": False, "reason": "no graded scenes left from the last training — retrain"}
        grid = {g: {v: 0 for v in classes} for g in classes}
        for _, g, o in judged:
            grid[g][o["tier"]] += 1
        agree = sum(1 for _, g, o in judged if g == o["tier"])
        per_tier = {g: {"n": sum(grid[g].values()), "agree": grid[g][g]} for g in classes}
        # calibration: how often the verdict is right at each confidence
        bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        calib = []
        for lo, hi in bins:
            xs = [g == o["tier"] for _, g, o in judged if lo <= o["conf"] < hi]
            if xs:
                calib.append({"lo": lo, "hi": min(hi, 1.0), "n": len(xs), "right": round(sum(xs) / len(xs), 3)})
        return {
            "ok": True, "n": len(judged), "agree": round(agree / len(judged), 3),
            "classes": classes, "names": {c: names.get(_tier(c), c) for c in classes},
            "grid": grid, "per_tier": per_tier, "calibration": calib,
            "clusters": self._arb_clusters(judged, live),
            "drivers": self._arb_drivers(m, rows),
            "words": self._arb_tier_words(rows, classes, names),
            "trained_at": (st.get("report") or {}).get("trained_at"),
        }

    def _arb_clusters(self, judged, live, k: int = 8) -> list[dict]:
        """Traits where your grades and Peaks' verdicts part ways in one direction."""
        pics = self._picture_entries()
        groups: dict[tuple[str, str], list[int]] = {}
        for r, g, o in judged:
            d = live[o["tier"]] - live[g]
            traits = [("studio", r.get("studio") or "")]
            traits += [("performer", p) for p in r.get("performers") or []]
            traits.append(("resolution", r["quality"].get("res") or ""))
            band = (pics.get(r["scene_id"]) or {}).get("band")
            traits.append(("picture", band or ""))
            for t in traits:
                if t[1]:
                    groups.setdefault(t, []).append(d)
        base = sum(1 for _, g, o in judged if g != o["tier"]) / len(judged)
        out = []
        for (kind, val), ds in groups.items():
            if len(ds) < CLUSTER_MIN_N:
                continue
            up, down = sum(1 for d in ds if d > 0), sum(1 for d in ds if d < 0)
            dis = up + down
            if dis < 2 or dis / len(ds) < max(2 * base, 0.15):   # clearly above the usual disagreement
                continue
            way = "higher" if up >= down else "lower"
            same = max(up, down)
            out.append({"kind": kind, "value": val, "n": len(ds), "disagree": dis, "way": way,
                        "same_way": same, "rate": round(dis / len(ds), 3),
                        "score": round(same * (dis / len(ds) - base), 3)})
        out.sort(key=lambda c: -c["score"])
        return out[:k]

    def _arb_drivers(self, m, rows) -> dict:
        """Average push of each group on the Légendaire-vs-Exceptionnelle split
        across your graded keepers — what actually separates the two."""
        from ..tier_model import picture_features, who_of

        cached = self.__dict__.get("_arb_drivers_cache")
        if cached and cached[0] == id(m):
            return cached[1]
        a, b = "legendaire", "exceptionnelle"
        out: dict = {}
        if a in m.classes_ and b in m.classes_:
            keep = [r for r in rows if r["tier"] in (a, b)]
            feats = self._scene_features(keep)
            pics = self._picture_entries()
            tot: dict[str, float] = {}
            n = 0
            for r in keep:
                f = feats.get(r["scene_id"])
                if f is None:
                    continue
                ex = m.explain(a, b, f[0], f[1], who=who_of(r), own=r["tier"],
                               pic=picture_features(pics.get(r["scene_id"])))
                for g in ex["groups"]:
                    tot[g["group"]] = tot.get(g["group"], 0.0) + abs(g["push"])
                n += 1
            if n:
                s = sum(tot.values()) or 1.0
                out = {"a": a, "b": b, "n": n,
                       "groups": [{"group": g, "label": GROUP_NAMES.get(g, g), "share": round(v / s, 3)}
                                  for g, v in sorted(tot.items(), key=lambda x: -x[1])]}
        self._arb_drivers_cache = (id(m), out)
        return out

    def _arb_tier_words(self, rows, classes, names) -> list[dict]:
        a, b = "legendaire", "exceptionnelle"
        if a not in classes or b not in classes:
            return []
        sep, _ = self._arb_separating(a, b, rows)
        return [{"term": w, "d": round(d, 2), "favours": names.get(a if d > 0 else b, a if d > 0 else b)}
                for w, d, _ in sep[:10]]

    # --- the grid's drill-down ----------------------------------------------------------

    def arbiter_cell(self, pool: list[dict], preds: dict, cell: str) -> list[dict]:
        """Scenes you graded `g` that Peaks honestly calls `v` ("g>v")."""
        from ..tier_model import train_class

        g, _, v = (cell or "").partition(">")
        out = [r for r in pool if train_class(r["tier"]) == g and (preds.get(r["scene_id"]) or {}).get("honest")
               and preds[r["scene_id"]]["tier"] == v]
        return sorted(out, key=lambda r: -preds[r["scene_id"]]["conf"])
