"""Teach Peaks: which ungraded scenes would teach the grade models most.

Grading in any order teaches slowly — most scenes are ones the model is already
sure about. This ranks ungraded scenes by how much an answer would help:

- **torn** — the model's top two grades are close (45% / 40%): your answer
  settles a boundary it hasn't learned;
- **new** — few or no grades yet from this studio or these performers: the
  records behind its suggestions are thin there;
- **spread** — at most SPREAD_STUDIO per studio in any window of 20, and nothing
  near-identical to a scene already picked, so one answer isn't spent twice.

For VR before its model exists there are no probabilities, so the ranking fills
the model's gaps instead: the grade it's short of (rejects first, by the
quality check's verdict) and studios with nothing graded yet.

Pure functions: rows in, ranked rows with a plain reason out.
"""

from __future__ import annotations

import numpy as np

SPREAD_STUDIO = 2
WINDOW = 20
DUP_COS = 0.97          # this similar (mean embedding) to a pick already chosen: skip it
W_NEW = 0.6             # novelty's weight against uncertainty (both 0..1)


def uncertainty(probs: dict | None) -> tuple[float, tuple[str, str] | None]:
    """1 − (top − second) — 1.0 is a coin toss between two grades — and the pair."""
    if not probs:
        return 0.0, None
    top = sorted(probs.items(), key=lambda kv: -kv[1])
    if len(top) < 2:
        return 0.0, None
    (a, pa), (b, pb) = top[0], top[1]
    return float(1.0 - (pa - pb)), (a, b)


def novelty(row: dict, studio_n: dict[str, int], perf_n: dict[str, int]) -> tuple[float, str | None]:
    """How little is known about this scene's studio / performers (1 = nothing)."""
    st = (row.get("studio") or "").strip()
    ns = studio_n.get(st.lower(), 0) if st else 0
    perfs = [str(p) for p in row.get("performer_ids") or []]
    nps = [perf_n.get(p, 0) for p in perfs]
    s_new = 1.0 / (1 + ns) if st else 0.0
    p_new = 1.0 / (1 + min(nps)) if nps else 0.0
    why = None
    if st and not ns:
        why = f"first {st} scene you'd grade"
    elif nps and min(nps) == 0:
        why = "a performer you haven't graded yet"
    return max(s_new, p_new), why


def grade_counts(graded: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    studio_n: dict[str, int] = {}
    perf_n: dict[str, int] = {}
    for r in graded:
        st = (r.get("studio") or "").strip().lower()
        if st:
            studio_n[st] = studio_n.get(st, 0) + 1
        for p in set(str(x) for x in r.get("performer_ids") or []):
            perf_n[p] = perf_n.get(p, 0) + 1
    return studio_n, perf_n


def rank(cands: list[dict], preds: dict[str, dict], graded: list[dict], names: dict[str, str],
         vecs: dict[str, np.ndarray] | None = None, limit: int | None = None) -> list[dict]:
    """Ungraded rows, most to learn from first, each with `teach` = {score, why}."""
    studio_n, perf_n = grade_counts(graded)
    scored = []
    for r in cands:
        p = preds.get(r["scene_id"]) or {}
        u, pair = uncertainty(p.get("probs"))
        nv, nwhy = novelty(r, studio_n, perf_n)
        why = None
        if pair and u >= 0.7:
            why = f"torn between {names.get(pair[0], pair[0])} and {names.get(pair[1], pair[1])}"
        why = why or nwhy or ("a guess it isn't sure of" if p else "no suggestion yet")
        scored.append((u + W_NEW * nv, r, why))
    scored.sort(key=lambda x: -x[0])
    return _spread([dict(r, teach={"score": round(s, 3), "why": w}) for s, r, w in scored], vecs, limit)


def _spread(rows: list[dict], vecs, limit) -> list[dict]:
    out, recent_studios, kept_vecs = [], [], []
    rest = []
    for r in rows:
        st = (r.get("studio") or "").strip().lower()
        window = recent_studios[-(WINDOW - 1):]
        if st and window.count(st) >= SPREAD_STUDIO:
            rest.append(r)
            continue
        v = (vecs or {}).get(r["scene_id"])
        if v is not None and kept_vecs:
            sims = np.asarray(kept_vecs) @ v
            if float(sims.max()) >= DUP_COS:
                rest.append(r)
                continue
        out.append(r)
        recent_studios.append(st)
        if v is not None:
            kept_vecs.append(v)
        if limit and len(out) >= limit:
            return out
    out += rest                                  # still listed — just after the spread-out ones
    return out[:limit] if limit else out


VERDICT_REJECTISH = {"below": 1.0, "borderline": 0.7, "unmeasured": 0.3, "meets": 0.1}


def rank_vr_gaps(cands: list[dict], vrq: dict[str, dict], graded: list[dict], needs: dict[str, int],
                 names: dict[str, str], limit: int | None = None) -> list[dict]:
    """VR before its model exists: the files that fill what it lacks soonest —
    likely rejects (by the quality check) when rejects are short, likely keepers
    when keepers are, and studios with nothing graded."""
    studio_n, _ = grade_counts(graded)
    want_rej = needs.get("reject", 0)
    want_keep = needs.get("exceptionnelle", 0) + needs.get("legendaire", 0)
    scored = []
    for r in cands:
        v = (vrq.get(r["scene_id"]) or {}).get("verdict", "unmeasured")
        rej = VERDICT_REJECTISH.get(v, 0.3)
        st = (r.get("studio") or "").strip()
        new = 1.0 if st and not studio_n.get(st.lower()) else 0.0
        s = want_rej * rej + want_keep * (1.0 - rej) + 3.0 * new
        if want_rej and v in ("below", "borderline"):
            why = "likely reject — the VR model needs rejects to learn from"
        elif want_keep and v == "meets":
            why = "likely keeper — the VR model needs more keepers"
        elif new:
            why = f"no VR from {st} graded yet"
        else:
            why = "fills a gap in the VR model"
        scored.append((s, r, why))
    scored.sort(key=lambda x: -x[0])
    return _spread([dict(r, teach={"score": round(s, 3), "why": w}) for s, r, w in scored], None, limit)
