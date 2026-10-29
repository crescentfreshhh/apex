"""Concepts: Peaks learns your words.

You name a concept in plain words ("close framing", "slow build"), mark a few
frames yes/no, and Peaks learns it: a linear probe on the frame embeddings of
the active index, trained on your marks. Because a probe is a direction in the
same space the search index already uses, finding a concept's frames is just
the index's own fast search — and every frame's probability is one dot product.

Concepts then turn up where Peaks explains itself (the Arbiter, gated the same
honest way as the CLIP words: only when a concept really separates two tiers),
as Catalogue filters and sorts, and in search.

State: state/concepts.json (the registry), state/concept_labels.json (your
marks — kept apart from your taste labels so they never mix), and
state/concepts/<slug>.npz (each probe).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

MIN_EACH = 3              # yes and no marks each before a concept is trained
SEED_N = 12               # frames offered per teaching round


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s[:40]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reliability(acc: float | None, n: int) -> str:
    if acc is None:
        return "needs more examples"
    if acc >= 0.85 and n >= 12:
        return "taught well"
    if acc >= 0.7:
        return "getting there"
    return "needs more examples"


class ConceptsMixin:
    # --- storage -----------------------------------------------------------------------

    def _cx_lock(self) -> threading.RLock:
        return self.__dict__.setdefault("_cx_lk", threading.RLock())

    def _cx_reg_path(self) -> Path:
        return self._state_dir() / "concepts.json"

    def _cx_dir(self) -> Path:
        return self._state_dir() / "concepts"

    def _cx_labels_path(self) -> Path:
        return self._state_dir() / "concept_labels.json"

    def _cx_reg(self) -> dict:
        try:
            d = json.loads(self._cx_reg_path().read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _cx_save_reg(self, d: dict) -> None:
        from .. import atomic

        p = self._cx_reg_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(p, d, indent=1)

    def _cx_store(self):
        from ..labels import LabelStore

        return LabelStore(self._cx_labels_path())

    def _cx_slug_of(self, name_or_slug: str) -> str:
        reg = self._cx_reg()
        s = _slug(name_or_slug)
        if s in reg:
            return s
        for k, v in reg.items():
            if v.get("name", "").strip().lower() == (name_or_slug or "").strip().lower():
                return k
        raise LookupError(f"no concept called “{name_or_slug}”")

    # --- the list ------------------------------------------------------------------------

    def concepts(self) -> list[dict]:
        """Visual concepts, then your named sound / motion concepts (kind tells them apart)."""
        out = self._visual_concepts()
        try:
            out += self.sense_concepts()
        except Exception:  # noqa: BLE001 — the senses are extra
            pass
        return out

    def _visual_concepts(self) -> list[dict]:
        reg = self._cx_reg()
        store = self._cx_store()
        out = []
        for slug, v in sorted(reg.items(), key=lambda kv: kv[1].get("created", 0)):
            pos, neg = store.counts(slug)
            rep = v.get("report") or {}
            out.append({"slug": slug, "name": v.get("name", slug), "kind": "visual", "yes": pos, "no": neg,
                        "trained": bool(rep.get("trained")), "accuracy": rep.get("accuracy"),
                        "reliability": rep.get("reliability") or reliability(None, pos + neg),
                        "trained_at": rep.get("at")})
        return out

    def concept_create(self, name: str) -> dict:
        name = (name or "").strip()[:60]
        slug = _slug(name)
        if not slug:
            raise ValueError("give the concept a name — a few plain words")
        with self._cx_lock():
            reg = self._cx_reg()
            if slug not in reg:
                reg[slug] = {"name": name, "created": time.time(), "report": None}
                self._cx_save_reg(reg)
        return next(c for c in self._visual_concepts() if c["slug"] == slug)

    def concept_delete(self, name_or_slug: str) -> dict:
        if (name_or_slug or "").split(":", 1)[0] in ("sound", "motion"):
            return self.sense_concept_delete(name_or_slug)
        slug = self._cx_slug_of(name_or_slug)
        from ..labels import LabelStore

        with self._cx_lock():
            reg = self._cx_reg()
            reg.pop(slug, None)
            self._cx_save_reg(reg)
            with LabelStore.editing(self._cx_labels_path()) as store:
                store.remove(slug)
            (self._cx_dir() / f"{slug}.npz").unlink(missing_ok=True)
            self.__dict__.setdefault("_cx_cache", {}).pop(slug, None)
        return {"deleted": slug}

    # --- teaching ------------------------------------------------------------------------

    def concept_candidates(self, name_or_slug: str, n: int = SEED_N) -> list[dict]:
        """Frames to mark yes/no. Before a concept is trained: half from a CLIP
        text match on its name (when CLIP is embedded), half deliberately varied.
        After: mostly the frames it's least sure about, plus a few it's sure of."""
        slug = self._cx_slug_of(name_or_slug)
        idx = self.index(self._model_name())
        if idx.size == 0:
            return []
        done = self._cx_store().labeled_ids(slug)
        rng = np.random.default_rng()
        picked: list[int] = []
        seen_scene: set = set()

        def take(rows, k):
            for i in rows:
                i = int(i)
                sid = idx.scene_ids[i]
                if (idx.keys[i], round(float(idx.times[i]), 2)) in done or sid in seen_scene:
                    continue
                picked.append(i)
                seen_scene.add(sid)
                if len(picked) >= k:
                    return

        probe = self._cx_probe(slug)
        if probe is not None:
            p = self._cx_frame_probs(slug)
            order = np.argsort(np.abs(p - 0.5))[: max(400, n * 20)]          # least sure first
            take(order, n - n // 3)
            take(np.argsort(-p)[: max(400, n * 20)], n)                      # a few it's sure of
        else:
            try:
                if self.has_clip_index():
                    clip = self.index(self._clip_name())
                    hits = clip.search(self._clip_query_vector(self._cx_reg()[slug]["name"], template=True),
                                       top_k=n * 4, per_scene=1)
                    rows = []
                    for h in hits:
                        span = idx._key_rows.get(h.key)
                        if span:
                            a, b = span
                            rows.append(a + int(np.argmin(np.abs(idx.times[a:b] - h.time))))
                    take(rows, n // 2)
            except Exception:  # noqa: BLE001 — CLIP is a seed, not a requirement
                pass
            take(rng.permutation(idx.size)[: n * 40], n)
        return [{"key": idx.keys[i], "t": round(float(idx.times[i]), 2), "scene_id": idx.scene_ids[i],
                 "thumb": f"/api/frame?key={idx.keys[i]}&t={float(idx.times[i]):g}"} for i in picked[:n]]

    def concept_label(self, name_or_slug: str, key: str, t: float, label: int,
                      scene_id: str | None = None) -> dict:
        from ..labels import LabelStore

        slug = self._cx_slug_of(name_or_slug)
        with LabelStore.editing(self._cx_labels_path()) as store:
            store.add(key, float(t), int(label), slug, scene_id=scene_id)
            pos, neg = store.counts(slug)
        out = {"slug": slug, "yes": pos, "no": neg}
        if pos >= MIN_EACH and neg >= MIN_EACH:
            out["report"] = self.concept_train(slug)             # small: retrain on every mark
        return out

    # --- the probe -----------------------------------------------------------------------

    def concept_train(self, name_or_slug: str) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        slug = self._cx_slug_of(name_or_slug)
        model = self._model_name()
        idx = self.index(model)
        X, y = [], []
        for lab in self._cx_store().for_profile(slug):
            v = idx.vector_at(lab.key, lab.time)
            if v is not None:
                X.append(v)
                y.append(int(lab.label))
        pos, neg = sum(y), len(y) - sum(y)
        if pos < MIN_EACH or neg < MIN_EACH:
            rep = {"trained": False, "yes": pos, "no": neg,
                   "reason": f"mark at least {MIN_EACH} yes and {MIN_EACH} no"}
        else:
            X = np.stack(X).astype(np.float32)
            y = np.asarray(y)
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000)
            k = max(2, min(5, pos, neg))
            acc = float(np.mean(cross_val_score(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0))))
            clf.fit(X, y)
            p = self._cx_dir()
            p.mkdir(parents=True, exist_ok=True)
            np.savez(p / f"{slug}.npz", w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
                     model=np.array(model))
            rep = {"trained": True, "yes": pos, "no": neg, "accuracy": round(acc, 3),
                   "reliability": reliability(acc, pos + neg), "at": time.time(), "model": model}
        with self._cx_lock():
            reg = self._cx_reg()
            if slug in reg:
                reg[slug]["report"] = rep
                self._cx_save_reg(reg)
            self.__dict__.setdefault("_cx_cache", {}).pop(slug, None)
        return rep

    def _cx_probe(self, slug: str):
        """(w, b) for the active model, or None (untrained, or trained on another model)."""
        p = self._cx_dir() / f"{slug}.npz"
        try:
            z = np.load(p)
        except (OSError, ValueError):
            return None
        if str(z["model"]) != self._model_name():
            return None
        return z["w"].astype(np.float32), float(z["b"])

    def _cx_frame_probs(self, slug: str) -> np.ndarray | None:
        """P(concept) for every frame of the active index (cached per index build)."""
        probe = self._cx_probe(slug)
        if probe is None:
            return None
        idx = self.index(self._model_name())
        cache = self.__dict__.setdefault("_cx_cache", {})
        hit = cache.get(slug)
        if hit and hit[0] == id(idx):
            return hit[1]
        w, b = probe
        probs = _sigmoid(idx.dot(w) + b).astype(np.float32)
        cache[slug] = (id(idx), probs, None)
        return probs

    def concept_scene_scores(self, name_or_slug: str) -> dict[str, float]:
        """scene_id → how strongly the scene shows the concept (mean of its top 10% frames)."""
        if (name_or_slug or "").split(":", 1)[0] in ("sound", "motion"):
            return self.sense_concept_scene_scores(name_or_slug)
        slug = self._cx_slug_of(name_or_slug)
        probs = self._cx_frame_probs(slug)
        if probs is None:
            return {}
        idx = self.index(self._model_name())
        cache = self.__dict__.setdefault("_cx_cache", {})
        hit = cache.get(slug)
        if hit and hit[2] is not None:
            return hit[2]
        out = {}
        for key, (a, b) in idx._key_rows.items():
            sid = (idx.key_meta.get(key) or {}).get("scene_id")
            if sid is None or b <= a:
                continue
            seg = np.sort(probs[a:b])[::-1]
            out[str(sid)] = round(float(seg[: max(1, (b - a) // 10)].mean()), 4)
        cache[slug] = (hit[0], hit[1], out) if hit else (id(idx), probs, out)
        return out

    def concept_search(self, name_or_slug: str, top_k: int | None = 60, per_scene: int | None = 3):
        """Frames showing the concept most strongly — the index's own search along
        the probe's direction, with each score turned into the concept's probability."""
        from ..search import Hit

        slug = self._cx_slug_of(name_or_slug)
        probe = self._cx_probe(slug)
        if probe is None:
            raise LookupError("teach this concept first — mark a few yes and no")
        w, b = probe
        norm = float(np.linalg.norm(w)) or 1.0
        hits = self.index(self._model_name()).search(w, top_k=top_k, per_scene=per_scene)
        return [Hit(scene_id=h.scene_id, key=h.key, time=h.time, score=float(_sigmoid(h.score * norm + b)))
                for h in hits]

    def trained_concepts(self) -> list[tuple[str, str]]:
        """[(slug, name)] of concepts with a probe for the active model."""
        return [(c["slug"], c["name"]) for c in self._visual_concepts() if self._cx_probe(c["slug"]) is not None]
