"""The Lab's Sound and Motion explorers: hear it, see it move, say "more of this".

Each sense keeps a small in-memory bank of clips — sound: CLAP vectors of 10 s
blocks (at most BANK_PER_SCENE per scene, evenly spread); motion: V-JEPA vectors
of 4 s clips — and you judge them one at a time, sound on. Your marks
("more of this" / "less of this") teach a taste per sense: a linear probe on the
clip vectors, with random clips as soft negatives. It drives:

  - the decks: "discover" (spread across everything), "more" (mostly what your
    taste likes, some discovery so it never narrows to one corner), and "teach"
    (the clips a named concept is least sure about);
  - named sound / motion concepts — your likes saved under a word, searchable
    ("sound: breathless") and usable as Catalogue filters;
  - two moment features, sound_like and motion_like (and one per named concept),
    which the moment model uses only where its honest held-out test says so.

State: state/sense_labels.jsonl (your marks), state/sense_concepts.json (named
concepts, with their probes inline), state/sense_taste-<sense>.npz.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

SENSES = ("sound", "motion")
BANK_PER_SCENE = {"sound": 30, "motion": 20}
HALF = {"sound": 5.0, "motion": 4.0}          # a deck clip plays centre ± this
MIN_LIKES = 3                                  # likes before a taste is learned
SOFT_NEG = 200                                 # random bank clips as soft negatives
SOFT_WEIGHT = 0.2
MORE_SHARE = 0.75                              # of a "more" deck; the rest is discovery
PER_SCENE = 2                                  # clips per scene in one deck
BANK_TTL = 1800                                # an unused bank is dropped after this
NEAR = 2.5                                     # seconds: a mark applies to the clip this close


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")[:40]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class SenseExploreMixin:
    # --- marks -----------------------------------------------------------------------------

    def _se_lock(self) -> threading.RLock:
        return self.__dict__.setdefault("_se_lk", threading.RLock())

    def _se_labels_path(self) -> Path:
        return self._state_dir() / "sense_labels.jsonl"

    def _se_marks(self, sense: str, concept: str | None = None) -> list[dict]:
        """Your current marks for a sense (the latest per clip; undone ones dropped).
        `concept`: only marks made for that named concept; None: the general taste."""
        try:
            lines = self._se_labels_path().read_text().splitlines()
        except OSError:
            return []
        latest: dict[tuple, dict] = {}
        for ln in lines:
            try:
                m = json.loads(ln)
            except ValueError:
                continue
            if m.get("sense") != sense or (m.get("concept") or None) != concept:
                continue
            latest[(m["scene_id"], round(float(m["t"]) / NEAR))] = m
        return [m for m in latest.values() if m.get("label") in (1, -1)]

    def sense_mark(self, sense: str, scene_id: str, t: float, label: int, concept: str | None = None) -> dict:
        """+1 more of this, −1 less of this, 0 undo."""
        if sense not in SENSES or label not in (1, -1, 0):
            raise ValueError("unknown sense or label")
        p = self._se_labels_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._se_lock(), p.open("a") as fh:
            fh.write(json.dumps({"sense": sense, "scene_id": str(scene_id), "t": round(float(t), 2),
                                 "label": int(label), "concept": concept or None, "at": time.time()}) + "\n")
        marks = self._se_marks(sense, concept)
        if concept:
            self._se_concept_train(concept)
        elif len(marks) % 5 == 0 or label == 0:
            self.__dict__.setdefault("_se_taste_cache", {}).pop(sense, None)
        self._se_drop_feature_caches()
        return self.sense_taste(sense)

    def _se_drop_feature_caches(self) -> None:
        self.__dict__.pop("_se_like_cache", None)

    # --- the clip bank ---------------------------------------------------------------------

    def _se_scene_clips(self, sense: str, sid: str):
        """(times, unit vectors) of every clip of a scene, or None."""
        if sense == "sound":
            s = self._sound_load(sid)
            return None if s is None or not len(s[2]) else (s[2], s[3])
        return self._motion_load(sid)

    def _se_bank(self, sense: str):
        """(vectors float16, scene ids, times) — capped per scene, rejects left out."""
        cache = self.__dict__.setdefault("_se_bank_cache", {})
        hit = cache.get(sense)
        sig = self._se_signature(sense)
        if hit and hit["sig"] == sig:
            hit["used"] = time.time()
            return hit["bank"]
        rejected = {r["scene_id"] for r in self._catalogue_all() if r["tier"] == "rejected"}
        cap = BANK_PER_SCENE[sense]
        V, S, T = [], [], []
        for sid in self._se_scene_ids(sense):
            if sid in rejected:
                continue
            c = self._se_scene_clips(sense, sid)
            if c is None:
                continue
            t, v = c
            pick = np.unique(np.linspace(0, len(t) - 1, min(cap, len(t))).round().astype(int))
            V.append(np.asarray(v[pick], np.float16))
            S += [sid] * len(pick)
            T.append(np.asarray(t[pick], np.float32))
        bank = (np.concatenate(V), np.asarray(S), np.concatenate(T)) if V else None
        cache[sense] = {"sig": sig, "bank": bank, "used": time.time()}
        for k, v in list(cache.items()):                  # forget banks nobody's using
            if time.time() - v["used"] > BANK_TTL and k != sense:
                cache.pop(k, None)
        return bank

    def _se_scene_ids(self, sense: str) -> list[str]:
        if sense == "sound":
            return list(self._hear_keys())
        from .motion import MOTION_NS

        watched = set(self._motion_cache().keys(MOTION_NS))     # one directory listing, not a look per scene
        return [sid for sid, e in self._zoom_view()["scenes"].items() if e.get("key") in watched]

    def _se_signature(self, sense: str):
        try:
            if sense == "sound":
                from .hear import ENV_NS
                return ("sound", len(self._hear_cache().keys(ENV_NS)))
            from .motion import MOTION_NS
            return ("motion", len(self._motion_cache().keys(MOTION_NS)))
        except Exception:  # noqa: BLE001
            return (sense, 0)

    def _se_vectors_of(self, sense: str, marks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Vectors of marked clips (nearest clip in the scene) and their labels."""
        X, y = [], []
        for m in marks:
            c = self._se_scene_clips(sense, m["scene_id"])
            if c is None:
                continue
            j = int(np.argmin(np.abs(c[0] - float(m["t"]))))
            X.append(np.asarray(c[1][j], np.float32))
            y.append(1 if m["label"] == 1 else 0)
        return (np.stack(X), np.asarray(y)) if X else (np.zeros((0, 1), np.float32), np.zeros(0))

    # --- learning a taste ------------------------------------------------------------------

    def _se_fit(self, sense: str, marks: list[dict]):
        """(w, b) of a probe on your marks + random soft negatives, or None."""
        from sklearn.linear_model import LogisticRegression

        X, y = self._se_vectors_of(sense, marks)
        if int((y == 1).sum()) < MIN_LIKES:
            return None
        bank = self._se_bank(sense)
        w = list(np.ones(len(y)))
        if bank is not None and len(bank[0]):
            rng = np.random.default_rng(len(marks))
            pick = rng.choice(len(bank[0]), size=min(SOFT_NEG, len(bank[0])), replace=False)
            X = np.vstack([X, bank[0][pick].astype(np.float32)])
            y = np.r_[y, np.zeros(len(pick))]
            w += [SOFT_WEIGHT] * len(pick)
        if len(set(y)) < 2:
            return None
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X, y, sample_weight=np.asarray(w))
        return m.coef_[0].astype(np.float32), float(m.intercept_[0])

    def _se_taste(self, sense: str):
        cache = self.__dict__.setdefault("_se_taste_cache", {})
        marks = self._se_marks(sense)
        key = (len(marks) // 5, self._se_signature(sense))
        hit = cache.get(sense)
        if hit and hit[0] == key:
            return hit[1]
        probe = self._se_fit(sense, marks)
        cache[sense] = (key, probe)
        if probe is not None:
            p = self._state_dir() / f"sense_taste-{sense}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, w=probe[0], b=np.array([probe[1]]))
        return probe

    def sense_taste(self, sense: str) -> dict:
        marks = self._se_marks(sense)
        likes = sum(1 for m in marks if m["label"] == 1)
        return {"sense": sense, "likes": likes, "dislikes": len(marks) - likes,
                "trained": self._se_taste(sense) is not None, "needs": max(0, MIN_LIKES - likes)}

    @staticmethod
    def _se_score(V: np.ndarray, probe) -> np.ndarray:
        w, b = probe
        out = np.empty(len(V), np.float32)
        for a in range(0, len(V), 50000):                 # widen a slice at a time
            out[a: a + 50000] = _sig(V[a: a + 50000].astype(np.float32) @ w + b)
        return out

    # --- decks -----------------------------------------------------------------------------

    def _se_marked(self, sense: str) -> set:
        try:
            lines = self._se_labels_path().read_text().splitlines()
        except OSError:
            return set()
        out = set()
        for ln in lines:
            try:
                m = json.loads(ln)
            except ValueError:
                continue
            if m.get("sense") == sense:
                out.add((m["scene_id"], round(float(m["t"]) / NEAR)))
        return out

    def _se_discover(self, V, S, T, free: np.ndarray, n: int, seed: int) -> list[int]:
        """Spread across the bank: farthest-point picks over a random subset, one per scene."""
        idx = np.flatnonzero(free)
        if not len(idx):
            return []
        rng = np.random.default_rng(seed)
        sub = rng.choice(idx, size=min(5000, len(idx)), replace=False)
        X = V[sub].astype(np.float32)
        chosen = [int(rng.integers(len(sub)))]
        dist = 1.0 - X @ X[chosen[0]]
        seen = {S[sub[chosen[0]]]}
        while len(chosen) < min(n, len(sub)):
            d = dist.copy()
            for j in range(len(sub)):
                if S[sub[j]] in seen:
                    d[j] = -1
            j = int(np.argmax(d))
            if d[j] < 0:
                break
            chosen.append(j)
            seen.add(S[sub[j]])
            dist = np.minimum(dist, 1.0 - X @ X[j])
        return [int(sub[j]) for j in chosen]

    def sense_deck(self, sense: str, mode: str = "discover", n: int = 12, concept: str | None = None) -> dict:
        if sense not in SENSES:
            raise ValueError("unknown sense")
        bank = self._se_bank(sense)
        if bank is None:
            return {"sense": sense, "mode": mode, "items": [], "reason":
                    "nothing heard yet — Lab → Models & senses → Hearing → Run now" if sense == "sound"
                    else "nothing watched yet — Lab → Models & senses → Motion → Run now"}
        V, S, T = bank
        marked = self._se_marked(sense)
        free = np.array([(s, round(float(t) / NEAR)) not in marked for s, t in zip(S, T)])
        seed = int(time.time()) // 60
        picks: list[tuple[int, str]] = []
        note = None
        if mode in ("more", "teach"):
            probe = self._se_concept_probe(concept) if mode == "teach" and concept else self._se_taste(sense)
            if probe is None:
                note = (f"teach “{concept}” a few more first" if mode == "teach"
                        else f"mark {self.sense_taste(sense)['needs']} more “more of this” first — discovering meanwhile")
            else:
                p = self._se_score(V, probe)
                if mode == "teach":
                    order = np.argsort(np.abs(p - 0.5))           # least sure first
                    k_main = max(1, int(round(n * 2 / 3)))
                    tops = [i for i in np.argsort(-p) if free[i]][: n - k_main]
                    picks += [(int(i), "Peaks thinks this is it — right?") for i in tops]
                    why = "Peaks can't tell — is this it?"
                else:
                    order = np.argsort(-p)
                    k_main = int(round(n * MORE_SHARE))
                    why = "like what you picked"
                per: dict[str, int] = {}
                main = []
                for i in order:
                    if len(main) >= k_main:
                        break
                    if not free[i] or per.get(S[i], 0) >= PER_SCENE or any(i == j for j, _ in picks):
                        continue
                    per[S[i]] = per.get(S[i], 0) + 1
                    main.append((int(i), why))
                picks = main + picks
        taken = {i for i, _ in picks}
        if len(picks) < n:
            f2 = free.copy()
            f2[list(taken)] = False
            picks += [(i, "something different") for i in self._se_discover(V, S, T, f2, n - len(picks), seed)]
        if mode == "more":
            rng = np.random.default_rng(seed)
            rng.shuffle(picks)                                    # discovery mixed in, not tacked on
        return {"sense": sense, "mode": mode, "concept": concept, "note": note,
                "items": [self._se_item(sense, S[i], float(T[i]), why) for i, why in picks[:n]],
                "taste": self.sense_taste(sense)}

    def _se_item(self, sense: str, sid: str, t: float, why: str) -> dict:
        half = HALF[sense]
        start = max(0.0, t - half)
        key = self._hear_keys().get(sid)
        row = self._se_rows().get(sid) or {}
        return {"scene_id": sid, "t": round(t, 2), "start": round(start, 2), "end": round(t + half, 2),
                "why": why, "title": row.get("title") or f"Scene {sid}", "tier": row.get("tier"),
                "stream": self.stream_url(sid, start=0),
                "thumb": f"/api/frame?key={key}&t={t:g}" if key else None}

    def _se_rows(self) -> dict:
        hit = self.__dict__.get("_se_rows_cache")
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        rows = {r["scene_id"]: r for r in self._catalogue_all()}
        self._se_rows_cache = (time.time(), rows)
        return rows

    # --- named sound / motion concepts -----------------------------------------------------

    def _se_reg_path(self) -> Path:
        return self._state_dir() / "sense_concepts.json"

    def _se_reg(self) -> dict:
        try:
            d = json.loads(self._se_reg_path().read_text())
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _se_save_reg(self, d: dict) -> None:
        from .. import atomic

        p = self._se_reg_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_json(p, d, indent=1)

    def sense_concept_key(self, sense: str, name: str) -> str | None:
        """"sound:breathless" for a named concept, matched by name or key; else None."""
        reg = self._se_reg()
        k = f"{sense}:{_slug(name)}"
        if k in reg:
            return k
        for key, v in reg.items():
            if v["sense"] == sense and v["name"].strip().lower() == (name or "").strip().lower():
                return key
        return None

    def sense_concepts(self, sense: str | None = None) -> list[dict]:
        out = []
        for key, v in sorted(self._se_reg().items(), key=lambda kv: kv[1].get("created", 0)):
            if sense and v["sense"] != sense:
                continue
            marks = self._se_marks(v["sense"], key)
            yes = sum(1 for m in marks if m["label"] == 1)
            out.append({"slug": key, "name": v["name"], "kind": v["sense"], "yes": yes, "no": len(marks) - yes,
                        "trained": bool(v.get("w")), "reliability": "taught" if v.get("w") else "needs more examples"})
        return out

    def sense_concept_save(self, sense: str, name: str) -> dict:
        """Save your current likes and dislikes as a named concept, and train it."""
        name = (name or "").strip()[:60]
        if sense not in SENSES or not _slug(name):
            raise ValueError("give it a name — a few plain words")
        key = f"{sense}:{_slug(name)}"
        with self._se_lock():
            reg = self._se_reg()
            reg.setdefault(key, {"sense": sense, "name": name, "created": time.time()})
            self._se_save_reg(reg)
            p = self._se_labels_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as fh:
                for m in self._se_marks(sense):
                    fh.write(json.dumps({**m, "concept": key, "at": time.time()}) + "\n")
        self._se_concept_train(key)
        return next(c for c in self.sense_concepts() if c["slug"] == key)

    def _se_concept_train(self, key: str) -> None:
        reg = self._se_reg()
        v = reg.get(key)
        if not v:
            return
        probe = self._se_fit(v["sense"], self._se_marks(v["sense"], key))
        v["w"] = None if probe is None else [round(float(x), 6) for x in probe[0]]
        v["b"] = None if probe is None else probe[1]
        reg[key] = v
        self._se_save_reg(reg)
        self.__dict__.pop("_se_cscore_cache", None)
        self._se_drop_feature_caches()

    def _se_concept_probe(self, key: str | None):
        v = self._se_reg().get(key or "")
        if not v or not v.get("w"):
            return None
        return np.asarray(v["w"], np.float32), float(v["b"])

    def sense_concept_delete(self, key: str) -> dict:
        with self._se_lock():
            reg = self._se_reg()
            reg.pop(key, None)
            self._se_save_reg(reg)
        self.__dict__.pop("_se_cscore_cache", None)
        self._se_drop_feature_caches()
        return {"deleted": key}

    def sense_concept_scene_scores(self, key: str) -> dict[str, float]:
        """scene_id → how strongly the scene has it (mean of its top 10 % clips in the bank)."""
        v = self._se_reg().get(key)
        probe = self._se_concept_probe(key)
        if not v or probe is None:
            return {}
        cache = self.__dict__.setdefault("_se_cscore_cache", {})
        bank = self._se_bank(v["sense"])
        if bank is None:
            return {}
        sig = (key, id(bank))
        if sig in cache:
            return cache[sig]
        p = self._se_score(bank[0], probe)
        out: dict[str, float] = {}
        S = bank[1]
        order = np.argsort(S, kind="stable")
        Ss, ps = S[order], p[order]
        cuts = np.flatnonzero(np.r_[True, Ss[1:] != Ss[:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:]):
            seg = np.sort(ps[a:b])[::-1]
            out[str(Ss[a])] = round(float(seg[: max(1, (b - a) // 10)].mean()), 4)
        cache[sig] = out
        return out

    def sense_concept_search(self, key: str, top_k: int = 60) -> list:
        from ..search import Hit

        v = self._se_reg().get(key)
        probe = self._se_concept_probe(key)
        bank = self._se_bank(v["sense"]) if v else None
        if probe is None or bank is None:
            return []
        p = self._se_score(bank[0], probe)
        keys = self._hear_keys()
        best: dict[str, int] = {}
        for i in np.argsort(-p):
            sid = str(bank[1][i])
            if sid not in best:
                best[sid] = int(i)
            if len(best) >= top_k:
                break
        return [Hit(scene_id=sid, key=keys.get(sid, ""), time=float(max(0.0, bank[2][i] - HALF[v["sense"]])),
                    score=float(p[i])) for sid, i in best.items()]

    # --- what the moment model gets --------------------------------------------------------

    def sense_window_likes(self, sense: str, sid: str, centre: float, half: float) -> dict:
        """{sound_like / motion_like, and one per trained named concept}: the window's
        best clip under your taste and each named concept (0 when not learned)."""
        c = self._se_scene_clips(sense, sid)
        if c is None:
            return {}
        t, v = c
        m = np.abs(t - centre) <= half + HALF[sense]
        if not m.any():
            return {}
        vw = np.asarray(v[m], np.float32)
        out = {}
        probe = self._se_taste(sense)
        out[f"{sense}_like"] = float(_sig(vw @ probe[0] + probe[1]).max()) if probe is not None else 0.0
        for k in self.sense_concept_feature_names(sense):
            pr = self._se_concept_probe(k)
            out[k] = float(_sig(vw @ pr[0] + pr[1]).max()) if pr is not None else 0.0
        return out

    def sense_concept_feature_names(self, sense: str) -> list[str]:
        return [k for k, v in self._se_reg().items() if v["sense"] == sense and v.get("w")]
