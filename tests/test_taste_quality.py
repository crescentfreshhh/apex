"""The taste engine's accuracy work: the held-out benchmark, weak labels from
grades/markers, PU-cleaned background, model selection, temporal context,
engagement signals and smarter active learning — on a synthetic library whose
taste is multi-modal and whose peaks last several frames, like the real thing."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from fakestash import FakeStash  # noqa: E402
from peaks.cache import EmbeddingCache  # noqa: E402
from peaks.classifier import TasteClassifier, with_context  # noqa: E402
from peaks.config import Config  # noqa: E402
from peaks.labels import LabelStore  # noqa: E402
from peaks.pipeline import WeakLabel, _spy_filter, sample_background_frames, train_profile  # noqa: E402
from peaks.taste_eval import grouped_oof  # noqa: E402

DIM, FRAMES = 24, 30


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def _library(tmp_path, n_scenes=60, seed=0, xor=False, model="dinov2"):
    """Scenes of random frames; every third scene holds a 6-frame peak near one
    of two taste modes (or, with `xor`, a taste a hyperplane can't separate).
    Returns (cache, peaks {key: peak start time}, taste scenes)."""
    rng = np.random.default_rng(seed)
    cache = EmbeddingCache(str(tmp_path / "cache"))
    modes = _unit(rng.normal(0, 1, (2, DIM)))
    peaks = {}
    for i in range(n_scenes):
        key = f"k{i}"
        v = rng.normal(0, 1, (FRAMES, DIM))
        if i % 3 == 0:
            s = int(rng.integers(3, FRAMES - 9))
            for j in range(s, s + 6):
                if xor:
                    sign = 1 if (i // 3) % 2 else -1
                    v[j, :2] = [3 * sign, 3 * sign]
                else:
                    v[j] = v[j] * 0.8 + 3.2 * modes[(i // 3) % 2] * np.sqrt(DIM) / 4
            peaks[key] = float(s + 2)
        elif xor and i % 3 == 1:
            s = int(rng.integers(3, FRAMES - 9))
            sign = 1 if (i // 3) % 2 else -1
            for j in range(s, s + 6):
                v[j, :2] = [3 * sign, -3 * sign]    # near the taste, on the wrong diagonal
        cache.save(key, model, np.arange(FRAMES, dtype=np.float32), _unit(v),
                   meta={"scene_id": str(i)})
    return cache, peaks


def _labels(tmp_path, peaks, n_pos=12, n_neg=12, seed=0):
    rng = np.random.default_rng(seed)
    store = LabelStore(tmp_path / "labels.json")
    for key, t in list(peaks.items())[:n_pos]:
        store.add(key, t, 1, "apex", scene_id=key[1:])
    others = [f"k{i}" for i in range(60) if f"k{i}" not in peaks]
    for key in others[:n_neg]:
        store.add(key, float(rng.integers(0, FRAMES)), 0, "apex", scene_id=key[1:])
    return store


# --- building blocks -------------------------------------------------------------

def test_context_features_and_backward_compatible_models(tmp_path):
    v = _unit(np.random.default_rng(0).normal(0, 1, (5, 4)))
    f = with_context(v, 1)
    assert f.shape == (5, 8)
    assert np.allclose(f[0, 4:], v[:2].mean(axis=0))      # clipped at the scene edge
    assert np.allclose(f[2, 4:], v[1:4].mean(axis=0))
    y = np.array([1, 0, 1, 0, 1])
    m = TasteClassifier(kind="logreg", context=1).train(f, y)
    # raw frames are read as one scene's sequence; ready-made context features also work
    assert np.allclose(m.predict_proba(v), m.predict_proba(f))
    m.save(tmp_path / "m.pkl")
    assert TasteClassifier.load(tmp_path / "m.pkl").context == 1
    import pickle
    old = pickle.loads((tmp_path / "m.pkl").read_bytes())
    old.pop("context")
    (tmp_path / "old.pkl").write_bytes(pickle.dumps(old))
    assert TasteClassifier.load(tmp_path / "old.pkl").context == 0


def test_holdout_never_trains_on_a_test_scene():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 4)).astype(np.float32)
    y = np.array([1, 0] * 40)
    groups = np.array([f"g{i // 4}" for i in range(80)])
    bg = np.zeros(80, dtype=bool)
    bg[-8:] = True                                   # background of scenes g18, g19
    seen = []

    def fit(Xa, ya, wa):
        seen.append({tuple(r) for r in Xa.round(5)})
        return lambda Xb: Xb[:, 0]

    res = grouped_oof(fit, X, y, np.ones(80), groups, ~bg, bg)
    assert res and res["folds"] >= 2 and res["evaluated"] == 72
    assert len(seen) == res["folds"]
    # each fold's training rows never include a row from its test groups: every
    # labeled row is absent from exactly one fold's training set
    for i in range(72):
        assert sum(tuple(X[i].round(5)) not in s for s in seen) == 1


def test_background_skips_in_taste_scenes_and_spy_filter_drops_lookalikes(tmp_path):
    cache, peaks = _library(tmp_path)
    frames = sample_background_frames(cache, "dinov2", 200, exclude_keys=set(peaks))
    assert frames and not {k for k, _ in frames} & set(peaks)

    rng = np.random.default_rng(1)
    pos = _unit(rng.normal(0, 0.2, (40, 8)) + np.eye(8)[0] * 2)
    neg = _unit(rng.normal(0, 1, (60, 8)))
    planted = _unit(rng.normal(0, 0.2, (15, 8)) + np.eye(8)[0] * 2)   # taste hiding in bg
    X = np.vstack([pos, neg, planted])
    y = np.array([1] * 40 + [0] * 75)
    bg = np.array([False] * 40 + [True] * 75)
    keep = _spy_filter(X, y, np.ones(len(y)), bg)
    assert (~keep[-15:]).sum() >= 10                 # most planted positives dropped
    assert keep[40:100].mean() > 0.85                # real background mostly kept


def test_weak_rows_are_weighted_and_counted(tmp_path):
    cache, peaks = _library(tmp_path)
    store = _labels(tmp_path, peaks, n_pos=4, n_neg=6)
    tier = [k for k in peaks][4:10]
    weak = [WeakLabel(k, peaks[k], 1, 0.5, "tier") for k in tier]
    clf, stats = train_profile(store, cache, "dinov2", "apex", background_ratio=1.0,
                               weak=weak, exclude_bg_keys=set(tier))
    assert stats["sources"]["tier"] == 6 and stats["sources"]["explicit"] == 10
    assert stats["positives"] == 10


# --- the benchmark proves the upgrades -------------------------------------------

def test_benchmark_reports_and_upgrades_do_not_regress(tmp_path):
    cache, peaks = _library(tmp_path, n_scenes=90)
    store = _labels(tmp_path, peaks, n_pos=10, n_neg=15)
    base_clf, base = train_profile(store, cache, "dinov2", "apex", background_ratio=1.0)
    h = base["holdout"]
    assert {"auc", "p_at_50", "base_rate", "peak_hit"} <= set(h)
    assert h["auc"] > 0.7

    tier = list(peaks)[10:]
    weak = [WeakLabel(k, peaks[k], 1, 0.5, "tier") for k in tier]
    _, up = train_profile(store, cache, "dinov2", "apex", kind="auto", background_ratio=2.0,
                          weak=weak, exclude_bg_keys=set(tier), pu_filter=True, context="auto")
    assert up["holdout"]["auc"] >= h["auc"] - 0.02
    assert up["holdout"]["peak_hit"] >= h["peak_hit"]
    print("baseline", h, "\nupgraded", up["holdout"], up["kind"], up["context"])


def test_auto_picks_a_nonlinear_model_when_taste_is_not_linear(tmp_path):
    cache, peaks = _library(tmp_path, n_scenes=150, xor=True)
    store = LabelStore(tmp_path / "labels.json")
    rng = np.random.default_rng(0)
    for i in range(150):
        key = f"k{i}"
        times, vecs, _ = cache.load(key, "dinov2")
        if key in peaks:
            for t in range(int(peaks[key]) - 2, int(peaks[key]) + 3):
                store.add(key, float(t), 1, "apex")
        elif i % 3 == 1:
            for t in range(FRAMES):
                if abs(vecs[t, 0]) > 0.3:
                    store.add(key, float(t), 0, "apex")
        else:
            store.add(key, float(rng.integers(0, FRAMES)), 0, "apex")
    _, stats = train_profile(store, cache, "dinov2", "apex", kind="auto", context=0)
    assert stats["samples"] >= 200
    assert stats["kind"] == "mlp", stats.get("candidates")


# --- service: engagement, weak rows from Stash, active learning -------------------

@pytest.fixture
def svc(tmp_path, monkeypatch):
    import peaks.web.service as svc_mod

    cache, peaks = _library(tmp_path, model="dinov2-vitl14")
    scenes = {str(i): {"rating100": None, "o_counter": 0} for i in range(60)}
    for i in (3, 6, 9):
        scenes[str(i)].update(rating100=100, o_counter=18)     # Légendaire
    scenes["1"].update(rating100=20)                           # rejected
    stash = FakeStash(scenes)
    stash.markers = [{"scene_id": "12", "seconds": peaks["k12"], "marker_id": "m1"}]
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    s = svc_mod.Service(cfg)
    s._peaks = peaks
    return s


def test_training_learns_from_grades_markers_and_logs_quality(svc):
    for key in ("k0", "k15", "k18", "k21"):
        svc.add_label(key, svc._peaks[key], 1, scene_id=key[1:])
    for key in ("k2", "k4", "k5", "k7", "k8"):
        svc.add_label(key, 5.0, 0, scene_id=key[1:])
    svc.train_taste(mode="full")                       # first model → tier rows can be picked
    out = svc.train_taste(mode="full")
    src = out["sources"]
    assert src.get("marker") == 1 and src.get("tier", 0) >= 3 and src.get("reject", 0) >= 1
    q = svc.taste_quality()
    assert len(q["history"]) == 2 and q["latest"]["samples"] == out["samples"]


def test_engagement_is_soft_and_dropped_on_reject(svc, monkeypatch):
    r = svc.record_engagement("30", 12.0)
    assert r["kept"]
    assert svc.record_engagement("30", 15.0)["reason"] == "already known"   # within 10 s
    labs = svc._label_store().for_profile("apex")
    assert labs[0].source == "engage" and labs[0].weight == svc.ENGAGE_WEIGHT
    assert svc.label_counts()["positive"] == 0                      # not a 👍
    assert svc.list_labels()["total"] == 0                          # not in the editor
    svc.grade_scene("30", "reject")
    assert svc._label_store().for_profile("apex") == []


def test_active_learning_skips_rejects_and_varies(svc):
    svc.add_label("k0", svc._peaks["k0"], 1, scene_id="0")
    svc.add_label("k2", 3.0, 0, scene_id="2")
    svc.train_taste()
    svc.set_scene_hidden("5", True)
    asked = [svc.next_uncertain() for _ in range(8)]
    assert all(a and a["scene_id"] != "5" for a in asked)
    assert len({(a["key"], a["time"]) for a in asked}) >= 6


# --- memory & crash safety ---------------------------------------------------------

def test_training_does_not_pin_whole_scenes_in_memory(tmp_path, monkeypatch):
    """Regression: each training row was a view into its scene's full frame
    matrix, so every scene touched stayed resident — a real library OOM'd.
    (The jump-to-peak benchmark legitimately holds a capped sample of scenes;
    the cap is pinned low here so only a leak could blow the budget.)"""
    import tracemalloc

    import peaks.pipeline as pl

    monkeypatch.setattr(pl, "MAX_PEAK_SCENES", 4)
    rng = np.random.default_rng(0)
    cache = EmbeddingCache(str(tmp_path / "cache"))
    store = LabelStore(tmp_path / "labels.json")
    n, frames, dim = 80, 600, 64
    for i in range(n):
        v = _unit(rng.normal(0, 1, (frames, dim)))
        cache.save(f"k{i}", "m", np.arange(frames, dtype=np.float32), v, meta={"scene_id": str(i)})
        store.add(f"k{i}", float(rng.integers(0, frames)), int(i % 2 == 0), "a")
    library = n * frames * dim * 4
    kw = dict(background_ratio=2.0, context="auto", pu_filter=True)
    train_profile(store, cache, "m", "a", **kw)   # warm-up: sklearn's lazy imports aren't training
    tracemalloc.start()
    train_profile(store, cache, "m", "a", **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.35 * library, f"peak {peak / 1e6:.1f} MB vs library {library / 1e6:.1f} MB"


def test_grade_bursts_start_one_background_retrain(svc, monkeypatch):
    import threading

    started = []
    gate = threading.Event()
    monkeypatch.setattr(type(svc), "_tier_model_state", lambda self: {"model": object(), "report": None})

    def fake_train(self):
        started.append(1)
        gate.wait(2)
        self._tier_training = False
    monkeypatch.setattr(type(svc), "_train_tier_quietly", fake_train)
    for _ in range(80):                  # a fast grading burst
        svc._note_grade()
    gate.set()
    assert len(started) == 1


def test_health_line_names_in_process_work():
    from peaks.web import forensics

    with forensics.busy("tier-train"):
        assert "work=tier-train" in forensics.health_line()
    assert "work=" not in forensics.health_line()


# --- quick retrain vs. full measure, and when the full one runs ---------------------

def test_quick_retrain_reuses_the_measured_variant_and_logs_nothing(svc):
    for key in ("k0", "k15", "k18", "k21"):
        svc.add_label(key, svc._peaks[key], 1, scene_id=key[1:])
    for key in ("k2", "k4", "k5", "k7", "k8"):
        svc.add_label(key, 5.0, 0, scene_id=key[1:])
    quick = svc.train_taste()                          # default: quick, nothing measured yet
    assert quick["mode"] == "quick" and quick["kind"] == "logreg" and "holdout" not in quick
    assert svc.taste_quality()["history"] == []
    full = svc.train_taste(mode="full")
    assert full["mode"] == "full" and "holdout" in full
    q = svc.taste_quality()
    assert len(q["history"]) == 1 and q["schedule"]["signals"] == 0   # reset by the measure
    again = svc.train_taste()
    assert (again["kind"], again["context"]) == (full["kind"], full["context"])
    assert len(svc.taste_quality()["history"]) == 1


def test_overnight_measure_needs_the_hour_and_enough_new_signals(svc):
    import time

    svc.cfg.modeling.taste_measure_hour = 3
    svc.cfg.modeling.taste_measure_min_signals = 5
    t3 = time.mktime((2026, 9, 27, 3, 10, 0, 0, 0, -1))
    t14 = time.mktime((2026, 9, 27, 14, 10, 0, 0, 0, -1))
    assert not svc.measure_due(t3)                     # nothing new yet
    for i in range(3):
        svc.add_label("k0", float(i), 1, scene_id="0")
    svc.grade_scene("30", "legendaire")                # grades count too
    svc.grade_scene("31", "merveilleuse")
    assert svc.measure_schedule()["signals"] == 5
    assert svc.measure_due(t3) and not svc.measure_due(t14)
    svc._measure_state_update(reset=True)              # a measure just ran
    assert not svc.measure_due(t3)
    svc.cfg.modeling.taste_measure_hour = -1
    assert not svc.measure_due(t3)


def test_measure_runs_as_a_job_with_progress(svc, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    for key in ("k0", "k15", "k18"):
        svc.add_label(key, svc._peaks[key], 1, scene_id=key[1:])
    for key in ("k2", "k4", "k5"):
        svc.add_label(key, 5.0, 0, scene_id=key[1:])
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    jid = api.post("/api/taste/measure").json()["job"]
    assert api.post("/api/taste/measure").status_code in (200, 409)
    for _ in range(300):
        job = next(j for j in api.get("/api/jobs").json() if j["id"] == jid)
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert job["result"]["mode"] == "full" and job["progress"]["pct"] >= 0.9
    assert api.get("/api/taste/quality").json()["latest"] is not None


# --- megaboard: taste scores, model-ranked tier boards, unreviewed channel, cancel --

def _trained(svc):
    for key in ("k0", "k15", "k18", "k21"):
        svc.add_label(key, svc._peaks[key], 1, scene_id=key[1:])
    for key in ("k2", "k4", "k5", "k7", "k8"):
        svc.add_label(key, 5.0, 0, scene_id=key[1:])
    svc.train_taste()
    return svc


def test_taste_scores_for_matches_the_library_scores(svc):
    _trained(svc)
    model = svc._model_name()
    scores, by = svc._taste_scores(model)
    idx = svc.index(model)
    s, e = idx._key_rows["k3"]
    i = s + 7
    out = svc.taste_scores_for([("3", float(idx.times[i])), ("999", 1.0)])
    assert by == "classifier" and out["scored_by"] == "classifier"
    assert out["scores"][0] == pytest.approx(float(scores[i]), abs=1e-4)
    assert out["scores"][1] is None                      # not embedded


def test_tier_board_is_ranked_by_the_trained_model(svc):
    _trained(svc)
    _settle(svc)       # a first-ever score pass doesn't block the board — wait for it here
    r = svc.tier_board("legendaire", per_scene=3)
    assert r["scenes"] == 3 and r["hits"]
    got = [h.score for h in r["hits"]]
    assert got == sorted(got, reverse=True)                # best-first across scenes
    by_scene: dict = {}
    for h in r["hits"]:
        by_scene.setdefault(h.scene_id, []).append(h.time)
    for times in by_scene.values():                        # ≥10 s apart within a scene
        times.sort()
        assert all(b - a >= 10 for a, b in zip(times, times[1:]))
    scores, _ = svc._taste_scores(svc._model_name())
    assert max(got) <= float(scores.max()) + 1e-6


def test_each_tier_channel_plays_only_its_own_tier(svc):
    st = svc.client()
    st.s["12"].update(rating100=100, o_counter=17)            # Exceptionnelle
    st.s["15"].update(rating100=100, o_counter=16)            # Merveilleuse
    svc.invalidate_meta()
    svc._cat_cache = None
    keys = [t["key"] for t in svc.board_sources()["tiers"]]
    assert keys == ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "unreviewed"]
    exc = svc.tier_board("exceptionnelle", per_scene=2)
    assert exc["scenes"] == 1 and {h.scene_id for h in exc["hits"]} == {"12"}
    mer = svc.tier_board("merveilleuse", per_scene=2)
    assert mer["scenes"] == 1 and {h.scene_id for h in mer["hits"]} == {"15"}
    both = svc.tier_board("exceptionnelle,legendaire", per_scene=2)   # an old combined link
    assert {h.scene_id for h in both["hits"]} == {"3", "6", "9", "12"}


def test_unreviewed_channel_only_holds_ungraded_scenes(svc):
    r = svc.tier_board("unreviewed", per_scene=2)
    sids = {h.scene_id for h in r["hits"]}
    assert r["label"].startswith("Unreviewed") and sids
    assert not sids & {"1", "3", "6", "9"}                 # rejected / graded
    assert any(t["key"] == "unreviewed" for t in svc.board_sources()["tiers"])
    with pytest.raises(ValueError):
        svc.tier_label("unreviewed")                       # exports stay keeper-only


def test_cancelled_measure_leaves_the_model_alone(svc):
    _trained(svc)
    path = svc._taste_path(svc.cfg.markers.tag_name, svc._model_name())
    before = path.read_bytes()

    class _Job:
        cancelled = True
        progress: dict = {}

    with pytest.raises(RuntimeError, match="cancelled"):
        svc.train_taste(mode="full", job=_Job())
    assert path.read_bytes() == before


# --- pivots stay fast: ratings don't re-page markers or re-score the library ------

def test_rating_keeps_marker_cache_and_model_scores(svc, monkeypatch):
    _trained(svc)
    stash = svc.client()
    pages = []
    real = stash.iter_markers_by_tag
    monkeypatch.setattr(stash, "iter_markers_by_tag", lambda tag: pages.append(tag) or real(tag))
    model = svc._model_name()
    svc._taste_centroid(model)
    scores, by = svc._taste_scores(model)
    assert by == "classifier"
    n = len(pages)
    for i in range(3):                                   # rate from the board
        svc.add_label("k30", float(i), 1, scene_id="30")
        svc._taste_centroid(model)
    assert len(pages) == n                               # markers not re-fetched from Stash
    assert svc._taste_scores(model)[0] is scores         # model scores kept (no retrain yet)


def test_save_and_unsave_update_the_marker_cache(svc, monkeypatch):
    stash = svc.client()
    stash.markers = []
    monkeypatch.setattr(stash, "find_or_create_tag", lambda name: type("T", (), {"id": "t1"})(), raising=False)
    monkeypatch.setattr(stash, "create_scene_marker", lambda **kw: {"id": "m5"}, raising=False)
    monkeypatch.setattr(stash, "markers_for_scene",
                        lambda sid: [{"marker_id": "m5", "seconds": 40.0, "primary_tag": svc.cfg.markers.tag_name}],
                        raising=False)
    monkeypatch.setattr(stash, "destroy_scene_markers", lambda ids: len(ids), raising=False)
    tag = svc.cfg.markers.tag_name
    assert svc._taste_markers(tag) == []
    svc.create_apex("30", 40.0)
    assert [m["marker_id"] for m in svc._taste_markers(tag)] == ["m5"]
    svc.remove_apex("30", 40.0, marker_id="m5")
    assert svc._taste_markers(tag) == []


def test_stream_urls_reuse_one_client(svc, monkeypatch):
    import peaks.web.service as svc_mod

    made = []
    real = svc_mod.Service.client
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: made.append(1) or real(self))
    for i in range(50):
        svc.stream_url(str(i), start=1.0)
    assert len(made) == 1


def test_slow_requests_are_logged(tmp_path):
    from peaks.web import forensics

    lines = []
    orig = forensics._write
    forensics._write = lambda line, echo=True: lines.append(line)
    try:
        forensics.note_duration("GET", "/api/performer/best", 0.4)
        forensics.note_duration("GET", "/api/performer/best", 12.34)
    finally:
        forensics._write = orig
    assert len(lines) == 1 and "[slow] 12.3s GET /api/performer/best" in lines[0]


# --- saved moments vs. the legacy scorer's auto markers ----------------------------

def test_auto_marker_detection():
    from peaks.playlist import is_auto_marker

    assert is_auto_marker("apex 0.873", "apex")
    assert is_auto_marker("apex 1", "apex")
    assert not is_auto_marker("apex (saved)", "apex")
    assert not is_auto_marker("my favourite bit", "apex")
    assert not is_auto_marker("heels 0.5", "apex")          # another tag's score


def test_only_your_saves_feed_taste_and_the_audit_counts_them(svc):
    tag = svc.cfg.markers.tag_name
    svc.client().markers = [
        {"marker_id": "s1", "scene_id": "0", "seconds": svc._peaks["k0"], "title": f"{tag} (saved)"},
        {"marker_id": "s2", "scene_id": "15", "seconds": svc._peaks["k15"], "title": f"{tag} (saved)"},
        {"marker_id": "a1", "scene_id": "21", "seconds": 4.0, "title": f"{tag} 0.912"},     # auto
        {"marker_id": "a2", "scene_id": "24", "seconds": 8.0, "title": f"{tag} 0.871"},     # auto
        {"marker_id": "s3", "scene_id": "999", "seconds": 1.0, "title": f"{tag} (saved)"},  # not embedded
    ]
    for m in svc.client().markers:
        m["end_seconds"] = None
    svc._forget_markers()
    svc.add_label("k0", svc._peaks["k0"], 1, scene_id="0")      # the 👍 the save made
    _, sources = svc._taste_sources(svc._model_name())
    apex_scenes = {s["scene_id"] for s in sources if s["kind"] == "apex"}
    assert apex_scenes == {"0", "15"}                          # auto markers aren't "loved"
    a = svc.saved_moments_audit()
    assert a == {"tag": tag, "saved": 3, "rated": 1, "added": 1, "not_embedded": 1, "auto_ignored": 2}
    weak, _ = svc._weak_taste_rows(tag, svc._model_name())
    assert {w.key for w in weak if w.source == "marker"} == {"k15"}
    assert "k0" in svc._saved_at and "k21" not in svc._saved_at
    board = svc.board_apexes()
    assert board["count"] == 3                                 # Saved moments channel: yours only


def test_a_saved_moment_counts_double(tmp_path, monkeypatch):
    import peaks.classifier as clf_mod

    cache, peaks = _library(tmp_path)
    store = _labels(tmp_path, peaks, n_pos=6, n_neg=6)
    seen = {}
    real = clf_mod.TasteClassifier.train

    def spy(self, X, y, sample_weight=None):
        seen["w"] = None if sample_weight is None else list(sample_weight)
        return real(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(clf_mod.TasteClassifier, "train", spy)
    first = list(peaks)[0]
    train_profile(store, cache, "dinov2", "apex", evaluate=False, boost={first: [peaks[first]]})
    w = seen["w"]
    labs = list(store.for_profile("apex"))
    i = next(j for j, lab in enumerate(labs) if lab.key == first)
    others = [w[j] for j, lab in enumerate(labs) if lab.label == 1 and lab.key != first]
    assert w[i] == pytest.approx(2 * others[0])


# --- channels never wait on a library re-score ------------------------------------

def _settle(svc, timeout=20.0):
    """Wait for any background re-score to finish."""
    import time
    t0 = time.time()
    while svc.__dict__.get("_score_refreshing") and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_retrain_serves_old_scores_until_the_rescore_lands(svc):
    _trained(svc)
    _settle(svc)
    model = svc._model_name()
    old, by = svc._taste_scores(model)
    assert by == "classifier"
    svc.add_label("k33", svc._peaks["k33"], 1, scene_id="33")
    svc.train_taste()                                      # new model file
    now, _ = svc._taste_scores(model)                      # immediately: no wait
    assert now is old or now.shape == old.shape
    _settle(svc)
    fresh, _ = svc._taste_scores(model)
    assert fresh is not old                                # swapped in by the background pass


def test_scores_survive_a_restart(svc, monkeypatch):
    import peaks.web.service as svc_mod

    _trained(svc)
    _settle(svc)
    model = svc._model_name()
    before, _ = svc._taste_scores(model)
    again = svc_mod.Service(svc.cfg)
    monkeypatch.setattr(svc_mod.Service, "_taste_proba_all",
                        lambda self, clf, idx: (_ for _ in ()).throw(AssertionError("recomputed")))
    after, by = again._taste_scores(model)
    assert by == "classifier" and np.allclose(after, before)


def test_new_embeds_keep_scores_aligned(svc, tmp_path):
    _trained(svc)
    _settle(svc)
    model = svc._model_name()
    idx = svc.index(model)
    before, _ = svc._taste_scores(model)
    s0, e0 = idx._key_rows["k5"]
    k5 = before[s0:e0].copy()
    cache = EmbeddingCache(svc.cfg.embedding.cache_dir)
    v = _unit(np.random.default_rng(9).normal(0, 1, (FRAMES, DIM)))
    cache.save("a_new", model, np.arange(FRAMES, dtype=np.float32), v, meta={"scene_id": "777"})
    svc.invalidate_index(model)
    import peaks.web.service as svc_mod
    orig = svc_mod.Service._refresh_scores_bg
    svc_mod.Service._refresh_scores_bg = lambda self, m, p: None     # observe the served (re-aligned) copy
    try:
        idx2 = svc.index(model)
        now, _ = svc._taste_scores(model)
    finally:
        svc_mod.Service._refresh_scores_bg = orig
    s1, e1 = idx2._key_rows["k5"]
    assert now.shape[0] == idx2.size and np.allclose(now[s1:e1], k5)
    a, b = idx2._key_rows["a_new"]
    assert (now[a:b] == 0).all()                            # unknown until the refresh reaches it


def test_tier_board_does_not_wait_for_a_first_score(svc, monkeypatch):
    import time

    import peaks.web.service as svc_mod

    _trained(svc)
    _settle(svc)
    fresh = svc_mod.Service(svc.cfg)
    from pathlib import Path

    for f in Path(svc.cfg.modeling.dir).rglob("*.scores.*"):   # nothing saved to fall back on
        f.unlink()
    monkeypatch.setattr(svc_mod.Service, "_refresh_scores_bg", lambda self, m, p: None)
    monkeypatch.setattr(svc_mod.Service, "_taste_proba_all",
                        lambda self, clf, idx: (time.sleep(30), None)[1])
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: svc.client())
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: svc.client())
    t0 = time.time()
    assert fresh._top_taste_moments_for_scenes(["3", "6"], 2, fresh._model_name()) is None
    assert fresh.taste_scores_for([("3", 1.0)])["scored_by"] == "pending"
    assert time.time() - t0 < 5


def test_catalogue_serves_stale_rows_while_refreshing(svc, monkeypatch):
    import peaks.web.service as svc_mod

    rows = svc._catalogue_all()
    svc._cat_cache = (svc._cat_cache[0] - 10_000, rows)      # well past its freshness
    started = []
    monkeypatch.setattr(svc_mod.Service, "_refresh_catalogue_bg", lambda self: started.append(1))
    assert svc._catalogue_all() is rows and started == [1]


def test_fast_context_matches_the_running_sum():
    rng = np.random.default_rng(0)

    def ref(v, w):
        n = v.shape[0]
        c = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0, dtype=np.float64)])
        i = np.arange(n)
        lo, hi = np.clip(i - w, 0, n), np.clip(i + w + 1, 0, n)
        return np.hstack([v, ((c[hi] - c[lo]) / (hi - lo)[:, None]).astype(np.float32)])

    for n in (1, 2, 3, 4, 7, 60):
        v = rng.normal(0, 1, (n, 16)).astype(np.float32)
        for w in (1, 2, 3):
            assert np.allclose(with_context(v, w), ref(v, w), atol=1e-5)


# --- saves → Légendaire, the trim list, recommendations, library health ------------

def _curated(svc):
    """Ages + saves on the fixture library: scene 30 is new (added 5 days ago),
    everything else was added in 2024; saves on 0 (unreviewed), 15 (upscale),
    1 (rejected) and 3 (already Légendaire); an auto marker on 18."""
    import datetime as dt

    st = svc.client()
    tag = svc.cfg.markers.tag_name
    fresh = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).isoformat()
    for sid, v in st.s.items():
        v["created_at"] = fresh if sid == "30" else "2024-03-01T10:00:00Z"
    st.s["15"].update(rating100=100, o_counter=0)            # Upscale
    st.markers = [
        {"marker_id": "m0", "scene_id": "0", "seconds": 4.0, "end_seconds": None, "title": f"{tag} (saved)"},
        {"marker_id": "m0b", "scene_id": "0", "seconds": 9.0, "end_seconds": None, "title": f"{tag} (saved)"},
        {"marker_id": "m15", "scene_id": "15", "seconds": 3.0, "end_seconds": None, "title": f"{tag} (saved)"},
        {"marker_id": "m1", "scene_id": "1", "seconds": 3.0, "end_seconds": None, "title": f"{tag} (saved)"},
        {"marker_id": "m3", "scene_id": "3", "seconds": 3.0, "end_seconds": None, "title": f"{tag} (saved)"},
        {"marker_id": "a18", "scene_id": "18", "seconds": 3.0, "end_seconds": None, "title": f"{tag} 0.91"},
    ]
    svc._forget_markers()
    svc.invalidate_meta()
    svc._cat_cache = None
    _trained(svc)
    _settle(svc)
    return svc


def test_scene_signals(svc):
    _curated(svc)
    rows = svc._catalogue_all()
    sig = svc._scene_signals(rows)
    assert sig["0"]["saves"] == 2 and sig["15"]["saves"] == 1
    assert sig["18"]["saves"] == 0                            # the legacy auto marker isn't a save
    assert sig["30"]["new"] and not sig["2"]["new"] and sig["2"]["age_days"] > 300
    model = svc._model_name()
    scores, _ = svc._taste_scores(model)
    idx = svc.index(model)
    a, b = idx._key_rows["k7"]
    assert sig["7"]["best"] == pytest.approx(float(scores[a:b].max()), abs=1e-4)


def test_saved_view_and_trim_view(svc):
    _curated(svc)
    d = svc.catalogue(view="saved", limit=100)
    assert [r["scene_id"] for r in d["items"]] == ["0", "15"]   # rejected 1 and Légendaire 3 left out; most saves first
    assert all(r["suggest"]["grade"] == "legendaire" for r in d["items"])
    t = svc.catalogue(view="trim", limit=100)
    ids = [r["scene_id"] for r in t["items"]]
    assert ids and not {"0", "15", "1", "3", "6", "9", "30"} & set(ids)   # saves, rejects, Légendaire, new
    bests = [r["signals"]["best"] for r in t["items"]]
    assert bests == sorted(bests)                               # weakest first
    assert t["grace"]["days"] == 30 and t["grace"]["new_excluded"] >= 1


def test_new_scenes_are_never_pushed_toward_reject(svc):
    _curated(svc)
    rows = {r["scene_id"]: r for r in svc._catalogue_all()}
    sig = svc._scene_signals(list(rows.values()))
    weak = {**sig["30"], "best": -1.0}                          # weakest possible, but new
    assert svc._recommend(rows["30"], {"keeper": 0.1}, weak) is None
    old = {**sig["2"], "best": -1.0}
    assert svc._recommend(rows["2"], {"keeper": 0.1}, old)["grade"] == "reject"
    assert svc._recommend(rows["2"], {"keeper": 0.9}, old) is None   # the tier model still believes in it


def test_saving_promotes_the_scene(svc, monkeypatch):
    _curated(svc)
    st = svc.client()
    monkeypatch.setattr(st, "create_scene_marker", lambda **kw: {"id": "new"}, raising=False)
    from peaks.web.app import create_app
    from fastapi.testclient import TestClient
    import peaks.web.app as app_mod
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(create_app(svc.cfg))
    r = api.post("/api/scene/15/apex", params={"t": 12.0}).json()      # Upscale → Légendaire
    assert r["promoted_from"] == "upscale"
    assert st.s["15"]["rating100"] == 100 and st.s["15"]["o_counter"] == 18
    r = api.post("/api/scene/1/apex", params={"t": 12.0}).json()       # a 1★ stays yours
    assert r["promoted_from"] is None and st.s["1"]["rating100"] == 20
    api.post("/api/library/curation", params={"auto_legendaire_on_save": False})
    r = api.post("/api/scene/2/apex", params={"t": 12.0}).json()
    assert r["promoted_from"] is None
    assert any("auto: saved a moment" in (e.get("source") or "") for e in svc.action_log().tail(20))


def test_library_health_adds_up(svc):
    _curated(svc)
    h = svc.library_health()
    assert h["scenes"] == 60 and h["with_saves"] == 4 and h["saves_total"] == 5
    assert sum(h["save_buckets"].values()) == 60
    assert h["legendaire_candidates"] == 2
    assert h["trim"]["count"] == len(svc.catalogue(view="trim", limit=500)["items"])
    assert h["per_tier"]["legendaire"]["scenes"] == 3


def test_score_threads_follow_the_env(svc, monkeypatch):
    import concurrent.futures as cf

    seen = []
    real = cf.ThreadPoolExecutor

    class Spy(real):
        def __init__(self, max_workers=None, **kw):
            seen.append(max_workers)
            super().__init__(max_workers=max_workers, **kw)
    monkeypatch.setattr(cf, "ThreadPoolExecutor", Spy)
    monkeypatch.setenv("PEAKS_SCORE_THREADS", "7")
    monkeypatch.setattr(type(svc), "_SCORE_CHUNK", 16)     # the fixture is tiny: split it up
    _trained(svc)
    _settle(svc)
    model = svc._model_name()
    clf = svc._taste_model(svc.cfg.markers.tag_name, model)
    svc._taste_proba_all(clf, svc.index(model))
    assert 7 in seen


# --- performer / studio track records ----------------------------------------

def _who_graded(svc):
    """Scenes 0–59 graded: evens Légendaire with performer Ann (id 1) at studio
    Good, odds rejected with Bo (id 2) at studio Bad. Unembedded, unreviewed
    scenes 100 (Ann) and 101 (Bo, Bad) and 102 (a newcomer) arrive."""
    st = svc.client()
    for sid, v in st.s.items():
        i = int(sid)
        good = i % 2 == 0
        v.update(rating100=100 if good else 20, o_counter=18 if good else 0,
                 studio="Good" if good else "Bad",
                 performers=["Ann" if good else "Bo"],
                 performers_detail=[{"id": "1" if good else "2", "name": "Ann" if good else "Bo"}])
    st.s["100"] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                   "fingerprint": "fp100", "studio": "", "performers": ["Ann"],
                   "performers_detail": [{"id": "1", "name": "Ann"}]}
    st.s["101"] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                   "fingerprint": "fp101", "studio": "Bad", "performers": ["Bo"],
                   "performers_detail": [{"id": "2", "name": "Bo"}]}
    st.s["102"] = {"rating100": None, "o_counter": 0, "tag_ids": [], "organized": False,
                   "fingerprint": "fp102", "studio": "Indie", "performers": ["Cy"],
                   "performers_detail": [{"id": "3", "name": "Cy"}]}
    svc.invalidate_meta()
    svc._cat_cache = None
    return svc


def test_tier_model_uses_performer_and_studio_history(svc):
    _who_graded(svc)
    rep = svc.train_tier_model()
    assert rep["trained"] and "who_gain" in rep and "use_who" in rep
    assert rep["fallback"]["trained"] and rep["fallback"]["who_gain"] > 0.1
    rows = svc._catalogue_all()
    preds = svc._tier_predictions(rows)
    assert preds["100"]["from"] == "who" and preds["100"]["tier"] == "legendaire"
    assert preds["101"]["from"] == "who" and preds["101"]["tier"] == "reject"
    assert "from" not in preds["0"]                          # embedded: the full model
    # "Likely keepers" can rank the not-yet-embedded newcomer by who's in it
    likely = [r["scene_id"] for r in svc.catalogue(view="likely")["items"]]
    assert likely.index("100") < likely.index("101")


def test_catalogue_cards_carry_performer_and_studio_reasons(svc):
    _who_graded(svc)
    items = {i["scene_id"]: i for i in svc.catalogue(limit=500)["items"]}
    lines = [x["text"] for x in items["100"]["who"]["lines"]]
    assert "Ann — a favourite: all of 30 graded were Exceptionnelle+" in lines
    ann = items["100"]["who"]["lines"][0]
    assert ann["detail"] == "30 of 30 Exceptionnelle+ · 100%"
    bad = items["101"]["who"]["lines"]
    assert [x["tone"] for x in bad] == ["bad", "bad"]
    assert any(x["text"] == "Bad — usually a miss: you've kept none of 30, none Exceptionnelle+"
               and x["detail"] == "keeps 0% · 0% Exceptionnelle+" for x in bad)
    new = [x["text"] for x in items["102"]["who"]["lines"]]
    assert new == ["New performer: Cy — no history", "New studio: Indie — no history"]
    assert items["100"]["performer_ids"] == ["1"]


def test_library_health_lists_best_and_worst_performers_and_studios(svc):
    _who_graded(svc)
    h = svc.library_health()["who"]
    assert [p["name"] for p in h["performers"]["top"]] == ["Ann"]
    assert [p["name"] for p in h["performers"]["bottom"]] == ["Bo"]
    assert h["studios"]["top"][0] == {**h["studios"]["top"][0], "name": "Good", "n": 30, "keep": 1.0}
    assert h["studios"]["bottom"][0]["name"] == "Bad" and h["studios"]["bottom"][0]["keep"] == 0.0
    assert h["performers"]["count"] == 2 and h["min_graded"] == 3
    assert h["performers"]["top"][0]["verdict"] == "a favourite" and h["studios"]["top"][0]["kept_words"] == "all"
    assert h["performers"]["bottom"][0]["verdict"] == "usually a miss"


def test_new_scenes_are_never_pushed_to_reject_by_who(svc):
    _who_graded(svc)
    svc.train_tier_model()
    import datetime as dt
    st = svc.client()
    st.s["101"]["created_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat()
    svc.invalidate_meta()
    svc._cat_cache = None
    item = next(i for i in svc.catalogue(limit=500)["items"] if i["scene_id"] == "101")
    assert item["pred"]["tier"] == "reject"                   # the forecast is honest…
    assert (item["suggest"] or {}).get("grade") != "reject"   # …but no Reject suggestion


def test_taste_scale_and_best_moment_words(svc):
    before = svc.taste_scale()                                 # saved moments alone: your taste modes
    assert before is None or before["by"] == "modes"
    _trained(svc)
    _settle(svc)
    sc = svc.taste_scale()
    m = sc["moment"]
    assert sc["by"] == "classifier" and m["p50"] <= m["p75"] <= m["p90"] <= m["p95"] <= m["p99"]
    assert sc["scene"]["p50"] >= m["p50"]                      # a scene's BEST beats a typical moment
    assert svc.taste_scale() is sc                             # cached per score array
    rows = svc._catalogue_all()
    sig = svc._scene_signals(rows)
    words = {v["band_word"] for v in sig.values() if v["best"] is not None}
    assert words and words <= {"Standout", "Strong", "Good", "Decent", "So-so", "Weak"}
    top = max((v for v in sig.values() if v["best"] is not None), key=lambda v: v["best"])
    assert top["band"] == "standout"
    health = svc.library_health()
    assert all("with_saves_words" in t for t in health["per_tier"].values())
