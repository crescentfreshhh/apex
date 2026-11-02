"""Faces: the pure detector/aligner helpers, and who-is-in-this-scene from a
fake face model (no onnxruntime, no ffmpeg)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_tier_drain_orphans import _orphan_svc  # noqa: E402

pytest.importorskip("sklearn")


def test_similarity_transform_recovers_a_rotation_scale_and_shift():
    from peaks.faces import ARCFACE_DST, similarity_transform

    a = np.deg2rad(20)
    R = 1.7 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = ARCFACE_DST @ R.T + [30, -12]
    m = similarity_transform(src, ARCFACE_DST)
    back = src @ m[:, :2].T + m[:, 2]
    assert np.allclose(back, ARCFACE_DST, atol=1e-3)


def test_nms_and_scrfd_decoding():
    from peaks.faces import decode_scrfd, nms

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]
    outs = []
    for st in (8, 16, 32):
        n = (640 // st) ** 2 * 2
        outs.append(np.zeros((n, 1), np.float32))
    for st in (8, 16, 32):
        outs.append(np.ones(((640 // st) ** 2 * 2, 4), np.float32) * 4)       # a 8·st box around the anchor
    for st in (8, 16, 32):
        outs.append(np.zeros(((640 // st) ** 2 * 2, 10), np.float32))
    outs[2][10] = 0.9                                                         # one face, stride 32, cell 5
    faces = decode_scrfd(outs, scale=0.5)
    assert len(faces) == 1 and faces[0].score == pytest.approx(0.9)
    x1, y1, x2, y2 = faces[0].box
    assert x2 - x1 == pytest.approx(2 * 4 * 32 / 0.5)                          # back in the full frame's pixels


IDS = {"1": 0, "2": 1, "3": 2, "stranger": 3}


class FakeFaces:
    """A frame's faces: each scene shows the identity planted for it."""

    dim = 512

    def __init__(self, plan):
        self.plan, self.rng = plan, np.random.default_rng(0)

    def faces(self, img):
        from peaks.faces import Face

        sid, t = img
        out = []
        for who in self.plan.get(sid, []):
            v = np.zeros(512, np.float32)
            v[IDS[who]] = 1.0
            v += 0.15 * self.rng.standard_normal(512).astype(np.float32) / np.sqrt(512)
            out.append(Face((0, 0, 100, 100), 0.9, np.zeros((5, 2)), v / np.linalg.norm(v)))
        return out


def _wire(svc, monkeypatch, plan):
    import peaks.web.service as svc_mod

    emb = FakeFaces(plan)
    monkeypatch.setattr(svc_mod.Service, "_face_embedder", lambda self: emb)
    monkeypatch.setattr(svc_mod.Service, "_face_frame", lambda self, path, t: (path.rsplit("/", 1)[-1].split(".")[0], t))
    monkeypatch.setattr(svc_mod.Service, "_stash_performers", lambda self, refresh=False: [
        {"id": "1", "name": "Ava", "has_image": False}, {"id": "2", "name": "Bea", "has_image": False},
        {"id": "3", "name": "Cleo", "has_image": False}])
    return emb


def test_faces_name_an_orphan_and_find_uncredited_scenes(tmp_path, monkeypatch):
    svc, stash = _orphan_svc(tmp_path, monkeypatch)
    for sid in stash.s:
        stash.s[sid]["path"] = f"/data/{sid}.mp4"
        stash.s[sid]["duration"] = 300
    plan = {}
    for sid, v in stash.s.items():
        ids = [p["id"] for p in v.get("performers_detail") or [] if p["id"] != "9"]
        plan[sid] = ids
    plan["u1"] = ["3"]                       # an orphan whose face is Cleo's (it *looks* like Bea's scenes)
    credited_ava = next(s for s, v in stash.s.items() if any(p["id"] == "1" for p in v.get("performers_detail") or []))
    plan[credited_ava] = ["1", "3"]          # Cleo is in one of Ava's scenes, uncredited
    _wire(svc, monkeypatch, plan)
    assert svc.faces_todo()[0]["scene_id"] in {"u1", "u2", "a1"}            # orphans first
    r = svc.run_faces(limit=500)
    assert r["seen"] >= 30 and not r["failed"] and not svc.faces_todo()
    m = svc.face_matches("u1")["matches"]
    assert m[0]["name"] == "Cleo" and m[0]["strength"] == "strong" and not m[0]["credited"]
    s = svc.orphan_suggestions("u1")["suggestions"]
    assert s[0]["name"] == "Cleo" and s[0]["face"]["faces"] >= 2                # by face, ahead of the look-alike
    unc = svc.face_uncredited("3")["scenes"]
    assert credited_ava in [x["scene_id"] for x in unc] and "u1" in [x["scene_id"] for x in unc]
    svc.orphan_not_them("u1", "3")
    assert all(x["id"] != "3" for x in svc.face_matches("u1")["matches"])      # "not her" sticks
    st = svc.faces_status()
    assert st["seen"] == st["embedded"] and st["last_run"]["done"] >= 30


def test_faces_take_a_turn_in_the_gpu_lane(tmp_path):
    import peaks.web.app as app_mod

    assert "faces" in app_mod.PAUSABLE and "faces" in app_mod.GPU_JOBS and "faces" in app_mod.RESUME_ORDER
