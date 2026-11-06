"""Performer pictures from the Faces scan: only a frame holding exactly one face —
hers — is used, cropped 2:3 around it; her Stash photo and your choices stay; an
automatic cover/frame picture is upgraded once the scan has found her."""

from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from peaks.config import Config  # noqa: E402
from peaks.web.performers import portrait_box  # noqa: E402
from peaks.web.service import Service  # noqa: E402


def _unit(i):
    v = np.zeros(512, np.float32)
    v[i] = 1.0
    return v


HER, OTHER = _unit(0), _unit(1)


def _svc(tmp_path, monkeypatch, scenes, has_image=False, gallery=True):
    cfg = Config()
    cfg.modeling.dir = str(tmp_path / "models")
    s = Service(cfg)
    rows = [{"scene_id": sid, "tier": "legendaire", "title": f"S{sid}", "path": f"/v/{sid}.mp4",
             "performer_ids": ["7"]} for sid in scenes]
    monkeypatch.setattr(Service, "_her_scenes", lambda self, pid: rows)
    monkeypatch.setattr(Service, "_catalogue_all", lambda self, refresh=False: rows)
    monkeypatch.setattr(Service, "_scene_faces", lambda self, sid: scenes.get(str(sid)))
    monkeypatch.setattr(Service, "_face_gallery", lambda self: {
        "M": np.stack([HER]) if gallery else np.zeros((0, 512), np.float32),
        "owners": np.asarray(["7"] if gallery else []), "owners_set": set(), "photo": set()})
    monkeypatch.setattr(Service, "_perf_info", lambda self, pid: {"has_image": has_image})
    monkeypatch.setattr(Service, "_her_frames", lambda self, pid, n=8: [])
    grabbed = []

    def frame(self, path, t):
        grabbed.append((path, t))
        img = Image.new("RGB", (960, 540), (20, 20, 20))
        img.paste((250, 200, 180), (400, 100, 560, 260))         # "her face" at the box below
        return img
    monkeypatch.setattr(Service, "_face_frame", frame)
    monkeypatch.setattr(Service, "scene_cover", lambda self, sid: (_jpeg((0, 0, 255)), "image/jpeg"))
    s._grabbed = grabbed
    return s


def _jpeg(color):
    b = BytesIO()
    Image.new("RGB", (640, 360), color).save(b, format="JPEG")
    return b.getvalue()


def _faces(*frames):
    """frames: [(t, [(vec, box, score), ...]), ...] → what the scan keeps."""
    ts, vs, boxes, scores = [], [], [], []
    for t, fs in frames:
        for v, b, sc in fs:
            ts.append(t)
            vs.append(v)
            boxes.append(list(b))
            scores.append(sc)
    return np.asarray(ts, np.float32), np.stack(vs), {"boxes": boxes, "scores": scores}


BOX = (400, 100, 560, 260)


def test_only_a_frame_with_one_face_and_it_hers_is_used(tmp_path, monkeypatch):
    scenes = {
        "a": _faces((10.0, [(HER, BOX, 0.9), (OTHER, (100, 100, 260, 260), 0.9)])),   # two people
        "b": _faces((20.0, [(OTHER, BOX, 0.95)])),                                    # one face, not hers
        "c": _faces((30.0, [(HER, (0, 0, 40, 40), 0.9)]), (31.0, [(HER, BOX, 0.5)])),  # too small / unsure
        "d": _faces((5.0, [(HER, BOX, 0.9), (OTHER, (0, 0, 100, 100), 0.9)]), (40.0, [(HER, BOX, 0.92)])),
    }
    s = _svc(tmp_path, monkeypatch, scenes)
    got = s._face_portraits("7")
    assert [(p["scene_id"], p["t"]) for p in got] == [("d", 40.0)]


def test_the_crop_is_2_by_3_around_her_face():
    x1, y1, x2, y2 = portrait_box(960, 540, BOX)
    w, h = x2 - x1, y2 - y1
    assert abs(w / h - 2 / 3) < 0.01 and h == 400
    assert x1 <= 400 and x2 >= 560 and y1 <= 100 and y2 >= 260               # the whole face is in
    assert abs((x1 + x2) / 2 - 480) <= 1                                     # centred on it
    x1, y1, x2, y2 = portrait_box(960, 540, (900, 400, 960, 540))            # at the frame's corner
    assert x2 <= 960 and y2 <= 540 and x1 >= 0 and y1 >= 0


def test_her_face_beats_a_cover_and_her_stash_photo_stays_first(tmp_path, monkeypatch):
    scenes = {"d": _faces((40.0, [(HER, BOX, 0.92)]))}
    s = _svc(tmp_path, monkeypatch, scenes)
    kinds = [x["kind"] for x in s._photo_sources("7")]
    assert kinds[0] == "face" and kinds.index("face") < kinds.index("cover")
    assert s._build_performer_photo("7") and s._photo_srcs()["7"] == "face"
    img = Image.open(BytesIO(s.performer_photo("7")))
    assert abs(img.width / img.height - 2 / 3) < 0.02
    assert s._grabbed == [("/v/d.mp4", 40.0)]
    s2 = _svc(tmp_path / "x", monkeypatch, scenes, has_image=True)
    assert [x["kind"] for x in s2._photo_sources("7")][0] == "stash"


def test_no_face_data_falls_back_as_before(tmp_path, monkeypatch):
    s = _svc(tmp_path, monkeypatch, {"d": None}, gallery=False)          # a scene, never face-scanned
    assert s._build_performer_photo("7") and s._photo_srcs()["7"] == "cover"


def test_an_automatic_cover_is_upgraded_but_a_chosen_picture_never(tmp_path, monkeypatch):
    scenes = {"d": _faces((40.0, [(HER, BOX, 0.92)]))}
    s = _svc(tmp_path, monkeypatch, scenes, gallery=False)
    s._build_performer_photo("7")
    assert s._photo_srcs()["7"] == "cover"
    monkeypatch.setattr(Service, "_face_gallery", lambda self: {"M": np.stack([HER]), "owners": np.asarray(["7"])})
    assert s.upgrade_performer_photos() == 1 and s._photo_srcs()["7"] == "face"
    # a picture you chose stays yours
    s.choose_performer_photo("7", {"kind": "cover", "scene_id": "d"})
    before = s.performer_photo("7")
    assert s.upgrade_performer_photos() == 0 and s.performer_photo("7") == before


def test_choosing_her_face_saves_and_rebuilds(tmp_path, monkeypatch):
    scenes = {"d": _faces((40.0, [(HER, BOX, 0.92)])), "e": _faces((12.5, [(HER, BOX, 0.9)]))}
    s = _svc(tmp_path, monkeypatch, scenes)
    opts = [o for o in s.performer_photo_options("7")["options"] if o["kind"] == "face"]
    assert {o["scene_id"] for o in opts} == {"d", "e"} and all("face-crop" in o["thumb"] for o in opts)
    out = s.choose_performer_photo("7", {"kind": "face", "scene_id": "e", "t": 12.5})
    assert out["photo"] and s._photo_choices()["7"] == {"kind": "face", "scene_id": "e", "t": 12.5}
    assert s._grabbed[-1] == ("/v/e.mp4", 12.5)
    assert s.face_crop("7", "e", 12.5) and s.face_crop("7", "e", 99.0) is None
    chosen = [o for o in s.performer_photo_options("7")["options"] if o.get("chosen")]
    assert [(o["kind"], o["scene_id"]) for o in chosen] == [("face", "e")]
