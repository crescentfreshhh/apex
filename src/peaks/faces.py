"""Faces: who is in a frame, by face — InsightFace's buffalo_l models (SCRFD
detector + ArcFace recogniser) run through onnxruntime, so no compiler is needed
in the container. Weights are InsightFace's (non-commercial licence — fine for a
personal library) and download once into the model cache.

`FaceEmbedder.faces(image)` → [Face(box, score, kps, vec)] with `vec` a unit
512-d ArcFace embedding. Two faces of the same person sit around cosine 0.5+;
strangers below ~0.3.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_FILE, REC_FILE = "det_10g.onnx", "w600k_r50.onnx"
DET_SIZE = 640
DET_THRESH = 0.5
NMS_THRESH = 0.4
MIN_FACE = 40             # px: smaller faces are too blurry to recognise reliably
DIM = 512

# where the five landmarks sit in a 112×112 ArcFace crop (insightface's template)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


@dataclass
class Face:
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in the frame
    score: float
    kps: np.ndarray                            # (5, 2) landmarks
    vec: np.ndarray | None = None              # unit ArcFace embedding

    @property
    def size(self) -> float:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


# --- pure helpers (tested without any model) -------------------------------------------------

def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2×3 matrix of the similarity transform (rotation, uniform scale, shift)
    taking `src` points onto `dst` in the least-squares sense (Umeyama)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    r = u @ sign @ vt
    scale = np.trace(np.diag(sig) @ sign) / max(float((s ** 2).sum() / len(src)), 1e-12)
    t = md - scale * r @ ms
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align(img, kps: np.ndarray, size: int = 112):
    """The face as ArcFace wants it: 112×112, eyes / nose / mouth in place."""
    from PIL import Image

    m = similarity_transform(kps, ARCFACE_DST * (size / 112.0))
    full = np.vstack([m, [0, 0, 1]])
    inv = np.linalg.inv(full)[:2]                     # PIL maps output → input
    return img.transform((size, size), Image.AFFINE, tuple(inv.flatten()), resample=Image.BILINEAR)


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float = NMS_THRESH) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


def decode_scrfd(outs: list[np.ndarray], scale: float, thresh: float = DET_THRESH,
                 size: int = DET_SIZE) -> list[Face]:
    """SCRFD's raw outputs (scores, boxes, landmarks per stride 8/16/32, two
    anchors a cell) → faces in the original frame's pixels."""
    strides, fmc, na = (8, 16, 32), 3, 2
    boxes_all, scores_all, kps_all = [], [], []
    for i, st in enumerate(strides):
        sc = outs[i].reshape(-1)
        bb = outs[i + fmc].reshape(-1, 4) * st
        kp = outs[i + 2 * fmc].reshape(-1, 10) * st
        h = w = size // st
        cy, cx = np.mgrid[:h, :w]
        centers = (np.stack([cx, cy], -1).reshape(-1, 2) * st).astype(np.float32)
        centers = np.repeat(centers, na, axis=0)
        pos = np.where(sc >= thresh)[0]
        if not len(pos):
            continue
        c = centers[pos]
        b = bb[pos]
        boxes_all.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], -1))
        k = kp[pos].reshape(-1, 5, 2) + c[:, None, :]
        kps_all.append(k)
        scores_all.append(sc[pos])
    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all) / scale
    kps = np.concatenate(kps_all) / scale
    scores = np.concatenate(scores_all)
    return [Face(tuple(map(float, boxes[i])), float(scores[i]), kps[i]) for i in nms(boxes, scores)]


# --- the models -----------------------------------------------------------------------------

def model_dir() -> Path:
    base = os.environ.get("PEAKS_FACE_MODELS") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "insightface", "buffalo_l")
    return Path(base)


def ensure_models(log=print) -> Path:
    """buffalo_l's detector + recogniser on disk (downloaded once, ~280 MB)."""
    d = model_dir()
    if (d / DET_FILE).exists() and (d / REC_FILE).exists():
        return d
    import requests

    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "buffalo_l.zip.part"
    log("faces: downloading InsightFace buffalo_l (once, ~280 MB)…")
    with requests.get(BUFFALO_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base in (DET_FILE, REC_FILE):
                (d / base).write_bytes(z.read(name))
    tmp.unlink(missing_ok=True)
    if not ((d / DET_FILE).exists() and (d / REC_FILE).exists()):
        raise RuntimeError("the buffalo_l download didn't contain the face models")
    return d


class FaceEmbedder:
    """SCRFD + ArcFace through onnxruntime — on the GPU when onnxruntime-gpu can
    find CUDA (torch's bundled CUDA libraries are offered to it), else the CPU."""

    dim = DIM

    def __init__(self, log=print):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("faces need onnxruntime (pip install onnxruntime-gpu)") from exc
        d = ensure_models(log)
        try:
            import torch  # noqa: F401 — loads CUDA/cuDNN that onnxruntime-gpu can then use
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception:  # noqa: BLE001
            pass
        want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(d / DET_FILE), so, providers=want)
        self.rec = ort.InferenceSession(str(d / REC_FILE), so, providers=want)
        self.device = "cuda" if "CUDAExecutionProvider" in self.det.get_providers() else "cpu"
        self._din = self.det.get_inputs()[0].name
        self._rin = self.rec.get_inputs()[0].name

    def detect(self, img) -> list[Face]:
        w, h = img.size
        scale = DET_SIZE / max(w, h)
        small = img.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), np.float32)
        a = np.asarray(small, np.float32)
        canvas[: a.shape[0], : a.shape[1]] = a
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self._din: blob})
        return [f for f in decode_scrfd(outs, scale) if f.size >= MIN_FACE]

    def embed(self, img, faces: list[Face]) -> list[Face]:
        if not faces:
            return []
        crops = np.stack([np.asarray(align(img.convert("RGB"), f.kps), np.float32) for f in faces])
        blob = ((crops - 127.5) / 127.5).transpose(0, 3, 1, 2)
        v = self.rec.run(None, {self._rin: blob})[0]
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        for f, x in zip(faces, v):
            f.vec = x.astype(np.float32)
        return faces

    def faces(self, img) -> list[Face]:
        return self.embed(img, self.detect(img))
