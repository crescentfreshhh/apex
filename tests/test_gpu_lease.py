"""One owner for the GPU: a job that loads a model holds the lease while it
holds the model (a second waits, never overlaps); text / sound search borrow
the card between a job's batches (a search waits one batch at most, never a
whole job); the text towers are residents that leave the GPU when idle."""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from peaks.gpu import GpuLease, gpu_owned  # noqa: E402


class Model:
    """A fake model: each call is one 'GPU batch', recorded as (who, start, end)."""

    def __init__(self, name, spans, secs=0.05):
        self.name, self.spans, self.secs, self.batch_size = name, spans, secs, 16

    def embed_array(self, x=None):
        t0 = time.monotonic()
        time.sleep(self.secs)
        self.spans.append((self.name, t0, time.monotonic()))
        return x

    embed_text = embed_array


def _overlaps(spans):
    return [(a, b) for i, a in enumerate(spans) for b in spans[i + 1:]
            if a[0] != b[0] and a[1] < b[2] and b[1] < a[2]]


class Job:
    def __init__(self):
        self.cancelled, self.lines = False, []

    def log(self, m):
        self.lines.append(m)


def test_two_senses_never_hold_the_gpu_at_once():
    lease, spans = GpuLease(free=lambda: None), []
    jobs = {k: Job() for k in ("hear", "motion")}

    def run(kind):
        m = lease.guard(Model(kind, spans))
        lease.run_owned(kind, jobs[kind], lambda: [m.embed_array() for _ in range(6)])
    ts = [threading.Thread(target=run, args=(k,)) for k in jobs]
    for t in ts:
        t.start()
        time.sleep(0.01)
    for t in ts:
        t.join(10)
    assert len(spans) == 12 and _overlaps(spans) == []
    assert spans[0][0] == "hear" and spans[-1][0] == "motion"          # one after the other, whole
    assert any("waiting for the GPU — hear has it" in m for m in jobs["motion"].lines)
    assert lease.holder() is None


def test_a_search_answers_within_one_batch_while_an_embed_runs():
    lease, spans = GpuLease(free=lambda: None), []
    m = lease.guard(Model("embed", spans, secs=0.3))
    t = threading.Thread(target=lambda: lease.run_owned("embed", Job(), lambda: [m.embed_array() for _ in range(15)]))
    t.start()
    time.sleep(1.0)
    q = lease.guard(Model("text", spans, secs=0.01), borrow=True)
    t0 = time.monotonic()
    q.embed_text(["a red dress"])
    took = time.monotonic() - t0
    t.join(10)
    assert took < 2.0 and took <= 0.35                                   # at most one batch's wait
    assert _overlaps(spans) == []                                       # between batches, never inside one


def test_the_lease_is_reentrant_and_a_stopped_waiter_never_runs():
    lease, ran = GpuLease(free=lambda: None), []
    with lease.owner("ingest"):
        with lease.owner("embed"):                                      # an ingest's embed stage: same thread
            ran.append("nested")
        held = threading.Event()
        job = Job()

        def wait_then_stop():
            out = lease.run_owned("hear", job, lambda: ran.append("hear ran"))
            ran.append(("returned", out))
            held.set()
        th = threading.Thread(target=wait_then_stop)
        th.start()
        time.sleep(0.3)
        job.cancelled = True                                            # stopped while waiting
        assert held.wait(5)
    assert ran == ["nested", ("returned", None)]
    assert any("stopped while waiting" in m for m in job.lines)


def test_job_methods_take_the_lease_and_report_a_stop_while_waiting():
    class Svc:
        def __init__(self):
            self.gpu = GpuLease(free=lambda: None)

        @gpu_owned("hear", "Hearing")
        def run_hear(self, job=None, limit=None):
            return {"holder": self.gpu.holder()["kind"]}
    s = Svc()
    assert s.run_hear(Job()) == {"holder": "hear"}
    with s.gpu.owner("embed"):
        j = Job()
        out = {}
        th = threading.Thread(target=lambda: out.update(r=s.run_hear(job=j)))
        th.start()
        time.sleep(0.2)
        j.cancelled = True
        th.join(5)
    assert out["r"] == {"stopped": True, "waited_for_gpu": True}


def test_residents_leave_when_idle_or_when_a_job_takes_the_gpu():
    now = {"t": 1000.0}
    lease, gone = GpuLease(clock=lambda: now["t"], free=lambda: None), []
    lease.resident("text-models", lambda: gone.append("text"))
    assert lease.unload_idle(600) == [] and gone == []                  # just used: stays
    now["t"] += 30
    with lease.owner("embed"):
        pass
    assert gone == []                                                   # idle 30 s: a job doesn't evict it
    now["t"] += 60
    with lease.owner("embed"):
        assert gone == ["text"]                                         # idle 90 s: the job's model gets the room
    assert lease.residents() == []


def test_the_service_wires_the_lease(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    assert isinstance(svc.gpu, GpuLease) and svc.gpu is svc.gpu
    spans = []

    class FakeClip:
        def __init__(self, **k):
            pass

        def text_only(self):
            return self

        def embed_text(self, prompts):
            import numpy as np
            spans.append(("text", time.monotonic(), time.monotonic()))
            return np.ones((len(prompts), 24), np.float32)
    import peaks.embedding as E
    monkeypatch.setattr(E, "ClipEmbedder", FakeClip)
    v = svc._clip_text_vector("a red dress")
    assert v.shape == (24,) and spans and svc.gpu.residents() == ["text-models"]
    assert svc.unload_idle_text_models(idle_s=0) == ["clip-text"] and svc.gpu.residents() == []


def test_sound_search_keeps_the_text_tower_only(tmp_path, monkeypatch):
    import numpy as np

    from peaks.embedding import ClapAudioEmbedder

    emb = object.__new__(ClapAudioEmbedder)

    class Model:
        audio_model, audio_projection, text_model = object(), object(), object()
    emb.model = Model()
    emb.text_only()
    assert emb.model.audio_model is None and emb.model.audio_projection is None and emb.model.text_model is not None
    assert np.isfinite(1.0)


def test_vram_is_back_to_baseline_after_a_lease():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from peaks.embedding import free_gpu

    lease = GpuLease(free=free_gpu)
    free_gpu()
    base = torch.cuda.memory_allocated()
    with lease.owner("embed"):
        x = torch.ones((4096, 4096), device="cuda")
        assert torch.cuda.memory_allocated() > base
        del x
    assert torch.cuda.memory_allocated() == base
