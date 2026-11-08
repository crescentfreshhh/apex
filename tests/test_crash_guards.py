"""The crashes while grading: a background model fit ran native BLAS/LAPACK on
OpenBLAS's worker pool beside the web server's threads. The guards: the image
keeps OpenBLAS single-threaded (the start script must never raise it), and every
heavy fit — tier retrain, taste, the moment model, the criteria models — takes
the one process-wide training lock, so two never run at once."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(__file__).resolve().parents[1]


def _thread_env(env: dict) -> dict:
    text = (ROOT / "docker" / "entrypoint.sh").read_text()
    block = text[text.index('if [ -z "$OMP_NUM_THREADS" ]'):]
    block = block[:block.index("\n\n")]                         # the thread caps, nothing else
    out = subprocess.run(["sh", "-c", block + '\necho "$OMP_NUM_THREADS $OPENBLAS_NUM_THREADS"'],
                         env={"PATH": os.environ["PATH"], **env}, capture_output=True, text=True, check=True)
    omp, blas = out.stdout.split()
    return {"omp": omp, "blas": blas}


def test_the_start_script_keeps_openblas_single_threaded():
    assert _thread_env({})["blas"] == "1"                          # nothing set: one thread
    assert _thread_env({"OPENBLAS_NUM_THREADS": "1"})["blas"] == "1"   # the Dockerfile's value stands
    assert _thread_env({"OPENBLAS_NUM_THREADS": "4"})["blas"] == "4"   # your container variable wins
    assert int(_thread_env({})["omp"]) >= 1
    docker = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^ENV OPENBLAS_NUM_THREADS=1$", docker, re.M)


def test_every_heavy_fit_holds_the_one_training_lock(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    held = {}
    monkeypatch.setattr(type(svc), "_train_moment_model", lambda self: held.setdefault("moment", self._train_lock._is_owned()) and {})
    monkeypatch.setattr(type(svc), "_crit_build_locked", lambda self: held.setdefault("criteria", self._train_lock._is_owned()) and {})
    svc.train_moment_model()
    svc._crit_build()
    assert held == {"moment": True, "criteria": True}
    with svc._train_lock:                                          # re-entrant: a retrain may read the criteria models
        svc._crit_build()
