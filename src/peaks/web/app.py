"""FastAPI app: control panel + explorer.

create_app(cfg) wires a Service (shared core) and a JobManager (background
embed/score) to HTTP routes and serves the static frontend. An optional
scheduler thread runs recurring incremental embeds (e.g. CPU passes for newly
added scenes once the GPU is gone).
"""

from __future__ import annotations

import threading
from pathlib import Path

from .jobs import JobManager
from .service import Service

STATIC_DIR = Path(__file__).parent / "static"


def _scene_edit_model():
    """Request body for scene edits (module-level so FastAPI resolves it as a
    body, not a query param)."""
    from pydantic import BaseModel

    class SceneEdit(BaseModel):
        rating100: int | None = None
        organized: bool | None = None
        title: str | None = None
        date: str | None = None
        details: str | None = None

    return SceneEdit


SceneEdit = _scene_edit_model()


def _hit_payload(service: Service, hits) -> list[dict]:
    meta = service.scene_meta([h.scene_id for h in hits if h.scene_id])
    out = []
    for h in hits:
        m = meta.get(h.scene_id, {}) if h.scene_id else {}
        out.append(
            {
                "scene_id": h.scene_id,
                "key": h.key,
                "time": round(h.time, 2),
                "score": round(h.score, 4),
                "thumb": f"/api/frame?key={h.key}&t={h.time:g}",
                "stream": (
                    service.stream_url(h.scene_id, start=h.time) if h.scene_id else None
                ),
                "title": m.get("title", ""),
                "studio": m.get("studio", ""),
                "performers": m.get("performers", []),
                "date": m.get("date", ""),
                "tags": m.get("tags", []),
                "rating100": m.get("rating100"),
                "o_counter": m.get("o_counter", 0),
                "organized": m.get("organized", False),
            }
        )
    return out


def create_app(cfg=None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    service = Service(cfg)
    jobs = JobManager()
    app = FastAPI(title="peaks", docs_url="/api/docs")

    # --- meta ---------------------------------------------------------------

    @app.get("/api/stats")
    def stats():
        return service.stats()

    @app.get("/api/capabilities")
    def capabilities():
        idx = service.index()
        return {
            "indexed_frames": idx.size,
            "has_clip": service.has_clip_index(),
            "embed_running": jobs.running("embed") is not None,
            "score_running": jobs.running("score") is not None,
        }

    # --- jobs ---------------------------------------------------------------

    @app.post("/api/embed")
    def start_embed(limit: int = Query(0)):
        try:
            job = jobs.start("embed", lambda j: service.run_embed(j, limit=limit))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/score")
    def start_score(tag: str | None = None, write: bool = False):
        try:
            job = jobs.start(
                "score", lambda j: service.run_score(j, tag=tag, write=write)
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/sync")
    def start_sync(prune: bool = True):
        try:
            job = jobs.start("sync", lambda j: service.run_sync(j, prune=prune))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.get("/api/failures")
    def failures():
        from ..failures import failure_log_for

        return {"failures": failure_log_for(service.cfg).entries()}

    @app.post("/api/fix")
    def start_fix(limit: int = Query(0)):
        try:
            job = jobs.start("fix", lambda j: service.run_fix(j, limit=limit))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.get("/api/jobs")
    def list_jobs():
        return [j.as_dict(log_tail=1) for j in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.as_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        job.request_cancel()
        return job.as_dict(log_tail=1)

    # --- search -------------------------------------------------------------

    @app.get("/api/search/similar")
    def search_similar(key: str, t: float, top_k: int = 60):
        return _hit_payload(service, service.search_by_frame(key, t, top_k=top_k))

    @app.get("/api/search/text")
    def search_text(q: str, top_k: int = 60):
        if not service.has_clip_index():
            raise HTTPException(
                400, "no CLIP index — run an embed pass with PEAKS_MODEL=clip"
            )
        try:
            hits = service.search_text(q, top_k=top_k)
        except ImportError as exc:
            raise HTTPException(500, f"CLIP unavailable: {exc}")
        return _hit_payload(service, hits)

    # --- scene metadata (two-way sync with Stash) ---------------------------

    @app.get("/api/scene/{scene_id}")
    def get_scene(scene_id: str):
        return service.scene_meta([scene_id]).get(scene_id, {})

    @app.patch("/api/scene/{scene_id}")
    def edit_scene(scene_id: str, body: SceneEdit):
        fields = body.model_dump(exclude_none=True)
        if not fields:
            raise HTTPException(400, "no fields to update")
        try:
            return service.update_scene(scene_id, **fields)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Stash update failed: {exc}")

    @app.post("/api/scene/{scene_id}/o")
    def add_o(scene_id: str):
        try:
            return {"o_counter": service.add_o(scene_id)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Stash update failed: {exc}")

    @app.delete("/api/scene/{scene_id}/o")
    def remove_o(scene_id: str):
        try:
            return {"o_counter": service.remove_o(scene_id)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Stash update failed: {exc}")

    # --- thumbnails ---------------------------------------------------------

    @app.get("/api/frame")
    def frame(key: str, t: float, size: int = 320):
        path = service.path_for_key(key)
        if not path:
            raise HTTPException(404, "unknown scene key")
        try:
            data = service.frame_jpeg(path, t, size=size)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"could not decode frame: {exc}")
        return Response(content=data, media_type="image/jpeg")

    # --- frontend -----------------------------------------------------------

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # the standalone megaboard (its playlist.json is generated by `peaks
    # playlist`); mount it if its directory exists.
    import os as _os

    board_dir = Path(_os.environ.get("PEAKS_WEBAPP_DIR", "webapp"))
    if board_dir.is_dir():
        app.mount("/megaboard", StaticFiles(directory=board_dir, html=True), name="board")

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return JSONResponse({"peaks": "running", "ui": "not built"})

    # --- scheduler (recurring incremental embeds) ---------------------------

    watch_seconds = service.cfg.schedule.embed_seconds
    if watch_seconds and watch_seconds > 0:
        _start_scheduler(app, service, jobs, watch_seconds)

    app.state.service = service
    app.state.jobs = jobs
    return app


def _start_scheduler(app, service: Service, jobs: JobManager, seconds: float):
    stop = threading.Event()

    sync = service.cfg.schedule.sync
    prune = service.cfg.schedule.prune

    def _embed_then_sync(job):
        stats = service.run_embed(job)
        if sync:
            job.log("--- reconciling cache with Stash (sync) ---")
            stats["sync"] = service.run_sync(job, prune=prune)
        return stats

    def _loop():
        # small initial delay so startup isn't slammed
        if stop.wait(30):
            return
        while not stop.wait(seconds):
            if jobs.running("embed") is None:
                try:
                    jobs.start("embed", _embed_then_sync)
                except RuntimeError:
                    pass  # a run is already going

    threading.Thread(target=_loop, daemon=True, name="peaks-scheduler").start()
    app.state._scheduler_stop = stop
