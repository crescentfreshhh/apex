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

# Self-contained login screen (no /static needed — those assets are gated too),
# served for any unauthenticated non-API request when a password is configured.
_LOGIN_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Opus — sign in</title><link rel="icon" href="data:,">
<style>
  :root{--bg:#0b0b0d;--panel:#141417;--panel2:#1b1b20;--fg:#e8e8ea;--dim:#8a8a92;
    --line:#2a2a30;--accent:#c8a24a;--bad:#e0604d}
  *{box-sizing:border-box}html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);display:flex;align-items:center;
    justify-content:center;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  form{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:32px 28px;width:300px;text-align:center}
  .brand{font-weight:700;color:var(--accent);letter-spacing:.5px;font-size:26px}
  .sub{color:var(--dim);margin-bottom:20px;font-size:13px}
  input{width:100%;background:var(--panel2);color:var(--fg);border:1px solid var(--line);
    border-radius:8px;padding:11px 12px;font-size:15px;margin-bottom:12px}
  button{width:100%;background:var(--accent);color:#1a1400;border:none;border-radius:8px;
    padding:11px;font-size:15px;font-weight:600;cursor:pointer}
  button:hover{filter:brightness(1.06)}
  #err{color:var(--bad);font-size:13px;min-height:18px;margin-top:10px}
</style></head><body>
<form id="f" autocomplete="on">
  <div class="brand">Opus</div><div class="sub">peaks — sign in</div>
  <input id="pw" type="password" placeholder="Password" autofocus autocomplete="current-password">
  <button type="submit">Unlock</button>
  <div id="err"></div>
</form>
<script>
  const f=document.getElementById('f'),pw=document.getElementById('pw'),err=document.getElementById('err');
  f.addEventListener('submit',async e=>{e.preventDefault();err.textContent='';
    try{const r=await fetch('/api/login',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({password:pw.value})});
      if(r.ok){location.reload();}else{err.textContent='Wrong password';pw.value='';pw.focus();}}
    catch{err.textContent='Something went wrong';}});
</script></body></html>"""


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


def _collection_model():
    from pydantic import BaseModel

    class CollectionIn(BaseModel):
        name: str
        apexes: list = []
        query: str | None = None      # how it was built (for a future refresh)
        params: dict | None = None

    return CollectionIn


CollectionIn = _collection_model()


def _vocab_model():
    from pydantic import BaseModel

    class VocabIn(BaseModel):
        vocab: str = ""

    return VocabIn


VocabIn = _vocab_model()


def _models_model():
    from pydantic import BaseModel

    class ModelsIn(BaseModel):
        dino_model: str | None = None
        clip_model: str | None = None

    return ModelsIn


ModelsIn = _models_model()


def _login_model():
    from pydantic import BaseModel

    class LoginIn(BaseModel):
        password: str = ""

    return LoginIn


LoginIn = _login_model()


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


def _hit_light(service: Service, h) -> dict:
    """A hit with just what the grid/board need — no Stash metadata lookup, so
    an unbounded result set costs no per-scene network calls."""
    return {
        "scene_id": h.scene_id,
        "key": h.key,
        "time": round(h.time, 2),
        "score": round(h.score, 4),
        "thumb": f"/api/frame?key={h.key}&t={h.time:g}",
        "stream": service.stream_url(h.scene_id, start=h.time) if h.scene_id else None,
        "title": "",
    }


def _search_payload(service: Service, hits, enrich: int = 300) -> dict:
    """Return the FULL result set, but only fetch Stash metadata for the first
    `enrich` hits (the on-screen preview). The rest stay lightweight so "all
    matches" (no cap) never means thousands of Stash lookups."""
    hits = list(hits)
    items = _hit_payload(service, hits[:enrich]) if enrich else []
    items += [_hit_light(service, h) for h in hits[enrich:]]
    return {"total": len(hits), "items": items}


def create_app(cfg=None):
    import secrets
    import time as _time

    from fastapi import Cookie, FastAPI, HTTPException, Query, Request
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        Response,
    )
    from fastapi.staticfiles import StaticFiles

    service = Service(cfg)
    jobs = JobManager()
    app = FastAPI(title="peaks", docs_url="/api/docs")

    # --- auth (optional password gate over the whole app) -------------------
    _auth_pw = service.cfg.auth.password
    _auth_on = bool(_auth_pw)
    _session_ttl = max(60.0, service.cfg.auth.session_hours * 3600.0)
    _sessions: dict[str, float] = {}  # token -> expiry epoch (sliding)

    def _session_valid(token: str | None) -> bool:
        if not token:
            return False
        exp = _sessions.get(token)
        if exp is None:
            return False
        if _time.time() > exp:
            _sessions.pop(token, None)
            return False
        _sessions[token] = _time.time() + _session_ttl  # refresh on activity
        return True

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):
        if not _auth_on or request.url.path == "/api/login":
            return await call_next(request)
        if _session_valid(request.cookies.get("peaks_session")):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "authentication required"}, status_code=401)
        return HTMLResponse(_LOGIN_HTML, status_code=200)

    @app.middleware("http")
    async def _revalidate_ui(request: Request, call_next):
        """Make the browser revalidate the UI (HTML/JS/CSS) on every load so a
        redeploy is picked up without a manual hard-refresh — the recurring
        "stale app.js" trap. no-cache still allows efficient 304s when unchanged."""
        resp = await call_next(request)
        p = request.url.path
        if p == "/" or p.startswith("/static/") or p.startswith("/megaboard"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.post("/api/login")
    def login(body: LoginIn):
        if not _auth_on:
            return {"ok": True}
        if secrets.compare_digest(body.password or "", _auth_pw):
            token = secrets.token_urlsafe(32)
            _sessions[token] = _time.time() + _session_ttl
            resp = JSONResponse({"ok": True})
            resp.set_cookie(
                "peaks_session", token, max_age=int(_session_ttl),
                httponly=True, samesite="lax", path="/",
            )
            return resp
        _time.sleep(0.5)  # gentle brute-force damper
        raise HTTPException(401, "wrong password")

    @app.post("/api/logout")
    def logout(peaks_session: str | None = Cookie(default=None)):
        _sessions.pop(peaks_session or "", None)
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("peaks_session", path="/")
        return resp

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
            "auth": _auth_on,
        }

    # --- jobs ---------------------------------------------------------------

    @app.get("/api/defaults")
    def defaults():
        """Current embed/sampling settings, so the Advanced form pre-fills to
        what the container is configured with."""
        s, e, sc = service.cfg.sampling, service.cfg.embedding, service.cfg.scoring
        return {
            "model": e.model,
            "mode": s.mode,
            "interval": s.interval_seconds,
            "hwaccel": s.hwaccel,
            "pipeline": s.pipeline,
            "workers": e.workers,
            "timeout": s.scene_timeout,
            "tag": service.cfg.markers.tag_name,
            "high": sc.high,
            "low": sc.low,
            "reduce": sc.reduce,
            "max_duration": sc.max_duration,
            "normalize": sc.normalize,
        }

    @app.post("/api/embed")
    def start_embed(
        limit: int = Query(0),
        model: str | None = None,
        mode: str | None = None,
        interval: float | None = None,
        hwaccel: str | None = None,
        pipeline: str | None = None,
        workers: int | None = None,
        timeout: float | None = None,
    ):
        # sampling knobs actually supplied; absent ones fall back to config.
        # The active backbone/variant come from the saved model settings.
        sampling = {
            k: v
            for k, v in dict(
                mode=mode, interval=interval, hwaccel=hwaccel,
                pipeline=pipeline, workers=workers, scene_timeout=timeout,
            ).items()
            if v is not None
        }
        # `model` may be a comma-list ("dino,clip") to queue passes back-to-back
        models = [m.strip() for m in model.split(",") if m.strip()] if model else []
        if len(models) > 1:
            target = lambda j: service.run_embed_multi(j, models=models, limit=limit, **sampling)  # noqa: E731
        else:
            overrides = dict(sampling)
            if models:
                overrides["model"] = models[0]
            target = lambda j: service.run_embed(j, limit=limit, **overrides)  # noqa: E731
        try:
            job = jobs.start("embed", target)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/score")
    def start_score(
        tag: str | None = None,
        write: bool = False,
        high: float | None = None,
        low: float | None = None,
        reduce: str | None = None,
        max_duration: float | None = None,
        normalize: str | None = None,
    ):
        try:
            job = jobs.start(
                "score",
                lambda j: service.run_score(
                    j, tag=tag, write=write, high=high, low=low, reduce=reduce,
                    max_duration=max_duration, normalize=normalize,
                ),
            )
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/playlist")
    def start_playlist(tag: str | None = None):
        tags = [tag] if tag else None
        try:
            job = jobs.start("playlist", lambda j: service.run_playlist(j, tags=tags))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/reel")
    def start_reel(tag: str | None = None, limit: int = Query(0)):
        try:
            job = jobs.start("reel", lambda j: service.export_reel(j, tag=tag, limit=limit))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.get("/api/reels")
    def list_reels():
        return {"reels": service.reels()}

    # --- megaboard sources --------------------------------------------------

    @app.get("/api/board/sources")
    def board_sources():
        return service.board_sources()

    @app.get("/api/board/scenes")
    def board_scenes(refresh: bool = False):
        return {"scenes": service.scene_pool(refresh=refresh)}

    @app.get("/api/board/apexes")
    def board_apexes(tag: str | None = None):
        return service.board_apexes(tag)

    @app.post("/api/collection")
    def save_collection(body: CollectionIn):
        if not body.name.strip() or not body.apexes:
            raise HTTPException(400, "need a name and at least one moment")
        return service.save_collection(
            body.name.strip(), body.apexes, query=body.query, params=body.params
        )

    @app.get("/api/collections")
    def collections():
        return {"collections": service.list_collections()}

    @app.get("/api/collection")
    def get_collection(name: str):
        c = service.load_collection(name)
        if not c:
            raise HTTPException(404, "no such collection")
        return c

    @app.delete("/api/collection")
    def delete_collection(name: str):
        r = service.delete_collection(name)
        if not r["removed"]:
            raise HTTPException(404, "no such collection")
        return r

    @app.post("/api/collection/export")
    def export_collection(name: str, limit: int = Query(200)):
        try:
            job = jobs.start("reel", lambda j: service.export_collection(j, name=name, limit=limit))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/collection/rename")
    def rename_collection(body: dict):
        try:
            return service.rename_collection(body.get("name", ""), body.get("new_name", ""))
        except FileNotFoundError:
            raise HTTPException(404, "no such collection")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/reel/download")
    def download_reel(name: str):
        path = service.reel_path(name)
        if not path:
            raise HTTPException(404, "no such reel")
        return FileResponse(path, media_type="video/mp4", filename=name)

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
    def search_similar(
        key: str | None = None, t: float = 0.0, scene_id: str | None = None,
        top_k: int = 60, taste: bool = False, per_scene: int = 3,
        min_score: float = 0.0, enrich: int = 300,
    ):
        # the megaboard knows scene_id, not the cache key — resolve it (same path
        # /api/classify uses) so "more like this moment" works from a board tile.
        if key is None and scene_id is not None:
            key = service._key_for_scene(scene_id, service._model_name())
        if key is None:
            return {"total": 0, "items": []}
        tk = None if min_score > 0 else top_k       # min_score → return ALL matches
        hits = service.search_by_frame(
            key, t, top_k=tk, taste=taste,
            per_scene=(per_scene if per_scene > 0 else None),
            min_score=(min_score if min_score > 0 else None),
        )
        return _search_payload(service, hits, enrich=enrich)

    @app.get("/api/search/text")
    def search_text(
        q: str, top_k: int = 60, taste: bool = False, per_scene: int = 3,
        min_score: float = 0.0, neg_weight: float = 0.5, enrich: int = 300,
    ):
        if not service.has_clip_index():
            raise HTTPException(
                400, "no CLIP index — run an embed pass with PEAKS_MODEL=clip"
            )
        tk = None if min_score > 0 else top_k       # min_score → return ALL matches
        try:
            hits = service.search_text(
                q, top_k=tk, taste=taste, neg_weight=neg_weight,
                per_scene=(per_scene if per_scene > 0 else None),
                min_score=(min_score if min_score > 0 else None),
            )
        except ImportError as exc:
            raise HTTPException(500, f"CLIP unavailable: {exc}")
        return _search_payload(service, hits, enrich=enrich)

    # --- taste (explicit thumbs → personalized ranking) ---------------------

    @app.get("/api/labels")
    def labels(profile: str | None = None):
        counts = service.label_counts(profile)
        counts["has_model"] = service.has_taste(profile)
        return counts

    @app.post("/api/label")
    def add_label(
        t: float, label: int, key: str | None = None,
        scene_id: str | None = None, profile: str | None = None,
    ):
        # the megaboard rates by scene_id+time (it has no cache key) — resolve it
        # the same way "more like this moment" does.
        if key is None and scene_id is not None:
            key = service._key_for_scene(scene_id, service._model_name())
        if key is None:
            raise HTTPException(400, "need a key or a resolvable scene_id")
        res = service.add_label(key, t, label, profile=profile, scene_id=scene_id)
        # hands-off training: once enough new ratings pile up, retrain the taste
        # model in the background so "Train now" is optional. One train at a time.
        res["autotrain"] = False
        if service.autotrain_due(profile):
            try:
                jobs.start("train", lambda j: service.train_taste(profile=profile))
                service.reset_labels_since_train()
                res["autotrain"] = True
            except RuntimeError:
                pass  # a train is already running; the next rating retries
        return res

    @app.post("/api/taste/delete")
    def delete_taste(
        within_minutes: float | None = None,
        profile: str | None = None,
        purge_apexes: bool = False,
    ):
        """Erase learned taste — all of it (within_minutes omitted/0) or just the
        last N minutes. Retrains from what's left, or drops the model. With
        `purge_apexes` on a full wipe, also deletes the ⭐ apex markers in Stash."""
        return service.delete_taste(
            profile=profile, within_minutes=within_minutes, purge_apexes=purge_apexes
        )

    @app.post("/api/train")
    def train_taste(profile: str | None = None, model: str | None = None):
        try:
            out = service.train_taste(profile=profile, model=model)
            service.reset_labels_since_train()
            return out
        except Exception as exc:  # noqa: BLE001 — surface training issues to the UI
            raise HTTPException(400, str(exc))

    # --- "For You": taste-centroid recommendations + active-learning trainer -

    @app.get("/api/foryou")
    def foryou(top_k: int = 60, recent: int = 0, rebuild: bool = False):
        r = service.recommend(top_k=top_k, recent=recent, rebuild=rebuild)
        return {
            "items": _hit_payload(service, r["hits"]),
            "sources": r["sources"], "total": r["total"], "model": r["model"],
            "reranked": r["reranked"], "diversified": r.get("diversified", False),
        }

    @app.get("/api/foryou/next")
    def foryou_next():
        # deliberately lightweight (no scene_meta fetch) so rapid swiping never
        # blocks on Stash — the viewer loads full metadata on click.
        hit = service.next_uncertain()
        if not hit:
            return {"item": None}
        sid = hit["scene_id"]
        return {"item": {
            "scene_id": sid, "key": hit["key"], "time": hit["time"], "score": 0.0,
            "thumb": f"/api/frame?key={hit['key']}&t={hit['time']:g}",
            "stream": service.stream_url(sid, start=hit["time"]) if sid else None,
            "title": "", "performers": [],
        }}

    @app.get("/api/foryou/sample")
    def foryou_sample(count: int = 10):
        # Taste Picker: a batch of random frames to eyeball and tap. Lightweight
        # (no scene_meta) like /api/foryou/next so the collage loads instantly.
        items = []
        for hit in service.sample_frames(count=count):
            sid = hit["scene_id"]
            items.append({
                "scene_id": sid, "key": hit["key"], "time": hit["time"], "score": 0.0,
                "thumb": f"/api/frame?key={hit['key']}&t={hit['time']:g}",
                "stream": service.stream_url(sid, start=hit["time"]) if sid else None,
            })
        return {"items": items}

    @app.get("/api/foryou/words")
    def foryou_words(recent: int = 0, top_k: int = 8):
        return service.taste_words(top_k=top_k, recent=recent)

    @app.get("/api/taste/metrics")
    def taste_metrics(threshold: float | None = None):
        """Taste-score distribution + how many moments/scenes meet your bar."""
        return service.taste_metrics(threshold=threshold)

    @app.get("/api/foryou/board")
    def foryou_board(count: int = 400, exclude: str = "", min_score: float = 0.0):
        """The endless For You megaboard's supply — one peak per scene across your
        whole vetted library (≥1 moment for every scene), scored by your trained
        model when you have one. `min_score` is an optional tightener (0 = off →
        every scene's peak; raise it → only your strongest). `exclude` marches the
        stream forward. Returns the coverage numbers (`scenes`/`moments` matching at
        this floor, and how the score was computed) so the board can show them."""
        seen = {s for s in exclude.split(",") if s}
        r = service.board_pool(
            count=count, per_scene=4, exclude=seen, min_score=min_score,
        )
        return {
            "items": _hit_payload(service, r["hits"]),
            "scenes": r.get("scenes", 0), "moments": r.get("moments", 0),
            "scored_by": r.get("scored_by"),
        }

    @app.get("/api/board/performer")
    def board_performer(scene_id: str, count: int = 300):
        """Moments from other scenes featuring this scene's lead performer —
        the megaboard's 'more from this actress' pivot."""
        r = service.performer_board(scene_id, count=count)
        return {"performer": r["performer"], "items": _hit_payload(service, r["hits"])}

    @app.get("/api/performer/best")
    def performer_best(
        name: str | None = None, id: str | None = None, scene_id: str | None = None,
        count: int = 200, per_scene: int = 6, query: str | None = None,
    ):
        """A performer's best moments (taste-ranked, or by a focus `query`) — the
        preview you save as a collection."""
        r = service.performer_best(
            name=name, performer_id=id, scene_id=scene_id,
            count=count, per_scene=per_scene, query=query or None,
        )
        return {"performer": r["performer"], "items": _hit_payload(service, r["hits"])}

    @app.get("/api/performers")
    def performers(sort: str = "moments", refresh: bool = False):
        rows = service.performer_stats(rebuild=refresh)
        keyf = {
            "moments": lambda r: r.get("moments", 0),
            "scenes": lambda r: r.get("scenes", 0),
            "taste": lambda r: (r.get("taste_best") or 0),
            "affinity": lambda r: (r.get("affinity") or 0),
            "engagement": lambda r: r.get("o_counter", 0),
        }.get(sort, lambda r: r.get("moments", 0))
        return {"performers": sorted(rows, key=keyf, reverse=True)}

    @app.get("/api/performer/detail")
    def performer_detail(id: str | None = None, name: str | None = None):
        d = service.performer_detail(name=name, performer_id=id)
        d["items"] = _hit_payload(service, d.pop("hits"))
        return d

    @app.get("/api/performers/similar")
    def performers_similar(id: str, k: int = 8):
        return {"similar": service.similar_performers(id, k=k)}

    @app.get("/api/performer/roulette")
    def performer_roulette():
        return service.performer_roulette()

    @app.post("/api/performers/hall-of-fame")
    def hall_of_fame(top_n: int = 10):
        return service.hall_of_fame(top_n=top_n)

    @app.get("/api/performer/{pid}/image")
    def performer_image(pid: str):
        try:
            got = service.client().performer_image(pid)
        except Exception:  # noqa: BLE001
            got = None
        if not got:
            raise HTTPException(404, "no image")
        data, ctype = got
        return Response(content=data, media_type=ctype)

    @app.get("/api/radio")
    def radio(exclude: str = "", count: int = 30):
        """Next batch for Taste Radio: top taste moments minus the scene_ids in
        `exclude` (comma-separated), so the stream never replays what you've seen."""
        seen = {s for s in exclude.split(",") if s}
        r = service.recommend(top_k=count, exclude=seen)
        return {"items": _hit_payload(service, r["hits"])}

    # --- galaxy map ---------------------------------------------------------

    @app.get("/api/galaxy")
    def galaxy(space: str = "dino"):
        return service.get_galaxy(space)

    @app.post("/api/galaxy/build")
    def galaxy_build(space: str = "dino"):
        try:
            job = jobs.start("galaxy", lambda j: service.build_galaxy(j, space))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.post("/api/autotag")
    def autotag(top: int = Query(5), min_score: float = Query(0.0), limit: int = Query(0)):
        try:
            job = jobs.start("autotag", lambda j: service.auto_tag(j, top=top, min_score=min_score, limit=limit))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return job.as_dict()

    @app.get("/api/duplicates")
    def duplicates(key: str, t: float, threshold: float = 0.9, model: str | None = None):
        return _hit_payload(service, service.find_duplicates(key, t, threshold=threshold, model=model))

    @app.get("/api/classify")
    def classify(t: float, key: str | None = None, scene_id: str | None = None, top_k: int = 6):
        try:
            return service.classify_frame(key=key, time=t, scene_id=scene_id, top_k=top_k)
        except Exception:  # noqa: BLE001 — classification is cosmetic
            return {"labels": []}

    @app.get("/api/vocab")
    def get_vocab():
        return service.get_vocab()

    @app.post("/api/vocab")
    def save_vocab(body: VocabIn):
        return service.save_vocab(body.vocab)

    @app.get("/api/models")
    def get_models():
        return service.get_models()

    @app.post("/api/models")
    def save_models(body: ModelsIn):
        try:
            return service.save_models(
                dino_model=body.dino_model, clip_model=body.clip_model
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/timeline")
    def timeline(
        key: str,
        model: str | None = None,
        q: str | None = None,
        ref_key: str | None = None,
        ref_t: float | None = None,
    ):
        return service.scene_timeline(
            key, model=model, text=q, ref_key=ref_key, ref_t=ref_t
        )

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

    @app.post("/api/scene/{scene_id}/apex")
    def save_apex(scene_id: str, t: float, end: float | None = None, tag: str | None = None):
        try:
            return {"marker": service.create_apex(scene_id, t, end=end, tag=tag)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Stash marker create failed: {exc}")

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
