/* Galaxy map — a zoomable 2D star-field of the library.
   Reuses app.js globals: $, api, toast, heatColor, openViewer, sendToMegaboard. */
(() => {
  const cv = $("#gx-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const tip = $("#gx-tip");
  const G = {
    space: "dino", color: "taste", data: null,
    scale: 1, ox: 0, oy: 0, W: 0, H: 0, dpr: 1,
    hover: -1, drag: null, sel: null, moved: false,
    imgs: new Map(), inflight: 0, ratings: new Map(),
  };
  const CLUSTER_COLORS = ["#c8a24a","#6ea8fe","#4ac888","#e0604d","#b07de0",
    "#e0a94a","#4ac8c8","#e06da8","#8ac84a","#a0a0ff","#e0c04a","#5ad0a0"];

  // world([0,1]) <-> screen(px)
  const sx = (x) => G.ox + x * G.scale;
  const sy = (y) => G.oy + y * G.scale;
  const wx = (px) => (px - G.ox) / G.scale;
  const wy = (py) => (py - G.oy) / G.scale;

  function resize() {
    const r = $("#gx-stage").getBoundingClientRect();
    G.dpr = window.devicePixelRatio || 1;
    G.W = r.width; G.H = r.height;
    cv.width = Math.round(r.width * G.dpr); cv.height = Math.round(r.height * G.dpr);
    cv.style.width = r.width + "px"; cv.style.height = r.height + "px";
    ctx.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
    draw();
  }
  function fitView() {
    const s = Math.max(50, Math.min(G.W, G.H) - 80);
    G.scale = s; G.ox = (G.W - s) / 2; G.oy = (G.H - s) / 2;
  }

  function colorFor(p) {
    if (G.color === "cluster") return p.c < 0 ? "#3a3a42" : CLUSTER_COLORS[p.c % CLUSTER_COLORS.length];
    if (G.color === "rating") {
      const rt = G.ratings.get(String(p.scene_id));
      return rt == null || rt < 0 ? "#3a3a42" : heatColor(Math.max(0, Math.min(1, rt / 100)));
    }
    return heatColor(p.taste || 0);
  }

  let raf = 0;
  function draw() { if (!raf) raf = requestAnimationFrame(render); }
  function render() {
    raf = 0;
    ctx.fillStyle = "#0b0b0d"; ctx.fillRect(0, 0, G.W, G.H);
    const d = G.data; if (!d || !d.scenes) return;
    const thumbs = G.scale > 2600;
    const rDot = G.scale > 1200 ? 3.5 : 2.2;
    for (let i = 0; i < d.scenes.length; i++) {
      const p = d.scenes[i], px = sx(p.x), py = sy(p.y);
      if (px < -40 || px > G.W + 40 || py < -40 || py > G.H + 40) continue;
      if (thumbs) { drawThumb(p, px, py); continue; }
      ctx.beginPath();
      ctx.arc(px, py, i === G.hover ? rDot + 2.5 : rDot, 0, 6.2832);
      ctx.fillStyle = colorFor(p); ctx.fill();
    }
    if (!thumbs && d.clusters) {
      ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
      for (const c of d.clusters) {
        if (!c.label) continue;
        const px = sx(c.cx), py = sy(c.cy);
        if (px < 0 || px > G.W || py < 0 || py > G.H) continue;
        const w = ctx.measureText(c.label).width + 12;
        ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(px - w / 2, py - 9, w, 17);
        ctx.fillStyle = "#e8e8ea"; ctx.fillText(c.label, px, py + 3);
      }
    }
    if (G.sel) {
      const s = G.sel, x = Math.min(s.x0, s.x1), y = Math.min(s.y0, s.y1);
      const w = Math.abs(s.x1 - s.x0), h = Math.abs(s.y1 - s.y0);
      ctx.fillStyle = "rgba(200,162,74,.12)"; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#c8a24a"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
    }
  }
  function drawThumb(p, px, py) {
    const sz = Math.max(24, Math.min(110, G.scale / 38));
    const w = sz, h = sz * 0.56;
    let img = G.imgs.get(p.key);
    if (img === undefined && G.inflight < 24) {
      img = new Image(); G.imgs.set(p.key, img); G.inflight++;
      img.onload = () => { G.inflight--; draw(); };
      img.onerror = () => { G.imgs.set(p.key, "err"); G.inflight--; };
      img.src = `/api/frame?key=${encodeURIComponent(p.key)}&t=${p.t}&size=128`;
    }
    if (img instanceof Image && img.complete && img.naturalWidth) {
      ctx.drawImage(img, px - w / 2, py - h / 2, w, h);
      ctx.strokeStyle = colorFor(p); ctx.lineWidth = 2;
      ctx.strokeRect(px - w / 2, py - h / 2, w, h);
    } else {
      ctx.beginPath(); ctx.arc(px, py, 3, 0, 6.2832); ctx.fillStyle = colorFor(p); ctx.fill();
    }
  }

  function nearest(px, py, maxd = 12) {
    const d = G.data; if (!d) return -1;
    let best = -1, bd = maxd * maxd;
    for (let i = 0; i < d.scenes.length; i++) {
      const dx = sx(d.scenes[i].x) - px, dy = sy(d.scenes[i].y) - py, dd = dx * dx + dy * dy;
      if (dd < bd) { bd = dd; best = i; }
    }
    return best;
  }

  function showTip(i, cx, cy) {
    if (i < 0) { tip.hidden = true; return; }
    const p = G.data.scenes[i];
    tip.innerHTML =
      `<img src="/api/frame?key=${encodeURIComponent(p.key)}&t=${p.t}&size=160" onerror="this.style.display='none'"/>` +
      `<div class="dim">scene ${p.scene_id ?? "?"} · ${((p.taste || 0) * 100).toFixed(0)}% taste</div>`;
    tip.style.left = Math.min(cx + 14, window.innerWidth - 190) + "px";
    tip.style.top = (cy + 14) + "px";
    tip.hidden = false;
  }

  // --- interactions ---
  cv.addEventListener("mousedown", (e) => {
    G.moved = false;
    if (e.shiftKey) G.sel = { x0: e.offsetX, y0: e.offsetY, x1: e.offsetX, y1: e.offsetY };
    else G.drag = { px: e.offsetX, py: e.offsetY, ox: G.ox, oy: G.oy };
  });
  window.addEventListener("mousemove", (e) => {
    if (!G.data) return;
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    if (G.drag) { G.moved = true; G.ox = G.drag.ox + (px - G.drag.px); G.oy = G.drag.oy + (py - G.drag.py); draw(); return; }
    if (G.sel) { G.moved = true; G.sel.x1 = px; G.sel.y1 = py; draw(); return; }
    if (px < 0 || py < 0 || px > G.W || py > G.H) { if (G.hover !== -1) { G.hover = -1; tip.hidden = true; draw(); } return; }
    const h = nearest(px, py);
    if (h !== G.hover) { G.hover = h; showTip(h, e.clientX, e.clientY); draw(); }
  });
  window.addEventListener("mouseup", () => {
    if (G.sel) { finishSelect(); G.sel = null; draw(); }
    if (G.drag) { G.drag = null; ensureRatings(); }
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const mx = e.offsetX, my = e.offsetY, wxu = wx(mx), wyu = wy(my);
    G.scale = Math.max(50, Math.min(200000, G.scale * f));
    G.ox = mx - wxu * G.scale; G.oy = my - wyu * G.scale;
    draw(); ensureRatings();
  }, { passive: false });
  cv.addEventListener("click", (e) => {
    if (G.moved) return;
    const h = nearest(e.offsetX, e.offsetY);
    if (h < 0) return;
    const p = G.data.scenes[h];
    if (!p.url) return toast("No stream for this scene");
    openViewer({ key: p.key, scene_id: p.scene_id, time: p.t, stream: p.url, title: "" });
  });

  function finishSelect() {
    const s = G.sel; if (!s) return;
    const x0 = Math.min(s.x0, s.x1), x1 = Math.max(s.x0, s.x1);
    const y0 = Math.min(s.y0, s.y1), y1 = Math.max(s.y0, s.y1);
    if (x1 - x0 < 4 && y1 - y0 < 4) return;
    const hits = [];
    for (const p of G.data.scenes) {
      const px = sx(p.x), py = sy(p.y);
      if (px >= x0 && px <= x1 && py >= y0 && py <= y1 && p.scene_id && p.url)
        hits.push({ scene_id: p.scene_id, time: p.t, stream: p.url, score: p.taste || 1, title: "" });
    }
    if (!hits.length) return toast("No scenes in that region");
    toast(`${hits.length} scene${hits.length === 1 ? "" : "s"} → megaboard`);
    sendToMegaboard(hits, "mb_galaxy", "galaxy");
  }

  // rating colour: lazy-fetch ratings for on-screen scenes (capped)
  function ensureRatings() {
    if (G.color !== "rating" || !G.data) return;
    let n = 0;
    for (const p of G.data.scenes) {
      if (n > 150) break;
      const px = sx(p.x), py = sy(p.y);
      if (px < 0 || px > G.W || py < 0 || py > G.H || !p.scene_id) continue;
      if (G.ratings.has(String(p.scene_id))) continue;
      G.ratings.set(String(p.scene_id), -1); n++;
      api("/api/scene/" + p.scene_id)
        .then((m) => { G.ratings.set(String(p.scene_id), m.rating100 ?? 0); draw(); })
        .catch(() => {});
    }
  }

  // --- data / controls ---
  async function pollJob(id) {
    for (;;) {
      const j = await api("/api/jobs/" + id);
      $("#gx-status").textContent = "building… " + ((j.log || []).slice(-1)[0] || "");
      if (j.status !== "running") { if (j.status === "error") throw new Error(j.error); return; }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  async function load(rebuild) {
    $("#gx-status").textContent = rebuild ? "building…" : "loading…";
    try {
      if (rebuild) { const job = await api("/api/galaxy/build?space=" + G.space, { method: "POST" }); await pollJob(job.id); }
      const d = await api("/api/galaxy?space=" + G.space);
      if (!d.built) { G.data = null; $("#gx-status").textContent = "Not built yet — hit “Rebuild map”."; draw(); return; }
      G.data = d; G.imgs.clear(); G.ratings.clear(); fitView();
      const nc = (d.clusters || []).length;
      $("#gx-status").textContent =
        `${d.scenes.length} scenes · ${nc} cluster${nc === 1 ? "" : "s"} · ${d.space}` +
        (d.has_taste ? "" : " · (train taste to colour by it)");
      draw();
    } catch (e) { $("#gx-status").textContent = e.message; }
  }

  $("#gx-space").addEventListener("change", (e) => { G.space = e.target.value; load(false); });
  $("#gx-color").addEventListener("change", (e) => { G.color = e.target.value; ensureRatings(); draw(); });
  $("#gx-rebuild").addEventListener("click", () => load(true));
  $("#gx-reset").addEventListener("click", () => { fitView(); draw(); });
  window.addEventListener("resize", () => { if ($("#galaxy").classList.contains("active")) resize(); });

  let inited = false;
  window.openGalaxy = function () {
    resize();
    if (!inited) { inited = true; load(false); } else { draw(); }
  };
})();
