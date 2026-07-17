/* Opus / peaks control panel + explorer. Vanilla JS, no build step. */

const $ = (s) => document.querySelector(s);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const toast = (msg, bad) => {
  const t = $("#toast");
  t.textContent = msg; t.className = bad ? "bad" : ""; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 3500);
};

// --- tabs -------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#" + b.dataset.view).classList.add("active");
    if (b.dataset.view === "dashboard") refreshDashboard();
  })
);

// --- dashboard --------------------------------------------------------------
async function refreshDashboard() {
  try {
    const [stats, caps] = await Promise.all([
      api("/api/stats"), api("/api/capabilities"),
    ]);
    $("#conn").textContent = "connected";
    $("#stat-cards").innerHTML = [
      ["Cached scenes", stats.cached_scenes],
      ["Indexed moments", caps.indexed_frames.toLocaleString()],
      ["Model", stats.model],
      ["Device", stats.device],
      ["CLIP / text search", caps.has_clip ? "ready" : "not yet"],
      ["Failed scenes", stats.failures || 0],
      ["Library", stats.library_path],
    ].map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    // surface the failures panel only when there are casualties to retry
    const nf = stats.failures || 0;
    $("#fail-panel").hidden = nf === 0;
    $("#fail-count").textContent = nf ? `· ${nf}` : "";
  } catch (e) {
    $("#conn").textContent = "disconnected"; toast("Cannot reach backend: " + e.message, true);
  }
  if (typeof refreshReels === "function") refreshReels();
}

function wireJob(btn, statusEl, logEl, start, stopBtn) {
  btn.addEventListener("click", async () => {
    btn.disabled = true; statusEl.textContent = "starting…"; logEl.hidden = false; logEl.textContent = "";
    try {
      const job = await start();
      if (stopBtn) {
        stopBtn.hidden = false; stopBtn.disabled = false;
        stopBtn.onclick = async () => {
          stopBtn.disabled = true; statusEl.textContent = "stopping…";
          try { await api("/api/jobs/" + job.id + "/cancel", { method: "POST" }); }
          catch (e) { toast(e.message, true); }
        };
      }
      poll(job.id, statusEl, logEl, btn, stopBtn);
    } catch (e) {
      btn.disabled = false; statusEl.textContent = ""; toast(e.message, true);
    }
  });
}
async function poll(id, statusEl, logEl, btn, stopBtn) {
  const done = () => { btn.disabled = false; if (stopBtn) stopBtn.hidden = true; };
  try {
    const j = await api("/api/jobs/" + id);
    const p = j.progress || {};
    statusEl.textContent = `${j.status} · ${p.done ?? 0}/${p.total ?? "?"} · ${j.elapsed}s`;
    logEl.textContent = (j.log || []).join("\n"); logEl.scrollTop = logEl.scrollHeight;
    if (j.status === "running") return setTimeout(() => poll(id, statusEl, logEl, btn, stopBtn), 1000);
    done();
    if (j.status === "error") toast("Job failed: " + j.error, true);
    else if (j.status === "cancelled") { toast("Stopped."); refreshDashboard(); }
    else { toast("Done: " + JSON.stringify(j.result || {})); refreshDashboard(); }
  } catch (e) { done(); toast(e.message, true); }
}
// --- embed advanced overrides (per-run model / sampling, no restart) --------
let defaultsLoaded = false;
(async () => {
  try {
    const d = await api("/api/defaults");
    $("#adv-model").value = d.model;
    $("#adv-mode").value = d.mode;
    $("#adv-hwaccel").value = d.hwaccel || "";
    $("#adv-interval").value = d.interval;
    $("#adv-workers").value = d.workers;
    $("#adv-timeout").value = d.timeout;
    // scoring thresholds
    $("#adv-high").value = d.high;
    $("#adv-low").value = d.low;
    $("#adv-maxdur").value = d.max_duration;
    $("#adv-reduce").value = d.reduce;
    $("#adv-normalize").value = d.normalize;
    defaultsLoaded = true;
  } catch {}
})();
function wireToggle(btnSel, panelSel, hintSel) {
  $(btnSel).addEventListener("click", () => {
    const a = $(panelSel), open = a.hidden;
    a.hidden = !open;
    if (hintSel) $(hintSel).hidden = !open;
    $(btnSel).textContent = open ? "Advanced ▴" : "Advanced ▾";
  });
}
wireToggle("#toggle-adv", "#embed-adv", "#adv-hint");
wireToggle("#toggle-score-adv", "#score-adv", null);
function embedQuery() {
  // only override once we know the current defaults; selects (incl. hwaccel="")
  // are always sent, numbers only when non-empty (avoids a 422 on blanks)
  if (!defaultsLoaded) return "";
  const qs = new URLSearchParams();
  qs.set("model", $("#adv-model").value);
  qs.set("mode", $("#adv-mode").value);
  qs.set("hwaccel", $("#adv-hwaccel").value);
  for (const [k, sel] of [["interval", "#adv-interval"], ["workers", "#adv-workers"], ["timeout", "#adv-timeout"]]) {
    const v = $(sel).value; if (v !== "") qs.set(k, v);
  }
  return qs.toString();
}
wireJob($("#btn-embed"), $("#embed-status"), $("#embed-log"), () => {
  const q = embedQuery();
  return api("/api/embed" + (q ? "?" + q : ""), { method: "POST" });
}, $("#btn-embed-stop"));
wireJob($("#btn-sync"), $("#sync-status"), $("#sync-log"), () => {
  const prune = $("#sync-prune").checked;
  return api("/api/sync?prune=" + (prune ? "true" : "false"), { method: "POST" });
});
wireJob($("#btn-fix"), $("#fix-status"), $("#fix-log"), () => api("/api/fix", { method: "POST" }), $("#btn-fix-stop"));
$("#btn-fail-list").addEventListener("click", async () => {
  const el = $("#fail-list");
  if (!el.hidden) { el.hidden = true; return; }
  try {
    const { failures } = await api("/api/failures");
    el.textContent = failures.length
      ? failures.map((f) => `scene ${f.scene_id}  [${f.mode}/${f.hwaccel || "off"}/${f.pipeline}]  ${f.path}\n    ${f.error}`).join("\n\n")
      : "(none)";
    el.hidden = false;
  } catch (e) { toast(e.message, true); }
});
wireJob($("#btn-score"), $("#score-status"), $("#score-log"), () => {
  const tag = $("#score-tag").value.trim();
  const write = $("#score-write").checked;
  const qs = new URLSearchParams();
  if (tag) qs.set("tag", tag);
  if (write) qs.set("write", "true");
  if (defaultsLoaded && !$("#score-adv").hidden) {
    if ($("#adv-high").value !== "") qs.set("high", $("#adv-high").value);
    if ($("#adv-low").value !== "") qs.set("low", $("#adv-low").value);
    if ($("#adv-maxdur").value !== "") qs.set("max_duration", $("#adv-maxdur").value);
    qs.set("reduce", $("#adv-reduce").value);
    qs.set("normalize", $("#adv-normalize").value);
  }
  return api("/api/score?" + qs, { method: "POST" });
}, $("#btn-score-stop"));
wireJob($("#btn-playlist"), $("#playlist-status"), $("#playlist-log"), () => {
  const tag = $("#board-tag").value.trim();
  return api("/api/playlist" + (tag ? "?tag=" + encodeURIComponent(tag) : ""), { method: "POST" });
});
wireJob($("#btn-reel"), $("#reel-status"), $("#reel-log"), () => {
  const tag = $("#board-tag").value.trim();
  return api("/api/reel" + (tag ? "?tag=" + encodeURIComponent(tag) : ""), { method: "POST" });
}, $("#btn-reel-stop"));
async function refreshReels() {
  try {
    const { reels } = await api("/api/reels");
    $("#reels").innerHTML = reels.length
      ? "<div class='dim' style='margin:8px 0 4px'>Exported reels</div>" + reels.map((r) =>
          `<a class="reel-item" href="/api/reel/download?name=${encodeURIComponent(r.name)}" download>
             ⬇ ${esc(r.name)} <span class="dim">${(r.bytes / 1e6).toFixed(0)} MB</span></a>`).join("")
      : "";
  } catch {}
}
refreshReels();

// --- explore / search -------------------------------------------------------
function stars(rating100) {
  const filled = Math.round((rating100 || 0) / 20);
  let s = "";
  for (let i = 1; i <= 5; i++)
    s += `<span class="star ${i <= filled ? "on" : ""}" data-r="${i * 20}">★</span>`;
  return s;
}
let lastHits = [];
function renderHits(hits) {
  const g = $("#results");
  lastHits = hits || [];
  $("#btn-board-search").disabled = !lastHits.some((h) => h.scene_id && h.stream);
  if (!hits.length) { g.innerHTML = '<p class="dim">No results.</p>'; return; }
  g.innerHTML = hits.map((h) => {
    const perf = (h.performers || []).slice(0, 3).join(", ");
    const sub = [h.studio, perf].filter(Boolean).join(" · ") || `scene ${h.scene_id ?? "?"}`;
    const title = h.title || `scene ${h.scene_id ?? "?"}`;
    const sid = h.scene_id ?? "";
    return `<div class="tile" data-sid="${sid}">
      <div class="thumbwrap">
        <img loading="lazy" src="${h.thumb}" alt="" onerror="this.style.opacity=.15" />
        <span class="score">${(h.score * 100).toFixed(0)}%</span>
        <span class="t">${fmt(h.time)}</span>
      </div>
      <div class="meta">
        <div class="title" title="${esc(title)}">${esc(title)}</div>
        <div class="sub" title="${esc(sub)}">${esc(sub)}</div>
        <div class="edit">
          <span class="rating" title="rating">${stars(h.rating100)}</span>
          <span class="ospacer"></span>
          <button class=" obtn" title="O-count (click +, shift-click −)">⊙ ${h.o_counter ?? 0}</button>
          <button class="orgbtn ${h.organized ? "on" : ""}" title="organized">✓</button>
        </div>
      </div>
      <div class="actions">
        <button class="thumb up" title="More like this (👍)">👍</button>
        <button class="thumb down" title="Less like this (👎)">👎</button>
        <button data-key="${h.key}" data-t="${h.time}">Find similar</button>
        ${h.stream ? `<button class="play-btn">Play ▸</button>` : ""}
      </div>
    </div>`;
  }).join("");
  g.querySelectorAll("button[data-key]").forEach((b) =>
    b.addEventListener("click", () => similar(b.dataset.key, b.dataset.t)));
  g.querySelectorAll(".tile").forEach((tile, i) => {
    wireTileEdits(tile);
    const h = lastHits[i];
    const open = () => openViewerAt(i);
    tile.querySelector(".play-btn")?.addEventListener("click", open);
    tile.querySelector(".thumbwrap")?.addEventListener("click", open);
    tile.querySelector(".thumb.up")?.addEventListener("click", (e) => thumb(h.key, h.time, 1, h.scene_id, e.currentTarget));
    tile.querySelector(".thumb.down")?.addEventListener("click", (e) => thumb(h.key, h.time, 0, h.scene_id, e.currentTarget));
  });
}

// --- taste: explicit thumbs → trained preference ranking -------------------
async function thumb(key, time, label, sceneId, btn) {
  if (btn) { btn.classList.add("flash"); setTimeout(() => btn.classList.remove("flash"), 600); }
  try {
    const qs = new URLSearchParams({ key, t: (+time).toFixed(2), label });
    if (sceneId) qs.set("scene_id", sceneId);
    const c = await api("/api/label?" + qs, { method: "POST" });
    toast(label ? "👍 More like this — noted" : "👎 Less like this — noted");
    updateTasteUI(c);
  } catch (e) { toast(e.message, true); }
}
function updateTasteUI(c) {
  if (c && c.positive != null) $("#btn-train").textContent = `Train (${c.positive + c.negative})`;
}
$("#btn-train").addEventListener("click", async () => {
  const btn = $("#btn-train"); btn.disabled = true;
  try {
    const s = await api("/api/train", { method: "POST" });
    toast(`Trained on ${s.samples} labels (${s.positives}+)` + (s.cv_auc ? ` · AUC ${s.cv_auc}` : ""));
  } catch (e) { toast(e.message, true); }
  btn.disabled = false;
});
(async () => { try { updateTasteUI(await api("/api/labels")); } catch {} })();

async function patchScene(sid, body) {
  return api(`/api/scene/${sid}`, {
    method: "PATCH", headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}
// editable rating/O/organized, shared by result tiles and the scene viewer
function wireStars(sid, root) {
  root.querySelectorAll(".star").forEach((s) =>
    s.addEventListener("click", async () => {
      try {
        const m = await patchScene(sid, { rating100: +s.dataset.r });
        const rt = root.querySelector(".rating");
        if (rt) { rt.innerHTML = stars(m.rating100); wireStars(sid, root); }
        toast("rating saved");
      } catch (e) { toast(e.message, true); }
    }));
}
function wireSceneEdits(sid, root) {
  if (!sid) return;
  wireStars(sid, root);
  const org = root.querySelector(".orgbtn");
  if (org) org.addEventListener("click", async () => {
    try {
      const m = await patchScene(sid, { organized: !org.classList.contains("on") });
      org.classList.toggle("on", !!m.organized); toast("organized " + (m.organized ? "on" : "off"));
    } catch (e) { toast(e.message, true); }
  });
  const ob = root.querySelector(".obtn");
  if (ob) ob.addEventListener("click", async (e) => {
    try {
      const r = await api(`/api/scene/${sid}/o`, { method: e.shiftKey ? "DELETE" : "POST" });
      ob.textContent = `⊙ ${r.o_counter}`;
    } catch (err) { toast(err.message, true); }
  });
}
function wireTileEdits(tile) { wireSceneEdits(tile.dataset.sid, tile); }

let currentContext = {}; // what produced the current hits → drives the heatmap
const tasteOn = () => ($("#taste-toggle").checked ? "&taste=true" : "");
async function similar(key, t) {
  setActiveView("explore");
  currentContext = { kind: "frame", key, t };
  $("#results").innerHTML = '<p class="dim">Finding similar moments…</p>';
  try { renderHits(await api(`/api/search/similar?key=${key}&t=${t}&top_k=60` + tasteOn())); }
  catch (e) { toast(e.message, true); }
}
async function textSearch() {
  const q = $("#q").value.trim(); if (!q) return;
  currentContext = { kind: "text", q };
  $("#results").innerHTML = '<p class="dim">Searching…</p>';
  try { renderHits(await api("/api/search/text?q=" + encodeURIComponent(q) + "&top_k=60" + tasteOn())); }
  catch (e) { $("#results").innerHTML = ""; toast(e.message, true); }
}

// --- scene viewer (in-app player + score heatmap + save-a-moment) ----------
function sceneStreamUrl(u) {
  try { const x = new URL(u, location.href); x.searchParams.set("start", "0"); return x.toString(); }
  catch { return u; }
}
function heatColor(x) {
  x = Math.max(0, Math.min(1, x));
  const a = [34, 34, 42], b = [200, 162, 74]; // panel → apex gold
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * x));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
async function renderHeat(hit) {
  const heat = $("#viewer-heat"); heat.innerHTML = "";
  const v = $("#viewer-v"); const dur = v.duration || 0;
  let url = "/api/timeline?key=" + encodeURIComponent(hit.key);
  if (currentContext.kind === "text") url += "&q=" + encodeURIComponent(currentContext.q);
  else if (currentContext.kind === "frame")
    url += "&ref_key=" + encodeURIComponent(currentContext.key) + "&ref_t=" + currentContext.t;
  let data; try { data = await api(url); } catch { return; }
  const pts = data.points || []; if (pts.length < 2 || !dur) return;
  const ss = pts.map((p) => p[1]); const mn = Math.min(...ss), mx = Math.max(...ss); const span = (mx - mn) || 1;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < pts.length; i++) {
    const t = pts[i][0], next = i + 1 < pts.length ? pts[i + 1][0] : dur;
    const left = Math.max(0, Math.min(100, (t / dur) * 100));
    const width = Math.max(0.2, ((Math.min(next, dur) - t) / dur) * 100);
    const seg = document.createElement("span");
    seg.style.cssText = `position:absolute;top:0;bottom:0;left:${left}%;width:${width}%;background:${heatColor((pts[i][1] - mn) / span)};`;
    frag.appendChild(seg);
  }
  heat.appendChild(frag);
}
function wireViewerTransport(v) {
  const play = $("#viewer-play"), seek = $("#viewer-seek"), time = $("#viewer-time");
  play.onclick = () => { if (v.paused) v.play().catch(() => {}); else v.pause(); };
  v.onplay = () => (play.textContent = "❚❚");
  v.onpause = () => (play.textContent = "▶");
  let drag = false;
  seek.oninput = () => { drag = true; if (v.duration) time.textContent = `${fmt(seek.value / 1000 * v.duration)} / ${fmt(v.duration)}`; };
  seek.onchange = () => { if (v.duration) v.currentTime = seek.value / 1000 * v.duration; drag = false; };
  v.ontimeupdate = () => {
    if (drag || !v.duration) return;
    seek.value = Math.round(v.currentTime / v.duration * 1000);
    time.textContent = `${fmt(v.currentTime)} / ${fmt(v.duration)}`;
  };
}
async function loadViewerMeta(sid) {
  const edit = $("#viewer-edit");
  $("#viewer-title").textContent = "loading…"; $("#viewer-sub").textContent = ""; edit.innerHTML = "";
  let m = {}; try { m = await api("/api/scene/" + sid); } catch {}
  $("#viewer-title").textContent = m.title || `scene ${sid}`;
  const perf = (m.performers || []).slice(0, 6).join(", ");
  $("#viewer-sub").textContent =
    [m.studio, perf, m.date, (m.tags || []).slice(0, 6).join(", ")].filter(Boolean).join("  ·  ") || "—";
  edit.innerHTML = `<span class="rating">${stars(m.rating100)}</span>
    <button class="obtn" title="O-count (click +, shift-click −)">⊙ ${m.o_counter ?? 0}</button>
    <button class="orgbtn ${m.organized ? "on" : ""}">✓ organized</button>`;
  wireSceneEdits(sid, edit);
}
async function saveMoment(sid, t) {
  if (!sid) return toast("no scene id for this result", true);
  try { await api(`/api/scene/${sid}/apex?t=${(t || 0).toFixed(2)}`, { method: "POST" }); toast("Saved apex @ " + fmt(t)); }
  catch (e) { toast(e.message, true); }
}
let viewerIndex = -1;
let currentHit = null;
let classifyTimer = null;
function openViewerAt(i) {
  if (i < 0 || i >= lastHits.length) return;
  viewerIndex = i;
  openViewer(lastHits[i]);
}
function nextViewer() { if (lastHits.length) openViewerAt((viewerIndex + 1) % lastHits.length); }
function prevViewer() { if (lastHits.length) openViewerAt((viewerIndex - 1 + lastHits.length) % lastHits.length); }
async function similarFromViewer() {
  if (!currentHit) return;
  const v = $("#viewer-v"); const t = v.currentTime || +currentHit.time;
  currentContext = { kind: "frame", key: currentHit.key, t };
  try {
    const hits = await api(`/api/search/similar?key=${currentHit.key}&t=${t.toFixed(2)}&top_k=60` + tasteOn());
    if (!hits.length) return toast("no similar moments found");
    renderHits(hits); openViewerAt(0); toast("More like this moment");
  } catch (e) { toast(e.message, true); }
}
async function classifyCurrent(hit) {
  const el = $("#viewer-clip"); const v = $("#viewer-v");
  let d; try { d = await api(`/api/classify?key=${encodeURIComponent(hit.key)}&t=${(v.currentTime || +hit.time).toFixed(2)}`); }
  catch { el.innerHTML = ""; return; }
  const labs = d.labels || [];
  el.innerHTML = labs.length
    ? `<span class="dim">CLIP sees</span> ` + labs.map(([l, s]) => `<span class="clip-chip" title="${(s * 100).toFixed(0)}% match">${esc(l)}</span>`).join("")
    : "";
}
function openViewer(hit) {
  if (!hit || !hit.stream) return;
  currentHit = hit;
  const V = $("#viewer"), v = $("#viewer-v");
  V.hidden = false;
  const startAt = +hit.time || 0;
  v.src = sceneStreamUrl(hit.stream);
  v.onloadedmetadata = () => {
    try { v.currentTime = Math.min(startAt, (v.duration || startAt) - 0.1); } catch {}
    renderHeat(hit); classifyCurrent(hit);
  };
  v.onseeked = () => { clearTimeout(classifyTimer); classifyTimer = setTimeout(() => classifyCurrent(hit), 350); };
  v.onclick = () => { if (v.paused) v.play().catch(() => {}); else v.pause(); }; // click video = play/pause
  v.play().catch(() => {});
  wireViewerTransport(v);
  $("#viewer-prev").onclick = prevViewer;
  $("#viewer-next").onclick = nextViewer;
  $("#viewer-similar").onclick = similarFromViewer;
  $("#viewer-save").onclick = () => saveMoment(hit.scene_id, v.currentTime);
  $("#viewer-up").onclick = (e) => thumb(hit.key, v.currentTime, 1, hit.scene_id, e.currentTarget);
  $("#viewer-down").onclick = (e) => thumb(hit.key, v.currentTime, 0, hit.scene_id, e.currentTarget);
  try { $("#viewer-stash").href = new URL(hit.stream, location.href).origin + "/scenes/" + hit.scene_id; }
  catch { $("#viewer-stash").href = "#"; }
  loadViewerMeta(hit.scene_id);
}
function closeViewer() {
  const V = $("#viewer"), v = $("#viewer-v");
  try { v.pause(); } catch {}
  v.removeAttribute("src"); v.load(); V.hidden = true;
}
$("#viewer-close").addEventListener("click", closeViewer);
$("#viewer-heat").addEventListener("click", (e) => {
  const v = $("#viewer-v"); if (!v.duration) return;
  const r = e.currentTarget.getBoundingClientRect();
  v.currentTime = ((e.clientX - r.left) / r.width) * v.duration;
});
document.addEventListener("keydown", (e) => {
  if ($("#viewer").hidden) return;
  const v = $("#viewer-v");
  if (e.key === "Escape") closeViewer();
  else if (e.key === "ArrowRight") nextViewer();
  else if (e.key === "ArrowLeft") prevViewer();
  else if (e.key === " ") { e.preventDefault(); if (v.paused) v.play().catch(() => {}); else v.pause(); }
  else if (e.key === "s" || e.key === "S") saveMoment(currentHit?.scene_id, v.currentTime);
});
$("#btn-text").addEventListener("click", textSearch);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") textSearch(); });

// hand the current results to the megaboard: each tile starts at its matched
// moment (the stream URL already carries start=<time>). Passed via localStorage
// (same origin) so we don't clobber the saved apex playlist.json.
const BOARD_CLIP_SECONDS = 20;
$("#btn-board-search").addEventListener("click", () => {
  const apexes = lastHits
    .filter((h) => h.scene_id && h.stream)
    .map((h) => ({
      scene_id: h.scene_id,
      start: +h.time,
      end: +h.time + BOARD_CLIP_SECONDS,
      duration: BOARD_CLIP_SECONDS,
      url: h.stream,
      score: h.score ?? 1,
      title: h.title || "",
    }));
  if (!apexes.length) return;
  localStorage.setItem("mb_search", JSON.stringify({ tag: "search", count: apexes.length, apexes }));
  window.open("/megaboard/?src=search", "_blank");
});

function setActiveView(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.view === name));
  document.querySelectorAll(".view").forEach((x) => x.classList.toggle("active", x.id === name));
}
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// hint about CLIP availability for text search
(async () => {
  try {
    const caps = await api("/api/capabilities");
    if (!caps.has_clip)
      $("#explore-hint").textContent = "text search needs a CLIP embed pass";
  } catch {}
})();

refreshDashboard();
