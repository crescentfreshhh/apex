/* Opus / peaks control panel + explorer. Vanilla JS, no build step. */

const $ = (s) => document.querySelector(s);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.reload(); throw new Error("session expired"); }
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
document.querySelectorAll(".tab[data-view]").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#" + b.dataset.view).classList.add("active");
    if (b.dataset.view === "dashboard") refreshDashboard();
    if (b.dataset.view === "foryou") openForYou();
  })
);

// --- dashboard --------------------------------------------------------------
async function refreshDashboard() {
  try {
    const [stats, caps] = await Promise.all([
      api("/api/stats"), api("/api/capabilities"),
    ]);
    $("#conn").textContent = "connected";
    const dino = (stats.dino_model || "").replace("dinov2_", "") || stats.model;
    const clip = `${stats.clip_model || "?"} · ${stats.clip_cached ? stats.clip_cached.toLocaleString() + " cached" : "not embedded"}`;
    $("#stat-cards").innerHTML = [
      ["Cached scenes", stats.cached_scenes],
      ["Indexed moments", caps.indexed_frames.toLocaleString()],
      ["DINOv2 backbone", dino],
      ["CLIP model", clip],
      ["Device", stats.device],
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
  if (typeof refreshCollections === "function") refreshCollections();
  if (typeof reattachJobs === "function") reattachJobs();
}

function wireJob(btn, statusEl, logEl, start, stopBtn) {
  btn.addEventListener("click", async () => {
    btn.disabled = true; statusEl.textContent = "starting…"; logEl.hidden = false; logEl.textContent = "";
    try {
      const job = await start();
      tracked.add(job.id);
      wireStop(stopBtn, statusEl, job.id);
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

// --- reattach to jobs already running on the server (survives page refresh
//     and shows up on any device — the server, not the tab, owns the job) ----
const JOB_PANELS = {
  embed: { btn: "#btn-embed", status: "#embed-status", log: "#embed-log", stop: "#btn-embed-stop" },
  score: { btn: "#btn-score", status: "#score-status", log: "#score-log", stop: "#btn-score-stop" },
  sync: { btn: "#btn-sync", status: "#sync-status", log: "#sync-log" },
  fix: { btn: "#btn-fix", status: "#fix-status", log: "#fix-log", stop: "#btn-fix-stop" },
  reel: { btn: "#btn-reel", status: "#reel-status", log: "#reel-log", stop: "#btn-reel-stop" },
  autotag: { btn: "#btn-autotag", status: "#autotag-status", log: "#autotag-log", stop: "#btn-autotag-stop" },
  playlist: { btn: "#btn-playlist", status: "#playlist-status", log: "#playlist-log" },
};
const tracked = new Set(); // job ids we're already polling in this tab
function wireStop(stopBtn, statusEl, id) {
  if (!stopBtn) return;
  stopBtn.hidden = false; stopBtn.disabled = false;
  stopBtn.onclick = async () => {
    stopBtn.disabled = true; statusEl.textContent = "stopping…";
    try { await api("/api/jobs/" + id + "/cancel", { method: "POST" }); }
    catch (e) { toast(e.message, true); }
  };
}
async function reattachJobs() {
  let jobs;
  try { jobs = await api("/api/jobs"); } catch { return; }
  for (const j of jobs) {
    if (j.status !== "running" || tracked.has(j.id)) continue;
    const panel = JOB_PANELS[j.kind];
    if (!panel) continue;
    tracked.add(j.id);
    const btn = $(panel.btn), statusEl = $(panel.status), logEl = $(panel.log);
    const stopBtn = panel.stop ? $(panel.stop) : null;
    if (btn) btn.disabled = true;
    if (logEl) logEl.hidden = false;
    wireStop(stopBtn, statusEl, j.id);
    poll(j.id, statusEl, logEl, btn, stopBtn);
  }
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
// --- active models (persisted DINOv2 backbone + CLIP variant) ---------------
async function loadModels() {
  try {
    const m = await api("/api/models");
    $("#sel-dino").value = m.dino_model;
    $("#sel-clip").value = m.clip_model;
    const bits = [];
    if (m.dino_saved) bits.push("DINO override");
    if (m.clip_saved) bits.push("CLIP override");
    $("#models-status").textContent = bits.length ? "saved: " + bits.join(" · ") : "container defaults";
  } catch {}
}
async function saveModels(patch) {
  try {
    const m = await api("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    $("#sel-dino").value = m.dino_model;
    $("#sel-clip").value = m.clip_model;
    toast("Active model saved — re-embed to populate its cache");
    refreshDashboard();
    loadModels();
  } catch (e) {
    toast("Couldn't save model: " + e.message, true);
    loadModels();
  }
}
$("#sel-dino")?.addEventListener("change", (e) => saveModels({ dino_model: e.target.value }));
$("#sel-clip")?.addEventListener("change", (e) => saveModels({ clip_model: e.target.value }));
loadModels();

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
wireJob($("#btn-autotag"), $("#autotag-status"), $("#autotag-log"), () => {
  const top = $("#autotag-top").value || 5;
  return api("/api/autotag?top=" + top, { method: "POST" });
}, $("#btn-autotag-stop"));
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
// --- CLIP vocabulary editor -------------------------------------------------
$("#btn-vocab-edit").addEventListener("click", async () => {
  const ed = $("#vocab-editor");
  if (!ed.hidden) { ed.hidden = true; return; }
  try {
    const d = await api("/api/vocab");
    $("#vocab-text").value = d.vocab;
    $("#vocab-status").textContent = `${d.count} terms${d.from_file ? "" : " (defaults)"}`;
    ed.hidden = false;
  } catch (e) { toast(e.message, true); }
});
$("#btn-vocab-reset").addEventListener("click", async () => {
  try { $("#vocab-text").value = (await api("/api/vocab")).vocab; } catch {}
});
$("#btn-vocab-save").addEventListener("click", async () => {
  try {
    const r = await api("/api/vocab", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ vocab: $("#vocab-text").value }),
    });
    toast(`Saved ${r.count} terms`); $("#vocab-status").textContent = `${r.count} terms`;
  } catch (e) { toast(e.message, true); }
});

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
function renderHits(hits, container) {
  const g = container || $("#results");
  lastHits = hits || [];
  const playable = lastHits.some((h) => h.scene_id && h.stream);
  $("#btn-board-search").disabled = !playable;
  $("#btn-save-collection").disabled = !playable;
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
        <button class="apex-btn" title="Mark this moment as an apex in Stash">⚑ Apex</button>
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
    tile.querySelector(".apex-btn")?.addEventListener("click", (e) => { saveMoment(h.scene_id, h.time); e.currentTarget.classList.add("flash"); });
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
let classifyInterval = null;
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
async function dupesFromViewer() {
  if (!currentHit) return;
  const v = $("#viewer-v"); const t = v.currentTime || +currentHit.time;
  try {
    const hits = await api(`/api/duplicates?key=${encodeURIComponent(currentHit.key)}&t=${t.toFixed(2)}`);
    if (!hits.length) return toast("no near-duplicates found");
    currentContext = { kind: "frame", key: currentHit.key, t };
    closeViewer(); renderHits(hits); setActiveView("explore");
    toast(`${hits.length} near-duplicate${hits.length > 1 ? "s" : ""}`);
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
  // live "CLIP sees": reclassify the moment as it plays
  clearInterval(classifyInterval);
  classifyInterval = setInterval(() => {
    if (!$("#viewer").hidden && !v.paused) classifyCurrent(currentHit);
  }, 1600);
  v.play().catch(() => {});
  wireViewerTransport(v);
  $("#viewer-prev").onclick = prevViewer;
  $("#viewer-next").onclick = nextViewer;
  $("#viewer-similar").onclick = similarFromViewer;
  $("#viewer-dupes").onclick = dupesFromViewer;
  $("#viewer-save").onclick = () => saveMoment(hit.scene_id, v.currentTime);
  $("#viewer-up").onclick = (e) => thumb(hit.key, v.currentTime, 1, hit.scene_id, e.currentTarget);
  $("#viewer-down").onclick = (e) => thumb(hit.key, v.currentTime, 0, hit.scene_id, e.currentTarget);
  try { $("#viewer-stash").href = new URL(hit.stream, location.href).origin + "/scenes/" + hit.scene_id; }
  catch { $("#viewer-stash").href = "#"; }
  loadViewerMeta(hit.scene_id);
}
function closeViewer() {
  const V = $("#viewer"), v = $("#viewer-v");
  clearInterval(classifyInterval);
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
function hitsToApexes(hits) {
  return (hits || []).filter((h) => h.scene_id && h.stream).map((h) => ({
    scene_id: h.scene_id, start: +h.time, end: +h.time + BOARD_CLIP_SECONDS,
    duration: BOARD_CLIP_SECONDS, url: h.stream, score: h.score ?? 1, title: h.title || "",
  }));
}
$("#btn-board-search").addEventListener("click", () => {
  const apexes = hitsToApexes(lastHits);
  if (!apexes.length) return;
  localStorage.setItem("mb_search", JSON.stringify({ tag: "search", count: apexes.length, apexes }));
  window.open("/megaboard/?src=search", "_blank");
});
$("#btn-save-collection").addEventListener("click", async () => {
  const apexes = hitsToApexes(lastHits);
  if (!apexes.length) return;
  const name = prompt("Name this collection:");
  if (!name) return;
  try {
    const r = await api("/api/collection", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, apexes }),
    });
    toast(`Saved "${r.name}" (${r.count} moments)`); refreshCollections();
  } catch (e) { toast(e.message, true); }
});
async function refreshCollections() {
  try {
    const { collections } = await api("/api/collections");
    $("#collections").innerHTML = collections.length
      ? "<div class='dim' style='margin:8px 0 4px'>Collections</div>" + collections.map((c) =>
          `<a class="reel-item" href="/megaboard/?collection=${encodeURIComponent(c.safe)}" target="_blank">▶ ${esc(c.name)} <span class="dim">${c.count}</span></a>`).join("")
      : "";
  } catch {}
}
refreshCollections();

// --- For You: taste-centroid recommender + active-learning swipe trainer ----
function recentN() { return $("#foryou-recent")?.checked ? 40 : 0; }

async function loadForYou(rebuild) {
  const grid = $("#foryou-results");
  grid.innerHTML = '<p class="dim">Reading your taste…</p>';
  try {
    const qs = new URLSearchParams({ top_k: 80, recent: recentN(), rebuild: rebuild ? "true" : "false" });
    const d = await api("/api/foryou?" + qs);
    if (!d.items.length) {
      grid.innerHTML = "";
      $("#foryou-status").textContent =
        "No taste yet — save some apexes (⚑) or thumb up moments, then rebuild.";
      return;
    }
    $("#foryou-status").textContent =
      `Built from ${d.sources} loved moment${d.sources === 1 ? "" : "s"}` +
      (recentN() ? " (lately)" : "") + ` · ${d.model}`;
    currentContext = { kind: "foryou" };
    renderHits(d.items, grid);
  } catch (e) { grid.innerHTML = ""; toast(e.message, true); }
}

async function loadTasteWords() {
  const el = $("#foryou-words");
  try {
    const d = await api("/api/foryou/words?recent=" + recentN());
    if (!d.labels || !d.labels.length) { el.innerHTML = ""; return; }
    el.innerHTML = `<span class="dim">${recentN() ? "Lately you're into" : "What you're into"}:</span> ` +
      d.labels.map(([w, s]) => `<span class="fy-chip" title="${(s * 100).toFixed(0)}% match">${esc(w)}</span>`).join("");
  } catch { el.innerHTML = ""; }
}

let swipeHit = null;
async function loadNextSwipe() {
  const card = $("#swipe-card");
  card.classList.add("dim"); card.textContent = "Finding a frame to rate…";
  try {
    const d = await api("/api/foryou/next");
    swipeHit = d.item;
    if (!swipeHit) {
      card.textContent = "Embed some scenes first, then come back to train.";
      return;
    }
    card.classList.remove("dim");
    const title = swipeHit.title || `scene ${swipeHit.scene_id ?? "?"}`;
    card.innerHTML = `<img src="${swipeHit.thumb}" alt="" onerror="this.style.opacity=.15" />
      <div class="swipe-meta"><div class="title">${esc(title)}</div>
      <div class="dim">${fmt(swipeHit.time)}</div></div>`;
    card.onclick = () => openViewer(swipeHit);
  } catch (e) { card.textContent = e.message; }
}
async function swipeRate(label) {
  if (!swipeHit) return;
  try {
    const c = await api("/api/label?" + new URLSearchParams({
      key: swipeHit.key, t: (+swipeHit.time).toFixed(2), label,
      ...(swipeHit.scene_id ? { scene_id: swipeHit.scene_id } : {}),
    }), { method: "POST" });
    updateTasteUI(c);
    $("#swipe-status").textContent = `${c.positive}👍 / ${c.negative}👎`;
  } catch (e) { toast(e.message, true); }
  loadNextSwipe();
}

async function openForYou() {
  loadNextSwipe();
  await loadForYou(true);   // rebuild on open so fresh apexes/thumbs count
  loadTasteWords();
}
$("#btn-foryou-rebuild")?.addEventListener("click", () => { loadForYou(true); loadTasteWords(); });
$("#foryou-recent")?.addEventListener("change", () => { loadForYou(false); loadTasteWords(); });
$("#btn-swipe-yes")?.addEventListener("click", () => swipeRate(1));
$("#btn-swipe-no")?.addEventListener("click", () => swipeRate(0));
$("#btn-swipe-skip")?.addEventListener("click", () => loadNextSwipe());
$("#btn-swipe-train")?.addEventListener("click", async () => {
  const btn = $("#btn-swipe-train"); btn.disabled = true;
  try {
    const s = await api("/api/train", { method: "POST" });
    toast(`Trained on ${s.samples} labels (${s.positives}+)` + (s.cv_auc ? ` · AUC ${s.cv_auc}` : ""));
    loadNextSwipe();
  } catch (e) { toast(e.message, true); }
  btn.disabled = false;
});
// keyboard shortcuts while the For You tab is the active view
document.addEventListener("keydown", (e) => {
  if (!$("#viewer").hidden) return;                       // viewer owns keys when open
  if (!$("#foryou")?.classList.contains("active")) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") { e.preventDefault(); swipeRate(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); swipeRate(0); }
  else if (e.key === "ArrowDown") { e.preventDefault(); loadNextSwipe(); }
});

function setActiveView(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.view === name));
  document.querySelectorAll(".view").forEach((x) => x.classList.toggle("active", x.id === name));
}
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// hint about CLIP availability for text search + reveal the lock button
(async () => {
  try {
    const caps = await api("/api/capabilities");
    if (!caps.has_clip)
      $("#explore-hint").textContent = "text search needs a CLIP embed pass";
    if (caps.auth) $("#btn-logout").hidden = false;
  } catch {}
})();
$("#btn-logout").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch {}
  location.reload();
});

refreshDashboard();
