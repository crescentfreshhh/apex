/* Peaks control panel + explorer. Vanilla JS, no build step. */

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

// --- taste profiles ---------------------------------------------------------
// The active taste profile scopes the whole For You surface (feed, radio, labels,
// visual taste) and the tag saved moments are filed under. Default = the server's
// configured marker tag; other profiles carry their own 👍/👎, saved moments, and
// trained model. Persisted per-browser.
const PROFILE = {
  name: (() => { try { return localStorage.getItem("peaks_profile") || ""; } catch { return ""; } })(),
  default: "",   // filled in by loadProfiles()
  isDefault() { return !this.name || this.name === this.default; },
  set(name) {
    this.name = name || "";
    try { localStorage.setItem("peaks_profile", this.name); } catch { /* ignore */ }
  },
};
// query fragment for taste calls — omitted for the default so existing behaviour
// (and cache keys) are untouched.
function pparam(extra = {}) {
  return PROFILE.isDefault() ? { ...extra } : { profile: PROFILE.name, ...extra };
}
// the Stash marker tag a saved moment is filed under (undefined = server default)
function ptag() { return PROFILE.isDefault() ? undefined : PROFILE.name; }

// --- tabs -------------------------------------------------------------------
document.querySelectorAll(".tab[data-view]").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#" + b.dataset.view).classList.add("active");
    if (b.dataset.view === "dashboard") refreshDashboard();
    if (b.dataset.view === "foryou") openForYou();
    if (b.dataset.view === "galaxy" && window.openGalaxy) window.openGalaxy();
    if (b.dataset.view === "performers") openPerformers();
    if (b.dataset.view === "statistics") openStatistics();
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
  if (typeof loadSchedule === "function") loadSchedule();
  if (typeof reattachJobs === "function") reattachJobs();
}

// recurring-embed schedule + how much of the library is embedded
async function loadSchedule() {
  try {
    const d = await api("/api/schedule");
    const pend = $("#embed-pending");
    if (pend) pend.textContent = d.total == null
      ? `${(d.embedded || 0).toLocaleString()} scenes embedded (Stash unreachable for a total)`
      : `${(d.embedded || 0).toLocaleString()} / ${d.total.toLocaleString()} scenes embedded · ${(d.pending || 0).toLocaleString()} not yet embedded`;
    const on = $("#sched-on"), h = $("#sched-hours"), sy = $("#sched-sync"), pr = $("#sched-prune");
    if (on) on.checked = d.embed_hours > 0;
    if (h) h.value = d.embed_hours > 0 ? d.embed_hours : 6;
    if (sy) sy.checked = !!d.sync;
    if (pr) pr.checked = !!d.prune;
  } catch { /* dashboard offline */ }
}
$("#btn-sched-save")?.addEventListener("click", async () => {
  const on = $("#sched-on").checked;
  const hours = on ? (parseFloat($("#sched-hours").value) || 6) : 0;
  try {
    await api("/api/schedule?" + new URLSearchParams({
      embed_hours: hours, sync: $("#sched-sync").checked, prune: $("#sched-prune").checked,
    }), { method: "POST" });
    $("#sched-status").textContent = on ? `on · every ${hours}h` : "off";
    loadSchedule();
  } catch (e) { toast(e.message, true); }
});

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
      ? "<div class='dim' style='margin:8px 0 4px'>Exported videos</div>" + reels.map((r) =>
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
function renderHits(hits, container, previewMax) {
  const g = container || $("#results");
  lastHits = hits || [];   // the FULL set (save/push use this)
  const playable = lastHits.some((h) => h.scene_id && h.stream);
  $("#btn-board-search").disabled = !playable;
  $("#btn-save-collection").disabled = !playable;
  if (!lastHits.length) { g.innerHTML = '<p class="dim">No results.</p>'; return; }
  // render only a preview slice when asked (search can return thousands)
  const shown = (previewMax && lastHits.length > previewMax) ? lastHits.slice(0, previewMax) : lastHits;
  g.innerHTML = shown.map((h) => {
    const perf = (h.performers || []).slice(0, 3).join(", ");
    const sub = [h.studio, perf].filter(Boolean).join(" · ") || `scene ${h.scene_id ?? "?"}`;
    const title = h.title || `scene ${h.scene_id ?? "?"}`;
    const sid = h.scene_id ?? "";
    return `<div class="tile" data-sid="${sid}">
      <div class="thumbwrap">
        <img loading="lazy" src="${h.thumb}" alt="" onerror="this.style.opacity=.15" />
        <span class="score ${g.id === "foryou-results" ? scoreBandClass(h.score) : ""}" data-score="${h.score}">${(h.score * 100).toFixed(0)}%</span>
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
        <button class="thumb up" title="Add to my taste (trains your model)">👍</button>
        <button class="thumb down" title="Not my taste">👎</button>
        <button data-key="${h.key}" data-t="${h.time}">Find similar</button>
        <button class="apex-btn" title="Save moment — Stash marker + adds to your taste">★ Save</button>
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
    const qs = new URLSearchParams(pparam({ key, t: (+time).toFixed(2), label }));
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
    const s = await api("/api/train?" + new URLSearchParams(pparam()), { method: "POST" });
    toast(`Trained on ${s.samples} labels (${s.positives}+)` + (s.kind ? ` · ${s.kind}` : "") + (s.cv_auc ? ` · AUC ${s.cv_auc}` : ""));
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
const PREVIEW_MAX = 300; // tiles rendered; the full set still lives in lastHits
const tasteOn = () => ($("#taste-toggle").checked ? "&taste=true" : "");
let searchResults = [];  // the full ranked pool from the last search
// the Explore "search options" — per-scene cap, negation (match % is a display filter)
function searchParams() {
  const per = $("#s-per-scene") ? (parseInt($("#s-per-scene").value, 10) || 0) : 3;
  const neg = $("#s-neg") ? (parseFloat($("#s-neg").value) || 0) : 0.5;
  return { min: matchMin(), per, neg };
}
function searchQuery() {
  const { per, neg } = searchParams();
  return `&per_scene=${per}&neg_weight=${neg}&enrich=${PREVIEW_MAX}&top_k=1000` + tasteOn();
}
// the match-% slider is the SAME number printed on the tiles; it live-hides weaker ones
const matchMin = () => parseFloat($("#match-min")?.value) || 0;
function onSearchResults(items) {
  searchResults = (items || []).slice().sort((a, b) => b.score - a.score);
  const bar = $("#results-bar");
  const sl = $("#match-min");
  if (!searchResults.length) { if (bar) bar.hidden = true; renderHits([]); return; }
  const scores = searchResults.map((h) => h.score);
  const lo = Math.floor(Math.min(...scores) * 100) / 100;
  const hi = Math.ceil(Math.max(...scores) * 100) / 100;
  sl.min = lo; sl.max = hi; sl.step = 0.01; sl.value = lo;   // start showing everything
  if (bar) bar.hidden = false;
  applyMatchFilter();
}
function applyMatchFilter() {
  const v = matchMin();
  $("#match-min-val").textContent = `${Math.round(v * 100)}%`;
  const shown = searchResults.filter((h) => h.score >= v);
  renderHits(shown, null, PREVIEW_MAX);   // sets lastHits to the filtered set (save/push use it)
  const el = $("#search-count");
  if (el) el.textContent = `Showing ${Math.min(shown.length, PREVIEW_MAX).toLocaleString()} of ${shown.length.toLocaleString()}` +
    (shown.length !== searchResults.length ? ` (of ${searchResults.length.toLocaleString()} matches)` : " matches");
}
$("#match-min")?.addEventListener("input", applyMatchFilter);
async function similar(key, t) {
  setActiveView("explore");
  currentContext = { kind: "frame", key, t };
  $("#results").innerHTML = '<p class="dim">Finding similar moments…</p>';
  try {
    const d = await api(`/api/search/similar?key=${key}&t=${t}` + searchQuery());
    onSearchResults(d.items);
  } catch (e) { toast(e.message, true); }
}
async function textSearch() {
  const q = $("#q").value.trim(); if (!q) return;
  currentContext = { kind: "text", q };
  $("#results").innerHTML = '<p class="dim">Searching…</p>';
  try {
    const d = await api("/api/search/text?q=" + encodeURIComponent(q) + searchQuery());
    onSearchResults(d.items);
  } catch (e) { $("#results").innerHTML = ""; toast(e.message, true); }
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
  // volume: restore the last-used level, and reflect mute state in the icon
  const vol = $("#viewer-vol"), muteBtn = $("#viewer-mute");
  const stored = parseFloat(localStorage.getItem("peaks_vol"));
  v.volume = isNaN(stored) ? 1 : stored;
  v.muted = localStorage.getItem("peaks_muted") === "1";
  const paintVol = () => {
    vol.value = v.muted ? 0 : v.volume;
    muteBtn.textContent = v.muted || v.volume === 0 ? "🔇" : v.volume < 0.5 ? "🔉" : "🔊";
  };
  paintVol();
  vol.oninput = () => {
    v.volume = parseFloat(vol.value);
    v.muted = v.volume === 0;
    localStorage.setItem("peaks_vol", v.volume);
    localStorage.setItem("peaks_muted", v.muted ? "1" : "0");
    paintVol();
  };
  muteBtn.onclick = () => {
    v.muted = !v.muted;
    localStorage.setItem("peaks_muted", v.muted ? "1" : "0");
    paintVol();
  };
  v.onvolumechange = paintVol;
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
  try {
    const p = { t: (t || 0).toFixed(2) };
    if (ptag()) p.tag = ptag();   // file the moment under the active profile's tag
    await api(`/api/scene/${sid}/apex?` + new URLSearchParams(p), { method: "POST" });
    toast("Saved moment + added to taste @ " + fmt(t) + (PROFILE.isDefault() ? "" : ` · ${PROFILE.name}`));
  } catch (e) { toast(e.message, true); }
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
    const d = await api(`/api/search/similar?key=${currentHit.key}&t=${t.toFixed(2)}` + searchQuery());
    if (!d.items.length) return toast("no similar moments found");
    setActiveView("explore"); onSearchResults(d.items); openViewerAt(0); toast("More like this moment");
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
  if (typeof stopRadio === "function") stopRadio();
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
  else if (e.key === "m" || e.key === "M") $("#viewer-mute").click();
});
$("#btn-text").addEventListener("click", textSearch);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") textSearch(); });
wireToggle("#toggle-search-adv", "#search-adv", null);

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
function sendToMegaboard(hits, storeKey, src) {
  const apexes = hitsToApexes(hits);
  if (!apexes.length) { toast("Nothing playable to send to the megaboard yet"); return; }
  localStorage.setItem(storeKey, JSON.stringify({ tag: src, count: apexes.length, apexes }));
  window.open("/megaboard/?src=" + src, "_blank");
}
// soft net: no hard cap, but confirm before committing a very large set
const SOFT_MAX = 2000;
function bigSetOk(n, verb) {
  if (n <= SOFT_MAX) return true;
  return confirm(`This will ${verb} ${n.toLocaleString()} moments (~${Math.round(n * 0.2)} KB` +
    `${n > 8000 ? " — that's a lot" : ""}). Continue?`);
}
$("#btn-board-search").addEventListener("click", () => {
  if (!lastHits.length) return;
  if (!bigSetOk(lastHits.length, "play")) return;
  // for a text search, let the board RE-FETCH the full set (no localStorage cap)
  if (currentContext.kind === "text" && currentContext.q) {
    const { min, per, neg } = searchParams();
    const qs = new URLSearchParams({ src: "search", q: currentContext.q, per_scene: per, neg: neg });
    if (min > 0) qs.set("min_score", min);
    if ($("#taste-toggle").checked) qs.set("taste", "true");
    window.open("/megaboard/?" + qs.toString(), "_blank");
  } else {
    sendToMegaboard(lastHits, "mb_search", "search");   // frame-similar / small: snapshot
  }
});
$("#btn-save-collection").addEventListener("click", async () => {
  if (!lastHits.length) return;
  if (!bigSetOk(lastHits.length, "save")) return;
  const apexes = hitsToApexes(lastHits);
  if (!apexes.length) return;
  const name = prompt("Name this playlist:");
  if (!name) return;
  // remember how it was built (query + filters) for a future refresh
  const meta = currentContext.kind === "text"
    ? { query: currentContext.q, params: searchParams() } : {};
  try {
    const r = await api("/api/collection", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, apexes, ...meta }),
    });
    toast(`Saved "${r.name}" (${r.count} moments)`); refreshCollections();
  } catch (e) { toast(e.message, true); }
});
async function refreshCollections() {
  try {
    const { collections } = await api("/api/collections");
    const el = $("#collections");
    el.innerHTML = collections.length
      ? "<div class='dim' style='margin:8px 0 4px'>Playlists</div><div class='pl-grid'>" + collections.map((c) =>
          `<div class="coll-row pl-card" data-safe="${esc(c.safe)}" data-name="${esc(c.name)}">
            <a class="pl-cover" href="/megaboard/?collection=${encodeURIComponent(c.safe)}" target="_blank" title="Play on megaboard">
              ${c.thumb ? `<img loading="lazy" src="${c.thumb}" onerror="this.style.display='none'" />` : ""}
              <span class="pl-play">▶</span>
            </a>
            <div class="pl-meta">
              <span class="pl-name" title="${esc(c.name)}">${c.live ? '<span class="pl-live" title="Live — re-derives on open">🔴</span> ' : ""}${esc(c.name)}</span>
              <span class="dim pl-count">${c.live ? "live" : c.count + " moments"}</span>
              <span class="pl-actions">
                <button class="coll-rename" title="Rename">✏️</button>
                <button class="coll-export" title="Export to a video file">⬇</button>
                <button class="coll-del" title="Delete">🗑</button>
              </span>
            </div>
          </div>`).join("") + "</div>"
      : "";
  } catch {}
}
// --- Performers tab: leaderboard + best-of collections ----------------------
let perfLoaded = false;
async function openPerformers(refresh) {
  const grid = $("#perf-grid"); if (!grid) return;
  showPerfDetail(false);   // always land on the grid
  if (perfLoaded && !refresh) return;   // cached; use ↻ Rebuild to re-scan
  grid.innerHTML = '<p class="dim">Reading performers…</p>';
  $("#perf-status").textContent = "";
  try {
    const sort = $("#perf-sort").value;
    const d = await api(`/api/performers?sort=${sort}` + (refresh ? "&refresh=true" : ""));
    perfLoaded = true;
    renderPerformers(d.performers || []);
  } catch (e) { grid.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}
// a single static (lazy-loaded) cover thumb + hover-plays her #1 clip
function perfPhotoHTML(r) {
  const thumb = (r.top || []).map((t) => t.thumb).filter(Boolean)[0];
  const stream = (r.top && r.top[0] && r.top[0].stream) || "";
  const img = thumb
    ? `<img loading="lazy" src="${thumb}" onerror="this.src='/api/performer/${encodeURIComponent(r.id)}/image'" />`
    : `<img loading="lazy" src="/api/performer/${encodeURIComponent(r.id)}/image" onerror="this.style.display='none'" />`;
  const hover = stream ? `<video class="perf-hover" muted loop playsinline preload="none" data-stream="${stream}"></video>` : "";
  return `<div class="perf-photo">${img}${hover}</div>`;
}
function renderPerformers(rows) {
  const grid = $("#perf-grid");
  if (!rows.length) { grid.innerHTML = '<p class="dim">No performers found — embed some scenes with performers assigned in Stash.</p>'; return; }
  const maxMoments = Math.max(...rows.map((r) => r.moments)) || 1;
  grid.innerHTML = rows.map((r) => {
    const pct = Math.round((r.moments / maxMoments) * 100);
    const taste = r.affinity != null ? `<span class="perf-taste" title="mean taste affinity">★ ${Math.round(r.affinity * 100)}%</span>` : "";
    const eng = r.o_counter ? ` · ⊙ ${r.o_counter}` : "";
    return `<div class="perf-card" data-id="${esc(r.id)}" data-name="${esc(r.name)}">
      ${perfPhotoHTML(r)}
      <div class="perf-body">
        <div class="perf-name" title="${esc(r.name)}">${esc(r.name)} ${taste}</div>
        <div class="perf-bar"><span style="width:${pct}%"></span></div>
        <div class="dim perf-stats">${r.moments.toLocaleString()} moments · ${r.scenes} scenes${eng}</div>
        <div class="perf-actions">
          <button class="perf-detail-btn">Open</button>
          <button class="perf-best">⭐ Best of</button>
          <button class="perf-play ghost">▶ Board</button>
        </div>
      </div>
    </div>`;
  }).join("");
  wirePerfHover(grid);
  applyPerfFilter();   // keep the name filter applied across re-sorts/rebuilds
}
// live-filter the performer grid by the "Find a performer by name" box
function applyPerfFilter() {
  const q = ($("#perf-search")?.value || "").trim().toLowerCase();
  document.querySelectorAll("#perf-grid .perf-card").forEach((card) => {
    const name = (card.dataset.name || "").toLowerCase();
    card.hidden = !!q && !name.includes(q);
  });
}
function wirePerfHover(container) {
  container.querySelectorAll(".perf-card, .perf-hero").forEach((card) => {
    const v = card.querySelector(".perf-hover"); if (!v) return;
    card.addEventListener("mouseenter", () => { if (!v.src) v.src = v.dataset.stream; v.style.opacity = 1; v.play().catch(() => {}); });
    card.addEventListener("mouseleave", () => { v.pause(); v.style.opacity = 0; });
  });
}
async function performerBestOf(id, name) {
  const query = $("#perf-query").value.trim();
  setActiveView("explore");
  currentContext = { kind: "performer", id, name, query };
  $("#results").innerHTML = `<p class="dim">Finding ${esc(name)}'s best moments…</p>`;
  const qs = new URLSearchParams({ per_scene: 6, count: 500 });
  if (id) qs.set("id", id);
  if (name) qs.set("name", name);   // resolve by id or by name; name also labels the result
  if (query) qs.set("query", query);
  try {
    const d = await api("/api/performer/best?" + qs.toString());
    renderHits(d.items, null, PREVIEW_MAX);
    $("#search-count").textContent = `${(d.items || []).length.toLocaleString()} best moments for ${d.performer || name}` +
      (query ? ` · focus: ${query}` : "") + " — Save playlist to keep them";
    if (!d.items.length) toast(`No embedded moments for ${name}`);
  } catch (e) { toast(e.message, true); }
}
function playPerformerBoard(id, name) {
  const qs = new URLSearchParams({ src: "performer", id: id || "", name: name || "" });
  const query = $("#perf-query").value.trim(); if (query) qs.set("pq", query);
  window.open("/megaboard/?" + qs.toString(), "_blank");
}
$("#perf-grid")?.addEventListener("click", (e) => {
  const card = e.target.closest(".perf-card"); if (!card) return;
  const { id, name } = card.dataset;
  if (e.target.closest(".perf-best")) performerBestOf(id, name);
  else if (e.target.closest(".perf-play")) playPerformerBoard(id, name);
  else openPerformerDetail(id);   // "Open" button or card body → detail page
});
$("#btn-perf-search")?.addEventListener("click", async () => {
  const name = $("#perf-search").value.trim(); if (!name) return;
  await performerBestOf("", name);   // id blank → backend resolves by name
});
$("#perf-search")?.addEventListener("input", applyPerfFilter);   // type to narrow the grid
$("#perf-search")?.addEventListener("keydown", (e) => { if (e.key === "Enter") $("#btn-perf-search").click(); });
$("#perf-sort")?.addEventListener("change", () => { perfLoaded = false; openPerformers(); });
$("#btn-perf-refresh")?.addEventListener("click", () => openPerformers(true));
$("#btn-perf-roulette")?.addEventListener("click", async () => {
  try { const r = await api("/api/performer/roulette"); if (r.id) playPerformerBoard(r.id, r.name); else toast("No performers yet"); }
  catch (e) { toast(e.message, true); }
});
$("#btn-perf-hof")?.addEventListener("click", async () => {
  if (!confirm("Auto-build best-of playlists for your top 10 performers?")) return;
  $("#perf-status").textContent = "building hall of fame…";
  try { const r = await api("/api/performers/hall-of-fame", { method: "POST" }); $("#perf-status").textContent = `🏆 created ${r.created.length} playlists`; refreshCollections(); }
  catch (e) { toast(e.message, true); $("#perf-status").textContent = ""; }
});

// --- performer detail page --------------------------------------------------
function showPerfDetail(on) {
  $("#perf-grid").hidden = on; $("#perf-detail").hidden = !on;
}
async function openPerformerDetail(id) {
  const box = $("#perf-detail");
  setActiveView("performers");
  showPerfDetail(true);
  box.innerHTML = '<p class="dim">Loading…</p>';
  try {
    const d = await api("/api/performer/detail?id=" + encodeURIComponent(id));
    renderPerfDetail(d);
  } catch (e) { box.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}
function renderPerfDetail(d) {
  const box = $("#perf-detail");
  const items = d.items || [];
  const thumbs = items.slice(0, 12).map((h) => h.thumb);
  const stream = items[0] && items[0].stream;
  const s = d.stats || {};
  const stat = (lbl, v) => v == null ? "" : `<span class="pd-stat">${lbl} <b>${v}</b></span>`;
  const dist = d.distribution ? sparkHTML(d.distribution.counts) : "";
  const fp = (d.fingerprint || []).map(([w]) => `<span class="fy-chip">${esc(w)}</span>`).join("");
  const sim = (d.similar || []).map((p) =>
    `<button class="pd-sim" data-id="${esc(p.id)}"><img loading="lazy" src="${(p.top && p.top[0] && p.top[0].thumb) || `/api/performer/${encodeURIComponent(p.id)}/image`}" onerror="this.style.opacity=.15"/><span>${esc(p.name)}</span></button>`).join("");
  box.innerHTML = `
    <div class="row"><button id="pd-back" class="ghost">← Performers</button></div>
    <div class="pd-head">
      <div class="perf-hero">
        <img loading="lazy" src="${thumbs[0] || `/api/performer/${encodeURIComponent(d.id)}/image`}" />
        ${stream ? `<video class="perf-hover" muted loop playsinline preload="none" data-stream="${stream}"></video>` : ""}
      </div>
      <div class="pd-info">
        <h2>${esc(d.performer || "performer")}</h2>
        <div class="pd-stats">
          ${stat("moments", (s.moments || 0).toLocaleString())}
          ${stat("scenes", s.scenes)}
          ${stat("★ taste", s.affinity != null ? Math.round(s.affinity * 100) + "%" : null)}
          ${stat("⊙", s.o_counter)}
          ${stat("✩", s.rating)}
        </div>
        ${dist ? `<div class="dim" style="margin-top:6px">how on-taste her moments are</div>${dist}` : ""}
        ${fp ? `<div class="dim" style="margin:8px 0 4px">known for</div><div class="fy-words">${fp}</div>` : ""}
        <div class="perf-actions" style="margin-top:10px">
          <button id="pd-best" class="primary">⭐ Save best-of</button>
          <button id="pd-board" class="ghost">▶ Endless channel</button>
          <button id="pd-compare" class="ghost">⚔ Compare</button>
        </div>
      </div>
    </div>
    ${sim ? `<div class="dim" style="margin:14px 0 6px">if you like her, try…</div><div class="pd-similar">${sim}</div>` : ""}
    <div class="dim" style="margin:14px 0 6px">her best moments</div>
    <div id="pd-strip" class="grid"></div>`;
  renderHits(items, $("#pd-strip"), 60);   // sets lastHits to her best (for save/play)
  wirePerfHover(box);
  $("#pd-back").onclick = () => showPerfDetail(false);
  $("#pd-board").onclick = () => playPerformerBoard(d.id, d.performer);
  $("#pd-best").onclick = () => saveCollectionPrompt(items, `${d.performer} — best of`);
  $("#pd-compare").onclick = () => addToCompare(d.id, d.performer);
  box.querySelectorAll(".pd-sim").forEach((b) => b.onclick = () => openPerformerDetail(b.dataset.id));
}
function sparkHTML(counts) {
  const hi = Math.max(...counts) || 1;
  return `<div class="m-spark">${counts.map((c) => `<span class="mbar" style="height:${Math.round((c / hi) * 100)}%"></span>`).join("")}</div>`;
}
async function saveCollectionPrompt(items, defName) {
  const apexes = hitsToApexes(items);
  if (!apexes.length) return toast("nothing to save");
  const name = prompt("Save playlist:", defName); if (!name) return;
  try { const r = await api("/api/collection", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, apexes }) }); toast(`Saved "${r.name}" (${r.count})`); refreshCollections(); }
  catch (e) { toast(e.message, true); }
}

// --- compare two performers -------------------------------------------------
let compareList = [];
async function addToCompare(id, name) {
  if (compareList.some((c) => c.id === id)) return;
  compareList.push({ id, name });
  if (compareList.length > 2) compareList.shift();
  const tray = $("#compare-tray");
  tray.hidden = false;
  tray.innerHTML = "compare: " + compareList.map((c) => esc(c.name)).join(" vs ") +
    (compareList.length === 2 ? ` <button id="cmp-go" class="ghost">⚔ Go</button>` : " · add one more") +
    ` <button id="cmp-clear" class="ghost">clear</button>`;
  $("#cmp-clear").onclick = () => { compareList = []; tray.hidden = true; };
  if ($("#cmp-go")) $("#cmp-go").onclick = renderCompare;
}
async function renderCompare() {
  const box = $("#perf-detail"); showPerfDetail(true);
  box.innerHTML = '<p class="dim">Loading compare…</p>';
  try {
    const [a, b] = await Promise.all(compareList.map((c) => api("/api/performer/detail?id=" + encodeURIComponent(c.id))));
    const col = (d) => {
      const s = d.stats || {};
      const fp = (d.fingerprint || []).slice(0, 8).map(([w]) => `<span class="fy-chip">${esc(w)}</span>`).join("");
      const thumb = (d.items[0] && d.items[0].thumb) || `/api/performer/${encodeURIComponent(d.id)}/image`;
      return `<div class="cmp-col">
        <img src="${thumb}" onerror="this.style.opacity=.15"/>
        <h3>${esc(d.performer)}</h3>
        <div class="dim">${(s.moments || 0).toLocaleString()} moments · ${s.scenes || 0} scenes</div>
        <div class="dim">★ ${s.affinity != null ? Math.round(s.affinity * 100) + "%" : "—"} · ⊙ ${s.o_counter || 0} · ✩ ${s.rating ?? "—"}</div>
        <div class="fy-words" style="margin-top:8px">${fp}</div>
      </div>`;
    };
    box.innerHTML = `<div class="row"><button id="pd-back" class="ghost">← Performers</button></div>
      <div class="cmp-grid">${col(a)}${col(b)}</div>`;
    $("#pd-back").onclick = () => showPerfDetail(false);
  } catch (e) { box.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}

// delegated collection actions (the list is innerHTML-rendered)
$("#collections")?.addEventListener("click", async (e) => {
  const row = e.target.closest(".coll-row"); if (!row) return;
  const name = row.dataset.name;
  if (e.target.closest(".coll-rename")) {
    e.preventDefault();
    const nn = prompt("Rename collection:", name); if (!nn || nn === name) return;
    try { await api("/api/collection/rename", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, new_name: nn }) }); refreshCollections(); }
    catch (err) { toast(err.message, true); }
  } else if (e.target.closest(".coll-del")) {
    e.preventDefault();
    if (!confirm(`Delete playlist "${name}"?`)) return;
    try { await api("/api/collection?name=" + encodeURIComponent(row.dataset.safe), { method: "DELETE" }); refreshCollections(); }
    catch (err) { toast(err.message, true); }
  } else if (e.target.closest(".coll-export")) {
    e.preventDefault();
    exportCollection(name, row.dataset.safe, e.target.closest(".coll-export"));
  }
});
async function exportCollection(name, safe, btn) {
  btn.textContent = "…"; btn.disabled = true;
  try {
    await api("/api/collection/export?name=" + encodeURIComponent(name), { method: "POST" });
    toast(`Exporting "${name}" to video…`);
    const file = safe + ".mp4";
    // poll the exports list until the file appears (stream-copy is usually quick)
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      const { reels } = await api("/api/reels").catch(() => ({ reels: [] }));
      if ((reels || []).some((r) => r.name === file)) {
        btn.outerHTML = `<a class="coll-dl" href="/api/reel/download?name=${encodeURIComponent(safe)}" title="Download video">⬇ video</a>`;
        toast(`"${name}" exported`);
        return;
      }
    }
    btn.textContent = "⬇"; btn.disabled = false;
    toast("Export is taking a while — check back (exports dir).");
  } catch (err) { btn.textContent = "⬇"; btn.disabled = false; toast(err.message, true); }
}
refreshCollections();

// --- For You: taste-centroid recommender + active-learning swipe trainer ----
function recentN() { return $("#foryou-recent")?.checked ? 40 : 0; }
let foryouItems = [];  // the current feed, for the "Play on megaboard" handoff

async function loadForYou(rebuild) {
  const grid = $("#foryou-results");
  grid.innerHTML = '<p class="dim">Reading your taste…</p>';
  try {
    const qs = new URLSearchParams(pparam({ top_k: 80, recent: recentN(), rebuild: rebuild ? "true" : "false" }));
    const d = await api("/api/foryou?" + qs);
    foryouItems = d.items || [];
    if (!d.items.length) {
      grid.innerHTML = "";
      $("#foryou-status").textContent =
        "No taste yet — save some moments (★) or thumb up moments, then rebuild.";
      return;
    }
    $("#foryou-status").textContent =
      `Built from ${d.sources} loved moment${d.sources === 1 ? "" : "s"}` +
      (recentN() ? " (lately)" : "") +
      (d.reranked ? " · reranked by your taste model" : " · taste centroid (train to sharpen)") +
      (d.diversified ? " · diverse mix" : "") +
      ` · ${d.model}`;
    currentContext = { kind: "foryou" };
    renderHits(d.items, grid);
  } catch (e) { grid.innerHTML = ""; toast(e.message, true); }
}

// --- Visual taste: what the model thinks you like, shown as frames ----------
// (replaces the old CLIP "what you're into" words, which weren't accurate)
async function loadTasteVisual() {
  const el = $("#foryou-words");
  if (!el) return;
  try {
    const d = await api("/api/taste/visual?" + new URLSearchParams(pparam({ per_mode: 6 })));
    if (!d.modes || !d.modes.length) { el.innerHTML = ""; return; }
    el.innerHTML =
      `<div class="dim" style="margin-bottom:6px">What you like — your taste as frames
        (${d.sources} loved moments). Hit ✕ on any that are <em>wrong</em> to correct it.</div>` +
      d.modes.map((m) => `<div class="taste-strip">` + m.frames.map((f) =>
        `<span class="taste-frame">
           <img loading="lazy" src="${f.thumb}" title="${(f.score * 100).toFixed(0)}% match"
             onerror="this.closest('.taste-frame').style.display='none'" />
           <button class="tf-x" title="Not my taste — down-vote this frame"
             data-key="${esc(f.key)}" data-t="${f.time}" data-sid="${esc(f.scene_id || "")}">✕</button>
         </span>`).join("") + `</div>`).join("");
  } catch { el.innerHTML = ""; }
}
// cheap header counts (no frame decodes) — shown even while the gallery is collapsed
async function loadLabelCounts() {
  const el = $("#taste-labels-counts");
  if (!el) return;
  try { const d = await api("/api/labels?" + new URLSearchParams(pparam())); el.textContent = `${d.positive}👍 / ${d.negative}👎`; }
  catch { /* leave as-is */ }
}
// the editable label gallery: your explicit 👍/👎, flip or remove — paginated so
// it doesn't decode every thumbnail at once. reset=true reloads from the top.
const LABELS_PAGE = 30;
let labelsOffset = 0;
async function loadTasteLabels(reset = true) {
  const el = $("#taste-labels");
  if (!el) return;
  if (reset) { labelsOffset = 0; el.innerHTML = '<span class="dim">Loading…</span>'; }
  try {
    const d = await api("/api/taste/labels?" + new URLSearchParams(pparam({ limit: LABELS_PAGE, offset: labelsOffset })));
    const cnt = $("#taste-labels-counts");
    if (cnt) cnt.textContent = `${d.positive}👍 / ${d.negative}👎`;
    if (reset) el.innerHTML = "";
    if (!d.total) { if (reset) el.innerHTML = '<span class="dim">No labels yet — save moments (★) or thumb some up.</span>'; return; }
    el.insertAdjacentHTML("beforeend", d.labels.map((l) =>
      `<span class="taste-label ${l.label ? "pos" : "neg"}" data-key="${esc(l.key)}" data-t="${l.time}" data-sid="${esc(l.scene_id || "")}" data-label="${l.label}">
         <img loading="lazy" src="${l.thumb}" onerror="this.closest('.taste-label').style.opacity=.25" />
         <button class="tl-flip" title="Flip 👍/👎">${l.label ? "👍" : "👎"}</button>
         <button class="tl-x" title="Remove this label">✕</button>
       </span>`).join(""));
    labelsOffset += d.labels.length;
    const more = $("#btn-taste-labels-more");
    if (more) more.hidden = labelsOffset >= d.total;
  } catch (e) { if (reset) el.innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
}
// collapse handles for the two lazy taste panels (assigned at module load)
let tasteVisualCollapse = null, tasteLabelsCollapse = null;
// retrain so label edits take effect (shared by the swipe trainer + label editor)
async function trainNow(btn) {
  if (btn) btn.disabled = true;
  try {
    const s = await api("/api/train?" + new URLSearchParams(pparam()), { method: "POST" });
    toast(`Trained on ${s.samples} labels (${s.positives}+)` + (s.kind ? ` · ${s.kind}` : "") + (s.cv_auc ? ` · AUC ${s.cv_auc}` : ""));
    loadForYou(false); loadLabelCounts();
    tasteVisualCollapse?.reloadIfOpen();   // refresh only the panels you're actually viewing
  } catch (e) { toast(e.message, true); }
  if (btn) btn.disabled = false;
}
// down-vote a "what you like" frame (correct a wrong association)
$("#foryou-words")?.addEventListener("click", async (e) => {
  const x = e.target.closest(".tf-x"); if (!x) return;
  const { key, t, sid } = x.dataset;
  try {
    await api("/api/label?" + new URLSearchParams(pparam({ key, t, label: 0, ...(sid ? { scene_id: sid } : {}) })), { method: "POST" });
    x.closest(".taste-frame").style.display = "none";
    toast("👎 noted — Train now to apply to similar frames");
  } catch (err) { toast(err.message, true); }
});
// flip / remove an explicit label
$("#taste-labels")?.addEventListener("click", async (e) => {
  const cell = e.target.closest(".taste-label"); if (!cell) return;
  const { key, t, sid } = cell.dataset;
  if (e.target.closest(".tl-flip")) {
    const next = cell.dataset.label === "1" ? 0 : 1;
    try {
      await api("/api/label?" + new URLSearchParams(pparam({ key, t, label: next, ...(sid ? { scene_id: sid } : {}) })), { method: "POST" });
      cell.dataset.label = String(next);
      cell.classList.toggle("pos", !!next); cell.classList.toggle("neg", !next);
      cell.querySelector(".tl-flip").textContent = next ? "👍" : "👎";
      loadLabelCounts();   // cheap counts refresh (no re-decode of the page)
    } catch (err) { toast(err.message, true); }
  } else if (e.target.closest(".tl-x")) {
    try {
      await api("/api/label?" + new URLSearchParams(pparam({ key, t })), { method: "DELETE" });
      cell.remove();
      loadLabelCounts();
    } catch (err) { toast(err.message, true); }
  }
});
$("#btn-taste-train")?.addEventListener("click", (e) => trainNow(e.currentTarget));
$("#btn-taste-labels-refresh")?.addEventListener("click", () => loadTasteLabels(true));
$("#btn-taste-labels-more")?.addEventListener("click", () => loadTasteLabels(false));

// collapsible sections with a lazy first-open loader + remembered state
function wireCollapse(btnSel, panelSel, opts = {}) {
  const btn = $(btnSel), panel = $(panelSel);
  if (!btn || !panel) return null;
  let loaded = false;
  const set = (open) => {
    panel.hidden = !open;
    btn.textContent = (open ? "▾ " : "▸ ") + (opts.label || "");
    try { if (opts.storeKey) localStorage.setItem(opts.storeKey, open ? "1" : "0"); } catch { /* ignore */ }
    if (opts.onToggle) opts.onToggle(open);
    if (open && !loaded) { loaded = true; opts.onFirstOpen?.(); }
  };
  btn.addEventListener("click", () => set(panel.hidden));
  let remembered = false;
  try { remembered = opts.storeKey ? localStorage.getItem(opts.storeKey) === "1" : false; } catch { /* ignore */ }
  set(!!remembered);
  return { isOpen: () => !panel.hidden, reloadIfOpen: () => { if (!panel.hidden) opts.onFirstOpen?.(); } };
}
tasteVisualCollapse = wireCollapse("#toggle-taste-visual", "#foryou-words",
  { label: "What you like", storeKey: "fy_show_visual", onFirstOpen: loadTasteVisual });
tasteLabelsCollapse = wireCollapse("#toggle-taste-labels", "#taste-labels",
  { label: "your labels", storeKey: "fy_show_labels",
    onFirstOpen: () => loadTasteLabels(true),
    onToggle: (open) => { const r = $("#btn-taste-labels-refresh"); if (r) r.hidden = !open; } });

// --- Taste bands: colour For You tiles by how on-taste each moment is --------
let tasteBands = null;   // [{pct,cutoff}] high→low, for tile coloring
let metricsCdf = null;   // {thresholds,moments_ge,scenes_ge} for the Statistics coverage slider
const pctText = (v) => `${(v * 100).toFixed(0)}%`;

function scoreBandClass(score) {
  if (!tasteBands) return "";
  for (const b of tasteBands) if (score >= b.cutoff) return "band-top" + b.pct;
  return "band-low";
}
function recolorForYouTiles() {
  document.querySelectorAll("#foryou-results .score[data-score]").forEach((el) => {
    el.className = "score " + scoreBandClass(+el.dataset.score);
  });
}
// The old "Taste Metrics" panel is gone; we still fetch its bands so For You tiles
// stay colour-graded, and its useful coverage read-out now lives on Statistics.
async function loadTasteBands() {
  try {
    const m = await api("/api/taste/metrics");
    tasteBands = m.has_taste ? m.bands.map((b) => ({ pct: b.pct, cutoff: b.cutoff })) : null;
    recolorForYouTiles();
  } catch { /* leave tiles uncolored */ }
}

// The taste floor is shared with the megaboard (and its tabs) via localStorage.
const FLOOR_KEY = "peaks_taste_floor";
function readFloor() {
  try { const v = parseFloat(localStorage.getItem(FLOOR_KEY)); return isFinite(v) ? v : null; }
  catch { return null; }
}
function writeFloor(v) { try { localStorage.setItem(FLOOR_KEY, String(v)); } catch { /* ignore */ } }
window.addEventListener("storage", (e) => {   // board changed the floor → reflect it live
  if (e.key !== FLOOR_KEY || e.newValue == null) return;
  const slider = $("#stats-floor");
  if (!slider) return;
  const v = Math.min(Math.max(parseFloat(e.newValue), +slider.min), +slider.max);
  if (isFinite(v) && v !== +slider.value) { slider.value = v; updateThreshOut(); }
});
function updateThreshOut() {
  const slider = $("#stats-floor"), out = $("#stats-floor-out");
  if (!slider || !out || !metricsCdf) return;
  const v = +slider.value;
  const th = metricsCdf.thresholds;
  let i = 0; while (i < th.length - 1 && th[i + 1] <= v) i++;   // nearest step ≤ v
  const mo = metricsCdf.moments_ge[i], sc = metricsCdf.scenes_ge[i];
  const frames = metricsCdf.moments_ge[0] || 1;
  const pctile = Math.round((1 - mo / frames) * 100);
  out.innerHTML = `≥ <b>${pctText(v)}</b> → <b>${mo.toLocaleString()}</b> moments ·
    <b>${sc.toLocaleString()}</b> scenes · your <b>${pctile}th</b> percentile`;
}

// --- Statistics tab ---------------------------------------------------------
const num = (n) => (n == null ? "—" : Number(n).toLocaleString());
function agoText(unixSec) {
  if (!unixSec) return "";
  const s = Date.now() / 1000 - unixSec;
  if (s < 3600) return Math.max(1, Math.round(s / 60)) + "m ago";
  if (s < 86400) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
}
// ▶ button that opens a distinct megaboard playlist for a stat
function statBoardBtn(metric, id, label) {
  const qs = new URLSearchParams({ src: "stat", metric });
  if (id) qs.set("id", id);
  return `<a class="ghost stat-play" target="_blank" href="/megaboard/?${qs.toString()}">▶ ${esc(label || "Play")}</a>`;
}
async function openStatistics(refresh) {
  const body = $("#stats-body");
  if (!body) return;
  if (!refresh && body.dataset.loaded) return;
  $("#stats-status").textContent = "";
  body.innerHTML = '<p class="dim">Crunching your library…</p>';
  try {
    const [st, metrics] = await Promise.all([
      api("/api/statistics"),
      api("/api/taste/metrics").catch(() => null),
    ]);
    renderStatistics(st, metrics);
    body.dataset.loaded = "1";
  } catch (e) { body.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}
$("#btn-stats-refresh")?.addEventListener("click", () => openStatistics(true));

function renderStatistics(st, metrics) {
  const body = $("#stats-body");
  const b = st.build, f = st.freshness;
  const cov = (b.library_scenes && b.library_scenes > 0)
    ? Math.round((b.embedded_scenes / b.library_scenes) * 100) : null;
  const hi = Math.max(...(f.timeline_weeks || [0]), 1);
  const spark = (f.timeline_weeks || []).map((c) =>
    `<span class="mbar" style="height:${Math.round((c / hi) * 100)}%" title="${c} scene(s)"></span>`).join("");
  const last = f.last_analyzed;

  // build health
  const buildCard = `
    <div class="panel stat-card">
      <h3>Peaks build</h3>
      <div class="pd-stats">
        ${statTile("scenes analyzed", num(b.embedded_scenes) + (cov != null ? ` <span class="dim">/ ${num(b.library_scenes)} · ${cov}%</span>` : ""))}
        ${b.backlog != null && b.backlog > 0 ? statTile("awaiting analysis", num(b.backlog)) : ""}
        ${statTile("total peaks", num(b.total_peaks))}
        ${statTile("frames indexed", num(b.frames))}
        ${statTile("performers", num(b.performers))}
        ${b.failures ? statTile("failures", `<span class="warn">${num(b.failures)}</span>`) : ""}
      </div>
      <p class="dim" style="margin-top:8px">peaks scored via ${esc(st.peak_source)}.</p>
    </div>`;

  // freshness / ongoing incorporation
  const freshCard = `
    <div class="panel stat-card">
      <h3>Still ingesting your library</h3>
      <p class="dim">Scenes Peaks has analyzed into the build, by week (oldest → newest).</p>
      <div class="m-spark">${spark || '<span class="dim">no data yet</span>'}</div>
      <div class="pd-stats" style="margin-top:10px">
        ${statTile("last 24h", `${num(f.last_24h.scenes)} scenes · ${num(f.last_24h.peaks)} peaks`)}
        ${statTile("last 7 days", `${num(f.last_7d.scenes)} scenes · ${num(f.last_7d.peaks)} peaks`)}
        ${statTile("last 30 days", `${num(f.last_30d.scenes)} scenes · ${num(f.last_30d.peaks)} peaks`)}
      </div>
      ${last ? `<p class="dim" style="margin-top:8px">last analyzed:
        <b>${esc(last.title)}</b>${last.performers ? " · " + esc(last.performers) : ""}
        <span class="dim">${agoText(last.at)}</span></p>` : ""}
      <div class="perf-actions">${statBoardBtn("fresh", null, "Play fresh peaks")}</div>
    </div>`;

  // performers by peaks
  const top = st.top_actress_by_peaks;
  const rows = (st.leaderboard || []).map((r, i) =>
    `<tr><td class="dim">${i + 1}</td><td>${esc(r.name || "—")}</td>
      <td><b>${num(r.peaks)}</b> peaks</td><td class="dim">${num(r.scenes)} scenes</td>
      <td>${r.taste != null ? "★ " + pctText(r.taste) : ""}</td>
      <td>${statBoardBtn("actress", r.id, "Play")}</td></tr>`).join("");
  const perfCard = `
    <div class="panel stat-card">
      <h3>Peaks by performer</h3>
      ${top ? `<p>Most peaks: <b>${esc(top.name)}</b> — <b>${num(top.peaks)}</b> peaks across
        ${num(top.scenes)} scenes ${statBoardBtn("most_peaks_actress", null, "Play her peaks")}</p>` : '<p class="dim">No peaks yet.</p>'}
      ${st.top_actress_by_taste ? `<p class="dim">Most on-taste: <b>${esc(st.top_actress_by_taste.name)}</b>
        (★ ${pctText(st.top_actress_by_taste.taste)}) ${statBoardBtn("most_ontaste_actress", null, "Play")}</p>` : ""}
      ${rows ? `<table class="m-bands stat-lb">${rows}</table>` : ""}
    </div>`;

  // most on-taste scene
  const sc = st.most_ontaste_scene;
  const sceneCard = sc ? `
    <div class="panel stat-card">
      <h3>Most on-taste scene</h3>
      <p>Your library's single highest peak — <b>${esc(sc.title)}</b>${sc.performers ? " · " + esc(sc.performers) : ""}
        ${sc.score != null ? `<span class="dim">(peak ${pctText(sc.score)})</span>` : ""}</p>
      <div class="perf-actions">${statBoardBtn("most_ontaste_scene", sc.scene_id, "Play its best moments")}</div>
    </div>` : "";

  // taste coverage (salvaged from Taste Metrics) — the shared floor slider
  let coverageCard = "";
  if (metrics && metrics.has_taste) {
    metricsCdf = metrics.cdf;
    const d = metrics.distribution;
    const v = Math.min(Math.max(readFloor() ?? d.p90, d.min), d.max);
    coverageCard = `
      <div class="panel stat-card">
        <h3>Taste coverage</h3>
        <p class="dim">How much of your library clears a taste bar — the same floor the megaboard uses.</p>
        <div class="m-thresh"><label>Count moments at ≥
          <input type="range" id="stats-floor" min="${d.min}" max="${d.max}" step="0.005" value="${v}" /></label>
          <span id="stats-floor-out" class="dim"></span></div>
      </div>`;
  }

  body.innerHTML = buildCard + freshCard + perfCard + sceneCard + coverageCard;

  const slider = $("#stats-floor");
  if (slider) {
    slider.addEventListener("input", () => { updateThreshOut(); writeFloor(+slider.value); });
    updateThreshOut();
  }
}
function statTile(label, value) {
  return `<span class="pd-stat">${label} <b>${value}</b></span>`;
}

let swipeHit = null;
async function loadNextSwipe() {
  const card = $("#swipe-card");
  card.classList.add("dim"); card.textContent = "Finding a frame to rate…";
  try {
    const d = await api("/api/foryou/next?" + new URLSearchParams(pparam()));
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
    const c = await api("/api/label?" + new URLSearchParams(pparam({
      key: swipeHit.key, t: (+swipeHit.time).toFixed(2), label,
      ...(swipeHit.scene_id ? { scene_id: swipeHit.scene_id } : {}),
    })), { method: "POST" });
    updateTasteUI(c);
    $("#swipe-status").textContent = `${c.positive}👍 / ${c.negative}👎` + (c.autotrain ? " · training…" : "");
    if (c.autotrain) {
      // the model refits in the background (a second or two); refresh the feed +
      // taste words shortly after so the freshly-sharpened ranking shows.
      setTimeout(() => { loadForYou(false); loadTasteVisual(); }, 4000);
    }
  } catch (e) { toast(e.message, true); }
  loadNextSwipe();
}

// --- taste-profile switcher -------------------------------------------------
let profilesLoaded = false;
async function loadProfiles() {
  const sel = $("#profile-select");
  if (!sel) return;
  try {
    const d = await api("/api/profiles");
    PROFILE.default = d.default || "";
    const names = d.profiles || [PROFILE.default];
    // drop a stale stored profile that no longer exists → fall back to default
    if (PROFILE.name && !names.includes(PROFILE.name)) PROFILE.set("");
    const cur = PROFILE.isDefault() ? PROFILE.default : PROFILE.name;
    sel.innerHTML = names.map((n) =>
      `<option value="${esc(n)}"${n === cur ? " selected" : ""}>${esc(n)}${n === PROFILE.default ? " (default)" : ""}</option>`).join("");
    $("#btn-profile-del").hidden = PROFILE.isDefault();
    profilesLoaded = true;
  } catch { /* profiles are optional — leave the default in effect */ }
}
// re-scope the whole For You surface to the active profile
function reloadForProfile() {
  $("#btn-profile-del").hidden = PROFILE.isDefault();
  loadNextSwipe(); loadForYou(false); loadTasteBands(); loadLabelCounts();
  tasteVisualCollapse?.reloadIfOpen(); tasteLabelsCollapse?.reloadIfOpen();
}
$("#profile-select")?.addEventListener("change", (e) => {
  const v = e.target.value;
  PROFILE.set(v === PROFILE.default ? "" : v);
  reloadForProfile();
});
$("#btn-profile-new")?.addEventListener("click", async () => {
  const name = (prompt("Name for the new taste profile:") || "").trim();
  if (!name) return;
  try {
    await api("/api/profiles?" + new URLSearchParams({ name }), { method: "POST" });
    PROFILE.set(name);
    await loadProfiles();
    reloadForProfile();
    toast(`Switched to “${name}” — save moments or thumb some up to teach it.`);
  } catch (e) { toast(e.message, true); }
});
$("#btn-profile-del")?.addEventListener("click", async () => {
  if (PROFILE.isDefault()) return;
  const name = PROFILE.name;
  if (!confirm(`Delete taste profile “${name}”?\n\nIts 👍/👎 ratings and trained model are erased. Saved ⭐ moments stay in Stash.`)) return;
  try {
    await api("/api/profiles?" + new URLSearchParams({ name }), { method: "DELETE" });
    PROFILE.set("");
    await loadProfiles();
    reloadForProfile();
    toast(`Deleted “${name}” — back to the default profile.`);
  } catch (e) { toast(e.message, true); }
});

async function openForYou() {
  if (!profilesLoaded) await loadProfiles();
  loadNextSwipe();
  loadPicks();
  await loadForYou(false);   // cached taste = fast open; "Rebuild" forces a fresh rebuild
  loadTasteBands();          // colours the For You tiles by taste band (cheap)
  loadLabelCounts();         // header counts only; the taste panels load frames on expand
}
$("#btn-foryou-rebuild")?.addEventListener("click", () => {
  loadForYou(true); loadTasteBands(); loadLabelCounts();
  tasteVisualCollapse?.reloadIfOpen(); tasteLabelsCollapse?.reloadIfOpen();
});
$("#foryou-recent")?.addEventListener("change", () => { loadForYou(false); tasteVisualCollapse?.reloadIfOpen(); });
// the For You megaboard pulls its own big, varied pool from /api/foryou/board
// (endless, non-repeating). Carry the shared taste floor (set on the Statistics
// tab or the board itself) so it opens as selective as you set it.
$("#btn-foryou-board")?.addEventListener("click", () => {
  const floor = readFloor();
  window.open("/megaboard/?src=foryou" + (floor ? "&min_score=" + floor : "")
    + (PROFILE.isDefault() ? "" : "&profile=" + encodeURIComponent(PROFILE.name)), "_blank");
});
$("#btn-foryou-radio")?.addEventListener("click", () => startRadio());

// --- Taste Picker: a random-frame collage you tap to build your taste --------
let pickItems = [], selectedPicks = new Set();
const pickId = (it) => `${it.key}@${it.time}`;

function refreshPickAdd() {
  const btn = $("#btn-pick-add");
  if (!btn) return;
  const n = selectedPicks.size;
  btn.disabled = n === 0;
  btn.textContent = n ? `＋ Add ${n} to my taste` : "＋ Add to my taste";
}

function renderPicks() {
  const grid = $("#pick-grid");
  grid.innerHTML = "";
  pickItems.forEach((it, i) => {
    const id = pickId(it);
    const tile = document.createElement("div");
    tile.className = "pick-tile" + (selectedPicks.has(id) ? " selected" : "");
    tile.innerHTML = `<img loading="lazy" src="${it.thumb}" alt="" onerror="this.style.opacity=.15" />
      <span class="pick-t">${fmt(it.time)}</span>
      <button class="pick-play" title="Preview">▶</button>`;
    tile.addEventListener("click", (e) => {
      if (e.target.closest(".pick-play")) { openViewer(it); return; }
      if (selectedPicks.has(id)) selectedPicks.delete(id); else selectedPicks.add(id);
      tile.classList.toggle("selected");
      refreshPickAdd();
    });
    grid.appendChild(tile);
  });
}

async function loadPicks() {
  const grid = $("#pick-grid");
  if (!grid) return;
  selectedPicks.clear(); refreshPickAdd();
  grid.innerHTML = `<div class="dim" style="padding:8px">Shuffling frames…</div>`;
  try {
    const d = await api("/api/foryou/sample?count=10");
    pickItems = d.items || [];
    if (!pickItems.length) { grid.innerHTML = `<div class="dim" style="padding:8px">Embed some scenes first, then shuffle.</div>`; return; }
    renderPicks();
  } catch (e) { grid.innerHTML = `<div class="dim" style="padding:8px">${esc(e.message)}</div>`; }
}

async function addPicks() {
  const chosen = pickItems.filter((it) => selectedPicks.has(pickId(it)));
  if (!chosen.length) return;
  const btn = $("#btn-pick-add"); btn.disabled = true;
  try {
    // sequential, not Promise.all: each /api/label reloads+saves the label file,
    // so concurrent posts would clobber each other (lost updates).
    let c = null, trained = false;
    for (const it of chosen) {
      c = await api("/api/label?" + new URLSearchParams(pparam({
        key: it.key, t: (+it.time).toFixed(2), label: 1,
        ...(it.scene_id ? { scene_id: it.scene_id } : {}),
      })), { method: "POST" });
      trained = trained || !!(c && c.autotrain);
    }
    updateTasteUI(c);
    $("#pick-status").textContent = `+${chosen.length} added · ${c.positive}👍` + (trained ? " · training…" : "");
    if (trained) setTimeout(() => { loadForYou(false); loadTasteVisual(); }, 4000);
  } catch (e) { toast(e.message, true); }
  loadPicks();   // fresh collage
}
$("#btn-pick-shuffle")?.addEventListener("click", () => loadPicks());
$("#btn-pick-add")?.addEventListener("click", () => addPicks());

// --- Taste Radio: endless, auto-advancing, live-adapting personal stream -----
let radioOn = false, radioQueue = [], radioPos = 0, radioSeen = new Set();
let radioTick = null, radioClipStart = -1, radioThumbs = 0, radioCurrentHit = null;
const RADIO_CLIP_SECS = 20;

async function radioFetch() {
  const ex = encodeURIComponent([...radioSeen].join(","));
  const d = await api("/api/radio?" + new URLSearchParams(pparam({ count: 30 })) + "&exclude=" + ex);
  return (d.items || []).filter((h) => h.scene_id && h.stream);
}
async function startRadio() {
  radioSeen = new Set(); radioThumbs = 0;
  let q;
  try { q = await radioFetch(); } catch (e) { return toast(e.message, true); }
  if (!q.length) return toast("No taste yet — save (⚑) or 👍 some moments first.");
  q.forEach((h) => radioSeen.add(String(h.scene_id)));
  radioQueue = q; radioOn = true; lastHits = radioQueue;
  toast("📻 Taste Radio — lean back");
  radioShow(0);
}
function radioShow(i) {
  radioPos = i;
  openViewerAt(i);              // reuse the viewer (stream, volume, "CLIP sees")
  radioCurrentHit = currentHit;
  radioClipStart = -1;         // captured on the first playing tick
  bindRadioControls();
  const badge = $("#viewer-radio"); if (badge) badge.hidden = false;
  if (!radioTick) radioTick = setInterval(radioTickFn, 1000);
}
function radioTickFn() {
  if (!radioOn) return;
  if ($("#viewer").hidden || currentHit !== radioCurrentHit) { stopRadio(); return; }
  const v = $("#viewer-v");
  if (v.paused || !v.duration) return;   // pausing the video pauses the stream
  if (radioClipStart < 0) radioClipStart = v.currentTime;
  if (v.ended || v.currentTime - radioClipStart >= RADIO_CLIP_SECS) radioNext();
}
async function radioNext() {
  if (!radioOn) return;
  radioPos++;
  if (radioPos >= radioQueue.length - 2) {
    try {
      const more = await radioFetch();
      if (more.length) {
        more.forEach((h) => radioSeen.add(String(h.scene_id)));
        radioQueue = radioQueue.concat(more); lastHits = radioQueue;
      } else { radioSeen = new Set(); }   // seen it all → loop your taste again
    } catch {}
  }
  if (radioPos >= radioQueue.length) radioPos = 0;
  radioShow(radioPos);
}
function radioPrev() { if (radioOn) radioShow(Math.max(0, radioPos - 1)); }
function bindRadioControls() {
  const v = $("#viewer-v");
  const up = $("#viewer-up"), down = $("#viewer-down");
  const next = $("#viewer-next"), prev = $("#viewer-prev");
  if (up) up.onclick = (e) => { thumb(currentHit.key, v.currentTime, 1, currentHit.scene_id, e.currentTarget); radioThumbs++; radioMaybeTrimTail(); };
  if (down) down.onclick = (e) => { thumb(currentHit.key, v.currentTime, 0, currentHit.scene_id, e.currentTarget); radioThumbs++; radioNext(); };
  if (next) next.onclick = radioNext;
  if (prev) prev.onclick = radioPrev;
}
function radioMaybeTrimTail() {
  // every few thumbs, drop the unplayed tail so the next refill re-ranks against
  // the freshly auto-retrained model (👍/👎 already hit /api/label → autotrain)
  if (radioThumbs % 6 === 0) radioQueue = radioQueue.slice(0, radioPos + 1);
}
function stopRadio() {
  radioOn = false;
  if (radioTick) { clearInterval(radioTick); radioTick = null; }
  const badge = $("#viewer-radio"); if (badge) badge.hidden = true;
}
$("#btn-swipe-yes")?.addEventListener("click", () => swipeRate(1));
$("#btn-swipe-no")?.addEventListener("click", () => swipeRate(0));
$("#btn-swipe-skip")?.addEventListener("click", () => loadNextSwipe());
$("#btn-swipe-train")?.addEventListener("click", async () => {
  const btn = $("#btn-swipe-train"); btn.disabled = true;
  try {
    const s = await api("/api/train?" + new URLSearchParams(pparam()), { method: "POST" });
    toast(`Trained on ${s.samples} labels (${s.positives}+)` + (s.kind ? ` · ${s.kind}` : "") + (s.cv_auc ? ` · AUC ${s.cv_auc}` : ""));
    loadNextSwipe();
  } catch (e) { toast(e.message, true); }
  btn.disabled = false;
});
// delete / undo learned taste
document.querySelectorAll("#taste-manage [data-del]").forEach((b) =>
  b.addEventListener("click", async () => {
    const v = b.dataset.del, isAll = v === "all", purge = b.dataset.purge === "1";
    const msg = purge
      ? "FULL RESET — permanently delete your ENTIRE taste profile (every 👍/👎 rating and the trained model) AND every ⭐ apex marker saved in Stash?\n\nThis deletes your curated favourites from Stash itself and CANNOT be undone."
      : isAll
      ? "Permanently delete your ENTIRE taste profile — every 👍/👎 rating and the trained model?\n\nThis cannot be undone. (Your saved apex moments in Stash are kept.)"
      : `Delete all ratings from the ${b.textContent.replace("Undo last ", "last ")} and retrain on what's left?`;
    if (!confirm(msg)) return;
    b.disabled = true;
    try {
      const params = pparam(isAll ? (purge ? { purge_apexes: 1 } : {}) : { within_minutes: v });
      const r = await api("/api/taste/delete?" + new URLSearchParams(params), { method: "POST" });
      const tail = r.retrained ? " · retrained" : r.model_deleted ? " · taste model cleared" : "";
      const apexTail = r.apex_error
        ? " · ⚠ apex delete failed"
        : (typeof r.apexes_removed === "number" ? ` · ${r.apexes_removed} apex marker${r.apexes_removed === 1 ? "" : "s"} deleted` : "");
      toast(`Deleted ${r.removed} rating${r.removed === 1 ? "" : "s"}${tail}${apexTail}`, !!r.apex_error);
      $("#taste-manage-status").textContent = `${r.positive}👍 / ${r.negative}👎 left`;
      updateTasteUI({ positive: r.positive, negative: r.negative });
      loadNextSwipe(); loadForYou(true); loadTasteVisual();
    } catch (e) { toast(e.message, true); }
    b.disabled = false;
  })
);
// keyboard shortcuts while the For You tab is the active view
document.addEventListener("keydown", (e) => {
  if (!$("#viewer").hidden) return;                       // viewer owns keys when open
  if (!$("#foryou")?.classList.contains("active")) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") { e.preventDefault(); swipeRate(1); }   // → = 👍 love
  else if (e.key === "ArrowLeft") { e.preventDefault(); swipeRate(0); }  // ← = 👎 pass
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

refreshDashboard();  // conn status + job reattach (runs even though it's not the landing view)
openForYou();        // For You is the home page — populate it on load
