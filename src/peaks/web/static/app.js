/* Peaks control panel + explorer. Vanilla JS, no build step. */

// plain words for numbers (static/words.js); fig(): an exact figure, shown small or hidden
const W = window.Words, fig = W.num;
// your library's taste cutoffs (the trained model's scores) — words for tiles and floors
let SCALE = null;
W.tasteScale().then((x) => { SCALE = x; });
const $ = (s) => document.querySelector(s);
// writes that change what the library-management counts show
const LIBRARY_WRITES = /^\/api\/(catalogue\/(grade|restore|grade-bulk|restore-bulk|delete|tag-sync|train|keep)|verdicts\/|duplicates\/(resolve|ignore)|backups\/|ingest|scene\/)/;
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.reload(); throw new Error("session expired"); }
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  const method = ((opts && opts.method) || "GET").toUpperCase();
  if (method !== "GET" && LIBRARY_WRITES.test(path) && typeof refreshSidebar === "function") refreshSidebar();
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

// --- navigation: sidebar pages, remembered in the URL hash ----------------------
const VIEWS = ["foryou", "board", "explore", "performers", "catalogue", "review", "dupes",
  "statistics", "taste", "activity", "seedbox", "dashboard"];
// show a page without side effects (used when another action lands on it)
function showView(name) {
  if (!VIEWS.includes(name)) name = "foryou";
  document.querySelectorAll(".nav[data-view]").forEach((x) => x.classList.toggle("active", x.dataset.view === name));
  document.querySelectorAll(".view").forEach((x) => x.classList.toggle("active", x.id === name));
  const smart = $("#side-smart"); if (smart) smart.hidden = !["catalogue", "review"].includes(name);
  document.body.classList.remove("nav-open");
  $("#content")?.scrollTo(0, 0);
  if (location.hash !== "#/" + name) history.replaceState(null, "", "#/" + name);
}
// open a page and load what it shows
function go(name) {
  showView(name);
  if (name === "activity") { refreshDashboard(); loadRenamerMoves(); loadCleanup(); loadCopies(); loadWatch(); }
  if (name === "foryou") openForYou();
  if (name === "performers") openPerformers();
  if (name === "catalogue" && !cat.loaded) openCatalogue();
  if (name === "review") openReview();
  if (name === "dupes") openDupes();
  if (name === "statistics") openStatistics();
  if (name === "taste") openTaste();
  if (name === "dashboard") { loadTierNames(); loadTierTags(); }
  if (name === "seedbox") openSeedbox();
}
document.querySelectorAll(".nav[data-view]").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
// any [data-go] button jumps to a page (optionally a Settings section)
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-go]"); if (!b) return;
  go(b.dataset.go);
  if (b.dataset.sec) showSettingsSection(b.dataset.sec);
});
$("#nav-toggle")?.addEventListener("click", (e) => { e.stopPropagation(); document.body.classList.toggle("nav-open"); });
$(".main")?.addEventListener("click", () => document.body.classList.remove("nav-open"));
// tabs inside a page: .seg.tabs [data-tab] ↔ siblings with [data-pane]
function wireTabs(tabsSel, onShow) {
  const tabs = $(tabsSel); if (!tabs) return;
  tabs.addEventListener("click", (e) => {
    const t = e.target.closest("[data-tab]"); if (!t) return;
    tabs.querySelectorAll("[data-tab]").forEach((x) => x.classList.toggle("on", x === t));
    tabs.parentElement.querySelectorAll(":scope > [data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== t.dataset.tab; });
    if (onShow) onShow(t.dataset.tab);
  });
}
// --- Settings → Backup & restore -------------------------------------------------------
async function loadBackup() {
  let b; try { b = await api("/api/backup"); } catch (e) { $("#bk-where").textContent = e.message; return; }
  const free = b.free != null ? ` · ${fmtBytes(b.free)} free` : "";
  $("#bk-where").innerHTML = `<code>${esc(b.root)}</code>${free}<br>` + (b.writable
    ? `<span class="ok">✓ writable</span>` + (b.links_ok === true ? ` · <span class="ok">✓ hard links working</span> (unchanged files take no extra space)`
      : b.links_ok === false ? ` · <span class="warn">⚠ hard links not supported here — every snapshot is a full copy</span>` : "")
    : `<span class="warn">⚠ ${esc(b.reason || "not writable")}</span> — backups are off until this is fixed`);
  $("#bk-on").checked = !!b.backup_on; $("#bk-day").value = String(b.backup_day);
  $("#bk-time").value = `${String(b.backup_hour).padStart(2, "0")}:${String(b.backup_minute).padStart(2, "0")}`;
  $("#bk-keep").value = b.backup_keep; $("#bk-now").disabled = !b.writable;
  const s = b.snapshots || [];
  $("#bk-list").innerHTML = s.length ? s.map((x) => `<div class="bk-row" data-name="${esc(x.name)}">
      <div><b>${esc(new Date((x.created || 0) * 1000).toLocaleString())}</b>${x.label ? ` <span class="faint">· ${esc(x.label)}</span>` : ""}
        <div class="faint small">${x.files.toLocaleString()} files · ${fmtBytes(x.bytes)} in all · ${fmtBytes(x.new_bytes)} new${x.links_ok ? "" : " · full copy"}</div></div>
      <span class="grow"></span>
      <a class="btn sm ghost" href="/api/backup/${encodeURIComponent(x.name)}/download" title="Everything except the embeddings, as one small file">⬇ Small</a>
      <button class="btn sm" data-bk-restore="${esc(x.name)}">Restore…</button></div>`).join("")
    : '<span class="faint">No snapshots yet.</span>';
}
$("#bk-save")?.addEventListener("click", async () => {
  const [h, m] = ($("#bk-time").value || "04:10").split(":").map((x) => +x);
  const qs = new URLSearchParams({ backup_on: $("#bk-on").checked, backup_day: $("#bk-day").value, backup_hour: h, backup_minute: m, backup_keep: $("#bk-keep").value });
  try { await api("/api/backup?" + qs, { method: "POST" }); toast("Backup schedule saved"); loadBackup(); } catch (e) { toast(e.message, true); }
});
$("#bk-now")?.addEventListener("click", async () => {
  $("#bk-now").disabled = true;
  try {
    const job = await api("/api/backup?run=true", { method: "POST" });
    const j = await waitJob(job.id, (x) => { const p = x.progress || {}; $("#bk-status").textContent = `backing up ${p.done ?? 0}/${p.total ?? "?"}…`; });
    if (j.status === "error") throw new Error(j.error);
    const r = j.result || {};
    $("#bk-status").textContent = `✓ ${r.name}: ${fmtBytes(r.new_bytes || 0)} new, ${(r.linked || 0).toLocaleString()} files unchanged`;
  } catch (e) { toast(e.message, true); $("#bk-status").textContent = ""; }
  loadBackup();
});
$("#bk-list")?.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-bk-restore]"); if (!b) return;
  const name = b.dataset.bkRestore;
  let p; try { p = await api(`/api/backup/${encodeURIComponent(name)}/preview`); } catch (err) { return toast(err.message, true); }
  const lines = Object.entries(p.parts).map(([k, v]) => `  ${k}: ${v.snapshot.toLocaleString()} files (you have ${v.now.toLocaleString()} now)`).join("\n");
  if (!confirm(`Restore the snapshot from ${new Date(p.created * 1000).toLocaleString()}?\n\n${lines}\n\nPeaks first takes a safety snapshot of how things are now, then puts this one back (taste, models, settings, playlists, logs, embeddings) and reloads — no restart.`)) return;
  const grades = confirm("Also re-apply the grades in this snapshot to Stash?\n\nOnly needed for a new or rebuilt Stash. Cancel = leave Stash as it is.");
  try {
    const job = await api(`/api/backup/${encodeURIComponent(name)}/restore?confirm=true&grades=${grades}`, { method: "POST" });
    const j = await waitJob(job.id, () => { $("#bk-status").textContent = "restoring…"; });
    if (j.status === "error") throw new Error(j.error);
    toast(`Restored ${(j.result.restored || 0).toLocaleString()} files — safety snapshot ${j.result.safety}`);
    $("#bk-status").textContent = "";
  } catch (err) { toast(err.message, true); }
  loadBackup();
});
$("#bk-upload")?.addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const r = await api("/api/backup/upload", { method: "POST", headers: { "content-type": "application/gzip" }, body: await f.arrayBuffer() });
    toast(`Uploaded as ${r.name} — press Restore on it`); loadBackup();
  } catch (err) { toast(err.message, true); }
  e.target.value = "";
});
function showSettingsSection(sec) {
  if (sec === "backup") loadBackup();
  document.querySelectorAll("#set-nav [data-sec]").forEach((b) => b.classList.toggle("on", b.dataset.sec === sec));
  document.querySelectorAll(".sets .sec").forEach((p) => { p.hidden = p.dataset.sec !== sec; });
}
$("#set-nav")?.addEventListener("click", (e) => {
  const b = e.target.closest("[data-sec]"); if (b) showSettingsSection(b.dataset.sec);
});

// --- dashboard --------------------------------------------------------------
async function refreshDashboard() {
  try {
    const [stats, caps, mem] = await Promise.all([
      api("/api/stats"), api("/api/capabilities"), api("/api/memory").catch(() => null),
    ]);
    $("#conn").textContent = "Stash connected"; $("#conn-dot")?.classList.remove("off");
    const dino = (stats.dino_model || "").replace("dinov2_", "") || stats.model;
    const clip = `${stats.clip_model || "?"} · ${stats.clip_cached ? stats.clip_cached.toLocaleString() + " cached" : "not embedded"}`;
    $("#stat-cards").innerHTML = [
      ["Cached scenes", stats.cached_scenes],
      ["Indexed moments", caps.indexed_frames.toLocaleString()],
      ["Visual model", dino],
      ["Text-search model", clip],
      ["Device", stats.device],
      ["Failed scenes", stats.failures || 0],
      ...(mem && mem.rss_mb ? [["Peaks memory", `${(mem.rss_mb / 1024).toFixed(1)}${mem.limit_mb ? " / " + (mem.limit_mb / 1024).toFixed(1) : ""} GB`]] : []),
      ["Library", stats.library_path],
    ].map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    // surface the failures panel only when there are casualties to retry
    const nf = stats.failures || 0;
    $("#fail-panel").hidden = nf === 0;
    $("#fail-count").textContent = nf ? `· ${nf}` : "";
  } catch (e) {
    $("#conn").textContent = "disconnected"; $("#conn-dot")?.classList.add("off");
    toast("Cannot reach backend: " + e.message, true);
  }
  if (typeof refreshReels === "function") refreshReels();
  if (typeof refreshCollections === "function") refreshCollections();
  if (typeof loadSchedule === "function") loadSchedule();
  if (typeof loadHistory === "function") loadHistory();
  if (typeof loadCrashReport === "function") loadCrashReport();
  if (typeof reattachJobs === "function") reattachJobs();
}

// recurring-embed schedule + how much of the library is embedded
async function loadSchedule() {
  try {
    const d = await api("/api/schedule");
    const pend = $("#embed-pending");
    // counts are at the library's CURRENT sampling; scenes still at older
    // settings (mid re-embed) are called out instead of counted as done
    const at = d.mode ? ` at ${d.mode} · ${d.interval}s` : "";
    const stale = d.stale ? ` · ${d.stale.toLocaleString()} still at older settings` : "";
    if (pend) pend.textContent = d.total == null
      ? `${(d.embedded || 0).toLocaleString()} scenes embedded${at}${stale} (Stash unreachable for a total)`
      : `${(d.embedded || 0).toLocaleString()} / ${d.total.toLocaleString()} scenes embedded${at}${stale} · ${(d.pending || 0).toLocaleString()} to embed`;
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
  playlist: { btn: "#btn-playlist", status: "#playlist-status", log: "#playlist-log" },
  ingest: { btn: "#btn-ingest", status: "#ingest-status", log: "#ingest-log", stop: "#btn-ingest-stop" },
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
// Sampling/interval pre-fill from the LIBRARY'S saved sampling (not config), so a
// reload never silently reverts them; changing them requires a confirm.
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
    if ($("#adv-batch")) $("#adv-batch").value = d.batch_size;
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
    savedPeakPool = !!m.peak_pool;
    const pp = $("#chk-peak-pool"); if (pp) pp.checked = savedPeakPool;
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
$("#chk-peak-pool")?.addEventListener("change", async (e) => {
  try {
    await api("/api/models", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ peak_pool: e.target.checked }),
    });
    wholePeakOverride = null;   // clear any inline compare override → follow the saved default
    toast(e.target.checked ? "Matching on the whole peak" : "Matching on a single frame");
  } catch (err) { toast(err.message, true); loadModels(); }
});
loadModels();

// --- export quality (reel re-encode target) ---------------------------------
async function loadExportSettings() {
  try {
    const e = await api("/api/export-settings");
    if ($("#exp-res")) $("#exp-res").value = e.res;
    if ($("#exp-fps")) $("#exp-fps").value = e.fps;
    if ($("#exp-codec")) $("#exp-codec").value = e.codec;
  } catch {}
}
async function saveExportSettings() {
  try {
    await api("/api/export-settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        res: $("#exp-res").value, fps: $("#exp-fps").value, codec: $("#exp-codec").value,
      }),
    });
    $("#export-status").textContent = "saved";
    setTimeout(() => { if ($("#export-status")) $("#export-status").textContent = ""; }, 1500);
  } catch (err) { toast(err.message, true); }
}
$("#btn-export-save")?.addEventListener("click", saveExportSettings);
loadExportSettings();

// --- tier display names (Settings) --------------------------------------------
const TIER_NAME_KEYS = [["legendaire", "18"], ["exceptionnelle", "17"], ["merveilleuse", "16"],
  ["upscale", "0"], ["anomaly", "other O"], ["unreviewed", "unrated"], ["rejected", "1★"]];
async function loadCuration() {
  const cb = $("#cur-auto-leg"); if (!cb) return;
  try {
    const c = await api("/api/library/curation");
    cb.checked = !!c.auto_legendaire_on_save;
    if ($("#cur-follow")) $("#cur-follow").checked = c.follow_renamer !== false;
    if ($("#cur-goal") && c.today_goal) $("#cur-goal").value = c.today_goal;
  } catch {}
  loadVerdicts();
}
async function loadVerdicts() {
  const el = $("#vd-count"); if (!el) return;
  try { const v = await api("/api/verdicts"); el.textContent = `${plural(v.answered, "scene")} answered`; } catch {}
}
$("#btn-vd-clear")?.addEventListener("click", async () => {
  if (!confirm("Forget every answered suggestion? Scenes you've already decided on can be suggested again.")) return;
  try { await api("/api/verdicts/clear", { method: "POST" }); loadVerdicts(); toast("Answered suggestions cleared"); }
  catch (e) { toast(e.message, true); }
});
$("#cur-goal")?.addEventListener("change", async (e) => {
  try { const r = await api("/api/library/curation?today_goal=" + Math.max(1, +e.target.value || 20), { method: "POST" }); toast(`For You: ${r.today_goal} decisions a day (from tomorrow's set)`); }
  catch (err) { toast(err.message, true); }
});
$("#cur-follow")?.addEventListener("change", async (e) => {
  try { await api("/api/library/curation?follow_renamer=" + e.target.checked, { method: "POST" }); toast(e.target.checked ? "Following renamer moves" : "Renamer moves: Sync by hand"); }
  catch (err) { toast(err.message, true); }
});
// Activity → Maintenance: files your renamer moved after a grade, followed automatically
async function loadRenamerMoves() {
  const el = $("#moves-status"); if (!el) return;
  let m; try { m = await api("/api/library/moves"); } catch { return; }
  if (!m.on) { el.innerHTML = 'Off — turn it on in Settings → Tiers &amp; renamer, or Sync by hand.'; return; }
  const bits = [`<span class="mv-n">${m.today ? m.today.toLocaleString() : "No"}</span> move${m.today === 1 ? "" : "s"} followed today`];
  if (m.watching) bits.push(`watching ${m.watching} just-graded scene${m.watching === 1 ? "" : "s"}`);
  const last = (m.recent || [])[0];
  let html = bits.join(" · ") + (last ? `<br><span class="faint">latest: ${esc(last.title || "scene " + last.scene_id)} → ${esc(last.path)}</span>` : "");
  if ((m.unresolved || []).length)
    html += `<br><span class="warn">⚠ ${m.unresolved.length} file${m.unresolved.length === 1 ? "" : "s"} moved where Stash can't see ${m.unresolved.length === 1 ? "it" : "them"} yet</span> — scan in Stash, then <button class="btn sm" id="moves-sync">↻ Sync</button>`;
  el.innerHTML = html;
  $("#moves-sync")?.addEventListener("click", () => $("#btn-sync").click());
}
// Settings → Display: exact numbers beside the words (this browser; megaboard too)
if ($("#set-show-nums")) {
  $("#set-show-nums").checked = W.numbersShown();
  $("#set-show-nums").addEventListener("change", (e) => W.setNumbersShown(e.target.checked));
}
$("#cur-auto-leg")?.addEventListener("change", async (e) => {
  try {
    const r = await api("/api/library/curation?auto_legendaire_on_save=" + e.target.checked, { method: "POST" });
    $("#cur-status").textContent = r.auto_legendaire_on_save ? "On — saving promotes the scene" : "Off";
  } catch (err) { toast(err.message, true); }
});
async function loadTierNames() {
  loadCuration();
  const box = $("#tier-names"); if (!box) return;
  try { TIER_NAMES = { ...TIER_NAMES, ...(await api("/api/catalogue/names")) }; } catch {}
  box.innerHTML = TIER_NAME_KEYS.map(([k, hint]) =>
    `<label title="${esc(k)}">${esc(hint)} <input data-k="${k}" value="${esc(TIER_NAMES[k] || "")}" class="tier-name-in" /></label>`).join("");
}
$("#btn-tier-names-save")?.addEventListener("click", async () => {
  const names = {};
  document.querySelectorAll("#tier-names input").forEach((i) => { names[i.dataset.k] = i.value; });
  try {
    TIER_NAMES = { ...TIER_NAMES, ...(await api("/api/catalogue/names", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ names }),
    })) };
    $("#tier-names-status").textContent = "saved";
    setTimeout(() => { if ($("#tier-names-status")) $("#tier-names-status").textContent = ""; }, 1500);
    loadTierNames();
    if (cat.loaded) { renderCatChips(); renderCatList(); }
  } catch (e) { toast(e.message, true); }
});
// (loadTierNames() is called after TIER_NAMES is declared, below)

// --- tier tags for the renamer (Settings) ---------------------------------------
let TIER_TAGS = { legendaire: "legendaire", exceptionnelle: "exceptionnelle",
  merveilleuse: "merveilleuse", upscale: "personal upscale" };
const TIER_TAG_KEYS = [["legendaire", "18"], ["exceptionnelle", "17"], ["merveilleuse", "16"], ["upscale", "5★ O 0"]];
async function loadTierTags() {
  try { TIER_TAGS = { ...TIER_TAGS, ...(await api("/api/catalogue/tier-tags")) }; } catch {}
  const box = $("#tier-tags"); if (!box) return;
  box.innerHTML = TIER_TAG_KEYS.map(([k, hint]) =>
    `<label title="Stash tag for ${esc(k)}">${esc(hint)} <input data-k="${k}" value="${esc(TIER_TAGS[k] || "")}" class="tier-name-in" /></label>`).join("");
}
$("#btn-tier-tags-save")?.addEventListener("click", async () => {
  const tags = {};
  document.querySelectorAll("#tier-tags input").forEach((i) => { tags[i.dataset.k] = i.value; });
  if (!confirm("Change the tier tag names?\n\nScenes already tagged keep their old tag until you run Check & sync — and a renamed tag must match your renamer's rules.")) return;
  try {
    TIER_TAGS = { ...TIER_TAGS, ...(await api("/api/catalogue/tier-tags", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tags }),
    })) };
    $("#tier-tags-status").textContent = "saved";
    setTimeout(() => { if ($("#tier-tags-status")) $("#tier-tags-status").textContent = ""; }, 1500);
    loadTierTags();
  } catch (e) { toast(e.message, true); }
});
$("#btn-tag-sync")?.addEventListener("click", async () => {
  const box = $("#tag-sync-box");
  $("#tier-tags-status").textContent = "checking every graded scene…";
  try {
    const d = await api("/api/catalogue/tag-sync");
    $("#tier-tags-status").textContent = "";
    box.hidden = false;
    if (!d.count) {
      box.innerHTML = `<div class="backup-diff"><p>Every Légendaire / Exceptionnelle / Merveilleuse / Upscale scene already carries its one tier tag and is organized. Nothing to do.</p>
        <button class="ghost" id="btn-tag-sync-close">Close</button></div>`;
    } else {
      const rows = d.items.map((r) => `<div class="hist-row"><span class="hist-what" title="${esc(r.path)}">${esc(r.title || r.path)}</span>
        <span>${r.present.length ? esc(r.present.map((t) => d.tags[t]).join(" + ")) + " → " : ""}<b>${esc(d.tags[r.tier])}</b>${r.organized ? "" : " + organized"}</span></div>`).join("");
      box.innerHTML = `<div class="backup-diff">
        <p><b>${plural(d.count, "scene")}</b> would change · the renamer will move about <b>${plural(d.moves, "file")}</b>.
          Grades are not touched — only the tier tag (other tier tags removed) and organized.</p>
        ${rows}${d.count > d.items.length ? `<div class="dim">…and ${d.count - d.items.length} more</div>` : ""}
        <div class="row"><button id="btn-tag-sync-apply" class="primary">Tag ${plural(d.count, "scene")}</button>
          <button id="btn-tag-sync-close" class="ghost">Close</button></div></div>`;
    }
    $("#btn-tag-sync-close").onclick = () => { box.hidden = true; };
    const apply = $("#btn-tag-sync-apply");
    if (apply) apply.onclick = async () => {
      if (!confirm(`Tag ${plural(d.count, "scene")} and mark them organized?\n\nYour renamer plugin will move about ${plural(d.moves, "file")}.`)) return;
      apply.disabled = true;
      try {
        const job = await api("/api/catalogue/tag-sync", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }),
        });
        const j = await waitJob(job.id, (x) => {
          const p = x.progress || {};
          $("#tier-tags-status").textContent = `tagging ${p.done ?? 0}/${p.total ?? d.count}…`;
        });
        $("#tier-tags-status").textContent = "";
        if (j.status === "error") toast("Tag sync failed: " + j.error, true);
        else toast(`Tagged ${plural(j.result.synced, "scene")}${j.result.failed.length ? ` · ${j.result.failed.length} failed` : ""}`);
        box.hidden = true; loadHistory(); if (cat.loaded) openCatalogue({ refresh: true });
      } catch (e) { apply.disabled = false; toast(e.message, true); }
    };
  } catch (e) { $("#tier-tags-status").textContent = ""; toast(e.message, true); }
});
loadTierTags();

// --- moment length (smart clip drift threshold) -----------------------------
async function loadClipSettings() {
  try {
    const c = await api("/api/clip-settings");
    if ($("#clip-sim")) { $("#clip-sim").value = c.similarity; }
    if ($("#clip-sim-val")) $("#clip-sim-val").innerHTML = esc(W.clipLength(+c.similarity)) + fig((+c.similarity).toFixed(2));
    if (c.min != null) document.querySelectorAll("#clip-min").forEach((e) => e.textContent = c.min);
    if (c.max != null) document.querySelectorAll("#clip-max, #clip-max2").forEach((e) => e.textContent = c.max);
  } catch {}
}
async function saveClipSettings() {
  try {
    await api("/api/clip-settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ similarity: parseFloat($("#clip-sim").value) }),
    });
    $("#clip-status").textContent = "saved — reload the board to see it";
    setTimeout(() => { if ($("#clip-status")) $("#clip-status").textContent = ""; }, 2500);
  } catch (err) { toast(err.message, true); }
}
$("#clip-sim")?.addEventListener("input", (e) => {
  if ($("#clip-sim-val")) $("#clip-sim-val").innerHTML = esc(W.clipLength(+e.target.value)) + fig((+e.target.value).toFixed(2));
});
$("#btn-clip-save")?.addEventListener("click", saveClipSettings);
loadClipSettings();

// --- library safety net: action history + grade backups (Settings) -------------
// Resolves with the finished job; `onTick(job)` sees each poll while it runs.
async function waitJob(id, onTick, every = 800) {
  for (;;) {
    const j = await api("/api/jobs/" + id);
    if (onTick) onTick(j);
    if (j.status !== "running") return j;
    await new Promise((r) => setTimeout(r, every));
  }
}
const fmtBytes = (b) => {
  b = +b || 0;
  for (const [u, n] of [["TB", 1e12], ["GB", 1e9], ["MB", 1e6]]) if (b >= n) return `${(b / n).toFixed(b >= 10 * n ? 0 : 1)} ${u}`;
  return `${Math.round(b / 1e3)} KB`;
};
const plural = (n, one, many = one + "s") => `${(+n).toLocaleString()} ${n === 1 ? one : many}`;
const tierLabel = (t) => (typeof TIER_NAMES !== "undefined" && TIER_NAMES[t]) || t || "—";
const HIST_VERB = { grade: "Graded", restore: "Restored", delete: "Deleted",
  duplicate: "Duplicate resolved", "tag-sync": "Tier tag synced", ingest: "Ingest" };
function histLine(e) {
  const when = (e.ts || "").replace("T", " ").slice(0, 16);
  const what = e.title || (e.path || "").split("/").pop() || (e.scene_id ? `scene ${e.scene_id}` : "");
  let change = "";
  if (e.before && e.after) change = `${esc(tierLabel(e.before.tier))} → <b>${esc(tierLabel(e.after.tier))}</b>`;
  else if (e.after) change = `→ <b>${esc(tierLabel(e.after.tier))}</b>`;
  if (e.detail) change += ` ${esc(e.detail)}`;
  const src = e.source && e.source !== "catalogue" ? ` <span class="dim">(${esc(e.source)})</span>` : "";
  return `<div class="hist-row"><span class="dim">${esc(when)}</span>
    <span class="hist-act hist-${esc(e.action)}">${esc(HIST_VERB[e.action] || e.action)}</span>
    <span class="hist-what" title="${esc(e.path || "")}">${esc(what)}</span>
    <span>${change}${src}</span></div>`;
}
async function loadHistory() {
  const box = $("#hist-list"); if (!box) return;
  try {
    const [h, b] = await Promise.all([api("/api/history?limit=200"), api("/api/backups")]);
    box.innerHTML = h.items.length ? h.items.map(histLine).join("")
      : `<span class="dim">Nothing logged yet — grades made in Peaks will appear here.</span>`;
    const sel = $("#backup-sel");
    sel.innerHTML = b.items.length
      ? b.items.map((x) => `<option value="${esc(x.name)}">${esc(x.created.replace("T", " ").slice(0, 16))} · ${x.count.toLocaleString()} graded</option>`).join("")
      : `<option value="">no backups yet</option>`;
    $("#btn-backup-preview").disabled = !b.items.length;
  } catch (e) { box.textContent = "History unavailable: " + e.message; }
}
$("#btn-hist-refresh")?.addEventListener("click", loadHistory);
$("#btn-backup-now")?.addEventListener("click", async () => {
  try {
    const r = await api("/api/backups", { method: "POST" });
    toast(`Backed up ${r.count.toLocaleString()} graded scenes`); loadHistory();
  } catch (e) { toast(e.message, true); }
});
$("#btn-backup-preview")?.addEventListener("click", async () => {
  const name = $("#backup-sel").value, box = $("#backup-diff"); if (!name) return;
  $("#backup-status").textContent = "comparing with Stash…";
  try {
    const d = await api(`/api/backups/${encodeURIComponent(name)}/preview`);
    $("#backup-status").textContent = "";
    const rows = d.changes.slice(0, 200).map((c) => `<div class="hist-row">
      <span class="hist-what" title="${esc(c.path)}">${esc(c.title || c.path)}</span>
      <span>${esc(tierLabel(c.from.tier))} → <b>${esc(tierLabel(c.to.tier))}</b></span></div>`).join("");
    box.hidden = false;
    box.innerHTML = `<div class="backup-diff">
      <p><b>${plural(d.changes.length, "scene")}</b> ${d.changes.length === 1 ? "differs" : "differ"} from this backup ·
        ${d.same.toLocaleString()} already match · ${plural(d.missing, "file")} no longer in Stash.</p>
      ${rows}${d.changes.length > 200 ? `<div class="dim">…and ${d.changes.length - 200} more</div>` : ""}
      <div class="row">${d.changes.length ? `<button id="btn-backup-apply" class="primary">Restore ${plural(d.changes.length, "grade")}</button>` : ""}
        <button id="btn-backup-close" class="ghost">Close</button></div></div>`;
    $("#btn-backup-close").onclick = () => { box.hidden = true; };
    const apply = $("#btn-backup-apply");
    if (apply) apply.onclick = async () => {
      if (!confirm(`Re-apply ${plural(d.changes.length, "grade")} from this backup in Stash?`)) return;
      apply.disabled = true;
      try {
        const job = await api(`/api/backups/${encodeURIComponent(name)}/restore`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }),
        });
        const j = await waitJob(job.id, (x) => {
          const p = x.progress || {};
          $("#backup-status").textContent = `restoring ${p.done ?? 0}/${p.total ?? d.changes.length}…`;
        });
        $("#backup-status").textContent = "";
        if (j.status === "error") toast("Restore failed: " + j.error, true);
        else toast(`Restored ${j.result.restored} grades${j.result.failed.length ? ` · ${j.result.failed.length} failed` : ""}`);
        box.hidden = true; loadHistory(); if (cat.loaded) openCatalogue({ refresh: true });
      } catch (e) { apply.disabled = false; toast(e.message, true); }
    };
  } catch (e) { $("#backup-status").textContent = ""; toast(e.message, true); }
});

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
  for (const [k, sel] of [["interval", "#adv-interval"], ["workers", "#adv-workers"], ["timeout", "#adv-timeout"], ["batch_size", "#adv-batch"]]) {
    const v = $(sel).value; if (v !== "") qs.set(k, v);
  }
  return qs.toString();
}
// Start an embed. If the Advanced sampling differs from the library's saved
// sampling, the server refuses with needs_confirm (it would re-embed every scene
// cached at the old settings) — ask, then retry with confirm=true.
async function startEmbed() {
  const q = embedQuery();
  const url = "/api/embed" + (q ? "?" + q : "");
  const r = await fetch(url, { method: "POST" });
  if (r.status === 409) {
    let d = null;
    try { d = (await r.clone().json()).detail; } catch { /* not json */ }
    if (d && d.needs_confirm) {
      if (!confirm(d.message)) throw new Error("Cancelled — library sampling unchanged");
      const job = await api(url + (q ? "&" : "?") + "confirm=true", { method: "POST" });
      loadSchedule();   // the library's sampling just changed — re-base the counter
      return job;
    }
  }
  if (r.status === 401) { location.reload(); throw new Error("session expired"); }
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  const job = await r.json();   // the form already shows the (now saved) library sampling
  loadSchedule();               // refresh the counter — a confirmed change re-bases it
  return job;
}
wireJob($("#btn-embed"), $("#embed-status"), $("#embed-log"), startEmbed, $("#btn-embed-stop"));
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
$("#btn-fail-clean")?.addEventListener("click", async () => {
  try {
    const r = await api("/api/failures/reconcile", { method: "POST" });
    const why = Object.entries(r.reasons).filter(([, n]) => n).map(([k, n]) => `${n} ${{ deleted: "deleted from Stash",
      embedded: "embedded since", replaced: "file replaced", no_file: "no file in Stash", gone: "file gone" }[k]}`).join(", ");
    toast(r.dropped ? `Cleared ${r.dropped} ${r.dropped === 1 ? "entry" : "entries"}: ${why} · ${r.left} still failing` : `Nothing to clear · ${r.left} still failing`);
    refreshDashboard();
    if (!$("#fail-list").hidden) { $("#fail-list").hidden = true; $("#btn-fail-list").click(); }
  } catch (e) { toast(e.message, true); }
});
// Activity → Maintenance: zips and empty folders under /data, cleaned on each Sync
async function loadCleanup(action) {
  const el = $("#cleanup-status"); if (!el) return;
  let c;
  try { c = await api("/api/library/cleanup" + (action ? "?action=" + action : ""), action ? { method: "POST" } : undefined); }
  catch (e) { el.textContent = e.message; return; }
  const L = c.last;
  const what = (x) => `${x.folders.toLocaleString()} empty folder${x.folders === 1 ? "" : "s"} · ${x.zips.toLocaleString()} zip${x.zips === 1 ? "" : "s"}${x.bytes ? " · " + fmtBytes(x.bytes) : ""}`;
  let html = "";
  if (!c.on) html = `Off. <button class="btn sm" data-cl="on">Turn on</button>`;
  else if (!L) html = `Runs at the end of each Sync in <code>${esc(c.root)}</code>. <button class="btn sm" data-cl="preview">Check now</button>`;
  else if (L.skipped) html = `<span class="warn">⚠ ${esc(L.skipped)}</span> <button class="btn sm" data-cl="preview">Check again</button>`;
  else if (L.applied) html = `Last cleanup removed ${what(L)}${L.errors.length ? ` · <span class="warn">${L.errors.length} couldn't be removed</span>` : ""}. <button class="btn sm ghost" data-cl="run">Clean up now</button>`;
  else if (!L.zips && !L.folders) html = `Nothing to clean in <code>${esc(c.root)}</code>.`;
  else {
    const sample = L.sample.slice(0, 8).map((x) => `<div class="faint">${x.kind === "zip" ? "🗜" : "📁"} ${esc(x.path)}</div>`).join("");
    html = `Found ${what(L)}${c.approved ? "" : " — first time, so nothing was deleted yet"}.${sample}
      ${c.approved ? `<button class="btn sm" data-cl="run">Clean up now</button>` : `<button class="btn pri sm" data-cl="approve">Approve &amp; delete — then automatic on every Sync</button>`}`;
  }
  if (c.on) html += ` <button class="btn sm ghost" data-cl="off" title="Stop cleaning up on Sync">Turn off</button>`;
  el.innerHTML = html;
  el.querySelectorAll("[data-cl]").forEach((b) => b.onclick = () => {
    if (b.dataset.cl === "approve" && !confirm(`Delete ${what(L)} under ${c.root}?\n\nZip files and empty folders are removed for good. After this, every Sync cleans up automatically (turn it off here any time).`)) return;
    loadCleanup(b.dataset.cl);
  });
}
// Same-file copies: NOT the Duplicates tool — one scene, the same file twice.
async function loadCopies(action) {
  const el = $("#copies-status"); if (!el) return;
  let c;
  try { c = await api("/api/library/copies" + (action ? "?action=" + action : ""), action ? { method: "POST" } : undefined); }
  catch (e) { el.textContent = e.message; return; }
  const L = c.last;
  const link = (id, t) => c.stash ? `<a href="${esc(c.stash)}/scenes/${esc(id)}" target="_blank" rel="noopener">${esc(t)}</a>` : esc(t);
  const what = (n, b) => `${n.toLocaleString()} extra file${n === 1 ? "" : "s"}${b ? " · " + fmtBytes(b) : ""}`;
  const fold = (title, rows) => rows.length ? `<details class="cp-more"><summary>${title} (${rows.length.toLocaleString()})</summary>${rows.join("")}</details>` : "";
  const plans = (L?.plans || []).slice(0, 40).map((p) => `<div class="cp-plan">
      <div>${link(p.scene_id, p.title || "Scene " + p.scene_id)} <span class="faint">· ${esc(p.why)}</span></div>
      <div class="cp-keep">✓ keep ${esc(p.keep.path)}</div>
      ${p.delete.map((d) => `<div class="cp-drop">✕ ${esc(d.path)}</div>`).join("")}</div>`);
  const looks = (L?.needs_look || []).map((n) => `<div class="cp-plan"><div>${link(n.scene_id, n.title || "Scene " + n.scene_id)} <span class="faint">· ${esc(n.reason)}</span></div>${n.files.map((f) => `<div class="faint">${esc(f)}</div>`).join("")}</div>`);
  const vers = (L?.versions || []).map((v) => `<div class="cp-plan"><div>${link(v.scene_id, v.title || "Scene " + v.scene_id)} <span class="faint">· different sizes, kept both</span></div>${v.files.map((f) => `<div class="faint">${esc(f.path)} · ${fmtBytes(f.size)}</div>`).join("")}</div>`);
  const extras = fold("Same size, can't prove identical — your call", looks) + fold("Different versions — your call", vers);
  let html = "";
  if (!c.on) html = `Off. <button class="btn sm" data-cp="on">Turn on</button>`;
  else if (!L) html = `Runs at the end of each Sync and Ingest. <button class="btn sm" data-cp="preview">Check now</button>`;
  else if (L.applied) html = `Last run removed ${what(L.removed, L.freed)}${L.method === "direct" ? " (deleted on disk — Stash drops them on its next scan)" : ""}${L.errors.length ? ` · <span class="warn">${L.errors.length} scene${L.errors.length === 1 ? "" : "s"} couldn't be cleaned: ${esc(L.errors[0].error)}</span>` : ""}. <button class="btn sm ghost" data-cp="run">Check now</button>${extras}`;
  else if (!L.files) html = `No identical copies found. <button class="btn sm ghost" data-cp="preview">Check again</button>${extras}`;
  else html = `Found ${what(L.files, L.bytes)} across ${L.plans.length.toLocaleString()} scene${L.plans.length === 1 ? "" : "s"}${c.approved ? "" : " — first time, so nothing was deleted yet"}.
      ${c.approved ? `<button class="btn sm" data-cp="run">Remove now</button>` : `<button class="btn pri sm" data-cp="approve">Approve &amp; remove — then automatic on every Sync / Ingest</button>`}
      <div class="cp-list">${plans.join("")}${L.plans.length > 40 ? `<div class="faint">…and ${(L.plans.length - 40).toLocaleString()} more scenes</div>` : ""}</div>${extras}`;
  if (c.on) html += ` <button class="btn sm ghost" data-cp="off" title="Stop removing copies on Sync / Ingest">Turn off</button>`;
  el.innerHTML = html;
  el.querySelectorAll("[data-cp]").forEach((b) => b.onclick = () => {
    if (b.dataset.cp === "approve" && !confirm(`Delete ${what(L.files, L.bytes)} from disk?\n\nOnly files with the exact same size and content hash as a file the scene keeps. Scenes, grades, markers and tags stay. After this, every Sync and Ingest does it automatically (turn it off here any time).`)) return;
    b.disabled = true;
    loadCopies(b.dataset.cp);
  });
}
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
  const all = $("#reel-all")?.checked;
  if (all && !confirm("Export EVERY saved moment under this tag into one video? With a large library this can be many GB and take a while.\n\nCancel to export just the 300 most recent instead.")) {
    return Promise.reject(new Error("Export cancelled"));
  }
  const qs = new URLSearchParams();
  if (tag) qs.set("tag", tag);
  if (all) qs.set("limit", "0");   // 0 = all; omitted = server default cap (300)
  const q = qs.toString();
  return api("/api/reel" + (q ? "?" + q : ""), { method: "POST" });
}, $("#btn-reel-stop"));
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

// --- tiers: the O-count is a GRADE above 5★, not an event count --------------
// Mirrors peaks/tiers.py (tier_of) — keep the two in sync.
const TIER_ORDER = ["unreviewed", "anomaly", "upscale", "merveilleuse", "exceptionnelle", "legendaire", "rejected"];
let TIER_NAMES = {
  unreviewed: "Unreviewed", rejected: "Rejected", anomaly: "Anomaly", upscale: "Upscale",
  merveilleuse: "Merveilleuse", exceptionnelle: "Exceptionnelle", legendaire: "Légendaire",
};
const GRADES = ["reject", "upscale", "merveilleuse", "exceptionnelle", "legendaire"];   // keys 1–5, worst → best
function tierOf(rating100, o) {
  const r = +rating100 || 0;
  if (r <= 0) return "unreviewed";
  if (r <= 20) return "rejected";
  if (r < 100) return "unreviewed";
  return ({ 0: "upscale", 16: "merveilleuse", 17: "exceptionnelle", 18: "legendaire" })[+o || 0] || "anomaly";
}
function gradeName(g) { return g === "reject" ? "Reject (1★)" : TIER_NAMES[g]; }
function tierBadge(rating100, o, { showUnreviewed = false } = {}) {
  const t = tierOf(rating100, o);
  if (t === "unreviewed" && !showUnreviewed) return "";
  return `<span class="tier tier-${t}" title="${esc(TIER_NAMES[t])} · ${rating100 == null ? "unrated" : Math.round(rating100 / 20) + "★"} · O ${o ?? 0}">${esc(TIER_NAMES[t])}</span>`;
}
loadTierNames();   // fetches the saved names + fills the Settings panel

// a performer's graded keepers, best tier first, e.g. "3 Légendaire · 5 Exceptionnelle"
function tierTally(tiers, { compact = false } = {}) {
  const parts = ["legendaire", "exceptionnelle", "merveilleuse"]
    .filter((t) => (tiers || {})[t])
    .map((t) => compact
      ? `<span class="tier tier-${t}" title="${esc(TIER_NAMES[t])}">${tiers[t]}</span>`
      : `${tiers[t]} ${esc(TIER_NAMES[t])}`);
  return parts.join(compact ? " " : " · ");
}

// grade undo stack, shared by the Catalogue and the viewer's grade menu
const gradeUndo = [];
async function applyGrade(sid, grade, undoKey = "Z") {
  const r = await api("/api/catalogue/grade", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: String(sid), grade }),
  });
  gradeUndo.push({ sid: String(sid), prev: r.previous, grade });
  toast(`${gradeName(grade)} — ${undoKey} to undo`);
  return r.scene;
}
// "Keep as is": answer a suggestion at the current tier — nothing changes in Stash;
// the scene leaves the suggestion lists until there's new evidence
async function keepAsIs(sid, undoKey = "Z") {
  const r = await api(`/api/catalogue/keep?scene_id=${encodeURIComponent(sid)}`, { method: "POST" });
  gradeUndo.push({ sid: String(sid), keep: true });
  toast(`Kept as ${gradeName(r.scene.tier === "rejected" ? "reject" : r.scene.tier)} — won't be suggested again without new evidence · ${undoKey} to undo`);
  return r.scene;
}
async function undoGrade() {
  const u = gradeUndo.pop();
  if (!u) { toast("nothing to undo"); return null; }
  if (u.bulk) return undoBulkGrade(u);
  if (u.keep) {
    try {
      const r = await api(`/api/catalogue/keep?scene_id=${encodeURIComponent(u.sid)}&undo=true`, { method: "POST" });
      toast("Undone — it can be suggested again"); return r.scene;
    } catch (e) { gradeUndo.push(u); toast(e.message, true); return null; }
  }
  try {
    const r = await api("/api/catalogue/restore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_id: u.sid, ...u.prev }),
    });
    toast(`Undone (${gradeName(u.grade)})`);
    return r.scene;
  } catch (e) { gradeUndo.push(u); toast(e.message, true); return null; }
}
// a whole bulk grade is one undo step; restored scene by scene server-side
async function undoBulkGrade(u) {
  try {
    const job = await api("/api/catalogue/restore-bulk", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: u.items }),
    });
    const j = await waitJob(job.id, (x) => {
      const p = x.progress || {};
      toast(`Undoing ${p.done ?? 0}/${p.total ?? u.items.length}…`);
    });
    if (j.status === "error") throw new Error(j.error);
    toast(`Undone: ${plural(j.result.restored, "scene")} back to how they were`);
    return { bulk: true, ids: u.items.map((x) => x.scene_id) };
  } catch (e) { gradeUndo.push(u); toast(e.message, true); return null; }
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
        ${g.id === "foryou-results" ? tasteChip(h.score)
          : `<span class="score" data-score="${h.score}" title="How closely this frame matches">${(h.score * 100).toFixed(0)}%</span>`}
        <span class="t">${fmt(h.time)}</span>
      </div>
      <div class="meta">
        <div class="title" title="${esc(title)}">${esc(title)}</div>
        <div class="sub" title="${esc(sub)}">${esc(sub)}</div>
        <div class="edit">
          <span class="rating" title="rating">${stars(h.rating100)}</span>
          <span class="ospacer"></span>
          ${tierBadge(h.rating100, h.o_counter)}
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
    toast(trainSummary(s)); loadTasteQuality();
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
}
function wireTileEdits(tile) { wireSceneEdits(tile.dataset.sid, tile); }

let currentContext = {}; // what produced the current hits → drives the heatmap
const PREVIEW_MAX = 300; // tiles rendered; the full set still lives in lastHits
const tasteOn = () => ($("#taste-toggle").checked ? "&taste=true" : "");
let searchResults = [];  // the full ranked pool from the last search
// peak-level matching: savedPeakPool = the Dashboard default; wholePeakOverride =
// the inline Explore compare toggle (null → follow the saved default).
let savedPeakPool = false, wholePeakOverride = null;
const effectivePeak = () => (wholePeakOverride === null ? savedPeakPool : wholePeakOverride);
// the Explore "search options" — per-scene cap, negation (match % is a display filter)
function searchParams() {
  const per = $("#s-per-scene") ? (parseInt($("#s-per-scene").value, 10) || 0) : 3;
  const neg = $("#s-neg") ? (parseFloat($("#s-neg").value) || 0) : 0.5;
  return { min: matchMin(), per, neg };
}
function searchQuery() {
  const { per, neg } = searchParams();
  let q = `&per_scene=${per}&neg_weight=${neg}&enrich=${PREVIEW_MAX}&top_k=1000` + tasteOn();
  if (currentContext.kind === "frame") q += "&whole_peak=" + (effectivePeak() ? "true" : "false");
  return q;
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
  // the "whole peak" compare toggle only applies to similar-moment (frame) results
  const pw = $("#rb-peak-wrap"), pc = $("#rb-peak");
  if (pw) pw.hidden = currentContext.kind !== "frame";
  if (pc) pc.checked = effectivePeak();
  applyMatchFilter();
}
$("#rb-peak")?.addEventListener("change", (e) => {
  wholePeakOverride = e.target.checked;
  if (currentContext.kind === "frame" && currentContext.key != null) {
    similar(currentContext.key, currentContext.t);   // re-run the same query both ways
  }
});
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
    ${tierBadge(m.rating100, m.o_counter, { showUnreviewed: true })}
    <select class="grade-sel" title="Grade this scene (5★ + O-count) — Z undoes">
      <option value="">Grade…</option>
      ${GRADES.map((g) => `<option value="${g}">${esc(gradeName(g))}</option>`).join("")}
    </select>
    <button class="orgbtn ${m.organized ? "on" : ""}">✓ organized</button>`;
  wireSceneEdits(sid, edit);
  const gs = edit.querySelector(".grade-sel");
  if (gs) gs.addEventListener("change", async () => {
    if (!gs.value) return;
    try { await applyGrade(sid, gs.value); loadViewerMeta(sid); }
    catch (e) { toast(e.message, true); gs.value = ""; }
  });
}
async function saveMoment(sid, t) {
  if (!sid) return toast("no scene id for this result", true);
  try {
    const p = { t: (t || 0).toFixed(2) };
    if (ptag()) p.tag = ptag();   // file the moment under the active profile's tag
    const r = await api(`/api/scene/${sid}/apex?` + new URLSearchParams(p), { method: "POST" });
    toast("Saved moment + added to taste @ " + fmt(t) + (PROFILE.isDefault() ? "" : ` · ${PROFILE.name}`)
      + (r && r.promoted_from ? ` · scene promoted to ${gradeName("legendaire")}` : ""));
    if (r && r.promoted_from) refreshSidebar();
  } catch (e) { toast(e.message, true); }
}
let viewerIndex = -1;
let currentHit = null;
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
function openViewer(hit) {
  if (!hit || !hit.stream) return;
  currentHit = hit;
  const V = $("#viewer"), v = $("#viewer-v");
  V.hidden = false;
  const startAt = +hit.time || 0;
  v.src = sceneStreamUrl(hit.stream);
  v.onloadedmetadata = () => {
    try { v.currentTime = Math.min(startAt, (v.duration || startAt) - 0.1); } catch {}
    renderHeat(hit);
  };
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
  return (hits || []).filter((h) => h.scene_id && h.stream).map((h) => {
    // prefer the server's smart, content-aware clip length (clip_span);
    // fall back to the fixed BOARD_CLIP_SECONDS for older payloads.
    const dur = (Number.isFinite(+h.duration) && +h.duration > 0) ? +h.duration : BOARD_CLIP_SECONDS;
    return {
      scene_id: h.scene_id, start: +h.time, end: +h.time + dur,
      duration: dur, url: h.stream, score: h.score ?? 1, title: h.title || "",
    };
  });
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
// --- Performers: every performer in Stash, searchable, with cached pictures ----
// The directory (/api/performers/directory) is small enough to hold in the
// browser, so search, filters and sorting are instant; the grid is built 60
// cards at a time as you scroll.
let perfDir = [], perfList = [], perfShown = 0, perfDirTries = 0;
const perfFilters = new Set();
const pfold = (t) => String(t || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
async function openPerformers(refresh) {
  const grid = $("#perf-grid"); if (!grid) return;
  showPerfDetail(false);   // always land on the directory
  if (perfLoaded && !refresh) { renderPerformers(); return; }
  if (!perfDir.length) grid.innerHTML = '<p class="dim">Reading performers…</p>';
  try {
    const d = await api("/api/performers/directory" + (refresh ? "?refresh=true" : ""));
    perfDir = d.performers || [];
    perfDir.forEach((p) => { p._names = [p.name, ...(p.aliases || [])].map(fold); });
    perfAffinities = perfDir.map((p) => p.taste && p.taste.affinity).filter((a) => a != null).sort((x, y) => y - x);
    perfLoaded = true;
    renderPerformers();
    // Peaks' own figures (moments, taste) are built in the background the first time
    const again = !d.stats_ready || d.photos_missing;
    if (again && perfDirTries++ < 12) setTimeout(() => { perfLoaded = false; if ($("#performers.active")) openPerformers(); else perfLoaded = true; }, 6000);
    else perfDirTries = 0;
  } catch (e) { grid.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}
// a performer's taste in words: where she ranks among all your performers
let perfAffinities = [];     // every performer's mean taste, best first
function perfTasteHTML(aff, { cls = "perf-taste" } = {}) {
  if (aff == null) return "";
  const p = Math.round(aff * 100) + "%";
  if (!perfAffinities.length) return `<span class="${cls}" title="mean taste affinity">★ ${p}</span>`;
  const at = perfAffinities.findIndex((a) => a <= aff);
  const [k, w] = W.rank((at < 0 ? perfAffinities.length : at) / perfAffinities.length);
  return `<span class="${cls} rk-${k}" title="Mean taste ${p} — ranked ${at + 1} of ${perfAffinities.length} performers">★ ${esc(w)}${fig(p)}</span>`;
}
function perfPhotoURL(p) {
  if (!p.photo && !p.lib && !p.img) return "";
  return `/api/performer/${encodeURIComponent(p.id)}/photo?v=${p.photo || 0}`;
}
function perfInitials(name) {
  return esc(String(name || "?").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase());
}
function perfPicHTML(p, { clip = true } = {}) {
  const url = perfPhotoURL(p);
  const img = url ? `<img loading="lazy" src="${url}" alt="" onerror="this.remove()" />` : "";
  const hover = clip && p.clip ? `<video class="perf-hover" muted loop playsinline preload="none" data-stream="${esc(p.clip)}"></video>` : "";
  return `<div class="pf-pic"><span class="pf-initials">${perfInitials(p.name)}</span>${img}${hover}${p.fav ? '<span class="pf-fav" title="Favourite in Stash">♥</span>' : ""}</div>`;
}
// her record + taste in plain words (figures small beside them)
function perfWordsHTML(p) {
  const bits = [];
  if (p.record) bits.push(`<span class="wr-${p.record.tone}" title="Your grades: ${p.record.n} scenes · kept ${Math.round(p.record.keep * 100)}% · ${Math.round(p.record.top * 100)}% ${esc(className("exceptionnelle"))}+">${esc(p.record.verdict)}</span>`);
  if (p.taste) bits.push(`<span class="rk-${p.taste.key}" title="Mean taste ${Math.round(p.taste.affinity * 100)}%">★ ${esc(p.taste.word)}</span>`);
  return bits.length ? `<div class="pf-words">${bits.join(" · ")}</div>` : "";
}
function perfCardHTML(p, q = "") {
  const alias = q && !pfold(p.name).includes(q) ? (p.aliases || []).find((a) => pfold(a).includes(q)) : null;
  const counts = [p.scenes ? `${p.scenes.toLocaleString()} scene${p.scenes === 1 ? "" : "s"}` : "", p.lib ? `${p.lib.toLocaleString()} in Peaks` : ""].filter(Boolean).join(" · ");
  return `<div class="pf-card perf-card" data-id="${esc(p.id)}" data-name="${esc(p.name)}">
    ${perfPicHTML(p)}
    <div class="pf-quick"><button class="perf-best" title="Her best moments">⭐ Best</button><button class="perf-play" title="Endless megaboard channel">▶ Board</button><button class="perf-reel" title="Export a single video of her top 300 moments">⬇</button></div>
    <div class="pf-info">
      <div class="pf-name" title="${esc(p.name)}">${esc(p.name)}</div>
      ${alias ? `<div class="pf-alias">aka ${esc(alias)}</div>` : ""}
      ${perfWordsHTML(p)}
      ${counts ? `<div class="pf-counts">${counts}</div>` : ""}
    </div>
  </div>`;
}
// search: prefix of a name/alias > start of a word > anywhere; then more scenes first
function perfMatch(p, q) {
  let best = 0;
  for (const n of p._names) {
    if (n.startsWith(q)) best = Math.max(best, 3);
    else if (n.includes(" " + q) || n.includes("-" + q)) best = Math.max(best, 2);
    else if (n.includes(q)) best = Math.max(best, 1);
  }
  return best;
}
const VERDICT_RANK = { "a favourite": 6, "a good bet": 5, "mixed": 4, "early days": 3, "hit and miss": 2, "usually a miss": 1 };
function perfSorted(list) {
  const sort = $("#perf-sort").value;
  const by = {
    taste: (p) => [p.taste ? p.taste.affinity : -9, p.lib, p.scenes],
    record: (p) => [p.record ? VERDICT_RANK[p.record.verdict] || 0 : -1, p.record ? p.record.n : 0, p.lib],
    scenes: (p) => [p.scenes, p.lib],
    added: (p) => [p.created || ""],
    az: null,
  }[sort];
  if (!by) return list.slice().sort((a, b) => a.name.localeCompare(b.name));
  return list.slice().sort((a, b) => {
    const x = by(a), y = by(b);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] < y[i] ? 1 : -1;
    return a.name.localeCompare(b.name);
  });
}
function renderPerformers() {
  const grid = $("#perf-grid");
  const q = pfold(($("#perf-search")?.value || "").trim());
  const min = +($("#perf-min")?.value || 0);
  let list = perfDir.filter((p) => (!perfFilters.has("fav") || p.fav) && (!perfFilters.has("graded") || p.record)
    && (!perfFilters.has("lib") || p.lib) && p.scenes >= min);
  if (q) {
    const scored = list.map((p) => [perfMatch(p, q), p]).filter(([m]) => m > 0);
    scored.sort((a, b) => b[0] - a[0] || b[1].lib - a[1].lib || b[1].scenes - a[1].scenes);
    list = scored.map(([, p]) => p);
  } else list = perfSorted(list);
  perfList = list; perfShown = 0;
  const browsing = !q && !perfFilters.size && !min;
  $("#perf-sections").innerHTML = browsing ? perfSectionsHTML() : "";
  wirePerfHover($("#perf-sections"));
  $("#perf-gridtitle").textContent = q ? "Results" : browsing ? "Everyone" : "Matching";
  $("#perf-count").textContent = `${list.length.toLocaleString()} of ${perfDir.length.toLocaleString()}`;
  $("#perf-sub").textContent = `${perfDir.length.toLocaleString()} in your Stash · ${perfDir.filter((p) => p.lib).length.toLocaleString()} with scenes in Peaks`;
  grid.innerHTML = list.length ? "" : `<p class="dim">${q ? `No one matches “${esc(q)}”.` : "No performers match these filters."}</p>`;
  perfMore();
}
function perfMore() {
  const grid = $("#perf-grid"), q = pfold(($("#perf-search")?.value || "").trim());
  if (perfShown >= perfList.length) return;
  const next = perfList.slice(perfShown, perfShown + 60);
  perfShown += next.length;
  grid.insertAdjacentHTML("beforeend", next.map((p) => perfCardHTML(p, q)).join(""));
  wirePerfHover(grid);
}
new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting) && $("#performers.active") && !$("#perf-home").hidden) perfMore(); },
  { rootMargin: "800px" }).observe($("#perf-more"));
function perfSectionsHTML() {
  const row = (title, sub, list) => list.length ? `<div class="pf-section"><h3>${title}</h3><div class="dim">${sub}</div>
    <div class="pf-row">${list.slice(0, 16).map((p) => perfCardHTML(p)).join("")}</div></div>` : "";
  const favs = perfDir.filter((p) => (p.record && p.record.verdict === "a favourite") || p.fav)
    .sort((a, b) => (b.record ? b.record.n : 0) - (a.record ? a.record.n : 0));
  const fresh = perfDir.filter((p) => p.new_to_you).sort((a, b) => b.lib - a.lib);
  const stale = perfDir.filter((p) => p.stale).sort((a, b) => (b.last_seen_days ?? 9999) - (a.last_seen_days ?? 9999));
  return row("Your favourites", "the performers you grade highest, and your ♥ in Stash", favs)
    + row("New to you", "in scenes added lately — nothing graded yet", fresh)
    + row("Not seen in a while", "ones you grade well that the megaboard hasn't shown you in two months", stale);
}
function wirePerfHover(container) {
  container.querySelectorAll(".perf-card, .perf-hero, .pf-detail-head").forEach((card) => {
    const v = card.querySelector(".perf-hover"); if (!v || v.dataset.wired) return;
    v.dataset.wired = "1";
    card.addEventListener("mouseenter", () => { if (!v.src) v.src = v.dataset.stream; v.style.opacity = 1; v.play().catch(() => {}); });
    card.addEventListener("mouseleave", () => { v.pause(); v.style.opacity = 0; });
  });
}
async function performerBestOf(id, name, query = "") {
  query = (query || "").trim();
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
function playPerformerBoard(id, name, query = "") {
  const qs = new URLSearchParams({ src: "performer", id: id || "", name: name || "" });
  if ((query || "").trim()) qs.set("pq", query.trim());
  window.open("/megaboard/?" + qs.toString(), "_blank");
}
// one-click: stitch a performer's top-N taste-ranked moments into a single video.
// Reuses the reel builder, so it honors the Export-quality setting (mixed sources
// → re-encode). The finished file lands under Megaboard → Exported videos.
async function exportPerformerReel(id, name, count = 300) {
  if (!confirm(`Export ${name}'s top ${count} moments as one video?\n\n` +
    `This stitches clips from many scenes and re-encodes to your current ` +
    `Export-quality setting (Settings → Export quality), so it can take a while. ` +
    `The file appears under Megaboard → Exported videos when done.`)) return;
  const qs = new URLSearchParams({ count });
  if (id) qs.set("id", id);
  if (name) qs.set("name", name);
  try {
    const j = await api("/api/performer/reel?" + qs.toString(), { method: "POST" });
    toast(`Building ${name}'s top ${count}… (Megaboard → Exported videos when ready)`);
    if (j && j.id) pollPerformerReel(j.id, name);
  } catch (e) { toast(e.message, true); }
}
async function pollPerformerReel(id, name) {
  try {
    const j = await api("/api/jobs/" + id);
    if (j.status === "running") return setTimeout(() => pollPerformerReel(id, name), 3000);
    if (j.status === "error") toast(`${name}'s reel failed: ${j.error}`, true);
    else if (j.status === "cancelled") toast(`${name}'s reel stopped.`);
    else {
      const r = j.result || {};
      toast(`✅ ${name}'s reel ready: ${r.clips || "?"} clips → Megaboard → Exported videos`);
    }
  } catch { /* transient; the job still finishes server-side and shows in the reel list */ }
}
$("#perf-home")?.addEventListener("click", (e) => {
  const card = e.target.closest(".perf-card"); if (!card) return;
  const { id, name } = card.dataset;
  if (e.target.closest(".perf-best")) performerBestOf(id, name);
  else if (e.target.closest(".perf-play")) playPerformerBoard(id, name);
  else if (e.target.closest(".perf-reel")) exportPerformerReel(id, name);
  else openPerformerDetail(id);   // the card itself → her page
});
let perfSearchT = null;
$("#perf-search")?.addEventListener("input", () => {   // live: every keystroke (lightly debounced)
  clearTimeout(perfSearchT); perfSearchT = setTimeout(renderPerformers, 60);
});
$("#perf-search")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { const first = perfList[0]; if (first) openPerformerDetail(first.id); }
  if (e.key === "Escape") { e.target.value = ""; renderPerformers(); }
});
document.querySelectorAll(".pf-f").forEach((b) => b.addEventListener("click", () => {
  const f = b.dataset.f; perfFilters.has(f) ? perfFilters.delete(f) : perfFilters.add(f);
  b.classList.toggle("on", perfFilters.has(f)); renderPerformers();
}));
$("#perf-min")?.addEventListener("change", renderPerformers);
$("#perf-sort")?.addEventListener("change", renderPerformers);
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
  $("#perf-home").hidden = on; $("#perf-detail").hidden = !on;
  document.querySelector("#performers .pf-bar").hidden = on;
}
async function openPerformerDetail(id) {
  const box = $("#perf-detail");
  setActiveView("performers");
  showPerfDetail(true);
  box.innerHTML = '<p class="dim">Loading…</p>';
  // her library facts come fast; the moments (taste ranking) can take a moment longer
  const [prof, det] = await Promise.all([
    api(`/api/performer/${encodeURIComponent(id)}/profile`).catch(() => null),
    api("/api/performer/detail?id=" + encodeURIComponent(id)).catch(() => null),
  ]);
  if (!prof && !det) { box.innerHTML = '<p class="dim">Couldn\'t load this performer.</p>'; return; }
  renderPerfDetail(det || { id, performer: prof.name, items: [], stats: {} }, prof || {});
}
const PF_TIERS = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected"];   // best first
function renderPerfDetail(d, prof = {}) {
  const box = $("#perf-detail");
  const items = d.items || [];
  const s = d.stats || {};
  const me = perfDir.find((p) => p.id === String(d.id)) || { id: String(d.id), name: d.performer || prof.name, photo: prof.photo, lib: 1, fav: prof.fav };
  const name = d.performer || prof.name || me.name || "performer";
  const stream = items[0] && items[0].stream;
  const dist = d.distribution ? sparkHTML(d.distribution.counts) : "";
  const fp = (d.fingerprint || []).map(([w]) => `<span class="fy-chip">${esc(w)}</span>`).join("");
  const sim = (d.similar || []).map((p) => {
    const pp = perfDir.find((x) => x.id === String(p.id)) || { id: p.id, name: p.name, lib: 1 };
    return `<button class="pd-sim" data-id="${esc(p.id)}">${perfPicHTML(pp, { clip: false })}<span>${esc(p.name)}</span></button>`;
  }).join("");
  const r = prof.record;
  const tiers = prof.tiers || {};
  const total = Object.values(tiers).reduce((a, b) => a + b, 0) || 1;
  const bars = PF_TIERS.filter((t) => tiers[t]).map((t) =>
    `<i class="tier-bg-${t}" style="width:${(100 * tiers[t] / total).toFixed(1)}%" title="${esc(TIER_NAMES[t] || t)}: ${tiers[t]}"></i>`).join("");
  const tally = PF_TIERS.filter((t) => tiers[t]).map((t) => `${tierChip(t)} ${tiers[t]}`).join(" · ");
  const record = r ? `<div class="pf-record"><b class="wr-${r.tone}">${esc(cap(r.verdict))}</b> — of ${r.n} graded, you kept ${esc(r.keep_words)}${fig(Math.round(r.keep * 100) + "%")},
      and ${esc(r.top_words)} were ${esc(className("exceptionnelle"))} or better${fig(Math.round(r.top * 100) + "%")}.</div>`
    : `<div class="pf-record dim">No graded scenes yet — her record starts with your first grade.</div>`;
  const studios = (prof.studios || []).map((x) => `<button class="cat-chip pf-studio" data-studio="${esc(x.name)}">${esc(x.name)} <span class="n">${x.n}</span></button>`).join("");
  const scenes = (prof.scenes || []).map((x) => `<div class="pf-scene" data-sid="${esc(x.scene_id)}">${tierChip(x.tier)}<span class="t" title="${esc(x.title)}">${esc(x.title)}</span><span class="faint small">${esc((x.date || "").slice(0, 4))}</span></div>`).join("");
  box.innerHTML = `
    <div class="row"><button id="pd-back" class="ghost">← Performers</button></div>
    <div class="pf-detail-head">
      <div>
        ${perfPicHTML({ ...me, clip: stream }, { clip: !!stream })}
        <div class="row" style="margin-top:8px"><button id="pd-pick" class="btn sm">🖼 Choose picture</button></div>
      </div>
      <div class="pd-info">
        <h2>${esc(name)}${me.fav ? ' <span class="pf-favtxt" title="Favourite in Stash">♥</span>' : ""}</h2>
        ${(prof.aliases || []).length ? `<div class="dim small">also known as ${esc(prof.aliases.slice(0, 6).join(", "))}</div>` : ""}
        ${record}
        ${bars ? `<div class="pf-tierbars">${bars}</div><div class="small">${tally}</div>` : ""}
        <div class="pd-stats" style="margin-top:8px">
          ${s.moments ? `<span class="pd-stat">moments <b>${(s.moments || 0).toLocaleString()}</b></span>` : ""}
          ${s.affinity != null ? `<span class="pd-stat">taste <b>${perfTasteHTML(s.affinity, { cls: "" })}</b></span>` : ""}
        </div>
        ${dist ? `<div class="dim" style="margin-top:6px">how on-taste her moments are</div>${dist}` : ""}
        ${fp ? `<div class="dim" style="margin:8px 0 4px">known for</div><div class="fy-words">${fp}</div>` : ""}
        <div class="pf-focus"><input id="pd-focus" class="search" placeholder="Her best, focused on… (e.g. lingerie) — optional" />
          <button id="pd-bestof" class="btn pri">⭐ Best of</button></div>
        <div class="perf-actions" style="margin-top:10px">
          <button id="pd-board" class="ghost">▶ Endless channel</button>
          <button id="pd-best" class="ghost">💾 Save best-of</button>
          <button id="pd-reel" class="ghost" title="Export a single video of her top 300 taste-ranked moments">⬇ Reel</button>
          <button id="pd-compare" class="ghost">⚔ Compare</button>
          <button id="pd-catalogue" class="ghost">☰ Her scenes in Catalogue</button>
        </div>
      </div>
    </div>
    ${studios ? `<h3 style="margin:18px 0 0">Studios</h3><div class="pf-studios">${studios}</div>` : ""}
    ${sim ? `<h3 style="margin:18px 0 6px">If you like her, try…</h3><div class="pd-similar">${sim}</div>` : ""}
    ${items.length ? `<h3 style="margin:18px 0 6px">Her best moments</h3><div id="pd-strip" class="grid"></div>` : ""}
    ${scenes ? `<h3 style="margin:18px 0 0">Her scenes <span class="dim small">${(prof.scenes || []).length} · best first</span></h3><div class="pf-scenes">${scenes}</div>` : ""}`;
  if (items.length) renderHits(items, $("#pd-strip"), 60);   // sets lastHits to her best (for save/play)
  wirePerfHover(box);
  const focus = () => $("#pd-focus").value;
  $("#pd-back").onclick = () => showPerfDetail(false);
  $("#pd-bestof").onclick = () => performerBestOf(d.id, name, focus());
  $("#pd-focus").onkeydown = (e) => { if (e.key === "Enter") performerBestOf(d.id, name, focus()); };
  $("#pd-board").onclick = () => playPerformerBoard(d.id, name, focus());
  $("#pd-reel").onclick = () => exportPerformerReel(d.id, name);
  $("#pd-best").onclick = () => saveCollectionPrompt(items, `${name} — best of`);
  $("#pd-compare").onclick = () => addToCompare(d.id, name);
  $("#pd-pick").onclick = () => openPhotoPicker(d.id, name);
  $("#pd-catalogue").onclick = () => { go("catalogue"); applyCatParams({ performer: name }); };
  box.querySelectorAll(".pd-sim").forEach((b) => b.onclick = () => openPerformerDetail(b.dataset.id));
  box.querySelectorAll(".pf-studio").forEach((b) => b.onclick = () => { go("catalogue"); applyCatParams({ studio: b.dataset.studio, performer: name }); });
  box.querySelectorAll(".pf-scene").forEach((b) => b.onclick = () => { go("catalogue"); applyCatParams({ performer: name, q: b.querySelector(".t").textContent }); });
}
// 🖼 Choose picture: her Stash photo, covers of her scenes, her best frames
async function openPhotoPicker(id, name) {
  const pk = $("#perf-picker");
  pk.hidden = false;
  pk.innerHTML = `<div class="box"><div class="row between"><h3 style="margin:0">Picture for ${esc(name)}</h3><button class="ghost" id="pk-x">✕</button></div><p class="dim">Loading…</p></div>`;
  $("#pk-x").onclick = () => { pk.hidden = true; };
  let d;
  try { d = await api(`/api/performer/${encodeURIComponent(id)}/photo/options`); } catch (e) { pk.querySelector("p").textContent = e.message; return; }
  const opts = d.options || [];
  pk.querySelector(".box").innerHTML = `<div class="row between"><h3 style="margin:0">Picture for ${esc(name)}</h3>
      <span><button class="ghost" id="pk-auto" title="Let Peaks choose (Stash photo first)">Automatic</button> <button class="ghost" id="pk-x">✕</button></span></div>
    <p class="dim small">Her Stash photo, covers of her scenes (solo ones marked), and her best moments. Click one to use it everywhere.</p>
    <div class="pf-opts">${opts.map((o, i) => `<div class="pf-opt ${o.chosen ? "on" : ""} ${o.kind !== "stash" ? "wide" : ""}" data-i="${i}">
      <img loading="lazy" src="${o.thumb}" onerror="this.parentNode.remove()" /><span>${o.kind === "stash" ? "📷 Stash photo" : o.kind === "cover" ? (o.solo ? "🎬 solo · " : "🎬 ") + esc(o.label) : "✨ a moment"}</span></div>`).join("") || '<p class="dim">Nothing to choose from yet.</p>'}</div>`;
  $("#pk-x").onclick = () => { pk.hidden = true; };
  const done = async (body) => {
    try {
      const r = await api(`/api/performer/${encodeURIComponent(id)}/photo`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const p = perfDir.find((x) => x.id === String(id)); if (p) p.photo = r.photo;
      document.querySelectorAll(`img[src*="/api/performer/${encodeURIComponent(id)}/photo"]`).forEach((im) => { im.src = `/api/performer/${encodeURIComponent(id)}/photo?v=${r.photo}`; });
      pk.hidden = true; toast("Picture updated");
    } catch (e) { toast(e.message, true); }
  };
  $("#pk-auto").onclick = () => done({});
  pk.querySelectorAll(".pf-opt").forEach((el) => el.onclick = () => {
    const o = opts[+el.dataset.i]; done({ kind: o.kind, scene_id: o.scene_id, key: o.key, t: o.t });
  });
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
      const thumb = `/api/performer/${encodeURIComponent(d.id)}/photo?v=${(perfDir.find((p) => p.id === String(d.id)) || {}).photo || 0}`;
      return `<div class="cmp-col">
        <img src="${thumb}" onerror="this.style.opacity=.15"/>
        <h3>${esc(d.performer)}</h3>
        <div class="dim">${(s.moments || 0).toLocaleString()} moments · ${s.scenes || 0} scenes</div>
        <div class="dim">${s.affinity != null ? perfTasteHTML(s.affinity, { cls: "" }) : "★ —"} · ${tierTally(s.tiers, { compact: true }) || "no tiered scenes"} · ✩ ${s.rating ?? "—"}</div>
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
    const qs = new URLSearchParams(pparam({ top_k: 12, recent: recentN(), rebuild: rebuild ? "true" : "false",
      daily: rebuild ? "false" : "true" }));
    const d = await api("/api/foryou?" + qs);
    foryouItems = d.items || [];
    if (!d.items.length) {
      grid.innerHTML = "";
      renderHero(null);
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
    renderHits(d.items, grid, 12);
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
           <img loading="lazy" src="${f.thumb}" title="${(f.score * 100).toFixed(0)}% like your taste"
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
    toast(trainSummary(s)); loadTasteQuality();
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
let metricsCuts = null;  // that slider's own percentile cutoffs, for its words
const pctText = (v) => `${(v * 100).toFixed(0)}%`;

// a For You tile's score chip: the band word, the % small beside it
function tasteChip(score) {
  const b = W.tasteBand(score, SCALE && SCALE.moment);
  const p = (score * 100).toFixed(0) + "%";
  if (!b) return `<span class="score ${scoreBandClass(score)}" data-score="${score}">${p}</span>`;
  return `<span class="score ${scoreBandClass(score)}" data-score="${score}" title="Taste ${p} — ${W.TASTE_HINT[b[0]]}">${b[1]}${fig(p)}</span>`;
}
function scoreBandClass(score) {
  const b = W.tasteBand(score, SCALE && SCALE.moment);    // same scale as the words
  if (b) return { standout: "band-top99", strong: "band-top95", good: "band-top90", decent: "band-top75" }[b[0]] || "band-low";
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
  const word = W.floorWord(v, metricsCuts);
  out.innerHTML = `<b>${esc(word)}</b>${fig(`≥ ${pctText(v)} · your ${pctile}th percentile`)} → <b>${mo.toLocaleString()}</b> moments ·
    <b>${sc.toLocaleString()}</b> scenes`;
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
    const [st, metrics, storage] = await Promise.all([
      api("/api/statistics"),
      api("/api/taste/metrics").catch(() => null),
      api("/api/storage").catch(() => null),
    ]);
    renderStatistics(st, metrics, storage);
    body.dataset.loaded = "1";
  } catch (e) { body.innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
}
$("#btn-stats-refresh")?.addEventListener("click", () => openStatistics(true));

function renderStatistics(st, metrics, storage) {
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
        ${statTile("scenes analyzed", cov != null ? `${esc(cap(W.share(cov / 100)))} of your library${fig(`${num(b.embedded_scenes)} / ${num(b.library_scenes)} · ${cov}%`)}` : num(b.embedded_scenes))}
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
  // taste in words: where each performer ranks among these for your taste
  const tastes = (st.leaderboard || []).map((r) => r.taste).filter((t) => t != null).sort((a, b) => b - a);
  const tasteWord = (t) => {
    if (t == null || !tastes.length) return "";
    const [k, w] = W.rank(tastes.indexOf(t) / tastes.length);
    return `<span class="rk rk-${k}" title="Mean taste ${pctText(t)}">★ ${esc(w)}</span>${fig(pctText(t))}`;
  };
  const rows = (st.leaderboard || []).map((r, i) =>
    `<tr><td class="dim">${i + 1}</td><td>${esc(r.name || "—")}</td>
      <td><b>${num(r.peaks)}</b> peaks</td><td class="dim">${num(r.scenes)} scenes</td>
      <td>${tasteWord(r.taste)}</td>
      <td>${statBoardBtn("actress", r.id, "Play")}</td></tr>`).join("");
  const perfCard = `
    <div class="panel stat-card">
      <h3>Peaks by performer</h3>
      ${top ? `<p>Most peaks: <b>${esc(top.name)}</b> — <b>${num(top.peaks)}</b> peaks across
        ${num(top.scenes)} scenes ${statBoardBtn("most_peaks_actress", null, "Play her peaks")}</p>` : '<p class="dim">No peaks yet.</p>'}
      ${st.top_actress_by_taste ? `<p class="dim">Most on-taste: <b>${esc(st.top_actress_by_taste.name)}</b>
        ${fig("★ " + pctText(st.top_actress_by_taste.taste))} ${statBoardBtn("most_ontaste_actress", null, "Play")}</p>` : ""}
      ${rows ? `<table class="m-bands stat-lb">${rows}</table>` : ""}
    </div>`;

  // most on-taste scene
  const sc = st.most_ontaste_scene;
  const sceneCard = sc ? `
    <div class="panel stat-card">
      <h3>Most on-taste scene</h3>
      <p>Your library's single highest peak — <b>${esc(sc.title)}</b>${sc.performers ? " · " + esc(sc.performers) : ""}
        ${sc.score != null ? fig(`peak ${pctText(sc.score)}`) : ""}</p>
      <div class="perf-actions">${statBoardBtn("most_ontaste_scene", sc.scene_id, "Play its best moments")}</div>
    </div>` : "";

  // taste coverage (salvaged from Taste Metrics) — the shared floor slider
  let coverageCard = "";
  if (metrics && metrics.has_taste) {
    metricsCdf = metrics.cdf;
    const d = metrics.distribution;
    metricsCuts = d;
    const v = Math.min(Math.max(readFloor() ?? d.p90, d.min), d.max);
    coverageCard = `
      <div class="panel stat-card">
        <h3>Taste coverage</h3>
        <p class="dim">How much of your library clears a taste bar — the same floor the megaboard uses.</p>
        <div class="m-thresh"><label>Count moments that are
          <input type="range" id="stats-floor" min="${d.min}" max="${d.max}" step="0.005" value="${v}" /></label>
          <span id="stats-floor-out" class="dim"></span></div>
      </div>`;
  }

  body.innerHTML = buildCard + freshCard + perfCard + sceneCard + coverageCard + storageCard(storage);

  const slider = $("#stats-floor");
  if (slider) {
    slider.addEventListener("input", () => { updateThreshOut(); writeFloor(+slider.value); });
    updateThreshOut();
  }
}
// disk space by tier, and the biggest files in tiers you might trim
function storageCard(d) {
  if (!d) return "";
  const order = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected"];
  const tot = d.total.bytes || 1;
  const bars = order.filter((t) => d.tiers[t]).map((t) => {
    const x = d.tiers[t];
    return `<div class="stor-row"><span class="tier tier-${t}">${esc(TIER_NAMES[t])}</span>
      <span class="stor-bar"><i class="tier-bg-${t}" style="width:${Math.max(1, Math.round(100 * x.bytes / tot))}%"></i></span>
      <span class="dim">${fmtBytes(x.bytes)} · ${plural(x.count, "scene")}${t === "rejected" ? " · to delete" : ""}</span></div>`;
  }).join("");
  const big = (d.largest_low || []).map((r) => `<div class="hist-row"><span class="hist-what" title="${esc(r.path || "")}">${esc(r.title || r.path)}</span>
    <span><span class="tier tier-${r.tier}">${esc(TIER_NAMES[r.tier])}</span> <span class="dim">${esc((r.quality || {}).res || "")} ${fmtBytes(r.size)}</span></span></div>`).join("");
  return `<div class="panel stat-card">
    <h3>Storage</h3>
    <p class="dim">${fmtBytes(d.total.bytes)} across ${plural(d.total.count, "scene")}, by tier.</p>
    ${bars}
    ${big ? `<h4>Largest files outside your top tiers</h4><div class="dlg-list">${big}</div>` : ""}
  </div>`;
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
    const mini = $("#fy-teach-img");
    if (mini) { mini.innerHTML = `<img src="${swipeHit.thumb}" alt="" />`; mini.onclick = () => openViewer(swipeHit); }
  } catch (e) { card.textContent = e.message; }
}
$("#fy-teach-yes")?.addEventListener("click", () => swipeRate(1));
$("#fy-teach-no")?.addEventListener("click", () => swipeRate(0));
$("#fy-teach-skip")?.addEventListener("click", () => loadNextSwipe());
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

// the For You hero: the single best moment right now, big
function renderHero(h) {
  const el = $("#fy-hero"); if (!el) return;
  if (!h) {
    el.onclick = null;
    el.innerHTML = `<div class="hero-empty"><h2>Your feed starts with a few likes</h2>
      <p class="muted">Tap 👍 on the card to the right, ★-save moments while you watch, or pick frames in Taste → Picker. Then ↻ rebuild.</p>
      <button class="btn pri" data-go="taste">Teach your taste →</button></div>`;
    return;
  }
  const title = h.title || `scene ${h.scene_id ?? "?"}`;
  const who = (h.performers || []).slice(0, 2).join(", ");
  el.innerHTML = `<img src="${h.thumb}" alt="" onerror="this.style.opacity=.15" />
    <div class="ov"><div class="row">${tierBadge(h.rating100, h.o_counter)}<span class="muted">${W.tasteBand(h.score, SCALE && SCALE.moment) ? esc(W.tasteBand(h.score, SCALE.moment)[1]) + " match" + fig(Math.round(h.score * 100) + "%") : Math.round(h.score * 100) + "% match"} · ${fmt(h.time)}</span></div>
      <h2>${esc(who ? who + " — " + title : title)}</h2>
      <div class="row"><button class="btn pri" data-hero="play">▶ Play</button><button class="btn" data-hero="similar">⟳ More like this</button>
        <button class="btn" data-hero="save">★ Save</button></div></div>`;
  el.onclick = (e) => {
    const a = e.target.closest("[data-hero]")?.dataset.hero;
    if (a === "similar") similar(h.key, h.time);
    else if (a === "save") saveMoment(h.scene_id, h.time);
    else openViewerAt(0);
  };
}
// "your library today": counts that point at the next job to do
async function loadLibraryToday() {
  const box = $("#fy-today-body"); if (!box) return;
  try {
    const d = await api("/api/catalogue?limit=1&tier=unreviewed");
    const likely = (d.views || {}).likely || 0, rej = (d.storage || {}).rejected;
    let dupes = 0;
    try { const x = await api("/api/duplicates"); dupes = x.groups ? x.groups.length : 0; } catch {}
    const n = (v) => (+v || 0).toLocaleString();
    box.innerHTML = `<div><b>${n(d.counts.unreviewed)}</b><span>to review</span></div>
      <div><b>${n(likely)}</b><span>likely keepers</span></div>
      <div><b class="bad">${rej ? fmtBytes(rej.bytes) : "0"}</b><span>rejects to delete</span></div>
      <div><b class="gold">${n(dupes)}</b><span>duplicate groups</span></div>`;
    setNavCounts(d);
  } catch { box.innerHTML = '<span class="faint">Stash unreachable</span>'; }
}
function openTaste() { loadNextSwipe(); loadLabelCounts(); loadTasteQuality(); }

// --- taste quality: the held-out benchmark, in plain words -------------------
function trainSummary(s) {
  const h = s.holdout || {};
  if (s.mode === "quick")
    return `Retrained on ${s.samples.toLocaleString()} moments · ${s.kind === "mlp" ? "non-linear" : "linear"} model` +
      (s.context ? `, ±${s.context} frames` : "") + " (from the last measure)";
  const bits = [`Trained on ${s.samples.toLocaleString()} moments`];
  if (h.peak_hit != null) bits.push(`jump-to-peak lands on your moment ${W.howOften(h.peak_hit)} (${Math.round(h.peak_hit * 100)}%)`);
  if (h.auc != null) bits.push(`tells love from pass: ${W.auc(h.auc).toLowerCase()} (AUC ${h.auc}` + (s.auc_delta ? `, ${s.auc_delta > 0 ? "+" : ""}${s.auc_delta}` : "") + ")");
  return bits.join(" · ");
}
const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
function tqSpark(hist, key) {
  const pts = hist.map((h) => h[key]).filter((x) => x != null);
  if (pts.length < 2) return "";
  const lo = Math.min(...pts) - 0.03, hi = Math.max(...pts) + 0.03, W = 120, H = 28;
  const xy = pts.map((p, i) => `${(i / (pts.length - 1) * W).toFixed(1)},${(H - 2 - (p - lo) / (hi - lo || 1) * (H - 4)).toFixed(1)}`);
  return `<svg class="tq-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><polyline points="${xy.join(" ")}" /></svg>`;
}
const TQ_SRC = [["explicit", "your ratings"], ["marker", "saved moments with no 👍 yet"], ["tier", "graded-scene moments"],
  ["reject", "reject moments"], ["engage", "from watching"], ["background", "library background"]];
// the full measure runs as a background job; the card follows it
const tqJob = { id: null, timer: null };
async function startMeasure() {
  try {
    const r = await api("/api/taste/measure?" + new URLSearchParams(pparam()), { method: "POST" });
    tqJob.id = r.job;
    toast("Measuring in the background — follow it in the sidebar tray");
  } catch (e) { toast(e.message, true); }
  followMeasure();
}
async function followMeasure() {
  clearTimeout(tqJob.timer);
  let job = null;
  try {
    const jobs = await api("/api/jobs");
    job = tqJob.id ? jobs.find((j) => j.id === tqJob.id) : jobs.find((j) => j.kind === "taste-measure" && j.status === "running");
  } catch { return; }
  if (!job) { renderMeasureRun(null); return; }
  tqJob.id = job.id;
  if (job.status === "running") {
    renderMeasureRun(job);
    tqJob.timer = setTimeout(followMeasure, 3000);
    return;
  }
  tqJob.id = null;
  renderMeasureRun(null);
  if (job.status === "done" && job.result) { toast(trainSummary(job.result)); loadTasteQuality(); loadForYou(false); }
  else if (job.status === "error") toast("Measure failed: " + (job.error || "unknown error"), true);
}
function renderMeasureRun(job) {
  const run = $("#tq-run");
  const p = (job && job.progress) || {}, pct = Math.round(100 * (p.pct || 0));
  document.querySelectorAll(".tq-train").forEach((b) => {
    b.disabled = !!job;
    b.classList.toggle("running", !!job);
    b.style.setProperty("--fill", job ? pct + "%" : "0%");
    b.textContent = job ? `Measuring… ${pct}%` : "⟳ Train & measure";
  });
  if (job) { const cap = $("#tq-train-cap"); if (cap) cap.textContent = p.stage || "Starting…"; }
  else tqCaption();
  if (!run) return;
  run.hidden = !job;
  if (!job) return;
  run.innerHTML = `<div class="row between small"><span>${esc(p.stage || "Starting…")}</span>
    <span class="muted">${pct}% · ${Math.round(job.elapsed / 60)} min
      <button class="btn ghost sm" data-cancel-job="${esc(job.id)}">Cancel</button></span></div>
    <div class="bar"><i style="width:${pct}%"></i></div>`;
}
// the header button's caption: when the last full measure ran
let tqLast = null;
function ago(ts) {
  const m = Math.max(0, (Date.now() / 1000 - ts) / 60);
  if (m < 1) return "just now";
  if (m < 60) return `${Math.round(m)} min ago`;
  if (m < 48 * 60) return `${Math.round(m / 60)} h ago`;
  return `${Math.round(m / 1440)} days ago`;
}
function tqCaption() {
  const cap = $("#tq-train-cap"); if (!cap) return;
  cap.textContent = tqLast ? `Last measured ${ago(tqLast)}` : "Never measured";
}
function tqScheduleLine(sch) {
  if (!sch) return "";
  const quick = "Quick retrain after every 25 ratings.";
  if (sch.hour < 0) return `${quick} Full measure: only when you click it.`;
  const at = new Date(); at.setHours(sch.hour, 0, 0, 0);
  const when = at.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const left = Math.max(0, sch.needed - sch.signals);
  return `${quick} Next full measure: overnight at ${when}` +
    (left ? ` once ${left.toLocaleString()} more ratings or grades come in (${sch.signals.toLocaleString()} / ${sch.needed.toLocaleString()}).` : " — enough is new, so tonight.");
}
async function loadTasteQuality() {
  const box = $("#taste-quality"); if (!box) return;
  let q;
  try { q = await api("/api/taste/quality?" + new URLSearchParams(pparam())); } catch { return; }
  const L = q.latest, hist = q.history || [];
  if (!L) {
    box.innerHTML = `<div class="tq-empty"><b>How well does Peaks know your taste?</b>
      <span class="muted">Click <b>Train &amp; measure</b> (top right) — it retrains and tests itself on scenes it didn't learn from. A background job; it can take a few minutes.</span>
      </div><div id="tq-run" class="tq-run" hidden></div>`;
    tqLast = null; tqCaption();
    followMeasure();
    return;
  }
  tqLast = L.ts; tqCaption();
  const prev = hist.length > 1 ? hist[hist.length - 2] : null;
  const delta = (k) => {        // percentage points, or raw for AUC
    if (!prev || prev[k] == null || L[k] == null) return "";
    const d = Math.round((L[k] - prev[k]) * 100);
    const txt = k === "auc" ? (Math.abs(d) / 100).toFixed(2) : `${Math.abs(d)} pts`;
    const word = W.trend(L[k] - prev[k]);
    return d ? `<span class="tq-d ${d > 0 ? "up" : "down"}" title="${d > 0 ? "▲" : "▼"} ${txt} since the previous training">${d > 0 ? "▲" : "▼"} ${esc(word)}${fig(txt)}</span>` : "";
  };
  const model = `${L.kind === "mlp" ? "Non-linear" : "Linear"} model` + (L.context ? ` · sees ±${L.context} frames of context` : " · single frames");
  const src = TQ_SRC.filter(([k]) => (L.sources || {})[k]).map(([k, n]) => `<span><b>${L.sources[k].toLocaleString()}</b> ${n}</span>`).join("");
  box.innerHTML = `
    <div class="tq-head"><b>How well Peaks knows your taste</b>
      <span class="faint small">measured on scenes it didn't train on · ${new Date(L.ts * 1000).toLocaleString()}</span></div>
    <div class="tq-stats">
      <div class="tq-stat" title="For scenes holding a moment you loved: how often the model's single best moment lands within 10 s of it — i.e. how often 'jump to peak' lands on your moment.">
        <span class="tq-l">Jump-to-peak lands on your moment</span><span class="tq-v tq-words">${L.peak_hit != null ? esc(cap(W.howOften(L.peak_hit))) + fig(pct(L.peak_hit)) : "—"}${delta("peak_hit")}</span>
        <span class="faint small">${L.peak_scenes || 0} held-out scenes</span></div>
      <div class="tq-stat" title="Of the held-out moments it ranks highest, the share you actually loved.">
        <span class="tq-l">Its top picks you love</span><span class="tq-v tq-words">${L.p_at_50 != null ? esc(cap(W.lift(L.p_at_50, L.base_rate) || pct(L.p_at_50))) + fig(pct(L.p_at_50)) : "—"}${delta("p_at_50")}</span>
        <span class="faint small">${L.p_at_50 != null ? `${esc(W.share(L.p_at_50))} of its top picks are moments you loved · ${pct(L.base_rate)} by chance` : ""}</span></div>
      <div class="tq-stat" title="ROC-AUC on your held-out ratings: 1.0 = always ranks a loved moment above a passed one, 0.5 = coin flip.">
        <span class="tq-l">Tells love from pass</span><span class="tq-v tq-words">${L.auc != null ? esc(W.auc(L.auc)) + fig(L.auc.toFixed(2)) : "—"}${delta("auc")}</span>
        ${tqSpark(hist, "auc") || '<span class="faint small">AUC · trend after 2 trainings</span>'}</div>
    </div>
    <div class="tq-src"><span class="faint">Learned from</span>${src}${L.pu_dropped ? `<span class="faint">· ${L.pu_dropped} look-alike background frames set aside</span>` : ""}</div>
    <div class="tq-saved small" id="tq-saved"></div>
    <div class="faint small">${model}</div>
    <div class="faint small">${esc(tqScheduleLine(q.schedule))}</div>
    <div id="tq-run" class="tq-run" hidden></div>`;
  followMeasure();
  loadSavedAudit();
}
// your saved moments: how they feed the taste model (live from Stash)
async function loadSavedAudit() {
  const el = $("#tq-saved"); if (!el) return;
  let a;
  try { a = await api("/api/taste/saved?" + new URLSearchParams(pparam())); } catch { return; }
  const n = (x) => x.toLocaleString();
  const parts = [`<b>${n(a.saved)}</b> saved moments`];
  if (a.rated) parts.push(`${n(a.rated)} already count as a 👍 — weighted double`);
  if (a.added) parts.push(`${n(a.added)} added from the save alone`);
  if (a.not_embedded) parts.push(`${n(a.not_embedded)} on scenes not embedded yet`);
  el.innerHTML = parts.join(" · ")
    + (a.auto_ignored ? `<br><span class="faint">Ignoring ${n(a.auto_ignored)} auto-detected markers the old “Write markers” scorer left under “${esc(a.tag)}” — they aren't your picks.</span>` : "");
}
document.addEventListener("click", async (e) => {
  if (e.target.closest(".tq-train")) startMeasure();
  const c = e.target.closest("[data-cancel-job]");
  if (c) {
    c.disabled = true;
    try { await api(`/api/jobs/${encodeURIComponent(c.dataset.cancelJob)}/cancel`, { method: "POST" }); toast("Cancelling after the current step…"); }
    catch (err) { toast(err.message, true); }
  }
});
wireTabs("#taste-tabs", (t) => { if (t === "picker" && !pickItems.length) loadPicks(); if (t === "teach") loadNextSwipe(); });
wireTabs("#ins-tabs", (t) => { if (t === "coverage") openExperimental(); if (t === "health") openHealth(); });

// --- Library health: saves, the model's view of each tier, the trim pool -------------
async function openHealth() {
  const box = $("#health-body"); if (!box) return;
  box.innerHTML = '<p class="dim">Reading your library…</p>';
  let h;
  try { h = await api("/api/library/health"); } catch (e) { box.innerHTML = `<p class="dim">${esc(e.message)}</p>`; return; }
  const n = (x) => (x ?? 0).toLocaleString();
  const pct = (a, b) => (b ? Math.round(100 * a / b) + "%" : "—");
  const order = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected"];
  const rows = order.filter((t) => h.per_tier[t]).map((t) => {
    const x = h.per_tier[t];
    const w = x.scenes ? 100 * x.with_saves / x.scenes : 0;
    return `<tr><td>${tierChip(t)}</td><td>${n(x.scenes)}</td>
      <td><div class="hb"><i style="width:${w.toFixed(1)}%"></i></div><span>${esc(x.with_saves_words || "")}${fig(pct(x.with_saves, x.scenes))}</span></td>
      <td>${x.median_best != null ? (x.median_word ? `<span class="tw tw-${x.median_band}">${esc(x.median_word)}</span>${fig(Math.round(x.median_best * 100) + "%")}` : Math.round(x.median_best * 100) + "%") : "—"}</td></tr>`;
  }).join("");
  const bk = h.save_buckets, bmax = Math.max(1, ...Object.values(bk));
  const bars = Object.entries(bk).map(([k, v]) =>
    `<div class="hbk"><i style="height:${(100 * v / bmax).toFixed(1)}%"></i><b>${n(v)}</b><span>${k}</span></div>`).join("");
  const t = h.trim, a = t.ages;
  box.innerHTML = `
    <div class="hgrid">
      <div class="card pad hcard"><span class="tq-l">Scenes with saved moments</span>
        <span class="tq-v tq-words">${esc(cap(h.with_saves_words || ""))}${fig(pct(h.with_saves, h.scenes))}</span>
        <span class="faint small">${n(h.with_saves)} of ${n(h.scenes)} scenes · ${n(h.saves_total)} saves</span></div>
      <div class="card pad hcard hlink" data-open-view="saved"><span class="tq-l">Ready for ${esc(gradeName("legendaire"))}</span>
        <span class="tq-v">${n(h.legendaire_candidates)}</span>
        <span class="faint small">scenes with saves that aren't ${esc(gradeName("legendaire"))} yet → review</span></div>
      <div class="card pad hcard hlink" data-open-view="passed"><span class="tq-l">Passed over on the megaboard</span>
        <span class="tq-v">${n(h.passed_over)}</span>
        <span class="faint small">shown often, never saved or explored → review</span></div>
      <div class="card pad hcard hlink" data-open-view="trim"><span class="tq-l">Trim pool</span>
        <span class="tq-v">${n(t.count)}</span>
        <span class="faint small">no saves, past ${h.grace_days} days · ${fmtBytes(t.bytes)} · ${n(t.suggest_reject)} suggested to reject → review</span></div>
    </div>
    <div class="hgrid two">
      <div class="card pad"><h3>By tier</h3>
        <table class="htab"><thead><tr><th>Tier</th><th>Scenes</th><th>With saved moments</th><th title="How the taste model rates a typical scene in this tier (its median best moment, against all your scenes) — do your grades and the model agree?">Typical best moment</th></tr></thead>
        <tbody>${rows}</tbody></table>
        ${h.weak_legendaire ? `<p class="faint small">${n(h.weak_legendaire)} ${esc(gradeName("legendaire"))} scenes have a best moment in the library's bottom quarter — the model disagrees with you there (your grade wins; it's a hint for the model).</p>` : ""}</div>
      <div class="card pad"><h3>Saved moments per scene</h3><div class="hbars">${bars}</div>
        <h3 style="margin-top:14px">Scenes without saves, by age</h3>
        <p class="small">${n(a.new)} new (under ${h.grace_days} days, protected) · ${n(a["30d-6mo"])} 30 days–6 months · ${n(a["6mo+"])} over 6 months${a.unknown ? ` · ${n(a.unknown)} unknown age` : ""}${t.unembedded ? ` · ${n(t.unembedded)} not embedded yet (can't be judged)` : ""}</p></div>
    </div>${whoHealth(h.who)}`;
}
// Library health: the performers and studios you grade best and worst
function whoHealth(w) {
  if (!w) return "";
  const tab = (list, kind) => list.length ? `<table class="htab wtab"><thead><tr><th>${kind === "studio" ? "Studio" : "Performer"}</th><th title="Graded scenes (incl. deleted rejects)">Graded</th><th title="Share not rejected">Kept</th><th>${esc(className("exceptionnelle"))}+</th></tr></thead><tbody>${
    list.map((x) => `<tr class="hlink" data-who-${kind}="${esc(x.name)}" title="Open in Catalogue"><td>${esc(x.name)}<br><span class="wr-${x.tone} small">${esc(x.verdict || "")}</span></td><td>${x.n}</td><td>${esc(x.kept_words || "")}${fig(Math.round(x.keep * 100) + "%")}</td><td>${esc(x.top_words || "")}${fig(Math.round(x.top * 100) + "%")}</td></tr>`).join("")}</tbody></table>`
    : '<p class="faint small">None yet.</p>';
  const block = (d, kind, label) => `<div class="card pad"><h3>${label}</h3>
      <p class="faint small">${d.count.toLocaleString()} with ${w.min_graded}+ graded scenes · ranked against your library average</p>
      <h4>Best</h4>${tab(d.top, kind)}<h4>Worst</h4>${tab(d.bottom, kind)}</div>`;
  return `<div class="hgrid two">${block(w.performers, "performer", "Performers")}${block(w.studios, "studio", "Studios")}</div>
    <p class="faint small">Records are smoothed: a handful of grades counts for little until there are more. Click a row to review them in Catalogue.</p>`;
}
const cap = (t) => (t ? t[0].toUpperCase() + t.slice(1) : t);
function tierChip(t) { return `<span class="tchip tc-${t}">${esc(TIER_NAMES[t] || t)}</span>`; }
document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-who-performer],[data-who-studio]"); if (!c) return;
  go("catalogue");
  applyCatParams({ performer: c.dataset.whoPerformer || "", studio: c.dataset.whoStudio || "" });
});
document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-open-view]"); if (!c) return;
  cat.view = c.dataset.openView; cat.tier = ""; cat.isNew = false;
  go("catalogue"); openCatalogue();
});

async function openForYou() {
  if (!profilesLoaded) await loadProfiles();
  tdLoad();                  // today's decisions + the to-do board
  await loadForYou(false);   // the small top-moments strip (changes daily)
  loadTasteBands();          // colours the strip's tiles by taste band (cheap)
}

// --- For You: today's decisions -------------------------------------------------
// One decision at a time from Peaks' suggestions (saves → Légendaire, let-go
// suggestions, new scenes, second opinions…), a fixed mixed set of 20 per day.
// The video autoplays the scene's best moments; 1–5 grade, Enter accepts Peaks'
// pick, S skips, U undoes. Every ~6 answers, a quick "teach Peaks" 👍/👎.
const td = { queue: [], answered: [], job: null, progress: null, sinceTeach: 0, teach: null,
  moments: [], mi: 0, hop: null, loading: false, board: null, labels: {} };
const TD_TEACH_EVERY = 6;
const TD_SHORT = { reject: "Reject", upscale: "Upscale", merveilleuse: "Merv.", exceptionnelle: "Except.", legendaire: "Légend." };
const TD_QUESTION = {
  saved: "You saved moments here — make it Légendaire?",
  reject: "Let it go?",
  new: "New scene — what tier is it?",
  promote: "Peaks rates it above your grade — promote it?",
  second: "Peaks rates it below your grade — still this tier?",
  anomaly: "5★ with an unusual O-count — which tier is it?",
  trim: "No saved moments — keep it?",
};
async function tdLoad({ more = false, keep = false } = {}) {
  if (td.loading) return;
  td.loading = true;
  try {
    const qs = new URLSearchParams({ n: 4 });
    if (td.job) qs.set("job", td.job);
    if (more) qs.set("more", "true");
    const d = await api("/api/today?" + qs);
    const have = new Set(td.queue.map((x) => x.scene_id));
    td.queue = keep ? td.queue.concat(d.next.filter((x) => !have.has(x.scene_id))) : d.next;
    td.progress = d.progress; td.remaining = d.remaining; td.board = d.board; td.labels = d.labels || {};
    tdWeek(d.week); tdBoard();
    if (!keep) tdShow();
  } catch (e) { $("#td-body").innerHTML = `<p class="dim">${esc(e.message)}</p>`; }
  finally { td.loading = false; }
}
function tdWeek(w) {
  if (!w) return;
  const bits = [`${w.decisions.toLocaleString()} decision${w.decisions === 1 ? "" : "s"}`];
  if (w.promoted) bits.push(`${w.promoted} promoted`);
  if (w.let_go) bits.push(`${w.let_go} let go`);
  if (w.freed) bits.push(`${fmtBytes(w.freed)} freed`);
  $("#td-week").textContent = "This week: " + bits.join(" · ");
}
function tdProgress() {
  const p = td.progress, el = $("#td-prog"), bar = $("#td-bar");
  if (td.job) { el.textContent = `${(td.remaining || 0).toLocaleString()} left in this list`; bar.style.width = "0"; return; }
  if (!p) return;
  el.textContent = p.done >= p.goal ? `${p.done} today — goal met 🎉` : `${p.done} of ${p.goal} today`;
  bar.style.width = Math.min(100, 100 * p.done / Math.max(1, p.goal)) + "%";
}
function tdStopVideo() {
  clearInterval(td.hop); td.hop = null;
  const v = $("#td-video"); v.pause(); v.removeAttribute("src"); v.load();
  v.hidden = true; v.dataset.sid = ""; $("#td-spin").hidden = true;
}
function tdShow() {
  tdProgress();
  $("#td-job").textContent = td.job ? (td.labels[td.job] || td.job) + " · " : "";
  if (!td.job && td.sinceTeach >= TD_TEACH_EVERY) return tdShowTeach();
  $("#td-teach").hidden = true;
  const x = td.queue[0];
  if (!x) return tdShowDone();
  $("#td-job").textContent = td.job ? `${td.labels[td.job] || td.job} · just this list` : (td.labels[x.job] || x.job);
  const sub = [(x.performers || []).slice(0, 3).join(", "), x.studio, (x.date || "").slice(0, 4)].filter(Boolean).join(" · ");
  const sug = x.suggest || (x.why ? { grade: x.pick, why: x.why, detail: x.detail } : null);
  const why = [
    sug && sug.why ? `<div class="cat-sug">${sug.grade ? `Suggest <b>${esc(gradeName(sug.grade))}</b> — ` : ""}${textNum(sug.why, sug.detail)}</div>` : "",
    x.pred ? `<div class="cat-pred">${predHTML(x.pred)}${keeperHTML(x.pred) ? " · " + keeperHTML(x.pred) : ""}</div>` : "",
    sigLine(x), whoLines(x, 4),
  ].filter(Boolean).join("");
  const cur = x.tier === "rejected" ? "reject" : x.tier;
  $("#td-body").innerHTML = `
    <h2 class="td-title">${tierBadge(x.rating100, x.o_counter, { showUnreviewed: true })} ${esc(x.title || "scene " + x.scene_id)}</h2>
    <div class="dim small">${esc(sub)}${x.duration ? " · " + fmt(x.duration) : ""}${x.size ? " · " + fmtBytes(x.size) : ""}</div>
    <div class="td-q">${esc(TD_QUESTION[x.job] || "What tier is it?")}</div>
    <div class="td-why">${why}</div>
    <div class="td-grades">${RV_GRADES.map(([g], k) => `<button data-g="${g}" class="g-${g} ${g === x.pick ? "sug" : ""} ${g === cur ? "cur" : ""}" title="${esc(gradeName(g))} (${k + 1})"><b>${k + 1}</b>${esc(TD_SHORT[g])}</button>`).join("")}</div>
    <div class="td-actions">
      ${x.pick ? `<button class="btn pri sm" id="td-accept">✓ ${esc(gradeName(x.pick))} <kbd>Enter</kbd></button>` : ""}
      ${TD_KEEPABLE.has(x.job) ? `<button class="btn sm" id="td-keep" title="It's right where it is — don't suggest it again unless something new happens">Keep as ${esc(gradeName(cur))} <kbd>0</kbd></button>` : ""}
      <button class="btn sm ghost" id="td-skip">Skip <kbd>S</kbd></button>
      <button class="btn sm ghost" id="td-undo" ${td.answered.length ? "" : "disabled"}>Undo <kbd>U</kbd></button>
      <button class="btn sm ghost" id="td-open" title="Open in the Review player">Open in Review</button>
    </div>
    <div class="td-keys">1–5 grade · Enter accept · 0 keep as is · S skip · U undo · P / click the still: play peaks · Space pause · M sound · ←/→ moments</div>`;
  $("#td-body").querySelectorAll(".td-grades button").forEach((b) => b.onclick = () => tdAnswer(b.dataset.g));
  $("#td-accept")?.addEventListener("click", () => tdAnswer(x.pick));
  $("#td-keep")?.addEventListener("click", () => tdAnswer(null, true));
  $("#td-skip").onclick = tdSkip; $("#td-undo").onclick = tdUndo;
  $("#td-open").onclick = () => { go("catalogue"); applyCatParams({ q: x.title || "" }); };
  tdPlay(x);
}
// a still of the scene first (its best moment); click it — or a moment, or P —
// to play its peak moments, best first, hopping every 12 s (muted until M)
function tdPlay(x) {
  tdStopVideo();
  const ms = (x.moments || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
  td.moments = ms.length ? ms.map((m) => m.t) : [0];
  td.streams = ms.map((m) => m.stream);          // each moment's own stream (works when Stash transcodes)
  td.mi = 0;
  td.still = ms[0] && ms[0].thumb ? ms[0].thumb : `/api/scene/${encodeURIComponent(x.scene_id)}/cover`;
  const cover = `/api/scene/${encodeURIComponent(x.scene_id)}/cover`;
  const st = $("#td-still");
  st.hidden = false;
  st.innerHTML = `<img class="td-still-img" src="${td.still}" alt="" onerror="if(!this.dataset.f){this.dataset.f=1;this.src='${cover}'}" />
    <div class="td-play"><span>▶</span>Play its peak moments</div>
    ${ms.length > 1 ? `<div class="td-thumbs">${ms.map((m, i) => `<img data-i="${i}" src="${m.thumb}" title="${fmt(m.t)}" alt="" onerror="this.remove()" />`).join("")}</div>` : ""}`;
  st.onclick = (e) => { const t = e.target.closest(".td-thumbs img"); tdStart(t ? +t.dataset.i : 0); };
  $("#td-dots").innerHTML = "";
}
function tdStart(i = 0) {
  const x = td.queue[0]; if (!x) return;
  td.mi = Math.max(0, Math.min(i, td.moments.length - 1));
  $("#td-still").hidden = true;
  const v = $("#td-video");
  v.hidden = false; v.poster = td.still || "";
  $("#td-dots").innerHTML = td.moments.length > 1 ? td.moments.map((_, k) => `<i data-i="${k}" class="${k === td.mi ? "on" : ""}"></i>`).join("") : "";
  $("#td-dots").querySelectorAll("i").forEach((d) => d.onclick = () => tdJump(+d.dataset.i - td.mi));
  tdLoadMoment();
  clearInterval(td.hop);
  td.hop = setInterval(() => { if (!v.paused) tdJump(1); }, 12000);
}
// load one moment and play it. Direct play serves the whole file (seek to the
// moment); a transcode serves a stream that starts AT the moment (play from 0) —
// told apart by its length, as the megaboard does.
function tdLoadMoment() {
  const x = td.queue[0]; if (!x) return;
  const v = $("#td-video"), t = td.moments[td.mi] || 0;
  const url = td.streams[td.mi] || x.stream;
  $("#td-spin").hidden = false; $("#td-spin").textContent = "loading…";
  v.onplaying = () => { $("#td-spin").hidden = true; };
  v.onwaiting = () => { $("#td-spin").hidden = false; };
  v.onerror = () => {
    $("#td-spin").hidden = false;
    $("#td-spin").innerHTML = 'Couldn\'t play this here — <a href="#" id="td-openrv">open in Review</a>';
    $("#td-openrv")?.addEventListener("click", (e) => { e.preventDefault(); $("#td-open")?.click(); });
  };
  v.onloadedmetadata = () => {
    const full = x.duration ? v.duration >= 0.9 * x.duration : !isFinite(v.duration) ? false : v.duration > t + 30;
    if (full && t) { try { v.currentTime = t; } catch { /* not seekable: plays from the start */ } }
    v.play().catch(() => {});
  };
  v.preload = "auto";
  v.src = url;
  v.load();
  v.play().catch(() => {});        // the click is the user gesture; loading starts now
  $("#td-dots").querySelectorAll("i").forEach((d, k) => d.classList.toggle("on", k === td.mi));
}
function tdJump(step) {
  if (!td.moments.length) return;
  if ($("#td-video").hidden) return tdStart(Math.max(0, step > 0 ? 0 : td.moments.length - 1));
  td.mi = (td.mi + step + td.moments.length) % td.moments.length;
  tdLoadMoment();
}
const TD_KEEPABLE = new Set(["saved", "reject", "promote", "second", "trim"]);
async function tdAnswer(grade, keep = false) {
  const x = td.queue[0]; if (!x || (!grade && !keep)) return;
  if (keep && !TD_KEEPABLE.has(x.job)) return;
  try { if (keep) await keepAsIs(x.scene_id, "U"); else await applyGrade(x.scene_id, grade, "U"); }
  catch (e) { return toast(e.message, true); }
  td.queue.shift(); td.answered.push(x); td.sinceTeach++;
  if (td.progress && !td.job) td.progress.done++;
  if (td.job) td.remaining = Math.max(0, (td.remaining || 1) - 1);
  api(`/api/today/done?scene_id=${encodeURIComponent(x.scene_id)}`, { method: "POST" }).catch(() => {});
  tdShow();
  if (td.queue.length < 2) tdLoad({ keep: true });
}
async function tdSkip() {
  const x = td.queue.shift(); if (!x) return;
  await api(`/api/today/skip?scene_id=${encodeURIComponent(x.scene_id)}`, { method: "POST" }).catch(() => {});
  if (td.queue.length < 2) await tdLoad({ keep: true });
  if (td.queue[0] && td.queue[0].scene_id === x.scene_id && td.queue.length > 1) td.queue.push(td.queue.shift());
  tdShow();
}
async function tdUndo() {
  const x = td.answered.pop(); if (!x) return toast("nothing to undo");
  const fresh = await undoGrade(); if (!fresh) { td.answered.push(x); return; }
  await api(`/api/today/done?scene_id=${encodeURIComponent(x.scene_id)}&undo=true`, { method: "POST" }).catch(() => {});
  td.queue.unshift({ ...x, ...fresh });
  if (td.progress && !td.job) td.progress.done = Math.max(0, td.progress.done - 1);
  td.sinceTeach = Math.max(0, td.sinceTeach - 1);
  tdShow();
}
function tdShowDone() {
  tdStopVideo();
  const p = td.progress || {};
  $("#td-body").innerHTML = td.job
    ? `<div class="td-done"><h2>Nothing left here 🎉</h2><p class="dim">That list is clear.</p>
        <button class="btn pri" id="td-back">← Back to today</button></div>`
    : `<div class="td-done"><h2>Done for today 🎉</h2><p class="dim">${p.done || 0} decisions made. Your library thanks you.</p>
        <button class="btn pri" id="td-more">Keep going →</button></div>`;
  $("#td-back")?.addEventListener("click", () => { td.job = null; td.queue = []; tdLoad(); });
  $("#td-more")?.addEventListener("click", () => tdLoad({ more: true }));
}
// "teach Peaks": the frame the taste model is least sure about
async function tdShowTeach() {
  tdStopVideo();
  $("#td-job").textContent = "Teach Peaks";
  let d; try { d = await api("/api/foryou/next?" + new URLSearchParams(pparam())); } catch { d = null; }
  td.teach = d && d.item;
  if (!td.teach) { td.sinceTeach = 0; return tdShow(); }
  $("#td-still").hidden = true;
  const t = $("#td-teach"); t.hidden = false;
  t.innerHTML = `<img src="${td.teach.thumb}" alt="" />`;
  $("#td-dots").innerHTML = "";
  $("#td-body").innerHTML = `<div class="td-q">Quick one: is this your kind of moment?</div>
    <div class="td-why"><span>Peaks is least sure about moments like this — one tap teaches it.</span>
      <span class="faint">${esc(td.teach.title || "")} · ${fmt(td.teach.time)}</span></div>
    <div class="td-actions" style="margin-top:14px"><button class="btn pri" id="td-love">👍 Love <kbd>↑</kbd></button>
      <button class="btn" id="td-pass">👎 Pass <kbd>↓</kbd></button><button class="btn ghost" id="td-tskip">Not now</button></div>
    <div class="td-keys">↑ love · ↓ pass · S not now</div>`;
  $("#td-love").onclick = () => tdTeach(1); $("#td-pass").onclick = () => tdTeach(0);
  $("#td-tskip").onclick = () => { td.sinceTeach = 0; tdShow(); };
}
async function tdTeach(label) {
  const h = td.teach; if (!h) return;
  try {
    await api("/api/label?" + new URLSearchParams(pparam({ key: h.key, t: (+h.time).toFixed(2), label,
      ...(h.scene_id ? { scene_id: h.scene_id } : {}) })), { method: "POST" });
    toast(label ? "👍 noted — more like that" : "👎 noted — less like that");
  } catch (e) { toast(e.message, true); }
  td.teach = null; td.sinceTeach = 0; tdShow();
}
// the to-do board: every job with its count; upkeep opens the right screen
function tdBoard() {
  const b = td.board; if (!b) return;
  const j = b.jobs || {}, n = (x) => (x ?? 0).toLocaleString();
  const card = (grp, key, label, count, desc, act) => `<div class="td-card ${td.job === key ? "on" : ""} ${count ? "" : "zero"}" data-act="${act}" data-key="${key}">
      <span class="g">${grp}</span><span class="n">${n(count)}</span><span class="l">${esc(label)}</span><span class="d">${desc}</span></div>`;
  $("#td-board").innerHTML = [
    card("Confirm", "saved", "Saved → Légendaire", j.saved, "you saved moments in these", "job"),
    card("Confirm", "reject", "Suggested to let go", j.reject, "passed over or among your weakest", "job"),
    card("Grade", "new", "New scenes", j.new, `${n(b.likely)} look like keepers`, "job"),
    card("Second opinion", "promote", "Promotion candidates", j.promote, "Peaks rates them above your grade", "job"),
    card("Second opinion", "second", "Second look", j.second, "Peaks rates them below your grade", "job"),
    card("Second opinion", "anomaly", "Anomalies", j.anomaly, "5★ with an unusual O-count", "job"),
    card("Trim", "trim", "No saved moments", j.trim, "past 30 days, nothing saved", "job"),
    card("Upkeep", "dupes", "Duplicates", b.dupes, b.dupes == null ? "not checked yet — open to scan" : "groups to resolve", "dupes"),
    card("Upkeep", "rejected", "Rejected, awaiting delete", b.rejected.count, b.rejected.bytes ? `free ${fmtBytes(b.rejected.bytes)}` : "nothing to free", "rejected"),
    card("Upkeep", "conflict", "Tag conflicts", b.conflicts, "tier tags your renamer can't act on", "view"),
    card("Upkeep", "quality", "Quality check", b.quality, "below the quality you usually keep", "view"),
    card("Teach", "teach", "Teach Peaks", 1, "a quick 👍/👎 on what it's unsure of", "teach"),
  ].join("");
}
$("#td-board")?.addEventListener("click", (e) => {
  const c = e.target.closest(".td-card"); if (!c) return;
  const { act, key } = c.dataset;
  if (act === "job") { td.job = td.job === key ? null : key; td.queue = []; tdLoad(); window.scrollTo(0, 0); $("#content")?.scrollTo(0, 0); }
  else if (act === "dupes") go("dupes");
  else if (act === "rejected") { cat.view = ""; cat.tier = "rejected"; go("catalogue"); openCatalogue(); }
  else if (act === "view") { cat.view = key; cat.tier = ""; go("catalogue"); openCatalogue(); }
  else if (act === "teach") { td.sinceTeach = TD_TEACH_EVERY; tdShow(); }
});
$("#td-sound")?.addEventListener("click", () => {
  const v = $("#td-video"); v.muted = !v.muted; $("#td-sound").textContent = v.muted ? "🔇" : "🔊";
});
document.addEventListener("keydown", (e) => {
  if (!$("#foryou")?.classList.contains("active") || !$("#viewer").hidden || !$("#cmdk").hidden) return;
  if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase(), v = $("#td-video");
  if (td.teach && !$("#td-teach").hidden) {
    if (k === "arrowup") { e.preventDefault(); tdTeach(1); }
    else if (k === "arrowdown") { e.preventDefault(); tdTeach(0); }
    else if (k === "s") { e.preventDefault(); td.teach = null; td.sinceTeach = 0; tdShow(); }
    return;
  }
  const x = td.queue[0];
  if (k >= "1" && k <= "5") { e.preventDefault(); tdAnswer(RV_GRADES[+k - 1][0]); }
  else if (k === "enter" && x && x.pick) { e.preventDefault(); tdAnswer(x.pick); }
  else if (k === "s") { e.preventDefault(); tdSkip(); }
  else if (k === "u") { e.preventDefault(); tdUndo(); }
  else if (k === "0") { e.preventDefault(); tdAnswer(null, true); }
  else if (k === "p") { e.preventDefault(); if (v.hidden) tdStart(0); else { tdStopVideo(); if (td.queue[0]) tdPlay(td.queue[0]); } }
  else if (k === " ") { e.preventDefault(); if (v.hidden) tdStart(0); else if (v.paused) v.play().catch(() => {}); else v.pause(); }
  else if (k === "m") { e.preventDefault(); $("#td-sound").click(); }
  else if (k === "arrowright") { e.preventDefault(); tdJump(1); }
  else if (k === "arrowleft") { e.preventDefault(); tdJump(-1); }
});
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
  openViewerAt(i);              // reuse the viewer (stream, volume, heat)
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
    toast(trainSummary(s)); loadTasteQuality();
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
      ? "Permanently delete your ENTIRE taste profile — every 👍/👎 rating and the trained model?\n\nThis cannot be undone. (Your saved moments in Stash are kept.)"
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
  const teachOpen = $("#taste")?.classList.contains("active") && !$("#swipe-panel").hidden;
  if (!teachOpen) return;                                  // (For You has its own keys now)
  if (!$("#cmdk").hidden || e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") { e.preventDefault(); swipeRate(1); }   // → = 👍 love
  else if (e.key === "ArrowLeft") { e.preventDefault(); swipeRate(0); }  // ← = 👎 pass
  else if (e.key === "ArrowDown") { e.preventDefault(); loadNextSwipe(); }
});

// --- Experimental: taste coverage / validation -----------------------------
let expData = null;
function updateExpFloorLabel() {
  const v = +$("#exp-floor").value;
  $("#exp-floor-val").innerHTML = v > 0 ? esc(W.floorWord(v, SCALE && SCALE.moment)) + fig(Math.round(v * 100) + "%") : "off";
}
async function openExperimental() {
  const body = $("#exp-body");
  const f = readFloor();
  if (f != null && isFinite(f)) $("#exp-floor").value = Math.min(Math.max(f, 0), 0.95);
  updateExpFloorLabel();
  body.innerHTML = '<p class="dim">Analyzing your taste coverage…</p>';
  $("#exp-status").textContent = "";
  try {
    SCALE = (await W.tasteScale()) || SCALE;
    expData = await api("/api/experimental/taste" + (PROFILE.isDefault() ? "" : `?profile=${encodeURIComponent(PROFILE.name)}`));
  } catch (e) { body.innerHTML = `<p class="dim">${esc(e.message)}</p>`; return; }
  renderExperimental();
}
function expCoverageAt(floor) {
  if (floor <= 0) return expData.scenes;
  let covered = expData.scenes;
  for (const p of (expData.coverage_curve || [])) if (p.floor <= floor) covered = p.covered;
  return covered;
}
function renderExpHealth() {
  const h = expData.health || {};
  const cards = [
    ["Embedded scenes", h.embedded_scenes ?? "—"],
    ["Library total", h.library_total ?? "—"],
    ["Not yet embedded", h.pending ?? "—"],
    ["Failed scenes", h.failed ?? 0],
    ["Taste examples", h.taste_examples ?? 0],
    ["Ratings 👍 / 👎", `${h.taste_positive ?? "—"} / ${h.taste_negative ?? "—"}`],
    ["Scorer", h.scorer || "—"],
    ["Text search (CLIP)", h.has_clip ? "yes" : "no"],
  ];
  const warn = [
    (h.taste_examples != null && h.taste_examples < 25) ? `⚠ Your taste is built from only ${h.taste_examples} example(s) — it will naturally miss most of the library until you rate more.` : "",
    h.pending ? `⚠ ${h.pending} scene(s) aren't embedded yet, so they can't be scored.` : "",
    h.failed ? `⚠ ${h.failed} scene(s) failed to embed (see Settings → Failed scenes).` : "",
  ].filter(Boolean).join(" ");
  return `<div class="panel"><h3 class="exp-h">Pipeline health</h3>
    <div class="cards">${cards.map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${esc(String(v))}</div></div>`).join("")}</div>
    ${warn ? `<p class="dim" style="margin-top:8px">${warn}</p>` : ""}</div>`;
}
function renderExperimental() {
  const body = $("#exp-body");
  if (!expData || !expData.ready) {
    body.innerHTML = `<div class="panel"><p class="dim">${esc(expData?.reason || "Not ready.")}</p></div>` + renderExpHealth();
    return;
  }
  const floor = +$("#exp-floor").value;
  const total = expData.scenes || 0;
  const covered = expCoverageAt(floor);
  const uncovered = total - covered;
  const pct = total ? Math.round((uncovered / total) * 100) : 0;
  const floorTxt = floor > 0 ? Math.round(floor * 100) + "%" : "0%";
  const floorWords = W.floorWord(floor, SCALE && SCALE.moment).toLowerCase();
  const headline = floor > 0
    ? `${esc(cap(W.share(total ? covered / total : 0)))} of your scenes <span class="dim">have a moment that's ${esc(floorWords)}${fig(`${covered.toLocaleString()} of ${total.toLocaleString()} · ${100 - pct}% · ≥ ${floorTxt}`)}</span>`
    : `Every scene counts <span class="dim">— raise the taste floor to see how much of your library clears it</span>`;
  const coverage = `<div class="panel">
    <div class="exp-big">${headline}</div>
    <div class="dim">${covered.toLocaleString()} scene(s) are covered by your taste at this floor${fig(`scores span ${Math.round(expData.score_range[0] * 100)}%–${Math.round(expData.score_range[1] * 100)}%`)}.</div></div>`;

  const dist = expData.distribution || [];
  const dmax = Math.max(1, ...dist.map((b) => b.n));
  const histBars = dist.map((b) => {
    const hh = Math.round((b.n / dmax) * 100);
    const bw = W.tasteBand(b.lo, SCALE && SCALE.scene);
    return `<div class="exp-bar ${b.lo >= floor ? "" : "below"}" style="height:${hh}%" title="${bw ? bw[1] + " · " : ""}${Math.round(b.lo * 100)}%+ · ${b.n} scene(s)"></div>`;
  }).join("");
  const histogram = `<div class="panel"><h3 class="exp-h">Per-scene best-score distribution</h3>
    <div class="exp-hist">${histBars}</div><div class="exp-axis"><span>0%</span><span>100%</span></div>
    <p class="dim">Each scene's single best moment. Grey bars (left of the floor) are the uncovered scenes.</p></div>`;

  const curve = expData.coverage_curve || [];
  const curveBars = curve.map((p) => {
    const hh = total ? Math.round((p.covered / total) * 100) : 0;
    return `<div class="exp-bar" style="height:${hh}%" title="floor ${W.floorWord(p.floor, SCALE && SCALE.moment)} (${Math.round(p.floor * 100)}%) · ${W.share(hh / 100)} covered (${p.covered.toLocaleString()} · ${hh}%)"></div>`;
  }).join("");
  const curvePanel = `<div class="panel"><h3 class="exp-h">Coverage vs. floor</h3>
    <div class="exp-hist">${curveBars}</div><div class="exp-axis"><span>floor 5%</span><span>95%</span></div>
    <p class="dim">How much of the library stays covered as the floor rises — the knee is a sensible floor.</p></div>`;

  const de = expData.density || {};
  const density = `<div class="panel"><h3 class="exp-h">Sampling density</h3>
    <p class="dim">Moments sampled per scene — typically ${de.median}${fig(`min ${de.min} · max ${de.max}`)}. ${de.sparse_scenes ? `⚠ ${de.sparse_scenes} scene(s) have fewer than 4 sampled moments (few chances to score — a denser embed interval would help).` : "Every scene has a healthy number of sampled moments."}</p></div>`;

  const w = expData.wall || [];
  const tiles = w.map((it) => `<div class="exp-tile" data-stream="${esc(it.stream || "")}" data-key="${esc(it.key)}" data-t="${it.t}" data-sid="${esc(String(it.scene_id))}">
      <img loading="lazy" src="${it.thumb}" onerror="this.style.display='none'" />
      <span class="exp-score">${W.tasteHTML(it.score, SCALE && SCALE.scene)}</span></div>`).join("");
  const wall = w.length ? `<div class="panel"><h3 class="exp-h">Least on-taste scenes <span class="dim">(their own best moment)</span></h3>
    <p class="dim">The ${w.length} scenes your taste scores lowest. Do they genuinely look "not you"? If good scenes are here, your taste needs more examples — not a bug. Click a still to play it.</p>
    <div class="exp-wall">${tiles}</div></div>` : "";

  body.innerHTML = coverage + renderExpHealth() + histogram + curvePanel + density + wall;
  body.querySelectorAll(".exp-tile").forEach((el) => el.addEventListener("click", () => {
    const hit = { stream: el.dataset.stream, time: +el.dataset.t, scene_id: el.dataset.sid, key: el.dataset.key, score: 0 };
    if (hit.stream) openViewer(hit);
  }));
}
$("#exp-floor")?.addEventListener("input", () => { updateExpFloorLabel(); if (expData && expData.ready) renderExperimental(); });
$("#exp-floor")?.addEventListener("change", () => writeFloor(+$("#exp-floor").value));
$("#btn-exp-refresh")?.addEventListener("click", () => openExperimental());

// --- Catalogue: grade & filter the library with your tiers -------------------
const CAT_PAGE = 60;
const CAT_CHIPS = ["unreviewed", "anomaly", "upscale", "merveilleuse", "exceptionnelle", "legendaire", "rejected"];
const cat = { tier: "unreviewed", view: "", items: [], total: 0, counts: {}, views: {}, model: null,
              focus: 0, loaded: false, busy: false, sel: new Set(), anchor: null, bulkBusy: false,
              isNew: false };
const CAT_VIEWS = [
  ["likely", "Likely keepers", "Unreviewed scenes that look most like your best tiers"],
  ["quality", "Quality check", "Unreviewed scenes below the quality of everything you've tiered — quick reject candidates"],
  ["promote", "Promotion candidates", "Merveilleuse scenes that look like your Exceptionnelle/Légendaire ones"],
  ["second", "Second look", "Exceptionnelle/Légendaire scenes that look more like Merveilleuse or below"],
  ["anomaly", "Anomaly suggestions", "5★ scenes with an O-count outside your scheme, with a suggested tier"],
  ["conflict", "Tag conflicts", "Scenes carrying two tier tags, or a tier tag that disagrees with their grade — the renamer can't file these. Re-grade to fix."],
  ["saved", "Saved → Légendaire", "Scenes you've saved moments in that aren't Légendaire yet — most saves first. One click grades them all."],
  ["trim", "No saved moments", "Scenes past their grace period with no saved moments, weakest best moment first — trim candidates. Légendaire and rejects are left out."],
  ["passed", "Passed over", "Scenes the megaboard keeps showing you that you never save, 👍, explore, pin or enlarge — most-shown first. Légendaire, rejects and new scenes are left out."],
];
const MODEL_FREE_VIEWS = new Set(["quality", "anomaly", "conflict", "saved", "trim", "passed"]);
const className = (c) => TIER_NAMES[c === "reject" ? "rejected" : c] || c;

const CAT_FILTERS = [["performer", "#cat-perf"], ["studio", "#cat-studio"], ["tag", "#cat-tag"],
  ["date_from", "#cat-from"], ["date_to", "#cat-to"], ["dur_min", "#cat-dmin"], ["dur_max", "#cat-dmax"]];
function catParams(offset, refresh) {
  const qs = new URLSearchParams({ offset, limit: CAT_PAGE, sort: $("#cat-sort").value });
  for (const [k, sel] of CAT_FILTERS) { const v = ($(sel)?.value || "").trim(); if (v) qs.set(k, v); }
  if (cat.view) qs.set("view", cat.view);
  else if (cat.tier) qs.set("tier", cat.tier);
  if (cat.isNew) qs.set("new", "true");
  const q = $("#cat-q").value.trim(); if (q) qs.set("q", q);
  const res = $("#cat-res").value; if (res) qs.set("res", res);
  const mb = $("#cat-mbps").value; if (mb !== "") qs.set("min_mbps", mb);
  if (refresh) qs.set("refresh", "true");
  return qs.toString();
}
async function openCatalogue({ append = false, refresh = false } = {}) {
  if (cat.busy) return;
  cat.busy = true;
  const offset = append ? cat.items.length : 0;
  $("#cat-status").textContent = refresh ? "re-reading grades from Stash…" : "loading…";
  if (!append) $("#cat-list").innerHTML = '<p class="dim">Reading your library…</p>';
  try {
    const d = await api("/api/catalogue?" + catParams(offset, refresh));
    TIER_NAMES = { ...TIER_NAMES, ...(d.names || {}) };
    cat.items = append ? cat.items.concat(d.items) : d.items;
    cat.total = d.total; cat.counts = d.counts; cat.views = d.views || {}; cat.model = d.model;
    cat.grace = d.grace || cat.grace;
    renderCatViewbar();
    setNavCounts(d);
    const nb = $("#btn-cat-new"), ing = d.ingest || {};
    if (nb) {
      nb.classList.toggle("on", cat.isNew);
      nb.title = ing.finished ? `Unreviewed scenes from the last ingest (${ing.finished.replace("T", " ").slice(0, 16)})` : "Unreviewed scenes from the last ingest";
    }
    cat.loaded = true;
    if (!append) { cat.focus = 0; cat.sel.clear(); cat.anchor = null; }
    renderCatChips(); renderCatTriage(); renderCatList(); renderCatBulk(); renderCatStorage(d.storage);
    const nf = CAT_FILTERS.filter(([, sel]) => ($(sel)?.value || "").trim()).length;
    $("#btn-cat-filters").textContent = `Filters${nf ? ` (${nf})` : ""} ${$("#cat-filters").hidden ? "▾" : "▴"}`;
    $("#cat-status").textContent = `${cat.items.length.toLocaleString()} of ${d.total.toLocaleString()}`;
  } catch (e) {
    $("#cat-list").innerHTML = `<p class="dim">${esc(e.message)}</p>`;
    $("#cat-status").textContent = "";
  } finally { cat.busy = false; }
}
// sidebar counts — library-wide, refreshed after every library change (grades,
// undos, bulk, deletes, duplicates, tag syncs, restores, training, ingest), when a
// background job finishes, and every 30 s (changes from another tab or device).
// Grades made directly in Stash show after Catalogue → ⋯ → Re-read from Stash.
const side = { views: null, model: null, timer: null, busy: false, again: false };
function refreshSidebar(delay = 250) {
  clearTimeout(side.timer);
  side.timer = setTimeout(loadSidebar, delay);
}
async function loadSidebar() {
  if (side.busy) { side.again = true; return; }
  side.busy = true;
  try {
    const d = await api("/api/catalogue/summary");
    side.views = d.views; side.model = d.model;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v ? (+v).toLocaleString() : ""; };
    set("#nav-ct-cat", d.total);
    set("#nav-ct-review", d.counts.unreviewed);
    if (d.dupes != null) setDupeCount(d.dupes);
    const nb = $("#btn-cat-new");
    if (nb) {
      nb.hidden = !d.new && !cat.isNew;
      nb.innerHTML = `✦ New <span class="n">${(d.new || 0).toLocaleString()}</span>`;
    }
    if (!cat.model) cat.model = d.model;
    renderCatTriage();
  } catch { /* Stash unreachable — keep the last numbers */ }
  side.busy = false;
  if (side.again) { side.again = false; refreshSidebar(); }
}
setInterval(() => { if (!document.hidden) refreshSidebar(0); }, 30000);
// kept for existing callers: counts now come from the library-wide summary
function setNavCounts() { refreshSidebar(); }
function setDupeCount(n) {
  const el = $("#nav-ct-dupes"); if (!el) return;
  el.hidden = !n; el.textContent = n || "";
}
function renderCatChips() {
  const all = Object.values(cat.counts).reduce((a, b) => a + b, 0);
  const chip = (t, label, n) =>
    `<button class="cat-chip ${!cat.view && cat.tier === t ? "on" : ""} ${t ? "tier-" + t : ""}" data-t="${t}">${esc(label)} <span class="n">${(n || 0).toLocaleString()}</span></button>`;
  $("#cat-chips").innerHTML = chip("", "All", all) + CAT_CHIPS.map((t) => chip(t, TIER_NAMES[t], cat.counts[t])).join("");
  // ▶ Board / ⬇ Reel act on one keeper tier (the selected chip)
  const tierOk = !cat.view && ["legendaire", "exceptionnelle", "merveilleuse", "upscale"].includes(cat.tier);
  const del = $("#btn-cat-delete");
  if (del) {
    const n = cat.counts.rejected || 0;
    del.hidden = cat.view || cat.tier !== "rejected" || !n;
    del.textContent = `🗑 Delete all ${plural(n, "rejected scene")}`;
  }
  for (const [id, verb] of [["#btn-cat-board", "▶ Board"], ["#btn-cat-reel", "⬇ Reel"]]) {
    const b = $(id); if (!b) continue;
    b.disabled = !tierOk;
    b.textContent = tierOk ? `${verb}: ${TIER_NAMES[cat.tier]}` : verb;
  }
}
function renderCatTriage() {
  const m = side.model || cat.model || {}, rep = m.report;
  const trained = !!m.trained;
  const views = side.views || cat.views || {};
  $("#cat-views").innerHTML = CAT_VIEWS.map(([v, label, tip]) => {
    const needsModel = !MODEL_FREE_VIEWS.has(v) && !trained;
    return `<button class="cat-chip cat-view ${cat.view === v ? "on" : ""}" data-v="${v}" title="${esc(tip)}${needsModel ? " (train the model first)" : ""}" ${needsModel ? "disabled" : ""}>${esc(label)} <span class="n">${(views[v] || 0).toLocaleString()}</span></button>`;
  }).join("");
  const btn = $("#btn-cat-train");
  btn.textContent = m.training ? "Training…" : trained ? "Retrain" : "Train on my grades";
  btn.disabled = !!m.training;
  let txt = "";
  if (rep && rep.trained) {
    // plain words first ("usually right, almost always within a tier"), the
    // cross-validated figures small beside them
    const pct = (x) => Math.round(x * 100) + "%";
    const pts = (g) => `${g >= 0 ? "+" : ""}${Math.round(g * 100)} pts`;
    const qg = rep.quality_gain || 0;
    txt = `Tier guesses are <b>${W.accuracy(rep.cv.exact)}</b>${fig(pct(rep.cv.exact) + " exact")} and ${W.withinOne(rep.cv.within_one)}${fig(pct(rep.cv.within_one))}` +
      ` · file quality ${qg > 0 ? W.gain(qg) : "no clear help"}${fig(pts(qg))}` + whoGainText(rep) +
      ` · learned from ${rep.n.toLocaleString()} graded scenes (${esc(rep.trained_at)})` +
      (m.grades_since_train ? ` · ${m.grades_since_train} grades since` : "") +
      (rep.remembered_rejects ? ` · incl. ${plural(rep.remembered_rejects, "deleted reject")} remembered` : "");
    const short = Object.entries(rep.short || {}).map(([c, n]) => `${className(c)} (${n})`);
    if (short.length) txt += ` · too few to learn: ${esc(short.join(", "))}`;
  } else if (rep && !rep.trained) {
    txt = esc(rep.reason);
  } else {
    txt = "Not trained yet — learns your tiers from the scenes you've graded.";
  }
  $("#cat-model").innerHTML = txt;
}
// did performer/studio history earn its place in the model (measured, not assumed)?
function whoGainText(rep) {
  if (rep.who_gain == null) return "";
  const g = rep.who_gain, sg = `${g >= 0 ? "+" : ""}${Math.round(g * 100)} pts`;
  let t = rep.use_who ? ` · performer/studio history ${W.gain(g)}${fig(sg)}`
    : ` · performer/studio history not used yet — no clear help${fig(sg)}`;
  const fb = rep.fallback || {};
  if (fb.trained) t += ` · scenes not embedded yet are judged by who's in them (${W.accuracy(fb.cv.exact)}${fig(Math.round(fb.cv.exact * 100) + "% exact")})`;
  return t;
}
function renderCatStorage(st) {
  const el = $("#cat-storage"); if (!el || !st) return;
  const order = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "anomaly", "unreviewed", "rejected"];
  el.innerHTML = order.filter((t) => st[t] && st[t].bytes).map((t) =>
    `<span class="stor-chip"><span class="tier-dot tier-bg-${t}"></span>${esc(TIER_NAMES[t])} <b>${fmtBytes(st[t].bytes)}</b>${t === "rejected" ? " awaiting delete" : ""}</span>`).join("");
}
// saved views: named filter sets, one click to re-apply
function catCurrentParams() {
  const p = { sort: $("#cat-sort").value, q: $("#cat-q").value.trim(), res: $("#cat-res").value,
    min_mbps: $("#cat-mbps").value };
  if (cat.view) p.view = cat.view; else if (cat.tier) p.tier = cat.tier;
  if (cat.isNew) p.new = true;
  for (const [k, sel] of CAT_FILTERS) p[k] = ($(sel)?.value || "").trim();
  return p;
}
function applyCatParams(p) {
  setDupeMode(false);
  cat.view = p.view || ""; cat.tier = p.view ? "" : (p.tier || ""); cat.isNew = !!p.new;
  $("#cat-sort").value = p.sort || "date"; $("#cat-q").value = p.q || "";
  $("#cat-res").value = p.res || ""; $("#cat-mbps").value = p.min_mbps || "";
  for (const [k, sel] of CAT_FILTERS) if ($(sel)) $(sel).value = p[k] || "";
  if (CAT_FILTERS.some(([k]) => p[k])) $("#cat-filters").hidden = false;
  openCatalogue();
}
let catSaved = [];
async function loadSavedViews() {
  try { catSaved = (await api("/api/catalogue/saved-views")).items; } catch { catSaved = []; }
  renderSavedViews();
}
function renderSavedViews() {
  const box = $("#cat-saved"); if (!box) return;
  box.innerHTML = catSaved.map((v, i) => `<span class="cat-chip saved-view" data-i="${i}" title="${esc(JSON.stringify(v.params))}">★ ${esc(v.name)}<b class="sv-x" title="Delete this saved view">×</b></span>`).join("") +
    `<button id="btn-cat-saveview" class="cat-chip" title="Save the current tier, filters and sort as a named view">＋ Save view</button>`;
}
$("#cat-saved")?.addEventListener("click", async (e) => {
  if (e.target.closest("#btn-cat-saveview")) {
    const name = prompt("Name this view (tier, filters and sort are saved):");
    if (!name || !name.trim()) return;
    try {
      catSaved = (await api("/api/catalogue/saved-views", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, params: catCurrentParams() }) })).items;
      renderSavedViews(); toast(`Saved view “${name.trim()}”`);
    } catch (err) { toast(err.message, true); }
    return;
  }
  const chip = e.target.closest(".saved-view"); if (!chip) return;
  const v = catSaved[+chip.dataset.i];
  if (e.target.closest(".sv-x")) {
    if (!confirm(`Delete the saved view “${v.name}”?`)) return;
    catSaved = (await api("/api/catalogue/saved-views?name=" + encodeURIComponent(v.name), { method: "DELETE" })).items;
    renderSavedViews(); return;
  }
  applyCatParams(v.params);
});
let catFacetsLoaded = false;
async function loadCatFacets() {
  if (catFacetsLoaded) return;
  try {
    const f = await api("/api/catalogue/facets");
    const fill = (id, list) => { const dl = $(id); if (dl) dl.innerHTML = list.map(([n, c]) => `<option value="${esc(n)}">${c}</option>`).join(""); };
    fill("#dl-perf", f.performers); fill("#dl-studio", f.studios); fill("#dl-tag", f.tags);
    catFacetsLoaded = true;
  } catch { /* typeahead is optional */ }
}
$("#btn-cat-filters")?.addEventListener("click", () => {
  const box = $("#cat-filters"); box.hidden = !box.hidden;
  if (!box.hidden) loadCatFacets();
  $("#btn-cat-filters").textContent = $("#btn-cat-filters").textContent.replace(/[▾▴]$/, box.hidden ? "▾" : "▴");
});
$("#btn-cat-clear")?.addEventListener("click", () => {
  for (const [, sel] of CAT_FILTERS) if ($(sel)) $(sel).value = "";
  openCatalogue();
});
for (const [, sel] of CAT_FILTERS) $(sel)?.addEventListener("change", () => openCatalogue());
loadSavedViews();
function qualityLine(q) {
  return [q.res, q.mbps != null ? `${q.mbps} Mbps` : null, (q.codec || "").toUpperCase() || null,
          q.fps ? `${Math.round(q.fps)}fps` : null].filter(Boolean).join(" · ") || "quality unknown";
}
// quality as chips: the things you judge a file by, at a glance
function qualityChips(r) {
  const q = r.quality || {};
  const chip = (txt, cls = "") => txt ? `<span class="q ${cls}">${esc(txt)}</span>` : "";
  const hiRes = ["4K", "5K", "6K", "7K", "8K"].includes(q.res), lowRes = q.res === "SD" || q.res === "720p";
  return chip(q.res || "?", hiRes ? "hi" : lowRes ? "warn" : "") +
    chip(q.mbps != null ? `${q.mbps} Mbps` : "", r.flag ? "warn" : (q.mbps >= 30 ? "hi" : "")) +
    chip((q.codec || "").toUpperCase()) + chip(q.fps ? `${Math.round(q.fps)}fps` : "") +
    chip(r.size ? fmtBytes(r.size) : "");
}
// saved moments · the model's best moment · how long it's been in the library
function sigLine(r) {
  const s = r.signals; if (!s) return "";
  const bits = [];
  if (s.saves) bits.push(`<span class="sg-saves" title="Moments you saved in this scene">★ ${s.saves} saved</span>`);
  else bits.push(`<span class="faint" title="You haven't saved a moment in this scene">no saved moments</span>`);
  if (s.best != null) bits.push(`<span>best moment ${bestHTML(s)}</span>`);
  if (s.age_days != null) bits.push(ageHTML(s));
  if (s.passed) bits.push(passedHTML(s));
  if (r.size && cat.view === "trim") bits.push(`<span class="faint">${fmtBytes(r.size)}</span>`);
  return `<div class="cat-sig">${bits.join(" · ")}</div>`;
}
// --- plain words for the model's opinions (static/words.js) ---------------------
// the scene's best moment against every other scene's best: "Standout 97%"
function bestHTML(s) {
  if (s.best == null) return "";
  const hint = W.TASTE_HINT[s.band] ? ` — ${W.TASTE_HINT[s.band].replace(" of your library", "")} of your scenes` : "";
  return s.band_word ? `<b class="tw tw-${s.band}" title="The taste model's best moment here: ${W.pct(s.best)}${hint}">${s.band_word}</b>${fig(W.pct(s.best))}`
    : W.pct(s.best);
}
// shown a lot on the megaboard, never responded to
function passedHTML(s) {
  return `<span class="sg-passed" title="The megaboard keeps showing it and you never save, 👍, explore, pin or enlarge it">often passed over on the megaboard</span>${fig(`shown ${Math.round(s.showings)}× on ${s.shown_days} days`)}`;
}
function ageHTML(s) {
  const a = W.age(s.age_days); if (!a) return "";
  return s.new ? `<span class="sg-new" title="Inside the grace period — never suggested for trimming (${s.age_days} days)">new · ${a}</span>`
    : `<span class="faint" title="${s.age_days} days">${a}</span>`;
}
// "Probably Légendaire, maybe Exceptionnelle 56%"
function predHTML(p) {
  const ranked = Object.entries(p.probs || {}).sort((a, b) => b[1] - a[1]);
  const second = ranked[1];
  const tc = (c) => `<span class="tc-${c === "reject" ? "rejected" : c}">${esc(className(c))}</span>`;
  const conf = W.confidence(p.conf);
  const lead = conf === "Hard to call" ? `Hard to call — ${tc(p.tier)}` : `${conf} ${tc(p.tier)}`;
  const maybe = second && W.closeRunnerUp(p.conf, second[1]) ? `, maybe ${tc(second[0])}` : "";
  const title = ranked.map(([c, v]) => `${className(c)} ${W.pct(v)}`).join(" · ");
  return `<span title="${esc(title)}">${lead}${maybe}</span>${fig(W.pct(p.conf))}`;
}
function keeperHTML(p) {
  if (!p || p.keeper == null || p.tier === "reject") return "";
  const w = W.keeperWord(p.keeper); if (!w) return "";
  return `<span title="Chance it's not a reject: ${W.pct(p.keeper)}">${w}</span>${fig(W.pct(p.keeper))}`;
}
function textNum(text, detail) { return esc(text) + fig(detail); }
// performer / studio track records: how you've graded who's in this scene
function whoLines(r, max = 3) {
  const ls = ((r.who || {}).lines || []).slice(0, max);
  if (!ls.length) return "";
  return `<div class="cat-whorec">${ls.map((x) => `<span class="wr wr-${x.tone}" title="Your grades for ${x.kind === "studio" ? "this studio" : "this performer"} (Library health lists the best and worst)">${x.kind === "studio" ? "🏢" : "👤"} ${textNum(x.text, x.detail)}</span>`).join("")}</div>`;
}
function ageText(d) {
  if (d < 45) return `${d} days`;
  if (d < 540) return `${Math.round(d / 30)} months`;
  return `${(d / 365).toFixed(1)} years`;
}
// a line under the tier chips for the saves-driven views: what it is + its action
function renderCatViewbar() {
  const bar = $("#cat-viewbar"); if (!bar) return;
  const n = (cat.views || {})[cat.view] || 0;
  if (cat.view === "saved") {
    bar.hidden = false;
    bar.innerHTML = `<span>Scenes with saved moments that aren't Légendaire yet. From now on, saving a moment promotes its scene automatically (Settings → Tiers).</span>
      <button class="btn pri sm" id="btn-saved-all" ${n ? "" : "disabled"}>★ Grade all ${n.toLocaleString()} as ${esc(gradeName("legendaire"))}</button>`;
  } else if (cat.view === "trim") {
    const g = cat.grace || {};
    bar.hidden = false;
    bar.innerHTML = `<span>No saved moments — ones you keep passing over on the megaboard first, then weakest best moment. New scenes (under ${g.days ?? 30} days) are left out${g.new_excluded ? ` — ${g.new_excluded.toLocaleString()} right now` : ""}. Select the ones to let go and press 1 (Reject), then “Delete all rejected” when you're ready.</span>`;
  } else if (cat.view === "passed") {
    bar.hidden = false;
    bar.innerHTML = `<span>The megaboard keeps showing you these and you never save, 👍, explore, pin or enlarge them — most-shown first. Grading a scene starts its record over. Select the ones to let go and press 1 (Reject).</span>`;
  } else { bar.hidden = true; bar.innerHTML = ""; }
}
async function gradeAllSaved() {
  let d;
  try { d = await api("/api/catalogue?" + new URLSearchParams({ view: "saved", ids_only: "true", limit: 500 })); }
  catch (e) { return toast(e.message, true); }
  const ids = d.ids || [];
  if (!ids.length) return toast("Nothing to promote");
  if (!confirm(`Grade ${plural(ids.length, "scene")} with saved moments as ${gradeName("legendaire")}?\n\nEach gets the "${TIER_TAGS.legendaire || "legendaire"}" tag (other tier tags removed) and is marked organized — your renamer will move the files.\n\nZ undoes the whole batch.`)) return;
  const btn = $("#btn-saved-all"); if (btn) btn.disabled = true;
  try {
    const job = await api("/api/catalogue/grade-bulk", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_ids: ids, grade: "legendaire" }),
    });
    const j = await waitJob(job.id, (x) => {
      const p = x.progress || {};
      if (btn) btn.textContent = `Grading ${p.done ?? 0}/${p.total ?? ids.length}…`;
    });
    if (j.status === "error") throw new Error(j.error);
    const r = j.result;
    if (r.previous.length) gradeUndo.push({ bulk: true, items: r.previous, grade: "legendaire" });
    toast(`${plural(r.graded, "scene")} → ${gradeName("legendaire")}${r.failed.length ? ` · ${r.failed.length} failed` : ""} — Z to undo`, !!r.failed.length);
  } catch (e) { toast(e.message, true); }
  openCatalogue();
}
document.addEventListener("click", (e) => { if (e.target.closest("#btn-saved-all")) gradeAllSaved(); });

function catCardHTML(r, i) {
  const who = [r.performers.slice(0, 3).join(", "), r.studio, (r.date || "").slice(0, 4)].filter(Boolean).join(" · ");
  const moments = (r.moments || []).slice(0, 3).map((m, j) =>
    `<img class="cat-m" loading="lazy" src="${m.thumb}" data-j="${j}" title="${fmt(m.t)}${m.score != null ? " · taste " + Math.round(m.score * 100) + "%" : ""}" />`).join("");
  const cur = r.tier === "rejected" ? "reject" : r.tier;   // highlight the grade it already has
  const p = r.pred;
  const sug = r.suggest ? r.suggest.grade : (p && r.tier === "unreviewed" ? p.tier : null);
  const short = { legendaire: "Lég", exceptionnelle: "Exc", merveilleuse: "Mer", upscale: "Ups", reject: "Rej" };
  const grades = GRADES.map((g, k) =>
    `<button class="cat-g g-${g} ${g === cur ? "cur" : ""} ${g === sug ? "sug" : ""}" data-g="${g}" title="${esc(gradeName(g))} (key ${k + 1})"><b>${k + 1}</b><span>${short[g]}</span></button>`).join("");
  const predLine = p
    ? `<div class="cat-pred">${predHTML(p)}` + (keeperHTML(p) ? ` · ${keeperHTML(p)}` : "") +
      (p.from === "who" ? ` <span class="faint" title="Not embedded yet: this guess comes only from how you've graded these performers and this studio">(from performer/studio history — not seen yet)</span>` : "") + `</div>`
    : "";
  const flag = (r.flag ? `<div class="cat-flag">⚠ ${textNum(r.flag, r.flag_detail)}</div>` : "") +
    (r.dupe ? `<div class="cat-flag">⧉ Stash thinks this has a duplicate</div>` : "");
  const keepable = r.tier !== "unreviewed" && r.tier !== "anomaly"
    && (r.suggest || ["promote", "second", "saved", "trim", "passed"].includes(cat.view));
  const suggest = (r.suggest ? `<div class="cat-sug">Suggest <b>${esc(gradeName(r.suggest.grade))}</b> — ${textNum(r.suggest.why, r.suggest.detail)}</div>` : "")
    + (keepable && !r._graded ? `<button class="btn sm ghost cat-keep" data-keep="${esc(r.scene_id)}" title="It's right where it is — don't suggest it again unless something new happens">✓ Keep as is</button>` : "");
  const sel = cat.sel.has(r.scene_id);
  return `<div class="cat-card ${i === cat.focus ? "focus" : ""} ${r._graded ? "graded" : ""} ${sel ? "sel" : ""}" data-i="${i}">
    <label class="cat-selbox" title="Select (X) · shift-click selects a range"><input type="checkbox" class="cat-sel" ${sel ? "checked" : ""} /></label>
    <div class="cat-coverwrap"><img class="cat-cover" loading="lazy" src="/api/scene/${encodeURIComponent(r.scene_id)}/cover" onerror="this.style.visibility='hidden'" title="Watch" />
      ${r.duration ? `<span class="dur">${fmt(r.duration)}</span>` : ""}</div>
    <div class="cat-body">
      <div class="cat-title">${tierBadge(r.rating100, r.o_counter, { showUnreviewed: true })} <span title="${esc(r.path)}">${esc(r.title)}</span></div>
      <div class="cat-who">${esc(who)}</div>
      <div class="qual">${qualityChips(r)}</div>
      ${sigLine(r)}${predLine}${whoLines(r)}${flag}${suggest}${tagLine(r)}
    </div>
    <div class="cat-moments">${moments || '<span class="faint small">moments appear once embedded</span>'}</div>
    <div class="cat-grades">${grades}</div>
  </div>`;
}
// the renamer files a scene by its ONE tier tag — surface anything it can't act on
function tagLine(r) {
  const st = r.tag_state; if (!st) return "";
  const names = st.present.map((t) => (TIER_TAGS[t] || t)).join(" + ");
  if (st.conflict) return `<div class="cat-flag">🏷 Tag conflict: ${esc(names)} — ${esc(st.conflict)}. Re-grade to fix.</div>`;
  if (st.needs_sync && !r._graded) {
    const why = [st.present.length ? `tagged ${names}` : "no tier tag", r.organized ? "" : "not organized"].filter(Boolean).join(", ");
    return `<div class="cat-tag dim" title="Settings → Tier tags → Check &amp; sync fixes these in one go">🏷 Not filed for the renamer: ${esc(why)}</div>`;
  }
  return "";
}
function renderCatList() {
  const list = $("#cat-list");
  list.innerHTML = cat.items.length
    ? cat.items.map(catCardHTML).join("")
    : '<p class="dim">Nothing here with these filters.</p>';
  $("#btn-cat-more").hidden = cat.items.length >= cat.total;
}
function renderCatCard(i) {
  const old = $(`#cat-list .cat-card[data-i="${i}"]`);
  if (old) old.outerHTML = catCardHTML(cat.items[i], i);
}
function focusCat(i, scroll = true) {
  if (!cat.items.length) return;
  const prev = cat.focus;
  cat.focus = Math.max(0, Math.min(cat.items.length - 1, i));
  $(`#cat-list .cat-card[data-i="${prev}"]`)?.classList.remove("focus");
  const el = $(`#cat-list .cat-card[data-i="${cat.focus}"]`);
  el?.classList.add("focus");
  if (scroll) el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (cat.focus >= cat.items.length - 3 && cat.items.length < cat.total) openCatalogue({ append: true });
}
function catBumpCounts(from, to) {
  if (from === to) return;
  cat.counts[from] = Math.max(0, (cat.counts[from] || 0) - 1);
  cat.counts[to] = (cat.counts[to] || 0) + 1;
  renderCatChips();
}
async function gradeCat(i, grade) {
  const r = cat.items[i]; if (!r) return;
  try {
    const fresh = await applyGrade(r.scene_id, grade);
    catBumpCounts(r.tier, fresh.tier);
    // stays in place (dimmed) so the list doesn't jump and Z still has context
    cat.items[i] = { ...r, ...fresh, moments: r.moments, stream: r.stream, pred: r.pred, _graded: true };
    renderCatCard(i);
    focusCat(i + 1);
  } catch (e) { toast(e.message, true); }
}
async function keepCat(i) {
  const r = cat.items[i]; if (!r) return;
  try {
    const fresh = await keepAsIs(r.scene_id);
    cat.items[i] = { ...r, ...fresh, moments: r.moments, stream: r.stream, pred: r.pred, suggest: null, _graded: true };
    renderCatCard(i);
    focusCat(i + 1);
  } catch (e) { toast(e.message, true); }
}
async function undoCatGrade() {
  const sid = gradeUndo.length ? gradeUndo[gradeUndo.length - 1].sid : null;
  const fresh = await undoGrade();
  if (!fresh) return;
  if (fresh.bulk) { openCatalogue(); return; }
  const i = cat.items.findIndex((r) => r.scene_id === sid);
  if (i < 0) return;
  catBumpCounts(cat.items[i].tier, fresh.tier);
  cat.items[i] = { ...cat.items[i], ...fresh, _graded: false };
  renderCatCard(i); focusCat(i);
}
// watching a Catalogue scene opens the Review player on this same list (Likely
// keepers, Quality check, a tier…), at that scene — or at the moment clicked
function watchCat(i, j = null) {
  const r = cat.items[i]; if (!r) return;
  const m = j != null ? (r.moments || [])[j] : null;
  rv.startAt = m ? { sid: r.scene_id, t: m.t } : null;
  cat.focus = i;
  rv.fromCatalogue = true;
  go("review");
}
$("#cat-chips")?.addEventListener("click", (e) => {
  const b = e.target.closest(".cat-chip"); if (!b) return;
  setDupeMode(false);
  cat.isNew = false;
  cat.tier = b.dataset.t; cat.view = ""; openCatalogue();
});
$("#cat-views")?.addEventListener("click", (e) => {
  const b = e.target.closest(".cat-view"); if (!b || b.disabled) return;
  showView("catalogue");
  cat.isNew = false;
  cat.view = cat.view === b.dataset.v ? "" : b.dataset.v;
  openCatalogue();
});
$("#btn-cat-board")?.addEventListener("click", () => {
  if (!cat.tier) return;
  window.open("/megaboard/?src=" + encodeURIComponent("tier:" + cat.tier), "_blank");
});
$("#btn-cat-reel")?.addEventListener("click", async () => {
  if (!cat.tier) return;
  const name = TIER_NAMES[cat.tier], count = 300;
  if (!confirm(`Export the best ${count} moments across your ${name} scenes as one video?\n\n` +
    `Moments are picked for variety across scenes, each clip held until the picture changes, ` +
    `and re-encoded to your Export-quality setting — it can take a while. ` +
    `The file appears under Megaboard → Exported videos.`)) return;
  try {
    const j = await api(`/api/catalogue/reel?tiers=${encodeURIComponent(cat.tier)}&count=${count}`, { method: "POST" });
    toast(`Building the ${name} reel… (Megaboard → Exported videos when ready)`);
    if (j && j.id) pollPerformerReel(j.id, name);
  } catch (e) { toast(e.message, true); }
});
$("#btn-cat-train")?.addEventListener("click", async () => {
  const btn = $("#btn-cat-train");
  btn.disabled = true; btn.textContent = "Training…";
  $("#cat-model").textContent = "learning your tiers from every graded scene — this can take a minute on a big library…";
  try {
    const r = await api("/api/catalogue/train", { method: "POST" });
    toast(r.trained ? `Trained on ${r.n} scenes — tier guesses are ${W.accuracy(r.cv.exact)} (${Math.round(r.cv.exact * 100)}% exact)` : r.reason, !r.trained);
  } catch (err) { toast(err.message, true); }
  openCatalogue();
});
$("#cat-list")?.addEventListener("click", (e) => {
  const card = e.target.closest(".cat-card"); if (!card) return;
  const i = +card.dataset.i;
  if (e.target.closest(".cat-selbox")) {
    e.preventDefault();
    toggleCatSel(i, e.shiftKey);
    return;
  }
  const g = e.target.closest(".cat-g");
  if (g) { focusCat(i, false); gradeCat(i, g.dataset.g); return; }
  if (e.target.closest(".cat-keep")) { focusCat(i, false); keepCat(i); return; }
  const m = e.target.closest(".cat-m");
  if (m) { focusCat(i, false); watchCat(i, +m.dataset.j); return; }
  if (e.target.closest(".cat-cover")) { focusCat(i, false); watchCat(i); return; }
  focusCat(i, false);
});
let catQT;
$("#cat-q")?.addEventListener("input", () => { clearTimeout(catQT); catQT = setTimeout(() => openCatalogue(), 300); });
for (const sel of ["#cat-res", "#cat-sort", "#cat-mbps"]) $(sel)?.addEventListener("change", () => openCatalogue());
$("#btn-cat-refresh")?.addEventListener("click", () => openCatalogue({ refresh: true }));
$("#btn-cat-more")?.addEventListener("click", () => openCatalogue({ append: true }));
document.addEventListener("keydown", (e) => {
  if (!$("#catalogue")?.classList.contains("active") || !$("#viewer").hidden) return;
  if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k >= "1" && k <= "5") { e.preventDefault(); gradeCat(cat.focus, GRADES[+k - 1]); }
  else if (k === "j" || k === "arrowdown") { e.preventDefault(); focusCat(cat.focus + 1); }
  else if (k === "k" || k === "arrowup") { e.preventDefault(); focusCat(cat.focus - 1); }
  else if (k === "enter") { e.preventDefault(); watchCat(cat.focus); }
  else if (k === "z") { e.preventDefault(); undoCatGrade(); }
  else if (k === "x") { e.preventDefault(); toggleCatSel(cat.focus, e.shiftKey); }
  else if (k === "escape" && cat.sel.size) { e.preventDefault(); cat.sel.clear(); renderCatList(); renderCatBulk(); }
});

// --- Catalogue: select several scenes and grade them together ------------------
function toggleCatSel(i, range) {
  const r = cat.items[i]; if (!r) return;
  const on = !cat.sel.has(r.scene_id);
  const [a, b] = range && cat.anchor != null ? [Math.min(cat.anchor, i), Math.max(cat.anchor, i)] : [i, i];
  for (let k = a; k <= b; k++) {
    const id = cat.items[k].scene_id;
    if (on) cat.sel.add(id); else cat.sel.delete(id);
    $(`#cat-list .cat-card[data-i="${k}"]`)?.classList.toggle("sel", on);
    const box = $(`#cat-list .cat-card[data-i="${k}"] .cat-sel`); if (box) box.checked = on;
  }
  cat.anchor = i;
  renderCatBulk();
}
function renderCatBulk() {
  const bar = $("#cat-bulk"); if (!bar) return;
  const n = cat.sel.size;
  bar.hidden = n === 0 && !cat.bulkBusy;
  if (cat.bulkBusy) return;
  $("#cat-bulk-n").textContent = plural(n, "scene") + " selected";
  const allRejected = n > 0 && cat.items.filter((r) => cat.sel.has(r.scene_id)).every((r) => r.tier === "rejected");
  const ds = $("#btn-cat-delsel"); if (ds) ds.hidden = !allRejected;
  $("#cat-bulk-grades").innerHTML = GRADES.map((g) =>
    `<button class="cat-g g-${g}" data-g="${g}">${esc(gradeName(g))}</button>`).join("");
}
async function gradeSelected(grade) {
  const ids = cat.items.filter((r) => cat.sel.has(r.scene_id)).map((r) => r.scene_id);
  if (!ids.length || cat.bulkBusy) return;
  const tagged = grade !== "reject";
  const note = tagged
    ? `\n\nEach one gets the "${TIER_TAGS[grade] || grade}" tag (other tier tags removed) and is marked organized — your renamer will move ${ids.length === 1 ? "the file" : "these files"}.`
    : "\n\nOnly the rating changes (1★); tags and organized are left alone.";
  if (!confirm(`Grade ${plural(ids.length, "scene")} as ${gradeName(grade)}?${note}\n\nOne Z undoes the whole batch.`)) return;
  cat.bulkBusy = true;
  $("#cat-bulk-grades").innerHTML = "";
  try {
    const job = await api("/api/catalogue/grade-bulk", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_ids: ids, grade }),
    });
    const j = await waitJob(job.id, (x) => {
      const p = x.progress || {};
      $("#cat-bulk-n").textContent = `Grading ${p.done ?? 0}/${p.total ?? ids.length} as ${gradeName(grade)}…`;
    });
    if (j.status === "error") throw new Error(j.error);
    const r = j.result;
    if (r.previous.length) gradeUndo.push({ bulk: true, items: r.previous, grade });
    toast(`${plural(r.graded, "scene")} → ${gradeName(grade)}${r.failed.length ? ` · ${r.failed.length} failed` : ""} — Z to undo`, !!r.failed.length);
  } catch (e) { toast(e.message, true); }
  cat.bulkBusy = false;
  cat.sel.clear();
  openCatalogue();
}
$("#cat-bulk")?.addEventListener("click", (e) => {
  const g = e.target.closest(".cat-g");
  if (g) { gradeSelected(g.dataset.g); return; }
  if (e.target.closest("#btn-cat-selall")) {
    cat.items.forEach((r) => cat.sel.add(r.scene_id));
    renderCatList(); renderCatBulk();
  } else if (e.target.closest("#btn-cat-selnone")) {
    cat.sel.clear(); renderCatList(); renderCatBulk();
  }
});
// --- deleting rejects (files included) — preview, tick, then delete ---------------
// One confirm dialog for every deletion: lists the files; "Delete the video
// files too" starts ticked (like Stash) — unticked removes the scenes from Stash
// but leaves the files on disk. `summary` / `goLabel` may be functions of it.
function showDeleteDialog({ title, summary, note, items, total, goLabel, capable = true, run }) {
  const dlg = $("#del-dlg");
  const val = (x, df) => (typeof x === "function" ? x(df) : x);
  $("#del-title").textContent = title;
  $("#del-note").textContent = note || "";
  $("#del-list").innerHTML = items.map((r) => `<div class="hist-row"><span class="hist-what" title="${esc(r.path || "")}">${esc(r.title || r.path)}</span>
    <span class="dim">${r.size ? fmtBytes(r.size) : ""}</span></div>`).join("") +
    (total > items.length ? `<div class="dim">…and ${total - items.length} more</div>` : "");
  const ack = $("#del-ack"), go = $("#btn-del-go"), cancel = $("#btn-del-cancel");
  ack.checked = true; cancel.disabled = false;
  $("#del-status").textContent = capable ? "" : "Your Stash version can't delete scenes from the API — update Stash to use this.";
  ack.disabled = !capable; go.disabled = !capable;
  const sync = () => {
    const df = ack.checked;
    $("#del-summary").innerHTML = val(summary, df);
    go.textContent = val(goLabel, df);
    go.classList.toggle("keep-files", !df);
    $("#del-keep-note").hidden = df;
  };
  sync();
  ack.onchange = sync;
  cancel.onclick = () => dlg.close();
  go.onclick = async () => {
    const deleteFile = ack.checked;
    go.disabled = true; ack.disabled = true; cancel.disabled = true;
    try {
      await run((msg) => { $("#del-status").textContent = msg; }, deleteFile);
      dlg.close();
    } catch (e) { $("#del-status").textContent = e.message; toast(e.message, true); }
    cancel.disabled = false;
  };
  dlg.showModal();
}
async function openDeleteDialog(ids) {
  let d;
  try {
    d = await api("/api/catalogue/delete-preview" + (ids ? "?ids=" + encodeURIComponent(ids.join(",")) : ""));
  } catch (e) { toast(e.message, true); return; }
  if (!d.count) { toast("Nothing rated 1★ to delete"); return; }
  showDeleteDialog({
    title: ids ? "Delete selected rejects" : "Delete all rejected scenes",
    summary: (df) => df
      ? `<b>${plural(d.count, "scene")}</b> · <b>${fmtBytes(d.bytes)}</b> will be deleted from Stash, with their files.`
      : `<b>${plural(d.count, "scene")}</b> will be removed from Stash — the files (${fmtBytes(d.bytes)}) stay on disk.`,
    note: "Each scene is re-checked right before deletion — anything no longer rated 1★ is skipped. " +
      "Every deleted file is recorded in Settings → History. Peaks remembers what the rejects looked like, " +
      "so keeper triage keeps learning from them.",
    items: d.items, total: d.count, capable: d.capable,
    goLabel: (df) => df ? `Delete ${plural(d.count, "file")}` : `Remove ${plural(d.count, "scene")} from Stash`,
    run: async (status, deleteFile) => {
      const job = await api("/api/catalogue/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene_ids: d.ids, confirm: true, delete_file: deleteFile }),   // exactly what was previewed
      });
      const j = await waitJob(job.id, (x) => {
        const p = x.progress || {};
        status(`deleting ${p.done ?? 0}/${p.total ?? d.count}…`);
      });
      if (j.status === "error") throw new Error(j.error);
      const r = j.result;
      const extra = [r.refused.length ? `${r.refused.length} skipped (no longer 1★)` : "",
        r.failed.length ? `${r.failed.length} failed` : ""].filter(Boolean).join(" · ");
      toast((r.files_deleted === false
        ? `Removed ${plural(r.deleted, "scene")} from Stash · files kept`
        : `Deleted ${plural(r.deleted, "scene")} · freed ${fmtBytes(r.freed_bytes)}`) + (extra ? " · " + extra : ""), !!r.failed.length);
      cat.sel.clear(); openCatalogue(); loadHistory();
    },
  });
}
// --- duplicates: Stash's phash groups, judged on file quality -----------------
const dupe = { on: false, data: null };
// duplicates live on their own page now; kept for callers that toggle the old mode
function setDupeMode(on) {
  dupe.on = on;
  if (on) go("dupes");
}
function dupeCopyHTML(r, g) {
  const q = r.quality || {};
  const rec = r.scene_id === g.keep;
  const added = (r.created_at || "").slice(0, 10);
  return `<div class="dupe-copy ${rec ? "rec" : ""}" data-sid="${esc(r.scene_id)}">
    <img class="cat-cover" loading="lazy" src="/api/scene/${encodeURIComponent(r.scene_id)}/cover" onerror="this.style.visibility='hidden'" />
    <div class="dupe-facts">
      <div>${rec ? '<span class="dupe-rec">★ Recommended</span> ' : ""}${tierBadge(r.rating100, r.o_counter, { showUnreviewed: true })}</div>
      <div class="dupe-q"><b>${esc(q.res || "?")}</b>${q.w ? ` <span class="muted">${q.w}×${q.h}</span>` : ""} · <b>${q.mbps != null ? q.mbps + " Mbps" : "? Mbps"}</b> · ${esc((q.codec || "").toUpperCase())} ${q.fps ? Math.round(q.fps) + "fps" : ""}</div>
      <div class="dim">${r.size ? fmtBytes(r.size) : "size ?"} · ${r.duration ? fmt(r.duration) : "?"}${added ? " · added " + esc(added) : ""}</div>
      <div class="dim dupe-path" title="${esc(r.path)}">${esc(r.path)}</div>
    </div>
    <button class="${rec ? "primary" : "ghost"} dupe-keep">Keep this, delete ${g.scenes.length === 2 ? "the other" : "the others"}</button>
  </div>`;
}
function renderDupes() {
  const d = dupe.data, box = $("#dupe-list");
  if (!d || !d.groups) { box.innerHTML = ""; return; }
  setDupeCount(d.groups.length);
  $("#dupe-status").textContent = `${plural(d.groups.length, "group")} · ${fmtBytes(d.reclaim)} reclaimable` +
    (d.ignored ? ` · ${d.ignored} marked not duplicates` : "") + ` · checked ${d.checked_at}`;
  box.innerHTML = d.groups.length ? d.groups.map((g, gi) => `<div class="dupe-group" data-g="${gi}">
      <div class="dupe-head"><b>${esc(g.scenes[0].title)}</b>
        <span class="dim">${g.scenes.length} copies · ${fmtBytes(g.reclaim)} to reclaim${g.best_grade ? " · graded " + esc(TIER_NAMES[g.best_grade]) : ""}</span>
        <span class="grow"></span><button class="ghost dupe-ignore">Not duplicates</button></div>
      <div class="dupe-copies">${g.scenes.map((r) => dupeCopyHTML(r, g)).join("")}</div></div>`).join("")
    : '<p class="dim">No duplicates at this accuracy. 🎉</p>';
}
async function openDupes() {
  try { const d = await api("/api/duplicates"); if (d.groups) { dupe.data = d; renderDupes(); } } catch {}
  pollDupeQueue();
  if (!dupe.data) $("#dupe-list").innerHTML = '<div class="empty">Pick an accuracy and <b>Find duplicates</b> — Stash compares phashes across the library.</div>';
}
$("#btn-cat-new")?.addEventListener("click", () => {
  showView("catalogue");
  cat.isNew = !cat.isNew;
  if (cat.isNew) { cat.tier = ""; cat.view = ""; }
  openCatalogue();
});
// POST that may come back 409 needs_confirm: ask, then retry with confirm=true
async function postConfirmed(url) {
  const r = await fetch(url, { method: "POST" });
  if (r.status === 409) {
    let d = null;
    try { d = (await r.clone().json()).detail; } catch { /* not json */ }
    if (d && d.needs_confirm) {
      if (!confirm(d.message)) throw new Error("Cancelled");
      return api(url + (url.includes("?") ? "&" : "?") + "confirm=true", { method: "POST" });
    }
  }
  if (r.status === 401) { location.reload(); throw new Error("session expired"); }
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}
async function startIngest() {
  try {
    const job = await postConfirmed("/api/ingest");
    tracked.add(job.id);
    const btn = $("#btn-ingest"), statusEl = $("#ingest-status"), logEl = $("#ingest-log"), stop = $("#btn-ingest-stop");
    if (btn) { btn.disabled = true; logEl.hidden = false; wireStop(stop, statusEl, job.id); }
    const cb = $("#btn-cat-ingest"); if (cb) cb.disabled = true;
    const j = await waitJob(job.id, (x) => {
      renderIngestSteps(x);
      const m = (x.log || []).join("\n").match(/scan done: (\d+) new/);
      const rb = $("#btn-ingest-review");
      if (rb && m && +m[1] > 0) { rb.hidden = false; rb.textContent = `▶ Review ${plural(+m[1], "new scene")} now`; }
      const p = x.progress || {}, line = (x.log || []).slice(-1)[0] || "starting…";
      $("#cat-status").textContent = "Ingest: " + line;
      if (statusEl) statusEl.textContent = `${x.status}${p.stage && p.stage !== "done" ? " · " + p.stage : ""} · ${x.elapsed}s`;
      if (logEl) { logEl.textContent = (x.log || []).join("\n"); logEl.scrollTop = logEl.scrollHeight; }
    }, 1500);
    if (btn) { btn.disabled = false; if (stop) stop.hidden = true; }
    if (cb) cb.disabled = false;
    renderIngestSteps(j);
    if (j.status === "error") { toast("Ingest failed: " + j.error, true); $("#cat-status").textContent = ""; return; }
    if (j.status === "cancelled") { toast("Ingest stopped."); return; }
    const r = j.result;
    toast(`Ingest done: ${plural(r.new, "new scene")}${r.duplicates ? ` · ${plural(r.duplicates, "duplicate group")}` : ""}`);
    loadHistory();
    if (r.new) { setDupeMode(false); cat.isNew = true; cat.tier = ""; cat.view = ""; }
    if (cat.loaded || r.new) openCatalogue({ refresh: true });
  } catch (e) { toast(e.message, true); }
}
// the five ingest steps light up from the job's log ("2/5 identify: …", "scan done: …")
const INGEST_STEPS = ["scan", "identify", "auto tag", "embed", "duplicates"];
function renderIngestSteps(job) {
  const box = $("#ingest-steps"); if (!box) return;
  const log = (job.log || []).join("\n");
  let cur = -1;
  INGEST_STEPS.forEach((st, i) => { if (log.includes(`${i + 1}/5 ${st}`)) cur = i; });
  const finished = job.status !== "running";
  box.querySelectorAll(".st").forEach((el, i) => {
    const note = (job.result && job.result.stages || {})[INGEST_STEPS[i]];
    el.classList.toggle("done", i < cur || (finished && job.status === "done" && (i <= cur || !!note)));
    el.classList.toggle("run", !finished && i === cur);
    el.classList.toggle("fail", finished && job.status === "error" && i === cur);
    if (note) el.querySelector("span").textContent = note;
  });
}
$("#btn-ingest")?.addEventListener("click", startIngest);
$("#btn-ingest-review")?.addEventListener("click", () => { rv.forceIngest = true; go("review"); });
$("#btn-top-ingest")?.addEventListener("click", () => {
  if (confirm("Ingest new files?\n\nStash scans the library (phashes on), identifies and auto-tags the new scenes with your saved task defaults, then Peaks embeds them and checks for duplicates.")) {
    go("activity"); startIngest();
  }
});
$("#btn-cat-ingest")?.addEventListener("click", () => {
  if (confirm("Ingest new files?\n\nStash scans the library (phashes on), identifies and auto-tags the new scenes with your saved task defaults, then Peaks embeds them and checks for duplicates.")) startIngest();
});
$("#btn-cat-dupes")?.addEventListener("click", () => go("dupes"));
$("#btn-dupe-scan")?.addEventListener("click", async () => {
  const btn = $("#btn-dupe-scan"); btn.disabled = true;
  try {
    const qs = new URLSearchParams({ accuracy: $("#dupe-acc").value, duration_diff: $("#dupe-dur").value });
    const job = await api("/api/duplicates/scan?" + qs, { method: "POST" });
    $("#dupe-status").textContent = "Stash is comparing phashes — this can take a while on a big library…";
    const j = await waitJob(job.id, null, 1500);
    if (j.status === "error") throw new Error(j.error);
    dupe.data = j.result; renderDupes();
  } catch (e) { $("#dupe-status").textContent = ""; toast(e.message, true); }
  btn.disabled = false;
});
$("#dupe-list")?.addEventListener("click", async (e) => {
  const gEl = e.target.closest(".dupe-group"); if (!gEl) return;
  const g = dupe.data.groups[+gEl.dataset.g];
  if (e.target.closest(".dupe-ignore")) {
    try {
      await api("/api/duplicates/ignore", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene_ids: g.scenes.map((r) => r.scene_id) }) });
      dupe.data.groups = dupe.data.groups.filter((x) => x !== g); dupe.data.ignored = (dupe.data.ignored || 0) + 1;
      renderDupes(); toast("Marked as not duplicates — this group won't show again");
    } catch (err) { toast(err.message, true); }
    return;
  }
  const copy = e.target.closest(".dupe-copy");
  if (e.target.closest(".dupe-keep") && copy) {
    // no dialog, no waiting: the decision is queued and the group leaves the list
    const keep = copy.dataset.sid;
    const others = g.scenes.filter((r) => r.scene_id !== keep).map((r) => r.scene_id);
    const df = dupeDeleteFiles();
    dupe.data.groups = dupe.data.groups.filter((x) => x !== g);
    dupe.data.reclaim -= g.reclaim;
    renderDupes();
    try {
      const r = await api("/api/duplicates/queue", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keep, delete: others, delete_file: df }) });
      renderDupeQueue(r);
      toast(`Queued: keeping 1, ${df ? "deleting" : "removing"} ${plural(others.length, "copy", "copies")} — Undo in the bar for 5 s`);
    } catch (err) {
      toast(err.message, true);
      dupe.data.groups.push(g); dupe.data.groups.sort((x, y) => y.reclaim - x.reclaim);
      dupe.data.reclaim += g.reclaim; renderDupes();
    }
  }
});
// --- the delete queue: decisions go in, a background worker works through them ---
function dupeDeleteFiles() {
  const el = $("#dupe-files"); return el ? el.checked : true;
}
(() => {
  const el = $("#dupe-files"); if (!el) return;
  try { el.checked = localStorage.getItem("peaks_dupe_files") !== "0"; } catch {}
  el.addEventListener("change", () => { try { localStorage.setItem("peaks_dupe_files", el.checked ? "1" : "0"); } catch {} });
})();
let dqTimer = null, dqSeen = 0;
function renderDupeQueue(q) {
  const bar = $("#dupe-queue"); if (!bar || !q) return;
  const busy = q.queued.length + q.running.length;
  const doneNow = q.done.filter((x) => x.finished_at > dqSeen);
  if (!busy && !q.failed.length && !doneNow.length) { bar.hidden = true; clearInterval(dqTimer); dqTimer = null; return; }
  bar.hidden = false;
  const total = busy + doneNow.length;
  const head = busy
    ? (q.paused ? `⏸ Paused · ${plural(q.queued.length, "group")} waiting`
       : !q.running.length ? `🕐 ${plural(q.queued.length, "group")} queued — starting in a moment`
       : `🗑 Deleting ${doneNow.length + 1} of ${total}${q.queued.length ? ` · ${q.queued.length} more queued` : ""}`)
    : `✓ Done · ${plural(doneNow.length, "group")} resolved`;
  const freed = doneNow.reduce((a, x) => a + (+(x.result || {}).freed_bytes || 0), 0);
  const now = Date.now() / 1000;
  const undoable = q.queued.filter((x) => x.not_before > now);
  bar.innerHTML = `<div class="dq-head"><b>${head}</b>${freed ? ` · ${fmtBytes(freed)} freed` : ""}
      ${q.queued_bytes ? ` · ${fmtBytes(q.queued_bytes)} to go` : ""}<span class="grow"></span>
      ${busy ? (q.paused ? `<button class="btn sm" data-dq="resume">Resume</button>` : `<button class="btn sm ghost" data-dq="stop" title="Finish the current group, then pause">Stop</button>`) : ""}
      ${q.queued.length ? `<button class="btn sm ghost" data-dq="clear" title="Drop what hasn't started — those groups come back">Clear queued</button>` : ""}
      ${!busy ? `<button class="btn sm ghost" data-dq="hide">Hide</button>` : ""}</div>
    ${q.running.map((x) => `<div class="dq-row">⏳ ${esc(x.title)}</div>`).join("")}
    ${undoable.map((x) => `<div class="dq-row">🕐 ${esc(x.title)} <button class="btn sm" data-dq-undo="${esc(x.id)}">Undo</button></div>`).join("")}
    ${q.failed.map((x) => `<div class="dq-row bad">⚠ ${esc(x.title)} — ${esc(x.error || "failed")}
      <button class="btn sm" data-dq-retry="${esc(x.id)}">Retry</button><button class="btn sm ghost" data-dq-dismiss="${esc(x.id)}">Dismiss</button></div>`).join("")}`;
  if ((busy || undoable.length) && !dqTimer) dqTimer = setInterval(pollDupeQueue, 2000);
}
async function pollDupeQueue() {
  try {
    const q = await api("/api/duplicates/queue");
    renderDupeQueue(q);
    if (!q.queued.length && !q.running.length) { clearInterval(dqTimer); dqTimer = null; loadHistory(); }
  } catch {}
}
$("#dupe-queue")?.addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  try {
    let q;
    if (b.dataset.dq === "hide") { dqSeen = Date.now() / 1000; $("#dupe-queue").hidden = true; return; }
    if (b.dataset.dq) q = await api("/api/duplicates/queue/" + b.dataset.dq, { method: "POST" });
    for (const k of ["undo", "retry", "dismiss"]) {
      const id = b.dataset["dq" + k[0].toUpperCase() + k.slice(1)];
      if (id) q = await api(`/api/duplicates/queue/${encodeURIComponent(id)}/${k}`, { method: "POST" });
    }
    if (b.dataset.dqUndo || b.dataset.dq === "clear") {      // those groups come back
      const d = await api("/api/duplicates"); if (d.groups) { dupe.data = d; renderDupes(); }
    }
    renderDupeQueue(q || await api("/api/duplicates/queue"));
  } catch (err) { toast(err.message, true); }
});
$("#btn-dupe-all")?.addEventListener("click", async () => {
  const gs = dupe.data?.groups || [];
  if (!gs.length) return;
  const df = dupeDeleteFiles();
  const bytes = gs.reduce((a, g) => a + g.reclaim, 0);
  if (!confirm(`Keep the recommended copy in all ${gs.length} groups and ${df
    ? `delete the other ${plural(gs.reduce((a, g) => a + g.scenes.length - 1, 0), "copy", "copies")} with their files (${fmtBytes(bytes)})`
    : "remove the other copies from Stash (files stay on disk)"}?\n\nEach kept copy takes its group's highest tier first. Deletions run in the background — you can Stop or Clear the queue any time.`)) return;
  try {
    const r = await api("/api/duplicates/queue/recommended?delete_file=" + df, { method: "POST" });
    const d = await api("/api/duplicates"); if (d.groups) { dupe.data = d; renderDupes(); }
    renderDupeQueue(r);
    toast(`Queued ${plural(r.added, "group")}${r.skipped ? ` · ${r.skipped} skipped (already queued)` : ""}`);
  } catch (err) { toast(err.message, true); }
});
$("#btn-cat-delete")?.addEventListener("click", () => openDeleteDialog(null));
$("#btn-cat-delsel")?.addEventListener("click", () =>
  openDeleteDialog(cat.items.filter((r) => cat.sel.has(r.scene_id)).map((r) => r.scene_id)));
$("#btn-cat-more-menu")?.addEventListener("click", (e) => { e.stopPropagation(); $("#cat-menu").hidden = !$("#cat-menu").hidden; });
document.addEventListener("click", (e) => { if (!e.target.closest(".menu-wrap")) { const m = $("#cat-menu"); if (m) m.hidden = true; } });
$("#btn-cat-review")?.addEventListener("click", () => { rv.fromCatalogue = true; go("review"); });
$("#btn-cat-select")?.addEventListener("click", () => {
  cat.items.forEach((r) => cat.sel.add(r.scene_id));
  renderCatList(); renderCatBulk();
});
// Z also undoes a grade made from the viewer's grade menu
document.addEventListener("keydown", (e) => {
  if ($("#viewer").hidden || e.key.toLowerCase() !== "z" || e.target.closest("input, select, textarea")) return;
  e.preventDefault();
  undoGrade().then((s) => { if (s && currentHit) loadViewerMeta(currentHit.scene_id); });
});

function setActiveView(name) {
  showView(name);
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


// --- live job tray (sidebar): whatever is running, from any page -------------------
const JOB_LABEL = { embed: "Embedding", ingest: "Ingest", score: "Writing markers", sync: "Syncing",
  fix: "Retrying failed", reel: "Exporting video", playlist: "Building board", library: "Updating Stash",
  dupes: "Finding duplicates", train: "Training taste", "taste-measure": "Measuring taste",
  warmup: "Starting up", backup: "Backing up", restore: "Restoring", "perf-photos": "Performer photos" };
async function pollJobTray() {
  const tray = $("#job-tray");
  if (!tray || document.hidden) return;
  let jobs;
  try { jobs = await api("/api/jobs"); } catch { return; }
  const running = jobs.filter((j) => j.status === "running");
  const kinds = new Set(running.map((j) => j.kind));
  const LIB = ["library", "ingest", "dupes", "train", "taste-measure", "embed", "fix", "sync", "warmup"];
  if ((pollJobTray.prev || []).some((k) => LIB.includes(k) && !kinds.has(k)) || kinds.has("ingest")) refreshSidebar();
  if ((pollJobTray.prev || []).includes("sync") && !kinds.has("sync") && $("#activity")?.classList.contains("active")) {
    loadCleanup(); loadCopies(); loadRenamerMoves();     // a Sync just finished: show what it tidied
  }
  pollJobTray.prev = [...kinds];
  const badge = $("#nav-ct-jobs");
  if (badge) { badge.hidden = !running.length; badge.textContent = running.length || ""; }
  tray.hidden = !running.length;
  tray.innerHTML = running.slice(0, 3).map((j) => {
    const p = j.progress || {};
    const pct = p.total ? Math.round(100 * (p.done || 0) / p.total) : (typeof p.pct === "number" ? Math.round(100 * p.pct) : null);
    const detail = p.total ? `${(p.done || 0).toLocaleString()} / ${p.total.toLocaleString()}` : (p.stage || `${Math.round(j.elapsed)}s`);
    return `<div class="tray-job"><div class="row between"><b>${esc(JOB_LABEL[j.kind] || j.kind)}</b><span class="muted">${pct != null ? pct + "%" : ""}</span></div>
      <div class="muted small">${esc(detail)}</div><div class="bar"><i style="width:${pct ?? 100}%" class="${pct == null ? "indet" : ""}"></i></div></div>`;
  }).join("");
}
$("#job-tray")?.addEventListener("click", () => go("activity"));
pollJobTray();
setInterval(pollJobTray, 4000);

// --- ⌘K command palette: jump anywhere, run anything, or search ---------------------
const CMDS = [
  ["For You", "page", () => go("foryou")], ["Megaboard", "page", () => go("board")],
  ["Search moments", "page", () => go("explore")], ["Performers", "page", () => go("performers")],
  ["Catalogue", "page", () => go("catalogue")], ["Review queue", "page", () => go("review")],
  ["Duplicates", "page", () => go("dupes")], ["Insights", "page", () => go("statistics")],
  ["Taste — teach", "page", () => go("taste")], ["Activity", "page", () => go("activity")],
  ["Settings", "page", () => go("dashboard")],
  ["Ingest new files", "run", () => { go("activity"); startIngest(); }],
  ["Embed new scenes", "run", () => { go("activity"); $("#btn-embed").click(); }],
  ["Sync with Stash", "run", () => { go("activity"); $("#btn-sync").click(); }],
  ["Find duplicates", "run", () => { go("dupes"); $("#btn-dupe-scan").click(); }],
  ["Train tier model on my grades", "run", () => { go("catalogue"); $("#btn-cat-train").click(); }],
  ["Rebuild For You", "run", () => { go("foryou"); $("#btn-foryou-rebuild").click(); }],
  ["Taste Radio", "run", () => startRadio()],
  ["Check tier tags…", "run", () => { go("dashboard"); showSettingsSection("tiers"); $("#btn-tag-sync").click(); }],
  ["Back up grades now", "run", () => { go("activity"); $("#btn-backup-now").click(); }],
  ["Delete rejected scenes…", "run", () => openDeleteDialog(null)],
  ...["legendaire", "exceptionnelle", "merveilleuse", "upscale", "unreviewed", "anomaly", "rejected"].map((t) =>
    [`Catalogue: ${t}`, "tier", () => { cat.tier = t; cat.view = ""; cat.isNew = false; go("catalogue"); openCatalogue(); }]),
];
const cmdk = { items: [], idx: 0 };
function openCmdk() {
  $("#cmdk").hidden = false;
  const inp = $("#cmdk-in"); inp.value = ""; renderCmdk(); inp.focus();
}
function closeCmdk() { $("#cmdk").hidden = true; }
const fold = (x) => x.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();   // é → e
function renderCmdk() {
  const q = $("#cmdk-in").value.trim(), ql = fold(q);
  const label = (c) => c[0].replace(/^Catalogue: (\w+)/, (_, t) => "Catalogue: " + (TIER_NAMES[t] || t));
  let items = CMDS.filter((c) => !ql || fold(label(c)).includes(ql))
    .map((c) => ({ text: label(c), kind: c[1], run: c[2] }));
  if (q) {
    items.push({ text: `Search moments for “${q}”`, kind: "search", run: () => { go("explore"); $("#q").value = q; $("#btn-text").click(); } });
    items.push({ text: `Find performer “${q}”`, kind: "search", run: () => { go("performers"); $("#perf-search").value = q; $("#btn-perf-search").click(); } });
    items.push({ text: `Filter catalogue by “${q}”`, kind: "search", run: () => { go("catalogue"); $("#cat-q").value = q; openCatalogue(); } });
  }
  cmdk.items = items.slice(0, 12); cmdk.idx = 0;
  const icon = { page: "→", run: "▸", tier: "●", search: "⌕" };
  $("#cmdk-list").innerHTML = cmdk.items.map((it, i) =>
    `<div class="cmdk-it ${i === cmdk.idx ? "on" : ""}" data-i="${i}"><span class="ci">${icon[it.kind]}</span>${esc(it.text)}<span class="ck">${it.kind === "page" ? "Go to" : it.kind === "run" ? "Run" : it.kind === "tier" ? "Filter" : "Search"}</span></div>`).join("")
    || '<div class="faint" style="padding:12px">No matches</div>';
}
function runCmdk(i) { const it = cmdk.items[i]; if (!it) return; closeCmdk(); it.run(); }
$("#cmd-open")?.addEventListener("click", openCmdk);
$("#cmdk")?.addEventListener("click", (e) => {
  const it = e.target.closest(".cmdk-it");
  if (it) runCmdk(+it.dataset.i); else if (e.target.id === "cmdk") closeCmdk();
});
$("#cmdk-in")?.addEventListener("input", renderCmdk);
$("#cmdk-in")?.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    cmdk.idx = (cmdk.idx + (e.key === "ArrowDown" ? 1 : -1) + cmdk.items.length) % Math.max(1, cmdk.items.length);
    document.querySelectorAll(".cmdk-it").forEach((el, i) => el.classList.toggle("on", i === cmdk.idx));
  } else if (e.key === "Enter") { e.preventDefault(); runCmdk(cmdk.idx); }
  else if (e.key === "Escape") closeCmdk();
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); $("#cmdk").hidden ? openCmdk() : closeCmdk(); }
  else if (e.key === "/" && !e.target.closest("input, textarea, select") && $("#cmdk").hidden && $("#viewer").hidden) { e.preventDefault(); openCmdk(); }
});

// --- Review queue: one scene at a time, big player, 1–5 to grade -------------------
const rv = { items: [], i: 0, label: "", fromCatalogue: false, peaks: [], source: "",
  graded: new Set(), live: null, forceIngest: false };
// same order as GRADES: key 1 = Reject … key 5 = Légendaire
const RV_GRADES = [
  ["reject", "1★ · queued for deletion"], ["upscale", "5★ · O 0 · tag + organize"],
  ["merveilleuse", "5★ · O 16 · tag + organize"], ["exceptionnelle", "5★ · O 17 · tag + organize"],
  ["legendaire", "5★ · O 18 · tag + organize"],
];
function catListLabel() {
  if (cat.isNew) return "New from ingest";
  if (cat.view) return (CAT_VIEWS.find((v) => v[0] === cat.view) || [0, cat.view])[1];
  return cat.tier ? TIER_NAMES[cat.tier] : "All scenes";
}
// new scenes from the running (or last) ingest, in the order they're embedded
async function loadIngestQueue() {
  const d = await api("/api/catalogue?" + new URLSearchParams({ new: "true", tier: "unreviewed", sort: "added", limit: 500 }));
  if (!d.items.length) return false;
  const idn = (x) => +x.scene_id || 0;
  rv.items = d.items.sort((a, b) => idn(a) - idn(b)); rv.i = 0; rv.graded = new Set();   // = embed order
  rv.label = "New from ingest"; rv.source = "ingest";
  return true;
}
async function ingestInfo() {
  try { const d = await api("/api/ingest"); return { ...(d.last || {}), live: !!(d.running || (d.last || {}).running) }; }
  catch { return {}; }
}
async function openReview() {
  const ing = await ingestInfo();
  const force = rv.forceIngest; rv.forceIngest = false;
  if (rv.fromCatalogue && cat.items.length) {
    rv.items = cat.items.slice(); rv.i = Math.max(0, Math.min(cat.focus, rv.items.length - 1));
    rv.label = catListLabel(); rv.source = "catalogue";
  } else if ((force || (ing.live && rv.source !== "ingest") || (!rv.items.length && (ing.new || []).length))
             && await loadIngestQueue().catch(() => false)) {
    /* reviewing the ingest's new scenes */
  } else if (!rv.items.length) {
    $("#rv-title").textContent = "Loading your queue…";
    try {
      let d = await api("/api/catalogue?" + new URLSearchParams({ view: "likely", limit: 200 }));
      rv.label = "Likely keepers";
      if (!d.items.length) {
        d = await api("/api/catalogue?" + new URLSearchParams({ tier: "unreviewed", limit: 200, sort: "date" }));
        rv.label = TIER_NAMES.unreviewed;
      }
      rv.items = d.items; rv.i = 0; rv.source = "default";
    } catch (e) { rv.items = []; toast(e.message, true); }
  }
  rv.fromCatalogue = false;
  renderReview();
  rvLiveTick(ing);
}
// while an ingest runs, keep the queue's scenes fresh: titles/performers after
// identify + auto tag, moments + the model's opinion once each is embedded
async function rvLiveTick(known) {
  clearTimeout(rv.live);
  const ing = known || await ingestInfo();
  const chip = $("#rv-ingest");
  if (chip) {
    chip.hidden = !ing.live;
    if (ing.live) {
      chip.textContent = `⤓ Ingesting · ${ing.stage || "…"}`;
      chip.title = "New scenes are reviewable now. Titles and performers fill in after identify / auto tag, peaks once each scene is embedded. Click for Activity.";
    }
  }
  if (rv.source === "ingest" && !known) await rvMergeFresh();
  if (ing.live && $("#review")?.classList.contains("active")) rv.live = setTimeout(() => rvLiveTick(), 8000);
  else if (!known && rv.source === "ingest") await rvMergeFresh();     // one last refresh at the end
}
async function rvMergeFresh() {
  let d;
  try { d = await api("/api/catalogue?" + new URLSearchParams({ new: "true", sort: "added", limit: 500 })); } catch { return; }
  const fresh = new Map(d.items.map((x) => [x.scene_id, x]));
  const have = new Set(rv.items.map((x) => x.scene_id));
  rv.items = rv.items.map((x) => {
    const f = fresh.get(x.scene_id);
    if (!f || rv.graded.has(x.scene_id)) return x;
    return { ...x, title: f.title, performers: f.performers, studio: f.studio, tags: f.tags, date: f.date,
      moments: f.moments, pred: f.pred, flag: f.flag, dupe: f.dupe, suggest: f.suggest, path: f.path, quality: f.quality,
      who: f.who, performer_ids: f.performer_ids, signals: f.signals };
  });
  d.items.filter((f) => !have.has(f.scene_id) && f.tier === "unreviewed")
    .sort((a, b) => (+a.scene_id || 0) - (+b.scene_id || 0)).forEach((f) => rv.items.push(f));
  if ($("#review")?.classList.contains("active")) renderReview();
}
function renderReview() {
  const r = rv.items[rv.i];
  $("#rv").hidden = !r; $("#rv-empty").hidden = !!r;
  const v = $("#rv-v");
  if (!r) {
    v.removeAttribute("src"); v.load();
    $("#rv-empty").innerHTML = `<h2>Queue done 🎉</h2><p class="muted">Nothing left in ${esc(rv.label || "this list")}.</p>
      <button class="btn pri" data-go="catalogue">Back to Catalogue</button>`;
    return;
  }
  $("#rv-title").textContent = r.title;
  $("#rv-sub").textContent = [r.performers.slice(0, 3).join(", "), r.studio, r.date].filter(Boolean).join(" · ");
  const total = rv.source === "catalogue" ? Math.max(cat.total || 0, rv.items.length) : rv.items.length;
  $("#rv-pos").textContent = `${rv.label} · ${rv.i + 1} of ${total.toLocaleString()}`;
  // player: start at the best moment, peaks marked on the timeline
  rv.peaks = (r.moments || []).map((m) => m.t).sort((a, b) => a - b);
  let start = (r.moments && r.moments.length) ? r.moments.reduce((a, b) => ((b.score || 0) > (a.score || 0) ? b : a)).t : 0;
  if (rv.startAt && rv.startAt.sid === r.scene_id) {
    start = rv.startAt.t; rv.startAt = null;
    if (v.dataset.sid === r.scene_id) { v.currentTime = start; v.play().catch(() => {}); }
  }
  if (v.dataset.sid !== r.scene_id) {
    v.dataset.sid = r.scene_id;
    v.src = r.stream;
    v.onloadedmetadata = () => { v.currentTime = start; v.play().catch(() => {}); };
  }
  const dur = r.duration || 1;
  $("#rv-nopeaks").hidden = rv.peaks.length > 0;
  $("#rv-scrub").innerHTML = `<div class="rv-track"><i id="rv-prog"></i></div>` +
    rv.peaks.map((t) => `<span class="pk" style="left:calc(16px + (100% - 32px) * ${(t / dur).toFixed(4)})" data-t="${t}" title="${fmt(t)}"></span>`).join("");
  // grades: current one outlined, the model's pick highlighted
  const cur = r.tier === "rejected" ? "reject" : r.tier;
  const probs = (r.pred || {}).probs || {};
  const sug = r.suggest ? r.suggest.grade : (r.pred ? r.pred.tier : null);
  $("#rv-grades").innerHTML = RV_GRADES.map(([g, d], k) =>
    `<button class="g g-${g} ${g === cur ? "cur" : ""} ${g === sug ? "sug" : ""}" data-g="${g}"><span class="k">${k + 1}</span>
      <span class="n">${esc(gradeName(g))}</span><span class="d">${d}</span></button>`).join("");
  const order = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "reject"];
  $("#rv-probs").innerHTML = r.pred ? order.map((c) => {
    const p = Math.round(100 * (probs[c] || 0)), t = c === "reject" ? "rejected" : c;
    return `<div class="pb" title="${esc(className(c))} ${p}%"><span>${esc(className(c))}</span><div class="b"><i class="tier-bg-${t}" style="width:${p}%"></i></div><span class="muted num">${p}%</span></div>`;
  }).join("") : (r.moments && r.moments.length
    ? '<span class="faint">The tier model isn\'t trained yet — Catalogue → ⋯ → Train.</span>'
    : '<span class="faint">Not embedded yet — the model\'s opinion appears once it is.</span>');
  $("#rv-why").innerHTML = [r.pred ? `<b class="rv-verdict">${predHTML(r.pred)}</b>` : "", keeperHTML(r.pred),
    r.pred && r.pred.from === "who" ? '<span class="faint">From performer/studio history — not seen yet</span>' : "",
    r.flag ? `<span class="warn">⚠ ${textNum(r.flag, r.flag_detail)}</span>` : "",
    r.suggest ? `Suggest <b>${esc(gradeName(r.suggest.grade))}</b> — ${textNum(r.suggest.why, r.suggest.detail)}` : "",
    r.dupe ? "⧉ Stash thinks this has a duplicate" : ""].filter(Boolean).join("<br>") + whoLines(r, 6)
    + (rvKeepable(r) ? `<div><button class="btn sm" id="rv-keep" title="It's right where it is — don't suggest it again unless something new happens">✓ Keep as ${esc(gradeName(cur))} <kbd>0</kbd></button></div>` : "");
  $("#rv-keep")?.addEventListener("click", rvKeep);
  const q = r.quality || {};
  const sg = r.signals;
  $("#rv-facts").innerHTML = (sg ? `<span>Saved</span><span>${sg.saves ? `★ ${sg.saves} moment${sg.saves === 1 ? "" : "s"}` : '<span class="faint">none yet</span>'}</span>
    ${sg.best != null ? `<span>Best moment</span><span>${bestHTML(sg)} <span class="faint">(taste model)</span></span>` : ""}
    ${sg.age_days != null ? `<span>In library</span><span>${ageHTML(sg)}</span>` : ""}
    ${sg.showings ? `<span>Megaboard</span><span>${sg.passed ? `shown often — you never saved or explored it${fig(`${Math.round(sg.showings)}× on ${sg.shown_days} days`)}`
      : `shown${fig(`${Math.round(sg.showings)}× on ${sg.shown_days} day${sg.shown_days === 1 ? "" : "s"}`)}`}</span>` : ""}` : "")
    + `<span>Quality</span><span>${esc(qualityLine(q))}${q.w ? ` <span class="faint">${q.w}×${q.h}</span>` : ""}</span>
    <span>Size</span><span>${r.size ? fmtBytes(r.size) : "?"} · ${r.duration ? fmt(r.duration) : "?"}</span>
    <span>Grade</span><span>${tierBadge(r.rating100, r.o_counter, { showUnreviewed: true })}</span>
    ${r.tags && r.tags.length ? `<span>Tags</span><span>${esc(r.tags.slice(0, 8).join(", "))}</span>` : ""}
    <span>Path</span><span class="faint path" title="${esc(r.path)}">${esc(r.path)}</span>`;
  $("#rv-queue").innerHTML = rv.items.slice(rv.i + 1, rv.i + 7).map((x, k) => `<div class="qi" data-j="${rv.i + 1 + k}">
      <div class="pic"><img loading="lazy" src="/api/scene/${encodeURIComponent(x.scene_id)}/cover" onerror="this.style.opacity=.1" /></div>
      <div><b>${esc(x.title)}</b><span class="muted">${esc(x.performers.slice(0, 2).join(", "))}${x.pred ? " · " + esc(W.confidence(x.pred.conf).toLowerCase()) + " " + esc(className(x.pred.tier)) : ""}</span></div></div>`).join("")
    || '<span class="faint">Last one in this list.</span>';
}
// a suggestion to answer: the scene has one, or you're working an answerable list
function rvKeepable(r) {
  return !!r && r.tier !== "unreviewed" && r.tier !== "anomaly"
    && (!!r.suggest || (rv.source === "catalogue" && ["promote", "second", "saved", "trim", "passed"].includes(cat.view)));
}
async function rvKeep() {
  const r = rv.items[rv.i]; if (!rvKeepable(r)) return;
  try {
    const fresh = await keepAsIs(r.scene_id);
    rv.graded.add(r.scene_id);
    rv.items[rv.i] = { ...r, ...fresh, suggest: null, moments: r.moments, stream: r.stream, pred: r.pred };
    const ci = rv.source === "catalogue" ? cat.items.findIndex((x) => x.scene_id === r.scene_id) : -1;
    if (ci >= 0) cat.items[ci] = { ...cat.items[ci], suggest: null, _graded: true };
    else cat.loaded = false;
    rvMove(1);
  } catch (e) { toast(e.message, true); }
}
async function rvGrade(grade) {
  const r = rv.items[rv.i]; if (!r) return;
  try {
    const fresh = await applyGrade(r.scene_id, grade);
    rv.graded.add(r.scene_id);
    rv.items[rv.i] = { ...r, ...fresh, moments: r.moments, stream: r.stream, pred: r.pred };
    const ci = rv.source === "catalogue" ? cat.items.findIndex((x) => x.scene_id === r.scene_id) : -1;
    if (ci >= 0) {                      // same list: mark it graded there, keep your place
      catBumpCounts(cat.items[ci].tier, fresh.tier);
      cat.items[ci] = { ...cat.items[ci], ...fresh, _graded: true };
    } else cat.loaded = false;          // the Catalogue re-reads when you go back
    rvMove(1);
  } catch (e) { toast(e.message, true); }
}
function rvMove(d) {
  const n = rv.i + d;
  if (n < 0) return;
  rv.i = Math.min(n, rv.items.length);
  renderReview();
  if (rv.source === "catalogue" && rv.i >= rv.items.length - 3 && cat.items.length < cat.total) rvMoreFromCatalogue();
}
async function rvMoreFromCatalogue() {
  if (rv.loadingMore) return;
  rv.loadingMore = true;
  try {
    await openCatalogue({ append: true });
    const have = new Set(rv.items.map((x) => x.scene_id));
    cat.items.forEach((x) => { if (!have.has(x.scene_id)) rv.items.push(x); });
    if (rv.i >= rv.items.length - 1 || !$("#rv").hidden) renderReview();
  } finally { rv.loadingMore = false; }
}
// leave Review: back to the list you came from, on the scene you were at
function rvExit() {
  $("#rv-v").pause();
  if (rv.source === "catalogue" && cat.loaded) {
    const cur = rv.items[Math.min(rv.i, rv.items.length - 1)];
    const ci = cur ? cat.items.findIndex((x) => x.scene_id === cur.scene_id) : -1;
    showView("catalogue");
    renderCatChips(); renderCatList();
    if (ci >= 0) focusCat(ci);
  } else go("catalogue");
}
function rvJump(dir) {
  const v = $("#rv-v"), t = v.currentTime;
  if (!rv.peaks.length) { v.currentTime = Math.max(0, t + 30 * dir); rvLanded(v.currentTime); return; }   // not embedded yet
  const next = dir > 0 ? rv.peaks.find((p) => p > t + 1) : [...rv.peaks].reverse().find((p) => p < t - 1);
  if (next != null) { v.currentTime = next; rvLanded(next); }
}
// Implicit taste signal: after YOU seek somewhere, watching on for 6 s tells
// Peaks that moment mattered — sent once as a soft "engage" (never a 👍, and
// dropped again if you reject the scene).
const rvEngage = { sid: null, t: null, sent: new Set() };
function rvLanded(t) {
  const r = rv.items[rv.i]; if (!r) return;
  rvEngage.sid = r.scene_id; rvEngage.t = t;
}
function rvEngageTick(v) {
  if (rvEngage.t == null || v.paused || v.dataset.sid !== rvEngage.sid) return;
  const d = v.currentTime - rvEngage.t;
  if (d < 0 || d > 60) { rvEngage.t = null; return; }       // they moved on
  if (d < 6) return;
  const t = rvEngage.t + 1, sig = `${rvEngage.sid}@${Math.round(t / 10)}`;
  rvEngage.t = null;
  if (rvEngage.sent.has(sig)) return;
  rvEngage.sent.add(sig);
  api(`/api/taste/engage?${new URLSearchParams({ scene_id: rvEngage.sid, t: t.toFixed(2) })}`, { method: "POST" }).catch(() => {});
}
$("#rv-grades")?.addEventListener("click", (e) => { const b = e.target.closest("[data-g]"); if (b) rvGrade(b.dataset.g); });
$("#rv-queue")?.addEventListener("click", (e) => { const q = e.target.closest("[data-j]"); if (q) { rv.i = +q.dataset.j; renderReview(); } });
// the whole bottom band of the player — the bar and everything below it — seeks
// (click or drag); only clicks above it play/pause
function rvSeekTo(e) {
  const v = $("#rv-v"), r = rv.items[rv.i]; if (!r) return;
  const pk = e.type === "pointerdown" ? e.target.closest(".pk") : null;
  const box = $("#rv-scrub .rv-track").getBoundingClientRect();   // the bar itself, not the band's padding
  const frac = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
  v.currentTime = pk ? +pk.dataset.t : frac * (v.duration || r.duration || 0);
  const bar = $("#rv-prog"); if (bar) bar.style.width = (100 * frac).toFixed(2) + "%";
}
$("#rv-scrub")?.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  const el = $("#rv-scrub");
  el.setPointerCapture(e.pointerId); el.classList.add("drag");
  rvSeekTo(e);
  const move = (ev) => rvSeekTo(ev);
  const up = () => { el.classList.remove("drag"); el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up);
    rvLanded($("#rv-v").currentTime); };
  el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
});
$("#rv-scrub")?.addEventListener("click", (e) => e.stopPropagation());
$("#rv-v")?.addEventListener("timeupdate", () => {
  const v = $("#rv-v"), bar = $("#rv-prog");
  if (bar && v.duration) bar.style.width = (100 * v.currentTime / v.duration).toFixed(2) + "%";
  rvEngageTick(v);
});
$("#rv-v")?.addEventListener("click", () => { const v = $("#rv-v"); v.paused ? v.play() : v.pause(); });
// volume: a mute button + slider, remembered per browser (starts muted so the
// first scene can autoplay; any change here is a user gesture that unlocks sound)
const rvAudio = (() => {
  try { return JSON.parse(localStorage.getItem("peaks_rv_audio")) || { vol: 0.8, muted: true }; }
  catch { return { vol: 0.8, muted: true }; }
})();
function rvApplyAudio(save = true) {
  const v = $("#rv-v"); if (!v) return;
  v.volume = rvAudio.vol; v.muted = rvAudio.muted || rvAudio.vol === 0;
  const b = $("#rv-mute"); if (b) b.textContent = v.muted ? "🔇" : rvAudio.vol < 0.5 ? "🔉" : "🔊";
  const r = $("#rv-volume"); if (r) r.value = v.muted ? 0 : rvAudio.vol;
  if (save) { try { localStorage.setItem("peaks_rv_audio", JSON.stringify(rvAudio)); } catch { /* ignore */ } }
}
function rvToggleMute() {
  if (rvAudio.muted && rvAudio.vol === 0) rvAudio.vol = 0.8;
  rvAudio.muted = !rvAudio.muted; rvApplyAudio();
}
$("#rv-mute")?.addEventListener("click", rvToggleMute);
$("#rv-volume")?.addEventListener("input", (e) => {
  rvAudio.vol = +e.target.value; rvAudio.muted = rvAudio.vol === 0; rvApplyAudio();
});
rvApplyAudio(false);
$("#rv-exit")?.addEventListener("click", rvExit);
// theater mode: the biggest 16:9 player that fits the window (default on — big screens)
function setTheater(on) {
  $("#rv")?.classList.toggle("theater", on);
  $("#review")?.classList.toggle("theater", on);
  $("#rv-theater")?.classList.toggle("on", on);
  try { localStorage.setItem("peaks_theater", on ? "1" : "0"); } catch { /* ignore */ }
}
function rvFullscreen() {
  const box = $(".rv-player");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else box?.requestFullscreen?.().catch(() => {});
}
$("#rv-theater")?.addEventListener("click", () => setTheater(!$("#rv").classList.contains("theater")));
$("#rv-full")?.addEventListener("click", rvFullscreen);
setTheater((() => { try { return localStorage.getItem("peaks_theater") !== "0"; } catch { return true; } })());
document.addEventListener("keydown", (e) => {
  if (!$("#review")?.classList.contains("active") || !$("#viewer").hidden || !$("#cmdk").hidden) return;
  if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase(), v = $("#rv-v");
  if (k >= "1" && k <= "5") { e.preventDefault(); rvGrade(RV_GRADES[+k - 1][0]); }
  else if (k === "0") { e.preventDefault(); rvKeep(); }
  else if (k === "j") { e.preventDefault(); rvMove(1); }
  else if (k === "k") { e.preventDefault(); rvMove(-1); }
  else if (k === "arrowright") { e.preventDefault(); rvJump(1); }
  else if (k === "arrowleft") { e.preventDefault(); rvJump(-1); }
  else if (k === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (k === "m") { e.preventDefault(); rvToggleMute(); toast(v.muted ? "🔇 muted" : "🔊 sound on"); }
  else if (k === "z") {
    e.preventDefault();
    const sid = gradeUndo.length ? gradeUndo[gradeUndo.length - 1].sid : null;
    undoGrade().then((fresh) => {
      if (!fresh || fresh.bulk) return;
      const i = rv.items.findIndex((x) => x.scene_id === sid);
      if (i >= 0) { rv.items[i] = { ...rv.items[i], ...fresh }; rv.i = i; renderReview(); }
    });
  } else if (k === "t") { e.preventDefault(); setTheater(!$("#rv").classList.contains("theater")); }
  else if (k === "f") { e.preventDefault(); rvFullscreen(); }
  else if (k === "escape") {
    e.preventDefault();
    if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); return; }
    rvExit();
  }
});

// --- Settings → Ingest: what Stash generates when Peaks scans ---------------------
// Rows in Stash's own order and wording; "animated image previews" is a sub-option
// of previews, and video phashes are locked on (duplicates need them).
const SCAN_ROWS = [
  ["scanGenerateCovers", "Generate scene covers"],
  ["scanGeneratePreviews", "Generate previews"],
  ["scanGenerateImagePreviews", "Generate animated image previews", "sub"],
  ["scanGenerateSprites", "Generate scrubber sprites"],
  ["scanGeneratePhashes", "Generate video perceptual hashes", "lock"],
  ["scanGenerateThumbnails", "Generate thumbnails for images"],
  ["scanGenerateImagePhashes", "Generate image perceptual hashes"],
  ["scanGenerateClipPreviews", "Generate previews for image clips"],
  ["rescan", "Rescan files"],
];
let scanOpts = {};
function renderScanToggles() {
  const box = $("#ingest-scan"); if (!box) return;
  box.innerHTML = SCAN_ROWS.map(([k, label, kind]) => {
    const off = (kind === "sub" && !scanOpts.scanGeneratePreviews) || kind === "lock";
    return `<label class="tog-row ${kind === "sub" ? "sub" : ""} ${off ? "off" : ""}">
      <span>${esc(label)}${kind === "lock" ? ' <span class="faint small">· needed for duplicates</span>' : ""}</span>
      <input type="checkbox" class="switch" data-k="${k}" ${scanOpts[k] ? "checked" : ""} ${off ? "disabled" : ""} /></label>`;
  }).join("");
  const on = SCAN_ROWS.filter(([k]) => scanOpts[k]).map(([, l]) => l.replace(/^Generate /, "").replace("video perceptual hashes", "video phashes"));
  const cap = $('#ingest-steps [data-st="scan"] span');
  if (cap) cap.textContent = on.join(" + ") || "nothing extra";
}
async function loadScanOptions() {
  try { scanOpts = (await api("/api/ingest/scan-options")).options; renderScanToggles(); } catch {}
}
$("#ingest-scan")?.addEventListener("change", (e) => {
  const k = e.target.dataset.k; if (!k) return;
  scanOpts[k] = e.target.checked;
  if (k === "scanGeneratePreviews" && !e.target.checked) scanOpts.scanGenerateImagePreviews = false;
  renderScanToggles();
});
$("#btn-ingest-scan-save")?.addEventListener("click", async () => {
  try {
    scanOpts = (await api("/api/ingest/scan-options", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ options: scanOpts }) })).options;
    renderScanToggles();
    $("#ingest-scan-status").textContent = "saved";
    setTimeout(() => { $("#ingest-scan-status").textContent = ""; }, 1500);
  } catch (e) { toast(e.message, true); }
});
loadScanOptions();

// --- Seedbox: the seedbox → library pipeline at a glance (read-only) -------------------
const SB_WORD = { green: "GREEN", yellow: "YELLOW", red: "RED" };
let sbTimer = null;
function sbAgo(sec) {
  if (sec == null) return "never";
  const m = Math.round(sec / 60);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = m / 60;
  return h < 48 ? `${h.toFixed(h < 10 ? 1 : 0)} h ago` : `${Math.round(h / 24)} days ago`;
}
function sbWhen(epoch, now) {
  if (!epoch) return '<span class="faint">never</span>';
  return `${esc(new Date(epoch * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" }))} <span class="faint">· ${sbAgo(now - epoch)}</span>`;
}
const sbNA = (reason) => `<div class="sb-na">Unavailable — ${esc(reason || "unknown reason")}</div>`;
const sbGB = (gb) => gb >= 1000 ? `${(gb / 1000).toFixed(2)} TB` : `${(+gb).toFixed(gb < 10 ? 2 : 1)} GB`;
// what a run accomplished, from runs.log's 4th field (prune only has it if the script logs it)
function sbDid(name, n) {
  if (n == null) return `<span class="faint" title="seedbox-${name} doesn't write a count to runs.log's 4th field">count not logged</span>`;
  return name === "pull" ? `pulled ${plural(n, "file")}` : `pruned ${plural(n, "torrent")}`;
}
function sbScript(name, h, now) {
  const state = h.failing ? '<span class="sb-pill red">failing now</span>'
    : h.last_run_ok === false ? '<span class="sb-pill yellow">last run failed</span>'
    : h.last_ok ? '<span class="sb-pill green">ok</span>' : '<span class="sb-pill red">no success yet</span>';
  return `<div class="sb-script"><div class="row between"><b>seedbox-${name}</b>${state}</div>
    <div class="sb-kv"><span>Last run</span><span>${h.last_run ? `${sbWhen(h.last_run, now)} · ${h.last_run_ok ? "" : '<span class="bad">failed</span> · '}${sbDid(name, h.last_run_count)}` : '<span class="faint">never</span>'}</span>
      <span>Last success</span><span>${sbWhen(h.last_ok, now)}</span>
      <span>Last 24 h</span><span>${h.ok_24h} ok${h.failed_24h ? ` · <span class="bad">${h.failed_24h} failed</span>` : " · 0 failed"}${h.count_24h != null ? ` · ${sbDid(name, h.count_24h)}` : ""}</span></div>
    ${h.errors && h.errors.length ? `<div class="dim small" style="margin-top:6px">Latest errors in pull.log</div><pre class="log sb-errors">${esc(h.errors.join("\n"))}</pre>` : ""}</div>`;
}
function renderSeedbox(d) {
  const now = d.now, v = d.verdict;
  $("#sb-verdict").className = `panel sb-verdict ${v.status}`;
  $("#sb-verdict").innerHTML = `<div class="sb-light"></div><div><div class="sb-status">${SB_WORD[v.status]} · ${esc(v.headline)}</div>
    ${v.reasons.length ? `<ul>${v.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : '<div class="dim">Pulls and prunes are on schedule, qBittorrent answers, and nothing is stuck.</div>'}</div>`;
  const dot = $("#nav-sb-dot");
  if (dot) { dot.hidden = false; dot.className = `sb-dot ${v.status}`; dot.title = `${SB_WORD[v.status]}: ${v.headline}`; }
  const sc = d.scripts, pool = d.pool, inbox = d.inbox;
  $("#sb-refreshed").textContent = `files read ${new Date(now * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` +
    (pool.fetched_at ? ` · qBittorrent ${pool.ok ? `${sbAgo(now - pool.fetched_at)} (cached ${Math.round(d.pool_ttl / 60)} min)`
      : `checked ${sbAgo(now - pool.fetched_at)} (retried every minute)`}` : "");
  // script health
  $("#sb-scripts .sb-body").innerHTML = sc.ok
    ? sbScript("pull", sc.pull, now) + sbScript("prune", sc.prune, now)
      + (sc.malformed ? `<div class="faint small">${plural(sc.malformed, "unreadable line")} in runs.log skipped</div>` : "")
    : sbNA(sc.reason);
  // throughput
  if (sc.ok) {
    const t = sc.throughput, max = Math.max(1, ...t.per_day.map((x) => x.files));
    const ever = Object.entries(sc.ever_pulled || {}).filter(([, n]) => n != null).map(([c, n]) => `${c}: ${n.toLocaleString()}`).join(" · ");
    $("#sb-through .sb-body").innerHTML = `<div class="cards sb-cards">
        <div class="card"><div class="k">Last 24 h</div><div class="v">${plural(t.last_24h, "file")}</div></div>
        <div class="card"><div class="k">Last 7 days</div><div class="v">${plural(t.last_7d, "file")}</div></div></div>
      <div class="sb-chart">${t.per_day.map((x) => `<div class="sb-bar" title="${esc(x.day)}: ${plural(x.files, "file")}">
        <span class="n">${x.files || ""}</span><i style="height:${Math.round(100 * x.files / max)}%"></i><span class="d">${esc(x.day.split(" ")[0])}</span></div>`).join("")}</div>
      ${ever ? `<div class="faint small">All-time pulled (from the .done lists) — ${esc(ever)}</div>` : ""}`;
  } else $("#sb-through .sb-body").innerHTML = sbNA(sc.reason);
  // pool
  if (pool.ok) {
    const cats = Object.entries(pool.by_category).map(([c, b]) =>
      `<span>${esc(c === "1" ? "1 (norating)" : c)}</span><span>${plural(b.count, "torrent")} · ${sbGB(b.gb)}</span>`).join("");
    $("#sb-pool .sb-body").innerHTML = `<div class="cards sb-cards">
        <div class="card"><div class="k">Total</div><div class="v">${sbGB(pool.total.gb)}<span class="faint small"> · ${plural(pool.total.count, "torrent")}</span></div></div>
        <div class="card"><div class="k">Seeding / paused</div><div class="v">${pool.seeding} / ${pool.paused}${pool.other ? `<span class="faint small"> · ${pool.other} other</span>` : ""}</div></div>
        <div class="card"><div class="k">Waiting to be pruned</div><div class="v">${pool.waiting_prune.count} · ${sbGB(pool.waiting_prune.gb)}</div></div></div>
      <div class="sb-kv">${cats}</div>
      ${pool.stale_paused.length ? `<div class="warn small" style="margin-top:8px">Paused longer than 4 days (prune should have taken them):</div>
        <div class="sb-kv small">${pool.stale_paused.slice(0, 6).map((s) => `<span title="${esc(s.name)}">${esc(s.name)}</span><span>${esc(s.category)} · ${s.age_days} d · ${sbGB(s.gb)}</span>`).join("")}</div>` : ""}`;
  } else $("#sb-pool .sb-body").innerHTML = sbNA(pool.reason);
  // inbox
  $("#sb-inbox .sb-body").innerHTML = inbox.ok
    ? `<div class="cards sb-cards"><div class="card"><div class="k">Waiting to be graded</div><div class="v">${plural(inbox.files, "file")} · ${sbGB(inbox.gb)}</div></div></div>
       <div class="faint small"><code>${esc(inbox.path)}</code></div>`
    : sbNA(inbox.reason);
}
async function openSeedbox(refresh) {
  try {
    const d = await api("/api/seedbox" + (refresh ? "/refresh" : ""), refresh ? { method: "POST" } : undefined);
    renderSeedbox(d);
  } catch (e) { $("#sb-verdict").className = "panel sb-verdict red"; $("#sb-verdict").textContent = "Couldn't load: " + e.message; }
  clearInterval(sbTimer);
  sbTimer = setInterval(() => {
    if (document.hidden || !$("#seedbox")?.classList.contains("active")) return;
    openSeedbox();
  }, 60000);
}
$("#btn-sb-refresh")?.addEventListener("click", () => openSeedbox(true));
// the nav dot: one quiet check at start-up (cached server-side, so it's cheap)
setTimeout(() => api("/api/seedbox").then((d) => {
  if (!(d.scripts.ok || d.pool.ok)) return;            // nothing set up yet: no dot
  const dot = $("#nav-sb-dot"); if (!dot) return;
  dot.hidden = false; dot.className = `sb-dot ${d.verdict.status}`; dot.title = `${SB_WORD[d.verdict.status]}: ${d.verdict.headline}`;
}).catch(() => {}), 4000);

// --- folder watch: new downloads ingest themselves --------------------------------
let watchCfg = null;
function watchLine(w) {
  if (!w.watch_on || !w.watch_paths.length) return "";
  const where = w.watch_paths.map((p) => `<code>${esc(p)}</code>`).join(", ");
  const ago = w.checked ? Math.max(0, Math.round((w.now || Date.now() / 1000) - w.checked)) : null;
  const bits = [`👁 Watching ${where}` + (ago == null ? "" : ` <span class="faint">· checked ${ago < 90 ? ago + " s" : Math.round(ago / 60) + " min"} ago</span>`)];
  if (w.check_error) bits.push(`<span class="bad">⚠ folder check failing: ${esc(w.check_error.error)}</span>`);
  if (w.settling) bits.push(`${plural(w.settling, "new file")} still arriving`);
  else if (w.settled) bits.push(`${plural(w.settled, "new file")} ready — ingesting once the folder is quiet`);
  if (w.last_run) bits.push(`last auto-ingest ${ago(w.last_run.at)}: ${plural(w.last_run.files, "file")}`
    + (w.last_ingest && w.last_ingest.stages?.scan ? ` → ${esc(w.last_ingest.stages.scan)}` : ""));
  else if (w.checked) bits.push("nothing new yet");
  if (w.unreadable?.length) bits.push(`<span class="warn">can't read ${esc(w.unreadable.join(", "))}</span>`);
  if (w.last_error) bits.push(`<span class="bad">last auto-ingest failed: ${esc(w.last_error.error)}`
    + (w.retry_after ? ` · retrying at ${new Date(w.retry_after * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "") + "</span>");
  if (w.embed_owed) bits.push(`${plural(w.embed_owed, "new scene")} waiting to be embedded — next as soon as nothing else runs`);
  let html = bits.join(" · ");
  if (w.ingest_warnings?.length) html += w.ingest_warnings.map((x) => `<div class="warn">⚠ ${esc(x)}</div>`).join("");
  return html;
}
async function loadWatch(fresh) {
  let w;
  try { w = fresh || await api("/api/watch"); } catch { return; }
  const line = $("#watch-line");
  if (line) { const h = watchLine(w); line.innerHTML = h; line.hidden = !h; }
  if (!$("#watch-on")) return;
  if (!watchCfg || fresh) {
    watchCfg = { on: w.watch_on, paths: [...w.watch_paths], settle: w.watch_settle, quiet: w.watch_quiet };
    $("#watch-on").checked = watchCfg.on;
    $("#watch-settle").value = String(watchCfg.settle);
    $("#watch-quiet").value = String(watchCfg.quiet);
  }
  renderWatchPaths();
  $("#watch-warn").innerHTML = (w.warnings || []).map((x) => `⚠ ${esc(x)}`).join("<br>");
}
function renderWatchPaths() {
  const el = $("#watch-paths"); if (!el || !watchCfg) return;
  el.innerHTML = watchCfg.paths.length
    ? watchCfg.paths.map((p, i) => `<div class="watch-path"><code>${esc(p)}</code><button class="btn sm ghost" data-wrm="${i}" title="Stop watching">✕</button></div>`).join("")
    : `<div class="faint small">No folder yet — add the one your downloads land in.</div>`;
  el.querySelectorAll("[data-wrm]").forEach((b) => b.onclick = () => { watchCfg.paths.splice(+b.dataset.wrm, 1); renderWatchPaths(); });
}
async function watchDirs() {
  const v = $("#watch-new").value.trim();
  const under = v.endsWith("/") ? v.replace(/\/+$/, "") || "/" : v.includes("/") ? v.slice(0, v.lastIndexOf("/")) || "/" : "";
  try {
    const d = await api("/api/watch/dirs" + (under ? "?under=" + encodeURIComponent(under) : ""));
    $("#watch-dirs").innerHTML = d.dirs.map((x) => `<option value="${esc(x)}">`).join("");
  } catch {}
}
$("#watch-new")?.addEventListener("focus", watchDirs);
$("#watch-new")?.addEventListener("input", () => { clearTimeout(watchDirs.t); watchDirs.t = setTimeout(watchDirs, 200); });
$("#watch-add")?.addEventListener("click", () => {
  const v = $("#watch-new").value.trim().replace(/\/+$/, "");
  if (!v || !watchCfg) return;
  if (!watchCfg.paths.includes(v)) watchCfg.paths.push(v);
  $("#watch-new").value = ""; renderWatchPaths();
});
$("#watch-save")?.addEventListener("click", async () => {
  if (!watchCfg) return;
  watchCfg.on = $("#watch-on").checked;
  watchCfg.settle = +$("#watch-settle").value; watchCfg.quiet = +$("#watch-quiet").value;
  if (watchCfg.on && !watchCfg.paths.length) { toast("Add the folder to watch first", true); return; }
  try {
    const w = await api("/api/watch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(watchCfg) });
    loadWatch(w);
    $("#watch-set-status").textContent = w.watch_on ? "saved — watching (files already there count as seen)" : "saved — off";
  } catch (e) { toast(e.message, true); }
});
$("#watch-check")?.addEventListener("click", async () => {
  try {
    const w = await api("/api/watch?action=check", { method: "POST" });
    loadWatch(w);
    $("#watch-set-status").textContent = w.started ? "new files found — ingest started (see Activity)"
      : w.settling ? `${plural(w.settling, "file")} still arriving — it'll ingest once they've settled`
      : w.watch_on ? "nothing new" : "the watch is off";
  } catch (e) { toast(e.message, true); }
});
loadWatch();
setInterval(() => { if (!document.hidden && $("#activity")?.classList.contains("active")) loadWatch(); }, 30000);

// --- "Loading your library…": the embeddings index (~a minute to load) isn't in
// memory yet — say so instead of showing blank cards, and refresh when it's in
let warmTimer = null, warmWasLoading = false;
async function checkWarm(start) {
  let w;
  try { w = await api("/api/warm" + (start ? "?start=true" : "")); } catch { return; }
  const el = $("#warm-banner"); if (!el) return;
  if (w.loading) {
    const pct = w.pct != null ? Math.round(100 * w.pct)
      : (w.total ? Math.round(100 * (w.done || 0) / w.total) : null);
    const what = w.stage || (w.total ? `Loading embeddings · ${(w.done || 0).toLocaleString()} / ${w.total.toLocaleString()} scenes` : "Loading your library");
    el.innerHTML = `<span class="spin"></span><b>Loading your library</b> · ${esc(what)}${pct != null ? ` · ${pct}%` : ""}
      <span class="faint">— pages fill in on their own when it's ready</span>
      <div class="warm-bar"><i style="width:${pct || 3}%"></i></div>`;
    el.hidden = false;
    warmWasLoading = true;
    if (!warmTimer) warmTimer = setInterval(() => checkWarm(false), 2000);
  } else {
    el.hidden = true;
    clearInterval(warmTimer); warmTimer = null;
    if (warmWasLoading) {                       // it just finished: fill the open page
      warmWasLoading = false;
      const v = document.querySelector(".view.active")?.id;
      if (v && ["foryou", "board", "explore", "performers", "statistics", "taste"].includes(v)) go(v);
    }
  }
}
checkWarm(true);
// coming back to a tab after a while: the index may have been dropped meanwhile
document.addEventListener("visibilitychange", () => { if (!document.hidden) checkWarm(true); });

// (last, so every page's code is defined before the first route runs)
// land on the page in the URL (#/catalogue …), else For You — the home page
refreshSidebar(0);   // sidebar counts from the start, not only once the Catalogue opens
go((location.hash.match(/^#\/(\w+)/) || [])[1] || "foryou");
window.addEventListener("hashchange", () => {
  const v = (location.hash.match(/^#\/(\w+)/) || [])[1];
  if (v && !$("#" + v)?.classList.contains("active")) go(v);
});


// --- crash forensics: if the previous run ended abruptly, say how -----------------
async function loadCrashReport() {
  const card = $("#crash-card"); if (!card) return;
  let d;
  try { d = await api("/api/crash-report"); } catch { return; }
  card.hidden = !d.abrupt;
  if (!d.abrupt) return;
  card.innerHTML = `<div class="row between"><h3 class="warn">⚠ The previous run ended abruptly</h3>
      <span class="row"><a class="btn sm ghost" href="/api/crashlog" download>⬇ crash.log</a>
      <button class="btn sm ghost" id="btn-crash-dismiss">Dismiss</button></span></div>
    <p class="cap">Started ${esc(d.started || "?")} · last sign of life ${esc(d.ended_after || "?")} · <b>${esc(d.kind)}</b>.
      Run <code>docker inspect -f '{{.State.OOMKilled}} {{.State.ExitCode}}' &lt;peaks&gt;</code> too
      (true / 137 = out of memory · 139 = segfault).</p>
    ${d.trace ? `<pre class="log">${esc(d.trace)}</pre>` : ""}
    ${(d.last_health || []).length ? `<div class="small muted">Last health readings before it stopped:</div>
      <pre class="log">${esc(d.last_health.join("\n"))}</pre>` : ""}`;
  $("#btn-crash-dismiss").onclick = async () => {
    try { await api("/api/crash-report/dismiss", { method: "POST" }); } catch {}
    card.hidden = true;
  };
}
