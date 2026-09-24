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
    if (b.dataset.view === "performers") openPerformers();
    if (b.dataset.view === "catalogue" && !cat.loaded) openCatalogue();
    if (b.dataset.view === "statistics") openStatistics();
    if (b.dataset.view === "experimental") openExperimental();
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
      ["Visual model", dino],
      ["Text-search model", clip],
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
  if (typeof loadHistory === "function") loadHistory();
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
async function loadTierNames() {
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
    if ($("#clip-sim-val")) $("#clip-sim-val").textContent = (+c.similarity).toFixed(2);
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
  if ($("#clip-sim-val")) $("#clip-sim-val").textContent = (+e.target.value).toFixed(2);
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
const GRADES = ["legendaire", "exceptionnelle", "merveilleuse", "upscale", "reject"];   // keys 1–5
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
async function applyGrade(sid, grade) {
  const r = await api("/api/catalogue/grade", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_id: String(sid), grade }),
  });
  gradeUndo.push({ sid: String(sid), prev: r.previous, grade });
  toast(`${gradeName(grade)} — Z to undo`);
  return r.scene;
}
async function undoGrade() {
  const u = gradeUndo.pop();
  if (!u) { toast("nothing to undo"); return null; }
  if (u.bulk) return undoBulkGrade(u);
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
        <span class="score ${g.id === "foryou-results" ? scoreBandClass(h.score) : ""}" data-score="${h.score}">${(h.score * 100).toFixed(0)}%</span>
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
    await api(`/api/scene/${sid}/apex?` + new URLSearchParams(p), { method: "POST" });
    toast("Saved moment + added to taste @ " + fmt(t) + (PROFILE.isDefault() ? "" : ` · ${PROFILE.name}`));
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
    const eng = tierTally(r.tiers, { compact: true }) ? ` · ${tierTally(r.tiers, { compact: true })}` : "";
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
          <button class="perf-reel ghost" title="Export a single video of this performer's top 300 taste-ranked moments">⬇ Reel</button>
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
$("#perf-grid")?.addEventListener("click", (e) => {
  const card = e.target.closest(".perf-card"); if (!card) return;
  const { id, name } = card.dataset;
  if (e.target.closest(".perf-best")) performerBestOf(id, name);
  else if (e.target.closest(".perf-play")) playPerformerBoard(id, name);
  else if (e.target.closest(".perf-reel")) exportPerformerReel(id, name);
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
          ${stat("🏆", tierTally(s.tiers) || null)}
          ${stat("✩", s.rating)}
        </div>
        ${dist ? `<div class="dim" style="margin-top:6px">how on-taste her moments are</div>${dist}` : ""}
        ${fp ? `<div class="dim" style="margin:8px 0 4px">known for</div><div class="fy-words">${fp}</div>` : ""}
        <div class="perf-actions" style="margin-top:10px">
          <button id="pd-best" class="primary">⭐ Save best-of</button>
          <button id="pd-board" class="ghost">▶ Endless channel</button>
          <button id="pd-reel" class="ghost" title="Export a single video of her top 300 taste-ranked moments">⬇ Reel</button>
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
  $("#pd-reel").onclick = () => exportPerformerReel(d.id, d.performer);
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
        <div class="dim">★ ${s.affinity != null ? Math.round(s.affinity * 100) + "%" : "—"} · ${tierTally(s.tiers, { compact: true }) || "no tiered scenes"} · ✩ ${s.rating ?? "—"}</div>
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
  if (!$("#foryou")?.classList.contains("active")) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") { e.preventDefault(); swipeRate(1); }   // → = 👍 love
  else if (e.key === "ArrowLeft") { e.preventDefault(); swipeRate(0); }  // ← = 👎 pass
  else if (e.key === "ArrowDown") { e.preventDefault(); loadNextSwipe(); }
});

// --- Experimental: taste coverage / validation -----------------------------
let expData = null;
function updateExpFloorLabel() {
  const v = +$("#exp-floor").value;
  $("#exp-floor-val").textContent = v > 0 ? Math.round(v * 100) + "%" : "off";
}
async function openExperimental() {
  const body = $("#exp-body");
  const f = readFloor();
  if (f != null && isFinite(f)) $("#exp-floor").value = Math.min(Math.max(f, 0), 0.95);
  updateExpFloorLabel();
  body.innerHTML = '<p class="dim">Analyzing your taste coverage…</p>';
  $("#exp-status").textContent = "";
  try {
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
  const coverage = `<div class="panel">
    <div class="exp-big">${uncovered.toLocaleString()} of ${total.toLocaleString()} scenes <span class="dim">have no moment above ${floorTxt} (${pct}%)</span></div>
    <div class="dim">${covered.toLocaleString()} scene(s) are covered by your taste at this floor · scores span ${Math.round(expData.score_range[0] * 100)}%–${Math.round(expData.score_range[1] * 100)}%.</div></div>`;

  const dist = expData.distribution || [];
  const dmax = Math.max(1, ...dist.map((b) => b.n));
  const histBars = dist.map((b) => {
    const hh = Math.round((b.n / dmax) * 100);
    return `<div class="exp-bar ${b.lo >= floor ? "" : "below"}" style="height:${hh}%" title="${Math.round(b.lo * 100)}%+ · ${b.n} scene(s)"></div>`;
  }).join("");
  const histogram = `<div class="panel"><h3 class="exp-h">Per-scene best-score distribution</h3>
    <div class="exp-hist">${histBars}</div><div class="exp-axis"><span>0%</span><span>100%</span></div>
    <p class="dim">Each scene's single best moment. Grey bars (left of the floor) are the uncovered scenes.</p></div>`;

  const curve = expData.coverage_curve || [];
  const curveBars = curve.map((p) => {
    const hh = total ? Math.round((p.covered / total) * 100) : 0;
    return `<div class="exp-bar" style="height:${hh}%" title="floor ${Math.round(p.floor * 100)}% · ${p.covered.toLocaleString()} covered (${hh}%)"></div>`;
  }).join("");
  const curvePanel = `<div class="panel"><h3 class="exp-h">Coverage vs. floor</h3>
    <div class="exp-hist">${curveBars}</div><div class="exp-axis"><span>floor 5%</span><span>95%</span></div>
    <p class="dim">How much of the library stays covered as the floor rises — the knee is a sensible floor.</p></div>`;

  const de = expData.density || {};
  const density = `<div class="panel"><h3 class="exp-h">Sampling density</h3>
    <p class="dim">Moments sampled per scene — min ${de.min}, median ${de.median}, max ${de.max}. ${de.sparse_scenes ? `⚠ ${de.sparse_scenes} scene(s) have fewer than 4 sampled moments (few chances to score — a denser embed interval would help).` : "Every scene has a healthy number of sampled moments."}</p></div>`;

  const w = expData.wall || [];
  const tiles = w.map((it) => `<div class="exp-tile" data-stream="${esc(it.stream || "")}" data-key="${esc(it.key)}" data-t="${it.t}" data-sid="${esc(String(it.scene_id))}">
      <img loading="lazy" src="${it.thumb}" onerror="this.style.display='none'" />
      <span class="exp-score">${Math.round(it.score * 100)}%</span></div>`).join("");
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
];
const MODEL_FREE_VIEWS = new Set(["quality", "anomaly", "conflict"]);
const className = (c) => TIER_NAMES[c === "reject" ? "rejected" : c] || c;

function catParams(offset, refresh) {
  const qs = new URLSearchParams({ offset, limit: CAT_PAGE, sort: $("#cat-sort").value });
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
    const nb = $("#btn-cat-new"), ing = d.ingest || {};
    if (nb) {
      nb.hidden = !ing.new && !cat.isNew;
      nb.classList.toggle("on", cat.isNew);
      nb.innerHTML = `✦ New <span class="n">${(ing.new || 0).toLocaleString()}</span>`;
      nb.title = ing.finished ? `Scenes added by the last ingest (${ing.finished.replace("T", " ").slice(0, 16)})` : "Scenes added by the last ingest";
    }
    cat.loaded = true;
    if (!append) { cat.focus = 0; cat.sel.clear(); cat.anchor = null; }
    renderCatChips(); renderCatTriage(); renderCatList(); renderCatBulk();
    $("#cat-status").textContent = `${cat.items.length.toLocaleString()} of ${d.total.toLocaleString()}`;
  } catch (e) {
    $("#cat-list").innerHTML = `<p class="dim">${esc(e.message)}</p>`;
    $("#cat-status").textContent = "";
  } finally { cat.busy = false; }
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
  const m = cat.model || {}, rep = m.report;
  const trained = !!m.trained;
  $("#cat-views").innerHTML = CAT_VIEWS.map(([v, label, tip]) => {
    const needsModel = !MODEL_FREE_VIEWS.has(v) && !trained;
    return `<button class="cat-chip cat-view ${cat.view === v ? "on" : ""}" data-v="${v}" title="${esc(tip)}${needsModel ? " (train the model first)" : ""}" ${needsModel ? "disabled" : ""}>${esc(label)} <span class="n">${(cat.views[v] || 0).toLocaleString()}</span></button>`;
  }).join("");
  const btn = $("#btn-cat-train");
  btn.textContent = m.training ? "Training…" : trained ? "Retrain" : "Train on my grades";
  btn.disabled = !!m.training;
  let txt = "";
  if (rep && rep.trained) {
    const pct = (x) => Math.round(x * 100) + "%";
    const gain = Math.round((rep.quality_gain || 0) * 100);
    txt = `Trained on ${rep.n} scenes (${rep.trained_at}) · ${pct(rep.cv.exact)} exact · ${pct(rep.cv.within_one)} within one tier` +
      ` · file quality ${gain >= 0 ? "+" : ""}${gain} pts` +
      (m.grades_since_train ? ` · ${m.grades_since_train} grades since` : "") +
      (rep.remembered_rejects ? ` · incl. ${plural(rep.remembered_rejects, "deleted reject")} remembered` : "");
    const short = Object.entries(rep.short || {}).map(([c, n]) => `${className(c)} (${n})`);
    if (short.length) txt += ` · too few to learn: ${short.join(", ")}`;
  } else if (rep && !rep.trained) {
    txt = rep.reason;
  } else {
    txt = "Not trained yet — learns your tiers from the scenes you've graded.";
  }
  $("#cat-model").textContent = txt;
}
function qualityLine(q) {
  return [q.res, q.mbps != null ? `${q.mbps} Mbps` : null, (q.codec || "").toUpperCase() || null,
          q.fps ? `${Math.round(q.fps)}fps` : null].filter(Boolean).join(" · ") || "quality unknown";
}
function catCardHTML(r, i) {
  const who = [r.performers.slice(0, 4).join(", "), r.studio, r.date, r.duration ? fmt(r.duration) : ""].filter(Boolean).join(" · ");
  const moments = (r.moments || []).map((m, j) =>
    `<img class="cat-m" loading="lazy" src="${m.thumb}" data-j="${j}" title="${fmt(m.t)}${m.score != null ? " · taste " + Math.round(m.score * 100) + "%" : ""}" />`).join("");
  const cur = r.tier === "rejected" ? "reject" : r.tier;   // highlight the grade it already has
  const sug = r.suggest ? r.suggest.grade : null;
  const grades = GRADES.map((g, k) =>
    `<button class="cat-g g-${g} ${g === cur ? "cur" : ""} ${g === sug ? "sug" : ""}" data-g="${g}" title="${esc(gradeName(g))} (key ${k + 1})"><kbd>${k + 1}</kbd> ${esc(gradeName(g))}</button>`).join("");
  const p = r.pred;
  const predLine = p
    ? `<div class="cat-pred">Looks like <span class="tier tier-${p.tier === "reject" ? "rejected" : p.tier}">${esc(className(p.tier))}</span> ${Math.round(p.conf * 100)}%` +
      (p.keeper != null ? ` · keeper ${Math.round(p.keeper * 100)}%` : "") + `</div>`
    : "";
  const flag = (r.flag ? `<div class="cat-flag">⚠ ${esc(r.flag)}</div>` : "") +
    (r.dupe ? `<div class="cat-flag">⧉ Stash thinks this has a duplicate — see ⧉ Duplicates</div>` : "");
  const suggest = r.suggest ? `<div class="cat-sug">Suggest <b>${esc(gradeName(r.suggest.grade))}</b> — ${esc(r.suggest.why)}</div>` : "";
  const sel = cat.sel.has(r.scene_id);
  return `<div class="cat-card ${i === cat.focus ? "focus" : ""} ${r._graded ? "graded" : ""} ${sel ? "sel" : ""}" data-i="${i}">
    <label class="cat-selbox" title="Select (X) · shift-click selects a range"><input type="checkbox" class="cat-sel" ${sel ? "checked" : ""} /></label>
    <img class="cat-cover" loading="lazy" src="/api/scene/${encodeURIComponent(r.scene_id)}/cover" onerror="this.style.visibility='hidden'" title="Watch" />
    <div class="cat-body">
      <div class="cat-title">${tierBadge(r.rating100, r.o_counter, { showUnreviewed: true })} <span title="${esc(r.path)}">${esc(r.title)}</span></div>
      <div class="dim cat-who">${esc(who)}</div>
      <div class="cat-q">${esc(qualityLine(r.quality))}</div>
      ${predLine}${flag}${suggest}${tagLine(r)}
      <div class="cat-moments">${moments || '<span class="dim">not embedded yet — no moments</span>'}</div>
    </div>
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
function watchCat(i, j = null) {
  const r = cat.items[i]; if (!r) return;
  const hits = cat.items.map((x) => {
    const m = (x.moments || [])[0];
    return m ? { scene_id: x.scene_id, key: m.key, time: m.t, stream: m.stream }
             : { scene_id: x.scene_id, key: "", time: 0, stream: x.stream };
  });
  const m = j != null ? (r.moments || [])[j] : null;
  if (m) hits[i] = { scene_id: r.scene_id, key: m.key, time: m.t, stream: m.stream };
  lastHits = hits;          // viewer ← / → walk the catalogue page
  openViewerAt(i);
}
$("#cat-chips")?.addEventListener("click", (e) => {
  const b = e.target.closest(".cat-chip"); if (!b) return;
  setDupeMode(false);
  cat.isNew = false;
  cat.tier = b.dataset.t; cat.view = ""; openCatalogue();
});
$("#cat-views")?.addEventListener("click", (e) => {
  const b = e.target.closest(".cat-view"); if (!b || b.disabled) return;
  setDupeMode(false);
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
    toast(r.trained ? `Trained on ${r.n} scenes — ${Math.round(r.cv.exact * 100)}% exact` : r.reason, !r.trained);
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
// One confirm dialog for every file deletion: lists the files, and the delete
// button only enables after ticking "delete the files from disk too".
function showDeleteDialog({ title, summary, note, items, total, goLabel, capable = true, run }) {
  const dlg = $("#del-dlg");
  $("#del-title").textContent = title;
  $("#del-summary").innerHTML = summary;
  $("#del-note").textContent = note || "";
  $("#del-list").innerHTML = items.map((r) => `<div class="hist-row"><span class="hist-what" title="${esc(r.path || "")}">${esc(r.title || r.path)}</span>
    <span class="dim">${r.size ? fmtBytes(r.size) : ""}</span></div>`).join("") +
    (total > items.length ? `<div class="dim">…and ${total - items.length} more</div>` : "");
  const ack = $("#del-ack"), go = $("#btn-del-go"), cancel = $("#btn-del-cancel");
  ack.checked = false; go.disabled = true; cancel.disabled = false;
  $("#del-status").textContent = capable ? "" : "Your Stash version can't delete scenes from the API — update Stash to use this.";
  ack.disabled = !capable;
  go.textContent = goLabel;
  ack.onchange = () => { go.disabled = !ack.checked; };
  cancel.onclick = () => dlg.close();
  go.onclick = async () => {
    go.disabled = true; ack.disabled = true; cancel.disabled = true;
    try {
      await run((msg) => { $("#del-status").textContent = msg; });
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
    summary: `<b>${plural(d.count, "scene")}</b> · <b>${fmtBytes(d.bytes)}</b> will be deleted from Stash, with their files.`,
    note: "Each scene is re-checked right before deletion — anything no longer rated 1★ is skipped. " +
      "Every deleted file is recorded in Settings → History. Peaks remembers what the rejects looked like, " +
      "so keeper triage keeps learning from them.",
    items: d.items, total: d.count, capable: d.capable, goLabel: `Delete ${plural(d.count, "file")}`,
    run: async (status) => {
      const job = await api("/api/catalogue/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene_ids: d.ids, confirm: true }),   // exactly what was previewed
      });
      const j = await waitJob(job.id, (x) => {
        const p = x.progress || {};
        status(`deleting ${p.done ?? 0}/${p.total ?? d.count}…`);
      });
      if (j.status === "error") throw new Error(j.error);
      const r = j.result;
      const extra = [r.refused.length ? `${r.refused.length} skipped (no longer 1★)` : "",
        r.failed.length ? `${r.failed.length} failed` : ""].filter(Boolean).join(" · ");
      toast(`Deleted ${plural(r.deleted, "scene")} · freed ${fmtBytes(r.freed_bytes)}${extra ? " · " + extra : ""}`, !!r.failed.length);
      cat.sel.clear(); openCatalogue(); loadHistory();
    },
  });
}
// --- duplicates: Stash's phash groups, judged on file quality -----------------
const dupe = { on: false, data: null };
function setDupeMode(on) {
  dupe.on = on;
  $("#dupe-box").hidden = !on;
  $("#cat-list").hidden = on;
  $("#btn-cat-more").hidden = on || cat.items.length >= cat.total;
  $("#btn-cat-dupes")?.classList.toggle("on", on);
  if (on) { cat.sel.clear(); renderCatBulk(); }
}
function dupeCopyHTML(r, g) {
  const q = r.quality || {};
  const rec = r.scene_id === g.keep;
  const added = (r.created_at || "").slice(0, 10);
  return `<div class="dupe-copy ${rec ? "rec" : ""}" data-sid="${esc(r.scene_id)}">
    <img class="cat-cover" loading="lazy" src="/api/scene/${encodeURIComponent(r.scene_id)}/cover" onerror="this.style.visibility='hidden'" />
    <div class="dupe-facts">
      <div>${rec ? '<span class="dupe-rec">★ Recommended</span> ' : ""}${tierBadge(r.rating100, r.o_counter, { showUnreviewed: true })}</div>
      <div class="dupe-q"><b>${esc(q.res || "?")}</b> · <b>${q.mbps != null ? q.mbps + " Mbps" : "? Mbps"}</b> · ${esc((q.codec || "").toUpperCase())} ${q.fps ? Math.round(q.fps) + "fps" : ""}</div>
      <div class="dim">${r.size ? fmtBytes(r.size) : "size ?"} · ${r.duration ? fmt(r.duration) : "?"}${added ? " · added " + esc(added) : ""}</div>
      <div class="dim dupe-path" title="${esc(r.path)}">${esc(r.path)}</div>
    </div>
    <button class="${rec ? "primary" : "ghost"} dupe-keep">Keep this, delete ${g.scenes.length === 2 ? "the other" : "the others"}</button>
  </div>`;
}
function renderDupes() {
  const d = dupe.data, box = $("#dupe-list");
  if (!d || !d.groups) { box.innerHTML = ""; return; }
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
  setDupeMode(true);
  try { const d = await api("/api/duplicates"); if (d.groups) { dupe.data = d; renderDupes(); } } catch {}
}
$("#btn-cat-new")?.addEventListener("click", () => {
  setDupeMode(false);
  cat.isNew = !cat.isNew;
  if (cat.isNew) { cat.tier = ""; cat.view = ""; }
  openCatalogue();
});
async function startIngest() {
  try {
    const job = await api("/api/ingest", { method: "POST" });
    tracked.add(job.id);
    const btn = $("#btn-ingest"), statusEl = $("#ingest-status"), logEl = $("#ingest-log"), stop = $("#btn-ingest-stop");
    if (btn) { btn.disabled = true; logEl.hidden = false; wireStop(stop, statusEl, job.id); }
    const cb = $("#btn-cat-ingest"); if (cb) cb.disabled = true;
    const j = await waitJob(job.id, (x) => {
      const p = x.progress || {}, line = (x.log || []).slice(-1)[0] || "starting…";
      $("#cat-status").textContent = "Ingest: " + line;
      if (statusEl) statusEl.textContent = `${x.status}${p.stage && p.stage !== "done" ? " · " + p.stage : ""} · ${x.elapsed}s`;
      if (logEl) { logEl.textContent = (x.log || []).join("\n"); logEl.scrollTop = logEl.scrollHeight; }
    }, 1500);
    if (btn) { btn.disabled = false; if (stop) stop.hidden = true; }
    if (cb) cb.disabled = false;
    if (j.status === "error") { toast("Ingest failed: " + j.error, true); $("#cat-status").textContent = ""; return; }
    if (j.status === "cancelled") { toast("Ingest stopped."); return; }
    const r = j.result;
    toast(`Ingest done: ${plural(r.new, "new scene")}${r.duplicates ? ` · ${plural(r.duplicates, "duplicate group")}` : ""}`);
    loadHistory();
    if (r.new) { setDupeMode(false); cat.isNew = true; cat.tier = ""; cat.view = ""; }
    if (cat.loaded || r.new) openCatalogue({ refresh: true });
  } catch (e) { toast(e.message, true); }
}
$("#btn-ingest")?.addEventListener("click", startIngest);
$("#btn-cat-ingest")?.addEventListener("click", () => {
  if (confirm("Ingest new files?\n\nStash scans the library (phashes on), identifies and auto-tags the new scenes with your saved task defaults, then Peaks embeds them and checks for duplicates.")) startIngest();
});
$("#btn-cat-dupes")?.addEventListener("click", () => (dupe.on ? (setDupeMode(false), openCatalogue()) : openDupes()));
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
    const keep = g.scenes.find((r) => r.scene_id === copy.dataset.sid);
    const others = g.scenes.filter((r) => r !== keep);
    const order = ["upscale", "merveilleuse", "exceptionnelle", "legendaire"];
    const carry = g.best_grade && order.indexOf(keep.tier) < order.indexOf(g.best_grade);
    const bytes = others.reduce((a, r) => a + (+r.size || 0), 0);
    showDeleteDialog({
      title: "Keep one copy, delete the others",
      summary: `Keeping <b>${esc(keep.quality.res || "?")} · ${keep.quality.mbps ?? "?"} Mbps</b> — ${esc(keep.path)}.<br>` +
        `Deleting <b>${plural(others.length, "copy", "copies")}</b> (${fmtBytes(bytes)}) with their files:` +
        (carry ? `<br>The kept copy is graded <b>${esc(TIER_NAMES[g.best_grade])}</b> first (tier tag + organized), so the grade isn't lost.` : ""),
      note: "Duplicate copies aren't counted as rejects. Every deleted file is recorded in Settings → History.",
      items: others, total: others.length, goLabel: `Delete ${plural(others.length, "copy", "copies")}`,
      run: async (status) => {
        const job = await api("/api/duplicates/resolve", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keep: keep.scene_id, delete: others.map((r) => r.scene_id), confirm: true }),
        });
        const j = await waitJob(job.id, () => status("deleting…"));
        if (j.status === "error") throw new Error(j.error);
        const r = j.result;
        toast(`Kept 1 · deleted ${r.deleted} · freed ${fmtBytes(r.freed_bytes)}${r.carried_grade ? " · grade carried over" : ""}`);
        dupe.data.groups = dupe.data.groups.filter((x) => x !== g);
        dupe.data.reclaim -= g.reclaim;
        renderDupes(); loadHistory();
      },
    });
  }
});
$("#btn-cat-delete")?.addEventListener("click", () => openDeleteDialog(null));
$("#btn-cat-delsel")?.addEventListener("click", () =>
  openDeleteDialog(cat.items.filter((r) => cat.sel.has(r.scene_id)).map((r) => r.scene_id)));
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
