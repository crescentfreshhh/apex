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
        <button data-key="${h.key}" data-t="${h.time}">Find similar</button>
        ${h.stream ? `<a href="${h.stream}" target="_blank">Play</a>` : ""}
      </div>
    </div>`;
  }).join("");
  g.querySelectorAll("button[data-key]").forEach((b) =>
    b.addEventListener("click", () => similar(b.dataset.key, b.dataset.t)));
  g.querySelectorAll(".tile").forEach(wireTileEdits);
}

async function patchScene(sid, body) {
  return api(`/api/scene/${sid}`, {
    method: "PATCH", headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}
function wireTileEdits(tile) {
  const sid = tile.dataset.sid;
  if (!sid) return;
  // rating stars
  tile.querySelectorAll(".star").forEach((s) =>
    s.addEventListener("click", async () => {
      const r = +s.dataset.r;
      try {
        const m = await patchScene(sid, { rating100: r });
        tile.querySelector(".rating").innerHTML = stars(m.rating100);
        wireTileEdits(tile); toast("rating saved");
      } catch (e) { toast(e.message, true); }
    }));
  // organized toggle
  const org = tile.querySelector(".orgbtn");
  org.addEventListener("click", async () => {
    try {
      const m = await patchScene(sid, { organized: !org.classList.contains("on") });
      org.classList.toggle("on", !!m.organized); toast("organized " + (m.organized ? "on" : "off"));
    } catch (e) { toast(e.message, true); }
  });
  // O-count: click +, shift-click −
  const ob = tile.querySelector(".obtn");
  ob.addEventListener("click", async (e) => {
    try {
      const r = await api(`/api/scene/${sid}/o`, { method: e.shiftKey ? "DELETE" : "POST" });
      ob.textContent = `⊙ ${r.o_counter}`;
    } catch (err) { toast(err.message, true); }
  });
}
async function similar(key, t) {
  setActiveView("explore");
  $("#results").innerHTML = '<p class="dim">Finding similar moments…</p>';
  try { renderHits(await api(`/api/search/similar?key=${key}&t=${t}&top_k=60`)); }
  catch (e) { toast(e.message, true); }
}
async function textSearch() {
  const q = $("#q").value.trim(); if (!q) return;
  $("#results").innerHTML = '<p class="dim">Searching…</p>';
  try { renderHits(await api("/api/search/text?q=" + encodeURIComponent(q) + "&top_k=60")); }
  catch (e) { $("#results").innerHTML = ""; toast(e.message, true); }
}
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
