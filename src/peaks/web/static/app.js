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
      ["Library", stats.library_path],
    ].map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  } catch (e) {
    $("#conn").textContent = "disconnected"; toast("Cannot reach backend: " + e.message, true);
  }
}

function wireJob(btn, statusEl, logEl, start) {
  btn.addEventListener("click", async () => {
    btn.disabled = true; statusEl.textContent = "starting…"; logEl.hidden = false; logEl.textContent = "";
    try {
      const job = await start();
      poll(job.id, statusEl, logEl, btn);
    } catch (e) {
      btn.disabled = false; statusEl.textContent = ""; toast(e.message, true);
    }
  });
}
async function poll(id, statusEl, logEl, btn) {
  try {
    const j = await api("/api/jobs/" + id);
    const p = j.progress || {};
    statusEl.textContent = `${j.status} · ${p.done ?? 0}/${p.total ?? "?"} · ${j.elapsed}s`;
    logEl.textContent = (j.log || []).join("\n"); logEl.scrollTop = logEl.scrollHeight;
    if (j.status === "running") return setTimeout(() => poll(id, statusEl, logEl, btn), 1000);
    btn.disabled = false;
    if (j.status === "error") toast("Job failed: " + j.error, true);
    else { toast("Done: " + JSON.stringify(j.result || {})); refreshDashboard(); }
  } catch (e) { btn.disabled = false; toast(e.message, true); }
}
wireJob($("#btn-embed"), $("#embed-status"), $("#embed-log"), () => api("/api/embed", { method: "POST" }));
wireJob($("#btn-score"), $("#score-status"), $("#score-log"), () => {
  const tag = $("#score-tag").value.trim();
  const write = $("#score-write").checked;
  const qs = new URLSearchParams(); if (tag) qs.set("tag", tag); if (write) qs.set("write", "true");
  return api("/api/score?" + qs, { method: "POST" });
});

// --- explore / search -------------------------------------------------------
function renderHits(hits) {
  const g = $("#results");
  if (!hits.length) { g.innerHTML = '<p class="dim">No results.</p>'; return; }
  g.innerHTML = hits.map((h) => {
    const perf = (h.performers || []).slice(0, 3).join(", ");
    const sub = [h.studio, perf].filter(Boolean).join(" · ") || `scene ${h.scene_id ?? "?"}`;
    const title = h.title || `scene ${h.scene_id ?? "?"}`;
    return `<div class="tile">
      <div class="thumbwrap">
        <img loading="lazy" src="${h.thumb}" alt="" onerror="this.style.opacity=.15" />
        <span class="score">${(h.score * 100).toFixed(0)}%</span>
        <span class="t">${fmt(h.time)}</span>
      </div>
      <div class="meta">
        <div class="title" title="${esc(title)}">${esc(title)}</div>
        <div class="sub" title="${esc(sub)}">${esc(sub)}</div>
      </div>
      <div class="actions">
        <button data-key="${h.key}" data-t="${h.time}">Find similar</button>
        ${h.stream ? `<a href="${h.stream}" target="_blank">Play</a>` : ""}
      </div>
    </div>`;
  }).join("");
  g.querySelectorAll("button[data-key]").forEach((b) =>
    b.addEventListener("click", () => similar(b.dataset.key, b.dataset.t)));
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
