/* Opus megaboard — a grid of looping "apex" tiles that continuously cycles.
 *
 * Click a tile to ENLARGE it in place: the video swaps to the whole scene so
 * you can scrub freely (this scene's other apexes show as chapter ticks), and a
 * panel shows the Stash stats — rating, O-count and organized are editable and
 * write straight back to Stash. Click the video again (or Esc) to collapse.
 *
 * Playlist data (with baked stream URLs) comes from playlist.json. Live
 * metadata comes from the same-origin /api (present when served by `peaks web`);
 * if that API isn't there the board still plays, it just hides the stats panel.
 */

const State = {
  apexes: [],
  tiles: [],
  playing: true,
  big: null, // the tile currently enlarged
  n: 3,
  shuffle: false, // whole-library random mode
  pool: null, // scene pool for shuffle
  searchMode: false,
  source: null, // current source id (for Refresh)
};

// --- weighted, no-immediate-repeat picker ---------------------------------

function makePicker(apexes) {
  const weights = apexes.map((a) => Math.max(0.0001, a.score ?? 1));
  const total = weights.reduce((s, w) => s + w, 0);
  let lastIdx = -1;
  return function pick() {
    if (apexes.length === 1) return apexes[0];
    for (let attempt = 0; attempt < 8; attempt++) {
      let r = Math.random() * total;
      let idx = 0;
      while (r > weights[idx] && idx < weights.length - 1) {
        r -= weights[idx];
        idx++;
      }
      if (idx !== lastIdx) {
        lastIdx = idx;
        return apexes[idx];
      }
    }
    return apexes[Math.floor(Math.random() * apexes.length)];
  };
}

let pickApex = () => null;

// --- same-origin API (best-effort; the board works without it) -------------

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}

// --- tiles ----------------------------------------------------------------

function loadApex(tile) {
  const apex = pickApex();
  if (!apex) return;
  const v = tile.video;
  tile.apex = apex;
  tile.mode = "offset"; // re-detected per stream on loadedmetadata
  tile.label.textContent = `#${apex.scene_id} · ${fmt(apex.start)} (${apex.duration.toFixed(0)}s)`;
  v.loop = false;
  v.src = apex.url;
  v.muted = true;
  v.load();
  if (State.playing) v.play().catch(() => {});
}

function advance(tile) {
  if (!State.playing || State.big === tile) return; // enlarged tile plays free
  const now = performance.now();
  if (tile.lastAdvance && now - tile.lastAdvance < 500) return;
  tile.lastAdvance = now;
  loadApex(tile);
}

function makeTile(index) {
  const el = document.createElement("div");
  el.className = "tile";

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  const label = document.createElement("span");
  label.className = "label";

  el.append(video, label);
  const tile = { el, video, label, index, apex: null, mode: "offset", lastAdvance: 0 };

  video.addEventListener("loadedmetadata", () => {
    if (!tile.apex || State.big === tile) return; // big mode drives seeking itself
    if (Number.isFinite(video.duration) && video.duration > tile.apex.duration + 5) {
      tile.mode = "absolute";
      video.currentTime = tile.apex.start;
    } else {
      tile.mode = "offset";
    }
  });

  video.addEventListener("timeupdate", () => {
    if (!tile.apex || State.big === tile) return;
    const end = tile.mode === "absolute" ? tile.apex.end : tile.apex.duration;
    if (video.currentTime >= end - 0.25) advance(tile);
  });
  video.addEventListener("ended", () => advance(tile));
  video.addEventListener("error", () => {
    if (State.big !== tile) setTimeout(() => advance(tile), 500);
  });

  el.addEventListener("click", (e) => {
    if (e.target.closest(".mb-controls, .mb-meta")) return; // interacting, not toggling
    if (State.big === tile) return collapse(tile);
    if (State.big) collapse(State.big);
    expand(tile);
  });
  return tile;
}

// --- enlarge in place ------------------------------------------------------

function bigSpan(n) {
  return Math.min(n, Math.max(2, Math.ceil(n * 0.6)));
}

function layoutBig(tile, on) {
  const board = document.getElementById("board");
  const n = State.n;
  if (on) {
    const s = bigSpan(n);
    const r = Math.floor(tile.index / n);
    const c = tile.index % n;
    const r0 = Math.min(r, n - s);
    const c0 = Math.min(c, n - s);
    tile.el.style.gridArea = `${r0 + 1} / ${c0 + 1} / span ${s} / span ${s}`;
    board.style.gridAutoFlow = "dense";
    // keep exactly the cells that remain filled; hide the overflow tiles
    let room = n * n - s * s;
    for (const t of State.tiles) {
      if (t === tile) continue;
      t.el.style.display = room-- > 0 ? "" : "none";
    }
  } else {
    tile.el.style.gridArea = "";
    board.style.gridAutoFlow = "";
    for (const t of State.tiles) t.el.style.display = "";
  }
}

function sceneStreamUrl(apexUrl) {
  // rewrite start=<apex> to start=0 so the whole scene is seekable (apikey kept)
  try {
    const u = new URL(apexUrl, location.href);
    u.searchParams.set("start", "0");
    return u.toString();
  } catch {
    return apexUrl;
  }
}

function expand(tile) {
  State.big = tile;
  tile.el.classList.add("big");
  layoutBig(tile, true);
  buildBigUI(tile);

  const v = tile.video;
  const apex = tile.apex;
  v.loop = false;
  v.muted = false;
  v.src = sceneStreamUrl(apex.url);
  v.load();
  const onMeta = () => {
    v.removeEventListener("loadedmetadata", onMeta);
    if (Number.isFinite(v.duration) && v.duration > 0) {
      tile.sceneDuration = v.duration;
      try {
        v.currentTime = Math.min(apex.start, v.duration - 0.1);
      } catch {}
      setupScrubber(tile);
    }
  };
  v.addEventListener("loadedmetadata", onMeta);
  if (State.playing) v.play().catch(() => {});
  loadMeta(tile);
  clearInterval(tile.clipTimer);
  tile.clipTimer = setInterval(() => {
    if (State.big === tile && !tile.video.paused) classifyBig(tile);
  }, 1900);
  classifyBig(tile);
}

function collapse(tile) {
  tile.el.classList.remove("big");
  layoutBig(tile, false);
  teardownBigUI(tile);
  State.big = null;
  tile.video.muted = true;
  loadApex(tile); // resume cycling with a fresh apex
}

// --- the enlarged tile's overlay UI (controls + editable stats) ------------

function buildBigUI(tile) {
  const meta = document.createElement("div");
  meta.className = "mb-meta";
  meta.innerHTML = `<div class="mb-title">#${tile.apex.scene_id}</div>
    <div class="mb-sub dim">loading…</div>
    <div class="mb-clip"></div>
    <div class="mb-edit"></div>`;

  const controls = document.createElement("div");
  controls.className = "mb-controls";
  controls.innerHTML = `
    <button class="mb-play" title="play / pause">❚❚</button>
    <div class="mb-seekwrap"><input class="mb-seek" type="range" min="0" max="1000" value="0" />
      <div class="mb-ticks"></div></div>
    <span class="mb-time dim">0:00 / 0:00</span>
    <button class="mb-mute" title="mute">🔊</button>`;

  tile.el.append(meta, controls);
  tile.ui = { meta, controls };

  const v = tile.video;
  controls.querySelector(".mb-play").addEventListener("click", () => {
    if (v.paused) v.play().catch(() => {}); else v.pause();
  });
  v.addEventListener("play", () => (controls.querySelector(".mb-play").textContent = "❚❚"));
  v.addEventListener("pause", () => (controls.querySelector(".mb-play").textContent = "▶"));
  const mute = controls.querySelector(".mb-mute");
  mute.addEventListener("click", () => {
    v.muted = !v.muted;
    mute.textContent = v.muted ? "🔇" : "🔊";
  });
  mute.textContent = v.muted ? "🔇" : "🔊";
}

function teardownBigUI(tile) {
  clearInterval(tile.clipTimer);
  tile.ui?.meta.remove();
  tile.ui?.controls.remove();
  tile.ui = null;
}

// live "CLIP sees" for the enlarged tile — classify by scene_id (no key here)
async function classifyBig(tile) {
  const el = tile.ui && tile.ui.meta.querySelector(".mb-clip");
  if (!el || !tile.apex) return;
  try {
    const d = await api(`/api/classify?scene_id=${encodeURIComponent(tile.apex.scene_id)}&t=${(tile.video.currentTime || 0).toFixed(2)}`);
    const labs = d.labels || [];
    el.innerHTML = labs.length
      ? `<span class="dim">CLIP sees</span> ` + labs.slice(0, 5).map(([l]) => `<span class="mb-chip">${esc(l)}</span>`).join("")
      : "";
  } catch {}
}

function setupScrubber(tile) {
  const v = tile.video;
  const dur = tile.sceneDuration || v.duration || 0;
  if (!tile.ui || !dur) return;
  const seek = tile.ui.controls.querySelector(".mb-seek");
  const time = tile.ui.controls.querySelector(".mb-time");
  const ticks = tile.ui.controls.querySelector(".mb-ticks");
  seek.max = 1000;

  // chapter ticks: every apex belonging to this scene
  ticks.innerHTML = "";
  for (const a of State.apexes) {
    if (String(a.scene_id) !== String(tile.apex.scene_id)) continue;
    const tick = document.createElement("span");
    tick.className = "mb-tick";
    tick.style.left = `${(a.start / dur) * 100}%`;
    tick.title = `apex @ ${fmt(a.start)}`;
    tick.addEventListener("click", (e) => {
      e.stopPropagation();
      try { v.currentTime = a.start; } catch {}
    });
    ticks.appendChild(tick);
  }

  let dragging = false;
  const fmtTime = () => (time.textContent = `${fmt(v.currentTime)} / ${fmt(dur)}`);
  seek.addEventListener("input", () => {
    dragging = true;
    time.textContent = `${fmt((seek.value / 1000) * dur)} / ${fmt(dur)}`;
  });
  seek.addEventListener("change", () => {
    try { v.currentTime = (seek.value / 1000) * dur; } catch {}
    dragging = false;
  });
  v.addEventListener("timeupdate", () => {
    if (dragging || State.big !== tile) return;
    seek.value = Math.round((v.currentTime / dur) * 1000);
    fmtTime();
  });
  fmtTime();
}

// --- editable Stash stats (rating / O-count / organized) -------------------

function starsHTML(r) {
  const filled = Math.round((r || 0) / 20);
  let s = "";
  for (let i = 1; i <= 5; i++)
    s += `<span class="mb-star ${i <= filled ? "on" : ""}" data-r="${i * 20}">★</span>`;
  return s;
}

async function loadMeta(tile) {
  const sid = tile.apex.scene_id;
  const meta = tile.ui?.meta;
  if (!meta) return;
  let m;
  try {
    m = await api("/api/scene/" + sid);
  } catch {
    meta.querySelector(".mb-sub").textContent = "stats unavailable (open via the web app)";
    meta.querySelector(".mb-edit").innerHTML = "";
    return;
  }
  const perf = (m.performers || []).slice(0, 4).join(", ");
  const sub = [m.studio, perf].filter(Boolean).join(" · ");
  meta.querySelector(".mb-title").textContent = m.title || `scene ${sid}`;
  meta.querySelector(".mb-sub").textContent =
    [sub, m.date, (m.tags || []).slice(0, 4).join(", ")].filter(Boolean).join("  ·  ");
  const edit = meta.querySelector(".mb-edit");
  edit.innerHTML = `
    <span class="mb-rating">${starsHTML(m.rating100)}</span>
    <button class="mb-o" title="O-count (click +, shift-click −)">⊙ ${m.o_counter ?? 0}</button>
    <button class="mb-org ${m.organized ? "on" : ""}" title="organized">✓ organized</button>`;
  wireStatEdits(tile, edit);
}

async function patchScene(sid, body) {
  return api("/api/scene/" + sid, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function wireStatEdits(tile, edit) {
  const sid = tile.apex.scene_id;
  edit.querySelectorAll(".mb-star").forEach((s) =>
    s.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        const m = await patchScene(sid, { rating100: +s.dataset.r });
        edit.querySelector(".mb-rating").innerHTML = starsHTML(m.rating100);
        wireStatEdits(tile, edit);
      } catch {}
    })
  );
  const o = edit.querySelector(".mb-o");
  o.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      const r = await api("/api/scene/" + sid + "/o", { method: e.shiftKey ? "DELETE" : "POST" });
      o.textContent = `⊙ ${r.o_counter}`;
    } catch {}
  });
  const org = edit.querySelector(".mb-org");
  org.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      const m = await patchScene(sid, { organized: !org.classList.contains("on") });
      org.classList.toggle("on", !!m.organized);
    } catch {}
  });
}

// --- board layout ---------------------------------------------------------

const STAGGER_MS = 250;

function buildBoard(n) {
  if (State.big) collapse(State.big);
  const board = document.getElementById("board");
  board.innerHTML = "";
  board.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  board.style.gridTemplateRows = `repeat(${n}, 1fr)`;
  board.style.gridAutoFlow = "";
  State.tiles = [];
  State.n = n;
  const generation = (State.boardGen = (State.boardGen || 0) + 1);
  for (let i = 0; i < n * n; i++) {
    const tile = makeTile(i);
    State.tiles.push(tile);
    board.appendChild(tile.el);
    setTimeout(() => {
      if (State.boardGen === generation) loadApex(tile);
    }, i * STAGGER_MS);
  }
  updateStatus();
}

// --- controls -------------------------------------------------------------

function setPlaying(on) {
  State.playing = on;
  document.getElementById("toggle").textContent = on ? "Pause" : "Play";
  for (const t of State.tiles) {
    if (on) t.video.play().catch(() => {});
    else t.video.pause();
  }
}

function reshuffle() {
  if (State.big) collapse(State.big);
  State.tiles.forEach((t, i) => setTimeout(() => loadApex(t), i * STAGGER_MS));
}

function updateStatus() {
  const label = State.shuffle
    ? `shuffle · ${(State.pool || []).length} scenes`
    : `${State.apexes.length} ${State.searchMode ? "moments" : "apexes"}`;
  document.getElementById("status").textContent = `${label} · ${State.tiles.length} tiles`;
}

function fmt(sec) {
  sec = Math.max(0, sec || 0);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function wireControls() {
  document.getElementById("grid").addEventListener("change", (e) => {
    buildBoard(parseInt(e.target.value, 10));
  });
  document.getElementById("toggle").addEventListener("click", () => setPlaying(!State.playing));
  document.getElementById("reshuffle").addEventListener("click", reshuffle);
  document.getElementById("refresh-lib").addEventListener("click", () => {
    const src = State.source || document.getElementById("source").value;
    loadSource(src, { refresh: true }); // rebuild pool/apexes from Stash
  });
  document.getElementById("mute").addEventListener("click", () => {
    if (State.big) {
      State.big.video.muted = true;
      const mb = State.big.ui?.controls.querySelector(".mb-mute");
      if (mb) mb.textContent = "🔇";
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && State.big) collapse(State.big);
  });
}

// --- sources: shuffle-all / apex tag / saved collection / search handoff ----

const SHUFFLE_MIN = 7, SHUFFLE_MAX = 22; // random clip length (s) => random intervals
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function withStart(url, start) {
  try { const u = new URL(url, location.href); u.searchParams.set("start", Math.round(start)); return u.toString(); }
  catch { return url; }
}
// shuffle: a fresh random moment (random scene, random start, random length)
function randomMoment() {
  const pool = State.pool;
  if (!pool || !pool.length) return null;
  const s = pool[Math.floor(Math.random() * pool.length)];
  const dur = s.duration || 60;
  const clip = SHUFFLE_MIN + Math.random() * (SHUFFLE_MAX - SHUFFLE_MIN);
  const start = Math.max(0, Math.random() * Math.max(1, dur - clip));
  return {
    scene_id: s.scene_id, start: Math.round(start), end: Math.round(start + clip),
    duration: Math.round(clip), url: withStart(s.url, start),
  };
}

async function loadSource(src, opts = {}) {
  State.source = src;
  State.shuffle = false; State.searchMode = false; State.pool = null; State.apexes = [];
  document.getElementById("error").hidden = true;
  document.getElementById("status").textContent = "loading…";
  try {
    if (src === "shuffle") {
      const d = await api("/api/board/scenes" + (opts.refresh ? "?refresh=true" : ""));
      State.pool = d.scenes || [];
      if (!State.pool.length) return showError("No scenes found in your library scope.");
      State.shuffle = true;
      pickApex = randomMoment;
    } else if (src.startsWith("collection:")) {
      const pl = await api("/api/collection?name=" + encodeURIComponent(src.slice(11)));
      State.apexes = pl.apexes || []; State.searchMode = true;
      if (!State.apexes.length) return showError("This collection has no moments.");
      pickApex = makePicker(State.apexes);
    } else if (src === "search") {
      let pl = null; try { pl = JSON.parse(localStorage.getItem("mb_search") || "null"); } catch {}
      State.apexes = (pl && pl.apexes) || []; State.searchMode = true;
      if (!State.apexes.length) return showError("No search results to play. Run a search and hit 'Play on megaboard'.");
      pickApex = makePicker(State.apexes);
    } else if (src === "foryou") {
      let pl = null; try { pl = JSON.parse(localStorage.getItem("mb_foryou") || "null"); } catch {}
      State.apexes = (pl && pl.apexes) || []; State.searchMode = true;
      if (!State.apexes.length) return showError("No For You feed to play. Open the For You tab and hit 'Play on megaboard'.");
      pickApex = makePicker(State.apexes);
    } else if (src === "galaxy") {
      let pl = null; try { pl = JSON.parse(localStorage.getItem("mb_galaxy") || "null"); } catch {}
      State.apexes = (pl && pl.apexes) || []; State.searchMode = true;
      if (!State.apexes.length) return showError("No galaxy selection to play. Select a region in the Galaxy tab.");
      pickApex = makePicker(State.apexes);
    } else {
      const tag = src.startsWith("tag:") ? src.slice(4) : src;
      const pl = await api("/api/board/apexes?tag=" + encodeURIComponent(tag));
      State.apexes = pl.apexes || [];
      if (!State.apexes.length)
        return showError(`No apexes for tag "${tag}".\n\nMark some with ⚑ Apex in Explore — or pick "Shuffle" above to play everything.`);
      pickApex = makePicker(State.apexes);
    }
  } catch (err) {
    return showError("Couldn't load this source.\n\n(" + err.message + ")");
  }
  buildBoard(parseInt(document.getElementById("grid").value, 10));
}

async function initSources(initial) {
  const sel = document.getElementById("source");
  const opts = [`<option value="shuffle">Shuffle — whole library</option>`];
  try {
    const s = await api("/api/board/sources");
    opts.push(`<option value="tag:${esc(s.tag)}">Apexes: ${esc(s.tag)}</option>`);
    for (const c of s.collections || [])
      opts.push(`<option value="collection:${esc(c.safe)}">Collection: ${esc(c.name)} (${c.count})</option>`);
  } catch {}
  if (initial === "search") opts.push(`<option value="search">Search results</option>`);
  if (initial === "foryou") opts.push(`<option value="foryou">For You — your taste</option>`);
  if (initial === "galaxy") opts.push(`<option value="galaxy">Galaxy selection</option>`);
  sel.innerHTML = opts.join("");
  sel.addEventListener("change", () => loadSource(sel.value));
  return sel;
}

// --- boot -----------------------------------------------------------------

async function main() {
  wireControls();
  const params = new URLSearchParams(location.search);
  let initial = null;
  if (params.get("src") === "search") initial = "search";
  else if (params.get("src") === "foryou") initial = "foryou";
  else if (params.get("src") === "galaxy") initial = "galaxy";
  else if (params.get("collection")) initial = "collection:" + params.get("collection");
  const sel = await initSources(initial);
  if (initial) sel.value = initial;
  else {
    const tagOpt = [...sel.options].find((o) => o.value.startsWith("tag:"));
    sel.value = tagOpt ? tagOpt.value : "shuffle"; // default to apexes, else shuffle
  }
  loadSource(sel.value);
}

function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = msg;
  el.hidden = false;
}

main();
