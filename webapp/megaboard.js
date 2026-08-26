/* Peaks megaboard — a grid of looping moment tiles that continuously cycles.
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
  hover: null, // last-hovered tile (keyboard actions target this / big)
  overTile: null, // tile the cursor is over RIGHT NOW (drives hover-to-hear)
  menuTile: null, // tile whose right-click menu is open (keeps its audio playing while it's up)
  n: 3,
  shuffle: false, // whole-library random mode
  pool: null, // scene pool for shuffle
  searchMode: false,
  source: null, // current source id (for Refresh)
  sourceSpec: null, // {kind,…} descriptor of what's driving the board (for a live playlist)
  entryFloor: null, // taste floor when the source was entered (Refresh restores it, leaving a pivot)
  pivotApplyFloor: null, // how the current pivot re-runs on a floor change (null = floor N/A here)
  muted: false, // master mute; when off, hover-to-hear plays the tile under the cursor
  volume: 1, // volume for whichever single tile is audible
  profile: "", // active taste profile (from ?profile=…) — scopes the For You board + saves
};
try { State.profile = new URLSearchParams(location.search).get("profile") || ""; } catch { /* ignore */ }
// query fragment for the active profile (empty = server default)
function profileParam() { return State.profile ? "&profile=" + encodeURIComponent(State.profile) : ""; }
// same, as the apex-marker `tag` (saved moments are filed under the profile's tag)
function tagParam() { return State.profile ? "&tag=" + encodeURIComponent(State.profile) : ""; }
try {
  const mv = JSON.parse(localStorage.getItem("mb_audio") || "null");
  if (mv) { State.muted = mv.muted === true; State.volume = isFinite(mv.volume) ? mv.volume : 1; }
} catch { /* ignore */ }
function persistAudio() {
  try { localStorage.setItem("mb_audio", JSON.stringify({ muted: State.muted, volume: State.volume })); }
  catch { /* ignore */ }
}
// the tile keyboard actions target: the enlarged tile, else whatever's hovered
function focusedTile() {
  const t = State.big || State.hover;
  return t && t.apex && t.apex.scene_id ? t : null;
}
// Grid audio is single-source: the ONE tile that's audible is the one under the
// cursor (unless something is enlarged, which owns audio itself, or master mute).
// Moving off all tiles (overTile null) silences everything — instantly.
function applyGridAudio() {
  // the menu's tile keeps audio while its right-click menu is open (the cursor is
  // over the menu, not the tile, so overTile goes null) — fall back to it.
  const audible = State.overTile || State.menuTile;
  for (const t of State.tiles) {
    if (t === State.big || !t.video) continue;   // enlarged tile manages its own audio
    const on = !State.muted && !State.big && t === audible;
    t.video.muted = !on;
    if (on) t.video.volume = State.volume;
  }
}
// apply master mute/volume to the enlarged tile + toolbar UI, then refresh the grid
function applyAudio() {
  const t = State.big;
  if (t && t.video) { t.video.muted = State.muted; t.video.volume = State.volume; }
  const btn = document.getElementById("mute");
  if (btn) btn.textContent = State.muted ? "🔇 Muted" : "🔊 Sound";
  const vol = document.getElementById("volume");
  if (vol && +vol.value !== State.volume) vol.value = State.volume;
  applyGridAudio();
}

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
  tile.extended = false;          // a fresh cycling clip is never pinned
  v.loop = false;
  tile.el.classList.remove("extended");
  tile.apex = apex;
  tile.mode = "offset"; // re-detected per stream on loadedmetadata
  tile.label.textContent = `#${apex.scene_id} · ${fmt(apex.start)} (${apex.duration.toFixed(0)}s)`;
  v.loop = false;
  v.src = apex.url;
  v.muted = true;
  v.load();
  if (State.playing) v.play().catch(() => {});
  if (!State.big) applyGridAudio();   // keep the hovered tile audible across clip changes
}

function advance(tile) {
  if (!State.playing || State.big === tile) return; // enlarged tile plays free
  const now = performance.now();
  if (tile.lastAdvance && now - tile.lastAdvance < 500) return;
  tile.lastAdvance = now;
  loadApex(tile);
}

// pin a good clip: stop cycling and let the scene play out (looping indefinitely)
function extendTile(tile) {
  if (!tile.apex) return;
  const v = tile.video;
  tile.extended = true;
  tile.el.classList.add("extended");
  v.loop = true; // play to the end of the scene, then repeat — never auto-advance
  if (tile.mode !== "absolute") {
    // offset = short-clip stream with nothing past the moment; reload the full scene
    // so it can play forward. The loadedmetadata handler seeks back to apex.start.
    v.src = sceneStreamUrl(tile.apex.url);
    v.load();
  }
  if (State.playing) v.play().catch(() => {});
}

// release the pin and resume normal cycling with a fresh apex
function unextendTile(tile) {
  tile.extended = false;
  tile.el.classList.remove("extended");
  tile.video.loop = false;
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
  const tile = { el, video, label, index, apex: null, mode: "offset", lastAdvance: 0, extended: false };

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
    if (!tile.apex || State.big === tile || tile.extended) return; // extended: play on, never cycle
    const end = tile.mode === "absolute" ? tile.apex.end : tile.apex.duration;
    if (video.currentTime >= end - 0.25) advance(tile);
  });
  video.addEventListener("ended", () => { if (!tile.extended) advance(tile); });
  video.addEventListener("error", () => {
    if (State.big !== tile) setTimeout(() => advance(tile), 500);
  });

  el.addEventListener("click", (e) => {
    if (e.target.closest(".mb-controls, .mb-meta, .mb-menu")) return; // interacting, not toggling
    if (State.big === tile) return collapse(tile);
    if (State.big) collapse(State.big);
    expand(tile);
  });
  el.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    if (tile.apex && tile.apex.scene_id) openTileMenu(tile, e.clientX, e.clientY);
  });
  el.addEventListener("mouseenter", () => { State.hover = tile; State.overTile = tile; applyGridAudio(); });
  el.addEventListener("mouseleave", () => { if (State.overTile === tile) State.overTile = null; applyGridAudio(); });
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
  applyGridAudio();   // enlarged tile owns audio now — silence the rest of the grid
  buildBigUI(tile);

  const v = tile.video;
  const apex = tile.apex;
  v.loop = false;
  v.muted = State.muted; v.volume = State.volume;   // enlarged tile carries the global audio
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
  applyGridAudio(); // hover-to-hear resumes on the grid
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
  v.muted = State.muted; v.volume = State.volume;   // enlarged tile carries the global audio
  controls.querySelector(".mb-play").addEventListener("click", () => {
    if (v.paused) v.play().catch(() => {}); else v.pause();
  });
  v.addEventListener("play", () => (controls.querySelector(".mb-play").textContent = "❚❚"));
  v.addEventListener("pause", () => (controls.querySelector(".mb-play").textContent = "▶"));
  const mute = controls.querySelector(".mb-mute");
  mute.addEventListener("click", () => { State.muted = !State.muted; persistAudio(); applyAudio(); mute.textContent = State.muted ? "🔇" : "🔊"; });
  mute.textContent = State.muted ? "🔇" : "🔊";
  applyAudio();
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
    tick.title = `moment @ ${fmt(a.start)}`;
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
  if (State.source === "foryou") fyRefetch(false);   // pull a genuinely fresh batch
  State.tiles.forEach((t, i) => setTimeout(() => loadApex(t), i * STAGGER_MS));
}

const STEER = "right-click, or hover + ↑/↓ to rate · S save · Space/M/R";
function updateStatus() {
  const pivot = State.pivot ? `${State.pivot} · ` : "";
  if (State.pivot) {   // a "more like this"/performer pivot — show ITS count, not For You coverage
    // the label already carries any ≥NN% floor; only add the "widens/tightens" hint
    // where the floor actually applies (moment pivot or a floor-aware pivot).
    const hint = (State.pivotSeed || State.pivotApplyFloor) ? " · floor widens/tightens" : "";
    document.getElementById("status").textContent =
      `${pivot}${State.apexes.length} moments · ${State.tiles.length} tiles${hint} · ${STEER}`;
    return;
  }
  if (State.source === "foryou" && fyTotals) {
    // separate "matches your taste" (the whole board) from "loaded so far".
    const by = fyTotals.scored_by === "classifier" ? "your trained model"
      : fyTotals.scored_by === "modes" ? "your taste modes (nearest of your interests)"
      : fyTotals.scored_by === "centroid" ? "taste centroid (no trained model yet)" : "your taste";
    const floor = fyMinScore > 0 ? ` ≥ ${Math.round(fyMinScore * 100)}%` : "";
    document.getElementById("status").textContent =
      `${pivot}Scored by ${by}${floor} · ≈${fmtN(fyTotals.scenes)} scenes / ${fmtN(fyTotals.moments)} moments match`
      + ` · ${State.apexes.length} loaded · ${State.tiles.length} tiles · ${STEER}`;
    return;
  }
  const label = State.shuffle
    ? `shuffle · ${(State.pool || []).length} scenes`
    : `${State.apexes.length} ${State.searchMode ? "moments" : "saved moments"}`;
  document.getElementById("status").textContent =
    `${pivot}${label} · ${State.tiles.length} tiles · ${STEER}`;
}
function fmtN(n) { return (n ?? 0).toLocaleString(); }

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
  const floor = document.getElementById("floor");
  if (floor) {
    floor.addEventListener("input", () => {
      fyMinScore = parseFloat(floor.value) || 0;
      const v = document.getElementById("floor-val");
      if (v) v.textContent = fyMinScore > 0 ? Math.round(fyMinScore * 100) + "%" : "off";
    });
    floor.addEventListener("change", () => {   // commit → re-apply the floor to what's driving the board
      persistFloor(fyMinScore);
      applyFloor();
    });
  }
  document.getElementById("refresh-lib").addEventListener("click", () => {
    const src = State.source || document.getElementById("source").value;
    // From a pivot, Refresh returns to the board you entered with — restore the
    // entry-time floor too, then rebuild that source (also picks up new scenes).
    if ((State.pivot || State.pivotSeed) && State.entryFloor != null && State.entryFloor !== fyMinScore) {
      fyMinScore = State.entryFloor;
      syncFloorUI(); persistFloor(fyMinScore);
    }
    loadSource(src, { refresh: true });   // exits any pivot → rebuild the entering source
  });
  document.getElementById("save-board")?.addEventListener("click", saveCurrentBoard);
  document.getElementById("mute").addEventListener("click", toggleMute);
  const vol = document.getElementById("volume");
  if (vol) {
    vol.value = State.volume;
    vol.addEventListener("input", () => {
      State.volume = parseFloat(vol.value);
      if (State.volume > 0 && State.muted) State.muted = false;   // nudging volume un-mutes
      persistAudio(); applyAudio();
      const mb = State.big?.ui?.controls.querySelector(".mb-mute");
      if (mb) mb.textContent = State.muted ? "🔇" : "🔊";
    });
  }
  applyAudio();
  document.addEventListener("keydown", onKey);
}

function toggleMute() {
  State.muted = !State.muted; persistAudio(); applyAudio();
  const mb = State.big?.ui?.controls.querySelector(".mb-mute");
  if (mb) mb.textContent = State.muted ? "🔇" : "🔊";
}
function nudgeFloor(delta) {
  const s = document.getElementById("floor");
  const max = s ? parseFloat(s.max) : 0.95;
  fyMinScore = Math.min(max, Math.max(0, +(fyMinScore + delta).toFixed(2)));
  syncFloorUI(); persistFloor(fyMinScore); applyFloor();
}
// board keyboard: drive and train without leaving the grid
function onKey(e) {
  if (e.key === "Escape") { closeTileMenu(); if (State.big) collapse(State.big); return; }
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select" || tag === "button") return;  // let fields/buttons handle their own keys
  const ft = focusedTile();
  switch (e.key) {
    case " ": e.preventDefault(); setPlaying(!State.playing); break;
    case "m": toggleMute(); break;
    case "r": reshuffle(); break;
    case "s": if (ft) saveApex(ft.apex.scene_id, tileTime(ft)); break;
    case "ArrowUp": if (ft) { e.preventDefault(); rateMoment(ft.apex.scene_id, tileTime(ft), 1); } break;
    case "ArrowDown": if (ft) { e.preventDefault(); rateMoment(ft.apex.scene_id, tileTime(ft), 0); } break;
    case "[": nudgeFloor(-0.05); break;
    case "]": nudgeFloor(0.05); break;
    default: break;
  }
}
function tileTime(t) {
  return (t.video && isFinite(t.video.currentTime)) ? t.video.currentTime : (t.apex?.start || 0);
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

// --- For You: an endless, non-repeating taste channel -----------------------
// Instead of the old ~80-item feed + memoryless weighted picker (which starved
// the tail and looped in ~1-2 min), pull a big varied pool from the API, play
// each moment once (weighted shuffle) before repeating, and refetch a fresh
// batch — excluding scenes already shown — when the queue runs low. Endless.
const FY_CLIP = 20;                      // clip seconds per For You tile
const PIVOT_PER_SCENE = 100;             // performer board: moments sampled per scene, well-spaced
const PIVOT_COUNT = 6000;                // performer board: max moments pulled (tab "Board" + pivot)
const fyState = { exclude: new Set(), queue: [], recent: [], refetching: false };
let fyMinScore = 0;                        // the "taste floor" tightener (0 = off → every scene's peak)
let fyTotals = null;                       // {scenes, moments, scored_by} for the current floor
let boardSearch = null;                   // {q,min,per,neg,taste} when replaying a search
let boardPerformer = null;                // {id,name,query} when playing a performer's best
let boardStat = null;                     // {metric,id,label} when playing a Statistics-tab stat
function fyReset() { fyState.exclude.clear(); fyState.queue = []; fyState.recent = []; fyState.refetching = false; }
function fyBoardUrl(withExclude) {
  const ex = withExclude ? [...fyState.exclude].join(",") : "";
  return "/api/foryou/board?count=400"
    + (ex ? "&exclude=" + encodeURIComponent(ex) : "")
    + "&min_score=" + fyMinScore    // always sent, so 0 truly means "off"
    + profileParam();
}
function syncFloorUI() {
  const s = document.getElementById("floor"), v = document.getElementById("floor-val");
  if (s) s.value = fyMinScore;
  if (v) v.textContent = fyMinScore > 0 ? Math.round(fyMinScore * 100) + "%" : "off";
}
// the taste floor is shared with the app (Taste Metrics slider) via localStorage,
// so it stays in sync across the tab and the board.
const FLOOR_KEY = "peaks_taste_floor";
function persistFloor(v) { try { localStorage.setItem(FLOOR_KEY, String(v)); } catch { /* ignore */ } }
function readPersistedFloor() {
  try { const v = parseFloat(localStorage.getItem(FLOOR_KEY)); return isFinite(v) ? v : null; } catch { return null; }
}
// re-apply the current floor to whatever is driving the board: a moment pivot
// re-filters that moment; For You rebuilds its pool; static sources ignore it.
function applyFloor() {
  // a seeded moment pivot re-filters THAT moment
  if (State.pivotSeed) { moreLikeThis(State.pivotSeed.scene_id, State.pivotSeed.t); return; }
  // any other pivot re-runs only how it registered — NEVER fall through to a source
  // rebuild, which would swap the board out from under the pivot (the taste-floor bug).
  if (State.pivot) { if (State.pivotApplyFloor) State.pivotApplyFloor(); return; }
  if (State.source === "performer") { loadSource("performer"); return; }
  if (State.source === "foryou") {
    fyReset();
    fyRefetch(true).then(() => { updateStatus(); reshuffle(); });
    return;
  }
  // static sources (shuffle / collection / search / galaxy / tag:apex): floor is a no-op (and hidden)
}

function fyHitToApex(h) {
  const start = +h.time || 0;
  return {
    scene_id: h.scene_id, start: Math.round(start), end: Math.round(start + FY_CLIP),
    duration: FY_CLIP, url: h.stream, score: h.score ?? 1, title: h.title || "",
  };
}
// order key = random^(1/score): higher-taste moments tend to come earlier, but
// every item still appears exactly once before the queue is exhausted.
// play-each-once picker over a STATIC pool (collections, search results): every
// clip plays before any repeats, so a big saved set actually shows all of it.
function makeQueuePicker(apexes) {
  let queue = fyWeightedShuffle(apexes);
  const recent = [];
  return function () {
    if (!queue.length) queue = fyWeightedShuffle(apexes);   // exhausted → reshuffle
    for (let i = 0; i < queue.length && i < 6; i++) {
      if (!recent.includes(String(queue[i].scene_id))) {
        const c = queue.splice(i, 1)[0];
        recent.push(String(c.scene_id)); if (recent.length > 8) recent.shift();
        return c;
      }
    }
    return queue.shift();
  };
}
function fyWeightedShuffle(items) {
  return items
    .map((a) => [a, Math.pow(Math.random(), 1 / Math.max(a.score ?? 1, 0.05))])
    .sort((x, y) => y[1] - x[1])
    .map((o) => o[0]);
}
// For You ordering: with a taste floor set, the floor is the quality gate → order
// uniformly at random (no score bias) so the feed is truly random within it. With
// the floor off, keep the score-weighted shuffle that surfaces your strongest first.
function fyShuffle(items) {
  if (fyMinScore > 0) {
    const a = items.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  return fyWeightedShuffle(items);
}
async function fyRefetch(reset) {
  if (fyState.refetching) return;
  fyState.refetching = true;
  try {
    let d;
    try { d = await api(fyBoardUrl(true)); }
    catch { d = { items: [] }; }
    let items = (d.items || []).filter((h) => h.scene_id && h.stream).map(fyHitToApex);
    if (!items.length && fyState.exclude.size) {   // whole library cycled — start fresh
      fyState.exclude.clear();
      try { d = await api(fyBoardUrl(false)); } catch { d = { items: [] }; }
      items = (d.items || []).filter((h) => h.scene_id && h.stream).map(fyHitToApex);
      reset = true;
    }
    if (d.scenes != null) fyTotals = { scenes: d.scenes, moments: d.moments, scored_by: d.scored_by };
    if (reset) { State.apexes = items; fyState.queue = []; }
    else State.apexes = State.apexes.concat(items);
    for (const a of items) fyState.exclude.add(String(a.scene_id));
    fyState.queue = fyState.queue.concat(fyShuffle(items));
    updateStatus();
  } finally { fyState.refetching = false; }
}
function fyPick() {
  if (fyState.queue.length < 6 && !fyState.refetching) fyRefetch(false);   // prefetch ahead
  if (!fyState.queue.length) {
    if (!State.apexes.length) return null;
    fyState.queue = fyShuffle(State.apexes);   // nothing fresh yet: recycle the pool
  }
  // don't show the same scene as the last few picks (across tiles)
  for (let i = 0; i < fyState.queue.length && i < 6; i++) {
    if (!fyState.recent.includes(String(fyState.queue[i].scene_id))) {
      const cand = fyState.queue.splice(i, 1)[0];
      fyState.recent.push(String(cand.scene_id));
      if (fyState.recent.length > 8) fyState.recent.shift();
      return cand;
    }
  }
  return fyState.queue.shift();   // all recent: take the head anyway
}

// --- in-the-moment right-click menu: drive the board without leaving ---------
function closeTileMenu() {
  document.querySelectorAll(".mb-menu").forEach((m) => m.remove());
  document.removeEventListener("click", closeTileMenu, true);
  if (State.menuTile) { State.menuTile = null; applyGridAudio(); }   // resume normal hover-to-hear
}
function openTileMenu(tile, x, y) {
  closeTileMenu();
  State.menuTile = tile;   // keep this tile audible while its menu is up
  const apex = tile.apex;
  const t = (tile.video && isFinite(tile.video.currentTime)) ? tile.video.currentTime : apex.start;
  // deep-link out to the real Stash scene page, seeked to this moment (?t=seconds)
  let stashHref = null;
  try { stashHref = new URL(apex.url, location.href).origin + "/scenes/" + apex.scene_id + "?t=" + Math.round(t); }
  catch { stashHref = null; }
  // "Save moment" flips to "Unsave moment" once we confirm this moment is already
  // saved (checked async below, anchored on the moment's canonical start).
  const apexState = { is: false };
  const items = [
    // taste actions
    ["👍 More of this (my taste)", () => rateMoment(apex.scene_id, t, 1)],
    ["👎 Less of this", () => rateMoment(apex.scene_id, t, 0)],
    ["★ Save moment", () => apexState.is ? removeApex(apex.scene_id, apex.start) : saveApex(apex.scene_id, t), "apex"],
    // explore more — narrowest to broadest scope
    ["🎞 More moments in this scene", () => moreInThisScene(apex.scene_id)],
    ["🔎 More like this moment", () => moreLikeThis(apex.scene_id, t)],
    ["🎬 More of this moment (same actress)", () => moreMomentSameActress(apex.scene_id, t)],
    ["🎬 More from this actress", () => moreFromActress(apex.scene_id)],
  ];
  if (tile.extended) items.push(["⏹ Stop extending (resume cycling)", () => unextendTile(tile)]);
  else items.push(["⏱ Extend (keep this scene playing)", () => extendTile(tile)]);
  if (stashHref) items.push(["📺 Open in Stash (this moment)", () => window.open(stashHref, "_blank", "noopener")]);
  items.push(
    ["⭐ Save her best as a playlist", () => saveHerBest(apex.scene_id)],
    ["💾 Save this board as a playlist", () => saveCurrentBoard()],
  );
  if (State.sourceSpec) items.push(["🔴 Save as LIVE playlist (re-derives each open)", () => saveLivePlaylist()]);
  if (State.pivot) items.push(["← Back to my taste", () => backToTaste()]);
  // always last: flag the whole scene for later cleanup by 1-starring it in Stash
  items.push(["🗑 Mark for deletion (1★ in Stash)", () => markForDeletion(apex.scene_id)]);

  const menu = document.createElement("div");
  menu.className = "mb-menu";
  items.forEach(([label, fn, role]) => {
    const b = document.createElement("button");
    b.textContent = label;
    if (role) b.dataset.role = role;
    b.addEventListener("click", (e) => { e.stopPropagation(); closeTileMenu(); fn(); });
    menu.appendChild(b);
  });
  document.body.appendChild(menu);
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  menu.style.left = Math.min(x, innerWidth - mw - 8) + "px";
  menu.style.top = Math.min(y, innerHeight - mh - 8) + "px";
  applyGridAudio();   // keep this tile audible now, before mouseleave nulls overTile
  setTimeout(() => document.addEventListener("click", closeTileMenu, true), 0);
  // confirm apex status out-of-band; flip the toggle to "Remove" only if it is one
  checkApex(apex.scene_id, apex.start).then((is) => {
    if (!is || !menu.isConnected) return;
    apexState.is = true;
    const b = menu.querySelector('[data-role="apex"]');
    if (b) b.textContent = "✖ Unsave moment";
  });
}

function apexFromHit(h) {
  const start = +h.time || 0;
  return {
    scene_id: h.scene_id, start: Math.round(start), end: Math.round(start + FY_CLIP),
    duration: FY_CLIP, url: h.stream, score: h.score ?? 1, title: h.title || "",
  };
}
// swap the whole board to a new pool live — no reload (mirrors fyRefetch)
function pivotBoard(apexes, label) {
  if (!apexes.length) { flashStatus("nothing to show"); return; }
  if (State.big) collapse(State.big);
  State.apexes = apexes; State.searchMode = true; State.pivot = label;
  pickApex = makePicker(State.apexes);
  reshuffle(); updateStatus();
}
async function moreLikeThis(scene_id, t) {
  // remember the seed so the taste-floor slider and Refresh re-filter THIS moment,
  // not the original board. With a floor set, return every match above it (the
  // count grows/shrinks with the slider); with it off, a generous default.
  State.pivotSeed = { scene_id, t };
  State.sourceSpec = { kind: "similar", scene_id, t };
  State.pivotApplyFloor = null;   // pivotSeed drives the re-run for this pivot
  syncFloorVisibility();
  flashStatus("finding similar…");
  try {
    const qs = new URLSearchParams({ scene_id, t });
    if (fyMinScore > 0) qs.set("min_score", fyMinScore);
    else qs.set("top_k", 300);
    const d = await api("/api/search/similar?" + qs.toString());
    const items = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
    const flr = fyMinScore > 0 ? ` ≥ ${Math.round(fyMinScore * 100)}%` : "";
    pivotBoard(items, `🔎 more like that moment${flr}`);
    flashStatus(`🔎 ${items.length} moments like that${flr}`);
  } catch (e) { flashStatus(e.message); }
}
async function saveHerBest(scene_id) {
  flashStatus("finding her best…");
  let d;
  try { d = await api(`/api/performer/best?scene_id=${encodeURIComponent(scene_id)}&count=500`); }
  catch (e) { return flashStatus(e.message); }
  const apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
  if (!apexes.length) return flashStatus(d.performer ? `no embedded moments for ${d.performer}` : "no performer here");
  const name = prompt(`Save ${d.performer || "her"}'s best as a playlist:`, (d.performer || "performer") + " — best of");
  if (!name) return;
  try {
    await api("/api/collection", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, apexes }) });
    flashStatus(`saved "${name}" (${apexes.length})`);
  } catch (e) { flashStatus(e.message); }
}
// Train from the board: shape your taste (👍/👎) or bank a moment (★ apex) right
// where you're watching, using the same endpoints Explore/the viewer use.
async function rateMoment(scene_id, t, label) {
  try {
    await api(`/api/label?scene_id=${encodeURIComponent(scene_id)}&t=${(+t || 0).toFixed(2)}&label=${label}` + profileParam(),
      { method: "POST" });
    flashStatus(label ? "👍 added to your taste" : "👎 noted — less like that");
  } catch (e) { flashStatus(e.message); }
}
async function saveApex(scene_id, t) {
  try {
    await api(`/api/scene/${encodeURIComponent(scene_id)}/apex?t=${(+t || 0).toFixed(2)}` + tagParam(), { method: "POST" });
    flashStatus("★ saved moment @ " + fmt(t));
  } catch (e) { flashStatus(e.message); }
}
// does an apex marker sit at this moment? (drives the menu's Save/Remove toggle)
async function checkApex(scene_id, t) {
  try {
    const d = await api(`/api/scene/${encodeURIComponent(scene_id)}/apex?t=${(+t || 0).toFixed(2)}` + tagParam());
    return !!(d && d.is_apex);
  } catch { return false; }
}
async function removeApex(scene_id, t) {
  try {
    await api(`/api/scene/${encodeURIComponent(scene_id)}/apex?t=${(+t || 0).toFixed(2)}` + tagParam(), { method: "DELETE" });
    flashStatus("✖ unsaved moment @ " + fmt(t));
  } catch (e) { flashStatus(e.message); }
}
// flag a scene for later cleanup: set its Stash rating to 1★ (rating100 = 20)
async function markForDeletion(scene_id) {
  try {
    await api(`/api/scene/${encodeURIComponent(scene_id)}`, {
      method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ rating100: 20 }),
    });
    flashStatus("🗑 marked for deletion (1★) · #" + scene_id);
  } catch (e) { flashStatus(e.message); }
}
// Save whatever is currently driving the board — a "more like this" rabbit hole,
// a performer pivot, a search, For You, anything — as a collection you can replay.
async function saveCurrentBoard() {
  const apexes = (State.apexes || []).filter((a) => a.scene_id && (a.url || a.stream));
  if (!apexes.length) return flashStatus("nothing on the board to save yet");
  const base = State.pivot ? State.pivot.replace(/^[^A-Za-z0-9]+/, "").trim() : "megaboard";
  const name = prompt(`Save these ${apexes.length} moments as a playlist:`, base + " — playlist");
  if (!name) return;
  try {
    await api("/api/collection", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, apexes }),
    });
    flashStatus(`saved "${name}" (${apexes.length} moments)`);
  } catch (e) { flashStatus(e.message); }
}
// Save the board's SOURCE (not a frozen snapshot): a live playlist re-derives its
// moments from your current taste + library every time you open it.
async function saveLivePlaylist() {
  if (!State.sourceSpec) return flashStatus("this board can't be saved live");
  const apexes = (State.apexes || []).filter((a) => a.scene_id && (a.url || a.stream));
  const base = State.pivot ? State.pivot.replace(/^[^A-Za-z0-9]+/, "").trim() : "live";
  const name = prompt("Save as a LIVE playlist (re-derives each open):", base + " — live");
  if (!name) return;
  try {
    await api("/api/collection", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, apexes, live: true, source: State.sourceSpec }),
    });
    flashStatus(`🔴 saved live playlist "${name}"`);
  } catch (e) { flashStatus(e.message); }
}
// Broad by default: her FULL spread across every embedded scene, floor OFF —
// pivoting shouldn't confine you to the curated taste feed. The floor slider stays
// available as an optional tightener (tighten=true, via State.pivotApplyFloor).
async function moreFromActress(scene_id, tighten = false) {
  State.pivotSeed = null;   // performer pivot isn't seeded by a single moment
  State.sourceSpec = { kind: "performer", scene_id };
  State.pivotApplyFloor = () => moreFromActress(scene_id, true);
  syncFloorVisibility();
  flashStatus("finding her scenes…");
  try {
    const useFloor = tighten && fyMinScore > 0;   // only tighten when you nudge the slider
    const qs = new URLSearchParams({ scene_id, count: PIVOT_COUNT, per_scene: PIVOT_PER_SCENE });
    if (useFloor) qs.set("min_score", fyMinScore);
    else qs.set("spread", "true");   // broad, unfiltered span of her scenes
    const d = await api("/api/performer/best?" + qs.toString());
    const apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
    if (!apexes.length) return flashStatus(d.performer ? `no embedded moments for ${d.performer}` : "no performer info for this scene");
    const tag = useFloor ? ` ≥ ${Math.round(fyMinScore * 100)}%` : " · full spread";
    pivotBoard(apexes, "🎬 " + (d.performer || "this actress") + tag);
    flashStatus(`🎬 ${apexes.length} moments from ${d.performer || "her"}${useFloor ? ` ≥ ${Math.round(fyMinScore * 100)}%` : " · full spread"}`);
  } catch (e) { flashStatus(e.message); }
}
// similar moments to THIS one, but only across this scene's performer
async function moreMomentSameActress(scene_id, t) {
  State.pivotSeed = null;
  State.sourceSpec = { kind: "performer_moment", scene_id, t };
  State.pivotApplyFloor = null;   // diverse coverage (score 0) — floor N/A, hidden
  syncFloorVisibility();
  flashStatus("finding her moments like this…");
  try {
    const qs = new URLSearchParams({ scene_id, t });
    const d = await api("/api/board/performer_moment?" + qs.toString());
    const apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
    if (!apexes.length) return flashStatus(d.performer ? `no similar moments for ${d.performer}` : "no performer info for this scene");
    pivotBoard(apexes, "🎬 more like this · " + (d.performer || "same actress"));
  } catch (e) { flashStatus(e.message); }
}
// stay in this scene: a diverse spread of its moments
async function moreInThisScene(scene_id) {
  State.pivotSeed = null;
  State.sourceSpec = { kind: "scene", scene_id };
  State.pivotApplyFloor = null;   // single-scene spread (score 0) — floor N/A, hidden
  syncFloorVisibility();
  flashStatus("gathering this scene…");
  try {
    const d = await api(`/api/scene/${encodeURIComponent(scene_id)}/moments`);
    const apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
    if (!apexes.length) return flashStatus("no other embedded moments in this scene");
    pivotBoard(apexes, "🎞 more in scene #" + scene_id);
  } catch (e) { flashStatus(e.message); }
}
function backToTaste() { loadSource("foryou"); }
function flashStatus(msg) {
  const el = document.getElementById("status");
  if (el) el.textContent = msg;
}

function syncFloorVisibility() {
  const fw = document.getElementById("floor-wrap");
  // Key on the PIVOT first — State.source is stale during a pivotBoard pivot. Show the
  // floor only where it means something: a moment pivot (closeness), a floor-aware pivot
  // (moreFromActress), or the taste sources For You / Performer. Hide it for the
  // diverse-coverage pivots and all static sources.
  const show = State.pivotSeed ? true
    : State.pivot ? (State.pivotApplyFloor != null)
    : (State.source === "foryou" || State.source === "performer");
  if (fw) fw.hidden = !show;
}
// the re-derivable descriptor for a live playlist (null = this board can't go live)
function specForSource(src) {
  if (src === "performer") return { kind: "performer", id: boardPerformer?.id || null, name: boardPerformer?.name || null, query: boardPerformer?.query || null };
  if (src === "search" && boardSearch) return { kind: "search", q: boardSearch.q, min: boardSearch.min, per: boardSearch.per, neg: boardSearch.neg, taste: boardSearch.taste };
  if (src === "foryou") return { kind: "foryou" };
  if (src === "stat") return { kind: "stat", metric: boardStat?.metric || "fresh", id: boardStat?.id || null };
  return null;  // shuffle / tag / collection / galaxy — not live-derivable
}
async function loadSource(src, opts = {}) {
  State.source = src;
  State.sourceSpec = specForSource(src);   // pivots override this below
  State.entryFloor = fyMinScore;   // remember the floor at entry; pivots never re-run this, so Refresh can restore it
  State.shuffle = false; State.searchMode = false; State.pool = null; State.apexes = [];
  State.pivot = null; State.pivotSeed = null; State.pivotApplyFloor = null;
  document.getElementById("error").hidden = true;
  syncFloorVisibility();
  document.getElementById("status").textContent = "loading…";
  try {
    if (src === "shuffle") {
      const d = await api("/api/board/scenes" + (opts.refresh ? "?refresh=true" : ""));
      State.pool = d.scenes || [];
      if (!State.pool.length) return showError("No scenes found in your library scope.");
      State.shuffle = true;
      pickApex = randomMoment;
    } else if (src.startsWith("collection:")) {
      const safe = src.slice(11);
      const pl = await api("/api/collection?name=" + encodeURIComponent(safe));
      let apexes = pl.apexes || [];
      if (pl.live) {   // re-derive from the saved source spec, not the frozen snapshot
        try {
          const d = await api("/api/collection/derive?name=" + encodeURIComponent(safe));
          const fresh = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
          if (fresh.length) apexes = fresh;
        } catch (e) { /* fall back to the stored snapshot */ }
      }
      State.apexes = apexes; State.searchMode = true;
      if (!State.apexes.length) return showError("This playlist has no moments.");
      pickApex = makeQueuePicker(State.apexes);   // play every clip before repeating
    } else if (src === "search") {
      State.searchMode = true;
      if (boardSearch) {
        // re-fetch the FULL filtered set from the API — no localStorage size cap
        const qs = new URLSearchParams({
          q: boardSearch.q, per_scene: boardSearch.per ?? 3,
          neg_weight: boardSearch.neg ?? 0.5, enrich: 0,
        });
        if (boardSearch.min > 0) qs.set("min_score", boardSearch.min);
        if (boardSearch.taste) qs.set("taste", "true");
        let d;
        try { d = await api("/api/search/text?" + qs.toString()); }
        catch (e) { return showError("Search failed.\n\n(" + e.message + ")"); }
        State.apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
        if (!State.apexes.length) return showError("No matches at this match strength.");
      } else {
        let pl = null; try { pl = JSON.parse(localStorage.getItem("mb_search") || "null"); } catch {}
        State.apexes = (pl && pl.apexes) || [];
        if (!State.apexes.length) return showError("No search results to play. Run a search and hit 'Play on megaboard'.");
      }
      pickApex = makeQueuePicker(State.apexes);
    } else if (src === "performer") {
      State.searchMode = true;
      const qs = new URLSearchParams({ count: PIVOT_COUNT, per_scene: PIVOT_PER_SCENE });
      if (boardPerformer?.id) qs.set("id", boardPerformer.id);
      else if (boardPerformer?.name) qs.set("name", boardPerformer.name);
      if (boardPerformer?.query) qs.set("query", boardPerformer.query);
      // floor at 0% → a diverse span of her scenes; above 0 → her taste-matching moments
      if (fyMinScore > 0) qs.set("min_score", fyMinScore);
      else qs.set("spread", "true");
      let d;
      try { d = await api("/api/performer/best?" + qs.toString()); }
      catch (e) { return showError("Couldn't load this performer.\n\n(" + e.message + ")"); }
      State.apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
      if (!State.apexes.length) return showError("No embedded moments for this performer.");
      pickApex = makeQueuePicker(State.apexes);
      document.getElementById("status").textContent = "🎬 " + (d.performer || "performer") + " · best of";
    } else if (src === "stat") {
      State.searchMode = true;
      const qs = new URLSearchParams({ metric: boardStat?.metric || "fresh" });
      if (boardStat?.id) qs.set("id", boardStat.id);
      let d;
      try { d = await api("/api/stats/board?" + qs.toString()); }
      catch (e) { return showError("Couldn't load this statistic.\n\n(" + e.message + ")"); }
      State.apexes = (d.items || []).filter((h) => h.scene_id && h.stream).map(apexFromHit);
      if (!State.apexes.length) return showError("No moments for this statistic yet.");
      pickApex = makeQueuePicker(State.apexes);
      document.getElementById("status").textContent = "📈 " + (boardStat?.label || boardStat?.metric || "stat");
    } else if (src === "foryou") {
      State.searchMode = true;
      fyReset();
      // seed instantly from the on-screen feed if it's there, then pull the big
      // varied pool from the API for an endless, non-repeating channel.
      let seed = null; try { seed = JSON.parse(localStorage.getItem("mb_foryou") || "null"); } catch {}
      if (seed && seed.apexes && seed.apexes.length) {
        State.apexes = seed.apexes.slice();
        for (const a of State.apexes) fyState.exclude.add(String(a.scene_id));
        fyState.queue = fyShuffle(State.apexes);
      }
      // always pull a fresh batch (reset) so each open is a new random draw rather
      // than replaying the stale on-screen seed's top moments first.
      fyState.exclude.clear();
      await fyRefetch(true);
      if (!State.apexes.length) return showError("No For You taste yet. Thumb up moments or save apexes (⚑), then try again.");
      pickApex = fyPick;
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
        return showError(`No saved moments for tag "${tag}".\n\nSave some with ★ in Explore — or pick "Shuffle" above to play everything.`);
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
    opts.push(`<option value="tag:${esc(s.tag)}">Saved moments</option>`);
    for (const c of s.collections || [])
      opts.push(`<option value="collection:${esc(c.safe)}">Playlist: ${esc(c.name)} (${c.count})</option>`);
  } catch {}
  if (initial === "search") opts.push(`<option value="search">Search results</option>`);
  if (initial === "performer") opts.push(`<option value="performer">Performer: ${esc(boardPerformer?.name || "best of")}</option>`);
  if (initial === "stat") opts.push(`<option value="stat">Stat: ${esc(boardStat?.label || "statistic")}</option>`);
  // For You is always available — you can pivot to your taste from any source.
  opts.push(`<option value="foryou">For You — your taste</option>`);
  if (initial === "galaxy") opts.push(`<option value="galaxy">Galaxy selection</option>`);
  sel.innerHTML = opts.join("");
  sel.addEventListener("change", () => loadSource(sel.value));
  return sel;
}

// --- boot -----------------------------------------------------------------

async function main() {
  wireControls();
  const params = new URLSearchParams(location.search);
  const ms = parseFloat(params.get("min_score"));
  const shared = readPersistedFloor();
  // priority: explicit URL floor (from "Play on megaboard") > last shared value > off
  if (ms > 0) fyMinScore = ms;
  else if (shared != null) fyMinScore = shared;
  syncFloorUI();
  // live-sync with the app's taste slider in other tabs
  window.addEventListener("storage", (e) => {
    if (e.key !== FLOOR_KEY || e.newValue == null) return;
    const v = parseFloat(e.newValue);
    if (!isFinite(v) || v === fyMinScore) return;
    fyMinScore = v; syncFloorUI(); applyFloor();
  });
  let initial = null;
  if (params.get("src") === "search") {
    initial = "search";
    if (params.get("q")) boardSearch = {   // re-run the Explore search on the board
      q: params.get("q"),
      min: parseFloat(params.get("min_score")) || 0,
      per: parseInt(params.get("per_scene"), 10),
      neg: parseFloat(params.get("neg")),
      taste: params.get("taste") === "true",
    };
  }
  else if (params.get("src") === "performer") {
    initial = "performer";
    boardPerformer = { id: params.get("id"), name: params.get("name"), query: params.get("pq") };
  }
  else if (params.get("src") === "stat") {
    initial = "stat";
    const metric = params.get("metric") || "fresh";
    const LABELS = { fresh: "fresh peaks", actress: "top actress", most_peaks_actress: "most peaks",
      most_ontaste_actress: "most on-taste actress", most_ontaste_scene: "most on-taste scene" };
    boardStat = { metric, id: params.get("id"), label: LABELS[metric] || metric };
  }
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
