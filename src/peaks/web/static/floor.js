// The factory floor: the whole Peaks pipeline, live, drawn in plain ASCII.
// Files ride belts from the seedbox through pull -> inbox -> ingest -> embed ->
// picture -> duplicates -> the library shelves; purge, upgrades, backup, the
// report and taste work off the main line. Everything comes from /api/floor
// (real jobs and logs); crates carry a category and a percentage, never a name.
// Only ASCII characters are ever drawn: colour comes from CSS classes.
"use strict";

const FLOOR_W = 142, FLOOR_H = 40, FLOOR_FPS = 12;
const fl = { data: null, timer: null, raf: null, frame: 0, poll: null, replay: null, sparks: [], crates: [], last: 0 };

// one cell = [char, class]
function flGrid() {
  return Array.from({ length: FLOOR_H }, () => Array.from({ length: FLOOR_W }, () => [" ", ""]));
}
function flPut(g, x, y, s, cls = "") {
  if (y < 0 || y >= FLOOR_H) return;
  for (let i = 0; i < s.length; i++) {
    const xx = x + i;
    if (xx < 0 || xx >= FLOOR_W) continue;
    const ch = s.charCodeAt(i) < 32 || s.charCodeAt(i) > 126 ? "?" : s[i];   // ASCII, always
    g[y][xx] = [ch, cls];
  }
}
function flBox(g, x, y, w, h, cls) {
  flPut(g, x, y, "+" + "-".repeat(w - 2) + "+", cls);
  for (let i = 1; i < h - 1; i++) { flPut(g, x, y + i, "|", cls); flPut(g, x + w - 1, y + i, "|", cls); }
  flPut(g, x, y + h - 1, "+" + "-".repeat(w - 2) + "+", cls);
}
function flCenter(g, x, w, y, s, cls) { flPut(g, x + Math.max(0, Math.floor((w - s.length) / 2)), y, s.slice(0, w), cls); }
function flGauge(pct, w) {
  if (pct == null) return "[" + ".".repeat(w - 2) + "]";
  const n = Math.round((w - 2) * Math.max(0, Math.min(1, pct)));
  return "[" + "#".repeat(n) + "-".repeat(w - 2 - n) + "]";
}
const FL_SPIN = ["|", "/", "-", "\\"];

// the stations on the main line: x, label, width
const FL_MAIN = [
  { k: "seedbox", x: 1, w: 13, label: "SEEDBOX" },
  { k: "inbox", x: 36, w: 11, label: "INBOX" },
  { k: "ingest", x: 53, w: 13, label: "INGEST" },
  { k: "embed", x: 71, w: 13, label: "EMBED" },
  { k: "picture", x: 89, w: 13, label: "PICTURE" },
  { k: "dupes", x: 107, w: 13, label: "DUPES" },
  { k: "library", x: 125, w: 16, label: "LIBRARY" },
];
const FL_SIDE = [
  { k: "purge", x: 1, w: 15, label: "PURGE" },
  { k: "upgrades", x: 36, w: 17, label: "UPGRADES" },
  { k: "backup", x: 62, w: 15, label: "BACKUP" },
  { k: "taste", x: 84, w: 15, label: "TASTE" },
  { k: "report", x: 105, w: 15, label: "REPORT" },
];
const MAIN_Y = 6, MAIN_H = 13, SIDE_Y = 22, SIDE_H = 9;

function flState(k) {
  const st = (fl.data && fl.data.stations && fl.data.stations[k]) || { state: "idle" };
  if (k === "inbox") return { state: fl.data && (fl.data.stations.pull || {}).state === "running" ? "running" : "idle" };
  if (k === "library") return { state: fl.data && Object.keys(fl.data.shelves || {}).length ? "running" : "idle" };
  return st;
}
function flCls(state) { return { running: "f-run", error: "f-err", done: "f-done", idle: "f-dim" }[state] || "f-dim"; }

// what each machine does while it works (inside its box)
function flMachine(g, s, x, y, w, h, st, f) {
  const cx = x + Math.floor(w / 2), on = st.state === "running", cls = on ? "f-run" : "f-dim";
  const mid = y + Math.floor(h / 2);
  switch (s.k) {
    case "seedbox": {
      for (let i = 0; i < 6; i++) {
        const blink = on && ((f + i * 3) % 7 < 3);
        flPut(g, x + 2, y + 2 + i, "[" + (blink ? ":#:#:" : "#####") + "]", blink ? "f-hot" : cls);
        flPut(g, x + 10, y + 2 + i, blink ? "*" : ".", blink ? "f-hot" : "f-dim");
      }
      break;
    }
    case "inbox":
      flPut(g, x + 1, y + 2, "\\" + " ".repeat(w - 4) + "/", cls);
      flPut(g, x + 2, y + 3, "\\" + " ".repeat(w - 6) + "/", cls);
      flPut(g, x + 3, y + 4, "\\" + (on ? "o".repeat(Math.max(1, (f >> 1) % (w - 8))) : "").padEnd(w - 8) + "/", on ? "f-crate" : cls);
      flPut(g, cx - 1, y + 5, "| |", cls);
      flPut(g, cx, y + 6, on && f % 4 < 2 ? "o" : " ", "f-crate");
      break;
    case "ingest": {
      const p = on ? [0, 1, 2, 3, 2, 1][f % 6] : 0;
      flPut(g, cx - 2, y + 1, "=====", cls);
      for (let i = 0; i < p; i++) flPut(g, cx, y + 2 + i, "|", cls);
      flPut(g, cx - 2, y + 2 + p, "#####", on ? "f-hot" : cls);
      flPut(g, x + 2, y + 7, "_".repeat(w - 4), cls);
      flPut(g, x + 2, y + 8, ["SCAN", "IDENT", "TAG"][on ? Math.floor(f / 10) % 3 : 0].padEnd(w - 4), "f-dim");
      break;
    }
    case "embed": {
      const sp = on ? FL_SPIN[f % 4] : "|";
      flPut(g, cx - 3, mid - 2, " .---. ", cls);
      flPut(g, cx - 3, mid - 1, "/  " + sp + "  \\", on ? "f-hot" : cls);
      flPut(g, cx - 3, mid, "| " + sp + "O" + sp + " |", on ? "f-hot" : cls);
      flPut(g, cx - 3, mid + 1, "\\  " + sp + "  /", on ? "f-hot" : cls);
      flPut(g, cx - 3, mid + 2, " '---' ", cls);
      flPut(g, x + 2, y + 9, "GPU", "f-dim");
      if (on) flPut(g, x + 6, y + 9, "~".repeat(1 + (f % 5)), "f-hot");
      break;
    }
    case "picture": {
      const span = w - 4, bx = on ? (f % (span * 2)) : 0, pos = bx < span ? bx : span * 2 - bx - 1;
      for (let r = 0; r < 5; r++) flPut(g, x + 2, y + 2 + r, ":".repeat(span), "f-dim");
      if (on) for (let r = 0; r < 5; r++) flPut(g, x + 2 + pos, y + 2 + r, bx < span ? ">" : "<", "f-hot");
      break;
    }
    case "dupes":
      flPut(g, cx - 3, mid - 2, on && f % 6 < 3 ? "<=  =>" : " <==> ", on ? "f-hot" : cls);
      flPut(g, cx - 3, mid, "[KEEP]", cls);
      flPut(g, cx - 1, mid + 2, "\\/", cls);
      break;
    case "library": {
      const sh = (fl.data && fl.data.shelves) || {};
      const rows = [["LEG", "legendaire"], ["EXC", "exceptionnelle"], ["MER", "merveilleuse"], ["ANO", "anomaly"],
        ["NEW", "unreviewed"], ["REJ", "rejected"]];
      const max = Math.max(1, ...rows.map(([, t]) => sh[t] || 0));
      rows.forEach(([lab, t], i) => {
        // "LEG ####  1978": a 4-wide bar and the count in 5 (never cut; 100k+ as "123k")
        const n = sh[t] || 0, bar = Math.round(4 * n / max);
        const num = n > 99999 ? `${Math.floor(n / 1000)}k` : String(n);
        flPut(g, x + 1, y + 2 + i, lab + " " + "#".repeat(bar).padEnd(4, "."), t === "rejected" ? "f-err" : "f-lib");
        flPut(g, x + 9, y + 2 + i, num.padStart(5), "f-dim");
      });
      break;
    }
    case "purge": {
      const dry = fl.data && fl.data.stations.purge && fl.data.stations.purge.dry;
      const shut = on ? (f % 8 < 4) : false;
      flPut(g, x + 2, y + 2, shut ? " \\    / " : "\\      /", on ? "f-hot" : cls);
      flPut(g, x + 2, y + 3, shut ? "  \\  /  " : " \\    / ", on ? "f-hot" : cls);
      flPut(g, x + 2, y + 4, shut ? "   ><   " : "  \\  /  ", on ? "f-hot" : cls);
      if (dry) flPut(g, x + w - 6, y + 2, "DRY", "f-warn");
      break;
    }
    case "upgrades": {
      const ang = on ? f % 8 : 2;
      const arm = ["|", "/", "-", "\\", "|", "/", "-", "\\"][ang];
      const off = [[0, -2], [2, -1], [3, 0], [2, 1], [0, 2], [-2, 1], [-3, 0], [-2, -1]][ang];
      flPut(g, cx - 4, mid - 2, " .--+--. ", "f-dim");
      flPut(g, cx - 4, mid + 2, " '--+--' ", "f-dim");
      flPut(g, cx, mid, "o", on ? "f-hot" : cls);
      flPut(g, cx + off[0], mid + off[1], arm, on ? "f-hot" : cls);
      const n = (fl.data && fl.data.counts && fl.data.counts.upgrades) || 0;
      if (n) flPut(g, x + 2, y + h - 2, `FOUND ${n}`, "f-warn");
      break;
    }
    case "backup": {
      const wheel = on ? ["o", "O", "0", "O"][f % 4] : "o";
      flPut(g, cx - 3, mid - 2, "+-----+", cls);
      flPut(g, cx - 3, mid - 1, "|  " + "|" + "  |", cls);
      flPut(g, cx - 3, mid, "|--" + wheel + "--|", on ? "f-hot" : cls);
      flPut(g, cx - 3, mid + 1, "|  " + "|" + "  |", cls);
      flPut(g, cx - 3, mid + 2, "+-----+", cls);
      break;
    }
    case "taste": {
      const wave = "_.-'~'-.";
      let line = "";
      for (let i = 0; i < w - 4; i++) line += on ? wave[(i + f) % wave.length] : "_";
      flPut(g, x + 2, mid, line, on ? "f-hot" : cls);
      break;
    }
    case "report": {
      flPut(g, x + 3, y + 2, "[=======]", cls);
      const feed = on ? (f >> 1) % 4 : 0;
      for (let i = 0; i < 3; i++) flPut(g, x + 4, y + 3 + i, i < feed ? "~~~~~~~" : "       ", "f-hot");
      break;
    }
  }
}

function flStation(g, s, y, h, f) {
  const st = flState(s.k), cls = flCls(st.state);
  flBox(g, s.x, y, s.w, h, cls);
  flCenter(g, s.x, s.w, y, ` ${s.label} `, st.state === "running" ? "f-title-run" : cls === "f-err" ? "f-err" : "f-title");
  flMachine(g, s, s.x, y, s.w, h, st, f);
  if (st.state === "running" && st.pct != null) flPut(g, s.x + 1, y + h - 2, flGauge(st.pct, s.w - 2), "f-run");
  else if (st.state === "idle" && s.k !== "library") flPut(g, s.x + s.w - 4, y + h - 2, (f >> 3) % 2 ? "zz" : "z ", "f-dim");
  if (st.state === "error") for (let i = 0; i < 3; i++) {
    const sx = s.x + ((f * 7 + i * 5) % s.w), sy = y + ((f * 3 + i * 4) % h);
    flPut(g, sx, sy, ["*", "x", "+"][(f + i) % 3], "f-err");
  }
}

// a belt between two stations; moving when the work flows through it
function flBelt(g, x1, x2, y, moving, f, crates) {
  const w = x2 - x1;
  let s = "";
  for (let i = 0; i < w; i++) s += moving ? ((i + f) % 3 === 0 ? ">" : "=") : "=";
  flPut(g, x1, y, s, moving ? "f-belt" : "f-dim");
  for (const c of crates) flPut(g, x1 + Math.min(w - 4, Math.floor(c * (w - 4))), y - 1, "[##]", "f-crate");
}

function flDraw() {
  const g = flGrid(), f = fl.frame, d = fl.data || { stations: {}, lanes: [], ticker: [] };
  const clock = new Date().toTimeString().slice(0, 8);
  flPut(g, 1, 0, "PEAKS :: FACTORY FLOOR", "f-title-run");
  flPut(g, FLOOR_W - 22, 0, fl.replay ? `REPLAY ${fl.replay.clock}` : `LIVE  ${clock}`, fl.replay ? "f-warn" : "f-run");
  if (!fl.replay) flPut(g, FLOOR_W - 24, 0, f % 12 < 6 ? "*" : " ", "f-err");
  flPut(g, 1, 1, "=".repeat(FLOOR_W - 2), "f-dim");
  for (const s of FL_MAIN) flStation(g, s, MAIN_Y, MAIN_H, f);
  for (const s of FL_SIDE) flStation(g, s, SIDE_Y, SIDE_H, f);

  // pull: three lanes from the seedbox to the inbox, one crate per file in flight
  const pullOn = (d.stations.pull || {}).state === "running";
  const lanes = d.lanes || [];
  flPut(g, 16, MAIN_Y - 1, "PULL" + (pullOn ? ` ${FL_SPIN[f % 4]}` : "  "), pullOn ? "f-title-run" : "f-title");
  if ((d.stations.pull || {}).queued) flPut(g, 23, MAIN_Y - 1, `+${d.stations.pull.queued} queued`, "f-dim");
  for (let i = 0; i < 3; i++) {
    const ly = MAIN_Y + 3 + i * 3, lane = lanes[i];
    flBelt(g, 14, 36, ly, pullOn && !!lane, f, []);
    if (lane) {
      const at = 14 + Math.floor(lane.pct * 15);
      flPut(g, at, ly - 1, `[${(lane.cat || "?").slice(0, 2)}${String(Math.round(lane.pct * 100)).padStart(3)}%]`, "f-crate");
    }
  }
  // the main belts: crates move while the station they feed is working
  const beltY = MAIN_Y + Math.floor(MAIN_H / 2);
  for (let i = 1; i < FL_MAIN.length - 1; i++) {
    const a = FL_MAIN[i], b = FL_MAIN[i + 1], st = flState(b.k).state === "running";
    flBelt(g, a.x + a.w, b.x, beltY, st, f, st ? [((f % 24) / 24)] : []);
  }
  // the duplicate chute: removed copies fall into the purge-side crusher line
  const dx = FL_MAIN[5].x + 6, dupOn = flState("dupes").state === "running";
  for (let y = MAIN_Y + MAIN_H; y < SIDE_Y - 1; y++) flPut(g, dx, y, dupOn && (y + f) % 3 === 0 ? "x" : "|", dupOn ? "f-err" : "f-dim");
  flPut(g, dx - 3, SIDE_Y - 1, "[BIN]", "f-dim");
  // the purge hangs off the seedbox
  for (let y = MAIN_Y + MAIN_H; y < SIDE_Y; y++) flPut(g, 7, y, flState("purge").state === "running" && (y + f) % 2 ? ":" : "|", "f-dim");

  // ticker
  const ty = SIDE_Y + SIDE_H + 1;
  flBox(g, 0, ty, FLOOR_W, FLOOR_H - ty, "f-dim");
  flPut(g, 2, ty, " LOG ", "f-title");
  const lines = (fl.replay ? fl.replay.log : (d.ticker || []).map((t) => t.text)).slice(-(FLOOR_H - ty - 2));
  lines.forEach((t, i) => flPut(g, 2, ty + 1 + i, ("> " + t).slice(0, FLOOR_W - 4), i === lines.length - 1 ? "f-run" : "f-dim"));
  if (!lines.length) flPut(g, 2, ty + 1, "> quiet. nothing moving right now.", "f-dim");
  if (f % 12 < 6) flPut(g, 4 + Math.min(FLOOR_W - 8, ((lines[lines.length - 1] || "").length + 2)), ty + Math.max(1, lines.length), "_", "f-run");

  // render: one span per run of the same class
  const out = [];
  for (const row of g) {
    let cur = null, buf = "";
    for (const [ch, cls] of row) {
      if (cls !== cur) { if (buf) out.push(cur ? `<span class="${cur}">${esc(buf)}</span>` : esc(buf)); buf = ""; cur = cls; }
      buf += ch;
    }
    if (buf) out.push(cur ? `<span class="${cur}">${esc(buf)}</span>` : esc(buf));
    out.push("\n");
  }
  $("#floor-grid").innerHTML = out.join("");
}

function flFit() {
  const el = $("#floor-grid"); if (!el) return;
  const box = el.parentElement.getBoundingClientRect();
  const byW = (box.width - 8) / (FLOOR_W * 0.6), byH = (window.innerHeight - el.getBoundingClientRect().top - 12) / (FLOOR_H * 1.15);
  el.style.fontSize = Math.max(6, Math.min(22, Math.floor(Math.min(byW, byH) * 10) / 10)) + "px";
}
function flLoop(ts) {
  fl.raf = requestAnimationFrame(flLoop);
  if (ts - fl.last < 1000 / FLOOR_FPS) return;
  fl.last = ts; fl.frame++;
  if (fl.replay) flReplayStep();
  flDraw();
}
async function flPoll() {
  if (fl.replay) return;
  try { fl.data = await api("/api/floor"); } catch { /* keep drawing the last picture */ }
}
function openFloor() {
  const kiosk = /kiosk=1/.test(location.hash);
  document.body.classList.toggle("kiosk", kiosk);
  stopFloor(true);
  flPoll(); fl.poll = setInterval(() => { if (!document.hidden) flPoll(); }, 2000);
  flFit(); window.addEventListener("resize", flFit);
  fl.raf = requestAnimationFrame(flLoop);
}
function stopFloor(keepKiosk) {
  if (fl.raf) cancelAnimationFrame(fl.raf);
  clearInterval(fl.poll); fl.raf = null; fl.poll = null; fl.replay = null;
  window.removeEventListener("resize", flFit);
  if (!keepKiosk) document.body.classList.remove("kiosk");
}
// the last 24 hours, fast-forwarded into about a minute
async function flStartReplay() {
  let ev;
  try { ev = (await api("/api/floor/replay?hours=24")).events; } catch (e) { toast(e.message, true); return; }
  const now = Date.now() / 1000, start = now - 24 * 3600;
  fl.replay = { ev, i: 0, t: start, start, end: now, step: (24 * 3600) / (60 * FLOOR_FPS), clock: "", log: [], hot: {} };
  fl.data = { stations: {}, lanes: [], shelves: (fl.data && fl.data.shelves) || {}, counts: (fl.data && fl.data.counts) || {}, ticker: [] };
}
function flReplayStep() {
  const r = fl.replay;
  r.t += r.step;
  r.clock = new Date(r.t * 1000).toTimeString().slice(0, 5);
  while (r.i < r.ev.length && r.ev[r.i].at <= r.t) {
    const e = r.ev[r.i++];
    r.hot[e.station] = fl.frame + 18;
    r.log.push(`${new Date(e.at * 1000).toTimeString().slice(0, 5)} ${e.station}${e.n > 1 ? " x" + e.n : ""}${e.ok ? "" : " FAILED"}`);
    r.log = r.log.slice(-6);
  }
  const st = {};
  for (const k of ["pull", "purge", "ingest", "embed", "picture", "dupes", "upgrades", "backup", "taste", "report"]) {
    st[k] = { state: (r.hot[k] || 0) > fl.frame ? "running" : "idle" };
  }
  fl.data.stations = st;
  fl.data.lanes = st.pull.state === "running" ? [{ cat: "1", pct: ((fl.frame % 18) / 18) }] : [];
  if (r.t >= r.end) { fl.replay = null; flPoll(); toast("Replay done — back to live"); }
}
$("#floor-replay")?.addEventListener("click", flStartReplay);
$("#floor-kiosk")?.addEventListener("click", () => {
  location.hash = "#/floor?kiosk=1"; openFloor();
  (document.documentElement.requestFullscreen || (() => {})).call(document.documentElement);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.classList.contains("kiosk")) { document.body.classList.remove("kiosk"); flFit(); }
});

// opened straight on #/floor: app.js routed before this file loaded
if (/^#\/floor/.test(location.hash) && $("#floor")?.classList.contains("active")) openFloor();
