// The control room: Peaks' pipeline as a factory floor, drawn by FE (engine.js).
// Everything shown is real — machines run when their jobs run, crates are real
// files in flight, every number comes from /api/floor and /api/floor/day.
// Never a file name, a title or a frame of video.
"use strict";

const FW = {
  data: null, day: null, prev: null, prevDay: null, raf: 0, last: 0, poll: 0, dpoll: 0,
  crates: [], drops: [], shake: 0, lastActive: Date.now(), replay: null, typed: { text: "", shown: 0 },
  fps: 30, frames: 0, fpsAt: 0, measured: 0, forklift: { x: 30, dir: 1 },
};

// --- layout (world cells) -----------------------------------------------------------------
const MAIN_Y = 8, MAIN_H = 12, LOW_Y = 31, LOW_H = 10;
const ST = {
  seedbox: { x: 2, y: MAIN_Y, w: 15, h: MAIN_H, label: "SEEDBOX", tag: "01" },
  inbox: { x: 42, y: MAIN_Y, w: 12, h: MAIN_H, label: "INBOX", tag: "02" },
  ingest: { x: 58, y: MAIN_Y, w: 15, h: MAIN_H, label: "INGEST", tag: "03" },
  embed: { x: 77, y: MAIN_Y, w: 15, h: MAIN_H, label: "EMBED", tag: "04" },
  picture: { x: 96, y: MAIN_Y, w: 15, h: MAIN_H, label: "PICTURE", tag: "05" },
  dupes: { x: 115, y: MAIN_Y, w: 14, h: MAIN_H, label: "DUPES", tag: "06" },
  library: { x: 133, y: MAIN_Y, w: 22, h: MAIN_H, label: "LIBRARY", tag: "07" },
  purge: { x: 2, y: LOW_Y, w: 17, h: LOW_H, label: "PURGE", tag: "08" },
  upgrades: { x: 26, y: LOW_Y, w: 21, h: LOW_H, label: "UPGRADES", tag: "09" },
  backup: { x: 54, y: LOW_Y, w: 17, h: LOW_H, label: "BACKUP", tag: "10" },
  taste: { x: 78, y: LOW_Y, w: 17, h: LOW_H, label: "TASTE", tag: "11" },
  report: { x: 101, y: LOW_Y, w: 15, h: LOW_H, label: "REPORT", tag: "12" },
  crusher: { x: 122, y: LOW_Y, w: 18, h: LOW_H, label: "CRUSHER", tag: "13" },
};
const LANES_X0 = 17, LANES_X1 = 42, LANE_Y = [MAIN_Y + 3, MAIN_Y + 6, MAIN_Y + 9];
const BELT_Y = MAIN_Y + 7;
const MAIN_ORDER = ["inbox", "ingest", "embed", "picture", "dupes", "library"];
const WALK_Y = 25;

// --- palette: day amber, night teal (follows your clock) ------------------------------------
function mix(a, b, k) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const c = [16, 8, 0].map((s) => Math.round(((pa >> s) & 255) * (1 - k) + ((pb >> s) & 255) * k));
  return "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("");
}
function dayness(d = new Date()) {
  const h = d.getHours() + d.getMinutes() / 60;
  return Math.max(0, Math.min(1, (Math.sin(((h - 6.5) / 12) * Math.PI) + 0.25) / 1.1));
}
function palette(night, k) {
  const D = { bg: "#070604", floor: "#2c2414", grid: "#3a2e18", frame: "#a07a34", face: "#4a3a1c", side: "#2a2010",
              text: "#e0b45a", dim: "#6a5428", hot: "#ffe08a", run: "#8dffae", crate: "#7fe8ff", err: "#ff5a4a",
              warn: "#ffb347", light: "rgba(255,190,90,ALPHA)", plate: "#c99a42" };
  const N = { bg: "#030608", floor: "#13252a", grid: "#1a3238", frame: "#3f8f8a", face: "#183236", side: "#0e1f22",
              text: "#7fd6cc", dim: "#2f5b58", hot: "#c8fff4", run: "#8dffae", crate: "#9fdcff", err: "#ff6a5a",
              warn: "#ffc46a", light: "rgba(90,220,210,ALPHA)", plate: "#5fb3ab" };
  const out = {};
  for (const key of Object.keys(D)) out[key] = key === "light" ? (k > 0.5 ? D.light : N.light) : mix(N[key], D[key], k);
  if (night) for (const key of ["frame", "text", "plate"]) out[key] = mix(out[key], out.bg, 0.25);
  return out;
}
const TIER_COL = { legendaire: "#ffd257", exceptionnelle: "#c79bff", merveilleuse: "#7cb4ff", anomaly: "#ffb347",
                   unreviewed: "#9aa3a0", rejected: "#ff6a5a" };

// --- small helpers -------------------------------------------------------------------------
const P = FE.putW;
const stOf = (k) => (FW.data && FW.data.stations && FW.data.stations[k]) || { state: "idle" };
const running = (k) => stOf(k).state === "running";
function fmtAgo(sec) {
  if (sec == null) return "--";
  if (sec < 90) return `${Math.round(sec)}s`;
  if (sec < 5400) return `${Math.round(sec / 60)}m`;
  if (sec < 172800) return `${Math.round(sec / 3600)}h`;
  return `${Math.round(sec / 86400)}d`;
}
function fmtNum(n) {
  if (n == null) return "--";
  if (n >= 100000) return `${Math.floor(n / 1000)}k`;
  return String(n);
}
function gauge(pct, w) {
  if (pct == null) return "[" + ".".repeat(w - 2) + "]";
  const n = Math.round((w - 2) * Math.max(0, Math.min(1, pct)));
  return "[" + "#".repeat(n) + "-".repeat(w - 2 - n) + "]";
}
const SPIN = ["|", "/", "-", "\\"];

// a machine block: front face, top face (depth), right side, shaded
function block(ctx, s, pal, depth = 2) {
  const { x, y, w, h } = s;
  for (let r = 1; r < h - 1; r++) P(ctx, x + 1, y + r, ":".repeat(w - 2), pal.face, 0.35);
  P(ctx, x, y, "+" + "-".repeat(w - 2) + "+", pal.frame);
  for (let r = 1; r < h - 1; r++) { P(ctx, x, y + r, "|", pal.frame); P(ctx, x + w - 1, y + r, "|", pal.frame); }
  P(ctx, x, y + h - 1, "+" + "-".repeat(w - 2) + "+", pal.frame);
  // depth: top face and right side
  for (let d = 1; d <= depth; d++) {
    P(ctx, x + d, y - d, "_".repeat(w - 1), pal.side, 0.9);
    P(ctx, x + d - 1, y - d + 1, "/", pal.frame, 0.8);
    P(ctx, x + w - 1 + d, y - d + 1, "/", pal.frame, 0.8);
    for (let r = 0; r < h - 1; r++) P(ctx, x + w - 1 + d, y + r - d + 1, d === depth ? "|" : "#", pal.side, d === depth ? 0.8 : 0.5);
  }
  P(ctx, x + w - 1 + depth, y + h - depth, "/", pal.frame, 0.8);
  // name plate on the top face
  P(ctx, x + 2, y - 1, ` ${s.label} `, pal.plate, 1, true);
  P(ctx, x + w - 3, y - 1, s.tag, pal.dim, 0.9);
}

// --- the static layer: floor, walkway, machine bodies, pipes ---------------------------------
function drawStatic(pal, night) {
  const E = FE, ctx = E.sctx;
  ctx.fillStyle = pal.bg; ctx.fillRect(0, 0, E.stat.width, E.stat.height);
  for (let y = 0; y < E.WROWS; y++) {
    const a = 0.12 + 0.18 * (y / E.WROWS);
    let row = "";
    for (let x = 0; x < E.WCOLS; x++) row += (x % 8 === 0 && y % 4 === 0) ? "+" : (y % 4 === 0 ? "." : (x % 8 === 0 ? ":" : " "));
    P(ctx, 0, y, row, pal.grid, a);
  }
  // the walkway with its rails and stencil
  P(ctx, 0, WALK_Y - 1, "=".repeat(E.WCOLS), pal.dim, 0.7);
  P(ctx, 0, WALK_Y + 2, "=".repeat(E.WCOLS), pal.dim, 0.7);
  for (let x = 0; x < E.WCOLS; x += 12) P(ctx, x + 2, WALK_Y, "/ / /", pal.dim, 0.35);
  P(ctx, 64, WALK_Y + 1, "P E A K S   W O R K S   ::   L I N E   1", pal.dim, 0.55, true);
  // pipes from the main line down to the plant below
  for (const k of ["purge", "upgrades", "backup", "taste", "report"]) {
    const s = ST[k], px = s.x + Math.floor(s.w / 2);
    for (let y = MAIN_Y + MAIN_H + 1; y < LOW_Y - 2; y++) if (y < WALK_Y - 1 || y > WALK_Y + 2) P(ctx, px, y, "|", pal.side, 0.8);
  }
  // the chutes into the crusher (from duplicates and from the reject shelf)
  for (const cx of [ST.dupes.x + 7, ST.library.x + 4]) {
    for (let y = MAIN_Y + MAIN_H; y < LOW_Y - 1; y++) if (y < WALK_Y - 1 || y > WALK_Y + 2) P(ctx, cx - 1, y, "| |", pal.side, 0.8);
  }
  for (const k of Object.keys(ST)) block(ctx, ST[k], pal);
  // the pull bay: three rails between the seedbox and the inbox
  P(ctx, LANES_X0, MAIN_Y + 1, "P U L L   B A Y", pal.dim, 0.8, true);
  for (const ly of LANE_Y) { P(ctx, LANES_X0, ly + 1, "-".repeat(LANES_X1 - LANES_X0), pal.side, 0.8); }
  if (night) P(ctx, 4, 48, "night shift", pal.dim, 0.5);
}

// --- machines in motion ---------------------------------------------------------------------
function lamp(ctx, s, state, pal, f) {
  const at = s.x + s.w - 5, y = s.y;
  if (state === "running") P(ctx, at, y, f % 12 < 8 ? "(*)" : "( )", pal.run, 1, true);
  else if (state === "error") P(ctx, at, y, f % 6 < 3 ? "(!)" : "( )", pal.err, 1, true);
  else if (state === "done") P(ctx, at, y, "(+)", pal.run, 0.8);
  else P(ctx, at, y, "(.)", pal.dim, 0.7);
}
function plate(ctx, s, text, pal, hot) {
  const line = text.slice(0, s.w - 2);
  P(ctx, s.x + 1, s.y + s.h + 1, "[" + line.padEnd(s.w - 4) + "]", hot ? pal.hot : pal.plate, hot ? 1 : 0.85);
}

function machines(ctx, pal, f, t) {
  const d = FW.data || { stations: {}, lanes: [], shelves: {}, counts: {} };
  const S = d.stations || {};
  // 01 seedbox: a rack of drives, LEDs chatter while it lists
  {
    const s = ST.seedbox, on = (S.seedbox || {}).state === "running" || running("pull");
    for (let i = 0; i < 7; i++) {
      const blink = on && ((f + i * 5) % 9 < 4);
      P(ctx, s.x + 2, s.y + 2 + i, "[=======]", blink ? pal.hot : pal.text, blink ? 1 : 0.6);
      P(ctx, s.x + 11, s.y + 2 + i, blink ? "*" : ".", blink ? pal.run : pal.dim);
    }
    P(ctx, s.x + 6, s.y - 3, on ? "\\|/" : " | ", on ? pal.hot : pal.dim);
    if (on && f % 10 === 0) FE.emit(s.x + 7, s.y - 3, 1, { chars: ")>-", color: pal.crate, vy: -0.3, spread: 0.6, g: 0, life: 0.8 });
    lamp(ctx, s, on ? "running" : "idle", pal, f);
    plate(ctx, s, (S.seedbox || {}).state === "running" ? "LISTING" : on ? "SENDING" : "IDLE", pal, on);
  }
  // pull lanes: one crate per file in flight, filling with bytes
  {
    const lanes = d.lanes || [];
    LANE_Y.forEach((ly, i) => {
      const lane = lanes[i], on = !!lane;
      let belt = "";
      for (let x = 0; x < LANES_X1 - LANES_X0; x++) belt += on ? ((x + f) % 4 === 0 ? ">" : "=") : "=";
      P(ctx, LANES_X0, ly, belt, on ? pal.text : pal.dim, on ? 0.9 : 0.5);
      if (lane) {
        const c = FW.lanes[lane.id] || (FW.lanes[lane.id] = { x: LANES_X0 });
        const target = LANES_X0 + lane.pct * (LANES_X1 - LANES_X0 - 9);
        c.x += (target - c.x) * 0.08;
        const fill = Math.round(lane.pct * 4);
        P(ctx, c.x, ly - 1, `[${"#".repeat(fill).padEnd(4, ".")}]`, pal.crate, 1, true);
        FE.putW(FE.tctx, c.x, ly - 1, "[####]", pal.crate, 0.5);
        P(ctx, LANES_X0 + 1, ly + 1, `${(lane.cat || "?").slice(0, 2).padEnd(3)}${lane.mbps ? lane.mbps.toFixed(1) + " MB/s" : ""}  ${Math.round(lane.pct * 100)}%`, pal.dim, 0.9);
      }
    });
    if ((lanes.length || 0) > 3) P(ctx, LANES_X0 + 2, LANE_Y[2] + 2, `+${lanes.length - 3} more`, pal.dim);
    const q = (S.pull || {}).queued;
    if (q) P(ctx, LANES_X0 + 16, MAIN_Y + 1, `queue ${q}`, pal.dim);
  }
  // 02 inbox: a hopper that fills while files land
  {
    const s = ST.inbox, on = running("pull") || running("ingest");
    const lvl = on ? 1 + (Math.floor(f / 8) % 4) : 0;
    for (let r = 0; r < 5; r++) {
      const w = s.w - 3 - r * 2;
      if (w < 2) break;
      const fill = r >= 5 - lvl;
      P(ctx, s.x + 1 + r, s.y + 2 + r, "\\" + (fill ? "o".repeat(w - 1) : " ".repeat(w - 1)) + "/", fill ? pal.crate : pal.text, fill ? 0.9 : 0.7);
    }
    P(ctx, s.x + Math.floor(s.w / 2) - 1, s.y + 8, "|_|", pal.text, 0.7);
    if (on && f % 6 < 3) P(ctx, s.x + Math.floor(s.w / 2), s.y + 9, "o", pal.crate);
    lamp(ctx, s, on ? "running" : "idle", pal, f);
    plate(ctx, s, on ? "FILLING" : "EMPTY", pal, on);
  }
  // 03 ingest: a press stamping each file into Stash
  {
    const s = ST.ingest, ss = S.ingest || {}, on = ss.state === "running";
    const p = on ? [0, 1, 2, 3, 3, 2, 1][f % 7] : 0;
    const cx = s.x + Math.floor(s.w / 2);
    P(ctx, cx - 4, s.y + 1, "=========", pal.text, 0.9);
    for (let i = 0; i < p; i++) P(ctx, cx, s.y + 2 + i, "||", pal.text, 0.8);
    P(ctx, cx - 3, s.y + 2 + p, "#######", on ? pal.hot : pal.text, on ? 1 : 0.6);
    P(ctx, s.x + 2, s.y + 7, "_".repeat(s.w - 4), pal.text, 0.6);
    if (on && p === 3 && f % 7 === 3) FE.emit(cx, s.y + 6, 5, { chars: "*+.", color: pal.hot, vy: -0.6, spread: 1.4, life: 0.5 });
    const stage = ["SCAN", "IDENTIFY", "AUTO TAG", "EMBED", "DUPES"][on ? Math.floor(t / 2500) % 5 : 0];
    P(ctx, s.x + 2, s.y + 9, on ? stage : "READY", on ? pal.text : pal.dim);
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, on ? `INGEST ${ss.pct != null ? Math.round(ss.pct * 100) + "%" : ""}` : "IDLE", pal, on);
  }
  // 04 embed: the furnace — a turbine and the GPU's flame
  {
    const s = ST.embed, ss = S.embed || {}, on = ss.state === "running";
    const cx = s.x + Math.floor(s.w / 2), cy = s.y + 4;
    const sp = on ? SPIN[f % 4] : "|", sp2 = on ? SPIN[(f + 2) % 4] : "-";
    P(ctx, cx - 3, cy - 2, " .---. ", pal.text, 0.8);
    P(ctx, cx - 3, cy - 1, `/ ${sp} ${sp2} \\`, on ? pal.hot : pal.text);
    P(ctx, cx - 3, cy, `|${sp2} (O) ${sp}`.slice(0, 7) + "|", on ? pal.hot : pal.text);
    P(ctx, cx - 3, cy + 1, `\\ ${sp2} ${sp} /`, on ? pal.hot : pal.text);
    P(ctx, cx - 3, cy + 2, " '---' ", pal.text, 0.8);
    let flame = "";
    for (let i = 0; i < s.w - 4; i++) flame += on ? "^'\"`^."[(i * 7 + f) % 6] : "_";
    P(ctx, s.x + 2, s.y + 9, flame, on ? pal.warn : pal.dim, on ? 0.9 : 0.5);
    if (on && f % 5 === 0) FE.emit(cx, s.y - 2, 1, { chars: "~'.", color: pal.dim, vy: -0.4, spread: 0.4, g: -0.05, life: 1.6 });
    lamp(ctx, s, ss.state, pal, f);
    const g = ss.gpu ? ` GPU${ss.gpu.util}%` : "";
    plate(ctx, s, on ? `${ss.done ?? 0}/${ss.total ?? "?"}${ss.rate ? " " + ss.rate + "/m" : ""}${g}` : `IDLE${g}`, pal, on);
  }
  // 05 picture: a scanner beam sweeping a grid
  {
    const s = ST.picture, ss = S.picture || {}, on = ss.state === "running";
    const span = s.w - 4, k = on ? f % (span * 2) : 0, pos = k < span ? k : span * 2 - k - 1;
    for (let r = 0; r < 6; r++) {
      P(ctx, s.x + 2, s.y + 2 + r, ".:".repeat(span).slice(0, span), pal.text, 0.35);
      if (on) {
        P(ctx, s.x + 2 + pos, s.y + 2 + r, "|", pal.hot, 1, true);
        FE.putW(FE.tctx, s.x + 2 + pos, s.y + 2 + r, "|", pal.hot, 0.6);
      }
    }
    P(ctx, s.x + 2, s.y + 9, on ? "MEASURING" : "STANDBY", on ? pal.text : pal.dim);
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, on && ss.total ? `${ss.done}/${ss.total} FILES` : "IDLE", pal, on);
  }
  // 06 duplicates: a sorter flipping copies to KEEP or the chute
  {
    const s = ST.dupes, ss = S.dupes || {}, on = ss.state === "running" || (ss.queued || 0) > 0;
    const flip = on && f % 10 < 5;
    P(ctx, s.x + 2, s.y + 2, "KEEP", pal.run, 0.8); P(ctx, s.x + s.w - 6, s.y + 2, "DROP", pal.err, 0.8);
    P(ctx, s.x + 4, s.y + 4, flip ? "\\____" : "____/", on ? pal.hot : pal.text);
    P(ctx, s.x + 5, s.y + 6, on ? "<=[]=>" : " [] ", on ? pal.crate : pal.dim);
    P(ctx, s.x + 6, s.y + 9, "\\  /", pal.text, 0.7);
    lamp(ctx, s, ss.state === "running" ? "running" : ss.state, pal, f);
    plate(ctx, s, `${ss.groups ?? 0} GRP ${ss.queued ? ss.queued + "Q" : ""}`, pal, on);
  }
  // 07 library: the shelves, filling as you grade
  {
    const s = ST.library, sh = d.shelves || {};
    const rows = [["LEG", "legendaire"], ["EXC", "exceptionnelle"], ["MER", "merveilleuse"], ["ANO", "anomaly"],
      ["NEW", "unreviewed"], ["REJ", "rejected"]];
    const max = Math.max(1, ...rows.map(([, t]) => sh[t] || 0));
    rows.forEach(([lab, t], i) => {
      const n = sh[t] || 0, bar = Math.round(7 * n / max);
      P(ctx, s.x + 2, s.y + 2 + i, lab, TIER_COL[t], 1, true);
      P(ctx, s.x + 6, s.y + 2 + i, "#".repeat(bar).padEnd(7, "."), TIER_COL[t], 0.85);
      P(ctx, s.x + 14, s.y + 2 + i, fmtNum(n).padStart(6), pal.text, 0.9);
    });
    const total = rows.reduce((a, [, t]) => a + (sh[t] || 0), 0);
    lamp(ctx, s, total ? "done" : "idle", pal, f);
    plate(ctx, s, `${fmtNum(total)} SCENES`, pal, false);
  }
  // 08 purge: the crusher at the seedbox
  {
    const s = ST.purge, ss = S.purge || {}, on = ss.state === "running";
    const shut = on && f % 8 < 4;
    P(ctx, s.x + 3, s.y + 2, shut ? "  \\    /  " : "\\        /", on ? pal.hot : pal.text);
    P(ctx, s.x + 3, s.y + 3, shut ? "   \\  /   " : " \\      / ", on ? pal.hot : pal.text);
    P(ctx, s.x + 3, s.y + 4, shut ? "    ><    " : "  \\____/  ", on ? pal.hot : pal.text);
    P(ctx, s.x + 2, s.y + 7, ss.dry === false ? "LIVE" : "DRY RUN", ss.dry === false ? pal.err : pal.warn, 1, true);
    if (on && f % 8 === 4) FE.emit(s.x + 8, s.y + 5, 6, { chars: "x*.", color: pal.err, vy: -0.3, spread: 1.6, life: 0.5 });
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, ss.ready != null ? `${ss.ready} READY ${ss.gb ? ss.gb + "GB" : ""}` : "IDLE", pal, on);
  }
  // 09 upgrades: the radar dish
  {
    const s = ST.upgrades, ss = S.upgrades || {}, on = ss.state === "running";
    const cx = s.x + Math.floor(s.w / 2), cy = s.y + 4, R = 3;
    for (let a = 0; a < 24; a++) {
      const ang = a / 24 * Math.PI * 2;
      P(ctx, cx + Math.round(Math.cos(ang) * R * 2), cy + Math.round(Math.sin(ang) * R), ".", pal.text, 0.5);
    }
    const sweep = (t / (on ? 450 : 2600)) % (Math.PI * 2);
    for (let r = 1; r <= R; r++) {
      const px = cx + Math.round(Math.cos(sweep) * r * 2), py = cy + Math.round(Math.sin(sweep) * r);
      P(ctx, px, py, "*", on ? pal.hot : pal.text, on ? 1 : 0.6);
      FE.putW(FE.tctx, px, py, "*", on ? pal.hot : pal.text, 0.5);
    }
    P(ctx, cx, cy, "o", pal.hot);
    const found = ss.found || 0;
    for (let i = 0; i < Math.min(found, 6); i++) {
      const ang = i * 2.1 + 0.4;
      P(ctx, cx + Math.round(Math.cos(ang) * 4), cy + Math.round(Math.sin(ang) * 2), (f + i * 3) % 14 < 9 ? "+" : " ", pal.run);
    }
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, `FOUND ${found}  ${ss.last ? fmtAgo(Date.now() / 1000 - ss.last) + " AGO" : ""}`, pal, found > 0);
  }
  // 10 backup: the vault
  {
    const s = ST.backup, ss = S.backup || {}, on = ss.state === "running";
    const cx = s.x + Math.floor(s.w / 2), cy = s.y + 4;
    const sp = on ? SPIN[Math.floor(f / 2) % 4] : "+";
    P(ctx, cx - 4, cy - 2, " .-----. ", pal.text, 0.8);
    P(ctx, cx - 4, cy - 1, "/   |   \\", pal.text, 0.8);
    P(ctx, cx - 4, cy, `|---${sp}---|`, on ? pal.hot : pal.text);
    P(ctx, cx - 4, cy + 1, "\\   |   /", pal.text, 0.8);
    P(ctx, cx - 4, cy + 2, " '-----' ", pal.text, 0.8);
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, on ? "SEALING" : `LAST ${ss.last ? fmtAgo(Date.now() / 1000 - ss.last) : "--"}`, pal, on);
  }
  // 11 taste: the loom
  {
    const s = ST.taste, ss = S.taste || {}, on = ss.state === "running";
    for (let r = 0; r < 5; r++) {
      let line = "";
      for (let i = 0; i < s.w - 4; i++) line += on ? "/\\|-~"[(i + r * 2 + f) % 5] : (r % 2 ? "-" : "|");
      P(ctx, s.x + 2, s.y + 2 + r, line, on ? pal.hot : pal.text, on ? 0.85 : 0.35);
    }
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, on ? `LEARNING ${ss.pct != null ? Math.round(ss.pct * 100) + "%" : ""}` : "IDLE", pal, on);
  }
  // 12 report: the printer
  {
    const s = ST.report, ss = S.report || {}, on = ss.state === "running";
    P(ctx, s.x + 2, s.y + 2, "[=========]", pal.text, 0.8);
    P(ctx, s.x + 3, s.y + 3, on ? "  >>>>   " : "         ", pal.hot);
    const feed = on ? (Math.floor(f / 2) % 6) : 0;
    for (let i = 0; i < feed; i++) P(ctx, s.x + 3, s.y + 4 + i, "|~~~~~~~|", pal.hot, 0.9 - i * 0.1);
    lamp(ctx, s, ss.state, pal, f);
    plate(ctx, s, on ? "PRINTING" : "06:00", pal, on);
  }
  // 13 crusher: duplicates and rejects end here
  {
    const s = ST.crusher, hit = FW.shake > 0.05;
    P(ctx, s.x + 3, s.y + 2, hit ? " \\\\  // " : "\\\\    //", hit ? pal.err : pal.text);
    P(ctx, s.x + 3, s.y + 3, hit ? "  \\\\//  " : " \\\\  // ", hit ? pal.err : pal.text);
    P(ctx, s.x + 3, s.y + 5, "#".repeat(Math.min(s.w - 6, 1 + (FW.day?.today?.rejected || 0) % (s.w - 6))), pal.err, 0.6);
    lamp(ctx, s, hit ? "error" : "idle", pal, f);
    plate(ctx, s, `REJ TODAY ${FW.day?.today?.rejected ?? 0}`, pal, hit);
  }
}

// the main belts between machines: crates roll toward whoever is working
function belts(ctx, pal, f, dt) {
  for (let i = 0; i < MAIN_ORDER.length - 1; i++) {
    const a = ST[MAIN_ORDER[i]], b = ST[MAIN_ORDER[i + 1]], on = running(MAIN_ORDER[i + 1]);
    const x0 = a.x + a.w + 2, x1 = b.x;
    let s = "";
    for (let x = x0; x < x1; x++) s += on ? ((x + f) % 3 === 0 ? ">" : "=") : "=";
    P(ctx, x0, BELT_Y, s, on ? pal.text : pal.dim, on ? 0.9 : 0.45);
    // crates: one every ~2 s while the next machine works
    const key = MAIN_ORDER[i + 1];
    const mine = FW.crates.filter((c) => c.belt === key);
    if (on && mine.length < 2 && Math.random() < dt * 0.6) FW.crates.push({ belt: key, x: x0, x1, cat: "" });
  }
  FW.crates = FW.crates.filter((c) => {
    c.x += dt * 5;
    if (c.x >= c.x1 - 3) { FE.emit(c.x1 - 1, BELT_Y - 1, 3, { chars: ".'", color: pal.crate, vy: -0.4, life: 0.4 }); return false; }
    P(ctx, c.x, BELT_Y - 1, "[##]", pal.crate, 1, true);
    FE.putW(FE.tctx, c.x, BELT_Y - 1, "[##]", pal.crate, 0.45);
    return true;
  });
  // things falling down the chutes (rejects, removed copies)
  FW.drops = FW.drops.filter((d) => {
    d.vy += dt * 18; d.y += d.vy * dt;
    if (d.y >= LOW_Y + 2) {
      FW.shake = 0.6;
      FE.emit(d.x, LOW_Y + 2, 14, { chars: "x*+.", color: pal.err, vy: -1, spread: 2.2, life: 0.7 });
      if (window.FA) FA.thud();
      return false;
    }
    P(ctx, d.x - 1, d.y, "[xx]", pal.err, 1, true);
    FE.putW(FE.tctx, d.x - 1, d.y, "[xx]", pal.err, 0.5);
    return true;
  });
}

// the night-shift forklift that patrols the walkway
function forklift(ctx, pal, dt, idle) {
  const fk = FW.forklift;
  fk.x += fk.dir * dt * (idle ? 2.2 : 4);
  if (fk.x > FE.WCOLS - 12) fk.dir = -1;
  if (fk.x < 2) fk.dir = 1;
  const body = fk.dir > 0 ? "[#]=o" : "o=[#]";
  P(ctx, fk.x, WALK_Y, body, pal.warn, 0.9, true);
  P(ctx, fk.x + (fk.dir > 0 ? 0 : 1), WALK_Y + 1, "o  o", pal.dim, 0.9);
  FE.light(ctx, fk.x + (fk.dir > 0 ? 7 : -2), WALK_Y + 0.5, 5, pal.light, 0.12);
}

// --- the screen-space instrument panels ------------------------------------------------------
const BIG = {   // 3x5 block digits for the clock
  "0": ["###", "# #", "# #", "# #", "###"], "1": [" # ", "## ", " # ", " # ", "###"], "2": ["###", "  #", "###", "#  ", "###"],
  "3": ["###", "  #", "###", "  #", "###"], "4": ["# #", "# #", "###", "  #", "  #"], "5": ["###", "#  ", "###", "  #", "###"],
  "6": ["###", "#  ", "###", "# #", "###"], "7": ["###", "  #", "  #", "  #", "  #"], "8": ["###", "# #", "###", "# #", "###"],
  "9": ["###", "# #", "###", "  #", "###"], ":": ["   ", " # ", "   ", " # ", "   "],
};
const bytes = (n) => n >= 1e12 ? (n / 1e12).toFixed(1) + "TB" : (n / 1e9).toFixed(0) + "GB";
function bigText(col, row, s, color) {
  for (let r = 0; r < 5; r++) {
    // each lit pixel two characters wide (square-ish), the unlit ones a faint LED dot
    let c = col;
    for (const ch of s) {
      const px = ch === ":" ? " # # "[r] : (BIG[ch] || ["   ", "   ", "   ", "   ", "   "])[r];
      for (const p of px) {
        if (p === "#") FE.putS(c, row + r, "@@", color, 1, true);
        else FE.putS(c, row + r, "..", color, 0.15);
        c += 2;
      }
      c += 1;
    }
  }
}
// Braille sparkline: two values per character, four dots tall
function braille(vals, width) {
  const v = vals.slice(-width * 2);
  const max = Math.max(1e-9, ...v);
  let out = "";
  for (let i = 0; i < v.length; i += 2) {
    const a = Math.round(4 * (v[i] || 0) / max), b = Math.round(4 * (v[i + 1] || 0) / max);
    const L = [0, 0x40, 0x44, 0x46, 0x47][a], R = [0, 0x80, 0xa0, 0xb0, 0xb8][b];
    out += String.fromCharCode(0x2800 + (L | R));
  }
  return out.padStart(width, String.fromCharCode(0x2800));
}
const SHADE = " .:-=+*#%@";

function panels(pal, f, t) {
  const E = FE, cols = Math.floor(E.W / E.cw), rows = Math.floor(E.H / E.ch);
  const d = FW.data || { stations: {} }, day = FW.day || {}, today = day.today || {};
  const LW = 32, RW = 34;
  // header
  E.putS(1, 0, "PEAKS // CONTROL ROOM", pal.hot, 1, true);
  const now = FW.replay ? new Date(FW.replay.t * 1000) : new Date();
  const hhmmss = now.toTimeString().slice(0, 8);
  E.putS(cols - 22, 0, FW.replay ? `REPLAY ${hhmmss.slice(0, 5)}` : `LIVE ${hhmmss}`, FW.replay ? pal.warn : pal.run, 1, true);
  if (!FW.replay) E.putS(cols - 24, 0, f % 30 < 15 ? "*" : " ", pal.err, 1, true);
  const busy = Object.entries(d.stations || {}).filter(([, v]) => v.state === "running").map(([k]) => k.toUpperCase());
  const mid = busy.length ? `RUNNING: ${busy.join("  ")}` : (FW.idle ? "ALL QUIET -- NIGHT SHIFT" : "ALL QUIET");
  E.putS(Math.max(LW + 2, Math.floor(cols / 2 - mid.length / 2)), 0, mid, busy.length ? pal.run : pal.dim, 1, busy.length > 0);
  E.putS(0, 1, "=".repeat(cols), pal.dim, 0.6);

  // left: the clock, the line, today
  let r = 3;
  bigText(1, r, now.toTimeString().slice(0, 5), pal.text); r += 6;
  E.putS(1, r++, now.toDateString().toUpperCase(), pal.dim);
  r++;
  E.putS(1, r++, "-- LINE STATUS " + "-".repeat(LW - 16), pal.dim);
  for (const k of ["seedbox", "pull", "ingest", "embed", "picture", "dupes", "purge", "upgrades", "backup", "taste", "report"]) {
    const s = (d.stations || {})[k] || { state: "idle" };
    const lampc = s.state === "running" ? pal.run : s.state === "error" ? pal.err : s.state === "done" ? pal.run : pal.dim;
    const word = { running: "RUNNING", error: "FAULT", done: "DONE", idle: "idle" }[s.state] || s.state;
    E.putS(1, r, s.state === "running" ? (f % 10 < 6 ? "[*]" : "[ ]") : s.state === "error" ? "[!]" : "[.]", lampc, 1, true);
    E.putS(5, r, k.toUpperCase().padEnd(10), s.state === "running" ? pal.text : pal.dim);
    E.putS(16, r, word.padEnd(8), lampc);
    if (s.pct != null) E.putS(24, r, gauge(s.pct, 8), pal.run);
    r++;
  }
  r++;
  E.putS(1, r++, "-- TODAY " + "-".repeat(LW - 10), pal.dim);
  const T = [["pulled", `${today.pulled ?? 0} files`], ["in", `${today.pulled_gb ?? 0} GB`], ["embedded", today.embedded ?? 0],
    ["graded", today.graded ?? 0], ["rejected", today.rejected ?? 0], ["freed", `${today.freed_gb ?? 0} GB`],
    ["upgrades", today.upgrades ?? 0]];
  for (const [k, v] of T) { E.putS(2, r, k.padEnd(10, "."), pal.dim); E.putS(13, r, String(v), pal.text); r++; }

  // right: throughput, the day, storage
  const rc = cols - RW;
  r = 3;
  E.putS(rc, r++, "-- THROUGHPUT (60 MIN) " + "-".repeat(RW - 24), pal.dim);
  const ser = d.series || {};
  const pull = ser.pull_mbps || [];
  E.putS(rc, r, "pull MB/s", pal.dim); E.putS(rc + 10, r, braille(pull, 18), pal.crate);
  E.putS(rc + 29, r++, String(pull.length ? pull[pull.length - 1] : 0).slice(0, 5), pal.text);
  const emb = ser.embeds || [];
  const embRate = emb.map((v, i) => (i ? Math.max(0, v - emb[i - 1]) : 0));
  E.putS(rc, r, "embeds/m", pal.dim); E.putS(rc + 10, r, braille(embRate, 18), pal.run);
  E.putS(rc + 29, r++, String(embRate.length ? embRate[embRate.length - 1] : 0), pal.text);
  const lib = (day.buckets || {}).library || [];
  E.putS(rc, r, "grades", pal.dim); E.putS(rc + 10, r, braille(lib.slice(-36), 18), TIER_COL.legendaire);
  E.putS(rc + 29, r++, String(lib.slice(-6).reduce((a, b) => a + b, 0)) + "/h", pal.text);
  r++;
  E.putS(rc, r++, "-- LAST 24 HOURS " + "-".repeat(RW - 18), pal.dim);
  const B = day.buckets || {};
  const width = RW - 11;
  for (const k of ["pull", "ingest", "embed", "picture", "library", "dupes", "purge", "upgrades", "backup"]) {
    const b = B[k] || [];
    const per = Math.max(1, Math.ceil(144 / width));
    const max = Math.max(1, ...b);
    let line = "";
    for (let i = 0; i < width; i++) {
      const v = b.slice(i * per, i * per + per).reduce((a, c) => a + c, 0) / per;
      line += SHADE[Math.min(9, Math.round(9 * Math.sqrt(v / max)))];
    }
    E.putS(rc, r, (k === "library" ? "grades" : k).padEnd(10), pal.dim);
    E.putS(rc + 10, r++, line, k === "library" ? TIER_COL.legendaire : pal.text, 0.95);
  }
  E.putS(rc + 10, r++, "-24h" + " ".repeat(width - 8) + "now", pal.dim, 0.7);
  r++;
  E.putS(rc, r++, "-- STORAGE " + "-".repeat(RW - 12), pal.dim);
  const disks = day.disks || {};
  for (const [k, v] of Object.entries(disks)) {
    if (v.total) {
      const used = 1 - v.free / v.total;
      E.putS(rc, r, k.padEnd(9), pal.dim);
      E.putS(rc + 9, r, gauge(used, 14), used > 0.9 ? pal.err : used > 0.8 ? pal.warn : pal.run);
      E.putS(rc + 24, r++, `${bytes(v.free)} free`, pal.text);
    } else {
      E.putS(rc, r, k.padEnd(9), pal.dim);
      E.putS(rc + 9, r++, `${bytes(v.free)} free`, pal.text);
    }
  }

  // bottom: the log, typed out
  const tickRows = 5, ty = rows - tickRows - 1;
  E.putS(0, ty - 1, "=".repeat(cols), pal.dim, 0.6);
  E.putS(1, ty - 1, " LOG ", pal.text, 1, true);
  const lines = FW.replay ? FW.replay.log : ((d.ticker || []).map((x) => x.text));
  const show = lines.slice(-tickRows);
  show.forEach((ln, i) => {
    const last = i === show.length - 1;
    let text = "> " + ln;
    if (last) {
      if (FW.typed.text !== text) { FW.typed.text = text; FW.typed.shown = 2; }
      FW.typed.shown = Math.min(text.length, FW.typed.shown + 2);
      text = text.slice(0, FW.typed.shown);
    }
    E.putS(1, ty + i, text.slice(0, cols - 3), last ? pal.run : pal.dim, last ? 1 : 0.85);
    if (last && f % 20 < 10) E.putS(1 + Math.min(text.length, cols - 3), ty + i, "_", pal.run);
  });
  if (!show.length) E.putS(1, ty, "> quiet. the line is waiting for work.", pal.dim);
  return { left: LW + 1, right: rc - 1, top: 2, bottom: ty - 2 };
}

// --- events: what changed since the last poll -----------------------------------------------
function events(prev, next, pal) {
  if (!prev) return;
  const ps = prev.stations || {}, ns = next.stations || {};
  const was = (k) => (ps[k] || {}).state, is = (k) => (ns[k] || {}).state;
  const ids = new Set((next.lanes || []).map((l) => l.id));
  for (const l of prev.lanes || []) {
    if (!ids.has(l.id)) {                      // a file landed: it drops into the inbox
      FE.emit(LANES_X1 - 1, LANE_Y[0], 6, { chars: "o.", color: pal.crate, vx: 0.4, vy: -0.3, spread: 0.8, life: 0.6 });
      if (window.FA) FA.click();
    }
  }
  if ((next.lanes || []).length > (prev.lanes || []).length) FE.focus(28, MAIN_Y + 6, 1.35);
  if (is("seedbox") === "running" && was("seedbox") !== "running") FE.focus(ST.seedbox.x + 7, MAIN_Y + 5, 1.3);
  for (const k of ["purge", "backup", "report", "upgrades", "ingest"]) {
    if (is(k) === "running" && was(k) !== "running") {
      const s = ST[k];
      FE.focus(s.x + s.w / 2, s.y + s.h / 2, 1.3, 4);
      if (k === "report" && window.FA) FA.printer();
      if (k === "purge" && window.FA) FA.thud();
    }
  }
  const fu = (ns.upgrades || {}).found || 0, pu = (ps.upgrades || {}).found || 0;
  if (fu > pu) {
    const s = ST.upgrades;
    FE.focus(s.x + s.w / 2, s.y + 4, 1.35, 4);
    FE.emit(s.x + s.w / 2, s.y + 4, 16, { chars: "()+.", color: pal.run, vy: 0, spread: 2.4, g: 0, life: 1 });
    if (window.FA) FA.ping();
  }
}
function dayEvents(prev, next) {
  if (!prev || !next) return;
  const add = ((next.today || {}).rejected || 0) - ((prev.today || {}).rejected || 0);
  for (let i = 0; i < Math.min(add, 5); i++) {
    FW.drops.push({ x: ST.library.x + 4, y: MAIN_Y + MAIN_H + i * 1.5, vy: 0 });
  }
  if (add > 0) FE.focus(ST.crusher.x + 9, LOW_Y - 2, 1.25, 3);
}

// --- the loop ---------------------------------------------------------------------------------
function frame(ts) {
  FW.raf = requestAnimationFrame(frame);
  const minDt = 1000 / FW.fps;
  if (ts - FW.last < minDt - 1) return;
  const dt = Math.min(0.1, (ts - (FW.last || ts)) / 1000) || 1 / FW.fps;
  FW.last = ts;
  FW.f = (FW.f || 0) + 1;
  FW.frames++;
  if (ts - FW.fpsAt > 1000) {
    FW.measured = FW.frames * 1000 / (ts - FW.fpsAt); FW.frames = 0; FW.fpsAt = ts;
    // a machine that can't keep up gets a lighter world canvas instead of a stutter
    FW.slow = FW.measured < 20 ? (FW.slow || 0) + 1 : 0;
    if (FW.slow >= 3 && FE.lighten()) FW.slow = 0;
  }
  const E = FE, f = FW.f;
  if (FW.replay) replayStep(dt);
  const anyRun = FW.data && Object.values(FW.data.stations || {}).some((s) => s.state === "running");
  if (anyRun) FW.lastActive = Date.now();
  FW.idle = Date.now() - FW.lastActive > 10 * 60 * 1000;
  const k = dayness(FW.replay ? new Date(FW.replay.t * 1000) : new Date());
  const pal = palette(FW.idle, k);
  const key = `${E.W}x${E.H}|${Math.round(k * 8)}|${FW.idle}`;
  if (E.statKey !== key) { drawStatic(pal, FW.idle); E.statKey = key; }
  // world
  const w = E.wctx;
  w.globalCompositeOperation = "copy";
  w.drawImage(E.stat, 0, 0);
  w.globalCompositeOperation = "source-over";
  for (const name of Object.keys(ST)) {
    const s = ST[name], on = running(name) || (name === "inbox" && running("pull")) || (name === "seedbox" && running("pull"));
    if (on) E.light(w, s.x + s.w / 2, s.y + s.h, s.w * 0.9, pal.light, (FW.idle ? 0.12 : 0.22) + 0.04 * Math.sin(ts / 300));
  }
  if (FW.idle) for (let i = 0; i < 2; i++) if (Math.random() < 0.05) E.emit(Math.random() * E.WCOLS, Math.random() * E.WROWS, 1, { chars: ".", color: pal.dim, vy: -0.05, g: 0, spread: 0.1, life: 3 });
  E.fadeTrail(dt);
  belts(w, pal, f, dt);
  machines(w, pal, f, ts);
  forklift(w, pal, dt, FW.idle);
  E.stepParticles(w, dt);
  w.globalCompositeOperation = "lighter";
  w.drawImage(E.trail, 0, 0);
  w.globalCompositeOperation = "source-over";
  // screen
  const c = E.ctx;
  c.fillStyle = pal.bg; c.fillRect(0, 0, E.W, E.H);
  const box = panels(pal, f, ts);
  E.view = { x: box.left * E.cw, y: box.top * E.ch, w: (box.right - box.left) * E.cw, h: (box.bottom - box.top) * E.ch };
  E.stepCamera(dt, ts, FW.idle);
  if (FW.shake > 0) { FW.shake = Math.max(0, FW.shake - dt); E.view.x += (Math.random() - 0.5) * 10 * FW.shake; E.view.y += (Math.random() - 0.5) * 6 * FW.shake; }
  E.blitWorld();
  E.post(ts);
  if (window.FA) FA.hum(Object.values((FW.data || {}).stations || {}).filter((s) => s.state === "running").length);
}

// --- data -------------------------------------------------------------------------------------
async function poll() {
  if (FW.replay) return;
  try {
    const d = await api("/api/floor");
    const pal = palette(FW.idle, dayness());
    events(FW.data, d, pal);
    FW.data = d;
  } catch { /* keep drawing the last picture */ }
}
async function pollDay() {
  try {
    const d = await api("/api/floor/day");
    dayEvents(FW.day, d);
    FW.day = d;
  } catch { /* ignore */ }
}

// --- replay: the last 24 hours in about a minute -----------------------------------------------
async function startReplay() {
  let ev;
  try { ev = (await api("/api/floor/replay?hours=24")).events; } catch (e) { toast(e.message, true); return; }
  const now = Date.now() / 1000;
  FW.replay = { ev, i: 0, t: now - 86400, end: now, step: 86400 / (60 * FW.fps), log: [], hot: {} };
  FW.saved = FW.data;
  FW.data = { stations: {}, lanes: [], shelves: (FW.saved || {}).shelves || {}, counts: {}, ticker: [] };
}
function replayStep() {
  const r = FW.replay;
  r.t += r.step;
  while (r.i < r.ev.length && r.ev[r.i].at <= r.t) {
    const e = r.ev[r.i++];
    r.hot[e.station] = FW.f + 20;
    r.log.push(`${new Date(e.at * 1000).toTimeString().slice(0, 5)}  ${e.station}${e.n > 1 ? " x" + e.n : ""}${e.ok ? "" : "  FAILED"}`);
    r.log = r.log.slice(-6);
    if (e.station === "library" && Math.random() < 0.15) FW.drops.push({ x: ST.library.x + 4, y: MAIN_Y + MAIN_H, vy: 0 });
  }
  const sts = {};
  for (const k of ["pull", "purge", "ingest", "embed", "picture", "dupes", "upgrades", "backup", "taste", "report"]) {
    sts[k] = { state: (r.hot[k] || 0) > FW.f ? "running" : "idle" };
  }
  FW.data.stations = sts;
  FW.data.lanes = sts.pull.state === "running" ? [{ id: 1, cat: "1", pct: (FW.f % 30) / 30, mbps: 0 }] : [];
  if (r.t >= r.end) { FW.replay = null; FW.data = FW.saved; toast("Replay done — back to live"); poll(); }
}

// --- open / close -------------------------------------------------------------------------------
function openFloor() {
  const cv = document.getElementById("floor-canvas"); if (!cv) return;
  const kiosk = /kiosk=1/.test(location.hash);
  document.body.classList.toggle("kiosk", kiosk);
  stopFloor(true);
  if (!FE.canvas) FE.init(cv); else FE.resize();
  FW.lanes = FW.lanes || {};
  try { FE.cam.locked = localStorage.getItem("peaks_floor_cam") === "1"; FE.crt = +(localStorage.getItem("peaks_floor_crt") ?? 2); } catch { /* defaults */ }
  poll(); pollDay();
  FW.poll = setInterval(() => { if (!document.hidden) poll(); }, 2000);
  FW.dpoll = setInterval(() => { if (!document.hidden) pollDay(); }, 60000);
  window.addEventListener("resize", onResize);
  FW.raf = requestAnimationFrame(frame);
  floorControls();
}
function onResize() { if (FE.canvas) FE.resize(); }
function stopFloor(keepKiosk) {
  cancelAnimationFrame(FW.raf); clearInterval(FW.poll); clearInterval(FW.dpoll);
  FW.raf = 0; FW.replay = null;
  window.removeEventListener("resize", onResize);
  if (window.FA) FA.hum(0);
  if (!keepKiosk) document.body.classList.remove("kiosk");
}

// --- the toolbar (shows on mouse move) and keys -----------------------------------------------
function floorControls() {
  const bar = document.getElementById("floor-bar"); if (!bar || bar.dataset.wired) return;
  bar.dataset.wired = "1";
  const sec = document.getElementById("floor");
  let hideAt = 0;
  const show = () => { bar.classList.add("on"); clearTimeout(hideAt); hideAt = setTimeout(() => bar.classList.remove("on"), 3000); };
  sec.addEventListener("mousemove", show);
  const label = () => {
    bar.querySelector('[data-a="cam"]').textContent = FE.cam.locked ? "camera: locked" : "camera: moving";
    bar.querySelector('[data-a="crt"]').textContent = "crt: " + ["off", "low", "full"][FE.crt];
    bar.querySelector('[data-a="sound"]').textContent = window.FA && FA.on ? "sound: on" : "sound: off";
  };
  label();
  const act = (a) => {
    if (a === "full") { location.hash = "#/floor?kiosk=1"; document.body.classList.add("kiosk"); (document.documentElement.requestFullscreen || (() => {})).call(document.documentElement); setTimeout(onResize, 300); }
    if (a === "replay") startReplay();
    if (a === "cam") { FE.cam.locked = !FE.cam.locked; try { localStorage.setItem("peaks_floor_cam", FE.cam.locked ? "1" : "0"); } catch { /* ok */ } }
    if (a === "crt") { FE.crt = (FE.crt + 1) % 3; try { localStorage.setItem("peaks_floor_crt", String(FE.crt)); } catch { /* ok */ } }
    if (a === "sound" && window.FA) FA.toggle();
    if (a === "back") { document.body.classList.remove("kiosk"); if (document.fullscreenElement) document.exitFullscreen(); go("foryou"); }
    label();
  };
  bar.addEventListener("click", (e) => { const b = e.target.closest("[data-a]"); if (b) act(b.dataset.a); });
  const vol = bar.querySelector("#floor-vol");
  if (vol && window.FA) { vol.value = FA.volume; vol.oninput = () => FA.setVolume(+vol.value); }
  document.addEventListener("keydown", (e) => {
    if (!document.getElementById("floor")?.classList.contains("active") || e.target.closest("input, textarea, select")) return;
    const k = e.key.toLowerCase();
    const map = { f: "full", r: "replay", m: "sound", c: "cam" };
    if (map[k]) { e.preventDefault(); act(map[k]); show(); }
    if (k === "escape" && document.body.classList.contains("kiosk")) { document.body.classList.remove("kiosk"); setTimeout(onResize, 100); }
  });
}

// opened straight on #/floor: app.js routed before this file loaded
if (/^#\/floor/.test(location.hash) && document.getElementById("floor")?.classList.contains("active")) openFloor();
