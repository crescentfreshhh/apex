// The control room's engine: a canvas that draws characters (ASCII first,
// Braille for graphs) from a cached glyph atlas, a 2.5D world with a gentle
// row perspective and a camera, light pools, particles, and a CRT finish
// (bloom, scanlines, vignette, phosphor afterglow, flicker).
//
// Layers per frame:
//   static world (floor, machine bodies) -> cached canvas, redrawn on resize / lighting change
//   dynamic world (animations, crates, particles, readouts) -> world canvas
//   world canvas -> screen through the camera (pan + zoom)
//   screen-space panels (clock, gauges, sparklines, ticker)
//   post: bloom (blurred copy, added), afterglow, scanlines, vignette, flicker
"use strict";

const FE = (() => {
  const E = {
    canvas: null, ctx: null, dpr: 1, W: 0, H: 0,          // screen (device px)
    cw: 8, ch: 16, font: "",                               // a screen cell
    WCOLS: 160, WROWS: 52,                                 // the world grid
    world: null, wctx: null, stat: null, sctx: null, statKey: "",
    glow: null, gctx: null, trail: null, tctx: null,
    q: 1.25,                                               // world canvas oversampling (crisp when the camera zooms)
    cam: { x: 80, y: 26, z: 1, tx: 80, ty: 26, tz: 1, locked: false },
    view: { x: 0, y: 0, w: 0, h: 0 },                      // the world's rectangle on screen
    dprCap: 2,                                             // lowered by lighten() on slow machines
    crt: 2,                                                // 0 off · 1 low · 2 full
    particles: [],
    atlas: null, actx: null, slots: new Map(), next: 0, perRow: 0,
  };

  // --- glyph atlas: any character, any colour, drawn once ---------------------------------
  function resetAtlas() {
    E.atlas = document.createElement("canvas");
    E.atlas.width = 2048; E.atlas.height = 2048;
    E.actx = E.atlas.getContext("2d");
    E.slots.clear(); E.next = 0;
    E.perRow = Math.floor(2048 / E.cw);
  }
  function glyph(ch, color, bold) {
    const key = ch + "\u0000" + color + (bold ? "b" : "");
    let s = E.slots.get(key);
    if (s) return s;
    const i = E.next++;
    const rows = Math.floor(2048 / E.ch);
    if (i >= E.perRow * rows) { resetAtlas(); return glyph(ch, color, bold); }   // full: start over
    const sx = (i % E.perRow) * E.cw, sy = Math.floor(i / E.perRow) * E.ch;
    const a = E.actx;
    a.font = (bold ? "700 " : "500 ") + E.font;
    a.textBaseline = "middle"; a.textAlign = "center";
    a.fillStyle = color;
    a.fillText(ch, sx + E.cw / 2, sy + E.ch / 2 + 0.5);
    s = { sx, sy };
    E.slots.set(key, s);
    return s;
  }

  // --- sizing -----------------------------------------------------------------------------
  function resize() {
    const c = E.canvas;
    E.dpr = Math.min(E.dprCap, window.devicePixelRatio || 1);
    const r = c.getBoundingClientRect();
    E.W = Math.max(320, Math.floor(r.width * E.dpr));
    E.H = Math.max(240, Math.floor(r.height * E.dpr));
    c.width = E.W; c.height = E.H;
    // ~236 cells across the screen: big enough to read on 1080p, sharp on 4K
    E.cw = Math.max(5, Math.floor(E.W / 236));
    E.ch = Math.round(E.cw * 1.85);
    E.font = `${Math.round(E.ch * 0.78)}px ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, Consolas, monospace`;
    resetAtlas();
    // the world: base size in its own canvas (oversampled), fitted between the panels
    const ww = Math.ceil(E.WCOLS * E.cw * E.q), wh = Math.ceil(rowY(E.WROWS) * E.q) + 4;
    for (const k of ["world", "stat", "trail"]) {
      E[k] = document.createElement("canvas");
      E[k].width = ww; E[k].height = wh;
    }
    E.wctx = E.world.getContext("2d"); E.sctx = E.stat.getContext("2d"); E.tctx = E.trail.getContext("2d");
    E.glow = document.createElement("canvas");
    E.glow.width = Math.ceil(E.W / (4 * E.dpr)); E.glow.height = Math.ceil(E.H / (4 * E.dpr));
    E.gctx = E.glow.getContext("2d");
    E.statKey = "";
    E.scan = null;
  }

  // --- the 2.5D world: rows further back are a little smaller and closer together ---------
  const S0 = 0.80, S1 = 0.20;                 // scale at the back row, extra at the front
  function rowScale(y) { return S0 + S1 * (y / E.WROWS); }
  function rowY(y) { return E.ch * (S0 * y + (S1 / 2) * y * y / E.WROWS); }      // integral of the scale
  // world cell (x, y) -> world-canvas px (centre of the cell)
  function wp(x, y) {
    const s = rowScale(y);
    return { px: (E.WCOLS / 2 * E.cw + (x - E.WCOLS / 2) * E.cw * s + E.cw * s / 2) * E.q,
             py: (rowY(y) + E.ch * s / 2) * E.q, s: s * E.q };
  }
  // text into a world-layer context at world cell coordinates (fractional allowed)
  function putW(ctx, x, y, str, color, alpha = 1, bold = false) {
    if (alpha <= 0.01) return;
    const prev = ctx.globalAlpha;
    if (alpha !== 1) ctx.globalAlpha = prev * alpha;
    for (let i = 0; i < str.length; i++) {
      const c = str[i];
      if (c === " ") continue;
      const p = wp(x + i, y), g = glyph(c, color, bold);
      const w = E.cw * p.s, h = E.ch * p.s;
      ctx.drawImage(E.atlas, g.sx, g.sy, E.cw, E.ch, p.px - w / 2, p.py - h / 2, w, h);
    }
    ctx.globalAlpha = prev;
  }
  // text in screen space (panels), by screen cell
  function putS(col, row, str, color, alpha = 1, bold = false) {
    const ctx = E.ctx;
    const prev = ctx.globalAlpha;
    if (alpha !== 1) ctx.globalAlpha = prev * alpha;
    for (let i = 0; i < str.length; i++) {
      const c = str[i];
      if (c === " ") continue;
      const g = glyph(c, color, bold);
      ctx.drawImage(E.atlas, g.sx, g.sy, E.cw, E.ch, (col + i) * E.cw, row * E.ch, E.cw, E.ch);
    }
    ctx.globalAlpha = prev;
  }
  // a soft pool of light on the floor around a world point
  function light(ctx, x, y, radiusCells, color, a) {
    if (a <= 0.01) return;
    const p = wp(x, y), r = radiusCells * E.cw * p.s;
    const g = ctx.createRadialGradient(p.px, p.py, 0, p.px, p.py, r);
    g.addColorStop(0, color.replace("ALPHA", String(a)));
    g.addColorStop(1, color.replace("ALPHA", "0"));
    const prev = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(p.px, p.py, r, r * 0.55, 0, 0, Math.PI * 2); ctx.fill();
    ctx.globalCompositeOperation = prev;
  }

  // --- particles: ASCII sparks, steam, paper, dust -----------------------------------------
  function emit(x, y, n, opts) {
    for (let i = 0; i < n; i++) {
      E.particles.push({
        x, y,
        vx: (opts.vx ?? 0) + (Math.random() - 0.5) * (opts.spread ?? 1),
        vy: (opts.vy ?? -0.5) + (Math.random() - 0.5) * (opts.spread ?? 1),
        g: opts.g ?? 0.6, life: (opts.life ?? 1) * (0.6 + Math.random() * 0.8), age: 0,
        chars: opts.chars || "*+.", color: opts.color || "#ffd27a",
      });
    }
    if (E.particles.length > 900) E.particles.splice(0, E.particles.length - 900);
  }
  function stepParticles(ctx, dt) {
    const keep = [];
    for (const p of E.particles) {
      p.age += dt;
      if (p.age >= p.life) continue;
      p.vy += p.g * dt; p.x += p.vx * dt * 6; p.y += p.vy * dt * 6;
      const k = p.age / p.life, ch = p.chars[Math.min(p.chars.length - 1, Math.floor(k * p.chars.length))];
      putW(ctx, p.x, p.y, ch, p.color, 1 - k);
      keep.push(p);
    }
    E.particles = keep;
  }

  // --- camera -------------------------------------------------------------------------------
  function focus(x, y, z, hold = 3.5) {
    if (E.cam.locked) return;
    E.cam.tx = x; E.cam.ty = y; E.cam.tz = z;
    E.cam.until = performance.now() + hold * 1000;
  }
  function stepCamera(dt, t, idle) {
    const c = E.cam;
    if (c.locked || (c.until && t > c.until)) {
      c.until = 0;
      // home: the whole floor, drifting a little when nothing happens
      const drift = c.locked ? 0 : 1;
      c.tx = E.WCOLS / 2 + drift * Math.sin(t / 23000) * (idle ? 6 : 2.5);
      c.ty = E.WROWS / 2 + drift * Math.cos(t / 31000) * (idle ? 2.5 : 1);
      c.tz = 1 + drift * (idle ? 0.04 + 0.03 * Math.sin(t / 41000) : 0);
    }
    const k = 1 - Math.pow(0.0015, dt);          // frame-rate independent easing
    c.x += (c.tx - c.x) * k; c.y += (c.ty - c.y) * k; c.z += (c.tz - c.z) * k;
  }
  // world canvas -> screen rectangle, through the camera
  function blitWorld() {
    const v = E.view, ctx = E.ctx;
    const base = Math.min(v.w / E.world.width, v.h / E.world.height);
    const z = base * E.cam.z;
    const f = wp(E.cam.x, E.cam.y);
    const dx = v.x + v.w / 2 - f.px * z, dy = v.y + v.h / 2 - f.py * z;
    // keep the floor from sliding off its rectangle
    const minX = v.x + v.w - E.world.width * z, minY = v.y + v.h - E.world.height * z;
    const ox = E.world.width * z <= v.w ? v.x + (v.w - E.world.width * z) / 2 : Math.min(v.x, Math.max(minX, dx));
    const oy = E.world.height * z <= v.h ? v.y + (v.h - E.world.height * z) / 2 : Math.min(v.y, Math.max(minY, dy));
    ctx.save();
    ctx.beginPath(); ctx.rect(v.x, v.y, v.w, v.h); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(E.world, ox, oy, E.world.width * z, E.world.height * z);
    ctx.restore();
  }

  // --- CRT finish -----------------------------------------------------------------------------
  function post(t) {
    const ctx = E.ctx;
    if (E.crt > 0) {
      // bloom: a quarter-size blurred copy, added back on top
      const g = E.gctx;
      g.globalCompositeOperation = "copy";
      g.filter = "blur(3px)";
      g.drawImage(E.canvas, 0, 0, E.glow.width, E.glow.height);
      g.filter = "none";
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = E.crt === 2 ? 0.55 : 0.3;
      ctx.drawImage(E.glow, 0, 0, E.W, E.H);
      ctx.restore();
      // scanlines
      if (!E.scan) {
        const s = document.createElement("canvas");
        s.width = 4; s.height = Math.max(2, Math.round(3 * E.dpr));
        const sc = s.getContext("2d");
        sc.fillStyle = "rgba(0,0,0,0.28)"; sc.fillRect(0, s.height - Math.max(1, Math.round(E.dpr)), 4, Math.max(1, Math.round(E.dpr)));
        E.scan = ctx.createPattern(s, "repeat");
      }
      ctx.fillStyle = E.scan;
      ctx.globalAlpha = E.crt === 2 ? 1 : 0.5;
      ctx.fillRect(0, 0, E.W, E.H);
      ctx.globalAlpha = 1;
    }
    // vignette (always a little: it frames the room)
    const v = ctx.createRadialGradient(E.W / 2, E.H / 2, Math.min(E.W, E.H) * 0.35, E.W / 2, E.H / 2, Math.max(E.W, E.H) * 0.75);
    v.addColorStop(0, "rgba(0,0,0,0)");
    v.addColorStop(1, `rgba(0,0,0,${E.crt ? 0.55 : 0.3})`);
    ctx.fillStyle = v; ctx.fillRect(0, 0, E.W, E.H);
    if (E.crt === 2) {                       // the faintest flicker
      const fl = 0.025 * (Math.sin(t / 37) * 0.5 + 0.5) * (Math.random() < 0.08 ? 2 : 1);
      ctx.fillStyle = `rgba(0,0,0,${fl})`; ctx.fillRect(0, 0, E.W, E.H);
    }
  }

  // phosphor afterglow: moving things leave a fading trail (world layer)
  function fadeTrail(dt) {
    const t = E.tctx;
    t.globalCompositeOperation = "destination-out";
    t.fillStyle = `rgba(0,0,0,${Math.min(1, dt * 7)})`;
    t.fillRect(0, 0, E.trail.width, E.trail.height);
    t.globalCompositeOperation = "source-over";
  }

  // one step lighter: a softer world canvas first, then fewer screen pixels (false = nothing left)
  function lighten() {
    if (E.q > 0.75) E.q -= 0.25;
    else if (E.dpr > 1) E.dprCap = Math.max(1, E.dpr - 0.5);
    else return false;
    resize();
    return true;
  }

  function init(canvas) {
    E.canvas = canvas;
    E.ctx = canvas.getContext("2d", { alpha: false });
    resize();
  }
  return Object.assign(E, { init, resize, glyph, putW, putS, light, wp, rowScale, emit, stepParticles, focus,
                            stepCamera, blitWorld, post, fadeTrail, lighten });
})();
