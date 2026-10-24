// The floor's ambience — synthesised with WebAudio, no sound files. Off by default;
// on/off and volume are remembered in this browser. Every sound is a few oscillators
// and a filtered noise burst, so it costs next to nothing and never surprises anyone.
"use strict";

window.FA = (function () {
  const A = { on: false, volume: 0.5, ctx: null, master: null, humGain: null, humN: 0, last: {} };
  try {
    A.on = localStorage.getItem("peaks_floor_sound") === "1";
    const v = parseFloat(localStorage.getItem("peaks_floor_vol"));
    if (!isNaN(v)) A.volume = Math.max(0, Math.min(1, v));
  } catch { /* defaults */ }

  // browsers only allow audio after a gesture: built lazily from toggle()/a click
  function ensure() {
    if (A.ctx) { if (A.ctx.state === "suspended") A.ctx.resume(); return true; }
    const C = window.AudioContext || window.webkitAudioContext; if (!C) return false;
    A.ctx = new C();
    A.master = A.ctx.createGain(); A.master.gain.value = A.volume * 0.6; A.master.connect(A.ctx.destination);
    // the hum: two detuned low sines through a lowpass, with a slow wobble
    const lp = A.ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 220;
    A.humGain = A.ctx.createGain(); A.humGain.gain.value = 0;
    for (const f of [55, 55.7, 110.3]) {
      const o = A.ctx.createOscillator(); o.type = f > 100 ? "triangle" : "sine"; o.frequency.value = f;
      const g = A.ctx.createGain(); g.gain.value = f > 100 ? 0.12 : 0.4; o.connect(g); g.connect(lp); o.start();
    }
    const lfo = A.ctx.createOscillator(); lfo.frequency.value = 0.15;
    const lg = A.ctx.createGain(); lg.gain.value = 40; lfo.connect(lg); lg.connect(lp.frequency); lfo.start();
    lp.connect(A.humGain); A.humGain.connect(A.master);
    return true;
  }
  const ready = () => A.on && A.ctx && A.ctx.state === "running";
  // the same sound at most every `gap` ms, so a burst of events is one sound
  function rate(name, gap) { const t = performance.now(); if (t - (A.last[name] || 0) < gap) return false; A.last[name] = t; return true; }

  function noise(dur) {
    const n = Math.floor(A.ctx.sampleRate * dur), b = A.ctx.createBuffer(1, n, A.ctx.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    const s = A.ctx.createBufferSource(); s.buffer = b; return s;
  }
  function env(node, t, peak, attack, decay) {
    const g = A.ctx.createGain(); g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + attack); g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
    node.connect(g); g.connect(A.master); return g;
  }

  A.hum = function (n) {
    A.humN = n;
    if (!A.humGain) return;
    const target = A.on ? Math.min(0.35, n ? 0.12 + 0.05 * n : 0.03) : 0;
    A.humGain.gain.setTargetAtTime(target, A.ctx.currentTime, 1.2);
  };
  A.click = function () {
    if (!ready() || !rate("click", 120)) return;
    const t = A.ctx.currentTime, s = noise(0.03), bp = A.ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = 2400 + Math.random() * 800; bp.Q.value = 4;
    s.connect(bp); env(bp, t, 0.25, 0.002, 0.03); s.start(t);
  };
  A.ping = function () {
    if (!ready() || !rate("ping", 1500)) return;
    const t = A.ctx.currentTime;
    for (const [f, d, p] of [[1320, 0, 0.18], [1320, 0.35, 0.07], [1320, 0.7, 0.03]]) {
      const o = A.ctx.createOscillator(); o.type = "sine"; o.frequency.value = f;
      env(o, t + d, p, 0.005, 0.9); o.start(t + d); o.stop(t + d + 1);
    }
  };
  A.thud = function () {
    if (!ready() || !rate("thud", 400)) return;
    const t = A.ctx.currentTime, o = A.ctx.createOscillator();
    o.type = "sine"; o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(38, t + 0.35);
    env(o, t, 0.6, 0.004, 0.45); o.start(t); o.stop(t + 0.5);
    const s = noise(0.15), lp = A.ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 500;
    s.connect(lp); env(lp, t, 0.2, 0.002, 0.12); s.start(t);
  };
  A.printer = function () {
    if (!ready() || !rate("printer", 5000)) return;
    const t0 = A.ctx.currentTime;
    for (let i = 0; i < 26; i++) {
      const t = t0 + i * 0.07 + (i % 6 === 5 ? 0.12 : 0), s = noise(0.025), bp = A.ctx.createBiquadFilter();
      bp.type = "bandpass"; bp.frequency.value = 1700 + (i % 3) * 300; bp.Q.value = 6;
      s.connect(bp); env(bp, t, 0.12, 0.002, 0.03); s.start(t);
    }
  };
  A.toggle = function () {
    A.on = !A.on;
    try { localStorage.setItem("peaks_floor_sound", A.on ? "1" : "0"); } catch { /* ok */ }
    if (A.on && !ensure()) { A.on = false; return; }
    A.hum(A.humN);
  };
  A.setVolume = function (v) {
    A.volume = Math.max(0, Math.min(1, v));
    try { localStorage.setItem("peaks_floor_vol", String(A.volume)); } catch { /* ok */ }
    if (A.master) A.master.gain.setTargetAtTime(A.volume * 0.6, A.ctx.currentTime, 0.05);
  };
  // remembered "on": the context still needs a gesture, so wake it on the first click/key
  if (A.on) {
    const wake = () => { ensure(); A.hum(A.humN); removeEventListener("pointerdown", wake); removeEventListener("keydown", wake); };
    addEventListener("pointerdown", wake); addEventListener("keydown", wake);
  }
  return A;
})();
