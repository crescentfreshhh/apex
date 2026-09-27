// Plain words for Peaks' numbers — the browser half of src/peaks/words.py (keep
// the two identical: tests/test_words.py runs both and compares). Every judgment
// reads as a phrase first; `num()` wraps the exact figure, which Settings →
// Display → "Show exact numbers" can hide app-wide (megaboard included).
(function () {
  const TASTE_BANDS = [["standout", "Standout", "p99"], ["strong", "Strong", "p95"], ["good", "Good", "p90"],
    ["decent", "Decent", "p75"], ["soso", "So-so", "p50"]];
  const TASTE_HINT = { standout: "top 1% of your library", strong: "top 5%", good: "top 10%",
    decent: "top 25%", soso: "top half", weak: "bottom half" };
  function tasteBand(score, cuts) {
    if (score == null || !cuts) return null;
    for (const [key, word, q] of TASTE_BANDS) if (cuts[q] != null && score >= cuts[q]) return [key, word];
    return ["weak", "Weak"];
  }
  function confidence(conf) {
    if (conf >= 0.75) return "Almost certainly";
    if (conf >= 0.5) return "Probably";
    if (conf >= 0.35) return "Leaning";
    return "Hard to call";
  }
  function closeRunnerUp(conf, second) { return second != null && conf < 0.75 && second >= 0.6 * conf; }
  function keeperWord(k) {
    if (k == null || k >= 0.9) return null;
    if (k >= 0.65) return "Likely a keeper";
    if (k >= 0.35) return "Could go either way";
    return "Likely a reject";
  }
  function share(p) {
    if (p <= 0) return "none";
    if (p >= 1) return "all";
    for (const [cut, w] of [[0.1, "hardly any"], [0.25, "a few"], [0.4, "about a third"], [0.6, "about half"],
      [0.75, "about two-thirds"], [0.9, "most"]]) if (p < cut) return w;
    return "nearly all";
  }
  function verdict(n, shrunk, keep, base) {
    if (!n) return ["new to you", "new"];
    if (n < 3) return ["early days", "neutral"];
    const d = (shrunk != null ? shrunk : base) - base;
    if (keep != null && keep < 0.5) return ["usually a miss", "bad"];
    if (d >= 0.6) return ["a favourite", "good"];
    if (d >= 0.2) return ["a good bet", "good"];
    if (d > -0.2) return ["mixed", "neutral"];
    if (d > -0.6) return ["hit and miss", "neutral"];
    return ["usually a miss", "bad"];
  }
  function accuracy(x) { return x >= 0.8 ? "very reliable" : x >= 0.6 ? "usually right" : x >= 0.4 ? "often close" : "rough guesses"; }
  function withinOne(p) { return p >= 0.9 ? "almost always within a tier" : p >= 0.75 ? "usually within a tier" : "often more than a tier off"; }
  function gain(g) {
    const pts = g * 100;
    return pts >= 8 ? "helps a lot" : pts >= 3 ? "helps" : pts > 0 ? "helps a little" : "no clear help yet";
  }
  function auc(a) { return a >= 0.9 ? "Excellent" : a >= 0.8 ? "Good" : a >= 0.7 ? "Fair" : a >= 0.6 ? "Weak" : "No better than a coin"; }
  function lift(p, base) {
    if (!base) return "";
    const x = p / base;
    if (x < 1.15) return "no better than chance";
    return x < 3 ? `${x.toFixed(1)}× better than chance` : `${Math.round(x)}× better than chance`;
  }
  function howOften(p) { return p >= 0.75 ? "most of the time" : p >= 0.5 ? "more often than not" : p >= 0.3 ? "some of the time" : "rarely"; }
  function trend(d, eps = 0.02) { return d > eps ? "better than last time" : d < -eps ? "slipped since last time" : "about the same as last time"; }
  function age(days) {
    if (days == null) return null;
    if (days <= 0) return "added today";
    if (days === 1) return "added yesterday";
    if (days < 7) return "added this week";
    if (days < 45) { const w = Math.round(days / 7); return `added ${w} week${w !== 1 ? "s" : ""} ago`; }
    if (days < 540) return `added ${Math.round(days / 30)} months ago`;
    return `added ${(days / 365).toFixed(1)} years ago`;
  }
  function rank(pct) {
    if (pct < 0.1) return ["top", "a favourite"];
    if (pct < 0.3) return ["strong", "strong"];
    if (pct < 0.6) return ["mixed", "mixed"];
    return ["low", "rarely your taste"];
  }
  function clipLength(sim) { return sim >= 0.8 ? "very tight" : sim >= 0.6 ? "tight" : sim >= 0.45 ? "balanced" : "long"; }

  // --- exact figures: shown small beside the words, or hidden app-wide --------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function num(text) { return text == null || text === "" ? "" : ` <span class="num">${esc(text)}</span>`; }
  const pct = (x) => Math.round(x * 100) + "%";
  function numbersShown() { try { return localStorage.getItem("peaks_hide_numbers") !== "1"; } catch { return true; } }
  function setNumbersShown(on) {
    try { localStorage.setItem("peaks_hide_numbers", on ? "0" : "1"); } catch {}
    applyNumbers();
  }
  function applyNumbers() { document.documentElement.classList.toggle("hide-nums", !numbersShown()); }
  // the taste scale: your library's cutoffs for the scores cards, For You and the
  // megaboard show — fetched once per page, shared by everything that asks
  let scaleP = null;
  function tasteScale(force) {
    if (!scaleP || force) scaleP = fetch("/api/taste/scale").then((r) => (r.ok ? r.json() : null)).catch(() => null);
    return scaleP;
  }
  // "Strong 82%" as HTML — word first, the figure as a small number, meaning on hover
  function tasteHTML(score, cuts, { cls = "tw" } = {}) {
    if (score == null) return "";
    const b = tasteBand(score, cuts);
    if (!b) return `<span class="${cls}">${pct(score)}</span>`;
    return `<span class="${cls} tw-${b[0]}" title="Taste ${pct(score)} — ${TASTE_HINT[b[0]]}">${b[1]}${num(pct(score))}</span>`;
  }
  // how the floor slider reads: "Strong and up"
  function floorWord(v, cuts) {
    if (!v) return "off";
    const b = tasteBand(v, cuts);
    if (!b) return pct(v);
    return b[0] === "standout" ? "Standouts only" : b[0] === "weak" ? "Nearly everything" : `${b[1]} and up`;
  }
  applyNumbers();
  window.Words = { TASTE_HINT, tasteBand, confidence, closeRunnerUp, keeperWord, share, verdict, accuracy,
    withinOne, gain, auc, lift, howOften, trend, age, rank, clipLength, num, pct, numbersShown,
    setNumbersShown, applyNumbers, tasteScale, tasteHTML, floorWord };
})();
