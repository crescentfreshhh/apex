// Peaks service worker — what makes Peaks an installable phone app.
// It only looks after the app shell (the page, its scripts, styles and icons):
// always fresh from Peaks when reachable, the last copy when not, so the app
// opens and says why instead of Chrome's dinosaur. It never touches the API,
// video streams (those come from Stash) or the megaboard's media.
const CACHE = "peaks-shell-v3";
const SHELL = ["/", "/static/app.js", "/static/floor/engine.js", "/static/floor/audio.js", "/static/floor/world.js", "/static/words.js", "/static/style.css", "/static/ui.css",
  "/static/peaks.svg", "/static/icons/icon-192.png", "/megaboard/", "/megaboard/megaboard.js",
  "/megaboard/megaboard.css"];

self.addEventListener("install", (e) => {
  self.skipWaiting();
  // best effort: behind the password gate these are only cached once signed in
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(SHELL.map((u) =>
    fetch(u, { credentials: "same-origin" }).then((r) => (r.ok && !r.redirected ? c.put(u, r) : null)).catch(() => null)))));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

const OFFLINE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Peaks — offline</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0b0b0d;color:#ececf1;font:15px/1.5 system-ui,sans-serif;text-align:center;padding:24px}
b{color:#d4ae55;font-size:24px}button{margin-top:18px;background:#d4ae55;color:#1b1505;border:0;
border-radius:9px;padding:11px 22px;font:inherit;font-weight:600}</style></head><body><div>
<b>Peaks</b><p>Peaks isn't reachable right now.<br>Are you on your home Wi-Fi or Tailscale?</p>
<button onclick="location.reload()">Try again</button></div></body></html>`;

function isShell(url) {
  if (url.origin !== self.location.origin) return false;
  const p = url.pathname;
  if (p.startsWith("/api/") || p === "/sw.js") return false;
  return p === "/" || p.startsWith("/static/") || p === "/megaboard/" || /^\/megaboard\/[\w.-]+\.(js|css|svg|html)$/.test(p);
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const nav = req.mode === "navigate" && url.origin === self.location.origin;
  if (!nav && !isShell(url)) return;                 // API, streams, Stash: straight through
  e.respondWith((async () => {
    try {
      const r = await fetch(req);
      // keep a copy of good shell responses (not the sign-in page, not errors)
      const login = (r.headers.get("content-type") || "").includes("text/html") && url.pathname.startsWith("/static/");
      if (r.ok && !login && isShell(url)) {
        const copy = r.clone();
        caches.open(CACHE).then((c) => c.put(url.pathname === "/" ? "/" : req, copy)).catch(() => {});
      }
      return r;
    } catch {
      const hit = await caches.match(nav && url.pathname === "/" ? "/" : req, { ignoreSearch: true });
      if (hit) return hit;
      if (nav) return new Response(OFFLINE, { headers: { "content-type": "text/html; charset=utf-8" } });
      return Response.error();
    }
  })());
});
