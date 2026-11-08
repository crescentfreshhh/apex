"""Stash tells Peaks when its library changes (GraphQL subscriptions).

Peaks used to learn about new scenes only by asking — one light id query every
30 minutes while idle, every 5 while pages are open. Stash's GraphQL server
also pushes events over a websocket. This listener keeps one connection open
and subscribes to the scan finishing (`scanCompleteSubscribe`) and to Stash's
job list (`jobsSubscribe`: a Scan / Identify / Auto Tag / Clean finishing).
When one arrives, `on_event(reason)` — the service re-reads the scene list in
the background, and the scheduler and open pages hear about it.

Both websocket protocols Stash has spoken are handled
(`graphql-transport-ws`, and the older `graphql-ws`); the API key goes in the
upgrade headers and the init payload. A dropped connection is retried with
back-off (5 s → 5 min). Nothing depends on it: the periodic check stays as the
fallback. Stash has no "a scene was edited" subscription, so grades changed in
Stash's own pages still arrive with that check (or Catalogue → Re-read).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable

SCAN_SUB = "subscription { scanCompleteSubscribe }"
JOBS_SUB = "subscription { jobsSubscribe { type job { id status description } } }"
LIBRARY_JOBS = ("scan", "identify", "auto tag", "autotag", "clean", "import")   # not previews / sprites: the list is the same


def ws_url(graphql_url: str) -> str:
    if graphql_url.startswith("https://"):
        return "wss://" + graphql_url[len("https://"):]
    if graphql_url.startswith("http://"):
        return "ws://" + graphql_url[len("http://"):]
    return graphql_url


def library_job_done(payload: dict) -> str | None:
    """A jobsSubscribe update that means "the library changed": a finished
    Scan / Identify / Auto Tag / Clean. Returns a short reason, or None."""
    upd = ((payload or {}).get("data") or {}).get("jobsSubscribe") or {}
    job = upd.get("job") or {}
    desc = str(job.get("description") or "").lower()
    if str(job.get("status") or "").upper() in ("FINISHED",) or upd.get("type") == "REMOVE":
        if any(w in desc for w in LIBRARY_JOBS):
            return f"stash job finished: {job.get('description') or 'library job'}"
    return None


class StashEvents:
    BACKOFF = (5, 15, 30, 60, 120, 300)

    def __init__(self, graphql_url: str, api_key: str, on_event: Callable[[str], None],
                 log: Callable[[str], None] | None = None, subs: tuple[str, ...] = ("scan", "jobs")):
        self.url = ws_url(graphql_url)
        self.api_key = api_key or ""
        self.on_event = on_event
        self.log = log or (lambda *_: None)
        self.subs = subs
        self.status = {"connected": False, "since": None, "events": 0, "last": None, "error": None,
                       "protocol": None}
        self._stop = threading.Event()
        self._ws = None

    # --- lifecycle --------------------------------------------------------------------------

    def start(self) -> "StashEvents":
        threading.Thread(target=self._loop, daemon=True, name="peaks-stash-events").start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        tries = 0
        while not self._stop.is_set():
            try:
                self._session()
                tries = 0                          # it ran (then dropped): start the back-off over
            except Exception as exc:  # noqa: BLE001 — Stash down, restarting, an old version…
                self.status.update(connected=False, error=f"{type(exc).__name__}: {exc}"[:200])
            if self._stop.is_set():
                return
            wait = self.BACKOFF[min(tries, len(self.BACKOFF) - 1)]
            tries += 1
            self._stop.wait(wait)

    # --- one connection -----------------------------------------------------------------------

    def _session(self) -> None:
        from websockets.sync.client import connect

        headers = {"ApiKey": self.api_key} if self.api_key else {}
        with connect(self.url, additional_headers=headers, open_timeout=10, close_timeout=2,
                     subprotocols=["graphql-transport-ws", "graphql-ws"]) as ws:
            self._ws = ws
            proto = ws.subprotocol or "graphql-ws"
            modern = proto == "graphql-transport-ws"
            ws.send(json.dumps({"type": "connection_init", "payload": {"ApiKey": self.api_key} if self.api_key else {}}))
            deadline = time.time() + 10
            while True:                                         # wait for the ack
                msg = json.loads(ws.recv(timeout=max(0.1, deadline - time.time())))
                if msg.get("type") == "connection_ack":
                    break
                if msg.get("type") in ("connection_error", "error"):
                    raise ConnectionError(str(msg.get("payload"))[:200])
            queries = {"scan": SCAN_SUB, "jobs": JOBS_SUB}
            for i, name in enumerate(self.subs, start=1):
                ws.send(json.dumps({"id": str(i), "type": "subscribe" if modern else "start",
                                    "payload": {"query": queries[name]}}))
            names = {str(i): n for i, n in enumerate(self.subs, start=1)}
            self.status.update(connected=True, since=time.time(), error=None, protocol=proto)
            self.log(f"listening to Stash ({proto})")
            while not self._stop.is_set():
                try:
                    raw = ws.recv(timeout=30)
                except TimeoutError:
                    continue
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "ping":
                    ws.send(json.dumps({"type": "pong"}))
                    continue
                if t in ("ka", "pong", "complete"):
                    continue
                if t in ("error", "connection_error"):
                    if names.get(str(msg.get("id"))) == "jobs":
                        continue                                # an older Stash without jobsSubscribe: scans only
                    raise ConnectionError(str(msg.get("payload"))[:200])
                if t not in ("next", "data"):
                    continue
                name = names.get(str(msg.get("id")))
                reason = "stash scan finished" if name == "scan" else library_job_done(msg.get("payload") or {})
                if reason:
                    self.status.update(events=self.status["events"] + 1, last=time.time())
                    try:
                        self.on_event(reason)
                    except Exception:  # noqa: BLE001
                        pass
        self.status.update(connected=False)
