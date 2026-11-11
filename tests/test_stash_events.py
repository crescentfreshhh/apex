"""Stash tells Peaks when its library changes: one websocket, subscribed to the
scan finishing and to Stash's jobs; an event re-reads the scene list in the
background (only the new scenes' details), the scheduler and the open pages
hear about it; a dropped connection comes back; no socket, no problem — the
periodic check stays."""

import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("websockets")


class FakeStashWS:
    """A tiny graphql-transport-ws server: acks, records subscriptions, and
    pushes whatever the test emits."""

    def __init__(self, proto="graphql-transport-ws", jobs_ok=True):
        from websockets.sync.server import serve

        self.proto, self.jobs_ok = proto, jobs_ok
        self.conns, self.subs, self.headers = [], [], []
        self._srv = serve(self._handler, "127.0.0.1", 0, subprotocols=[proto])
        self.port = self._srv.socket.getsockname()[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/graphql"

    def _handler(self, ws):
        self.headers.append({k.lower(): v for k, v in ws.request.headers.raw_items()})
        self.conns.append(ws)
        try:
            for raw in ws:
                m = json.loads(raw)
                if m["type"] == "connection_init":
                    self.init = m.get("payload")
                    ws.send(json.dumps({"type": "connection_ack"}))
                elif m["type"] in ("subscribe", "start"):
                    self.subs.append((m["id"], m["payload"]["query"]))
                    if "jobsSubscribe" in m["payload"]["query"] and not self.jobs_ok:
                        ws.send(json.dumps({"id": m["id"], "type": "error", "payload": [{"message": "unknown field"}]}))
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.conns.remove(ws)

    def wait_subscribed(self, n=2, timeout=5):
        t = time.time() + timeout
        while len(self.subs) < n and time.time() < t:
            time.sleep(0.02)
        return len(self.subs) >= n

    def emit(self, sub_id, data):
        t = "next" if self.proto == "graphql-transport-ws" else "data"
        for ws in list(self.conns):
            ws.send(json.dumps({"id": sub_id, "type": t, "payload": {"data": data}}))

    def drop(self):
        for ws in list(self.conns):
            ws.close()

    def close(self):
        self._srv.shutdown()


def test_the_listener_reports_a_finished_scan_and_library_jobs():
    from peaks.stash_events import StashEvents

    srv = FakeStashWS()
    got = []
    ev = StashEvents(srv.url, "secret", got.append).start()
    try:
        assert srv.wait_subscribed(2)
        assert srv.headers[0].get("apikey") == "secret" and srv.init == {"ApiKey": "secret"}
        assert ev.status["connected"] and ev.status["protocol"] == "graphql-transport-ws"
        srv.emit("1", {"scanCompleteSubscribe": True})
        srv.emit("2", {"jobsSubscribe": {"type": "UPDATE", "job": {"id": "9", "status": "RUNNING", "description": "Scanning..."}}})
        srv.emit("2", {"jobsSubscribe": {"type": "UPDATE", "job": {"id": "9", "status": "FINISHED", "description": "Identifying scenes"}}})
        srv.emit("2", {"jobsSubscribe": {"type": "UPDATE", "job": {"id": "8", "status": "FINISHED", "description": "Generating previews… "}}})
        t = time.time() + 5
        while len(got) < 2 and time.time() < t:
            time.sleep(0.02)
        time.sleep(0.3)
        assert got[0] == "stash scan finished" and "Identifying" in got[1]
        assert len(got) == 2                                    # a running scan and previews don't change the list
    finally:
        ev.stop()
        srv.close()


def test_an_old_stash_without_jobs_subscriptions_still_hears_scans_and_a_drop_reconnects(monkeypatch):
    from peaks.stash_events import StashEvents

    monkeypatch.setattr(StashEvents, "BACKOFF", (0.2,))
    srv = FakeStashWS(proto="graphql-ws", jobs_ok=False)
    got = []
    ev = StashEvents(srv.url, "", got.append).start()
    try:
        assert srv.wait_subscribed(2)
        srv.emit("1", {"scanCompleteSubscribe": True})
        t = time.time() + 5
        while not got and time.time() < t:
            time.sleep(0.02)
        assert got == ["stash scan finished"] and ev.status["protocol"] == "graphql-ws"
        srv.drop()                                              # Stash restarted
        assert srv.wait_subscribed(4, timeout=8)                # …and the listener came back
        srv.emit("1", {"scanCompleteSubscribe": True})
        t = time.time() + 5
        while len(got) < 2 and time.time() < t:
            time.sleep(0.02)
        assert len(got) == 2
    finally:
        ev.stop()
        srv.close()


def test_a_new_scene_appears_within_five_seconds_without_a_poll(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    svc._catalogue_all(refresh=True)
    asked = []
    real_ids, real_details = stash.all_scene_ids, stash.scene_details
    stash.all_scene_ids = lambda: asked.append("ids") or real_ids()
    stash.scene_details = lambda ids: asked.append(("details", sorted(ids))) or real_details(ids)
    srv = FakeStashWS()
    ver = svc.__dict__.get("_data_version", 0)
    woke = []
    svc._on_change = woke.append
    ev = svc.start_stash_events(url=srv.url, api_key="")
    try:
        assert srv.wait_subscribed(2)
        time.sleep(0.5)
        assert "ids" not in asked                               # connected: the library isn't polled
        asked.clear()                                           # (stray one-scene reads under load aren't polls)
        stash.s["new1"] = {"rating100": None, "o_counter": 0, "width": 1920, "height": 1080, "tag_ids": [],
                           "organized": False, "fingerprint": "fpnew1", "bit_rate": 5e6, "frame_rate": 30,
                           "video_codec": "h264", "date": "2025-02-02"}
        t0 = time.time()
        srv.emit("1", {"scanCompleteSubscribe": True})
        while time.time() - t0 < 5 and not any(r["scene_id"] == "new1" for r in svc._catalogue_all()):
            time.sleep(0.05)
        took = time.time() - t0
        assert any(r["scene_id"] == "new1" for r in svc._catalogue_all()) and took < 5
        assert asked.count("ids") == 1 and ("details", ["new1"]) in asked     # one light id list + the newcomer
        assert all(len(a[1]) <= 1 for a in asked if a != "ids")                # never a whole-library re-read
        assert "catalogue" in woke and svc.__dict__.get("_data_version", 0) > ver   # scheduler and open pages hear it
        assert svc.stash_events_status()["events"] == 1
    finally:
        ev.stop()
        srv.close()


def test_the_scene_list_changing_tells_open_pages(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from test_catalogue import _triage_svc

    svc, stash = _triage_svc(tmp_path, monkeypatch)
    v = svc.__dict__.get("_data_version", 0)
    svc._cat_changed()
    assert svc.__dict__["_data_version"] == v + 1
