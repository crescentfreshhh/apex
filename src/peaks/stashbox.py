"""StashDB (any stash-box) — read-only: the scenes a performer or studio has
released. The endpoint and API key come from your own Stash's configuration
(Settings → Metadata Providers → Stash-box), so Peaks needs no key of its own.
"""

from __future__ import annotations

import requests

SCENE_FIELDS = """id title release_date duration
  studio { id name }
  performers { performer { id name gender } }
  images { url width }
  urls { url }"""

_QUERY = """query Q($input: SceneQueryInput!) {
  queryScenes(input: $input) { count scenes { %s } }
}"""


class StashBoxError(RuntimeError):
    pass


def stash_boxes(stash) -> list[dict]:
    """[{endpoint, api_key, name}] configured in your Stash."""
    d = stash.execute("query { configuration { general { stashBoxes { endpoint api_key name } } } }")
    return [b for b in (d.get("configuration") or {}).get("general", {}).get("stashBoxes") or []
            if b.get("endpoint") and b.get("api_key")]


def pick_box(boxes: list[dict]) -> dict | None:
    """StashDB if it's there, else the first stash-box."""
    return next((b for b in boxes if "stashdb.org" in b["endpoint"]), boxes[0] if boxes else None)


class StashBoxClient:
    def __init__(self, endpoint: str, api_key: str, timeout: float = 30.0, session=None):
        self.endpoint, self.timeout = endpoint, timeout
        self.s = session or requests.Session()
        self.s.headers.update({"ApiKey": api_key, "Content-Type": "application/json"})

    def _q(self, query: str, variables: dict) -> dict:
        try:
            r = self.s.post(self.endpoint, json={"query": query, "variables": variables}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise StashBoxError(f"can't reach {self.endpoint}: {exc}") from exc
        if r.status_code in (401, 403):
            raise StashBoxError("the stash-box refused the API key in your Stash settings")
        r.raise_for_status()
        d = r.json()
        if d.get("errors"):
            raise StashBoxError("; ".join(e.get("message", "") for e in d["errors"])[:300])
        return d["data"]

    def scenes(self, performer: str | None = None, studio: str | None = None,
               per_page: int = 40, page: int = 1) -> list[dict]:
        """Newest first: a performer's scenes, or a studio's."""
        inp: dict = {"page": page, "per_page": per_page, "sort": "DATE", "direction": "DESC"}
        if performer:
            inp["performers"] = {"value": [performer], "modifier": "INCLUDES"}
        if studio:
            inp["studios"] = {"value": [studio], "modifier": "INCLUDES"}
        data = self._q(_QUERY % SCENE_FIELDS, {"input": inp})
        return [scene_row(s) for s in (data.get("queryScenes") or {}).get("scenes") or []]


def scene_row(s: dict) -> dict:
    """A stash-box scene as Peaks keeps it."""
    imgs = sorted(s.get("images") or [], key=lambda i: -(i.get("width") or 0))
    return {"id": s["id"], "title": s.get("title") or "", "date": s.get("release_date") or "",
            "duration": s.get("duration"), "studio": (s.get("studio") or {}).get("name") or "",
            "studio_id": (s.get("studio") or {}).get("id"),
            "performers": [p["performer"]["name"] for p in s.get("performers") or [] if p.get("performer")],
            "performer_ids": [p["performer"]["id"] for p in s.get("performers") or [] if p.get("performer")],
            "image": imgs[0]["url"] if imgs else None,
            "url": next((u["url"] for u in s.get("urls") or [] if u.get("url")), None)}
