"""Minimal Stash GraphQL client.

Step 1 scope:
  - connection / version check
  - count + iterate scenes (paginated) with their files and existing markers
  - helper to find-or-create a tag and create a scene marker (used in step 3)

Auth: Stash uses an `ApiKey` request header. If your server has no
authentication configured, leave the key empty.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from .models import Scene, Tag


class StashError(RuntimeError):
    """Raised when Stash returns a GraphQL error or an unexpected response."""


# --- GraphQL documents -------------------------------------------------------

_VERSION_QUERY = """
query Version {
  version { version build_time hash }
}
"""

# A page of scenes with the fields we care about for sampling + marker reads.
_FIND_SCENES_QUERY = """
query FindScenes($filter: FindFilterType) {
  findScenes(filter: $filter) {
    count
    scenes {
      id
      title
      files {
        path
        duration
        width
        height
        frame_rate
        video_codec
        size
        fingerprints { type value }
      }
      scene_markers {
        id
        seconds
        end_seconds
        title
        primary_tag { id name }
      }
    }
  }
}
"""

_FIND_SCENE_MARKERS_QUERY = """
query FindSceneMarkers($filter: FindFilterType, $marker_filter: SceneMarkerFilterType) {
  findSceneMarkers(filter: $filter, scene_marker_filter: $marker_filter) {
    count
    scene_markers {
      id
      seconds
      end_seconds
      title
      scene { id }
      primary_tag { id name }
    }
  }
}
"""

_FIND_TAGS_QUERY = """
query FindTags($filter: FindFilterType, $tag_filter: TagFilterType) {
  findTags(filter: $filter, tag_filter: $tag_filter) {
    tags { id name }
  }
}
"""

_TAG_CREATE = """
mutation TagCreate($input: TagCreateInput!) {
  tagCreate(input: $input) { id name }
}
"""

_SCENE_MARKER_CREATE = """
mutation SceneMarkerCreate($input: SceneMarkerCreateInput!) {
  sceneMarkerCreate(input: $input) { id seconds end_seconds title }
}
"""

_SCENE_MARKERS_DESTROY = """
mutation SceneMarkersDestroy($ids: [ID!]!) {
  sceneMarkersDestroy(ids: $ids)
}
"""


class StashClient:
    # transient network errors are retried with these sleeps between attempts;
    # a multi-hour embed run pages the scene list lazily and shouldn't die at
    # hour five because one request hit a blip. GraphQL/HTTP errors (bad auth,
    # bad query) are NOT retried — those won't fix themselves.
    RETRY_SLEEPS: tuple[float, ...] = (1.0, 4.0, 10.0)

    def __init__(self, url: str, api_key: str = "", timeout: int = 30):
        self.base_url = url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["ApiKey"] = api_key

    # --- core ----------------------------------------------------------------

    def execute(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        for attempt, sleep in enumerate((*self.RETRY_SLEEPS, None)):
            try:
                resp = self.session.post(
                    self.graphql_url,
                    json={"query": query, "variables": variables or {}},
                    timeout=self.timeout,
                )
                break
            except requests.RequestException as exc:
                if sleep is None:
                    raise StashError(
                        f"Could not reach Stash at {self.graphql_url} "
                        f"(after {attempt + 1} attempts): {exc}"
                    ) from exc
                time.sleep(sleep)

        if resp.status_code != 200:
            raise StashError(
                f"Stash returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        payload = resp.json()
        if payload.get("errors"):
            raise StashError(f"GraphQL error: {payload['errors']}")
        return payload["data"]

    # --- reads ---------------------------------------------------------------

    def version(self) -> dict:
        return self.execute(_VERSION_QUERY)["version"]

    def scene_count(self) -> int:
        data = self.execute(
            _FIND_SCENES_QUERY,
            {"filter": {"per_page": 0}},
        )
        return data["findScenes"]["count"]

    @staticmethod
    def _under_path(scene: Scene, prefix: str) -> bool:
        """True if the scene's file is inside `prefix` (folder, incl. subdirs).
        Client-side so it works regardless of Stash's filter schema."""
        if not prefix:
            return True
        p = scene.path
        if not p:
            return False
        base = prefix.rstrip("/")
        return p == base or p.startswith(base + "/")

    def iter_scenes(
        self, page_size: int = 100, path_prefix: str = ""
    ) -> Iterator[Scene]:
        """Yield every scene, transparently paginating. When `path_prefix` is
        set, only scenes whose file lives under that folder are yielded."""
        page = 1
        seen = 0
        while True:
            data = self.execute(
                _FIND_SCENES_QUERY,
                {
                    "filter": {
                        "per_page": page_size,
                        "page": page,
                        "sort": "id",
                        "direction": "ASC",
                    }
                },
            )
            result = data["findScenes"]
            scenes = result["scenes"]
            for s in scenes:
                scene = Scene.from_dict(s)
                if self._under_path(scene, path_prefix):
                    yield scene
            seen += len(scenes)
            if not scenes or seen >= result["count"]:
                break
            page += 1

    def iter_markers_by_tag(
        self, tag_name: str, page_size: int = 200
    ) -> Iterator[dict]:
        """Yield scene markers whose tags include `tag_name`.

        Each yielded dict: {marker_id, scene_id, seconds, end_seconds, title}.
        Returns nothing if the tag doesn't exist yet.
        """
        tag = self.find_tag_by_name(tag_name)
        if tag is None:
            return
        marker_filter = {
            "tags": {"value": [tag.id], "modifier": "INCLUDES", "depth": 0}
        }
        page = 1
        seen = 0
        while True:
            data = self.execute(
                _FIND_SCENE_MARKERS_QUERY,
                {
                    "filter": {
                        "per_page": page_size,
                        "page": page,
                        "sort": "scene_id",
                        "direction": "ASC",
                    },
                    "marker_filter": marker_filter,
                },
            )
            result = data["findSceneMarkers"]
            markers = result["scene_markers"]
            for m in markers:
                scene = m.get("scene") or {}
                pt = m.get("primary_tag") or {}
                yield {
                    "marker_id": str(m["id"]),
                    "scene_id": str(scene.get("id")) if scene.get("id") else None,
                    "seconds": float(m.get("seconds") or 0.0),
                    "end_seconds": (
                        float(m["end_seconds"]) if m.get("end_seconds") else None
                    ),
                    "title": m.get("title", ""),
                    "primary_tag": pt.get("name", ""),
                }
            seen += len(markers)
            if not markers or seen >= result["count"]:
                break
            page += 1

    # --- writes (used in step 3) --------------------------------------------

    def find_tag_by_name(self, name: str) -> Tag | None:
        data = self.execute(
            _FIND_TAGS_QUERY,
            {
                "filter": {"per_page": 1},
                "tag_filter": {"name": {"value": name, "modifier": "EQUALS"}},
            },
        )
        tags = data["findTags"]["tags"]
        return Tag.from_dict(tags[0]) if tags else None

    def find_or_create_tag(self, name: str) -> Tag:
        existing = self.find_tag_by_name(name)
        if existing:
            return existing
        data = self.execute(_TAG_CREATE, {"input": {"name": name}})
        return Tag.from_dict(data["tagCreate"])

    def create_scene_marker(
        self,
        scene_id: str,
        seconds: float,
        primary_tag_id: str,
        title: str = "",
        end_seconds: float | None = None,
    ) -> dict:
        """Create a marker on a scene. `end_seconds` makes it a ranged marker."""
        input_obj: dict[str, Any] = {
            "scene_id": scene_id,
            "seconds": seconds,
            "primary_tag_id": primary_tag_id,
            "title": title,
        }
        if end_seconds is not None:
            input_obj["end_seconds"] = end_seconds
        data = self.execute(_SCENE_MARKER_CREATE, {"input": input_obj})
        return data["sceneMarkerCreate"]

    def destroy_scene_markers(self, marker_ids: list[str], chunk: int = 100) -> int:
        """Delete markers by id (chunked). Returns how many ids were submitted."""
        for i in range(0, len(marker_ids), chunk):
            self.execute(
                _SCENE_MARKERS_DESTROY, {"ids": marker_ids[i : i + chunk]}
            )
        return len(marker_ids)

    # --- playback helpers (used by the megaboard) ---------------------------

    def stream_url(self, scene_id: str, start: float | None = None) -> str:
        """Direct-stream URL for a scene, optionally seeked to `start` seconds.

        The megaboard points <video> tiles at these. `apikey` is appended as a
        query param because <video> tags can't send the ApiKey header.
        """
        url = f"{self.base_url}/scene/{scene_id}/stream"
        params = []
        if start is not None:
            params.append(f"start={start:g}")
        api_key = self.session.headers.get("ApiKey")
        if api_key:
            params.append(f"apikey={api_key}")
        if params:
            url += "?" + "&".join(params)
        return url

    @classmethod
    def from_config(cls, cfg) -> "StashClient":
        return cls(url=cfg.stash.url, api_key=cfg.stash.api_key, timeout=cfg.stash.timeout)
