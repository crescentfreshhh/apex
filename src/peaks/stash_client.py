"""Minimal Stash GraphQL client.

Step 1 scope:
  - connection / version check
  - count + iterate scenes (paginated) with their files and existing markers
  - helper to find-or-create a tag and create a scene marker (used in step 3)

Auth: Stash uses an `ApiKey` request header. If your server has no
authentication configured, leave the key empty.
"""

from __future__ import annotations

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


class StashClient:
    def __init__(self, url: str, api_key: str = "", timeout: int = 30):
        self.base_url = url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["ApiKey"] = api_key

    # --- core ----------------------------------------------------------------

    def execute(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        try:
            resp = self.session.post(
                self.graphql_url,
                json={"query": query, "variables": variables or {}},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise StashError(f"Could not reach Stash at {self.graphql_url}: {exc}") from exc

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

    def iter_scenes(self, page_size: int = 100) -> Iterator[Scene]:
        """Yield every scene, transparently paginating."""
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
                yield Scene.from_dict(s)
            seen += len(scenes)
            if not scenes or seen >= result["count"]:
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

    @classmethod
    def from_config(cls, cfg) -> "StashClient":
        return cls(url=cfg.stash.url, api_key=cfg.stash.api_key, timeout=cfg.stash.timeout)
