"""Client logic tested against a fake GraphQL transport (no real Stash)."""

import pytest

from peaks.models import Scene
from peaks.stash_client import StashClient


class FakeClient(StashClient):
    """StashClient with `execute` replaced by canned responses + call capture."""

    def __init__(self, responses):
        super().__init__(url="http://stash.test:6969", api_key="secret")
        self._responses = list(responses)
        self.calls = []

    def execute(self, query, variables=None):
        self.calls.append((query, variables))
        return self._responses.pop(0)


def _scene(id_, *, markers=0, fp=True):
    return {
        "id": id_,
        "title": f"scene {id_}",
        "files": [
            {
                "path": f"/m/{id_}.mp4",
                "duration": 600.0,
                "width": 1920,
                "height": 1080,
                "frame_rate": 30.0,
                "video_codec": "h264",
                "size": 123,
                "fingerprints": ([{"type": "oshash", "value": f"hash{id_}"}] if fp else []),
            }
        ],
        "scene_markers": [
            {
                "id": f"{id_}m{j}",
                "seconds": 10.0 * j,
                "end_seconds": 10.0 * j + 5,
                "title": "x",
                "primary_tag": {"id": "1", "name": "apex"},
            }
            for j in range(markers)
        ],
    }


def _scene_at(id_, path):
    return {
        "id": id_,
        "title": f"scene {id_}",
        "files": [{"path": path, "duration": 600.0, "fingerprints": [
            {"type": "oshash", "value": f"hash{id_}"}]}],
        "scene_markers": [],
    }


def test_iter_scenes_paginates_until_count():
    page1 = {"findScenes": {"count": 3, "scenes": [_scene("1"), _scene("2")]}}
    page2 = {"findScenes": {"count": 3, "scenes": [_scene("3")]}}
    client = FakeClient([page1, page2])

    scenes = list(client.iter_scenes(page_size=2))
    assert [s.id for s in scenes] == ["1", "2", "3"]
    # two pages requested
    assert len(client.calls) == 2
    assert client.calls[0][1]["filter"]["page"] == 1
    assert client.calls[1][1]["filter"]["page"] == 2


def test_iter_scenes_path_filter_includes_subfolders():
    page = {
        "findScenes": {
            "count": 4,
            "scenes": [
                _scene_at("1", "/data/Rando/a.mp4"),
                _scene_at("2", "/data/Rando/sub/deep/b.mp4"),
                _scene_at("3", "/data/VR/c.mp4"),
                _scene_at("4", "/data/RandoVR/d.mp4"),  # sibling, must NOT match
            ],
        }
    }
    client = FakeClient([page])
    ids = [s.id for s in client.iter_scenes(path_prefix="/data/Rando")]
    assert ids == ["1", "2"]  # folder + subfolders only; not VR, not RandoVR


def test_iter_scenes_no_filter_returns_all():
    page = {
        "findScenes": {
            "count": 2,
            "scenes": [_scene_at("1", "/data/VR/a.mp4"), _scene_at("2", "/x/b.mp4")],
        }
    }
    client = FakeClient([page])
    assert [s.id for s in client.iter_scenes(path_prefix="")] == ["1", "2"]


def test_iter_scenes_stops_on_empty_page():
    page = {"findScenes": {"count": 99, "scenes": []}}
    client = FakeClient([page])
    assert list(client.iter_scenes()) == []


def test_scene_parsing_exposes_fingerprint_and_duration():
    s = Scene.from_dict(_scene("7", markers=2))
    assert s.duration == 600.0
    assert s.fingerprint == "hash7"
    assert len(s.markers) == 2
    assert s.markers[0].primary_tag.name == "apex"


def test_scene_fingerprint_falls_back_to_none_when_absent():
    s = Scene.from_dict(_scene("8", fp=False))
    assert s.fingerprint is None


def test_find_or_create_tag_reuses_existing():
    found = {"findTags": {"tags": [{"id": "42", "name": "apex"}]}}
    client = FakeClient([found])
    tag = client.find_or_create_tag("apex")
    assert tag.id == "42"
    assert len(client.calls) == 1  # no create call made


def test_find_or_create_tag_creates_when_missing():
    none = {"findTags": {"tags": []}}
    created = {"tagCreate": {"id": "99", "name": "apex:heels"}}
    client = FakeClient([none, created])
    tag = client.find_or_create_tag("apex:heels")
    assert tag.id == "99"
    assert len(client.calls) == 2


def test_create_scene_marker_builds_ranged_input():
    client = FakeClient([{"sceneMarkerCreate": {"id": "m1"}}])
    client.create_scene_marker(
        scene_id="5", seconds=12.5, primary_tag_id="42", title="apex", end_seconds=30.0
    )
    _, variables = client.calls[0]
    inp = variables["input"]
    assert inp["scene_id"] == "5"
    assert inp["seconds"] == 12.5
    assert inp["primary_tag_id"] == "42"
    assert inp["end_seconds"] == 30.0


def test_create_scene_marker_omits_end_when_none():
    client = FakeClient([{"sceneMarkerCreate": {"id": "m1"}}])
    client.create_scene_marker(scene_id="5", seconds=1.0, primary_tag_id="42")
    assert "end_seconds" not in client.calls[0][1]["input"]


@pytest.mark.parametrize(
    "start,expected_contains",
    [(None, "/scene/5/stream"), (42.0, "start=42"), (12.5, "start=12.5")],
)
def test_stream_url(start, expected_contains):
    client = StashClient("http://stash.test:6969/", api_key="secret")
    url = client.stream_url("5", start=start)
    assert expected_contains in url
    assert "apikey=secret" in url  # key travels as query param for <video>
