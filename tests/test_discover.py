"""Discover: favourites from your grades, StashDB scenes you don't have ranked by
fit, then Find on Prowlarr and Grab into qBittorrent — only on your click."""

import pytest

pytest.importorskip("fastapi")

from test_upgrades import FakeProwlarr, FakeQbt, _rel, stash, svc  # noqa: E402,F401
from peaks.stashbox import pick_box, scene_row  # noqa: E402


class FakeBox:
    def __init__(self, by):
        self.by, self.calls = by, []

    def scenes(self, performer=None, studio=None, per_page=40, page=1):
        self.calls.append(performer or studio)
        return [scene_row(s) for s in self.by.get(performer or studio, [])]


def _sb(i, title, studio, perfs, date="2024-02-03"):
    return {"id": i, "title": title, "release_date": date, "duration": 1800, "studio": {"id": "st-" + studio, "name": studio},
            "performers": [{"performer": {"id": pid, "name": name}} for pid, name in perfs],
            "images": [{"url": f"http://img/{i}.jpg", "width": 1280}], "urls": [{"url": f"http://site/{i}"}]}


def test_pick_box_prefers_stashdb():
    boxes = [{"endpoint": "https://pmvstash.org/graphql", "api_key": "k"},
             {"endpoint": "https://stashdb.org/graphql", "api_key": "k", "name": "StashDB"}]
    assert pick_box(boxes)["name"] == "StashDB" and pick_box([]) is None


def test_favourites_come_from_your_grades(svc, monkeypatch):
    import peaks.web.service as svc_mod

    rows = ([{"tier": "legendaire", "performer_ids": ["1"], "studio": "Vixen"}] * 4
            + [{"tier": "rejected", "performer_ids": ["2"], "studio": "Blacked"}] * 4
            + [{"tier": "exceptionnelle", "performer_ids": ["2"], "studio": "Blacked"}] * 2)
    monkeypatch.setattr(svc_mod.Service, "_catalogue_all", lambda self, refresh=False: rows)
    monkeypatch.setattr(svc_mod.Service, "_stash_performers", lambda self, refresh=False: [
        {"id": "1", "name": "Jane Doe"}, {"id": "2", "name": "Mia Lee"}])
    f = svc.discover_favourites()
    assert [p["name"] for p in f["performers"]] == ["Jane Doe"]       # mostly rejected: not a favourite
    assert [s["name"] for s in f["studios"]] == ["Vixen"]


def test_discover_lists_what_you_dont_have_best_fit_first_then_find_and_grab(svc, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setattr(svc_mod.Service, "discover_favourites", lambda self: {
        "performers": [{"id": "1", "name": "Jane Doe", "score": 2.5, "keeps": 4},
                       {"id": "7", "name": "Ava Moon", "score": 1.0, "keeps": 2},
                       {"id": "8", "name": "Unlinked One", "score": 1.0, "keeps": 2}],
        "studios": [{"name": "Vixen", "score": 1.5, "keeps": 5}]})
    monkeypatch.setattr(svc_mod.Service, "_disc_local", lambda self, ep: {
        "performers": {"1": "sb-jane", "7": "sb-ava", "8": None}, "studios": {"Vixen": "sb-vixen"},
        "owned": {"s-owned"}})
    box = FakeBox({
        "sb-jane": [_sb("s-owned", "Already Mine", "Vixen", [("sb-jane", "Jane Doe")]),
                    _sb("s-new", "Sunset Dreams", "Vixen", [("sb-jane", "Jane Doe"), ("sb-ava", "Ava Moon")], "2023-05-14")],
        "sb-ava": [_sb("s-ava", "Pool Day", "Tushy", [("sb-ava", "Ava Moon")])],
        "sb-vixen": [_sb("s-new", "Sunset Dreams", "Vixen", [("sb-jane", "Jane Doe"), ("sb-ava", "Ava Moon")], "2023-05-14")],
    })
    r = svc.run_discover(box=(box, {"endpoint": "https://stashdb.org/graphql", "name": "StashDB"}))
    assert r == {"found": 2, "sources": 3, "unlinked": 1, "errors": 0}
    d = svc.discover()
    ids = [x["id"] for x in d["items"]]
    assert ids == ["s-new", "s-ava"] and "s-owned" not in ids               # owned dropped; best fit first
    assert set(d["items"][0]["because"]) == {"Jane Doe", "Vixen"} and d["unlinked"] == ["Unlinked One"]

    svc.save_upgrade_settings({"url": "http://prowlarr", "api_key": "k"})
    fake, qbt = FakeProwlarr([_rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p", guid="g1"),
                              _rel("Vixen.23.06.01.Other.Scene.2160p", guid="g2")]), FakeQbt()
    got = svc.discover_search("s-new", client=fake)
    assert [c["guid"] for c in got["releases"]] == ["g1"]                  # only this scene, only healthy
    assert not qbt.added                                                     # finding never downloads
    with pytest.raises(LookupError):
        svc.discover_grab("s-new", "g2", client=fake, qbt=qbt)
    svc.discover_grab("s-new", "g1", client=fake, qbt=qbt)
    assert len(qbt.added) == 1 and svc.discover()["items"][0]["state"] == "grabbed"
    svc.discover_dismiss("s-ava")
    assert [x["id"] for x in svc.discover()["items"]] == ["s-new"]
