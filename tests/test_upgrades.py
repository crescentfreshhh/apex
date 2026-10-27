"""Upgrades: which keepers are wanted, which Prowlarr releases are this scene,
healthy and better, the daily pass, grab → arrived / stalled, and settings
that never hand the API key back."""

import time

import pytest

pytest.importorskip("fastapi")

from fakestash import FakeStash  # noqa: E402
from peaks import prowlarr as pw  # noqa: E402
from peaks.config import Config  # noqa: E402

# --- the pure rules -------------------------------------------------------------------

SCENE = {"studio": "Vixen", "date": "2023-05-14", "performers": ["Jane Doe"], "title": "Sunset Dreams"}


@pytest.mark.parametrize("title,want", [
    ("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.XXX.2160p.MP4-WRB", True),
    ("Vixen 2023-05-14 Jane Doe 1080p", True),
    ("Vixen - Jane Doe - Sunset Dreams [4K]", True),            # no date, but performer + title
    ("Vixen.23.06.01.Jane.Doe.Other.Scene.2160p", False),        # same performer, another scene
    ("Blacked.23.05.14.Jane.Doe.2160p", False),                  # another studio, same day
])
def test_a_release_must_be_this_scene(title, want):
    assert pw.matches_scene(title, SCENE) is want


def test_release_names_say_resolution_and_vr():
    assert pw.parse_release("X.2160p.HEVC") == {"height": 2160, "vr": False}
    assert pw.parse_release("X [4K]")["height"] == 2160 and pw.parse_release("X UHD")["height"] == 2160
    assert pw.parse_release("X.1080p")["height"] == 1080 and pw.parse_release("X")["height"] is None
    assert pw.parse_release("VixenVR.23.05.14.Oculus.5K")["vr"]


def test_healthy_means_seeders_not_just_peers():
    assert pw.is_healthy({"seeders": 12, "leechers": 3}, 5)
    assert not pw.is_healthy({"seeders": 2, "leechers": 0}, 5)          # too few
    assert not pw.is_healthy({"seeders": 6, "leechers": 40}, 5)         # a crowd waiting on six
    assert not pw.is_healthy({"leechers": 3}, 5)                        # unknown
    assert not pw.is_healthy({"seeders": 50, "protocol": "usenet"}, 5)


def test_what_counts_as_an_upgrade():
    assert pw.is_upgrade(2160, 5e9, "1080p", 3e9, poor=False)
    assert not pw.is_upgrade(1080, 9e9, "1080p", 3e9, poor=False)        # bigger, same res, fine picture
    assert pw.is_upgrade(2160, 9e9, "4K", 5e9, poor=True)                # an upscale: same res, 1.8× size
    assert not pw.is_upgrade(2160, 6e9, "4K", 5e9, poor=True)            # only 1.2×
    assert not pw.is_upgrade(None, 9e9, "1080p", 3e9, poor=True)         # name doesn't say


def test_queries_most_specific_first():
    assert pw.queries_for(SCENE) == ["Vixen 23 05 14", "Vixen Jane Doe Sunset Dreams"]


# --- the service ----------------------------------------------------------------------

class FakeProwlarr:
    def __init__(self, releases):
        self.releases, self.searches, self.grabs = releases, [], []

    def search(self, query, categories=None):
        self.searches.append(query)
        return list(self.releases)

    def grab(self, guid, indexer_id):          # Prowlarr's own grab: Peaks must never use it now
        raise AssertionError("Peaks asked Prowlarr to grab (it has no download client)")

    def fetch_torrent(self, download_url, magnet_url=None):
        self.grabs.append(download_url)
        return "torrent", b"d4:infod4:name3:abcee"


class FakeQbt:
    def __init__(self):
        self.added = []

    def add_torrent(self, data=None, magnet=None, category=""):
        self.added.append({"data": data, "magnet": magnet, "category": category})


def _rel(title, seeders=20, leechers=2, size=8e9, guid=None):
    g = guid or title
    return {"guid": g, "indexerId": 3, "indexer": "Tracker", "title": title, "size": size,
            "seeders": seeders, "leechers": leechers, "protocol": "torrent",
            "downloadUrl": f"http://prowlarr/3/download?link={g}"}


@pytest.fixture
def stash():
    base = {"rating100": 100, "date": "2023-05-14", "performers": ["Jane Doe"], "size": int(3e9)}
    return FakeStash({
        "1": {**base, "o_counter": 18, "title": "Sunset Dreams", "studio": "Vixen", "width": 1920, "height": 1080},
        "2": {**base, "o_counter": 17, "title": "Night Swim", "studio": "Tushy", "width": 3840, "height": 2160},
        "3": {**base, "o_counter": 16, "title": "Garden", "studio": "Blacked", "width": 1280, "height": 720},
        "4": {**base, "rating100": None, "title": "Unrated", "studio": "Vixen", "width": 1280, "height": 720},
        "5": {**base, "o_counter": 18, "title": "VR One", "studio": "Vixen", "width": 5760, "height": 2880,
              "tags": ["VR"]},
    })


@pytest.fixture
def svc(tmp_path, stash, monkeypatch):
    import peaks.web.service as svc_mod

    monkeypatch.setenv("PEAKS_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.delenv("PEAKS_PROWLARR_URL", raising=False)
    monkeypatch.delenv("PEAKS_PROWLARR_KEY", raising=False)
    cfg = Config()
    cfg.embedding.cache_dir = str(tmp_path / "cache")
    cfg.modeling.dir = str(tmp_path / "models")
    cfg.modeling.labels_path = str(tmp_path / "labels.json")
    monkeypatch.setattr(svc_mod.Service, "client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "_meta_client", lambda self: stash)
    monkeypatch.setattr(svc_mod.Service, "picture_scores",
                        lambda self: {"2": {"score": 30, "band": "Poor", "upscaled": True, "real": 1080}})
    return svc_mod.Service(cfg)


def test_wanted_keepers_below_4k_poor_or_upscaled(svc):
    w = {x["row"]["scene_id"]: x["reasons"] for x in svc.upgrade_wanted()}
    assert set(w) == {"1", "2", "3"}                       # 4 isn't a keeper, 5 is VR
    assert w["1"] == ["1080p"] and w["3"] == ["720p"]
    assert w["2"] == ["picture Poor 30", "upscale — real detail of ~1080p"]
    assert [x["row"]["scene_id"] for x in svc.upgrade_wanted()][0] == "1"     # Légendaire first
    svc.save_upgrade_settings({"excluded_studios": ["vixen"]})              # your excluded studios
    assert "1" not in {x["row"]["scene_id"] for x in svc.upgrade_wanted()}


def test_search_keeps_this_scene_healthy_and_better(svc):
    fake = FakeProwlarr([
        _rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p", seeders=30),
        _rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p.ALT", seeders=2),    # unhealthy
        _rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.1080p"),                   # not better
        _rel("Vixen.23.06.01.Jane.Doe.Other.2160p"),                           # another scene
        _rel("VixenVR.23.05.14.Jane.Doe.Oculus.2160p"),                        # VR
    ])
    r = svc.upgrade_search("1", client=fake)
    assert [c["title"] for c in r["found"]] == ["Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p"]
    assert fake.searches[0] == "Vixen 23 05 14"
    up = svc.upgrades()
    assert [c["scene_id"] for c in up["found"]] == ["1"] and up["found"][0]["found"][0]["seeders"] == 30


def test_an_upscale_upgrade_at_the_same_resolution_needs_a_bigger_file(svc):
    fake = FakeProwlarr([_rel("Tushy.23.05.14.Jane.Doe.Night.Swim.2160p", size=6e9, guid="a"),
                         _rel("Tushy.23.05.14.Jane.Doe.Night.Swim.2160p.REAL", size=3.3e9, guid="b")])
    assert [c["guid"] for c in svc.upgrade_search("2", client=fake)["found"]] == ["a"]


def test_the_daily_pass_is_capped_and_waits_before_searching_again(svc):
    svc.save_upgrade_settings({"url": "http://p:9696", "api_key": "k", "auto": True, "per_day": 2})
    fake = FakeProwlarr([])
    slept = []
    out = svc.run_upgrade_pass(client=fake, sleep=slept.append)
    assert out["searched"] == 2 and slept == [3.0]
    assert not svc.upgrade_due()                           # ran today
    out2 = svc.run_upgrade_pass(client=fake, sleep=slept.append)
    assert out2["searched"] == 1                           # the two from before wait 14 days


def test_grab_only_found_releases_then_follow_it(svc):
    fake = FakeProwlarr([_rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p", guid="g1")])
    svc.upgrade_search("1", client=fake)
    qbt = FakeQbt()
    with pytest.raises(LookupError):
        svc.upgrade_grab("1", "made-up", client=fake, qbt=qbt)
    out = svc.upgrade_grab("1", "g1", client=fake, qbt=qbt)
    assert fake.grabs == ["http://prowlarr/3/download?link=g1"]
    assert qbt.added == [{"data": b"d4:infod4:name3:abcee", "magnet": None, "category": out["category"]}]
    assert "upgrade-grab" in [e["action"] for e in svc.history()]
    assert [c["scene_id"] for c in svc.upgrades()["grabbed"]] == ["1"]
    # the new copy arrives and the duplicate check pairs it with the old one
    big = {"scene_id": "9", "size": int(9e9)}
    svc._dupe_cache = {"groups": [{"scenes": [svc._cat_update_row("1"), big]}]}
    assert [c["scene_id"] for c in svc.upgrades()["arrived"]] == ["1"]


def test_a_grab_that_never_arrives_stalls_and_can_be_retried(svc):
    fake = FakeProwlarr([_rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p", guid="g1"),
                         _rel("Vixen.23.05.14.Jane.Doe.Sunset.Dreams.2160p.v2", guid="g2")])
    svc.upgrade_search("1", client=fake)
    svc.upgrade_grab("1", "g1", client=fake, qbt=FakeQbt())
    d = svc._upg_load()
    d["scenes"]["1"]["grabbed"]["at"] = time.time() - 15 * 86400
    svc._upg_save(d)
    assert svc.upgrades()["grabbed"][0]["state"] == "stalled"
    e = svc.upgrade_retry("1")
    assert e["state"] == "found" and [c["guid"] for c in e["found"]] == ["g2"]


def test_dismissed_scenes_leave_the_lists(svc):
    svc.upgrade_dismiss("1")
    up = svc.upgrades()
    assert "1" not in {c["scene_id"] for k in ("found", "wanted", "grabbed", "arrived") for c in up[k]}


def test_settings_never_return_the_api_key(svc, monkeypatch):
    from fastapi.testclient import TestClient

    import peaks.web.app as app_mod

    monkeypatch.setenv("PEAKS_WARM_ON_START", "0")
    monkeypatch.setattr(app_mod, "Service", lambda cfg=None: svc)
    api = TestClient(app_mod.create_app(svc.cfg))
    s = api.post("/api/upgrades/settings", json={"url": "http://prowlarr:9696/", "api_key": "secret1234"}).json()
    assert s["api_key"] == "••••••••1234" and s["has_key"] and s["url"] == "http://prowlarr:9696"
    s2 = api.post("/api/upgrades/settings", json={"api_key": s["api_key"], "min_seeders": 8}).json()
    assert svc.upgrade_settings(reveal=True)["api_key"] == "secret1234"     # the masked echo kept it
    assert s2["min_seeders"] == 8 and "secret" not in api.get("/api/upgrades").text
    assert api.post("/api/upgrades/grab", json={"scene_id": "1", "guid": "x"}).status_code == 404


def test_not_configured_says_so(svc):
    assert svc.upgrade_test() == {"ok": False, "error":
                                  "Prowlarr isn't set up — add its URL and API key in Settings → Upgrades"}
    assert not svc.upgrade_due()


class ForgetfulProwlarr(FakeProwlarr):
    """Prowlarr's links are minted per search and can go stale; `reguid` mimics
    indexers that also mint a new guid on every search."""

    def __init__(self, releases, reguid=False):
        super().__init__(releases)
        self.live, self.reguid, self.n = set(), reguid, 0

    def search(self, query, categories=None):
        self.n += 1
        out = []
        for r in self.releases:
            r = dict(r, guid=f"{r['guid']}#{self.n}") if self.reguid else dict(r)
            r["downloadUrl"] = f"http://prowlarr/3/download?link={r['guid']}&s={self.n}"
            self.live.add(r["downloadUrl"])
            out.append(r)
        self.searches.append(query)
        return out

    def forget(self):
        self.live.clear()

    def fetch_torrent(self, download_url, magnet_url=None):
        if download_url not in self.live:
            raise pw.ProwlarrError("Prowlarr couldn't fetch the torrent (HTTP 404)", 404)
        return super().fetch_torrent(download_url, magnet_url)


def test_a_stale_link_searches_again(svc):
    fake, qbt = ForgetfulProwlarr([_rel("Vixen 23 05 14 Jane Doe 2160p", guid="g1")]), FakeQbt()
    svc.upgrade_search("1", client=fake)
    fake.forget()                                          # hours later
    svc.upgrade_grab("1", "g1", client=fake, qbt=qbt)
    assert len(qbt.added) == 1                             # found again and added, once
    assert [c["scene_id"] for c in svc.upgrades()["grabbed"]] == ["1"]


def test_a_new_guid_on_the_second_search_is_still_the_same_release(svc):
    fake, qbt = ForgetfulProwlarr([_rel("Vixen 23 05 14 Jane Doe 2160p", guid="g1")], reguid=True), FakeQbt()
    first = svc.upgrade_search("1", client=fake)["found"][0]["guid"]
    fake.forget()
    svc.upgrade_grab("1", first, client=fake, qbt=qbt)
    assert len(qbt.added) == 1 and svc.upgrades()["grabbed"][0]["grabbed"]["guid"] != first


def test_a_release_gone_from_the_indexer_says_so_and_refreshes(svc):
    fake, qbt = ForgetfulProwlarr([_rel("Vixen 23 05 14 Jane Doe 2160p", guid="g1")]), FakeQbt()
    svc.upgrade_search("1", client=fake)
    fake.forget()
    fake.releases = [_rel("Vixen 23 05 14 Jane Doe 2160p REPACK", guid="g2", size=9e9)]
    with pytest.raises(LookupError, match="isn't on Tracker any more"):
        svc.upgrade_grab("1", "g1", client=fake, qbt=qbt)
    assert qbt.added == []
    assert [c["guid"] for c in svc.upgrades()["found"][0]["found"]] == ["g2"]   # the list is fresh


def test_releases_found_before_links_were_kept_are_found_again(svc):
    fake, qbt = FakeProwlarr([_rel("Vixen 23 05 14 Jane Doe 2160p", guid="g1")]), FakeQbt()
    svc.upgrade_search("1", client=fake)
    d = svc._upg_load()
    for c in d["scenes"]["1"]["found"]:
        c.pop("download_url"); c.pop("magnet_url")       # as stored by the previous version
    svc._upg_save(d)
    svc.upgrade_grab("1", "g1", client=fake, qbt=qbt)
    assert len(qbt.added) == 1


def test_grab_category_defaults_to_the_first_flat_pull_category_and_needs_qbt(svc):
    svc.save_seedjob_settings({"categories": [{"cat": "vr", "dest": "/x"}, {"cat": "1", "dest": "/y"}]})
    assert svc._upg_category() == "1"                       # never VR on its own
    svc.save_upgrade_settings({"qbt_category": "upgrades"})
    assert svc._upg_category() == "upgrades"
    fake = FakeProwlarr([_rel("Vixen 23 05 14 Jane Doe 2160p", guid="g1")])
    svc.upgrade_search("1", client=fake)
    svc.cfg.seedbox.qbt_url = ""
    with pytest.raises(ValueError, match="qBittorrent"):
        svc.upgrade_grab("1", "g1", client=fake)


def test_prowlarr_errors_carry_its_own_explanation():
    class R:
        status_code, content, text = 404, b"x", ""

        def json(self):
            return {"message": "Couldn't find requested release in cache, try searching again"}

    class S:
        headers = {}

        def request(self, *a, **k):
            return R()

    c = pw.ProwlarrClient("http://p", "k", session=S())
    with pytest.raises(pw.ProwlarrError) as e:
        c.grab("g", 1)
    assert e.value.status == 404 and "try searching again" in str(e.value)


def test_fetch_torrent_file_magnet_redirect_and_fallback():
    class R:
        def __init__(self, status=200, content=b"", headers=None, text=""):
            self.status_code, self.content, self.headers, self.text = status, content, headers or {}, text

        def json(self):
            raise ValueError

    class S:
        def __init__(self, answers):
            self.headers, self.answers, self.urls = {}, list(answers), []

        def get(self, url, **k):
            self.urls.append(url)
            return self.answers.pop(0)

    s = S([R(200, b"d4:infoe")])
    c = pw.ProwlarrClient("http://jarvis:9696", "k", session=s)
    assert c.fetch_torrent("http://localhost:9696/3/download?link=x") == ("torrent", b"d4:infoe")
    assert s.urls == ["http://jarvis:9696/3/download?link=x"]        # through the address Peaks uses
    c = pw.ProwlarrClient("http://jarvis:9696", "k", session=S([R(302, headers={"Location": "magnet:?xt=1"})]))
    assert c.fetch_torrent("http://jarvis:9696/3/download?link=x") == ("magnet", "magnet:?xt=1")
    c = pw.ProwlarrClient("http://jarvis:9696", "k", session=S([R(500, text="boom")]))
    assert c.fetch_torrent("http://jarvis:9696/3/download?link=x", "magnet:?xt=2") == ("magnet", "magnet:?xt=2")
    c = pw.ProwlarrClient("http://jarvis:9696", "k", session=S([R(404, text="gone")]))
    with pytest.raises(pw.ProwlarrError) as e:
        c.fetch_torrent("http://jarvis:9696/3/download?link=x")
    assert e.value.status == 404
