"""Prowlarr: search your indexers for a better copy of a scene, and — only when
you press Grab — ask Prowlarr to send one release to the download client you
configured there. Peaks never talks to the download client itself.

Pure helpers decide what a release is (its resolution, VR or not), whether it
is *this* scene (studio + release date, or studio + performer + title words —
the same performer in a different scene doesn't count), whether its torrent
is healthy (enough seeders, not leechers only) and whether it beats the copy
you have.
"""

from __future__ import annotations

import re
import unicodedata


class ProwlarrError(RuntimeError):
    """`status` is Prowlarr's HTTP status when it answered with an error (else None)."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def _why(r) -> str:
    """Prowlarr's own explanation of an error answer, short."""
    try:
        j = r.json()
        msg = j.get("message") if isinstance(j, dict) else (j[0].get("errorMessage") if isinstance(j, list) and j else None)
    except (ValueError, AttributeError, IndexError):
        msg = None
    msg = (msg or (r.text or "")).strip().replace("\n", " ")
    return msg[:160]


class ProwlarrClient:
    """Prowlarr's v1 API with your API key: health, indexers, search, grab."""

    def __init__(self, url: str, api_key: str, timeout: float = 60.0, session=None):
        import requests

        if not url or not api_key:
            raise ProwlarrError("Prowlarr isn't set up — add its URL and API key in Settings → Upgrades")
        self.base = url.rstrip("/")
        self.timeout = timeout
        self.s = session if session is not None else requests.Session()
        self.s.headers.update({"X-Api-Key": api_key, "Accept": "application/json"})

    def _call(self, method: str, path: str, **kw):
        try:
            r = self.s.request(method, f"{self.base}{path}", timeout=self.timeout, **kw)
        except Exception as exc:  # noqa: BLE001 — network: report, don't raise raw
            raise ProwlarrError(f"can't reach Prowlarr at {self.base} ({type(exc).__name__})") from exc
        if r.status_code == 401:
            raise ProwlarrError("Prowlarr refused the API key (401)", 401)
        if r.status_code >= 400:
            why = _why(r)
            raise ProwlarrError(f"Prowlarr answered HTTP {r.status_code} for {path}" + (f": {why}" if why else ""),
                                r.status_code)
        try:
            return r.json() if r.content else None
        except ValueError as exc:
            raise ProwlarrError(f"Prowlarr's answer to {path} wasn't JSON — is the URL right?") from exc

    def health(self) -> list[dict]:
        return self._call("GET", "/api/v1/health") or []

    def indexers(self) -> list[dict]:
        return self._call("GET", "/api/v1/indexer") or []

    def search(self, query: str, categories: list[int] | None = None) -> list[dict]:
        params: list[tuple[str, str]] = [("query", query), ("type", "search")]
        params += [("categories", str(c)) for c in categories or []]
        out = self._call("GET", "/api/v1/search", params=params)
        return out if isinstance(out, list) else []

    def grab(self, guid: str, indexer_id: int) -> dict:
        """Prowlarr sends this release to its download client."""
        return self._call("POST", "/api/v1/search", json={"guid": guid, "indexerId": int(indexer_id)}) or {}


# --- what a release is ------------------------------------------------------------

_RES = [(re.compile(r"\b(4320p|8k)\b"), 4320), (re.compile(r"\b(2160p|4k|uhd)\b"), 2160),
        (re.compile(r"\b1440p\b"), 1440), (re.compile(r"\b1080[pi]\b"), 1080),
        (re.compile(r"\b720p\b"), 720), (re.compile(r"\b(576p|540p|480p|sd)\b"), 480)]
_VR = re.compile(r"\b(vr|vr180|vr360|sbs|180x180|oculus|gearvr|quest|psvr|lr_180|tb_360)\b")

# a library resolution class → the height a release name would carry
RES_HEIGHT = {"SD": 480, "720p": 720, "1080p": 1080, "1440p": 1440, "4K": 2160,
              "5K": 2880, "6K": 2880, "7K": 3240, "8K": 4320}


def _words(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _squash(s: str) -> str:
    return _words(s).replace(" ", "")


def parse_release(title: str) -> dict:
    """{height (None if the name doesn't say), vr}."""
    w = _words(title)
    height = next((h for rx, h in _RES if rx.search(w)), None)
    return {"height": height, "vr": bool(_VR.search(w))}


def _date_in(title: str, date: str) -> bool:
    """The scene's release date in the name, as YY.MM.DD, YYYY.MM.DD or YYYY-MM-DD."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", date or "")
    if not m:
        return False
    y, mo, d = m.groups()
    nums = re.findall(r"(\d{2,4})[ ._-](\d{2})[ ._-](\d{2})", title or "")
    return any((a in (y, y[2:])) and b == mo and c == d for a, b, c in nums)


_STOP = {"with", "and", "the", "from", "that", "this", "your", "her", "his", "for", "she", "gets", "into"}


def matches_scene(title: str, scene: dict) -> bool:
    """Is this release this scene? Studio + date, or studio + a performer + two
    title words (a scene without a studio: date + performer)."""
    sq, words = _squash(title), set(_words(title).split())
    studio = _squash(scene.get("studio") or "")
    has_studio = bool(studio) and studio in sq
    dated = _date_in(title, scene.get("date") or "")
    performer = any(_squash(p) and _squash(p) in sq for p in scene.get("performers") or [])
    tw = [w for w in _words(scene.get("title") or "").split() if len(w) >= 4 and w not in _STOP]
    titled = len(tw) > 0 and sum(w in words for w in tw) >= min(2, len(tw))
    if studio:
        return has_studio and (dated or (performer and titled))
    return dated and performer


def queries_for(scene: dict) -> list[str]:
    """Search texts, most specific first."""
    out = []
    studio = (scene.get("studio") or "").strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", scene.get("date") or "")
    if studio and m:
        y, mo, d = m.groups()
        out.append(f"{studio} {y[2:]} {mo} {d}")
    perf = next((p for p in scene.get("performers") or [] if p), "")
    if studio and (perf or scene.get("title")):
        out.append(" ".join(x for x in (studio, perf, scene.get("title") or "") if x))
    elif perf and m:
        out.append(f"{perf} {m.group(1)[2:]} {m.group(2)} {m.group(3)}")
    seen, uniq = set(), []
    for q in out:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q[:200])
    return uniq


def is_healthy(rel: dict, min_seeders: int = 5) -> bool:
    """Enough seeders, and not a swarm of leechers waiting on too few."""
    if (rel.get("protocol") or "torrent") != "torrent":
        return False
    s, le = rel.get("seeders"), rel.get("leechers")
    if s is None:
        return False
    return int(s) >= int(min_seeders) and int(s) >= int(le or 0)


def is_upgrade(rel_height: int | None, rel_size: int | None, have_res: str | None,
               have_size: int | None, poor: bool) -> bool:
    """A higher resolution than your copy — or, when your picture is poor or an
    upscale, the same resolution at 1.5× the size or more."""
    have_h = RES_HEIGHT.get(have_res or "", 0)
    if not rel_height:
        return False
    if rel_height > have_h:
        return True
    return bool(poor and rel_height == have_h and rel_size and have_size and rel_size >= 1.5 * have_size)
