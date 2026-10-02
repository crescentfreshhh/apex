"""A fake Stash for library-management tests. It keeps scene state (rating,
O-count, tags, organized, file facts) and records every mutation in `calls`,
so tests can assert the exact sequence of writes Peaks makes."""

from types import SimpleNamespace


class FakeStash:
    def __init__(self, scenes: dict, tags: dict | None = None):
        self.s = scenes              # sid -> {rating100, o_counter, tag_ids, organized, ...}
        for sid, v in self.s.items():
            v.setdefault("rating100", None)
            v.setdefault("o_counter", 0)
            v.setdefault("tag_ids", [])
            v.setdefault("organized", False)
            v.setdefault("fingerprint", f"fp{sid}")
        self.tags = dict(tags or {})  # tag id -> name
        self.markers: list[dict] = []  # {scene_id, seconds, marker_id} under the taste tag
        self.calls: list = []
        self.caps = {"scenesDestroy": True, "findDuplicateScenes": True,
                     "metadataScan": True, "metadataIdentify": True,
                     "metadataAutoTag": True, "findJob": True, "configuration": True,
                     "deleteFiles": True}

    # --- reads ---------------------------------------------------------------

    def iter_scenes(self, path_prefix=None):
        for sid in self.s:
            yield SimpleNamespace(id=sid)

    def iter_markers_by_tag(self, tag):
        yield from self.markers

    def scene_details(self, ids):
        out = {}
        for i in ids:
            v = self.s.get(str(i))
            if v is None:
                continue
            out[str(i)] = {"path": f"/data/{i}.mp4", "title": f"Scene {i}",
                           "performers": ["Jane"], **v,
                           "tag_ids": list(v["tag_ids"]),
                           "tags": v.get("tags") or [self.tags[t] for t in v["tag_ids"]]}
        return out

    def capabilities(self):
        return dict(self.caps)

    # performers: set `self.performers` to [{id, name, aliases, has_image, …}]
    performers: list = []

    def iter_performers(self, page_size=500):
        for p in self.performers:
            yield {"aliases": [], "image": None, "has_image": False, "scene_count": 0,
                   "favorite": False, "rating100": None, "created_at": "", **p}

    def performer_image(self, pid):
        p = next((x for x in self.performers if x["id"] == str(pid)), None)
        if not p or not p.get("has_image"):
            return None
        return _png(200, 300, (200, 60, 90)), "image/png"

    def scene_screenshot(self, sid):
        return (_png(320, 180, (40, 90, 160)), "image/png") if str(sid) in self.s else None

    def find_tag_by_name(self, name):
        for tid, n in self.tags.items():
            if n.lower() == name.lower():
                return SimpleNamespace(id=tid, name=n)
        return None

    def find_or_create_tag(self, name):
        t = self.find_tag_by_name(name)
        if t:
            return t
        tid = str(900 + len(self.tags))
        self.tags[tid] = name
        self.calls.append(("tag_create", name))
        return SimpleNamespace(id=tid, name=name)

    def stream_url(self, sid, start=None):
        return f"http://stash/{sid}?t={start}"

    # --- writes --------------------------------------------------------------

    def update_scene(self, scene_id, clear=(), **fields):
        rec = {k: v for k, v in fields.items() if v is not None}
        if clear:
            rec["clear"] = tuple(clear)
        self.calls.append(("update", scene_id, rec))
        v = self.s[scene_id]
        if "rating100" in clear:
            v["rating100"] = None
        for k in ("rating100", "organized"):
            if fields.get(k) is not None:
                v[k] = fields[k]
        if fields.get("tag_ids") is not None:
            v["tag_ids"] = [str(t) for t in fields["tag_ids"]]
        return {}

    def scene_add_o(self, sid):
        self.calls.append(("add_o", sid))
        self.s[sid]["o_counter"] += 1
        return self.s[sid]["o_counter"]

    def scene_delete_o(self, sid):
        self.calls.append(("del_o", sid))
        self.s[sid]["o_counter"] -= 1
        return self.s[sid]["o_counter"]

    def scene_reset_o(self, sid):
        self.calls.append(("reset_o", sid))
        self.s[sid]["o_counter"] = 0
        return 0

    def destroy_scenes(self, scene_ids, delete_file=True, delete_generated=True):
        self.calls.append(("destroy", tuple(scene_ids), delete_file, delete_generated))
        for sid in scene_ids:
            self.s.pop(str(sid), None)
        return len(scene_ids)

    # same-file copies: a scene's `files` = [{id, path, size, oshash?, md5?,
    # mod_time?}]; files[0] is primary. Scenes with 2+ files are listed.

    def multi_file_scenes(self, page_size=200):
        out = []
        for sid, v in self.s.items():
            fs = v.get("files") or []
            if len(fs) < 2:
                continue
            out.append({"id": sid, "title": v.get("title") or f"Scene {sid}",
                        "rating100": v["rating100"], "o_counter": v["o_counter"],
                        "files": [{"id": f["id"], "path": f["path"],
                                   "basename": f["path"].rsplit("/", 1)[-1], "size": f["size"],
                                   "mod_time": f.get("mod_time", ""),
                                   "fingerprints": {k: f[k] for k in ("oshash", "md5") if f.get(k)}}
                                  for f in fs]})
        return out

    def set_primary_file(self, scene_id, file_id):
        self.calls.append(("primary", str(scene_id), str(file_id)))
        fs = self.s[str(scene_id)]["files"]
        f = next(x for x in fs if x["id"] == str(file_id))
        fs.remove(f)
        fs.insert(0, f)

    def delete_files(self, file_ids):
        self.calls.append(("delete_files", tuple(file_ids)))
        ids = {str(i) for i in file_ids}
        for v in self.s.values():
            if v.get("files"):
                v["files"] = [f for f in v["files"] if f["id"] not in ids]
        return True

    def duplicate_groups(self, distance=0, duration_diff=-1.0):
        self.calls.append(("dupes", distance, duration_diff))
        return [[s for s in g if s in self.s] for g in getattr(self, "dupes", [])
                if len([s for s in g if s in self.s]) > 1]

    # --- tasks / jobs (ingest) -------------------------------------------------
    # `arriving` scenes appear when a scan runs; each started task is a job
    # that reports RUNNING once, then `job_outcome.get(kind, "FINISHED")`.

    def all_scene_ids(self):
        return set(self.s)

    def config_defaults(self):
        return getattr(self, "defaults", {"scan": None, "identify": None, "autoTag": None})

    def fit_input(self, value, type_name):
        return {k: v for k, v in (value or {}).items() if v is not None}

    def input_has(self, type_name, field):
        return True

    def _task(self, kind, inp):
        self.calls.append((kind, inp))
        jobs = self.__dict__.setdefault("jobs", {})
        jid = str(len(jobs) + 1)
        jobs[jid] = {"kind": kind, "polls": 0}
        if kind == "scan":
            for sid, v in getattr(self, "arriving", {}).items():
                self.s[sid] = {"rating100": None, "o_counter": 0, "tag_ids": [],
                               "organized": False, "fingerprint": f"fp{sid}", **v}
        return jid

    def metadata_scan(self, inp):
        return self._task("scan", inp)

    def metadata_identify(self, inp):
        return self._task("identify", inp)

    def metadata_auto_tag(self, inp):
        return self._task("auto_tag", inp)

    def find_job(self, jid):
        j = self.jobs[jid]
        j["polls"] += 1
        if j["polls"] == 1:
            return {"id": jid, "status": "RUNNING", "progress": 0.5}
        return {"id": jid, "status": getattr(self, "job_outcome", {}).get(j["kind"], "FINISHED"),
                "progress": 1.0, "error": "boom"}

    def stop_job(self, jid):
        self.calls.append(("stop", jid))


def _png(w, h, rgb):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")
    return buf.getvalue()
