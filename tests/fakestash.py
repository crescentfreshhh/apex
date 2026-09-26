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
                     "metadataAutoTag": True, "findJob": True, "configuration": True}

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
