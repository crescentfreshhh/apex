"""A week of how Peaks spent its time, in one page (the Speed panel's "This week").

Every few minutes a small sample is saved (Peaks' own SQLite store): request
times per route since the last sample, Stash's latency, whether the GPU was
busy, how much the process read and wrote. Each job, when it finishes, adds how
long it ran and what it read and wrote; each retrain adds what each input gained
in accuracy. Once a night the last 7 days become one report: the slowest
routes, Stash, the GPU, every kind of job — and one plain line per background
job saying whether it earns its keep.

Disk numbers come from /proc/self/io for the whole process: a job that ran
alongside another shares the bill (marked approximate). Without nvidia-smi
there's no GPU line; without /proc, no disk numbers.
"""

from __future__ import annotations

import time

SAMPLES, JOBS, GAINS = "health", "jobs_io", "gains"
KEEP_DAYS = 8
WEEK = 7 * 86400

# a job's accuracy gain lives under these report keys (fractions of exact CV accuracy)
GAIN_OF = {"hear": "sound_gain", "motion": "motion_gain", "picture": "picture_gain", "zoom": "moment_gain"}
LABEL = {"hear": "Hearing", "motion": "Motion", "picture": "Picture quality", "zoom": "Moment zoom",
         "faces": "Faces", "embed": "Embedding", "ingest": "Ingest", "fix": "Fix", "sync": "Sync",
         "lab": "Lab trials", "backup": "Backup", "warmup": "Loading the library", "taste-measure": "Taste measure",
         "upgrades": "Upgrades", "seed-pull": "Seedbox pull"}


# --- reading the machine ------------------------------------------------------------------

def proc_io() -> dict | None:
    """Bytes this process has read from / written to storage so far."""
    try:
        with open("/proc/self/io") as fh:
            kv = dict(ln.split(":", 1) for ln in fh.read().splitlines() if ":" in ln)
        return {"read": int(kv["read_bytes"]), "write": int(kv["write_bytes"])}
    except (OSError, KeyError, ValueError):
        return None


def io_delta(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return None
    return {"read": max(0, b["read"] - a["read"]), "write": max(0, b["write"] - a["write"])}


# --- the 5-minute sample ------------------------------------------------------------------

def sample(since: float, routes: list[dict], stash: dict | None, gpu: dict | None, io: dict | None,
           running: list[str], now: float | None = None) -> dict:
    """One row: routes as {route: [n, p50_ms, p95_ms]} for requests since `since`."""
    return {"at": now or time.time(), "since": since,
            "routes": {r["route"]: [int(r["n"]), r.get("p50_ms"), r.get("p95_ms")] for r in routes if r.get("n")},
            "stash": {k: (stash or {}).get(k) for k in ("calls", "median_ms", "p90_ms")} if stash else None,
            "gpu": gpu, "io": io, "jobs": sorted(set(running))}


# --- the week ----------------------------------------------------------------------------

def _wpct(pairs: list[tuple[float, int]], q: float) -> float | None:
    """Weighted percentile of (value, weight) pairs."""
    pairs = sorted((v, w) for v, w in pairs if v is not None and w > 0)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    acc = 0
    for v, w in pairs:
        acc += w
        if acc >= q * total:
            return v
    return pairs[-1][0]


def latest_gains(gains: list[dict]) -> dict:
    """The most recent value of each gain key across the retrains logged."""
    out: dict[str, float] = {}
    for g in sorted(gains, key=lambda x: x.get("at") or 0):
        for k, v in g.items():
            if k.endswith("_gain") and v is not None:
                out[k] = float(v)
    return out


def verdict(kind: str, hours: float, read_gb: float, gain: float | None, costly: bool) -> str:
    name = LABEL.get(kind, kind)
    cost = f"{hours:.1f} h" + (f" and {read_gb:.0f} GB read" if read_gb >= 1 else "")
    if gain is None:
        return f"{name}: {cost} this week — no accuracy measured yet"
    pts = gain * 100
    if pts < 1.0 and costly:
        return f"{name}: {cost} this week for {pts:+.1f} points of grade accuracy — consider turning it off"
    if pts < 1.0:
        return f"{name}: {cost} this week, {pts:+.1f} points of accuracy — cheap, so harmless"
    return f"{name}: {cost} this week, {pts:+.1f} points of grade accuracy — earns its keep"


def build_report(samples: list[dict], jobs: list[dict], gains: list[dict], now: float | None = None,
                 days: int = 7) -> dict:
    now = now or time.time()
    cut = now - days * 86400
    samples = [s for s in samples if (s.get("at") or 0) >= cut]
    jobs = [j for j in jobs if (j.get("at") or j.get("started") or 0) >= cut]

    # routes: weighted percentiles of each sample's p50 / p95 by its request count
    per: dict[str, dict] = {}
    for s in samples:
        for route, (n, p50, p95) in (s.get("routes") or {}).items():
            e = per.setdefault(route, {"n": 0, "p50": [], "p95": []})
            e["n"] += n
            e["p50"].append((p50, n))
            e["p95"].append((p95, n))
    routes = sorted(({"route": r, "n": e["n"], "p50_ms": _wpct(e["p50"], 0.5), "p95_ms": _wpct(e["p95"], 0.95)}
                     for r, e in per.items()), key=lambda x: -(x["p95_ms"] or 0))

    st = [s["stash"] for s in samples if s.get("stash") and s["stash"].get("calls")]
    stash = {"calls": sum(x["calls"] for x in st),
             "median_ms": _wpct([(x.get("median_ms"), x["calls"]) for x in st], 0.5),
             "p90_ms": _wpct([(x.get("p90_ms"), x["calls"]) for x in st], 0.9)} if st else None

    g = [s["gpu"] for s in samples if s.get("gpu") and s["gpu"].get("util") is not None]
    gpu = {"busy_pct": round(100 * sum(1 for x in g if x["util"] > 10) / len(g), 1), "samples": len(g),
           "peak_mem_mb": max((x.get("mem_mb") or 0) for x in g)} if g else None

    io = [s["io"] for s in samples if s.get("io")]
    disk = {"read_gb": round(sum(x["read"] for x in io) / 1e9, 1),
            "write_gb": round(sum(x["write"] for x in io) / 1e9, 1)} if io else None

    by: dict[str, dict] = {}
    for j in jobs:
        e = by.setdefault(j["kind"], {"kind": j["kind"], "runs": 0, "hours": 0.0, "read_gb": 0.0, "write_gb": 0.0,
                                      "approximate": False})
        e["runs"] += 1
        e["hours"] += (j.get("elapsed_s") or 0) / 3600
        e["read_gb"] += (j.get("read") or 0) / 1e9
        e["write_gb"] += (j.get("write") or 0) / 1e9
        e["approximate"] |= bool(j.get("overlapped"))
    kinds = sorted(by.values(), key=lambda e: -e["hours"])
    gain_now = latest_gains(gains)
    senses = [e for e in kinds if e["kind"] in GAIN_OF]
    costs = sorted(e["hours"] for e in senses)
    median_cost = costs[len(costs) // 2] if costs else 0.0
    for e in kinds:
        e["hours"], e["read_gb"], e["write_gb"] = round(e["hours"], 2), round(e["read_gb"], 1), round(e["write_gb"], 1)
        if e["kind"] in GAIN_OF:
            gain = gain_now.get(GAIN_OF[e["kind"]])
            e["gain_pts"] = None if gain is None else round(gain * 100, 1)
            e["verdict"] = verdict(e["kind"], e["hours"], e["read_gb"], gain, costly=e["hours"] >= median_cost and e["hours"] > 0.5)
    return {"at": now, "days": days, "samples": len(samples), "routes": routes[:10], "stash": stash, "gpu": gpu,
            "disk": disk, "jobs": kinds, "verdicts": [e["verdict"] for e in kinds if e.get("verdict")]}
