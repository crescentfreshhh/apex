"""The user's keeper grading scheme, as one pure source of truth.

Stash ratings stop at 5 stars, so every keeper ties at the top. The O-counter is
used as the grade *above* 5 stars — not as an event count:

    unrated / 2-4 stars        → unreviewed
    1 star  (rating100 <= 20)  → rejected (Peaks hides these everywhere)
    5 stars, O = 0             → upscale        (lower-tier keeper)
    5 stars, O = 16            → merveilleuse
    5 stars, O = 17            → exceptionnelle
    5 stars, O = 18            → legendaire
    5 stars, any other O       → anomaly        (to revalidate)

Tier keys are stable ASCII ids; display names are configurable (settings.json
`tier_names`) and default to the user's own names.
"""

from __future__ import annotations

# ordered worst → best among graded keepers, with the review states around them
TIERS: tuple[str, ...] = (
    "unreviewed", "rejected", "anomaly",
    "upscale", "merveilleuse", "exceptionnelle", "legendaire",
)

KEEPER_TIERS: tuple[str, ...] = ("upscale", "merveilleuse", "exceptionnelle", "legendaire")

DEFAULT_NAMES: dict[str, str] = {
    "unreviewed": "Unreviewed",
    "rejected": "Rejected",
    "anomaly": "Anomaly",
    "upscale": "Upscale",
    "merveilleuse": "Merveilleuse",
    "exceptionnelle": "Exceptionnelle",
    "legendaire": "Légendaire",
}

# grade → (rating100, o_counter). o_counter None = leave the O-count untouched.
GRADES: dict[str, tuple[int, int | None]] = {
    "legendaire": (100, 18),
    "exceptionnelle": (100, 17),
    "merveilleuse": (100, 16),
    "upscale": (100, 0),
    "reject": (20, None),
}

_O_TIER = {0: "upscale", 16: "merveilleuse", 17: "exceptionnelle", 18: "legendaire"}

# weights for ranking by tier (performer leaderboard, statistics)
TIER_WEIGHT: dict[str, int] = {"merveilleuse": 1, "exceptionnelle": 2, "legendaire": 3}


def tier_of(rating100, o_counter) -> str:
    """The tier for a scene's Stash rating (0-100 scale, None = unrated) and
    O-count."""
    try:
        r = int(rating100) if rating100 is not None else 0
    except (TypeError, ValueError):
        r = 0
    if r <= 0:
        return "unreviewed"
    if r <= 20:
        return "rejected"
    if r < 100:
        return "unreviewed"
    try:
        o = int(o_counter or 0)
    except (TypeError, ValueError):
        o = 0
    return _O_TIER.get(o, "anomaly")


def tier_names(overrides: dict | None = None) -> dict[str, str]:
    """Display names, with any user overrides applied (unknown keys ignored)."""
    names = dict(DEFAULT_NAMES)
    for k, v in (overrides or {}).items():
        if k in names and isinstance(v, str) and v.strip():
            names[k] = v.strip()[:40]
    return names


# --- tier tags (drive the user's renamer plugin) ---------------------------------
# Grading into one of these tiers gives the scene exactly ONE of these tags (all
# others removed) and marks it organized; the renamer plugin then files it into
# that tier's folder. Two tier tags at once break the plugin. Names are
# configurable (settings.json `tier_tags`); these are the defaults.

TIER_TAGS: dict[str, str] = {
    "legendaire": "legendaire",
    "exceptionnelle": "exceptionnelle",
    "merveilleuse": "merveilleuse",
    "upscale": "personal upscale",
}


def tier_tags(overrides: dict | None = None) -> dict[str, str]:
    tags = dict(TIER_TAGS)
    for k, v in (overrides or {}).items():
        if k in tags and isinstance(v, str) and v.strip():
            tags[k] = v.strip()[:80]
    return tags


def tag_state(tier: str, tag_names: list[str], organized: bool,
              tags: dict[str, str] | None = None) -> dict:
    """How a scene's tier tags line up with its grade.

    {present: [tiers whose tag it carries], conflict: str|None, needs_sync: bool}
    conflict: two or more tier tags, or a tier tag that disagrees with the
    O-count grade. needs_sync: a tagged tier missing its tag / organized flag,
    or carrying another tier's tag (fixable by the explicit tag sync)."""
    tags = tags or TIER_TAGS
    have = {n.strip().lower() for n in tag_names or []}
    present = [t for t, name in tags.items() if name.lower() in have]
    conflict = None
    if len(present) >= 2:
        conflict = "several tier tags"
    elif present and present[0] != tier:
        conflict = "tier tag disagrees with the grade"
    needs_sync = tier in tags and (present != [tier] or not organized)
    return {"present": present, "conflict": conflict, "needs_sync": needs_sync}


# --- file quality ------------------------------------------------------------

RES_CLASSES: tuple[str, ...] = ("SD", "720p", "1080p", "1440p", "4K", "5K", "6K", "7K", "8K")


def res_class(width, height) -> str | None:
    """Resolution class. Up to 1440p it goes by the SHORT side, so vertical
    video classifies like landscape (1080x1920 is 1080p). From 4K up it goes by
    the LONG side — how 4K–8K (and VR, often 2:1 side-by-side: 5760x2880 is 6K,
    8192x4096 is 8K) are named — so a 3840x1920 VR file is 4K, not 1080p."""
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    short, long_ = min(w, h), max(w, h)
    if short <= 0:
        return None
    if long_ >= 3800 or short >= 2000:
        if long_ >= 7600:
            return "8K"
        if long_ >= 6600:
            return "7K"
        if long_ >= 5600:
            return "6K"
        if long_ >= 4800:
            return "5K"
        return "4K"
    if short >= 1400:
        return "1440p"
    if short >= 1000:
        return "1080p"
    if short >= 700:
        return "720p"
    return "SD"


def quality_of(meta: dict) -> dict:
    """Quality facts for a scene from its Stash file metadata: resolution class,
    megabits/s, frame rate, codec, and bits per pixel per frame (bpp) — bitrate
    normalized by resolution and fps, so a 1080p and a 4K file are comparable."""
    w, h = meta.get("width"), meta.get("height")
    try:
        br = float(meta.get("bit_rate") or 0)
    except (TypeError, ValueError):
        br = 0.0
    try:
        fps = float(meta.get("frame_rate") or 0)
    except (TypeError, ValueError):
        fps = 0.0
    bpp = None
    try:
        if br > 0 and fps > 0 and int(w) > 0 and int(h) > 0:
            bpp = br / (int(w) * int(h) * fps)
    except (TypeError, ValueError):
        bpp = None
    try:
        dims = (int(w), int(h)) if int(w) > 0 and int(h) > 0 else None
    except (TypeError, ValueError):
        dims = None
    return {
        "res": res_class(w, h),
        "w": dims[0] if dims else None,
        "h": dims[1] if dims else None,
        "mbps": round(br / 1e6, 1) if br > 0 else None,
        "fps": round(fps, 2) if fps > 0 else None,
        "codec": (meta.get("video_codec") or "").lower() or None,
        "bpp": round(bpp, 4) if bpp is not None else None,
    }
