"""Default vocabulary for CLIP zero-shot moment classification.

These are neutral, descriptive prompts scored against a frame's CLIP embedding
to surface "what CLIP sees" in a moment — framing, setting, attire, count,
posture, lighting. It's deliberately general; users can override the whole list
by dropping a `vocab.txt` (one prompt per line) into /config.

Kept plain and functional on purpose: this drives interpretable feedback and,
later, optional auto-tagging back to Stash.
"""

from __future__ import annotations

DEFAULT_VOCAB: list[str] = [
    # framing / camera
    "close-up shot", "wide shot", "point of view", "overhead angle", "mirror reflection",
    # setting
    "bedroom", "bathroom", "shower", "kitchen", "living room", "office",
    "outdoors", "poolside", "beach", "car interior", "hotel room",
    # attire
    "lingerie", "dress", "skirt", "bikini", "swimsuit", "high heels",
    "boots", "stockings", "nude", "partially clothed", "casual clothes",
    # count
    "one person", "two people", "group of people",
    # posture / motion
    "standing", "sitting", "kneeling", "lying down", "bending over", "dancing",
    # appearance
    "blonde hair", "brunette hair", "red hair", "tattoos", "glasses",
    # lighting / mood
    "dim lighting", "bright lighting", "natural light", "neon lighting",
]
