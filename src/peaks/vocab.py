"""Default vocabulary for CLIP zero-shot moment classification.

These descriptive prompts are scored against a frame's CLIP embedding to surface
"what CLIP sees" in a moment and to drive auto-tagging. It's deliberately broad
and caption-flavoured (CLIP was trained on web captions, so natural noun-phrases
score better than bare keywords). Users override the whole list by dropping a
`vocab.txt` (one prompt per line) into /config — editable live from the GUI.

Kept descriptive and functional on purpose.
"""

from __future__ import annotations

DEFAULT_VOCAB: list[str] = [
    # --- camera / framing ---
    "a close-up shot", "an extreme close-up", "a medium shot", "a wide shot",
    "a full body shot", "a point-of-view shot", "an overhead angle",
    "a low angle looking up", "a mirror reflection", "a selfie", "a webcam view",
    "a handheld camera", "a professional studio shot", "an amateur home video",
    # --- setting / location ---
    "a bedroom", "a bathroom", "a shower", "a bathtub", "a kitchen",
    "a living room", "an office", "a hotel room", "a locker room", "a gym",
    "a swimming pool", "a hot tub", "a beach", "outdoors in nature", "a garden",
    "a car interior", "a public place", "a nightclub", "a stage", "a plain backdrop",
    # --- wardrobe ---
    "wearing lingerie", "wearing a bra and panties", "wearing a dress",
    "wearing a skirt", "wearing a bikini", "wearing a swimsuit", "wearing jeans",
    "wearing yoga pants", "wearing a uniform", "wearing a costume", "wearing high heels",
    "wearing boots", "wearing stockings", "wearing fishnets", "wearing a collar",
    "topless", "fully nude", "partially clothed", "wearing casual clothes",
    "wet clothing", "wearing glasses",
    # --- appearance ---
    "a blonde woman", "a brunette woman", "a woman with red hair",
    "a woman with black hair", "dyed or colorful hair", "long hair", "short hair",
    "a person with tattoos", "a person with piercings", "an athletic body",
    "a curvy body", "a slim body", "tan skin", "pale skin",
    # --- count / people ---
    "one person alone", "two people together", "a group of people", "a man and a woman",
    "two women",
    # --- pose / action ---
    "standing", "sitting", "kneeling", "lying on a bed", "bending over",
    "on hands and knees", "arching the back", "spreading the legs", "dancing",
    "stripping off clothes", "posing for the camera", "touching herself",
    "kissing", "an embrace", "a massage",
    # --- lighting / mood / style ---
    "dim mood lighting", "bright lighting", "natural daylight", "neon lighting",
    "candlelight", "a colorful background", "a dark background",
    "a high quality photo", "a grainy low quality photo", "black and white",
]
