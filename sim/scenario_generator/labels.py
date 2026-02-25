"""
Language label system for scenario generator.

assign_label(faces, target_idx, face_attrs, rng) selects the most informative
semantic condition (per a priority order) and returns a randomly chosen label
template for that condition.
"""

import random
from typing import Optional


# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------

LABEL_REGISTRY: dict[str, list[str]] = {

    # ── Multi-face: distinguishing attribute ──────────────────────────────────
    # {attr} is substituted at generation time.

    "multi_attr": [
        "track the person wearing the {attr}",
        "follow the one with the {attr}",
        "track the {attr}",
        "follow the {attr}",
        "look at the one with the {attr} on",
        "keep your eye on the person with the {attr}",
        "find the {attr} and follow them",
    ],

    # ── Multi-face: positional ────────────────────────────────────────────────

    "multi_left": [
        "follow the person on the left",
        "track the leftmost person",
        "look to the left one",
        "keep your eye on the person furthest left",
        "the one on the left — follow them",
    ],

    "multi_right": [
        "follow the person on the right",
        "track the rightmost person",
        "look to the right one",
        "keep your eye on the person furthest right",
        "the one on the right — follow them",
    ],

    # ── Multi-face: proximity ─────────────────────────────────────────────────

    "multi_closest": [
        "track the closest person",
        "follow the nearest one",
        "keep the person closest to you centered",
        "track whoever is nearest",
        "focus on the closest face",
        "stay with the one right in front of you",
    ],
}


# ---------------------------------------------------------------------------
# Attribute display strings
# ---------------------------------------------------------------------------

ATTR_DISPLAY: dict[str, str] = {
    "baseball_cap":         "baseball cap",
    "beanie":               "beanie",
    "fedora":               "fedora",
    "wide_brim":            "sun hat",
    "pirate_hat":           "pirate hat",
    "cowboy_hat":           "cowboy hat",
    "reading":              "glasses",
    "round":                "round glasses",
    "rectangular":          "glasses",
    "thick_rimmed":         "thick-rimmed glasses",
    "sunglasses":           "sunglasses",
    "stubble":              "stubble",
    "beard":                "beard",
    "mustache":             "mustache",
    "goatee":               "goatee",
    "over_ear_headphones":  "headphones",
    "scarf":                "scarf",
}


# ---------------------------------------------------------------------------
# Distinguishing attribute resolver
# ---------------------------------------------------------------------------

def _find_distinguishing_attr(
    target_id: str,
    other_ids: list[str],
    face_attrs: dict[str, dict],
) -> Optional[str]:
    """
    Return a display string for the first attribute that is non-null for the
    target face and null (or absent) for ALL other faces.  Returns None if no
    such attribute exists or if target_id is not in face_attrs.

    Priority: hat > sunglasses > any glasses > facial_hair > accessory.
    """
    if target_id not in face_attrs:
        return None
    t = face_attrs[target_id]

    def others_lack(attr_key: str) -> bool:
        """True if all other faces have attr_key == None (or are unknown)."""
        for fid in other_ids:
            if face_attrs.get(fid, {}).get(attr_key) is not None:
                return False
        return True

    # 1. Hat
    hat = t.get("hat")
    if hat is not None and others_lack("hat"):
        return ATTR_DISPLAY.get(hat, hat)

    # 2. Sunglasses specifically (more distinctive than plain glasses)
    glasses = t.get("glasses")
    if glasses == "sunglasses":
        others_no_sunglasses = all(
            face_attrs.get(fid, {}).get("glasses") != "sunglasses"
            for fid in other_ids
        )
        if others_no_sunglasses:
            return ATTR_DISPLAY["sunglasses"]

    # 3. Any glasses
    if glasses is not None and others_lack("glasses"):
        return ATTR_DISPLAY.get(glasses, glasses)

    # 4. Facial hair
    fh = t.get("facial_hair")
    if fh is not None and others_lack("facial_hair"):
        return ATTR_DISPLAY.get(fh, fh)

    # 5. Accessory
    acc = t.get("accessory")
    if acc is not None and others_lack("accessory"):
        return ATTR_DISPLAY.get(acc, acc)

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assign_label(
    faces: list,             # list[FaceConfig] — from scenario.py
    target_idx: int,
    face_attrs: dict,        # {face_id: attr_dict} from face_descriptions*.json
    rng: random.Random,
) -> tuple[str, str]:
    """
    Select the most informative semantic label for this scenario.

    Returns (label_key, language_label).

    Single-face: always returns ("track", "track the face") — no disambiguation needed.

    Multi-face priority (first match wins):
        1. multi_attr    — target has distinguishing hat/glasses/beard
        2. multi_left    — target is leftmost (min initial_y)
           multi_right   — target is rightmost (max initial_y)
        3. multi_closest — fallback
    """
    target = faces[target_idx]
    n = len(faces)

    if n == 1:
        return "track", "track the face"

    # Multi-face ──────────────────────────────────────────────────────────────

    # 1. Distinguishing attribute
    other_ids = [f.face_id for i, f in enumerate(faces) if i != target_idx]
    attr = _find_distinguishing_attr(target.face_id, other_ids, face_attrs)
    if attr is not None:
        template = rng.choice(LABEL_REGISTRY["multi_attr"])
        return "multi_attr", template.format(attr=attr)

    # 2. Leftmost / rightmost by initial_y
    by_y = sorted(range(n), key=lambda i: faces[i].initial_y)
    if by_y[0] == target_idx:
        key = "multi_left"
    elif by_y[-1] == target_idx:
        key = "multi_right"
    else:
        # 3. Closest (min initial_x) — or fallback if also not closest
        key = "multi_closest"

    template = rng.choice(LABEL_REGISTRY[key])
    return key, template
