#!/usr/bin/env python3
"""
Face description generator.

Produces a list of natural-language prompts suitable for AI image generation
(DALL-E, Stable Diffusion, etc.).  Each prompt describes one face texture for
the sim face billboard.  Attributes are stored alongside the prompt so the
scenario generator can use them for language label generation later
(e.g. "follow the person in the hat").

Usage (as a module):
    from sim.scenario_generator.face_descriptions import generate_face_descriptions
    faces = generate_face_descriptions(count=80, seed=42)

Usage (as a script):
    python3 -m sim.scenario_generator.face_descriptions --count 80 --out sim/faces/
    python3 sim/scenario_generator/face_descriptions.py --count 80 --out sim/faces/
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Attribute pools
# ---------------------------------------------------------------------------

# (internal_key, display_string, relative_weight)
_GENDER = [
    ("man",   "man",   1),
    ("woman", "woman", 1),
]

_AGE = [
    ("young_adult", "young",        1),
    ("middle_aged", "middle-aged",  1),
    ("older",       "older",        1),
]

_SKIN_TONE = [
    ("light",       "light",        1),
    ("medium_light","medium-light", 1),
    ("medium",      "medium",       1),
    ("olive",       "olive",        1),
    ("brown",       "brown",        1),
    ("dark",        "dark",         1),
]

# hair_length: None means bald (handled separately)
_HAIR_LENGTH = [
    (None,      None,            1),   # bald
    ("cropped", "very short",    2),
    ("short",   "short",         3),
    ("medium",  "medium-length", 2),
    ("long",    "long",          2),
]

_HAIR_COLOR = [
    ("black",       "black",      3),
    ("dark_brown",  "dark brown", 3),
    ("brown",       "brown",      3),
    ("auburn",      "auburn",     1),
    ("blonde",      "blonde",     2),
    ("red",         "red",        1),
    ("gray",        "gray",       1),
    ("white",       "white",      1),
]

_HAIR_STYLE = [
    ("straight", "straight",   4),
    ("wavy",     "wavy",       2),
    ("curly",    "curly",      2),
    ("afro",     "afro",       1),
    ("braided",  "braided",    1),
    ("dreadlocks","dreadlocked",1),
]

# Facial hair — only sampled for men; None = clean-shaven
_FACIAL_HAIR_MAN = [
    (None,        None,             4),   # clean-shaven
    ("stubble",   "light stubble",  2),
    ("beard",     "a full beard",   2),
    ("mustache",  "a mustache",     1),
    ("goatee",    "a goatee",       1),
]

# Hats — sampled for all; None = no hat (higher weight)
_HAT = [
    (None,           None,                    5),
    ("baseball_cap", "a baseball cap",        2),
    ("beanie",       "a beanie",              1),
    ("fedora",       "a fedora",              1),
    ("wide_brim",    "a wide-brimmed sun hat",1),
    ("pirate_hat",   "a pirate hat",          1),
    ("cowboy_hat",   "a cowboy hat",          1),
]

# Glasses — None = no glasses (higher weight)
_GLASSES = [
    (None,          None,                     4),
    ("reading",     "reading glasses",        2),
    ("round",       "round glasses",          1),
    ("rectangular", "rectangular glasses",    2),
    ("thick_rimmed","thick-rimmed glasses",   1),
    ("sunglasses",  "dark sunglasses",        1),
]

_EXPRESSION = [
    ("neutral",       "a neutral expression",  3),
    ("slight_smile",  "a slight smile",        2),
    ("big_grin",      "a big grin",            1),
    ("serious",       "a serious expression",  1),
]

# Shirt type and color — only used in prompts for chest_up / waist_up shots
_SHIRT_TYPE = [
    ("t_shirt",  "t-shirt", 2),
    ("hoodie",   "hoodie",  1),
    ("sweater",  "sweater", 1),
]

_SHIRT_COLOR = [
    ("red",   "red",   1),
    ("blue",  "blue",  1),
    ("green", "green", 1),
    ("black", "black", 1),
    ("white", "white", 1),
]

# Over-clothing accessories — None = nothing (higher weight)
_ACCESSORY = [
    (None,                   None,                   5),
    ("over_ear_headphones",  "over-ear headphones",  1),
    ("scarf",                "a scarf",              1),
]

# How much of the body is visible.  Randomized so the bottom cutoff line is
# never a reliable cue the policy can latch onto.
# Weights give: waist_up 50%, chest_up 30%, neck_up 20%.
_CROP_LEVEL = [
    ("neck_up",   "portrait from the neck up",                                           2),
    ("chest_up",  "upper body portrait showing face, neck, shoulders, and upper chest",  3),
    ("waist_up",  "half-body portrait showing full figure from crown to hips, entire torso and both arms visible down to the waist", 5),
]

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, pool):
    """Pick one item from a weighted pool list of (key, display, weight)."""
    keys    = [p[0] for p in pool]
    weights = [p[2] for p in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaceDescription:
    face_id:      str
    gender:       str
    age_range:    str
    skin_tone:    str
    # hair
    hair_length:  Optional[str]   # None → bald
    hair_color:   Optional[str]   # None → bald
    hair_style:   Optional[str]   # None → bald
    # accessories
    facial_hair:  Optional[str]   # None → clean-shaven / N/A for women
    hat:          Optional[str]   # None → no hat
    glasses:      Optional[str]   # None → no glasses
    accessory:    Optional[str]   # None → none; over_ear_headphones | scarf
    # clothing — None when crop_level is neck_up
    shirt:        Optional[str]   # None when not visible; e.g. "red t-shirt"
    # affect
    expression:   str
    # composition — randomized so bottom cutoff is never a consistent cue
    crop_level:   str             # neck_up | chest_up | waist_up
    # derived
    prompt:       str             # ready-to-use image-gen prompt


def _build_prompt(attrs: dict) -> str:
    """
    Convert a dict of display strings (not internal keys) into a single
    image-generation prompt.
    """
    parts = []

    # Opening line
    age   = attrs["age_display"]
    skin  = attrs["skin_display"]
    gender = attrs["gender"]
    article = "an" if age[0] in "aeiou" else "a"
    parts.append(f"Portrait photo of {article} {age} {skin}-skinned {gender}")

    # Hair
    if attrs["hair_length_display"] is None:
        parts.append("who is completely bald")
    else:
        hair_bits = [attrs["hair_length_display"]]
        if attrs["hair_style_display"] and attrs["hair_style_display"] != "straight":
            hair_bits.append(attrs["hair_style_display"])
        if attrs["hair_color_display"]:
            hair_bits.append(attrs["hair_color_display"])
        parts.append("with " + " ".join(hair_bits) + " hair")

    # Facial hair
    if attrs.get("facial_hair_display"):
        parts.append(f"and {attrs['facial_hair_display']}")

    # Hat
    if attrs.get("hat_display"):
        parts.append(f"wearing {attrs['hat_display']}")

    # Glasses
    if attrs.get("glasses_display"):
        parts.append(f"wearing {attrs['glasses_display']}")

    # Shirt (only included for chest_up / waist_up)
    if attrs.get("shirt_display"):
        parts.append(f"wearing a {attrs['shirt_display']}")

    # Over-clothing accessory
    if attrs.get("accessory_display"):
        parts.append(f"wearing {attrs['accessory_display']}")

    # Expression
    parts.append(f"with {attrs['expression_display']}")

    # Quality / composition suffix — crop_level controls how much body is visible
    crop_desc = attrs.get("crop_level_display", "upper body portrait showing face, neck, shoulders, and upper chest")
    parts.append(
        f"facing the camera, photorealistic, {crop_desc}, "
        f"complete head fully in frame with crown and top of hair visible, "
        f"soft professional lighting, high resolution"
    )

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_face_descriptions(
    count: int = 80,
    seed: int = 42,
) -> list[FaceDescription]:
    """
    Generate `count` diverse face descriptions.

    Diversity strategy
    ------------------
    We cycle through the major axes (gender, age, skin_tone) in a round-robin
    before sampling the rest randomly.  This guarantees even coverage even for
    small `count` values, while still leaving the secondary attributes (hair,
    accessories) fully random for variety.
    """
    rng = random.Random(seed)
    faces: list[FaceDescription] = []

    # Build cycling pools as (key, display) tuples, weighted then shuffled
    gender_cycle = [(k, k) for k, _, w in _GENDER     for _ in range(w)]
    age_cycle    = [(k, d) for k, d, w in _AGE        for _ in range(w)]
    skin_cycle   = [(k, d) for k, d, w in _SKIN_TONE  for _ in range(w)]
    rng.shuffle(gender_cycle)
    rng.shuffle(age_cycle)
    rng.shuffle(skin_cycle)

    for i in range(count):
        face_id = f"face_{i+1:03d}"

        # Primary axes — cycled for guaranteed even coverage
        gender_key              = gender_cycle[i % len(gender_cycle)][0]
        age_key,  age_disp      = age_cycle[i   % len(age_cycle)]
        skin_key, skin_disp     = skin_cycle[i  % len(skin_cycle)]

        # Hair
        hair_len_entry = _weighted_choice(rng, _HAIR_LENGTH)
        hair_len_key, hair_len_disp = hair_len_entry[0], hair_len_entry[1]
        bald = (hair_len_key is None)

        if bald:
            hair_color_key = hair_color_disp = None
            hair_style_key = hair_style_disp = None
        else:
            hc = _weighted_choice(rng, _HAIR_COLOR)
            hair_color_key, hair_color_disp = hc[0], hc[1]
            hs = _weighted_choice(rng, _HAIR_STYLE)
            hair_style_key, hair_style_disp = hs[0], hs[1]

        # Facial hair (men only)
        if gender_key == "man":
            fh = _weighted_choice(rng, _FACIAL_HAIR_MAN)
            facial_hair_key, facial_hair_disp = fh[0], fh[1]
        else:
            facial_hair_key = facial_hair_disp = None

        # Accessories
        hat = _weighted_choice(rng, _HAT)
        hat_key, hat_disp = hat[0], hat[1]

        gl = _weighted_choice(rng, _GLASSES)
        glasses_key, glasses_disp = gl[0], gl[1]

        acc = _weighted_choice(rng, _ACCESSORY)
        accessory_key, accessory_disp = acc[0], acc[1]

        # Expression
        expr = _weighted_choice(rng, _EXPRESSION)
        expr_key, expr_disp = expr[0], expr[1]

        # Crop level — randomized so bottom cutoff is never a consistent cue
        crop = _weighted_choice(rng, _CROP_LEVEL)
        crop_key, crop_disp = crop[0], crop[1]

        # Shirt — always sampled, but only shown in prompt for chest_up / waist_up
        st = _weighted_choice(rng, _SHIRT_TYPE)
        shirt_type_key, shirt_type_disp = st[0], st[1]
        sc = _weighted_choice(rng, _SHIRT_COLOR)
        shirt_color_key, shirt_color_disp = sc[0], sc[1]
        shirt_key  = f"{shirt_color_key}_{shirt_type_key}"
        shirt_disp = f"{shirt_color_disp} {shirt_type_disp}" if crop_key != "neck_up" else None

        # Build prompt
        prompt_attrs = dict(
            gender              = gender_key,
            age_display         = age_disp,
            skin_display        = skin_disp,
            hair_length_display = hair_len_disp,
            hair_color_display  = hair_color_disp,
            hair_style_display  = hair_style_disp,
            facial_hair_display = facial_hair_disp,
            hat_display         = hat_disp,
            glasses_display     = glasses_disp,
            accessory_display   = accessory_disp,
            shirt_display       = shirt_disp,
            expression_display  = expr_disp,
            crop_level_display  = crop_disp,
        )
        prompt = _build_prompt(prompt_attrs)

        faces.append(FaceDescription(
            face_id      = face_id,
            gender       = gender_key,
            age_range    = age_key,
            skin_tone    = skin_key,
            hair_length  = hair_len_key,
            hair_color   = hair_color_key,
            hair_style   = hair_style_key,
            facial_hair  = facial_hair_key,
            hat          = hat_key,
            glasses      = glasses_key,
            accessory    = accessory_key,
            shirt        = shirt_key if crop_key != "neck_up" else None,
            expression   = expr_key,
            crop_level   = crop_key,
            prompt       = prompt,
        ))

    return faces


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate face description prompts for AI image generation."
    )
    parser.add_argument(
        "--count", type=int, default=80,
        help="Number of face descriptions to generate (default: 80)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("sim/faces"),
        help="Output directory (default: sim/faces/)"
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = generate_face_descriptions(count=args.count, seed=args.seed)

    # Write full JSON (attributes + prompts)
    json_path = out_dir / "face_descriptions.json"
    with open(json_path, "w") as f:
        json.dump([asdict(face) for face in faces], f, indent=2)
    print(f"Wrote {len(faces)} face descriptions → {json_path}")

    # Print summary
    print()
    print("=== Attribute coverage ===")
    from collections import Counter
    attrs_to_check = ["gender", "age_range", "skin_tone", "crop_level", "hat", "glasses", "facial_hair", "accessory", "shirt"]
    for attr in attrs_to_check:
        counts = Counter(getattr(f, attr) for f in faces)
        print(f"  {attr}:")
        for val, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {str(val):<20s} {n:3d}  ({100*n/len(faces):.0f}%)")


if __name__ == "__main__":
    main()
