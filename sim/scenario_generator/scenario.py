"""
ScenarioConfig dataclass + ScenarioGenerator for synthetic data collection.

Each scenario is deterministically reproducible from a single integer seed.
ScenarioGenerator.sample(seed) draws all parameters from seeded random, then
calls labels.assign_label() to attach a natural-language instruction.

Standard project layout assumed:
  sim/scenario_generator/    ← faces_dir (contains face_descriptions*.json)
  sim/assets/faces/          ← face PNG images (faces_dir.parent / "assets" / "faces")
  sim/assets/backgrounds/    ← backgrounds_dir (image files; DVC-tracked)
  sim/scenario_generator/backgrounds_manifest.json  ← git-tracked metadata
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

# Keep faces within camera vertical FOV — same constants as motion.py.
_VFOV_HALF = math.radians(20)
_CAM_Z     = 0.07   # camera z in world coords (m)


@dataclass
class FaceConfig:
    face_id:      str    # "face_042" — stem of PNG in sim/assets/faces/
    texture_path: str    # absolute path, resolved at sample time
    initial_x:    float  # distance in front of robot [2.0–4.0 m]
    initial_y:    float  # lateral offset [-1.0–1.0 m]
    initial_z:    float  # height — depth-dependent FOV range, floor 0.3 m
    motion:       str    # static | linear_drift | sinusoidal | random_walk
    speed:        float  # m/s [0.05–0.5] — peak velocity for all patterns
    period:       float  # seconds [6.0–20.0] — sinusoidal; ignored otherwise


@dataclass
class DistractorConfig:
    shape:     str    # sphere | box
    color_rgb: tuple  # (r, g, b) each channel [0.2–0.9]
    initial_x: float
    initial_y: float
    initial_z: float
    speed:     float  # [0.02–0.2 m/s]


@dataclass
class ScenarioConfig:
    scenario_id:              str
    seed:                     int
    faces:                    list   # list[FaceConfig]
    target_face_idx:          int
    background_id:            str
    background_path:          str
    lighting_azimuth_deg:     float  # [0–360]
    lighting_elevation_deg:   float  # [15–75]
    lighting_intensity:       float  # [0.5–2.0]
    key_color_rgb:            tuple  # (r, g, b) each [0.85–1.0] — subtle warm/cool tint
    ambient_rgb:              tuple  # (r, g, b) each [0.2–0.8] — fill light color
    fill_intensity:           float  # [0.2–0.9] — independent key-to-fill ratio
    distractor_count:         int    # [0–2]
    distractors:              list   # list[DistractorConfig]
    camera_noise_sigma:       float  # [0.0–0.015]
    camera_brightness_offset: float  # [-20–+20]
    label_key:                str
    language_label:           str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        d = dict(d)
        d["faces"] = [FaceConfig(**f) for f in d["faces"]]
        d["distractors"] = [
            DistractorConfig(
                shape=dist["shape"],
                color_rgb=tuple(dist["color_rgb"]),
                initial_x=dist["initial_x"],
                initial_y=dist["initial_y"],
                initial_z=dist["initial_z"],
                speed=dist["speed"],
            )
            for dist in d["distractors"]
        ]
        d["ambient_rgb"] = tuple(d["ambient_rgb"])
        d.setdefault("key_color_rgb", (1.0, 1.0, 1.0))
        d["key_color_rgb"] = tuple(d["key_color_rgb"])
        d.setdefault("fill_intensity", 0.6)
        return cls(**d)


class ScenarioGenerator:
    """
    Generates randomized ScenarioConfig instances from integer seeds.

    Args:
        faces_dir:       Directory containing face_descriptions*.json files.
                         Also used to derive the faces assets dir:
                         faces_dir.parent / "assets" / "faces".
        backgrounds_dir: Directory containing backgrounds_manifest.json and
                         the actual background image files.
    """

    # Motion distribution: 30% static, 70% moving split equally.
    # weights=[9,7,7,7] → 9/30 = 30% static.
    _MOTION_TYPES = ["static", "linear_drift", "sinusoidal", "random_walk"]
    _MOTION_WEIGHTS = [9, 7, 7, 7]

    def __init__(self, faces_dir: Path, backgrounds_dir: Path):
        faces_dir = Path(faces_dir)
        backgrounds_dir = Path(backgrounds_dir)

        # Glob all face_descriptions*.json, merge, and deduplicate by face_id
        # (face_descriptions_003.json is a superset that overlaps earlier files).
        faces_assets_dir = faces_dir.parent / "assets" / "faces"
        seen_ids: set[str] = set()
        raw_faces: list[dict] = []
        for p in sorted(faces_dir.glob("face_descriptions*.json")):
            for entry in json.loads(p.read_text()):
                fid = entry["face_id"]
                if fid not in seen_ids:
                    seen_ids.add(fid)
                    raw_faces.append(entry)
        if not raw_faces:
            raise ValueError(f"No face_descriptions*.json found in {faces_dir}")
        # Keep only faces whose PNG asset is present on disk.
        self._faces: list[dict] = [
            f for f in raw_faces
            if (faces_assets_dir / f"{f['face_id']}.png").exists()
        ]
        if not self._faces:
            raise ValueError(f"No face PNG assets found in {faces_assets_dir}")

        # Load background manifest. Lives in scenario_generator/ (git-tracked),
        # not in assets/ (DVC-tracked), so it's available after a plain git clone.
        manifest_path = Path(__file__).resolve().parent / "backgrounds_manifest.json"
        if manifest_path.exists():
            self._backgrounds: list[dict] = json.loads(manifest_path.read_text())
        else:
            # Graceful fallback: synthesise a single plain-white entry so the
            # generator can be used without downloading background assets.
            self._backgrounds = [{"id": "plain_white", "tags": ["plain"],
                                   "file": "plain_white.png"}]

        self._backgrounds_dir = backgrounds_dir
        self._faces_assets_dir = faces_assets_dir

    def sample(self, seed: int) -> "ScenarioConfig":
        from sim.scenario_generator.labels import assign_label  # lazy import avoids cycles

        rng = random.Random(seed)

        # ── Number of faces ──────────────────────────────────────────────────
        # ~60% single, ~30% two, ~10% three
        n_faces = rng.choices([1, 2, 3], weights=[6, 3, 1], k=1)[0]

        # ── Face configs ─────────────────────────────────────────────────────
        faces: list[FaceConfig] = []
        for _ in range(n_faces):
            entry = rng.choice(self._faces)
            fid = entry["face_id"]
            texture_path = str(
                (self._faces_assets_dir / f"{fid}.png").resolve()
            )
            face_x = rng.uniform(2.0, 4.0)
            face_y = rng.uniform(-1.0, 1.0)
            _dz    = face_x * math.tan(_VFOV_HALF)
            face_z = rng.uniform(max(0.3, _CAM_Z - _dz), min(1.5, _CAM_Z + _dz))
            faces.append(FaceConfig(
                face_id=fid,
                texture_path=texture_path,
                initial_x=face_x,
                initial_y=face_y,
                initial_z=face_z,
                motion=rng.choices(
                    self._MOTION_TYPES, weights=self._MOTION_WEIGHTS, k=1
                )[0],
                speed=rng.uniform(0.05, 0.5),
                period=rng.uniform(6.0, 20.0),
            ))

        # For multi-face scenarios, constrain the target to be the leftmost or
        # rightmost face by initial_y.  This guarantees assign_label() always
        # finds a position-based label (multi_left / multi_right) — or an
        # attribute-based one if the face happens to have a distinguishing
        # attribute.  A target in the middle of 3 faces with no attribute
        # cannot be described unambiguously, so we never produce that case.
        if n_faces > 1:
            by_y = sorted(range(n_faces), key=lambda i: faces[i].initial_y)
            target_face_idx = rng.choice([by_y[0], by_y[-1]])
        else:
            target_face_idx = 0

        # ── Background ───────────────────────────────────────────────────────
        bg = rng.choice(self._backgrounds)
        background_id = bg["id"]
        background_path = str(
            (self._backgrounds_dir / bg["file"]).resolve()
        )

        # ── Lighting ─────────────────────────────────────────────────────────
        lighting_azimuth_deg   = rng.uniform(0, 360)
        lighting_elevation_deg = rng.uniform(15, 75)
        lighting_intensity     = rng.uniform(0.5, 2.0)
        # Subtle warm/cool tint on the key light — each channel [0.85, 1.0]
        # so the deviation from pure white is at most 0.15 per channel.
        key_color_rgb = (
            rng.uniform(0.85, 1.0),
            rng.uniform(0.85, 1.0),
            rng.uniform(0.85, 1.0),
        )
        ambient_rgb = (
            rng.uniform(0.2, 0.8),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.2, 0.8),
        )
        fill_intensity = rng.uniform(0.2, 0.9)

        # ── Distractors ──────────────────────────────────────────────────────
        distractor_count = rng.randint(0, 2)
        distractors: list[DistractorConfig] = []
        for _ in range(distractor_count):
            distractors.append(DistractorConfig(
                shape=rng.choice(["sphere", "box"]),
                color_rgb=(
                    rng.uniform(0.2, 0.9),
                    rng.uniform(0.2, 0.9),
                    rng.uniform(0.2, 0.9),
                ),
                initial_x=rng.uniform(1.0, 3.0),
                initial_y=rng.uniform(-1.0, 1.0),
                initial_z=rng.uniform(0.3, 1.5),
                speed=rng.uniform(0.02, 0.2),
            ))

        # ── Camera augmentation ──────────────────────────────────────────────
        camera_noise_sigma       = rng.uniform(0.0, 0.015)
        camera_brightness_offset = rng.uniform(-20.0, 20.0)

        # ── Language label ───────────────────────────────────────────────────
        face_attrs = {f["face_id"]: f for f in self._faces}
        label_key, language_label = assign_label(
            faces, target_face_idx, face_attrs, rng
        )

        # ── scenario_id: SHA1[:8] of all params (for dedup / split) ─────────
        config_for_hash = {
            "seed": seed,
            "faces": [asdict(f) for f in faces],
            "target_face_idx": target_face_idx,
            "background_id": background_id,
            "background_path": background_path,
            "lighting_azimuth_deg": lighting_azimuth_deg,
            "lighting_elevation_deg": lighting_elevation_deg,
            "lighting_intensity": lighting_intensity,
            "key_color_rgb": list(key_color_rgb),
            "ambient_rgb": list(ambient_rgb),
            "fill_intensity": fill_intensity,
            "distractor_count": distractor_count,
            "distractors": [asdict(d) for d in distractors],
            "camera_noise_sigma": camera_noise_sigma,
            "camera_brightness_offset": camera_brightness_offset,
            "label_key": label_key,
            "language_label": language_label,
        }
        scenario_id = hashlib.sha1(
            json.dumps(config_for_hash, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        return ScenarioConfig(
            scenario_id=scenario_id,
            seed=seed,
            faces=faces,
            target_face_idx=target_face_idx,
            background_id=background_id,
            background_path=background_path,
            lighting_azimuth_deg=lighting_azimuth_deg,
            lighting_elevation_deg=lighting_elevation_deg,
            lighting_intensity=lighting_intensity,
            key_color_rgb=key_color_rgb,
            ambient_rgb=ambient_rgb,
            fill_intensity=fill_intensity,
            distractor_count=distractor_count,
            distractors=distractors,
            camera_noise_sigma=camera_noise_sigma,
            camera_brightness_offset=camera_brightness_offset,
            label_key=label_key,
            language_label=language_label,
        )
