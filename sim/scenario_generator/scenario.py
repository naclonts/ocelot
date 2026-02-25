"""
ScenarioConfig dataclass + ScenarioGenerator for Phase 2 data collection.

Each scenario is deterministically reproducible from a single integer seed.
ScenarioGenerator.sample(seed) draws all parameters from seeded random, then
calls labels.assign_label() to attach a natural-language instruction.

Standard project layout assumed:
  sim/scenario_generator/    ← faces_dir (contains face_descriptions*.json)
  sim/assets/faces/          ← face PNG images (faces_dir.parent / "assets" / "faces")
  sim/assets/backgrounds/    ← backgrounds_dir (contains backgrounds_manifest.json)
"""

import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FaceConfig:
    face_id:      str    # "face_042" — stem of PNG in sim/assets/faces/
    texture_path: str    # absolute path, resolved at sample time
    initial_x:    float  # distance in front of robot [1.0–3.0 m]
    initial_y:    float  # lateral offset [-1.0–1.0 m]
    initial_z:    float  # height [0.5–1.5 m]
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
    ambient_rgb:              tuple  # (r, g, b) each [0.2–0.8]
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

        # Glob all face_descriptions*.json and merge into one pool.
        self._faces: list[dict] = []
        for p in sorted(faces_dir.glob("face_descriptions*.json")):
            self._faces.extend(json.loads(p.read_text()))
        if not self._faces:
            raise ValueError(f"No face_descriptions*.json found in {faces_dir}")

        # Load background manifest.
        manifest_path = backgrounds_dir / "backgrounds_manifest.json"
        if manifest_path.exists():
            self._backgrounds: list[dict] = json.loads(manifest_path.read_text())
        else:
            # Graceful fallback: synthesise a single plain-white entry so the
            # generator can be used without downloading background assets.
            self._backgrounds = [{"id": "plain_white", "tags": ["plain"],
                                   "file": "plain_white.png"}]

        self._backgrounds_dir = backgrounds_dir
        self._faces_assets_dir = faces_dir.parent / "assets" / "faces"

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
            faces.append(FaceConfig(
                face_id=fid,
                texture_path=texture_path,
                initial_x=rng.uniform(1.0, 3.0),
                initial_y=rng.uniform(-1.0, 1.0),
                initial_z=rng.uniform(0.5, 1.5),
                motion=rng.choices(
                    self._MOTION_TYPES, weights=self._MOTION_WEIGHTS, k=1
                )[0],
                speed=rng.uniform(0.05, 0.5),
                period=rng.uniform(6.0, 20.0),
            ))

        target_face_idx = rng.randint(0, len(faces) - 1)

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
        ambient_rgb = (
            rng.uniform(0.2, 0.8),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.2, 0.8),
        )

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
            "ambient_rgb": list(ambient_rgb),
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
            ambient_rgb=ambient_rgb,
            distractor_count=distractor_count,
            distractors=distractors,
            camera_noise_sigma=camera_noise_sigma,
            camera_brightness_offset=camera_brightness_offset,
            label_key=label_key,
            language_label=language_label,
        )
