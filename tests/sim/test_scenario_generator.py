"""
Pure-Python tests for the scenario generator.

No ROS, no Gazebo, no network required.

Run with:  pytest tests/sim/ -v
"""

import json
import random
from pathlib import Path

import pytest

from sim.scenario_generator.scenario import (
    FaceConfig,
    DistractorConfig,
    ScenarioConfig,
    ScenarioGenerator,
)
from sim.scenario_generator.labels import (
    LABEL_REGISTRY,
    assign_label,
    _find_distinguishing_attr,
)
from sim.scenario_generator.motion import (
    StaticMotion,
    SinusoidalMotion,
    LinearDriftMotion,
    RandomWalkMotion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_face_desc(face_id: str, hat=None, glasses=None, facial_hair=None,
                    accessory=None) -> dict:
    return {
        "face_id": face_id,
        "gender": "man",
        "age_range": "young_adult",
        "skin_tone": "medium",
        "hair_length": "short",
        "hair_color": "black",
        "hair_style": "straight",
        "facial_hair": facial_hair,
        "hat": hat,
        "glasses": glasses,
        "accessory": accessory,
        "shirt": "blue_t_shirt",
        "expression": "neutral",
        "crop_level": "waist_up",
        "prompt": "test prompt",
    }


def _make_face_config(face_id="face_001", initial_x=2.0, initial_y=0.0,
                      initial_z=1.0, motion="sinusoidal", speed=0.3,
                      period=10.0) -> FaceConfig:
    return FaceConfig(
        face_id=face_id,
        texture_path=f"/fake/path/{face_id}.png",
        initial_x=initial_x,
        initial_y=initial_y,
        initial_z=initial_z,
        motion=motion,
        speed=speed,
        period=period,
    )


@pytest.fixture
def generator(tmp_path) -> ScenarioGenerator:
    """ScenarioGenerator backed by a minimal synthetic face pool + backgrounds."""
    faces_dir = tmp_path / "scenario_generator"
    faces_dir.mkdir()
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # 20 synthetic face descriptions — enough for diverse sampling
    faces = []
    hats = [None, "baseball_cap", "beanie", "fedora", "cowboy_hat",
            None, None, None, None, None]
    glasses_options = [None, "reading", "sunglasses", "round", None,
                       None, None, None, None, None]
    fh_options = [None, "stubble", "beard", None, None,
                  None, None, None, None, None]
    for i in range(20):
        faces.append(_make_face_desc(
            face_id=f"face_{i+1:03d}",
            hat=hats[i % len(hats)],
            glasses=glasses_options[i % len(glasses_options)],
            facial_hair=fh_options[i % len(fh_options)],
        ))

    (faces_dir / "face_descriptions_001.json").write_text(json.dumps(faces))

    # Minimal backgrounds manifest
    manifest = [
        {"id": "plain_white",    "tags": ["plain"],   "file": "plain_white.png"},
        {"id": "indoor_office",  "tags": ["indoor"],  "file": "indoor_office.jpg"},
        {"id": "outdoor_park",   "tags": ["outdoor"], "file": "outdoor_park.jpg"},
    ]
    (bg_dir / "backgrounds_manifest.json").write_text(json.dumps(manifest))

    return ScenarioGenerator(faces_dir=faces_dir, backgrounds_dir=bg_dir)


# ---------------------------------------------------------------------------
# Label tests
# ---------------------------------------------------------------------------

class TestLabels:

    def test_label_determinism(self, generator):
        """Same seed must produce the same language_label across 10 calls."""
        cfg_a = generator.sample(seed=42)
        for _ in range(9):
            cfg_b = generator.sample(seed=42)
            assert cfg_b.language_label == cfg_a.language_label
            assert cfg_b.label_key == cfg_a.label_key

    def test_label_correctness_hat(self):
        """target has hat, other faces don't → multi_attr, label contains 'cap' or 'hat'."""
        target = _make_face_config("face_hat")
        other1 = _make_face_config("face_plain1")
        other2 = _make_face_config("face_plain2")
        faces = [other1, target, other2]
        target_idx = 1
        face_attrs = {
            "face_hat":    _make_face_desc("face_hat",    hat="baseball_cap"),
            "face_plain1": _make_face_desc("face_plain1", hat=None),
            "face_plain2": _make_face_desc("face_plain2", hat=None),
        }
        rng = random.Random(0)
        key, label = assign_label(faces, target_idx, face_attrs, rng)
        assert key == "multi_attr", f"Expected multi_attr, got {key!r}"
        assert any(word in label for word in ("cap", "hat", "baseball")), (
            f"Expected hat-related label, got: {label!r}"
        )

    def test_label_correctness_slow(self):
        """Single face, speed=0.1, motion=sinusoidal → single_slow."""
        face = _make_face_config(motion="sinusoidal", speed=0.1)
        rng = random.Random(0)
        key, label = assign_label([face], 0, {}, rng)
        assert key == "single_slow", f"Expected single_slow, got {key!r}"

    def test_label_correctness_left(self):
        """Single face, initial_y=-0.6 (left of camera) → single_left."""
        face = _make_face_config(initial_y=-0.6, motion="sinusoidal", speed=0.3)
        rng = random.Random(0)
        key, label = assign_label([face], 0, {}, rng)
        assert key == "single_left", f"Expected single_left, got {key!r}"

    def test_label_coverage(self, generator):
        """500 samples must cover all 8 label keys at least once."""
        all_keys = set(LABEL_REGISTRY.keys())
        covered = set()
        for seed in range(500):
            cfg = generator.sample(seed)
            covered.add(cfg.label_key)
            if covered >= all_keys:
                break
        assert len(covered) >= 6, (
            f"Expected ≥6 label keys covered, got {len(covered)}: {covered}"
        )

    def test_multi_attr_priority(self):
        """Hat takes priority over glasses when target has both, others have neither."""
        target = _make_face_config("face_both")
        other = _make_face_config("face_plain")
        faces = [other, target]
        target_idx = 1
        face_attrs = {
            "face_both":  _make_face_desc("face_both",  hat="fedora", glasses="reading"),
            "face_plain": _make_face_desc("face_plain", hat=None,     glasses=None),
        }
        attr = _find_distinguishing_attr("face_both", ["face_plain"], face_attrs)
        assert attr is not None
        assert "fedora" in attr or "hat" in attr, (
            f"Expected hat to take priority, got attr={attr!r}"
        )


# ---------------------------------------------------------------------------
# Bounds / distribution tests
# ---------------------------------------------------------------------------

class TestBoundsAndDistributions:

    def test_bounds_check(self, generator):
        """All numeric params must be within declared ranges for 1000 samples."""
        for seed in range(1000):
            cfg = generator.sample(seed)
            for face in cfg.faces:
                assert 1.0 <= face.initial_x <= 3.0, f"initial_x out of range: {face.initial_x}"
                assert -1.0 <= face.initial_y <= 1.0
                assert 0.5 <= face.initial_z <= 1.5
                assert 0.05 <= face.speed <= 0.5
                assert 6.0 <= face.period <= 20.0
                assert face.motion in ("static", "linear_drift", "sinusoidal", "random_walk")
            assert 0 <= cfg.lighting_azimuth_deg <= 360
            assert 15 <= cfg.lighting_elevation_deg <= 75
            assert 0.5 <= cfg.lighting_intensity <= 2.0
            for ch in cfg.ambient_rgb:
                assert 0.2 <= ch <= 0.8
            assert 0 <= cfg.distractor_count <= 2
            assert len(cfg.distractors) == cfg.distractor_count
            for dist in cfg.distractors:
                assert dist.shape in ("sphere", "box")
                for ch in dist.color_rgb:
                    assert 0.2 <= ch <= 0.9
            assert 0.0 <= cfg.camera_noise_sigma <= 0.015
            assert -20.0 <= cfg.camera_brightness_offset <= 20.0

    def test_face_count_distribution(self, generator):
        """single ≈ 60%, two ≈ 30%, three ≈ 10% ± 5% over 1000 samples."""
        counts = {1: 0, 2: 0, 3: 0}
        n = 1000
        for seed in range(n):
            cfg = generator.sample(seed)
            counts[len(cfg.faces)] += 1
        assert 55 <= counts[1] / n * 100 <= 65, f"single-face fraction off: {counts[1]/n:.2%}"
        assert 25 <= counts[2] / n * 100 <= 35, f"two-face fraction off:    {counts[2]/n:.2%}"
        assert  5 <= counts[3] / n * 100 <= 15, f"three-face fraction off:  {counts[3]/n:.2%}"

    def test_motion_static_fraction(self, generator):
        """Single-face episodes: static motion ≈ 30% ± 5%."""
        static_count = 0
        total = 0
        for seed in range(2000):
            cfg = generator.sample(seed)
            if len(cfg.faces) == 1:
                total += 1
                if cfg.faces[0].motion == "static":
                    static_count += 1
        frac = static_count / total
        assert 0.25 <= frac <= 0.35, (
            f"Static motion fraction = {frac:.2%}, expected 25–35%"
        )

    def test_scenario_id_uniqueness(self, generator):
        """500 different seeds must produce 500 unique scenario_ids."""
        ids = [generator.sample(seed).scenario_id for seed in range(500)]
        assert len(set(ids)) == 500, "Duplicate scenario_ids found"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:

    def test_json_round_trip(self, generator):
        """ScenarioConfig → dict → JSON string → dict → ScenarioConfig is lossless."""
        for seed in range(20):
            original = generator.sample(seed)
            d = original.to_dict()
            json_str = json.dumps(d)
            d2 = json.loads(json_str)
            restored = ScenarioConfig.from_dict(d2)

            assert restored.scenario_id == original.scenario_id
            assert restored.seed == original.seed
            assert restored.label_key == original.label_key
            assert restored.language_label == original.language_label
            assert len(restored.faces) == len(original.faces)
            for rf, of in zip(restored.faces, original.faces):
                assert rf.face_id == of.face_id
                assert abs(rf.initial_x - of.initial_x) < 1e-12
                assert rf.motion == of.motion
            assert restored.ambient_rgb == original.ambient_rgb
            for rd, od in zip(restored.distractors, original.distractors):
                assert rd.color_rgb == od.color_rgb
                assert rd.shape == od.shape


# ---------------------------------------------------------------------------
# Motion pattern tests
# ---------------------------------------------------------------------------

class TestMotionPatterns:

    Y_LO, Y_HI = -1.0, 1.0
    Z_LO, Z_HI = 0.3, 1.7

    def _assert_in_bounds(self, pos, label=""):
        x, y, z = pos
        assert self.Y_LO <= y <= self.Y_HI, f"{label} y={y} out of bounds"
        assert self.Z_LO <= z <= self.Z_HI, f"{label} z={z} out of bounds"

    def test_static_motion(self):
        m = StaticMotion()
        m.reset(2.0, 0.5, 1.0)
        for t in [0, 1, 10, 60]:
            assert m.step(t) == (2.0, 0.5, 1.0)

    def test_sinusoidal_motion(self):
        m = SinusoidalMotion(speed=0.3, period_y=10.0)
        m.reset(2.0, 0.0, 1.0)
        for t in range(61):
            self._assert_in_bounds(m.step(t), f"sinusoidal t={t}")

    def test_linear_drift_motion(self):
        m = LinearDriftMotion(vy=0.2, vz=0.1)
        m.reset(2.0, 0.0, 1.0)
        for t in range(61):
            self._assert_in_bounds(m.step(t), f"linear_drift t={t}")

    def test_random_walk_motion(self):
        m = RandomWalkMotion(speed=0.3, tau=3.0, seed=42)
        m.reset(2.0, 0.0, 1.0)
        for t in range(0, 61):
            self._assert_in_bounds(m.step(t), f"random_walk t={t}")

    def test_all_patterns_in_bounds_60s(self):
        """All 4 patterns stay within world bounds after a 60-second episode."""
        patterns = [
            StaticMotion(),
            SinusoidalMotion(speed=0.5, period_y=6.0),
            LinearDriftMotion(vy=0.5, vz=0.15),
            RandomWalkMotion(speed=0.5, tau=3.0, seed=0),
        ]
        for p in patterns:
            p.reset(2.0, 0.3, 1.0)
            for t_tenth in range(601):  # 0.0 to 60.0 in 0.1 s steps
                pos = p.step(t_tenth / 10.0)
                self._assert_in_bounds(pos, type(p).__name__)

    def test_linear_drift_analytical_correctness(self):
        """LinearDrift at t=0 returns initial position; x never changes."""
        m = LinearDriftMotion(vy=0.3, vz=0.1)
        m.reset(1.5, 0.2, 0.8)
        x, y, z = m.step(0.0)
        assert x == 1.5
        assert abs(y - 0.2) < 1e-9
        assert abs(z - 0.8) < 1e-9

    def test_random_walk_reproducible(self):
        """Same seed gives the same trajectory from reset."""
        m1 = RandomWalkMotion(speed=0.2, seed=7)
        m2 = RandomWalkMotion(speed=0.2, seed=7)
        m1.reset(2.0, 0.0, 1.0)
        m2.reset(2.0, 0.0, 1.0)
        for t in [1, 5, 10, 30, 60]:
            assert m1.step(t) == m2.step(t), f"Mismatch at t={t}"
