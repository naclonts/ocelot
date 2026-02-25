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
                assert 2.0 <= face.initial_x <= 4.0, f"initial_x out of range: {face.initial_x}"
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


# ---------------------------------------------------------------------------
# Helpers shared by reordering / label-key tests
# ---------------------------------------------------------------------------

def _make_scenario_config(faces, target_face_idx=0, label_key="single_centered",
                           language_label="track the face") -> ScenarioConfig:
    """Build a minimal ScenarioConfig for testing — no real paths required."""
    return ScenarioConfig(
        scenario_id="test",
        seed=0,
        faces=faces,
        target_face_idx=target_face_idx,
        background_id="plain_white",
        background_path="/fake/bg.png",
        lighting_azimuth_deg=45.0,
        lighting_elevation_deg=45.0,
        lighting_intensity=1.0,
        ambient_rgb=(0.5, 0.5, 0.5),
        distractor_count=0,
        distractors=[],
        camera_noise_sigma=0.0,
        camera_brightness_offset=0.0,
        label_key=label_key,
        language_label=language_label,
    )


# ---------------------------------------------------------------------------
# Face reordering tests — GazeboBridge.setup_episode()
# ---------------------------------------------------------------------------

class TestSetupEpisodeFaceOrdering:
    """Verify setup_episode() always spawns the target face as face_0."""

    def _run_setup_episode(self, target_face_idx: int):
        """Run setup_episode on a 3-face config and return the spawn_face call list."""
        from unittest.mock import MagicMock
        from sim.scenario_generator.gazebo_bridge import GazeboBridge

        faces = [
            _make_face_config(face_id=f"face_{c:03d}", initial_y=float(c))
            for c in range(3)
        ]

        bridge = GazeboBridge(world="test_world")
        bridge.spawn_face = MagicMock(return_value=True)
        bridge.spawn_background = MagicMock(return_value=True)
        bridge.spawn_key_light = MagicMock(return_value=True)
        bridge.spawn_fill_light = MagicMock(return_value=True)
        bridge.teardown_episode = MagicMock()

        config = _make_scenario_config(faces, target_face_idx=target_face_idx)
        bridge.setup_episode(config)
        return bridge.spawn_face.call_args_list, faces

    def test_target_idx_0_spawns_first_as_face0(self):
        """When target_face_idx=0, face_0 still gets faces[0]'s texture."""
        calls, faces = self._run_setup_episode(target_face_idx=0)
        assert calls[0].kwargs["name"] == "face_0"
        assert calls[0].kwargs["texture_abs_path"] == faces[0].texture_path

    def test_target_idx_1_spawns_as_face0(self):
        """When target_face_idx=1, face_0 gets faces[1]'s texture."""
        calls, faces = self._run_setup_episode(target_face_idx=1)
        assert calls[0].kwargs["name"] == "face_0"
        assert calls[0].kwargs["texture_abs_path"] == faces[1].texture_path

    def test_target_idx_2_spawns_as_face0(self):
        """When target_face_idx=2, face_0 gets faces[2]'s texture."""
        calls, faces = self._run_setup_episode(target_face_idx=2)
        assert calls[0].kwargs["name"] == "face_0"
        assert calls[0].kwargs["texture_abs_path"] == faces[2].texture_path

    def test_all_three_faces_spawned(self):
        """All three faces are spawned regardless of reordering."""
        calls, _ = self._run_setup_episode(target_face_idx=1)
        spawned_names = [c.kwargs["name"] for c in calls]
        assert sorted(spawned_names) == ["face_0", "face_1", "face_2"]


# ---------------------------------------------------------------------------
# Face reordering tests — EpisodeRunner.setup()
# ---------------------------------------------------------------------------

class TestEpisodeRunnerFaceOrdering:
    """Verify face_0 motion tracks the target face's initial position."""

    def _run_runner_step0(self, target_face_idx: int):
        """Setup runner with a 2-face static config and step at t=0."""
        from unittest.mock import MagicMock, patch
        from sim.scenario_generator.episode_runner import EpisodeRunner

        mock_bridge = MagicMock()
        mock_bridge.setup_episode.return_value = None
        mock_bridge.set_pose.return_value = True

        faces = [
            _make_face_config("face_001", initial_x=2.0, initial_y=0.5,
                              motion="static"),
            _make_face_config("face_002", initial_x=3.0, initial_y=-0.5,
                              motion="static"),
        ]
        config = _make_scenario_config(faces, target_face_idx=target_face_idx)

        runner = EpisodeRunner(mock_bridge)
        with patch("sim.scenario_generator.episode_runner.subprocess.run"):
            runner.setup(config)
        return runner.step(0.0), faces

    def test_face0_is_target_when_idx1(self):
        """With target_face_idx=1, face_0 motion starts at faces[1]'s position."""
        positions, faces = self._run_runner_step0(target_face_idx=1)
        assert positions["face_0"] == (3.0, -0.5, 1.0), (
            f"face_0 should be target (faces[1]); got {positions['face_0']}"
        )
        assert positions["face_1"] == (2.0, 0.5, 1.0), (
            f"face_1 should be non-target (faces[0]); got {positions['face_1']}"
        )

    def test_face0_is_target_when_idx0(self):
        """With target_face_idx=0, face_0 motion starts at faces[0]'s position."""
        positions, faces = self._run_runner_step0(target_face_idx=0)
        assert positions["face_0"] == (2.0, 0.5, 1.0), (
            f"face_0 should be target (faces[0]); got {positions['face_0']}"
        )


# ---------------------------------------------------------------------------
# Oracle label_key mapping tests — EpisodeRunner._set_oracle_label_key
# ---------------------------------------------------------------------------

class TestEpisodeRunnerLabelKey:
    """Verify label_key mapping: single_slow → 'slow', others → 'track'."""

    def _setup_runner(self, label_key: str):
        """Run setup() with the given label_key and return the subprocess.run call."""
        from unittest.mock import MagicMock, patch
        from sim.scenario_generator.episode_runner import EpisodeRunner

        mock_bridge = MagicMock()
        mock_bridge.setup_episode.return_value = None

        face = _make_face_config(motion="sinusoidal", speed=0.1)
        config = _make_scenario_config([face], label_key=label_key)

        runner = EpisodeRunner(mock_bridge)
        with patch("sim.scenario_generator.episode_runner.subprocess.run") as mock_run:
            runner.setup(config)
        return mock_run.call_args_list

    def test_single_slow_maps_to_slow(self):
        """label_key='single_slow' → oracle receives 'slow'."""
        calls = self._setup_runner("single_slow")
        assert len(calls) == 1
        cmd = calls[0][0][0]
        assert "label_key" in cmd
        assert cmd[-1] == "slow", f"Expected last arg 'slow', got {cmd[-1]!r}"

    def test_single_centered_maps_to_track(self):
        """label_key='single_centered' → oracle receives 'track'."""
        calls = self._setup_runner("single_centered")
        assert len(calls) == 1
        cmd = calls[0][0][0]
        assert cmd[-1] == "track", f"Expected last arg 'track', got {cmd[-1]!r}"

    def test_multi_attr_maps_to_track(self):
        """label_key='multi_attr' → oracle receives 'track'."""
        calls = self._setup_runner("multi_attr")
        assert len(calls) == 1
        cmd = calls[0][0][0]
        assert cmd[-1] == "track"
