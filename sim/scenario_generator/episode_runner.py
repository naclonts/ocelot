"""
Episode runner for scenario generator.

Orchestrates a single simulation episode: spawns all entities for a
ScenarioConfig, ticks motion patterns at each time step, and tears down
on completion.

Usage (inside the sim container):
    from sim.scenario_generator.gazebo_bridge import GazeboBridge
    from sim.scenario_generator.episode_runner import EpisodeRunner

    bridge = GazeboBridge(world="scenario_world")
    runner = EpisodeRunner(bridge)
    runner.setup(config)
    for tick in range(n_ticks):
        positions = runner.step(tick * dt)
    runner.teardown()

The runner does NOT touch ROS — it only speaks gz.transport via GazeboBridge.
collect_data.py (Step 7) owns the ROS subscription loop and calls step() in
sync with the camera topic.
"""

import random

from sim.scenario_generator.motion import make_motion, RandomWalkMotion

# Constant added to config.seed when constructing the episode-runner RNG.
# Keeps this stream independent from the scenario-generator sampling stream.
_RNG_OFFSET = 0x5A5A5A5A


class EpisodeRunner:
    """Spawn, animate, and despawn all entities for one simulation episode.

    A single EpisodeRunner instance can be reused across multiple episodes
    by calling setup() / teardown() repeatedly.  setup() always calls
    teardown() internally first, so explicit teardown is only required at
    shutdown.
    """

    def __init__(self, bridge):
        """
        Args:
            bridge: GazeboBridge instance shared with the caller.
        """
        self._bridge = bridge
        # List of (entity_name, MotionPattern) pairs — rebuilt each episode.
        self._face_motions: list = []
        self._distractor_motions: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, config) -> None:
        """Spawn all entities for this episode and initialise motion patterns.

        Calls bridge.setup_episode(config), then constructs a MotionPattern
        for every face and every distractor, seeded deterministically from
        config.seed.

        Args:
            config: ScenarioConfig from sim.scenario_generator.scenario.
        """
        # Spawn all Gazebo entities (also tears down any previous episode).
        self._bridge.setup_episode(config)

        # Seeded RNG for deriving motion parameters not stored in FaceConfig
        # (linear_drift direction signs, random_walk internal seed).
        # The offset keeps this stream independent from scenario.py's stream.
        rng = random.Random(config.seed + _RNG_OFFSET)

        # One motion pattern per face.
        self._face_motions = []
        for i, face in enumerate(config.faces):
            pattern = make_motion(face.motion, face.speed, face.period, rng)
            pattern.reset(face.initial_x, face.initial_y, face.initial_z)
            self._face_motions.append((f"face_{i}", pattern))

        # Distractors always use random walk (no motion field in DistractorConfig).
        self._distractor_motions = []
        for i, dist in enumerate(config.distractors):
            dist_seed = rng.randint(0, 2**31)
            pattern = RandomWalkMotion(speed=dist.speed, seed=dist_seed)
            pattern.reset(dist.initial_x, dist.initial_y, dist.initial_z)
            self._distractor_motions.append((f"distractor_{i}", pattern))

    def step(self, t: float) -> dict:
        """Advance all motion patterns to time t and push poses to Gazebo.

        Args:
            t: Elapsed time in seconds since episode start.

        Returns:
            dict mapping entity name → (x, y, z) for every animated entity.
        """
        positions = {}
        for name, pattern in self._face_motions:
            x, y, z = pattern.step(t)
            self._bridge.set_pose(name, x, y, z)
            positions[name] = (x, y, z)
        for name, pattern in self._distractor_motions:
            x, y, z = pattern.step(t)
            self._bridge.set_pose(name, x, y, z)
            positions[name] = (x, y, z)
        return positions

    def teardown(self) -> None:
        """Despawn all episode-scoped entities.  Idempotent."""
        self._bridge.teardown_episode()
        self._face_motions = []
        self._distractor_motions = []
