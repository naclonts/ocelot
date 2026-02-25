"""Gazebo bridge for scenario generator — spawns, moves, and despawns episode entities.

Uses gz.transport13 Python bindings (single persistent Node at module level) —
same pattern as sim/move_face.py.  A persistent Node avoids repeated ZMQ teardown
which causes 'Host unreachable' log spam in Gazebo when using ephemeral nodes.

All gz service calls are synchronous with a 2-second timeout.  Operations that
time out log a warning and return False; they never raise exceptions, so the
episode runner can continue and attempt cleanup.

Usage (inside the sim container, with scenario_world or tracker_world running):

    from sim.scenario_generator.gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(world="scenario_world")
    bridge.spawn_face("face_0", (2.0, 0.0, 0.8), "/abs/path/face.png")
    bridge.set_pose("face_0", 2.0, 0.3, 0.8)
    bridge.despawn("face_0")

Requires: python3-gz-transport13  python3-gz-msgs10
  (both installed in deploy/docker/Dockerfile.sim — rebuild if missing)
"""

import logging
import math

log = logging.getLogger(__name__)

try:
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.pose_pb2 import Pose

    _GZ_AVAILABLE = True
except ImportError as _exc:
    log.warning(
        "gz-transport Python bindings not available: %s. "
        "GazeboBridge will operate as a no-op stub (all calls return True). "
        "Fix: install python3-gz-transport13 and python3-gz-msgs10 in the sim container.",
        _exc,
    )
    _GZ_AVAILABLE = False

# Single persistent node for the process lifetime.
# Avoids ZMQ teardown / 'Host unreachable' log spam from ephemeral nodes.
_node = Node() if _GZ_AVAILABLE else None

# Timeout for gz service calls in milliseconds.
_TIMEOUT_MS = 2000

# Entity type integers from gz.msgs10.entity_pb2.Entity.Type
# NONE=0, LIGHT=1, MODEL=2, LINK=3, ...
_ENTITY_LIGHT = 1
_ENTITY_MODEL = 2

# ---------------------------------------------------------------------------
# SDF templates
# ---------------------------------------------------------------------------

_POSE_PUBLISHER_PLUGIN = """\

    <plugin filename="gz-sim-pose-publisher-system"
            name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose>
      <publish_nested_model_pose>false</publish_nested_model_pose>
      <publish_model_pose>true</publish_model_pose>
      <update_rate>10</update_rate>
    </plugin>"""

_FACE_SDF = """\
<sdf version="1.10">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>{pose_publisher_plugin}
    <link name="link">
      <visual name="face_visual">
        <transparency>1</transparency>
        <geometry>
          <box><size>0.002 0.5 0.5</size></box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <pbr><metal>
            <albedo_map>{texture_abs_path}</albedo_map>
            <metalness>0.0</metalness>
            <roughness>1.0</roughness>
          </metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

_BACKGROUND_SDF = """\
<sdf version="1.10">
  <model name="background_wall">
    <static>true</static>
    <pose>50 0 20 0 0 0</pose>
    <link name="link">
      <visual name="wall_visual">
        <geometry>
          <box><size>0.002 120.0 40.0</size></box>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1</ambient>
          <diffuse>1.0 1.0 1.0 1</diffuse>
          <pbr><metal>
            <albedo_map>{texture_abs_path}</albedo_map>
            <metalness>0.0</metalness>
            <roughness>1.0</roughness>
          </metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

_LIGHT_SDF = """\
<sdf version="1.10">
  <light type="point" name="{name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <diffuse>1 1 1 1</diffuse>
    <specular>0.1 0.1 0.1 1</specular>
    <intensity>{intensity}</intensity>
    <attenuation>
      <range>20</range>
      <constant>0.1</constant>
      <linear>0.01</linear>
      <quadratic>0.001</quadratic>
    </attenuation>
    <cast_shadows>false</cast_shadows>
  </light>
</sdf>"""

_FILL_LIGHT_SDF = """\
<sdf version="1.10">
  <light type="point" name="{name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <diffuse>{r} {g} {b} 1</diffuse>
    <specular>0.05 0.05 0.05 1</specular>
    <intensity>0.6</intensity>
    <attenuation>
      <range>20</range>
      <constant>0.2</constant>
      <linear>0.02</linear>
      <quadratic>0.002</quadratic>
    </attenuation>
    <cast_shadows>false</cast_shadows>
  </light>
</sdf>"""

_DISTRACTOR_SDF = """\
<sdf version="1.10">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <visual name="vis">
        <geometry>
          {geom}
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

_GEOM_SPHERE = "<sphere><radius>0.1</radius></sphere>"
_GEOM_BOX = "<box><size>0.2 0.2 0.2</size></box>"

# All entity names that are episode-scoped (used by teardown_episode).
_EPISODE_ENTITY_NAMES = [
    "face_0", "face_1", "face_2",
    "distractor_0", "distractor_1",
    "episode_light_key", "episode_light_fill",
    "background_wall",
]


# ---------------------------------------------------------------------------
# GazeboBridge
# ---------------------------------------------------------------------------

class GazeboBridge:
    """Spawn, move, and despawn Gazebo entities for simulation episodes.

    A single instance is typically shared across multiple episodes.  Call
    setup_episode() at the start of each episode and teardown_episode() at
    the end (setup_episode calls teardown_episode internally, so explicit
    teardown is only needed at shutdown).
    """

    def __init__(self, world: str = "scenario_world"):
        self._world = world
        self._create_srv   = f"/world/{world}/create"
        self._remove_srv   = f"/world/{world}/remove"
        self._set_pose_srv = f"/world/{world}/set_pose"
        # name → entity type int; used by despawn() to choose the right type.
        self._spawned: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _spawn_sdf(self, sdf: str, entity_type: int, name: str) -> bool:
        """Submit an EntityFactory request with the given SDF string."""
        if not _GZ_AVAILABLE:
            log.debug("gz not available — skipping spawn of '%s'", name)
            return True  # no-op stub always reports success

        ef = EntityFactory()
        ef.sdf = sdf
        ef.allow_renaming = False
        result, _rep = _node.request(
            self._create_srv, ef, EntityFactory, Boolean, _TIMEOUT_MS
        )
        if result:
            self._spawned[name] = entity_type
            log.debug("spawned '%s'", name)
        else:
            log.warning(
                "spawn '%s': EntityFactory request timed out or service "
                "returned False (is the sim running?)",
                name,
            )
        return result

    def _remove_entity(self, name: str, entity_type: int) -> bool:
        """Submit a remove request for a named entity."""
        if not _GZ_AVAILABLE:
            return True

        ent = Entity()
        ent.name = name
        ent.type = entity_type
        result, _rep = _node.request(
            self._remove_srv, ent, Entity, Boolean, _TIMEOUT_MS
        )
        if result:
            log.debug("despawned '%s'", name)
        else:
            log.warning(
                "despawn '%s': remove request timed out or returned False",
                name,
            )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn_face(self, name: str, pos: tuple, texture_abs_path: str) -> bool:
        """Spawn a face billboard with the given texture at pos=(x, y, z).

        Only face_0 receives the PosePublisher plugin — the oracle node
        subscribes to /model/face_0/pose to track the primary target.
        """
        plugin = _POSE_PUBLISHER_PLUGIN if name == "face_0" else ""
        sdf = _FACE_SDF.format(
            name=name,
            x=pos[0], y=pos[1], z=pos[2],
            texture_abs_path=texture_abs_path,
            pose_publisher_plugin=plugin,
        )
        return self._spawn_sdf(sdf, _ENTITY_MODEL, name)

    def despawn(self, name: str) -> bool:
        """Remove a named entity using its tracked type.  Idempotent."""
        if name not in self._spawned:
            return True  # nothing to do
        entity_type = self._spawned.pop(name)
        return self._remove_entity(name, entity_type)

    def set_pose(self, name: str, x: float, y: float, z: float) -> bool:
        """Set the absolute world position of a model.  Rotation stays at identity."""
        if not _GZ_AVAILABLE:
            return True

        req = Pose()
        req.name = name
        req.position.x = x
        req.position.y = y
        req.position.z = z
        result, _rep = _node.request(
            self._set_pose_srv, req, Pose, Boolean, _TIMEOUT_MS
        )
        if not result:
            log.warning(
                "set_pose '%s': request timed out or returned False", name
            )
        return result

    def spawn_background(self, background_abs_path: str) -> bool:
        """Despawn any existing background_wall, then spawn one with the given texture.

        Geometry: 0.002 × 120.0 × 40.0 m box at x=50, z=20.  At 50 m with a
        60° HFOV the wall subtends the full frame, reading as a distant backdrop
        rather than a visible foreground object.
        """
        if "background_wall" in self._spawned:
            self.despawn("background_wall")
        sdf = _BACKGROUND_SDF.format(texture_abs_path=background_abs_path)
        return self._spawn_sdf(sdf, _ENTITY_MODEL, "background_wall")

    def spawn_key_light(
        self, azimuth_deg: float, elevation_deg: float, intensity: float
    ) -> bool:
        """Spawn a point light at a hemisphere position above the scene.

        Spherical → Cartesian:
            r = 6.0 m
            x = r * cos(el) * cos(az) + 1.5   (offset toward scene centre)
            y = r * cos(el) * sin(az)
            z = r * sin(el)

        Diffuse = intensity × white; range 20 m; no shadow casting.
        Despawns any previous key light first.
        """
        name = "episode_light_key"
        if name in self._spawned:
            self.despawn(name)

        r = 6.0
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        x = r * math.cos(el) * math.cos(az) + 1.5
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)

        sdf = _LIGHT_SDF.format(
            name=name,
            x=x, y=y, z=z,
            intensity=intensity,
        )
        return self._spawn_sdf(sdf, _ENTITY_LIGHT, name)

    def spawn_fill_light(
        self, azimuth_deg: float, elevation_deg: float, ambient_rgb: tuple
    ) -> bool:
        """Spawn a soft fill light on the hemisphere opposite the key light.

        Position: same elevation as the key light, azimuth rotated 180°.
        Diffuse = ambient_rgb — tints the scene fill with the sampled ambient colour.
        Intensity is fixed at 0.6, always softer than the key light (0.5–2.0 range).
        Despawns any previous fill light first.
        """
        name = "episode_light_fill"
        if name in self._spawned:
            self.despawn(name)

        r = 6.0
        az = math.radians((azimuth_deg + 180) % 360)
        el = math.radians(elevation_deg)
        x = r * math.cos(el) * math.cos(az) + 1.5
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)

        cr, cg, cb = ambient_rgb
        sdf = _FILL_LIGHT_SDF.format(
            name=name,
            x=x, y=y, z=z,
            r=cr, g=cg, b=cb,
        )
        return self._spawn_sdf(sdf, _ENTITY_LIGHT, name)

    def spawn_distractor(
        self, name: str, pos: tuple, shape: str, color_rgb: tuple
    ) -> bool:
        """Spawn a colored sphere or box distractor primitive.

        No PosePublisher — the oracle must not identify it as a tracking target.
        shape:     "sphere" or "box"
        color_rgb: (r, g, b), each channel in [0.2, 0.9]
        """
        geom = _GEOM_SPHERE if shape == "sphere" else _GEOM_BOX
        r, g, b = color_rgb
        sdf = _DISTRACTOR_SDF.format(
            name=name,
            x=pos[0], y=pos[1], z=pos[2],
            geom=geom,
            r=r, g=g, b=b,
        )
        return self._spawn_sdf(sdf, _ENTITY_MODEL, name)

    def setup_episode(self, config) -> None:
        """Tear down any previous episode, then spawn all entities for config.

        config: ScenarioConfig from sim.scenario_generator.scenario.
        """
        self.teardown_episode()

        self.spawn_background(config.background_path)
        self.spawn_key_light(
            config.lighting_azimuth_deg,
            config.lighting_elevation_deg,
            config.lighting_intensity,
        )
        self.spawn_fill_light(
            config.lighting_azimuth_deg,
            config.lighting_elevation_deg,
            config.ambient_rgb,
        )
        for i, face_cfg in enumerate(config.faces):
            self.spawn_face(
                name=f"face_{i}",
                pos=(face_cfg.initial_x, face_cfg.initial_y, face_cfg.initial_z),
                texture_abs_path=face_cfg.texture_path,
            )
        for i, dist_cfg in enumerate(config.distractors):
            self.spawn_distractor(
                name=f"distractor_{i}",
                pos=(dist_cfg.initial_x, dist_cfg.initial_y, dist_cfg.initial_z),
                shape=dist_cfg.shape,
                color_rgb=dist_cfg.color_rgb,
            )

    def teardown_episode(self) -> None:
        """Despawn all episode-scoped entities.  Idempotent."""
        for name in _EPISODE_ENTITY_NAMES:
            if name in self._spawned:
                self.despawn(name)
