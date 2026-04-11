#!/usr/bin/env python3
"""sim/preview_episode.py — record one episode to mp4 with live annotation overlay.

Runs a single scenario episode and saves /camera/image_raw frames with an
in-process overlay (crosshair, cmd_vel, projected face-position dot) to an mp4
file. The overlay is rendered here rather than reusing /camera/image_annotated
so preview video is not distorted by asynchronous visualizer timing.

Default: seed=280  (sinusoidal, outdoor_forest, n_faces=1, no perturbation)

Usage (inside sim container with sim_launch.py use_oracle:=true world:=scenario_world running):

    python3 /ws/src/ocelot/sim/preview_episode.py
    python3 /ws/src/ocelot/sim/preview_episode.py --seed 280 --out /ws/src/ocelot/preview.mp4
    python3 /ws/src/ocelot/sim/preview_episode.py --seed 33
"""

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Twist
from rcl_interfaces.msg import Parameter as RclParameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.scenario_generator.gazebo_bridge import GazeboBridge
from sim.scenario_generator.episode_runner import EpisodeRunner
from sim.scenario_generator.scenario import ScenarioGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CAMERA_HZ          = 10
EPISODE_SECS       = 10.0
WARMUP_SECS        = 4.0
FRAMES_PER_EPISODE = int(EPISODE_SECS * CAMERA_HZ)

# Camera intrinsics from URDF (640×480, HFOV=60°=1.0472 rad)
_IMG_W, _IMG_H = 640, 480
_CX, _CY       = _IMG_W // 2, _IMG_H // 2
_FX             = (_IMG_W / 2) / math.tan(1.0472 / 2)   # ≈ 554.3 px
_FY             = _FX                                     # square pixels

# FK constants — mirror oracle_node.py
_PAN_Z  = 0.03
_TILT_Z = 0.03
_CAM_X  = 0.02
_CAM_Z  = 0.01

# Annotation colours (BGR)
_CYAN   = (255, 200, 0)
_RED    = (0, 60, 220)
_GREY   = (180, 180, 180)
_YELLOW = (80, 230, 255)
_WHITE  = (255, 255, 255)

_ORACLE_LABEL_MAP = {
    "track":       "track",
    "multi_left":  "multi_left",
    "multi_right": "multi_right",
    "multi_attr":  "track",
    "centered":    "track",
    "no_face":     "track",
}


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _project_face(face_pos: np.ndarray, pan: float, tilt: float) -> "tuple[int, int] | None":
    """Project Gazebo face position to (u, v) pixel coords via FK.

    Returns None if the face is behind the camera.
    Camera frame convention (from oracle_node FK):
        +X = forward, +Y = left, +Z = up.
    Image convention: +u = right, +v = down → u = cx - fx*(dy/dx), v = cy - fy*(dz/dx).
    """
    R_cam = _rz(pan) @ _ry(tilt)
    P_tilt = np.array([0.0, 0.0, _PAN_Z + _TILT_Z])
    P_cam  = P_tilt + R_cam @ np.array([_CAM_X, 0.0, _CAM_Z])
    d_cam  = R_cam.T @ (face_pos - P_cam)

    if d_cam[0] <= 0.01:
        return None

    u = int(round(_CX - _FX * d_cam[1] / d_cam[0]))
    v = int(round(_CY - _FY * d_cam[2] / d_cam[0]))
    return u, v


def _annotate_frame(
    frame: np.ndarray,
    face_pos: "np.ndarray | None",
    pan: float,
    tilt: float,
    cmd: Twist,
    episode_cmd: str,
) -> np.ndarray:
    """Overlay crosshair, cmd_vel text, and projected face pose onto frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 30, _WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    arm = 14
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), _GREY, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), _GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"pan  {cmd.angular.z:+.2f}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, f"tilt {cmd.angular.y:+.2f}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)
    if episode_cmd:
        cv2.putText(frame, episode_cmd,
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

    if face_pos is None:
        return frame
    uv = _project_face(face_pos, pan, tilt)
    if uv is None:
        return frame
    u, v = uv
    # error vector: center → projected face
    cv2.arrowedLine(frame, (cx, cy), (u, v), _RED, 2, cv2.LINE_AA, tipLength=0.15)
    # face projection dot
    cv2.circle(frame, (u, v), 8, _CYAN, -1, cv2.LINE_AA)
    cv2.circle(frame, (u, v), 8, _WHITE, 1, cv2.LINE_AA)

    return frame


class PreviewNode(Node):
    def __init__(self):
        super().__init__("preview_episode")
        self._bridge = CvBridge()
        self._lock   = threading.Lock()

        self._latest_raw: np.ndarray | None       = None
        self._face_pos: np.ndarray | None          = None
        self._pan: float  = 0.0
        self._tilt: float = 0.0
        self._latest_cmd = Twist()
        self._latest_episode_cmd = ""

        self._frame_event      = threading.Event()
        self._joints_received  = threading.Event()

        self._cmd_pub = self.create_publisher(String, "/episode/cmd", 1)
        self._oracle_client = self.create_client(
            SetParameters, "/oracle_node/set_parameters"
        )
        self._oracle_ready = False

        self.create_subscription(Image,      "/camera/image_raw", self._on_image, 10)
        self.create_subscription(JointState, "/joint_states",     self._on_joints, 10)
        self.create_subscription(Pose,       "/model/face_0/pose", self._on_face, 10)
        self.create_subscription(Twist,      "/cmd_vel",           self._on_cmd_vel, 10)

    def _on_image(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._latest_raw = frame
        self._frame_event.set()

    def _on_joints(self, msg: JointState) -> None:
        with self._lock:
            for i, name in enumerate(msg.name):
                if name == "pan_joint":
                    self._pan = float(msg.position[i])
                elif name == "tilt_joint":
                    self._tilt = float(msg.position[i])
        self._joints_received.set()

    def _on_face(self, msg: Pose) -> None:
        with self._lock:
            self._face_pos = np.array(
                [msg.position.x, msg.position.y, msg.position.z]
            )

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._lock:
            self._latest_cmd = msg

    def wait_ready(self, timeout: float = 90.0) -> bool:
        return self._joints_received.wait(timeout=timeout)

    def wait_for_frame(self, timeout: float = 5.0) -> bool:
        self._frame_event.clear()
        return self._frame_event.wait(timeout=timeout)

    def get_frame_with_face(self) -> "np.ndarray | None":
        """Return a raw frame with a local overlay based on latest control/state."""
        with self._lock:
            base = None if self._latest_raw is None else self._latest_raw.copy()
            pos = self._face_pos.copy() if self._face_pos is not None else None
            pan, tilt = self._pan, self._tilt
            cmd = self._latest_cmd
            episode_cmd = self._latest_episode_cmd
        if base is None:
            return None
        return _annotate_frame(base, pos, pan, tilt, cmd, episode_cmd)

    def publish_cmd(self, text: str) -> None:
        msg = String()
        msg.data = text
        with self._lock:
            self._latest_episode_cmd = text
        self._cmd_pub.publish(msg)

    def configure_oracle(self, label_key: str, num_faces: int,
                         timeout: float = 10.0) -> None:
        oracle_label = _ORACLE_LABEL_MAP.get(label_key, "track")
        if not self._oracle_ready:
            if not self._oracle_client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(
                    f"/oracle_node/set_parameters not available after {timeout:.0f}s"
                )
            self._oracle_ready = True

        lk = RclParameter()
        lk.name  = "label_key"
        lk.value = ParameterValue(
            type=ParameterType.PARAMETER_STRING, string_value=oracle_label
        )
        nf = RclParameter()
        nf.name  = "num_faces"
        nf.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER, integer_value=num_faces
        )
        req = SetParameters.Request()
        req.parameters = [lk, nf]

        future = self._oracle_client.call_async(req)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError("configure_oracle: timed out")
            time.sleep(0.01)
        for r in future.result().results:
            if not r.successful:
                raise RuntimeError(f"configure_oracle: {r.reason}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Record one oracle-driven episode to mp4 (annotated frames)."
    )
    p.add_argument("--seed", type=int, default=280,
                   help="Scenario seed. Default 280 = sinusoidal / outdoor_forest / 1 face.")
    p.add_argument("--out", default="/ws/src/ocelot/preview.mp4",
                   help="Output mp4 path. Default: /ws/src/ocelot/preview.mp4")
    p.add_argument(
        "--skip-oracle-config",
        action="store_true",
        help=(
            "Skip /oracle_node parameter configuration. "
            "Use this when sim_launch.py is running with use_vla:=true."
        ),
    )
    p.add_argument(
        "--faces-dir",
        default=str(Path(__file__).resolve().parent / "scenario_generator"),
    )
    p.add_argument(
        "--bg-dir",
        default=str(Path(__file__).resolve().parent / "assets" / "backgrounds"),
    )
    args = p.parse_args()

    rclpy.init()
    node = PreviewNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    log.info("Waiting for sim stack (/joint_states) …")
    if not node.wait_ready(timeout=90.0):
        log.error("Timed out — is sim_launch.py running?")
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)
    log.info("Sim ready.")

    generator = ScenarioGenerator(
        faces_dir=Path(args.faces_dir),
        backgrounds_dir=Path(args.bg_dir),
    )
    bridge = GazeboBridge(world="scenario_world")
    runner = EpisodeRunner(bridge)

    config = generator.sample(args.seed)
    log.info(
        "seed=%d  n_faces=%d  bg=%s  motion=%s  label=%s",
        args.seed, len(config.faces), config.background_id,
        config.faces[0].motion if config.faces else "?", config.label_key,
    )

    if args.skip_oracle_config:
        log.info("Skipping oracle parameter config (--skip-oracle-config).")
    else:
        node.configure_oracle(config.label_key, len(config.faces))
    runner.setup(config)
    node.publish_cmd(config.language_label)

    # ── Warmup ────────────────────────────────────────────────────────────────
    log.info("Warmup %.0fs …", WARMUP_SECS)
    warmup_start = time.monotonic()
    warmup_frame = 0
    while (time.monotonic() - warmup_start) < WARMUP_SECS:
        if not node.wait_for_frame(timeout=5.0):
            log.warning("Frame timeout during warmup")
            break
        runner.step(warmup_frame / CAMERA_HZ)
        warmup_frame += 1

    # ── Record ────────────────────────────────────────────────────────────────
    log.info("Recording %d frames …", FRAMES_PER_EPISODE)
    frames: list[np.ndarray] = []
    for frame_idx in range(FRAMES_PER_EPISODE):
        if not node.wait_for_frame(timeout=5.0):
            log.warning("Frame %d: timeout", frame_idx)
            break
        f = node.get_frame_with_face()
        runner.step(WARMUP_SECS + frame_idx / CAMERA_HZ)
        if f is not None:
            frames.append(f)

    runner.teardown()
    log.info("Teardown done. Captured %d frames.", len(frames))

    if not frames:
        log.error("No frames — mp4 not saved.")
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    # ── Save mp4 ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(CAMERA_HZ),
        (w, h),
    )
    for f in frames:
        writer.write(f)
    writer.release()

    log.info("Saved → %s  (%d frames @ %d Hz)", out_path, len(frames), CAMERA_HZ)
    log.info("On host: xdg-open %s", out_path.name)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
