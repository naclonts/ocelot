#!/usr/bin/env python3
"""Camera node — captures frames from Pi Camera V2 and publishes to /camera/image_raw.

Camera capture is delegated to capture_worker.py running under Python 3.11,
working around the libcamera Python bindings being compiled for Python 3.11
while ROS Jazzy uses Python 3.12.
"""

import os
import struct
import subprocess
import sys
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

_WORKER = os.path.join(os.path.dirname(__file__), 'capture_worker.py')
_PYTHON311 = '/usr/bin/python3.11'


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('fps', 15)
        self.declare_parameter('resolution', [640, 480])

        fps = self.get_parameter('fps').value
        resolution = self.get_parameter('resolution').value
        self._width, self._height = resolution[0], resolution[1]
        self._frame_bytes = self._width * self._height * 3

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, '/camera/image_raw', 10)

        interpreter = _PYTHON311 if os.path.exists(_PYTHON311) else sys.executable
        self._proc = subprocess.Popen(
            [interpreter, _WORKER, str(self._width), str(self._height)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._latest_frame = None
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_frames, daemon=True)
        self._reader.start()

        self.create_timer(1.0 / fps, self._publish)
        self.get_logger().info(
            f'Camera node started ({self._width}x{self._height} @ {fps} fps, '
            f'worker interpreter: {interpreter})'
        )

    def _read_frames(self):
        while self._proc.poll() is None:
            try:
                header = self._proc.stdout.read(4)
                if len(header) < 4:
                    break
                size = struct.unpack('>I', header)[0]
                data = b''
                while len(data) < size:
                    chunk = self._proc.stdout.read(size - len(data))
                    if not chunk:
                        break
                    data += chunk
                if len(data) == size:
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(
                        (self._height, self._width, 3)
                    )
                    with self._lock:
                        self._latest_frame = frame
            except Exception as e:
                self.get_logger().error(f'Frame read error: {e}')
                break
        stderr = self._proc.stderr.read().decode(errors='replace')
        if stderr:
            self.get_logger().error(f'Capture worker stderr:\n{stderr}')

    def _publish(self):
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return
        msg = self._bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self._pub.publish(msg)

    def destroy_node(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
