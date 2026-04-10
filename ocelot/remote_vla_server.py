#!/usr/bin/env python3
"""HTTP server that runs the VLA model offboard and returns velocities."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from ocelot.vla_inference import VLAInferenceEngine


def _build_handler(
    engine: VLAInferenceEngine,
    default_command: str,
    max_vel: float,
    deadband: float,
):
    class RemoteVLAHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/health":
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "status": "ok",
                    "provider": engine.provider,
                    "checkpoint": str(engine.checkpoint),
                }
            )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/infer":
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self._send_json(
                    {"error": "empty request body"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            body = self.rfile.read(content_length)
            image = np.frombuffer(body, dtype=np.uint8)

            import cv2

            bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if bgr is None:
                self._send_json(
                    {"error": "failed to decode jpeg"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            qs = parse_qs(parsed.query)
            command = qs.get("command", [default_command])[0]

            start = time.perf_counter()
            result = engine.predict_bgr(bgr, command)
            total_latency_ms = (time.perf_counter() - start) * 1000.0

            pan_vel = float(np.clip(result["pan_vel"], -max_vel, max_vel))
            tilt_vel = float(np.clip(result["tilt_vel"], -max_vel, max_vel))

            if abs(pan_vel) < deadband:
                pan_vel = 0.0
            if abs(tilt_vel) < deadband:
                tilt_vel = 0.0

            self._send_json(
                {
                    "command": result["command"],
                    "pan_vel": pan_vel,
                    "tilt_vel": tilt_vel,
                    "inference_latency_ms": result["inference_latency_ms"],
                    "total_latency_ms": total_latency_ms,
                }
            )

        def log_message(self, fmt: str, *args) -> None:
            return

    return RemoteVLAHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-cache")
    parser.add_argument("--command", default="track the face")
    parser.add_argument("--max-vel", type=float, default=0.3)
    parser.add_argument("--deadband", type=float, default=0.03)
    args = parser.parse_args()

    engine = VLAInferenceEngine(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
    )
    actual_command = engine.resolve_command(args.command)
    handler = _build_handler(
        engine=engine,
        default_command=actual_command,
        max_vel=args.max_vel,
        deadband=args.deadband,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Remote VLA server listening on http://{args.host}:{args.port} "
        f"(provider={engine.provider}, command={actual_command!r})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
