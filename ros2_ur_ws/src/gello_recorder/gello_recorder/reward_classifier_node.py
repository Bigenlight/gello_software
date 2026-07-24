#!/usr/bin/env python3
"""Live two-camera HIL-SERL reward-classifier inference.

The node is deliberately separate from ``policy_run_gui``: JAX initialization
and inference must never block the Qt event loop or the policy control service
clients.  It subscribes to the same compressed RealSense topics as the policy,
reproduces the cube-in-cup classifier input contract, and publishes read-only
JSON status for visualization.

Status topic (std_msgs/String), ``/reward_classifier/status``::

    {
      "ready": true, "probability": 0.91, "success": true,
      "threshold": 0.5, "cam_skew_ms": 4.2, "inference_ms": 18.1,
      "message": "ok"
    }

No robot command, reward, reset, or termination signal is published.
"""

import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from gello_recorder.reward_classifier_runtime import (
    IMAGE_KEYS,
    make_observation,
    sigmoid_probability,
    status_json,
    validate_threshold,
)

STATUS_TOPIC = "/reward_classifier/status"


def ros_stamp_ns(msg: CompressedImage) -> int:
    """Return a ROS header timestamp in nanoseconds."""
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(
        msg.header.stamp.nanosec
    )


class RewardClassifierNode(Node):
    def __init__(self) -> None:
        super().__init__("reward_classifier_node")
        self.declare_parameter(
            "cam1_topic", "/cam1/cam1/color/image_raw/compressed"
        )
        self.declare_parameter(
            "cam2_topic", "/cam2/cam2/color/image_raw/compressed"
        )
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("hil_serl_root", "")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("inference_hz", 10.0)
        self.declare_parameter("max_camera_age_s", 0.5)
        self.declare_parameter("max_camera_skew_s", 0.10)

        self._threshold = validate_threshold(
            self.get_parameter("threshold").value
        )
        inference_hz = float(self.get_parameter("inference_hz").value)
        self._max_age_s = float(self.get_parameter("max_camera_age_s").value)
        self._max_skew_s = float(self.get_parameter("max_camera_skew_s").value)

        checkpoint = str(self.get_parameter("checkpoint_path").value).strip()
        checkpoint = checkpoint or os.environ.get(
            "REWARD_CLASSIFIER_CHECKPOINT", ""
        )
        hil_serl_root = str(self.get_parameter("hil_serl_root").value).strip()
        hil_serl_root = hil_serl_root or os.environ.get("HIL_SERL_ROOT", "")
        self._classifier = self._load_classifier(hil_serl_root, checkpoint)

        self._lock = threading.Lock()
        self._frames = {"cam1": None, "cam2": None}
        self._publisher = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("cam1_topic").value),
            lambda msg: self._on_image("cam1", msg),
            10,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("cam2_topic").value),
            lambda msg: self._on_image("cam2", msg),
            10,
        )
        self.create_timer(1.0 / max(inference_hz, 0.1), self._infer_tick)
        self._publish_status(
            ready=False,
            probability=None,
            success=False,
            threshold=self._threshold,
            message="waiting for synchronized camera frames",
        )
        self.get_logger().info(
            "reward classifier loaded; checkpoint=%s threshold=%.3f status=%s"
            % (checkpoint, self._threshold, STATUS_TOPIC)
        )

    def _load_classifier(self, hil_serl_root: str, checkpoint: str):
        if not checkpoint:
            raise RuntimeError(
                "checkpoint_path parameter or REWARD_CLASSIFIER_CHECKPOINT "
                "environment variable is required"
            )
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.exists(checkpoint):
            raise RuntimeError("classifier checkpoint not found: %s" % checkpoint)

        if hil_serl_root:
            launcher_root = os.path.join(
                os.path.abspath(os.path.expanduser(hil_serl_root)), "serl_launcher"
            )
            if launcher_root not in sys.path:
                sys.path.insert(0, launcher_root)

        import jax
        from serl_launcher.networks.reward_classifier import load_classifier_func

        sample = {
            "state": np.zeros((1, 1), dtype=np.float32),
            "cam1": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "cam2": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        }
        classifier = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=sample,
            image_keys=list(IMAGE_KEYS),
            checkpoint_path=checkpoint,
        )
        # Compile before subscribing/operating so the first displayed result is
        # not confused with JAX's one-time compilation latency.
        warmup_logit = classifier(sample)
        np.asarray(warmup_logit).item()
        return classifier

    def _on_image(self, key: str, msg: CompressedImage) -> None:
        item = (bytes(msg.data), ros_stamp_ns(msg), time.monotonic())
        with self._lock:
            self._frames[key] = item

    def _snapshot_pair(self):
        with self._lock:
            return self._frames["cam1"], self._frames["cam2"]

    def _publish_status(self, **values) -> None:
        msg = String()
        msg.data = status_json(**values)
        self._publisher.publish(msg)

    def _infer_tick(self) -> None:
        cam1, cam2 = self._snapshot_pair()
        now = time.monotonic()
        base = {"threshold": self._threshold, "success": False}
        if cam1 is None or cam2 is None:
            self._publish_status(
                **base,
                ready=False,
                probability=None,
                message="waiting for cam1/cam2",
            )
            return

        age = max(now - cam1[2], now - cam2[2])
        skew_s = abs(cam1[1] - cam2[1]) / 1e9
        if age > self._max_age_s:
            self._publish_status(
                **base,
                ready=False,
                probability=None,
                cam_skew_ms=skew_s * 1000.0,
                message="camera frame stale (%.3fs)" % age,
            )
            return
        if skew_s > self._max_skew_s:
            self._publish_status(
                **base,
                ready=False,
                probability=None,
                cam_skew_ms=skew_s * 1000.0,
                message="camera timestamp skew too large",
            )
            return

        started = time.monotonic()
        try:
            obs = make_observation(cam1[0], cam2[0])
            logit = float(np.asarray(self._classifier(obs)).item())
            probability = sigmoid_probability(logit)
        except Exception as exc:  # keep publishing diagnostics; never kill GUI
            self.get_logger().error("classifier inference failed: %s" % exc)
            self._publish_status(
                **base,
                ready=False,
                probability=None,
                cam_skew_ms=skew_s * 1000.0,
                message="inference error: %s" % exc,
            )
            return
        inference_ms = (time.monotonic() - started) * 1000.0
        self._publish_status(
            ready=True,
            probability=probability,
            success=probability > self._threshold,
            threshold=self._threshold,
            cam_skew_ms=skew_s * 1000.0,
            inference_ms=inference_ms,
            message="ok",
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RewardClassifierNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
