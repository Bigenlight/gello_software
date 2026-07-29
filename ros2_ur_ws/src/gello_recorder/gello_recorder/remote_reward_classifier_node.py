#!/usr/bin/env python3
"""ROS camera client for a reward classifier reached through an SSH tunnel."""

from collections import deque
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .remote_classifier_runtime import (
    DEFAULT_ENDPOINT, RemoteInferenceWorker, oldest_pair_receipt_monotonic)
from .reward_classifier_node import STATUS_TOPIC, ros_stamp_ns
from .reward_classifier_runtime import (
    default_threshold, status_json, validate_threshold)


class RemoteRewardClassifierNode(Node):
    def __init__(self):
        super().__init__("remote_reward_classifier_node")
        self.declare_parameter("cam1_topic", "/cam1/cam1/color/image_raw/compressed")
        self.declare_parameter("cam2_topic", "/cam2/cam2/color/image_raw/compressed")
        self.declare_parameter("endpoint", DEFAULT_ENDPOINT)
        self.declare_parameter("threshold", default_threshold())
        self.declare_parameter("request_hz", 10.0)
        self.declare_parameter("timeout_s", 2.0)
        self.declare_parameter("max_camera_age_s", 0.5)
        self.declare_parameter("max_camera_skew_s", 0.10)
        self._threshold = validate_threshold(self.get_parameter("threshold").value)
        self._max_age = float(self.get_parameter("max_camera_age_s").value)
        self._max_skew_ns = int(
            float(self.get_parameter("max_camera_skew_s").value) * 1e9
        )
        self._queues = {"cam1": deque(maxlen=12), "cam2": deque(maxlen=12)}
        self._latest_pair = None
        self._last_submitted_stamps = None
        self._pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.create_subscription(
            CompressedImage, str(self.get_parameter("cam1_topic").value),
            lambda msg: self._on_image("cam1", msg), 10)
        self.create_subscription(
            CompressedImage, str(self.get_parameter("cam2_topic").value),
            lambda msg: self._on_image("cam2", msg), 10)
        self._worker = RemoteInferenceWorker(
            str(self.get_parameter("endpoint").value),
            float(self.get_parameter("timeout_s").value),
            self._threshold)
        self._worker.start()
        hz = max(0.1, float(self.get_parameter("request_hz").value))
        self.create_timer(1.0 / hz, self._tick)
        self._publish(False, None, False, "waiting for synchronized cameras")

    def _on_image(self, key, msg):
        self._queues[key].append(
            (bytes(msg.data), ros_stamp_ns(msg), time.monotonic()))
        left, right = self._queues["cam1"], self._queues["cam2"]
        if not left or not right:
            return
        # Select the globally nearest pair in the small bounded queues.
        best = min(
            ((abs(a[1] - b[1]), a, b) for a in left for b in right),
            key=lambda item: item[0])
        if best[0] <= self._max_skew_ns:
            self._latest_pair = (best[1][0], best[1][1], best[2][0], best[2][1],
                                 oldest_pair_receipt_monotonic(
                                     best[1][2], best[2][2]))
            # Frames older than this accepted pair cannot improve a later pairing.
            while left and left[0][1] <= best[1][1]:
                left.popleft()
            while right and right[0][1] <= best[2][1]:
                right.popleft()

    def _publish(self, ready, probability, success, message, **extra):
        msg = String()
        msg.data = status_json(
            ready=ready, probability=probability, success=success,
            threshold=self._threshold, message=message, remote=True, **extra)
        self._pub.publish(msg)

    def _tick(self):
        result = self._worker.poll()
        if result is not None:
            if result.get("ok"):
                probability = float(result["probability"])
                capture_age_ms = float(result.get("capture_age_ms", 0.0))
                if capture_age_ms > self._max_age * 1000.0:
                    self._publish(
                        False, None, False, "remote result stale",
                        capture_age_ms=capture_age_ms,
                        inference_ms=float(result.get("inference_ms", 0.0)),
                        roundtrip_ms=float(result.get("roundtrip_ms", 0.0)))
                    return
                self._publish(
                    True, probability, probability > self._threshold, "remote ok",
                    inference_ms=float(result.get("inference_ms", 0.0)),
                    roundtrip_ms=float(result.get("roundtrip_ms", 0.0)),
                    capture_age_ms=capture_age_ms,
                    cam_skew_ms=float(result.get("cam_skew_ms", 0.0)))
            else:
                self._publish(False, None, False, result.get("message", "remote error"))

        pair = self._latest_pair
        if pair is None:
            return
        age = time.monotonic() - pair[4]
        skew_ms = abs(pair[1] - pair[3]) / 1e6
        if age > self._max_age:
            self._publish(False, None, False, "camera frame stale", cam_skew_ms=skew_ms)
            return
        stamps = (pair[1], pair[3])
        if stamps != self._last_submitted_stamps:
            self._worker.submit(pair)
            self._last_submitted_stamps = stamps

    def destroy_node(self):
        self._worker.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RemoteRewardClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
