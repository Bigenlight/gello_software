#!/usr/bin/env python3
"""Read-only ROS node backing the standalone reward-classifier monitor GUI."""

import json
import threading
import time

import rclpy
from std_msgs.msg import String

from gello_recorder.gello_gui_node import GelloRecorderGuiNode

CLASSIFIER_STATUS_TOPIC = "/reward_classifier/status"

class ClassifierViewGuiNode(GelloRecorderGuiNode):
    """Existing camera/robot-state subscriptions plus classifier status only.

    There are deliberately no policy start/hold clients in this class.
    """

    def __init__(
        self,
        cam1_topic="/cam1/cam1/color/image_raw/compressed",
        cam2_topic="/cam2/cam2/color/image_raw/compressed",
        node_name="classifier_view_gui_node",
    ):
        super().__init__(
            cam1_topic=cam1_topic,
            cam2_topic=cam2_topic,
            camera_fps=30.0,
            camera_warmup_s=0.0,
            output_root="~/gello_recordings",
            node_name=node_name,
        )
        self._classifier_lock = threading.Lock()
        self._classifier_status = None
        self._classifier_status_t = None
        self.create_subscription(
            String, CLASSIFIER_STATUS_TOPIC, self._on_classifier_status, 10
        )
        self.get_logger().info(
            "standalone classifier viewer listening on %s"
            % CLASSIFIER_STATUS_TOPIC
        )

    def _on_classifier_status(self, msg):
        try:
            status = json.loads(msg.data)
            if not isinstance(status, dict):
                raise ValueError("status must be a JSON object")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            status = {
                "ready": False,
                "message": "invalid classifier status: %s" % exc,
            }
        with self._classifier_lock:
            self._classifier_status = status
            self._classifier_status_t = time.monotonic()

    def get_classifier_status(self):
        with self._classifier_lock:
            status = (
                dict(self._classifier_status)
                if self._classifier_status is not None
                else {
                    "ready": False,
                    "message": "classifier node not connected",
                }
            )
            received = self._classifier_status_t
        status["status_age_s"] = (
            None if received is None else time.monotonic() - received
        )
        return status


def main(args=None):
    rclpy.init(args=args)
    node = ClassifierViewGuiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
