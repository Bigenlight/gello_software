"""Synthetic camera/gripper observations for fake-hardware integration tests only."""

import base64

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


# Valid 1x1 black JPEG. The remote server decodes and resizes it exactly as it
# does a camera JPEG; no OpenCV/RealSense dependency is needed on the laptop.
_BLACK_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/"
    "2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAEDASIAAhEBAxEB/"
    "8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2Jy"
    "ggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLD"
    "xMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3"
    "AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6"
    "goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+ii"
    "igD/2Q=="
)


class FakeDiffusionObservationNode(Node):
    """Publish deterministic inputs missing from ros2_control fake hardware.

    The fake UR driver already publishes ``/joint_states``. This node supplies
    only the two compressed camera topics and gripper-position feedback needed
    by ``policy_leader_node``. It must never be used for a physical-robot run.
    """

    def __init__(self):
        super().__init__("fake_diffusion_observations")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("cam1_topic", "/cam1/cam1/color/image_raw/compressed")
        self.declare_parameter("cam2_topic", "/cam2/cam2/color/image_raw/compressed")
        self.declare_parameter("gripper_position", 0.0)

        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        gripper_position = float(self.get_parameter("gripper_position").value)
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("gripper_position must be in [0, 1]")
        self._gripper_position = gripper_position
        cam1_topic = str(self.get_parameter("cam1_topic").value)
        cam2_topic = str(self.get_parameter("cam2_topic").value)

        self._cam1_pub = self.create_publisher(CompressedImage, cam1_topic, 10)
        self._cam2_pub = self.create_publisher(CompressedImage, cam2_topic, 10)
        self._grip_pub = self.create_publisher(
            Float32, "/robotiq_gripper/position_percent", 10
        )
        self._timer = self.create_timer(1.0 / rate, self._publish)
        self.get_logger().warning(
            "FAKE observations enabled: black JPEGs and fixed gripper feedback; "
            "use only with use_fake_hardware:=true."
        )

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        for publisher, frame_id in (
            (self._cam1_pub, "fake_cam1"),
            (self._cam2_pub, "fake_cam2"),
        ):
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id
            msg.format = "jpeg"
            msg.data = _BLACK_JPEG
            publisher.publish(msg)

        grip = Float32()
        grip.data = self._gripper_position
        self._grip_pub.publish(grip)


def main(args=None):
    rclpy.init(args=args)
    node = FakeDiffusionObservationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
