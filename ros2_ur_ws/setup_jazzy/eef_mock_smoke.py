"""Opt-in Jazzy EEF service smoke; launches only isolated GenericSystem hardware."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from ur_gello_bringup.fake_gello_node import START_POSE, UR_JOINT_NAMES


def main():
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise RuntimeError("Set ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")
    if not 100 <= int(os.environ.get("ROS_DOMAIN_ID", "0")) <= 200:
        raise RuntimeError("Select an unused ROS_DOMAIN_ID between 100 and 200")
    workspace = Path(__file__).resolve().parents[1]
    log_path = workspace / "log/jazzy_port_20260916/eef_mock_smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("eef_mock_smoke")
    state = {}
    actual = []
    descriptions = []
    commands = []
    process = None

    def spin(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)

    def until(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"Timeout; latest EEF state: {state}")
            spin(0.05)

    def receive_joints(message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in UR_JOINT_NAMES):
            actual[:] = [positions[name] for name in UR_JOINT_NAMES]

    def call(name, success=True, reason=None):
        client = node.create_client(Trigger, "/gello_ur_bridge/" + name)
        try:
            assert client.wait_for_service(timeout_sec=5), name
            future = client.call_async(Trigger.Request())
            until(future.done, timeout=5)
            response = future.result()
            assert response.success == success, response.message
            if reason:
                assert reason in response.message, response.message
            print(f"{name}: {response.message}", flush=True)
        finally:
            node.destroy_client(client)

    try:
        spin(1)
        assert node.get_node_names() == [node.get_name()], "ROS domain is not empty"
        node.create_subscription(String, "/gello_ur_bridge/eef/state",
                                 lambda message: state.update(json.loads(message.data)), 10)
        node.create_subscription(JointState, "/joint_states", receive_joints, 10)
        node.create_subscription(String, "/robot_description",
                                 lambda message: descriptions.append(message.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        node.create_subscription(Float64MultiArray, "/forward_position_controller/commands",
                                 lambda message: commands.append(list(message.data)), 10)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                ["/opt/ros/jazzy/bin/ros2", "launch", "ur_gello_bringup",
                 "ur7e_gello_eef_mock.launch.py", "source:=fake", "pattern:=hold",
                 "handshake:=true", "use_fake_hardware:=true", "launch_rviz:=false",
                 "robot_ip:=127.0.0.1"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            until(lambda: state.get("state") == "HOLD" and not state.get("paused")
                  and actual and descriptions)
            import xml.etree.ElementTree as ET
            plugins = [element.text for element in
                       ET.fromstring(descriptions[-1]).findall(".//ros2_control/hardware/plugin")]
            assert plugins == ["mock_components/GenericSystem"], plugins
            baseline = np.array(actual)
            spin(1)
            np.testing.assert_allclose(actual, baseline, atol=1e-8)
            call("eef_engage", success=False, reason="singular_anchor")
            call("pause")
            publisher = node.create_publisher(
                Float64MultiArray, "/forward_position_controller/commands", 10)
            until(lambda: publisher.get_subscription_count() > 0)
            seed = list(START_POSE)
            seed[0] = 0.3
            for repeat in range(10):
                publisher.publish(Float64MultiArray(data=seed))
                spin(0.05)
            until(lambda: np.max(np.abs(np.array(actual) - seed)) < 1e-6)
            node.destroy_publisher(publisher)
            call("eef_resume")
            spin(1)
            baseline = np.array(actual)
            call("eef_engage")
            until(lambda: state.get("state") == "ENGAGED")
            spin(0.4)
            np.testing.assert_allclose(actual, baseline, atol=1e-5)
            call("eef_reclutch")
            leader = node.create_publisher(Float64MultiArray, "/fake_gello/set_pose", 10)
            until(lambda: leader.get_subscription_count() > 0)
            target = list(START_POSE)
            target[0] += 0.02
            for repeat in range(5):
                leader.publish(Float64MultiArray(data=target))
                spin(0.1)
            until(lambda: np.max(np.abs(np.array(actual) - baseline)) > 0.002)
            assert state.get("state") == "ENGAGED", state
            assert all(np.isfinite(command).all() for command in commands)
            call("eef_disengage")
            until(lambda: state.get("paused") is True)
            spin(0.2)
            baseline = np.array(actual)
            target[0] += 0.02
            leader.publish(Float64MultiArray(data=target))
            spin(0.6)
            np.testing.assert_allclose(actual, baseline, atol=1e-8)
            call("eef_engage", success=False, reason="bridge_paused")
            call("eef_resume")
            spin(1)
            call("eef_engage")
            spin(0.3)
            np.testing.assert_allclose(actual, baseline, atol=1e-5)
            call("eef_disengage")
            print(f"PASS: STRICT handoff, singularity refusal, zero-jump engage/re-engage, "
                  f"reclutch, delta tracking, disengaged hold; log={log_path}")
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
