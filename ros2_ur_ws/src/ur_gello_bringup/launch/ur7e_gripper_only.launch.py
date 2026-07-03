#!/usr/bin/env python3
"""Gripper-ONLY bringup for the RWH UR7e's Robotiq 2F-85 (Modbus over tool-comm).

Launches just the ``robotiq_gripper_modbus`` node — no arm, no ros2_control. The node
talks Modbus RTU straight over the UR controller's RS485 tool-communication TCP port
(``robot_ip:54321``), which requires the UR "RS485 / tool communication" URCap on
PolyScope (already installed on this robot) and the robot to be POWERED ON.

Usage:
    ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py
    ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py robot_ip:=192.168.10.11 force:=60

Then, from another terminal:
    # open / close via the convenience service
    ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"    # close
    ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"   # open
    # or the standard action (position in METERS: 0.0 closed .. 0.085 open)
    ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
        control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 40.0}}"
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    speed = LaunchConfiguration("speed")
    force = LaunchConfiguration("force")
    serial_port = LaunchConfiguration("serial_port")

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.10.11",
                              description="UR controller IP (RS485 tool-comm on :54321)."),
        DeclareLaunchArgument("speed", default_value="150",
                              description="Gripper speed 0-255."),
        DeclareLaunchArgument("force", default_value="50",
                              description="Default gripper force 0-255 (used when action max_effort=0)."),
        DeclareLaunchArgument(
            "serial_port", default_value="",
            description=(
                "Leave EMPTY to talk Modbus directly over TCP to robot_ip:54321 "
                "(gripper-only, node owns the single-client forwarder). Set to a serial "
                "device (e.g. /tmp/ttyUR) to instead share the bridge created by the ARM "
                "driver's use_tool_communication:=true — REQUIRED when the ur_robot_driver "
                "is also running, since :54321 allows only one client."),
        ),
        Node(
            package="ur_gello_bringup",
            executable="robotiq_gripper_modbus",
            name="robotiq_gripper",
            output="screen",
            parameters=[{
                "robot_ip": robot_ip,
                "tool_comm_port": 54321,
                "serial_port": serial_port,
                "speed": speed,
                "force": force,
                "connect_on_start": True,
                "activate_on_connect": True,
            }],
        ),
    ])
