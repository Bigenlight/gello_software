#!/usr/bin/env python3
"""Bring up a UR7e in RViz with mock/fake hardware and drive it from GELLO.

This is the turnkey FAKE/mock entrypoint for the UR7e. It lets you
VISUALIZE GELLO -> UR7e teleoperation in RViz WITHOUT a real robot and
WITHOUT the GELLO leader arm attached. It:

  1. Includes the official ``ur_control.launch.py`` from ``ur_robot_driver``
     with ``ur_type:=ur7e`` and ``use_fake_hardware:=true``. The ros2_control
     mock hardware simply mirrors position commands back to the joint states,
     so any command sent to the ``forward_position_controller`` makes the
     RViz model move.
  2. A few seconds later (once the controller is spawned), starts our
     ``gello_ur_bridge`` node, which subscribes to the GELLO joint stream and
     republishes it on ``/forward_position_controller/commands``
     (Float64MultiArray, 6 values, UR joint order:
     shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3).
  3. Also starts the GELLO source:
       * ``source:=fake``  -> ``fake_gello`` (synthetic joint stream, no hardware)
       * ``source:=gello`` -> ``gello_publisher`` (reads the real GELLO arm)
  4. Optionally the Robotiq gripper node (harmless logging by default).

The mock arm cannot be damaged, so this FAKE launch does NOT run the
gello_move_to_start handshake (that safety step is only needed on real HW,
in ur7e_gello_real.launch.py).

SAFETY: GELLO is ALWAYS a passive, read-only leader. The driver never applies
torque to the GELLO Dynamixels; it only reads their joint angles.

Example:
    ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py                # fake source
    ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello  # real GELLO
    ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    declared_arguments = [
        DeclareLaunchArgument(
            "source",
            default_value="fake",
            choices=["fake", "gello"],
            description=(
                "GELLO command source: 'fake' runs the synthetic fake_gello "
                "node (no hardware), 'gello' runs the real gello_publisher."
            ),
        ),
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur7e",
            description="Type/series of the UR robot (e.g. ur3e, ur5e, ur7e, ur10e).",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.56.101",
            description=(
                "IP address of the robot. Dummy value; unused when "
                "use_fake_hardware is true."
            ),
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description=(
                "Use ros2_control mock hardware instead of a real robot. "
                "Mock hardware mirrors commands to states so RViz moves."
            ),
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz for visualization.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("ur_gello_bringup"),
                    "config",
                    "ur7e_gello.yaml",
                ]
            ),
            description="YAML parameter file for the GELLO bridge / publisher.",
        ),
    ]

    # ------------------------------------------------------------------ #
    # Substitutions
    # ------------------------------------------------------------------ #
    source = LaunchConfiguration("source")
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    params_file = LaunchConfiguration("params_file")

    # ------------------------------------------------------------------ #
    # Official UR driver bring-up (ros2_control + RViz + mock hardware)
    # ------------------------------------------------------------------ #
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur_robot_driver"),
                    "launch",
                    "ur_control.launch.py",
                ]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            # ur_robot_driver >= 3.x (Jazzy) calls this 'use_mock_hardware';
            # Humble's ur_control.launch.py uses 'use_fake_hardware'. Pass both
            # — unknown launch_arguments are silently ignored by the include.
            "use_mock_hardware": use_fake_hardware,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": launch_rviz,
            # We drive the arm with raw position commands, not trajectories.
            "initial_joint_controller": "forward_position_controller",
            "activate_joint_controller": "true",
            # headless_mode avoids needing the External Control URCap program
            # to be running (relevant under fake hardware / no pendant).
            "headless_mode": "true",
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # GELLO -> UR bridge: subscribes to GELLO stream, publishes to
    # /forward_position_controller/commands.
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        parameters=[params_file],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # GELLO source node (chosen by 'source' argument)
    # ------------------------------------------------------------------ #
    is_fake = IfCondition(
        PythonExpression(["'", source, "' == 'fake'"])
    )
    is_gello = IfCondition(
        PythonExpression(["'", source, "' == 'gello'"])
    )

    fake_gello_node = Node(
        package="ur_gello_bringup",
        executable="fake_gello",
        output="screen",
        condition=is_fake,
    )

    gello_publisher_node = Node(
        package="ur_gello_bringup",
        executable="gello_publisher",
        parameters=[params_file],
        output="screen",
        condition=is_gello,
    )

    # ------------------------------------------------------------------ #
    # Optional Robotiq gripper node (harmless; connect_on_start=false).
    # ------------------------------------------------------------------ #
    gripper_node = Node(
        package="ur_gello_bringup",
        executable="robotiq_urcap",
        parameters=[params_file, {"connect_on_start": False}],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Delay our nodes so the controller is spawned before we command it.
    # ------------------------------------------------------------------ #
    delayed_nodes = TimerAction(
        period=5.0,
        actions=[
            bridge_node,
            fake_gello_node,
            gello_publisher_node,
            gripper_node,
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            ur_control_launch,
            delayed_nodes,
        ]
    )
