#!/usr/bin/env python3
"""Bring up a UR7e in RViz with mock/fake hardware, driven via the EEF
(end-effector delta-pose) bridge control mode. NO REAL ROBOT REQUIRED.

This is the turnkey validation entrypoint for `control_mode:=eef` on
`gello_ur_bridge`. It mirrors `ur7e_gello_rviz.launch.py` (mock ros2_control +
GELLO source + bridge + RViz) but forces the bridge into eef mode and layers
`config/ur7e_gello_eef.yaml` on top of `params_file`, so the full engage/
reclutch/disengage gate stack (G1-G9), the eef delta controller, and the
zero-jump BOOTSTRAP -> ENGAGED state machine can be exercised end-to-end
against safe, undamageable mock hardware:

  1. Includes the official ``ur_control.launch.py`` from ``ur_robot_driver``
     with ``ur_type:=ur7e`` and ``use_fake_hardware:=true``. The ros2_control
     mock hardware simply mirrors position commands back to the joint states,
     so any command sent to the ``forward_position_controller`` makes the
     RViz model move -- nothing here can be damaged.
  2. A few seconds later, starts our ``gello_ur_bridge`` node with
     ``control_mode:=eef``, parameters layered as
     ``[params_file, ur7e_gello_eef.yaml, {control_mode: eef}]`` so the eef
     overlay always wins on key conflicts. The bridge boots in BOOTSTRAP
     (joint passthrough) until ``~/eef_engage`` is called; use
     ``~/eef_engage``, ``~/eef_reclutch``, ``~/eef_disengage``,
     ``~/eef_to_joint`` (all std_srvs/Trigger) to drive the state machine and
     watch ``~/eef/state`` (std_msgs/String, JSON) plus
     ``~/eef/{leader,desired,commanded}_pose`` (PoseStamped) for the reject
     reasons / gate failures (NO_IK, BRANCH_JUMP, JOINT_LIMIT, GEOM_KEEPOUT,
     EXCURSION, ESTOP, BAD_INPUT) without ever risking real hardware.
  3. Also starts the GELLO source:
       * ``source:=fake``  -> ``fake_gello`` (synthetic joint stream, no
         hardware). Exposes ``pattern`` (forwarded to fake_gello's
         ``pattern`` parameter) so a reviewer can pick which synthetic motion
         drives the eef gates (e.g. a held-still pose to pass the
         leader-quasi-still gate G5, or a sweep to exercise HOLD/reject
         paths). Consult fake_gello's own parameter docs for the exact set of
         supported values; this launch only forwards whatever string you
         pass through unmodified.
       * ``source:=gello`` -> ``gello_publisher`` (reads the real, PASSIVE
         GELLO leader arm; still no real UR robot is involved).
  4. Optionally the Robotiq gripper node (harmless logging by default,
     ``connect_on_start:=false``) -- the gripper path is unchanged by
     control_mode and is included here only for structural parity with
     ``ur7e_gello_rviz.launch.py``.

This mock launch does NOT run the gello_move_to_start handshake (mock
hardware cannot be damaged by a first-command jump); that safety step is only
mandatory on real HW, in ur7e_gello_real.launch.py.

SAFETY: GELLO is ALWAYS a passive, read-only leader. The driver never applies
torque to the GELLO Dynamixels; it only reads their joint angles.

Example:
    ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py
    ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py source:=gello
    ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py \
        source:=fake pattern:=hold launch_rviz:=false
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
                "node (no hardware), 'gello' runs the real gello_publisher "
                "against the physical (passive, read-only) GELLO leader."
            ),
        ),
        DeclareLaunchArgument(
            "pattern",
            default_value="sweep",
            description=(
                "Forwarded verbatim to fake_gello's 'pattern' parameter "
                "(only consulted when source:=fake). Lets a reviewer pick "
                "which synthetic leader motion drives the eef engage gates "
                "(e.g. a still pose to satisfy the leader-quasi-still gate, "
                "or a sweep to exercise HOLD / reject paths). See "
                "fake_gello_node.py for the supported values."
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
                "Mock hardware mirrors commands to states so RViz moves. "
                "This launch is meant to run with this true (no real robot)."
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
            description="Base YAML parameter file for the GELLO bridge / publisher.",
        ),
        DeclareLaunchArgument(
            "eef_params_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("ur_gello_bringup"),
                    "config",
                    "ur7e_gello_eef.yaml",
                ]
            ),
            description=(
                "EEF overlay YAML, layered on TOP of params_file for the "
                "bridge only (later entries in a Node's `parameters` list "
                "win on key conflicts)."
            ),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Substitutions
    # ------------------------------------------------------------------ #
    source = LaunchConfiguration("source")
    pattern = LaunchConfiguration("pattern")
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    params_file = LaunchConfiguration("params_file")
    eef_params_file = LaunchConfiguration("eef_params_file")

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
            # -- unknown launch_arguments are silently ignored by the include.
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
    # GELLO -> UR bridge, forced into control_mode:=eef. The eef overlay
    # yaml is layered AFTER params_file so it wins on any key overlap; the
    # trailing dict override guarantees control_mode is "eef" regardless of
    # what either yaml file says.
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        parameters=[
            params_file,
            eef_params_file,
            {"control_mode": "eef"},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # GELLO source node (chosen by 'source' argument)
    # ------------------------------------------------------------------ #
    is_fake = IfCondition(PythonExpression(["'", source, "' == 'fake'"]))
    is_gello = IfCondition(PythonExpression(["'", source, "' == 'gello'"]))

    fake_gello_node = Node(
        package="ur_gello_bringup",
        executable="fake_gello",
        parameters=[{"pattern": pattern}],
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
    # Structural parity with ur7e_gello_rviz.launch.py; the eef control mode
    # only affects the 6-DoF arm path, never the gripper.
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
