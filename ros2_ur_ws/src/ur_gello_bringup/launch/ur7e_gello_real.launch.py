#!/usr/bin/env python3
"""Bring up a REAL UR7e and drive it from the physical GELLO leader arm.

    ############################################################################
    #  READ THIS BEFORE RUNNING ON REAL HARDWARE                               #
    #                                                                          #
    #  SAFETY INVARIANT (non-negotiable):                                      #
    #    GELLO is ALWAYS a PASSIVE, READ-ONLY leader. We NEVER apply torque    #
    #    to the GELLO Dynamixels. The gello_publisher driver initialises       #
    #    torque OFF and only READS joint positions. Do not change this.        #
    #                                                                          #
    #  MANDATORY move-to-start HANDSHAKE:                                      #
    #    On real hardware the forward_position_controller must NOT be the      #
    #    initially-active controller. If it were, the very first command it    #
    #    receives would snap the UR7e from its current pose to the GELLO pose  #
    #    in a single step -> a violent, dangerous jump.                        #
    #                                                                          #
    #    Instead we bring the robot up on the                                  #
    #    scaled_joint_trajectory_controller (active) and run the               #
    #    gello_move_to_start node. That node commands a smooth, time-          #
    #    parameterised trajectory from the robot's current pose to the GELLO   #
    #    leader's current pose (via                                            #
    #    /scaled_joint_trajectory_controller/follow_joint_trajectory), and     #
    #    ONLY once the robot has caught up does it switch control to the       #
    #    forward_position_controller (STRICT switch:                           #
    #      activate=[forward_position_controller],                             #
    #      deactivate=[scaled_joint_trajectory_controller]).                   #
    #    Only after that switch does gello_ur_bridge begin streaming raw       #
    #    position commands. This eliminates the start-up jump.                 #
    ############################################################################

Bring-up sequence (staggered TimerActions):

  (1) t=0s    ur_control.launch.py -> UR7e driver + ros2_control.
                  * scaled_joint_trajectory_controller is loaded ACTIVE
                    (initial_joint_controller + activate_joint_controller).
                  * forward_position_controller is loaded INACTIVE (it lives
                    in the driver's controllers.yaml; we spawn it stopped so
                    gello_move_to_start can switch to it later).
  (2) t=6s    gello_publisher -> reads the PASSIVE GELLO arm and publishes
                  /gello/joint_states (6 arm joints, radians) +
                  /gripper/... width percent. Started early so a live GELLO
                  pose is available for the handshake target.
  (3) t=8s    gello_move_to_start -> MANDATORY handshake. Waits for the External
                  Control program to be running (scaled_joint_trajectory_controller
                  active), smoothly drives the UR7e to the current GELLO pose,
                  then STRICT-switches to forward_position_controller. Exits 0.
  (4) on handshake success -> gello_ur_bridge streams /gello/joint_states onto
                  /forward_position_controller/commands (publish_rate_hz, 250 Hz
                  in ur7e_gello.yaml). The bridge seeds from the robot's ACTUAL
                  /joint_states, so it ramps from the real pose to GELLO with no
                  snap. Started by an OnProcessExit handler (NOT a fixed timer).
  (5) with the bridge -> the Robotiq 2F-85 gripper comes up via Modbus RTU over
                  the UR RS485 tool-communication bus. The arm driver is started
                  with use_tool_communication:=true, tool_voltage:=24,
                  tool_device_name:=/tmp/ttyUR, so the driver both POWERS the
                  tool (24V) and starts the UR tool_communication (socat)
                  forwarder that OWNS robot_ip:54321 and exposes it as the
                  serial device /tmp/ttyUR. The gripper node SHARES that bridge
                  with serial_port:=/tmp/ttyUR (NOT direct TCP), so exactly one
                  client owns :54321 = the driver's socat forwarder.
                  gello_gripper_bridge maps the GELLO width topic
                  (0=OPEN..1=CLOSED) onto /robotiq_gripper/command_percent
                  (0=open..1=closed, no inversion) so closing the GELLO hand
                  closes the robot gripper. The ROBOT MUST BE POWERED ON so tool
                  voltage powers the 2F-85.

Prerequisites on this machine (real GELLO path):
    sudo apt install ros-humble-dynamixel-sdk
    export GELLO_REPO_ROOT=/home/laptop3/gello_software   # gello_publisher import root

Example:
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=192.168.10.11
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
        robot_ip:=192.168.10.11 \
        kinematics_params_file:=/path/to/my_robot_calibration.yaml \
        headless_mode:=true

This launch file is DOCUMENTED / real-robot-only; it is NOT verifiable on the
mock-hardware CI machine (no robot, no GELLO serial). For the mock/RViz path
use ur7e_gello_rviz.launch.py (source:=fake).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur7e",
            description="Type/series of the UR robot. This runbook targets ur7e.",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            # NO default_value on purpose: on real hardware the robot IP MUST
            # be supplied explicitly, e.g. robot_ip:=192.168.10.11. Launch
            # will error out if it is omitted, which is the desired behaviour.
            description="REQUIRED. IP address of the physical UR7e controller.",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description=(
                "If true, the driver runs without requiring the External "
                "Control URCap program to be started from the pendant. On a "
                "real robot the usual flow is headless_mode:=false and you "
                "start the External Control program on the teach pendant."
            ),
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz to visualise the robot state.",
        ),
        DeclareLaunchArgument(
            "kinematics_params_file",
            # Default to the nominal per-ur_type kinematics that ships with
            # ur_description (e.g. config/ur7e/default_kinematics.yaml). Do NOT
            # default to "" — an empty value is forwarded to the description
            # xacro, which then falls back to a non-existent 'ur5x' default and
            # the launch fails. Override with the ur_calibration output for a
            # per-robot-accurate description.
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("ur_description"),
                    "config",
                    LaunchConfiguration("ur_type"),
                    "default_kinematics.yaml",
                ]
            ),
            description=(
                "Path to a kinematics calibration YAML. Defaults to the nominal "
                "ur_description kinematics for the chosen ur_type. Strongly "
                "recommended to override with the per-robot ur_calibration "
                "output for accurate Cartesian behaviour."
            ),
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
            description=(
                "YAML parameter file for the GELLO publisher / bridge / "
                "move-to-start / gripper nodes."
            ),
        ),
        DeclareLaunchArgument(
            "start_mode",
            default_value="gello",
            description=(
                "Handshake style for gello_move_to_start. 'gello' (default): the "
                "arm moves straight to the leader's current pose, then streams. "
                "'init_align': the arm moves to a fixed init_pose (from the params "
                "file), then waits for you to align the GELLO leader to it before "
                "streaming (safer first motion)."
            ),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Substitutions
    # ------------------------------------------------------------------ #
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_rviz = LaunchConfiguration("launch_rviz")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")
    params_file = LaunchConfiguration("params_file")

    # ------------------------------------------------------------------ #
    # (1) Official UR driver bring-up (REAL hardware, ros2_control).
    #
    #  * use_fake_hardware:=false            -> talk to the real UR7e.
    #  * initial_joint_controller:=scaled_joint_trajectory_controller
    #    + activate_joint_controller:=true  -> trajectory controller ACTIVE
    #                                          at start (needed by the
    #                                          move-to-start handshake).
    #  * forward_position_controller is present in the driver's
    #    ur_controllers.yaml and is loaded INACTIVE by the controller
    #    spawner (it is not in the driver's active set). gello_move_to_start
    #    later switches to it via /controller_manager/switch_controller
    #    (STRICT: activate=[forward_position_controller],
    #             deactivate=[scaled_joint_trajectory_controller]).
    #
    #    If your installed ur_robot_driver does NOT auto-load
    #    forward_position_controller in a stopped state, load it manually
    #    after the driver is up, BEFORE the handshake, e.g.:
    #        ros2 control load_controller forward_position_controller \
    #            --set-state inactive
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
            # Real hardware: mock/fake both OFF. Pass both spellings so the
            # value is honoured across driver versions (unknown launch
            # arguments are silently ignored by the include).
            "use_mock_hardware": "false",
            "use_fake_hardware": "false",
            "launch_rviz": launch_rviz,
            "headless_mode": headless_mode,
            # Bring the robot up on the trajectory controller (ACTIVE) so the
            # move-to-start handshake can command a smooth catch-up motion.
            # forward_position_controller must NOT be active initially.
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
            # Optional per-robot calibration (ur_calibration output). Forwarded
            # as-is; empty string -> driver uses nominal kinematics.
            "kinematics_params_file": kinematics_params_file,
            # Tool communication for the Robotiq 2F-85 gripper. This makes the
            # DRIVER (not the pendant Installation tab) provision the tool:
            #   * use_tool_communication:=true starts the UR tool_communication
            #     (socat) forwarder, which OWNS robot_ip:54321 and exposes it as
            #     the serial device /tmp/ttyUR.
            #   * tool_voltage:=24 powers the 2F-85 (24V) from the driver, so it
            #     stays up while External Control is PLAYING (applying tool
            #     voltage via the pendant would be cut by EC starting, and
            #     re-applying it via the pendant would STOP EC).
            #   * tool_device_name:=/tmp/ttyUR is the serial device the gripper
            #     Modbus node then SHARES (serial_port:=/tmp/ttyUR below), so
            #     exactly ONE client owns :54321 = this socat forwarder.
            "use_tool_communication": "true",
            "tool_voltage": "24",
            "tool_device_name": "/tmp/ttyUR",
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # (2) GELLO source: reads the PASSIVE leader arm (torque OFF) and
    #     publishes /gello/joint_states. Started before the handshake so a
    #     live GELLO pose is available as the move-to-start target.
    # ------------------------------------------------------------------ #
    gello_publisher_node = Node(
        package="ur_gello_bringup",
        executable="gello_publisher",
        parameters=[params_file],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (3) MANDATORY move-to-start handshake. Drives the UR7e smoothly to the
    #     current GELLO pose via
    #     /scaled_joint_trajectory_controller/follow_joint_trajectory, then
    #     STRICT-switches control to forward_position_controller.
    # ------------------------------------------------------------------ #
    move_to_start_node = Node(
        package="ur_gello_bringup",
        executable="gello_move_to_start",
        # start_mode launch arg overrides the value in params_file (last wins).
        parameters=[params_file, {"start_mode": LaunchConfiguration("start_mode")}],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (4) GELLO -> UR bridge. Streams /gello/joint_states onto
    #     /forward_position_controller/commands (publish_rate_hz, 250 Hz). Only meaningful
    #     AFTER the handshake has switched to forward_position_controller.
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        parameters=[params_file],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (5) Robotiq 2F-85 gripper over Modbus RTU (SHARED socat bridge /tmp/ttyUR).
    #
    #  The gripper is driven via Modbus RTU over the UR RS485 tool-
    #  communication bus. Because the arm bring-up (ur_control_launch above)
    #  is started with use_tool_communication:=true / tool_voltage:=24 /
    #  tool_device_name:=/tmp/ttyUR, the DRIVER powers the tool (24V) AND runs
    #  the tool_communication (socat) forwarder that OWNS robot_ip:54321 and
    #  exposes it as the serial device /tmp/ttyUR. So we do NOT use direct TCP
    #  here — instead the gripper node SHARES the driver's bridge with
    #  serial_port:=/tmp/ttyUR, keeping exactly ONE client on :54321 (the socat
    #  forwarder). The ROBOT MUST BE POWERED ON so tool voltage powers the
    #  2F-85; a powered-off robot gives no Modbus response and the node simply
    #  keeps retrying (not a bug).
    #
    #  Interfaces preserved: action /robotiq_gripper_controller/gripper_cmd,
    #  service /robotiq_gripper/set_closed, status ~/position_percent +
    #  ~/joint_states. Streaming command topic ~/command_percent (Float32,
    #  0=open..1=closed) is what the bridge drives.
    # ------------------------------------------------------------------ #
    gripper_modbus_node = Node(
        package="ur_gello_bringup",
        executable="robotiq_gripper_modbus",
        parameters=[
            params_file,
            # serial_port=/tmp/ttyUR -> SHARE the driver's socat bridge (NOT
            # direct TCP). The driver's tool_communication forwarder owns
            # :54321; this node reads/writes the serial device it exposes.
            # robot_ip is injected here (not in the yaml). connect_on_start
            # defaults True in the node so it opens the device on start.
            {"robot_ip": robot_ip, "serial_port": "/tmp/ttyUR"},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (6) GELLO -> gripper bridge. Subscribes the GELLO width topic
    #     /gripper/gripper_client/target_gripper_width_percent and republishes
    #     it onto /robotiq_gripper/command_percent with the CORRECT direction
    #     (width 0=OPEN..1=CLOSED -> command_percent 0=open..1=closed,
    #     invert=False). GELLO stays passive; this node only reads a topic and
    #     publishes a Float32 (no hardware access).
    # ------------------------------------------------------------------ #
    gello_gripper_bridge_node = Node(
        package="ur_gello_bringup",
        executable="gello_gripper_bridge",
        parameters=[params_file],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Start-up sequencing.
    #   * gello_publisher after 6s (driver/ros2_control up; publishes the live
    #     GELLO pose the handshake needs).
    #   * move_to_start after 8s. It now WAITS internally for
    #     scaled_joint_trajectory_controller to become active (i.e. for the
    #     External Control program to be PLAYING on the pendant), so a fixed
    #     delay is no longer timing-critical.
    #   * arm bridge + Modbus gripper + gello_gripper_bridge are started ONLY
    #     when move_to_start EXITS SUCCESSFULLY (return code 0 == handshake done,
    #     forward_position_controller active). This replaces the old fixed
    #     bridge_start_delay timer, which could fire before the
    #     (variable-duration) handshake had switched controllers and leave the
    #     arm commanded on an inactive controller (== no motion).
    # ------------------------------------------------------------------ #
    gello_publisher_delayed = TimerAction(
        period=6.0,
        actions=[gello_publisher_node],
    )
    move_to_start_delayed = TimerAction(
        period=8.0,
        actions=[move_to_start_node],
    )

    def _on_handshake_exit(event, context):
        if event.returncode == 0:
            return [
                LogInfo(
                    msg=(
                        "Move-to-start handshake SUCCEEDED "
                        "(forward_position_controller active); starting "
                        "gello_ur_bridge streaming + Modbus gripper "
                        "(shared socat bridge /tmp/ttyUR) + gello_gripper_bridge."
                    )
                ),
                bridge_node,
                gripper_modbus_node,
                gello_gripper_bridge_node,
            ]
        return [
            LogInfo(
                msg=(
                    "Move-to-start handshake FAILED (exit "
                    f"{event.returncode}); NOT starting the bridge. The robot "
                    "will stay put. Check that the External Control program is "
                    "PLAYING on the pendant, then re-launch."
                )
            )
        ]

    bridge_after_handshake = RegisterEventHandler(
        OnProcessExit(
            target_action=move_to_start_node,
            on_exit=_on_handshake_exit,
        )
    )

    return LaunchDescription(
        declared_arguments
        + [
            ur_control_launch,
            gello_publisher_delayed,
            move_to_start_delayed,
            bridge_after_handshake,
        ]
    )
