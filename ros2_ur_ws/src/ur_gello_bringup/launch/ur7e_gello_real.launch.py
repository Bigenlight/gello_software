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
    #    The bridge is pre-spawned PAUSED (start_paused:=true) at t=6s so its  #
    #    ~/resume service already exists at handover; it HOLDS (publishes      #
    #    nothing) until gello_move_to_start calls ~/resume AFTER the STRICT    #
    #    switch. HONEST GUARANTEE (not a zero-snap claim): handover happens    #
    #    only once per-joint |live GELLO - actual| <= chase_tol, sustained     #
    #    chase_dwell_s; any residual gap is then closed by the soft-started    #
    #    slew clamp (<= max_step_rad*rate, ramped over soft_start_s): a        #
    #    bounded ease-in, never a single-cycle jump.                           #
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
      t=6s    gello_ur_bridge PRE-SPAWNED PAUSED (start_paused:=true). It
                  subscribes to /gello/joint_states + the robot's actual
                  /joint_states and brings up its ~/pause + ~/resume services,
                  but publishes NOTHING on /forward_position_controller/commands
                  while paused. Pre-spawning removes the old OnProcessExit
                  cold-start window during which the leader kept drifting (which
                  widened the handover gap). It is inert until resumed.
  (3) t=8s    gello_move_to_start -> MANDATORY handshake. Waits for the External
                  Control program to be running (scaled_joint_trajectory_controller
                  active), CHASES the live GELLO pose with duration-sized catch-up
                  trajectories until per-joint |live GELLO - actual| <= chase_tol
                  sustained chase_dwell_s, then STRICT-switches to
                  forward_position_controller. Run with resume_bridge:=true, so
                  AFTER a successful switch it calls the bridge's ~/resume service
                  (alignment-gated in the bridge) to release streaming. Exits 0.
  (4) on ~/resume -> gello_ur_bridge begins streaming /gello/joint_states onto
                  /forward_position_controller/commands (publish_rate_hz, 250 Hz
                  in ur7e_gello.yaml). Resume re-seeds from the robot's ACTUAL
                  /joint_states and soft-starts (soft_start_s ramp of the slew
                  clamp), so the residual gap is eased in, not snapped. The bridge
                  is NOT (re)started on handshake exit — it already exists, paused;
                  the resume call (not a launch timer) releases it strictly after
                  the controller switch, so it never streams into an inactive
                  controller.
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
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
        robot_ip:=127.0.0.1 use_fake_hardware:=true   # MOCK validation, no real robot/gripper
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
        robot_ip:=192.168.10.11 control_mode:=eef     # EEF delta-pose control

control_mode:=eef layers config/ur7e_gello_eef.yaml on top of params_file for the
gello_ur_bridge node ONLY (move-to-start, publisher, and gripper nodes are
unaffected). control_mode:=joint (default) is byte-for-byte the existing
behaviour -- no eef overlay is loaded and the bridge lazy-imports no eef
dependencies.

With use_fake_hardware:=true AND a real physical GELLO leader attached, the
arm-side handshake/bridge pipeline (move-to-start convergence gate, controller
switching, streaming) IS now verifiable here against ros2_control mock hardware
— no real robot required, and headless_mode is forced true automatically. Only
the Robotiq gripper still requires real hardware (Modbus tool bus needs a
powered robot) and is auto-skipped in that mode. ur7e_gello_rviz.launch.py
(source:=fake) remains the right choice when there is no physical GELLO either
(fully synthetic).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
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
            "use_fake_hardware",
            default_value="false",
            description=(
                "PRE-REAL-HARDWARE VALIDATION SWITCH. Default 'false' -> real "
                "UR7e (real-hardware behaviour is COMPLETELY unaffected unless a "
                "caller explicitly passes this arg). When 'true', the underlying "
                "ur_control.launch.py is switched to ros2_control mock/fake "
                "hardware so the FULL move-to-start convergence-gate handshake "
                "plus the bridge pre-spawn/resume sequence can be exercised "
                "end-to-end against a real physical GELLO leader WITHOUT any real "
                "robot attached (safe, undamageable validation). In this mode "
                "headless_mode is forced true (mock hardware has no pendant / "
                "External Control Play to wait on) and the Robotiq 2F-85 gripper "
                "is automatically SKIPPED because its Modbus bridge needs a "
                "powered real robot to supply 24V tool voltage."
            ),
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
                "arm CHASES the live leader (gap-sized catch-ups) and streams only "
                "once caught up within chase_tol, sustained chase_dwell_s. "
                "'init_align': the arm moves to a fixed init_pose (from the params "
                "file), then waits for you to align the GELLO leader to it before "
                "streaming (safer first motion)."
            ),
        ),
        DeclareLaunchArgument(
            "control_mode",
            default_value="joint",
            choices=["joint", "eef"],
            description=(
                "gello_ur_bridge control mode. 'joint' (default): existing "
                "per-joint passthrough, UNCHANGED behaviour -- the bridge does "
                "not import any eef (ur_kin/eef_delta) dependency in this mode. "
                "'eef': the bridge additionally loads "
                "config/ur7e_gello_eef.yaml (layered on top of params_file) and "
                "drives an end-effector delta-pose controller gated behind the "
                "eef_engage/eef_disengage/eef_reclutch/eef_to_joint services. "
                "Only the gello_ur_bridge node is affected; move-to-start, the "
                "GELLO publisher, and the gripper nodes are unchanged."
            ),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Substitutions
    # ------------------------------------------------------------------ #
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_rviz = LaunchConfiguration("launch_rviz")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")
    params_file = LaunchConfiguration("params_file")
    control_mode = LaunchConfiguration("control_mode")

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
            # Real hardware (default use_fake_hardware:=false): mock/fake both
            # OFF. Pass both spellings so the value is honoured across driver
            # versions (unknown launch arguments are silently ignored by the
            # include). When use_fake_hardware:=true both go ON, switching the
            # driver to ros2_control mock hardware for pre-real-HW validation.
            "use_mock_hardware": use_fake_hardware,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": launch_rviz,
            # Force headless_mode true under fake hardware (mock HW has no
            # pendant / External Control Play, so waiting for it would hang
            # forever); otherwise pass the user's headless_mode value through
            # unchanged. Equivalent to:
            #     'true' if use_fake_hardware == 'true' else headless_mode
            "headless_mode": PythonExpression(
                ["'true' if '", use_fake_hardware, "' == 'true' else '", headless_mode, "'"]
            ),
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
            # Disable tool communication (and therefore the gripper's Modbus
            # socat bridge) under fake hardware — there is no real robot to
            # power the tool or run the socat forwarder — while preserving
            # "true" unchanged for the real-hardware default. Equivalent to:
            #     'false' if use_fake_hardware == 'true' else 'true'
            "use_tool_communication": PythonExpression(
                ["'false' if '", use_fake_hardware, "' == 'true' else 'true'"]
            ),
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
        # resume_bridge:=True -> after a SUCCESSFUL STRICT switch, this node
        # calls the pre-spawned bridge's ~/resume service (with retries) to
        # release streaming. The resume is issued strictly AFTER the switch, so
        # the bridge never streams into an inactive controller; the bridge's own
        # ~/resume is alignment-gated + soft-started. Kept a launch override (not
        # in yaml) so standalone move_to_start runs do not try to poke a bridge.
        parameters=[
            params_file,
            {
                "start_mode": LaunchConfiguration("start_mode"),
                "resume_bridge": True,
            },
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (4) GELLO -> UR bridge. Streams /gello/joint_states onto
    #     /forward_position_controller/commands (publish_rate_hz, 250 Hz).
    #
    #     PRE-SPAWNED PAUSED: start_paused:=True is a LAUNCH-LEVEL override
    #     (NOT in the yaml, so gripper-only / standalone bridge runs keep the
    #     old unpaused behaviour). It comes up at t=6s alongside the publisher
    #     so its ~/resume service exists at handover, but publishes nothing
    #     until gello_move_to_start (resume_bridge:=True) calls ~/resume AFTER
    #     the STRICT switch. This closes the cold-start drift window that the
    #     old OnProcessExit start left open. On resume the bridge re-seeds from
    #     the actual /joint_states and soft-starts the slew clamp, so the
    #     residual gap eases in rather than snapping.
    # ------------------------------------------------------------------ #
    # control_mode:=eef layers config/ur7e_gello_eef.yaml on TOP of params_file
    # (later entries in the `parameters` list win on key conflicts). This is
    # ONLY consulted when control_mode=="eef" (IfCondition below); the
    # control_mode=="joint" node is built with params_file alone, so the joint
    # code path is completely untouched by the presence of this overlay.
    eef_overlay_file = PathJoinSubstitution(
        [
            FindPackageShare("ur_gello_bringup"),
            "config",
            "ur7e_gello_eef.yaml",
        ]
    )
    is_joint_mode = IfCondition(
        PythonExpression(["'", control_mode, "' == 'joint'"])
    )
    is_eef_mode = IfCondition(
        PythonExpression(["'", control_mode, "' == 'eef'"])
    )

    # Two conditioned Node actions (rather than branching Python logic) so the
    # control_mode=="joint" path is byte-for-byte the pre-existing behaviour:
    # same params_file, same overrides, no eef overlay anywhere near it.
    bridge_node_joint = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        # start_paused override kept here (not in yaml) so ONLY this integrated
        # launch starts the bridge held; other launches loading the same params
        # file are unaffected.
        parameters=[params_file, {"start_paused": True, "control_mode": control_mode}],
        output="screen",
        condition=is_joint_mode,
    )

    bridge_node_eef = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        parameters=[
            params_file,
            eef_overlay_file,
            {"start_paused": True, "control_mode": control_mode},
        ],
        output="screen",
        condition=is_eef_mode,
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
    #   * gello_ur_bridge after 6s TOO, but PAUSED (start_paused:=True). It only
    #     brings up its ~/pause + ~/resume services and its subscriptions; it
    #     publishes nothing until resumed. Pre-spawning (vs. cold-starting it on
    #     handshake exit) removes the window in which the leader used to keep
    #     drifting, widening the handover gap.
    #   * move_to_start after 8s. It WAITS internally for
    #     scaled_joint_trajectory_controller to become active (i.e. for the
    #     External Control program to be PLAYING on the pendant), so a fixed
    #     delay is not timing-critical. On a SUCCESSFUL switch it calls the
    #     bridge's ~/resume itself (resume_bridge:=True) — so the bridge is
    #     released strictly AFTER the controller switch, never before.
    #   * Modbus gripper + gello_gripper_bridge are started ONLY when
    #     move_to_start EXITS SUCCESSFULLY (return code 0 == handshake done,
    #     forward_position_controller active + bridge resumed). The bridge is NO
    #     LONGER started here — it already exists (paused) and resumes itself.
    #     If the handshake FAILS the bridge simply stays paused and silent
    #     forever (fail-safe: no streaming from an unsafe state).
    # ------------------------------------------------------------------ #
    gello_publisher_delayed = TimerAction(
        period=6.0,
        actions=[gello_publisher_node],
    )
    bridge_paused_delayed = TimerAction(
        period=6.0,
        # Both entries are IfCondition-gated on control_mode; exactly one
        # spawns at runtime (the other's condition evaluates false and it is
        # never launched).
        actions=[bridge_node_joint, bridge_node_eef],
    )
    move_to_start_delayed = TimerAction(
        period=8.0,
        actions=[move_to_start_node],
    )

    def _on_handshake_exit(event, context):
        # Resolve whether ros2_control mock/fake hardware is in use. Under fake
        # hardware there is no real powered robot behind the Modbus tool bus, so
        # the Robotiq gripper nodes must be skipped (see below).
        fake_hardware = use_fake_hardware.perform(context) == "true"
        if event.returncode == 0:
            if fake_hardware:
                return [
                    LogInfo(
                        msg=(
                            "Move-to-start handshake SUCCEEDED "
                            "(forward_position_controller active; bridge resumed "
                            "via ~/resume). use_fake_hardware:=true -- skipping "
                            "the Robotiq gripper (Modbus needs a real powered "
                            "robot); arm handshake validation complete."
                        )
                    ),
                ]
            return [
                LogInfo(
                    msg=(
                        "Move-to-start handshake SUCCEEDED "
                        "(forward_position_controller active; bridge resumed via "
                        "~/resume). Starting Modbus gripper (shared socat bridge "
                        "/tmp/ttyUR) + gello_gripper_bridge. The arm bridge was "
                        "pre-spawned paused and has already been released."
                    )
                ),
                gripper_modbus_node,
                gello_gripper_bridge_node,
            ]
        return [
            LogInfo(
                msg=(
                    "Move-to-start handshake FAILED (exit "
                    f"{event.returncode}); the pre-spawned bridge stays PAUSED "
                    "and silent (fail-safe) and the grippers are NOT started. "
                    "The robot will stay put. Check that the External Control "
                    "program is PLAYING on the pendant, then re-launch. Manual "
                    "recovery once aligned: re-align the leader and call "
                    "'ros2 service call /gello_ur_bridge/resume "
                    "std_srvs/srv/Trigger' after switching to "
                    "forward_position_controller."
                )
            )
        ]

    grippers_after_handshake = RegisterEventHandler(
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
            bridge_paused_delayed,
            move_to_start_delayed,
            grippers_after_handshake,
        ]
    )
