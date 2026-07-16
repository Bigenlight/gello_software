#!/usr/bin/env python3
r"""Bring up a REAL UR7e and drive it AUTONOMOUSLY from the trained Diffusion policy.

    ############################################################################
    #  READ THIS BEFORE RUNNING ON REAL HARDWARE                               #
    #                                                                          #
    #  THE DIFFUSION POLICY DRIVES THE ARM AUTONOMOUSLY.                       #
    #    This launch REPLACES the physical GELLO leader with policy_leader_node #
    #    (package gello_policy), a SYNTHETIC leader. Instead of reading a human #
    #    GELLO arm it queries the py3.12 Diffusion inference server over        #
    #    localhost ZMQ and publishes the policy's joint targets onto            #
    #    /gello/joint_states (+ the gripper). Everything downstream             #
    #    (gello_ur_bridge handshake, move-to-start, Robotiq Modbus) is          #
    #    UNCHANGED and unaware the leader is a policy, not a person.             #
    #                                                                          #
    #  HANDSHAKE CHASES THE POLICY'S HELD START POSE:                          #
    #    At boot policy_leader_node is in HOLD: every tick it publishes its     #
    #    fixed start_pose (from diffusion_deploy.yaml) perfectly still and does #
    #    NOT query the server. gello_move_to_start therefore chases that        #
    #    constant start pose and, once converged (chase_tol, sustained          #
    #    chase_dwell_s), STRICT-switches to forward_position_controller and     #
    #    resumes the bridge exactly as in the human-GELLO launch. NO autonomous #
    #    motion happens during the handshake — the leader is holding still.     #
    #                                                                          #
    #  THE OPERATOR MUST EXPLICITLY START EXECUTION:                           #
    #    Autonomous Diffusion motion begins ONLY when the operator calls        #
    #        ros2 service call /policy_leader_node/start_execution \            #
    #            std_srvs/srv/Trigger                                           #
    #    (unless auto_start_on_stream:=true in the params file). On start the   #
    #    leader RESETs the server and enters EXECUTE: each tick it queries the  #
    #    Diffusion server, SAFETY-CLAMPS the returned target (1.2x joint        #
    #    envelope, then per-joint max-deviation from the live pose) and         #
    #    publishes it. On any ZMQ timeout / server error it FAULTS: it STOPS    #
    #    publishing, so the bridge's staleness watchdog halts the arm           #
    #    (fail-silent — never a stale held target through an outage). Recover   #
    #    with ~/start_execution again.                                          #
    #                                                                          #
    #  Keep the teach-pendant E-STOP within reach at all times. Ctrl-C on the   #
    #  run script tears down BOTH the Diffusion server and this launch cleanly. #
    ############################################################################

Bring-up sequence (staggered TimerActions) — identical topology to
ur7e_gello_real.launch.py, with the physical GELLO source swapped for the
Diffusion synthetic leader:

  (1) t=0s    ur_control.launch.py -> UR7e driver + ros2_control.
                  * scaled_joint_trajectory_controller ACTIVE (handshake).
                  * forward_position_controller loaded INACTIVE.
                  * use_tool_communication:=true / tool_voltage:=24 /
                    tool_device_name:=/tmp/ttyUR -> the DRIVER powers the tool
                    and owns robot_ip:54321 as the serial device /tmp/ttyUR
                    (the Robotiq node SHARES it).
  (2) t=6s    policy_leader_node (SYNTHETIC DIFFUSION LEADER) -> publishes
                  /gello/joint_states + /robotiq_gripper/command_percent. In
                  HOLD it emits its fixed start_pose so a still handshake target
                  exists. (Replaces gello_publisher.)
      t=6s    gello_ur_bridge PRE-SPAWNED PAUSED (start_paused:=true) — brings up
                  ~/pause + ~/resume, publishes nothing until resumed.
  (3) t=8s    gello_move_to_start -> MANDATORY handshake. Chases the HELD start
                  pose, STRICT-switches to forward_position_controller, then
                  (resume_bridge:=true) calls the bridge ~/resume. Exits 0.
  (4) on ~/resume -> gello_ur_bridge streams /gello/joint_states onto
                  /forward_position_controller/commands (250 Hz, soft-started).
  (5) on handshake exit 0 -> the Robotiq 2F-85 Modbus node starts (shared
                  /tmp/ttyUR socat bridge). policy_leader_node publishes the
                  gripper command DIRECTLY (0=open..1=closed), so
                  gello_gripper_bridge is NOT started (no dual writers).

Autonomous motion is still GATED behind the operator's ~/start_execution call
(see the banner above); the handshake alone only parks the arm on the start pose.

Prerequisites:
    * The py3.12 Diffusion inference server must be reachable at act_host:act_port
      (start it first — run_ur7e_diffusion_real.sh launches it automatically
      before this launch, or run scripts/run_diffusion_server.sh standalone).
    * A trained checkpoint (Bigenlight/diffusion_banana_in_pot_joint; download via
      scripts/download_diffusion_checkpoint.sh).

Example:
    ros2 launch gello_policy ur7e_diffusion_real.launch.py robot_ip:=192.168.10.11
    ros2 launch gello_policy ur7e_diffusion_real.launch.py \
        robot_ip:=192.168.10.11 act_port:=5592 \
        checkpoint_path:=/path/to/pretrained_model
    ros2 launch gello_policy ur7e_diffusion_real.launch.py \
        robot_ip:=127.0.0.1 use_fake_hardware:=true   # MOCK arm validation
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
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
            # be supplied explicitly, e.g. robot_ip:=192.168.10.11.
            description="REQUIRED. IP address of the physical UR7e controller.",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description=(
                "PRE-REAL-HARDWARE VALIDATION SWITCH. Default 'false' -> real "
                "UR7e. When 'true', ur_control.launch.py is switched to "
                "ros2_control mock/fake hardware so the move-to-start handshake + "
                "bridge pre-spawn/resume sequence can be exercised end-to-end "
                "WITHOUT a real robot. In this mode headless_mode is forced true "
                "and the Robotiq 2F-85 gripper is SKIPPED (its Modbus bridge needs "
                "a powered real robot). This validates the arm bring-up + handshake "
                "only: full autonomous Diffusion execution ADDITIONALLY needs the "
                "two RealSense cameras running AND the gripper position topic faked "
                "(e.g. `ros2 topic pub /robotiq_gripper/position_percent "
                "std_msgs/Float32 \"{data: 0.0}\" -r 10`); otherwise the leader "
                "FAULTs on missing observations (obs-freshness watchdog) instead "
                "of moving, because the skipped gripper leaves "
                "/robotiq_gripper/position_percent unpublished."
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
            # ur_description. Override with the ur_calibration output for a
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
                "recommended to override with the per-robot ur_calibration output."
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            # Default to THIS package's Diffusion deploy params (drives
            # policy_leader_node, gello_ur_bridge, gello_move_to_start and
            # robotiq_gripper from one file).
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("gello_policy"),
                    "config",
                    "diffusion_deploy.yaml",
                ]
            ),
            description=(
                "YAML parameter file for the Diffusion leader / bridge / "
                "move-to-start / gripper nodes. Defaults to "
                "gello_policy/config/diffusion_deploy.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "start_mode",
            default_value="gello",
            description=(
                "Handshake style for gello_move_to_start. 'gello' (default): the "
                "arm CHASES the leader's HELD start pose and streams once caught up "
                "within chase_tol, sustained chase_dwell_s. 'init_align': the arm "
                "moves to a fixed init_pose (from the params file) first."
            ),
        ),
        DeclareLaunchArgument(
            "act_host",
            default_value="127.0.0.1",
            description=(
                "Host of the py3.12 Diffusion inference server (ZMQ REQ/REP). "
                "Overrides policy_leader_node.act_host in the params file."
            ),
        ),
        DeclareLaunchArgument(
            "act_port",
            default_value="5592",
            description=(
                "TCP port of the Diffusion inference server. Overrides "
                "policy_leader_node.act_port in the params file. "
                "run_ur7e_diffusion_real.sh passes the same port it started the "
                "server on."
            ),
        ),
        DeclareLaunchArgument(
            "inference_transport",
            default_value="zmq",
            description="Inference transport: legacy local 'zmq' or remote 'grpc'.",
        ),
        DeclareLaunchArgument(
            "grpc_port",
            default_value="50051",
            description="Remote Diffusion gRPC port when inference_transport:=grpc.",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value="",
            description=(
                "LOGGING ONLY. The Diffusion checkpoint the server was started "
                "with; this launch does NOT load the checkpoint (the separate "
                "py3.12 server does). Passed only so the effective checkpoint is "
                "echoed in the launch log for provenance."
            ),
        ),
        DeclareLaunchArgument(
            "start_pose",
            default_value="",
            description=(
                "OPTIONAL / LOGGING. The authoritative held start pose is "
                "policy_leader_node.start_pose in the params file (a float list, "
                "which launch cannot reliably coerce from a CLI string). To change "
                "the held pose, edit diffusion_deploy.yaml. If set here it is only "
                "echoed in the launch log so the intended pose is visible."
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
    act_host = LaunchConfiguration("act_host")
    act_port = LaunchConfiguration("act_port")
    inference_transport = LaunchConfiguration("inference_transport")
    grpc_port = LaunchConfiguration("grpc_port")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    start_pose = LaunchConfiguration("start_pose")

    # ------------------------------------------------------------------ #
    # (1) Official UR driver bring-up (REAL hardware, ros2_control).
    #     Identical to ur7e_gello_real.launch.py — the Diffusion deploy reuses the
    #     arm stack unchanged (trajectory controller ACTIVE for the handshake;
    #     forward_position_controller INACTIVE; tool communication for the 2F-85).
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
            "use_mock_hardware": use_fake_hardware,
            "use_fake_hardware": use_fake_hardware,
            "launch_rviz": launch_rviz,
            # Force headless_mode true under fake hardware (mock HW has no pendant /
            # External Control Play); otherwise pass the user's value through.
            "headless_mode": PythonExpression(
                ["'true' if '", use_fake_hardware, "' == 'true' else '", headless_mode, "'"]
            ),
            # Bring up on the trajectory controller (ACTIVE) so the handshake can
            # command a smooth catch-up; forward_position_controller INACTIVE.
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
            "kinematics_params_file": kinematics_params_file,
            # Tool communication for the Robotiq 2F-85 gripper (driver owns the
            # tool bus; the gripper node shares /tmp/ttyUR). Disabled under fake
            # hardware (no real robot to power/forward the tool).
            "use_tool_communication": PythonExpression(
                ["'false' if '", use_fake_hardware, "' == 'true' else 'true'"]
            ),
            "tool_voltage": "24",
            "tool_device_name": "/tmp/ttyUR",
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # (2) SYNTHETIC DIFFUSION LEADER (this package). Replaces gello_publisher:
    #     instead of reading a physical GELLO arm it queries the py3.12 Diffusion
    #     server over ZMQ and publishes the policy's targets on /gello/joint_states
    #     (+ the gripper command DIRECTLY). In HOLD it publishes its fixed
    #     start_pose so the handshake below has a still target to chase; it only
    #     queries the server after the operator calls ~/start_execution (or
    #     auto_start_on_stream).
    #
    #     act_host / act_port are typed dict OVERRIDES on top of the params file so
    #     the run script can point the node at the port it started the server on.
    #     start_pose / joint limits / clamps come from the params file.
    #
    #     This node is POLICY-AGNOSTIC and UNCHANGED from the ACT launch — the
    #     diffusion model emits the identical 7-D joint action contract.
    # ------------------------------------------------------------------ #
    policy_leader_node = Node(
        package="gello_policy",
        executable="policy_leader_node",
        parameters=[
            params_file,
            {
                "act_host": ParameterValue(act_host, value_type=str),
                "act_port": ParameterValue(act_port, value_type=int),
                "inference_transport": ParameterValue(inference_transport, value_type=str),
                "grpc_port": ParameterValue(grpc_port, value_type=int),
            },
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (3) MANDATORY move-to-start handshake. Chases the leader's HELD start pose,
    #     then STRICT-switches to forward_position_controller and (resume_bridge)
    #     resumes the pre-spawned bridge. Unchanged from the human-GELLO launch.
    # ------------------------------------------------------------------ #
    move_to_start_node = Node(
        package="ur_gello_bringup",
        executable="gello_move_to_start",
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
    # (4) GELLO -> UR bridge, PRE-SPAWNED PAUSED (start_paused:=True launch-level
    #     override). Streams /gello/joint_states onto
    #     /forward_position_controller/commands only after ~/resume. Unchanged.
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        package="ur_gello_bringup",
        executable="gello_ur_bridge",
        parameters=[params_file, {"start_paused": True}],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # (5) Robotiq 2F-85 gripper over Modbus RTU (SHARED socat bridge /tmp/ttyUR).
    #     The arm driver runs with use_tool_communication:=true / tool_voltage:=24
    #     / tool_device_name:=/tmp/ttyUR, so the DRIVER powers the tool and owns
    #     :54321; this node shares the exposed serial device. robot_ip injected.
    #     NOTE: gello_gripper_bridge is DROPPED for Diffusion — policy_leader_node
    #     publishes /robotiq_gripper/command_percent directly (single writer).
    # ------------------------------------------------------------------ #
    gripper_modbus_node = Node(
        package="ur_gello_bringup",
        executable="robotiq_gripper_modbus",
        parameters=[
            params_file,
            {"robot_ip": robot_ip, "serial_port": "/tmp/ttyUR"},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Start-up sequencing (same timings as the human-GELLO launch).
    # ------------------------------------------------------------------ #
    provenance_banner = LogInfo(
        msg=[
            "Diffusion deploy: policy_leader_node will query the Diffusion server at ",
            act_host, ":", act_port,
            " | checkpoint (server-side): ", checkpoint_path,
            " | start_pose override (yaml is authoritative): ", start_pose,
        ]
    )

    policy_leader_delayed = TimerAction(
        period=6.0,
        actions=[policy_leader_node],
    )
    bridge_paused_delayed = TimerAction(
        period=6.0,
        actions=[bridge_node],
    )
    move_to_start_delayed = TimerAction(
        period=8.0,
        actions=[move_to_start_node],
    )

    def _on_handshake_exit(event, context):
        # Under fake hardware there is no real powered robot behind the Modbus
        # tool bus, so the Robotiq gripper node is skipped.
        fake_hardware = use_fake_hardware.perform(context) == "true"
        if event.returncode == 0:
            if fake_hardware:
                return [
                    LogInfo(
                        msg=(
                            "Move-to-start handshake SUCCEEDED "
                            "(forward_position_controller active; bridge resumed "
                            "via ~/resume). use_fake_hardware:=true -- skipping the "
                            "Robotiq gripper (Modbus needs a real powered robot). "
                            "The arm/handshake are now validated and the arm is "
                            "parked on the held start pose. NOTE: autonomous "
                            "Diffusion execution additionally requires the two "
                            "RealSense cameras running AND the gripper position "
                            "topic faked (the gripper node was skipped, so "
                            "/robotiq_gripper/position_percent has no publisher), "
                            "e.g.:  ros2 topic pub /robotiq_gripper/position_percent "
                            "std_msgs/Float32 \"{data: 0.0}\" -r 10 . Without those, "
                            "~/start_execution will FAULT on missing observations "
                            "rather than move the (mock) arm."
                        )
                    ),
                ]
            return [
                LogInfo(
                    msg=(
                        "Move-to-start handshake SUCCEEDED "
                        "(forward_position_controller active; bridge resumed via "
                        "~/resume). Starting Modbus gripper (shared socat bridge "
                        "/tmp/ttyUR). The Diffusion leader publishes command_percent "
                        "directly, so gello_gripper_bridge is NOT started. The arm "
                        "is parked on the held start pose; call 'ros2 service call "
                        "/policy_leader_node/start_execution std_srvs/srv/Trigger' "
                        "to begin AUTONOMOUS Diffusion motion."
                    )
                ),
                gripper_modbus_node,
            ]
        return [
            LogInfo(
                msg=(
                    "Move-to-start handshake FAILED (exit "
                    f"{event.returncode}); the pre-spawned bridge stays PAUSED "
                    "and silent (fail-safe) and the gripper is NOT started. The "
                    "robot will stay put and the Diffusion leader stays in HOLD. "
                    "Check that the External Control program is PLAYING on the "
                    "pendant, then re-launch."
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
            provenance_banner,
            ur_control_launch,
            policy_leader_delayed,
            bridge_paused_delayed,
            move_to_start_delayed,
            grippers_after_handshake,
        ]
    )
