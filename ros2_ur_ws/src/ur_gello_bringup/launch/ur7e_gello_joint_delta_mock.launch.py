#!/usr/bin/env python3
"""Bring up a UR7e in RViz with mock/fake hardware, driven via the JOINT_DELTA
(start-anchored joint-space delta) bridge control mode. NO REAL ROBOT REQUIRED.

This is the turnkey validation entrypoint for `control_mode:=joint_delta` on
`gello_ur_bridge`. It mirrors `ur7e_gello_eef_mock.launch.py` (mock ros2_control
+ GELLO source + bridge + RViz) but forces the bridge into joint_delta mode and
layers `config/ur7e_gello_joint_delta.yaml` on top of `params_file`, so the full
engage/clutch/reclutch gate stack (G0-G7), the delta accumulator, the joint-limit
and excursion clamps, and the
JOINT_BOOTSTRAP -> ENGAGED -> CLUTCHED -> ENGAGED state machine can be exercised
end-to-end against safe, undamageable mock hardware:

  1. Includes the official ``ur_control.launch.py`` from ``ur_robot_driver``
     with ``ur_type:=ur7e`` and ``use_fake_hardware:=true``. The ros2_control
     mock hardware simply mirrors position commands back to the joint states,
     so any command sent to the ``forward_position_controller`` makes the
     RViz model move -- nothing here can be damaged.
  2. A few seconds later, starts our ``gello_ur_bridge`` node with
     ``control_mode:=joint_delta``, parameters layered as
     ``[params_file, ur7e_gello_joint_delta.yaml, {control_mode: joint_delta}]``
     so the joint_delta overlay always wins on key conflicts. The bridge boots
     in JOINT_BOOTSTRAP (ordinary absolute joint passthrough -- NOT the eef
     3D-pen HOLD) until ``~/joint_delta_engage`` is called. Drive the state
     machine with ``~/joint_delta_engage``, ``~/joint_delta_clutch``,
     ``~/joint_delta_reclutch``, ``~/joint_delta_disengage``,
     ``~/joint_delta_to_joint``, ``~/joint_delta_start`` (all std_srvs/Trigger)
     and watch ``~/joint_delta/state`` (std_msgs/String, JSON) for the gate
     failures and reject reasons (BAD_INPUT, ESTOP, LEADER_JUMP) plus the
     per-joint ``delta`` / ``limited`` / ``slewed`` / ``excursion_rad`` /
     ``jump_absorbed`` telemetry, without ever risking real hardware.
  3. Also starts the GELLO source:
       * ``source:=fake``  -> ``fake_gello`` (synthetic joint stream, no
         hardware). ``pattern`` is forwarded to fake_gello's ``pattern``
         parameter, which is the whole point of using this launch for
         joint_delta validation:
             pattern:=hold           -> a genuinely still leader; satisfies the
                                        leader-quasi-still gate G5 so an engage
                                        can actually be accepted.
             pattern:=step           -> repeated discrete leader jumps; exercises
                                        jd_leader_jump_max_rad (absorbed jumps)
                                        and the slew clamp.
             pattern:=full_rotation  -> wrist_3 through +2*pi with no cap; this
                                        is THE regression rehearsal for the
                                        accumulate-past-pi property (a naive
                                        wrap-against-the-anchor implementation
                                        would fold and drive the arm backwards).
       * ``source:=gello`` -> ``gello_publisher`` (reads the real, PASSIVE
         GELLO leader arm; still no real UR robot is involved).
  4. Optionally the Robotiq gripper node (harmless logging by default,
     ``connect_on_start:=false``) -- the gripper path is unchanged by
     control_mode and is included here only for structural parity.

By default this mock launch does NOT run the gello_move_to_start handshake (mock
hardware cannot be damaged by a first-command jump); that safety step is only
mandatory on real HW, in ur7e_gello_real.launch.py.

handshake:=true MIRRORS THE REAL JOINT_DELTA BRING-UP (no real robot needed):
    Pass ``handshake:=true`` to reproduce ur7e_gello_real.launch.py's
    control_mode:=joint_delta bring-up here. Like eef mode, that bring-up is now a
    NO-MOTION one: the driver comes up on ``scaled_joint_trajectory_controller``
    (so forward_position_controller is loaded INACTIVE), the bridge is pre-spawned
    PAUSED (with ``jd_start_allow_unstreamed:=True``), and ``gello_move_to_start``
    runs with ``start_mode:=switch_only`` (a STRICT controller switch IN PLACE --
    no trajectory is built or sent, the arm does not move) +
    ``bridge_resume_service:=/gello_ur_bridge/joint_delta_start`` (anchor at the
    arm's ACTUAL current pose and release into ENGAGED delta streaming, WITHOUT
    any chase). From the first tick the arm holds its own pose and thereafter
    moves ONLY by the leader's delta; with ``jd_gain:=0.0`` it does not move at
    all. No manual ``~/joint_delta_engage`` call is needed to start delta control;
    that service remains the manual RE-anchor for an already-streaming bridge.

    handshake:=false (the default) starts forward_position_controller active and
    the bridge unpaused, which is the fastest way to exercise the state machine.

SAFETY: GELLO is ALWAYS a passive, read-only leader. The driver never applies
torque to the GELLO Dynamixels; it only reads their joint angles.

Staged-bring-up override (jd_gain) is accepted here with exactly the same
semantics as in ur7e_gello_real.launch.py, so the mock can exercise the same code
path before it is used on the real arm: it defaults to the EMPTY STRING = "not
overridden" (config/ur7e_gello_joint_delta.yaml wins), and a supplied value is
coerced to a real Python float before it reaches the node.

Example:
    ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py
    ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py \\
        source:=fake pattern:=hold launch_rviz:=false
    ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py \\
        pattern:=hold jd_gain:=0.0            # mock rehearsal of the P1 stage
    ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py \\
        pattern:=full_rotation launch_rviz:=false   # accumulate-past-pi check
    ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py \\
        handshake:=true launch_rviz:=false pattern:=hold
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
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


# Staged-bring-up override. Same semantics and the same float-coercion handling
# as ur7e_gello_real.launch.py -- see the long comment in
# _eef_bridge_parameter_overrides() there for the full rationale. Short version:
# gello_ur_bridge declares jd_gain as a DOUBLE parameter, but launch_ros runs
# substitution-valued parameters through yaml.safe_load(), so the all-digit CLI
# value `jd_gain:=0` (the first staged bring-up stage!) would become an INTEGER
# and crash the node with a parameter-type mismatch. We therefore .perform() the
# LaunchConfiguration ourselves and apply an explicit float();
# normalize_parameter_dict() keeps a real Python float as-is, so the node always
# receives a DOUBLE.
_JD_DOUBLE_OVERRIDES = ("jd_gain",)


def _jd_double_overrides(context):
    """Return {name: float} for every staged-bring-up arg actually supplied.

    An empty value (the declared default) means NOT SUPPLIED -> the key is
    omitted -> the joint_delta overlay yaml value wins, unchanged.
    """
    overrides = {}
    for name in _JD_DOUBLE_OVERRIDES:
        raw = LaunchConfiguration(name).perform(context).strip()
        if not raw:
            continue
        try:
            overrides[name] = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Launch argument {name}:={raw!r} is not a number. It overrides "
                f"the '{name}' double parameter in "
                f"config/ur7e_gello_joint_delta.yaml; pass a numeric value "
                f"(e.g. {name}:=0.0) or omit the argument to keep the yaml value."
            ) from exc
    # Same launch-time range check as ur7e_gello_real.launch.py: an out-of-range
    # gain makes JointDeltaController's constructor raise, which would kill the
    # bridge silently mid-bring-up instead of failing the launch outright.
    gain = overrides.get("jd_gain")
    if gain is not None and not (0.0 <= gain <= 2.0):
        raise RuntimeError(
            f"Launch argument jd_gain:={gain} is out of range — it must be in "
            "[0.0, 2.0]. Nothing was started."
        )
    return overrides


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
                "(only consulted when source:=fake). For joint_delta the "
                "interesting values are 'hold' (still leader -> gate G5 can "
                "pass, so an engage is actually accepted), 'step' (discrete "
                "jumps -> exercises jd_leader_jump_max_rad and the slew clamp) "
                "and 'full_rotation' (wrist_3 through +2*pi -> the "
                "accumulate-past-pi regression rehearsal). See "
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
            "joint_delta_params_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("ur_gello_bringup"),
                    "config",
                    "ur7e_gello_joint_delta.yaml",
                ]
            ),
            description=(
                "JOINT_DELTA overlay YAML, layered on TOP of params_file for the "
                "bridge only (later entries in a Node's `parameters` list win on "
                "key conflicts)."
            ),
        ),
        DeclareLaunchArgument(
            "jd_gain",
            default_value="",
            description=(
                "Leader->robot joint gain (dimensionless), overriding jd_gain in "
                "the joint_delta overlay yaml. Empty (default) = use the yaml "
                "value. Staged bring-up: P1/P4 0.0 (the arm must not move at all "
                "after engage), then 0.25 / 0.5 / 1.0."
            ),
        ),
        # ---------------------------------------------------------------- #
        # Mirror of the real launch's joint_delta bring-up path (see module
        # docstring). Default 'false' = the fast state-machine rehearsal.
        # ---------------------------------------------------------------- #
        DeclareLaunchArgument(
            "handshake",
            default_value="false",
            choices=["true", "false"],
            description=(
                "If true, reproduce ur7e_gello_real.launch.py's "
                "control_mode:=joint_delta bring-up against mock hardware: bring "
                "the driver up on scaled_joint_trajectory_controller (so "
                "forward_position_controller is loaded INACTIVE), pre-spawn the "
                "bridge PAUSED (jd_start_allow_unstreamed:=True), and run "
                "gello_move_to_start with start_mode:=switch_only (a STRICT "
                "controller switch IN PLACE -- no trajectory, the arm does not "
                "move) + bridge_resume_service:=/gello_ur_bridge/joint_delta_start "
                "(anchor at the arm's ACTUAL current pose and release into ENGAGED "
                "delta streaming, no chase) -- i.e. the NO-MOTION bring-up, "
                "matching eef mode. Default 'false' = forward_position_controller "
                "active from the start, no handshake, bridge unpaused."
            ),
        ),
        DeclareLaunchArgument(
            "start_mode",
            # Empty == AUTO. joint_delta uses the SAME derived default as eef mode
            # ('switch_only'), NOT joint mode's 'gello' chase.
            default_value="",
            description=(
                "handshake:=true ONLY. Handshake style for gello_move_to_start. "
                "Empty (default) = AUTO -> 'switch_only' (the STRICT switch IN "
                "PLACE, no motion), which is what control_mode:=joint_delta uses "
                "on real hardware too. Explicit 'gello' / 'init_align' still win."
            ),
        ),
        DeclareLaunchArgument(
            "bridge_resume_service",
            default_value="",
            description=(
                "handshake:=true ONLY. Service gello_move_to_start calls after a "
                "successful STRICT switch. Empty (default) = AUTO -> "
                "'/gello_ur_bridge/joint_delta_start' (anchor at the arm's actual "
                "pose and release into ENGAGED delta streaming without any chase), "
                "matching the real joint_delta bring-up."
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
    joint_delta_params_file = LaunchConfiguration("joint_delta_params_file")
    handshake = LaunchConfiguration("handshake")

    is_handshake = IfCondition(handshake)

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
            # Default: we drive the arm with raw position commands, not
            # trajectories, so forward_position_controller is ACTIVE from the
            # start. With handshake:=true we instead mirror the real launch and
            # come up on scaled_joint_trajectory_controller -- which leaves
            # forward_position_controller loaded INACTIVE, so gello_move_to_start
            # has a genuine STRICT switch to perform.
            "initial_joint_controller": PythonExpression(
                [
                    "'scaled_joint_trajectory_controller' if '", handshake,
                    "' == 'true' else 'forward_position_controller'",
                ]
            ),
            "activate_joint_controller": "true",
            # headless_mode avoids needing the External Control URCap program
            # to be running (relevant under fake hardware / no pendant).
            "headless_mode": "true",
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # GELLO -> UR bridge, forced into control_mode:=joint_delta. The overlay
    # yaml is layered AFTER params_file so it wins on any key overlap; the
    # trailing dict override guarantees control_mode is "joint_delta"
    # regardless of what either yaml file says.
    # ------------------------------------------------------------------ #
    # Built inside an OpaqueFunction so jd_gain can be resolved to a real Python
    # float before the Node is constructed (see _jd_double_overrides above).
    def _make_bridge_node(context):
        overrides = {
            "control_mode": "joint_delta",
            **_jd_double_overrides(context),
        }
        # handshake:=true mirrors the real launch: the bridge is PRE-SPAWNED
        # PAUSED so its services exist at handover but it publishes nothing until
        # gello_move_to_start releases it AFTER the STRICT switch. Without the
        # handshake there is nothing to wait for, so it starts unpaused.
        if LaunchConfiguration("handshake").perform(context) == "true":
            overrides["start_paused"] = True
            # Mirror the real launch's no-motion bring-up: gello_move_to_start
            # does a STRICT switch IN PLACE (start_mode:=switch_only) and then
            # calls ~/joint_delta_start, which would otherwise refuse because the
            # bridge has never streamed. The switch-first ordering makes that gate
            # redundant, so opt in here exactly as the real launch does.
            overrides["jd_start_allow_unstreamed"] = True
        return [
            Node(
                package="ur_gello_bringup",
                executable="gello_ur_bridge",
                parameters=[
                    params_file,
                    joint_delta_params_file,
                    overrides,
                ],
                output="screen",
            )
        ]

    bridge_node = OpaqueFunction(function=_make_bridge_node)

    # ------------------------------------------------------------------ #
    # NO-MOTION handshake (handshake:=true only).
    #
    # Same node and parameter shape as ur7e_gello_real.launch.py's
    # move_to_start_node. control_mode is a constant "joint_delta" here, and its
    # AUTO-derived defaults mirror the real launch's joint_delta bring-up:
    # start_mode:=switch_only (STRICT switch IN PLACE, no trajectory, arm does not
    # move) and bridge_resume_service:=/gello_ur_bridge/joint_delta_start (anchor
    # at the arm's actual pose and release into ENGAGED delta streaming, no
    # chase). Because control_mode is a constant here the derivation collapses to
    # literals and no PythonExpression is needed.
    # ------------------------------------------------------------------ #
    def _make_move_to_start_node(context):
        start_mode = (
            LaunchConfiguration("start_mode").perform(context).strip()
            or "switch_only"
        )
        resume_service = (
            LaunchConfiguration("bridge_resume_service").perform(context).strip()
            or "/gello_ur_bridge/joint_delta_start"
        )
        return [
            Node(
                package="ur_gello_bringup",
                executable="gello_move_to_start",
                parameters=[
                    params_file,
                    {
                        "start_mode": start_mode,
                        "bridge_resume_service": resume_service,
                        "resume_bridge": True,
                    },
                ],
                output="screen",
            )
        ]

    move_to_start_node = OpaqueFunction(
        function=_make_move_to_start_node,
        condition=is_handshake,
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
    # Structural parity with the other mock launches; the joint_delta control
    # mode only affects the 6-DoF arm path, never the gripper (the gripper
    # mapping stays ABSOLUTE).
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

    # Handshake AFTER the bridge + GELLO source exist (mirrors the real launch's
    # 6s bridge / 8s move_to_start stagger). Inert when handshake:=false (the
    # OpaqueFunction's condition yields no actions).
    move_to_start_delayed = TimerAction(
        period=8.0,
        actions=[move_to_start_node],
    )

    return LaunchDescription(
        declared_arguments
        + [
            ur_control_launch,
            delayed_nodes,
            move_to_start_delayed,
        ]
    )
