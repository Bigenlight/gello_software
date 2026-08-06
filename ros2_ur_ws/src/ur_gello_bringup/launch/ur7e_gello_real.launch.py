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
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
        robot_ip:=192.168.10.11 control_mode:=eef \
        pos_scale:=0.0 v_max:=0.01 w_max:=0.05        # staged bring-up stage P6
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
        robot_ip:=192.168.10.11 gripper_mode:=discrete  # snap-to-endpoint gripper

DISCRETE GRIPPER MODE (gripper_mode / gripper_close_at / gripper_open_at):
    gripper_mode:=discrete folds the GELLO trigger to the two endpoints before
    it reaches /robotiq_gripper/command_percent: >= gripper_close_at -> exactly
    1.0 (CLOSED), <= gripper_open_at -> exactly 0.0 (OPEN), and in between the
    previous state is HELD (hysteresis -- a single threshold would chatter at
    the boundary). Default 'continuous' is today's raw passthrough, byte-for-
    byte unchanged.

    WHY: the trigger is spring + encoder and is already ~binary (90.95% of
    52,607 samples across 125 takes are below 0.02 or above 0.98), but roughly
    once or twice per session it RESTS PARTWAY OPEN -- worst measured 0.229 and
    0.243 -- and the bridge faithfully forwards that, so the Robotiq only opens
    ~76% and the operator sees a gripper that "didn't fully open". Not a
    calibration problem: all 125 takes still reach exactly 0.000. Full evidence
    and the per-date table: docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md §5.1.

    The two thresholds are optional and follow the same empty-string "not
    overridden" sentinel as the eef / joint_delta ones (the yaml wins when
    omitted), but they must be supplied TOGETHER and satisfy
    0.0 < open_at < close_at < 1.0, which is RANGE-CHECKED at launch time
    before anything spawns -- a threshold the trigger can no longer cross makes
    the gripper stop responding SILENTLY. The latched state is published on
    /gello_gripper_bridge/discrete_state and shown live in the EEF operator GUI
    (run_eef_gui.sh) so that failure is visible immediately.

STAGED EEF BRING-UP OVERRIDES (pos_scale / v_max / w_max):
    docs/ros2/GELLO_UR7E_EEF_MODE.md §3 prescribes a low-gain, gated first
    real-robot bring-up (P6: pos_scale=0.0, v_max=0.01, w_max=0.05;
    P7: v_max 0.02 -> 0.05; P9: v_max=0.08). Those three values used to live
    ONLY in config/ur7e_gello_eef.yaml, so stepping through the stages meant
    editing the yaml AND re-running `colcon build` every time (colcon COPIES
    config into install/, it does not symlink it). They are now real launch
    arguments.

    Semantics: each defaults to the EMPTY STRING = "not overridden", in which
    case the yaml value wins exactly as it does today (no duplicated default is
    hardcoded here that could silently drift from the yaml). Supplying a value
    overrides the yaml for the EEF bridge node ONLY; control_mode:=joint is
    completely unaffected (the joint bridge node never sees these).

control_mode:=eef layers config/ur7e_gello_eef.yaml on top of params_file for the
gello_ur_bridge node ONLY. control_mode:=joint (default) is byte-for-byte the
existing behaviour -- no eef overlay is loaded and the bridge lazy-imports no eef
dependencies.

control_mode:=joint_delta layers config/ur7e_gello_joint_delta.yaml the same way
(START-ANCHORED JOINT DELTA: the arm follows how much GELLO MOVED since an
anchor, not where GELLO is). Its BRING-UP IS NO-MOTION, exactly like eef mode
(see below): start_mode / bridge_resume_service auto-derive to 'switch_only' and
/gello_ur_bridge/joint_delta_start (NOT the joint-mode 'gello' chase). The driver
comes up on scaled_joint_trajectory_controller, gello_move_to_start performs a
STRICT controller switch to forward_position_controller IN PLACE (no trajectory
is built or sent, the arm does not move), and then calls
/gello_ur_bridge/joint_delta_start, which anchors at the arm's ACTUAL current
pose and releases the PAUSED bridge into ENGAGED delta streaming WITHOUT any
chase. From the first tick the arm holds its own pose and moves ONLY by the
leader's delta thereafter. The first published command equals the arm's actual
pose ALGEBRAICALLY (zero jump, not within a tolerance), so no protective stop can
occur. No manual ~/joint_delta_engage call is needed to start delta control;
~/joint_delta_engage remains the manual RE-anchor for an already-streaming
bridge. The joint_delta bridge is spawned with jd_start_allow_unstreamed:=True so
joint_delta_start accepts the never-streamed switch_only startup (the switch-
first ordering makes the ~/joint_delta_start _has_streamed gate redundant -- the
controller is already ACTIVE before the service is called, exactly as eef_resume
relies on with no such gate). Staged bring-up: control_mode:=joint_delta
jd_gain:=0.0 first -- with gain 0 the arm must not move at all.
    ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \\
        robot_ip:=192.168.10.11 control_mode:=joint_delta jd_gain:=0.0

EEF BRING-UP IS NO-MOTION (start_mode / bridge_resume_service auto-derive):
    control_mode:=eef ALSO changes how gello_move_to_start brings the arm up,
    because the joint-mode handshake is actively wrong there. In eef mode the
    bridge drives an end-effector DELTA and the delta math is pose-independent:
    only the leader's CHANGE since the anchor matters. The operator therefore
    holds GELLO as a free-floating "3D pen", permanently in a different joint
    configuration from the robot. The default start_mode:=gello would open
    bring-up by dragging the arm all the way to that unrelated leader joint pose
    (observed on the real UR7e: "Chasing live GELLO: gap 0.623 rad at pan ->
    1.25s catch-up") -- a large, unwanted motion that then has to be undone.

    So with control_mode:=eef the move-to-start node now defaults to
    start_mode:=switch_only (STRICT controller switch in place; NO trajectory is
    ever built or sent, the arm does not move) and bridge_resume_service
    defaults to /gello_ur_bridge/eef_resume (the bridge's re-arm WITHOUT the
    joint-alignment gate, which in eef mode would refuse forever while
    protecting nothing).

    Both remain plain defaults: start_mode:=gello / start_mode:=init_align and
    an explicit bridge_resume_service:=... still override them, so the old
    behaviour is one CLI argument away. control_mode:=joint is untouched
    (start_mode 'gello' + /gello_ur_bridge/resume, exactly as before).

    The arm HOLDING across a switch_only switch is not an assumption: the
    outgoing scaled_joint_trajectory_controller holds position from activation
    onward, so the position command interfaces already carry the current pose,
    and forward_position_controller (a ForwardCommandController) writes nothing
    until its first /forward_position_controller/commands message. Verified
    against ros2_control mock hardware.

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
    OpaqueFunction,
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


# --------------------------------------------------------------------------- #
# Staged EEF bring-up overrides (see module docstring): launch args that
# override config/ur7e_gello_eef.yaml for the EEF bridge node only.
#
# Every name here is declared on gello_ur_bridge with a FLOAT default
# (declare_parameter("v_max", 0.08) etc., see gello_ur_bridge_node.py), i.e.
# ROS type PARAMETER_DOUBLE.
# --------------------------------------------------------------------------- #
_EEF_DOUBLE_OVERRIDES = ("v_max", "w_max", "pos_scale")

# Staged JOINT_DELTA bring-up overrides: launch args that override
# config/ur7e_gello_joint_delta.yaml for the joint_delta bridge node only.
# Same DOUBLE-parameter / int-coercion trap as the eef ones above -- and jd_gain
# is EXACTLY the parameter an operator types as `jd_gain:=0` and `jd_gain:=1`
# while stepping through the staged bring-up, so it must go through the same
# .perform(context) + float() treatment. See _eef_bridge_parameter_overrides().
_JD_DOUBLE_OVERRIDES = ("jd_gain",)

# DISCRETE GRIPPER threshold overrides: {launch argument -> node parameter}.
#
# The names deliberately DIFFER on the two sides. The launch-argument namespace
# is flat and shared with the ARM bridge's overrides above, where a bare
# `close_at:=` / `open_at:=` would not say which node it lands on; the gripper_
# prefix does. On the node side the parameters live inside the
# gello_gripper_bridge namespace already, so they are named for what they do.
#
# Same DOUBLE-parameter treatment as the eef / joint_delta overrides: the value
# is resolved with .perform(context) and converted with an explicit float(), so
# it can never reach the node as an INTEGER. See
# _eef_bridge_parameter_overrides() for the full rationale.
_GRIPPER_DOUBLE_OVERRIDES = {
    "gripper_close_at": "discrete_close_at",
    "gripper_open_at": "discrete_open_at",
}


# --------------------------------------------------------------------------- #
# control_mode-derived handshake defaults.
#
# WHY: `control_mode:=joint` and `control_mode:=eef` want OPPOSITE bring-up
# behaviour, and until now both got the joint one.
#
#   joint -> the bridge is a per-joint passthrough, so the robot's joint pose
#            MUST match the leader's before streaming. start_mode:=gello chases
#            the live leader until it does. Unchanged, forever.
#   eef   -> the bridge drives an end-effector DELTA, and the delta math is
#            pose-independent: only the leader's *change* since the anchor
#            matters, never its absolute joint pose. The operator therefore uses
#            GELLO as a free-floating "3D pen", permanently in a different joint
#            configuration from the robot. Dragging the arm to the leader's joint
#            pose at bring-up (what start_mode:=gello does) is then not a safety
#            step at all — it is a large, unwanted autonomous motion that has to
#            be undone before eef teleop is usable. start_mode:=switch_only does
#            the STRICT controller switch in place, moving nothing.
#            For the same reason the resume must NOT go through the bridge's
#            joint-alignment-gated ~/resume (it would refuse forever while
#            protecting nothing) but through ~/eef_resume, which re-arms without
#            that gate.
#   joint_delta -> START-ANCHORED joint delta: the arm follows the leader's
#            *change* since an anchor, and the operator wants the robot to STAY AT
#            ITS OWN CURRENT POSE at bring-up (never chase to the leader). This is
#            the same no-motion requirement as eef, so joint_delta takes the same
#            switch_only path, but through the joint_delta-specific re-arm service
#            ~/joint_delta_start (which anchors at the arm's ACTUAL pose and
#            releases the paused bridge into ENGAGED delta streaming without any
#            chase) rather than ~/eef_resume.
#
# Both are DEFAULTS ONLY: the start_mode / bridge_resume_service launch
# arguments still win when supplied, so forcing the old behaviour in eef or
# joint_delta mode (start_mode:=gello) remains possible.
# --------------------------------------------------------------------------- #
def _auto_start_mode(control_mode):
    """start_mode launch arg, or the control_mode-derived default when empty."""
    return PythonExpression(
        [
            "'", LaunchConfiguration("start_mode"), "'.strip() or ",
            "('switch_only' if '", control_mode, "' in ('eef', 'joint_delta') else 'gello')",
        ]
    )


def _auto_resume_service(control_mode):
    """bridge_resume_service launch arg, or the control_mode-derived default."""
    return PythonExpression(
        [
            "'", LaunchConfiguration("bridge_resume_service"), "'.strip() or ",
            "('/gello_ur_bridge/eef_resume' if '", control_mode, "' == 'eef' ",
            "else '/gello_ur_bridge/joint_delta_start' if '", control_mode,
            "' == 'joint_delta' else '/gello_ur_bridge/resume')",
        ]
    )


def _eef_bridge_parameter_overrides(context):
    """Build the EEF bridge's launch-level parameter dict.

    Returns the dict of parameter overrides that is layered LAST (after
    params_file and the eef overlay yaml) on the eef bridge node.

    ############################################################################
    # WHY THIS IS AN OpaqueFunction AND NOT `{"v_max": LaunchConfiguration(..)}`
    #
    # Known repo-wide gotcha (already bit us once as the tick_budget_us
    # param-type mismatch in 2905925): launch_ros does NOT pass substitution-
    # valued parameters through as strings. In
    #   launch_ros/utilities/evaluate_parameters.py::evaluate_parameter_dict()
    # a substitution value is performed to a string and then run through
    # `yaml.safe_load()`, and whatever Python type that yields becomes the ROS
    # parameter type. So an operator typing
    #       v_max:=1        -> yaml.safe_load("1")   -> int   -> INTEGER
    #       pos_scale:=0    -> yaml.safe_load("0")   -> int   -> INTEGER
    # would hand gello_ur_bridge an INTEGER for a parameter it declared as a
    # DOUBLE -> rclpy InvalidParameterTypeException at startup -> the bridge
    # dies mid-bring-up on the real robot. `v_max:=1.0` would work and
    # `v_max:=1` would not, which is exactly the trap to remove.
    #
    # Fix: resolve the LaunchConfiguration to a string OURSELVES here
    # (`.perform(context)`) and convert with an explicit Python `float()`. The
    # dict we return therefore holds real Python floats, and
    #   launch_ros/utilities/normalize_parameters.py::normalize_parameter_dict()
    # keeps `(float, bool, int)` values "as is" -- no yaml re-parsing, no type
    # inference. `v_max:=1`, `v_max:=1.0`, `pos_scale:=0` and `pos_scale:=0.0`
    # all arrive as PARAMETER_DOUBLE.
    #
    # Empty string (the declared default) means NOT SUPPLIED: the key is simply
    # omitted from the dict, so the yaml value wins exactly as it does today.
    # No yaml default is duplicated in this file.
    ############################################################################
    """
    overrides = {
        # Pre-existing launch-level overrides, unchanged.
        "start_paused": True,
        "control_mode": LaunchConfiguration("control_mode").perform(context),
    }
    for name in _EEF_DOUBLE_OVERRIDES:
        raw = LaunchConfiguration(name).perform(context).strip()
        if not raw:
            # Not supplied -> do NOT set the parameter -> yaml value wins.
            continue
        try:
            overrides[name] = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Launch argument {name}:={raw!r} is not a number. It overrides "
                f"the '{name}' double parameter in config/ur7e_gello_eef.yaml; "
                f"pass a numeric value (e.g. {name}:=0.01) or omit the argument "
                f"to keep the yaml value."
            ) from exc
    return overrides


def _joint_delta_bridge_parameter_overrides(context):
    """Build the JOINT_DELTA bridge's launch-level parameter dict.

    Structural copy of _eef_bridge_parameter_overrides() above, against
    _JD_DOUBLE_OVERRIDES and config/ur7e_gello_joint_delta.yaml. The whole
    int-coercion rationale documented there applies VERBATIM here: `jd_gain:=0`
    (the "the robot must not move at all" stage of the staged bring-up) would
    otherwise reach the node as an INTEGER against a parameter declared DOUBLE
    and kill the bridge mid-bring-up. Resolving the LaunchConfiguration
    ourselves and applying an explicit float() is what makes `jd_gain:=0`,
    `jd_gain:=0.0`, `jd_gain:=1` and `jd_gain:=1.0` all arrive as
    PARAMETER_DOUBLE.

    Empty string (the declared default) means NOT SUPPLIED: the key is omitted
    so the yaml value wins, and no yaml default is duplicated in this file.
    """
    # The _has_streamed gate opt-in is scoped to the SWITCH_ONLY no-motion
    # bring-up ONLY. In that path gello_move_to_start does a STRICT switch IN
    # PLACE and then calls ~/joint_delta_start; that switch-first ordering makes
    # the gate redundant (the "controller must already be active" invariant the
    # _has_streamed latch proxies is guaranteed structurally, identical to how
    # ~/eef_resume ships with no such gate). But start_mode is a supported
    # override: an operator can force start_mode:=gello to restore the old chase,
    # and in that path the chase runs BEFORE the switch, so the gate must stay
    # live to refuse a mis-sequenced manual ~/joint_delta_start streaming into
    # the still-inactive controller. We therefore derive the opt-in from the
    # SAME resolved start_mode the move_to_start node will use, not from the
    # control_mode alone. Default (empty start_mode) resolves to switch_only, so
    # the ordinary no-motion bring-up keeps the opt-in.
    resolved_start_mode = (
        LaunchConfiguration("start_mode").perform(context).strip() or "switch_only"
    )
    overrides = {
        "start_paused": True,
        "control_mode": LaunchConfiguration("control_mode").perform(context),
        # True only for the switch_only no-motion path (see rationale above).
        # This override dict is built ONLY on the is_joint_delta_mode branch, so
        # the joint and eef bridges never receive this parameter at all.
        "jd_start_allow_unstreamed": (resolved_start_mode == "switch_only"),
    }
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
    # RANGE-CHECK HERE, AT LAUNCH TIME, not at node construction. JointDeltaController
    # raises ValueError for a gain outside [0.0, 2.0] — correct for the module, but if
    # that raise happens inside GelloUrBridge.__init__ the bridge simply never appears,
    # and gello_move_to_start's resume then fires at t=8s against a service that does
    # not exist: a confusing half-started bring-up, which is the exact failure class the
    # int-coercion comment above was written to prevent. Failing HERE aborts the launch
    # before anything spawns, with a message that names the offending argument.
    gain = overrides.get("jd_gain")
    if gain is not None and not (0.0 <= gain <= 2.0):
        raise RuntimeError(
            f"Launch argument jd_gain:={gain} is out of range — it must be in "
            "[0.0, 2.0] (0.0 = the robot never moves, the first stage of the "
            "staged bring-up; 1.0 = 1:1 with the leader's travel). Nothing was "
            "started."
        )
    return overrides


def _gripper_bridge_parameter_overrides(context):
    """Build the GELLO->gripper bridge's launch-level parameter dict.

    Maps `gripper_mode:=continuous|discrete` onto the bridge's `discrete_mode`
    bool, and the optional `gripper_close_at:=` / `gripper_open_at:=` onto its
    `discrete_close_at` / `discrete_open_at` doubles.

    DISCRETE MODE, in one line: the leader trigger is folded to the two
    endpoints -- trigger >= discrete_close_at -> exactly 1.0 (CLOSED),
    trigger <= discrete_open_at -> exactly 0.0 (OPEN), in between the previous
    state is HELD (hysteresis; a single threshold chatters at the boundary).
    WHY it exists: the spring+encoder trigger is already ~binary, but roughly
    once or twice per session it RESTS PARTWAY OPEN (worst measured 0.229 and
    0.243) and the bridge faithfully forwards that, so the Robotiq only opens
    ~76%. Evidence: docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md §5.1.

    discrete_mode is set on EVERY launch, unlike the eef / joint_delta doubles
    above: `gripper_mode` has a REAL default ("continuous") rather than the
    empty "not overridden" sentinel, exactly like `control_mode` on the arm
    bridge, so the launch always pins it. With no new arguments that pins the
    value the yaml already carries -> today's behaviour, unchanged.

    The two THRESHOLDS keep the empty-string sentinel: omitted -> key absent ->
    the yaml value wins, and no yaml number is duplicated here where it could
    silently drift.
    """
    mode = LaunchConfiguration("gripper_mode").perform(context).strip()
    if mode not in ("continuous", "discrete"):
        # DeclareLaunchArgument(choices=...) already refuses anything else at
        # launch; this keeps the helper honest when called on a bare context.
        raise RuntimeError(
            f"Launch argument gripper_mode:={mode!r} is not one of "
            "'continuous' / 'discrete'."
        )
    overrides = {"discrete_mode": mode == "discrete"}

    for arg_name, param_name in _GRIPPER_DOUBLE_OVERRIDES.items():
        raw = LaunchConfiguration(arg_name).perform(context).strip()
        if not raw:
            # Not supplied -> do NOT set the parameter -> yaml value wins.
            continue
        try:
            overrides[param_name] = float(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Launch argument {arg_name}:={raw!r} is not a number. It "
                f"overrides the '{param_name}' double parameter in the "
                "gello_gripper_bridge block of config/ur7e_gello.yaml; pass a "
                f"numeric value (e.g. {arg_name}:=0.3) or omit the argument to "
                "keep the yaml value."
            ) from exc

    # RANGE-CHECK HERE, AT LAUNCH TIME. A mis-set threshold does not crash
    # anything -- it makes the gripper SILENTLY STOP RESPONDING (the trigger can
    # no longer cross it), which is the worst failure mode this feature can have.
    # Failing here aborts the launch before anything spawns.
    close_at = overrides.get("discrete_close_at")
    open_at = overrides.get("discrete_open_at")
    if (close_at is None) != (open_at is None):
        # BOTH OR NEITHER. The invariant is a RELATION between the two, so half
        # an override cannot be checked without duplicating the yaml number for
        # the other half -- which this file deliberately never does (a
        # duplicated default silently drifts). Refusing keeps the invariant
        # checkable, and the fix is one extra argument.
        raise RuntimeError(
            "Launch arguments gripper_close_at / gripper_open_at must be given "
            "TOGETHER or not at all (got "
            f"gripper_close_at={close_at}, gripper_open_at={open_at}). They "
            "define ONE window and the invariant 0.0 < open_at < close_at < 1.0 "
            "is a relation between them, so a half-override cannot be "
            "range-checked. Pass both, or neither and keep the yaml values."
        )
    if close_at is not None and not (0.0 < open_at < close_at < 1.0):
        raise RuntimeError(
            f"Launch arguments gripper_open_at:={open_at} / "
            f"gripper_close_at:={close_at} are out of range -- they must "
            "satisfy 0.0 < open_at < close_at < 1.0 (defaults 0.3 / 0.7, see "
            "docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md §5.1). A threshold the "
            "trigger cannot cross makes the gripper stop responding silently. "
            "Nothing was started."
        )
    return overrides


def _validate_gripper_arguments(context):
    """Resolve + range-check the gripper arguments AT LAUNCH TIME, at t=0.

    The gripper bridge is only built when the move-to-start handshake EXITS
    (OnProcessExit) -- i.e. after the arm has already moved. Validating there
    would report a typo'd threshold minutes late, with the robot mid-session.
    This OpaqueFunction is visited immediately, before the driver include, so a
    bad argument aborts the launch before anything spawns. It adds no actions.
    """
    _gripper_bridge_parameter_overrides(context)
    return []


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
            # Empty == AUTO: derived from control_mode (joint -> 'gello',
            # eef -> 'switch_only'). See _START_MODE_FOR_CONTROL_MODE below.
            # Any explicit value still wins, so an operator can force the old
            # chase behaviour in eef mode with start_mode:=gello.
            default_value="",
            description=(
                "Handshake style for gello_move_to_start. Empty (default) = AUTO: "
                "control_mode:=joint -> 'gello', control_mode:=eef -> "
                "'switch_only'. Explicit values: 'gello' -- the arm CHASES the "
                "live leader (gap-sized catch-ups) and streams only once caught up "
                "within chase_tol, sustained chase_dwell_s. 'init_align' -- the arm "
                "moves to a fixed init_pose (from the params file), then waits for "
                "you to align the GELLO leader to it before streaming (safer first "
                "motion). 'switch_only' -- the arm NEVER MOVES during bring-up: no "
                "trajectory is built or sent, the STRICT controller switch happens "
                "in place and the bridge is resumed immediately. Intended for "
                "control_mode:=eef, where the leader is used as a free-floating '3D "
                "pen' and is DELIBERATELY in a different joint configuration from "
                "the robot, so joint alignment is meaningless (the eef delta math "
                "is pose-independent) and both moving modes would only produce an "
                "unwanted large motion."
            ),
        ),
        DeclareLaunchArgument(
            "bridge_resume_service",
            # Empty == AUTO: derived from control_mode (joint -> ~/resume,
            # eef -> ~/eef_resume). Escape hatch: point it back at
            # /gello_ur_bridge/resume if the eef_resume service is unavailable.
            default_value="",
            description=(
                "Service gello_move_to_start calls after a successful STRICT "
                "switch to release the pre-spawned paused bridge. Empty (default) "
                "= AUTO: control_mode:=joint -> '/gello_ur_bridge/resume' (the "
                "joint-alignment-gated re-arm), control_mode:=eef -> "
                "'/gello_ur_bridge/eef_resume' (re-arms WITHOUT that gate, which in "
                "eef mode would refuse forever while protecting nothing). Override "
                "explicitly to pin one of them."
            ),
        ),
        DeclareLaunchArgument(
            "control_mode",
            default_value="joint",
            choices=["joint", "eef", "joint_delta"],
            description=(
                "gello_ur_bridge control mode. 'joint' (default): existing "
                "per-joint passthrough, UNCHANGED behaviour -- the bridge does "
                "not import any eef (ur_kin/eef_delta) dependency in this mode. "
                "'eef': the bridge additionally loads "
                "config/ur7e_gello_eef.yaml (layered on top of params_file) and "
                "drives an end-effector delta-pose controller gated behind the "
                "eef_engage/eef_disengage/eef_reclutch/eef_to_joint services. "
                "'joint_delta': START-ANCHORED JOINT DELTA -- the bridge loads "
                "config/ur7e_gello_joint_delta.yaml (layered on top of "
                "params_file) and commands "
                "q_robot_anchor + jd_gain * (leader travel since the anchor), "
                "gated behind the joint_delta_engage / joint_delta_clutch / "
                "joint_delta_reclutch / joint_delta_disengage / "
                "joint_delta_to_joint / joint_delta_start services. The arm "
                "follows how much GELLO MOVES, not where it is, so the operator "
                "can 'mouse lift' (clutch, reposition the leader, reclutch) and "
                "leader drift stops mattering. Bring-up is NO-MOTION in this "
                "mode, exactly like eef: start_mode auto-derives to "
                "'switch_only' and bridge_resume_service to "
                "'/gello_ur_bridge/joint_delta_start', so gello_move_to_start "
                "does a STRICT controller switch IN PLACE (no chase, the arm "
                "does not move) and the bridge anchors at the arm's actual pose "
                "and starts delta streaming -- no manual ~/joint_delta_engage is "
                "needed to begin. "
                "Only the gello_ur_bridge node is affected; move-to-start, the "
                "GELLO publisher, and the gripper nodes are unchanged."
            ),
        ),
        # ---------------------------------------------------------------- #
        # STAGED EEF BRING-UP OVERRIDES (see module docstring).
        #
        # Default "" == NOT OVERRIDDEN: the value from
        # config/ur7e_gello_eef.yaml is used, byte-for-byte as before. We
        # deliberately do NOT repeat the yaml numbers as defaults here -- a
        # duplicated default would silently drift from the yaml.
        #
        # These are consumed ONLY by bridge_node_eef (control_mode:=eef).
        # ---------------------------------------------------------------- #
        DeclareLaunchArgument(
            "v_max",
            default_value="",
            description=(
                "EEF MODE ONLY. Max commanded EEF linear speed (m/s), "
                "overriding v_max in config/ur7e_gello_eef.yaml. Empty "
                "(default) = use the yaml value. Staged bring-up: P6 0.01, "
                "P7 0.02 -> 0.05, P9 0.08. Ignored in control_mode:=joint."
            ),
        ),
        DeclareLaunchArgument(
            "w_max",
            default_value="",
            description=(
                "EEF MODE ONLY. Max commanded EEF angular speed (rad/s), "
                "overriding w_max in config/ur7e_gello_eef.yaml. Empty "
                "(default) = use the yaml value. Staged bring-up: P6 0.05. "
                "Ignored in control_mode:=joint."
            ),
        ),
        DeclareLaunchArgument(
            "pos_scale",
            default_value="",
            description=(
                "EEF MODE ONLY. Leader->follower position gain (dimensionless), "
                "overriding pos_scale in config/ur7e_gello_eef.yaml. Empty "
                "(default) = use the yaml value. Staged bring-up: P6 0.0 (the "
                "robot must not move at all). Ignored in control_mode:=joint."
            ),
        ),
        # ---------------------------------------------------------------- #
        # STAGED JOINT_DELTA BRING-UP OVERRIDE. Same semantics as the three
        # above: default "" == NOT OVERRIDDEN (the joint_delta overlay yaml
        # wins), and a supplied value is coerced to a real Python float before
        # it reaches the node.
        # ---------------------------------------------------------------- #
        DeclareLaunchArgument(
            "jd_gain",
            default_value="",
            description=(
                "JOINT_DELTA MODE ONLY. Leader->robot joint gain "
                "(dimensionless), overriding jd_gain in "
                "config/ur7e_gello_joint_delta.yaml. Empty (default) = use the "
                "yaml value. Staged bring-up: P4 0.0 (the arm must not move at "
                "all after engage), P5 0.25, P6 0.5, P7 1.0. Ignored in "
                "control_mode:=joint / control_mode:=eef."
            ),
        ),
        # ---------------------------------------------------------------- #
        # DISCRETE GRIPPER MODE. Independent of control_mode: it lands on the
        # gello_gripper_bridge node in every mode. Default 'continuous' is
        # today's behaviour, unchanged.
        # ---------------------------------------------------------------- #
        DeclareLaunchArgument(
            "gripper_mode",
            default_value="continuous",
            choices=["continuous", "discrete"],
            description=(
                "gello_gripper_bridge trigger handling. 'continuous' "
                "(default): the leader trigger value is passed through as-is "
                "onto /robotiq_gripper/command_percent -- UNCHANGED behaviour. "
                "'discrete': the trigger is folded to the two endpoints, "
                ">= gripper_close_at -> exactly 1.0 (CLOSED), "
                "<= gripper_open_at -> exactly 0.0 (OPEN), in between the "
                "previous state is HELD (hysteresis, so the boundary cannot "
                "chatter). WHY: the spring+encoder trigger is already ~binary, "
                "but roughly once or twice per session it RESTS PARTWAY OPEN "
                "(worst measured 0.229 / 0.243) and the bridge faithfully "
                "forwards that, so the Robotiq only opens ~76% and the "
                "operator sees a gripper that 'didn't fully open'. Evidence: "
                "docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md §5.1. The latched "
                "state is published on /gello_gripper_bridge/discrete_state "
                "and shown live in the EEF operator GUI."
            ),
        ),
        DeclareLaunchArgument(
            "gripper_close_at",
            default_value="",
            description=(
                "DISCRETE GRIPPER ONLY. Trigger value at/above which the "
                "gripper latches CLOSED, overriding discrete_close_at in the "
                "gello_gripper_bridge block of config/ur7e_gello.yaml (0.7). "
                "Empty (default) = use the yaml value. Must be supplied "
                "TOGETHER with gripper_open_at and satisfy "
                "0.0 < open_at < close_at < 1.0; range-checked at launch."
            ),
        ),
        DeclareLaunchArgument(
            "gripper_open_at",
            default_value="",
            description=(
                "DISCRETE GRIPPER ONLY. Trigger value at/below which the "
                "gripper latches OPEN, overriding discrete_open_at in the "
                "gello_gripper_bridge block of config/ur7e_gello.yaml (0.3). "
                "Empty (default) = use the yaml value. Must be supplied "
                "TOGETHER with gripper_close_at and satisfy "
                "0.0 < open_at < close_at < 1.0; range-checked at launch. "
                "Observed worst-case margin to 0.3 was 0.057 (§5.1) -- re-read "
                "that table before moving it."
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
        # start_mode + bridge_resume_service are AUTO-derived from control_mode
        # when their launch arguments are left empty (see _auto_start_mode /
        # _auto_resume_service above): joint keeps 'gello' + ~/resume exactly as
        # before, eef gets 'switch_only' + ~/eef_resume so bring-up never drags
        # the arm to the leader's (deliberately different) joint pose.
        parameters=[
            params_file,
            {
                "start_mode": _auto_start_mode(control_mode),
                "bridge_resume_service": _auto_resume_service(control_mode),
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
    # control_mode:=joint_delta layers config/ur7e_gello_joint_delta.yaml the
    # same way, and is likewise ONLY consulted under its own IfCondition.
    joint_delta_overlay_file = PathJoinSubstitution(
        [
            FindPackageShare("ur_gello_bringup"),
            "config",
            "ur7e_gello_joint_delta.yaml",
        ]
    )
    is_joint_mode = IfCondition(
        PythonExpression(["'", control_mode, "' == 'joint'"])
    )
    is_eef_mode = IfCondition(
        PythonExpression(["'", control_mode, "' == 'eef'"])
    )
    is_joint_delta_mode = IfCondition(
        PythonExpression(["'", control_mode, "' == 'joint_delta'"])
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

    # The eef bridge is built inside an OpaqueFunction so the staged-bring-up
    # launch args (v_max / w_max / pos_scale) can be resolved to real Python
    # floats before they are handed to the Node -- see
    # _eef_bridge_parameter_overrides() above for why that matters (int
    # coercion of all-digit CLI values). The OpaqueFunction carries the same
    # is_eef_mode condition the Node used to, so in control_mode:=joint it is
    # never even evaluated and the joint path is untouched.
    def _make_bridge_node_eef(context):
        return [
            Node(
                package="ur_gello_bringup",
                executable="gello_ur_bridge",
                parameters=[
                    params_file,
                    eef_overlay_file,
                    # Layered LAST -> wins over both yaml files. Contains an
                    # entry for v_max / w_max / pos_scale ONLY if the operator
                    # actually passed that launch argument.
                    _eef_bridge_parameter_overrides(context),
                ],
                output="screen",
            )
        ]

    bridge_node_eef = OpaqueFunction(
        function=_make_bridge_node_eef,
        condition=is_eef_mode,
    )

    # The joint_delta bridge is built the same way and for the same reason (the
    # jd_gain staged-bring-up arg must be a real Python float, not an int). It
    # carries is_joint_delta_mode, so in control_mode:=joint / :=eef it is never
    # even evaluated and neither of those paths is touched.
    def _make_bridge_node_joint_delta(context):
        return [
            Node(
                package="ur_gello_bringup",
                executable="gello_ur_bridge",
                parameters=[
                    params_file,
                    joint_delta_overlay_file,
                    # Layered LAST -> wins over both yaml files. Contains an
                    # entry for jd_gain ONLY if the operator actually passed
                    # that launch argument.
                    _joint_delta_bridge_parameter_overrides(context),
                ],
                output="screen",
            )
        ]

    bridge_node_joint_delta = OpaqueFunction(
        function=_make_bridge_node_joint_delta,
        condition=is_joint_delta_mode,
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
    #
    #     gripper_mode:=discrete additionally snaps the trigger to the two
    #     endpoints (see _gripper_bridge_parameter_overrides). Built by a
    #     factory rather than a module-level Node so the launch arguments can be
    #     resolved to a real Python bool/floats -- the same int-coercion trap the
    #     eef / joint_delta overrides go through. The factory is called from the
    #     handshake-exit handler below, which already carries a LaunchContext;
    #     the VALIDATION of those arguments happens far earlier, at t=0, via the
    #     _validate_gripper_arguments OpaqueFunction.
    # ------------------------------------------------------------------ #
    def _make_gello_gripper_bridge_node(context):
        return Node(
            package="ur_gello_bringup",
            executable="gello_gripper_bridge",
            parameters=[
                params_file,
                # Layered LAST -> wins over the yaml. Always carries
                # discrete_mode; carries a threshold only if the operator
                # actually passed that launch argument.
                _gripper_bridge_parameter_overrides(context),
            ],
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
        # All three entries are IfCondition-gated on control_mode; exactly one
        # spawns at runtime (the others' conditions evaluate false and they are
        # never launched).
        actions=[bridge_node_joint, bridge_node_eef, bridge_node_joint_delta],
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
                _make_gello_gripper_bridge_node(context),
            ]
        # Quote the SAME resume service the handshake would have called, so the
        # manual-recovery hint is correct in eef mode (~/eef_resume) too.
        resume_svc = _auto_resume_service(control_mode).perform(context)
        return [
            LogInfo(
                msg=(
                    "Move-to-start handshake FAILED (exit "
                    f"{event.returncode}); the pre-spawned bridge stays PAUSED "
                    "and silent (fail-safe) and the grippers are NOT started. "
                    "The robot will stay put. Check that the External Control "
                    "program is PLAYING on the pendant, then re-launch. Manual "
                    "recovery once aligned: re-align the leader and call "
                    f"'ros2 service call {resume_svc} "
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
            # FIRST, before the driver include: resolve + range-check the
            # gripper arguments so a bad threshold aborts at t=0 rather than at
            # handshake exit, with the arm already moved.
            OpaqueFunction(function=_validate_gripper_arguments),
            ur_control_launch,
            gello_publisher_delayed,
            bridge_paused_delayed,
            move_to_start_delayed,
            grippers_after_handshake,
        ]
    )
