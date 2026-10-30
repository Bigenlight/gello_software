"""ROS 2 I/O backend for UR7eEnv.

FrankaEnv talks to a Flask robot server over HTTP; we talk to the ROS 2 graph
from ur_gello_bringup directly. This class owns a rclpy node spun in a
background thread and exposes the same conceptual surface FrankaEnv gets from
its server:

    get_joint_state()      <- /joint_states        (POST /getstate analog)
    get_gripper_percent()  <- ~/position_percent
    get_gello_state()      <- /gello/joint_states + the leader trigger topic
    get_gello_trigger()    <- /gripper/gripper_client/target_gripper_width_percent
    get_image()            <- /camX/.../compressed (realsense2_camera driver)
    send_joint_command()   -> /forward_position_controller/commands
    send_gripper_percent() -> ~/command_percent

The leader arrives on TWO topics, not one: gello_publisher_node publishes only
the 6 arm joints on /gello/joint_states (``position`` has length 6) and the
trigger separately as std_msgs/Float32 on
/gripper/gripper_client/target_gripper_width_percent (0.0 open .. 1.0 closed,
30 Hz). fake_gello_node mirrors that split exactly. get_gello_state() joins the
two streams so callers still see one (7,) leader vector; see merge_gello_state
for the staleness policy.

Cameras are consumed the same way as the rest of the UR7e line
(policy_leader_node, recorder, camera_viewer): JPEG CompressedImage topics
published by launch_cameras.sh's realsense2_camera drivers. The backend caches
the newest raw JPEG per camera; decoding happens in the env at step rate.

All getters return the latest message (with reception timestamp) — the env's
fixed-rate step loop does the pacing, so no getter ever blocks.

UNTESTED SKELETON — wiring (QoS, joint name ordering, staleness thresholds)
to be validated together on the real stack.
"""

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float32, Float64MultiArray

    _ROS_AVAILABLE = True
except ImportError:  # allow import (e.g. fake_env on a non-ROS machine)
    _ROS_AVAILABLE = False


# How long the spin loop blocks in one executor wait before re-checking the
# stop flag. Costs nothing in callback latency — spin_once returns as soon as
# any entity is ready — it only bounds how long close() waits for the thread.
SPIN_POLL_S = 0.1

# close() waits this long for each worker thread. The upsampler checks its stop
# flag every 1/up_hz s (4 ms at the default 250 Hz) and the spin loop every
# SPIN_POLL_S, so a join that needs seconds means a thread is wedged, not
# merely slow — we warn rather than hang the caller's finally block forever.
CLOSE_JOIN_TIMEOUT_S = 2.0


# A single gripper datagram is not a reliable command. Two independent
# mechanisms swallow it, both measured on the real stack (2026-08-06,
# ros2_ur_ws/gello_logs/diag_gripper_halfopen_20260806_173307):
#
#   * DDS discovery. A publish issued before the subscription has matched is
#     dropped with no error anywhere — ``ros2 topic pub -1`` lost 3 of 7 sends.
#   * robotiq_gripper_modbus_node's rate limiter DISCARDS the freshest setpoint
#     when it arrives inside ``command_min_period`` of the last accepted one,
#     and does NOT update its ``_last_cmd_pct`` cache when it does. The driver's
#     own view therefore stays self-consistent: it believes the gripper is where
#     it last wrote, and it is right. Only the publisher knows something else was
#     wanted — and a one-shot publisher never says so again. 7 of 14 open cycles
#     parked at 0.043..0.42 instead of 0.012; re-asserting the setpoint for 1 s
#     failed 0 of 6 times.
#
# So the command path re-asserts the newest setpoint for a short window instead
# of sending it once. Three properties make this cheap and safe:
#
#   * NO extra Modbus traffic. The driver deadbands an unchanged setpoint before
#     it touches the single-client :54321 bus, so repeats cost ROS traffic only.
#   * NOTHING at rest. The re-assert exists only while a command is in flight;
#     with no pending setpoint the 250 Hz worker publishes nothing at all.
#   * LATEST WINS. A new setpoint replaces the pending one, so a re-assert can
#     never fight a command issued after it.
#
# Setting GRIPPER_REASSERT_S (or _HZ) to 0.0 restores exactly the historical
# one-publish-per-call behaviour.
GRIPPER_REASSERT_S = 1.0
GRIPPER_REASSERT_HZ = 10.0


# UR joint order used by ur_gello_bringup command path.
UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Leader trigger topic published by gello_publisher_node / fake_gello_node.
# Kept as a module default (not a config.py key) so the backend works against
# the existing DefaultUR7eEnvConfig.ROS dict unchanged; ros_cfg may still
# override it with the same key name.
GELLO_TRIGGER_TOPIC = "/gripper/gripper_client/target_gripper_width_percent"

# Newest trigger older than this -> treated as absent. Matches
# GelloIntervention.LEADER_STALE_S: both streams come from the same 30 Hz node
# timer, so one going quiet while the other keeps up means the trigger read
# itself failed, and a stale trigger must not be replayed as human intent.
GELLO_TRIGGER_STALE_S = 0.3


def merge_gello_state(
    q,
    age_q: float,
    trigger: Optional[float],
    age_trigger: float,
    stale_s: float = GELLO_TRIGGER_STALE_S,
):
    """Join the two leader streams into one ``((7,), age)`` reading.

    ``arr[:6]`` are the leader joints, ``arr[6]`` the trigger in [0, 1]
    (0 = open, 1 = closed), or **NaN** when no usable trigger exists.

    Decisions, and why:

    * NaN, not 0.0, for a missing trigger. 0.0 is a legal trigger value
      ("fully open"), so using it as the sentinel would silently command the
      gripper open every time the trigger topic dies — exactly the failure the
      caller must be able to detect. NaN is unambiguous and float-typed, so the
      return stays a plain (7,) ndarray and every existing ``arr[:6]`` caller
      is unaffected.
    * The returned age is the JOINT age only. The joints gate whether the human
      may drive the arm at all (GelloIntervention.LEADER_STALE_S); folding an
      infinite trigger age into it would disable arm teleop entirely whenever
      the trigger is merely absent. Trigger freshness is handled here, locally,
      and is also exposed raw via URRosBackend.get_gello_trigger().
    * Legacy fallback: if the trigger topic has NEVER produced a message but
      the JointState carries a 7th position element, that element is used. No
      publisher in this repo does that today (both gello_publisher_node and
      fake_gello_node send exactly 6), but it is the layout the previous
      ``arr[6] if len(arr) > 6`` code assumed, so honouring it keeps any
      out-of-tree or replayed 7-element publisher working. It deliberately does
      NOT apply once the trigger topic has spoken and then gone stale — that is
      a live-signal failure, and falling back there would mask it.
    """
    if q is None:
        return None, age_q
    q = np.asarray(q, dtype=float).ravel()
    if q.size < 6:
        # Malformed leader message: report "no leader" rather than hand back a
        # short array the caller would silently slice into a bad q_lead.
        return None, float("inf")
    if trigger is None:
        grip = float(q[6]) if q.size > 6 else float("nan")  # legacy fallback
    elif age_trigger > stale_s:
        grip = float("nan")
    else:
        grip = float(trigger)
    return np.concatenate([q[:6], [grip]]), age_q


class AccelerationLimitedJointStream:
    """Pure fixed-rate joint command trajectory generator.

    The state is the last published position and its per-tick displacement.
    Keeping displacement (rather than estimating velocity from wall-clock
    callback jitter) makes both command limits exact at the configured publish
    rate::

        abs(q[k] - q[k-1]) <= max_step_rad
        abs(q[k] - 2*q[k-1] + q[k-2]) * hz**2 <= max_accel_rad_s2

    Tracking is per joint. Before accelerating, ``_safe_step_for_distance``
    asks how large the next step may be while still leaving enough distance to
    brake by ``max_accel_rad_s2`` on every later tick. That discrete braking
    distance is what prevents a fixed target from being crossed at the end of
    a move. A target which jumps behind a joint that is already moving cannot
    be obeyed without either overshoot or an acceleration discontinuity; in
    that case acceleration continuity wins and the joint brakes through the
    reversal.

    This helper deliberately sees only joint targets. It does not filter policy
    actions (no EMA/One-Euro), so the action stored in replay remains the action
    selected by the policy/intervention path; this is the actuator-side command
    governor already represented by the environment dynamics.
    """

    UNSEEDED = "UNSEEDED"
    TRACKING = "TRACKING"
    BRAKING = "BRAKING"
    HOLD = "HOLD"

    def __init__(
        self,
        hz: float,
        max_step_rad: float,
        max_accel_rad_s2: float,
        soft_start_s: float,
        target_stale_s: float,
        soft_start_fraction: float = 0.15,
    ):
        self.hz = float(hz)
        self.max_step_rad = float(max_step_rad)
        self.max_accel_rad_s2 = float(max_accel_rad_s2)
        self.soft_start_s = float(soft_start_s)
        self.target_stale_s = float(target_stale_s)
        self.soft_start_fraction = float(soft_start_fraction)

        if self.hz <= 0.0:
            raise ValueError("hz must be positive")
        if self.max_step_rad <= 0.0:
            raise ValueError("max_step_rad must be positive")
        if self.max_accel_rad_s2 <= 0.0:
            raise ValueError("max_accel_rad_s2 must be positive")
        if self.soft_start_s < 0.0:
            raise ValueError("soft_start_s must be non-negative")
        if self.target_stale_s <= 0.0:
            raise ValueError("target_stale_s must be positive")
        if not 0.0 <= self.soft_start_fraction <= 1.0:
            raise ValueError("soft_start_fraction must be in [0, 1]")

        # A change of this many radians/tick corresponds exactly to a change
        # of max_accel_rad_s2 in the nominal command velocity.
        self._max_step_delta = self.max_accel_rad_s2 / (self.hz * self.hz)
        self._q: Optional[np.ndarray] = None
        self._step: Optional[np.ndarray] = None
        self._soft_start_time: Optional[float] = None
        self.mode = self.UNSEEDED

    @property
    def seeded(self) -> bool:
        return self._q is not None

    @property
    def position(self) -> Optional[np.ndarray]:
        return None if self._q is None else self._q.copy()

    @property
    def velocity(self) -> Optional[np.ndarray]:
        return None if self._step is None else self._step.copy() * self.hz

    def braking_endpoint(self) -> np.ndarray:
        """Return the position reached by acceleration-limited braking.

        This is a prediction only; it does not mutate the stream.  Giving the
        returned point back as the target makes an explicit HOLD decelerate to
        zero without continuing toward an older, lagging policy/GELLO target.
        """
        if self._q is None or self._step is None:
            raise RuntimeError("seed() must be called before braking_endpoint()")
        q = self._q.copy()
        step = self._step.copy()
        while np.any(step != 0.0):
            step = step + np.clip(
                -step, -self._max_step_delta, self._max_step_delta
            )
            step[np.abs(step) < 1e-15] = 0.0
            q = q + step
        return q

    def reset(self):
        """Forget all trajectory state; the next command must seed measured q."""
        self._q = None
        self._step = None
        self._soft_start_time = None
        self.mode = self.UNSEEDED

    def seed(self, measured_q: np.ndarray, now: float) -> np.ndarray:
        """Seed at the measured joints with zero velocity and return them exact."""
        q = np.asarray(measured_q, dtype=float).ravel()
        if q.size == 0 or not np.all(np.isfinite(q)):
            raise ValueError("measured_q must be a non-empty finite vector")
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        self._q = q.copy()
        self._step = np.zeros_like(q)
        self._soft_start_time = float(now)
        self.mode = self.TRACKING
        return self._q.copy()

    def _soft_step_cap(self, now: float) -> float:
        if self.soft_start_s == 0.0 or self._soft_start_time is None:
            return self.max_step_rad
        fraction = np.clip(
            (float(now) - self._soft_start_time) / self.soft_start_s,
            0.0,
            1.0,
        )
        scale = self.soft_start_fraction + (1.0 - self.soft_start_fraction) * fraction
        return self.max_step_rad * float(scale)

    def _stopping_distance(self, step: float) -> float:
        """Minimum distance including ``step`` before reaching zero velocity."""
        if step <= 0.0:
            return 0.0
        ticks = int(np.ceil(step / self._max_step_delta))
        return (
            ticks * step
            - self._max_step_delta * ticks * (ticks - 1) / 2.0
        )

    def _safe_step_for_distance(self, distance: float, step_cap: float) -> float:
        """Largest next step whose discrete braking distance fits ``distance``."""
        if distance <= 0.0:
            return 0.0
        if self._stopping_distance(step_cap) <= distance:
            return step_cap

        # Stopping distance is continuous and monotone even where ceil() adds a
        # braking tick. Bisection avoids a continuous-time sqrt(2*a*d)
        # approximation, whose half-tick error can overshoot small targets.
        low, high = 0.0, step_cap
        for _ in range(48):
            mid = (low + high) / 2.0
            if self._stopping_distance(mid) <= distance:
                low = mid
            else:
                high = mid
        return low

    def _move_step_toward(self, desired: np.ndarray) -> np.ndarray:
        assert self._step is not None
        delta = np.clip(
            desired - self._step,
            -self._max_step_delta,
            self._max_step_delta,
        )
        return self._step + delta

    def advance(
        self,
        target_q: np.ndarray,
        now: float,
        target_time: Optional[float],
    ) -> np.ndarray:
        """Advance one 1/hz tick, braking to HOLD when ``target_time`` is stale."""
        if self._q is None or self._step is None:
            raise RuntimeError("seed() must be called before advance()")
        target = np.asarray(target_q, dtype=float).ravel()
        if target.shape != self._q.shape or not np.all(np.isfinite(target)):
            raise ValueError("target_q must match the seeded finite vector")
        if not np.isfinite(now):
            raise ValueError("now must be finite")

        stale = (
            target_time is None
            or not np.isfinite(target_time)
            or float(now) - float(target_time) > self.target_stale_s
        )
        if stale:
            # Ignore the obsolete target. Ramp velocity to zero, continue
            # publishing the deceleration trajectory, then repeat the final
            # stream position indefinitely as HOLD.
            self._step = self._move_step_toward(np.zeros_like(self._step))
            self._step[np.abs(self._step) < 1e-15] = 0.0
            self._q = self._q + self._step
            self.mode = self.HOLD if np.all(self._step == 0.0) else self.BRAKING
            return self._q.copy()

        # A command stream recovered after reaching stale HOLD starts a fresh
        # 0.7 s ease-in, but remains anchored at the held command position.
        if self.mode == self.HOLD:
            self._soft_start_time = float(now)

        error = target - self._q
        step_cap = self._soft_step_cap(now)
        desired = np.empty_like(error)
        for joint, joint_error in enumerate(error):
            magnitude = self._safe_step_for_distance(abs(joint_error), step_cap)
            desired[joint] = np.sign(joint_error) * magnitude

        self._step = self._move_step_toward(desired)
        self._q = self._q + self._step
        self.mode = self.TRACKING
        return self._q.copy()


class URRosBackend:
    #: How long, and how fast, a gripper setpoint is re-asserted after it is
    #: issued (see GRIPPER_REASSERT_S above for why one publish is not enough).
    #: Instance attributes, so a caller or a test can retune or disable them
    #: without touching the module: either at 0.0 gives back the historical
    #: single publish exactly.
    grip_reassert_s: float = GRIPPER_REASSERT_S
    grip_reassert_hz: float = GRIPPER_REASSERT_HZ

    def __init__(
        self,
        ros_cfg: Dict[str, str],
        camera_topics: Optional[Dict[str, str]] = None,
        upsampler_cfg: Optional[Dict[str, float]] = None,
        dry_run: bool = True,
    ):
        if not _ROS_AVAILABLE:
            raise RuntimeError("rclpy not available — use fake_env=True")

        self.dry_run = dry_run
        self._lock = threading.Lock()
        self._shutdown = False
        # Teardown state. _closed makes close() idempotent (env.close() and a
        # finally: block both call it); _node_alive is the publish guard — once
        # it is False nothing may touch the node's publishers again.
        self._close_lock = threading.Lock()
        self._closed = False
        self._node_alive = True
        self._owns_rclpy = False

        # latest-state caches: (data, wall-clock reception time)
        self._q: Optional[Tuple[np.ndarray, float]] = None          # (6,) robot joints
        self._dq: Optional[Tuple[np.ndarray, float]] = None         # (6,) robot joint vel
        self._gripper: Optional[Tuple[float, float]] = None         # 0.0 open..1.0 closed
        self._gello: Optional[Tuple[np.ndarray, float]] = None      # (6,) leader q
        self._gello_trigger: Optional[Tuple[float, float]] = None   # leader trigger 0..1
        self._wrench: Optional[Tuple[np.ndarray, float]] = None     # (6,) fx..tz, TCP F/T
        self._tcp_pose: Optional[Tuple[np.ndarray, float]] = None   # (7,) xyz + quat
        self._images: Dict[str, Tuple[bytes, float]] = {}           # name -> raw JPEG

        # Only shut the context down in close() if we were the ones who brought
        # it up — see close() for why that distinction matters.
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = Node("ur7e_env_backend")

        self._node.create_subscription(
            JointState, ros_cfg["joint_states_topic"], self._on_joint_states, 10
        )
        self._node.create_subscription(
            JointState, ros_cfg["gello_topic"], self._on_gello, 10
        )
        # Leader trigger: a SEPARATE topic, not a 7th element of gello_topic.
        # Subscribing here (rather than widening gello_publisher's JointState)
        # leaves the /gello/joint_states contract — position length 6 — intact
        # for its existing consumers (gello_ur_bridge, gello_gripper_bridge,
        # the recorder, the GUI), which is the lower-risk side of the change.
        self._node.create_subscription(
            Float32,
            ros_cfg.get("gello_gripper_topic", GELLO_TRIGGER_TOPIC),
            self._on_gello_trigger,
            10,
        )
        self._node.create_subscription(
            Float32, ros_cfg["gripper_state_topic"], self._on_gripper, 10
        )
        if "wrench_topic" in ros_cfg:
            self._node.create_subscription(
                WrenchStamped, ros_cfg["wrench_topic"], self._on_wrench, 10
            )
        if "tcp_pose_topic" in ros_cfg:
            self._node.create_subscription(
                PoseStamped, ros_cfg["tcp_pose_topic"], self._on_tcp_pose, 10
            )
        # depth-1 subscriptions: only the newest frame is ever interesting.
        # VERIFY(hw): all subscriptions here use default (reliable) QoS. If a
        # publisher is best-effort the subscription silently never fires —
        # check every topic actually delivers on first bring-up (ros2 topic hz).
        for cam_name, topic in (camera_topics or {}).items():
            self._node.create_subscription(
                CompressedImage,
                topic,
                self._make_image_cb(cam_name),
                1,
            )

        self._cmd_pub = self._node.create_publisher(
            Float64MultiArray, ros_cfg["command_topic"], 10
        )
        self._gripper_pub = self._node.create_publisher(
            Float32, ros_cfg["gripper_command_topic"], 10
        )
        # Newest gripper setpoint still owed a re-assert, as
        # (value, window_deadline, next_publish_at) on the monotonic clock.
        # None = nothing pending, which is also the resting state.
        self._grip_pending: Optional[Tuple[float, float, float]] = None

        # Own executor rather than rclpy.spin(node) (which uses the GLOBAL
        # executor): close() can then stop *our* spin loop with
        # executor.shutdown() without shutting down a context that other nodes
        # in this process may be sharing. rclpy.spin() offers no such handle —
        # the only way to make it return is to kill the whole context.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # ---- 10 Hz targets -> velocity/acceleration-limited 250 Hz stream ---- #
        ucfg = upsampler_cfg or {"hz": 250.0, "max_step_rad": 0.0025}
        self._up_hz = float(ucfg["hz"])
        self._up_step = float(ucfg["max_step_rad"])
        self._up_accel = float(ucfg.get("max_accel_rad_s2", 8.0))
        self._up_soft_start_s = float(ucfg.get("soft_start_s", 0.7))
        self._up_soft_start_fraction = float(
            ucfg.get("soft_start_fraction", 0.15)
        )
        self._target_stale_s = float(ucfg.get("target_stale_s", 0.3))
        self._q_target: Optional[np.ndarray] = None   # latest 10 Hz goal
        self._q_target_time: Optional[float] = None   # receipt, monotonic clock
        self._q_stream: Optional[np.ndarray] = None   # what we're publishing now
        self._up_streamer = AccelerationLimitedJointStream(
            hz=self._up_hz,
            max_step_rad=self._up_step,
            max_accel_rad_s2=self._up_accel,
            soft_start_s=self._up_soft_start_s,
            target_stale_s=self._target_stale_s,
            soft_start_fraction=self._up_soft_start_fraction,
        )
        self._up_thread = threading.Thread(target=self._upsample_loop, daemon=True)
        self._up_thread.start()

    # ------------------------------------------------------------------ #
    # callbacks                                                           #
    # ------------------------------------------------------------------ #
    def _spin(self):
        """Pump callbacks until close() sets _shutdown.

        Deliberately a bounded ``spin_once`` loop rather than ``executor.spin()``
        (or the original ``rclpy.spin(node)``). Both of those block in the
        executor indefinitely, and the only documented way to break them out —
        ``Executor.shutdown()`` — is itself unsafe to call from another thread
        while this one is inside the wait: ``_wait_for_ready_callbacks`` does
        ``guards.append(self._guard)`` and hands that guard's handle to a live
        rcl wait set, while ``shutdown()`` concurrently does
        ``self._guard.destroy(); self._guard = None``. That is a use-after-free
        in C, and it segfaulted 1 run in 50 on the real stack.

        Polling the stop flag inverts the ordering: this thread leaves the
        executor on its own, close() joins it, and only then is the executor
        torn down — by which point nobody is inside it. Throughput is
        unaffected; ``SingleThreadedExecutor.spin()`` is itself just a
        ``spin_once()`` loop, and ``spin_once`` returns the moment any entity is
        ready, so SPIN_POLL_S is an idle timeout, not a callback delay.
        """
        while not self._shutdown:
            try:
                self._executor.spin_once(timeout_sec=SPIN_POLL_S)
            except Exception as exc:  # noqa: BLE001
                if self._shutdown or not rclpy.ok():
                    break  # teardown, not a fault
                # A raising callback must not silently kill the whole state
                # stream — report it and keep pumping (throttled, so a
                # persistently bad callback cannot spin hot).
                print(f"URRosBackend spin: {type(exc).__name__}: {exc}")
                time.sleep(SPIN_POLL_S)

    def _on_joint_states(self, msg: "JointState"):
        # Reorder by name — joint_state_broadcaster order is not guaranteed.
        try:
            idx = [msg.name.index(n) for n in UR_JOINT_NAMES]
        except ValueError:
            return  # not the UR arm (e.g. gripper joint-only message)
        q = np.array([msg.position[i] for i in idx])
        dq = (
            np.array([msg.velocity[i] for i in idx])
            if len(msg.velocity) == len(msg.name)
            else np.zeros(6)
        )
        now = time.monotonic()
        with self._lock:
            self._q = (q, now)
            self._dq = (dq, now)

    def _on_gello(self, msg: "JointState"):
        # 6 arm joints; the trigger arrives on its own topic (see _on_gello_trigger).
        arr = np.asarray(msg.position, dtype=float)
        with self._lock:
            self._gello = (arr, time.monotonic())

    def _on_gello_trigger(self, msg: "Float32"):
        """Leader trigger, 0.0 = OPEN .. 1.0 = CLOSED (same span as the URCap)."""
        with self._lock:
            self._gello_trigger = (float(msg.data), time.monotonic())

    def _on_gripper(self, msg: "Float32"):
        with self._lock:
            self._gripper = (float(msg.data), time.monotonic())

    def _on_wrench(self, msg: "WrenchStamped"):
        w = msg.wrench
        arr = np.array(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        )
        with self._lock:
            self._wrench = (arr, time.monotonic())

    def _on_tcp_pose(self, msg: "PoseStamped"):
        p, o = msg.pose.position, msg.pose.orientation
        arr = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])  # xyz + quat (xyzw)
        with self._lock:
            self._tcp_pose = (arr, time.monotonic())

    def _make_image_cb(self, cam_name: str):
        def _cb(msg: "CompressedImage"):
            with self._lock:
                self._images[cam_name] = (bytes(msg.data), time.monotonic())

        return _cb

    # ------------------------------------------------------------------ #
    # getters (never block; return (data, age_seconds))                   #
    # ------------------------------------------------------------------ #
    def _aged(self, slot):
        if slot is None:
            return None, float("inf")
        data, t = slot
        return data, time.monotonic() - t

    def get_joint_state(self):
        with self._lock:
            q, age_q = self._aged(self._q)
            dq, _ = self._aged(self._dq)
        return q, dq, age_q

    def get_gripper_percent(self):
        with self._lock:
            return self._aged(self._gripper)

    def get_gello_state(self):
        """Returns ((7,) leader joints + trigger, joint age) — see merge_gello_state.

        arr[6] is NaN when the trigger topic has not delivered a usable value;
        callers must test it (GelloExpert.get_leader turns NaN into None).
        """
        with self._lock:
            q, age_q = self._aged(self._gello)
            trig, age_trig = self._aged(self._gello_trigger)
        return merge_gello_state(
            q, age_q, trig, age_trig, stale_s=GELLO_TRIGGER_STALE_S
        )

    def get_gello_trigger(self):
        """Returns (trigger 0..1 or None, age_seconds) — raw, no staleness policy."""
        with self._lock:
            return self._aged(self._gello_trigger)

    def get_wrench(self):
        with self._lock:
            return self._aged(self._wrench)

    def get_tcp_pose(self):
        with self._lock:
            return self._aged(self._tcp_pose)

    def get_image(self, cam_name: str):
        """Returns (raw JPEG bytes or None, age_seconds)."""
        with self._lock:
            return self._aged(self._images.get(cam_name))

    # ------------------------------------------------------------------ #
    # commands                                                            #
    # ------------------------------------------------------------------ #
    def send_joint_command(self, q_cmd: np.ndarray):
        """Update the upsampler's target (does NOT publish directly)."""
        if self._closed:
            # A late command after teardown must not linger as a target: the
            # upsampler is gone, so it would never be acted on here, but it
            # would silently become the seed for anything that inspected
            # _q_target afterwards.
            return
        target = np.asarray(q_cmd, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(target)):
            raise ValueError("joint command must contain only finite values")
        target_time = time.monotonic()
        with self._lock:
            self._q_target = target
            self._q_target_time = target_time

    def reset_command_stream(self):
        """Drop target/stream so the next target re-seeds from measured joints.

        Call on env.reset(): a stale stream from the previous episode must not
        race the new one.
        """
        with self._lock:
            self._q_target = None
            self._q_target_time = None
            self._q_stream = None
            self._up_streamer.reset()

    def request_hold(self) -> np.ndarray:
        """Replace the current goal with its finite-acceleration stop point.

        The operation is atomic with respect to the 250 Hz worker.  A wrapper
        can therefore stop chasing an accumulated target while preserving the
        same acceleration bound; the returned joints are also the exact point
        to which the task-space command integrator must be re-anchored.
        """
        now = time.monotonic()
        with self._lock:
            if self._up_streamer.seeded:
                hold = self._up_streamer.braking_endpoint()
            elif self._q is not None:
                hold = np.asarray(self._q[0], dtype=float).reshape(6).copy()
            else:
                raise RuntimeError("no measured/streamed joints available for HOLD")
            self._q_target = hold.copy()
            self._q_target_time = now
            return hold.copy()

    def _upsample_loop(self):
        """250 Hz worker: acceleration-limit the latest timestamped target.

        Seeding: the stream starts at the *measured* joints, so the first
        published command is exactly measured q (zero jump and zero velocity).
        With no target yet (or after reset_command_stream) nothing is published.
        Fresh targets retain the existing max_step_rad ceiling and add a finite
        per-joint acceleration plus braking-distance planning. If target updates
        stop, the last target is not chased forever: the stream decelerates and
        keeps publishing its final position as HOLD. There remains exactly one
        joint-command publisher, owned by this worker.
        """
        period = 1.0 / self._up_hz
        while not self._shutdown:
            t0 = time.monotonic()
            stream = None
            with self._lock:
                target = self._q_target
                target_time = self._q_target_time
                q_meas = self._q[0] if self._q is not None else None
                if target is not None:
                    if not self._up_streamer.seeded:
                        if q_meas is not None:
                            # Seed is itself this tick's output. Advancing here
                            # would turn the first publish into a one-step jump.
                            stream = self._up_streamer.seed(q_meas, t0)
                    else:
                        stream = self._up_streamer.advance(
                            target, t0, target_time
                        )
                    if stream is not None:
                        self._q_stream = stream.copy()
            if stream is not None:
                # _node_alive, not just dry_run: close() flips _shutdown and
                # _node_alive together, but this thread may already be inside
                # the body when that happens. Re-checking immediately before
                # the publish keeps us off a node close() is about to destroy.
                if not self.dry_run and self._node_alive:
                    msg = Float64MultiArray()
                    msg.data = [float(v) for v in stream]
                    self._cmd_pub.publish(msg)
            # Same worker, same lifetime guarantees: close() stops this loop
            # before it destroys the node, so no re-assert can outlive the
            # backend either. Costs nothing when no setpoint is pending.
            self._tick_gripper_reassert(t0)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def send_gripper_percent(self, fraction: float):
        """Robotiq command_percent convention: 0.0 = OPEN .. 1.0 = CLOSED.

        Publishes immediately AND arms a bounded re-assert of the same value on
        the 250 Hz worker (``grip_reassert_s`` at ``grip_reassert_hz``), because
        one datagram is not a reliable command — see GRIPPER_REASSERT_S for the
        two measured loss mechanisms and why the repeats cost no bus traffic.

        Non-blocking by construction: the caller publishes once and returns, so
        this is usable from the RL step loop (~2 Hz measured) and from the
        follower thread without adding latency to either.

        Re-asserting is NOT a position watchdog. It repeats the value that was
        COMMANDED and never reads ``position_percent``, so it cannot re-command
        a closing setpoint because the fingers stopped short — that is a
        successful grasp, and nothing here can turn it into a squeeze.
        """
        if self.dry_run or not self._node_alive:
            return
        value = float(np.clip(fraction, 0.0, 1.0))
        window = float(self.grip_reassert_s or 0.0)
        hz = float(self.grip_reassert_hz or 0.0)
        now = time.monotonic()
        msg = Float32()
        msg.data = value
        # Publish and re-arm under ONE lock hold, and let the worker publish
        # under the same one. LATEST WINS is otherwise only true of the stored
        # state, not of the wire: the worker decides to re-assert, leaves the
        # lock, and is then descheduled; this call publishes CLOSE and stores
        # it; the worker wakes and publishes its stale OPEN *after* it. The
        # driver's big-jump exemption (|dv| >= 0.5) waves that inversion
        # straight through to the single-client bus, so the fingers open for
        # one re-assert period in the middle of a grasp. Serialising the two
        # publishers is what makes the newest setpoint the last one on the
        # wire. This is the ONLY publish held under the lock: it is ~2 Hz and
        # the joint stream deliberately stays outside (see _upsample_loop).
        with self._lock:
            self._gripper_pub.publish(msg)
            if window <= 0.0 or hz <= 0.0:
                # Disabled: still drop any older pending setpoint, so turning
                # the feature off mid-flight cannot leave a stale value looping.
                self._grip_pending = None
            else:
                # LATEST WINS: this setpoint supersedes whatever was pending.
                # Without that, a re-assert of "open" could keep firing after
                # the policy has asked to close 0.6 s later.
                self._grip_pending = (value, now + window, now + 1.0 / hz)

    def _tick_gripper_reassert(self, now: float) -> None:
        """Re-publish the pending gripper setpoint; called by the 250 Hz worker.

        Publishes INSIDE the lock, which is what makes "latest wins" true of
        the wire and not merely of the stored state — see send_gripper_percent
        for the inversion that costs a grasp otherwise. ``_node_alive`` is
        re-checked immediately before, so close() cannot be raced into
        publishing onto a destroyed node.

        The unlocked pre-check keeps the resting state free: with nothing
        pending this costs one attribute read per 250 Hz tick and never
        contends for the lock at all.
        """
        pending = getattr(self, "_grip_pending", None)
        if pending is None:
            return  # resting state: no traffic at all
        with self._lock:
            pending = self._grip_pending
            if pending is None:
                return
            value, deadline, next_at = pending
            if now >= deadline:
                self._grip_pending = None
                return
            if now < next_at:
                return
            hz = float(self.grip_reassert_hz or 0.0)
            period = (1.0 / hz) if hz > 0.0 else (deadline - now)
            self._grip_pending = (value, deadline, now + period)
            if self.dry_run or not self._node_alive:
                return
            msg = Float32()
            msg.data = value
            self._gripper_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # teardown                                                            #
    # ------------------------------------------------------------------ #
    def close(self, join_timeout: float = CLOSE_JOIN_TIMEOUT_S):
        """Ordered, idempotent teardown. Stop the threads, THEN destroy the node.

        The order is the whole point. The previous version did only::

            self._shutdown = True
            self._node.destroy_node()

        which never joined either worker and never shut the context down, so
        destroy_node() raced a live 250 Hz _upsample_loop and a live spin
        thread. Two things came out of that, both observed on the real stack:

        * ``terminate called without an active exception`` + a core dump on
          roughly 1 in 30 clean exits. A read-only probe that shut down
          normally still aborted, which makes every real session end
          indistinguishable from a genuine crash.
        * The safety one. _upsample_loop republishes the last target at 250 Hz;
          if it outlives the env (an exception on the way out, say) the robot
          keeps being driven. DRY_RUN=True hides this today because nothing is
          published — with DRY_RUN=False it is a moving arm nobody is holding.

        So:

        1. ``_shutdown``/``_node_alive`` — both loops stop; the command loop
           stops publishing.
        2. join ``_up_thread`` — no command can still be in flight.
        3. join ``_spin_thread`` — no callback can still be touching the node,
           and nothing is left inside the executor.
        4. ``executor.shutdown()`` then ``destroy_node()`` — safe only now, in
           this order; see _spin for why shutting the executor down while the
           spin thread is still in it is a use-after-free.
        5. ``rclpy.shutdown()`` — only if we called ``rclpy.init()`` ourselves.

        On (5): the constructor inits the context only ``if not rclpy.ok()``,
        i.e. it explicitly supports being dropped into a process that already
        has a ROS context, where other nodes are live on it. Shutting that
        context down unconditionally would tear those nodes out from under
        their owners on our way out. Owning the executor (step 3) is what lets
        us stop *our* spin thread without needing the context down at all, so
        the shutdown becomes a pure ownership question: we brought it up, we
        take it down; we borrowed it, we leave it. Skipping it when we do own
        it is not an option — leaving the context up means rcl/rmw teardown
        happens during interpreter finalization instead, which is the exact
        window the abort came from.

        Threads that miss ``join_timeout`` are reported and left alone rather
        than hung on; close() is routinely called from a ``finally`` block and
        must not be the thing that wedges a shutdown.
        """
        with self._close_lock:
            if self._closed:
                return  # env.close() and a finally: block may both call us
            self._closed = True

        # 1. stop the workers and close the publish gate before anything else.
        self._shutdown = True
        self._node_alive = False

        # 2./3. join both workers, so nothing is mid-callback or mid-publish
        # and nothing is left inside the executor.
        stuck = []
        self._up_thread.join(timeout=join_timeout)
        if self._up_thread.is_alive():
            stuck.append("upsampler")
        self._spin_thread.join(timeout=join_timeout)
        if self._spin_thread.is_alive():
            stuck.append("spin")

        if stuck:
            # Everything below frees memory those threads may still be reading.
            # A leaked node in a process that is on its way out is harmless; a
            # segfault at exit is not, and is indistinguishable from a real
            # crash — which is the entire reason this method was rewritten.
            print(
                "URRosBackend.close(): thread(s) did not stop within "
                f"{join_timeout}s: {', '.join(stuck)} — skipping node/context "
                "teardown to avoid destroying objects still in use"
            )
            return

        # 4. nobody is inside the executor and nothing references the node.
        try:
            self._executor.shutdown(timeout_sec=join_timeout)
        except Exception:  # noqa: BLE001 — already-shutdown executor
            pass
        self._node.destroy_node()

        # 5. context, ours only.
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
