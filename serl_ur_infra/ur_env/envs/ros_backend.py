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


class URRosBackend:
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

        # Own executor rather than rclpy.spin(node) (which uses the GLOBAL
        # executor): close() can then stop *our* spin loop with
        # executor.shutdown() without shutting down a context that other nodes
        # in this process may be sharing. rclpy.spin() offers no such handle —
        # the only way to make it return is to kill the whole context.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # ---- command upsampler: 10 Hz targets -> slew-limited 250 Hz stream ---- #
        ucfg = upsampler_cfg or {"hz": 250.0, "max_step_rad": 0.002}
        self._up_hz = float(ucfg["hz"])
        self._up_step = float(ucfg["max_step_rad"])
        self._q_target: Optional[np.ndarray] = None   # latest 10 Hz goal
        self._q_stream: Optional[np.ndarray] = None   # what we're publishing now
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
        with self._lock:
            self._q_target = np.asarray(q_cmd, dtype=float).reshape(6).copy()

    def reset_command_stream(self):
        """Drop target/stream so the next target re-seeds from measured joints.

        Call on env.reset(): a stale stream from the previous episode must not
        race the new one.
        """
        with self._lock:
            self._q_target = None
            self._q_stream = None

    def _upsample_loop(self):
        """250 Hz worker: slew the published stream toward the latest target.

        Seeding: the stream starts at the *measured* joints, so the first
        published command is jump-free by construction. With no target yet (or
        after reset_command_stream) nothing is published — same "never emit an
        unearned command" rule as eef_delta's DISENGAGED state.

        Once at the target it keeps publishing the held pose; if the env dies
        the robot simply holds position. An orderly exit stops this loop —
        close() joins the thread before destroying the node — but an abrupt one
        (process killed, thread starved) does not. TODO(together):
        target-staleness policy (stop publishing after N s without a fresh
        target?), to be decided with the other safe-stop cases.
        """
        period = 1.0 / self._up_hz
        while not self._shutdown:
            t0 = time.monotonic()
            with self._lock:
                target = self._q_target
                stream = self._q_stream
                q_meas = self._q[0] if self._q is not None else None
            if target is not None:
                if stream is None:
                    if q_meas is None:
                        time.sleep(period)
                        continue
                    stream = q_meas.copy()
                stream = stream + np.clip(
                    target - stream, -self._up_step, self._up_step
                )
                with self._lock:
                    self._q_stream = stream
                # _node_alive, not just dry_run: close() flips _shutdown and
                # _node_alive together, but this thread may already be inside
                # the body when that happens. Re-checking immediately before
                # the publish keeps us off a node close() is about to destroy.
                if not self.dry_run and self._node_alive:
                    msg = Float64MultiArray()
                    msg.data = [float(v) for v in stream]
                    self._cmd_pub.publish(msg)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def send_gripper_percent(self, fraction: float):
        """Robotiq command_percent convention: 0.0 = OPEN .. 1.0 = CLOSED."""
        if self.dry_run or not self._node_alive:
            return
        msg = Float32()
        msg.data = float(np.clip(fraction, 0.0, 1.0))
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
