#!/usr/bin/env python3
"""Synthetic GELLO leader driven by an ACT policy server.

This rclpy node impersonates the physical GELLO leader arm: it publishes the
EXACT same ``/gello/joint_states`` + gripper contract ``gello_publisher_node``
does, so the existing ``gello_ur_bridge`` + ``gello_move_to_start`` handshake and
the Robotiq modbus stack run UNMODIFIED. The joint targets come from a trained
ACT policy that lives in a separate py3.12 process, reached over localhost ZMQ
(REQ here / REP there) -- keeping torch/lerobot out of this py3.10 ROS process.

State machine (BUILD_SPEC §5):
  HOLD  (boot default): publish a constant held pose (start_pose) + start_gripper
        every tick. Never queries the server. This is the pose
        ``gello_move_to_start`` chases to complete the startup handshake.
  EXECUTE: each tick assemble the observation (7-vector state + 2 raw JPEG
        frames), ZMQ-ACT the server, apply the two safety clamps, and publish the
        clamped target. On any ZMQ timeout/error or ok:false -> FAULT.
  FAULT: STOP publishing entirely (fail-silent). The bridge's staleness watchdog
        then trips and halts the arm. Stay here until re-armed via
        ``~/start_execution``. NEVER hold-last-target-forever through an outage.

Services (std_srvs/Trigger):
  ~/start_execution : HOLD (or FAULT) -> EXECUTE. Guards that the live arm pose is
                      within ~0.1 rad of start_pose, then ZMQ-RESETs the server.
  ~/hold            : -> HOLD, republishing the current live pose held constant.

Distro note: written to be distro-agnostic (std sensor_msgs/std_msgs/std_srvs +
rclpy). Verified to construct under ROS2 Jazzy on the dev PC; the target robot PC
runs Humble -- no Jazzy-only APIs are used (see module docstring caveats below).
"""

import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray, String
from std_srvs.srv import Trigger

import zmq

from gello_policy import obs_assembler
from gello_policy.joint_angles import angular_deviations, positions_near_reference
from gello_policy.obs_assembler import UR_JOINT_ORDER
from gello_policy.remote_policy_client import (
    ImageSnapshot,
    ObservationSnapshot,
    ServerContract,
    create_worker,
)

_N = len(UR_JOINT_ORDER)

# --- States ------------------------------------------------------------------
HOLD = "HOLD"
EXECUTE = "EXECUTE"
FAULT = "FAULT"
ARMING = "ARMING"

# Live-pose gate for arming: the arm must already be sitting on start_pose (the
# handshake has converged) before we hand control to the policy.
START_GATE_RAD = 0.1


class PolicyLeaderNode(Node):
    """ACT-driven synthetic GELLO leader with a HOLD/EXECUTE/FAULT state machine."""

    def __init__(self):
        super().__init__("policy_leader_node")

        # --- Parameters (BUILD_SPEC §5 defaults) -------------------------
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter(
            "start_pose", [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]
        )
        self.declare_parameter("start_gripper", 0.0)
        self.declare_parameter("act_host", "127.0.0.1")
        self.declare_parameter("act_port", 5591)
        self.declare_parameter("act_timeout_s", 0.5)
        self.declare_parameter("inference_transport", "zmq")
        self.declare_parameter("grpc_port", 50051)
        self.declare_parameter("camera_width", 1280)
        self.declare_parameter("camera_height", 720)
        self.declare_parameter("max_camera_skew_s", 0.1)
        self.declare_parameter("max_jpeg_bytes", 4194304)
        self.declare_parameter("expected_model_id", "Bigenlight/diffusion_banana_in_pot_joint")
        self.declare_parameter("expected_checkpoint_revision", "unknown")
        self.declare_parameter("expected_scheduler", "DDIM")
        self.declare_parameter("expected_inference_steps", 10)
        self.declare_parameter("expected_action_steps", 32)
        self.declare_parameter("expected_resize_height", 360)
        self.declare_parameter("expected_resize_width", 640)
        self.declare_parameter(
            "joint_limits_lo", [2.40, -2.53, 1.05, -3.34, -2.18, -5.22]
        )
        self.declare_parameter(
            "joint_limits_hi", [3.67, -0.88, 2.62, -1.01, -1.25, -1.52]
        )
        self.declare_parameter("max_dev_rad", 0.5)
        # Max age of any observation before EXECUTE FAULTs. Guards against a frozen
        # camera / hung joint-state stream driving the policy on a stale frame
        # (review: obs-freshness watchdog). ~0.5 s = a real stall, not a dropped frame.
        self.declare_parameter("obs_timeout_s", 0.5)
        self.declare_parameter("auto_start_on_stream", False)
        # Camera topics (match the recorder defaults; overridable).
        self.declare_parameter("cam1_topic", "/cam1/cam1/color/image_raw/compressed")
        self.declare_parameter("cam2_topic", "/cam2/cam2/color/image_raw/compressed")

        gp = self.get_parameter
        self._publish_rate_hz = gp("publish_rate_hz").value
        self._start_pose = [float(x) for x in gp("start_pose").value]
        self._start_gripper = float(gp("start_gripper").value)
        self._act_host = str(gp("act_host").value)
        self._act_port = int(gp("act_port").value)
        self._act_timeout_s = float(gp("act_timeout_s").value)
        self._transport = str(gp("inference_transport").value).lower()
        self._grpc_port = int(gp("grpc_port").value)
        self._camera_width = int(gp("camera_width").value)
        self._camera_height = int(gp("camera_height").value)
        self._max_camera_skew_s = float(gp("max_camera_skew_s").value)
        self._max_jpeg_bytes = int(gp("max_jpeg_bytes").value)
        self._lo = [float(x) for x in gp("joint_limits_lo").value]
        self._hi = [float(x) for x in gp("joint_limits_hi").value]
        self._max_dev = float(gp("max_dev_rad").value)
        self._obs_timeout_s = float(gp("obs_timeout_s").value)
        self._auto_start = bool(gp("auto_start_on_stream").value)
        cam1_topic = str(gp("cam1_topic").value)
        cam2_topic = str(gp("cam2_topic").value)
        if self._transport not in ("zmq", "grpc"):
            raise ValueError("inference_transport must be 'zmq' or 'grpc'")

        # Validate array param lengths early -- a wrong-length limit vector would
        # silently disable a joint's clamp.
        for name, vec in (
            ("start_pose", self._start_pose),
            ("joint_limits_lo", self._lo),
            ("joint_limits_hi", self._hi),
        ):
            if len(vec) != _N:
                raise ValueError(
                    f"param '{name}' must have {_N} entries, got {len(vec)}"
                )

        # --- Latest observation store (None until first message) ---------
        self._live_q = None        # 6 arm joints (rad), UR order, from /joint_states
        self._grip_pos = None      # gripper position percent (0..1)
        self._cam1_jpeg = None     # raw JPEG bytes, cam1
        self._cam2_jpeg = None     # raw JPEG bytes, cam2
        self._cam1_ros_stamp_ns = 0
        self._cam2_ros_stamp_ns = 0
        # Arrival time (time.monotonic()) of each obs; None until first message.
        # Used by the EXECUTE freshness watchdog to catch a frozen/hung stream.
        self._live_q_t = None
        self._grip_pos_t = None
        self._cam1_t = None
        self._cam2_t = None

        # HOLD pose (boot: start_pose; ~/hold overwrites with the live pose).
        self._hold_pose = list(self._start_pose)
        # HOLD gripper (boot: start_gripper=open; ~/hold overwrites with the LAST
        # commanded grip so pausing mid-grasp doesn't drop the object).
        self._hold_gripper = self._start_gripper
        self._last_grip_cmd = self._start_gripper

        self._state = HOLD
        self._arming_generation = 0
        self._arming_origin = HOLD
        self._arming_result = None

        # --- Publishers (EXACT synthetic-leader contract) ----------------
        # Plain depth-10 publishers (default QoS), matching gello_publisher_node.
        self._js_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        # Public, read-only state used by robotless integration validation and
        # operator diagnostics.  In particular, command values alone cannot
        # distinguish asynchronous gRPC ARMING from EXECUTE when a policy
        # legitimately returns the held start pose.
        self._state_pub = self.create_publisher(String, "~/state", 10)
        self._grip_pub = self.create_publisher(
            Float32, "/robotiq_gripper/command_percent", 10
        )

        # --- Subscriptions (observation sources; match recorder depths) --
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 100)
        self.create_subscription(
            Float32, "/robotiq_gripper/position_percent", self._on_grip_pos, 20
        )
        self.create_subscription(CompressedImage, cam1_topic, self._on_cam1, 10)
        self.create_subscription(CompressedImage, cam2_topic, self._on_cam2, 10)
        # Optional auto-start trigger: detect the bridge streaming commands.
        self._auto_start_fired = False
        self.create_subscription(
            Float64MultiArray,
            "/forward_position_controller/commands",
            self._on_fpc_commands,
            50,
        )

        # --- Services (Trigger) ------------------------------------------
        self.create_service(Trigger, "~/start_execution", self._srv_start_execution)
        self.create_service(Trigger, "~/hold", self._srv_hold)

        # --- ZMQ REQ client ----------------------------------------------
        self._zmq_ctx = zmq.Context.instance()
        self._sock = None
        self._grpc_worker = None
        if self._transport == "zmq":
            self._connect_socket()
        else:
            self._grpc_contract = ServerContract(
                str(gp("expected_model_id").value),
                str(gp("expected_checkpoint_revision").value),
                str(gp("expected_scheduler").value),
                int(gp("expected_inference_steps").value),
                int(gp("expected_action_steps").value),
                int(gp("expected_resize_height").value),
                int(gp("expected_resize_width").value),
            )
            self._grpc_worker = create_worker(
                f"{self._act_host}:{self._grpc_port}",
                client_id=self.get_name(),
                rpc_deadline_s=self._act_timeout_s,
                max_response_age_s=self._obs_timeout_s,
                max_camera_skew_s=self._max_camera_skew_s,
                max_jpeg_bytes=self._max_jpeg_bytes,
            )
            self._grpc_worker.start()

        # --- Timer (single timer drives BOTH publishes each tick) --------
        rate = self._publish_rate_hz
        if rate <= 0.0:
            self.get_logger().warn(
                f"publish_rate_hz={rate} invalid; falling back to 30.0 Hz"
            )
            rate = 30.0
        self._timer = self.create_timer(1.0 / rate, self._on_timer)

        endpoint = (
            f"{self._act_host}:{self._grpc_port} (gRPC)"
            if self._transport == "grpc"
            else f"tcp://{self._act_host}:{self._act_port} (ZMQ)"
        )
        self.get_logger().info(
            f"policy_leader_node up in HOLD @ {rate:.1f} Hz. "
            f"Inference server {endpoint}, "
            f"timeout={self._act_timeout_s}s, max_dev={self._max_dev} rad, "
            f"auto_start_on_stream={self._auto_start}. "
            f"Holding start_pose; call ~/start_execution to begin."
        )

    # ---- ZMQ socket lifecycle ------------------------------------------
    def _connect_socket(self):
        """(Re)create the REQ socket. REQ's strict send/recv lockstep means a
        timed-out recv leaves the socket unusable, so we tear it down and rebuild
        on every failure rather than trying to reuse it."""
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:  # noqa: BLE001
                pass
        sock = self._zmq_ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self._act_timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self._act_timeout_s * 1000))
        sock.connect(f"tcp://{self._act_host}:{self._act_port}")
        self._sock = sock

    def _zmq_roundtrip(self, req_frames):
        """Send a multipart REQ and return the reply frames, or raise on any
        timeout/error. Rebuilds the socket on failure so the next call is clean."""
        try:
            self._sock.send_multipart(req_frames)
            return self._sock.recv_multipart()
        except zmq.error.Again as exc:  # timeout
            self._connect_socket()
            raise TimeoutError(f"ZMQ timeout after {self._act_timeout_s}s") from exc
        except zmq.ZMQError as exc:
            self._connect_socket()
            raise RuntimeError(f"ZMQ error: {exc}") from exc

    # ---- subscription callbacks ----------------------------------------
    def _on_joint_states(self, msg: JointState):
        pos = obs_assembler.positions_in_ur_order(msg)
        if pos is not None:
            self._live_q = pos
            self._live_q_t = time.monotonic()

    def _on_grip_pos(self, msg: Float32):
        self._grip_pos = float(msg.data)
        self._grip_pos_t = time.monotonic()

    def _on_cam1(self, msg: CompressedImage):
        self._cam1_jpeg = bytes(msg.data)
        self._cam1_t = time.monotonic()
        self._cam1_ros_stamp_ns = self._stamp_ns(msg)

    def _on_cam2(self, msg: CompressedImage):
        self._cam2_jpeg = bytes(msg.data)
        self._cam2_t = time.monotonic()
        self._cam2_ros_stamp_ns = self._stamp_ns(msg)

    @staticmethod
    def _stamp_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _stale_obs(self, now):
        """Return a list of (name) for observations that are missing OR older than
        obs_timeout_s. Empty list => all obs are present and fresh."""
        stale = []
        for name, value, stamp in (
            ("joint_states", self._live_q, self._live_q_t),
            ("grip_pos", self._grip_pos, self._grip_pos_t),
            ("cam1", self._cam1_jpeg, self._cam1_t),
            ("cam2", self._cam2_jpeg, self._cam2_t),
        ):
            if value is None or stamp is None or (now - stamp) > self._obs_timeout_s:
                stale.append(name)
        return stale

    def _on_fpc_commands(self, msg: Float64MultiArray):
        # Used ONLY to detect the bridge has started streaming (optional auto-start).
        if not self._auto_start or self._auto_start_fired:
            return
        if self._state != HOLD:
            return
        self._auto_start_fired = True
        self.get_logger().info(
            "auto_start_on_stream: bridge is streaming -> attempting start_execution."
        )
        ok, msg_text = self._try_start_execution()
        if not ok:
            self.get_logger().warn(f"auto-start refused: {msg_text}")

    # ---- services -------------------------------------------------------
    def _srv_start_execution(self, request, response):
        ok, msg_text = self._try_start_execution()
        response.success = ok
        response.message = msg_text
        return response

    def _srv_hold(self, request, response):
        # Republish the current live pose held (fall back to start_pose if unseen).
        if self._live_q is not None:
            self._hold_pose = list(self._live_q)
            src = "live pose"
        else:
            self._hold_pose = list(self._start_pose)
            src = "start_pose (no live pose yet)"
        # Hold the LAST commanded gripper, not start_gripper -- pausing mid-grasp must
        # not open the gripper and drop the object (review).
        self._hold_gripper = self._last_grip_cmd
        self._arming_generation += 1
        self._arming_result = None
        self._state = HOLD
        if self._grpc_worker is not None:
            self._grpc_worker.disarm()
        self._auto_start_fired = True  # don't auto-rearm after a manual hold
        self.get_logger().info(
            f"~/hold -> HOLD, holding {src}, gripper={self._hold_gripper:.3f}."
        )
        response.success = True
        response.message = f"HOLD (holding {src}, grip {self._hold_gripper:.3f})"
        return response

    def _try_start_execution(self):
        """Guard the live pose, ZMQ-RESET the server, then enter EXECUTE.

        Returns (ok, message). On refusal/failure the state is unchanged (stays
        HOLD/FAULT) so the operator can safely retry."""
        if self._state == ARMING:
            return False, "refused: gRPC server check/reset already in progress"
        if self._live_q is None:
            return False, "refused: no live /joint_states yet"
        # Require a COMPLETE, FRESH observation set before arming: otherwise EXECUTE
        # would arm with e.g. no cameras running and never actually move (review:
        # arming must verify the full obs, not just joint states).
        stale = self._stale_obs(time.monotonic())
        if stale:
            return False, (
                f"refused: observations missing/stale {stale} "
                f"(are the RealSense cameras + gripper running?)."
            )
        # UR joint-state publishers may wrap a joint at +/-pi.  Compare periodic
        # equivalents so, for example, +3.088 and -3.195 are treated as the same
        # physical wrist pose rather than as a spurious 2*pi deviation.
        devs = angular_deviations(self._live_q, self._start_pose)
        worst = max(devs)
        if worst > START_GATE_RAD:
            j = devs.index(worst)
            return False, (
                f"refused: live pose not within {START_GATE_RAD} rad of start_pose "
                f"(worst joint {j}: {worst:.3f} rad). Let the handshake converge first."
            )
        if self._transport == "grpc":
            self._arming_generation += 1
            generation = self._arming_generation
            self._arming_origin = self._state
            self._arming_result = None
            self._state = ARMING
            threading.Thread(
                target=self._grpc_arm,
                args=(generation,),
                name="remote-diffusion-arming",
                daemon=True,
            ).start()
            return True, "ARMING (checking/resetting remote inference server)"

        # RESET the legacy ZMQ policy (clears its action queue).
        try:
            reply = self._zmq_roundtrip(obs_assembler.build_reset_request())
            obj = obs_assembler.parse_reply(reply)
        except Exception as exc:  # noqa: BLE001
            return False, f"refused: inference RESET failed ({exc})"
        if not obj.get(obs_assembler.KEY_OK, False):
            return False, f"refused: server RESET returned ok:false ({obj})"
        self._state = EXECUTE
        self.get_logger().info("~/start_execution: RESET ok -> EXECUTE.")
        return True, "EXECUTE"

    def _grpc_arm(self, generation):
        try:
            self._grpc_worker.get_server_info(self._grpc_contract)
            if generation != self._arming_generation:
                return
            self._grpc_worker.reset_episode()
            result = (generation, True, "remote server ready; RESET ok")
        except Exception as exc:  # noqa: BLE001 - reported by timer thread
            result = (generation, False, str(exc))
        if generation == self._arming_generation and self._state == ARMING:
            self._arming_result = result
        elif self._state != ARMING:
            # HOLD/FAULT may have cancelled ARMING while ResetEpisode was on
            # the wire. Ensure that late success cannot leave the worker armed.
            self._grpc_worker.disarm()

    # ---- timer / state machine -----------------------------------------
    def _on_timer(self):
        self._state_pub.publish(String(data=self._state))
        if self._state == HOLD:
            self._tick_hold()
        elif self._state == ARMING:
            self._tick_arming()
        elif self._state == EXECUTE:
            self._tick_execute()
        else:  # FAULT: fail-silent, publish nothing.
            return

    def _tick_arming(self):
        result = self._arming_result
        if result is None:
            if self._arming_origin == HOLD:
                self._tick_hold()
            return
        generation, ok, detail = result
        self._arming_result = None
        if generation != self._arming_generation:
            return
        if ok:
            self._state = EXECUTE
            self.get_logger().info(f"gRPC ARMING complete -> EXECUTE ({detail}).")
        else:
            self._state = self._arming_origin
            self.get_logger().error(f"gRPC ARMING failed: {detail}")

    def _tick_hold(self):
        # Publish the held pose + held gripper constantly. Never query server.
        self._publish_arm(self._hold_pose)
        self._publish_gripper(self._hold_gripper)

    def _tick_execute(self):
        # FAULT if any observation is missing or stale (frozen camera / hung stream):
        # driving the policy on a stale frame is unsafe, and a bare skip would let the
        # arm silently auto-resume when the stream returns (review: obs watchdog).
        stale = self._stale_obs(time.monotonic())
        if stale:
            self._enter_fault(f"observation missing/stale {stale} (>{self._obs_timeout_s}s)")
            return

        # Keep observations in the checkpoint/dataset branch defined by start_pose.
        # The same representation is used by the max-deviation clamp below, so a
        # wrapped /joint_states sample cannot corrupt either inference or safety.
        live_q = positions_near_reference(self._live_q, self._start_pose)
        state = live_q + [float(self._grip_pos)]
        if self._transport == "grpc":
            error = self._grpc_worker.error()
            if error is not None:
                self._enter_fault(error)
                return
            try:
                result = self._grpc_worker.take_result(self._obs_timeout_s)
                oldest_arrival = min(
                    self._live_q_t, self._grip_pos_t, self._cam1_t, self._cam2_t
                )
                observation = ObservationSnapshot(
                    created_monotonic_ns=int(oldest_arrival * 1e9),
                    state=tuple(state),
                    cam1=ImageSnapshot(
                        self._cam1_ros_stamp_ns, self._camera_width,
                        self._camera_height, self._cam1_jpeg,
                    ),
                    cam2=ImageSnapshot(
                        self._cam2_ros_stamp_ns, self._camera_width,
                        self._camera_height, self._cam2_jpeg,
                    ),
                )
                self._grpc_worker.submit(observation)
            except Exception as exc:  # noqa: BLE001 -- any failure => FAULT
                self._enter_fault(str(exc))
                return
            if result is None:
                return
            action = result.action
        else:
            req = obs_assembler.build_act_request(state, self._cam1_jpeg, self._cam2_jpeg)
            try:
                reply = self._zmq_roundtrip(req)
                action = obs_assembler.parse_action_reply(reply)
            except Exception as exc:  # noqa: BLE001 -- any failure => FAULT
                self._enter_fault(str(exc))
                return

        # --- SAFETY CLAMP (order matters; BUILD_SPEC §5) -----------------
        target = list(action[:_N])
        clamped_limit = False
        clamped_dev = False
        for i in range(_N):
            # (a) per-joint clip to the 1.2x dataset envelope.
            v = target[i]
            if v < self._lo[i]:
                v = self._lo[i]
                clamped_limit = True
            elif v > self._hi[i]:
                v = self._hi[i]
                clamped_limit = True
            # (b) per-joint clip so |target - live_q| <= max_dev_rad.
            lo_d = live_q[i] - self._max_dev
            hi_d = live_q[i] + self._max_dev
            if v < lo_d:
                v = lo_d
                clamped_dev = True
            elif v > hi_d:
                v = hi_d
                clamped_dev = True
            target[i] = v

        if clamped_limit or clamped_dev:
            reasons = []
            if clamped_limit:
                reasons.append("joint_limits (OOD)")
            if clamped_dev:
                reasons.append(f"max_dev>{self._max_dev}rad")
            self.get_logger().warn(
                f"SAFETY CLAMP engaged: {', '.join(reasons)}.",
                throttle_duration_sec=1.0,
            )

        grip = action[_N]
        grip = 0.0 if grip < 0.0 else (1.0 if grip > 1.0 else grip)

        self._publish_arm(target)
        self._publish_gripper(grip)

    def _enter_fault(self, reason):
        self._arming_generation += 1
        self._arming_result = None
        self._state = FAULT
        if self._grpc_worker is not None:
            self._grpc_worker.disarm()
        self.get_logger().error(
            f"FAULT: {reason}. STOPPING /gello/joint_states publishing "
            f"(bridge staleness watchdog will halt the arm). "
            f"Re-arm with ~/start_execution."
        )

    # ---- publish helpers (EXACT contract) ------------------------------
    def _publish_arm(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = UR_JOINT_ORDER
        msg.position = [float(p) for p in positions]
        # velocity / effort left EMPTY.
        self._js_pub.publish(msg)

    def _publish_gripper(self, value):
        msg = Float32()
        msg.data = float(value)
        self._grip_pub.publish(msg)
        self._last_grip_cmd = float(value)

    # ---- shutdown -------------------------------------------------------
    def destroy_node(self):
        if self._grpc_worker is not None:
            self._grpc_worker.close()
        try:
            if self._sock is not None:
                self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyLeaderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
