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
                      within ~0.1 rad of start_pose, then ZMQ-RESETs the server and
                      checks the v2 handshake (action_space / state_dim / action_dim)
                      against this node's configuration.
  ~/hold            : -> HOLD, republishing the current live pose held constant.

Action spaces (``action_space`` parameter, CARROT_EEF_DESIGN.md §3):
  "joint"     (default) -- the reply IS the joint target: q1..q6 + grip. UNCHANGED,
              byte for byte, from the banana/cube ACT + FM + Diffusion deploys.
  "eef_delta" -- the observation is the 16-D EEF state and the reply is a Cartesian
              delta; ``eef_executor.EefDeltaExecutor`` accumulates it on a target
              TCP pose, clamps it (step / lead / workspace box, all
              norm-proportional) and inverts it to joints. The resulting joint
              target then goes through the SAME joint envelope + max-deviation
              clamps and the SAME publishers -- the Cartesian layer is strictly
              upstream of the existing position safety, never a replacement for it.
              ZMQ only: the gRPC remote-diffusion transport carries a joint action.

Distro note: written to be distro-agnostic (std sensor_msgs/std_msgs/std_srvs +
rclpy). Verified to construct under ROS2 Jazzy on the dev PC; the target robot PC
runs Humble -- no Jazzy-only APIs are used (see module docstring caveats below).
"""

import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import Trigger

import zmq

from gello_policy import obs_assembler
from gello_policy.joint_angles import angular_deviations, positions_near_reference
from gello_policy.obs_assembler import UR_JOINT_ORDER

# EEF mode only. Imported defensively so a missing numpy / ur_gello_bringup can
# never break the joint path that drives the robot today; eef mode refuses to
# construct with the captured error instead.
try:
    from gello_policy import eef_space
    from gello_policy.eef_executor import EefDeltaExecutor, EefExecutorConfig
    _EEF_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - reported at construction, not import
    eef_space = None
    EefDeltaExecutor = None
    EefExecutorConfig = None
    _EEF_IMPORT_ERROR = _exc
from gello_policy.remote_diffusion_client import (
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

# --- Action spaces (parameter values, NOT wire ids) --------------------------
# The wire id for eef mode is "eef_delta_v1" (obs_assembler.EEF_ACTION_SPACE_ID /
# eef_space.ACTION_SPACE_ID) and lives in `expected_action_space`, so the policy
# space can be versioned without renaming the operator-facing mode switch.
ACTION_SPACE_JOINT = "joint"
ACTION_SPACE_EEF = "eef_delta"


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
        # --- EEF-delta mode (CARROT_EEF_DESIGN.md §3; names are a contract with
        # the config yamls -- do not rename). All are inert when
        # action_space == "joint".
        self.declare_parameter("action_space", ACTION_SPACE_JOINT)
        self.declare_parameter("expected_action_space", obs_assembler.EEF_ACTION_SPACE_ID)
        self.declare_parameter("eef_max_step_m", 0.02)
        self.declare_parameter("eef_max_step_rad", 0.10)
        self.declare_parameter("eef_max_lead_m", 0.06)
        self.declare_parameter("eef_max_lead_rad", 0.30)
        self.declare_parameter("workspace_lo", [0.35, -0.30, 0.00])
        self.declare_parameter("workspace_hi", [0.78, 0.45, 0.40])
        self.declare_parameter("eef_ik_fail_fault_n", 10)
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
        self._action_space = str(gp("action_space").value).strip().lower()
        self._expected_action_space = str(gp("expected_action_space").value)
        if self._transport not in ("zmq", "grpc"):
            raise ValueError("inference_transport must be 'zmq' or 'grpc'")
        if self._action_space not in (ACTION_SPACE_JOINT, ACTION_SPACE_EEF):
            raise ValueError(
                f"action_space must be '{ACTION_SPACE_JOINT}' or '{ACTION_SPACE_EEF}', "
                f"got {self._action_space!r}"
            )
        self._eef_mode = self._action_space == ACTION_SPACE_EEF
        if self._eef_mode and self._transport != "zmq":
            # The gRPC RemoteDiffusion contract carries a 7-D JOINT action and has
            # no chunk_start / handshake fields; there is no way to run eef mode
            # over it. Refuse loudly at construction rather than silently feeding
            # Cartesian deltas to the joint clamps.
            raise ValueError(
                "action_space 'eef_delta' requires inference_transport 'zmq' "
                "(the gRPC remote-inference path is joint-space only)"
            )
        if self._eef_mode and eef_space is None:
            raise ValueError(
                f"action_space 'eef_delta' needs gello_policy.eef_space "
                f"(numpy + ur_gello_bringup.ur_kin), which failed to import: "
                f"{_EEF_IMPORT_ERROR}"
            )

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

        # Observation/action widths on the wire. Joint mode keeps the historic 7;
        # eef mode starts at eef_space's 16 and is then re-confirmed (never
        # silently widened) by the RESET handshake before the first act request.
        self._state_len = obs_assembler.STATE_LEN
        self._action_len = obs_assembler.ACTION_LEN
        self._expected_wire_space = obs_assembler.JOINT_ACTION_SPACE_ID
        self._eef = None
        if self._eef_mode:
            self._state_len = eef_space.STATE_DIM
            self._action_len = eef_space.ACTION_DIM
            self._expected_wire_space = self._expected_action_space
            self._eef = EefDeltaExecutor(EefExecutorConfig(
                start_pose=tuple(self._start_pose),
                max_step_m=float(gp("eef_max_step_m").value),
                max_step_rad=float(gp("eef_max_step_rad").value),
                max_lead_m=float(gp("eef_max_lead_m").value),
                max_lead_rad=float(gp("eef_max_lead_rad").value),
                workspace_lo=tuple(float(x) for x in gp("workspace_lo").value),
                workspace_hi=tuple(float(x) for x in gp("workspace_hi").value),
                ik_fail_fault_n=int(gp("eef_ik_fail_fault_n").value),
            ))
            if self._eef.start_branch < 0:
                # branch_id returns -1 when the analytic IK cannot invert the start
                # pose at all. ik_tcp_nearest then runs UNLOCKED (any branch), so
                # say so instead of pretending the lock is in place.
                self.get_logger().warn(
                    "start_pose has no closed-form IK branch (branch_id=-1): EEF "
                    "targets will be inverted WITHOUT a branch lock. Check start_pose."
                )

        self._state = HOLD
        self._arming_generation = 0
        self._arming_origin = HOLD
        self._arming_result = None

        # --- Publishers (EXACT synthetic-leader contract) ----------------
        # Plain depth-10 publishers (default QoS), matching gello_publisher_node.
        self._js_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
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
        if self._eef_mode:
            c = self._eef.cfg
            self.get_logger().info(
                f"action_space=eef_delta (expect {self._expected_wire_space}, "
                f"state {self._state_len}-D): step<={c.max_step_m} m/{c.max_step_rad} rad, "
                f"lead<={c.max_lead_m} m/{c.max_lead_rad} rad, box {list(c.workspace_lo)}"
                f"..{list(c.workspace_hi)}, IK branch {self._eef.start_branch}, "
                f"fault after {c.ik_fail_fault_n} consecutive IK failures."
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
        if self._eef is not None:
            # Drop the accumulated Cartesian target: after a hold the arm is
            # wherever the operator left it, and a stale target would be executed
            # as one large jump on the next arm.
            self._eef.reset()
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
            handshake = obs_assembler.parse_reset_reply(reply)
        except Exception as exc:  # noqa: BLE001
            return False, f"refused: inference RESET failed ({exc})"
        if not handshake[obs_assembler.KEY_OK]:
            return False, f"refused: server RESET returned ok:false ({handshake['raw']})"
        ok, detail = self._check_handshake(handshake)
        if not ok:
            return False, f"refused: {detail}"
        if self._eef is not None:
            self._eef.reset(positions_near_reference(self._live_q, self._start_pose))
        self._state = EXECUTE
        self.get_logger().info(f"~/start_execution: RESET ok -> EXECUTE{detail}.")
        return True, "EXECUTE"

    def _check_handshake(self, handshake):
        """Verify a v2 RESET reply describes the checkpoint we are configured for.

        Returns (ok, detail); detail is a refusal reason when ok is False and a
        short log suffix otherwise. Design §3.7: "a joint checkpoint can never
        drive eef mode or vice versa".

        Missing fields are handled asymmetrically ON PURPOSE:
          * eef mode REFUSES a reply without ``action_space``/dims -- that is a
            pre-v2 server, and a 7-D joint checkpoint answering a 16-D request
            would be a silent shape error at best and a wrong-space motion at
            worst;
          * joint mode ignores absent fields (every server that exists today
            predates the handshake, and this path must stay byte-for-byte as it
            is) but still refuses a server that explicitly declares a space or a
            state width that is not joint/7.
        """
        space = handshake[obs_assembler.KEY_ACTION_SPACE]
        state_dim = handshake[obs_assembler.KEY_STATE_DIM]
        action_dim = handshake[obs_assembler.KEY_ACTION_DIM]
        if self._eef_mode and (space is None or state_dim is None or action_dim is None):
            return False, (
                "server RESET reply has no action_space/state_dim/action_dim -- it "
                "predates protocol v2, so it cannot be confirmed to serve "
                f"{self._expected_wire_space!r}. Start the server with "
                f"--action-space {self._expected_wire_space}."
            )
        if space is not None and space != self._expected_wire_space:
            return False, (
                f"server action_space {space!r} != expected "
                f"{self._expected_wire_space!r} (a joint checkpoint can never drive "
                f"eef mode or vice versa)"
            )
        if state_dim is not None and state_dim != self._state_len:
            return False, (
                f"server state_dim {state_dim} != expected {self._state_len} for "
                f"action_space {self._expected_wire_space!r}"
            )
        if action_dim is not None and action_dim != self._action_len:
            return False, (
                f"server action_dim {action_dim} != expected {self._action_len} for "
                f"action_space {self._expected_wire_space!r}"
            )
        if space is None or state_dim is None or action_dim is None:
            # Pre-v2 (or partially v2) server: nothing to adopt. Only reachable in
            # joint mode -- eef mode refused a few lines above.
            return True, ""
        # Adopt the handshaken width: from here on every act request is validated
        # against what the server said, not against a compile-time constant.
        self._state_len = int(state_dim)
        self._action_len = int(action_dim)
        return True, (
            f" (server: action_space={space}, state_dim={state_dim}, "
            f"action_dim={action_dim}, policy={handshake[obs_assembler.KEY_POLICY_TYPE]})"
        )

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
        if self._eef_mode:
            # 16-D: [q1..q6, tcp_xyz, 6-D rotation, grip_pos] (design §2).
            state = [float(v) for v in eef_space.state_vector(live_q, float(self._grip_pos))]
        else:
            state = live_q + [float(self._grip_pos)]
        chunk_start = False
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
            req = obs_assembler.build_act_request(
                state, self._cam1_jpeg, self._cam2_jpeg, state_len=self._state_len
            )
            try:
                reply = self._zmq_roundtrip(req)
                action, chunk_start = obs_assembler.parse_action_reply(
                    reply, action_len=self._action_len
                )
            except Exception as exc:  # noqa: BLE001 -- any failure => FAULT
                self._enter_fault(str(exc))
                return

        # --- EEF delta -> joint target (design §3 items 3-5) -------------
        # Everything below this block is the ORIGINAL joint path: whatever the
        # action space, `action[:6]` has become a joint target by now and the
        # existing envelope / max-deviation clamps are the last word.
        if self._eef_mode:
            q_eef, eef_info = self._eef.step(live_q, action, chunk_start)
            eef_reasons = EefDeltaExecutor.clamp_reasons(eef_info)
            if eef_reasons:
                self.get_logger().warn(
                    f"EEF CLAMP engaged: {', '.join(eef_reasons)}.",
                    throttle_duration_sec=1.0,
                )
            if q_eef is None:
                if eef_info["ik_fault"]:
                    self._enter_fault(
                        f"EEF IK failed on {eef_info['ik_fail_streak']} consecutive "
                        f"ticks (target {eef_info['target_pos']})"
                    )
                    return
                self.get_logger().warn(
                    f"EEF IK failed (streak "
                    f"{eef_info['ik_fail_streak']}/{self._eef.cfg.ik_fail_fault_n}); "
                    f"holding the live pose this tick.",
                    throttle_duration_sec=1.0,
                )
                action = list(live_q) + list(action[_N:])
            else:
                action = [float(v) for v in q_eef] + list(action[_N:])

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
