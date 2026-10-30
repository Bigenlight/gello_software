"""Opt-in instrumentation for the existing ZMQ PolicyLeaderNode.

The factory allows non-ROS tests against a fake base. It never changes policy
requests, replies, safety limits, controller targets, or policy random keys.
"""
import json
import time
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace
from recording import EpisodeWriter


def launch_metadata():
    meta = {"render": True, "method": "svf", "profile": "ifqlparity",
            "run_dir": os.environ.get("SVF_RUN_DIR"), "step": os.environ.get("SVF_STEP"),
            "files": {}}
    run = Path(os.environ["SVF_RUN_DIR"])
    files = [run / "flags.json", run / f"params_{os.environ['SVF_STEP']}.pkl",
             Path(os.environ["SVF_NORM_STATS"]), Path(os.environ["IFQL_PARAMS_FILE"])]
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        meta["files"][str(path)] = digest.hexdigest()
    norm = json.loads(Path(os.environ["SVF_NORM_STATS"]).read_text())
    meta["encoder_identity"] = norm.get("encoder_identity")
    meta["flags"] = json.loads((run / "flags.json").read_text())
    meta["svf_diagnostic"] = os.environ.get("SVF_DIAGNOSTIC_VIDEO", "1") == "1"
    meta["diagnostic_python"] = os.environ.get("SVF_PY", "/home/junhyeong/miniconda3/envs/il/bin/python")
    source = Path(os.environ["SVF_SERVER_PY"]).parent
    runtime = Path(os.environ["SVF_QFLOW_DIR"])
    meta["diagnostic_source"] = str(source)
    meta["diagnostic_runtime"] = str(runtime)
    meta["diagnostic_source_files"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (source, runtime / "agents", runtime / "utils")
        for p in folder.rglob("*.py")}
    for path in Path(__file__).parent.glob("*.py"):
        meta["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def native_camera_period_ns(policy_hz):
    """Camera-native stream policy (2026-09-23). Returns 0 = keep every frame, -1 = do not record,
    or a minimum source-stamp spacing in ns.

    At the 30 Hz policy clock every frame is kept (unchanged). Below it (the 15 Hz deployments)
    native_cam1/2 are NOT recorded by default: the policy's own input frames are stored in cam1/cam2
    at every request and are byte-identical camera frames (user request: drop what is redundant).
    REAL_EVAL_NATIVE_CAM_HZ overrides: "0" = keep every frame, "<hz>" = keep ~<hz> (e.g. "15").
    """
    override = os.environ.get("REAL_EVAL_NATIVE_CAM_HZ")
    if override not in (None, ""):
        hz = float(override)
        return 0 if hz <= 0 or hz >= 29.9 else int(0.75 * 1e9 / hz)
    return 0 if float(policy_hz or 30.0) >= 29.9 else -1


def instrumented_leader(base, execute_state, root, metadata=None):
    class RecordingLeader(base):
        def __init__(self, *args, **kwargs):
            self._eval_writer = None
            self._eval_finished = []
            self._eval_request = 0
            self._eval_raw_action = None
            self._eval_reset_reply = None
            self._eval_native_last = {}
            super().__init__(*args, **kwargs)
            self._eval_native_period_ns = native_camera_period_ns(getattr(self, "_publish_rate_hz", 30.0))

        def _keep_native(self, stream, stamp_ns):
            period = self._eval_native_period_ns
            if period < 0:
                return False
            key = (id(self._eval_writer), stream)
            last = self._eval_native_last.get(key)
            if period and stamp_ns is not None and last is not None and stamp_ns - last < period:
                return False
            self._eval_native_last = {k: v for k, v in self._eval_native_last.items() if k[0] == key[0]}
            self._eval_native_last[key] = stamp_ns
            return True

        def _record(self, stream, payload):
            writer = self._eval_writer
            if writer is not None:
                try:
                    ok = writer.append(stream, payload)
                except Exception as exc:
                    writer.error = str(exc)
                    ok = False
                if not ok:
                    self.get_logger().error("Eval recorder failed or overflowed; episode invalid")
            return writer

        def _close_eval(self, reason):
            writer, self._eval_writer = self._eval_writer, None
            if writer is not None:
                writer.stop(reason)
                self._eval_finished.append(writer)

        def _try_start_execution(self):
            if self._state == execute_state:
                return False, "refused: already executing"
            if any(not w._done.is_set() for w in self._eval_finished):
                return False, "refused: eval recording still finalizing"
            # Disk creation is before the existing arming transaction; no control
            # callback during EXECUTE performs filesystem setup.
            try:
                prepared = EpisodeWriter(root, metadata)
            except Exception as exc:
                return False, f"refused: eval recorder unavailable ({exc})"
            try:
                result = super()._try_start_execution()
            except Exception:
                prepared.stop("start_exception")
                self._eval_finished.append(prepared)
                raise
            if not result[0] or self._state != execute_state:
                prepared.stop("start_rejected")
                self._eval_finished.append(prepared)
            else:
                prepared.start_ns = time.monotonic_ns()
                prepared.start_wall_ns = time.time_ns()
                prepared.metadata["accepted_start"] = True
                prepared.metadata["native_camera_min_spacing_ns"] = self._eval_native_period_ns
                prepared.metadata["reset_reply"] = self._eval_reset_reply
                if self._eval_reset_reply and self._eval_reset_reply.get("log_dir"):
                    prepared.metadata["server_npz"] = str(
                        Path(self._eval_reset_reply["log_dir"]) /
                        f"ep_{int(self._eval_reset_reply['reset_counter']):04d}.npz")
                self._eval_writer = prepared
                self.get_logger().info(f"Recording started: {prepared.path}")
                self._record("lifecycle", {"event": "accepted_start",
                                           "accepted_ns": prepared.start_ns})
            return result

        def _on_timer(self):
            writer = self._eval_writer
            if writer is not None and (writer.error or writer.dropped):
                self._record("lifecycle", {"event": "recording_error_hold"})
                self._close_eval("recording_error")
                self._srv_hold(None, SimpleNamespace(success=False, message=""))
                return
            return super()._on_timer()

        def _srv_go_to_start(self, request, response):
            self._close_eval("go_to_start")
            return super()._srv_go_to_start(request, response)

        def _srv_hold(self, request, response):
            # Freeze data interval before base enters HOLD and schedules RESET.
            # stop() only sets an event; no disk flush or join here.
            self._close_eval("operator_hold")
            return super()._srv_hold(request, response)

        def _enter_fault(self, reason):
            self._record("lifecycle", {"event": "fault", "reason": reason})
            self._close_eval("fault")
            return super()._enter_fault(reason)

        def _zmq_roundtrip(self, frames):
            writer = self._eval_writer
            self._eval_request += 1
            request_id = self._eval_request
            start = time.monotonic_ns()
            if writer is not None:
                self._record("request", {"request_id": request_id,
                                          "control": json.loads(frames[0]),
                                          "cam1_source_ns": getattr(self, "_cam1_ros_stamp_ns", None),
                                          "cam2_source_ns": getattr(self, "_cam2_ros_stamp_ns", None),
                                          "live_q": getattr(self, "_live_q", None),
                                          "observation_arrival_monotonic_s": {
                                              k: getattr(self, k, None) for k in
                                              ("_live_q_t", "_grip_pos_t", "_cam1_t", "_cam2_t")}})
                for stream, payload in zip(("cam1", "cam2"), frames[1:]):
                    writer.append_image(stream, payload, start)
            try:
                result = super()._zmq_roundtrip(frames)
            except Exception as exc:
                self._record("reply_error", {"request_id": request_id, "error": str(exc)})
                raise
            parsed = json.loads(result[0])
            if json.loads(frames[0]).get("cmd") == "reset":
                self._eval_reset_reply = parsed
            self._eval_raw_action = parsed.get("action")
            if writer is not None and writer is self._eval_writer:
                self._record("reply", {"request_id": request_id,
                                        "duration_ns": time.monotonic_ns() - start,
                                        "control": json.loads(result[0])})
            return result

        def _publish_arm(self, positions):
            result = super()._publish_arm(positions)
            if self._state == execute_state:
                payload = {"request_id": self._eval_request,
                           "position": list(positions),
                           "policy_action": self._eval_raw_action,
                           "clamp_delta": ([float(a) - float(b) for a, b in
                                            zip(positions, self._eval_raw_action[:6])]
                                           if self._eval_raw_action else None)}
                # TCP Z-floor fields (2026-09-21): present only when the leader
                # evaluates the floor (tcp_z_clamp_enabled); older leaders lack them.
                if getattr(self, "_tcp_floor_enabled", False):
                    payload.update({"tcp_z_raw": self._last_tcp_z_raw,
                                    "tcp_z": self._last_tcp_z,
                                    "tcp_alpha": self._last_tcp_alpha,
                                    "tcp_clamped": bool(self._last_tcp_clamped)})
                self._record("leader_command", payload)
            return result

        def _on_grip_pos(self, msg):
            result = super()._on_grip_pos(msg)
            if self._state == execute_state:
                self._record("gripper_state", {"position": float(msg.data)})
            return result

        def _on_cam1(self, msg):
            result = super()._on_cam1(msg)
            if self._eval_writer is not None and self._keep_native("cam1", self._cam1_ros_stamp_ns):
                self._eval_writer.append_image("native_cam1", bytes(msg.data))
                self._record("native_cam1_meta", {"source_stamp_ns": self._cam1_ros_stamp_ns})
            return result

        def _on_cam2(self, msg):
            result = super()._on_cam2(msg)
            if self._eval_writer is not None and self._keep_native("cam2", self._cam2_ros_stamp_ns):
                self._eval_writer.append_image("native_cam2", bytes(msg.data))
                self._record("native_cam2_meta", {"source_stamp_ns": self._cam2_ros_stamp_ns})
            return result

        def _publish_gripper(self, value):
            result = super()._publish_gripper(value)
            if self._state == execute_state:
                self._record("gripper_command", {"request_id": self._eval_request,
                                                  "value": float(value)})
            return result

        def _on_joint_states(self, msg):
            result = super()._on_joint_states(msg)
            if self._state == execute_state:
                self._record("robot_state", {
                    "name": list(msg.name), "position": list(msg.position),
                    "velocity": list(msg.velocity), "effort": list(msg.effort),
                    "source_stamp_ns": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec})
            return result

        def _on_fpc_commands(self, msg):
            result = super()._on_fpc_commands(msg)
            if self._state == execute_state:
                self._record("controller_command", {"data": list(msg.data)})
            return result

        def destroy_node(self):
            self._close_eval("shutdown")
            result = super().destroy_node()
            for writer in self._eval_finished:
                try:
                    writer.wait(10)
                except Exception as exc:
                    print(f"Recording finalization error: {exc}")
            return result

    return RecordingLeader
