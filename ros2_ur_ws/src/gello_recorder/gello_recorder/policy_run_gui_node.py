#!/usr/bin/env python3
"""rclpy Node backing the diffusion-policy run recorder GUI.

Subclass of :class:`GelloRecorderGuiNode` (gello_gui_node.py) -- it inherits ALL
of that node's always-active subscriptions, live preview/state snapshot getters,
camera-warmup gating, and start_recording()/stop_recording() take lifecycle
unchanged. On top of that it adds what a *policy inference* run (as opposed to a
human teleop demo) needs:

  1. Two std_srvs/Trigger service CLIENTS for policy_leader_node's private
     ``~/start_execution`` and ``~/hold`` services (the exact same fully
     qualified names camera_viewer.py's button bar calls), so the GUI can arm /
     pause the autonomous policy without a separate ``ros2 service call``
     terminal. This node only *creates* the clients and exposes thread-safe
     ready-check / call_async wrappers -- the non-blocking Future polling state
     machine lives on the Qt side (policy_run_gui.py), mirroring how
     camera_viewer.py polls ``future.done()`` from its cv2 loop. The futures
     complete on the background rclpy spin thread this node already runs in
     (see gello_gui_node.py's thread-safety docstring); the GUI thread only ever
     reads ``.done()``/``.result()``.

  2. ``write_label(take_dir, success)``: writes a small ``label.json``
     (success/fail verdict) into a finished take directory. The write is atomic
     (temp file in the same directory, then ``os.replace``) so a crash mid-write
     can never leave a corrupt / partial label.json behind -- either the old
     state (no file) or the complete new file exists.

Unlike gello_recorder_gui.py this workflow does NOT launch RealSense
subprocesses: in the diffusion deploy the cameras are already running (started
via launch_cameras.sh before the diffusion server / this GUI come up), exactly
the assumption camera_viewer.py makes. This node therefore only *subscribes* to
the existing camera topics (inherited behavior); a second RealSense driver
against the same USB device would conflict with the running one.
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime

import rclpy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from gello_recorder.gello_gui_node import GelloRecorderGuiNode

# Full service names (policy_leader_node declares them as ~/... i.e. private).
# Identical to camera_viewer.py's START_SERVICE / HOLD_SERVICE constants.
START_SERVICE = "/policy_leader_node/start_execution"
HOLD_SERVICE = "/policy_leader_node/hold"
CLASSIFIER_STATUS_TOPIC = "/reward_classifier/status"


class PolicyRunGuiNode(GelloRecorderGuiNode):
    """GelloRecorderGuiNode + policy_leader_node Trigger clients + take labeling."""

    def __init__(
        self,
        cam1_topic: str = "/cam1/cam1/color/image_raw/compressed",
        cam2_topic: str = "/cam2/cam2/color/image_raw/compressed",
        camera_fps: float = 30.0,
        camera_warmup_s: float = 1.0,
        output_root: str = "~/gello_recordings",
        node_name: str = "policy_run_gui_node",
    ) -> None:
        # Parent wires up every subscription, lock, and the recording-session
        # state machine; nothing camera-subprocess-related lives in the node.
        super().__init__(
            cam1_topic=cam1_topic,
            cam2_topic=cam2_topic,
            camera_fps=camera_fps,
            camera_warmup_s=camera_warmup_s,
            output_root=output_root,
            node_name=node_name,
        )

        # Trigger CLIENTS (not servers). Created once here, before the spin
        # thread starts, so no locking is needed around their construction.
        self._start_exec_client = self.create_client(Trigger, START_SERVICE)
        self._hold_client = self.create_client(Trigger, HOLD_SERVICE)
        self._classifier_lock = threading.Lock()
        self._classifier_status = None
        self._classifier_status_t = None
        self.create_subscription(
            String,
            CLASSIFIER_STATUS_TOPIC,
            self._on_classifier_status,
            10,
        )

        self.get_logger().info(
            f"policy_run_gui_node up. trigger clients: {START_SERVICE} , {HOLD_SERVICE}; "
            f"classifier status: {CLASSIFIER_STATUS_TOPIC}"
        )

    def _on_classifier_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            if not isinstance(status, dict):
                raise ValueError("status must be a JSON object")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            status = {
                "ready": False,
                "message": "invalid classifier status: {}".format(exc),
            }
        with self._classifier_lock:
            self._classifier_status = status
            self._classifier_status_t = time.monotonic()

    def get_classifier_status(self) -> dict:
        """Thread-safe classifier snapshot for the Qt refresh timer."""
        with self._classifier_lock:
            status = (
                dict(self._classifier_status)
                if self._classifier_status is not None
                else {
                    "ready": False,
                    "message": "classifier node not connected",
                }
            )
            received = self._classifier_status_t
        status["status_age_s"] = (
            None if received is None else time.monotonic() - received
        )
        return status

    # ---- Trigger-service access (thread-safe; called from the Qt thread) ----
    @staticmethod
    def _client_ready(client) -> bool:
        """service_is_ready(), hardened against the shutdown race exactly like
        camera_viewer.py's _TriggerButton.refresh_ready(): a Ctrl-C/SIGTERM can
        invalidate the rclpy context between the caller's check and this call --
        report "not ready" instead of crashing the GUI."""
        if not rclpy.ok():
            return False
        try:
            return bool(client.service_is_ready())
        except Exception:  # noqa: BLE001 -- shutdown race, not a real fault
            return False

    def start_execution_ready(self) -> bool:
        return self._client_ready(self._start_exec_client)

    def hold_ready(self) -> bool:
        return self._client_ready(self._hold_client)

    def call_start_execution(self):
        """Fire ~/start_execution. Returns the rclpy Future (poll .done())."""
        return self._start_exec_client.call_async(Trigger.Request())

    def call_hold(self):
        """Fire ~/hold. Returns the rclpy Future (poll .done())."""
        return self._hold_client.call_async(Trigger.Request())

    # ---- take labeling ------------------------------------------------------
    def write_label(self, take_dir: str, success: bool) -> str:
        """Atomically write ``take_dir/label.json`` with a success/fail verdict.

        Write-to-temp-then-``os.replace()`` in the SAME directory (os.replace is
        only atomic within one filesystem), so a crash mid-write never leaves a
        corrupt or partial label.json. Returns the final label.json path.
        """
        if not os.path.isdir(take_dir):
            raise RuntimeError(f"take directory does not exist: {take_dir}")

        payload = {
            "result": "success" if success else "fail",
            "success": bool(success),
            "t_wall": time.time(),
            "labeled_at": datetime.now().isoformat(timespec="seconds"),
            "take_dir": os.path.basename(os.path.normpath(take_dir)),
        }

        final_path = os.path.join(take_dir, "label.json")
        fd, tmp_path = tempfile.mkstemp(
            prefix=".label_", suffix=".json.tmp", dir=take_dir
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
        except Exception:
            # Best-effort temp cleanup; the final path was never touched.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self.get_logger().info(f"Labeled take: {final_path} -> {payload['result']}")
        return final_path


def main(args=None):
    """Headless spin, mirroring gello_gui_node.main() -- mostly for debugging;
    the real entry point is policy_run_gui.py which owns the Qt loop."""
    rclpy.init(args=args)
    node = PolicyRunGuiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
