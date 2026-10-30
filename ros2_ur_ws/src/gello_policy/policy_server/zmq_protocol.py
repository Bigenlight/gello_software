#!/usr/bin/env python3
"""Canonical constants for the localhost ZMQ link between the py3.10 rclpy
`policy_leader_node` (REQ client) and the py3.12 `act_server` (REP server).

This module is intentionally tiny and dependency-free (stdlib only) so that BOTH
sides -- which live in different virtualenvs / python versions -- can import the
exact same frame keys, command names and endpoint defaults. See BUILD_SPEC.md §3.

Wire format (ZMQ REQ/REP, MULTIPART, stdlib-json control frame, raw JPEG bytes):

  Request  (ROS node -> server)
    RESET:  [ b'{"cmd":"reset"}' ]
    ACT:    [ b'{"cmd":"act","state":[f,f,f,f,f,f,f]}', <cam1_jpeg>, <cam2_jpeg> ]

  Reply    (server -> ROS node), always a SINGLE json frame
    RESET ok: [ b'{"ok":true}' ]
    ACT ok:   [ b'{"ok":true,"action":[f,f,f,f,f,f,f]}' ]   # q1..q6 rad + grip 0..1
    error:    [ b'{"ok":false,"err":"..."}' ]
"""

# ---- Endpoint (override via CLI/env/ROS param on both sides) -----------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5591


def default_endpoint(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Build the tcp:// endpoint string used by both bind() and connect()."""
    return f"tcp://{host}:{port}"


# ---- Control-frame command values (JSON "cmd" field) ------------------------
CMD_RESET = "reset"
CMD_ACT = "act"

# ---- Control-frame JSON keys ------------------------------------------------
KEY_CMD = "cmd"        # request: which command ("reset" | "act")
KEY_STATE = "state"    # request (act): list[7] float -> observation.state
KEY_OK = "ok"          # reply: bool success flag
KEY_ACTION = "action"  # reply (act): list[7] float -> q1..q6 (rad) + grip (0..1)
KEY_ERR = "err"        # reply: error string when ok=false

# ---- Multipart layout -------------------------------------------------------
# ACT request has exactly 3 frames: [control_json, cam1_jpeg, cam2_jpeg].
ACT_REQUEST_NFRAMES = 3

# ---- Observation dict keys the policy expects (dataset feature names) --------
# These MUST match the trained checkpoint's config.json input_features exactly.
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"

# ---- Fixed dimensions -------------------------------------------------------
STATE_DIM = 7   # [q1..q6, grip_pos]
ACTION_DIM = 7  # [q1..q6, grip_cmd]
