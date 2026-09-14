#!/usr/bin/env python3
"""Pure, testable helpers for the ACT synthetic-leader node.

This module holds everything that does NOT require rclpy or a live ROS graph so
it can be unit-tested standalone (``python3 -m gello_policy.obs_assembler`` runs
a tiny self-check, and ``test_obs_assembler`` exercises ``_reorder``):

  * ``UR_JOINT_ORDER`` + ``_reorder`` -- copied VERBATIM from the recorder
    (gello_recorder/gello_ur_recorder_node.py) so the policy observation uses the
    IDENTICAL reorder-by-name semantics the dataset was recorded with.
  * ZMQ request/reply framing -- packs the multipart REQ frames and parses the
    single-frame JSON replies, matching the wire contract in BUILD_SPEC §3 (and
    the string literals builder-1's ``zmq_protocol.py`` must agree with).

No ``import zmq`` here on purpose: the socket lives in the node, this stays pure.
Only stdlib ``json`` is used, with compact separators so the emitted bytes match
the spec literals exactly (``b'{"cmd":"reset"}'`` -- no spaces).
"""

import json
from collections import namedtuple

# --- Recorder contract (VERBATIM from gello_ur_recorder_node.py) -------------
# Canonical UR joint order; GELLO and the UR driver both publish these names.
UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_N = len(UR_JOINT_ORDER)


def _reorder(msg):
    """Return (pos, vel, eff) lists in UR_JOINT_ORDER, or None if joints missing."""
    idx = {n: i for i, n in enumerate(msg.name)}
    if any(j not in idx for j in UR_JOINT_ORDER):
        return None
    pos = [float(msg.position[idx[j]]) if msg.position else None for j in UR_JOINT_ORDER]
    vel = [float(msg.velocity[idx[j]]) if msg.velocity else None for j in UR_JOINT_ORDER]
    eff = [float(msg.effort[idx[j]]) if msg.effort else None for j in UR_JOINT_ORDER]
    return pos, vel, eff


def positions_in_ur_order(msg):
    """Convenience wrapper: just the 6 positions in UR order, or None if incomplete.

    Returns None if any of the 6 UR joints is absent from ``msg.name`` OR if the
    message carries no position array (mirrors the recorder's "silently dropped"
    semantics -- the policy must never act on a partial joint state).
    """
    r = _reorder(msg)
    if r is None:
        return None
    pos, _, _ = r
    if any(p is None for p in pos):
        return None
    return pos


# --- ZMQ wire protocol (BUILD_SPEC §3; must match builder-1's zmq_protocol.py)-
# Command / key string literals. Kept as module constants so a divergence from
# the server shows up as one obvious edit site (do NOT import across the
# py3.10/py3.12 boundary -- just keep these literals in sync).
CMD_RESET = "reset"
CMD_ACT = "act"
KEY_CMD = "cmd"
KEY_STATE = "state"
KEY_OK = "ok"
KEY_ACTION = "action"
KEY_ERR = "err"
# --- protocol v2 additions (CARROT_EEF_DESIGN.md §4) -------------------------
# ACT reply: true on the tick the network actually re-ran (its action queue was
# empty before select_action), i.e. "this is a fresh chunk". The EEF executor
# re-anchors on it. ABSENT from an old server's reply -> parsed as False, which is
# the safe reading for joint mode (it ignores the flag entirely).
KEY_CHUNK_START = "chunk_start"
# RESET reply handshake: what the loaded checkpoint actually is. All three are
# ABSENT on a pre-v2 server -> parse_reset_reply reports None so the caller can
# tell "old server" apart from "server that disagrees with me".
KEY_ACTION_SPACE = "action_space"
KEY_STATE_DIM = "state_dim"
KEY_ACTION_DIM = "action_dim"
KEY_POLICY_TYPE = "policy_type"
KEY_CHECKPOINT = "checkpoint"

# Action-space ids on the wire. EEF_ACTION_SPACE_ID must equal
# gello_policy.eef_space.ACTION_SPACE_ID; it is repeated as a literal instead of
# imported so this module stays stdlib-only (eef_space pulls in numpy + ur_kin,
# which the py3.12 server side does not necessarily want). test_obs_assembler_v2
# pins the two together.
JOINT_ACTION_SPACE_ID = "joint"
EEF_ACTION_SPACE_ID = "eef_delta_v1"

STATE_LEN = 7   # 6 arm joints (rad) + grip_pos (0..1)  -- joint space, the default
ACTION_LEN = 7  # 6 arm joints (rad) + grip_cmd (0..1)  -- BOTH spaces are 7-D

# Per-space observation/action widths (design §4: the hardcoded 7 becomes a table
# or, better, the handshake). The node prefers the handshaken state_dim; this
# table is what it validates that handshake against.
STATE_LEN_BY_ACTION_SPACE = {
    JOINT_ACTION_SPACE_ID: 7,    # [q1..q6, grip_pos]
    EEF_ACTION_SPACE_ID: 16,     # [q1..q6, tcp_xyz, rot6d, grip_pos]
}
ACTION_LEN_BY_ACTION_SPACE = {
    JOINT_ACTION_SPACE_ID: 7,    # [q1..q6, grip_cmd]
    EEF_ACTION_SPACE_ID: 7,      # [dx,dy,dz, drx,dry,drz, grip_cmd]
}

# parse_action_reply result: a namedtuple so callers can write
# ``action, chunk_start = parse_action_reply(frames)`` or use the fields.
ActionReply = namedtuple("ActionReply", ["action", "chunk_start"])

# Compact separators => emitted bytes match the spec literals exactly
# (b'{"cmd":"reset"}' with no whitespace). The server parses JSON so spacing is
# not load-bearing for it, but matching the literals keeps the contract crisp.
_SEP = (",", ":")


def build_reset_request():
    """Multipart REQ frames for a RESET: [ b'{"cmd":"reset"}' ]."""
    return [json.dumps({KEY_CMD: CMD_RESET}, separators=_SEP).encode("utf-8")]


def build_act_request(state, cam1_jpeg, cam2_jpeg, state_len=STATE_LEN):
    """Multipart REQ frames for an ACT step.

    Frames: [ b'{"cmd":"act","state":[N floats]}', <cam1 jpeg>, <cam2 jpeg> ].
    ``cam*_jpeg`` are the RAW JPEG bytes straight from each CompressedImage msg
    (no decode here).

    ``state_len`` is the width the loaded checkpoint expects: 7 in joint space
    (the default, so every pre-existing caller is unchanged) and 16 in
    ``eef_delta_v1``. The node passes the value it HANDSHOOK with the server at
    RESET, not a compile-time constant, so a checkpoint/config mismatch is caught
    at arming instead of being silently padded or truncated on the wire.
    """
    state_list = [float(x) for x in state]
    if len(state_list) != int(state_len):
        raise ValueError(
            f"ACT state must have {int(state_len)} floats, got {len(state_list)}"
        )
    header = json.dumps(
        {KEY_CMD: CMD_ACT, KEY_STATE: state_list}, separators=_SEP
    ).encode("utf-8")
    return [header, bytes(cam1_jpeg), bytes(cam2_jpeg)]


def parse_reply(frames):
    """Parse a single-frame JSON reply into a dict.

    Returns the decoded dict (with at least an ``ok`` bool). Raises ValueError on
    a malformed / empty / non-JSON reply so the caller can treat it as a fault.
    """
    if not frames:
        raise ValueError("empty ZMQ reply (no frames)")
    try:
        obj = json.loads(bytes(frames[0]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed ZMQ reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("ZMQ reply is not a JSON object")
    return obj


def parse_action_reply(frames, action_len=ACTION_LEN):
    """Parse an ACT reply into ``ActionReply(action, chunk_start)``.

    ``action`` is a list of ``action_len`` floats; ``chunk_start`` is the v2 flag
    (False when the server does not send it -- an old server simply never
    re-anchors a chunk, which is exactly what joint mode wants).

    Raises ValueError if ok is not true, the action is missing, or its length is
    wrong (the caller turns any ValueError into a FAULT).
    """
    obj = parse_reply(frames)
    if not obj.get(KEY_OK, False):
        raise ValueError(f"server returned ok:false ({obj.get(KEY_ERR)})")
    action = obj.get(KEY_ACTION)
    if action is None:
        raise ValueError("ACT reply missing 'action'")
    action = [float(x) for x in action]
    if len(action) != int(action_len):
        raise ValueError(
            f"ACT reply action must have {int(action_len)} floats, got {len(action)}"
        )
    return ActionReply(action, bool(obj.get(KEY_CHUNK_START, False)))


def parse_action_list(frames, action_len=ACTION_LEN):
    """Backward-compatible helper: just the action list, v1 semantics."""
    return parse_action_reply(frames, action_len=action_len).action


def parse_reset_reply(frames):
    """Parse a RESET reply into the v2 handshake dict.

    Returns ``{"ok": bool, "action_space": str|None, "state_dim": int|None,
    "action_dim": int|None, "policy_type": str|None, "checkpoint": str|None,
    "raw": <full reply dict>}``.

    A MISSING key is reported as None, never defaulted: "this server predates the
    handshake" and "this server says joint/7" are different facts, and eef mode
    must refuse the first one rather than assume it agrees (design §3.7). A
    present-but-unparseable ``*_dim`` is also None -- same refusal path, and it
    keeps a malformed reply from raising here where the caller expects a dict.
    """
    obj = parse_reply(frames)

    def _int_or_none(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _str_or_none(value):
        return None if value is None else str(value)

    return {
        KEY_OK: bool(obj.get(KEY_OK, False)),
        KEY_ACTION_SPACE: _str_or_none(obj.get(KEY_ACTION_SPACE)),
        KEY_STATE_DIM: _int_or_none(obj.get(KEY_STATE_DIM)),
        KEY_ACTION_DIM: _int_or_none(obj.get(KEY_ACTION_DIM)),
        KEY_POLICY_TYPE: _str_or_none(obj.get(KEY_POLICY_TYPE)),
        KEY_CHECKPOINT: _str_or_none(obj.get(KEY_CHECKPOINT)),
        "raw": obj,
    }


if __name__ == "__main__":
    # Tiny self-check (no ROS/zmq needed).
    assert build_reset_request() == [b'{"cmd":"reset"}'], build_reset_request()
    frames = build_act_request([1, 2, 3, 4, 5, 6, 0.5], b"\xff\xd8jpg1", b"\xff\xd8jpg2")
    assert frames[0] == b'{"cmd":"act","state":[1.0,2.0,3.0,4.0,5.0,6.0,0.5]}', frames[0]
    assert frames[1] == b"\xff\xd8jpg1" and frames[2] == b"\xff\xd8jpg2"
    assert parse_action_list([b'{"ok":true,"action":[0,1,2,3,4,5,0.9]}']) == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9,
    ]
    act, chunk = parse_action_reply([b'{"ok":true,"action":[0,1,2,3,4,5,0.9],"chunk_start":true}'])
    assert act[-1] == 0.9 and chunk is True
    assert parse_action_reply([b'{"ok":true,"action":[0,1,2,3,4,5,0.9]}']).chunk_start is False
    assert parse_reply([b'{"ok":true}'])["ok"] is True
    hs = parse_reset_reply([b'{"ok":true,"action_space":"eef_delta_v1","state_dim":16,"action_dim":7}'])
    assert hs["action_space"] == "eef_delta_v1" and hs["state_dim"] == 16
    assert parse_reset_reply([b'{"ok":true}'])["action_space"] is None
    frames16 = build_act_request(list(range(16)), b"a", b"b", state_len=16)
    assert frames16[0].startswith(b'{"cmd":"act","state":[0.0,')
    print("obs_assembler self-check OK")
