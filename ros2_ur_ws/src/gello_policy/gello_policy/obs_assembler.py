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

STATE_LEN = 7   # 6 arm joints (rad) + grip_pos (0..1)
ACTION_LEN = 7  # 6 arm joints (rad) + grip_cmd (0..1)

# Compact separators => emitted bytes match the spec literals exactly
# (b'{"cmd":"reset"}' with no whitespace). The server parses JSON so spacing is
# not load-bearing for it, but matching the literals keeps the contract crisp.
_SEP = (",", ":")


def build_reset_request():
    """Multipart REQ frames for a RESET: [ b'{"cmd":"reset"}' ]."""
    return [json.dumps({KEY_CMD: CMD_RESET}, separators=_SEP).encode("utf-8")]


def build_act_request(state, cam1_jpeg, cam2_jpeg):
    """Multipart REQ frames for an ACT step.

    Frames: [ b'{"cmd":"act","state":[7 floats]}', <cam1 jpeg>, <cam2 jpeg> ].
    ``state`` must be length-7 (6 joints rad + grip_pos). ``cam*_jpeg`` are the
    RAW JPEG bytes straight from each CompressedImage msg (no decode here).
    """
    state_list = [float(x) for x in state]
    if len(state_list) != STATE_LEN:
        raise ValueError(
            f"ACT state must have {STATE_LEN} floats, got {len(state_list)}"
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


def parse_action_reply(frames):
    """Parse an ACT reply and return a length-7 action list.

    Raises ValueError if ok is not true, the action is missing, or it is not
    length-7 (the caller turns any ValueError into a FAULT).
    """
    obj = parse_reply(frames)
    if not obj.get(KEY_OK, False):
        raise ValueError(f"server returned ok:false ({obj.get(KEY_ERR)})")
    action = obj.get(KEY_ACTION)
    if action is None:
        raise ValueError("ACT reply missing 'action'")
    action = [float(x) for x in action]
    if len(action) != ACTION_LEN:
        raise ValueError(
            f"ACT reply action must have {ACTION_LEN} floats, got {len(action)}"
        )
    return action


if __name__ == "__main__":
    # Tiny self-check (no ROS/zmq needed).
    assert build_reset_request() == [b'{"cmd":"reset"}'], build_reset_request()
    frames = build_act_request([1, 2, 3, 4, 5, 6, 0.5], b"\xff\xd8jpg1", b"\xff\xd8jpg2")
    assert frames[0] == b'{"cmd":"act","state":[1.0,2.0,3.0,4.0,5.0,6.0,0.5]}', frames[0]
    assert frames[1] == b"\xff\xd8jpg1" and frames[2] == b"\xff\xd8jpg2"
    assert parse_action_reply([b'{"ok":true,"action":[0,1,2,3,4,5,0.9]}']) == [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9,
    ]
    assert parse_reply([b'{"ok":true}'])["ok"] is True
    print("obs_assembler self-check OK")
