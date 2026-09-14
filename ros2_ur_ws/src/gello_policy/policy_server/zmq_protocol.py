#!/usr/bin/env python3
"""Canonical constants for the localhost ZMQ link between the py3.10 rclpy
`policy_leader_node` (REQ client) and the py3.12 inference servers (REP servers:
`act_server`, `fm_server`, `diffusion_server`).

This module is deliberately tiny so that BOTH sides -- which live in different
virtualenvs / python versions -- can import the exact same frame keys, command
names and endpoint defaults. See BUILD_SPEC.md §3 and CARROT_EEF_DESIGN.md §4.

The only non-stdlib dependency is ``gello_policy.eef_space`` (numpy + the ROS-free
``ur_gello_bringup.ur_kin``), imported for the EEF action-space identity and its
state dimension: the whole point of CARROT_EEF_DESIGN §2 is that "eef_delta_v1"
and "16" are defined in exactly ONE place. See ``_import_eef_space`` for why the
import has a path-repair fallback.

Wire format v2 (ZMQ REQ/REP, MULTIPART, stdlib-json control frame, raw JPEG bytes)
==================================================================================
v2 is a strict superset of v1: every v1 key keeps its meaning and every v1 client
still parses a v2 reply (it just ignores the new keys). What v2 adds is a
HANDSHAKE (so a joint checkpoint can never be driven as an EEF policy, or vice
versa) and a CHUNK marker (so the client knows when the network actually re-ran).

  Request  (ROS node -> server)            -- unchanged from v1
    RESET:  [ b'{"cmd":"reset"}' ]
    ACT:    [ b'{"cmd":"act","state":[f, ... ]}', <cam1_jpeg>, <cam2_jpeg> ]
            `state` is STATE_DIM_BY_SPACE[action_space] floats:
              joint         ->  7 = [q1..q6 rad, grip_pos 0..1]
              eef_delta_v1  -> 16 = eef_space.state_vector(q, grip_pos)

  Reply    (server -> ROS node), always a SINGLE json frame
    RESET ok: [ b'{"ok":true,"action_space":"joint","state_dim":7,"action_dim":7,
                   "policy_type":"act","checkpoint":"<path>"}' ]
    ACT ok:   [ b'{"ok":true,"action":[f x7],"chunk_start":true}' ]
              action is ALWAYS 7 floats, but its MEANING depends on action_space:
                joint         -> [q1..q6 rad, grip_cmd 0..1]  (absolute targets)
                eef_delta_v1  -> [dx,dy,dz,drx,dry,drz, grip_cmd]  (base-frame delta)
              chunk_start is true when the policy's internal action queue was EMPTY
              before this select_action, i.e. the network re-ran on THIS observation
              and everything the client pops until the next chunk_start belongs to
              one trajectory predicted from this tick's frames. The EEF executor
              re-anchors its integration target on chunk_start (DESIGN §3.3).
    error:    [ b'{"ok":false,"err":"..."}' ]

  Handshake contract
    The client sends RESET at arming and REFUSES to run unless the reply's
    `action_space` matches what it was configured for and `state_dim`/`action_dim`
    match what it packs/unpacks. The server independently REFUSES TO START (exit 4)
    when `--action-space` contradicts the checkpoint's own feature shapes, so the
    mismatch is caught at launch rather than on the arm.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _import_eef_space():
    """Import ``gello_policy.eef_space`` from any of the three ways we are launched.

    * ``python -m gello_policy.policy_server.act_server`` / colcon-installed ->
      the package is already importable, the plain import wins.
    * ``scripts/run_act_server.sh`` runs the server as a BARE SCRIPT, so sys.path[0]
      is ``.../gello_policy/policy_server`` and neither ``gello_policy`` nor
      ``ur_gello_bringup`` is importable. Rather than duplicate ACTION_SPACE_ID /
      STATE_DIM here (the one thing DESIGN §2 forbids), repair sys.path from this
      file's own location: policy_server/ -> gello_policy/ -> src/, and
      src/ur_gello_bringup for the ROS-free ur_kin that eef_space needs.
    """
    try:
        from gello_policy import eef_space  # type: ignore
        return eef_space
    except ImportError:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_dir = os.path.dirname(pkg_dir)
        for cand in (pkg_dir, os.path.join(src_dir, "ur_gello_bringup")):
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        from gello_policy import eef_space  # type: ignore
        return eef_space


_eef_space = _import_eef_space()

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
KEY_STATE = "state"    # request (act): list[state_dim] float -> observation.state
KEY_OK = "ok"          # reply: bool success flag
KEY_ACTION = "action"  # reply (act): list[7] float (joint targets OR eef delta)
KEY_ERR = "err"        # reply: error string when ok=false

# ---- v2 additions: RESET handshake + ACT chunk marker -----------------------
KEY_ACTION_SPACE = "action_space"  # reply (reset): "joint" | "eef_delta_v1"
KEY_STATE_DIM = "state_dim"        # reply (reset): int, length of the state list
KEY_ACTION_DIM = "action_dim"      # reply (reset): int, always 7
KEY_POLICY_TYPE = "policy_type"    # reply (reset): lerobot cfg.type ("act", ...)
KEY_CHECKPOINT = "checkpoint"      # reply (reset): the loaded checkpoint path
KEY_CHUNK_START = "chunk_start"    # reply (act): bool, the net re-ran this tick

# ---- Multipart layout -------------------------------------------------------
# ACT request has exactly 3 frames: [control_json, cam1_jpeg, cam2_jpeg].
ACT_REQUEST_NFRAMES = 3

# ---- Observation dict keys the policy expects (dataset feature names) --------
# These MUST match the trained checkpoint's config.json input_features exactly.
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"

# Key of the ACTION entry in a lerobot config's output_features (lerobot's
# `lerobot.utils.constants.ACTION`; spelled out here so this module stays
# importable without lerobot, e.g. from the py3.10 ROS side).
POLICY_ACTION_FEATURE = "action"

# ---- Action spaces ----------------------------------------------------------
# "joint": the original banana/cube contract -- state 7, action 7 absolute joint
#          targets. "eef_delta_v1": CARROT_EEF_DESIGN §2 -- state 16, action 7
#          base-frame Cartesian delta. ACTION_SPACE_EEF is NOT a literal here on
#          purpose: eef_space owns the identity string and bumps it when the
#          definition changes.
ACTION_SPACE_JOINT = "joint"
ACTION_SPACE_EEF = _eef_space.ACTION_SPACE_ID  # "eef_delta_v1"
ACTION_SPACES = (ACTION_SPACE_JOINT, ACTION_SPACE_EEF)

# ---- Fixed dimensions -------------------------------------------------------
# STATE_DIM is the JOINT-space value, kept for v1 callers. Anything that can run
# in either space must use STATE_DIM_BY_SPACE / the RESET handshake instead.
STATE_DIM = 7   # [q1..q6, grip_pos]                       (joint space only)
ACTION_DIM = 7  # both spaces: 6 dof + gripper

STATE_DIM_BY_SPACE = {
    ACTION_SPACE_JOINT: STATE_DIM,          # 7
    ACTION_SPACE_EEF: _eef_space.STATE_DIM,  # 16
}

# Exit code a server uses when --action-space contradicts the checkpoint. Distinct
# from 2 (bad CLI) and 3 (CUDA unavailable) so a launcher can tell them apart.
EXIT_ACTION_SPACE_MISMATCH = 4


def state_dim_for_space(action_space: str) -> int:
    """Length of the `state` list for an action space. Raises on an unknown space."""
    try:
        return STATE_DIM_BY_SPACE[action_space]
    except KeyError:
        raise ValueError(
            f"unknown action_space {action_space!r}; expected one of {list(ACTION_SPACES)}"
        ) from None


def action_space_conflict(action_space: str, state_dim: int, action_dim: int) -> Optional[str]:
    """Return a human-readable reason the checkpoint cannot serve `action_space`.

    Returns None when the checkpoint's own feature shapes agree with the requested
    space. The dimensions come from the checkpoint (config.input_features /
    output_features), never from the CLI, so this is a real cross-check and not a
    restatement of the operator's intent.
    """
    if action_space not in STATE_DIM_BY_SPACE:
        return (f"unknown action space {action_space!r}; expected one of "
                f"{list(ACTION_SPACES)}")
    want_state = STATE_DIM_BY_SPACE[action_space]
    if state_dim != want_state:
        other = [s for s, d in STATE_DIM_BY_SPACE.items() if d == state_dim]
        hint = f" (that state_dim is {other[0]!r})" if other else ""
        return (f"checkpoint has observation.state shape ({state_dim},) but "
                f"--action-space {action_space} requires ({want_state},){hint}")
    if action_dim != ACTION_DIM:
        return (f"checkpoint has action shape ({action_dim},) but every supported "
                f"action space requires ({ACTION_DIM},)")
    return None


def enforce_action_space(action_space: str, state_dim: int, action_dim: int,
                         checkpoint: str, log_prefix: str) -> None:
    """Refuse to start (SystemExit 4) when the checkpoint contradicts the CLI.

    This is THE guard that makes "a joint checkpoint can never drive eef mode or
    vice versa" true at launch time rather than at the first tick on the arm. The
    client-side RESET handshake check (DESIGN §3.7) is the second, independent
    half of the same contract.
    """
    reason = action_space_conflict(action_space, state_dim, action_dim)
    if reason is None:
        return
    print(
        f"{log_prefix} ERROR: action-space / checkpoint mismatch.\n"
        f"{log_prefix}   checkpoint    : {checkpoint}\n"
        f"{log_prefix}   --action-space: {action_space}\n"
        f"{log_prefix}   {reason}\n"
        f"{log_prefix} Refusing to start: serving the wrong action space would send "
        f"joint targets to a Cartesian executor (or vice versa) and move the arm "
        f"in a way nothing downstream can detect.",
        file=sys.stderr, flush=True,
    )
    raise SystemExit(EXIT_ACTION_SPACE_MISMATCH)


def handshake(action_space: str, state_dim: int, action_dim: int,
              policy_type: str, checkpoint: str) -> dict:
    """Build the v2 RESET reply body (see the wire-format docstring above)."""
    return {
        KEY_OK: True,
        KEY_ACTION_SPACE: action_space,
        KEY_STATE_DIM: int(state_dim),
        KEY_ACTION_DIM: int(action_dim),
        KEY_POLICY_TYPE: policy_type,
        KEY_CHECKPOINT: checkpoint,
    }
