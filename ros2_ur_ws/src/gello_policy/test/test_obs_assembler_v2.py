"""ZMQ protocol v2 framing/parsing in obs_assembler (CARROT_EEF_DESIGN.md §4).

Runs WITHOUT ROS:

    PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy \\
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider \\
      test/test_obs_assembler_v2.py

v1 compatibility is a first-class requirement here: the banana/cube ACT, FM and
Diffusion servers in production today send NEITHER ``chunk_start`` NOR the RESET
handshake fields, and must keep working byte for byte.
"""
import json

import pytest

from gello_policy import obs_assembler as oa

JPEG1 = b"\xff\xd8jpg1"
JPEG2 = b"\xff\xd8jpg2"


def frame(obj):
    return [json.dumps(obj, separators=(",", ":")).encode("utf-8")]


# --------------------------------------------------------------------------- #
# constants: one definition of the eef space, repeated as a literal
# --------------------------------------------------------------------------- #
def test_wire_ids_agree_with_eef_space():
    eef_space = pytest.importorskip("gello_policy.eef_space")
    assert oa.EEF_ACTION_SPACE_ID == eef_space.ACTION_SPACE_ID
    assert oa.STATE_LEN_BY_ACTION_SPACE[oa.EEF_ACTION_SPACE_ID] == eef_space.STATE_DIM
    assert oa.ACTION_LEN_BY_ACTION_SPACE[oa.EEF_ACTION_SPACE_ID] == eef_space.ACTION_DIM


def test_joint_space_widths_are_unchanged():
    assert oa.STATE_LEN == oa.ACTION_LEN == 7
    assert oa.STATE_LEN_BY_ACTION_SPACE[oa.JOINT_ACTION_SPACE_ID] == 7
    assert oa.ACTION_LEN_BY_ACTION_SPACE[oa.JOINT_ACTION_SPACE_ID] == 7


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #
def test_reset_request_bytes_are_unchanged():
    assert oa.build_reset_request() == [b'{"cmd":"reset"}']


def test_act_request_defaults_to_the_v1_seven_floats():
    frames = oa.build_act_request([1, 2, 3, 4, 5, 6, 0.5], JPEG1, JPEG2)
    assert frames[0] == b'{"cmd":"act","state":[1.0,2.0,3.0,4.0,5.0,6.0,0.5]}'
    assert frames[1:] == [JPEG1, JPEG2]


def test_act_request_accepts_a_sixteen_wide_state():
    state = [float(i) for i in range(16)]
    frames = oa.build_act_request(state, JPEG1, JPEG2, state_len=16)
    assert json.loads(frames[0])["state"] == state
    assert frames[1:] == [JPEG1, JPEG2]


@pytest.mark.parametrize("state,state_len", [
    ([0.0] * 16, 7),      # eef state sent to a joint server
    ([0.0] * 7, 16),      # joint state sent to an eef server
    ([0.0] * 6, 7),
])
def test_act_request_refuses_a_state_of_the_wrong_width(state, state_len):
    with pytest.raises(ValueError, match="floats"):
        oa.build_act_request(state, JPEG1, JPEG2, state_len=state_len)


# --------------------------------------------------------------------------- #
# act replies
# --------------------------------------------------------------------------- #
def test_action_reply_unpacks_as_action_and_chunk_start():
    action, chunk_start = oa.parse_action_reply(
        frame({"ok": True, "action": [0, 1, 2, 3, 4, 5, 0.9], "chunk_start": True})
    )
    assert action == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9]
    assert chunk_start is True


def test_action_reply_fields_are_named_too():
    reply = oa.parse_action_reply(frame({"ok": True, "action": [0.0] * 7}))
    assert reply.action == [0.0] * 7
    assert reply.chunk_start is False


def test_missing_chunk_start_is_false_not_an_error():
    """A pre-v2 server never re-anchors a chunk -- which is what joint mode wants."""
    assert oa.parse_action_reply(frame({"ok": True, "action": [0.0] * 7})).chunk_start is False


def test_chunk_start_is_coerced_to_bool():
    assert oa.parse_action_reply(
        frame({"ok": True, "action": [0.0] * 7, "chunk_start": 1})
    ).chunk_start is True


def test_backward_compatible_helper_returns_the_bare_list():
    assert oa.parse_action_list(
        frame({"ok": True, "action": [0, 1, 2, 3, 4, 5, 0.9]})
    ) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9]


@pytest.mark.parametrize("frames,match", [
    (frame({"ok": False, "err": "boom"}), "ok:false"),
    (frame({"ok": True}), "missing 'action'"),
    (frame({"ok": True, "action": [0.0] * 6}), "floats"),
    ([b"not json"], "malformed"),
    ([], "empty"),
])
def test_bad_act_replies_raise(frames, match):
    with pytest.raises(ValueError, match=match):
        oa.parse_action_reply(frames)


def test_action_len_is_overridable():
    with pytest.raises(ValueError, match="floats"):
        oa.parse_action_reply(frame({"ok": True, "action": [0.0] * 7}), action_len=16)


# --------------------------------------------------------------------------- #
# R2/F3: non-finite actions are refused, in BOTH action spaces
#
# json.loads accepts the bare `NaN` / `Infinity` / `-Infinity` literals BY DEFAULT,
# so these are literally what a non-conforming server puts on the wire -- hence the
# raw byte strings below rather than json.dumps of a float('nan'). Without the check:
#   * action[6] reaches /robotiq_gripper/command_percent verbatim (the node's
#     0..1 clip `0.0 if g < 0 else (1.0 if g > 1 else g)` is NaN-transparent);
#   * in eef mode action[:6] costs eef_ik_fail_fault_n held ticks before the
#     IK-failure streak faults;
#   * in joint mode it is a NaN joint target.
# The caller turns the ValueError into a FAULT.
# --------------------------------------------------------------------------- #
NONFINITE_WIRE_REPLIES = [
    b'{"ok":true,"action":[NaN,0,0,0,0,0,0.0]}',            # NaN in the position delta
    b'{"ok":true,"action":[0,0,0,NaN,0,0,0.0]}',            # NaN in the rotation delta
    b'{"ok":true,"action":[0,0,0,0,0,0,NaN]}',              # NaN in the GRIPPER command
    b'{"ok":true,"action":[Infinity,0,0,0,0,0,0.0]}',
    b'{"ok":true,"action":[0,0,0,0,0,0,-Infinity]}',
]


@pytest.mark.parametrize("raw", NONFINITE_WIRE_REPLIES)
def test_non_finite_actions_are_refused(raw):
    with pytest.raises(ValueError, match="not finite"):
        oa.parse_action_reply([raw])


@pytest.mark.parametrize("raw", NONFINITE_WIRE_REPLIES)
def test_the_bare_json_literals_really_do_parse(raw):
    """The premise of the check: stdlib json accepts these, so they ARE reachable."""
    values = json.loads(raw.decode())["action"]
    assert len(values) == 7
    assert not all(v == v and abs(v) != float("inf") for v in values)


def test_non_finite_is_refused_in_the_sixteen_wide_eef_space_too():
    """One check covers both spaces -- the eef action is 7-D like the joint one."""
    raw = b'{"ok":true,"action":[0,0,0,0,0,0,NaN],"chunk_start":true}'
    with pytest.raises(ValueError, match="not finite"):
        oa.parse_action_reply([raw], action_len=oa.ACTION_LEN_BY_ACTION_SPACE[
            oa.EEF_ACTION_SPACE_ID])


def test_finite_actions_are_still_accepted_unchanged():
    """The guard must not have narrowed the accepted set (regression on v1)."""
    action, chunk = oa.parse_action_reply(
        frame({"ok": True, "action": [-1e6, 0.0, 1e6, -3.14, 0.0, 3.14, 1.0]}))
    assert action == [-1e6, 0.0, 1e6, -3.14, 0.0, 3.14, 1.0]
    assert chunk is False


def test_the_length_check_still_runs_first():
    """A short reply reports its LENGTH, not its finiteness (clearer diagnosis)."""
    with pytest.raises(ValueError, match="floats"):
        oa.parse_action_reply([b'{"ok":true,"action":[NaN,0,0,0,0,0]}'])


# --------------------------------------------------------------------------- #
# reset handshake
# --------------------------------------------------------------------------- #
def test_reset_reply_reports_the_full_handshake():
    hs = oa.parse_reset_reply(frame({
        "ok": True, "action_space": "eef_delta_v1", "state_dim": 16, "action_dim": 7,
        "policy_type": "multi_task_dit", "checkpoint": "/ckpt/080000",
    }))
    assert hs["ok"] is True
    assert hs["action_space"] == "eef_delta_v1"
    assert hs["state_dim"] == 16
    assert hs["action_dim"] == 7
    assert hs["policy_type"] == "multi_task_dit"
    assert hs["checkpoint"] == "/ckpt/080000"


def test_missing_handshake_fields_are_none_not_defaults():
    """"old server" and "server says joint/7" must be distinguishable: eef mode
    refuses the first, joint mode tolerates it."""
    hs = oa.parse_reset_reply(frame({"ok": True}))
    assert hs["ok"] is True
    assert hs["action_space"] is None
    assert hs["state_dim"] is None
    assert hs["action_dim"] is None
    assert hs["policy_type"] is None
    assert hs["checkpoint"] is None


def test_reset_reply_keeps_the_raw_object_for_error_messages():
    hs = oa.parse_reset_reply(frame({"ok": False, "err": "no checkpoint"}))
    assert hs["ok"] is False
    assert hs["raw"] == {"ok": False, "err": "no checkpoint"}


def test_unparseable_dims_become_none_rather_than_raising():
    hs = oa.parse_reset_reply(frame({"ok": True, "action_space": "joint",
                                     "state_dim": "seven", "action_dim": None}))
    assert hs["state_dim"] is None and hs["action_dim"] is None
    assert hs["action_space"] == "joint"


def test_joint_server_handshake_round_trips():
    hs = oa.parse_reset_reply(frame({"ok": True, "action_space": "joint",
                                     "state_dim": 7, "action_dim": 7,
                                     "policy_type": "act"}))
    assert (hs["action_space"], hs["state_dim"], hs["action_dim"]) == ("joint", 7, 7)


@pytest.mark.parametrize("frames", [[b"{"], []])
def test_malformed_reset_replies_raise(frames):
    with pytest.raises(ValueError):
        oa.parse_reset_reply(frames)


# --------------------------------------------------------------------------- #
# cross-interpreter contract: the server's copy of the constants
# --------------------------------------------------------------------------- #
def test_server_side_constants_are_identical():
    """``policy_server/zmq_protocol.py`` runs in a DIFFERENT interpreter (py3.12
    act_venv) and therefore duplicates these literals on purpose. Nothing at
    runtime can catch a divergence, so pin it here -- both files are in this repo.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "policy_server", "zmq_protocol.py",
    )
    if not os.path.exists(path):
        pytest.skip("policy_server/zmq_protocol.py not present")
    spec = importlib.util.spec_from_file_location("_server_zmq_protocol", path)
    proto = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(proto)
    except Exception as exc:  # noqa: BLE001 - A5's module, py3.12-oriented
        pytest.skip(f"cannot import the server-side protocol here: {exc}")

    for key in ("CMD_RESET", "CMD_ACT", "KEY_CMD", "KEY_STATE", "KEY_OK",
                "KEY_ACTION", "KEY_ERR"):
        assert getattr(proto, key) == getattr(oa, key), key
    for key in ("KEY_CHUNK_START", "KEY_ACTION_SPACE", "KEY_STATE_DIM",
                "KEY_ACTION_DIM", "KEY_POLICY_TYPE", "KEY_CHECKPOINT"):
        if hasattr(proto, key):       # A5 may not have landed this key yet
            assert getattr(proto, key) == getattr(oa, key), key
    if hasattr(proto, "STATE_DIM_BY_SPACE"):
        assert proto.STATE_DIM_BY_SPACE == oa.STATE_LEN_BY_ACTION_SPACE
    if hasattr(proto, "handshake"):
        hs = oa.parse_reset_reply(
            [json.dumps(proto.handshake("eef_delta_v1", 16, 7, "act", "/ckpt"),
                        separators=(",", ":")).encode()]
        )
        assert (hs["ok"], hs["action_space"], hs["state_dim"], hs["action_dim"],
                hs["policy_type"], hs["checkpoint"]) == (
            True, "eef_delta_v1", 16, 7, "act", "/ckpt")
