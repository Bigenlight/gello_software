"""EEF-delta execution semantics (CARROT_EEF_DESIGN.md §3 items 3-5).

Runs WITHOUT ROS:

    PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy \\
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider \\
      test/test_eef_executor.py

The last two tests are the load-bearing ones: they replay REAL recorded joint
frames from the carrot takes through ``eef_space.delta_action`` -> executor ->
IK and require the original ``q_{t+1}`` back, single-step and accumulated over a
chunk. Everything else is a property of the clamps in isolation.
"""
import ast
import glob
import math
import os

import numpy as np
import pytest

from gello_policy import eef_space
from gello_policy import obs_assembler as oa
from gello_policy.eef_executor import (
    DEFAULT_IK_FAIL_FAULT_N,
    DEFAULT_MAX_LEAD_M,
    DEFAULT_MAX_LEAD_RAD,
    DEFAULT_MAX_STEP_M,
    DEFAULT_MAX_STEP_RAD,
    DEFAULT_WORKSPACE_HI,
    DEFAULT_WORKSPACE_LO,
    EefDeltaExecutor,
    EefExecutorConfig,
    apply_delta_pose,
)
from ur_gello_bringup import ur_kin

# Mean frame-0 pose of the 54 carrot takes (design §1). Branch 2.
START_POSE = (-3.193, -1.490, 1.693, -1.798, -1.590, -3.269)
TAKES_GLOB = "/home/laptop3/youngwoong_ws/Put_carrot_in_pot/take_*/vectors.h5"

NODE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gello_policy", "policy_leader_node.py",
)


def cfg(**kw):
    """Config with the design defaults unless a test overrides them."""
    kw.setdefault("start_pose", START_POSE)
    return EefExecutorConfig(**kw)


def wide_cfg(**kw):
    """Config whose clamps can never engage -- isolates accumulation from clamping."""
    kw.setdefault("max_step_m", 10.0)
    kw.setdefault("max_step_rad", 10.0)
    kw.setdefault("max_lead_m", 10.0)
    kw.setdefault("max_lead_rad", 10.0)
    kw.setdefault("workspace_lo", (-10.0, -10.0, -10.0))
    kw.setdefault("workspace_hi", (10.0, 10.0, 10.0))
    return cfg(**kw)


# --------------------------------------------------------------------------- #
# config contract
# --------------------------------------------------------------------------- #
def test_defaults_match_the_design_table():
    c = cfg()
    assert (c.max_step_m, c.max_step_rad) == (0.02, 0.10)
    assert (c.max_lead_m, c.max_lead_rad) == (0.06, 0.30)
    assert c.workspace_lo == (0.35, -0.30, 0.00)
    assert c.workspace_hi == (0.78, 0.45, 0.40)
    assert c.ik_fail_fault_n == 10


def _declared_params(path):
    """{name: default} for every declare_parameter(...) literal in the node source.

    Parsed from the AST instead of imported: the node needs rclpy, this test must
    run without ROS, and the parameter NAMES are a contract with A6's yamls.
    """
    tree = ast.parse(open(path).read())
    # Module-level constants of the node (ACTION_SPACE_JOINT = "joint", ...) so a
    # default written as a named constant still resolves.
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    def value_of(node):
        try:
            return ast.literal_eval(node)
        except ValueError:
            pass
        if isinstance(node, ast.Name) and node.id in consts:
            return consts[node.id]
        # obs_assembler.<CONST> -- resolved against the real module.
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "obs_assembler"):
            return getattr(oa, node.attr)
        return "<non-literal>"

    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)):
            out[node.args[0].value] = value_of(node.args[1])
    return out


def test_node_declares_the_design_parameter_names_and_defaults():
    params = _declared_params(NODE_SRC)
    assert params["action_space"] == "joint"
    assert params["expected_action_space"] == eef_space.ACTION_SPACE_ID
    assert params["eef_max_step_m"] == DEFAULT_MAX_STEP_M
    assert params["eef_max_step_rad"] == DEFAULT_MAX_STEP_RAD
    assert params["eef_max_lead_m"] == DEFAULT_MAX_LEAD_M
    assert params["eef_max_lead_rad"] == DEFAULT_MAX_LEAD_RAD
    assert tuple(params["workspace_lo"]) == DEFAULT_WORKSPACE_LO
    assert tuple(params["workspace_hi"]) == DEFAULT_WORKSPACE_HI
    assert params["eef_ik_fail_fault_n"] == DEFAULT_IK_FAIL_FAULT_N
    # The joint path's own parameters must still be there, untouched.
    assert params["max_dev_rad"] == 0.5
    assert params["inference_transport"] == "zmq"


@pytest.mark.parametrize("kw", [
    {"start_pose": (0.0,) * 5},
    {"workspace_lo": (0.0, 0.0)},
    {"workspace_hi": (0.0, 0.0, 0.0, 0.0)},
    {"workspace_lo": (0.5, 0.0, 0.0), "workspace_hi": (0.4, 0.1, 0.1)},
    {"max_step_m": 0.0},
    {"max_step_rad": -1.0},
    {"max_lead_m": 0.0},
    {"max_lead_rad": 0.0},
    {"ik_fail_fault_n": 0},
])
def test_config_refuses_nonsense(kw):
    with pytest.raises(ValueError):
        cfg(**kw)


def test_start_branch_is_derived_from_start_pose_and_overridable():
    ex = EefDeltaExecutor(cfg())
    assert ex.start_branch == ur_kin.branch_id(np.asarray(START_POSE)) == 2
    assert EefDeltaExecutor(cfg(start_branch=5)).start_branch == 5


def test_apply_delta_pose_is_eef_space_apply_delta_on_a_pose():
    """Pins the one piece of §2 arithmetic that has no pose-level entry point."""
    rng = np.random.default_rng(0)
    q = np.asarray(START_POSE)
    for _ in range(20):
        a = np.concatenate([rng.normal(scale=0.05, size=3),
                            rng.normal(scale=0.05, size=3), [0.0]])
        assert np.allclose(apply_delta_pose(eef_space.tcp_pose(q), a),
                           eef_space.apply_delta(q, a), atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# chunk accumulation
# --------------------------------------------------------------------------- #
def test_accumulates_on_the_target_while_the_arm_stalls():
    """§3.3: N ticks of the same delta must move the target N deltas, even if the
    live pose never moves (the whole reason accumulation is on the target)."""
    ex = EefDeltaExecutor(wide_cfg())
    q = np.asarray(START_POSE)
    p0 = eef_space.tcp_pose(q)[:3, 3]
    d = np.array([0.003, -0.002, 0.001, 0.0, 0.0, 0.0, 0.0])
    for i in range(1, 11):
        q_t, info = ex.step(q, d, chunk_start=(i == 1))
        assert q_t is not None
        assert np.allclose(ex.target_pose[:3, 3], p0 + i * d[:3], atol=1e-12)
        # ...and the IK'd joints really are that target, not the live pose.
        assert np.allclose(eef_space.tcp_pose(q_t)[:3, 3], p0 + i * d[:3], atol=1e-9)
    assert info["lead_pos_norm"] == pytest.approx(np.linalg.norm(10 * d[:3]))


def test_chunk_start_re_anchors_on_the_live_pose():
    ex = EefDeltaExecutor(wide_cfg())
    q = np.asarray(START_POSE)
    d = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        ex.step(q, d, chunk_start=False)
    assert ex.target_pose[:3, 3][0] == pytest.approx(eef_space.tcp_pose(q)[:3, 3][0] + 0.025)

    q2 = q + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, info = ex.step(q2, d, chunk_start=True)
    assert info["re_anchored"] is True
    # anchor == TCP(live), target == anchor + ONE delta.
    assert np.allclose(ex.anchor_pose, eef_space.tcp_pose(q2))
    assert np.allclose(ex.target_pose[:3, 3], eef_space.tcp_pose(q2)[:3, 3] + d[:3])


def test_first_tick_after_reset_anchors_even_without_chunk_start():
    ex = EefDeltaExecutor(wide_cfg())
    q = np.asarray(START_POSE)
    ex.step(q, np.zeros(7), chunk_start=True)
    ex.reset()
    assert ex.target_pose is None
    _, info = ex.step(q, np.zeros(7), chunk_start=False)
    assert info["re_anchored"] is True
    assert np.allclose(ex.target_pose, eef_space.tcp_pose(q))


def test_reset_with_a_live_pose_anchors_immediately():
    ex = EefDeltaExecutor(wide_cfg())
    q = np.asarray(START_POSE)
    ex.reset(q)
    assert np.allclose(ex.target_pose, eef_space.tcp_pose(q))
    _, info = ex.step(q, np.zeros(7), chunk_start=False)
    assert info["re_anchored"] is False
    assert ex.ik_fail_streak == 0


# --------------------------------------------------------------------------- #
# clamps: each engages, and each preserves direction
# --------------------------------------------------------------------------- #
def test_per_tick_step_clamp_scales_the_delta_and_keeps_its_direction():
    ex = EefDeltaExecutor(wide_cfg(max_step_m=0.01))
    q = np.asarray(START_POSE)
    p0 = eef_space.tcp_pose(q)[:3, 3]
    raw = np.array([0.06, -0.08, 0.0])          # norm 0.1 -> 10x over the clamp
    _, info = ex.step(q, np.concatenate([raw, np.zeros(4)]), chunk_start=True)
    assert info["clamped_step_pos"] is True and info["clamped_step_rot"] is False
    moved = ex.target_pose[:3, 3] - p0
    assert np.linalg.norm(moved) == pytest.approx(0.01)
    # direction preserved: parallel, not per-axis truncated.
    assert np.allclose(moved / np.linalg.norm(moved), raw / np.linalg.norm(raw))
    assert info["delta_pos_norm"] == pytest.approx(0.1)
    assert info["delta_pos_norm_clamped"] == pytest.approx(0.01)


def test_per_tick_step_clamp_scales_rotation_and_keeps_its_axis():
    ex = EefDeltaExecutor(wide_cfg(max_step_rad=0.05))
    q = np.asarray(START_POSE)
    R0 = eef_space.tcp_pose(q)[:3, :3]
    raw = np.array([0.2, 0.1, -0.2])
    _, info = ex.step(q, np.concatenate([np.zeros(3), raw, [0.0]]), chunk_start=True)
    assert info["clamped_step_rot"] is True and info["clamped_step_pos"] is False
    applied = ur_kin.so3_log(ex.target_pose[:3, :3] @ R0.T)
    assert np.linalg.norm(applied) == pytest.approx(0.05, abs=1e-9)
    assert np.allclose(applied / np.linalg.norm(applied), raw / np.linalg.norm(raw))


def test_lead_clamp_stops_the_target_running_away_from_a_stalled_arm():
    ex = EefDeltaExecutor(wide_cfg(max_lead_m=0.03))
    q = np.asarray(START_POSE)
    p_live = eef_space.tcp_pose(q)[:3, 3]
    d = np.array([0.008, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.01 m per tick
    infos = [ex.step(q, d, chunk_start=(i == 0))[1] for i in range(8)]
    assert infos[0]["clamped_lead_pos"] is False
    assert infos[-1]["clamped_lead_pos"] is True
    lead = ex.target_pose[:3, 3] - p_live
    assert np.linalg.norm(lead) == pytest.approx(0.03)
    assert np.allclose(lead / np.linalg.norm(lead), d[:3] / np.linalg.norm(d[:3]))
    # written BACK into the accumulator: the lead never exceeds the cap again.
    for _ in range(5):
        ex.step(q, d, chunk_start=False)
        assert np.linalg.norm(ex.target_pose[:3, 3] - p_live) <= 0.03 + 1e-12


def test_lead_clamp_bounds_rotation_too():
    ex = EefDeltaExecutor(wide_cfg(max_lead_rad=0.06))
    q = np.asarray(START_POSE)
    R_live = eef_space.tcp_pose(q)[:3, :3]
    d = np.concatenate([np.zeros(3), [0.0, 0.0, 0.03], [0.0]])
    for i in range(6):
        _, info = ex.step(q, d, chunk_start=(i == 0))
    assert info["clamped_lead_rot"] is True
    lead_rot = ur_kin.so3_log(ex.target_pose[:3, :3] @ R_live.T)
    assert np.linalg.norm(lead_rot) == pytest.approx(0.06, abs=1e-9)
    assert np.allclose(lead_rot / np.linalg.norm(lead_rot), [0.0, 0.0, 1.0], atol=1e-9)


def test_workspace_box_clips_the_target_position():
    q = np.asarray(START_POSE)
    p0 = eef_space.tcp_pose(q)[:3, 3]
    hi = (p0[0] + 0.005, 10.0, 10.0)            # only x is tight
    ex = EefDeltaExecutor(wide_cfg(workspace_lo=(-10.0, -10.0, -10.0), workspace_hi=hi))
    _, info = ex.step(q, np.array([0.02, 0.01, 0.0, 0, 0, 0, 0]), chunk_start=True)
    assert info["clamped_box"] is True
    assert ex.target_pose[0, 3] == pytest.approx(p0[0] + 0.005)
    assert ex.target_pose[1, 3] == pytest.approx(p0[1] + 0.01)   # y untouched
    assert info["any_clamped"] is True


def test_no_clamp_no_flags():
    ex = EefDeltaExecutor(cfg())
    _, info = ex.step(np.asarray(START_POSE), np.array([0.001, 0, 0, 0, 0, 0, 0.0]),
                      chunk_start=True)
    assert info["any_clamped"] is False
    assert EefDeltaExecutor.clamp_reasons(info) == []


def test_clamp_reasons_names_every_clamp():
    info = {"clamped_step_pos": True, "clamped_step_rot": True,
            "clamped_lead_pos": True, "clamped_lead_rot": True, "clamped_box": True,
            "delta_pos_norm": 1.0, "delta_rot_norm": 1.0,
            "lead_pos_norm": 1.0, "lead_rot_norm": 1.0}
    assert len(EefDeltaExecutor.clamp_reasons(info)) == 5


# --------------------------------------------------------------------------- #
# IK failure
# --------------------------------------------------------------------------- #
def test_unreachable_target_returns_none_and_counts_up_to_the_fault():
    """Real kinematics: a workspace box 2.9 m out in x cannot be inverted."""
    ex = EefDeltaExecutor(cfg(
        max_step_m=1.0, max_step_rad=1.0, max_lead_m=100.0, max_lead_rad=10.0,
        workspace_lo=(2.9, -0.3, 0.0), workspace_hi=(3.1, 0.45, 0.4),
        ik_fail_fault_n=3,
    ))
    q = np.asarray(START_POSE)
    for expected in (1, 2, 3, 4):
        q_t, info = ex.step(q, np.zeros(7), chunk_start=(expected == 1))
        assert q_t is None
        assert info["ik_ok"] is False
        assert info["ik_fail_streak"] == expected == ex.ik_fail_streak
        assert info["ik_fault"] is (expected >= 3)
    ex.reset()
    assert ex.ik_fail_streak == 0


def test_a_successful_tick_clears_the_streak(monkeypatch):
    ex = EefDeltaExecutor(cfg(ik_fail_fault_n=5))
    q = np.asarray(START_POSE)
    real = eef_space.ik_tcp_nearest
    fail = {"on": True}

    def fake(*args, **kw):
        if fail["on"]:
            return None, {"n_solutions": 0, "branch": -1, "on_branch": False,
                          "max_dev": float("nan")}
        return real(*args, **kw)

    monkeypatch.setattr(eef_space, "ik_tcp_nearest", fake)
    for _ in range(4):
        ex.step(q, np.zeros(7), chunk_start=False)
    assert ex.ik_fail_streak == 4
    fail["on"] = False
    q_t, info = ex.step(q, np.zeros(7), chunk_start=False)
    assert q_t is not None and info["ik_fail_streak"] == 0 and info["ik_fault"] is False


def test_info_carries_the_ik_diagnostics():
    ex = EefDeltaExecutor(cfg())
    _, info = ex.step(np.asarray(START_POSE), np.zeros(7), chunk_start=True)
    assert info["ik_ok"] is True
    assert info["ik_branch"] == info["start_branch"] == 2
    assert info["ik_on_branch"] is True
    assert info["ik_n_solutions"] >= 1
    assert info["ik_max_dev"] == pytest.approx(0.0, abs=1e-9)
    assert info["tick"] == 1 and info["chunk_start"] is True


def test_action_shorter_than_six_is_refused():
    ex = EefDeltaExecutor(cfg())
    with pytest.raises(ValueError):
        ex.step(np.asarray(START_POSE), np.zeros(5), chunk_start=True)


# --------------------------------------------------------------------------- #
# real recorded data: dataset action -> executor -> the joints it came from
# --------------------------------------------------------------------------- #
def _take_joints(n_frames=None):
    h5py = pytest.importorskip("h5py")
    takes = sorted(glob.glob(TAKES_GLOB))
    if not takes:
        pytest.skip(f"no carrot takes at {TAKES_GLOB}")
    with h5py.File(takes[0], "r") as f:
        g = f["ur_joint_states"]
        q = np.stack([np.asarray(g[f"q{i}"]) for i in range(1, 7)], axis=1)
    return q if n_frames is None else q[:n_frames]


def test_real_frames_round_trip_single_step():
    """delta_action(q_t, q_t+1) -> step(chunk_start) must give back q_{t+1}."""
    q = _take_joints()
    ex = EefDeltaExecutor(wide_cfg(start_branch=int(ur_kin.branch_id(q[0]))))
    worst = 0.0
    idx = np.linspace(0, len(q) - 2, 40).astype(int)
    for t in idx:
        a = eef_space.delta_action(q[t], q[t + 1], 0.0)
        q_t, info = ex.step(q[t], a, chunk_start=True)
        assert q_t is not None, f"IK failed at frame {t}"
        assert info["any_clamped"] is False
        worst = max(worst, float(np.max(np.abs(q_t - q[t + 1]))))
    assert worst < 1e-6, f"worst joint error {worst} rad"


def test_real_frames_accumulate_over_a_chunk_while_the_arm_stalls():
    """10 deltas applied to the target from ONE anchor land on q_{t+10}.

    This is the property §3.3 exists for: the live pose is held at q_t the whole
    time (a lagging arm), yet the target still walks the full recorded trajectory.
    """
    q = _take_joints()
    k = 10
    ex = EefDeltaExecutor(wide_cfg(start_branch=int(ur_kin.branch_id(q[0]))))
    worst = 0.0
    for t in (0, 50, 100, 150, 200):
        if t + k >= len(q):
            continue
        q_t = None
        for j in range(k):
            a = eef_space.delta_action(q[t + j], q[t + j + 1], 0.0)
            q_t, info = ex.step(q[t], a, chunk_start=(j == 0))
            assert q_t is not None and info["any_clamped"] is False
        worst = max(worst, float(np.max(np.abs(q_t - q[t + k]))))
        assert np.allclose(ex.target_pose, eef_space.tcp_pose(q[t + k]), atol=1e-9)
    assert worst < 1e-6, f"worst joint error {worst} rad"


def test_real_frame_deltas_are_within_the_shipped_step_clamp():
    """Sanity on the §3 defaults: real 30 fps motion must not be clamped.

    Design §1: 1-frame |dp| max 9.6 mm, |drot| max 0.03 rad vs the 0.02 m / 0.10
    rad defaults. If this fails the defaults are throttling the policy itself.
    """
    q = _take_joints()
    c = cfg()
    dp = []
    dr = []
    for t in range(len(q) - 1):
        a = eef_space.delta_action(q[t], q[t + 1], 0.0)
        dp.append(np.linalg.norm(a[:3]))
        dr.append(np.linalg.norm(a[3:6]))
    assert max(dp) < c.max_step_m, f"|dp| max {max(dp)} >= {c.max_step_m}"
    assert max(dr) < c.max_step_rad, f"|drot| max {max(dr)} >= {c.max_step_rad}"
    assert not math.isnan(max(dp))


# --------------------------------------------------------------------------- #
# node wiring -- SKIPPED without ROS (this whole file must run ROS-free), but it
# runs in the overlay suite of CARROT_EEF_DESIGN.md §6. Construct-only: no timer
# ever fires because nothing spins, so no topic is ever published.
# --------------------------------------------------------------------------- #
CARROT_PARAMS = [
    "-p", "start_pose:=[-3.193,-1.490,1.693,-1.798,-1.590,-3.269]",
    "-p", "joint_limits_lo:=[-4.0,-2.6,0.9,-3.0,-2.2,-4.2]",
    "-p", "joint_limits_hi:=[-2.4,-0.8,2.5,-0.9,-1.0,-2.4]",
]


def _node(argv):
    rclpy = pytest.importorskip("rclpy")
    from gello_policy.policy_leader_node import PolicyLeaderNode
    rclpy.init(args=["--ros-args"] + argv)
    return rclpy, PolicyLeaderNode


@pytest.fixture
def node_factory():
    made = []
    rclpy_mod = []

    def make(argv):
        rclpy, cls = _node(argv)
        rclpy_mod.append(rclpy)
        n = cls()
        made.append(n)
        return n

    yield make
    for n in made:
        n.destroy_node()
    for rclpy in rclpy_mod:
        if rclpy.ok():
            rclpy.shutdown()


def _reset_frames(**fields):
    import json
    return [json.dumps(dict(ok=True, **fields)).encode()]


def test_node_eef_mode_handshake_matrix(node_factory):
    from gello_policy import obs_assembler as oa
    n = node_factory(["-p", "action_space:=eef_delta"] + CARROT_PARAMS)
    assert n._eef_mode is True and n._state_len == 16 and n._action_len == 7
    assert n._eef.start_branch == 2

    def check(**fields):
        return n._check_handshake(oa.parse_reset_reply(_reset_frames(**fields)))

    assert check(action_space="eef_delta_v1", state_dim=16, action_dim=7)[0] is True
    assert check()[0] is False                                    # pre-v2 server
    assert check(action_space="joint", state_dim=7, action_dim=7)[0] is False
    assert check(action_space="eef_delta_v1", state_dim=7, action_dim=7)[0] is False
    assert check(action_space="eef_delta_v1", state_dim=16, action_dim=6)[0] is False


def test_node_joint_mode_tolerates_a_pre_v2_server(node_factory):
    from gello_policy import obs_assembler as oa
    n = node_factory(CARROT_PARAMS)
    assert n._eef_mode is False and n._eef is None and n._state_len == 7

    def check(**fields):
        return n._check_handshake(oa.parse_reset_reply(_reset_frames(**fields)))

    assert check() == (True, "")                                  # today's servers
    assert check(action_space="joint")[0] is True                 # partial v2
    assert check(action_space="joint", state_dim=7, action_dim=7)[0] is True
    assert check(action_space="eef_delta_v1", state_dim=16, action_dim=7)[0] is False


@pytest.mark.parametrize("argv,match", [
    (["-p", "action_space:=eef_delta", "-p", "inference_transport:=grpc"], "zmq"),
    (["-p", "action_space:=cartesian"], "action_space must be"),
])
def test_node_refuses_impossible_configurations(argv, match):
    rclpy, cls = _node(argv + CARROT_PARAMS)
    try:
        with pytest.raises(ValueError, match=match):
            cls()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
