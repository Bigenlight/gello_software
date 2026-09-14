"""Node-level guards added after the R2 deploy-path review -- NO robot, NO checkpoint.

    cd <wt>/ros2_ur_ws && colcon build --packages-select ur_gello_bringup gello_policy \
        --symlink-install && source install/setup.bash
    cd src/gello_policy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python3 -m pytest -q -p no:cacheprovider test/test_node_eef_guards.py

Findings covered (ids from the R2 report):

  F1  the joint envelope clipping an IK SOLUTION is uncommanded Cartesian motion.
      A per-axis-clipped IK solution is not an IK solution, so the executed TCP
      leaves the workspace box the Cartesian clamps just enforced -- the box only
      ever constrained the TARGET pose.  The node now HOLDS the live pose on such a
      tick and FAULTs on a streak of ``eef_ik_fail_fault_n``.
  F2B ``_check_handshake``'s success detail names the server's CHECKPOINT, so the
      log can answer "which weights drove the arm".
  F5  ``ik_in_limits`` reaches the node and a lost branch / out-of-limit solution
      WARNs (warn only -- integrator decision).
  F8  degenerate ``chunk_start`` patterns WARN instead of being silent.

Plus the two things the above must NOT break: the node-level IK-failure streak
(which used to be covered through a NaN action, now rejected earlier by F3) and
joint mode, which must be byte-for-byte what it was.

The fakes (in-process ZMQ REP policy + perfectly tracking arm) are imported from
``test_e2e_eef_fake_server``; see that module's docstring for the design.  Unlike
that file NOTHING here needs the LeRobot dataset: every action is synthetic, so
these tests run anywhere the ROS overlay does.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# Sibling-module import: pytest's default (prepend) import mode puts this directory
# on sys.path because there is no __init__.py, but say it explicitly so the file also
# works when run with a different rootdir or import mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing the e2e module pins ROS_DOMAIN_ID/ROS_LOCALHOST_ONLY and performs the
# rclpy/pyzmq importorskip, so a machine without the overlay skips this file too.
import test_e2e_eef_fake_server as e2e  # noqa: E402

rclpy = e2e.rclpy  # noqa: N816  (already importorskip'ed by the module above)

from gello_policy import eef_space  # noqa: E402
from gello_policy import obs_assembler as oa  # noqa: E402
from gello_policy import policy_leader_node as pln  # noqa: E402
from gello_policy.policy_leader_node import EXECUTE, FAULT, HOLD  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CARROT_YAML = e2e.CARROT_YAML
JOINT_YAML = os.path.join(e2e.PKG, "config", "act_deploy.yaml")

# One synthetic action, repeated: a pure +x Cartesian delta far above
# eef_max_step_m, so the per-tick clamp pins it at the cap and the target marches
# straight at the far x wall of the workspace box.  R2's OOD probe used exactly this
# signal; it is the cheapest reproduction of "the policy drifts reach-further".
PUSH_X = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PULL_X = [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
NO_MOVE = [0.0] * 7

# Measured offline against the shipped carrot_eef_act.yaml with a perfectly tracking
# arm (see the module docstring of test_e2e_eef_fake_server for what "perfect" buys):
# the envelope starts clipping on tick 10 of a sustained +x drive.
FIRST_CLIP_TICK = 10


# =============================================================================
# fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def _rclpy_is_shut_down():
    """A test that dies before Harness.__exit__ must not poison the next one."""
    yield
    try:
        rclpy.try_shutdown()
    except Exception:  # noqa: BLE001
        pass


def _params(path):
    if yaml is None or not os.path.exists(path):
        pytest.skip(f"{path} missing (or pyyaml unavailable)")
    with open(path) as fh:
        return yaml.safe_load(fh)["policy_leader_node"]["ros__parameters"]


@pytest.fixture(scope="module")
def carrot_params():
    """policy_leader_node's section of the real carrot EEF deploy yaml."""
    return _params(CARROT_YAML)


@pytest.fixture(scope="module")
def joint_params():
    """policy_leader_node's section of the banana JOINT deploy yaml (regression)."""
    return _params(JOINT_YAML)


# =============================================================================
# helpers
# =============================================================================
def _eef_harness(carrot_params, port, **overrides):
    """Harness armed on the yaml's own start pose (the gate's target)."""
    return e2e.make_harness(
        carrot_params, port,
        list(carrot_params["start_pose"]), float(carrot_params["start_gripper"]),
        **overrides)


def _arm(h):
    h.prime_observations()
    result = h.call(h.start_cli)
    assert result.success, f"~/start_execution refused: {result.message}"
    assert h.leader._state == EXECUTE
    h.arm.follow = True
    h.arm.targets.clear()
    h.arm.grip_cmds.clear()
    return result


def _tick(h, timeout=2.0):
    """One tick; if the node published, wait until the fake arm has adopted it.

    ``Harness.tick_until_arm_follows`` cannot be used directly here because these
    tests deliberately run PAST a FAULT, and a faulted node publishes nothing.
    """
    before = len(h.arm.targets)
    h.pump(0.01)
    h.leader._on_timer()
    if h.leader._state != EXECUTE:
        h.pump(0.05)
        return
    h.wait(lambda: len(h.arm.targets) > before, timeout=timeout,
           message="the arm to receive the published target")
    target = h.arm.targets[-1]
    h.wait(lambda: h.leader._live_q is not None
           and max(abs(a - b) for a, b in zip(h.leader._live_q, target)) < 1e-12,
           timeout=timeout, message="the leader to see the tracked joint state")


def _run(h, n):
    """Tick ``n`` times or until the node leaves EXECUTE; returns the ticks used."""
    for i in range(n):
        if h.leader._state != EXECUTE:
            return i
        _tick(h)
    return n


def _warnings_matching(h, needle):
    return [w for w in h.warnings if needle in w]


# =============================================================================
# F1 -- an envelope-clipped IK solution holds, and a streak FAULTs
# =============================================================================
def test_f1_envelope_clip_holds_the_live_pose_then_faults(carrot_params):
    """A sustained OOD +x drive must FREEZE the arm, not slide it.

    Before this guard the node published the clipped IK solution and the tool slid
    (R2 measured 168 mm DOWN with the commanded target z CONSTANT, ending 96 mm off
    target) while nothing faulted.
    """
    fault_n = int(carrot_params["eef_ik_fail_fault_n"])
    actions = [PUSH_X] * (FIRST_CLIP_TICK + fault_n + 5)
    # chunk_len far beyond the test: a re-anchor would reset the target onto the live
    # pose and dissolve exactly the condition under test.
    with e2e.FakeEefServer(actions, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            used = _run(h, len(actions))

            assert h.leader._state == FAULT, (
                f"no FAULT after {used} ticks; "
                f"streak={h.leader._eef_env_clip_streak}")
            assert any("hit the joint envelope on" in e for e in h.errors), h.errors
            assert f"{fault_n} consecutive ticks" in " ".join(h.errors)

            # FAULT lands exactly fault_n clipping ticks after the first one.
            assert used == FIRST_CLIP_TICK + fault_n

            # Every clipping tick published the LIVE pose: the arm is frozen, which
            # is the whole point (a clipped IK solution is not an IK solution).
            held = np.asarray(h.arm.targets[FIRST_CLIP_TICK:], dtype=float)
            assert len(held) >= fault_n - 1
            assert np.allclose(held, held[0], atol=1e-12), (
                "the arm kept moving while the envelope was clipping")

            clip_warns = _warnings_matching(h, "EEF joint-envelope clip")
            assert len(clip_warns) == fault_n - 1, clip_warns
            assert f"(streak 1/{fault_n})" in clip_warns[0]


def test_f1_a_clean_tick_resets_the_streak(carrot_params):
    """Back off before the streak matures and the count starts over from zero."""
    fault_n = int(carrot_params["eef_ik_fail_fault_n"])
    n_clip = 5                      # < fault_n, so no FAULT yet
    n_back = 6                      # measured: enough to pull the target back in
    actions = ([PUSH_X] * (FIRST_CLIP_TICK + n_clip)
               + [PULL_X] * n_back
               + [PUSH_X] * (FIRST_CLIP_TICK + fault_n + 5))
    with e2e.FakeEefServer(actions, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)

            _run(h, FIRST_CLIP_TICK + n_clip)
            assert h.leader._state == EXECUTE, "faulted too early"
            assert h.leader._eef_env_clip_streak == n_clip

            # Back off: the first tick whose IK solution fits the envelope again
            # zeroes the counter.
            _run(h, n_back)
            assert h.leader._state == EXECUTE
            assert h.leader._eef_env_clip_streak == 0, "a clean tick must reset"

            # ...and the FAULT now needs a fresh, FULL streak.
            _run(h, len(actions))
            assert h.leader._state == FAULT
            assert f"{fault_n} consecutive ticks" in " ".join(h.errors)


def test_f1_streak_is_cleared_by_hold_and_by_rearming(carrot_params):
    """``~/hold`` and a re-arm are episode boundaries; suspicion must not carry over."""
    actions = [PUSH_X] * 200
    with e2e.FakeEefServer(actions, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, FIRST_CLIP_TICK + 3)
            assert h.leader._eef_env_clip_streak == 3

            assert h.call(h.hold_cli).success
            assert h.leader._state == HOLD
            assert h.leader._eef_env_clip_streak == 0

            # Re-arming: the 0.1 rad pose gate runs BEFORE the RESET, so put the
            # fake arm back on start_pose first (on the real robot that is what
            # gello_move_to_start does). Then take the REAL arming path.
            h.leader._eef_env_clip_streak = 7
            h.leader._chunk_start_silent_ticks = 7
            h.arm.follow = False
            h.arm.q = list(h.leader._start_pose)
            h.prime_observations()
            result = h.call(h.start_cli)
            assert result.success, result.message
            assert h.leader._state == EXECUTE
            assert h.leader._eef_env_clip_streak == 0
            assert h.leader._chunk_start_silent_ticks == 0


def test_f1_does_not_exist_in_joint_mode(joint_params):
    """Joint mode must be byte-for-byte what it was: no hold, no streak, no FAULT.

    The same clip in joint space is harmless -- a clipped JOINT target is still a
    valid joint pose -- so the guard is gated on eef mode and joint mode keeps
    publishing the clipped target exactly as before.
    """
    lo = list(joint_params["joint_limits_lo"])
    start = list(joint_params["start_pose"])
    # A joint-space "action" IS the joint target. Ask for one 1 rad below the
    # envelope floor on joint 0: the node must clip it and PUBLISH the clipped value.
    ood = [lo[0] - 1.0] + start[1:] + [0.0]
    with e2e.FakeEefServer([ood] * 40, action_space=oa.JOINT_ACTION_SPACE_ID,
                           state_dim=7, action_dim=7, chunk_len=10_000) as server:
        with e2e.Harness(JOINT_YAML, {"act_port": server.port}, start,
                         float(joint_params["start_gripper"])) as h:
            assert h.leader._eef_mode is False and h.leader._eef is None
            _arm(h)
            used = _run(h, 20)

            assert used == 20 and h.leader._state == EXECUTE, "joint mode must not fault"
            assert not _warnings_matching(h, "EEF joint-envelope clip")
            assert not _warnings_matching(h, "chunk_start")
            assert _warnings_matching(h, "SAFETY CLAMP engaged: joint_limits (OOD)")
            # The clipped target -- not the live pose -- is what reaches the arm.
            assert h.arm.targets[-1][0] == pytest.approx(lo[0], abs=1e-9)


# =============================================================================
# F5 -- ik_in_limits / ik_on_branch are surfaced, and the node warns
# =============================================================================
def test_f5_executor_info_carries_ik_in_limits(carrot_params):
    """The flag exists on the executor's info dict (it used to stop at eef_space)."""
    from gello_policy.eef_executor import EefDeltaExecutor, EefExecutorConfig

    ex = EefDeltaExecutor(EefExecutorConfig(start_pose=tuple(carrot_params["start_pose"])))
    q, info = ex.step(np.asarray(carrot_params["start_pose"], dtype=float),
                      np.zeros(7), chunk_start=True)
    assert q is not None
    assert info["ik_in_limits"] is True
    assert info["ik_on_branch"] is True


def _patch_ik(monkeypatch, *, ok=True, on_branch=True, in_limits=True, branch=2):
    """Replace eef_space.ik_tcp_nearest (the executor calls it by module attribute)."""
    def fake(T_tcp, q_seed, branch=None, limit_margin=0.0):
        info = {"n_solutions": 1, "branch": int(branch if branch is not None else -1),
                "on_branch": on_branch, "max_dev": 0.0, "in_limits": in_limits}
        info["branch"] = 3 if not on_branch else 2
        if not ok:
            return None, {"n_solutions": 0, "branch": -1, "on_branch": False,
                          "max_dev": float("nan"), "in_limits": False}
        return np.asarray(q_seed, dtype=float).copy(), info

    monkeypatch.setattr(eef_space, "ik_tcp_nearest", fake)


@pytest.mark.parametrize("on_branch,in_limits", [(False, True), (True, False),
                                                 (False, False)])
def test_f5_node_warns_on_a_lost_branch_or_an_out_of_limit_solution(
        carrot_params, monkeypatch, on_branch, in_limits):
    _patch_ik(monkeypatch, on_branch=on_branch, in_limits=in_limits)
    with e2e.FakeEefServer([NO_MOVE] * 10, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, 3)
            assert h.leader._state == EXECUTE, "F5 is a WARN, never a hold or a FAULT"
            warns = _warnings_matching(h, "EEF IK left the pinned branch")
            assert warns, h.warnings
            assert f"in_limits={in_limits}" in warns[0]


def test_f5_is_silent_on_a_healthy_tick(carrot_params):
    with e2e.FakeEefServer([NO_MOVE] * 10, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, 5)
            assert not _warnings_matching(h, "EEF IK left the pinned branch")


def test_the_ik_failure_streak_still_faults(carrot_params, monkeypatch):
    """Coverage displaced from the e2e NaN test, which F3 now short-circuits.

    A finite action whose target simply has no IK solution still costs
    ``eef_ik_fail_fault_n`` held ticks and then FAULTs.
    """
    fault_n = int(carrot_params["eef_ik_fail_fault_n"])
    _patch_ik(monkeypatch, ok=False)
    with e2e.FakeEefServer([NO_MOVE] * (fault_n + 5), chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            used = _run(h, fault_n + 5)
            assert h.leader._state == FAULT
            assert used == fault_n
            assert any("EEF IK failed on" in e for e in h.errors), h.errors


# =============================================================================
# F8 -- chunk_start anomalies
# =============================================================================
def test_f8_thresholds_are_above_any_real_horizon():
    """The shipped constants, and why they cannot fire on a healthy run.

    The node has no ``n_action_steps`` parameter (the server owns the horizon and
    does not report it in the handshake), so the silent bound is a fixed tick count
    rather than ``2 x n_action_steps``. It must clear 2x the largest horizon these
    servers actually run: ACT 30, FM 32, diffusion 32 -> 64.
    """
    assert pln.CHUNK_START_SILENT_WARN_TICKS == 200
    assert pln.CHUNK_START_ALWAYS_WARN_TICKS == 60
    assert pln.CHUNK_START_SILENT_WARN_TICKS > 2 * 32


def test_f8_warns_when_chunk_start_is_true_on_every_tick(carrot_params):
    """The degenerate "the net re-runs every tick" case: 169 mm p50 tracking error."""
    n = pln.CHUNK_START_ALWAYS_WARN_TICKS
    with e2e.FakeEefServer([NO_MOVE] * (n + 5), chunk_len=1) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, n - 1)
            assert not _warnings_matching(h, "chunk_start has been TRUE")
            _run(h, 2)
            warns = _warnings_matching(h, "chunk_start has been TRUE")
            assert warns, h.warnings
            assert f"on {n} consecutive ticks" in warns[0]
            assert h.leader._state == EXECUTE, "F8 warns, it never faults"
            assert all(i["chunk_start"] for i in h.infos)


def test_f8_warns_when_chunk_start_never_arrives(carrot_params, monkeypatch):
    """The other degenerate case: one anchor, integrating forever, silently.

    The threshold is monkeypatched down; ``test_f8_thresholds_are_above_any_real_horizon``
    pins the shipped value. The logic under test is identical either way.
    """
    monkeypatch.setattr(pln, "CHUNK_START_SILENT_WARN_TICKS", 8)
    # chunk_len far beyond the run => chunk_start is true on tick 0 only.
    with e2e.FakeEefServer([NO_MOVE] * 30, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, 1)
            assert h.leader._chunk_start_silent_ticks == 0, "tick 0 re-anchors"
            _run(h, 7)
            assert not _warnings_matching(h, "no chunk_start in")
            _run(h, 1)
            warns = _warnings_matching(h, "no chunk_start in")
            assert warns, h.warnings
            assert "no chunk_start in 8 ticks" in warns[0]
            assert h.leader._state == EXECUTE, "F8 warns, it never faults"


def test_f8_a_healthy_chunked_stream_is_silent(carrot_params):
    """ACT's real pattern (true every 30th tick) must never warn."""
    with e2e.FakeEefServer([NO_MOVE] * 100, chunk_len=30) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, 70)
            assert not _warnings_matching(h, "chunk_start has been TRUE")
            assert not _warnings_matching(h, "no chunk_start in")


def test_f8_counters_are_cleared_on_hold(carrot_params):
    with e2e.FakeEefServer([NO_MOVE] * 40, chunk_len=10_000) as server:
        with _eef_harness(carrot_params, server.port) as h:
            _arm(h)
            _run(h, 6)
            assert h.leader._chunk_start_silent_ticks == 5
            assert h.call(h.hold_cli).success
            assert h.leader._chunk_start_silent_ticks == 0
            assert h.leader._chunk_start_run_ticks == 0


# =============================================================================
# F2B -- the handshake detail names the checkpoint
# =============================================================================
def _handshake(**kw):
    body = {"ok": True, "action_space": oa.EEF_ACTION_SPACE_ID, "state_dim": 16,
            "action_dim": 7, "policy_type": "act",
            "checkpoint": "/ckpt/carrot_eef_act_50k"}
    body.update(kw)
    return oa.parse_reset_reply([json.dumps(body).encode("utf-8")])


def test_f2b_handshake_detail_names_the_checkpoint(carrot_params):
    """Without it the log cannot prove WHICH weights answered (stale-server case)."""
    with _eef_harness(carrot_params, 1) as h:
        ok, detail = h.leader._check_handshake(_handshake())
        assert ok
        assert "checkpoint=/ckpt/carrot_eef_act_50k" in detail
        # the fields that were already there are untouched
        for needle in ("action_space=eef_delta_v1", "state_dim=16", "action_dim=7",
                       "policy=act"):
            assert needle in detail


def test_f2b_a_missing_checkpoint_field_is_reported_as_none_not_hidden(carrot_params):
    """A v2 server that omits `checkpoint` must say so, not silently drop the field."""
    with _eef_harness(carrot_params, 1) as h:
        ok, detail = h.leader._check_handshake(_handshake(checkpoint=None))
        assert ok
        assert "checkpoint=None" in detail


def test_f2b_joint_mode_detail_is_unchanged_for_a_pre_v2_server(joint_params):
    """Joint mode + today's servers: the detail is still the empty string."""
    with e2e.Harness(JOINT_YAML, {}, list(joint_params["start_pose"]),
                     float(joint_params["start_gripper"])) as h:
        ok, detail = h.leader._check_handshake(
            oa.parse_reset_reply([b'{"ok":true}']))
        assert ok and detail == ""


def test_f2b_refusals_are_unchanged(carrot_params):
    """The new field must not have softened any gate."""
    with _eef_harness(carrot_params, 1) as h:
        for bad in ({"action_space": oa.JOINT_ACTION_SPACE_ID},
                    {"state_dim": 7}, {"action_dim": 6}):
            ok, detail = h.leader._check_handshake(_handshake(**bad))
            assert not ok, (bad, detail)
        ok, detail = h.leader._check_handshake(
            oa.parse_reset_reply([b'{"ok":true}']))
        assert not ok and "protocol v2" in detail
