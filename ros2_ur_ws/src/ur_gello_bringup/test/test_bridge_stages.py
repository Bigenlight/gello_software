"""Bit-identity + control-mode-branch proof for the refactored command pipeline.

The bridge's ``_on_timer`` was refactored from ONE per-joint interleaved loop
(filter then clamp per joint) into three ordered stages
(``filter_stage_joint`` -> optional eef -> ``clamp_stage``) so an EEF path can be
added WITHOUT regressing the joint-passthrough teleop that runs on the real
robot every day.

The top safety requirement is that JOINT MODE stays **bit-identical** to the
legacy loop. This suite reproduces the exact legacy interleaved loop (both the
one_euro path and the ema+deadband path) and asserts the split stages produce
**byte-for-byte equal** output (``==`` on floats, not a tolerance) over many
randomized trajectories — including clamp-saturating and non-saturating steps,
deadband on/off, and the speed-adaptive one_euro cutoff.

It also proves the ``command_pipeline`` control-mode branch: in joint mode the
eef stage is a no-op (any ``eef_command`` is ignored); in eef+ENGAGED it
overwrites the filtered vector with the controller's q_cmd (or holds on None).
The same proof is repeated for the additive ``joint_delta`` stage (2b), plus the
two guards that matter most for it: joint mode ignores ``joint_delta_command``
entirely, and eef output is byte-identical whether that argument is absent or
garbage (stage 2b is an ``elif`` after the eef branch, so eef never reaches it).

Pure module — no rclpy — so it runs in a plain venv (the code tested IS the code
the node runs).
"""

import random

import pytest

from ur_gello_bringup.bridge_stages import (
    OneEuro,
    clamp_stage,
    command_pipeline,
    filter_stage_joint,
)

N = 6


# --------------------------------------------------------------------------- #
# Reference: the ORIGINAL interleaved _on_timer loop, reproduced verbatim.     #
# --------------------------------------------------------------------------- #
def legacy_tick_euro(raw, filtered, last_published, euro, step):
    """Byte-for-byte the pre-refactor one_euro tick loop."""
    out = []
    for i in range(N):
        filtered[i] = euro[i](raw[i])
        delta = filtered[i] - last_published[i]
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        out.append(last_published[i] + delta)
    return out


def legacy_tick_ema(raw, filtered, gated, last_published, alpha, deadband, step):
    """Byte-for-byte the pre-refactor ema+deadband tick loop."""
    out = []
    for i in range(N):
        if abs(raw[i] - gated[i]) > deadband:
            gated[i] = raw[i]
        filtered[i] = (1.0 - alpha) * filtered[i] + alpha * gated[i]
        delta = filtered[i] - last_published[i]
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        out.append(last_published[i] + delta)
    return out


# --------------------------------------------------------------------------- #
# Split: filter_stage_joint + clamp_stage (what the node now runs).            #
# --------------------------------------------------------------------------- #
def _new_euro_bank(dt=1.0 / 250.0, min_cutoff=1.0, beta=2.0, d_cutoff=1.0):
    return [OneEuro(dt, min_cutoff, beta, d_cutoff) for _ in range(N)]


# --------------------------------------------------------------------------- #
# Bit-identity: one_euro path.                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", list(range(12)))
@pytest.mark.parametrize("step", [1e-4, 0.0025, 1.0])
def test_bit_identical_one_euro(seed, step):
    """filter_stage_joint+clamp_stage == legacy interleaved loop (one_euro)."""
    rng = random.Random(seed)

    # Two INDEPENDENT euro banks, seeded identically; fed identical inputs so
    # their state evolutions stay in lock-step (OneEuro is deterministic).
    ref_euro = _new_euro_bank()
    spl_euro = _new_euro_bank()
    seed_pose = [rng.uniform(-3.0, 3.0) for _ in range(N)]
    for i in range(N):
        ref_euro[i].seed(seed_pose[i])
        spl_euro[i].seed(seed_pose[i])

    ref_filtered = list(seed_pose)
    spl_filtered = list(seed_pose)
    ref_lp = list(seed_pose)
    spl_lp = list(seed_pose)

    raw = list(seed_pose)
    for tick in range(300):
        # Random walk with occasional big jumps to exercise the clamp.
        for i in range(N):
            raw[i] += rng.uniform(-0.02, 0.02)
            if rng.random() < 0.03:
                raw[i] += rng.uniform(-0.5, 0.5)
        # Occasionally feed the speed estimate at the source cadence (same on
        # both banks) so the adaptive cutoff is genuinely exercised.
        if rng.random() < 0.3:
            dt = rng.uniform(0.02, 0.05)
            for i in range(N):
                ref_euro[i].update_input(raw[i], dt)
                spl_euro[i].update_input(raw[i], dt)

        out_ref = legacy_tick_euro(raw, ref_filtered, ref_lp, ref_euro, step)
        filter_stage_joint(raw, spl_filtered, [0.0] * N, spl_euro, 0.0, 0.0)
        out_spl = clamp_stage(spl_filtered, spl_lp, step)

        assert out_ref == out_spl, f"tick {tick}: {out_ref} != {out_spl}"
        assert ref_filtered == spl_filtered
        ref_lp = out_ref
        spl_lp = out_spl


# --------------------------------------------------------------------------- #
# Bit-identity: ema + deadband path.                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", list(range(12)))
@pytest.mark.parametrize("step", [1e-4, 0.0025, 1.0])
@pytest.mark.parametrize("deadband", [0.0, 0.004, 0.02])
def test_bit_identical_ema(seed, step, deadband):
    """filter_stage_joint+clamp_stage == legacy interleaved loop (ema+deadband)."""
    rng = random.Random(seed * 100 + int(deadband * 1000))
    alpha = rng.uniform(0.1, 0.9)

    seed_pose = [rng.uniform(-3.0, 3.0) for _ in range(N)]
    ref_filtered = list(seed_pose)
    spl_filtered = list(seed_pose)
    ref_gated = list(seed_pose)
    spl_gated = list(seed_pose)
    ref_lp = list(seed_pose)
    spl_lp = list(seed_pose)

    raw = list(seed_pose)
    for tick in range(300):
        for i in range(N):
            raw[i] += rng.uniform(-0.02, 0.02)
            if rng.random() < 0.03:
                raw[i] += rng.uniform(-0.5, 0.5)

        out_ref = legacy_tick_ema(
            raw, ref_filtered, ref_gated, ref_lp, alpha, deadband, step
        )
        filter_stage_joint(raw, spl_filtered, spl_gated, None, alpha, deadband)
        out_spl = clamp_stage(spl_filtered, spl_lp, step)

        assert out_ref == out_spl, f"tick {tick}: {out_ref} != {out_spl}"
        assert ref_filtered == spl_filtered
        assert ref_gated == spl_gated
        ref_lp = out_ref
        spl_lp = out_spl


# --------------------------------------------------------------------------- #
# clamp_stage: exact boundary behavior.                                        #
# --------------------------------------------------------------------------- #
def test_clamp_stage_saturation_and_passthrough():
    last = [0.0] * N
    step = 0.1
    # Below step -> pass through exactly.
    filtered = [0.05, -0.05, 0.0, 0.09999, -0.09999, 0.1]
    out = clamp_stage(filtered, last, step)
    assert out == [0.05, -0.05, 0.0, 0.09999, -0.09999, 0.1]
    # Above step -> clamp to +-step relative to last_published.
    filtered = [1.0, -1.0, 0.5, -0.5, 0.2, -0.2]
    out = clamp_stage(filtered, last, step)
    for v in out:
        assert abs(v) <= step + 1e-15
    assert out[0] == step and out[1] == -step


# --------------------------------------------------------------------------- #
# command_pipeline: control-mode branch.                                       #
# --------------------------------------------------------------------------- #
def test_pipeline_joint_mode_ignores_eef_command():
    """In joint mode the eef stage is a NO-OP: any eef_command is ignored and
    the result equals filter_stage_joint+clamp_stage."""
    rng = random.Random(7)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    filtered_a = [0.0] * N
    filtered_b = [0.0] * N
    gated_a = [0.0] * N
    gated_b = [0.0] * N
    last = [0.1 * i for i in range(N)]
    step = 0.05

    # Reference: filter + clamp directly.
    filter_stage_joint(raw, filtered_a, gated_a, None, 0.5, 0.0)
    ref = clamp_stage(filtered_a, last, step)

    # Pipeline in joint mode with a BOGUS eef_command that must be ignored.
    out = command_pipeline(
        "joint", True, raw, filtered_b, gated_b, None, 0.5, 0.0, last, step,
        eef_command=[999.0] * N,
    )
    assert out == ref
    assert filtered_b == filtered_a  # filtered NOT overwritten by eef_command


def test_pipeline_joint_mode_not_engaged_matches():
    """control_mode joint, engaged flag irrelevant."""
    rng = random.Random(11)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    fa, fb = [0.0] * N, [0.0] * N
    ga, gb = [0.0] * N, [0.0] * N
    last = [0.0] * N
    step = 1.0
    filter_stage_joint(raw, fa, ga, None, 0.3, 0.0)
    ref = clamp_stage(fa, last, step)
    out = command_pipeline("joint", False, raw, fb, gb, None, 0.3, 0.0, last, step,
                           eef_command=[42.0] * N)
    assert out == ref


def test_pipeline_eef_engaged_overwrites_with_eef_command():
    """eef mode + engaged: filtered is overwritten by the eef controller output,
    then clamped relative to last_published."""
    raw = [0.7] * N
    filtered = [0.0] * N
    gated = [0.0] * N
    last = [0.0] * N
    step = 10.0  # large so the clamp does not bind
    eef_cmd = [0.11, -0.22, 0.33, -0.44, 0.05, -0.06]
    out = command_pipeline(
        "eef", True, raw, filtered, gated, None, 0.5, 0.0, last, step,
        eef_command=eef_cmd,
    )
    # filtered was overwritten by the eef command...
    assert filtered == eef_cmd
    # ...and the (non-binding) clamp passed it through.
    assert out == eef_cmd


def test_pipeline_eef_engaged_clamp_still_binds():
    """Even in eef mode the downstream clamp still bounds the per-tick step."""
    last = [0.0] * N
    step = 0.01
    eef_cmd = [1.0] * N  # far beyond step
    out = command_pipeline(
        "eef", True, [0.0] * N, [0.0] * N, [0.0] * N, None, 0.5, 0.0, last, step,
        eef_command=eef_cmd,
    )
    assert out == [step] * N


def test_pipeline_eef_engaged_none_holds():
    """eef mode + engaged + eef_command None (HOLD): output == last_published
    (zero delta), regardless of what the joint filter produced."""
    raw = [0.9] * N
    filtered = [0.0] * N
    gated = [0.0] * N
    last = [0.05 * i for i in range(N)]
    step = 1.0
    out = command_pipeline(
        "eef", True, raw, filtered, gated, None, 0.5, 0.0, last, step,
        eef_command=None,
    )
    assert out == last  # held exactly


def test_pipeline_eef_not_engaged_is_joint_passthrough():
    """eef mode but NOT engaged (BOOTSTRAP) behaves as joint passthrough:
    eef_command ignored, joint filter drives the output."""
    rng = random.Random(3)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    fa, fb = [0.0] * N, [0.0] * N
    ga, gb = [0.0] * N, [0.0] * N
    last = [0.0] * N
    step = 1.0
    filter_stage_joint(raw, fa, ga, None, 0.4, 0.0)
    ref = clamp_stage(fa, last, step)
    out = command_pipeline("eef", False, raw, fb, gb, None, 0.4, 0.0, last, step,
                           eef_command=[7.0] * N)
    assert out == ref


# --------------------------------------------------------------------------- #
# command_pipeline: joint_delta stage (2b) — and the guards proving the two     #
# pre-existing modes are untouched by its presence.                            #
# --------------------------------------------------------------------------- #
def test_pipeline_joint_mode_ignores_joint_delta_command():
    """In joint mode stage 2b is a NO-OP: any joint_delta_command is ignored and
    the result equals filter_stage_joint+clamp_stage (mirror of the eef guard)."""
    rng = random.Random(17)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    filtered_a = [0.0] * N
    filtered_b = [0.0] * N
    gated_a = [0.0] * N
    gated_b = [0.0] * N
    last = [0.1 * i for i in range(N)]
    step = 0.05

    filter_stage_joint(raw, filtered_a, gated_a, None, 0.5, 0.0)
    ref = clamp_stage(filtered_a, last, step)

    out = command_pipeline(
        "joint", True, raw, filtered_b, gated_b, None, 0.5, 0.0, last, step,
        joint_delta_command=[999.0] * N,
    )
    assert out == ref
    assert filtered_b == filtered_a  # NOT overwritten


def test_pipeline_eef_engaged_unaffected_by_joint_delta_command():
    """eef output is byte-identical whether joint_delta_command is None or
    garbage — stage 2b is an elif AFTER the eef branch, so eef never sees it."""
    eef_cmd = [0.11, -0.22, 0.33, -0.44, 0.05, -0.06]
    kwargs = dict(raw_target=[0.7] * N, gated_target=[0.0] * N, euro=None,
                  ema_alpha=0.5, deadband=0.0, last_published=[0.0] * N,
                  step=10.0)
    a = command_pipeline("eef", True, kwargs["raw_target"], [0.0] * N,
                         list(kwargs["gated_target"]), None, 0.5, 0.0,
                         kwargs["last_published"], kwargs["step"],
                         eef_command=eef_cmd)
    b = command_pipeline("eef", True, kwargs["raw_target"], [0.0] * N,
                         list(kwargs["gated_target"]), None, 0.5, 0.0,
                         kwargs["last_published"], kwargs["step"],
                         eef_command=eef_cmd,
                         joint_delta_command=[-42.0] * N)
    assert a == b == eef_cmd


def test_pipeline_joint_delta_engaged_overwrites():
    """joint_delta + engaged: filtered is overwritten by the controller output,
    then clamped relative to last_published."""
    raw = [0.7] * N
    filtered = [0.0] * N
    gated = [0.0] * N
    last = [0.0] * N
    step = 10.0  # large so the clamp does not bind
    jd_cmd = [0.11, -0.22, 0.33, -0.44, 0.05, -0.06]
    out = command_pipeline(
        "joint_delta", True, raw, filtered, gated, None, 0.5, 0.0, last, step,
        joint_delta_command=jd_cmd,
    )
    assert filtered == jd_cmd
    assert out == jd_cmd


def test_pipeline_joint_delta_engaged_none_holds():
    """joint_delta + engaged + command None (HOLD): output == last_published."""
    raw = [0.9] * N
    filtered = [0.0] * N
    gated = [0.0] * N
    last = [0.05 * i for i in range(N)]
    step = 1.0
    out = command_pipeline(
        "joint_delta", True, raw, filtered, gated, None, 0.5, 0.0, last, step,
        joint_delta_command=None,
    )
    assert out == last  # held exactly


def test_pipeline_joint_delta_not_engaged_is_joint_passthrough():
    """joint_delta mode but NOT engaged (JOINT_BOOTSTRAP) is ordinary absolute
    joint passthrough — NOT the eef 3D-pen hold."""
    rng = random.Random(23)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    fa, fb = [0.0] * N, [0.0] * N
    ga, gb = [0.0] * N, [0.0] * N
    last = [0.0] * N
    step = 1.0
    filter_stage_joint(raw, fa, ga, None, 0.4, 0.0)
    ref = clamp_stage(fa, last, step)
    out = command_pipeline("joint_delta", False, raw, fb, gb, None, 0.4, 0.0,
                           last, step, joint_delta_command=[7.0] * N)
    assert out == ref


def test_pipeline_joint_delta_clamp_still_binds():
    """Even in joint_delta mode the downstream clamp still bounds the per-tick
    step — the controller's own pre-clamp is defence in depth, not a bypass."""
    last = [0.0] * N
    step = 0.01
    jd_cmd = [1.0] * N  # far beyond step
    out = command_pipeline(
        "joint_delta", True, [0.0] * N, [0.0] * N, [0.0] * N, None, 0.5, 0.0,
        last, step, joint_delta_command=jd_cmd,
    )
    assert out == [step] * N


# =========================================================================== #
# 3D-PEN HOLD (hold_when_not_engaged) — pure pipeline.                        #
#                                                                             #
# In eef mode the leader and the robot deliberately live in DIFFERENT joint   #
# configurations forever ("GELLO as a 3D pen"), so the pre-engage state must   #
# HOLD, never mirror the leader's joint shape. The kwarg defaults to False so  #
# joint mode — and the deliberate ~/eef_to_joint hand-back — are unchanged.    #
# =========================================================================== #
@pytest.mark.parametrize("seed", list(range(12)))
@pytest.mark.parametrize("step", [1e-4, 0.0025, 1.0])
@pytest.mark.parametrize("hold_flag", [False, True])
def test_joint_mode_bit_identical_regardless_of_hold_flag(seed, step, hold_flag):
    """JOINT MODE IS UNTOUCHED by the hold flag — still byte-for-byte the legacy
    interleaved loop over 300 ticks, with hold_when_not_engaged either way.

    This is the same pinning contract as test_bit_identical_one_euro, re-run
    against the NEW kwarg: stage 2 is skipped entirely when control_mode is
    "joint", so no value of the flag (and no eef_command) can reach the output.
    """
    rng = random.Random(seed)
    ref_euro = _new_euro_bank()
    spl_euro = _new_euro_bank()
    seed_pose = [rng.uniform(-3.0, 3.0) for _ in range(N)]
    for i in range(N):
        ref_euro[i].seed(seed_pose[i])
        spl_euro[i].seed(seed_pose[i])

    ref_filtered = list(seed_pose)
    spl_filtered = list(seed_pose)
    ref_lp = list(seed_pose)
    spl_lp = list(seed_pose)

    raw = list(seed_pose)
    for tick in range(300):
        for i in range(N):
            raw[i] += rng.uniform(-0.02, 0.02)
            if rng.random() < 0.03:
                raw[i] += rng.uniform(-0.5, 0.5)
        if rng.random() < 0.3:
            dt = rng.uniform(0.02, 0.05)
            for i in range(N):
                ref_euro[i].update_input(raw[i], dt)
                spl_euro[i].update_input(raw[i], dt)

        out_ref = legacy_tick_euro(raw, ref_filtered, ref_lp, ref_euro, step)
        # engaged=True AND a bogus eef_command AND the hold flag: in joint mode
        # every one of them must be inert.
        out_spl = command_pipeline(
            "joint", True, raw, spl_filtered, [0.0] * N, spl_euro, 0.0, 0.0,
            spl_lp, step, eef_command=[999.0] * N,
            hold_when_not_engaged=hold_flag,
        )
        assert out_ref == out_spl, f"tick {tick}: {out_ref} != {out_spl}"
        assert ref_filtered == spl_filtered
        ref_lp = out_ref
        spl_lp = out_spl


@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize("deadband", [0.0, 0.004])
def test_joint_mode_ema_bit_identical_regardless_of_hold_flag(seed, deadband):
    """Same pinning contract on the ema+deadband path."""
    rng = random.Random(seed * 977 + int(deadband * 1000))
    alpha = rng.uniform(0.1, 0.9)
    step = 0.0025
    seed_pose = [rng.uniform(-3.0, 3.0) for _ in range(N)]
    ref_filtered, spl_filtered = list(seed_pose), list(seed_pose)
    ref_gated, spl_gated = list(seed_pose), list(seed_pose)
    ref_lp, spl_lp = list(seed_pose), list(seed_pose)

    raw = list(seed_pose)
    for tick in range(300):
        for i in range(N):
            raw[i] += rng.uniform(-0.02, 0.02)
        out_ref = legacy_tick_ema(
            raw, ref_filtered, ref_gated, ref_lp, alpha, deadband, step
        )
        out_spl = command_pipeline(
            "joint", False, raw, spl_filtered, spl_gated, None, alpha, deadband,
            spl_lp, step, hold_when_not_engaged=True,
        )
        assert out_ref == out_spl, f"tick {tick}: {out_ref} != {out_spl}"
        assert ref_gated == spl_gated
        ref_lp = out_ref
        spl_lp = out_spl


def test_pipeline_eef_hold_when_not_engaged_holds_exactly():
    """eef + NOT engaged + hold flag: output is EXACTLY last_published.

    Not "close to" — the clamp delta must be identically zero, so the arm does
    not creep even one max_step per tick toward the leader.
    """
    last = [0.31, -1.02, 1.44, -0.07, 2.10, -3.00]
    raw = [v + 2.5 for v in last]  # leader miles away (different configuration)
    filtered = list(last)
    gated = list(last)
    out = command_pipeline(
        "eef", False, raw, filtered, gated, None, 0.5, 0.0, last, 0.0025,
        hold_when_not_engaged=True,
    )
    assert out == last


def test_pipeline_eef_hold_repeated_ticks_never_drift():
    """Holding for many ticks with a far-away leader produces NO motion at all."""
    last = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    raw = [2.0, -0.5, 0.9, -2.0, 1.0, 3.0]
    filtered = list(last)
    gated = list(last)
    lp = list(last)
    for _ in range(500):
        lp = command_pipeline(
            "eef", False, raw, filtered, gated, None, 0.5, 0.0, lp, 0.0025,
            hold_when_not_engaged=True,
        )
    assert lp == last  # bit-for-bit the pose we started from


def test_pipeline_eef_hold_flag_defaults_to_passthrough():
    """OMITTING the kwarg keeps the legacy eef BOOTSTRAP passthrough.

    This is the ~/eef_to_joint path: after the operator asks to go back to joint
    teleop, the arm must once again mirror the leader so it can be re-aligned.
    """
    rng = random.Random(23)
    raw = [rng.uniform(-1, 1) for _ in range(N)]
    fa, fb = [0.0] * N, [0.0] * N
    ga, gb = [0.0] * N, [0.0] * N
    last = [0.0] * N
    step = 1.0
    filter_stage_joint(raw, fa, ga, None, 0.4, 0.0)
    ref = clamp_stage(fa, last, step)
    out = command_pipeline("eef", False, raw, fb, gb, None, 0.4, 0.0, last, step)
    assert out == ref
    assert out != last  # it really did move toward the leader


def test_pipeline_eef_engaged_ignores_hold_flag():
    """ENGAGED wins over the hold flag: the controller command still drives."""
    last = [0.0] * N
    eef_cmd = [0.001 * (i + 1) for i in range(N)]
    out = command_pipeline(
        "eef", True, [5.0] * N, [0.0] * N, [0.0] * N, None, 0.5, 0.0, last, 1.0,
        eef_command=eef_cmd, hold_when_not_engaged=True,
    )
    assert out == eef_cmd


def test_pipeline_hold_still_advances_the_joint_filter_stage():
    """Stage 1 keeps running while stage 2 holds.

    The joint filter bank must stay live so a later return to passthrough starts
    from current state; observable through gated_target, which stage 1 mutates.
    """
    last = [0.0] * N
    raw = [1.0] * N
    gated = [0.0] * N
    filtered = [0.0] * N
    out = command_pipeline(
        "eef", False, raw, filtered, gated, None, 0.5, 0.0, last, 0.0025,
        hold_when_not_engaged=True,
    )
    assert out == last          # ... the arm still did not move ...
    assert gated == [1.0] * N   # ... but the joint filter tracked the leader.


# =========================================================================== #
# G6 ROOT CAUSE: OneEuro output only advances on __call__.                     #
# =========================================================================== #
def test_one_euro_output_frozen_when_only_update_input_is_called():
    """update_input() advances the SPEED estimate, NOT the output.

    This is exactly the engage-gate bug: the bridge fed the leader filter via
    update_input() on every GELLO message but never CALLED it outside the
    ENGAGED stage, so at engage time the filter output was still the seed. Gate
    G6 then compared that seed against the live leader, making "settled" mean
    "the leader has not moved since the seed" — measured refusals on the real
    rig were 0.29288 / 0.97110 / 1.30802 / 1.35317 rad against a 0.005 tol.
    """
    f = OneEuro(1.0 / 250.0, 1.0, 2.0, 1.0)
    f.seed(0.0)
    x = 0.0
    for _ in range(30):           # ~1 s of GELLO samples at 30 Hz
        x += 0.01                 # a slow, deliberate 0.3 rad reposition
        f.update_input(x, 1.0 / 30.0)
    # Nothing has advanced the output: it is still the seed, 0.3 rad away.
    assert f._x_prev == 0.0
    settle = abs(f._x_prev - x)
    assert settle > 0.25
    assert settle > 0.005         # ... i.e. G6 could never pass. QED.


def test_one_euro_lag_settles_when_stepped_every_tick():
    """Stepped once per publish tick, the residual lag is a real, small lag.

    With the shipped tuning (250 Hz, min_cutoff 1.0, beta 2.0) a leader held
    quasi-still (here 0.02 rad/s, well inside resume_chase_still_speed=0.10)
    settles to a lag comfortably below filter_settled_tol=0.005 — so the fixed
    G6 is a gate a human can actually satisfy, and it now measures what it
    claims to measure (filter convergence), not "has the operator moved".
    """
    dt = 1.0 / 250.0
    f = OneEuro(dt, 1.0, 2.0, 1.0)
    x = 0.0
    f.seed(x)
    speed = 0.02  # rad/s — a hand holding the leader "still"
    out = x
    for tick in range(2000):      # 8 s
        x += speed * dt
        if tick % 8 == 0:         # GELLO cadence ~30 Hz
            f.update_input(x, 8 * dt)
        out = f(x)                # <-- the per-tick step the bridge now does
    lag = abs(out - x)
    assert lag < 0.005, f"steady-state lag {lag:.5f} >= filter_settled_tol"
    # And the frozen-seed comparison over the same motion is hopeless:
    assert abs(0.0 - x) > 0.15


# =========================================================================== #
# NODE-LEVEL tests (rclpy). These exercise the BRIDGE ITSELF — the lifecycle   #
# plumbing that the pure stage tests above cannot reach: the /joint_states     #
# staleness watchdog, the 3D-pen hold bootstrap, the per-tick leader-filter    #
# step behind gate G6, and the ~/eef_resume re-arm path.                       #
#                                                                             #
# They construct the real node and call its callbacks directly (no executor,   #
# no spinning, no robot), so they stay fast and deterministic. Skipped when    #
# rclpy is unavailable, which keeps the pure-venv promise of this file.        #
# =========================================================================== #
rclpy = pytest.importorskip("rclpy", reason="node-level tests need a ROS 2 env")

from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from ur_gello_bringup.gello_ur_bridge_node import (  # noqa: E402
    UR_JOINT_ORDER,
    GelloUrBridge,
)

# A well-conditioned, away-from-singularity arm pose (UR "ready"-ish).
ARM_POSE = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
# A leader pose in a COMPLETELY different joint configuration — the 3D-pen
# premise. Every joint differs by ~0.9 to ~3.0 rad.
LEADER_POSE = [2.0, -0.5, 0.9, -2.0, 1.0, 3.0]


class _Bridge:
    """Context manager: a real GelloUrBridge with captured publishes."""

    def __init__(self, **params):
        self._params = params
        self.node = None
        self.published = []
        self.states = []

    def __enter__(self):
        args = ["--ros-args"]
        for k, v in self._params.items():
            args += ["-p", f"{k}:={v}"]
        rclpy.init(args=args)
        self.node = GelloUrBridge()
        self.node._publish = lambda p: self.published.append(list(p))
        self.node._state_pub = _CapturePub(self.states)
        return self

    def __exit__(self, *exc):
        try:
            self.node.destroy_node()
        finally:
            rclpy.shutdown()
        return False

    # -- input helpers -------------------------------------------------
    def leader(self, pose):
        self.node._on_joint_state(_js(pose))

    def arm(self, pose):
        self.node._on_actual_joint_state(_js(pose))

    def ticks(self, n):
        for _ in range(n):
            self.node._on_timer()

    def call(self, name):
        resp = getattr(self.node, name)(Trigger.Request(), Trigger.Response())
        return resp


class _CapturePub:
    def __init__(self, sink):
        self._sink = sink

    def publish(self, msg):
        self._sink.append(msg.data)


def _js(pose):
    m = JointState()
    m.name = list(UR_JOINT_ORDER)
    m.position = [float(p) for p in pose]
    return m


def _eef_params(**over):
    p = dict(
        control_mode="eef",
        publish_rate_hz=250.0,
        max_step_rad=0.0025,
        soft_start_s=0.0,
        filter_type="one_euro",
    )
    p.update(over)
    return p


def _jd_params(**over):
    p = dict(
        control_mode="joint_delta",
        publish_rate_hz=250.0,
        max_step_rad=0.0025,
        soft_start_s=0.0,
        filter_type="one_euro",
        jd_gain=1.0,
    )
    p.update(over)
    return p


def _seed_still_leader_history(node, pose, window_s=0.3):
    """Inject two window-spanning, zero-speed leader samples so the
    leader_quasi_still gate (c) in ~/joint_delta_start can POSITIVELY gate.

    A rapid burst of _on_joint_state() calls in a unit test spans far less
    than window_s*0.5, so the span guard would (correctly) fail closed; this
    helper stamps a real time span onto two identical poses instead.
    """
    import time as _time
    now = _time.monotonic()
    node._gello_history.clear()
    node._gello_history.append((now - window_s, list(pose)))
    node._gello_history.append((now, list(pose)))


# --------------------------------------------------------------------------- #
# TASK 3: eef mode boots into HOLD and never mirrors the leader's joints.      #
# --------------------------------------------------------------------------- #
def test_eef_mode_boots_into_hold_and_never_mirrors_the_leader():
    with _Bridge(**_eef_params()) as b:
        assert b.node._eef_state == "HOLD"
        assert b.node._eef_hold_active() is True
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(500)                      # 2 s at 250 Hz
        assert b.published, "bridge published nothing at all"
        # Seeded from the ARM, and never moved a single step toward the leader.
        assert b.published[0] == ARM_POSE
        assert b.published[-1] == ARM_POSE
        assert all(p == ARM_POSE for p in b.published)


def test_eef_to_joint_restores_joint_passthrough():
    """~/eef_to_joint is the ONE documented exit from the 3D-pen hold: it must
    still hand back to joint passthrough so the operator can re-align."""
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(10)
        b.call("_on_eef_to_joint")
        assert b.node._eef_state == "JOINT_BOOTSTRAP"
        assert b.node._eef_hold_active() is False
        # eef_to_joint pauses; the operator resumes (leader is far, so use the
        # gated chase path's state transition directly rather than its gate).
        b.node._paused = False
        n_before = len(b.published)
        b.ticks(400)
        moved = [
            abs(b.published[-1][i] - b.published[n_before][i])
            for i in range(len(UR_JOINT_ORDER))
        ]
        assert max(moved) > 0.05, "JOINT_BOOTSTRAP must chase the leader again"


# --------------------------------------------------------------------------- #
# TASK 3 + joint-mode safety: joint mode is unchanged at the NODE level.       #
# --------------------------------------------------------------------------- #
def test_joint_mode_node_output_matches_the_pure_pipeline():
    """The node's published sequence in joint mode is byte-for-byte what the
    pure (legacy-pinned) pipeline produces — i.e. none of the eef/hold/watchdog
    work changed a single published number in the healthy joint case."""
    with _Bridge(control_mode="joint", publish_rate_hz=250.0,
                 max_step_rad=0.0025, soft_start_s=0.0, ema_alpha=0.5) as b:
        arm = list(ARM_POSE)
        # Keep the leader within +-pi of the arm so wrapped_nearest is identity
        # and the reference below needs no re-anchoring.
        lead = [arm[i] + 0.4 for i in range(len(arm))]
        b.arm(arm)
        b.leader(lead)
        b.ticks(200)

    # Reference: seed tick publishes the actual pose, then the pure pipeline.
    ref = [list(arm)]
    filtered = list(arm)
    gated = list(lead)
    lp = list(arm)
    for _ in range(199):
        lp = command_pipeline(
            "joint", False, lead, filtered, gated, None, 0.5, 0.0, lp, 0.0025,
        )
        ref.append(list(lp))
    assert b.published == ref


# --------------------------------------------------------------------------- #
# TASK 2: /joint_states staleness watchdog.                                    #
# --------------------------------------------------------------------------- #
def test_actual_joint_state_watchdog_stops_publishing_and_reseeds():
    """A frozen /joint_states must stop the stream AND force a re-seed, so the
    arm is never re-seeded from a pose it no longer holds."""
    import time as _time
    with _Bridge(control_mode="joint", publish_rate_hz=250.0,
                 max_step_rad=0.0025, soft_start_s=0.0,
                 actual_staleness_timeout_s=0.2) as b:
        b.arm(ARM_POSE)
        b.leader([v + 0.3 for v in ARM_POSE])
        b.ticks(20)
        assert len(b.published) == 20          # healthy: publishing normally

        # External Control drops: no new /joint_states, pose freezes.
        b.node._actual_pose_time = _time.monotonic() - 1.0
        n = len(b.published)
        b.ticks(50)
        assert len(b.published) == n, "kept publishing against a stale robot pose"
        assert b.node._last_published is None   # re-seed armed
        assert b.node._filtered is None

        # Recovery — and the arm turns out to have MOVED while we were blind.
        moved_pose = [v + 0.25 for v in ARM_POSE]
        b.arm(moved_pose)
        b.ticks(1)
        # The re-seed used the NEW pose, not the fossil: zero jump preserved.
        assert b.published[-1] == moved_pose


def test_actual_joint_state_watchdog_is_inert_when_healthy():
    """Healthy joint mode is untouched: a fresh /joint_states never trips it."""
    with _Bridge(control_mode="joint", publish_rate_hz=250.0,
                 max_step_rad=0.0025, soft_start_s=0.0) as b:
        b.arm(ARM_POSE)
        b.leader([v + 0.3 for v in ARM_POSE])
        for _ in range(100):
            b.arm(ARM_POSE)      # broadcaster keeps streaming
            b.node._on_timer()
        assert len(b.published) == 100
        assert b.node._last_published is not None


def test_actual_staleness_autodisengages_in_eef_with_a_distinct_reason():
    import time as _time
    with _Bridge(**_eef_params(actual_staleness_timeout_s=0.2)) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(5)
        b.node._eef_state = "ENGAGED"          # pretend a live session
        b.node._actual_pose_time = _time.monotonic() - 1.0
        b.ticks(1)
        assert b.node._eef_state == "DISENGAGED"
        assert b.node._paused is True
        assert b.node._eef_auto_reason == "robot_joint_state_stale"


# --------------------------------------------------------------------------- #
# TASK 1: the leader filter is stepped every tick and G6 reads the cache.      #
# --------------------------------------------------------------------------- #
def test_leader_filter_is_stepped_every_tick_in_eef_mode():
    """The cache must ADVANCE on every tick with no service call involved.

    Before the fix the only __call__ outside the ENGAGED stage was inside gate
    G6 itself, so between engages the filter output never moved at all.
    """
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(1)                       # seed tick: cache deliberately voided
        assert b.node._q_lead_f is None
        # Leader is repositioned (the clutch use-case) and then held.
        moved = [v + 0.3 for v in LEADER_POSE]
        b.leader(moved)
        b.ticks(1)
        first = list(b.node._q_lead_f)
        b.ticks(1)
        second = list(b.node._q_lead_f)
        # It MOVED between ticks -> the filter is genuinely advancing, and it is
        # converging on the LIVE leader (not stuck at its seed).
        assert first != second
        assert abs(second[0] - moved[0]) < abs(first[0] - moved[0])
        b.ticks(600)
        assert b.node._q_lead_f[0] == pytest.approx(moved[0], abs=1e-4)


def test_g6_settles_against_a_live_leader_and_is_not_a_leader_stillness_test():
    """After the arm holds and the leader sits still, G6's residual is tiny —
    even though the leader is ~2 rad away from the arm in joint space.

    The old implementation compared the filter SEED to the live leader, so any
    leader reposition (the entire point of a clutch) made G6 unsatisfiable.
    """
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(600)
        settle = max(
            abs(b.node._q_lead_f[i] - b.node._raw_target[i])
            for i in range(len(UR_JOINT_ORDER))
        )
        assert settle < b.node.filter_settled_tol, f"settle={settle:.6f}"
        # And this held while the leader/arm joint gap stayed enormous.
        gap = max(
            abs(LEADER_POSE[i] - ARM_POSE[i]) for i in range(len(UR_JOINT_ORDER))
        )
        assert gap > 2.0


def test_g6_fails_closed_when_the_filter_is_not_being_stepped():
    """A paused bridge stops stepping the filter, so the cache goes stale and
    G6 refuses rather than certifying a frozen filter."""
    import time as _time
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(50)
        # Satisfy the EARLIER gate G5 (leader quasi-still) with a genuine, time-
        # spanning still history, so the refusal we observe is unambiguously G6.
        now = _time.monotonic()
        b.node._gello_history.clear()
        for dt in (0.30, 0.15, 0.0):
            b.node._gello_history.append((now - dt, list(LEADER_POSE)))
        ok, _key, _detail, _, _ = b.node._run_eef_gates(run_baseline=False)
        assert ok is True, (_key, _detail)          # control: gates pass now

        b.node._q_lead_f_time = _time.monotonic() - 5.0   # bridge was paused
        ok, key, detail, _, _ = b.node._run_eef_gates(run_baseline=False)
        assert ok is False
        assert key == "filter_not_running", (key, detail)

        b.node._q_lead_f = None                          # never stepped at all
        b.node._q_lead_f_time = None
        ok, key, _detail, _, _ = b.node._run_eef_gates(run_baseline=False)
        assert ok is False and key == "filter_not_running"


# --------------------------------------------------------------------------- #
# TASK 4: engage/_paused trap, eef_resume, reclutch.                           #
# --------------------------------------------------------------------------- #
def test_engage_is_refused_while_paused():
    """The measured trap: engage used to succeed on a PAUSED bridge, reporting
    ENGAGED while the arm was dead ({"eef":"ENGAGED","bridge":"PAUSED"})."""
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(50)
        b.call("_on_eef_disengage")
        assert b.node._paused is True
        resp = b.call("_on_eef_engage")
        assert resp.success is False
        assert "bridge_paused" in resp.message
        assert "eef_resume" in resp.message
        assert b.node._eef_state == "DISENGAGED"   # NOT flipped to ENGAGED


def test_eef_state_topic_never_reports_engaged_while_paused():
    import json as _json
    with _Bridge(**_eef_params()) as b:
        sink = []
        b.node._eef_state_pub = _CapturePub(sink)
        b.node._eef_state = "ENGAGED"     # force the invariant violation
        b.node._paused = True
        b.node._on_eef_state_timer()
        payload = _json.loads(sink[-1])
        assert payload["state"] == "DISENGAGED"
        assert payload["paused"] is True


def test_eef_resume_lands_in_hold_not_joint_passthrough():
    """The safety-critical property: re-arming after a fault must NOT route
    through joint passthrough, which would slew the arm across the cell toward
    the leader with none of the engaged-mode protections."""
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(50)
        # A fault: any route ends here.
        b.node._eef_state = "ENGAGED"
        b.node._eef_autodisengage("hold_latched (test)")
        assert b.node._paused is True

        resp = b.call("_on_eef_resume")
        assert resp.success is True, resp.message
        assert b.node._eef_state == "HOLD"
        assert b.node._eef_state != "JOINT_BOOTSTRAP"
        assert b.node._eef_hold_active() is True
        assert b.node._paused is False

        n = len(b.published)
        b.ticks(500)
        after = b.published[n:]
        assert after, "eef_resume did not restart publishing"
        # Re-seeded from the arm's actual pose and held there — zero motion.
        assert all(p == ARM_POSE for p in after)


def test_eef_resume_is_refused_when_the_robot_pose_is_stale():
    import time as _time
    with _Bridge(**_eef_params(actual_staleness_timeout_s=0.2)) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(20)
        b.call("_on_eef_disengage")
        b.node._actual_pose_time = _time.monotonic() - 5.0
        resp = b.call("_on_eef_resume")
        assert resp.success is False
        assert "STALE" in resp.message
        assert b.node._paused is True          # stays paused, publishes nothing


def test_eef_resume_is_refused_when_the_leader_is_stale():
    import time as _time
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(20)
        b.call("_on_eef_disengage")
        b.node._last_good_msg_time = _time.monotonic() - 5.0
        resp = b.call("_on_eef_resume")
        assert resp.success is False
        assert "GELLO" in resp.message
        assert b.node._paused is True


def test_eef_reclutch_still_requires_engaged():
    """Reclutch skips G3/G4, so it must stay confined to a live engaged session;
    the fault-recovery path is eef_resume + eef_engage (full G0..G9)."""
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(20)
        b.call("_on_eef_disengage")
        resp = b.call("_on_eef_reclutch")
        assert resp.success is False
        assert "eef_resume" in resp.message


# --------------------------------------------------------------------------- #
# joint_delta switch_only no-motion bring-up: ~/joint_delta_start must arm a    #
# PAUSED, NEVER-STREAMED bridge when (and only when) jd_start_allow_unstreamed  #
# is set — the additive gate relaxation for the switch_only startup path.       #
# --------------------------------------------------------------------------- #
def test_joint_delta_start_arms_never_streamed_bridge_when_opted_in():
    """The switch_only bring-up: the bridge is pre-spawned start_paused, the
    STRICT switch happens IN PLACE (nothing streams), then joint_delta_start is
    called. With jd_start_allow_unstreamed:=True the _has_streamed gate (a2) is
    bypassed and the service arms straight into ENGAGED with a ZERO-jump first
    command == the arm's ACTUAL pose, regardless of the leader's pose."""
    with _Bridge(**_jd_params(start_paused=True,
                              jd_start_allow_unstreamed=True)) as b:
        # The switch_only scenario: paused from spawn, never streamed.
        assert b.node._paused is True
        assert b.node._has_streamed is False
        assert b.node._jd_state == "JOINT_BOOTSTRAP"

        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)                 # fresh leader (raw_target + time)
        _seed_still_leader_history(b.node, LEADER_POSE)

        resp = b.call("_on_jd_start")
        assert resp.success is True, resp.message
        assert b.node._jd_state == "ENGAGED"
        assert b.node._paused is False

        # First published command is the arm's actual pose, ALGEBRAICALLY zero
        # jump — never the (completely different) leader pose.
        n = len(b.published)
        b.ticks(300)
        after = b.published[n:]
        assert after, "joint_delta_start did not restart publishing"
        assert after[0] == ARM_POSE
        # Leader held still after the anchor, so the arm never moves off it.
        assert all(p == ARM_POSE for p in after)


def test_joint_delta_start_still_refuses_never_streamed_by_default():
    """Default (jd_start_allow_unstreamed unset == False): the manual-recovery
    gate is UNCHANGED. A never-streamed bridge is refused, stays PAUSED, and
    publishes nothing — the exact behaviour every existing caller relies on."""
    with _Bridge(**_jd_params(start_paused=True)) as b:
        assert b.node.jd_start_allow_unstreamed is False
        assert b.node._has_streamed is False

        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        _seed_still_leader_history(b.node, LEADER_POSE)

        resp = b.call("_on_jd_start")
        assert resp.success is False
        assert "never streamed" in resp.message
        assert b.node._paused is True
        assert b.node._jd_state == "JOINT_BOOTSTRAP"
        n = len(b.published)
        b.ticks(50)
        assert b.published[n:] == [], "refused start must publish nothing"


# --------------------------------------------------------------------------- #
# TASK 3: ~/state must not say CHASING forever under a hold bootstrap.         #
# --------------------------------------------------------------------------- #
def test_state_topic_reports_hold_under_the_pen_bootstrap():
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(20)
        b.node._on_state_timer()
        assert b.states[-1] == "HOLD"


def test_state_topic_reports_eef_engaged_not_chasing():
    with _Bridge(**_eef_params()) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(20)
        b.node._eef_state = "ENGAGED"
        b.node._on_state_timer()
        assert b.states[-1] == "EEF_ENGAGED"


def test_state_topic_reports_stale_robot():
    import time as _time
    with _Bridge(control_mode="joint", publish_rate_hz=250.0,
                 actual_staleness_timeout_s=0.2) as b:
        b.arm(ARM_POSE)
        b.leader(ARM_POSE)
        b.ticks(5)
        b.node._actual_pose_time = _time.monotonic() - 5.0
        b.node._on_state_timer()
        assert b.states[-1] == "STALE_ROBOT"


def test_state_topic_joint_mode_still_reports_chasing_and_following():
    """Joint mode keeps its original labels."""
    with _Bridge(control_mode="joint", publish_rate_hz=250.0,
                 max_step_rad=0.0025, soft_start_s=0.0,
                 state_chase_done_tol=0.1) as b:
        b.arm(ARM_POSE)
        b.leader([v + 0.5 for v in ARM_POSE])
        b.ticks(5)
        b.node._on_state_timer()
        assert b.states[-1] == "CHASING"
        b.ticks(1500)                       # let it converge
        b.node._on_state_timer()
        assert b.states[-1] == "FOLLOWING"


# --------------------------------------------------------------------------- #
# Tick-budget watchdog: SOFT budget warns, HARD budget fails closed.           #
# --------------------------------------------------------------------------- #
class _SlowEef:
    """Stand-in EefDeltaController whose step() burns a fixed wall time."""

    def __init__(self, micros):
        self._us = micros
        self.disengaged = 0
        self.keepout = {}

    def step(self, q_lead_f, step_eff):
        import time as _t
        t0 = _t.monotonic()
        while (_t.monotonic() - t0) * 1e6 < self._us:
            pass
        return None, {"state": "OK", "reject_reason": None}

    def disengage(self):
        self.disengaged += 1


def test_soft_budget_overrun_warns_but_does_not_disengage():
    """Measured reality: step() runs ~2054us mean / 2249us p99 while the leader
    moves, so the shipped 1000us budget fired within ~20 ms of any motion. The
    soft tier must count and warn WITHOUT dumping the operator into recovery."""
    with _Bridge(**_eef_params(tick_budget_us=1000.0, tick_overrun_limit=5)) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(5)
        b.node._eef = _SlowEef(2200.0)         # over soft, under hard (3200us)
        b.node._eef_state = "ENGAGED"
        b.ticks(40)
        assert b.node._eef_state == "ENGAGED", "nuisance disengage came back"
        assert b.node._paused is False
        assert b.node._tick_soft_overruns >= 20
        assert b.node._tick_overruns == 0


def test_hard_budget_overrun_still_fails_closed():
    """SUSTAINED slowness (bucket reaches tick_overrun_limit) still fails closed.
    With limit=5 and an unbroken run of over-hard ticks the bucket climbs +1/tick
    and tears down on the 5th, exactly as a genuinely stuck loop must."""
    with _Bridge(**_eef_params(tick_budget_us=1000.0, tick_overrun_limit=5)) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(5)
        b.node._eef = _SlowEef(5000.0)         # over the 3200us hard budget
        b.node._eef_state = "ENGAGED"
        b.ticks(10)
        assert b.node._eef_state == "DISENGAGED"
        assert b.node._paused is True
        assert "tick_budget SUSTAINED" in (b.node._eef_auto_reason or "")


def test_transient_hard_overrun_degrades_without_teardown():
    """A brief compute spike (fewer over-hard ticks than the bucket limit) must
    NOT tear down the session: the arm HOLDs each slow tick (anchor retained,
    still ENGAGED, not paused) and the leaky bucket only counts up. This is the
    real-HW nuisance-disengage that the redesign fixes."""
    with _Bridge(**_eef_params(tick_budget_us=1000.0, tick_overrun_limit=250)) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(5)
        b.node._eef = _SlowEef(5000.0)         # over the 3200us hard budget
        b.node._eef_state = "ENGAGED"
        held = list(b.node._last_published)
        b.ticks(10)                            # 10 over-hard ticks << 250 limit
        assert b.node._eef_state == "ENGAGED", "transient spike tore down session"
        assert b.node._paused is False
        assert b.node._tick_overruns == pytest.approx(10.0)
        # DEGRADE means HOLD: the arm did not move across the slow burst.
        assert list(b.node._last_published) == pytest.approx(held)


def test_leaky_bucket_drains_on_healthy_ticks():
    """Isolated spikes are forgiven: after a slow burst, healthy ticks drain the
    bucket by tick_overrun_leak each, so it never accumulates toward teardown."""
    with _Bridge(
        **_eef_params(tick_budget_us=1000.0, tick_overrun_limit=250,
                      tick_overrun_leak=1.0)
    ) as b:
        b.arm(ARM_POSE)
        b.leader(LEADER_POSE)
        b.ticks(5)
        b.node._eef = _SlowEef(5000.0)         # over hard
        b.node._eef_state = "ENGAGED"
        b.ticks(10)
        assert b.node._tick_overruns == pytest.approx(10.0)
        b.node._eef = _SlowEef(100.0)          # healthy, under soft budget
        b.ticks(10)
        assert b.node._tick_overruns == pytest.approx(0.0)
        assert b.node._eef_state == "ENGAGED"
        assert b.node._paused is False


def test_hard_budget_is_derived_from_the_publish_period():
    with _Bridge(**_eef_params(publish_rate_hz=250.0)) as b:
        assert b.node._tick_hard_us == pytest.approx(3200.0)
    with _Bridge(**_eef_params(publish_rate_hz=125.0)) as b:
        assert b.node._tick_hard_us == pytest.approx(6400.0)
    with _Bridge(**_eef_params(tick_hard_budget_us=1500.0)) as b:
        assert b.node._tick_hard_us == pytest.approx(1500.0)


def test_tick_watchdog_is_retunable_at_runtime():
    """tick_budget_us is read once at construction and is not a launch argument,
    so during bring-up the only way to test a value was to stop the launch and
    re-run the bridge by hand. These three knobs now apply live."""
    from rclpy.parameter import Parameter
    with _Bridge(**_eef_params()) as b:
        assert b.node._tick_hard_us == pytest.approx(3200.0)
        res = b.node.set_parameters([
            Parameter("tick_hard_budget_us", Parameter.Type.DOUBLE, 2600.0),
            Parameter("tick_budget_us", Parameter.Type.DOUBLE, 1800.0),
            Parameter("tick_overrun_limit", Parameter.Type.INTEGER, 9),
            Parameter("tick_overrun_leak", Parameter.Type.DOUBLE, 0.5),
        ])
        assert all(r.successful for r in res)
        assert b.node._tick_hard_us == pytest.approx(2600.0)
        assert b.node.tick_budget_us == pytest.approx(1800.0)
        assert b.node.tick_overrun_limit == 9
        assert b.node.tick_overrun_leak == pytest.approx(0.5)
        # An unrelated parameter is accepted but not applied live.
        res = b.node.set_parameters(
            [Parameter("state_chase_done_tol", Parameter.Type.DOUBLE, 0.42)]
        )
        assert all(r.successful for r in res)
