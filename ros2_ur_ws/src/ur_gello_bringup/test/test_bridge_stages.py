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
