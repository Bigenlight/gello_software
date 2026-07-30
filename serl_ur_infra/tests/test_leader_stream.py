"""Tests for the leader resampling filter + intervention displacement budget.

Three of these tests are load-bearing; the rest are guard rails.

* ``test_one_euro_is_bit_identical_to_the_bridge_filter`` (and its bundle-level
  twin ``test_leader_filter_matches_a_hand_rolled_bridge_filter_bundle``) — the
  ported :class:`OneEuro` is the *proven* teleop filter, not a re-derivation of
  it.  Asserted exactly, with no tolerance, against
  ``ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/bridge_stages.py``.  If
  someone "cleans up" the arithmetic, this fails.
* ``test_coupling_update_input_to_every_tick_makes_the_cutoff_pulse`` — the
  reason :meth:`LeaderFilter.note_sample` and :meth:`LeaderFilter.filtered` are
  separate methods at all (``bridge_stages.py:56-59``: the filter "would open up
  in visible pulses").  Quantified, not asserted by assertion of faith.
* ``test_diagonal_saturation_is_not_a_per_axis_clip`` — the regression that
  keeps ``InterventionBudget.take`` a proportional (norm) clamp.  A per-axis
  clip bends a saturated diagonal, which the operator feels as the arm going
  somewhere they did not point (``ur_env/envs/wrappers.py:316-324``).

Pure numpy/stdlib: no rclpy, no ROS runtime.  The bridge reference module is
stdlib-only, so it imports from the overlay source tree directly.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_INFRA_ROOT = _HERE.parent
_REPO_ROOT = _INFRA_ROOT.parent
if os.fspath(_INFRA_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_INFRA_ROOT))

# bridge_stages imports only ``math``, so the overlay source tree is importable
# without a ROS environment (same trick as tests/test_clip_safety_box.py:24-28).
_OVERLAY = _REPO_ROOT / "ros2_ur_ws" / "src" / "ur_gello_bringup"
if _OVERLAY.is_dir() and os.fspath(_OVERLAY) not in sys.path:
    sys.path.insert(0, os.fspath(_OVERLAY))

from ur_env.envs.leader_stream import (  # noqa: E402
    NOMINAL_CONTROL_HZ,
    ONE_EURO_DEFAULTS,
    InterventionBudget,
    LeaderFilter,
    OneEuro,
)

_bridge = pytest.importorskip(
    "ur_gello_bringup.bridge_stages",
    reason="needs the ros2_ur_ws/src/ur_gello_bringup overlay on sys.path",
)
ReferenceOneEuro = _bridge.OneEuro

# cube_in_cup's measured ACTION_SCALE (ur_env/envs/config.py:73).  Hard-coded on
# purpose: these tests must keep testing the real magnitudes even if a future
# retune changes the config, and a silent drift should show up as a failure here.
ACTION_SCALE = np.array([0.0125, 0.0625, 1.0])
SUBSTEP_HZ = 30.0
OUTPUT_HZ = 250.0
LEADER_HZ = 30.0


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _leader_trajectory(kind: str, n: int, dt: float, seed: int = 0):
    """A leader joint signal sampled at ``dt``: list of ``(t, value)``."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = k * dt
        if kind == "still":
            x = 0.31
        elif kind == "ramp":                      # 0.15 rad/s, the measured case
            x = 0.15 * t
        elif kind == "tremor":                    # at-rest hand/Dynamixel noise
            x = 0.31 + rng.normal(0.0, 0.002)
        elif kind == "reversal":                  # out and back, sign change
            x = 0.4 * math.sin(2.0 * math.pi * 0.8 * t)
        elif kind == "burst":                     # fast move, then hold
            x = 0.0 if t < 0.2 else (0.9 * (t - 0.2) if t < 0.6 else 0.36)
        elif kind == "noisy_ramp":
            x = 0.15 * t + rng.normal(0.0, 0.0008)
        else:  # pragma: no cover - guard against typos in the parametrisation
            raise AssertionError(f"unknown trajectory {kind!r}")
        out.append((t, float(x)))
    return out


def _run_two_cadence(filt, samples, output_hz: float, leader_hz: float):
    """Drive ``filt`` the way the real loop does: samples < ticks.

    ``update_input`` fires only when a new leader sample lands; ``__call__``
    fires on every output tick with the most recent (held) sample.
    """
    out_dt = 1.0 / output_hz
    samp_dt = 1.0 / leader_hz
    ticks = int(len(samples) * samp_dt / out_dt)
    held = samples[0][1]
    idx = 0
    last_t = None
    ys = []
    for k in range(ticks):
        t = k * out_dt
        while idx < len(samples) and samples[idx][0] <= t + 1e-12:
            s_t, held = samples[idx]
            filt.update_input(held, None if last_t is None else s_t - last_t)
            last_t = s_t
            idx += 1
        ys.append(filt(held))
    return np.array(ys)


# --------------------------------------------------------------------------- #
# (1) the port is the proven filter, bit for bit                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind", ["still", "ramp", "tremor", "reversal", "burst", "noisy_ramp"]
)
def test_one_euro_is_bit_identical_to_the_bridge_filter(kind):
    """Exact equality against ``bridge_stages.OneEuro`` — no tolerance.

    The hand feel this module copies was measured with THAT implementation, so
    the port is only allowed to be a copy.  Both cadences are exercised
    (30 Hz samples, 250 Hz ticks) because the two entry points hold separate
    state.
    """
    samples = _leader_trajectory(kind, 90, 1.0 / LEADER_HZ, seed=7)
    mine = OneEuro(1.0 / OUTPUT_HZ, **ONE_EURO_DEFAULTS)
    ref = ReferenceOneEuro(1.0 / OUTPUT_HZ, **ONE_EURO_DEFAULTS)

    got = _run_two_cadence(mine, samples, OUTPUT_HZ, LEADER_HZ)
    want = _run_two_cadence(ref, samples, OUTPUT_HZ, LEADER_HZ)

    assert got.shape == want.shape and got.size > 100
    assert list(got) == list(want), "ported OneEuro drifted from the bridge filter"


def test_one_euro_matches_the_bridge_on_the_degenerate_dt_branches():
    """First sample (``dt=None``) and a stalled clock (``dt<=1e-6``).

    These branches exist so a clock hiccup can never divide by ~0 and blow the
    speed estimate up; they are the ones a "tidier" port silently drops.
    """
    mine = OneEuro(1.0 / OUTPUT_HZ, **ONE_EURO_DEFAULTS)
    ref = ReferenceOneEuro(1.0 / OUTPUT_HZ, **ONE_EURO_DEFAULTS)
    for x, dt in [(0.1, None), (0.2, 0.0), (0.3, 1e-9), (0.4, 1e-6),
                  (0.5, 1.0 / LEADER_HZ), (0.55, 1.0 / LEADER_HZ)]:
        mine.update_input(x, dt)
        ref.update_input(x, dt)
        assert mine(x) == ref(x)

    # seed() must also agree: same preload, same first output.
    mine.seed(1.234)
    ref.seed(1.234)
    assert mine(1.3) == ref(1.3)


def test_ported_defaults_match_the_operational_yaml():
    """``ONE_EURO_DEFAULTS`` are the shipped teleop gains, not new tuning.

    Parsed straight out of ``config/ur7e_gello.yaml`` (line-oriented regex, no
    yaml dependency).  The node's ``declare_parameter`` defaults
    (``gello_ur_bridge_node.py:401-408``) already agree with that file, so
    pinning against the yaml pins both.
    """
    yaml_path = _OVERLAY / "config" / "ur7e_gello.yaml"
    if not yaml_path.is_file():  # pragma: no cover - overlay absent
        pytest.skip(f"{yaml_path} not present")
    text = yaml_path.read_text()

    for key, expected in ONE_EURO_DEFAULTS.items():
        m = re.search(rf"^\s*one_euro_{key}\s*:\s*([0-9.eE+-]+)\s*$",
                      text, re.MULTILINE)
        assert m, f"one_euro_{key} not found in {yaml_path}"
        assert float(m.group(1)) == expected

    # The deadband is deliberately NOT ported: filter_type selects one_euro, so
    # filter_stage_joint's deadband arm is dead code on the proven path
    # (bridge_stages.py:134-143).  If the operational config ever switches to
    # "ema" this assertion fires and the "One-Euro alone" claim in the module
    # docstring has to be revisited.
    assert re.search(r'^\s*filter_type\s*:\s*"one_euro"\s*$', text, re.MULTILINE)


# --------------------------------------------------------------------------- #
# (2) why the two cadences must not be conflated                               #
# --------------------------------------------------------------------------- #
def test_coupling_update_input_to_every_tick_makes_the_cutoff_pulse():
    """Calling ``update_input`` per output tick pulses the adaptive cutoff.

    A constant-speed leader (0.15 rad/s, the measured case) sampled at 30 Hz is
    a zero-order-hold staircase at the 250 Hz output.  Fed at the sample
    cadence, the speed estimate is a constant 0.15 and the cutoff is steady.
    Fed at every tick, the same motion looks like seven zero-velocity ticks plus
    one ~8x jump, so the estimate — and with it the cutoff, i.e. the filter's
    aperture — ripples at 30 Hz.  That ripple is the "visible pulses" of
    ``bridge_stages.py:56-59``.

    The metric is derived from the PUBLIC signals only: the effective per-tick
    smoothing factor ``a = (y[k]-y[k-1]) / (x[k]-y[k-1])`` is observable from
    input and output, and it is exactly the quantity the adaptive cutoff sets.
    Measured in steady state (t > 1.5 s): std 3.1e-07 (sample cadence) vs
    4.2e-04 (per-tick) — a ~1360x wider aperture swing — and output increment
    variance 2.15e-09 vs 2.97e-09.  The gate below is 100x, well under the
    measured ratio and well over any float noise.
    """
    out_dt = 1.0 / OUTPUT_HZ
    samp_dt = 1.0 / LEADER_HZ
    total_s = 3.0
    settle_s = 1.5   # skip the filter's start-up transient
    speed = 0.15

    def run(couple_to_ticks: bool):
        filt = OneEuro(out_dt, **ONE_EURO_DEFAULTS)
        held = 0.0
        next_sample = 0.0
        last_t = None
        xs, ys, ts = [], [], []
        for k in range(int(total_s / out_dt)):
            t = k * out_dt
            if t >= next_sample - 1e-12:
                held = speed * t
                if not couple_to_ticks:
                    filt.update_input(held, None if last_t is None else t - last_t)
                    last_t = t
                next_sample += samp_dt
            if couple_to_ticks:
                # THE MISTAKE: the output period is not the sample period.
                filt.update_input(held, out_dt)
            ts.append(t)
            xs.append(held)
            ys.append(filt(held))
        ts, xs, ys = np.array(ts), np.array(xs), np.array(ys)
        gap = xs[1:] - ys[:-1]
        keep = (np.abs(gap) > 1e-9) & (ts[1:] > settle_s)
        alpha = (ys[1:] - ys[:-1])[keep] / gap[keep]
        incr = np.diff(ys)[ts[1:] > settle_s]
        return alpha, incr

    alpha_ok, incr_ok = run(False)
    alpha_bad, incr_bad = run(True)

    assert alpha_ok.size > 200 and alpha_bad.size > 200
    # Same mean aperture (this is a ripple, not a bias) ...
    assert alpha_bad.mean() == pytest.approx(alpha_ok.mean(), rel=1e-3)
    # ... but a vastly noisier one when the cadences are conflated.
    assert alpha_bad.std() > 100.0 * alpha_ok.std()
    # And the pulsing does reach the output: strictly larger increment variance.
    assert np.var(incr_bad) > 1.2 * np.var(incr_ok)


def test_leader_filter_matches_a_hand_rolled_bridge_filter_bundle():
    """The 6-joint bundle is bit-identical to per-joint bridge filters.

    Same assertion as the scalar bit-identity test, one level up: this is what
    proves ``LeaderFilter`` only sequences the proven filter (and does not, say,
    accidentally share state between joints or reorder the two cadences).
    """
    n_joints = 6
    lf = LeaderFilter(output_hz=OUTPUT_HZ, n_joints=n_joints)
    ref = [ReferenceOneEuro(1.0 / OUTPUT_HZ, **ONE_EURO_DEFAULTS)
           for _ in range(n_joints)]

    rng = np.random.default_rng(11)
    q = np.array([0.1, -0.4, 0.9, -1.2, 0.3, 0.0])
    out_dt = 1.0 / OUTPUT_HZ
    samp_dt = 1.0 / LEADER_HZ
    next_sample = 0.0
    last_t = None
    held = q.copy()
    compared = 0
    for k in range(int(2.0 / out_dt)):
        t = k * out_dt
        if t >= next_sample - 1e-12:
            held = q + 0.15 * t + rng.normal(0.0, 0.001, size=n_joints)
            dt = None if last_t is None else t - last_t
            lf.note_sample(held, dt)
            for i in range(n_joints):
                ref[i].update_input(float(held[i]), dt)
            last_t = t
            next_sample += samp_dt
        got = lf.filtered(held)
        want = [ref[i](float(held[i])) for i in range(n_joints)]
        assert list(got) == want
        compared += 1
    assert compared > 400


def test_note_sample_does_not_move_the_output_and_filtered_does():
    """The API split is real: each method touches only its own state."""
    lf = LeaderFilter(output_hz=SUBSTEP_HZ)
    q0 = np.zeros(6)
    lf.seed(q0)                       # align state, no motion
    q1 = np.full(6, 0.05)

    # Many samples, no ticks -> the speed estimate moves, the output does not.
    for k in range(20):
        lf.note_sample(np.full(6, 0.05 * (k + 1)), 1.0 / LEADER_HZ)
    first = lf.filtered(q1)

    # A second tick with the SAME input must advance the output further: the
    # output state is what filtered() owns.
    second = lf.filtered(q1)
    assert np.all(first > 0.0) and np.all(second > first)
    assert np.all(second < 0.05 + 1e-12)   # still a low-pass, never an overshoot


# --------------------------------------------------------------------------- #
# LeaderFilter state + validation                                              #
# --------------------------------------------------------------------------- #
def test_reset_reanchors_on_the_live_leader():
    """After ``reset()`` the next tick returns the input verbatim.

    That is the engage/re-anchor requirement: the leader may be anywhere while
    the policy was driving, and a filter that low-passed across that gap would
    ramp the arm toward a pose nobody commanded.
    """
    lf = LeaderFilter(output_hz=SUBSTEP_HZ)
    for k in range(10):
        q = np.full(6, 0.01 * k)
        lf.note_sample(q, 1.0 / LEADER_HZ)
        lf.filtered(q)

    far = np.full(6, 2.5)
    assert not np.allclose(lf.filtered(far), far)   # pre-reset: heavy lag

    lf.reset()
    assert np.array_equal(lf.filtered(far), far)    # post-reset: verbatim

    # ... and the post-reset sequence equals a brand-new filter's sequence.
    fresh = LeaderFilter(output_hz=SUBSTEP_HZ)
    lf.reset()
    seq_a, seq_b = [], []
    for k in range(30):
        q = np.full(6, 2.5 + 0.02 * k)
        lf.note_sample(q, 1.0 / LEADER_HZ)
        fresh.note_sample(q, 1.0 / LEADER_HZ)
        seq_a.append(lf.filtered(q))
        seq_b.append(fresh.filtered(q))
    assert np.array_equal(np.array(seq_a), np.array(seq_b))


def test_leader_filter_rejects_bad_construction_and_bad_samples():
    with pytest.raises(ValueError):
        LeaderFilter(output_hz=0.0)
    with pytest.raises(ValueError):
        LeaderFilter(output_hz=float("inf"))
    with pytest.raises(ValueError):
        LeaderFilter(output_hz=SUBSTEP_HZ, n_joints=0)
    with pytest.raises(TypeError):
        LeaderFilter(output_hz=SUBSTEP_HZ, min_cutof=1.0)   # typo must not pass
    with pytest.raises(ValueError):
        LeaderFilter(output_hz=SUBSTEP_HZ, beta=-1.0)

    lf = LeaderFilter(output_hz=SUBSTEP_HZ)
    with pytest.raises(ValueError):
        lf.filtered(np.zeros(5))
    with pytest.raises(ValueError):
        lf.note_sample(np.zeros(7), 0.03)
    # A NaN in a low-pass is permanent (the state feeds back into itself), so it
    # must be refused at the boundary while the caller can still HOLD.
    with pytest.raises(ValueError):
        lf.note_sample(np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0]), 0.03)
    with pytest.raises(ValueError):
        lf.filtered(np.array([0.0, 0.0, np.inf, 0.0, 0.0, 0.0]))
    # ... and the filter is still usable afterwards (no poisoned state).
    assert np.array_equal(lf.filtered(np.zeros(6)), np.zeros(6))


def test_defaults_dict_is_not_mutated_through_an_instance():
    lf = LeaderFilter(output_hz=SUBSTEP_HZ, beta=5.0)
    assert lf.euro_params["beta"] == 5.0
    assert ONE_EURO_DEFAULTS["beta"] == 2.0
    lf.euro_params["beta"] = 99.0
    assert ONE_EURO_DEFAULTS["beta"] == 2.0


# --------------------------------------------------------------------------- #
# (3) InterventionBudget                                                       #
# --------------------------------------------------------------------------- #
def _budget():
    return InterventionBudget(ACTION_SCALE, SUBSTEP_HZ)


def test_requests_inside_the_budget_pass_through_untouched():
    """No per-substep throttling: credit remains, so the request is honoured.

    The budget is a ceiling on the WINDOW (config.py:55-73), so an operator
    moving fast inside one window must not be slowed down — that would make the
    stored action overstate the motion, the mirror image of the bug this class
    prevents.
    """
    b = _budget()
    b.begin_window()
    xi = np.array([0.003, -0.002, 0.001, 0.01, 0.0, -0.005])
    allowed, spent = b.take(xi)
    assert np.array_equal(allowed, xi)
    assert spent is False and b.exhausted is False
    # A second small request also passes: 0.003+... is still under 0.0125 m.
    allowed2, spent2 = b.take(xi)
    assert np.array_equal(allowed2, xi)
    assert spent2 is False


def test_over_budget_request_is_scaled_by_norm_preserving_direction():
    b = _budget()
    b.begin_window()
    # 10x the position budget, in a deliberately non-axis-aligned direction.
    pos = np.array([3.0, -4.0, 12.0])          # norm 13
    pos = pos / 13.0 * (10.0 * ACTION_SCALE[0])
    rot = np.array([0.0, 1.0, 1.0])
    rot = rot / math.sqrt(2.0) * (4.0 * ACTION_SCALE[1])
    allowed, spent = b.take(np.concatenate([pos, rot]))

    ap, ar = allowed[:3], allowed[3:]
    assert np.linalg.norm(ap) == pytest.approx(ACTION_SCALE[0], rel=1e-12)
    assert np.linalg.norm(ar) == pytest.approx(ACTION_SCALE[1], rel=1e-12)
    # DIRECTION PRESERVED: unit vectors unchanged (cos == 1).
    for got, want in ((ap, pos), (ar, rot)):
        cos = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want)))
        assert cos == pytest.approx(1.0, abs=1e-12)
    assert spent is True


def test_exhaustion_latches_and_later_takes_return_zero():
    """Once the credit is gone the honest answer is HOLD, i.e. exactly zero."""
    b = _budget()
    b.begin_window()
    step = np.array([ACTION_SCALE[0] / 2.0, 0.0, 0.0,
                     ACTION_SCALE[1] / 2.0, 0.0, 0.0])
    a1, s1 = b.take(step)
    assert np.array_equal(a1, step) and s1 is False
    a2, s2 = b.take(step)                       # lands exactly on the budget
    assert np.array_equal(a2, step) and s2 is True
    for _ in range(5):
        a, s = b.take(step)
        assert np.array_equal(a, np.zeros(6))
        assert s is True
    assert b.exhausted is True
    assert np.allclose(b.remaining(), [0.0, 0.0])


def test_consumed_action_is_in_the_unit_box_and_hits_one_when_spent():
    """The stored action: legal by construction, and truthful at saturation."""
    b = _budget()
    b.begin_window()
    # Collinear substeps summing to exactly one full budget in x and rx.
    n = 7
    step = np.array([ACTION_SCALE[0] / n, 0.0, 0.0,
                     ACTION_SCALE[1] / n, 0.0, 0.0])
    for _ in range(n):
        b.take(step)
        act = b.consumed_action()
        assert np.all(np.abs(act) <= 1.0 + 1e-12)

    act = b.consumed_action()
    assert np.linalg.norm(act[:3]) == pytest.approx(1.0, rel=1e-9)
    assert np.linalg.norm(act[3:]) == pytest.approx(1.0, rel=1e-9)
    assert act[0] == pytest.approx(1.0, rel=1e-9)
    assert act[3] == pytest.approx(1.0, rel=1e-9)


def test_diagonal_saturation_is_not_a_per_axis_clip():
    """REGRESSION (wrappers.py:316-324): ``[2.0, 0.5]`` must stay 4:1.

    A per-axis clip of this saturated diagonal returns ``[1.0, 0.5]`` — ratio
    2:1 — bending the path the operator drew.  The norm clamp keeps 4:1 and
    caps the magnitude instead.
    """
    b = _budget()
    b.begin_window()
    xi = np.zeros(6)
    xi[0] = 2.0 * ACTION_SCALE[0]
    xi[1] = 0.5 * ACTION_SCALE[0]
    allowed, _ = b.take(xi)

    assert allowed[0] / allowed[1] == pytest.approx(4.0, rel=1e-12)
    assert np.linalg.norm(allowed[:3]) == pytest.approx(ACTION_SCALE[0], rel=1e-12)
    act = b.consumed_action()
    # What a per-axis clip would have reported, and must NOT:
    assert not np.allclose(act[:2], [1.0, 0.5])
    assert act[0] / act[1] == pytest.approx(4.0, rel=1e-12)
    assert np.linalg.norm(act[:3]) == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.abs(act) <= 1.0 + 1e-12)


def test_position_exhaustion_latches_hold_while_rotation_credit_remains():
    """Either channel spent => HOLD, because one hand drives both.

    The rotation budget is genuinely untouched (independent accounting), but the
    flag latches anyway: letting rotation continue while translation is frozen
    would distort the coupled motion the operator commanded — the same
    direction-distortion argument that rules out per-axis clipping.
    """
    b = _budget()
    b.begin_window()
    allowed, spent = b.take(np.array([2.0 * ACTION_SCALE[0], 0, 0, 0, 0, 0]))
    assert spent is True
    rem = b.remaining()
    assert rem[0] == pytest.approx(0.0, abs=1e-15)
    assert rem[1] == pytest.approx(ACTION_SCALE[1], rel=1e-12)
    assert np.allclose(b.consumed_action()[3:], 0.0)


def test_out_and_back_is_charged_for_both_legs():
    """Path-length accounting is the conservative, monotone choice.

    Budgeting the NET displacement would let the flag flicker back to False on a
    reversal and would allow unbounded travel inside one window.  Charging the
    path keeps ``exhausted`` latched and keeps the reported action bounded; the
    price, asserted here, is that an out-and-back pays twice.
    """
    b = _budget()
    b.begin_window()
    half = np.array([ACTION_SCALE[0] / 2.0, 0, 0, 0, 0, 0])
    b.take(half)
    b.take(-half)                                # returns to the anchor
    assert b.exhausted is True                   # path spent, though net == 0
    assert np.allclose(b.consumed_action(), 0.0)  # ... and reported as net 0
    a, s = b.take(half)
    assert np.array_equal(a, np.zeros(6)) and s is True


def test_begin_window_restores_the_full_budget():
    b = _budget()
    b.take(np.array([9.0 * ACTION_SCALE[0], 0, 0, 9.0 * ACTION_SCALE[1], 0, 0]))
    assert b.exhausted is True

    b.begin_window()
    assert b.exhausted is False
    assert np.allclose(b.remaining(), [ACTION_SCALE[0], ACTION_SCALE[1]])
    assert np.allclose(b.consumed_action(), np.zeros(6))
    small = np.array([0.001, 0.0, 0.0, 0.004, 0.0, 0.0])
    allowed, spent = b.take(small)
    assert np.array_equal(allowed, small) and spent is False


def test_fresh_budget_starts_open_without_an_explicit_begin_window():
    """``__init__`` opens the first window, so a caller cannot leak state."""
    b = _budget()
    assert b.exhausted is False
    assert np.allclose(b.remaining(), [ACTION_SCALE[0], ACTION_SCALE[1]])
    assert np.allclose(b.consumed_action(), np.zeros(6))


def test_zero_request_neither_spends_nor_exhausts():
    b = _budget()
    allowed, spent = b.take(np.zeros(6))
    assert np.array_equal(allowed, np.zeros(6))
    assert spent is False
    assert np.allclose(b.remaining(), [ACTION_SCALE[0], ACTION_SCALE[1]])


def test_budget_validates_its_arguments():
    with pytest.raises(ValueError):
        InterventionBudget(np.array([0.0125, 0.0625]), SUBSTEP_HZ)      # (2,)
    with pytest.raises(ValueError):
        InterventionBudget(np.array([0.0, 0.0625, 1.0]), SUBSTEP_HZ)    # pos 0
    with pytest.raises(ValueError):
        InterventionBudget(np.array([0.0125, -1.0, 1.0]), SUBSTEP_HZ)   # rot < 0
    with pytest.raises(ValueError):
        InterventionBudget(ACTION_SCALE, 0.0)
    b = _budget()
    with pytest.raises(ValueError):
        b.take(np.zeros(7))
    with pytest.raises(ValueError):
        b.take(np.array([np.nan, 0, 0, 0, 0, 0]))


def test_action_scale_is_copied_not_aliased():
    scale = ACTION_SCALE.copy()
    b = InterventionBudget(scale, SUBSTEP_HZ)
    scale[0] = 99.0
    assert b.pos_budget == ACTION_SCALE[0]


def test_nominal_substep_share_is_advisory_and_consistent():
    b = _budget()
    assert b.substep_hz == SUBSTEP_HZ
    assert b.substep_dt == pytest.approx(1.0 / SUBSTEP_HZ)
    assert b.nominal_substeps == round(SUBSTEP_HZ / NOMINAL_CONTROL_HZ) == 3
    share = b.nominal_substep_share
    assert np.allclose(share * b.nominal_substeps,
                       [ACTION_SCALE[0], ACTION_SCALE[1]])
    # Advisory only: a single substep larger than the share is still honoured.
    big = np.array([2.0 * share[0], 0, 0, 0, 0, 0])
    allowed, spent = b.take(big)
    assert np.array_equal(allowed, big) and spent is False


def test_random_substeps_never_break_the_stored_action_invariant():
    """Fuzz: whatever the operator does, the report stays legal.

    Invariants checked on every substep of every window:
      * the allowed increment is never longer than the request (and never
        longer than the remaining credit),
      * direction is preserved whenever the request is non-zero,
      * path spent never exceeds the budget,
      * ``consumed_action()`` stays inside ``[-1,1]`` with no clipping needed.
    """
    rng = np.random.default_rng(2026)
    b = _budget()
    for _ in range(60):
        b.begin_window()
        for _ in range(rng.integers(1, 12)):
            xi = np.concatenate([
                rng.normal(0.0, 0.01, size=3),    # ~ one budget per substep
                rng.normal(0.0, 0.05, size=3),
            ])
            before = b.remaining()
            allowed, spent = b.take(xi)

            n_req_p = np.linalg.norm(xi[:3])
            n_got_p = np.linalg.norm(allowed[:3])
            assert n_got_p <= n_req_p + 1e-15
            assert n_got_p <= before[0] + 1e-15
            if n_req_p > 0 and n_got_p > 0:
                cos = float(np.dot(allowed[:3], xi[:3]) / (n_got_p * n_req_p))
                assert cos == pytest.approx(1.0, abs=1e-9)

            n_req_r = np.linalg.norm(xi[3:])
            n_got_r = np.linalg.norm(allowed[3:])
            assert n_got_r <= n_req_r + 1e-15
            assert n_got_r <= before[1] + 1e-15
            if n_req_r > 0 and n_got_r > 0:
                cos = float(np.dot(allowed[3:], xi[3:]) / (n_got_r * n_req_r))
                assert cos == pytest.approx(1.0, abs=1e-9)

            act = b.consumed_action()
            assert np.all(np.abs(act) <= 1.0)
            assert np.linalg.norm(act[:3]) <= 1.0 + 1e-12
            assert np.linalg.norm(act[3:]) <= 1.0 + 1e-12
            assert spent == b.exhausted
            assert np.all(b.remaining() >= 0.0)
