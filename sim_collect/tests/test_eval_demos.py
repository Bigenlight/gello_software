"""Kinematic validation of the success predicate + dwell latch on recorded demos, and one
dynamic ReplayPolicy episode through EvalWorld (DESIGN.md §2.4)."""
import os

import pytest

from sim_collect.eval.validate_success_on_demos import kinematic_validate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAKES = os.path.join(ROOT, "ros2_ur_ws", "gello_logs", "sim")


def _takes(n):
    if not os.path.isdir(TAKES):
        pytest.skip("no recorded sim takes")
    takes = sorted(os.path.join(TAKES, d) for d in os.listdir(TAKES)
                   if os.path.isfile(os.path.join(TAKES, d, "vectors.h5")))
    if len(takes) < n:
        pytest.skip(f"need {n} takes, found {len(takes)}")
    return takes[:n]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_kinematic_validation_on_three_takes(idx):
    take = _takes(3)[idx]
    r = kinematic_validate(take, dwell_s=1.0)
    assert r["n_rows"] > 500
    assert r["success_at_t0"] is False
    assert r["success_at_end"] is True and r["latched"]
    assert r["first_ok_t"] is not None and r["first_ok_t"] > 3.0
    assert r["first_success_t"] >= r["first_ok_t"] + 1.0 - 1e-9
    assert r["first_success_t"] < r["duration_s"]
    assert r["recorder_flag"] is True


def test_dwell_shorter_latches_earlier():
    take = _takes(1)[0]
    a = kinematic_validate(take, dwell_s=0.2)
    b = kinematic_validate(take, dwell_s=1.0)
    assert a["first_ok_t"] == b["first_ok_t"] and a["first_success_t"] < b["first_success_t"]


def test_dynamic_replay_of_first_take_succeeds():
    from sim_collect.eval.policies import ReplayPolicy
    from sim_collect.eval.run_eval import run_episode
    from sim_collect.eval.world import EvalWorld
    take = _takes(1)[0]
    w = EvalWorld(os.path.join(ROOT, "sim_collect", "configs", "carrot_in_pot_sim.yaml"))
    try:
        rec = run_episode(w, ReplayPolicy(take), "t", None, 600, None, verbose=False)
    finally:
        w.close()
    assert rec["outcome"] == "success" and 9.0 < rec["t_success_s"] < 20.0
    assert rec["layout"]["_override"] is True
    assert rec["clamp_hits"]["envelope"] == 0            # the sim-derived envelope contains the demos
