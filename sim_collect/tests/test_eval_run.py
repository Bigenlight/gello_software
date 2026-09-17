"""run_eval end-to-end against the stub ZMQ server, zero policy on all default seeds, and the
fault / PolicyRefused accounting (reviewer items 1, 2, 4)."""
import json
import os

import numpy as np
import pytest

from sim_collect.eval.policies import PolicyRefused, ZeroPolicy, ZmqPolicy
from sim_collect.eval.run_eval import main as run_eval_main, run_episode, summarize
from sim_collect.eval.world import EvalWorld
from sim_collect.tests.conftest import needs_display
from sim_collect.tests.stub_policy_server import StubPolicyServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = os.path.join(ROOT, "sim_collect", "configs", "carrot_in_pot_sim.yaml")


@pytest.fixture(scope="module")
def world():
    w = EvalWorld(CFG, video=False)
    yield w
    w.close()


@needs_display
def test_run_eval_main_end_to_end_with_stub(tmp_path):
    with StubPolicyServer(port=0) as s:
        rc = run_eval_main(["--policy", s.endpoint, "--seeds", "0-1", "--max-steps", "15", "--out", str(tmp_path),
                            "--quiet", "--config", CFG])
    assert rc == 0
    eps = [json.loads(l) for l in open(tmp_path / "episodes.jsonl")]
    summ = json.load(open(tmp_path / "summary.json"))
    assert (tmp_path / "summary.md").is_file()
    assert [e["seed"] for e in eps] == [0, 1] and all(e["outcome"] == "timeout" and e["n_steps"] == 15 for e in eps)
    assert all((tmp_path / f"ep_{e['episode']}.h5").is_file() for e in eps)
    assert summ["n_episodes"] == 2 and summ["n_success"] == summ["counts"]["success"] == 0
    assert summ["success_rate"] == summ["n_success"] / summ["n_episodes"]
    assert sum(summ["counts"].values()) == summ["n_episodes"]
    assert summ["policy_meta"]["server"]["state_dim"] == 7 and summ["policy_meta"]["timeout_s"] == 0.6
    assert "NOT carrot_eef_limits.json" in open(tmp_path / "summary.md").read()
    assert s.n_reset == 2 and s.n_act == 30


def test_zero_policy_on_all_default_seeds(world):
    seeds = world.ev["seeds"]
    assert seeds == list(range(100, 120))   # held-out default (seeds 0-19 == training layouts)
    max_steps = 120                                 # 4 s each: the arm never moves, so the outcome cannot change
    recs = [run_episode(world, ZeroPolicy(), str(sd), sd, max_steps, None, verbose=False) for sd in seeds]
    assert all(r["outcome"] == "timeout" and r["n_steps"] == max_steps for r in recs)
    s = summarize(recs, policy_spec="zero", world=world, args={}, wall_s=0.0)
    assert s["n_success"] == 0 and s["n_episodes"] == 20 and s["success_rate"] == 0.0
    assert s["wilson_95"]["lo"] == 0.0 and s["wilson_95"]["hi"] < 0.2
    assert s["clamp_hits"]["envelope"] == 0 and s["clamp_hits"]["max_dev"] == 0


def test_timeout_fault_counts_in_the_denominator(world):
    with StubPolicyServer(port=0, delay_s=0.5) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.2)
        rec = run_episode(world, p, "f", 0, 5, None, verbose=False)
        p.close()
    assert rec["outcome"] == "fault" and "timeout" in rec["fault"] and rec["n_steps"] == 0
    s = summarize([rec], policy_spec="x", world=world, args={}, wall_s=0.0)
    assert s["n_episodes"] == 1 and s["counts"]["fault"] == 1 and s["success_rate"] == 0.0


def test_raising_policy_is_a_fault_and_the_run_continues(world):
    class Boom(ZeroPolicy):
        def act(self, obs):
            raise RuntimeError("kaboom")
    rec = run_episode(world, Boom(), "b", 0, 5, None, verbose=False)
    assert rec["outcome"] == "fault" and "kaboom" in rec["fault"]
    rec2 = run_episode(world, ZeroPolicy(), "ok", 0, 3, None, verbose=False)
    assert rec2["outcome"] == "timeout" and rec2["n_steps"] == 3


def test_policy_refused_aborts_instead_of_faulting(world, tmp_path):
    with StubPolicyServer(port=0, reset_extra={"policy_type": "act", "state_dim": 16, "action_dim": 7}) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.5)
        with pytest.raises(PolicyRefused):
            run_episode(world, p, "r", 0, 5, None, verbose=False)
        p.close()
        rc = run_eval_main(["--policy", s.endpoint, "--seeds", "0-2", "--max-steps", "5", "--out", str(tmp_path),
                            "--quiet", "--config", CFG])
    assert rc == 2
    assert json.load(open(tmp_path / "summary.json"))["n_episodes"] == 0
    assert "aborted" in json.load(open(tmp_path / "summary.json"))["args"]


def test_run_episode_snapshots_policy_metadata_after_reset(world):
    class ResetMetadataPolicy(ZeroPolicy):
        def __init__(self):
            super().__init__()
            self.meta = {"policy": "test", "reset_reply": {"reset_counter": 40}}

        def reset(self, world_info):
            self.meta["reset_reply"] = {"reset_counter": 41, "seed": 123, "log_dir": "/trusted/local"}
            return super().reset(world_info)

    rec = run_episode(world, ResetMetadataPolicy(), "meta", 0, 1, None, verbose=False)
    assert rec["policy"]["reset_reply"]["reset_counter"] == 41
    assert rec["policy_reset"] == {"reset_counter": 41, "seed": 123, "log_dir": "/trusted/local"}
