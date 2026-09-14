"""Deploy-time configuration guards from the R2 review -- ROS-free, no robot.

    PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider \
      test/test_deploy_config_guards.py

These are text/yaml assertions on purpose: every finding here is a property of a
launcher script or a parameter file, and each one of them is the kind of thing that
silently rots back to its old value in a copy-paste.

  F2A  ``run_ur7e_act_real.sh`` must refuse to start when its ZMQ port is already
       held. The v2 handshake cannot catch a stale server: it verifies the action
       space and the dims, never WHICH CHECKPOINT is loaded, and a stale EEF server
       answers all three perfectly.
  F6   carrot ACT: ``staleness_timeout_s`` must leave margin over ``act_timeout_s``.
  F7   carrot FM: the inert ``task:`` node parameter must stay gone.
"""

import os
import re

import pytest

yaml = pytest.importorskip("yaml")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CONFIG = os.path.join(PKG, "config")
WS = os.path.dirname(os.path.dirname(PKG))  # .../ros2_ur_ws


def _load(name):
    with open(os.path.join(CONFIG, name)) as fh:
        return yaml.safe_load(fh)


def _script(name):
    path = os.path.join(WS, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    with open(path) as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# F2A: stale-port refusal in every launcher that spawns a policy server
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script,port_var,spawn_marker", [
    ("run_ur7e_act_real.sh", "ACT_PORT", "Start the ACT inference server"),
    ("run_ur7e_fm_real.sh", "FM_PORT", "Start the FM inference server"),
    ("run_ur7e_diffusion_real.sh", "DIFFUSION_PORT",
     "Start the Diffusion inference server"),
])
def test_launchers_refuse_a_port_that_is_already_listening(script, port_var,
                                                           spawn_marker):
    text = _script(script)
    guard = re.search(
        r"if command -v ss .*?ss -ltn .*?:\$\{%s\}.*?exit 1\nfi" % port_var,
        text, re.S)
    assert guard, f"{script} has no stale-port guard for {port_var}"

    # It has to run BEFORE the server is spawned, or it guards nothing: the whole
    # point is that the NEW server dies on bind while the readiness loop below is
    # satisfied by the STALE one's LISTEN.
    spawn = text.index(spawn_marker)
    assert guard.start() < spawn, f"{script}'s port guard runs after the spawn"

    # ...and it must be fatal, not a warning.
    assert "exit 1" in guard.group(0)
    assert "stale" in guard.group(0)


def test_the_act_guard_matches_the_fm_one_it_was_copied_from():
    """Same idiom, same failure mode, same operator instructions -- only names differ."""
    act = _script("run_ur7e_act_real.sh")
    fm = _script("run_ur7e_fm_real.sh")
    for text, port, proc in ((act, "ACT_PORT", "act_server.py"),
                             (fm, "FM_PORT", "fm_server.py")):
        assert (f'ss -ltn 2>/dev/null | grep -qE ":${{{port}}}([^0-9]|$)"') in text
        assert f"pkill -f {proc} ; ss -ltnp | grep :${{{port}}}" in text
        assert "Starting now would leave the arm driven by that stale server" in text


# --------------------------------------------------------------------------- #
# F6: the ACT watchdog ordering needs margin, not equality
# --------------------------------------------------------------------------- #
def test_carrot_act_staleness_leaves_margin_over_the_zmq_timeout():
    """act_timeout_s == staleness_timeout_s is zero margin.

    The node is single-threaded, so a ZMQ round trip BLOCKS /gello/joint_states for
    its whole duration. At 0.5 == 0.5 a reply at ~0.45 s does not FAULT the node but
    has already starved the bridge past its watchdog -> re-seed + 0.7 s soft-start
    ramp, i.e. a visible stutter with nothing logged, while the EEF target keeps
    integrating.
    """
    cfg = _load("carrot_eef_act.yaml")
    act_timeout = float(cfg["policy_leader_node"]["ros__parameters"]["act_timeout_s"])
    staleness = float(cfg["gello_ur_bridge"]["ros__parameters"]["staleness_timeout_s"])
    assert staleness == 0.8
    assert staleness > act_timeout, "the LEADER must fault before the bridge halts"
    assert staleness - act_timeout >= 0.2, "mirror the FM file's margin"


def test_both_carrot_yamls_use_the_same_bridge_margin():
    act = _load("carrot_eef_act.yaml")
    fm = _load("carrot_eef_fm.yaml")
    assert (act["gello_ur_bridge"]["ros__parameters"]["staleness_timeout_s"]
            == fm["gello_ur_bridge"]["ros__parameters"]["staleness_timeout_s"] == 0.8)


def test_the_bridge_slew_ceiling_was_not_touched():
    """The one bridge value that must never move while adjusting the watchdogs."""
    for name in ("carrot_eef_act.yaml", "carrot_eef_fm.yaml", "act_deploy.yaml"):
        bridge = _load(name)["gello_ur_bridge"]["ros__parameters"]
        assert bridge["max_step_rad"] == 0.0025, name
        assert bridge["soft_start_s"] == 0.7, name


def test_the_joint_act_yaml_is_untouched():
    """act_deploy.yaml drives the banana JOINT deploy: F6 must not have reached it."""
    cfg = _load("act_deploy.yaml")
    assert cfg["gello_ur_bridge"]["ros__parameters"]["staleness_timeout_s"] == 0.5


# --------------------------------------------------------------------------- #
# F7: the inert `task` parameter
# --------------------------------------------------------------------------- #
def test_carrot_fm_yaml_declares_no_task_parameter():
    """`policy_leader_node` never declares `task`, so an override there is read by
    nobody -- while LOOKING like the CLIP prompt knob for a text-conditioned model."""
    params = _load("carrot_eef_fm.yaml")["policy_leader_node"]["ros__parameters"]
    assert "task" not in params


def test_carrot_fm_yaml_still_documents_where_the_prompt_comes_from():
    """Deleting the line silently would just move the trap; the comment must stay."""
    with open(os.path.join(CONFIG, "carrot_eef_fm.yaml")) as fh:
        text = fh.read()
    assert "FM_TASK" in text
    assert "default_task_for_space" in text
    assert '#     task: "Put carrot in pot"' in text


def test_the_fm_server_default_prompt_is_the_dataset_task_string():
    """The value the yaml comment promises is actually what the server defaults to."""
    fm_server = os.path.join(PKG, "policy_server", "fm_server.py")
    with open(fm_server) as fh:
        text = fh.read()
    assert "Put carrot in pot" in text
    assert "def default_task_for_space(" in text


def test_no_other_deploy_yaml_grew_an_undeclared_task_parameter():
    node_src = os.path.join(PKG, "gello_policy", "policy_leader_node.py")
    with open(node_src) as fh:
        declared = set(re.findall(r'declare_parameter\(\s*"([^"]+)"', fh.read()))
    assert "task" not in declared, "if the node ever declares it, revisit this file"
    for name in ("carrot_eef_act.yaml", "carrot_eef_fm.yaml", "act_deploy.yaml",
                 "fm_deploy.yaml", "diffusion_deploy.yaml"):
        params = _load(name).get("policy_leader_node", {}).get("ros__parameters", {})
        undeclared = sorted(set(params) - declared)
        assert "task" not in undeclared, f"{name} carries an inert `task` override"
