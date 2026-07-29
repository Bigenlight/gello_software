"""Which deadman source actually ends up in the actor's wrapper chain.

WHY THIS FILE EXISTS
--------------------
The Stage B PASS criterion is "press ENGAGE in the HIL GUI -> move the GELLO ->
the server's intervention counter goes up".  There is a path where that fails
*silently*:

    ur_experiments/cube_in_cup.py:257   env = GelloIntervention(env, deadman=deadman)
    ...with deadman defaulting to None, and
    scripts/run_remote_rlpd_actor.py:93 never passing it.

    ur_env/envs/wrappers.py:189
        self.deadman = deadman if deadman is not None else SpacebarDeadman()

So ``deadman=None`` is NOT "no deadman" and NOT the GUI topic -- it constructs a
``SpacebarDeadman``, a global pynput key listener that ignores ``/hil/deadman``
entirely.  The GUI publishes happily, the actor subscribes to nothing, no error
is raised anywhere, and the operator is left hunting for a counter that can
never move.  (Tracked as G12 in docs/testing/08_OPEN_GAPS.md, which covers the
``run_rviz_hil.py`` path; the actor entry point is worse -- it has no
``--deadman`` flag at all, so the topic deadman is *unreachable* from it.)

These tests pin down three separate things:
  1. the actual resolution of ``deadman=None`` (test_* below assert the real
     class, so a future fix must update them deliberately);
  2. the ``RosTopicDeadman`` <-> GUI frozen contract (parse, staleness
     fail-stop, gain clamp) so the GUI side can be trusted once it IS wired;
  3. the intervention bookkeeping the server counts on
     (``intervened`` / ``intervene_action``), including the stuck-ON case.

Everything here is ROS-free and robot-free: no rclpy node, no serial port, no
/hil/deadman publisher, no camera.  ``std_msgs`` and ``pynput`` are stubbed so
the file behaves identically on the robot laptop and on a bare dev box -- and,
importantly, so running the suite never installs a *global keyboard listener*
on an operator's desktop.
"""

import importlib.util
import os
import re
import sys
import time
import types
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.join(_HERE, "..")
_REPO = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _INFRA)
# ur_gello_bringup.ur_kin (fk / so3_log) is pure numpy -> importable without ROS
sys.path.insert(0, os.path.join(_REPO, "ros2_ur_ws", "src", "ur_gello_bringup"))

from ur_env.envs.wrappers import (  # noqa: E402
    DeadmanHeartbeatStaleError,
    DeadmanSource,
    GelloExpert,
    GelloIntervention,
    RosTopicDeadman,
    SpacebarDeadman,
)

Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])


# --------------------------------------------------------------------------- #
# safety fixtures: never touch the operator's keyboard, never need ROS         #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _block_pynput(monkeypatch):
    """Stop any code path under test from starting a real global key listener.

    ``SpacebarDeadman.__init__`` and ``UR7eEnv.__init__`` both build a pynput
    listener inside a bare ``except Exception``.  This box has DISPLAY set, so
    without this fixture merely *constructing* the default deadman would grab
    every keystroke the operator types anywhere on the desktop for the rest of
    the pytest process.  ``None`` in sys.modules is the canonical import block.
    """

    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)


class Float32MultiArray:
    """Stand-in for std_msgs.msg.Float32MultiArray (``data`` is the contract).

    Named exactly like the real message so the subscription assertion below
    checks the type the GUI publishes, not just "some class".
    """

    def __init__(self, data=()):
        self.data = list(data)


class _FakeNode:
    """Records what RosTopicDeadman subscribes to, instead of joining a graph."""

    def __init__(self):
        self.subscriptions = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append(
            {"type": msg_type, "topic": topic, "callback": callback, "qos": qos}
        )
        return object()


@pytest.fixture
def ros_node(monkeypatch):
    """A fake rclpy node + a fake ``std_msgs`` so RosTopicDeadman constructs.

    Deliberately stubs unconditionally rather than using a real ``std_msgs``
    when one happens to be importable: the test must assert the same things on
    the robot laptop and on a dev box with no ROS at all.
    """

    pkg = types.ModuleType("std_msgs")
    msg_mod = types.ModuleType("std_msgs.msg")
    msg_mod.Float32MultiArray = Float32MultiArray
    pkg.msg = msg_mod
    monkeypatch.setitem(sys.modules, "std_msgs", pkg)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", msg_mod)
    return _FakeNode()


def _publish(deadman, *data):
    """Deliver one /hil/deadman message, exactly as the ROS callback would."""

    deadman._on_msg(Float32MultiArray(data))


# --------------------------------------------------------------------------- #
# doubles for the GelloIntervention behaviour tests                            #
# --------------------------------------------------------------------------- #
class _FakeDeadman(DeadmanSource):
    """Operator signal under direct test control (no keyboard, no topic)."""

    def __init__(self, engaged=False, gain=1.0):
        self.engaged = bool(engaged)
        self._gain = float(gain)

    def is_engaged(self):
        return self.engaged

    def gain(self):
        return self._gain


class _Backend:
    """Leader source: /gello/joint_states + trigger, already merged."""

    def __init__(self, q=Q6, age=0.0, trigger=float("nan")):
        self.q = np.asarray(q, dtype=float)
        self.age = float(age)
        self.trigger = trigger

    def get_gello_state(self):
        return np.concatenate([self.q, [self.trigger]]), self.age


class _Controller:
    def tcp_cmd(self):
        return np.eye(4)


class _StubEnv(gym.Env):
    def __init__(self, backend):
        self.backend = backend
        self.controller = _Controller()
        self.action_scale = np.array([0.01, 0.05, 1.0])
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.last_action = None

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return {}, 0.0, False, False, {}


def _wrapper(deadman, backend=None):
    return GelloIntervention(_StubEnv(backend or _Backend()), deadman=deadman)


# =========================================================================== #
# 1. What deadman=None actually resolves to                                   #
# =========================================================================== #
def test_deadman_none_falls_back_to_spacebar_not_the_gui_topic():
    """The headline fact: ``deadman=None`` silently selects the SPACEBAR.

    Not "no deadman" (which would be inert but obvious) and not the GUI's
    ``/hil/deadman`` -- a keyboard listener that cannot ever see the GUI.
    """

    expert = GelloExpert(_Backend(), deadman=None)

    assert isinstance(expert.deadman, SpacebarDeadman)
    assert not isinstance(expert.deadman, RosTopicDeadman)
    # ...and it is a real, live object, so nothing downstream ever errors out.
    assert expert.is_engaged() is False
    assert expert.gain() == 1.0


def test_gello_intervention_default_is_the_same_spacebar_fallback():
    """The wrapper adds no defaulting of its own -- it forwards None straight on."""

    env = _wrapper(deadman=None)
    assert isinstance(env.expert.deadman, SpacebarDeadman)


def test_spacebar_deadman_has_no_heartbeat_watchdog():
    """Why the spacebar fallback is unsafe on real hardware, not merely wrong.

    RosTopicDeadman stops the actor when the signal goes stale (STALE_S). The
    spacebar has no such notion: if a key-release event is ever missed (X11
    focus loss, dropped listener), ``_engaged`` stays True forever and the
    human keeps "intervening" with a leader arm nobody is holding.
    """

    assert hasattr(RosTopicDeadman, "STALE_S")
    assert not hasattr(SpacebarDeadman, "STALE_S")


# =========================================================================== #
# 2. The chain the actor really builds                                        #
# =========================================================================== #
class _ChainBackend:
    """Minimal URRosBackend stand-in: enough to build the chain, no ROS."""

    dry_run = True

    def get_joint_state(self):
        return Q6.copy(), np.zeros(6), 0.0

    def get_gello_state(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")

    def get_wrench(self):
        return None, float("inf")

    def get_gripper_percent(self):
        return 0.0, 0.0

    def reset_command_stream(self):
        pass

    def close(self):
        pass


def _find_gello_intervention(env):
    node = env
    while node is not None:
        if isinstance(node, GelloIntervention):
            return node
        node = getattr(node, "env", None)
    return None


def _build_actor_chain(monkeypatch, **kwargs):
    """``CubeInCupConfig.get_environment`` with the ROS backend swapped out."""

    from ur_experiments import cube_in_cup as task_mod

    class _Commissioned(task_mod.CubeInCupEnvConfig):
        DISPLAY_IMAGE = False

    real_cls = task_mod.UR7eEnv

    def _factory(*, fake_env, save_video, config):
        return real_cls(
            fake_env=fake_env,
            save_video=save_video,
            config=config,
            backend=_ChainBackend(),
        )

    monkeypatch.setattr(task_mod, "UR7eEnv", _factory)

    config = task_mod.CubeInCupConfig.__new__(task_mod.CubeInCupConfig)
    config.robot_config = _Commissioned()
    return config.get_environment(fake_env=False, **kwargs)


def test_actor_chain_default_gets_the_spacebar_deadman(monkeypatch):
    """THE regression this file is for.

    ``run_remote_rlpd_actor.py`` calls ``get_environment(fake_env=..., save_video=
    ..., classifier=False)`` and nothing else, so this is byte-for-byte the
    deadman the real actor runs with.  It is the spacebar -- pressing ENGAGE in
    the HIL GUI can never raise the intervention counter.
    """

    pytest.importorskip(
        "serl_launcher.wrappers.serl_obs_wrappers",
        reason="third_party/hil-serl submodule is not checked out",
    )
    env = _build_actor_chain(monkeypatch, save_video=False, classifier=False)
    try:
        intervention = _find_gello_intervention(env)
        assert intervention is not None, "GelloIntervention missing from the chain"
        assert isinstance(intervention.expert.deadman, SpacebarDeadman)
        assert not isinstance(intervention.expert.deadman, RosTopicDeadman)
    finally:
        env.close()


def test_actor_chain_propagates_an_explicit_deadman(monkeypatch):
    """The plumbing is fine -- only the *caller* never supplies a source.

    This is what makes the fix a one-line entry-point change rather than a
    wrapper rewrite: get_environment already forwards whatever it is given.
    """

    pytest.importorskip(
        "serl_launcher.wrappers.serl_obs_wrappers",
        reason="third_party/hil-serl submodule is not checked out",
    )
    supplied = _FakeDeadman()
    env = _build_actor_chain(
        monkeypatch, save_video=False, classifier=False, deadman=supplied
    )
    try:
        assert _find_gello_intervention(env).expert.deadman is supplied
    finally:
        env.close()


def test_fake_env_chain_has_no_intervention_wrapper_at_all(monkeypatch):
    """Guard against 'proving' the wiring with --fake-env.

    ``get_environment`` only inserts GelloIntervention when ``fake_env`` is
    False, so a --fake-env smoke run can never exercise the deadman path and
    must not be accepted as evidence that the GUI works.
    """

    pytest.importorskip(
        "serl_launcher.wrappers.serl_obs_wrappers",
        reason="third_party/hil-serl submodule is not checked out",
    )
    from ur_experiments import cube_in_cup as task_mod

    class _Commissioned(task_mod.CubeInCupEnvConfig):
        DISPLAY_IMAGE = False

    config = task_mod.CubeInCupConfig.__new__(task_mod.CubeInCupConfig)
    config.robot_config = _Commissioned()
    env = config.get_environment(fake_env=True)
    try:
        assert _find_gello_intervention(env) is None
    finally:
        env.close()


def test_actor_entry_point_can_select_the_topic_deadman():
    """G12 is closed: the entry point exposes --deadman and passes it through.

    This was a strict xfail while the flag was missing.  It is kept as a
    regression guard because the failure it protects against is silent: with
    no ``deadman=``, ``GelloIntervention`` falls back to ``SpacebarDeadman``
    and the rig still runs -- just with a global, watchdog-less listener where
    SPACE in any window engages intervention.
    """

    source = open(
        os.path.join(_INFRA, "scripts", "run_remote_rlpd_actor.py"),
        encoding="utf-8",
    ).read()
    assert "--deadman" in source
    assert re.search(r"deadman\s*=", source)
    # The default must be the watchdog-backed source, not the global listener.
    assert re.search(r'"--deadman"[\s\S]{0,400}default="topic"', source)


# =========================================================================== #
# 3. RosTopicDeadman <-> HIL GUI frozen contract                              #
# =========================================================================== #
def test_subscribes_to_the_topic_the_gui_publishes(ros_node):
    deadman = RosTopicDeadman(ros_node)

    assert len(ros_node.subscriptions) == 1
    sub = ros_node.subscriptions[0]
    assert sub["topic"] == "/hil/deadman"
    assert sub["type"].__name__ == "Float32MultiArray"
    assert sub["qos"] == 10
    assert sub["callback"] == deadman._on_msg


def test_never_engages_before_the_first_message(ros_node):
    """No message yet must mean 'policy drives', never 'human drives'."""

    deadman = RosTopicDeadman(ros_node)
    assert deadman.is_engaged() is False
    assert deadman.gain() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "engaged_field, expected",
    [(1.0, True), (0.0, False), (0.5, True), (0.49, False), (2.0, True)],
)
def test_parses_the_engaged_field(ros_node, engaged_field, expected):
    deadman = RosTopicDeadman(ros_node)
    _publish(deadman, engaged_field, 1.0)
    assert deadman.is_engaged() is expected


def test_parses_the_gain_field(ros_node):
    deadman = RosTopicDeadman(ros_node)
    _publish(deadman, 1.0, 0.42)
    assert deadman.gain() == pytest.approx(0.42)


@pytest.mark.parametrize(
    "raw, clamped",
    [(5.0, 1.00), (1.00, 1.00), (0.10, 0.10), (0.0, 0.10), (-3.0, 0.10)],
)
def test_clamps_gain_into_the_published_range(ros_node, raw, clamped):
    """A GUI bug (or a hand-published topic) must not scale the leader 5x."""

    deadman = RosTopicDeadman(ros_node)
    _publish(deadman, 1.0, raw)
    assert deadman.gain() == pytest.approx(clamped)


def test_short_message_degrades_to_safe_defaults(ros_node):
    """[engaged] with no gain -> gain 1.0; [] -> disengaged."""

    deadman = RosTopicDeadman(ros_node)
    _publish(deadman, 1.0)
    assert deadman.is_engaged() is True
    assert deadman.gain() == pytest.approx(1.0)

    _publish(deadman)
    assert deadman.is_engaged() is False


@pytest.mark.parametrize("engaged_field", [0.0, 1.0])
def test_stale_heartbeat_stops_instead_of_falling_back_to_policy(
    ros_node, engaged_field
):
    """The GUI beats at 20 Hz; 0.5 s of silence is ~10 lost messages.

    A fresh 0.0 is an explicit hand-back to policy.  Silence is not: after the
    first message, losing either a fresh ENGAGE or fresh DISENGAGE stream must
    abort the actor rather than authorize a policy action.
    """

    deadman = RosTopicDeadman(ros_node)
    assert RosTopicDeadman.STALE_S == 0.5

    _publish(deadman, engaged_field, 1.0)
    assert deadman.is_engaged() is (engaged_field >= 0.5)

    # still fresh (0.4 s < 0.5 s): a couple of dropped heartbeats are tolerated
    deadman._last_rx = time.monotonic() - 0.4
    assert deadman.is_engaged() is (engaged_field >= 0.5)

    # stale: fail closed, WITHOUT treating silence as a policy hand-back
    deadman._last_rx = time.monotonic() - 0.6
    with pytest.raises(DeadmanHeartbeatStaleError) as caught:
        deadman.is_engaged()
    assert caught.value.age_s > RosTopicDeadman.STALE_S
    assert caught.value.stale_s == RosTopicDeadman.STALE_S
    assert "refusing policy fallback" in str(caught.value)

    # A fresh message makes the source live again; normal state semantics resume.
    _publish(deadman, 1.0, 1.0)
    assert deadman.is_engaged() is True


def test_stale_heartbeat_never_forwards_the_policy_action(ros_node):
    """Fresh release forwards policy; silence cannot impersonate that release."""

    deadman = RosTopicDeadman(ros_node)
    env = _wrapper(deadman)
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)

    _publish(deadman, 0.0, 1.0)
    env.step(policy_action)
    np.testing.assert_array_equal(env.unwrapped.last_action, policy_action)

    env.unwrapped.last_action = None
    deadman._last_rx = time.monotonic() - 0.6
    with pytest.raises(DeadmanHeartbeatStaleError):
        env.step(policy_action)

    assert env.unwrapped.last_action is None


def test_topic_deadman_rejects_startup_after_no_first_heartbeat(monkeypatch):
    """The pre-first-message state remains covered by the existing 15 s guard."""

    from ur_experiments import cube_in_cup as task_mod

    class _NeverReceives:
        def __init__(self, node):
            del node
            self._last_rx = None

    clock = iter((0.0, 16.0))
    monkeypatch.setattr(task_mod, "RosTopicDeadman", _NeverReceives)
    monkeypatch.setattr(
        task_mod,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None),
    )

    env = SimpleNamespace(backend=SimpleNamespace(_node=object()))
    with pytest.raises(
        RuntimeError, match=r"no /hil/deadman message after 15 s"
    ):
        task_mod.CubeInCupConfig._resolve_deadman("topic", env)


def test_deadman_stale_error_exits_actor_and_closes_resources(monkeypatch):
    """``run_remote_actor`` propagates the stop into the CLI's ``finally``."""

    script = os.path.join(_INFRA, "scripts", "run_remote_rlpd_actor.py")
    spec = importlib.util.spec_from_file_location(
        "test_deadman_stale_actor_cleanup", script
    )
    assert spec is not None and spec.loader is not None
    actor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor_module)

    class _ActorEnv:
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        def __init__(self):
            self.closed = False

        def reset(self):
            return {}, {"timestamp_ns": 1}

        def step(self, action):
            del action
            raise DeadmanHeartbeatStaleError(age_s=0.6, stale_s=0.5)

        def close(self):
            self.closed = True

    class _Network:
        def __init__(self):
            self.closed = False

        def health(self):
            return True, True, "ok"

        def get_server_info(self):
            return SimpleNamespace(model_id="test-policy", protocol_version=2)

        def begin_episode(self, observation, **kwargs):
            del observation, kwargs
            return SimpleNamespace(
                action=np.zeros(7, dtype=np.float32), policy_version=0
            )

        def close(self):
            self.closed = True

    class _Config:
        NETWORK = {}
        CLASSIFIER_SIDECAR = {"enabled": False}
        max_steps = 1
        random_steps = 0
        buffer_period = 0

        def __init__(self):
            self.robot_config = SimpleNamespace(DRY_RUN=True)

    args = SimpleNamespace(
        exp_name="task",
        ur_config_module=None,
        checkpoint_path=None,
        save_video=False,
        fake_env=False,
        actor_id="test-actor",
        network_type=None,
        server_host=None,
        server_port=None,
        timeout_s=None,
        max_response_age_s=None,
        observation_schema_hash=None,
        expected_model_id=None,
        expected_reward_authority=None,
        expected_reward_model_id=None,
        deadman="topic",
        arm=True,
        no_classifier_sidecar=False,
        classifier_sidecar_interval=None,
        classifier_stationary_speed_max=None,
        classifier_escalate_probability=None,
        mock_policy_noise=0.0,
    )
    env = _ActorEnv()
    network = _Network()
    monkeypatch.setattr(actor_module, "_parse_args", lambda: args)
    monkeypatch.setattr(
        actor_module, "_load_config_mapping", lambda module: {"task": _Config}
    )
    monkeypatch.setattr(actor_module, "_build_actor_environment", lambda *_: env)
    monkeypatch.setattr(actor_module, "_preflight_command_topics", lambda *_: None)
    monkeypatch.setattr(actor_module, "create_actor_network", lambda *_, **__: network)

    with pytest.raises(DeadmanHeartbeatStaleError):
        actor_module.main()

    assert env.closed
    assert network.closed


# =========================================================================== #
# 4. Intervention bookkeeping (what the server's counter is built on)         #
# =========================================================================== #
def test_disengaged_deadman_records_no_intervention():
    env = _wrapper(_FakeDeadman(engaged=False))
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)

    _, _, _, _, info = env.step(policy_action)

    assert info["intervened"] == 0
    assert "intervene_action" not in info
    np.testing.assert_array_equal(env.unwrapped.last_action, policy_action)


def test_engaged_deadman_records_the_executed_action():
    """저장 액션 불변식: info['intervene_action'] IS what the robot ran."""

    backend = _Backend(q=Q6)
    deadman = _FakeDeadman(engaged=True)
    env = _wrapper(deadman, backend=backend)
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)

    # step 1 anchors (leader == anchor -> zero delta), step 2 has real motion
    env.step(policy_action)
    backend.q = Q6 + np.array([0.05, 0.03, -0.02, 0.0, 0.0, 0.0])
    _, _, _, _, info = env.step(policy_action)

    assert info["intervened"] == 1
    assert "intervene_action" in info
    executed = env.unwrapped.last_action
    np.testing.assert_array_equal(info["intervene_action"], executed)
    # the human action really replaced the policy action, and the policy
    # action survives as the counterfactual record
    assert not np.array_equal(executed, policy_action)
    np.testing.assert_array_equal(info["policy_action"], policy_action)
    assert np.any(np.abs(np.asarray(info["intervene_action"])[:6]) > 0.0)
    assert np.all(np.abs(np.asarray(info["intervene_action"])[:6]) <= 1.0)


def test_engaged_deadman_with_a_stale_leader_does_not_intervene():
    """Deadman ON but the GELLO topic dead: refuse, do not freeze on old joints."""

    backend = _Backend(q=Q6, age=GelloIntervention.LEADER_STALE_S + 0.1)
    _, _, _, _, info = _wrapper(_FakeDeadman(engaged=True), backend).step(
        np.zeros(7, dtype=np.float32)
    )
    assert info["intervened"] == 0
    assert "intervene_action" not in info


def test_gain_is_latched_at_the_engage_edge():
    """Moving the GUI slider mid-takeover must not rescale an in-flight delta.

    The leader nudge is deliberately small (0.005 rad ~= 2.5 mm of TCP travel,
    a quarter of ACTION_SCALE[0]=0.01 m) so neither run hits the wrapper's
    proportional anti-windup clamp -- otherwise both saturate at |act|=1 and the
    test would pass for the wrong reason.
    """

    nudge = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _takeover(gain, slider_moves_to=None):
        backend = _Backend(q=Q6)
        deadman = _FakeDeadman(engaged=True, gain=gain)
        env = _wrapper(deadman, backend=backend)
        env.step(np.zeros(7, dtype=np.float32))  # anchors, latching `gain`
        if slider_moves_to is not None:
            deadman._gain = slider_moves_to     # operator drags the slider
        backend.q = Q6 + nudge
        _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
        return np.asarray(info["intervene_action"], dtype=float)[:3]

    latched = _takeover(0.5, slider_moves_to=1.0)
    full = _takeover(1.0)

    assert np.linalg.norm(full) < 1.0, "clamp saturated; nudge is too large"
    # the mid-motion slider move was ignored: still exactly the latched 0.5x
    np.testing.assert_allclose(latched, 0.5 * full, rtol=1e-5, atol=1e-8)


# =========================================================================== #
# 5. Stuck-ON at startup: there is NO wrapper-level defence                   #
# =========================================================================== #
def test_deadman_already_engaged_at_startup_intervenes_on_the_first_step():
    """Documents a real gap: the wrapper has no require-a-release arming rule.

    If the operator left the GUI ENGAGEd (or a stale latched spacebar), the very
    first ``step`` after construction anchors and hands the robot to a leader
    arm nobody is holding.  The only guard that exists anywhere is a *preflight*
    check in ``tests/run_real_hil.py`` (``sys.exit`` if already engaged); it is
    absent from ``run_rviz_hil.py`` and from ``scripts/run_remote_rlpd_actor.py``.
    """

    env = _wrapper(_FakeDeadman(engaged=True))
    _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))

    assert info["intervened"] == 1, (
        "if this ever becomes 0, an arming rule was added -- update this test "
        "and drop the mitigation note in docs/testing/08_OPEN_GAPS.md G12"
    )
    # No arming/latching state exists to be defeated.
    for attr in ("armed", "_armed", "require_release", "_require_release"):
        assert not hasattr(env, attr)
        assert not hasattr(env.expert, attr)


def test_reset_does_not_re_arm_against_a_stuck_on_deadman():
    """reset() disengages the anchor, but re-engages instantly if still held."""

    env = _wrapper(_FakeDeadman(engaged=True))
    env.step(np.zeros(7, dtype=np.float32))
    assert env._anchored is True

    env.reset()
    assert env._anchored is False  # anchor dropped...

    _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert info["intervened"] == 1  # ...but takeover resumes with no release


def test_only_run_real_hil_preflights_the_stuck_on_case():
    """Pin where the single existing guard lives, so a refactor cannot lose it."""

    def _read(*parts):
        return open(os.path.join(_INFRA, *parts), encoding="utf-8").read()

    guarded = _read("tests", "run_real_hil.py")
    assert re.search(r"if\s+deadman\.is_engaged\(\)", guarded), (
        "run_real_hil.py lost its 'already ENGAGED -> abort' preflight"
    )

    for unguarded in (("tests", "run_rviz_hil.py"), ("scripts", "run_remote_rlpd_actor.py")):
        source = _read(*unguarded)
        if re.search(r"is_engaged\(\)", source):
            pytest.fail(
                f"{'/'.join(unguarded)} gained a stuck-ON check -- good, but "
                "this test and the report must be updated"
            )
