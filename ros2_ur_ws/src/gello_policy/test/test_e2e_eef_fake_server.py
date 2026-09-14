"""End-to-end proof of the carrot EEF-delta deploy path -- NO robot, NO checkpoint.

    cd <wt>/ros2_ur_ws && colcon build --packages-select ur_gello_bringup gello_policy \\
        --symlink-install && source install/setup.bash
    cd src/gello_policy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        python3 -m pytest -q -p no:cacheprovider test/test_e2e_eef_fake_server.py

What this file closes
---------------------
Every other test in this package exercises ONE layer (``eef_space`` arithmetic,
``eef_executor`` clamps, ``obs_assembler`` framing, ``zmq_protocol`` handshake).
Nothing proved that the layers, wired together by the real rclpy node, reproduce a
real trajectory.  Here the whole deploy path runs for real -- real
``PolicyLeaderNode`` built from the real ``config/carrot_eef_act.yaml``, real ROS
publishers/subscriptions/services, real ZMQ REQ/REP multipart -- against two fakes:

  * a fake POLICY: an in-process ZMQ REP server speaking protocol v2 that replays
    the recorded actions of a HELD-OUT episode (>= 48, i.e. outside the training
    split ``--dataset.eval_split=0.111`` = the last 6 of 54) instead of running a
    network.  A trained checkpoint is therefore not needed to test the transport,
    the handshake, the chunk marker or the executor.
  * a fake ARM: a node that publishes ``/joint_states`` + gripper + two JPEG
    ``CompressedImage`` streams and, on every ``/gello/joint_states`` target,
    snaps its joints onto that target -- a PERFECTLY tracking arm with exactly one
    tick of delay (what the bridge would do with zero slew limit and zero lag).

With a perfect arm the expected result is exact, and that is what makes this an
assertion rather than a smoke test: the dataset action is
``TCP(q_{t+1}) (-) TCP(q_t)``, the node accumulates the same deltas onto its target
pose, so the traced TCP MUST land on the dataset's own ``observation.state[6:9]``
to numerical precision.  Re-anchoring on ``chunk_start`` is a no-op for a perfect
arm (the anchor is the previous target), so the chunk boundaries every 30 actions
do not perturb the trace -- exactly the property the accumulate-on-the-target rule
of DESIGN §3.3 exists to give.

Clock
-----
The trajectory tests CANCEL the node's 30 Hz rclpy timer and call ``_on_timer()``
themselves.  That is deliberate: the claim under test is "tick k executes dataset
action k", and a wall-clock timer racing the arming service would make the mapping
from published target to dataset frame depend on scheduling.  The real timer is
exercised separately by ``test_execute_publishes_at_the_configured_rate_and_hold_freezes_it``.

ROS isolation
-------------
The module pins ``ROS_DOMAIN_ID``/``ROS_LOCALHOST_ONLY`` before ``rclpy.init`` so a
fake ``/joint_states`` can never reach a real robot graph, and so a real UR
bringup on the default domain can never drive this test's node.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import pytest

# --- ROS graph isolation: MUST happen before rclpy.init() creates the context. --
os.environ["ROS_DOMAIN_ID"] = os.environ.get("HIL_TEST_ROS_DOMAIN_ID", "87")
os.environ["ROS_LOCALHOST_ONLY"] = "1"

rclpy = pytest.importorskip(
    "rclpy", reason="needs the ROS 2 overlay (source install/setup.bash)")
zmq = pytest.importorskip("zmq", reason="needs pyzmq")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CompressedImage, JointState  # noqa: E402
from std_msgs.msg import Float32  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from gello_policy import eef_space  # noqa: E402
from gello_policy import obs_assembler as oa  # noqa: E402
from gello_policy.policy_leader_node import (  # noqa: E402
    EXECUTE, FAULT, HOLD, PolicyLeaderNode,
)

try:  # yaml ships with the ROS overlay; guard anyway so collection never explodes.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CARROT_YAML = os.path.join(PKG, "config", "carrot_eef_act.yaml")

# A1's conversion of the 54 carrot takes (DESIGN §5). Episodes 48..53 are the
# held-out tail of the train/eval split, so replaying one of them here is replaying
# data no checkpoint was fitted on.
DATASET = os.environ.get(
    "CARROT_EEF_DATASET", "/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot"
)
HELD_OUT_EPISODE = int(os.environ.get("CARROT_EEF_EPISODE", "48"))

# Interpreters that may own pandas/pyarrow when the ROS python3 does not (it does
# not: the Humble system interpreter has neither, and DESIGN §6 forbids new pip
# deps). The episode is then extracted in a subprocess instead of skipping the
# whole file -- the dataset is the point of the test.
PARQUET_PYTHONS = [
    p for p in (
        os.environ.get("CARROT_EEF_PARQUET_PYTHON"),
        "/home/laptop3/youngwoong_ws/lr_env/bin/python",
    ) if p
]

# Chunk length of the fake policy: ACT's receding horizon (n_action_steps=30), so
# chunk_start is true on tick 0, 30, 60 ... exactly like the real act_server.
CHUNK_LEN = 30
# Ticks executed by the trajectory tests. > 60 (the window the error is measured
# over) and > 2*CHUNK_LEN so at least three chunk boundaries are crossed.
N_TICKS = 70
TRACE_WINDOW = 60
# The whole point of a perfect arm: the trace is exact. 5 mm is the contract in the
# task; the measured value is ~1e-7 m (float32 dataset precision).
MAX_MEAN_TCP_ERR_M = 0.005


# =============================================================================
# dataset access
# =============================================================================
_EXTRACT_SRC = r"""
import glob, sys
import numpy as np, pandas as pd
files = sorted(glob.glob(sys.argv[1] + "/data/**/*.parquet", recursive=True))
frames = []
for f in files:
    df = pd.read_parquet(f, columns=["episode_index", "frame_index",
                                     "observation.state", "action"])
    df = df[df["episode_index"] == int(sys.argv[2])]
    if len(df):
        frames.append(df)
df = pd.concat(frames).sort_values("frame_index")
np.savez(sys.argv[3],
         state=np.stack(df["observation.state"].values).astype(np.float64),
         action=np.stack(df["action"].values).astype(np.float64))
"""


def _read_episode(dataset: str, episode: int):
    """(state (n,16), action (n,7)) float64 for one episode of A1's LeRobot dataset.

    pandas/pyarrow in-process when available, otherwise a one-shot subprocess into
    an interpreter that has them (the ROS python3 has neither and must not grow a
    dependency for a test).
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        files = sorted(glob.glob(os.path.join(dataset, "data", "**", "*.parquet"),
                                 recursive=True))
        parts = []
        for f in files:
            df = pd.read_parquet(f, columns=["episode_index", "frame_index",
                                             "observation.state", "action"])
            df = df[df["episode_index"] == episode]
            if len(df):
                parts.append(df)
        if not parts:
            return None
        df = pd.concat(parts).sort_values("frame_index")
        return (np.stack(df["observation.state"].values).astype(np.float64),
                np.stack(df["action"].values).astype(np.float64))

    for python in PARQUET_PYTHONS:
        if not os.path.exists(python):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ep.npz")
            proc = subprocess.run(
                [python, "-c", _EXTRACT_SRC, dataset, str(episode), out],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0 or not os.path.exists(out):
                continue
            with np.load(out) as d:
                return d["state"].astype(np.float64), d["action"].astype(np.float64)
    return None


@pytest.fixture(scope="module")
def episode():
    """The held-out episode, or a skip explaining exactly what is missing."""
    if not os.path.isdir(DATASET):
        pytest.skip(f"carrot EEF LeRobot dataset not found at {DATASET}")
    if not glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True):
        pytest.skip(f"no parquet under {DATASET}/data")
    data = _read_episode(DATASET, HELD_OUT_EPISODE)
    if data is None:
        pytest.skip(
            "cannot read parquet: this interpreter has no pandas/pyarrow and none "
            f"of {PARQUET_PYTHONS} worked (set CARROT_EEF_PARQUET_PYTHON)"
        )
    state, action = data
    assert state.shape[1] == eef_space.STATE_DIM
    assert action.shape[1] == eef_space.ACTION_DIM
    assert len(state) > N_TICKS + 1, f"episode {HELD_OUT_EPISODE} is too short"
    return state, action


@pytest.fixture(autouse=True)
def _rclpy_is_shut_down():
    """A test that dies before Harness.__exit__ must not poison the next one."""
    yield
    try:
        rclpy.try_shutdown()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="module")
def carrot_params():
    """policy_leader_node's section of the real deploy yaml (A6's file)."""
    if yaml is None or not os.path.exists(CARROT_YAML):
        pytest.skip(f"{CARROT_YAML} missing (or pyyaml unavailable)")
    with open(CARROT_YAML) as fh:
        return yaml.safe_load(fh)["policy_leader_node"]["ros__parameters"]


# =============================================================================
# fake policy server (ZMQ REP, protocol v2)
# =============================================================================
class FakeEefServer:
    """In-process REP server that replays recorded actions, protocol v2.

    It is deliberately NOT the production ``act_server``: the point is to test the
    ROS side against the WIRE CONTRACT (zmq_protocol.py) with a policy whose output
    is known exactly, so any deviation in the trace is the node's, not a network's.
    """

    def __init__(self, actions, *, action_space=oa.EEF_ACTION_SPACE_ID, state_dim=16,
                 action_dim=7, chunk_len=CHUNK_LEN, nan_after=None, omit_handshake=False):
        self._actions = np.asarray(actions, dtype=float)
        self._action_space = action_space
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._chunk_len = int(chunk_len)
        self._nan_after = nan_after
        self._omit_handshake = omit_handshake

        self.states = []      # every `state` list an ACT request carried
        self.chunk_starts = []
        self.protocol_errors = []   # anything the client sent that violates the wire contract
        self.n_resets = 0
        self._cursor = 0
        self._stop = threading.Event()

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._thread = threading.Thread(target=self._serve, name="fake-eef-server",
                                        daemon=True)

    # -- lifecycle --------------------------------------------------------
    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._sock.close(0)
        return False

    @property
    def n_acts(self):
        return len(self.states)

    # -- REP loop ---------------------------------------------------------
    def _serve(self):
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(50):
                continue
            frames = self._sock.recv_multipart()
            try:
                reply = self._handle(frames)
            except Exception as exc:  # noqa: BLE001 - a REP socket must always answer
                self.protocol_errors.append(f"{type(exc).__name__}: {exc}")
                reply = {oa.KEY_OK: False, oa.KEY_ERR: str(exc)}
            self._sock.send_multipart([json.dumps(reply).encode("utf-8")])

    def _handle(self, frames):
        ctrl = json.loads(frames[0].decode("utf-8"))
        cmd = ctrl.get(oa.KEY_CMD)

        if cmd == oa.CMD_RESET:
            if len(frames) != 1:
                self.protocol_errors.append(f"reset had {len(frames)} frames")
            self.n_resets += 1
            self._cursor = 0
            if self._omit_handshake:
                return {oa.KEY_OK: True}  # a pre-v2 server
            return {
                oa.KEY_OK: True,
                oa.KEY_ACTION_SPACE: self._action_space,
                oa.KEY_STATE_DIM: self._state_dim,
                oa.KEY_ACTION_DIM: self._action_dim,
                oa.KEY_POLICY_TYPE: "act",
                oa.KEY_CHECKPOINT: "<fake: replayed dataset episode>",
            }

        if cmd == oa.CMD_ACT:
            if len(frames) != 3:
                self.protocol_errors.append(f"act had {len(frames)} frames, want 3")
            for i, jpeg in enumerate(frames[1:3], start=1):
                if not bytes(jpeg).startswith(b"\xff\xd8"):
                    self.protocol_errors.append(f"cam{i} frame is not a JPEG")
            state = ctrl.get(oa.KEY_STATE)
            if not isinstance(state, list) or len(state) != self._state_dim:
                self.protocol_errors.append(
                    f"state has {None if state is None else len(state)} floats, "
                    f"want {self._state_dim}")
            self.states.append(list(state or []))

            chunk_start = (self._cursor % self._chunk_len) == 0
            self.chunk_starts.append(chunk_start)
            idx = min(self._cursor, len(self._actions) - 1)
            action = list(self._actions[idx])
            if self._nan_after is not None and self._cursor >= self._nan_after:
                action = [float("nan")] * 6 + [action[6]]
            self._cursor += 1
            return {oa.KEY_OK: True, oa.KEY_ACTION: [float(x) for x in action],
                    oa.KEY_CHUNK_START: bool(chunk_start)}

        raise ValueError(f"unknown cmd {cmd!r}")


# =============================================================================
# fake arm + cameras
# =============================================================================
def _jpeg(seed: int) -> bytes:
    """A small but genuine JPEG (SOI marker checked by the fake server)."""
    try:
        import cv2
        img = np.full((48, 64, 3), (seed * 7) % 255, dtype=np.uint8)
        img[:, :, 1] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return bytes(buf.tobytes())
    except ImportError:
        pass
    # Minimal valid-enough JPEG envelope; nothing in this test decodes it.
    return b"\xff\xd8\xff\xdb" + bytes([seed % 251]) * 64 + b"\xff\xd9"


class FakeArm(Node):
    """Publishes the observation half of the graph and tracks the leader perfectly.

    ``follow`` is off until the test arms: before hand-over the arm sits wherever
    the operator left it (the leader's HOLD pose is a chase target for
    ``gello_move_to_start``, which is not part of this path), and the arming gate
    only requires the two to agree to within ``START_GATE_RAD``.
    """

    def __init__(self, q0, grip0, cam1_topic, cam2_topic):
        super().__init__("fake_arm")
        self.q = [float(v) for v in q0]
        self.grip_pos = float(grip0)
        self.follow = False
        self.targets = []     # every /gello/joint_states position, in order
        self.grip_cmds = []   # every /robotiq_gripper/command_percent value
        self._seq = 0

        self._js_pub = self.create_publisher(JointState, "/joint_states", 100)
        self._grip_pub = self.create_publisher(
            Float32, "/robotiq_gripper/position_percent", 20)
        self._cam1_pub = self.create_publisher(CompressedImage, cam1_topic, 10)
        self._cam2_pub = self.create_publisher(CompressedImage, cam2_topic, 10)

        self.create_subscription(JointState, "/gello/joint_states", self._on_target, 10)
        self.create_subscription(Float32, "/robotiq_gripper/command_percent",
                                 self._on_grip_cmd, 20)

    # -- leader -> arm ----------------------------------------------------
    def _on_target(self, msg: JointState):
        self.targets.append([float(v) for v in msg.position])
        if self.follow:
            self.q = [float(v) for v in msg.position]
        self.publish_obs()

    def _on_grip_cmd(self, msg: Float32):
        self.grip_cmds.append(float(msg.data))
        if self.follow:
            self.grip_pos = float(msg.data)

    # -- arm -> leader ----------------------------------------------------
    def publish_obs(self):
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = list(oa.UR_JOINT_ORDER)
        js.position = list(self.q)
        self._js_pub.publish(js)

        grip = Float32()
        grip.data = float(self.grip_pos)
        self._grip_pub.publish(grip)

        self._seq += 1
        for pub, cam in ((self._cam1_pub, 1), (self._cam2_pub, 2)):
            img = CompressedImage()
            img.header.stamp = now
            img.header.frame_id = f"cam{cam}"
            img.format = "jpeg"
            img.data = _jpeg(self._seq * 2 + cam)
            pub.publish(img)


# =============================================================================
# harness
# =============================================================================
class Harness:
    """rclpy context + leader + fake arm + a Trigger client, driven by the test."""

    def __init__(self, params_file, overrides, q0, grip0, manual_clock=True):
        args = ["pytest", "--ros-args", "--params-file", params_file]
        for key, value in overrides.items():
            args += ["-p", f"{key}:={value}"]
        rclpy.init(args=args)
        self.leader = PolicyLeaderNode()
        if manual_clock:
            # Deterministic frame mapping (see the module docstring).
            for timer in self.leader.timers:
                timer.cancel()

        self.logs = []
        self.warnings = []
        self.errors = []
        self._patch_logger()
        self.infos = []
        self._spy_executor()

        self.arm = FakeArm(
            q0, grip0,
            self.leader.get_parameter("cam1_topic").value,
            self.leader.get_parameter("cam2_topic").value,
        )
        self.client_node = rclpy.create_node("e2e_test_client")
        self.start_cli = self.client_node.create_client(
            Trigger, "/policy_leader_node/start_execution")
        self.hold_cli = self.client_node.create_client(Trigger, "/policy_leader_node/hold")

        self.executor = SingleThreadedExecutor()
        for node in (self.leader, self.arm, self.client_node):
            self.executor.add_node(node)

    # -- instrumentation --------------------------------------------------
    def _patch_logger(self):
        logger = self.leader.get_logger()
        info, warn, error = logger.info, logger.warn, logger.error

        def _info(msg, *a, **kw):
            self.logs.append(str(msg))
            return info(msg, *a, **kw)

        def _warn(msg, *a, **kw):
            self.warnings.append(str(msg))
            return warn(msg, *a, **kw)

        def _error(msg, *a, **kw):
            self.errors.append(str(msg))
            return error(msg, *a, **kw)

        logger.info = _info
        logger.warn = _warn
        logger.error = _error

    def _spy_executor(self):
        """Record every EefDeltaExecutor.step info dict (clamp flags, IK diagnostics)."""
        if self.leader._eef is None:
            return
        step = self.leader._eef.step

        def _step(*a, **kw):
            q, info = step(*a, **kw)
            self.infos.append(info)
            return q, info

        self.leader._eef.step = _step

    # -- lifecycle --------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for node in (self.leader, self.arm, self.client_node):
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        rclpy.try_shutdown()
        return False

    # -- spinning ---------------------------------------------------------
    def pump(self, seconds=0.05):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.005)

    def wait(self, predicate, timeout=5.0, message="condition"):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=0.01)
        raise AssertionError(f"timed out after {timeout}s waiting for {message}")

    def call(self, client, timeout=10.0):
        assert client.wait_for_service(timeout_sec=0.0) or self.wait(
            lambda: client.service_is_ready(), timeout=timeout, message="service")
        future = client.call_async(Trigger.Request())
        self.executor.spin_until_future_complete(future, timeout_sec=timeout)
        assert future.done(), "Trigger call did not complete"
        return future.result()

    # -- driving ----------------------------------------------------------
    def prime_observations(self):
        """Publish one full observation set and wait until the leader has all of it."""
        def _ready():
            self.arm.publish_obs()
            return not self.leader._stale_obs(time.monotonic())

        self.wait(_ready, timeout=10.0,
                  message="a complete fresh observation set at the leader")

    def tick(self):
        """One leader timer callback, with the graph flushed on both sides."""
        self.pump(0.01)
        self.leader._on_timer()
        self.pump(0.01)

    def tick_until_arm_follows(self):
        """Tick once and wait until the arm has adopted the freshly published target."""
        before = len(self.arm.targets)
        self.leader._on_timer()
        self.wait(lambda: len(self.arm.targets) > before, timeout=2.0,
                  message="the arm to receive the published target")
        target = self.arm.targets[-1]
        self.wait(
            lambda: self.leader._live_q is not None
            and max(abs(a - b) for a, b in zip(self.leader._live_q, target)) < 1e-12,
            timeout=2.0, message="the leader to see the tracked joint state",
        )


def make_harness(carrot_params, server_port, q0, grip0, **overrides):
    over = {"act_port": server_port}
    over.update(overrides)
    return Harness(CARROT_YAML, over, q0, grip0)


# =============================================================================
# helpers
# =============================================================================
def _tcp_p(q):
    return eef_space.tcp_pose(q)[:3, 3]


def _arm_and_run(h, n_ticks):
    """Prime observations, arm, then run ``n_ticks`` EXECUTE ticks in lockstep."""
    h.prime_observations()
    result = h.call(h.start_cli)
    assert result.success, f"~/start_execution refused: {result.message}"
    assert h.leader._state == EXECUTE
    h.arm.follow = True
    h.arm.targets.clear()
    h.arm.grip_cmds.clear()
    for _ in range(n_ticks):
        h.tick_until_arm_follows()
    return result


def _joint_clamp_report(h, params):
    """(n_envelope_hits, n_maxdev_hits) recomputed from the published targets.

    Independent of the node's own clamp flags on purpose: if the joint envelope or
    the max-deviation clamp had engaged, the published value would be sitting ON a
    limit, and that is what is checked here.  With a perfectly tracking arm the
    live pose IS the previous target, so consecutive targets are the right pair to
    measure ``max_dev_rad`` on.
    """
    lo = np.asarray(params["joint_limits_lo"], dtype=float)
    hi = np.asarray(params["joint_limits_hi"], dtype=float)
    max_dev = float(params["max_dev_rad"])
    envelope = max_dev_hits = 0
    prev = None
    for target in h.arm.targets:
        t = np.asarray(target, dtype=float)
        if np.any(t < lo - 1e-12) or np.any(t > hi + 1e-12):
            envelope += 1
        if prev is not None and np.any(np.abs(t - prev) > max_dev + 1e-12):
            max_dev_hits += 1
        prev = t
    return envelope, max_dev_hits


# =============================================================================
# 1. the trace: a perfect arm reproduces the held-out episode
# =============================================================================
def test_eef_deploy_traces_the_held_out_episode(episode, carrot_params, capsys):
    """The full path reproduces the recorded TCP trajectory to well under 5 mm.

    The arm is parked on the EPISODE's own first pose (0.074 rad from the yaml
    ``start_pose``, i.e. inside the 0.1 rad arming gate -- what a real arm sitting
    on the start pose looks like), so the accumulation anchor is the dataset's own
    frame-0 TCP and the comparison against ``observation.state[6:9]`` is absolute.
    """
    state, action = episode
    q0 = state[0, :6]

    with FakeEefServer(action) as server:
        with make_harness(carrot_params, server.port, q0, float(state[0, 15])) as h:
            result = _arm_and_run(h, N_TICKS)

            # --- handshake ------------------------------------------------
            assert server.n_resets == 1
            assert result.message == "EXECUTE"
            # The handshake detail is LOGGED, not returned in the Trigger message.
            armed = [line for line in h.logs if "~/start_execution" in line]
            assert len(armed) == 1, h.logs
            assert f"action_space={oa.EEF_ACTION_SPACE_ID}" in armed[0]
            assert "state_dim=16" in armed[0] and "action_dim=7" in armed[0]
            assert h.leader._state_len == 16 and h.leader._action_len == 7
            assert server.protocol_errors == []
            assert server.n_acts == N_TICKS
            assert [i for i, c in enumerate(server.chunk_starts) if c] == [0, 30, 60]

            # --- what the server saw is the 16-D state of DESIGN §2 --------
            seen = np.asarray(server.states, dtype=float)
            assert seen.shape == (N_TICKS, 16)
            # state[k] is the live pose at tick k = the target published at k-1 =
            # dataset frame k (the anchor at k=0).
            state_err = np.linalg.norm(seen[:, 6:9] - state[:N_TICKS, 6:9], axis=1)

            # --- the published targets ------------------------------------
            targets = np.asarray(h.arm.targets, dtype=float)
            assert targets.shape == (N_TICKS, 6)
            traced = np.array([_tcp_p(q) for q in targets])
            # target k = dataset frame k+1 (the first tick executes action 0).
            err = np.linalg.norm(traced - state[1:N_TICKS + 1, 6:9], axis=1)

            # --- no fault, no clamp, no IK failure ------------------------
            assert h.leader._state == EXECUTE
            assert h.errors == [], h.errors
            assert not [w for w in h.warnings if "CLAMP" in w], h.warnings
            assert len(h.infos) == N_TICKS
            assert not any(i["any_clamped"] for i in h.infos)
            assert all(i["ik_ok"] and i["ik_on_branch"] for i in h.infos)
            assert all(i["ik_branch"] == 2 for i in h.infos)
            assert [i for i, info in enumerate(h.infos) if info["re_anchored"]] == [0, 30, 60]
            assert _joint_clamp_report(h, carrot_params) == (0, 0)

            # --- the gripper is the absolute recorded command -------------
            grip = np.asarray(h.arm.grip_cmds, dtype=float)
            assert grip.shape == (N_TICKS,)
            assert np.allclose(grip, np.clip(action[:N_TICKS, 6], 0.0, 1.0), atol=1e-6)

    mean60 = float(err[:TRACE_WINDOW].mean())
    with capsys.disabled():
        print(f"\n[e2e] TCP trace vs dataset episode {HELD_OUT_EPISODE}: "
              f"mean over first {TRACE_WINDOW} frames {mean60 * 1e3:.6f} mm, "
              f"max over {N_TICKS} {err.max() * 1e3:.6f} mm; "
              f"observation seen by the server {state_err.max() * 1e3:.6f} mm max")
    assert mean60 < MAX_MEAN_TCP_ERR_M      # the contract
    assert err.max() < MAX_MEAN_TCP_ERR_M
    assert state_err.max() < MAX_MEAN_TCP_ERR_M
    # ... and the real claim. A perfect arm reproduces the episode EXACTLY, so the
    # only error left is the dataset's float32 rounding (~1e-7 m measured). This
    # tight gate is what would catch a systematic error too small for the 5 mm one:
    # scaling every delta by 0.999 already lands at 6e-5 m here.
    assert err.max() < 1e-5, "the accumulation is no longer exact for a perfect arm"


def test_arming_anywhere_in_the_gate_replays_the_same_motion(episode, carrot_params,
                                                             capsys):
    """Armed on the yaml ``start_pose`` the trace is the SAME motion, rigidly offset.

    This is the honest reading of "the arm is on start_pose": the policy's deltas
    are relative, so the executed path is the dataset path translated by
    ``TCP(start_pose) - TCP(dataset frame 0)``.  Asserting that the OFFSET-REMOVED
    trace still matches to micrometres is what proves the accumulation is exact and
    anchor-independent -- and it quantifies how far a real arm parked at the gate's
    edge ends up from the demonstrated path.
    """
    state, action = episode
    q0 = np.asarray(carrot_params["start_pose"], dtype=float)
    assert np.max(np.abs(q0 - state[0, :6])) < 0.1  # inside START_GATE_RAD

    with FakeEefServer(action) as server:
        with make_harness(carrot_params, server.port, q0,
                          float(carrot_params["start_gripper"])) as h:
            _arm_and_run(h, N_TICKS)
            targets = np.asarray(h.arm.targets, dtype=float)
            traced = np.array([_tcp_p(q) for q in targets])
            anchor = _tcp_p(q0)
            offset = anchor - state[0, 6:9]
            rel_err = np.linalg.norm(traced - (state[1:N_TICKS + 1, 6:9] + offset), axis=1)
            assert h.leader._state == EXECUTE
            assert not any(i["any_clamped"] for i in h.infos)
            assert _joint_clamp_report(h, carrot_params) == (0, 0)

    with capsys.disabled():
        print(f"[e2e] armed on the yaml start_pose: constant offset "
              f"{np.linalg.norm(offset) * 1e3:.2f} mm, offset-removed trace error "
              f"max {rel_err.max() * 1e6:.3f} um")
    assert rel_err.max() < 1e-4  # 0.1 mm; measured ~3e-8 m


# =============================================================================
# 2. the real 30 Hz timer, and ~/hold
# =============================================================================
def test_execute_publishes_at_the_configured_rate_and_hold_freezes_it(
        episode, carrot_params):
    """The node's OWN timer drives EXECUTE at ~30 Hz; ``~/hold`` stops the motion.

    ``~/hold`` does not stop publishing (that would trip the bridge's staleness
    watchdog) -- it stops publishing CHANGES: the same live pose, forever, and not
    one further query to the policy server.
    """
    state, action = episode
    with FakeEefServer(action) as server:
        h = Harness(CARROT_YAML, {"act_port": server.port}, state[0, :6],
                    float(state[0, 15]), manual_clock=False)
        with h:
            h.prime_observations()
            assert h.call(h.start_cli).success
            h.arm.follow = True
            h.arm.targets.clear()

            t0 = time.monotonic()
            h.wait(lambda: len(h.arm.targets) >= 30, timeout=5.0,
                   message="30 published targets")
            rate = len(h.arm.targets) / (time.monotonic() - t0)
            assert 15.0 < rate < 45.0, f"publish rate {rate:.1f} Hz is not ~30 Hz"
            assert h.leader._state == EXECUTE

            assert h.call(h.hold_cli).success
            assert h.leader._state == HOLD
            acts_at_hold = server.n_acts
            n_at_hold = len(h.arm.targets)
            h.pump(0.5)

    frozen = np.asarray(h.arm.targets[n_at_hold + 1:], dtype=float)
    assert len(frozen) >= 5, "HOLD must keep publishing (the bridge watchdog needs it)"
    assert np.allclose(frozen, frozen[0]), "HOLD must publish a CONSTANT pose"
    assert server.n_acts == acts_at_hold, "HOLD must not query the policy server"
    assert h.leader._eef.target_pose is None, "HOLD must drop the accumulated target"


# =============================================================================
# 3. refusals and faults
# =============================================================================
def test_a_joint_space_server_is_refused_at_arming(episode, carrot_params):
    """A joint checkpoint can never drive eef mode (DESIGN §3.7) -- refused, not moved."""
    state, action = episode
    with FakeEefServer(action, action_space=oa.JOINT_ACTION_SPACE_ID,
                       state_dim=7) as server:
        with make_harness(carrot_params, server.port, state[0, :6],
                          float(state[0, 15])) as h:
            h.prime_observations()
            result = h.call(h.start_cli)
            assert not result.success
            assert "action_space" in result.message and "joint" in result.message
            assert h.leader._state == HOLD          # unchanged: the operator may retry
            assert server.n_resets == 1
            assert server.n_acts == 0               # never sent an observation
            h.tick()
            assert server.n_acts == 0


def test_a_pre_v2_server_is_refused_at_arming(episode, carrot_params):
    """A RESET reply without the handshake is a pre-v2 server: refuse, never assume."""
    state, action = episode
    with FakeEefServer(action, omit_handshake=True) as server:
        with make_harness(carrot_params, server.port, state[0, :6],
                          float(state[0, 15])) as h:
            h.prime_observations()
            result = h.call(h.start_cli)
            assert not result.success
            assert "protocol v2" in result.message
            assert h.leader._state == HOLD
            assert server.n_acts == 0


def test_a_nan_action_drives_fault_and_stops_publishing(episode, carrot_params):
    """A non-finite action FAULTs the node ON THE FIRST REPLY that carries one.

    UPDATED for R2/F3.  Before the finiteness check in ``parse_action_reply``, a NaN
    delta reached the executor, could not be inverted, and cost
    ``eef_ik_fail_fault_n`` held ticks before the IK-failure streak faulted -- while
    ``action[6]`` (the gripper), which no IK ever touches, was published verbatim as
    ``Float32(nan)``.  ``json.loads`` accepts the bare ``NaN`` literal by default, so
    this is reachable from any non-conforming server.  Now the reply is rejected at
    the parser and the FAULT is immediate, in BOTH action spaces.

    FAULT is fail-silent by contract: the node stops publishing so the bridge's
    staleness watchdog halts the arm.  The node-level IK-failure streak is covered
    separately in ``test_node_eef_guards.py``.
    """
    state, action = episode
    nan_after = 5
    with FakeEefServer(action, nan_after=nan_after) as server:
        with make_harness(carrot_params, server.port, state[0, :6],
                          float(state[0, 15])) as h:
            h.prime_observations()
            assert h.call(h.start_cli).success
            h.arm.follow = True
            h.arm.targets.clear()
            h.arm.grip_cmds.clear()
            for _ in range(nan_after):
                h.tick_until_arm_follows()
            assert h.leader._state == EXECUTE

            h.tick()  # the FIRST non-finite reply is enough now
            assert h.leader._state == FAULT, [i["ik_fail_streak"] for i in h.infos]
            assert any("is not finite" in e for e in h.errors), h.errors
            # The executor never saw the NaN: the parser rejected the reply first.
            assert len(h.infos) == nan_after

            # every published target AND gripper command is finite, and publishing
            # has stopped
            h.pump(0.3)  # flush every straggler before the counts are frozen
            published = np.asarray(h.arm.targets, dtype=float)
            assert np.all(np.isfinite(published))
            assert np.all(np.isfinite(np.asarray(h.arm.grip_cmds, dtype=float)))
            n = len(h.arm.targets)
            acts = server.n_acts
            for _ in range(5):
                h.tick()
            assert len(h.arm.targets) == n, "FAULT must stop publishing"
            assert server.n_acts == acts, "FAULT must stop querying the server"
