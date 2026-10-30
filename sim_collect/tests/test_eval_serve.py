"""Tests for sim_collect/eval/serve_policy.sh.

Two things are exercised for real:

  * the **probe** (the RESET round-trip that decides "the server is up"), against
    F1's ``sim_collect/tests/stub_policy_server.py`` when it is importable (so both
    suites drive the same server), and otherwise against the small fallback below.
    The refusal case (RESET answering ``ok:false``) always uses the fallback --
    F1's stub cannot express it.
  * the **--check** preview and the argument validation, which is the only way to
    see the remote command lines without starting anything on kanu.

Nothing here starts a real lerobot server (torch + a checkpoint is GBs of RAM)
and nothing here ssh's anywhere: the remote previews are asserted against a host
name that does not resolve, which is itself the proof that --check makes no
connection.

    cd /home/laptop3/gello_software && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      .venv/bin/python -m pytest -q -p no:cacheprovider \\
      sim_collect/tests/test_eval_serve.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import pytest

zmq = pytest.importorskip("zmq")

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "sim_collect" / "eval" / "serve_policy.sh"
UNROUTABLE_HOST = "sim-eval-no-such-host.invalid"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run_script(*args: str, env_extra: dict[str, str] | None = None, timeout: float = 60.0):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=str(REPO),
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# F1 owns the canonical stub; use it when it is there so this test and
# test_eval_policies.py exercise the same server.  The fallback below keeps this
# file standalone (and is the only stub that can refuse a RESET, which F1's
# cannot express).
try:  # pragma: no cover - exercised by whichever branch is live
    from sim_collect.tests.stub_policy_server import StubPolicyServer  # type: ignore
except Exception:  # pragma: no cover
    StubPolicyServer = None  # type: ignore[assignment]


class _MiniStub:
    """Fallback REP server speaking the real RESET/ACT protocol (DESIGN §1.1)."""

    def __init__(self, port: int = 0, *, reset_ok: bool = True, reset_extra: dict | None = None):
        self.port = int(port) or free_port()
        self.reset_ok = reset_ok
        self.reset_extra = reset_extra or {}
        self.n_reset = 0
        self._stop = threading.Event()
        self._bound = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{self.port}")
        self._bound.set()
        try:
            while not self._stop.is_set():
                if not sock.poll(100):
                    continue
                frames = sock.recv_multipart()
                cmd = json.loads(frames[0].decode("utf-8")).get("cmd")
                if cmd == "reset":
                    self.n_reset += 1
                    reply = ({"ok": True, **self.reset_extra} if self.reset_ok
                             else {"ok": False, "err": "stub refuses"})
                else:
                    reply = {"ok": True, "action": [0.0] * 7}
                sock.send_multipart([json.dumps(reply).encode("utf-8")])
        finally:
            sock.close(linger=0)
            ctx.term()

    def __enter__(self) -> "_MiniStub":
        self._thread.start()
        assert self._bound.wait(5.0), "stub server never bound"
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def ok_stub(**kwargs):
    """A stub that answers RESET with ok:true -- F1's when available."""
    if StubPolicyServer is not None:
        return StubPolicyServer(port=free_port(), **kwargs)
    return _MiniStub(port=free_port(), **kwargs)


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------
def test_script_is_executable_and_valid_bash():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_help_lists_every_mode():
    result = run_script("--help")
    assert result.returncode == 0
    for flag in ("--type", "--checkpoint", "--remote", "--gpus", "--sync",
                 "--stop", "--check", "--probe", "--remote-venv", "--remote-repo"):
        assert flag in result.stdout, flag


@pytest.mark.parametrize("policy_type,port,nsteps", [("act", "5591", "30"),
                                                     ("diffusion", "5592", "32"),
                                                     ("fm", "5593", "24")])
def test_per_type_defaults(policy_type, port, nsteps):
    result = run_script("--type", policy_type, "--check", "--checkpoint", "/x/pretrained_model")
    assert result.returncode == 0
    assert f"port          : {port}" in result.stdout
    assert f"n_action_steps: {nsteps}" in result.stdout


def test_bad_type_and_unknown_flag_are_usage_errors():
    assert run_script("--type", "bogus", "--check").returncode == 2
    assert run_script("--type", "act", "--nonsense").returncode == 2
    # start without a checkpoint must refuse before touching anything
    result = run_script("--type", "act")
    assert result.returncode == 2 and "--checkpoint" in result.stderr


def test_probe_without_type_needs_a_port():
    result = run_script("--probe")
    assert result.returncode == 2
    assert "--type" in result.stderr or "--port" in result.stderr


# ---------------------------------------------------------------------------
# --check previews (no ssh: the host deliberately does not resolve)
# ---------------------------------------------------------------------------
def test_local_check_previews_the_run_server_script():
    result = run_script("--type", "act", "--check", "--checkpoint", "/ckpt/pretrained_model")
    assert result.returncode == 0
    out = result.stdout
    assert "run_act_server.sh" in out
    assert "'--device' 'cpu'" in out                  # laptop3 default
    assert "'--port' '5591'" in out
    assert "--task" not in out.split("1) run in the foreground")[1].split("then:")[0]
    assert "run_eval" in out and "zmq://127.0.0.1:5591" in out


def test_remote_check_prints_the_exact_kanu_command_lines():
    result = run_script(
        "--type", "fm", "--remote", UNROUTABLE_HOST, "--gpus", "6,7", "--sync",
        "--check", "--checkpoint", "/remote/ckpt/pretrained_model",
    )
    assert result.returncode == 0
    out = result.stdout
    # code sync
    assert "rsync -az --delete --exclude __pycache__/" in out
    assert "policy_server/ " in out and f"{UNROUTABLE_HOST}:" in out
    # detached launch, one GPU set, loopback bind, log under ~/logs
    assert "nohup env" in out
    assert "CUDA_VISIBLE_DEVICES='6,7'" in out
    assert "'--host' '127.0.0.1'" in out
    assert "'--port' '5593'" in out
    assert "'--task' 'Put carrot in pot'" in out
    assert "$HOME/logs/sim_eval_fm_5593_" in out
    assert "</dev/null &" in out
    # FM needs the offline flags run_fm_server.sh sets
    assert "HF_HUB_OFFLINE=1" in out and "TRANSFORMERS_OFFLINE=1" in out
    # tunnel
    assert "ssh -N -T" in out and "-o ExitOnForwardFailure=yes" in out
    assert "-L 127.0.0.1:5593:127.0.0.1:5593" in out
    # teardown hint
    assert "--stop" in out


def test_remote_check_defaults_point_at_the_kanu_lerobot_venv():
    result = run_script("--type", "act", "--remote", UNROUTABLE_HOST, "--check",
                        "--checkpoint", "/remote/ckpt")
    out = result.stdout
    assert "/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/lr_env" in out
    assert "sim_eval_policy_server" in out
    assert "'--device' 'cuda'" in out                 # remote default flips to cuda
    assert "HF_HUB_OFFLINE" not in out               # act/diffusion never set it


def test_remote_check_honours_overrides():
    result = run_script("--type", "diffusion", "--remote", UNROUTABLE_HOST,
                        "--remote-venv", "/opt/myvenv", "--remote-repo", "/scratch/srv",
                        "--port", "5700", "--local-port", "5701",
                        "--n-action-steps", "8", "--check", "--checkpoint", "/c")
    out = result.stdout
    assert "/opt/myvenv/bin/python" in out
    assert "/scratch/srv/policy_server/diffusion_server.py" in out
    assert "'--n-action-steps' '8'" in out
    assert "-L 127.0.0.1:5701:127.0.0.1:5700" in out


def test_extra_args_after_double_dash_are_forwarded():
    result = run_script("--type", "fm", "--remote", UNROUTABLE_HOST, "--check",
                        "--checkpoint", "/c", "--", "--num-integration-steps", "10")
    assert "'--num-integration-steps' '10'" in result.stdout


def test_task_on_a_non_fm_type_warns_and_is_not_forwarded():
    result = run_script("--type", "act", "--check", "--checkpoint", "/c",
                        "--task", "Put carrot in pot")
    assert result.returncode == 0
    assert "FM-only" in result.stderr
    assert "'--task'" not in result.stdout


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def test_probe_succeeds_against_a_stub_server():
    with ok_stub(reset_extra={"policy_type": "stub", "state_dim": 7}) as stub:
        result = run_script("--port", str(stub.port), "--probe", "--wait-s", "15")
    assert result.returncode == 0, result.stderr
    assert "SIM_EVAL_PROBE=ok" in result.stdout
    # the optional v2 RESET fields are surfaced, not swallowed
    assert '"state_dim": 7' in result.stdout
    assert stub.n_reset == 1


def test_probe_reports_a_refusing_server_immediately():
    started = time.time()
    with _MiniStub(reset_ok=False) as stub:   # F1's stub cannot refuse a RESET
        result = run_script("--port", str(stub.port), "--probe", "--wait-s", "60")
    assert result.returncode == 1
    assert "SIM_EVAL_PROBE=refused" in result.stderr
    assert "stub refuses" in result.stderr
    # ok:false is a real failure, so it must NOT sit out the 60 s deadline
    assert time.time() - started < 20.0


def test_probe_times_out_on_a_dead_port():
    port = free_port()
    result = run_script("--port", str(port), "--probe", "--wait-s", "2", timeout=40)
    assert result.returncode == 1
    assert "SIM_EVAL_PROBE=timeout" in result.stderr


def test_probe_uses_the_real_obs_assembler_framing():
    """The probe must send the deploy's byte-exact RESET frame, not its own."""
    port = free_port()
    seen: list[list[bytes]] = []

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")

    def serve():
        if sock.poll(20000):
            frames = sock.recv_multipart()
            seen.append(frames)
            sock.send_multipart([b'{"ok":true}'])

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    result = run_script("--port", str(port), "--probe", "--wait-s", "15")
    thread.join(timeout=5.0)
    sock.close(linger=0)
    ctx.term()

    assert result.returncode == 0, result.stderr
    assert seen and seen[0] == [b'{"cmd":"reset"}']


# ---------------------------------------------------------------------------
# --stop
# ---------------------------------------------------------------------------
def test_stop_without_state_or_remote_is_a_no_op(tmp_path):
    result = run_script("--type", "fm", "--stop",
                        env_extra={"SIM_EVAL_STATE_DIR": str(tmp_path)})
    assert result.returncode == 0
    assert "no local tunnel state" in result.stdout
    assert "nothing to stop" in result.stdout


def test_stop_refuses_to_kill_a_pid_that_is_not_our_tunnel(tmp_path):
    """A pid number alone is never enough -- the cmdline has to prove it."""
    victim = subprocess.Popen(["sleep", "30"])
    try:
        state = tmp_path / f"local_fm_5593.tunnel"
        state.write_text(f"pid={victim.pid}\n", encoding="utf-8")
        result = run_script("--type", "fm", "--stop",
                            env_extra={"SIM_EVAL_STATE_DIR": str(tmp_path)})
        assert result.returncode == 0
        assert "REFUSING to kill" in result.stderr
        assert victim.poll() is None, "the unrelated process was killed"
    finally:
        victim.terminate()
        victim.wait(timeout=10)


def test_stop_kills_a_matching_tunnel_process(tmp_path):
    """A process whose cmdline carries our -L spec IS ours and gets stopped."""
    forward = "127.0.0.1:5593:127.0.0.1:5593"
    # `sleep` with the -L spec in argv stands in for the ssh tunnel: --stop must
    # match on the forward spec, which is what makes the tunnel identifiable.
    victim = subprocess.Popen(["bash", "-c", f'exec -a "ssh -N -L {forward} kanu" sleep 30'])
    try:
        (tmp_path / "local_fm_5593.tunnel").write_text(f"pid={victim.pid}\n", encoding="utf-8")
        result = run_script("--type", "fm", "--stop",
                            env_extra={"SIM_EVAL_STATE_DIR": str(tmp_path)})
        assert result.returncode == 0, result.stderr
        assert f"stopped tunnel PID {victim.pid}" in result.stdout
        victim.wait(timeout=10)
        assert not (tmp_path / "local_fm_5593.tunnel").exists()
    finally:
        if victim.poll() is None:
            victim.terminate()
            victim.wait(timeout=10)


def test_stop_needs_a_type_for_a_remote(tmp_path):
    result = run_script("--port", "5593", "--remote", UNROUTABLE_HOST, "--stop",
                        env_extra={"SIM_EVAL_STATE_DIR": str(tmp_path)})
    assert result.returncode == 2
    assert "--type" in result.stderr


# ---------------------------------------------------------------------------
# guard rails that must not regress
# ---------------------------------------------------------------------------
def test_script_never_uses_pkill_or_killall():
    code = "\n".join(line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))
    for banned in ("pkill", "killall", "kill -9 $(", "fuser -k"):
        assert banned not in code, f"{banned}: kanu hosts other people's jobs"


def test_script_never_installs_packages():
    source = SCRIPT.read_text(encoding="utf-8")
    # The pip line is PRINTED for the operator, never executed.
    assert "pip install" in source
    for line in source.splitlines():
        stripped = line.strip()
        if "pip install" in stripped:
            assert stripped.startswith(("#", "echo", "PIPNAMES")), stripped
