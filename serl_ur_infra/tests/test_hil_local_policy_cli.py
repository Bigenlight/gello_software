"""``HIL_POLICY_MODE`` is opt-in, and "opt-in" has to be provable.

Local inference moves policy inference off the learner host and onto laptop3.
The whole design's first constraint is that an operator who does NOT ask for it
gets byte-identical behaviour, so these tests do not read the shell scripts and
nod at them: they EXTRACT the command-building blocks, expand them in a stub
shell, and compare the resulting argv with the mode off, explicitly ``remote``,
and ``local``.  Off and ``remote`` must be identical to each other AND to the
literal command this rig spells out (that literal is the pre-change command --
change either side and this fails).

The session script is checked through its own ``--plan`` mode, which prints
every command it would run and starts nothing, so the comparison is over the
real thing rather than a re-implementation of it.

Nothing here starts a proxy, opens 50253, contacts a host, or touches the
robot.  ``ssh`` is shadowed by a stub whose invocation count is itself an
assertion, and the "venv python" is a shell script that records its argv.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_WS = _ROOT / "ros2_ur_ws"
_ACTOR_SH = _WS / "run_hil_actor.sh"
_SESSION_SH = _WS / "run_hil_session.sh"
_PROXY_SH = _WS / "run_hil_local_policy.sh"

#: The proxy's own model id.  Pinned in the design spec and in
#: ``ur_env/local_policy/runtime.py``; restated here on purpose so a rename has
#: to be a deliberate edit in three places rather than a silent drift in one.
LOCAL_MODEL_ID = "hil-serl-local-policy-resnet10-manual-v1"
REMOTE_MODEL_ID = "hil-serl-hybrid-sac-resnet10-trunk-cache-v1"


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def _clean_env(**overrides: str) -> dict[str, str]:
    """os.environ minus everything these scripts read, plus explicit values."""

    drop = {
        key
        for key in os.environ
        if key.startswith(("HIL_", "SERVER_", "EXPECTED_", "LOCAL_POLICY_"))
    }
    drop |= {"TIMEOUT_S", "MAX_RESPONSE_AGE_S", "OBS_SCHEMA_HASH", "EXP_NAME"}
    env = {key: value for key, value in os.environ.items() if key not in drop}
    env.update(overrides)
    return env


def _block(path: Path, first: str, last: str) -> str:
    """Lines from the one starting with ``first`` to the one starting ``last``."""

    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith(first)
    )
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].startswith(last)
    )
    return "\n".join(lines[start : end + 1])


def _actor_config_block() -> str:
    return _block(_ACTOR_SH, 'POLICY_MODE="${HIL_POLICY_MODE', "EXPECTED_REWARD_MODEL_ID=")


def _actor_command_block() -> str:
    return _block(_ACTOR_SH, 'ACTOR_CMD=("$ACTOR_PY"', '    "${PASSTHRU[@]}")')


def _expand_actor_command(tmp_path: Path, mode: str | None, **extra: str) -> list[str]:
    """Run the actor script's own config + ACTOR_CMD blocks and capture argv."""

    capture = tmp_path / f"argv-{mode or 'unset'}.nul"
    harness = tmp_path / f"actor-{mode or 'unset'}.sh"
    harness.write_text(
        "set -eu\n"
        'ACTOR_PY="/stub/venv/bin/python"\n'
        'ACTOR_SCRIPT="/stub/run_remote_rlpd_actor.py"\n'
        "PASSTHRU=(--arm --deadman topic)\n"
        f"{_actor_config_block()}\n"
        f"{_actor_command_block()}\n"
        f'printf "%s\\0" "${{ACTOR_CMD[@]}}" >{str(capture)!r}\n',
        encoding="utf-8",
    )
    env = _clean_env(**extra)
    if mode is not None:
        env["HIL_POLICY_MODE"] = mode
    result = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    return [token for token in capture.read_bytes().decode().split("\0") if token]


def _run_plan(mode: str | None, *args: str) -> subprocess.CompletedProcess:
    env = _clean_env()
    if mode is not None:
        env["HIL_POLICY_MODE"] = mode
    return subprocess.run(
        ["bash", str(_SESSION_SH), "--plan", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_WS),
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# 0. The scripts parse.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [_ACTOR_SH, _SESSION_SH, _PROXY_SH])
def test_scripts_are_syntactically_valid(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_proxy_wrapper_is_executable() -> None:
    # run_hil_session.sh refuses to start in local mode otherwise, and a file
    # that lost its +x bit in a patch transfer is a real thing that happens.
    assert os.access(_PROXY_SH, os.X_OK)


# ---------------------------------------------------------------------------
# 1. Default-mode purity: run_hil_actor.sh
# ---------------------------------------------------------------------------


def _expected_remote_argv() -> list[str]:
    return [
        "/stub/venv/bin/python",
        "/stub/run_remote_rlpd_actor.py",
        "--exp-name",
        "cube_in_cup",
        "--ur-config-module",
        "ur_experiments.mappings",
        "--network-type",
        "grpc",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "50153",
        "--timeout-s",
        "1.5",
        "--max-response-age-s",
        "2.0",
        "--observation-schema-hash",
        "3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903",
        "--expected-model-id",
        REMOTE_MODEL_ID,
        "--expected-reward-authority",
        "server_classifier",
        "--expected-reward-model-id",
        "cube-in-cup-all3-ckpt150+sidecar-v1",
        "--arm",
        "--deadman",
        "topic",
    ]


def test_actor_command_with_the_mode_unset_is_the_pre_change_command(tmp_path):
    assert _expand_actor_command(tmp_path, None) == _expected_remote_argv()


def test_actor_command_is_identical_unset_and_explicitly_remote(tmp_path):
    assert _expand_actor_command(tmp_path, None) == _expand_actor_command(
        tmp_path, "remote"
    )


def test_local_mode_switches_the_port_and_the_model_id_and_nothing_else(tmp_path):
    remote = _expand_actor_command(tmp_path, "remote")
    local = _expand_actor_command(tmp_path, "local")
    assert len(remote) == len(local)
    differing = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(remote, local))
        if before != after
    ]
    assert [(before, after) for _, before, after in differing] == [
        ("50153", "50253"),
        (REMOTE_MODEL_ID, LOCAL_MODEL_ID),
    ]
    # Spelled out, because these are the two the operator actually cares about.
    assert local[local.index("--server-port") + 1] == "50253"
    assert local[local.index("--expected-model-id") + 1] == LOCAL_MODEL_ID
    # The proxy MIRRORS the server's reward fields and observation hash; if
    # local mode ever rewrote them, the proxy<->server handshake would stop
    # being able to catch a genuine contract mismatch.
    assert local[local.index("--server-host") + 1] == "127.0.0.1"
    assert local[local.index("--expected-reward-authority") + 1] == "server_classifier"
    assert (
        local[local.index("--observation-schema-hash") + 1]
        == remote[remote.index("--observation-schema-hash") + 1]
    )
    assert (
        local[local.index("--expected-reward-model-id") + 1]
        == remote[remote.index("--expected-reward-model-id") + 1]
    )


def test_local_mode_honours_hil_local_policy_port(tmp_path):
    argv = _expand_actor_command(tmp_path, "local", HIL_LOCAL_POLICY_PORT="50999")
    assert argv[argv.index("--server-port") + 1] == "50999"


def test_an_explicit_server_port_still_wins_in_both_modes(tmp_path):
    for mode in (None, "remote", "local"):
        argv = _expand_actor_command(tmp_path, mode, SERVER_PORT="51234")
        assert argv[argv.index("--server-port") + 1] == "51234"


def test_an_explicit_expected_model_id_still_wins_in_both_modes(tmp_path):
    for mode in (None, "remote", "local"):
        argv = _expand_actor_command(tmp_path, mode, EXPECTED_MODEL_ID="custom-id")
        assert argv[argv.index("--expected-model-id") + 1] == "custom-id"


def test_the_actor_script_no_longer_hardcodes_either_default_twice():
    """The literals must live in exactly one place: the mode case block."""

    text = _ACTOR_SH.read_text(encoding="utf-8")
    assert text.count('DEFAULT_SERVER_PORT="50153"') == 1
    assert text.count(f'DEFAULT_EXPECTED_MODEL_ID="{REMOTE_MODEL_ID}"') == 1
    assert 'SERVER_PORT="${SERVER_PORT:-$DEFAULT_SERVER_PORT}"' in text
    assert 'EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-$DEFAULT_EXPECTED_MODEL_ID}"' in text


# ---------------------------------------------------------------------------
# 2. Default-mode purity: run_hil_session.sh (through its own --plan)
# ---------------------------------------------------------------------------


def test_session_plan_is_identical_unset_and_explicitly_remote():
    unset = _run_plan(None)
    remote = _run_plan("remote")
    assert unset.returncode == 0, unset.stderr
    assert remote.returncode == 0, remote.stderr
    assert unset.stdout == remote.stdout
    assert "local" not in unset.stdout.lower()


@pytest.mark.parametrize("extra", [(), ("--no-arm",)])
def test_session_plan_local_only_adds_the_proxy_and_changes_no_command(extra):
    remote = _run_plan("remote", *extra)
    local = _run_plan("local", *extra)
    assert local.returncode == 0, local.stderr

    def commands(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.startswith("  /") or line.startswith("  env ")]

    added = [line for line in commands(local.stdout) if line not in commands(remote.stdout)]
    assert len(added) == 1
    assert "run_hil_local_policy.sh" in added[0]
    # Every command the remote plan had is still there, unchanged and in order.
    assert [line for line in commands(local.stdout) if line != added[0]] == commands(
        remote.stdout
    )


def _session_mode_block_env(mode: str | None, tmp_path: Path) -> dict[str, str]:
    """Run the session's mode block alone and report what it exported."""

    body = _block(_SESSION_SH, 'POLICY_MODE="${HIL_POLICY_MODE', "esac")
    harness = tmp_path / f"mode-{mode or 'unset'}.sh"
    harness.write_text(
        "set -Eeuo pipefail\n"
        + body
        + "\n"
        + 'printf "POLICY_LOCAL=%s\\n" "$POLICY_LOCAL"\n'
        + "env | grep '^HIL_POLICY_MODE=' || true\n",
        encoding="utf-8",
    )
    env = _clean_env()
    if mode is not None:
        env["HIL_POLICY_MODE"] = mode
    result = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def test_remote_mode_adds_nothing_to_the_child_environment(tmp_path):
    """Purity is not just argv: an added env var is a changed child too."""

    reported = _session_mode_block_env(None, tmp_path)
    assert reported["POLICY_LOCAL"] == "0"
    assert "HIL_POLICY_MODE" not in reported


def test_local_mode_exports_the_mode_so_the_actor_wrapper_sees_it(tmp_path):
    reported = _session_mode_block_env("local", tmp_path)
    assert reported["POLICY_LOCAL"] == "1"
    assert reported["HIL_POLICY_MODE"] == "local"


# ---------------------------------------------------------------------------
# 3. Fail-closed on an unknown mode (mirrors HIL_STARTUP_DEADMAN)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bogus", ["Local", "LOCAL", "1", "on", "yes", " local"])
def test_session_refuses_an_unknown_mode_before_starting_anything(bogus):
    result = _run_plan(bogus)
    assert result.returncode == 2, result.stdout
    assert "HIL_POLICY_MODE" in result.stderr
    # --plan prints the command table; a refusal must beat it to the terminal.
    assert "[plan]" not in result.stdout


@pytest.mark.parametrize("bogus", ["Local", "LOCAL", "1", "on", "yes", " local"])
def test_actor_refuses_an_unknown_mode_before_any_preflight(bogus):
    result = subprocess.run(
        ["bash", str(_ACTOR_SH), "--help"],
        capture_output=True,
        text=True,
        env=_clean_env(HIL_POLICY_MODE=bogus),
        cwd=str(_WS),
    )
    assert result.returncode == 2, result.stdout
    assert "HIL_POLICY_MODE must be 'remote' or 'local'" in result.stderr
    assert result.stdout == ""


def test_an_empty_mode_reads_as_unset_exactly_like_hil_startup_deadman(tmp_path):
    """``${VAR:-remote}``: an exported-but-empty variable is not a typo class.

    Stated rather than discovered, because the fail-closed case block above
    deliberately does NOT see it -- the same is already true of
    HIL_STARTUP_DEADMAN and HIL_ACTOR_EXIT_MAP, and one shell idiom across all
    three is worth more than a special case here.
    """

    assert _expand_actor_command(tmp_path, "") == _expand_actor_command(tmp_path, None)
    assert _run_plan("").stdout == _run_plan(None).stdout


def test_a_valid_mode_does_not_break_the_actor_help_path():
    for mode in (None, "remote", "local"):
        env = _clean_env()
        if mode is not None:
            env["HIL_POLICY_MODE"] = mode
        result = subprocess.run(
            ["bash", str(_ACTOR_SH), "--help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_WS),
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 4. run_hil_local_policy.sh
# ---------------------------------------------------------------------------


class _ProxyRig:
    """A stub world for the wrapper: fake venv, fake ssh, fake --check."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.venv = tmp_path / "venv"
        (self.venv / "bin").mkdir(parents=True)
        self.python_argv = tmp_path / "python.argv"
        self.ssh_argv = tmp_path / "ssh.argv"
        self.ssh_calls_file = tmp_path / "ssh.calls"
        self.check_argv = tmp_path / "check.argv"
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self._write_python(jax_ok=True)
        self._write_ssh(stdout="")
        self.server_script = tmp_path / "run_hil_server.sh"
        self.write_server_check(stdout="", returncode=3)
        # A listening socket standing in for Terminal 1's tunnel.
        self.tunnel = socket.socket()
        self.tunnel.bind(("127.0.0.1", 0))
        self.tunnel.listen(4)
        self.tunnel_port = int(self.tunnel.getsockname()[1])
        self.proxy_port = _free_port()

    def close(self) -> None:
        self.tunnel.close()

    # -- stubs ---------------------------------------------------------- #
    def _write_python(self, *, jax_ok: bool) -> None:
        path = self.venv / "bin" / "python"
        path.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "-c" ]; then\n'
            + ("  echo 0.5.3-stub\n  exit 0\n" if jax_ok else "  echo 'ModuleNotFoundError: jax' >&2\n  exit 1\n")
            + "fi\n"
            f"printf '%s\\0' \"$@\" >{str(self.python_argv)!r}\n"
            "exit 0\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def break_jax(self) -> None:
        self._write_python(jax_ok=False)

    def _write_ssh(self, *, stdout: str) -> None:
        out = self.tmp / "ssh.stdout"
        out.write_text(stdout, encoding="utf-8")
        path = self.bin / "ssh"
        path.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >>{str(self.ssh_argv)!r}\n"
            f"echo call >>{str(self.ssh_calls_file)!r}\n"
            f"cat {str(out)!r}\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def write_ssh(self, *, stdout: str) -> None:
        self._write_ssh(stdout=stdout)

    def write_server_check(self, *, stdout: str, returncode: int) -> None:
        out = self.tmp / "check.stdout"
        out.write_text(stdout, encoding="utf-8")
        self.server_script.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >>{str(self.check_argv)!r}\n"
            f"cat {str(out)!r}\n"
            f"exit {returncode}\n",
            encoding="utf-8",
        )
        self.server_script.chmod(0o755)

    # -- driving --------------------------------------------------------- #
    def run(self, *args: str, **overrides: str) -> subprocess.CompletedProcess:
        env = _clean_env(
            LOCAL_POLICY_VENV=str(self.venv),
            HIL_LOCAL_POLICY_PORT=str(self.proxy_port),
            HIL_LOCAL_POLICY_REMOTE_TARGET=f"127.0.0.1:{self.tunnel_port}",
            HIL_SERVER_SCRIPT=str(self.server_script),
            HIL_SSH_HOST="stub-host",
            HIL_REMOTE_DATA_ROOT="/stub/hil-serl-data",
        )
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        env.update(overrides)
        return subprocess.run(
            ["bash", str(_PROXY_SH), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_WS),
        )

    # -- reading --------------------------------------------------------- #
    def proxy_argv(self) -> list[str]:
        assert self.python_argv.exists(), "the wrapper never ran the entrypoint"
        return [
            token
            for token in self.python_argv.read_bytes().decode().split("\0")
            if token
        ]

    def ssh_calls(self) -> int:
        if not self.ssh_calls_file.exists():
            return 0
        return len(self.ssh_calls_file.read_text().split())


@pytest.fixture()
def rig(tmp_path):
    made = _ProxyRig(tmp_path)
    try:
        yield made
    finally:
        made.close()


def test_proxy_wrapper_help_needs_no_venv_and_no_network(rig):
    result = rig.run("--help", LOCAL_POLICY_VENV="/nonexistent")
    assert result.returncode == 0, result.stderr
    assert "run_hil_local_policy.sh" in result.stdout
    assert not rig.ssh_argv.exists()


def test_missing_venv_is_refused_and_points_at_the_setup_script(rig):
    result = rig.run(LOCAL_POLICY_VENV=str(rig.tmp / "absent"))
    assert result.returncode == 2
    assert "setup_local_policy_venv.sh" in result.stderr
    assert not rig.python_argv.exists()
    assert not rig.ssh_argv.exists()


def test_a_venv_whose_jax_does_not_import_is_refused(rig):
    rig.break_jax()
    result = rig.run(HIL_PARAMS_REMOTE_DIR="/srv/run/params_live")
    assert result.returncode == 2
    assert "jax" in result.stderr
    assert "setup_local_policy_venv.sh" in result.stderr
    assert not rig.python_argv.exists()


def test_a_closed_tunnel_port_is_refused_with_the_right_terminal(rig):
    closed = _free_port()
    result = rig.run(
        HIL_LOCAL_POLICY_REMOTE_TARGET=f"127.0.0.1:{closed}",
        HIL_PARAMS_REMOTE_DIR="/srv/run/params_live",
    )
    assert result.returncode == 2
    assert "run_hil_server.sh" in result.stderr
    assert not rig.python_argv.exists()


def test_an_occupied_proxy_port_is_refused(rig):
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        result = rig.run(
            HIL_LOCAL_POLICY_PORT=str(holder.getsockname()[1]),
            HIL_PARAMS_REMOTE_DIR="/srv/run/params_live",
        )
    finally:
        holder.close()
    assert result.returncode == 2
    assert "run_local_policy_proxy.py" in result.stderr
    assert not rig.python_argv.exists()


def test_an_explicit_params_dir_resolves_without_touching_ssh(rig):
    result = rig.run(HIL_PARAMS_REMOTE_DIR="/srv/run/params_live/")
    assert result.returncode == 0, result.stderr + result.stdout
    argv = rig.proxy_argv()
    assert argv[argv.index("--params-remote-dir") + 1] == "/srv/run/params_live"
    assert rig.ssh_calls() == 0, "an explicit params dir still shelled out to ssh"
    assert not rig.check_argv.exists(), "an explicit params dir still ran --check"


def test_an_explicit_run_root_resolves_without_touching_ssh(rig):
    result = rig.run(HIL_PARAMS_RUN_ROOT="/srv/runs/lineage")
    assert result.returncode == 0, result.stderr + result.stdout
    argv = rig.proxy_argv()
    assert argv[argv.index("--params-remote-dir") + 1] == "/srv/runs/lineage/params_live"
    assert rig.ssh_calls() == 0
    assert not rig.check_argv.exists()


def test_the_run_root_comes_from_run_hil_server_check_when_not_given(rig):
    """The authoritative source: the live learner's own argv, via --check."""

    rig.write_server_check(
        stdout=(
            "HIL_SERVER_PID=4242\n"
            "HIL_SERVER_RUN_ROOT=/home/junhyeong/hil-serl-data/runs/cube_in_cup_real_1\n"
            "HIL_SERVER_RESULT=healthy\n"
            "CHECK PASS: one exact production learner is healthy; no tunnel was opened.\n"
        ),
        returncode=0,
    )
    result = rig.run()
    assert result.returncode == 0, result.stderr + result.stdout
    argv = rig.proxy_argv()
    assert argv[argv.index("--params-remote-dir") + 1] == (
        "/home/junhyeong/hil-serl-data/runs/cube_in_cup_real_1/params_live"
    )
    check = rig.check_argv.read_bytes().decode().split("\0")
    assert "--check" in check
    assert rig.ssh_calls() == 0, "the authoritative source was used AND the fallback ran"
    assert "run_hil_server.sh --check" in result.stdout


def test_the_ssh_listing_is_only_a_fallback_and_says_so(rig):
    rig.write_server_check(stdout="No production learner is running.\n", returncode=3)
    rig.write_ssh(stdout="/stub/hil-serl-data/runs/newest/params_live\n")
    result = rig.run()
    assert result.returncode == 0, result.stderr + result.stdout
    argv = rig.proxy_argv()
    assert argv[argv.index("--params-remote-dir") + 1] == (
        "/stub/hil-serl-data/runs/newest/params_live"
    )
    assert rig.ssh_calls() >= 1
    ssh_tokens = rig.ssh_argv.read_bytes().decode().split("\0")
    assert "stub-host" in ssh_tokens
    listing = next(token for token in ssh_tokens if token.startswith("ls -td"))
    assert "/stub/hil-serl-data" in listing and "runs/*/params_live" in listing
    # The operator must be told the fallback cannot prove the lineage.
    assert "fallback" in result.stdout


def test_no_params_directory_anywhere_is_a_refusal_not_a_guess(rig):
    rig.write_server_check(stdout="", returncode=3)
    rig.write_ssh(stdout="")
    result = rig.run()
    assert result.returncode == 2
    assert "HIL_PARAMS_RUN_ROOT" in result.stderr
    assert not rig.python_argv.exists()


def test_the_entrypoint_is_run_from_the_local_policy_venv_with_the_pinned_flags(rig):
    result = rig.run(
        "--status-interval-s",
        "0",
        HIL_PARAMS_REMOTE_DIR="/srv/run/params_live",
        HIL_PARAMS_POLL_S="3.5",
        HIL_LOCAL_POLICY_DEVICE="cpu",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    argv = rig.proxy_argv()
    assert argv[0].endswith("/venv/bin/python") is False  # argv[0] is the script
    assert argv[0] == str(
        _ROOT / "serl_ur_infra" / "scripts" / "run_local_policy_proxy.py"
    )
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == str(rig.proxy_port)
    assert argv[argv.index("--remote-target") + 1] == f"127.0.0.1:{rig.tunnel_port}"
    assert argv[argv.index("--params-fetch") + 1] == "ssh"
    assert argv[argv.index("--poll-interval-s") + 1] == "3.5"
    assert argv[argv.index("--device") + 1] == "cpu"
    # Unrecognised arguments pass straight through to the entrypoint.
    assert argv[-2:] == ["--status-interval-s", "0"]


def test_the_wrapper_forwards_hil_latency_profile_by_inheritance(rig):
    """It is inherited, not re-spelled -- but it must actually arrive."""

    marker = rig.tmp / "child.env"
    path = rig.venv / "bin" / "python"
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then echo 0.5.3-stub; exit 0; fi\n'
        f"printf '%s' \"${{HIL_LATENCY_PROFILE:-<unset>}}\" >{str(marker)!r}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    result = rig.run(
        HIL_PARAMS_REMOTE_DIR="/srv/run/params_live", HIL_LATENCY_PROFILE="1"
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert marker.read_text() == "1"


@pytest.mark.parametrize("bogus", ["0", "70000", "abc", "50253x"])
def test_a_malformed_proxy_port_is_refused(rig, bogus):
    result = rig.run(HIL_LOCAL_POLICY_PORT=bogus)
    assert result.returncode == 2
    assert "HIL_LOCAL_POLICY_PORT" in result.stderr


@pytest.mark.parametrize("bogus", ["127.0.0.1", "127.0.0.1:", ":50153", "host:port"])
def test_a_malformed_remote_target_is_refused(rig, bogus):
    result = rig.run(HIL_LOCAL_POLICY_REMOTE_TARGET=bogus)
    assert result.returncode == 2
    assert "HIL_LOCAL_POLICY_REMOTE_TARGET" in result.stderr


def test_a_malformed_params_fetch_is_refused(rig):
    result = rig.run(HIL_PARAMS_FETCH="scp")
    assert result.returncode == 2
    assert "HIL_PARAMS_FETCH" in result.stderr


def test_local_fetch_without_a_directory_is_refused_rather_than_ssh_resolved(rig):
    result = rig.run(HIL_PARAMS_FETCH="local")
    assert result.returncode == 2
    assert rig.ssh_calls() == 0
    assert not rig.check_argv.exists()


# ---------------------------------------------------------------------------
# 5. Session wiring facts that are cheaper to assert than to run
# ---------------------------------------------------------------------------


def test_the_session_owns_the_proxy_lifecycle_like_the_cameras():
    text = _SESSION_SH.read_text(encoding="utf-8")
    # Started as its own process group, verified as a session leader, and torn
    # down by the same helper the cameras and the GUI use.
    assert 'setsid "$LOCAL_POLICY_SCRIPT"' in text
    assert 'verify_session_leader "$LOCAL_POLICY_PID" "local policy proxy"' in text
    assert (
        'stop_child_group "$LOCAL_POLICY_PID" "local policy proxy" '
        '"$LOCAL_POLICY_DRAIN_GRACE_S"' in text
    )


def test_the_proxy_is_started_before_the_cameras_and_gated_after_them():
    """Order is the point: warm-up overlaps the RealSense bring-up."""

    lines = _SESSION_SH.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if 'setsid "$LOCAL_POLICY_SCRIPT"' in line
    )
    cameras = next(
        index for index, line in enumerate(lines) if 'setsid "$CAMERA_SCRIPT"' in line
    )
    gate = next(
        index
        for index, line in enumerate(lines)
        if "hil_wait_for_local_policy" in line and "if !" in line
    )
    camera_ready = next(
        index for index, line in enumerate(lines) if "camera launcher READY" in line
    )
    assert start < cameras < camera_ready < gate


def test_the_readiness_gate_watches_the_entrypoints_own_event_names():
    """A renamed event must break here, not silently hang for 420 s."""

    session = _SESSION_SH.read_text(encoding="utf-8")
    entrypoint = (
        _ROOT / "serl_ur_infra" / "scripts" / "run_local_policy_proxy.py"
    ).read_text(encoding="utf-8")
    for event in ("local_policy_proxy_ready", "local_policy_proxy_startup_failed"):
        assert f'"event":"{event}"' in session
        assert re.search(rf'_emit\(\s*\n?\s*"{event}"', entrypoint), event


def _run_readiness_gate(
    tmp_path: Path, log_text: str, *, timeout_s: int = 5, alive: bool = True
) -> subprocess.CompletedProcess:
    """Run the session script's own gate function against a canned log."""

    log = tmp_path / "local_policy.log"
    log.write_text(log_text, encoding="utf-8")
    harness = tmp_path / "gate.sh"
    body = _block(_SESSION_SH, "hil_wait_for_local_policy() {", "}")
    harness.write_text(
        "set -Eeuo pipefail\n"
        f"LOCAL_POLICY_LOG={str(log)!r}\n"
        f"LOCAL_POLICY_READY_TIMEOUT_S={timeout_s}\n"
        # $$ is this harness: alive.  A never-allocated pid stands in for a
        # proxy that already exited.
        + ("LOCAL_POLICY_PID=$$\n" if alive else "LOCAL_POLICY_PID=999999\n")
        + f"{body}\n"
        "hil_wait_for_local_policy\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )


def test_the_readiness_gate_passes_on_the_real_emitted_line(tmp_path):
    import json

    line = json.dumps(
        {
            "event": "local_policy_proxy_ready",
            "host": "127.0.0.1",
            "port": 50253,
            "alive": True,
            "ready": True,
            "detail": "",
            "model_id": LOCAL_MODEL_ID,
            "params_version": 0,
            "remote_target": "127.0.0.1:50153",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    result = _run_readiness_gate(tmp_path, f"noise\n{line}\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_readiness_gate_fails_fast_on_startup_failed(tmp_path):
    import json

    line = json.dumps(
        {"event": "local_policy_proxy_startup_failed", "detail": "no blob"},
        sort_keys=True,
        separators=(",", ":"),
    )
    # timeout_s is large; a pass here means it did NOT wait it out.
    result = _run_readiness_gate(tmp_path, line + "\n", timeout_s=600)
    assert result.returncode == 1
    assert "startup_failed" in result.stdout


def test_the_readiness_gate_fails_when_the_proxy_is_gone(tmp_path):
    result = _run_readiness_gate(tmp_path, "starting up\n", timeout_s=600, alive=False)
    assert result.returncode == 1
    assert "READY 전에 종료" in result.stdout


def test_the_readiness_gate_gives_up_rather_than_hanging_forever(tmp_path):
    result = _run_readiness_gate(tmp_path, "warming up\n", timeout_s=2)
    assert result.returncode == 1
    assert "READY를 확인하지 못했다" in result.stdout


def test_the_drain_grace_is_not_shorter_than_the_entrypoints_drain_timeout():
    """Kill the proxy mid-drain and queued transitions are lost for good."""

    session = _SESSION_SH.read_text(encoding="utf-8")
    grace = int(
        re.search(r'LOCAL_POLICY_DRAIN_GRACE_S="\$\{HIL_LOCAL_POLICY_DRAIN_GRACE_S:-(\d+)\}"', session).group(1)
    )
    entrypoint = (
        _ROOT / "serl_ur_infra" / "scripts" / "run_local_policy_proxy.py"
    ).read_text(encoding="utf-8")
    drain = float(
        re.search(
            r'"--drain-timeout-s",\s*\n\s*type=float,\s*\n\s*default=([\d.]+),',
            entrypoint,
        ).group(1)
    )
    assert grace >= drain
