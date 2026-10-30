"""The kanu fallback is an ENV override, and "only env" has to be provable.

TEST-BRANCH-ONLY (``test/kanu-learner-fallback``), alongside
``ros2_ur_ws/run_hil_server_kanu.sh``.  junhyeong_ai's GPU is occupied, so the
learner temporarily runs on the older ``kanu`` host.  The whole point of doing
that with a six-line wrapper instead of a second launcher is that NO guard
moves: the process contract, the checkpoint SHA pin, the GPU-occupancy refusal,
the cross-host HEAD identity check and the run lock all stay inside
``run_hil_server.sh``.  That claim is only worth anything if the wrapper really
does nothing but export five variables and hand argv over untouched, so these
tests expand the real wrapper against a stub launcher and diff BOTH sides:
the argv it forwards, and the environment it changed.

The second half proves the Terminal-3 path, which is where a host override can
silently half-apply: ``run_hil_local_policy.sh`` asks
``run_hil_server.sh --check`` which run root is live, and that probe must ask
kanu -- otherwise the proxy would hunt junhyeong_ai's run-root path on kanu and
find nothing.  Pointing ``HIL_SERVER_SCRIPT`` at this wrapper is what makes the
probe carry the kanu profile, so the rig here drives the real
``run_hil_local_policy.sh`` through the real wrapper with only the launcher
itself stubbed.

Nothing here contacts kanu, opens 50053/50253, starts a learner or a proxy, or
touches the robot.  ``ssh`` is shadowed by a stub and the "venv python" is a
shell script that records its argv and environment.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_INFRA = _ROOT / "serl_ur_infra"
_WS = _ROOT / "ros2_ur_ws"
_KANU_SH = _WS / "run_hil_server_kanu.sh"
_SERVER_SH = _WS / "run_hil_server.sh"
_PROXY_SH = _WS / "run_hil_local_policy.sh"
_SESSION_SH = _WS / "run_hil_session.sh"
_ACTOR_SH = _WS / "run_hil_actor.sh"

sys.path.insert(0, str(_INFRA))

from ur_env.local_policy.params_sync import (  # noqa: E402
    DEFAULT_SSH_HOST,
    SSH_HOST_ENV_VAR,
    SshParamsFetcher,
)

#: The kanu profile, restated once here so a silent edit of the wrapper is a
#: test failure rather than a different server.
KANU_PROFILE = {
    "HIL_SSH_HOST": "kanu",
    "HIL_GPU_INDEX": "1",
    "HIL_REMOTE_REPO": "/home/junhyeong/gello_software_hil_current",
    "HIL_REMOTE_PYTHON": "/home/junhyeong/miniconda3/envs/il/bin/python",
    "HIL_REMOTE_DATA_ROOT": "/home/junhyeong/hil-serl-data",
}

#: Variables bash rewrites for its own bookkeeping on every invocation.  They
#: differ between "wrapper then exec" and "stub directly" for reasons that have
#: nothing to do with what the wrapper exports.
_SHELL_BOOKKEEPING = {"_", "SHLVL", "PWD", "OLDPWD", "BASH_EXECUTION_STRING"}


# ---------------------------------------------------------------------------
# Helpers
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_nul(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [token for token in path.read_bytes().decode().split("\0") if token]


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in path.read_bytes().decode().split("\0"):
        if not entry:
            continue
        name, _, value = entry.partition("=")
        out[name] = value
    return out


class _WrapperRig:
    """A copy of the real wrapper next to a launcher stub that records itself.

    The wrapper resolves its sibling through ``BASH_SOURCE``, so copying it
    into a scratch directory is enough to intercept the exec without editing
    the wrapper, adding an indirection hook to it, or running the real
    launcher (which would ssh to kanu).
    """

    def __init__(self, tmp_path: Path, *, exit_code: int = 7) -> None:
        self.dir = tmp_path / "ws"
        self.dir.mkdir(parents=True)
        self.wrapper = self.dir / _KANU_SH.name
        shutil.copy2(_KANU_SH, self.wrapper)
        self.argv = tmp_path / "server.argv"
        self.env = tmp_path / "server.env"
        self.stub = self.dir / "run_hil_server.sh"
        self.stub.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >{str(self.argv)!r}\n"
            f"env -0 >{str(self.env)!r}\n"
            f"exit {exit_code}\n",
            encoding="utf-8",
        )
        self.stub.chmod(0o755)

    def run(self, *args: str, **overrides: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.wrapper), *args],
            capture_output=True,
            text=True,
            env=_clean_env(**overrides),
            cwd=str(self.dir),
        )

    def run_stub_directly(self, *args: str, **overrides: str):
        return subprocess.run(
            ["bash", str(self.stub), *args],
            capture_output=True,
            text=True,
            env=_clean_env(**overrides),
            cwd=str(self.dir),
        )


@pytest.fixture()
def wrapper(tmp_path):
    return _WrapperRig(tmp_path)


# ---------------------------------------------------------------------------
# 0. The scripts still parse.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script", [_KANU_SH, _SERVER_SH, _PROXY_SH, _SESSION_SH, _ACTOR_SH]
)
def test_touched_scripts_are_syntactically_valid(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_wrapper_is_executable_and_says_it_is_test_branch_only() -> None:
    assert os.access(_KANU_SH, os.X_OK), f"{_KANU_SH} is not executable"
    text = _KANU_SH.read_text(encoding="utf-8")
    assert "TEST-BRANCH-ONLY" in text
    assert "test/kanu-learner-fallback" in text


# ---------------------------------------------------------------------------
# 1. Wrapper purity: env only, argv verbatim.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--check",),
        ("--new-lineage", "--gpu", "2", "--run-id", "cube_in_cup_real_20260810_0000"),
        ("--gpu=1",),
        # A value with a space and one with a glob character: the wrapper must
        # not word-split or expand what the operator typed.
        ("--run-id", "a b", "*"),
    ],
)
def test_argv_reaches_the_launcher_verbatim(wrapper, argv) -> None:
    result = wrapper.run(*argv)
    assert result.returncode == 7, result.stderr
    assert _read_nul(wrapper.argv) == list(argv)


def test_the_wrapper_changes_exactly_the_five_kanu_variables(wrapper) -> None:
    wrapper.run("--check")
    through_wrapper = _read_env(wrapper.env)
    wrapper.run_stub_directly("--check")
    direct = _read_env(wrapper.env)

    changed = {
        name: value
        for name, value in through_wrapper.items()
        if name not in _SHELL_BOOKKEEPING and direct.get(name) != value
    }
    removed = {
        name
        for name, value in direct.items()
        if name not in _SHELL_BOOKKEEPING and name not in through_wrapper
    }
    assert changed == KANU_PROFILE
    assert removed == set()


def test_the_launcher_exit_code_is_the_wrappers_exit_code(tmp_path) -> None:
    for code in (0, 2, 3, 42):
        rig = _WrapperRig(tmp_path / f"code{code}", exit_code=code)
        assert rig.run("--check").returncode == code


@pytest.mark.parametrize("name, value", sorted(KANU_PROFILE.items()))
def test_an_explicit_override_beats_the_wrapper_default(
    wrapper, name: str, value: str
) -> None:
    override = "2" if name == "HIL_GPU_INDEX" else f"/operator/override/{name}"
    if name == "HIL_SSH_HOST":
        override = "operator-host"
    wrapper.run("--check", **{name: override})
    seen = _read_env(wrapper.env)
    assert seen[name] == override
    for other, expected in KANU_PROFILE.items():
        if other != name:
            assert seen[other] == expected


def test_the_wrapper_does_not_follow_hil_server_script(tmp_path) -> None:
    """T3 sets HIL_SERVER_SCRIPT to this wrapper; it must not exec itself."""

    rig = _WrapperRig(tmp_path)
    bomb = tmp_path / "bomb.sh"
    bomb.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    bomb.chmod(0o755)
    result = rig.run("--check", HIL_SERVER_SCRIPT=str(bomb))
    assert result.returncode == 7, result.stderr
    assert _read_nul(rig.argv) == ["--check"]


def test_the_wrapper_carries_no_logic_of_its_own() -> None:
    """Every guard stays in run_hil_server.sh; this file only exports and execs."""

    body = [
        line.strip()
        for line in _KANU_SH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert body[-1] == 'exec "$SCRIPT_DIR/run_hil_server.sh" "$@"'
    exports = [line for line in body if line.startswith("export ")]
    assert len(exports) == len(KANU_PROFILE)
    for name, value in KANU_PROFILE.items():
        assert f'export {name}="${{{name}:-{value}}}"' in exports
    # No ssh, no nvidia-smi, no process inspection, no conditionals.
    for forbidden in ("ssh ", "nvidia-smi", "pgrep", "if ", "while ", "case "):
        assert not any(line.startswith(forbidden) for line in body), forbidden


# ---------------------------------------------------------------------------
# 2. The values the wrapper injects survive run_hil_server.sh's own validators.
#    Extracted from the launcher rather than restated, so a tightened guard
#    fails here instead of failing on kanu at 2 a.m.
# ---------------------------------------------------------------------------


def _data_root_pattern() -> re.Pattern[str]:
    for line in _SERVER_SH.read_text(encoding="utf-8").splitlines():
        found = re.search(r'\[\[ "\$REMOTE_DATA_ROOT" =~ (\S+) \]\]', line)
        if found:
            return re.compile(found.group(1))
    raise AssertionError("run_hil_server.sh no longer validates REMOTE_DATA_ROOT")


def test_the_injected_data_root_passes_the_launchers_own_validator() -> None:
    root = KANU_PROFILE["HIL_REMOTE_DATA_ROOT"]
    assert _data_root_pattern().match(root)
    assert ".." not in root


def test_the_injected_gpu_index_passes_the_launchers_own_validator() -> None:
    assert re.fullmatch(r"[0-9]+", KANU_PROFILE["HIL_GPU_INDEX"])


@pytest.mark.parametrize(
    "name", ["HIL_REMOTE_REPO", "HIL_REMOTE_PYTHON", "HIL_REMOTE_DATA_ROOT"]
)
def test_the_injected_paths_are_absolute_and_shell_safe(name: str) -> None:
    value = KANU_PROFILE[name]
    assert value.startswith("/")
    assert not re.search(r"[\s'\"$`\\;&|<>()*?]", value)


# ---------------------------------------------------------------------------
# 3. Terminal 3: the run-root probe has to ask kanu, and the proxy has to fetch
#    from kanu.
# ---------------------------------------------------------------------------


class _T3Rig:
    """run_hil_local_policy.sh driven with the real wrapper as HIL_SERVER_SCRIPT."""

    RUN_ROOT = "/home/junhyeong/hil-serl-data/runs/cube_in_cup_real_20260810_120000"
    FALLBACK_DIR = "/home/junhyeong/hil-serl-data/runs/older_run/params_live"

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.venv = tmp_path / "venv"
        (self.venv / "bin").mkdir(parents=True)
        self.python_argv = tmp_path / "python.argv"
        self.python_env = tmp_path / "python.env"
        path = self.venv / "bin" / "python"
        path.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "-c" ]; then echo 0.5.3-stub; exit 0; fi\n'
            f"printf '%s\\0' \"$@\" >{str(self.python_argv)!r}\n"
            f"env -0 >{str(self.python_env)!r}\n"
            "exit 0\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.ssh_calls = tmp_path / "ssh.calls"
        ssh = self.bin / "ssh"
        ssh.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >>{str(self.ssh_calls)!r}\n"
            f"echo {self.FALLBACK_DIR!r}\n",
            encoding="utf-8",
        )
        ssh.chmod(0o755)

        # The wrapper, copied next to a launcher stub that answers --check the
        # way a live kanu learner would and records the env it was asked in.
        self.wrap = _WrapperRig(tmp_path, exit_code=0)
        self.wrap.stub.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >{str(self.wrap.argv)!r}\n"
            f"env -0 >{str(self.wrap.env)!r}\n"
            f"echo 'HIL_SERVER_RUN_ROOT={self.RUN_ROOT}'\n",
            encoding="utf-8",
        )
        self.wrap.stub.chmod(0o755)

        self.tunnel = socket.socket()
        self.tunnel.bind(("127.0.0.1", 0))
        self.tunnel.listen(4)
        self.tunnel_port = int(self.tunnel.getsockname()[1])
        self.proxy_port = _free_port()

    def close(self) -> None:
        self.tunnel.close()

    def run(self, **overrides: str) -> subprocess.CompletedProcess:
        env = _clean_env(
            LOCAL_POLICY_VENV=str(self.venv),
            HIL_LOCAL_POLICY_PORT=str(self.proxy_port),
            HIL_LOCAL_POLICY_REMOTE_TARGET=f"127.0.0.1:{self.tunnel_port}",
            HIL_SERVER_SCRIPT=str(self.wrap.wrapper),
        )
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        env.update(overrides)
        return subprocess.run(
            ["bash", str(_PROXY_SH)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_WS),
        )


@pytest.fixture()
def t3(tmp_path):
    made = _T3Rig(tmp_path)
    try:
        yield made
    finally:
        made.close()


def test_the_run_root_probe_runs_with_the_kanu_profile(t3) -> None:
    result = t3.run(HIL_SSH_HOST="kanu")
    assert result.returncode == 0, result.stderr
    assert _read_nul(t3.wrap.argv) == ["--check"]
    probe_env = _read_env(t3.wrap.env)
    for name, value in KANU_PROFILE.items():
        assert probe_env[name] == value, name


def test_the_proxy_is_pointed_at_the_run_root_the_probe_reported(t3) -> None:
    result = t3.run(HIL_SSH_HOST="kanu")
    assert result.returncode == 0, result.stderr
    argv = _read_nul(t3.python_argv)
    assert "--params-remote-dir" in argv
    assert argv[argv.index("--params-remote-dir") + 1] == f"{t3.RUN_ROOT}/params_live"
    # The authoritative probe answered, so no ssh guessing happened.
    assert not t3.ssh_calls.exists()


def test_hil_ssh_host_reaches_the_proxy_process(t3) -> None:
    """The proxy passes no --params-ssh-host, so SshParamsFetcher reads env."""

    result = t3.run(HIL_SSH_HOST="kanu")
    assert result.returncode == 0, result.stderr
    assert "--params-ssh-host" not in _read_nul(t3.python_argv)
    assert _read_env(t3.python_env)[SSH_HOST_ENV_VAR] == "kanu"


def test_the_ssh_fallback_also_follows_hil_ssh_host(t3) -> None:
    """When the probe reports nothing, the ls fallback must still ask kanu."""

    t3.wrap.stub.write_text(
        "#!/usr/bin/env bash\necho 'no learner'\nexit 3\n", encoding="utf-8"
    )
    t3.wrap.stub.chmod(0o755)
    result = t3.run(HIL_SSH_HOST="kanu")
    assert result.returncode == 0, result.stderr
    ssh_argv = _read_nul(t3.ssh_calls)
    assert "kanu" in ssh_argv
    assert any(KANU_PROFILE["HIL_REMOTE_DATA_ROOT"] in token for token in ssh_argv)
    argv = _read_nul(t3.python_argv)
    assert argv[argv.index("--params-remote-dir") + 1] == t3.FALLBACK_DIR


def test_the_session_hands_its_environment_to_the_proxy_wrapper() -> None:
    """T3's `HIL_SSH_HOST=kanu ...` prefix reaches the proxy by inheritance."""

    text = _SESSION_SH.read_text(encoding="utf-8")
    spawn = [line for line in text.splitlines() if 'setsid "$LOCAL_POLICY_SCRIPT"' in line]
    assert len(spawn) == 1, spawn
    assert "env " not in spawn[0]
    assert not re.search(r"\benv\s+-[iu]\b", text)
    assert not re.search(r"^\s*unset\s+HIL_", text, re.MULTILINE)


# ---------------------------------------------------------------------------
# 4. The last link: the fetcher itself.
# ---------------------------------------------------------------------------


def test_ssh_params_fetcher_follows_hil_ssh_host() -> None:
    assert SshParamsFetcher(env={SSH_HOST_ENV_VAR: "kanu"}).host == "kanu"
    assert SshParamsFetcher(env={}).host == DEFAULT_SSH_HOST


def test_an_explicit_fetcher_host_still_wins_over_the_env() -> None:
    fetcher = SshParamsFetcher("explicit-host", env={SSH_HOST_ENV_VAR: "kanu"})
    assert fetcher.host == "explicit-host"


def test_the_actor_tunnel_hint_names_the_host_actually_in_use() -> None:
    """Only a message, but it told the operator to tunnel to the wrong host."""

    text = _ACTOR_SH.read_text(encoding="utf-8")
    assert "${HIL_SSH_HOST:-junhyeong_ai}" in text
    assert not re.search(r"50053 junhyeong_ai", text)
