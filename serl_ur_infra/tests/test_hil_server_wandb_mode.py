"""The W&B transport is operator-selectable and its guard cannot desync.

`run_hil_server.sh` both LAUNCHES the production learner and, on every later
invocation, RE-VALIDATES a running learner against the exact command it would
have launched.  Those are two descriptions of one thing, and this repo has
already paid for a guard that kept passing while the thing it guarded moved
(the kanu `origin`-points-at-a-local-path incident).  So these tests do not
assert on hardcoded strings: they EXTRACT the launch command and the contract
guard out of the shell script, expand both from one variable table, and check
that the guard accepts the process the launch command would have produced.
Hardcode `offline` back into either side and the online cases fail.

Nothing here starts a learner, opens port 50053, or talks to any host.  The
"learner" is a sleeping script that only has to own the right /proc entry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SH = _ROOT / "ros2_ur_ws" / "run_hil_server.sh"

_MODES = ("offline", "online", "disabled")


# ---------------------------------------------------------------------------
# Extraction helpers: read the launcher rather than restate it.
# ---------------------------------------------------------------------------


def _lines() -> list[str]:
    return _SERVER_SH.read_text(encoding="utf-8").splitlines()


def _heredoc(function_name: str) -> tuple[str, list[str]]:
    """Return (python source, ordered $VAR names passed as its argv)."""

    lines = _lines()
    start = lines.index(f"{function_name}() {{")
    opener = next(
        index
        for index in range(start, len(lines))
        if lines[index].rstrip().endswith("<<'PY'")
    )
    terminator = next(
        index for index in range(opener + 1, len(lines)) if lines[index] == "PY"
    )
    invocation = "\n".join(lines[start : opener + 1])
    # Everything after the `-` that makes Python read the program from stdin.
    invocation = invocation.split(" - ", 1)[1]
    argv_names = re.findall(r'"\$(\w+)"', invocation)
    return "\n".join(lines[opener + 1 : terminator]), argv_names


def _launch_block() -> str:
    lines = _lines()
    start = lines.index("nohup env \\")
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].rstrip().endswith("9>&- &")
    )
    return "\n".join(lines[start : end + 1])


def _referenced_variables(block: str) -> set[str]:
    return set(re.findall(r'\$\{?([A-Za-z_][A-Za-z_0-9]*)\}?', block))


# ---------------------------------------------------------------------------
# A rig shaped exactly like the remote script's own variables.
# ---------------------------------------------------------------------------


def _rig(tmp_path: Path, *, wandb_mode: str) -> dict[str, str]:
    repo = tmp_path / "repo"
    (repo / "serl_ur_infra" / "scripts").mkdir(parents=True)
    (repo / "third_party" / "hil-serl").mkdir(parents=True)
    run_id = "cube_in_cup_real_20260731_000000"
    run_root = tmp_path / "runs" / run_id
    for name in ("logs", "wandb", "assets"):
        (run_root / name).mkdir(parents=True)
    data_root = tmp_path / "data"
    return {
        "GPU_INDEX": "0",
        "RUN_ID": run_id,
        "REMOTE_REPO": str(repo),
        "REMOTE_PYTHON": sys.executable,
        "REMOTE_PORT": "50053",
        "CLASSIFIER": str(data_root / "classifier_ckpt" / "checkpoint_150"),
        "CLASSIFIER_SHA256": "512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d",
        "REAL_DEMO": str(data_root / "demos" / "demo.pkl"),
        "RESNET_SOURCE": str(
            repo / "third_party" / "hil-serl" / "examples" / "experiments"
            / "resnet10_params.pkl"
        ),
        "REWARD_THRESHOLD": "0.5",
        "REWARD_MODEL_ID": "cube-in-cup-all3-ckpt150+sidecar-v1",
        "WANDB_MODE": wandb_mode,
        "checkpoint_root": str(run_root / "checkpoints"),
        "jsonl_path": str(run_root / "logs" / "learner.jsonl"),
        "memory_path": str(run_root / "logs" / "memory-preflight.jsonl"),
        "wandb_dir": str(run_root / "wandb"),
        "resnet_cache": str(run_root / "assets" / "resnet10_params.pkl"),
        "stdout_path": str(run_root / "logs" / "stdout.log"),
    }


def _expand_launch_command(tmp_path: Path, rig: dict[str, str]) -> list[str]:
    """Run the launcher's own nohup block with `nohup` stubbed out."""

    block = _launch_block()
    missing = _referenced_variables(block) - set(rig)
    assert not missing, f"launch block gained unmodelled variables: {sorted(missing)}"
    capture = tmp_path / "argv.nul"
    assignments = "\n".join(f"{key}={value!r}" for key, value in rig.items())
    harness = tmp_path / "capture_launch.sh"
    harness.write_text(
        "set -euo pipefail\n"
        f"CAPTURE={str(capture)!r}\n"
        "nohup() { printf '%s\\0' \"$@\" >\"$CAPTURE\"; }\n"
        f"{assignments}\n"
        f"{block}\n"
        'wait "$!" || true\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["bash", str(harness)],
        check=True,
        capture_output=True,
        text=True,
    )
    tokens = capture.read_bytes().split(b"\0")
    return [token.decode() for token in tokens if token]


def _split_env_prefix(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    assert tokens[0] == "env", tokens[:1]
    environment: dict[str, str] = {}
    index = 1
    while "=" in tokens[index] and not tokens[index].startswith("-"):
        key, value = tokens[index].split("=", 1)
        environment[key] = value
        index += 1
    return environment, tokens[index:]


def _spawn_fake_learner(rig: dict[str, str], argv: list[str], environment: dict[str, str]):
    """Own a /proc entry with the launcher's exact argv, cwd and environment."""

    repo = Path(rig["REMOTE_REPO"])
    entrypoint = repo / "serl_ur_infra" / "scripts" / "run_rlpd_learner_server.py"
    entrypoint.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
    child_env = dict(os.environ)
    child_env.update(environment)
    return subprocess.Popen(
        argv,
        cwd=str(repo),
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def _run_contract_guard(pid: int, rig: dict[str, str]) -> subprocess.CompletedProcess:
    source, argv_names = _heredoc("validate_process_contract")
    assert argv_names[0] == "pid", argv_names
    values = [str(pid)]
    for name in argv_names[1:]:
        assert name in rig, f"contract guard reads unmodelled variable ${name}"
        values.append(rig[name])
    return subprocess.run(
        [sys.executable, "-c", source, *values],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# The coupling itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _MODES)
def test_contract_guard_accepts_the_command_the_launcher_builds(tmp_path, mode):
    rig = _rig(tmp_path, wandb_mode=mode)
    environment, argv = _split_env_prefix(_expand_launch_command(tmp_path, rig))
    assert "--wandb-mode" in argv
    assert argv[argv.index("--wandb-mode") + 1] == mode

    process = _spawn_fake_learner(rig, argv, environment)
    try:
        deadline = time.time() + 10.0
        while True:
            result = _run_contract_guard(process.pid, rig)
            if result.returncode == 0 or time.time() > deadline:
                break
            time.sleep(0.05)
        assert result.returncode == 0, result.stderr
        assert f"HIL_SERVER_PID={process.pid}" in result.stdout
    finally:
        process.kill()
        process.wait(timeout=10)


def test_contract_guard_rejects_a_learner_started_in_another_mode(tmp_path):
    """The guard is the reason a lineage's W&B mode is fixed at launch."""

    launched = _rig(tmp_path, wandb_mode="offline")
    environment, argv = _split_env_prefix(_expand_launch_command(tmp_path, launched))
    process = _spawn_fake_learner(launched, argv, environment)
    try:
        asked = dict(launched, WANDB_MODE="online")
        result = _run_contract_guard(process.pid, asked)
        assert result.returncode != 0
        assert "--wandb-mode" in result.stderr
        assert "HIL_WANDB_MODE" in result.stderr
    finally:
        process.kill()
        process.wait(timeout=10)


def test_neither_side_hardcodes_a_wandb_mode_literal():
    """Belt and braces for the parametrised tests above.

    A future edit that reintroduces a literal on either side would be caught by
    the online/disabled cases, but only if someone runs them.  This states the
    invariant directly so the intent survives in the source.
    """

    block = _launch_block()
    assert '--wandb-mode "$WANDB_MODE" \\' in block
    for mode in _MODES:
        assert f"--wandb-mode {mode}" not in block

    guard, _ = _heredoc("validate_process_contract")
    assert re.search(r'exact\(\s*"--wandb-mode",\s*wandb_mode,', guard)
    for mode in _MODES:
        assert f'exact("--wandb-mode", "{mode}")' not in guard


# ---------------------------------------------------------------------------
# Laptop-side option handling.
# ---------------------------------------------------------------------------


def _run_launcher(tmp_path, *args, **environment) -> subprocess.CompletedProcess:
    """Run run_hil_server.sh with ssh shadowed so no host is ever contacted."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    ssh_log = tmp_path / "ssh.args"
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\0' \"$@\" >{str(ssh_log)!r}\n"
        "cat >/dev/null\n"
        "exit 3\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "HIL_WANDB_MODE"
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.update(environment)
    result = subprocess.run(
        ["bash", str(_SERVER_SH), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_ROOT),
    )
    result.ssh_log = ssh_log  # type: ignore[attr-defined]
    return result


def test_invalid_wandb_mode_dies_before_any_remote_contact(tmp_path):
    result = _run_launcher(tmp_path, "--check", HIL_WANDB_MODE="Online")
    assert result.returncode == 1
    assert "HIL_WANDB_MODE must be offline, online or disabled" in result.stderr
    assert not result.ssh_log.exists(), "a rejected mode still reached ssh"


@pytest.mark.parametrize("mode", _MODES)
def test_valid_wandb_mode_is_marshalled_to_the_learner_host(tmp_path, mode):
    result = _run_launcher(tmp_path, "--check", HIL_WANDB_MODE=mode)
    # The stub ssh reports "no learner", which --check maps to exit 3.
    assert result.returncode == 3, result.stderr
    tokens = [
        token.decode()
        for token in result.ssh_log.read_bytes().split(b"\0")
        if token
    ]
    positionals = tokens[tokens.index("--") + 1 :]
    assert positionals[-1] == mode
    assert len(positionals) == 14


def test_default_wandb_mode_is_still_offline(tmp_path):
    """Deliberate: online is opt-in until a W&B stall cannot stall the learner."""

    result = _run_launcher(tmp_path, "--check")
    assert result.returncode == 3, result.stderr
    tokens = [
        token.decode()
        for token in result.ssh_log.read_bytes().split(b"\0")
        if token
    ]
    assert tokens[-1] == "offline"


# ---------------------------------------------------------------------------
# Reporting the run URL: the whole point of online mode.
# ---------------------------------------------------------------------------


def _write_ready_evidence(run_root: Path, ready_extra: dict) -> None:
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    warmup = {
        "event": "learner_update_warmup_complete",
        "warmup_outer_steps": 3,
        "warmup_gradient_updates": 6,
        "learner_step": 0,
        "gradient_step": 0,
        "policy_version": 0,
        "production_state_advanced": False,
    }
    ready = {
        "event": "learner_process_ready",
        "jax_backend": "gpu",
        "demo_count": 2037,
        "synthetic_acceptance_demo_count": 0,
        "restored_checkpoint": None,
        "learner_step": 0,
        "gradient_step": 0,
        "policy_version": 0,
        **ready_extra,
    }
    (run_root / "logs" / "learner.jsonl").write_text(
        json.dumps(warmup) + "\n" + json.dumps(ready) + "\n",
        encoding="utf-8",
    )


def _run_ready_evidence(run_root: Path) -> subprocess.CompletedProcess:
    source, argv_names = _heredoc("validate_ready_evidence")
    assert argv_names == ["run_root"], argv_names
    return subprocess.run(
        [sys.executable, "-c", source, str(run_root)],
        capture_output=True,
        text=True,
    )


def test_ready_evidence_reports_the_live_wandb_url(tmp_path):
    url = "https://wandb.ai/junhyeong/hil-serl/runs/abc12345"
    _write_ready_evidence(
        tmp_path,
        {"wandb_mode": "online", "wandb_url": url, "wandb_run_path": "junhyeong/hil-serl/abc12345"},
    )
    result = _run_ready_evidence(tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"HIL_SERVER_WANDB_URL={url}" in result.stdout
    assert "HIL_SERVER_WANDB_MODE=online" in result.stdout


def test_ready_evidence_accepts_a_learner_that_predates_the_wandb_fields(tmp_path):
    """Reuse must not break for a lineage started before this feature existed."""

    _write_ready_evidence(tmp_path, {})
    result = _run_ready_evidence(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "HIL_SERVER_WANDB_URL=none" in result.stdout
    assert "HIL_SERVER_WANDB_MODE=unknown" in result.stdout


# ---------------------------------------------------------------------------
# The learner side of the same question: where does the URL come from at all.
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, **attributes):
        self._attributes = attributes
        self.finished = False

    def __getattr__(self, name):
        if name in self._attributes:
            value = self._attributes[name]
            if isinstance(value, Exception):
                raise value
            return value
        raise AttributeError(name)

    def log(self, record, step):  # pragma: no cover - not exercised here
        del record, step

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self, run):
        self.run = run
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def _logger(tmp_path, run, **overrides):
    from ur_env.learner import JsonlWandbLogger

    options = dict(
        wandb_dir=tmp_path / "wandb",
        wandb_module=_FakeWandb(run) if run is not None else None,
    )
    options.update(overrides)
    return JsonlWandbLogger(tmp_path / "learner.jsonl", **options)


def test_online_run_metadata_is_addressable(tmp_path):
    run = _FakeRun(
        id="abc12345",
        name="cube_in_cup_real-hil-5000",
        entity="junhyeong",
        project="hil-serl",
        url="https://wandb.ai/junhyeong/hil-serl/runs/abc12345",
    )
    logger = _logger(tmp_path, run, wandb_mode="online")
    try:
        metadata = logger.wandb_metadata
    finally:
        logger.close()
    assert metadata["mode"] == "online"
    assert metadata["url"] == "https://wandb.ai/junhyeong/hil-serl/runs/abc12345"
    assert metadata["run_path"] == "junhyeong/hil-serl/abc12345"
    assert metadata["dir"] == str((tmp_path / "wandb").resolve())


def test_offline_run_has_no_url_to_report(tmp_path):
    run = _FakeRun(id="abc12345", entity="junhyeong", project="hil-serl", url=None)
    logger = _logger(tmp_path, run, wandb_mode="offline")
    try:
        metadata = logger.wandb_metadata
    finally:
        logger.close()
    assert metadata["mode"] == "offline"
    assert metadata["url"] is None
    assert metadata["run_path"] == "junhyeong/hil-serl/abc12345"


def test_disabled_mode_reports_only_the_mode(tmp_path):
    logger = _logger(tmp_path, None, wandb_mode="disabled", enable_wandb=False)
    try:
        metadata = logger.wandb_metadata
    finally:
        logger.close()
    assert metadata["mode"] == "disabled"
    assert set(metadata) - {"mode"} == {
        "dir",
        "id",
        "name",
        "entity",
        "project",
        "url",
        "run_path",
    }
    assert all(value is None for key, value in metadata.items() if key != "mode")


def test_unreadable_run_identity_does_not_take_the_learner_down(tmp_path):
    """Reporting the URL is diagnostics; it must never become a startup fault."""

    run = _FakeRun(
        id="abc12345",
        entity="junhyeong",
        project="hil-serl",
        url=RuntimeError("run url is not available yet"),
    )
    logger = _logger(tmp_path, run, wandb_mode="online")
    try:
        metadata = logger.wandb_metadata
    finally:
        logger.close()
    assert metadata["url"] is None
    assert metadata["run_path"] == "junhyeong/hil-serl/abc12345"


def test_metadata_is_a_copy(tmp_path):
    run = _FakeRun(id="abc12345", entity="e", project="p", url="https://example/x")
    logger = _logger(tmp_path, run, wandb_mode="online")
    try:
        logger.wandb_metadata["url"] = "tampered"
        assert logger.wandb_metadata["url"] == "https://example/x"
    finally:
        logger.close()
