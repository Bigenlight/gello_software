from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "sim_collect" / "run_sim_collect.sh"

FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  exit 0
fi
case " $* " in
  *" -m sim_collect.sim_main "*) role=sim ;;
  *" -m sim_collect.capture "*) role=capture ;;
  *" -m sim_collect.gui "*) role=gui ;;
  *) role=unknown ;;
esac
process_group="$(ps -o pgid= -p $$ | tr -d ' ')"
line="start|$role|${MUJOCO_GL:-}|${SIM_COLLECT_OUTPUT_ROOT:-}|$$|$process_group"
for arg in "$@"; do
  line="$line|$arg"
done
printf '%s\n' "$line" >> "$FAKE_PY_LOG"
stop() {
  state=after_capture
  if [[ "$role" == capture ]]; then
    : > "$FAKE_CAPTURE_STOPPED"
  elif [[ ! -f "$FAKE_CAPTURE_STOPPED" ]]; then
    state=before_capture
  fi
  printf 'term|%s|%s\n' "$role" "$state" >> "$FAKE_PY_LOG"
  exit 0
}
trap stop TERM INT
while true; do
  sleep 0.05
done
"""


def _fake_env(tmp_path: Path, *, backend: str | None = None) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "fake env" / "bin"
    bin_dir.mkdir(parents=True)
    fake_python = bin_dir / "python"
    fake_python.write_text(FAKE_PYTHON)
    fake_python.chmod(0o755)
    log = tmp_path / "fake-python.log"
    env = dict(os.environ)
    env.update({
        "SIM_COLLECT_PY": str(fake_python),
        "FAKE_PY_LOG": str(log),
        "FAKE_CAPTURE_STOPPED": str(tmp_path / "capture-stopped"),
        "LOG_DIR": str(tmp_path / "logs"),
        "SIM_COLLECT_OUTPUT_ROOT": str(tmp_path / "takes"),
        "SIM_COLLECT_IPC": "ipc",
    })
    if backend is None:
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = backend
    return env, log


def _wait_for_roles(log: Path, expected: set[str], timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    while time.monotonic() < deadline:
        if log.exists():
            lines = log.read_text().splitlines()
            roles = {line.split("|")[1] for line in lines if line.startswith("start|")}
            if expected <= roles:
                return lines
        time.sleep(0.05)
    raise AssertionError(f"launcher roles {expected} did not start; records={lines}")


def _stop(proc: subprocess.Popen[str], signum: int = signal.SIGTERM) -> tuple[str, str]:
    proc.send_signal(signum)
    return proc.communicate(timeout=15)


def _start_records(lines: list[str]) -> list[list[str]]:
    return [line.split("|") for line in lines if line.startswith("start|")]


def _assert_service_groups_are_isolated(records: list[list[str]]) -> None:
    services = [parts for parts in records if parts[1] in {"sim", "capture", "gui"}]
    assert services
    assert all(parts[4] == parts[5] for parts in services)
    assert len({parts[5] for parts in services}) == len(services)


def test_launcher_is_valid_bash_and_help_lists_runtime_contract():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    result = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    for item in ("--headless", "--no-viewer", "--no-gui", "SIM_COLLECT_PY", "MUJOCO_GL"):
        assert item in result.stdout


@pytest.mark.parametrize(
    "args,message",
    [
        (["--config"], "--config requires a value"),
        (["--root", "--depth"], "--root requires a value"),
        (["--control-mode", "cartesian"], "--control-mode must be eef or joint"),
        (["--unknown"], "unknown arg: --unknown"),
    ],
)
def test_bad_arguments_fail_before_python_resolution(args: list[str], message: str):
    env = dict(os.environ)
    env.pop("SIM_COLLECT_PY", None)
    result = subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("signum,exit_code", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_headless_sets_egl_skips_gui_and_finalizes_on_signal(tmp_path: Path, signum: int, exit_code: int):
    env, log = _fake_env(tmp_path)
    config = tmp_path / "scene.yaml"
    config.write_text("scene: fake\n")
    output_root = tmp_path / "takes"
    proc = subprocess.Popen(
        [str(SCRIPT), "--headless", "--fake-leader", "--config", str(config),
         "--root", str(output_root), "--depth"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        lines = _wait_for_roles(log, {"sim", "capture"})
        assert not any(line.startswith("start|gui|") for line in lines)
        starts = _start_records(lines)
        _assert_service_groups_are_isolated(starts)
        assert all(parts[2] == "egl" for parts in starts)
        assert all(parts[3] == str(output_root) for parts in starts)
        sim = next(parts for parts in starts if parts[1] == "sim")
        capture = next(parts for parts in starts if parts[1] == "capture")
        assert "--no-viewer" in sim
        assert str(config.resolve()) in sim
        assert "--depth" in capture
        stdout, stderr = _stop(proc, signum)
        assert proc.returncode == exit_code, (stdout, stderr)
        records = log.read_text().splitlines()
        assert "term|capture|after_capture" in records
        assert "term|sim|after_capture" in records
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_relative_paths_default_glfw_and_capture_stops_before_interactive_processes(tmp_path: Path):
    env, log = _fake_env(tmp_path)
    config = tmp_path / "scene file.yaml"
    config.write_text("scene: fake\n")
    proc = subprocess.Popen(
        [str(SCRIPT), "--config", config.name, "--root", "take root"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        lines = _wait_for_roles(log, {"sim", "capture", "gui"})
        starts = _start_records(lines)
        _assert_service_groups_are_isolated(starts)
        assert all(parts[2] == "glfw" for parts in starts)
        assert all(parts[3] == str(tmp_path / "take root") for parts in starts)
        sim = next(parts for parts in starts if parts[1] == "sim")
        assert str(config.resolve()) in sim
        stdout, stderr = _stop(proc)
        assert proc.returncode == 143, (stdout, stderr)
        records = log.read_text().splitlines()
        assert "term|capture|after_capture" in records
        assert "term|sim|after_capture" in records
        assert "term|gui|after_capture" in records
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_explicit_backend_and_python_override_are_preserved(tmp_path: Path):
    env, log = _fake_env(tmp_path, backend="osmesa")
    config = tmp_path / "scene.yaml"
    config.write_text("scene: fake\n")
    proc = subprocess.Popen(
        [str(SCRIPT), "--no-viewer", "--no-gui", "--config", str(config)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        lines = _wait_for_roles(log, {"sim", "capture"})
        starts = _start_records(lines)
        _assert_service_groups_are_isolated(starts)
        assert all(parts[2] == "osmesa" for parts in starts)
        stdout, stderr = _stop(proc)
        assert proc.returncode == 143, (stdout, stderr)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_named_conda_env_resolves_to_its_python(tmp_path: Path):
    env, log = _fake_env(tmp_path)
    fake_python = tmp_path / "fake env" / "bin" / "python"
    conda_dir = tmp_path / "fake conda"
    conda_dir.mkdir()
    conda_log = tmp_path / "conda.log"
    conda = conda_dir / "conda"
    conda.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$FAKE_CONDA_LOG\"\n"
        "printf '%s\\n' \"$FAKE_CONDA_PY\"\n"
    )
    conda.chmod(0o755)
    config = tmp_path / "scene.yaml"
    config.write_text("scene: fake\n")
    env.pop("SIM_COLLECT_PY")
    env.pop("CONDA_DEFAULT_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env.update({
        "PATH": str(conda_dir) + os.pathsep + env["PATH"],
        "FAKE_CONDA_LOG": str(conda_log),
        "FAKE_CONDA_PY": str(fake_python),
    })
    proc = subprocess.Popen(
        [str(SCRIPT), "--headless", "--config", str(config)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_roles(log, {"sim", "capture"})
        stdout, stderr = _stop(proc)
        assert proc.returncode == 143, (stdout, stderr)
        assert "(conda env gello-sim)" in stdout
        assert conda_log.read_text().strip() == "run -n gello-sim python -c import sys; print(sys.executable)"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_cleanup_does_not_kill_matching_python_in_callers_group(tmp_path: Path):
    env, log = _fake_env(tmp_path)
    fake_python = tmp_path / "fake env" / "bin" / "python"
    config = tmp_path / "scene.yaml"
    config.write_text("scene: fake\n")
    unrelated = subprocess.Popen(
        [str(fake_python), "-c", "from multiprocessing import resource_tracker"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    proc = subprocess.Popen(
        [str(SCRIPT), "--headless", "--config", str(config)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        lines = _wait_for_roles(log, {"sim", "capture", "unknown"})
        _assert_service_groups_are_isolated(_start_records(lines))
        stdout, stderr = _stop(proc)
        assert proc.returncode == 143, (stdout, stderr)
        assert unrelated.poll() is None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)
