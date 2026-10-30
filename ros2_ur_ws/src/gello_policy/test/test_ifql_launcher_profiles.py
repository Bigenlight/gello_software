import os
from pathlib import Path
import subprocess
import sys

import pytest


WORKSPACE = Path(__file__).resolve().parents[3]
LAUNCHER = WORKSPACE / "run_ur7e_ifql_real.sh"
CONFIG_DIR = WORKSPACE / "src" / "gello_policy" / "config"
RECORDING_PREFLIGHT = WORKSPACE / "setup_jazzy" / "real_eval_recording_preflight.py"


def _write_fixture(tmp_path: Path, task: str) -> dict[str, str]:
    run_dir = tmp_path / "run"
    server_dir = tmp_path / "server"
    qflow_dir = tmp_path / "qflow"
    run_dir.mkdir()
    server_dir.mkdir()
    (qflow_dir / "agents").mkdir(parents=True)
    (run_dir / "flags.json").write_text(
        '{"agent": {"encoder": "state"}, "env_name": "fixture_real.npz"}\n'
    )
    (run_dir / "norm_stats_fixture.json").write_text("{}\n")
    (run_dir / "params_100000.pkl").write_bytes(b"")
    server = server_dir / "ifql_server.py"
    server_started = server_dir / "server-started"
    server.write_text(
        "from pathlib import Path\n"
        f"Path({str(server_started)!r}).touch()\n"
        'raise SystemExit("dry-run must not execute the server")\n'
    )
    renderer = server_dir / "real_eval_renderer.py"
    renderer.write_text("# Team 1 hook fixture; dry-run never imports it.\n")
    ffmpeg = tmp_path / "ffmpeg-with-libx264"
    ffmpeg.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-hide_banner\" ] && [ \"${2:-}\" = \"-encoders\" ]; then\n"
        "  echo ' V.... libx264             H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10'\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    ffmpeg.chmod(0o755)
    params = tmp_path / f"{task}.yaml"
    params.write_text(
        (CONFIG_DIR / ("ifql_deploy_orange.yaml" if task == "orange" else "ifql_deploy.yaml"))
        .read_text()
        .replace("act_port: 5595", "act_port: 5589")
    )
    return {
        "IFQL_TASK": task,
        "IFQL_DRY_RUN": "1",
        "IFQL_COMPAT": "0",
        "IFQL_DEVICE": "cpu",
        "IFQL_PORT": "5589",
        "IFQL_ROOT": str(tmp_path / "root"),
        "IFQL_PY": sys.executable,
        "IFQL_SERVER_PY": str(server),
        "IFQL_RUN_DIR": str(run_dir),
        "IFQL_PARAMS_FILE": str(params),
        "IFQL_LOG_DIR": str(tmp_path / "logs"),
        "REAL_EVAL_FFMPEG": str(ffmpeg),
        "REAL_EVAL_RENDERER_HOOK": str(renderer),
        "REAL_EVAL_MIN_FREE_GIB": "0",
        "QFLOW_DIR": str(qflow_dir),
        "_SERVER_STARTED": str(server_started),
    }


@pytest.mark.parametrize("task", ["carrot", "orange"])
def test_ifql_task_profile_dry_run_starts_nothing(tmp_path, task):
    jazzy_setup = Path("/opt/ros/jazzy/setup.bash")
    overlay_setup = WORKSPACE / "install/setup.bash"
    if not jazzy_setup.exists() or not overlay_setup.exists():
        pytest.skip("ROS 2 Jazzy overlay is not built")
    env = os.environ.copy()
    env.update(_write_fixture(tmp_path, task))
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=WORKSPACE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert not Path(env["_SERVER_STARTED"]).exists(), result.stdout
    assert "IFQL_DRY_RUN=1" in result.stdout
    assert f"task={task} " in result.stdout
    assert "act_port=5589" in result.stdout
    assert "ros2 launch gello_policy ur7e_diffusion_real.launch.py" in result.stdout
    assert "--hdf5-log-dir" in result.stdout
    assert "--renderer-hook" in result.stdout
    assert "--renderer-output-path" in result.stdout
    manifest_text = next(line for line in result.stdout.splitlines() if " manifest=" in line)
    manifest = Path(manifest_text.split("manifest=", 1)[1].split()[0])
    assert manifest.is_file()
    data = __import__("json").loads(manifest.read_text())
    assert data["phase"] == "launching"
    assert data["recording"]["enabled"] is True
    assert data["policy"]["type"] == "ifql"
    assert data["artifacts"]["run_dir"] == str(manifest.parent)


def test_recording_preflight_resolves_explicit_ffmpeg_and_rejects_missing_renderer(tmp_path):
    """This runs only a pure-Python preflight: it cannot start ROS, robot, or cameras."""
    pytest.importorskip("h5py")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("#!/usr/bin/env bash\n[ \"${2:-}\" = \"-encoders\" ] && echo libx264\n")
    ffmpeg.chmod(0o755)
    hook = tmp_path / "hook.py"
    hook.write_text("# fixture\n")
    run = tmp_path / "run"
    common = [
        sys.executable, str(RECORDING_PREFLIGHT),
        "--inference-python", sys.executable, "--run-dir", str(run),
        "--hdf5-log-dir", str(run / "hdf5"), "--mp4-path", str(run / "policy_render.mp4"),
        "--renderer-hook", str(hook), "--ffmpeg", str(ffmpeg), "--min-free-gib", "0",
    ]
    resolved = subprocess.run(common + ["--print-ffmpeg"], text=True, capture_output=True, check=False)
    assert resolved.returncode == 0, resolved.stderr
    assert Path(resolved.stdout.strip()) == ffmpeg.resolve()
    passed = subprocess.run(common, text=True, capture_output=True, check=False)
    assert passed.returncode == 0, passed.stderr
    hook.unlink()
    failed = subprocess.run(common, text=True, capture_output=True, check=False)
    assert failed.returncode != 0
    assert "renderer hook is missing" in (failed.stdout + failed.stderr)
    # The pre-mkdir ffmpeg-resolution pass must perform the same dependency checks;
    # otherwise a bad hook would leave a useless empty run directory behind.
    failed_before_mkdir = subprocess.run(
        common + ["--print-ffmpeg"], text=True, capture_output=True, check=False
    )
    assert failed_before_mkdir.returncode != 0
    assert "renderer hook is missing" in (failed_before_mkdir.stdout + failed_before_mkdir.stderr)


def test_ifql_unknown_task_is_refused_before_start(tmp_path):
    env = os.environ.copy()
    env.update({"IFQL_TASK": "typo", "IFQL_ROOT": str(tmp_path)})
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=WORKSPACE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "IFQL_TASK must be carrot|orange" in result.stdout
