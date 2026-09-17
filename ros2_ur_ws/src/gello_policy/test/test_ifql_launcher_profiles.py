import os
from pathlib import Path
import subprocess

import pytest


WORKSPACE = Path(__file__).resolve().parents[3]
LAUNCHER = WORKSPACE / "run_ur7e_ifql_real.sh"
CONFIG_DIR = WORKSPACE / "src" / "gello_policy" / "config"


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
    server.write_text('raise SystemExit("dry-run must not execute the server")\n')
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
        "IFQL_PY": "/usr/bin/python3",
        "IFQL_SERVER_PY": str(server),
        "IFQL_RUN_DIR": str(run_dir),
        "IFQL_PARAMS_FILE": str(params),
        "IFQL_LOG_DIR": str(tmp_path / "logs"),
        "QFLOW_DIR": str(qflow_dir),
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
    assert "IFQL_DRY_RUN=1" in result.stdout
    assert f"task={task} " in result.stdout
    assert "act_port=5589" in result.stdout
    assert "ros2 launch gello_policy ur7e_diffusion_real.launch.py" in result.stdout


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
