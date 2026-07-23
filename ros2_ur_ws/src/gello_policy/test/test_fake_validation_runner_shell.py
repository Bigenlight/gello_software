import os
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_remote_policy_fake_ur7e_validation.sh"
)


def _run(tmp_path: Path, venv: Path) -> subprocess.CompletedProcess[str]:
    params = tmp_path / "params.yaml"
    params.write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "",
            "REMOTE_CLIENT_VENV": str(venv),
            "SYSTEM_PYTHON": sys.executable,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), str(params), "127.0.0.1", "1"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_remote_client_venv_fails_before_network(tmp_path):
    venv = tmp_path / "missing-venv"

    result = _run(tmp_path, venv)

    assert result.returncode == 1
    assert f"remote-client environment is missing: {venv}" in result.stderr
    assert "Preflight: waiting for TCP endpoint" not in result.stdout


def test_wrong_overlay_grpc_version_fails_before_network(tmp_path):
    venv = tmp_path / "venv"
    purelib = venv / "lib" / "python" / "site-packages"
    grpc_package = purelib / "grpc"
    grpc_package.mkdir(parents=True)
    (grpc_package / "__init__.py").write_text(
        '__version__ = "1.30.2"\n',
        encoding="utf-8",
    )
    venv_python = venv / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' '{purelib}'\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)

    result = _run(tmp_path, venv)

    assert result.returncode == 1
    assert "gRPC client preflight: grpcio=1.30.2" in result.stdout
    assert "expected grpcio 1.74.0, loaded 1.30.2" in result.stderr
    assert "Preflight: waiting for TCP endpoint" not in result.stdout


def test_pinned_overlay_grpc_version_reaches_network_gate(tmp_path):
    venv = tmp_path / "venv"
    purelib = venv / "lib" / "python" / "site-packages"
    grpc_package = purelib / "grpc"
    grpc_package.mkdir(parents=True)
    (grpc_package / "__init__.py").write_text(
        '__version__ = "1.74.0"\n',
        encoding="utf-8",
    )
    venv_python = venv / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' '{purelib}'\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)

    result = _run(tmp_path, venv)

    assert result.returncode == 1
    assert "gRPC client preflight: grpcio=1.74.0" in result.stdout
    assert "Preflight: waiting for TCP endpoint 127.0.0.1:1" in result.stdout
    assert "generic policy server endpoint is unreachable" in result.stderr
