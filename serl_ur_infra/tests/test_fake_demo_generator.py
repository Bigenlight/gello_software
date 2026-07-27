"""Canonical fake-demo generator and CLI tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
_SCRIPT = _INFRA / "scripts" / "generate_fake_canonical_demo.py"
sys.path.insert(0, str(_INFRA))

from ur_env.learner import (  # noqa: E402
    DemoContractError,
    build_fake_demo_payload,
    load_demo_object,
    load_demo_pickle,
    write_fake_demo_pickle,
)
from ur_env.learner.demo import SYNTHETIC_ACCEPTANCE_ONLY_KEY  # noqa: E402


_SPEC = importlib.util.spec_from_file_location(
    "generate_fake_canonical_demo_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_fake_demo_is_deterministic_and_uses_strict_loader(tmp_path):
    first_path = write_fake_demo_pickle(tmp_path / "first.pkl")
    second_path = write_fake_demo_pickle(tmp_path / "second.pkl")

    assert first_path.read_bytes() == second_path.read_bytes()
    loaded = load_demo_pickle(first_path)
    assert len(loaded) == 2
    assert loaded.sidecars[0].metadata["success"] is True
    assert loaded.sidecars[1].metadata["success"] is False
    assert loaded.sidecars[1].metadata["run_id"] == "fake-acceptance-run"
    assert loaded.sidecars[1].metadata["intervened"] == 1
    assert all(
        sidecar.metadata[SYNTHETIC_ACCEPTANCE_ONLY_KEY] is True
        for sidecar in loaded.sidecars
    )
    for transition in loaded.transitions:
        assert set(transition) == {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "masks",
            "grasp_penalty",
        }
        assert transition["actions"].dtype == np.float32
        assert transition["actions"].shape == (7,)
        assert transition["observations"]["state"].shape == (1, 19)
        assert transition["observations"]["cam1"].dtype == np.uint8


def test_fake_demo_provenance_marker_is_strict_and_consistent():
    malformed = build_fake_demo_payload()
    malformed[0][SYNTHETIC_ACCEPTANCE_ONLY_KEY] = "yes"
    with pytest.raises(DemoContractError, match=SYNTHETIC_ACCEPTANCE_ONLY_KEY):
        load_demo_object(malformed)

    conflicting = build_fake_demo_payload()
    conflicting[1]["transition"][SYNTHETIC_ACCEPTANCE_ONLY_KEY] = False
    with pytest.raises(DemoContractError, match="markers disagree"):
        load_demo_object(conflicting)


def test_fake_demo_never_overwrites_existing_path(tmp_path):
    output = tmp_path / "demo.pkl"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError):
        write_fake_demo_pickle(output)

    assert output.read_bytes() == b"keep-me"


def test_fake_demo_cli_reports_loader_validated_artifact(tmp_path, capsys):
    output = tmp_path / "nested" / "demo.pkl"

    assert _MODULE.main(["--output", str(output)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["demo_path"] == str(output.resolve())
    assert result["transition_count"] == 2
    assert result[SYNTHETIC_ACCEPTANCE_ONLY_KEY] is True
    assert result["synthetic_transition_count"] == 2
    assert "SYNTHETIC ACCEPTANCE-ONLY" in result["warning"]
    assert len(result["sha256"]) == 64
    assert len(load_demo_pickle(result["demo_path"])) == 2
