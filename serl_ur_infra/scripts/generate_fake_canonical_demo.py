#!/usr/bin/env python3
"""Generate a small canonical fake demo for learner acceptance runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))

from ur_env.learner import load_demo_pickle, write_fake_demo_pickle  # noqa: E402
from ur_env.learner.demo import SYNTHETIC_ACCEPTANCE_ONLY_KEY  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a deterministic SYNTHETIC ACCEPTANCE-ONLY canonical demo "
            "pickle. The learner server rejects this artifact for real "
            "serving and accepts it only with --dry-run or "
            "--synthetic-e2e. Existing files are "
            "never overwritten."
        )
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = write_fake_demo_pickle(args.output)
    loaded = load_demo_pickle(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    synthetic_count = sum(
        sidecar.metadata.get(SYNTHETIC_ACCEPTANCE_ONLY_KEY) is True
        for sidecar in loaded.sidecars
    )
    print(
        json.dumps(
            {
                "demo_path": str(output),
                "sha256": digest,
                "transition_count": len(loaded),
                SYNTHETIC_ACCEPTANCE_ONLY_KEY: True,
                "synthetic_transition_count": synthetic_count,
                "warning": (
                    "SYNTHETIC ACCEPTANCE-ONLY: use as --demo-path only "
                    "with run_rlpd_learner_server.py --dry-run or "
                    "--synthetic-e2e"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
