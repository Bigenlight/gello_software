#!/usr/bin/env python3
"""Convert gello_recorder ``vectors.h5`` + MP4 takes to learner demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFRA_ROOT = _REPO_ROOT / "serl_ur_infra"
_UR_GELLO_PACKAGE = (
    _REPO_ROOT / "ros2_ur_ws" / "src" / "ur_gello_bringup"
)
for path in (_INFRA_ROOT, _UR_GELLO_PACKAGE):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from ur_env.learner.recorded_demo import (  # noqa: E402
    DEFAULT_GRASP_PENALTY,
    DEFAULT_MAX_CAMERA_AGE_S,
    DEFAULT_MAX_SIGNAL_AGE_S,
    DEFAULT_SAMPLE_RATE_HZ,
    convert_recorded_takes,
    write_recorded_demo_pickle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more gello_recorder take directories into the "
            "strict canonical pickle consumed by run_rlpd_learner_server.py."
        )
    )
    parser.add_argument(
        "takes",
        nargs="+",
        help="take directories, each containing vectors.h5, cam1.mp4, cam2.mp4",
    )
    parser.add_argument("--output", required=True, help="new .pkl path; never overwritten")
    parser.add_argument(
        "--outcome",
        required=True,
        choices=("success", "truncated"),
        help=(
            "explicit terminal label for every supplied take: success gives the "
            "last transition reward=1/mask=0; truncated keeps reward=0/mask=1"
        ),
    )
    parser.add_argument("--episode-id-start", type=int, default=0)
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument(
        "--max-signal-age-s", type=float, default=DEFAULT_MAX_SIGNAL_AGE_S
    )
    parser.add_argument(
        "--max-camera-age-s", type=float, default=DEFAULT_MAX_CAMERA_AGE_S
    )
    parser.add_argument("--grasp-penalty", type=float, default=DEFAULT_GRASP_PENALTY)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    converted = convert_recorded_takes(
        args.takes,
        outcome=args.outcome,
        episode_id_start=args.episode_id_start,
        sample_rate_hz=args.sample_rate_hz,
        max_signal_age_s=args.max_signal_age_s,
        max_camera_age_s=args.max_camera_age_s,
        grasp_penalty=args.grasp_penalty,
    )
    output = write_recorded_demo_pickle(args.output, converted)
    summary = converted.summary()
    summary.update(
        {
            "output": str(output),
            "output_bytes": output.stat().st_size,
            "output_sha256": _sha256(output),
        }
    )
    saturated = sum(
        item.saturated_action_count for item in converted.takes
    )
    if saturated:
        summary["warnings"] = [
            "recorded teleoperation exceeded one 10 Hz RL action step in "
            f"{saturated}/{len(converted.transitions)} transitions; translation "
            "and rotation were direction-preserving norm-clamped and per-take "
            "fractions are reported"
        ]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
