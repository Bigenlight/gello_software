#!/usr/bin/env python3
"""Fail-closed local checks for a paired IFQL/SVF real-eval recording run.

This program has no ROS imports and never starts a camera, robot, renderer, or
policy process.  The launcher calls it before it creates a run directory, then
again after creation so an operator gets an actionable failure before actuation.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def fail(message: str) -> None:
    raise SystemExit(f"real-eval recording preflight refused: {message}")


def resolve_ffmpeg(inference_python: Path, requested: str | None) -> Path:
    """Prefer an explicit/PATH ffmpeg, then imageio_ffmpeg from inference Python."""
    if requested:
        candidate = Path(requested)
    else:
        on_path = shutil.which("ffmpeg")
        if on_path:
            candidate = Path(on_path)
        else:
            probe = subprocess.run(
                [str(inference_python), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if probe.returncode != 0 or not probe.stdout.strip():
                fail(
                    "ffmpeg is absent from PATH and inference Python could not resolve "
                    "imageio_ffmpeg; set REAL_EVAL_FFMPEG to an executable"
                )
            candidate = Path(probe.stdout.strip())
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        fail(f"ffmpeg is not an executable file: {candidate}")
    return candidate.resolve()


def check_x264(ffmpeg: Path) -> None:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0 or "libx264" not in result.stdout:
        fail(f"ffmpeg has no libx264 encoder: {ffmpeg}")


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def check_writable(path: Path) -> None:
    parent = nearest_existing_parent(path)
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        fail(f"recording parent is not writable: {parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-python", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hdf5-log-dir", type=Path, required=True)
    parser.add_argument("--mp4-path", type=Path, required=True)
    parser.add_argument("--renderer-hook", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--print-ffmpeg", action="store_true")
    args = parser.parse_args()

    if not args.inference_python.is_file() or not os.access(args.inference_python, os.X_OK):
        fail(f"inference Python is not executable: {args.inference_python}")
    ffmpeg = resolve_ffmpeg(args.inference_python, args.ffmpeg)
    import_probe = subprocess.run(
        [str(args.inference_python), "-c", "import h5py; print(h5py.__version__)"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if import_probe.returncode != 0:
        fail(f"inference Python cannot import h5py: {import_probe.stderr.strip() or args.inference_python}")
    check_x264(ffmpeg)
    if not args.renderer_hook.is_file():
        fail(f"renderer hook is missing: {args.renderer_hook}")

    run_dir = args.run_dir.resolve(strict=False)
    hdf5_dir = args.hdf5_log_dir.resolve(strict=False)
    mp4_path = args.mp4_path.resolve(strict=False)
    if hdf5_dir.parent != run_dir or mp4_path.parent != run_dir:
        fail("HDF5 and MP4 paths must be direct children of the unique run directory")
    check_writable(run_dir)
    free = shutil.disk_usage(nearest_existing_parent(run_dir)).free
    required = int(args.min_free_gib * 1024**3)
    if free < required:
        fail(f"only {free} free bytes; need at least {required} bytes")
    stale = [p for p in (hdf5_dir, mp4_path, mp4_path.with_suffix(mp4_path.suffix + ".part")) if p.exists()]
    if stale:
        fail("stale partial/artifact conflict: " + ", ".join(map(str, stale)))

    # The launcher calls this once before mkdir to validate every dependency without
    # leaving an empty run directory behind, then once again after mkdir to make the
    # unique directory itself part of the checked contract.  Keep --print-ffmpeg as
    # machine-readable output, but never let it bypass the safety checks above.
    if args.print_ffmpeg:
        print(ffmpeg)
        return

    print(
        "real-eval recording preflight OK: "
        f"h5py={import_probe.stdout.strip()} ffmpeg={ffmpeg} "
        f"run_dir={run_dir} free_bytes={free}"
    )


if __name__ == "__main__":
    main()
