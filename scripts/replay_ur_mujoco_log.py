#!/usr/bin/env python3
"""Replay recorded GELLO -> UR logs in the MuJoCo viewer.

This is a visualization tool, not a controller. It loads the MuJoCo Menagerie
UR5e model and writes recorded joint positions directly into qpos over time.
That makes the logged UR motion inspectable without ROS2 or robot hardware.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from ur_command_smoothing import accel_limited_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    PROJECT_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "universal_robots_ur5e"
    / "scene.xml"
)
DEFAULT_LOG_ROOT = PROJECT_ROOT / "ros2_ur_ws" / "gello_logs"

SOURCES = {
    "ur": ("ur_joint_states.csv", "q"),
    "command": ("command.csv", "cmd"),
    "smooth-command": ("command.csv", "cmd"),
    "gello": ("gello_joint_states.csv", "q"),
}


def latest_session(log_root: Path) -> Path:
    sessions = sorted(p for p in log_root.glob("session_*") if p.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No session_* directories found under {log_root}")
    return sessions[-1]


def load_trajectory(
    session: Path,
    source: str,
    smooth_rate_hz: float,
    smooth_max_step_rad: float,
    smooth_accel_rad_s2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    filename, prefix = SOURCES[source]
    csv_path = session / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    cols = [f"{prefix}{i}" for i in range(1, 7)]
    missing = [col for col in ["t_rel_s", *cols] if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    t = df["t_rel_s"].to_numpy(dtype=float)
    q = df[cols].to_numpy(dtype=float)
    keep = np.isfinite(t) & np.isfinite(q).all(axis=1)
    t = t[keep]
    q = q[keep]
    if len(t) < 2:
        raise ValueError(f"{csv_path} does not contain enough finite samples")

    if source == "smooth-command":
        q, _ = accel_limited_command(
            q,
            smooth_rate_hz,
            smooth_max_step_rad,
            smooth_accel_rad_s2,
        )

    source_start_t = float(t[0])
    t = t - source_start_t
    return t, q, source_start_t


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    n = min(6, model.nq, len(q))
    data.qpos[:n] = q[:n]
    data.qvel[: min(6, model.nv)] = 0.0
    if model.nu >= n:
        data.ctrl[:n] = q[:n]
    mujoco.mj_forward(model, data)


def sample_pose(t: np.ndarray, q: np.ndarray, elapsed: float) -> np.ndarray:
    idx = int(np.searchsorted(t, elapsed, side="right") - 1)
    idx = max(0, min(idx, len(t) - 1))
    return q[idx]


def sample_index(t: np.ndarray, elapsed: float) -> int:
    idx = int(np.searchsorted(t, elapsed, side="right") - 1)
    return max(0, min(idx, len(t) - 1))


class SpaceMarkerLogger:
    def __init__(
        self,
        csv_path: Path,
        source: str,
        source_start_t: float,
        t: np.ndarray,
        q: np.ndarray,
    ) -> None:
        self.csv_path = csv_path.resolve()
        self.source = source
        self.source_start_t = source_start_t
        self.t = t
        self.q = q
        self.loop_count = 0
        self.elapsed = 0.0
        self.idx = 0
        self._last_mark_wall = 0.0

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._file = self.csv_path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(
                [
                    "mark_wall_s",
                    "source",
                    "loop",
                    "playback_t_s",
                    "source_t_rel_s",
                    "source_sample_t_rel_s",
                    "sample_index",
                    "q1",
                    "q2",
                    "q3",
                    "q4",
                    "q5",
                    "q6",
                ]
            )

    def close(self) -> None:
        self._file.close()

    def update(self, elapsed: float, idx: int, loop_count: int) -> None:
        self.elapsed = float(elapsed)
        self.idx = int(idx)
        self.loop_count = int(loop_count)

    def mark(self) -> None:
        now = time.time()
        # Avoid multiple rows from keyboard repeat when the spacebar is held.
        if now - self._last_mark_wall < 0.20:
            return
        self._last_mark_wall = now
        idx = self.idx
        source_t = self.source_start_t + self.elapsed
        source_sample_t = self.source_start_t + float(self.t[idx])
        row = [
            f"{now:.6f}",
            self.source,
            self.loop_count,
            f"{self.elapsed:.6f}",
            f"{source_t:.6f}",
            f"{source_sample_t:.6f}",
            idx,
        ] + [f"{v:.9f}" for v in self.q[idx, :6]]
        self._writer.writerow(row)
        self._file.flush()
        print(
            f"MARK {self.source}: playback_t={self.elapsed:.3f}s "
            f"source_t_rel={source_t:.3f}s idx={idx} -> {self.csv_path}",
            flush=True,
        )


def render_mp4(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: np.ndarray,
    q: np.ndarray,
    args: argparse.Namespace,
) -> None:
    out_path = args.render_mp4.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_duration = t[-1] / max(args.speed, 1e-6)
    if args.max_video_seconds is not None:
        video_duration = min(video_duration, args.max_video_seconds)
    n_frames = max(1, int(video_duration * args.render_fps))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer = imageio.get_writer(str(out_path), fps=args.render_fps)
    try:
        for frame in range(n_frames):
            video_t = frame / args.render_fps
            traj_t = min(video_t * args.speed, t[-1])
            apply_pose(model, data, sample_pose(t, q, traj_t))
            renderer.update_scene(data)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"Wrote {out_path}")


def replay(args: argparse.Namespace) -> None:
    xml_path = args.xml.resolve()
    if not xml_path.exists():
        raise FileNotFoundError(
            f"MuJoCo UR XML not found: {xml_path}\n"
            "Initialize the menagerie submodule first:\n"
            "  git submodule update --init --recursive third_party/mujoco_menagerie"
        )

    session = args.session
    if session is None:
        session = latest_session(args.log_root)
    session = session.resolve()

    t, q, source_start_t = load_trajectory(
        session,
        args.source,
        args.smooth_rate_hz,
        args.smooth_max_step_rad,
        args.smooth_accel_rad_s2,
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    apply_pose(model, data, q[0])

    print(f"Loaded model: {xml_path}")
    print(f"Loaded session: {session}")
    print(f"Source: {args.source} ({len(t)} samples, {t[-1]:.2f}s)")
    if args.source == "smooth-command":
        print(
            "Smooth command params: "
            f"rate={args.smooth_rate_hz:g}Hz, "
            f"max_step={args.smooth_max_step_rad:g}rad, "
            f"max_accel={args.smooth_accel_rad_s2:g}rad/s^2"
        )
    if args.check_only:
        print(f"Model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
        print("Check complete; viewer was not launched.")
        return

    if args.render_mp4 is not None:
        render_mp4(model, data, t, q, args)
        return

    marker_path = (
        args.markers_csv
        if args.markers_csv is not None
        else session / f"markers_{args.source}.csv"
    )
    marker_logger = SpaceMarkerLogger(
        marker_path, args.source, source_start_t, t, q
    )

    print("Close the MuJoCo viewer window to stop.")
    print(f"Press SPACE to save a marker to: {marker_logger.csv_path}")

    def on_key(key: int) -> None:
        # GLFW_KEY_SPACE == 32. Avoid importing glfw; mujoco's callback only
        # gives the key code.
        if key == 32:
            marker_logger.mark()

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=on_key
        ) as viewer:
            start_wall = time.monotonic()
            loop_count = 0
            while viewer.is_running():
                elapsed = (time.monotonic() - start_wall) * args.speed
                if args.loop and elapsed > t[-1]:
                    start_wall = time.monotonic()
                    loop_count += 1
                    elapsed = 0.0
                elif elapsed > t[-1]:
                    elapsed = t[-1]

                idx = sample_index(t, elapsed)
                marker_logger.update(elapsed, idx, loop_count)
                apply_pose(model, data, q[idx])

                viewer.sync()
                time.sleep(max(0.0, 1.0 / args.fps))
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            raise RuntimeError(
                "MuJoCo interactive viewer on macOS requires mjpython. Run:\n"
                "  mjpython scripts/replay_ur_mujoco_log.py --source ur --loop\n"
                "If mjpython aborts in this environment, render a video instead:\n"
                "  conda run -n base python scripts/replay_ur_mujoco_log.py "
                "--source ur --render-mp4 /tmp/ur_replay.mp4"
            ) from exc
        raise
    finally:
        marker_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a GELLO/UR log session in the MuJoCo viewer."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Path to ros2_ur_ws/gello_logs/session_*; default: latest session.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Directory containing session_* logs.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="UR MuJoCo XML path.",
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="ur",
        help=(
            "Trajectory to replay: ur actual, final command, smoothed final "
            "command, or gello raw."
        ),
    )
    parser.add_argument(
        "--smooth-rate-hz",
        type=float,
        default=250.0,
        help="Nominal command rate used by --source smooth-command.",
    )
    parser.add_argument(
        "--smooth-max-step-rad",
        type=float,
        default=0.0025,
        help="Per-sample position step cap used by --source smooth-command.",
    )
    parser.add_argument(
        "--smooth-accel-rad-s2",
        type=float,
        default=8.0,
        help="Acceleration limit used by --source smooth-command.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Viewer update rate.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback instead of holding the last pose.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the model and log, then exit without opening the viewer.",
    )
    parser.add_argument(
        "--render-mp4",
        type=Path,
        default=None,
        help="Render playback to an mp4 instead of opening the interactive viewer.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="FPS for --render-mp4.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width for --render-mp4.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height for --render-mp4.",
    )
    parser.add_argument(
        "--max-video-seconds",
        type=float,
        default=None,
        help="Optional cap on rendered video duration for quick previews.",
    )
    parser.add_argument(
        "--markers-csv",
        type=Path,
        default=None,
        help="CSV path for SPACE-key markers. Default: session/markers_<source>.csv.",
    )
    return parser.parse_args()


def main() -> None:
    replay(parse_args())


if __name__ == "__main__":
    main()
