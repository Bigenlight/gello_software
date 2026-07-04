#!/usr/bin/env python3
"""Generate per-session GELLO -> UR joint tracking plots.

The recorder logs raw GELLO joints, final commands, and UR actual joints. It
does not log the bridge's internal _filtered state, so this script reconstructs
an approximate pre-clamp filter output from the GELLO stream and bridge params.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots

from ur_command_smoothing import accel_limited_command


ROOT = Path("ros2_ur_ws/gello_logs")
OUT = ROOT / "plots"
PARAMS_FILE = Path("ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml")
SMOOTH_ACCEL_RAD_S2 = 8.0
MARKER_LOOKBACK_S = 1.5

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MARKER_STYLES = {
    "ur": {
        "color": "rgba(21,147,91,0.85)",
        "dash": "dash",
        "fill": "rgba(21,147,91,0.08)",
    },
    "command": {
        "color": "rgba(207,63,63,0.85)",
        "dash": "dot",
        "fill": "rgba(207,63,63,0.08)",
    },
    "smooth-command": {
        "color": "rgba(38,103,255,0.90)",
        "dash": "dashdot",
        "fill": "rgba(38,103,255,0.10)",
    },
}


class OneEuroRecordingStyle:
    """Reconstruct the pre-clamp One Euro output used by recorded sessions.

    The current worktree has a timing fix, but the existing logs were collected
    before that change. The measured command trace is still the source of truth;
    this is only an estimate of the internal filter output before step clamp.
    """

    def __init__(self, dt: float, min_cutoff: float, beta: float, d_cutoff: float):
        self.dt = float(dt)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev = 0.0

    @staticmethod
    def alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def seed(self, x: float) -> None:
        self.x_prev = float(x)
        self.dx_prev = 0.0

    def __call__(self, x: float) -> float:
        x = float(x)
        if self.x_prev is None:
            self.seed(x)
            return x
        dx = (x - self.x_prev) / self.dt
        a_d = self.alpha(self.d_cutoff, self.dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff, self.dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def bridge_params() -> dict:
    data = yaml.safe_load(PARAMS_FILE.read_text())
    return data["gello_ur_bridge"]["ros__parameters"]


def reconstruct_filter_estimate(
    gello_df: pd.DataFrame,
    command_df: pd.DataFrame,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    t_g = gello_df["t_rel_s"].to_numpy(float)
    q_g = gello_df[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
    t_c = command_df["t_rel_s"].to_numpy(float)
    cmd = command_df[[f"cmd{i}" for i in range(1, 7)]].to_numpy(float)

    if len(t_g) == 0 or len(t_c) == 0:
        return t_c, np.empty((0, 6))

    publish_rate = float(params.get("publish_rate_hz", 250.0))
    dt = 1.0 / publish_rate
    filters = [
        OneEuroRecordingStyle(
            dt,
            float(params.get("one_euro_min_cutoff", 1.0)),
            float(params.get("one_euro_beta", 2.0)),
            float(params.get("one_euro_d_cutoff", 1.0)),
        )
        for _ in range(6)
    ]

    seed = cmd[0] if len(cmd) else q_g[0]
    for joint, filt in enumerate(filters):
        filt.seed(seed[joint])

    out = np.zeros((len(t_c), 6), dtype=float)
    for row, t in enumerate(t_c):
        idx = np.searchsorted(t_g, t, side="right") - 1
        idx = max(idx, 0)
        raw = q_g[idx]
        for joint in range(6):
            out[row, joint] = filters[joint](raw[joint])
    return t_c, out


def add_trace(
    fig,
    row: int,
    col: int,
    x,
    y,
    name: str,
    color: str,
    width: float = 1.5,
    dash: str | None = None,
    unit: str = "rad",
) -> None:
    line = {"color": color, "width": width}
    if dash is not None:
        line["dash"] = dash
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=(row == 1),
            line=line,
            hovertemplate=f"t=%{{x:.3f}}s<br>value=%{{y:.5f}} {unit}<extra>"
            + name
            + "</extra>",
        ),
        row=row,
        col=col,
    )


def finite_diff_velocity(t, q: np.ndarray, nominal_dt: float | None = None) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    v = np.full_like(q, np.nan, dtype=float)
    dq = np.diff(q, axis=0)
    if nominal_dt is None:
        dt = np.diff(t)
        valid = dt > 1e-6
        rows = np.flatnonzero(valid) + 1
        v[rows] = dq[valid] / dt[valid, None]
    else:
        v[1:] = dq / nominal_dt
    return v


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return (
        pd.DataFrame(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def load_markers(session: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(session.glob("markers_*.csv")):
        markers = pd.read_csv(path)
        if "source_sample_t_rel_s" in markers:
            marker_t = markers["source_sample_t_rel_s"]
        elif "source_t_rel_s" in markers:
            marker_t = markers["source_t_rel_s"]
        else:
            continue
        source = markers["source"] if "source" in markers else path.stem.removeprefix("markers_")
        frames.append(
            pd.DataFrame(
                {
                    "t_rel_s": pd.to_numeric(marker_t, errors="coerce"),
                    "source": source.astype(str),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["t_rel_s", "source"])
    return pd.concat(frames, ignore_index=True).dropna().sort_values("t_rel_s")


def add_marker_lines(fig, markers: pd.DataFrame) -> None:
    for _, marker in markers.iterrows():
        source = str(marker["source"])
        style = MARKER_STYLES.get(
            source,
            {
                "color": "rgba(80,88,105,0.75)",
                "dash": "dash",
                "fill": "rgba(80,88,105,0.08)",
            },
        )
        x = float(marker["t_rel_s"])
        x0 = max(0.0, x - MARKER_LOOKBACK_S)
        for row in range(1, 7):
            for col in (1, 2):
                fig.add_vrect(
                    x0=x0,
                    x1=x,
                    row=row,
                    col=col,
                    fillcolor=style["fill"],
                    line_width=0,
                    layer="below",
                )
                fig.add_vline(
                    x=x,
                    row=row,
                    col=col,
                    line_width=1.4,
                    line_dash=style["dash"],
                    line_color=style["color"],
                )


def make_session_plot(session: Path, params: dict) -> tuple[Path, float | str, dict]:
    meta = json.loads((session / "metadata.json").read_text())
    gello = pd.read_csv(session / "gello_joint_states.csv")
    command = pd.read_csv(session / "command.csv")
    ur = pd.read_csv(session / "ur_joint_states.csv")
    markers = load_markers(session)
    t_filter, filter_est = reconstruct_filter_estimate(gello, command, params)
    cmd_q = command[[f"cmd{i}" for i in range(1, 7)]].to_numpy(float)
    ur_vel = ur[[f"qd{i}" for i in range(1, 7)]].to_numpy(float)
    publish_rate = float(params.get("publish_rate_hz", 250.0))
    nominal_dt = 1.0 / publish_rate
    max_step_rad = float(params.get("max_step_rad", 0.0025))
    smooth_cmd_q, _ = accel_limited_command(
        cmd_q,
        publish_rate,
        max_step_rad,
        float(params.get("max_accel_rad_s2", SMOOTH_ACCEL_RAD_S2)),
    )
    cmd_vel = finite_diff_velocity(command["t_rel_s"], cmd_q, nominal_dt=nominal_dt)
    smooth_cmd_vel = finite_diff_velocity(
        command["t_rel_s"], smooth_cmd_q, nominal_dt=nominal_dt
    )
    filter_vel = (
        finite_diff_velocity(t_filter, filter_est, nominal_dt=nominal_dt)
        if len(filter_est)
        else filter_est
    )
    # The driver-reported qd is useful but visually spiky. A short centered
    # median keeps real motion trends visible while preserving the raw trace.
    ur_vel_smooth = rolling_median(ur_vel, window=9)

    fig = make_subplots(
        rows=6,
        cols=2,
        shared_xaxes="columns",
        vertical_spacing=0.035,
        horizontal_spacing=0.08,
        subplot_titles=sum(
            (
                [
                    f"J{i + 1}: {name} position",
                    f"J{i + 1}: {name} velocity",
                ]
                for i, name in enumerate(JOINT_NAMES)
            ),
            [],
        ),
    )

    for joint in range(6):
        row = joint + 1
        if len(filter_est):
            add_trace(
                fig,
                row,
                1,
                t_filter,
                filter_est[:, joint],
                "filter estimate before clamp",
                "#c97913",
                1.25,
                "dot",
                "rad",
            )
            add_trace(
                fig,
                row,
                2,
                t_filter,
                filter_vel[:, joint],
                "filter velocity estimate",
                "#c97913",
                1.15,
                "dot",
                "rad/s",
            )
        add_trace(
            fig,
            row,
            1,
            command["t_rel_s"],
            command[f"cmd{row}"],
            "actual command after clamp",
            "#cf3f3f",
            1.35,
            unit="rad",
        )
        add_trace(
            fig,
            row,
            2,
            command["t_rel_s"],
            cmd_vel[:, joint],
            "command velocity after clamp, nominal 250Hz",
            "#cf3f3f",
            1.25,
            unit="rad/s",
        )
        add_trace(
            fig,
            row,
            1,
            command["t_rel_s"],
            smooth_cmd_q[:, joint],
            "smooth command, accel/brake limited",
            "#2667ff",
            1.25,
            "dash",
            "rad",
        )
        add_trace(
            fig,
            row,
            2,
            command["t_rel_s"],
            smooth_cmd_vel[:, joint],
            "smooth command velocity",
            "#2667ff",
            1.2,
            "dash",
            "rad/s",
        )
        add_trace(
            fig,
            row,
            2,
            ur["t_rel_s"],
            ur_vel[:, joint],
            "UR actual velocity raw",
            "rgba(21,147,91,0.32)",
            0.9,
            unit="rad/s",
        )
        add_trace(
            fig,
            row,
            1,
            ur["t_rel_s"],
            ur[f"q{row}"],
            "UR actual position",
            "#15935b",
            1.35,
            unit="rad",
        )
        add_trace(
            fig,
            row,
            2,
            ur["t_rel_s"],
            ur_vel_smooth[:, joint],
            "UR actual velocity median 9 samples",
            "#15935b",
            1.55,
            unit="rad/s",
        )
        fig.update_yaxes(title_text="rad", row=row, col=1, zeroline=False)
        fig.update_yaxes(title_text="rad/s", row=row, col=2, zeroline=False)

    if len(markers):
        add_marker_lines(fig, markers)

    counts = meta.get("message_counts", {})
    duration = meta.get("duration_s", "?")
    marker_counts = markers["source"].value_counts().to_dict() if len(markers) else {}
    marker_text = (
        "markers="
        + ", ".join(
            f"{html.escape(str(source))}:{count}"
            for source, count in sorted(marker_counts.items())
        )
        + ". "
        if marker_counts
        else ""
    )
    title = (
        f"{session.name} | filter estimate vs command vs UR actual"
        f"<br><sup>duration={duration}s, "
        f"gello={counts.get('gello_joint_states', '?')} msgs, "
        f"command={counts.get('command', '?')} msgs, "
        f"ur={counts.get('ur_joint_states', '?')} msgs. "
        f"{marker_text}"
        "Raw GELLO is hidden; command velocity uses nominal 250Hz timing to avoid recorder callback timestamp spikes. "
        f"Smooth command uses accel limit {float(params.get('max_accel_rad_s2', SMOOTH_ACCEL_RAD_S2)):g}rad/s^2. "
        f"Marker bands show the {MARKER_LOOKBACK_S:g}s before each SPACE press.</sup>"
    )
    fig.update_layout(
        title=title,
        height=1780,
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 72, "r": 28, "t": 128, "b": 58},
        font={"family": "Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(title_text="time (s)", row=6, col=1)
    fig.update_xaxes(title_text="time (s)", row=6, col=2)

    out_file = OUT / f"{session.name}_joint_position_velocity.html"
    pio.write_html(fig, out_file, include_plotlyjs=True, full_html=True)
    return out_file, duration, counts


def write_index(outputs: list[tuple[Path, Path, float | str, dict]]) -> Path:
    cards = []
    for session, out_file, duration, counts in outputs:
        cards.append(
            f"""
      <a class="card" href="{html.escape(out_file.name)}">
        <strong>{html.escape(session.name)}</strong>
        <span>duration {html.escape(str(duration))}s</span>
        <small>gello {counts.get('gello_joint_states', '?')} · command {counts.get('command', '?')} · ur {counts.get('ur_joint_states', '?')}</small>
      </a>
            """
        )

    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GELLO UR Position and Velocity Plots</title>
  <style>
    body {{ margin: 0; background: #f7f8fb; color: #172033; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(900px, calc(100vw - 32px)); margin: 0 auto; padding: 34px 0 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ margin: 0 0 18px; color: #647084; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ display: grid; gap: 5px; min-height: 104px; padding: 16px; border: 1px solid #c7cedb; border-radius: 8px; background: #fff; color: inherit; text-decoration: none; box-shadow: 0 12px 30px rgba(23,32,51,.10); }}
    .card strong {{ font-size: 17px; }}
    .card span {{ color: #344158; }}
    .card small {{ color: #647084; }}
    .note {{ margin-top: 18px; padding: 14px 16px; border-left: 4px solid #2667ff; border-radius: 6px; background: #eef4ff; color: #263956; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <main>
    <h1>GELLO to UR joint position and velocity plots</h1>
    <p>Each episode contains six rows. The left column is joint position and the right column is joint velocity. Raw GELLO is hidden. Lines: <code>filter estimate before clamp</code>, <code>actual command after clamp</code>, <code>smooth command</code>, and <code>UR actual</code>. Marker bands come from <code>markers_*.csv</code> and cover the short lookback window before each SPACE press.</p>
    <section class="grid">
      {''.join(cards)}
    </section>
    <div class="note">Note: the recorder did not log bridge internal <code>_filtered</code>, so the filter output and filter velocity are reconstructed from GELLO logs and bridge params. The actual controller input is the measured <code>command.csv</code> trace. The blue smooth command is an offline acceleration/braking-limited replay candidate, not a recorded robot signal. Command velocity uses nominal 250Hz timing instead of callback timestamps because the recorder can receive command messages in bursts. UR velocity shows both raw <code>qd</code> and a centered 9-sample median. Marker lines use <code>source_sample_t_rel_s</code> when available; marker bands show the preceding lookback window.</div>
  </main>
</body>
</html>
"""
    out_file = OUT / "index.html"
    out_file.write_text(index)
    return out_file


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = bridge_params()
    outputs = []
    for session in sorted(p for p in ROOT.glob("session_*") if p.is_dir()):
        required = [
            session / "metadata.json",
            session / "gello_joint_states.csv",
            session / "command.csv",
            session / "ur_joint_states.csv",
        ]
        if not all(path.exists() for path in required):
            continue
        out_file, duration, counts = make_session_plot(session, params)
        outputs.append((session, out_file, duration, counts))
        print(f"wrote {out_file}")

    index = write_index(outputs)
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
