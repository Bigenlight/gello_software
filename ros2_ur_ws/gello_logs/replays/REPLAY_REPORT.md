# GELLO -> UR7e Teleop Log Replay Report

**Verdict: Reproduction works fully headless.** All 8 reproductions (2 sessions x 4 sources) rendered to MP4 in MuJoCo with NO ROS2, NO robot, and NO physical GELLO attached — offscreen via `MUJOCO_GL=osmesa`.

- Date: 2026-07-05
- Tool: `scripts/replay_ur_mujoco_log.py --render-mp4`
- Output directory: `/home/theo/gello_software/ros2_ur_ws/gello_logs/replays/`

---

## How to view / reproduce

All commands run from the repo root inside the project venv with the osmesa (software) GL backend so no display/GPU is required.

**Render an MP4 (offscreen, headless — what this report used):**
```bash
source .venv/bin/activate
MUJOCO_GL=osmesa python scripts/replay_ur_mujoco_log.py \
    ros2_ur_ws/gello_logs/session_20260703_165530 \
    --source smooth-command \
    --render-mp4
```

**Interactive viewer (on a machine with a display):**
```bash
source .venv/bin/activate
MUJOCO_GL=osmesa python scripts/replay_ur_mujoco_log.py \
    ros2_ur_ws/gello_logs/session_20260703_165530 \
    --source ur
```

Useful flags:
- `--source {ur, command, smooth-command, gello}` — pick which recorded stream to replay.
- `--loop` — replay the trajectory repeatedly.
- `--speed <factor>` — play back faster/slower than real time.

Sources explained:
- **ur** — actual UR7e post-controller joint states (ground truth, 100 Hz).
- **command** — raw bridge-to-controller command stream (250 Hz).
- **smooth-command** — accel-limited smoothed command (max_step 0.0025 rad, max_accel 8 rad/s^2).
- **gello** — raw GELLO leader-arm angles, pre-filter (~30 Hz).

---

## The 8 outputs

| Session | Source | MP4 (bytes) | Samples | Rate (Hz) | Dur (s) | Peak speed (rad/s) | Jerk | Large steps | NaN |
|---|---|---|---|---|---|---|---|---|---|
| 165530 | ur | 1,138,188 | 3648 | 100 | 36.47 | 0.9083 | 0.0077 | 0 | 0 |
| 165530 | command | 598,706 | 9337 | 250 | 37.47 | 17.5472 | 0.0276 | 0 | 0 |
| 165530 | smooth-command | 1,157,991 | 9337 | 250 | 37.47 | 17.5472 (→12.07 smoothed) | 0.0276 (→0.0183) | 0 | 0 |
| 165530 | gello | 1,156,833 | 1124 | 30.03 | 37.43 | 53.5359 | 0.1015 | 68 | 0 |
| 171323 | ur | 1,239,206 | 4233 | 100 | 42.32 | 0.7844 | 0.0066 | 0 | 0 |
| 171323 | command | 536,701 | 10751 | 250 | 42.99 | 5.1540 | 0.0104 | 0 | 0 |
| 171323 | smooth-command | 1,260,702 | 10751 | 250 | 42.99 | 5.1540 (→3.84 smoothed) | 0.0104 (→0.006) | 0 | 0 |
| 171323 | gello | 1,251,204 | 1290 | 30.03 | 42.97 | 0.7898 | 0.0325 | 11 | 0 |

MP4 paths (all under the output directory above):
- `session_20260703_165530__ur.mp4`
- `session_20260703_165530__command.mp4`
- `session_20260703_165530__smooth-command.mp4`
- `session_20260703_165530__gello.mp4`
- `session_20260703_171323__ur.mp4`
- `session_20260703_171323__command.mp4`
- `session_20260703_171323__smooth-command.mp4`
- `session_20260703_171323__gello.mp4`

Every entry reproduced successfully (`reproduces: true`, MP4 written, exit 0, 0 NaNs).

---

## The pipeline / vibration story: gello -> command -> smooth-command -> ur

The chain localizes where teleop vibration is born and where it dies.

**Session 165530 (monotonic cleanup):**
1. **ORIGIN — gello (raw leader, 30.03 Hz):** vibration is born here — peak_speed 53.54 rad/s (~59x the robot's 0.91), jerk 0.1015 (~13x the robot's 0.0077), and 68 large steps across 1124 samples (a discontinuity roughly every 16 samples). The coarse 30 Hz encoder stream, not the robot, is the source.
2. **BRIDGE — command (250 Hz interpolation):** the dominant fix. Peak_speed drops 53.54 -> 17.55 rad/s (-67%), jerk 0.1015 -> 0.0276 (-73%), and n_large_steps collapses 68 -> 0. Upsampling removes all step discontinuities and most jerk but leaves a residual velocity spike (17.55 rad/s).
3. **SMOOTHING — smooth-command:** a real but secondary polish. Peak_speed 17.55 -> 12.07 rad/s (-31%), jerk 0.0276 -> 0.0183 (-34%), joint ranges unchanged (damps velocity/jerk without distorting the path). Still 12.07 rad/s > 5 rad/s safe band, so command-side filtering alone does not fully tame the transient.
4. **ACTUAL ROBOT — ur (100 Hz):** by far the smoothest — peak_speed 0.9083 rad/s, jerk 0.0077, 0 large steps. Controller + physical inertia absorb the remaining spike; the real motion is safe.

**Session 171323 (reshaped, not just reduced):**
1. **ORIGIN — gello (30.03 Hz):** roughest link by jerk — jerk 0.0325 (chain max, ~5x the robot's 0.0066) with 11 large steps, but peak speed only 0.79 rad/s. The problem is coarse 30 Hz sampling producing jerky quantized angle jumps, not fast motion.
2. **BRIDGE — command (250 Hz):** cleans discontinuities completely (11 -> 0 large steps, jerk 0.0325 -> 0.0104) but injects a ~6.5x peak-speed spike (0.79 -> 5.154 rad/s), edging past the ~5 rad/s guideline. The interpolator, not the operator, creates these velocity peaks.
3. **SMOOTHING — smooth-command:** genuinely effective — peak_speed 5.154 -> 3.84 rad/s (~25%, back under 5 rad/s) and jerk 0.0104 -> 0.006 (~42%).
4. **ACTUAL ROBOT — ur (100 Hz):** cleanest — jerk 0.0066 (essentially matching the smoothed command's 0.006), 0 discontinuities, peak speed back to 0.784 rad/s. The robot faithfully tracks the smoothed command's jerk profile.

**Bottom line:** vibration originates in the raw ~30 Hz leader stream; 250 Hz bridge interpolation is the dominant cleanup (removes all discontinuities) but can inject velocity spikes; accel-limited smoothing adds a measurable, warranted reduction (~25-42%); and the physical robot/controller absorbs the rest so actual motion is smooth and safe.

---

## Anomalies

**Safety — peak-speed spikes above the ~5 rad/s guideline (all in the raw/command streams, never in the actual robot):**
- 165530 / command: raw peak_speed 17.55 rad/s (transient spike in the unfiltered command).
- 165530 / smooth-command: raw 17.55 -> smoothed 12.07 rad/s — still above the safe band.
- 165530 / gello: peak_speed 53.54 rad/s and 68 large steps — raw pre-filter leader discontinuities.
- 171323 / command: peak_speed 5.154 rad/s, marginally over (joint range 1.53 rad).
- 171323 / smooth-command: raw 5.154 -> smoothed 3.84 rad/s (pulled back under 5 rad/s).
- The actual robot (`ur`) peaks at only 0.78-0.91 rad/s in both sessions — well within safe limits.

**Data-quality bug — FIXED (both sessions):** `analyze_replay` originally reported `vibration_reduction_pct = 0.0` on smooth-command because it was computed from `n_large_steps` (already 0 at 250 Hz). The QA/render agents flagged this; the helper now derives reduction from the jerk metric (and also reports `peak_speed_reduction_pct`). Verified values: session 165530 → jerk −33.7% / peak-speed −31.2%; session 171323 → jerk −42.3% / peak-speed −25.5%.

**Not anomalies (expected):** the 11-68 large steps and elevated jerk on the `gello` source reflect the coarse ~30 Hz pre-filter leader signal that downstream smoothing is designed to suppress. Zero NaNs across all 8 reproductions.
