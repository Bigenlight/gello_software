# GELLO Startup-Snap — Experiment Verdict

**Scope:** Empirically test the diagnosed cause of a real UR7e's sudden fast motion right after the GELLO move-to-start handshake, and validate fixes. All work done on THIS PC with NO GELLO / NO robot (pure-math harness, real bridge node with synthetic ROS inputs, source-code inspection, recorded-data analysis).

---

## 1. Headline Verdict

**CONFIRMED.** The diagnosed mechanism — a frozen move-to-start target opens a position gap `G` at handover, which the streaming bridge closes as a **bounded, clamp-limited catch-up sweep pinned at exactly `max_step_rad × publish_rate_hz = 0.0025 × 250 = 0.625 rad/s (35.8 deg/s)`** — is reproduced exactly across all experiments. It is a slew catch-up, never a step jump. Realistic hold-still drift (0.155–0.25 rad) lands squarely in the fully-saturated regime, so real handoffs routinely produce a several-hundred-ms 35.8 deg/s sweep from a dead stop right after 5s of smooth trajectory motion — exactly the operator-reported "sudden fast motion."

## 2. In-Stack REAL Bridge vs Pure-Math Harness — Model Validated?

**YES — validated to within ~1%.** The real `gello_ur_bridge` ROS2 node, run standalone with synthetic ROS inputs, matched the harness:

- Sustained catch-up speed **0.6267 rad/s** (real node) vs **0.625 rad/s** (harness prediction) → **0.27% error**.
- Gap-close time for G=0.2 rad: **0.3223 s** measured vs **0.320 s** predicted (G/0.625) → **0.7% error**.
- Zero-jump seed confirmed: first command == actual `/joint_states` pose exactly.
- `max_step_rad=0.0025` clamp enforced EXACTLY every publish cycle (max |per-cycle delta| = 0.0025000 rad, never exceeded), across 1485 commands @ 249.6 Hz over 6.006 s.
- Staleness watchdog fired correctly (~2.5 s after probe stopped).

The pure-math harness therefore faithfully models the production code path.

## 3. RANK1 Competing Cause (FPC-switch injects an additional jump) — Ruled In or Out?

**RULED OUT** by two independent lines of evidence:

- **Empirically:** every simulated + in-stack run peaked at exactly **0.625 rad/s** (worst naive per-sample spike = 0.894 rad/s, a timer-jitter artifact) — **zero occurrences approaching the >1 rad/s single-cycle discriminator** that would indicate an FPC-activation jump.
- **By source inspection of the exact installed software:** `forward_position_controller` is `position_controllers::JointGroupPositionController` (a thin `ForwardControllersBase` subclass, installed 2.53.1); its `on_activate()` only clears the RT command buffer and `update()` no-ops until the first `~/commands` message — it is architecturally incapable of writing any command before the bridge's first publish. And `ur_robot_driver` 2.13.2's `perform_command_mode_switch()` unconditionally reseeds `urcl_position_commands_ = urcl_joint_positions_` (the just-read ACTUAL pose) at the switch (`hardware_interface.cpp:864/1322/1366`), so the switch is provably a stationary hold.

The unfed-FPC window is not a jump source — though it does add GELLO-leader drift time that feeds gap `G`, additive to the 5s trajectory window. *Caveat: the ros2_controllers side was cited from the humble-branch HEAD (no exact 2.53.1 tag on GitHub), not a version-pinned tag; a ready-to-run rosbag discriminator recipe exists but is unexecuted pending real hardware.*

## 4. Fix Effectiveness Ranking

Ceiling in all cases = `max_step_rad × publish_rate_hz = 0.625 rad/s`. Fixes evaluated over the measured drift range 0.155–0.25 rad.

| Rank | Fix | Peak speed | Fast-sweep duration (≥90% ceiling) | Notes |
|------|-----|-----------|-----------------------------------|-------|
| **1** | **Fix #1 — re-latch move-to-start target to LIVE gello pose** | 0.468 rad/s mean (25.2% ↓ from 0.625); below-ceiling in 2/3 cases | **0.0067 s mean vs 0.299 s baseline → 97.8% ↓** | Shrinks G ~3× (0.15–0.30 → 0.05–0.10 rad). Cheapest, biggest single win. Does NOT fully eliminate (worst G=0.10 still pegs clamp ~20 ms). |
| **2** | **Soft-start — time-ramped `max_step`** | 0.420 / 0.485 / 0.542 rad/s across 0.155/0.2/0.25 rad (fixed 1.0 s ramp) | replaces step-onset with hundreds-of-ms ramp | One parameter, no extra trajectory action; degrades most gracefully across drift range; small secondary bump at ramp-completion edge. |
| **3** | **Fix #2 — catch-up trajectory before streaming** | 0.283 / 0.377 / 0.482 rad/s (retuned 0.75 s catch-up) | lowest peak for headline G=0.2 rad (0.471 rad/s @ 0.6 s) | Best peak reduction BUT duration-dependent: fixed 0.4 s catch-up made it WORSE than baseline (0.707 rad/s); must size duration = gap / velocity-budget from the *measured* gap or it silently degrades. |

All three keep final tracking error ≤ 0.0046 rad (~0.26 deg) — no steady-state accuracy tradeoff. Fix #1 + (soft-start or fix #2) stack for full elimination.

**Top-fix one-line code change (Fix #1):** In `ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/gello_move_to_start_node.py`, re-read the live GELLO pose immediately before building the 5 s trajectory instead of using the pose latched at first-sight/t=0 — i.e., set the trajectory target = current `/gello_joint_states` sample at the instant the trajectory is dispatched (not the cached t=0 pose). This removes the Play-press-wait aging from gap `G`, shrinking it from 0.15–0.30 rad to the 5 s-only residual of 0.05–0.10 rad.

## 5. Key Numbers Across Experiments

| Quantity | Value | Source experiment |
|---|---|---|
| Theoretical / realized max slew | 0.6250 rad/s = 35.8 deg/s | all (14/14 saturated runs) |
| Clamp enforcement | max |Δcmd| = 0.0025000 rad (machine-precision) | sweep (14/14), in-stack |
| one_euro saturation onset | G ≥ 0.15 rad (0.6140 @ G=0.10 → 0.6250 @ G=0.15) | sweep-signature |
| ema saturation | immediate, 0.6250 rad/s at all G (0.02–0.40) | sweep-signature |
| In-stack sustained speed | 0.6267 rad/s (0.27% vs harness) | instack-real-bridge |
| In-stack gap-close (G=0.2) | 0.3223 s vs 0.320 s predicted (0.7%) | instack-real-bridge |
| In-stack publish rate | 249.6 Hz over 1485 cmds / 6.006 s | instack-real-bridge |
| Worst per-sample spike | 0.894 rad/s (timer-jitter artifact, < 1 rad/s) | instack-real-bridge |
| Quiet-window drift, session 165530 | 0.250 rad = 14.32 deg (q6) | data-drift-distribution |
| Quiet-window drift, session 171323 | 0.158 rad = 9.05 deg (q4) | data-drift-distribution |
| No 5 s window below | ~0.15 rad (2113 windows total) | data-drift-distribution |
| Settle scaling | 0.24–0.48 s @ G=0.158; 0.48–0.84 s @ G=0.250 | data-drift-distribution |
| Fix #1 peak speed | 0.468 rad/s mean (25.2% ↓); {0.301,0.478,0.625} @ G={0.05,0.075,0.10} | fix1-refresh-target |
| Fix #1 fast-sweep duration | 0.0067 s vs 0.299 s baseline (97.8% ↓) | fix1-refresh-target |
| Fix #2 catch-up (0.75 s) | 0.283/0.377/0.482 rad/s @ drift 0.155/0.2/0.25 | fix2-catchup-and-softstart |
| Fix #2 mis-tuned (0.4 s) | 0.707 rad/s — WORSE than baseline | fix2-catchup-and-softstart |
| Soft-start (1.0 s ramp) | 0.420/0.485/0.542 rad/s @ drift 0.155/0.2/0.25 | fix2-catchup-and-softstart |
| Final tracking error, all fixes | ≤ 0.0046 rad (~0.26 deg) | fix2-catchup-and-softstart |
| FPC set_value() calls before first cmd | 0 (architecturally incapable) | fpc-switch-probe |
| Installed ur_robot_driver | 2.13.2; ros2_controllers 2.53.1 | fpc-switch-probe |
| FPC-jump discriminator | ≤0.625 rad/s = slew; >1 rad/s single-cycle = FPC jump | all |

## 6. Artifacts Produced

**sweep-signature**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/sweep-signature.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/sweep-signature.png`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/sweep-signature_driver.py`

**fix1-refresh-target**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix1.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix1_run_script.py`

**fix2-catchup-and-softstart**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix2_sim.py`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix2.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix2.png`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix2_metrics.json`

**data-drift-distribution**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/data-drift-distribution_analyze.py`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/data-drift.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/data-drift_windows_session_20260703_165530.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/data-drift_windows_session_20260703_171323.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/data-drift_hist.png`

**instack-real-bridge**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack_probe.py`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack_commands.csv`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack_snap_plot.png`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack_bridge_stdout.log`
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack-summary.txt`

**fpc-switch-probe**
- `/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fpc-switch-probe.md`
