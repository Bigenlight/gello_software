# GELLO -> UR7e "startup snap" — experiment & validation index

This directory holds the diagnosis, experiments, and hardware-free validation
behind the **convergence-gated handover** fix for the GELLO -> UR7e teleop
"startup snap" (one fast unprompted arm motion right after the move-to-start
handshake switches to `forward_position_controller`).

**Everything here was produced with NO real robot and NO real GELLO** — pure-math
harnesses, the REAL bridge/handshake nodes run against synthetic ROS inputs, a
mock `ros2_control` stack, and recorded-session analysis. The real-robot test is
still **pending** (see caveats at the bottom).

The code the artifacts validate lives in
`../../src/ur_gello_bringup/` (nodes `gello_move_to_start_node.py`,
`gello_ur_bridge_node.py`; launch `ur7e_gello_real.launch.py`; config
`config/ur7e_gello.yaml`).

---

## Read in this order

| # | File | What it is / proves |
|---|------|---------------------|
| 1 | `../replays/STARTUP_JERK_DIAGNOSIS.md` | **Diagnosis** (lives one dir up in `replays/`). Root-cause narrative: a frozen move-to-start target opens a position gap `G` at handover that the streaming bridge closes as a bounded, clamp-limited catch-up sweep. |
| 2 | `EXPERIMENT_RESULTS.md` | **Headline verdict.** Confirms the mechanism empirically, validates the pure-math harness against the REAL bridge to ~0.27%, rules the FPC-switch competing cause OUT, and ranks the candidate fixes. |
| 3 | `fpc-switch-probe.md` | **RANK1 competing cause ruled out.** Static analysis (exact installed versions) showing the `scaled_joint_trajectory_controller` -> `forward_position_controller` switch cannot itself inject motion: FPC `on_activate()` only clears the RT buffer; `ur_robot_driver` 2.13.2 reseeds `position_commands = actual` at the switch. |
| 4 | `SOLUTION_PROPOSAL.md` | **The chosen design** (convergence-gated chase loop L1 + pre-spawned paused bridge / resume L2), layered, with the alternatives table. |
| 5 | `SOLUTION_REVIEW.md` | **Adversarial review** of the proposal (independent re-check of every file:line citation). Verdict: ENDORSE WITH CHANGES; lists the concrete defects that were then folded into the implemented node/bridge/launch. |

---

## Analysis scripts + their data (mechanism characterization)

These established that gap `G` is real, sized it, and validated the 0.625 rad/s
slew-catch-up model against the actual bridge code.

| Script | Data / plot it produces | Proves |
|--------|-------------------------|--------|
| `data-drift-distribution_analyze.py` | `data-drift.csv`, `data-drift_hist.png`, `data-drift_windows_session_20260703_*.csv` | The passive leader is a **moving target**: worst-joint hold-still drift over the quietest 5 s windows is 0.155–0.25 rad (no 5 s window under ~0.15 rad across 2113 windows) — i.e. a non-trivial `G` always exists at handover. |
| `../../../scripts/sim_bridge_snap.py` | (offline, JSON to stdout) | Faithful offline sim of `gello_ur_bridge._on_timer` that **imports the REAL `_OneEuro` filter** from the node. Baseline model of the post-handoff sweep with no ROS/robot. |
| `instack_probe.py` | `instack_commands.csv`, `instack_snap_plot.png`, `instack-summary.txt`, `instack_bridge_stdout.log` | **In-stack REAL-bridge probe**: feeds the real node synthetic `/joint_states` + `/gello/joint_states` and measures the published slew. Matched the harness to ~0.27% (0.6267 vs 0.625 rad/s); confirmed zero-jump seed and the exact per-cycle `max_step_rad` clamp. |
| `sweep-signature_driver.py` | `sweep-signature.csv`, `sweep-signature.png` | Characterizes the fast-sweep signature (saturated slew from a dead stop) so it can be told apart from ordinary teleop motion in recorded data. |

---

## Fix experiments (candidate fixes, ranked in EXPERIMENT_RESULTS.md)

| Script | Data / plot | Fix evaluated |
|--------|-------------|---------------|
| `fix1_run_script.py` | `fix1.csv` | **Fix #1** — re-latch the move-to-start target to the LIVE gello pose (shrinks `G` ~3x). Biggest single win, but does not fully eliminate. |
| `fix2_sim.py` | `fix2.csv`, `fix2.png`, `fix2_metrics.json` | **Fix #2** — short catch-up trajectory + **soft-start** time-ramped clamp. Copies the validated `simulate()` math from `scripts/sim_bridge_snap.py`. Shows catch-up must be **duration-sized from the measured gap** or it degrades. |
| `bridge_softstart_test.py` | (stdout JSON) | Confirms the **REAL bridge's soft-start ramp**: with a constant 0.3 rad gap the per-cycle slew ramps up over `soft_start_s` instead of jumping straight to the full `max_step_rad` clamp. |

The final design combines these: a **convergence-gated chase** in
`gello_move_to_start` (re-reads the live leader, sizes each catch-up by the
measured gap, hands over only when `|live gello - actual| <= chase_tol` sustained)
plus the always-on **soft-start** slew ramp in `gello_ur_bridge` on every (re)seed.

---

## End-to-end validation of the implemented fix

### Convergence-gated handshake (hardware-free, subprocess node + mock robot)

| File | Role |
|------|------|
| `mock_handshake_test.py` | Runs the **REAL modified `gello_move_to_start`** as a subprocess against a mock FollowJointTrajectory server + list/switch_controller services, with a scripted leader. Three scenarios: `quiet` (holds -> SWITCH, tiny gap), `abrupt` (0.4 rad mid-approach jump -> SWITCH only AFTER it settles, >=2 chase goals), `never_settle` (oscillates forever -> NO SWITCH, timeout). Proves the gate hands over only on sustained convergence and fails safe otherwise. |

### L2 paused-bridge -> resume handoff (REAL `ros2_control` mock stack)

`test_l2_resume.sh` is the end-to-end test of the launch-level L2 change: the
bridge is pre-spawned `start_paused:=true` and must publish **nothing** on
`/forward_position_controller/commands` until `gello_move_to_start`
(`resume_bridge:=true`) calls `~/resume` **after** the STRICT controller switch.
Runs against the mock ur_control stack (`ROS_DOMAIN_ID=62`).

Helpers used by the real-stack scenario runs:

| Helper | Role |
|--------|------|
| `gello_pub.py` | Scripted `/gello/joint_states` publisher (scenarios `quiet` / `abrupt` / `never_settle`) for the real mock stack. |
| `js_recorder.py` | Records `/joint_states` to CSV for a fixed duration and reports peak per-joint speed (verify the post-switch window has no snap). |
| `count_topic.py` | Counts messages on `/forward_position_controller/commands` for N s (robust `ros2 topic echo` replacement — used to prove the paused bridge publishes 0). |
| `stream_analyze.py` | Analyzes a `stream_*.csv` for snap-relevant metrics (raw peak, 50 ms-window peak, excursion) to distinguish a real catch-up SWEEP from jitter. |
| `../../../scripts/analyze_replay.py` | Deterministic trajectory metrics (motion range, peak speed, jerk proxy, discontinuity/NaN counts) from recorded CSVs; JSON to stdout. Also reports metrics after accel-limited smoothing. |

### Run outputs (logs + captured streams)

Per-scenario captures from the real mock-stack runs:

- `run_{quiet,abrupt,never_settle}.log` — `gello_move_to_start` handshake output per scenario.
- `stack_{quiet,abrupt,never_settle}.log` — full mock `ros2_control` stack log per scenario.
- `gellopub_{quiet,abrupt,never_settle}.log` — scripted leader output.
- `bridge_{quiet,abrupt}.log` — bridge output.
- `stream_{quiet,abrupt}.csv` — recorded `/joint_states` during post-switch streaming (feed to `stream_analyze.py`).
- `l2_resume_{stack,mts,bridge,gello}.log` — outputs of `test_l2_resume.sh`.
- `ur_control_sim.log` — mock ur_control / `ros2_control` bring-up log.

---

## Caveats (real hardware still pending)

- **Everything here is hardware-free.** No `stream_*.csv` / handshake log was
  captured on a real UR7e or real GELLO. Treat the numbers as model-validated,
  not field-validated.
- **`chase_tol` (0.025) must exceed the real JTC steady-state arrival error** or
  the convergence gate can livelock — the mock arrives exactly, real HW may not.
  Calibration step: record one rosbag of `/joint_states` + the controller
  commands across a real handshake, read off the steady-state error, and set
  `chase_tol` (and possibly loosen `arrival_tolerance`) above it.
- The honest guarantee is **not** "zero snap": handover happens only once the
  follower has caught the LIVE leader within `chase_tol`, sustained; any residual
  is closed by the soft-started, rate-limited slew (`<= max_step_rad * rate`). A
  moving/never-settling operator **delays** handover rather than causing a snap.
