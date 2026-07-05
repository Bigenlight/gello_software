# Startup "snap" after move-to-start handshake — root-cause diagnosis

Synthesized from 6 independent investigations (frozen-target, bridge-handoff, data-drift,
switch-timing, bridge-seeding, red-team) + independent re-verification of every load-bearing
file:line and re-measurement of the CSV data on 2026-07-05.

---

## 1. Root cause (confidence: HIGH)

**A compound mechanism: (A) the move-to-start target is frozen at first sight of the GELLO
and ages while the operator waits + during the 5 s move, so a real position gap exists at
handover; (B) the streaming bridge then closes that gap at exactly its slew ceiling of
0.625 rad/s — the fastest speed the arm ever moves in teleop — producing one brief,
full-speed catch-up sweep that reads as a "snap", after which tracking is normal.**

Causal chain, end to end:

1. **Target frozen at t=0.** `gello_move_to_start_node.py:249-251` latches `_gello_target`
   only while it is `None` and never refreshes it; the live pose `_gello_latest` (line 248)
   is unused in the default `"gello"` mode. Critically, `run()` captures the target
   (`_wait_for_gello_target`, line 616) **before** `_wait_for_source_active` (line 621), so
   the freeze precedes the wait for the operator to press Play on the pendant (up to
   `activation_timeout` = 120 s). The frozen pose then drives a single 5 s trajectory
   (`trajectory_duration: 5.0`, `ur7e_gello.yaml:78`; sent at line 650).
2. **Gap accumulates.** While the arm moves, the human holding the passive GELLO drifts
   (tremor, fatigue, repositioning) plus Dynamixel encoder noise. `arrival_tolerance: 0.05`
   rad/joint (`ur7e_gello.yaml:79`, applied at `gello_move_to_start_node.py:347`) adds up to
   0.05 rad/joint of trajectory-controller slack on top.
3. **Handover extends the gap window.** The STRICT switch to `forward_position_controller`
   happens (`gello_move_to_start_node.py:653`, switch at ~597), move_to_start **exits**, and
   only then does the launch spawn the bridge as a **fresh process**
   (`ur7e_gello_real.launch.py:399`, `OnProcessExit`). Bridge cold-start (rclpy init,
   discovery, first `/joint_states`) adds ~0.5-2 s during which the arm holds the frozen
   arrival setpoint (FPC inherits the shared position command interface) and the GELLO keeps
   drifting.
4. **Bridge closes the gap at max speed.** The bridge seeds its first command from the
   robot's **actual** `/joint_states` pose, not the GELLO (`gello_ur_bridge_node.py:306-327`)
   — zero jump at seed — then every cycle slews toward the live (one-euro-filtered) GELLO
   pose under an **unconditional** per-joint clamp of `max_step_rad = 0.0025` rad/cycle
   (`gello_ur_bridge_node.py:353-359`, `ur7e_gello.yaml:59`) at 250 Hz (`yaml:70`)
   = **0.625 rad/s (~36 deg/s)**. A gap G closes in G/0.625 s: a 0.1-0.3 rad gap is a
   0.16-0.5 s sweep at the arm's maximum sustained teleop speed. That is the observed
   "one sudden fast motion, then normal."

**Why "sometimes":** the gap magnitude varies run to run — how long the operator took to
press Play after the GELLO pose was latched, how still they held the leader, encoder noise
in the single captured first sample, plus 0-0.05 rad arrival slack. Small gap → invisible;
0.1-0.3 rad gap → a visible full-speed snap.

**Measured numbers (re-verified today):**
- Quietest genuine 5 s window, worst-joint peak-to-peak GELLO drift: **0.25 rad (14.3°)**
  in `session_20260703_165530`, **0.155 rad (8.9°)** in `session_20260703_171323`
  (`gello_joint_states.csv`, ~30 Hz). These are active-teleop upper bounds; realistic
  hold-still drift is ~0.05-0.15 rad over the capture-to-handover window — still 1-4x the
  arrival tolerance and plainly visible at 0.625 rad/s.
- Command stream slew clamp holds empirically: max per-cycle |Δcmd| = **0.0025 rad exactly**
  over 10,751 samples (171323); 0.0074 rad worst in 165530, explained by 2-3-cycle dropped
  samples (still ≤ clamp per elapsed cycle). 250 Hz confirmed (median dt = 4.0 ms).
- Note: both CSV sessions are replay/steady-state diagnostics — they confirm the clamp and
  drift magnitudes but do **not** contain the handshake instant itself.

**Investigator disagreement resolved:** the data-drift lens computed a 12.5 rad/s snap from
the code **default** `max_step_rad = 0.05`; that default is overridden by
`ur7e_gello.yaml:59` to 0.0025 in every real launch, so 0.625 rad/s is correct. All six
lenses agree on the frozen-target premise; the four that checked the bridge agree it cannot
emit a per-cycle jump above the clamp.

## 2. Verdict on the user's hypothesis: **IN — premise confirmed, mechanism corrected**

- **Confirmed:** capture-once-at-t=0 (`gello_move_to_start_node.py:249`), 5 s move to the
  frozen pose (line 650), drift of 0.05-0.25 rad by handover (CSV numbers above), and the
  robot closing that accumulated gap when streaming starts. The user even underestimated the
  window: the freeze happens **before** the Play-press wait (line 616 vs 621), so the target
  can age far longer than 5 s.
- **Corrected:** the gap is NOT closed "suddenly"/unboundedly. The bridge seeds from the
  arm's actual pose (`gello_ur_bridge_node.py:320`) and hard-clamps every cycle
  (`:353-359`), so the closure is a rate-limited sweep at 0.625 rad/s. It *reads* as a snap
  because 0.625 rad/s is the maximum speed teleop ever produces and it fires unprompted right
  after 5 s of smooth slow motion. It is a UX defect, not an unbounded/safety jerk.

## 3. Competing causes and how to tell them apart

| Candidate | Status | Discriminator |
|---|---|---|
| **FPC handover window** (red-team RANK1): `forward_position_controller` active but bridge not yet spawned (`launch:399`); if the driver/controller latched a stale command-interface value at activation, one **non-interpolated, non-slew-limited** jump. | Plausible but unconfirmed; code suggests FPC inherits the arrival setpoint via the shared position interface, and the bridge seed is zero-jump. Residual uncertainty lives in the installed `ros2_controllers`/`ur_robot_driver` `on_activate` behavior, not in this repo. | **Measure the snap's joint speed.** If |qd| ≤ ~0.625 rad/s and it lasts G/0.625 s → slew catch-up (root cause above). If it completes in 1-2 control cycles (>1 rad/s spike) → handover window. A rosbag of `/joint_states` + `/forward_position_controller/commands` across the switch decides it in one run. |
| **2π/branch wraparound** (q6 straddles +π: 2.543-3.201 rad in 165530). | Ruled out for THIS symptom: a wrapped target would corrupt the smooth 5 s move itself, not produce a post-arrival snap; and any wrap gap at handover would still be slew-clamped. | Snap timing: wraparound shows *during* move-to-start; the reported snap is *after* arrival. |
| **Single Dynamixel encoder spike in the one captured sample** (spikes up to 2.9° observed in 165530). | Contributor, not root cause: it inflates the frozen-target error the same way drift does, and is closed by the same clamped sweep. | Indistinguishable from drift at handover; fixed by the same target-refresh fix. |
| **Protective stop / velocity fault.** | Ruled out: "then behaves normally" — a UR protective stop would halt, not self-recover. | Pendant log would show it. |

## 4. Fix (ranked)

1. **Refresh the target to the live pose before moving** — `gello_move_to_start_node.py`,
   `run()`: after `_wait_for_source_active()` returns (line 621) and immediately before
   `_send_trajectory(...)` (line 650), re-latch `self._gello_target = list(self._gello_latest)`.
   One-line change; removes the entire Play-press-wait aging (potentially minutes), leaving
   only ~5 s of drift. Highest value/risk ratio.
2. **Verify alignment at arrival, catch up before switching** — same node: after the
   trajectory succeeds, compare `self._gello_latest` to the arrived pose per joint; if any
   joint exceeds a small threshold (e.g. 0.03 rad), send one short (~0.5-1 s) catch-up
   trajectory to the refreshed live pose, then switch. This is init_align's guarantee applied
   to `"gello"` mode and removes the visible sweep entirely.
3. **Shrink the cold-start window** — stop spawning the bridge via `OnProcessExit`
   (`ur7e_gello_real.launch.py:399`); start `gello_ur_bridge` at launch, paused, and flip a
   "streaming enabled" flag (it already has pause/resume services,
   `gello_ur_bridge_node.py:365`) when the handshake succeeds. The bridge then seeds and
   starts closing the residual gap the instant FPC activates, and no unfed-FPC window exists
   (also neutralizes the RANK1 competing cause).
4. **Operational alternative:** `start_mode: "init_align"` (`ur7e_gello.yaml:92`,
   `gello_move_to_start_node.py:624-647`) — structurally gap-free (handover gated on live
   alignment within `alignment_tolerance` = 0.2 rad), at the cost of operator interaction.
5. **Minor:** tighten `arrival_tolerance` (`ur7e_gello.yaml:79`) from 0.05 toward 0.02 rad
   to cut the trajectory-slack contribution.
6. **Do NOT raise `max_step_rad`** to make catch-up "snappier" — it is the safety bound
   (yaml:44-58 documents the coalescing math); raising it converts this cosmetic sweep into
   a genuine velocity-limit risk.

## 5. Synthetic test (this PC, no GELLO / no robot)

**Test A — pure-math repro of the mechanism (no ROS needed):**
Replicate the bridge loop (one-euro seed + `max_step_rad` clamp, `gello_ur_bridge_node.py:329-362`,
params from `ur7e_gello.yaml`) in a ~40-line script:
1. Build a synthetic 30 Hz GELLO stream: hold pose `q0` with tremor (e.g. Gaussian σ=0.003 rad
   + 0.00153 rad Dynamixel quantization) for 5 s, drifting linearly to `q0 + G` (sweep
   G ∈ {0.05, 0.15, 0.30} rad on one joint — the measured range).
2. "Move-to-start" = arm arrives at frozen `q(t=0)`; seed the bridge state from that arrival
   pose (mimics seeding from `/joint_states`).
3. Run the 250 Hz loop against the live (drifted) stream; record the command trace.
4. **Assert:** max per-cycle |Δcmd| == 0.0025 rad; catch-up duration ≈ G/0.625 s
   (e.g. G=0.15 → ~0.24 s at a constant 0.625 rad/s ramp). This is the "snap" signature.
5. Re-run with fix #1/#2 applied (target = live pose at handover): G collapses to the ~5 s
   drift only / ~0, and the ramp disappears — confirming the fix.

**Test B — in-stack repro (sourced ROS 2, still hardware-free):**
Launch `gello_publisher` replaced by `fake_gello_node` modified (or remapped through a small
relay) to hold-then-jump as above, plus a mock `/joint_states` publisher that echoes the last
`/forward_position_controller/commands` (a 10-line node standing in for the UR), then run
`gello_move_to_start` (against a mocked FollowJointTrajectory server) → `gello_ur_bridge`.
Record `command.csv`-style output and apply the same assertions. This additionally exercises
the OnProcessExit cold-start window: timestamp FPC-activation vs first bridge publish to
bound the unfed-FPC interval for the RANK1 discriminator.

**Real-hardware discriminator (when the robot is available):** rosbag
`/joint_states`, `/forward_position_controller/commands`, and controller_manager switch
events across one handshake. One trace distinguishes slew catch-up (≤0.625 rad/s ramp,
starts at first bridge publish) from an FPC activation jump (single-cycle step at switch
time, before the first bridge command).
