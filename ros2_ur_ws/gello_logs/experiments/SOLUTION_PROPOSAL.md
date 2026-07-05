# Startup-Snap Solution Proposal — Convergence-Gated Handover (refined, layered)

Reviewer pass 2026-07-05. Grounded in a fresh read of:
`gello_move_to_start_node.py`, `gello_ur_bridge_node.py`, `ur7e_gello_real.launch.py`,
`ur7e_gello.yaml`, `STARTUP_JERK_DIAGNOSIS.md`, `EXPERIMENT_RESULTS.md`.

---

## 1. Problem restated; root cause confirmed (with two corrections of emphasis)

**Symptom:** right after the ~5 s move-to-start trajectory arrives and control STRICT-switches
to `forward_position_controller`, the arm sometimes does one fast unprompted motion, then
tracks normally.

**Root cause — CONFIRMED as diagnosed.** Verified in code:

- `gello_move_to_start_node.py:249-251` latches `_gello_target` exactly once, the first
  complete `/gello/joint_states` message; `run()` does this (`:616`) **before**
  `_wait_for_source_active()` (`:621`, pendant-Play wait, `activation_timeout` up to 120 s),
  then sends one 5 s trajectory to that frozen pose (`:650`). The live pose `_gello_latest`
  (`:248`) is unused in `"gello"` mode.
- The passive leader is a **moving target** (measured 0.155–0.25 rad worst-joint drift over
  the quietest 5 s windows; no 5 s window under ~0.15 rad in 2113 windows), so a gap `G`
  exists at handover.
- The bridge seeds from the robot's **actual** pose (`gello_ur_bridge_node.py:312-327`,
  zero jump) and then closes `G` under the unconditional clamp
  `max_step_rad(0.0025) × publish_rate_hz(250) = 0.625 rad/s`
  (`:353-359`, `ur7e_gello.yaml:59,70`) — one bounded full-speed sweep of ~`G/0.625` s.
  In-stack test matched the model to 0.27%. Bounded, never near the 3.14 rad/s
  driver limit: a **predictability/UX defect, not a safety jerk**.
- Competing cause (switch injects a jump) is correctly ruled out: FPC `on_activate` only
  clears the RT buffer; `ur_robot_driver` 2.13.2 reseeds `position_commands = actual` at the
  mode switch — a stationary hold.

**Two points the prior analysis under-weighted:**

1. **The gap window is larger than the frozen-target window.** After the switch,
   `gello_move_to_start` *exits*, and only then does the launch spawn the bridge as a fresh
   process (`ur7e_gello_real.launch.py:399-404`, `OnProcessExit`, spawning at `:384`).
   Bridge cold start (rclpy init + discovery + first `/joint_states` + first GELLO msg) is
   ~0.5–2 s during which the leader keeps moving and the arm holds. At the measured drift
   rate (~0.03–0.05 rad/s while "holding still") that window alone contributes
   **0.03–0.10 rad** — the same order as the visible-snap threshold. Any fix that
   guarantees alignment only *at the switch instant* but leaves this window open **fails to
   guarantee alignment at the bridge's first command**, which is the instant that actually
   matters. This is why Fix#1 alone measured a residual pegged-clamp episode.
2. **The invariant to enforce must be stated precisely.** Once streaming, a fast follower
   motion in response to a fast *live* leader motion is *correct teleop*, not a snap. The
   defect is exclusively **unprompted** motion: closing a gap that accumulated while not
   streaming. So the target invariant is:

   > **At the instant the bridge publishes its first (and first post-resume) command,
   > `|leader_live − follower_actual| ≤ ε` per joint, with the leader demonstrably
   > quasi-still over the preceding dwell window.**

   After that instant, every follower motion is a rate-limited response to live leader
   motion — by construction, never a snap.

The user's critical insight is correct and is the decisive design constraint: **any
single-snapshot scheme (including Fix#1) races a moving target and loses whenever the
operator moves deliberately during the approach.** Tuning snapshot timing cannot fix a
category error; the handover must be **gated on a measured, sustained condition**, not on
a captured pose.

---

## 2. Evaluation of the convergence-gated handover candidate

**Mechanism:** loop { read live GELLO pose → short catch-up trajectory toward it → on
arrival check per-joint `|live − actual|` } ; switch to streaming only when the error is
within a tight tolerance **sustained for a dwell time**.

### Verdict: correct core mechanism — GOOD, and *nearly* sufficient. It is the only
candidate on the table that enforces the invariant above rather than approximating it.
But as stated it has five gaps that must be engineered out:

| # | Failure mode / edge case | Severity | Resolution |
|---|---|---|---|
| A | **Post-gate window re-opens the gap.** Gate passes → switch (~0.1–0.5 s) → node exits → `OnProcessExit` spawns bridge (~0.5–2 s cold start) → first command. Leader drift alone re-accumulates 0.03–0.10 rad; an abrupt operator move re-creates the full snap. **The gate guarantees the invariant at the wrong instant.** | HIGH — defeats the fix | Pre-spawn the bridge **paused** at launch; `move_to_start` resumes it via the existing `~/resume` service (`gello_ur_bridge_node.py:373`) immediately after the switch. Window shrinks to ~0.15 s (switch + service call + one 4 ms timer tick); drift over 0.15 s < 0.01 rad. Resume already re-seeds from actual pose (`:379-380`) — zero jump by construction. |
| B | **Never settles.** Operator keeps moving the leader → the gate never passes. | MEDIUM — availability, not safety | This is *correct fail-safe behavior* (streaming must not start), but needs UX: throttled prompt "hold the leader still to start teleop", iteration cap + `converge_timeout` → abort with the node's existing fail-safe exit (no switch, exit code 1, launch refuses to start the bridge — `ur7e_gello_real.launch.py:388-397`). Never silently loop forever without telling the operator why. |
| C | **Dynamixel encoder spikes** (observed up to 2.9° ≈ 0.05 rad single-sample) vs a tight ε (~0.02–0.03 rad). A spike during dwell falsely resets the gate (annoying, retries); a spike can never falsely *pass* a properly designed dwell but naive per-sample checking makes the gate flaky. | MEDIUM — UX flakiness | Evaluate dwell on a **median-of-last-N (N≈5)** per-joint error at the 30 Hz GELLO cadence. Median rejects single-sample spikes in both directions. ε must stay > quantization (0.0015 rad) + tremor band; 0.02–0.03 rad is right. |
| D | **Chase-trajectory sizing** — the Fix#2 hazard (fixed 0.4 s catch-up measured *worse* than baseline, 0.707 rad/s). A one-point `FollowJointTrajectory` goal with zero end velocity is interpolated by the JTC as a spline whose peak velocity ≈ 1.5–2 × gap/duration. | MEDIUM | Compute each iteration's duration from the **measured** gap: `T = clip(gap_max / v_budget, T_min, trajectory_duration)` with `v_budget ≈ 0.3–0.4 rad/s` (below the 0.625 rad/s stream ceiling → the approach is never faster than steady-state teleop) and `T_min ≈ 0.75 s`. This is Fix#2 done robustly: sized per-iteration from live measurement, and *verified* afterwards by the gate instead of trusted. |
| E | **Chase-loop stability / JTC re-goal behavior.** Rapid goal preemption of the scaled JTC mid-motion has replan edge cases. | LOW if designed sequentially | Make the loop **strictly sequential**: send goal → block on SUCCEEDED (the node already does exactly this, `_send_trajectory` `:316-385`) → measure → dwell or re-chase. No preemption ever occurs, so no re-goal edge cases. Convergence: each iteration's residual ≈ leader motion during T; a still leader converges in ≤2 iterations; a moving leader (v·T > ε) never converges — which is the *desired* outcome (gate holds). Bounded velocity, zero-velocity endpoints ⇒ no oscillation/divergence possible. |

Other checks:
- **ros2_control / driver constraints:** the whole loop runs while
  `scaled_joint_trajectory_controller` is active; the switch itself is unchanged and proven
  jump-free (driver reseed). Publishing to `/forward_position_controller/commands` while FPC
  is inactive (paused-bridge architecture never does this anyway, but even if resumed early)
  is inert: FPC's `on_activate` clears the RT buffer.
- **Latency:** 30 Hz GELLO sampling + serial latency is irrelevant inside a dwell that
  requires stillness; ε=0.025 rad at 30 Hz resolves leader speeds down to ~0.05 rad/s.
- **Residual theoretical hole:** an abrupt operator move inside the ~0.15 s post-gate window
  cannot be prevented (the leader is uncontrolled). Consequence: residual ≤ 0.15 s × leader
  speed, closed at ≤ 0.625 rad/s — and since the *user is moving* at that moment, the
  follower's fast response reads as teleop, not as an unprompted snap. With layer L3 below
  the onset is additionally ramped. This is the irreducible minimum for any architecture in
  which the leader cannot be commanded.

**Conclusion:** the convergence gate is the right mechanism, but it is only sufficient when
paired with closing the cold-start window (A). Gate without (A) = Fix#1's residual all over
again; (A) without gate = snapshot race. Together they enforce the invariant at the correct
instant for **all** operator behaviors.

### Alternatives weighed (and why they lose as primary mechanism)

| Alternative | Verdict |
|---|---|
| **Bridge streams from t=0, no handshake move** (it already seeds-from-actual + rate-limits) | *Relocates* the snap: the whole approach from the power-on pose becomes one multi-second full-speed (0.625 rad/s) unprompted sweep along an uncontrolled joint-space line — strictly worse than the 5 s interpolated trajectory for large initial gaps, and streaming during operator setup is the wrong safety posture. Reject. |
| **Fix#1 (re-latch once, later)** | Measured 97.8% sweep-duration reduction but still a snapshot: loses to any deliberate move during the 5 s approach (the user's case). Insufficient alone; **subsumed** by the gate (its first iteration *is* Fix#1). |
| **Soft-start ramp only** | Fixes onset abruptness, not the sweep (peaks 0.42–0.54 rad/s measured). Insufficient alone; excellent cheap defense-in-depth layer. |
| **One-shot catch-up trajectory (Fix#2)** | Correct instinct, fragile execution: unverified, mis-sizing measured *worse* than baseline (0.707 rad/s). Subsumed by the gate loop (sized-from-measurement + verified + repeated). |
| **Stillness detection only (no alignment check)** | Leader can be still but displaced (moved during approach, then stopped early) → full gap remains → snap. Must be stillness **and** alignment ⇒ that conjunction *is* the convergence gate. |
| **init_align (operator-gated)** | Structurally sound but `alignment_tolerance: 0.2` rad (`ur7e_gello.yaml:99`) still permits a 0.32 s pegged-clamp sweep, and it costs operator interaction on every start. Keep as optional mode; tighten its tolerance; the gate is "init_align automated, tightened, and pointed at the live leader instead of a fixed pose". |
| **Accept the bounded slew, fix perception (announce/beep)** | Honest but fails the requirement; keep the announcement as a UX layer. |
| **Continuous JTC tracking / MoveIt Servo rewrite** | Solves it but is a large-risk rewrite of a working, verified streaming path for zero additional guarantee over gate+prespawn. Reject. |
| **Raise `max_step_rad` for a quicker catch-up** | Never: it is the coalescing-safe velocity bound (`ur7e_gello.yaml:44-58`); raising it converts a cosmetic defect into a protective-stop risk. |

---

## 3. Recommended solution (ranked, layered)

### L1 — NECESSARY: convergence-gated handover in `gello_move_to_start` ("gello" mode)

Replace the single frozen-target trajectory (`gello_move_to_start_node.py:648-651`) with a
sequential converge loop:

```
_wait_for_source_active()                      # unchanged (Play gate)
loop (≤ max_converge_iters, ≤ converge_timeout):
    target = list(self._gello_latest)          # LIVE pose, re-read every iteration (:248)
    gap    = max_i |target[i] − actual[i]|     # actual from a new /joint_states sub
    T      = clip(gap / catchup_speed_budget, catchup_min_duration, trajectory_duration)
    _send_trajectory(target, "live GELLO pose", duration=T)   # blocks until SUCCEEDED
    # dwell: sample at GELLO cadence for handover_dwell_s;
    # err_i = median of last 5 samples of |gello_latest[i] − actual[i]|
    if all joints err_i ≤ handover_tolerance for the whole dwell → break (converged)
    else → log which joints / how far (reuse _alignment_report style :395-405), re-loop
on timeout/iters exhausted → throttled operator prompt; then abort fail-safe (exit 1, no switch)
_switch_controllers()                          # unchanged (:574-611)
call /gello_ur_bridge/resume                   # L2 below
```

Concrete edits:
- **`gello_move_to_start_node.py`**
  - Add a `/joint_states` subscription tracking `self._actual_pose` (mirror of
    `gello_ur_bridge_node.py:222-224, 274-284`).
  - Give `_send_trajectory` (`:316`) a per-call `duration` argument (today it hardcodes
    `self.trajectory_duration`, `:333-335`).
  - New parameters (declare near `:117-122`):
    `handover_tolerance: 0.025` rad, `handover_dwell_s: 0.4`,
    `catchup_speed_budget: 0.35` rad/s, `catchup_min_duration: 0.75` s,
    `max_converge_iters: 10`, `converge_timeout: 60.0` s.
  - Replace the `else` branch of `run()` (`:648-651`) with the loop above. `_gello_target`
    (`:249-251`) becomes unused in "gello" mode (or keep as a sanity log only).
- **`ur7e_gello.yaml`**: add the six parameters under `gello_move_to_start` (`:74`).

**Invariant guaranteed at the switch:** for the `handover_dwell_s` window immediately
preceding the switch, every joint satisfied `|leader_live − follower_actual| ≤ 0.025` rad
(median-filtered), which also bounds leader speed to ≲ 2ε/dwell ≈ 0.125 rad/s. Worst-case
unprompted post-switch motion ≤ ε, closing in ≤ ε/0.625 ≈ **40 ms** — sub-perceptual.
If the operator moves abruptly during the approach, the loop simply re-chases and the gate
holds: **no user behavior can produce a snap; the failure direction is "won't start yet",
never "moves unexpectedly".**

### L2 — NECESSARY: close the cold-start window (bridge pre-spawned, paused)

Without this, L1's invariant expires during the 0.5–2 s `OnProcessExit` bridge cold start
and drift re-opens a visible gap (§2-A).

- **`gello_ur_bridge_node.py`**: new parameter `start_paused` (default `false` for
  back-compat); when true, initialize `self._paused = True` (`:205`). The existing
  `~/resume` (`:373-389`) already re-seeds from actual pose — no other change.
- **`ur7e_gello_real.launch.py`**: start `bridge_node` (`:292-297`) at launch (e.g. in the
  same 8 s `TimerAction` as move_to_start, `:368-371`) with `start_paused: true`; remove it
  from `_on_handshake_exit` (`:384`), keeping the gripper nodes there. On handshake failure
  the bridge stays paused forever — fail-safe preserved (and doubly so: FPC never activates).
- **`gello_move_to_start_node.py`**: after `_switch_controllers()` succeeds (`:653`), call
  `/gello_ur_bridge/resume` (Trigger client, `wait_for_service` ≤ 10 s; on failure log
  ERROR — arm holds position, operator can call resume manually).

Result: gate-pass → first bridge command ≈ switch (~0.1 s) + service (~10 ms) + one
250 Hz tick (4 ms) ≈ **0.15 s**; drift re-accumulation < 0.01 rad. This also permanently
neutralizes the residual RANK1 "unfed-FPC window" uncertainty.

### L3 — NICE-TO-HAVE: soft-start slew ramp on every (re)seed in the bridge

Defense-in-depth for whatever residual/reopened gap remains (including after operator
pause/resume with a moved leader — an *existing* snap path today, `:373-389`):
in `_on_timer`, scale the clamp `step` (`:330, :353-359`) by
`min(1, (now − seed_time)/soft_start_s)` with `soft_start_s ≈ 0.7` and a small floor
(e.g. 0.15) so motion is never fully frozen; record `seed_time` at the seed block
(`:320-326`). ~6 lines, one parameter. Measured effect: onset becomes a ramp, peak
0.42–0.54 rad/s for legacy-size gaps — irrelevant after L1+L2 except as insurance.

### L4 — NICE-TO-HAVE: consistency tightening

- `ur7e_gello.yaml:99` `alignment_tolerance: 0.2 → 0.05` rad (init_align mode has the same
  defect class; 0.2 rad = 0.32 s pegged sweep), and apply the same dwell check to its GATE 2.
- `ur7e_gello.yaml:79` `arrival_tolerance: 0.05 → 0.02` rad (trajectory slack feeds ε budget).
- Operator UX: log "CATCHING UP to leader (iter k, worst joint X.XX rad)" and
  "HOLD LEADER STILL to start" prompts; announce handover.

### Do NOT
- Raise `max_step_rad` (safety bound, `ur7e_gello.yaml:44-58`).
- Stream from t=0 / drop the trajectory phase (relocates the snap, §2 table).
- Ship a fixed-duration catch-up trajectory (measured worse-than-baseline when mis-sized).

---

## 4. Hardware-free validation plan (this PC, in-stack)

Extend the proven in-stack harness (`instack_probe.py` pattern, which validated the bridge
to 0.27%) into a closed-loop handshake rig:

1. **Mock robot** (~60 lines): a `FollowJointTrajectory` action server named
   `/scaled_joint_trajectory_controller/follow_joint_trajectory` that linearly (or cubically)
   interpolates to the goal over `time_from_start` while publishing `/joint_states` at
   250 Hz; after "switch", it instead echoes `/forward_position_controller/commands` into
   `/joint_states` (the existing echo trick). Mock
   `/controller_manager/{list_controllers,switch_controller}` services (list returns
   "active", switch returns ok and flips the echo mode).
2. **Scripted leader** on `/gello/joint_states` @ 30 Hz with tremor (σ=0.003 rad) +
   quantization (0.00153 rad), scenarios:
   - S1 baseline drift: quasi-still, 0.03 rad/s wander (regression of the original bug);
   - S2 **abrupt move during approach**: at t = 40% of the first trajectory, ramp one joint
     0.4 rad in 0.5 s, then still — the user's critical case;
   - S3 **never settles**: 0.3 rad, 0.2 Hz sinusoid forever;
   - S4 encoder spikes: 0.05 rad single-sample spikes at 1 Hz during dwell;
   - S5 pause/resume with leader moved 0.3 rad while paused (L3 check).
3. Run the **real modified nodes**: `gello_move_to_start` (L1) + `gello_ur_bridge`
   (`start_paused:=true`, L2/L3), production `ur7e_gello.yaml`. Record
   `/forward_position_controller/commands` + `/joint_states` + node logs to CSV.
4. **Assertions:**
   - A1 (invariant): at first bridge command, per-joint `|leader − actual| ≤ 0.03` rad in
     every converging scenario (S1, S2, S4, S5).
   - A2 (no unprompted sweep): zero intervals where command speed ≥ 0.9 × 0.625 rad/s for
     > 50 ms without leader motion ≥ 0.05 rad in the preceding 200 ms.
   - A3 (clamp): max per-cycle |Δcmd| ≤ 0.0025 rad, machine-exact (regression).
   - A4 (S2): ≥ 2 chase iterations observed; switch occurs only after the leader stops;
     A1/A2 hold.
   - A5 (S3): NO switch within `converge_timeout`; node exits 1; bridge never resumes;
     prompt logged. Fail-safe intact.
   - A6 (S4): spikes never *pass* the gate early (median filter), and dwell completes
     within ≤ 3 extra iterations (flakiness bound).
   - A7 (timing): gate-pass → first bridge command < 0.3 s (L2 window).
5. Parameter sweeps (ε × dwell × v_budget) on the existing pure-math harness
   (`fix2_sim.py` lineage) to confirm 0.025/0.4/0.35 before the in-stack run.

Real-hardware acceptance later: rosbag one handshake per scenario S1/S2; assert A1/A2 on
the bag (the discriminator recipe from `STARTUP_JERK_DIAGNOSIS.md` §5 applies unchanged).

---

## 5. Summary table

| Layer | Change | Files | Status |
|---|---|---|---|
| L1 converge-gate | chase loop + dwell + live target | `gello_move_to_start_node.py:648-651` (+ new sub, params, `_send_trajectory` duration arg `:333`) | **Necessary** |
| L2 close window | bridge pre-spawned paused + resume call | `gello_ur_bridge_node.py:205` (param), `ur7e_gello_real.launch.py:368-404`, move_to_start resume client after `:653` | **Necessary** |
| L3 soft-start | ramped clamp after every seed | `gello_ur_bridge_node.py:320-326, 353-359` | Nice-to-have |
| L4 tighten | init_align tol 0.05, arrival tol 0.02, UX logs | `ur7e_gello.yaml:79,99` | Nice-to-have |

The invariant L1+L2 jointly guarantee: **streaming never begins (and never resumes) unless
leader and follower agree within 0.025 rad per joint, sustained 0.4 s, re-verified within
~0.15 s of the first streamed command.** Under that invariant an unprompted fast motion is
impossible for any operator behavior; the only remaining fast motions are rate-limited
responses to live leader motion — i.e., teleop working as designed.
