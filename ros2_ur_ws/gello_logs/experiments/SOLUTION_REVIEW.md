# Adversarial Review of SOLUTION_PROPOSAL.md (convergence-gated handover)

Independent reviewer pass, 2026-07-05. Grounded in a fresh read of
`gello_move_to_start_node.py`, `gello_ur_bridge_node.py`, `ur7e_gello_real.launch.py`,
`ur7e_gello.yaml`, `STARTUP_JERK_DIAGNOSIS.md`, `EXPERIMENT_RESULTS.md`. I did NOT take the
proposal's citations on trust; every file:line it leans on was re-checked.

---

## VERDICT: ENDORSE WITH CHANGES

The core architecture is right and I found no competing design that beats it:

- **Root-cause reliance is correct.** Latch at `gello_move_to_start_node.py:249-251`, latched
  before the Play wait (`run()` :616 vs :621), one frozen 5 s trajectory (:650), bridge
  seed-from-actual (:312-327) + unconditional clamp (:353-359) at 0.0025 × 250 = 0.625 rad/s.
  All citations verified accurate, including the bridge `_paused` init (:205), `~/resume`
  re-seed (:373-389, clears state at :379-380), `_send_trajectory` hardcoded duration
  (:333-335), `OnProcessExit` spawn (`ur7e_gello_real.launch.py:373-404`, bridge at :384),
  and the yaml values (:59, :70, :79, :99). No citation errors found.
- **The two "under-weighted" findings are genuine and important.** (1) The cold-start window
  is real: the bridge is a fresh process spawned only after move_to_start exits; alignment
  guaranteed at the switch instant is guaranteed at the wrong instant. (2) The invariant
  restatement (unprompted motion is the defect; leader-driven fast motion is correct teleop)
  is the right frame and is what makes the design judgeable at all.
- **L1 (chase loop + tight gate + dwell) is the correct primary mechanism.** It is the only
  candidate that enforces a *measured, sustained* condition rather than trusting a snapshot,
  and it correctly makes the failure direction "won't start yet", never "moves unexpectedly".
  The alternatives table is sound; I independently agree with every rejection in it
  (stream-from-t=0 relocates the snap; stillness-only misses still-but-displaced; fixed
  catch-up measured worse at 0.707 rad/s; raising max_step_rad is a real safety bound per
  the coalescing math at yaml:44-58).
- **L2 (pre-spawn bridge paused, resume from move_to_start) is necessary and minimal.**
  I looked for a simpler variant (merge nodes; gate in the bridge alone) — the bridge cannot
  close a gap without the JTC, and merging is a bigger refactor for no additional guarantee.
  `start_paused` default-false preserves the mock/rviz launch paths. Endorsed as-is modulo
  R1 below.

However, the proposal **over-claims its invariant** and has **four concrete defects** that
must be fixed before implementation. None invalidates the architecture; all are reachable
states on the real stack.

---

## REQUIRED CHANGES

### R1 (HIGH) — The claimed invariant is false for two existing bridge paths; L3 must be
### promoted to NECESSARY and generalized, or the resume gate closed explicitly.

The summary claims: *"streaming never begins (and never resumes) unless leader and follower
agree within 0.025 rad per joint."* Nothing in L1+L2 enforces the "resumes" half:

1. **Manual `~/resume` with a misaligned leader.** The operator console can call
   `/gello_ur_bridge/resume` at ANY time with the leader ANYWHERE (that is its documented
   use, :373-389: "Keep clear if GELLO moved while paused"). L2 makes this service *more*
   load-bearing, not less. Result: the exact original snap (full-speed sweep of the
   accumulated gap) through a door L1 never guards. The proposal knows this path exists
   (it cites it as an "existing snap path today") yet leaves the mitigation (L3) as
   nice-to-have. That is inconsistent with the "ALL operator behaviors" success criterion.
2. **Staleness recovery — worse, L3 as specced never fires there.** Verified in code: when
   GELLO input goes stale, `_on_timer` returns early (:299-304) but `_filtered` /
   `_last_published` are RETAINED. When data resumes (serial hiccup, unplug/replug,
   publisher restart), the bridge does NOT re-seed — it slews from the last published pose
   toward the now-arbitrary live leader at the full 0.625 rad/s. There is **no seed event**,
   so L3's "ramp after every (re)seed" trigger never occurs on this path. An unplugged-and-
   moved leader mid-session reproduces the full snap today AND under the proposal as written.

**Required:** (a) promote L3 to NECESSARY; (b) redefine its trigger as "after any publishing
gap > staleness_timeout_s OR any re-seed", i.e. record `seed_time` also when recovering from
staleness (equivalently: on staleness expiry, clear `_filtered`/`_last_published` so recovery
re-seeds from actual, then soft-start); (c) preferably ALSO gate `_on_resume` on per-joint
`|_raw_target − _actual_pose| ≤ tolerance` — the bridge already holds both vectors — refusing
with a per-joint report (mirror `_alignment_report`, move_to_start :395-405) and offering the
existing override pattern. With (c), the resume path gets the same guarantee as startup
instead of merely a softened sweep.

### R2 (HIGH) — The catch-up sizing formula violates its own velocity budget for large gaps.

`T = clip(gap / 0.35, 0.75 s, trajectory_duration=5 s)`: the UPPER clip means any gap
> 0.35 × 5 = 1.75 rad is traversed faster than the budget — e.g. a wrist_3 wraparound /
branch gap of ~6.28 rad (q6 straddles +π in session 165530; the user's case (d)) gives a 5 s
trajectory at 1.26 rad/s *average*, and the JTC's cubic interpolation with zero endpoint
velocities peaks at ~1.5-2× the average → **~2-2.4 rad/s peak**, approaching the 3.14 rad/s
protective-stop limit. The proposal's claim "the approach is never faster than steady-state
teleop" is arithmetically false above 1.75 rad. (The legacy code has the same flaw — but the
proposal explicitly advertises the guarantee, so it must actually hold.)

**Required:** `T = max(gap / v_budget, T_min)` with NO upper clip, plus a hard refusal
(abort fail-safe with per-joint report, reusing the `alignment_hard_limit` concept, :149-154)
when any single-joint gap exceeds a `chase_hard_limit` (~1.0-1.5 rad is a sane default: a
leader that far from the arm after the first approach indicates a wrap/calibration fault, not
drift — moving there autonomously is the wrong reflex). Additionally, pick `v_budget`
acknowledging the spline peak factor: mean 0.35 rad/s → ~0.55-0.7 rad/s peak, at/above the
0.625 rad/s teleop ceiling. Use v_budget ≈ 0.25-0.3 if the "never faster than teleop" claim
is to survive on the actual interpolated profile, or state the claim in terms of mean speed.

### R3 (MEDIUM) — Gate re-chase iterations on leader stillness; otherwise the "never settles"
### case makes the arm shadow the moving leader for up to 60 s.

As specced, every loop iteration sends a trajectory toward wherever the leader currently is.
In scenario S3 (0.3 rad sinusoid forever) the arm therefore *continuously follows the
sinusoid* at the v_budget via back-to-back trajectories until `converge_timeout` — a minute
of autonomous motion tracking an operator who has not authorized streaming. That is
materially worse UX/safety posture than today's single move, and assertion A5 ("NO switch")
would PASS while this misbehavior occurs — the validation as written cannot catch it.

**Required:** for iterations ≥ 2, require the leader quasi-still (same median-filtered
stillness estimator, e.g. worst-joint speed < ~0.1 rad/s over ~0.3 s) BEFORE dispatching the
next catch-up trajectory; while the leader moves, HOLD and prompt. The loop becomes:
wait-for-still → move to it → dwell-verify → switch. Same convergence property (a still
leader converges in ≤ 2 moves), strictly less unexpected motion. Add an assertion to S3:
after the first trajectory completes, total additional arm travel < ~0.05 rad.

### R4 (MEDIUM) — Convergence livelock risk vs real JTC arrival error; the mock cannot
### detect it; two loop-robustness omissions.

ε = 0.025 rad must exceed (real scaled-JTC steady-state arrival error) + (30 Hz sampling
noise floor). The action's `goal_tolerance` (currently 0.05, proposed 0.02) is an *abort
threshold*, not an arrival guarantee: SUCCEEDED permits up to that much standing error. If
the real arm systematically settles 0.02-0.04 rad off on any joint (gravity sag, friction),
every iteration re-sends essentially the same target, never converges, and after
`max_converge_iters` the node aborts — with the prompt "hold leader still", which is the
WRONG diagnosis (the leader IS still). Also note the coupling hazard in L4: tightening
`arrival_tolerance` to 0.02 while ε = 0.025 means a 0.021 rad settle now ABORTS the goal
outright — the two knobs can deadlock teleop startup between them.

**Required:** (a) validate ε against measured real-arm arrival error before freezing 0.025
(cheap: one rosbag of an ordinary handshake already contains it); keep
`arrival_tolerance ≥ ~1.5×` the measured settle error rather than blind 0.02; (b) detect
"leader still AND gap not shrinking across ≥ 2 iterations" and emit a calibration/arrival
diagnostic distinct from the hold-still prompt; (c) the mock robot must support a
configurable arrival BIAS (e.g. 0.015/0.03 rad undershoot) and a mid-chase goal-ABORT
scenario (models pendant stop / protective stop) asserting fail-safe exit with no switch and
no retry-after-abort — the proposed mock arrives *exactly* at the goal, so it structurally
cannot exercise the livelock, the abort path, or the ε floor. This is the main validation
gap; S1-S5 otherwise do genuinely exercise the moving-leader cases, including the decisive
S2.

Two loop omissions to fold in: **(i) `~/abort` is never consulted in "gello" mode** (checked
only inside `_wait_for_operator`, :543-545); a 60 s chase loop needs the operator console's
stop to work — poll `self._abort` each iteration and during dwell. **(ii) GELLO staleness
during the loop:** move_to_start has no staleness watchdog. If the gello stream dies,
`_gello_latest` freezes, the arm converges to the frozen pose, the DWELL PASSES (a frozen
value is perfectly "still"), and the switch proceeds on a dead stream — the bridge then holds
(its own watchdog), and on stream recovery R1's staleness path closes the by-then-arbitrary
gap. Require: gello message age < ~0.5 s for every dwell sample and before each dispatch,
else abort fail-safe.

---

## Sufficiency scorecard against the four adversarial behaviors (post-changes)

| Behavior | Verdict |
|---|---|
| (a) abrupt move during approach | **Handled by construction** (gate holds; loop re-chases when still again). This is the decisive improvement over Fix#1/#2 and is real, not asserted: the gate is a measured condition, not a snapshot. |
| (b) never settles | Handled: no switch, prompt, timeout → exit 1, bridge stays paused. With R3 the arm also stops *moving* after the first approach. |
| (c) fast move at the handover instant | **Irreducible residual, but the proposal understates it.** Median-of-5 at 30 Hz adds ~70-100 ms detection lag on top of the ~0.15 s switch window, so worst case is ~(0.2-0.25 s × leader speed) of gap — a hard 2 rad/s flick timed exactly at gate-pass yields ~0.4-0.5 rad closed at 0.625 rad/s (~0.7 s). Defensible because it strictly coincides with the operator's own deliberate fast motion (and normal teleop already rate-limits behind any >0.625 rad/s leader move), and L3 ramps the onset — but the "≤0.025 rad, ~40 ms, sub-perceptual for ALL behaviors" claim must be corrected to this honest bound. |
| (d) large single-joint / wraparound offset | **Broken as specced (R2)**; handled after R2 (budget-true sizing + hard-limit refusal with per-joint report, matching init_align's protection against un-eyeballable base offsets). |

Safety posture overall: sound. All approach motion is JTC-interpolated with (post-R2)
budgeted durations; all streaming motion stays under the untouched 0.0025 rad clamp; every
failure path (timeout, goal abort/E-stop, service unavailable, operator abort) degrades to
"scaled controller active or FPC holding, bridge paused, exit 1" — no path commands
unbounded motion. The resume-on-failure story in L2 (bridge stays paused forever) is correct
and I verified the launch keeps the failure branch (:388-397) intact.

## Implementability check (real stack)

- Strictly-sequential goal dispatch (send → block on SUCCEEDED → measure) reuses the exact
  proven `_send_trajectory` pattern (:316-385) — no preemption, no re-goal semantics ever
  exercised. Realistic on ur_robot_driver 2.13.2 + Humble JTC.
- The switch itself is unchanged and independently proven jump-free (driver reseeds
  `position_commands = actual` at mode switch; FPC `on_activate` only clears the RT buffer).
- `start_paused` + resume-as-trigger uses only existing, tested bridge machinery; default
  false keeps every other launch path byte-identical. The 10 s `wait_for_service` with
  manual-resume fallback is an acceptable degradation.
- The single-threaded spin architecture handles the dwell fine (spin_once at GELLO cadence).
- One practical note: with L2 the bridge and move_to_start now start in the same window;
  the resume client should be created before the (up to 120 s) Play wait so discovery cost
  is hidden.

## Missed-simpler-solution check

I attempted to break the "L1 is necessary" claim and could not: every cheaper scheme
(snapshot refresh, one verified catch-up, stillness-only, soft-start-only, tolerance
tightening) fails at least one of behaviors (a)-(d), consistent with the measured numbers in
EXPERIMENT_RESULTS.md. A bounded special case of L1 (max_converge_iters=2 with the gate) is
the same code with a smaller constant — not a distinct simpler design. L2 has no cheaper
equivalent that closes the 0.5-2 s cold start. Nothing recommended is unnecessary — except
that L4's `arrival_tolerance: 0.02` must be re-derived per R4, not asserted.

---

## Bottom line

The proposal correctly identifies that the fix must be a *gated invariant*, not a better
snapshot, and correctly identifies the cold-start window as the second necessary half. But
its headline guarantee ("never begins AND NEVER RESUMES ... for ALL operator behaviors") is
not delivered by the specced L1+L2: the manual-resume and staleness-recovery doors remain
wide open (R1), the catch-up sizing formula contradicts its own velocity budget exactly in
the large-offset case it needs to cover (R2), the chase loop shadows a moving leader (R3),
and the validation mock is structurally blind to the most likely real-hardware failure
(arrival-error livelock, R4). All four are cheap to fix within the proposed architecture.

**ENDORSE WITH CHANGES — R1-R4 mandatory before implementation; with them, the design
delivers the stated invariant (with the corrected case-(c) bound) and I know of no residual
operator behavior that produces unprompted fast motion.**
