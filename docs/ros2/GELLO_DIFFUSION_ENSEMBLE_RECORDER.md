# Diffusion Policy Inference Recorder + 16-Sample Uncertainty Ensemble

> **STATUS (2026-07-09): implemented and code-reviewed, but NOT YET SAFE to enable on
> the robot PC's GPU.** The ensemble side-channel is built, verified correct, and
> default-OFF (`DIFFUSION_ENSEMBLE_K=0`) — the existing ACT/Diffusion deploy paths are
> unaffected either way. Enabling it (`DIFFUSION_ENSEMBLE_K=16`) currently **fails** the
> mandatory pre-hardware safety benchmark on this machine's GPU (RTX 3060 Laptop, 6GB).
> Do not set `DIFFUSION_ENSEMBLE_K` to a nonzero value on the real arm until a later
> entry in this doc says PASS.

## TL;DR

| | |
|---|---|
| Goal | Record everything useful from a real diffusion-policy run (video, joint/TCP/wrench, gripper) **plus** a 16-noise-sample trajectory ensemble per refill, for offline uncertainty (divergence-across-samples) research. |
| Data recorder | Reuses `gello_recorder` (HDF5 `vectors.h5` + `cam1.mp4`/`cam2.mp4`) unmodified. New PyQt5 GUI (`policy_run_gui`) auto-starts recording on `~/start_execution`, manual Stop, then forces a SUCCESS/FAIL label click before the next take. |
| Ensemble side-channel | New, **default-off**, additive-only hook in `diffusion_server.py`. Runs on a background thread + private CUDA stream + private noise-scheduler instance, strictly *after* the real ZMQ reply is sent — cannot affect what the robot does. |
| Safety gate | `scripts/benchmark_ensemble_batch16.py` — pure GPU/software, never moves the robot. **Currently reports FAIL** on this machine (see §6). |
| Bug caught in review | A latent crash in the *existing, already-verified* diffusion launch path (`int("")` on an exported-but-empty env var) — found and fixed before it could ever reach hardware. See §7. |

## 1. Motivation

The user is running real diffusion-policy inference on the UR7e (`Bigenlight/diffusion_banana_in_pot_joint`, see [GELLO_UR7E_DIFFUSION_DEPLOY.md](GELLO_UR7E_DIFFUSION_DEPLOY.md)) and wants offline data to study whether **divergence across multiple noise-seeded trajectory samples from the same observation** is a usable real-time uncertainty/confidence signal for the policy (more divergence = less confident) — plus general-purpose video/vector logging reusable for other future research.

Hard constraint: **the extra sampling must never be able to add latency to, or otherwise perturb, the real robot-control ZMQ round trip** (`act_timeout_s = 0.6s` in `config/diffusion_deploy.yaml`).

## 2. How this was built

Built via a two-stage agent workflow this session:
1. **Research + design + review** (13 agents: 6 research, 3 design proposals, 3 reviews + 1 synthesis) produced a concrete spec — see the synthesis's own risk list, largely reproduced in this doc.
2. **Implementation + review** (2 build agents + 3 review agents) wrote the code against that spec, then independently verified it against real files (not just their own summaries), ran real imports under both interpreters, and ran a real `colcon build`.

## 3. Architecture

**Zero changes to the real-time control path.** `policy_leader_node.py`, `zmq_protocol.py`, `act_server.py`, `camera_viewer.py`, and every launch file are untouched. The only robot-adjacent file touched is `diffusion_server.py`, and only additively.

### New files
| File | Role |
|---|---|
| `src/gello_policy/policy_server/ensemble_sampler.py` | Runs inside `diffusion_server.py`'s process. Owns a **private** `DDIMScheduler` instance (never touches the real policy's scheduler — they'd corrupt each other's state if shared), a **dedicated `torch.cuda.Stream()`**, and a 1-worker `ThreadPoolExecutor` that **drops, never queues**, a job if the previous one hasn't finished. `run_batch()` mirrors lerobot's `conditional_sample` exactly, batch=16, conditioned on the real observation's already-computed `global_cond` (the expensive vision-encoder pass is NOT re-run — only the cheap `expand(16, -1)` + a batched 10-step DDIM loop). |
| `src/gello_policy/policy_server/ensemble_logger.py` | Plain `h5py` writer (no `gello_recorder` import — stays on the py3.12 side of the interpreter boundary). Writes `ensemble_<start_wall>.h5` next to `gello_logs/`. |
| `src/gello_policy/scripts/benchmark_ensemble_batch16.py` | The safety gate — see §6. |
| `src/gello_recorder/gello_recorder/policy_run_gui_node.py`, `policy_run_gui.py` | New PyQt5 GUI, see §5. |

### Edited files (all additive / default-off)
| File | Change |
|---|---|
| `src/gello_policy/policy_server/diffusion_server.py` | `--ensemble-k` (default 0, env `DIFFUSION_ENSEMBLE_K`), `--ensemble-dir`. When `k=0`, zero extra code runs — verified by the review agent by reading the actual branch conditions. The ensemble `submit()` call happens in `serve()` **strictly after** the real ZMQ reply (`_reply()`) has already been sent — this ordering is the single most load-bearing correctness property of the whole design, and it was independently re-verified line-by-line during review. |
| `run_ur7e_diffusion_real.sh`, `scripts/run_diffusion_server.sh` | Thread `DIFFUSION_ENSEMBLE_K`/`DIFFUSION_ENSEMBLE_DIR` through, same pattern as the existing `DIFFUSION_N_ACTION_STEPS` etc. |
| `src/gello_recorder/{setup.py,package.xml}` | New `policy_run_gui` console-script entry; added missing `std_srvs` exec_depend. |
| `src/gello_recorder/gello_recorder/gello_gui_node.py` | One backward-compatible line: `node_name` is now an optional constructor param (default unchanged) so the new GUI's node doesn't collide with the teleop recorder's node name. |
| `src/gello_policy/policy_server/requirements-diffusion.lock` | Added `h5py==3.16.0` (was missing from `act_venv`). |

## 4. What gets recorded

Per take (`policy_runs/take_NN_<timestamp>/`, via the reused `gello_recorder` machinery, unmodified):
- `cam1.mp4`, `cam2.mp4` — dual camera video.
- `vectors.h5` — the same 9 tables `gello_recorder` has always written: `synchronized`, `gello_joint_states`, `ur_joint_states`, `command`, `gripper`, `wrench`, `tcp_pose`, `cam1_frames`, `cam2_frames`. Wrench/TCP-pose topics were confirmed to already publish during a real diffusion run with **no launch-file changes needed**.
- `label.json` — written atomically (temp file + `os.replace`) by the SUCCESS/FAIL buttons; the START EXECUTION button is structurally disabled between Stop and a label click, so an unlabeled take cannot happen.

Separately, one `ensemble_<server-start-timestamp>.h5` per `diffusion_server.py` process lifetime (can span multiple takes), written only when `--ensemble-k > 0`:
```
/ensemble_trajectories/meta            t_rel_s, t_wall, refill_idx, ensemble_ms, dropped_flag
/ensemble_trajectories/trajectories    (N, 16, horizon=64, action_dim=7) float32 — the 16 samples
/ensemble_trajectories/committed_chunk (N, n_action_steps=32, action_dim=7) float32 — what actually got queued for the robot
/ensemble_trajectories/obs_state       (N, n_obs_steps=2, state_dim=7) float32 — conditioning window
```
All values are in lerobot's **normalized** space (matches the checkpoint's training normalization) — offline analysis must unnormalize using the checkpoint's saved processor stats before interpreting magnitudes.

### Alignment caveat: episode-level, not tick-level

The ensemble file and `vectors.h5` are only joined by **wall-clock timestamp** (`t_wall` in both, `metadata.json`'s `start_wall`/`stop_wall` for the take boundary). This is enough to know **which take an ensemble sample belongs to**, and to see **trend-level** patterns ("did divergence rise over the course of this episode"). It is **not** precise enough to pin one ensemble sample to one exact robot-control tick/synchronized-table row (e.g. "did divergence spike at the exact tick the protective stop fired"), because:
1. `diffusion_server.py` and `policy_leader_node.py` are independent processes with **no shared sequence counter** — each only stamps its own wall clock.
2. A refill itself is a ~185–210ms **computation window**, not an instant, so "the timestamp of this ensemble sample" is inherently fuzzy relative to any single 30Hz control tick.
3. Network/driver latency between a computed action and the arm actually executing it adds further slop.

Getting true tick-level correlation would require a small additive change to `policy_leader_node.py` (a tick counter/timestamp publisher) — deliberately deferred to a Phase 2, to keep the first version's risk surface (and its diff footprint on the real-time-adjacent code) as small as possible.

## 5. GUI flow (`policy_run_gui`)

A new PyQt5 window, sibling to `gello_recorder_gui` (does **not** replace `camera_viewer.py`, which remains available for ACT/teleop use). Critically, unlike `gello_recorder_gui.py`, **it does not launch its own RealSense camera subprocesses** — cameras must already be running via `./launch_cameras.sh` first, exactly like `camera_viewer.py` assumes, otherwise two processes would fight over the same USB camera.

```
launch_cameras.sh (cameras up)
  -> run_ur7e_diffusion_real.sh (diffusion server + arm driver up, arm parks)
  -> policy_run_gui (dual preview + state panel + START EXECUTION / HOLD / Stop Recording)

[START EXECUTION] succeeds  -> recording auto-starts (no separate manual step)
[HOLD] or a protective stop -> does NOT stop recording (explicit user requirement)
[Stop Recording] (manual)   -> [SUCCESS] [FAIL] buttons appear
                                START EXECUTION stays disabled until one is clicked
click SUCCESS or FAIL       -> label.json written, ready for next take
```

## 6. Safety benchmark results (this machine, 2026-07-09)

`scripts/benchmark_ensemble_batch16.py` measures the real single-sample refill latency (`select_action()`, DDIM-10) while the production `EnsembleSampler.run_batch()` code path runs **continuously** in the background (worst-case stress test, not average-case) on its own CUDA stream. PASS requires contended p99 < 500ms (margin under the real 600ms `act_timeout_s`).

| Ensemble batch K | Baseline p99 | Contended p99 | Slowdown | Verdict |
|---|---|---|---|---|
| 16 | 214 ms | 811 ms | 3.79x | **FAIL** |
| 4  | 217 ms | 581 ms | 2.67x | **FAIL** |
| 2  | 217 ms | 587 ms | 2.70x | **FAIL** |
| 1  | 210 ms | 545 ms | 2.60x | **FAIL** |

**Why even K=1 fails so badly:** this is the important finding. A single-sample DDIM-10 refill (194ms) already leaves this laptop GPU (RTX 3060, 6GB, limited SM count) with very little spare compute headroom. Separate CUDA streams let two kernel sequences be *dispatched* concurrently, but they cannot conjure extra physical compute units — if the GPU's compute is already close to saturated by one full DDIM-10 UNet loop, a second concurrent one is largely forced to time-slice on the same silicon, roughly doubling wall-clock cost regardless of batch size. A desktop/server-class GPU with real spare headroom would likely behave very differently; this result is specific to this laptop GPU.

**Caveat on the test itself:** the benchmark's "contended" phase runs the background ensemble job **back-to-back with no gaps**, deliberately worse than reality — real refills only happen once every ~1.07s (`n_action_steps=32` @ 30Hz), and one ensemble job takes ~300–500ms on average, so there is naturally ~600–800ms of idle GPU time per cycle it could fit into. A more realistic "one job per real 1.07s cycle" probe is the immediate next step (in progress) to check whether real-world contention is actually this severe or much smaller — but until that is run and shown to PASS, treat the worst-case FAIL above as the operative answer for hardware safety purposes.

## 7. Bug caught during review (pre-existing path, not the new feature)

The integration reviewer found that `run_ur7e_diffusion_real.sh`'s `VAR="${VAR}" bash script.sh` pattern **always exports** `DIFFUSION_ENSEMBLE_K` into the child environment, even when its value is empty — and `diffusion_server.py`'s original `int(os.environ.get("DIFFUSION_ENSEMBLE_K", "0"))` only substitutes `"0"` when the key is *absent*, not when it's present-but-empty. A present-but-empty value hits `int("")` and crashes at argument-parsing time. This meant the "additive, default-off" feature would have broken **every normal diffusion launch**, including ones that never touch the new feature. Fixed to `int(os.environ.get("DIFFUSION_ENSEMBLE_K", "0") or "0")` and re-verified (default/unset, explicit-empty, and `=16` all now parse correctly).

## 8. Open next steps

1. **Realistic-cadence re-benchmark** (in progress): fire one ensemble job per real ~1.07s cycle (not continuously) and measure whether the real refill immediately after is ever actually delayed. If this passes with real margin, the current in-process design may be usable as-is despite the worst-case FAIL above.
2. **Separate-GPU/process fallback** (leading alternative if #1 doesn't pass): run the ensemble sampler as a fully separate process — ideally on a **different machine's GPU** on the same network — that independently loads the checkpoint and receives observations via a side-channel, so it shares no compute with the robot-control GPU at all. This was flagged as the safest-but-most-complex option during the original design phase; worth revisiting now that in-process contention is confirmed real on this hardware. Requires knowing whether a second GPU machine is actually available.
3. **Reduce ensemble DDIM steps independently of the real path's 10** (not yet implemented — the sampler currently reuses the same step count as real inference): fewer steps for the ensemble-only pass would cut its compute cost further; combine with #1's realistic timing to see if it's sufficient.
4. **Tick-level alignment** (§4 caveat): a small `policy_leader_node.py` addition if/when precise tick correlation becomes necessary for the research.
5. **Multi-modality confound**: high sample-divergence can mean "two equally valid grasps exist," not "the policy is unconfident in a bad way." No part of this design resolves that — worth a simple offline clustering pass (e.g. k-means on final actions per refill) before trusting divergence as a pure confidence signal.

## 9. Before ever setting `DIFFUSION_ENSEMBLE_K` nonzero on the real arm

Run, on the actual robot PC, every time the GPU/driver/checkpoint changes:
```bash
cd ros2_ur_ws
act_venv/bin/python src/gello_policy/scripts/benchmark_ensemble_batch16.py \
    --checkpoint "$(pwd)/src/gello_policy/checkpoints/diffusion_banana_in_pot_joint" \
    --device cuda --ensemble-k 16
```
It never moves the robot. Only proceed to a real hardware run if it prints `VERDICT: PASS`.
