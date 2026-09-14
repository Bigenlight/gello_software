# carrot_in_pot — EEF-delta imitation learning (ACT + Flow Matching) — design contract

Status: v1, 2026-09-14. Integrator owns this file; implementers work against it and
report deviations instead of silently changing the contract.
Worktree: `/home/laptop3/gello_software_humble` (branch `feat/carrot-eef-il`, based on
`feat/gello-ur7e-humble-22.04`). **Edit only inside this worktree.** The main checkout
`/home/laptop3/gello_software` belongs to another session — never touch it.

## 0. Goal

Train ACT and Flow-Matching (lerobot 0.6.1 `multi_task_dit`, `objective=flow_matching`)
on the 54-take `Put carrot in pot` dataset with a **Cartesian (EEF) delta action space**,
on **kanu (≤ 2 GPUs)**, and run the trained policies on the real UR7e through the existing
`gello_policy` deploy stack (`policy_leader_node` → `gello_ur_bridge` joint mode → UR).
No depth. The existing joint-space path must keep working unchanged (`action_space: joint`).

## 1. Facts (measured 2026-09-14; do not re-derive)

| fact | value |
| --- | --- |
| Raw takes | `/home/laptop3/youngwoong_ws/Put_carrot_in_pot/take_*` (54, EEF-teleop mode, 30 fps cams, `vectors.h5` + `cam1.mp4` + `cam2.mp4` + `depth.h5`). Existing joint-space LeRobot conversion: `/home/laptop3/youngwoong_ws/carrot_in_pot_lerobot` (18,557 frames). |
| Converter env | `/home/laptop3/youngwoong_ws/lr_env/bin/python` (py3.12, lerobot 0.6.1 pin 8a74e0a). Needs `PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy` (both are ROS-free modules). Verified: `from ur_gello_bringup import ur_kin` works there. |
| TCP | flange (`ur_kin.fk`) ⊕ 0.174 m along flange +Z. Matches the recorder's `tcp_pose` table to ~1 mm. Defined once in `gello_policy/eef_space.py` (`T_TOOL`). |
| Recorded lag | In this EEF-mode session `command` LEADS the measured joints by **~0.9 s** (61 mm median at τ=0, 1 mm at τ=+0.9 s, all takes). Joint-mode cube takes show ~0 lag (3–5 mm). So `command` is NOT "next pose"; **the action is defined from the ACHIEVED motion, never from `command`** (§2). Root cause is open (owner A7). |
| Achieved motion per frame (30 fps) | |Δp| p50 2.2 mm, p99 6.6 mm, max 9.6 mm; |Δrot| p99 0.02 rad, max 0.03 rad. Over 30 frames: |Δp| p50 64 mm, p99 159 mm, max 161 mm; |Δrot| max 0.46 rad. |
| TCP workspace (all frames) | x 0.415–0.709, y −0.214–0.389, z 0.017–0.306 m (base frame). |
| Start pose (frame 0, 54 takes) | mean `[-3.193, -1.490, 1.693, -1.798, -1.590, -3.269]` rad, std ≤ 0.113, max |dev| 0.28. Start TCP ≈ (0.528, 0.101, 0.235). Gripper open (grip_pos ≈ 0.012). |
| IK branch | every frame of every take is closed-form branch **2** (`ur_kin.branch_id`). Round trip `delta → apply → ik_tcp_nearest` recovers the target to 3e-15 rad, 0 failures on 1,079 sampled frames. |
| Deploy stack facts | `policy_leader_node` (py3.10 rclpy) publishes `/gello/joint_states` at 30 Hz as a synthetic leader; bridge runs in JOINT mode (slew 0.625 rad/s) → `forward_position_controller`. Servers (py3.12 `ros2_ur_ws/act_venv`, symlinked into the worktree) speak ZMQ REQ/REP: ACT 5591, FM 5593. Config yaml is selected by `params_file` launch arg (`FM_PARAMS_FILE` env in `run_ur7e_fm_real.sh`; ACT script has no such env yet). `checkpoints/` is symlinked into the worktree too. |
| Banana values to REPLACE | `start_pose [3.106, …]`, `joint_limits_lo/hi` (shoulder_pan +2.40..+3.67 — would clip carrot's −3.19 to +2.40), task string "put the right banana in the pot". |
| kanu | ssh alias `kanu`, 8× RTX A4000 16 GB. **Use at most 2 GPUs, pick idle ones with `nvidia-smi` right before launch** (GPU 4 was busy). Disk 98 % full / 51 GB free — do not write outside `~/workspace/youngwoong/carrot_eef/`. lerobot env already there: `~/workspace/youngwoong/cube_flow_matching/training/lr_env` (torch 2.11 cu128, CUDA OK, lerobot 0.6.1); CLIP ViT-B/16 in HF cache. Reuse it via `LEROBOT_VENV`; do not create a new env. |
| Reference train recipes | ACT: `/home/laptop3/youngwoong_ws/banana-in-pot-experiments/train_act.sh` (+ `train_act_valdiag.sh`, `eval_offline.py`). FM: `/home/laptop3/youngwoong_ws/cube_flow_matching_training/train_flow_matching.sh` (+ preflight/launch_detached/eval_checkpoints). Both are plain `lerobot-train`. |

## 2. Policy space — `gello_policy/eef_space.py` is the single source of truth

Provided by the integrator. **Import it; never re-implement the arithmetic.**

```
state  (16) = [q1..q6, tcp_x, tcp_y, tcp_z, r11, r21, r31, r12, r22, r32, grip_pos]
              q as observed (UR order, rad); TCP in base frame; 6-D rotation = first two
              COLUMNS of R_tcp (continuous; the carrot TCP is ~175° from identity so a
              rotvec would wrap); grip_pos 0=open..1=closed.
action (7)  = [dx, dy, dz, drx, dry, drz, grip_cmd]
              dataset frame t (30 fps master clock):  Δ_t = TCP(q_{t+1}) ⊖ TCP(q_t)
                 dp = p_{t+1} − p_t ; drot = so3_log(R_{t+1} R_tᵀ)   (base frame, left-mult)
              last frame of an episode: dp = drot = 0.
              grip_cmd = ABSOLUTE recorded gripper command (0=open..1=closed), same
              resampling (ffill/bfill) as the existing converter.
ACTION_SPACE_ID = "eef_delta_v1"
```

Helpers: `state_vector(q, grip_pos)`, `delta_action(q_a, q_b, grip)`, `apply_delta(q_live, action)`
(= target TCP pose), `ik_tcp_nearest(T_tcp, q_seed, branch)`, `clamp_delta`, `clamp_position_box`,
`rot_to_6d`/`rot_from_6d`. `delta_action(q_t, q_{t+1}, grip)` IS the dataset action.

## 3. Deploy semantics (`policy_leader_node`, `action_space: eef_delta`)

Per 30 Hz tick in EXECUTE:
1. `live_q = positions_near_reference(/joint_states, start_pose)`; `state = state_vector(live_q, grip_pos)` (16).
2. ZMQ `act` → reply `{"ok":true,"action":[7],"chunk_start":bool}`.
3. **Chunk accumulation.** `chunk_start` (server re-ran the network) or first tick after
   arming → `anchor = TCP(live_q)` and `target = anchor`. Every tick: `target = target ⊕ Δ`
   (`apply_delta` on the *target*, not on live). Rationale: a chunked policy predicts the
   trajectory from the observation it saw; applying each Δ to a lagging live pose would
   shrink the motion every tick.
4. **Cartesian clamps, in order** (all norm-proportional, never per-axis; each logs a
   throttled WARN): (a) per-tick `clamp_delta(Δ, eef_max_step_m, eef_max_step_rad)`;
   (b) lead clamp: if `|p_target − p_live| > eef_max_lead_m` scale the lead vector down to
   that norm (same for rotation vs `eef_max_lead_rad`) — the target may never run away from
   a stalled arm; (c) `clamp_position_box(p_target, workspace_lo, workspace_hi)`.
5. `q_target, info = ik_tcp_nearest(target, live_q, branch=start_branch)` where `start_branch =
   branch_id(start_pose)` computed once at arming. `None` → publish `live_q` (hold) this tick,
   count; `eef_ik_fail_fault_n` consecutive → FAULT.
6. Existing joint clamps unchanged: envelope `joint_limits_lo/hi`, then `max_dev_rad` vs live.
   Then publish exactly as today (`_publish_arm`, `_publish_gripper` with grip clipped 0..1).
7. `~/hold` and FAULT behave as today. Arming (`~/start_execution`) additionally sends RESET
   and **refuses** unless the RESET reply reports `action_space == expected_action_space`
   and `state_dim == 16`, `action_dim == 7` — a joint checkpoint can never drive eef mode
   or vice versa.

New parameters (declare with these exact names; defaults in parentheses):
`action_space` ("joint"), `expected_action_space` ("eef_delta_v1"), `eef_max_step_m` (0.02),
`eef_max_step_rad` (0.10), `eef_max_lead_m` (0.06), `eef_max_lead_rad` (0.30),
`workspace_lo` ([0.35, -0.30, 0.00]), `workspace_hi` ([0.78, 0.45, 0.40]),
`eef_ik_fail_fault_n` (10). Values for the carrot yaml come from A2's JSON (§5).
`action_space: joint` must be byte-for-byte the current behaviour (regression tests).

## 4. ZMQ protocol v2 (backward compatible for joint mode)

- RESET reply gains: `{"ok":true,"action_space":"joint"|"eef_delta_v1","state_dim":N,"action_dim":7,
  "policy_type":"act"|"multi_task_dit","checkpoint":"<path>"}`. Servers derive `state_dim` from
  `policy.config.input_features["observation.state"].shape[0]`; `action_space` from
  `--action-space` CLI/env (`ACT_ACTION_SPACE`/`FM_ACTION_SPACE`, default joint) **and** must
  refuse to start if it contradicts the checkpoint (state_dim 16 ⇔ eef_delta_v1, 7 ⇔ joint).
- ACT reply gains `"chunk_start": bool` (true when the queue was empty before `select_action`,
  i.e. the network re-ran this tick).
- `state` length is validated against `state_dim` (not the hardcoded 7). `STATE_DIM` constants in
  `zmq_protocol.py` / `obs_assembler.py` become per-space tables or are replaced by the handshake.
- FM: `--task` default must be the dataset task string **"Put carrot in pot"** when
  `--action-space eef_delta_v1`; the node/launch pass it explicitly anyway.
- Image handling untouched (ACT 360×640 resize + BGR→RGB; FM native + internal 224).

## 5. Ownership

| id | owner delivers | files (worktree-relative) |
| --- | --- | --- |
| **A1 dataset** | `convert_carrot_to_lerobot_eef.py` (no depth; state 16 / action 7 per §2; RGB + master-clock resampling identical to the existing converter; `meta/source_takes.json`; `meta/eef_space.json` = {ACTION_SPACE_ID, TOOL_Z_M, names}), an **independent** validator `validate_carrot_eef_conversion.py` (re-derives state/action straight from h5 + ur_kin without importing the converter: Δ=0 tolerance 1e-6, frame counts, 6-D orthonormality, last-frame zero, task string), run the conversion to `/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot` (repo_id `Bigenlight/carrot_in_pot_eef_lerobot_v3`), run the validator, print `meta/info.json` features. **Do not upload to HF.** | `scripts/dataset/convert_carrot_to_lerobot_eef.py`, `scripts/dataset/validate_carrot_eef_conversion.py`, `scripts/dataset/carrot_eef_validation_2026-09-14.json` |
| **A2 limits** | `make_carrot_eef_limits.py`: from raw takes → `config/carrot_eef_limits.json` with `start_pose` (mean of frame-0 q, unwrapped near the first take), `start_branch`, `joint_raw_min/max`, `joint_limits_lo/hi` (1.2× symmetric about centre, same formula/note as the banana json), `workspace_raw`, `workspace_lo/hi` (raw ± 0.03 m, z_lo = max(raw_z_min − 0.01, 0.0)), delta stats (1-frame / 30-frame p50/p99/max), `eef_max_step_m/rad` = 1.5 × 1-frame max rounded up, `eef_max_lead_m/rad` = 30-frame p50 rounded up to 0.01, `n_takes`, `n_frames`, git commit, generated-at. Also emits the exact yaml snippet A6 pastes. Unit test on a synthetic take. | `scripts/dataset/make_carrot_eef_limits.py`, `ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json`, `ros2_ur_ws/src/gello_policy/test/test_carrot_eef_limits.py` |
| **A3 training** | `scripts/training/carrot_eef/`: `train_act_carrot_eef.sh`, `train_fm_carrot_eef.sh` (mirror the reference recipes; `--dataset.eval_split=0.111` = last 6 of 54; ACT 50k steps / FM 100k, save every 10k, `--wandb.enable=false`; `GPU_ID` required; `DRY_RUN=1` prints the command), `preflight.sh` (GPU idle, dataset features == §2, disk ≥ 20 GB, `lerobot-train --help` works), `launch_detached.sh` (setsid nohup, pid + log under `runtime/`), `sync_to_kanu.sh` (rsync bundle + dataset to `kanu:~/workspace/youngwoong/carrot_eef/`, `--dry-run` first), `eval_offline_eef.py` (open-loop metrics on held-out episodes: position MAE m, rotation MAE rad, gripper acc, per checkpoint; ACT and FM), `README_KO.md`. Run sync + preflight + `DRY_RUN=1` on kanu as soon as A1's dataset exists; **do not start training** — the integrator triggers it. | `scripts/training/carrot_eef/**` |
| **A4 node** | `policy_leader_node.py` eef_delta mode per §3 with the pure math in a new rclpy-free `gello_policy/eef_executor.py` (class holding anchor/target/counters; `step(live_q, action, chunk_start) -> (q_target or None, info)`); `obs_assembler.py` 16-D packing + v2 reply parsing + RESET handshake check; unit tests for the executor (accumulation, all clamps, IK fail → hold, re-anchor on chunk_start, joint-mode regression) runnable without ROS. | `ros2_ur_ws/src/gello_policy/gello_policy/{policy_leader_node,obs_assembler,eef_executor}.py`, `ros2_ur_ws/src/gello_policy/test/test_eef_executor.py`, `test_obs_assembler_v2.py` |
| **A5 servers** | §4 in `zmq_protocol.py`, `act_server.py`, `fm_server.py` (+ `diffusion_server.py` only the RESET/act reply additions so it stays protocol-compatible), `scripts/run_act_server.sh` / `run_fm_server.sh` env passthrough (`*_ACTION_SPACE`, `FM_TASK`). Tests with a stub policy object (no checkpoint needed) in act_venv: handshake fields, state-dim validation, chunk_start flag. | `ros2_ur_ws/src/gello_policy/policy_server/*`, `ros2_ur_ws/src/gello_policy/test/test_zmq_protocol_v2.py` |
| **A6 configs+launch+docs** | `config/carrot_eef_act.yaml`, `config/carrot_eef_fm.yaml` (copy of act/fm_deploy.yaml with `action_space: eef_delta`, `expected_action_space`, §3 params, A2's `start_pose`/limits/workspace, FM `task: "Put carrot in pot"`), `run_ur7e_act_real.sh` gains `ACT_PARAMS_FILE` (mirror `FM_PARAMS_FILE`), launch passes the new node params through, `scripts/run_*_server.sh` accept the action-space env; `docs/ros2/GELLO_UR7E_CARROT_EEF_DEPLOY_KO.md` (operator runbook: dataset → kanu training → checkpoint fetch → latency check → deploy, env vars, safety clamps, what is NOT yet verified) + CLAUDE.md pointer (one table row). `colcon build --packages-select gello_policy ur_gello_bringup` in the worktree must pass. | `ros2_ur_ws/src/gello_policy/config/carrot_eef_*.yaml`, `ros2_ur_ws/run_ur7e_act_real.sh`, launch files, `docs/ros2/GELLO_UR7E_CARROT_EEF_DEPLOY_KO.md`, `CLAUDE.md` |
| **A7 lag + eef_space review** | (1) Explain the 0.9 s command-vs-joint lag of the EEF-mode session from the bridge/driver code and the raw data (which topic `command` is, whether the arm was slew/velocity limited, whether the same lag will exist when `policy_leader_node` drives the bridge in joint mode). Write `docs/ros2/CARROT_EEF_LAG_NOTE_KO.md`. (2) Adversarial review of `eef_space.py` (6-D convention, so3 conventions, branch handling near merge points, `wrapped_nearest` elbow rule, clamp semantics) with a test file `test/test_eef_space.py` (round trip on ≥ 3 raw takes, rot_from_6d∘rot_to_6d, apply/delta inverse, clamp direction preservation). Report bugs to the integrator; do not edit `eef_space.py` unless the bug is unambiguous (then note it in the report). | `docs/ros2/CARROT_EEF_LAG_NOTE_KO.md`, `ros2_ur_ws/src/gello_policy/test/test_eef_space.py` |

Wave 2 (after wave 1): A8 integration (end-to-end fake ZMQ server ↔ node in eef_delta mode
without a robot; round trip dataset action → executor → recovers q_{t+1}; full test suites),
A9 launch + monitor training on kanu, R1–R3 cross reviews.

## 6. Test commands (record the interpreter with every passed count)

```bash
# gello_policy + ur_gello_bringup (system python3, ROS overlay). Build first in the worktree:
cd /home/laptop3/gello_software_humble/ros2_ur_ws && set +u && source /opt/ros/humble/setup.bash && set -u \
  && colcon build --packages-select ur_gello_bringup gello_policy --symlink-install && source install/setup.bash
cd /home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy \
  && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider test
# rclpy-free modules without the overlay:
PYTHONPATH=/home/laptop3/gello_software_humble/ros2_ur_ws/src/ur_gello_bringup:/home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider test/test_eef_space.py
# servers (act_venv): same PYTHONPATH, interpreter /home/laptop3/gello_software_humble/ros2_ur_ws/act_venv/bin/python
# converter/validator: /home/laptop3/youngwoong_ws/lr_env/bin/python with the same PYTHONPATH
```

Rules: no new pip deps; no edits outside your rows without telling the integrator; never
touch the main checkout, `act_venv`, or kanu paths other than `~/workspace/youngwoong/carrot_eef/`;
never launch training; never move the robot (no robot in this phase). Korean or English is fine
in docs; code comments say WHY. Type hints + docstrings. Report: what you built, how you verified
it (commands + counts), open risks.
