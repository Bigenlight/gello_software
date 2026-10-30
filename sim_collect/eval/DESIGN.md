# sim_collect/eval — closed-loop success-rate evaluation in MuJoCo (design contract)

Status: design v1, 2026-09-15. Owners work against this file; the integrator changes it.

## 0. Goal

Measure the **success rate (SR)** of a trained carrot-in-pot policy in the `sim_collect` MuJoCo
scene: a fixed list of seeds (default 20), one episode per seed, same reset protocol as data
collection, success = carrot inside the pot (validated against the 22 recorded human demos),
and a JSONL/summary report. The policy is reached through the **same ZMQ protocol the real
deployment uses**, so a checkpoint served on kanu (GPU) is evaluated by laptop3's MuJoCo.
Also provide oracle/replay/zero policies so the harness itself is validated without a model.

## 1. Facts (researched; trust, do not re-derive)

### 1.1 Real remote-policy protocol (ZMQ REQ/REP) — reuse verbatim
- Schema: `ros2_ur_ws/src/gello_policy/policy_server/zmq_protocol.py` (stdlib-only). Client framing
  helpers (stdlib-only, byte-exact): `ros2_ur_ws/src/gello_policy/gello_policy/obs_assembler.py`
  `build_reset_request / build_act_request / parse_action_reply`.
  ```
  RESET: [ b'{"cmd":"reset"}' ]                       -> [ b'{"ok":true, ...v2 fields optional...}' ]
  ACT:   [ b'{"cmd":"act","state":[7 floats]}', cam1_jpeg, cam2_jpeg ]
                                                       -> [ b'{"ok":true,"action":[7 floats]}' ]  | {"ok":false,"err":..}
  ```
  `state = [q1..q6 (rad, UR order, re-branched near start_pose), grip_pos (0 open..1 closed)]`,
  `action = [q1..q6 target rad, grip 0..1]`. Images: **raw JPEG, full 1280×720, uncropped, BGR-encoded
  by cv2** (server converts to RGB and resizes: ACT/Diffusion → 360×640; FM resizes internally to 224²).
  One 7-vector per request (server keeps the action chunk queue; `n_action_steps` set server-side).
  Task string for FM: **"Put carrot in pot"** (CLIP-conditioned; must match the dataset). ACT/Diffusion ignore it.
  Servers: `policy_server/{act,fm,diffusion}_server.py` + `scripts/run_{act,fm,diffusion}_server.sh`
  (venv `ros2_ur_ws/act_venv`: lerobot 0.6.1, torch cu128, pyzmq, transformers). Ports 5591/5592/5593.
  Do **not** apply the HIL-SERL crops `(20,670,340,990)`/`(0,720,420,1140)` — those belong to the jax stack.
- Timeouts in the real client: REQ `RCVTIMEO` 0.5–0.6 s; a timeout is a FAULT (episode ends, counted as failure).
- One RESET per episode, before the first ACT.

### 1.2 How the real deploy executes actions (`gello_policy/policy_leader_node.py::_tick_execute`)
30 Hz: after the ACT reply, **clamp in this order**: (1) per-joint envelope clip to `joint_limits_lo/hi`;
(2) per-joint clip so `|target − live_q| ≤ max_dev_rad` (0.5); (3) `grip = clip(action[6], 0, 1)` published
as-is (no threshold). The joint target then goes through the unmodified 250 Hz bridge: One-Euro
(policy-deploy params `min_cutoff 3.0, beta 4.0, d_cutoff 1.0`, dt 1/250) → `bridge_stages.clamp_stage`
(`max_step_rad 0.0025` ⇒ 0.625 rad/s), `deadband_rad 0.004`, `soft_start_s 0.7`. Params:
`ros2_ur_ws/src/ur_gello_bringup/config/{act,diffusion,fm}_deploy.yaml`.
**Carrot branch**: the carrot data lives on the −π shoulder_pan branch. Use
`/home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json`
(`start_pose [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]`, `joint_limits_lo/hi` J1 ∈ [−3.80, −2.58],
`start_grip_pos ≈ 0.012`) — NOT the banana +π values. The sim home `[-3.302, -1.563, 1.607, -1.523, -1.615, -3.118]`
is on the right branch. For the SIM the episode start pose is the sim `home_joints` (what the demos started from).
- No episode-length limit exists upstream; the harness adds one (default 600 steps = 20 s @30 Hz).

### 1.3 Environments (forced split)
| | `.venv` (py3.11) | `ros2_ur_ws/act_venv` (py3.12) | kanu |
|---|---|---|---|
| mujoco 3.10 / dm_control / pyzmq / cv2 / h5py | ✅ | ✗ mujoco, ✅ pyzmq cv2 | — |
| torch + lerobot 0.6.1 + transformers | ✗ | ✅ (CPU only on laptop3: no usable GPU) | ✅ `~/workspace/youngwoong/cube_flow_matching/training/lr_env` (8× A4000) |
The eval world + client run in `.venv`; the policy server runs in `act_venv` (CPU, slow) or on kanu behind
`ssh -N -L 127.0.0.1:<port>:127.0.0.1:<port> kanu` (pattern: `ros2_ur_ws/src/gello_policy/scripts/run_fm_server.sh`).
Checkpoints: stock lerobot layout `<out>/checkpoints/<step>/pretrained_model/{config.json, model.safetensors,
policy_pre/postprocessor*}`; on kanu e.g. `/home/junhyeong/workspace/youngwoong/carrot_eef/outputs/*/checkpoints/last/pretrained_model`.
The training session ("laptop3 학습") will hand over a kanu path. Read `input_features` from `config.json`
(state may be 7 (joint) or 16 (EEF variant) — the harness supports **joint-space 7/7 only** in v1 and must
refuse an EEF checkpoint with a clear message).

### 1.4 Simulation facts (read `sim_collect/README.md`, `DESIGN.md`)
- Scene/config: `sim_collect/configs/carrot_in_pot_sim.yaml` (carrot left +y, pot right −y, seeded jitter
  `random_xy_radius`/`random_yaw_deg`); `scene.build_scene(cfg, layout)`, `scene.sample_layout(cfg, seed, attempt)`.
- Arm drive: position actuators, `d.ctrl[:6] = q_target`, gripper `d.ctrl[6] = grip_cmd*255`;
  `gripper.grip_pos_from_driver(q_driver, 0.0, 0.871)` gives grip_pos 0..1. Physics 2 ms; control 250 Hz.
- Success predicate: `sim_collect/task.py::TaskEvaluator.evaluate(data, grip_cmd) -> (bool, detail)`:
  food origin inside the container opening cylinder (container frame, `-0.005 < rel_z < depth`, `r_xy < radius`)
  AND `|v_z| < 0.05` AND `grip_cmd < 0.3`. This IS the "transparent cylinder trigger" the operator described.
- Rendering: `cameras.CameraRig(xml, assets, render=cfg['render'])` → `mirror(qpos, qvel)`, `render_color(cam)`
  RGB (720,1280,3); JPEG-encode BGR with quality 92 like `capture.py`. `mujoco.Renderer` must not share a
  process with `mujoco.viewer` — the eval runs **headless, no viewer** (`MUJOCO_GL=glfw DISPLAY=:0`).
- Demos: 22 takes in `ros2_ur_ws/gello_logs/sim/take_*` (and `sim_retimed/`): `/sim_scene` (exact MJCF),
  `/sim_mj_state` (qpos/qvel/ctrl @125 Hz), `sim_object_poses` (30 Hz), `command` (q_cmd @125 Hz),
  `gripper` (grip_cmd @62 Hz), `sim_meta` (JSON; `layout_seed` is WRONG (0) for these 22 — use
  `sim_object_poses` row 0 / `sim_mj_state` row 0 for the initial layout). Replay API:
  `sim_collect/tools/replay_take.py` `load_scene/load_model/load_state/set_row`.

## 2. Architecture (all in `.venv`, one process, lockstep — no real-time pacing)

```
run_eval.py ──► EvalWorld (MuJoCo, headless)        ──► success/latch, step budget, logs
                 │  reset(seed): home pose + seeded layout + settle + wrench tare
                 │  observe(): render cam1/cam2 (30 Hz), state[7]
                 │  apply(action7): real deploy clamps → 250 Hz One-Euro+clamp_stage upsampler → ctrl
                 │  step(1/30 s): 16-17 physics steps of 2 ms with the 250 Hz tick in between
                 ▼
             Policy (interface: reset(), act(obs) -> action7 | None on fault)
               ├── ZmqPolicy     : real protocol → local act_venv server or kanu tunnel
               ├── ReplayPolicy  : replays a recorded take's `command` (+grip_cmd) at 30 Hz (harness validation)
               ├── ScriptedPolicy: oracle pick-and-place from ground-truth poses (IK via ur_kin) (positive control)
               └── ZeroPolicy    : holds the start pose (negative control, SR must be 0)
```
Lockstep means a slow policy (CPU) just makes the run slower, never changes the result. Determinism:
given (seed list, scene config, policy), the world is deterministic; ZMQ policies may be stochastic
(diffusion/FM sampling) — record `torch` seed if the server exposes it, else note it.

### 2.1 `EvalWorld` (`sim_collect/eval/world.py`)
- Build once from the yaml (+ `--config`), `mujoco.MjModel` + `MjData`, `CameraRig` for cam1/cam2 sharing the
  same xml/assets (separate MjData is fine; mirror qpos each render).
- `reset(seed) -> obs`: teleport arm to `home_joints` (ctrl = qpos, zero velocities), open gripper,
  `sample_layout(cfg, seed, attempt)` with the same collision-rejection as `sim_main._place_objects_safely`
  (import/reuse it — refactor into `scene.py` if needed, do not copy-paste), settle 0.5 s, tare wrench,
  reset the upsampler (One-Euro seeded at the actual pose, soft-start), reset the success latch.
- `observe() -> {"cam1_jpeg", "cam2_jpeg", "state": [q1..q6, grip_pos], "t": sim time, "rgb": {...} (only if video requested)}`.
  State branch: the sim starts at home on the −π branch; report `q` as-is (already near start_pose).
- `apply(action7)`: clamps (1)-(3) from §1.2 with `joint_limits_lo/hi` from the yaml `eval:` block
  (default = carrot_eef_limits.json values, copied into the yaml with provenance) and `max_dev_rad 0.5`;
  gripper ctrl = clip(a[6],0,1)*255 (identity, no threshold — as the real deploy).
- `step()`: advance 1/30 s: loop physics steps of 2 ms; every 4 ms (250 Hz) run One-Euro + `clamp_stage`
  toward the latest target and write `d.ctrl[:6]`; track `task.evaluate` each tick; success **latch** requires
  the predicate to hold for `dwell_s` (default 1.0 s) continuously; also detect failure modes: carrot fell off
  the floor plane bounds, arm joint limit violations, NaN.
- Logging: per-episode `sim_mj_state`-style npz/h5 (qpos/qvel/ctrl @125 Hz) so any episode can be replayed
  with `replay_take.py`-like tooling; optional mp4 of cam1 (and cam2) at 30 Hz via cv2 mp4v.

### 2.2 Policies (`sim_collect/eval/policies.py`)
- `ZmqPolicy(endpoint, task, timeout_s=0.6, act_timeout_faults=True)` using `obs_assembler` verbatim
  (add the path `ros2_ur_ws/src/gello_policy` to sys.path; import only stdlib modules from it — verify it does
  not import rclpy). Parse optional v2 RESET fields (`policy_type`, `checkpoint`, `state_dim`, `action_dim`);
  refuse if `state_dim != 7` or `action_dim != 7` (EEF checkpoints are out of scope in v1).
- `ReplayPolicy(take_dir)`: at each 30 Hz step return the take's `command` row nearest to the episode time
  (plus `grip_cmd` from the `gripper` table); `reset()` also returns the take's initial layout so the world
  can start from the recorded object poses (world API: `reset(seed=None, layout_override=...)`).
- `ScriptedPolicy(world_ground_truth)`: F2's oracle (see §3).
- `ZeroPolicy`: returns the home pose + grip 0.

### 2.3 `run_eval.py` CLI
```
.venv/bin/python -m sim_collect.eval.run_eval --policy zmq://127.0.0.1:5593 --task "Put carrot in pot" \
    --seeds 0-19 --max-steps 600 --dwell-s 1.0 --out sim_collect/eval/runs/<name> [--video] [--config ...]
.venv/bin/python -m sim_collect.eval.run_eval --policy scripted --seeds 0-19 --out ...
.venv/bin/python -m sim_collect.eval.run_eval --policy replay --takes ros2_ur_ws/gello_logs/sim --out ...
.venv/bin/python -m sim_collect.eval.run_eval --policy zero --seeds 0-4 --out ...
```
Outputs in `--out`: `episodes.jsonl` (`seed, outcome ∈ {success, timeout, fault, failure_reason}, n_steps,
t_success_s, detail, policy meta, wall time`), `summary.json` (SR, Wilson 95 % CI, mean steps, per-outcome
counts, config sha, git commit, checkpoint id), `summary.md` (table), per-episode `ep_<seed>.h5` (state) and
`ep_<seed>_cam1.mp4` when `--video`. Exit code 0 always (SR is data, not a test).

### 2.4 Demo validation (`sim_collect/eval/validate_success_on_demos.py`)
For each recorded take: kinematic replay of `/sim_mj_state` through `TaskEvaluator` with the dwell latch →
must report **not success at t=0** and **success by the end** for all 22 (print time-to-success per take,
and where the predicate first fires vs the recorder's `task_success_at_stop`). This is the operator's
requested check ("replay the dataset in MuJoCo and the method will show itself"). Also run
`ReplayPolicy` (dynamic replay of the recorded commands from the recorded initial layout) on all 22 and
report how many succeed — physics divergence is expected; report the number honestly.

## 3. Ownership
| owner | files | deliverable |
|---|---|---|
| F1 (fable) | `eval/__init__.py, world.py, policies.py (Zmq/Replay/Zero), run_eval.py, validate_success_on_demos.py`, tests `tests/test_eval_world.py, test_eval_policies.py, test_eval_demos.py`, yaml `eval:` block | headless world with exact deploy semantics; zero policy SR 0/5; demo predicate validation 22/22; ReplayPolicy result reported; run_eval works against a **stub ZMQ server** (F1 writes `tests/stub_policy_server.py` speaking the protocol) |
| F2 (fable) | `eval/scripted_policy.py`, `eval/ik.py` (thin wrapper over `ur_kin`), tests `tests/test_eval_scripted.py`; may add a debug-only transparent trigger-cylinder visual to renders behind a flag (never in policy observations) | oracle pick-and-place from ground truth: reach above carrot → descend → close → lift → move over pot → descend a bit → open → retreat, all as 30 Hz joint targets through `EvalWorld.apply` (so the same clamps/upsampler apply); target SR ≥ 15/20 on seeds 0-19; report failure modes |
| O1 (opus) | `eval/serve_policy.sh` (wraps run_{act,fm,diffusion}_server.sh with carrot params; `--remote kanu --gpus 6,7 --port` opens the tunnel), `eval/report.py` (aggregate runs → markdown/CSV, compare checkpoints), `eval/README.md` (Korean runbook), CLAUDE.md pointer row, `tests/test_eval_report.py` | server/tunnel wrapper tested with the stub server locally; report tool tested on synthetic runs |

Rules: `.venv` python for everything under `sim_collect/eval` (the policy server is the only act_venv/kanu process);
no new pip deps; no git commit (integrator commits); do not touch files outside your list except `scene.py`
refactors F1 needs (announce them); tests via
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider sim_collect/tests/test_eval_*.py`.
