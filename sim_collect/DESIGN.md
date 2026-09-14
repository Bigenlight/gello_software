# sim_collect — GELLO-teleoperated MuJoCo data collection (design contract)

Status: design v1, 2026-09-14. This file is the contract every implementer works
against. Change it only through the integrator (the session that owns this branch).

## 0. Goal

Collect demonstration takes in MuJoCo with the **physical GELLO** leader arm, in the
**same file family and format as the real-robot recorder** (`gello_recorder`), so the
existing converters (`scripts/dataset/convert_carrot_to_lerobot.py`,
`serl_ur_infra/ur_env/learner/recorded_demo.py`, `make_carrot_raw_stats.py`) consume sim
takes unchanged. First task: **move one vegetable/fruit into a pot or bowl on the floor.**

The existing `configs/rwh_ur.yaml` + `experiments/launch_yaml.py` sim is a *playground*
and stays untouched. `sim_collect/` is the production collection environment.

## 1. Facts established (do not re-derive)

| fact | value / source |
| --- | --- |
| Python | `/home/laptop3/gello_software/.venv/bin/python` (3.11, mujoco 3.10.0, dm_control, zmq, h5py 3.16, opencv-python-headless 5.0, scipy, pyyaml, **tkinter OK**, no PyQt5, no PIL) |
| Offscreen render | **Only `MUJOCO_GL=glfw` works** (EGL/OSMESA broken, no NVIDIA driver loaded; software GL). `DISPLAY=:0` required. A `mujoco.Renderer` **must live in a different process** from `mujoco.viewer.launch_passive` (same-process combination hangs — measured). Model needs `<visual><global offwidth="1280" offheight="720"/></visual>`. Measured: 2×1280×720 RGB = 12.6 ms, mp4v encode 3 ms/frame, 848×480 depth render+PNG 4.8 ms. 20 CPU cores. |
| GELLO | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0`, XL330 ×7 (ids 1..6 arm, 7 gripper), baud 57600. Calibration for MuJoCo: `configs/rwh_ur.yaml` (`joint_offsets [3.142,1.571,4.712,4.712,4.712,3.142]`, `joint_signs [1,1,-1,1,1,1]`, `gripper_config [7, 210.649609375, 168.849609375]`). Read via `gello.agents.gello_agent.DynamixelRobotConfig(...).make_robot(port, start_joints)` → `get_joint_state()` = 7 floats, gripper **0=open 1=closed**. **The Dynamixel driver kills any process holding the port** — only one reader process ever. Never enable torque. |
| Robot frame | `ur_kin.fk(q)` (base_link→flange, DH d=[0.1625,0,0,0.1333,0.0997,0.0996], a=[0,−0.425,−0.3922,0,0,0]) ≡ MuJoCo world pose of menagerie `ur5e.xml` `attachment_site` when the arm body is at the world origin with **no extra rotation** (the xml's `base quat="0 0 0 -1"` absorbs the UR flip). No X/Y flip anywhere. TCP = flange ⊕ 0.174 m along flange +Z (`T_tool_R`). |
| Start pose | GELLO calibration pose read today: `[-0.160, -1.563, 1.607, -1.523, -1.615, -3.118]`. At that pose (sim): joint-5 (`wrist_2_link`) origin z = **0.571 m**, TCP ≈ (−0.48, −0.05, 0.31), arm faces **−x**. |
| Reusable, ROS-free modules | `ur_gello_bringup.{ur_kin, eef_delta, bridge_stages, joint_delta, angle_utils}` (PYTHONPATH `ros2_ur_ws/src/ur_gello_bringup`); `gello_recorder.{recording_session, hdf5_writer, video_writer, depth_writer}` (PYTHONPATH `ros2_ur_ws/src/gello_recorder`); `gello.agents.gello_agent`, `gello.robots.dynamixel`. **Reuse verbatim; do not copy-paste them.** `discrete_latch`/`validate_discrete_thresholds` live in an rclpy module — re-implement those two (10 lines) in `sim_collect/gripper.py` with a bit-identity test against the originals' semantics. |
| EEF config | `ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello_eef.yaml` — `pos_scale 1.0`, `r_align_rpy [0,0,0]`, `tool_l = tool_r = [0,0,0.174,0,0,0]`, `v_max 0.16`, `w_max 1.0`, `sigma_warn 0.10`, `sigma_stop 0.03`, `hold_latch_s 2.0`, `ik_backend analytic`, `max_step_rad 0.0025 @ 250 Hz`, One-Euro `dt=1/250, min_cutoff 1.0, beta 2.0, d_cutoff 1.0`, `soft_start_s 0.7`, `anchor_agree_tol 0.02`, `filter_settled_tol 0.005`. Load the same yaml (the `gello_ur_bridge` section) so sim and real share numbers. |
| Real data format | See §4. Real cam1 = **fixed scene camera**, cam2 = **wrist camera** (`serl_ur_infra/ur_experiments/cube_in_cup.py:176`). Color 1280×720@30 mp4v BGR; depth 848×480 uint16 mm PNG in `depth.h5`, **not aligned to color**, depth K = `[426.74,0,423.94, 0,426.74,233.15, 0,0,1]` (cam1) / `[425.26,0,425.04, 0,425.26,232.82, 0,0,1]` (cam2), D = zeros, extrinsics depth→color t ≈ `[0.015, 0, 0]` m, R ≈ I. |

## 2. Processes and IPC

Three processes, launched by `sim_collect/run_sim_collect.sh` (bash, sets `MUJOCO_GL=glfw`,
`DISPLAY`, PYTHONPATH, tears everything down on Ctrl-C). All IPC is ZMQ over
`ipc:///tmp/sim_collect_<user>/…` (fallback tcp on 127.0.0.1:67xx). Messages are
**pickled dicts** (same convention as `gello/zmq_core`); keys below are the contract.

```
 GELLO(USB) ──► [A] sim_main.py ──PUB state@250Hz──► [B] capture.py (render 2 cams, record take)
                     ▲  REP :6701                          ▲  REP :6702        │ PUB preview jpeg 10 Hz
                     │                                     │                   ▼
                     └──────────── [C] gui.py (tkinter) ───┘◄──────────────────┘
                     (MuJoCo passive viewer window belongs to [A])
```

### 2.1 [A] `sim_collect/sim_main.py` — physics + leader + controller (owner: F1)

- Loads `SceneConfig` (yaml, §3), builds the MJCF via `sim_collect/scene.py`, runs
  `mujoco.viewer.launch_passive` (physics thread = the main loop, 2 ms timestep, real-time paced).
- **Leader thread** (30 Hz): reads GELLO through `DynamixelRobotConfig.make_robot(...)`,
  publishes the latest `q_lead_raw[6]`, `trigger` (0..1), timestamp into a lock-protected slot.
  Unwrap with `ur_kin.wrapped_nearest` against the previous unwrapped sample (as the ROS bridge does).
- **Control tick** (every 2nd physics step = 250 Hz): exactly the ROS bridge pipeline:
  - `joint` mode: `bridge_stages.command_pipeline(control_mode="joint", ...)` with
    `filter_stage_joint` + `clamp_stage(max_step_rad)`; when disengaged the arm HOLDS
    (`hold_when_not_engaged=True`) — the sim arm never follows an un-engaged leader.
  - `eef` mode (default): One-Euro bank (`bridge_stages.OneEuro`, dt=1/250) → `EefDeltaController.step(q_lead_f, step_eff)`
    with soft-start ramp; `None` → hold. Engage = `controller.engage(q_anchor=q_actual, q_lead_f_anchor=q_lead_f)`
    after the same gates as ROS G2–G8 (fresh leader, leader quasi-still, filter settled, IK self-test,
    `sigma_min > sigma_warn`). Disengage is immediate and ungated.
  - Gripper: `trigger` → `gripper.py` (continuous with `deadband 0.02`, or discrete latch 0.3/0.7 when
    `gripper_mode: discrete`) → `grip_cmd ∈ [0,1]` (0 open) → 2F-85 actuator ctrl `grip_cmd * 255`.
  - Writes `d.ctrl[:6] = q_cmd`, `d.ctrl[6] = grip_ctrl`. Position actuators from the menagerie xml.
- **State PUB** (every control tick, topic `state`):
  ```
  {"t": time.time(), "sim_t": d.time, "tick": int,
   "control_mode": "eef"|"joint", "eef_state": "DISENGAGED"|"HOLD"|"ENGAGED"|..., "eef_info": {...controller info dict...},
   "engaged": bool, "pos_scale": float,
   "q_lead_raw": [6], "q_lead_unwrapped": [6], "q_lead_f": [6], "qd_lead": [6], "trigger": float, "leader_t": float,
   "q_cmd": [6], "q": [6], "qd": [6], "eff": [6] (actuator_force),
   "tcp_pos": [3], "tcp_quat_xyzw": [4] (= fk(q) @ T_tool_R, world frame),
   "cmd_tcp_pos": [3], "cmd_tcp_quat_xyzw": [4] (= fk(q_cmd) @ T_tool_R),
   "wrench": [6] (fx fy fz tx ty tz from the flange force/torque sensors, sensor-site frame),
   "grip_cmd": float, "grip_pos": float (0 open..1 closed, from the 2F-85 driver joint / its range),
   "qpos_full": [nq], "qvel_full": [nv],           # for the capture process to mirror the scene
   "objects": {name: {"pos":[3], "quat_wxyz":[4]}},
   "task": {"success": bool, "detail": str}}      # geometric success test (§3.4), evaluated every tick
  ```
- **REP :6701** commands (`{"cmd": ..., ...}` → `{"ok": bool, "msg": str, ...}`):
  `get_status`, `engage`, `disengage`, `set_pos_scale {value}`, `set_control_mode {mode}` (only while disengaged),
  `reset_scene {seed?}` (disengage → teleport arm to the **current leader pose** (joint-space, like
  `MujocoRobotServer.reset_joint_state`) → open gripper → re-sample object poses → zero velocities),
  `home` (disengage → teleport to `scene.home_joints`), `gripper_pause`, `gripper_resume`, `get_scene_meta`
  (everything needed for `sim_meta`: model name, config, object list, seed, git commit, camera specs).
- Startup: exactly like the playground fix — read the leader once and teleport the sim arm there
  (`init_from_leader: true`), start **DISENGAGED**.

### 2.2 [B] `sim_collect/capture.py` — cameras + recorder (owner: F2)

- Builds the **same MJCF** (calls `scene.build()` with the same config; scene must be deterministic given
  config + object layout received from [A] via `get_scene_meta` / the `objects` field) and mirrors the state:
  on each `state` message set `d.qpos[:] = qpos_full; d.qvel[:] = qvel_full; mj_forward`.
- Renders at 30 Hz (own pacing, latest state wins): per camera one **color** render 1280×720 (fovy from config)
  and one **depth** render 848×480 from the co-located depth camera (offset by the real extrinsics t=[0.015,0,0]).
  Depth: `float32 m` → `uint16 mm`, `0` for anything beyond `depth_max_m` (10 m) or invalid; PNG-encode.
  May use two worker processes (one per camera) if a single loop cannot hold 30 Hz — measure and decide.
- **Take lifecycle** via REP :6702: `start_take {note?}`, `stop_take`, `discard_last_take`, `get_status`
  (recording?, take dir, frames, duration, rows), `snapshot` (returns current jpeg previews).
  Take directory: `<root>/take_{NN:02d}_{YYYYmmdd_HHMMSS}/` where `root` defaults to
  `ros2_ur_ws/gello_logs/sim/` (env `SIM_COLLECT_OUTPUT_ROOT`). NN = per-process counter from 1 (as real).
- Writes with **`gello_recorder.recording_session.RecordingSession` verbatim** (its `Hdf5TableWriter`,
  `Mp4FrameWriter`, `DepthH5Writer`). Fill **all nine** tables including `synchronized` (the real GUI leaves
  it empty; filling it is strictly better and compatible) plus the sim extras of §4.3.
- **PUB preview** (topic `preview`, 10 Hz): `{"cam1": jpeg_bytes(320×180), "cam2": jpeg_bytes, "recording": bool, ...}`.

### 2.3 [C] `sim_collect/gui.py` — operator GUI, tkinter (owner: O2)

Mirrors the real `gello_eef_gui` + `gello_recorder_gui` (Korean labels OK, same words as the real GUIs):
- Big primary button: `ENGAGE` (two-click confirm within 3 s, like real) / `DISENGAGE` (single click).
- `pos_scale` slider 0.10..1.00 (applies on next engage / via `set_pos_scale`).
- `Gripper PAUSE` / `Gripper Resume`.
- `RESET SCENE` (calls `reset_scene`; refuses while a take is recording), `HOME`.
- `START TAKE` / `STOP TAKE` / `DISCARD LAST`, take counter + elapsed time + rows/frames.
- Control-mode radio `eef | joint` (disabled while engaged).
- Two camera previews (cam1 scene, cam2 wrist) from the preview PUB, rendered with `tk.PhotoImage(data=PPM)`
  (no PIL) — convert jpeg→ppm with cv2 in the GUI.
- Status line: eef_state, reject/auto reason, sigma_min, task success flag.
- Keyboard: `space` = engage/disengage, `r` = start/stop take, `n` = reset scene.
- Never blocks on ZMQ (REQ with timeouts; polls at 10 Hz).

## 3. Scene (`sim_collect/scene.py` + `sim_collect/configs/*.yaml`)

### 3.1 Robot
- `robot: ur7e` — the UR7e MJCF produced by O3 under `sim_collect/assets/robots/ur7e/` (menagerie
  `ur5e.xml` structure, ur7e kinematics/limits; if kinematics are identical it is an explicit alias with a
  README note). Gripper `third_party/mujoco_menagerie/robotiq_2f85/2f85.xml` attached at `attachment_site`
  (reuse `gello.robots.sim_robot.attach_hand_to_arm`). Arm body at the world origin, base on the floor (z=0).
- Sensors: `<site name="ft_site">` at the flange + `<force>` + `<torque>` sensors → `wrench`.
- `<visual><global offwidth="1280" offheight="720"/></visual>`.
- Keyframe `home` = §1 start pose (arm) + gripper open.

### 3.2 Floor
- One `plane` geom at z=0 covering ≥ 4×4 m, textured with a **LIBERO table wood texture** (vendored under
  `sim_collect/assets/textures/`, attribution file next to it), `texrepeat` so the grain is ~real scale.
  No table geometry: the workspace *is* the floor.

### 3.3 Objects and layout ("left = food, right = containers")
Convention: **left/right are as seen from the robot base looking along the arm's forward direction (−x)**,
i.e. left = **−y**, right = **+y** (the config exposes both regions so the user can swap).
- Food candidates (all vendored under `sim_collect/assets/objects/`, each its own MJCF with a `<body>` with
  `freejoint`, convex collision, realistic mass 0.05–0.3 kg, listed in the yaml; user picks later):
  carrot (procedural if no mesh), plus 3–5 meshes from O1's research (e.g. YCB apple/banana/lemon/orange).
- Containers on the right: one **pot** and one **bowl** (from O1), open side up, resting on the floor.
- `layout:` yaml lists each object with `nominal_pos`, `random_xy_radius`, `random_yaw`; `reset_scene`
  samples from a seeded RNG, drops objects 2 cm above the floor and settles physics for 0.5 s (fast-forward,
  not real time) before reporting ready. Objects must not spawn intersecting each other or the robot.

### 3.4 Task success (`sim_collect/task.py`)
`carrot_in_pot`-style: success when the chosen food's origin is inside the container's opening cylinder
(above container floor, below rim, within radius) AND its z-velocity < 0.05 m/s AND the gripper is open
(`grip_cmd < 0.3`). Evaluated every tick in [A]; recorded per row in the extras table and as
`task_success` in `sim_meta` at stop_take. The success test is informational (converters take `--outcome`
from the human).

### 3.5 Cameras (named exactly `cam1` = scene, `cam2` = wrist, matching real data)
- `cam1` (front/scene): position **0.70 m from the base origin along −x, z = 0.571 m** (joint-5 height at the
  start pose), looking back toward +x and pitched down at the workspace centre ≈ (−0.45, 0, 0). Color fovy
  **42°** (D435 color vertical FOV), depth camera co-located (+0.015 m along its x) with fovy **58.7°**
  (matches depth K fx=426.7 at 848×480).
- `cam2` (wrist): body attached to `wrist_3_link`, mounted on the side of the flange **farthest from the base**
  in the start pose (i.e. offset along the flange's local axis that points along −x world at start pose),
  ~0.05 m radial offset, looking along the tool +Z with a ~15° pitch toward the fingertips so the fingertips
  are visible at the bottom of the frame. Same fovy pair. All numbers in the yaml (`cameras:` block) — they are
  the user's to tune.
- The capture process writes real-looking `camera_info` (copy the real K/D/R/P values from §1 for each cam,
  width 848, height 480, `distortion_model plumb_bob`, `frame_id cam{N}_depth_optical_frame`) and
  `extrinsics_depth_to_color` (`rotation` = I column-major, `translation` = [0.015, 0, 0]) — identical across takes.

## 4. Recorded format (must match `gello_recorder` byte-for-byte in structure)

### 4.1 Take dir: `vectors.h5`, `cam1.mp4`, `cam2.mp4`, `depth.h5` — nothing else.
### 4.2 `vectors.h5` — the nine groups written by `RecordingSession`:
`synchronized` (56 cols, sample at `sample_rate_hz` 100), `gello_joint_states` (`t_rel_s,q1..6,qd1..6` = leader
raw joints + finite-diff qd, at leader rate 30 Hz), `ur_joint_states` (`t_rel_s,q1..6,qd1..6,eff1..6`),
`command` (`t_rel_s,cmd1..6` = `q_cmd`), `gripper` (`t_rel_s,gello_grip,grip_cmd,grip_pos`, 0 open..1 closed),
`wrench` (`t_rel_s,fx..tz`), `tcp_pose` (`t_rel_s,x,y,z,qx,qy,qz,qw` = actual TCP = `fk(q) @ T_tool_R`),
`cam1_frames`, `cam2_frames` (`t_rel_s,frame_idx`, one row per written mp4 frame, 0-based contiguous).
Rates: write `ur_joint_states/command/tcp_pose/wrench` at **125 Hz** (every 2nd state message; real is 63 Hz),
`gripper` at ≥30 Hz, no stream may gap > 0.2 s. `t_rel_s` = `time.time() - t0` of the RecordingSession, `.4f`.
Joint order = UR order. `qd` = actual `d.qvel[:6]`, `eff` = `d.actuator_force[:6]`.
### 4.3 Sim extras (new groups, same table convention, leading `t_rel_s`), prefix `sim_`:
- `sim_object_poses`: `t_rel_s, <obj>_x,_y,_z,_qx,_qy,_qz,_qw` for every object, 30 Hz.
- `sim_control`: `t_rel_s, engaged, eef_state_code, pos_scale, sigma_min, gamma, ls_scale, task_success`.
- `sim_leader_filtered`: `t_rel_s, qf1..qf6` (One-Euro output) — lets us reproduce the controller offline.
- File-level attrs on `vectors.h5`: `sim_meta` = JSON string {`sim_collect_version`, `git_commit`, `robot`,
  `control_mode`, `gripper_mode`, `pos_scale`, `scene_config` (full yaml dump), `layout_seed`, `objects`,
  `chosen_food`, `container`, `task_success_at_stop`, `cameras` (poses, fovy), `mujoco_version`}.
  The real files have **no** file attrs; adding attrs breaks no consumer.
### 4.4 Videos/depth: exactly as real (§1 row "Real data format"). BGR into `cv2.VideoWriter`.

## 5. Ownership, files, tests

| owner | files | must deliver |
| --- | --- | --- |
| F1 sim core | `sim_collect/{sim_main.py, scene.py, leader.py, controller.py, gripper.py, ipc.py, task.py}`, `sim_collect/configs/carrot_in_pot_sim.yaml` | runs with the real GELLO and with `--fake-leader` (scripted leader for tests); joint+eef modes; unit tests in `sim_collect/tests/` (fake leader, no GELLO, headless: `MUJOCO_GL=glfw` still needs DISPLAY — tests that need rendering are marked and skipped without DISPLAY; controller/scene/task tests must not need a display) |
| F2 capture | `sim_collect/capture.py`, `sim_collect/recorder.py`, `sim_collect/cameras.py` | format parity: a recorded fake-leader take passes `scripts/dataset/validate_carrot_conversion.py`-style checks and `recorded_demo.convert_recorded_take` + `convert_carrot_to_lerobot.load_take_arrays`; tests |
| O1 assets | `sim_collect/assets/**`, `sim_collect/assets/ATTRIBUTION.md`, `sim_collect/tools/fetch_assets.py` | tested-loading MJCF for floor texture, ≥4 foods, pot, bowl; sizes; licenses |
| O2 gui | `sim_collect/gui.py` | works against a **stub** server (`sim_collect/tests/stub_servers.py` that O2 writes from §2 contract) before integration |
| O3 ur7e | **DONE by integrator** — `sim_collect/assets/robots/ur7e/ur7e.xml` (menagerie ur5e structure, exact URDF offsets, meshes via `meshdir` to menagerie; FK == `ur_kin.fk` to 0.000 mm). Load with `mujoco.MjModel.from_xml_path` or dm_control `mjcf.from_path`. | — |
| O4 docs+launcher+tests | `sim_collect/README.md` (Korean, operator runbook), `sim_collect/run_sim_collect.sh`, `sim_collect/tests/test_format_parity.py`, CLAUDE.md pointer | passes on this machine |

### 5.0 Integration changes after the first implementation round (2026-09-14 evening)
- `ipc.py` frames are single-part (`topic\0pickle`) so `Subscriber(..., conflate=True)` (ZMQ_CONFLATE) can hand
  consumers the truly newest state: render workers and the GUI preview use it. Before this the 30 Hz render
  workers saw states 0.5–2.4 s old (kernel socket buffers queue frames regardless of RCVHWM). The recorder
  subscribes without conflate (needs every message) with a deep pipe (`Publisher` SNDHWM default 4000).
- `recorder.py` swaps `RecordingSession`'s table writer for `BufferedHdf5TableWriter` (same on-disk layout;
  rows buffered and written in blocks every 1 s / on flush / before close): 904 → 6.6 µs per row. Without it
  the 125 Hz tables fell to ~78 Hz under CPU contention.
- `cameras.py` render quality from the yaml `render:` block (`capture_offsamples` 0, `capture_shadows` true,
  `capture_reflections` false, `capture_skybox` false). Measured on the real scene at 1280×720: 20.1 ms
  baseline → 16.3 (no MSAA) → 3.6 ms (also no shadow/reflection/skybox). The viewer's own shadowmap is
  `render.viewer_shadowsize` (F1).
- Wrench is tared after every teleport+settle (startup/reset/home) like the real UR's zeroed F/T; raw in `wrench_raw`.
- Leader calibration: the ROS `ur7e_gello.yaml` (`gello_publisher`) is the single source; home = the real
  robot's home (J1 ≈ −3.30, TCP on base-frame +x, matching `take_18`); scene/cameras mirrored to +x (F1, R1 #1).

### 5.1 Provided by the integrator (use, do not rewrite)
- `sim_collect/ipc.py` — `endpoint(name)`, `Publisher(name).send(topic, dict)`, `Subscriber(name, topic).latest()/recv()`,
  `Server(name).poll(handler, timeout_ms)`, `Client(name, timeout_ms).call(cmd, **kw)`, `wait_for(client)`.
  Endpoint names: `sim_rep`, `capture_rep`, `state_pub`, `preview_pub`. ipc:// by default, `SIM_COLLECT_IPC=tcp` for ports.
- `sim_collect/tests/conftest.py` — sys.path setup + `needs_display` marker. **Do not create another conftest.**
- `sim_collect/assets/robots/ur7e/ur7e.xml` — the robot.

### 5.2 Asset interface (O1 delivers, F1 consumes)
- Floor texture: `sim_collect/assets/textures/<name>.png` (power-of-two, ≤ 2048²), referenced from the yaml
  `floor.texture`.
- Each object: `sim_collect/assets/objects/<name>/<name>.xml` — a complete `<mujoco model="<name>">` file whose
  `<worldbody>` holds **exactly one top-level `<body name="<name>">`** (no freejoint inside; `scene.py` attaches it
  with dm_control `mjcf.from_path` → `arena.worldbody.attach(obj)` and adds `freejoint` on the attachment frame),
  `<asset>` meshes relative to that file (`meshdir="."` or `assets/`), collision geoms convex (mesh or primitives,
  `group="3"`, `contype/conaffinity` default), visual geoms `group="2"` `contype="0" conaffinity="0"`, a
  realistic `mass` on the body via inertial or geom density, and an `<site name="<name>_center">` at the body
  origin. Containers additionally provide `<site name="<name>_opening" pos="0 0 <rim_z>" size="<radius>">`
  and `<site name="<name>_floor" pos="0 0 <inner_floor_z>">` so `task.py` can test "inside" geometrically.
  Object origin at the bottom of the object (resting on z=0 means body pos z=0).
- `sim_collect/assets/ATTRIBUTION.md` lists every asset with source URL and license.
- Test: `sim_collect/tests/test_assets.py` loads every object xml standalone, drops it from 5 cm and asserts it
  settles (|v| < 1e-3) with z ≥ −1e-3 after 1 s.

### 5.3 Test file ownership (no shared test files besides conftest.py)
F1: `test_controller.py, test_scene.py, test_task.py, test_leader.py, test_gripper.py`; F2: `test_recorder.py,
test_cameras.py, test_capture.py`; O1: `test_assets.py`; O2: `test_gui.py, stub_servers.py`; O4:
`test_format_parity.py, test_launcher.py`. Integrator: `test_ipc.py`.

Shared rules: `.venv` python only; no new pip deps without the integrator's OK (allowed already: zmq, h5py,
cv2-headless, scipy, yaml, tkinter); no edits outside `sim_collect/` except the CLAUDE.md pointer (O4) and
`third_party/mujoco_menagerie` is read-only; every ZMQ key above is the contract — add keys freely, never
rename; type hints + docstrings; tests via
`cd /home/laptop3/gello_software && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider sim_collect/tests` (the ROS overlay in the shell otherwise injects launch_testing plugins that need lark).
