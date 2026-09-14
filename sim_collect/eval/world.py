"""EvalWorld — headless MuJoCo world with the REAL policy-deploy execution semantics
(sim_collect/eval/DESIGN.md §2.1).

One process, one `MjModel`/`MjData`, one `CameraRig`, lockstep (no real-time pacing):

    reset(seed) -> obs          teleport arm to home_joints (or q0), open gripper, seeded
                                layout with sim_main's collision rejection
                                (scene.place_objects_safely), settle, wrench tare,
                                upsampler re-seed (One-Euro at the actual pose, soft start),
                                success latch reset
    observe() -> obs            {"cam1_jpeg", "cam2_jpeg", "state": [q1..q6, grip_pos], "t"}
                                JPEGs are full 1280x720, uncropped, BGR-encoded by cv2 (q 92),
                                exactly what the real client sends over ZMQ
    apply(action7)              the deploy client's clamps IN ORDER (policy_leader_node
                                `_tick_execute`): (1) envelope joint_limits_lo/hi,
                                (2) |target - live_q| <= max_dev_rad, (3) grip clip 0..1;
                                the joint target then becomes the 250 Hz bridge's raw target
    step() -> info              advance exactly 1/30 s: physics at 2 ms, every 4 ms the
                                bridge tick (One-Euro + bridge_stages.clamp_stage via
                                `command_pipeline`, joint mode — bit-identical to the ROS
                                bridge) writes d.ctrl[:6]; d.ctrl[6] = grip*255; the task
                                predicate is evaluated every tick and latched after `dwell_s`

Failure detection (episode ends, outcome "failure"): NaN in the state, an arm joint
outside the model's joint range, an object outside `object_bounds_xy_m` / below
`object_min_z_m`. Per-episode state log at 125 Hz (qpos/qvel/ctrl, the recorder's
`sim_mj_state` columns) via `save_episode`; `load_episode_state` reads it back into a
`replay_take.TakeState`. Optional cam1/cam2 mp4 (mp4v, 30 fps, BGR) with `video=True`.

Everything numeric comes from the yaml `eval:` block (see carrot_in_pot_sim.yaml for the
provenance of the envelope). `mujoco.viewer` is never imported here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402

from sim_collect.controller import load_bridge_params  # noqa: E402
from sim_collect.gripper import grip_pos_from_driver  # noqa: E402
from sim_collect.scene import (REPO_ROOT, SceneConfig, apply_layout, build_scene, place_objects_safely,  # noqa: E402
                               robot_body_ids, sample_layout)
from sim_collect.task import TaskConfig, TaskEvaluator, set_free_body_pose  # noqa: E402
from ur_gello_bringup import ur_kin  # noqa: E402
from ur_gello_bringup.angle_utils import wrapped_nearest  # noqa: E402
from ur_gello_bringup.bridge_stages import OneEuro, command_pipeline  # noqa: E402

N_JOINTS = 6
STATE_LOG_HZ = 125.0
COLOR_SIZE = (1280, 720)

EVAL_DEFAULTS: Dict[str, Any] = {
    "joint_limits_lo": [-3.8049, -1.8055, 1.2649, -2.6913, -1.9309, -4.6440],
    "joint_limits_hi": [-2.5787, -0.8823, 2.1442, -1.4379, -1.2014, -1.8222],
    "max_dev_rad": 0.5,
    "start_pose": [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692],
    "bridge": {
        "filter_type": "one_euro", "one_euro_min_cutoff": 3.0, "one_euro_beta": 4.0, "one_euro_d_cutoff": 1.0,
        "ema_alpha": 0.4, "max_step_rad": 0.0025, "deadband_rad": 0.004, "soft_start_s": 0.7,
        "publish_rate_hz": 250.0,
    },
    "policy_hz": 30.0, "max_steps": 600, "dwell_s": 1.0, "settle_s": 0.5,
    "seeds": list(range(20)), "object_bounds_xy_m": 1.5, "object_min_z_m": -0.05, "jpeg_quality": 92,
}


def _merge_defaults(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = json.loads(json.dumps(EVAL_DEFAULTS))
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def config_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Outcome:
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FAULT = "fault"
    FAILURE = "failure"


class EvalWorld:
    """See module docstring. Not thread-safe; one instance per process (memory)."""

    def __init__(self, config_path: str = "sim_collect/configs/carrot_in_pot_sim.yaml", *,
                 video: bool = False, render: Optional[bool] = None) -> None:
        self.cfg = SceneConfig.load(config_path)
        self.config_path = self.cfg.path or os.path.abspath(config_path)
        self.config_sha256 = config_sha256(self.config_path)
        self.ev = _merge_defaults(self.cfg.raw.get("eval"))
        self.video = bool(video)
        self._render_enabled = True if render is None else bool(render)

        # ---- model (built at the nominal layout; reset() places the objects) ----
        self.built = build_scene(self.cfg, sample_layout(self.cfg, int(self.cfg.layout.get("seed", 0))))
        self.meta = self.built.meta
        self.xml_sha256 = hashlib.sha256(self.built.xml.encode()).hexdigest()
        self.model = self.built.load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = float(m.opt.timestep)
        self.bridge_hz = float(self.ev["bridge"].get("publish_rate_hz", 250.0))
        self.steps_per_tick = max(1, int(round(1.0 / (self.bridge_hz * self.dt))))
        self.bridge_hz = 1.0 / (self.steps_per_tick * self.dt)
        self.policy_hz = float(self.ev.get("policy_hz", 30.0))
        self.frame_dt = 1.0 / self.policy_hz
        self.steps_per_log = max(1, int(round(1.0 / (STATE_LOG_HZ * self.dt))))
        names = self.meta["names"]
        self.grip_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, names["gripper_driver_joint"])
        self.grip_qadr = int(m.jnt_qposadr[self.grip_jnt])
        self.grip_range = [float(v) for v in names["gripper_driver_range"]]
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, names["gripper_actuator"])
        fs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, names["force_sensor"])
        ts = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, names["torque_sensor"])
        self.force_adr = int(m.sensor_adr[fs])
        self.torque_adr = int(m.sensor_adr[ts])
        self.obj_frame_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["frame_body"])
                              for n, o in self.meta["objects"].items()}
        self.gripper_qpos = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                             if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and j >= N_JOINTS]
        self.robot_bodies = robot_body_ids(m, self.meta)
        self.arm_range = np.asarray(m.jnt_range[:N_JOINTS], dtype=float)
        ctl = self.cfg.control
        self.bridge_params = load_bridge_params(ctl.get("bridge_params", []), str(ctl.get("bridge_node", "gello_ur_bridge")))
        self.T_tool_R = ur_kin.xyz_rpy_to_T(self.bridge_params["tool_r_xyz_rpy"])
        self.task_cfg = TaskConfig.from_scene(self.cfg.task, self.meta)
        self.task = TaskEvaluator(m, self.task_cfg)

        # ---- deploy clamps + bridge params ----
        self.home_joints = self.cfg.home_joints.copy()
        self.start_pose = np.asarray(self.ev["start_pose"], dtype=float).reshape(N_JOINTS)
        self.joint_limits_lo = np.asarray(self.ev["joint_limits_lo"], dtype=float).reshape(N_JOINTS)
        self.joint_limits_hi = np.asarray(self.ev["joint_limits_hi"], dtype=float).reshape(N_JOINTS)
        self.max_dev_rad = float(self.ev["max_dev_rad"])
        if np.any(self.home_joints < self.joint_limits_lo) or np.any(self.home_joints > self.joint_limits_hi):
            raise ValueError(f"home_joints {self.home_joints.tolist()} outside eval joint_limits")
        b = self.ev["bridge"]
        self.use_euro = str(b.get("filter_type", "one_euro")) == "one_euro"
        self.max_step_rad = float(b["max_step_rad"])
        self.soft_start_s = float(b["soft_start_s"])
        self.deadband_rad = float(b["deadband_rad"])
        self.ema_alpha = float(b["ema_alpha"])
        self.dwell_s = float(self.ev["dwell_s"])
        self.settle_s = float(self.ev.get("settle_s", self.cfg.layout.get("settle_s", 0.5)))
        self.jpeg_quality = int(self.ev.get("jpeg_quality", 92))
        self.object_bounds_xy = float(self.ev.get("object_bounds_xy_m", 1.5))
        self.object_min_z = float(self.ev.get("object_min_z_m", -0.05))

        # ---- runtime state ----
        self._rig = None
        self._euro: Optional[List[OneEuro]] = None
        self._filtered: List[float] = []
        self._gated: List[float] = []
        self._last_published: List[float] = []
        self._raw_target: Optional[List[float]] = None
        self._last_input_t: Optional[float] = None
        self._seed_time = 0.0
        self._grip_cmd = 0.0
        self.step_eff = self.max_step_rad
        self.wrench_tare = np.zeros(6)
        self.layout: Dict[str, Any] = {}
        self.seed: Optional[int] = None
        self.reset_note = ""
        self._t0 = 0.0
        self._phys = 0
        self._tick = 0
        self.frame = 0
        self.n_clamped_limit = 0
        self.n_clamped_dev = 0
        self.clamp_hits: Dict[str, Any] = self._new_clamp_hits()
        self.last_apply: Dict[str, Any] = {}
        self.last_tick_max_delta = 0.0
        self._log_rows: List[np.ndarray] = []
        self._video: Dict[str, Any] = {}
        self._video_dir: Optional[str] = None
        self._video_tag = ""
        self._reset_latch()

    @staticmethod
    def _new_clamp_hits() -> Dict[str, Any]:
        """Per-episode clamp statistics: steps on which each clamp bound (+ per joint)."""
        return {"steps": 0, "envelope": 0, "max_dev": 0, "grip": 0,
                "envelope_per_joint": [0] * N_JOINTS, "max_dev_per_joint": [0] * N_JOINTS}

    # ------------------------------------------------------------------ #
    # Info for policies (ScriptedPolicy reads ground truth through these) #
    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "home_joints": self.home_joints.tolist(), "start_pose": self.start_pose.tolist(),
            "joint_limits_lo": self.joint_limits_lo.tolist(), "joint_limits_hi": self.joint_limits_hi.tolist(),
            "max_dev_rad": self.max_dev_rad, "policy_hz": self.policy_hz, "dwell_s": self.dwell_s,
            "objects": {n: {"kind": o["kind"], "radius_m": o["radius_m"], "size": o["size"]}
                        for n, o in self.meta["objects"].items()},
            "task": {"name": self.task_cfg.name, "food": self.task_cfg.food, "container": self.task_cfg.container},
            "layout": self.layout, "seed": self.seed, "tool_r_xyz_rpy": list(self.bridge_params["tool_r_xyz_rpy"]),
            "config_sha256": self.config_sha256, "xml_sha256": self.xml_sha256,
        }

    def set_envelope(self, lo: Optional[Sequence[float]], hi: Optional[Sequence[float]]) -> None:
        """Replace clamp (1)'s envelope; `None, None` = the model's own joint range (i.e. only
        the max-deviation clamp binds). The yaml default is the REAL dataset's 1.2x envelope,
        which the SIM demos leave on 24 % of their steps (2026-09-15 replay measurement) —
        use this deliberately when evaluating a sim-trained checkpoint."""
        if lo is None or hi is None:
            self.joint_limits_lo = self.arm_range[:, 0].copy()
            self.joint_limits_hi = self.arm_range[:, 1].copy()
        else:
            self.joint_limits_lo = np.asarray(lo, dtype=float).reshape(N_JOINTS)
            self.joint_limits_hi = np.asarray(hi, dtype=float).reshape(N_JOINTS)
        self.ev["joint_limits_lo"] = self.joint_limits_lo.tolist()
        self.ev["joint_limits_hi"] = self.joint_limits_hi.tolist()

    def q(self) -> np.ndarray:
        """Current arm joints (rad, UR order, as simulated — on the -pi branch)."""
        return np.asarray(self.data.qpos[:N_JOINTS], dtype=float).copy()

    def grip_pos(self) -> float:
        return grip_pos_from_driver(float(self.data.qpos[self.grip_qadr]), *self.grip_range)

    def tcp_pose(self) -> np.ndarray:
        """4x4 world pose of the tool point (fk(q) @ T_tool_R — same as sim_main's `tcp_pos`)."""
        return ur_kin.fk(self.q()) @ self.T_tool_R

    def object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """(pos[3], quat_wxyz[4]) of object `name`'s attachment frame (= its free joint)."""
        bid = self.obj_frame_ids[name]
        return (np.asarray(self.data.xpos[bid], dtype=float).copy(),
                np.asarray(self.data.xquat[bid], dtype=float).copy())

    def object_site_pos(self, name: str, which: str = "center") -> np.ndarray:
        """World position of `<name>_center` / `_opening` / `_floor` (KeyError if absent)."""
        key = {"center": "center_site", "opening": "opening_site", "floor": "floor_site"}[which]
        site = self.meta["objects"][name].get(key)
        if not site:
            raise KeyError(f"object {name!r} has no {which} site")
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        return np.asarray(self.data.site_xpos[sid], dtype=float).copy()

    def set_object_pose(self, name: str, pos: Sequence[float], quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport an object (tests / debugging only — never part of an evaluation)."""
        set_free_body_pose(self.model, self.data, self.meta["objects"][name]["frame_body"], pos, quat_wxyz)
        mujoco.mj_forward(self.model, self.data)

    @property
    def t(self) -> float:
        """Episode time (s) since reset()."""
        return float(self.data.time) - self._t0

    def wrench(self) -> np.ndarray:
        return self._wrench_raw() - self.wrench_tare

    def _wrench_raw(self) -> np.ndarray:
        d = self.data
        return np.concatenate([d.sensordata[self.force_adr:self.force_adr + 3],
                               d.sensordata[self.torque_adr:self.torque_adr + 3]]).astype(float)

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #
    def _teleport_arm(self, q: Sequence[float], open_gripper: bool = True) -> None:
        d = self.data
        d.qpos[:N_JOINTS] = np.asarray(q, dtype=float)
        d.qvel[:N_JOINTS] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[:N_JOINTS] = np.asarray(q, dtype=float)
        if open_gripper:
            for a in self.gripper_qpos:
                d.qpos[a] = 0.0
            d.ctrl[self.grip_act] = 0.0
            self._grip_cmd = 0.0
        mujoco.mj_forward(self.model, d)

    def _settle(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.dt))):
            mujoco.mj_step(self.model, self.data)

    def _reset_latch(self) -> None:
        self._ok_since: Optional[float] = None
        self.success = False
        self.t_first_ok: Optional[float] = None
        self.t_success: Optional[float] = None
        self.failure: Optional[str] = None
        self.task_detail = ""

    def _reset_upsampler(self) -> None:
        """Seed the bridge chain from the ACTUAL pose (zero jump), restart the soft start —
        the bridge's SEED branch after a (re)seed (controller.TeleopController._tick)."""
        q = [float(v) for v in self.data.qpos[:N_JOINTS]]
        b = self.ev["bridge"]
        if self.use_euro:
            self._euro = [OneEuro(1.0 / self.bridge_hz, float(b["one_euro_min_cutoff"]), float(b["one_euro_beta"]),
                                  float(b["one_euro_d_cutoff"])) for _ in range(N_JOINTS)]
            for i in range(N_JOINTS):
                self._euro[i].seed(q[i])
        else:
            self._euro = None
        self._filtered = list(q)
        self._gated = list(q)
        self._last_published = list(q)
        self._raw_target = None
        self._last_input_t = None
        self._seed_time = float(self.data.time)
        self.step_eff = self.max_step_rad * (0.15 if self.soft_start_s > 0 else 1.0)

    def reset(self, seed: Optional[int] = None, layout_override: Optional[Dict[str, Any]] = None,
              q0: Optional[Sequence[float]] = None, video_dir: Optional[str] = None,
              video_tag: str = "") -> Dict[str, Any]:
        """Start an episode. `seed` -> seeded layout (sim_main's rejection sampling);
        `layout_override` {name: {"pos", "quat_wxyz"|"yaw"}} places the objects exactly
        there instead (ReplayPolicy); `q0` overrides the arm start pose (default home_joints).
        Returns the first observation."""
        self._close_video()
        mujoco.mj_resetData(self.model, self.data)
        q_start = self.home_joints if q0 is None else np.asarray(q0, dtype=float).reshape(N_JOINTS)
        self._teleport_arm(q_start, open_gripper=True)
        self.reset_note = ""
        if layout_override is not None:
            self.seed = None if seed is None else int(seed)
            self.layout = {n: dict(v) for n, v in layout_override.items() if not n.startswith("_")}
            self.layout["_override"] = True
            apply_layout(self.model, self.data, self.meta, self.layout)
        else:
            self.seed = int(self.cfg.layout.get("seed", 0)) if seed is None else int(seed)
            self.layout, self.reset_note, _ = place_objects_safely(
                self.cfg, self.model, self.data, self.meta, self.seed, robot_bodies=self.robot_bodies,
                T_tool_R=self.T_tool_R, teleport_home=lambda: self._teleport_arm(self.home_joints, open_gripper=True))
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._settle(self.settle_s)
        self.wrench_tare = self._wrench_raw()
        self._t0 = float(self.data.time)
        self._phys = 0
        self._tick = 0
        self.frame = 0
        self.n_clamped_limit = 0
        self.n_clamped_dev = 0
        self.clamp_hits = self._new_clamp_hits()
        self.last_apply = {}
        self.last_tick_max_delta = 0.0
        self._log_rows = []
        self._reset_upsampler()
        self._reset_latch()
        self._log_row()
        if self.video:
            self._open_video(video_dir, video_tag)
        return self.observe()

    # ------------------------------------------------------------------ #
    # Observe                                                              #
    # ------------------------------------------------------------------ #
    def _get_rig(self):
        if self._rig is None:
            from sim_collect.cameras import CameraRig
            self._rig = CameraRig(self.built.xml, self.built.assets, render=self.cfg.render)
        return self._rig

    def render(self, cam: str) -> np.ndarray:
        """RGB uint8 (720, 1280, 3) of `cam` at the current state."""
        rig = self._get_rig()
        rig.mirror(self.data.qpos, self.data.qvel)
        return rig.render_color(cam)

    def observe(self, images: bool = True) -> Dict[str, Any]:
        """{"cam1_jpeg", "cam2_jpeg", "state": [q1..q6, grip_pos], "t"} (+ "rgb" with video).
        `images=False` skips rendering (for policies that do not look)."""
        state = [float(v) for v in self.data.qpos[:N_JOINTS]] + [self.grip_pos()]
        obs: Dict[str, Any] = {"state": state, "t": self.t, "frame": self.frame, "cam1_jpeg": None, "cam2_jpeg": None}
        if images and self._render_enabled:
            import cv2
            rig = self._get_rig()
            rig.mirror(self.data.qpos, self.data.qvel)
            rgb: Dict[str, np.ndarray] = {}
            for cam in ("cam1", "cam2"):
                img = rig.render_color(cam)
                bgr = np.ascontiguousarray(img[:, :, ::-1])
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    raise RuntimeError(f"JPEG encode failed for {cam}")
                obs[f"{cam}_jpeg"] = buf.tobytes()
                if self.video:
                    rgb[cam] = img
                    w = self._video.get(cam)
                    if w is not None:
                        w.write(bgr)
            if self.video:
                obs["rgb"] = rgb
        return obs

    # ------------------------------------------------------------------ #
    # Apply (deploy clamps) + step (bridge upsampler + physics)            #
    # ------------------------------------------------------------------ #
    def apply(self, action7: Sequence[float]) -> Dict[str, Any]:
        """policy_leader_node._tick_execute's safety clamp, in order; the result becomes
        the 250 Hz bridge's raw target (bridge ingest = wrapped_nearest to the previous
        raw target + One-Euro speed-estimate update at the 30 Hz cadence)."""
        a = [float(v) for v in action7]
        if len(a) != N_JOINTS + 1 or not all(math.isfinite(v) for v in a):
            raise ValueError(f"action must be 7 finite floats, got {action7!r}")
        live_q = wrapped_nearest([float(v) for v in self.data.qpos[:N_JOINTS]], self.start_pose.tolist())
        target = list(a[:N_JOINTS])
        clamped_limit = False
        clamped_dev = False
        ch = self.clamp_hits
        ch["steps"] += 1
        for i in range(N_JOINTS):
            v = target[i]
            if v < self.joint_limits_lo[i]:          # (a) envelope (OOD guard)
                v = float(self.joint_limits_lo[i]); clamped_limit = True; ch["envelope_per_joint"][i] += 1
            elif v > self.joint_limits_hi[i]:
                v = float(self.joint_limits_hi[i]); clamped_limit = True; ch["envelope_per_joint"][i] += 1
            lo_d = live_q[i] - self.max_dev_rad      # (b) max deviation from the live pose
            hi_d = live_q[i] + self.max_dev_rad
            if v < lo_d:
                v = lo_d; clamped_dev = True; ch["max_dev_per_joint"][i] += 1
            elif v > hi_d:
                v = hi_d; clamped_dev = True; ch["max_dev_per_joint"][i] += 1
            target[i] = v
        grip = a[N_JOINTS]
        grip_clipped = grip < 0.0 or grip > 1.0
        grip = 0.0 if grip < 0.0 else (1.0 if grip > 1.0 else grip)   # (c) grip clip, no threshold
        self.n_clamped_limit += int(clamped_limit)
        self.n_clamped_dev += int(clamped_dev)
        ch["envelope"] += int(clamped_limit)
        ch["max_dev"] += int(clamped_dev)
        ch["grip"] += int(grip_clipped)
        # bridge ingest (_on_joint_state): one unwrap chain + One-Euro input-speed update
        prev = self._raw_target
        unwrapped = target if prev is None else wrapped_nearest(target, prev)
        now = float(self.data.time)
        dt = None if self._last_input_t is None else now - self._last_input_t
        if self._euro is not None:
            for i in range(N_JOINTS):
                self._euro[i].update_input(unwrapped[i], dt)
        self._raw_target = list(unwrapped)
        self._last_input_t = now
        self._grip_cmd = float(grip)
        self.data.ctrl[self.grip_act] = 255.0 * self._grip_cmd
        self.last_apply = {"target": list(target), "grip": self._grip_cmd,
                           "clamped_limit": clamped_limit, "clamped_dev": clamped_dev}
        return self.last_apply

    def _bridge_tick(self) -> None:
        """One 250 Hz publish tick = the bridge's `_on_timer` after the seed: soft-start
        step, filter stage (One-Euro) + clamp stage. Holds (no ctrl change) until the
        first apply() — a policy that never acts leaves the arm exactly where it is."""
        if self._raw_target is None:
            return
        step = self.max_step_rad
        if self.soft_start_s > 0.0:
            frac = (float(self.data.time) - self._seed_time) / self.soft_start_s
            if frac < 1.0:
                step = self.max_step_rad * (0.15 + 0.85 * max(0.0, frac))
        self.step_eff = step
        out = command_pipeline("joint", False, self._raw_target, self._filtered, self._gated, self._euro,
                               self.ema_alpha, self.deadband_rad, self._last_published, step)
        delta = max(abs(out[i] - self._last_published[i]) for i in range(N_JOINTS))
        self.last_tick_max_delta = max(self.last_tick_max_delta, delta)
        self._last_published = out
        self.data.ctrl[:N_JOINTS] = out

    def _log_row(self) -> None:
        d = self.data
        self._log_rows.append(np.concatenate([[self.t, float(d.time), float(self._tick)],
                                              d.qpos, d.qvel, d.ctrl]).astype(np.float64))

    def _check_failure(self) -> Optional[str]:
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return "nan_state"
        q = d.qpos[:N_JOINTS]
        if np.any(q < self.arm_range[:, 0] - 1e-6) or np.any(q > self.arm_range[:, 1] + 1e-6):
            return "joint_limit_violation"
        for name, bid in self.obj_frame_ids.items():
            p = d.xpos[bid]
            if float(np.hypot(p[0], p[1])) > self.object_bounds_xy or float(p[2]) < self.object_min_z:
                return f"object_out_of_bounds:{name}"
        return None

    def _update_latch(self) -> None:
        ok, detail = self.task.evaluate(self.data, self._grip_cmd)
        self.task_detail = detail
        t = self.t
        if ok:
            if self._ok_since is None:
                self._ok_since = t
                if self.t_first_ok is None:
                    self.t_first_ok = t
            if not self.success and (t - self._ok_since) >= self.dwell_s - 1e-9:
                self.success = True
                self.t_success = t
        else:
            self._ok_since = None

    def step(self) -> Dict[str, Any]:
        """Advance exactly one policy period (1/30 s) in lockstep. Returns
        {"t", "frame", "success", "t_success_s", "failure", "detail", "done", "outcome"}."""
        self.last_tick_max_delta = 0.0
        t_end = self._t0 + (self.frame + 1) * self.frame_dt
        while float(self.data.time) + 0.5 * self.dt < t_end:
            if self._phys % self.steps_per_tick == 0:
                self._bridge_tick()
                self._tick += 1
            mujoco.mj_step(self.model, self.data)
            self._phys += 1
            if self._phys % self.steps_per_tick == 0:
                self._update_latch()
            if self._phys % self.steps_per_log == 0:
                self._log_row()
        self.frame += 1
        if self.failure is None:
            self.failure = self._check_failure()
        done = self.success or self.failure is not None
        outcome = Outcome.SUCCESS if self.success else (Outcome.FAILURE if self.failure else None)
        return {"t": self.t, "frame": self.frame, "success": self.success, "t_success_s": self.t_success,
                "t_first_ok_s": self.t_first_ok, "failure": self.failure, "detail": self.task_detail,
                "done": done, "outcome": outcome, "step_eff": self.step_eff}

    # ------------------------------------------------------------------ #
    # Episode logs                                                         #
    # ------------------------------------------------------------------ #
    def episode_state(self) -> Dict[str, np.ndarray]:
        rows = np.asarray(self._log_rows, dtype=np.float64) if self._log_rows else np.zeros((0, 3 + self.model.nq + self.model.nv + self.model.nu))
        nq, nv = self.model.nq, self.model.nv
        return {"t_rel_s": rows[:, 0], "sim_t": rows[:, 1], "tick": rows[:, 2],
                "qpos": rows[:, 3:3 + nq], "qvel": rows[:, 3 + nq:3 + nq + nv], "ctrl": rows[:, 3 + nq + nv:]}

    def save_episode(self, path: str, attrs: Optional[Dict[str, Any]] = None) -> str:
        """Write `sim_mj_state` (recorder columns/attrs) + `sim_scene` (xml, layout, config)
        into one h5 so `load_episode_state` / replay_take-style tooling can replay it."""
        import h5py
        st = self.episode_state()
        nq, nv, nu = self.model.nq, self.model.nv, self.model.nu
        cols = (["t_rel_s", "sim_t", "tick"] + [f"qpos{i}" for i in range(nq)]
                + [f"qvel{i}" for i in range(nv)] + [f"ctrl{i}" for i in range(nu)])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group("sim_mj_state")
            g.attrs["columns"] = json.dumps(cols)
            g.attrs["nq"], g.attrs["nv"], g.attrs["nu"] = nq, nv, nu
            g.attrs["ctrl_note"] = "ctrl = q_cmd[6] + grip_ctrl (2F-85 actuator, 0..255)"
            g.attrs["rate_hz"] = STATE_LOG_HZ
            for k in ("t_rel_s", "sim_t", "tick"):
                g.create_dataset(k, data=st[k])
            for i in range(nq):
                g.create_dataset(f"qpos{i}", data=st["qpos"][:, i])
            for i in range(nv):
                g.create_dataset(f"qvel{i}", data=st["qvel"][:, i])
            for i in range(nu):
                g.create_dataset(f"ctrl{i}", data=st["ctrl"][:, i])
            s = f.create_group("sim_scene")
            s.create_dataset("xml", data=self.built.xml)
            s.attrs["xml_sha256"] = self.xml_sha256
            s.attrs["layout"] = json.dumps(self.layout)
            s.attrs["config"] = json.dumps(self.cfg.raw)
            s.attrs["config_path"] = os.path.relpath(self.config_path, REPO_ROOT)
            s.attrs["mujoco_version"] = mujoco.__version__
            e = f.create_group("eval")
            meta = {"seed": self.seed, "layout": self.layout, "success": self.success, "t_success_s": self.t_success,
                    "t_first_ok_s": self.t_first_ok, "failure": self.failure, "frames": self.frame,
                    "n_clamped_limit": self.n_clamped_limit, "n_clamped_dev": self.n_clamped_dev,
                    "clamp_hits": self.clamp_hits, "eval_config": self.ev, "wrench_tare": self.wrench_tare.tolist()}
            meta.update(attrs or {})
            e.attrs["meta"] = json.dumps(meta, default=str)
        return path

    # ------------------------------------------------------------------ #
    # Video                                                                #
    # ------------------------------------------------------------------ #
    def _open_video(self, video_dir: Optional[str], tag: str) -> None:
        import cv2
        self._video_dir = video_dir or os.path.join(REPO_ROOT, "sim_collect", "eval", "runs", "untitled")
        os.makedirs(self._video_dir, exist_ok=True)
        self._video_tag = tag or (f"ep_{self.seed}" if self.seed is not None else "ep")
        self.video_paths: Dict[str, str] = {}
        for cam in ("cam1", "cam2"):
            p = os.path.join(self._video_dir, f"{self._video_tag}_{cam}.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), self.policy_hz, COLOR_SIZE)
            if not w.isOpened():
                raise RuntimeError(f"cannot open video writer {p}")
            self._video[cam] = w
            self.video_paths[cam] = p

    def _close_video(self) -> None:
        for w in self._video.values():
            try:
                w.release()
            except Exception:  # noqa: BLE001
                pass
        self._video = {}

    def close(self) -> None:
        self._close_video()
        if self._rig is not None:
            self._rig.close()
            self._rig = None


def load_episode_state(path: str):
    """Read an `ep_*.h5` (or any h5 with the recorder's `sim_mj_state` group) into a
    `replay_take.TakeState` (t_rel_s, sim_t, tick, qpos, qvel, ctrl)."""
    import h5py
    from sim_collect.tools.replay_take import TakeState
    with h5py.File(path, "r") as f:
        g = f["sim_mj_state"]
        nq, nv, nu = int(g.attrs["nq"]), int(g.attrs["nv"]), int(g.attrs["nu"])
        col = {c: g[c][:] for c in json.loads(g.attrs["columns"])}
    n = len(col["t_rel_s"])
    qpos = np.stack([col[f"qpos{i}"] for i in range(nq)], 1) if n else np.zeros((0, nq))
    qvel = np.stack([col[f"qvel{i}"] for i in range(nv)], 1) if n else np.zeros((0, nv))
    ctrl = np.stack([col[f"ctrl{i}"] for i in range(nu)], 1) if n else np.zeros((0, nu))
    return TakeState(col["t_rel_s"], col["sim_t"], col["tick"], qpos, qvel, ctrl)
