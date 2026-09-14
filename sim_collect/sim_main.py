"""Process [A]: physics + GELLO leader + teleop controller (DESIGN.md §2.1).

    MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.sim_main \
        --config sim_collect/configs/carrot_in_pot_sim.yaml [--fake-leader] [--no-viewer]

Main thread = physics loop (2 ms timestep, real-time paced, `mujoco.viewer.launch_passive`
unless --no-viewer, in which case `mujoco.viewer` is never imported). A leader
thread reads the GELLO at 30 Hz. Every 2nd physics step (250 Hz) the control tick
runs `TeleopController.tick`, maps the trigger through `GripperMapper`, writes
`d.ctrl`, evaluates the task, publishes the `state` message and services one
pending REP command. Startup reads the leader once and TELEPORTS the arm there
(`init_from_leader`), starting DISENGAGED (state HOLD = holding, ready to engage).

REP commands (`{"cmd": ...}` -> `{"ok", "msg", ...}`): get_status, engage,
disengage, reclutch, set_pos_scale{value}, set_control_mode{mode}, reset_scene{seed?},
home, gripper_pause, gripper_resume, get_scene_meta, get_layout, shutdown.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from sim_collect import ipc
from sim_collect.controller import TeleopController, load_bridge_params
from sim_collect.gripper import GripperMapper, grip_pos_from_driver
from sim_collect.leader import LeaderSample, make_leader, resolve_leader_config
from sim_collect.scene import (REPO_ROOT, SceneConfig, build_scene, layout_conflicts, place_objects_safely,
                               robot_body_ids, robot_contacts, sample_layout, tcp_over_objects)
from sim_collect.scene import apply_layout as scene_apply_layout
from sim_collect.task import TaskConfig, TaskEvaluator
from ur_gello_bringup import ur_kin

SIM_COLLECT_VERSION = 1


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=2.0).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class SimMain:
    def __init__(self, cfg: SceneConfig, control_mode: Optional[str] = None, fake_leader: bool = False,
                 no_viewer: bool = False, seed: Optional[int] = None, fake_wiggle: float = 0.0,
                 realtime: bool = True, publish: bool = True) -> None:
        self.cfg = cfg
        self.no_viewer = no_viewer
        self.realtime = realtime
        self.fake_leader = fake_leader
        ctl = cfg.control
        self.control_mode = str(control_mode or ctl.get("control_mode", "eef"))
        self.control_hz = float(ctl.get("control_hz", 250))
        self.leader_hz = float(ctl.get("leader_hz", 30))
        self.seed = int(cfg.layout.get("seed", 0)) if seed is None else int(seed)

        # scene
        self.layout = sample_layout(cfg, self.seed)
        self.built = build_scene(cfg, self.layout)
        self.meta = self.built.meta
        self.model = self.built.load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = float(m.opt.timestep)
        self.steps_per_tick = max(1, int(round(1.0 / (self.control_hz * self.dt))))
        self.control_hz = 1.0 / (self.steps_per_tick * self.dt)
        names = self.meta["names"]
        self.grip_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, names["gripper_driver_joint"])
        self.grip_qadr = int(m.jnt_qposadr[self.grip_jnt])
        self.grip_range = names["gripper_driver_range"]
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, names["gripper_actuator"])
        self.site_att = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, names["attachment_site"])
        fs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, names["force_sensor"])
        ts = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, names["torque_sensor"])
        self.force_adr = int(m.sensor_adr[fs])
        self.torque_adr = int(m.sensor_adr[ts])
        self.obj_frame_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["frame_body"])
                              for n, o in self.meta["objects"].items()}
        # gripper qpos block = every joint between the arm and the first object
        self.gripper_qpos = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                             if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and j >= 6]
        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)

        # leader (constructed first: the controller re-anchors its unwrap chain)
        self.leader = make_leader(cfg.leader, fake_leader, cfg.home_joints, self.leader_hz, fake_wiggle)
        self.last_sample: Optional[LeaderSample] = None
        try:
            self.leader_config = resolve_leader_config(cfg.leader, cfg.home_joints) if not fake_leader else {
                "source": "fake", "start_joints": list(map(float, cfg.home_joints)) + [0.0]}
        except Exception as e:  # noqa: BLE001 report, never block a fake run
            self.leader_config = {"source": f"unresolved: {e}"}

        # controller + gripper + task
        self.params = load_bridge_params(ctl.get("bridge_params", []), str(ctl.get("bridge_node", "gello_ur_bridge")))
        self.controller = TeleopController(self.params, self.control_mode, self.control_hz,
                                           float(ctl.get("joint_engage_max_gap_rad", 1.5)),
                                           reanchor=self.leader.reanchor)
        g = cfg.gripper
        self.gripper = GripperMapper(str(g.get("mode", "continuous")), float(g.get("deadband", 0.02)),
                                     float(g.get("open_at", 0.3)), float(g.get("close_at", 0.7)),
                                     resume_ramp_s=float(g.get("resume_ramp_s", 2.0)),
                                     resume_slew_per_s=float(g.get("resume_slew_per_s", 0.6)),
                                     staleness_timeout_s=float(self.params.get("staleness_timeout_s", 0.5)))
        self.task_cfg = TaskConfig.from_scene(cfg.task, self.meta)
        self.task = TaskEvaluator(m, self.task_cfg)
        self.task_result = (False, "not evaluated")
        self.T_tool_R = self.controller.T_tool_R
        # robot = every body between world and the first object attachment frame
        self.robot_bodies = robot_body_ids(m, self.meta)
        self.startup_note = ""
        self.last_reset_note = ""

        # ipc
        self.pub = ipc.Publisher("state_pub") if publish else None
        self.server = ipc.Server("sim_rep") if publish else None
        self.tick = 0
        self._stop = False
        self._git = git_commit()
        self.wall_start = 0.0
        self.sim_start = 0.0
        self.rt_ratio = 1.0
        self.t_started = time.time()
        # Wrench tare: the real UR zeroes its F/T reading at startup, so the recorded
        # `wrench` is small at rest (take_18: |f| ~ 2 N). We mirror that by taring the
        # flange sensor (which sees the full 2F-85 weight, ~10 N) after every
        # teleport+settle (startup / reset_scene / home). `wrench_raw` keeps the raw value.
        self.wrench_tare = np.zeros(6)

    # ------------------------------------------------------------------ #
    # Teleports / settle                                                   #
    # ------------------------------------------------------------------ #
    def _teleport_arm(self, q: np.ndarray, open_gripper: bool = True) -> None:
        d = self.data
        d.qpos[:6] = q
        d.qvel[:6] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[:6] = q
        if open_gripper:
            for a in self.gripper_qpos:
                d.qpos[a] = 0.0
            self.gripper.reset(0.0)
            d.ctrl[self.grip_act] = 0.0
        mujoco.mj_forward(self.model, d)
        self.controller.reseed()

    def _apply_layout(self, layout: Dict[str, Any]) -> None:
        scene_apply_layout(self.model, self.data, self.meta, layout)

    def _settle(self, seconds: float) -> None:
        """Fast-forward physics with the current ctrl held (not real time)."""
        n = int(round(seconds / self.dt))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.wall_start = time.monotonic()
        self.sim_start = float(self.data.time)
        self.wrench_tare = self._wrench_raw()

    def _wrench_raw(self) -> np.ndarray:
        d = self.data
        return np.concatenate([d.sensordata[self.force_adr:self.force_adr + 3],
                               d.sensordata[self.torque_adr:self.torque_adr + 3]]).astype(float)

    # The placement helpers live in scene.py (shared with sim_collect/eval); these
    # wrappers keep SimMain's private API (tests call `_robot_contacts`).
    def _robot_contacts(self) -> List[str]:
        """Robot<->(floor|object) contacts in the CURRENT kinematic state (call after mj_forward)."""
        return robot_contacts(self.model, self.data, self.robot_bodies)

    def _tcp_over_objects(self, layout: Dict[str, Any]) -> List[str]:
        """Objects whose footprint (+3 cm) is under a LOW TCP (z < 0.25 m)."""
        return tcp_over_objects(self.data, self.meta, layout, self.T_tool_R)

    def _layout_conflicts(self, layout: Dict[str, Any]) -> List[str]:
        return layout_conflicts(self.model, self.data, self.meta, layout, self.robot_bodies, self.T_tool_R)

    def _place_objects_safely(self, seed: int) -> Dict[str, Any]:
        """Sample layouts for `seed` until none touches the arm; else send the arm home."""
        layout, note, fell_back = place_objects_safely(
            self.cfg, self.model, self.data, self.meta, seed, robot_bodies=self.robot_bodies,
            T_tool_R=self.T_tool_R, teleport_home=lambda: self._teleport_arm(self.cfg.home_joints, open_gripper=True),
            sampler=sample_layout)  # module attribute, looked up at call time (tests monkeypatch it)
        self.last_reset_note = note
        if fell_back:
            print("[sim_main] reset_scene: " + self.last_reset_note)
        return layout

    def _leader_pose_nearest(self, ref: np.ndarray) -> Optional[np.ndarray]:
        s = self.leader.latest()
        if s is None:
            return None
        self.last_sample = s
        return ur_kin.wrapped_nearest(s.q_raw, ref)

    def startup(self) -> None:
        """Read the leader once and teleport the arm there; start DISENGAGED."""
        self.leader.start()
        home = self.cfg.home_joints
        q0 = home.copy()
        if self.cfg.init_from_leader:
            deadline = time.monotonic() + 10.0
            while self.leader.latest() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            q = self._leader_pose_nearest(home)
            if q is None:
                raise RuntimeError("no leader sample within 10 s at startup")
            q0 = q
            print("[sim_main] init_from_leader: teleporting arm to " + ", ".join(f"{v:.4f}" for v in q0))
        self._teleport_arm(q0)
        conflicts = self._layout_conflicts(self.layout)
        if conflicts:
            self.startup_note = (f"leader pose collides ({', '.join(conflicts[:4])}); arm teleported to "
                                 f"home_joints instead — move the GELLO to a free pose and reset_scene")
            print("[sim_main] startup: " + self.startup_note)
            self._teleport_arm(home)
            conflicts = self._layout_conflicts(self.layout)
            if conflicts:
                self.layout = self._place_objects_safely(self.seed)
        self._settle(float(self.cfg.layout.get("settle_s", 0.5)))
        self.controller.reseed()

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.controller.disengage()
        home = self.cfg.home_joints
        q = self._leader_pose_nearest(home)
        if q is None:
            q = np.asarray(self.data.qpos[:6], dtype=float).copy()
        self._teleport_arm(q, open_gripper=True)
        self.seed = int(seed) if seed is not None else self.seed + 1
        self.last_reset_note = ""
        self.layout = self._place_objects_safely(self.seed)
        self.data.qvel[:] = 0.0
        self._settle(float(self.cfg.layout.get("settle_s", 0.5)))
        self.controller.reseed()
        return self.layout

    def home(self) -> None:
        self.controller.disengage()
        self._teleport_arm(self.cfg.home_joints, open_gripper=True)
        self.data.qvel[:] = 0.0
        self._settle(0.2)
        self.controller.reseed()

    # ------------------------------------------------------------------ #
    # Control tick                                                         #
    # ------------------------------------------------------------------ #
    def control_tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        d = self.data
        s = self.leader.latest()
        if s is not None:
            self.last_sample = s
            self.controller.ingest(s)
            self.gripper.update(s.trigger, now)
        q_actual = np.asarray(d.qpos[:6], dtype=float)
        q_cmd = self.controller.tick(q_actual, now)
        if q_cmd is not None:
            d.ctrl[:6] = q_cmd
        d.ctrl[self.grip_act] = self.gripper.ctrl
        self.task_result = self.task.evaluate(d, self.gripper.grip_cmd)
        self.tick += 1

    # ------------------------------------------------------------------ #
    # State message                                                        #
    # ------------------------------------------------------------------ #
    def state_message(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.monotonic() if now is None else now
        d, m = self.data, self.model
        q = np.asarray(d.qpos[:6], dtype=float)
        T = ur_kin.fk(q) @ self.T_tool_R
        cmd_p, cmd_q = self.controller.cmd_tcp()
        s = self.last_sample
        c = self.controller
        q_cmd = c.q_cmd if c.q_cmd is not None else q
        objects = {}
        for name, bid in self.obj_frame_ids.items():
            objects[name] = {"pos": d.xpos[bid].tolist(), "quat_wxyz": d.xquat[bid].tolist()}
        info = c.info(now)
        lead_raw = s.q_raw.tolist() if s is not None else [0.0] * 6
        return {
            "t": time.time(), "sim_t": float(d.time), "tick": self.tick,
            "control_mode": c.mode, "eef_state": c.eef_state, "eef_info": info,
            "engaged": c.engaged, "pos_scale": c.pos_scale,
            "q_lead_raw": lead_raw,
            "q_lead_unwrapped": s.q_unwrapped.tolist() if s is not None else lead_raw,
            "q_lead_f": list(map(float, c.q_lead_f)) if c.q_lead_f is not None else (
                list(map(float, c._raw_target)) if c._raw_target is not None else lead_raw),
            "qd_lead": s.qd.tolist() if s is not None else [0.0] * 6,
            "trigger": float(s.trigger) if s is not None else 0.0,
            "leader_t": float(s.t) if s is not None else 0.0,
            "leader_seq": int(s.seq) if s is not None else -1,
            "leader_age": (now - s.t) if s is not None else None,
            "q_cmd": q_cmd.tolist(), "q": q.tolist(), "qd": d.qvel[:6].tolist(),
            "eff": d.actuator_force[:6].tolist(),
            "tcp_pos": T[:3, 3].tolist(), "tcp_quat_xyzw": ur_kin.mat_to_quat_xyzw(T[:3, :3]).tolist(),
            "cmd_tcp_pos": cmd_p.tolist(), "cmd_tcp_quat_xyzw": cmd_q.tolist(),
            "wrench": (self._wrench_raw() - self.wrench_tare).tolist(),   # tared like the real UR (see __init__)
            "wrench_raw": self._wrench_raw().tolist(),
            "grip_cmd": float(self.gripper.grip_cmd), "grip_ctrl": float(d.ctrl[self.grip_act]),
            "grip_pos": grip_pos_from_driver(float(d.qpos[self.grip_qadr]), *self.grip_range),
            "gripper_paused": self.gripper.paused, "gripper_mode": self.gripper.mode,
            "gripper_ramping": self.gripper.ramping,
            "qpos_full": d.qpos.tolist(), "qvel_full": d.qvel.tolist(),
            "objects": objects,
            "task": {"success": bool(self.task_result[0]), "detail": self.task_result[1], "name": self.task_cfg.name},
            "layout_seed": self.seed, "rt_ratio": self.rt_ratio, "viewer": not self.no_viewer,
            "leader_fake": bool(getattr(self.leader, "is_fake", False)),
        }

    def scene_meta(self) -> Dict[str, Any]:
        objs = {n: dict(o) for n, o in self.meta["objects"].items()}
        return {
            "sim_collect_version": SIM_COLLECT_VERSION, "git_commit": self._git,
            "mujoco_version": mujoco.__version__, "model_name": self.built.xml.split('model="', 1)[1].split('"', 1)[0],
            "config_path": self.cfg.path, "config": self.cfg.raw, "control_mode": self.controller.mode,
            "gripper_mode": self.gripper.mode, "pos_scale": self.controller.pos_scale,
            "robot": self.cfg.robot.get("name", "ur7e"), "home_joints": self.meta["home_joints"],
            "objects": objs, "object_names": list(objs), "chosen_food": self.task_cfg.food,
            "container": self.task_cfg.container, "task": self.task_cfg.name,
            "cameras": self.meta["cameras"], "floor": self.meta["floor"], "names": self.meta["names"],
            "layout": self.layout, "layout_seed": self.seed, "timestep": self.dt,
            "control_hz": self.control_hz, "leader_hz": self.leader_hz,
            "nq": int(self.model.nq), "nv": int(self.model.nv), "nu": int(self.model.nu),
            "bridge_params": self.params, "leader_fake": bool(getattr(self.leader, "is_fake", False)),
            "leader_calibration_source": self.leader_config.get("source"),
            "leader_config": {k: v for k, v in self.leader_config.items() if k != "source"},
            "startup_note": self.startup_note, "render": self.cfg.render,
            "xml_sha256": __import__("hashlib").sha256(self.built.xml.encode()).hexdigest(),
        }

    # ------------------------------------------------------------------ #
    # REP handler                                                          #
    # ------------------------------------------------------------------ #
    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = str(req.get("cmd", ""))
        c = self.controller
        now = time.monotonic()
        if cmd == "get_status":
            return {"ok": True, "msg": "alive", "engaged": c.engaged, "eef_state": c.eef_state,
                    "control_mode": c.mode, "pos_scale": c.pos_scale, "tick": self.tick, "sim_t": float(self.data.time),
                    "leader_ok": self.last_sample is not None and (now - self.last_sample.t) < c.staleness_timeout_s,
                    "leader_fake": bool(getattr(self.leader, "is_fake", False)), "viewer": not self.no_viewer,
                    "gripper_paused": self.gripper.paused, "task_success": bool(self.task_result[0]),
                    "layout_seed": self.seed, "rt_ratio": self.rt_ratio, "eef_info": c.info(now),
                    "uptime_s": time.time() - self.t_started, "startup_note": self.startup_note,
                    "gripper_ramping": self.gripper.ramping}
        if cmd == "engage":
            ok, key, msg = c.engage(self.data.qpos[:6], now)
            return {"ok": ok, "key": key, "msg": msg, "eef_state": c.eef_state}
        if cmd == "disengage":
            ok, key, msg = c.disengage()
            return {"ok": ok, "key": key, "msg": msg, "eef_state": c.eef_state}
        if cmd == "reclutch":
            ok, key, msg = c.reclutch(now)
            return {"ok": ok, "key": key, "msg": msg, "eef_state": c.eef_state}
        if cmd == "set_pos_scale":
            ok, key, msg = c.set_pos_scale(req.get("value"))
            return {"ok": ok, "key": key, "msg": msg, "pos_scale": c.pos_scale}
        if cmd == "set_control_mode":
            ok, key, msg = c.set_control_mode(str(req.get("mode", "")))
            return {"ok": ok, "key": key, "msg": msg, "control_mode": c.mode}
        if cmd == "reset_scene":
            layout = self.reset_scene(req.get("seed"))
            msg = f"scene reset (seed {self.seed})" + (f"; {self.last_reset_note}" if self.last_reset_note else "")
            return {"ok": True, "msg": msg, "layout": layout, "seed": self.seed,
                    "layout_attempt": int(layout.get("_attempt", 0)),
                    "arm_moved_home": "teleported to home" in self.last_reset_note}
        if cmd == "home":
            self.home()
            return {"ok": True, "msg": "arm teleported to home", "layout": self.layout, "seed": self.seed}
        if cmd == "gripper_pause":
            self.gripper.pause()
            return {"ok": True, "msg": f"gripper paused at grip_cmd={self.gripper.grip_cmd:.2f}"}
        if cmd == "gripper_resume":
            self.gripper.resume(now)
            return {"ok": True, "msg": f"gripper resumed (slew ramp {self.gripper.resume_ramp_s:.1f}s)"}
        if cmd == "get_scene_meta":
            return {"ok": True, "msg": "scene meta", "meta": self.scene_meta()}
        if cmd == "get_layout":
            return {"ok": True, "msg": "layout", "layout": self.layout, "seed": self.seed}
        if cmd == "shutdown":
            self._stop = True
            return {"ok": True, "msg": "stopping"}
        return {"ok": False, "msg": f"unknown cmd {cmd!r}"}

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    def _pace(self) -> None:
        if not self.realtime:
            return
        ahead = (self.wall_start + (float(self.data.time) - self.sim_start)) - time.monotonic()
        if ahead > 0.0005:
            time.sleep(ahead)
        elif ahead < -0.25:  # hopelessly behind (e.g. a blocking reset): drop, do not catch up
            self.wall_start = time.monotonic()
            self.sim_start = float(self.data.time)

    def run(self, duration_s: Optional[float] = None) -> None:
        self.startup()
        viewer = None
        if not self.no_viewer:
            import importlib  # mujoco.viewer is imported ONLY here, never on the headless path
            viewer_mod = importlib.import_module("mujoco.viewer")
            viewer = viewer_mod.launch_passive(self.model, self.data)
        print(f"[sim_main] running: mode={self.controller.mode} control {self.control_hz:.0f} Hz "
              f"({self.steps_per_tick} steps/tick), leader {'FAKE' if self.fake_leader else 'GELLO'} "
              f"{self.leader_hz:.0f} Hz, viewer={'off' if self.no_viewer else 'on'}, seed={self.seed}")
        self.wall_start = time.monotonic()
        self.sim_start = float(self.data.time)
        t_end = None if duration_s is None else time.monotonic() + float(duration_s)
        step = 0
        last_report = time.monotonic()
        try:
            while not self._stop:
                if viewer is not None and not viewer.is_running():
                    break
                if t_end is not None and time.monotonic() >= t_end:
                    break
                if step % self.steps_per_tick == 0:
                    now = time.monotonic()
                    self.control_tick(now)
                    if self.pub is not None:
                        self.pub.send("state", self.state_message(now))
                    if self.server is not None:
                        self.server.poll(self.handle, timeout_ms=0)
                mujoco.mj_step(self.model, self.data)
                step += 1
                if viewer is not None and step % 16 == 0:   # ~31 Hz: the software-GL viewer competes with the capture workers
                    viewer.sync()
                if step % 500 == 0:
                    now = time.monotonic()
                    self.rt_ratio = 1.0 / max(1e-6, now - last_report)
                    last_report = now
                self._pace()
        except KeyboardInterrupt:
            pass
        finally:
            if viewer is not None:
                viewer.close()
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.leader.close()
        finally:
            if self.server is not None:
                self.server.close()
            if self.pub is not None:
                self.pub.close()
        print("[sim_main] stopped; leader port closed")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sim_collect/configs/carrot_in_pot_sim.yaml")
    ap.add_argument("--fake-leader", action="store_true", help="scripted leader (no GELLO)")
    ap.add_argument("--fake-wiggle", type=float, default=0.0, help="fake leader sinusoid amplitude (rad) on joint 0")
    ap.add_argument("--no-viewer", action="store_true", help="headless (mujoco.viewer is not imported)")
    ap.add_argument("--control-mode", choices=["eef", "joint"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="exit after N seconds")
    ap.add_argument("--no-realtime", action="store_true", help="run physics as fast as possible")
    args = ap.parse_args(argv)
    if args.no_viewer and "mujoco.viewer" in sys.modules:
        raise RuntimeError("mujoco.viewer was imported on the --no-viewer path")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    cfg = SceneConfig.load(args.config)
    app = SimMain(cfg, control_mode=args.control_mode, fake_leader=args.fake_leader, no_viewer=args.no_viewer,
                  seed=args.seed, fake_wiggle=args.fake_wiggle, realtime=not args.no_realtime)

    def _sig(signum, _frame):
        print(f"[sim_main] signal {signum}: stopping")
        app._stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    app.run(args.duration)
    return 0


def _hard_exit_after(seconds: float) -> None:
    """Watchdog: if interpreter teardown hangs (GLFW viewer thread / ZMQ context term
    were observed to keep a real-GELLO+viewer process alive after cleanup), exit hard."""
    import threading

    def _bang():
        time.sleep(seconds)
        sys.stdout.write("[sim_main] teardown watchdog: forcing exit\n")
        sys.stdout.flush()
        os._exit(0)

    threading.Thread(target=_bang, daemon=True).start()


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        # cleanup is done inside run(); do not let a stuck non-daemon thread or a
        # blocking zmq/GLFW teardown keep the process (and the launcher) waiting
        _hard_exit_after(3.0)
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(rc)
