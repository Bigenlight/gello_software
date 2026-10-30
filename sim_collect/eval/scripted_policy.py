"""Scripted oracle pick-and-place for the carrot-in-pot eval (DESIGN.md §2.2 / §3, F2).

Positive control for the harness: from GROUND-TRUTH object poses it emits
30 Hz joint targets + grip through the normal `EvalWorld.apply()` path, so the
real-deploy clamps (envelope clip, max_dev, One-Euro + 250 Hz slew) shape the
motion exactly as they would a learned policy's. No teleports, no qpos writes.

Phases (each advances on a pose tolerance on the ACTUAL joints, or a timeout):
    PREGRASP  above the grasp point (top-down, wrist aligned with the carrot heading)
    DESCEND   straight down to the grasp height
    CLOSE     grip 1.0, wait for grip_pos to settle
    LIFT      straight up; then re-read the carrot pose: slip -> retry once
    TRANSFER  straight line to the release pose above the pot (wrist re-oriented)
    LOWER     straight down to the release height
    OPEN      grip 0.0
    RETREAT   straight up
    HOLD      stay

Why there is a candidate SEARCH instead of one fixed grasp/release pose: the
eval joint envelope (`eval.joint_limits_lo/hi`, from the REAL carrot takes) is
tight around the pot -- a top-down TCP over a pot centre at y < -0.22 needs
J1 < -3.80 -- so the oracle enumerates grasp points along the carrot, wrist
yaw offsets, small tilts and (for the release) TCP positions on the reachable
side of the pot, keeps the candidates whose whole straight-line path is IK
feasible inside the envelope, and takes the cheapest. Holding the carrot
off-centre and yawing the wrist lets the carrot's centre hang over the pot
while the TCP stays inside the envelope. With looser limits the same search
simply picks the centred, exactly-aligned grasp.

Ground truth needed from the world: `object_pose(name) -> (pos, quat_wxyz)`
(free-joint pose = the object's bottom-centre origin) for 'carrot' and 'pot';
the pot rim height is taken from `world.meta['objects'][pot]['size']['rim_z']`
when present (else the `pot_rim_z` param). `tcp_pose()` is NOT needed: the
TCP is derived from the observed joints with the same kinematics the IK uses.
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ur_gello_bringup import ur_kin

from sim_collect.eval import ik
from sim_collect.scene import quat_wxyz_to_mat

_Z_DOWN = np.array([0.0, 0.0, -1.0])

# Carrot collision capsules (assets/objects/carrot/carrot.xml): (x_from, x_to, radius),
# body-local, lying along +x with the bottom at z=0. Used for the grasp height and to
# predict where the tips are when the held carrot is over the pot.
CARROT_SEGMENTS = ((-0.0864, -0.0514, 0.0059), (-0.0514, -0.0164, 0.0098),
                   (-0.0164, 0.0186, 0.0137), (0.0186, 0.0536, 0.0176))
CARROT_X_MIN = CARROT_SEGMENTS[0][0] - CARROT_SEGMENTS[0][2]
CARROT_X_MAX = CARROT_SEGMENTS[-1][1] + CARROT_SEGMENTS[-1][2]


def carrot_radius_at(x_local: float) -> float:
    for x0, x1, r in CARROT_SEGMENTS:
        if x0 - 1e-9 <= x_local <= x1 + 1e-9:
            return r
    return CARROT_SEGMENTS[0][2] if x_local < 0 else CARROT_SEGMENTS[-1][2]


DEFAULT_PARAMS: Dict[str, Any] = {
    # --- grasp geometry ---
    "approach_height": 0.15,      # pre-grasp TCP height above the grasp point [m]
    "grasp_tcp_z": 0.006,         # TCP height above the floor at the grasp (closed pads span TCP+0.0005..+0.038)
    "grasp_x_candidates": (0.0, 0.02, -0.02, 0.04, 0.01, -0.01, 0.03),  # carrot-local x of the grasp point
    "yaw_err_candidates_deg": (0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0),  # wrist yaw off the carrot heading
    "tilt_candidates_deg": (0.0, 10.0, -10.0, 20.0, -20.0),               # tool tilt about the carrot axis
    "lift_height": 0.20,
    # --- place geometry ---
    "pot_rim_z": 0.11,            # fallback when the world has no scene meta
    "release_dz": 0.04,           # TCP height above the rim at release
    "transfer_height": 0.22,      # TCP height (world z) during the transfer
    "place_xy_grid": 0.02,        # TCP xy grid step around the pot centre [m]
    "place_xy_radius": 0.08,
    "place_yaw_candidates_deg": (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0),
    "place_pitch_candidates_deg": (0.0, 15.0, -15.0, 30.0, -30.0),     # tool pitch about the closing axis (pitches the carrot)
    "retreat_height": 0.10,
    # --- motion / timing (30 Hz steps) ---
    "step_m": 0.007,              # Cartesian waypoint spacing -> 0.21 m/s
    "step_rad": 0.02,             # rotation waypoint spacing -> 0.6 rad/s (the bridge slews 0.625 rad/s)
    "pos_tol": 0.010, "rot_tol": 0.08,
    "phase_timeout_steps": 120,
    "close_settle_steps": 4, "close_settle_eps": 0.003, "close_timeout_steps": 45, "close_min_grip_pos": 0.15,
    "open_steps": 12,
    "slip_min_rise": 0.05,        # carrot origin must have risen this much after LIFT
    "max_retries": 1,
    "retry_deeper": 0.003,        # lower the grasp TCP by this on a retry
}


@dataclasses.dataclass
class _Phase:
    name: str
    path: List[np.ndarray]        # joint waypoints to emit, one per step
    grip: float
    goal_T: Optional[np.ndarray]  # TCP pose to reach (None = no pose check)
    timeout: int
    i: int = 0
    steps: int = 0


def _top_down_T(p: np.ndarray, y_axis: np.ndarray, tilt_about_y: float = 0.0,
                pitch_about_x: float = 0.0) -> np.ndarray:
    """TCP pose with tool +Z down, tool +Y = `y_axis` (horizontal unit vector), then
    rotated by `tilt_about_y` about tool Y and `pitch_about_x` about tool X."""
    y = np.array([y_axis[0], y_axis[1], 0.0]); y /= np.linalg.norm(y)
    z = _Z_DOWN.copy()
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    if tilt_about_y:
        R = R @ ur_kin.so3_exp(np.array([0.0, tilt_about_y, 0.0]))
    if pitch_about_x:
        R = R @ ur_kin.so3_exp(np.array([pitch_about_x, 0.0, 0.0]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def _pose(pos: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def _limits_from(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(lo, hi) from a dict (`world.info()`) or an object with joint_limits_lo/hi attributes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        lo, hi = obj.get("joint_limits_lo"), obj.get("joint_limits_hi")
    else:
        lo, hi = getattr(obj, "joint_limits_lo", None), getattr(obj, "joint_limits_hi", None)
    if lo is None or hi is None:
        return None
    return np.asarray(lo, dtype=float).reshape(6), np.asarray(hi, dtype=float).reshape(6)


class ScriptedPolicy:
    """Implements the harness `Policy` protocol (policies.py): `needs_images`, `meta`,
    `reset(world_info) -> None`, `act(obs) -> list[7]` (joint targets + grip 0..1).
    Never returns None (the oracle has no fault path); on an unplannable scene it
    holds the last target with the gripper open and records why in `log` / `meta`.

    `limits` (lo, hi) defaults to the world's `joint_limits_lo/hi` (or the `world_info`
    handed to `reset`), else the carrot deploy envelope — the SAME envelope the world
    clips with, so every emitted target is inside it and the clamps stay idle."""

    needs_images = False

    def __init__(self, world: Any, params: Optional[Dict[str, Any]] = None,
                 limits: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                 food: str = "carrot", container: str = "pot") -> None:
        self.world = world
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.food, self.container = food, container
        self._explicit_limits = limits is not None
        lim = (np.asarray(limits[0], float).reshape(6), np.asarray(limits[1], float).reshape(6)) if limits is not None \
            else _limits_from(world)
        self.limits = ik.default_limits() if lim is None else lim
        self.meta: Dict[str, Any] = {"policy": "scripted", "food": food, "container": container,
                                     "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.p.items()}}
        self.log: List[str] = []
        self._phases: List[_Phase] = []
        self._retries = 0
        self._grasp_deeper = 0.0
        self._q_last: Optional[np.ndarray] = None
        self._branch = -1
        self._T_food_in_tcp: Optional[np.ndarray] = None
        self._grasp_T: Optional[np.ndarray] = None
        self.phase_name = "INIT"
        self.done = False
        self.step_count = 0

    # ------------------------------------------------------------------ #
    # ground truth                                                         #
    # ------------------------------------------------------------------ #
    def _object_T(self, name: str) -> np.ndarray:
        pos, quat = self.world.object_pose(name)
        return _pose(pos, quat)

    def _rim_z(self) -> float:
        meta = getattr(self.world, "meta", None) or getattr(self.world, "scene_meta", None)
        try:
            return float(meta["objects"][self.container]["size"]["rim_z"])
        except (TypeError, KeyError):
            return float(self.p["pot_rim_z"])

    # ------------------------------------------------------------------ #
    # planning helpers                                                     #
    # ------------------------------------------------------------------ #
    def _ik(self, T: np.ndarray, q_seed: np.ndarray) -> Optional[np.ndarray]:
        return ik.tcp_to_joints(T, q_seed, self.limits, branch=self._branch, allow_branch_fallback=False)

    def _path(self, T_from: np.ndarray, T_to: np.ndarray, q_seed: np.ndarray) -> Optional[List[np.ndarray]]:
        """Straight-line (pos linear, rot geodesic) joint path, None if any waypoint is infeasible."""
        dp, dr = ik.pose_error(T_from, T_to)
        n = max(1, int(math.ceil(max(dp / self.p["step_m"], dr / self.p["step_rad"]))))
        out: List[np.ndarray] = []
        q = q_seed
        for T in ik.interpolate_pose(T_from, T_to, n):
            q = self._ik(T, q)
            if q is None:
                return None
            out.append(q)
        return out

    def _feasible(self, T: np.ndarray, q_seed: np.ndarray) -> Optional[np.ndarray]:
        return self._ik(T, q_seed)

    def _plan_grasp(self, q_now: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]]:
        """-> (T_pre, T_grasp, path_to_pre, path_pre_to_grasp, grasp_x) or None."""
        p = self.p
        T_c = self._object_T(self.food)
        R_c, c = T_c[:3, :3], T_c[:3, 3]
        h = R_c[:, 0].copy(); h[2] = 0.0
        if np.linalg.norm(h) < 1e-6:
            h = np.array([1.0, 0.0, 0.0])
        h /= np.linalg.norm(h)
        T_now = ik.tcp_pose_from_q(q_now)
        best = None
        for gx in p["grasp_x_candidates"]:
            r = carrot_radius_at(gx)
            centre = c + R_c @ np.array([gx, 0.0, r])
            for yaw_err in p["yaw_err_candidates_deg"]:
                cy, sy = math.cos(math.radians(yaw_err)), math.sin(math.radians(yaw_err))
                hy = np.array([cy * h[0] - sy * h[1], sy * h[0] + cy * h[1]])
                for flip in (1.0, -1.0):
                    for tilt in p["tilt_candidates_deg"]:
                        cost = abs(gx) * 10.0 + abs(yaw_err) / 30.0 + abs(tilt) / 20.0
                        if best is not None and cost >= best[0]:
                            continue
                        pg = np.array([centre[0], centre[1], p["grasp_tcp_z"] - self._grasp_deeper])
                        T_g = _top_down_T(pg, flip * hy, tilt_about_y=math.radians(tilt))
                        T_pre = T_g.copy(); T_pre[2, 3] += p["approach_height"]
                        q_pre = self._feasible(T_pre, q_now)
                        if q_pre is None:
                            continue
                        q_g = self._feasible(T_g, q_pre)
                        if q_g is None:
                            continue
                        cost += 0.05 * float(np.max(np.abs(q_pre - q_now)))
                        if best is not None and cost >= best[0]:
                            continue
                        path_pre = self._path(T_now, T_pre, q_now)
                        if path_pre is None:
                            continue
                        path_desc = self._path(T_pre, T_g, path_pre[-1])
                        if path_desc is None:
                            continue
                        best = (cost, T_pre, T_g, path_pre, path_desc, gx, yaw_err, tilt, flip)
        if best is None:
            return None
        _, T_pre, T_g, path_pre, path_desc, gx, yaw_err, tilt, flip = best
        self.log.append(f"grasp plan: x_local={gx:+.3f} yaw_err={yaw_err:+.0f}deg tilt={tilt:+.0f}deg flip={flip:+.0f} "
                        f"pre {len(path_pre)} + descend {len(path_desc)} steps")
        return T_pre, T_g, path_pre, path_desc, gx

    def _predict_food_tips(self, T_tcp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(origin, tip_lo, tip_hi) of the held carrot for a TCP pose, world frame."""
        T_f = T_tcp @ self._T_food_in_tcp
        o = T_f[:3, 3]
        a = T_f @ np.array([CARROT_X_MIN, 0.0, 0.012, 1.0])
        b = T_f @ np.array([CARROT_X_MAX, 0.0, 0.012, 1.0])
        lo, hi = (a[:3], b[:3]) if a[2] <= b[2] else (b[:3], a[:3])
        return o, lo, hi

    def _plan_place(self, q_now: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]]:
        """-> (T_transfer, T_release, path_to_transfer, path_transfer_to_release) or None."""
        p = self.p
        T_pot = self._object_T(self.container)
        pot_c = T_pot[:3, 3]
        rim_z = pot_c[2] + self._rim_z()
        radius = 0.09
        T_now = ik.tcp_pose_from_q(q_now)
        g = p["place_xy_grid"]; R = p["place_xy_radius"]
        offsets = [(dx, dy) for dx in np.arange(-R, R + 1e-9, g) for dy in np.arange(-R, R + 1e-9, g)
                   if dx * dx + dy * dy <= R * R + 1e-9]
        offsets.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
        base_yaw = math.atan2(T_now[1, 1], T_now[0, 1])  # current tool-Y heading
        best = None
        for yaw in p["place_yaw_candidates_deg"]:
            ang = base_yaw + math.radians(yaw)
            y_axis = np.array([math.cos(ang), math.sin(ang)])
            for pitch in p["place_pitch_candidates_deg"]:
                for dx, dy in offsets:
                    T_rel = _top_down_T(np.array([pot_c[0] + dx, pot_c[1] + dy, rim_z + p["release_dz"]]),
                                        y_axis, pitch_about_x=math.radians(pitch))
                    o, lo, hi = self._predict_food_tips(T_rel)
                    r_o = float(np.hypot(*(o[:2] - pot_c[:2])))
                    r_lo = float(np.hypot(*(lo[:2] - pot_c[:2])))
                    r_hi = float(np.hypot(*(hi[:2] - pot_c[:2])))
                    if r_o > radius - 0.02:
                        continue
                    if lo[2] < rim_z - 0.002 and r_lo > radius - 0.01:
                        continue  # the low tip is already below the rim but outside the pot -> would hit the wall
                    geo = r_o + 0.5 * max(0.0, r_lo - (radius - 0.015)) + 0.5 * max(0.0, r_hi - (radius - 0.015))
                    cost = geo * 10.0 + abs(yaw) / 180.0 + abs(pitch) / 60.0
                    if best is not None and cost >= best[0]:
                        continue
                    T_tr = T_rel.copy(); T_tr[2, 3] = max(p["transfer_height"], rim_z + p["release_dz"] + 0.05)
                    q_tr = self._feasible(T_tr, q_now)
                    if q_tr is None:
                        continue
                    q_rel = self._feasible(T_rel, q_tr)
                    if q_rel is None:
                        continue
                    cost += 0.05 * float(np.max(np.abs(q_tr - q_now)))
                    if best is not None and cost >= best[0]:
                        continue
                    path_tr = self._path(T_now, T_tr, q_now)
                    if path_tr is None:
                        continue
                    path_dn = self._path(T_tr, T_rel, path_tr[-1])
                    if path_dn is None:
                        continue
                    best = (cost, T_tr, T_rel, path_tr, path_dn, yaw, pitch, dx, dy, r_o, r_lo, r_hi)
        if best is None:
            return None
        _, T_tr, T_rel, path_tr, path_dn, yaw, pitch, dx, dy, r_o, r_lo, r_hi = best
        self.log.append(f"place plan: yaw={yaw:+.0f}deg pitch={pitch:+.0f}deg tcp offset=({dx:+.2f},{dy:+.2f}) "
                        f"pred carrot r_o={r_o:.3f} r_lo={r_lo:.3f} r_hi={r_hi:.3f}; transfer {len(path_tr)} + lower {len(path_dn)} steps")
        return T_tr, T_rel, path_tr, path_dn

    # ------------------------------------------------------------------ #
    # phase machinery                                                      #
    # ------------------------------------------------------------------ #
    def _push(self, name: str, path: List[np.ndarray], grip: float, goal_T: Optional[np.ndarray],
              timeout: Optional[int] = None) -> None:
        self._phases.append(_Phase(name, list(path), grip, goal_T,
                                   int(self.p["phase_timeout_steps"] if timeout is None else timeout)))

    def _fail(self, why: str, q: np.ndarray) -> None:
        self.log.append("FAIL: " + why)
        self._phases = [_Phase("HOLD", [], 0.0, None, 10 ** 9)]
        self.done = True

    def reset(self, world_info: Optional[Dict[str, Any]] = None) -> None:
        """Harness protocol: called once per episode after `world.reset`. Returns None
        (the oracle never overrides the layout / start pose)."""
        if not self._explicit_limits:
            lim = _limits_from(world_info) or _limits_from(self.world)
            if lim is not None:
                self.limits = lim
        self._phases = []
        self._retries = 0
        self._grasp_deeper = 0.0
        self._q_last = None
        self._T_food_in_tcp = None
        self._branch = -1
        self.log = []
        self.meta["log"] = self.log  # live reference: the plan/slip/timeout notes of the current episode
        self.phase_name = "INIT"
        self.done = False
        self.step_count = 0
        return None

    def _start_grasp(self, q: np.ndarray) -> None:
        plan = self._plan_grasp(q)
        if plan is None:
            self._fail("no envelope-feasible grasp pose for the carrot", q)
            return
        T_pre, T_g, path_pre, path_desc, _gx = plan
        self._grasp_T = T_g
        self._push("PREGRASP", path_pre, 0.0, T_pre)
        self._push("DESCEND", path_desc, 0.0, T_g, timeout=90)
        self._push("CLOSE", [], 1.0, None, timeout=self.p["close_timeout_steps"])
        T_lift = T_g.copy(); T_lift[2, 3] += self.p["lift_height"]
        path_lift = self._path(T_g, T_lift, path_desc[-1])
        if path_lift is None:
            # lift as far as the envelope allows
            for frac in (0.75, 0.5, 0.3):
                T_lift = T_g.copy(); T_lift[2, 3] += frac * self.p["lift_height"]
                path_lift = self._path(T_g, T_lift, path_desc[-1])
                if path_lift is not None:
                    break
        if path_lift is None:
            self._fail("no feasible lift path", q)
            return
        self._push("LIFT", path_lift, 1.0, T_lift)

    def _after_lift(self, q: np.ndarray) -> None:
        T_food = self._object_T(self.food)
        rise = float(T_food[2, 3])
        if rise < self.p["slip_min_rise"]:
            self.log.append(f"slip: carrot origin z={rise:.3f} after LIFT")
            if self._retries < int(self.p["max_retries"]):
                self._retries += 1
                self._grasp_deeper += float(self.p["retry_deeper"])
                self._push("OPEN", [], 0.0, None, timeout=self.p["open_steps"])
                self._phases.append(_Phase("REPLAN_GRASP", [], 0.0, None, 1))
                return
            self._fail("carrot not lifted after retry", q)
            return
        T_tcp = ik.tcp_pose_from_q(q)
        self._T_food_in_tcp = np.linalg.inv(T_tcp) @ T_food
        plan = self._plan_place(q)
        if plan is None:
            self._fail("no envelope-feasible release pose over the pot", q)
            return
        T_tr, T_rel, path_tr, path_dn = plan
        self._push("TRANSFER", path_tr, 1.0, T_tr, timeout=150)
        self._push("LOWER", path_dn, 1.0, T_rel, timeout=90)
        self._push("OPEN", [], 0.0, None, timeout=self.p["open_steps"])
        T_up = T_rel.copy(); T_up[2, 3] += self.p["retreat_height"]
        path_up = self._path(T_rel, T_up, path_dn[-1]) or []
        self._push("RETREAT", path_up, 0.0, T_up if path_up else None, timeout=60)
        self._phases.append(_Phase("HOLD", [], 0.0, None, 10 ** 9))

    def _phase_complete(self, ph: _Phase, q: np.ndarray, grip_pos: float) -> bool:
        """True when `ph` (already emitted >= 1 step) has reached its goal or timed out."""
        if ph.name == "HOLD":
            return False
        if ph.steps <= 0:
            return False
        if ph.name == "CLOSE":
            settled = (ph.steps >= self.p["close_settle_steps"] and grip_pos >= self.p["close_min_grip_pos"]
                       and abs(grip_pos - self._grip_prev) < self.p["close_settle_eps"])
            self._grip_prev = grip_pos
            if settled:
                self.log.append(f"CLOSE: settled grip_pos={grip_pos:.3f} after {ph.steps} steps")
                return True
            if ph.steps >= ph.timeout:
                self.log.append(f"CLOSE: timeout grip_pos={grip_pos:.3f}")
                return True
            return False
        if ph.i < len(ph.path):
            return False  # still emitting waypoints
        if ph.goal_T is None:
            return ph.steps >= ph.timeout
        dp, dr = ik.pose_error(ik.tcp_pose_from_q(q), ph.goal_T)
        if dp <= self.p["pos_tol"] and dr <= self.p["rot_tol"]:
            return True
        if ph.steps >= ph.timeout:
            self.log.append(f"{ph.name}: timeout (pose err {dp * 1000:.0f} mm / {math.degrees(dr):.1f} deg)")
            return True
        return False

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=float).reshape(7)
        q, grip_pos = state[:6], float(state[6])
        self.step_count += 1
        if self._branch < 0:
            self._branch = ur_kin.branch_id(q)
        if not self._phases:
            self._q_last = q.copy()
            self._start_grasp(q)
        # -- phase transitions (bounded loop: each iteration pops one phase) ------
        for _ in range(16):
            ph = self._phases[0]
            if ph.name == "REPLAN_GRASP":
                self._phases.pop(0)
                self._start_grasp(q)
                continue
            if not self._phase_complete(ph, q, grip_pos):
                break
            finished = self._phases.pop(0)
            if finished.name == "LIFT":
                self._after_lift(q)
            if not self._phases:
                self._phases.append(_Phase("HOLD", [], 0.0, None, 10 ** 9))
            if self._phases[0].name == "CLOSE":
                self._grip_prev = grip_pos
        # -- emit ------------------------------------------------------------------
        ph = self._phases[0]
        self.phase_name = ph.name
        if ph.i < len(ph.path):
            q_cmd = ph.path[ph.i]
            ph.i += 1
            self._q_last = q_cmd
        else:
            q_cmd = self._q_last if self._q_last is not None else q
        ph.steps += 1
        if ph.name == "HOLD":
            self.done = True
        return [float(v) for v in q_cmd] + [float(ph.grip)]

    _grip_prev: float = 0.0


# --------------------------------------------------------------------------- #
# Debug-only render helper (never for policy observations)                     #
# --------------------------------------------------------------------------- #
_OPENING_SITE_RE = re.compile(r'<site name="(?P<name>[^"]*_opening)"(?P<attrs>[^>]*)/>')


def debug_trigger_xml(xml: str, rgba: Sequence[float] = (0.1, 0.6, 1.0, 0.25)) -> str:
    """Return a COPY of a built scene xml where every container `<name>_opening` site
    (an invisible sphere site whose size = the inner radius, at rim height) is turned
    into a translucent CYLINDER site spanning the container's floor..rim — the
    `TaskEvaluator` success volume made visible. Sites never collide and the change is
    purely visual, but this xml must only feed a separate debug renderer
    (`--debug-trigger`), never the model whose cameras the policy observes."""
    def repl(m: "re.Match[str]") -> str:
        attrs = m.group("attrs")
        size_m = re.search(r'size="([^"]*)"', attrs)
        pos_m = re.search(r'pos="([^"]*)"', attrs)
        radius = float(size_m.group(1).split()[0]) if size_m else 0.05
        pos = [float(v) for v in pos_m.group(1).split()] if pos_m else [0.0, 0.0, 0.0]
        rim_z = pos[2] if len(pos) == 3 else 0.0
        half_h = max(rim_z / 2.0, 1e-3)
        attrs = re.sub(r'\s(rgba|size|pos|type)="[^"]*"', "", attrs)
        rgba_s = " ".join(f"{float(v):g}" for v in rgba)
        return (f'<site name="{m.group("name")}"{attrs} type="cylinder" size="{radius:g} {half_h:g}" '
                f'pos="{pos[0]:g} {pos[1]:g} {rim_z - half_h:g}" rgba="{rgba_s}"/>')
    return _OPENING_SITE_RE.sub(repl, xml)


__all__ = ["ScriptedPolicy", "DEFAULT_PARAMS", "carrot_radius_at", "CARROT_SEGMENTS"]
