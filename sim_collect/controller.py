"""Teleop controller = the ROS `gello_ur_bridge` command pipeline, ROS-free.

Mirrors `ur_gello_bringup/gello_ur_bridge_node.py` (`_on_joint_state`, `_on_timer`,
`_run_eef_gates`, `_on_eef_engage/disengage/reclutch`) using the SAME pure
modules the node runs — `bridge_stages` (OneEuro, filter_stage_joint,
clamp_stage, command_pipeline), `eef_delta.EefDeltaController`, `ur_kin`,
`angle_utils` — and the SAME parameter files (`load_bridge_params`). The node
itself needs rclpy, so its per-tick logic is mirrored here rather than imported.

Per control tick (250 Hz), exactly as the node:
  1. leader staleness watchdog (`staleness_timeout_s`) -> re-seed on recovery,
     auto-disengage if engaged;
  2. SEED branch: first tick / after a re-seed the command chain is seeded from
     the arm's ACTUAL pose (zero jump) and the unwrapped leader is re-anchored on
     the branch nearest the arm; soft-start ramp restarts;
  3. soft-start: effective slew `step` ramps 15 % -> 100 % of `max_step_rad`
     over `soft_start_s`;
  4. the dedicated leader One-Euro bank is stepped once per tick (eef mode);
  5. eef stage (ENGAGED only): `EefDeltaController.step(q_lead_f, step)`,
     HOLD latch (`hold_latch_s`) -> auto-disengage, exception -> auto-disengage;
  6. `bridge_stages.command_pipeline(...)` with `hold_when_not_engaged=True`
     in eef mode ("3D pen": the arm never mirrors an un-engaged leader).

Sim-specific differences (documented, deliberate):
  * joint mode also HOLDS while not engaged (DESIGN §2.1). `command_pipeline`
    cannot express that in joint mode (its stage 2 is skipped there), so this
    controller runs `filter_stage_joint` alone (keeps the bank live, identical
    arithmetic) and holds `last_published`. Engaging in joint mode re-seeds from
    the actual pose (like the node's `~/resume`) so the glide toward the leader
    is jump-free and soft-started; it is gated by fresh + quasi-still leader and
    a per-joint circular gap <= `joint_engage_max_gap_rad`.
  * there is no "paused" bridge: DISENGAGED simply holds the last command.
  * `set_pos_scale` while ENGAGED is deferred to the next engage/reclutch (an
    immediate change would re-scale the live delta).

States (`eef_state`): HOLD (seeded, holding, ready to engage — the node's boot
state), ENGAGED, DISENGAGED (after an operator/auto disengage; still holding).
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ur_gello_bringup import ur_kin
from ur_gello_bringup.angle_utils import circular_dist, leader_quasi_still, wrapped_nearest
from ur_gello_bringup.bridge_stages import OneEuro, clamp_stage, command_pipeline, filter_stage_joint
from ur_gello_bringup.eef_delta import EefDeltaController

from sim_collect.leader import LeaderSample
from sim_collect.scene import resolve_path

N_JOINTS = 6

_BRIDGE_DEFAULTS: Dict[str, Any] = {
    "filter_type": "one_euro", "one_euro_min_cutoff": 1.0, "one_euro_beta": 2.0, "one_euro_d_cutoff": 1.0,
    "ema_alpha": 0.4, "max_step_rad": 0.0025, "deadband_rad": 0.004, "staleness_timeout_s": 0.5,
    "soft_start_s": 0.7, "resume_chase_still_speed": 0.10, "resume_chase_still_window_s": 0.3,
    "publish_rate_hz": 250.0, "anchor_agree_tol": 0.02, "filter_settled_tol": 0.005, "hold_latch_s": 2.0,
    "tick_budget_us": 1000.0, "pos_scale": 1.0, "r_align_rpy": [0.0, 0.0, 0.0],
    "tool_l_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "tool_r_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "v_max": 0.08, "w_max": 0.5, "sigma_warn": 0.10, "sigma_stop": 0.03, "gamma_min": 0.05,
    "char_length": 0.30, "branch_tol": 0.25, "branch_weights": [1.0] * 6, "limit_margin_rad": 0.05,
    "s_floor": 0.02, "lag_max_pose": [0.05, 0.3], "max_excursion_m": 0.5, "ik_backend": "analytic",
    "keepout_json": "{}",
}


def load_bridge_params(paths: Sequence[str], node: str = "gello_ur_bridge") -> Dict[str, Any]:
    """Merge `<node>.ros__parameters` from the given ROS param yamls, in order
    (later files overlay earlier ones — the eef yaml overlays the base yaml)."""
    out: Dict[str, Any] = dict(_BRIDGE_DEFAULTS)
    for p in paths:
        with open(resolve_path(p), "r") as f:
            d = yaml.safe_load(f) or {}
        sec = d.get(node, {}).get("ros__parameters", {})
        out.update(sec)
    return out


class TeleopController:
    """See module docstring. Not thread-safe: call from the physics thread."""

    def __init__(self, params: Dict[str, Any], control_mode: str = "eef", control_hz: float = 250.0,
                 joint_engage_max_gap_rad: float = 1.5, reanchor=None) -> None:
        """`reanchor(shift)` (optional): called at every seed with the 2*pi*k per-joint
        shift that moved the leader stream onto the arm's branch; wire it to
        `leader.reanchor` so the leader's unwrap chain IS the consumed chain."""
        if control_mode not in ("eef", "joint"):
            raise ValueError(f"control_mode must be eef|joint, got {control_mode!r}")
        self.params = dict(params)
        self.mode = control_mode
        self.hz = float(control_hz)
        self.dt = 1.0 / self.hz
        p = self.params
        self.max_step_rad = float(p["max_step_rad"])
        self.soft_start_s = float(p["soft_start_s"])
        self.staleness_timeout_s = float(p["staleness_timeout_s"])
        self.ema_alpha = float(p["ema_alpha"])
        self.deadband_rad = float(p["deadband_rad"])
        self.still_speed = float(p["resume_chase_still_speed"])
        self.still_window_s = float(p["resume_chase_still_window_s"])
        self.anchor_agree_tol = float(p["anchor_agree_tol"])
        self.filter_settled_tol = float(p["filter_settled_tol"])
        self.hold_latch_s = float(p["hold_latch_s"])
        self.tick_budget_us = float(p["tick_budget_us"])
        self.joint_engage_max_gap = float(joint_engage_max_gap_rad)
        self._q_lead_f_max_age_s = max(4.0 / self.hz, 0.05)
        self.use_euro = str(p.get("filter_type", "one_euro")) == "one_euro"
        self.T_tool_R = ur_kin.xyz_rpy_to_T(p["tool_r_xyz_rpy"])

        # leader ingest state
        self._raw_target: Optional[List[float]] = None
        self._last_leader_t: Optional[float] = None
        self._last_seq: int = -1
        self._history: Deque[Tuple[float, List[float]]] = deque(maxlen=64)
        # command chain
        self._filtered: Optional[List[float]] = None
        self._gated_target: Optional[List[float]] = None
        self._last_published: Optional[List[float]] = None
        self._seed_time: Optional[float] = None
        self._was_stale = False
        self.q_cmd: Optional[np.ndarray] = None
        self.q_lead_f: Optional[List[float]] = None
        self._q_lead_f_time: Optional[float] = None
        # eef
        self._eef: Optional[EefDeltaController] = None
        self._euro: Optional[List[OneEuro]] = None
        self._euro_lead: Optional[List[OneEuro]] = None
        self._eef_state = "HOLD"
        self._hold_since: Optional[float] = None
        self._auto_reason: Optional[str] = None
        self._eef_info: Dict[str, Any] = {}
        self._pending_pos_scale: Optional[float] = None
        self.pos_scale = float(p["pos_scale"])
        self._last_gate: Tuple[str, str] = ("", "")
        self.tick_us = 0.0
        self.step_eff = self.max_step_rad
        self.tick_count = 0
        self._reanchor_cb = reanchor
        self._last_sample_unwrapped: Optional[np.ndarray] = None
        self._build_banks()

    # ------------------------------------------------------------------ #
    def _build_banks(self) -> None:
        p = self.params
        mk = lambda: OneEuro(self.dt, float(p["one_euro_min_cutoff"]), float(p["one_euro_beta"]),  # noqa: E731
                             float(p["one_euro_d_cutoff"]))
        self._euro = [mk() for _ in range(N_JOINTS)] if self.use_euro else None
        if self.mode == "eef":
            import json
            try:
                keepout = json.loads(p.get("keepout_json") or "{}")
            except (ValueError, TypeError):
                keepout = {}
            self._eef = EefDeltaController({
                "pos_scale": self.pos_scale, "r_align_rpy": p["r_align_rpy"],
                "tool_l_xyz_rpy": p["tool_l_xyz_rpy"], "tool_r_xyz_rpy": p["tool_r_xyz_rpy"],
                "v_max": float(p["v_max"]), "w_max": float(p["w_max"]), "sigma_warn": float(p["sigma_warn"]),
                "sigma_stop": float(p["sigma_stop"]), "gamma_min": float(p["gamma_min"]),
                "char_length": float(p["char_length"]), "branch_tol": float(p["branch_tol"]),
                "branch_weights": list(p["branch_weights"]), "limit_margin_rad": float(p["limit_margin_rad"]),
                "s_floor": float(p["s_floor"]), "lag_max_pose": list(p["lag_max_pose"]),
                "max_excursion_m": float(p["max_excursion_m"]), "dt": self.dt, "keepout": keepout,
                "ik_backend": str(p["ik_backend"]),
            })
            self._euro_lead = [mk() for _ in range(N_JOINTS)]
        else:
            self._eef = None
            self._euro_lead = None

    # ------------------------------------------------------------------ #
    # Leader ingest (the node's _on_joint_state)                          #
    # ------------------------------------------------------------------ #
    def ingest(self, sample: Optional[LeaderSample]) -> bool:
        """Feed the newest leader sample (idempotent per `seq`). Returns True if new."""
        if sample is None or sample.seq == self._last_seq:
            return False
        self._last_seq = sample.seq
        # ONE unwrap chain: consume the leader's q_unwrapped verbatim. After the
        # seed re-anchored that chain (reanchor callback) the safety net below is
        # an identity; without a callback it only bridges the seed's branch shift.
        lead = [float(v) for v in sample.q_unwrapped]
        self._last_sample_unwrapped = np.asarray(lead, dtype=float)
        unwrapped = lead if self._raw_target is None else wrapped_nearest(lead, self._raw_target)
        now = float(sample.t)
        _dt = None if self._last_leader_t is None else now - self._last_leader_t
        if self._euro is not None:
            for i in range(N_JOINTS):
                self._euro[i].update_input(unwrapped[i], _dt)
        if self._euro_lead is not None:
            for i in range(N_JOINTS):
                self._euro_lead[i].update_input(unwrapped[i], _dt)
        self._raw_target = list(unwrapped)
        self._last_leader_t = now
        self._history.append((now, list(unwrapped)))
        return True

    def leader_age(self, now: float) -> Optional[float]:
        return None if self._last_leader_t is None else now - self._last_leader_t

    # ------------------------------------------------------------------ #
    # Control tick (the node's _on_timer)                                  #
    # ------------------------------------------------------------------ #
    def reseed(self) -> None:
        """Forget the command chain: the next tick seeds from the arm's actual
        pose with a soft start (after a teleport / reset / mode switch)."""
        if self._eef is not None:
            self._eef.disengage()
        self._filtered = None
        self._gated_target = None
        self._last_published = None
        self._seed_time = None
        self.q_lead_f = None
        self._q_lead_f_time = None
        self._eef_state = "HOLD"
        self._hold_since = None
        self._auto_reason = None
        self._eef_info = {}

    def _autodisengage(self, reason: str) -> None:
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "DISENGAGED"
        self._hold_since = None
        self._auto_reason = reason

    def tick(self, q_actual: Sequence[float], now: Optional[float] = None) -> Optional[np.ndarray]:
        """One 250 Hz tick. Returns the joint command (None until seeded)."""
        t_start = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        self.tick_count += 1
        q_actual_l = [float(v) for v in q_actual]
        try:
            return self._tick(q_actual_l, now)
        finally:
            self.tick_us = (time.perf_counter() - t_start) * 1e6

    def _tick(self, q_actual: List[float], now: float) -> Optional[np.ndarray]:
        if self._raw_target is None or self._last_leader_t is None:
            return self.q_cmd  # nothing valid received yet: hold
        age = now - self._last_leader_t
        if age > self.staleness_timeout_s:
            self._filtered = None
            self._last_published = None
            self._was_stale = True
            if self._eef_state == "ENGAGED":
                self._autodisengage("leader_stale")
            return self.q_cmd
        # SEED branch: zero-jump start from the ACTUAL pose.
        if self._filtered is None or self._last_published is None:
            anchored = wrapped_nearest(self._raw_target, q_actual)
            shift = [a - r for a, r in zip(anchored, self._raw_target)]
            self._raw_target = anchored
            if self._reanchor_cb is not None and any(abs(v) > 1e-9 for v in shift):
                self._reanchor_cb(shift)  # move the leader's chain onto this branch too
                if self._last_sample_unwrapped is not None:
                    self._last_sample_unwrapped = self._last_sample_unwrapped + np.asarray(shift)
            self._filtered = list(q_actual)
            self._last_published = list(q_actual)
            self._gated_target = list(anchored)
            if self._euro is not None:
                for i in range(N_JOINTS):
                    self._euro[i].seed(q_actual[i])
            if self._euro_lead is not None:
                for i in range(N_JOINTS):
                    self._euro_lead[i].seed(anchored[i])
                self.q_lead_f = None
                self._q_lead_f_time = None
            self._seed_time = now
            self._was_stale = False
            self.q_cmd = np.asarray(self._last_published, dtype=float)
            return self.q_cmd

        step = self.max_step_rad
        if self.soft_start_s > 0.0 and self._seed_time is not None:
            frac = (now - self._seed_time) / self.soft_start_s
            if frac < 1.0:
                step = self.max_step_rad * (0.15 + 0.85 * max(0.0, frac))
        self.step_eff = step

        if self._euro_lead is not None:
            self.q_lead_f = [self._euro_lead[i](self._raw_target[i]) for i in range(N_JOINTS)]
            self._q_lead_f_time = now

        eef_command = None
        if self.mode == "eef" and self._eef_state == "ENGAGED":
            try:
                if self.q_lead_f is None:
                    raise RuntimeError("leader filter cache empty")
                q_cmd, info = self._eef.step(self.q_lead_f, step)
                self._eef_info = info
                if q_cmd is not None:
                    eef_command = [float(v) for v in q_cmd]
                if info.get("state") == "HOLD":
                    if self._hold_since is None:
                        self._hold_since = now
                    elif (now - self._hold_since) > self.hold_latch_s:
                        self._autodisengage(f"hold_latched (reason={info.get('reject_reason')})")
                        eef_command = None
                else:
                    self._hold_since = None
            except Exception as exc:  # noqa: BLE001 fail closed on ANY eef fault
                self._autodisengage(f"exception: {exc}")
                eef_command = None

        engaged = self._eef_state == "ENGAGED"
        if self.mode == "joint" and not engaged:
            # sim addition: joint mode HOLDS when not engaged (keep the bank live)
            filter_stage_joint(self._raw_target, self._filtered, self._gated_target, self._euro,
                               self.ema_alpha, self.deadband_rad)
            out = list(self._last_published)
        else:
            out = command_pipeline(
                self.mode, engaged, self._raw_target, self._filtered, self._gated_target, self._euro,
                self.ema_alpha, self.deadband_rad, self._last_published, step,
                eef_command=eef_command, hold_when_not_engaged=(self.mode == "eef"),
            )
        self._last_published = out
        self.q_cmd = np.asarray(out, dtype=float)
        return self.q_cmd

    # ------------------------------------------------------------------ #
    # Gates + clutch                                                       #
    # ------------------------------------------------------------------ #
    def _selftest(self, q_anchor: Sequence[float]) -> bool:
        """G7: IK(FK(q)) == q round trip (solver integrity)."""
        try:
            qa = np.asarray(q_anchor, dtype=float).reshape(6)
            T = ur_kin.fk(qa)
            if str(self.params.get("ik_backend", "analytic")) == "analytic":
                cand = [ur_kin.wrapped_nearest(np.asarray(s, float), qa) for s in ur_kin.ik_analytic(T)]
                cand = [c for c in cand if bool(np.all(np.isfinite(c)))]
                if not cand:
                    return False
                best = min(cand, key=lambda c: float(np.max(np.abs(c - qa))))
            else:
                best = ur_kin.ik_numeric(T, qa)
                if best is None:
                    return False
                best = np.asarray(best, dtype=float).reshape(6)
            if float(np.max(np.abs(best - qa))) >= 1e-6:
                return False
            xi = ur_kin.se3_log(np.linalg.inv(ur_kin.fk(best)) @ T)
            return float(np.linalg.norm(xi[:3])) < 1e-4 and float(np.linalg.norm(xi[3:])) < 1e-4
        except Exception:  # noqa: BLE001
            return False

    def _common_gates(self, now: float) -> Tuple[bool, str, str]:
        # G2: fresh leader
        age = self.leader_age(now)
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            a = "n/a" if age is None else f"{age:.2f}s"
            return False, "leader_stale", f"age={a} > {self.staleness_timeout_s:.2f}s"
        # G3: command baseline
        if self._last_published is None:
            return False, "no_command_baseline", "command chain not seeded yet"
        # G5: leader quasi-still
        if not leader_quasi_still(list(self._history), self.still_window_s, self.still_speed):
            return False, "leader_moving", "leader not demonstrably still"
        return True, "ok", ""

    def _eef_gates(self, q_actual: List[float], now: float, run_baseline: bool):
        if self.mode != "eef":
            return False, "not_eef_mode", "control_mode != eef", None, None
        ok, key, detail = self._common_gates(now)
        if not ok:
            return False, key, detail, None, None
        if run_baseline:  # G4: command chain agrees with the arm
            agree = max(abs(self._last_published[i] - q_actual[i]) for i in range(N_JOINTS))
            if agree > self.anchor_agree_tol:
                return (False, "chains_disagree", f"max|cmd-actual|={agree:.4f} > {self.anchor_agree_tol:.4f}",
                        None, None)
        q_anchor = list(self._last_published)
        # G6: leader filter running + settled
        q_lead_f = self.q_lead_f
        lead_age = None if self._q_lead_f_time is None else now - self._q_lead_f_time
        if q_lead_f is None or lead_age is None or lead_age > self._q_lead_f_max_age_s:
            a = "n/a" if lead_age is None else f"{lead_age:.3f}s"
            return False, "filter_not_running", f"leader filter cache age={a}", None, None
        settle = max(abs(q_lead_f[i] - self._raw_target[i]) for i in range(N_JOINTS))
        if settle >= self.filter_settled_tol:
            return (False, "filter_not_settled", f"max|q_lead_f-raw|={settle:.5f} >= {self.filter_settled_tol:.5f}",
                    None, None)
        # G7: kinematics self-test
        if not self._selftest(q_anchor):
            return False, "kinematics_selftest", "IK(FK(q))!=q round-trip failed", None, None
        # G8: not singular
        sig = float(ur_kin.sigma_min(np.asarray(q_anchor, dtype=float), float(self.params["char_length"])))
        if sig <= float(self.params["sigma_warn"]):
            return False, "singular_anchor", f"sigma_min={sig:.4f} <= sigma_warn={self.params['sigma_warn']:.4f}", None, None
        # G9: keepout
        if not ur_kin.keepout_ok(np.asarray(q_anchor, dtype=float), self._eef.keepout):
            return False, "keepout", "anchor pose violates keepout", None, None
        return True, "ok", "", q_anchor, list(q_lead_f)

    def engage(self, q_actual: Sequence[float], now: Optional[float] = None) -> Tuple[bool, str, str]:
        """Clutch in. eef: gates G2..G9 then anchor (zero jump). joint: gates +
        re-seed so the arm glides to the leader jump-free."""
        now = time.monotonic() if now is None else float(now)
        q_actual_l = [float(v) for v in q_actual]
        if self._eef_state == "ENGAGED":
            return True, "already_engaged", "already ENGAGED"
        if self.mode == "eef":
            ok, key, detail, q_anchor, q_lead_f = self._eef_gates(q_actual_l, now, run_baseline=True)
            self._last_gate = (key, detail)
            if not ok:
                return False, key, detail
            if self._pending_pos_scale is not None:
                self.pos_scale = self._pending_pos_scale
                self._pending_pos_scale = None
            self._eef.pos_scale = self.pos_scale
            summary = self._eef.engage(q_anchor, q_lead_f)
            self._eef_state = "ENGAGED"
            self._hold_since = None
            self._auto_reason = None
            p = summary["T_r_anchor"][:3, 3]
            return True, "ok", (f"ENGAGED anchor p=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) branch={summary['branch0']} "
                                f"sigma_min={summary['sigma_min']:.3f} pos_scale={self.pos_scale:.2f}")
        # joint mode
        ok, key, detail = self._common_gates(now)
        self._last_gate = (key, detail)
        if not ok:
            return False, key, detail
        gap = max(circular_dist(self._raw_target[i], q_actual_l[i]) for i in range(N_JOINTS))
        if gap > self.joint_engage_max_gap:
            return False, "gap_too_large", f"max circular gap {gap:.3f} > {self.joint_engage_max_gap:.3f} rad"
        self._filtered = None  # re-seed from actual on the next tick (soft-started glide)
        self._last_published = None
        self._eef_state = "ENGAGED"
        self._auto_reason = None
        return True, "ok", f"ENGAGED (joint) gap={gap:.3f} rad, slew {self.max_step_rad * self.hz:.2f} rad/s max"

    def disengage(self) -> Tuple[bool, str, str]:
        """Ungated, immediate. The arm holds its last command."""
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "DISENGAGED"
        self._hold_since = None
        self._auto_reason = None
        return True, "ok", "DISENGAGED — anchor discarded; arm holds"

    def reclutch(self, now: Optional[float] = None) -> Tuple[bool, str, str]:
        """Re-anchor at the current leader/command pose (zero delta). ENGAGED only."""
        now = time.monotonic() if now is None else float(now)
        if self.mode != "eef":
            return False, "not_eef_mode", "reclutch is an eef-mode operation"
        if self._eef_state != "ENGAGED":
            return False, "not_engaged", f"state={self._eef_state}"
        ok, key, detail, _q_anchor, q_lead_f = self._eef_gates(list(self._last_published), now, run_baseline=False)
        if not ok:
            return False, key, detail
        if self._pending_pos_scale is not None:
            self.pos_scale = self._pending_pos_scale
            self._pending_pos_scale = None
            self._eef.pos_scale = self.pos_scale
        self._eef.reclutch(q_lead_now=q_lead_f, q_cmd_now=self._last_published)
        self._hold_since = None
        self._auto_reason = None
        return True, "ok", "RE-CLUTCHED — anchor reset at current pose"

    def set_pos_scale(self, value: float) -> Tuple[bool, str, str]:
        v = float(value)
        if not math.isfinite(v) or not (0.05 <= v <= 2.0):
            return False, "bad_value", f"pos_scale must be in [0.05, 2.0], got {value!r}"
        if self._eef_state == "ENGAGED":
            self._pending_pos_scale = v
            return True, "deferred", f"pos_scale={v:.2f} applies at the next engage/reclutch"
        self.pos_scale = v
        self._pending_pos_scale = None
        if self._eef is not None:
            self._eef.pos_scale = v
        return True, "ok", f"pos_scale={v:.2f}"

    def set_control_mode(self, mode: str) -> Tuple[bool, str, str]:
        if mode not in ("eef", "joint"):
            return False, "bad_mode", f"mode must be eef|joint, got {mode!r}"
        if self._eef_state == "ENGAGED":
            return False, "engaged", "disengage before switching control mode"
        if mode == self.mode:
            return True, "ok", f"already {mode}"
        self.mode = mode
        self._build_banks()
        self.reseed()
        return True, "ok", f"control_mode={mode} (re-seeded)"

    # ------------------------------------------------------------------ #
    @property
    def engaged(self) -> bool:
        return self._eef_state == "ENGAGED"

    @property
    def eef_state(self) -> str:
        return self._eef_state

    def cmd_tcp(self) -> Tuple[np.ndarray, np.ndarray]:
        """(pos, quat_xyzw) of fk(q_cmd) @ T_tool_R; zeros if no command yet."""
        if self.q_cmd is None:
            return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        T = ur_kin.fk(self.q_cmd) @ self.T_tool_R
        return T[:3, 3].copy(), ur_kin.mat_to_quat_xyzw(T[:3, :3])

    def info(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Diagnostics for the state message (`eef_info`)."""
        now = time.monotonic() if now is None else float(now)
        i = self._eef_info or {}
        settle = None
        if self.q_lead_f is not None and self._raw_target is not None:
            settle = max(abs(self.q_lead_f[k] - self._raw_target[k]) for k in range(N_JOINTS))
        return {
            "mode": self.mode,
            "state": self._eef_state,
            "ctrl_state": i.get("state"),
            "reject_reason": i.get("reject_reason"),
            "auto_reason": self._auto_reason,
            "last_gate": self._last_gate[0], "last_gate_detail": self._last_gate[1],
            "sigma_min": i.get("sigma_min"), "gamma": i.get("gamma"), "ls_scale": i.get("ls_scale"),
            "ik_residual": i.get("ik_residual"), "lag_pos": i.get("lag_pos"), "lag_rot": i.get("lag_rot"),
            "lag": [i.get("lag_pos"), i.get("lag_rot")],
            "excursion": i.get("excursion"), "branch_id": i.get("branch_id"),
            "n_ik_solutions": i.get("n_ik_solutions"),
            "pos_scale": self.pos_scale, "pending_pos_scale": self._pending_pos_scale,
            "step_eff": self.step_eff, "soft_start_active": bool(
                self._seed_time is not None and (now - self._seed_time) < self.soft_start_s),
            "leader_age": self.leader_age(now), "filter_settle": settle,
            "seeded": self._last_published is not None,
            # 2*pi*k offset between the consumed target and the leader's published
            # chain; exactly zero when `reanchor` is wired (one chain).
            "lead_branch_shift": (
                [float(a - b) for a, b in zip(self._raw_target, self._last_sample_unwrapped)]
                if self._raw_target is not None and self._last_sample_unwrapped is not None else None),
            "tick_us": self.tick_us, "tick_over_budget": self.tick_us > self.tick_budget_us,
            "hold_when_not_engaged": True,
        }


__all__ = ["TeleopController", "load_bridge_params", "clamp_stage"]
