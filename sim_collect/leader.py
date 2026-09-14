"""GELLO leader reader (DESIGN.md §2.1 "Leader thread") + a scripted fake.

Both leaders publish `LeaderSample`s into a lock-protected slot at `hz` (30):
    q_raw[6]        joints straight from the driver (UR convention, calibrated)
    q_unwrapped[6]  branch-continuous: `angle_utils.wrapped_nearest(q_raw, previous)`
                    per joint, exactly as the ROS bridge's `_on_joint_state` does.
                    This is THE ONE unwrap chain: the controller consumes it as is,
                    and re-anchors it (`reanchor`) onto the arm's branch at seed
                    time, so the recorded stream and the consumed stream coincide
    qd[6]           finite difference of q_unwrapped between samples
    trigger         gripper 0.0 = open .. 1.0 = closed (driver's normalised value)
    t               time.monotonic() of the read
    seq             sample counter (a consumer detects "new sample" by seq)

Calibration comes from the ROS `gello_publisher` parameters (`load_leader_calibration`)
so sim and real share one source; the config's `leader.overrides` may deviate.

SAFETY: torque is NEVER enabled here. `DynamixelRobot.__init__` disables it on
connect and nothing in this module calls `set_torque_mode`. Only ONE process may
own the serial port (the driver kills other holders on connect).
"""
from __future__ import annotations

import dataclasses
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from ur_gello_bringup.angle_utils import wrapped_nearest  # all-joint unwrap, as the ROS bridge

from sim_collect.scene import resolve_path


@dataclasses.dataclass(frozen=True)
class LeaderSample:
    q_raw: np.ndarray
    q_unwrapped: np.ndarray
    qd: np.ndarray
    trigger: float
    t: float
    seq: int


class _LeaderBase:
    """Thread scaffolding shared by the real and fake leaders."""

    def __init__(self, hz: float = 30.0, name: str = "leader") -> None:
        self.hz = float(hz)
        self._lock = threading.Lock()
        self._latest: Optional[LeaderSample] = None
        self._prev_unwrapped: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._name = name
        self.read_errors = 0

    # -- subclass API -----------------------------------------------------
    def _read_once(self, t: float):
        """Return (q6, trigger) or raise."""
        raise NotImplementedError

    # -- unwrap + publish ---------------------------------------------------
    def _ingest(self, q: Sequence[float], trigger: float, t: float) -> LeaderSample:
        q_raw = np.asarray(q, dtype=float).reshape(6).copy()
        if self._prev_unwrapped is None:
            q_un = q_raw.copy()
            qd = np.zeros(6)
        else:
            q_un = np.asarray(wrapped_nearest(q_raw, self._prev_unwrapped), dtype=float)
            dt = t - (self._prev_t if self._prev_t is not None else t)
            qd = (q_un - self._prev_unwrapped) / dt if dt > 1e-6 else np.zeros(6)
        self._prev_unwrapped = q_un.copy()
        self._prev_t = t
        self._seq += 1
        s = LeaderSample(q_raw=q_raw, q_unwrapped=q_un, qd=qd,
                         trigger=float(min(max(trigger, 0.0), 1.0)), t=t, seq=self._seq)
        with self._lock:
            self._latest = s
        return s

    def reanchor(self, shift: Sequence[float]) -> None:
        """Shift the unwrap chain by `shift` (multiples of 2*pi per joint) so
        subsequent samples continue on the consumer's branch (controller seed)."""
        sh = np.asarray(shift, dtype=float).reshape(6)
        with self._lock:
            if self._prev_unwrapped is not None:
                self._prev_unwrapped = self._prev_unwrapped + sh
            if self._latest is not None:
                s = self._latest
                self._latest = LeaderSample(q_raw=s.q_raw, q_unwrapped=s.q_unwrapped + sh, qd=s.qd,
                                            trigger=s.trigger, t=s.t, seq=s.seq)

    def sample_now(self, t: Optional[float] = None) -> LeaderSample:
        """Synchronous read (no thread needed) — used at startup and in tests."""
        t = time.monotonic() if t is None else t
        q, trig = self._read_once(t)
        return self._ingest(q, trig, t)

    def latest(self) -> Optional[LeaderSample]:
        with self._lock:
            return self._latest

    # -- thread ---------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_now()
            except Exception as e:  # noqa: BLE001 keep the thread alive; consumer sees staleness
                self.read_errors += 1
                if self.read_errors <= 5 or self.read_errors % 100 == 0:
                    print(f"[{self._name}] read error #{self.read_errors}: {e}")
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def close(self) -> None:
        self.stop()


class FakeLeader(_LeaderBase):
    """Scripted leader: holds `q0`, optional sinusoidal wiggle, settable trigger.

    `set_pose(q)` / `set_offset(dq)` retarget it (tests move it by 5 cm through
    IK); `set_trigger(v)` drives the gripper. `sample_now(t)` lets tests drive
    it with an explicit clock, no thread.
    """

    def __init__(self, q0: Sequence[float], hz: float = 30.0, wiggle_amp: float = 0.0,
                 wiggle_hz: float = 0.2, wiggle_joints: Sequence[int] = (0,), trigger: float = 0.0) -> None:
        super().__init__(hz=hz, name="fake_leader")
        self._q0 = np.asarray(q0, dtype=float).reshape(6).copy()
        self._offset = np.zeros(6)
        self._trigger = float(trigger)
        self.wiggle_amp = float(wiggle_amp)
        self.wiggle_hz = float(wiggle_hz)
        self.wiggle_joints = tuple(int(j) for j in wiggle_joints)
        self._t0: Optional[float] = None
        self.is_fake = True

    def set_pose(self, q: Sequence[float]) -> None:
        with self._lock:
            self._q0 = np.asarray(q, dtype=float).reshape(6).copy()
            self._offset[:] = 0.0

    def set_offset(self, dq: Sequence[float]) -> None:
        with self._lock:
            self._offset = np.asarray(dq, dtype=float).reshape(6).copy()

    def set_trigger(self, v: float) -> None:
        with self._lock:
            self._trigger = float(v)

    def _read_once(self, t: float):
        with self._lock:
            q = self._q0 + self._offset
            trig = self._trigger
            if self._t0 is None:
                self._t0 = t
            if self.wiggle_amp > 0.0:
                w = self.wiggle_amp * math.sin(2 * math.pi * self.wiggle_hz * (t - self._t0))
                q = q.copy()
                for j in self.wiggle_joints:
                    q[j] += w
        return q, trig


def load_leader_calibration(path: str, node: str = "gello_publisher") -> Dict[str, Any]:
    """Read the GELLO calibration from the ROS parameter yaml (`<node>.ros__parameters`):
    port, joint_ids, joint_offsets, joint_signs, gripper_config (id cast to int)."""
    with open(resolve_path(path), "r") as f:
        d = yaml.safe_load(f) or {}
    sec = d.get(node, {}).get("ros__parameters")
    if not sec:
        raise KeyError(f"{path}: no {node}.ros__parameters section")
    gc = sec["gripper_config"]
    return {
        "port": str(sec["port"]),
        "joint_ids": [int(i) for i in sec["joint_ids"]],
        "joint_offsets": [float(v) for v in sec["joint_offsets"]],
        "joint_signs": [int(v) for v in sec["joint_signs"]],
        "gripper_config": [int(gc[0]), float(gc[1]), float(gc[2])],
        "source": resolve_path(path),
    }


def port_holders(port: str) -> List[Dict[str, Any]]:
    """Processes (other than us) holding `port` open, via /proc/<pid>/fd.
    Returns [{"pid", "cmdline"}]. The Dynamixel driver runs `fuser -k` on the
    port when it looks busy, which would KILL a ROS gello_publisher or the
    playground sim — so we refuse to construct it while anyone holds the port."""
    target = os.path.realpath(port)
    me = os.getpid()
    out: List[Dict[str, Any]] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    try:
                        with open(f"/proc/{pid}/cmdline", "rb") as f:
                            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
                    except OSError:
                        cmd = "?"
                    out.append({"pid": int(pid), "cmdline": cmd})
                    break
            except OSError:
                continue
    return out


class GelloLeader(_LeaderBase):
    """Physical GELLO via `gello.agents.gello_agent.DynamixelRobotConfig.make_robot`.

    `get_joint_state()` -> 7 floats (6 arm + gripper 0..1). Startup: the driver
    prints `warning, comm failed` for its first reads; `warmup_reads` samples are
    discarded and read errors are retried until `connect_timeout_s`. A driver
    that silently fell back to its FAKE implementation (port busy / missing) is
    detected and rejected — a fake leader must be asked for explicitly.
    """

    def __init__(self, port: str, joint_ids: Sequence[int], joint_offsets: Sequence[float],
                 joint_signs: Sequence[int], gripper_config: Sequence[float],
                 start_joints: Optional[Sequence[float]] = None, hz: float = 30.0,
                 warmup_reads: int = 5, connect_timeout_s: float = 15.0, baudrate: int = 57600) -> None:
        super().__init__(hz=hz, name="gello_leader")
        from gello.agents.gello_agent import DynamixelRobotConfig  # lazy: pulls dynamixel_sdk

        self.port = port
        self.is_fake = False
        holders = port_holders(port)
        if holders:
            who = "; ".join(f"pid {h['pid']}: {h['cmdline'][:120]}" for h in holders)
            raise RuntimeError(f"GELLO port {port} is held by another process ({who}). Refusing to start: "
                               "the Dynamixel driver would `fuser -k` it. Stop that process first.")
        cfg = DynamixelRobotConfig(
            joint_ids=tuple(int(i) for i in joint_ids),
            joint_offsets=tuple(float(o) for o in joint_offsets),
            joint_signs=tuple(int(s) for s in joint_signs),
            gripper_config=(int(gripper_config[0]), float(gripper_config[1]), float(gripper_config[2])),
        )
        sj = None if start_joints is None else np.asarray(start_joints, dtype=float)
        if sj is not None and sj.shape[0] == 6:
            sj = np.concatenate([sj, [0.0]])
        # NOTE: make_robot(port=..., start_joints=...) constructs DynamixelRobot(real=True) whose
        # driver is DynamixelDriver(use_fake_fallback=True) — hence the fake check below.
        self._robot = cfg.make_robot(port=port, start_joints=sj)
        drv = getattr(self._robot, "_driver", None)
        if getattr(drv, "_is_fake", False):
            self.close()
            raise RuntimeError(f"GELLO driver fell back to the FAKE driver on {port} (port missing/busy?). "
                               "Refusing to run: use --fake-leader if you want a scripted leader.")
        if baudrate != 57600:
            print(f"[gello_leader] note: baudrate {baudrate} requested; DynamixelRobotConfig.make_robot "
                  "uses the driver default (57600)")
        # warm-up: discard the first reads, retrying through the driver's comm warnings
        deadline = time.monotonic() + float(connect_timeout_s)
        good = 0
        while good < int(warmup_reads):
            try:
                js = np.asarray(self._robot.get_joint_state(), dtype=float)
                if js.shape[0] >= 7 and np.all(np.isfinite(js)):
                    good += 1
            except Exception as e:  # noqa: BLE001
                if time.monotonic() > deadline:
                    self.close()
                    raise RuntimeError(f"GELLO warm-up failed on {port}: {e}") from e
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"GELLO warm-up timed out after {connect_timeout_s}s on {port}")
            time.sleep(0.02)

    def _read_once(self, t: float):
        js = np.asarray(self._robot.get_joint_state(), dtype=float)
        if js.shape[0] < 7 or not np.all(np.isfinite(js)):
            raise RuntimeError(f"bad GELLO sample {js}")
        return js[:6], float(js[6])

    def close(self) -> None:
        self.stop()
        robot = getattr(self, "_robot", None)
        drv = getattr(robot, "_driver", None)
        if drv is not None and hasattr(drv, "close"):
            try:
                drv.close()
            except Exception as e:  # noqa: BLE001
                print(f"[gello_leader] close: {e}")
        self._robot = None


def resolve_leader_config(cfg_leader: Dict[str, Any], home: Sequence[float]) -> Dict[str, Any]:
    """Merge the ROS calibration (`leader.calibration_source`) with `leader.overrides`
    and the sim-only keys; start_joints = home (6) + gripper 0 unless overridden."""
    out: Dict[str, Any] = {}
    src = cfg_leader.get("calibration_source")
    if src:
        out.update(load_leader_calibration(src, str(cfg_leader.get("calibration_node", "gello_publisher"))))
    for k in ("port", "joint_ids", "joint_offsets", "joint_signs", "gripper_config", "start_joints"):
        if k in cfg_leader:  # legacy inline keys still honoured
            out[k] = cfg_leader[k]
    out.update(cfg_leader.get("overrides") or {})
    out.setdefault("start_joints", list(map(float, home)) + [0.0])
    out["warmup_reads"] = int(cfg_leader.get("warmup_reads", 5))
    out["connect_timeout_s"] = float(cfg_leader.get("connect_timeout_s", 15.0))
    out["baudrate"] = int(cfg_leader.get("baudrate", 57600))
    for k in ("port", "joint_ids", "joint_offsets", "joint_signs", "gripper_config"):
        if k not in out:
            raise KeyError(f"leader config: missing {k!r} (set leader.calibration_source or leader.overrides)")
    return out


def make_leader(cfg_leader: dict, fake: bool, home: Sequence[float], hz: float,
                fake_wiggle: float = 0.0) -> _LeaderBase:
    """Factory used by sim_main: FakeLeader at `home` or the physical GELLO."""
    if fake:
        return FakeLeader(home, hz=hz, wiggle_amp=fake_wiggle)
    lc = resolve_leader_config(cfg_leader, home)
    return GelloLeader(
        port=str(lc["port"]), joint_ids=lc["joint_ids"], joint_offsets=lc["joint_offsets"],
        joint_signs=lc["joint_signs"], gripper_config=lc["gripper_config"], start_joints=lc["start_joints"],
        hz=hz, warmup_reads=lc["warmup_reads"], connect_timeout_s=lc["connect_timeout_s"], baudrate=lc["baudrate"],
    )
