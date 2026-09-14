"""Policies for the eval harness (DESIGN.md §2.2).

    class Policy(Protocol):
        needs_images: bool                       # False -> the world skips rendering
        meta: dict                               # recorded into summary.json
        def reset(self, world_info: dict) -> dict | None
            # may return {"layout_override": {...}, "q0": [6], "max_steps_hint": int}
        def act(self, obs: dict) -> list[7] | None   # None = FAULT (episode ends)

* ZmqPolicy    — the REAL remote-policy protocol via `gello_policy.obs_assembler` (stdlib
                 only, imported from ros2_ur_ws/src/gello_policy) and `policy_server.zmq_protocol`.
                 REQ socket with RCVTIMEO/SNDTIMEO = timeout, rebuilt on every failure exactly
                 like policy_leader_node._connect_socket / _zmq_roundtrip; a timeout/error is
                 a FAULT (act -> None). Optional v2 RESET fields (policy_type, checkpoint,
                 state_dim, action_dim) are parsed; state_dim/action_dim != 7 is refused.
* ReplayPolicy — replays a recorded sim take's `command` (q_cmd @125 Hz) + `gripper.grip_cmd`
                 at 30 Hz (nearest row to the episode time); reset() hands the world the
                 take's INITIAL layout from `sim_object_poses` row 0 (position + quaternion —
                 the takes' `layout_seed` metadata is wrong) and the arm's recorded start
                 pose (`sim_mj_state` row 0; the demos started at the GELLO pose, not home).
* ZeroPolicy   — holds the home pose + grip 0 (negative control).

ScriptedPolicy (F2) lives in scripted_policy.py; run_eval imports it lazily.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GELLO_POLICY = os.path.join(_ROOT, "ros2_ur_ws", "src", "gello_policy")
for _p in (_ROOT, _GELLO_POLICY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gello_policy import obs_assembler  # noqa: E402  (stdlib-only module; verified: no rclpy import)
from policy_server import zmq_protocol  # noqa: E402  (stdlib-only constants)

N_JOINTS = 6
ACTION_LEN = 7


@runtime_checkable
class Policy(Protocol):
    needs_images: bool
    meta: Dict[str, Any]

    def reset(self, world_info: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...

    def act(self, obs: Dict[str, Any]) -> Optional[List[float]]: ...


def _finite7(a: Sequence[float]) -> Optional[List[float]]:
    try:
        out = [float(v) for v in a]
    except (TypeError, ValueError):
        return None
    if len(out) != ACTION_LEN or not all(math.isfinite(v) for v in out):
        return None
    return out


# --------------------------------------------------------------------------- #
# ZMQ (real protocol)                                                           #
# --------------------------------------------------------------------------- #
class PolicyRefused(RuntimeError):
    """The server describes a checkpoint this harness cannot drive (e.g. EEF 16-D state)."""


class ZmqPolicy:
    """REQ client speaking the exact wire format of policy_leader_node (ZMQ transport).

    `endpoint`: "zmq://host:port" | "tcp://host:port" | "host:port".
    `task` is recorded in `meta` only — the wire RESET is the verbatim b'{"cmd":"reset"}'
    (the FM server's task string is a server-side setting, as on the real robot).
    `act_timeout_faults=False` turns a timed-out ACT into "repeat the last action" instead
    of a fault (diagnostic only; the real client always faults).
    """

    needs_images = True

    def __init__(self, endpoint: str, task: str = "Put carrot in pot", timeout_s: float = 0.6,
                 act_timeout_faults: bool = True) -> None:
        import zmq
        self._zmq = zmq
        ep = str(endpoint)
        for pre in ("zmq://", "tcp://"):
            if ep.startswith(pre):
                ep = ep[len(pre):]
        host, _, port = ep.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"endpoint must be zmq://host:port, got {endpoint!r}")
        self.endpoint = zmq_protocol.default_endpoint(host, int(port))
        self.task = str(task)
        self.timeout_s = float(timeout_s)
        self.act_timeout_faults = bool(act_timeout_faults)
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._connect_socket()
        self.last_error: Optional[str] = None
        self.last_action: Optional[List[float]] = None
        self.n_faults = 0
        self.n_acts = 0
        self.rtt_ms: List[float] = []
        self.meta: Dict[str, Any] = {"policy": "zmq", "endpoint": self.endpoint, "task": self.task,
                                     "timeout_s": self.timeout_s, "transport": "zmq"}

    # -- socket lifecycle: verbatim semantics of policy_leader_node --------------
    def _connect_socket(self) -> None:
        zmq = self._zmq
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:  # noqa: BLE001
                pass
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        sock.connect(self.endpoint)
        self._sock = sock

    def _roundtrip(self, req_frames: List[bytes]) -> List[bytes]:
        zmq = self._zmq
        try:
            self._sock.send_multipart(req_frames)
            return self._sock.recv_multipart()
        except zmq.error.Again as exc:
            self._connect_socket()
            raise TimeoutError(f"ZMQ timeout after {self.timeout_s}s") from exc
        except zmq.ZMQError as exc:
            self._connect_socket()
            raise RuntimeError(f"ZMQ error: {exc}") from exc

    # -- Policy API -----------------------------------------------------------------
    def reset(self, world_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        reply = obs_assembler.parse_reply(self._roundtrip(obs_assembler.build_reset_request()))
        if not reply.get(obs_assembler.KEY_OK, False):
            raise RuntimeError(f"RESET refused by server: {reply.get(obs_assembler.KEY_ERR)}")
        v2 = {k: reply[k] for k in ("policy_type", "checkpoint", "state_dim", "action_dim", "torch_seed",
                                     "n_action_steps", "checkpoint_revision") if k in reply}
        for k in ("state_dim", "action_dim"):
            if k in v2 and int(v2[k]) != ACTION_LEN:
                raise PolicyRefused(f"server reports {k}={v2[k]}; this harness drives joint-space 7/7 only "
                                    f"(an EEF checkpoint is out of scope in v1)")
        self.meta.update({"server": v2, "reset_reply": {k: v for k, v in reply.items() if k != obs_assembler.KEY_OK}})
        self.last_action = None
        return None

    def act(self, obs: Dict[str, Any]) -> Optional[List[float]]:
        if obs.get("cam1_jpeg") is None or obs.get("cam2_jpeg") is None:
            self.last_error = "observation has no images (world.observe(images=False)?)"
            self.n_faults += 1
            return None
        req = obs_assembler.build_act_request(obs["state"], obs["cam1_jpeg"], obs["cam2_jpeg"])
        t0 = time.perf_counter()
        try:
            action = obs_assembler.parse_action_reply(self._roundtrip(req))
        except TimeoutError as exc:
            self.last_error = str(exc)
            self.n_faults += 1
            if not self.act_timeout_faults and self.last_action is not None:
                return list(self.last_action)
            return None
        except Exception as exc:  # noqa: BLE001 -- any failure => FAULT (as the real client)
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.n_faults += 1
            return None
        self.rtt_ms.append((time.perf_counter() - t0) * 1e3)
        out = _finite7(action)
        if out is None:
            self.last_error = f"non-finite/invalid action {action!r}"
            self.n_faults += 1
            return None
        self.last_action = out
        self.n_acts += 1
        return out

    def stats(self) -> Dict[str, Any]:
        r = np.asarray(self.rtt_ms, dtype=float)
        return {"n_acts": self.n_acts, "n_faults": self.n_faults, "last_error": self.last_error,
                "rtt_ms_median": float(np.median(r)) if r.size else None,
                "rtt_ms_p95": float(np.percentile(r, 95)) if r.size else None,
                "rtt_ms_max": float(r.max()) if r.size else None}

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:  # noqa: BLE001
                pass
            self._sock = None


# --------------------------------------------------------------------------- #
# Replay of a recorded take                                                     #
# --------------------------------------------------------------------------- #
class ReplayPolicy:
    """Replays `command` (q_cmd @125 Hz) + `gripper.grip_cmd` of one sim take at 30 Hz.

    Episode time 0 = the take's first `sim_mj_state` row (the same instant the initial
    layout / arm pose are taken from), so the replayed command stream and the world start
    in the same state. After the take ends the last command is held; `max_steps_hint`
    lets run_eval stop `dwell + 1 s` after the take's end instead of at --max-steps.
    """

    needs_images = False

    def __init__(self, take_dir: str, policy_hz: float = 30.0) -> None:
        import h5py
        self.take_dir = os.path.abspath(take_dir)
        self.take_name = os.path.basename(self.take_dir.rstrip("/"))
        self.policy_hz = float(policy_hz)
        with h5py.File(os.path.join(self.take_dir, "vectors.h5"), "r") as f:
            c = f["command"]
            self.cmd_t = c["t_rel_s"][:]
            self.cmd_q = np.stack([c[f"cmd{i + 1}"][:] for i in range(N_JOINTS)], 1)
            g = f["gripper"]
            self.grip_t = g["t_rel_s"][:]
            self.grip_cmd = g["grip_cmd"][:]
            s = f["sim_mj_state"]
            self.t_base = float(s["t_rel_s"][0])
            self.q0 = np.array([float(s[f"qpos{i}"][0]) for i in range(N_JOINTS)])
            o = f["sim_object_poses"]
            cols = json.loads(o.attrs["columns"])
            names = sorted({c_[:-2] for c_ in cols if c_.endswith("_x")})
            self.layout0: Dict[str, Dict[str, Any]] = {}
            for name in names:
                pos = [float(o[f"{name}_{a}"][0]) for a in ("x", "y", "z")]
                qx, qy, qz, qw = (float(o[f"{name}_{a}"][0]) for a in ("qx", "qy", "qz", "qw"))
                self.layout0[name] = {"pos": pos, "quat_wxyz": [qw, qx, qy, qz]}
            self.t_obj0 = float(o["t_rel_s"][0])
            meta = json.loads(f.attrs["sim_meta"]) if "sim_meta" in f.attrs else {}
        self.duration_s = float(self.cmd_t[-1] - self.t_base)
        self.recorder_flag = meta.get("task_success_at_stop")
        self.meta: Dict[str, Any] = {"policy": "replay", "take": self.take_name, "take_dir": self.take_dir,
                                     "duration_s": self.duration_s, "recorder_task_success_at_stop": self.recorder_flag,
                                     "layout_seed_in_meta": meta.get("layout_seed"),
                                     "t_base_s": self.t_base, "q0": self.q0.tolist(), "layout0": self.layout0}
        if not np.all(np.isfinite(self.cmd_q)):
            raise ValueError(f"{self.take_name}: non-finite command rows")

    def reset(self, world_info: Dict[str, Any]) -> Dict[str, Any]:
        dwell = float(world_info.get("dwell_s", 1.0))
        return {"layout_override": self.layout0, "q0": self.q0.tolist(),
                "max_steps_hint": int(math.ceil((self.duration_s + dwell + 1.0) * self.policy_hz))}

    def act(self, obs: Dict[str, Any]) -> Optional[List[float]]:
        t = float(obs["t"]) + self.t_base
        i = int(np.argmin(np.abs(self.cmd_t - t)))
        j = int(np.argmin(np.abs(self.grip_t - t)))
        return [float(v) for v in self.cmd_q[i]] + [float(self.grip_cmd[j])]


# --------------------------------------------------------------------------- #
# Zero (negative control)                                                       #
# --------------------------------------------------------------------------- #
class ZeroPolicy:
    """Holds the start pose (home_joints) with the gripper open. SR must be 0."""

    needs_images = False

    def __init__(self) -> None:
        self._hold: Optional[List[float]] = None
        self.meta: Dict[str, Any] = {"policy": "zero"}

    def reset(self, world_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._hold = [float(v) for v in world_info["home_joints"]] + [0.0]
        return None

    def act(self, obs: Dict[str, Any]) -> Optional[List[float]]:
        return list(self._hold) if self._hold is not None else [float(v) for v in obs["state"][:N_JOINTS]] + [0.0]


def make_policy(spec: str, *, task: str = "Put carrot in pot", timeout_s: float = 0.6, world=None,
                take_dir: Optional[str] = None):
    """`zmq://host:port` | `replay` (needs take_dir) | `scripted` (F2) | `zero`."""
    s = str(spec)
    if s.startswith("zmq://") or s.startswith("tcp://"):
        return ZmqPolicy(s, task=task, timeout_s=timeout_s)
    if s == "zero":
        return ZeroPolicy()
    if s == "replay":
        if not take_dir:
            raise ValueError("--policy replay needs a take directory")
        return ReplayPolicy(take_dir)
    if s == "scripted":
        try:
            from sim_collect.eval.scripted_policy import ScriptedPolicy  # F2's deliverable
        except ImportError as exc:
            raise ImportError("--policy scripted needs sim_collect/eval/scripted_policy.py (owner F2); "
                              f"import failed: {exc}") from exc
        return ScriptedPolicy(world)
    raise ValueError(f"unknown policy spec {spec!r} (zmq://host:port | replay | scripted | zero)")
