"""Helpers shared by F2's tests (test_cameras / test_recorder / test_capture).

Not a test module. Provides a tiny self-contained scene (plane + box + one hinge body +
one free body + the four contract cameras) and a synthetic ``state`` message generator
that follows DESIGN.md §2.1, so the recorder/capture can be exercised without the sim
process, the GELLO or F1's scene.
"""
from __future__ import annotations

import math
import struct
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

# No <visual><global> on purpose: cameras.build_model must raise the framebuffer itself.
TINY_SCENE_XML = """
<mujoco model="f2_tiny">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.6 0.5 0.4 1"/>
    <body name="arm" pos="0 0 0.3">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" rgba="0.2 0.4 0.9 1"/>
    </body>
    <body name="carrot" pos="-0.45 0 0.05">
      <freejoint/>
      <geom name="carrot_geom" type="box" size="0.05 0.05 0.05" rgba="1 0.4 0.1 1"/>
    </body>
    <camera name="cam1" pos="-0.7 0 0.571" xyaxes="0 -1 0 0.45 0 0.65" fovy="42"/>
    <camera name="cam1_depth" pos="-0.7 -0.015 0.571" xyaxes="0 -1 0 0.45 0 0.65" fovy="58.7"/>
    <camera name="cam2" pos="-0.3 0.3 0.4" xyaxes="0 -1 0 0.45 0 0.65" fovy="42"/>
    <camera name="cam2_depth" pos="-0.3 0.285 0.4" xyaxes="0 -1 0 0.45 0 0.65" fovy="58.7"/>
  </worldbody>
</mujoco>
"""
TINY_NQ = 1 + 7
TINY_NV = 1 + 6
OBJECT_NAMES = ("carrot", "pot")


def make_state(tick: int, t: float, sim_t: float, leader_t: Optional[float] = None,
               engaged: bool = True, success: bool = False) -> Dict[str, Any]:
    """One DESIGN §2.1 ``state`` message with smooth synthetic signals."""
    ph = 0.7 * sim_t
    q = [0.3 * math.sin(ph + k) for k in range(6)]
    qd = [0.3 * 0.7 * math.cos(ph + k) for k in range(6)]
    q_cmd = [v + 0.01 for v in q]
    q_lead = [v + 0.02 for v in q]
    trig = 0.5 + 0.5 * math.sin(0.5 * sim_t)
    qpos = [q[0]] + [-0.45, 0.0, 0.05 + 0.02 * math.sin(ph), 1.0, 0.0, 0.0, 0.0]
    qvel = [qd[0]] + [0.0] * 6
    return {
        "t": t, "sim_t": sim_t, "tick": tick,
        "control_mode": "eef", "eef_state": "ENGAGED" if engaged else "DISENGAGED",
        "eef_info": {"state": "ENGAGED" if engaged else "DISENGAGED",
                     "sigma_min": 0.12 + 0.01 * math.sin(ph), "gamma": 1.0, "ls_scale": 1.0},
        "engaged": engaged, "pos_scale": 1.0,
        "q_lead_raw": q_lead, "q_lead_unwrapped": q_lead, "q_lead_f": [v + 0.001 for v in q_lead],
        "qd_lead": qd, "trigger": trig, "leader_t": leader_t if leader_t is not None else t,
        "q_cmd": q_cmd, "q": q, "qd": qd, "eff": [2.0 * v for v in qd],
        "tcp_pos": [-0.48 + 0.05 * math.sin(ph), -0.05, 0.31],
        "tcp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
        "cmd_tcp_pos": [-0.47, -0.05, 0.31], "cmd_tcp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
        "wrench": [0.1 * k + 0.05 * math.sin(ph) for k in range(6)],
        "grip_cmd": trig, "grip_pos": 0.9 * trig,
        "qpos_full": qpos, "qvel_full": qvel,
        "objects": {"carrot": {"pos": qpos[1:4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    "pot": {"pos": [-0.45, 0.25, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        "task": {"success": success, "detail": "synthetic"},
    }


class StateStream:
    """Generates 250 Hz state messages with a 30 Hz leader clock, like the sim."""

    def __init__(self, hz: float = 250.0, leader_hz: float = 30.0):
        self.hz = hz
        self.leader_hz = leader_hz
        self.tick = 0
        self.t0 = time.time()
        self._leader_t = None

    def next(self, success: bool = False) -> Dict[str, Any]:
        sim_t = self.tick / self.hz
        now = time.time()
        leader_slot = math.floor(sim_t * self.leader_hz) / self.leader_hz
        leader_t = self.t0 + leader_slot
        msg = make_state(self.tick, now, sim_t, leader_t=leader_t, success=success)
        self.tick += 1
        return msg


def synthetic_jpeg(seq: int, w: int = 1280, h: int = 720) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = (seq * 7) % 256
    img[:, :, 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def synthetic_depth_png(seq: int, w: int = 848, h: int = 480) -> bytes:
    arr = (np.linspace(300, 3000, w, dtype=np.uint16)[None, :]
           + np.uint16(seq % 50)).astype(np.uint16)
    arr = np.repeat(arr, h, axis=0)
    arr[: h // 5, :] = 0
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert ok
    return buf.tobytes()


def compressed_depth(png: bytes) -> bytes:
    return struct.pack("<iff", 0, 0.0, 0.0) + png
