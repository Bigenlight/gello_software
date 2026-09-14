"""Pure helpers for the read-only GELLO leader probe (``scripts/gello_probe.py``).

Everything in this module is hardware-free and testable: config parsing, the
calibration formula, the boot-time 2*pi auto-wrap, serial-port matching,
inventory classification and the reference-pose diagnosis.  The only functions
that touch a Dynamixel bus are ``ping_inventory`` / ``read_positions`` and they
take an already-open ``(PortHandler, PacketHandler)`` pair and issue **ping and
read instructions only** — there is no write anywhere in this file.

The calibration math deliberately mirrors ``gello.robots.dynamixel.DynamixelRobot``
(``get_joint_state`` and the ``start_joints`` offset wrap in ``__init__``) so the
probe prints exactly what ``gello_publisher`` would publish at boot.  A test in
``gello/dynamixel/tests/test_gello_probe.py`` pins that equivalence against the
real class.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Control-table addresses (XL330 / protocol 2.0). Read-only use here.
ADDR_FIRMWARE_VERSION = 6
ADDR_TORQUE_ENABLE = 64  # driver.py:21
ADDR_PRESENT_POSITION = 132  # driver.py:24
TICKS_PER_REV = 4096  # driver.py: ticks / 2048 * pi

# Dynamixel model numbers -> names. Only the ones plausibly on a GELLO.
MODEL_NAMES: Dict[int, str] = {
    1200: "XL330-M288-T",
    1190: "XL330-M077-T",
    1230: "XC330-T288-T",
    1220: "XC330-T181-T",
    1210: "XC330-M288-T",
    1240: "XC330-M181-T",
    1060: "XL430-W250-T",
    1020: "XM430-W350-T",
    1030: "XM430-W210-T",
    1120: "XM540-W150-T",
    1130: "XM540-W270-T",
}

# The calibrated individual (docs/testing/02_GELLO_LEADER.md §2, measured 2026-07-27):
# arm IDs 1..6 are XL330-M288 (1200), trigger ID 7 is XL330-M077 (1190).
DOCUMENTED_FINGERPRINT: Dict[int, int] = {1: 1200, 2: 1200, 3: 1200, 4: 1200, 5: 1200, 6: 1200, 7: 1190}

ROS_CONFIG_REL = "ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml"
MUJOCO_CONFIG_REL = "configs/rwh_ur.yaml"
CONFIG_SOURCES = {"ros": ROS_CONFIG_REL, "mujoco": MUJOCO_CONFIG_REL}


@dataclass(frozen=True)
class LeaderConfig:
    """The calibration block a GELLO consumer needs, source-agnostic."""

    source: str
    path: str
    port: str
    joint_ids: Tuple[int, ...]
    joint_offsets: Tuple[float, ...]
    joint_signs: Tuple[int, ...]
    gripper_config: Optional[Tuple[int, float, float]]  # (id, open_deg, close_deg)
    start_joints: Optional[Tuple[float, ...]]  # arm joints (+ trailing gripper 0..1 if present)

    @property
    def gripper_id(self) -> Optional[int]:
        return None if self.gripper_config is None else int(self.gripper_config[0])

    @property
    def all_ids(self) -> Tuple[int, ...]:
        return self.joint_ids + ((self.gripper_id,) if self.gripper_id is not None else ())

    @property
    def start_arm_joints(self) -> Optional[Tuple[float, ...]]:
        if self.start_joints is None:
            return None
        return tuple(self.start_joints[: len(self.joint_ids)])


# --------------------------------------------------------------------------- config


def _as_leader_config(source: str, path: str, port: str, block: Mapping, start_joints) -> LeaderConfig:
    gc = block.get("gripper_config")
    gripper = None if gc is None else (int(round(float(gc[0]))), float(gc[1]), float(gc[2]))
    return LeaderConfig(
        source=source,
        path=path,
        port=str(port),
        joint_ids=tuple(int(i) for i in block["joint_ids"]),
        joint_offsets=tuple(float(o) for o in block["joint_offsets"]),
        joint_signs=tuple(int(s) for s in block["joint_signs"]),
        gripper_config=gripper,
        start_joints=None if start_joints is None else tuple(float(s) for s in start_joints),
    )


def parse_ros_config(doc: Mapping, path: str = ROS_CONFIG_REL) -> LeaderConfig:
    """``ur7e_gello.yaml`` -> LeaderConfig (``gello_publisher.ros__parameters``)."""
    params = doc["gello_publisher"]["ros__parameters"]
    return _as_leader_config("ros", path, params["port"], params, params.get("start_joints"))


def parse_mujoco_config(doc: Mapping, path: str = MUJOCO_CONFIG_REL) -> LeaderConfig:
    """``configs/rwh_ur.yaml`` -> LeaderConfig (``agent.dynamixel_config``)."""
    agent = doc["agent"]
    return _as_leader_config(
        "mujoco", path, agent["port"], agent["dynamixel_config"], agent.get("start_joints")
    )


def load_config(source: str, repo_root: str) -> LeaderConfig:
    import yaml  # PyYAML — checked at CLI start

    rel = CONFIG_SOURCES[source]
    path = os.path.join(repo_root, rel)
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    parser = parse_ros_config if source == "ros" else parse_mujoco_config
    return parser(doc, path)


def compare_configs(a: LeaderConfig, b: LeaderConfig, atol: float = 1e-6) -> List[str]:
    """Human-readable differences in the calibration block between two sources.

    The values are stored in (at least) three places in this repo — the ROS yaml,
    the mujoco yaml and the ``gello_publisher_node.py`` parameter defaults — so
    drift between them is a known trap; the probe warns whenever these two differ.
    """
    diffs: List[str] = []
    if a.port != b.port:
        diffs.append(f"port: {a.source}={a.port!r} vs {b.source}={b.port!r}")
    if a.joint_ids != b.joint_ids:
        diffs.append(f"joint_ids: {a.source}={list(a.joint_ids)} vs {b.source}={list(b.joint_ids)}")
    if a.joint_signs != b.joint_signs:
        diffs.append(f"joint_signs: {a.source}={list(a.joint_signs)} vs {b.source}={list(b.joint_signs)}")
    if len(a.joint_offsets) != len(b.joint_offsets):
        diffs.append(
            f"joint_offsets length: {a.source}={len(a.joint_offsets)} vs {b.source}={len(b.joint_offsets)}"
        )
    else:
        for i, (x, y) in enumerate(zip(a.joint_offsets, b.joint_offsets)):
            if abs(x - y) > atol:
                d = y - x
                diffs.append(
                    f"joint_offsets[{i}] (J{i + 1}): {a.source}={x:.3f} vs {b.source}={y:.3f}"
                    f"  (diff {d:+.3f} rad = {d / (math.pi / 2):+.2f} x pi/2)"
                )
    if (a.gripper_config is None) != (b.gripper_config is None):
        diffs.append(f"gripper_config: {a.source}={a.gripper_config} vs {b.source}={b.gripper_config}")
    elif a.gripper_config is not None and b.gripper_config is not None:
        if a.gripper_config[0] != b.gripper_config[0] or any(
            abs(x - y) > atol for x, y in zip(a.gripper_config[1:], b.gripper_config[1:])
        ):
            diffs.append(f"gripper_config: {a.source}={a.gripper_config} vs {b.source}={b.gripper_config}")
    if a.start_joints != b.start_joints:
        sa = None if a.start_joints is None else list(a.start_joints)
        sb = None if b.start_joints is None else list(b.start_joints)
        if sa is None or sb is None or len(sa) != len(sb) or any(abs(x - y) > atol for x, y in zip(sa, sb)):
            diffs.append(f"start_joints: {a.source}={sa} vs {b.source}={sb}")
    return diffs


# --------------------------------------------------------------------------- serial ports


def list_serial_by_id(by_id_dir: str = "/dev/serial/by-id") -> List[Tuple[str, str]]:
    """``[(link_name, resolved_target), ...]`` for every entry in ``/dev/serial/by-id``."""
    if not os.path.isdir(by_id_dir):
        return []
    out = []
    for name in sorted(os.listdir(by_id_dir)):
        link = os.path.join(by_id_dir, name)
        out.append((name, os.path.realpath(link)))
    return out


def match_port(configured: str, entries: Sequence[Tuple[str, str]], by_id_dir: str = "/dev/serial/by-id"):
    """Return ``(present: bool, resolved_target: Optional[str])`` for a configured port.

    A configured ``/dev/serial/by-id/...`` path matches by link name; a raw
    ``/dev/ttyUSBn`` path matches by resolved target.
    """
    base = os.path.basename(configured)
    if os.path.dirname(configured) == by_id_dir.rstrip("/"):
        for name, target in entries:
            if name == base:
                return True, target
        return False, None
    for _name, target in entries:
        if os.path.basename(target) == base:
            return True, target
    return False, None


# --------------------------------------------------------------------------- calibration


def ticks_to_rad(ticks: int) -> float:
    """4-byte present-position ticks -> rad, exactly as ``DynamixelDriver.get_joints``."""
    if ticks > 0x7FFFFFFF:
        ticks -= 0x100000000
    return ticks / (TICKS_PER_REV / 2) * math.pi


def calibrate(
    raw_rad: Sequence[float],
    joint_offsets: Sequence[float],
    joint_signs: Sequence[int],
    gripper_open_close_deg: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """``DynamixelRobot.get_joint_state`` without the EMA: ``(raw - offset) * sign``.

    ``raw_rad`` is the arm joints followed by the gripper motor (if configured);
    ``joint_offsets``/``joint_signs`` are the *arm-only* config lists — the gripper
    gets offset 0 / sign +1 appended exactly like ``DynamixelRobot.__init__`` does.
    The gripper element is mapped to [0, 1] between open_deg and close_deg and clamped.
    """
    offsets = list(joint_offsets)
    signs = list(joint_signs)
    if gripper_open_close_deg is not None:
        offsets.append(0.0)
        signs.append(1)
    raw = np.asarray(raw_rad, dtype=float)
    if raw.shape != (len(offsets),):
        raise ValueError(f"raw_rad has {raw.shape[0]} elements, config expects {len(offsets)}")
    pos = (raw - np.asarray(offsets, dtype=float)) * np.asarray(signs, dtype=float)
    if gripper_open_close_deg is not None:
        open_rad = gripper_open_close_deg[0] * math.pi / 180
        close_rad = gripper_open_close_deg[1] * math.pi / 180
        g = (pos[-1] - open_rad) / (close_rad - open_rad)
        pos[-1] = min(max(0.0, g), 1.0)
    return pos


def wrap_offsets_to_start(
    current_arm_rad: Sequence[float],
    joint_offsets: Sequence[float],
    joint_signs: Sequence[int],
    start_joints: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """The boot-time 2*pi wrap from ``DynamixelRobot.__init__`` (start_joints branch).

    ``current_arm_rad`` is the *calibrated* (pre-wrap) arm reading at boot.  Returns
    ``(new_offsets, k)`` where ``k[i]`` is the integer number of full turns folded
    into offset ``i``.  Only 2*pi multiples ever change, so a pi/2 miscalibration
    survives this — which is exactly what the reference check relies on.
    """
    cur = np.asarray(current_arm_rad, dtype=float)
    start = np.asarray(start_joints, dtype=float)
    if cur.shape != start.shape:
        raise ValueError(f"current {cur.shape} vs start_joints {start.shape}")
    k = np.round((-start + cur) / (2 * math.pi))
    new_offsets = 2 * math.pi * k * np.asarray(joint_signs, dtype=float) + np.asarray(joint_offsets, dtype=float)
    return new_offsets, k.astype(int)


def wrap_pi(x: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    r = (x + math.pi) % (2 * math.pi) - math.pi
    return math.pi if r == -math.pi else float(r)


# --------------------------------------------------------------------------- inventory


@dataclass(frozen=True)
class MotorInfo:
    dxl_id: int
    model: int
    firmware: Optional[int]
    torque_enable: Optional[int]
    error: int = 0

    @property
    def model_name(self) -> str:
        return MODEL_NAMES.get(self.model, "unknown")


def classify_inventory(
    responding: Mapping[int, MotorInfo],
    cfg: LeaderConfig,
    fingerprint: Mapping[int, int] = DOCUMENTED_FINGERPRINT,
) -> Tuple[str, List[str]]:
    """``(verdict, notes)`` from which IDs answered and what they are.

    7 motors on IDs 1..7 -> UR-style (6 joints + trigger); 8 on IDs 1..8 ->
    Franka-style (7 + trigger).  ``notes`` lists everything that disagrees with
    the config or with the documented model fingerprint of the calibrated unit.
    """
    ids = sorted(responding)
    notes: List[str] = []
    n = len(ids)
    if n == 0:
        verdict = "응답 모터 0개 — 전원(5 V)/케이블/baud/포트를 먼저 의심"
    elif ids == list(range(1, 8)):
        verdict = "UR용 GELLO 형상 (6관절 ID1..6 + 그리퍼 ID7 = 7개)"
    elif ids == list(range(1, 9)):
        verdict = "Franka용 GELLO 형상 (7관절 ID1..7 + 그리퍼 ID8 = 8개) — UR config와 맞지 않음"
    else:
        verdict = f"알 수 없는 형상 — 응답 ID {ids} ({n}개)"

    missing = [i for i in cfg.all_ids if i not in responding]
    if missing:
        notes.append(f"config가 요구하는 ID 중 무응답: {missing}")
    extra = [i for i in ids if i not in cfg.all_ids]
    if extra:
        notes.append(f"config에 없는 ID가 응답: {extra}")
    for dxl_id, expected_model in fingerprint.items():
        info = responding.get(dxl_id)
        if info is not None and info.model != expected_model:
            notes.append(
                f"ID {dxl_id}: model {info.model} ({info.model_name}) — 문서 지문은 "
                f"{expected_model} ({MODEL_NAMES.get(expected_model, '?')})"
            )
    for dxl_id in ids:
        te = responding[dxl_id].torque_enable
        if te not in (None, 0):
            notes.append(f"ID {dxl_id}: torque_enable={te} — 이 리포의 어떤 경로도 토크를 켜지 않는다. 즉시 조사")
    return verdict, notes


# --------------------------------------------------------------------------- reference diagnosis


@dataclass(frozen=True)
class JointDiagnosis:
    index: int
    calibrated: float
    reference: float
    error: float  # wrapped to (-pi, pi]
    ok: bool
    kind: str  # "ok" | "offset" | "sign" | "other"
    hint: str
    suggested_offset: Optional[float] = None


def diagnose_reference(
    calibrated_arm: Sequence[float],
    reference: Sequence[float],
    joint_offsets: Sequence[float],
    joint_signs: Sequence[int],
    tol: float = 0.15,
) -> List[JointDiagnosis]:
    """Per-joint comparison of the calibrated reading against a known physical pose.

    ``joint_offsets`` are the offsets actually used to produce ``calibrated_arm``
    (i.e. after the boot wrap) so the suggested offset is directly usable.
    """
    out: List[JointDiagnosis] = []
    for i, (c, r) in enumerate(zip(calibrated_arm, reference)):
        err = wrap_pi(float(c) - float(r))
        sign = int(joint_signs[i])
        off = float(joint_offsets[i])
        if abs(err) <= tol:
            out.append(JointDiagnosis(i, float(c), float(r), err, True, "ok", "OK"))
            continue
        k = int(round(err / (math.pi / 2)))
        residual = err - k * (math.pi / 2)
        # calib = sign * (raw - off)  =>  to move calib by -err, off += sign * err
        suggested = off + sign * err
        if k != 0 and abs(residual) <= tol:
            out.append(
                JointDiagnosis(
                    i, float(c), float(r), err, False, "offset",
                    f"pi/2 x {k:+d} 만큼 어긋남 -> joint_offsets[{i}] 문제 "
                    f"(현재 {off:.3f}, 제안 {suggested:.3f}; 정본 도구는 scripts/gello_get_offset.py)",
                    suggested,
                )
            )
        elif abs(r) > tol and abs(wrap_pi(float(c) + float(r))) <= tol:
            out.append(
                JointDiagnosis(
                    i, float(c), float(r), err, False, "sign",
                    f"기준의 부호 반대 ({c:+.3f} vs {r:+.3f}) -> joint_signs[{i}] 문제 (현재 {sign:+d})",
                )
            )
        else:
            out.append(
                JointDiagnosis(
                    i, float(c), float(r), err, False, "other",
                    "pi/2 배수도 부호 반전도 아님 -> 다른 개체이거나 재조립/기어 슬립. "
                    "기준 자세를 정확히 맞췄는지도 확인",
                )
            )
    return out


def reference_verdict(diags: Sequence[JointDiagnosis]) -> Tuple[bool, str]:
    bad = [d for d in diags if not d.ok]
    if not bad:
        return True, "PASS — 모든 관절이 허용 오차 안"
    parts = ", ".join(f"J{d.index + 1} {d.error:+.3f} rad ({d.kind})" for d in bad)
    return False, f"FAIL — {len(bad)}개 관절 초과: {parts}"


# --------------------------------------------------------------------------- bus I/O (ping + read only)


def ping_inventory(port_handler, packet_handler, ids: Iterable[int]) -> Dict[int, MotorInfo]:
    """Ping each ID; for responders read firmware version and torque_enable. No writes."""
    from dynamixel_sdk.robotis_def import COMM_SUCCESS

    found: Dict[int, MotorInfo] = {}
    for dxl_id in ids:
        model, comm, err = packet_handler.ping(port_handler, dxl_id)
        if comm != COMM_SUCCESS:
            continue
        fw, c2, _ = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_FIRMWARE_VERSION)
        te, c3, _ = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE)
        found[dxl_id] = MotorInfo(
            dxl_id=dxl_id,
            model=int(model),
            firmware=int(fw) if c2 == COMM_SUCCESS else None,
            torque_enable=int(te) if c3 == COMM_SUCCESS else None,
            error=int(err),
        )
    return found


def read_positions(port_handler, packet_handler, ids: Sequence[int]) -> Dict[int, int]:
    """Present-position ticks (signed) per ID. Raises on a comm failure. No writes."""
    from dynamixel_sdk.robotis_def import COMM_SUCCESS

    out: Dict[int, int] = {}
    for dxl_id in ids:
        ticks, comm, err = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_PRESENT_POSITION)
        if comm != COMM_SUCCESS:
            raise RuntimeError(
                f"ID {dxl_id} present-position read failed: {packet_handler.getTxRxResult(comm)}"
            )
        if err != 0:
            raise RuntimeError(f"ID {dxl_id} packet error: {packet_handler.getRxPacketError(err)}")
        if ticks > 0x7FFFFFFF:
            ticks -= 0x100000000
        out[dxl_id] = int(ticks)
    return out
