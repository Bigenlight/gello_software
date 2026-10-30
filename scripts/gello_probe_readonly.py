#!/usr/bin/env python3
"""READ-ONLY GELLO probe: compare a GELLO's raw motor angles against a calibration.

Reads Present Position / Torque Enable / Drive Mode from every Dynamixel on the
bus and NEVER writes anything (GELLO is a passive leader; torque must stay off).
Hold the arm at the calibration pose (UR [0, -90, 90, -90, -90, 0] deg) first.

Reports, per joint, the UR-equivalent angle under the given offsets/signs, the
branch-cut-safe error vs ``--start-joints``, and the pi/2-snapped offset the
measurement implies. If the snapped offsets equal ``--joint-offsets``, this arm
can share that calibration.

CAVEAT: the expected raw angles at the UR calibration pose are 0 / 180 deg,
where a reversed motor (360 - raw) reads the SAME value. A static read cannot
detect a flipped direction -- move each joint (e.g. in the MuJoCo sim) to check.

Example:
    python scripts/gello_probe_readonly.py \
        --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
"""

import argparse

import numpy as np
from dynamixel_sdk import PacketHandler, PortHandler

ADDR_DRIVE_MODE = 10
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
TICKS_PER_REV = 4096

# Defaults = ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml
DEFAULT_OFFSETS = [0.0, 1.571, 4.712, 4.712, 4.712, 3.142]
DEFAULT_SIGNS = [1, 1, -1, 1, 1, 1]
DEFAULT_START = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", required=True)
    p.add_argument("--baudrate", type=int, default=57600)
    p.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--joint-offsets", type=float, nargs=6, default=DEFAULT_OFFSETS)
    p.add_argument("--joint-signs", type=int, nargs=6, default=DEFAULT_SIGNS)
    p.add_argument("--start-joints", type=float, nargs=6, default=DEFAULT_START)
    args = p.parse_args()

    port, packet = PortHandler(args.port), PacketHandler(2.0)
    if not (port.openPort() and port.setBaudRate(args.baudrate)):
        raise SystemExit(f"could not open {args.port} (busy? e.g. DYNAMIXEL Wizard)")

    raw = {}
    try:
        for i in args.ids:
            model, rc, _ = packet.ping(port, i)
            if rc != 0:
                print(f"id{i}: no response")
                continue
            pos, _, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
            if pos & (1 << 31):
                pos -= 1 << 32
            torque, _, _ = packet.read1ByteTxRx(port, i, ADDR_TORQUE_ENABLE)
            drive, _, _ = packet.read1ByteTxRx(port, i, ADDR_DRIVE_MODE)
            raw[i] = pos
            deg = pos * 360 / TICKS_PER_REV
            print(
                f"id{i}: model={model} ticks={pos:6d} deg={deg % 360:7.2f} "
                f"torque={torque} drive_mode={drive} (reverse={drive & 1})"
            )
    finally:
        port.closePort()

    arm_ids = args.ids[:6]
    if not all(i in raw for i in arm_ids):
        return
    r = np.array([raw[i] for i in arm_ids]) * 2 * np.pi / TICKS_PER_REV
    offsets, signs = np.array(args.joint_offsets), np.array(args.joint_signs)
    start = np.array(args.start_joints)
    err = ((r - offsets) * signs - start + np.pi) % (2 * np.pi) - np.pi
    snapped = np.round((r - start * signs) / (np.pi / 2)) * (np.pi / 2) % (2 * np.pi)
    print("UR-equiv q (wrapped):", np.round(start + err, 3))
    print("err vs start (deg)  :", np.round(np.rad2deg(err), 1))
    print("pi/2-snapped offsets:", np.round(snapped, 3))
    print("matches calibration :", bool(np.allclose(snapped, offsets % (2 * np.pi), atol=0.01)))


if __name__ == "__main__":
    main()
