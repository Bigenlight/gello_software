#!/usr/bin/env python3
"""Robotiq 2F-85 MODBUS RTU driver over the UR RS485 tool-communication forwarder.

The UR e-Series exposes the tool-flange RS485 line (to which the 2F-85 is wired) as a
raw TCP passthrough on the controller at ``ROBOT_IP:54321`` (the "RS485 / tool
communication" URCap must be installed on PolyScope). This driver speaks Modbus RTU
(with CRC) straight over that TCP socket — so it needs neither ``socat``/``/tmp/ttyUR``
nor libmodbus nor any C++ build. Optionally it can also open a real serial device
(e.g. a socat-created ``/tmp/ttyUR``) if you prefer that path.

Only ONE client may own the :54321 forwarder at a time. Close the connection cleanly
on shutdown (``close()``); a hard-killed connection leaves the controller holding a
dead socket that starves reconnects until it times out.

IMPORTANT: the gripper is only powered when the robot is POWERED ON (tool voltage is
off in POWER_OFF). A powered-off robot => no Modbus response, which is not a bug here.

Register layout (Robotiq 2F-85/2F-140 manual, slave id 0x09):
  OUT (write FC16 @0x03E8, 3 regs / 6 bytes): [action, 0, 0, rPR, rSP, rFR]
      action bits: bit0=rACT, bit3=rGTO, bit4=rATR, bit5=rARD
  IN  (read  FC03 @0x07D0, 3 regs / 6 bytes): [status, 0, gFLT, gPR, gPO, gCU]
      status bits: bit0=gACT, bit3=gGTO, bit4-5=gSTA, bit6-7=gOBJ
  Position: 0 = OPEN, 255 = CLOSED.

All bus I/O raises ``GripperError`` on any transport OR protocol failure, so callers
can funnel every failure into a single reconnect path.
"""
import socket
import threading
import time

SLAVE = 0x09
OUT_ADDR = 0x03E8
IN_ADDR = 0x07D0


def crc16(data: bytes) -> bytes:
    """Modbus RTU CRC-16, returned low-byte-first (as appended to the frame)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


class GripperError(IOError):
    pass


class Robotiq2F85:
    """Modbus-RTU-over-TCP (default) or -over-serial driver for a Robotiq 2F-85."""

    def __init__(self, host="192.168.10.11", port=54321, serial_port=None,
                 timeout=1.5):
        self._host = host
        self._port = port
        self._serial_port = serial_port  # if set, use pyserial instead of TCP
        self._timeout = timeout
        self._sock = None
        self._ser = None
        self._lock = threading.Lock()  # serialize all bus transactions

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def connect(self):
        with self._lock:
            self._connect_locked()

    def _connect_locked(self):
        self._close_locked()
        try:
            if self._serial_port:
                import serial  # lazy import; only needed for the serial path
                self._ser = serial.Serial(self._serial_port, baudrate=115200,
                                         bytesize=8, parity="N", stopbits=1,
                                         timeout=self._timeout)
            else:
                self._sock = socket.create_connection((self._host, self._port),
                                                     timeout=5)
                self._sock.settimeout(self._timeout)
        except OSError as exc:
            raise GripperError(f"connect failed: {exc}") from exc
        time.sleep(0.2)

    def close(self):
        with self._lock:
            self._close_locked()

    def _close_locked(self):
        for obj in (self._sock, self._ser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._sock = None
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._sock is not None or self._ser is not None

    def _drain_tcp(self, sock):
        """Discard any stale bytes left by a previously timed-out transaction."""
        sock.setblocking(False)
        try:
            while True:
                if not sock.recv(4096):
                    break  # peer closed
        except (BlockingIOError, socket.timeout, OSError):
            pass
        finally:
            sock.settimeout(self._timeout)

    def _rw(self, frame: bytes, expect: int) -> bytes:
        """One raw transaction. Raises GripperError on any transport failure."""
        payload = frame + crc16(frame)
        try:
            if self._ser is not None:
                self._ser.reset_input_buffer()
                self._ser.write(payload)
                self._ser.flush()
                time.sleep(0.03)
                return self._ser.read(expect)
            sock = self._sock
            if sock is None:
                raise GripperError("socket not connected")
            self._drain_tcp(sock)
            sock.sendall(payload)
            buf = b""
            t0 = time.time()
            while len(buf) < expect and time.time() - t0 < self._timeout:
                try:
                    chunk = sock.recv(expect - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    break
            return buf
        except GripperError:
            raise
        except OSError as exc:
            raise GripperError(f"transport error: {exc}") from exc

    @staticmethod
    def _frame_ok(resp: bytes, slave, func, length) -> bool:
        return (len(resp) == length and resp[0] == slave and resp[1] == func
                and crc16(resp[:-2]) == resp[-2:])

    # ------------------------------------------------------------------ #
    # Modbus transactions (retried a few times; the first RTU frame after a
    # fresh connection, or a frame during auto-cal, is occasionally dropped)
    # ------------------------------------------------------------------ #
    def read_status(self, retries=5) -> dict:
        req = bytes([SLAVE, 0x03, IN_ADDR >> 8, IN_ADDR & 0xFF, 0x00, 0x03])
        with self._lock:
            resp = b""
            for _ in range(retries):
                resp = self._rw(req, 11)
                if self._frame_ok(resp, SLAVE, 0x03, 11) and resp[2] == 0x06:
                    break
                time.sleep(0.05)
            else:
                raise GripperError(
                    f"no/invalid status response ({resp.hex(' ') if resp else 'none'}). "
                    "Is the robot POWERED ON (tool voltage) and the RS485 URCap installed?"
                )
        d = resp[3:9]  # wire order: [status, _, gFLT, gPR, gPO, gCU]
        s = d[0]
        return {
            "gACT": s & 1, "gGTO": (s >> 3) & 1, "gSTA": (s >> 4) & 3,
            "gOBJ": (s >> 6) & 3, "gFLT": d[2], "gPR": d[3], "gPO": d[4], "gCU": d[5],
        }

    def _write_out(self, action, pos=0, speed=0, force=0, retries=3):
        pos = max(0, min(255, int(pos)))
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        data = bytes([action & 0xFF, 0x00, 0x00, pos, speed, force])
        req = bytes([SLAVE, 0x10, OUT_ADDR >> 8, OUT_ADDR & 0xFF, 0x00, 0x03, 0x06]) + data
        with self._lock:
            resp = b""
            for _ in range(retries):
                resp = self._rw(req, 8)
                if self._frame_ok(resp, SLAVE, 0x10, 8):
                    return
                time.sleep(0.05)
        raise GripperError(f"write failed ({resp.hex(' ') if resp else 'none'})")

    # ------------------------------------------------------------------ #
    # High-level
    # ------------------------------------------------------------------ #
    def activate(self, timeout=8.0):
        """rACT 0->1, wait for gSTA==3. Clears prior faults (e.g. 0x09 comm-loss).

        Activation triggers the gripper's internal auto-calibration open/close sweep.
        """
        self._write_out(0x00)          # rACT=0 (reset / clear fault)
        time.sleep(0.5)
        self._write_out(0x01)          # rACT=1 (activate)
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.read_status()
            if st["gACT"] == 1 and st["gSTA"] == 3:
                return st
            time.sleep(0.1)
        raise GripperError(f"activation did not complete: {self.read_status()}")

    def is_active(self) -> bool:
        st = self.read_status()
        return st["gACT"] == 1 and st["gSTA"] == 3

    def move(self, pos, speed=150, force=50):
        """Command a position (0=open .. 255=closed). action = rACT|rGTO = 0x09."""
        self._write_out(0x09, pos, speed, force)

    def stop(self):
        """Clear rGTO so the gripper holds its current position (halts motion)."""
        self._write_out(0x01)  # rACT=1, rGTO=0

    def move_and_wait(self, pos, speed=150, force=50, timeout=3.0):
        """Move then poll until motion stops (object contact or target reached)."""
        self.move(pos, speed, force)
        t0 = time.time()
        st = self.read_status()
        while time.time() - t0 < timeout:
            st = self.read_status()
            if st["gOBJ"] != 0:  # 1/2 = stopped on object, 3 = reached target
                break
            time.sleep(0.02)
        return st

    def get_position(self) -> int:
        """Current position 0=open .. 255=closed."""
        return self.read_status()["gPO"]
