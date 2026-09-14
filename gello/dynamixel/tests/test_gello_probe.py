"""Hardware-free tests for ``gello.dynamixel.probe`` (the read-only GELLO probe).

Run with the system interpreter (the anyio plugin trap applies here too):

    cd /home/laptop3/gello_software
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider gello/dynamixel/tests
"""
import math
import os

import numpy as np
import pytest

from gello.dynamixel import probe as P

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

UR_START = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0)


def _cfg(**over) -> P.LeaderConfig:
    base = dict(
        source="ros",
        path="x.yaml",
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0",
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(0.0, 1.571, 4.712, 4.712, 4.712, 3.142),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 210.649609375, 168.849609375),
        start_joints=UR_START + (0.0,),
    )
    base.update(over)
    return P.LeaderConfig(**base)


class _StubDriver:
    """Stands in for FakeDynamixelDriver: returns a fixed raw reading (rad)."""

    def __init__(self, raw):
        self._raw = np.asarray(raw, dtype=float)

    def get_joints(self):
        return self._raw.copy()

    def set_torque_mode(self, enable):  # DynamixelRobot never calls this for real=False
        raise AssertionError("no writes expected")


def _robot_with_raw(monkeypatch, raw, cfg: P.LeaderConfig, start_joints=None):
    """A real ``DynamixelRobot`` whose driver reads ``raw`` — the reference implementation."""
    import gello.dynamixel.driver as drv
    from gello.robots.dynamixel import DynamixelRobot

    monkeypatch.setattr(drv, "FakeDynamixelDriver", lambda ids: _StubDriver(raw))
    return DynamixelRobot(
        joint_ids=list(cfg.joint_ids),
        joint_offsets=list(cfg.joint_offsets),
        joint_signs=list(cfg.joint_signs),
        real=False,
        gripper_config=cfg.gripper_config,
        start_joints=None if start_joints is None else np.asarray(start_joints),
    )


# --------------------------------------------------------------------------- config parsing


def test_parse_ros_and_mujoco_documents():
    ros_doc = {
        "gello_publisher": {
            "ros__parameters": {
                "port": "/dev/serial/by-id/X",
                "joint_ids": [1, 2, 3, 4, 5, 6],
                "joint_offsets": [0.0, 1.571, 4.712, 4.712, 4.712, 3.142],
                "joint_signs": [1, 1, -1, 1, 1, 1],
                "gripper_config": [7.0, 210.6, 168.8],
                "start_joints": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0],
            }
        }
    }
    mj_doc = {
        "agent": {
            "port": "/dev/serial/by-id/X",
            "dynamixel_config": {
                "joint_ids": [1, 2, 3, 4, 5, 6],
                "joint_offsets": [0.0, 1.571, 4.712, 4.712, 4.712, 3.142],
                "joint_signs": [1, 1, -1, 1, 1, 1],
                "gripper_config": [7, 210.6, 168.8],
            },
            "start_joints": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0],
        }
    }
    a = P.parse_ros_config(ros_doc)
    b = P.parse_mujoco_config(mj_doc)
    assert a.gripper_config == (7, 210.6, 168.8)  # 7.0 round-trips to int like the node does
    assert a.gripper_id == 7 and a.all_ids == (1, 2, 3, 4, 5, 6, 7)
    assert a.start_arm_joints == tuple(UR_START)
    assert P.compare_configs(a, b) == []


def test_load_real_repo_configs_and_known_divergence():
    ros = P.load_config("ros", REPO_ROOT)
    mj = P.load_config("mujoco", REPO_ROOT)
    assert ros.port == mj.port and "FTBEO6QK" in ros.port
    assert ros.joint_ids == mj.joint_ids == (1, 2, 3, 4, 5, 6)
    assert ros.gripper_id == mj.gripper_id == 7
    # The two files are known to disagree on J1's offset (0.0 vs 3.142); the probe must say so.
    diffs = P.compare_configs(ros, mj)
    assert any(d.startswith("joint_offsets[0]") and "pi/2" in d for d in diffs), diffs


def test_compare_configs_reports_each_field():
    a = _cfg()
    b = _cfg(joint_signs=(1, 1, 1, 1, 1, 1), gripper_config=(7, 200.0, 168.849609375), port="/dev/ttyUSB9")
    diffs = P.compare_configs(a, b)
    assert any(d.startswith("port:") for d in diffs)
    assert any(d.startswith("joint_signs:") for d in diffs)
    assert any(d.startswith("gripper_config:") for d in diffs)
    assert not any(d.startswith("joint_offsets") for d in diffs)


# --------------------------------------------------------------------------- serial ports


def test_match_port_by_id_link_and_raw_target():
    entries = [("usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0", "/dev/ttyUSB0")]
    assert P.match_port(_cfg().port, entries) == (True, "/dev/ttyUSB0")
    assert P.match_port("/dev/ttyUSB0", entries) == (True, "/dev/ttyUSB0")
    assert P.match_port("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0", entries) == (False, None)
    assert P.match_port("/dev/ttyUSB1", entries) == (False, None)
    assert P.match_port(_cfg().port, []) == (False, None)


def test_list_serial_by_id_missing_dir(tmp_path):
    assert P.list_serial_by_id(str(tmp_path / "nope")) == []


# --------------------------------------------------------------------------- calibration math


def test_ticks_to_rad_matches_driver_formula():
    assert P.ticks_to_rad(2048) == pytest.approx(math.pi)
    assert P.ticks_to_rad(0) == 0.0
    assert P.ticks_to_rad(4096) == pytest.approx(2 * math.pi)
    # 32-bit two's complement: 0xFFFFFFFF is -1 tick
    assert P.ticks_to_rad(0xFFFFFFFF) == pytest.approx(-math.pi / 2048)


def test_calibrate_matches_dynamixel_robot_get_joint_state(monkeypatch):
    cfg = _cfg()
    raw = [0.3, 1.9, 4.0, 5.0, 4.5, 3.0, math.radians(190.0)]
    robot = _robot_with_raw(monkeypatch, raw, cfg)
    expected = robot.get_joint_state()  # first call: no EMA yet
    got = P.calibrate(raw, cfg.joint_offsets, cfg.joint_signs, cfg.gripper_config[1:])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert 0.0 < got[-1] < 1.0


@pytest.mark.parametrize("gdeg, expect", [(215.0, 0.0), (210.649609375, 0.0), (168.849609375, 1.0), (160.0, 1.0)])
def test_calibrate_gripper_is_clamped(gdeg, expect):
    cfg = _cfg()
    raw = [0.0] * 6 + [math.radians(gdeg)]
    got = P.calibrate(raw, cfg.joint_offsets, cfg.joint_signs, cfg.gripper_config[1:])
    assert got[-1] == pytest.approx(expect)


def test_calibrate_without_gripper_and_length_check():
    cfg = _cfg(gripper_config=None)
    got = P.calibrate([1.0] * 6, cfg.joint_offsets, cfg.joint_signs, None)
    assert got.shape == (6,)
    with pytest.raises(ValueError):
        P.calibrate([1.0] * 7, cfg.joint_offsets, cfg.joint_signs, None)


def test_wrap_offsets_matches_dynamixel_robot_start_joints_branch(monkeypatch):
    cfg = _cfg()
    # Raw such that, before wrapping, J1 reads +2pi off, J3 (sign -1) reads -2pi off, others exact.
    calib_target = np.array(UR_START, dtype=float)
    offs = np.array(cfg.joint_offsets)
    signs = np.array(cfg.joint_signs, dtype=float)
    raw_arm = calib_target * signs + offs  # calib = (raw - off) * sign  =>  raw = calib*sign + off
    raw_arm[0] += 2 * math.pi
    raw_arm[2] += 2 * math.pi  # sign -1 => calib shifts by -2pi
    raw = list(raw_arm) + [math.radians(200.0)]

    robot = _robot_with_raw(monkeypatch, raw, cfg, start_joints=UR_START + (0.0,))
    pre = P.calibrate(raw, cfg.joint_offsets, cfg.joint_signs, cfg.gripper_config[1:])
    new_off, k = P.wrap_offsets_to_start(pre[:6], cfg.joint_offsets, cfg.joint_signs, UR_START)
    np.testing.assert_allclose(new_off, robot._joint_offsets[:6], atol=1e-12)
    assert list(k) == [1, 0, -1, 0, 0, 0]
    post = P.calibrate(raw, new_off, cfg.joint_signs, cfg.gripper_config[1:])
    np.testing.assert_allclose(post[:6], calib_target, atol=1e-9)


def test_wrap_does_not_hide_a_quarter_turn_error():
    cfg = _cfg()
    cur = np.array(UR_START) + np.array([math.pi / 2, 0, 0, 0, 0, 0])
    new_off, k = P.wrap_offsets_to_start(cur, cfg.joint_offsets, cfg.joint_signs, UR_START)
    assert list(k) == [0] * 6
    np.testing.assert_allclose(new_off, cfg.joint_offsets)


def test_wrap_pi():
    assert P.wrap_pi(0.0) == 0.0
    assert P.wrap_pi(2 * math.pi + 0.1) == pytest.approx(0.1)
    assert P.wrap_pi(-2 * math.pi - 0.1) == pytest.approx(-0.1)
    assert P.wrap_pi(math.pi) == pytest.approx(math.pi)
    assert P.wrap_pi(-math.pi) == pytest.approx(math.pi)


# --------------------------------------------------------------------------- inventory


def _inv(ids, model_of=lambda i: 1190 if i == 7 else 1200, torque=0):
    return {i: P.MotorInfo(i, model_of(i), 53, torque) for i in ids}


def test_classify_inventory_ur_shape_and_fingerprint():
    verdict, notes = P.classify_inventory(_inv(range(1, 8)), _cfg())
    assert verdict.startswith("UR용")
    assert notes == []


def test_classify_inventory_franka_shape():
    verdict, notes = P.classify_inventory(_inv(range(1, 9), model_of=lambda i: 1190 if i == 8 else 1200), _cfg())
    assert verdict.startswith("Franka용")
    assert any("config에 없는 ID" in n and "[8]" in n for n in notes)
    assert any("ID 7: model 1200" in n for n in notes)  # fingerprint says ID 7 is the 1190 trigger


def test_classify_inventory_missing_and_torque():
    verdict, notes = P.classify_inventory(_inv([5, 7], torque=1), _cfg())
    assert "알 수 없는 형상" in verdict
    assert any("무응답: [1, 2, 3, 4, 6]" in n for n in notes)
    assert sum("torque_enable=1" in n for n in notes) == 2


def test_classify_inventory_empty():
    verdict, notes = P.classify_inventory({}, _cfg())
    assert verdict.startswith("응답 모터 0개")
    assert any("무응답" in n for n in notes)


def test_model_names():
    assert P.MotorInfo(1, 1200, None, None).model_name == "XL330-M288-T"
    assert P.MotorInfo(7, 1190, None, None).model_name == "XL330-M077-T"
    assert P.MotorInfo(9, 4242, None, None).model_name == "unknown"


# --------------------------------------------------------------------------- reference diagnosis


def test_diagnose_reference_pass():
    cfg = _cfg()
    cal = np.array(UR_START) + 0.05
    diags = P.diagnose_reference(cal, UR_START, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    assert all(d.ok for d in diags)
    ok, text = P.reference_verdict(diags)
    assert ok and text.startswith("PASS")


def test_diagnose_reference_offset_error_suggests_fixed_offset():
    cfg = _cfg()
    cal = np.array(UR_START)
    cal[2] += math.pi / 2  # J3 (sign -1) off by a quarter turn
    diags = P.diagnose_reference(cal, UR_START, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    d = diags[2]
    assert not d.ok and d.kind == "offset"
    assert "joint_offsets[2]" in d.hint and "gello_get_offset.py" in d.hint
    # Applying the suggested offset must remove the error: calib = sign*(raw - off).
    raw3 = cal[2] * cfg.joint_signs[2] + cfg.joint_offsets[2]
    assert cfg.joint_signs[2] * (raw3 - d.suggested_offset) == pytest.approx(UR_START[2])
    ok, text = P.reference_verdict(diags)
    assert not ok and "J3" in text and "offset" in text


def test_diagnose_reference_sign_error():
    cfg = _cfg()
    cal = np.array(UR_START)
    cal[1] = -cal[1]  # J2 reads +1.57 instead of -1.57
    diags = P.diagnose_reference(cal, UR_START, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    # |err| = 3.14 = 2 x pi/2 exactly: the offset rule would also match, so a pure pi
    # flip on a +/-pi/2 joint is ambiguous by construction; we report it as offset
    # (k=+2) — either config edit fixes it. Use a non-quarter-turn sign case for "sign".
    assert diags[1].kind in ("offset", "sign")
    cal = np.array([0.0, -1.0, 1.0, -1.0, -1.0, 0.0])
    ref = [0.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    diags = P.diagnose_reference(cal, ref, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    assert diags[2].kind == "sign" and "joint_signs[2]" in diags[2].hint


def test_diagnose_reference_other():
    cfg = _cfg()
    cal = np.array(UR_START)
    cal[4] += 0.6  # neither a pi/2 multiple nor a sign flip
    diags = P.diagnose_reference(cal, UR_START, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    assert diags[4].kind == "other" and "다른 개체" in diags[4].hint


def test_diagnose_reference_error_is_wrapped():
    cfg = _cfg()
    cal = np.array(UR_START)
    cal[0] += 2 * math.pi  # a full turn is not an error
    diags = P.diagnose_reference(cal, UR_START, cfg.joint_offsets, cfg.joint_signs, tol=0.15)
    assert diags[0].ok and diags[0].error == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- bus I/O against a fake packet handler


class _FakePacket:
    """Answers pings/reads for a scripted bus. Any write attribute access is an error."""

    COMM_SUCCESS = 0

    def __init__(self, motors, positions):
        self.motors = motors  # id -> model
        self.positions = positions  # id -> ticks (unsigned 32)

    def ping(self, port, dxl_id):
        if dxl_id in self.motors:
            return self.motors[dxl_id], 0, 0
        return 0, -3001, 0  # COMM_RX_TIMEOUT

    def read1ByteTxRx(self, port, dxl_id, addr):
        if addr == P.ADDR_FIRMWARE_VERSION:
            return 53, 0, 0
        if addr == P.ADDR_TORQUE_ENABLE:
            return 0, 0, 0
        raise AssertionError(addr)

    def read4ByteTxRx(self, port, dxl_id, addr):
        assert addr == P.ADDR_PRESENT_POSITION
        if dxl_id in self.positions:
            return self.positions[dxl_id], 0, 0
        return 0, -3001, 0

    def getTxRxResult(self, comm):
        return f"comm {comm}"

    def getRxPacketError(self, err):
        return f"err {err}"

    def __getattr__(self, name):
        if "write" in name.lower() or "sync" in name.lower():
            raise AssertionError(f"probe must never call {name}")
        raise AttributeError(name)


def test_ping_inventory_and_read_positions_use_only_ping_and_reads():
    pk = _FakePacket({i: (1190 if i == 7 else 1200) for i in range(1, 8)}, {1: 2048, 7: 0xFFFFFFFF})
    found = P.ping_inventory(None, pk, range(1, 13))
    assert sorted(found) == list(range(1, 8))
    assert found[7].model == 1190 and found[7].firmware == 53 and found[7].torque_enable == 0
    pos = P.read_positions(None, pk, [1, 7])
    assert pos == {1: 2048, 7: -1}
    with pytest.raises(RuntimeError):
        P.read_positions(None, pk, [1, 2])
