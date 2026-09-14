#!/usr/bin/env python3
"""GELLO 리더팔 개체 확인 — 읽기 전용 진단.

"지금 꽂혀 있는 GELLO가 우리가 캘리브레이션해 둔 그 개체(UR용 6-DOF, FTDI FTBEO6QK)가
맞는가"에 실행 한 번으로 답한다.  버스에는 **ping과 read만** 보낸다 — 토크는 절대 켜지
않고, 어떤 레지스터에도 쓰지 않으며, ``gello.dynamixel.driver.DynamixelDriver``도 쓰지
않는다(그 드라이버는 초기화에서 포트 점유 프로세스를 ``fuser -k``로 죽이고 실패 시 조용히
가짜 드라이버로 대체한다 — 진단 도구에 어울리지 않는다).

    python3 scripts/gello_probe.py                 # 포트 · 모터 인벤토리 · 현재 자세
    python3 scripts/gello_probe.py --reference     # + 기준 자세(config start_joints) 대조
    python3 scripts/gello_probe.py --watch         # 2 Hz로 캘리브레이션 관절각 스트리밍
    python3 scripts/gello_probe.py --config mujoco # configs/rwh_ur.yaml 기준으로

시스템 python3(3.10)에서 돈다. 순수 로직은 ``gello/dynamixel/probe.py``에 있고 거기에
테스트가 붙어 있다(``gello/dynamixel/tests/test_gello_probe.py``).
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import yaml  # noqa: F401  (PyYAML)
except ImportError:  # pragma: no cover - environment check
    sys.exit("PyYAML이 없다: `python3 -m pip install --user pyyaml`")
try:
    from dynamixel_sdk import PacketHandler, PortHandler
except ImportError:  # pragma: no cover - environment check
    sys.exit("dynamixel_sdk가 없다: `python3 -c 'import dynamixel_sdk'`로 확인 (~/.local에 있어야 한다)")

import numpy as np

from gello.dynamixel import probe as P

DEFAULT_BAUD = 57600  # driver.py:166
PROTOCOL = 2.0  # driver.py:251


def _port_users(path: str) -> List[str]:
    """Other processes holding the serial port (never kills anything)."""
    real = os.path.realpath(path)
    lines: List[str] = []
    if shutil.which("fuser"):
        r = subprocess.run(["fuser", "-v", real], capture_output=True, text=True)
        # fuser prints to stderr; rc 0 means at least one user
        if r.returncode == 0:
            lines += [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    if not lines and shutil.which("lsof"):
        r = subprocess.run(["lsof", real], capture_output=True, text=True)
        if r.returncode == 0:
            body = [ln for ln in r.stdout.strip().splitlines() if ln and not ln.startswith("COMMAND")]
            lines += body
    return lines


def _fmt_list(xs: Sequence[float], fmt: str = "{:+.3f}") -> str:
    return "[" + ", ".join(fmt.format(x) for x in xs) + "]"


def section(title: str) -> None:
    print(f"\n[{title}]")


# --------------------------------------------------------------------------- sections


def report_ports(configs: Dict[str, P.LeaderConfig], chosen: str, port_override: Optional[str], allow_busy: bool = False) -> str:
    section("1. 시리얼 포트")
    entries = P.list_serial_by_id()
    if not entries:
        print("  /dev/serial/by-id/ 비어 있음 — USB 어댑터가 안 보인다")
    for name, target in entries:
        print(f"  /dev/serial/by-id/{name} -> {target}")
    for src, cfg in configs.items():
        present, target = P.match_port(cfg.port, entries)
        mark = "✅" if present else "❌"
        label = "gello_publisher.port" if src == "ros" else "agent.port"
        tail = f" (-> {target})" if present else " (목록에 없음)"
        print(f"  {mark} {src:6s} {label} = {os.path.basename(cfg.port)}{tail}")
    port = port_override or configs[chosen].port
    if port_override:
        print(f"  --port 지정: {port}")
    users = _port_users(port)
    if users:
        print("  ❌ 포트를 다른 프로세스가 잡고 있다 (Dynamixel 버스는 단일 마스터 — 같이 쓰면 ping/read가 서로 깨진다):")
        for ln in users:
            print(f"     {ln}")
        print("  gello_publisher / run_hil_hardware.sh / 텔레옵 launch를 먼저 내리고 다시 실행하라. "
              "이 스크립트는 아무것도 죽이지 않는다.")
        if not allow_busy:
            sys.exit(3)
        print("  (--allow-busy: 계속 진행 — 결과를 믿지 말 것)")
    else:
        print("  포트 점유: 없음")
    return port


def open_bus(port: str, baud: int):
    ph = PortHandler(port)
    pk = PacketHandler(PROTOCOL)
    if not ph.openPort():
        users = _port_users(port)
        msg = f"포트를 열 수 없다: {port}"
        if users:
            msg += "\n  다른 프로세스가 잡고 있다 (gello_publisher / run_hil_hardware.sh 등을 먼저 내려라):\n    " + "\n    ".join(users)
        elif not os.path.exists(port):
            msg += "\n  경로가 없다 — USB 연결/시리얼(by-id 이름)을 확인"
        else:
            msg += "\n  권한? `id -nG | grep dialout`"
        sys.exit(msg)
    if not ph.setBaudRate(baud):
        ph.closePort()
        sys.exit(f"baudrate {baud} 설정 실패")
    return ph, pk


def report_inventory(ph, pk, cfg: P.LeaderConfig, max_id: int) -> Dict[int, P.MotorInfo]:
    section(f"2. 모터 인벤토리 (ID 1..{max_id} ping, baud {ph.getBaudRate()}, protocol {PROTOCOL})")
    found = P.ping_inventory(ph, pk, range(1, max_id + 1))
    if found:
        print("   ID  model  name           fw   torque_enable")
        for dxl_id in sorted(found):
            m = found[dxl_id]
            fw = "?" if m.firmware is None else str(m.firmware)
            te = "?" if m.torque_enable is None else str(m.torque_enable)
            flag = "" if m.torque_enable in (None, 0) else "  ⚠️"
            print(f"  {dxl_id:3d}  {m.model:5d}  {m.model_name:13s}  {fw:>3s}  {te}{flag}")
    verdict, notes = P.classify_inventory(found, cfg)
    print(f"  응답 {len(found)}개 → {verdict}")
    fp_ok = all(found.get(i) is not None and found[i].model == m for i, m in P.DOCUMENTED_FINGERPRINT.items())
    if fp_ok and sorted(found) == sorted(P.DOCUMENTED_FINGERPRINT):
        print("  ✅ 모델 지문이 문서(02_GELLO_LEADER.md §2: 1200×6 + 1190×1)와 일치")
    for n in notes:
        print(f"  ⚠️ {n}")
    return found


def read_calibrated(ph, pk, cfg: P.LeaderConfig, offsets: Sequence[float]):
    ticks = P.read_positions(ph, pk, cfg.all_ids)
    raw = [P.ticks_to_rad(ticks[i]) for i in cfg.all_ids]
    goc = None if cfg.gripper_config is None else (cfg.gripper_config[1], cfg.gripper_config[2])
    cal = P.calibrate(raw, offsets, cfg.joint_signs, goc)
    return ticks, raw, cal


def report_pose(ph, pk, cfg: P.LeaderConfig, use_wrap: bool):
    """Print the current pose; return (calibrated_arm, offsets_used)."""
    section("3. 현재 자세 (config offsets/signs 적용)")
    n = len(cfg.joint_ids)
    ticks, raw, cal0 = read_calibrated(ph, pk, cfg, cfg.joint_offsets)
    offsets = np.asarray(cfg.joint_offsets, dtype=float)
    k = np.zeros(n, dtype=int)
    if use_wrap and cfg.start_arm_joints is not None:
        offsets, k = P.wrap_offsets_to_start(cal0[:n], cfg.joint_offsets, cfg.joint_signs, cfg.start_arm_joints)
        _, _, cal = read_calibrated(ph, pk, cfg, offsets)
    else:
        cal = cal0
    print("  joint  id   tick     raw(rad)  offset   sign   calib(rad)  calib(deg)  2πwrap")
    for i, dxl_id in enumerate(cfg.joint_ids):
        wrap = f"{k[i]:+d}" if k[i] != 0 else "-"
        print(
            f"  J{i + 1:<4d} {dxl_id:3d}  {ticks[dxl_id]:6d}  {raw[i]:+9.3f}  {offsets[i]:+7.3f}  "
            f"{cfg.joint_signs[i]:+3d}   {cal[i]:+9.3f}  {math.degrees(cal[i]):+9.1f}   {wrap}"
        )
    if cfg.gripper_config is not None:
        gid, od, cd = cfg.gripper_config
        graw = raw[n]
        print(
            f"  grip  {gid:3d}  {ticks[gid]:6d}  {graw:+9.3f}  ({math.degrees(graw):.2f} deg; "
            f"open {od:.2f} / close {cd:.2f} deg)  -> {cal[n]:.3f}  (0=열림 1=닫힘)"
        )
    if use_wrap and cfg.start_arm_joints is not None:
        print(
            "  (2πwrap: gello_publisher 부팅 시 start_joints 기준 ±2π 정규화와 동일. "
            "π/2 오차는 이걸로 안 사라진다. --no-wrap으로 끌 수 있다)"
        )
    return cal[:n], offsets


def report_reference(cal_arm, offsets, cfg: P.LeaderConfig, reference: Sequence[float], tol: float) -> bool:
    section(f"4. 기준 자세 대조 (tol {tol:.3f} rad = {math.degrees(tol):.1f} deg)")
    print(f"  기준: {_fmt_list(reference)}")
    diags = P.diagnose_reference(cal_arm, reference, offsets, cfg.joint_signs, tol)
    print("  joint  calib(rad)  ref(rad)   err(rad)  err(deg)  판정")
    for d in diags:
        mark = "✅" if d.ok else "❌"
        print(
            f"  J{d.index + 1:<4d} {d.calibrated:+10.3f} {d.reference:+9.3f}  {d.error:+9.3f} "
            f"{math.degrees(d.error):+8.1f}  {mark} {d.hint}"
        )
    ok, text = P.reference_verdict(diags)
    print(f"  => {text}")
    if not ok:
        print(
            "  힌트: π/2 배수만큼 틀렸으면 offset, 부호가 반대면 sign, 그 외는 다른 개체 또는 재조립. "
            "재캘리브레이션은 scripts/gello_get_offset.py"
        )
    return ok


def watch(ph, pk, cfg: P.LeaderConfig, offsets: Sequence[float], hz: float) -> None:
    section(f"5. watch ({hz:g} Hz) — 관절을 하나씩 움직여 어느 J가 어느 방향으로 반응하는지 본다. Ctrl-C로 종료")
    n = len(cfg.joint_ids)
    period = 1.0 / hz
    _, _, first = read_calibrated(ph, pk, cfg, offsets)
    prev = np.array(first, dtype=float)
    t0 = time.monotonic()
    hdr = "   t(s)  " + "  ".join(f"J{i + 1:<6d}" for i in range(n)) + "  grip   | 직전 대비 최대 변화"
    print(hdr)
    try:
        while True:
            _, _, cal = read_calibrated(ph, pk, cfg, offsets)
            cal = np.array(cal, dtype=float)
            d = cal[:n] - prev[:n]
            j = int(np.argmax(np.abs(d)))
            move = f"J{j + 1} {d[j]:+.3f}" if abs(d[j]) > 0.01 else "-"
            since = cal[:n] - np.array(first[:n])
            js = int(np.argmax(np.abs(since)))
            tot = f"  (시작 대비 최대 J{js + 1} {since[js]:+.3f})" if abs(since[js]) > 0.01 else ""
            grip = f"{cal[n]:.2f}" if cfg.gripper_config is not None else "  -  "
            print(
                f"  {time.monotonic() - t0:6.1f}  " + "  ".join(f"{v:+7.3f}" for v in cal[:n])
                + f"  {grip}  | {move}{tot}",
                flush=True,
            )
            prev = cal
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  종료 (write 없음)")


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(P.CONFIG_SOURCES), default="ros",
                    help="캘리브레이션 값을 읽을 config (ros=ur7e_gello.yaml, mujoco=configs/rwh_ur.yaml)")
    ap.add_argument("--port", default=None, help="config의 포트 대신 이 포트를 연다")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--max-id", type=int, default=12, help="ping할 ID 상한 (1..N)")
    ap.add_argument("--reference", nargs="*", type=float, default=None, metavar="RAD",
                    help="기준 자세 대조. 값 없이 주면 config의 start_joints(UR 기본 [0,-1.571,1.571,-1.571,-1.571,0]), "
                         "또는 관절 수만큼 rad를 직접 준다")
    ap.add_argument("--tol", type=float, default=0.15, help="--reference 허용 오차 (rad)")
    ap.add_argument("--watch", action="store_true", help="캘리브레이션 관절각을 계속 출력 (Ctrl-C 종료)")
    ap.add_argument("--watch-hz", type=float, default=2.0)
    ap.add_argument("--allow-busy", action="store_true",
                    help="포트를 다른 프로세스가 잡고 있어도 진행 (진단용; 결과 신뢰 불가)")
    ap.add_argument("--no-wrap", action="store_true", help="부팅 시 start_joints 기준 ±2π 정규화를 적용하지 않는다")
    args = ap.parse_args(argv)

    print("GELLO probe — 읽기 전용 (ping/read만, write 0, 토크 절대 켜지 않음)")
    print(f"  repo: {REPO_ROOT}")

    configs: Dict[str, P.LeaderConfig] = {}
    for src in ("ros", "mujoco"):
        try:
            configs[src] = P.load_config(src, REPO_ROOT)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ⚠️ {src} config 로드 실패 ({P.CONFIG_SOURCES[src]}): {exc}")
    if args.config not in configs:
        return 2
    cfg = configs[args.config]
    print(f"  config: {args.config} = {os.path.relpath(cfg.path, REPO_ROOT)}")
    print(f"    joint_ids={list(cfg.joint_ids)} offsets={_fmt_list(cfg.joint_offsets, '{:.3f}')} "
          f"signs={list(cfg.joint_signs)} gripper={cfg.gripper_config}")

    section("6. config 중복 비교 (ros vs mujoco)")
    if len(configs) == 2:
        diffs = P.compare_configs(configs["ros"], configs["mujoco"])
        if diffs:
            print("  ⚠️ 두 config의 캘리브레이션 값이 다르다 — 값이 세 곳(ros yaml / mujoco yaml / "
                  "gello_publisher_node.py 기본값)에 중복 저장돼 있다는 알려진 함정. 어느 쪽이 정본인지 확인:")
            for d in diffs:
                print(f"     - {d}")
        else:
            print("  ✅ 두 config의 캘리브레이션 값 일치")
    else:
        print("  (한쪽 config를 못 읽어 비교 생략)")

    port = report_ports(configs, args.config, args.port, allow_busy=args.allow_busy)
    ph, pk = open_bus(port, args.baud)
    rc = 0
    try:
        found = report_inventory(ph, pk, cfg, args.max_id)
        missing = [i for i in cfg.all_ids if i not in found]
        if missing:
            print(f"\n  ❌ config ID {missing}가 응답하지 않아 자세 읽기를 건너뛴다")
            return 1
        cal_arm, offsets = report_pose(ph, pk, cfg, use_wrap=not args.no_wrap)

        if args.reference is not None:
            if len(args.reference) == 0:
                ref = cfg.start_arm_joints
                if ref is None:
                    ref = (0.0, -1.571, 1.571, -1.571, -1.571, 0.0)[: len(cfg.joint_ids)]
            else:
                ref = tuple(args.reference)
            if len(ref) != len(cfg.joint_ids):
                print(f"  ❌ --reference는 관절 수({len(cfg.joint_ids)})만큼 필요, {len(ref)}개 받음")
                return 2
            if not report_reference(cal_arm, offsets, cfg, ref, args.tol):
                rc = 1
        if args.watch:
            watch(ph, pk, cfg, offsets, args.watch_hz)
    finally:
        ph.closePort()
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  중단 (write 없음)")
        sys.exit(130)
