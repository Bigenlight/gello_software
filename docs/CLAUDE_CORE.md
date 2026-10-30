# CLAUDE_CORE.md — slim 기본 브랜치(`ur7e-jazzy`)의 CLAUDE.md 사본

> `research/hil-serl` 머지 시 보존한 사본이다. 링크는 `docs/` 기준으로 고쳤다. 원본은 `ur7e-jazzy`의 루트 `CLAUDE.md`.

문서 색인이다. 상세와 근거는 링크된 문서에 있다.

## 이 리포

GELLO 리더(수동, 읽기 전용 Dynamixel) → **UR7e + Robotiq 2F-85** 텔레옵, RealSense D435 ×2, 데이터 기록,
LeRobot 변환. 브랜치 `ur7e-jazzy` = Ubuntu 24.04 / ROS 2 Jazzy의 **GELLO 필수 기능만** 담은 slim 기본 브랜치.
사람용 입구는 [`README.md`](../README.md), 설치 정본은 [`docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`](ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md).

| 브랜치 | 내용 |
|---|---|
| `research/hil-serl` | HIL-SERL 온라인 RL (`serl_ur_infra/`, `run_hil_*`, `docs/testing/00,04~11`) |
| `research/policy-deploy` | ACT/Diffusion/FM/IFQL/SVF/DSRL 배포 + 실기 평가 (`gello_policy`) |
| `research/offline-rl` / `research/sim-collect` | 오프라인 IFQL 학습 / MuJoCo 수집 sim(`sim_collect/`) |
| `feat/gello-ur7e-jazzy-24.04` / `feat/gello-ur7e-humble-22.04` | 분리 전 전체 스냅샷 / 22.04 Humble 사용자용 |

## 구조

```
ros2_ur_ws/src/ur_gello_bringup/   publisher, gello_ur_bridge(joint/eef/joint_delta), move_to_start, 그리퍼, GUI
ros2_ur_ws/src/gello_recorder/     레코더 (gello_ur_recorder, gello_recorder_gui)
ros2_ur_ws/*.sh                    실행 래퍼 (ROS/conda PATH·GELLO_REPO_ROOT를 스스로 잡음)
ros2_ur_ws/setup_jazzy/            install_jazzy.sh, 이더넷, ToolComm 진단, EEF mock 스모크
gello/  scripts/  scripts/dataset/ GELLO 코어 / probe·offset / LeRobot 변환 3종
docs/ros2  docs/hardware  docs/testing(01~03)  docs/sim
```

## 실행 (`ros2_ur_ws/`에서)

```bash
./build_ur7e.sh                                          # ur_gello_bringup + gello_recorder
./run_ur7e_gello_sim.sh | ./run_ur7e_gello_mock.sh       # mock (실로봇 접속 불가)
HEADLESS=true ./run_ur7e_gello_real.sh [control_mode:=eef]   # 실기 — 팔이 움직인다
./run_eef_gui.sh | ./run_eef_collection_gui.sh           # EEF GUI / EEF 수집 통합 GUI
./run_ur7e_gripper.sh  ./launch_cameras.sh [--check]  CAMS=true ./run_recorder.sh
```

**실기를 움직이는 명령은 사용자가 명시적으로 요청할 때만 실행한다.** mock·`--check`·`--show-args`·pytest는 안전.

## 반드시 지킬 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다. 전원(5 V) ≠ 토크.
- **그리퍼 값 규약은 둘이고 서로 반대다 — 뒤집어도 런타임 경고가 없다.** 토픽·녹화 컬럼(`gello_grip`/`grip_cmd`/
  `grip_pos`, `/robotiq_gripper/*_percent`)은 **0.0=열림 / 1.0=닫힘**. `GripperCommand.position`은 미터(0.085=열림).
  RL `action[6]`(+1=열림/−1=닫힘)은 research 브랜치의 오프라인 변환에만 있다. **레코더를 ±1로 바꾸지 말 것.**
  표: [`docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md`](ros2/GELLO_UR7E_UNITS_REFERENCE.md) §5.
- **그리퍼 이산화 임계값은 `0.3 / 0.7`(히스테리시스: `>=0.7` 닫힘, `<=0.3` 열림, 사이 유지) — 새로 정하지 말 것.**
  방아쇠가 세션당 1~2회 0.2대에서 멈추는 산발적 사건을 이 값이 흡수한다(take 125개 전수 검증, units §5.1).
  같은 값이 research 브랜치 변환기에도 있으므로 브리지 파라미터만 옮기면 학습 데이터와 어긋난다.
- **`grip_pos`(측정 위치)는 이산화하거나 버리지 말 것.** 쥐면 ~0.5에서 멈춰 잡음/헛잡음을 구분하는 유일한 채널이다.
- **그리퍼 셋포인트 유실은 드라이버에서 안 보인다.** rate limiter는 최신 값을 버리지 말고 **합쳐야(coalesce)** 한다
  (`robotiq_gripper_modbus_node.py`, 2026-08-06 수정). "명령 vs 실제" 워치독으로 고치려 하지 말 것 — 거부된 값은
  `_last_cmd_pct`를 안 바꾸므로 불일치가 안 보인다. `trig`≈0.00인데 덜 열림 = 이 문제, `trig` 0.2대 = 방아쇠 문제.
  근거: [`docs/ros2/GELLO_UR7E_GRIPPER.md`](ros2/GELLO_UR7E_GRIPPER.md) §9.
- **검증된 브리지·핸드셰이크 로직을 다시 쓰지 말 것.** `gello_ur_bridge`의 실측 `/joint_states` seed, soft-start,
  step clamp(`max_step_rad` 0.0025 @ 250 Hz — 더 빠르게는 `publish_rate_hz`부터), deadband, One-Euro 필터와
  `gello_move_to_start`의 수렴 게이트 chase + STRICT 전환은 실기 protective stop을 겪고 고친 결과다. 튜닝은 yaml
  파라미터로 하고 근거는 [`GELLO_UR7E_SETUP_CLI.md`](ros2/GELLO_UR7E_SETUP_CLI.md) §5를 먼저 읽는다.
- **카메라 시리얼은 module serial이다**(`rs-enumerate-devices -s`). `lsusb`/sysfs/`journalctl`의 ASIC 시리얼은
  `serial_no`와 절대 매칭되지 않고, 잘못된 시리얼은 에러 없이 프레임 0을 낸다.
  기본값은 `ros2_ur_ws/camera_serials.env` 한 곳(셸 `load_camera_serial_defaults` + Python `gello_recorder.camera_defaults`)이고 export한 `CAM1_SERIAL`/`CAM2_SERIAL`이 이긴다.
- **ROS는 시스템 `/usr/bin/python3`(3.12)로.** conda `python3`가 PATH 앞에 있으면 rclpy/colcon이 깨진다.
- **공유 브랜치다.** 머지·리베이스·푸시 전에 확인을 받을 것.

## 테스트 (인터프리터와 passed 수를 같이 적을 것)

```bash
cd ros2_ur_ws && ./build_ur7e.sh
export PATH=/usr/bin:/bin:$PATH; set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen
(cd src/ur_gello_bringup && /usr/bin/python3 -m pytest -q -p no:cacheprovider test)
(cd src/gello_recorder   && /usr/bin/python3 -m pytest -q -p no:cacheprovider test)
cd .. && /usr/bin/python3 -m pytest -q -p no:cacheprovider gello/dynamixel/tests
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`이 없으면 `~/.local`의 anyio 플러그인 때문에 collection에서 죽는다.
2026-10-30 머지 후 실측(clean `colcon build`, `/usr/bin/python3` 3.12.3, 위 플래그 그대로): `ur_gello_bringup` **625 passed**,
`gello_recorder` **221 passed**(실패 0), `gello/dynamixel/tests` **31 passed**. 세 숫자를 합치지 말 것 — 실행 디렉터리가 다르다.
