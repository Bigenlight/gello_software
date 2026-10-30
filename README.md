# GELLO for UR7e (ROS 2 Jazzy)

수동(passive) **GELLO 리더 암**을 손으로 움직여 **UR7e + Robotiq 2F-85**를 원격 조종하고, RealSense 카메라 2대와
함께 데모 데이터를 녹화해 **LeRobot 데이터셋**으로 바꾸는 RWH 랩의 GELLO 포크입니다.
원본은 [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)입니다.

이 브랜치(`ur7e-jazzy`)는 **GELLO 필수 기능만** 담은 기본 브랜치입니다 — 조인트 텔레옵, EEF 텔레옵, 그리퍼,
move-to-start 핸드셰이크, 카메라, 데이터 기록, LeRobot 변환, 작은 시뮬레이션 예제. 연구용 스택(HIL-SERL,
정책 배포, 오프라인 RL, MuJoCo 데이터 수집)은 아래 브랜치로 옮겨졌습니다.

> **안전 불변식:** GELLO는 언제나 **읽기 전용**입니다. GELLO Dynamixel에 **절대 토크를 걸지 마세요.**
> 드라이버는 토크를 OFF로 초기화하고 관절 각도만 읽습니다. 실기 UR7e는 실제로 움직입니다 — E-STOP을 손 닿는 곳에.

---

## 브랜치 지도

| 브랜치 | 내용 |
|---|---|
| **`ur7e-jazzy`** (이 브랜치, 기본) | GELLO 필수 기능 — Ubuntu 24.04 / ROS 2 Jazzy |
| `research/hil-serl` | HIL-SERL 온라인 RL (`serl_ur_infra/`, `run_hil_*.sh`, HIL 검증 런북 `docs/testing/`) |
| `research/policy-deploy` | ACT / Diffusion / FM / IFQL / SVF / DSRL 정책 배포 + 실기 평가 (`gello_policy`) |
| `research/offline-rl` | 오프라인 IFQL 학습 |
| `research/sim-collect` | MuJoCo 데이터 수집 sim (`sim_collect/`), carrot 전용 변환 스크립트 |
| `feat/gello-ur7e-jazzy-24.04` | 분리 전 전체 스냅샷(Jazzy). 아카이브 문서(EEF 설계 PLAN, UR ROS2 PLAN 등)도 여기 |
| `feat/gello-ur7e-humble-22.04` | Ubuntu 22.04 / ROS 2 **Humble** 사용자용 |

## 하드웨어

| 구성 | 이 리그의 값 |
|---|---|
| 팔로워 | **UR7e** (e-Series, PolyScope 5), 기본 IP `192.168.10.11` (`ROBOT_IP=`로 변경) |
| 리더 | GELLO UR형 — Dynamixel XL330 ×7 (팔 6 = XL330-M288, 트리거 1 = XL330-M077), U2D2(FTDI) USB 어댑터, baud 57600, **5 V 별도 전원** |
| 그리퍼 | Robotiq 2F-85 — UR tool 포트 RS485, Modbus RTU over tool-communication(`:54321`) |
| 카메라 | Intel RealSense D435 ×2 — cam1 = SCENE(삼각대), cam2 = WRIST(그리퍼 장착), USB3 |
| 제어 PC | Ubuntu 24.04, ROS 2 Jazzy, 로봇 전용 이더넷 NIC(`192.168.10.100/24`) |

다른 UR 리더 개체로 바꿔 끼우는 경우: [`docs/hardware/GELLO_UR_ARM_INTERCHANGE.md`](docs/hardware/GELLO_UR_ARM_INTERCHANGE.md).

---

## 설치

정본은 [`docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`](docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md)입니다. 요약:

```bash
git clone <this-repo> ~/gello_software && cd ~/gello_software
git checkout ur7e-jazzy
git submodule update --init --recursive                 # DynamixelSDK, mujoco_menagerie

bash ros2_ur_ws/setup_jazzy/install_jazzy.sh            # sudo 1회: ROS 2 Jazzy desktop, UR 드라이버 3.8.0,
                                                        # RealSense + udev 룰, dynamixel-sdk, rosdep
cd ros2_ur_ws && ./build_ur7e.sh                        # ur_gello_bringup + gello_recorder
```

**시리얼 권한 (GELLO).** config는 `/dev/serial/by-id/usb-FTDI_...` 경로를 씁니다(재부팅해도 고정).

```bash
ls -l /dev/serial/by-id/
sudo usermod -aG dialout $USER     # 영구 — 로그아웃/재로그인 필요
sudo chmod 666 /dev/ttyUSB0        # 임시 — 재연결하면 풀림
```

**먼저 모터부터.** 소프트웨어 전에 [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
또는 읽기 전용 프로브 `python3 scripts/gello_probe.py`로 모터 7개가 baud 57600에서 응답하는지 확인합니다
([`docs/testing/02_GELLO_LEADER.md`](docs/testing/02_GELLO_LEADER.md)).

**conda 주의.** conda `base`가 자동 활성화되는 PC에서는 `python3`가 3.12가 아니어서 rclpy/colcon이 깨집니다.
`run_*.sh` 래퍼는 알아서 피하지만, `ros2`를 직접 칠 때는 PATH에서 miniconda를 빼세요(설치 문서 §2).

실로봇 네트워크·URCap: [`ros2_ur_ws/setup_jazzy/network_and_urcap.md`](ros2_ur_ws/setup_jazzy/network_and_urcap.md) ·
[`UR7e_Remote_Control_ROS2.md`](UR7e_Remote_Control_ROS2.md)(Remote 모드).

---

## 빠른 시작

모든 명령은 `~/gello_software/ros2_ur_ws`에서 실행합니다. 래퍼가 ROS 환경과 `GELLO_REPO_ROOT`를 알아서 잡습니다.

### 0. 로봇 없이 먼저 (mock)

```bash
./run_ur7e_gello_sim.sh                 # 물리 GELLO → mock UR7e (RViz)
./run_ur7e_gello_mock.sh                # 실기와 같은 launch를 mock hardware로 — 핸드셰이크 리허설 (물리 GELLO 필요)
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake    # GELLO도 없이 (합성 리더)
```

### 1. 조인트 텔레옵 (실기)

```bash
HEADLESS=true ./run_ur7e_gello_real.sh  # Method B: 펜던트 Remote 모드, Play 불필요 (권장)
./run_ur7e_gello_real.sh                # Method A: 펜던트에서 External Control 프로그램 Play
```

기동 후 로봇이 **살아 있는 GELLO 자세를 따라잡을 때까지** 보간 이동하고(수렴 게이트 move-to-start), 관절별
오차가 `chase_tol` 이내로 유지될 때만 스트리밍으로 넘어갑니다. GELLO를 중립 근처에 잡고 시작하세요.
리더를 계속 움직이면 핸드오버가 **지연될 뿐 스냅하지 않습니다.** 대시보드 헬퍼는 `source remote_helpers.sh && ur_help`.
`START_MODE=init_align` + `./run_operator_console.sh`로 단계별 승인 기동도 가능합니다.

→ [`GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md)(턴키·튜닝·프리플라이트) ·
[`GELLO_UR7E_REAL_ROBOT.md`](docs/ros2/GELLO_UR7E_REAL_ROBOT.md)(핸드셰이크·캘리브레이션) ·
[`GELLO_UR7E_JOINT_DELTA_MODE.md`](docs/ros2/GELLO_UR7E_JOINT_DELTA_MODE.md)(앵커 상대 조인트 모드)

### 2. EEF 텔레옵 (Cartesian delta)

```bash
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef    # 터미널 1 — 기동 시 팔은 움직이지 않음
./run_eef_gui.sh                                            # 터미널 2 — ENGAGE(두 번 클릭/SPACE SPACE) 후에만 움직임
```

마우스처럼 ENGAGE → 이동 → DISENGAGE → 리더 재배치 → ENGAGE(새 앵커). 속도 상한 `v_max`/`w_max`는
`ros2 param set /gello_ur_bridge v_max <값>`으로 라이브 조정. → [`GELLO_UR7E_EEF_MODE.md`](docs/ros2/GELLO_UR7E_EEF_MODE.md)

### 3. 그리퍼

실기 텔레옵(`run_ur7e_gello_real.sh`)은 그리퍼를 **함께** 띄웁니다. 그리퍼만 따로 시험할 때는 팔 드라이버를 내리고:

```bash
./run_ur7e_gripper.sh                   # 로봇 POWER ON 필요 (tool 전압)
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"   # 닫기
```

`:54321`은 단일 클라이언트라 둘을 동시에 띄우면 안 됩니다. 트리거 → 그리퍼는 기본 연속(`gripper_mode:=continuous`),
`gripper_mode:=discrete`면 0.3/0.7 히스테리시스로 열림/닫힘만. → [`GELLO_UR7E_GRIPPER.md`](docs/ros2/GELLO_UR7E_GRIPPER.md)

### 4. 카메라

```bash
./launch_cameras.sh --check             # 연결된 시리얼이 설정값과 맞는지만 (스트림 안 염)
./launch_cameras.sh                     # 2대 + 뷰어 (좌 = cam1 SCENE, 우 = cam2 WRIST)
```

기본 시리얼은 `ros2_ur_ws/camera_serials.env` 한 곳에 있습니다(셸 스크립트와 녹화 GUI가 같이 읽음).
다른 리그에서는 그 파일을 고치거나 `CAM1_SERIAL`/`CAM2_SERIAL`(또는 `GELLO_CAMERA_SERIALS_FILE`)을 export하세요.
설정된 두 시리얼이 모두 연결돼 있지 않으면 **기동 전에 실패**합니다(잘못된 시리얼은 조용히 프레임 0을 냄).
뷰어에서 팔을 움직여 우측 창의 배경이 흐르는지로 손목 카메라를 확인하세요.
→ [`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md)

### 5. 데이터 기록

텔레옵을 띄운 뒤 두 번째 터미널에서 셋 중 하나:

```bash
CAMS=true ./run_recorder.sh                         # 헤드리스 (gello_ur_recorder)
ros2 run gello_recorder gello_recorder_gui          # GUI — take 시작/정지·삭제·태그·GO HOME
./run_eef_collection_gui.sh                         # EEF 수집 통합 GUI (R 녹화 / SPACE engage / G 그리퍼)
```

take마다 `vectors.h5`(관절·커맨드·그리퍼·F/T·TCP, 헤더 `stamp_s` 포함) + `cam1.mp4` / `cam2.mp4` + `metadata.json`이
`ros2_ur_ws/gello_logs/` 아래에 생깁니다. depth는 기본 OFF(`ENABLE_DEPTH=1`).
→ [`GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md) ·
[`GELLO_EEF_COLLECTION_JAZZY.md`](docs/ros2/GELLO_EEF_COLLECTION_JAZZY.md) ·
[`gello_recorder/README.md`](ros2_ur_ws/src/gello_recorder/README.md)

### 6. LeRobot 변환

[`scripts/dataset/`](scripts/dataset/)의 범용 도구 3개(현재 레코더로 찍은 RGB take용):

| 스크립트 | 역할 |
|---|---|
| `make_raw_stats.py` | raw take 폴더 → `dataset_stats.json` (프레임 수·NaN·주기·`stamp_s` 검사) |
| `convert_takes_to_lerobot.py` | take → LeRobot v3.0. `observation.state` = `ur_q1..6` + `grip_pos`, `action` = `cmd1..6` + `grip_cmd`, 마스터 클럭 = cam1 헤더 스탬프 |
| `validate_takes_conversion.py` | 변환기를 import하지 않는 독립 검증 |

lerobot은 **0.6.1(git 핀 `8a74e0a`, PyPI에 없음)** 을 별도 venv에 설치해 씁니다. 업로드 후 `v3.0` 태그를 따로 만들어야
`LeRobotDataset(repo_id)`로 열립니다. 절차: [`GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md) 「허깅페이스 업로드」.

### 7. 시뮬레이션 예제

ROS mock + RViz(UR7e 기구학)와 upstream MuJoCo(UR5e + 2F-85) 두 가지 — **GELLO 없이도** 돌릴 수 있습니다.
→ [`docs/sim/QUICKSTART_KO.md`](docs/sim/QUICKSTART_KO.md) · 상세: [`GELLO_UR_SIM_TELEOP.md`](docs/sim/GELLO_UR_SIM_TELEOP.md),
[`GELLO_PANDA_SIM_TELEOP.md`](docs/sim/GELLO_PANDA_SIM_TELEOP.md)

---

## 안전과 규약

- **E-STOP**을 항상 손 닿는 곳에. 실기 기동 전에 작업 공간을 비우고, 펜던트에서 Remote/Local과 Real Robot 모드를 확인.
  프리플라이트 체크리스트: [`GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md) §6.
- **GELLO 토크는 절대 켜지 않는다.** XL330은 읽기에도 5 V 전원이 필요하지만 **전원 ≠ 토크**. 7 V 초과 금지.
- **그리퍼 값 규약:** 토픽과 녹화 컬럼(`gello_grip`/`grip_cmd`/`grip_pos`, `/robotiq_gripper/*_percent`)은
  **0.0 = 열림 / 1.0 = 닫힘**. 액션 API(`GripperCommand.position`)는 미터 단위라 **0.085 = 열림**으로 방향이 반대입니다.
  RL 액션 `action[6]`의 **+1 = 열림 / −1 = 닫힘** 규약은 research 브랜치에만 있고, 레코더는 언제나 0/1 규약입니다.
  전 계층 표: [`GELLO_UR7E_UNITS_REFERENCE.md`](docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md) §5.
- **mock PASS는 필요조건일 뿐입니다.** mock hardware는 속도 제한도 protective stop도 강제하지 않습니다.

## 문서 지도

| 하려는 일 | 문서 |
|---|---|
| 설치·빌드·무동작 스모크 | [`docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md`](docs/ros2/GELLO_UR7E_JAZZY_24.04_SETUP.md) |
| 실기 턴키 실행·튜닝·프리플라이트 | [`docs/ros2/GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md) |
| 핸드셰이크·URCap·캘리브레이션 | [`docs/ros2/GELLO_UR7E_REAL_ROBOT.md`](docs/ros2/GELLO_UR7E_REAL_ROBOT.md) |
| mock 브링업·브리지 설정 레퍼런스·일시정지/재개 | [`docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md) |
| EEF 텔레옵 | [`docs/ros2/GELLO_UR7E_EEF_MODE.md`](docs/ros2/GELLO_UR7E_EEF_MODE.md) |
| 조인트 델타 모드 | [`docs/ros2/GELLO_UR7E_JOINT_DELTA_MODE.md`](docs/ros2/GELLO_UR7E_JOINT_DELTA_MODE.md) |
| 그리퍼 | [`docs/ros2/GELLO_UR7E_GRIPPER.md`](docs/ros2/GELLO_UR7E_GRIPPER.md) |
| 데이터 기록·LeRobot 변환 | [`docs/ros2/GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md) · [`docs/ros2/GELLO_EEF_COLLECTION_JAZZY.md`](docs/ros2/GELLO_EEF_COLLECTION_JAZZY.md) |
| 단위·그리퍼 규약 | [`docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md`](docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md) |
| 하드웨어 검증 런북 (그리퍼·리더·EEF) | [`docs/testing/README.md`](docs/testing/README.md) |
| 카메라 문제 | [`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md) |
| UR Remote 모드·대시보드 | [`UR7e_Remote_Control_ROS2.md`](UR7e_Remote_Control_ROS2.md) |
| 드라이버 3.8.0 검토 | [`ros2_ur_ws/setup_jazzy/DRIVER_3_8_REVIEW.md`](ros2_ur_ws/setup_jazzy/DRIVER_3_8_REVIEW.md) |
| 패키지 README | [`ur_gello_bringup`](ros2_ur_ws/src/ur_gello_bringup/README.md) · [`gello_recorder`](ros2_ur_ws/src/gello_recorder/README.md) |
| UR5e RViz 경로(원본 launch) | [`docs/ros2/GELLO_UR_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR_ROS2_BRINGUP.md) |

## 리포 구조

```
ros2_ur_ws/
  src/ur_gello_bringup/     GELLO publisher, UR 브리지(조인트/EEF/조인트 델타), move-to-start, 그리퍼, GUI
  src/gello_recorder/       레코더 (headless 노드 + GUI)
  run_ur7e_gello_{real,sim,mock,joint_delta_mock}.sh, run_eef_gui.sh, run_eef_collection_gui.sh,
  run_ur7e_gripper.sh, run_recorder.sh, launch_cameras.sh, run_operator_console.sh, remote_helpers.sh
  setup_jazzy/              install_jazzy.sh, 이더넷 설정, ToolComm 진단, EEF mock 스모크
gello/                      GELLO 코어 (Dynamixel 드라이버·에이전트·sim 로봇)
scripts/                    gello_probe.py(읽기 전용 개체 확인), gello_get_offset.py(캘리브레이션), dataset/
configs/, experiments/      upstream MuJoCo sim 설정과 실행 엔트리포인트
docs/                       ros2/ · hardware/ · testing/ · sim/
ros2/                       upstream Franka FR3 스택 (참고용)
third_party/                DynamixelSDK, mujoco_menagerie (submodule)
```

---

## 데이터셋 (Hugging Face)

이 리그(UR7e + GELLO + Robotiq 2F-85 + RealSense 2대)로 수집해 공개한 데이터셋입니다. 수집 날짜·조명·텔레옵 모드와
레코더 버전에 따라 변환 규칙이 다릅니다. LeRobot 행 수는 cam1 마스터 프레임 수입니다.

| 태스크 | 수집 | 규모 | raw | LeRobot v3 |
|---|---|---|---|---|
| put the right banana in the pot | 2026-07 (joint) | 51 / 21,524 | [raw](https://huggingface.co/datasets/Bigenlight/banana_in_pot_raw) | [joint](https://huggingface.co/datasets/Bigenlight/banana_in_pot_lerobot_v3) · [EE 관측](https://huggingface.co/datasets/Bigenlight/banana_in_pot_ee_lerobot_v3) |
| put the cube in the cup (파일럿) | 2026-07-20 (joint) | 23 / 6,177 | [raw](https://huggingface.co/datasets/Bigenlight/cube_in_cup_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/cube_in_cup_lerobot_v3) |
| put carrot in pot (**depth 포함**) | 2026-09-14 (EEF) | 54 / 18,557 | [raw](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_lerobot_v3) |
| put the orange bowl into the purple bowl | 2026-09-15/16 (EEF) | 52 / 16,009 | [raw](https://huggingface.co/datasets/Bigenlight/orange_bowl_in_purple_bowl_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/orange_bowl_in_purple_bowl_lerobot_v3) |
| put the carrot in the pot, 추가 조명 | 2026-09-18 | 65 / 20,472 | [raw](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_lighting_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_lighting_lerobot_v3) |
| stack the orange bowl on the purple bowl, 추가 조명 | 2026-09-18 | 71 / 17,945 | [raw](https://huggingface.co/datasets/Bigenlight/bowl_stack_lighting_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/bowl_stack_lighting_lerobot_v3) |
| stack the orange bowls and place them into the purple bowl | 2026-09-19 | 60 / 36,030 | [raw](https://huggingface.co/datasets/Bigenlight/bowl_stack_triple_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/bowl_stack_triple_lerobot_v3) |
| stack the red cube on the blue cube | 2026-09-19 | 60 / 20,177 | [raw](https://huggingface.co/datasets/Bigenlight/cube_stack_raw) | [LeRobot](https://huggingface.co/datasets/Bigenlight/cube_stack_lerobot_v3) |

- 규모 = episodes / frames. raw는 take별 `vectors.h5` + `cam1.mp4` / `cam2.mp4`(+ carrot은 `depth.h5`)와 `DATA_DICTIONARY.md`·`dataset_stats.json`.
- **carrot(2026-09-14)** 은 옛 레코더로 찍혀 `ur_joint_states`가 0.9 s 늦게 기록됐고, LeRobot판은 그 지연을 보정했습니다
  (`meta/source_takes.json`의 `ur_joint_states_lag_s`). depth는 HEVC gray12le 무손실, 12-bit 선형(±1.25 mm), 무효 0 보존.
  상세: [`GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md) 「타임스탬프 아티팩트」.
- **orange bowl부터는 수정된 레코더**(`stamp_s`)라 보정 없이 `convert_takes_to_lerobot.py`로 변환했습니다.
- `cube_in_cup` raw 24 take 중 `take_23`은 녹화 오작동(1.64 s)이라 LeRobot판에서 제외했습니다.
- `recover`, `recover_twice`, `other_side` 등 폴더 태그는 raw와 LeRobot 메타데이터(`folder_tag`, `folder_labels`,
  `recovery_label`)에 보존됩니다. 태그는 성공/실패 라벨이 **아닙니다.** 기본 `train` 범위는 전체 코퍼스이므로
  평가 split은 별도로 고정하세요.
- banana/cube 변환기는 별도 리포 [`Bigenlight/banana-in-pot-experiments`](https://github.com/Bigenlight/banana-in-pot-experiments)에 있습니다.

---

## Credits & License

원본 GELLO 프로젝트: [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)
([Project Website](https://wuphilipp.github.io/gello_site/), [Hardware Repository](https://github.com/wuphilipp/gello_mechanical)).
upstream의 YAM / xArm / Franka 사용법은 [upstream README](https://github.com/wuphilipp/gello_software)를 참고하세요.

### Citation

```bibtex
@misc{wu2023gello,
    title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
    author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
    year={2023},
}
```

### License

This project is licensed under the MIT License (see `LICENSE` file).

이 프로젝트는 'FACTR Teleop: Low-Cost Force-Feedback Teleoperation' (Apache-2.0)의 컴포넌트를 사용합니다.
https://github.com/RaindragonD/factr_teleop/ 참고.
