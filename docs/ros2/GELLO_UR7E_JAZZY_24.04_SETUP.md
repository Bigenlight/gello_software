# GELLO → UR7e 설치 (Ubuntu 24.04 / ROS 2 Jazzy)

이 리포의 **설치 정본**이다. 새 PC에서 소프트웨어를 깔고, 워크스페이스를 빌드하고, **로봇을 전혀
움직이지 않는** 스모크 테스트까지 끝내는 것이 범위다. 그 다음 단계는:

- 실로봇 handshake·URCap·캘리브레이션 → [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)
- 턴키 실행(Method A/B)·튜닝·프리플라이트 체크리스트 → [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md)
- 로봇 전용 이더넷·URCap 역접속 → [`ros2_ur_ws/setup_jazzy/network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md)

> Ubuntu 22.04 / ROS 2 Humble을 쓴다면 이 브랜치가 아니라 `feat/gello-ur7e-humble-22.04` 브랜치를
> 쓴다. 이 문서의 경로는 리포를 `~/gello_software`에 clone했다고 가정한다.

## 0. 기준 환경

| 항목 | 값 |
|---|---|
| Ubuntu | **24.04 noble** |
| ROS 2 | **Jazzy** (`ros-jazzy-desktop`) |
| `ur_robot_driver` | **3.8.0** (apt). Humble 시절 2.x에서 major 점프 — 검토 결과는 [`DRIVER_3_8_REVIEW.md`](../../ros2_ur_ws/setup_jazzy/DRIVER_3_8_REVIEW.md) |
| `realsense2_camera` / `librealsense2` | 4.58.4 / 2.58.4 (apt) |
| 시스템 python | **3.12** — colcon/rclpy C 확장이 이 인터프리터로 빌드된다(§2 conda 함정) |
| `dynamixel-sdk` | pip `--user --break-system-packages` (24.04 PEP 668 제약) |
| RMW | `rmw_fastrtps_cpp`(기본) |

배포판과 무관하게 그대로인 것: GELLO 리더는 **passive read-only**(토크를 절대 켜지 않는다),
그리퍼 토픽 규약은 **0.0=열림 / 1.0=닫힘** — 전 계층 대조표는
[`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) §5.

## 1. 설치 — `install_jazzy.sh`

전부 [`ros2_ur_ws/setup_jazzy/install_jazzy.sh`](../../ros2_ur_ws/setup_jazzy/install_jazzy.sh)
하나가 한다. **sudo가 필요한 유일한 스크립트**이고 한 번만 손으로 돌린다. 워크스페이스를 빌드하지
않으며 로봇도 건드리지 않는다. 하는 일:

1. **conda를 PATH에서 제거**(§2) → `/usr/bin/python3`가 아니면 중단 → `VERSION_CODENAME=noble` 확인.
2. **ROS 2 apt repo 등록** (`ros.key` + `ros2.list`).
3. **`ros-jazzy-desktop`**(RViz·rqt 포함) + colcon/rosdep/vcstool/pip/pytest + `python3-h5py` 등
   Python 런타임 의존성.
4. **UR 드라이버** — `ros-jazzy-ur` / `-ur-robot-driver` / `-ur-controllers` / `-ur-description` /
   `-ur-calibration` / `-ros2-control` / `-ros2-controllers`.
5. **RealSense** — `ros-jazzy-realsense2-camera` / `-camera-msgs` / `-librealsense2` /
   compressed image transport + **udev 룰 설치**(§3).
6. **GELLO** — `ros-jazzy-dynamixel-sdk` + pip `dynamixel-sdk`(gello 드라이버가 import하는 것은
   pip 모듈이다. 그래서 `build_ur7e.sh`의 rosdep은 `--skip-keys dynamixel_sdk`).
7. **`rosdep init` + `update`**.

```bash
git clone <this-repo> ~/gello_software && cd ~/gello_software
git checkout ur7e-jazzy
git submodule update --init --recursive     # DynamixelSDK, mujoco_menagerie
bash ros2_ur_ws/setup_jazzy/install_jazzy.sh
```

### 1.1 GELLO 시리얼 권한 (upstream 공통)

GELLO는 U2D2(FTDI) USB-시리얼 어댑터로 붙는다. by-id 경로가 재부팅·재연결에도 고정이라 config는
그것을 쓴다(`ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml`의 `port`).

```bash
ls -l /dev/serial/by-id/            # usb-FTDI_..._<시리얼>-if00-port0
sudo usermod -aG dialout $USER      # 영구. 로그아웃/재로그인 필요
sudo chmod 666 /dev/ttyUSB0         # (임시) 재연결하면 풀린다
```

선택: FTDI latency timer를 1 ms로 줄이면 리더 읽기가 부드러워진다(기본 16 ms, 재연결 시 복귀).

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

소프트웨어 전에 [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)이나
`python3 scripts/gello_probe.py`(읽기 전용)로 **모터 7개가 baud 57600에서 응답하는지** 먼저 본다 —
절차는 [`docs/testing/02_GELLO_LEADER.md`](../testing/02_GELLO_LEADER.md).

## 2. conda 함정 — 가장 시간을 잡아먹는 지점

로그인 시 miniconda `base`가 자동 활성화되는 PC라면 그 `python3`(예: 3.13)가 PATH 맨 앞에 온다.
Jazzy의 rclpy/colcon C 확장은 시스템 `python3`(**3.12**)용이므로 조용히 또는 시끄럽게 깨진다.

**증상 (이걸 보면 conda부터 의심):**
- `import rclpy` → `ImportError`(메시지가 매번 다를 수 있음).
- `colcon build`가 `Could not find "ament_package"` 류로 실패하거나, 빌드는 됐는데 이후
  `ros2 run`/`ros2 launch`가 rclpy를 못 찾음.
- `python3 --version`이 3.12가 아님.

**ROS 터미널을 열 때마다 맨 먼저:**

```bash
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"; export PATH
which python3          # /usr/bin/python3
python3 --version      # 3.12.x
```

`install_jazzy.sh`, `build_ur7e.sh`, `run_*.sh` 래퍼는 이 처리를 스스로 한다. 직접 `ros2 ...`를
칠 때만 필요하다. 같은 PC에서 conda를 다른 작업(GPU 학습 등)에 쓴다면 auto-activate를 끄기보다
터미널 단위로 PATH를 벗기는 위 주문이 안전하다.

## 3. RealSense udev 룰

`ros-jazzy-librealsense2` apt 패키지는 usbfs 접근용 udev 룰을 **설치하지 않는다.** 룰이 없으면
카메라 노드가 `LIBUSB_ERROR_ACCESS`로 죽는다. `install_jazzy.sh`가 upstream(`IntelRealSense/librealsense`)
에서 `99-realsense-libusb.rules`를 받아 설치하고 `udevadm control --reload-rules` + `trigger`까지 한다.

**설치 후 카메라 2대를 물리적으로 다시 꽂을 것** — 룰은 재열거 시점에 적용된다.

카메라 시리얼·cam1/cam2 역할·스트리밍 장애는
[`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](../hardware/REALSENSE_D435_TROUBLESHOOTING.md).

## 4. 하드웨어 인벤토리 — 매 세션 다시 확인

USB 토폴로지가 바뀌면 값이 바뀐다. 아래 명령으로 확인하고, 숫자를 스크립트에 옮겨 적지 않는다.

| 항목 | 기대 | 확인 명령 |
|---|---|---|
| GELLO FTDI | `/dev/serial/by-id/usb-FTDI_..._<시리얼>-if00-port0` → `/dev/ttyUSBn` | `ls -la /dev/serial/by-id/` |
| GELLO 권한 | `dialout` 소속 | `groups`, `ls -la /dev/ttyUSB0` |
| RealSense D435 ×2 | `8086:0b07` 2개, USB3(`5000M`). `8086:0adb`면 DFU 복구 모드 → 재연결 | `lsusb \| grep 8086`, `lsusb -t`, `rs-enumerate-devices -s` |
| 로봇 NIC | 로봇 전용 NIC에 `192.168.10.x/24`, carrier 1 | `ip -br addr`, [`network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md)의 `setup_ur7e_ethernet.sh --check` |
| 그리퍼 ToolComm | 읽기 전용 진단 | `ros2_ur_ws/setup_jazzy/check_robotiq_toolcomm.sh` |

## 5. 빌드

```bash
cd ~/gello_software/ros2_ur_ws
./build_ur7e.sh        # ur_gello_bringup + gello_recorder (rosdep 검사 포함)
```

부분 빌드는 `GELLO_BUILD_PACKAGES=ur_gello_bringup ./build_ur7e.sh`. yaml·launch를 고친 뒤에도
다시 빌드해야 `install/`에 반영된다.

## 6. 첫 스모크 테스트 — 로봇은 움직이지 않는다

각 단계가 통과해야 다음으로 넘어간다.

1. **패키지 등록**
   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/gello_software/ros2_ur_ws/install/setup.bash
   ros2 pkg list | grep -E "ur_gello_bringup|gello_recorder"    # 두 줄
   ```
2. **카메라만** (USB만 검증)
   ```bash
   cd ~/gello_software/ros2_ur_ws
   ./launch_cameras.sh --check          # 시리얼 매핑만, 스트림 안 염
   ./launch_cameras.sh                  # 두 스트림 + 뷰어
   ros2 topic hz /cam1/cam1/color/image_raw/compressed     # ~30 Hz
   ```
3. **mock UR7e + 합성 리더** (`source:=fake` 사인파, GELLO 불필요). `ur7e_gello_rviz.launch.py`는
   공식 `ur_control.launch.py`를 ros2_control mock hardware(`use_mock_hardware`)로 띄우고 브리지를
   붙인다 — 실로봇에 접속하지 않는다.
   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py --show-args
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=true
   ```
4. **mock UR7e + 실물 GELLO 리더** (읽기 전용)
   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello launch_rviz:=true
   ```
   GELLO를 손으로 움직여 RViz의 팔이 따라오는지 본다.
5. **ros2_control mock으로 실기 기동 시퀀스 리허설** (실로봇 접속 불가가 스크립트에 박혀 있다. 둘 다 **물리 GELLO 필요**)
   ```bash
   ./run_ur7e_gello_mock.sh       # driver → move-to-start → STRICT 전환 → bridge, mock hardware
   ./run_ur7e_gello_sim.sh        # 물리 GELLO가 mock UR7e를 RViz에서 구동
   ```
   Jazzy 3.8.0 + `GenericSystem` mock에서 SJTC→FPC STRICT 전환은 `ok=True`, friction controller는
   active 유지로 확인됐다(2026-09-16). mock 성공은 실제 UR hardware interface의 재시딩을 측정한
   것이 **아니다.** EEF 경로의 mock 스모크는 `setup_jazzy/eef_mock_smoke.py`.
6. **실로봇** — 여기서부터는 이 문서 범위 밖. 네트워크/URCap
   ([`network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md)) → 
   [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2(수렴 게이트 handshake).

## 반드시 지킬 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 드라이버는 토크를 OFF로 초기화하고 관절 각도를 읽기만
  한다 — mock/실로봇, Humble/Jazzy와 무관하게 항상 참이다.
- **그리퍼 값 규약은 두 개고 서로 반대다.** 토픽/녹화 컬럼(`gello_grip`/`grip_cmd`/`grip_pos`,
  `/robotiq_gripper/*_percent`)은 **0.0=열림 / 1.0=닫힘**. RL 액션 `action[6]`(+1=열림 / −1=닫힘)은
  research 브랜치에만 있고 뒤집기는 오프라인 변환에서만 일어난다. 레코더는 언제나 0/1 규약이다.
