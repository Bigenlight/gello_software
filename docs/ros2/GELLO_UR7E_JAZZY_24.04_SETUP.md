# GELLO → UR7e (ROS 2 Jazzy / Ubuntu 24.04) — 이 PC(`junhyeong`) 세팅 델타

이 문서는 [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md)(Humble/22.04, `laptop3`)와
[`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)(실기 handshake·URCap·캘리브레이션)를
**대체하지 않는다.** 두 문서는 지금도 절차 정본이고, 이 문서는 **이 PC(호스트 `junhyeong`,
Ubuntu 24.04 noble, ROS 2 Jazzy, RTX 5070 Ti, 시스템 python 3.12)에서 무엇이 달라지는지**만 적는다.
로봇 handshake·URCap 페어링·캘리브레이션 절차 자체는 배포판이 바뀌어도 그대로이므로 거기서
가져다 쓴다. 네트워크/URCap 재페어링 델타는 별도 문서
[`ros2_ur_ws/setup_jazzy/network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md)로
분리했다 — 이 문서는 소프트웨어/하드웨어 브링업만 다룬다.

> 🛑 이 리포 최상위 `CLAUDE.md`에 "`/home/junhyeong/gello_software`(접미사 없음)는 다른 사람의
> 작업 트리이니 읽지도 쓰지도 말 것"이라는 문구가 있다. **그 경고는 이 작업 브랜치·워크트리에는
> 적용되지 않는다** — 이 문서가 사는 곳은 별도 워크트리 `/home/junhyeong/gello_software_jazzy`,
> 브랜치 `feat/gello-ur7e-jazzy-24.04`다. 같은 호스트에 다른 브랜치용 워크트리가 여러 개
> 공존한다는 뜻이니 혼동하지 말 것.

## 0. 무엇이 바뀌었나 (Humble/laptop3 → Jazzy/이 PC)

| 항목 | laptop3 (Humble, 22.04) | 이 PC (Jazzy, 24.04) |
|---|---|---|
| Ubuntu | 22.04 jammy | **24.04 noble** (측정: `VERSION_CODENAME=noble`) |
| ROS 2 배포판 | Humble | **Jazzy** |
| `ur_robot_driver` | `ros-humble-ur` **2.8.1**(`GELLO_UR7E_REAL_ROBOT.md`) / 일부 문서는 2.13.2 표기 | **`ros-jazzy-ur-robot-driver` 3.8.0** — **major 버전 점프**(2.x → 3.x). 최초 실기 연결 시 switch/handshake 동작이 바뀌었을 수 있으므로 "1) 첫 스모크 테스트 순서"를 생략하지 말 것 |
| `realsense2_camera` | (laptop3 문서에 버전 미기재) | **4.58.4** |
| `librealsense2` | (laptop3 문서에 버전 미기재) | **2.58.4** |
| 시스템 python | 3.10 | **3.12** — colcon/rclpy C 확장이 이 인터프리터로 빌드된다(§2 conda 함정 참고) |
| `dynamixel-sdk` | pip `--user` 설치(GELLO 리더용) | 동일 — pip `--user --break-system-packages`(24.04 PEP 668 externally-managed 제약, 아래 참고) |
| RMW | `rmw_fastrtps_cpp`(기본) | `rmw_fastrtps_cpp`(기본) — **변경 없음**, 두 배포판 다 동일 기본값 |
| GPU | (laptop3는 GPU 약함 — 정책 추론을 원격 서버로 위임) | **RTX 5070 Ti** — 이 PC 자체가 그 "원격 서버" 쪽이었다(CLAUDE.md `junhyeong_ai`). ROS 스택을 얹는다고 그 역할이 없어지지 않으니 GPU 프로세스와의 리소스 경합에 주의 |

두 GELLO Dynamixel·그리퍼 규약(§4 아래)은 배포판과 무관하게 **그대로**다 — 리더는 언제나
passive read-only, 토크를 걸지 않는다. `grip_cmd`/`grip_pos`/`/robotiq_gripper/*_percent`는
**0.0=열림 / 1.0=닫힘**, HIL-SERL RL 액션 `action[6]`은 그 반대로 **+1=열림 / −1=닫힘**(0=유지) —
자세한 전 계층 대조표는 [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) §5,
Modbus 경로 근거는 [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md).

## 1. 설치

전부 [`ros2_ur_ws/setup_jazzy/install_jazzy.sh`](../../ros2_ur_ws/setup_jazzy/install_jazzy.sh)
하나가 한다. **sudo가 필요한 유일한 스크립트**이고 한 번만 손으로 돌린다 — 워크스페이스 빌드는
별도 non-sudo 단계(§3)다. 블록별 요약:

1. **conda를 PATH에서 제거**(§2) → `/usr/bin/python3`인지 확인 → 아니면 즉시 중단(`set -euo pipefail`
   + 명시적 `case` 가드). → `VERSION_CODENAME=noble` 확인.
2. **ROS 2 apt repo 등록** — `ros.key` + `ros2.list`. 표준 절차, Humble과 동일한 패턴.
3. **`ros-jazzy-desktop`** (ros-base가 아니라 desktop — 이 리포의 여러 launch가 RViz를 띄우고
   첫 하드웨어 스모크에 rqt가 쓸모 있어서) + colcon/rosdep/vcstool/pip 빌드 툴체인 및
   `python3-grpcio` / `python3-protobuf` / `python3-h5py` / `python3-zmq`.
4. **UR7e 드라이버 6종** — `ros-jazzy-ur` / `-ur-robot-driver` / `-ur-controllers` /
   `-ur-description` / `-ur-calibration` / `ros2-control` / `ros2-controllers`. 버전은 apt가 고정,
   현재 Jazzy 배포판 기준 **3.8.0**(§0).
5. **RealSense** — `ros-jazzy-realsense2-camera` / `-camera-msgs` / `-librealsense2` 설치 +
   **udev 룰 설치**(아래 별도 항목 — 이 PC에서 가장 밟기 쉬운 함정).
6. **GELLO/기타** — `ros-jazzy-dynamixel-sdk`(ROS 메시지 패키지) + pip `dynamixel-sdk`(gello
   드라이버가 임포트하는 건 이 pip 모듈 쪽이다. `build_ur7e.sh`도 이 이유로 rosdep에서
   `--skip-keys dynamixel_sdk`를 준다). 24.04는 PEP 668 externally-managed 환경이라
   `--break-system-packages` 없이는 시스템 pip 설치가 거부된다 — 스크립트가 이미 그 플래그를 준다.
7. **`rosdep init` + `update`**.

```bash
cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws/setup_jazzy
bash install_jazzy.sh
```

현재 `install_jazzy.sh`는 위 네 Python 런타임 의존성까지 apt로 설치한다. 스크립트는
워크스페이스를 빌드하지 않으며 로봇도 건드리지 않는다.

## 2. conda 함정 — 이 PC에서 가장 시간을 잡아먹을 지점

**이 PC는 로그인 시 miniconda `base`가 자동 활성화된다.** 그 `python3`는 **3.13**이다. Jazzy의
rclpy/colcon C 확장은 시스템 `python3`(**3.12**)로 빌드돼 있으므로, ROS 작업을 하는 **모든
터미널**에서 PATH 맨 앞에 miniconda가 얹혀 있으면 조용히 또는 시끄럽게 깨진다.

측정: `which python3` → `/home/junhyeong/miniconda3/bin/python3` (`CONDA_DEFAULT_ENV=base`,
Python 3.13.11)이 `base` 활성 상태의 기본값이다.

**증상 (이걸 보면 먼저 conda부터 의심할 것):**
- `import rclpy` → `ImportError` (심볼/버전 불일치, 메시지가 매번 다를 수 있음).
- `colcon build`가 `Could not find "ament_package"` 류로 실패하거나, 빌드는 성공한 듯 보여도
  `ros2 run`/`ros2 launch`가 이후 rclpy를 못 찾음.
- `python3 --version`이 3.12가 아니라 3.13/3.1x을 출력.

**고치는 주문 — ROS 터미널을 열 때마다 맨 먼저:**

```bash
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
export PATH
which python3   # /usr/bin/python3 여야 함
python3 --version   # 3.12.x 여야 함
```

`install_jazzy.sh`와 `build_ur7e.sh`는 이 처리를 각각 자동으로 한다(맨 위 설명 참고). 새로
여는 터미널에서 ROS 명령을 직접 실행할 때는 위 PATH 정리가 필요하다. `.bashrc`에서 conda
auto-activate 자체를 끄는 것도 방법이지만(`conda config --set auto_activate_base false`), 이 PC가
GPU 학습 서버 역할도 겸하므로(CLAUDE.md `junhyeong_ai`) **끄지 말 것** — 다른 작업이 conda
기본 활성화에 의존할 수 있다. PATH를 터미널 단위로 벗기는 위 주문이 안전하다.

🪤 **ROS 러너와 pytest 인터프리터 규칙이 반대 방향이라는 것도 laptop3 문서(CLAUDE.md `반드시
지킬 것`)와 동일하게 여기서도 성립한다** — ROS 실행은 conda/PYTHONPATH를 걷어내야 하고, gRPC
관련 venv 실행은 반대로 PYTHONPATH를 덮어써야 한다. 이 PC에서 그 venv들을 아직 만들지 않았다면
해당 없음.

## 3. RealSense udev — 설치 전 관찰과 현재 설치

설치 전 관찰에서 `ros-jazzy-librealsense2` apt 패키지만으로는 **usbfs 접근용 udev 룰이
없었다.** 이 PC에서 설치 전 `/etc/udev/rules.d/99-realsense-libusb.rules`가 **존재하지
않는 것을 확인했다**(`ls /etc/udev/rules.d/ | grep -i realsense` → 매치 없음). 그 상태에서
카메라 노드를 띄우면
usbfs 노드가 `root:root`라 **`LIBUSB_ERROR_ACCESS`**로 죽는다.

현재 `install_jazzy.sh` §4가 upstream(`IntelRealSense/librealsense` `master`)에서 이 룰 파일을
받아 설치하고 `udevadm control --reload-rules` + `trigger`를 실행한다. **설치 후에는 카메라를
물리적으로 재연결(replug)해야 한다** — reload-rules만으로는 이미 열려 있는 장치 노드의 권한이
갱신되지 않는다(udev 룰은 재연결(재-enumerate) 시점에 적용됨). 카메라 2대 모두 재연결할 것.

## 4. 이 PC의 하드웨어 인벤토리 (실측, 세션마다 재확인 권장)

아래는 이 세션에서 read-only 프로브로 직접 측정한 값이다. **그대로 믿지 말고 매 세션 다시
확인하라** — 특히 카메라 시리얼과 latency_timer는 USB 토폴로지가 바뀌면 달라진다.

| 항목 | 측정값 | 확인 명령 |
|---|---|---|
| GELLO FTDI | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` → **`/dev/ttyUSB0`** | `ls -la /dev/serial/by-id/` |
| GELLO 권한 | `crw-rw---- root dialout`, 사용자 `junhyeong`은 `dialout` 그룹 **소속 확인됨** | `groups`, `ls -la /dev/ttyUSB0` |
| GELLO latency_timer | **16**(기본값. laptop3 문서의 저지연 튜닝을 아직 적용 안 한 상태) | `cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` |
| RealSense D435 | **1대만 연결됨.** `lsusb`: `Bus 002 Device 003: ID 8086:0b07 Intel Corp. RealSense D435`. 카메라 시리얼(측정, `udevadm info`): **`143322071682`**; ASIC 식별자는 **`143623022572`**. USB **3.0 SuperSpeed, 5000 Mbps**(`/sys/bus/usb/devices/2-5.1/speed` = `5000`) 링크에 물려 있음(허브 `174c:3074` 경유) | `lsusb`, `udevadm info -a -n /dev/bus/usb/002/003 \| grep serial`, `cat /sys/bus/usb/devices/2-5.1/speed` |
| RealSense 2번 카메라 | **미연결** — `lsusb`에 두 번째 `8086:0b07` 없음 | `lsusb \| grep -i 8086` |
| 이더넷 `enp12s0` | **1 Gbps 대응 NIC**이나 이 세션 측정 시점엔 **carrier=0(케이블 미연결, DOWN)** — 케이블이 랩 스위치/로봇 사이로 옮겨 다니는 중이라 그렇다. 상세는 [`network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md) | `ip -br addr`, `cat /sys/class/net/enp12s0/carrier` |

## 5. 빌드

빌드는 [`ros2_ur_ws/build_ur7e.sh`](../../ros2_ur_ws/build_ur7e.sh)가 한다 — 이 문서에서 절차를
중복하지 않는다. 2026-09-16 현재 `/usr/bin/colcon`을 사용한 세 패키지
(`gello_policy`, `gello_recorder`, `ur_gello_bringup`) 빌드는 이미 통과했다. 드라이버 3.8.0
소스·mock 검토 결과는 [`DRIVER_3_8_REVIEW.md`](../../ros2_ur_ws/setup_jazzy/DRIVER_3_8_REVIEW.md)에
기록돼 있다.

```bash
cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws
./build_ur7e.sh
```

빌드가 성공했는지는 최종적으로 **§6의 `ros2 pkg list`**로 검증한다.

## 6. 첫 스모크 테스트 순서 — 로봇은 절대 움직이지 않는다

아래는 **로봇을 전혀 움직이지 않는** 순서다. 각 단계가 통과해야 다음으로 넘어간다. 실로봇
handshake(§2 "Move-to-start" 수렴 게이트)는 이 순서를 전부 통과한 **뒤에**,
[`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) 절차를 그대로 따른다 — 여기서
반복하지 않는다.

1. **빌드 확인 (§5).**
2. **패키지 등록 확인** (움직임 없음):
   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/gello_software_jazzy/ros2_ur_ws/install/setup.bash
   ros2 pkg list | grep gello
   # gello_policy / gello_recorder / ur_gello_bringup 세 줄이 나와야 함
   ```
3. **카메라만 단독** (움직임 없음, USB만 검증):
   ```bash
   cd /home/junhyeong/gello_software_jazzy/ros2_ur_ws
   ./launch_cameras.sh
   ros2 topic hz /cam1/cam1/color/image_raw/compressed   # ~30 Hz
   ```
   `LIBUSB_ERROR_ACCESS`가 나오면 §3(udev + replug)을 다시 확인.
4. **GELLO publisher 단독** (읽기 전용 리더 — 토크 없음, 로봇 미연결):
   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello launch_rviz:=true
   ```
   GELLO를 손으로 움직여 RViz의 조인트 상태가 따라오는지만 확인한다. **이 단계에서 UR7e는
   전혀 관련이 없다** — 관절 값을 읽기만 한다.
5. **RViz GELLO display-only 확인** (`source:=fake` — 로봇 미연결; RViz 표시 경로이며
   ros2_control mock hardware가 아님):
   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=true
   ```
   `--show-args`가 에러 없이 나오는지도 여기서 같이 확인:
   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py --show-args
   ```
   실제 `ros2_control` mock 검증은 다음처럼 별도로 실행한다(`GenericSystem`, RViz 없음):
   ```bash
   ROS_DOMAIN_ID=173 ros2 launch gello_policy ur_control_fake_safe.launch.py \
     ur_type:=ur7e robot_ip:=127.0.0.1 use_mock_hardware:=true launch_rviz:=false
   ```
   `ROS_DOMAIN_ID=173`으로 격리한 실제 mock 검증에서 SJTC→FPC STRICT 전환은 `ok=True`였고
   friction controller는 active 상태를 유지했다. 이는 실제 UR hardware interface나 실기
   handshake 검증은 아니다.
6. **실로봇 — 여기서부터는 이 문서 범위 밖.** 네트워크/URCap 재페어링이 먼저 끝나야 한다
   ([`network_and_urcap.md`](../../ros2_ur_ws/setup_jazzy/network_and_urcap.md)). 그 다음
   [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2(수렴 게이트 handshake)부터
   그대로 따른다 — 설치된 `ur_robot_driver`가 3.8.0이므로, 첫 연결은 이 버전으로 별도 확인한다.

## 반드시 지킬 것 (laptop3 CLAUDE.md에서 그대로 옮김 — 배포판이 바뀌어도 무효화되지 않음)

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다. 드라이버는 토크를 OFF로
  초기화하고 관절 각도를 읽기만 한다 — 이 성질은 mock/실로봇, Humble/Jazzy와 무관하게 항상
  참이다.
- **그리퍼 값 규약은 두 개고 서로 반대다.** 토픽/녹화 컬럼(`gello_grip`/`grip_cmd`/`grip_pos`,
  `/robotiq_gripper/*_percent`)은 **0.0=열림 / 1.0=닫힘**이고, RL 액션 `action[6]`은
  **+1=열림 / −1=닫힘**(0=유지)이다. 두 범위 다 유효해서 뒤집어도 런타임 경고가 없다 — 레코더는
  언제나 0/1 규약을 쓰고, 뒤집기는 오프라인 변환 단계에서만 일어난다. 근거·전 계층 표는
  [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) §5.
