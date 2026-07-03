# GELLO -> UR7e (ROS2 Humble / Ubuntu 22.04) — 턴키 세팅 CLI & 체크리스트

이 문서는 **UR7e + GELLO**를 Ubuntu 22.04 / ROS2 Humble 환경에서 처음부터 끝까지 "턴키(turnkey)"로 세팅하기 위한 실전 CLI 가이드입니다. 사전 요구사항 점검 → 의존성 설치/빌드 → 시뮬레이션 검증 → 실로봇 실행(Method A/B) → 튜닝 → 프리플라이트 체크리스트 → 트러블슈팅 순서로, 각 단계를 복사-붙여넣기 가능한 명령과 함께 정리했습니다. 실제 UR7e 로봇 팔을 물리 GELLO leader로 조인트 공간(joint-space) 텔레옵으로 구동하는 것이 목표이며, 시뮬레이션(RViz2 mock robot) 경로와 실로봇 경로를 모두 다룹니다.

## 목차 (Table of Contents)

- [0) 개요 + 사전 요구사항(하드웨어/OS/네트워크/권한)](#0-개요--사전-요구사항하드웨어os네트워크권한)
- [1) 의존성 설치 + 빌드](#1-의존성-설치--빌드)
- [2) 시뮬레이션 먼저 검증 (실로봇 없이)](#2-시뮬레이션-먼저-검증-실로봇-없이)
- [3) 실로봇 — Method A (펜던트 Play)](#3-실로봇--method-a-펜던트-play)
- [4) 실로봇 — Method B (Remote 헤드리스, Play 불필요)](#4-실로봇--method-b-remote-헤드리스-play-불필요)
- [5) 튜닝 — 떨림/속도 (파라미터 표)](#5-튜닝--떨림속도-파라미터-표)
- [6) 실행 전 프리플라이트 체크리스트 (안전)](#6-실행-전-프리플라이트-체크리스트-안전)
- [7) 트러블슈팅 + 빠른 명령 치트시트](#7-트러블슈팅--빠른-명령-치트시트)

---

## 0) 개요 + 사전 요구사항(하드웨어/OS/네트워크/권한)

### 0.1 이 문서는 무엇인가

이 문서는 **UR7e + GELLO**를 Ubuntu 22.04 / ROS2 Humble 환경에서 처음부터 끝까지 "턴키(turnkey)"로 세팅하기 위한 실전 가이드입니다. 실제 UR7e 로봇 팔을 물리 GELLO leader로 조인트 공간(joint-space) 텔레옵으로 구동하는 것이 목표이며, 시뮬레이션(RViz2 mock robot) 경로와 실로봇 경로를 모두 다룹니다.

- 작업 브랜치: `feat/gello-ur7e-humble-22.04` (`feat/gello-panda-sim-teleop`에서 분기; 그쪽은 24.04/Jazzy 라인이므로 혼동하지 말 것)
- 저장소 경로: `/home/laptop3/gello_software`
- 워크스페이스: `ros2_ur_ws` (colcon), 패키지 `ur_gello_bringup` (ament_python, ur5e용을 재사용, 기본 `ur_type=ur7e`)

먼저 브랜치를 확인/체크아웃합니다:

```bash
cd /home/laptop3/gello_software
git checkout feat/gello-ur7e-humble-22.04
```

### 0.2 하드웨어 체크리스트

| 항목 | 값 / 확인 방법 |
|---|---|
| 팔(follower) | UR7e (e-Series), PolyScope 5.24.1, `robot_ip = 192.168.10.11` |
| GELLO leader | Dynamixel 기반 leader arm, 시리얼 ID `FTBEO6QK` |
| GELLO 시리얼 포트 | `/dev/ttyUSB0` (== `/dev/serial/by-id/...FTBEO6QK...if00-port0`) |
| 네트워크 | 제어 PC와 UR7e 펜던트가 같은 서브넷에서 `192.168.10.11`로 ping 가능해야 함 |
| E-STOP | 항상 손이 닿는 위치에 두고 작업 (실로봇은 실제로 움직입니다) |

네트워크 연결 확인:

```bash
ping -c 3 192.168.10.11
```

GELLO leader 포트 확인:

```bash
ls -l /dev/serial/by-id/ | grep -i FTBEO6QK
# 보통 /dev/ttyUSB0 로 심볼릭 링크됨
```

### 0.3 OS / ROS 가정

- OS: Ubuntu 22.04 LTS
- ROS2 배포판: Humble
- `ros-humble-ur` 패키지: 버전 2.8.1 기준으로 이 문서를 검증함 (자세한 verified/partial 구분은 0.5절 참고)

Humble이 설치·소싱되어 있는지 확인:

```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

### 0.4 권한: dialout 그룹 + 시리얼 퍼미션

GELLO leader는 USB 시리얼(`/dev/ttyUSB0`)로 연결됩니다. 이 장치에 접근하려면 사용자가 `dialout` 그룹에 속해 있어야 합니다.

```bash
groups $USER | grep dialout || sudo usermod -aG dialout $USER
# 그룹에 새로 추가했다면 로그아웃/로그인(또는 재부팅) 필요
```

정상적으로 `dialout` 그룹에 속해 있다면 별도 `chmod` 없이 `/dev/ttyUSB0`에 접근할 수 있어야 합니다. 다만 환경에 따라(udev 규칙 미적용 등) 권한 문제가 발생할 수 있으며, 이런 경우를 위해 저장소 루트 `README.md`에 **임시방편으로 `chmod 666 /dev/ttyUSB0` 워크어라운드**가 문서화되어 있습니다. 이는 임시 조치이며 재부팅/재연결 시마다 다시 적용해야 하므로, 가능하면 `dialout` 그룹 가입 쪽을 우선 시도하세요.

```bash
# 워크어라운드 (임시, 매 재연결 시 재실행 필요) — 루트 README.md 참고
sudo chmod 666 /dev/ttyUSB0
```

### 0.5 "Verified" vs "Partial" 지원 범위 (ur7e on ros-humble-ur 2.8.1)

`ros-humble-ur` 2.8.1에서 `ur_type:=ur7e`를 사용할 때의 지원 범위를 명확히 구분합니다:

- **Verified (완전 검증됨) — 조인트 공간 텔레옵**: `ur_type:=ur7e`가 정상적으로 로드/구동되며, joint-space GELLO 텔레옵은 완전히 정확합니다. 이는 ur5e와 ur7e의 `joint_limits`가 동일하기 때문입니다 (모든 조인트 180 deg/s = 3.14159 rad/s). 이 문서의 핵심 시나리오(GELLO로 조인트를 따라 움직이는 것)는 이 범위에 있습니다.
- **Partial (부분 지원) — 시각화/기구학**: `config/ur7e`의 visual_parameters가 실제로는 `meshes/ur5e`를 가리키고 있어 RViz에서 ur5e 모델이 표시됩니다. 또한 `ur7e_update_rate.yaml` 파일이 없어 무해한 경고(warning)가 출력됩니다.
- 정확한 ur7e 비주얼/기구학이 필요한 경우에만 `ros-humble-ur` 2.13.2로 업그레이드하세요. 조인트 텔레옵 목적이라면 ur5e/ur7e는 상호 교환 가능(interchangeable)합니다.

다음 절부터는 의존성 설치, 빌드, 시뮬레이션/실로봇 실행 순서로 이어집니다.

---

## 1) 의존성 설치 + 빌드

### 1-1. ROS2 UR 드라이버 (apt)

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ur \
  ros-humble-ur-robot-driver \
  ros-humble-ur-controllers
```

- `ros-humble-ur` 2.8.1 기준으로 `ur_type:=ur7e`가 로드/구동은 되지만, 일부 리소스(비주얼 메시, `ur7e_update_rate.yaml`)가 아직 ur5e를 재사용하는 부분 지원 상태입니다. 조인트 텔레옵(GELLO)은 ur5e/ur7e의 `joint_limits`가 동일(전 조인트 180 deg/s = 3.14159 rad/s)하므로 문제 없이 정확합니다. RViz에서 ur5e 메시가 보이는 것과 `ur7e_update_rate.yaml` missing 경고는 무시해도 됩니다.
- 정확한 ur7e 비주얼/기구학이 필요하면 `ros-humble-ur` 2.13.2로 업그레이드하세요 (조인트 텔레옵만 할 거면 불필요).

### 1-2. dynamixel_sdk (GELLO 리더암 SDK, pip — sudo 불필요)

```bash
pip install --user dynamixel-sdk
```

- apt 패키지가 아니라 pip으로 설치합니다 (검증된 버전: 4.0.5). `sudo`가 필요 없고, 시스템 파이썬을 건드리지 않습니다.
- 리더암 시리얼: `/dev/ttyUSB0` (== `/dev/serial/by-id/...FTBEO6QK...if00-port0`). 사용자 계정이 `dialout` 그룹에 속해 있어야 합니다.

### 1-3. GELLO_REPO_ROOT 환경변수

`ur_gello_bringup`의 `gello_publisher` 노드가 `source:=gello`로 실행될 때 이 저장소의 GELLO 드라이버를 import합니다. 셸 프로필(`~/.bashrc`)에 추가해 두세요.

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
echo 'export GELLO_REPO_ROOT=/home/laptop3/gello_software' >> ~/.bashrc
```

### 1-4. colcon 빌드

`ros2_ur_ws/` 안에 있는 스크립트로 한 번에 빌드합니다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./build_ur7e.sh
```

`build_ur7e.sh`는 내부적으로 아래와 동일합니다 (수동으로 하려면):

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

빌드 후에는 항상 `install/setup.bash`를 source해야 새 터미널/스크립트에서 `ur_gello_bringup` 패키지가 보입니다.

```bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
```

### 1-5. 패키지 인식 확인 (verify)

```bash
ros2 pkg list | grep ur_gello_bringup
```

정상이면 `ur_gello_bringup` 한 줄이 출력됩니다. 런치 파일의 인자까지 확인하려면:

```bash
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py --show-args
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py --show-args
```

`--show-args`가 `robot_ip`, `use_fake_hardware`, `kinematics_params_file` 등 인자 목록을 에러 없이 출력하면 빌드/소스 설정이 정상입니다. `Package 'ur_gello_bringup' not found` 에러가 나면 `install/setup.bash`를 source하지 않았거나 빌드가 실패한 것이니 1-4를 다시 확인하세요.

---

## 2) 시뮬레이션 먼저 검증 (실로봇 없이)

실로봇(UR7e, 192.168.10.11)에 연결하기 **전에** 반드시 시뮬레이션(mock UR7e + RViz2)으로 먼저 검증합니다. 이 단계를 건너뛰고 바로 실로봇으로 가지 마세요.

### 2-1. 셀프테스트 (source:=fake, 사인파)

GELLO 리더 없이 소프트웨어 스택만 확인하는 자체 테스트입니다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_ur7e_gello_sim.sh source:=fake
```

- 내부적으로 `ur7e_gello_rviz.launch.py source:=fake` 를 실행하며, `use_fake_hardware`로 mock UR7e를 띄우고 `forward_position_controller`를 active 상태로 만듭니다.
- RViz2가 뜨면 UR7e(현재 ur5e 메시로 표시됨, 조인트 값은 정확) 모델의 각 조인트가 사인파 형태로 천천히 움직이는지 확인합니다. 이것만 되면 컨트롤러/토픽 배선은 정상입니다.

### 2-2. 실제 GELLO로 mock UR7e 구동 (source:=gello)

물리적인 GELLO 리더 암을 실제로 손으로 움직여 RViz의 UR7e가 따라오는지 확인하는, 실로봇 투입 전 필수 드라이런입니다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
export GELLO_REPO_ROOT=/home/laptop3/gello_software   # gello 드라이버 import 에 필요
./run_ur7e_gello_sim.sh source:=gello
```

- GELLO 리더의 USB 시리얼(`/dev/ttyUSB0`, by-id에 `FTBEO6QK`)이 인식되어 있어야 합니다. dialout 그룹 권한 필요.
- RViz2에서 확인할 것: **GELLO 암을 손으로 움직이면 mock UR7e 모델이 실시간으로 동일하게 따라 움직여야 합니다.** 6개 조인트 전부 방향과 속도가 자연스럽게 매칭되는지 확인 (부호 반전, 뒤집힌 축, 튐 현상이 없어야 함).
- 대기 상태(GELLO를 가만히 들고 있을 때)에서 RViz의 팔이 미세하게 떨리면(tremor) `config/ur7e_gello.yaml`의 `deadband_rad`(기본 0.004)를 0.006~0.01로 올려서 재확인합니다.

### 2-3. 컨트롤러/토픽 상태로 이중 확인

RViz만으로 판단하지 말고, 아래 명령으로 컨트롤러와 토픽이 실제로 정상 동작 중인지 반드시 교차 확인합니다.

```bash
# 새 터미널에서 (같은 workspace source 필요)
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash

# 컨트롤러 상태 확인: forward_position_controller가 active 여야 함
ros2 control list_controllers
```

기대 출력 예 (핵심은 `forward_position_controller` 줄이 `active`):
```
forward_position_controller[position_controllers/JointGroupPositionController] active
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
scaled_joint_trajectory_controller[...]  inactive  (혹은 unconfigured)
```

```bash
# 명령 토픽이 브릿지 발행 주기대로 나가는지 확인 (source:=gello 로 GELLO를 움직이면서)
ros2 topic hz /forward_position_controller/commands
```

- `average rate: ~250.00` 근처(브릿지 `publish_rate_hz=250.0` 기준)가 안정적으로 찍히면 정상입니다. 이 커맨드 토픽은 30 Hz GELLO 리더 스트림과 **분리(decoupled)**되어 브릿지가 250 Hz로 발행합니다. rate가 0이거나 끊기면 GELLO 시리얼 연결, `GELLO_REPO_ROOT` 환경변수, USB 권한을 다시 점검하세요.

```bash
# (선택) 30 Hz GELLO 리더 스트림 자체를 확인하고 싶으면 이 토픽을 봅니다
ros2 topic hz /gello/joint_states
```

- `average rate: ~30.00` 근처(`gello_publisher`의 리더 읽기 주기 30 Hz)가 나오면 정상입니다. 커맨드 토픽(250 Hz)과 리더 스트림(30 Hz)은 서로 다른 주기임에 유의하세요.

### 통과 기준 (다음 단계로 넘어가도 되는 조건)

- [ ] `source:=fake` 사인파 자체 테스트가 RViz에서 정상 동작
- [ ] `source:=gello` 로 물리 GELLO를 움직이면 RViz UR7e가 6축 모두 자연스럽게 추종
- [ ] `ros2 control list_controllers` 에서 `forward_position_controller` = `active`
- [ ] `ros2 topic hz /forward_position_controller/commands` 가 끊기지 않고 지속 발행 (~250 Hz, 브릿지 `publish_rate_hz=250.0`)
- [ ] (선택) `ros2 topic hz /gello/joint_states` 가 ~30 Hz 로 발행 (GELLO 리더 스트림)
- [ ] 대기 상태에서 떨림(tremor) 없음

위 5가지가 모두 확인되어야 실로봇(`./run_ur7e_gello_real.sh`) 단계로 진행합니다.

---

## 3) 실로봇 — Method A (펜던트 Play)

Method A는 UR 펜던트에서 **External Control** 프로그램을 사람이 직접 Load + Play 하는 방식입니다. 헤드리스(Method B, `HEADLESS=true`)와 절대 섞어서 쓰지 마세요.

### 3-1. 최초 1회 설정

- **로봇 IP**: `192.168.10.11` (UR7e, e-Series, PolyScope 5.24.1). `run_ur7e_gello_real.sh`의 기본값이므로 보통 그대로 두면 됩니다. 다른 IP를 쓸 경우:
  ```bash
  ROBOT_IP=192.168.10.11 ./run_ur7e_gello_real.sh
  ```
- **(선택) UR 캘리브레이션** — TCP/Cartesian 정확도를 위한 것으로, 조인트 텔레옵 자체에는 필요 없습니다. 조인트 스페이스 팔로잉만 쓸 거면 건너뛰어도 됩니다.
  ```bash
  ros2 launch ur_calibration calibration_correction.launch.py \
    robot_ip:=192.168.10.11 \
    target_filename:="${HOME}/ur7e_calibration.yaml"
  ```
  캘리브레이션을 만들었다면 실행 스크립트에 `CALIB`로 넘겨줍니다:
  ```bash
  CALIB="${HOME}/ur7e_calibration.yaml" ./run_ur7e_gello_real.sh
  ```

### 3-2. 시작 시퀀스가 하는 일 (내부 동작 이해)

`ur7e_gello_real.launch.py`는 노드들을 순차 타이머로 띄웁니다:

1. **t=0s** — `ur_robot_driver` 기동. `scaled_joint_trajectory_controller`가 ACTIVE 상태로, `forward_position_controller`는 로드는 되지만 INACTIVE 상태로 올라옵니다.
2. **t=6s** — `gello_publisher` 기동 (물리 GELLO 리더암에서 조인트 각도 30Hz로 퍼블리시).
3. **t=8s** — `gello_move_to_start` (핸드셰이크 노드) 기동:
   - `scaled_joint_trajectory_controller`가 ACTIVE가 될 때까지 대기 (= 펜던트에서 External Control Play를 누르는 시점).
   - ACTIVE가 확인되면, 현재 GELLO 리더암의 포즈를 읽고 `/scaled_joint_trajectory_controller/follow_joint_trajectory` 액션으로 로봇을 그 자세까지 부드럽게(보간, `trajectory_duration=5.0`, `arrival_tolerance=0.05`) 이동시킵니다.
   - 도착 확인 후 **STRICT 컨트롤러 스위치**: `forward_position_controller`를 activate, `scaled_joint_trajectory_controller`를 deactivate.
4. 핸드셰이크 노드가 **returncode==0**으로 정상 종료한 경우에만, `RegisterEventHandler(OnProcessExit(...))`를 통해 **브리지(`gello_ur_bridge`) + 그리퍼**가 자동으로 기동됩니다. (실패 시 브리지는 절대 켜지지 않음 — 안전장치.)

이 핸드셰이크는 **필수**입니다. 생략하면 GELLO의 현재 포즈로 로봇이 순간이동(스냅)하려다 velocity-limit 프로텍티브 스탑이 걸립니다.

### 3-3. 실행

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_ur7e_gello_real.sh
```

스크립트가 대기 상태에 들어가면(약 t=8s 부근, `move_to_start`가 컨트롤러 ACTIVE를 기다리는 시점), **펜던트에서**:

1. Program 탭에서 External Control 프로그램(예: `external_control.urp` 등, 드라이버가 지정한 프로그램)을 **Load**.
2. 하단 **Play(▶)** 버튼을 누릅니다.

Play를 누르면 `scaled_joint_trajectory_controller`가 ACTIVE로 전환되고, 대기하던 `move_to_start`가 즉시 이어서 진행됩니다.

> GELLO 리더암은 항상 **패시브 read-only**입니다 — Dynamixel에는 토크가 절대 인가되지 않고 읽기만 합니다. 실제로 움직이는 것은 UR7e 팔로워뿐입니다.

### 3-4. 정상 로그 확인

정상 부팅 시 대략 다음과 같은 순서로 로그가 보여야 합니다:

```
[ur_robot_driver] ... scaled_joint_trajectory_controller ... active
[gello_publisher] Publishing joint states at 30.0 Hz (port ...FTBEO6QK...)
[gello_move_to_start] Waiting for scaled_joint_trajectory_controller to become ACTIVE...
# <-- 여기서 펜던트 Load + Play 수행 -->
[gello_move_to_start] Controller active, moving robot to GELLO start pose...
[gello_move_to_start] Trajectory goal reached (within arrival_tolerance=0.05)
[gello_move_to_start] Switching controllers: activate=forward_position_controller, deactivate=scaled_joint_trajectory_controller
[gello_move_to_start] Handshake complete (returncode=0)
[gello_ur_bridge] Bridge started (ema_alpha=0.4, max_step_rad=0.0025, deadband_rad=0.004, publish_rate_hz=250.0)
[gripper] Gripper node started
```

`Handshake complete` 로그와 브리지/그리퍼 기동 로그가 뒤따라야 정상입니다. 이 로그가 안 뜨면 브리지가 켜지지 않은 것이므로 GELLO를 움직여도 로봇은 반응하지 않습니다 (의도된 안전 동작).

### 3-5. 실제 로봇 움직임 시점

- 로봇은 **핸드셰이크가 시작되는 시점(런치 기준 약 t=8초, 실제로는 펜던트 Play를 누른 직후)**에 GELLO의 현재 포즈까지 5초간 보간 이동하며 처음으로 움직입니다.
- **E-STOP을 항상 손 닿는 곳에 두고, 작업 공간을 비우고, Play를 누르기 전에 GELLO 리더암을 로봇의 현재 자세(중립 부근)와 비슷하게 잡아두세요** — 로봇은 GELLO가 어디에 있든 그 자세로 이동합니다.
- 이후 컨트롤러가 `forward_position_controller`로 전환되면 브리지가 GELLO 움직임을 실시간으로(30Hz 입력 → 250Hz 퍼블리시, EMA/데드밴드/스텝 제한 적용) 따라갑니다.

### 3-6. 문제 발생 시 복구

- Play 이후 연결이 끊긴 경우(Method A): 펜던트에서 다시 **Play**를 누르면 재개됩니다.
- `remote_helpers.sh`를 source해두면 `ur_play`, `ur_resend`, `ur_unlock` 등으로 대시보드에서도 복구 가능합니다:
  ```bash
  source remote_helpers.sh
  ur_play
  ```
- Method A와 Method B(`HEADLESS=true`)는 한 세션에서 절대 혼용하지 마세요.

---

## 4) 실로봇 — Method B (Remote 헤드리스, Play 불필요)

Method B는 펜던트를 **Remote 모드**로 켜두면, 드라이버가 URScript를 직접 전송하고
reverse interface가 자동 연결되어 `scaled_joint_trajectory_controller`가 자동 활성화되는
방식입니다. 매 실행마다 펜던트에서 Play를 누를 필요가 없습니다 (핸즈프리). 실행 시
`HEADLESS=true` 환경변수를 주면 launch에 `headless_mode:=true` 인자가 전달됩니다.

> 주의: Method A(수동 Play)와 Method B(헤드리스)를 **한 세션에서 섞어 쓰지 마세요.**
> Remote → Local 전환은 반드시 **펜던트에서만** 수행합니다.

### 4.1 원타임 설정 — 펜던트 Remote 모드 활성화

펜던트에서 최초 1회만 설정합니다 (이후에는 그대로 유지됨):

1. 햄버거 메뉴 > **Settings > System > Remote Control > Enable**
2. 화면 우측 상단의 **Local/Remote** 토글을 **REMOTE**로 전환
   - 전환 후 Play/Load 버튼이 회색으로 비활성화되는 것이 **정상**입니다.
3. 화면 우측 하단이 **Real Robot**(시뮬레이션 아님)인지 확인

### 4.2 헤드리스 실행

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
HEADLESS=true ./run_ur7e_gello_real.sh
```

- 기본 `ROBOT_IP=192.168.10.11`. 다른 로봇이면 오버라이드:
  ```bash
  ROBOT_IP=192.168.10.11 HEADLESS=true ./run_ur7e_gello_real.sh
  ```
- 캘리브레이션 파일을 사용하는 경우:
  ```bash
  CALIB=/path/to/ur7e_calibration.yaml HEADLESS=true ./run_ur7e_gello_real.sh
  ```
- 내부적으로 `headless_mode:=true` launch 인자가 전달되어 배너/로그에 Method B로 표시됩니다.
- 부팅 시퀀스(자동): driver 기동(t0) → reverse interface 자동 연결 → `scaled_joint_trajectory_controller` 자동 ACTIVE → `gello_publisher`(t6) → `gello_move_to_start` 핸드셰이크(t8, 로봇이 GELLO 자세로 실제 이동) → 핸드셰이크 성공 시 bridge + gripper 시작.

**안전 수칙**: 로봇이 t≈8초 부근에서 실제로 움직입니다. E-STOP을 손 닿는 곳에 두고,
작업 공간을 비우고, GELLO를 중립 자세 근처에 든 상태로 시작하세요
(로봇은 GELLO가 있는 자세로 그대로 따라옵니다).

### 4.3 remote_helpers.sh — 대시보드 헬퍼

새 터미널에서 소싱하여 사용합니다:

```bash
source remote_helpers.sh
ur_help          # 사용 가능한 함수 목록 출력
```

주요 함수:

| 함수 | 설명 | 서비스 타입 |
|---|---|---|
| `ur_mode` | 현재 로봇 모드 조회 | `ur_dashboard_msgs/srv/GetRobotMode` |
| `ur_program` | 로드된 프로그램 상태 조회 | `ur_dashboard_msgs/srv/GetLoadedProgram` |
| `ur_powerup` | 전원 on + 브레이크 해제 | `std_srvs/srv/Trigger` |
| `ur_load PROG` | `.urp` 프로그램 로드 | `ur_dashboard_msgs/srv/Load {filename:}` |
| `ur_play` | Play (Method A 전용) | `std_srvs/srv/Trigger` |
| `ur_stop` | Stop | `std_srvs/srv/Trigger` |
| `ur_unlock` | Protective stop 해제 | `std_srvs/srv/Trigger` |
| `ur_resend` | 헤드리스 연결 끊김 복구 | `std_srvs/srv/Trigger` |
| `ur_headless_driver` | 드라이버 단독 연결성 점검 (움직임 없음) | — |

### 4.4 장애 복구

- **Method B(헤드리스) 도중 연결 끊김**: `ur_resend` 호출 →
  `/io_and_status_controller/resend_robot_program` 트리거.
- **Method A 도중 끊김**: 펜던트에서 다시 `ur_play`.
- **Remote → Local 전환**: ROS2/터미널에서는 불가능하며, **펜던트 우측 상단 토글에서만** 수행.

```bash
source remote_helpers.sh
ur_resend
```

### 4.5 Bare-driver 연결성 점검 (모션 없음)

로봇을 움직이지 않고 드라이버만 실제 로봇과 통신되는지 확인하고 싶을 때:

```bash
source remote_helpers.sh
ur_headless_driver
```

이 점검은 드라이버-로봇 간 연결/리버스 인터페이스 상태만 확인하며, 어떠한 조인트 명령도
전송하지 않습니다. GELLO 핸드셰이크나 컨트롤러 전환 전에 하드웨어 연결만 빠르게
검증하고 싶을 때 사용하세요.

---

## 5) 튜닝 — 떨림/속도 (파라미터 표)

브리지 노드(`gello_ur_bridge`)는 GELLO 리더 → UR7e 팔로워 커맨드를 안전 속도로 필터링해서 내보냅니다. 떨림(tremor)이나 반응 지연이 느껴지면 아래 표의 파라미터를 조정하세요.

### 설정 파일

```
ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml
```

### 파라미터 표

| 파라미터 | 기본값 | 역할 | 조정 가이드 |
|---|---|---|---|
| `deadband_rad` | `0.004` | 정지 상태에서 Dynamixel 노이즈로 생기는 미세 떨림을 무시하는 불감대 | 팔이 가만히 있는데도 떨리면 `0.006~0.01`로 올리기; 반응이 둔하고 끈적하면 `~0.002`까지 낮추기 |
| `ema_alpha` | `0.4` | 지수이동평균(EMA) 스무딩 계수 — 낮을수록 부드럽지만 지연(lag) 증가 | 더 부드럽게 하려면 `0.2~0.3`으로 낮추기; 반응성을 높이려면 올리기 |
| `max_step_rad` | `0.0025` | 한 사이클당 허용 최대 관절 이동량(속도 상한) | **250 Hz 기준 `0.003` 이하 유지** — 이 값을 올리기 전에 `publish_rate_hz`부터 올릴 것 |
| `publish_rate_hz` | `250.0` | 브리지의 커맨드 발행 주기 | 더 민첩한 반응이 필요하면 `max_step_rad`를 올리기 전에 먼저 `500`으로 올리기 |

> 참고: 위 `publish_rate_hz`는 **브리지**의 발행 주기입니다. `gello_publisher`(리더암 읽기)는 별도로 `publish_rate_hz=30`으로 동작합니다.

### 속도-안전성 계산 (건드리지 말고 이해만 할 것)

UR 드라이버는 매 커맨드를 `delta / 0.002s` (500 Hz 서보 주기) 로 환산해 관절 속도 한도 `3.14159 rad/s` (180°/s) 와 비교합니다.

- 사이클당 안전 예산: `0.00628 rad` (=3.14159 × 0.002)
- 250 Hz 코얼레싱(발행 주기가 서보 주기의 절반) 고려 시 실질 예산: `0.00314 rad`
- 기본값 `max_step_rad=0.0025` → 단일 스텝 `1.25 rad/s`, 코얼레싱 시 최대 `2.5 rad/s` (한도 내 안전)
- 지속 속도 상한: `0.0025 × 250 = 0.625 rad/s`

`max_step_rad`를 임의로 올리면 이 여유분이 줄어들어 "External Control speed limit" 프로텍티브 스탑(순간 스냅)이 재발할 수 있습니다. 반드시 `publish_rate_hz`를 먼저 올려 예산을 확보한 뒤에만 `max_step_rad`를 조정하세요.

### 수정 → 반영 절차

1. YAML 편집:

```bash
nano /home/laptop3/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml
```

2. 리빌드 (colcon 캐시 덕분에 ~0.7초):

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./build_ur7e.sh
```

3. 루프 재실행 (시뮬레이션 또는 실물):

```bash
./run_ur7e_gello_sim.sh
# 또는
./run_ur7e_gello_real.sh
```

### 권장 튜닝 순서

1. **`deadband_rad` 먼저** — 정지 시 떨림이 없어질 때까지 조금씩 올린다 (0.004 → 0.006 → 0.008…).
2. **그다음 `ema_alpha`** — 그래도 움직임 중 미세 떨림/노이즈가 남으면 낮춘다 (0.4 → 0.3 → 0.2). 단, 너무 낮추면 반응 지연이 체감된다.
3. `max_step_rad`, `publish_rate_hz`는 반응성(속도) 문제일 때만, 위 순서를 마친 후 건드린다. `max_step_rad`는 250 Hz 기준 `0.003` 이하로 유지하고, 더 민첩하게 하려면 `publish_rate_hz`를 먼저 `500`으로 올린다.

---

## 6) 실행 전 프리플라이트 체크리스트 (안전)

실제 UR7e(192.168.10.11)를 GELLO로 조작하기 전, 매 세션마다 아래 체크리스트를 순서대로 확인한다. 하나라도 체크되지 않으면 실행하지 않는다.

### [pre-power] 전원 인가 전

- [ ] **E-STOP 즉시 접근 가능**: 로봇 E-stop 버튼(펜던트 또는 외부 버튼)을 손이 닿는 위치에서 확인했다.
- [ ] **작업공간(workspace) 클리어**: UR7e 도달 반경 내에 사람, 장애물, 케이블이 없다.
- [ ] **GELLO는 항상 수동(passive)**: GELLO 리더 암은 토크가 절대 걸리지 않는다 (driver가 torque OFF로 초기화하고 READ-only로만 사용). GELLO Dynamixel에 절대 토크를 걸지 말 것 — 실제로 움직이는 것은 팔로워(UR7e)뿐이다.
- [ ] **GELLO를 중립 자세 근처로 이동**: 로봇은 핸드셰이크 시 "현재 GELLO 자세"로 이동한다. 시작 전 GELLO를 대략 `start_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]` 근처(중립/안전 자세)에 두어 갑작스러운 큰 이동을 방지한다.

### [pre-launch] 실행 전 (pendant / 시뮬레이션)

- [ ] **펜던트 상태 = Real Robot (NOT Simulation)**: 하단 우측이 "Real Robot"인지 확인 (Simulation 모드면 실기 구동 안 됨).
- [ ] **Safety state = Normal**: 펜던트 안전 상태가 Normal인지 확인 (Protective Stop / Fault 상태면 먼저 `ur_unlock` 또는 펜던트에서 해제).
- [ ] **ROBOT_IP 확인**: 기본값은 `192.168.10.11`. 다르면 `ROBOT_IP=<ip>`로 명시적으로 오버라이드했다.
  ```bash
  echo $ROBOT_IP   # 비어있으면 스크립트 기본값 192.168.10.11 사용됨
  ```
- [ ] **시뮬레이션(mock) 드라이런 통과**: 실기 실행 전, 아래로 RViz mock 경로에서 GELLO 추종이 정상임을 먼저 확인했다.
  ```bash
  cd /home/laptop3/gello_software/ros2_ur_ws
  ./run_ur7e_gello_sim.sh
  ```
- [ ] **전체 스윕 경로가 장애물 없음(handshake move 대비)**: `move_to_start`는 로봇 현재 포즈 → GELLO 현재 포즈까지 조인트 공간을 단순 보간(interpolated)으로 이동하며 **장애물/충돌 인식이 전혀 없다**. 현재 포즈에서의 클리어런스만이 아니라, 두 포즈를 잇는 **전체 조인트 스윕 경로 전 구간**이 사람·장애물·케이블 없이 비어 있는지 확인했다.
  - **Method A (펜던트 Play 수동)** — 선행조건: 펜던트에서 External Control 프로그램을 Load하고 대기 중이어야 함. 실행:
    ```bash
    ./run_ur7e_gello_real.sh
    ```
    → 이후 펜던트에서 **Play** 버튼을 직접 눌러야 handshake가 진행된다.
  - **Method B (headless, Remote 모드)** — 선행조건: 펜던트가 **REMOTE** 모드(hamburger > Settings > System > Remote Control 활성화 + 우측상단 토글 REMOTE)이고 Play/Load가 회색(비활성)으로 표시되어야 함. 실행:
    ```bash
    HEADLESS=true ./run_ur7e_gello_real.sh
    ```
    → Play를 누르지 않아도 driver가 URScript를 직접 전송, scaled_jtc가 자동 활성화됨.
  - **Method A와 B를 한 세션에서 섞어 쓰지 말 것.**

### [during] 실행 중

- [ ] **첫 움직임은 약 t=8s 부근**: 런치 내 staggered TimerAction 순서상 driver(t0) → gello_publisher(t6) → gello_move_to_start(t8) 이므로, 약 8초 뒤 로봇이 GELLO 자세로 서서히 이동을 시작하는 것이 정상이다. 그 전에 로봇이 움직이면 비정상이니 즉시 정지.
  ```bash
  # 참고: handshake 성공 시에만 브릿지 + gripper가 이어서 시작됨
  # (RegisterEventHandler OnProcessExit(move_to_start), returncode==0 조건)
  ```
  이 단계에서 GELLO를 잡고 천천히, 예측 가능한 범위 내에서만 움직인다. 급격한 동작 금지.
- [ ] **핸드셰이크 스킵 금지**: gello_move_to_start 노드를 건너뛰거나 강제 종료하지 않는다 (건너뛰면 속도 제한 protective stop 위험).
- [ ] **세션 전체 동안 E-STOP 근접 유지**: 첫 t≈8s 이동 구간뿐 아니라 **세션이 끝날 때까지 내내** E-STOP에 손이 닿는 위치를 유지하고 작업 공간 상황을 계속 주시한다. 텔레옵이 활성인 동안 로봇은 GELLO를 따라 언제든 움직인다.

### [abort] 비상 정지 / 중단

- [ ] **1차 대응**: 실행 중인 터미널에서 **Ctrl-C** 로 launch를 즉시 중단한다.
- [ ] **2차/즉시 위험 시**: 주저 없이 물리 **E-STOP**을 누른다 (Ctrl-C보다 우선).
- [ ] **재개 절차**:
  - Method B(headless) 통신 끊김 후 재개 →
    ```bash
    source remote_helpers.sh
    ur_resend   # /io_and_status_controller/resend_robot_program
    ```
  - Method A(Play) 드롭 후 재개 →
    ```bash
    ur_play
    ```
  - Protective Stop 발생 시 →
    ```bash
    ur_unlock   # /dashboard_client/unlock_protective_stop
    ```
  - Remote → Local 전환은 반드시 **펜던트에서** 수행 (소프트웨어로 불가).

---

## 7) 트러블슈팅 + 빠른 명령 치트시트

### 7.1 자주 발생하는 문제와 해결법

**"Controller is not running" / 핸드셰이크(handshake) 실패**
- 원인: 펜던트에서 External Control 프로그램의 Play를 누르지 않았거나, 펜던트가 Remote 모드가 아님.
- 확인: `gello_move_to_start` 노드는 `scaled_joint_trajectory_controller`가 ACTIVE가 될 때까지 대기합니다. Play를 누르지 않으면 영원히 대기 상태로 멈춥니다.
- 해결:
  - **Method A (기본, Play 수동)**: 펜던트에서 External Control 프로그램의 ▶ Play를 누른다.
  - **Method B (헤드리스, Remote 모드)**: 펜던트를 REMOTE로 전환하고 아래처럼 헤드리스 실행:
    ```bash
    HEADLESS=true ./run_ur7e_gello_real.sh
    ```
  - Method A/B를 한 번의 실행에서 섞지 말 것 (DO NOT MIX).

**External Control 속도 제한(Speed Limit) 보호 정지**
- 원인: 브릿지가 첫 명령을 GELLO의 현재 포즈로 바로 점프시키면 스냅(snap)이 발생해 순간 속도가 조인트 한계(3.14159 rad/s)를 초과함. 이미 `config/ur7e_gello.yaml`에서 **로봇의 실제 `/joint_states`로 첫 명령을 시딩(seed)** 하고 `max_step_rad` 클램프로 해결되어 있어야 정상입니다.
- 만약 그래도 재발하면:
  1. `publish_rate_hz` 확인 (기본 250.0) — 낮추면 안 됨, 오히려 `max_step_rad`를 더 낮춰야 함.
  2. `max_step_rad`를 0.0025보다 낮춰본다 (예: 0.0015~0.002), 단 250 Hz 기준 0.003 이하 유지 권장.
  3. `gello_publisher`의 `publish_rate_hz=30`과 브릿지 `publish_rate_hz=250.0`이 설정대로인지 `config/ur7e_gello.yaml`에서 재확인.
  4. 시딩 로직(시작 시 GELLO 포즈가 아닌 로봇 실제 `/joint_states`에서 시작)이 정상 동작하는지 로그 확인.

**로봇이 전혀 움직이지 않음**
- `forward_position_controller`가 비활성(inactive) 상태일 수 있음 — 정상 부팅 시퀀스는 `scaled_joint_trajectory_controller`가 ACTIVE로 시작하고, 핸드셰이크 성공 후에만 `forward_position_controller`로 전환(activate)되고 `scaled_jtc`가 deactivate 됩니다.
- `gello_move_to_start`의 리턴코드(returncode)가 0이 아니면 브릿지와 그리퍼가 아예 시작되지 않습니다 (`RegisterEventHandler(OnProcessExit(...))`로 returncode==0 조건부 실행). 런치 로그에서 handshake 노드 종료 코드를 확인할 것.
- 컨트롤러 상태 확인:
  ```bash
  ros2 control list_controllers
  ```

**GELLO 스트림 끊김 / staleness 워치독 (arm holds, does not fault)**
- 동작: 브릿지(`gello_ur_bridge`)에는 staleness 워치독(`staleness_timeout_s=0.5`)이 있습니다. GELLO 스트림이 0.5초 넘게 stale해지면(예: USB 분리, 리더 노드 죽음) 브릿지는 **새 setpoint 발행을 멈추고** `"GELLO stale, holding — not publishing"` 로그를 남깁니다.
- 결과: `forward_position_controller`는 마지막으로 명령받은 포즈를 그대로 **홀드**합니다 — 팔은 그 자리에 멈춘 채 유지됩니다(감속 램프 없음, fault/protective stop 없음). GELLO 메시지가 다시 들어오면 브릿지가 **자동으로 재개**되어 실시간 추종을 이어갑니다.
- **배포 전 1회 테스트 (권장)**: 로봇이 안전하게 자세를 유지할 수 있는 상태에서, 세션 중 GELLO USB를 **분리**하고 팔이 드리프트/폴트 없이 **그 자리에 홀드**하는지 확인합니다. 그런 다음 다시 **연결**하여 추종이 자동으로 **재개**되는지 확인합니다.

**시리얼 포트 권한 오류 (GELLO 리더암 연결 안 됨)**
- 증상: `/dev/ttyUSB0` 접근 시 Permission denied.
- 원인: 사용자가 `dialout` 그룹에 없거나 그룹 반영 전(재로그인 필요).
- 해결:
  ```bash
  sudo usermod -aG dialout $USER   # 이후 재로그인 필요
  # 임시 우회 (재부팅/재연결 시마다 재적용 필요):
  sudo chmod 666 /dev/ttyUSB0
  ```
- 포트 확인:
  ```bash
  ls -l /dev/serial/by-id/
  ```

**dynamixel_sdk import 오류 (`ModuleNotFoundError: dynamixel_sdk`)**
- 원인: `GELLO_REPO_ROOT` 환경변수 미설정 (gello 드라이버 임포트 경로 필요) 또는 `dynamixel-sdk` 패키지 미설치.
- 해결:
  ```bash
  export GELLO_REPO_ROOT=/home/laptop3/gello_software
  pip install --user dynamixel-sdk
  ```
- sudo/apt 불필요, `--user` pip 설치로 충분 (검증된 버전 v4.0.5).

**캘리브레이션 체크섬(checksum) 경고**
- 예: `Calibration checksum mismatch` 류의 경고 로그.
- **무해(harmless)** — 조인트 텔레옵 정확도에는 영향 없음. 필요 시 아래로 재생성:
  ```bash
  ros2 launch ur_calibration calibration_correction.launch.py \
    robot_ip:=192.168.10.11 \
    target_filename:="${HOME}/ur7e_calibration.yaml"
  ```
- 이후 실행 스크립트에 `CALIB` 환경변수로 전달:
  ```bash
  CALIB=${HOME}/ur7e_calibration.yaml ./run_ur7e_gello_real.sh
  ```

**UR7e 메시(mesh) 경고 (RViz에 ur5e 모델 표시)**
- 원인: `ros-humble-ur` 2.8.1은 ur7e를 **부분 지원**만 함 — `ur_type:=ur7e`는 정상 로드/구동되고 조인트 텔레옵은 완전히 정확(ur5e/ur7e의 joint_limits 동일: 전 조인트 180 deg/s = 3.14159 rad/s)하지만, `config/ur7e` 의 visual_parameters가 `meshes/ur5e`를 가리키고 `ur7e_update_rate.yaml`이 없어서 경고가 뜸.
- **무해** — 시각적 표현(RViz 외형)에만 영향, 조인트 제어에는 영향 없음. 정확한 ur7e 외형/기구학이 필요하면 `ros-humble-ur` 2.13.2로 업그레이드.

---

### 7.2 빠른 명령 치트시트

```bash
# ---- 빌드 ----
cd /home/laptop3/gello_software/ros2_ur_ws
./build_ur7e.sh

# ---- 시뮬레이션 (실물 로봇 없이, RViz2 mock) ----
./run_ur7e_gello_sim.sh                     # source:=gello (실제 GELLO 리더로 mock 구동)

# ---- 실물 로봇: Method A (펜던트 Play 수동) ----
./run_ur7e_gello_real.sh

# ---- 실물 로봇: Method B (헤드리스, Remote 모드, Play 불필요) ----
HEADLESS=true ./run_ur7e_gello_real.sh

# ---- IP / 캘리브레이션 오버라이드 ----
ROBOT_IP=192.168.10.11 CALIB=/home/laptop3/ur7e_calibration.yaml ./run_ur7e_gello_real.sh

# ---- 대시보드 헬퍼 (source 해서 함수로 사용) ----
source remote_helpers.sh
ur_mode              # 현재 로봇 모드 조회 (GetRobotMode)
ur_program           # 현재 로드된 프로그램 확인
ur_powerup           # 파워온 + 브레이크 해제
ur_load PROG.urp     # 프로그램 로드 (Load {filename:})
ur_play              # Play (Trigger)
ur_stop              # Stop (Trigger)
ur_unlock            # 보호정지 해제 (unlock_protective_stop, Trigger)
ur_resend            # 헤드리스 드롭 복구 (resend_robot_program, Trigger)
ur_headless_driver   # 헤드리스 드라이버 관련 헬퍼
ur_help              # 전체 헬퍼 목록

# ---- 시리얼 권한 ----
sudo usermod -aG dialout $USER      # 영구 (재로그인 필요)
sudo chmod 666 /dev/ttyUSB0         # 임시 우회

# ---- dynamixel_sdk / GELLO 임포트 ----
export GELLO_REPO_ROOT=/home/laptop3/gello_software
pip install --user dynamixel-sdk

# ---- 캘리브레이션 재생성 ----
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.10.11 target_filename:="${HOME}/ur7e_calibration.yaml"

# ---- 컨트롤러 상태 점검 ----
ros2 control list_controllers

# ---- 헤드리스 드롭 후 복구 (Method B) ----
source remote_helpers.sh && ur_resend

# ---- Method A 드롭 후 복구 ----
source remote_helpers.sh && ur_play
```

**복구 규칙 요약**: 헤드리스(Method B) 연결이 끊기면 `ur_resend`, Method A(수동 Play)에서 끊기면 `ur_play`. Remote → Local 전환은 반드시 펜던트에서만 수행. Method A와 B를 한 실행에서 혼용하지 말 것.

---

## 관련 문서

- [GELLO_UR7E_ROS2_BRINGUP.md](./GELLO_UR7E_ROS2_BRINGUP.md) — 시뮬레이션/mock 경로 브링업 (검증됨)
- [GELLO_UR7E_REAL_ROBOT.md](./GELLO_UR7E_REAL_ROBOT.md) — 실로봇 런북(runbook)
- [UR7e_Remote_Control_ROS2.md](../../UR7e_Remote_Control_ROS2.md) — Remote 모드 레퍼런스
- [README.md](../../README.md) — 저장소 개요 및 시리얼 권한 워크어라운드
