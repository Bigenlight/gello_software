# GELLO for RWH

수동(passive) GELLO 리더 암을 손으로 움직여 **Franka Panda**와 **UR5e**를 원격 조종하는 RWH 랩용 GELLO 포크입니다. 조종 대상은 두 가지 경로로 동작합니다. (1) **MuJoCo 시뮬레이션** — Panda와 UR5e(+ Robotiq 2F-85 그리퍼)를 실시간 뷰어에서 1:1로 미러링, (2) **ROS2 / RViz2** — UR5e는 로컬 native Jazzy, Panda는 docker 이미지로 RViz에 표시. GELLO 자체는 **항상 읽기 전용**이며, 모터에 토크를 인가하지 않습니다.

원본 프로젝트는 [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)이며, 이 문서는 RWH 랩에서 실제로 쓰는 **Panda + UR** 경로만 다룹니다.

> **브랜치 안내:** 이 문서는 **`feat/gello-ur7e-humble-22.04`** 브랜치(Ubuntu **22.04** / ROS2 **Humble**, UR7e 라인) 기준입니다. 이 라인은 Panda·UR5e sim 자료가 있는 24.04 / **Jazzy** 라인(`feat/gello-panda-sim-teleop`)에서 갈라져 나왔고, 두 경로 모두 이 문서에 함께 담겨 있습니다. 실제 UR7e 턴키 경로는 아래 [UR7e — ROS2 Humble 턴키](#ur7e--ros2-humble-2204-턴키) 섹션을 참고하세요.

---

## Step 0 — 먼저 모터부터 확인 ⚠️

소프트웨어를 건드리기 전에, **모든 Dynamixel 서보가 정상 응답하는지 먼저 확인**하세요. [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)으로 스캔해서 각 모터의 **ID가 올바른지**, **baud가 57600인지** 확인합니다. 여기서 모터가 안 잡히면 이후 어떤 단계도 진행되지 않습니다.

- Wizard 사용법 자체는 이 문서에서 다루지 않습니다 (위 공식 링크 참고).
- **안전 불변식: GELLO는 passive read-only입니다. GELLO Dynamixel에 절대 토크를 인가하지 마세요.** 드라이버는 토크를 OFF로 초기화하며, 관절 각도를 읽기만 합니다.

---

## USB / Serial 세팅

GELLO는 U2D2(FTDI) USB-시리얼 어댑터로 연결됩니다. 포트를 찾는 방법:

```bash
ls -l /dev/serial/by-id/     # 안정적인 by-id 경로 (권장)
ls /dev/ttyUSB*              # enumeration 순서대로 잡히는 임시 경로
dmesg | grep -i tty          # 방금 꽂은 장치 확인
```

config는 `/dev/serial/by-id/usb-FTDI...` 형태의 **by-id 경로**를 사용합니다. `/dev/ttyUSBn`은 재부팅·재연결 시 번호가 바뀌지만 by-id 경로는 장치에 고정되어 있어 안정적입니다. 다른 어댑터를 쓰는 경우 yaml의 `agent.port`만 자신의 by-id 경로로 바꾸면 됩니다.

시리얼 접근 권한이 없으면(permission denied), 사용자를 `dialout` 그룹에 추가한 뒤 **로그아웃/로그인**하세요.

```bash
sudo usermod -aG dialout $USER   # 이후 로그아웃 후 다시 로그인 (또는 재부팅) — 영구 해결

# 로그아웃 없이 지금 당장 쓰고 싶으면 임시로 권한 부여 (재부팅/재연결 시 초기화됨):
sudo chmod 666 /dev/ttyUSB0      # 자신의 장치 노드로 (ls /dev/ttyUSB* 로 확인)
```

> `dialout` 그룹 추가가 영구적인 권장 방법이고, `chmod 666`은 재부팅하면 풀리는 임시 방편입니다.

> 물리적 Dynamixel 배선/조립은 이 문서 범위가 아닙니다 (하드웨어 문서 별도).

---

## 시뮬레이션 (메인)

Sim은 **docker를 쓰지 않습니다.** native `uv` venv(python 3.11)로 실행합니다. 이유: MuJoCo 인터랙티브 GLFW 뷰어(X11/GL) + 실시간 Dynamixel USB 조합은 docker 패스스루가 가장 취약한 케이스입니다. `uv`는 호스트 Ubuntu와 무관하게 py3.11을 제공하고, mujoco는 manylinux wheel로 배포되므로 native가 더 안정적입니다.

### One-time setup

bare 머신이라면 먼저 GL/뷰어 관련 apt 패키지를 설치합니다.

```bash
sudo apt install libgl1 libglfw3 libosmesa6
```

그다음 리포를 준비합니다.

```bash
git clone <repo> && cd gello_software
git submodule update --init --recursive    # mujoco_menagerie(로봇 XML) + DynamixelSDK

curl -LsSf https://astral.sh/uv/install.sh | sh   # uv가 없다면
uv venv --python 3.11 && source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python  # 필수 — dynamixel_sdk는 여기서만 설치됨
```

> 검증된 조합은 **numpy 2.3.5 + mujoco 3.10.0 + python 3.11**입니다. `numpy<2`로 핀하지 마세요 (known-good 환경을 깨뜨립니다).

### 실행

한 프로세스가 MuJoCo 서버 스레드와 GELLO 클라이언트 루프를 함께 돌립니다. `<your-display>`는 자신의 세션 X 디스플레이입니다 (일반 모니터라면 보통 `:0`; 이 headless 빌드 PC 예시에서는 `:1`).

```bash
# Panda
DISPLAY=<your-display> python experiments/launch_yaml.py --left-config-path configs/rwh_panda.yaml

# UR5e (+ Robotiq 2F-85)
DISPLAY=<your-display> python experiments/launch_yaml.py --left-config-path configs/rwh_ur.yaml
```

> `MUJOCO_GL`은 **설정하지 마세요** (인터랙티브 GLFW 뷰어를 씀). `MUJOCO_GL=glx`는 크래시합니다.

**무엇이 보여야 하는가:** MuJoCo 뷰어 창이 뜨고, GELLO를 손으로 움직이면 sim 로봇 팔이 1:1로 따라 움직입니다. GELLO 그리퍼 레버를 여닫으면 sim 그리퍼도 여닫힙니다.

### 상세 문서 (config 필드 표 · 캘리브레이션 · 트러블슈팅)

README에는 실행 one-liner만 두었습니다. 전체 config 필드, `joint_offsets` 캘리브레이션(`scripts/gello_get_offset.py`), 트러블슈팅은 아래 딥 문서를 참고하세요.

| 대상 | 문서 |
|------|------|
| Panda (MuJoCo sim) | [`docs/sim/GELLO_PANDA_SIM_TELEOP.md`](docs/sim/GELLO_PANDA_SIM_TELEOP.md) |
| UR5e + 2F-85 (MuJoCo sim) | [`docs/sim/GELLO_UR_SIM_TELEOP.md`](docs/sim/GELLO_UR_SIM_TELEOP.md) |

---

## UR7e — ROS2 Humble (22.04) 턴키

실제 **UR7e**(e-Series, PolyScope 5)를 수동 GELLO로 **관절 원격 조종**하는 턴키 경로입니다. 모든 진입점이 `ros2_ur_ws/` 아래 스크립트 하나로 감싸져 있으며, Ubuntu **22.04 + ROS2 Humble** 전용입니다(위 브랜치 안내 참고). 패키지는 `ur_gello_bringup`(기본 `ur_type=ur7e`)이고, sim과 real이 동일한 노드·캘리브레이션을 공유합니다.

> ### 안전 — 시작 전에 반드시 ⚠️
> - **GELLO는 항상 passive read-only** — 드라이버가 Dynamixel 토크를 OFF로 초기화하고 관절만 읽습니다. GELLO에 절대 토크를 인가하지 마세요.
> - **실제 UR7e는 물리적으로 움직입니다** (약 t≈8s에 기동하여 로봇이 GELLO의 현재 자세로 이동). E-STOP을 손 닿는 곳에 두고 작업공간을 비운 뒤, GELLO를 중립 근처로 잡고 시작하세요.
> - **핸드셰이크는 필수** — 부팅 시 `scaled_joint_trajectory_controller`가 로봇을 GELLO 현재 자세로 안전 보간 이동시킨 뒤 `forward_position_controller`로 전환합니다(`gello_move_to_start`). 이 과정을 건너뛰면 속도 리밋 protective stop이 납니다.

### 0. 사전 준비 (한 번)

```bash
pip install --user dynamixel-sdk                       # v4.0.5, sudo/apt 불필요
# 시리얼 접근 권한은 위 USB/Serial 섹션 참고 (dialout 그룹). 리더 = /dev/ttyUSB0 (FTBEO6QK)
export GELLO_REPO_ROOT=/home/laptop3/gello_software    # publisher가 gello 드라이버를 import (필수)
```

### 1. 빌드

```bash
cd ros2_ur_ws
./build_ur7e.sh        # colcon build --packages-select ur_gello_bringup + source install/setup.bash
```

### 2. 시뮬 (mock UR7e, RViz2 — 실제 로봇 없음)

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
cd ros2_ur_ws
./run_ur7e_gello_sim.sh        # source:=gello — 실제 물리 GELLO가 mock UR7e를 구동 (use_fake_hardware)
# 자가 테스트(사인파, GELLO 불필요)는 내부 source:=fake 경로를 사용합니다.
```

### 3. 실제 로봇 (UR7e 물리 이동)

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
cd ros2_ur_ws
./run_ur7e_gello_real.sh                  # Method A: 펜던트 External Control에서 Play를 누름
HEADLESS=true ./run_ur7e_gello_real.sh    # Method B: 펜던트 REMOTE 모드, Play 불필요 (권장, 핸즈프리)
# 환경변수: ROBOT_IP=192.168.10.11(기본), CALIB=/path/ur7e_calibration.yaml
```

- 대시보드 helper(load/play/stop/powerup/resend 등)가 필요하면 `source ros2_ur_ws/remote_helpers.sh` 후 `ur_help`.
- **Method A와 B를 한 실행에서 섞지 마세요.** 헤드리스 드롭 복구는 `ur_resend`, Method A 드롭은 `ur_play`.

> **드라이버 안전성 노트:** `gello_ur_bridge`는 첫 명령을 GELLO 자세가 아닌 로봇의 **실제 `/joint_states`**로 시드해 기동 스냅(과거 "External Control speed limit" protective stop의 원인)을 없애고, EMA smoothing · step limiting · **deadband 노이즈 게이트**(정지 시 Dynamixel 떨림 억제) · staleness guard를 적용한 뒤 `/forward_position_controller/commands`로 퍼블리시합니다.

> **ros-humble-ur 2.8.1 부분 지원:** `ur_type:=ur7e`로 **joint-space teleop은 완전히 정확**합니다(ur5e/ur7e 관절 리밋 동일). 다만 RViz visuals가 아직 ur5e mesh로 렌더되고(`config/ur7e` visual_parameters 미갱신), `ur7e_update_rate.yaml` 누락 경고(무해)가 있습니다. 정확한 ur7e visuals/kinematics가 필요하면 `ros-humble-ur >=2.13.2`로 업그레이드하세요.

### 런북 & 상세 문서

| 대상 | 문서 |
|------|------|
| UR7e — 턴키 셋업 CLI + 프리플라이트 체크리스트 | [`docs/ros2/GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md) |
| UR7e — sim/mock 브링업 (검증됨) | [`docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md) |
| UR7e — 2F-85 그리퍼 (Modbus over tool-comm, 검증됨) | [`docs/ros2/GELLO_UR7E_GRIPPER.md`](docs/ros2/GELLO_UR7E_GRIPPER.md) |
| UR7e — 실제 로봇 런북 (코드 완료·실기 미검증) | [`docs/ros2/GELLO_UR7E_REAL_ROBOT.md`](docs/ros2/GELLO_UR7E_REAL_ROBOT.md) |
| UR7e — Remote Control 모드 (헤드리스/대시보드) 레퍼런스 | [`UR7e_Remote_Control_ROS2.md`](UR7e_Remote_Control_ROS2.md) |

---

## ROS2 / RViz2

**UR5e**는 **로컬 native ROS2 Jazzy**(Ubuntu 24.04)로 실행합니다 — docker 없음. UR 드라이버가 최신 Ubuntu에서 잘 동작하기 때문입니다. `ur_gello_bringup` 패키지가 `use_mock_hardware`로 RViz2에 UR5e를 띄우며, `source:=fake`(로봇/GELLO 없이 sine sweep)와 `source:=gello`(실제 GELLO 연결, mock UR) 두 모드가 있습니다. 실제 UR5e는 별도 PC에 있고, 이 빌드 PC는 mock hardware로 RViz만 검증합니다. 빌드/실행 절차와 함정(orphan publisher, 포트, 전원)은 딥 문서에 있습니다.

**UR7e**는 같은 `ur_gello_bringup` 패키지로 **ROS2 Humble**(Ubuntu 22.04)에서 실행하며, sim/mock부터 실제 로봇까지 턴키 스크립트로 감싸져 있습니다 — 전용 절차·런북은 위 [UR7e — ROS2 Humble 턴키](#ur7e--ros2-humble-2204-턴키) 섹션을 참고하세요.

**Franka**는 "어느 Franka인가"를 반드시 구분해야 합니다. 랩의 실제 Panda를 RViz로 보려면 조민제님의 **minje227 docker 이미지**로 `multipanda_ros2` 기반 launch를 실행하는 것이 현재의 turnkey 경로입니다 — docker를 쓰는 이유는 franka_ros2 v2.1.0 / libfranka 0.18.2가 ROS2 Humble / Ubuntu 22.04에 하드락되어 있기 때문입니다. **이 이미지는 이후 수정 + 재빌드/재배포가 필요한 "현재 경로"일 뿐 최종본이 아닙니다.** 또한 이 경로는 private docker 이미지와 물리 GELLO가 모두 있어야 동작합니다(오프라인·완전 로컬이 아님). 반면 리포의 `ros2/` 폴더는 upstream Franka 스택으로 **FR3 전용**(관절명 `fr3_joint1..7` 하드코딩, FR3 impedance gain)이며 RViz에 뜨는 것은 **랩의 Panda와 다른 로봇(FR3)**입니다 — 참고용 reference로만 봅니다.

### 어느 Franka인가 (용어 구분)

- **Panda (sim)** — MuJoCo 시뮬레이션의 Panda (위 시뮬레이션 섹션).
- **Panda (minje227 docker, RViz)** — 랩의 실제 Panda 모델을 RViz로 보는 현재 경로 (재빌드 예정).
- **FR3 (repo `ros2/`, upstream reference)** — upstream 스택이 띄우는 다른 로봇, 참고용.

| 경로 | 문서 |
|------|------|
| UR5e — 로컬 native Jazzy `ur_gello_bringup` (검증됨) | [`docs/ros2/GELLO_UR_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR_ROS2_BRINGUP.md) |
| UR5e — 실제 로봇 계획 (아직 미구현) | [`docs/ros2/GELLO_UR_ROS2_PLAN.md`](docs/ros2/GELLO_UR_ROS2_PLAN.md) |
| UR7e (Humble, sim/mock·실제 로봇 턴키) | 위 [UR7e — ROS2 Humble 턴키](#ur7e--ros2-humble-2204-턴키) 섹션 참고 |
| Franka — Panda를 RViz로 (minje227 docker, 현재 경로) | [`docs/ros2/GELLO_FRANKA_RVIZ.md`](docs/ros2/GELLO_FRANKA_RVIZ.md) |
| FR3-vs-Panda 비교 reference | [`docs/ros2/GELLO_ROS2_CONTROL_REFERENCE.md`](docs/ros2/GELLO_ROS2_CONTROL_REFERENCE.md) |

---

## Repo Layout

```
README.md                                   # 이 문서 (front door)
configs/rwh_panda.yaml                       # Panda sim config
configs/rwh_ur.yaml                          # UR5e sim config
experiments/launch_yaml.py                   # sim 실행 엔트리포인트
scripts/gello_get_offset.py                  # joint_offsets 캘리브레이션
gello/                                        # GELLO 코어 패키지 (드라이버·로봇·에이전트)
ros2_ur_ws/                                   # UR5e/UR7e ROS2 워크스페이스 (ur_gello_bringup; 기본 ur_type=ur7e)
  build_ur7e.sh / run_ur7e_gello_sim.sh       #   UR7e 턴키 스크립트 (빌드 · sim/mock)
  run_ur7e_gello_real.sh / remote_helpers.sh  #   UR7e 실제 로봇 · 대시보드 헬퍼
ros2/                                         # upstream Franka 스택 (FR3 전용, 참고용)
UR7e_Remote_Control_ROS2.md                   # UR7e Remote Control(헤드리스/대시보드) 레퍼런스
docs/sim/                                     # 시뮬레이션 딥 문서
docs/ros2/                                    # ROS2 딥 문서 (UR7e 셋업 CLI·브링업·실제 로봇 런북 포함)
docs/reference/                              # 동료 셋업 가이드 등
```

> **안전 불변식 (재강조): GELLO는 언제나 passive read-only입니다. GELLO Dynamixel에 절대 토크를 인가하지 마세요.** 드라이버는 토크 OFF로 초기화하고 관절만 읽습니다.

---

## Credits & License

원본 GELLO 프로젝트: [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) ([Project Website](https://wuphilipp.github.io/gello_site/), [Hardware Repository](https://github.com/wuphilipp/gello_mechanical)).

> 이 RWH 문서는 **Panda + UR만** 다룹니다. upstream의 YAM / xArm 관련 내용은 제거했으며, 필요하면 [upstream README](https://github.com/wuphilipp/gello_software)를 참고하세요.

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

이 프로젝트는 'FACTR Teleop: Low-Cost Force-Feedback Teleoperation' (Apache-2.0)의 컴포넌트를 사용합니다. https://github.com/RaindragonD/factr_teleop/ 참고.
