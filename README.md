# GELLO for RWH

수동(passive) GELLO 리더 암을 손으로 움직여 **Franka Panda**와 **UR5e**를 원격 조종하는 RWH 랩용 GELLO 포크입니다. 조종 대상은 두 가지 경로로 동작합니다. (1) **MuJoCo 시뮬레이션** — Panda와 UR5e(+ Robotiq 2F-85 그리퍼)를 실시간 뷰어에서 1:1로 미러링, (2) **ROS2 / RViz2** — UR5e는 로컬 native Jazzy, Panda는 docker 이미지로 RViz에 표시. GELLO 자체는 **항상 읽기 전용**이며, 모터에 토크를 인가하지 않습니다.

원본 프로젝트는 [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)이며, 이 문서는 RWH 랩에서 실제로 쓰는 **Panda + UR** 경로만 다룹니다.

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
sudo usermod -aG dialout $USER   # 이후 로그아웃 후 다시 로그인 (또는 재부팅)
```

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

## ROS2 / RViz2

**UR5e**는 **로컬 native ROS2 Jazzy**(Ubuntu 24.04)로 실행합니다 — docker 없음. UR 드라이버가 최신 Ubuntu에서 잘 동작하기 때문입니다. `ur_gello_bringup` 패키지가 `use_mock_hardware`로 RViz2에 UR5e를 띄우며, `source:=fake`(로봇/GELLO 없이 sine sweep)와 `source:=gello`(실제 GELLO 연결, mock UR) 두 모드가 있습니다. 실제 UR5e는 별도 PC에 있고, 이 빌드 PC는 mock hardware로 RViz만 검증합니다. 빌드/실행 절차와 함정(orphan publisher, 포트, 전원)은 딥 문서에 있습니다.

**Franka**는 "어느 Franka인가"를 반드시 구분해야 합니다. 랩의 실제 Panda를 RViz로 보려면 조민제님의 **minje227 docker 이미지**로 `multipanda_ros2` 기반 launch를 실행하는 것이 현재의 turnkey 경로입니다 — docker를 쓰는 이유는 franka_ros2 v2.1.0 / libfranka 0.18.2가 ROS2 Humble / Ubuntu 22.04에 하드락되어 있기 때문입니다. **이 이미지는 이후 수정 + 재빌드/재배포가 필요한 "현재 경로"일 뿐 최종본이 아닙니다.** 또한 이 경로는 private docker 이미지와 물리 GELLO가 모두 있어야 동작합니다(오프라인·완전 로컬이 아님). 반면 리포의 `ros2/` 폴더는 upstream Franka 스택으로 **FR3 전용**(관절명 `fr3_joint1..7` 하드코딩, FR3 impedance gain)이며 RViz에 뜨는 것은 **랩의 Panda와 다른 로봇(FR3)**입니다 — 참고용 reference로만 봅니다.

### 어느 Franka인가 (용어 구분)

- **Panda (sim)** — MuJoCo 시뮬레이션의 Panda (위 시뮬레이션 섹션).
- **Panda (minje227 docker, RViz)** — 랩의 실제 Panda 모델을 RViz로 보는 현재 경로 (재빌드 예정).
- **FR3 (repo `ros2/`, upstream reference)** — upstream 스택이 띄우는 다른 로봇, 참고용.

| 경로 | 문서 |
|------|------|
| UR5e — 로컬 native Jazzy `ur_gello_bringup` (검증됨) | [`docs/ros2/GELLO_UR_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR_ROS2_BRINGUP.md) |
| UR5e — 실제 로봇 계획 (아직 미구현) | [`docs/ros2/GELLO_UR_ROS2_PLAN.md`](docs/ros2/GELLO_UR_ROS2_PLAN.md) |
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
ros2_ur_ws/                                   # UR5e ROS2 워크스페이스 (ur_gello_bringup)
ros2/                                         # upstream Franka 스택 (FR3 전용, 참고용)
docs/sim/                                     # 시뮬레이션 딥 문서
docs/ros2/                                    # ROS2 딥 문서
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
