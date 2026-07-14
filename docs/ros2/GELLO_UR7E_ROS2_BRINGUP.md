# GELLO → UR7e ROS2 텔레오퍼레이션 (로컬 · Humble · mock/fake · 검증됨)

이 문서는 **ROS2 Humble + `ros-humble-ur` 2.8.1**(도커 없음) 환경에서 GELLO 리더 암으로 UR7e를 텔레오퍼레이션하는 파이프라인을, **이 PC에서 mock hardware(`source:=fake`)로 검증하는** 경로를 다룹니다. 사용하는 패키지는 `ros2_ur_ws/src/ur_gello_bringup` 입니다.

> **이 문서가 검증하는 것**: 이 PC에는 물리 GELLO가 **연결되어 있지 않으므로**, `ur7e_gello_rviz.launch.py`를 **mock UR7e(`use_fake_hardware:=true`)** 위에서 `source:=fake`로 띄워 브릿지 · 컨트롤러 · 토픽 · RViz2까지의 파이프라인 전체를 검증합니다. 여기가 **로컬에서 실제로 검증된(verified-here) 경로**입니다.

> **실물 UR7e로 가려면**: 실제 GELLO 리더 암 + 실 UR7e 구동 절차는 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)를 따르세요. 그 경로는 이 PC에서는 검증할 수 없습니다.

> **UR5e 원본과의 관계**: UR5e용 원본 경로(`ur_gello_rviz.launch.py`, `config/ur_gello.yaml`)와 노드 코드는 그대로 두고, ur7e 전용 launch/config만 새로 추가한 것입니다. 설계 배경은 [`GELLO_UR_ROS2_BRINGUP.md`](./GELLO_UR_ROS2_BRINGUP.md) 및 [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md)를 참고하세요.

---

## TL;DR

이 PC(GELLO 미연결)에서 GELLO → UR7e 파이프라인을 mock으로 검증하는 최소 절차:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur_gello_bringup && source install/setup.bash
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false
```

별도 터미널에서 검증(각 터미널마다 `source /opt/ros/humble/setup.bash && source install/setup.bash` 먼저):

```bash
ros2 control list_controllers          # joint_state_broadcaster, forward_position_controller 둘 다 active
ros2 topic hz /forward_position_controller/commands   # ~250 Hz
ros2 param get /robot_state_publisher robot_description | grep -o ur7e   # ur7e
```

- **핵심**: `source:=fake`는 dynamixel_sdk도, GELLO 시리얼도, `GELLO_REPO_ROOT`도 필요 없습니다.
- **UR5e → UR7e 변경점**: 노드 로직은 동일(joint_limits 동일). 새 launch/config에서 `ur_type:=ur7e`, 그리고 실기 안전을 위해 bridge를 velocity-safe·anti-tremor로 조정: `max_step_rad` **0.0025**, `deadband_rad` **0.004**(노이즈 게이트), `publish_rate_hz` **250.0**, 그리고 첫 명령을 로봇 실제 `/joint_states`에서 seed(스타트업 스냅 방지).
- **안전 불변식**: GELLO는 항상 passive read-only — Dynamixel에 절대 토크를 걸지 않음.

---

## 목차

- [안전 — GELLO는 항상 passive read-only](#안전--gello는-항상-passive-read-only)
- [1. Overview — 무엇이 실행되는가 (ur7e, ROS2 Humble)](#1-overview--무엇이-실행되는가-ur7e-ros2-humble)
- [2. 사전 준비 (Prerequisites, ROS2 Humble)](#2-사전-준비-prerequisites-ros2-humble)
- [3. 빌드 & 실행 (이 PC — fake/mock 경로, 검증됨)](#3-빌드--실행-이-pc--fakemock-경로-검증됨)
- [4. 설정 파일 레퍼런스 — `config/ur7e_gello.yaml`](#4-설정-파일-레퍼런스--configur7e_gelloyaml)
- [5. 트러블슈팅](#5-트러블슈팅)
- [6. 일시정지·재개 (두 손 자유 씬 리셋)](#6-일시정지재개-두-손-자유-씬-리셋)

---

## 안전 — GELLO는 항상 passive read-only

GELLO는 **수동(passive) 모션캡처 리더 암**입니다. Dynamixel 모터에 **절대 토크를 걸지 않습니다.** 이 원칙은 UR5e든 UR7e든, 시뮬레이션(mock)이든 실로봇이든 예외 없이 항상 적용됩니다.

- `gello_publisher_node.py`는 드라이버 초기화 시 **토크를 OFF 상태로 초기화**하고 관절 각도를 **읽기만** 합니다.
- `fake_gello_node.py`(source:=fake) 경로는 GELLO 하드웨어 자체를 쓰지 않으므로 토크 문제와 무관합니다.
- 노드 종료 시에도 GELLO 쪽에서 끌 전원 자체가 없습니다 — "GELLO is passive; nothing to power down."

즉, 이 문서의 어떤 절차를 따라도 GELLO 모터에 힘이 들어가는 경로는 존재하지 않습니다. GELLO는 손으로 자유롭게 움직이는 입력 장치이며, 로봇(UR7e)이 그 자세를 **따라가는** 방향으로만 힘이 전달됩니다.

---

## 1. Overview — 무엇이 실행되는가 (ur7e, ROS2 Humble)

이 런북은 **ROS2 Humble + `ros-humble-ur` 2.8.1** 환경에서, `ur7e_gello_rviz.launch.py`로 GELLO → UR7e 텔레오퍼레이션을 mock hardware(`source:=fake` 또는 `source:=gello`) 위에서 검증하는 경로를 다룹니다. UR5e용 원본 경로(`ur_gello_rviz.launch.py`, `config/ur_gello.yaml`)는 그대로 두고, ur7e 전용 launch/config만 새로 추가합니다. 노드 코드 자체(`gello_publisher_node.py`, `gello_ur_bridge_node.py`, `fake_gello_node.py`, `robotiq_urcap_node.py`)는 UR5e와 UR7e의 joint_limits가 동일하므로 **변경 없이 그대로 재사용**됩니다.

```
[GELLO source]                   [bridge]                          [mock UR7e]                    [RViz2]
gello_publisher   ─(30Hz)──▶   gello_ur_bridge   ─(250Hz)──▶   forward_position_    ──▶  /joint_states ──▶ RViz2
 또는 fake_gello                 EMA + deadband +                controller (mock,          (UR7e 모델이 움직임)
 (source:=gello|fake)            step-clamp + staleness           use_fake_hardware)
                                 watchdog, seed from
                                 /joint_states

gello_publisher ──▶ /gripper/gripper_client/target_gripper_width_percent ──▶ robotiq_urcap (기본: 로그만)
```

`ur7e_gello_rviz.launch.py`는 다음 순서로 노드를 띄웁니다.

1. UR 드라이버(`ur_control.launch.py`, `ur_type:=ur7e`, `use_fake_hardware:=true` + `use_mock_hardware:=true`, `headless_mode:=true`)를 include → `joint_state_broadcaster`, `forward_position_controller`가 활성화된 mock UR7e가 뜸.
2. RViz2(`launch_rviz` 인자, 기본 true).
3. **~5초 TimerAction 이후**: `source:=fake`면 `fake_gello`, `source:=gello`면 `gello_publisher`를 실행하고, 곧이어 `gello_ur_bridge`를 실행. (컨트롤러가 먼저 완전히 spawn된 뒤에 소스/브리지가 명령을 흘려보내도록 지연시킨 것.)

### 노드별 역할

| 노드 (executable) | 역할 |
| --- | --- |
| `gello_publisher` | 물리 GELLO를 시리얼로 읽어 6개 관절을 UR 순서의 `sensor_msgs/JointState`로 `/gello/joint_states`에 발행(기본 30Hz). 그리퍼 폭은 `std_msgs/Float32`(0..1)로 `/gripper/gripper_client/target_gripper_width_percent`에 발행. **읽기 전용, 토크 OFF.** `source:=gello`에서만 사용, 리포 루트를 `GELLO_REPO_ROOT` 환경변수로 알려줘야 함(2절 참고). |
| `fake_gello` | 하드웨어 없이 파이프라인을 시험하는 **테스트 전용** 노드. `gello_publisher`와 동일한 토픽에 느린 sine sweep을 발행. GELLO/UR 실물이 전혀 필요 없음. **이 PC에서 GELLO 시리얼이 연결되어 있지 않으므로 `source:=fake`가 기본 검증 경로.** |
| `gello_ur_bridge` | `/gello/joint_states`를 구독해 이름 기준으로 UR 관절 순서로 재정렬한 뒤, EMA 스무딩(`ema_alpha`) + deadband 노이즈 게이트(`deadband_rad`) + step-clamp(`max_step_rad`) + staleness watchdog(`staleness_timeout_s`)을 적용하여 `/forward_position_controller/commands`(`std_msgs/Float64MultiArray`, 6 doubles)로 **250Hz** 상향 샘플링해 발행. **첫 명령은 GELLO 포즈가 아니라 로봇 실제 `/joint_states`(`joint_states_topic`)에서 seed**하여 스타트업 스냅(→ External Control 속도 제한 protective stop)을 방지. |
| `robotiq_urcap` | Robotiq 2F-85 그리퍼를 UR URCap 소켓(기본 포트 63352)으로 구동. 이 PC에는 그리퍼가 없으므로 기본 `connect_on_start=False` — 소켓을 열지 않고 로그만 남김. |
| `gello_move_to_start` *(신규, 실로봇 경로 전용)* | **수렴-게이트 핸드셰이크** 노드. `scaled_joint_trajectory_controller`가 active가 되길 기다린 뒤(External Control Play), **라이브** GELLO 포즈와 로봇 실제 `/joint_states`를 매 반복 다시 읽어 gap 기반 길이(`T=max(gap/chase_v_budget, min_traj_duration)`)의 캐치업 트래젝토리를 반복 전송하고, 관절별 `\|라이브GELLO−실제\| ≤ chase_tol`이 `chase_dwell_s` 유지될 때에만 `/controller_manager/switch_controller`(STRICT)로 `forward_position_controller`로 전환한다. (`chase_hard_limit` 초과 gap 거부, `chase_timeout_s` 내 미수렴/stale 리더면 exit 1 = 스위치 안 함.) 스위치 성공 후 `resume_bridge:=true`로 브릿지 `~/resume`를 호출해 스트리밍을 시작. 목표는 `start_joints`가 아니라 구독한 라이브 GELLO 자세임. **mock 경로(이 PC의 검증 대상)에서는 사용하지 않음** — `ur7e_gello_rviz.launch.py`는 처음부터 `forward_position_controller`를 active로 올리므로 필요 없음. 파라미터 표는 [4절](#4-설정-파일-레퍼런스--configur7e_gelloyaml), 절차는 실로봇 런북([`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)) 참고. |

### 토픽 계약

| 토픽 | 타입 | 비고 |
| --- | --- | --- |
| `/gello/joint_states` | `sensor_msgs/JointState` | 6 관절, UR 순서, radian |
| `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` | 0..1 (0=open, 1=closed) |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 6 doubles, UR 순서 |

UR 관절 순서(UR5e/UR7e 동일):
`[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`

> UR5e와 UR7e는 joint_limits가 동일(모든 관절 max_velocity 180°/s ≈ 3.14159 rad/s)하므로 bridge/publisher 노드 코드는 변경할 필요가 없습니다. 다만 실로봇 velocity-safe·anti-tremor를 위해 ur7e 설정(`config/ur7e_gello.yaml`)의 bridge는 `max_step_rad=0.0025`, `deadband_rad=0.004`, `publish_rate_hz=250.0`로 조정되어 있고, 첫 명령을 로봇 실제 `/joint_states`에서 seed합니다(mock에서는 velocity 제한이 물리적으로 무의미하므로 무해). 자세한 이유는 [4절](#4-설정-파일-레퍼런스--configur7e_gelloyaml)을 참고하세요.

---

## 2. 사전 준비 (Prerequisites, ROS2 Humble)

이 패키지를 빌드/실행하기 전에 아래 항목을 먼저 확인·설치한다.

### 2-1. 시스템 / ROS2 배포판

- OS: Ubuntu 22.04
- ROS2: Humble (다른 배포판에서는 검증되지 않음)
- Python: **시스템 python3.10** 사용 (venv/conda 등 별도 가상환경 금지 — `rclpy` 및 ROS2 Humble 파이썬 바인딩은 시스템 python3.10과 함께 설치되어 있어야 정상 동작한다.)

### 2-2. 필수 apt 패키지 설치

```bash
sudo apt update
sudo apt install ros-humble-ur
pip install --user dynamixel-sdk   # source:=gello 경로에서만 필요 (v4.0.5, sudo/apt 불필요)
```

- `ros-humble-ur`: UR 드라이버/설명(description) 패키지. 설치된 2.8.1 버전에 `ur7e` 구성(`config/ur7e`)이 이미 포함되어 있으므로 **버전을 임의로 올리지 않는다.**
- `dynamixel-sdk`: GELLO 리더암(Dynamixel 서보)과 통신하기 위한 SDK. `pip install --user dynamixel-sdk`로 설치(v4.0.5, **sudo/apt 불필요**). **`source:=fake` 경로(이 머신에서 검증하는 경로)에서는 필요 없다.** 실제 GELLO 하드웨어를 연결해 `source:=gello`로 실행할 때만 필요하다.

### 2-3. ROS2 환경 소싱

매 새 터미널에서 워크스페이스 빌드/실행 전에 반드시 실행:

```bash
source /opt/ros/humble/setup.bash
```

패키지 빌드 후에는 오버레이도 함께 소싱한다:

```bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
```

### 2-4. `source:=gello` (실제 GELLO 리더암) 사용 시 필수 환경 변수

`gello_publisher_node.py`에는 리포지토리 루트를 찾지 못했을 때 사용하는 하드코딩된 폴백 경로 `/home/theo_lab/gello_software`가 들어 있다. **이 머신의 실제 경로와 다르므로**, GELLO 실물 하드웨어를 사용하는 모든 실행(`source:=gello`) 전에 반드시 아래 환경 변수를 설정한다:

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
```

- 이 변수를 설정하지 않으면 노드가 잘못된 경로(`/home/theo_lab/...`)를 참조하여 캘리브레이션/설정 파일을 찾지 못하고 실패할 수 있다.
- `source:=fake` 경로는 실제 GELLO 장치를 읽지 않으므로 이 환경 변수가 필요 없다.
- 노드 코드 자체(`gello_publisher_node.py`)는 FROZEN 상태이므로 수정하지 않는다 — 반드시 환경 변수로 우회한다.

### 2-5. GELLO 시리얼 포트 권한 (dialout / chmod 666)

GELLO 리더암은 아래 시리얼 포트로 연결된다 (물리 리더암 FTBEO6QK 기준, `config/ur_gello.yaml`과 동일):

```
/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
```

리눅스에서 일반 사용자 계정이 이 시리얼 장치에 접근하려면 `dialout` 그룹 권한이 필요하다:

```bash
sudo usermod -aG dialout $USER
# 이후 로그아웃/재로그인 (또는 재부팅) 필요
```

그룹 적용이 번거롭거나 즉시 테스트가 필요한 경우, 임시방편으로 장치 파일 권한을 직접 여는 방법도 있다 (재부팅/재연결 시마다 초기화되므로 매번 재실행 필요, 운영 환경에는 `dialout` 그룹 방식을 권장):

```bash
sudo chmod 666 /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
```

이 항목 역시 `source:=gello`(실제 GELLO 하드웨어) 경로에만 해당하며, `source:=fake` 검증 경로에서는 불필요하다.

### 2-6. 안전 불변식 (모든 문서 공통 — 필독)

**GELLO 리더암은 항상 수동(passive) 읽기 전용 장치다. GELLO의 Dynamixel 서보에는 어떤 경우에도 토크를 인가하지 않는다.** 드라이버는 기동 시 토크를 OFF로 초기화하며 오직 조인트 값을 읽기만 한다. 이 불변식은 fake/real 어떤 실행 경로에서도 예외 없이 유지된다.

---

## 3. 빌드 & 실행 (이 PC — fake/mock 경로, 검증됨)

> 이 PC에는 물리 GELLO가 연결되어 있지 않습니다. 아래 절차는 **mock hardware**(`source:=fake`)로 파이프라인 전체(브릿지 · 컨트롤러 · RViz2)를 검증하는 경로입니다. 실물 GELLO + 실 UR7e 절차는 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)를 참고하세요.

### 3-1. 빌드

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

### 3-2. 실행 — mock hardware + RViz2 (기본)

```bash
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake
```

- `source:=fake` → `fake_gello_node`가 `/gello/joint_states`에 sine sweep을 발행합니다(물리 GELLO 불필요).
- `ur_type` 기본값은 `ur7e`, `use_fake_hardware:=true`(및 호환용 `use_mock_hardware:=true`)가 기본 적용됩니다.
- RViz2 창이 함께 뜨고, UR7e 모델이 fake 시퀀스를 따라 움직이면 정상입니다.

### 3-3. 헤드리스 실행 (RViz2 없이, CI/원격 세션용)

```bash
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false
```

### 3-4. 실물 GELLO를 이 PC에 연결해 fake 대신 실제 리더로 구동하고 싶다면

> 주의: 이 PC 검증 시나리오는 원칙적으로 GELLO가 없다고 가정합니다. 만약 실물 GELLO(FTBEO6QK)를 이 PC에 임시로 연결해 시험한다면:

```bash
pip install --user dynamixel-sdk   # dynamixel_sdk 미설치 상태이므로 gello 소스 사용 시 필요 (sudo/apt 불필요)
export GELLO_REPO_ROOT=/home/laptop3/gello_software   # gello_publisher_node의 하드코딩 fallback 경로(/home/theo_lab/gello_software) 우회
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello
```

- `source:=gello` → `gello_publisher` 실행(하드웨어 시리얼 포트: `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0`, `config/ur7e_gello.yaml`에 정의됨).
- **안전 불변식**: GELLO는 이 경로에서도 항상 passive read-only입니다 — Dynamixel에 절대 토크를 걸지 않습니다(드라이버가 토크 OFF로 초기화 후 읽기만 수행).

### 3-5. 검증 체크리스트

launch 후 별도 터미널에서(각 터미널마다 `source /opt/ros/humble/setup.bash && source install/setup.bash` 먼저 실행):

```bash
# (1) 컨트롤러 상태 — joint_state_broadcaster, forward_position_controller 둘 다 active
ros2 control list_controllers
```
기대 출력 예시:
```
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
forward_position_controller[position_controllers/JointGroupPositionController] active
scaled_joint_trajectory_controller[...] inactive
```

```bash
# (2) 브릿지가 250Hz로 명령을 상향 샘플링해 발행하는지 확인
ros2 topic hz /forward_position_controller/commands
```
기대: `average rate: 250.0` 근방(±수 Hz 오차는 정상).

```bash
# (3) 로드된 로봇 모델이 ur7e인지 확인 — 드라이버 로그가 가장 확실하다.
#     (robot_description grep은 부정확: 2.8.1의 ur7e 서술이 ur5e 메시를 참조하므로
#      grep ur5e도 매칭됨 — 아래 §5 "알려진 제약" 참고. 하드웨어 로드 로그로 판정할 것.)
grep -m1 "Loading hardware" ~/.ros/log/latest/ros2_control_node*.log 2>/dev/null || \
  echo "launch 콘솔에서 \"Loading hardware 'ur7e'\" 라인을 확인"
```
기대: `Loading hardware 'ur7e'` (드라이버가 ur7e HW를 로드).

세 가지 모두 통과하면 이 PC에서의 fake/mock 경로 검증이 완료된 것입니다.

> **실측 검증(2026-07-03, 이 PC).** `source:=fake launch_rviz:=false`로 실행 시: 드라이버가 `Loading hardware 'ur7e'` → `joint_state_broadcaster` **active** + `forward_position_controller` **active** + `scaled_joint_trajectory_controller` inactive, `/forward_position_controller/commands` **~250 Hz**(현재 config `publish_rate_hz=250.0`), fake 소스 `/gello/joint_states` **29.99 Hz**. 파이프라인 정상 확인됨.

### 3-6. 명령 요약

| 시나리오 | 명령 |
| --- | --- |
| 빌드 | `colcon build --packages-select ur_gello_bringup` |
| RViz2 포함(fake) | `ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake` |
| 헤드리스(fake) | `ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false` |
| 실물 GELLO 소스 | `GELLO_REPO_ROOT=/home/laptop3/gello_software ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello` |
| 컨트롤러 확인 | `ros2 control list_controllers` |
| 발행 주기 확인 | `ros2 topic hz /forward_position_controller/commands` |
| 로봇 모델 확인 | `ros2 param get /robot_state_publisher robot_description \| grep -o ur7e` |

---

## 4. 설정 파일 레퍼런스 — `config/ur7e_gello.yaml`

`ur7e_gello.yaml`은 `config/ur_gello.yaml`(UR5e용, FROZEN)과 **동일한 물리 리더 암(FTBEO6QK)의 캘리브레이션 값을 그대로 재사용**하되, `gello_ur_bridge`를 UR7e 실기용으로 velocity-safe·anti-tremor로 조정한 버전입니다.

```yaml
gello_publisher:
  ros__parameters:
    port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0"
    joint_ids: [1, 2, 3, 4, 5, 6]
    joint_offsets: [0.0, 1.571, 4.712, 4.712, 4.712, 0.0]   # w3 원래 3.142; J2/J3 remount 후 recalib
    joint_signs: [1, 1, -1, 1, 1, 1]
    gripper_config: [7.0, 210.649609375, 168.849609375]
    start_joints: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]
    publish_rate_hz: 30.0

gello_ur_bridge:
  ros__parameters:
    filter_type: "one_euro"          # anti-tremor 주 필터 (ema로 fallback 가능)
    one_euro_min_cutoff: 1.0
    one_euro_beta: 2.0
    one_euro_d_cutoff: 1.0
    ema_alpha: 0.4                    # filter_type: ema 일 때만 적용
    max_step_rad: 0.0025             # 사이클당 슬루 제한(≈속도 캡)
    deadband_rad: 0.004              # ema 경로 노이즈 게이트
    staleness_timeout_s: 0.5
    soft_start_s: 0.7                # 모든 (재)시딩 후 슬루 클램프 램프 (anti-snap)
    start_paused: false              # 통합 launch가 true로 오버라이드(선기동-일시정지)
    publish_rate_hz: 250.0
    joint_states_topic: "/joint_states"
    # resume_align_tol 은 yaml에 없음 → 노드 기본값 0.05 사용

gello_move_to_start:                 # 실로봇 경로 전용 (mock 미사용)
  ros__parameters:
    source_controller: "scaled_joint_trajectory_controller"
    target_controller: "forward_position_controller"
    trajectory_duration: 5.0
    arrival_tolerance: 0.05
    start_mode: "gello"              # 수렴-게이트 추격 후 자동 인계
    chase_tol: 0.025
    chase_dwell_s: 0.4
    chase_v_budget: 0.5
    min_traj_duration: 0.5
    chase_hard_limit: 4.0
    chase_timeout_s: 30.0
    gello_staleness_s: 0.5
    # resume_bridge / bridge_resume_service 는 통합 launch가 주입(true / /gello_ur_bridge/resume)
    alignment_tolerance: 0.2         # init_align 모드 전용
    alignment_hard_limit: 0.5
    alignment_timeout: 0.0
```

### 필드 설명

| 파라미터 | 값 | 설명 |
| --- | --- | --- |
| `port` | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` | GELLO 리더 암의 USB-시리얼 고정 경로. `by-id` 심볼릭 링크를 사용하므로 어느 USB 포트에 꽂아도 동일하게 인식됨. |
| `joint_ids` | `[1, 2, 3, 4, 5, 6]` | Dynamixel 버스 상의 관절 모터 ID (그리퍼 ID 7은 `gripper_config`에서 별도 처리). |
| `joint_offsets` | `[3.142, 1.571, 4.712, 4.712, 4.712, 3.142]` | 각 모터의 0-rad 기준 오프셋(라디안). `config/ur_gello.yaml`(UR5e)에서 **그대로 재사용**한 값 — 동일 리더 암이므로 재보정 불필요, **하드웨어를 재조립하지 않는 한 변경 금지**. |
| `joint_signs` | `[1, 1, -1, 1, 1, 1]` | 모터 회전 방향과 UR 관절 방향의 부호 일치용. J3만 반전. |
| `gripper_config` | `[7.0, 210.649609375, 168.849609375]` | `[그리퍼 모터 ID, 열림 raw값, 닫힘 raw값]`. **반드시 모두 double**이어야 함 — rcl yaml 파서가 하나의 배열에 int/double이 섞이면 파싱을 거부함(ID 7도 7.0으로 기입, 노드 내부에서 다시 int로 라운드트립됨). |
| `start_joints` | `[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]` | **`gello_publisher` 전용** 파라미터(Dynamixel 리더 초기 포즈 참조, 관절 6개 + 그리퍼 1개, 라디안). ⚠️ `gello_move_to_start`는 이 값을 목표로 쓰지 **않는다** — handshake는 항상 **구독한 현재 GELLO 자세**로 이동한다([`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2). |
| `publish_rate_hz` (publisher) | `30.0` | GELLO 원시 관절 읽기 발행 주기. |
| `ema_alpha` | `0.4` | bridge의 관절별 EMA 저역 통과 필터 계수(낮을수록 더 부드럽고 지연↑, 0.2~0.3으로 낮춰 더 스무딩). |
| `max_step_rad` | **`0.0025`** | bridge 발행 주기(250Hz)당 명령이 이동할 수 있는 최대 각도(슬루 제한 ≈ 속도 캡). 250Hz에서 ≤0.003 유지. |
| `deadband_rad` | **`0.004`** | at-rest 노이즈 게이트. Dynamixel/손 떨림 지터를 억제(흔들리면 0.006~0.01로↑, 뻑뻑하면 ~0.002로↓). |
| `staleness_timeout_s` | `0.5` | 이 시간 동안 새 GELLO 메시지가 없으면 bridge가 명령 발행을 멈춤(fail-safe). |
| `publish_rate_hz` (bridge) | `250.0` | `/forward_position_controller/commands` 발행 주기(더 스냅하게 하려면 `max_step_rad`를 올리기 전에 500으로 먼저 올릴 것). |
| `joint_states_topic` | `/joint_states` | bridge가 **첫 명령을 seed**할 로봇 실제 관절 상태 토픽. GELLO 포즈가 아니라 이 토픽의 실제 자세에서 시작하여 스타트업 스냅(velocity-limit protective stop)을 방지. |
| `soft_start_s` | `0.7` | **모든** (재)시딩(startup/resume/staleness 복구) 후 이 시간 동안 슬루 클램프를 ~15%→`max_step_rad`로 램프 — 정지 상태에서 풀슬루로 튀지 않게 잔차를 이즈인. `0.0`=off. |
| `start_paused` | `false` *(통합 launch: `true`)* | `true`면 시딩돼도 `~/resume` 전까지 발행 안 함(HOLD). yaml은 `false` 유지 — `ur7e_gello_real.launch.py`만 launch 레벨에서 `true`로 선기동-일시정지. |
| `resume_align_tol` | `0.05` *(yaml에 없음 = 노드 기본값)* | `~/resume` 정렬 게이트: `\|GELLO−실제\| ≤ 이 값`(rad)일 때만 재개 허용(=2×`chase_tol`). 방금 수렴한 팔은 통과, 오정렬 수동 resume은 거부(정지 유지). |

### `gello_move_to_start` 파라미터 레퍼런스 (실로봇 경로 전용, 수렴-게이트 핸드셰이크)

mock 경로(`ur7e_gello_rviz.launch.py`)는 이 노드를 **쓰지 않습니다**(처음부터 `forward_position_controller` active). 아래는 실로봇 런북에서 시작 스냅을 막는 수렴-게이트 핸드셰이크(`start_mode=gello`) 파라미터로, **config 값이 노드 기본값을 오버라이드**합니다. `resume_bridge` / `bridge_resume_service`는 yaml에 없고 통합 launch가 주입합니다.

| 파라미터 | 배포값(config) | 노드 기본값 | 한 줄 의미 |
| --- | --- | --- | --- |
| `start_mode` | `gello` | `gello` | `gello`=라이브 리더로 수렴 후 자동 인계 / `init_align`=고정 init_pose 이동 후 오퍼레이터 정렬 게이트 |
| `source_controller` | `scaled_joint_trajectory_controller` | (동일) | 캐치업 트래젝토리를 보내는(보간) 컨트롤러 |
| `target_controller` | `forward_position_controller` | (동일) | 인계 후 스트리밍 컨트롤러 |
| `trajectory_duration` | `5.0` | `5.0` | init_align의 init_pose 이동 시간(s). `gello` 캐치업 길이는 gap 기반 별도 산정이라 미사용 |
| `arrival_tolerance` | `0.05` | `0.05` | FollowJointTrajectory 관절별 도착 허용오차(rad). 실 JTC 도착정확도에 따라 완화(↑) 필요할 수 있음 |
| `chase_tol` | `0.025` | `0.025` | 인계 게이트: 관절별 `\|라이브GELLO−실제\| ≤ 이 값`(≈1.4°). **실기 캘리브 필요** ↓ |
| `chase_dwell_s` | `0.4` | `0.4` | 수렴이 이 시간 동안 유지(median-of-5, 새 샘플 도착 필수)돼야 인계 |
| `chase_v_budget` | `0.5` | `0.3` | 캐치업 길이 산정 속도예산(rad/s): `T=max(gap/budget, min_traj_duration)`, 상한 클램프 없음 |
| `min_traj_duration` | `0.5` | `0.75` | 캐치업 트래젝토리 최소 길이(s) — 작은 gap도 급스텝 대신 부드럽게 |
| `chase_hard_limit` | `4.0` | `4.0` | 이보다 큰 gap은 자동추격 거부(래핑/그로스-오포즈 백스톱, **접근 한계 아님**; 정상 접근 ~π 허용) |
| `chase_timeout_s` | `30.0` | `30.0` | 전체 수렴 시간예산(s). 초과 시 **exit 1**(스위치 안 함). `≤0`=무한 대기 |
| `gello_staleness_s` | `0.5` | `0.5` | 이보다 오래된 GELLO 샘플=stale. 죽은/멈춘 리더는 dwell 통과 못 하고 타임아웃(재개 스냅 방지) |
| `resume_bridge` | *(launch: `true`)* | `false` | 스위치 성공 후 브릿지 `~/resume` 호출로 스트리밍 시작. 통합 launch가 `true` 주입 |
| `bridge_resume_service` | `/gello_ur_bridge/resume` | (동일) | resume를 호출할 브릿지 Trigger 서비스명 |
| `alignment_tolerance` | `0.2` | `0.15` | *(init_align 전용)* 정렬 게이트 관절별 허용오차(rad) |
| `alignment_hard_limit` | `0.5` | `0.5` | *(init_align 전용)* `~/override_follow` 허용 상한(rad) |
| `alignment_timeout` | `0.0` | `0.0` | *(init_align 전용)* 게이트 대기 상한(s). `≤0`=무한 |

> **정직한 보증(“제로 스냅” 아님):** 인계는 팔로워가 라이브 리더를 `chase_tol` 이내로 따라잡고 유지될 때에만 발생하며, 남은 잔차는 소프트스타트·레이트리밋 슬루(`≤ max_step_rad × publish_rate_hz`, `soft_start_s` 램프)로 닫힙니다. 움직이는 리더는 스냅이 아니라 **인계 지연**으로 나타납니다.

> **실기 캘리브 주의(미검증):** 이 파이프라인은 **ros2_control mock 스택에서만 검증**되었고 실로봇 테스트는 미완료입니다. mock JTC는 목표에 *정확히* 도착하지만 실 HW는 정상상태 잔차가 남을 수 있어, `chase_tol`(0.025)이 그 잔차보다 작으면 게이트가 **라이브락**(영영 인계 안 함)될 수 있습니다. 핸드셰이크 1회 동안 `/joint_states` + 명령 토픽을 **rosbag 1개로 기록**해 관절별 정상상태 잔차를 측정하고, 그 최대치보다 `chase_tol`(및 필요 시 `arrival_tolerance`)을 약간 크게(예: `0.03`) 잡으세요.

> **sim/mock 케이던스 주의 (fake hardware):** `use_fake_hardware:=true` mock 경로에서는 궤적 컨트롤러가 `scaled_joint_trajectory_controller`가 **아니라** `joint_trajectory_controller`로 뜹니다(`scaled_`는 실 e-Series 드라이버 전용). mock은 `gello_move_to_start`를 쓰지 않으므로 무해합니다. 또한 mock 부팅 로그의 **`io_and_status_controller` spawner 실패는 무해**합니다(mock에는 IO 인터페이스 없음). 둘 다 정상이며 조인트 텔레옵에 영향 없음.

### 왜 `max_step_rad=0.0025` @ 250Hz 인가 (velocity-safe 예산)

- `forward_position_controller`는 보간을 하지 않고 **받은 위치를 즉시 명령**하므로, 사이클당 이동 한도(`max_step_rad`)가 사실상 속도 제한으로 작동한다.
- UR e-Series 드라이버는 각 명령을 500Hz 서보 사이클(0.002s)당 델타로 보고 관절 속도 한계 3.14159 rad/s와 비교한다. 사이클당 안전 예산 = 3.14159 × 0.002 ≈ 0.00628 rad. 250Hz로 발행하면 드라이버가 두 명령을 coalesce할 수 있으므로 이 예산을 절반(≈0.00314 rad)으로 잡는다.
- `max_step_rad=0.0025`는 이 절반 예산 안쪽 값으로, 단일 사이클 ≈ 1.25 rad/s, coalesce 시 ≈ 2.5 rad/s(안전), 지속 이동은 0.0025 × 250 = 0.625 rad/s에 해당한다. 실기 관절 속도 한계(3.14159 rad/s) 아래에서 손 떨림·시리얼 튐·EMA 워밍업 스파이크를 흡수한다.
- **스타트업 스냅 방지(필독 안전 노트)**: bridge는 첫 명령을 GELLO 포즈가 아니라 **로봇 실제 `/joint_states`**(`joint_states_topic`)에서 seed한다. 이전에는 첫 GELLO 메시지로 seed하여 부팅 시 ~0.42 rad(한 사이클 ≈ 210 rad/s)의 스냅이 발생, 실 UR7e에서 "External Control 속도 제한" protective stop을 유발했다. 실제 자세에서 시작하면 이 순간 점프가 사라진다. 이는 튜닝 편의가 아니라 **반드시 지켜야 할 안전 설계**다.
- `deadband_rad=0.004` 노이즈 게이트는 정지 상태의 Dynamixel/손 떨림 지터가 관절 명령으로 새는 것을 막는다.
- mock hardware(`source:=fake`, 이 PC에서 검증하는 경로)에서는 속도 제한이 물리적으로 의미가 없으므로 이 값들은 **무해**하다 — "여기서 검증되는 값"이 아니라 "실기로 넘어갈 때 그대로 안전하게 쓰기 위한 값"이다.
- 실기에서의 초기 접근은 이 seed(bridge 레벨)에 더해 `gello_move_to_start` 핸드셰이크([`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)의 move-to-start 절 참고)가 담당하여, scaled_jtc로 보간 이동 후 forward_position_controller로 STRICT 전환한다.

---

## 5. 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `gello_publisher` 실행 시 시리얼 포트 `Permission denied` | 현재 사용자가 `dialout` 그룹에 없거나, `/dev/serial/by-id/...` 심볼릭 링크가 가리키는 `/dev/ttyUSB*`에 그룹 rw 권한이 없음 | `sudo usermod -aG dialout $USER` 후 **재로그인**(또는 `newgrp dialout`). 즉시 임시로 풀려면 `sudo chmod 666 /dev/ttyUSB0`(재부팅/재연결 시 초기화되는 임시방편 — README의 "임시 chmod 666 워크어라운드" 참고). |
| 런치를 여러 번 실행/중단하다 보니 이전 노드가 남아 두 개의 `gello_publisher` 또는 `gello_ur_bridge`가 동시에 시리얼/토픽을 잡음(고스트/오번 노드) | `Ctrl+C`로 launch를 종료해도 자식 프로세스가 완전히 죽지 않고 남는 경우가 있음(특히 시리얼 I/O 중 블로킹) | `ros2 node list`로 중복 노드 확인 → `pkill -f gello_publisher`, `pkill -f gello_ur_bridge`, `pkill -f fake_gello` 로 정리 후 재실행. 재실행 전 `ros2 topic hz /gello/joint_states`로 발행자가 하나뿐인지 확인. |
| `gello_publisher` / `robotiq_urcap` 실행 시 `ModuleNotFoundError: No module named 'gello'` | 노드가 리포 루트의 python `gello` 패키지를 import하는데, 하드코드된 폴백 경로가 `/home/theo_lab/gello_software`로 되어 있어 **이 머신 경로와 다름** | 노드 코드는 수정하지 않는다. 대신 셸에 `export GELLO_REPO_ROOT=/home/laptop3/gello_software`를 설정한 뒤(예: `~/.bashrc`에 추가하거나 launch 실행 전 매번 export) 노드를 실행한다. |
| `ros2 control list_controllers`에서 `forward_position_controller`가 `inactive`이거나, `/forward_position_controller/commands`에 아무것도 발행되지 않음 | launch 내에서 bridge/GELLO source 노드가 컨트롤러 스포너보다 먼저 뜨는 레이스 컨디션 — `ur7e_gello_rviz.launch.py`는 컨트롤러가 spawn될 시간을 벌기 위해 bridge/source 노드를 **~5초 `TimerAction`으로 지연 기동**한다 | 5초를 기다린 뒤 다시 확인. 그래도 `inactive`라면 `ros2 control list_controllers`로 실제 상태를 먼저 확인하고, `ros2 control switch_controllers`로 수동 재시도하거나 launch를 재시작한다. 5초보다 컨트롤러 스폰이 오래 걸리는 저사양 환경이면 launch의 `TimerAction` 지연 값을 늘리는 것을 고려(단, `ur7e_gello_rviz.launch.py`는 FROZEN이 아니므로 수정 가능하나 이 작업 범위 밖). |
| RViz2 창이 뜨지 않거나 `qt.qpa.xcb: could not connect to display` 등 DISPLAY 관련 에러로 launch 자체가 실패 | 헤드리스 환경(SSH, 컨테이너 등)에서 DISPLAY가 없는데 RViz2를 강제로 띄우려 함 | `launch_rviz:=false`로 실행: `ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false`. 파이프라인(컨트롤러/브리지/토픽)만 검증하고 싶을 때 표준적으로 쓰는 방법이며, VERIFY-HERE 절차도 이 옵션을 기본으로 사용한다. |

### 5-1. 알려진 제약 — `ros-humble-ur` 2.8.1의 ur7e는 "부분 지원" (⚠️ 중요)

실측 결과, 설치된 `ros-humble-ur` **2.8.1**에서 `ur_type:=ur7e`는 **로드·구동되지만 서술(description)이 완전하지 않다**:

- `ur_description`의 `config/ur7e/`(joint_limits·kinematics·physical·visual_parameters)는 존재하나, **`visual_parameters.yaml`이 메시를 `meshes/ur5e/...`로 참조**한다 (ur7e 전용 메시 디렉토리가 없음). 그래서 `robot_description`에 `ur5e` 문자열이 14회 등장하고 RViz에는 **ur5e 메시로 렌더링**된다.
- 드라이버 config에 **`ur7e_update_rate.yaml`이 없어** launch 시 `Parameter file path is not a file: .../ur7e_update_rate.yaml` **경고**가 뜬다(제어는 기본 rate로 정상 동작 — RTDE 500Hz, bridge 명령 250Hz 확인됨).

**영향 범위 판단:**
- ✅ **joint-space GELLO 텔레오퍼에는 문제 없음.** 관절명(`shoulder_pan_joint`…`wrist_3_joint`)과 관절 리밋은 ur5e/ur7e가 동일하고, 우리는 관절 각도를 명령하므로 메시/DH 차이는 **RViz 겉모습과 TCP/Cartesian 정확도에만** 영향을 준다. (이것이 실기에서 `ur_type:=ur5e`로 잘못 실행해도 관절 추종이 정상 동작하는 이유다.)
- ⚙️ **정확한 ur7e 비주얼/기구학이 필요하면** `ros-humble-ur`를 **2.13.2로 업그레이드**한다(apt 후보 존재: `sudo apt install ros-humble-ur=2.13.2-*` 후 재빌드 불필요, 드라이버만 갱신). 업그레이드 시 컨트롤러 이름이 바뀔 수 있으니 이 문서의 `list_controllers` 기대값을 재확인할 것.

> 요약: **관절 텔레오퍼 목적이면 2.8.1로 충분(검증 완료)**, 정밀 ur7e 모델이 필요하면 2.13.2.

---

**안전 재확인**: 위 어떤 트러블슈팅 단계를 수행하더라도 GELLO는 **항상 passive read-only** 입력 장치다. Dynamixel 드라이버는 토크를 OFF로 초기화하고 관절 각도를 읽기만 하며, 이 브링업의 어떤 노드도 GELLO 모터에 토크 명령을 보내지 않는다.

---

## 6. 일시정지·재개 (두 손 자유 씬 리셋)

텔레옵 중에 물체를 제자리에 놓거나 지그를 옮기는 등 **씬을 리셋하려면 리더암에서 손을 떼야** 합니다. 그런데 GELLO 리더는 **수동(passive)** 이라 손을 놓으면 중력에 처지고, 그 처지는 관절값이 그대로 로봇으로 흘러 들어갑니다(실기에서는 스냅/충돌 위험). 이 기능은 **리더→로봇 신호 경로를 확실히 끊는(pause)** 스위치와, 리셋 후 로봇을 **부드럽고 속도-제한된 글라이드로 되돌리는(resume)** 경로를 제공합니다.

- **Pause** 는 무조건(unconditional) 스트리밍을 멈춥니다. 팔은 마지막 명령 자세를 유지하고(`forward_position_controller` 셋포인트 홀드), Robotiq 그리퍼는 마지막 위치를 온보드로 유지합니다. **발행하지 않음이 안전 상태**이므로 pause는 게이팅/지연/거부되지 않습니다 — 한 번 클릭, 항상 성공.
- **Resume** 는 **게이팅 + fail-closed** 입니다: 거부되거나 실패한 재개는 **일시정지 상태를 유지하고 아무것도 발행하지 않습니다.** 점프 없이 재시작 가능할 때만 진행하며, 첫 명령은 팔/그리퍼가 이미 있는 위치와 같습니다(zero-jump). 이후 속도-제한 글라이드로 gap을 닫습니다.

이 기능은 mock 경로(`source:=fake`)에서도 서비스/토픽이 그대로 존재하므로 **이 PC에서 리허설 가능**합니다(§끝의 리허설 참고).

### 오퍼레이터 워크플로우

권장 UI는 `gello_recorder`의 **Teleop 바**입니다(`ros2 run gello_recorder gello_recorder_gui`, 녹화 컨트롤 바로 위). `arm:` / `gripper:` 상태 라벨과 **Pause Teleop** · **Resume Teleop** 버튼이 있습니다.

1. **Pause** — **Pause Teleop** 클릭(한 번). 팔·그리퍼 브릿지 모두 정지, 라벨이 빨강(`PAUSED`).
2. **씬 리셋** — 두 손을 자유롭게. 일시정지 중에는 리더가 처지든 옮겨지든 로봇에 아무것도 전달되지 않음.
3. **리더 재정렬** — 리더를 로봇의 **멈춰 있는(frozen) 자세**에 대략 다시 맞추고 약 0.5초간 **가만히 유지**. (로봇은 움직이지 않았으므로, 로봇을 리더에 맞추는 게 아니라 리더를 로봇에 맞추는 것.)
4. **Resume** — **Resume Teleop** 클릭. 실 로봇을 움직이므로 **두 번 클릭 확인**입니다: 버튼이 주황색 *"Confirm Resume (robot will move!)"* 로 바뀌고, **3초 안에** 다시 클릭해야 실제로 재개 요청이 나갑니다(시간 초과 시 원상복귀 — 다시 클릭해 재무장).
5. 수락되면 팔은 실제 자세에서 재시딩(zero-jump)→소프트스타트→리더 자세로 **글라이드**(`CHASING`→`FOLLOWING`), 그리퍼는 실제 위치에서 ~2초 램프(`RAMPING`→`FOLLOWING`). 둘 다 `FOLLOWING` 이 될 때까지 로봇에서 물러서고 리더를 가만히 유지.

**Resume Teleop** 버튼은 팔이 `PAUSED` 일 때만 활성화됩니다. 재개가 **거부**되면 로봇은 움직이지 않고 브릿지는 일시정지 상태로 남으며, 거부 사유가 상태바에 ~6초간 표시됩니다 — 읽고, 그 한 가지를 고친 뒤(§트러블슈팅 표), 다시 재개하세요.

**터미널 대체 수단(GUI 없이):**

```bash
# Pause (무조건 — 항상 성공)
ros2 service call /gello_ur_bridge/pause        std_srvs/srv/Trigger
ros2 service call /gello_gripper_bridge/pause   std_srvs/srv/Trigger

# ... 씬 리셋, 리더를 로봇 자세에 재정렬, 가만히 유지 ...

# Resume (게이팅 — success/message 확인. 거부되면 PAUSED 유지)
ros2 service call /gello_ur_bridge/resume_chase std_srvs/srv/Trigger
ros2 service call /gello_gripper_bridge/resume  std_srvs/srv/Trigger

# 실시간 상태
ros2 topic echo /gello_ur_bridge/state
ros2 topic echo /gello_gripper_bridge/state
```

> ⚠️ 수동 리셋에는 `/gello_ur_bridge/resume`(엄격 게이트)가 아니라 **`/gello_ur_bridge/resume_chase`** 를 쓰세요. 전자는 시작 핸드셰이크 전용이라 조금이라도 gap이 있으면 거부합니다. `success: false` 는 **에러가 아니라 거부** — 로봇은 안 움직였고 브릿지는 여전히 PAUSED입니다.

### 서비스 · 토픽 · 상태 (contract)

모든 서비스는 `std_srvs/srv/Trigger`, 모든 state 토픽은 `std_msgs/msg/String` 5 Hz. 노드: `gello_ur_bridge`, `gello_gripper_bridge`.

| 이름 | 동작 |
| --- | --- |
| `/gello_ur_bridge/pause` | 무조건 정지. 팔 홀드. 항상 성공. |
| `/gello_ur_bridge/resume` | **엄격** 게이트(`resume_align_tol` 이내). 시작 핸드셰이크 전용 — 수동 리셋엔 쓰지 말 것. |
| `/gello_ur_bridge/resume_chase` | **수동 리셋용 재개.** 리더가 quasi-still일 때 bounded gap을 글라이드로 닫음. |
| `/gello_ur_bridge/state` | `PAUSED` / `WAITING` / `STALE` / `CHASING` / `FOLLOWING` (이 우선순위). |
| `/gello_gripper_bridge/pause` | 무조건 정지. Robotiq 온보드 홀드. 항상 성공. |
| `/gello_gripper_bridge/resume` | 게이팅(fresh 리더 그리퍼 샘플 + 실제 그리퍼 위치 필요). 실제 위치에서 시딩 후 램프. |
| `/gello_gripper_bridge/state` | `PAUSED` / `WAITING` / `RAMPING` / `FOLLOWING`. |

### 안전 게이트와 이유

`resume_chase` 는 **모두 만족할 때만** 수락(아니면 PAUSED 유지, 무발행):

| 게이트 | 기본값 | 이유 |
| --- | --- | --- |
| (a) fresh 리더 샘플 | age ≤ `staleness_timeout_s`(0.5s) | 죽은/멈춘 리더 스트림이 모션을 승인하면 안 됨(크래시한 `gello_publisher`가 stale 타깃으로 재개하는 것 차단). |
| (b) 로봇 실제 자세 인지 | `/joint_states` 수신 | gap 측정과 zero-jump 시딩 모두 실제 자세 필요. |
| (c) 리더 quasi-still | 최악 관절 속도 ≤ `resume_chase_still_speed`(0.10 rad/s), `resume_chase_still_window_s`(0.3s) 창 | 움직이는 리더로 재개하면 로봇이 움직이는 타깃을 추격. **fail-closed:** 샘플/시간 커버리지 부족 = *not still* → 거부. 리더를 ~0.5초 가만히. |
| (d) 관절별 gap ≤ 상한 | `resume_chase_max_gap`(1.5 rad) | 글라이드를 유계화. 1.5 rad ≈ 글라이드 ~2.4s + 이즈인 0.7s. 더 큰 오정렬은 거부 → 로봇의 긴 자율 스윕 방지. 리더를 더 가깝게. |

수락 시 상태 변경은 브릿지의 **기존 seed 브랜치**로 떨어뜨리는 것뿐입니다: 다음 250 Hz tick이 리더 타깃을 팔의 실제 자세에 가장 가까운 분기로 재앵커(`wrapped_nearest` — 항상 짧은 쪽, ~2π 스핀 없음)하고, 실제 자세에서 시딩(첫 명령 = 현재 팔 위치, zero-jump)하며, `soft_start_s`(0.7s) 램프를 재시작하고, 이후 사이클당 최대 `max_step_rad`(0.0025 rad @ 250 Hz = 지속 0.625 rad/s)로 슬루합니다. **새 발행 경로는 도입되지 않으며**, 시작 핸드셰이크·staleness 복구가 이미 쓰는 그 machinery입니다. 그리퍼 resume은 실제 위치에서 시딩(`invert`는 crush-hazard 방지로 `false` 유지) 후 `resume_ramp_s` 동안 리더로 슬루-제한(램프 중 deadband 우회)하여 점프 없음.

### 트러블슈팅 (일시정지·재개)

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| resume 거부: `leader is moving (or stillness not yet established)` | 게이트 (c): 리더가 아직 quasi-still로 입증되지 않음(움직이거나 창이 덜 참). | GELLO를 ~0.5초 **가만히** 잡고 다시 재개. |
| resume 거부: `gap too large: max … > resume_chase_max_gap 1.500` | 게이트 (d): 리더가 멈춰 있는 팔 자세에서 1.5 rad 넘게 벗어남(메시지가 최악 관절·관절별 gap 표기). | 리더를 로봇의 멈춘 자세에 **더 가깝게** 맞춘 뒤 재개. |
| resume 거부: `no fresh GELLO sample (age=… > 0.50s)` / state가 `STALE` | 리더 스트림이 stale/dead. | `gello_publisher`(또는 `fake_gello`)가 살아 `/gello/joint_states`를 발행 중인지 확인(`ros2 topic hz /gello/joint_states`). |
| resume 거부: `robot actual pose unknown (no /joint_states yet)` | 로봇 드라이버/`joint_state_broadcaster` 미기동. | `/joint_states` 복구를 기다림. |
| GUI 그리퍼 라벨이 `gripper: n/a` | 그리퍼 브릿지/state가 아직 없음. | 정상일 수 있음 — 그리퍼 브릿지는 **핸드셰이크 완료 후에 기동**하므로, 팔 텔레옵이 스트리밍을 시작한 뒤 라벨이 채워짐. 팔 텔레옵을 먼저 붙일 것. |
| Pause를 눌렀는데 팔이 계속 움직임 | 두 개의 `gello_ur_bridge`가 떠 있거나(고스트 노드), state를 잘못된 네임스페이스에서 봄. | `ros2 node list`로 중복 확인 후 정리(§5 트러블슈팅 표의 고스트/오번 노드 항목). `ros2 topic echo /gello_ur_bridge/state`가 `PAUSED`인지 확인. |
| 그리퍼 resume 거부: `no actual gripper position` | `/robotiq_gripper/position_percent` 피드백이 없고 아직 아무것도 발행 안 됨. | Robotiq 노드가 떠서 position feedback을 발행하는지 확인. |

### 리허설 (mock, 필수 주의)

로봇 없이 pause/거부/글라이드 전체를 mock에서 리허설할 수 있습니다. 한 터미널에서 fake 스택을 띄우고(`ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake` 또는 `./run_ur7e_gello_sim.sh`), 다른 터미널에서:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./check_pause_resume_sim.sh
```

`fake_gello`의 `/fake_gello/hold` · `/fake_gello/sweep` · `/fake_gello/collapse` · `/fake_gello/set_pose` 를 구동해 pause, fail-closed 거부(움직이는 리더 / gap>1.5 rad), 그리고 수락된 ~0.8 rad 글라이드를 단계별 PASS/FAIL로 검증합니다.

> ⚠️ **SIM PASS는 필요조건이지 충분조건이 아님.** mock 하드웨어는 속도 제한도 protective stop도 **강제하지 않습니다.** 속도-제한 및 부드러움 안전은 데이터 수집에 쓰기 전 **실 UR7e에서 사람이 반드시 재검증**해야 합니다.

---

## 관련 문서

- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실물 UR7e + 실 GELLO 구동 런북(이 PC에서는 검증 불가).
- [`GELLO_UR_ROS2_BRINGUP.md`](./GELLO_UR_ROS2_BRINGUP.md) — UR5e 원본 브링업(설계 기준 문서).
- [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md) — GELLO ↔ UR ROS2 통합 설계/계획 문서.
