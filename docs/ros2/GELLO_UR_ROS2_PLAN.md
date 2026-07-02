# GELLO → 실제 UR5e + Robotiq 2F-85 ROS2 텔레오퍼레이션 — 세팅 계획 (PLAN)

> ⚠️ **이 문서는 실제 UR5e를 ROS2로 구동하기 위한 계획서(PLAN)이며, 아직 완전히 구현·검증되지 않았습니다.**
> 실제 UR 하드웨어는 별도 PC에 있고 이 문서의 bridge/gripper 신규 노드, 안전 handshake, 실기 전환(Phase 4)은 아직 미구현입니다.
> **현재 실제로 검증된 로컬 경로**는 mock-hardware 기반 RViz2 시각화뿐이며 → [GELLO_UR_ROS2_BRINGUP.md](GELLO_UR_ROS2_BRINGUP.md)를 참고하세요 (`ur_gello_bringup` 패키지, `use_mock_hardware`, 실 UR 미접속).
>
> **Humble vs Jazzy 불일치 안내**: 아래 계획 본문은 **ROS2 Humble + URSim**을 타깃으로 서술되어 있으나, 실제로 빌드·RViz2 검증된 `ur_gello_bringup` 패키지는 **ROS2 Jazzy + `use_mock_hardware`** 환경에서 동작이 확인되었습니다. 아키텍처(name-based reorder → bridge → forward_position_controller), 안전 handshake, 캘리브레이션 절차는 두 distro에서 그대로 이어지며 **차이는 distro(패키지 이름의 `ros-humble-*` ↔ `ros-jazzy-*`, apt 대상)뿐**입니다. 실기 전환 시 이 계획의 Humble 기준 명령을 Jazzy로 치환하면 됩니다.

물리 **GELLO**(6-DOF UR 리더 암)로 **실제 UR5e + Robotiq 2F-85**를 **ROS2 Humble** 위에서 텔레오퍼레이션하기 위한 구축 계획 문서. 아직 실제 UR이 연결되지 않은 상태를 전제로, **URSim(도커) 우선 검증 → 실로봇** 순서로 진행한다.

- 선행 문서: [GELLO_UR_SIM_TELEOP.md](../sim/GELLO_UR_SIM_TELEOP.md) (MuJoCo sim으로 GELLO 연결·캘리브레이션 검증 완료), [GELLO_ROS2_CONTROL_REFERENCE.md](GELLO_ROS2_CONTROL_REFERENCE.md) (기존 `ros2/` Franka 스택 분석). 세부 근거·URL은 해당 리서치 결과에 있음.
- **안전 대전제 (변함없음)**: GELLO는 **항상 passive read-only** — `dynamixel_torque_enable` 전부 0, 종료 시 `disable_torque()`. ROS2 경로에서도 동일하게 유지한다.

## 요약 (TL;DR)

| 항목 | 결정 |
|---|---|
| UR 제어 | **공식 `Universal_Robots_ROS2_Driver`** (`ros-humble-ur`) + `forward_position_controller`(servoj 스트리밍) |
| 팔 명령 경로 | GELLO publisher(기존 `ros2/` 패키지 소폭 수정) → **신규 bridge node**(rclpy, name-based reorder + EMA + step clamp + watchdog) → `/forward_position_controller/commands` |
| 그리퍼 | **신규 rclpy 노드**가 percent 토픽 구독 → 리포의 `gello/robots/robotiq_gripper.py`(URCap TCP 63352)로 실 2F-85 구동 — 추가 하드웨어 불필요 (Option B) |
| 기존 `ros2/` Franka 컨트롤러 | **재사용 불가**(effort/torque 기반, libfranka 전용) — 대체하되 move-to-start·staleness deadman 등 안전 패턴은 이식 |
| 로봇 없는 테스트 | **URSim 도커**(권장, External Control 흐름까지 검증) 또는 `use_fake_hardware:=true` |
| 최대 리스크 | `forward_position_controller`는 활성화 시점에 로봇 자세 ≠ GELLO 자세면 **joint-velocity-limit fault**(확인된 실버그) → §6의 **안전 handshake 필수** |

### 목차

1. [목표 & 아키텍처](#1-목표--아키텍처-architecture)
2. [재사용 vs 신규 정리](#2-재사용-vs-신규-정리-reuse-vs-build)
3. [사전 준비](#3-사전-준비-prerequisites)
4. [단계별 세팅 로드맵](#4-단계별-세팅-로드맵-phase-0--4)
5. [신규 노드 상세](#5-신규-노드-상세-new-nodes)
6. [안전 handshake & watchdog](#6-안전-handshake--watchdog-mandatory)
7. [열린 결정 / 리스크](#7-열린-결정--리스크-open-questions)

---

## 1. 목표 & 아키텍처 (Architecture)

목표: GELLO를 손으로 움직이면 실제 UR5e 6축이 1:1(radian)로 추종하고, GELLO 그리퍼 레버로 실 2F-85가 개폐되는 상태. 전 구간 ROS2 Humble.

```
GELLO (Dynamixel XL330 ×7, passive, U2D2 FTBEO6QK @57600)
  │ USB
  ▼
[gello publisher]  franka_gello_state_publisher (기존 패키지, 6-DOF 수정)  ~25 Hz
  ├─ /gello/joint_states                                   (sensor_msgs/JointState, UR 6축, name 포함)
  └─ /gripper/gripper_client/target_gripper_width_percent  (std_msgs/Float32, 0~1)
  │
  ├────────► [팔] ur_gello_bridge  (신규 rclpy 노드)
  │             · name 기반 reorder → UR 순서 [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
  │             · EMA low-pass + per-cycle max-step clamp
  │             · staleness watchdog (>0.5 s → fail-closed)
  │             · (옵션) ~100–125 Hz 업샘플
  │             ▼
  │           /forward_position_controller/commands  (std_msgs/Float64MultiArray, 6)
  │             ▼
  │           [UR 공식 드라이버] ur_robot_driver + forward_position_controller
  │             ▼  External Control URCap ↔ 내부 ~500 Hz servoj
  │           UR5e 실기
  │
  └────────► [그리퍼] robotiq_urcap_node  (신규 rclpy 노드)
                · percent(0~1) → Robotiq POS(0~255) 변환
                · gello/robots/robotiq_gripper.py 재사용 (URCap TCP socket, UR IP:63352)
                ▼
              Robotiq 2F-85 실기
```

핵심 설계 판단:

- **왜 `forward_position_controller`인가** — GELLO 같은 외부 position 스트리밍(moveit_servo류)에 맞게 설계된 컨트롤러. 25–30 Hz 입력도 hold-last-value로 수용하고 로봇 내부 servoj(~500 Hz)에 태워진다. 단 smoothing/safety가 전혀 없으므로 smoothing·clamp·watchdog을 bridge가 담당한다(§5, §6).
- **왜 기존 Franka 컨트롤러를 못 쓰는가** — `franka_fr3_arm_controllers`는 τ = K(q_gello − q) − D·dq_filt 형태의 **effort(토크) 임피던스 컨트롤러**로, libfranka의 1 kHz FCI 토크 인터페이스 전제. UR5e는 position/velocity 인터페이스만 노출하므로 구조적으로 이식 불가. 대신 그 안의 좋은 패턴(move-to-start speed_factor 0.2, 유효 GELLO 수신 전 명령 금지, 0.5 s staleness → shutdown)을 bridge/handshake에 이식한다.
- **왜 그리퍼는 URCap 소켓 직결인가(Option B)** — 리포에 이미 실동작하는 `gello/robots/robotiq_gripper.py`(UR IP의 TCP 63352, URCap 프로토콜)가 있어 추가 하드웨어·드라이버 없이 가장 단순·확실. 기존 `ros2/src/franka_gripper_manager`의 robotiq client는 PickNik `ros2_robotiq_gripper`(ros2_control + Modbus/RS-485 시리얼) 전제라 **그대로는 사용 불가**(§7 Option A 참고).

## 2. 재사용 vs 신규 정리 (Reuse vs Build)

| 컴포넌트 | 파일 | 분류 | 내용 |
|---|---|---|---|
| Dynamixel 드라이버 + XL330 모터 설정 | `ros2/src/franka_gello_state_publisher/` (driver, `xl330.yaml`) | **재사용 (config만)** | arm-agnostic. `num_arm_joints`, `joint_signs`, `assembly_offsets`, `gripper_range_rad`, `com_port`, `dynamixel_*` 배열 전부 YAML/launch 파라미터로 주입 가능 |
| offset 측정 스크립트 | `ros2/.../scripts/get_offsets.py` | **재사용 (그대로)** | arm-agnostic — 값 6개만 넘기면 됨 |
| GELLO publisher 노드 | `ros2/src/franka_gello_state_publisher/franka_gello_state_publisher/gello_publisher.py` (57–71행) | **수정 (소규모)** | 하드코딩된 `JOINT_NAMES=["fr3_joint1"..7]` + `frame_id="fr3_link0"` → UR joint 이름 6개 + UR base frame으로 교체 (ROS2 파라미터화 권장) |
| GELLO hardware 가공 | `ros2/.../gello_hardware.py` (75–86행) | **수정 (필수, 크래시)** | 7×2 FR3 `JOINT_POSITION_LIMITS`(+`MID_JOINT_POSITIONS`) 하드코딩 — `num_arm_joints=6`이면 `np.clip`에서 **numpy shape-mismatch 크래시**. 6행 UR5e limits(±2π, elbow ±π)로 교체하거나 클램프 제거/config화 |
| UR용 publisher config | `ros2/src/franka_gello_state_publisher/config/rwh_ur_ros2.yaml` | **신규 (YAML만)** | `configs/rwh_ur.yaml`의 검증된 캘리브레이션 이식 (§4 Phase 1) |
| 팔 컨트롤러 | `ros2/src/franka_fr3_arm_controllers/` | **대체 (재사용 불가)** | effort 전용 → UR 공식 드라이버 + `forward_position_controller` + 신규 bridge로 대체. 안전 패턴만 이식 |
| bridge 노드 | `ur_gello_bridge` (신규, §5.1) | **신규 #1** | name-based reorder + EMA + step clamp + watchdog → Float64MultiArray publish |
| 그리퍼 노드 | `robotiq_urcap_node` (신규, §5.2) | **신규 #2** | percent → POS 0–255, `gello/robots/robotiq_gripper.py` 재사용 |
| 그리퍼 client (기존) | `ros2/src/franka_gripper_manager/.../robotiq_gripper_client.py` | **미사용** | PickNik ros2_control/Modbus 스택 전제 — URCap 소켓과 불일치 |
| UR 드라이버 | `ros-humble-ur` (apt) | **설치** | §3 |

## 3. 사전 준비 (Prerequisites)

### 3.1 UR 공식 드라이버 설치 (호스트, ROS2 Humble)

```bash
sudo apt update
sudo apt install ros-humble-ur        # Universal_Robots_ROS2_Driver 메타패키지
```

소스 빌드가 필요해지면(패치 등) 공식 문서의 `vcs import ...repos` + `colcon build` 절차 사용 — 세부는 리서치 결과 참조. v1은 apt 바이너리로 충분.

### 3.2 UR 로봇 측 준비 (실기 도착 시)

1. **External Control URCap** 설치 — 티치펜던트에 `externalcontrol-1.0.5.urcap` 설치 후, URCap의 Installation 노드에 **ROS PC의 IP** 입력.
2. 펜던트를 **Remote Control 모드**로 전환 (또는 드라이버 `headless_mode` 사용).
3. 네트워크: ROS PC ↔ UR 컨트롤박스 이더넷 직결/동일 서브넷, 상호 ping 확인.

### 3.3 kinematics 캘리브레이션 추출 (실기, 일회성)

공장 캘리브레이션을 추출해 드라이버에 넘겨야 TCP 정확도가 나온다:

```bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=<UR_IP> target_filename:=$HOME/my_ur5e_calib.yaml
```

이후 모든 bringup에 `kinematics_params_file:=$HOME/my_ur5e_calib.yaml`을 붙인다.

### 3.4 GELLO 측 (기존과 동일)

- 5V 외부 전원 + U2D2 USB, 포트 `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` (baud 57600).
- `dialout` 권한: `sudo usermod -aG dialout $USER` 후 재로그인 (또는 임시 `chmod 666`).

## 4. 단계별 세팅 로드맵 (Phase 0 → 4)

### Phase 0 — 로봇 없이 UR 드라이버 검증 (URSim / fake hardware)

**목적**: GELLO 없이 드라이버·컨트롤러 스위칭·`forward_position_controller` 스트리밍을 먼저 손에 익힌다.

**A. URSim 도커 (권장 — External Control/dashboard 실흐름까지 검증)**

```bash
# URCap 사전 설치 + 고정 IP(192.168.56.101)로 URSim 기동
ros2 run ur_client_library start_ursim.sh -m ur5e

# 별도 터미널: 드라이버 bringup (기본 scaled_joint_trajectory_controller 활성)
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=192.168.56.101 launch_rviz:=true
```

브라우저에서 `http://192.168.56.101:6080/vnc.html`(URSim 웹 펜던트)로 프로그램 실행(External Control) 확인.

**B. fake hardware (더 가벼움, ros2_control mock — state가 command를 그대로 미러링)**

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true launch_rviz:=true
```

**공통 확인 절차**:

```bash
ros2 control list_controllers            # scaled_joint_trajectory_controller [active] 확인
# 궤적 이동 스모크 테스트
ros2 launch ur_robot_driver test_scaled_joint_trajectory_controller.launch.py
# 컨트롤러 스위치 (position command interface는 한 번에 한 컨트롤러만 소유)
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate forward_position_controller
# 현재 관절값 근처로 수동 스트리밍 테스트 (작은 값 변화만!)
ros2 topic pub -r 50 /forward_position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"
```

**성공 기준**: ① bringup 후 `/joint_states` 수신, ② RViz에서 궤적 이동 재생, ③ 컨트롤러 스위치 성공, ④ `forward_position_controller`로 관절 스트리밍 시 fault 없이 추종. URSim에서 **큰 점프를 일부러 줘서 velocity-limit fault가 나는 것도 확인**해 둘 것(§6의 근거 체감).

### Phase 1 — GELLO publisher 6-DOF 수정 + UR config

**목적**: `/gello/joint_states`에 UR 6축이 올바른 이름으로 publish되게 한다.

1. **`gello_publisher.py` (57–71행)**: `JOINT_NAMES`를 UR 이름으로, `frame_id`를 UR base(예: `base_link`)로 — 하드코딩 대신 ROS2 파라미터(`joint_names`, `frame_id`)로 빼는 것을 권장.

   ```python
   UR_JOINT_NAMES = [
       "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
   ]
   ```

2. **`gello_hardware.py` (75–86행)**: FR3 7×2 limits → UR5e 6×2로. UR5e는 전 관절 ±2π(elbow는 ±π)이고 leader는 사람이 든 그대로 보고하는 게 맞으므로, **클램프 자체를 제거하거나 config-driven으로** 바꾸는 쪽 권장(조민제 도커도 클램프를 제거했음 — [레퍼런스 §4](GELLO_ROS2_CONTROL_REFERENCE.md)).
3. **신규 config** `ros2/src/franka_gello_state_publisher/config/rwh_ur_ros2.yaml` — sim에서 검증된 값 이식:

   | 파라미터 | 값 | 출처 |
   |---|---|---|
   | `num_arm_joints` | `6` | UR 6-DOF |
   | `com_port` | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` | 기존 |
   | `joint_signs` | `[1, 1, -1, 1, 1, 1]` | `configs/rwh_ur.yaml` |
   | `assembly_offsets` | `[3.142, 4.712, 1.571, 4.712, 4.712, 3.142]` (= π, 3π/2, π/2, 3π/2, 3π/2, π) | `configs/rwh_ur.yaml` — ⚠️ 아래 주의 |
   | `gripper_range_rad` | `[2.936, 3.666]` (= close 168.234°, open 210.034°) | `gripper_config` deg→rad 환산 |
   | `dynamixel_torque_enable` | `[0, 0, 0, 0, 0, 0, 0]` | **안전 — 절대 변경 금지** |

   > ⚠️ **offset 재측정 플래그**: Python 경로와 ROS2 publisher는 offset 처리 컨벤션이 미묘하게 다르다(Python은 init 때 `start_joints` 기준 2π 래핑, ROS2는 `assembly_offsets` + 선택적 align-to-start — [레퍼런스 §3](GELLO_ROS2_CONTROL_REFERENCE.md)). 위 값을 우선 넣되, Phase 1 검증에서 각도가 π/2 배수만큼 어긋나면 **`ros2/.../scripts/get_offsets.py`(6값 전달)로 재측정**한다.

4. 빌드 & 실행:

   ```bash
   cd /home/theo_lab/gello_software/ros2
   colcon build --packages-select franka_gello_state_publisher && source install/setup.bash
   ros2 launch franka_gello_state_publisher main.launch.py \
     gello_config_file:=rwh_ur_ros2.yaml \
     com_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
   ```

**성공 기준**: `ros2 topic echo /gello/joint_states`에서 ① `name`이 UR 6개 이름, ② GELLO 캘리 포즈(0, −90°, +90°, −90°, −90°, 0°)에서 position ≈ `[0, -1.57, 1.57, -1.57, -1.57, 0]`, ③ 각 관절을 손으로 움직이면 해당 축만 매끄럽게 변함, ④ shape-mismatch 크래시 없음. 그리퍼 레버로 `/gripper/.../target_gripper_width_percent`가 0↔1로 스윕되는지도 확인.

### Phase 2 — bridge node (팔 스트리밍)

**목적**: `/gello/joint_states` → `/forward_position_controller/commands` 연결. 상세 설계·코드는 §5.1.

1. 신규 패키지(가칭 `ur_gello_bringup`, §7 참고) 생성 후 `ur_gello_bridge` 노드 구현.
2. **URSim + 실제 GELLO**로 통합 테스트:

   ```bash
   # T1: URSim         ros2 run ur_client_library start_ursim.sh -m ur5e
   # T2: UR 드라이버    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.56.101 launch_rviz:=true
   # T3: GELLO publisher (Phase 1 명령)
   # T4: handshake(§6 — v1은 수동) 후 bridge
   ros2 run ur_gello_bringup ur_gello_bridge
   ```

3. topic_tools `transform`으로 1줄 스모크 테스트를 먼저 해볼 수는 있으나(빠른 배선 확인용), **name 순서 보장이 없어 실사용 금지** — 반드시 bridge로 대체.

**성공 기준**: ① URSim 속 UR5e가 GELLO를 1:1 추종(RViz/웹 펜던트), ② 급하게 흔들어도 velocity fault 없음(clamp 동작), ③ GELLO publisher를 kill하면 watchdog이 0.5 s 내 fail-closed(스트리밍 중지 + 명시적 로그/종료), ④ joint 이름을 섞은 가짜 JointState를 넣어도 올바른 축에 매핑(name-based reorder 검증).

### Phase 3 — gripper node

**목적**: GELLO 레버 → 실 2F-85. 상세는 §5.2. **2F-85는 URSim에 없으므로 이 Phase의 실검증은 실기에서만 가능** — URSim 단계에서는 노드 기동 + 변환 로직 단위 테스트(percent→POS)까지만.

**성공 기준(실기)**: ① 노드 기동 시 gripper activate 성공, ② 레버 당김 → 닫힘, 놓음 → 열림(극성 확인 — 반대면 §5.2의 변환식 부호 수정), ③ 레버 연타에도 소켓 에러 없음(명령 rate-limit 동작).

### Phase 4 — 실로봇 전환 + 안전 handshake

**목적**: URSim에서 검증된 전체 스택을 실 UR5e로. **§6의 handshake 프로토콜을 그대로 따른다.**

```bash
# T1: 드라이버 (실기, 캘리브레이션 파일 포함; scaled_jtc 활성 + forward_position_controller는 inactive 로드)
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e robot_ip:=<UR_IP> \
  kinematics_params_file:=$HOME/my_ur5e_calib.yaml
# 펜던트에서 External Control 프로그램 ▶ (또는 headless_mode)

# T2: GELLO publisher (Phase 1)
# T3: gripper node
# T4: handshake 스크립트/수동 절차(§6) → bridge 기동
```

**성공 기준**: ① move-to-start가 저속(≤0.2–0.3 rad/s)으로 GELLO 자세까지 이동 후 SUCCEEDED, ② 컨트롤러 스위치 후 fault 없이 스트리밍 개시, ③ 1:1 추종 + 그리퍼 개폐, ④ GELLO USB를 뽑는 테스트에서 fail-closed 동작, ⑤ protective stop 유발 → §6-7 복구 절차로 재개 가능. **첫 실기 테스트는 반드시 감속 모드(펜던트 speed slider 낮춤) + e-stop 손 닿는 곳에서.**

## 5. 신규 노드 상세 (New Nodes)

### 5.1 `ur_gello_bridge` (rclpy)

설계 원칙:

- **name-based reorder 필수** — `Float64MultiArray`에는 joint 이름이 없다. 순서를 blind index로 넘기면 **조용히 엉뚱한 관절이 움직이는** 최악의 실패 모드가 된다. 반드시 JointState의 `name`으로 UR 순서에 재배열.
- **v1은 커스텀 ros2_control 컨트롤러가 아닌 평범한 rclpy 노드**로 — 단순·디버깅 용이. GELLO 콜백(~25 Hz) 이벤트 구동 + 타이머로 ~100–125 Hz 업샘플(servoj 입력이 더 매끄러움).
- **로봇 현재 자세로 시딩** — 첫 명령이 로봇 실제 자세에서 출발하도록 `/joint_states`로 `last_cmd`를 초기화(handshake의 2차 방어선).

코드 스케치:

```python
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

UR_JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

class UrGelloBridge(Node):
    def __init__(self):
        super().__init__("ur_gello_bridge")
        self.alpha       = self.declare_parameter("ema_alpha", 0.3).value        # EMA 계수
        self.max_step    = self.declare_parameter("max_step_rad", 0.05).value    # 사이클당 최대 이동 [rad]
        self.stale_sec   = self.declare_parameter("stale_timeout_s", 0.5).value  # watchdog
        rate             = self.declare_parameter("publish_rate_hz", 100.0).value

        self.target = None      # GELLO 최신 목표 (UR 순서)
        self.filtered = None    # EMA 상태
        self.last_cmd = None    # 마지막 publish 명령 (로봇 현재 자세로 시딩)
        self.last_gello_t = None

        self.create_subscription(JointState, "/gello/joint_states", self.gello_cb, 1)
        self.create_subscription(JointState, "/joint_states", self.robot_cb, 1)   # 시딩용
        self.pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 1)
        self.create_timer(1.0 / rate, self.tick)

    @staticmethod
    def reorder(msg):
        """name 기반 재배열 — blind index 금지."""
        idx = {n: i for i, n in enumerate(msg.name)}
        return np.array([msg.position[idx[j]] for j in UR_JOINT_ORDER])

    def robot_cb(self, msg):
        if self.last_cmd is None and all(j in msg.name for j in UR_JOINT_ORDER):
            self.last_cmd = self.reorder(msg)   # 로봇 현재 자세로 1회 시딩

    def gello_cb(self, msg):
        try:
            self.target = self.reorder(msg)
        except KeyError as e:
            self.get_logger().error(f"joint missing in /gello/joint_states: {e}")
            return
        self.last_gello_t = self.get_clock().now()

    def tick(self):
        if self.target is None or self.last_cmd is None:
            return  # 유효 GELLO/로봇 상태 수신 전에는 아무것도 publish하지 않음
        age = (self.get_clock().now() - self.last_gello_t).nanoseconds * 1e-9
        if age > self.stale_sec:
            # fail-closed: 스트리밍 중지 + 종료. 조용히 hold하지 않는다.
            self.get_logger().fatal(f"GELLO stale {age:.2f}s > {self.stale_sec}s — shutting down")
            # (선택) /controller_manager/switch_controller로 scaled_jtc 복귀 호출
            raise SystemExit
        if self.filtered is None:
            self.filtered = self.last_cmd.copy()
        self.filtered = (1.0 - self.alpha) * self.filtered + self.alpha * self.target  # EMA
        step = np.clip(self.filtered - self.last_cmd, -self.max_step, self.max_step)  # step clamp
        self.last_cmd = self.last_cmd + step
        out = Float64MultiArray(); out.data = self.last_cmd.tolist()
        self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(UrGelloBridge())
```

파라미터 튜닝 가이드: `max_step_rad=0.05` @100 Hz ≈ 최대 5 rad/s — 실기 초기에는 0.02(≈2 rad/s)로 낮춰 시작 권장. `ema_alpha`는 낮을수록 부드럽고 지연 증가.

### 5.2 `robotiq_urcap_node` (rclpy)

리포에서 이미 실동작하는 URCap 소켓 드라이버를 그대로 재사용한다 — ROS2 의존성이 전혀 없는 순수 Python 클래스라 rclpy 노드에서 바로 import 가능.

```python
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from gello.robots.robotiq_gripper import RobotiqGripper   # 기존 코드 재사용

class RobotiqUrcapNode(Node):
    def __init__(self):
        super().__init__("robotiq_urcap_node")
        ip    = self.declare_parameter("robot_ip", "192.168.1.102").value
        self.speed = self.declare_parameter("speed", 255).value
        self.force = self.declare_parameter("force", 100).value
        self.min_delta = self.declare_parameter("min_delta_pos", 3).value  # 명령 rate-limit

        self.gripper = RobotiqGripper()
        self.gripper.connect(ip, 63352)      # URCap TCP socket — UR 컨트롤러 IP 그대로
        self.gripper.activate()
        self.last_pos = None

        self.create_subscription(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", self.cb, 1)

    def cb(self, msg):
        percent = float(np.clip(msg.data, 0.0, 1.0))     # 1.0 = 완전 open (width percent)
        pos = int(round((1.0 - percent) * 255))          # Robotiq: 0=open, 255=closed
        if self.last_pos is not None and abs(pos - self.last_pos) < self.min_delta:
            return                                       # 소켓 플러딩 방지
        self.gripper.move(pos, self.speed, self.force)
        self.last_pos = pos
```

- 극성: publisher의 percent는 **width**(1=열림), Robotiq POS는 0=열림/255=닫힘이므로 `1-percent` 반전이 기본. Phase 3에서 실기로 확인 후 반대면 부호만 수정.
- 2F-85는 UR 컨트롤박스에 물려 있고 URCap이 63352 포트를 열어주므로 **ROS PC에는 아무 하드웨어도 추가되지 않는다**.

## 6. 안전 handshake & watchdog (MANDATORY)

**근거**: `forward_position_controller`는 (target − current)/dt로 암묵적 속도를 계산한다. **활성화 시점에 로봇이 GELLO의 현재 자세에 있지 않거나, 한 사이클에 큰 점프가 들어오면 joint-velocity-limit fault로 즉시 정지**한다(공식 드라이버에서 확인된 실버그, issue #1277). 아래 절차는 생략 불가.

### 기동 프로토콜 (매 세션)

1. **드라이버 bringup** — `scaled_joint_trajectory_controller` 활성, `forward_position_controller`는 `--inactive`로 로드만.
2. **GELLO publisher 기동** — `/gello/joint_states`가 **신선하고(stale 아님) 일관된** 샘플 수 회 이상 들어올 때까지 대기.
3. **move-to-start** — `/scaled_joint_trajectory_controller`에 **GELLO의 현재 자세**를 목표로 하는 단일-포인트 FollowJointTrajectory goal 1개 전송. `time_from_start`는 보수적으로(관절 속도 ≤ 0.2–0.3 rad/s — Franka 스택의 MotionGenerator speed_factor 0.2에 상응). SUCCEEDED 대기 후 실제 관절값이 tolerance 이내인지 재확인.
4. **컨트롤러 스위치**:

   ```bash
   ros2 control switch_controllers \
     --deactivate scaled_joint_trajectory_controller \
     --activate forward_position_controller
   ```

5. **스트리밍 개시** — bridge 기동: GELLO → name reorder → EMA + step clamp → `/forward_position_controller/commands`.
6. **watchdog** — `/gello/joint_states`가 **0.3–0.5 s 이상 stale이면 publish 즉시 중단**하고 `scaled_joint_trajectory_controller`로 복귀 스위치 및/또는 bridge 종료 (**fail-closed**). 마지막 값을 조용히 hold하는 것 금지.
7. **protective stop 발생 시** — 자동 해제 금지. 원인 해결 → 드라이버 서비스로 dismiss → `resend_robot_program` → **2번부터 handshake 재실행** (정지 동안 로봇 자세가 GELLO와 어긋났을 수 있음).

### 백스톱 (절대 우회 금지)

- **하드웨어 e-stop** — 항상 손 닿는 곳에.
- **UR 온보드 dual-channel Safety Configuration** — PolyScope에서 joint/speed/force limit을 작업공간에 맞게 설정. 소프트웨어가 다 실패해도 이것이 마지막 방어선.
- ⚠️ `forward_position_controller`/`forward_velocity_controller`는 **speed-scaling을 존중하지 않는다**(펜던트 슬라이더·안전 감속 무시). `speed_scaling_state_broadcaster`를 자체 모니터링해 감속 상태에서는 스트리밍을 멈추거나 clamp를 더 조일 것.
- **GELLO passive 원칙** — `dynamixel_torque_enable` 전부 0, 종료 시 torque disable. 어떤 Phase에서도 예외 없음.

## 7. 열린 결정 / 리스크 (Open Questions)

| # | 항목 | 내용 | 권장 |
|---|---|---|---|
| 1 | **offset 컨벤션 (최대 캘리 리스크)** | Python(`configs/rwh_ur.yaml`)과 ROS2 publisher의 offset 처리 방식이 다름(2π 래핑 vs assembly_offsets + align-to-start). sim에서 검증한 `[π, 3π/2, π/2, 3π/2, 3π/2, π]`가 ROS2 경로에서 그대로 맞는다는 보장 없음 | 우선 이식 → Phase 1 검증에서 어긋나면 `ros2/.../scripts/get_offsets.py`로 재측정. **URSim 단계에서 반드시 판명** (실기 전) |
| 2 | 패키징 | bridge·gripper 노드·launch·config를 어디에 두나 | 신규 ros2 패키지 **`ur_gello_bringup`** 생성 권장 (`ros2/src/` 하위) — 기존 Franka 패키지 오염 방지, launch 하나로 publisher+bridge+gripper 일괄 기동 가능 |
| 3 | 그리퍼 Option B vs A | B = URCap 소켓 직결(본 계획, 하드웨어 추가 0) vs A = PickNik `ros2_robotiq_gripper` + UR Tool Communication Forwarder URCap(`/tmp/ttyUR`) | **v1은 B**. MoveIt/ros2_control 수준의 그리퍼 통합이 필요해지는 시점에만 A 재검토 |
| 4 | move-to-start 위치 | handshake 3단계를 bridge 내부에 넣나(액션 클라이언트 내장, Franka 컨트롤러 방식) vs 별도 handshake 스크립트로 두나 | v1은 **별도 스크립트/수동** — 상태머신 단순화, 단계별 사람 확인 가능. 운용이 안정되면 bridge에 내장해 원커맨드화 |
| 5 | staleness 시 복귀 방식 | watchdog 발동 시 단순 종료 vs `switch_controller`로 scaled_jtc 자동 복귀 | v1은 종료(fail-closed) + 수동 복귀. 자동 복귀 서비스 호출은 v2 |
| 6 | 업샘플 rate | 25 Hz 그대로 vs 100–125 Hz 업샘플 | 100 Hz 타이머로 시작(코드 스케치 기준) — hold-last-value 대비 servoj 입력이 매끄러움. URSim에서 비교 후 확정 |
| 7 | speed-scaling 미준수 | forward 계열 컨트롤러가 감속/스케일링 무시 | `speed_scaling_state_broadcaster` 구독 모니터를 v1 범위에 포함할지 결정 — 최소한 값 로깅은 넣을 것 |
| 8 | 실기 IP/네트워크 | UR 컨트롤박스 IP·서브넷 미정 | 실기 도착 시 확정, `<UR_IP>` 플레이스홀더 치환 |
