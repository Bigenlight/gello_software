# GELLO + ROS2 제어 레퍼런스

> **이 문서는 레퍼런스/비교 문서입니다** (FR3 스택 vs Panda 도커 스택의 구조·게인·데이터 흐름·마이그레이션 로드맵). 실제로 **GELLO로 Franka를 RViz2에 띄우는 단계별 how-to**는 → **[GELLO_FRANKA_RVIZ.md](./GELLO_FRANKA_RVIZ.md)**를 보세요. 여기(control-reference)는 그 배경·근거를 담습니다.

> 실제 Franka를 ROS2로 붙이기 위한 조사 결과. 소스 두 곳:
> ① 이 리포의 `ros2/` 패키지 (공식 `franka_ros2`, **FR3** 타겟)
> ② 조민제 도커 이미지 `minje227/gello:RWH_Gello_PandaV1.0` (`multipanda_ros2`, **Panda** 타겟)
>
> **안전 (변함없음):** GELLO는 항상 passive(읽기 전용). 양쪽 ROS2 스택 모두 `dynamixel_torque_enable: [0]*8`로 토크 비활성. 종료 시에도 `disable_torque()` 호출. XL330에 토크/파워 인가 금지 원칙 유지됨.

---

## 0. 결론 먼저 (TL;DR)

- **gello + ROS2 제어 레퍼런스는 풍부하게 존재한다.** 두 개의 독립적인 완성형 스택이 있음 (코드 공유는 전혀 없음).
- **우리 기체는 Panda** (`panda.xml`, `configs/rwh_panda.yaml`, `panda_arm.urdf.xacro`). → **조민제 도커의 `multipanda_ros2` Panda 스택이 정답 경로.** 이 리포의 `ros2/` FR3 스택은 Panda에 그대로 안 맞음 (joint 이름·게인·libfranka 버전 다름).
- **캘리브레이션은 이미 재사용 가능.** 도커의 `rwh_panda_ros2.yaml`이 우리 `configs/rwh_panda.yaml`과 동일 값 (offsets/signs/gripper). 재측정 불필요.
- **로봇 없이 테스트 가능.** 도커 `rwh_gello_rviz.launch.py` = 턴키 RViz 모드. 실제 GELLO 흔들면 RViz 속 Panda가 1:1로 따라옴.

---

## 1. 두 스택 비교

| 항목 | 이 리포 `ros2/` (FR3) | 도커 multipanda (Panda) |
|---|---|---|
| 로봇 | Franka Research 3 | **Franka Panda** ✅(우리) |
| HW 인터페이스 | 공식 `franka_ros2 v2.1.0` + `franka_hardware` | `franka_control2` (multipanda) |
| 컨트롤러 | `JointImpedanceController` (C++) | `RwhGelloJointImpedanceController` (C++) |
| 제어 루프 | 1000 Hz (`ros2_control_node`) | 1000 Hz (`franka_control2_node`, SCHED_FIFO pri=50) |
| k_gains (J1–7) | **[240,240,240,240,100,60,20]** | [24,24,24,24,10,6,2] |
| d_gains | [20,20,20,10,10,10,5] | [2,2,2,1,1,1,0.5] |
| libfranka | 0.18.2 (FR3용) | ~0.9.x (Panda FCI) |
| URDF / joint 이름 | `fr3/fr3.urdf.xacro` / `fr3_joint1..7` | `panda_arm.urdf.xacro` / `panda_joint1..7` |
| 실행 진입점 | `franka_fr3_arm_controllers.launch.py` | `rwh_gello_realtime.launch.py` (+ 대시보드 `.sh`) |
| GELLO config | `franka_gello_single.yaml` | `rwh_panda_ros2.yaml` (우리 캘리브레이션) |
| 로봇 없이 테스트 | YAML에서 `use_fake_hardware:=true` 편집 | `rwh_gello_rviz.launch.py` 턴키 |
| 검증 상태 | 유지보수됨 | **실동작 로그 있음** (2026-06-18, robot_ip 172.29.0.2) |

> ⚠️ **게인 섞지 말 것.** FR3 게인(240)을 Panda에 넣으면 위험. Panda는 [24,...] 사용.

---

## 2. 공통 데이터 흐름 (양쪽 동일 구조)

```
GELLO (XL330 ×8, passive, U2D2 FTBEO75A @57600)
  │ USB
  ▼
gello_publisher  [franka_gello_state_publisher]  25 Hz
  ├─ gello/joint_states                              (sensor_msgs/JointState, 7축)
  └─ gripper/gripper_client/target_gripper_width_percent  (std_msgs/Float32, 0~1)
  │
  ├──────────────► [팔] 임피던스 컨트롤러 (controller_manager, 1000 Hz)
  │                  gello/joint_states 구독 (queue 1, index로 읽음)
  │                  기동: 유효 GELLO 대기 → MotionGenerator로 부드럽게
  │                        현재 자세→GELLO 자세 이동 (speed_factor 0.2)
  │                  제어식: τ = K·(q_gello − q) − D·dq_filt
  │                          dq_filt = (1−α)·dq_filt + α·dq  (α=0.99)
  │                  명령: panda_joint{1..7}/effort (순수 토크)
  │                  안전: GELLO 0.5s 끊기면 rclcpp::shutdown()
  │                          → franka_control2/ros2_control → libfranka FCI → 로봇
  │
  └──────────────► [그리퍼] gripper_client  [franka_gripper_manager]
                     기동 시 Homing 액션 → max_width 측정 (Panda 실측 0.0806 m)
                     percent×max_width → panda_gripper/move 액션 (단위: m)
                       → franka_gripper → 로봇 핸드
```

**rate 매칭:** GELLO 25Hz publish vs 컨트롤러 1000Hz → 사이엔 보간 없음. 마지막 `q_goal`을 40번 재사용.

---

## 3. GELLO publisher (양쪽 공통, `franka_gello_state_publisher`)

- 노드 `gello_publisher` @25Hz. Dynamixel `DynamixelDriver`로 8모터(arm 1~7 + gripper 8) 읽음, baud **57600**, **XL330**(`xl330.yaml`), U2D2/OpenRB-150 둘 다 지원.
- 1ms 백그라운드 폴링(GroupSyncRead) → 25Hz 콜백에서 가공 후 publish.
- 팔 가공: `assembly_offsets`·`joint_signs` 적용 + **delta 누적 추적**(전원-자세 무관). 그리퍼: `(raw − range[0])/(range[1] − range[0])` → [0,1] 클립.
- **안전 3중:** config `dynamixel_torque_enable:[0]*8` + 코드 기본값 `[0]*7` + 종료 훅 `disable_torque()`.

### 캘리브레이션 — Python ↔ ROS2 대응 (개념 동일)

| Python (`configs/rwh_panda.yaml`) | ROS2 (`rwh_panda_ros2.yaml`) |
|---|---|
| `joint_offsets:[4.712,3.142,3.142,3.142,0,3.142,3.142]` | `assembly_offsets:` 동일값 |
| `joint_signs:[1,-1,1,-1,1,-1,1]` | 동일 |
| `gripper_config:[8,115.376,73.576]` (deg) | `gripper_range_rad:[1.2841,2.0137]` (rad 환산) |
| `start_joints:[0,0,0,-1.57,0,1.57,0,1.0]` | 동일 |
| port FTBEO75A | 동일 |

> 차이: Python은 init 때 `start_joints`로 2π 래핑, ROS2는 `_align_offsets_to_start_joints()`로 offset에 흡수(전원-자세 독립). **값은 같음 → 재측정 불필요.**

---

## 4. 조민제 Panda 실로봇 스택 (도커, 권장 경로)

### 한 방 실행 — `rwh_gello_realtime_dashboard.sh`
환경변수(기본): `ROBOT_IP=172.29.0.2`, `USE_RVIZ=true`, `GELLO_CONFIG_FILE=rwh_panda_ros2.yaml`, `GELLO_COM_PORT=/dev/ttyUSB0`.
순차 기동: ① GELLO publisher(`/gello/ros2`) → (3s) ② 팔 `rwh_franka.launch.py` → (0.5s) ③ 그리퍼 client. 각 단계 로그로 readiness 확인 후 진행.

### 컨트롤러 게인 — `rwh_single_controllers.yaml`
```yaml
rwh_gello_joint_impedance_controller:   # @1000 Hz
  arm_id: panda
  k_alpha: 0.99
  k_gains: [24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0]
  d_gains: [ 2.0,  2.0,  2.0,  1.0,  1.0, 1.0, 0.5]
```
순수 토크 명령(`panda_joint{1..7}/effort`), move-to-start speed_factor 0.2, staleness 0.5s 초과 시 `rclcpp::shutdown()`.

### 도커가 upstream에서 고친 부분 (`franka_gello_state_publisher`)
1. **`start_joints` 파라미터 추가** + `_align_offsets_to_start_joints()` — 전원 시 자세에 무관하게 published 위치 일정.
2. **팔 클램핑 제거** — `process_arm_joint_positions`에서 FR3 limit `np.clip` 삭제 (leader는 사람이 든 그대로 보고).
3. **`com_port` launch 인자 + `resolve_com_port()`** — `com_port:=/dev/ttyUSB0` 런타임 오버라이드.
4. **`rwh_panda_ros2.yaml`** — 우리 캘리브레이션을 ROS2 포맷으로 포팅.

---

## 5. 로봇 없이 테스트하기

### A. 도커 턴키 RViz (가장 쉬움 — 권장)
```bash
xhost +local:docker
docker run --rm -it --privileged \
  --device=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0 \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  minje227/gello:RWH_Gello_PandaV1.0 bash -c "
    source /workspace/panda_ros2_ws/install/setup.bash && \
    source /gello/ros2/install/setup.bash && \
    ros2 launch franka_bringup rwh_gello_rviz.launch.py \
      gello_config_file:=rwh_panda_ros2.yaml \
      gello_com_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0 \
      launch_gello:=true"
```
→ RViz 속 Panda가 실제 GELLO를 실시간 미러링. Franka 불필요. (`use_fake_hardware:=true`, `robot_ip:=dummy`)

### B. 도커 full-stack fake-hardware (컨트롤러까지 검증)
`rwh_gello_realtime.launch.py`에 `use_fake_hardware:=true fake_sensor_commands:=true use_rviz:=true` → 컨트롤러+publisher+gripper 파이프라인 전체를 로봇 없이 점검.

### C. 도커 MuJoCo 물리 sim
`franka_sim.launch.py` (`mujoco_ros2_control`, `franka_description/mujoco/franka/scene.xml`) — multipanda 빌드 필요.

> 이 리포 `ros2/`(FR3)도 `use_fake_hardware:=true`+`use_rviz:=true` 지원하지만 Gazebo는 없음.

---

## 6. 마이그레이션 로드맵 (현재 MuJoCo → ROS2 실로봇)

**현재:** `experiments/launch_yaml.py` + `configs/rwh_panda.yaml` (ROS2 아님, ZMQ). 팔 7축+그리퍼 sim 텔레오퍼레이션 동작 검증됨.

| 단계 | 내용 |
|---|---|
| **0. 로봇 모델 확인** | Panda 맞는지 확정 (정황상 Panda). 이게 모든 후속 단계를 가름. |
| **1. RViz 테스트** | §5-A 도커 턴키. GELLO 흔들어 RViz Panda 추종 확인 → 캘리브레이션 검증 끝. |
| **2. fake-hw full-stack** | §5-B로 컨트롤러 게인·move-to-start·staleness 동작 확인. |
| **3. 실로봇 준비** | Desk UI에서 joint unlock + **FCI 활성화**, 로봇 IP(172.29.0.2) 이더넷, **PREEMPT-RT 커널** 권장(1kHz 안정성). |
| **4. 실로봇 실행** | `rwh_gello_realtime.launch.py robot_ip:=172.29.0.2 use_fake_hardware:=false ...` 또는 대시보드 `.sh`. **GELLO를 편한 자세로 잡고** 시작(로봇이 그 자세로 부드럽게 이동). |
| **5. 캘리브레이션** | `rwh_panda_ros2.yaml` 그대로 사용. 재측정은 GELLO 재조립 시에만. |

### 권장: 빌드 대신 도커 재사용
multipanda는 Panda 전용 libfranka(~0.9.x)에 묶여있어 호스트 빌드는 버전 리스크 큼. **`docker run --network=host`로 도커를 그대로 실로봇에 쓰는 게 안전.**

### 열린 결정/리스크
- **FR3 vs Panda**: 이 리포 `ros2/`는 FR3 전용(`gello_publisher.py`가 `fr3_joint1..7` 하드코딩). Panda면 도커 경로.
- **libfranka 버전**: Panda ≤0.9.x vs FR3 0.13+. 로봇 펌웨어와 일치 필수.
- **RT 커널**: 실로봇 FCI 1kHz 안정에 필요.
- **`rwh_panda_ros2.yaml`**: 로컬 리포엔 없음 → 도커에서 복사하거나 우리 값으로 생성.

---

## 부록: 핵심 파일 경로

**이 리포 (FR3):**
- `ros2/src/franka_gello_state_publisher/...` — publisher 노드, dynamixel 드라이버(XL330 yaml 기반, Python `gello/`와 별개)
- `ros2/src/franka_fr3_arm_controllers/src/joint_impedance_controller.cpp` / `config/controllers.yaml`
- `ros2/src/franka_gripper_manager/franka_gripper_manager/franka_gripper_client.py` (Franka Hand) / `robotiq_gripper_client.py` (2F-85)
- `ros2/README.md`, `ros2/.devcontainer/Dockerfile` (Humble, libfranka 0.18.2, franka_ros2 v2.1.0, franka_description 1.3.0)

**도커 (Panda):**
- `/workspace/panda_ros2_ws/.../franka_bringup/launch/real/{rwh_gello_realtime,rwh_franka,rwh_gello_rviz}.launch.py`
- `/workspace/panda_ros2_ws/.../franka_example_controllers/src/subscriber/rwh_gello_joint_impedance_controller.cpp`
- `/workspace/panda_ros2_ws/.../franka_bringup/config/real/rwh_single_controllers.yaml`
- `/workspace/panda_ros2_ws/.../franka_control2/src/franka_control2_node.cpp` (커스텀 RT 루프)
- `/gello/ros2/src/franka_gello_state_publisher/config/rwh_panda_ros2.yaml` (우리 캘리브레이션)
- `/workspace/panda_ros2_ws/{,src/multipanda_ros2/franka_bringup/scripts/}rwh_gello_realtime_dashboard.sh`
