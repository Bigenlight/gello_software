# GELLO ↔ UR7e 단위(Unit) 레퍼런스

> **목적:** teleop(조인트/EEF) · 녹화(HDF5) · 제어(UR7e 인터페이스) 전 경로에서 각 물리량이 **어떤 단위**로 표현/전달/저장되는지 코드 근거로 전수 정리한 문서.
> 데이터셋 변환, 정책 학습 입력 정규화, 디버깅 시 "이 값이 rad냐 deg냐 / m냐 mm냐"를 확정하기 위한 단일 참조점.
>
> 조사 기준 브랜치: `feat/gello-ur7e-humble-22.04`. 모든 경로/라인은 조사 시점 기준이며, 리팩터 시 라인 번호는 이동할 수 있음(심볼명으로 재확인 권장).

---

## 0. 결론 먼저 (TL;DR)

- **표준 규약: 각도 = rad, 각속도 = rad/s, 위치 = m(미터), 선속도 = m/s, 그리퍼 = 0~1 정규화.** mm·deg는 시스템 어디에도 흐르지 않는다.
- `deg`가 등장하는 곳은 **단 1곳**(GELLO 그리퍼 open/close config 입력) → 즉시 rad로 변환 후 다시 0~1 정규화.
- `mm`는 **어디에도 없다.** UR/Robotiq 규약상 미터를 쓴다 (그리퍼 stroke = 0.085 m = 85 mm이지만 코드 값은 **0.085 m**).
- EEF 모드라도 UR로 나가는 **최종 명령은 Cartesian(servoL)이 아니라 관절 위치(rad)**. Cartesian pose는 진단 토픽에만 존재.
- HDF5 파일에는 **단위 attribute가 저장되지 않는다** → 단위는 이 문서/컬럼명 규약에 의존.

---

## 1. 조인트(Joint) Teleop 경로

| 신호 | 단위 | 변환/근거 |
|---|---|---|
| GELLO 리더 조인트(6축) | **rad** | Dynamixel tick→rad: `/2048*π` (=2π/4096) — `gello/dynamixel/driver.py:520` |
| 리더 조인트 속도 | **rad/s** | `driver.py:517` `*0.229*2π/60` |
| joint offsets | **rad** (π/2 배수) | `gello/robots/dynamixel.py:110`, config는 `np.pi` 배수 |
| joint signs | **±1 (무차원)** | `dynamixel.py:68-70` `abs==1` assert |
| 팔로워 명령(6축) | **rad** | 비-ROS `dynamixel.py:130` / ROS `Float64MultiArray` → `/forward_position_controller/commands` |
| 그리퍼 채널 | **0~1 정규화** (0=open, 1=closed) | `dynamixel.py:113-119` clamp[0,1] |

**`joint_delta` (start-anchored relative) 모드** — `ur_gello_bringup/joint_delta.py`

| 신호 | 단위 | 근거 |
|---|---|---|
| 델타 δ | **rad** | `joint_delta.py:369` `wrap_to_pi(q_in - q_lead_prev)` |
| `jd_gain` | **무차원** | `joint_delta.py:396` `anchor + gain*delta`; 검증범위 0.0~2.0; 기본 1.0 |
| 출력 `q_cmd` | **rad** | → `Float64MultiArray` |
| `jd_limit_margin_rad` / `jd_max_excursion_rad` / `jd_leader_jump_max_rad` / `jd_delta_deadband_rad` | **rad** | `config/ur7e_gello_joint_delta.yaml:61,92,113,139` |
| `jd_max_excursion_rad: 1.0` | 1.0 rad ≈ 57° | yaml:92 |
| `jd_leader_jump_max_rad: 0.35` | 0.35 rad/30Hz ≈ 10.5 rad/s | yaml:113 |

> 내부/ROS 일치: `JointState.position`(numpy 배열)과 `Float64MultiArray` 모두 **rad**, 변환 없음. `joint_delta.py`는 stdlib `math`만 사용(리스트 기반).

---

## 2. EEF (Cartesian delta) Teleop 경로

핵심 파일: `ur_kin.py`(FK/IK/SE3), `eef_delta.py`(앵커/델타/거버너), `config/ur7e_dh.yaml`, `config/ur7e_gello_eef.yaml`

| 신호 | 단위 | 근거 |
|---|---|---|
| FK로 계산한 EEF 위치 (`p_g`, `p_des` …) | **m (미터)** | DH `d,a` = metres — `ur7e_dh.yaml:19-20`, `ur_kin.py:317` |
| 회전 표현 (주) | **3×3 회전행렬** | `eef_delta.py:462-468` |
| 회전 증분 (내부) | **axis-angle, rad** (SE(3) log/exp) | `eef_delta.py:473-486`, `ur_kin.py:111-118` `so3_log` |
| 회전 입력 파라미터 (`r_align_rpy`, `tool_*_rpy`) | **euler rpy, rad** | `eef_delta.py:160-162` |
| 위치 델타 | **m** | `eef_delta.py:466` |
| `pos_scale` | **무차원 게인** (위치항 전용), 기본 1.0 | `eef_delta.py:466`, yaml:74 |
| `rot_scale` | **무차원**, 1.0 고정(다른 값 거부) | `eef_delta.py:152-153` |
| `tool_l`/`tool_r` z = **0.174** | **m** (=174 mm) | `ur7e_gello_eef.yaml:207,213` ("174 쓰지 말 것" 명시) |
| `v_max` (선속도 캡) | **m/s** (기본 0.08) | `eef_delta.py:84,504` |
| `w_max` (각속도 캡) | **rad/s** (기본 0.5) | `eef_delta.py:85,506` |
| `max_excursion_m: 0.5` | **m** | `eef_delta.py:95` |
| `lag_max_pose` = [0.05, 0.3] | **[m, rad]** | `eef_delta.py:94` |
| `max_step_rad` (IK 출력 per-tick 관절 스텝) | **rad/tick** (기본 0.05) | `gello_ur_bridge_node.py:126,1130` |
| `dt` | **초(s)** = 1/publish_rate_hz | `eef_delta.py:96` |

> ⚠️ **EEF 모드의 실제 UR 명령은 관절 위치(rad)**: `EefDeltaController.step()`이 Cartesian 델타를 내부에서 계산 후 branch-locked 해석적 IK로 `q_cmd`(rad)를 반환 → `Float64MultiArray`로 발행. `servoL`/`speedL`/`moveL` 문자열은 저장소 전체에 **0건**. Cartesian pose(PoseStamped, base frame **미터**)는 `~/eef/leader_pose`·`desired_pose`·`commanded_pose` **진단 토픽에만** 게시.

---

## 3. 녹화 (Recording) — HDF5 `vectors.h5`

- 모든 값은 **float64**. `synchronized` 테이블은 기본 **100 Hz** 스냅샷, 카메라 **30 fps**.
- ⚠️ **파일에 단위 attribute 없음** (`attrs["columns"]` 컬럼명 JSON만 저장). 단위는 소스 토픽 규약에서 유래 → 이 문서가 유일한 단위 근거.

| 컬럼 (synchronized) | 단위 | 소스 |
|---|---|---|
| `t_rel_s` | **초(s)**, 세션 시작 기준 상대, `.4f` | `recording_session.py:235` |
| `t_wall` | **초(s)**, Unix epoch, `.4f` | `recording_session.py:235` |
| `gello_q1..6` | **rad**, `.6f` | GELLO leader |
| `gello_qd1..6` | **rad/s**, `.5f` (유한차분) | `gello_ur_recorder_node.py:198-203` |
| `gello_grip` | **0~1** (0=open,1=closed), `.4f` | `gello_hardware.py:249-257` |
| `cmd1..6` | **rad** (UR 조인트 위치 명령), `.6f` | `/forward_position_controller/commands` |
| `ur_q1..6` | **rad**, `.6f` | `/joint_states.position` |
| `ur_qd1..6` | **rad/s**, `.6f` | `/joint_states.velocity` |
| `ur_eff1..6` | **N·m** (추정), `.4f` | `/joint_states.effort` |
| `grip_cmd`, `grip_pos` | **0~1**, `.4f` | `/robotiq_gripper/*_percent` |
| `fx,fy,fz` | **N**, `.5f` | `/force_torque_sensor_broadcaster/wrench` |
| `tx,ty,tz` | **N·m**, `.5f` | 동일 wrench.torque |
| `tcp_x,tcp_y,tcp_z` | **m**, `.6f` | `/tcp_pose_broadcaster/pose.position` |
| `tcp_qx,qy,qz,qw` | **단위 쿼터니언 (무차원)**, `.6f` | 동일 pose.orientation |
| `cam1_frame_idx, cam2_frame_idx` | **정수 프레임 인덱스** (None→NaN) | `recording_session.py:246` |

> 네이티브-레이트 세부 테이블(`gello_joint_states`, `ur_joint_states`, `command`, `gripper`, `wrench`, `tcp_pose`, `cam*_frames`)도 동일 단위 규약. 각 토픽 콜백당 1행.
>
> **주의:** `/joint_positions`, `/control`, `/observations` 같은 LeRobot 계열 데이터셋 이름은 이 녹화 경로에 **없다**. 실제 이름은 위 컬럼(`gello_q*`, `ur_q*`, `cmd*`, `tcp_*` …).

---

## 4. 제어 (Control) — UR7e 인터페이스

**두 개의 분리된 경로가 존재하며 서로 함께 쓰이지 않음:**

| 경로 | 위치 | 드라이버 | 명령 방식 |
|---|---|---|---|
| (A) 레거시 GELLO 원본 | `gello/robots/ur.py` **(유일한 ur_rtde 호출부)** | ur_rtde 직접 | `servoJ` |
| (B) 현재 주력 ROS2 | `ur_gello_bringup/` | ROS2 `ur_robot_driver` + `forward_position_controller` | ROS 토픽 `Float64MultiArray` |

| 항목 | 단위 | 근거 / UR 규약 대비 |
|---|---|---|
| 조인트 명령 `servoJ` / `forward_position_controller` | **rad** | UR 규약 일치 ✓ (`ur.py:83`, node:477) |
| 조인트 상태 `getActualQ` / `/joint_states` | **rad** | UR 규약 일치 ✓ (`ur.py:61`) |
| `servoJ` velocity / acceleration | **rad/s** / **rad/s²** | `ur.py:75-76` (velocity=0.5, accel=0.5) |
| `servoJ` dt / lookahead | **초(s)** | dt=0.002(2ms), lookahead=0.2 |
| 조인트 속도 한계 | **rad/s** (π=3.14 rad/s = 180°/s) | `ur_kin.py:77`, node docstring:13-15 |
| ROS 지속 텔레옵 속도 | max_step_rad × publish_rate_hz = 0.0025×250 = **0.625 rad/s** | `config/ur7e_gello.yaml` |
| Cartesian pose (UR 규약 = m + axis-angle rad) | RTDE로 **미사용** | `getActualTCPPose`/`servoL` 등 호출 0건 |

> ⚠️ **`gello/robots/ur.py::get_observations()` 부정확** (코드상 명백): `joint_velocities`에 조인트 **위치**를 넣고(`ur.py:118`), `ee_pos_quat = np.zeros(7)`로 TCP를 항상 0 반환(`ur.py:119`, 미구현). 이 레거시 경로에서 속도·TCP 단위는 실질적으로 무의미.

---

## 5. 그리퍼 — 계층별 단위 (가장 헷갈리는 부분)

동일 물리량(그리퍼 개폐)이 계층마다 다른 단위로 표현됨:

| 계층 | 단위 | 규약 | 근거 |
|---|---|---|---|
| GELLO 리더 폭 | **0~1** | 0=open, 1=closed | `dynamixel.py:113-119` |
| teleop 스트리밍 (`command_percent`, `target_gripper_width_percent`) | **0~1 fraction** (≠ 0~100%) | 0=open, 1=closed | `gello_gripper_bridge_node.py:16,205-209` |
| 하드웨어 Modbus 레지스터 | **0~255 정수** | 0=OPEN, 255=CLOSED | `robotiq_2f85_modbus.py:23,217` |
| ROS `GripperCommand` 액션 `.position` | **미터(m)** | 0.0=CLOSED, **0.085=OPEN** (mm 아님) | `robotiq_gripper_modbus_node.py:18-19,74` |
| ROS `GripperCommand` `.max_effort` | **뉴턴(N)** | max_force_n=235.0 | `robotiq_gripper_modbus_node.py:75,226-229` |
| JointState 발행 (knuckle) | **rad** | 0~0.8 rad | `robotiq_gripper_modbus_node.py:76,395` |

**변환 함수 위치** (`robotiq_gripper_modbus_node.py`):
- `_m_to_pos`(:219): `round((1 - m/0.085)*255)` — 미터 → 255 tick
- `_pos_to_m`(:223): `(1 - pos/255)*0.085` — 255 tick → 미터
- `_effort_to_force`(:226): N ↔ 255 tick
- percent → tick(:369): `round(pct*255)`

> ⚠️ 토픽 이름에 `percent`가 들어가지만 값은 **0.0~1.0 fraction**이지 0~100%가 아니다.

---

## 6. deg / mm 가 나타나는 유일한 지점

| 위치 | 무엇 | 처리 |
|---|---|---|
| `gello/robots/dynamixel.py:41-42` | GELLO 그리퍼 open/close config가 **deg**로 입력 | `*π/180`로 즉시 rad 변환 → 다시 0~1 정규화. rad는 중간 단계일 뿐, 최종 채널값은 0~1 |

- **조인트 각도의 deg 입력 경로 없음** — 전부 rad.
- **mm 경로 없음** — 그리퍼 stroke는 0.085 m로 표현(85 mm를 미터로).

---

## 7. 내부 numpy vs ROS 메시지 단위 대조표

| 신호 | 내부 (numpy/python) | ROS2 메시지 | 단위 |
|---|---|---|---|
| 리더 조인트(6축) | `get_joint_state()[:6]` | `JointState.position` (`/gello/joint_states`) | rad (동일) |
| 그리퍼 | `get_joint_state()[6]` | `Float32` (`.../target_gripper_width_percent`) | 0~1 (동일) |
| 팔로워 명령(6축) | `command_joint_state` / servoJ | `Float64MultiArray` (`/forward_position_controller/commands`) | rad (동일) |
| joint_delta | δ, gain, q_cmd (stdlib list) | q_cmd → `Float64MultiArray` | δ=rad, gain=무차원, q_cmd=rad |
| grip 명령 | `joint_state[-1]*255` (ur.py) | `Float32` command_percent → `*255` | 0~1 → 0~255 tick |

---

## 8. 추정(코드로 단정 못 한 부분)

- `cmd*`(UR position command)의 rad, `ur_eff*`의 N·m, wrench N/N·m, TCP m — 파일에 단위 리터럴이 없고 **ROS 토픽/컨트롤러 규약 기반 추정**(broadcaster 원본 코드는 이 저장소가 아닌 ur_robot_driver 제공). 단위 판단 자체는 강한 근거.
- EEF FK가 실제 UR7e `getActualTCPPose`와 수치적으로 일치하는지는 **실기 검증 필요** — `config/ur7e_dh.yaml`이 "nominal 값, 실 로봇 factory calibration으로 교체 필요"라고 스스로 명시.

---

## 참조 파일

- `gello/dynamixel/driver.py` — tick↔rad, rad/s
- `gello/robots/dynamixel.py` — 리더 rad / offsets / 그리퍼 0~1 / deg2rad
- `gello/robots/ur.py` — 레거시 ur_rtde(servoJ rad)
- `ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/`: `gello_ur_bridge_node.py`, `joint_delta.py`, `eef_delta.py`, `ur_kin.py`, `angle_utils.py`, `robotiq_gripper_modbus_node.py`, `gello_gripper_bridge_node.py`
- `ros2_ur_ws/src/ur_gello_bringup/config/`: `ur7e_gello.yaml`, `ur7e_gello_joint_delta.yaml`, `ur7e_gello_eef.yaml`, `ur7e_dh.yaml`
- `ros2_ur_ws/src/gello_recorder/gello_recorder/`: `gello_ur_recorder_node.py`, `recording_session.py`, `hdf5_writer.py`
