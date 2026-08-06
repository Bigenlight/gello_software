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
| **RL 액션** `action[6]` (HIL-SERL) | **−1 / 0 / +1** | 🔴 **+1=OPEN, −1=CLOSED — 위 전부와 반대** (0=유지) | `recorded_demo.py:559-570`, `ur7e_env.py:1843-1867` |

> 🔴 **규약이 두 개고 서로 반대다.** 토픽·녹화 컬럼(`gello_grip`/`grip_cmd`/`grip_pos`)은 **0=열림, 1=닫힘**인데,
> RL 액션 `action[6]`은 **+1=열림, −1=닫힘**이다. 변환기가 뒤집는다:
> `trigger >= 0.7 → latch = −1.0`(닫힘), `trigger <= 0.3 → latch = +1.0`(열림).
> **녹화는 언제나 0/1 규약이다** — ±1은 오프라인 RL 데모 변환 단계에서만 나타난다.
> 이 뒤집기를 놓치면 그리퍼가 정확히 거꾸로 동작하고, 값 범위가 둘 다 유효해서 **에러가 나지 않는다.**

**변환 함수 위치** (`robotiq_gripper_modbus_node.py`):
- `_m_to_pos`(:219): `round((1 - m/0.085)*255)` — 미터 → 255 tick
- `_pos_to_m`(:223): `(1 - pos/255)*0.085` — 255 tick → 미터
- `_effort_to_force`(:226): N ↔ 255 tick
- percent → tick(:369): `round(pct*255)`

> ⚠️ 토픽 이름에 `percent`가 들어가지만 값은 **0.0~1.0 fraction**이지 0~100%가 아니다.

### 5.1 이산화(열림/닫힘) 임계값 — **0.3 / 0.7**, 리더 방아쇠 기준

연속 방아쇠값을 열림/닫힘 두 상태로 접을 때 쓰는 정본 상수다. 이미
`recorded_demo.py:559-570`이 오프라인 RL 데모 변환에서 쓰고 있고, 실시간 teleop의
discrete 모드도 **같은 값을 재사용한다**(새로 정하지 말 것):

```
trigger >= 0.7  ->  CLOSED
trigger <= 0.3  ->  OPEN
그 사이         ->  직전 상태 유지        <- 히스테리시스. 단일 임계값이면 경계에서 채터링한다
```

**실측 근거 (2026-07-31, take 125개 / 52,607 샘플 전수):** 방아쇠가 **정지해 있는**
구간(연속 8샘플 이상 변동 <0.02)을 전부 뽑아 보면 값이 **양 극단에만 있고 중간에는
하나도 없다.** 오늘(07-31) 34개 take 기준 정지 구간 106개 = 열림 70(<0.05) / 닫힘
36(>0.95) / 중간 **0**. 전체 샘플의 90.95%가 0.02 미만 또는 0.98 초과이며, 중간
9%는 특정 값에 몰리지 않고 **균등하게 퍼져 있다** — 머문 게 아니라 지나가는 중이라는 뜻.
알고리즘을 전 데이터에 재생하면 임계값에 도달하지 못한 take는 **0개**, 불감대 체류
시간은 평균 1.8%, 최장 모호 구간 0.35 s, 전환은 take당 정확히 2회(잡고→놓고 1사이클)다.

> 🪤 **이 임계값이 존재하는 이유 — 방아쇠가 가끔 덜 열린 채로 멈춘다.** 날짜별 최악의
> '열림 정지값'과, 그 값이 **얼마나 드문지**를 함께 봐야 한다:
>
> | 날짜 | 열림 최악 | 0.3까지 여유 | 열림 정지구간 중 <0.05 | 0.15~0.30 | take 최소값이 0.05 넘는 take |
> |---|---|---|---|---|---|
> | 2026-07-07 | 0.229 | 0.071 | 122 / 125 | **1** | **0 / 51** |
> | 2026-07-20 | **0.243** | **0.057** | 48 / 49 | **1** | **0 / 24** |
> | 2026-07-24 | 0.000 | 0.300 | 17 / 17 | 0 | 0 / 16 |
> | 2026-07-31 | 0.001 | 0.299 | 70 / 70 | 0 | 0 / 34 |
>
> ⚠️ **캘리브레이션 문제가 아니다.** take **125개 전부**에서 방아쇠가 정확히 `0.000`에
> 도달한다(마지막 열: 최소값이 0.05를 넘는 take는 한 개도 없다). 캘리브레이션이 밀렸다면
> 바닥이 통째로 들려 0에 못 간다. 높은 정지값은 **세션당 1~2회 나오는 산발적 사건**이다
> (07-07은 125개 중 1개, 07-20은 49개 중 1개) — 스프링 복원이 덜 됐거나 손이 방아쇠에
> 걸친 것으로 보이며, 원인이 무엇이든 **GELLO가 "24% 닫힘" 값을 내보내고 브릿지가 그걸
> 충실히 전달해 그리퍼가 덜 열린다.** 이산화가 이것을 고친다: 0.3 미만은 전부 정확히
> `0.0`이 된다.
>
> ✅ **관측된 이상값은 전부 0.3 아래다** — 열림쪽 정지구간이 `0.30~0.50`에 놓인 사례는
> 125개 take 전체에서 **0건**. 만약 0.3을 넘었다면 이산화는 오히려 직전 상태(닫힘)를
> 유지해 더 나빴을 것이므로, 이 사실이 임계값 선택의 근거다.
> 그래도 여유가 0.057까지 좁아진 적이 있으므로 임계값은 상수로 박지 말고 **파라미터로
> 노출**하고, **현재 이산 상태를 GUI에 표시**해 조작자가 즉시 알아채게 한다.
>
> 🔴 **"그리퍼가 덜 열린다"의 원인은 둘이고, 이산화는 그중 하나만 고친다.** 위는 **방아쇠가
> 0.2대 값을 내보내는** 경우다. 이와 **똑같아 보이지만 전혀 다른** 두 번째 원인이 있다 —
> 방아쇠는 `0.00`을 제대로 내는데 **셋포인트가 전달 경로에서 유실되는 것**(2026-08-06 실기:
> 14회 중 7회). 이산화로는 안 고쳐진다. 가르는 법은 **`trig` 값 하나**다:
> 0.2대면 여기(§5.1), `0.00`인데 덜 열려 있으면
> [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) §9다.
> 그리퍼를 재캘리브레이션하면 이 표를 다시 재라.

> ⚠️ **측정값 `grip_pos`는 이산화되지 않는다 — 그리고 1.0에 도달하지도 않는다.**
> 물체를 쥐면 손가락이 거기서 멈추므로 실측 최댓값은 **0.506**이었다(빈 손 완전 닫힘은
> ~0.90 = POS 229, `robotiq_gripper_modbus_node.py:305`). 즉 위치만으로는 **잡았다와
> 헛잡았다를 구분할 수 없다.** 이 연속 신호가 그 정보를 담은 유일한 채널이므로
> **이산화하거나 버리지 말 것.** 이산화는 **명령(`grip_cmd`)에만** 적용한다.

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
