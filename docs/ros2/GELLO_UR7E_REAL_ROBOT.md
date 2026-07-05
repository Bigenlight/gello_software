# GELLO → 실제 UR7e + Robotiq 2F-85 ROS2 텔레오퍼레이션 — 실기 런북 (REAL)

> ⚠️ **이 문서는 실제 UR7e 하드웨어를 ROS2 Humble로 구동하기 위한 실기(real-robot) 런북입니다.**
> 실 UR7e는 이 빌드-검증 PC가 아니라 **별도의 로봇 PC**에 연결되어 있으며, 이 문서의 절차(move-to-start handshake, External Control/URCapX, 기구학 캘리브레이션, 실 그리퍼 소켓)는 **여기(mock) 환경에서는 검증할 수 없습니다.** 로봇에 붙기 전 반드시 mock 경로로 파이프라인을 먼저 확인하세요.
>
> - **검증 완료된 mock 경로 (여기서 실행 가능)**: [`GELLO_UR7E_ROS2_BRINGUP.md`](./GELLO_UR7E_ROS2_BRINGUP.md) — `source:=fake` + `use_fake_hardware`, RViz2 시각화, 실 UR/GELLO 미접속.
> - **설계·근거 문서**: [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md) (아키텍처·안전 handshake·리스크 결정), [`GELLO_UR_ROS2_BRINGUP.md`](./GELLO_UR_ROS2_BRINGUP.md) (mock UR 브링업 상세), [`GELLO_ROS2_CONTROL_REFERENCE.md`](./GELLO_ROS2_CONTROL_REFERENCE.md).

물리 **GELLO**(6-DOF UR 리더 암, U2D2 `FTBEO6QK`)로 **실제 UR7e + Robotiq 2F-85**를 **ROS2 Humble** 위에서 1:1(radian) 텔레오퍼레이션한다. UR7e는 `ros-humble-ur` **2.8.1**에서 정식 지원되며(`config/ur7e` 존재), 관절 리밋이 UR5e와 동일하므로 노드 코드는 **변경 없이** 그대로 쓴다. 브리지는 로봇 실제 `/joint_states`에서 seed해 시작 스냅을 없애고, 실기 속도 안전을 위해 `max_step_rad=0.0025` + `publish_rate_hz=250`으로 스트리밍한다(스냅 원인·근거는 §2 참고).

> **⭐ 이 버전의 핵심 변경 — 수렴 게이트(convergence-gated) 핸드셰이크.** `gello_move_to_start`의 `gello` 시작 모드는 더 이상 "고정된 첫 GELLO 자세로 **한 번** 이동"하지 않는다. 대신 매 반복마다 **살아있는(live) GELLO 자세 + 로봇 실측 `/joint_states`를 다시 읽어** 따라가는(chase) 궤적을 쏘고, per-joint `|live GELLO − actual| <= chase_tol`가 `chase_dwell_s` 동안 유지될 때에만 핸드오버한다. 동시에 브리지는 `start_paused:=true`로 **미리(paused) 스폰**되어 있다가, STRICT 전환 성공 직후 `gello_move_to_start`가 브리지 `~/resume`를 호출해야 비로소 스트리밍을 시작한다. **정직한 보장(이건 "zero snap" 주장이 아님):** 핸드오버는 팔로워가 **살아있는 리더를 chase_tol 이내로 따라잡아 유지**했을 때만 일어나며, 남는 잔차는 soft-start된 rate-limited slew(`<= max_step_rad*rate`)로 닫힌다. 운영자가 계속 움직이거나 리더가 안 멈추면 **스냅이 아니라 핸드오버가 지연**될 뿐이다. 자세한 로그·복구·캘리브레이션은 §2 참고.
>
> **⚠️ 실기 미검증.** 이 수렴 게이트 경로 전체는 ros2_control **mock** 스택에서만 검증되었고 **실 GELLO+로봇에서는 아직 미검증**이다. mock은 JTC가 정확히 도달하지만 실기는 정상상태 오차가 남을 수 있어, 실기 첫 구동 전 §2의 캘리브레이션 체크리스트(`chase_tol` vs JTC arrival error)를 반드시 수행할 것.

> **안전 대전제 (변함없음, 절대 위반 불가)**: GELLO는 **항상 passive read-only** 입력 장치이다. GELLO Dynamixel에는 **어떤 경우에도 토크를 인가하지 않는다.** 드라이버는 토크를 OFF 상태로 초기화하고 관절 각도를 **읽기만** 한다. 이 런북의 모든 전환·소켓·캘리브레이션 절차는 오직 **UR7e 팔로워 로봇** 쪽에만 적용된다.

## 요약 (TL;DR)

| 항목 | 결정 / 값 |
|---|---|
| 로봇 | **UR7e** (`ur_type:=ur7e`), `ros-humble-ur` **2.8.1** 유지(업그레이드 금지) |
| launch | `ur7e_gello_real.launch.py` — `robot_ip` **필수**(기본값 없음), `use_fake_hardware:=false` |
| 부팅 컨트롤러 | `scaled_joint_trajectory_controller` **active** + `forward_position_controller` **inactive** 로드 |
| **필수 시퀀스** | 드라이버 기동 → `gello_ur_bridge` **pre-spawn PAUSED** → `gello_move_to_start` **수렴 게이트 chase** → STRICT 전환 → `gello_move_to_start`가 브리지 `~/resume` 호출 → 스트리밍 (**§2, 건너뛰면 protective stop**) |
| 팔 명령 경로 | `gello_publisher` → `gello_ur_bridge`(actual-`/joint_states` seed + One-Euro 필터 + step-clamp 0.0025 + **soft-start 0.7s** + watchdog) → `/forward_position_controller/commands` (250 Hz) |
| 그리퍼 | `robotiq_urcap` → UR URCap TCP 소켓(63352) raw 접속, `robot_ip` + `connect_on_start:=true` |
| 캘리브레이션 | GELLO 캘리브(`FTBEO6QK`)는 **그대로 재사용**; UR7e **기구학** 캘리브는 로봇당 1회 별도 추출 |
| 최대 리스크 | `forward_position_controller`는 보간 없이 즉시 명령 → 활성화 시점 로봇 자세 ≠ GELLO 자세면 **joint-velocity-limit protective stop** → §2 handshake 필수 |

### 목차

1. [안전 — GELLO는 항상 passive read-only](#1-안전--gello는-항상-passive-read-only)
2. [Move-to-start handshake (필수)](#2-move-to-start-handshake-필수)
3. [External Control: PolyScope 5 vs PolyScope X](#3-external-control-polyscope-5-vs-polyscope-x)
4. [UR7e 기구학 캘리브레이션 — 최초 1회](#4-ur7e-기구학kinematics-캘리브레이션--최초-1회)
5. [Robotiq 2F-85 그리퍼 — URCap 소켓 (63352)](#5-robotiq-2f-85-그리퍼--urcap-소켓-63352-raw-tcp)
6. [열린 질문 체크리스트 (실기 연결 전 확인)](#6-열린-질문-체크리스트-실기-연결-전-확인)

---

## 1. 안전 — GELLO는 항상 passive read-only

GELLO는 **수동(passive) 모션캡처 리더 암**입니다. Dynamixel 모터에 **절대 토크를 걸지 않습니다.** 이 원칙은 UR5e든 UR7e든, mock이든 실로봇이든 예외 없이 항상 적용됩니다.

- `gello_publisher_node.py`는 `DynamixelRobotConfig.make_robot`으로 드라이버를 초기화하는데, 이 드라이버는 **토크를 OFF 상태로 초기화**하고 관절 각도를 **읽기만** 합니다.
- 노드 종료 시에도 "GELLO is passive; nothing to power down"으로 명시되어 있습니다 — 끌 전원 자체가 없습니다.
- 실기 연결에서 힘이 들어갈 수 있는 쪽은 오직 **UR7e 팔로워 로봇**뿐입니다. 따라서 아래 §2 handshake와 protective-stop 회피가 이 런북의 핵심 안전 절차입니다.

즉, 이 경로 어디에서도 GELLO 모터에는 힘이 들어가지 않습니다. GELLO는 손으로 자유롭게 움직이는 입력 장치이며, 안전 책임은 전적으로 UR7e 측 명령 스트림에 있습니다.

---

## 2. Move-to-start handshake (필수)

**이 절은 실기 구동의 가장 중요한 안전 절차이며, 단 한 단계도 생략할 수 없습니다.**

### 왜 필요한가 — protective stop의 원인

`forward_position_controller`(fpc)는 **보간(interpolation)을 하지 않고, 받은 관절 위치를 즉시 로봇에 명령**합니다. 그래서 fpc가 활성화되는 **바로 그 순간** 로봇의 현재 자세와 GELLO가 지시하는 자세가 다르면, 로봇은 그 큰 각도 차이를 **한 제어 주기 안에 점프(snap)**하려 시도합니다. 이 점프는 순간 관절 속도가 UR7e의 관절 속도 한계(전 관절 180 deg/s)를 초과하게 만들어 UR 컨트롤러의 **joint-velocity-limit protective stop(보호 정지)**을 유발합니다. (이는 계획 문서 §6에서 확인된 실버그입니다 — [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md) 참고.)

해결책은 **fpc를 켜기 전에 로봇을 GELLO의 현재 자세로 부드럽게(보간되는 궤적으로) 먼저 데려다 놓는 것**입니다. 그 역할을 `scaled_joint_trajectory_controller`(궤적 보간 컨트롤러)와 `gello_move_to_start` 노드가 담당합니다.

### 무엇이 바뀌었나 — "한 번 이동"에서 "수렴 게이트 chase"로

이전(舊) `gello` 모드는 **첫 GELLO 메시지 하나를 latch해 그 고정 자세로 궤적 한 번**을 쏘고, SUCCEEDED가 되면 곧바로 STRICT 전환·스트리밍을 시작했습니다. 문제는 **리더가 움직이는 타깃**이라는 점입니다: 팔이 캡처된 그 옛 자세로 이동하는 동안(그리고 운영자가 계속 GELLO를 잡고 있는 동안) 리더는 드리프트/이동하므로, 스트리밍이 시작되는 순간 `live GELLO ≠ 도착 자세`인 **잔차 gap이 남아 fpc 첫 명령에서 snap**이 재발했습니다.

지금(現) `gello` 모드는 `_converge_and_handover()`로 **수렴할 때까지 살아있는 리더를 따라갑니다(chase):**

1. 매 반복마다 **live GELLO 자세**(`/gello/joint_states`)와 **로봇 실측 자세**(`/joint_states`)를 **다시 읽습니다.**
2. per-joint gap을 재고, `chase_tol`보다 크면 **그 gap 크기에 비례한 길이**의 catch-up 궤적을 쏩니다: `T = max(gap / chase_v_budget, min_traj_duration)`. **상한 clamp 없음** — gap이 크면 더 *빠르게*가 아니라 더 *길게* 움직여 JTC 스플라인 peak 속도가 3.14 rad/s protective-stop 한계 아래로 유지됩니다.
3. gap이 `chase_hard_limit`(≈4.0 rad, **wraparound/gross-mispose 백스톱**, approach 제한 아님)을 넘으면 **자동 chase를 거부**하고 fail-safe로 종료(스위치 없음)합니다.
4. per-joint `|live GELLO − actual| <= chase_tol`가 **`chase_dwell_s` 동안 유지**(median-of-5 필터 + dwell 중 *새* 샘플 도착 요구)되면 **"Converged" 로그 후 핸드오버**합니다.
5. `~abort`를 폴링하고, `chase_timeout_s` 안에 수렴 못 하거나(계속 움직이는/안 멈추는 리더) 리더 스트림이 **stale(dead)**이면 **exit 1 fail-safe로 종료(스위치 없음)**합니다.

즉 실패 방향은 항상 **"아직 시작 안 함"**이지 **"예상 못한 이동"**이 아닙니다. 남는 잔차는 브리지의 soft-start된 slew(`<= max_step_rad*rate`)로 닫힙니다 (§2 파라미터 표·복구 참고).

### 실행 전 프리플라이트 (첫 동작 전 필수)

> **경고 — 실 UR7e는 물리적으로 움직입니다. 아래 게이트를 모두 통과하기 전에는 Play/headless 시작을 하지 마세요.** 팔은 대략 **t≈8s**(`gello_move_to_start`가 첫 catch-up 궤적을 쏘는 순간)부터 **살아있는 GELLO 자세를 따라 움직이기 시작**하며, 리더가 멈춰 수렴할 때까지 여러 번의 catch-up 이동이 이어질 수 있습니다.

- [ ] **E-STOP(비상정지)이 손 닿는 거리**에 있는가.
- [ ] **작업공간 / 전체 스윕 경로가 비어 있는가** — 현재 자세의 여유뿐 아니라, 로봇 현재 자세 → GELLO 현재 자세로 이어지는 **관절공간 궤적 전체**가 장애물이 없어야 함(move-to-start는 충돌을 인지하지 못함 — 아래 3단계 경고 참고).
- [ ] **펜던트 우측 하단이 `Real Robot`인가** (`Simulation` 아님) — Simulation이면 실팔이 안 움직임.
- [ ] **GELLO가 `start_joints` 근처 중립 자세로 손에 잡혀 있는가** — handshake가 이 자세로 팔을 이동시키므로, GELLO가 극단 자세면 팔도 그만큼 크게 움직임.
- [ ] **Safety state = Normal인가** — 활성 protective stop / fault 없음(펜던트 안전 상태 확인).
- [ ] **Installation > Payload(질량/CoG)가 Robotiq 2F-85 그리퍼 포함해 올바르게 설정되었는가** (§5) — 페이로드 미설정 시 보호 정지·처짐을 유발.

### 자동 시퀀스 (런치 한 줄)

> **중요 — 아래 전 과정은 `ur7e_gello_real.launch.py`가 staggered `TimerAction` + 이벤트 핸들러로 자동 수행합니다.**
> 런치 한 줄이면: 드라이버 기동(t=0) → `gello_publisher`(t=6s, `TimerAction`) **및 `gello_ur_bridge`가 `start_paused:=true`로 함께 pre-spawn**(t=6s, `TimerAction`) → `gello_move_to_start`(t=8s, `TimerAction`, `resume_bridge:=true`)가 **수렴 게이트 chase → STRICT 전환 → 브리지 `~/resume` 호출**을 노드 내부에서 순서대로 수행 → handshake 프로세스가 **returncode==0 으로 종료되는 순간** `RegisterEventHandler(OnProcessExit)` 콜백으로 **그리퍼(`robotiq_gripper_modbus` + `gello_gripper_bridge`)**가 기동됩니다.
>
> **핵심 차이(구 버전 대비):** 팔 브리지(`gello_ur_bridge`)는 이제 `OnProcessExit`에서 **cold-start되지 않습니다.** 이미 t=6s에 **paused 상태로 존재**하고, `gello_move_to_start`가 STRICT 전환 성공 직후 **직접 `~/resume`를 호출**해 스트리밍을 시작합니다(런치 타이머가 아님). 이렇게 하면 브리지가 **inactive 컨트롤러로 스트리밍하는 창**도, 옛 OnProcessExit 방식의 **cold-start 드리프트 창**(리더가 계속 움직여 핸드오버 gap이 벌어지던)도 사라집니다. 아래 개별 명령은 **각 단계에서 무슨 일이 일어나는지 이해하고, 문제 시 수동 검증/복구**하기 위한 것입니다 (정상 운용 시 개별 실행 불필요).

1. **부팅 — 궤적 컨트롤러 active, fpc inactive, 브리지 pre-spawn PAUSED.**
   `ur7e_gello_real.launch.py`는 드라이버를 `initial_joint_controller:=scaled_joint_trajectory_controller`(+ activate)로 띄우고, `forward_position_controller`는 **INACTIVE(로드만)** 상태로 함께 스폰합니다. t=6s에는 `gello_ur_bridge`가 **`start_paused:=true`(런치 레벨 override, yaml 아님)**로 떠서 `~/pause`·`~/resume` 서비스와 구독만 올리고 **아무것도 발행하지 않습니다.** 이 시점에는 fpc가 명령을 받지 않으므로 스냅이 발생할 수 없습니다.

   ```bash
   # 이 랩 실기 예시 IP: 192.168.10.11 (펜던트에서 실제 값 확인)
   ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
     robot_ip:=192.168.10.11 \
     kinematics_params_file:="/home/laptop3/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_calibration.yaml"
   ```

2. **컨트롤러 상태 확인(sanity).**
   전환 전에 두 컨트롤러가 기대한 상태인지 확인합니다.

   ```bash
   ros2 control list_controllers
   ```

   기대값:
   - `joint_state_broadcaster` → **active**
   - `scaled_joint_trajectory_controller` → **active**
   - `forward_position_controller` → **inactive**

3. **`gello_move_to_start` — 살아있는 GELLO 자세로 수렴 게이트 chase.**
   이 노드는 매 반복마다 **live GELLO 자세**(`/gello/joint_states`)와 **로봇 실측 자세**(`/joint_states`)를 다시 읽어, gap이 `chase_tol`보다 크면 gap 크기에 비례한 길이의 catch-up 궤적을 `/scaled_joint_trajectory_controller/follow_joint_trajectory`(`control_msgs/action/FollowJointTrajectory`)로 반복해 쏩니다. per-joint `|live GELLO − actual| <= chase_tol`가 `chase_dwell_s` 동안 유지되면 **"Converged"** 후 핸드오버합니다. (수렴 로직의 상세는 위 "무엇이 바뀌었나" 참고.)

   ```bash
   # (수동 재현용) — 이 노드는 ros2_control 액션/서비스로만 동작하므로 robot_ip가 필요 없습니다.
   # 목표는 start_joints가 아니라 "매 반복 다시 읽는 live GELLO 자세"입니다.
   # 단독 실행 시 resume_bridge는 기본 False라 브리지를 건드리지 않습니다(런치가 True로 override).
   ros2 run ur_gello_bringup gello_move_to_start --ros-args \
     --params-file /home/laptop3/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml
   ```

   > 관절 순서는 UR 표준 `[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`.
   >
   > ⚠️ **경고 — chase 궤적은 충돌을 인지하지 않습니다(collision-UNAWARE).** 각 catch-up 이동은 단순한 point-to-point **관절공간 보간(interpolated)** 이동일 뿐, 장애물/충돌 회피 계획이 전혀 없습니다. 게다가 리더가 움직이면 **여러 번**의 catch-up이 이어집니다. Play/headless 시작 전에, 로봇 **현재 자세**의 여유만이 아니라 **로봇 → live GELLO로 이어지는 스윕 경로 전체**가 장애물 없이 비어 있는지 반드시 확인하세요.

4. **수렴 시 STRICT 컨트롤러 전환.**
   수렴 게이트를 통과하면 `/controller_manager/switch_controller`(`controller_manager_msgs/srv/SwitchController`)를 **strictness=STRICT(2)**로 호출해 원자적으로 전환합니다.

   - `activate=[forward_position_controller]`
   - `deactivate=[scaled_joint_trajectory_controller]`

   STRICT이므로 둘 중 하나라도 전환에 실패하면 서비스가 실패로 떨어져, "반쯤 전환된" 위험 상태로 진행되지 않습니다. 이 시점의 로봇 자세와 live GELLO가 `chase_tol` 이내로 일치하므로 첫 fpc 명령에서 스냅이 없습니다.

5. **STRICT 전환 성공 직후 `gello_move_to_start`가 브리지 `~/resume` 호출 → 스트리밍 시작.**
   `resume_bridge:=true`이면 노드는 전환 성공 **직후** pre-spawn된 브리지의 `~/resume`(`std_srvs/srv/Trigger`)를 **최대 3회 재시도**하며 호출합니다. 브리지의 `~/resume`는 **정렬 게이트(alignment-gated)** — `|raw GELLO − actual| <= resume_align_tol`(기본 0.05 = `chase_tol`의 2배)일 때만 재개 — 이므로 방금 수렴한 팔은 통과합니다. 재개 시 브리지는 **로봇 실제 `/joint_states`에서 재-seed**(jump-free)하고 **soft-start(`soft_start_s=0.7s`)**로 slew clamp를 램프업하므로, 남은 잔차 gap은 한 사이클 점프가 아니라 **완만히 ease-in**됩니다. 정상 로그는 `"Bridge resumed (...); teleop is now streaming."`입니다. 이후 `/gello/joint_states` → `/forward_position_controller/commands`(250 Hz) 스트리밍이 시작됩니다.

   > **staleness watchdog 동작(`staleness_timeout_s=0.5`).** GELLO 스트림이 끊기거나 늦어져 0.5s를 초과하면, 브리지는 **새 setpoint 발행을 멈추고** `"GELLO stale, holding — not publishing"`을 로그로 남깁니다. 이때 `forward_position_controller`는 **마지막으로 명령된 자세를 그대로 유지**합니다 — 즉 **팔이 그 자리에 정지(freeze)**하며, 감속 램프도, fault도, protective stop도 없습니다. GELLO 메시지가 다시 들어오면 **자동으로 재-seed + soft-start로 스트리밍을 재개**합니다(복구 시 리더가 이동했더라도 full-slew 스냅 없음).
   >
   > **배포 전 1회 테스트(권장).** 세션 중간에 **GELLO USB를 뽑아** 팔이 드리프트/fault 없이 그 자리에 **정지(hold)**하는지 확인하고, 다시 **연결해** 스트리밍이 **재개**되는지 확인하세요.

### 운영자가 보는 것 — 정상 핸드셰이크 로그

정상 핸드오버 시 `gello_move_to_start`(그리고 브리지) 콘솔에 대략 다음 순서로 로그가 흐릅니다. 실제 문구는 코드의 로거 메시지 기준입니다.

```text
# (1) 소스 컨트롤러가 active 되기를 대기 (헤드리스가 아니면 펜던트 Play 대기)
[gello_move_to_start] scaled_joint_trajectory_controller is not active yet. Method A: on the pendant START (Play) ...
[gello_move_to_start] scaled_joint_trajectory_controller is ACTIVE; proceeding with move-to-start.

# (2) 수렴 게이트 chase — 리더가 멀거나 움직이면 catch-up 궤적을 반복해서 쏨
[gello_move_to_start] Chasing live GELLO: gap 0.612 rad at lift -> 1.22s catch-up.
[gello_move_to_start] Sending trajectory to live GELLO pose over 1.22s ...
[gello_move_to_start] Goal accepted; waiting for arm to arrive...
[gello_move_to_start] Arm arrived at live GELLO pose (trajectory SUCCEEDED).
[gello_move_to_start] Chasing live GELLO: gap 0.083 rad at w1 -> 0.50s catch-up.   # 리더가 그새 조금 움직였음
...

# (3) 리더가 멈춰 gap이 chase_tol 이내로 dwell 동안 유지되면 수렴
[gello_move_to_start] Converged: max gap 0.0121 rad held 0.42s (<= 0.025 for 0.4s). Handing over to streaming.

# (4) STRICT 전환
[gello_move_to_start] Switching controllers (STRICT): activate=forward_position_controller, deactivate=scaled_joint_trajectory_controller...
[gello_move_to_start] Controller switch OK: forward_position_controller active. Bridge may now stream.

# (5) 전환 직후 브리지 resume (정렬 게이트 통과)
[gello_move_to_start] Bridge resumed (Following RESUMED — aligned (max 0.012 rad <= 0.050). ...); teleop is now streaming.
[gello_move_to_start] Move-to-start handshake complete; shutting down.
```

- **`Chasing live GELLO: gap ... -> ...s catch-up`** 가 여러 번 보이는 것은 **정상**입니다 — 리더가 움직이는 한 계속 따라갑니다.
- **`Converged`** 로그가 나오기 전에는 절대 fpc로 전환되지 않습니다. **리더를 멈추고 `chase_tol` 이내로 들어와야** 수렴합니다.
- **`Bridge resumed ... teleop is now streaming.`** 이 나와야 실제 teleop이 시작된 것입니다. 이 줄이 없으면 아직 스트리밍이 아닙니다(아래 복구 참고).

### 복구 절차 — resume 실패 / 게이트 미수렴

세 가지 실패 양상과 대응:

**A. 수렴 게이트가 계속 안 넘어감 (chase만 반복하다 timeout).**
증상: `Chasing live GELLO ...`가 계속 반복되거나, 결국 `Did not converge within 30s (leader still moving / never settled?); aborting (fail-safe, no switch).` 후 노드가 exit 1로 종료. STRICT 전환·스트리밍 없음, 팔은 `scaled_joint_trajectory_controller`가 잡은 마지막 자세에서 정지.
- **원인 1 — 운영자가 계속 GELLO를 움직임.** 리더를 **가만히** 잡고 있으세요. dwell(`chase_dwell_s=0.4s`) 동안 정지해야 수렴합니다.
- **원인 2 — 실기 JTC 정상상태 오차 > `chase_tol`(0.025).** mock은 정확히 도달하지만 실기는 catch-up 후에도 per-joint 잔차가 남아 gap이 절대 `chase_tol` 밑으로 안 내려갈 수 있습니다(**livelock**). → 아래 **캘리브레이션 체크리스트**로 실측 오차를 재고 `chase_tol`(및 필요 시 `arrival_tolerance`)을 그 위로 올리세요.
- **원인 3 — 리더 스트림 dead/stale.** `Waiting for a fresh GELLO sample + robot /joint_states before convergence ...` 가 반복되면 GELLO USB/`gello_publisher`를 확인하세요.
- **복구:** 실패 후에는 launch를 다시 올려야 합니다(노드가 종료됨). 파라미터를 바꿔야 하면 `chase_tol`을 올린 뒤 재-launch.

**B. STRICT 전환 성공했으나 `~/resume`가 안 됨.**
증상: `Controller switch OK` 는 떴는데, `Bridge did NOT resume after 3 attempts (leader likely not aligned within resume_align_tol). forward_position_controller is ACTIVE and HOLDING; no teleop.` 로 끝남. **이 경우 handshake는 성공(exit 0)으로 간주**되어 그리퍼는 올라오지만, **팔 teleop은 아직 스트리밍 안 함.** fpc가 도착 자세를 **잡고 있어(HOLDING)** 안전합니다.
- **복구:** GELLO 리더를 로봇 현재 자세에 **정렬**(`resume_align_tol=0.05` 이내)시킨 뒤, 아래를 수동 호출:
  ```bash
  ros2 service call /gello_ur_bridge/resume std_srvs/srv/Trigger
  ```
  성공 시 `Following RESUMED — aligned ...` 응답과 함께 스트리밍이 시작됩니다. 정렬이 안 되어 있으면 `Resume REFUSED — leader/arm misaligned: max ... > resume_align_tol ...` 로 **거부**되며 어느 관절이 얼마나 어긋났는지 출력하므로, 그 관절을 맞춘 뒤 다시 호출하세요.

**C. handshake 자체가 실패로 종료(exit != 0).**
증상: launch 로그에 `Move-to-start handshake FAILED (exit N); the pre-spawned bridge stays PAUSED and silent (fail-safe) and the grippers are NOT started.` **브리지는 paused로 남아 아무것도 발행하지 않고, 팔은 정지.**
- 흔한 원인: External Control 프로그램이 펜던트에서 **Play되지 않아** `scaled_joint_trajectory_controller`가 active가 안 됨(`activation_timeout` 120s 초과), 또는 gap이 `chase_hard_limit`(4.0 rad) 초과(리더가 크게 mis-pose/wraparound — `Live gap ... exceeds chase_hard_limit ...`).
- **복구:** 원인을 해소(펜던트 Play / GELLO를 로봇 근처로 re-pose)한 뒤 **launch를 재기동**. 이미 fpc로 전환된 상태가 아니므로 브리지 수동 resume은 의미 없음.

**공통 — 즉시 중단(Abort).**
- teleop/handshake 중 위험하면 **launch 터미널 Ctrl-C**로 스트리밍을 멈추고, 조금이라도 위험하면 **물리 E-STOP**(Ctrl-C보다 우선). (수동 실행 중 `init_align` 모드라면 `ros2 service call /gello_move_to_start/abort std_srvs/srv/Trigger`로도 fail-safe 종료 가능.)

### 실기 캘리브레이션 체크리스트 — `chase_tol` vs JTC 도달 오차

> **왜 필요한가.** 수렴 게이트는 `|live GELLO − actual| <= chase_tol`(**0.025 rad**)을 요구합니다. **mock JTC는 명령 자세에 정확히 도달**하므로 gap이 0으로 떨어져 항상 수렴하지만, **실기 JTC는 정상상태 오차(steady-state error)가 남을 수 있어** catch-up 후에도 gap이 `chase_tol` 밑으로 안 내려가면 게이트가 **livelock**(무한 chase → timeout → exit 1)합니다. 또 `arrival_tolerance`(FollowJointTrajectory goal tolerance, **0.05**)보다 실기 도달 오차가 크면 각 catch-up 궤적이 **SUCCEEDED되지 못해** `_send_trajectory`가 실패 → 수렴 자체가 중단됩니다. 그래서 실기 첫 구동 전 **한 번의 핸드셰이크에 걸쳐 rosbag을 떠서 실측 도달 오차를 재고, `chase_tol`(과 필요 시 `arrival_tolerance`)이 그 위인지 확인**해야 합니다.

1. **핸드셰이크 1회에 걸쳐 rosbag 기록.** launch를 올리고(또는 수동으로 `gello_move_to_start` 실행), 핸드셰이크가 진행되는 동안 다음을 녹화합니다:
   ```bash
   ros2 bag record -o handshake_cal \
     /joint_states \
     /forward_position_controller/commands \
     /gello/joint_states
   ```
   리더를 멈춰 수렴("Converged")까지 간 뒤 정지합니다. (수렴이 안 되면 그 자체가 `chase_tol`이 너무 빡빡하다는 신호 — 아래 3 참고.)
2. **catch-up 궤적 도달 후의 정상상태 오차 측정.** 마지막 catch-up이 SUCCEEDED된 직후 구간에서, `/joint_states`(실측)와 그 catch-up의 목표(로그의 live GELLO 자세, 또는 `/gello/joint_states`가 그 순간 정지해 있었다면 그 값)의 **per-joint 차이의 정상상태 값**을 봅니다. 이 잔차가 실기 JTC 도달 오차 `e`입니다.
   ```bash
   # 빠른 확인용: 팔이 궤적 도달 후 정지한 구간에서 두 토픽을 비교
   ros2 topic echo /joint_states --once
   ros2 topic echo /gello/joint_states --once   # 리더가 정지해 있을 때
   ```
3. **판정 및 튜닝.**
   - `chase_tol`(0.025)이 실측 `e`보다 **커야** 합니다. 여유 있게 `e`의 약 1.5~2배로 잡으세요. `e ≈ 0.02`면 `chase_tol: 0.03~0.04`.
   - `arrival_tolerance`(0.05)도 `e`보다 커야 catch-up 궤적이 SUCCEEDED됩니다. 실기에서 궤적이 자꾸 abort되면 `0.05` → `0.08` 등으로 **완화**하세요.
   - 두 값은 `ur7e_gello.yaml`의 `gello_move_to_start:` 블록에서 조정 후 `colcon build`(또는 install된 config 갱신)합니다.
   - **주의:** `chase_tol`을 너무 키우면 그만큼 큰 잔차 gap이 handover 시점에 남고, 그 gap은 브리지 soft-start slew로 닫힙니다(`<= max_step_rad*rate`). `resume_align_tol`(0.05)이 `chase_tol`의 상한 역할을 하므로, `chase_tol`을 `resume_align_tol` 근처까지 키우면 resume 정렬 게이트가 빡빡해집니다 — 필요 시 `resume_align_tol`도 함께 올리세요(단 클수록 resume 시 허용 잔차가 커짐).

### 새 파라미터 레퍼런스 (`gello_move_to_start:` / `gello_ur_bridge:`)

배포 기본값은 `config/ur7e_gello.yaml` 기준이며, config는 노드 코드 기본값을 **override**합니다. (노드 기본값이 다르면 괄호로 표기.)

**`gello_move_to_start:` — 수렴 게이트 (`gello` 모드)**

| 파라미터 | 배포 기본값 | 의미 / 튜닝 시점 |
|---|---|---|
| `chase_tol` | `0.025` (≈1.4°) | 핸드오버에 요구되는 per-joint 일치(rad). **실기 JTC 정상상태 오차보다 커야** 함(안 그러면 livelock). 위 캘리브레이션으로 조정. |
| `chase_dwell_s` | `0.4` | 위 일치가 **유지**돼야 하는 시간(s, median-of-5 + dwell 중 새 샘플 요구). 낮추면 더 빨리 수렴하나 스파이크 통과 위험↑. |
| `chase_v_budget` | `0.5` (노드 기본 `0.3`) | catch-up 궤적 길이를 정하는 속도 예산(rad/s): `T = max(gap/v, min_traj_duration)`. 낮추면 더 느리고 안전, 높이면 빠름. |
| `min_traj_duration` | `0.5` (노드 기본 `0.75`) | 각 catch-up 궤적 최소 길이(s). 작은 gap도 near-step이 아니라 부드럽게. |
| `chase_hard_limit` | `4.0` | 이보다 큰 gap은 **자동 chase 거부**(fail-safe). **wraparound/gross-mispose 백스톱**이지 approach 제한 아님(팔은 최대 ~π에서 정상 출발). 실기에서 **줄이지 말 것**(정상 approach를 막음). |
| `chase_timeout_s` | `30.0` | 전체 수렴 시간 예산(s). 초과 시 exit 1 fail-safe. `<=0`이면 무한 대기. |
| `gello_staleness_s` | `0.5` | GELLO 샘플이 이보다 오래되면 **stale** — dead 스트림이 dwell을 통과해 복구 시 스냅을 여는 것을 방지. |
| `arrival_tolerance` | `0.05` | catch-up 궤적의 FollowJointTrajectory **goal tolerance**(rad). 실기 도달 오차가 이보다 크면 궤적이 abort → 수렴 중단. 실기에서 필요 시 완화. |
| `trajectory_duration` | `5.0` | `init_align` 모드의 init-pose 이동 길이(s). `gello` 모드 chase는 duration-sized라 미사용. |
| `resume_bridge` | *(yaml 없음)* 노드 기본 `False` | STRICT 전환 성공 후 브리지 `~/resume` 호출 여부. **런치가 `True`로 override**. 단독 실행은 `False`(브리지 안 건드림). |
| `bridge_resume_service` | *(yaml 없음)* 노드 기본 `/gello_ur_bridge/resume` | resume를 호출할 서비스 이름. |
| `activation_timeout` | *(yaml 없음)* 노드 기본 `120.0` | 소스 컨트롤러가 active(=펜던트 Play) 되기를 기다리는 최대 시간(s). |

> **`init_align` 모드 파라미터**(`start_mode: init_align`일 때만): `init_pose`(배포 `[1.5708, -1.57, 1.57, -1.57, -1.57, 0.0]`, 노드 기본 pan=0.0), `alignment_tolerance`(배포 **0.2**, 노드 기본 0.15), `alignment_hard_limit`(0.5), `alignment_timeout`(0.0=무한). 기본 `gello` 모드에서는 수렴 게이트가 대신 동작하므로 이들은 미사용.

**`gello_ur_bridge:` — pre-spawn + resume + soft-start**

| 파라미터 | 배포 기본값 | 의미 / 튜닝 시점 |
|---|---|---|
| `start_paused` | `false` (**런치가 `True`로 override**) | yaml은 `false` 유지 필수(다른 launch가 이 파일을 로드해도 안 멈추도록). 통합 실기 launch만 `True`로 pre-spawn PAUSED. |
| `resume_align_tol` | *(yaml 없음)* 노드 기본 `0.05` | `~/resume` 정렬 게이트(rad): `|raw GELLO − actual| <= 이 값`일 때만 재개. `chase_tol`의 2배라 방금 수렴한 팔은 통과, 어긋난 수동 resume은 거부. `chase_tol`을 크게 키우면 이것도 함께 올림. |
| `soft_start_s` | `0.7` | 모든 (re)seed(startup/resume/staleness 복구) 후 slew clamp를 ~15%→100%로 램프하는 시간(s). 잔차 gap을 한 사이클이 아니라 완만히 닫음. `0.0`=off. |
| `max_step_rad` | `0.0025` | per-cycle slew 상한(rad). 250 Hz에서 지속 0.625 rad/s, 최악 coalescing 2.5 rad/s(< 3.14 한계). **250 Hz에서 0.003 초과 금지**; 더 빠르게 하려면 먼저 `publish_rate_hz`를 500으로. |
| `staleness_timeout_s` | `0.5` | 이보다 GELLO가 오래 안 오면 발행 중단(hold) + 복구 시 재-seed. |
| `filter_type` | `one_euro` (노드 기본 `ema`) | 스무딩 필터. `one_euro`는 속도적응 저역통과(정지 시 강한 스무딩, 빠를 때 저지연). `deadband_rad`·`ema_alpha`는 `ema` 모드에서만 사용. |
| `publish_rate_hz` | `250.0` (노드 기본 `125.0`) | 30 Hz GELLO를 500 Hz 드라이버로 2:1 업샘플. |

> **요약(현 버전 순서)**: `scaled_joint_trajectory_controller`(부팅·active) + `gello_ur_bridge` **pre-spawn PAUSED** → list_controllers 확인 → `gello_move_to_start` **수렴 게이트 chase**(live 리더 따라감) → **"Converged"** → STRICT 전환(fpc active / stjc deactivate) → `gello_move_to_start`가 브리지 **`~/resume` 호출**(정렬 게이트 통과) → **"Bridge resumed ... streaming"**. 이 순서를 어기고 fpc를 먼저 켜면 protective stop이 발생합니다.
>
> **정직한 보장(재확인):** 핸드오버는 팔로워가 **살아있는 리더를 `chase_tol` 이내로 따라잡아 유지**했을 때만 일어나며, 남는 잔차는 soft-start된 rate-limited slew로 닫힙니다. 계속 움직이는 리더는 스냅이 아니라 **핸드오버 지연**을 유발합니다("zero snap"이 아님).
>
> 안전 불변식(재확인): 이 handshake는 전부 **UR7e 팔로워** 쪽 절차입니다. GELLO는 이 과정 내내 토크 없이 **읽히기만** 합니다.

---

## 3. External Control: PolyScope 5 vs PolyScope X

`scaled_joint_trajectory_controller` → `forward_position_controller` 전환 및 GELLO 스트리밍이 정상 동작하려면, UR 폴리스코프(Polyscope) 펜던트에 **External Control** 프로그램 노드가 반드시 올라가 있어야 한다. 이 노드의 설치 방식은 펜던트 소프트웨어 버전에 따라 완전히 다르므로, 실기 연결 전 **반드시 펜던트에서 버전을 먼저 확인**해야 한다.

> **이 랩 실기 관찰(2026-07-03).** `ros2 launch ur_robot_driver ur_control.launch.py robot_ip:=192.168.10.11` 실행 시 Dashboard 서버 접속 + RTDE 500Hz + `robot mode RUNNING / safety NORMAL`로 **클래식 e-Series 흐름이 정상 동작**했다. 이는 대상 로봇이 **PolyScope 5(§3.1 classic `externalcontrol` URCap 경로)** 일 가능성이 높음을 시사한다. 다만 드라이버 연결만으로 5 vs X를 확정할 수는 없으므로, 펜던트 `Settings > About`으로 **반드시 최종 확인**할 것.

### 3.0 사전 확인 (필수)

펜던트에서 `Settings > About` 메뉴로 들어가 소프트웨어 버전을 확인한다.

- **PolyScope 5** (예: 5.x, e-Series 구버전 UI) 인지
- **PolyScope X** (버전 **10.8.0 이상**, e-Series 신규 UI) 인지

를 먼저 확정하라. 아래 두 경로는 서로 호환되지 않으며, 잘못된 URCap을 설치하면 프로그램 노드 자체가 보이지 않거나 로드에 실패한다.

### 3.1 PolyScope 5 (기존 경로) — classic `externalcontrol` URCap

- Universal_Robots_ExternalControl_URCap (예: `externalcontrol-1.0.5.urcap` 등, classic `.urcap` 확장자)을 USB로 펜던트에 사이드로드한다.
- `Settings > System > URCaps`에서 `+` 버튼으로 설치 후 펜던트 재시작.
- 프로그램 탭에서 `URCaps > External Control` 노드를 추가하고, 노드 파라미터에 ROS2 PC의 IP(`robot_ip`가 아니라 **ROS2 컨트롤 PC의 IP**, 즉 `ur_control.launch.py`를 실행하는 머신)와 **Reverse Port** (드라이버 기본값 **50001**, `ur_control.launch.py`의 `reverse_port` 인자와 반드시 일치해야 함 — 참고로 50002는 `script_sender_port`이므로 혼동 금지)를 입력한다.
- 프로그램을 저장하고 로드한 뒤 하단 **Play(▶)** 버튼으로 실행하면 ROS2 드라이버가 로봇에 연결된다.

### 3.2 PolyScope X (10.8.0 이상) — 별도의 `ExternalControl` URCapX 필요

PolyScope X는 URCap 프레임워크 자체가 classic Polyscope 5와 다르다 (URCapX). **기존 `externalcontrol` URCap은 PolyScope X에서 동작하지 않는다.** 반드시 UR가 별도로 배포하는 **ExternalControl URCapX** 패키지(확장자 `.urcapx`)를 사용해야 한다.

설치 절차:

1. UR 공식 배포처(또는 로봇 벤더 제공 매체)에서 `externalcontrol-x-<version>.urcapx` 파일을 확보한다.
2. USB에 담아 펜던트에 꽂고, PolyScope X의 `Settings > URCaps` (또는 `System > URCapX Manager`, 버전별 메뉴명 상이) 화면에서 사이드로드/설치한다.
3. 설치 후 펜던트 재시작이 필요할 수 있다.
4. 프로그램 편집기에서 `External Control` 노드를 추가하고, ROS2 PC IP와 Reverse Port를 classic 경로와 동일하게 입력한다.

**가장 중요한 함정 — "Update program" 단계 누락:**

PolyScope X에서 External Control 노드의 파라미터(IP, 포트 등)를 수정하거나 프로그램을 새로 불러온 뒤에는, 실행(Play) 전에 반드시 **"Update program"**(프로그램 갱신) 단계를 눌러 변경 사항을 실제 실행 바이너리에 반영해야 한다. 이 "Update program" 버튼을 누르지 않고 바로 Play를 누르면:

- ROS2 드라이버(`ur_robot_driver`)가 리버스 포트에서 연결을 계속 기다리며 타임아웃되거나,
- 펜던트 쪽은 "실행 중"으로 보이지만 실제로는 이전 파라미터(또는 빈 설정)로 떠 있어 소켓이 열리지 않는 등,

**"연결이 걸린 것처럼 보이는(stuck-connection)" 증상의 1순위 원인**이 된다. 즉 순서는 항상:

```
프로그램 파라미터 수정/로드 → [Update program] 클릭 → [Play] 클릭
```

이 순서를 지키지 않으면 GELLO → UR 브리지(`gello_ur_bridge_node`)가 정상 기동되어도 로봇이 반응하지 않는다. 실기 연결 트러블슈팅 시 가장 먼저 재확인할 항목이다.

### 3.3 `robot_ip` / Reverse Port 기본 개념 (두 경로 공통)

- `robot_ip`: `ur7e_gello_real.launch.py` 실행 시 **필수** 인자로, UR 컨트롤러(로봇 본체)의 IP 주소. 기본값 없음 — 반드시 명시적으로 전달해야 한다.
  ```
  ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<로봇_IP> ...
  ```
- Reverse Port: UR 컨트롤러가 펜던트의 External Control 프로그램을 통해 ROS2 PC로 다시 접속(reverse connection)할 때 사용하는 TCP 포트(드라이버 기본 **50001**; 50002는 `script_sender_port`로 별개). 펜던트 노드에 입력한 값과 `ur_control.launch.py`가 여는 포트가 반드시 일치해야 하며, 방화벽에서 열려 있어야 한다.
- ROS2 PC IP: 펜던트 External Control 노드에 입력하는 IP는 `robot_ip`가 아니라 **ROS2 드라이버를 실행 중인 PC의 IP**임에 유의한다 (역할이 반대라 혼동하기 쉽다).

### 3.4 요약 체크리스트

- [ ] 펜던트 `Settings > About`에서 PolyScope 버전 확인 (5 vs X, X면 10.8.0 이상인지)
- [ ] PolyScope 5 → classic `externalcontrol-*.urcap` 사이드로드
- [ ] PolyScope X (>=10.8.0) → 별도의 `externalcontrol-x-*.urcapx` (URCapX) 사이드로드, classic URCap 재사용 금지
- [ ] External Control 노드에 ROS2 PC IP + Reverse Port(드라이버 기본 **50001**, launch 인자와 일치; 50002 아님) 입력
- [ ] (PolyScope X) 파라미터 변경/로드 후 **Update program** 클릭 → 그 다음 **Play**
- [ ] `robot_ip` 없이는 `ur7e_gello_real.launch.py`가 기동되지 않음 — 반드시 명시

> 안전 불변식: 이 절의 어떤 단계도 GELLO 리더 암 쪽에는 적용되지 않는다. GELLO는 항상 수동(passive) 읽기 전용 장치이며, GELLO Dynamixel에는 어떤 경우에도 토크를 인가하지 않는다. 여기서 다루는 External Control / URCapX 설정은 오직 **UR 팔로워 로봇(ur7e)** 쪽 연결 절차이다.

---

## Remote 모드 — 펜던트 Play 없이 시작

> **목적:** 매번 티칭 펜던트에서 프로그램을 로드하고 손으로 **Play(▶)** 를 눌러야 하던 절차를 없애고, **ROS2 launch 한 줄**로 GELLO teleop 을 완전 hands-free 로 시작한다.
> **루트 참조 문서:** 이 랩 PC의 사용자 자체 레퍼런스 [`../../UR7e_Remote_Control_ROS2.md`](../../UR7e_Remote_Control_ROS2.md) (Method A = dashboard play, Method B = headless ⭐). 이 절은 그 문서를 본 실기 런북 파이프라인(§2 move-to-start handshake)과 어떻게 맞물리는지에 초점을 맞춘 요약이다.

우리 실기 launch(`ur7e_gello_real.launch.py`)는 이미 `headless_mode` 인자를 노출한다(**기본값 `false`**). 여기에는 두 가지 시작 방식이 있으며, **한 실행 안에서 A와 B를 절대 섞지 않는다.**

### 방식 B — Headless (⭐ 권장): 프로그램·Play 자체가 불필요

`headless_mode:=true` 를 주면 `ur_robot_driver` 가 URScript 를 로봇에 **직접 전송**하므로, 펜던트의 External Control URCap 프로그램도, Play 버튼도 필요 없다. launch 하는 순간 리버스 인터페이스가 **자동 접속**된다.

- **전제조건(펜던트, 최초 1회 세팅):** 로봇이 반드시 **Remote 모드**여야 하고 **Real Robot**(Simulation 아님) 상태여야 한다.
  - 우측 상단 **햄버거(≡) → Settings → System → Remote Control → Enable**, 그런 다음 헤더의 Local/Remote 토글을 **Remote** 로 전환.
  - 우측 하단이 **`Simulation`** 이면 실제 팔이 안 움직이므로 **`Real Robot`** 으로 토글.
  - Remote 모드에서는 펜던트 `Load Program` / ▶Play / ■Stop 버튼이 **회색(비활성)** 인 것이 정상이다.
- **실행 (권장 경로 — 런처 사용):** `run_ur7e_gello_real.sh` 에서 `HEADLESS=true` 로 켠다.

  ```bash
  HEADLESS=true ./run_ur7e_gello_real.sh          # robot_ip 기본 192.168.10.11
  # 또는 직접:
  ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
    robot_ip:=192.168.10.11 headless_mode:=true
  ```

- **§2 handshake 와의 결합 (완전 hands-free):** headless + Remote 에서는 리버스 인터페이스가 startup 에 자동 접속되므로 `scaled_joint_trajectory_controller` 가 **자동으로 active** 가 된다. 우리 `gello_move_to_start` 노드는 이 컨트롤러가 active 가 되기를 **기다렸다가**(`_wait_for_source_active`) 진행하도록 되어 있으므로, 사람이 Play 를 누르지 않아도 handshake 가 그대로 이어진다:
  **launch → (headless URScript 자동 전송) → 리버스 인터페이스 자동 접속 → `scaled_joint_trajectory_controller` active → `gello_move_to_start` 가 live GELLO를 수렴 게이트로 chase → "Converged" → STRICT 전환(fpc active) → 노드가 브리지 `~/resume` 호출 → `gello_ur_bridge` 스트리밍.** 전 과정에 펜던트 조작이 없다. (단, 리더가 계속 움직이면 수렴할 때까지 핸드오버가 지연됨 — §2 복구 A 참고.)

### 방식 A — External Control URCap + Dashboard 원격 Play (fallback)

headless 를 쓸 수 없는 경우(예: headless 를 막는 사이트 정책, URCap 프로그램을 유지해야 하는 워크플로우)의 대안. 펜던트에 External Control URCap 프로그램(예: `ur_caps.urp`)을 두고, **Play 만 ROS2 가 dashboard 서비스로 대신** 눌러준다. 이때 launch 는 `headless_mode:=false`(기본값) 로 둔다.

```bash
# 터미널 1 — 우리 실기 launch (headless_mode 기본 false)
./run_ur7e_gello_real.sh          # 또는 ros2 launch ... ur7e_gello_real.launch.py robot_ip:=192.168.10.11

# 터미널 2 — 프로그램 로드 후 원격 Play (펜던트 Play 버튼 대신)
ros2 service call /dashboard_client/load_program \
  ur_dashboard_msgs/srv/Load "{filename: ur_caps.urp}"
ros2 service call /dashboard_client/play std_srvs/srv/Trigger {}
```

Play 가 성공하면 리버스 인터페이스가 붙어 `scaled_joint_trajectory_controller` 가 active 가 되고, 위 §2 handshake 가 동일하게 이어진다. `filename` 은 펜던트에 실제 저장된 프로그램 이름에 맞춘다.

### 리버스 인터페이스 끊김 복구

실행 중 리버스 인터페이스가 끊기면(`Connection to reverse interface dropped.`) 시작 방식에 맞는 복구 명령을 쓴다:

| 방식 | 복구 명령 |
|---|---|
| **B (headless)** | `ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger {}` |
| **A (dashboard)** | `ros2 service call /dashboard_client/play std_srvs/srv/Trigger {}` |

보조 명령: 필요 시 `remote_helpers.sh` 에 위 dashboard/status 서비스 호출 래퍼가 모여 있다(load/play/stop/power_on/brake_release/resend). 서비스 **타입**을 반드시 지킬 것 — `load_program` = `ur_dashboard_msgs/srv/Load`(`filename` 필드), `play`/`stop`/`power_on`/`brake_release`/`resend_robot_program` = `std_srvs/srv/Trigger`(빈 `{}`).

### Abort / 비상정지

teleop 중 위험을 감지하면 아래 순서로 즉시 중단한다.

1. **launch 터미널에서 Ctrl-C** — 스트리밍(브리지)을 멈춰 새 setpoint 발행을 중단한다.
2. **조금이라도 의심스럽거나 위험하면 즉시 물리 E-STOP(비상정지)** — 물리 E-STOP은 Ctrl-C보다 **우선**한다. 망설이지 말 것.
3. **래치된 protective stop 해제** — 안전이 확보된 뒤 `ur_unlock`(=`/dashboard_client/unlock_protective_stop`, `std_srvs/srv/Trigger`)으로 보호 정지를 해제한다.
4. **재개** — 방식 A면 **Play**(dashboard play), 방식 B면 `ur_resend`(=`/io_and_status_controller/resend_robot_program`)로 리버스 프로그램을 다시 보낸다. (`ur_unlock` / `ur_resend` / `ur_play` 는 `remote_helpers.sh` 에 래핑되어 있음.)

> 안전 불변식: abort 전 과정에서도 GELLO는 토크 없이 **읽히기만** 한다(§1). E-STOP·보호 정지 처리는 오직 **UR7e 팔로워** 쪽 절차다.

### 주의 (비협상)

- ⚠️ **방식 A 와 B 를 한 실행에서 섞지 말 것.** headless 로 URScript 를 직접 쏘는 동시에 External Control 프로그램을 Play 하면 두 제어 스트림이 충돌한다. 둘 중 하나만.
- ⚠️ **Remote → Local 전환은 오직 펜던트에서만.** 원격으로 강제 전환할 수 없는 것이 안전 설계다. 손으로 팔을 옮기고 싶으면 펜던트에서 잠깐 Local 로 토글한다(그동안 teleop 은 멈춘다).
- ⚠️ 이 절은 전부 **UR7e 팔로워** 쪽 시작 절차다. GELLO 는 이 과정 내내 토크 없이 **읽히기만** 한다(§1 안전 불변식).
- 물리적인 "no-Play" 실기 테스트(펜던트에서 Remote 로 토글 후 팔이 실제로 움직이는지)는 **사용자(운영자)** 의 책임이다. 빌드-검증 PC 에서는 launch `--show-args`, dashboard/controller 서비스 이름·타입 존재 확인, `colcon build`, 그리고 teleop 노드 없이 **드라이버만** headless 로 띄워 "Robot ready to receive control commands." 도달까지만(=팔 무동작) ROS2-side 배선을 검증한다.

---

## 4. UR7e 기구학(kinematics) 캘리브레이션 — 최초 1회

> **주의 — 이건 GELLO 캘리브레이션이 아닙니다.** `config/ur7e_gello.yaml`의 `joint_offsets` / `joint_signs` / `gripper_config`는 **물리 GELLO 리더 암(FTBEO6QK)** 고유의 캘리브레이션이며, 이번 절과 무관하게 **그대로(verbatim) 재사용**합니다. 여기서 다루는 것은 그와 완전히 별개인 **UR7e 로봇 팔 자체의 기구학 캘리브레이션**(DH 파라미터 보정)입니다. UR 로봇은 팔마다 제조 편차가 있어, 실기체 정밀도를 얻으려면 `ur_calibration`으로 그 로봇 고유의 보정값을 뽑아내야 합니다.

> **joint-space 텔레오퍼에서의 영향 범위.** GELLO teleop은 **관절공간(joint-space)** 명령(관절 각도 → `forward_position_controller`)이므로, 기구학 캘리브레이션 불일치는 주로 **TCP/Cartesian 좌표 정확도**에만 영향을 주고 **관절 추종 자체는 정상 동작**합니다. 실제로 이 랩에서 `ur_type:=ur5e`로 잘못 실행해도 관절 추종은 문제없이 동작했고, 드라이버는 `calibration checksum mismatch`를 **경고**로만 남기고 정상 기동합니다(`System successfully started!`). 그럼에도 실기 정밀도·안전을 위해 아래 추출을 **권장(최초 1회)** 합니다.

두 캘리브레이션은 독립적입니다.

| 구분 | 대상 | 값 | 재사용 여부 |
| --- | --- | --- | --- |
| GELLO 캘리브레이션 | 리더 암(Dynamixel, FTBEO6QK) | `joint_offsets`, `joint_signs`, `gripper_config` (`config/ur7e_gello.yaml`) | 이 로봇(ur7e)에서도 **그대로 재사용** — 리더 암은 바뀌지 않았으므로 |
| UR 기구학 캘리브레이션 | 팔로워 로봇(UR7e 실기체) | `ur7e_calibration.yaml` (DH 보정) | **로봇마다 새로 추출** — 이번 절의 내용 |

### 4.1 캘리브레이션 추출 (최초 1회, 로봇당 1회)

UR7e 컨트롤러가 켜져 있고 네트워크로 접근 가능한 상태에서 실행합니다. (`ur_calibration`은 `ros-humble-ur` 메타패키지에 포함되어 있으므로 추가 설치가 필요 없습니다.)

```bash
source /opt/ros/humble/setup.bash

ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.10.11 \
  target_filename:="/home/laptop3/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_calibration.yaml"
```

- `robot_ip`는 실제 UR7e 컨트롤러(펜던트)의 IP로 교체합니다. **이 랩 실기 예시: `192.168.10.11`** (2026-07-03 연결 확인됨).
- `target_filename`은 저장소 내부 경로(`ros2_ur_ws/src/ur_gello_bringup/config/ur7e_calibration.yaml`)로 지정해, 다른 설정 파일들과 함께 버전 관리되도록 합니다.
- 이 명령은 로봇에 짧게 통신해 실제 DH 파라미터를 읽어온 뒤 위 경로에 YAML로 저장하고 종료합니다. UR7e 실기체가 교체/재조정되지 않는 한 **한 번만** 실행하면 됩니다.

### 4.2 실주행 launch에 반영

추출된 `ur7e_calibration.yaml`은 `ur7e_gello_real.launch.py`의 `kinematics_params_file` 인자로 전달합니다.

```bash
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py \
  robot_ip:=<UR7e_IP> \
  kinematics_params_file:="/home/laptop3/gello_software/ros2_ur_ws/src/ur_gello_bringup/config/ur7e_calibration.yaml"
```

`kinematics_params_file`을 생략하면 UR 드라이버는 공장 기본 DH 파라미터를 사용합니다 — 텔레옵 자체는 동작하지만, TCP 절대 위치 정밀도가 그 UR7e 실기체의 실제 치수와 어긋날 수 있습니다. 실기체 정밀 작업(그리퍼 접근, 파지 등) 전에는 반드시 4.1을 먼저 수행해 두는 것을 권장합니다.

> 참고: 이 절은 **로봇(UR7e)이 연결된 환경에서만** 수행 가능하며, `source:=fake` 검증 경로에서는 필요하지도, 수행할 수도 없습니다.

---

## 5. Robotiq 2F-85 그리퍼 — URCap 소켓 (63352, raw TCP)

`robotiq_urcap_node`는 ROS 토픽/서비스가 아니라 **UR 컨트롤러(PolyScope)가 띄우는 URCap TCP 소켓(기본 포트 63352)에 직접 연결**해 그리퍼를 구동합니다. 즉 이 구간은 `ros2_control`/컨트롤러 매니저와 무관하게 동작하는 **raw socket 경로**이며, §2의 `forward_position_controller` / `scaled_joint_trajectory_controller` 전환과는 완전히 독립적입니다.

구독 토픽은 기존 그대로입니다.

| 토픽 | 타입 | 비고 |
| --- | --- | --- |
| `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` | 0..1 (0=open, 1=closed) |

### 5.1 PolyScope 5 (기존 검증된 UR 컨트롤러) — 그대로 동작

로봇 PC(UR 컨트롤러와 같은 네트워크)에서 다음 파라미터만 맞추면 별도 코드 수정 없이 그대로 씁니다.

```bash
ros2 run ur_gello_bringup robotiq_urcap --ros-args \
  -p robot_ip:=<UR7e_IP> \
  -p connect_on_start:=true
```

- `connect_on_start:=false`(기본값, 이 PC/빌드-검증 PC 기준)에서는 소켓을 열지 않고 "이 위치로 움직였을 것"만 로그로 남깁니다 — 그리퍼가 물리적으로 없는 이 검증 환경에서는 이 상태가 정상입니다.
- 실 로봇 PC에서 `connect_on_start:=true` + 올바른 `robot_ip`를 주면 URCap 소켓(63352)에 접속해 실제로 그리퍼를 구동합니다.

### 5.2 PolyScope X (신형 컨트롤러) — 63352 소켓 생존 여부를 먼저 검증할 것

PolyScope X는 URCap 런타임/네트워크 스택이 PolyScope 5와 다르므로, **63352 포트가 동일하게 열려 있다고 가정하지 말고** 실 로봇에 붙이기 전에 반드시 확인합니다.

```bash
nc <UR7e_IP> 63352
```

- **응답이 오면** (연결이 열리고 그리퍼 상태 질의에 반응): PolyScope 5와 동일하게 `robotiq_urcap` 노드를 `robot_ip` + `connect_on_start:=true`로 그대로 사용할 수 있습니다.
- **응답이 없으면/연결이 거부되면**: PolyScope X에서는 이 raw-socket URCap 경로가 죽어 있는 것이므로, `robotiq_urcap_node`를 신뢰하지 말고 **ToolComm/URCapX 기반 그리퍼 경로**(PolyScope X의 새 Tool Communication 인터페이스를 통한 그리퍼 제어)를 후속 조사 항목으로 남깁니다. 이 경우 이 노드를 수정하기 전에 먼저 별도 스파이크로 PolyScope X 쪽 그리퍼 통신 방식을 확인해야 합니다.

### 5.3 안전 메모

이 소켓 경로는 **그리퍼 전용**이며 GELLO passive read-only 불변식(§1 참고)과는 무관합니다 — GELLO 쪽에는 항상 어떤 경우에도 토크가 걸리지 않습니다.

---

## 6. 열린 질문 체크리스트 (실기 연결 전 확인)

실 UR7e에 붙이기 직전, 아래 항목을 순서대로 확정하세요. 하나라도 불명확하면 그 항목이 실기 첫 구동 실패의 1순위 원인입니다.

- [ ] **PolyScope 버전 확정 (5 vs X).** 펜던트 `Settings > About` 확인. PolyScope 5면 classic `externalcontrol-*.urcap`, PolyScope X(>=10.8.0)면 `externalcontrol-x-*.urcapx` — 경로가 완전히 갈립니다(§3). PolyScope X면 그리퍼 63352 소켓 생존 여부(§5.2 `nc`)도 함께 확인.
- [ ] **`robot_ip` 확정.** `ur7e_gello_real.launch.py`의 **필수** 인자, 기본값 없음. UR7e 컨트롤러 본체 IP. External Control 노드에 넣는 IP는 이것이 아니라 **ROS2 PC IP**임에 유의(§3.3).
- [ ] **`dynamixel_sdk` 설치 확인.** 실 GELLO(`source:=gello`, `gello_publisher`) 경로는 `dynamixel_sdk`가 필요합니다. 실기 GELLO 연결 전 `pip install --user dynamixel-sdk` (v4.0.5, sudo/apt 불필요)로 설치하고 `python3 -c "import dynamixel_sdk"`로 확인합니다. (mock `source:=fake` 경로는 불필요.)
- [ ] **`GELLO_REPO_ROOT` export.** `gello_publisher`에는 잘못된 하드코딩 fallback 경로(`/home/theo_lab/gello_software`)가 있으므로, `source:=gello` 실행 전 반드시 `export GELLO_REPO_ROOT=/home/laptop3/gello_software`를 설정합니다. (노드는 FROZEN — 수정 금지, docs로 우회.)
- [ ] **`max_step_rad = 0.0025` (250 Hz) 유지.** 드라이버는 각 명령을 `delta/0.002s`로 속도 한계(3.14 rad/s)와 비교하므로, 안전 예산은 `0.00628 rad`(250 Hz에서 coalescing 대비 절반 `0.00314`)입니다. `0.0025`면 단발 1.25 / 최악 2.5 rad/s로 안전합니다. **더 빠른 텔레오퍼가 필요하면 먼저 `publish_rate_hz`를 500으로 올린 뒤** `max_step_rad`를 키우세요(250 Hz에서 `0.003` 초과 금지). 시작 스냅은 브리지가 실제 `/joint_states`에서 seed하므로 이미 제거됨.
- [ ] **(권장) UR7e 기구학 캘리브레이션 추출.** 정밀 작업 전 `ur7e_calibration.yaml`을 뽑아 `kinematics_params_file`로 전달(§4).
- [ ] **부팅 컨트롤러 상태 확인.** 기동 직후 `ros2 control list_controllers`로 `scaled_joint_trajectory_controller` active + `forward_position_controller` inactive를 확인한 뒤에만 §2 handshake를 진행.

> **최종 안전 불변식 (재확인, 비협상):** GELLO는 **항상 passive read-only** 입력 장치이며, GELLO Dynamixel에는 **어떤 경우에도 토크를 인가하지 않습니다.** 드라이버는 토크 OFF로 초기화하고 관절만 읽습니다. 이 런북의 모든 전환·소켓·캘리브레이션 절차는 오직 UR7e 팔로워 로봇 쪽에만 적용됩니다.

---

### 관련 문서

- mock/검증 경로(여기서 실행 가능): [`GELLO_UR7E_ROS2_BRINGUP.md`](./GELLO_UR7E_ROS2_BRINGUP.md)
- 설계·아키텍처·리스크: [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md)
- mock UR 브링업 상세: [`GELLO_UR_ROS2_BRINGUP.md`](./GELLO_UR_ROS2_BRINGUP.md)
- ros2_control 레퍼런스: [`GELLO_ROS2_CONTROL_REFERENCE.md`](./GELLO_ROS2_CONTROL_REFERENCE.md)

---

### 팔 + 그리퍼 동시 구동 (2F-85 포함)

`ur7e_gello_real.launch.py`는 이제 UR7e 팔 텔레오퍼와 **Robotiq 2F-85 그리퍼**를 함께 올립니다. 그리퍼는 GELLO 리더의 그리퍼 축(width 토픽)에서 구동됩니다 — GELLO 손을 **닫으면 로봇 그리퍼도 닫힙니다**(방향 반전 없음, crush 방지).

**공존 모델 (핵심):** 그리퍼용 tool voltage와 RS485 버스는 **드라이버**가 제공합니다(펜던트 Installation 탭 아님).

- 런치의 `ur_control.launch.py` include에 `use_tool_communication:=true`, `tool_voltage:=24`, `tool_device_name:=/tmp/ttyUR`를 전달합니다. 드라이버가 (1) 툴에 24V를 인가하고 (2) `robot_ip:54321`을 소유하는 tool_communication(socat) 포워더를 띄워 시리얼 장치 `/tmp/ttyUR`로 노출합니다.
- `robotiq_gripper_modbus` 노드는 **직접 TCP가 아니라** 이 브리지를 공유합니다: per-node 오버라이드 `serial_port:=/tmp/ttyUR`. 따라서 `:54321`의 클라이언트는 **드라이버 socat 포워더 단 하나**입니다.
- 그리퍼(`robotiq_gripper_modbus`)와 `gello_gripper_bridge`는 handshake 성공 핸들러(`OnProcessExit(move_to_start)`, returncode==0)에서 시작합니다 — handshake 완료 후에만 올라옵니다. (팔 브리지 `gello_ur_bridge`는 이 핸들러에서 시작하지 **않습니다**: 이미 t=6s에 pre-spawn PAUSED로 존재하고, `gello_move_to_start`가 STRICT 전환 직후 `~/resume`로 직접 releases합니다 — §2 참고.)

**왜 펜던트가 아니라 드라이버인가:** 펜던트 Installation 탭에서 tool voltage를 인가하면 External Control이 시작될 때 그것을 **끊어버리고**, EC 실행 중에 다시 인가하면 EC가 **멈춥니다**. 드라이버 인자(`tool_voltage:=24`)로 공급해야 EC PLAY 중에도 유지됩니다.

**필수 조건 / 체크:**

- **로봇 전원 ON** 필수. POWER_OFF면 tool voltage가 없어 Modbus 응답이 없고 그리퍼가 움직이지 않습니다(버그 아님, 노드는 계속 재시도).
- **1회성 확인 (블로킹 아님):** External Control을 Play한 뒤 `ros2 topic echo /robotiq_gripper/position_percent`가 계속 갱신되는지 확인 — 즉 드라이버가 인가한 tool voltage가 EC Play 중에도 살아있는지. 끊기면 fallback은 [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) 참조.
- **connect 시 auto-cal(open/close sweep)** — 손가락/물체를 치우세요.
- **방향 반전 = crush 위험:** `gello_gripper_bridge.invert`는 반드시 `false` 유지.
- **깨끗한 종료:** 항상 Ctrl-C(SIGINT)로 종료해 드라이버가 `:54321` / `/tmp/ttyUR`를 반납하도록 합니다.
