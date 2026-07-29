# 01 — Robotiq 2F-85 그리퍼 검증  ✅ PASS

**상태: 실기 검증 완료.** 방향 육안 확인까지 끝나 crush 게이트가 해소되었다.

정본 문서: `docs/ros2/GELLO_UR7E_GRIPPER.md`. 이 문서는 HIL 관점의 검증 절차/판정 기준만 다룬다.

```bash
export WT=/home/laptop3/gello_software     # 2026-07-29 머지(3f199d4) 이후 통합 checkout이 정본
```

---

## 1. 경로 개요 (verified)

```
robotiq_gripper 노드 ──Modbus RTU over TCP──► 192.168.10.11:54321 ──RS485──► 2F-85
                       (tool-communication URCap)
```

- `socat` / `/tmp/ttyUR` **불필요**. 노드가 TCP로 직접 Modbus를 말한다
  (`ur7e_gripper_only.launch.py:3-7`, `robotiq_gripper_modbus_node.py:6`).
- **로봇이 POWER ON이어야 한다.** POWER_OFF면 tool 전압이 안 나와 그리퍼가 죽어 있다.
- **`:54321`은 단일 클라이언트다.** → §5의 충돌 규칙이 이 문서에서 가장 중요한 부분이다.

### 1.1 인터페이스와 단위 계약

| 인터페이스 | 타입 | 단위 계약 | 근거 |
|---|---|---|---|
| `/robotiq_gripper_controller/gripper_cmd` | `control_msgs/action/GripperCommand` | `position` = **미터 gap**. `0.0` = 완전 **닫힘**, `0.085` = 완전 **열림** | `robotiq_gripper_modbus_node.py:16-19`, `_m_to_pos():219` |
| `/robotiq_gripper/set_closed` | `std_srvs/srv/SetBool` | `data: true` = **닫기** | `:20`, `:127` |
| `/robotiq_gripper/command_percent` (sub) | `std_msgs/Float32` | `0.0` = **열림** … `1.0` = **닫힘** | `:23`, `:131-132` |
| `/robotiq_gripper/position_percent` (pub) | `std_msgs/Float32` | `pos255/255.0`. `0.0` = 열림, `1.0` = 닫힘 | `:135`(선언), `:399`(발행) |
| `/robotiq_gripper/joint_states` (pub) | `sensor_msgs/JointState` | `pos255/255 * knuckle_closed_rad(0.8)` | `:396`, `:76` |

> ⚠️ **미터(액션)와 percent(토픽)는 방향이 반대다.** 액션은 `0.085 = 열림`, percent는 `1.0 = 닫힘`.
> 두 개를 섞어 쓰다 부호가 뒤집히면 그게 곧 crush다. 코드에서 이 변환은
> `_m_to_pos()` 하나에만 있다 (`:219`).

---

## 2. 단독 검증 (완료, 재현 절차)

### 2.1 기동

```bash
cd $WT/ros2_ur_ws
./run_ur7e_gripper.sh
# 또는  ROBOT_IP=192.168.10.11 ./run_ur7e_gripper.sh
```

정상 로그:

```
robotiq_gripper up (tcp:192.168.10.11:54321). action=/robotiq_gripper_controller/gripper_cmd service=/robotiq_gripper/set_closed speed=150 force=50
```

### 2.2 열기 / 닫기 (서비스)

```bash
# 다른 터미널
source /opt/ros/humble/setup.bash && source $WT/ros2_ur_ws/install/setup.bash

ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"  # 열기
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"   # 닫기
```

### 2.3 액션 (미터 단위)

```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "{command: {position: 0.085, max_effort: 40.0}}"   # 열기
```

### 2.4 피드백 관측

```bash
ros2 topic echo /robotiq_gripper/position_percent
ros2 topic hz   /robotiq_gripper/position_percent
```

---

## 3. 📌 실측 결과 — 기록 (PASS)

> **이 표는 2026-07-27 세션의 관측 기록이다.** 재현 목표치로 읽되, 설정값으로
> 옮겨 적을 것은 없다(전부 읽기 결과다).

| 판정 항목 | 기대 | **실측** | 판정 |
|---|---|---|---|
| 열림 위치 | ≈ 0.0 | `position_percent` = **0.0118** | PASS |
| 빈손 완전닫힘 | ≈ 1.0 (물리적으로 255에 못 미침이 정상) | **0.8980** (= 229/255) | PASS |
| 액션 열기 `position: 0.085` | `reached_goal: true` | **`reached_goal: true`** | PASS |
| 피드백 주기 | `status_rate_hz` 기본 5.0 Hz (`:78`) | **5.000 Hz, std 1.3 ms** | PASS |
| **방향 육안** | 닫기 명령 → 손가락이 **닫힌다** | **육안 확인 완료** | **PASS** |

> **왜 "빈손 완전닫힘 = 0.898"이 정상인가:** `gPO`는 실제 손가락 위치의 raw 값(0~255)이고,
> 빈손 완전 폐합에서 기구적으로 229 부근에 멈춘다. 코드는 `abs(cur - target) <= 3`
> (`:256`) / `<= 5` (`:271`) 또는 `gOBJ in (1,2)`(물체 접촉 = stalled)로 도달을 판정하므로,
> 255에 도달하지 않아도 `reached_goal`은 정상적으로 뜬다.

> **방향 육안 확인이 왜 게이트였나:** `_send_gripper_command()`(`ur7e_env.py:709`)의
> docstring `:723-725`에 명시적으로
> `VERIFY(hw): on first hardware bring-up confirm direction and scale by eye —
> a silent inversion here is a crush-or-drop hazard`라고 적혀 있었다. 이 게이트는 해소됐다.
> (주석은 코드에 그대로 남아 있다 — 주석을 보고 "미검증"이라고 판단하지 말 것.)

---

## 4. HIL(RL) 통합 시 배선

RL env는 그리퍼를 **percent 토픽**으로 붙는다:

| env 설정 키 | 값 | 상대편 |
|---|---|---|
| `gripper_command_topic` | `/robotiq_gripper/command_percent` | 모드버스 노드의 `~/command_percent` sub |
| `gripper_state_topic` | `/robotiq_gripper/position_percent` | 모드버스 노드의 `~/position_percent` pub |

근거: `serl_ur_infra/ur_env/envs/config.py:116-117`(`ROS` dict), `ros_backend.py:91`, `:117`.
노드 이름이 `robotiq_gripper`(`ur7e_gripper_only.launch.py`의 `name=`)이므로 `~/`가 그대로 맞는다.

### 4.1 RL의 그리퍼 액션은 3-state 이산이다

`action[6] * ACTION_SCALE[2]` 값이

- `<= -0.5` → 닫기 (`send_gripper_percent(1.0)`), 단 현재 위치 `< 0.85`일 때만
- `>= +0.5` → 열기 (`send_gripper_percent(0.0)`), 단 현재 위치 `> 0.15`일 때만
- 그 사이 → **홀드(아무것도 안 함)**

그리고 `GRIPPER_SLEEP = 0.6 s` 디바운스가 걸린다
(`ur7e_env.py:709-741`, 임계는 `:728`/`:731`; `config.py:146`).
`ACTION_SCALE[2]`의 기본값은 **1.0**이다 (`config.py:73` = `[0.0125, 0.0625, 1.0]`).

> 텔레옵(연속 스트리밍)과 달리 RL은 **이산**이다. 이유: hil-serl의 하이브리드 에이전트가
> grasp를 별도 이산 critic으로 학습하고, `GripperPenaltyWrapper`가 open/close **이벤트**를
> 가정하며, 2F-85가 물리적으로 1회 작동에 ~0.5 s 걸리기 때문 (`ur7e_env.py:714-722`).
>
> ⚠️ 단, 실기 러너 `run_real_hil.py`는 기본적으로 `ACTION_SCALE[2] = 0.0`으로 그리퍼를
> **끈다**. `--gripper`를 줘야 켜진다 → `04_HIL_INTERVENTION.md` §6.3.

### 4.2 스트리밍 보호 장치

`~/command_percent`는 단일 클라이언트 버스를 보호하려고 rate limit + deadband가 걸려 있다:

- `command_rate_hz` 기본 20.0 → 최소 50 ms 간격 (`:82`)
- `command_deadband` 기본 0.01 → 그보다 작은 변화는 버림 (`:83`)
- **예외:** 변화량 `>= 0.5`인 큰 점프는 rate limit을 즉시 통과한다 —
  비상 완전개방/폐쇄가 지연되지 않도록 (`:343-345` docstring, 구현 `:355-358`)

---

## 5. 🛑 `:54321` 단일 클라이언트 — 가장 흔한 실패

**같은 순간에 `:54321`에 붙을 수 있는 프로세스는 하나뿐이다.**

| 시나리오 | 결과 | 해결 |
|---|---|---|
| `run_ur7e_gripper.sh` 두 개 동시 실행 | 두 번째가 `no/invalid status response` | 하나만 띄운다 |
| 이전 세션이 Ctrl-C로 안 죽고 남음 | 새 세션이 못 붙음 | `pgrep -af robotiq_gripper_modbus` 후 정리. 노드는 `destroy_node()`에서 `close()`로 포트를 놓는다 (`:401-411`) — SIGKILL로 죽이면 로봇이 포트를 늦게 회수한다 |
| **팔 드라이버(`ur_robot_driver`)와 그리퍼 노드를 동시에** | 드라이버가 `use_tool_communication:=true`로 `:54321`을 이미 점유 | 그리퍼를 **TCP가 아니라 시리얼**로 붙인다: `serial_port:=/tmp/ttyUR` (`ur7e_gripper_only.launch.py:41-42`, `:57`; 노드 쪽 파라미터는 `robotiq_gripper_modbus_node.py:69` — 비어 있지 않으면 TCP 대신 serial) |

> **HIL 세션에서 반드시 기억할 것:** `run_ur7e_gello_real.sh`는 팔 드라이버 + 그리퍼를
> **함께** 띄우고, 그리퍼는 드라이버가 만든 공유 `/tmp/ttyUR` 브리지를 쓴다
> (`run_ur7e_gello_real.sh:24-27`). 그 상태에서 `run_ur7e_gripper.sh`를 **추가로**
> 띄우면 안 된다. 그리퍼 단독 테스트는 팔 드라이버를 내린 상태에서만 한다.

### 5.1 진단

| 증상 | 원인 | 확인 |
|---|---|---|
| `no/invalid status response` | ① 로봇 POWER_OFF (가장 흔함), ② RS485 URCap 미설치, ③ `:54321` 타 클라이언트 점유 | `echo -e 'robotmode\n' \| nc 192.168.10.11 29999` → RUNNING 확인. `pgrep -af "robotiq\|tool_communication"` |
| 붙었다 끊겼다 반복 | 중간에 I/O 실패 → `_drop_connection()` → 재연결 루프 (`:203-215`, `:155-164`) | 노드 로그의 `gripper move failed:` 확인 |
| 액션은 되는데 스트리밍이 씹힘 | deadband/rate limit | §4.2. 액션 실행 후 `_last_cmd_pct`를 동기화하는 코드가 있다 (`:255-261`) — 없으면 스트리밍이 재개 안 되는 버그였음 |

---

## 6. 완료 판정 기준 (전부 충족됨)

- [x] `robotmode` = RUNNING, 그리퍼 노드가 `tcp:...:54321`로 연결 로그를 냄
- [x] `set_closed true/false`로 열고 닫힘
- [x] 액션 `position: 0.085` → `reached_goal: true`
- [x] `position_percent`가 5 Hz로 안정 발행 (std 1.3 ms)
- [x] 열림 0.0118 / 빈손닫힘 0.8980 재현
- [x] **방향 육안 확인** — 닫기 명령이 실제로 닫는다

---

## 7. 남은 것 (이 문서 범위에서 미검증)

> 🔧 **2026-07-27 재확인:** §3의 실측치(열림 0.0118 / 빈손닫힘 0.8980 = 229/255 /
> 피드백 5.000 Hz)는 이번 실기 세션에서도 그대로 재현됐다. **§6의 완료 판정은 유효하다.**
> 아래 미검증 항목은 여전히 미검증이다 — §6과 섞지 말 것.
>
> 🔧 **2026-07-29 머지 반영:** 머지(`3f199d4`)는 그리퍼 경로를 바꾸지 않았다.
> 2026-07-28 실기 세션이 팔을 구동했지만 `run_real_hil.py`의 기본은 그리퍼 **비활성**
> (`ACTION_SCALE[2]=0.0`)이므로 아래 미검증 항목은 **하나도 해소되지 않았다.**

- [ ] **RL 경로**(`/robotiq_gripper/command_percent`)로 그리퍼가 실제로 움직이는지 —
      env에서 액션 `[0,0,0,0,0,0,-1]` / `[0,...,+1]`을 쏴서 확인. `07_FAILURE_INJECTION.md` E12.
- [ ] **개입 경로**(리더 트리거 → `intervene_action[6]` → 그리퍼). 배선 코드는 커밋됐지만
      (`08_OPEN_GAPS.md` G3) 하드웨어 미확인. 판정은 `04_HIL_INTERVENTION.md` §6.3.
- [ ] `run_hil_actor.sh`의 preflight [7b]가 `/robotiq_gripper/position_percent`를 확인하지만
      **WARN이지 FAIL이 아니다** — 그리퍼 없이도 actor가 뜬다. 19-D state의 그리퍼 채널이
      조용히 0.0("열림")으로 채워진다는 뜻이다 (`08_OPEN_GAPS.md` G8).
- [ ] `GRIPPER_SLEEP=0.6 s` 디바운스가 10 Hz 정책 루프에서 실제로 어떻게 보이는지 (6 스텝에 1번만 반영).
- [ ] 물체를 쥔 상태에서 `gOBJ`(stalled) 판정과 `GripperPenaltyWrapper` 보상 연동.
