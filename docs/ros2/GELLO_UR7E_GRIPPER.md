# UR7e Robotiq 2F-85 그리퍼 제어 (Modbus over tool-comm · Humble · 검증됨)

이 문서는 **UR7e의 tool 포트에 배선된 Robotiq 2F-85 그리퍼**를, UR 컨트롤러의 RS485 tool-communication TCP 포트를 통해 **Modbus RTU로 직접** 구동하는 ROS 2 Humble 경로를 다룹니다. 실기(UR7e, PolyScope 5.24)에서 **오늘 실제로 열고/닫고/상태를 읽어 검증한(verified) 경로**입니다. 사용하는 패키지는 `ros2_ur_ws/src/ur_gello_bringup`이며, 그리퍼만 단독으로 브링업합니다 — 팔도, ros2_control도 띄우지 않습니다.

> **이 문서가 검증하는 것**: 이 PC(eno1 = 192.168.10.10)에서 UR7e(192.168.10.11, PolyScope **5.24** — PolyScope X 아님)의 2F-85를 `ur7e_gripper_only.launch.py`로 브링업하여, `set_closed` 서비스 / `GripperCommand` 액션 / 상태 토픽까지 실기에서 동작 확인했습니다. **팔(GELLO 텔레오퍼) 경로와는 완전히 독립**입니다 — 그리퍼만 필요할 때 이 문서를 쓰면 됩니다.

> **팔 텔레오퍼로 가려면**: GELLO → UR7e 관절 원격 조종은 [`GELLO_UR7E_ROS2_BRINGUP.md`](./GELLO_UR7E_ROS2_BRINGUP.md)(sim/mock) 및 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)(실기)를 참고하세요.

---

## 빠른 시작 (Quick Start)

전제: 리포는 이미 빌드되어 있고, **로봇은 전원이 켜져 있어야 합니다**(아래 [사전 준비](#1-사전-준비-prerequisites) 참고 — powered-off이면 그리퍼가 응답하지 않습니다).

```bash
# 1) 로봇이 켜져 있는지 확인 (RUNNING 또는 최소한 IDLE/powered — POWER_OFF면 안 됨)
echo -e 'robotmode\n' | nc 192.168.10.11 29999

# 2) 빌드 (한 번)
cd /home/laptop3/gello_software/ros2_ur_ws
./build_ur7e.sh

# 3) 그리퍼 노드 실행
source install/setup.bash
./run_ur7e_gripper.sh
#   또는 직접:
#   ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py robot_ip:=192.168.10.11
```

다른 터미널에서 제어 (각 터미널마다 `source /opt/ros/humble/setup.bash && source install/setup.bash` 먼저):

```bash
# 편의 서비스로 열기/닫기 (가장 간단)
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"    # 닫기
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"   # 열기

# 표준 액션 (position은 METERS: 0.0=닫힘 .. 0.085=열림, max_effort는 Newton)
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
    control_msgs/action/GripperCommand "{command: {position: 0.085, max_effort: 40.0}}"

# 현재 위치(0.0 열림 .. 1.0 닫힘) 한 번 읽기
ros2 topic echo --once /robotiq_gripper/position_percent
```

- **핵심**: socat도, `/tmp/ttyUR`도, C++ 빌드도, 추가 ROS 패키지도 필요 없습니다. 노드가 TCP 소켓 위로 Modbus를 직접 말합니다.
- **가장 흔한 실패 원인은 "로봇이 꺼져 있음"**입니다 (그리퍼는 tool 전압으로 구동되고, tool 전압은 POWER_OFF 상태에서 꺼집니다). 응답이 없으면 먼저 전원부터 의심하세요.
- **정상 종료는 반드시 `Ctrl-C`(clean SIGINT)** — 노드가 `:54321` 소켓을 깔끔히 반납해야 다음 실행이 바로 붙습니다.

---

## 목차

- [0. 이 경로가 무엇인가 — "Path B"](#0-이-경로가-무엇인가--path-b)
- [1. 사전 준비 (Prerequisites)](#1-사전-준비-prerequisites)
- [2. 빌드 & 실행](#2-빌드--실행)
- [3. ROS 2 인터페이스 레퍼런스](#3-ros-2-인터페이스-레퍼런스)
- [4. 노드 파라미터](#4-노드-파라미터)
- [5. 프로토콜 부록 (Modbus / Robotiq 레지스터)](#5-프로토콜-부록-modbus--robotiq-레지스터)
- [6. 트러블슈팅](#6-트러블슈팅)
- [7. 팔 제어와 동시 사용 (⚠️ tool 전압 ↔ External Control)](#7-팔-제어와-동시-사용--tool-전압--external-control)

---

## 0. 이 경로가 무엇인가 — "Path B"

Robotiq 2F-85는 UR7e의 **tool 플랜지 RS485 라인**에 배선되어 있습니다. 이 라인을 ROS 2로 구동하는 방법은 크게 두 가지입니다.

- **Path A (URCap 소켓, 이 로봇의 config 아님)** — PolyScope에 Robotiq **"Grippers" URCap**을 설치하면 컨트롤러가 그리퍼 서버를 **포트 63352**로 노출합니다. 리포에는 이 경로용 대체 노드(`robotiq_urcap` / `robotiq_gripper_action`)도 들어 있지만, **이 로봇의 설정이 아닙니다.**
- **Path B (RS485 tool-comm, 이 문서 · 검증됨)** — PolyScope에 UR **"RS485 / tool communication" URCap**을 설치하면 컨트롤러가 tool 플랜지 RS485 라인을 **`ROBOT_IP:54321`** 의 raw TCP 패스스루로 노출합니다. 노드는 이 TCP 소켓 위로 **Modbus RTU(+CRC)를 직접** 말합니다. 이 로봇에는 RS485 URCap이 **이미 설치되어 있고(포트 54321이 열려 있음)**, 이것이 검증된 경로입니다.

> **⚠️ 두 URCap은 상호 배타적(mutually exclusive)입니다.** RS485 tool-communication URCap(포트 54321)과 Robotiq "Grippers" URCap(포트 63352)을 **동시에 설치할 수 없습니다** — 둘 다 tool RS485 라인을 독점하기 때문입니다. 이 로봇은 Path B(54321)로 잡혀 있으므로, 이 문서의 절차만 사용하세요.

> **socat 브리지는 대안일 뿐 기본이 아님.** 고전적인 `ros2 run ur_robot_driver tool_communication.py` socat 브리지(`/tmp/ttyUR` 시리얼 장치를 만들어 줌)로도 같은 RS485 라인에 붙을 수 있고, 노드의 `serial_port` 파라미터가 이 경로를 지원합니다(비우면 TCP, 채우면 시리얼). 하지만 **direct-TCP가 기본이자 가장 단순**하므로 특별한 이유가 없으면 socat을 쓰지 마세요. 또한 socat 브리지와 이 노드는 **둘 다 `:54321`을 점유**하므로 동시에 띄우면 충돌합니다([트러블슈팅](#6-트러블슈팅) 참고).

---

## 1. 사전 준비 (Prerequisites)

### 1-1. 네트워크 / 하드웨어

- 이 PC: `eno1` = **192.168.10.10**
- UR7e 컨트롤러: **192.168.10.11**, PolyScope **5.24** (PolyScope X 아님)
- 그리퍼는 UR7e의 **tool 포트에 물리적으로 배선**되어 있음.

### 1-2. PolyScope URCap

- UR **"RS485 / tool communication" URCap**이 설치되어 있어야 하며(이 로봇은 이미 설치됨 — 포트 54321 열려 있음), 그러면 tool RS485가 `192.168.10.11:54321`로 노출됩니다.
- Robotiq **"Grippers" URCap(63352)은 설치하지 마세요** — 위 [0절](#0-이-경로가-무엇인가--path-b)의 상호 배타성 참고.

### 1-3. ⚠️ 로봇 전원 (가장 중요한 사전 조건)

**그리퍼는 tool 전압으로 구동되고, tool 전압은 로봇이 POWER_OFF일 때 꺼집니다.** 따라서 전원이 꺼진 로봇은 Modbus에 **아무 응답도 하지 않습니다 — 이것은 버그가 아닙니다.**

실행 전에 대시보드로 robotmode를 확인하세요:

```bash
echo -e 'robotmode\n' | nc 192.168.10.11 29999
# 기대: RUNNING (또는 최소한 IDLE/powered). POWER_OFF면 그리퍼가 죽어 있음.
```

`POWER_OFF`라면 펜던트로 전원을 켜거나, 대시보드로 켭니다:

```bash
echo -e 'power on\n'       | nc 192.168.10.11 29999
echo -e 'brake release\n'  | nc 192.168.10.11 29999
```

> 노드는 붙지 못하면 **백그라운드에서 재접속을 계속 재시도**하므로, 노드를 먼저 띄워 두고 로봇 전원을 나중에 켜도 됩니다 — 전원이 들어오는 순간 자동으로 연결됩니다.

### 1-4. ROS 2 환경

- OS: Ubuntu 22.04, ROS 2: **Humble**.
- 매 새 터미널에서: `source /opt/ros/humble/setup.bash`, 빌드 후에는 `source install/setup.bash`.
- 이 경로는 **GELLO도, dynamixel-sdk도, `GELLO_REPO_ROOT`도 필요 없습니다** — 그리퍼 노드는 GELLO/팔과 무관합니다.

---

## 2. 빌드 & 실행

### 2-1. 빌드

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./build_ur7e.sh
#   또는 직접:
#   source /opt/ros/humble/setup.bash
#   colcon build --packages-select ur_gello_bringup
#   source install/setup.bash
```

### 2-2. 실행

```bash
source install/setup.bash
./run_ur7e_gripper.sh
#   또는 직접:
#   ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py robot_ip:=192.168.10.11
```

launch 인자(기본값): `robot_ip:=192.168.10.11`, `speed:=150`(0-255), `force:=50`(0-255, 액션의 `max_effort=0`일 때 사용).

```bash
# 예: force를 올려 실행
ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py robot_ip:=192.168.10.11 force:=60
```

정상 기동 시 노드가 다음과 같이 로그합니다: 연결되면 `Gripper connected & ready: {...}` (필요 시 activation 자동 스윕 후). 연결이 안 되면 `Gripper not ready (...). Is the robot POWERED ON?`를 주기적으로 남기며 재시도합니다.

> **⚠️ 정상 종료는 `Ctrl-C`(clean SIGINT).** 노드는 종료 시 `:54321` 소켓을 깔끔히 닫아 컨트롤러가 포워더를 반납하게 합니다(`destroy_node`가 `close()` 호출). 하드킬(`kill -9`)하면 컨트롤러가 죽은 소켓을 붙들어(FIN-WAIT-2) 잠시 재접속이 막힙니다 — [트러블슈팅](#6-트러블슈팅) 참고.

---

## 3. ROS 2 인터페이스 레퍼런스

노드 이름은 `robotiq_gripper`입니다. 인터페이스는 **연결 여부와 무관하게 항상 광고(advertise)** 되며, 연결이 안 된 상태에서 명령하면 노드가 안전하게 거부(abort / success=false)합니다.

### 3-1. 액션 — `/robotiq_gripper_controller/gripper_cmd`

타입: `control_msgs/action/GripperCommand`

- `goal.command.position` — **미터(m) 단위 gap**. **0.0 = 완전 닫힘, 0.085 = 완전 열림** (ROS 관례). 내부에서 0(open)..255(closed) raw 위치로 변환됩니다.
- `goal.command.max_effort` — **Newton**. `0`이면 노드 기본 `force` 파라미터를 사용. `max_force_n`(기본 235N)을 255에 매핑해 rFR로 변환합니다.
- 결과/피드백: `.position`(m), `.stalled`(물체 접촉 = `gOBJ` 1/2), `.reached_goal`.

```bash
# 완전 열기 (position=0.085)
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
    control_msgs/action/GripperCommand "{command: {position: 0.085, max_effort: 40.0}}"

# 완전 닫기 (position=0.0)
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
    control_msgs/action/GripperCommand "{command: {position: 0.0, max_effort: 40.0}}"
```

### 3-2. 서비스 — `/robotiq_gripper/set_closed`

타입: `std_srvs/srv/SetBool` — 열기/닫기 편의 서비스.

- `data: true` → **닫기**(POS=255), `data: false` → **열기**(POS=0).

```bash
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"    # 닫기
ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"   # 열기
```

연결이 안 된 상태에서 호출하면 `success: false`와 함께 `not connected (...); robot powered on?` 메시지를 반환합니다.

### 3-3. 토픽

| 토픽 | 타입 | 비고 |
| --- | --- | --- |
| `/robotiq_gripper/position_percent` | `std_msgs/Float32` | **0.0 = 열림 .. 1.0 = 닫힘** (raw `gPO`/255). 기본 5Hz 발행. |
| `/robotiq_gripper/joint_states` | `sensor_msgs/JointState` | 관절명 `robotiq_85_left_knuckle_joint`, position = `gPO`/255 × `knuckle_closed_rad`(기본 0.8 rad). |

```bash
ros2 topic echo --once /robotiq_gripper/position_percent
ros2 topic echo --once /robotiq_gripper/joint_states
```

> 위치 규약이 두 가지라 헷갈리기 쉽습니다: **액션의 position은 미터(0=닫힘, 0.085=열림)**, **`position_percent` 토픽은 0..1(0=열림, 1=닫힘)** 입니다. 방향이 반대이니 주의하세요.

---

## 4. 노드 파라미터

`ur7e_gripper_only.launch.py`가 설정하는 주요 파라미터 외에, 노드(`robotiq_gripper_modbus_node.py`)가 선언하는 전체 파라미터입니다.

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `robot_ip` | `192.168.10.11` | UR7e 컨트롤러 IP. |
| `tool_comm_port` | `54321` | RS485 tool-comm TCP 포트. |
| `serial_port` | `""` (비움) | 비우면 **TCP(기본)**. 채우면(예: `/tmp/ttyUR`) socat 시리얼 경로 사용. |
| `speed` | `150` | rSP, 0-255. |
| `force` | `50` | rFR, 0-255. 액션 `max_effort=0`일 때 기본 힘. |
| `connect_on_start` | `True` | 기동 시 재접속 루프 시작. |
| `activate_on_connect` | `True` | 연결 후 미활성(gACT/gSTA)이면 activation 자동 스윕 실행. |
| `stroke_m` | `0.085` | 완전 스트로크(m). 미터↔raw 변환 기준. |
| `max_force_n` | `235.0` | max_effort(N)→rFR(255) 매핑 기준. |
| `knuckle_closed_rad` | `0.8` | joint_states의 닫힘 각도(rad). |
| `joint_name` | `robotiq_85_left_knuckle_joint` | joint_states 관절명. |
| `status_rate_hz` | `5.0` | 상태 토픽 발행 주기. |
| `move_timeout_s` | `3.0` | 액션 이동 폴링 타임아웃. |
| `reconnect_period_s` | `3.0` | 재접속 재시도 간격. |

> 노드는 **동시에 하나의 재접속 루프만** 돌리도록 락(`_connecting`)으로 보호합니다 — 여러 루프가 `:54321`을 storm하는 것을 막기 위함입니다. 마찬가지로 드라이버(`Robotiq2F85`)는 모든 버스 트랜잭션을 내부 락으로 직렬화합니다.

---

## 5. 프로토콜 부록 (Modbus / Robotiq 레지스터)

Robotiq 2F-85/2F-140 매뉴얼 기준. 드라이버(`robotiq_2f85_modbus.py`)는 Modbus RTU(+CRC16)를 TCP/시리얼 위로 직접 프레이밍합니다.

- **슬레이브 ID**: `0x09`.
- **읽기 (상태)** — FC03 `@0x07D0`, 3 레지스터 / 6 바이트: `[status, _, gFLT, gPR, gPO, gCU]`
  - `status` 비트: `bit0=gACT`, `bit3=gGTO`, `bit4-5=gSTA`, `bit6-7=gOBJ`.
- **쓰기 (명령)** — FC16 `@0x03E8`, 3 레지스터 / 6 바이트: `[action, 0, 0, rPR, rSP, rFR]`
  - `action` 비트: `bit0=rACT`, `bit3=rGTO`, `bit4=rATR`, `bit5=rARD`.
  - `move()`는 `action = rACT|rGTO = 0x09`로 위치/속도/힘을 함께 씀.
- **위치(POS)**: **0 = 열림, 255 = 닫힘** (드라이버/gripper 원시값). 액션 API의 미터 규약(0=닫힘)과 반대이니 변환에 주의.
- **Activation**: `rACT` 0→1 후 `gSTA==3`이 될 때까지 대기. activation은 그리퍼 내부 **auto-calibration 열림/닫힘 스윕**을 트리거하며, 이전 fault(예: 0x09 comm-loss)도 클리어합니다.
- **gOBJ 판정**: `1`/`2` = 물체를 잡아 정지(object grasped), `3` = 목표 도달 / 물체 없음(reached, no object), `0` = 이동 중.

> 첫 RTU 프레임은 fresh connection 직후 간혹 드롭되므로, 드라이버는 read/write 트랜잭션을 각각 몇 회 재시도합니다.

---

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| **상태 응답 없음 / "not connected"** (`no/invalid status response`) | ① **로봇이 POWER_OFF** (가장 흔함 — tool 전압이 꺼져 그리퍼가 죽어 있음), ② RS485 tool-communication URCap 미설치, ③ 다른 클라이언트가 이미 `:54321`을 점유 | 먼저 `echo -e 'robotmode\n' \| nc 192.168.10.11 29999`로 **RUNNING/powered 확인** → 아니면 전원 켜기(`power on` + `brake release`). URCap/포트 확인. 다른 그리퍼 노드나 `tool_communication.py`가 떠 있지 않은지 확인. |
| 노드를 하드킬 후 재실행하니 한동안 연결이 안 됨 | **`:54321`은 한 번에 한 클라이언트만** 소유 가능. 하드킬된 노드가 stale connection(FIN-WAIT-2)을 남겨 재접속을 잠시 굶김 | **약 30-45초 기다리면** 노드의 재시도 루프가 자동 복구합니다. 애초에 **항상 `Ctrl-C`(clean SIGINT)로 종료**해 소켓을 반납하세요. |
| 두 개의 그리퍼 노드 / socat 브리지가 동시에 뜸 | 같은 로봇의 `:54321`을 여러 클라이언트가 두고 다툼 | **하나만** 실행. 이 노드를 쓰면 `tool_communication.py` socat 브리지를 같은 로봇에 **동시에 돌리지 마세요**. |
| 빈 손으로 완전 닫기 했는데 100%가 아니라 90% 근처(POS ≈ 229)에서 멈춤 | **2F-85의 정상 동작** — 빈 완전 닫힘은 POS 255가 아니라 ~229에서 안착 | 정상. 노드는 `gOBJ==3`(reached/no-object)로 이 경우를 성공으로 판정하므로 거리만 보고 실패로 오판하지 않음. |
| fault `gFLT=0x09` ("no communication >1s") | 통신이 1초 이상 끊겼을 때 뜨는 **양성(benign) fault** | 무해. (재)activation 시 자동으로 클리어됩니다. |
| **팔 teleop을 시작(External Control Play)하자마자 그리퍼가 죽음** — read 실패/`not ready` 반복 | External Control 프로그램이 **tool 전압을 리셋/차단**했을 수 있음 (아래 §7 참고) | tool 전압을 **Installation 탭이 아니라 드라이버 launch의 `tool_voltage:=24`로** 설정. 그래도 끊기면 §7의 검증 절차로 확인. |

---

## 7. 팔 제어와 동시 사용 (⚠️ tool 전압 ↔ External Control)

> 그리퍼만 쓸 때는 이 절과 무관합니다. **팔 teleop(External Control)과 그리퍼를 동시에** 쓸 때만 해당됩니다.

**핵심**: 이건 "데이터 통신 꼬임"이 아니라 **전원(tool 전압) 충돌**입니다. 팔 제어(External Control)는 관절 제어라 그리퍼의 tool RS485 라인과 **물리적으로 별개**입니다 — 팔이 움직여도 그리퍼 Modbus 데이터는 꼬이지 않습니다. 유일한 커플링은 그리퍼를 구동하는 **tool 전압(24V)** 하나입니다.

알려진 충돌(이전 RS485+2F-85 셋업 경험):
1. Installation 탭 드롭다운으로 tool 전압을 수동 인가하면 → **External Control이 켜질 때 그 전압이 차단**될 수 있음 → 그리퍼 전원 끊김.
2. External Control 실행 중에 Installation 탭에서 전압을 재인가하면 → **External Control이 중단**됨.

이 리포의 완화책 (위 충돌을 피하는 올바른 구성):
- **Robotiq "Grippers" URCap(63352)이 아니라 RS485 tool-comm URCap** 사용 → RS485 라인 경합이 원천적으로 없음.
- tool 전압을 **Installation 탭이 아니라 드라이버 launch 인자로** 인가:
  ```bash
  ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.10.11 \
      use_tool_communication:=true tool_voltage:=24 use_mock_hardware:=false launch_rviz:=false
  ```
  그리고 그리퍼 노드는 `serial_port:=/tmp/ttyUR`로 이 드라이버의 socat 브리지를 공유:
  ```bash
  ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py serial_port:=/tmp/ttyUR
  ```
- **노드가 전원 끊김을 안전하게 드러냄**: tool 전압이 빠지면 Modbus read가 실패 → `not ready` 경고 + 자동 재접속, 전원 복구 시 자동 재연결. 조용히 오작동하지 않음.

**아직 미검증인 지점 (반드시 한 번 확인하세요)**: 지금까지 검증된 것은 (a) 그리퍼 단독, (b) 팔 드라이버는 떠 있으나 **External Control은 정지** 상태에서의 그리퍼 동작입니다. **External Control이 실제 Play 중일 때** tool 전압이 유지되는지는 셋업별로 다를 수 있으므로, 팔+그리퍼 동시 운용 전 아래를 1회 검증하세요:

1. 펜던트에서 **Remote Control ON**.
2. 위 `ur_control.launch.py ... tool_voltage:=24` 로 팔 드라이버 기동.
3. `serial_port:=/tmp/ttyUR`로 그리퍼 노드 기동 → `ros2 topic echo /robotiq_gripper/position_percent`가 갱신되는지(=그리퍼 전원 ON) 확인.
4. **External Control 프로그램을 Play**(펜던트 또는 헤드리스)로 시작.
5. 그리퍼가 **여전히 응답하는지** 재확인. 멈추면 → External Control이 tool 전압을 끊은 것(충돌 확정), 계속 응답하면 → 동시 운용 안전.

참고: [UR ToolComm Forwarder URCap](https://github.com/UniversalRobots/Universal_Robots_ToolComm_Forwarder_URCap), [tool communication 설정 문서](https://docs.ros.org/en/humble/p/ur_robot_driver/doc/setup_tool_communication.html).

---

**안전 재확인**: 이 그리퍼 경로는 GELLO/팔과 독립이지만, 리포의 안전 불변식은 그대로 유지됩니다 — **GELLO는 언제나 passive read-only이며, Dynamixel에 절대 토크를 인가하지 않습니다.** 그리퍼 작업 시에도 로봇이 물리적으로 움직일 수 있으니(전원 ON 상태), E-STOP을 손 닿는 곳에 두고 tool 주변에 손가락/물체가 끼지 않도록 작업공간을 확보하세요.

---

## 관련 문서

- [`GELLO_UR7E_ROS2_BRINGUP.md`](./GELLO_UR7E_ROS2_BRINGUP.md) — GELLO → UR7e 관절 텔레오퍼(sim/mock, 검증됨).
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실물 UR7e + 실 GELLO 구동 런북.
- [`../../UR7e_Remote_Control_ROS2.md`](../../UR7e_Remote_Control_ROS2.md) — UR7e Remote Control(헤드리스/대시보드) 레퍼런스.
