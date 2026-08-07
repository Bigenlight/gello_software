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
- [8. 그리퍼 DISCRETE 모드 (GELLO 텔레오퍼 전용, opt-in)](#8-그리퍼-discrete-모드-gello-텔레오퍼-전용-opt-in)

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

## 8. 그리퍼 DISCRETE 모드 (GELLO 텔레오퍼 전용, opt-in)

> **이 절만 대상 노드가 다릅니다.** §1~§7은 `robotiq_gripper`(Modbus 단독 경로)를 다루지만, 이 절은 **GELLO 텔레오퍼 중에만 도는 `gello_gripper_bridge`** — 리더 방아쇠값을 읽어 `/robotiq_gripper/command_percent`로 흘려 주는 노드 — 의 동작을 바꿉니다. 그리퍼 단독 운용(§2의 `run_ur7e_gripper.sh`)에는 **아무 영향이 없습니다.**

**기본값은 꺼짐(`continuous`)입니다.** 켜지 않으면 파이프라인은 이전과 완전히 동일하게 연속값을 그대로 전달합니다.

### 8-1. 무엇을 고치는 모드인가 (실측 근거)

GELLO 방아쇠는 **스프링이 Dynamixel 엔코더에 대고 되돌아오는** 구조입니다. take 125개 / 샘플 52,607개 전수 측정 결과 이 신호는 **이미 거의 이진**입니다(90.95%가 0.02 미만 또는 0.98 초과). 문제는 나머지입니다 — **세션당 대략 1~2회, 스프링이 끝까지 돌아오지 못하고 방아쇠가 "덜 열린 채" 멈춥니다.** 관측된 최악의 정지값은 **0.229**(2026-07-07)와 **0.243**(2026-07-20)이고, 브릿지는 그 값을 **의심 없이 그대로 전달**하므로 Robotiq은 76%만 열립니다. 조작자에게는 **"그리퍼가 다 안 열렸다"**로 보입니다.

DISCRETE 모드는 **0.3 이하를 전부 정확히 `0.0`으로 스냅**합니다. **관측된 이상값은 전부 0.3 아래**이므로 지금까지 나온 사례가 **전부** 이것으로 해결됩니다.

⚠️ **캘리브레이션 문제가 아닙니다.** take **125개 전부**에서 방아쇠가 어느 시점엔가 정확히 `0.000`에 도달합니다 — 바닥이 통째로 들려 있는 것이 아니라 **산발적으로 덜 돌아오는 것**입니다. 그래서 재캘리브레이션으로는 없어지지 않습니다.

🟡 **대신 잃는 것 — 속도와 부분 파지입니다.** 이산 모드에서 브릿지가 내보내는 것은 `0.0` 아니면 `1.0`, 즉 **한 번의 전(全)스트로크 명령**뿐입니다. 그 명령은 드라이버에 설정된 **고정 속도·고정 힘**으로 실행되므로, **방아쇠를 천천히 당겨 천천히 닫거나 중간에서 멈춰 살짝 쥐는 것이 불가능해집니다.** 이것은 버그가 아니라 이 모드의 정의입니다. 받아들일 만한 이유는 위 실측입니다 — 샘플의 **90.95%가 이미 0.02 미만 또는 0.98 초과**라 조작자는 사실상 부분 파지를 쓰지 않고 있었습니다. 다만 **하드웨어 앞에서 발견할 일이 아니라 미리 알고 켜야 하는 사실**이라 여기 적어 둡니다. 부분 파지가 필요한 작업이면 `gripper_mode`를 기본 `continuous`로 두세요.

> 📎 **숫자의 정본은 [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) §5.1입니다** — 날짜별 최악값·0.3까지의 여유·빈도 표, 그리고 임계값 선택 근거가 거기 있습니다. **여기에 옮겨 적지 않았으니** 수치가 필요하면 그쪽을 보세요. 여유가 **0.057**까지 좁아진 적이 있으므로 임계값이 상수가 아니라 파라미터인 것이고, **그리퍼를 재캘리브레이션하면 그 표를 다시 재야 합니다.**

### 8-2. 켜는 법

launch 인자 `gripper_mode`(기본 `continuous`) 하나입니다. 팔 텔레오퍼 명령줄 **끝에 그대로 덧붙이면** 됩니다(`run_ur7e_gello_real.sh`가 뒤에 붙은 인자를 `ros2 launch`로 그대로 넘깁니다).

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
source install/setup.bash

# EEF 모드 + DISCRETE 그리퍼
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef gripper_mode:=discrete

# joint 모드에도 똑같이 붙습니다
HEADLESS=true ./run_ur7e_gello_real.sh gripper_mode:=discrete
```

`gripper_mode`는 **그리퍼 브릿지 파라미터만** 건드립니다 — `control_mode`, EEF 델타, 컨트롤러 전환 등 **팔 경로와는 직교**합니다.

실제로 켜졌는지 확인 (다른 터미널):

```bash
ros2 param get /gello_gripper_bridge discrete_mode
ros2 topic echo --once /gello_gripper_bridge/discrete_state
```

브릿지 파라미터:

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `discrete_mode` | `False` | `gripper_mode:=discrete`가 이것을 `True`로 만듭니다. `False`면 이 절 전체가 무효(연속 전달). |
| `discrete_close_at` | `0.7` | 방아쇠가 이 값 **이상**이면 CLOSED로 래치 → 명령 `1.0`. |
| `discrete_open_at` | `0.3` | 방아쇠가 이 값 **이하**면 OPEN으로 래치 → 명령 `0.0`. |

> 임계값은 `0.0 < discrete_open_at < discrete_close_at < 1.0`을 만족해야 합니다. 양 끝의 등호 배제도 의미가 있습니다 — `open_at = 0.0`이면 **정확히 0.000인 샘플에서만** 열림으로 래치되어 0.229 같은 이상값이 영영 스냅되지 않습니다(이 기능의 존재 이유가 통째로 사라집니다).

### 8-3. 임계값과 히스테리시스, 그리고 `UNKNOWN`

```
방아쇠 >= 0.7   ->  CLOSED   (command_percent 1.0)
방아쇠 <= 0.3   ->  OPEN     (command_percent 0.0)
그 사이         ->  직전 상태 유지          <- 히스테리시스
첫 임계 통과 전 ->  UNKNOWN  (새로 명령하는 것이 아무것도 없습니다)
일시정지–재개 뒤 -> UNKNOWN  (래치를 되돌립니다 — §8-6의 GO HOME 항목)
```

- **왜 임계값이 두 개인가**: 하나뿐이면 방아쇠가 그 근처를 지날 때 출력이 **채터링**합니다. 0.3~0.7 밴드 안에서는 직전 상태를 기억하는 것이 그 대책입니다.
- **0.3 / 0.7은 새로 정한 값이 아닙니다.** 오프라인 RL 데모 변환기가 이미 쓰고 있던 **정본 상수**이며, 근거는 §5.1입니다. **다른 값을 발명해 넣지 마세요.**
- **`UNKNOWN`은 기동 직후의 정상 상태이며, 일시정지 뒤에도 여기로 돌아옵니다.** 래치는 **어느 쪽으로도 추정하지 않습니다** — "아마 열려 있겠지"로 시작하면 **추측을 근거로 실제 모션을 명령**하게 되고, "아마 닫혀 있겠지"는 끼임입니다. 조작자의 방아쇠가 임계값을 **직접 한 번 넘을 때까지** 브릿지는 새 명령을 내지 않습니다. 비용은 사실상 0입니다: take 125개 중 **임계값에 도달하지 못한 take는 0개**이고, 보통 첫 잡기 사이클 안에 해소됩니다.

> ⚠️ **규약 주의 — 이 토픽들에서는 `0.0 = 열림`, `1.0 = 닫힘`입니다.** (§3-3의 `position_percent`와 같은 방향이고, 액션 API의 미터 규약과는 반대입니다.) 그리고 **HIL-SERL의 RL 액션은 이것과 정확히 반대**(+1=열림, −1=닫힘)입니다. 그 뒤집기는 오프라인 데모 변환 단계에서만 일어나고 **녹화는 언제나 0/1 규약**입니다 — 계층별 전체 대조표는 [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) **§5**에 있으니 여기서 다시 설명하지 않습니다.

### 8-4. `discrete_state` / `discrete_trigger` 토픽과 GUI 표시등 — **없으면 조용히 실패합니다**

`/gello_gripper_bridge/discrete_state` (`std_msgs/String`, 5 Hz) — **다섯 단어입니다**:

| 값 | 뜻 |
| --- | --- |
| `DISABLED` | 이산 모드가 **꺼져 있음** — 연속값을 그대로 전달 중. |
| `UNKNOWN` | 켜져 있으나 **아직 임계값을 한 번도 넘지 않음** → 새 명령을 내지 않음. |
| `OPEN` | 마지막 래치가 열림 — `0.0`을 명령 중. |
| `CLOSED` | 마지막 래치가 닫힘 — `1.0`을 명령 중. |
| `RAMPING` | 래치는 이미 정해졌고, **재개 직후의 슬루 구간**을 지나는 중 — 출력이 실측 위치에서 래치 끝점으로 기어가고 있습니다. 최대 `resume_ramp_s`(2.0초)로 **유한하며 고장이 아닙니다.** 이 구간의 중간값은 목표가 아니라 통과점입니다. |

`/gello_gripper_bridge/discrete_trigger` (`std_msgs/Float32`, 같은 5 Hz) — **래치가 판정 중인 원본(클램프된) 방아쇠값**(`0.0`=열림 .. `1.0`=닫힘, 위와 같은 규약)입니다. 래치는 **결론**이고 이쪽은 **근거**입니다.

EEF 조작 GUI에 이 둘이 **한 줄짜리 라이브 표시등**으로 뜹니다 — `GRIP: CLOSED  trig 0.94` 형태로 래치 옆에 방아쇠값이 같이 나옵니다. 그리고 **방아쇠가 두 임계값 사이(경계 제외)에 1.5초 넘게 계속 머무르면 `in band 1.8s`가 붙고 색이 `UNKNOWN`과 같은 호박색으로 바뀝니다.** 어느 쪽 토픽이든 2초간 조용하면 각각 `no signal` / `trig --`로 떨어집니다. GUI가 밴드를 아는 방법은 기동 시 `/gello_gripper_bridge`에 `GetParameters`를 **한 번** 던져 `discrete_open_at`/`discrete_close_at`을 읽는 것이고, 그 읽기가 실패하면(구버전 브릿지 등) **방아쇠값은 그대로 보여 주고 `in band` 판정만 조용히 끕니다** — 밴드를 모르면서 추측해 깃발을 세우지는 않습니다.

> 기존 `/gello_gripper_bridge/state`(**PAUSED / WAITING / RAMPING / FOLLOWING**)는 **그대로입니다.** 이산 상태를 그쪽 어휘에 섞지 않고 **별도 토픽**으로 뺀 이유는, `state`의 네 단어를 파싱하는 다른 코드가 있어 새 토큰이 늘어나면 안 되기 때문입니다. 두 토픽은 보는 것이 다릅니다 — `state`는 **브릿지가 흐르고 있는가**, `discrete_state`는 **래치가 어디에 있는가**. (`RAMPING`이라는 단어가 양쪽에 다 있지만 같은 구간을 각자의 관점에서 부르는 것입니다.)

🛑 **왜 이것들이 필수인가 — 임계값을 잘못 잡으면 증상이 "무증상"입니다.** 예를 들어 `discrete_close_at`을 조작자의 방아쇠가 실제로는 도달하지 못하는 값으로 올려 두면, 래치는 영영 넘어가지 않고 브릿지는 **새 명령을 아예 내지 않습니다.** 그리퍼는 그냥 **반응을 멈춥니다 — 에러도, 경고도, 거부 응답도 없습니다.** 로그를 뒤져도 나올 것이 없습니다(코드는 계약대로 동작하고 있으니까요).

⚠️ **그런데 래치 표시등만으로는 이것을 잡을 수 없습니다.** 래치가 `UNKNOWN`에 멈추는 것은 이 실패의 **기동 직후 얼굴**일 뿐입니다. 더 고약한 얼굴은 **완전히 정상으로 보이는 래치**입니다:

```
discrete_open_at 이 너무 낮게 잡혀 있고, 방아쇠가 0.243에서 쉬고 있다
  -> 0.243 은 어느 임계값도 넘지 않는다
  -> 래치는 직전 값 CLOSED 를 그대로 유지하고, 토픽은 계속 CLOSED 를 발행한다
  -> 조작자가 손을 펴도 그리퍼는 닫힌 채이고, 표시등은 멀쩡한 CLOSED 다
```

**"쥐고 있어서 CLOSED"와 "방아쇠가 열림 임계값에 더는 닿지 못해서 CLOSED"를 래치 단어만으로는 구분할 수 없습니다.** 실제 탐지 수단은 셋입니다:

1. **표시등의 라이브 방아쇠값** — `trig 0.24`인데 손은 펴져 있다면 그 자리에서 끝납니다.
2. **표시등의 `in band` 깃발** — 방아쇠가 밴드 안에서 1.5초 넘게 머무르면 호박색으로 바뀝니다. 밴드 안에 머무는 것 자체가 "아무것도 넘고 있지 않다"는 신호입니다.
3. **브릿지의 스로틀된 경고 로그** — 같은 1.5초 조건으로 브릿지가 직접 경고를 냅니다(GUI를 안 띄웠거나 사후에 로그를 볼 때의 경로).

래치가 `UNKNOWN`에 멈춰 있거나 방아쇠를 끝까지 당겨도 `OPEN`↔`CLOSED`가 바뀌지 않는 것은 **여전히 유효한 신호이지만, 유일한 신호가 아니며 위 시나리오는 잡지 못합니다.**

⚠️ 임계값이 위 계약을 위반하면 브릿지는 **죽지 않고 연속 모드로 폴백**합니다(팔 텔레오퍼 전체를 그리퍼 파라미터 오타 하나로 내리지 않기 위한 선택입니다). 그때는 기동 로그에 오류가 크게 남고 `discrete_state`는 `DISABLED`가 됩니다 — **`discrete`로 띄웠는데 `DISABLED`가 보이면 임계값이 거부된 것입니다.**

🔢 **임계값 0.3/0.7은 이제 세 군데에 있습니다** — 오프라인 데모 변환기(`recorded_demo.py`), HIL 개입 경로(`wrappers.py`), 그리고 이 브릿지. 앞의 둘은 코드 상수이고 브릿지만 런타임 파라미터입니다. 즉 **브릿지 파라미터를 옮기면 canonical 데모 세트를 만들어 낸 변환기와 값이 어긋납니다** — 조작자가 보는 이산화 경계와 학습 데이터의 이산화 경계가 달라지는 것이므로, 브릿지에서 임계값을 바꿀 일이 생기면 나머지 두 곳도 같이 봐야 합니다.

### 8-5. 녹화에는 무엇이 남는가 — **명령만 바뀝니다** (가장 흔한 오해)

DISCRETE 모드가 이산화하는 것은 **브릿지가 내보내는 명령 하나뿐**입니다. 녹화 take의 HDF5 `synchronized` 테이블(컬럼 정의는 [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) §3) 기준으로:

| 컬럼 | discrete 모드에서 | 왜 |
| --- | --- | --- |
| `grip_cmd` | **이진 (0.0 / 1.0)** | 브릿지 출력 = `/robotiq_gripper/command_percent`. **이산화되는 유일한 채널입니다.** |
| `gello_grip` | **연속 그대로** | 리더 방아쇠 원신호. 브릿지 **상류**라 이 기능이 닿지 않습니다. |
| `grip_pos` | **연속 그대로** | 그리퍼 **실측 위치**. 명령이 아니라 측정값입니다. |

- **아무것도 잃지 않습니다.** `gello_grip`이 원본 연속 신호를 그대로 보존하므로, 이산화는 **원시 컬럼에서 언제든 되돌리거나 다시 계산할 수 있습니다.**
- **`grip_pos`는 애초에 1.0에 도달하지 않습니다.** 물체를 쥐면 손가락이 거기서 멈추기 때문이고, 실측 최댓값은 **0.506**이었습니다(빈 손 완전 닫힘조차 ~0.90 = POS 229에서 안착합니다 — §6 트러블슈팅 표의 "90% 근처에서 멈춤" 항목과 같은 현상입니다). 즉 이 연속 신호는 **"잡았다"와 "헛잡았다"를 구분할 수 있는 유일한 채널**이므로 **이산화하거나 버려서는 안 됩니다.** 이 모드는 그 채널을 건드리지 않습니다.

### 8-6. 트러블슈팅 — "그리퍼가 안 열린다"

| 증상 | 먼저 볼 것 |
| --- | --- |
| 방아쇠를 놓았는데 그리퍼가 **끝까지 안 열린다** (연속 모드) | 🔴 **원인이 둘이고 이 모드는 그중 하나만 고칩니다.** 먼저 표시등의 **`trig` 값**을 보세요. **`trig`가 0.2대에 떠 있으면** 방아쇠(스프링) 문제이고 이 모드가 고칩니다 — §8-2로 켜세요. **`trig`가 0.00인데 그리퍼만 덜 열려 있으면 전혀 다른 버그입니다** — 명령이 전달 경로에서 유실된 것이고, §9를 보세요. |
| discrete로 켰는데 **그리퍼가 아무 반응이 없다** | ① 표시등의 **`trig` 값**을 먼저 보세요 — 손을 폈는데 `trig`가 0.2대에 머물러 있거나 `in band`가 떠 있으면 **방아쇠가 임계값에 닿지 못하는 것**입니다(래치 단어는 `CLOSED`로 멀쩡해 보일 수 있습니다). GUI가 없으면 `ros2 topic echo /gello_gripper_bridge/discrete_trigger`와 브릿지의 스로틀된 경고 로그가 같은 것을 말해 줍니다. ② `discrete_state`가 **`UNKNOWN`에 멈춰 있는가** → 기동 후 아직 한 번도 임계값을 못 넘은 것. ③ **`DISABLED`인가** → 임계값이 거부되어 폴백한 것(기동 로그 확인). ④ `ros2 param get /gello_gripper_bridge discrete_close_at` / `discrete_open_at`로 **실효값** 확인 — 기본 0.7 / 0.3. |
| 방아쇠 중간에서 **출력이 떨린다** | 두 임계값이 서로 가까워져 **히스테리시스 밴드가 좁아진** 경우입니다. 기본 0.3 / 0.7로 되돌리세요. |
| `state`는 `FOLLOWING`인데 그리퍼가 안 움직인다 | 두 토픽은 **별개**입니다(§8-4). `FOLLOWING`은 "브릿지가 흐르고 있다"일 뿐이고, 래치가 `UNKNOWN`이면 흐를 것이 없습니다. |

> ⚠️ **이산 모드와 무관하지만 똑같아 보이는 함정 — 일시정지 뒤에는 그리퍼를 팔과 별도로 재개해야 합니다.** 새 task recorder의 `GO HOME`(그리고 모든 teleop 일시정지)은 팔 브릿지와 **그리퍼 브릿지를 둘 다** PAUSE합니다. 그런데 **EEF GUI의 `ENGAGE`는 팔만 재개합니다** — 그리퍼는 계속 멈춰 있으므로 `Gripper Resume`을 따로 눌러야 합니다. 그리고 재개는 실측 위치에서 시드해 **약 2초에 걸쳐 라이브 방아쇠값으로 램프**합니다(그동안 `discrete_state`는 `RAMPING`).
>
> 🔄 **일시정지–재개 주기는 이제 래치를 `UNKNOWN`으로 되돌립니다(동결이 아니라).** 예전에는 일시정지 동안 마지막 래치가 그대로 남아 있어서, 방아쇠를 쥔 채 `Gripper Resume`을 누르면 그 기억된 `CLOSED`가 곧바로 전(全)스트로크 닫기로 실행됐습니다. 지금은 재개 직후 브릿지가 **임계값을 다시 한 번 넘을 때까지 아무것도 발행하지 않고**, Robotiq은 **현재 위치를 그대로 유지**합니다. 조작자에게 보이는 차이는 이것입니다 — **`GO HOME` 뒤에 방아쇠가 밴드 안에 있으면 그리퍼는 닫히지 않고 열린 채로 있습니다.** (`discrete_state`는 잠깐 `UNKNOWN`으로 보입니다. 정상입니다.)
>
> **그래서 "누르기 전에 GELLO 방아쇠를 열린 상태로 잡고 있으라"는 조언은 여전히 좋은 습관이지만, 이제 조작자와 전력 닫힘 사이에 서 있는 유일한 것이 아닙니다.** 잊고 눌러도 예전처럼 곧바로 닫히지는 않습니다. 다만 방아쇠를 쥐고 있다가 `discrete_close_at`을 **넘겨 버리면** 그때는 정상적으로 닫히므로, 습관 자체를 버리라는 뜻은 아닙니다.

---

## 9. 🔴 "명령은 0.00인데 그리퍼가 덜 열린다" — 셋포인트 유실 (2026-08-06 발견·수정)

**이것은 §8의 이산 모드가 고치는 문제가 아닙니다.** 방아쇠는 멀쩡히 `0.00`을 내는데
그리퍼만 중간에 서 있으면 여기입니다. 이산 모드로는 **절대 안 고쳐집니다** — 오히려
이산 모드는 발행 횟수가 더 적어서 같은 위험에 그대로 노출됩니다.

### 9-1. 증상과 실측

실기 재현(2026-08-06, 증거는 `ros2_ur_ws/gello_logs/diag_gripper_halfopen_20260806_173307/`
— rosbag + 실험 로그 9개):

| 관측 | 값 |
| --- | --- |
| 얼어붙은 순간의 상태 | 입력 trigger **0.0**, 브릿지 **FOLLOWING**, `command_percent` **4초간 발행 0건**, 실제 위치 **0.4196** |
| 닫기/열기 14회 반복 | **7회 실패** (0.0431 / 0.0549 / 0.1098에서 정지. 완전 열림은 **0.0118** = 3/255) |
| 마지막 셋포인트를 1초간 재발행 | **6회 중 0회 실패** |

로봇이 아니라 **소프트웨어**입니다. 하드웨어는 명령만 도착하면 매번 끝까지 엽니다.

### 9-2. 원인 — rate limiter가 가장 **최신** 값을 버렸다

세 가지가 겹칩니다.

1. **브릿지는 값이 바뀔 때 딱 한 번만 발행합니다.** 타이머가 없습니다 —
   `publish_rate_hz` 파라미터는 선언만 되고 코드에서 읽히지 않는 죽은 값이었습니다.
2. **드라이버는 직전 채택 후 50 ms(`command_rate_hz: 20`) 안에 온 셋포인트를 버렸습니다.**
   30 Hz 스트림은 ~20 Hz로만 채택되므로, 방아쇠를 놓는 동작의 **마지막 값** — 바로 그
   "완전 열림" — 이 그 창에 떨어질 확률이 절반쯤 됩니다.
3. **아무도 다시 보내지 않습니다.** rate limit으로 거부된 셋포인트는 드라이버의
   `_last_cmd_pct` 캐시를 갱신하지 않으므로, **드라이버 입장에서는 모든 것이 일관됩니다** —
   자기가 마지막에 쓴 값에 그리퍼가 정확히 가 있으니까요. 어긋난 것은 발행자의 의도뿐이고
   그것은 다시 말해지지 않습니다.

> 🪤 **그래서 드라이버에 "명령 vs 실제 위치" 워치독을 달아도 이 버그는 안 잡힙니다.**
> 드라이버가 보기엔 불일치가 **없습니다.** 이 함정을 한 번 밟았으니 적어 둡니다.

**닫힘이 멀쩡해 보였던 이유**: 손가락이 어차피 물리적 한계(~0.898)에서 멈추므로, 닫는
명령의 마지막 몇 %가 유실돼도 **결과가 똑같습니다.** 여는 쪽에는 그런 정지점이 없어서
유실된 만큼 그대로 벌어진 채 남습니다. 같은 확률로 씹히는데 **한쪽만 눈에 보였던** 것입니다.

### 9-3. 수정

| 위치 | 무엇을 |
| --- | --- |
| `robotiq_gripper_modbus_node.py` | **rate limiter가 버리지 않고 합칩니다(coalesce).** 창이 닫혀 있으면 최신 셋포인트를 보류했다가 창이 열릴 때 발행합니다. rate limiter가 최신 샘플을 버리는 것 자체가 버그였습니다. **여기 하나로 텔레옵·GO HOME·RL·정책 배포가 전부 고쳐집니다.** 재연결 시 명령 캐시도 무효화합니다(`activate()`의 auto-cal 스윕이 손가락을 옮겨 놓기 때문). |
| `gello_gripper_bridge_node.py` | 방어 2선. 출력이 바뀐 뒤 **짧은 창 동안 같은 값을 재발행**하고 조용해집니다(실기 0/6 검증). resume이 발행 없이 `_last_pub`만 세우던 유령 시드도 고쳤습니다. |
| 원샷 발행자들 | 한 번만 publish하면 DDS discovery에서 유실될 수 있습니다(실측: `ros2 topic pub -1`이 7회 중 3회 유실). 짧은 창 동안 반복 발행하고, 피드백이 있는 곳은 확인합니다. |

⚠️ **닫는 셋포인트는 손가락이 덜 닫혔다는 이유로 절대 재명령하지 않습니다.** 그건 실패가
아니라 **물체를 잡은 것**입니다(`grip_pos`는 물체를 쥐면 최대 0.506에서 멈춥니다 —
`GELLO_UR7E_UNITS_REFERENCE.md` §5). 재발행 로직이 그리퍼를 더 세게 조이거나 진동시키면
안 됩니다.

✅ **실기 확인 (2026-08-07, `499707a`).** 조작자가 실제 UR7e + 2F-85에서 EEF 텔레옵으로
확인했다 — **정상 동작.** 수정 직전 같은 조작이 절반 확률로 중간에 멈췄던 것이
(§9-1) 재현되지 않았다.
🟡 **검증 수준**: 조작자의 실기 확인이고, §9-1처럼 **사이클 수를 센 통계는 다시 재지
않았다.** 이 결함은 확률적이므로(14회 중 7회) 재발이 의심되면 세는 것부터 한다 —
`ros2 topic echo /robotiq_gripper/position_percent`로 방아쇠를 빠르게 쥐었다 놓기를
10회 이상 반복하고 매번 **0.0118**에 닿는지 본다.

### 9-4. 되돌리기 / 끄기 — **"전부 끌 수 있다"가 아닙니다**

| 무엇 | 끄는 법 |
| --- | --- |
| 브릿지 settle 재발행 | `settle_reassert_s=0` (또는 `settle_reassert_hz=0`) |
| 브릿지 reconcile | `reconcile_after_s=0` (또는 `reconcile_tol=0`) |
| 드라이버 coalescing | `command_rate_hz=0`이면 애초에 rate limit이 없어 자동 비활성 |
| **resume force-publish** | **끌 수 없습니다 — 일부러 노브를 만들지 않았습니다** |

위 네 개를 0으로 두면 그 로직들은 정확히 예전 동작으로 돌아갑니다.
**단 resume force-publish는 예외입니다.** `_on_resume`이 발행 없이 `_last_pub = seed`만
세우던 옛 동작은, 리더가 이미 `seed`의 `deadband` 안에 있으면 **브릿지가 영원히 침묵하고
그리퍼가 아무 명령도 못 받는** 조용한 행업입니다. 되돌릴 가치가 있는 동작이 아니라서
노브를 두지 않았습니다.

> 🪤 **이 표는 한 번 틀리게 적혔다가 적대적 검수에서 잡혔습니다.** "새 동작은 전부
> 파라미터이고 0으로 두면 정확히 예전 동작"이라고 썼는데, resume force-publish에는
> 파라미터가 아예 없었습니다. 지금은 테스트가 이 사실 자체를 고정합니다
> (`test_the_resume_force_publish_has_no_off_switch`).

---

**안전 재확인**: 이 그리퍼 경로는 GELLO/팔과 독립이지만, 리포의 안전 불변식은 그대로 유지됩니다 — **GELLO는 언제나 passive read-only이며, Dynamixel에 절대 토크를 인가하지 않습니다.** 그리퍼 작업 시에도 로봇이 물리적으로 움직일 수 있으니(전원 ON 상태), E-STOP을 손 닿는 곳에 두고 tool 주변에 손가락/물체가 끼지 않도록 작업공간을 확보하세요.

---

## 관련 문서

- [`GELLO_UR7E_ROS2_BRINGUP.md`](./GELLO_UR7E_ROS2_BRINGUP.md) — GELLO → UR7e 관절 텔레오퍼(sim/mock, 검증됨).
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실물 UR7e + 실 GELLO 구동 런북.
- [`../../UR7e_Remote_Control_ROS2.md`](../../UR7e_Remote_Control_ROS2.md) — UR7e Remote Control(헤드리스/대시보드) 레퍼런스.
- [`GELLO_UR7E_UNITS_REFERENCE.md`](./GELLO_UR7E_UNITS_REFERENCE.md) — 계층별 그리퍼 단위·규약 대조표(**§5**)와 이산화 임계값 0.3/0.7의 실측 근거(**§5.1**). §8의 숫자는 전부 여기서 옵니다.
