# GELLO → UR5e ROS2 텔레오퍼레이션 (로컬 · RViz2 · 검증됨)

이 문서는 **native ROS2 Jazzy(도커 없음)** 환경에서 GELLO 리더 암으로 UR5e를 텔레오퍼레이션하고, 그 결과를 **RViz2**에서 시각화하는 검증된 경로를 다룹니다. 사용하는 패키지는 `ros2_ur_ws/src/ur_gello_bringup` 입니다.

> **왜 도커가 아닌가**: UR 드라이버(`ur_robot_driver`)는 최신 Ubuntu 24.04 / ROS2 Jazzy에서 안정적으로 동작합니다. 도커로 감쌀 이유가 없어 native가 더 단순하고 신뢰성이 높습니다. (반대로 Franka 계열은 버전 락 때문에 도커가 필요합니다 — 별도 문서 참고.)

> **이 경로가 검증하는 것**: 이 PC에서는 **mock hardware(가짜 UR)** 로 RViz2 시각화만 확인합니다. 실제 UR5e는 **별도의 PC**에 연결되어 있으며, 실로봇 구동 절차는 [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md)에 정리되어 있습니다.

---

## 안전 — GELLO는 항상 passive read-only

GELLO는 **수동(passive) 모션캡처 리더 암**입니다. Dynamixel 모터에 **절대 토크를 걸지 않습니다.**

- `gello_publisher_node.py`는 `DynamixelRobotConfig.make_robot`으로 드라이버를 초기화하는데, 이 드라이버는 **토크를 OFF 상태로 초기화**하고 관절 각도를 **읽기만** 합니다.
- 노드 종료(`main`의 `finally`)에서도 "GELLO is passive; nothing to power down"이라고 명시되어 있습니다 — 끌 전원 자체가 없습니다.

즉, 이 경로 어디에서도 GELLO 모터에 힘이 들어가지 않습니다. GELLO는 손으로 자유롭게 움직이는 입력 장치입니다.

---

## 1. Overview — 무엇이 실행되는가

`ur_gello_rviz.launch.py`는 다음을 순서대로 띄웁니다.

```
[GELLO source]                [bridge]                       [mock UR]                 [RViz2]
gello_publisher  ─(30Hz)─▶  gello_ur_bridge  ─(125Hz)─▶  forward_position_    ─▶  /joint_states ─▶ RViz2
  또는 fake_gello              EMA + step-clamp             controller (mock)         (UR5e 모델이 움직임)
                              + staleness watchdog
gello_publisher ─▶ /gripper/.../target_gripper_width_percent ─▶ robotiq_urcap (기본: 로그만)
```

### 노드별 역할

| 노드 (executable) | 역할 |
| --- | --- |
| `gello_publisher` | 물리 GELLO를 시리얼로 읽어 6개 관절을 UR 순서의 `sensor_msgs/JointState`로 `/gello/joint_states`에 발행(기본 30Hz). 그리퍼 폭은 `std_msgs/Float32`(0..1)로 `/gripper/gripper_client/target_gripper_width_percent`에 발행. **읽기 전용, 토크 OFF.** |
| `gello_ur_bridge` | `/gello/joint_states`를 구독해 **이름 기준으로 UR 관절 순서에 재정렬**한 뒤, EMA 스무딩 + step-clamp(슬루 제한) + staleness watchdog을 적용하여 `/forward_position_controller/commands`(`std_msgs/Float64MultiArray`, 6 doubles)로 **125Hz** 상향 샘플링해 발행. |
| `fake_gello` | 하드웨어 없이 파이프라인을 시험하는 **테스트 전용** 노드. 실제 `gello_publisher`와 동일한 토픽에 느린 sine sweep을 발행. GELLO/UR 실물이 전혀 필요 없음. |
| `robotiq_urcap` | Robotiq 2F-85 그리퍼를 UR URCap 소켓(기본 포트 63352)으로 구동. **이 PC에는 그리퍼가 없으므로 기본 `connect_on_start=False`** — 소켓을 열지 않고 "이 위치로 움직였을 것"만 로그로 남김. 실 그리퍼는 로봇 PC에서 `connect_on_start:=true`로 실행. |

### bridge의 3가지 안전장치 (`gello_ur_bridge_node.py`)

`forward_position_controller`는 보간을 하지 않고 **보낸 위치를 즉시 명령**하므로, bridge가 다음을 담당합니다.

- **EMA 스무딩** (`ema_alpha`): 관절별 저역 통과 필터로 노이즈/떨림 완화.
- **MAX-STEP CLAMP** (`max_step_rad`): 매 발행 사이클당 명령이 움직일 수 있는 최대 각도를 제한(슬루 제한). 단, 이 clamp는 **첫 GELLO 포즈로의 초기 접근을 램프업하지 않습니다.** bridge는 첫 유효 GELLO 메시지에서 필터와 last-published 명령을 **GELLO 포즈 전체 값으로 seed한 뒤 그대로 발행**합니다. 즉 mock hardware에서는 RViz 모델이 한 스텝에 GELLO 포즈로 스냅합니다(sim이라 무해). clamp는 그 이후 **실시간 추종 중 사이클당 이동량을 제한하고 스파이크를 걸러내는** 역할만 합니다. (실로봇에서의 초기 점프 방지는 clamp가 아니라 7절의 move-to-start 핸드셰이크가 담당합니다.)
- **STALENESS WATCHDOG** (`staleness_timeout_s`): GELLO 입력이 끊기면(뽑힘/크래시/드라이버 행) 마지막 명령을 반복하지 않고 **발행을 멈춤**. position controller에서는 "스트리밍 중단"이 fail-safe 상태. (판정에 `time.monotonic()`을 써서 wall-clock/`use_sim_time` 변화에 영향받지 않음.)

### 토픽 계약

| 토픽 | 타입 | 비고 |
| --- | --- | --- |
| `/gello/joint_states` | `sensor_msgs/JointState` | 6 관절, UR 순서, radian |
| `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` | 0..1 (0=open, 1=closed) |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 6 doubles, UR 순서 |

UR 관절 순서:
`[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`

---

## 2. Prerequisites

### 2-1. apt 패키지 (ROS2 Jazzy)

```bash
sudo apt install ros-jazzy-ur ros-jazzy-dynamixel-sdk
```

- `ros-jazzy-ur` — `ur_robot_driver` / `ur_description` / `ur_controllers`(launch가 include하는 공식 UR 드라이버).
- `ros-jazzy-dynamixel-sdk` — 시스템 python3.12에서 Dynamixel 접근용.

### 2-2. python `gello` 패키지를 import 가능하게 만들기 (중요)

`gello_publisher` / `robotiq_urcap` 노드는 리포 루트의 python `gello` 패키지를 import합니다. 여기서 **인터프리터가 갈립니다.**

- **시뮬레이션(MuJoCo)** 경로는 uv venv의 **python 3.11**을 씁니다.
- **ROS2 Jazzy**는 **시스템 python 3.12**를 씁니다.

따라서 sim용 venv에 설치해둔 `gello`는 ROS2에서 보이지 않습니다. 다음 중 하나로 시스템 python3.12에서 `gello`를 import 가능하게 만드세요.

**방법 A — 환경변수 (portable, 권장)**

```bash
export GELLO_REPO_ROOT=/home/theo_lab/gello_software
```

노드는 `gello`를 못 찾으면 `GELLO_REPO_ROOT`를 `sys.path`에 추가합니다. 셸 프로파일(`~/.bashrc` 등)에 넣어두면 편합니다.

**방법 B — 시스템 python3.12에 editable 설치**

```bash
# sim용 uv venv가 아니라, 시스템 python3.12로 설치해야 합니다.
pip3 install -e /home/theo_lab/gello_software
```

> **주의**: 노드에는 `/home/theo_lab/gello_software`라는 하드코딩 fallback 경로가 있습니다. 동료의 다른 머신에서는 이 경로가 존재하지 않으므로, **반드시 `GELLO_REPO_ROOT`를 본인 체크아웃 경로로 지정하거나 방법 B로 설치**하세요. 하드코딩 fallback에 의존하지 마세요.

---

## 3. Build

```bash
source /opt/ros/jazzy/setup.bash
cd /home/theo_lab/gello_software/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

빌드 후 매 새 셸에서 `source /opt/ros/jazzy/setup.bash` + `source .../ros2_ur_ws/install/setup.bash` 두 줄을 다시 실행해야 합니다.

---

## 4. Run — 두 가지 모드

두 모드 모두 UR 드라이버를 `use_fake_hardware:=true`(mock hardware)로 include합니다. mock은 보낸 position 명령을 그대로 joint state로 되돌려주므로, `forward_position_controller`에 명령을 보내면 **RViz 모델이 움직입니다.** 실제 UR5e는 전혀 건드리지 않습니다.

> `DISPLAY`는 **본인 세션의 X 디스플레이**로 지정하세요. 일반 모니터가 달린 PC라면 보통 `:0` 입니다.
> (아래 예시의 `:1`은 **이 headless 빌드 PC 전용**의 가상 X 서버라서 그렇습니다 — 그대로 따라 쓰지 마세요.)

### 4-1. Fake 모드 — 하드웨어 없이 RViz 확인

GELLO/UR 실물 없이 파이프라인 전체(sine sweep → bridge → mock UR → RViz)를 검증합니다.

```bash
DISPLAY=:0 ros2 launch ur_gello_bringup ur_gello_rviz.launch.py
# 또는 명시적으로:
DISPLAY=:0 ros2 launch ur_gello_bringup ur_gello_rviz.launch.py source:=fake
```

**무엇이 보여야 하는가**
- RViz2 창이 열리고 UR5e 모델이 표시됩니다.
- 컨트롤러가 spawn되고 약 5초 뒤(launch의 `TimerAction period=5.0`), UR5e가 시작 포즈 주변에서 **천천히 sine으로 흔들립니다.**
- `robotiq_urcap`는 그리퍼가 없으므로 "would move to POS=..." 로그만 남깁니다(정상).

### 4-2. Gello 모드 — 실제 GELLO로 조종 (mock UR)

물리 GELLO를 읽어 mock UR을 조종합니다. GELLO에 **전원(5V)** 이 들어와 있어야 하고, 시리얼 포트/캘리브레이션은 `config/ur_gello.yaml`에서 읽습니다. (**GELLO는 여전히 read-only, 토크 OFF.**)

```bash
DISPLAY=:0 ros2 launch ur_gello_bringup ur_gello_rviz.launch.py source:=gello
```

**무엇이 보여야 하는가**
- RViz2에 UR5e 모델이 뜨고, **GELLO를 손으로 움직이면 RViz의 UR5e가 1:1로 따라 움직입니다.**
- bridge 로그에 `gello_ur_bridge started | ema_alpha=... max_step_rad=... publish_rate_hz=125.0`가 찍힙니다.
- mock UR은 all-zero에서 시작하지만, bridge가 첫 유효 GELLO 메시지에서 필터와 last-published 명령을 GELLO 포즈 전체로 seed해 곧바로 발행하므로, RViz의 UR5e 모델은 한 스텝에 GELLO 포즈로 스냅한 뒤 실시간 추종으로 넘어갑니다(mock hardware라 이 스냅은 무해). 이후 `max_step_rad` clamp는 사이클당 이동량을 제한하며 스파이크를 걸러냅니다.

---

## 5. Config 필드 레퍼런스 (`config/ur_gello.yaml`)

```yaml
gello_publisher:
  ros__parameters:
    port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0"
    joint_ids: [1, 2, 3, 4, 5, 6]
    joint_offsets: [3.142, 1.571, 4.712, 4.712, 4.712, 3.142]  # J2/J3 재장착 후 재캘리브레이션 값
    joint_signs: [1, 1, -1, 1, 1, 1]
    # NOTE: 반드시 ALL doubles — rcl yaml 파서는 int/double 혼합 배열을 거부함.
    # (그리퍼 id 7은 노드 내부에서 다시 int로 round-trip 됨.)
    gripper_config: [7.0, 210.649609375, 168.849609375]
    start_joints: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]
    publish_rate_hz: 30.0

gello_ur_bridge:
  ros__parameters:
    ema_alpha: 0.5
    max_step_rad: 0.05
    staleness_timeout_s: 0.5
    publish_rate_hz: 125.0
```

### `gello_publisher`

| 필드 | 의미 |
| --- | --- |
| `port` | GELLO FTDI 시리얼 포트. `/dev/serial/by-id/...` 경로는 **replug/reboot에도 안정적**(`/dev/ttyUSBn`은 열거 순서라 바뀜). 다른 어댑터를 쓰면 이 값을 본인 by-id 경로로 교체. baud는 57600 고정. |
| `joint_ids` | Dynamixel 관절 모터 ID 6개. |
| `joint_offsets` | 캘리브레이션 오프셋(rad, π/2의 배수). `scripts/gello_get_offset.py` 출력값을 붙여넣음. |
| `joint_signs` | 관절별 부호(GELLO ↔ UR 회전 방향). |
| `gripper_config` | `[gripper_id, open_deg, close_deg]`. **반드시 doubles** — id도 `7.0`처럼 써야 함(노드가 내부에서 int로 반올림). |
| `start_joints` | 캘리브레이션 기준 시작 포즈(6 관절 + 그리퍼, rad). |
| `publish_rate_hz` | GELLO 발행 주기(기본 30Hz). |

### `gello_ur_bridge`

| 필드 | 의미 |
| --- | --- |
| `ema_alpha` | EMA 저역 통과 계수(0~1). 클수록 새 입력 반영이 빠르고(덜 부드러움), 작을수록 더 매끄럽지만 지연 증가. |
| `max_step_rad` | 한 발행 사이클당 관절 최대 이동각(rad). 급격한 점프/스파이크 억제. |
| `staleness_timeout_s` | 이 시간(초) 넘게 GELLO 입력이 없으면 발행 중단(watchdog). |
| `publish_rate_hz` | bridge 상향 샘플링 발행 주기(기본 125Hz — UR servoj 루프에 매끄럽게 공급). |

> **rcl YAML gotcha**: rcl의 YAML 파서는 **int/double이 섞인 배열을 거부**합니다. 실수(double)가 하나라도 들어가는 배열은 모든 원소를 double로 적으세요. 예: `gripper_config: [7.0, 210.649609375, 168.849609375]` (`7`이 아니라 `7.0`).

---

## 6. Troubleshooting

### 시리얼 포트를 다른 프로세스가 잡고 있음 (orphan port holder)
launch가 죽었는데 `gello_publisher`가 좀비로 남아 포트를 붙들고 있으면, 새 실행에서 GELLO 연결에 실패합니다.

```bash
lsof /dev/ttyUSB1          # 또는 by-id 경로. 포트를 잡은 PID 확인
kill <PID>
```

### `ros2 node list`에 유령 노드 (ghost nodes)
종료된 노드가 목록에 계속 보이면 ROS2 데몬을 재시작합니다.

```bash
ros2 daemon stop && ros2 daemon start
```

### GELLO가 `-3001` 등 이상값을 읽음
GELLO에 **5V 전원**이 들어와 있어야 합니다. 전원이 없으면 Dynamixel 읽기가 실패해 `-3001` 같은 값이 나옵니다. 전원 어댑터/USB 허브 전원을 확인하세요. (전원은 **읽기용**일 뿐, 모터 토크와 무관 — GELLO는 여전히 passive.)

### 시리얼 접근 권한 (dialout)
사용자가 `dialout` 그룹에 없으면 포트 접근이 거부됩니다.

```bash
sudo usermod -aG dialout $USER
# 그런 다음 로그아웃 후 다시 로그인해야 반영됩니다.
```

### `gello`를 import하지 못함 (`ModuleNotFoundError: gello`)
2-2 절을 따르지 않은 경우입니다. **시스템 python3.12** 기준으로 `GELLO_REPO_ROOT`를 export하거나 `pip3 install -e .`를 하세요. sim용 uv venv(python3.11)에 설치한 것은 ROS2에서 보이지 않습니다.

### RViz는 뜨는데 UR5e가 안 움직임
컨트롤러 spawn 전에 명령이 나가지 않도록 launch가 5초 지연(`TimerAction`)을 둡니다. 잠시 기다리세요. 그래도 안 움직이면 `ros2 topic echo /forward_position_controller/commands`로 bridge가 발행 중인지, `ros2 control list_controllers`로 `forward_position_controller`가 active인지 확인하세요.

---

## 7. 실 UR5e로 가려면

이 문서의 경로는 **mock hardware로 RViz2 시각화만** 검증합니다. 실제 UR5e는 **별도 PC**에 연결되어 있습니다.

실로봇 구동은 [`GELLO_UR_ROS2_PLAN.md`](./GELLO_UR_ROS2_PLAN.md)를 따르세요. 핵심 차이는:

- `use_fake_hardware:=false` + 실제 `robot_ip` 지정,
- UR 티치펜던트의 **External Control URCap** 프로그램 실행,
- **필수 move-to-start 핸드셰이크**: `forward_position_controller`는 보간 없이 즉시 명령하므로, 먼저 `scaled_joint_trajectory_controller`로 현재 GELLO 포즈까지 부드럽게 이동시킨 뒤 컨트롤러를 전환하고 bridge를 시작해야 합니다. (이 절차를 건너뛰면 실로봇이 급격히 점프할 수 있습니다.)

실 그리퍼를 쓰려면 로봇 PC에서 `robotiq_urcap` 노드를 `connect_on_start:=true`(및 올바른 `robot_ip`)로 실행합니다.
