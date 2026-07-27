# 02 — GELLO 리더 (Dynamixel) 검증  ✅ PASS

**상태: 실기 검증 완료.** 모터 응답, 발행 주기, 드롭, 트리거 스팬 전부 통과.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## 1. 하드웨어 계약 (verified)

| 항목 | 값 | 근거 |
|---|---|---|
| 포트 | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` | `config/ur7e_gello.yaml:9`, 노드 기본값 `gello_publisher_node.py:71-74` |
| Baud | **57600** | `gello/dynamixel/driver.py:166` (기본값), `docs/ros2/GELLO_ROS2_CONTROL_REFERENCE.md:46` |
| 프로토콜 | Dynamixel **2.0** | `driver.py:251` |
| 팔 모터 ID | 1, 2, 3, 4, 5, 6 | `ur7e_gello.yaml:10` |
| 그리퍼(트리거) ID | **7** | `ur7e_gello.yaml:24` (`gripper_config[0]`) |
| 토크 | **항상 OFF (passive read-only)** | `driver.py:190`(`_torque_enabled = False`), `:289`, `gello_publisher_node.py:129` 주석 |
| 발행 주기 | 30.0 Hz | `ur7e_gello.yaml:27`, `gello_publisher_node.py:84` |

> ### 🔒 불변식: GELLO에 토크/파워를 인가하지 않는다
> 드라이버는 초기화 시 모든 서보의 토크를 **비활성화**하고(`driver.py:288-289`),
> 발행 노드는 읽기만 한다 (`gello_publisher_node.py:165` — `# Read-only: never enables torque`).
> XL330은 5 V 서보다. **7 V 초과 금지** (`docs/sim/GELLO_UR_SIM_TELEOP.md:50`).
> XL330은 읽기조차 별도 5 V 전원이 필요하다 — **전원 ≠ 토크**
> (`docs/sim/GELLO_UR_SIM_TELEOP.md:274`).

---

## 2. 실측 결과 (PASS)

| 판정 항목 | 기대 | **실측** | 판정 |
|---|---|---|---|
| 모터 스캔 (baud 57600) | ID 1~7 전부 응답 | **ID1~6 = model 1200, ID7 = model 1190. 7/7 응답** | PASS |
| `/gello/joint_states` 발행 주기 | 30 Hz | **30.004 Hz (std 0.15 ms)** | PASS |
| 30초 샘플 수 | ≈ 900 | **901** | PASS |
| 드롭 | 0 | **0** | PASS |
| `warning, comm failed` 로그 | 0회 | **0회** | PASS |
| 트리거 스팬 | 0 → 1 전 구간 도달 | **0.000 ~ 1.000** | PASS |

- model **1200** = XL330-M288, model **1190** = XL330-M077 (Dynamixel 모델 번호).
  즉 팔 6축은 M288, 트리거는 M077이다.
- `warning, comm failed: <code>`는 `driver.py:474`에서 나온다. 이게 뜨면 물리 배선/전원/baud 문제다.

---

## 3. Dynamixel 진단 스캔 방법

**언제 쓰나:** 노드가 `FATAL: could not connect to GELLO on port ...`
(`gello_publisher_node.py:135-141`)로 죽었을 때, 또는 특정 관절만 안 움직일 때.

### 3.1 물리 계층 먼저

```bash
# 포트가 보이는가
ls -l /dev/serial/by-id/ | grep FTDI

# 다른 프로세스가 잡고 있는가 (드라이버가 자동으로 죽이려 시도하지만 확인은 직접)
sudo fuser -v /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0

# dialout 권한
id -nG | tr ' ' '\n' | grep -x dialout
```

> `driver.py:222-226`에 포트 점유 확인(`_check_port_availability`)과 점유 프로세스 kill
> 시도(`_kill_processes_using_port`)가 들어 있다. 그래도 안 되면 위 명령으로 수동 확인.

### 3.2 ID/모델 스캔 (읽기 전용)

**⚠️ 이 스크립트는 GELLO 시리얼 포트를 점유한다. `gello_publisher`가 떠 있으면
먼저 내려야 한다** (두 프로세스가 같은 FTDI를 못 쓴다).

`$WT/../scan.py` 같은 임시 위치에 저장하고 실행한다 (리포에 커밋하지 말 것):

```python
#!/usr/bin/env python3
"""GELLO Dynamixel 진단 스캔 — 읽기 전용. 토크를 절대 켜지 않는다."""
from dynamixel_sdk import PortHandler, PacketHandler

PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0"
BAUD = 57600
ADDR_PRESENT_POSITION = 132   # driver.py:24
ADDR_TORQUE_ENABLE = 64       # driver.py:21

ph, pk = PortHandler(PORT), PacketHandler(2.0)
assert ph.openPort(), "openPort failed"
assert ph.setBaudRate(BAUD), "setBaudRate failed"

for dxl_id in range(1, 9):
    model, comm, err = pk.ping(ph, dxl_id)
    if comm != 0:
        continue
    pos, c2, _ = pk.read4ByteTxRx(ph, dxl_id, ADDR_PRESENT_POSITION)
    trq, c3, _ = pk.read1ByteTxRx(ph, dxl_id, ADDR_TORQUE_ENABLE)
    print(f"ID {dxl_id}: model={model} err={err} pos={pos if c2==0 else 'ERR'} "
          f"torque_enable={trq if c3==0 else 'ERR'}")
ph.closePort()
```

```bash
python3 /tmp/gello_scan.py
```

기대 출력(검증된 정상 상태):

```
ID 1: model=1200 err=0 pos=...  torque_enable=0
ID 2: model=1200 err=0 pos=...  torque_enable=0
ID 3: model=1200 err=0 pos=...  torque_enable=0
ID 4: model=1200 err=0 pos=...  torque_enable=0
ID 5: model=1200 err=0 pos=...  torque_enable=0
ID 6: model=1200 err=0 pos=...  torque_enable=0
ID 7: model=1190 err=0 pos=...  torque_enable=0
```

판정:

- **0개 응답** → 전원 미인가가 압도적으로 흔하다. XL330은 읽기에도 5 V가 필요하다.
  다음으로 baud 불일치, 포트 점유.
- **일부만 응답** → 해당 ID의 데이지체인 케이블/커넥터. 응답이 끊기는 첫 번째 ID의
  **바로 앞 케이블**을 의심한다.
- **`torque_enable`이 0이 아님** → 즉시 조사. 이 리포의 어떤 경로도 GELLO 토크를 켜지 않는다.
  이전 세션의 다른 툴이 켜 놓았을 수 있다.
- **model이 1200/1190이 아님** → 하드웨어 구성이 문서와 다르다. 에스컬레이션.

---

## 4. 발행 검증 (완료, 재현 절차)

### 4.1 리더만 띄우기 (로봇 브리지 없음, 팔 미동작)

```bash
source /opt/ros/humble/setup.bash
source $WT/ros2_ur_ws/install/setup.bash

GELLO_REPO_ROOT=$WT \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $WT/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

> `GELLO_REPO_ROOT` 없이는 `No module named 'gello'`로 죽는다 → `00_SETUP_AND_SAFETY.md` §3.1.
> **params-file은 `src/`가 아니라 `install/` 아래 것을 쓴다** — yaml을 고쳤으면 재빌드해야
> `install/`에 반영된다.

정상 로그:

```
Building GELLO config: port=/dev/serial/by-id/..., joint_ids=[1, 2, 3, 4, 5, 6], joint_signs=[...], gripper_config=(7, 210.6..., 168.8...)
gello_publisher started, publishing at 30.0 Hz.
```

### 4.2 판정 명령

```bash
# 주기
ros2 topic hz /gello/joint_states                                   # 기대 ~30.0
ros2 topic hz /gripper/gripper_client/target_gripper_width_percent  # 기대 ~30.0

# 내용
ros2 topic echo --once /gello/joint_states
ros2 topic echo /gripper/gripper_client/target_gripper_width_percent
```

### 4.3 트리거 스팬 확인

트리거를 완전히 풀었다가 완전히 쥐면서 값이 **0.000 → 1.000 전 구간**을 지나는지 본다.
전 구간이 안 나오면 `gripper_config`의 `open_deg`/`close_deg` 캘리브레이션이 틀린 것이다
(`ur7e_gello.yaml:24` = `[7.0, 210.649609375, 168.849609375]`).

---

## 5. 🛑 `/gello/joint_states`의 `position` 길이는 **6**이다

```python
js_msg.position = js[:6].tolist()      # gello_publisher_node.py:190
...
gripper_msg.data = float(js[6])        # :194  ← 트리거는 별도 토픽으로 나간다
self._gripper_pub.publish(gripper_msg)
```

즉:

| 데이터 | 토픽 | 타입 |
|---|---|---|
| 팔 6축 | `/gello/joint_states` | `sensor_msgs/JointState`, `position` 길이 **6** |
| 트리거 | `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` |

**이것이 HIL 개입의 그리퍼 데드코드 원인이었다.** RL 백엔드가 `/gello/joint_states`만 구독해서
`GelloExpert.get_leader()`의 `grip = float(arr[6]) if len(arr) > 6 else None`이
**항상 `None`**이었다.

> ⚠️ 흔한 오해 정정: "publisher가 트리거를 버린다"가 아니다. 트리거는 **다른 토픽으로
> 정상 발행되고 있다.** 문제는 RL 백엔드가 그 토픽을 구독하지 않은 것이었다.

### 5.1 수정됨 (2026-07-27, 커밋 안 됨, 하드웨어 미검증)

`URRosBackend`가 트리거 토픽을 **추가로 구독**하고 두 스트림을 7-요소로 합친다:

| | |
|---|---|
| 트리거 토픽 상수 | `GELLO_TRIGGER_TOPIC = "/gripper/gripper_client/target_gripper_width_percent"` (`ros_backend.py:68`) |
| 병합 함수 | `merge_gello_state(q, age_q, trigger, age_trigger, stale_s)` (`ros_backend.py:77-124`) |
| 트리거 스테일 임계 | `GELLO_TRIGGER_STALE_S = 0.3` (`ros_backend.py:74`) |
| **트리거 부재 표현** | **`NaN`** — `0.0`은 "완전 열림"이라는 **정당한 값**이므로 센티널로 쓰면 토픽이 죽을 때마다 그리퍼를 조용히 열어버린다 (`ros_backend.py:91-96`) |
| 반환 age | **관절 age만.** 트리거가 없다고 팔 텔레옵까지 막지 않기 위해 (`:97-101`) |
| 레거시 폴백 | 트리거 토픽이 **한 번도** 안 왔고 JointState에 7번째가 있으면 그걸 쓴다. 토픽이 말한 뒤 스테일된 경우엔 **폴백하지 않는다** (실신호 장애를 가리지 않기 위해, `:102-109`) |
| 원시 접근 | `URRosBackend.get_gello_trigger()` → `(0..1 or None, age)` (`ros_backend.py:306-309`) |

**중요:** `/gello/joint_states`의 계약(`position` 길이 6)은 **일부러 그대로 뒀다**.
기존 소비자(`gello_ur_bridge`, `gello_gripper_bridge`, 레코더, GUI)를 건드리지 않는
쪽이 위험이 낮기 때문이다 (`ros_backend.py:162-166`).

회귀 테스트 (rclpy·시리얼 불필요):

```bash
cd $WT/serl_ur_infra
python3 -m pytest tests/test_gello_gripper_wiring.py -q -p no:anyio
```

**하드웨어에서는 아직 확인되지 않았다.** 실기 판정은 `04_HIL_INTERVENTION.md` §6.

---

## 6. 실패 모드

| 증상 | 원인 | 확인/복구 |
|---|---|---|
| `FATAL: could not connect to GELLO on port ...` | 미연결 / 전원 없음 / 포트 점유 / dialout 권한 | §3.1 |
| `warning, comm failed: <code>` 반복 | 배선·전원·baud | §3.2 스캔 |
| `Unexpected joint state length (got N, expected 7); skipping cycle.` | 모터 하나가 응답 안 함 → 사이클 스킵. **노드는 안 죽고 토픽만 끊긴다** | `gello_publisher_node.py:174-181`. §3.2로 어느 ID인지 특정 |
| `get_joint_state() failed, skipping cycle: ...` | 일시적 읽기 글리치. 2초 throttle 로그 | `:169-173` |
| 토픽은 나오는데 값이 이상 | `joint_offsets` / `joint_signs` 캘리브레이션 | `ur7e_gello.yaml:11-22`. `scripts/gello_get_offset.py` |

> **중요:** 위 두 skip 경로 모두 **노드를 죽이지 않는다.** 토픽이 조용히 멈출 뿐이다.
> 그래서 "GELLO가 끊겼다"는 프로세스 존재 여부가 아니라 **`ros2 topic hz`로** 확인해야 한다.
> 장애 주입 시나리오 E1 참조 (`07_FAILURE_INJECTION.md`).

---

## 7. 완료 판정 기준 (전부 충족됨)

- [x] ID 1~7 전부 응답, model 1200×6 + 1190×1
- [x] 전 ID `torque_enable = 0`
- [x] `/gello/joint_states` 30.004 Hz, std 0.15 ms
- [x] 30초 901샘플, 드롭 0, `comm failed` 0회
- [x] 트리거 0.000~1.000 전 스팬
