# GELLO → Franka Panda MuJoCo 텔레오퍼레이션 셋업

물리 **GELLO** 리더 암으로 **MuJoCo 시뮬레이션 속 Franka Panda**를 텔레오퍼레이션하는 셋업 문서. 실제 Franka 하드웨어 없이 시뮬레이션만으로 동작하며, GELLO는 **passive read-only(토크 미인가)** 로만 사용한다.

- **실행 한 줄**: `DISPLAY=:1 .venv/bin/python experiments/launch_yaml.py --left-config-path configs/rwh_panda.yaml`
- **결과**: 팔 7축 1:1 추종 + 그리퍼 정방향 개폐, GELLO 모터는 끝까지 통전되지 않음.
- **참고**: 동료(조민제)의 가이드 + 도커 이미지 `minje227/gello:RWH_Gello_PandaV1.0` (캘리브레이션 값 재사용).

## 요약 (TL;DR)

| 항목 | 상태 |
|---|---|
| MuJoCo Panda 시뮬 (`panda.xml`, nu=8) | ✅ |
| GELLO 8축 읽기 (XL330, 5V 전원 후) | ✅ torque OFF 확정 |
| 캘리브레이션 | ✅ 동료 값 재사용 (재측정 불필요) |
| 팔 7축 텔레오퍼레이션 | ✅ 라디안 1:1 추종 |
| 그리퍼 개폐 | ✅ `gripper_builtin` + `gripper_invert`로 정상화 |
| 배경 씬 & 큐브 | ✅ `add_scene`(바닥+스카이박스+조명) + `add_cube`(로봇 앞 4cm 큐브), opt-in |
| 안전 (GELLO 토크 미인가) | ✅ init부터 끝까지 passive |

### 변경/추가 파일

| 파일 | 종류 | 내용 |
|---|---|---|
| `gello/dynamixel/driver.py` | 수정 | `set_torque_mode`: 토크 disable 시 XL330 alert 비트(128) 무시 |
| `gello/robots/dynamixel.py` | 수정 | init의 `set_torque_mode(False)`를 try/except로 감쌈 |
| `gello/robots/sim_robot.py` | 수정 | `gripper_builtin` / `gripper_invert` 옵션 + `add_scene` / `add_cube`(배경 씬·큐브) |
| `configs/rwh_panda.yaml` | 신규 | launch_yaml 설정 (동료 캘리브레이션 + 그리퍼 + 씬/큐브 옵션) |
| `scripts/sim_panda_scripted_demo.py` | 신규 | GELLO 없이 sim 검증용 사인 스윕 + GIF 데모 |

### 목차

1. [셋업 & 재현 가이드](#셋업--재현-가이드-setup--runbook)
2. [코드 변경 상세](#코드-변경-상세-code-changes)
3. [하드웨어 & 캘리브레이션](#하드웨어--캘리브레이션-hardware--calibration)
4. [개발 여정 & 트러블슈팅](#개발-여정--트러블슈팅-development-journey)
5. [향후 작업 (실로봇 ROS2)](#향후-작업-실로봇-ros2)

---

## 셋업 & 재현 가이드 (Setup & Runbook)

아래 순서대로 따라하면 GELLO(물리 leader) → Franka Panda MuJoCo 시뮬레이션 텔레오퍼레이션을 그대로 재현할 수 있음. 작업 디렉토리는 전부 `/home/theo_lab/gello_software` 기준이고, 명령어는 그 루트에서 실행한다고 가정함.

### 0) 사전 요건 (Prerequisites)

- **OS / 디스플레이**: Ubuntu, 실제 디스플레이가 `DISPLAY=:1`에 붙어 있음 (인터랙티브 MuJoCo 뷰어는 GLFW 창을 띄우므로 실제 X 디스플레이가 필요함). GL/EGL 라이브러리는 이미 설치돼 있음.
- **uv**: 0.11.8 설치돼 있음. 없으면 먼저 설치:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **하드웨어**: GELLO 본체(Dynamixel XL330 서보 8개), U2D2 USB 인터페이스(FTDI FT232H), **별도 5V 외부 전원**. 자세한 건 3) 참고.
- Python 3.11 venv는 `.venv`에 만들 거고, 이 venv에는 `pip` 모듈이 없음 → 패키지 설치는 무조건 `uv pip`로 한다.

### 1) 서브모듈 초기화 (Submodules)

`panda.xml`은 리포에 없고 `mujoco_menagerie` 서브모듈에서 옴. DynamixelSDK도 서브모듈이라 둘 다 받아야 함.

```bash
cd /home/theo_lab/gello_software
git submodule update --init third_party/mujoco_menagerie third_party/DynamixelSDK
```

체크: `third_party/mujoco_menagerie/franka_emika_panda/panda.xml`이 존재해야 함 (이 모델은 `nu=8`: 팔 7축 + 핸드 액추에이터 1개).

### 2) venv + 의존성 설치 (venv & deps)

```bash
cd /home/theo_lab/gello_software
uv venv --python 3.11
uv pip install -r requirements.txt mujoco
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python
```

주의할 점:
- **`mujoco`는 `requirements.txt`에 없음** → 위처럼 명시적으로 같이 깔아줘야 함.
- `DynamixelSDK/python` 설치하면서 `pyserial`이 같이 딸려옴.
- 검증된 설치 버전: mujoco 3.10.0, numpy 2.3.5, dynamixel-sdk 3.7.51, pyserial 3.5, pyzmq 27.1.0.

### 3) 하드웨어 연결 & 전원 (Hardware / Power)

순서가 중요함:

1. GELLO의 서보 버스에 **외부 5V 전원**을 연결한다. USB만으로는 서보가 어떤 baud에서도 응답하지 않음 — 전원이 반드시 따로 필요함.
   - **경고**: XL330은 5V 서보임. 최대 7V를 넘기지 말 것 (전압 잘못 넣으면 서보 손상).
2. U2D2를 호스트 USB에 연결한다. `lsusb`에서 `0403:6014` (FTDI FT232H)로 보임.
3. 포트 경로(고정):
   ```
   /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0
   ```
   (= `/dev/ttyUSB0`, baud 57600). 이 by-id 경로가 config에 그대로 박혀 있음.

참고로 GELLO는 **read-only(passive leader)로만** 쓰임. 위치를 읽으려면 전원은 필요하지만 토크는 켜지 않음 — DynamixelRobot이 torque OFF 상태로 init하고, 명령은 sim 로봇에만 흐르지 GELLO 모터로는 안 간다.

### 4) 시리얼 권한 (Serial permission)

현재 유저가 `dialout` 그룹에 없음. 둘 중 하나:

```bash
# 방법 A (영구): 그룹 추가 후 재로그인 필요
sudo usermod -aG dialout $USER

# 방법 B (임시): 권한만 열기
sudo chmod 666 /dev/ttyUSB0
```

(선택) 동작을 더 부드럽게 하려면 USB latency timer를 줄여줌:

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### 5) (일회성) 캘리브레이션 — 보통 생략 (One-time calibration)

`configs/rwh_panda.yaml`의 `joint_offsets` / `joint_signs` / `gripper_config`는 **조민제의 Docker 이미지에서 가져온 값을 재사용**함 (동일한 물리 GELLO, 같은 포트 FTBEO75A). **GELLO 조립/서보 마운팅이 바뀌지 않았다면 재캘리브레이션 불필요** → 이 단계는 건너뛰면 됨.

만약 다시 캘리브해야 하면, GELLO를 Panda home 포즈로 잡은 상태에서:

```bash
.venv/bin/python scripts/gello_get_offset.py \
  --start-joints 0 0 0 -1.57 0 1.57 0 \
  --joint-signs 1 -1 1 -1 1 -1 1 \
  --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0
```

출력된 offset 값을 `configs/rwh_panda.yaml`의 `joint_offsets`에 옮겨 적는다.

(선택) read-only 검증: 모터 스캔 시 ID 1–8이 잡혀야 함 (1–7은 model 1200 = XL330-M288 팔 축, ID 8은 model 1190 = XL330-M077 그리퍼). 전부 `torque_enable=0`이어야 정상.

### 6) 실행 (Launch)

단일 명령으로 sim 서버 + GELLO 에이전트 + 뷰어가 한 번에 뜸:

```bash
cd /home/theo_lab/gello_software
DISPLAY=:1 .venv/bin/python experiments/launch_yaml.py \
  --left-config-path configs/rwh_panda.yaml
```

- **`MUJOCO_GL=glx`를 설정하지 말 것** (잘못된 값임). 인터랙티브 뷰어는 GLFW를 쓴다.
- 헤드리스 offscreen 렌더가 필요한 경우에만 `MUJOCO_GL=egl`을 쓴다 (이 런북에서는 불필요).
- 내부 동작: `MujocoRobotServer`가 `127.0.0.1:6001` ZMQ 서버로 뜨고, GELLO `GelloAgent`가 30Hz로 관절값을 읽어 sim에 흘려보냄. config의 `start_joints`(`[0,0,0,-1.57,0,1.57,0, 1.0]`)로 먼저 이동한 뒤 control loop가 돌기 시작함.

### 7) 정상 동작 확인 (What you should see)

- `:1` 디스플레이에 MuJoCo 뷰어 창이 뜨고 Franka Panda가 **체커보드 바닥 + 스카이박스 배경** 위에 보임. 로봇 앞(+x)에 **작은 빨간 큐브**가 바닥에 놓여 있음 (`add_scene`/`add_cube`).
- GELLO를 손으로 움직이면 **시뮬 Panda 팔이 1:1(radian)로 따라옴** (7축).
- GELLO 그리퍼 레버를 당기면 **sim 핸드가 닫히고**, 놓으면 열림 (`gripper_builtin: true` + `gripper_invert: true`로 GELLO `[0,1]` → panda actuator8 `[0,255]`로 리스케일, 255=open이라 invert).
- 터미널에는 `Server ready!`, `Launching robot: MujocoRobotServer, agent: GelloAgent`, `Control loop: 30 Hz` 같은 로그가 찍힘.

### 8) 종료 (Stop)

```bash
# 실행 중인 터미널에서:
Ctrl-C
```

`launch_yaml.py`는 SIGINT/SIGTERM 핸들러로 ZMQ 서버/스레드를 정리함. 혹시 프로세스가 남아 포트(6001)가 안 풀리면:

```bash
pkill -f launch_yaml.py
```

---

## 코드 변경 상세 (Code Changes)

동작에 영향을 주는 변경은 4개 파일(+ 검증용 데모 스크립트 1개)이다. 토크 관련 패치 2개는 GELLO를 안전하게 read-only로 쓰기 위한 것이고 (조민제 docker 커밋 `f67af69`을 미러링, 업스트림에는 없음), 나머지는 panda.xml의 내장 그리퍼를 sim에 연결하고(`sim_robot.py` + `rwh_panda.yaml`) 배경 씬·큐브를 추가하는 변경이다.

### 1. `gello/dynamixel/driver.py` — `set_torque_mode()`: disable 시 alert 비트 무시

XL330 서보는 토크 enable/disable 쓰기를 할 때 alert 비트(`dxl_error == 128`, 0x80)를 올리는 경우가 있다 (예: input-voltage 플래그). 기존 코드는 통신 실패와 servo error를 한 덩어리로 묶어서 검사했기 때문에, 이 alert 비트만 떠도 무조건 `RuntimeError`로 죽었다. 이걸 두 갈래로 쪼개서, **토크를 끄는 방향(`enable=False`)일 때 alert 비트(128)는 무시하고 넘어가도록** 했다. disable은 fail-safe 방향이라 alert가 있어도 토크를 끄는 것 자체는 안전하기 때문.

```diff
                 dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                     self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                 )
-                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
+                if dxl_comm_result != COMM_SUCCESS:
+                    print(dxl_comm_result)
+                    print(dxl_error)
+                    raise RuntimeError(
+                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
+                    )
+                if dxl_error != 0:
+                    # When DISABLING torque, ignore the Dynamixel alert bit (128 / 0x80).
+                    # XL330 servos can raise an alert (e.g. input-voltage flag) on the
+                    # torque write; tolerating it on disable is safe (disable is the
+                    # fail-safe direction) and matches the RWH GELLO hardware fix.
+                    if not enable and dxl_error == 128:
+                        print(
+                            f"Warning: ignoring alert bit while disabling torque "
+                            f"for Dynamixel ID {dxl_id}"
+                        )
+                        continue
                     print(dxl_comm_result)
                     print(dxl_error)
                     raise RuntimeError(
```

포인트:
- 통신 자체 실패(`dxl_comm_result != COMM_SUCCESS`)는 여전히 무조건 `RuntimeError`. 무시하지 않는다.
- alert 비트 무시는 `not enable and dxl_error == 128`일 때만. 즉 **토크를 켜는 방향에서는** alert가 뜨면 그대로 에러로 죽는다 (켜는 건 위험 방향이니까 보수적으로). 다른 종류의 error 코드(128이 아닌 값)도 무시하지 않는다.
- `continue`로 다음 서보 ID로 넘어간다.

### 2. `gello/robots/dynamixel.py` — init의 `set_torque_mode(False)`를 try/except로 감싸기

방어를 한 겹 더 둔 것(defense in depth). driver 레벨에서 disable+alert는 이미 무시되지만, init 단계에서 토크를 끄려다 그래도 `RuntimeError`가 나면 그걸 잡아서 경고만 찍고 진행한다. GELLO는 어차피 passive read-only라 토크를 못 껐다고 해서 초기화를 통째로 실패시킬 이유가 없다.

```diff
         if real:
             self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
-            self._driver.set_torque_mode(False)
+            try:
+                self._driver.set_torque_mode(False)
+            except RuntimeError as exc:
+                print(f"Warning: failed to disable torque during init: {exc}")
         else:
             self._driver = FakeDynamixelDriver(joint_ids)
         self._torque_on = False
```

`self._torque_on = False`는 그대로라서, 내부 상태는 항상 "토크 꺼짐"으로 유지된다.

### 3. `gello/robots/sim_robot.py` — `MujocoRobotServer`에 `gripper_builtin` / `gripper_invert` 추가

문제 상황: 기존 그리퍼 처리 분기는 별도 `gripper_xml`이 붙어 있을 때(`self._has_gripper`)만 동작하면서 마지막 채널에 `* 255`를 곱했다. 그런데 franka `panda.xml`은 별도 그리퍼 xml 없이 손(hand) 액추에이터를 본체 xml 안에 번들로 갖고 있다 (actuator8, ctrlrange `[0, 255]`). 그래서 GELLO가 보내는 정규화된 `[0,1]` 그리퍼 명령을 **마지막 액추에이터의 실제 ctrlrange로 리스케일**하는 새 분기가 필요했다.

생성자 시그니처에 파라미터 2개 추가:

```diff
         host: str = "127.0.0.1",
         port: int = 5556,
         print_joints: bool = False,
+        gripper_builtin: bool = False,
+        gripper_invert: bool = False,
     ):
         self._has_gripper = gripper_xml_path is not None
+        # Some models (e.g. franka panda.xml) bundle the gripper actuator in the
+        # main xml instead of attaching a separate gripper_xml. In that case the
+        # GELLO's normalized [0,1] gripper command must be rescaled to the last
+        # actuator's ctrlrange (panda actuator8 is [0, 255]).
+        self._gripper_builtin = gripper_builtin
+        # Invert the gripper polarity when the model's "open" end is the high end
+        # of the ctrlrange (panda actuator8: 255 = open) but GELLO sends 1 = closed.
+        self._gripper_invert = gripper_invert
```

모델 로드 후 `nu` 계산 직후, 마지막 액추에이터의 ctrlrange를 캐시한다:

```diff
         self._num_joints = self._model.nu

+        if self._gripper_builtin:
+            self._gripper_ctrlrange = self._model.actuator_ctrlrange[-1].copy()
+
         self._joint_state = np.zeros(self._num_joints)
         self._joint_cmd = self._joint_state
```

명령 처리 분기에 `elif self._gripper_builtin:` 케이스 추가. 정규화 값 `g`를 받아서 `gripper_invert`면 `1.0 - g`로 뒤집고, `lo + g*(hi-lo)`로 ctrlrange에 매핑한다:

```diff
             _joint_state = joint_state.copy()
             _joint_state[-1] = _joint_state[-1] * 255
             self._joint_cmd = _joint_state
+        elif self._gripper_builtin:
+            # map normalized [0,1] gripper command to the actuator's ctrlrange
+            _joint_state = joint_state.copy()
+            g = _joint_state[-1]
+            if self._gripper_invert:
+                g = 1.0 - g
+            lo, hi = self._gripper_ctrlrange
+            _joint_state[-1] = lo + g * (hi - lo)
+            self._joint_cmd = _joint_state
         else:
             self._joint_cmd = joint_state.copy()
```

극성(polarity)에 대해: panda actuator8은 **255가 열림(open)**이다. 그런데 GELLO는 레버를 당겨 쥐었을 때 **1을 보낸다(= 닫힘 의도)**. 그래서 `gripper_invert=true`로 뒤집어야 한다. GELLO 1(쥠) → `1-g=0` → ctrlrange의 lo쪽(0) → 액추에이터 닫힘. 즉 레버를 당기면 sim 그리퍼가 닫힌다.

### 4. `configs/rwh_panda.yaml` (신규) — launch 설정

`experiments/launch_yaml.py`가 읽는 launch 설정. `MujocoRobotServer`(panda.xml, 내장 그리퍼, port 6001)와 `GelloAgent`를 한 파일에서 묶는다.

```yaml
robot:
  _target_: gello.robots.sim_robot.MujocoRobotServer
  xml_path: "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
  gripper_xml_path: null
  gripper_builtin: true   # panda.xml bundles the hand: rescale GELLO [0,1] -> actuator8 [0,255]
  gripper_invert: true    # panda actuator8 255=open; invert so pulling the GELLO lever closes
  host: "127.0.0.1"
  port: 6001

agent:
  _target_: gello.agents.gello_agent.GelloAgent
  port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0"
  dynamixel_config:
    _target_: gello.agents.gello_agent.DynamixelRobotConfig
    joint_ids: [1, 2, 3, 4, 5, 6, 7]
    joint_offsets: [4.712, 3.142, 3.142, 3.142, 0.0, 3.142, 3.142]
    joint_signs: [1, -1, 1, -1, 1, -1, 1]
    gripper_config: [8, 115.376171875, 73.576171875]
  start_joints: [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 1.0]

hz: 30
```

설정 해석:
- `gripper_xml_path: null` + `gripper_builtin: true` 조합이 위 3번 분기를 켜는 트리거. ctrlrange는 모델에서 자동으로 `[0,255]`를 읽어온다 (하드코딩 아님).
- arm 7관절은 라디안 단위로 1:1 teleop. `joint_ids [1..7]`이 panda의 arm 7축에 대응하고, 8번 채널(`gripper_config`의 첫 값 `8`)이 그리퍼.
- `joint_offsets`/`joint_signs`/`gripper_config` 값은 전부 **조민제의 docker 이미지**(`minje227/gello:RWH_Gello_PandaV1.0`)에서 그대로 가져온 것. 같은 물리 GELLO(port `FTBEO75A`, XL330 서보)이므로 재캘리브레이션이 필요 없다. GELLO 조립/서보 장착이 바뀌지 않는 한 유효.
- `start_joints`는 8개(7 arm + 1 gripper). 앞 7개 `[0,0,0,-1.57,0,1.57,0]`은 panda의 표준 home 포즈에 가깝고, 마지막 `1.0`이 그리퍼 정규화 시작값.
- `hz: 30` — teleop 루프 주파수.

### 5. `scripts/sim_panda_scripted_demo.py` (신규) — GELLO 없이 sim 먼저 검증

GELLO 하드웨어/시리얼 포트를 전혀 건드리지 않는 순수 시뮬레이션 데모. `panda.xml`을 직접 로드해서 7개 arm 액추에이터에 사인 스윕을 주고 그리퍼는 open/close 사이클을 돌린 뒤, offscreen 렌더로 GIF + PNG 키프레임을 뽑는다. teleop 배선을 붙이기 **전에** sim 자체(모델 로드, 액추에이터 매핑, 렌더링)가 동작하는지 먼저 확인하는 용도.

핵심 동작:
- 헤드리스 GL 백엔드를 `egl` → `osmesa` 순으로 시도해서 `mujoco.Renderer`를 만든다.
- ctrlrange와 joint range를 모델에서 읽어, 모든 명령을 `np.clip`으로 안전 한계 안에 가둔다.
- 그리퍼(`grip_act = model.nu - 1`)는 ctrlrange 중앙값 기준으로 진폭을 잡아 `[0,255]` 안에서 여닫는다.
- 매 step `np.all(np.isfinite(data.qpos))`로 발산을 감시하고, 시작/끝 qpos 차이(`max_delta`)를 찍어 실제로 움직였는지 증거를 남긴다.

이 데모로 panda.xml의 `nu=8`(arm 7 + 그리퍼 1), actuator8 ctrlrange `[0,255]` 같은 사실을 먼저 눈으로 확인한 뒤, 그 정보를 3·4번 설정에 반영했다.

### 6. `gello/robots/sim_robot.py` — 배경 씬 & 큐브 (`build_scene` opt-in)

기본 `build_scene`은 빈 arena에 팔만 붙여서 **바닥도 조명도 없는 까만 배경**이었다. 여기에 **체커보드 바닥 + 그라데이션 스카이박스 + 조명**과 **로봇 앞 작은 큐브**를 추가하는 opt-in 옵션 두 개(`add_scene`, `add_cube`)를 달았다. 기본값이 `False`라 다른 로봇(sim_ur/yam/xarm)은 영향 없음.

`MujocoRobotServer.__init__`와 `build_scene`에 파라미터를 추가하고, 씬/큐브는 dm_control `mjcf` API로 구성한다 (값은 menagerie `scene.xml` 미감을 미러링):

```python
def build_scene(robot_xml_path, gripper_xml_path=None, add_scene=False, add_cube=False):
    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)
    if gripper_xml_path is not None:
        attach_hand_to_arm(arm_simulate, mjcf.from_path(gripper_xml_path))
    if add_scene:
        # headlight/haze/global + skybox(gradient) + checker groundplane material + light + floor plane
        ...
        arena.worldbody.add("geom", name="floor", type="plane", material="groundplane", size=[5, 5, 0.1])
    arena.worldbody.attach(arm_simulate)
    if add_cube:
        # 팔 attach 뒤에 추가해야 cube freejoint qpos가 arm 뒤로 붙음 (qpos[:8]=arm 보존)
        cube = arena.worldbody.add("body", name="cube", pos=[0.5, 0.0, 0.025])
        cube.add("freejoint", name="cube_free")
        cube.add("geom", type="box", size=[0.02, 0.02, 0.02], rgba=[0.8, 0.2, 0.2, 1.0],
                 mass=0.05, condim=3, friction=[1.0, 0.005, 0.0001])
    return arena
```

핵심 불변식 (검수로 확인):
- **`nu`는 8 그대로** — 큐브는 액추에이터가 없어서 actuator 수 불변 → GELLO 8채널 teleop·`command_joint_state`의 `len==nu` assert 영향 없음.
- **팔 qpos 보존** — 큐브 body를 `attach(arm)` **뒤에** 추가해서 cube freejoint qpos가 arm/hand 블록(`qpos[:9]` = arm 7 + finger 2) **뒤**(`qpos[9:16]`)로 붙음. 서버는 `qpos[:nu]`(=`qpos[:8]`)로 읽으므로 그대로 팔을 가리킴. `nq`는 9 → 16 (+7 freejoint), `nv` 9 → 15.
- **큐브 안착** — 100 step 후 z=0.0199(half-extent 0.02), `[0.5, 0, 0.0199]`에서 안정. 발산/관통 없음.
- panda 기준 +x 0.5m → 팔 도달범위(~0.85m) 안, 4cm라 Franka 핸드로 그랩 가능.

config `robot:`에 `add_scene: true`, `add_cube: true` 추가.

---

## 하드웨어 & 캘리브레이션 (Hardware & Calibration)

### 서보 구성

| ID | 모델 | model num | 역할 |
|---|---|---|---|
| 1~7 | XL330-M288-T | 1200 | arm 관절 |
| 8 | XL330-M077-T | 1190 | gripper |

- **전원:** 5V (operating 3.7~6.0V, 절대 최대 7.0V). 읽기조차 전원 필수.
- **통신:** baud **57600**, protocol **2.0**.

### 인터페이스

| 항목 | 값 |
|---|---|
| 어댑터 | U2D2 (FTDI FT232H 기반) |
| USB id | `0403:6014` |
| 포트 basename | `usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0` |

### 캘리브레이션 값

조민제 도커의 `/gello/configs/rwh_panda.yaml`을 그대로 재사용 (물리적으로 **같은 GELLO 개체**라 유효).

| 항목 | 값 |
|---|---|
| `joint_offsets` | `[4.712, 3.142, 3.142, 3.142, 0.0, 3.142, 3.142]` (모두 π/2의 배수) |
| `joint_signs` | `[1, -1, 1, -1, 1, -1, 1]` |
| `gripper_config` | `[8, 115.376, 73.576]` (id, open°, closed°) |
| `start_joints` | `[0, 0, 0, -1.57, 0, 1.57, 0, 1.0]` |

### gello_get_offset.py 동작 원리

- **offset 탐색:** π/2의 배수 후보들에 대해 brute-force로 `|sign*(raw - offset) - start_joint|`를 최소화하는 offset을 찾음.
- **gripper:** raw 엔코더에서 open/close 각도를 읽음 (대략 42° travel).
- offset은 GELLO 조립체의 **기계적 성질**이라, 서보나 horn을 다시 장착하지 않는 한 **재사용 가능**. (그래서 같은 개체면 위 값을 그대로 씀.)

### 안전 (SAFETY)

- GELLO는 **패시브 리더**임. init 단계에서 토크가 disable 되고, **이후 절대 enable 하지 않음**.
- 텔레오퍼 명령은 **시뮬 로봇으로만** 흐름. GELLO 모터로는 절대 가지 않음.
- **FACTR gravity_compensation 등 active-GELLO 계열 설정은 실행 금지.** (모터에 토크가 걸리는 구성은 이 셋업의 안전 전제를 깨뜨림.)

---

## 개발 여정 & 트러블슈팅 (Development Journey)

> 목표: 실제 Franka는 없이, **MuJoCo 시뮬레이션 속 Franka Panda**를 물리 GELLO 리더 암으로 텔레오퍼레이션 하는 것. 참고 자료는 동료(조민제)의 가이드와 그가 만든 도커 이미지 `minje227/gello:RWH_Gello_PandaV1.0`.

작업은 대략 "환경 셋업 → 시뮬레이터 단독 검증 → GELLO 통신 → 그리퍼 매핑"의 순서로 진행됐고, 단계마다 막힌 지점이 있었음. 시간순으로 정리한다.

### 1. 환경이 비어 있었음 (submodule / venv / 패키지)

처음 받은 상태는 그대로 돌릴 수 없었음.

- `mujoco_menagerie`, `DynamixelSDK` submodule이 init 안 돼 있었음 (빈 디렉토리)
- venv 자체가 없었고, `mujoco` / `dynamixel-sdk`도 미설치

**해결:** submodule 초기화 + `uv venv`로 가상환경 생성 + 필요한 패키지 설치. 이걸로 일단 import 단계는 통과.

### 2. DOF / actuator 개수 의심 (panda.xml의 "9" 문제)

`panda.xml`을 보다가 actuator가 **9개**처럼 보여서, GELLO의 8채널(7 arm + 1 gripper)과 안 맞는 것 아닌가 걱정했음. 그리퍼 매핑이 어긋나면 텔레오퍼가 통째로 틀어지니까 여기서 확실히 짚고 넘어가야 했음.

**해결:** 눈으로 XML 세는 대신 mujoco에 직접 모델을 로드해서 권위 있게 확인 → **`nu=8`** (7 arm + 1 gripper). GELLO 8채널과 정확히 1:1. 보였던 "9번째"는 `<default>` 클래스 안의 `<general>` 엘리먼트라서 실제 actuator로 카운트되지 않는 **false positive**였음. (즉 문제 없음, 안심하고 진행.)

### 3. GELLO 없이 시뮬레이터부터 검증

하드웨어를 붙이기 전에 시뮬 쪽이 멀쩡한지 먼저 못 박아 두는 게 중요했음. 두 단계로 검증:

1. **스크립트 sine-sweep 데모** + 오프스크린 GIF 렌더(`MUJOCO_GL=egl`) → 관절이 의도대로 움직이는지 시각 확인
2. **실제 경로 그대로** `launch_nodes.py --robot sim_panda`를 띄우고, 스크립트로 만든 ZMQ 클라이언트를 붙여서 명령 전송 → 라이브 뷰어 스크린샷으로 Panda가 실제로 움직이는 것 확인

여기까지 통과해서, 이후 문제가 생기면 "시뮬 탓"이 아니라 "GELLO/매핑 탓"으로 범위를 좁힐 수 있었음.

### 4. GELLO 무응답 — 알고 보니 전원 미인가

GELLO를 붙이고 모터 스캔을 돌렸는데 **모든 baudrate에서 모터 0개**. 포트(`/dev/...`)는 열리는데 서보 ACK가 하나도 안 옴.

**진단:** 포트가 열린다는 건 U2D2(USB-시리얼)까지는 정상이라는 뜻. 응답이 0이라는 건 **서보 버스에 전원이 안 들어온 것**. XL330은 별도 5V 외부 전원이 필요하고, **단순히 값을 읽는 것조차 전원이 있어야 함** (토크 인가와는 별개의 얘기 — 전원 ≠ 토크).

**해결:** 사용자가 전원을 켜자 → ID 1~8 모터 **8개 전부** 57600 baud에서 정상 인식, 전부 `torque_enable=0` 상태.

### 5. GelloAgent 초기화 크래시 — XL330 error 128 (alert bit)

전원이 들어왔는데 이번엔 `GelloAgent` 초기화에서 죽음:

```
RuntimeError: Failed to set torque mode for Dynamixel with ID 1
```

드라이버가 `0`을 찍고 곧이어 `128`을 찍은 뒤 예외를 던졌음.

**근본 원인:** XL330은 토크 관련 write에 대해 `dxl_error=128`(alert bit)을 응답으로 돌려주는데, 로컬 드라이버는 **error가 0이 아니면 무조건 raise**하도록 돼 있었음. 그래서 정상 동작인데도 크래시.

**해결:** 조민제의 도커에 이미 같은 패치(commit `f67af69`)가 있었음 — **토크를 DISABLE 할 때는 error 128을 무시**하는 처리. 동일하게 2개 파일에 패치 적용(`driver.py` + `dynamixel.py`). 이후 GELLO는 깔끔하게 읽혔고, **torque OFF 상태 확인**.

### 6. 그리퍼 버그 #1 — 스케일링 (값이 ~0.0007에 멈춤)

이제 텔레오퍼가 돌긴 도는데, 시뮬 그리퍼가 거의 안 움직였음. 사용자가 Control 패널 값을 보니 **~0.0007에 사실상 고정**.

**원인:** `sim_panda`는 `gripper_xml=null`이라 `_has_gripper=False`가 되고, 그 결과 그리퍼 값에 `*255` 리스케일 단계가 **스킵**됨. 그래서 GELLO의 `[0,1]` 정규화 값이 그대로(raw) panda actuator8(ctrlrange `[0,255]`)로 들어가서, 최대로 당겨도 1 근처 → 거의 닫힘 0 수준으로만 보였던 것.

**해결:** `gripper_builtin` 옵션을 추가해서 `[0,1]` → actuator ctrlrange로 리스케일. 검증: `0.0→0`, `0.5→127.5`, `1.0→255`.

### 7. 그리퍼 버그 #2 — 극성 반전 (당기면 열림)

스케일을 고치고 나니, 이번엔 **GELLO 레버를 당기면 시뮬 그리퍼가 열리는** 반대 동작이 나왔음.

**원인:** 두 컨벤션이 정반대였음.

| | 0 (값 작음) | 1 / 255 (값 큼) |
|---|---|---|
| GELLO `g_pos` | 열림(open) | 닫힘/당김(closed) |
| panda actuator8 | 닫힘(closed) | 열림(open) |

**해결:** `gripper_invert` 옵션 추가. 검증: 레버 열면 → 255(open), 레버 당기면 → 0(closed). 이제 직관대로 동작.

### 8. 최종 결과

GELLO 패시브 텔레오퍼 풀 동작 완성:

- 7개 arm 관절이 시뮬 Panda를 **1:1로 추종**
- 그리퍼가 **올바른 방향으로** 열리고 닫힘
- **GELLO 모터는 한 번도 통전(torque enable)되지 않음** — 끝까지 passive leader로만 사용

---

## 향후 작업 (실로봇 ROS2)

실제 Franka FR3/Panda가 준비되면 ROS2 경로로 확장 가능. 동료 도커 이미지(`minje227/gello:RWH_Gello_PandaV1.0`) 안의 `multipanda_ros2` 기반 파이프라인이 레퍼런스다 (이 리포의 `ros2/`와는 별개 패키지).

- **실행 대시보드**: `ROBOT_IP=172.29.0.2 USE_RVIZ=true /workspace/panda_ros2_ws/rwh_gello_realtime_dashboard.sh` — GELLO publisher → arm controller → gripper client 순서로 띄움.
- **RViz 전용(실로봇 없이 시각화)**: `ros2 launch franka_bringup rwh_gello_rviz.launch.py gello_com_port:=/dev/ttyUSB0` (`use_fake_hardware:=true`).
- **Joint impedance controller** (1kHz): P gain `[24,24,24,24,10,6,2]`, D gain `[2,2,2,1,1,1,0.5]`. 실시간성 부족하면 이 게인을 조정.
- **GELLO publisher 설정**: `rwh_panda_ros2.yaml` (torque_enable 전부 0, com_port FTBEO75A, baud 57600).
- 그리퍼는 실로봇에서 `franka_gripper_client`가 width-percent로 따로 구동 (sim의 `gripper_builtin` 경로와 무관).

> 주의: ROS2 실로봇 경로도 GELLO는 동일하게 passive read-only. 토크 인가 금지 원칙은 그대로 유지한다.
