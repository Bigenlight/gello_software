# GELLO → UR5e + Robotiq 2F-85 MuJoCo 텔레오퍼레이션 셋업

물리 **GELLO**(6-DOF UR 리더 암)로 **MuJoCo 시뮬레이션 속 UR5e + Robotiq 2F-85 그리퍼**를 텔레오퍼레이션하는 셋업 문서. 실제 UR 하드웨어 없이 시뮬레이션만으로 동작하며, GELLO는 **passive read-only(토크 미인가)** 로만 사용한다. [Franka 판](GELLO_PANDA_SIM_TELEOP.md)의 UR 버전.

> **먼저 읽을 것**: 저장소 전체 개요와 공통 사전 준비(STEP-0 Dynamixel Wizard 모터 점검, `dialout` 시리얼 권한)는 [README](../../README.md)에 있음. 서브모듈·venv·의존성 설치는 [Franka 판](GELLO_PANDA_SIM_TELEOP.md)의 0~2단계와 동일하니 먼저 마쳐 둘 것.

- **실행 한 줄**: `DISPLAY=<your-display> .venv/bin/python experiments/launch_yaml.py --left-config-path configs/rwh_ur.yaml`
  - `<your-display>`는 세션의 X 디스플레이(일반 모니터면 보통 `:0`). 이 문서 예시의 `:1`은 **이 헤드리스 빌드 PC의 가상 X 서버**이므로 각자 값(`echo $DISPLAY`)으로 바꿀 것.
- **결과**: UR5e 6축 1:1 추종 + Robotiq 2F-85 그리퍼 정방향 개폐, 앞쪽 테이블에 빨강·초록·파랑 큐브 3개. GELLO 모터는 끝까지 통전되지 않음.

## 요약 (TL;DR)

| 항목 | 상태 |
|---|---|
| MuJoCo UR5e 시뮬 (`ur5e.xml`, nu=6) + 2F-85 attach (nu=7) | ✅ |
| GELLO 6축+그리퍼 읽기 (XL330, 5V 전원 후) | ✅ torque OFF 확정 |
| 캘리브레이션 | ✅ 이 물리 GELLO(FTBEO6QK)로 신규 측정 |
| 팔 6축 텔레오퍼레이션 | ✅ 라디안 1:1 추종 |
| 그리퍼 개폐 | ✅ 별도 `gripper_xml_path`(2F-85) 경로, 극성 정상(invert 불필요) |
| 배경 씬 & 큐브 | ✅ `add_scene`(바닥+스카이박스+조명) + `add_red_cube`(테이블+큐브 3개), opt-in |
| 안전 (GELLO 토크 미인가) | ✅ init부터 끝까지 passive |

### 변경/추가 파일

| 파일 | 종류 | 내용 |
|---|---|---|
| `configs/rwh_ur.yaml` | 신규 | launch_yaml 설정 (UR5e+2F-85 sim + 캘리브레이션 + 씬/큐브) |
| `gello/robots/sim_robot.py` | 수정 | `add_red_cube` opt-in 옵션 추가 (UR용 -x 방향 테이블 + 큐브 3개). 기존 `add_cube`(Franka)는 불변 |

> 토크 관련 패치(`driver.py`, `dynamixel.py`)는 Franka 셋업에서 이미 적용된 것을 그대로 사용 — [GELLO_PANDA_SIM_TELEOP.md](GELLO_PANDA_SIM_TELEOP.md) 참고.

### 목차

1. [셋업 & 재현 가이드](#셋업--재현-가이드-setup--runbook)
2. [코드/설정 상세](#코드설정-상세-code--config)
3. [하드웨어 & 캘리브레이션](#하드웨어--캘리브레이션-hardware--calibration)
4. [UR5e + 2F-85 결합 원리](#ur5e--2f-85-결합-원리-how-the-gripper-attaches)
5. [개발 여정 & 트러블슈팅](#개발-여정--트러블슈팅-development-journey)
6. [향후 작업 (실로봇)](#향후-작업-실로봇-ur)

---

## 셋업 & 재현 가이드 (Setup & Runbook)

작업 디렉토리는 전부 `/home/theo_lab/gello_software` 기준. Franka 셋업([GELLO_PANDA_SIM_TELEOP.md](GELLO_PANDA_SIM_TELEOP.md))의 0~2단계(서브모듈, venv, 의존성)를 이미 마쳤다고 가정한다. UR에 추가로 필요한 것은 없다 — `ur5e.xml`/`2f85.xml`은 이미 받은 `mujoco_menagerie` 서브모듈에 있고, 실로봇용 `ur_rtde`(v1.6.3)도 이미 `.venv`에 설치돼 있다(sim에는 불필요).

### 1) 하드웨어 연결 & 전원 (Hardware / Power)

1. GELLO(UR용, **6-DOF**)의 서보 버스에 **외부 5V 전원**을 연결. USB만으로는 서보가 응답하지 않음(전원 ≠ 토크).
   - **경고**: XL330은 5V 서보. 최대 7V 초과 금지.
2. U2D2를 호스트 USB에 연결. 포트 경로는 `ls -l /dev/serial/by-id/`로 확인한다. 이 빌드 PC의 UR GELLO는 아래 by-id 경로로 잡히고, `configs/rwh_ur.yaml`의 `agent.port`에 박혀 있음:
   ```
   /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
   ```
   (baud 57600, protocol 2.0). `ttyUSBx` 번호는 재접속마다 바뀔 수 있으나 by-id 경로는 불변이라 이걸 쓴다. **다른 어댑터/개체를 쓰는 동료는 자기 by-id 경로로 yaml의 `agent.port`를 바꿔야 함**(`FTBEO6QK`는 이 개체 고유 시리얼).
   - 참고: 이 GELLO는 원래 Franka용으로 쓰던 물리 개체를 UR용으로 재사용한 것(같은 U2D2/포트 FTBEO6QK). 그래서 캘리브레이션을 **UR 기준으로 새로** 측정했다(아래 3단계).

### 2) 시리얼 권한 (Serial permission)

포트 열기가 `Permission denied`로 실패하면 유저가 `dialout` 그룹에 없는 것. 둘 중 하나:

```bash
# 영구(권장): 그룹 추가 후 로그아웃/재로그인
sudo usermod -aG dialout $USER
# 임시: 권한만 열기 (ttyUSB 번호는 ls -l /dev/serial/by-id 로 확인)
sudo chmod 666 /dev/ttyUSB1
```

(공통 준비 항목이라 [README](../../README.md)에도 안내돼 있음.)

### 3) (일회성) 캘리브레이션 — GELLO 재조립 안 했으면 생략

`configs/rwh_ur.yaml`의 `joint_offsets`/`gripper_config`는 **이 물리 GELLO를 UR 포즈로 측정한 값**이라, GELLO 조립/서보 마운팅이 바뀌지 않았으면 재측정 불필요.

다시 캘리해야 하면, GELLO를 **UR 캘리 포즈**로 든 상태에서 실행:

```bash
.venv/bin/python scripts/gello_get_offset.py \
  --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
  --joint-signs 1 1 -1 1 1 1 \
  --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0
```

- **UR 캘리 포즈** = base 0°, shoulder −90°, elbow +90°, wrist1 −90°, wrist2 −90°, wrist3 0°. 참고 이미지: `imgs/robot_known_configuration.jpg`(UR 티치펜던트에 이 각도가 그대로 찍혀 있음), `imgs/gello_matching_joints.jpg`(UR GELLO 매칭 자세).
- 손으로 대충 들어도 됨 — 스크립트가 offset을 π/2(90°) 배수로 스냅해서 찾으므로 각 관절이 0/±90/±180° 중 가장 가까운 칸이면 정확히 잡힘.
- 출력된 `best offsets function of pi: [...]`를 `joint_offsets`에, `gripper open/close (degrees)`를 `gripper_config`에 반영.
- (선택) 그리퍼 open값 정밀화: 그리퍼를 **완전히 벌린 상태**로 다시 돌리면 open각이 정확해짐.

### 4) 실행 (Launch)

```bash
cd /home/theo_lab/gello_software
# DISPLAY은 세션 값으로 (일반 모니터면 :0). :1은 이 헤드리스 빌드 PC 예시.
DISPLAY=:1 .venv/bin/python experiments/launch_yaml.py \
  --left-config-path configs/rwh_ur.yaml
```

- **유효한 `DISPLAY` 필수.** `MujocoRobotServer.serve()`가 GLFW `launch_passive` 인터랙티브 뷰어를 무조건 열기 때문. `DISPLAY`가 잘못되면 뷰어가 안 뜨고 아무것도 안 움직이는 함정이 있음 → 하단 트러블슈팅 참고.
- **`MUJOCO_GL`은 설정하지 말 것**(인터랙티브 뷰어는 GLFW 사용). 헤드리스 offscreen 렌더가 필요할 때만 `MUJOCO_GL=egl`.
- 내부 동작: `MujocoRobotServer`가 `127.0.0.1:6001` ZMQ 서버로 뜨고, `GelloAgent`가 30Hz로 관절값을 읽어 sim에 흘려보냄. config의 `start_joints`로 먼저 이동 후 control loop 시작.

### 5) 정상 동작 확인 (What you should see)

- 지정한 X 디스플레이에 MuJoCo 뷰어가 뜨고 **UR5e + 2F-85 그리퍼**가 체커보드 바닥 + 스카이박스 위에 캘리 포즈로 서 있음. 로봇 앞(**−x 방향**)에 테이블 + **빨강/초록/파랑 큐브 3개**.
- GELLO를 손으로 움직이면 **UR5e 6축이 1:1(radian)로 따라옴**.
- GELLO 그리퍼 레버를 당기면 **sim 2F-85가 닫히고**, 놓으면 열림(극성 정상).
- 로그: `Server ready!` → `Moving robot to start position` → `Control loop: 30 Hz` → `Start 🚀`.
- ⚠️ **기동 직후 `warning, comm failed: -3002`가 1~수 회 찍히는 건 정상**(GroupSyncRead 워밍업). 몇 초 지나면 사라지고 `Time passed:`가 올라감. **이 워밍업 창을 보고 성급히 죽이지 말 것.**

### 6) 종료 (Stop)

```bash
pkill -f launch_yaml.py
```

포트(6001)가 안 풀리면 몇 초 기다렸다 재실행(rapid kill/relaunch는 U2D2 시리얼을 잠깐 불안정하게 만들 수 있음 — 재실행 후 워밍업 여유를 주면 회복됨).

---

## 코드/설정 상세 (Code & Config)

### 1. `configs/rwh_ur.yaml` (신규)

`experiments/launch_yaml.py`가 읽는 launch 설정. `launch_yaml.py`는 완전 robot-agnostic(`_target_`만 보고 동적 인스턴스화)이라, UR 지원은 순전히 이 config 파일의 존재 여부 문제였다.

```yaml
robot:
  _target_: gello.robots.sim_robot.MujocoRobotServer
  xml_path: "third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
  gripper_xml_path: "third_party/mujoco_menagerie/robotiq_2f85/2f85.xml"  # 별도 2F-85 모델 attach (builtin 아님)
  add_scene: true         # checker floor + skybox + lighting
  add_red_cube: true      # UR용 테이블 + 큐브 3개 (-x 방향)
  host: "127.0.0.1"
  port: 6001

agent:
  _target_: gello.agents.gello_agent.GelloAgent
  port: "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0"
  dynamixel_config:
    _target_: gello.agents.gello_agent.DynamixelRobotConfig
    joint_ids: [1, 2, 3, 4, 5, 6]                       # UR은 6-DOF; 그리퍼는 Dynamixel id 7
    joint_offsets: [3.142, 1.571, 4.712, 4.712, 4.712, 3.142]  # gello_get_offset.py 측정값 (J2/J3 재장착 후 재캘리브레이션)
    joint_signs: [1, 1, -1, 1, 1, 1]                    # UR joint-sign 컨벤션
    gripper_config: [7, 210.649609375, 168.849609375]   # [gripper_id, open_deg, close_deg]
  start_joints: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]  # 6 arm(캘리 포즈) + 1 정규화 그리퍼

hz: 30
```

설정 해석:
- **Franka와의 핵심 차이**: Franka는 `panda.xml`에 그리퍼가 번들돼 있어 `gripper_builtin: true`(내장 액추에이터에 `[0,1]→[0,255]` 리스케일)를 썼다. **UR5e 모델엔 그리퍼가 없어서** 별도 `robotiq_2f85/2f85.xml`을 `gripper_xml_path`로 붙인다(런타임 attach). 이 경로는 `_has_gripper=True` 분기를 타서 마지막 채널에 `*255`를 곱한다(2F-85 `fingers_actuator` ctrlrange `[0,255]`).
- **극성**: 2F-85는 0=열림/255=닫힘. GELLO 정규화도 열림→0/닫힘→1이라 자연히 맞음 → `gripper_invert` 불필요.
- **6-DOF**: `joint_ids [1..6]`(팔 6축) + 그리퍼 id 7. `start_joints`는 7개(6 arm + 1 gripper). Franka(7 arm + 1 = 8)와 개수가 다름.
- 값은 전부 이 물리 GELLO(FTBEO6QK)로 UR 포즈에서 새로 측정. `PORT_CONFIG_MAP`의 기존 Left/Right UR 엔트리(FT7WBEIA/FT7WBG6A)는 **다른 물리 개체**라 offset이 달라 재사용 불가.

### 2. `gello/robots/sim_robot.py` — `add_red_cube` opt-in 옵션 추가

기존 `add_cube`는 Franka용으로 **+x 0.5m에 테이블 + 큐브 3개**를 놓는다. UR5e는 base quat(`0 0 0 -1`) 때문에 start 포즈에서 그리퍼가 **−x 방향**을 향하므로(EE pinch ≈ `[-0.49, -0.13, 0.33]`), Franka용 `add_cube`를 그대로 켤 수 없다. Franka config가 기존 동작에 의존하므로 `add_cube`는 건드리지 않고, **독립적인 `add_red_cube` 플래그**를 새로 추가했다(기본값 `False`라 다른 로봇 무영향).

`build_scene()`과 `MujocoRobotServer.__init__` 시그니처에 `add_red_cube: bool = False`를 추가하고, 팔 attach **뒤에** 테이블(고정 슬래브) + 큐브 3개를 배치:

```python
if add_red_cube:
    TABLE_TOP = 0.2
    cx, cy = -0.5, -0.13   # UR5e start 포즈에서 그리퍼가 향하는 -x쪽
    table = arena.worldbody.add("body", name="table", pos=[cx, cy, TABLE_TOP / 2])
    table.add("geom", type="box", size=[0.2, 0.2, TABLE_TOP / 2],
              rgba=[0.55, 0.40, 0.25, 1.0], friction=[1.0, 0.05, 0.001])
    cubes = [
        ("cube",       [cx, cy,        TABLE_TOP + 0.022], [0.8, 0.2, 0.2, 1.0]),  # 빨강
        ("cube_green", [cx, cy + 0.15, TABLE_TOP + 0.022], [0.2, 0.7, 0.3, 1.0]),  # 초록
        ("cube_blue",  [cx, cy - 0.15, TABLE_TOP + 0.022], [0.2, 0.4, 0.8, 1.0]),  # 파랑
    ]
    for name, pos, rgba in cubes:
        cube = arena.worldbody.add("body", name=name, pos=pos)
        cube.add("freejoint", name=name + "_free")
        cube.add("geom", type="box", size=[0.022]*3, rgba=rgba, mass=0.05,
                 condim=6, friction=[2.0, 0.05, 0.001])  # condim=6 + 강마찰: 안 미끄러짐
```

핵심 불변식 (검증 완료):
- **`nu`는 7 그대로** — 큐브는 액추에이터가 없어 actuator 수 불변 → GELLO 7채널 teleop·`command_joint_state`의 `len==nu` assert 영향 없음.
- **팔 qpos 보존** — 큐브 body를 `attach(arm)` **뒤에** 추가해 freejoint qpos가 팔/그리퍼 블록 뒤로 붙음. 서버는 `qpos[:nu]`로 읽으므로 그대로 팔을 가리킴. `nq`: 14 → 35(+7×3 freejoint).
- **안착 확인** — 3개 큐브 모두 테이블 위 `z=0.222`에 안정(발산/낙하 없음, finite). 빨강 `[-0.5,-0.13]`, 초록 `[-0.5,0.02]`, 파랑 `[-0.5,-0.28]` — y로 0.15m씩 벌려 개별 파지 가능.

---

## 하드웨어 & 캘리브레이션 (Hardware & Calibration)

### 서보 구성 (UR GELLO)

| ID | 역할 |
|---|---|
| 1~6 | arm 관절 (UR은 6-DOF) |
| 7 | gripper |

- **전원:** 5V 필수(읽기조차). **통신:** baud 57600, protocol 2.0.
- 어댑터: U2D2 (FTDI FT232H), 포트 basename `usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0`.

### 캘리브레이션 값 (이 GELLO, UR 포즈 측정)

| 항목 | 값 |
|---|---|
| `joint_offsets` | `[3.142, 1.571, 4.712, 4.712, 4.712, 3.142]` (= `[π, π/2, 3π/2, 3π/2, 3π/2, π]`, J2/J3 재장착 후 재측정) |
| `joint_signs` | `[1, 1, -1, 1, 1, 1]` |
| `gripper_config` | `[7, 210.65, 168.85]` (id, open°, close°) |
| `start_joints` | `[0, -1.57, 1.57, -1.57, -1.57, 0, 0]` |

### 캘리브 원리 (요약)

- XL330은 **절대 엔코더**(0~4095 tick = 360°) — raw 읽기값은 전원 재부팅해도 고정. "랜덤"이 아님.
- 다만 raw zero ≠ 로봇 관절 zero. 그 차이가 두 기계적 성질에 담김:
  - **`joint_offset`**: 서보 혼을 끼운 방향(4가지, 각 π/2). `gello_get_offset.py`가 π/2 배수로 brute-force 탐색.
  - **`joint_sign`**: 서보 회전이 관절과 같은/반대 방향(±1). UR은 `1 1 -1 1 1 1`.
- offset은 조립체의 기계적 성질 → 서보/혼 재장착 안 하면 재사용 가능(같은 개체 한정).

### 안전 (SAFETY)

- GELLO는 **패시브 리더**. init에서 토크 disable, 이후 절대 enable 안 함. 텔레오퍼 명령은 sim으로만 흐르고 GELLO 모터로는 안 감.
- **active-GELLO 계열(FACTR gravity_compensation 등) 실행 금지** — 이 셋업의 안전 전제를 깸.

---

## UR5e + 2F-85 결합 원리 (How the gripper attaches)

**menagerie `universal_robots_ur5e/`엔 그리퍼가 없다.** standalone `robotiq_2f85/2f85.xml`만 따로 존재. 사전 결합된 xml도 리포/menagerie 어디에도 없다. 대신 **런타임에 붙인다**:

- `build_scene`이 `attach_hand_to_arm(arm, hand)`(`sim_robot.py:17-51`)를 호출 → dm_control의 `attachment_site.attach(hand_mjcf)`로 2F-85를 UR5e의 `<site name="attachment_site">`(`ur5e.xml:114`, `pos="0 0.1 0" quat="-1 1 0 0"`)에 그래프팅.
- 이건 **body만 옮기는 게 아니라 전체 모델 병합**이다. 2F-85가 작동에 필요한 root-level 요소들(`<equality>` 3개=손가락 4절링크 연결 2 + 좌우 커플링 1, `<tendon>` `split` 1개, `fingers_actuator`, `<contact>` exclude 6개)이 전부 `ur5e/robotiq_2f85/` 프리픽스로 병합돼 보존된다.
- 같은 메커니즘을 `gello/dm_control_tasks/arms/ur5e.py`(+ `ur5e_test.py`)도 사용 — 검증된 레퍼런스.

**실제 동작 검증**(합친 모델을 직접 step):
- `nu=7`(팔 6 + 그리퍼 1), `neq=3`, `ntendon=1` — attach 후에도 그리퍼 메커니즘 살아있음.
- 그리퍼 ctrl=255 → 손가락 +0.782 rad 닫힘, ctrl=0 → 다시 열림. 좌우 1e-6까지 동기(커플링 정상).
- 팔 6축이 캘리 포즈 목표를 ~0.01 rad 오차로 추종. NaN/폭발 없음.
- UR5e base는 world에 고정(freejoint 없음)이라 바닥 없어도 안정.

---

## 개발 여정 & 트러블슈팅 (Development Journey)

> 목표: 실제 UR 없이 **MuJoCo 속 UR5e + 2F-85**를 물리 GELLO 리더로 텔레오퍼레이션.

### 1. 리포에 이미 있던 UR 자산 확인

조사 결과 UR은 이미 상당 부분 지원됨: `gello/robots/ur.py`(실로봇, ur_rtde), `sim_ur`(`launch_nodes.py`), menagerie `ur5e.xml`+`2f85.xml`, `PORT_CONFIG_MAP`의 UR 엔트리, `gello_get_offset.py`의 UR 컨벤션. **없던 것은 `launch_yaml.py`용 `configs/rwh_ur.yaml`뿐**(조민제 가이드가 참조하지만 미커밋). → 이 파일을 새로 만드는 게 핵심 작업.

### 2. 캘리브레이션 — 어디가 "0"인지 모른다는 문제

Dynamixel 절대 위치가 로봇 관절 프레임과 안 맞음(혼 장착 방향). GELLO를 UR 캘리 포즈(`imgs/robot_known_configuration.jpg` = 0/−90/90/−90/−90/0°)로 들고 `gello_get_offset.py` 실행 → offset이 π/2 배수로 깔끔히 스냅됨(`[π, π/2, 3π/2, 3π/2, 3π/2, π]`). 모터 id 1~7 전부 읽혀서 **6-DOF UR GELLO 확정**. 기존 PORT_CONFIG_MAP UR 값과 달라(다른 개체) 새 값 채택. 이후 J2/J3 서보를 재장착하면서 해당 두 축 offset을 다시 측정해 `configs/rwh_ur.yaml`에 반영함(현재 커밋된 값 기준).

### 3. "menagerie UR엔 그리퍼가 없다" 우려

맞는 지적. 하지만 리포의 `attach_hand_to_arm`이 런타임에 standalone 2F-85를 UR5e attachment_site에 붙여주고(전체 모델 병합), equality/tendon/actuator가 전부 보존됨을 실제 sim step으로 검증(`nu=7, neq=3, ntendon=1`, 그리퍼 실개폐 확인). 별도 결합 xml을 만들 필요 없음.

### 4. 실행 함정 — `DISPLAY` 누락 & 워밍업 노이즈

- 유효한 `DISPLAY` 없이 실행하면 GLFW 뷰어 스레드만 죽고 물리 정지(로그는 정상처럼 보임). → 세션의 X 디스플레이 값을 지정할 것(일반 모니터 `:0`, 이 빌드 PC는 `:1`).
- 기동 직후 `comm failed: -3002`(GroupSyncRead 워밍업)를 보고 성급히 죽였다가, standalone 읽기 테스트로 시리얼이 멀쩡함을 확인. **워밍업 몇 초 기다리면 안정화**(정상 실행 시 경고 1~2회 후 사라짐).

### 5. 최종 결과

UR5e + 2F-85 패시브 텔레오퍼 완성 — 6축 1:1 추종, 그리퍼 정방향 개폐, 앞 테이블에 빨강/초록/파랑 큐브 3개(개별 파지 가능), GELLO 무통전.

### 6. 트러블슈팅 요약 (Troubleshooting)

| 증상 | 원인 / 해결 |
|---|---|
| 뷰어가 안 뜨고 아무것도 안 움직이는데 로그는 `Server ready!`까지 정상 | `DISPLAY`가 잘못됨/없음. GLFW 뷰어 스레드만 조용히 죽고 물리가 멈춘 것(ZMQ readiness는 실제 RPC를 안 해서 로그는 정상처럼 보임). `echo $DISPLAY`로 세션 값 확인 후 재실행(일반 모니터 `:0`). |
| 뷰어가 뜨자마자 크래시 | `MUJOCO_GL`이 설정돼 있음. `unset MUJOCO_GL`. 인터랙티브 뷰어는 GLFW를 쓰므로 이 변수를 건드리면 안 됨. offscreen 렌더 때만 `MUJOCO_GL=egl`. |
| 기동 직후 `warning, comm failed: -3002`가 1~수 회 | **정상**(GroupSyncRead 워밍업). 몇 초 지나면 사라짐. 성급히 죽이지 말 것. |
| 포트(6001) 안 풀림 / 재실행 실패 | 이전 프로세스 잔류. `pkill -f launch_yaml.py` 후 몇 초 기다렸다 재실행(rapid kill/relaunch는 U2D2 시리얼을 잠깐 불안정하게 함 — 워밍업 여유를 주면 회복). |
| `Permission denied`로 포트 못 엶 | `dialout` 그룹 미가입. `sudo usermod -aG dialout $USER` 후 재로그인(2단계 참고). |
| 모터 스캔 0개 / 서보 무응답 | **전원 미인가.** XL330은 읽기조차 별도 5V 전원 필요(전원 ≠ 토크). 5V 연결 후 재시도. |
| `RuntimeError: Failed to set torque mode ...` (초기화 크래시) | XL330 alert bit(128) 토크-disable 패치가 없는 상태. **이 RWH fork(`c375187`)에는 이미 들어있어 clone만으로는 안 남.** 업스트림 `wuphilipp/gello_software` 위로 rebase/pull 해서 패치가 덮인 경우에만 발생 → `driver.py`/`dynamixel.py` 패치 재적용([Franka 판](GELLO_PANDA_SIM_TELEOP.md) 코드 변경 1·2번). |

---

## 향후 작업 (실로봇 UR)

실제 UR이 준비되면 sim이 아닌 실로봇 경로로 확장 가능(ROS2 아님 — 이 리포 `ros2/`는 Franka 전용). UR은 Python(ZMQ) 경로:

- **의존성**: `ur_rtde`(이미 `.venv`에 v1.6.3 설치됨). 실로봇은 `gello/robots/ur.py`의 `URRobot(robot_ip, no_gripper)` 사용 — arm은 RTDE, Robotiq 그리퍼는 같은 IP의 TCP 63352.
- **실행(node 경로)**:
  ```bash
  # 터미널 1 — 실 UR 노드
  python experiments/launch_nodes.py --robot ur --robot_ip <UR_IP> --robot_port 6001
  # 터미널 2 — GELLO 컨트롤러
  python experiments/run_env.py --agent gello --robot_port 6001 --gello_port /dev/ttyUSB1
  ```
  또는 `configs/rwh_ur.yaml`의 `robot._target_`을 `gello.robots.ur.URRobot`(+`robot_ip`)로 바꿔 `launch_yaml.py`로도 가능.
- **바이매뉴얼 UR**: `launch_nodes.py --robot bimanual_ur`(IP·GELLO 포트가 소스에 하드코딩 — 편집 필요).

> 주의: 실로봇 경로도 GELLO는 동일하게 passive read-only. 토크 인가 금지 원칙 유지.
