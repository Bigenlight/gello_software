# 11. 환경이 바뀐 뒤 시뮬 리허설 — GELLO 개체 확인 → mock UR7e 핸드셰이크 → EEF 상태기계

> 2026-09-14 작성. 로봇 설치 위치·테이블·작업공간·카메라·물체가 **전부 바뀐** 뒤,
> 실기에 손을 대기 전에 **랩톱 + 실물 GELLO + mock(가짜) UR7e**만으로 확인할 수 있는 것을
> 순서대로 돈다. 이 문서의 0~2단계는 **2026-09-14에 실제로 PASS**했다(아래 상태표). 3단계는
> 선택이고, §3의 측정표와 §4는 **실기에서 해야 하는 일**을 미리 적어 둔 것이다.

```bash
export WT=/home/laptop3/gello_software
```

## 0. 목적 — 시뮬이 증명하는 것과 증명 못 하는 것

`./run_ur7e_gello_mock.sh`는 **실기 런처(`ur7e_gello_real.launch.py`)와 같은 파일**을
`use_fake_hardware:=true robot_ip:=127.0.0.1`로 띄운다. 그래서 다음은 실기와 **같은 코드
경로**로 검증된다:

| 검증되는 것 | 어디서 |
|---|---|
| **GELLO 개체·캘리브레이션** — 꽂힌 리더가 우리가 캘리브레이션한 그 개체이고 `joint_offsets`/`joint_signs`가 맞는가 | 0단계 probe + 1단계 RViz 형상 비교 |
| **기동 핸드셰이크** — driver → `gello_publisher` + PAUSED 브리지(t≈6 s) → `gello_move_to_start`(t≈8 s) → 수렴 → STRICT 컨트롤러 전환 → 브리지 resume | 1단계 |
| **컨트롤러 전환** — `scaled_joint_trajectory_controller` → `forward_position_controller` | 1단계 |
| **스트리밍** — `/gello/joint_states` 30 Hz → `/forward_position_controller/commands` 250 Hz | 1단계 |
| **EEF 상태기계** — `switch_only` 무동작 기동, `HOLD` → ENGAGE → `ENGAGED` → DISENGAGE → `DISENGAGED` → 재ENGAGE 클러치 | 2단계 |

⚠️ **다음은 시뮬에서 절대 검증되지 않는다.** mock 하드웨어는 명령 자세에 **정확히, 즉시**
도달하고 아무것도 거부하지 않는다:

- **속도 한계·protective stop** — mock에는 없다. 실기 3.14 rad/s 한계는 실기에서만 보인다
  (`check_pause_resume_sim.sh` 머리말도 같은 경고를 적어 둔다).
- **그리퍼** — mock에서는 런처가 Robotiq 노드를 **건너뛴다**(Modbus는 전원 켜진 실기가 필요).
  → [`01_GRIPPER.md`](01_GRIPPER.md).
- **충돌·실제 테이블 높이** — `keepout_json`은 `"{}"`이고(EEF 경로 G9만 게이트, joint 패스스루에는
  keepout 호출부가 아예 없다 — `keepout_ok` 호출은 `eef_delta.py`와 브리지의 engage 게이트 두
  곳뿐이다), RViz 바닥은 실제 테이블이 아니다.
- **베이스 정렬각 `r_align_rpy`** — 시뮬은 `[0,0,0]`을 그대로 쓴다. 로봇을 옮겼으면 **다시 재야
  한다**(§3).
- **JTC 도달 오차(dead-band livelock)** — mock JTC는 정상상태 오차 0이라 `chase_tol` vs
  `arrival_tolerance` 문제는 실기에서만 난다(`GELLO_UR7E_REAL_ROBOT.md` §"실기 캘리브레이션 체크리스트").

## 1. 단계별 절차

### 0단계 — GELLO 개체 확인 (`scripts/gello_probe.py`, 읽기 전용)

**터미널 1** (ROS 소싱 불필요, 시스템 `python3`):

```bash
cd $WT
python3 scripts/gello_probe.py                 # 포트 · 모터 인벤토리 · 현재 자세 · config 비교
```

정상 출력(2026-09-14 실측과 `02_GELLO_LEADER.md` §2 기록이 일치):

| 절 | 정상 |
|---|---|
| `[6] config 중복 비교` | `⚠️ 두 config의 캘리브레이션 값이 다르다` — **J1 offset 0.000(ros) vs 3.142(mujoco)**. 알려진 상태, 아래 🪤 |
| `[1] 시리얼 포트` | `FTBEO6QK -> /dev/ttyUSB0` ✅, `포트 점유: 없음` |
| `[2] 모터 인벤토리` | **7개** — ID 1~6 model **1200**(XL330-M288), ID 7 model **1190**(XL330-M077), `torque_enable` 전부 0, `모델 지문이 문서와 일치` |
| `[3] 현재 자세` | J1~J6 calib(rad/deg) + grip 0..1. 값 자체는 지금 리더가 놓인 자세라 판정 대상이 아니다 |

- 🪤 **exit 3 = 포트를 다른 프로세스가 잡고 있다.** `gello_publisher`(1·2단계 mock, `run_hil_hardware.sh`,
  텔레옵 launch 전부)가 떠 있으면 같은 FTDI를 두 마스터가 써서 ping이 서로 깨진다 — "ID 5, 7만
  응답" 같은 **거짓 결과**가 난다(2026-09-14 실측). 스크립트는 아무것도 죽이지 않는다. 먼저 내리고
  다시 돈다. `--allow-busy`는 진단용이고 결과를 믿으면 안 된다.
- 8개가 응답하면 Franka용 GELLO다. 6개 이하면 케이블/ID 문제 → `02_GELLO_LEADER.md` §3.
- 🪤 **J1 offset π 차이는 오늘 고치지 않는다.** RViz에서 ROS 값(`ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml`)이
  맞았으므로 **ROS yaml이 정본**이다. MuJoCo 쪽(`configs/rwh_ur.yaml`)이 다른 이유는 씬의 UR5e 베이스가
  `third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml`에서 `<body name="base" quat="0 0 0 -1">`
  (Z축 180°)로 돌아 있어서인 것으로 **추정**한다 — ⚠️ MuJoCo 씬을 실제로 띄워 대조하지는 않았다(미확인).
  세 번째 사본(`gello_publisher_node.py`의 `declare_parameter("joint_offsets", ...)` 기본값
  `[3.142, 4.712, 1.571, …]`)은 yaml이 덮어쓰므로 launch 경로에서는 안 보이지만, **`params_file` 없이
  노드를 직접 `ros2 run`하면 그 값이 살아난다.**

**기준 자세 대조** — GELLO를 **캘리브레이션 자세**로 들고 있을 때만 의미가 있다(책상에 놓고 돌리면
당연히 FAIL이고 그건 개체에 대한 증거가 아니다):

```bash
python3 scripts/gello_probe.py --reference     # 기준 = ur7e_gello.yaml start_joints 앞 6개 = [0, -1.571, 1.571, -1.571, -1.571, 0]
python3 scripts/gello_probe.py --reference 0 -1.571 1.571 -1.571 -1.571 0 --tol 0.15   # 값을 직접 줄 때
```

`[4] 기준 자세 대조`가 관절별 ✅/❌와 힌트를 찍는다. FAIL이면 exit 1. 판독:

- **π/2 배수만큼 어긋남 → `joint_offsets[i]`** (정본 도구 `scripts/gello_get_offset.py`)
- **부호 반대 → `joint_signs[i]`**
- **그 외 → 다른 개체 / 재조립 / 기어 슬립**, 또는 기준 자세를 정확히 안 잡은 것.

초보자는 이 자세를 손으로 정확히 잡기 어렵다. **1단계 RViz 비교가 더 직관적이므로 `--reference`는
1단계에서 특정 관절이 이상해 보일 때 숫자 힌트를 얻는 용도로 써도 된다.**

**관절별 방향 확인:**

```bash
python3 scripts/gello_probe.py --watch         # 2 Hz 스트리밍, Ctrl-C 종료
```

관절을 **하나씩** 움직이면 `| J3 +0.120` 식으로 어느 J가 어느 방향으로 반응했는지 찍힌다.
J 번호가 엇갈리면 배선/ID, 방향이 기대와 반대면 `joint_signs`. (이 판단은 UR 관절 부호 규약을
알아야 하므로, 모르면 1단계에서 RViz 팔이 **같은 쪽으로** 움직이는지로 대신 본다.)

### 1단계 — joint 모드 mock: 핸드셰이크 → 스트리밍 → 형상 비교

**터미널 1:**

```bash
cd $WT/ros2_ur_ws
./run_ur7e_gello_mock.sh                       # control_mode joint, RViz 켜짐
# ./run_ur7e_gello_mock.sh launch_rviz:=false  # 헤드리스(DISPLAY 없을 때)
# ./run_ur7e_gello_mock.sh --print-args        # 실제 ros2 launch 한 줄만 출력하고 종료
```

이 스크립트는 `robot_ip:=127.0.0.1 use_fake_hardware:=true`를 **박아 넣고** 덮어쓰기를 거부한다
(명령줄의 `robot_ip:=…`/`use_fake_hardware:=false`, 또는 loopback이 아닌 `ROBOT_IP` 환경변수 →
`REFUSED`, exit 2). `source:=fake`도 거부한다 — 이 런처는 `source` 인자가 없고 **항상 실물
`gello_publisher`**를 띄운다(합성 리더는 3단계 도구를 쓴다). 그 외 `key:=value`(`start_mode`,
`pos_scale`, `v_max`, `w_max`, `jd_gain`, `params_file` …)는 런처로 그대로 넘어간다.

**기동 순서 — 실기와 동일하다:**

```
t=0 s   ur_control(driver, mock 하드웨어) + 컨트롤러 스포너
t≈6 s   gello_publisher(실물 GELLO 30 Hz) + gello_ur_bridge(PAUSED로 선기동)
t≈8 s   gello_move_to_start: 라이브 GELLO 자세를 chase → Converged → STRICT 전환 → 브리지 resume
        → /forward_position_controller/commands 250 Hz 스트리밍
```

🛑 **실기에서도 t≈8 s에 팔이 실제로 GELLO 자세로 이동한다.** mock에서는 RViz 팔이 움직일 뿐이지만,
`run_ur7e_gello_real.sh`에서는 **GELLO가 어디에 있든 로봇이 거기로 간다.** 새 작업공간에서는 그
자세가 테이블·벽과 안 부딪히는지를 **런치 전에** 생각해야 한다(§3의 `start_joints`/작업공간 항목).

**정상 로그** (이 순서로 나와야 PASS. 2026-09-14 헤드리스 실측):

```text
[gello_move_to_start] Chasing live GELLO: gap … rad at … -> …s catch-up.     # 여러 번 나와도 정상
[gello_move_to_start] Converged: max gap … rad held …s (<= 0.06 for 0.4s). Handing over to streaming.
[gello_move_to_start] Switching controllers (STRICT): activate=forward_position_controller, …
[gello_move_to_start] Controller switch OK: forward_position_controller active. …
[gello_move_to_start] Bridge resumed (…); teleop is now streaming.
[launch] Move-to-start handshake SUCCEEDED (forward_position_controller active; bridge resumed via ~/resume).
         use_fake_hardware:=true -- skipping the Robotiq gripper (Modbus needs a real powered robot); …
```

**유량 확인** (터미널 2, 2026-09-14 실측값):

```bash
source /opt/ros/humble/setup.bash && source $WT/ros2_ur_ws/install/setup.bash
ros2 topic hz /gello/joint_states                       # ≈30 Hz
ros2 topic hz /joint_states                             # ≈100 Hz (mock 하드웨어)
ros2 topic hz /forward_position_controller/commands     # ≈250 Hz  ← 이게 나와야 스트리밍 중
```

`Converged`가 안 나오면: GELLO가 계속 움직이고 있거나(리더를 **멈춰야** 수렴), `/gello/joint_states`가
stale(`gello_staleness_s` 0.5 s — 0단계 exit 3처럼 포트를 다른 프로세스가 잡고 있는지 본다).
`chase_timeout_s`(30 s) 뒤 노드가 exit 1로 끝나고 **전환 없이** 멈추는 것이 fail-safe 동작이다.

**형상 비교 요령 (RViz):**

1. GELLO를 들고 **관절 하나씩** 움직인다. RViz 팔의 **같은 관절**이 **같은 방향**으로 따라오면 그
   관절은 PASS.
2. 반대 방향으로 가면 → `joint_signs[i]`. 90° 배수만큼 틀어진 자세면 → `joint_offsets[i]`
   (0단계 `--reference`로 숫자 확인).
3. 여섯 관절 전부 맞으면 "같은 팔"이다. **2026-09-14 사용자 육안 확인: 일치.**

- 🪤 **RViz 메시는 UR5e 껍데기다.** `ur_description`에 ur7e 메시가 없어 `visual_parameters.yaml`이
  ur5e 메시를 가리킨다. 기구학은 ur7e — 겉모양만 빌린 것이니 링크 길이가 "달라 보여도" 무시한다.
- 🪤 **X/Y가 뒤집혀 보이는 착시.** 스크립트가 띄우는 `rviz/hil_operator_view.rviz`는 조작자 시점
  (stock `view_robot.rviz`에서 Yaw+π)이다. 시점을 돌리면 화면의 두 수평축이 같이 뒤집혀 "좌우·앞뒤가
  반대"로 보이는데 **위아래는 맞다** — 코드 문제가 아니라 카메라 방위다. **코드에서 X/Y를 뒤집지
  말 것**(실기가 깨지고 기록 액션이 오염된다. 근거는 메모리 `hil-xy-flip-is-camera-not-code`).
  헷갈리면 RViz에서 시점을 돌려 본다.
- **종료:** 터미널 1에서 Ctrl-C. **~15 s 걸린다** — mock 하드웨어에서 `ros2_control_node`가 SIGINT를
  무시하고 `urscript_interface`가 SIGKILL 승격까지 가야 한다(실측, 무해). 다 끝나기 전에 다음 것을
  띄우지 말 것(포트 점유 → 0단계 exit 3 상황).
- 종료 뒤 `ros2 node list`에 잔상이 남으면 `ros2 daemon stop && ros2 daemon start`.

### 2단계 — eef 모드 mock: 무동작 기동 → ENGAGE → 클러치

**터미널 1:**

```bash
cd $WT/ros2_ur_ws
./run_ur7e_gello_mock.sh control_mode:=eef
```

`control_mode:=eef`면 런처가 `start_mode`를 **`switch_only`**로, `bridge_resume_service`를
**`/gello_ur_bridge/eef_resume`**로 자동 유도한다(`GELLO_UR7E_EEF_MODE.md` §3.5). 즉 t≈8 s에
**궤적을 만들지도 보내지도 않고** 컨트롤러만 제자리 STRICT 전환 → `eef_resume` → 브리지가
**`HOLD`**로 앉는다. 로그에 이 경고가 나오는 게 정상이다:

```text
[gello_move_to_start] start_mode=switch_only: NOT moving the arm and NOT checking joint alignment. …
[gello_move_to_start] Controller switch OK: forward_position_controller active. …
```

🛑 `start_mode:=gello`를 같이 주면 eef 모드에서도 **기동하자마자 팔이 GELLO 자세로 스윕**한다.
배너의 `start_mode=` 줄을 읽고 시작할 것.

**터미널 2 — 상태 모니터:**

```bash
source /opt/ros/humble/setup.bash && source $WT/ros2_ur_ws/install/setup.bash
ros2 topic echo /gello_ur_bridge/eef/state          # state 필드: HOLD / ENGAGED / DISENGAGED / JOINT_BOOTSTRAP
ros2 service list | grep eef                         # eef_engage / eef_disengage / eef_reclutch / eef_to_joint / eef_resume 다섯 개
```

**터미널 3 — GUI:**

```bash
cd $WT/ros2_ur_ws && ./run_eef_gui.sh
```

| 순서 | 조작 | 보여야 하는 것 | 아니면 의심 |
|---|---|---|---|
| a | 아무것도 안 누르고 GELLO를 움직인다 | 상태바 **HOLD**(황), RViz 팔 **무이동**, 큰 토글 문구 `ENGAGE  (press to start following)` | 팔이 따라오면 `start_mode`가 `gello`로 덮였거나 브리지가 `JOINT_BOOTSTRAP`이다 — 배너/`eef/state` 확인 |
| b | 큰 토글 1회 클릭 | 주황 `Click AGAIN to ENGAGE (robot WILL move)` (3 s 창) | — |
| c | 3 s 안에 2회째 클릭 | 상태바 **ENGAGED**(녹), 토글 `DISENGAGE (clutch up)`. GELLO를 펜처럼 밀면 RViz TCP가 **같은 방향·같은 크기**로 따라온다 | 방향이 회전돼 있으면 `r_align_rpy`(시뮬은 `[0,0,0]` 그대로) — 실기 측정 대상(§3). 여기서 고치지 말 것 |
| d | 토글 클릭(1회) | **DISENGAGED**(회), 팔 정지. GELLO를 움직여도 무이동 | 움직이면 FAIL — `eef_disengage` 거부 로그 확인 |
| e | GELLO를 다른 자리로 옮긴 뒤 토글 2회 클릭 | 토글 문구 `ENGAGE  (new reference here)` → ENGAGED. 팔은 **점프 없이** 지금 자리에서 새 델타를 따른다(새 앵커) | 점프하면 앵커 재설정 결함 — 재현 조건을 기록하고 멈춘다 |

GUI는 DISENGAGED에서 `eef_resume → pos_scale 커밋 → eef_engage`를 **자동으로 잇는다.**
GUI 없이 같은 것을 하려면(전부 `std_srvs/srv/Trigger`):

```bash
ros2 service call /gello_ur_bridge/eef_engage    std_srvs/srv/Trigger   # HOLD에서 켜기
ros2 service call /gello_ur_bridge/eef_disengage std_srvs/srv/Trigger   # 끄기(즉시 정지)
ros2 service call /gello_ur_bridge/eef_resume    std_srvs/srv/Trigger   # disengage 뒤 재무장 → HOLD (이걸 먼저 안 하면 eef_engage가 G0에서 거부된다)
```

**2026-09-14 실측:** `switch_only` 기동에서 팔 무이동, `/gello_ur_bridge/eef_resume` 서비스 존재를
확인했다. ⚠️ **표의 c~e(ENGAGE 뒤 방향 일치·클러치)는 이 문서 작성 시점에 아직 돌지 않았다** —
사용자가 지금 돌리는 중이면 결과를 이 표 옆에 날짜와 함께 적는다.

실기 P-단계(`03_EEF_MODE.md` §3: **P6** `pos_scale:=0.0 v_max:=0.01 w_max:=0.05` → **P7** 병진 →
**P8** 회전 → **P9a/P9b** 6-DoF)와 같은 인자를 mock에도 넘길 수 있다 — 예: `./run_ur7e_gello_mock.sh
control_mode:=eef pos_scale:=0.0`이면 ENGAGE 뒤에도 TCP 위치가 안 움직여야 한다(P6의 논리 부분).
다만 mock에는 속도 한계가 없으므로 `v_max`/`w_max`의 **체감**은 실기에서만 의미가 있다.

### 3단계 (선택) — 합성 리더(fake_gello)로 엣지 케이스

실물 GELLO 없이, 결정론적 패턴으로 돌린다. 핸드셰이크는 **없고** 각각 기존 문서가 정본이다.

| 도구 | 무엇 | 문서 |
|---|---|---|
| `./run_ur7e_gello_sim.sh` (+ 터미널 2 `./check_pause_resume_sim.sh`) | joint 모드 `ur7e_gello_rviz.launch.py source:=fake`. 두 번째 스크립트가 pause / resume-chase 거부·수락을 PASS/FAIL로 채점 | 스크립트 머리말 |
| `ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py pattern:=hold` | EEF mock. `pattern`은 `fake_gello`의 `_VALID_PATTERNS` = `sweep` / `hold` / `offset_hold` / `line_xyz` / `wrist_singularity` / `full_rotation` / `step` | [`GELLO_UR7E_EEF_MODE.md`](../ros2/GELLO_UR7E_EEF_MODE.md) §3 |
| `PATTERN=hold JD_GAIN=0.0 ./run_ur7e_gello_joint_delta_mock.sh` | joint_delta 모드(`SOURCE`/`PATTERN`/`JD_GAIN` 환경변수). `full_rotation`이 ±π 경계 검사 | [`GELLO_UR7E_JOINT_DELTA_MODE.md`](../ros2/GELLO_UR7E_JOINT_DELTA_MODE.md) |

⚠️ 셋 다 실물 GELLO 포트를 잡지 않는다(`source:=fake`가 기본) — 단 `SOURCE=gello`를 주면 잡는다.

## 3. 실기 전 측정표 — 환경이 바뀌었으니 다시 재야 하는 값

**yaml을 고치면 반드시 `cd $WT/ros2_ur_ws && ./build_ur7e.sh`** (`colcon build --packages-select
ur_gello_bringup`; 설치는 **복사식**이라 빌드 전에는 노드가 옛 값을 본다). `v_max` / `w_max` /
`pos_scale`만 launch 인자라 재빌드 없이 바꿀 수 있고, 나머지는 전부 yaml + 재빌드다.

| 값 | 왜 다시 재나 | 파일 · 파라미터 | 절차 / 주의 |
|---|---|---|---|
| **베이스 정렬각** `r_align_rpy` | 로봇과 GELLO의 상대 배치가 바뀌었다. 지금 `[0.0, 0.0, 0.0]`은 **측정된 적 없는 가정값**이고, 틀리면 "크기는 맞고 방향만 틀린" 고약한 증상(10~20°가 가장 위험 — 합리화되고 넘어간다) | `ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello_eef.yaml` → `gello_ur_bridge.ros__parameters.r_align_rpy` (rad, `[roll, pitch, yaw]`) | [`GELLO_UR7E_EEF_MODE.md`](../ros2/GELLO_UR7E_EEF_MODE.md) **§1.6** — 로봇 무이동, 다림줄 2개 + 실 + 각도기로 ψ를 재서 `[0, 0, ψ]`. ≤5°면 충분. launch 인자 **아님** → 재빌드 |
| **테이블 높이 / 금지 영역** `keepout_json` | 테이블·작업공간이 바뀌었다. 지금 `"{}"` = 게이트 없음(H1) | 같은 파일 `keepout_json` — **한 줄 JSON 문자열.** `ur_kin.keepout_ok` 키: `floor_z`, `margin`, `link_radius`, `tool` / `tool_xyz_rpy`, `base_cylinder{radius,height}`, `cylinder{point,axis,radius}`, `halfplane{point,normal}` | `floor_z`는 **base_link 기준 z(m)**로 물리 테이블 높이를 적고 `link_radius`로 팔 두께를 부풀린다(`table + tool + radius`를 손으로 계산하지 말 것 — docstring). ⚠️ **EEF 경로에서만** 검사된다(`eef_delta.step`과 engage 게이트 G9). joint 패스스루에는 keepout이 없다 |
| **시작 자세** | joint 모드는 t≈8 s에 **GELLO가 있는 자세로** 로봇이 간다. 새 셀에서 그 자세가 안전한지 사람이 보장해야 한다 | `ur7e_gello.yaml` → `gello_publisher.start_joints`(7개: 팔 6 + 그리퍼 1)는 **캘리브레이션 자세 = ±2π 정규화 기준**이지 로봇을 보내는 목표가 아니다. `gello_move_to_start.init_pose`(6개)는 **`start_mode: "init_align"`에서만** 쓰인다(기본 `"gello"`에서는 무시) | 둘 다 **재캘리브레이션하지 않았다면 건드리지 않는다.** 대신 런치 전에 GELLO를 새 셀에서 안전한 자세로 들고 있는 것이 통제 수단이다. 첫 실기는 팔이 갈 자리를 비워 두고 E-STOP에 손 |
| **툴/TCP** `tool_l_xyz_rpy` · `tool_r_xyz_rpy` | 그리퍼나 TCP가 바뀌었을 때만 | 같은 eef yaml. 지금 둘 다 `[0, 0, 0.174, 0, 0, 0]`(2F-85 실측, **미터**) | 🪤 **펜던트 TCP는 코드로 전파되지 않는다** — EEF 경로는 자체 FK(`ur_kin.fk`) + 이 값을 쓴다. 펜던트에서 TCP를 다시 잡으면 여기도 손으로 옮긴다. 둘은 같은 값으로 유지(§1.1 "가상 풀사이즈 UR7e" 논거) |
| 로봇 기구학 캘리브레이션 | **로봇 개체가 같으면 그대로.** 바뀌었을 때만 | `run_ur7e_gello_real.sh`의 `CALIB=/path/to/ur7e_calibration.yaml` | [`GELLO_UR7E_REAL_ROBOT.md`](../ros2/GELLO_UR7E_REAL_ROBOT.md) §4 |
| (RL 쪽) 워크스페이스 박스·RESET 자세 | 텔레옵과 무관. HIL/RL 세션을 다시 열 때 | `serl_ur_infra/ur_experiments/cube_in_cup.py`의 `ABS_POSE_LIMIT_LOW/HIGH` 등 | [`08_OPEN_GAPS.md`](08_OPEN_GAPS.md) G1. 이 문서 범위 밖 |

**카메라·물체 변경은 텔레옵과 무관하다.** 녹화/RL 쪽에서만 본다 — cam1/cam2 매핑 육안 확인(팔 한 번
흔들기)은 [`06_SENSORS.md`](06_SENSORS.md) **§1.3**, 정책 크롭은 `cube_in_cup.py`의 `IMAGE_CROP`
(측정값이다 — "안 맞으니" 지우지 말 것, 루트 `CLAUDE.md` "반드시 지킬 것").

## 4. 실기로 넘어갈 때 (짧게)

```bash
cd $WT/ros2_ur_ws
./run_ur7e_gello_real.sh                      # Method A: 펜던트에서 External Control 프로그램 Play
HEADLESS=true ./run_ur7e_gello_real.sh        # Method B: 펜던트 Play 불필요 — 단 Remote 모드 필수
```

- **Method A vs B, 하나만 고른다**(한 세션에 섞으면 두 URScript 주입 경로가 싸운다 — `remote_helpers.sh`
  머리말, [`00_SETUP_AND_SAFETY.md`](00_SETUP_AND_SAFETY.md) §5.3).
- 🪤 **2026-09-14 대시보드 조회 결과 `is in remote control: false`였다** — 이 상태로 `HEADLESS=true`는
  쓸 수 없다. 펜던트 **Settings › System › Remote Control › Enable** 후 헤더의 Local/Remote 토글을
  Remote로(Remote → Local 복귀는 펜던트에서만). 확인은:
  ```bash
  echo -e 'is in remote control\n' | nc 192.168.10.11 29999     # true 가 나와야 Method B
  echo -e 'robotmode\n'            | nc 192.168.10.11 29999     # RUNNING
  echo -e 'safetystatus\n'         | nc 192.168.10.11 29999     # NORMAL
  ```
- **E-STOP은 손 닿는 곳에.** Ctrl-C는 "그 자리에 세우는 것"이지 "안전한 곳으로 빼는 것"이 아니다
  (`forward_position_controller`가 마지막 명령 자세를 홀드) → `00` §6.
- joint 모드 실기 첫 구동은 [`GELLO_UR7E_REAL_ROBOT.md`](../ros2/GELLO_UR7E_REAL_ROBOT.md)
  (§2 핸드셰이크·프리플라이트·정상 로그), EEF는 [`03_EEF_MODE.md`](03_EEF_MODE.md) P6부터.
  세션 전 체크리스트 전체는 [`00_SETUP_AND_SAFETY.md`](00_SETUP_AND_SAFETY.md) §5.

## 5. 상태표 (2026-09-14)

| 항목 | 상태 | 근거 |
|---|---|---|
| GELLO 개체 확인 (`gello_probe.py`) | **PASS** | 7 모터(1200×6 + 1190×1), FTDI FTBEO6QK, 2026-07-27 기록과 일치. `--reference`/`--watch`는 도구 존재 확인 |
| joint mock 핸드셰이크 스모크 (헤드리스) | **PASS** | gello 30 Hz / joint_states 100 Hz / commands 250 Hz, `Converged` → `Controller switch OK` → `Bridge resumed` |
| joint mock RViz 형상 일치 | **PASS (사용자 육안)** | "같은 팔이다" |
| eef mock `switch_only` 무이동 + `eef_resume` 서비스 존재 | **PASS** | 기동 시 팔 무이동 확인 |
| eef mock ENGAGE 방향 일치 / DISENGAGE / 재ENGAGE 클러치 | ⚠️ **미검증** (이 문서 작성 시점) | §1 2단계 표 c~e |
| 3단계 합성 패턴 | 미실행 (선택) | — |
| §3 측정표 전 항목 | **실기 필요** | 시뮬로는 불가 |
