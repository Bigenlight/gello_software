# sim_collect — MuJoCo 안에서 GELLO로 데이터 수집하기 (조작자 런북)

**물리 GELLO 리더팔로 MuJoCo 속 UR7e + Robotiq 2F-85를 텔레옵해서, 실기 레코더와 똑같은
형식의 take를 녹화한다.** 첫 태스크는 "바닥의 채소/과일 하나를 냄비·그릇에 넣기"다.

> 이 문서는 **조작 절차서**다. 설계 계약(왜 이렇게 만들었나, 프로세스/IPC 키, 각 파일의 소유자)은
> [`DESIGN.md`](DESIGN.md)에 있다. 실기 HIL 세션(`run_hil_*.sh` 3-CLI)과는 **완전히 다른 스택이다** — 섞지 말 것.

## Ubuntu 24.04 / Jazzy 새 머신 이전

새 머신 `/home/junhyeong/gello_software_jazzy`는 전용 Conda 환경 `gello-sim`(Python 3.11),
RTX 5070 Ti / driver 590.48.01, `DISPLAY=:1`을 쓴다. 인터랙티브는 GLFW, headless는 EGL로
검증한다. 설치, 서브모듈, 에셋, launcher Python 우선순위, opt-in durable smoke 절차는
[`UBUNTU24_JAZZY_MIGRATION.md`](UBUNTU24_JAZZY_MIGRATION.md)가 정본이다.

아래의 `/home/laptop3/gello_software`, `.venv`, `DISPLAY=:0`, 소프트웨어 GLFW 성능 수치는
기존 laptop3 운용 기록이다. 새 머신 사실로 옮겨 읽지 않는다.

## 0. 실기와 같은 것 / 다른 것

| | 같다 |
| --- | --- |
| 파일 형식 | take 디렉터리 이름, `vectors.h5`의 9개 테이블, `cam1.mp4`/`cam2.mp4`, `depth.h5` — `gello_recorder.recording_session.RecordingSession`을 **그대로 호출**해서 쓴다 |
| EEF 제어 수학 | ROS 브릿지와 **같은 순수 모듈**(`bridge_stages`, `eef_delta`, `ur_kin`)에 **같은 파라미터 파일**(`config/ur7e_gello.yaml` + `ur7e_gello_eef.yaml`)을 먹인다 (`sim_collect/controller.py` 머리말) |
| 카메라 이름 | `cam1` = 정면(씬) 고정 카메라, `cam2` = 손목 카메라 — 실기 `cube_in_cup.py`와 같은 규약 |
| 그리퍼 값 규약 | `gello_grip`/`grip_cmd`/`grip_pos` 전부 **0.0 = 열림 / 1.0 = 닫힘** (CLAUDE.md 「그리퍼 값 규약」) |
| 이산 임계값 | 0.3 / 0.7 히스테리시스 (`gripper.mode: discrete`일 때) |

| | 다르다 |
| --- | --- |
| 장면 | 실물 테이블이 아니라 **텍스처 바닥(z=0)** 위에 시뮬 물체가 놓인다 |
| 추가 테이블 | `vectors.h5`에 `sim_object_poses` / `sim_control` / `sim_leader_filtered` 3개가 **더** 들어간다 |
| 파일 attr | `vectors.h5`에 `sim_meta` JSON attr이 붙는다(실기 파일에는 attr이 없다). 기존 소비자 3종은 attr을 읽지 않으므로 안 깨진다 |
| wrench | 플랜지 force/torque 센서값에서 **시작·리셋 시점 값을 tare(영점)해서** 기록한다 (`sim_main.py::_settle`) |
| 로봇 외형 | 기구학·한계는 UR7e지만 **메시는 menagerie의 UR5e 것**이다 (`assets/robots/ur7e/README.md`) |

---

## 1. 사전 조건 (한 번만)

### 1-1. GELLO에 **5 V 외부 전원**을 넣는다 ← 오늘 이것 때문에 서보가 무응답이었다

USB만 꽂으면 서보는 **응답하지 않는다**. 전원 ≠ 토크 — 전원을 넣어도 토크는 안 걸린다(read-only).
⚠️ XL330은 5 V 서보다. **7 V 초과 금지.**

확인은 읽기 전용 프로브로 한다(ping/read만 보내고 **토크는 절대 안 켠다**):

```bash
cd /home/laptop3/gello_software
python3 scripts/gello_probe.py --config mujoco
```

- **서보 7개(id 1~6 팔 + 7 그리퍼)가 다 보이면 정상.**
- **0개**면 5 V 전원이 안 들어온 것이다(USB는 잡히는데 ping이 하나도 안 돌아온다).
- 기준 자세와 대조하려면 `python3 scripts/gello_probe.py --config mujoco --reference`.
- 포트를 다른 프로세스가 잡고 있으면 **exit 3**으로 멈춘다(진단 도구는 남의 프로세스를 죽이지 않는다).

### 1-2. 포트를 잡은 다른 프로세스를 **먼저 끈다**

🛑 **Dynamixel 드라이버는 포트를 점유한 프로세스를 `fuser -k`로 죽인다.** 플레이그라운드
(`experiments/launch_yaml.py`)나 다른 `sim_collect`가 떠 있으면 서로 죽인다. 런처는 `sim_collect.sim_main`이
이미 떠 있으면 **실행을 거부**하지만, 플레이그라운드는 못 본다.

```bash
pgrep -af "launch_yaml.py|sim_collect.sim_main"   # 나오면 전부 종료한 뒤 시작
```

### 1-3. 나머지

| 항목 | 값 / 확인 |
| --- | --- |
| 새 머신 인터프리터 | Conda `gello-sim` Python 3.11. `SIM_COLLECT_PY`로 명시 가능 |
| laptop3 인터프리터 | `/home/laptop3/gello_software/.venv/bin/python` (legacy fallback) |
| 시리얼 권한 | `Permission denied`면 `dialout` 그룹 (`groups`로 확인, 로그아웃 후 재로그인) |
| 포트 | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0` (by-id는 불변. `configs/carrot_in_pot_sim.yaml`의 `leader.port`) |
| 새 머신 디스플레이 | interactive `DISPLAY=:1`, GLFW. headless `MUJOCO_GL=egl` |
| laptop3 디스플레이 | `DISPLAY=:0`, GLFW만 검증됨. EGL/OSMESA 실패는 laptop3 기록 |

---

## 2. 실행

```bash
cd /home/junhyeong/gello_software_jazzy
conda activate gello-sim
./sim_collect/run_sim_collect.sh
```

laptop3에서는 기존 `/home/laptop3/gello_software/.venv/bin/python` fallback을 계속 쓸 수 있다.

옵션:

| 옵션 | 뜻 |
| --- | --- |
| `--fake-leader` | GELLO 없이 **스크립트 리더**로 돈다 (GUI/녹화 연습, 장비 없는 테스트) |
| `--control-mode joint` | joint 모드로 시작 (기본 `eef`). GUI 라디오로도 바꾼다 |
| `--root <디렉터리>` | take 저장 위치 (기본 `ros2_ur_ws/gello_logs/sim/`) |
| `--config <yaml>` | 씬 설정 (기본 `sim_collect/configs/carrot_in_pot_sim.yaml`) |
| `--headless` | `--no-viewer` + `--no-gui`. backend 미지정 시 EGL |
| `--no-viewer` | MuJoCo viewer만 생략 |
| `--no-gui` | Tk 조작자 GUI만 생략 |

환경변수: `SIM_COLLECT_PY`(최우선 Python override) · `MUJOCO_GL`(호출자 값 보존) ·
`DISPLAY`(interactive 기본 `:0`) · `SIM_COLLECT_OUTPUT_ROOT`(take 루트) · `LOG_DIR`(로그) ·
`SIM_COLLECT_IPC=tcp`(ipc 소켓 대신 127.0.0.1 포트). 자동 Python 순서는
`SIM_COLLECT_PY` → 활성 `gello-sim` → 이름으로 찾은 `gello-sim` → legacy `.venv`다.

interactive에서 **뜨는 것 3개:**

1. **MuJoCo 뷰어 창** — 물리 프로세스 `sim_main`의 것. 마우스로 시점을 돌려도 시뮬에 영향 없다.
2. **조작자 GUI 창**(tkinter) — 아래 3절의 버튼이 전부 여기 있다.
3. **실행한 터미널** — `sim_main.log`를 `tail -F`로 흘린다.

**종료는 터미널에서 Ctrl-C** — 런처가 **캡처를 먼저** 내려 녹화 중이던 take를 마무리(mp4/HDF5 닫기)한 뒤 sim과 GUI를 내리고, 프로세스 그룹에 남은 렌더 워커까지 정리한다. 그래도 녹화 중이면 **STOP TAKE를 먼저 누르는 것이 정석**이다. 셋 중 하나라도 죽으면 런처가 나머지를 정리하고 끝난다.

로그: `sim_collect/logs/<YYYYmmdd_HHMMSS>/{sim_main,capture,gui}.log`. **무언가 안 될 때 제일 먼저 볼 곳이다.**
`--headless`에서는 `gui.log`가 생기지 않는다. GUI 없이 take를 제어하는 전체 경로는
`python -m sim_collect.tools.smoke_collect --output <새 디렉터리>`로 검증한다.

---

## 3. 조작 절차

### 3-1. 시작 상태

기동 시 GELLO를 한 번 읽어 **시뮬 팔을 GELLO의 현재 자세로 텔레포트**한 뒤 **DISENGAGED**로 선다
(`init_from_leader: true`). DISENGAGED에서 팔은 **마지막 명령을 유지**하고 리더를 따라가지 않는다 —
"3D 펜"이라 리더를 아무 데나 들고 있어도 팔은 안 움직인다.
⚠️ **그리퍼는 예외다** — 방아쇠는 engage 여부와 무관하게 계속 먹는다(`sim_main.control_tick`). 멈추려면
`Gripper PAUSE`.

### 3-2. ENGAGE (두 번 클릭)

큰 버튼을 누르면 **3초 안에 한 번 더** 누르라고 오렌지색으로 바뀐다(`CONFIRM ENGAGE (3 s) — 팔이 움직인다`).
두 번째 클릭에서 서버가 게이트를 돌린다. **DISENGAGE는 한 번 클릭**(안전한 방향이라).
단축키 `space`.

게이트가 하나라도 걸리면 engage는 **거부되고 팔은 안 움직인다.** 거부 사유는 GUI 상단 `reason`과 log 창에 그대로 뜬다:

| 사유 키 | 뜻 | 대처 |
| --- | --- | --- |
| `leader_stale` (G2) | 리더 샘플이 오래됨 | GELLO 전원/USB. `sim_main.log`에 읽기 예외가 있는지 |
| `no_command_baseline` (G3) | 명령 체인이 아직 seed 안 됨 | 1~2초 기다렸다 다시 |
| `leader_moving` (G5) | 리더가 "정지" 판정이 안 됨 | **GELLO에서 손을 떼고** 다시 누른다 |
| `chains_disagree` (G4) | 마지막 명령과 실제 팔이 `anchor_agree_tol`(0.02 rad)보다 벌어짐 | 팔이 아직 정착 중이다. 잠깐 기다린다 / `HOME` |
| `filter_not_running` · `filter_not_settled` (G6) | One-Euro 필터 출력이 원시값에 수렴 안 함 | 손 떼고 1초 기다린다 |
| `kinematics_selftest` (G7) | IK(FK(q)) 왕복 실패 | 그 자세가 특이하다. `HOME` 후 재시도 |
| `singular_anchor` (G8) | `sigma_min <= sigma_warn`(0.10) — 앵커 자세가 특이점 근처 | 팔꿈치/손목을 편 자세에서 벗어나게 `HOME` 후 재시도 |
| `keepout` (G9) | 앵커 자세가 keepout 위반 | `HOME` |
| `gap_too_large` | **joint 모드 전용** — 어떤 관절이 리더와 `joint_engage_max_gap_rad`(1.5 rad)보다 멀다 | GELLO를 팔 자세 근처로 가져간 뒤 재시도 |

거부 근거 코드: `sim_collect/controller.py::_common_gates` / `_eef_gates` / `engage`.

### 3-3. EEF 모드 = **GELLO를 3D 펜처럼** 쓴다

engage 순간 리더 EEF와 로봇 EEF를 각각 **앵커로 스냅샷**하고, 그 뒤로는 **앵커 대비 변화량만** 로봇에 더한다.
그래서 **engage 순간 팔은 절대 안 튄다(zero-jump)**, 그리고 **두 팔의 관절 자세가 영구히 달라도 정상**이다.

- `pos_scale` 슬라이더(0.10~1.00)는 **위치 항에만** 곱한다. **회전은 항상 1:1** — `pos_scale=0.10`이어도
  GELLO를 돌리면 팔은 그대로 돌아간다.
- 슬라이더 값은 **즉시 반영되지 않는다.** 다음 engage 때 커밋된다(스트로크 중간에 배율이 바뀌면 미끄러지므로).
  서버도 ENGAGED 중 `set_pos_scale`은 `deferred`로 답한다.

### 3-4. 리더 자세를 고쳐 잡기 (re-anchor)

팔은 많이 움직였는데 손이 불편한 자세가 됐다 →

```
DISENGAGE (팔은 그 자리에 선다) → GELLO를 편한 자세로 옮긴다 → ENGAGE (새 앵커)
```

이게 "펜을 들어 옮겨 다시 짚는" 동작이다. 몇 번을 해도 팔은 제자리에 있다.
(서버에는 `reclutch` 명령도 있지만 **GUI 버튼은 없다** — DISENGAGE/ENGAGE로 같은 일을 한다.)

### 3-5. 그리퍼

- 기본 `continuous`: 방아쇠 값이 그대로 `grip_cmd`가 된다(0=열림, 1=닫힘). `deadband 0.02` 미만 변화는 무시(떨림 억제).
- `configs/carrot_in_pot_sim.yaml`의 `gripper.mode: discrete`로 바꾸면 **0.3/0.7 히스테리시스 래치**:
  `>=0.7`이면 닫힘, `<=0.3`이면 열림, 그 사이는 **직전 상태 유지**.
- `Gripper PAUSE`는 현재 값에서 **얼린다**. `Gripper Resume`은 확인 클릭이 한 번 더 필요하다(재개 순간 방아쇠 값으로 튄다).

### 3-6. RESET SCENE / HOME

| 버튼 | 하는 일 |
| --- | --- |
| `RESET SCENE` (`n`) | **녹화 중에는 거부된다** (STOP TAKE 먼저). DISENGAGE → 팔을 **GELLO의 현재 자세**로 텔레포트 → 그리퍼 열기 → 물체 위치 재샘플 → 속도 0 → `settle_s`(0.5 s) 동안 물리 빨리감기. seed를 안 적으면 **직전 seed + 1**, 옆 `seed:` 칸에 정수를 넣으면 그 seed로 고정(같은 배치 재현) |
| `HOME` (`h`) | DISENGAGE → 팔을 `robot.home_joints`(GELLO 캘리브레이션 자세)로 텔레포트. **물체는 건드리지 않는다** |

둘 다 wrench tare를 다시 잡는다.

### 3-7. 녹화

| 버튼 | 하는 일 |
| --- | --- |
| `START TAKE` (`r`) | 새 take 디렉터리를 열고 기록 시작. 씬/렌더 워커가 준비 안 됐거나 sim state가 1초 이상 끊겼으면 거부한다 |
| `STOP TAKE` (`r`) | 파일을 닫고 `sim_meta`에 `task_success_at_stop`·`duration_s`를 적는다 |
| `DISCARD LAST` | 확인 클릭 2회. **마지막으로 끝낸 take 디렉터리를 통째로 지운다**(되돌릴 수 없다). 녹화 중에는 거부 |

GUI에 `Take: N`, `● RECORDING`, 경과 시간, `frames cam1/cam2 · rows`, take 경로가 뜬다.
상단 오른쪽 `TASK` 배지는 기하학적 성공 판정(음식이 용기 개구부 안 + 거의 정지 + 그리퍼 열림)이며 **참고용**이다 —
학습용 라벨은 변환할 때 사람이 `--outcome`으로 준다.

**단축키:** `space` ENGAGE/DISENGAGE · `r` START/STOP TAKE · `n` RESET SCENE · `h` HOME
(seed 입력칸에 커서가 있으면 단축키는 안 먹는다).

### 3-8. joint 모드는 무엇이 다른가

절대 미러링이다 — 관절각을 1:1로 따라간다. engage 게이트는 G2/G3/G5 + `gap_too_large`뿐이고,
앵커·`pos_scale`·reclutch 개념이 없다. engage하면 실제 자세에서 re-seed해서 soft-start로 리더까지
**미끄러져 간다**(튀지 않는다). ENGAGED 중에는 모드를 바꿀 수 없다.

---

## 4. 씬 — 무엇이 어디 있나

바닥은 LIBERO 나무 텍스처 평면(z=0, 10×10 m)이고 **테이블 지오메트리는 없다 — 작업면이 곧 바닥이다.**
로봇 베이스는 원점, 홈 자세에서 팔은 **월드 +x** 쪽을 향한다 — 실기와 같다(실기 take_18의
`ur_joint_states` J1 ≈ −3.30, `tcp_pose` x ≈ +0.49). GELLO 캘리브레이션도 ROS 실기 것
(`ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml`의 `gello_publisher`)을 **그대로 읽는다** —
시뮬 전용 오프셋은 없다. 정면 카메라 cam1은 베이스에서 +x로 0.70 m, 높이 0.571 m에서 로봇 쪽을 본다.

```
베이스에서 +x를 바라볼 때:   왼쪽(+y) = 음식        오른쪽(−y) = 용기
```

| 왼쪽(+y) | 오른쪽(−y) |
| --- | --- |
| **carrot**(절차적, 18.5 cm) | **pot**(절차적, 안지름 18 cm × 11 cm) |

**기본 씬에는 이 둘만 놓인다**(2026-09-14 결정). 매 RESET SCENE마다 seed로 둘의 위치(당근 반경 6 cm,
냄비 5 cm)와 방향(당근 ±35°, 냄비 임의)이 조금씩 달라지고, 같은 seed면 같은 배치다 — GUI의 seed 칸 또는
`sim_main --seed`. 나머지 후보(딸기·자두·레몬·복숭아·배·바나나·빵, 그릇·바구니)는 `assets/objects/`에
그대로 있고(`assets/objects/README.md`, `assets/preview.jpg`), yaml의 `objects:`와 `layout.items`에
한 줄씩 추가하면 놓인다(주석에 반지름이 적혀 있다).

전부 2F-85로 집을 수 있게 최소 치수 ≤ 7 cm다. 치수·질량·출처는
[`assets/objects/README.md`](assets/objects/README.md), 라이선스는 [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md),
전체 그림은 [`assets/preview.jpg`](assets/preview.jpg).

`configs/carrot_in_pot_sim.yaml`에서 바꾸는 법:

| 하고 싶은 것 | 고칠 곳 |
| --- | --- |
| 물체 위치 | `layout.items.<이름>.nominal_pos` (미터, `[x, y, z]`) · `random_xy_radius` · `random_yaw` |
| 태스크 대상 바꾸기 | `task.food` / `task.container` (예: `food: pear`, `container: bowl`) |
| 물체 빼기/넣기 | `objects:` 목록과 `layout.items:`에서 **같이** 지우거나 추가 |
| 카메라 | `cameras.cam1.pos` / `lookat`, `cameras.cam2.radial_m` / `pitch_deg`, `color_fovy_deg` |
| **좌우 뒤집기** | `layout.items`의 모든 `nominal_pos`에서 **y 부호만 뒤집는다** (yaml 머리말에 그렇게 적혀 있다) |

바꾼 뒤에는 런처를 다시 띄운다. 씬은 config로부터 결정적이라 `sim_main`과 `capture`가 같은 모델을 만든다.

---

## 5. 저장되는 데이터

```
ros2_ur_ws/gello_logs/sim/take_01_20260914_221530/
├── vectors.h5    # 9개 실기 테이블 + 시뮬 3개 + sim_meta attr
├── cam1.mp4      # 정면, 1280x720 @30 mp4v
├── cam2.mp4      # 손목, 동일
└── depth.h5      # cam1/cam2, 848x480 uint16 mm PNG (color에 정렬되지 않음)
```

이름은 `take_{NN:02d}_{YYYYmmdd_HHMMSS}`, `NN`은 **capture 프로세스당 1부터** 센다(실기와 같다).

| 테이블 | 내용 (레이트) |
| --- | --- |
| `synchronized` | 56컬럼 와이드 분석 테이블 (100 Hz 격자, 두 카메라가 다 들어온 뒤부터) |
| `gello_joint_states` | 리더 원시 관절 + 유한차분 속도 (30 Hz) |
| `ur_joint_states` · `command` · `tcp_pose` · `wrench` | 실제 관절/명령/TCP/렌치 (125 Hz) |
| `gripper` | `gello_grip, grip_cmd, grip_pos` (62.5 Hz, 전부 0=열림) |
| `cam1_frames` · `cam2_frames` | mp4 프레임당 한 행 |
| **`sim_object_poses`** | 물체별 `x,y,z,qx..qw` (30 Hz) |
| **`sim_control`** | `engaged, eef_state_code, pos_scale, sigma_min, gamma, ls_scale, task_success` |
| **`sim_leader_filtered`** | One-Euro 출력 `qf1..qf6` — 오프라인에서 컨트롤러 재현용 |

`vectors.h5`의 파일 attr **`sim_meta`**(JSON)에 `git_commit`, `mujoco_version`, 전체 씬 yaml, `layout_seed`,
물체 목록, `chosen_food`/`container`, 카메라 포즈·내부 파라미터, `eef_state_codes`, `task_success_at_stop`,
`duration_s`가 들어간다.

### 변환은 **실기 문서의 명령을 그대로** 쓴다

sim take는 실기 take와 구조가 같으므로 기존 변환기가 수정 없이 먹는다
(`sim_collect/tests/test_recorder.py`가 두 변환기의 로더를 실제 sim take에 돌려 확인한다).

```bash
# (1) learner용 offline demo pickle — 성공 라벨은 사람이 명시한다
python3 serl_ur_infra/scripts/convert_recorded_takes_to_demo.py \
  ros2_ur_ws/gello_logs/sim/take_01_20260914_221530 \
  --output /tmp/sim_demo_take01.pkl \
  --outcome success
```

```bash
# (2) raw 통계
python3 scripts/dataset/make_carrot_raw_stats.py --data $S
# (3) LeRobot 변환 (lr_env 인터프리터로만)
cd /home/laptop3/youngwoong_ws && lr_env/bin/python \
  /home/laptop3/gello_software/scripts/dataset/convert_carrot_to_lerobot.py \
  --data Put_carrot_in_pot --out carrot_in_pot_lerobot --procs 4 --threads 2
```

전체 절차(스테이징 하드링크, 검증, HF 업로드·태그)는
[`docs/ros2/GELLO_UR7E_RECORDING.md`](../docs/ros2/GELLO_UR7E_RECORDING.md) 「허깅페이스 업로드」절,
demo pickle 계약은
[`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](../serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md).

---

## 6. 테스트

```bash
cd /home/junhyeong/gello_software_jazzy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl DISPLAY=:1 \
  conda run -n gello-sim python -m pytest -q -p no:cacheprovider sim_collect/tests
```

- 2026-09-14 저녁 기준 **151 passed**(36 s). 한때 `test_capture.py::test_capture_records_a_take_end_to_end`가
  부하 아래서 떨어졌다("어떤 스트림도 0.2 s 이상 비지 않는다" 불변식, 125 Hz 테이블에 0.39 s 구멍) — 원인은
  실기 레코더의 행 단위 HDF5 resize(행당 904 µs)였고, 시뮬 레코더는 이제 행을 모아 블록으로 쓴다(6.6 µs).
  그래도 검수 프로세스 여러 개가 동시에 돌 때는 타이밍 검사가 흔들릴 수 있다.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`은 **필수**다. ROS overlay를 source한 셸에서는 `launch_testing`
  플러그인이 끼어들어 수집 단계에서 죽는다.
- 새 머신의 정본은 Conda `gello-sim`, laptop3의 정본은 `.venv/bin/python`이다.
  `venvs/gello-hil-actor`는 어느 쪽에서도 이 코드의 환경이 아니다.
- 렌더가 필요한 테스트는 `DISPLAY`가 없으면 skip된다.

---

## 6.2 MuJoCo 씬 재구성 — 무엇이 저장되고 어떻게 되살리나

take마다 `vectors.h5` 안에 두 가지가 더 들어간다.

| 위치 | 내용 |
| --- | --- |
| `/sim_scene` | sim이 컴파일한 **MJCF 원문**(`xml`), 그 sha256, 참조한 **에셋별 sha256·크기**(`assets_manifest`), `layout`(seed 포함), `config`, `git_commit`, mujoco 버전 |
| `/sim_mj_state` | 매 tick(125 Hz)의 **전체 일반화 상태** `qpos[nq]·qvel[nv]·ctrl[nu]` + `sim_t`·`tick` (nq/nv/nu는 attrs) |

메시·텍스처 바이트(35 MB)는 take에 복사하지 않는다. 재생 도구가 `config`+`layout`으로 씬을 다시 빌드해
에셋을 얻고 sha로 변하지 않았는지 확인한 뒤, **저장된 xml**을 그 에셋으로 컴파일한다. 그다음 행마다
`qpos/qvel`을 덮어쓰고 `mj_forward`만 하므로(물리 재시뮬 아님) 녹화 당시 상태가 그대로 재현되고, 어떤 카메라로도
다시 렌더할 수 있다.

```bash
cd ~/gello_software
.venv/bin/python -m sim_collect.tools.replay_take <take_dir> --check            # 재구성 검증(물체 자세 오차 mm)
MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.tools.replay_take <take_dir> --viewer   # 실시간 재생
MUJOCO_GL=glfw DISPLAY=:0 .venv/bin/python -m sim_collect.tools.replay_take <take_dir> --render cam1 cam2 --out /tmp/frames --every 15
```

## 6.3 `stamp_s` 컬럼

실기 레코더가 2026-09-14 타임스탬프 기아 문제를 고치면서 `gello_joint_states`·`ur_joint_states`·`wrench`·
`tcp_pose`·`cam1_frames`·`cam2_frames`에 **원본 메시지의 header stamp**인 `stamp_s`(unix 초) 컬럼을 맨 뒤에
붙였다. 시뮬도 같은 컬럼을 채운다 — 상태 테이블은 물리 스레드가 그 tick을 발행한 벽시계 시각, 리더 테이블은
GELLO 샘플 시각, 프레임 테이블은 렌더 캡처 시각이다(`t_rel_s`는 실기 규약대로 쓰기 시각). 그 수정 이전에
녹화된 실기 take(예: take_18)에는 이 컬럼이 없다.

## 6.4 depth — 옵션이고 기본은 끔

2026-09-14 결정: **depth는 기본으로 수집하지 않는다.** 기본 take는 `vectors.h5` + `cam1.mp4` + `cam2.mp4`
**3개 파일**이고 depth 렌더도 하지 않는다(카메라당 렌더 1회가 줄어 CPU도 아낀다). 켜는 방법 둘:

```bash
./sim_collect/run_sim_collect.sh --depth          # 이번 세션만
# 또는 configs/carrot_in_pot_sim.yaml:  cameras.record_depth: true
```

켜면 실기 레코더와 같은 4개 파일(`depth.h5`, 848×480 uint16 mm PNG + camera_info/extrinsics)이 된다.
`sim_meta.record_depth`에 어느 쪽이었는지 남는다. 소비자 차이: `recorded_demo.py`(HIL offline demo)는 depth가
필요 없고, depth feature를 넣는 `convert_carrot_to_lerobot.py`와 `make_carrot_raw_stats.py`(4개 파일 기대)는
**depth를 켠 take만** 받는다.

## 6.5 카메라 프레임레이트 — laptop3의 역사적 한계

laptop3 렌더는 소프트웨어 GL(NVIDIA 드라이버 미로드)이어서 **CPU 부하에 그대로 노출됐다.** 아래는
laptop3 실측(2026-09-14)이며 RTX 5070 Ti 새 머신의 성능 수치가 아니다.

| 상황 | cam1/cam2 프레임레이트 | 프레임당 렌더 |
| --- | --- | --- |
| 다른 작업 없음, 뷰어 켬 | **30.0 / 30.0 Hz** | 28 ms |
| 같은 PC에서 LeRobot 변환 5개 동시(load 16) | **8.0 / 8.1 Hz** | 116~120 ms |

mp4는 실기와 같이 항상 30 fps로 찍히므로 느리게 잡히면 **영상이 빨리 재생된다.** 레코더가 이것을 숨기지 않는다 —
GUI에 `SLOW`(주황), `stop_take` 응답과 `sim_meta.problems`에 `"cam1 captured at 8.0 fps but cam1.mp4 is stamped
30 (plays 3.75x fast)"`가 남고 `sim_meta.achieved_fps_take`에 실제 값이 있다. **`problems`가 비어 있지 않은 take는
학습에 쓰지 말 것.** 대책은 순서대로: 다른 무거운 작업을 끝낸 뒤 녹화, yaml `render.capture_shadows: false`
(렌더 16 → 3.6 ms), `render.viewer_shadowsize` 축소.

## 7. 문제가 생기면

| 증상 | 원인 / 대처 |
| --- | --- |
| 기동 중 `warning, comm failed`가 몇 줄 | **정상이다.** 드라이버 첫 읽기들이 그렇다 — `warmup_reads`(5)개를 버리고 `connect_timeout_s`(15 s)까지 재시도한다. 15초를 넘겨야 실패다 |
| `gello_probe`에서 서보 **0개** | 5 V 외부 전원 미인가가 1순위. 그다음 U2D2 케이블 |
| `GELLO driver fell back to the FAKE driver ... (port missing/busy?)` | 포트가 없거나 다른 프로세스가 잡고 있다. 1-2절 `pgrep`. **가짜 리더로 조용히 넘어가지 않게 일부러 실패시킨다** — 정말 원하면 `--fake-leader` |
| `a sim_collect process is already running` | 런처의 안전장치(sim_main·capture·gui, 고아 렌더 워커 포함). 표시된 pid를 `kill -TERM`하고 다시 시작한다 |
| 뷰어 창이 안 뜬다 | 새 머신 interactive는 `DISPLAY=:1` + GLFW, laptop3는 `DISPLAY=:0` + GLFW를 확인한다. headless는 `--headless` + EGL이다. `sim_main.log`의 GL 오류와 현재 `MUJOCO_GL`을 같이 본다 |
| GUI가 `sim: DISCONNECTED` / `capture: DISCONNECTED` | 그 프로세스가 죽었거나 재기동 중이다. `logs/<시각>/{sim_main,capture}.log`를 본다. GUI는 **스스로 계속 재접속**하므로 살아나면 알아서 붙는다 |
| `render`의 `dropped`가 는다 | 렌더가 30 Hz를 못 맞춘다(CPU 부하). 프레임을 버려서 **녹화가 sim보다 뒤처지지 않게** 하는 설계다. 다른 무거운 프로그램을 끄거나 `--fps`를 낮춘다 |
| ENGAGE가 안 먹는다 | 3-2절 표에서 `reason` 키를 찾는다. 가장 흔한 둘은 `leader_moving`(손을 떼라)과 `singular_anchor`(HOME 후 재시도) |
| **팔이 GELLO와 180° 다른 자세로 뜬다** | 시뮬은 ROS 실기 캘리브레이션(`ur7e_gello.yaml`, J1 오프셋 0.0)을 읽고 `home_joints` J1 = −3.30 기준으로 ±2π 정규화하므로 정상이라면 실기와 같은 자세다. 플레이그라운드(`configs/rwh_ur.yaml`, J1 = 3.142)와는 **π 다른 것이 의도**다. 그래도 어긋나면 `python3 scripts/gello_probe.py --config ros --reference`로 GELLO를 캘리 자세에 두고 대조하고, 필요하면 `configs/carrot_in_pot_sim.yaml`의 `leader.overrides`로 잠깐 덮어쓴다(정본은 ROS yaml) |
| 녹화 파일이 비어 보인다 | `START TAKE`가 거부됐을 수 있다 — GUI log 창을 본다(`scene/render workers not ready`, `no state from the sim in the last second`) |

---

## 8. 알려진 한계 · 아직 검증되지 않은 것

**정직하게 적는다. 아래는 "될 것이다"이지 "된다"가 아니다.**

1. 🔴 **조작자가 실제 GELLO로 ENGAGE 상태 텔레옵과 물체 파지를 아직 해 본 적이 없다.** 컨트롤러·게이트·녹화는
   단위 테스트와 `--fake-leader`로만 돌았다. 첫 실행은 **짧게, 물체 없이 팔만** 움직여 보는 것부터 시작할 것.
2. **긴 run(수십 분, take 수십 개) 미검증.** 메모리·디스크·렌더 드리프트를 본 적이 없다. depth 포함 용량은
   실기 기준 두 대 약 6 MB/s다.
3. **카메라 위치는 DESIGN §3.5 사양대로 배치했을 뿐, 그림을 보고 튜닝한 적이 없다.** cam1 높이/각도와 cam2
   손목 오프셋은 조작자가 `preview` 창을 보면서 yaml에서 맞춰야 한다.
4. **wrench는 시뮬 플랜지 센서값의 tare 결과**다. 실기 UR의 wrench와 프레임·부호가 같다는 것을 **측정으로 확인한
   적이 없다** — 학습에 쓰기 전에 확인할 것.
5. **UR7e 외형은 UR5e 메시다**(기구학은 UR7e, FK는 `ur_kin.fk`와 일치). 시각적 충돌 판단에 쓰지 말 것.
6. **성공 판정(`TASK` 배지)은 기하학적 근사**다. 학습 라벨이 아니다.
