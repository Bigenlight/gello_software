# 09 — HIL actor 기동 런북 (Stage A: fake-env / Stage B: 실센서 DRY_RUN)

> ### 🔧 정정 (2026-07-28) — actor entrypoint에 CLI 플래그 3개가 생겼다
> 이 문서는 그 이전에 쓰였다. 아래가 바뀌었다:
> - **`--arm`** — `DRY_RUN`을 CLI로 해제한다. "DRY_RUN 해제는 이 문서의 범위가 아니다"(§4)는
>   더 이상 맞지 않는다. `--arm` 사용 시 이중 퍼블리셔를 자동으로 거부한다.
> - **`--deadman {topic,spacebar}`**, 기본 `topic` — 이전에는 `deadman`이 전혀 전달되지 않아
>   `SpacebarDeadman`(전역 pynput, 워치독 없음)으로 조용히 fallback했다. **G12 해결.**
> - **`--mock-policy-noise SIGMA`** — zero-action 서버 상대로 로봇을 움직여 개입 경로를 실증한다.
>
> 또한 `clip_safety_box`는 **구현돼 있다**(§4의 "아직 구현되어 있지 않다"는 낡았다). 다만
> `run_real_hil.py` 경로에서는 `DefaultUR7eEnvConfig.ABS_POSE_LIMIT_*`가 0이라 **비활성**이다.
>
> 현재 상태는 [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](../../serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md)를 볼 것.

> 이 문서는 **로봇 랩톱(`laptop3`)에서 HIL-SERL actor 프로세스를 띄우는 절차**만 다룬다.
> 학습 알고리즘, 보상 분류기 학습, 정책 성능은 범위 밖이다.
> 실행은 전부 `ros2_ur_ws/run_hil_actor.sh` 하나로 통일한다. **손으로 python 명령을
> 치지 않는다** — 오늘 실기에서 그것 때문에만 4번 실패했다 (§0).

```bash
# 이 문서의 모든 명령이 쓰는 변수 (워크트리에서 작업 중이면 그 경로로 바꾼다)
export WT=/home/laptop3/gello_software
```

> `run_hil_actor.sh`는 **자기 파일 위치로부터 repo root를 계산한다.** 통합 checkout이든
> 워크트리든 스크립트가 들어 있는 트리를 그대로 쓴다. 경로를 하드코딩할 필요가 없다.

---

## 0. 왜 이 런북과 래퍼가 생겼나 — 오늘의 실패 4건

2026-07-27 실기 세션에서 actor를 띄우는 데 **코드가 아니라 실행 명령 때문에만** 4번 실패했다.

| # | 증상 | 진짜 원인 | 래퍼가 막는 방법 |
|---|---|---|---|
| 1 | 에러도 로그도 없이 **CPU 100%로 영구 정지** | `python3`(시스템 인터프리터)로 실행 → 시스템 grpcio `1.30.2`가 손상돼 있음 | 항상 `/home/laptop3/venvs/gello-hil-actor/bin/python`을 절대경로로 `exec`. preflight [2]가 grpcio 버전을 보고 1.30.2면 거부, [3]이 `cygrpc.CompletionQueue()`를 **타임아웃 건 서브프로세스**로 실제 호출해 코어 생존을 확인 |
| 2 | `ModuleNotFoundError: ur_gello_bringup` | `PYTHONPATH=serl_ur_infra:...` 로 **덮어써서** ROS 오버레이 경로가 날아감 | 항상 `"...${PYTHONPATH:+:$PYTHONPATH}"` 로 **이어붙임**. preflight [4]가 오버레이 경로가 PYTHONPATH에 남아 있는지 직접 확인하고, [5]가 네 모듈의 **실제 해석 경로**를 출력 |
| 3 | 오버레이 없음 | `source install/setup.bash` 누락 | 래퍼가 `/opt/ros/humble/setup.bash` → `<repo>/ros2_ur_ws/install/setup.bash` 순서로 항상 소스 |
| 4 | 상대경로 실패 | `cd` 안 함 | 래퍼가 repo root로 `cd` 하고 모든 경로를 절대경로로 만듦 |

추가로 preflight [5]는 오늘까지 드러나지 않았던 함정 하나를 더 막는다:
`serl-ur-infra`가 **다른 checkout에 editable 설치**돼 있으면 `ur_env`는 import되지만
지금 트리의 코드가 아니고, `ur_experiments`는 아예 없어서 `--ur-config-module`이 깨진다.
preflight는 네 모듈이 **이 트리 안에서** 해석됐는지 경로로 검증한다.

---

## 1. `run_hil_actor.sh` — 사용법

```bash
cd $WT/ros2_ur_ws

./run_hil_actor.sh --help              # 상단 주석(용도/안전/중단) 출력
./run_hil_actor.sh --dry-preflight     # 점검만. actor를 기동하지 않는다 (안전)
./run_hil_actor.sh --fake-env          # Stage A
./run_hil_actor.sh                     # Stage B
```

* `--dry-preflight`와 `--help`만 래퍼가 소비한다. **나머지 인자는 전부 그대로 통과**한다
  (`--fake-env`, `--save-video`, `--actor-id`, `--checkpoint-path`, …).
* 래퍼가 먼저 넣는 기본 인자보다 **뒤에** 붙으므로, 같은 옵션을 다시 주면 사용자 값이 이긴다
  (argparse는 뒤가 이김).
* `exec`로 프로세스를 대체하므로 **Ctrl-C가 곧바로 actor에게** 간다 (중간 래퍼 없음).

### 1.1 환경변수 오버라이드

| 변수 | 기본값 | 비고 |
|---|---|---|
| `ACTOR_VENV` | `/home/laptop3/venvs/gello-hil-actor` | `--system-site-packages` venv (rclpy 상속). grpcio 1.74.0 |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `50153` | 로컬 터널 입구. 원격은 Kanu `50053` |
| `EXP_NAME` | `cube_in_cup` | `ur_experiments/mappings.py`의 `CONFIG_MAPPING` 키 |
| `UR_CONFIG_MODULE` | `ur_experiments.mappings` | |
| `OBS_SCHEMA_HASH` | `3459098d…0352903` | 양쪽이 같아야 한다 (§2.1) |
| `EXPECTED_MODEL_ID` | `hil-serl-hybrid-sac-resnet10-trunk-cache-v1` | **서버 종류에 따라 반드시 바꾼다** (§2.3) |
| `EXPECTED_REWARD_AUTHORITY` | `server_classifier` | |
| `EXPECTED_REWARD_MODEL_ID` | `cube-in-cup-checkpoint-150` | 서버 `--reward-model-id`와 같아야 함 |
| `TIMEOUT_S` / `MAX_RESPONSE_AGE_S` | `0.6` / `0.8` | |
| `HZ_TIMEOUT_S` | `6` | 토픽당 `ros2 topic hz` 대기 시간 |
| `SKIP_ROS_CHECKS` | (미설정) | `1`이면 [7][8][9] 건너뜀. **실기에서는 쓰지 말 것** |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | |

### 1.2 preflight가 보는 것

| # | 점검 | 실패 시 |
|---|---|---|
| 1 | venv python 존재 + `sys.prefix`가 그 venv | FAIL |
| 2 | `grpcio` 버전 ≠ `1.30.2` (import 없이 메타데이터만 읽음) | FAIL |
| 3 | `cygrpc.CompletionQueue()` — **`timeout --signal=KILL 20` 서브프로세스** | 20초 초과 = FAIL. 인라인으로 부르면 이 스크립트도 같이 멈추므로 절대 그렇게 하지 않는다 |
| 4 | `/opt/ros/humble/setup.bash` + `<repo>/ros2_ur_ws/install/setup.bash` 존재·소스, PYTHONPATH 이어붙임 | 없으면 FAIL |
| 5 | `ur_env` / `ur_experiments` / `ur_gello_bringup` / `serl_launcher` + `rclpy`/`gymnasium`/`numpy`/`cv2` 해석 경로 | 못 찾거나 트리 밖이면 FAIL |
| 6 | `SERVER_HOST:SERVER_PORT` TCP connect (3s) | FAIL — 터널/서버 확인 |
| 7 | `/joint_states`, `/gello/joint_states`, cam1/cam2 compressed | 실기 모드 FAIL / fake-env WARN |
| 7b | `/robotiq_gripper/position_percent` | WARN (19-D state의 그리퍼 채널) |
| 8 | `forward_position_controller`가 `inactive`인지 | `active`면 **WARN** (실수 발행 시 팔이 즉시 움직임) |
| 9 | `/forward_position_controller/commands`의 **퍼블리셔 수** | 1개라도 있으면 **FAIL·거부** (이 리그 최대 하자) |

모든 점검은 **읽기 전용**이다 — `ros2 topic hz` / `ros2 topic info` /
`ros2 control list_controllers` / TCP connect뿐이고, 로봇에 아무것도 발행하지 않는다.

---

## 2. Kanu(서버) 쪽 — Stage A/B 공통 전제

### 2.1 오늘 확인된 값 (2026-07-27)

| 항목 | 값 |
|---|---|
| Kanu worktree | `/tmp/gello-hil-rl-receive-server-v2` @ `5fb716b` |
| overlay venv | `/tmp/gello-hil-rl-receive-overlay-v2` |
| GPU | 7 (`CUDA_VISIBLE_DEVICES=7`) |
| 서버 포트 | `50053` (loopback bind) |
| 로컬 터널 입구 | `50153` → 원격 `50053` |
| 분류기 checkpoint SHA-256 | `e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997` |
| observation schema hash | `3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903` (**양쪽 동일**) |

> 포트가 50053이 아니라 **50153**인 이유: 로컬 50053이 다른 프로세스에 잡혀 있었다.
> 터널 로컬 쪽만 바꾸고 원격 쪽은 50053 그대로 둔다.

### 2.2 Kanu에서 서버 띄우기

```bash
# Kanu에서
cd /tmp/gello-hil-rl-receive-server-v2
git rev-parse --short HEAD          # 아래 주석 참조 — 5fb716b 고정 아님
nvidia-smi                          # 쓰려는 GPU가 비어 있는지 매번 다시 확인

# 위 `5fb716b`는 2026-07-27 시점의 Kanu worktree HEAD였다. actor 브랜치가 머지되면
# 그 값이 아니게 되므로 특정 SHA를 기대하지 말 것 — 확인해야 할 것은 "이 트리가
# 랩톱 쪽과 같은 커밋인가"이지 특정 해시가 아니다. 랩톱에서 `git rev-parse --short HEAD`를
# 찍어 같은 값인지 대조하라. GPU도 7번 고정이 아니다(2026-07-28 기준 5/6/7 전부 유휴).

CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/tmp/gello-hil-rl-receive-overlay-v2/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150 \
  --expected-checkpoint-sha256 e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997 \
  --threshold 0.5 \
  --reward-model-id cube-in-cup-checkpoint-150 \
  --replay-capacity 50000 \
  --intervention-capacity 10000 \
  --require-jax-backend gpu
```

> **`--threshold`는 0.5다** (2026-07-29 정정, 이전 판은 0.85). 대상 브랜치 `53d5cf6`이
> `DEFAULT_REWARD_THRESHOLD`를 0.5로 낮췄고, 이 문서의 예시가 0.85로 남아 있으면 코드
> 기본값과 어긋난다. **단 그 근거 수치는 전부 크롭 없는 입력에서 측정된 것이라, 크롭이
> 활성인 상태에서는 재측정이 필요하다** (G15 — `08_OPEN_GAPS.md`).
>
> threshold는 learner fingerprint에 들어간다(`run_rlpd_learner_server.py:548-552`).
> 0.85로 학습된 checkpoint를 resume하려면 `--reward-threshold 0.85`를 명시해야 하며,
> 아니면 게이트에서 **fail-closed로 거부**된다.

근거: `serl_ur_infra/RL_RECEIVE_SERVER.md` §"Preferred Kanu runtime",
`serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md` §"Kanu 실행 환경".

> **주의 — 서버 종류가 두 가지다.**
> * `run_rlpd_receive_server.py` = **receive-only** 마일스톤. 정책 없음, 항상 zero action,
>   `model_id = "fake-zero-action-v0"` (`ur_env/rlpd_receive_server.py:169`).
> * `run_rlpd_learner_server.py` = 실제 RLPD learner. `model_id =
>   "hil-serl-hybrid-sac-resnet10-trunk-cache-v1"` (`ur_env/learner/config.py:14`).
>
> 어느 쪽을 띄웠는지에 따라 actor의 `EXPECTED_MODEL_ID`가 달라진다 (§2.3).

### 2.3 `EXPECTED_MODEL_ID`를 맞추는 법

actor는 **에피소드 시작마다** `GetServerInfo`를 다시 읽고 pin과 대조한다. 틀리면
첫 inference **전에** 죽는다 — 그게 설계된 동작이다. 에러 메시지에 서버가 광고한
실제 값이 들어 있다 (`ur_env/grpc_actor_transport.py:551-557`):

```
ActorProtocolError: server model_id is 'fake-zero-action-v0', expected
  'hil-serl-hybrid-sac-resnet10-trunk-cache-v1'
```

→ 그 값을 그대로 환경변수로 넘긴다:

```bash
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --fake-env
```

### 2.4 SSH 터널 (로컬 터미널 T0, 계속 띄워 둔다)

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50153:127.0.0.1:50053 kanu
```

확인 (다른 터미널):

```bash
ss -ltnp | grep 50153      # ssh가 127.0.0.1:50153에 LISTEN 중이어야 한다
```

`ExitOnForwardFailure=yes`가 핵심이다 — 포워딩에 실패하면 ssh가 조용히 붙어 있지 않고 죽는다.

---

## 3. Stage A — fake-env로 Kanu 왕복 (**로봇 사용 안 함**)

목적: **로봇을 전혀 건드리지 않고** wrapper 체인 · 관측 스키마 · gRPC 계약 ·
버퍼 삽입까지 왕복을 증명한다. `fake_env=True`면 `GelloIntervention` 래퍼가 아예
붙지 않고 ROS 백엔드도 열리지 않는다 (`ur_experiments/cube_in_cup.py`
`get_environment()`의 `if not fake_env:` 분기).

### 3.1 절차 (복붙)

```bash
# T0 — 터널 (§2.4). 그대로 띄워 둔다.
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:50153:127.0.0.1:50053 kanu
```

```bash
# T1 — preflight만 먼저. actor는 안 뜬다.
cd $WT/ros2_ur_ws
./run_hil_actor.sh --dry-preflight --fake-env
```

fake-env 모드에서는 센서 토픽이 없어도 **WARN으로만** 뜨고 통과한다.
반드시 통과해야 하는 것은 [1]~[6]이다.

```bash
# T2 — actor 기동 (Ctrl-C로 중단)
cd $WT/ros2_ur_ws
EXPECTED_MODEL_ID=<서버가 광고하는 값> ./run_hil_actor.sh --fake-env
```

### 3.2 순수 전송 계층만 보고 싶을 때 (receive server 인수 테스트)

actor/wrapper를 빼고 gRPC 전송과 버퍼 삽입만 보려면 100-step 인수 클라이언트를 쓴다.
**서버 버퍼에 synthetic transition 100개가 실제로 들어간다** — 프로덕션 수집 중에는 쓰지 않는다.

```bash
cd $WT
PYTHONPATH=$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH} \
  /home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50153 --steps 100
```

### 3.3 판정 — Stage A는 **통과했다** (2026-07-27, PASS)

서버 로그에서 확인된 값:

```
replay_insert_count : 100
state_shape         : [8, 1, 19]
```

* `replay_insert_count: 100` = 보낸 100개 transition이 전부 replay buffer에 삽입되고 ACK됨.
* `state_shape: [8, 1, 19]` = 서버가 replay에서 **실제로 batch를 샘플링해 본** 결과
  (batch 8 × chunk 1 × 19-D state). 19-D 순서는
  TCP position XYZ / Euler XYZ / linear vel XYZ / angular vel XYZ / force XYZ / torque XYZ / gripper
  (`serl_ur_infra/RL_RECEIVE_SERVER.md`, `ur_env/observation_schema.py`).
* 즉 **스키마 해시 일치 → 분류기 추론 → replay 삽입 → 샘플링 가능**까지 한 줄로 증명됐다.

---

## 4. Stage B — 실센서 + GELLO 개입 (DRY_RUN)

### 4.1 안전 전제 — 읽고 시작할 것

* **`clip_safety_box`는 아직 구현되어 있지 않다** (`08_OPEN_GAPS.md`). 워크스페이스
  박스는 명령을 자르지 않는다. → Stage B는 **DRY_RUN에서만** 한다.
* `cube_in_cup` 태스크 config의 `DRY_RUN`은 **기본 `True`**
  (`ur_experiments/cube_in_cup.py`). 이 래퍼는 그 값을 건드리지 않는다.
  DRY_RUN 해제는 이 문서의 범위가 **아니다**.
* **이중 퍼블리셔 금지.** `gello_ur_bridge`(텔레옵 브리지)를 띄운 채로 actor를 올리면
  두 퍼블리셔가 `/forward_position_controller/commands`를 서로 다른 목표로 때린다.
  preflight [9]가 이를 감지하고 **거부**한다. 거부당하면 브리지를 먼저 끈다.
* 펜던트 E-STOP을 손 닿는 곳에. 작업 공간을 비운다.

### 4.2 터미널 구성

```bash
# 모든 터미널 공통 (run_hil_actor.sh를 쓰는 T6은 예외 — 스크립트가 알아서 한다)
cd $WT && source /opt/ros/humble/setup.bash && source ros2_ur_ws/install/setup.bash
```

**T1 — UR 드라이버 (툴 통신 포함)**

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur7e \
  robot_ip:=192.168.10.11 \
  headless_mode:=true \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  use_tool_communication:=true \
  tool_voltage:=24 \
  tool_device_name:=/tmp/ttyUR
```

* `headless_mode:=true` → 펜던트를 **REMOTE 모드**로 둬야 한다.
* `initial_joint_controller:=scaled_joint_trajectory_controller` → 기동 시
  `forward_position_controller`는 **inactive**다. preflight [8]이 이걸 확인한다.
  (실측 2026-07-27: `forward_position_controller  …  inactive`)
* `use_tool_communication:=true`면 **드라이버가 `:54321`을 점유**하고 `/tmp/ttyUR`을 만든다.
  `:54321`은 **단일 클라이언트**이므로, 이 상태에서 그리퍼 노드는 TCP가 아니라
  **serial `/tmp/ttyUR`** 로 붙어야 한다 (`robotiq_gripper_modbus_node.py:69`의
  `serial_port` 파라미터: 비어 있지 않으면 TCP 대신 serial 사용).

**T2 — 그리퍼 (2F-85)** — 로봇이 **POWER ON** 이어야 툴 전압이 나온다.

```bash
ros2 run ur_gello_bringup robotiq_gripper_modbus \
  --ros-args -p serial_port:=/tmp/ttyUR
```

(실행 파일 이름은 `robotiq_gripper_modbus` — `_node` 없음.
`ros2_ur_ws/src/ur_gello_bringup/setup.py:32`)

확인: `/robotiq_gripper/position_percent`가 ≈5 Hz (실측 2026-07-27: 5.001 Hz).

**T3 — 카메라 2대**

```bash
cd $WT/ros2_ur_ws && VIEW=true ./launch_cameras.sh
```

뷰어로 **cam1 = 장면(삼각대), cam2 = 손목**을 눈으로 확인한다. 뒤바뀌면 관측이 조용히 망가진다.
확인: 두 토픽 다 ≈30 Hz.

**T4 — GELLO 리더** (`/dev/ttyUSB0` 점유 — 두 개 띄우지 말 것)

```bash
GELLO_REPO_ROOT=$WT \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $WT/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

확인: `/gello/joint_states` ≈30 Hz (실측 2026-07-27: 29.98 Hz).
`GELLO_REPO_ROOT`가 없으면 `No module named 'gello'`로 죽는다.

**T5 — HIL 데드맨 GUI**

```bash
cd $WT/ros2_ur_ws && ./run_hil_gui.sh
```

ENGAGE/DISENGAGE 버튼 + 감도 슬라이더. 20 Hz 하트비트를 `/hil/deadman`에 쏜다.
0.5 s 끊기면 env가 자동으로 개입을 해제한다. **스페이스바 데드맨은 쓰지 않는다**
(워치독이 없어 stuck-ON 위험 — `04_HIL_INTERVENTION.md` §1.1).

**T0 — 터널** (§2.4)

**T6 — preflight → actor**

```bash
cd $WT/ros2_ur_ws

# 1) 먼저 점검만. 여기서 초록불이 안 뜨면 actor를 띄우지 않는다.
./run_hil_actor.sh --dry-preflight

# 2) 통과하면 기동
EXPECTED_MODEL_ID=<서버가 광고하는 값> ./run_hil_actor.sh
```

### 4.3 실기 모드 preflight의 통과 기준 (2026-07-27 실측 예시)

```
[1] interpreter = /home/laptop3/venvs/gello-hil-actor/bin/python (python 3.10.12)
[2] grpcio 1.74.0 (손상판 아님)
[3] gRPC 코어 정상 (CompletionQueue 생성 성공)
[4] sourced /opt/ros/humble/setup.bash + <repo>/ros2_ur_ws/install/setup.bash
    ROS 오버레이가 PYTHONPATH에 살아 있다 (덮어쓰기 아님)
[5] ur_env / ur_experiments / ur_gello_bringup / serl_launcher 전부 이 트리에서 해석
[6] TCP 127.0.0.1:50153 — 연결 성공
[7] /joint_states ≈ 100.4 Hz · /gello/joint_states ≈ 30.0 Hz
    cam1 ≈ 30.0 Hz · cam2 ≈ 30.0 Hz · 그리퍼 ≈ 5.0 Hz
[8] forward_position_controller = inactive (안전한 기본 상태)
[9] 퍼블리셔 0개 — actor가 유일한 퍼블리셔가 된다
```

---

## 5. 실패 대응표

| preflight 메시지 | 조치 |
|---|---|
| `actor venv python이 없다` | `ACTOR_VENV` 확인. 없으면 `python3 -m venv --system-site-packages /home/laptop3/venvs/gello-hil-actor` 후 `serl_ur_infra/requirements-grpc.lock` 설치 |
| `grpcio 1.30.2 — 손상된 것으로 확인된 바로 그 버전` | venv 안에 최신 grpcio 재설치. **절대 이 상태로 띄우지 말 것** (조용히 영구 정지) |
| `cygrpc.CompletionQueue()가 20초 안에 돌아오지 않았다` | 위와 같은 증상의 직접 증거. grpcio 재설치 |
| `… 이 저장소 밖에서 해석됨` | 다른 checkout에 editable 설치된 `serl-ur-infra`가 이기고 있다. 지금 트리에서 스크립트를 실행하고 있는지 확인 |
| `ur_experiments: 찾을 수 없음` | 지금 checkout에 `serl_ur_infra/ur_experiments/`가 없다 = 브랜치가 틀렸다 |
| `TCP …:50153 연결 실패` | 터널이 죽었다. §2.4 재실행 → 그래도 안 되면 Kanu에서 서버가 살아 있는지 확인 |
| `cam1 … 에서 6s 동안 메시지가 없다` | 카메라 노드는 살아 있는데 스트림이 멈춘 상태일 수 있다. `ros2 topic info`의 Publisher count가 1인데 `hz`가 비면 **USB 재연결 후 `launch_cameras.sh` 재기동** |
| `forward_position_controller = ACTIVE` (WARN) | 의도한 것이 아니면 `ros2 control switch_controllers --deactivate forward_position_controller` |
| `퍼블리셔가 N개 있다` (FAIL) | 텔레옵 브리지/다른 러너가 살아 있다. `ros2 topic info -v /forward_position_controller/commands`로 범인을 찾아 끄고 재실행 |

### 5.1 `RESET_MAX_DIST_RAD` 게이트 (실기에서 자주 만난다)

리셋은 **현재 자세 → `RESET_JOINTS`** 사이를 250 Hz 업샘플러로 그대로 쓸고 지나간다.
그래서 그 거리가 `RESET_MAX_DIST_RAD`를 넘으면 **에러로 멈춘다 — 그게 의도된 안전 실패다.**
`cube_in_cup`의 게이트는 **`RESET_MAX_DIST_RAD = 0.5`** 로 조여져 있다
(`serl_ur_infra/ur_experiments/cube_in_cup.py:87`; 기본값 1.5보다 좁다).

증상 (`ur_env/envs/ur7e_env.py:550-555`):

```
RuntimeError: reset distance 1.23 rad exceeds RESET_MAX_DIST_RAD=0.5 —
  pre-position the arm with move-to-start first
```

원인: 사람이 GELLO 개입으로 팔을 시작 자세에서 멀리 옮긴 뒤 다음 에피소드를 리셋했다.

조치: **팔을 먼저 시작 자세 근처로 옮긴 뒤** 다시 시작한다.

```bash
# 현재 관절값이 RESET_JOINTS에 가까운지 먼저 눈으로 확인
ros2 topic echo /joint_states --once
```

멀다면 **actor를 내린 상태에서** 팔을 `RESET_JOINTS` 근처(0.5 rad 이내)로 사전 배치한다.
전용 도구가 같은 트리에 있다 (**다른 담당 영역** — 그 스크립트의 자체 안내를 따를 것):

```bash
cd $WT/ros2_ur_ws && ./run_hil_preposition.sh
```

`gello_move_to_start`의 `init_align` 모드를 재사용해 `scaled_joint_trajectory_controller`로
보간 이동하며, 조작자의 명시적 승인 뒤에만 움직인다. `gello_move_to_start`를 맨손으로
단독 실행하는 방법은 이 리그에서 검증되지 않았으므로 여기에 적지 않는다.

> ⚠️ 사전 배치 도구가 도는 동안에는 그것이 팔의 명령 소유자다. **끝난 뒤 반드시 내리고**
> actor preflight [8]이 `forward_position_controller = inactive`, [9]가 퍼블리셔 0개를
> 보고하는지 확인한 다음 actor를 띄운다.

게이트 값을 **키워서 통과시키지 말 것** — 그러면 리셋이 작업 공간을 가로질러 쓸고 간다.

---

## 6. 중단 · 복구

* **actor 중단:** T6에서 `Ctrl-C`. `exec`이므로 신호가 곧바로 actor에게 간다.
* **팔이 움직이는 중이라면 먼저 펜던트 E-STOP.**
* actor가 죽어도 `forward_position_controller`는 **마지막 명령을 유지**한다 → 팔은 그 자리에서
  홀드하고 튀지 않는다.
* GELLO 개입 중이면 T5 GUI에서 **DISENGAGE**.
* 터널이 죽으면 actor는 타임아웃으로 실패한다. §2.4를 다시 띄우고 actor를 재시작한다.
* Kanu 세션을 끝낼 때는 서버와 터널을 정상 종료해 **GPU 7과 포트 50053을 반납**한다.

---

## 7. 검증 상태표

| # | 항목 | 상태 | 근거 / 실측치 (2026-07-27) |
|---|---|---|---|
| A1 | `run_hil_actor.sh` preflight (실기 모드) | **PASS** | 1회차 [1]~[9] 전부 초록: `/joint_states` 99.9–100.4 Hz, `/gello/joint_states` 29.98–30.02 Hz, cam1 30.03 Hz, cam2 30.02 Hz, 그리퍼 5.001 Hz, fpc `inactive`, commands 퍼블리셔 0 |
| A1b | preflight가 **실제 장애를 잡아냈다** | **PASS** | 같은 세션 후반, cam1(이어서 cam2)의 노드는 살아 있고 `ros2 topic info`의 Publisher count도 1인데 데이터가 끊겼다. preflight [7]이 FAIL로 잡고 actor 기동을 막았다 → §5의 카메라 항목 |
| A2 | preflight 실패 경로 | **PASS** | 없는 venv → FAIL·중단(rc=1). 닫힌 포트 → `ConnectionRefusedError`로 FAIL. 격리 도메인 → 토픽 4건 FAIL + 컨트롤러/퍼블리셔 FAIL, actor 미기동 |
| A3 | `--fake-env`에서 센서 점검이 WARN으로 강등 | **PASS** | 격리 ROS 도메인에서 경고 4건 + 통과 |
| A4 | PYTHONPATH 이어붙임 | **PASS** | `PYTHONPATH=/pre/existing`를 미리 잡고 실행해도 `ur_gello_bringup`이 오버레이에서 해석됨 |
| A5 | 인자 통과 | **PASS** | `--fake-env --save-video --actor-id …`가 최종 argv 끝에 그대로 붙음 |
| B1 | Stage A (fake-env, Kanu 왕복) | **PASS** | 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]` |
| B2 | Stage B (실센서 + GELLO 개입, DRY_RUN) | **미검증(TODO)** | 절차는 §4에 있으나 아직 실행되지 않았다. PASS로 승격하지 말 것 |
| B3 | DRY_RUN 해제(실제 팔 구동) | **금지** | `clip_safety_box` 미구현 (`08_OPEN_GAPS.md`) |

---

## 8. 관련 문서

* `04_HIL_INTERVENTION.md` — 데드맨 / 앵커 / 좌표계 / 두 퍼블리셔 충돌
* `05_COMMS_GRPC.md` — gRPC 계약과 19-D state 레이아웃
* `06_SENSORS.md` — RealSense 2대
* `08_OPEN_GAPS.md` — `clip_safety_box` 등 실기 투입 전 미해결 갭
* `serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md` — Kanu learner 서버 전체 런북
* `serl_ur_infra/RL_RECEIVE_SERVER.md` / `HIL_RLPD_RECEIVE_SERVER_KO.md` — receive server 마일스톤
