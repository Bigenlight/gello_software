# GELLO → UR7e ACT 정책 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 학습된 ACT "put right banana in pot" 정책으로 실제 UR7e를 자율 구동하기 위한 실기(real-robot) 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)이며, 여기서 다루는 콜드스타트 절차(colcon build on Humble, 실제 핸드셰이크, 실제 카메라 시리얼, 실제 팔 동작)는 **이 문서를 작성한 개발 PC에서는 검증되지 않았습니다.** 반면 이미지 파이프라인 byte-parity, ZMQ 라운드트립, HOLD 유지, arming guard, 두 안전 클램프, fail-silent FAULT 전이는 개발 PC에서 오프라인으로 검증되었습니다 — 각 섹션에 어느 쪽인지 명시합니다.
>
> - **패키지 개요 / 빌드 방법**: [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md)
> - **설계 근거**: 프로젝트 루트 `DEPLOY_REPO_DECISION.md` (왜 `gello_software`를 확장하는지, 왜 synthetic-leader + ZMQ 분리인지, HIL-SERL로 가는 길)
> - **관련 실기 문서**: [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) (사람 GELLO 리더로 하는 handshake의 전신), [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) (이 정책을 학습시킨 데이터셋을 만든 레코더)

이 패키지(`gello_policy`)는 학습된 ACT 정책이 **사람 GELLO 리더 대신** `/gello/joint_states` + 그리퍼를 발행하게 해서, 기존 `gello_ur_bridge` + `gello_move_to_start` handshake + Robotiq Modbus 스택을 **코드 변경 없이** 그대로 재사용한다. 정책 추론은 별도의 py3.12 프로세스(torch + lerobot)에서 돌고, ROS py3.10 프로세스와는 localhost ZMQ로만 통신한다.

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 대상 로봇 | UR7e + Robotiq 2F-85, ROS2 **Humble** |
| 체크포인트 | HF `Bigenlight/act_banana_in_pot` (`scripts/download_checkpoint.sh`) |
| 두 프로세스 | ACT 서버(py3.12, torch+lerobot 0.6.1, `policy_server/act_server.py`) ⇄ ZMQ(localhost:5591) ⇄ `policy_leader_node`(py3.10 rclpy) |
| 정책 → 팔 경로 | `policy_leader_node`가 `/gello/joint_states`(30Hz) + `/robotiq_gripper/command_percent`를 발행 → 기존 `gello_ur_bridge`(250Hz 업샘플, slew clamp) + `gello_move_to_start` handshake + Robotiq Modbus **무변경** |
| 시작 절차 | 핸드셰이크가 HELD start pose로 팔을 파킹 → 오퍼레이터가 명시적으로 `ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger` 호출해야 자율 모션 시작 |
| receding horizon | k=30 (`n_action_steps`) — 30틱마다 net 재실행, 30Hz 발행 |
| 안전 | 속도는 브리지의 0.625 rad/s slew ceiling(불변), 위치는 `policy_leader_node`의 1.2x envelope 클램프 + live-pose 0.5rad 클램프, ZMQ 실패 **또는 관측 stale(`obs_timeout_s` 0.5s)** 시 fail-silent FAULT |
| 이 문서에서 검증됨 | 이미지 파이프라인 byte-parity, ZMQ round-trip, HOLD 유지, arming guard, 두 안전 클램프, fail-silent FAULT |
| 로봇 PC에서 검증 필요 | colcon build on Humble, 실제 handshake, 실제 카메라 시리얼, 실제 팔 동작 |

### 목차

1. [빅 픽처 / 데이터 흐름](#1-빅-픽처--데이터-흐름)
2. [하드웨어 / 환경](#2-하드웨어--환경)
3. [데이터셋 & 체크포인트 다운로드](#3-데이터셋--체크포인트-다운로드)
4. [셋업 절차 (로봇 PC)](#4-셋업-절차-로봇-pc)
5. [배포 실행하기](#5-배포-실행하기)
6. [안전 모델](#6-안전-모델)
7. [추론 동작 (receding horizon, 이미지 파이프라인)](#7-추론-동작-receding-horizon-이미지-파이프라인)
8. [검증 완료 vs 로봇 PC에서 검증 필요](#8-검증-완료-vs-로봇-pc에서-검증-필요)
9. [트러블슈팅](#9-트러블슈팅)
10. [HIL-SERL으로 가는 길](#10-hil-serl으로-가는-길)

---

## 1. 빅 픽처 / 데이터 흐름

ACT 서버(py3.12, torch+lerobot)와 `policy_leader_node`(py3.10, rclpy)는 서로 다른 파이썬/venv에 살기 때문에 **localhost ZMQ REQ/REP**로만 통신한다 — `lerobot`은 `pyproject.toml`에서 Python ≥3.12를 요구하는데 ROS2 Humble의 `rclpy`는 CPython 3.10으로 빌드되어 있어, 같은 인터프리터를 공유할 수 없다(자세한 이유는 `DEPLOY_REPO_DECISION.md` §3).

```
┌─────────────────────────────┐         ZMQ REQ/REP           ┌──────────────────────────────┐
│  act_server.py (py3.12)     │   tcp://127.0.0.1:5591         │  policy_leader_node (py3.10)  │
│  torch + lerobot 0.6.1      │◄───────────────────────────────┤  rclpy                       │
│                              │   reset: {"cmd":"reset"}       │                               │
│  ACTPolicy.from_pretrained   │──────────────────────────────► │  HOLD / EXECUTE / FAULT       │
│  n_action_steps=30           │   act: {"cmd":"act",           │  state machine                │
│  select_action (chunk queue) │        "state":[7 floats]},    │                               │
│  image resize 360x640+BGR2RGB│        <cam1 jpeg>,<cam2 jpeg> │                               │
│                              │◄─────────────────────────────  │                               │
│  reply: {"ok":true,          │   action = [q1..q6, grip_cmd]   │                               │
│          "action":[7 floats]}│                                 │                               │
└─────────────────────────────┘                                 └───────────────┬───────────────┘
                                                                                  │ publishes @30Hz
                                                                                  │ (safety-clamped)
                                                                                  ▼
                                                    /gello/joint_states  +  /robotiq_gripper/command_percent
                                                                                  │
                                                                                  ▼
                                              ┌───────────────────────────────────────────────────┐
                                              │  gello_ur_bridge (UNCHANGED)                        │
                                              │  250Hz 업샘플, One-Euro 필터, max_step_rad 슬루우클램프│
                                              │  (0.625 rad/s 상한), 0.5s staleness watchdog         │
                                              └───────────────────────┬─────────────────────────────┘
                                                                       │ /forward_position_controller/commands
                                                                       ▼
                                              ┌───────────────────────────────────────────────────┐
                                              │  gello_move_to_start handshake (UNCHANGED)          │
                                              │  실제 UR7e + Robotiq 2F-85 Modbus                     │
                                              └───────────────────────────────────────────────────┘

observation 방향 (팔 → 정책, 매 EXECUTE 틱):
  /joint_states (6 joints)  ──┐
  /robotiq_gripper/position_percent ─┤→ policy_leader_node가 state[7] 조립 → ZMQ "act" 요청
  /cam1/.../compressed (JPEG) ─┤   (raw JPEG 그대로 전달, py3.10 쪽에서 cv2/torch 사용 안 함)
  /cam2/.../compressed (JPEG) ─┘
```

핵심 포인트: `policy_leader_node`는 `gello_publisher_node`(사람 GELLO 리더)를 대체하는 **synthetic leader**일 뿐이다. 그 아래의 `gello_ur_bridge` + `gello_move_to_start` + Robotiq Modbus는 리더가 사람이든 정책이든 전혀 구분하지 못하며 완전히 동일한 코드가 돈다.

## 2. 하드웨어 / 환경

- **팔**: UR7e, `ros-humble-ur`. **그리퍼**: Robotiq 2F-85 (Modbus RTU, 드라이버가 소유한 socat 브리지 `/tmp/ttyUR` 공유).
- **카메라**: RealSense 2대, **시리얼로 바인딩** (혼동 시 정책이 조용히 열화됨 — §9 참고):
  - cam1 = Intel RealSense **D435**, 시리얼 `147122072740`
  - cam2 = Intel RealSense **D435if**, 시리얼 `243222072700`
  - 공통 컬러 프로파일 `1280x720x30` (해상도/FPS는 두 카메라 동일해야 함)

  > **⚠️ 물리적 카메라 배치 = 학습 리그와 반드시 일치.** 하드웨어 스펙상 한 대는 **삼각대(tripod)에 올려 씬/3인칭 시점**을, 다른 한 대는 **작업공간을 근접(close-up)**으로 본다. **두 물리 시점과 cam1/cam2 시리얼 할당이 학습 당시 리그와 동일**해야 하며, 어긋나면 정책이 **에러 없이 조용히 열화**된다(정책은 cam1=씬, cam2=근접 같은 고정 배치를 가정하고 학습됨). 실기 첫 구동 전에 **라이브 뷰(`rqt_image_view` 등)를 학습 셋업 사진과 대조**해 어느 카메라가 어느 시점인지, 시리얼 할당이 맞는지 확인한 뒤에야 정책을 신뢰할 것. (§9의 cam1/cam2 swap 경고와 함께 읽을 것.)

- **ROS2**: 로봇 PC는 **Humble** (rclpy py3.10). `ur_gello_bringup`/`gello_recorder`와 동일 워크스페이스. 필요한 ROS 패키지 설치:

  ```bash
  # UR 드라이버 + RealSense 카메라 스택 (+ librealsense2 SDK)
  sudo apt install ros-humble-ur ros-humble-realsense2-camera ros-humble-librealsense2
  # 워크스페이스 나머지 의존성 (ur_robot_driver 등) — dynamixel_sdk는 pip이라 skip
  cd gello_software/ros2_ur_ws
  rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
  ```

  > 실 로봇/ROS 최초 셋업 전체(펜던트 External Control 페어링, 기구학 캘리브레이션, 24V 그리퍼)는 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)를 참고. 이 문서는 그 위에 ACT 배포만 얹는다.

- **ACT 서버**: 별도 **py3.12 venv** (torch cu128 빌드 + lerobot 0.6.1). **GPU(CUDA) 필수 (권장이 아님).** `--device cuda`가 기본값이며, CUDA가 없으면 서버는 **자동 CPU 폴백을 하지 않고 종료 코드 3으로 거부**한다 — CPU ACT forward는 leader의 0.5s `act_timeout_s`를 넘겨 FAULT-loop를 유발하기 때문이다. CPU에서 굳이 돌리려면 `ACT_DEVICE=cpu`를 명시하고 동시에 `act_deploy.yaml`의 `act_timeout_s`를 크게 올려야 한다(§9 참고).

## 3. 데이터셋 & 체크포인트 다운로드

학습에 쓰인 Hugging Face 리포:

| 종류 | HF repo id | 비고 |
|---|---|---|
| 체크포인트 | `Bigenlight/act_banana_in_pot` | `--checkpoint` / `ACT_CHECKPOINT`로 넘길 것 |
| 데이터셋 (joint-space) | `Bigenlight/banana_in_pot_lerobot_v3` | 학습에 실제로 쓰인 LeRobot v3 데이터셋 |
| 데이터셋 (EE-space) | `Bigenlight/banana_in_pot_ee_lerobot_v3` | HIL-SERL/EE 실험용 변환 버전 |
| 원본 (raw) | `Bigenlight/banana_in_pot_raw` | HDF5 원본 로그 (recorder 출력) |

`scripts/download_checkpoint.sh`가 체크포인트(+ 옵션으로 데이터셋)를 gitignored 로컬 디렉터리로 받아온다:

```bash
cd gello_software/ros2_ur_ws/src/gello_policy

# 체크포인트만 (기본):
./scripts/download_checkpoint.sh

# 데이터셋도 함께:
WITH_DATASET=1 ./scripts/download_checkpoint.sh

# 바로 쓰기 (stdout 마지막 줄이 --checkpoint에 넘길 경로):
CKPT=$(./scripts/download_checkpoint.sh | tail -1)
ACT_CHECKPOINT="$CKPT" ./scripts/run_act_server.sh
```

환경변수: `ACT_VENV`(huggingface-cli를 찾을 py3.12 venv, 기본 `ros2_ur_ws/act_venv`), `DOWNLOAD_DIR`(기본 `<pkg>/checkpoints`), `CKPT_REPO`, `DATASET_REPO`, `WITH_DATASET`. `huggingface-cli`/`hf`가 `ACT_VENV`와 `PATH`에서 검색되며, 없으면 `pip install -U 'huggingface_hub[cli]'`를 안내하고 종료한다.

## 4. 셋업 절차 (로봇 PC)

1. **colcon 빌드 — `gello_policy` + `ur_gello_bringup`** (Humble, py3.10). `ur7e_act_real.launch.py`는 `ur_gello_bringup` 노드(`gello_ur_bridge`, `gello_move_to_start`, `robotiq_gripper_modbus`)를 돌리고 `ur_robot_driver`의 `ur_control.launch.py`를 include하므로, **`gello_policy`만 빌드하면 안 되고 `ur_gello_bringup`도 반드시 함께 빌드**해야 한다. `gello_policy`는 `ur_gello_bringup` + `ur_robot_driver`를 `exec_depend`로 선언하므로 `rosdep install --from-paths src`가 드라이버까지 끌어온다. (`policy_leader_node`만 이 빌드에 포함되며, `policy_server/`는 **설치되지 않고** 자기 venv에서 소스 디렉터리째로 실행됨.)

   ```bash
   cd gello_software/ros2_ur_ws
   # 의존성 해결 (ur_robot_driver, realsense2_camera 등); dynamixel_sdk는 pip이라 skip
   rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
   # 두 패키지 명시 빌드...
   colcon build --packages-select gello_policy ur_gello_bringup
   # ...또는 가장 간단하게 워크스페이스 전체:  colcon build
   source install/setup.bash
   ```

   > `ur_gello_bringup`의 `ros2_ur_ws/build_ur7e.sh`도 동일한 `rosdep install --skip-keys dynamixel_sdk` + `colcon build`를 수행하므로, 그 스크립트를 먼저 한 번 돌려 드라이버 의존성을 깔아두는 것과 동등하다.

2. **py3.12 venv 생성 + ACT 서버 의존성 설치** (`policy_server/requirements-act.lock` 참고 — torch는 **cu128** 빌드이므로 별도 index-url 필요). venv는 **`ros2_ur_ws/act_venv`에** 만든다 — 이것이 세 run 스크립트가 기대하는 기본 위치(`ACT_VENV=ros2_ur_ws/act_venv`)다. 다른 곳에 두려면 `export ACT_VENV=/your/path`:

   ```bash
   cd gello_software/ros2_ur_ws        # 여기서 venv 생성
   python3.12 -m venv act_venv
   act_venv/bin/pip install --upgrade pip
   act_venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   act_venv/bin/pip install -r src/gello_policy/policy_server/requirements-act.lock
   ```

   > **첫 실행 캐시 노트.** ACT 백본은 ImageNet 사전학습 ResNet18을 쓴다. 오프라인 로봇 PC(또는 방화벽 뒤)에서 첫 실행 시 torchvision이 이 가중치를 인터넷에서 받으려다 멈추거나 실패할 수 있다. 미리 `TORCH_HOME`을 세팅해 캐시 위치를 고정하고, 캐시가 이미 채워져 있다면 `HF_HUB_OFFLINE=1`도 함께 export할 것. **이 두 변수는 `run_ur7e_act_real.sh`를 실행하는 바로 그 셸에서 export해야 한다** — 스크립트는 환경을 상속만 하고 스스로 세팅하지 않는다:
   >
   > ```bash
   > # run_ur7e_act_real.sh를 띄우는 것과 같은 셸에서:
   > export TORCH_HOME=/path/to/torch_cache   # resnet18 가중치가 저장/캐시될 위치
   > export HF_HUB_OFFLINE=1                  # 체크포인트가 이미 로컬에 있다면 HF 접근 자체를 스킵
   > ```

3. **RealSense 카메라 2대를 정확한 시리얼→네임스페이스 매핑으로 기동.** 별도 launch 파일이 없으므로 `ros2_ur_ws/run_recorder.sh`(레코더)가 카메라를 띄우는 방식(시리얼별 `realsense2_camera` 노드, `serial_no:='<serial>'`처럼 **작은따옴표로 감싼** all-digit 문자열, 공통 `color_profile:='1280x720x30'`)을 그대로 재사용한다:

   ```bash
   # cam1 (D435, 147122072740) → 네임스페이스 cam1
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam1 camera_namespace:=cam1 \
       serial_no:="'147122072740'" rgb_camera.color_profile:="'1280x720x30'" &
   # cam2 (D435iF, 243222072700) → 네임스페이스 cam2
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam2 camera_namespace:=cam2 \
       serial_no:="'243222072700'" rgb_camera.color_profile:="'1280x720x30'" &
   ```

   기동 후 압축 이미지 토픽이 실제로 나오는지 확인:

   ```bash
   ros2 topic hz /cam1/cam1/color/image_raw/compressed
   ros2 topic hz /cam2/cam2/color/image_raw/compressed
   ```

## 5. 배포 실행하기

```bash
cd gello_software/ros2_ur_ws
ACT_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_act_real.sh
```

`run_ur7e_act_real.sh`는 먼저 py3.12 ACT 서버를 백그라운드로 띄우고(PID 저장, `trap 'kill $PID' EXIT`로 Ctrl-C 시 함께 정리), 서버가 살아있고 포트가 listen 중인지 확인한 뒤, Humble `ros2 launch gello_policy ur7e_act_real.launch.py`를 실행한다.

- **Method A (기본)**: 펜던트에서 External Control 프로그램을 로드하고 **Play** — `HEADLESS` 미설정/`false`.
- **Method B (헤드리스)**: `HEADLESS=true ACT_CHECKPOINT=/path/... ./run_ur7e_act_real.sh` — 펜던트 Play 없이 드라이버가 직접 URScript를 보냄. 로봇이 **REMOTE 모드**여야 함.

> **⚠️ 카메라를 먼저 띄울 것.** ACT 배포는 카메라 없이는 돌지 않는다: EXECUTE는 매 틱 fresh `cam1`/`cam2` 프레임을 요구하고, `~/start_execution`과 in-EXECUTE obs-freshness watchdog이 카메라가 없거나 stale하면 각각 arming을 거부/FAULT시킨다(§6). `run_ur7e_act_real.sh` 실행 **전에** §4의 카메라 기동 단계(3번)의 명령으로 두 RealSense를 시리얼 바인딩으로 띄우고 `ros2 topic hz`로 두 압축 토픽이 ~30Hz로 나오는지 확인하라.

> **Dry-run / mock (실팔 없이) — 원커맨드가 아님.** `use_fake_hardware:=true` **하나만으로는** mock 팔이 ACT로 움직이지 않는다. fake hardware에서는 Robotiq 노드가 건너뛰어져 `/robotiq_gripper/position_percent`를 발행하는 publisher가 아무도 없고 → `grip_pos` 관측이 빠지고 → EXECUTE가 obs-freshness watchdog에서 **FAULT**한다. mock 하드웨어로 전체 자율 경로를 dry-run하려면 (1) fake-hardware launch에 더해 (2) 두 RealSense 카메라를 **실제로** 띄우고, (3) 그리퍼 위치 토픽을 **가짜로** 발행해야 한다:
>
> ```bash
> # use_fake_hardware:=true launch + 두 RealSense 카메라가 돌고 있는 상태에서:
> ros2 topic pub /robotiq_gripper/position_percent std_msgs/Float32 "{data: 0.0}" -r 10
> ```
>
> 그런 다음에야 `~/start_execution`이 arming을 허용한다. 즉 mock 자율 경로는 절대 한 줄짜리가 아니다.

### 시작 handshake 타임라인

1. **t=0s** — `ur_control.launch.py`: UR7e 드라이버 + `scaled_joint_trajectory_controller` active, `forward_position_controller` inactive (로드만).
2. **t=6s** — `policy_leader_node` 기동, **HOLD** 상태로 진입. 매 틱 고정 `start_pose`
   `q = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` (rad, UR 관절 순서)를 완전히 정지된 상태로 발행 — 서버는 아직 쿼리하지 않음. 동시에 `gello_ur_bridge`가 `start_paused:=true`로 pre-spawn (아무것도 발행 안 함).
3. **t=8s** — `gello_move_to_start`가 이 HELD start pose로 수렴 게이트 chase 핸드셰이크를 수행 (자세한 chase/dwell/stillness-gate 메커니즘은 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2 참고 — 로직은 100% 동일하며, 다른 점은 리더가 사람 손이 아니라 이 고정 pose라는 것뿐). 수렴하면 STRICT 컨트롤러 전환 → 브리지 `~/resume` 호출.
4. **팔이 start_pose에서 정지(파킹)한다.** 이 시점까지 **자율 모션은 전혀 없다** — 리더가 완전히 정지해 있기 때문.
5. **오퍼레이터가 명시적으로** 자율 실행을 시작한다:

   ```bash
   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
   ```

   이 호출은 (a) 라이브 팔 자세가 `start_pose`에서 ~0.1 rad 이내인지 확인하고(핸드셰이크가 실제로 수렴했는지 재확인하는 가드), (b) ZMQ `reset`을 서버에 보내 ACT의 action queue를 비운 뒤, (c) **EXECUTE**로 진입한다. 이후부터 매 틱 서버를 쿼리하고 클램프된 타깃을 발행한다 — 이때부터가 진짜 자율 모션 시작이다.

6. **언제든 일시정지(pause)하려면**:

   ```bash
   ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
   ```

   현재 라이브 자세를 그대로 HOLD로 전환해 그 자리에서 팔을 정지시킨다. **그리퍼는 `start_gripper`(open)로 되돌아가지 않고 마지막으로 명령된 값을 그대로 유지**한다 — 즉 그랩 도중에 `~/hold`를 걸어도 물체를 놓치지 않고 잡은 상태를 유지한다(부팅 시점의 HOLD만 `start_gripper=0.0=open`을 씀). 재시작 시 arming guard는 여전히 `start_pose` 기준으로 평가되므로(다시 EXECUTE로 가려면 라이브 자세가 원래 `start_pose` ±0.1rad 근처여야 함), `~/hold`로 정지시킨 위치가 이미 그 근처가 아니면 재-arm이 거부될 수 있음에 유의.

## 6. 안전 모델

이 정책은 실제 로봇을 자율 구동하므로 이 섹션을 가볍게 넘기지 말 것.

- **속도 상한 — 절대 넘지 않음.** `policy_leader_node`가 무엇을 발행하든, 그 아래 **`gello_ur_bridge`는 무변경**이며 자신의 `max_step_rad` slew 클램프(배포 설정 `0.0025 rad` @ `250 Hz` = **지속 0.625 rad/s**, coalescing 최악 케이스 2.5 rad/s, UR 드라이버 한계 3.14 rad/s 대비 ~20% 마진)를 무조건 적용한다. ACT가 아무리 급격한 타깃을 내도 팔은 이 속도 이상으로 움직이지 않는다.
- **위치 클램프 — `policy_leader_node`가 매 EXECUTE 틱마다, 이 순서로:**
  1. **(a) 1.2x 데이터셋 envelope로 clip** (`joint_limits_lo`/`joint_limits_hi`, OOD 출력 가드):

     | 관절 | lo (rad) | hi (rad) |
     |---|---|---|
     | shoulder_pan | 2.40 | 3.67 |
     | shoulder_lift | -2.53 | -0.88 |
     | elbow | 1.05 | 2.62 |
     | wrist_1 | -3.34 | -1.01 |
     | wrist_2 | -2.18 | -1.25 |
     | wrist_3 | -5.22 | -1.52 |

     (`config/joint_safety_limits.json`에 원본 min/max와 함께 기록됨 — 관측/커맨드 union의 중심을 기준으로 대칭 1.2배 확장.)
  2. **(b) live `/joint_states` 대비 `max_dev_rad`(기본 0.5 rad)로 clip** — 실제 자세에서 한 틱에 튈 수 있는 최대량을 제한.

  둘 중 하나라도 발동하면 throttled `WARN`을 로그로 남긴다(어느 클램프가 발동했는지 구분해서).

- **ZMQ 실패 시 fail-silent FAULT — never hold-forever.** 서버 타임아웃(`act_timeout_s`, 기본 0.5s)이나 에러, 또는 `{"ok":false}` 응답을 받으면 `policy_leader_node`는 즉시 **FAULT**로 전이하고 `/gello/joint_states` 발행을 **완전히 중단**한다. 이는 의도적으로 `gello_ur_bridge`의 0.5s staleness watchdog을 트립시켜 브리지가 스트리밍을 멈추고 팔이 그 자리에 정지하게 만든다. 복구 시 브리지는 실제 라이브 자세에서 soft-start로 재-seed한다. **마지막 유효 타깃을 계속 재발행하는 방식은 절대 쓰지 않는다** — 그렇게 하면 장애 중 벌어진 gap을 복구 시 풀 슬루우로 닫아버리는 위험한 동작이 된다.

  > **FAULT 복구 절차 (review L2).** 에피소드 도중 FAULT가 발생하면 팔은 **그 자리에서 그대로 정지**한다. 재-arm은 라이브 자세가 다시 `start_pose` 근처(~0.1 rad 이내)여야만 허용된다 — `~/start_execution`의 arming guard가 이를 강제하기 때문이다. 즉 **"거기서부터 이어서 재개(resume-from-here)"는 불가능**하다. 복구 방법은 둘 중 하나:
  > 1. `~/hold`를 호출해 현재 자세를 HOLD로 만든 뒤, 팔을 (수동 조그 또는 재-handshake로) `start_pose` 근처로 되돌리고, 다시 `~/start_execution`을 호출한다.
  > 2. launch를 재기동해 handshake를 처음부터 다시 태운다(팔이 이미 `start_pose`에서 멀지 않다면 chase가 금방 수렴한다).
  >
  > 두 경로 모두 "FAULT 지점에서 이어서 계속" 하지 않고 항상 알려진 안전한 시작 자세로 되돌아간다는 점을 기억할 것.

- **Observation freshness watchdog — 얼어붙은 카메라도 FAULT로.** EXECUTE 상태에서, 4개 관측(`joint_states`, `grip_pos`, `cam1`, `cam2`) 중 **하나라도 없거나 `obs_timeout_s`(기본 0.5s, `act_deploy.yaml`의 신규 파라미터)보다 오래된** 프레임이면 노드는 즉시 **FAULT**로 전이해 `/gello/joint_states` 발행을 중단한다(→ 브리지 staleness watchdog이 팔을 정지). 이는 카메라 드라이버가 hang해서 **낡은(stale) 프레임으로 정책이 계속 돌아가는** 상황을 잡아낸다(ZMQ는 멀쩡히 응답하므로 §위의 ZMQ 타임아웃만으로는 못 잡는 케이스). **오퍼레이터 결과: 에피소드 도중 카메라가 죽으면 팔은 FAULT(정지)하고 자동 재개하지 않는다** — 다른 모든 FAULT와 동일하게 `start_pose` 근처에서 `~/start_execution`으로 재-arm해야 한다. 또한 `~/start_execution`은 이제 **완전한 fresh 관측 셋이 모두 존재할 때만** arming을 허용한다 — 즉 **카메라가 먼저 돌고 있어야** 실행을 시작할 수 있다(카메라 없이 arm하면 즉시 FAULT).
- **`~/hold`는 마지막 명령 그리퍼 값을 유지한다.** 일시정지 시 팔은 현재 라이브 자세에서 정지하되, 그리퍼는 `start_gripper`(open)로 되돌아가지 않고 **마지막으로 명령된 값을 그대로 잡고 있는다** — 그랩 도중 `~/hold`를 걸어도 물체를 놓치지 않는다. (부팅 시점의 HOLD만 `start_gripper=0.0=open`을 사용한다.)

- **오퍼레이터 게이트.** 핸드셰이크만으로는 절대 자율 모션이 시작되지 않는다 — 리더가 완전히 정지된 `start_pose`를 발행하는 동안만 chase가 일어나기 때문이다. `~/start_execution`을 명시적으로 호출해야 EXECUTE로 넘어간다(`auto_start_on_stream:=true`가 아닌 한).
- **첫 추론 워밍업.** ACT 서버는 로드 시점에 더미 관측으로 2회 추론을 미리 돌려 CUDA 커널을 컴파일/워밍업한다. 이렇게 하지 않으면 실제 첫 EXECUTE 틱에서 추론이 `act_timeout_s`(0.5s)를 넘겨 **가짜(spurious) FAULT**가 발생할 수 있다.
- **단일 그리퍼 writer.** `ur7e_act_real.launch.py`는 `gello_gripper_bridge`를 launch하지 않는다 — `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher다(이중 writer로 인한 충돌 방지).

## 7. 추론 동작 (receding horizon, 이미지 파이프라인)

- **Receding horizon k=30.** `act_server`는 로드 시 `policy.config.n_action_steps = 30`을 **`policy.reset()` 이전에** 설정한다(`reset()`이 `maxlen=n_action_steps`인 action deque를 만들기 때문 — 나중에 바꿔도 이미 만들어진 큐에는 반영되지 않는다). 결과적으로 net은 **30번의 `act` 호출마다** 재실행되고(fresh observation 기준), 그 사이에는 이전 청크에서 pop만 한다. `temporal_ensemble_coeff`는 이 체크포인트에서 `None`이어야 하며(청크 앙상블 없음 = 재쿼리가 유일한 피드백 채널), 서버가 로드 시 이를 assert로 확인한다.
- **30Hz 발행은 협상 불가.** 데이터셋 자체가 30Hz로 기록되었고 브리지의 필터 튜닝도 ~30Hz 리더를 가정한다. 이보다 빠르거나 느리게 발행하면 정책이 학습한 것과 다른 유효 다이나믹스가 된다.
- **이미지 전처리 — 필수, 체크포인트에 없음.** 저장된 preprocessor는 rename→batch→device→normalize 4단계뿐이며 **리사이즈도 색공간 변환도 하지 않는다.** 360x640 리사이즈와 BGR→RGB는 학습 시 dataloader 단(`image_transforms`)에서 적용됐다. 그래서 `image_preprocess.py`가 서버 측에서 반드시: `cv2.imdecode` → **BGR→RGB** → **torchvision `Resize([360,640])`**(bilinear + antialias, `eval_offline.py`의 `build_image_transforms()`와 동일한 단일 결정적 변환) 를 수행한 **뒤에** 저장된 preprocessor가 실행된다. 720x1280을 그대로 먹이거나 BGR을 남겨두면 **에러 없이 조용히** 정책이 열화된다.
- **카덴스(재쿼리 주기) 조정 가이드.** 기본은 `k=30`(1.0s). 그랩(grasp)을 자주 놓치는 등 **반응이 느려서** 실패하면 `k`를 **10~15**로 낮춘다(ACT forward는 CUDA에서 10~50ms라 재쿼리는 저렴함 — 30ms 스텝 예산 안에 충분히 들어옴). **청크 경계 아티팩트가 확실히 지배적**이고 반응성은 이미 충분함이 증명된 경우에만 `k`를 **100** 쪽으로 올린다. `n_action_steps`은 서버의 `--n-action-steps` CLI 플래그(또는 `$ACT_N_ACTION_STEPS`)로 조정한다:

  ```bash
  ACT_CHECKPOINT=/path/... ACT_N_ACTION_STEPS=15 ./scripts/run_act_server.sh
  ```

## 8. 검증 완료 vs 로봇 PC에서 검증 필요

**개발 PC에서 오프라인으로 검증됨** (로봇/카메라 없이):

- 이미지 파이프라인 byte-parity (`image_preprocess.py` vs `eval_offline.py`의 `build_image_transforms()`)
- ZMQ multipart round-trip (더미 클라이언트)
- HOLD가 start_pose를 정확히 유지하는지
- arming guard (`~/start_execution`의 라이브 자세 0.1rad 가드)
- 두 안전 클램프 (OOD → clip, max_dev → clip)
- fail-silent FAULT 전이 (ZMQ 타임아웃/에러 시 발행 중단)

**로봇 PC에서 반드시 검증할 것** (이 개발 PC에는 로봇도 카메라도 없음):

- Humble 위에서의 `colcon build --packages-select gello_policy`
- 실제 handshake (실기 JTC 정상상태 오차, `chase_tol` 체인 — [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2의 캘리브레이션 체크리스트 참고)
- 실제 카메라 시리얼 매핑 (cam1/cam2가 실제로 올바른 물리 카메라인지)
- 실제 팔 동작 (그랩 성공률, 청크 경계에서의 부드러움, 그리퍼 크러시 여부)

## 9. 트러블슈팅

- **cam1/cam2 시리얼이 뒤바뀜.** 반드시 **시리얼로 바인딩**할 것(`serial_no`) — 이름/포트 순서에 의존하면 두 RealSense를 구분할 수 없다. 잘못 바인딩되면 정책은 크래시하지 않고 **조용히 열화**된다(어느 카메라가 어느 관측 슬롯인지 학습 시와 달라지므로).
- **all-digit 시리얼의 정수 강제변환 버그.** `ros2 launch`의 CLI 인자 파서는 전부 숫자인 문자열(예: `147122072740`)을 정수로 오추론해 `serial_no`(string 파라미터)에 넣으려다 노드가 즉시 죽는다. 값을 **따옴표로 감쌀 것**: `serial_no:="'147122072740'"` (자세한 배경은 [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md)의 troubleshooting 항목 7 참고).
- **그리퍼 크러시(과도한 힘으로 닫힘).** `policy_leader_node`는 `action[6]`(0=open..1=closed)을 **그대로(identity)** `/robotiq_gripper/command_percent`에 발행한다 — threshold나 binarize를 절대 넣지 말 것(데이터셋의 grip_cmd가 연속값이므로). `ur_gello_bringup`의 `gello_gripper_bridge_node`에는 방향을 뒤집는 `invert` 파라미터(반드시 `false`로 유지해야 하는, crush hazard가 있는 옵션)가 있지만, **ACT 배포 경로는 이 노드를 아예 launch하지 않는다** — `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher이고 그 사이에 invert 로직이 전혀 없다. 만약 launch 파일을 손대다 `gello_gripper_bridge`를 다시 끼워 넣으면 이중 writer + 잠재적 invert 위험이 함께 재발하므로, 트러블슈팅 시 **`gello_gripper_bridge`가 여전히 빠져 있는지**부터 확인할 것.
- **첫 실행 시 torch 캐시 관련 멈춤/실패.** §4의 `TORCH_HOME`/`HF_HUB_OFFLINE=1` 참고 — ResNet18 백본 가중치를 오프라인 환경에서 인터넷으로 받으려다 멈추는 경우가 흔하다.
- **가짜(spurious) FAULT가 시작 직후에 뜬다.** 서버의 워밍업이 실패했거나 건너뛰어졌을 가능성. 서버 로그에 `warmup done`이 찍혔는지 확인.
- **`~/start_execution`이 계속 거부됨.** 라이브 자세가 `start_pose`에서 0.1rad 넘게 떨어져 있거나, **fresh 관측 셋이 완전하지 않다**는 뜻 — handshake가 실제로 수렴(`Converged` 로그)했는지, `~/hold`로 다른 자세에 파킹된 상태는 아닌지, 그리고 **두 카메라 + 그리퍼 위치 토픽이 모두 발행 중인지**(§5의 카메라 기동, mock이면 그리퍼 위치 fake 발행) 확인.
- **ACT 서버가 종료 코드 3 / "CUDA unavailable"로 죽음.** `act_server`의 `resolve_device()`는 `--device cuda`인데 CUDA가 없으면 **자동 CPU 폴백을 하지 않고 코드 3으로 거부**한다(CPU ACT forward가 0.5s 타임아웃을 넘겨 FAULT-loop를 유발하므로 의도적). 해결: (a) CUDA/드라이버를 고치거나, (b) 정말 CPU로 돌려야 한다면 `ACT_DEVICE=cpu`를 명시하고 **동시에** `act_deploy.yaml`의 `act_timeout_s`를 크게 올릴 것(그러지 않으면 느린 CPU 추론이 매 틱 FAULT를 낸다).

## 10. HIL-SERL으로 가는 길

이 패키지는 나중에 HIL-SERL로 확장될 예정이다. 무거운 RL 부분(actor/learner, replay buffer, EE-delta processors)은 계속 lerobot의 py3.12 쪽에 **패치 없이** 얹히고, ROS 쪽에는 **`policy_mux_node`** 하나만 추가되어 실제 GELLO 리더(사람 개입)와 정책 스트림을 중재한다. UR7e를 위해서는 `send_action`/`get_observation`이 이 ZMQ 프로토콜을 말하는 커스텀 `lerobot.Robot` 서브클래스와, GELLO 리더를 개입 장치로 노출하는 `Teleoperator` shim이 필요하다(둘 다 draccus `ChoiceRegistry`로 플러그인 등록 — lerobot 포크 불필요). 자세한 근거와 리스크는 `DEPLOY_REPO_DECISION.md` §5를 참고할 것.

---

## 관련 문서

- [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) — 패키지 개요, 노드/토픽 레퍼런스, 빌드
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 사람 GELLO 리더로 하는 move-to-start handshake의 전체 메커니즘(chase/dwell/stillness-gate, 캘리브레이션 체크리스트) — 이 문서의 handshake는 리더가 정책으로 바뀌었을 뿐 로직은 동일
- [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) — 이 정책을 학습시킨 데이터셋을 만든 레코더 (observation/action 컨트랙트의 출처)
