# GELLO → UR7e ACT 정책 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 학습된 ACT "put right banana in pot" 정책으로 실제 UR7e를 자율 구동하기 위한 실기(real-robot) 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)입니다. **2026-07-08, 실제 로봇 PC에서 콜드스타트 절차(colcon build on Humble, 실제 핸드셰이크, 실제 카메라 시리얼, 실제 팔 동작) 전체가 처음으로 실기 검증되었습니다** — §4.5, §5, §8 참고. 이미지 파이프라인 byte-parity, ZMQ 라운드트립, HOLD 유지, arming guard, 두 안전 클램프, fail-silent FAULT 전이는 그 이전에 개발 PC에서 오프라인으로 검증되었습니다. 단, **정책의 태스크 수행 신뢰도(그랩 성공률 등)는 아직 특성화되지 않았습니다** — §8 참고.
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
| 오프라인 검증됨 (개발 PC) | 이미지 파이프라인 byte-parity, ZMQ round-trip, HOLD 유지, arming guard, 두 안전 클램프, fail-silent FAULT |
| 실기 검증됨 (로봇 PC, 2026-07-08) | Humble colcon build, 실제 handshake(첫 시도 수렴, 튜닝 불필요), 실제 카메라 시리얼(cam1/cam2 ~30Hz), 실제 그리퍼(auto-cal, gFLT:0), GPU 추론, 엔드투엔드 자율 시도 1회 — **시스템 배관 레벨** |
| 아직 특성화 안 됨 | 그랩 성공률(첫 시도 미완수), 청크 경계 부드러움, 그리퍼 크러시 — **정책 태스크 성능은 미검증, 후속 튜닝/반복 예상** |

### 목차

1. [빅 픽처 / 데이터 흐름](#1-빅-픽처--데이터-흐름)
2. [하드웨어 / 환경](#2-하드웨어--환경)
3. [데이터셋 & 체크포인트 다운로드](#3-데이터셋--체크포인트-다운로드)
4. [셋업 절차 (로봇 PC)](#4-셋업-절차-로봇-pc)
5. [배포 실행하기](#5-배포-실행하기)
6. [안전 모델](#6-안전-모델)
7. [추론 동작 (receding horizon, 이미지 파이프라인)](#7-추론-동작-receding-horizon-이미지-파이프라인)
8. [검증 완료 (실기 포함) vs 아직 특성화 안 됨](#8-검증-완료-실기-포함-vs-아직-특성화-안-됨)
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

> ### 🔧 카메라 시리얼 정정 (2026-07-28)
>
> 카메라 **개체가 물리적으로 교체됐다.** 이전 판의 `147122072740` / `243222072700`은
> 이 PC가 커널 로그상 한 번도 열거한 적 없는 하드웨어다(2026-07-05까지 소급 확인).
> 이 문서의 시리얼은 실제 연결된 개체(`151623020789` / `322743060038`)로 갱신했다.
> **없는 시리얼로 바인딩하면 조용히 안 뜬다** — 오류가 아니라 "프레임 없음"으로 보인다.
> 모델 클래스(D435 / D435if)와 cam1·cam2 배정은 그대로지만, **어느 개체가 손목에 달렸는지는
> 미확정**이다. 팔을 흔들어 cam2 화면에서 손가락이 고정되는지 확인할 것.
> 아래 "검증 완료" 류의 과거 기록은 **옛 개체로 수행된 것**이라 그대로 두었다.

- **팔**: UR7e, `ros-humble-ur`. **그리퍼**: Robotiq 2F-85 (Modbus RTU, 드라이버가 소유한 socat 브리지 `/tmp/ttyUR` 공유).
- **카메라**: RealSense 2대, **시리얼로 바인딩** (혼동 시 정책이 조용히 열화됨 — §9 참고):
  - cam1 = Intel RealSense **D435**, 시리얼 `151623020789`
  - cam2 = Intel RealSense **D435if**, 시리얼 `322743060038`
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
   # cam1 (D435, 151623020789) → 네임스페이스 cam1
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam1 camera_namespace:=cam1 \
       serial_no:="'151623020789'" rgb_camera.color_profile:="'1280x720x30'" &
   # cam2 (D435if, 322743060038) → 네임스페이스 cam2
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam2 camera_namespace:=cam2 \
       serial_no:="'322743060038'" rgb_camera.color_profile:="'1280x720x30'" &
   ```

   기동 후 압축 이미지 토픽이 실제로 나오는지 확인:

   ```bash
   ros2 topic hz /cam1/cam1/color/image_raw/compressed
   ros2 topic hz /cam2/cam2/color/image_raw/compressed
   ```

### 4.5 이 로봇 PC에서 실제로 겪은 환경 이슈 (2026-07-08 실기 셋업 로그)

> 이 절은 **이 머신에서** ACT 배포 환경을 실제로 처음 세팅하며 부딪힌 환경 이슈와 그 해결을 기록한다. 위 §4의 정규 절차가 "무엇을 해야 하는가"라면, 이 절은 "이 PC에서는 그 과정에서 무엇이 실제로 막혔는가"다. 다음에 이 머신에서 GPU/ACT 작업을 하는 오퍼레이터가 같은 디버깅을 반복하지 않도록 하기 위한 것이다.

**1) GPU 드라이버 vs PREEMPT_RT 커널 충돌 — 가장 먼저 부딪히는 벽.**
이 머신은 실시간 로봇 제어용 커스텀 **PREEMPT_RT 커널(`6.8.2-rt11`)로 부팅되는 것이 기본값**이다. 그런데 NVIDIA 독점 드라이버는 **어떤** PREEMPT_RT 커널에서도 DKMS 커널 모듈 빌드를 거부한다 — 하드코딩된 sanity check가 `Failed PREEMPT_RT sanity check. Bailing out!`을 내며 죽는다. **이는 드라이버 버전과 무관하게 모든 버전에서 동일하며, 버그가 아니라 의도된 거부다.** 즉 RT 커널로 부팅된 채로는 `nvidia-smi`가 절대 뜨지 않고, 따라서 ACT 추론용 CUDA도 못 쓴다.

한편 GELLO↔UR7e teleop 스택 코드를 감사해 본 결과, **이 스택이 실제로 PREEMPT_RT를 요구한다는 증거는 없었다** — 평범한 ROS2 rate(리더 30Hz, 브리지 250Hz, UR servo cycle 500Hz)로 돌고, jitter를 견디는 watchdog / slew 클램프가 이미 설계에 들어가 있다. 그래서 **GPU 작업을 할 때만 non-RT(generic) 커널로 부팅하면** teleop 기능을 잃지 않고 CUDA를 되찾을 수 있다.

다행히 이 머신에는 **이미 non-RT 커널(`6.8.0-124-generic`)이 설치돼 있어서 새 커널 설치는 필요 없었다.** 이 커널로 부팅하는 법: GRUB 메뉴에서 **"Advanced options for Ubuntu" → "Ubuntu, with Linux 6.8.0-124-generic"** 를 선택한다.

> **⚠️ 기본값을 영구 변경하려 했으나 이 머신의 GRUB 설정 특성상 실패하여, 결국 매번 GRUB 메뉴에서 수동으로 generic 커널을 선택하는 방식으로 확정했다.** 즉 **RT 커널이 여전히 자동 기본값**이고, GPU 작업이 필요할 때만 오퍼레이터가 부팅 시 GRUB 메뉴에서 generic 커널을 직접 골라야 한다. (named menu-entry ID가 이 시스템의 `grub.cfg`에서 조용히 매칭 실패하는 별개의 GRUB 문제였고, 파고들 가치가 없어 이 수동 방식으로 확정.)

generic 커널로 부팅한 **뒤에** NVIDIA 모듈을 살린다:

```bash
sudo dpkg --configure -a   # RT 커널에서 미뤄졌던 NVIDIA DKMS 빌드를 마무리 (non-RT에서는 성공)
sudo modprobe nvidia       # 현재 세션에 모듈 로드 (이후 generic 부팅 시엔 자동 로드됨)
nvidia-smi                 # 확인
```

이 머신에서 확인된 결과: **NVIDIA GeForce RTX 3060 Mobile/Max-Q, driver 595.71.05, CUDA 13.2, VRAM 6GB.** ACT 추론은 ResNet18 백본 정책의 **batch-size-1 inference**만 필요하고 학습이 아니므로, 6GB로 차고 넘친다.

**2) `python3.12`가 stock Ubuntu 22.04에는 없다.**
`gello_policy`의 ACT 서버는 py3.12를 요구하는데(§1, `lerobot`이 Python ≥3.12) 22.04 기본 저장소에는 3.12가 없다. deadsnakes PPA로 설치:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

**3) rosdep이 플래그한 실제 시스템 의존성은 `python3-h5py` 하나뿐이었다.**
§2/§4의 `rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk`가 이 워크스페이스에서 실제로 잡아낸 시스템 의존성은 `python3-h5py` 단 하나다(`gello_recorder`가 쓰는 것으로, `gello_policy` 빌드 자체와는 무관하지만 같은 워크스페이스라 함께 걸린다):

```bash
sudo apt install -y python3-h5py
```

**4) 체크포인트 다운로드 — `hf` vs `huggingface-cli` 함정.**
`pip install --user 'huggingface_hub[cli]'`는 **정상 동작하는 `hf`** 와 **아무 것도 안 하는 deprecated `huggingface-cli` shim** 을 **둘 다** 깐다. 후자는 deprecation 경고만 찍고 **아무 작업도 없이 exit 0** 으로 조용히 끝난다. 그런데 `scripts/download_checkpoint.sh`의 CLI 탐지 루프(`for cand in ... huggingface-cli hf`)는, 둘 다 `$ACT_VENV` 밖(bare PATH)에 있으면 **`hf`보다 망가진 `huggingface-cli`를 먼저 찾는다.** 결과적으로 스크립트는 **"성공"(exit 0)한 것처럼 보이는데 실제로는 아무것도 받지 않아** 체크포인트 디렉터리가 텅 빈 채로 남는다.

> **⚠️ 조용한 no-op이라 특히 위험하다** — 에러가 없어 다음 단계까지 갔다가 로드 시점에서야 빈 디렉터리로 실패한다. 우회책은 둘 중 하나: (a) 래퍼 스크립트 대신 `hf download <repo> --local-dir <dir>`를 **직접** 호출하거나, (b) `ACT_VENV`가 **`hf`만 있는** venv를 가리키게 해서 탐지 루프가 `hf`를 집게 한다.

**5) `lerobot==0.6.1`(락파일에 핀됨)이 PyPI에 존재하지 않는다.**
`policy_server/requirements-act.lock`은 `lerobot==0.6.1`로 핀돼 있는데, **PyPI에 published된 최신은 `0.6.0`** 이고(`pip index versions lerobot`로 확인), `github.com/huggingface/lerobot`에도 **`v0.6.1` 태그가 없다.** `0.6.1`은 단지 `v0.6.0` 태그를 자른 직후 lerobot `main` 브랜치의 `pyproject.toml`에 박혀 있는 버전 문자열일 뿐이다(`raw.githubusercontent.com/huggingface/lerobot/main/pyproject.toml`에서 `version = "0.6.1"` 확인). 따라서 PyPI가 아니라 **GitHub main에서** 설치해야 한다:

```bash
act_venv/bin/pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@main" \
    numpy==2.2.6 pillow==12.3.0 opencv-python-headless==4.13.0.92 pyzmq==27.1.0
```

> **⚠️ torch/torchvision을 건드리게 두지 말 것.** torch/torchvision은 §4 step 2에서 **cu128 index-url로 이미 별도 설치**돼 있다. 위 lerobot 설치가 이들을 다운그레이드하지 않도록 주의하고, 설치 후 **torch가 `2.11.0+cu128`로 유지되고 `torch.cuda.is_available() == True`인지** 반드시 재확인했다(이 머신에서 확인 완료).

**6) 모든 설치 후 최종 환경 sanity check (한 줄).**
위 단계가 다 끝나면 아래 한 줄로 py3.12 venv가 온전한지 확인한다 — torch+CUDA, torchvision, lerobot, zmq, cv2, numpy가 모두 임포트되고 `ACTPolicy`까지 로드되면 배포 준비 완료:

```bash
act_venv/bin/python -c "
import torch, torchvision, lerobot, zmq, cv2, numpy
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
from lerobot.policies.act.modeling_act import ACTPolicy
print('ACTPolicy import OK')
"
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

### 실전 3-터미널 워크플로우 (이 머신에서 검증된 방식)

이 머신에서는 **Method B (`HEADLESS=true`)를 기본으로 쓴다.** 사람 GELLO 텔레옵도 이미 `HEADLESS=true ./run_ur7e_gello_real.sh`로 헤드리스 구동하고 있고, UR7e 펜던트를 **Remote Control 모드로 상시 켜 둔 상태**이기 때문이다.

> **펜던트 REMOTE 모드는 한 번만 켜면 됨 (persistent).** PolyScope 5에서 `Settings > System > Remote Control > Enable`로 활성화(우측 상단 토글로 on/off). 리부트/세션이 바뀌어도 유지되는 **1회성 설정**이다. 활성 상태 표시: **Play/Load 버튼이 회색으로 비활성**되고 우측 상단에 **REMOTE**라고 뜬다.

Method A와 Method B는 **코드상 완전히 동일**하다 — `HEADLESS=true`는 teleop launch와 **byte-for-byte 같은** `headless_mode` 패스-스루로 `ur_control.launch.py`에 넘어갈 뿐이며(같은 인자명·같은 기본값·같은 fake-hardware 오버라이드), `policy_leader_node.py`는 `headless_mode`를 **읽지도 분기하지도 않는다**. 즉 HOLD→`~/start_execution`→EXECUTE 오퍼레이터 게이트, 두 위치 안전 클램프, FAULT 처리 모두 A/B가 100% 동일하다. Method B가 없애는 것은 딱 하나 — 핸드셰이크 move-to-start 모션 직전에 사람이 펜던트 Play를 누르며 갖던 암묵적 일시정지뿐이다. **진짜 자율 ACT 모션은 어느 쪽이든 여전히 명시적 `~/start_execution` 게이트 뒤에 있다** — 헤드리스라고 그 게이트를 건너뛰지 않는다.

> **⚠️ REMOTE 모드가 꺼진 채로 `headless_mode:=true`를 쓰면 실패는 조용하지 않고 시끄럽다.** `Could not send program to robot` / `Could not resend robot program` 에러가 반복되고 **모션이 전혀 없다** — 부분적으로/위험하게 동작하는 일은 없다.

아래가 실제로 쓰는 순서다. 예전에는 자율 동작 시작/정지를 위한 별도 터미널(구 Terminal 3)이 필요했지만, 지금은 그 두 서비스 호출이 **Terminal 1 카메라 뷰어 창의 버튼으로 들어와 있어** 일반적인 대화식 세션에서는 **3개 터미널이면 충분**하다. 커맨드라인 호출은 자동화·스크립트용 fallback으로 여전히 문서화해 둔다(아래 "선택 사항" 참고).

**Terminal 1 — 카메라 + 실행 제어 버튼** (§4의 3번과 동일, ACT 실행 **전에 반드시 먼저**):

```bash
source /opt/ros/humble/setup.bash
cd ~/gello_software/ros2_ur_ws
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam1 camera_namespace:=cam1 \
    serial_no:="'151623020789'" rgb_camera.color_profile:="'1280x720x30'" &
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam2 camera_namespace:=cam2 \
    serial_no:="'322743060038'" rgb_camera.color_profile:="'1280x720x30'" &
```

확인: `ros2 topic hz /cam1/cam1/color/image_raw/compressed` / `/cam2/...`로 ~30Hz 나오는지.

> **⚠️ `&`로 백그라운드 실행하면 Ctrl-C로 안 죽는다.** 이렇게 띄우면 터미널의 SIGINT가 백그라운드 잡까지 전파되지 않아, 세션을 끝낼 때 `ps aux | grep realsense2_camera`로 PID를 찾아 수동으로 `kill`해야 하는 번거로움이 있다(실제로 이 방식 때문에 정리가 번거로웠던 사례 있음). 카메라를 **하나의 foreground launch + 뷰어**로 묶어 Ctrl-C 한 번에 깨끗이 종료되게 하는 개선판은 [`ros2_ur_ws/launch_cameras.sh`](../../ros2_ur_ws/launch_cameras.sh) 참고.

> **뷰어 창에 실행 제어 버튼이 내장되어 있다 (권장 경로).** `launch_cameras.sh`가 띄우는 뷰어 창([`ros2_ur_ws/camera_viewer.py`](../../ros2_ur_ws/camera_viewer.py))에는 이제 **`START EXECUTION`**·**`HOLD`** 두 버튼이 함께 떠 있다. 두 버튼은 구 Terminal 3이 손으로 치던 서비스(`/policy_leader_node/start_execution`·`/policy_leader_node/hold`)를 **똑같이** 호출한다 — 즉 일반 세션에서는 이 버튼만으로 자율 동작을 시작/정지하면 되고 별도 터미널이 필요 없다. 버튼 상태 표기:
> - **회색/딤(dimmed)** = 서비스가 아직 안 떠 있음 = Terminal 2의 `ros2 launch`가 `policy_leader_node`를 아직 안 올림. 이 상태에서는 눌러도 반응 없음(= 아직 arm 불가, 대기 중인 호출을 큐에 넣지도 않음).
> - **컬러(활성)** = 서비스 available = 이제 누를 수 있음.
> - **pending(호출 진행 중)** = 클릭 후 Trigger 응답을 기다리는 중(다시 눌러도 중복 호출 안 됨).
> - **초록/빨강(잠깐)** = Trigger 응답의 `success`/`message`를 색으로 표시(초록=성공, 빨강=실패 및 사유). 잠시 후 다시 평상시 활성 상태로 돌아간다.
>
> 따라서 **정상 흐름은**: Terminal 2 launch를 올리면 버튼이 회색→컬러로 살아나고, 핸드셰이크 수렴을 (Terminal 2 로그로) 확인한 뒤 뷰어에서 `START EXECUTION`을 눌러 자율 동작을 시작한다. 결과가 초록이면 EXECUTE 진입, 빨강이면 메시지(대개 arming guard 거부)를 확인하고 팔을 `start_pose` 근처로 되돌린 뒤 다시 누른다. 두 버튼은 서로 독립적으로 동작하므로(START가 pending이어도 HOLD는 항상 바로 눌림), 언제든 HOLD로 즉시 정지할 수 있다.

**Terminal 2 — ACT 서버 + 로봇 launch** (Method B, `HEADLESS=true` 추가):

```bash
cd ~/gello_software/ros2_ur_ws
HEADLESS=true ACT_CHECKPOINT=/home/laptop3/gello_software/ros2_ur_ws/src/gello_policy/checkpoints/act_banana_in_pot \
    ./run_ur7e_act_real.sh
```

펜던트 REMOTE 모드가 이미 활성화되어 있어야 함 — **Play 누를 필요 없음**. 이 터미널은 `ros2 launch`가 포그라운드로 도는 중이라 이후 명령은 다른 터미널/뷰어 버튼에서 내린다. **로그를 계속 봐야 하므로 이 터미널은 그대로 유지한다** — 로봇/그리퍼 연결, 핸드셰이크 수렴 여부를 여기서 확인한다.

**선택 사항 — 커맨드라인으로 직접 호출하고 싶다면 (구 Terminal 3, fallback):**

일반적인 대화식 세션에서는 위 **Terminal 1 뷰어의 버튼이 기본/권장 경로**다. 다만 자동화·스크립트에서 호출하거나 어떤 이유로 GUI 버튼을 쓸 수 없을 때를 위해, 버튼이 내부적으로 호출하는 것과 **동일한** 두 서비스를 별도 터미널에서 직접 부를 수도 있다:

```bash
source /opt/ros/humble/setup.bash
source ~/gello_software/ros2_ur_ws/install/setup.bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger   # 진짜 자율 동작 시작
ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger              # 그 자리에서 즉시 정지
```

> 버튼 클릭과 이 `ros2 service call`은 **완전히 같은 서비스·같은 Trigger**다 — 안전 게이트(arming guard, 클램프)도 동일하게 적용된다. 어느 쪽으로 부르든 결과는 같으니 상황에 맞게 섞어 써도 된다.

**Terminal 4 (선택, 실행 결과 로깅) — `gello_recorder`로 ACT 입출력을 HDF5+MP4 기록:**

`gello_recorder`는 원래 사람 GELLO 텔레옵을 기록하려고 만든 완전 **read-only** 패키지지만, `policy_leader_node`가 정확히 같은 토픽(`/gello/joint_states`, `/robotiq_gripper/command_percent`)에 발행하므로 **리더가 사람이든 ACT든 구분 없이 그대로 재사용 가능**하다:

```bash
cd ~/gello_software/ros2_ur_ws
./run_recorder.sh
```

> **`CAMS=true`를 붙이지 말 것.** 카메라는 이미 Terminal 1에서 떠 있으므로 recorder가 다시 launch를 시도하면 **USB 장치 충돌**이 난다. recorder는 그냥 이미 떠 있는 `/cam1`·`/cam2` 토픽을 구독한다. `~/start_execution` 호출 **전에** 이 recorder를 먼저 켜서 "recording" 로그를 확인한 뒤 시작하는 것을 권장.

기록되는 것: ACT의 관절 출력(`/gello/joint_states`), ACT의 그리퍼 출력(`/robotiq_gripper/command_percent`), 실제 로봇 상태(`/joint_states`), 브리지 최종 커맨드, 힘/토크, TCP pose, 카메라 2대 영상 — 전부 시간 동기화되어 `ros2_ur_ws/gello_logs/session_<타임스탬프>/`에 HDF5(`vectors.h5`) + `cam1.mp4`/`cam2.mp4`로 저장된다. `Ctrl-C`로 정지하면 flush되어 저장 완료.

### 리커버리 / start_pose로 리셋

- `~/hold`은 **현재 위치에서 즉시 정지**(freeze)시킬 뿐이다.
- 새 시도를 위해 팔을 완전히 `start_pose`로 되돌리는 가장 간단한 검증된 경로: **Terminal 2에서 `Ctrl-C`** (EXIT trap이 ACT 서버와 ros2 launch를 **둘 다** 정리) → 그런 다음 **똑같은 Terminal 2 명령을 그대로 다시 실행**한다. move-to-start 핸드셰이크가 현재 위치가 어디든 자동으로 다시 `start_pose`로 chase해 온다 (펜던트 수동 jog 불필요).

> **⚠️ 이 re-chase 모션은 충돌을 인지하지 못한다(collision-unaware).** 재시작 전에 팔 주변 상황을 한번 눈으로 확인하라.

### 첫 실기 종단(end-to-end) 성공 기록 — 2026-07-08

이 머신에서 첫 실기 전체 파이프라인이 성공했다: 핸드셰이크가 첫 실제 시도에서 깔끔하게 수렴(0.388 rad 갭을 0.78s에 chase, max gap 0.0001 rad로 수렴), 그리퍼가 fault 없이 연결·자동 캘리브레이션, 두 카메라 모두 arming 직전 ~29.7–29.9Hz로 라이브 확인, ACT 서버가 `--device cuda`로 `127.0.0.1:5591`에서 listen 확인. 오퍼레이터가 `~/start_execution`을 호출하자 로봇이 학습된 태스크("put right banana in pot")를 자율로 시도 — 안전 게이팅·실제 핸드셰이크·실제 카메라 매핑·ZMQ·GPU 추론·그리퍼 제어 전 구간이 실기에서 처음으로 종단까지 올바르게 동작했다. 다만 이 시점에서 **태스크 완수(성공률) 자체는 아직 안정적/반복 가능하지 않았다** — 파이프라인이 도는 것과 태스크를 매번 해내는 것은 별개다.

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

## 8. 검증 완료 (실기 포함) vs 아직 특성화 안 됨

> **2026-07-08 업데이트 — 실제 로봇 PC에서 첫 엔드투엔드 실기 구동 성공.** 이전까지 "로봇 PC에서 검증 필요"로 남아 있던 콜드스타트 경로(Humble colcon build, 실제 handshake, 실제 카메라 시리얼, 실제 팔 동작)가 실제 UR7e + Robotiq 2F-85 + RealSense 2대가 붙은 로봇 PC에서 **처음으로 실제 구동**되었다. 단, 이는 **시스템 배관(plumbing)이 실기에서 동작함을 확인한 것**이지 **정책의 태스크 성능/신뢰도를 특성화한 것이 아니다** — 아래 세 그룹을 분리해서 읽을 것.

**개발 PC에서 오프라인으로 검증됨** (로봇/카메라 없이 — 기존과 동일):

- 이미지 파이프라인 byte-parity (`image_preprocess.py` vs `eval_offline.py`의 `build_image_transforms()`)
- ZMQ multipart round-trip (더미 클라이언트)
- HOLD가 start_pose를 정확히 유지하는지
- arming guard (`~/start_execution`의 라이브 자세 0.1rad 가드)
- 두 안전 클램프 (OOD → clip, max_dev → clip)
- fail-silent FAULT 전이 (ZMQ 타임아웃/에러 시 발행 중단)

**실제 로봇 PC에서 실기 검증됨** (2026-07-08, 첫 성공 구동에서 확인 — 시스템 배관 레벨):

- **Humble colcon build** — `colcon build --packages-select gello_policy ur_gello_bringup`가 실제 로봇 PC의 Humble에서 성공(이전엔 가정이었으나 이제 실제로 수행됨).
- **실제 handshake** — 실제 UR7e에서 **첫 시도에** 깔끔하게 수렴: 0.388 rad gap을 0.78s 트래젝토리로 chase → max gap 0.0001 rad(`chase_tol` 0.06 rad을 크게 하회)로 수렴 → 0.41s 유지(`chase_dwell_s` 0.4s 요건 충족) → 컨트롤러가 `forward_position_controller`로 깔끔히 전환 → 브리지 resume. dead-band livelock 없음, 튜닝 불필요 — **디폴트 그대로 통과**. ([`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2의 tolerance 체인이 ACT 경로에서도 실기에서 성립함을 확인.)
- **실제 카메라 시리얼 매핑** — cam1(D435, `147122072740`)·cam2(D435if, `243222072700`) 모두 `rs-enumerate-devices`로 정상 열거, arm 직전 `ros2 topic hz`로 압축 토픽이 ~29.7–29.9Hz 라이브 스트리밍 확인 — 문서의 예상 물리 배치와 일치.
  > ⚠️ 이 시리얼은 **당시 장착돼 있던 개체**다. 2026-07-28에 카메라가 물리적으로 교체돼
  > 현재는 `151623020789` / `322743060038`이다(§2 정정 박스). 이 줄은 그때의 검증 기록이라
  > 일부러 그대로 두었다 — 오늘 이 값으로 실행하면 카메라가 조용히 안 뜬다.
- **실제 그리퍼** — 공유 Modbus socat 브리지로 연결 후 fault 없이 자동 캘리브레이션 성공(`gACT:1, gFLT:0`). ~1.7s 활성화 창 동안 일시적 "dropped streaming setpoint" 경고 1회가 떴으나 스스로 해소된 benign 현상(실제 문제 아님).
- **ACT 서버 / GPU 추론** — py3.12·`--device cuda`로 체크포인트 로드 후 `127.0.0.1:5591` listen 확인, GPU 추론 경로 라이브(RTX 3060 Mobile, CUDA 13.2, 드라이버 595.71.05).
- **엔드투엔드 자율 시도** — 오퍼레이터가 `~/start_execution`을 호출, 로봇이 학습된 "put right banana in pot" 태스크를 실제로 자율 시도. **전체 시스템(안전 게이팅 → 실제 handshake → 실제 카메라 매핑 → ZMQ 라운드트립 → 실제 GPU 추론 → 실제 그리퍼 제어)이 물리 하드웨어에서 엔드투엔드로 올바르게 동작한 첫 확인 사례**다.

**아직 특성화/튜닝 안 됨** (실기 배관은 확인됐으나 정책의 태스크 성능은 미검증):

- **그랩 성공률.** 첫 자율 시도는 태스크를 완전히/신뢰성 있게 완수하지 **못했다**(첫 시도에서 grab/placement 성공에 도달하지 못함). 성공률은 아직 측정된 바 없다 — **한 번의 성공적 구동은 "시스템이 돈다"는 뜻이지 "정책이 태스크를 푼다"는 뜻이 아니다.**
- **청크 경계에서의 부드러움** (receding horizon 재쿼리 경계) — 미특성화.
- **그리퍼 크러시 여부** — 미특성화.
- **후속 반복(iteration) 예상.** 그랩을 자주 놓치면 위 §7 가이드대로 `n_action_steps`/카덴스(재쿼리 주기)를 낮춰(`ACT_N_ACTION_STEPS=10~15`) 반응성을 올리거나, 경우에 따라 추가 학습 데이터가 필요할 수 있다. 현 시점에서 검증된 것은 **배포 파이프라인 자체가 엔드투엔드로 동작한다는 것**뿐이며, **정책의 태스크 수행 신뢰도는 아니다.**

> **알려진(수용된) 갭 — UR7e 기구학 캘리브레이션 파일 없음.** 이 머신에는 `ur7e_calibration.yaml`이 없어 배포는 팩토리/디폴트 기구학으로 돈다. 기존 문서대로 이는 **TCP/Cartesian 정확도에만 영향을 주고 joint-space ACT 제어에는 영향을 주지 않으므로**, 의도적으로 알려진·수용된 갭으로 남겨 두었다(블로킹 아님).

## 9. 트러블슈팅

- **cam1/cam2 시리얼이 뒤바뀜.** 반드시 **시리얼로 바인딩**할 것(`serial_no`) — 이름/포트 순서에 의존하면 두 RealSense를 구분할 수 없다. 잘못 바인딩되면 정책은 크래시하지 않고 **조용히 열화**된다(어느 카메라가 어느 관측 슬롯인지 학습 시와 달라지므로).
- **all-digit 시리얼의 정수 강제변환 버그.** `ros2 launch`의 CLI 인자 파서는 전부 숫자인 문자열(예: `151623020789`)을 정수로 오추론해 `serial_no`(string 파라미터)에 넣으려다 노드가 즉시 죽는다. 값을 **따옴표로 감쌀 것**: `serial_no:="'151623020789'"` (자세한 배경은 [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md)의 troubleshooting 항목 7 참고).
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
