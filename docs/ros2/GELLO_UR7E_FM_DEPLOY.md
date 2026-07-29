# GELLO → UR7e Flow-Matching Policy 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 학습된 Flow-Matching Policy(`multi_task_dit`, objective=`flow_matching`) "put the right banana in the pot" 정책으로 실제 UR7e를 자율 구동하기 위한 실기(real-robot) 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)입니다. 이 배포는 이미 실기 검증된 **ACT 배포**([`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md))·**Diffusion 배포**([`GELLO_UR7E_DIFFUSION_DEPLOY.md`](./GELLO_UR7E_DIFFUSION_DEPLOY.md))와 **같은 ROS2 패키지·같은 안전 스택**을 그대로 쓰고, **추론 서버만** Flow-Matching 정책(Euler-ODE 적분 샘플링)으로 교체한 형태입니다. JOINT FM 모델은 ACT/Diffusion과 **완전히 동일한 7-D 관절 액션 컨트랙트**를 내보내므로, 리더 노드·브리지·핸드셰이크·안전 클램프·그리퍼 스택은 **코드 변경 없이** 재사용됩니다 — FM 배포는 서버 교체 + yaml 재타게팅(포트 5593)이 전부입니다.
>
> **현재 상태 (정직하게):** 이 FM 경로의 정책 자체는 **오프라인 open-loop 평가로 체크포인트가 선택된 상태**이며(70k, poseMAE 0.0735 — §3), **실제 UR7e 팔 자율 구동, 로봇 PC에서의 colcon build, 타깃 GPU에서의 refill 레이턴시 측정은 아직 확정되지 않았습니다** — §8 참고. **안전상 중요한 ROS/브리지/핸드셰이크 코드는 ACT 실기에서 검증된 것과 바이트 단위로 동일**하지만, **FM 서버가 붙은 전체 경로의 실기 동작 자체는 아직 실물 하드웨어에서 확인되지 않았습니다.**
>
> - **패키지 개요 / 빌드 방법**: [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md)
> - **형제 문서 (실기 검증된 ACT 배포, 뼈대)**: [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md)
> - **형제 문서 (Diffusion 배포, 이 문서의 직계 템플릿)**: [`GELLO_UR7E_DIFFUSION_DEPLOY.md`](./GELLO_UR7E_DIFFUSION_DEPLOY.md)
> - **설계 근거**: 프로젝트 루트 `DEPLOY_REPO_DECISION.md` (왜 `gello_software`를 확장하는지, 왜 synthetic-leader + ZMQ 분리인지, HIL-SERL로 가는 길)
> - **관련 실기 문서**: [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) (사람 GELLO 리더로 하는 handshake의 전신), [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) (이 정책을 학습시킨 데이터셋을 만든 레코더)

이 패키지(`gello_policy`)는 학습된 정책이 **사람 GELLO 리더 대신** `/gello/joint_states` + 그리퍼를 발행하게 해서, 기존 `gello_ur_bridge` + `gello_move_to_start` handshake + Robotiq Modbus 스택을 **코드 변경 없이** 재사용한다. FM 추론은 별도의 py3.12 프로세스(torch + lerobot 0.6.1 + **transformers**/CLIP)에서 돌고, ROS py3.10 프로세스와는 localhost ZMQ로만 통신한다. Diffusion 배포와 다른 것은 **추론 서버**(`policy_server/fm_server.py`, Euler-ODE 샘플링)와 그것이 **ZMQ 포트 5593**을 쓴다는 점(ACT는 5591, Diffusion은 5592)뿐이다.

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 대상 로봇 | UR7e + Robotiq 2F-85, ROS2 **Humble** |
| 체크포인트 | HF `Bigenlight/flow_matching_banana_in_pot_joint`, best = **70k** |
| 액션 공간 | **JOINT 7-D** (`[cmd1..cmd6, grip_cmd]`, 6 UR 관절 rad + 그리퍼) — ACT/Diffusion과 완전히 동일한 컨트랙트 |
| 두 프로세스 | FM 서버(py3.12, torch+lerobot 0.6.1+**transformers**+diffusers, `policy_server/fm_server.py`) ⇄ ZMQ(localhost:**5593**) ⇄ `policy_leader_node`(py3.10 rclpy) |
| 아키텍처 | lerobot **`multi_task_dit`** (DiT), objective=`flow_matching`, CLIP ViT-B/16 (vision+text) 인코더 — **텍스트 조건부** |
| 샘플러 | **Euler-ODE 적분** (`integration_method=euler`, `num_integration_steps`). 학습값 100 → **배포 기본 Euler-10** (셸 런처가 `FM_NUM_INTEGRATION_STEPS=10` 세팅) |
| 정책 → 팔 경로 | `policy_leader_node`가 `/gello/joint_states`(30Hz) + `/robotiq_gripper/command_percent`를 발행 → 기존 `gello_ur_bridge`(250Hz 업샘플, slew clamp) + `gello_move_to_start` handshake + Robotiq Modbus **무변경** |
| 시작 절차 | 핸드셰이크가 HELD start pose로 팔을 파킹 → 오퍼레이터가 명시적으로 `ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger` 호출해야 자율 모션 시작 |
| receding horizon | `n_action_steps=24`, `n_obs_steps=2` — **24틱마다** Euler-ODE 재적분(≈ **0.8s @30Hz**), 그 사이엔 큐에서 pop만 |
| 안전 | 속도는 브리지의 0.625 rad/s slew ceiling(불변), 위치는 `policy_leader_node`의 1.2x envelope 클램프 + live-pose 0.5rad 클램프. 타임아웃 순서 `act_timeout_s=0.6` < `obs_timeout_s=0.7` < `staleness_timeout_s=0.8`로 리더의 ZMQ 타임아웃을 primary fault owner로 배치 (§6) |
| 오프라인 (open-loop) | held-out 롤아웃 MAE로 70k 선택 (poseMAE **0.0735**, Euler-10) — §3 |
| **아직 안 됨** | **실제 UR7e 팔 구동, 로봇 PC colcon build, 타깃 GPU refill 레이턴시 확정** — §8 |
| 실기 전 필수 | 타깃 GPU에서 **첫 팔 구동 전에** refill 레이턴시를 sanity-check (서버 `net refill` 로그 관찰; §6) |

### 목차

1. [빅 픽처 / 데이터 흐름](#1-빅-픽처--데이터-흐름)
2. [하드웨어 / 환경](#2-하드웨어--환경)
3. [모델 & 체크포인트 다운로드](#3-모델--체크포인트-다운로드)
4. [셋업 절차 (로봇 PC)](#4-셋업-절차-로봇-pc)
5. [배포 실행하기](#5-배포-실행하기)
6. [안전 모델](#6-안전-모델)
7. [추론 동작 (Euler-ODE 샘플링, receding horizon, 이미지 파이프라인)](#7-추론-동작-euler-ode-샘플링-receding-horizon-이미지-파이프라인)
8. [검증 완료 vs 아직 안 됨](#8-검증-완료-vs-아직-안-됨)
9. [트러블슈팅](#9-트러블슈팅)
10. [Diffusion/ACT 배포와의 관계 / HIL-SERL으로 가는 길](#10-diffusionact-배포와의-관계--hil-serl으로-가는-길)

---

## 1. 빅 픽처 / 데이터 흐름

FM 서버(py3.12, torch+lerobot+transformers/CLIP)와 `policy_leader_node`(py3.10, rclpy)는 서로 다른 파이썬/venv에 살기 때문에 **localhost ZMQ REQ/REP multipart**로만 통신한다 — `lerobot`은 Python ≥3.12를 요구하는데 ROS2 Humble의 `rclpy`는 CPython 3.10으로 빌드되어 있어 같은 인터프리터를 공유할 수 없다(자세한 이유는 `DEPLOY_REPO_DECISION.md` §3). **Diffusion 배포와 유일하게 다른 것은 이 서버가 Flow-Matching 정책을 Euler-ODE로 샘플링한다는 점과 ZMQ 포트가 5593(ACT는 5591, Diffusion은 5592)이라는 점뿐이다.** 포트를 분리한 이유는 세 서버(ACT/diffusion/FM)가 한 머신에 공존해도 충돌하지 않게 하기 위함이다. ZMQ 와이어 포맷(`zmq_protocol.py`)은 세 서버가 **완전히 동일**하다.

```
┌───────────────────────────────────────┐        ZMQ REQ/REP multipart     ┌──────────────────────────────┐
│  fm_server.py (py3.12)                 │   tcp://127.0.0.1:5593            │  policy_leader_node (py3.10)  │
│  torch + lerobot 0.6.1 + transformers  │◄────────────────────────────────┤  rclpy                       │
│  (CLIP ViT-B/16 vision+text)           │   reset: {"cmd":"reset"}         │                               │
│                                         │─────────────────────────────────►│  HOLD / EXECUTE / FAULT       │
│  get_policy_class(cfg.type)            │   act: {"cmd":"act",             │  state machine                │
│    → MultiTaskDiTPolicy (generic load) │        "state":[7 floats]},      │  (ACT/diffusion과 100% 동일)   │
│  cfg.num_integration_steps 오버라이드    │        <cam1 jpeg>,<cam2 jpeg>   │                               │
│  n_obs_steps=2, n_action_steps=24      │                                  │                               │
│  select_action (Euler-ODE 재적분)       │◄─────────────────────────────    │                               │
│  이미지 = NATIVE res (policy가 224 내부) │   action = [cmd1..cmd6, grip_cmd]│                               │
│  task="put the right banana in the pot"│                                  │                               │
│  reply: {"ok":true,"action":[7]}       │                                  │                               │
└───────────────────────────────────────┘                                  └───────────────┬───────────────┘
                                                                                            │ publishes @30Hz
                                                                                            │ (safety-clamped)
                                                                                            ▼
                                                    /gello/joint_states  +  /robotiq_gripper/command_percent
                                                                                            │
                                                                                            ▼
                                              ┌───────────────────────────────────────────────────┐
                                              │  gello_ur_bridge (UNCHANGED)                        │
                                              │  250Hz 업샘플, One-Euro 필터, max_step_rad 슬루우클램프│
                                              │  (0.625 rad/s 상한), staleness watchdog (0.8s)       │
                                              └───────────────────────┬─────────────────────────────┘
                                                                       │ /forward_position_controller/commands
                                                                       ▼
                                              ┌───────────────────────────────────────────────────┐
                                              │  gello_move_to_start handshake (UNCHANGED)          │
                                              │  실제 UR7e + Robotiq 2F-85 Modbus                     │
                                              └───────────────────────────────────────────────────┘

observation 방향 (팔 → 정책, 매 EXECUTE 틱):
  /joint_states (6 joints)  ──┐
  /robotiq_gripper/position_percent ─┤→ policy_leader_node가 state[7] 조립 → ZMQ "act" 요청 (port 5593)
  /cam1/.../compressed (JPEG) ─┤   (raw JPEG 그대로 전달, py3.10 쪽에서 cv2/torch 사용 안 함)
  /cam2/.../compressed (JPEG) ─┘
```

핵심 포인트: `policy_leader_node`는 `gello_publisher_node`(사람 GELLO 리더)를 대체하는 **synthetic leader**일 뿐이며, ACT/diffusion/FM 어느 쪽이든 **동일한 policy-agnostic 노드**다. 그 아래의 `gello_ur_bridge` + `gello_move_to_start` + Robotiq Modbus는 리더가 사람이든 어느 정책이든 전혀 구분하지 못하며 완전히 동일한 코드가 돈다. 실제로 `run_ur7e_fm_real.sh`는 **diffusion 런치 파일(`ur7e_diffusion_real.launch.py`)을 그대로 재사용**하고 `params_file:=fm_deploy.yaml`, `act_port:=5593`만 오버라이드한다(§5).

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

  > **⚠️ 물리적 카메라 배치 = 학습 리그와 반드시 일치.** 한 대는 씬/3인칭 시점을, 다른 한 대는 작업공간 근접(close-up)을 본다. **두 물리 시점과 cam1/cam2 시리얼 할당이 학습 당시 리그와 동일**해야 하며, 어긋나면 정책이 **에러 없이 조용히 열화**된다(정책은 `cam1`=씬, `cam2`=근접 같은 고정 배치를 가정하고 학습됨 — `cam1/cam2` 순서는 학습에 고정됨). 실기 첫 구동 전에 **라이브 뷰(`rqt_image_view` 등)를 학습 셋업 사진과 대조**해 확인할 것. (FM 서버는 네이티브 해상도 프레임을 받아 정책이 224로 내부 리사이즈하므로, 두 카메라의 시점·순서가 유일한 육안 검증 포인트다.)

- **ROS2**: 로봇 PC는 **Humble** (rclpy py3.10). `ur_gello_bringup`/`gello_recorder`와 동일 워크스페이스. 필요한 ROS 패키지 설치:

  ```bash
  # UR 드라이버 + RealSense 카메라 스택 (+ librealsense2 SDK)
  sudo apt install ros-humble-ur ros-humble-realsense2-camera ros-humble-librealsense2
  cd gello_software/ros2_ur_ws
  rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
  ```

  > 실 로봇/ROS 최초 셋업 전체(펜던트 External Control 페어링, 기구학 캘리브레이션, 24V 그리퍼)는 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)를 참고. 이 문서는 그 위에 FM 배포만 얹는다.

- **FM 서버**: 별도 **py3.12 venv** (torch cu128 빌드 + lerobot 0.6.1 + **transformers** + diffusers + pyzmq + opencv-python). **GPU(CUDA) 필수 (권장이 아님).** `--device cuda`가 기본값이며, CUDA가 없으면 서버는 **자동 CPU 폴백을 하지 않고 종료 코드 3으로 거부**한다(`resolve_device()`) — CPU에서의 full Euler-ODE 적분(DiT + CLIP 인코더 통과)은 leader의 `act_timeout_s`(0.6s)를 넘겨 FAULT-loop를 유발하기 때문이다. CPU에서 굳이 돌리려면 `--device cpu`(또는 `FM_DEVICE=cpu`)를 **명시**하고 동시에 `fm_deploy.yaml`의 타임아웃을 크게 올려야 한다(§9 참고).
  > **오프라인 CLIP 주의.** `multi_task_dit`는 `CLIPVisionModel`/`CLIPTextModel.from_pretrained`로 CLIP ViT-B/16 인코더를 로드한다. `run_fm_server.sh`가 **`HF_HUB_OFFLINE=1`·`TRANSFORMERS_OFFLINE=1`을 export**하므로, 서버는 시작 시 HF 허브를 절대 치지 않는다. 따라서 **CLIP 가중치가 HF 캐시에 있거나 체크포인트에 baked-in**되어 있어야 한다 — 없으면 오프라인 로드에서 실패한다(§4, §9).

## 3. 모델 & 체크포인트 다운로드

**모델: `Bigenlight/flow_matching_banana_in_pot_joint`** (Hugging Face). 핵심 모델 사실:

| 항목 | 값 |
|---|---|
| 아키텍처 | lerobot **`multi_task_dit`** — DiT 백본 + **CLIP ViT-B/16** vision·text 인코더. RGB 프레임은 인코더 내부에서 `image_resize_shape`(= [224, 224])로 리사이즈 |
| objective | **`flow_matching`** — velocity field를 Euler-ODE로 적분해 액션 시퀀스 생성 |
| 조건화 | **텍스트 조건부** — `task` 문자열을 CLIP text 인코더로 토크나이즈/임베딩. 단일 태스크이지만 모델이 text-conditioned라 **학습 문구 그대로** 넣어야 한다: `"put the right banana in the pot"` |
| 액션 공간 | **JOINT 7-D** (`[cmd1..cmd6, grip_cmd]`, 6 UR 관절 rad + 그리퍼, 절대값). 그리퍼 채널은 사실상 binary |
| 관측 | `observation.state` (7,), `observation.images.cam1/cam2` RGB (**네이티브 해상도로 서버에 전달 → policy 내부에서 224×224**), `task` (문자열) |
| 샘플러 | **Euler-ODE 적분**, `integration_method=euler`. 학습 `num_integration_steps=100` |
| **배포 적분 스텝** | **Euler-10** (셸 런처 기본 `FM_NUM_INTEGRATION_STEPS=10`) — 70k를 고른 오프라인 eval과 동일한 샘플러이며 refill을 `act_timeout_s`(0.6s) 아래로 유지 |
| **best checkpoint** | **step 70,000** |
| 정규화 | STATE/ACTION 정규화 + VISUAL = MEAN_STD. 통계·토크나이저는 저장된 pre/post-processor json에 baked-in |

> **@70k open-loop 성적:** held-out 롤아웃 **poseMAE 0.0735 rad** (Euler-10). 이는 **open-loop 지표이지 closed-loop 태스크 성공률이 아니다** — 실기 성공률은 아직 측정된 바 없다(§8). 배포 기본 Euler-10은 바로 이 70k 선택 eval이 쓴 샘플러이므로, 실기도 기본은 Euler-10로 시작하는 것이 학습/평가 parity 측면에서 맞다.

이 체크포인트 리포는 config.json + model.safetensors + `policy_preprocessor`/`policy_postprocessor` json이 함께 들어 있는 HF 모델 레이아웃이므로, 서버 `--checkpoint`(또는 `FM_CHECKPOINT`)는 **그 디렉터리를 직접 가리키면 된다**. 다운로드는 `huggingface-cli`/`hf`로:

```bash
cd gello_software/ros2_ur_ws/src/gello_policy

# CLIP 가중치까지 오프라인에서 쓰려면, 온라인일 때 미리 받아 캐시를 채워둘 것.
act_venv/bin/hf download Bigenlight/flow_matching_banana_in_pot_joint \
    --local-dir checkpoints/flow_matching_banana_in_pot_joint
# 위 --local-dir(pretrained_model 레이아웃)을 --checkpoint / FM_CHECKPOINT에 그대로 넘긴다.
```

> **다운로드 스크립트.** FM 전용 다운로드 스크립트는 아직 없다. 범용 `scripts/download_checkpoint.sh`는 `CKPT_REPO`로 리포를 바꿀 수 있으나 로컬 하위 디렉터리 이름이 `act_banana_in_pot`으로 하드코딩되어 있으니(경로만 다를 뿐 내용은 정상), 혼동을 피하려면 위처럼 `hf download`를 직접 쓰는 편이 깔끔하다. `huggingface-cli`/`hf`가 없으면 `pip install -U 'huggingface_hub[cli]'`.

관련 데이터셋: `Bigenlight/banana_in_pot_lerobot_v3` (학습에 쓰인 LeRobot v3 joint-space 데이터셋).

## 4. 셋업 절차 (로봇 PC)

1. **colcon 빌드 — `gello_policy` + `ur_gello_bringup`** (Humble, py3.10). `run_ur7e_fm_real.sh`는 `ur7e_diffusion_real.launch.py`를 그대로 재사용하며, 이 런치는 `ur_gello_bringup` 노드(`gello_ur_bridge`, `gello_move_to_start`, `robotiq_gripper_modbus`)를 돌리고 `ur_robot_driver`의 `ur_control.launch.py`를 include하므로, **`gello_policy`만 빌드하면 안 되고 `ur_gello_bringup`도 반드시 함께 빌드**해야 한다. **빌드는 `config/fm_deploy.yaml`을 share로 설치하기 위해서도 필요하다** — 런처는 설치된 `share/gello_policy/config/fm_deploy.yaml`을 먼저 찾고 없으면 src로 폴백한다. (`policy_leader_node`만 이 빌드에 포함되며 — ACT/diffusion과 **완전히 같은 노드** — `policy_server/`는 **설치되지 않고** 자기 venv에서 소스 디렉터리째로 실행됨.)

   ```bash
   cd gello_software/ros2_ur_ws
   rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
   colcon build --packages-select gello_policy ur_gello_bringup
   # ...또는 가장 간단하게 워크스페이스 전체:  colcon build
   source install/setup.bash
   ```

2. **py3.12 venv + FM 서버 의존성.** venv는 **`ros2_ur_ws/act_venv`에** 만든다 — run 스크립트가 기대하는 기본 위치(`FM_VENV=ros2_ur_ws/act_venv`)다. 다른 곳에 두려면 `export FM_VENV=/your/path`. FM 전용 lock은 아직 없고, **diffusion lock을 베이스로 쓰고 `transformers`(CLIP)만 확실히 있으면 된다**:

   ```bash
   cd gello_software/ros2_ur_ws        # 여기서 venv 생성 (ACT/diffusion과 공유 가능)
   python3.12 -m venv act_venv
   act_venv/bin/pip install --upgrade pip
   act_venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   act_venv/bin/pip install -r src/gello_policy/policy_server/requirements-diffusion.lock
   # CLIP 인코더용 — diffusers가 끌어오지만 명시적으로 확인:
   act_venv/bin/pip install transformers
   ```

   > **ACT/diffusion venv를 이미 만들었다면** 대부분의 핀이 동일하므로 같은 `act_venv`에 `transformers`만 있는지 확인하면 된다. FM 서버가 필요로 하는 것: **lerobot 0.6.1 (`multi_task_dit`를 제공하는 버전) + torch + transformers + diffusers + pyzmq + opencv-python**. (torch/torchvision을 다운그레이드시키지 말 것 — ACT 문서의 주의사항 동일.)
   >
   > **오프라인 CLIP 캐시.** 서버는 `HF_HUB_OFFLINE=1`로 뜨므로 **첫 실행 전에 CLIP ViT-B/16 가중치가 HF 캐시에 있어야 한다.** 온라인일 때 한 번 로드해 캐시를 채우거나, 체크포인트에 baked-in된 가중치를 쓰거나, `HF_HOME`/`TRANSFORMERS_CACHE`를 캐시가 있는 경로로 지정할 것. 오프라인 로봇 PC에서 캐시가 비어 있으면 CLIP `from_pretrained`가 실패한다(§9).

3. **RealSense 카메라 2대를 정확한 시리얼→네임스페이스 매핑으로 기동** (ACT/diffusion과 동일):

   ```bash
   # cam1 (D435, 151623020789) → 네임스페이스 cam1
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam1 camera_namespace:=cam1 \
       serial_no:="'151623020789'" rgb_camera.color_profile:="'1280x720x30'" &
   # cam2 (D435if, 322743060038) → 네임스페이스 cam2
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam2 camera_namespace:=cam2 \
       serial_no:="'322743060038'" rgb_camera.color_profile:="'1280x720x30'" &

   # 압축 이미지 토픽이 ~30Hz로 나오는지 확인:
   ros2 topic hz /cam1/cam1/color/image_raw/compressed
   ros2 topic hz /cam2/cam2/color/image_raw/compressed
   ```

4. **⚠️ 첫 팔 구동 전 필수 — refill 레이턴시 sanity-check.** Euler-10 refill이 이 로봇 PC의 GPU에서 안전 예산(`act_timeout_s=0.6s`) 안에 드는지 반드시 확인한다(§6). FM 전용 벤치마크 스크립트는 아직 없으므로(diffusion용 `benchmark_diffusion_latency.py`는 `DiffusionPolicy`를 로드하므로 FM에 그대로 쓸 수 없다), 서버를 standalone으로 띄워 **`net refill #...` 로그 사이의 실측 지연**을 관찰하거나(§5의 `run_fm_server.sh`), warmup 소요·첫 refill 타이밍이 예산 안인지 확인한다:

   ```bash
   FM_CHECKPOINT=/path/to/pretrained_model FM_DEVICE=cuda \
       FM_NUM_INTEGRATION_STEPS=10 \
       ./src/gello_policy/scripts/run_fm_server.sh
   # 로그의 "warmup done in Xs" 및 이후 "net refill" 간격이 0.6s를 넘지 않는지 볼 것.
   ```

## 5. 배포 실행하기

```bash
cd gello_software/ros2_ur_ws
FM_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_fm_real.sh
```

`run_ur7e_fm_real.sh`는 먼저 py3.12 FM 서버를 백그라운드로 띄우고(PID 저장, `trap 'kill $FM_SERVER_PID' EXIT`로 Ctrl-C 시 함께 정리), **시작 전에 포트 5593이 이미 점유돼 있으면 거부**하고(잔여 stale 서버로 팔이 구동되는 것을 막음), 서버가 살아 있고 **포트 5593**이 listen 중인지 헬스체크한 뒤, Humble `ros2 launch gello_policy ur7e_diffusion_real.launch.py`를 **`params_file:=fm_deploy.yaml act_port:=5593`으로** 실행한다. (전용 FM 런치 파일은 없다 — 정책-무관 diffusion 런치를 재사용한다.)

- **Method A (기본)**: 펜던트에서 External Control 프로그램을 로드하고 **Play** — `HEADLESS` 미설정/`false`.
- **Method B (헤드리스)**: `HEADLESS=true FM_CHECKPOINT=/path/... ./run_ur7e_fm_real.sh` — 펜던트 Play 없이 드라이버가 직접 URScript를 보냄. 로봇이 **REMOTE 모드**여야 함.

주요 환경변수(런처 배너에도 출력됨): `FM_CHECKPOINT`(**필수**), `ROBOT_IP`(기본 `192.168.10.11`), `FM_PORT`(기본 `5593`), `FM_DEVICE`(기본 `cuda`), `FM_VENV`(기본 `ros2_ur_ws/act_venv`), `FM_N_ACTION_STEPS`(기본 `24`), `FM_NUM_INTEGRATION_STEPS`(**런처 기본 `10`**), `FM_TASK`(기본 `"put the right banana in the pot"`), `FM_PARAMS_FILE`(기본 설치된 `config/fm_deploy.yaml`), `START_MODE`(`gello`|`init_align`), `CALIB`(옵션 기구학 YAML).

> **서버만 따로 띄우기 (bring-up/테스트).** ros2 없이 서버 half만 검증하려면:
> ```bash
> FM_CHECKPOINT=/path/to/pretrained_model ./src/gello_policy/scripts/run_fm_server.sh
> # --checkpoint/--host/--port/--device/--n-action-steps/--num-integration-steps/--task 플래그로 env 오버라이드 가능
> ```

> **⚠️ 카메라를 먼저 띄울 것.** FM 배포도 카메라 없이는 돌지 않는다: EXECUTE는 매 틱(최소 refill 틱마다) fresh `cam1`/`cam2` 프레임을 요구하고, `~/start_execution`과 in-EXECUTE obs-freshness watchdog이 카메라가 없거나 stale하면 각각 arming을 거부/FAULT시킨다(§6). 실행 **전에** §4의 카메라 기동(3번)으로 두 RealSense를 띄우고 `ros2 topic hz`로 확인하라.

### 시작 handshake 타임라인

1. **t=0s** — `ur_control.launch.py`: UR7e 드라이버 + `scaled_joint_trajectory_controller` active, `forward_position_controller` inactive (로드만).
2. **t≈6s** — `policy_leader_node` 기동, **HOLD** 상태로 진입. 매 틱 고정 `start_pose`
   `q = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` (rad, UR 관절 순서, `fm_deploy.yaml`)를 완전히 정지된 상태로 발행 — 서버는 아직 쿼리하지 않음. 그리퍼는 `start_gripper: 0.0`(OPEN) 유지. 동시에 `gello_ur_bridge`가 pre-spawn.
3. **t≈8s** — `gello_move_to_start`가 이 HELD start pose로 수렴 게이트 chase 핸드셰이크를 수행(메커니즘은 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2와 100% 동일 — 다른 점은 리더가 사람 손이 아니라 이 고정 pose라는 것뿐). 수렴하면 STRICT 컨트롤러 전환 → 브리지 `~/resume`.
4. **팔이 start_pose에서 정지(파킹)한다.** 이 시점까지 **자율 모션은 전혀 없다** — 리더가 완전히 정지해 있기 때문.
5. **오퍼레이터가 명시적으로** 자율 실행을 시작한다:

   ```bash
   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
   ```

   이 호출은 (a) 라이브 팔 자세가 `start_pose`에서 **`START_GATE_RAD`(0.1 rad) 이내**인지 확인하고(아니면 거부: "live pose not within 0.1 rad of start_pose"), (b) ZMQ `reset`을 서버에 보내 FM의 action queue를 비운 뒤, (c) **EXECUTE**로 진입한다. 이후부터 매 틱 서버를 쿼리하고(refill 틱에서만 실제 Euler-ODE 재적분, 나머지는 큐에서 pop) 클램프된 타깃을 발행한다 — 이때부터가 진짜 자율 모션 시작이다.

6. **언제든 일시정지(pause)하려면**:

   ```bash
   ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
   ```

   현재 라이브 자세를 그대로 HOLD로 전환해 그 자리에서 팔을 정지시킨다. **그리퍼는 마지막으로 명령된 값을 그대로 유지**한다. 재-arm은 여전히 `start_pose` ±0.1rad 근처를 요구한다.

(Method A/B의 코드 동등성, 4-터미널 워크플로우, `gello_recorder`로 결과 로깅, `Ctrl-C` 후 재실행으로 start_pose 리셋하는 리커버리 절차는 **ACT/Diffusion 문서 §5와 완전히 동일**하다 — 명령의 `run_ur7e_diffusion_real.sh` → `run_ur7e_fm_real.sh`, `DIFFUSION_CHECKPOINT` → `FM_CHECKPOINT`, 포트 5592 → 5593만 바뀐다. 재-chase 모션이 collision-unaware라는 경고도 동일.)

## 6. 안전 모델

이 정책은 실제 로봇을 자율 구동하므로 이 섹션을 가볍게 넘기지 말 것. **속도/위치 안전 스택은 ACT/Diffusion과 완전히 동일**하며(아래 처음 세 항목), 그 뒤에 **FM/receding-horizon 특유의 항목**이 이어진다. **이 액션은 절대 관절값(absolute joint)이다** — 델타가 아니라 팔이 가야 할 목표 관절각 그 자체이므로, 잘못된 한 틱이 그대로 큰 점프 명령이 될 수 있다. 그래서 아래 다층 클램프가 존재한다.

- **속도 상한 — 절대 넘지 않음.** `policy_leader_node`가 무엇을 발행하든, 그 아래 **`gello_ur_bridge`는 무변경**이며 자신의 `max_step_rad` slew 클램프(배포 설정 지속 **0.625 rad/s** = 0.0025 rad @ 250Hz, UR 한계 3.14 rad/s 대비 마진)를 무조건 적용한다. FM이 아무리 급격한 타깃을 내도 팔은 이 속도 이상으로 움직이지 않는다. **`max_step_rad`는 올리지 말 것.**
- **위치 클램프 — `policy_leader_node`가 매 EXECUTE 틱마다, 이 순서로:** (a) **1.2x 데이터셋 envelope로 clip**(`joint_limits_lo/hi`, `fm_deploy.yaml`; OOD 출력 가드) → (b) live `/joint_states` 대비 **`max_dev_rad`(기본 0.5 rad)로 clip**. 둘 중 하나라도 발동하면 throttled `WARN`을 남긴다. (이 클램프는 ACT/Diffusion 배포와 **바이트 단위로 동일한 config·코드**를 쓴다 — 액션 컨트랙트가 같으므로.)
- **첫 명령 점프 가드 = 시작 게이트 + per-tick 클램프 (특례 없음).** `~/start_execution`은 라이브 팔이 `start_pose`에서 **0.1 rad(`START_GATE_RAD`) 이내**일 때만 EXECUTE를 arm한다 → 팔은 항상 데이터셋 start pose **가까이에서** 자율을 시작한다. 그리고 **첫 EXECUTE 틱의 명령도 다른 틱과 똑같이** (a) envelope clip + (b) live 대비 `max_dev_rad`(0.5 rad) clip을 받으므로, 정책의 첫 액션이 아무리 튀어도 **라이브 자세에서 0.5 rad을 초과하는 즉발 점프는 발행되지 않는다**(그 위에 브리지 slew가 다시 0.625 rad/s로 상한). 즉 "start_pose 근처에서 시작 + 첫 명령을 per-tick 클램프"의 조합이 첫 명령 점프 가드다 — 첫 틱을 위한 별도 완화/특례는 **없다.**
- **관측 freshness watchdog + fail-silent FAULT.** EXECUTE에서 4개 관측(`joint_states`, `grip_pos`, `cam1`, `cam2`) 중 하나라도 stale하거나(`obs_timeout_s`), ZMQ 타임아웃/에러/`{"ok":false}` 응답을 받으면 노드는 즉시 **FAULT**로 전이해 `/gello/joint_states` 발행을 **완전히 중단**한다 → 브리지 staleness watchdog이 팔을 정지시킨다. **마지막 유효 타깃을 계속 재발행하지 않는다.** 복구 시 브리지는 라이브 자세에서 soft-start로 재-seed하며, 재-arm은 `start_pose` 근처에서만 허용된다.

### ⚠️ FM refill / 타임아웃 배치 (반드시 이해할 것)

- **리더는 refill 틱의 Euler-ODE 적분 동안 발행을 블로킹한다 (그리고 관측 갱신도 멈춘다).** 큐가 빈 refill 틱에서 `policy_leader_node`는 서버의 ZMQ 응답을 **동기적으로 기다리며 그동안 `/gello/joint_states`에 발행하지 못한다.** 게다가 **이 노드는 단일 스레드**라, refill 블록 동안 리더는 **카메라/관측 콜백도 서비스하지 못해** obs 타임스탬프가 refill 소요 시간 D만큼 그대로 **얼어붙는다.** FM refill은 `num_integration_steps`번 DiT+CLIP forward를 도는 full Euler 적분이므로 **ACT의 단일 트랜스포머 forward보다 무겁다**(diffusion DDIM refill과 동급 이상). 이 refill 갭이 워치독을 실수로 트립시키지 않도록 **세 워치독**의 순서를 **명시적으로** 배치했다(`fm_deploy.yaml`, diffusion의 넓힌 값을 그대로 계승):

  | 파라미터 | 값 | 소유자 | 의미 |
  |---|---|---|---|
  | `act_timeout_s` | **0.6s** | `policy_leader_node` (리더) | ZMQ REQ(recv) 타임아웃. 이 시간 안에 Euler refill이 안 끝나면 리더가 **먼저** FAULT — **primary fault owner** |
  | `obs_timeout_s` | **0.7s** | `policy_leader_node` (리더) | 관측 freshness watchdog. 단일 스레드 노드가 refill 블록 동안 카메라 콜백을 못 돌려 obs 타임스탬프가 D만큼 얼기 때문에 `act_timeout_s`보다 **위**에 둔다 — 정상 refill이 **가짜 obs-stale FAULT**를 내지 않도록 |
  | `staleness_timeout_s` | **0.8s** | `gello_ur_bridge` (브리지) | 발행이 끊긴 지 이만큼 지나면 브리지가 팔을 정지. 리더의 정상 ≤0.6s refill 갭보다 **넉넉히 위** |

  **불변식(invariant): `act_timeout_s` (0.6) < `obs_timeout_s` (0.7) < `staleness_timeout_s` (0.8).** 이렇게 배치해야 **리더의 ZMQ 타임아웃(0.6s)이 primary fault owner**가 되고, refill 블록으로 obs가 얼어도 정상 refill이 가짜 obs-stale FAULT를 내지 않으며, **진짜로 얼어붙은 카메라는 여전히 잡힌다**(다만 0.2s 늦게 = `obs_timeout_s`에서).

- **Euler-steps ↔ `act_timeout_s` 트레이드오프 (타깃 GPU에서 반드시 sanity-check).** 배포 기본은 **Euler-10**(셸 런처 `FM_NUM_INTEGRATION_STEPS=10`)이고, 이는 refill을 `act_timeout_s`(0.6s) 아래로 유지하도록 고른 값이자 70k를 선택한 eval의 샘플러다. **학습된 100 스텝 쪽으로 올리면** 샘플이 미세하게 매끄러워지지만 refill이 훨씬 무거워진다 — **레이턴시 여유가 확인된 경우에만** 올릴 것. 반대로 **측정한 refill p99가 예산(0.6s)을 넘으면, 워치독을 넓히지 말고 `FM_NUM_INTEGRATION_STEPS`를 줄여라**(예: 5). 타임아웃을 더 넓히는 것은 팔 워치독을 약화시키는 것이므로 **금지**한다. Euler 스텝 수를 줄이는 것이 refill을 예산 안으로 되돌리는 올바른 노브다. FM 전용 벤치마크 스크립트는 아직 없으므로(§4-4), **첫 팔 구동 전에 이 로봇 PC의 GPU에서** 서버 standalone `net refill` 로그로 실측 지연을 반드시 확인하라. (순수 추론 지연은 브리지가 겪는 발행 갭의 **하한**이다 — JPEG 디코드 + ZMQ/JSON 왕복 오버헤드가 빠져 있어 실제 갭은 조금 더 크다.)

- **오퍼레이터 게이트 + 워밍업.** 핸드셰이크만으로는 자율 모션이 시작되지 않으며 `~/start_execution`이 필수 게이트다. FM 서버는 로드 시 더미 관측으로 **2회 추론을 미리 돌려**(→ 그 뒤 `reset()`) **CUDA 커널 + CLIP vision/text 인코더 lazy-init을 워밍업**하므로(`_warmup()`, 로그 `warmup done in Xs`), 실제 첫 refill 틱이 `act_timeout_s`를 넘겨 가짜 FAULT를 내지 않는다. FM은 첫 EXECUTE에서 곧바로 refill을 한 번 하므로 이 워밍업이 특히 중요하다.
- **단일 그리퍼 writer.** 이 배포는 `gello_gripper_bridge`를 launch하지 않는다 — `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher다. (그리퍼 채널 `action[6]`은 identity로 발행.)
- **E-STOP.** 어떤 소프트웨어 클램프도 물리 안전을 대체하지 않는다. **펜던트 E-STOP을 항상 손 닿는 곳에** 두고, 이상 시 즉시 누른다. Ctrl-C는 FM 서버 + ros2 launch를 함께 정리(EXIT trap)하지만, 팔을 물리적으로 세우는 최종 수단은 E-STOP이다. 그리퍼는 연결 시 auto-calibrate(open/close sweep)하므로 **손가락을 치울 것.**

## 7. 추론 동작 (Euler-ODE 샘플링, receding horizon, 이미지 파이프라인)

> 이 절은 **FM 특유**이며 diffusion의 DDIM 체인과 다르다. diffusion은 노이즈 스케줄러를 DDIM으로 역확산하지만, FM은 **velocity field를 Euler-ODE로 적분**해 액션 시퀀스를 만든다. `fm_server.py`가 **diffusion_server.py와 다른 것은 정확히 네 가지**뿐이고, 나머지(ZMQ 프로토콜, 그리퍼 처리, 큐 로직, 헬스체크, 로깅)는 동일하다.

- **차이 (1) — GENERIC 정책 로드.** diffusion 서버는 `DiffusionPolicy`를 이름으로 하드코딩해 로드하지만, FM 서버는 **lerobot 팩토리로 클래스를 config에서 해석**한다: `cfg = PreTrainedConfig.from_pretrained(ckpt)` → `PolicyCls = get_policy_class(cfg.type)`(`"multi_task_dit"` → `MultiTaskDiTPolicy`) → `PolicyCls.from_pretrained(ckpt, config=cfg)`. 정책 클래스를 import 이름으로 못박지 않는다.
- **차이 (2) — Euler-ODE 적분, DDIM 아님.** FM은 `cfg.num_integration_steps`번의 Euler 스텝으로 ODE를 적분한다(`integration_method=euler`). **노이즈 스케줄러도 `num_inference_steps`도 DDIM 오버라이드도 없다.** 서버는 `--scheduler` 대신 **`--num-integration-steps`**(`FM_NUM_INTEGRATION_STEPS`)를 노출하며, config 변경은 **`from_pretrained` 이전에**, `hasattr(cfg,"num_integration_steps")` 가드 아래 이뤄진다(비-FM 체크포인트가 여기서 크래시하지 않도록). **주의:** `fm_server.py`의 CLI 기본은 `None`(= 체크포인트의 학습값 100 유지)이지만, **셸 런처(`run_fm_server.sh`/`run_ur7e_fm_real.sh`)는 `FM_NUM_INTEGRATION_STEPS=10`을 기본으로 세팅**하므로, 런처로 띄우면 실제로는 **Euler-10**이 걸린다. 100을 그대로 쓰려면 파이썬을 직접 호출하거나 env를 비워야 한다.
- **차이 (3) — 외부 리사이즈 없음.** diffusion 서버는 프레임을 360×640으로 pre-resize하고 `config.resize_shape`가 맞는지 assert했지만, FM 정책은 **매 forward마다 내부에서** `torchvision.Resize(config.image_resize_shape)`(= [224, 224])로 리사이즈한다. 그래서 서버 측 `image_preprocess_fm.decode_jpeg_to_rgb_float_chw`는 프레임을 **네이티브 해상도 그대로** 넘기고, diffusion 경로의 `resize_shape` assert도 **삭제**됐다.
- **차이 (4) — 실제 `task` 문자열 전송.** FM 모델은 **텍스트 조건부**(CLIP text)라, 단일 태스크 diffusion 서버가 `task=""`를 보낸 것과 달리 **매 틱 학습 문구를 그대로** 보낸다: `"put the right banana in the pot"`(`--task`/`FM_TASK`, 기본값 = `DEFAULT_TASK`). 저장된 preprocessor가 이 문자열을 토크나이즈한다.

그 외 추론 동작(diffusion과 공유되는 부분):

- **관측 큐 (`n_obs_steps=2`).** 정책은 2 프레임의 관측 컨텍스트를 조건화에 쓴다. 첫 틱에는 프레임이 1개뿐이므로 lerobot이 첫 관측을 repeat-padding한다 — 별도 priming 없이 첫 틱부터 동작한다.
- **Receding horizon (`n_action_steps=24`).** 서버는 로드 시 `policy.config.n_action_steps=24`를 **`policy.reset()` 이전에** 설정한다(`reset()`이 `maxlen=n_action_steps`인 action deque를 만들기 때문). 결과적으로 **net은 24번의 `act` 호출마다** Euler-ODE로 fresh 액션 시퀀스를 재적분하고(그 틱이 "refill", `policy._queues[ACTION]`이 빌 때만), 그 사이 23틱은 큐에서 pop만 한다. 30Hz 발행 기준 **재적분 주기 ≈ 24/30 = 약 0.8s**. 로드 시 가드로 `n_action_steps <= horizon - n_obs_steps + 1`(24 ≤ horizon−2+1)을 assert한다.
- **30Hz 발행은 협상 불가.** 데이터셋이 30Hz로 기록됐고 브리지 필터도 ~30Hz 리더를 가정한다. 이보다 빠르거나 느리게 발행하면 학습과 다른 유효 다이나믹스가 된다.
- **이미지 전처리 — 네이티브 해상도, byte-parity.** 서버 측 `image_preprocess_fm.py`는 `cv2.imdecode(jpeg)` → **BGR→RGB** → HWC uint8[0,255] → **CHW float32 [0,1]** 만 한다 — **리사이즈도 다른 색공간 변환도 하지 않는다.** 리사이즈(→224)는 **정책 내부**에서 일어난다. 저장된 preprocessor는 rename→batch→**tokenize(task)**→device→normalize(VISUAL=MEAN_STD)뿐이며 리사이즈/색변환을 하지 않으므로, **BGR→RGB는 반드시 여기(서버)에서** 일어나야 한다. BGR을 남기거나 다른 크기로 pre-resize하면 double-resize/색상 뒤집힘으로 **에러 없이 조용히** 정책이 열화된다.
- **카덴스 노브 (env vars / CLI).**

  | 노브 | env var / CLI | 기본 | 효과 |
  |---|---|---|---|
  | Euler 스텝 수 | `FM_NUM_INTEGRATION_STEPS` / `--num-integration-steps` | 셸 런처 **10** (파이썬 직접 호출 시 unset→ckpt 100) | ↓ 낮추면 refill이 빨라지지만 샘플 품질↓. **p99 refill이 예산 초과 시 여기부터 줄일 것(예: 5).** ↑ 100으로 올리려면 레이턴시 여유 확인 |
  | receding horizon | `FM_N_ACTION_STEPS` / `--n-action-steps` | `24` | ↓ 낮추면 재쿼리가 잦아져 반응성↑(단 refill 빈도↑ → 평균 부하↑); ↑ 높이면 부하↓·반응성↓ |
  | 디바이스 | `FM_DEVICE` / `--device` | `cuda` | CPU 폴백은 명시해야 하며 타임아웃 조정 필수(§9) |
  | 포트 | `FM_PORT` / `--port` | `5593` | ZMQ 엔드포인트 (ACT 5591, Diffusion 5592) |
  | task 문자열 | `FM_TASK` / `--task` | `"put the right banana in the pot"` | CLIP text 조건. **학습 문구에서 바꾸지 말 것** |

  > **반응성이 느려 그랩을 놓치면**: 먼저 `FM_N_ACTION_STEPS`를 낮춰(예: 16) 재쿼리를 잦게 한다. 단 refill 빈도가 늘어 평균 GPU 부하가 오르므로 예산 여유를 재확인할 것. refill이 무거운 게 병목이면 `FM_NUM_INTEGRATION_STEPS`를 줄이는 쪽이 더 직접적이다.

## 8. 검증 완료 vs 아직 안 됨

> **정직한 현재 상태.** FM 정책은 **오프라인 open-loop 평가로 체크포인트가 선택**되었고(70k, poseMAE 0.0735), 안전상 중요한 ROS/브리지/핸드셰이크 코드는 **ACT 실기에서 검증된 것과 바이트 단위로 동일**하다. 하지만 **FM 서버가 붙은 전체 경로로 실물 UR7e 팔을 자율 구동한 적은 아직 없다.**

**검증/근거가 있는 것:**

- **open-loop 체크포인트 선택** — held-out 롤아웃 MAE로 70k 선택(poseMAE 0.0735, Euler-10). 배포 기본 Euler-10은 이 eval과 동일한 샘플러.
- **안전/배관 코드 재사용** — `policy_leader_node`, `gello_ur_bridge`, `gello_move_to_start`, Robotiq 스택, 7-D 액션 컨트랙트가 ACT/Diffusion과 **동일 코드**이며, ACT 경로는 실기 엔드투엔드 검증됨(2026-07-08). FM은 이 스택 위에 서버만 교체.

**아직 검증/수행 안 됨:**

- **실제 UR7e 팔 구동** — 이 FM 서버 경로로 실물 팔을 자율 구동한 적이 **한 번도 없다.** handshake·안전 게이팅·클램프는 ACT와 같은 코드라 검증됐지만, **FM 서버가 붙은 전체 경로의 실기 동작은 미확인.**
- **로봇 PC colcon build** — `gello_policy` + `ur_gello_bringup`를 로봇 PC Humble에서 실제로 빌드하고 `fm_deploy.yaml`이 share로 설치되는지 확정 안 됨.
- **타깃 GPU refill 레이턴시** — Euler-10 refill의 p99가 이 로봇 PC의 GPU에서 `act_timeout_s`(0.6s) 안에 드는지 **아직 실측 안 됨.** §6대로 **첫 팔 구동 전에** 서버 `net refill` 로그로 확정할 것(FM 전용 벤치마크 스크립트는 아직 없음).
- **오프라인 CLIP 로드** — `HF_HUB_OFFLINE=1` 하에서 CLIP ViT-B/16이 로봇 PC 캐시/체크포인트에서 문제없이 로드되는지 실기 환경에서 미확인(§4-2).
- **closed-loop 태스크 성공률** — open-loop MAE(§3)는 있으나 실기 grab/placement 성공률은 미측정.

> **요약:** 첫 실기 세션은 (1) 로봇 PC colcon build → (2) 오프라인 CLIP 캐시 확인 → (3) 서버 standalone으로 refill 레이턴시 sanity-check → (4) 카메라/그리퍼 라이브 확인 → (5) 감독하에 `~/start_execution` 순서로 진행할 것.

## 9. 트러블슈팅

- **포트 5593 충돌 / 서버가 안 뜸.** FM=5593, Diffusion=5592, ACT=5591. `run_ur7e_fm_real.sh`는 시작 전에 5593이 이미 listen 중이면 **거부**한다(stale 서버로 팔이 구동되는 것을 막기 위함). 잔여 프로세스 정리: `pkill -f fm_server.py ; ss -ltnp | grep :5593`. 리더는 `fm_deploy.yaml`의 `act_port=5593`으로 서버를 찾으므로, 포트를 바꾸면 yaml과 서버(및 런처 `FM_PORT`) 양쪽을 맞춰야 한다.
- **오프라인 CLIP 로드 실패 (`from_pretrained`가 허브를 침 / 캐시 없음).** 서버는 `HF_HUB_OFFLINE=1`·`TRANSFORMERS_OFFLINE=1`로 뜨므로 `CLIPVisionModel`/`CLIPTextModel.from_pretrained`가 **로컬 캐시/체크포인트만** 본다. 캐시가 비어 있으면 실패한다. 해결: 온라인일 때 한 번 로드해 캐시를 채우거나, `HF_HOME`/`TRANSFORMERS_CACHE`를 캐시가 있는 경로로 지정하거나, 체크포인트에 CLIP 가중치가 baked-in되어 있는지 확인(§4-2).
- **refill이 느려 FAULT-loop / 팔이 자꾸 멈춤.** Euler refill이 `act_timeout_s`(0.6s)를 넘기고 있다는 뜻(또는 refill 블록이 obs를 얼려 `obs_timeout_s`(0.7s)를 넘긴 경우 — 둘 중 먼저 걸리는 쪽이 FAULT). **선호되는 해결책은 워치독을 넓히는 게 아니라**(§6) `FM_NUM_INTEGRATION_STEPS`를 낮추는 것이다(예: 5). 먼저 서버 `net refill` 로그로 실제 예산 초과인지 확인할 것. GPU가 아니라 CPU로 돌고 있으면 근본 원인이 그것이다(아래).
- **서버가 종료 코드 3 / "CUDA unavailable"로 죽음.** `resolve_device()`는 `--device cuda`인데 CUDA가 없으면 **자동 CPU 폴백을 하지 않고 코드 3으로 거부**한다(CPU Euler 적분이 0.6s 타임아웃을 넘겨 FAULT-loop를 유발하므로 의도적). 해결: (a) CUDA/드라이버를 고치거나, (b) 정말 CPU로 돌려야 한다면 `--device cpu`(또는 `FM_DEVICE=cpu`)를 명시하고 **동시에** `fm_deploy.yaml`의 세 워치독을 **모두** 크게 올릴 것 — `act_timeout_s`뿐 아니라 **`obs_timeout_s`도 반드시 함께** 올려 `act_timeout_s < obs_timeout_s < staleness_timeout_s` 순서를 유지해야 한다(그러지 않으면 느린 CPU refill이 obs를 오래 얼려 **가짜 obs-stale FAULT**를 낸다). 이는 어디까지나 CPU 임시방편이며, 정상 배포의 올바른 노브는 `FM_NUM_INTEGRATION_STEPS` 축소다.
- **resize / policy-type mismatch (조용한 열화 또는 로드 실패).** FM 경로는 **네이티브 해상도**를 서버에 넘기고 정책이 224로 내부 리사이즈한다 — diffusion처럼 360×640으로 pre-resize하면 **double-resize**로 조용히 열화된다. `image_preprocess_fm`(FM용)과 `image_preprocess`(diffusion용)를 헷갈리지 말 것. 또한 체크포인트가 `multi_task_dit`/`flow_matching`이 아니면 `get_policy_class(cfg.type)`가 다른 클래스를 로드하거나, `--num-integration-steps`가 `hasattr` 가드에 걸려 **경고만 남기고 무시**된다(로그 `WARNING: --num-integration-steps=... ignored`) — 로드 로그의 `policy class:`·`objective=`·`num_integration_steps=`·`image_resize_shape=` 줄로 올바른 체크포인트인지 확인하라.
- **cam1/cam2 시리얼이 뒤바뀜.** 반드시 **시리얼로 바인딩**할 것(`serial_no`). 잘못 바인딩되면 정책은 크래시하지 않고 **조용히 열화**된다. all-digit 시리얼은 따옴표로 감쌀 것: `serial_no:="'151623020789'"`.
- **가짜(spurious) FAULT가 시작 직후에 뜬다.** 서버 워밍업이 실패/스킵됐을 가능성(로그의 `warmup done` 확인), 또는 첫 refill이 예산을 넘김(→ Euler 스텝 줄이기). FM은 첫 EXECUTE에서 곧바로 refill을 한 번 하므로 워밍업(+CLIP lazy-init)이 특히 중요하다.
- **`~/start_execution`이 계속 거부됨.** 라이브 자세가 `start_pose`에서 0.1rad 넘게 떨어졌거나("live pose not within 0.1 rad" 로그), fresh 관측 셋(두 카메라 + 그리퍼 위치)이 완전하지 않다는 뜻. handshake 수렴 로그와 카메라·그리퍼 토픽 발행을 확인하라.
- **`--checkpoint` 누락.** `FM_CHECKPOINT`(또는 `--checkpoint`)가 없으면 서버는 종료 코드 2로 즉시 죽는다; 런처는 그 전에 `ERROR: FM_CHECKPOINT is required`로 거부한다.

## 10. Diffusion/ACT 배포와의 관계 / HIL-SERL으로 가는 길

**이 문서는 [`GELLO_UR7E_DIFFUSION_DEPLOY.md`](./GELLO_UR7E_DIFFUSION_DEPLOY.md)의 FM 형제(sibling)다.** 세 배포는 **같은 ROS2 패키지·같은 `policy_leader_node`·같은 브리지·같은 핸드셰이크·같은 안전 클램프·같은 그리퍼 스택**을 쓴다. 유일한 차이는:

| 항목 | ACT | Diffusion | Flow-Matching |
|---|---|---|---|
| 추론 서버 | `policy_server/act_server.py` | `policy_server/diffusion_server.py` | `policy_server/fm_server.py` |
| 정책 | `ACTPolicy` | `DiffusionPolicy` (하드코딩 로드) | `MultiTaskDiTPolicy` (**generic `get_policy_class` 로드**) |
| 샘플링 | 트랜스포머 forward 1회 | DDIM-10 역확산 | **Euler-ODE 적분** (`num_integration_steps`) |
| 조건화 | — | 단일 태스크 (`task=""`) | **텍스트 조건부 (CLIP text, 실제 `task` 문자열)** |
| 이미지 리사이즈 | (해당 경로) | 서버가 360×640 pre-resize + assert | **네이티브 전달, 정책이 224 내부 리사이즈 (assert 삭제)** |
| ZMQ 포트 | **5591** | **5592** | **5593** |
| receding horizon | `n_action_steps=30` | `n_action_steps=32` (≈1.07s) | `n_action_steps=24` (≈0.8s @30Hz) |
| 타임아웃 | `act_timeout_s=0.5`, staleness 0.5 | `0.6 < 0.7 < 0.8` | **`0.6 < 0.7 < 0.8`** (diffusion 값 계승; refill이 무겁고 단일 스레드라 refill 블록 동안 obs가 얼기 때문) |
| 추가 의존성 | `transformers` 불필요 | +`diffusers==0.35.2` | +**`transformers`**(CLIP) (+오프라인 `HF_HUB_OFFLINE=1`) |
| 체크포인트 | `Bigenlight/act_banana_in_pot` | `Bigenlight/diffusion_banana_in_pot_joint` (80k) | `Bigenlight/flow_matching_banana_in_pot_joint` (**70k**) |
| 런치 파일 | (자체) | `ur7e_diffusion_real.launch.py` | **재사용** `ur7e_diffusion_real.launch.py` (`params_file:=fm_deploy.yaml`, `act_port:=5593`) |
| 실기 상태 | 2026-07-08 엔드투엔드 실기 검증 | 오프라인(dev PC)만 | **오프라인 open-loop만** — 실기 미검증 (§8) |

액션 컨트랙트(7-D JOINT)가 동일하므로, 안전상 중요한 ROS 코드는 **한 줄도 새로 쓰이지 않았다** — FM 배포는 순전히 서버 교체(+`fm_server.py`/`image_preprocess_fm.py`) + yaml 재타게팅(포트/타임아웃) + diffusion 런치 재사용이다. `fm_server.py`가 `diffusion_server.py`와 다른 것은 정확히 **네 가지**(generic 로드 / Euler-ODE / 외부 리사이즈 없음 / 실제 `task` 전송)이며, 그 외 ZMQ·그리퍼·큐·헬스체크·로깅은 동일하다(§7).

**HIL-SERL으로 가는 길**은 ACT/Diffusion과 공유된다: 무거운 RL 부분(actor/learner, replay buffer, EE-delta processors)은 lerobot의 py3.12 쪽에 얹히고, ROS 쪽에는 `policy_mux_node` 하나만 추가되어 실제 GELLO 리더(사람 개입)와 정책 스트림을 중재한다. 자세한 근거와 리스크는 `DEPLOY_REPO_DECISION.md` §5를 참고할 것.

---

## 관련 문서

- [`GELLO_UR7E_DIFFUSION_DEPLOY.md`](./GELLO_UR7E_DIFFUSION_DEPLOY.md) — **직계 형제 Diffusion 배포**. 이 문서의 템플릿이며, 공유 절차의 상세 버전
- [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md) — **실기 검증된 ACT 배포**. Method A/B·4-터미널 워크플로우·recorder·리커버리 등 공유 절차의 원본
- [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) — 패키지 개요, 노드/토픽 레퍼런스, 빌드
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 사람 GELLO 리더로 하는 move-to-start handshake의 전체 메커니즘(chase/dwell/stillness-gate) — 이 문서의 handshake는 리더가 정책으로 바뀌었을 뿐 로직은 동일
- [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) — 이 정책을 학습시킨 데이터셋을 만든 레코더 (observation/action 컨트랙트의 출처)
- 모델 카드: HF `Bigenlight/flow_matching_banana_in_pot_joint` (아키텍처·학습·open-loop 결과 상세)
