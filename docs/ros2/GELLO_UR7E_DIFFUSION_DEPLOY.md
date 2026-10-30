# GELLO → UR7e Diffusion Policy 실기 배포 — 런북 (REAL)

> ⚠️ **이 문서는 학습된 Diffusion Policy "put right banana in pot" 정책으로 실제 UR7e를 자율 구동하기 위한 실기(real-robot) 런북입니다.**
> 대상은 **로봇 PC** (실제 UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 머신)입니다. 이 배포는 이미 실기 검증된 **ACT 배포**([`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md))와 **같은 ROS2 패키지·같은 안전 스택**을 그대로 쓰고, **추론 서버만** ACT 트랜스포머 대신 Diffusion Policy(DDIM-10 샘플링)로 교체한 형태입니다. JOINT diffusion 모델은 ACT와 **완전히 동일한 7-D 관절 액션 컨트랙트**를 내보내므로, 리더 노드·브리지·핸드셰이크·안전 클램프·그리퍼 스택은 **코드 변경 없이** 재사용됩니다.
>
> **현재 상태 (정직하게):** 이 diffusion 경로는 **개발 PC에서 CPU로 엔드투엔드 오프라인(ZMQ 라운드트립이 유효한 7-D 액션을 돌려주는 것)까지만 검증**되었습니다. **실제 UR7e 팔 구동, 로봇 PC에서의 colcon build, GPU 레이턴시 벤치마크는 아직 수행되지 않았습니다** — §8 참고. **실기 태스크 성능은 물론, 실기 파이프라인 동작 자체도 아직 실물 하드웨어에서 확인되지 않았습니다.** (참고: 형제 ACT 경로는 2026-07-08에 실기 엔드투엔드까지 검증됨.)
>
> - **패키지 개요 / 빌드 방법**: [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) (§ "Diffusion variant")
> - **형제 문서 (실기 검증된 ACT 배포, 이 문서의 뼈대)**: [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md)
> - **설계 근거**: 프로젝트 루트 `DEPLOY_REPO_DECISION.md` (왜 `gello_software`를 확장하는지, 왜 synthetic-leader + ZMQ 분리인지, HIL-SERL로 가는 길)
> - **관련 실기 문서**: [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) (사람 GELLO 리더로 하는 handshake의 전신), [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) (이 정책을 학습시킨 데이터셋을 만든 레코더)

이 패키지(`gello_policy`)는 학습된 정책이 **사람 GELLO 리더 대신** `/gello/joint_states` + 그리퍼를 발행하게 해서, 기존 `gello_ur_bridge` + `gello_move_to_start` handshake + Robotiq Modbus 스택을 **코드 변경 없이** 재사용한다. Diffusion 추론은 별도의 py3.12 프로세스(torch + lerobot + diffusers)에서 돌고, ROS py3.10 프로세스와는 localhost ZMQ로만 통신한다. ACT와 유일하게 다른 것은 **추론 서버**(`policy_server/diffusion_server.py`, DDIM-10 샘플링)와 그것이 **ZMQ 포트 5592**를 쓴다는 점(ACT는 5591)뿐이다.

## 요약 (TL;DR)

| 항목 | 값 |
|---|---|
| 대상 로봇 | UR7e + Robotiq 2F-85, ROS2 **Humble** |
| 체크포인트 | HF `Bigenlight/diffusion_banana_in_pot_joint` (`scripts/download_diffusion_checkpoint.sh`), best = **80k** |
| 액션 공간 | **JOINT 7-D** (6 UR 관절 rad + 그리퍼) — ACT와 완전히 동일한 컨트랙트 |
| 두 프로세스 | Diffusion 서버(py3.12, torch+lerobot 0.6.1+**diffusers 0.35.2**, `policy_server/diffusion_server.py`) ⇄ ZMQ(localhost:**5592**) ⇄ `policy_leader_node`(py3.10 rclpy) |
| 샘플러 | **DDPM으로 학습**(100 timesteps, `squaredcos_cap_v2`)했으나 **배포는 DDIM-10**으로 샘플링(속도) — DDPM-학습 ε 모델에 유효한 샘플러 |
| 정책 → 팔 경로 | `policy_leader_node`가 `/gello/joint_states`(30Hz) + `/robotiq_gripper/command_percent`를 발행 → 기존 `gello_ur_bridge`(250Hz 업샘플, slew clamp) + `gello_move_to_start` handshake + Robotiq Modbus **무변경** |
| 시작 절차 | 핸드셰이크가 HELD start pose로 팔을 파킹 → 오퍼레이터가 명시적으로 `ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger` 호출해야 자율 모션 시작 |
| receding horizon | `n_action_steps=32`, `n_obs_steps=2`, `horizon=64` — **32틱마다** DDIM 재샘플(≈ **1.07s @30Hz**), 그 사이엔 큐에서 pop만 |
| 안전 | 속도는 브리지의 0.625 rad/s slew ceiling(불변), 위치는 `policy_leader_node`의 1.2x envelope 클램프 + live-pose 0.5rad 클램프. **diffusion 신규**: 리더가 refill 틱의 DDIM 샘플링 동안 발행을 **블로킹**(+단일 스레드라 obs도 프리즈)하므로 `act_timeout_s=0.6` < `obs_timeout_s=0.7` < `staleness_timeout_s=0.8`로 리더의 ZMQ 타임아웃을 true primary fault owner로 배치 (§6) |
| 오프라인 검증됨 (개발 PC, CPU) | diffusion_server 로드 + RESET + 1회 ACT ZMQ 라운드트립이 finite (7,) 액션 반환 (엔드투엔드) |
| **아직 안 됨** | **실제 UR7e 팔 구동, 로봇 PC colcon build, GPU 레이턴시 벤치마크** — §8 |
| 실기 전 필수 (**게이트 A**, K=0) | `scripts/benchmark_diffusion_latency.py`를 로봇 PC에서 **첫 팔 구동 전에** 반드시 실행 (§6) — 이 배포의 기본(K=0) 경로에 적용, 앙상블 녹화 여부와 무관 |
| 앙상블 녹화(`DIFFUSION_ENSEMBLE_K>0`) 켤 때만 추가 필수 (**게이트 B**) | `scripts/benchmark_ensemble_batch16.py` — 게이트 A와 **독립적**인 별개 게이트 (§6). **현재 로봇 PC(6GB)에서 FAIL** — 녹화를 원할 때만 관련, 팔 구동 자체와는 무관 |

### 목차

1. [빅 픽처 / 데이터 흐름](#1-빅-픽처--데이터-흐름)
2. [하드웨어 / 환경](#2-하드웨어--환경)
3. [모델 & 체크포인트 다운로드](#3-모델--체크포인트-다운로드)
4. [셋업 절차 (로봇 PC)](#4-셋업-절차-로봇-pc)
5. [배포 실행하기](#5-배포-실행하기)
6. [안전 모델](#6-안전-모델)
7. [추론 동작 (DDIM 샘플링, receding horizon, 이미지 파이프라인)](#7-추론-동작-ddim-샘플링-receding-horizon-이미지-파이프라인)
8. [검증 완료 vs 아직 안 됨](#8-검증-완료-vs-아직-안-됨)
9. [트러블슈팅](#9-트러블슈팅)
10. [ACT 배포와의 관계 / HIL-SERL으로 가는 길](#10-act-배포와의-관계--hil-serl으로-가는-길)

---

## 1. 빅 픽처 / 데이터 흐름

Diffusion 서버(py3.12, torch+lerobot+diffusers)와 `policy_leader_node`(py3.10, rclpy)는 서로 다른 파이썬/venv에 살기 때문에 **localhost ZMQ REQ/REP**로만 통신한다 — `lerobot`은 Python ≥3.12를 요구하는데 ROS2 Humble의 `rclpy`는 CPython 3.10으로 빌드되어 있어 같은 인터프리터를 공유할 수 없다(자세한 이유는 `DEPLOY_REPO_DECISION.md` §3). **ACT 배포와 유일하게 다른 것은 이 서버가 Diffusion Policy를 DDIM-10으로 샘플링한다는 점과 ZMQ 포트가 5592(ACT는 5591)라는 점뿐이다.** 포트를 분리한 이유는 두 서버(ACT/diffusion)가 한 머신에 공존해도 충돌하지 않게 하기 위함이다.

```
┌───────────────────────────────────┐         ZMQ REQ/REP           ┌──────────────────────────────┐
│  diffusion_server.py (py3.12)     │   tcp://127.0.0.1:5592         │  policy_leader_node (py3.10)  │
│  torch + lerobot 0.6.1 + diffusers│◄───────────────────────────────┤  rclpy                       │
│                                    │   reset: {"cmd":"reset"}       │                               │
│  DiffusionPolicy.from_pretrained   │──────────────────────────────► │  HOLD / EXECUTE / FAULT       │
│  (config override: DDIM, steps=10) │   act: {"cmd":"act",           │  state machine                │
│  n_obs_steps=2, horizon=64         │        "state":[7 floats]},    │  (ACT와 100% 동일 노드)         │
│  n_action_steps=32 (receding hor.) │        <cam1 jpeg>,<cam2 jpeg> │                               │
│  select_action (DDIM-10 리샘플)     │◄─────────────────────────────  │                               │
│  image resize 360x640 + BGR2RGB    │   action = [q1..q6, grip_cmd]   │                               │
│  reply: {"ok":true,"action":[7]}   │                                 │                               │
└───────────────────────────────────┘                                 └───────────────┬───────────────┘
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
  /robotiq_gripper/position_percent ─┤→ policy_leader_node가 state[7] 조립 → ZMQ "act" 요청 (port 5592)
  /cam1/.../compressed (JPEG) ─┤   (raw JPEG 그대로 전달, py3.10 쪽에서 cv2/torch 사용 안 함)
  /cam2/.../compressed (JPEG) ─┘
```

핵심 포인트: `policy_leader_node`는 `gello_publisher_node`(사람 GELLO 리더)를 대체하는 **synthetic leader**일 뿐이며, ACT/diffusion 어느 쪽이든 **동일한 policy-agnostic 노드**다. 그 아래의 `gello_ur_bridge` + `gello_move_to_start` + Robotiq Modbus는 리더가 사람이든 ACT든 diffusion이든 전혀 구분하지 못하며 완전히 동일한 코드가 돈다.

## 2. 하드웨어 / 환경

> ### 🔧 카메라 시리얼 정정 (2026-07-29) — **직전 "물리 교체" 정정은 틀렸다**
>
> **이 리그의 RealSense는 한 쌍뿐이고, 한 번도 교체된 적이 없다.**
> 이 자리에 있던 2026-07-28 판은 "카메라 **개체가 물리적으로 교체됐다**. 이전 판의
> `147122072740` / `243222072700`은 이 PC가 커널 로그상 한 번도 열거한 적 없는 하드웨어다"라고
> 주장하며 문서 전체의 시리얼을 `151623020789` / `322743060038`으로 바꿨다.
> **그 주장은 사실이 아니다.** 리포가 "두 쌍"이라 부르던 것은 **같은 카메라 두 대의 서로 다른
> 시리얼 *필드*** 다. 2026-07-29에 **같은 물리 USB 포트**에서 두 값을 동시에 측정했다:
>
> | 포트 | `camera_info.serial_number` | `camera_info.asic_serial_number` | 장치 |
> |---|---|---|---|
> | `4-4.1` | **`147122072740`** | `151623020789` | plain D435 → cam1 **SCENE** (삼각대, 3인칭) |
> | `4-4.3` | **`243222072700`** | `322743060038` | D435IF → cam2 **WRIST** (그리퍼 장착) |
>
> `serial_no:=`는 **`serial_number`에만** 매칭된다. 직접 확인한 결과:
>
> ```
> rs.config().enable_device('151623020789')  ->  NO MATCH   # ASIC 시리얼
> rs.config().enable_device('147122072740')  ->  MATCHED    # 장치 시리얼
> ```
>
> 즉 07-28 판이 문서에 심어 놓은 ASIC 시리얼은 `realsense2_camera`가 **영원히 해석하지 못한다.**
> 이 문서의 시리얼은 전부 **장치 시리얼**(`147122072740` / `243222072700`)로 되돌렸다.
>
> **덫 1 — 저널 grep.** 커널 USB 디스크립터(`journalctl`, `/sys/bus/usb/devices/*/serial`)는
> **ASIC 시리얼**을 노출한다. 그래서 저널을 grep하면 `151623020789`/`322743060038`은 수백 건
> 나오고 `147122072740`/`243222072700`은 **0건**이라, 정확히 "이 호스트는 그 카메라를 본 적이
> 없다"처럼 읽힌다. 커밋 `607e541`이 그렇게 결론짓고 기본값을 ASIC 시리얼로 바꿨다.
> **저널 grep으로 카메라 정체를 다시 유도하지 마라** — 이 방법으로 **서로 반대 방향의** 틀린
> 결론에 도달한 세션이 이미 둘이다. 정체의 권위는 `rs-enumerate-devices` / `pyrealsense2`가
> 보고하는 `serial_number`이고, 코드에서는
> [`ros2_ur_ws/_resolve_camera_serials.sh`](../../ros2_ur_ws/_resolve_camera_serials.sh)다.
>
> **덫 2 — 조용한 실패.** 해석되지 않는 시리얼로 바인딩해도 **시끄럽게 실패하지 않는다.**
> 노드는 정상 기동하고 `ros2 topic info`는 publisher 1을 보고하는데, **프레임이 하나도
> 안 나온다.** 스택 어디에도 "카메라가 안 꽂혔다"와 "설정이 틀렸다"를 구분해 주는 신호가 없다.
> 그래서 시리얼 하드코딩보다
> [`ros2_ur_ws/launch_cameras.sh`](../../ros2_ur_ws/launch_cameras.sh)를 권한다 — 이쪽은
> `_resolve_camera_serials.sh`를 source해 **실제 USB 버스에 대고 시리얼을 해석**한다
> (커밋 `43ba314`, `fb48100`).
>
> cam1/cam2 ↔ 모델 클래스 ↔ 마운트 배정도 확정이다: plain D435 = cam1 = SCENE(삼각대),
> D435IF = cam2 = WRIST(그리퍼 장착). **손목 배정은 녹화 영상으로 증명됐다** — cam2 화면에서
> 그리퍼 손가락은 픽셀이 고정된 채 배경만 흐른다.
> 07-28 판이 "아래 과거 검증 기록은 옛 개체로 수행된 것"이라 적은 주석들도 함께 철회한다.

- **팔**: UR7e, `ros-humble-ur`. **그리퍼**: Robotiq 2F-85 (Modbus RTU, 드라이버가 소유한 socat 브리지 `/tmp/ttyUR` 공유).
- **카메라**: RealSense 2대, **시리얼로 바인딩** (혼동 시 정책이 조용히 열화됨 — §9 참고):
  - cam1 = Intel RealSense **D435**, 시리얼 `147122072740` (SCENE, 삼각대)
  - cam2 = Intel RealSense **D435if**, 시리얼 `243222072700` (WRIST, 그리퍼 장착)
  - 공통 컬러 프로파일 `1280x720x30` (해상도/FPS는 두 카메라 동일해야 함)
  - 위 값은 **장치 시리얼**이다. `journalctl`에 보이는 `151623020789` / `322743060038`은
    **ASIC 시리얼**이고 `serial_no:=`로는 절대 해석되지 않는다(위 정정 박스).

  > **⚠️ 물리적 카메라 배치 = 학습 리그와 반드시 일치.** 한 대는 씬/3인칭 시점을, 다른 한 대는 작업공간 근접(close-up)을 본다. **두 물리 시점과 cam1/cam2 시리얼 할당이 학습 당시 리그와 동일**해야 하며, 어긋나면 정책이 **에러 없이 조용히 열화**된다(정책은 cam1=씬, cam2=근접 같은 고정 배치를 가정하고 학습됨 — `cam1/cam2` 순서는 학습에 고정됨). 실기 첫 구동 전에 **라이브 뷰(`rqt_image_view` 등)를 학습 셋업 사진과 대조**해 확인할 것.

- **ROS2**: 로봇 PC는 **Humble** (rclpy py3.10). `ur_gello_bringup`/`gello_recorder`와 동일 워크스페이스. 필요한 ROS 패키지 설치:

  ```bash
  # UR 드라이버 + RealSense 카메라 스택 (+ librealsense2 SDK)
  sudo apt install ros-humble-ur ros-humble-realsense2-camera ros-humble-librealsense2
  cd gello_software/ros2_ur_ws
  rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
  ```

  > 실 로봇/ROS 최초 셋업 전체(펜던트 External Control 페어링, 기구학 캘리브레이션, 24V 그리퍼)는 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md)를 참고. 이 문서는 그 위에 diffusion 배포만 얹는다.

- **Diffusion 서버**: 별도 **py3.12 venv** (torch cu128 빌드 + lerobot 0.6.1 + **diffusers 0.35.2**). **GPU(CUDA) 필수 (권장이 아님).** `--device cuda`가 기본값이며, CUDA가 없으면 서버는 **자동 CPU 폴백을 하지 않고 종료 코드 3으로 거부**한다 — CPU에서의 DDIM-10 refill은 leader의 `act_timeout_s`(0.6s)를 넘겨 FAULT-loop를 유발하기 때문이다. CPU에서 굳이 돌리려면 `DIFFUSION_DEVICE=cpu`를 명시하고 동시에 `diffusion_deploy.yaml`의 타임아웃을 크게 올려야 한다(§9 참고). **DDIM refill은 ACT의 단일 트랜스포머 forward보다 무겁다**(10-step 역확산) — 그래서 반드시 §6의 레이턴시 벤치마크를 먼저 돌려야 한다.

## 3. 모델 & 체크포인트 다운로드

**모델: `Bigenlight/diffusion_banana_in_pot_joint`** (Hugging Face). 핵심 모델 사실:

| 항목 | 값 |
|---|---|
| 아키텍처 | LeRobot **DiffusionPolicy** — 카메라별 ResNet18(ImageNet 사전학습) + SpatialSoftmax(32 keypoints) → 1D-conv UNet 디노이저 |
| 액션 공간 | **JOINT 7-D** (6 UR 관절 rad + 그리퍼, 절대값). 그리퍼 채널은 사실상 binary |
| 관측 | `observation.state` (7,), `observation.images.cam1/cam2` RGB (네트워크 입력 **360×640**) |
| 학습 노이즈 모델 | **DDPM**, `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2`, `prediction_type=epsilon` |
| **배포 샘플러** | **DDIM, 10 inference steps** (DDPM-학습 ε 모델에 유효한 샘플러 — 같은 beta 스케줄, ~10× 빠른 롤아웃) |
| receding horizon | `horizon=64`, `n_obs_steps=2`, `n_action_steps=32` |
| **best checkpoint** | **step 80,000** |
| 정규화 | STATE/ACTION = MIN_MAX, VISUAL = MEAN_STD (ImageNet stats). 통계는 저장된 pre/post-processor json에 baked-in |

> **체크포인트 선택의 교훈 (읽고 넘어갈 것).** 이 체크포인트는 **open-loop 롤아웃 MAE로 선택**되었다 — **held-out `eval_loss`로 선택하지 않았다.** 학습 중 held-out **디노이징 `eval_loss`는 ~5× 상승**(0.029 → 0.149)해서 naive하게 보면 "심각한 오버핏"으로 보이지만, **배포에 실제로 중요한 open-loop 롤아웃 MAE는 80k까지 계속 개선**되었다. **Diffusion Policy에서 held-out 디노이징 loss는 잘못된(misleading) 오버핏/early-stop 신호다 — 체크포인트는 open-loop MAE로 고를 것.** (형제 ACT는 두 신호가 일치했으므로 이는 diffusion 특유의 함정이다.)
>
> **@80k open-loop 성적** (held-out eps 45–50, DDIM-10): **poseMAE 0.0845 rad, gripAcc 0.953** (poseMAE는 60k 이후 ~0.085에서 plateau, gripAcc는 80k에서 최고). 이는 **open-loop 지표이지 closed-loop 태스크 성공률이 아니다** — 실기 성공률은 아직 측정된 바 없다(§8).

`scripts/download_diffusion_checkpoint.sh`가 체크포인트를 gitignored 로컬 디렉터리로 받아온다. 받은 디렉터리 ROOT에 `config.json` + `model.safetensors` + `policy_pre/postprocessor` json이 함께 들어 있는 HF 모델 레이아웃이므로, 서버 `--checkpoint`는 **그 디렉터리를 직접 가리키면 된다**:

```bash
cd gello_software/ros2_ur_ws/src/gello_policy

# 체크포인트 다운로드 (stdout 마지막 줄이 --checkpoint에 넘길 경로):
CKPT=$(./scripts/download_diffusion_checkpoint.sh | tail -1)
DIFFUSION_CHECKPOINT="$CKPT" ./scripts/run_diffusion_server.sh
```

환경변수: `ACT_VENV`(huggingface-cli를 찾을 py3.12 venv, 기본 `ros2_ur_ws/act_venv`), `DOWNLOAD_DIR`(기본 `<pkg>/checkpoints`), `CKPT_REPO`(기본 `Bigenlight/diffusion_banana_in_pot_joint`). `huggingface-cli`/`hf`가 검색되며, 없으면 `pip install -U 'huggingface_hub[cli]'`를 안내하고 종료한다. (다운로드 CLI의 `hf` vs `huggingface-cli` 함정은 ACT 문서 §4.5-4를 참고 — 동일하게 적용됨.)

관련 데이터셋: `Bigenlight/banana_in_pot_lerobot_v3` (학습에 쓰인 LeRobot v3 joint-space 데이터셋, 51 episodes / 21,524 frames).

## 4. 셋업 절차 (로봇 PC)

1. **colcon 빌드 — `gello_policy` + `ur_gello_bringup`** (Humble, py3.10). `ur7e_diffusion_real.launch.py`는 `ur_gello_bringup` 노드(`gello_ur_bridge`, `gello_move_to_start`, `robotiq_gripper_modbus`)를 돌리고 `ur_robot_driver`의 `ur_control.launch.py`를 include하므로, **`gello_policy`만 빌드하면 안 되고 `ur_gello_bringup`도 반드시 함께 빌드**해야 한다. (`policy_leader_node`만 이 빌드에 포함되며 — ACT와 **완전히 같은 노드** — `policy_server/`는 **설치되지 않고** 자기 venv에서 소스 디렉터리째로 실행됨.)

   ```bash
   cd gello_software/ros2_ur_ws
   rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
   colcon build --packages-select gello_policy ur_gello_bringup
   # ...또는 가장 간단하게 워크스페이스 전체:  colcon build
   source install/setup.bash
   ```

2. **py3.12 venv 생성 + Diffusion 서버 의존성 설치** (`policy_server/requirements-diffusion.lock` 참고 — ACT lock과 동일 핀에 **`diffusers==0.35.2`가 추가**됨. torch는 **cu128** 빌드이므로 별도 index-url 필요). venv는 **`ros2_ur_ws/act_venv`에** 만든다 — run 스크립트가 기대하는 기본 위치(`ACT_VENV=ros2_ur_ws/act_venv`)다. 다른 곳에 두려면 `export ACT_VENV=/your/path`:

   ```bash
   cd gello_software/ros2_ur_ws        # 여기서 venv 생성
   python3.12 -m venv act_venv
   act_venv/bin/pip install --upgrade pip
   act_venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   act_venv/bin/pip install -r src/gello_policy/policy_server/requirements-diffusion.lock
   ```

   > **ACT venv를 이미 만들었다면** 대부분의 핀이 동일하므로 같은 venv에 `diffusers==0.35.2`만 추가 설치해도 된다. lock 파일 헤더 주석을 먼저 확인할 것(`lerobot==0.6.1`은 GitHub main에서 설치 — ACT 문서 §4.5-5의 주의사항이 동일하게 적용됨: torch/torchvision을 다운그레이드시키지 말 것).
   >
   > **첫 실행 캐시 노트.** ResNet18 백본은 ImageNet 사전학습 가중치를 쓴다. 오프라인/방화벽 뒤 로봇 PC에서 첫 실행 시 torchvision이 이 가중치를 받으려다 멈출 수 있다 — `run_ur7e_diffusion_real.sh`를 띄우는 셸에서 `export TORCH_HOME=/path/to/torch_cache` (및 캐시가 채워졌으면 `export HF_HUB_OFFLINE=1`)을 미리 설정할 것.

3. **RealSense 카메라 2대를 정확한 시리얼→네임스페이스 매핑으로 기동** (ACT와 동일).
   **권장은 시리얼을 직접 적지 않는 것** — [`ros2_ur_ws/launch_cameras.sh`](../../ros2_ur_ws/launch_cameras.sh)가
   `_resolve_camera_serials.sh`로 실제 USB 버스에 대고 시리얼을 해석하고, 두 스트림이 ~30 Hz로
   흐를 때까지 기다린 뒤 cam1/cam2 뷰어를 띄운다:

   ```bash
   cd ~/gello_software/ros2_ur_ws
   ./launch_cameras.sh              # 카메라 2대 + 뷰어 (Ctrl-C 한 번으로 정리)
   # VIEW=false ./launch_cameras.sh # 뷰어 없이 카메라만
   ```

   손으로 띄우는 fallback (값은 반드시 **장치 시리얼**, ASIC 시리얼이 아니다 — §2 정정 박스):

   ```bash
   # cam1 (D435, 147122072740 — SCENE) → 네임스페이스 cam1
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam1 camera_namespace:=cam1 \
       serial_no:="'147122072740'" rgb_camera.color_profile:="'1280x720x30'" &
   # cam2 (D435if, 243222072700 — WRIST) → 네임스페이스 cam2
   ros2 launch realsense2_camera rs_launch.py \
       camera_name:=cam2 camera_namespace:=cam2 \
       serial_no:="'243222072700'" rgb_camera.color_profile:="'1280x720x30'" &

   # 압축 이미지 토픽이 ~30Hz로 나오는지 확인 (수동 경로에서는 필수 —
   # `ros2 topic info`의 publisher 수는 시리얼이 안 맞아도 1로 나온다):
   ros2 topic hz /cam1/cam1/color/image_raw/compressed
   ros2 topic hz /cam2/cam2/color/image_raw/compressed
   ```

4. **⚠️ 첫 팔 구동 전 필수 — 레이턴시 벤치마크.** DDIM-10 refill이 이 로봇 PC의 GPU에서 안전 예산 안에 드는지 반드시 확인한다(§6). 자세한 절차/판정 기준은 §6에 있다:

   ```bash
   act_venv/bin/python src/gello_policy/scripts/benchmark_diffusion_latency.py \
       --checkpoint "$CKPT" --device cuda --num-inference-steps 10
   ```

## 5. 배포 실행하기

```bash
cd gello_software/ros2_ur_ws
DIFFUSION_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_diffusion_real.sh
```

`run_ur7e_diffusion_real.sh`는 먼저 py3.12 diffusion 서버를 백그라운드로 띄우고(PID 저장, `trap 'kill $PID' EXIT`로 Ctrl-C 시 함께 정리), 서버가 살아있고 **포트 5592**가 listen 중인지 확인한 뒤, Humble `ros2 launch gello_policy ur7e_diffusion_real.launch.py`를 실행한다.

- **Method A (기본)**: 펜던트에서 External Control 프로그램을 로드하고 **Play** — `HEADLESS` 미설정/`false`.
- **Method B (헤드리스)**: `HEADLESS=true DIFFUSION_CHECKPOINT=/path/... ./run_ur7e_diffusion_real.sh` — 펜던트 Play 없이 드라이버가 직접 URScript를 보냄. 로봇이 **REMOTE 모드**여야 함.

> **⚠️ 카메라를 먼저 띄울 것.** diffusion 배포도 카메라 없이는 돌지 않는다: EXECUTE는 매 틱(또는 최소 refill 틱마다) fresh `cam1`/`cam2` 프레임을 요구하고, `~/start_execution`과 in-EXECUTE obs-freshness watchdog이 카메라가 없거나 stale하면 각각 arming을 거부/FAULT시킨다(§6). 실행 **전에** §4의 카메라 기동(3번)으로 두 RealSense를 띄우고 `ros2 topic hz`로 확인하라. (mock/fake-hardware dry-run 시 그리퍼 위치 토픽을 fake 발행해야 하는 등의 주의사항은 ACT 문서 §5의 "Dry-run / mock" 항목과 동일하게 적용된다.)

### 시작 handshake 타임라인

1. **t=0s** — `ur_control.launch.py`: UR7e 드라이버 + `scaled_joint_trajectory_controller` active, `forward_position_controller` inactive (로드만).
2. **t=6s** — `policy_leader_node` 기동, **HOLD** 상태로 진입. 매 틱 고정 `start_pose`
   `q = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` (rad, UR 관절 순서)를 완전히 정지된 상태로 발행 — 서버는 아직 쿼리하지 않음. 동시에 `gello_ur_bridge`가 `start_paused:=true`로 pre-spawn.
3. **t=8s** — `gello_move_to_start`가 이 HELD start pose로 수렴 게이트 chase 핸드셰이크를 수행(메커니즘은 [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) §2와 100% 동일 — 다른 점은 리더가 사람 손이 아니라 이 고정 pose라는 것뿐). 수렴하면 STRICT 컨트롤러 전환 → 브리지 `~/resume` 호출.
4. **팔이 start_pose에서 정지(파킹)한다.** 이 시점까지 **자율 모션은 전혀 없다** — 리더가 완전히 정지해 있기 때문.
5. **오퍼레이터가 명시적으로** 자율 실행을 시작한다:

   ```bash
   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
   ```

   이 호출은 (a) 라이브 팔 자세가 `start_pose`에서 ~0.1 rad 이내인지 확인하고, (b) ZMQ `reset`을 서버에 보내 diffusion의 action queue를 비운 뒤, (c) **EXECUTE**로 진입한다. 이후부터 매 틱 서버를 쿼리하고(refill 틱에서만 실제 DDIM 재샘플, 나머지는 큐에서 pop) 클램프된 타깃을 발행한다 — 이때부터가 진짜 자율 모션 시작이다.

6. **언제든 일시정지(pause)하려면**:

   ```bash
   ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
   ```

   현재 라이브 자세를 그대로 HOLD로 전환해 그 자리에서 팔을 정지시킨다. **그리퍼는 마지막으로 명령된 값을 그대로 유지**한다(그랩 도중 `~/hold`를 걸어도 물체를 놓치지 않음). 재-arm은 여전히 `start_pose` ±0.1rad 근처를 요구한다.

(Method A/B의 코드 동등성, 4-터미널 워크플로우, `gello_recorder`로 결과 로깅, `Ctrl-C` 후 재실행으로 start_pose 리셋하는 리커버리 절차는 **ACT 문서 §5와 완전히 동일**하다 — 명령의 `run_ur7e_act_real.sh` → `run_ur7e_diffusion_real.sh`, `ACT_CHECKPOINT` → `DIFFUSION_CHECKPOINT`, 포트 5591 → 5592만 바뀐다. 재-chase 모션이 collision-unaware라는 경고도 동일.)

## 6. 안전 모델

이 정책은 실제 로봇을 자율 구동하므로 이 섹션을 가볍게 넘기지 말 것. **속도/위치 안전 스택은 ACT와 완전히 동일**하며(아래 처음 세 항목), 그 뒤에 **diffusion 특유의 신규 항목**이 추가된다.

- **속도 상한 — 절대 넘지 않음.** `policy_leader_node`가 무엇을 발행하든, 그 아래 **`gello_ur_bridge`는 무변경**이며 자신의 `max_step_rad` slew 클램프(배포 설정 지속 **0.625 rad/s**, UR 한계 3.14 rad/s 대비 마진)를 무조건 적용한다. diffusion이 아무리 급격한 타깃을 내도 팔은 이 속도 이상으로 움직이지 않는다.
- **위치 클램프 — `policy_leader_node`가 매 EXECUTE 틱마다, 이 순서로:** (a) **1.2x 데이터셋 envelope로 clip**(`joint_limits_lo/hi`, `config/joint_safety_limits.json`, OOD 출력 가드) → (b) live `/joint_states` 대비 **`max_dev_rad`(기본 0.5 rad)로 clip**. 둘 중 하나라도 발동하면 throttled `WARN`을 남긴다. (이 클램프는 ACT 배포와 **바이트 단위로 동일한 config**를 쓴다 — 액션 컨트랙트가 같으므로.)
- **관측 freshness watchdog + fail-silent FAULT.** EXECUTE에서 4개 관측(`joint_states`, `grip_pos`, `cam1`, `cam2`) 중 하나라도 stale하거나(`obs_timeout_s`), ZMQ 타임아웃/에러/`{"ok":false}` 응답을 받으면 노드는 즉시 **FAULT**로 전이해 `/gello/joint_states` 발행을 **완전히 중단**한다 → 브리지 staleness watchdog이 팔을 정지시킨다. **마지막 유효 타깃을 계속 재발행하지 않는다.** 복구 시 브리지는 라이브 자세에서 soft-start로 재-seed하며, 재-arm은 `start_pose` 근처에서만 허용된다.

### ⚠️ Diffusion 특유의 신규 안전 항목 (반드시 이해할 것)

- **리더는 refill 틱의 DDIM 샘플링 동안 발행을 블로킹한다 (그리고 관측 갱신도 멈춘다).** ACT의 트랜스포머 forward는 CUDA에서 10~50ms로 매우 빠르지만, **diffusion의 refill 틱은 DDIM-10 역확산 전체를 돌리므로 훨씬 무겁다.** 큐가 빈 refill 틱에서 `policy_leader_node`는 서버의 ZMQ 응답을 **동기적으로 기다리며 그동안 `/gello/joint_states`에 발행하지 못한다.** 게다가 **이 노드는 단일 스레드**라, refill 블록 동안 리더는 **카메라/관측 콜백도 서비스하지 못해** obs 타임스탬프가 refill 소요 시간 D만큼 그대로 **얼어붙는다.** 이 refill 갭이 obs-freshness watchdog이나 브리지 staleness watchdog을 실수로 트립시키지 않도록, **세 워치독**의 순서를 **명시적으로** 배치했다:

  | 파라미터 | 값 | 소유자 | 의미 |
  |---|---|---|---|
  | `act_timeout_s` | **0.6s** | `policy_leader_node` (리더) | ZMQ REQ(recv) 타임아웃. 이 시간 안에 DDIM refill이 안 끝나면 리더가 **먼저** FAULT — **true primary fault owner** |
  | `obs_timeout_s` | **0.7s** | `policy_leader_node` (리더) | 관측 freshness watchdog. **0.5→0.7로 상향** — 단일 스레드 노드가 refill 블록 동안 카메라 콜백을 돌리지 못해 obs 타임스탬프가 D만큼 얼기 때문. 반드시 `act_timeout_s`보다 **위**에 있어야 정상 refill(0.47~0.6s)이 **가짜 obs-stale FAULT**를 내지 않는다 |
  | `staleness_timeout_s` | **0.8s** | `gello_ur_bridge` (브리지) | 발행이 끊긴 지 이만큼 지나면 브리지가 팔을 정지. 리더의 정상 ≤0.6s refill 갭보다 **넉넉히 위** |

  **불변식(invariant): `act_timeout_s` (0.6) < `obs_timeout_s` (0.7) < `staleness_timeout_s` (0.8).** 이렇게 배치해야 **리더의 ZMQ 타임아웃(0.6s)이 진짜 primary fault owner**가 되고, 그 위에 obs 워치독을 두어 refill 블록으로 obs가 얼어도 정상 refill이 가짜 obs-stale FAULT를 내지 않으며, **진짜로 얼어붙은 카메라는 여전히 잡힌다**(다만 0.2s 늦게 = `obs_timeout_s`에서). 세 값 모두 `config/diffusion_deploy.yaml`에 있으며 — ACT의 값(양쪽 0.5s)보다 넓힌 이유는 DDIM refill의 추가 지연과 그 블록 동안의 obs 프리즈를 흡수하기 위해서다.

  > **각주(단일 스레드 블록의 파급).** rclpy는 발행 타이머가 밀리면 큐에 쌓인 카메라 프레임을 드레인하기 **전에** 밀린 발행 타이머를 먼저 돌린다. 그래서 **refill 직후 첫 틱은 obs가 ~D만큼 오래된 상태**로 관측을 조립한다. 이는 **한 틱짜리(one-tick)로 자기 보정**되며(다음 틱에서 fresh 프레임이 드레인됨), `obs_timeout_s`(0.7)가 `act_timeout_s`(0.6) 위에 있는 한 FAULT를 유발하지 않는다.

- **⚠️ 첫 팔 구동 전에 반드시 `scripts/benchmark_diffusion_latency.py`를 로봇 PC에서 돌릴 것.** 이 벤치마크는 diffusion_server와 **똑같은 방식으로**(DDIM-N) 정책을 로드해, 매번 큐를 강제로 비운 full-refill 추론을 여러 번 돌리고 min/mean/p50/p95/**p99** 레이턴시와 0.5s 브리지 예산 대비 판정을 출력한다:

  ```bash
  act_venv/bin/python src/gello_policy/scripts/benchmark_diffusion_latency.py \
      --checkpoint "$CKPT" --device cuda --num-inference-steps 10 --iters 50
  ```

  **판정 규칙: p99 refill 레이턴시가 ~0.5s를 넘으면, 워치독 타임아웃을 넓히지 말고 `DIFFUSION_NUM_INFERENCE_STEPS`를 줄여라**(예: 5). 타임아웃을 더 넓히는 것은 팔 워치독을 약화시키는 것이므로 **금지**한다. DDIM step 수를 줄이는 것이 refill을 예산 안으로 되돌리는 올바른 노브다. (참고: 벤치마크가 재는 순수 추론 레이턴시는 **브리지가 실제로 겪는 발행 갭의 하한(lower bound)**이다 — JPEG 디코드 + ZMQ/JSON 왕복 오버헤드(~10-30ms)가 빠져 있으므로, 실제 갭은 이보다 조금 더 크다.)

  ```bash
  # p99가 예산을 넘으면 — 타임아웃을 넓히지 말고 step을 줄인다:
  DIFFUSION_CHECKPOINT=/path/... DIFFUSION_NUM_INFERENCE_STEPS=5 ./run_ur7e_diffusion_real.sh
  ```

### ⚠️ 게이트 A vs 게이트 B — 앙상블 녹화(ensemble side-channel)와의 관계 (반드시 구분할 것)

이 배포 문서가 지금까지 다룬 것은 **게이트 A뿐**이다 — `DIFFUSION_ENSEMBLE_K`(기본 `0`, 즉 앙상블 side-channel 완전 OFF)로 팔을 구동하는 **기본 K=0 경로**의 안전 게이트다. 만약 실기에서 **[`GELLO_DIFFUSION_ENSEMBLE_RECORDER.md`](./GELLO_DIFFUSION_ENSEMBLE_RECORDER.md)**의 16-샘플 불확실성 앙상블 녹화 기능(`DIFFUSION_ENSEMBLE_K>0`)까지 켜고 싶다면, **완전히 별개의 두 번째 게이트**를 통과해야 한다. 둘을 혼동하면 안 된다 — 게이트 A는 로봇 팔을 구동하는 모든 diffusion 배포에 항상 필요하고, 게이트 B는 앙상블 녹화를 켤 때만 추가로 필요하다.

| | **게이트 A** (K=0, 팔 구동 전 필수) | **게이트 B** (K>0, 앙상블 녹화 켤 때만) |
|---|---|---|
| 스크립트 | `scripts/benchmark_diffusion_latency.py` | `scripts/benchmark_ensemble_batch16.py` |
| 언제 필요한가 | **모든** diffusion 배포 — 앙상블을 켜든 안 켜든, 이 게이트를 통과해야 팔을 움직일 수 있다 | `DIFFUSION_ENSEMBLE_K`를 `0`이 아닌 값으로 설정해 앙상블 녹화를 켜고 싶을 때만. 녹화 없이 K=0으로만 팔을 구동한다면 이 게이트는 불필요 |
| 무엇을 재는가 | 실시간 제어가 의존하는 단일-샘플 DDIM-N refill의 레이턴시 (`select_action` 1회) | 배경 스레드에서 계속 도는 batch-16 앙상블 샘플링이 **그 실시간 refill을 지연시키는지** (contention) |
| 정확한 명령 | `act_venv/bin/python src/gello_policy/scripts/benchmark_diffusion_latency.py --checkpoint "$CKPT" --device cuda --num-inference-steps 10 --iters 50` | `act_venv/bin/python src/gello_policy/scripts/benchmark_ensemble_batch16.py --checkpoint "$CKPT" --device cuda --num-inference-steps 10 --ensemble-k 16 --iters 50` |
| PASS 기준 | 종료 코드 0 — 스크립트가 `VERDICT: OK`(p99 ≤ 400ms) 또는 `VERDICT: MARGINAL`(p99 ≤ 500ms 브리지 예산)을 출력. `VERDICT: TOO SLOW`(p99 > 500ms, 종료 코드 1)면 진행 금지 — 위 규칙대로 `DIFFUSION_NUM_INFERENCE_STEPS`를 줄이고 재측정 | 종료 코드 0 — 스크립트가 `VERDICT: PASS`를 출력 (contended p99 **< 500ms**, `act_timeout_s=0.6s` 아래 마진). `VERDICT: FAIL`(종료 코드 1)이면 그 GPU에서 `DIFFUSION_ENSEMBLE_K`를 절대 켜지 말 것 |
| 현재 상태 | 로봇 PC에서 **아직 미실행** (§8 "아직 검증/수행 안 됨") | **현재 로봇 PC(RTX 3060 Laptop, 6GB)에서 FAIL** — K=16(contended p99 811ms)은 물론 **K=1조차 FAIL**(contended p99 545ms, baseline 210ms) — [`GELLO_DIFFUSION_ENSEMBLE_RECORDER.md`](./GELLO_DIFFUSION_ENSEMBLE_RECORDER.md) §6 |

**두 게이트는 서로 독립이다.** 게이트 A를 통과했다고 게이트 B가 자동으로 통과되는 것이 아니고, 그 반대도 마찬가지다 — 검증하는 대상 자체가 다르다(A: 단일 realtime 추론이 안전 예산 안에 드는가, B: 배경 앙상블이 그 realtime 추론을 지연시키지 않는가). GPU/드라이버/체크포인트가 바뀌면 두 게이트 모두 다시 실행해야 한다(`GELLO_DIFFUSION_ENSEMBLE_RECORDER.md` §9).

앙상블을 켜도 정책→ROS→팔의 실시간 제어 코드는 **한 줄도 바뀌지 않는다**(`diffusion_server.py`에만 additive-only 훅이 추가됨, `RECORDER.md` §3) — 그래서 앙상블을 켜기 위해 게이트 A를 다시 통과할 필요는 없지만, 게이트 B는 반드시 별도로 필요하다. **참고로 하드웨어 문제와 별개로**, K=64까지 스윕한 오프라인 연구(`GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md` TL;DR/§7)도 자체 결론으로 `DIFFUSION_ENSEMBLE_K=0` 유지를 권고한다 — across-sample variance는 유용한 실기 불확실성 신호라기보다 photometric-corruption tripwire에 가깝다는 것이 결론이다.

**앙상블 녹화를 실제로 켜는 법 (요약, 상세는 별도 문서).** 이 배포의 실행 명령(`run_ur7e_diffusion_real.sh`)에 `DIFFUSION_ENSEMBLE_K=16`을 추가하면 앙상블 side-channel이 켜지지만, **오직 위 게이트 B를 PASS한 뒤에만** 켤 것. 정확한 실행 절차(런처 순서, GUI, 파일 경로)는 이 문서의 범위가 아니다 — [`GELLO_DIFFUSION_ENSEMBLE_RECORDER.md`](./GELLO_DIFFUSION_ENSEMBLE_RECORDER.md) §5b "How to run it (real-robot recording)"를 참고할 것.

- **오퍼레이터 게이트 + 워밍업.** 핸드셰이크만으로는 자율 모션이 시작되지 않으며 `~/start_execution`이 필수 게이트다. diffusion_server는 로드 시 더미 관측으로 2회 추론을 미리 돌려(→ 그 뒤 `reset()`) CUDA 커널을 워밍업하므로, 실제 첫 refill 틱이 `act_timeout_s`를 넘겨 가짜 FAULT를 내지 않는다.
- **단일 그리퍼 writer.** `ur7e_diffusion_real.launch.py`는 `gello_gripper_bridge`를 launch하지 않는다 — `policy_leader_node`가 `/robotiq_gripper/command_percent`의 유일한 publisher다.

## 7. 추론 동작 (DDIM 샘플링, receding horizon, 이미지 파이프라인)

> 이 절은 **diffusion 특유**이며 ACT의 청크-큐 동작과 다르다. ACT는 한 번의 트랜스포머 forward로 청크를 만들지만, diffusion은 **DDIM 역확산 체인**으로 액션 시퀀스를 샘플한다.

- **DDIM-10 디노이징.** 모델은 **DDPM으로 학습**(`num_train_timesteps=100`, `squaredcos_cap_v2`, ε-prediction)되었지만, 배포 시 서버가 로드 시점에 스케줄러를 **DDIM으로, `num_inference_steps=10`으로 오버라이드**한다. DDIM은 DDPM-학습 ε 모델에 유효한 샘플러이며(같은 beta 스케줄) 전체 100-step DDPM보다 ~10× 빠르다. **평범한 `from_pretrained`는 config의 DDPM/100을 그대로 유지하므로**, 서버는 반드시 `PreTrainedConfig`를 로드해 `noise_scheduler_type`/`num_inference_steps`를 덮어쓴 뒤 `DiffusionPolicy.from_pretrained(config=cfg)`로 로드한다(이것이 ACT 대비 핵심 차이).
- **관측 큐 (`n_obs_steps=2`).** diffusion은 2 프레임의 관측 컨텍스트를 쌓아서 조건화한다. **첫 틱에는 프레임이 1개뿐이므로 lerobot이 첫 관측을 repeat-padding해 n_obs_steps=2를 채운다** — 별도 priming이 필요 없고 첫 틱부터 바로 동작한다.
- **Receding horizon (`n_action_steps=32`).** 서버는 로드 시 `policy.config.n_action_steps=32`를 **`policy.reset()` 이전에** 설정한다(`reset()`이 `maxlen=n_action_steps`인 action deque를 만들기 때문). 결과적으로 **net은 32번의 `act` 호출마다** DDIM-10으로 fresh 액션 시퀀스를 재샘플하고(그 틱이 "refill"), 그 사이 31틱은 큐에서 pop만 한다. 30Hz 발행 기준 **재샘플 주기 ≈ 32/30 = 약 1.07s**. 로드 시 가드로 `n_action_steps <= horizon - n_obs_steps + 1`(32 ≤ 64-2+1 = 63 ✓)을 assert한다.
- **30Hz 발행은 협상 불가.** 데이터셋이 30Hz로 기록됐고 브리지 필터도 ~30Hz 리더를 가정한다. 이보다 빠르거나 느리게 발행하면 학습과 다른 유효 다이나믹스가 된다.
- **이미지 전처리 — byte-parity, 체크포인트에 없음.** 저장된 preprocessor는 rename→batch→device→normalize뿐이며 **리사이즈도 색공간 변환도 하지 않는다.** 360×640 리사이즈와 BGR→RGB는 학습 시 dataloader 단(`image_transforms`)에서 적용됐다. 그래서 서버 측 `image_preprocess.py`가 반드시: `cv2.imdecode` → **BGR→RGB** → **torchvision `Resize([360,640])`**(bilinear+antialias, `eval_offline.py`의 `build_image_transforms()`와 동일한 단일 결정적 변환)를 수행한 **뒤에** 저장된 preprocessor가 실행된다. 720×1280을 그대로 먹이거나 BGR을 남기면 **에러 없이 조용히** 정책이 열화된다.
- **카덴스 노브 (env vars).**

  | 노브 | env var / CLI | 기본 | 효과 |
  |---|---|---|---|
  | DDIM step 수 | `DIFFUSION_NUM_INFERENCE_STEPS` / `--num-inference-steps` | `10` | ↓ 낮추면 refill이 빨라지지만 샘플 품질↓. **p99 refill이 예산 초과 시 여기부터 줄일 것(예: 5)** |
  | receding horizon | `DIFFUSION_N_ACTION_STEPS` / `--n-action-steps` | `32` | ↓ 낮추면 재쿼리가 잦아져 반응성↑(단 refill 빈도↑ → 평균 부하↑); ↑ 높이면 부하↓·반응성↓ |
  | 스케줄러 | `DIFFUSION_SCHEDULER` / `--scheduler` | `DDIM` | `DDIM`/`DDPM`/`asis`(config 그대로). 정상 배포는 `DDIM` 유지 |
  | 디바이스 | `DIFFUSION_DEVICE` / `--device` | `cuda` | CPU 폴백은 명시해야 하며 타임아웃 조정 필수(§9) |
  | 포트 | `DIFFUSION_PORT` / `--port` | `5592` | ZMQ 엔드포인트 (ACT는 5591) |

  > **반응성이 느려 그랩을 놓치면**: 먼저 `DIFFUSION_N_ACTION_STEPS`를 낮춰(예: 16) 재쿼리를 잦게 한다. 단 refill 빈도가 늘어 평균 GPU 부하가 오르므로 벤치마크로 예산 여유를 재확인할 것. refill이 무거운 게 병목이면 `DIFFUSION_NUM_INFERENCE_STEPS`를 줄이는 쪽이 더 직접적이다.

## 8. 검증 완료 vs 아직 안 됨

> **정직한 현재 상태.** 이 diffusion 경로는 **개발 PC에서 오프라인으로만** 검증되었다. **실물 UR7e 팔로는 아직 한 번도 구동되지 않았다.** (형제 ACT 경로는 2026-07-08에 실기 엔드투엔드까지 성공했으나, 그것은 ACT 서버 경로에 대한 검증이지 이 diffusion 서버 경로에 대한 검증이 아니다.)

**개발 PC에서 오프라인으로 검증됨** (로봇/카메라/GPU 없이, `--device cpu`):

- **엔드투엔드 ZMQ 라운드트립** — 로컬 joint 체크포인트로 `diffusion_server`를 CPU로 띄우고, ZMQ `reset` + 1회 `act` 라운드트립을 수행해 **finite한 (7,) 액션이 반환**됨을 확인. 즉 DDIM 오버라이드 로드 경로 → 관측 조립 → 이미지 전처리(360×640, BGR→RGB) → 저장된 preprocessor → `select_action`(DDIM-10) → postprocessor → 7-float 응답의 전 파이프라인이 dev PC에서 동작함.
- 스크립트 `bash -n` 문법 검사 + 서버/벤치마크 `py_compile`.

**아직 검증/수행 안 됨** (실기 배관·성능 모두 미검증):

- **실제 UR7e 팔 구동** — 이 diffusion 경로로 실물 팔을 자율 구동한 적이 **한 번도 없다.** handshake·안전 게이팅·클램프는 ACT와 같은 코드라 ACT 실기에서 검증됐지만, **diffusion 서버가 붙은 전체 경로의 실기 동작은 미확인.**
- **로봇 PC에서의 colcon build** — `gello_policy` + `ur_gello_bringup`를 로봇 PC Humble에서 실제로 빌드한 적 없음(dev PC와 환경이 다름).
- **GPU 레이턴시 벤치마크** — DDIM-10 refill의 p99가 이 로봇 PC의 GPU(예: RTX 3060 Mobile)에서 0.5s 예산 안에 드는지 **아직 측정 안 됨.** §6의 벤치마크를 **첫 팔 구동 전에 반드시** 돌려 `act_timeout_s=0.6`이 충분한지, 아니면 step을 줄여야 하는지 확정해야 한다.
- **closed-loop 태스크 성공률** — open-loop MAE(§3)는 있으나 실기 grab/placement 성공률은 미측정.

> **요약:** 지금 확실한 것은 **"diffusion_server가 유효한 액션을 dev PC에서 엔드투엔드로 반환한다"**는 것뿐이다. **"실기 파이프라인이 돈다"거나 "정책이 태스크를 푼다"는 아직 아니다.**

### 실기 첫 구동 프리플라이트 체크리스트

1. **하드웨어 존재 확인** — UR7e + Robotiq 2F-85 + RealSense 2대가 연결된 로봇 PC인지, cam1/cam2 시리얼 매핑과 물리 배치가 학습 리그와 일치하는지(§2).
2. **로봇 PC에서 colcon build** — `gello_policy` + `ur_gello_bringup` (§4-1). dev PC에서의 검증은 이를 대체하지 못한다(위 "아직 검증/수행 안 됨" 참고).
3. **게이트 A PASS (필수, 앙상블 여부와 무관)** — `benchmark_diffusion_latency.py`가 종료 코드 0(`VERDICT: OK`/`MARGINAL`)을 출력해야 팔을 구동할 수 있다(§6). `TOO SLOW`면 `DIFFUSION_NUM_INFERENCE_STEPS`를 줄이고 재측정.
4. **E-STOP / 펜던트 대기, 감독자 배치** — 이 diffusion 경로는 실물 UR7e에서 **아직 한 번도** 구동된 적이 없다. 형제 ACT는 2026-07-08에 실기 검증됐지만, 그 검증은 ACT 서버 경로에 대한 것이지 이 diffusion 서버 경로에 대한 것이 아니다(위 요약, §8).
5. **(선택) 앙상블 녹화(`DIFFUSION_ENSEMBLE_K>0`)를 켤 경우에만 — 게이트 B PASS + K 설정** — `benchmark_ensemble_batch16.py`가 `VERDICT: PASS`(종료 코드 0)를 출력해야 한다. 게이트 A 통과와 **완전히 독립**이며, **현재 로봇 PC(RTX 3060 Laptop, 6GB)에서는 K=1조차 FAIL**한다(§6 "게이트 A vs 게이트 B"). 녹화 없이 팔만 구동한다면 이 단계는 건너뛴다. 실행 상세는 [`GELLO_DIFFUSION_ENSEMBLE_RECORDER.md`](./GELLO_DIFFUSION_ENSEMBLE_RECORDER.md) §5b.
6. **카메라/그리퍼 라이브 확인** → **감독하에 `~/start_execution`**(§5).

## 9. 트러블슈팅

- **포트 5592 충돌 / 서버가 안 뜸.** diffusion 서버는 5592, ACT 서버는 5591을 쓴다. 두 서버가 동시에 떠 있어도 포트가 달라 충돌하지 않지만, 이전 diffusion 서버가 안 죽고 남아 있으면 5592가 이미 점유되어 새 서버가 bind에 실패한다 — `ss -ltnp | grep 5592`로 확인하고 잔여 프로세스를 정리하라. 리더는 `diffusion_deploy.yaml`의 `act_port=5592`로 서버를 찾으므로, 포트를 바꾸면 yaml과 서버 양쪽을 맞춰야 한다.
- **refill이 느려 FAULT-loop / 팔이 자꾸 멈춤.** DDIM refill이 `act_timeout_s`(0.6s)를 넘기고 있다는 뜻(또는 refill 블록이 obs를 얼려 `obs_timeout_s`(0.7s)를 넘긴 경우 — 둘 중 먼저 걸리는 쪽이 FAULT). **선호되는 해결책은 워치독을 넓히는 게 아니라**(§6) `DIFFUSION_NUM_INFERENCE_STEPS`를 낮추는 것이다(예: 5). 먼저 `benchmark_diffusion_latency.py`로 p99를 측정해 실제 예산 초과인지 확인할 것. GPU가 아니라 CPU로 돌고 있으면 근본 원인이 그것이다(아래).
- **서버가 종료 코드 3 / "CUDA unavailable"로 죽음.** `diffusion_server`의 `resolve_device()`는 `--device cuda`인데 CUDA가 없으면 **자동 CPU 폴백을 하지 않고 코드 3으로 거부**한다(CPU DDIM-10이 0.6s 타임아웃을 넘겨 FAULT-loop를 유발하므로 의도적). 해결: (a) CUDA/드라이버를 고치거나(RT 커널 vs NVIDIA 드라이버 충돌 등은 ACT 문서 §4.5-1 참고), (b) 정말 CPU로 돌려야 한다면 `DIFFUSION_DEVICE=cpu`를 명시하고 **동시에** `diffusion_deploy.yaml`의 세 워치독을 **모두** 크게 올릴 것 — `act_timeout_s`뿐 아니라 **`obs_timeout_s`도 반드시 함께** 올려 `act_timeout_s < obs_timeout_s < staleness_timeout_s` 순서를 유지해야 한다. `obs_timeout_s`를 그대로 두면 느린 CPU refill이 obs를 오래 얼려 **가짜 obs-stale FAULT**를 낸다. (단 이는 어디까지나 CPU 임시방편일 뿐이며, 정상 배포의 올바른 노브는 워치독 확대가 아니라 `DIFFUSION_NUM_INFERENCE_STEPS` 축소다.)
- **cam1/cam2 시리얼이 뒤바뀜.** 반드시 **시리얼로 바인딩**할 것(`serial_no`). 잘못 바인딩되면 정책은 크래시하지 않고 **조용히 열화**된다. all-digit 시리얼은 따옴표로 감쌀 것: `serial_no:="'147122072740'"` (정수 강제변환 버그 — ACT 문서 §9 / `GELLO_UR7E_RECORDING.md` troubleshooting 7 참고).
- **카메라가 조용히 안 뜬다 (노드는 살아 있는데 프레임이 0).** 십중팔구 **해석되지 않는 시리얼**을 바인딩한 것이다. `journalctl` / `/sys/bus/usb/devices/*/serial`이 보여주는 값은 **ASIC 시리얼**(`151623020789` / `322743060038`)이고, `serial_no:=`는 **장치 시리얼**(`147122072740` / `243222072700`)에만 매칭된다. 이 실패는 에러를 내지 않으며 `ros2 topic info`도 publisher 1을 보고한다 — `ros2 topic hz`로만 구분된다. **저널 grep으로 카메라 정체를 재유도하지 말 것**(그 방법으로 반대 방향의 틀린 결론에 도달한 세션이 둘 있다). 실제 값은 `rs-enumerate-devices`로 보거나, 그냥 [`launch_cameras.sh`](../../ros2_ur_ws/launch_cameras.sh)에 맡길 것.
- **`diffusers`가 없다 / import 에러.** ACT lock에는 `diffusers`가 빠져 있다. diffusion 서버는 `diffusers==0.35.2`가 필요하므로 반드시 `requirements-diffusion.lock`으로 설치했는지 확인(§4-2). ACT venv를 재사용했다면 `act_venv/bin/pip install diffusers==0.35.2`.
- **첫 실행 시 torch 캐시 관련 멈춤/실패.** §4의 `TORCH_HOME`/`HF_HUB_OFFLINE=1` 참고 — ResNet18 백본 가중치를 오프라인 환경에서 받으려다 멈추는 경우가 흔하다.
- **가짜(spurious) FAULT가 시작 직후에 뜬다.** 서버 워밍업이 실패/스킵됐을 가능성(로그의 `warmup done` 확인), 또는 첫 refill이 예산을 넘김(→ step 줄이기). diffusion은 첫 EXECUTE에서 곧바로 refill을 한 번 하므로 워밍업이 특히 중요하다.
- **`~/start_execution`이 계속 거부됨.** 라이브 자세가 `start_pose`에서 0.1rad 넘게 떨어졌거나, fresh 관측 셋(두 카메라 + 그리퍼 위치)이 완전하지 않다는 뜻. handshake 수렴(`Converged` 로그)과 카메라·그리퍼 토픽 발행을 확인하라(ACT 문서 §9와 동일).
- **그리퍼 크러시.** `policy_leader_node`는 `action[6]`을 그대로(identity) 발행하며 diffusion 경로도 `gello_gripper_bridge`(invert 옵션 보유)를 launch하지 않는다 — 트러블슈팅 시 그 노드가 여전히 빠져 있는지부터 확인(ACT 문서 §9와 동일).

## 10. ACT 배포와의 관계 / HIL-SERL으로 가는 길

**이 문서는 [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md)의 diffusion 형제(sibling)다.** 두 배포는 **같은 ROS2 패키지·같은 `policy_leader_node`·같은 브리지·같은 핸드셰이크·같은 안전 클램프·같은 그리퍼 스택**을 쓴다. 유일한 차이는:

| 항목 | ACT | Diffusion |
|---|---|---|
| 추론 서버 | `policy_server/act_server.py` | `policy_server/diffusion_server.py` |
| 정책 | `ACTPolicy` (트랜스포머 forward 1회) | `DiffusionPolicy` (DDIM-10 역확산) |
| 로드 시 핵심 차이 | — | `PreTrainedConfig`로 스케줄러를 **DDIM**·`num_inference_steps=10`으로 오버라이드 |
| ZMQ 포트 | **5591** | **5592** |
| receding horizon | `n_action_steps=30` | `n_action_steps=32` (재샘플 ≈1.07s @30Hz) |
| 타임아웃 | `act_timeout_s=0.5`, staleness 0.5 | **`act_timeout_s=0.6` < `obs_timeout_s=0.7` < `staleness_timeout_s=0.8`** (refill이 무겁고, 단일 스레드라 refill 블록 동안 obs가 얼기 때문) |
| 의존성 | `requirements-act.lock` | `requirements-diffusion.lock` (+`diffusers==0.35.2`) |
| 체크포인트 | `Bigenlight/act_banana_in_pot` | `Bigenlight/diffusion_banana_in_pot_joint` (best 80k) |
| 실기 상태 | 2026-07-08 엔드투엔드 실기 검증 | **오프라인(dev PC/CPU)만** — 실기 미검증 (§8) |

액션 컨트랙트(7-D JOINT)가 동일하므로, 안전상 중요한 ROS 코드는 **한 줄도 새로 쓰이지 않았다** — diffusion 배포는 순전히 서버 교체 + yaml 재타게팅(포트/타임아웃)이다.

**HIL-SERL으로 가는 길**은 ACT와 공유된다: 무거운 RL 부분(actor/learner, replay buffer, EE-delta processors)은 lerobot의 py3.12 쪽에 얹히고, ROS 쪽에는 `policy_mux_node` 하나만 추가되어 실제 GELLO 리더(사람 개입)와 정책 스트림을 중재한다. 자세한 근거와 리스크는 `DEPLOY_REPO_DECISION.md` §5를 참고할 것.

---

## 관련 문서

- [`GELLO_UR7E_ACT_DEPLOY.md`](./GELLO_UR7E_ACT_DEPLOY.md) — **형제 ACT 배포(실기 검증됨)**. 이 문서의 뼈대이며, Method A/B·4-터미널 워크플로우·recorder·리커버리 등 공유 절차의 상세 버전
- [`gello_policy/README.md`](../../ros2_ur_ws/src/gello_policy/README.md) — 패키지 개요, 노드/토픽 레퍼런스, 빌드 (§ "Diffusion variant")
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 사람 GELLO 리더로 하는 move-to-start handshake의 전체 메커니즘(chase/dwell/stillness-gate) — 이 문서의 handshake는 리더가 정책으로 바뀌었을 뿐 로직은 동일
- [`GELLO_UR7E_RECORDING.md`](./GELLO_UR7E_RECORDING.md) — 이 정책을 학습시킨 데이터셋을 만든 레코더 (observation/action 컨트랙트의 출처)
- 모델 카드: HF `Bigenlight/diffusion_banana_in_pot_joint` (아키텍처·학습·open-loop 결과 상세), 프로젝트 루트 `DIFFUSION_JOINT_OVERFIT.md` (open-loop MAE로 체크포인트를 고른 근거)
