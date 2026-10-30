# ⛔ 폐기된 문서 — 2026-07-24 시점의 기록 (실행하지 마라)

> **이 문서는 superseded 되었다. 현재 진입점은
> [`HANDOFF_NEXT_SESSION_KO.md`](./HANDOFF_NEXT_SESSION_KO.md) 다.**
> 이 파일은 **2026-07-24에 무엇을 알고 있었는지**의 기록으로만 남긴다. 사료로서의 가치는 있지만,
> **여기 적힌 설치 명령·포트·CLI를 그대로 따라 하면 실패한다.** 아래 대조표를 먼저 읽어라.
>
> 지금 유효한 문서: [`HANDOFF_NEXT_SESSION_KO.md`](./HANDOFF_NEXT_SESSION_KO.md)(진입점) ·
> [`HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md`](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)(learner) ·
> [`REMOTE_ACTOR_GRPC.md`](./REMOTE_ACTOR_GRPC.md)(전송) ·
> [`HIL_SERL_KANU_RUNBOOK_KO.md`](./HIL_SERL_KANU_RUNBOOK_KO.md)(서버 운용) ·
> [`REWARD_CLASSIFIER_THRESHOLD_KO.md`](./REWARD_CLASSIFIER_THRESHOLD_KO.md)(reward threshold) ·
> [`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)(actor 실기).

## ⛔ 이 문서에서 지금 틀린 것 (따라 하면 깨지는 순서대로)

| 이 문서의 서술 | 2026-07-29 현재 | 근거 |
| --- | --- | --- |
| **§3 "`jax==0.4.35`를 명시적으로 핀"** (`jax[cuda12_pip]==0.4.35` / `jax[cpu]==0.4.35`) | **틀렸다. 런타임이 `jax`/`jaxlib` `0.5.3`을 요구하며 fail-closed로 죽는다.** 0.4.35를 깔면 learner가 시작조차 못 한다 | `ur_env/learner/agent.py:104-150`(`validate_learner_dependencies`, 기대값 jax/jaxlib 0.5.3, flax 0.10.5, distrax 0.1.5, tensorflow_probability 0.25.0, wandb 0.26.0) ← `scripts/run_rlpd_learner_server.py:488` 이 호출 |
| **§2·§4 통신 = agentlace, 포트 5588(데이터)/5589(파라미터)** | **틀렸다. 전송은 gRPC 단일 포트 `50053`**(SSH 터널 로컬 `50153`)으로 교체됐다. `network.type='agentlace'`는 **명시적으로 거부**된다 | `ur_env/actor_network.py:1205-1215`(`agentlace` → `NotImplementedError`), `scripts/run_rlpd_learner_server.py:85`, `ur_experiments/cube_in_cup.py:273` |
| **§5 CLI `train_rlpd.py --learner` / `--actor --ip=...`** | 그런 진입점을 쓰지 않는다. 현재는 `scripts/run_rlpd_learner_server.py`(learner) / `scripts/run_remote_rlpd_actor.py`(actor)다 | 위 두 파일 |
| **§1·§4 🔴 `clip_safety_box` 미작동** | **구현됐고 명령 경로에 배선됐다**(`PolicyDeltaController(clip_pose=...)` → 명령 자세에만 적용, 관측에는 미적용). 다만 `ABS_POSE_LIMIT_*`를 채우지 않는 config에서는 **여전히 no-op**이다(기본값이 0 벡터) | `ur_env/envs/ur7e_env.py:334`(구현)·`:188`·`:357`(배선), `ur_env/envs/policy_delta_controller.py:140-141`, `tests/test_clip_safety_box.py`, `ur_env/envs/config.py:76-77`(0 기본값) |
| **§1·§4 🟠 wrapper 미포팅**(`RelativeFrame`·`Quat2Euler`·`Chunking`·`GripperPenalty`) | **포팅됐다** | `ur_env/envs/frame_wrappers.py`(RelativeFrame, Quat2EulerWrapper), `ur_env/envs/chunking.py`(obs_horizon=1만), `ur_env/envs/wrappers.py`(GripperPenaltyWrapper) |
| **§1 "태스크 config 없음"** | **생겼다** — `cube_in_cup`. **그리고 이 config가 `IMAGE_CROP`을 켜면서 reward classifier 전처리 불일치를 유입시켰다**(미해결) | `ur_experiments/cube_in_cup.py:211-214`, `ur_experiments/mappings.py`, [`REWARD_CLASSIFIER_THRESHOLD_KO.md`](./REWARD_CLASSIFIER_THRESHOLD_KO.md) |
| **§1 "리워드 분류기는 우리 범위 밖"** | 학습은 여전히 외부지만, **크롭 정합 재학습과 threshold는 이제 우리 문제다** | 같은 문서 |
| **§3.2 actor venv = `~/actor_venv`** | 실제 경로는 `/home/laptop3/venvs/gello-hil-actor` | [`HANDOFF_NEXT_SESSION_KO.md`](./HANDOFF_NEXT_SESSION_KO.md) |
| **부록 A 머신 팩트**(디스크 여유, `nvidia-smi` 드라이버 불일치 등) | 2026-07-24 정찰 스냅샷이다. 그 뒤 재확인하지 않았다 — **현재 상태는 미확인**이므로 직접 확인 없이 인용하지 마라 | — |

**반대로 아직 유효한 것**(즉 이 문서가 "미완"이라고 적은 것 중 지금도 미완인 것):
`GRASP_POSE` 미선언(트리 전체에 정의 없음) · `RANDOM_RESET`/`RANDOM_XY_RANGE` no-op
(`ur_env/envs/config.py:39-40`, `ur_env/envs/ur7e_env.py:536` TODO) · `PolicyDeltaController` 단순화판
(`ur_env/envs/policy_delta_controller.py` 상단 TODO 그대로) · `save_video` no-op
(`ur_env/envs/ur7e_env.py:776-778`) · 부트스트랩 데모 미수집.
이 항목들의 현재 상태도 이 문서가 아니라 위 진입점 문서에서 확인하라.

---

# HIL-SERL on UR7e+GELLO — 현재 상황 & 다음 할 일 (서버 핸드오프)

> ⛔ **아래는 2026-07-24 원문이다. 갱신하지 않았고, 그대로 실행하면 안 된다.** 위 대조표 참조.
>
> 작성 2026-07-24. 목적: **학습/추론은 서버(`kanu`)에서 돌린다**는 결정에 맞춰,
> 로봇 랩톱(`laptop3`)에서 준비한 내용과 남은 작업을 서버에서 그대로 이어받도록 정리.
> 배경 설계는 [`README.md`](README.md), env 상태 스냅샷은
> [`../docs/rl/GELLO_UR7E_SERL_ENV_STATUS.md`](../docs/rl/GELLO_UR7E_SERL_ENV_STATUS.md),
> upstream 구조는 [`../docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md`](../docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md).

---

## 0. 한눈에 (TL;DR)

- **학습(learner)·추론은 서버 `kanu`(166.104.35.33, GPU)에서.** 로컬 랩톱은 로봇을 구동하는 **actor**만 담당. 이 리포는 서버에도 동일하게 있으므로, 아래 설치 레시피/코드 gap을 채워 커밋하면 서버에서 그대로 진행 가능.
- **리워드 분류기**(성공 판별기)는 **다른 곳에서 학습 중** → 우리 범위 밖. 결과물 `classifier_ckpt/`만 받으면 됨.
- **베이스 폴리시 불필요.** HIL-SERL(RLPD)은 정책을 랜덤에서 시작하고 **데모 + 사람 개입**으로 학습. 사전학습 정책도, 기존 ACT/Diffusion/FM(PyTorch)도 SAC actor로 못 씀(프레임워크·알고리즘 상이).
- **지금까지 검증된 것:** `serl_ur_infra` env가 offline + RViz mock + (텔레옵 경로) 실기까지 동작. **아직 안 붙은 것:** RL 두뇌(`serl_launcher`/`agentlace`/learner) — 로컬 미설치(의도적).

---

## 1. 현재 상황

### 됨 ✅
- **hil-serl 서브모듈 init 완료** — `third_party/hil-serl` (rail-berkeley/hil-serl, 핀 `c32939b`, Apache-2.0). `serl_launcher/`, `serl_robot_infra/`, `examples/`, `docs/` 포함.
- **`serl_ur_infra` env 코어 5개 모듈** (`ur_env/envs/`): `ur7e_env.py`, `config.py`, `ros_backend.py`, `policy_delta_controller.py`, `wrappers.py`. FrankaEnv 관측/액션 계약을 복제 → hil-serl 학습 스택이 무수정으로 도는 것을 목표.
- **검증 사다리 3단** (`tests/`): `test_env_fake_backend.py`(offline, ROS 없이 통과) → `run_rviz_fake_rl.py`(mock+RViz) → `run_rviz_hil.py`(mock+진짜 GELLO 개입). 앞 두 단계 + 실기 EEF 텔레옵/개입은 이미 확인됨.
- **`GelloIntervention`이 `SpacemouseIntervention`의 drop-in 교체**로 설계됨 — `info["intervene_action"]`을 동일 계약으로 세팅하므로 actor 루프가 그대로 인식.

### 안 됨 / 없음 ❌
- 로컬에 `serl_launcher`/`jax`/`agentlace` 미설치 (서버에서 돌릴 것이므로 의도적).
- **태스크 config 없음** (hil-serl `examples/experiments/<task>/config.py` 대응물).
- **wrapper 미포팅**: `RelativeFrame`, `Quat2EulerWrapper`, `SERLObsWrapper`, `ChunkingWrapper`, `GripperPenaltyWrapper`, 분류기 wrapper — 아래 §4 gap 참고.
- **`clip_safety_box` 미작동** (안전) / PolicyDeltaController 단순화판 / fault recovery 없음.
- 부트스트랩 데모 미수집.

---

## 2. 아키텍처 — actor/learner 분리 + 서버 플랜

> ⛔ **[2026-07-29] 아래 다이어그램의 전송 계층은 폐기됐다.** agentlace 5588/5589가 아니라
> **gRPC 단일 포트 50053**(터널 로컬 50153)이다. actor/learner 분리라는 큰 그림만 유효하다.

```
   ┌─────────────── 로봇 랩톱 laptop3 (actor) ───────────────┐        ┌──── 서버 kanu (learner) ────┐
   │  UR7e + GELLO + RealSense                                │        │  GPU (RLPD/SAC 학습)        │
   │  ROS2 Humble (rclpy) + serl_ur_infra env + jax(actor)    │        │  순수 JAX, ROS 불필요       │
   │  agentlace TrainerClient  ──(5588 데이터 up)──────────►  │        │  agentlace TrainerServer    │
   │                           ◄─(5589 파라미터 broadcast)──  │        │  fake_env=True (관측/액션    │
   │  사람: GELLO로 개입(데드맨)                              │        │   공간만; 하드웨어 없음)     │
   └─────────────────────────────────────────────────────────┘        └─────────────────────────────┘
              actor는 --ip=<터널 localhost> 로 learner에 접속 (SSH 터널로 5588/5589 포워딩)
```

- **learner = 서버.** `train_rlpd.py --learner`. GPU에서 SAC 업데이트, replay/demo 버퍼 소유, 새 정책 파라미터를 actor로 push. `fake_env=True`라 로봇/ROS 불필요.
- **actor = 로봇 랩톱.** `train_rlpd.py --actor --ip=<learner>`. 실제 env·로봇 구동, 매 스텝 정책 실행 + 사람 개입 라우팅.
- **통신 = agentlace** (`youliangtan/agentlace@cf2c337`, pip git). 포트 **5588**(데이터 업로드) / **5589**(파라미터 broadcast). 두 포트를 SSH 터널로 포워딩하고 actor는 `--ip=127.0.0.1`.
  - 기존 remote-diffusion의 터널/프리플라이트/Docker 패턴 재사용 가능([[remote-diffusion-merged-into-ur7e]]). gRPC 코드는 제외.
- **프레임워크 공존:** learner는 순수 JAX. **actor는 한 프로세스에서 `rclpy`(ROS2) + `jax`를 동시 import** 해야 함 → §3.2의 `--system-site-packages` venv 전략 필수.

---

## 3. 설치 레시피 (정찰로 검증됨 — 그대로 쓰면 됨)

> ⛔ **[2026-07-29] 이 절의 명령을 실행하지 마라.** `jax==0.4.35` 핀은 이제 **틀렸다** —
> 런타임이 `0.5.3`을 요구하고 불일치 시 `LearnerDependencyError`로 죽는다
> (`ur_env/learner/agent.py:104-150`). "그대로 쓰면 됨"이라는 이 절의 제목은 07-24 기준이다.
> 현재 설치 절차는 [`HIL_SERL_KANU_RUNBOOK_KO.md`](./HIL_SERL_KANU_RUNBOOK_KO.md) 와
> [`HANDOFF_NEXT_SESSION_KO.md`](./HANDOFF_NEXT_SESSION_KO.md) 에 있다.

### ⚠️ 공통 landmine — jax 버전 반드시 핀
hil-serl README의 **CPU 설치 라인은 무핀**(`pip install --upgrade "jax[cpu]"`)인데, 이러면 **jax 0.6.2**가 깔리고 거기선 `jax.tree_map`이 **제거**되어 있어 `serl_launcher`(common/common.py, wrappers/chunking.py 등에서 `jax.tree_map` 직접 호출)가 **런타임 크래시**(`AttributeError`). `pip install`은 성공으로 보고하므로 더 함정.
→ **CPU든 GPU든 `jax==0.4.35`를 명시적으로 핀.** (서버 GPU는 `jax[cuda12_pip]==0.4.35`.)

기타 정찰 사실:
- `tensorflow`는 **필수**(`serl_launcher/common/typing.py`가 top-level `import tensorflow`) — ~GB 불가피.
- `agentlace`는 순수 파이썬(C 빌드 없음), github 공개, git+pip로 설치됨.
- `moviepy`는 requirements에 있으나 코드에서 **미사용**(dead weight). `orbax-checkpoint`는 jax 0.4.35 때문에 0.11.5로 backtrack(설치 느려도 정상).
- `serl_ur_infra`는 `--no-deps`로 설치(그 deps는 serl_launcher가 이미 충족). **설치 후 `pip freeze`로 lock 파일 남길 것**(느슨한 핀 드리프트 방지).

### 3.1 서버(learner) env — clean venv
```bash
python3.10 -m venv ~/serl_venv          # 서버 python3.10 기준
source ~/serl_venv/bin/activate
pip install --upgrade pip
pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd <repo>/third_party/hil-serl/serl_launcher
pip install -e .
pip install -r requirements.txt
cd <repo>/serl_ur_infra
pip install -e . --no-deps
pip freeze > <repo>/serl_ur_infra/serl_lock_server.txt
python -c "import jax, serl_launcher, agentlace, ur_env; print('learner stack OK', jax.__version__)"
```
> learner는 `fake_env=True`라 rclpy 불필요 → `--system-site-packages` 안 써도 됨.

### 3.2 로봇 랩톱(actor) env — `--system-site-packages` venv (rclpy + jax 공존)
정찰 결과: `rclpy`는 `/opt/ros/humble/local/lib/python3.10/dist-packages`에 있고, 그 apt 의존성(`yaml`, `netifaces`, `empy`, `lark` 등)은 PYTHONPATH로 안 잡혀서 **`--system-site-packages`** 라야 보임. venv는 `~/.local`을 안 보므로 `pynput`/`serl_ur_infra`를 venv pip로 재설치해야 함.
```bash
# 이 venv를 쓰는 모든 셸에서 매번 먼저:
source /opt/ros/humble/setup.bash
source ~/gello_software/ros2_ur_ws/install/setup.bash    # ur_gello_bringup 오버레이(ur_kin)

python3 -m venv --system-site-packages ~/actor_venv       # 시스템 python3.10(=ROS 3.10.12)
source ~/actor_venv/bin/activate
pip install --upgrade pip
pip install "jax[cpu]==0.4.35"
cd ~/gello_software/third_party/hil-serl/serl_launcher && pip install -e . && pip install -r requirements.txt
pip install pynput
pip install -e ~/gello_software/serl_ur_infra --no-deps
python -c "import cv2; print('cv2 OK', cv2.__version__)"   # numpy 그림자 후 ABI 확인
python -c "import rclpy, jax, ur_env; print('actor stack OK')"  # 반드시 ROS source 후
```
> **gotcha:** ① 새 셸마다 두 `setup.bash` 소스 필수(자동 안 됨). ② venv라 `~/.local` 패키지 안 보임 → 위처럼 재설치. ③ 시스템 numpy 1.21.5를 venv numpy가 가리면 compiled 확장(cv2 등) ABI 깨질 수 있음 → 깨지면 venv에 `opencv-python-headless` 설치.

---

## 4. 실기 학습 전 메워야 할 코드 gap (Franka 원본 대비)

> ⛔ **[2026-07-29] 이 표는 07-24 기준이다.** `clip_safety_box`와 wrapper 4종은 그 뒤 구현됐고,
> `PolicyDeltaController`·`save_video`·`GRASP_POSE`·`RANDOM_*`는 아직 열려 있다. 상단 대조표 참조.

| 심각도 | gap | 위치 / 내용 | 참고 |
| --- | --- | --- | --- |
| 🔴 안전 | `clip_safety_box` 미작동 | `ur7e_env.py:229-240` `_apply_action`에 `TODO`. `ABS_POSE_LIMIT_LOW/HIGH`가 config엔 있으나 명령 자세를 자르지 않음. RL은 사람 없이 탐색하므로 **실기 전 필수**. | Franka는 매 스텝 `clip_safety_box` 호출 |
| 🟠 차단 | wrapper 미포팅 | `RelativeFrame`(가장 중요, EEF프레임↔base 변환), `Quat2EulerWrapper`, `SERLObsWrapper`, `ChunkingWrapper`, `GripperPenaltyWrapper`, 분류기 wrapper가 `serl_ur_infra`에 없음. `franka_env.envs.*`에서 직접 import하면 `pyspacemouse` + 구식 `gym`(gymnasium 아님) 의존성이 딸려와 깨질 수 있음 → **UR 로컬 re-export 또는 클린 import 경로** 필요. | `franka_env/envs/wrappers.py`, `relative_env.py` |
| 🟡 config | `GRASP_POSE` 미선언 / `RANDOM_RESET·RANDOM_XY_RANGE·RANDOM_RZ_RANGE` 미사용 | Franka 태스크 config를 그대로 베끼면 `GRASP_POSE`는 `AttributeError`, `RANDOM_*`는 조용히 no-op. `config.py`/`go_to_reset` 배선 필요. | `config.py:39-41`, `ur7e_env.py:296` |
| 🟡 컨트롤러 | `PolicyDeltaController` 단순화판 | branch-lock 해석 IK·sigma_min 특이점 감속·keepout 없음. `README.md` 비교표대로 `eef_delta` 후반부 재사용으로 교체 → **DRY_RUN 해제 전 필수**. | `policy_delta_controller.py` 상단 TODO |
| 🟡 운영 | UR fault recovery 없음, `save_video` no-op | protective-stop 복구(`/clearerr` 대응) 미구현. `_save_video_recording`은 프레임만 비움. | `ur7e_env.py:17, 488` |

**task config에 채워야 할 값**(hil-serl `TrainConfig`/`EnvConfig` 대응): `RESET_JOINTS`·`RESET_POSE`(같은 물리자세), `ACTION_SCALE`, `ABS_POSE_LIMIT_LOW/HIGH`(바운딩박스 실측), `CAMERAS`+`IMAGE_CROP`, `image_keys`/`classifier_keys`/`proprio_keys`, `setup_mode`(`single-arm-learned-gripper`면 이산 grasp critic 사용). 그리퍼를 이산 3-state로 학습하려면 hybrid agent(`SACAgentHybridSingleArm`) 경로 → 액션 `[6 EEF, 1 gripper]`, transition에 `grasp_penalty` 필요.

---

## 5. HIL-SERL 실행 파이프라인 (순서)

> ⛔ **[2026-07-29] 아래 CLI는 존재하지 않는 진입점이다.** `train_rlpd.py --learner/--actor` 대신
> `scripts/run_rlpd_learner_server.py` / `scripts/run_remote_rlpd_actor.py` 를 쓴다.
> 실행 순서(라벨링 → classifier → 데모 → 학습)라는 큰 틀만 유효하다.

```
① record_success_fail.py   → 성공/실패 라벨링            [외부에서 진행 중]
② train_reward_classifier  → classifier_ckpt/            [외부에서 진행 중]
③ record_demos.py          → demo_data/*.pkl (~20, 실기)  [우리, actor에서]
④ train_rlpd.py --learner  (서버 kanu)  ┐ agentlace로 연결, 동시 실행
   train_rlpd.py --actor   (로봇 랩톱)  ┘ actor는 --ip=<터널>
```
- ③ `record_demos.py`: 정책 zero + 사람이 GELLO로 전 구간 시연, `info["succeed"]`인 성공 궤적만 저장. 실기 필요(actor env). wrapper 체인 + task config가 있어야 돎.
- ④ learner는 `--demo_path`로 데모 pkl 로드, 50/50(online+demo) RLPD 배치. actor는 개입 스텝을 online+demo 버퍼 양쪽에 라우팅.
- 예시 CLI(hil-serl `usb_pickup_insertion/run_*.sh` 기준):
  ```bash
  # 서버(learner)
  python train_rlpd.py --exp_name <task> --checkpoint_path <ckpt> --demo_path <demos.pkl> --learner
  # 로봇(actor)
  python train_rlpd.py --exp_name <task> --checkpoint_path <ckpt> --actor --ip=127.0.0.1   # 터널
  ```

---

## 6. 다음 할 일 (우선순위)

> ⛔ **[2026-07-29] 이 목록은 소진됐거나 폐기됐다.** 1·2번(task config, wrapper 체인)과 3번
> (`clip_safety_box`)은 완료됐고, 4번의 "agentlace 스모크 테스트"는 gRPC로 대체됐다.
> 현재 우선순위는 [`HANDOFF_NEXT_SESSION_KO.md`](./HANDOFF_NEXT_SESSION_KO.md) 를 보라.

1. **task config 1개 작성** — 최소 난이도 태스크(cube_in_cup 이하)로. §4의 값들 실측/기입. `examples/experiments/<task>/{config.py,wrapper.py}` 대응. `mappings.py`(우리 쪽 등가물)에 등록.
2. **wrapper 체인 정리** — `RelativeFrame`/`Quat2Euler`/`SERLObs`/`Chunking`/`GripperPenalty`를 UR 로컬에서 안전 import 가능하게(re-export 또는 최소 포팅). `pyspacemouse`/구식 `gym` 딸림 방지.
3. **`clip_safety_box`를 컨트롤러 게이트에 통합** (🔴 안전) + `ABS_POSE_LIMIT` 배선.
4. **서버에 learner env 구축**(§3.1) + **agentlace 스모크 테스트**(서버 TrainerServer ↔ 로컬 TrainerClient, SSH 터널 5588/5589) — 실 학습 전에 통신만 먼저 확인.
5. **로봇에 actor env 구축**(§3.2, `--system-site-packages`) + `import rclpy, jax, ur_env` 확인.
6. **mock 端-to-端 배관 검증** — actor(mock 하드웨어) + learner(fake_env) 로 `train_rlpd` 전체 루프가 도는지(실기 리스크 0). 여기서 wrapper/계약 문제 다 드러남.
7. **`PolicyDeltaController` → eef_delta 후반부 교체** (DRY_RUN 해제 전).
8. **데모 ~20개 수집**(③) → **실기 `train_rlpd`**(④). 안전 리밋/개입 인력 일정 반영.

---

## 부록 A. 머신 팩트 (2026-07-24 정찰)

- **laptop3 (actor):** RTX 3060 Mobile 물리적으로 있음. 현재 `nvidia-smi`가 드라이버 버전 불일치(로드된 커널모듈 595.71.05 < 설치 라이브러리 595.84)로 죽어 있음 — **재부팅하면 복구될** 전형적 케이스. python 3.10.12, gcc 11.4, python3.10-venv/dev 있음. **디스크 여유 ~36G/468G(93% 참) — 빠듯**, 무거운 설치 전 정리 권장.
- **서버 `kanu`:** `~/.ssh/config`에 `kanu → 166.104.35.33, user junhyeong` 설정됨. GPU learner 대상.
- 기존 venv: `.venv`(3.11), `ros2_ur_ws/act_venv`(3.12, 검증된 배포용 — **건드리지 말 것**). HIL-SERL용은 위처럼 **별도 3.10 venv** 신설.

## 부록 B. 근거 (조사 방법)

- upstream 구조·파이프라인·계약: hil-serl 4영역 병렬 조사 → 메모리 [[hil-serl-upstream-survey]].
- 설치 핀/의존성: PyPI 메타데이터 + `pip install --dry-run` 실검증(별도 스크래치 venv).
- ROS+jax 공존: rclpy 경로·venv semantics·`ros_backend.py` import 경계 실조사.
- 로컬 시험 설치(jax 0.4.35 CPU)는 성공 확인 후 **삭제**(로컬 미실행 결정).
