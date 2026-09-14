# carrot_in_pot EEF-delta 학습 번들 (A3)

`Put carrot in pot` 54 take 데이터셋을 **Cartesian(EEF) delta 액션 공간**
(`state 16 / action 7`, `ACTION_SPACE_ID = "eef_delta_v1"`)으로 ACT · Flow-Matching
두 정책에 학습시키는 스크립트 묶음이다. 설계 계약은
`ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md`가 정본이고, 이 문서는 **조작 절차**만 적는다.

참조 레시피(그대로 따라 만든 것):
`~/youngwoong_ws/banana-in-pot-experiments/train_act.sh`(+`train_act_valdiag.sh`, `eval_offline.py`),
`~/youngwoong_ws/cube_flow_matching_training/`(`train_flow_matching.sh`, `preflight.sh`,
`launch_detached.sh`, `sync_to_kanu.sh`, `eval_checkpoints.sh`, `eval_flow_matching.py`).

---

## 0. 한눈에

| 파일 | 하는 일 |
| --- | --- |
| `_common.sh` | 경로·기본값·NVIDIA lib 트릭·held-out episode 계산 (나머지가 source 한다) |
| `train_act_carrot_eef.sh` | ACT 50,000 step, batch 8, 360×640 resize, seed 1000 |
| `train_fm_carrot_eef.sh` | FM(`multi_task_dit`/`flow_matching`) 100,000 step, batch 8, 내부 224 |
| `preflight.sh` | 5개 게이트(아래 §2). **통과 못 하면 절대 띄우지 않는다** |
| `launch_detached.sh JOB` | `setsid nohup` 분리 실행 + `runtime/<job>/train.{pid,log}` |
| `sync_to_kanu.sh` | 번들 + 데이터셋 → `kanu:~/workspace/youngwoong/carrot_eef/` |
| `eval_offline_eef.py` | held-out episode open-loop 평가 (ACT/FM 공통) |
| `eval_checkpoints.sh JOB` | checkpoint 전체 순회 + `evaluations/<job>/summary.csv` |

**GPU 규칙 (DESIGN §1): kanu는 공용이다. 우리는 최대 2개만 쓴다.**
GPU는 **띄우기 직전에** `nvidia-smi`로 유휴인 것을 고른다. `GPU_ID`에 기본값은 없다 —
빠뜨리면 스크립트가 거부한다.

**학습 시작은 integrator가 한다.** A3는 sync · preflight · `DRY_RUN=1`까지만 한다.

---

## 1. laptop3 → kanu 전송

```bash
cd /home/laptop3/gello_software_humble/scripts/training/carrot_eef
./sync_to_kanu.sh --dry-run     # 무엇이 갈지 전부 출력, 전송 0바이트
./sync_to_kanu.sh               # 실제 전송

# 이미 보낸 데이터셋을 **폐기하고 다른 변환본으로 교체**할 때만:
./sync_to_kanu.sh --delete-dataset --dry-run
./sync_to_kanu.sh --delete-dataset
```

⚠️ `--delete-dataset`는 **데이터셋 디렉터리에만** `rsync --delete`를 건다. 새 변환본이
파일 수·프레임 수가 **더 적을 때** 이게 없으면 옛 파일이 kanu에 남아 두 변환본이 섞인다.
**번들 rsync에는 절대 `--delete`를 걸지 않는다** — `runtime/`·`outputs/`·`evaluations/`는
kanu에만 있고, 지우면 돌고 있는 job의 로그와 checkpoint가 날아간다.

보내는 것은 딱 둘이다.

| 로컬 | kanu |
| --- | --- |
| 이 번들 디렉터리 | `~/workspace/youngwoong/carrot_eef/training/` |
| `/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot/` | `~/workspace/youngwoong/carrot_eef/dataset/carrot_in_pot_eef_lerobot/` |

- 로컬 데이터셋에 `meta/info.json`이 없으면 **거부한다**(변환이 안 끝났다는 뜻).
- `runtime/ outputs/ evaluations/ .cache/`는 kanu에서 생기는 것이라 전송에서 제외한다.
- `REMOTE_ROOT`가 `workspace/youngwoong/carrot_eef`로 시작하지 않으면 거부한다 —
  **kanu에서 그 밖으로는 아무것도 쓰지 않는다.**

---

## 2. preflight (kanu에서)

```bash
cd ~/workspace/youngwoong/carrot_eef/training
nvidia-smi                       # ← 지금 유휴인 GPU 2개를 고른다
GPU_ID=<a> ./preflight.sh act
GPU_ID=<b> ./preflight.sh fm
```

거부 조건 5개:

1. **선택한 GPU에 CUDA compute 프로세스가 하나라도 있으면** 거부.
   메모리 임계값이 아니라 **프로세스**로 본다(banana `TROUBLESHOOTING.md` §6).
2. **우리 `lerobot-train` 프로세스가 이미 2개면** 거부 (`MAX_OUR_GPUS`, 기본 2).
   한 run이 GPU 하나를 쓰므로 프로세스 수 = GPU 수다. 남의 job은 1번이 본다.
3. `~/workspace/youngwoong/carrot_eef`가 있는 파일시스템 **여유 < 20 GiB**면 거부
   (`MIN_FREE_GIB`). kanu는 98 % 차 있다.
4. **데이터셋 feature 집합이 EEF 계약과 정확히 같지 않으면** 거부 —
   `{action[7], observation.state[16], observation.images.cam1, observation.images.cam2,
   timestamp, frame_index, episode_index, index, task_index}` 9개 **정확히**, 이름은
   DESIGN §2, `total_episodes == 54`, `fps == 30`, `codebase_version == v3.0`.
   🛑 **depth feature가 있으면 여기서 걸린다.** 학습 스크립트에 `--policy.input_features`
   override가 **없기 때문에** lerobot이 자동 탐지한 feature를 그대로 정책 입력으로 쓴다 —
   depth가 섞여 있으면 조용히 학습되어 버린다. 그래서 "없음"을 **명시적으로 단언**한다.
5. venv에서 `lerobot-train --help`가 실패하면 거부.
6. **`meta/source_takes.json`의 take가 전부 `n_tail_dropped == 0`이면 거부.**
   첫 carrot EEF 변환본은 take 끝마다 **frozen tail**(레코더 joint 캐시가 갱신을 멈춘 구간)을
   그대로 남겼고, 유한차분 액션이 그것을 **0 delta의 연속**으로 만든다 — 정책이 "마지막에는
   멈춘다"를 학습한다. 채택된 변환본은 `--drop-stale-tail`로 그것을 잘라내고 take별로 몇
   프레임을 버렸는지 기록한다. `total_frames`만으로는 두 변환본을 확실히 구분할 수 없어서
   **버림 자체를 명시적으로 단언한다.** 탈출구 `ALLOW_STALE_TAIL=1`.
   (`source_takes.json`이 없거나 `n_tail_dropped` 필드가 없으면 실패시키지 않고 그 사실을
   한 줄로 알린다.)

> 📌 **preflight는 프레임 수를 하드코딩하지 않는다.** `total_episodes == 54`(`TOTAL_EPISODES`로
> 덮어쓰기 가능)만 단언하고 `total_frames`는 `meta/info.json`에서 읽어 **보고만** 한다.
> 변환 사양이 바뀌어 프레임 수가 달라져도 게이트가 잘못 막지 않는다.

> 이름의 정본은 `gello_policy/eef_space.py`의 `STATE_NAMES` / `ACTION_NAMES`다
> (`ur_q1..ur_q6, tcp_x/y/z, tcp_r11..tcp_r32, grip_pos` / `dx..drz, grip_cmd`).
> DESIGN §2는 같은 16개를 축약해서(`q1..q6`, `r11..r32`) 적어 놓았을 뿐이라,
> preflight는 `{q, ur_q} × {r, tcp_r}` **4가지 표기를 모두 허용하고 실제 이름을 출력한다.**
> 그 밖의 이름은 **다른 물리량**이라는 뜻이므로 실패한다. integrator에게 알린 뒤에만
> `EXPECTED_STATE_NAMES=` / `EXPECTED_ACTION_NAMES=`(쉼표 구분)로 우회한다.

---

## 3. 명령 미리보기 (`DRY_RUN=1`)

아무것도 실행하지 않고 실제 `lerobot-train` 인자만 출력한다.

```bash
DRY_RUN=1 GPU_ID=<a> ./train_act_carrot_eef.sh
DRY_RUN=1 GPU_ID=<b> ./train_fm_carrot_eef.sh
```

두 recipe의 고정값:

| | ACT | FM |
| --- | --- | --- |
| policy | `--policy.type=act` | `--policy.type=multi_task_dit --policy.objective=flow_matching` |
| 이미지 | dataset 단계 `Resize [360,640]` (`max_num_transforms=1`) | 정책 **내부** `image_resize_shape=[224,224]`, `image_crop_shape=[224,224]` |
| steps | 50,000 | 100,000 |
| batch / seed | 8 / 1000 | 8 / 1000 |
| save_freq / eval_steps / log_freq | 10,000 / 2,000 / 200 | 10,000 / 2,000 / 200 |
| 기타 | `HF_HUB_OFFLINE=1` (ResNet18은 `TORCH_HOME` 캐시에서) | `num_integration_steps=100`, offline 강제 **안 함**(CLIP ViT-B/16을 HF 캐시/네트워크에서 받는다 — cube recipe와 동일) |
| 공통 | `--dataset.eval_split=0.111`, `--wandb.enable=false`, `--policy.push_to_hub=false`, `num_workers=4` | |

`--dataset.eval_split=0.111` → `ceil(54 × 0.111) = 6` → **마지막 6개 episode 48–53이
held-out**이다. `eval_offline_eef.py`의 기본 episode도 정확히 이 6개다.
**eval_split을 바꾸면 평가 episode도 같이 바꿔야 한다** — 안 그러면 학습에 쓴
episode를 평가하게 된다(data leakage).

환경변수로 덮어쓸 수 있는 것: `DATASET_ROOT REPO_ID LEROBOT_VENV OUTPUT_DIR SCRATCH_DIR
STEPS BATCH_SIZE SAVE_FREQ EVAL_STEPS LOG_FREQ EVAL_SPLIT SEED NUM_WORKERS JOB_NAME`
(FM은 추가로 `NUM_INTEGRATION_STEPS IMAGE_RESIZE_SHAPE IMAGE_CROP_SHAPE`).

기본 경로:

```
~/workspace/youngwoong/carrot_eef/
├── dataset/carrot_in_pot_eef_lerobot/     # A1 산출물
├── outputs/{act_carrot_eef,fm_carrot_eef}/  # checkpoint
├── .cache/{torch_home,hf_lerobot_home}/     # SCRATCH_DIR
└── training/                                # 이 번들
    ├── runtime/<job>/{train.pid,train.log}
    └── evaluations/<job>/
```

venv는 **새로 만들지 않는다.** `LEROBOT_VENV` 기본값이
`~/workspace/youngwoong/cube_flow_matching/training/lr_env`(torch 2.11+cu128,
lerobot 0.6.1, CLIP 캐시 보유)이고 **그것을 재사용한다.**

---

## 4. 실행 (integrator 전용)

```bash
nvidia-smi                              # 다시 한 번, 바로 직전에
GPU_ID=<a> ./launch_detached.sh act
GPU_ID=<b> ./launch_detached.sh fm
```

- 내부에서 `preflight.sh`를 먼저 돌리고, 통과해야 `setsid nohup`으로 분리한다.
  SSH 세션이 끊겨도, 에이전트 하네스가 자식을 회수해도 살아남는다.
- `runtime/<job>/train.pid`에는 **진짜 `lerobot-train` PID**가 들어간다.
  `setsid`가 돌려주는 것은 wrapper PID라서 그걸로 `kill`하면 아무 일도 안 일어난다 —
  그래서 `--job_name=<job>`로 argv를 다시 찾아 최소 PID를 기록한다.
- `OUTPUT_DIR`이 이미 있으면 거부한다(덮어쓰기 사고 방지).
  이어서 학습하려면 lerobot 네이티브 resume을 쓴다:
  `lerobot-train --config_path=<ckpt>/train_config.json --resume=true --steps=<N>`.

모니터링:

```bash
tail -f runtime/act_carrot_eef/train.log
cat runtime/act_carrot_eef/train.pid
nvidia-smi
ps -o pid,etime,%cpu,%mem,cmd -p "$(cat runtime/act_carrot_eef/train.pid)"
```

중지: `kill "$(cat runtime/<job>/train.pid)"`.

---

## 5. 평가 (open-loop, held-out 48–53)

**학습 중인 GPU가 아닌 다른 유휴 GPU**(또는 CPU)에서 돌린다.

```bash
GPU_ID=<c> ./eval_checkpoints.sh act
GPU_ID=<c> ./eval_checkpoints.sh fm
DEVICE=cpu ./eval_checkpoints.sh fm        # GPU가 없을 때
```

체크포인트 하나만:

```bash
~/workspace/youngwoong/cube_flow_matching/training/lr_env/bin/python eval_offline_eef.py \
  --checkpoint ~/workspace/youngwoong/carrot_eef/outputs/fm_carrot_eef/checkpoints/050000/pretrained_model \
  --dataset-root ~/workspace/youngwoong/carrot_eef/dataset/carrot_in_pot_eef_lerobot \
  --device cuda --num-integration-steps 10 --out evaluations/fm_carrot_eef/050000
```

- 정책 클래스는 **checkpoint의 config에서** 뽑는다(`PreTrainedConfig.from_pretrained` +
  `get_policy_class`) — `fm_server.py`와 같은 방식이라 ACT/FM 둘 다 같은 스크립트로 된다.
- 이미지 경로는 **학습과 같아야 한다**: `--image-resize auto`(기본)가 ACT면 360×640
  resize를, `multi_task_dit`이면 무변환(정책이 내부에서 224로)을 고른다.
- `--num-integration-steps` 기본 **10** — 배포(`fm_server.py`)와 같은 값. ACT에서는 무시된다.
- `--n-action-steps`로 receding-horizon 길이를 바꿀 수 있다(기본 = 학습값).
- 출력: `evaluations/<job>/<step>/{metrics.json,episodes.csv}` + `evaluations/<job>/summary.csv`.

지표(액션이 **30 fps 프레임당 delta**임을 잊지 말 것 — 실측 `|Δp|` 중앙값이 2.2 mm다):

| 이름 | 뜻 |
| --- | --- |
| `position_mae_m` | `action[:3]` 3축 평균 절대오차 [m] |
| `position_err_norm_m` | 프레임별 `‖dp_pred − dp_gt‖` 평균 [m] |
| `rotation_mae_rad` | 프레임별 **rotvec 차이의 norm** 평균 [rad] |
| `rotation_axis_mae_rad` | `action[3:6]` 3축 평균 절대오차 [rad] |
| `gripper_mae` / `gripper_accuracy` | 절대 그리퍼 명령 오차 / 0.5 임계 일치율 |
| `overall_l1` | 7차원 전체 평균 절대오차 |

> ⚠️ open-loop 지표는 **정답 관측을 계속 먹여 주는** 수치라 실제 rollout 성공률이 아니다.
> 오차가 누적되지 않으므로 배포 성능의 상한에 가깝다. 실기 성능은 로봇에서 확인해야 한다.

비디오 디코딩이 실패하면 `--video-backend pyav`(또는 `eval_checkpoints.sh`는
`eval_offline_eef.py`를 직접 호출하도록) 바꿔 본다.

---

## 6. checkpoint를 laptop3로 가져오기

배포 스택은 `ros2_ur_ws/src/gello_policy/checkpoints/<name>`을 본다
(worktree에서는 심링크). `pretrained_model` 디렉터리 **통째로** 가져온다 —
`config.json` · `model.safetensors` · `policy_preprocessor.json` ·
`policy_postprocessor.json` 넷이 다 있어야 서버가 뜬다.

```bash
# laptop3에서
CKPT=~/workspace/youngwoong/carrot_eef/outputs/fm_carrot_eef/checkpoints/100000/pretrained_model
DEST=/home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy/checkpoints/carrot_eef_fm_100k
mkdir -p "$DEST"
rsync -av --partial "kanu:$CKPT/" "$DEST/"
ls -la "$DEST"      # config.json model.safetensors policy_preprocessor.json policy_postprocessor.json
```

ACT면 `.../outputs/act_carrot_eef/checkpoints/050000/pretrained_model` →
`checkpoints/carrot_eef_act_50k`.

그 다음은 A5/A6의 배포 경로다 — 서버를 `--action-space eef_delta_v1`로 띄우고
(`ACT_ACTION_SPACE`/`FM_ACTION_SPACE`), RESET handshake가 `state_dim 16 / action_dim 7`을
보고해야 노드가 arming을 허용한다. 절차는
`docs/ros2/GELLO_UR7E_CARROT_EEF_DEPLOY_KO.md`(A6).

---

## 7. 함정 모음 (참조 레시피에서 그대로 물려받은 것)

- **`./train.sh &`로 띄우면 죽는다.** 세션·에이전트 턴 경계에서 회수된다.
  반드시 `launch_detached.sh`(= `setsid nohup … </dev/null`).
- **`setsid`가 준 PID로 `kill`하면 아무 일도 안 일어난다.** wrapper PID다.
  `runtime/<job>/train.pid`를 쓴다.
- **캐시를 `$HOME` 기본값으로 두지 말 것.** kanu의 `/`는 98 % 찼다.
  `SCRATCH_DIR`(기본 `carrot_eef/.cache`)로 `TORCH_HOME`·`HF_LEROBOT_HOME`을 돌린다.
- **pip NVIDIA 라이브러리 경로를 `LD_LIBRARY_PATH`에 넣어야 한다.** torch와 torchcodec이
  거기 링크되어 있다(CPU 평가에서도 torchcodec이 CUDA NPP를 찾는다). `_common.sh`가 처리한다.
- **batch 8 fp32는 이 계열 워크로드에서 ~9.7 GB.** A4000 16 GB면 여유가 있지만,
  GPU를 남과 나눠 쓰면 OOM이 난다 — 그래서 preflight가 compute 프로세스를 거부한다.
- **GPU 4는 남의 것일 수 있다.** 목록을 믿지 말고 **띄우기 직전에** `nvidia-smi`를 본다.
