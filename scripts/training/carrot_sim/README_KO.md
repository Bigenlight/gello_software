# carrot_in_pot **시뮬** 관절공간 학습 번들 (ACT / Diffusion / Flow-Matching)

MuJoCo 시뮬에서 모은 `carrot_in_pot` take를 **관절공간(joint-space)** 정책 3종으로
학습한다. 학습은 전부 **kanu**에서, `lerobot-train`(lerobot 0.6.1)으로 돈다.

> 🔴 **SIM 전용이다. 실기 데이터와 절대 섞지 않는다.**
> repo_id·데이터셋 경로·job root·job 이름이 전부 실기(`carrot_eef`)와 분리돼 있고,
> 이 번들의 어떤 스크립트도 `carrot_eef/`나 실기 데이터셋을 읽거나 쓰지 않는다.

---

## 1. 데이터셋

| | |
| --- | --- |
| repo_id | `Bigenlight/carrot_in_pot_sim_lerobot_v3` (tag **`v3.0`**, commit `b1e4c822…`) |
| 에피소드 / 프레임 | **22 / 10,725** (30 fps) |
| `observation.state` | float32 **[7]** = `ur_q1..ur_q6` + `grip_pos` |
| `action` | float32 **[7]** = `cmd1..cmd6` + `grip_cmd` |
| 카메라 | `observation.images.cam1`, `observation.images.cam2` — **720×1280 RGB**, av1 비디오 |
| depth | **없음** |
| task 문자열 | `"Put carrot in pot"` (22 에피소드 전부 동일, task 1개) |
| 추가 키 | `observation.sim.object_poses` float32 **[14]** (당근·냄비 pose) |

J1(shoulder_pan)은 실기 carrot 데이터와 같은 **−π 브랜치**(≈ −3.3 rad)에 있다.

### 🛑 `observation.sim.object_poses`는 정책 입력이 되면 안 된다 — 그래서 `input_features`를 명시한다

lerobot 0.6.1은 `observation`으로 시작하는 **모든** 키를 `FeatureType.STATE`로 분류하고
(`lerobot/utils/feature_utils.py::dataset_to_policy_features`),
`lerobot/policies/factory.py::make_policy`가

```python
cfg.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
if not cfg.input_features:                       # <-- 비어 있을 때만 자동 추론
    cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}
```

이므로, **그냥 두면 `object_poses`가 정책 입력에 들어간다.** 그건 MuJoCo만 만들 수 있는
ground truth라서 그걸 배운 정책은 **배포 불가능**하다(ZMQ 서버는 state + 이미지 2장만 보낸다).

그래서 세 스크립트 전부 아래를 **명시적으로** 넘긴다:

```bash
--policy.input_features='{"observation.state": {"type": "STATE", "shape": [7]}, "observation.images.cam1": {"type": "VISUAL", "shape": [3, 720, 1280]}, "observation.images.cam2": {"type": "VISUAL", "shape": [3, 720, 1280]}}'
```

* 이미지 shape는 **채널 우선 `[3,720,1280]`**이다. `meta/info.json`은 `[720,1280,3]`로
  적지만 `dataset_to_policy_features`가 전치하므로, info.json 순서를 그대로 쓰면 조용히 틀린다.
* `--policy.output_features`는 **넘기지 않는다.** `make_policy`가 데이터셋에서 무조건
  덮어쓰므로 CLI 값은 버려진다. (실측: 세 번의 dry run 모두 `config.json`의
  `output_features == {"action": ACTION [7]}`로 나왔다.)
* 이 한 줄이 빠지면 **`preflight.sh` 검사 [6]이 실패한다**(음성 테스트로 확인).

---

## 2. 파일

| 파일 | 하는 일 |
| --- | --- |
| `_common.sh` | 경로·기본값·`input_features` JSON·GPU/PID/디스크 헬퍼. **부작용 없음**(변수 정의만) |
| `train_act_carrot_sim.sh` | ACT 50k step, batch 8, save_freq 10k, eval_steps 2k, seed 1000, 데이터셋 레벨 Resize **[360,640]** |
| `train_diffusion_carrot_sim.sh` | Diffusion 100k step, `drop_n_last_frames=31`, `policy.resize_shape=[360,640]` + 같은 데이터셋 Resize |
| `train_fm_carrot_sim.sh` | Flow-Matching(`multi_task_dit`, `objective=flow_matching`) 100k step, 내부 **224²** resize/crop, `num_integration_steps=100` |
| `preflight.sh` | 발사 게이트 6종. **GPU 없이도 통과 가능**(`--no-gpu`) |
| `launch_detached.sh` | job 1개를 setsid+nohup으로 분리 실행 |
| `run_queue.sh` | **GPU 하나에서 여러 job을 순차 실행**(첫 실패에서 체인 중단) |
| `sync_to_kanu.sh` | 번들(+옵션으로 데이터셋)을 kanu로 rsync |

각 레시피의 **출처**: ACT·Diffusion은 `/home/laptop3/youngwoong_ws/banana-in-pot-experiments/`의
`train_act.sh` / `train_diffusion_joint_valdiag.sh`, FM은
`/home/laptop3/youngwoong_ws/cube_flow_matching_training/train_flow_matching.sh`.
바꾼 것은 데이터셋·job 이름·`eval_split`·`input_features` 넷뿐이고, 각 스크립트 머리말에
무엇을 왜 바꿨는지 적혀 있다.

---

## 3. 검증 분할 — 마지막 2개(20, 21)

`EVAL_SPLIT=0.09`. lerobot 0.6.1 `datasets/factory.py::make_train_eval_datasets`는
**task별로** `n_eval = ceil(len(eps) * eval_split)`를 잡고 **뒤에서** 그만큼 떼낸다.
이 데이터셋은 task가 하나뿐이라 그룹이 곧 전체다:

```
ceil(22 × 0.09) = ceil(1.98) = 2   →  held out = [20, 21]
"exactly 2"가 성립하는 구간: 1/22 < eval_split ≤ 2/22  =  (0.04546, 0.09091]
```

실측 로그: `Train/eval split: 20 train, 2 eval (eval_split=0.09, 1 tasks)`.

⚠️ task 문자열이 여러 개인 데이터셋으로 바뀌면 "마지막 2개"의 의미가 달라진다 —
`preflight.sh` 검사 [5]가 **task가 정확히 1개인지**를 확인해서 이 전제를 지킨다.

---

## 4. 2-GPU 운용 계획

kanu는 **8 GPU를 다른 사람들과 공유**한다. 우리 예산은 **동시 학습 2개**이고,
`preflight.sh` 검사 [2]가 `--job_name`으로 **우리 job만**
(`act_carrot_eef` / `fm_carrot_eef` / `*_carrot_sim`) 세어서 강제한다.

학습할 정책은 3개이므로 **GPU 하나에 둘을 사슬로 건다**:

```
GPU A :  act_carrot_sim  →(끝나면 자동으로)→  diffusion_carrot_sim      # run_queue.sh
GPU B :  fm_carrot_sim                                                  # launch_detached.sh
```

```bash
cd ~/workspace/youngwoong/carrot_sim/training
nvidia-smi                      # 지금 비어 있는 GPU를 고른다. 캐시된 정보를 믿지 말 것
./run_queue.sh 4 act diffusion  # GPU 4: ACT 먼저, 끝나면 Diffusion
GPU_ID=5 ./launch_detached.sh fm

./run_queue.sh --status 4       # 체인 상태 + 각 job의 status 파일
./run_queue.sh --stop 4         # 체인 감독 프로세스 → 현재 trainer 순으로 정지
tail -f runtime/act_carrot_sim/train.log
```

`run_queue.sh`는 **각 job 직전에 `preflight.sh`를 다시 돌린다.** 몇 시간 뒤 두 번째 job이
시작할 때 GPU를 누가 가져갔을 수 있기 때문이다. 첫 실패에서 체인은 멈추고
`runtime/<job>/status`에 `FAILED rc=<n>`이 남는다. **재시도·재개는 하지 않는다** —
100k step짜리 실패는 사람이 로그를 봐야 한다.

---

## 5. 🔴 디스크 — 지금 상태로는 Diffusion이 안 들어간다

lerobot은 **주기 체크포인트를 하나도 지우지 않는다**(`checkpoints/last`는 심링크일 뿐,
retention 옵션 자체가 `TrainPipelineConfig`에 없다). kanu 실측(`carrot_eef`):
ACT 체크포인트 **592 MiB**, FM 체크포인트 **2.4 GiB** → 학습 파라미터당 약 **12 B**
(fp32 가중치 + AdamW 모멘트 2개).

| job | 학습 파라미터 | 체크포인트 1개 | 개수 | 합계 |
| --- | --- | --- | --- | --- |
| `act_carrot_sim` | 51.6 M | 0.6 GiB | 5 | **≈ 3 GiB** |
| `diffusion_carrot_sim` | 277.9 M | 3.3 GiB | 10 | **≈ 33 GiB** |
| `fm_carrot_sim` | 186.0 M | 2.4 GiB | 10 | **≈ 24 GiB** |

**2026-09-15 기준 kanu 여유 공간은 43 GiB이고, 여기에 `fm_carrot_eef`가 아직 쓰는 중이다.**
그래서 `preflight.sh` 검사 [3]은 job별로 위 값(+여유)을 요구하며,
**`diffusion`은 36 GiB를 요구해 지금은 FAIL한다.** 선택지는 셋이다:

1. 공간을 비운다(오래된 `carrot_eef` 체크포인트 등 — **소유자 확인 후**),
2. `SAVE_FREQ=25000 ./run_queue.sh …`처럼 체크포인트 수를 줄인다(Diffusion 4개 ≈ 13 GiB),
3. `MIN_FREE_GIB=N`으로 게이트를 낮추고 **학습 중에 사람이 직접 오래된 체크포인트를 지운다.**

디스크가 떨어지면 체크포인트 저장 시점(safetensors 직렬화)에서 죽는다 — 몇 시간을 날린다.
(이 번들을 만드는 동안 laptop3에서 실제로 그렇게 죽었다.)

---

## 6. 명령 요약

### laptop3에서

```bash
cd /home/laptop3/gello_software_humble/scripts/training/carrot_sim
./sync_to_kanu.sh --dry-run --bundle-only     # 먼저 항상 dry-run
./sync_to_kanu.sh --bundle-only               # 데이터셋은 kanu가 Hub에서 직접 받는다
```

데이터셋까지 밀어야 할 때(kanu에 인터넷이 없을 때)는 `--bundle-only`를 빼면 된다.

### kanu에서

```bash
cd ~/workspace/youngwoong/carrot_sim/training
./preflight.sh --no-gpu                        # GPU 없이 스키마·feature 게이트만
DRY_RUN=1 GPU_ID=0 ./train_act_carrot_sim.sh   # 실행될 명령 그대로 출력, 아무것도 안 돈다
DRY_RUN=1 GPU_ID=0 ./train_diffusion_carrot_sim.sh
DRY_RUN=1 GPU_ID=0 ./train_fm_carrot_sim.sh
```

CPU 1-step 연기 테스트(정말 도는지 확인하고 싶을 때, GPU 불필요):

```bash
POLICY_DEVICE=cpu STEPS=1 BATCH_SIZE=2 NUM_WORKERS=0 \
  OUTPUT_DIR=/tmp/smoke_act ./train_act_carrot_sim.sh
```

---

## 7. 체크포인트가 어디에 떨어지나

```
~/workspace/youngwoong/carrot_sim/outputs/<job>/checkpoints/<step>/pretrained_model/
```

* `<job>` ∈ `act_carrot_sim`, `diffusion_carrot_sim`, `fm_carrot_sim`
* `<step>` = `010000`, `020000`, … (`SAVE_FREQ` 간격) + 마지막 step
* `checkpoints/last` 는 가장 최근 `<step>`을 가리키는 **심링크**
* 로그: `~/workspace/youngwoong/carrot_sim/training/runtime/<job>/train.log`
* 체인 로그: `…/runtime/queue_gpu<N>.log`, 상태: `…/runtime/<job>/status`

`pretrained_model/` 안에는 `config.json`, `model.safetensors`,
`policy_preprocessor.json`(+ 정규화 stats `.safetensors`), `policy_postprocessor.json`,
`train_config.json`이 들어 있다.

---

## 8. 다른 세션(서빙 담당)에게 — 서버가 무엇을 보내야 하나

**전처리 파이프라인에 resize 단계가 없다.** `policy_preprocessor.json` 실측:

| 정책 | preprocessor 단계 | 서버가 보낼 것 |
| --- | --- | --- |
| ACT | rename → to_batch → device → **normalizer** | 이미지를 **서버가 [360,640]으로 리사이즈해서** 보낸다 (실기 ACT 배포 서버와 동일). 학습 시 리사이즈가 **데이터셋 transform**에 있었기 때문에 정책은 스스로 줄이지 않는다 |
| Diffusion | rename → to_batch → device → **normalizer** | 정책 config에 `resize_shape=[360,640]`이 있어 **정책 내부에서** 줄인다. 720×1280 원본을 보내도 되고 [360,640]을 보내도 결과가 같다. `crop_shape=null`(크롭 없음) |
| FM | rename → to_batch → **tokenizer** → device → **normalizer** | 정책이 내부에서 **224×224**로 resize+crop한다. 추가로 **`task` 문자열이 필요하다** — `openai/clip-vit-base-patch16` 토크나이저, `max_length=77`. 문자열은 정확히 **`"Put carrot in pot"`** |

공통:

* 관측 키는 정확히 **`observation.state`(7), `observation.images.cam1`, `observation.images.cam2`** 셋뿐이다.
  `observation.sim.object_poses`는 **보내지 않는다**(정책이 모른다).
* 이미지 텐서는 **채널 우선 `[3,H,W]`**, `[0,1]` float (lerobot 규약).
* 출력은 `action` **[7]** = `cmd1..cmd6` + `grip_cmd`. postprocessor는
  `unnormalizer → device` 뿐이다.
* 정규화 규약이 정책마다 다르다 — ACT는 STATE/ACTION 모두 `MEAN_STD`, Diffusion·FM은
  `MIN_MAX`다. **체크포인트의 preprocessor가 알아서 처리하므로 서버가 손댈 것은 없다.**
  다만 정규화 통계는 **22 에피소드 전체**(held-out 20·21 포함)로 계산된 것임을 알아 둘 것.
* FM의 `num_integration_steps`는 학습 때 100이지만 **추론에서는 낮춰도 된다**(실기 FM 서버는 10).

---

## 9. 알려진 한계 / 미검증

* 실제 GPU 학습은 **아직 한 번도 돌지 않았다.** 검증된 것은 (a) laptop3 CPU에서 세 정책
  전부 1 step 완주 + `config.json` 확인, (b) kanu에서 `DRY_RUN=1`과 `preflight.sh`다.
* `run_queue.sh`의 **체인 전환**(job 1 종료 → job 2 시작)은 실제 장시간 실행으로
  검증되지 않았다. 로직은 `wait`의 종료코드 하나에 달려 있다.
* 수렴 품질에 대한 주장은 없다. 22 에피소드는 적고, held-out은 2개뿐이다.
  Diffusion의 held-out loss는 **확률적**이므로(매 forward마다 새 noise·timestep)
  단일 점이 아니라 **추세**로 읽어야 한다.
