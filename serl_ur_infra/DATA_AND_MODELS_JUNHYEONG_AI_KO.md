# 데이터·모델 위치 — 서버 `junhyeong_ai` (hostname `junhyeong`)

> # 🔴 먼저 읽어라 — **GPU 서버는 더 이상 `kanu`가 아니다**
>
> 2026-07-31에 HIL-SERL learner를 **`kanu` → `junhyeong_ai`** 로 옮겼다.
> 이 리포의 다른 문서 대부분은 `kanu`를 전제로 쓰여 있고, `run_hil_server.sh`의
> **기본값도 아직 kanu**다(코드 변경은 이전 작업의 마지막 단계다 — 아래 §6).
>
> | | 예전 | **지금** |
> | --- | --- | --- |
> | ssh 별칭 | `kanu` | **`junhyeong_ai`** |
> | 주소 | 166.104.35.33 | **166.104.146.29** |
> | hostname | `kanu` | **`junhyeong`** |
> | 계정 | junhyeong | junhyeong (동일) |
> | GPU | RTX A4000 **×8** (sm_86), learner는 GPU 5 | **RTX 5070 Ti ×1** (sm_120 Blackwell), **GPU 0** |
> | 디스크 | 73 GB 여유 (**96% 사용**) | **615 GB 여유** (63% 사용) |
> | RAM | 여유 약 140 GB | **60 GB (여유 약 37 GB)** |
>
> `kanu`는 **아직 살아 있고 아무것도 지우지 않았다.** 이전은 전부 **복사**였다.
> 이 문서의 모든 경로는 특별히 명시하지 않는 한 **`junhyeong_ai` 기준**이다.

작성일 2026-07-31. 작성 근거는 전부 그날 세 머신에서 직접 읽은 실측이다.

---

## 0. 접속

```bash
ssh junhyeong_ai          # laptop3의 ~/.ssh/config에 등록돼 있다
```

키 인증(`~/.ssh/id_ed25519`)이 **필수**다. `run_hil_server.sh`가 터널을
`-o BatchMode=yes`로 열기 때문에, 비밀번호 프롬프트는 "물어보기"가 아니라 **즉시 실패**다.

⚠️ **GPU 1장을 다른 사람과 공유한다.** 계정도 같은 `junhyeong`이다. GPU 작업 전에
`nvidia-smi`로 점유를 확인한다.

⚠️ **`/home/junhyeong/gello_software`는 다른 사람의 작업 트리다** — 커밋 안 된 변경이 있다.
읽지도 쓰지도 말 것. HIL이 쓰는 checkout은 `_hil` 접미사가 붙은 쪽이다(§2).

---

## 1. 한눈에 — 무엇이 어디에 있나

```
junhyeong_ai:/home/junhyeong/
├── gello_software_hil/          441M   HIL learner 코드 (§2)
├── gello_software/                     🚫 다른 사람 작업 트리 — 건드리지 말 것
├── hil-serl-data/              ~5.1G   ★ 데이터·모델 전부 여기 (§3)
│   ├── demos/                   195M     학습용 canonical demo
│   ├── classifier_ckpt/          43M     운영 reward classifier 모델
│   ├── datasets/                4.7G     classifier 재학습용 원재료
│   ├── runs/                       -     learner 출력 (여기에 쌓인다)
│   └── archive/                 222M     kanu 시절 이력 (읽기 전용)
└── miniconda3/envs/il/                 learner 파이썬 환경 (§4)
```

**설계 원칙 하나만 기억하면 된다: 데이터와 모델은 전부 `~/hil-serl-data` 아래에 있다.**
kanu 시절에는 demo·classifier·runs·lock이 서로 다른 네 군데(그중 하나는 FM 스택
디렉터리 안)에 흩어져 있었다. 그걸 이번에 한 뿌리로 모았다.

---

## 2. 코드 checkout

| | |
| --- | --- |
| 경로 | `/home/junhyeong/gello_software_hil` |
| 브랜치 | `feat/gello-ur7e-humble-22.04` |
| HEAD | `ed60212` (= laptop3 = GitHub tip, 2026-07-31 3자 일치 실측) |
| origin | `https://github.com/Bigenlight/gello_software.git` |
| submodule | `third_party/hil-serl` @ `c32939b` (이것만 init. DynamixelSDK·mujoco_menagerie는 kanu와 마찬가지로 미초기화) |
| 크기 | 441 MB |

**독립 clone이다** — `.git`이 디렉터리, `alternates` 없음, `worktree list` 1개,
partial/promisor 아님, `fsck` 통과. 전진은 laptop3와 똑같이 **`git pull --ff-only`**다.

🪤 **절대 하지 말 것: `--reference` clone, `git worktree add`, 두 번째 HIL checkout.**
2026-07-30에 kanu에서 그 셋 때문에 하루를 잃었다 — 자세한 사고 경위는
[`HIL_SERL_KANU_RUNBOOK_KO.md`](./HIL_SERL_KANU_RUNBOOK_KO.md) §1.1의 🪤.

`gello_software_hil`과 `gello_software`(다른 사람 것)가 **같은 머신에 둘 있는 것은 정상**이다.
금지된 것은 *같은 스택의* checkout이 둘인 것과, 하나가 다른 하나에 사슬로 물리는 것이다.

---

## 3. 데이터와 모델

### 3.1 학습 데이터 — canonical offline demo ★ 없으면 learner가 시작조차 안 한다

```
~/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl
```

| | |
| --- | --- |
| 크기 | 203,573,172 B (195 MB) |
| SHA256 | `f97185582401ce7570d44fddc33d1bd64b215d7e32d6384d5fe13e1b405032fa` |
| 내용 | 성공 take 23개 → **transition 2,037개** (`take_23` 제외) |
| 사본 | `kanu:~/hil-serl-data/demos/` · `laptop3:~/hil-serl-artifacts/demos/` — **총 3벌, 전부 동일 해시** |

`run_hil_server.sh`가 기동 전에 이 SHA를 검사한다. **learner는 offline demo가 0이면
학습을 시작하지 않는다.** 생성 절차는
[`RECORDED_TAKE_DEMO_CONVERSION_KO.md`](./RECORDED_TAKE_DEMO_CONVERSION_KO.md).

### 3.2 운영 모델 — reward classifier

```
~/hil-serl-data/classifier_ckpt/checkpoint_150/
```

| | |
| --- | --- |
| 크기 | 44,748,408 B (43 MB), 파일 14개 (orbax 체크포인트 **디렉터리**) |
| directory SHA256 | `512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d` |
| 모델 ID | `cube-in-cup-all3-ckpt150+sidecar-v1` |
| threshold | **0.5** (strict `p > 0.5`) |
| 사본 | `kanu:~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/` · `laptop3:gello_software/classifier_ckpt/cube_in_cup_all3/checkpoint_150` (gitignore됨) — **3벌 동일 해시** |

해시는 `sha256sum`이 아니라 **프로젝트 자체 함수**로 계산한다(서버가 검사하는 것도 이것이다):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/gello_software_hil/serl_ur_infra \
  python -c "from ur_env.classifier_sidecar import directory_sha256; \
             print(directory_sha256('$HOME/hil-serl-data/classifier_ckpt/checkpoint_150'))"
```

🚚 **kanu 시절 이 모델은 FM/diffusion 스택 디렉터리
(`workspace/youngwoong/dataset/…`) 안에 있었다.** 이번에 HIL 데이터 뿌리로 꺼냈다.
옛 경로를 가리키는 문서를 보면 그건 kanu 기준이다.

### 3.3 classifier 재학습용 원재료

```
~/hil-serl-data/datasets/
    cube_in_cup_all3/                          737M   ← checkpoint_150을 학습시킨 그 데이터 (train/ 포함)
    cube_in_cup_combined/                      763M
    cube_in_cup_cv/                            2.6G
    cube_in_cup_raw_0724/                      215M
    cube_in_cup_reward_classifier_data.tar.gz  168M
    success_0724.zip                           135M   ┐ 라벨된 성공/실패/검증 take.
    fail_0724.zip                               56M   ├ kanu에서는 dataset/ 밖
    take_fail_val.zip                          8.3M   ┘ (workspace/youngwoong/)에 있었다
```

**왜 옮겼나:** classifier 정확도가 부족해 성공 판정 기본값이 `MANUAL`이고, `AUTO`를 운영
기본으로 되돌리려면 **새 데이터로 재학습·재검증**해야 한다. 그 재학습의 재료가 이것이다.
kanu에 두고 왔으면 그 경로가 조용히 막혔을 것이다.

`datasets/cube_in_cup_all3/classifier_ckpt/`는 §3.2와 **의도적으로 중복**이다 —
위쪽은 서버가 경로로 로드하는 운영 artifact, 이쪽은 원래 레이아웃 보존용이다.

### 3.4 learner 출력 — run root

```
~/hil-serl-data/runs/<run_id>/
    checkpoints/   logs/   assets/   wandb/
```

learner가 기동할 때 스스로 만든다. **비어 있는 게 정상이다.**

### 3.5 kanu 시절 이력 (읽기 전용)

```
~/hil-serl-data/archive/
    kanu-runs/       160M   run root 8개 + ARCHIVE_PROVENANCE.txt
    kanu-dry-runs/    42M   warmup 증거 (문서가 경로로 인용한다)
    kanu-probes/      21M   no-submit probe 증거
    kanu-preflight/   8.0K
```

---

## 4. 🔴 없는 것 — 학습된 policy 체크포인트는 **존재하지 않는다**

기대할까 봐 명시한다. **kanu의 run root 8개 전부 `checkpoints/`가 비어 있었다**(6개는
72바이트 `.learner-writer.lock` 하나, 2개는 완전히 빈 디렉터리). 2026-07-31 실측이다.

```
checkpoint_period = 5000        ur_env/learner/config.py:31
--target-learner-step 5000
지금까지 최고 도달 learner step   301
```

**첫 체크포인트가 5000 step에 저장되는데 어떤 lineage도 근처에 가지 못했다.**
그래서 새 서버는 **canonical demo 2,037개로 깨끗한 새 lineage를 시작**한다.

이전 시점 kanu learner의 replay 907 / intervention 586과 가중치는 **그 프로세스의 RAM에만**
있었다. replay는 디스크에 저장되지 않으므로 이전 여부와 무관하게 프로세스 종료와 함께
사라진다. `archive/kanu-runs/`에 있는 것은 **로그·JSONL·wandb 이력이지 모델이 아니다.**

---

## 5. 파이썬 환경

| | |
| --- | --- |
| learner 인터프리터 | `/home/junhyeong/miniconda3/envs/il/bin/python` |
| 기준 | kanu `il` env의 `pip freeze` 그대로 (py 3.10.20) |

learner는 버전을 **exact로 검사하고 fail-closed**다
(`ur_env/learner/agent.py:125` `validate_learner_dependencies`):

```
jax 0.5.3 · jaxlib 0.5.3 · flax 0.10.5 · distrax 0.1.5
tensorflow_probability 0.25.0 · wandb 0.26.0
```

⚠️ **GPU가 Blackwell(sm_120)로 바뀌었다.** kanu는 sm_86이었다. 2026-07-31 측정으로
**jax 0.6.0은 이 GPU에서 정상 동작**이 확인됐다(같은 서버 `qc` env로 backend=gpu,
matmul·cuDNN conv 수치 정답). **jax 0.5.3의 sm_120 동작 여부는 별도 검증 대상이며,
안 되면 핀을 0.6.0으로 올리고 위 expected 딕셔너리와 테스트 기대값을 함께 고쳐야 한다.**

🚫 **다른 env(`base acg expo gr00t lerobot qc robocasa robodiff`)를 수정하지 말 것.**
다른 사람 것이다. 특히 `qc`의 jax를 0.5.3으로 강등시키면 그 사람 작업이 깨진다.

---

## 6. ⚠️ 코드는 아직 kanu를 기본값으로 본다

`ros2_ur_ws/run_hil_server.sh`가 이 서버를 기본으로 삼는 변경은 **이전 작업의 마지막
단계**이며 이 문서 작성 시점에 아직 들어가지 않았다. 그때까지는 환경변수로 지정한다:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
HIL_SSH_HOST=junhyeong_ai \
HIL_KANU_REPO=/home/junhyeong/gello_software_hil \
HIL_GPU_INDEX=0 \
./run_hil_server.sh --check
```

하드코딩돼 있어 **환경변수로 못 바꾸는 것 4개**(`run_hil_server.sh`):

```
:68   REMOTE_RUN_BASE=/home/junhyeong/hil-serl-data/runs                  ← 계정명이 같아 그대로 맞는다
:248  CLASSIFIER=/home/junhyeong/workspace/youngwoong/dataset/…/checkpoint_150   ← 임시 심링크로 우회 중
:250  REAL_DEMO=/home/junhyeong/hil-serl-data/demos/…23takes.pkl          ← 그대로 맞는다
:258  RUN_LOCK=/home/junhyeong/hil-serl-data/.run_hil_server.lock         ← 그대로 맞는다
```

🩹 **임시 심링크가 하나 있다** — 코드를 고치기 전에 옛 스크립트로 시험 기동을 할 수 있게
만든 것이다. **`HIL_REMOTE_DATA_ROOT` 변경이 들어가면 지운다.**

```
~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt
    -> ~/hil-serl-data/classifier_ckpt
```

---

## 7. kanu에 남겨 둔 것 (의도적)

| 항목 | 크기 | 왜 안 옮겼나 |
| --- | --- | --- |
| `workspace/youngwoong/cube_flow_matching/` | 9.4G | **FM/diffusion 스택** — CLAUDE.md §E "다른 스택, 섞지 말 것" |
| `workspace/youngwoong/models/diffusion_banana_in_pot_joint/` | 1.1G | 같은 이유 |
| `workspace/youngwoong/{hil-serl,gello_software,gello_software_remote_classifier}/` | 6.8G | 코드 — git에서 복원 가능 |
| `gello-rescue-backups/` | 333M | 07-29 사고 백업 bundle. 그 ref는 **이미 GitHub에** 있다 (`rescue/kanu-worktree-20260729-014633`, 태그 `kanu-local-c9c30c3-preserved`) |
| `exp/qam-endcorr-q0/` | 836M | 무관한 프로젝트 |
| `miniconda3/envs/il/` | 11G | env는 복사가 아니라 재생성한다 |

그리고 **kanu의 HIL 데이터 원본은 전부 그대로 있다.** 이전은 복사였고 kanu에서는
파일 하나도 지우거나 옮기지 않았다.

---

## 8. laptop3에 있는 것

| 항목 | 경로 |
| --- | --- |
| demo pickle 사본 | `~/hil-serl-artifacts/demos/cube_in_cup_20260720_success_23takes.pkl` |
| classifier 사본 | `gello_software/classifier_ckpt/cube_in_cup_all3/checkpoint_150` (gitignore) |
| 녹화 take 원본 | `gello_software/ros2_ur_ws/gello_logs/` |
| actor venv (gRPC 전용) | `/home/laptop3/venvs/gello-hil-actor/bin/python` |

---

## 9. 관련 문서

- [`../CLAUDE.md`](../CLAUDE.md) — 전체 색인
- [`HIL_SERL_KANU_RUNBOOK_KO.md`](./HIL_SERL_KANU_RUNBOOK_KO.md) — learner 기동 절차.
  **파일명과 본문이 전부 kanu 기준이다** — 서버 주소·GPU·경로는 이 문서가 우선한다
- [`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](./HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) — 현재 진입점
- [`RECORDED_TAKE_DEMO_CONVERSION_KO.md`](./RECORDED_TAKE_DEMO_CONVERSION_KO.md) — take → demo 변환
- [`REWARD_CLASSIFIER_THRESHOLD_KO.md`](./REWARD_CLASSIFIER_THRESHOLD_KO.md) — threshold 근거
