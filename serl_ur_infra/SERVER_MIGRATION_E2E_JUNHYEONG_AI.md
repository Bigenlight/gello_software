# laptop3 ↔ `junhyeong_ai` 실통신 수락 시험 (로봇 없음)

**결론 한 줄: PASS.** 2026-07-31, 실제 gRPC transport로 laptop3에서 새 서버
`junhyeong_ai`의 learner에 **transition 200개(100 × 2 run)** 를 보내
`gRPC → reward finalize → frozen-trunk feature replay → CTA update → policy publish →
checkpoint → 별도 프로세스 resume`까지 전 구간이 돌았다. 두 run 모두 서버가
`rlpd_learner_synthetic_e2e_passed`, 액터가 `fake_e2e_actor_passed`를 냈다.

이 문서는 **서버 이전이 통신까지 실제로 성립하는가**만 다룬다. 데이터·코드 이전 자체는
[`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md)가 정본이다.

| | |
| --- | --- |
| 브랜치 | `test/server-migration-junhyeong-ai` |
| 실행 시점 코드 | laptop3·서버 checkout **양쪽 `1acd903`** |
| 현재 브랜치 tip | `5eaab25` — `1acd903..5eaab25` diff는 셸 문구 5파일뿐이고 **이 시험이 부른 코드에는 한 줄도 없다**(`git diff --stat`로 확인) |
| 도구 | `serl_ur_infra/scripts/run_fake_e2e_actor.py` (수락 도구, 로봇 액터 아님) |
| 서버 모드 | `run_rlpd_learner_server.py --synthetic-e2e` (수동 기동. **`run_hil_server.sh`가 아니다**) |
| laptop3 인터프리터 | `/home/laptop3/venvs/gello-hil-actor/bin/python` (grpcio 1.74.0 / protobuf 3.20.3 / **numpy 2.2.6**) |
| 서버 인터프리터 | `/home/junhyeong/miniconda3/envs/il/bin/python` (`CUDA_VISIBLE_DEVICES=0`) |

---

## 1. 왜 이 명령인가 — 플래그의 근거

명령은 추측이 아니라 **두 스크립트의 인자 검증 코드**에서 역산했다.
`run_rlpd_learner_server.py::_validate_args`가 `--synthetic-e2e`에 서로 물린 제약을 건다.

| 플래그 | 값 | 왜 이 값이어야 하는가 (코드 근거) |
| --- | --- | --- |
| `--synthetic-transition-count` | `100` | `_validate_args`: `!= LearnerConfig().training_starts` 면 거부. `training_starts=100`이라 **100 외의 값은 불법**이다 |
| `--replay-capacity` | `128` | 같은 검사가 `>= training_starts`(100)를 요구. 128은 충족하면서 ring을 21 MB로 묶는다(운영 50,000은 7.32 GiB) |
| `--intervention-capacity` | `32` | 이 run은 `intervened=0`만 보내므로 개입 버퍼는 쓰이지 않는다. 양수 최소 규모 |
| `--target-learner-step` | `1` → `2` | `[1,10]` 범위 + `_validate_synthetic_progress_target`이 **정확히 `시작 step + 1`** 을 요구. fresh는 0→1이라 `1`, resume은 1→2라 `2` |
| `--synthetic-actor-id` / `--synthetic-run-id` | `fake-e2e-actor` / run별 고정 문자열 | 서버가 `allowed_actor_ids`/`allowed_run_ids`로 **정확히 이 한 쌍만** 수락한다. 액터의 `--actor-id`/`--run-id`와 글자까지 같아야 한다 |
| `--demo-path` | `synthetic-e2e/fake_canonical_demo.pkl` | `_validate_demo_serving_scope`가 `--synthetic-e2e`에서는 **모든 demo item이 synthetic 마커를 달 것**을 요구한다. 운영 2,037개 canonical demo는 이 모드에서 **거부된다**(그 반대도 성립: synthetic demo는 운영 serving에서 거부) |
| `--grasp-penalty` | `-0.02` | fake demo의 두 번째 transition이 `-0.02`를 들고 있어 `_validate_demo_grasp_penalty`가 `{0, -0.02}`만 허용한다. 액터도 같은 값을 보낸다 |
| `--require-jax-backend gpu` | — | CPU 폴백을 성능 저하가 아니라 **기동 실패**로 만든다 (Blackwell sm_120 회귀 감시) |
| `--synthetic-timeout-s` | `1500` | `[1, 1800]` 범위. deadline은 **server-ready 시점부터** 흐르므로 터널 개설 + 액터 실행을 넉넉히 덮는다 |
| `--resume-latest` (2번째 run만) | — | 같은 `--checkpoint-root`에서 `checkpoint_000000000001`을 복원해 **별도 프로세스 resume**을 만든다 |
| `--host 127.0.0.1` | — | `_validate_args`가 loopback만 허용한다. 외부 노출은 **SSH 터널이 유일한 경로**다 |
| `--expected-start-policy-version` (액터) | `0` → `1` | 서버가 되감기거나 엉뚱한 lineage에 붙는 것을 액터 쪽에서 즉시 거부하게 한다 |

`--synthetic-e2e`는 `publish_period=1`, `checkpoint_period=1`만 줄이고
**batch 256 · CTA 2 · optimizer · 모델은 운영과 동일**하게 둔다
(`_learner_config` 주석). 그래서 이 run은 "축소판 학습"이 아니라 **운영과 같은 1 step**이다.

또 서버는 정책 model id를
`hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`로 광고한다 —
**운영 액터가 실수로 붙는 것을 handshake에서 막기 위한 별도 id**이고,
`run_fake_e2e_actor.py`는 그 id를 pin한다.

### 실제 명령

서버 (`junhyeong_ai`, 전문은 `~/hil-serl-data/synthetic-e2e/launch_synthetic_learner.sh`):

```bash
cd /home/junhyeong/gello_software_runtime
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_SILENT=true WANDB_DISABLE_CODE=true \
PYTHONPATH=$REPO/serl_ur_infra:$REPO/third_party/hil-serl/serl_launcher \
/home/junhyeong/miniconda3/envs/il/bin/python \
  serl_ur_infra/scripts/run_rlpd_learner_server.py \
  --host 127.0.0.1 --port 50053 \
  --classifier-checkpoint ~/hil-serl-data/classifier_ckpt/checkpoint_150 \
  --expected-classifier-sha256 512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d \
  --reward-threshold 0.5 --reward-model-id cube-in-cup-all3-ckpt150+sidecar-v1 \
  --success-confirmations 1 \
  --demo-path ~/hil-serl-data/synthetic-e2e/fake_canonical_demo.pkl \
  --checkpoint-root  ~/hil-serl-data/synthetic-e2e/run/checkpoints \
  --checkpoint-reserve-gib 2 \
  --jsonl-path ~/hil-serl-data/synthetic-e2e/run/logs/learner.jsonl \
  --memory-preflight-path ~/hil-serl-data/synthetic-e2e/run/logs/memory-preflight.jsonl \
  --wandb-dir ~/hil-serl-data/synthetic-e2e/run/wandb --wandb-mode offline \
  --wandb-project hil-serl --run-name synthetic-e2e-fresh \
  --hil-serl-root $REPO/third_party/hil-serl \
  --resnet-source $REPO/third_party/hil-serl/examples/experiments/resnet10_params.pkl \
  --resnet-cache ~/hil-serl-data/synthetic-e2e/run/assets/resnet10_params.pkl \
  --replay-capacity 128 --intervention-capacity 32 \
  --feature-memory-reserve-gib 2 --demo-extraction-batch-size 64 \
  --grasp-penalty -0.02 --utd-ratio 1 --max-workers 4 \
  --max-message-bytes 16777216 --require-jax-backend gpu \
  --target-learner-step 1 --poll-interval 0.1 \
  --synthetic-e2e --synthetic-actor-id fake-e2e-actor \
  --synthetic-run-id fake-e2e-junhyeong-fresh-20260731 \
  --synthetic-transition-count 100 --synthetic-timeout-s 1500
# resume run: --resume-latest, --target-learner-step 2,
#             --synthetic-run-id fake-e2e-junhyeong-resume-20260731
```

터널 (laptop3):

```bash
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -N -L 127.0.0.1:50053:127.0.0.1:50053 junhyeong_ai
```

액터 (laptop3):

```bash
cd /home/laptop3/gello_software
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/serl_ur_infra \
/home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/run_fake_e2e_actor.py \
  --target 127.0.0.1:50053 --actor-id fake-e2e-actor \
  --run-id fake-e2e-junhyeong-fresh-20260731 \
  --transition-count 100 --expected-start-policy-version 0 \
  --expected-reward-model-id cube-in-cup-all3-ckpt150+sidecar-v1 \
  --grasp-penalty -0.02 --timeout-s 30 --max-response-age-s 120
```

🚫 **시스템 `python3`로 이 액터를 돌리지 말 것** — ROS Humble의 grpcio 1.30.2가
오류 없이 100 % CPU로 영원히 멈춘다. 프로젝트 하드 룰이다.

---

## 2. 측정 결과 — 단계별

### 2.1 기동 (fresh)

```
RAM preflight   available 37,148,053,504 B / required 2,168,748,416 B / margin 34,979,305,088 B  → accepted
jax_backend     gpu, device 1개
정책 model id   hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1
reward model id cube-in-cup-all3-ckpt150+sidecar-v1
classifier sha  512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d  (기동 시 검증 통과)
fingerprint     1a5c96f5a2e7f13f109617c8859e0cc6c5d6be949ac76a5de6b95e6808c4c844  (두 프로세스 동일)
CTA warm-up     44,229.5 ms / 3 outer step / 6 gradient update, single-feature warm-up 71.5 ms
demo            fake_canonical_demo.pkl, 2 transition, 394,896 B,
                sha256 6e8ad1dee2ed487301a43ab37eeed8773c988419f073460a8b410e03ad201ced
```

### 2.2 두 run의 전 구간 수치

| 항목 | fresh run | resume run |
| --- | --- | --- |
| 액터 결과 | `fake_e2e_actor_passed` | `fake_e2e_actor_passed` |
| 수락된 transition | **100 / 100** | **100 / 100** |
| `replay_insert_delta` | **100** (서버가 정확히 100을 강제) | **100** |
| `replay_size` | 100 | 100 |
| feature replay insert | 100 (= replay insert. raw pixel은 저장 안 됨) | 100 |
| 시작/끝 policy version (액터 관측) | 0 / 0 | 1 / 1 |
| learner step | **0 → 1** | **1 → 2** |
| gradient step | **0 → 2** (CTA 2:1) | **2 → 4** |
| policy version | **0 → 1** | **1 → 2** |
| `policy_published` | 1건 | 1건 |
| checkpoint | `checkpoint_000000000001` (306 MB) | `checkpoint_000000000002` |
| checkpoint round-trip | **verified: true** | **verified: true** |
| 프로세스 종료 | `learner_process_stopped exit_code=0` | `exit_code=0` |
| update loss 유한성 | 전 metric finite (non-finite 0건) | 전 metric finite (non-finite 0건) |
| `learner_step_ms` | 2,781.9 | 712.1 |
| `critic_update_ms` / `full_update_ms` / `sample_ms` | 47.4 / 50.2 / 17.0 | 90.8 / 46.3 / 18.8 |
| batch 구성 | online 100 + offline demo 2 (RLPD 50:50 = 128:128) | 동일 |

resume run은 `learner_process_ready`에
`restored_checkpoint=.../checkpoint_000000000001`, `learner_step=1`,
`gradient_step=2`, `policy_version=1`을 기록했다 — **fresh-process resume이 실제로
복원에서 출발했다.** fingerprint도 두 프로세스가 동일해서 lineage가 이어졌다.

### 2.3 액터 측 자체 보고 (도구 원본 출력)

```json
{"begin_round_trip_ms_max":82.362063,"begin_round_trip_ms_mean":57.984199059999995,
 "classifier_success_count":0,"event":"fake_e2e_actor_passed","first_policy_version":0,
 "last_policy_version":0,"replay_insert_delta":100,"replay_size":100,
 "run_id":"fake-e2e-junhyeong-fresh-20260731","target":"127.0.0.1:50053","transition_count":100}

{"begin_round_trip_ms_max":84.562271,"begin_round_trip_ms_mean":58.45641606,
 "classifier_success_count":0,"event":"fake_e2e_actor_passed","first_policy_version":1,
 "last_policy_version":1,"replay_insert_delta":100,"replay_size":100,
 "run_id":"fake-e2e-junhyeong-resume-20260731","target":"127.0.0.1:50053","transition_count":100}
```

---

## 3. RPC 지연 — 이 링크의 실측

fresh run에서 `GrpcActorNetwork._call`을 감싸는 **동작 무변경 계측 shim**으로 잰 값이다
(shim은 타이밍만 기록하고 도구의 `run()`을 그대로 호출한다).
resume run은 **shim 없이 원본 CLI**로 돌렸고, 도구가 스스로 보고한 BeginEpisode
mean 58.456 / max 84.562 ms가 shim의 57.984 / 82.362와 일치한다 →
**shim이 값을 왜곡하지 않았다.**

| RPC | n | mean | p50 | p95 | max | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BeginEpisode` | 100 | **57.7 ms** | 57.0 | 70.3 | **82.2** | 17.7 |
| `Step` | 100 | **156.1 ms** | 153.2 | 179.6 | **211.0** | 145.5 |
| `GetServerInfo` | 100 | 3.6 ms | 2.6 | 8.2 | 20.7 | 1.7 |
| `GetBufferStatus` | 2 | 28.1 ms | — | — | 53.4 | 2.9 |
| `Health` | 1 | 55.7 ms | — | — | 55.7 | 55.7 |
| **transition당 합계** (`BeginEpisode`+`Step`) | 100 | **213.8 ms** | 210.8 | 237.9 | **268.7** | 180.0 |

100 transition 벽시계 **21,996 ms → 4.55 transition/s**.

### 3.1 kanu와의 비교

| 비교 대상 | kanu | **junhyeong_ai** | 판정 |
| --- | --- | --- | --- |
| `BeginEpisode` mean (§5.5 schema v1) | 63.29 ms | **57.7 ms** | 약간 빠름 |
| `BeginEpisode` max (§5.5) | 91.16 ms | **82.2 ms** | 약간 빠름 |
| `BeginEpisode` mean / max (§5.6 schema v2) | 84.89 / **372.82 ms** | **57.7 / 82.2 ms** | **명확히 빠르고 tail이 4.5배 짧다** |
| 운영 로봇 루프 주기 (kanu 실측) | 512 ms 평균 / 854 ms 최대 (1.95 Hz) | 이 시험의 transition당 RPC 합계 **213.8 ms 평균 / 268.7 ms 최대** | 아래 주의 참조 |

📏 **네트워크는 원인이 아니다.** ICMP RTT는 두 호스트가 사실상 같다 —
junhyeong_ai `min/avg/max = 1.078/2.811/6.979 ms`, kanu `1.272/2.518/6.314 ms`
(각 10패킷). 즉 위 57 ms / 156 ms는 **거의 전부 서버 연산 + 직렬화**이고
링크 지연은 3 ms 수준이다. 서버를 옮겨도 네트워크 조건은 안 바뀌었다.

⚠️ **213.8 ms를 512 ms와 직접 빼서 읽지 말 것.** kanu의 512 ms는 카메라 디코드와
`env.step`의 100 ms 자체 페이싱을 포함한 **로봇 루프 전체 주기**이고, 여기 213.8 ms는
**RPC 두 개의 합**이다. 정직한 결론은 이것이다: **RPC 구간만 보면 새 서버가 kanu보다
느리지 않고, tail은 뚜렷하게 짧다.** 실기 루프가 실제로 얼마나 빨라지는지는
G21과 함께 **실기에서 다시 재야 한다 — 이 시험으로는 미검증이다.**

⚠️ 그리고 여기 `Step` 156 ms에는 **classifier 추론이 들어 있지 않다**(§4). 실기에서
sidecar가 붙는 약 2 Hz의 Step은 이 값보다 **느릴 수밖에 없다.**

---

## 4. 🛑 이 시험이 증명하지 **못한** 것 — classifier 추론

`classifier_success_count: 0`은 "분류기가 100장을 보고 전부 실패로 판정했다"는 뜻이
**아니다.** `run_fake_e2e_actor.py`는 `build_data()`에 **classifier sidecar를 붙이지
않는다.** `RewardTransitionFinalizer.__call__`은 `classifier_sidecar is None`이면
CNN을 **호출하지 않고** `evaluated=False`, `probability=0.0`, `reward_model_id=""`로
빠진다. 서버 로그가 그것을 명시적으로 말했다:

```
[reward-classifier] WARNING: 100 consecutive transitions carried no classifier sidecar;
  no reward can be produced while this continues
[reward-classifier] WARNING: session '...-session-99' finalized 1 transitions and
  classified NONE of them; every reward in it is 0 by default, not by verdict
```

**증명된 것:** classifier checkpoint가 이 서버에서 로드되고 directory SHA
`512b6575…`가 검증됐다 · `reward_model_id`가 handshake에 광고되고 액터가 pin했다 ·
서버 권위 reward finalize 경로가 200 transition 전부에 대해 돌았고 `masks`/`dones`
일관성 검사를 통과했다.

**미검증(UNVERIFIED):** 이 링크 위에서의 **per-transition classifier 추론** — 지연,
확률값, verdict. 이건 sidecar를 보내는 실기 액터에서만 나온다.
🪤 도구의 docstring "finalized by the server classifier"는 **finalize 경로**를 말하는
것이지 CNN이 매 transition 돌았다는 뜻이 아니다. 읽을 때 헷갈리지 말 것.

---

## 5. 정리 (cleanup) — 실측 확인

| 대상 | 상태 |
| --- | --- |
| synthetic learner 프로세스 | **없음.** 두 프로세스 모두 bounded `--target-learner-step`에 도달해 **스스로 종료**했다 (`exit_code=0`). **신호를 보낸 적이 없다 — TERM도, `-9`도 아니다** |
| 서버 port 50053 | **FREE** (`ss -ltn`에 없음) |
| 서버 GPU 0 | **17 MiB / 0 %**, compute app 0개 — 시험 전 상태로 복귀 |
| laptop3 SSH 터널 | 종료 (PID 326262 dead), laptop3 port 50053 listener **0개** |

### 디스크에 남긴 것 (전부 `~/hil-serl-data/synthetic-e2e/`, 합계 632 MB)

```
fake_canonical_demo.pkl        388 KB   synthetic 마커가 붙은 수락 전용 demo (388 KB라 삭제 안 함)
launch_synthetic_learner.sh    4 KB     이 시험의 기동 명령 원본
run/logs/                      64 KB    learner.jsonl · memory-preflight.jsonl · stdout-{fresh,resume}.log  ← 증거
run/checkpoints/               611 MB   checkpoint_000000000001, checkpoint_000000000002
run/assets/                    21 MB    resnet10 캐시
run/wandb/                     192 KB   offline
```

🗑️ **611 MB의 checkpoint 2개는 언제든 지워도 된다.** fingerprint와 model scope가
synthetic이라 **운영 lineage에 절대 쓸 수 없다.** 디스크 여유가 594 GB라 급하지 않아
증거로 남겼다.

**건드리지 않은 것:** `~/hil-serl-data/demos/` · `~/hil-serl-data/classifier_ckpt/`
(mtime 13:28 그대로, 시험은 14:37 시작) · `~/hil-serl-data/runs/`(burn-in 2개뿐,
이 시험은 하나도 넣지 않았다) · `/home/junhyeong/gello_software`(다른 사람 트리,
`.git/HEAD` mtime 2026-07-24 그대로) · kanu(접속조차 안 했다) · conda env 일절 무변경.

---

## 6. 실기 세션에 옮겨 갈 사항

1. **`numpy 2.2.6` vs `requirements-grpc.lock`의 `1.26.4` 차이는 문제가 되지 않았다.**
   액터 venv에서 200 transition 전송·직렬화·검증이 전부 통과했다. **실측이다.**
2. **`--synthetic-e2e` learner는 `run_hil_server.sh`가 절대 채택하지 않는다.**
   실제로 운영자가 그 사이 `./run_hil_server.sh`를 눌러
   `PID ... is not a fresh production learner: --synthetic-e2e`로 거부당했다 —
   **가드가 설계대로 동작한 것이다.** 다만 port 50053이 하나뿐이라
   **수락 시험과 실기 세션은 동시에 못 돈다.** 시험을 돌릴 거면 실기 앞뒤로 돌린다.
3. Blackwell sm_120에서 `--require-jax-backend gpu`가 통과했고 CTA update가
   실제 GPU에서 유한한 loss로 돌았다. warm-up 44 s 뒤 첫 실 update는 2.78 s였다.
4. 남은 미검증: **classifier 추론(§4)**, **실기 루프 주기(§3.1)**, 그리고 이 시험은
   개입(`intervened=1`) transition을 **한 건도** 보내지 않았으므로 **intervention
   버퍼 ingress와 RLPD 50:50의 개입 쪽 경로는 이 run으로 증명되지 않았다.**
