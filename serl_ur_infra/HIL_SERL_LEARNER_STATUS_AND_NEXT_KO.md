# HIL-SERL production learner 현황과 다음 작업

> # 🔴 2026-07-31 서버 이전 — 이 문서의 "Kanu"는 **거의 전부 역사다**
>
> learner가 **`kanu` → `junhyeong_ai`** 로 옮겨졌고 실기에서 동작이 확인됐다.
> 현행 호스트·경로·GPU·파이썬 환경의 정본은
> [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md),
> 새 링크의 실통신 수락 시험은
> [`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md)다.
>
> 🛑 **본문의 Kanu 수치는 하나도 지우지 않았다 — 지우면 안 되기 때문이다.** 여기 적힌
> fingerprint·RTT·run root·PID·GPU 표는 **kanu에서 실제로 잰 값**이고, 새 호스트 수치는
> 정확히 그것과 대조해야 의미가 생긴다. §5.5의 fingerprint `defda67b…`와 §5.6의
> RTT `84.89 / 372.82 ms`가 오늘 새 호스트 수치(§5.8)의 **유일한 비교 기준선**이다.
> **날짜가 붙은 블록은 그 날짜의 kanu 관측으로 읽어라.**
>
> | | kanu (~2026-07-30) | **junhyeong_ai (2026-07-31~)** |
> | --- | --- | --- |
> | ssh 별칭 | `kanu` | **`junhyeong_ai`** (166.104.146.29, hostname `junhyeong`) |
> | GPU | RTX A4000 ×8 (sm_86), learner는 **GPU 5** | **RTX 5070 Ti ×1 16 GB** (sm_120 Blackwell), **GPU 0** |
> | RAM / disk | 251 GB 중 여유 118~152 GB(조회 시점마다 다름) / 여유 73~90 GB (**95~96% 사용**) | **60 GB 중 약 37 GB** / **약 594 GB 여유** |
> | HIL checkout | `~/gello_software_hil_current` → `…_schema3_stage_20260730` | **`/home/junhyeong/gello_software_runtime`** (독립 clone, worktree 아님) |
> | 데이터·모델 | demo/classifier/runs/lock이 서로 다른 네 곳 | **`~/hil-serl-data/{demos,classifier_ckpt,datasets,runs,archive}` 한 뿌리** |
> | 터널 | `127.0.0.1:50153 → kanu:50053` | **`127.0.0.1:50153 → junhyeong_ai:50053`** |
> | wandb run 이름 접미사 | `-kanu-5000` | **`-hil-5000`** |
>
> `run_hil_server.sh`는 **환경변수 0개**로 새 서버를 쓴다(`5594d0e`). 옛 이름
> `HIL_KANU_REPO` / `HIL_KANU_PYTHON`은 **alias로 살아 있다**
> (`${HIL_REMOTE_REPO:-${HIL_KANU_REPO:-…}}`) — 새 이름은 `HIL_REMOTE_REPO` /
> `HIL_REMOTE_PYTHON`이고 `HIL_REMOTE_DATA_ROOT`가 신설됐다.
> ⚠️ **그 스크립트로는 이제 kanu를 몰 수 없다 — 의도된 결과다.** kanu의 classifier가
> `hil-serl-data` 밖(`workspace/youngwoong/…`)에 있어 **어떤 단일 `HIL_REMOTE_DATA_ROOT`도
> kanu를 만족시키지 못한다.** kanu는 `ssh kanu 'ps -p <pid> -o pid,etime'` 같은
> **읽기 전용** 조회로만 본다. kanu에서는 파일 하나도 지우거나 옮기지 않았다.
>
> 🟡 **jax 핀은 그대로 간다.** sm_120에서 XLA가 네이티브 `.target sm_120a`를 낸다(PTX
> 폴백 아님) — 즉 §4.9/§5의 `jax 0.5.3 / flax 0.10.5 / distrax 0.1.5 / tfp 0.25.0 /
> wandb 0.26.0` fail-closed 계약은 이전으로 **바뀌지 않았다.**
>
> 🔴 **kanu에서 잃은 학습 결과는 없다.** kanu run root 8개 전부 `checkpoints/`가
> 비어 있었다 — `checkpoint_period`가 5,000인데 최고 도달 learner step이 **301**이다.
> 새 서버는 같은 canonical demo 2,037개로 **깨끗한 새 lineage**를 시작한다.
>
> 새 호스트의 첫 실통신 재현은 **§5.8**이다.

> 🚩 **이어서 작업하러 왔다면 [HANDOFF_NEXT_SESSION_KO.md](./HANDOFF_NEXT_SESSION_KO.md)를 먼저 읽어라.** 이 문서는 전체 상태 기록이고, 그쪽이 "지금 무엇을 하면 되는가"다.
>
> 현행 기준일: **2026-07-30 19:41 KST** (아래 07-27~29 블록은 누적 측정 기록) — **단 호스트만 2026-07-31에 `junhyeong_ai`로 바뀌었다**(위 🔴 블록, §5.8)
>
> 최종 운영 checkout: `/home/laptop3/gello_software` · branch `feat/gello-ur7e-humble-22.04` (= `origin/HEAD`)
>
> **actor/하드웨어 branch는 머지됐다.** `test/hil-hardware-comms` 는 2026-07-29 merge `3f199d4`(부모 `3ff5f80` + `1a4f93d`)로 canonical branch에 들어왔다. `43ba314`와 아래 commit 표는 역사적 통합 기준점이다. **현행 계약은 특정 옛 HEAD가 아니라 이 문서 바로 아래의 2026-07-30 정본과 현재 코드로 판정한다.**
>
> 이전 learner/hardware 통합 merge: `248255f` (2026-07-27, schema v2). `5fb716b..3f199d4` 는 23 commit, `5fb716b..43ba314` 누적 diff는 70 files / +9,903 −395.
>
> worktree `/home/laptop3/gello_worktrees/hil-hardware-comms` 는 아직 디스크에 있고 `test/hil-hardware-comms` @ `1a4f93d` 를 가리키지만 **더 이상 작업 위치가 아니다.** 새 작업은 canonical checkout에서 한다.
>
> actor/하드웨어는 §11, **reward classifier 조사는 §12**
>
> learner 실행 절차: [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) — ⚠️ **파일명도 본문도 kanu 기준이다.** 호스트·GPU·경로는 [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md)가 우선한다
>
> reward threshold 근거와 classifier 실측: [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)
>
> 하드웨어/통신 테스트 절차: [../docs/testing/README.md](../docs/testing/README.md)

### 2026-07-30 현행 정본 — 이 블록이 아래 누적 기록보다 우선한다

현재 transport와 성공 판정 계약은 다음과 같다.

| 항목 | 현행 값 |
| --- | --- |
| gRPC protocol | **`2`** (`ur_env.actor_network.PROTOCOL_VERSION`) |
| transition/data schema | **`3`** (`SCHEMA_VERSION`); `Meta.auto_success`와 `Meta.operator_success`가 추가됨 |
| canonical observation schema | **v2**, hash `3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903` — transition schema 3과 별개이며 바뀌지 않음 |
| production reward threshold | **`0.5`** (`DEFAULT_REWARD_THRESHOLD`와 `run_hil_server.sh`가 같은 값을 pin) |
| GUI success mode | **MANUAL 기본** (`auto_success=false`) |
| 최종 success | `operator_success OR (auto_success AND classifier_success)`; 두 플래그를 동시에 true로 보내면 protocol error |

**MANUAL은 classifier OFF가 아니다.** sidecar가 붙은 transition은 MANUAL/AUTO와 무관하게
서버 classifier를 실행한다. 서버 replay provenance에는 `classifier_evaluated`, probability,
threshold, raw `classifier_success`, mode flags가 저장된다. actor의 로컬 transition에도
evaluated/probability/threshold/reward model/effective success가 반영되고 GUI에는
**LAST CLASSIFIER**로 표시된다. 차이는 classifier verdict가 episode를 끝낼 권한뿐이다.

- **MANUAL:** classifier는 계속 실행·표시·저장되지만 자동 success로 승격되지 않는다.
  조작자가 GUI의 **MARK SUCCESS**를 누르면 현재 `(run_id, episode_id)`에 묶인 one-shot
  `operator_success=true`가 다음 transition에 정확히 한 번 찍히고 reward 1 / terminal이 된다.
- **AUTO:** GUI에서 명시적으로 전환했을 때만 `auto_success=true`가 transition마다 찍힌다.
  이때 `classifier_probability > threshold`인 server verdict가 reward 1 / terminal을 만들며,
  수동 success 버튼은 비활성이다.
- sidecar가 없는 transition은 classifier 미평가로 명시된다. MANUAL의 operator success는
  그래도 유효하고, AUTO에는 성공 근거가 없으므로 zero-reward ordinary sample이다.

**🗄️ kanu 실측 스냅샷 (2026-07-30 19:41 KST) — 현행 호스트의 상태가 아니다.**
그날 laptop3에서 `./run_hil_server.sh --check`와 kanu JSONL을 **읽기 전용**으로 확인한 값이다.
2026-07-31 이후 learner는 `junhyeong_ai`에서 돌고, **아래 PID·GPU 5·run root·checkout은
전부 kanu의 것**이다. 지금 상태를 알고 싶으면 아래 표를 읽지 말고 `./run_hil_server.sh --check`를
다시 돌려라 — 그 스크립트는 이제 `junhyeong_ai`를 조회한다. 이 표를 남기는 이유는 그것이
**새 lineage가 비교당할 kanu 쪽 마지막 관측**이기 때문이다.

| 항목 | 확인값 |
| --- | --- |
| process / GPU / health | PID `1112465` / physical GPU `5` / gRPC `ready` |
| deployment checkout | `/home/junhyeong/gello_software_hil_current` → `/home/junhyeong/gello_software_hil_schema3_stage_20260730`, clean, `c9c30c3` |
| run root | `/home/junhyeong/hil-serl-data/runs/cube_in_cup_manual_schema3_thr05_20260730_1715` |
| server contract | protocol `2`, schema `3`, production model ID, reward authority `server_classifier`, threshold `0.5` |
| buffer status | replay `400 / 50,000`, intervention `225 / 10,000`, overwrite `0 / 0`; last env step `99` |
| learner status | learner step `301`, gradient step `602`, policy version `6`; fault event 없음, target step `5,000` |
| startup lineage | `restored_checkpoint=null`, learner/gradient/policy `0/0/0`에서 시작; JAX backend `gpu` |
| offline demo | **2,037 transition 보존·로드**, synthetic demo `0`, SHA `f9718558…032fa` |

이 lineage는 이전 실기/RAM 시험의 online replay를 복원하지 않았다. replay/intervention은
원래 process-local RAM이므로 옛 process 종료와 함께 **의도적으로 폐기**됐고, 현재 `400/225`는
새 schema-3 lineage에 다시 들어온 값이다. 반대로 사람 승인 offline demo pickle 2,037개는
영구 artifact이므로 그대로 보존되어 startup마다 다시 로드된다. startup RAM gate도
forecast `available=138,997,297,152 B`, `required=10,290,708,416 B`로 accepted였고 그 결정은
run root의 `logs/memory-preflight.jsonl`에 남아 있다.

정상 3-CLI 운용에서 **server terminal** 명령은 아래 하나다. raw learner CLI를 손으로 다시
조립하지 않는다. wrapper가 exact artifact/SHA/threshold/process 계약을 검증하고, healthy
learner가 있으면 재사용한 뒤 laptop `127.0.0.1:50153` tunnel만 소유한다.
**(2026-07-31~) 이 명령은 환경변수 0개로 `junhyeong_ai`를 조회한다** — 호스트를 지정하는
override를 붙이지 않는다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh
```

상태만 보고 learner나 tunnel을 만들지 않으려면:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh --check
```

### 📖 이 문서를 읽는 법

이 문서는 **누적 기록**이다. 지운 서술이 거의 없고, 대신 뒤집힌 서술에는 무엇이 그것을 대체했는지를 붙였다.

- 절 제목이나 문단 머리에 **날짜가 붙어 있으면 그 날짜의 관측**이다. 현재 상태가 아니다.
- `⛔ 폐기` / `⚠️ 대체됨` 표시가 붙은 블록은 **왜 틀렸는지를 남기려고 보존한 것**이다. 인용하지 마라.
- **숫자를 인용할 때는 옆에 붙은 조건(split·전처리·날짜·하드웨어)을 함께 옮겨라.** 조건 없는 숫자 하나가 이미 한 번 잘못된 결론을 만들었다(§12.3).
- **그 "하드웨어"에는 이제 호스트도 포함된다.** 이 문서의 시간·GPU·경로 수치는 **거의 전부 kanu(2026-07-30까지)** 에서 잰 것이다. 새 호스트 `junhyeong_ai` 값으로 **덮어쓰지 말고 옆에 붙여라** — 그래야 이전 효과를 읽을 수 있다. 새 호스트에서 실제로 잰 것은 **§5.8 하나뿐**이다.
- **"실기 검증"이라고 쓰인 것은 어느 코드 경로였는지 확인하라.** `run_real_hil.py`(수동 HIL 스크립트)와 `run_remote_rlpd_actor.py`(production actor)는 다른 경로다. 후자도 2026-07-29 첫 실물 E2E를 통과했으므로, 아래의 "아직 한 번도 돌지 않았다"는 문구는 07-29 이전 기록이다.

## 한눈에 보기

> **(2026-07-29) 이 절은 §11/§12와 정합을 맞춘 판이다.** 각 항목 끝의 `§` 가 근거 절이다.

- **actor 브랜치 머지가 끝났다** (`3f199d4`). 이 문서에서 "머지하면 …가 된다"로 쓰여 있던 예측은 전부 **일어난 일**로 바뀌었다. §12의 크롭 불일치도 그렇게 canonical branch에 실재하는 결함이 됐고, **2026-07-29에 sidecar 분리로 해소됐다**(재학습이 아니다). §12.1
- receive server, 실제 hybrid SAC learner, versioned policy, strict replay ingress, gripper penalty, checkpoint/resume, JSONL/W&B를 **하나의 production CLI**로 조립한다. 현행 구분은 **transport protocol 2 / transition schema 3 / canonical observation schema v2**다. 셋을 모두 "schema v2"라고 부르지 않는다.
- robot actor의 실제 실행 action을 기준으로 `grasp_penalty`를 생성하는 wrapper도 두 actor entrypoint에 배선됐다. learner ingress는 penalty 누락을 허용하지 않는다.
- 실제 `SACAgentHybridSingleArm`을 사용해 CTA update → publish → checkpoint → fresh agent restore → production composition 재조립 → action/RNG/counter 확인 → 추가 update/checkpoint까지 검증했다.
- 실제 reward classifier checkpoint(`e329986b...`, Jul-24 단일 파일 flax msgpack 87 MB)는 로컬에서 **SHA 검증, load, warm-up까지만** 성공했다. 이것은 artifact I/O 검증이지 분류 성능 검증이 아니며, **분류 성능은 당시 검증하지 않았다.** 이후 **2026-07-28 Kanu GPU 실측(0724 도메인 success 프레임 1,123장, 크롭 없는 입력, threshold 0.85)** 에서 이 checkpoint의 success recall이 `0.0%`(1,123장 중 0건, mean 확률 `0.007`)로 확인돼 **폐기 대상**이 됐다. ※ 같은 checkpoint가 **0720 test split**(무크롭)에서는 recall `93.4%` @0.85 / FPR `0.0%` 다(§12.3) — 도메인이 다르면 숫자가 이렇게 갈린다. **"recall 0%"를 조건 없이 인용하지 마라.** 그대로 실기에 물리면 로봇이 성공해도 reward가 영원히 0이고 학습이 시작되지 않는다. 근거는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md), 새 정본 경로는 [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) 1.3에 있다. annotation-only TensorFlow shim 때문에 Flax가 잘못된 TensorFlow I/O backend를 고르던 문제는 infra-owned local-I/O 설정으로 수정했고, 이 수정 자체는 계속 유효하다.
- fake canonical demo generator가 추가됐다. 이 raw artifact는 construction `--dry-run` 또는 명시적으로 bounded된 `--synthetic-e2e` acceptance에만 허용된다. 일반 robot-data learner serving은 계속 거부한다.
- **테스트 (📌 2026-07-30 실측, `gello-hil-actor` = numpy 2.2.6):** `serl_ur_infra/tests` = **`595 passed, 11 skipped, 1 xfailed`** (**다른 세션 작업 2파일 제외**). 명령은 §9. ⚠️ **인터프리터와 범위를 안 적은 passed 수는 무의미하다** — 트리 전체는 **`605 / 11 / 1`**, `hilserl`(numpy 1.26.4, jax) 트리 전체는 **`645 / 4 / 1`**다. ⚠️ **`1 xfailed`는 `08` G27의 알려진 결함 표식이므로 반드시 같이 적는다.**
  - 🚧 **같은 checkout에서 다른 세션이 동시에 작업 중이다** (P3 operator 상태 기계: `operator_session.py`, `hil_actor_status.py`, GUI/런처). 그래서 **트리 전체 개수에는 우리 것이 아닌 테스트가 섞이고 계속 움직인다.** 우리 기준선은 그 2파일(`test_operator_session.py`, `test_remote_actor_operator_session.py`)을 `--ignore`한 **595**다. 커밋 `4197f5b` 시점 값은 **`579 / 11`**이었다. `gello-hil-actor` 에서 **skipped는 정확히 `11`** 이고 변하지 않았다.
  - 계보(전부 `gello-hil-actor`): `43ba314` `333` → `40b99f8`(recorder take 변환) `337` → classifier sidecar `429` → 07-30 오전 `497` → `4197f5b`(30 Hz 서브스텝, +82 = leader_stream 28 / governor_dt 38 / intervention_substeps 16) `579` → 📌 07-30 `595`. **`333`·`429`·`588` 을 현행 기준선으로 인용하지 마라.**
  - 🪤 `third_party/hil-serl/serl_launcher` 를 `PYTHONPATH` 에서 빼면 passed 수가 **조용히 떨어지고**(`43ba314` 기준으로는 `300 passed, 13 skipped`였다) skip 사유가 "submodule is not checked out"이라고 거짓말한다. **기준선보다 낮은 passed 수 = `serl_launcher` 가 `PYTHONPATH` 에서 빠진 것**이고, 이때 skipped도 11을 벗어난다. **녹색이 아니라 passed 수를 볼 것.**
  - UR/GELLO `ur_gello_bringup` = **`436 passed`** (2026-07-29 `43ba314` 측정, 이번 변경은 이 suite를 건드리지 않아 재실행하지 않았다).
  - *(2026-07-27 `248255f` 시점의 역사값: `253 passed, 4 skipped, 6 warnings` / `436 passed`. 실제 frozen-trunk agent `2 passed`, checkpoint/resume `1 passed`, local fake E2E `1 passed`. §5.1)*
- 사용자가 현재 milestone 완료 조건으로 지정한 **fake data laptop→SSH tunnel→Kanu learning E2E**는 schema v2에서 exact 100 ingress→actual classifier→feature replay→CTA update→publish→checkpoint full-load roundtrip까지 통과했다. 새 Kanu process가 checkpoint를 `1/2/1`로 restore하고 policy version 1의 finite 7D action을 serving하는 것도 확인했다. 사용자 요청에 따라 v2 resume process의 불필요한 두 번째 SAC update는 생략했다.
- 실제 robot/task/camera의 짧은 E2E와 production 50-step publish는 확인됐다. 아직 **5,000-step checkpoint bounded run**, 장시간 GPU contention/latency와 운영 heartbeat는 미완료라 continuous 장기 운용 승인은 별도다.
- external policy/classifier는 canonical raw `uint8 (1,128,128,3)` image를 계속 받지만, replay/demo에는 frozen ResNet-10의 `stop_gradient` 직후 camera당 `float32 (1,4,4,512)` map의 current/next만 저장한다. GAP은 적용하지 않고 augmentation은 `none`이다.
- `SpatialLearnedEmbeddings(8) -> Dropout(0.1) -> Dense(256) -> LayerNorm -> tanh`는 동결하지 않았다. learner가 feature batch를 꺼낼 때 현재 weight로 적용하므로 critic/grasp critic CTA update가 유지된다.
- checkpoint만 영속화되고 replay/intervention buffer는 RAM-only다. checkpoint는 덮어쓰기·삭제·자동 pruning을 하지 않는다.
- **(2026-07-27 추가 · 2026-07-29 정정)** robot actor 쪽은 `5fb716b` 시점에 **실행 자체가 불가능**했다. 4개 층이 동시에 막고 있었고(§11.1) 전부 해소했다. 이제 actor가 기동해서 Kanu까지 왕복한다.
  - ⚠️ 07-27 판본은 여기에 *"workspace box와 reset branch-cut 두 안전 게이트가 실제로 동작한다"* 를 덧붙였는데, **"동작한다"는 단위 테스트 기준이다.** 두 게이트는 2026-07-29 현재도 **실기에서 한 번도 작동한 적이 없다** — 팔이 움직인 07-28 세션은 박스가 꺼진 `run_real_hil.py` 경로였다(§11.7-3).
- **(2026-07-27 실기 측정)** 검증 완료: 2F-85 gripper 개폐 방향, GELLO leader 7 모터, laptop→Kanu 100-step gRPC 왕복(`replay_insert_count:100`, schema hash 일치). 왕복 지연은 **WiFi·fake-env actor·zero-action receive server 조건**에서 RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**, step당 96.1 KiB → 7.9 Mbit/s(§11.4).
  - ⚠️ 같은 항목의 "미검증: 팔 실제 구동, 카메라(하드웨어 고장), 개입 루프 실기"는 **2026-07-28에 셋 다 뒤집혔다** — 팔은 움직였고 개입 루프도 64 스텝 돌았으며(§11.9), 카메라는 허브 고장이 아니었다(§11.10). **단 그 세션은 `run_real_hil.py` 경로다.**
- **(2026-07-30 갱신 · 🗄️ 2026-07-31 서버 이전으로 호스트 무효)** ~~Kanu에는 실제 policy/learner server가 떠 있다.~~ **그날 kanu에 떠 있었다는 기록으로 읽어라.** PID/run/health/buffer 스냅샷은 문서 최상단과 §11.5에 kanu 값으로 보존돼 있다. 사용자가 `take_23` 제외 23개를 success로 승인해 만든 2,037-transition 영구 demo artifact는 **이전 대상이었고 새 서버 `~/hil-serl-data/demos/`에 같은 SHA로 존재한다**(laptop3·kanu·junhyeong_ai 3벌 동일 해시).
  - ⛔ **폐기된 07-28~29 스냅샷:** 옛 zero-action PID 1096786이 종료된 뒤 잠시 HIL process가 0개였다. **2026-07-30에는 production learner가 healthy**이므로 이 문장을 현재 상태로 인용하지 않는다(§11.5, §12.5).
  - ⛔ **폐기:** `/home/laptop3/gello_software` 는 **GPU 서버에 존재하지 않는다.** kanu 시절 실제 경로는 §12.5 표, 현행 `junhyeong_ai` 경로는 [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md) §2(코드는 `/home/junhyeong/gello_software_runtime`).
- **(2026-07-28 실기 측정)** 실기에서 **팔이 처음으로 움직였다** — `run_real_hil.py --arm --scale 0.25`, 100 스텝 중 개입 64, `held=0`, 개입 불변식 4종 통과. 같은 세션에서 **frame-map 측정을 마쳤다**: 좌표계 매핑은 **단위행렬**이고 부호 뒤집힘도 축 교환도 없다(포화 표본 제외 잔차 0.093, 기준 0.15; §11.9). 속도 3층은 검증된 EEF 텔레옵 값에 균일 1.25배로 정렬했다(`a5f9890`).
- **(2026-07-28 추가)** actor entrypoint의 4개 결함을 고쳤다(`607e541`, `c069e79`): `--deadman {topic,spacebar}` 배선(이전에는 전역 스페이스바/ESC로 조용히 fallback), `--arm`(`DRY_RUN` CLI 해제), `--mock-policy-noise`(zero-action 서버 상대로 로봇을 움직여 개입 경로를 실증), 카메라 첫 프레임 대기, `--arm` 시 이중 퍼블리셔 거부 게이트.
  - ⛔ **07-29 오전 기록:** 당시에는 "실기 투입 가능" 단계였지만 같은 날 이후 `run_remote_rlpd_actor.py`의 첫 실물 E2E가 완료됐다(§1 판정표).
- **(2026-07-28 발견 · 2026-07-29 머지로 현실화 🔴 핵심)** **reward classifier가 학습 때와 다른 이미지를 받는다.** 학습은 크롭 없이 full-frame 1280×720을 128×128로 찌그러뜨렸는데, actor는 `IMAGE_CROP`을 적용한 뒤 리사이즈한다. 픽셀 대조(**MAE 0.00**(무크롭 가설, 비트 일치) **vs 21–35**(크롭))와 실제 체크포인트 실행(**성공 프레임 36장에서 recall@0.85 100.0% → 33.3%**)으로 증명했다.
  - 07-28 판본은 "현재 프로덕션은 안 망가져 있다 — 크롭을 켜는 task config가 kanu 체크아웃 브랜치에 없다. **actor 브랜치를 머지하는 순간 유입된다**"였다. **그 머지가 `3f199d4` 로 일어났다.** `serl_ur_infra/ur_experiments/cube_in_cup.py` 는 이제 canonical checkout에 존재하고 `IMAGE_CROP` 이 채워져 있다. **따라서 이것은 예측이 아니라 canonical branch의 실재 결함이었다.**
  - ✅ **(2026-07-29 해소) — 재학습이 아니라 *분리*로 고쳤다.** actor가 관측에 **무크롭 128×128 JPEG "sidecar"** 를 붙여 보내고, 서버는 **그것만** 분류한다. 정책은 측정된 `IMAGE_CROP` 을 **그대로** 유지한다. 하나의 이미지가 두 소비자를 섬기던 것을 그만둔 것이다. §12.1 / §12.7
    - **⚠️ 이 항목의 07-28~07-29 판본이 권고하던 "크롭에 맞춘 classifier 재학습"은 채택되지 않았다.** 아래 「⛔ 폐기」 표시가 붙은 서술을 근거로 재학습 작업을 시작하지 마라.
    - **그래서 §12.3의 held-out 수치 전부가 그대로 살아 있다.** 분류기는 여전히 **무크롭** 프레임을 먹으므로 측정 조건이 바뀌지 않았다. 재학습을 택했다면 threshold 결정에 쓰인 숫자를 **전부 다시 재야** 했다([REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)의 sweep 전체 포함). 이것이 분리 방식의 가장 큰 실질 이득이다.
    - **당시** `DEFAULT_REWARD_THRESHOLD` 는 0.2였다. **현행 production 값은 0.5**이며 문서 최상단과 §4.7이 우선한다.
    - **고쳐지지 않은 것: 팔 가림(occlusion) 병리.** `take_21` 은 @0.85 recall `0.0%`, @0.05 로 내려도 `57.9%` 다(§12.3 · 출처 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md) `:369`). 팔이 cam1 시야를 쓸고 지나가면 확률이 `0.005 ↔ 1.0` 으로 진동한다. **원인은 전처리도 라벨도 아니고 시야/가림이다.** sidecar의 정지 게이트가 완화할 뿐이고, 진짜 해법은 **팔이 가로지르지 않는 카메라 배치**다.
- **(2026-07-28 측정 · 2026-07-29 감사로 조건 명시)** Jul-27 `cube_in_cup_all3` 체크포인트(orbax 디렉터리, 43 MB)의 held-out 성능. **아래 두 줄은 같은 체크포인트·같은 threshold인데 숫자가 다르다. 분할이 다르기 때문이고 둘 다 맞다.**
  - **0720 test split만**(success n=166), threshold 0.5, **크롭 없는 입력**, 2026-07-28 Kanu GPU: recall **100.0%** / FPR **0.0%** / acc **100.0%** (§12.3)
  - **0720 held-out 전체**(test 166 + val 100 = success n=266, 6 takes), threshold 0.5, **크롭 없는 입력**: recall **86.8%** — [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md) 「처음 측정된 held-out success recall (0720)」
  - 차이는 전부 **val split에 들어 있는 병리 take `take_21_20260720_210234`**(n=38, @0.5 recall 7.9%)가 만든다. §12.3에 산술 대조가 있다.
  - **운영 기준은 보수적인 쪽(266프레임 = 86.8%)을 쓴다.** 두 수치 모두 **크롭 없는 입력**에서 쟀다. ~~크롭이 활성인 현재 actor 경로에는 그대로 적용되지 않는다.~~ → **2026-07-29 정정: 이제 그대로 적용된다.** sidecar가 분류기에 무크롭 프레임을 주므로 **측정 조건과 실행 조건이 다시 같아졌다**(§12.1).
- **(2026-07-29 역사)** `DEFAULT_REWARD_THRESHOLD` 가 0.2까지 내려왔었다(`1b02857`). **2026-07-30 현행 production은 보수적인 0.5로 복귀**했다. held-out FPR은 0.2와 0.5에서 모두 0%였고, 0.5가 hardest known negative와의 margin을 더 크게 유지한다. 어느 값이든 threshold는 learner fingerprint에 들어가므로 다른 값의 checkpoint resume은 fail-closed로 거부된다(§4.7).
- **(2026-07-29)** `launch_cameras.sh` 가 RealSense 시리얼을 상수로 믿지 않고 **실제 USB 버스에서 해석**하도록 바뀌었다(`43ba314`). 없는 시리얼 바인딩은 에러가 아니라 조용한 "프레임 없음"이라 이 방향은 옳다.
  - ✅ **해소 (2026-07-29 직접 측정).** 이 항목의 07-29 오전 판본은 *"커밋의 전제가 커널 저널과 어긋난다 — 기본 시리얼 쌍은 이 PC가 열거한 적 없는 하드웨어다"* 였다. **반증됐다. 카메라는 한 쌍뿐이고 `43ba314` 의 기본값이 옳다.** `147122072740`/`243222072700` 은 **모듈 시리얼**(= `serial_no:=` 가 매칭하는 필드), `151623020789`/`322743060038` 은 같은 두 대의 **ASIC 시리얼**(= 커널 USB 디스크립터가 노출하는 필드)이다. 🪤 **저널 grep으로 이 질문에 답하려 하지 마라** — 두 세션이 그렇게 해서 각각 정반대의 틀린 결론에 도달했다. §11.10
  - ⚠️ **2026-07-29 11:14에 두 카메라 모두 `uvcvideo … Non-zero status (-71)`(EPROTO)이 다시 났다.** §11.6에서 "허브 고장"으로 오판했던 것과 같은 증상이다. §11.10

---

## 1. 현재 판정

**판정 기준일 2026-07-30.** Kanu 수치는 19:41 KST 읽기 전용 스냅샷이다. 시간이 지나며 counter는 증가할 수 있으므로 재확인은 `ros2_ur_ws/run_hil_server.sh --check`로 한다.

> 🗄️ **아래 표의 "Kanu"가 붙은 행은 전부 kanu에서 낸 판정이다.** 2026-07-31에 learner가
> `junhyeong_ai`로 옮겨졌으므로 **"실행 중"·"배포·healthy"류의 현재형은 그 호스트에서
> 다시 세워져야 한다.** 새 호스트에서 오늘 무엇이 재현됐는지는 **§5.8**과
> [`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md)에 있다.
> 판정 자체(무엇이 구현·검증됐는가)는 호스트와 무관하므로 그대로 유효하다 — 바뀐 것은
> **어느 기계에서 도는가**뿐이다.

| 범위 | 판정 |
| --- | --- |
| learner 라이브러리 | 구현·자동 검증 |
| receive server + learner 단일 프로세스 composition | 구현·loopback 자동 검증 |
| 실제 agent checkpoint/resume/continue | opt-in 실제 agent 자동 검증 |
| actor gripper penalty wiring | 구현·자동 검증 |
| classifier 실제 checkpoint local restore | load/SHA/warm-up만 검증(2026-07-27, local CPU), 분류 성능 미검증 — 사용한 Jul-24 `e329986b...`는 이후 **0724 도메인·무크롭·@0.85 에서 recall `0.0%`** 로 폐기(§12.3) |
| **reward classifier 분류 성능 (Jul-27 `cube_in_cup_all3`)** | **측정 완료 (2026-07-28, Kanu GPU, 크롭 없는 입력)** — 0720 test split(n=166) @0.5 recall 100.0% / FPR 0.0%; 0720 held-out 전체(n=266) @0.5 recall 86.8%; §12.3 |
| **actor 경로의 classifier 입력 정합** | ✅ **정합·실물 전송 확인.** actor가 무크롭 JPEG sidecar를 별도로 보내고 서버는 그것만 분류한다. 정책의 `IMAGE_CROP`은 불변. 남은 것은 장시간 verdict/가림 분포 검증; §12.1 |
| **Jul-27 checkpoint를 gRPC 경로에 pin** | ✅ **가능 (2026-07-29, 코드 검증).** G19 해소 — `checkpoint_sha256()` 이 재귀 `directory_sha256()` 에 위임한다(`ur_env/classifier_sidecar.py::directory_sha256`). 기본 SHA는 `512b6575…`(= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`, 14개 파일). **이 세션에서 직접 재계산해 상수와 일치를 확인했다.** 단일 파일은 예전과 동일한 순수 content sha256이라 기존 pin도 유효; §12.6-3 |
| JSONL + 실제 W&B offline artifact | 자동 검증 |
| fake canonical demo | 생성기·strict loader·dry-run/synthetic-E2E scope gate 자동 검증 |
| Kanu GPU production dry-run | 통과 (2026-07-27) — actual classifier/agent, feature demo conversion, 128/32 RAM preflight. **classifier는 이후 폐기된 Jul-24였다** |
| Kanu GPU feature CTA smoke | 통과 (2026-07-27) — 1 learner step/2 gradient step, raw/cached action, trunk invariant |
| laptop→Kanu fake learning E2E | 통과 (2026-07-27) — unified schema v2 fresh actual update/checkpoint + fresh-process resume serving |
| Kanu GPU production learner | **(2026-07-30 기준) 실행 중 / bounded acceptance 진행 중.** health `ready`, target 5,000 중 learner `301`, gradient `602`, policy version `6`. 🗄️ **그 lineage는 kanu의 것이고 checkpoint를 하나도 남기지 못한 채(5,000 step 미도달) 이전됐다.** 5,000-step checkpoint/완주와 continuous 장기 운용 승인은 **새 호스트에서 다시** 남은 일이다 |
| 실제 robot actor → learner E2E | **2026-07-29 첫 실물 smoke 통과(kanu).** 201 transition / intervention 153 / learner 102 / policy version 2. 07-30 schema-3 lineage는 replay 400 / intervention 225까지 받았다. ✅ **2026-07-31 서버 이전 뒤 `junhyeong_ai` 상대로 조작자 실기 세션 재통과** — replay 316 / intervention 210 |
| frozen-trunk feature replay/demo | 구현·자동 검증; Kanu GPU dry-run/CTA smoke 통과 |
| **robot actor 기동 (wrapper chain + task config)** | **구현·자동 검증 (`ee3240e`); §11.2** |
| **actor → Kanu 실제 gRPC 왕복 100 step** | **실기 검증 (2026-07-27, zero-action receive server 상대, fake-env actor); §11.4** |
| **workspace box / reset branch-cut 안전 게이트** | **구현·단위 검증; 실기 미검증** — 팔이 움직인 07-28 세션은 `run_real_hil.py` 경로였고 거기서는 `DefaultUR7eEnvConfig.ABS_POSE_LIMIT`이 0이라 **박스가 아예 비활성**이었다. 즉 팔은 움직였지만 이 게이트는 여전히 실기에서 한 번도 작동한 적이 없다; §11.3, §11.7 |
| **2F-85 gripper, GELLO leader 하드웨어 경로** | **실기 검증 (2026-07-27)** |
| **팔 실제 구동 (`DRY_RUN=False`)** | 2026-07-28 `run_real_hil.py` 성공에 이어 **2026-07-29 production actor entrypoint도 실물 UR7e에서 구동** |
| **actor entrypoint 실기 투입** | **통과(짧은 E2E).** controller handoff/복귀와 actual policy+GELLO transition ingress 확인. 장시간 run과 publish 순간 latency는 별도 미완료 |
| **카메라 경로** | 두 RealSense의 30 Hz preflight와 production actor observation/sidecar 전송 확인. 07-29 EPROTO 이력은 있으므로 세션마다 probe 유지; §11.10 |
| **Kanu 실제 정책 서빙 (learner server)** | **(2026-07-30) 배포·healthy였다.** PID `1112465`, GPU 5, run `cube_in_cup_manual_schema3_thr05_20260730_1715`, protocol 2/schema 3, production model/reward pins 일치; §11.5. 🗄️ **현행 서버는 `junhyeong_ai`이며 같은 pin 세트로 2026-07-31 실통신 수락 시험을 통과했다(§5.8)** |
| **canonical robot demo** | ✅ **사람 승인 영구 artifact 완료.** `take_23` 제외 23개/2,037 transitions, SHA `f9718558…032fa`, laptop3/Kanu strict-load 통과. 새 actor 녹화의 `buffer_period=0` 문제는 별개로 남는다 |

즉, fake-data milestone뿐 아니라 실제 robot actor→Kanu policy/classifier/replay/learner의 짧은 E2E와 현행 protocol-2/schema-3 production serving까지 확인했다. 남은 큰 경계는 target 5,000 완주/checkpoint, 장시간 latency·자원 경합, 연속 운용이다.

다만 위 판정표에서 `classifier 실제 checkpoint local restore` 항목은 **artifact I/O 판정일 뿐 reward 품질 판정이 아니다.** 위 모든 fake/synthetic 통과 결과는 `e329986b...` checkpoint로 얻은 것이고, 그 checkpoint는 2026-07-28 실측에서 0724 도메인 recall `0.0%`(무크롭, @0.85)로 폐기됐다. synthetic/fake acceptance가 검증한 범위는 파이프라인 배선(load, warm-up, ingress, CTA, publish, checkpoint)이지 reward 품질이 아니므로 **위 통과 기록 자체는 그대로 유효하다.** 그러나 **실기 run에는 새 정본 classifier가 필요하다**(§4.8, §7 P0-0).

robot actor 쪽은 "실행 불가"에서 시작해 팔 실구동·카메라 복구·production actor 첫 E2E까지 마쳤다. 아래 07-29 목록 중 1번과 4번의 "미투입/미배포"는 완료됐으며, 현행 남은 경계는 장시간 운용과 계측이다.

1. ✅ **actor entrypoint(`run_remote_rlpd_actor.py`)의 실기 첫 투입 완료** — 2026-07-29 짧은 E2E. 다음은 장시간 run과 publish 순간 RPC latency 검증이다.
2. ~~**크롭에 맞춘 classifier 재학습**~~ → **⛔ 취소. 2026-07-29에 sidecar 분리로 해소했고 재학습은 하지 않는다**(§12.1, §12.7). 대신 남은 것은 **`stationary_speed_max` 실측**이다 — `cube_in_cup.py::CLASSIFIER_SIDECAR` 의 `0.05 m/s` 는 코드 주석이 스스로 `PLACEHOLDER` 라고 밝혀 둔 값이고, 녹화 take에서 실제로 재야 한다.
3. ✅ **canonical demo artifact 생성 완료** — 23개/2,037 transitions 사람 success 승인, laptop3/Kanu strict-load 통과. 현행 run은 online replay 400으로 `training_starts=100` gate도 이미 넘었다.
4. ✅ **진짜 정책 서버 배포 완료**(2026-07-30 kanu; §11.5). 새 schema-3 / threshold-0.5 lineage를 `restored_checkpoint=null`로 시작했으며 옛 RAM replay는 가져오지 않았다. 🗄️ **2026-07-31에 그 배포가 `junhyeong_ai`로 옮겨졌고, kanu lineage는 checkpoint를 남기지 못한 채(최고 step 301 / period 5,000) 끝났다.** target 5,000 checkpoint 완주는 새 호스트에서 남은 일이다.
5. **팔 가림에 강한 카메라 배치** — sidecar가 못 고치는 유일한 항목이다(`take_21` 계열). §12.8-3.

## 2. 작업 위치와 branch

### 2.1 최종 운영 위치와 통합 기준점

| 용도 | 위치 | branch/commit | 상태 |
| --- | --- | --- | --- |
| canonical 운영 checkout | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` (= `origin/HEAD`) | 최종 server/learner/actor/local-hardware 통합 위치 |
| learner/hardware 통합 기준점 | 위 canonical branch에 포함 | `248255f` (2026-07-27) | learner `8f242d8` 계열과 hardware `6a0b127`을 병합; schema v2 검증 완료 |
| **robot actor 통합 merge** | 위 canonical branch에 포함 | **`3f199d4` (2026-07-29)** | 부모 `3ff5f80` + `1a4f93d`. `test/hil-hardware-comms` 를 흡수. `5fb716b..3f199d4` = 23 commit |
| reward threshold 인하 (역사) | 위 canonical branch에 포함 | `1b02857` (2026-07-29) | 당시 `DEFAULT_REWARD_THRESHOLD` 0.5 → 0.2; **현행값 아님** |
| **기준 HEAD** | 위 canonical branch | **`43ba314` (2026-07-29)** | RealSense 시리얼을 실제 USB 버스에서 해석 (§11.10) |
| 🗄️ 구 Kanu deployment (은퇴) | kanu `~/gello_software_hil_current` symlink → `…_schema3_stage_20260730` | **`c9c30c3` (2026-07-30)** | protocol 2 / transition schema 3 / threshold 0.5 production learner, clean checkout. **2026-07-31 이전으로 더 이상 운영 경로가 아니다** |
| **현행 server deployment** | `junhyeong_ai:/home/junhyeong/gello_software_runtime` | `feat/gello-ur7e-humble-22.04`, laptop3·GitHub tip과 같은 커밋 유지 | **독립 clone(worktree 아님).** 커밋 id는 스냅샷이므로 pin하지 않는다 — `git rev-parse HEAD`로 읽는다. 데이터·모델은 `~/hil-serl-data/` |
| ⚠️ 구 actor worktree | `/home/laptop3/gello_worktrees/hil-hardware-comms` | `test/hil-hardware-comms` @ `1a4f93d` | **더 이상 작업 위치가 아니다.** 머지 완료. 디스크에 남아 있을 뿐 |

통합 lineage는 historical base `f0dd3e7` 위의 learner snapshot `8f242d8`, synthetic acceptance 문서 `5322119`, hardware contract fix `6a0b127`을 `248255f`에서 합치고, 그 위에 actor/하드웨어 계열 23 commit을 `3f199d4`에서 합쳤다. 최종 작업 branch는 `feat/gello-ur7e-humble-22.04` 하나다.

`5fb716b..43ba314` 누적 diff: **70 files changed, +9,903 −395.**

### 2.2 역사적 통합 이력

아래는 현재 작업 위치가 아니라 이전 milestone의 **역사적 기준점**이다.

```text
dc25cbe  robot-local intervention metadata
  -> 2b50d34  local RLPD actor adapter
  -> ec91bef..5709bb5  gRPC actor transport + SSH smoke
  -> 9cc994f..e42dbf3  server-authoritative receive server
  -> c67c327  local hybrid SAC learner foundation
  -> a01b665  real-agent checkpoint integration
  -> e5ec01b / f0dd3e7  canonical handoff/status
```

친구의 remote-inference split WIP은 `be0ffdc`이고 `archive/hil-grpc-actor-transport-wip-20260727` tag로 보존돼 있다. 현재 production learner lineage에 병합하지 않았다.

### 2.3 수정 금지 경계

- `third_party/hil-serl` submodule은 직접 수정하지 않는다.
- `.proto`와 generated `*_pb2.py`, `*_pb2_grpc.py`는 수정하지 않는다.
- 기존 checkpoint를 덮어쓰거나 삭제하지 않는다.
- replay/intervention buffer를 checkpoint에 포함한다고 가정하지 않는다.
- canonical checkout의 기존 dirty 변경을 통합 작업으로 임의 흡수하거나 덮어쓰지 않는다.

## 3. production data flow (목표 토폴로지)

> **2026-07-30에 이 토폴로지가 Kanu production learner로 실제 가동됐다.** 아래 그림은 더 이상 목표도만이 아니다. 짧은 실물 actor E2E는 통과했고, target 5,000 checkpoint/장시간 운용만 아직 미완료다(§1, §11.5).
>
> 📌 **토폴로지 자체는 서버 이전으로 바뀌지 않았다** — 프로세스 경계도, 단일 프로세스 제약도, SSH tunnel + loopback bind도 그대로다. 바뀐 것은 오른쪽 상자가 도는 **기계 이름**뿐이다(kanu → `junhyeong_ai`, GPU 5 → GPU 0). 2026-07-31에 이 전 구간이 새 호스트에서 200 transition으로 재현됐다(§5.8).

현재 구현은 RAM replay store를 별도 프로세스에서 sampling할 RPC가 없기 때문에, 첫 production topology를 단일 프로세스로 고정한다.

```text
robot laptop
  run_remote_rlpd_actor.py
  task env -> GripperPenaltyWrapper -> episode stats -> timestamp adapter
          |
          | SSH local forwarding / gRPC
          v
GPU 서버 loopback-only learner process   (kanu -> junhyeong_ai, 2026-07-31)
  raw uint8 observation
       |                  |
       v                  v
  RewardClassifierRuntime  raw policy inference
          |
          v
  ActorSessionService
          |
          v
  FrozenResNet10TrunkExtractor
  current/next: cam1/cam2 float32 (1,4,4,512)
          |
          v
  FaultGated FeatureReplayIngress
       |               |
       v               v
  online replay   online intervention pool
       |               |
       +-------+-------+
               |
  raw canonical demo -- one-time trunk conversion
               |
  feature offline demo pool
               |
               v
        RLPDBatchSampler
      online 50% + demo union 50%
               |
               v
       HILSERLLearner worker
       critic/grasp + all update
               |
       +-------+----------------+
       |                        |
 every 50 steps          every 5,000 steps
       v                        v
 VersionedPolicyRuntime    immutable checkpoint directory
       |
       +---- next actor inference
```

gRPC ingress, classifier, policy inference, replay stores, sampler, learner worker가 같은 process에 있다. 이것은 현재 RAM buffer 공유를 위한 의도된 제약이다. 향후 process split은 `PolicyPublisher` 또는 별도 replay service 경계 뒤에서 결정한다.

> 🔴 **그림의 `raw uint8 observation` 은 "카메라 원본"이 아니다.** actor가 `IMAGE_CROP` 을 적용한 뒤 128×128로 만든 **canonical observation**이다. ~~서버는 그것을 classifier에 **변환 없이** 넘긴다. classifier는 크롭 없는 프레임으로 학습됐으므로 **이 화살표가 §12.1의 결함이 물리는 지점**이다.~~
>
> **(2026-07-29 정정)** 앞 문장은 더 이상 맞지 않는다. **classifier는 이 화살표를 타지 않는다.** 관측에는 이제 **무크롭 128×128 JPEG sidecar**(`classifier` 예약 키)가 ~2 Hz로 함께 실려 오고, 서버는 canonical 검증 **전에** 그것을 벗겨내 **그것만** 분류한다. 그림의 canonical 관측은 **정책 전용**이다. schema hash는 그대로 `3459098d…` — sidecar는 hash 계산에 들어가지 않는다. §12.1

## 4. 구현 상세

### 4.1 production CLI와 composition root

`scripts/run_rlpd_learner_server.py`가 다음을 한 process에서 조립한다.

- dependency/version preflight
- 명시적 JAX backend 검사(`cpu` 또는 `gpu`; 생략 불가)
- canonical demo strict load와 provenance 검사
- verified ResNet asset/cache 준비
- reward classifier SHA 검사, local restore, warm-up
- 실제 dual-input `SACAgentHybridSingleArm` 생성
- raw demo의 one-time frozen-trunk conversion
- feature replay/demo 고정 tensor RAM preflight
- fresh/resume checkpoint state 준비
- learner-mode `FeatureReplayIngress`와 fault gate
- feature offline demo pool과 RLPD sampler
- versioned policy runtime과 actor service
- loopback-only gRPC server
- 정확히 한 개의 non-daemon learner worker
- JSONL과 W&B logger
- signal/shutdown 처리

CLI는 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부한다. 외부 공개 port 대신 SSH tunnel을 사용한다. `--require-jax-backend`는 필수라 GPU 서버에서 GPU가 CPU로 조용히 fallback하는 것을 허용하지 않는다. *(2026-07-31 `junhyeong_ai`의 Blackwell sm_120에서도 `--require-jax-backend gpu`가 그대로 통과했다 — §5.8.)*

다만 **3-CLI 실기 운용에서는 이 Python CLI를 직접 복사해 실행하지 않는다.** laptop3의
server terminal에서 `ros2_ur_ws/run_hil_server.sh`를 실행한다. 이 wrapper가 GPU 서버(현행
`junhyeong_ai`)의 exact Python argv, artifact SHA, threshold 0.5, RAM gate, 단일 process와
gRPC health를 검사하고 healthy process를 재사용하며 SSH tunnel을 연다. 전체 명령은 문서
최상단과 §11.5에 있다.

fresh run은 checkpoint root 아래에 기존 `checkpoint_*` entry가 있으면 거부한다. resume는 counters, CTA ratio, publish/checkpoint boundary, inference RNG, fingerprint를 조립 전에 검사한다. initial policy는 step 0/version 0부터 actor service와 learner가 동일한 parameter reference를 공유한다.

`--synthetic-e2e`는 일반 serving 옵션이 아니라 laptop→server 학습 수명주기를 끝까지 검증하는 명시적 acceptance scope다.

- `--dry-run`과 상호 배타적
- offline demo item 전체가 `synthetic_acceptance_only=true`여야 함; real/synthetic 혼합 거부
- `--target-learner-step` 필수, 범위 1..10
- `--replay-capacity >= 100`
- `--synthetic-transition-count` 정확히 100; pass 시 replay insert count도 정확히 100
- `--synthetic-actor-id`/`--synthetic-run-id`와 일치하는 actor/run만 server allowlist로 허용
- `--synthetic-timeout-s` 1..1,800초의 wall-clock deadline
- 실제 batch 256, online/demo 50:50, `training_starts=100`, CTA ratio 2, optimizer/model/discount는 production과 동일
- 검증 시간을 줄이기 위해 publish/checkpoint period만 1 step으로 단축
- gRPC bind, classifier, feature ingress, replay sampling, CTA, policy publish, checkpoint, process restart/resume를 실제로 실행
- target은 fresh/restored learner step에서 정확히 +1이어야 함
- gRPC stop, worker join, process-stopped log, logger close 후 full checkpoint load roundtrip/counter/trunk invariant까지 통과해야 pass event 출력

synthetic execution scope은 fingerprint의 `execution_scope=synthetic_laptop_server_e2e_v1`로 묶는다. 일반 robot lineage의 `production_robot_data_v1`과 다르므로 synthetic checkpoint를 production robot run으로 resume하거나 그 반대로 섞을 수 없다.

robot actor는 expected policy model ID, reward authority, reward model ID, observation schema hash를 pin할 수 있다. pin이 하나라도 설정되면 episode 시작마다 `GetServerInfo`를 새로 조회하고 mismatch를 첫 inference 전에 거부한다. 현재 production 값은 policy `hil-serl-hybrid-sac-resnet10-trunk-cache-v1`, reward authority `server_classifier`이며 reward model ID는 server CLI에 준 값이다.

현재 wire는 protocol `2`, transition schema `3`이다. `reward_authority=server_classifier`는
MANUAL에서도 서버가 reward를 최종화한다는 뜻이지 classifier가 반드시 자동 success를
결정한다는 뜻이 아니다. MANUAL의 one-shot operator assertion과 AUTO의 classifier verdict를
서버가 같은 보호된 식으로 결합한다.

synthetic E2E server는 production actor가 잘못 연결되지 않도록 별도 policy model ID `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`을 advertise한다. `run_fake_e2e_actor.py`는 이 ID를 exact pin하며 production robot actor는 production ID를 계속 pin한다.

### 4.2 학습 계약

- agent: upstream `SACAgentHybridSingleArm`
- external observation: `state float32 (1,19)`, `cam1/cam2 uint8 (1,128,128,3)`
- learner observation: explicit current/next `state float32 (1,19)`, `cam1/cam2 float32 (1,4,4,512)`
- feature cut: pretrained ResNet-10 `stop_gradient` 직후; no GAP, augmentation `none`
- sample-time trainable visual head: `SpatialLearnedEmbeddings8 + Dropout0.1 + Dense256 + LayerNorm + tanh`
- action: `float32 (7,)`; EEF 6D와 gripper `{-1,0,1}`
- seed: `42`
- discount: `0.97`
- batch: `256`
- online replay: `128`
- demo union: `128`
- CTA ratio: `2`
- training start: replay 100개 이상이며 offline demo가 1개 이상
- policy publish: learner step 50마다
- checkpoint: learner step 5,000마다, publish boundary에서

CTA learner step 하나는 critic/grasp-critic update 한 번과 all-network update 한 번을 수행한다. 따라서 정상 fresh lineage에서는 `gradient_step == learner_step * 2`다.

기본 production period는 publish 50/checkpoint 5,000이다. `--synthetic-e2e`에서만 둘 다 1로 바뀌며, 학습 batch/threshold/CTA는 줄이지 않는다. 따라서 synthetic step 1은 gradient step 2, policy version 1, `checkpoint_000000000001`을 동시에 만들어야 성공이다.

learner batch에는 아래 여섯 필드만 전달한다.

```text
observations
next_observations
actions
rewards
masks
grasp_penalty
```

timestamp, actor/run/episode/transition ID, intervention label과 success metadata는 학습 tensor payload에서 제거하고 logging/provenance sidecar에 유지한다.

#### transport protocol 2 / transition schema 3

`PROTOCOL_VERSION="2"`는 그대로이고 `SCHEMA_VERSION`만 **3**으로 올라갔다. schema 3의
`Meta`에는 transition마다 다음 두 bool이 포함된다.

- `auto_success`: false가 기본 MANUAL, true가 명시적 AUTO mode snapshot
- `operator_success`: MANUAL에서 GUI가 현재 episode에 발행한 one-shot success token

서버는 두 값이 동시에 true인 요청을 거부하고, classifier 결과와 함께
`operator_success OR (auto_success AND classifier_success)`를 독립 검증한다. 이 변경은 아래
canonical **observation schema v2**의 tensor 순서나 hash를 바꾸지 않는다. 즉 현행을 짧게
쓸 때는 **protocol 2 / data schema 3 / observation schema v2**라고 적는다.

#### final unified observation schema v2

hardware branch `6a0b127` 통합 후 canonical external state는 shape `(1,19)`를 유지하지만 gymnasium `Dict` flatten의 실제 정렬 계약에 맞게 순서가 바뀐다.

```text
schema id: hil-serl-ur-canonical-observation-v2
schema hash: 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
state groups: gripper_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel
gripper_position index: 0
```

즉 `state[0,-1]`은 gripper가 아니라 `tcp_angular_velocity_z`다. gripper는 `GRIPPER_POSITION_INDEX`/`gripper_position_from_state()`로만 읽어야 한다. shape가 같아도 ordered feature/schema hash가 다르므로 v1 checkpoint/fingerprint와 v2를 섞지 않는다. 5.5의 v1 Kanu E2E는 interim이고 5.6의 v2 결과만 최종 통합 근거로 사용한다.

external canonical observation shape와 learner storage shape는 구분한다. actor policy와 reward classifier는 raw image를 소비한다. ingress는 classifier가 reward/termination을 finalize한 뒤 current/next raw observation을 frozen trunk로 encode하고, ring에는 explicit `observations`/`next_observations`의 state와 두 camera map만 보유한다. raw image packing과 upstream `pack_batch()`는 feature learner path에서 사용하지 않는다.

offline demo와 online intervention은 하나의 logical demo union으로 취급한다. source draw를 크기 비례 multinomial로 뽑아 작은 intervention pool이 deterministic rounding 때문에 영구적으로 굶는 문제를 제거했다.

### 4.3 gripper penalty

`GripperPenaltyWrapper`는 policy proposal이 아니라 실제 실행 action을 기준으로 penalty를 계산한다. intervention이 있으면 `info["intervene_action"]`이 우선이다.

- 이미 열린 상태에서 open 반복: configured penalty
- 이미 닫힌 상태에서 close 반복: configured penalty
- 그 외: `0.0`
- 기본 task 값: `-0.02`

`run_remote_rlpd_actor.py`와 legacy `train_rlpd_actor.py` 모두 task config에서 값을 명시적으로 읽어 wrapper를 구성한다. env task config와 experiment config가 둘 다 값을 제공하면서 서로 다르면 실패한다.

learner mode ingress는 `grasp_penalty`의 존재, scalar float32 contract, finite 여부를 검사한다. server CLI의 `--grasp-penalty`가 task 값을 pin하며 offline demo와 online transition 모두 정확히 `0` 또는 configured non-positive penalty만 허용한다. 기본값은 `-0.02`다. 신규 transition/demo에서 누락된 값을 0으로 보정하지 않는다.

### 4.4 replay ingress fault 경계

`FaultGatedReplayIngress`는 acceptance와 learner sampling을 하나의 lock 경계로 직렬화한다. replay insert 이후 intervention insert가 실패하는 것처럼 부분 route가 생기는 경우, 최초 예외를 actor service에 그대로 전달한 뒤 fault를 영구 latch한다.

fault 이후에는:

- 추가 acceptance 거부
- learner sampling 거부
- status 조회만 진단용으로 허용
- actor service는 기존 fail-stop semantics 유지

따라서 learner가 replay-only half insert를 계속 sample하는 것을 막는다.

### 4.5 policy publish와 learner fault

`VersionedPolicyRuntime`은 parameter tree를 매 publish마다 serialize/deep-copy하지 않고 reference를 원자적으로 교체한다. publish 전 다음을 검사한다.

- parameter tree structure
- leaf dtype/shape compatibility
- 모든 parameter finite
- canonical observation의 deterministic action smoke
- action `float32 (7,)`, finite, gripper 3-state

검증 실패 snapshot은 publish하지 않는다. learner update가 non-finite이거나 exception을 내면 learner worker만 fault되고 last-known-good policy는 유지된다.

현재 운영상 중요한 제한이 있다. bounded run에서 worker fault는 exit code 2로 process를 종료하지만, continuous run에서는 `rlpd_learner_worker_fault` event를 한 번 출력한 뒤 last-known-good policy serving을 계속한다. 이 degraded 상태는 아직 gRPC health에 노출되지 않는다.

### 4.6 checkpoint와 resume

checkpoint는 전체 Flax train state를 저장한다.

- model params
- target params
- optimizer state
- agent RNG
- learner step
- gradient step
- policy version
- inference RNG
- fingerprint document와 SHA
- agent-state payload SHA

각 `checkpoint_<12-digit-step>` directory는 `agent_state.msgpack`, `metadata.json`을 새 파일로 기록하고 fsync한 뒤 `completion.json`을 마지막에 기록한다. 기존 path는 덮어쓰지 않는다. 실패 중간 directory도 자동 삭제하지 않는다.

`latest_path()`는 completion marker와 구조/checksum을 통과하는 최신 checkpoint만 고른다. 다만 더 높은 손상 entry가 같은 root에 있으면 lineage collision 방지를 위해 새 빈 checkpoint root로 explicit resume해야 한다.

production CLI는 `--resume-latest`뿐 아니라 explicit `--resume-path`에도 `completion.json`을 요구한다. markerless legacy v1을 읽는 `allow_legacy_markerless`는 library-only one-off migration escape hatch이며 CLI에 노출하지 않는다. 과거 checkpoint에 marker를 손으로 만들어 production resume하는 것은 허용하지 않는다.

추가 hardening:

- checkpoint root별 advisory single-writer lock
- 다음 payload 저장 뒤에도 남겨야 할 free-space reserve 검사
- CLI reserve 기본값 2 GiB
- fresh start와 resume lineage 혼합 거부
- resume 시 exact counter/publish boundary 검사

실제 agent checkpoint payload는 약 305 MiB였고 schema v1과 최종 v2 Kanu synthetic E2E에서 `320,100,609 B`를 관측했다(2026-07-31 `junhyeong_ai` 재현에서도 checkpoint 하나가 306 MB로 같은 자릿수였다 — §5.8). pruning이 없으므로 production 5,000-step checkpoint마다 이 정도가 누적된다고 가정하고 disk를 계획해야 한다. *(📌 kanu는 여유 73 GB / 96% 사용이라 이것이 실질 제약이었다. `junhyeong_ai`는 약 594 GB 여유이므로 제약이 훨씬 느슨하다 — 그래도 자동 pruning은 없다.)* synthetic scope는 검증용으로 period 1이므로 target을 1..10으로 제한한다.

### 4.7 fingerprint

현재 fingerprint는 external observation schema, learner algorithm config, ResNet SHA에 더해 아래 production run contract를 포함한다.

- `frozen_trunk_feature_hybrid_sac_v1` contract revision
- execution scope: `production_robot_data_v1` 또는 `synthetic_laptop_server_e2e_v1`
- learner representation `resnet10_frozen_trunk_map_f32_v1`
- cut point `pretrained_resnet10.stop_gradient`, camera key/shape/dtype, no-GAP 계약
- augmentation `none`
- policy model ID: production `hil-serl-hybrid-sac-resnet10-trunk-cache-v1`, synthetic `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`
- verified ResNet-10 asset SHA
- action dtype/shape/range/gripper values
- reward classifier SHA/threshold/model ID
- offline demo artifact SHA 목록과 transition 수
- exact grasp-penalty allowed values `[0, configured penalty]`
- JAX/JAXLIB/Flax/Distrax/TFP version

resume에서 document 또는 SHA가 다르면 즉시 실패한다. **reward threshold도 fingerprint에 들어간다**(`scripts/run_rlpd_learner_server.py`의 `run_contract`). 역사적으로 0.85 → 0.5(`53d5cf6`) → 0.2(`1b02857`)로 움직였지만, **2026-07-30 현행 production 값은 다시 0.5**다(`ur_env/rlpd_receive_server.py::DEFAULT_REWARD_THRESHOLD`; `run_hil_server.sh`도 exact 0.5를 pin하고 기존 process argv를 검사한다). held-out FPR은 0.85/0.5/0.2에서 모두 0%라 그 지표만으로 셋을 구분할 수 없었고, 0.5는 0.2보다 hardest known negative와의 margin을 더 보수적으로 유지한다. 따라서 다른 threshold로 만든 checkpoint를 현행 lineage로 resume하면 fail-closed로 거부되는 것이 정상이다. 이전 raw/random-crop checkpoint와 frozen-trunk/no-aug checkpoint도 fingerprint가 다르며 자동 migration하지 않는다. synthetic E2E와 production robot execution scope 역시 서로 resume하지 않는다.

### 4.8 reward classifier와 Flax local-I/O 수정

> ⚠️ **이 절이 다루는 checkpoint `e329986b...`(Jul-24)는 폐기 대상이다.** **2026-07-28 Kanu GPU 실측**에서 **0724 도메인 success 프레임 1,123장 / 크롭 없는 입력 / threshold 0.85** 기준 recall이 `0.0%`(1,123장 중 0건, mean 확률 `0.007`)였다. threshold를 아무리 낮춰도 살아나지 않는다. 실기 learner에 이 checkpoint를 물리면 로봇이 성공해도 reward가 영원히 0이므로 HIL-SERL 학습이 시작되지 않는다.
>
> ※ **같은 checkpoint가 0720 test split에서는 recall `93.4%` @0.85 / FPR `0.0%` 다**(§12.3). 즉 "이 모델은 아무것도 못 맞힌다"가 아니라 **"새 도메인(0724)으로 전혀 일반화하지 못한다"** 가 정확한 서술이다. 두 숫자를 조건 없이 나란히 인용하면 모순처럼 보인다.
>
> 아래에 남긴 SHA, 경로, load/warm-up 수치는 **2026-07-27 시점의 역사적 기록**이며 지우지 않는다. 다만 그때 검증한 것은 SHA/load/warm-up까지이고 **분류 성능은 검증하지 않았다.**
>
> **Flax local-I/O backend 수정 기록(아래 5단계)은 계속 유효하며 새 정본 checkpoint에도 그대로 적용된다.**
>
> 새 정본 경로와 orbax 디렉터리 제약은 [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) 1.3, 측정 근거는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 있다.

폐기된 classifier artifact(역사적 기록):

```text
local:
/home/laptop3/youngwoong_ws/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150

Kanu historical path:
/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150

SHA-256:
e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997
```

이 저장소는 upstream type annotation을 위해 full TensorFlow 대신 최소 shim을 제공한다. Flax 0.10.5는 `tensorflow`가 import 가능하다는 이유만으로 TensorFlow I/O mode를 고를 수 있었고, shim은 의도적으로 `tf.io`를 구현하지 않으므로 실제 classifier restore가 실패했다.

수정된 경계는 다음과 같다.

1. repository ResNet-10 asset의 SHA를 검증한다.
2. custom cache와 upstream fixed cache가 있으면 모두 같은 SHA인지 확인한다.
3. 다른 cache를 덮어쓰지 않고 verified bytes만 새로 생성한다.
4. `flax.io.BackendMode.DEFAULT`를 명시적으로 선택한다.
5. 그 뒤 upstream classifier checkpoint loader를 호출한다.

실제 local artifact의 load/warm-up이 성공했다. 별도 측정에서 전체 준비 약 7.89초, warm-up 약 674 ms였고 zero-image smoke probability는 약 `0.081304`였다. **이 timing은 2026-07-27 laptop3 로컬 CPU acceptance 관측값이며 GPU production latency 기준이 아니다** (참고: Jul-27 checkpoint의 Kanu GPU 실측은 load 13.5 s, 워밍업 후 1.03 ms/frame — §12.3). zero-image smoke는 runtime이 유한한 확률을 낸다는 것만 보이며 **분류 정확도/recall과는 무관하다** — 실제 성공 프레임에 대한 recall은 이때 측정하지 않았고, 이후 2026-07-28 실측에서 0724 도메인 `0.0%`로 확인됐다.

### 4.9 logging, W&B, protobuf

- JSONL은 항상 local path에 structured event를 기록한다.
- W&B 기본 mode는 `offline`이다.
- W&B `0.26.0`을 pin한다.
- 실제 W&B offline run directory 생성과 checked-in protobuf module 동시 import를 자동 테스트한다.
- protobuf 7.34.1 process에서는 generated binding import 전에 pure-Python compatibility mode를 선택한다.
- W&B online login/upload는 사용자가 credential을 제공한 뒤 별도 1회 검증한다.

실제 W&B offline artifact test와 protobuf co-import test는 존재한다. 장시간 learner/classifier/gRPC 동시 부하에서의 logging 안정성은 아직 GPU 서버 soak 범위다(kanu에서 못 했고, `junhyeong_ai`에서도 아직이다).

### 4.10 fake canonical demo

`scripts/generate_fake_canonical_demo.py`는 deterministic canonical raw transition 두 개를 pickle로 만든다. strict loader를 즉시 다시 통과시키고 SHA, transition 수, synthetic marker를 JSON으로 출력한다.

모든 item은 `synthetic_acceptance_only=true` provenance를 가진다. 생성기는 기존 파일을 덮어쓰지 않는다. production CLI는 이 marker를 construction `--dry-run`과 bounded `--synthetic-e2e`에서만 허용한다. 옵션 없는 일반 learner, continuous mode, production robot-data mode에서는 거부한다.

CLI는 demo SHA/provenance/penalty를 검증한 뒤 `--demo-extraction-batch-size` 단위로 verified frozen trunk를 한 번만 적용한다. 변환된 pool은 explicit current/next `float32 (1,4,4,512)` map과 learner tensor/provenance sidecar만 소유하고 raw demo reference는 live ring 할당 전에 해제한다. fake와 real canonical demo 모두 같은 one-time conversion 경계를 통과한다.

fake `--dry-run`으로 확인할 수 있는 것:

- canonical loader/schema
- classifier/agent construction
- dependency/backend/asset preflight
- fingerprint 생성
- production composition construction
- JSONL/W&B offline initialization

fake `--synthetic-e2e`가 추가로 실제 실행하는 것:

- gRPC bind와 SSH transport
- replay 100개 도달
- learner CTA update
- 1-step acceptance publish/checkpoint
- process stop, fresh process checkpoint resume, last-known policy serving

fake data로도 확인할 수 없는 것:

- production 기본 50-step publish/5,000-step checkpoint 장시간 lifecycle
- 실제 robot distribution의 reward/action 품질
- 실제 camera timing/skew, intervention 행동, robot safety/fault recovery

사용자는 현재 milestone의 완료 조건을 fake data로 laptop→Kanu gRPC/CTA/publish/checkpoint/resume까지 돌리는 것으로 명확히 정했다. 이를 위해 production robot gate를 제거하지 않고 별도 fingerprint의 bounded `--synthetic-e2e` scope를 구현·검증했다.

## 5. 검증 현황

### 5.1 회귀 suite

**현재값 (2026-07-29, `43ba314`, 이 문서 작성 중 재실행):**

```text
serl_ur_infra/tests:  333 passed, 11 skipped in 3.47s
ur_gello_bringup:     436 passed in 7.38s
```

`serl_ur_infra` 는 §9의 venv+overlay 명령으로, `ur_gello_bringup` 은 `-p no:launch_testing` 로 돌렸다. **actor 머지(`3f199d4`)로 테스트가 253 → 333으로 늘었다** — 늘어난 80개가 actor/wrapper/task-config 쪽이다.

🪤 `third_party/hil-serl/serl_launcher` 를 `PYTHONPATH` 에서 빼면 **조용히 `300 passed, 13 skipped`**로 떨어지고, 사라지는 것이 하필 `test_cube_in_cup_config.py` / `test_frame_wrappers.py` 다. 게다가 skip 사유가 "submodule is not checked out"이라고 **거짓말한다**(서브모듈은 체크아웃돼 있다). 새 worktree는 서브모듈 미초기화로 296/17이 된다. **녹색이 아니라 passed 수를 볼 것.**

**역사값 (2026-07-27, learner/hardware 통합 merge `248255f` 시점):**

```text
serl_ur_infra: 253 passed, 4 skipped, 6 warnings in 3.10s
ur_gello_bringup: 436 passed in 7.35s
```

skip은 JAX 비용이 큰 opt-in 실제 agent/E2E 경로다. 같은 unified tree에서 실제 frozen-feature agent `2 passed in 22.66s`, 실제 checkpoint `1 passed in 33.61s`, 실제 localhost fake E2E `1 passed in 32.77s`를 별도 실행했다. 실패는 0이었다. 당시 UR/GELLO suite는 system pytest/anyio plugin 충돌을 피하기 위해 `-p no:anyio`를 사용했다. **위 opt-in 3종은 2026-07-29 기준 HEAD에서 재실행하지 않았다** — 07-27 관측값이다.

### 5.2 opt-in 실제 agent checkpoint/resume

`tests/test_actual_agent_checkpoint_integration.py`는 실제 upstream `SACAgentHybridSingleArm`을 생성해 다음을 확인한다.

1. raw fake demo를 exact frozen-trunk map으로 one-time conversion
2. cached-feature batch로 CTA update
3. publish와 checkpoint 저장
4. fresh real-agent template 생성
5. checkpoint restore
6. `prepare_learner_state()`와 `compose_learner()`로 production 객체 재조립
7. train-state leaf, counters, agent/inference RNG 확인
8. raw observation deterministic/stochastic action exact 비교
9. resume learner에서 추가 CTA update/publish/checkpoint

비용을 줄이기 위해 test config의 publish/checkpoint period는 1이다. 따라서 boundary 구현을 검증하지만 default 50/5,000 장시간 run을 대체하지는 않는다.

opt-in 환경 변수는 `RUN_HIL_SERL_ACTUAL_CHECKPOINT=1`이다. unified schema v2에서 CPU JAX 0.5.3 frozen-trunk feature checkpoint save/resume/continued CTA 통합 검증 `1 passed in 33.61s`를 확인했다.

### 5.3 opt-in 실제 frozen-trunk agent

`tests/test_actual_frozen_trunk_feature_agent.py`는 `RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1`로 실행하며 unified schema v2에서 `2 passed in 22.66s`를 확인했다.

- raw path와 cached-map path의 deterministic/stochastic 7D action 수치 동치
- feature shape/dtype/finite/no-augmentation contract
- CTA 후 trainable camera head/proprio path가 변함
- CTA 후 online trunk가 verified initial trunk와 exact equal
- target trunk을 verified trunk으로 repin한 뒤 exact equal
- 변조된 trunk parameter snapshot/publish 거부

### 5.4 production dry-run과 실제 classifier

실제 classifier checkpoint와 생성한 fake canonical demo를 사용한 local production CLI dry-run이 frozen-feature 통합 뒤 통과했다. 아래 fingerprint는 `execution_scope` field 추가 전의 역사적 construction 결과이며 현재 synthetic/production checkpoint resume identity로 사용하지 않는다.

- event: `rlpd_learner_dry_run_passed`
- artifact root: `/tmp/hil-feature-production-final-zw6U7b` (local acceptance scratch; 영속 lineage로 사용하지 않음)
- backend: CPU
- W&B mode: `disabled`
- fake demo SHA-256: `6907f4e458e87001bc26dc3d9d8e9b7a4c5ae2ac1ee637bd56ce0e82372c9177`
- ResNet cache SHA-256: `175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b`
- production fingerprint: `fb48aee61c655300e19fc58d412534dbbf25752ce3ad792704c6ff1ab42754dd`
- ready state: learner 0, gradient 0, policy version 0, synthetic demo 2개

이전 pre-hardening dry-run에서는 wall time 약 15.03초와 peak RSS 약 1.51 GiB를 관측했다. 최신 hardening run에서는 이 timing/RSS를 다시 계측하지 않았으므로 역사적 참고값으로만 남긴다.

이 결과는 production construction의 실제 dependency/artifact smoke다. 최신 run은 W&B를 disabled로 실행했으므로 실제 W&B offline artifact 검증은 별도 자동 테스트가 근거다. server bind나 training을 실행하지 않으므로 Kanu/robot E2E 결과로 확대 해석하지 않는다.

> 🗄️ **아래 두 문단은 2026-07-27 kanu 관측이다.** SSH alias·GPU 개수·RAM·경로는 전부
> kanu의 것이며 **현행 서버는 `junhyeong_ai`(RTX 5070 Ti ×1, GPU 0)** 다. 파이썬 경로
> `/home/junhyeong/miniconda3/envs/il/bin/python` 는 **우연이 아니라 그대로 재현**된 것이다 —
> 새 서버의 `il` env를 kanu의 `pip freeze`로 만들었기 때문이고, 그래서 fail-closed 핀
> (jax 0.5.3 / flax 0.10.5 / distrax 0.1.5 / tfp 0.25.0)이 이전에도 그대로 통과한다.

처음 사용한 `kanu_junhyeong`은 등록되지 않은 이름이었고 당시 실제 SSH alias는 `kanu`였다(**현행 alias는 `junhyeong_ai`**). **2026-07-27 read-only preflight**에서 RAM 251 GiB(available 152 GiB — *공유 서버라 이 available 값은 그때의 순간값이다*), RTX A4000 16 GB 8장과 `/home/junhyeong/miniconda3/envs/il/bin/python`의 JAX/JAXLIB 0.5.3, Flax 0.10.5, backend `gpu`, device 8개를 확인했다.

Kanu의 기존 repository가 dirty detached 상태였으므로 그 workspace를 수정하지 않고 `/tmp/hil-feature-dryrun-BUJNWu`에 rsync/symlink로 일회성 검증 tree를 구성했다. `CUDA_VISIBLE_DEVICES=0`으로 실제 classifier + agent production dry-run(128/32 capacity)이 통과했고 당시 pre-execution-scope fingerprint는 `8465e464b3f4eb638513eaa4ab3daea85a9435a9ddf47bdbf841a2c2f2aacce9`였다. 별도 실제 GPU feature CTA smoke는 backend `gpu`, device 1개, feature `[1,4,4,512]`, `gradient_step=2`, `augmentation_function=None`(JSON `null`)을 확인했다. raw/cached deterministic action의 max absolute difference는 `0.00012614415027201176`였고 online/target trunk invariant도 통과했다. 5.5의 `defda67...`는 pre-hardware schema v1 interim이고, unified schema v2 authoritative fingerprint는 아래 `fa198537...`다. 어느 결과도 continuous server/robot E2E로 확대 해석하지 않는다.

### 5.5 laptop→Kanu 실제 fake-data learning E2E (schema v1 interim)

> 🗄️ **kanu 실측(2026-07-27). 새 호스트에서 같은 모양을 재현한 것이 §5.8이고, 아래 수치가
> 그 비교 기준선이다.** fingerprint `defda67b…`와 `BeginEpisode` RTT mean `63.29` /
> max `91.16 ms`는 **kanu에 관한 사실**이므로 그대로 둔다 — 새 호스트 값으로 덮어쓰면
> 비교 대상 자체가 사라진다.

`scripts/run_fake_e2e_actor.py`를 laptop3에서 SSH local tunnel 뒤의 Kanu GPU learner에 연결했다. server는 actual classifier, actual dual-input SAC agent, raw→frozen-feature ingress, batch 256/CTA 2 learner, versioned policy, full checkpoint를 실제로 사용했다.

execution fingerprint:

```text
defda67b4463526a6aca4fb397327ff01b93ffd07b9250cc538a46b105bf96cb
```

fresh run:

- laptop actor의 fresh canonical raw transition 100개 ACK/feature replay insert
- `BeginEpisode` RTT: max `91.158068 ms`, mean `63.2926349 ms`
- replay 100 + offline synthetic demo 2; update batch에서 실제 online/demo 128:128 구성 확인
- learner `step=1`, `gradient_step=2`, `policy_version=1`
- `checkpoint_000000000001`; cleanup 후 full load roundtrip/counter/trunk invariant 통과
- update loss 모두 finite, `policy_published`, `checkpoint_saved`, `learner_process_stopped(exit_code=0)` JSONL event 확인

fresh-process resume run:

- Kanu learner process를 새로 시작해 checkpoint 1을 restore하고 actor의 첫 action을 `policy_version=1`로 serving
- replay는 RAM-only이므로 새 transition 100개를 다시 ACK/insert
- `BeginEpisode` RTT: max `95.694507 ms`, mean `64.40843136 ms`
- learner `step=2`, `gradient_step=4`, `policy_version=2`
- `checkpoint_000000000002`; cleanup 후 full load roundtrip/counter/trunk invariant 통과
- update loss 모두 finite, publish/checkpoint/`learner_process_stopped(exit_code=0)` event 확인

위 결과는 schema v1 역사적 근거이며 v2 checkpoint lineage에 사용하지 않는다.

### 5.6 laptop→Kanu 최종 fake-data learning E2E (schema v2)

> 🗄️ **kanu 실측(2026-07-27). 아래 RTT `mean 84.89 / max 372.82 ms`가 서버 이전 후
> tail latency 비교의 정본 기준선**이다(§5.8에서 `57.7 / 82.2 ms`와 대조한다).
> **이 두 숫자는 kanu의 것이므로 절대 갱신하지 않는다.**

- fingerprint: `fa1985378ad2729f466783e4f112d54022e14090430374a6531e4fb715440fcd`
- sender: exact 100 ACK/insert, RTT mean `84.88630425 ms`, max `372.820324 ms`
- 실제 batch 256 CTA: learner/gradient/policy `0/0/0 -> 1/2/1`; online/demo 128:128, 모든 loss finite
- timing: learner `83,997.304 ms`, critic `36,496.960 ms`, full `36,822.142 ms`, sample `1,754.703 ms`
- checkpoint: `checkpoint_000000000001`, payload `320,100,609 B`; cleanup 후 full-load counter/trunk invariant roundtrip 통과
- fresh Kanu process `--resume-latest`: learner/gradient/policy `1/2/1` 복원, policy version 1의 finite `(7,)` deterministic action과 gripper `-1` serving, RTT `133.65432 ms`, JSONL clean stop `exit_code=0`

사용자 요청에 따라 resume process에서 같은 SAC update를 한 번 더 수행하지 않았다. 실제 continued-update/checkpoint 경계는 local actual integration test와 schema v1 Kanu full resume run에서 이미 검증됐다. RTT/timing은 일회 관측값이며 SLA가 아니다.

### 5.7 local opt-in actual fake E2E

`tests/test_actual_fake_data_e2e_learning.py`는 실제 localhost gRPC server/client, canonical raw pixels, frozen feature ingress, CTA, publish, checkpoint, server restart/resume를 하나의 opt-in test로 검증한다.

```text
RUN_HIL_SERL_FAKE_E2E=1
1 passed, 92 warnings in 32.77s
```

이 수치는 unified schema v2 통합 tree의 최종 재실행 값이다.

### 5.8 laptop3→`junhyeong_ai` 실통신 수락 시험 (2026-07-31, 서버 이전) — **§5.5의 모양이 새 호스트에서 재현됐다**

**결론: PASS.** 서버 이전 뒤 laptop3에서 새 서버의 learner에 **transition 200개**
(fresh 100 + **별도 프로세스 resume** 100)를 실제 gRPC로 보내
`gRPC → reward finalize → frozen-trunk feature replay → CTA update → policy publish →
checkpoint → resume` 전 구간이 돌았다. 두 run 모두 서버가
`rlpd_learner_synthetic_e2e_passed`, 액터가 `fake_e2e_actor_passed`를 냈다.

**전문(플래그 근거·명령·cleanup 실측 포함)은 이 문서가 아니라
[`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md)가 소유한다.**
여기 적는 것은 §5.5/§5.6과 **줄 세워 읽을 수 있는 최소 집합**이다.

| 항목 | fresh run | resume run |
| --- | --- | --- |
| learner step | **0 → 1** | **1 → 2** |
| gradient step | **0 → 2** (CTA 2:1) | **2 → 4** |
| policy version | **0 → 1** | **1 → 2** |
| checkpoint | `checkpoint_000000000001` | `checkpoint_000000000002` |
| checkpoint round-trip | **verified** | **verified** |
| 수락된 transition | 100 / 100 (`replay_insert_delta` 100) | 100 / 100 |
| 프로세스 종료 | `learner_process_stopped exit_code=0` | `exit_code=0` |

execution fingerprint (**두 프로세스 동일** — lineage가 이어졌다는 증거):

```text
1a5c96f5a2e7f13f109617c8859e0cc6c5d6be949ac76a5de6b95e6808c4c844
```

RPC 지연 — **kanu와의 대조. 왼쪽 열은 §5.5/§5.6의 kanu 값 그대로다.**

| | kanu | **junhyeong_ai (2026-07-31)** |
| --- | --- | --- |
| `BeginEpisode` mean / max (§5.5 schema v1) | 63.29 / 91.16 ms | **57.7 / 82.2 ms** |
| `BeginEpisode` mean / max (§5.6 schema v2) | 84.89 / **372.82 ms** | **57.7 / 82.2 ms** — tail이 약 **4.5배** 짧다 |

📏 **네트워크는 원인이 아니다.** ICMP RTT가 두 호스트에서 사실상 같다
(`junhyeong_ai` 1.078/2.811/6.979 ms vs kanu 1.272/2.518/6.314 ms, 각 10패킷).
즉 이 차이는 **호스트 연산**이지 링크가 아니다.

🛑 **이 시험이 증명하지 *못한* 것 — classifier 추론.** 액터 출력의
`classifier_success_count: 0`은 **"분류기가 100장을 보고 전부 실패로 판정했다"는 뜻이
아니다.** `run_fake_e2e_actor.py`는 `build_data()`에 **classifier sidecar를 붙이지 않고**,
그러면 `RewardTransitionFinalizer`가 CNN을 **호출하지 않는** `evaluated=False` 분기로 빠진다
(`probability=0.0`, `reward_model_id=""`). 서버 로그도 *"classified NONE of them; every
reward in it is 0 by default, not by verdict"* 로 명시했다.
🪤 **그 도구의 docstring("finalized by the server classifier")은 이 지점에서 오해를 부른다** —
그것은 **finalize 경로**를 말하는 것이지 매 transition CNN이 돌았다는 뜻이 아니다.

✅ **per-transition classifier 추론은 별도로 증명됐다** — 같은 날 조작자의 **실기 로봇 세션**
(sidecar를 실제로 보내는 production actor)에서 replay **316** / intervention **210**이
새 서버로 흘렀다. 이 시험이 못 덮은 두 구멍(진짜 classifier sidecar, `intervened=1` ingress)이
그쪽에서 닫혔다.

증명된 것: classifier checkpoint가 새 서버에서 로드되고 directory SHA `512b6575…`가
검증됐다 · `reward_model_id`가 handshake에 광고되고 액터가 pin했다 · 서버 권위 reward
finalize 경로가 200 transition 전부에 대해 돌며 `masks`/`dones` 일관성 검사를 통과했다 ·
Blackwell sm_120에서 `--require-jax-backend gpu`가 통과하고 CTA update가 실제 GPU에서
유한한 loss로 돌았다.

⚠️ **이 시험은 로봇 루프 주기를 재지 않았다.** transition당 RPC 합계 213.8 ms를 kanu의
로봇 루프 512 ms에서 그냥 빼지 마라 — 후자는 카메라 디코드와 `env.step`의 100 ms 자체
페이싱을 포함한 **전체 주기**다. 실기 루프가 실제로 얼마나 빨라지는지는 G21과 함께
**실기에서 다시 재야 한다**([`CONTROL_RATE_AND_ASYNC_PLAN_KO.md`](./CONTROL_RATE_AND_ASYNC_PLAN_KO.md)).

## 6. frozen-trunk feature replay와 메모리

### 6.1 현재 기본값

현재 CLI 기본 capacity는 다음과 같다.

- replay: 50,000 logical transitions
- intervention: 10,000 logical transitions

각 logical slot은 current/next, cam1/cam2의 `float32 (1,4,4,512)` map을 보유한다. replay 50,000 + intervention 10,000 기본 ring의 camera tensor는 정확히 `7,864,320,000 B = 7.32421875 GiB`다. state/action/reward/mask/grasp tensor, offline demo feature pool, NumPy/Python sidecar, JAX/XLA, classifier/learner model memory는 별도다.

CLI는 ring과 offline demo의 고정 tensor byte를 실제 할당 전에 계산하고 Linux `MemAvailable`에서 `--feature-memory-reserve-gib`(기본 2 GiB)를 남길 수 없으면 fail-closed한다. dry-run은 replay/update를 검증하지 않으므로 128/32 같은 작은 ring을 쓴다. RAM buffer는 process 종료 시 사라진다.

### 6.2 exact feature boundary

저장 경계는 pretrained image encoder 전체의 임의 bottleneck이 아니라 upstream ResNet-10의 기존 `jax.lax.stop_gradient` 직후다.

```text
canonical uint8 image
  -> ImageNet normalization
  -> pretrained ResNet-10 backbone
  -> 4x4x512 float32 map             [stop_gradient; frozen; replay/demo에 저장]
  -> SpatialLearnedEmbeddings
  -> Dropout(0.1)
  -> Dense(256) -> LayerNorm -> tanh [sample time; critic/grasp critic에서 trainable]
```

즉 **GAP/pooling을 사용하지 않고**, trainable 256-D head 뒤의 값도 저장하지 않는다. head weight가 CTA 중 바뀌어도 과거 feature가 오염되지 않도록 frozen 경계 직후를 cache한다. 두 camera head와 proprio head는 계속 학습된다.

### 6.3 invariant와 두 input path

- external actor policy: raw image를 받아 trunk + 현재 trainable head를 모두 실행
- reward classifier: raw image를 계속 사용
- learner update: cached trunk map을 받아 trunk를 skip하고 현재 trainable head부터 실행
- offline demo: startup에 verified trunk를 한 번만 통과
- online ingress: classifier finalize 후 current/next를 encode하고 raw array를 ring에 보유하지 않음
- augmentation: `none`

online trunk가 update로 변하면 cached feature 의미가 깨지므로 publish/checkpoint 경계에서 verified initial trunk과 exact tree equality를 검사한다. target-network Polyak update가 trunk leaf를 수치적으로 변형하지 않도록 각 CTA candidate의 target trunk를 verified trunk으로 repin한 뒤 invariant를 검사한다. raw/cached policy action 동치와 CTA 후 online/target trunk exact equality가 실제 JAX opt-in test로 검증됐다.

## 7. 남은 차단점

### P0 — 실기 production 승인 전 필수

0. ✅ **~~사용 가능한 reward classifier 부재 — 현재 최상위 차단점~~ → 2026-07-29 코드에서 해소.** *(아래는 갱신된 항목별 상태다. **남은 것은 실기 검증뿐이고, 코드 차단점은 없다.**)*
   - 지금까지 모든 dry-run/E2E가 사용한 Jul-24 `e329986b...`는 **0724 도메인·무크롭·@0.85 에서 recall `0.0%`** 로 폐기됐다(§12.3). 이 상태로 실기를 돌리면 reward가 영원히 0이라 학습이 시작조차 하지 않는다.
   - 새 정본은 2026-07-27 kanu에서 만든 `checkpoint_150`(약 43 MB, 파일 14개)이며 단일 파일이 아니라 **orbax 디렉터리 포맷**이다.
     - 🗄️ **kanu 시절 경로**(역사): `~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150` — FM/diffusion 스택 디렉터리 **안**이었다.
     - ✅ **현행 경로**: `junhyeong_ai:~/hil-serl-data/classifier_ckpt/checkpoint_150`. 같은 directory SHA `512b6575…`이고 `run_hil_server.sh`가 `HIL_REMOTE_DATA_ROOT`에서 파생해 자동으로 가리킨다. laptop3 사본은 `gello_software/classifier_ckpt/cube_in_cup_all3/checkpoint_150`(gitignore).
   - ✅ **~~새 정본의 recall/FPR 미측정~~ → 해소.** 2026-07-28 Kanu GPU에서 측정했다(§12.3): **크롭 없는 입력** 기준 0720 test split(n=166) @0.5 recall 100.0% / FPR 0.0%, 0720 held-out 전체(n=266) @0.5 recall 86.8%.
   - ✅ **~~단 그 수치는 크롭 없는 입력의 것이라 현재 actor 경로(크롭 적용, recall@0.85 100.0% → 33.3%)에는 적용되지 않는다. 해결은 크롭에 맞춘 재학습~~ → 2026-07-29 해소.** **재학습이 아니라 sidecar 분리로 고쳤다.** 분류기가 다시 무크롭 프레임을 먹으므로 **위 수치가 그대로 이 경로에 적용된다**(§12.1, §12.7).
   - ✅ **~~`checkpoint_sha256()` 이 `os.path.isfile()` 을 강제해 orbax 디렉터리를 못 받는다~~ → 2026-07-29 해소 (G19).** 재귀 `directory_sha256()`(`ur_env/classifier_sidecar.py`, ≈`:384`)에 위임한다. 단일 파일 digest는 예전과 동일하다.
   - ✅ **~~두 기본 SHA 상수가 폐기된 Jul-24(`e329986b…`)를 가리킨다~~ → 2026-07-29 교체됨.** `run_rlpd_learner_server.py::DEFAULT_CLASSIFIER_CHECKPOINT_SHA256` / `run_rlpd_receive_server.py::DEFAULT_CHECKPOINT_SHA256` 가 **`512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d`**(= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`, 파일 14개)를 가리킨다. **이 세션에서 직접 재계산해 일치를 확인했다.** 🪤 **G19 수정과 이 상수 교체는 반드시 같이 나가야 했다** — 해싱만 고치면 은퇴 체크포인트로 서버가 조용히 기동해 reward가 영구 0이 된다(§12.6-3).
   - ✅ **실물 actor 경로에도 투입됐다.** 2026-07-29 첫 production-model E2E에서 sidecar/classifier/replay/learner 왕복을 수행했다. 남은 것은 장시간 verdict 분포와 가림 조건 검증이다.
   - 현행 threshold는 **0.5**다(§4.7). `run_hil_server.sh`가 code default와 process argv를 둘 다 exact 검사한다.

1. **production lifecycle bounded/continuous GPU acceptance**
   - JAX/JAXLIB 0.5.3 GPU production dry-run, 단일 feature CTA, laptop→Kanu synthetic E2E step 1/resume step 2는 통과했다.
   - production 기본 50-step publish는 현행 run에서 policy version 6까지 확인했다. **5,000-step checkpoint/완주는 아직 미검증**이다.
   - 기본 ring camera map `7,864,320,000 B` + offline demo + reserve 할당 후 장시간 RSS/VMS/GPU memory/compile/contention을 계측한다.
   - CPU용 `requirements-learner.lock`을 GPU 서버의 공유 env에 그대로 설치하지 않는다. **`junhyeong_ai`도 공유 머신이다** — 계정이 같고 GPU가 1장뿐이며 다른 사람의 env(`base acg expo gr00t lerobot qc robocasa robodiff`)가 같은 conda에 있다. 🪤 즉흥 `pip install`은 의존성을 조용히 끌어올린다(측정: `pip install flax==0.10.5 optax==0.2.4`가 jax를 0.5.3 → 0.6.2로 올려 learner가 fail-closed로 죽었다). env 구성은 `pip install --no-deps -r <freeze>` 하나로만 한다.

2. ✅ **real serving용 canonical robot demo 라벨/영구 artifact 완료**
   - 사용자가 지정한 현재 fake-data acceptance는 완료됐다.
   - fake는 bounded `--synthetic-e2e`에서만 live learner에 허용되고 production robot-data scope에는 의도적으로 차단된다.
   - 사용자가 `take_23` 제외 23개를 success로 승인했고 2,037-transition 영구 artifact를 생성했다. laptop3/Kanu strict-load와 SHA 일치까지 확인했다.
   - **`--synthetic-e2e`로는 우회할 수 없다.** 그 모드는 서버가 `allowed_run_ids`를 화이트리스트로 강제하는데, `run_remote_rlpd_actor.py`는 **`--run-id`를 노출하지 않고**(2026-07-29 재확인: CLI에 없음) `remote_actor.py`의 `run_id or uuid.uuid4().hex`로 매번 새로 만든다. 실제 robot actor의 run_id를 서버에 미리 등록할 방법이 없어 `FailedPreconditionError`로 거부된다.
   - **생산 경로는 두 개다.** 기존 recorder take는 `scripts/convert_recorded_takes_to_demo.py`로 변환한다(`RECORDED_TAKE_DEMO_CONVERSION_KO.md`). 새 actor 녹화는 `remote_actor.py::_dump_data`가 `--checkpoint-path`를 받아 `<ckpt>/actor_data/<run_id>/replay/data_<step>.pkl`을 남기고, `load_demo_pickles`가 그 `{"meta":…, "transition":…}` 형식을 바로 받는다.
   - 🪤 **선결 조건: `buffer_period = 0`**(`ur_experiments/cube_in_cup.py` 의 `buffer_period`, 현재 :271). 0이면 `_dump_data` 가 호출되지 않아 `--checkpoint-path` 를 줘도 pickle이 **하나도** 안 쓰인다. 이 값을 뒤집는 CLI 플래그도 없다(§11.7).
   - 🪤 **`--mock-policy-noise` 로 채운 데이터는 demo로 쓰면 안 된다.** `1a4f93d` 이후 모든 전이에 `meta.policy_actions_synthetic=true` 가 박힌다.

3. **실제 robot/task/camera E2E**
   - task config의 `GRASP_PENALTY`(`-0.02`), action convention, camera key/shape/timing을 확인한다. → task config는 `ur_experiments/cube_in_cup.py`로 확정됐고 `GRASP_PENALTY`(`cube_in_cup.py::CubeInCupConfig.GRASP_PENALTY`, 현재 :229)도 서버 기본값과 일치한다(§11.2).
   - robot laptop → SSH tunnel → GPU 서버 → classifier/replay → policy response는 **2026-07-29 실제 production actor에서(kanu 상대로) 짧게 검증 완료**했고, **2026-07-31 서버 이전 뒤 `junhyeong_ai` 상대로 조작자 실기 세션에서 다시 통과**했다(replay **316** / intervention **210**). 후자가 §5.8의 합성 시험이 못 덮은 두 구멍 — **실제 classifier sidecar 추론**과 **`intervened=1` ingress** — 을 닫는다.
   - ⚠️ **~~카메라가 현재 물리적으로 차단 상태다~~ → 2026-07-28에 해소됐다**(§11.10 — 허브 고장이 아니었다). §11.6은 07-27 시점의 기록으로만 읽어라.
   - 카메라를 켠 canonical observation과 sidecar의 실물 왕복은 통과했다. 남은 것은 정지 게이트(`stationary_speed_max`)의 장시간 분포, ~2 Hz 첨부 latency, 팔 가림 조건의 verdict 정합이다.

4. **continuous learner degraded monitoring**
   - 현재 learner fault는 stdout/JSONL event로만 확인하며 gRPC health는 last-known-good serving 때문에 ready일 수 있다.
   - heartbeat/status/alert 또는 운영 supervisor 정책이 필요하다.

### P1 — production 장기 운용 전 보강

1. actor가 server model/reward/schema identity를 pin하는 기능은 구현됐다. bounded synthetic acceptance는 server가 exact actor/run allowlist를 강제한다. 반면 production robot scope에는 아직 actor ID/task/run reverse allowlist나 별도 인증이 없으며, 현재 production 보안 경계는 Kanu loopback + SSH access다.
2. run fingerprint에 exact source commit/upstream revision/task identity를 최종 포함해야 한다.
3. JAX update가 native backend에서 영구 hang하면 graceful shutdown이 worker join을 계속 기다릴 수 있다.
4. replay/intervention은 RAM-only라 restart 후 training distribution이 달라진다.
5. sampler NumPy RNG와 RAM buffer는 checkpoint에 없으므로 whole-process bitwise continuation은 아니다.
6. W&B online 인증/업로드와 장시간 offline disk 증가를 검증하지 않았다.
7. default 50-step publish/5,000-step checkpoint까지 실제 GPU bounded run이 필요하다.
8. checkpoint 약 305 MiB가 계속 누적되므로 run별 disk capacity/retention 운영 절차가 필요하다. 자동 pruning은 추가하지 않는다.
9. runtime dependency validator는 JAX/JAXLIB/Flax/Distrax/TFP와 W&B만 검사한다. NumPy/Optax/protobuf/grpcio/Orbax는 lock/known environment에는 고정돼 있지만 같은 fail-closed preflight에 아직 포함되지 않았다.

## 8. 권장 다음 순서

*(2026-07-29 재작성. 07-28 판본의 1~4번은 전부 완료됐다 — 아래 「완료된 항목」 참조.)*

> "지금 무엇을 하면 되는가"의 정본은 [HANDOFF_NEXT_SESSION_KO.md](./HANDOFF_NEXT_SESSION_KO.md)다. 이 절은 그것을 남은 차단점(§7)에 대응시킨 요약이다.

### ✅ 완료된 항목 (07-28 판본의 A/B 갈래)

| 07-28 판본의 지시 | 결과 |
| --- | --- |
| 1. move-then-hold로 frame-map 재측정 | ✅ 완료 — 매핑 = **단위행렬**, 포화 제외 잔차 0.093 (§11.9) |
| 2. actor에 `--deadman {topic,spacebar}` 배선 | ✅ 완료 (`607e541`), 기본 `topic` (§11.7) |
| 3. `DRY_RUN` CLI 해제 + `--arm --scale 0.25` 첫 실물 구동 | ✅ 완료 — 2026-07-28, 개입 64/100, `held=0` (§11.9). **단 workspace box는 이때 검증되지 않았다** — `run_real_hil.py` 경로에서 박스가 비활성이기 때문이다(§11.7-3) |
| 4. USB 허브 `4-4` 고장 복구 | ✅ 불필요했다 — **고장이 아니었다.** 재연결로 해결 (§11.10). ⚠️ 단 07-29 11:14에 같은 EPROTO가 재발했으므로 A-1 전에 확인할 것 |

### 남은 순서

**A. 실기 확인 (laptop)**

1. **카메라를 먼저 확인한다.** 07-29 11:14에 EPROTO가 재발했고(§11.10), `launch_cameras.sh` 의 기본 시리얼이 이 PC가 한 번도 본 적 없는 쌍으로 바뀌어 있다. **팔을 한 번 흔들어 cam2 창을 보면 개체 배정까지 같이 확정된다.**
2. **ZMQ classifier 뷰어로 실기 분포를 본다.** 텔레옵하면서 `p(success)` 곡선을 관찰한다. 이 경로는 크롭 없이 리사이즈하므로 **학습 분포와 일치**하고(§12.1의 결함과 무관), 카메라 조건이 흔들린 지금 가장 싸고 직접적인 확인이다.
   - `serl_ur_infra/run_remote_reward_classifier_server.sh` (GPU 서버, port 5594) + `ros2_ur_ws/run_remote_classifier_viewer.sh` (laptop)
   - ⚠️ **호스트가 바뀌었다.** 이 두 스크립트는 kanu 시절에 검증된 경로다. 현행 서버는 `junhyeong_ai`이고 체크포인트는 `~/hil-serl-data/classifier_ckpt/checkpoint_150`이다. **새 서버에서 이 뷰어 경로를 돌린 적은 아직 없다** — 돌릴 때 `REWARD_CLASSIFIER_CHECKPOINT`로 그 경로를 명시할 것.
3. ✅ **actor entrypoint(`run_remote_rlpd_actor.py`) 첫 실물 E2E 완료.** 다음 실행은 정상 3-CLI로 재현하고, MANUAL/AUTO verdict와 publish 순간 latency를 기록한다.
4. **workspace box를 실제로 무장한 채 돌린다.** `run_real_hil.py` 경로에서는 박스가 꺼져 있고, DRY RUN 300 스텝 중 **241 스텝(80%)** 이 측정 박스 밖이었으며 최대 **73.9 cm** 이탈했다(§11.7-3).

**B. classifier 크롭 정합 (§12.7) — ✅ 2026-07-29 완료. 5·6번은 실행하지 마라**

5. ⛔ ~~**크롭을 넣어 classifier를 재학습한다.**~~ — **취소됐다.** 채택된 것은 재학습이 아니라 **sidecar 분리**다(§12.1, §12.7). 파이프라인 인자(`--cam1-crop 340,20,990,670`, `--cam2-crop 420,0,1140,720`)와 "07-27 학습은 150 epoch에 46초"는 (a)를 나중에 다시 볼 사람을 위한 **자료로만** 남긴다. **재학습하면 §12.3과 threshold 문서의 측정값이 전부 무효가 된다** — 그걸 피한 것이 이번 결정의 요지다.
6. ✅ ~~**orbax 디렉터리 digest/load를 구현한다**~~ — **완료 (G19).** `directory_sha256()` 이 들어갔고 기본 SHA 상수 2개도 `512b6575…`(checkpoint_150, 14파일)로 교체됐다. §7 P0-0, §12.6-3.
   - **대신 남은 것:** `stationary_speed_max`(현재 `0.05 m/s`, **코드가 스스로 PLACEHOLDER라고 표시**)를 녹화 take에서 실측할 것.
   - **그리고 sidecar 경로 실기 첫 투입** — 위 3번과 같은 세션에서 처음 돈다.

**C. 합류 후 — canonical robot demo artifact**

7. ✅ 기존 2026-07-20 take의 사람 승인 canonical pickle 생성·복사·strict-load 완료. **2026-07-31 현재 같은 SHA로 3벌**(laptop3 `~/hil-serl-artifacts/demos/`, kanu, `junhyeong_ai:~/hil-serl-data/demos/`). 새 데이터를 actor 형식으로 다시 녹화할 때만 `buffer_period > 0` 배선을 추가한다.
8. ✅ production learner 배포 완료(2026-07-30 kanu → **2026-07-31 `junhyeong_ai`로 이전**). 정상 재접속은 raw Python CLI가 아니라 3-CLI의 server terminal에서 `cd /home/laptop3/gello_software/ros2_ur_ws && ./run_hil_server.sh`를 실행해 healthy process를 재사용한다(§11.5). **환경변수 override는 붙이지 않는다 — 기본값이 이미 새 서버다.**
9. production-scope 5,000-step bounded learner run을 완주하고, checkpoint와 장시간 RSS/VMS/GPU memory/compile/contention, W&B offline disk 증가를 계측한다. 🔴 **이 항목은 서버 이전으로 원점에서 다시 시작한다** — kanu에서 도달한 최고 learner step은 301이고 `checkpoint_period`가 5,000이라 **kanu run root 8개 전부 `checkpoints/`가 비어 있었다.** 새 호스트는 GPU가 1장뿐이고 RAM이 60 GB(available 약 37 GB)이므로 ring/RSS 예산도 다시 세운다.
10. learner fault heartbeat와 shutdown escalation을 보강한 뒤 continuous mode를 승인한다.

## 9. 재현 명령

상세 learner 기동 명령은 [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)에 있다.
⚠️ **그 런북은 파일명도 본문도 kanu 기준이다.** 호스트·GPU 인덱스·데이터 경로는
[`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md)가 우선한다.

**로컬 빠른 suite — 이 명령을 그대로 쓸 것 (📌 2026-07-30 `595 passed, 11 skipped, 1 xfailed` 실측, `gello-hil-actor`, 다른 세션 작업 2파일 `--ignore`):**

```bash
cd /home/laptop3/gello_software
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
  -p no:cacheprovider serl_ur_infra/tests \
  --ignore=serl_ur_infra/tests/test_operator_session.py \
  --ignore=serl_ur_infra/tests/test_remote_actor_operator_session.py
# -> 595 passed, 11 skipped, 1 xfailed   (📌 2026-07-30 / gello-hil-actor.
#    다른 세션 작업 제외: --ignore=tests/test_operator_session.py
#                        --ignore=tests/test_remote_actor_operator_session.py
#    트리 전체는 605 / 11 / 1 — 아래 참고)
```

> 🚧 **범위를 반드시 같이 적어라.** `595` 는 **다른 세션의 P3 operator 상태 기계 테스트 2파일을 제외**한 값이다. 같은 checkout에서 다른 세션이 동시에 작업 중이라 **트리 전체 수(605)는 변동 중이다.** 커밋 `4197f5b` 시점 값은 **`579 / 11`**이었다. **`11 skipped` 는 `gello-hil-actor` 에서 확정이다** — `43ba314` `333` 이후 skip 수는 한 번도 바뀌지 않았다.
> ⚠️ **`1 xfailed` 를 빼지 마라.** `08` **G27**(line search 축소가 저장 액션에 반영되지 않음)의 `strict=True` 표식이고, 그 갭이 고쳐지면 **XPASS 로 터진다** — 그게 의도된 알림이다.
> ⚠️ **인터프리터를 바꾸면 수가 바뀐다:** `hilserl`(numpy 1.26.4, jax) 트리 전체는 **`645 passed, 4 skipped, 1 xfailed`** 다. **인터프리터와 범위를 안 적은 passed 수는 판정에 쓸 수 없다.**
>
> 🪤 **`third_party/hil-serl/serl_launcher` 를 `PYTHONPATH` 에서 빼면 passed 수가 조용히 떨어진다.** `43ba314` 기준으로는 `300 passed, 13 skipped` 였다. 사라지는 것이 하필 `test_cube_in_cup_config.py` 와 `test_frame_wrappers.py` 이고, skip 사유가 "submodule is not checked out"이라고 **거짓말한다.** 새 worktree에서는 서브모듈 미초기화로 296/17이 된다. **기준선보다 낮은 passed 수와 11이 아닌 skipped 수는 둘 다 이 증상이다. 녹색이 아니라 passed 수를 볼 것.**
>
> 🪤 **시스템 `python3` 로 gRPC 코드를 실행하지 말 것** — 시스템 grpcio가 고장나 오류 없이 100% CPU로 무한 정지한다(§11.8).

UR/GELLO suite (2026-07-29 `436 passed` 확인):

```bash
cd /home/laptop3/gello_software
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -m pytest -q -p no:cacheprovider -p no:launch_testing \
  ros2_ur_ws/src/ur_gello_bringup
# -> 436 passed
```

아래 opt-in JAX test 3종은 learner 전용 venv `/tmp/gello-hil-rl-learner-venv` 를 쓴다(2026-07-29 존재 확인). ⚠️ `/tmp` 라 리부트하면 사라진다. **마지막 pass 확인은 2026-07-27 `248255f` 이고 `43ba314` 에서 재실행하지 않았다** — 아래 명령의 `PYTHONPATH` 에 `serl_launcher` 가 없는 것도 07-27 형태 그대로다.

> 🔴 **2026-07-31 확인: 그 venv는 이제 없다.** laptop3가 리부트해 `/tmp/gello-hil-rl-learner-venv`
> 가 사라졌으므로 **아래 세 명령을 그대로 복사하면 `No such file or directory` 로 죽는다.**
> `/tmp` 경고가 실현된 것이지 서버 이전 때문이 아니다.
> 대체 후보는 laptop3의 `/home/laptop3/venvs/hilserl/bin/python` 이다 — 2026-07-31 실측으로
> `jax/jaxlib 0.5.3, flax 0.10.5, distrax 0.1.5, tfp 0.25.0` 이 확인됐고
> `validate_learner_dependencies()` 를 통과한다
> ([`CONTROL_RATE_AND_ASYNC_PLAN_KO.md`](./CONTROL_RATE_AND_ASYNC_PLAN_KO.md) §9.2).
> 🟡 **다만 이 opt-in 3종을 그 인터프리터로 돌려 본 적은 없다 — 미검증이다.**

실제 agent opt-in test:

```bash
cd /home/laptop3/gello_software

RUN_HIL_SERL_ACTUAL_CHECKPOINT=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
MPLCONFIGDIR=/tmp/gello-hil-production-matplotlib \
PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests/test_actual_agent_checkpoint_integration.py
```

실제 frozen-trunk raw/cached agent opt-in test:

```bash
cd /home/laptop3/gello_software

RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
MPLCONFIGDIR=/tmp/gello-hil-production-matplotlib \
PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests/test_actual_frozen_trunk_feature_agent.py
```

실제 localhost gRPC fake learning/resume opt-in test:

```bash
cd /home/laptop3/gello_software

RUN_HIL_SERL_FAKE_E2E=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
MPLCONFIGDIR=/tmp/gello-hil-production-matplotlib \
PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests/test_actual_fake_data_e2e_learning.py
```

## 10. handoff 불변식

- fake demo는 construction `--dry-run` 또는 bounded `--synthetic-e2e` 전용이다.
- production robot serving에서 fake marker를 제거하거나 우회하지 않는다.
- synthetic E2E는 target 1..10이자 restored step +1, replay capacity 100 이상, exact 100 inserts, all-synthetic demo, exact actor/run allowlist, timeout 1..1,800초, batch 256/training-starts 100/CTA 2를 유지한다.
- synthetic-only policy model ID와 cleanup 후 checkpoint roundtrip/trunk invariant pass gate를 우회하지 않는다.
- synthetic E2E checkpoint와 production robot checkpoint의 execution-scope fingerprint를 섞지 않는다.
- loopback bind와 SSH tunnel 경계를 유지한다.
- GPU 서버에서는 GPU backend를 명시하고 CPU fallback을 허용하지 않는다(`--require-jax-backend gpu`). 호스트가 바뀌어도 이 불변식은 그대로다.
- `third_party/hil-serl`, proto, generated binding을 직접 수정하지 않는다.
- checkpoint를 덮어쓰기·삭제·pruning하지 않는다.
- resume는 fingerprint/counter/RNG 검사를 우회하지 않는다.
- learner fault가 나도 last-known-good policy를 오염시키지 않는다.
- replay가 RAM-only라는 사실을 checkpoint resume와 혼동하지 않는다.
- replay/demo에 raw image, GAP512, 또는 trainable 256-D head 출력을 저장하지 않는다.
- verified frozen trunk 직후 `float32 (1,4,4,512)` current/next 계약과 augmentation `none`을 유지한다.
- online/target trunk invariant 검사와 target repin을 우회하지 않는다.
- raw/random-crop checkpoint를 feature/no-aug lineage에 resume하지 않는다.
- **reward threshold는 fingerprint에 들어간다.** 값을 바꾸면 기존 lineage는 fail-closed로 거부된다 — 그건 의도다(§4.7).
- 🔴 **`ACTION_SCALE` 은 fingerprint에 없다.** 다른 `ACTION_SCALE` 로 녹화한 데이터를 이어받아도 **경고가 나오지 않는다.** resume 전에 손으로 대조할 것(§11.5).
- **`--mock-policy-noise` 로 만든 전이를 demo로 쓰지 않는다.** `meta.policy_actions_synthetic=true` 가 박혀 있다(§7 P0-2).

reward classifier 쪽 불변식 (§12):

- **크롭 불일치를 `IMAGE_CROP` 제거로 "고치지" 않는다.** 크롭 값은 데이터셋 실측이고 정책이 1차 소비자다. ~~해법은 크롭에 맞춘 **재학습**이다(§12.7).~~ → **2026-07-29 정정: 해법은 재학습이 아니라 sidecar 분리였고, 이미 적용됐다**(§12.1). `IMAGE_CROP` 을 건드리지 말라는 부분은 **그대로 유효하다** — 오히려 이번 변경의 전제다.
- **classifier에 크롭된 정책 관측을 먹이지 않는다.** 서버는 오직 sidecar 프레임만 분류한다(`validate_classifier_frames`). 정책 관측을 분류기에 넘기는 코드를 다시 만들면 recall@0.85 100% → 33.3% 로 되돌아간다.
- **sidecar를 크롭하지 않는다.** sidecar가 존재하는 이유 전체가 "무크롭 시야"다.
- **`reward_model_id` 와 `CLASSIFIER_INPUT_ID` 를 따로 움직이지 않는다.** 둘이 어긋나면 actor↔server 조합이 핸드셰이크에서 안 걸러지고, 양쪽이 서로 다른 픽셀로 reward를 계산한 채 세션이 끝까지 간다.
- **`checkpoint_sha256()` 의 디렉터리 지원과 pin 상수를 따로 배포하지 않는다.** 해싱만 고치면 은퇴한 recall-0% 체크포인트로 서버가 조용히 기동한다(§12.6-3).
- **`decode_classifier_image()`(ZMQ 뷰어)를 이 결함의 원인으로 지목하지 않는다.** gRPC 경로와 호출 관계가 없고, 그쪽 용도에서는 무크롭이 올바르다(§12.1). *(sidecar의 `decode_classifier_frames()` 는 이 함수와 **같은 레시피를 의도적으로 복제**한 것이다 — 뷰어 쪽이 ROS ament 패키지 안이라 Kanu learner의 `PYTHONPATH` 에 없기 때문이고, `tests/test_classifier_sidecar.py` 가 둘의 일치를 강제한다.)*
- **classifier 성능 수치를 split/전처리 표기 없이 옮기지 않는다.** 같은 체크포인트가 split에 따라 100.0% 와 86.8% 를 낸다(§12.3).

robot actor 쪽 불변식 (§11):

- GELLO는 수동 read-only 리더다. **Dynamixel에 절대 토크를 걸지 않는다.**
- `TCP_POSE_SOURCE` / `TCP_OFFSET_XYZ_RPY` / `ABS_POSE_LIMIT` 은 결합된 한 세트다. 하나만 바꾸지 않는다.
- RViz에서 좌우/전후가 뒤집혀 보인다는 이유로 `wrappers.py` 의 X/Y 부호를 뒤집지 않는다. 카메라 방위각 artifact이며 `R_align = I` 다 — **2026-07-28 실측으로 재확인됐다**(§11.9).
- 프레임 통일 명목으로 wrench(`tcp_force`/`tcp_torque`)를 회전시키지 않는다. upstream도 건드리지 않는다.
- `RelativeFrame.step` 의 두 시점(step 이전 행렬로 action 변환, 이후 행렬로 observation 변환)을 하나로 합치지 않는다.
- `wrapped_nearest` 가 elbow를 unwrap하지 않는 것은 의도다.
- gRPC 코드를 시스템 `python3` 로 실행하지 않는다.
- gripper는 `Ctrl-C` 로 종료한다. `kill -9` 하지 않는다.
- headless 세션에서 `ur_play`/`ur_load`/`ur_stop` 을 실행하지 않는다. `ur_resend` 만 쓴다.
- **RealSense 시리얼을 코드에 새로 하드코딩하지 않는다.** 이 리그에서 어느 쌍이 열거되는지가 이미 두 번 뒤집혔고, 없는 시리얼 바인딩은 **에러가 아니라 조용한 "프레임 없음"** 으로 나타난다. `43ba314` 의 버스 해석 방식을 따를 것(§11.10).

---

## 11. robot actor / 하드웨어·통신 현황 (2026-07-27 착수 → 2026-07-29 머지 완료)

이 절은 laptop 쪽 robot actor 작업의 현황이다.

> **⚠️ 작업 위치가 바뀌었다.** 이 절의 07-27 판본은 "작업 위치는 worktree `/home/laptop3/gello_worktrees/hil-hardware-comms`, branch `test/hil-hardware-comms`, HEAD `ee3240e`. canonical checkout은 건드리지 않았다"였다. **그 작업은 2026-07-29 merge `3f199d4` 로 canonical branch에 들어왔다.** 이제 이 절이 서술하는 코드는 전부 `/home/laptop3/gello_software` 에 있다. worktree는 `1a4f93d` 에서 멈춘 채 디스크에 남아 있을 뿐이다.

`5fb716b..3f199d4` = **23 commit**. `5fb716b..43ba314` 누적 diff **70 files / +9,903 −395**.

| commit | 날짜 | 내용 |
| --- | --- | --- |
| `cc13710` | 07-27 | `RecordEpisodeStatistics` 를 gymnasium 패키지 루트에서 import |
| `ebc77f7` | 07-27 | Franka task registry 없이도 actor가 기동하도록 |
| `14c57d8` | 07-27 | canonical observation을 만드는 wrapper 이식 |
| `50f65de` | 07-27 | UR7e task config와 `CONFIG_MAPPING` 추가 |
| `a4a2b0a` | 07-27 | robot state stream을 기다린 뒤 `__init__` 반환 |
| `d49d0f6` | 07-27 | workspace box 활성화 + reset이 먼 길로 돌지 않게 |
| `ee3240e` | 07-27 | z 바닥과 reset gate 정정, frame test를 실제로 물게 만듦 (§11.3) |
| `a5f9890` | 07-28 | 속도 3층을 검증된 EEF 텔레옵 한계에 정렬 (§11.9) |
| `607e541` | 07-28 | actor를 실기에서 돌 수 있게 — `--arm` / `--deadman` / 카메라 첫 프레임 대기 (§11.7) |
| `c069e79` | 07-28 | 다른 노드가 컨트롤러를 잡고 있으면 `--arm` 거부 |
| `53d5cf6` | 07-29 | reward threshold 0.85 → 0.5 |
| `1a4f93d` | 07-29 | `--mock-policy-noise` 실행에 식별자를 남김(`meta.policy_actions_synthetic`) + 카메라 대기 예산 분리 |
| **`3f199d4`** | **07-29** | **merge — 위 계열을 canonical branch로** |
| `1b02857` | 07-29 | reward threshold 0.5 → **0.2** (§4.7) |
| `43ba314` | 07-29 | RealSense 시리얼을 실제 USB 버스에서 해석 (§11.10) |

*(문서 커밋 `22b0df2` / `30d21c4` / `db01e64` / `142ea2a` / `921a155` / `af8b576` / `9934533` / `d65913a` / `3ff5f80` 은 생략했다. 전체 목록은 `git log --oneline 5fb716b..3f199d4`.)*

테스트: **2026-07-29 `43ba314` 에서 `333 passed, 11 skipped`** (역사값 — 현행 기준선은 §1의 계보. §5.1, §9). *(07-27 `ee3240e` worktree 시점 값은 `332 passed, 11 skipped, 1 xfailed` 였다. `607e541` 이 그 strict xfail을 제거하고 회귀 가드로 전환했다 — §11.7-2.)* UR/GELLO suite `436 passed`.

### 11.1 `5fb716b` 에서 actor가 실행 불가능했던 4개 층

전부 해소했다. 어느 하나만 고쳐도 다음 층에서 다시 멈췄다.

1. **wrapper 3종 부재** — `RelativeFrame`, `Quat2EulerWrapper`, `ChunkingWrapper` 가 없었다. 이것들이 없으면 state가 canonical 19-D가 되지 못한다(quat 7-D → euler 6-D 변환이 20-D를 19-D로 만든다). upstream에서 import할 수 없었던 이유는 §11.2.
2. **task config 부재** — `CONFIG_MAPPING` 에 UR7e task가 없어 `--config` 로 지정할 대상 자체가 없었다.
3. **import 시점 실패 2건** — actor가 upstream Franka registry를 무조건 import해서 `ModuleNotFoundError: jax`, 그리고 gymnasium 1.0에서 삭제된 `gymnasium.wrappers.record_episode_statistics` 경로.
4. **DDS discovery race** — `UR7eEnv.__init__` 이 `/joint_states` 도착 전에 반환해서 첫 reset이 `no /joint_states — cannot reset` 으로 죽었다. discovery에 약 1.0 s 걸린다.

### 11.2 이식한 wrapper와 task config

**wrapper를 import하지 않고 이식한 이유**: upstream `franka_env.envs.relative_env` 는 legacy `gym` 패키지를, `franka_env.envs.wrappers` 는 module scope에서 `pyspacemouse`/`hidapi`/`pyrealsense2` 를 끌고 온다. `serl_launcher.wrappers.chunking` 은 `jax.tree_map` 하나 때문에 `import jax` 를 한다. robot laptop은 의도적으로 jax가 없는 순수 rclpy/numpy 프로세스여야 한다. 동작은 upstream을 줄 단위로 옮겼다 — UR7e run과 Franka run이 같은 observation/action 의미를 보게 하는 것이 이 파일의 존재 이유이므로 그대로 유지할 것.

- `ur_env/envs/frame_wrappers.py` (208줄) — `RelativeFrame` + `Quat2EulerWrapper`. `ChunkingWrapper` 는 `ur_env/envs/chunking.py` (105줄), `obs_horizon=1` 외에는 `NotImplementedError`.
- `RelativeFrame` 의 타이밍 미묘함을 upstream 그대로 보존했다: action과 `info["intervene_action"]` 은 **step 이전** 행렬로 변환하고, 그 다음에 반환 observation용으로 행렬을 갱신한다. 두 용도는 의도적으로 한 제어 주기 떨어져 있다. 하나의 행렬로 "고치지" 말 것.
- `Quat2EulerWrapper` 는 upstream과 달리 observation space를 deep-copy한 뒤 수정한다. upstream은 공유 객체를 in-place로 바꿔서 내부 env가 만들지 못하는 space를 광고하게 된다.
- `tcp_force`/`tcp_torque` 는 건드리지 않는다. upstream도 그렇다. 우리 `/force_torque_sensor_broadcaster/wrench` 는 `tool0` 에, libfranka `K_F_ext_hat_K` 는 stiffness(EE) frame에 publish하므로 양쪽 다 tool frame wrench + tool frame `tcp_vel` 로 끝난다. **프레임 통일 명목으로 wrench를 회전시키지 말 것.**

**task config `serl_ur_infra/ur_experiments/cube_in_cup.py`** (2026-07-29 기준 357줄; 07-27 이식 당시 288줄). 패키지 이름이 `experiments` 가 아니라 `ur_experiments` 인 것은 upstream `examples` 가 sys.path에서 더 앞이기 때문이다. 미측정 값은 `UNSET = None` sentinel로 두고 `__init__` 에서 예외를 던진다 — 0으로 조용히 떨어지면 zero-volume workspace나 테이블을 가로지르는 수평 자세가 된다.

| 항목 | 값 | 근거 |
| --- | --- | --- |
| `RESET_JOINTS` | `[3.1382, −1.5276, 1.7168, −1.7592, −1.5216, −3.1331]` | cube_in_cup 데이터셋 시작 자세. `fk` = `(0.5036, 0.1365, 0.4138)`, box 내부 |
| `RESET_MAX_DIST_RAD` | `0.9` | §11.3 |
| `TCP_POSE_SOURCE` / `TCP_OFFSET_XYZ_RPY` | `"driver"` / `[0]*6` | **flange frame. 결합된 한 세트** |
| `ABS_POSE_LIMIT_LOW` | `[0.375, −0.229, 0.185, 2.60, −0.30, 1.10]` | §11.3 |
| `ABS_POSE_LIMIT_HIGH` | `[0.642, 0.272, 0.550, π, 0.35, 2.20]` | X/Y는 데이터 범위의 1.5배(탐사 여유), Z 상한은 데이터 최대 |
| `IMAGE_CROP` (`::IMAGE_CROP`, 현재 `:217-221`) | cam1 `[20:670, 340:990]` (650×650) / cam2 `[0:720, 420:1140]` (720×720) | **cam2가 손목 카메라**. ✅ **2026-07-29: 이 값은 옳고 그대로 간다.** 예전 `KNOWN CONFLICT` 주석(classifier 학습 전처리와 불일치)은 **`G15 ... RESOLVED -- BY DECOUPLING` 으로 대체됐다** — classifier는 이제 이 이미지를 아예 보지 않는다; §12.1 |
| `CLASSIFIER_SIDECAR` (≈`:303`) | `enabled=True`, `interval_steps=5`(10 Hz → ~2 Hz), `stationary_speed_max=0.05 m/s`, `escalate_probability=0.05` | classifier에 **무크롭** 프레임을 보내는 주기 정책. env config가 아니라 task config에 있다 — 소비자가 actor 루프뿐이기 때문. 🚧 `stationary_speed_max`는 실기 장시간 분포를 더 재야 한다. `escalate_probability`는 난수 확률이 아니라 **"매 스텝 분류로 전환하는 확률 임계"**이며 현행 reward threshold 0.5보다 충분히 낮다 |
| `MAX_EPISODE_LENGTH` | `100` (HZ=10 → 10 s) | |
| `GRASP_PENALTY` (`::GRASP_PENALTY`, 현재 `:229`) | `−0.02` | learner 서버 기본값과 일치해야 함 |
| `DRY_RUN` (`::DRY_RUN`, 현재 `:233`) | `True` | 파일 기본값은 여전히 `True` = 모든 로봇 명령 publish 차단. **`--arm`(`607e541`)이 런타임에 이걸 해제한다** — `_build_actor_environment` **이전에** 적용돼야 한다(§11.7) |
| `buffer_period` (현재 `:271`) | `0` | demo pickle이 하나도 안 쓰인다. canonical demo 녹화의 선결 조건(§7 P0-2) |

`TCP_POSE_SOURCE`/`TCP_OFFSET_XYZ_RPY`/`ABS_POSE_LIMIT` 세 설정은 **반드시 함께 움직인다**. box를 flange frame에서 측정했으므로, 실제 0.174 m tool offset을 넣거나 pose source를 `fk` 로 바꾸면 관측 pose가 오류 없이 17.4 cm 이동해 z 바닥이 테이블 아래로 내려간다. `test_pose_reference_point_stays_coupled` 가 이를 고정한다.

`get_environment(classifier=True)` 는 예외를 던진다 — reward는 서버가 결정한다.

### 11.3 검수로 뒤집힌 값 2개

두 값 모두 처음 산출값이 틀렸고, 독립 검수에서 잡혀 `ee3240e` 로 정정했다.

**z 바닥 `0.1785` → `0.185`.** `0.1785` 는 테이블 표면이 아니라 **충돌 깊이**였다. 그 최소값을 만든 40개 샘플이 전부 take_11이고, 해당 구간의 gripper 개구가 0.047–0.176(빈 손), `fz` 가 −133.2 ~ −24.2 N이다. 즉 빈 그리퍼로 테이블을 눌러 박은 자세다. 접촉 없는 샘플만 필터링하면 표면은 `0.1808`. 따라서 바닥은 그 위여야 하고, 동시에 가장 낮은 성공 그랩 `0.1941` 아래여야 과제가 도달 가능하다. `test_z_floor_clears_the_contact_free_table_surface` 가 양쪽을 고정한다.

**`RESET_MAX_DIST_RAD` `0.5` → `0.9`.** `0.5` 는 23개 take 중 **16개**의 정상 에피소드 종료 자세를 거부한다. 각 take 마지막 프레임에서 `RESET_JOINTS` 까지의 branch-safe 거리는 중앙값 0.615 rad, 최대 0.774 rad다. 게이트가 정상 종료를 막으면 매 에피소드가 수동 개입을 요구한다.

**reset branch-cut (H3).** `go_to_reset` 은 `ur_kin.wrapped_nearest(q, ref)` 를 쓴다. 실기에서 wrist_3가 `+3.1795`, 목표가 `−3.1331` 인 상황이 나왔다 — 실제로는 0.029 rad 떨어져 있는데 순진한 차분은 6.31 rad다. 게이트가 없으면 reset이 wrist_3를 **한 바퀴 통째로** 돌려 2F-85 tool-comm 케이블을 감는다. `wrapped_nearest` 는 elbow(index 2, ±π 한계)를 **의도적으로 unwrap하지 않는다**.

`clip_safety_box` 는 부호를 보존하는 abs-clip으로 구현했다. 두 게이트 모두 단위 테스트는 통과했고(`tests/test_clip_safety_box.py`, `tests/test_reset_branch_cut.py`) **실기 검증은 2026-07-29 현재도 아직**이다.

> **⚠️ 07-27 판본은 그 이유를 "`DRY_RUN=True` 라서"라고 적었다. 그건 07-28에 바뀌었다.** 팔은 `--arm` 으로 실제로 움직였다(§11.9). 그런데 **게이트는 여전히 실기에서 작동한 적이 없다** — 이유가 바뀌었을 뿐이다. 07-28 세션은 `run_real_hil.py` 경로였고 거기서는 `DefaultUR7eEnvConfig.ABS_POSE_LIMIT_*` 가 zeros(`config.py::DefaultUR7eEnvConfig.ABS_POSE_LIMIT_LOW/HIGH`, 현재 :84-85)라 코드가 0-부피 박스를 감지해 박스를 **끈다**. 실측 박스는 `cube_in_cup.py` 에만 있다. 자세한 것은 §11.7-3.

### 11.4 실기 검증된 것 (2026-07-27 측정)

- **2F-85 gripper** — Modbus RTU over UR tool-comm `:54321`. 개폐 및 방향을 눈으로 확인했다. 이로써 crush-hazard 게이트가 닫혔다. 로봇 전원이 켜져 있어야 하고(tool 24 V), `:54321` 은 클라이언트를 **하나만** 받는다. 반드시 `Ctrl-C` 로 종료할 것 — `kill -9` 는 FIN-WAIT-2로 재접속을 30–45초 굶긴다.
- **GELLO leader** — 7개 모터 전부 baud 57600에서 응답. **토크는 항상 OFF를 유지한다.**
- **laptop → SSH tunnel → Kanu 100-step 왕복** — `replay_insert_count:100`, `state_shape:[8,1,19]`, observation schema hash `3459098d…` 가 양쪽 일치. **상대는 zero-action receive server였고 actor는 fake-env였다** — 카메라도 팔도 개입하지 않은 전송 경로 단독 검증이다.
- **지연/대역폭** *(측정 조건: 2026-07-27, WiFi, fake-env actor, zero-action 서버, 10 Hz 100 스텝)* — RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**, step당 96.1 KiB → 7.9 Mbit/s. 병목은 WiFi 대역폭(약 13 Mbit/s)이다. **p99가 10 Hz 예산 100 ms를 거의 다 쓴다.** 유선 재측정 권장. ⚠️ 실제 정책 추론이 붙으면 서버 측 계산 시간이 더해지므로 이 수치는 **하한**이다.

### 11.5 Kanu 정책 서빙 — 2026-07-30 production learner (🗄️ 호스트 이전됨)

> # 🗄️ 이 절 전체가 **kanu 기준**이다 (2026-07-30 19:41 KST 스냅샷)
>
> **2026-07-31에 learner가 `junhyeong_ai`로 옮겨졌다.** 아래의 PID `1112465`, **GPU 5**,
> run root `cube_in_cup_manual_schema3_thr05_20260730_1715`, checkout
> `~/gello_software_hil_current`, 그리고 환경 점검 표(8× A4000 / RAM 251 GiB / disk 95%)는
> **전부 kanu에 관한 사실**이므로 그대로 둔다. **현행 호스트 상태를 이 절에서 읽지 마라** —
> `./run_hil_server.sh --check`를 돌리거나
> [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md)를 본다.
>
> 이전 뒤에도 **바뀌지 않은 것**: 서빙 스크립트(`run_rlpd_learner_server.py`), 단일 process
> 조립, loopback bind + SSH tunnel, model ID, 학습 시작 게이트(replay ≥ 100 AND demo ≥ 1),
> `XLA_PYTHON_CLIENT_PREALLOCATE=false` 요구, `--resnet-cache`/`preflight_checkpoint_run`
> 규칙, actor가 pin해야 하는 값 4개. **바뀐 것은 호스트·GPU 인덱스·경로뿐이다.**
>
> ⚠️ kanu의 옛 learner 프로세스는 **아직 살아 있고 사용자만 종료할 수 있다.** 우리는
> `ssh kanu 'ps -p <pid> -o pid,etime'` 같은 **읽기 전용** 조회만 한다 — 어떤 신호도 보내지 않는다.

**(2026-07-30 19:41 KST 읽기 전용 재확인):** `run_hil_server.sh --check`가 exact process
contract, gRPC health/ServerInfo/BufferStatus와 startup ready evidence를 모두 통과했다.
아래 counter는 스냅샷이므로 이후 transition에 따라 증가할 수 있다.

그날 Kanu에는 실제 `run_rlpd_learner_server.py`가 하나만 실행 중이었다.

- PID `1112465`, physical GPU `5`, port `127.0.0.1:50053`, health `ready`
- run root `~/hil-serl-data/runs/cube_in_cup_manual_schema3_thr05_20260730_1715`
- protocol `2`, transition schema `3`, production policy/reward model과 observation hash 일치
- replay `400`, intervention `225`, overwrite `0`; learner/gradient/policy `301/602/6`
- startup은 `restored_checkpoint=null`인 fresh lineage였고 offline demo **2,037개**를 로드함

이전 production/RAM 시험의 online replay는 checkpoint나 새 process로 옮기지 않았다.
replay/intervention은 RAM-only라 옛 process 종료와 함께 폐기되는 것이 계약이고, 영구
offline demo pickle 2,037개만 같은 SHA로 보존해 새 lineage에 다시 넣었다. 현재 400/225는
schema-3 process가 새로 받은 ingress다.

정상 3-CLI 운용의 server terminal은 다음 명령만 사용한다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh
```

이 wrapper는 healthy learner를 중단하거나 중복 기동하지 않고 **재사용**한 뒤
`127.0.0.1:50153 → junhyeong_ai:127.0.0.1:50053` tunnel을 연다
*(🗄️ 2026-07-30까지는 `→ kanu:127.0.0.1:50053`이었다. laptop 쪽 포트 50153은 그대로다)*. 읽기 전용 확인은
`./run_hil_server.sh --check`다. 새 lineage가 명시적으로 필요할 때만 기존 process를 운영자가
별도로 종료한 뒤 `--new-lineage --gpu N --run-id NAME`을 사용한다.

> **⛔ 역사 기록:** 2026-07-27에는 zero-action receive server PID 1096786만 있었고,
> 2026-07-28~29에는 HIL process가 하나도 없었다. 현재 상태 판단에 그 PID/"미배포" 문구를
> 재사용하지 않는다.

실제 서빙은 `run_rlpd_learner_server.py`가 한다. 이 스크립트는 receive server의 **엄격한 상위집합**이다 — 같은 gRPC ingress + 같은 classifier runtime에 더해 실제 `SACAgentHybridSingleArm` 추론(`ur_env/learner/composition.py`가 `VersionedPolicyRuntime`을 만들어 `build_actor_service(sample_action=runtime)`로 꽂는다), CTA 학습, feature replay, checkpoint를 한 프로세스에 조립한다. 현재 떠 있는 것이 바로 이 process다.

serving model ID (`ur_env/learner/config.py:14-17`):

- production: `hil-serl-hybrid-sac-resnet10-trunk-cache-v1`
- `--synthetic-e2e`: `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`

학습 시작 게이트는 `ur_env/learner/batches.py::RLPDBatchSampler.ready` — **online replay ≥ 100 AND offline demo ≥ 1**. 둘 다 필요하다. replay가 100 미만이면 서버는 정상적으로 뜬 채 policy version 0을 계속 서빙하고 워커는 대기한다.

**Kanu 환경 점검 결과 — 추가 설치 불필요, 단 베이스 env를 쓸 것.**

> **측정일에 주의.** 아래 표의 python/의존성/자산 행은 2026-07-27 점검값이고 GPU/RAM은 2026-07-30 현행 process와 함께 재확인했다. disk 행은 07-29 값이다. 공유 자원 수치는 언제든 바뀐다.
>
> 🗄️ **그리고 이 표는 kanu의 하드웨어다.** `junhyeong_ai`는 **RTX 5070 Ti 1장(sm_120, index 0)**,
> RAM 60 GB(available 약 37 GB), disk 여유 약 594 GB다 — GPU 수도, RAM 여유도, disk 여유도
> 전부 다르므로 **아래 숫자로 새 서버 용량을 계획하지 마라.** 반면 python 경로와
> jax/flax/distrax/tfp/wandb 행은 **그대로 재현됐다**(새 env를 kanu의 `pip freeze`로 만들었다).
> 🪤 **kanu의 learner도 A4000을 8장 쓴 적이 없다 — GPU 5 한 장이었다.** "8장에서 1장으로
> 줄었다"로 읽으면 warm-up 비교를 잘못하게 된다.

| 항목 | 결과 | 측정일 |
| --- | --- | --- |
| python | **`/home/junhyeong/miniconda3/envs/il/bin/python`** (베이스 `il`) | 07-27 |
| jax/jaxlib/flax/distrax/tfp/wandb | `0.5.3 / 0.5.3 / 0.10.5 / 0.1.5 / 0.25.0 / 0.26.0` — lock과 정확히 일치, `validate_learner_dependencies` 통과 | 07-27 |
| protobuf | `7.34.1`, implementation `python` | 07-27 |
| ResNet-10 자산 | source·`~/.serl` 캐시 모두 SHA `175745d4…07f57b` 일치 | 07-27 |
| GPU | 8× RTX A4000 (16 GiB). production learner는 **GPU 5**, 조회 시 4,335 MiB 사용 / 11,768 MiB free / util 0% | 07-30 19:42 |
| RAM | 251 GiB total, 조회 시 available 약 **118 GiB**. startup forecast gate는 available 138,997,297,152 B ≥ required 10,290,708,416 B로 accepted | 07-30 19:42 |
| disk | 단일 파일시스템 `/` 1.8 T, **95% 사용 / 여유 약 90 G**. checkpoint 305 MiB × N, **자동 pruning 없음** | 07-29 |
| 🔴 환경 드리프트 | numpy **2.2.5**(lock 1.26.4, 메이저 점프), orbax 0.11.12(0.11.5), grpcio 1.80.0(1.74.0). **런타임 fail-closed 대상이 jax/flax/distrax/tfp/wandb 뿐이라 이 3개는 자동으로 안 걸린다** (§12.6-5) | 07-28 |

> ⛔ **폐기된 07-27 행 2개** (기록용):
> - *"GPU 2/4/5/6 권장, GPU 7은 receive server가 12.3 GiB 점유 중"* → **더 이상 아무것도 점유하지 않는다.** 다만 그 12.3 GiB가 **왜** 생겼는지는 계속 유효한 교훈이다 — 아래 `XLA_PYTHON_CLIENT_PREALLOCATE` 항목.
> - *"RAM 251 G 중 available 73 G"*, *"disk 여유 127 G (93% 사용)"* → 07-29에 각각 재측정된 값이 위 표에 있다.
> - *"learner 파일 최신성: Kanu `5fb716b` 와 로컬 `ee3240e` 의 learner 파일 19개 전부 SHA256 동일"* → **`3f199d4` 머지로 무의미해졌다.** 이제 두 쪽이 비교할 대상 자체가 다르다. Kanu의 worktree `/tmp/gello-hil-rl-receive-server-v2` 는 여전히 `5fb716b` 이므로 **머지 이후의 actor/classifier 변경을 하나도 갖고 있지 않다.**

**overlay venv `/tmp/gello-hil-rl-receive-overlay-v2` 를 learner에 재사용하지 말 것.** protobuf를 `3.20.3` 으로 핀했는데 wandb 0.26.0은 `wandb/proto/` 에 v4~v7만 배포한다 → `ImportError: cannot import name 'Imports' from wandb.proto.wandb_telemetry_pb2`. `requirements-learner.lock` 도 protobuf 7.34.1을 요구한다.

기타 함정:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` **필수**. 07-27에 receive server가 GPU 7에서 12.3 GiB를 잡고 있던 것은 JAX 기본 75% preallocation(16376 × 0.75) 때문이었다. 16 GiB 카드에서 이걸 두면 learner + classifier가 한 GPU에 못 들어간다.
- `--resnet-cache` 는 run root 안의 새 경로로 지정할 것. `~/.serl/resnet10_params.pkl` 의 SHA가 다르면 `ResNetAssetError` 로 즉사하며 코드는 절대 덮어쓰지 않는다. (현재는 SHA가 일치하므로 기본값도 안전하다.)
- `preflight_checkpoint_run` (`ur_env/learner/composition.py::preflight_checkpoint_run`) 은 `--resume-*` 없이 시작할 때 checkpoint root에 기존 항목이 있으면 **거부**한다. 새 run은 반드시 빈 root.
- `--dry-run` 은 gRPC를 bind하지 않고 `rlpd_learner_dry_run_passed` 출력 후 즉시 exit한다. 구성 검증 전용이라 actor가 붙을 수 없다.
- `--target-learner-step` 을 생략하면 무한 continuous 학습이다. 첫 실전은 `5000` 을 권장한다.
- 학습 워커가 fault를 내도 서버는 last-known-good 정책을 계속 서빙하므로 gRPC health는 ready로 보인다. `rlpd_learner_worker_fault` 이벤트를 로그로 감시해야 한다(§7 P0-4).

actor가 pin해야 하는 값:

```text
--expected-model-id hil-serl-hybrid-sac-resnet10-trunk-cache-v1
--expected-reward-authority server_classifier
--expected-reward-model-id <서버 CLI 의 --reward-model-id 와 정확히 같은 문자열>
--observation-schema-hash 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

> 🪤 **`--expected-reward-model-id` 에 `cube-in-cup-checkpoint-150` 을 그대로 쓰지 마라.** 07-27 판본은 그 문자열을 예시로 박아 뒀는데, kanu에는 같은 12분 세션이 만든 **`checkpoint_150` 이 5개** 있다(§12.6-1). 이 값은 자유 문자열이고 서버 `--reward-model-id` 와의 **동일성만** 검사하므로, 어느 체크포인트인지 구별되는 이름(예: 날짜 + 데이터셋 이름)을 쓸 것.
>
> 📌 **현행 값은 `cube-in-cup-all3-ckpt150+sidecar-v1`** 이며 체크포인트 **와** 입력 계약(sidecar)을 같이 이름 짓는다(§12.7). `junhyeong_ai`에서는 운영 artifact가 `~/hil-serl-data/classifier_ckpt/checkpoint_150` **하나**로 정리돼 kanu의 5중 동명 함정 자체가 없다 — 다만 `~/hil-serl-data/datasets/cube_in_cup_all3/classifier_ckpt/` 에 원래 레이아웃 보존용 사본이 **의도적으로 중복**돼 있다. 서버가 로드하는 것은 앞의 경로다.

**미해결 위험**:

- **학습이 시작되기 전 초기화된 SAC 정책이 어떤 크기의 action을 내는지 아직 확인하지 않았다.** 첫 실물 구동에서는 `--mock-policy-noise` 또는 낮은 `--scale` 로 이를 먼저 관측할 것.
- 🔴 **`ACTION_SCALE` 은 learner fingerprint에 들어 있지 않다.** 따라서 `0.01` 로 녹화한 데이터를 `0.0125` 환경에서 재개해도 **경고 없이** 같은 액션이 25% 더 멀리 간다. resume 전에 손으로 확인할 것.

### 11.6 카메라 — 2026-07-27 기록 (⚠️ 대체됨 → §11.10)

> **이 절은 07-27 시점의 기록이다. 2026-07-28에 뒤집혔다 — 허브는 고장이 아니었고 재연결로 둘 다 정상 동작했다.** 현재 상태는 **§11.10**을 볼 것. 아래는 "왜 고장이라고 판단했는가"를 남기려고 보존한다.

*(2026-07-27)* cam1/cam2가 공유 USB 허브 `4-4` 에서 동시에 죽었다: `uvcvideo Non-zero status (-71)` (EPROTO). 운영자가 케이블을 분리한 상태였다. 그래서 당시에는 actor 전 경로를 돌릴 수 없었고, 팔 단독 검증만 `run_real_hil.py` 경로로 가능했다.

**cam2는 손목 카메라다.** 이전 문서와 한 에이전트가 고정 데스크 카메라로 오판한 적이 있다. 이 판단은 **07-28에 녹화 데이터로 독립 확인됐다**(§11.10).

### 11.7 남은 actor 쪽 결함 — 2026-07-29 갱신

**해결된 것 (2026-07-28):**

1. ~~frame-map 측정 무효~~ → **해결.** §11.9 참조.
2. ~~deadman GUI 무효~~ → **해결** (`607e541`). `--deadman {topic,spacebar}`, 기본 `topic`. 문자열 spec으로 넘기는 이유는 `RosTopicDeadman` 이 backend의 rclpy 노드에 구독하는데 그 노드가 `UR7eEnv` 생성 이후에야 존재하기 때문이다. `strict xfail` 은 제거하고 회귀 가드로 전환했다.
3. ~~`_await_robot_state` 에 카메라 첫 프레임 대기 없음~~ → **해결** (`607e541`). `_await_first_frames()`. `hasattr(backend, "get_image")` 가드는 필수다 — 테스트의 stub backend가 `get_image` 를 구현하지 않는다.
4. ~~`DRY_RUN` 을 CLI로 뒤집을 방법이 없음~~ → **해결** (`607e541`). `--arm`. `_build_actor_environment` **이전에** 설정해야 한다 — `DRY_RUN` 은 `URRosBackend` 생성 시 한 번만 읽힌다.

**남은 것 (2026-07-29 `43ba314` 에서 전부 미수정 확인):**

1. **전역 ESC 리스너** (`ur_env/envs/ur7e_env.py::UR7eEnv.__init__`의 pynput 블록, `grep -n "keyboard.Listener"`) — deadman 배선과 **별개**다. 아무 창에서 ESC를 누르면 `self.terminate` 가 서고 에피소드가 끝난다. 미수정.
2. **`run_real_hil.py` 의 frame-map 판정이 포화 표본을 걸러내지 않는다.** 포화 구간에서는 명령 델타가 리더 변위와 무관하게 `ACTION_SCALE` 노름에 고정되므로 `robot_dp ≈ M @ leader_dp` 모델 자체가 성립하지 않는데, 판정은 그 표본을 넣고 최소자승을 돌린 뒤 FAIL을 띄우며 "좌표계를 의심하라"고 안내한다 — **오진 유도다.** `held`(거버너 거부)만 세고 ACTION_SCALE 노름 클립은 안 세는 것이 사각지대. 미수정. *(07-28 측정은 포화 표본을 손으로 걸러내서 통과시켰다 — §11.9. 즉 결함은 판정 코드에 남아 있고, 측정 자체는 유효하다.)*
3. 🔴 **workspace box가 `run_real_hil.py` 에서 비활성이다.** `DefaultUR7eEnvConfig.ABS_POSE_LIMIT_*` 가 zeros(`config.py::ABS_POSE_LIMIT_LOW/HIGH`, 현재 :84-85)이고 측정된 박스는 `ur_experiments/cube_in_cup.py:162-167` 에만 있다. 코드가 0-부피 박스를 감지해 끄는 것은 올바른 처리지만, 실측 결과 **DRY RUN 300 스텝 중 241 스텝(80%)이 측정 박스 밖**이었고 시작점에서 **최대 73.9 cm** 이탈했다. `--arm` 이었다면 팔이 실제로 거기까지 갔다. 속도 제한은 *얼마나 빨리* 가는지만 막지 *어디로* 가는지는 안 막는다. **이것이 §1 판정표에서 "팔은 움직였는데 workspace box는 여전히 실기 미검증"인 이유다.**
4. **`buffer_period = 0`** (`ur_experiments/cube_in_cup.py` 의 `buffer_period`, 현재 :271) 이라 `--checkpoint-path` 를 줘도 actor-format demo pickle이 하나도 안 쓰인다. CLI 플래그도 없다. 기존 recorder take는 별도 converter로 살릴 수 있으므로, 이것은 **새 actor 직접 녹화 경로의 선결 조건**이다(§7 P0-2).

### 11.8 환경 함정 (반복 발생)

- **시스템 grpcio 1.30.2가 고장나 있다.** gRPC를 쓰면 오류 없이 100% CPU로 무한 정지한다(rc=124, 단일 스레드 `R`, `wchan` 비어 있음). 기계어 원인은 미확인이나 **행동은 100% 재현**된다. 반드시 `/home/laptop3/venvs/gello-hil-actor` (grpcio 1.74.0, rclpy용 `--system-site-packages`)를 쓸 것. 시스템 `python3` 로 gRPC 코드를 절대 실행하지 말 것.
- **PYTHONPATH는 덮어쓰지 말고 이어붙일 것** — `PYTHONPATH=…` 는 ROS overlay를 날려 `ModuleNotFoundError: ur_gello_bringup` 을 만든다. `:$PYTHONPATH` 를 붙인다. **반대로 pytest는 ROS PYTHONPATH가 있으면 깨진다.** 원인은 ROS의 pytest 플러그인이므로 `-p no:launch_testing` 으로 해결한다(`--ignore=` 로는 막히지 않는다 — 다음 파일이 같은 자리를 물려받는다).
- ROS setup.bash를 `set -u` 아래에서 source하면 `AMENT_TRACE_SETUP_FILES: unbound variable` 이 난다. `set +u` / `set -u` 로 감쌀 것.
- ⛔ **폐기:** 07-27 판본의 *"모든 절차는 worktree에서만 돈다. `/home/laptop3/gello_software` 에는 `serl_ur_infra/ur_experiments/` 가 아예 없다(다른 branch)"* 는 **`3f199d4` 머지 이후 거짓이다.** `serl_ur_infra/ur_experiments/{__init__,cube_in_cup,mappings}.py` 는 이제 canonical checkout에 있다(2026-07-29 확인). **모든 절차는 이제 `/home/laptop3/gello_software` 에서 돈다.**

### 11.9 실기 측정 결과 (2026-07-28, 실제 UR7e @ 192.168.10.11)

**frame-map — 매핑 = 단위행렬.** `--scale 1.0`, move-then-hold 프로토콜, 개입 267 스텝.

포화(스텝 델타가 `ACTION_SCALE` 노름 클립에 붙음) 51.3%를 **제외하면**:

| | 전체 267 | **포화 제외 130** |
| --- | --- | --- |
| 단위행렬 잔차 | 0.203 | **0.093** (기준 0.15) |
| alpha (추종이득) | 0.914 | **0.984** |

자유 9-파라미터 행렬이 전체 표본에서 0.165밖에 못 내려간다 — 단위행렬(파라미터 0개)의 0.203과 거의 차이가 없다. 진짜 회전이 있었다면 자유 행렬이 압도적으로 잘 맞았어야 한다. M의 대각 평균 0.871, 비대각 최대 0.084.

**결론: 부호 뒤집힘 없음, 축 교환 없음, 매핑 = I.** `R_align = I` 라는 기존 판단이 실측으로 재확인됐다. **RViz에서 뒤집혀 보인다는 이유로 X/Y를 뒤집지 말 것.**

포화 표본을 제외하는 것이 통계적으로 정당한 이유: 비포화 스텝 = 명령이 리더를 따라잡은 스텝 = move-then-hold의 "hold" 구간이다. 애초에 이 프로토콜이 만들어내려던 측정 조건 그 자체라 편향이 아니다.

**첫 실물 구동.** `run_real_hil.py --arm --scale 0.25`, 100 스텝 중 개입 64 스텝, `held=0`. 운영자 육안 확인 통과. 개입 불변식 4종(anchor-latch 0, gain-latch 0, action-exec 1.000, held-rate 0%) 전부 통과했고, 그중 **action-exec(저장 액션 == 실행 액션, 비율 1.000)** 이 특히 중요하다 — 이게 깨지면 학습 서버가 거짓 데이터로 학습한다.

> ⚠️ **이 세션은 `run_real_hil.py` 경로다. actor entrypoint(`run_remote_rlpd_actor.py`)가 아니다.** 두 경로는 wrapper chain도 config도 다르다. 따라서 이 결과는 "팔·개입·저장 정합"의 증거이지 **actor 경로가 실기에서 돈다는 증거가 아니다.** 그리고 이 config에서는 workspace box가 꺼져 있었다(§11.7-3).

**속도 정렬** (`a5f9890`). 3층을 **균일하게 1.25배**:

| | 텔레옵 (검증됨) | 이전 | 현재 |
| --- | --- | --- | --- |
| `max_step_rad` | 0.0025 | 0.0020 | **0.0025** ← 동일 |
| 유효 병진 | 0.16 m/s | 0.10 | **0.125** |
| 유효 회전 | 1.0 rad/s | 0.50 | **0.625** |

현재 3층 값 (2026-07-29 `43ba314` 에서 코드 확인):

- `ACTION_SCALE = [0.0125, 0.0625, 1.0]` — `config.py::DefaultUR7eEnvConfig.ACTION_SCALE` (:73)
- `GOVERNOR = {v_max 0.15, w_max 0.75, dq_step_max 0.0625}` — `config.py::GOVERNOR` (:94-98)
- 업샘플러 250 Hz, `max_step_rad 0.0025` — `ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml:64`
- 검증된 텔레옵 상한 `v_max 0.16` / `w_max 1.0` — `ur7e_gello_eef.yaml:238,241`

균일한 것이 핵심이다 — 한 층만 올리면 다음 층이 조용히 잘라먹어 버퍼 정합성 불변식이 깨진다. 헤드룸 1.200x 유지(거버너·업샘플러 조인트 레이트 둘 다 0.625 rad/s). 1.25배를 고른 이유는 `max_step_rad` 가 정확히 텔레옵 값에 떨어지기 때문이고, **더 올리면 안 된다** — 텔레옵의 0.16 m/s 를 내려면 `max_step_rad` 가 0.0032 여야 하는데 `ur7e_gello.yaml:51-63` 의 주석이 250 Hz 업샘플 + 500 Hz 드라이버 사이클의 coalescing-safe 천장을 **~0.00314** 로 못박아 뒀다.

### 11.10 카메라 — 복구됨. 시리얼은 이제 하드코딩하지 않는다 (2026-07-29 갱신)

허브 `4-4` 는 고장이 아니었다(§11.6은 그 오판의 기록). 재연결로 둘 다 정상 동작했다(`4-4.1`, `4-4.3`).

**카메라는 한 쌍뿐이다. "쌍이 두 벌"은 시리얼 *필드* 두 개를 개체 두 벌로 오독한 것이다.**
(2026-07-29 직접 측정으로 확정. 아래 ⛔ 블록이 그 오독의 기록이다.)

| 포트 | `camera_info.serial_number` | `camera_info.asic_serial_number` | 장치 | 역할 |
| --- | --- | --- | --- | --- |
| `4-4.1` | **`147122072740`** | `151623020789` | plain D435 | cam1 = SCENE |
| `4-4.3` | **`243222072700`** | `322743060038` | D435IF | cam2 = WRIST |

**같은 카메라 두 대가 필드 두 개로 나타난 것뿐이다.** `serial_no:=` 가 매칭하는 것은
**`serial_number`**(모듈 시리얼)이고, 커널 USB 디스크립터(`journalctl -k`,
`/sys/bus/usb/devices/*/serial`, `lsusb -v`)가 노출하는 것은 **ASIC 시리얼**이다.
그래서 저널을 grep하면 `151623020789`/`322743060038` 만 나오고
`147122072740`/`243222072700` 은 0회로 보인다 — **다른 카메라가 아니라 다른 필드다.**
정본은 [`../docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](../docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md)
`:69-70`, `:82-83` 이고, 코드 기본값도 이미 모듈 시리얼로 일치해 있다 —
`ros2_ur_ws/launch_cameras.sh:69-70`, `ros2_ur_ws/_resolve_camera_serials.sh`,
`ros2_ur_ws/src/gello_recorder/gello_recorder/gello_recorder_gui.py:55-56`.
즉 `43ba314` 가 기본값으로 되돌린 쌍이 **옳았다.**

> 🪤 **저널 grep으로 이걸 다시 유도하지 마라.** 서로 다른 두 세션이 각각 저널 grep만
> 보고 **정반대의 틀린 결론**에 도달했다("쌍 A는 존재하지 않는 하드웨어" / "쌍이 두 번
> 뒤집혔다"). 저널은 ASIC 필드만 보여주므로 이 질문에 답할 수 없는 증거다. 확인은
> `rs-enumerate-devices -s` 또는 `pyrealsense2` 로 **두 필드를 나란히** 찍어서 한다.

> **⛔ 폐기 (2026-07-29 판본, 반증됨) — 왜 틀렸는지를 남기려고 보존한다. 인용하지 마라.**
>
> *07-29 오전 판본은 이 자리에 아래 표와 결론을 실었다:*
>
> | 쌍 | cam1 (plain D435) | cam2 (D435if) | 이 PC의 커널 저널 기록 |
> | --- | --- | --- | --- |
> | A (리포에 오래 박혀 있던 값) | `147122072740` | `243222072700` | **0회.** 한 번도 열거된 적 없음 |
> | B | `151623020789` | `322743060038` | 3회 / 2회 (전부 2026-07-28) |
>
> *…그리고 "쌍 A는 이 PC가 열거한 적 없는 하드웨어이므로 `43ba314` 의 기본값은 존재하지
> 않는 쌍이고, 매번 fallback + WARN 경로를 탄다"고 결론지었다.*
>
> **틀렸다.** 관측(`journalctl -k | grep -cE "147122072740|243222072700"` → 0)은 사실이지만
> 해석이 틀렸다. 저널은 ASIC 시리얼만 싣는다. 0회는 "그 카메라가 없다"가 아니라
> **"저널에는 그 필드가 애초에 안 실린다"** 는 뜻이다. 같은 이유로 `43ba314` 커밋
> 메시지의 "쌍이 두 번 뒤집혔다"도 실재하는 개체 교체를 말한 것이 아니다.
> 따라서 fallback+WARN 상시 동작이라는 우려도 성립하지 않는다 — 선호 시리얼은 버스에
> 그대로 있다.

**시리얼 상수를 고치는 대신 해석 로직을 넣은 것 자체는 옳다** (`43ba314`, `ros2_ur_ws/launch_cameras.sh`).

- 선호 시리얼(`CAM1_SERIAL`/`CAM2_SERIAL`, 현재 기본값은 쌍 A)이 **둘 다 버스에 있으면 그대로** 쓴다.
- 아니면 **모델 클래스로 배정**한다: plain `D435` → cam1(SCENE), `D435I`/`D435if` → cam2(CLOSE-UP/wrist). 이때 크게 경고한다.
- 두 대가 같은 클래스면 시리얼 정렬 순서로 떨어뜨리고 **"녹화 전에 창을 확인하라"고 경고**한다.
- 2대 미만이면 하드 에러 + override 문법 출력. `pyrealsense2` 가 없으면 경고로 강등.
- 열거는 스트리밍 락을 잡지 않으므로 이미 카메라를 쥔 프로세스와 함께 돌려도 안전하다.

🪤 **이 수정이 왜 상수 변경보다 중요한가**: 없는 시리얼로 `realsense2_camera` 를 바인딩하면 **에러가 나지 않는다.** 노드는 정상적으로 뜨고, 아무것도 publish하지 않고, 모든 소비자에게는 "프레임 없음"으로 보인다. 스택 어디에도 *뽑힌 카메라*와 *잘못 설정된 카메라*를 구별하는 곳이 없었다.

**아직 쌍 B 시리얼이 상수로 남아 있는 곳** (2026-07-29 확인 — `launch_cameras.sh` 처럼 버스 해석을 하지 않는다):
`ros2_ur_ws/run_recorder.sh:64-65`, `ros2_ur_ws/camera_viewer.py:10-11`, `ros2_ur_ws/src/gello_recorder/gello_recorder/gello_recorder_gui.py:55-56`, `docs/testing/06_SENSORS.md`, `docs/ros2/GELLO_UR7E_{ACT,DIFFUSION,FM}_DEPLOY.md`, 그리고 별도 저장소 `gello_software_remote_classifier`.
**이 문서는 그 파일들을 소유하지 않는다. 고칠 때는 `43ba314` 의 해석 방식을 따를 것.**

**cam2 = 손목 카메라 확정 (역할 확정이지 개체 확정이 아니다).** 녹화 데이터셋 영상에서 cam2는 그리퍼 손가락이 같은 픽셀에 고정된 채 배경만 흐른다(정지 픽셀 비율 0.4–1.0%, cam1은 32–43%). 저분산 컬럼이 `x[476,536]`·`x[1008,1149]` 에 몰려 `ur_experiments/cube_in_cup.py:175-178` 주석의 손가락 측정치(fingertip `x 512-563` / `x 1000-1117`, 파지축 `x=781`)를 독립 재현했다. **따라서 `IMAGE_CROP` 값 자체는 정책 관점에서 올바르다** — §12.1의 결함은 크롭이 틀렸다는 뜻이 **아니다.**

**미확정 (07-29에도 그대로)**: 지금 연결된 두 대 중 **어느 물리 개체가 손목에 달렸는지.** `43ba314` 의 배정도 모델-클래스 추론이다. 팔을 한 번 흔들어 cam2 창을 보면 끝난다. 이것이 §12.8-2(실기 분포)의 선결 확인이기도 하다.

**⚠️ 2026-07-29 11:14에 EPROTO가 다시 났다.** `uvcvideo 4-4.1:1.4` / `4-4.3:1.1` / `4-4.3:1.4` 전부 `Non-zero status (-71)` — §11.6에서 "허브 고장"으로 오판했던 것과 **같은 증상**이다. 그때 결론은 "고장이 아니라 재연결로 해결"이었으므로 이번에도 하드웨어 고장으로 단정하지 말 것. 다만 **카메라 경로가 안정적이지 않다는 신호**이고, actor 전 경로 실기 검증(§7 P0-3) 전에 확인이 필요하다.

---

## 12. reward classifier 조사 (2026-07-28 측정 · 2026-07-29 머지 반영 · 2026-07-29 sidecar로 해소)

이 절은 reward classifier의 **출처·전처리·실측 성능**을 기록한다. 결론부터: **classifier 자체는 좋았고, 우리가 잘못된 이미지를 먹이고 있었다. 2026-07-29에 이미지를 분리해서 고쳤다.**

> threshold 측정·결정의 역사(0.85 → 0.5 → 0.2)는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)가 소유한다. **현행 실행값의 source of truth는 코드 상수이며 2026-07-30에는 0.5다.** 여기서는 결정에 쓰인 측정을 기록한다. sidecar 도입 후에도 그 측정 수치는 유효하다.

> 🚩 **읽는 순서 주의.** §12.1~§12.6 은 **결함이 살아 있던 시점의 조사 기록**이고, 지금은 §12.1 머리의 「해소」 블록과 §12.7 이 현재 상태다. 아래 본문의 "따라서 원인은 actor 쪽 `IMAGE_CROP` 이다" 같은 서술은 **당시 진단**이며 그 자체로는 여전히 맞다. 다만 **거기서 도출됐던 처방(크롭에 맞춘 재학습)은 채택되지 않았다.**

### 12.1 크롭 불일치 — 증명됨, 그리고 2026-07-29에 **분리로** 해소됨

> ### ✅ 해소 (2026-07-29) — 재학습이 아니라 **분리**다
>
> **한 이미지가 두 소비자를 섬기던 것을 그만뒀다.** actor가 관측에 **무크롭 128×128
> JPEG sidecar** 를 덧붙여 보내고, 서버는 **그것만** 분류한다. 정책은 측정된
> `IMAGE_CROP` 을 **그대로** 쓴다. 아래는 전부 이번 세션에 코드에서 확인한 것이다.
>
> | 항목 | 확인 결과 | 근거 |
> | --- | --- | --- |
> | proto 변경 | **없음** — 전송이 이미 generic named-tensor map이다 | `serl_ur_infra/proto/actor_transport.proto` |
> | **관측 schema hash** | **불변** `3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903` — hash는 `CANONICAL_OBSERVATION_SPEC` **문서**에서 나오고 wire payload에서 나오지 않는다. sidecar는 canonical 검증 **이전에** 벗겨진다 | 이 세션에서 `CANONICAL_OBSERVATION_SCHEMA_HASH` 직접 출력해 대조 · `ur_env/actor_network.py::_split_classifier_sidecar` |
> | 프로덕션 이미지 출처 | `get_im()` 이 **어차피 디코드하는** full-res BGR을 크롭 **직전**에 참조로 보관 | `ur_env/envs/ur7e_env.py` 의 `self._last_camera_frames[key] = bgr` (≈`:1064`, 크롭은 **바로 다음 줄**) |
> | 리사이즈 | 서버가 했을 것과 **같은** 결정적 `cv2.resize(bgr,(128,128))` 를 랩톱이 수행 후 JPEG 인코드 | `ur_env/classifier_sidecar.py::_resize_for_classifier` (producer/consumer 공용 단일 정의) |
> | threshold | **현행 0.5** | `ur_env/rlpd_receive_server.py::DEFAULT_REWARD_THRESHOLD`; 0.2는 07-29 역사값 |
>
> **왜 원본 JPEG을 그대로 넘기지 않았나** — 그게 최초 설계였고 **측정으로 기각됐다.**
> 카메라 노드가 `jpeg_quality=95` 라 720p 한 장이 206 KiB(cam1)/194 KiB(cam2), 쌍으로
> 400 KiB다. 2 Hz만 붙여도 6.55 Mbit/s 추가라 **13 Mbit/s WiFi 링크를 그대로 넘고**,
> 한 번의 첨부가 100 ms 예산에 **+252 ms** 를 얹는다. 랩톱에서 128×128로 줄여 다시
> 인코드하면 쌍당 **~10–15 KiB**(27배 감소)다. *(이 수치들은 `classifier_sidecar.py`
> 모듈 docstring이 근거로 제시한 값이다 — 이 세션에서 재측정하지는 않았다.)*
>
> **모든 transition이 분류되지 않는다. 그게 설계다.** sidecar는 **~2 Hz**, 그리고
> **팔이 정지해 있을 때만** 붙는다. 10 Hz 루프이므로 대다수 transition은 미분류로
> 도착한다. 미분류 transition의 필드는 이렇게 확정된다(`RewardTransitionFinalizer`):
>
> | 필드 | 미분류 시 |
> | --- | --- |
> | `rewards` | **0.0 강제** — 분류가 없으면 성공의 증거가 없다 |
> | `masks` / `dones` / `truncated` | 로컬 제안을 **그대로 통과**(mask/done 일관성 검사를 만족시키는 유일한 선택) |
> | `classifier_evaluated` | `0` |
> | `classifier_probability` / `classifier_threshold` / `classifier_success` | `0` (미평가 시 0이 아니면 wire 계약이 거부한다) |
> | `reward_model_id` | `""` (같은 이유) |
>
> 순효과: 미분류 transition은 **평범한 zero-reward 비종료 샘플**이다. 못 하는 일은
> 딱 하나 — **양의 reward로 episode를 끝내는 것**이고, 그건 아무도 분류하지 않은
> 스텝에서 빼앗아야 할 바로 그 권한이다.
>
> **희소 분류는 대역폭 트릭이 아니라 정확성 조치다.** 성공 판정의 flicker를 없애고
> (몇 Hz로만 갱신되는 판정은 10 Hz로 진동할 수 없다), 큐브를 놓은 뒤 장면이
> **가라앉을 시간**을 준다. 대역폭 절감은 부수 효과다.
>
> **N-of-M 평활은 있지만 기본 OFF다** (`--success-confirmations 1`,
> `run_rlpd_learner_server.py` 의 `--success-confirmations`). 따라서 보고되는 확률은 **순간 sigmoid** 이고
> **라이브 뷰어와 프레임 단위로 일치한다.** 일부러 그렇게 뒀다 — 서버가 몰래 평활하면
> 둘을 비교하는 사람이 로봇 대신 필터를 디버깅하게 된다.
>
> #### 왜 "재학습"이 아니라 "분리"인가 — 측정값 보존
>
> 🔴 **이것이 이번 결정의 가장 큰 실질 이득이다.** 분류기가 **여전히 무크롭 프레임을
> 먹기 때문에, threshold 작업의 측정 세트 전체가 이 변경을 그대로 통과해 살아남는다.**
> §12.3의 held-out 수치도, [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)의
> sweep도 조건이 바뀌지 않았다. **크롭에 맞춰 재학습했다면 그 숫자를 전부 무효화하고
> 전면 재측정을 강제했을 것이다**(§12.8-4가 정확히 그걸 예고하고 있었다).
>
> #### 🔴 이것이 고치지 **못하는** 것 — 팔 가림(occlusion) 병리
>
> **정직하게 적는다: sidecar는 `take_21` 을 구제하지 못한다.** `take_21_20260720_210234`
> 은 @0.85 recall `0.0%`, @0.05 까지 내려도 `57.9%` 다(출처:
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md) `:369` —
> 이 세션에서 재측정하지 않고 인용했다). 팔이 cam1 시야를 쓸고 지나가는 동안 확률이
> **`0.005 ↔ 1.0`** 으로 진동한다.
>
> **근본 원인은 시야/가림이지 전처리도 라벨도 아니다.** 따라서 크롭을 고쳐도, 재학습을
> 해도, threshold를 내려도 해결되지 않는다. sidecar의 **정지 게이트가 완화**할 뿐이다
> (움직이는 동안은 아예 분류하지 않으므로). **진짜 해법은 팔이 가로지르지 않는 카메라
> 배치**다. §12.8-3.
>
> #### 🪤 G19와 함께 나갔어야 하는 이유
>
> checkpoint 디렉터리 해싱만 고치고 상수를 안 바꿨다면 **정확히 최악의 실패**가 난다 —
> 서버가 은퇴한 recall-0% 체크포인트로 **깨끗하게 기동해서** 모든 transition에 reward 0을
> 영원히 내보내고, 어디에도 에러가 없다. 그래서 둘은 같이 나갔다. §12.6-3

**학습 전처리** (kanu `~/workspace/youngwoong/hil-serl/examples/cube_classifier_pipeline.py:290-297`):

```python
def preprocess_frame(frame_bgr, crop):
    if crop:                       # <- export_0724.py:32 는 crop=None 을 넘긴다
        x0, y0, x1, y1 = crop
        frame_bgr = frame_bgr[y0:y1, x0:x1]
    resized_bgr = cv2.resize(frame_bgr, (128, 128), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized_bgr[..., ::-1], dtype=np.uint8)
```

→ **크롭 없이** 1280×720 전체를 128×128로 찌그러뜨린다(가로가 0.5625배로 압축되는 종횡비 왜곡).

**추론 전처리** (우리 `ur_env/envs/ur7e_env.py::get_im()`): JPEG decode(BGR) → **`IMAGE_CROP` 적용**(cam1 `img[20:670,340:990]` 650×650, cam2 `img[0:720,420:1140]` 720×720) → 128×128 리사이즈 → BGR→RGB.

**서버는 이미지 변환을 하나도 하지 않는다.** `ur_env/rlpd_receive_server.py::validate_classifier_frames` / `::_classifier_model_input` 은 canonical `(1,128,128,3)` uint8 을 그대로 통과시키는 strict-validating passthrough다. `grep -n "1280\|720\|resize\|cv2\." rlpd_receive_server.py` 는 **0 hit**. 따라서 원인은 전적으로 **actor 쪽 `IMAGE_CROP`** 이다.

> **⚠️ 위 문단은 2026-07-28 시점의 코드 서술이다. 함수 이름과 입력이 07-29에 바뀌었다.**
> `_classifier_observation()` 은 **없어졌다.** 지금은 `_classifier_model_input()` 이고,
> 받는 것은 **canonical 관측이 아니라 디코드된 sidecar 프레임**이다
> (`validate_classifier_frames()` 가 계약을 소유한다). **"서버는 이미지 변환을 하지
> 않는다"도 더 이상 사실이 아니다** — 서버는 sidecar JPEG을 `decode_classifier_frames()`
> 로 디코드한다(라이브 뷰어와 같은 레시피). *진단 자체*("당시 원인은 actor 쪽 크롭")는
> 여전히 맞지만, **이 문단의 코드 좌표를 지금 코드에서 찾지 마라.**

> **🪤 원래 G15/B8 문구는 틀렸다.** `reward_classifier_runtime.decode_classifier_image()` 를 지목했는데, 그 함수(`ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_runtime.py:81-88`)는 **ZMQ GUI 뷰어(`tcp://127.0.0.1:5594`) 전용**이고 gRPC 경로(port 50053)와 **호출 관계가 전혀 없다.** 두 모듈은 서로 다른 파일·프로세스·전송이다.
>
> 그리고 그 함수가 **크롭 없이 리사이즈하는 것은 그쪽 용도에서 올바르다** — 뷰어는 raw 카메라 토픽을 직접 구독하고 canonical observation을 아예 보지 않으므로, 무크롭 리사이즈가 정확히 학습 전처리와 일치한다. **고칠 위치가 다르므로 이 문구를 근거로 서버나 뷰어 코드를 건드리면 안 된다.**
>
> 두 경로를 절대 섞지 말 것:
>
> | | gRPC 경로 (실제 RL) | ZMQ 뷰어 (사람 확인용) |
> | --- | --- | --- |
> | 엔드포인트 | port 50053 | `tcp://127.0.0.1:5594` |
> | 입력 (~~07-28~~ → **07-29 현재**) | ~~canonical observation (**크롭됨**)~~ → **sidecar 프레임 (무크롭 128×128 JPEG)** | raw 카메라 토픽 (**무크롭**) |
> | 학습 전처리와 | ~~🔴 불일치~~ → ✅ **일치** | ✅ 일치 |
> | 실행 스크립트 | §11.5 | `serl_ur_infra/run_remote_reward_classifier_server.sh` (GPU 서버 — kanu에서 검증, 현행 `junhyeong_ai`에서는 미검증) + `ros2_ur_ws/run_remote_classifier_viewer.sh` (laptop) |
>
> **(2026-07-29)** 두 경로는 **여전히 다른 프로세스·다른 전송**이라 섞으면 안 된다는 원칙은
> 그대로다. 바뀐 것은 **분류 입력이 이제 양쪽 다 무크롭이라는 점**이다. 그래서
> `--success-confirmations 1`(기본)에서 **서버 확률과 뷰어 확률이 프레임 단위로 일치한다** —
> 실기에서 둘을 대조할 수 있게 된 것도 이번 변경의 부수 효과다. 다만 gRPC 쪽은 ~2 Hz로만
> 분류하므로 **모든 프레임에 대응하는 서버 확률이 있는 것은 아니다.**

### 12.2 3중 독립 검증

*(전부 2026-07-28, Kanu GPU, Jul-27 `cube_in_cup_all3` 체크포인트 기준)*

| 방법 | 결과 |
| --- | --- |
| 코드 판독 | `export_0724.py:32` → `preprocess_frame(frame, None)` |
| **픽셀 대조** | full-frame 리사이즈 가설 **MAE 0.00 / 100% 비트 일치**, 우리 크롭 **MAE 21–35 / 일치 픽셀 ~4%** |
| **실제 체크포인트 실행** | **성공 프레임 36장**(무크롭 입력 vs 같은 프레임의 크롭 입력), **threshold 0.85**: recall **100.0% → 33.3%**, 평균 P 0.9990 → 0.6027, 개별 최저 P **0.0213** |

> ⚠️ 위 `100.0% → 33.3%` 은 **n=36 짜리 대조 실험**이다. §12.3의 held-out 성능 수치(n=166 / n=266)와는 표본도 목적도 다르다. **"recall이 33%"라고 단독 인용하지 마라** — 이 숫자가 말하는 것은 *"같은 프레임을 크롭해서 넣으면 절반 넘게 놓친다"* 라는 **상대적 손실**이다.

**반증 시도 6종이 전부 실패했다:**

1. "all3의 3개 소스 중 하나는 크롭됐을 것" → pkl 6개 전부 무크롭 확인(0720분도 비트 단위 검증)
2. "프레임 정렬이 틀렸을 것" → 100% 완전 일치는 정렬 오류 시 발생 불가
3. "체크포인트 출처가 다를 것" → wandb argv + 샘플 수 정확히 일치(§12.4)
4. "종횡비 왜곡이 크롭과 비슷할 것" → 두 가설의 상관 0.43–0.48. cam1은 1.97배 확대이고 프레임 면적의 45.8%만 남긴다
5. "학습 증강이 흡수할 것" → `batched_random_crop(padding=4)` 는 ±4 px 평행이동이고 프레임의 3.1%. 크롭은 44–54%를 버린다 — **증강 범위의 1.6–2.4배 초과**
6. "다른 전처리 경로가 있을 것" → 하나 찾았으나(`decode_classifier_image`) 그건 **올바른** 쪽이고 gRPC 경로와 무관

#### 머지 전후 — 이건 이제 예측이 아니다

> **⛔ 2026-07-28 판본 (예측형, 이제 무효):** *"현재 프로덕션은 안 망가져 있다. `ur_experiments/cube_in_cup.py` 가 kanu 체크아웃 브랜치(`5fb716b`)에 존재하지 않아 거기서는 `IMAGE_CROP = {}` 기본값이 적용된다 — 즉 오늘 기준으로는 우연히 학습과 일치한다. **actor 브랜치를 머지하는 순간 결함이 유입된다.**"*

**그 머지는 2026-07-29 `3f199d4` 로 일어났다.** 현재 상태:

- `serl_ur_infra/ur_experiments/cube_in_cup.py` 는 **canonical branch에 존재하고** `IMAGE_CROP` 이 채워져 있다(현재 `:217-221`).
- 따라서 **crop된 canonical observation이 classifier로 가는 것이 이제 기본 경로다.** "우연히 일치하던" 보호막은 사라졌다.
- 🪤 **단, 이것이 "지금 프로덕션이 깨지고 있다"는 뜻은 아니다** — 2026-07-29 기준 **kanu에는 classifier를 물고 있는 gRPC 서버가 아예 떠 있지 않고**(§11.5, §12.5), kanu의 worktree `/tmp/gello-hil-rl-receive-server-v2` 는 여전히 `5fb716b` 다. **즉 결함은 코드에 실재하고, 다음에 실기 run을 띄우는 순간 물린다.**

`ur_experiments/cube_in_cup.py:181-210` 에 `KNOWN CONFLICT` 주석이 있다: *"Harmless while reward is stubbed (our loop ignores it); must be resolved ... before any run that trusts the reward signal."* 작성자가 알고 무장해제 상태로 출하한 것이고, **그 "resolved before any run"의 시점이 바로 지금이다.**

### 12.3 classifier 성능 — held-out 실측

> **측정 조건 (모든 수치에 공통으로 붙는다):**
> **2026-07-28** · **Kanu GPU** (`CUDA_VISIBLE_DEVICES=6`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`) · **크롭 없는 입력**(학습 전처리와 동일) · 라벨은 사람이 프레임 단위로 리뷰한 정답 · split은 `dataset_manifest.json` 의 val/test(학습에 **한 번도 안 들어간** take) · 스크립트는 ssh stdin으로 파이프해 실행했고 **kanu에는 아무것도 쓰지 않았다**.
>
> 🔴 **위 수치는 크롭이 활성인 현재 actor 경로에는 적용되지 않는다**(§12.1). 크롭을 넣으면 같은 프레임에서 recall이 무너진다(§12.2, n=36).

학습 로그의 `Train Accuracy: 1.0000` 이 take 암기를 시사해 직접 재측정했다.

| 체크포인트 | split (success n) | thr | recall | **FPR** | acc |
| --- | --- | --- | --- | --- | --- |
| **Jul-27 `cube_in_cup_all3`** | 0720 **test**만 (n=166) | 0.5 | **100.0%** | **0.0%** | **100.0%** |
| Jul-27 `cube_in_cup_all3` | 0720 **test**만 (n=166) | 0.85 | 95.8% | 0.0% | 98.4% |
| Jul-24 (legacy msgpack) | 0720 **test**만 (n=166) | 0.85 | 93.4% | 0.0% | 97.6% |

#### ⚠️ 위의 `100.0%` 를 단독으로 인용하지 마라 — split이 작고 유리하다

같은 체크포인트를 **0720 held-out 전체**(test 166 + val 100 = **success n=266, 6 takes**)로 재면 **@0.5 recall 86.8%**, @0.85 83.1% 다. 정본 표는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md) 「처음 측정된 held-out success recall (0720)」에 있다.

**두 숫자는 모순이 아니다.** 아래는 두 문서의 표에서 **유도한** 대조다(새 측정이 아니라 산술 확인이며, 반올림 범위 안에서 정확히 맞는다):

```
test 166장 × 100.0%                       = 166 정답
pooled 266장 × 86.8%                      ≈ 231 정답
=> val 100장 중 정답 65장
   그중 병리 take_21_20260720_210234 (n=38) @0.5 recall 7.9% -> 정답 3장
   val 나머지 62장은 전원 정답  =>  62 + 3 = 65  ✔
```

즉 **차이는 전부 val split에 들어 있는 `take_21` 한 take가 만든다.** 그 take는 팔이 컵 위에 머무르거나 컵이 cam2에서 사라져서 실패하고, **threshold로는 구제되지 않는다**(0.05까지 내려도 57.9%인데 0.05는 negative를 오탐한다). take 단위 수치의 정본은 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)다.

> **운영 기준은 보수적인 쪽(266장 = 86.8%)을 쓴다.** 이 문서에서 조건 없이 `100%` 를 인용한 것이 이미 한 번 "classifier는 완벽하다"는 잘못된 결론을 만들었다.

#### 그 외

**FPR은 위 표의 두 체크포인트 모두 0720 test split에서 0.0%다.** 배포 체크포인트(`all3`)에 대해서는 **0720 held-out negative 470프레임**(test + val)에서 **@0.85 / 0.5 / 0.2 모두 0.0%** 로 확인됐고, 이것이 threshold를 0.2까지 내릴 수 있었던 근거다. 다만 **0.1과 0.05에서는 실제로 오탐이 발생한다.** 470프레임 표와 마진 분석의 정본은 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)다.

val failure 최대 확률은 Jul-27이 **0.086**, Jul-24가 **0.488** — 후자는 0.5 임계값에 위험할 만큼 붙어 있다.

에피소드별 시간축으로도 **사람이 찍은 성공 프레임과 사실상 같은 프레임에서 발화**한다(Jul-24 6/6 정확, Jul-27 5/6 정확 + 1개 2프레임 지연).

**채택: Jul-27 `cube_in_cup_all3`.** 근거는 (a) Jul-24는 학습 데이터를 복구할 수 없어 held-out 수치를 신뢰할 근거가 없고, (b) Jul-27의 음성 마진이 약 6배 넓다.
*(역사: 07-28의 0.5에서 2026-07-29에 0.2로 내려갔다(`1b02857`). **현행 production은 2026-07-30에 0.5로 복귀**했으며 §4.7과 코드 상수가 우선한다.)*
🪤 **채택했다고 배포된 것은 아니다.** `checkpoint_sha256()` 의 orbax 제약 때문에 코드 기본값은 아직 폐기된 Jul-24를 가리킨다(§7 P0-0, §12.6-3).

로드/추론 성능 (Kanu GPU, 2026-07-28): `LOAD OK 13.5s`, 워밍업 후 **1.03 ms/frame**. *(§4.8의 7.89 s / 674 ms 는 다른 체크포인트를 laptop3 로컬 CPU에서 잰 값이다 — 비교하지 말 것.)*

**약점**: `take_21` 은 양쪽 체크포인트 모두 약하다(Jul-27은 @0.85에서 recall 0.0%, 즉 아예 발화하지 않는다).

### 12.4 체크포인트 출처 — 확정

```
repo:   github.com/YWhero/hil-serl        ← rail-berkeley 가 아니다
경로:   kanu ~/workspace/youngwoong/hil-serl
branch: agent/cube-in-cup-classifier
commit: d753571702e8d0c0dc37e4989830f3c03a9a32db
실행:   2026-07-27 11:14:38 UTC
wandb:  examples/wandb/run-20260727_111438-yufxxoum/files/wandb-metadata.json
소요:   46초 (150 epoch, 7375 샘플, A4000 1장)
```

wandb metadata에 program 경로·git SHA·전체 argv가 그대로 남아 있어 확정됐다. `--checkpoint_dir` 이 `dataset/cube_in_cup_all3/classifier_ckpt` 를 정확히 가리킨다.

학습 데이터 정합도 맞아떨어진다: success `5186 = 527 + 1784 + 1752 + 1123`, failure `2189 = 601 + 1588` — 로그의 버퍼 크기와 pkl 6개의 샘플 수가 일치한다. "all3" = 0720 + 0724.

uncommitted 수정 2개가 실행 시점에 살아 있었고 **둘 다 동작에 영향이 없음**을 확인했다: `train_reward_classifier.py` 에 wandb 로깅 플래그 추가, `mappings.py` 의 franka import를 `try/except ImportError` 로 감싼 것.

**`5fb716b` 는 이 체크포인트를 만들 수 없다.** 서브모듈을 `c32939b`(rail-berkeley 업스트림 = `d753571` 의 **부모 커밋**)에 핀하고, 당시 어느 gello_software checkout에도 `cube_in_cup` 실험 디렉터리가 없었다. 이전 문서가 `5fb716b` 를 기준으로 가정한 것은 **틀렸다.** *(classifier 학습 코드는 지금도 이 리포에 없다 — kanu의 `hil-serl` fork에만 있다. §12.5)*

### 12.5 kanu 실제 경로 — 2026-07-30 SSH 읽기 전용 재확인 (🗄️ 이전 전 마지막 관측)

> # 🗄️ 이 표는 **kanu**의 디렉터리 배치다 — 현행 서버가 아니다
>
> 2026-07-31에 HIL이 `junhyeong_ai`로 옮겨졌다. 현행 경로는 아래 셋만 기억하면 된다
> (정본: [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](./DATA_AND_MODELS_JUNHYEONG_AI_KO.md)):
>
> | 용도 | `junhyeong_ai` 경로 |
> | --- | --- |
> | HIL 코드 checkout (유일) | `/home/junhyeong/gello_software_runtime` — **독립 clone, worktree 아님** |
> | 데이터·모델 전부 | `/home/junhyeong/hil-serl-data/{demos,classifier_ckpt,datasets,runs,archive}` |
> | learner python | `/home/junhyeong/miniconda3/envs/il/bin/python` (kanu freeze 재현) |
>
> 🚫 **`/home/junhyeong/gello_software` 는 다른 사람의 작업 트리다** — 이름이 kanu 시절과
> 겹치지만 같은 것이 아니다. **읽지도 쓰지도 말 것.** HIL은 `gello_software_runtime`을 쓴다.
>
> 아래 표는 지우지 않는다. classifier 학습처(`~/workspace/youngwoong/hil-serl` @ `d753571`)와
> 학습 데이터 원본은 **kanu에 그대로 남아 있고**(FM/diffusion 스택과 섞여 있어 옮기지 않았다),
> §12.4의 체크포인트 출처 증명이 그 경로들을 인용한다. 재학습용 데이터셋은
> `junhyeong_ai:~/hil-serl-data/datasets/` 로 **복사**됐다 — kanu에서는 아무것도 지우지 않았다.

`/home/laptop3/gello_software`는 laptop 경로이며 GPU 서버에는 없다. **(아래는 2026-07-30
kanu 기준)** production launcher가 쓰던 stable path는 `~/gello_software_hil_current`이고,
당시 schema-3 staging checkout을 가리켰다.

| 경로 (kanu, `~` = `/home/junhyeong`) | 상태 |
| --- | --- |
| `~/gello_software_hil_current` | production stable symlink → `~/gello_software_hil_schema3_stage_20260730` |
| `~/gello_software_hil_schema3_stage_20260730` | `feat/gello-ur7e-humble-22.04` @ `c9c30c3`, clean. 현재 learner cwd |
| `~/hil-serl-data/runs/cube_in_cup_manual_schema3_thr05_20260730_1715` | 현재 production run root; logs/checkpoints/W&B/assets |
| `~/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl` | 사람 승인 offline demo 2,037 transition, SHA `f9718558…032fa` |
| `~/workspace/youngwoong/hil-serl` | **`agent/cube-in-cup-classifier` @ `d753571`** (YWhero fork) — **classifier 학습처.** 학습·CV·평가 코드는 오직 여기에만 있다 |
| `~/workspace/youngwoong/dataset/cube_in_cup_all3/` | 학습 데이터 + **Jul-27 체크포인트** |
| `~/workspace/youngwoong/gello_software` | detached `0c8a5a8`, **dirty** — 쓰지 말 것 |
| `~/workspace/youngwoong/gello_software_remote_classifier` | `feat/remote-cube-classifier-viewer` @ `a2733ee` — ZMQ 뷰어 + Jul-24 체크포인트 |
| `/tmp/gello-hil-rl-receive-server-v2` | 옛 receive-only worktree `5fb716b`; production launcher가 사용하지 않음 |
| python | `/home/junhyeong/miniconda3/envs/il/bin/python` |

**2026-07-30 시점에는 PID 1112465 production learner가 port 50053/GPU 5에서 healthy였다.**
그 이전의 "HIL process 0개" 스냅샷은 그 시점 기준으로 이미 낡은 것이었다.
🗄️ **그리고 그 PID 자체가 이제 kanu 쪽 역사다** — 2026-07-31 현재 운영 learner는
`junhyeong_ai`에서 돈다. kanu의 옛 프로세스는 아직 살아 있으며 **사용자가 종료할 대상이지
우리가 신호를 보낼 대상이 아니다.** overlay venv `/tmp/gello-hil-rl-receive-overlay-v2`
(kanu)는 여전히 재사용 금지다(§11.5).

### 12.6 함정

1. **`checkpoint_150` 이 5개다.** 같은 12분 세션 산출물이고 크기까지 비슷하다:
   `cube_in_cup_combined`(11:04:16), `cv/fold_take_01`(11:10:41), `fold_take_02`(11:11:59), `fold_take_03`(11:13:22), **`cube_in_cup_all3`(11:15:24) ← 이것만 우리 것**
2. ⚠️ **~~서빙 코드는 세 번째 저장소에 있다~~ → 2026-07-29에 이 리포로 들어왔다** (`3ff5f80`, `3f199d4` 로 머지). 이제 ZMQ 뷰어 경로가 canonical checkout에 있다: `serl_ur_infra/remote_reward_classifier_server.py`, `serl_ur_infra/run_remote_reward_classifier_server.sh`, `ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_runtime.py`, `ros2_ur_ws/run_remote_classifier_viewer.sh`.
   - 07-28 판본이 경고했던 *"그 서버의 `--checkpoint` 기본값이 Jul-24 옛 모델을 가리킨다"* 는 **별도 저장소 `gello_software_remote_classifier @ a2733ee` 쪽 이야기이고 그쪽은 그대로다.** 이 리포의 `run_remote_reward_classifier_server.sh:24` 는 기본값이 `$REPO_ROOT/classifier_ckpt/cube_in_cup_all3` (Jul-27 계열)이고 `REWARD_CLASSIFIER_CHECKPOINT` 로 덮어쓸 수 있다. **어느 쪽 스크립트를 돌리는지 확인할 것.**
3. ✅ **~~`checkpoint_sha256()` 이 `os.path.isfile` 을 요구한다~~ → 해소 (G19, 2026-07-29).**
   - *07-29 오전 판본:* "`checkpoint_sha256()`(`ur_env/rlpd_receive_server.py:148-159`)이 `os.path.isfile` 을 요구한다. Jul-27 체크포인트는 **orbax OCDBT 디렉터리**라 그대로 넣으면 즉시 `FileNotFoundError`. 최신 체크포인트를 gRPC 경로에 쓰려면 이 코드를 먼저 고쳐야 한다 — 그래서 기본 SHA가 아직 폐기된 Jul-24(`e329986b…`)를 가리킨다." — **둘 다 고쳐졌다.**
   - 이제 `checkpoint_sha256()` 은 재귀 `directory_sha256()` 에 위임한다(`ur_env/classifier_sidecar.py::directory_sha256`). 파일/디렉터리 모두 받고, **단일 파일은 예전과 완전히 같은 순수 content sha256** 이라 이미 런북에 적힌 단일 파일 pin은 그대로 유효하다. 디렉터리는 POSIX relpath 정렬 순서로 `relpath\0size\0contents` 를 먹여 rename·재분할에 민감하다.
   - 기본 SHA 상수 2개(`run_rlpd_learner_server.py::DEFAULT_CLASSIFIER_CHECKPOINT_SHA256`, `run_rlpd_receive_server.py::DEFAULT_CHECKPOINT_SHA256`)가 **`512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d`** (= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`, **파일 14개**)로 교체됐다. **이 세션에서 `directory_sha256()` 을 직접 돌려 상수와 일치하는 것을 확인했다.**
   - 🪤 **G19를 G15와 따로 내보내면 안 됐던 이유**: 디렉터리 해싱만 고치고 상수를 그대로 뒀다면, 서버가 은퇴한 recall-0% 체크포인트로 **아무 에러 없이 깨끗하게 기동**해서 모든 transition에 reward 0을 영원히 내보낸다. RLPD는 계속 학습하면서 아무것도 배우지 않는다. **이 시스템 최악의 실패 모드가 정확히 그것**이라 pin은 운영자 플래그가 아니라 코드 기본값으로 박아 뒀다.
4. **`serl_launcher` 는 어디에도 pip 설치돼 있지 않다** — 순전히 `PYTHONPATH` 로 해결된다. 재현하려면 `PYTHONPATH=.../hil-serl/serl_launcher`, cwd `.../hil-serl/examples`.
5. **환경 드리프트** *(2026-07-28 kanu 측정)*: kanu `il` env의 numpy가 **2.2.5**인데 lock은 1.26.4(메이저 점프), orbax 0.11.12 vs 0.11.5, grpcio 1.80.0 vs 1.74.0. 런타임 fail-closed 강제 대상은 jax/flax/distrax/tfp/wandb뿐이라 **이 3개는 자동으로 안 걸린다.**
   - 📌 **서버 이전으로 이 함정이 사라지지는 않았다.** `junhyeong_ai`의 `il` env는 **kanu의 `pip freeze` 그대로** 재현했으므로 fail-closed 대상 5개는 정확히 일치하지만, **핀 밖의 세 패키지가 같은 값으로 따라왔는지는 이번에 재측정하지 않았다.** 알고 싶으면 새 서버에서 직접 읽어라 — 추정하지 말 것.
   - 🪤 같은 이유로 **새 서버에서 즉흥 `pip install` 금지**다(§7 P0-1). 실제로 검증 중 `pip install flax==0.10.5 optax==0.2.4`가 jax를 0.5.3 → 0.6.2로 조용히 올렸다.

### 12.7 수정 방향 — **채택된 것은 (b)다** (2026-07-29)

> ## 🔴 결론이 뒤집혔다 — 이 절 아래쪽의 "**채택은 (a)**"를 따르지 마라
>
> 07-28 판본은 대역폭을 근거로 **(a) 크롭으로 재학습**을 채택했다. **실제로 나간 것은
> (b) 분류기용 이미지 별도 전송이다.** 아래 대조표의 (b) 열 자체가 두 군데 틀렸고,
> 그 두 칸이 결정을 뒤집었다.
>
> | 07-28이 (b)에 대해 적은 것 | 실제 (코드 확인, 2026-07-29) |
> | --- | --- |
> | "canonical schema hash **변경**" | ❌ **불변.** hash는 `CANONICAL_OBSERVATION_SPEC` **문서**에서 파생되고 wire payload에서 파생되지 않는다. sidecar는 canonical 검증 **전에** 벗겨진다. proto도 안 바뀌었다 — 전송이 이미 generic named-tensor map이다 |
> | "gRPC 대역폭 **약 2배**" | ❌ **~2 Hz 게이팅 + 랩톱 측 128×128 다운스케일로 쌍당 ~10–15 KiB.** 매 스텝 96.1 KiB에 붙는 것이 아니다 |
>
> 07-28의 (b) 비용 추정이 틀린 이유는 명확하다 — **매 스텝, 원본 해상도**를 가정했다.
> 실제 설계는 둘 다 하지 않는다. 다만 **원본 JPEG 그대로 넘기기는 실제로 불가능한 게
> 맞았다**: q95 720p가 206/194 KiB, 쌍 400 KiB, 2 Hz에 6.55 Mbit/s로 13 Mbit/s 링크를
> 넘고 첨부 1회가 **+252 ms**(예산 100 ms)다. 그래서 랩톱이 먼저 128×128로 줄인다.
>
> **(a)를 안 택한 실질 이유** — 재학습은 §12.3과 threshold 문서의 **측정값을 전부
> 무효화**한다. (b)는 분류기 입력이 여전히 무크롭이라 **한 줄도 다시 재지 않아도 된다.**
>
> **결정 후속 조치 (전부 코드에 반영됨):**
> - 서빙 `reward_model_id` = **`cube-in-cup-all3-ckpt150+sidecar-v1`**. 체크포인트 **와**
>   입력 계약을 같이 이름 짓는다 — pre-sidecar actor ↔ post-sidecar server(또는 그 반대)
>   조합이 **핸드셰이크에서 거부**된다. 안 그러면 양쪽이 서로 다른 픽셀로 reward를
>   계산하면서 세션 전체를 돌게 된다.
> - `run_contract["reward_classifier"]` 에 **`input_contract`**(= `CLASSIFIER_INPUT_ID`)와
>   **`success_confirmations`** 가 추가됐고 둘 다 learner fingerprint에 들어간다
>   (`run_rlpd_learner_server.py` 의 `run_contract` 조립부).
> - 🪤 **따라서 옛 checkpoint resume은 fail-closed로 거부된다.** **1회성이고 의도된
>   단절**이다 — 옛 계보는 recall 0% 체크포인트가 매긴 reward로 학습됐으므로 구제할
>   가치가 없다. 새 `--checkpoint-root` 로 한 번 시작하면 fingerprint는 다시 안정된다.
>   `ur_env/learner/composition.py::prepare_learner_state` 가 이 상황을 **거부 메시지
>   안에서 설명**하므로 버그처럼 읽히지 않는다.
> - **`--cam1-crop`/`--cam2-crop` 재학습 절차(아래)는 실행하지 마라.** 참고 자료로만
>   남긴다. (a)를 나중에 다시 검토할 사람을 위한 기록이다.

*(이하 07-28 조사 원문 — upstream 패턴 근거로서는 여전히 유효하다.)*

upstream `usb_pickup_insertion` 은 **같은 물리 카메라를 두 키로 두 번 등록**해 분류기에 전용 크롭을 준다:

```python
"side_policy":     {"serial_number": "130322274175", ...}
"side_classifier": {"serial_number": "130322274175", ...}   # 같은 카메라
IMAGE_CROP = {"side_policy":     img[250:500, 350:650],
              "side_classifier": img[270:398, 500:628]}     # 정확히 128×128 — 리샘플링 0
image_keys      = ["side_policy", "wrist_1", "wrist_2"]
classifier_keys = ["side_classifier"]
```

`object_handover` 도 동일하다. 반면 `ram_insertion` 은 `classifier_keys == image_keys` — 분류기가 정책 크롭을 그대로 먹는다. **둘 다 upstream 계약 안이며, 선택은 우리 몫이다.**

> ### ⛔ 아래 비교표는 **폐기됐다** — (b)를 기각한 두 근거가 둘 다 틀렸다
> 원문을 삭제하지 않고 남긴다. 2026-07-30에 **(b)를 채택**했고, 그 과정에서 이 표의
> 결정적인 두 칸이 측정으로 뒤집혔다:
>
> | 이 표의 주장 | 실제 (2026-07-30 측정) |
> | --- | --- |
> | (b)는 schema hash를 **변경**한다 | **변경되지 않는다.** 해시는 `CANONICAL_OBSERVATION_SPEC`에서만 파생되고, 사이드카는 canonical 검증 **이전에** `actor_network.py`에서 분리된다. `3459098d…0352903` 그대로 |
> | (b)는 대역폭이 **약 2배** | **+2.8%.** "2배"는 *raw uint8을 매 스텝 10 Hz로* 보낸다는 가정이었다. 실제는 2 Hz + 랩톱 측 128×128 인코딩 = 쌍당 **13.32 KiB** |
>
> 그리고 (a)에는 이 표에 없던 비용이 있다 — **재학습은 threshold 측정 세트 전체를 무효화한다**
> (`REWARD_CLASSIFIER_THRESHOLD_KO.md`의 수치가 전부 무크롭 기준). (b)는 분류기가 계속
> 무크롭을 먹으므로 그 측정이 **그대로 살아 있다.** 이것이 결정을 뒤집은 가장 큰 요인이다.
> 아래 §12.7의 정정 표를 함께 볼 것.

| | (a) 크롭으로 재학습 | (b) 분류기용 이미지 별도 전송 |
| --- | --- | --- |
| canonical schema hash | 그대로 | **변경** (actor·server 동시 수정) |
| gRPC 대역폭 | 그대로 96.1 KiB/step | **약 2배** ← RTT p99가 이미 97.1 ms / 100 ms 예산 (§11.4) |
| upstream 패턴 | `ram_insertion` | `usb_pickup_insertion` |
| 비용 | **46초** 재학습 + 재export | 코드 변경 + 대역폭 |
| 위험 | 크롭 값이 체크포인트에 각인 | 없음 |

~~**대역폭이 결정적이다** — WiFi 병목이 약 13 Mbit/s인데 현재 7.9 Mbit/s를 쓰고 있어 이미지 쌍을 하나 더 보내면 예산을 넘긴다. 유선이면 (b)도 실현 가능하다. **채택은 (a)** — 크롭 값은 데이터셋 실측이고 정책이 1차 소비자다.~~

> **⛔ 위 결론은 뒤집혔다(이 절 머리의 블록 참고). 채택된 것은 (b)다.**
> 대역폭 분석의 전제가 "**매 스텝, 원본 해상도**"였는데 실제 설계는 **~2 Hz + 랩톱 측
> 128×128 다운스케일**이라 쌍당 ~10–15 KiB다. "크롭 값은 데이터셋 실측이고 정책이 1차
> 소비자"라는 문장 자체는 계속 맞다 — 그래서 `IMAGE_CROP` 을 **건드리지 않은** 쪽으로
> 갔다.

**⚠️ 아래는 채택되지 않은 (a) 경로의 실행 정보다. 자료로만 남긴다 — 실행하지 마라.**

*(📌 2026-07-31 서버 이전에서 **classifier 학습 코드는 kanu에 남겨 뒀다** — `~/workspace/youngwoong/hil-serl` @ `d753571`은 FM/diffusion 스택과 같은 디렉터리에 있고 git에서 복원 가능하다. 재학습용 **데이터**만 `junhyeong_ai:~/hil-serl-data/datasets/` 로 복사했다. 즉 아래 경로는 지금도 kanu 기준이다.)*

kanu `cube_classifier_pipeline.py` 에 **`--cam1-crop`/`--cam2-crop` 이 이미 배선돼 있다**(`preprocess_frame` 까지 연결됨 — 경로만 안 썼을 뿐). 변환할 때 **인자 순서가 뒤바뀐다는 점에 주의**:

```
이 리포는 y0,y1,x0,x1 / 파이프라인은 x0,y0,x1,y1

cam1  img[20:670, 340:990]  →  --cam1-crop 340,20,990,670
cam2  img[0:720, 420:1140]  →  --cam2-crop 420,0,1140,720
```

*(위 두 줄은 `ur_experiments/cube_in_cup.py:195-197` 주석의 값과 일치함을 2026-07-29에 대조했다. 파이프라인 쪽 인자 존재 여부는 kanu에 있어 이 세션에서 재확인하지 못했다 — 07-28 조사 결과를 그대로 옮긴 것이다.)*

**(a)를 택하면 반드시 크롭 값을 체크포인트 옆에 sidecar로 기록하고, 서버가 불일치 시 fail-closed 하게 할 것.** 안 그러면 나중에 `IMAGE_CROP` 을 바꿨을 때 또 조용히 깨진다 — 이 결함이 정확히 그렇게 생겼다.

> ⚠️ 07-28 판본은 *"미적용 패치 초안이 scratchpad에 있다: `01-actor-classifier-preprocessing-contract.diff`, `02-training-fork-crop-and-sidecar.diff`, `03-remote-classifier-repo-mirror.diff`"* 라고 적었다. **2026-07-29에 찾을 수 없다** — 세션 scratchpad는 영속 저장소가 아니다. 위 세 이름은 **의도의 기록으로만** 남긴다. 없는 파일을 찾느라 시간 쓰지 말 것.

### 12.8 남은 열린 질문

1. **whole-take 라벨링의 영향.** 0724 데이터는 take 전체를 통째로 성공/실패로 라벨링했다. held-out FPR 0.0%가 이를 상당 부분 방어하지만, 프레임 단위 라벨과 whole-take 라벨이 섞인 학습이 경계 근처 판정에 어떤 영향을 주는지는 미측정.
2. **실기 분포.** held-out은 전부 녹화 데이터다. 렌즈 개체차·색감·장착 각도 미세 차이가 실기에서 어떻게 작용하는지는 실제로 돌려봐야 안다. **ZMQ 뷰어로 텔레옵하며 `p(success)` 곡선을 보는 것이 가장 직접적인 확인이고, 이제는 gRPC 경로도 같은 무크롭 입력을 먹으므로 뷰어와 서버 판정이 일치한다**(§12.1). §8-A-1.
   - ✅ *"현재 붙어 있는 카메라가 녹화 개체와 같은지 확정되지 않았다"* 는 **해소됐다.** 카메라는 한 쌍뿐이고 시리얼 혼란은 필드 차이였다(§11.10). USB 포트 순서 차이는 여전히 사실이지만, 역할 배정은 `_resolve_camera_serials.sh` 가 버스에서 해석한다.
3. 🔴 **`take_21` 급 실패 모드 — 이번 변경으로 고쳐지지 않았다.** 팔이 컵 위에 머무르거나 컵이 cam2에서 사라지면 threshold로는 구제되지 않는다: @0.85 recall `0.0%`, @0.05 로 내려도 `57.9%` (§12.3, [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md) `:369`). 팔이 cam1을 쓸고 지나가는 동안 확률이 `0.005 ↔ 1.0` 으로 진동한다.
   - **근본 원인은 시야/가림이고 전처리도 라벨도 아니다.** sidecar 분리로도, 재학습으로도, threshold로도 안 없어진다.
   - **현재 완화책**: sidecar의 **정지 게이트**(`stationary_speed_max`, 움직이는 동안 아예 분류하지 않는다) + **~2 Hz 희소 분류**. 완화지 해결이 아니다.
   - **진짜 해법은 팔이 가로지르지 않는 카메라 배치다.** 이게 이 절에 남은 가장 큰 미해결 항목이다.
   - N-of-M 시간 평활 배선은 존재하지만 **기본 OFF**(`--success-confirmations 1`)다. 켜면 라이브 뷰어와 판정이 어긋나므로, 켤 거면 그 대가를 알고 켤 것.
4. ⛔ ~~**크롭 재학습 후 성능 재측정.**~~ — **불필요해졌다.** (a)를 채택하지 않았고 분류기 입력이 여전히 무크롭이므로 **§12.3의 held-out 수치는 그대로 유효하다**(§12.1 「왜 재학습이 아니라 분리인가」). 이 항목이 예고하던 전면 재측정 비용을 피한 것이 분리 방식의 핵심 이득이다.
5. **`stationary_speed_max` 실측.** `cube_in_cup.py::CLASSIFIER_SIDECAR` 의 `0.05 m/s` 는 **코드 주석이 스스로 `PLACEHOLDER` 라고 밝힌 값**이다. 의도는 "팔이 가라앉았다"이고, 옳은 값은 릴리스 후 조작자가 기다리는 동안 TCP가 실제로 머무는 속도다 — 녹화 take에서 재야 한다. 너무 낮으면 게이트가 안 열리고, 너무 높으면 모션 블러를 분류한다.
6. ✅ **sidecar의 실물 actor→learner 경로는 투입 완료** (2026-07-29 kanu, **2026-07-31 `junhyeong_ai`에서 재확인** — 조작자 실기 세션 replay 316 / intervention 210). 🪤 **같은 날의 합성 수락 시험(§5.8)은 이것을 증명하지 못한다** — 그 도구는 sidecar를 안 붙이므로 `classifier_success_count: 0`이 "분류기가 다 떨어뜨렸다"가 아니라 **"한 번도 안 불렸다"** 는 뜻이다. 남은 질문은 MANUAL/AUTO 각각의 장시간 verdict 분포, 정지 게이트와 팔 가림 조건에서의 품질이다(§1 판정표).
