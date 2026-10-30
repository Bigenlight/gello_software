# kanu learner 임시 대체 — **테스트 브랜치 전용**

> # 🛑 이 문서는 브랜치 `test/kanu-learner-fallback` 에서만 유효하다
>
> **왜 있나.** `junhyeong_ai`의 GPU 1장이 **무기한 점유**됐다. 그래서 HIL-SERL **learner만**
> 옛 서버 `kanu`(RTX A4000 ×8)로 **임시로** 옮겨 돌린다.
>
> **왜 지금은 참을 만한가.** 예전이라면 느린 learner는 곧 느린 로봇이었다 — 정책 추론이
> 제어 루프 **안**에 있었기 때문이다. `HIL_POLICY_MODE=local`(→
> [`HIL_LOCAL_INFERENCE_KO.md`](HIL_LOCAL_INFERENCE_KO.md)) 이후로는 아니다. kanu의 느린
> learner step은 **로봇 지연이 아니라 파라미터 노후(staleness)** 로 나타난다. 액션은 laptop3가
> 로컬에서 만든다. **이 조합이 아니면 kanu 대체는 하지 않는다.**
>
> **main 브랜치는 안 건드렸다.** kanu 전용 물건(래퍼 스크립트 + 이 문서)은
> `feat/gello-ur7e-humble-22.04`에 **하나도 올라가지 않았다.**
> **예외 딱 하나** — ssh `ControlPath` 수정(`15a467e`)은 **main에 올렸다.** 그건 kanu 전용이
> 아니라 이 시험이 **찾아낸 main 브랜치의 결함**이었다(§7).
>
> **`junhyeong_ai`에 한 일은 `git pull --ff-only` 한 번뿐이다.** 데이터 루트
> (`~/hil-serl-data`)에 쓰기 **0회**, 프로세스 **0개** — 실측으로 확인했다
> (`find ~/hil-serl-data -newermt 2026-08-10` 무출력). 그 pull(18:19:28,
> `6bc7644` → `15a467e`)은 위의 main 수정을 받아 간 것이고 **빼면 안 되는 것이다**:
> launcher가 laptop3 HEAD와 learner 호스트 HEAD를 비교하므로, main으로 돌아왔을 때
> 두 쪽이 어긋나 있으면 정상 서버로 새 lineage를 시작할 수 없다.
> ⚠️ junhyeong_ai는 **테스트 브랜치를 fetch한 적이 없다**(의도적) — 그래서 테스트 브랜치에
> 서 있는 동안 junhyeong_ai를 부르면 큰 배너가 뜬다. **§5 함정 6.**
>
> 🛑 **실기 미검증.** 아래 절차는 **dry run까지만** 검증됐다(§4는 그 dry run의 실측이다).
> **kanu 모드로 로봇 세션을 돌린 적은 아직 없다.**
>
> ⚠️ **쓰기 전에 전제부터 확인할 것.** 이 문서의 존재 이유는 "junhyeong_ai GPU가 무기한
> 점유됐다"이다. 그 전제는 **날짜가 붙은 관측**이지 항구적 사실이 아니다 — 2026-08-10 18:35
> 재확인 시점에 junhyeong_ai GPU 0은 **완전히 비어 있었다**(15 MiB / 16303 MiB, 0 %,
> compute app 0개). 비어 있으면 **kanu를 빌릴 이유가 없다**: main 브랜치로 평소대로 돌리는
> 쪽이 언제나 더 빠르고 검증도 더 많이 됐다. 확인:
> `ssh junhyeong_ai nvidia-smi --query-compute-apps=pid,used_memory --format=csv`

---

## 1. 한 줄

**learner와 reward 권위는 kanu로, 정책 추론은 laptop3로.** 터널의 로컬 끝
(`127.0.0.1:50153`)도, actor가 다이얼하는 프록시 포트(`50253`)도 **그대로**다 — 바뀌는 것은
터널 **반대쪽 호스트**뿐이다.

```
laptop3 (test 브랜치)                        kanu (test 브랜치 checkout)
  run_hil_server_kanu.sh ─ env 5개 ─────────► learner @ GPU 1, il python
  run_hil_session.sh HIL_POLICY_MODE=local     ~/hil-serl-data/
    · 로컬 정책 추론 (액션)                      ├ demos/            (SHA 핀 통과)
    · 전이 forward (백그라운드)   ── tunnel ──►  ├ classifier_ckpt/  ← K1이 새로 복사
    · params ssh pull  ◄────────── ssh ────────  └ runs/<run_root>/params_live/
```

---

## 2. kanu에 준비한 것 — **파일 하나뿐이다**

kanu의 `~/hil-serl-data/`에는 `classifier_ckpt/`가 **없었다.** 그 흩어짐이 바로
`DATA_AND_MODELS_JUNHYEONG_AI_KO.md` §6이 "kanu는 이제 이 스크립트로 못 본다"고 적은 이유다
(`HIL_REMOTE_DATA_ROOT` 하나로 kanu의 배치를 표현할 수 없었다). **그 구멍 하나만 메웠다.**

| | |
| --- | --- |
| 원본 (읽기 전용) | `kanu:~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150` |
| 새 사본 | **`kanu:~/hil-serl-data/classifier_ckpt/checkpoint_150`** |
| directory SHA256 | `512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d` — **검증 완료.** 서버 SHA 핀과 동일 |
| 방식 | `cp -a` (복사다. **원본은 그대로 있다**) |

🛑 **`~/workspace/youngwoong/**` 는 다른 스택(FM/diffusion)의 트리다 — 읽기만 했고 한 바이트도
쓰지 않았다.** 루트 `CLAUDE.md` §E의 "섞지 말 것"이 그대로 적용된다.

그 밖에 kanu에서 한 쓰기는 **HIL checkout의 `git fetch` + 테스트 브랜치 checkout**뿐이다
(`~/gello_software_hil_current` → `gello_software_hil_schema3_stage_20260730`). demo pickle
(`demos/cube_in_cup_20260720_success_23takes.pkl`, SHA `f9718558…032fa`)은 **이미 있었고 손대지
않았다.** conda `il` env도 그대로다 — `validate_learner_dependencies`의 exact 핀(jax 0.5.3 /
jaxlib 0.5.3 / flax 0.10.5 / distrax 0.1.5 / tfp 0.25.0 / wandb 0.26.0)을 **있는 그대로 통과**한다.
사실 junhyeong_ai의 `il` env가 **이 env를 복제한 것**이다.

---

## 3. 조작자 CLI — 터미널 3개 + 확인 1개

전부 `cd /home/laptop3/gello_software/ros2_ur_ws`에서 친다. 브랜치는
`test/kanu-learner-fallback`이어야 한다(`git branch --show-current`).

### T1 — kanu learner + 터널

```bash
HIL_PARAMS_EXPORT=1 HIL_EXTERNAL_POLICY_INGEST=1 HIL_LATENCY_PROFILE=1 \
  ./run_hil_server_kanu.sh
```

- `run_hil_server_kanu.sh`는 **환경변수 5개를 더하고 `exec`하는 얇은 래퍼**다
  (`HIL_SSH_HOST=kanu` · `HIL_GPU_INDEX=1` · `HIL_REMOTE_REPO=/home/junhyeong/gello_software_hil_current` ·
  `HIL_REMOTE_PYTHON=/home/junhyeong/miniconda3/envs/il/bin/python` ·
  `HIL_REMOTE_DATA_ROOT=/home/junhyeong/hil-serl-data`). **argv는 손대지 않는다** — `--check`,
  `--new-lineage`, `--run-id`, `--help` 전부 `run_hil_server.sh` 문서 그대로 동작한다.
  게이트(process contract · checkpoint SHA · GPU 점유 거부 · 교차 호스트 HEAD 비교 · lock)는
  **전부 원래 스크립트에 그대로 남아 있다.**
- 출력의 **`HIL_SERVER_RESULT=started`** 를 확인한다. `reused`면 §5 함정 1이다.
- 읽기 전용 확인은 `./run_hil_server_kanu.sh --check`.
- **GPU 1이 이미 차 있으면** 앞에 `HIL_GPU_INDEX=2`를 붙인다(명시 override가 이긴다):
  ```bash
  HIL_GPU_INDEX=2 HIL_PARAMS_EXPORT=1 HIL_EXTERNAL_POLICY_INGEST=1 HIL_LATENCY_PROFILE=1 \
    ./run_hil_server_kanu.sh
  ```
  launcher는 **compute app이 있는 GPU에서 기동을 거부**하므로, 낡은 선택은 조용히 남의 작업 위에
  올라타는 대신 **큰 소리로 실패**한다.

### T2 — 하드웨어 (**무변경**)

```bash
./run_hil_hardware.sh
```

### T3 — 세션 (로컬 추론 + kanu를 가리키게)

```bash
HIL_POLICY_MODE=local HIL_SSH_HOST=kanu \
  HIL_SERVER_SCRIPT="$PWD/run_hil_server_kanu.sh" \
  [HIL_LATENCY_PROFILE=1] ./run_hil_session.sh
```

🪤 **환경변수 2개가 필요하고, 두 번째가 안 뻔한 쪽이다.** T1의 export는 **자식 프로세스에서
부모로 거슬러 올라가지 않는다.**

| 변수 | 무엇이 읽나 | 없으면 |
| --- | --- | --- |
| `HIL_SSH_HOST=kanu` | `run_hil_local_policy.sh`의 ssh fallback과 프록시의 `SshParamsFetcher`(`params_sync.SSH_HOST_ENV_VAR`) | params를 `junhyeong_ai`에서 찾는다 |
| `HIL_SERVER_SCRIPT=…/run_hil_server_kanu.sh` | "지금 살아 있는 run root가 어디냐"를 판정하는 **정본 probe**가 `run_hil_server.sh --check`인데, 그 probe가 이 래퍼를 거쳐야 kanu에게 묻는다 | probe가 **junhyeong_ai의 run root 경로**를 답하고, 프록시는 그 경로를 **kanu에서** 찾는다 |

📌 래퍼 자신은 `HIL_SERVER_SCRIPT`를 **읽지 않는다**(항상 자기 형제 `run_hil_server.sh`를
exec한다). 그래서 위 사용법이 **재귀하지 않는다** — 테스트로 고정돼 있다.

📌 **ssh control-dir override는 이제 필요 없다.** `15a467e` 이전에는 필요했다(§7).

### T4 — 조용한 실패 점검 (**세션 시작 직후 한 번, 필수**)

```bash
grep -c 'TRANSITION NOT INGESTED' /tmp/hil_session_*/local_policy.log   # 0 이어야 한다
```

0이 아니면 **서버가 전이를 전부 거절하고 있는데 로봇은 완벽히 정상으로 도는 상태**다
(§5 함정 2). 세션 로그는 조작자 터미널이 아니라 `/tmp/hil_session_*/`에 있다.

---

## 4. 실측 기대치 — **2026-08-10 dry run** (로봇 없음)

kanu GPU 1에서 실제로 learner를 띄우고 잰 값이다. **junhyeong_ai 열은 비교용**이다.

| 항목 | **kanu (실측)** | junhyeong_ai | 읽는 법 |
| --- | --- | --- | --- |
| 기동 → health READY | **약 2.5분** (JAX 워밍업 91.2 s: 컴파일 사이클 48.05 / 42.68 / 0.439 s) | ~65 s | T1이 오래 걸리는 게 **정상이다.** 죽은 게 아니다 |
| learner step (정상 상태) | **≈ 439 ms** | ~110 ms | 약 **4배** 느리다. 이것이 아래 두 줄의 원인 전부다 |
| params publish 간격 | **≈ 22 s** (50 step마다) | ~5.8 s | learner step × 50 |
| params staleness | **≈ 22–25 s** | ~6–9 s | 🟢 **무해하다 — 액션은 로컬에서 만든다.** 정책 지연(policy lag)이지 제어 지연이 아니다 |
| 전이 소비 속도 | 더 느리다 | — | ⚠️ uploader **백로그가 더 크게 자라고**, `WAIT_HOME_APPROVAL`/`WAIT_SCENE_READY` 대기 화면에서 빠진다. 이 설계의 전제(`HIL_LOCAL_INFERENCE_KO.md` §5.1)가 **kanu에서 더 세게 당겨진다** |
| Health RPC (터널 경유) | **중앙값 4.8 ms** | — | 네트워크는 문제가 아니다 |
| params fetch (ssh) | **cold 0.697 s / warm 0.023 s** | cold 0.378 / warm 0.015 | ControlMaster 재사용이 살아 있다는 증거 |
| SIGINT → 종료 | **5.4 ms**, 클린 | — | dry run 후 kanu에 learner를 남기지 않았다 |

부가 관측(전부 실측): 기동 프로세스 env에 opt-in 3개가 모두 실렸고, offline demo **2,037개**
로드, `jax_backend=gpu`, `utd_ratio=5`, external policy ingest **enabled**,
`params_live/params_v00000000.msgpack` **12,385,459 B**.

📌 **v0 blob은 health-ready보다 91 s *먼저* 나왔다.** 즉 워밍업이 끝나기 전에도 프록시가 물
파라미터가 이미 존재한다. 다만 T3는 T1의 READY를 기다리는 것이 정상 절차다 — 이 사실은
"params가 없다"는 진단을 배제할 때 쓴다.

📌 **`logs/latency_server.jsonl`은 dry run 뒤에 없다. 정상이다.** `LatencyProfiler`는
디렉터리 생성도 파일 열기도 **첫 커밋 때 지연 수행**하는데, 로봇이 없으면 Step RPC가 0건이라
커밋할 표본이 없다. **profiling이 켜졌다는 증거는 파일이 아니라 로그 한 줄**이다 —
learner 기동 로그의 `{"event":"rlpd_learner_latency_profile","path":"…/latency_server.jsonl"}`
(dry run에서 확인함). 실기 세션에서는 첫 전이와 함께 파일이 생긴다.

---

## 5. 함정

### 1) 🪤 재사용된 learner에는 환경변수가 안 먹는다 (main과 동일)

`run_hil_server.sh`는 **살아 있는 learner를 재사용**한다(`HIL_SERVER_RESULT=reused`). 환경변수는
**프로세스 기동 시점에만** 전달되므로, 재사용된 learner에는 `HIL_PARAMS_EXPORT`도
`HIL_EXTERNAL_POLICY_INGEST`도 `HIL_LATENCY_PROFILE`도 **붙지 않는다.**
`params_live/`가 안 생기면 **T1 출력의 `HIL_SERVER_RESULT`부터 본다.**
로컬 모드 첫 세션은 **learner를 새로 띄워야 한다.**

### 2) 🛑 EXPORT만 켜고 INGEST를 빼면 **조용히** 반쪽으로 돈다 (main과 동일)

`HIL_PARAMS_EXPORT`가 빠지면 프록시가 health-ready가 안 되어 **즉시 안다.**
`HIL_EXTERNAL_POLICY_INGEST`가 빠지면 정반대다 — **팔이 움직이고 개입이 먹고 episode가 끝나는데
서버만 전이를 전부 거절한다.** 그래서 §3의 T4 `grep`이 절차의 일부다. 전문은
[`HIL_LOCAL_INFERENCE_KO.md`](HIL_LOCAL_INFERENCE_KO.md) §3의 🛑.

### 3) ⚠️ kanu GPU 8장은 **여러 사람이 공유한다** — 예의가 안전장치다

2026-08-10 실측: **GPU 1·2가 비어 있었고**, **3 / 4 / 7은 100% 사용 중, 6은 25%**, 0·5에 잔여
메모리. 우리 몫은 **GPU 1**이고, 막히면 **2**다.
🛑 **`HIL_GPU_INDEX`를 3/4/6/7로 override하지 말 것 — 남의 작업이다.** launcher가 비어 있지 않은
GPU에서의 기동을 거부하지만, **그 거부는 마지막 방어선이지 허가가 아니다.**

### 4) ⚠️ kanu 디스크가 **95% 사용 중**이다

run root가 `~/hil-serl-data/runs/` 아래에 계속 쌓인다. 시험이 길어지면 **옛 테스트 run root를
정리**한다. 지금 당장 막힌 것은 아니지만(여유 ~94 GB) 여기는 junhyeong_ai가 아니다.

### 5) 🛑 실기 미검증 스탬프

**dry run만 했다.** 로봇을 붙인 kanu 모드 세션은 **0회**다. §4의 숫자는 전부 로봇 없는 측정이고,
"실제 루프 주기가 얼마가 되는가"는 여전히 **미측정**이다
([`HIL_LOCAL_INFERENCE_KO.md`](HIL_LOCAL_INFERENCE_KO.md) §5.6과 같은 상태).

### 6) 🪤 이 브랜치에 서 있는 동안 **junhyeong_ai를 부르면 무서운 배너가 뜬다**

테스트 브랜치에서 평소의 `./run_hil_server.sh`(= junhyeong_ai)를 그냥 치면 이렇게 나온다
(2026-08-10 실측):

```
  !!  LEARNER HOST / LAPTOP3 CODE IDENTITY MISMATCH  !!
  laptop3 : 35efda2…  (test/kanu-learner-fallback)
  remote  : 15a467e…  (feat/gello-ur7e-humble-22.04)
  relation: remote_never_fetched
```

**이건 고장이 아니라 사실이다.** junhyeong_ai는 테스트 브랜치를 **fetch한 적이 없고**, 그게
의도다(그 서버는 main만 본다). 다만 배너 문구가 `HIL_SERL_KANU_RUNBOOK_KO.md` §1.1의 **옛
origin 사고**("fetch가 성공하고 아무것도 안 가져왔다")를 서술하도록 쓰여 있어서, 여기서는
**맞는 relation에 틀린 서사**가 붙는다.

| 무엇을 하려는가 | 결과 |
| --- | --- |
| `./run_hil_server.sh --check` | 배너 + 정상 보고. **죽지 않는다** (`--check`와 reuse는 불일치로 죽지 않는다) |
| `./run_hil_server.sh` (새 lineage) | **`remote_die`로 거부된다.** 새 lineage에서만 치명적인 것이 설계다 |

🛑 **`HIL_ACCEPT_HEAD_MISMATCH=1`로 뚫지 말 것.** 그러면 junhyeong_ai의 learner가 main 코드로
도는데 laptop3 actor는 테스트 브랜치 코드로 도는 lineage가 **영구히** 생긴다. 올바른 조치는
**§6대로 main으로 checkout하는 것**이고, 그러면 laptop3 = junhyeong_ai = `15a467e`로 맞는다.

---

## 6. 되돌리기 — **되돌릴 것이 거의 없다**

```bash
# laptop3
cd /home/laptop3/gello_software && git checkout feat/gello-ur7e-humble-22.04

# kanu (learner를 kanu에서 돌릴 때에만 의미가 있다. 안 돌릴 거면 그냥 둬도 된다)
ssh kanu 'cd /home/junhyeong/gello_software_hil_schema3_stage_20260730 \
          && git checkout feat/gello-ur7e-humble-22.04 && git pull --ff-only'
```

| 대상 | 조치 |
| --- | --- |
| laptop3 | main 브랜치로 checkout. 래퍼와 이 문서가 **함께 사라지고** junhyeong_ai 기본값이 통째로 돌아온다 |
| kanu checkout | 같은 방식으로 main으로. **급하지 않다** — 그 브랜치는 kanu에서 learner가 돌 때만 하중을 받는다 |
| kanu `~/hil-serl-data/classifier_ckpt/checkpoint_150` | **그냥 둔다.** 내용이 정확한 사본이고(SHA 검증됨) 아무도 안 읽으면 그냥 43 MB짜리 죽은 데이터다. 지우면 다음에 다시 복사해야 한다 |
| kanu `~/workspace/youngwoong/**` | **원래부터 무변경.** 되돌릴 것 없음 |
| `junhyeong_ai` | **되돌릴 것 없음** — 단 "무변경"은 아니다. checkout이 `15a467e`(main)로 **한 번 pull됐고 그대로 두는 것이 맞다.** laptop3가 main으로 돌아오면 양쪽이 `15a467e`로 일치해 §5 함정 6이 저절로 사라진다. 데이터 루트와 프로세스는 처음부터 무변경 |
| GitHub | 테스트 브랜치는 남겨 둔다(다시 필요할 때 checkout 하나로 돌아온다) |

---

## 7. 무엇이 main으로 갔나 — **ssh `ControlPath` 수정 하나** (`15a467e`)

이 시험이 찾아낸 것은 kanu 결함이 아니라 **main 브랜치 결함**이었다. 그래서 그것만 main에 올렸다.

**증상.** params-sync 폴링이 **매번** `rc=255 unix_listener: path too long`으로 실패했다.
그런데 **우아하게** 실패했다 — 경고 + 낡은 파라미터 — 그리고 프록시는 한참 뒤
`wait_for_initial_params`에서 죽으면서 **`HIL_PARAMS_EXPORT`가 없다고 탓했다.** 틀린 서사,
맞는 증상.

**원인 3개가 겹쳐 있었다.** (a) 기본 control dir가 checkout 안(66자)이었고, (b) 가드가
**템플릿** `cm-%r@%h`(75자)를 쟀는데 ssh는 `%r`/`%h`를 **조작자가 타이핑하지도 않는 값**으로
확장하며(`junhyeong_ai` 12자 → `junhyeong@166.104.146.29` 24자 → 실제 94자), (c) 임계값 100이
**ssh가 바인딩 중에 덧붙이는 접미사**(`.` + mkstemp 16자 = 17바이트)를 무시했다.

**수정.** 기본값을 `~/.ssh/hil_cm`(25자)로, 가드는 `ssh -G`로 **확장된 실제 경로**를
`107 − 17 = 90`에 대고 재고, 넘치면 경고에 그치지 않고 `/tmp/hil_cm_<uid>`(0700)로 옮기며,
거기서도 안 맞으면 ControlMaster를 아예 포기한다(폴링마다 ~0.44 s 핸드셰이크를 낸다).
**느린 폴링은 성능 문제지만 매 폴링 rc=255는 장애다 — 파라미터는 계속 흐른다.**

**두 호스트 모두에서 새 기본 경로로 실측 검증됐다**: junhyeong_ai cold 0.378 s / warm 0.015 s,
**kanu cold 0.697 s / warm 0.023 s.** 그래서 §3의 T3 명령에 **control-dir override가 없다.**

> 이것이 main에 올라간 **유일한** 변경이다. 래퍼 `run_hil_server_kanu.sh`, `run_hil_actor.sh`의
> 터널 힌트 문구(`HIL_SSH_HOST`를 쓰도록), 그리고 이 문서는 **테스트 브랜치에만** 있다.

---

## 8. 관련 문서

- [`HIL_LOCAL_INFERENCE_KO.md`](HIL_LOCAL_INFERENCE_KO.md) — **선행 조건.** 로컬 추론 모드가
  없으면 이 대체는 성립하지 않는다. 백로그·staleness·MANUAL 전용 제약의 정본
- [`DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](DATA_AND_MODELS_JUNHYEONG_AI_KO.md) — 정상 서버의 배치.
  §6의 "kanu는 이제 이 스크립트로 못 본다"가 **§2의 복사 하나로 해소된 상태**가 이 문서다
- [`HIL_SERL_KANU_RUNBOOK_KO.md`](HIL_SERL_KANU_RUNBOOK_KO.md) — 🗄️ kanu 시절 기록. **명령을 그대로
  치지 말 것.** 살아 있는 것은 게이트 의미(§1.4 / §11–13)와 §1.1의 origin 사고
- [`../CLAUDE.md`](../CLAUDE.md) — 전체 색인. ⚠️ 그 파일은 **main 기준**이라 kanu 대체를 모른다
