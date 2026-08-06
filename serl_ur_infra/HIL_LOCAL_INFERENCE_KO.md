# HIL-SERL 로컬 추론 (`HIL_POLICY_MODE=local`)

> # 🛑 실기 미검증 — **로봇에서 한 번도 돌지 않았다**
>
> 이 문서는 **설계와 계약**이지 측정 결과가 아니다. 코드와 오프라인 테스트까지만 있다.
> 아래 숫자 중 **실측인 것은 2026-08-06 feasibility study의 벤치 값뿐**이고, "그래서 루프
> 주기가 얼마가 된다"는 값은 **어디에도 없다** — 재보기 전에는 모른다.
>
> 🟢 **기본값은 하나도 안 바뀐다.** `HIL_POLICY_MODE`가 없거나 `remote`이면 3-CLI도, 프로세스
> 배치도, 생성되는 명령줄도 **예전 그대로**다. proto·pb2·`PROTOCOL_VERSION`·`SCHEMA_VERSION`·
> `remote_actor.py`는 **한 글자도 안 바뀌었다**(§7).

---

## 0. 한 줄

**서버는 학습·reward 권위·replay·classifier를 그대로 갖고, laptop3가 정책 추론만 가져온다.**
그래서 10 Hz 제어 루프에서 **블로킹 Step RPC가 빠진다.** 파라미터는 서버→laptop3로,
전이는 laptop3→서버로, **둘 다 루프 밖에서 비동기로** 흐른다.

---

## 1. 왜 — 숫자부터 (전부 2026-08-06 실측)

| 항목 | 값 | 출처 |
|---|---|---|
| 원격 `Step` RPC 1회 | **156.1 ms** | `SERVER_MIGRATION_E2E_JUNHYEONG_AI.md` (junhyeong_ai 실측) |
| 그중 네트워크 | ⚠️ **거의 없다** — 156 ms는 **서버 계산**이다. Step은 원시 픽셀 ~96 KiB를 싣지만 그건 링크에서 수십 ms다 | feasibility study |
| laptop3 로컬 추론 (`agent.sample_actions`) | **GPU p50 2.0 ms / CPU 9.6 ms** (RTX 3060 Laptop 6 GB) | feasibility study |
| JIT 워밍업 | 0.7–1.6 s, **세션당 1회** | 〃 |
| 파라미터 blob (frozen trunk 제외) | **12,383,812 B ≈ 12.4 MB** (전체 트리는 32.0 MB, 그중 ResNet-10 trunk 19.6 MB는 **laptop3에 SHA 동일 파일이 이미 있다**) | 〃 |
| learner publish 간격 | 50 learner step마다, **실측 벽시계 p50 5.77 s** | 〃 |
| laptop3↔서버 링크 | **~3.6 MB/s** → 12.4 MB에 **약 3.4 s** | 〃 |
| laptop3 여유 RAM | **~2 GB** — 새 프로세스는 가벼워야 한다 | 〃 |

**읽는 법이 하나 있다.** 156 ms가 **선이 아니라 서버 계산**이라는 사실이 이 설계의 전부다.
선을 빠르게 만드는 것으로는 아무것도 못 고친다 — 그 156 ms를 **제어 루프 밖으로 옮기는 것**만이
답이고, 옮길 수 있는 이유는 **actor가 매 스텝 서버로부터 정말로 필요로 하는 것이 액션 하나뿐**
이기 때문이다. replay 적재·reward 확정·학습은 전부 **나중에 해도 되는 일**이다.

> 🪤 **그러나 "서버가 빨라진다"는 뜻은 아니다.** 서버는 forward된 전이 하나마다 **여전히
> ~156 ms를 쓴다**(자기 추론까지 포함해서 — phase 1은 그 낭비를 받아들인다). 달라진 것은
> **누가 그 156 ms를 기다리느냐**뿐이다. 이것이 §5.1의 백로그다.

---

## 2. 구조 — 서로 분리된 세 경로

```
laptop3                                                junhyeong_ai
┌─────────────────────────────────────────┐            ┌──────────────────────────┐
│ remote_actor.py  (무수정, actor venv)     │           │ run_rlpd_learner_server  │
│   127.0.0.1:50253으로 다이얼 (local mode) │           │  (기존 + publish마다      │
│        │  기존 ActorTransport proto      │           │   params export)         │
│        ▼                                 │           │                          │
│ LOCAL POLICY PROXY (gello-local-policy)  │           │  <run_root>/params_live/ │
│  - Step 응답: 로컬 액션 (~2 ms GPU)       │           │    params_v%08d.msgpack  │
│    + 로컬 MANUAL verdict                 │           │    LATEST.json           │
│  - 원시 바이트 forward 큐 ───────────────┼──tunnel──▶│  gRPC :50053 (불변)      │
│  - param puller (ssh pull, hot-swap) ◀───┼──ssh──────│                          │
└─────────────────────────────────────────┘            └──────────────────────────┘
```

| # | 경로 | 주기 | 무엇을 하나 | 막히면 |
|---|---|---|---|---|
| 1 | **액션 (hot)** | 제어 루프 | actor → 프록시 `Step` → 로컬 추론 → 응답. **같은 wire protocol, 같은 프로세스 모양** — 프록시가 곧 ActorTransport 서버다 | 세션이 선다. 여기만 실시간이다 |
| 2 | **전이 (background)** | 큐 소진 속도 | 프록시가 `BeginEpisode`/`Step` **요청 바이트 원본**을 붙잡아 **순서대로, 동시 1개**로 진짜 서버에 재생. 서버는 아무것도 달라진 걸 모른다(같은 dedup·fingerprint) | 백로그가 자란다. **전이는 버리지 않는다**(§5.1) |
| 3 | **파라미터 (background)** | `HIL_PARAMS_POLL_S` 기본 2.0 s | learner가 publish마다 **trainable-only** blob을 쓰고, 프록시가 `LATEST.json`을 ssh로 폴링 → 새 버전만 fetch → sha 검증 → **로컬 frozen trunk를 접붙임** → 검증·스모크 → hot-swap | 정책이 낡아 간다. **skip-to-newest**라 밀린 blob을 줄줄이 적용하지 않는다 |

**셋 다 서로를 기다리지 않는다.** 2가 밀려도 1은 2 ms이고, 3이 실패해도 1은 **직전 버전으로 계속
서빙**한다(버전 번호는 단조 — fetch 중에는 옛 번호를 유지한다).

### 2.1 코드 위치

| 파일 | 역할 |
|---|---|
| `ur_env/learner/params_export.py` (+ `learner/runtime.py`의 publish 지점 훅) | 서버: trainable-only 트리를 원자적으로(tmp+rename) 쓰고 `LATEST.json` 갱신, 최근 2개만 보존. **learner ready 시점에 v0도 쓴다** — step 50 전에도 blob이 항상 하나는 있게 |
| `ur_env/local_policy/params_sync.py` | 폴링·fetch·sha 검증·trunk 접붙임·트리 모양 검증·hot-swap holder·staleness 계측 |
| `ur_env/local_policy/runtime.py` | `create_frozen_trunk_feature_agent`로 **서버와 같은 네트워크**를 만들고 `ActorSessionService`의 `sample_action` 자리에 들어간다 |
| `ur_env/local_policy/manual_finalize.py` | 서버 MANUAL 분기의 **오프라인 복제**. parity 단위 테스트가 진짜 서버 finalizer와 필드 단위로 비교한다 |
| `ur_env/local_policy/uploader.py` | FIFO 큐 + 배경 스레드 + 순서 보존 재생 + outcome 비교/divergence 로깅 + drain-on-shutdown |
| `ur_env/local_policy/proxy.py` · `scripts/run_local_policy_proxy.py` | 위 넷을 조립한 ActorTransport 서버와 entrypoint |
| `ros2_ur_ws/run_hil_local_policy.sh` | 프록시 단독 실행 래퍼 (venv 확인, RAM 경고, 터널 도달성 확인) |

📌 **인터프리터.** gRPC를 import하는 laptop3 코드는 **`/home/laptop3/venvs/gello-local-policy/bin/python`**
(jax 0.5.3 + CUDA plugin + grpcio 1.74.0 — GPU를 보는 것과 gRPC가 정상인 것 **둘 다 확인됨**)
또는 actor venv(`gello-hil-actor`, jax 없음)다. **시스템 `python3`는 절대 아니다** — 루트
`CLAUDE.md`의 grpcio 1.30.2 무한 정지 항목.

---

## 3. 조작자 CLI

```bash
# ── 원격 추론(기존, 기본값) — 아무것도 안 바뀜 ──
./run_hil_server.sh; ./run_hil_hardware.sh; ./run_hil_session.sh

# ── 로컬 추론(신규, opt-in) ──
HIL_PARAMS_EXPORT=1 HIL_EXTERNAL_POLICY_INGEST=1 [HIL_LATENCY_PROFILE=1] \
  ./run_hil_server.sh                                             # T1
./run_hil_hardware.sh                                             # T2
HIL_POLICY_MODE=local [HIL_LATENCY_PROFILE=1] ./run_hil_session.sh  # T3
#   └ 세션 스크립트가 로컬 프록시를 자동 기동·감시 (단독 실행: ./run_hil_local_policy.sh)
```

🛑 **T1의 환경변수는 둘 다 필수다. 하나만 켜면 조용히 반쪽으로 돈다.**
`HIL_PARAMS_EXPORT`는 파라미터를 **내보내는** 쪽이고, `HIL_EXTERNAL_POLICY_INGEST`는 그
파라미터로 만든 액션이 실린 전이를 **받아 주는** 쪽이다. 로컬 추론은 둘 다 있어야 성립한다.

터미널 수는 **여전히 3개**다. 프록시는 카메라·GUI처럼 `run_hil_session.sh`가 소유한다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `HIL_POLICY_MODE` | `remote` | `local`이면 actor가 `127.0.0.1:50253`을 다이얼하고 세션이 프록시를 띄운다 |
| `HIL_LOCAL_POLICY_PORT` | `50253` | 프록시 포트. 터널의 로컬 끝 `50153`(진짜 서버)과 **다른 번호다 — 헷갈리지 말 것** |
| `HIL_PARAMS_EXPORT` | (없음) | 서버 opt-in. `HIL_LATENCY_PROFILE`과 **똑같은 방식**으로 `run_hil_server.sh`가 넘긴다 |
| `HIL_EXTERNAL_POLICY_INGEST` | (없음) | 서버 opt-in. learner가 **프록시가 발행한** `meta.policy_action`/`policy_version`을 받아들이게 한다. **없으면 forward된 전이가 전부 거절된다** — 아래 🛑 |
| `HIL_PARAMS_POLL_S` | `2.0` | `LATEST.json` 폴링 주기 |
| `HIL_LATENCY_PROFILE` | (없음) | §6. 프록시/uploader/paramsync **세 role 전부**가 이 하나를 본다 |

> 🪤 **`HIL_PARAMS_EXPORT`도 재사용된 learner에는 안 먹는다.** `run_hil_server.sh`는 살아 있는
> learner를 재사용하고(`HIL_SERVER_RESULT=reused`), 환경변수는 **프로세스 기동 시점에만** 전달된다.
> `params_live/`가 안 생기면 T1 출력의 `HIL_SERVER_RESULT`부터 본다. `HIL_LATENCY_PROFILE`이
> 같은 이유로 겪는 함정과 **정확히 같다**(`docs/testing/10_LATENCY_PROFILING.md` 함정 1).

> # 🛑 `HIL_EXTERNAL_POLICY_INGEST` — 이 설계에서 **가장 조용한 실패**
>
> 기본 learner는 **chain of custody**를 강제한다: `meta.policy_action`/`policy_version`이
> **그 learner가 직전 관측에 대해 발행한 값과 같아야** 한다. 로컬 모드에서 그 액션을 발행한
> 것은 프록시이고, 서버가 같은 관측에 자기 추론을 돌려도 **같은 값이 나올 수 없다.**
> 그래서 플래그가 없으면 **온라인 전이가 하나도 안 들어가고 replay가 빈 채로 남는다.**
>
> **왜 조용한가.** `HIL_PARAMS_EXPORT`를 빼먹으면 프록시가 health-ready가 안 되어 세션이
> 아예 시작되지 않는다 — 즉시 안다. 이쪽은 정반대다. **프록시가 로컬에서 답하므로 actor는
> 완벽히 정상으로 돈다.** 팔이 움직이고, 개입이 먹고, GUI가 살아 있고, episode가 끝난다.
> 서버만 조용히 전부 거절하고 있다.
>
> **어디서 보나.** 프록시 로그의 `TRANSITION NOT INGESTED (#N): server refused ...` 줄
> (`ur_env/local_policy/uploader.py:_abandon`). `run_hil_session.sh`가 띄운 경우 그 로그는
> **조작자 터미널이 아니라** `/tmp/hil_session_*/local_policy.log`에 있다. 세션 시작 직후 한 번:
>
> ```bash
> grep -c 'TRANSITION NOT INGESTED' /tmp/hil_session_*/local_policy.log   # 0 이어야 한다
> ```
>
> ⚠️ **살아 있는 learner를 재사용하면 이 플래그를 켤 수 없다** — 위 🪤와 같은 이유다.
> 로컬 모드로 처음 도는 세션은 **learner를 새로 띄워야 한다**(`HIL_SERVER_RESULT=started`
> 확인). 재사용된 STRICT learner 밑에서 로컬 모드를 돌리면 위의 조용한 실패가 바로 그것이다.
> learner가 어느 쪽으로 떴는지는 그 run의 JSONL에 `accept_external_policy_meta`로 남는다
> (`rlpd_learner_external_policy_ingest` 이벤트 + run metadata).

---

## 4. MANUAL 전용 — AUTO는 fail-closed

**프록시는 classifier를 돌리지 않는다.** classifier 체크포인트·GPU·sidecar 디코드는 전부 서버에
있고, 그것이 이 설계가 유지하려는 경계다(reward 권위 = 서버).

그래서 phase 1의 프록시는 **서버 `_finalize_transition`의 MANUAL 분기만** 복제한다 —
success는 `operator_success` 토큰, done/truncated/masks는 같은 동일성 규칙, classifier 필드는
**not-evaluated로 표시**. 이 복제가 맞다는 것은 **진짜 서버 finalizer와 같은 입력을 넣어 필드를
비교하는 parity 단위 테스트**가 지킨다. 그게 이 phase의 **가장 load-bearing한 테스트**다.

🛑 **AUTO에서는 조용히 성공을 못 만든다 — 프로토콜 오류로 거절한다.** 전이에
`meta.auto_success == true`가 실려 오면 프록시는 응답 대신 오류를 내고 **actor가 fault한다.**
분류기를 못 돌리는 프로세스가 AUTO 세션에서 "성공 아님"을 계속 답하면, 그건 **성공 권한을
말없이 축소하는 것**이고 조작자는 영영 알 수 없다. 요란하게 죽는 쪽이 옳다.

📌 현재 운용 기본이 **MANUAL**이라는 것(루트 `CLAUDE.md`) 덕분에 이 제약이 실무를 막지 않는다.
AUTO를 되살리려면 이 문서가 아니라 **classifier 재학습**이 선행이다.

---

## 5. 정직한 한계 — 이 문단을 빼고 인용하지 말 것

### 5.1 서버는 전이당 여전히 ~156 ms를 쓴다 → **에피소드 중 백로그가 자란다**

forward된 `Step`은 서버에서 **평소와 똑같이** 처리된다. 자기 정책 추론까지 그대로 돈다(그 액션은
**버려진다** — phase 1이 받아들인 낭비다. 없애려면 proto에 submit-only RPC가 필요하고, 그건 §8).

즉 **생산은 루프 속도(최대 10 Hz), 소비는 ~6.4 Hz**다. 에피소드가 도는 동안 큐는 **자라고**,
`WAIT_HOME_APPROVAL` / `WAIT_SCENE_READY` 같은 조작자 대기 화면에서 **빠진다.** 이 설계는
**"에피소드 사이에 사람이 장면을 재배치하는 시간"이 존재한다는 사실에 기대고 있다.**

- 🟢 **전이는 절대 버리지 않는다.** 큐는 FIFO이고 RAM에서 unbounded이며, 에피소드 경계를 넘어
  보존되고, 프록시가 정상 종료를 보고하기 전에 **비운다**.
- ⚠️ **unbounded의 대가는 RAM이다.** laptop3 여유 RAM이 ~2 GB고 전이 하나가 원시 픽셀 ~96 KiB다.
  500건에서 큰 경고를 찍는 이유가 그것이고, **그 경고가 유일한 방어선이다.**
- 📌 **에피소드 사이에 큐가 ~0으로 안 돌아오면 이 설계의 전제가 깨진 것이다.** 분석기 보고서
  **7절**의 `backlog_depth` / `oldest_backlog_s`가 정확히 그것을 본다.

### 5.2 파라미터는 **한 전송분(≈3–9 s)** 뒤처진다

`staleness_s`는 "적용된 버전의 `LATEST.json` 시각으로부터 흐른 시간"이다. **0이 될 수 없다** —
12.4 MB가 3.6 MB/s 링크를 건너야 하고(≈3.4 s), 그 위에 폴링 주기(2 s)와 learner의 publish
간격(p50 5.77 s)이 얹힌다. 즉 **actor는 항상 조금 낡은 정책으로 행동한다.**

이것이 학습에 주는 영향은 **정책 지연(policy lag)**이고, off-policy RL에서 흔한 조건이지만
**공짜는 아니다.** 지금 상태에서 우리가 말할 수 있는 것은 "얼마나 낡았는지 잰다"까지다 —
`params_age_s`(프록시가 응답할 때) / `staleness_s`(sync가 적용할 때) 두 각도로 본다.
⚠️ **원격 모드에는 이 지연이 없었다**(서버가 자기 최신 파라미터로 추론했다). 로컬 모드가
**새로 만든 비용**이므로, 루프가 빨라진 이득과 **같이** 저울에 올려야 한다.

### 5.3 divergence 로깅 = parity 알람

프록시는 자기 MANUAL verdict로 actor에 답하고, uploader는 **서버가 돌려준 outcome과 그것을
비교**해 다르면 `outcome_divergence`를 남긴다. 이건 지표가 아니라 **알람**이다:

- **0이 정상이다.** MANUAL 분기는 결정론적이고 두 구현이 같은 입력을 본다.
- **1 이상이면 actor가 본 에피소드와 learner가 학습한 에피소드가 다르다.** 데이터가 오염된 건
  아니다(서버 쪽이 권위이고 그게 학습에 쓰인다) — 하지만 **로봇 위에서 벌어진 일과 버퍼에 적힌
  일이 갈라졌다**는 뜻이고, 그 간극이 바로 이 알람의 존재 이유다.
- 분석기가 `!!! OUTCOME DIVERGENCE` 배너로 크게 찍는다(§6).

### 5.4 프록시가 **못 하는** 것

| 못 하는 것 | 결과 |
|---|---|
| classifier 추론 | reward는 **서버가** 매긴다. 프록시 응답의 classifier 필드는 not-evaluated다 |
| AUTO 성공 판정 | fail-closed로 거절(§4) |
| replay 저장 | 서버만 저장한다. 프록시는 아무것도 영속하지 않는다 |
| 학습 | 없다. 파라미터는 **읽기만** 한다 |

### 5.5 재시도는 안전하지만 **바이트 동일할 때만**

uploader는 붙잡은 **원본 바이트**를 그대로 재생한다 — 같은 request id, 같은 fingerprint →
서버의 dedup이 재시도를 흡수한다. 🛑 **요청을 재조립하지 말 것.** 필드를 하나라도 다시 만들면
dedup이 안 잡고 **같은 전이가 두 번 학습된다.**

### 5.6 실기에서 아직 아무것도 확인되지 않았다

|  | 상태 |
|---|---|
| 오프라인 단위/통합 테스트 | 있음 |
| 벤치(추론·링크·blob 크기) | 실측 (§1) |
| **실기 세션 1회** | 🛑 **없음** |
| 루프 주기가 실제로 얼마가 되는가 | 🛑 **미측정** — 제일 궁금한 숫자가 아직 없다 |
| 장시간 백로그 거동 | 🛑 미측정 |
| divergence 0 여부(실기) | 🛑 미측정 |

---

## 6. 계측 — role 3개가 늘어난다

`HIL_LATENCY_PROFILE=1`이면 프록시·uploader·paramsync가 **각자** JSONL을 쓴다. 위치는 actor와
같은 `ros2_ur_ws/gello_logs/hil_latency/`이고 파일명에 role이 들어간다
(`<UTC>_<role>_<pid>.jsonl`).

| role | 필드 |
|---|---|
| `proxy` | `local_inference_ms` · `local_finalize_ms` · `total_ms` · `queue_depth` · `params_version` · `params_age_s` |
| `uploader` | `upload_rpc_ms` · `backlog_depth` · `oldest_backlog_s` · `outcome_divergence` |
| `paramsync` | `poll_ms` · `fetch_ms` · `load_ms` · `swap_ms` · `version` · `staleness_s` |

```bash
PY=/home/laptop3/venvs/gello-hil-actor/bin/python
$PY serl_ur_infra/scripts/analyze_hil_latency.py \
  --actor     ros2_ur_ws/gello_logs/hil_latency/<utc>_actor_<pid>.jsonl \
  --proxy     ros2_ur_ws/gello_logs/hil_latency/<utc>_proxy_<pid>.jsonl \
  --uploader  ros2_ur_ws/gello_logs/hil_latency/<utc>_uploader_<pid>.jsonl \
  --paramsync ros2_ur_ws/gello_logs/hil_latency/<utc>_paramsync_<pid>.jsonl
```

절 **6·7·8**이 붙는다: 프록시 phase + `params_age_s` 분포 + `params_version` 진행,
uploader 백로그 요약 + **divergence 배너**, paramsync fetch/load/swap + staleness 분포.
셋 다 **없으면 이유를 적고 강등**되며 exit code는 0이다. 절차 전문은
[`../docs/testing/10_LATENCY_PROFILING.md`](../docs/testing/10_LATENCY_PROFILING.md) §1.4·§4.

> 🛑 **로컬 모드에서 actor의 `step_rpc_ms`는 서버까지가 아니라 `127.0.0.1`까지다.**
> 원격 모드 파일의 같은 필드와 나란히 놓고 "서버가 빨라졌다"고 읽으면 틀린다 — 서버의 156 ms는
> **없어진 게 아니라 uploader로 옮겨 갔다.** 분석기가 `--proxy`가 있을 때 보고서 **3절
> (LOOP BUDGET)** 에 이 경고를 자동으로 찍는다(`step_rpc_ms measures the LOCAL hop`).

---

## 7. 바꾸지 않은 것 (이걸 바꿨으면 설계가 실패한 것이다)

| 불변 | 왜 |
|---|---|
| `proto/` · pb2 · `PROTOCOL_VERSION` · `SCHEMA_VERSION` · 기존 메시지 | **이 설계 전체가 그걸 피하려고 존재한다.** protobuf가 unknown field를 조용히 버리므로 반쪽 업그레이드는 무증상 오염이다 |
| `remote_actor.py` | **편집 0줄.** 모드 전환은 shell 래퍼가 기존 CLI 플래그/env로 한다 |
| 원격 모드 동작 | `HIL_POLICY_MODE` 미설정/`remote`에서 **생성되는 명령이 바이트 동일**해야 한다 |
| learner 기동 argv | `run_hil_server.sh --check`의 `validate_process_contract`가 argv를 문자열 동등성으로 비교한다 → 새 opt-in은 **전부 환경변수** |
| reward 권위 | 서버. 프록시는 MANUAL 복제일 뿐이고 classifier를 대신하지 않는다 |
| 전이 손실 | **0.** 큐는 FIFO·unbounded·drain-on-shutdown |

---

## 8. phase 2 아이디어 (아직 아무것도 결정되지 않았다)

1. **submit-only RPC.** 서버가 forward된 전이에 대해 **자기 추론을 돌리지 않게** 하면 전이당
   비용에서 policy inference가 빠져 소비 속도가 올라간다. **proto 신규 RPC + `PROTOCOL_VERSION`
   bump**가 필요하고 **양끝을 같이 올려야 한다** — §7의 이유로 이건 가볍게 할 수 있는 일이 아니다.
   먼저 **§5.1의 백로그를 실기에서 재고**, 그 숫자로 필요한지 정한다.
2. **전이 배치 전송.** 여러 전이를 한 RPC로 묶으면 왕복 수가 준다. 서버 핸들러 비용이 지배적
   이므로 이득은 위 1보다 작다.
3. **파라미터 델타/압축.** 12.4 MB 중 실제로 바뀌는 양은 훨씬 적을 수 있다. staleness가 문제로
   측정될 때만 의미가 있다.
4. **로컬 classifier.** AUTO를 로컬에서 되살리는 유일한 길이지만, **reward 권위를 laptop3로
   옮기는 것**이라 설계 결정이 필요하다. 지금은 명시적으로 범위 밖이다.

**넷 다 공통 선행 조건은 §5.6의 첫 실기 세션이다.** 측정 없이 손대지 않는다.

---

## 9. 상태

| 항목 | 상태 |
|---|---|
| 서버 params export · sync 클라이언트 · 로컬 런타임 · MANUAL finalizer · uploader | 코드 통합 |
| 프록시 조립 · entrypoint · 3-CLI 배선 | 코드 통합 (`proxy.py` · `run_local_policy_proxy.py` · `run_hil_local_policy.sh` + `HIL_POLICY_MODE`) |
| learner 측 외부 정책 meta 수용 (`HIL_EXTERNAL_POLICY_INGEST`) | 코드 통합 — **§3의 🛑을 읽을 것** |
| A→B→C→E→D→learner 한 프로세스 통합 스모크 | 있음 (2026-08-06 검수) |
| 분석기 절 6·7·8 + 오프라인 테스트 | 있음 |
| **실기 세션** | 🛑 **없다 — 미검증** |
| 루프 주기 개선폭 | 🛑 **미측정.** §1은 부품 벤치이지 루프 측정이 아니다 |

📌 **여기에 실측치를 적지 말 것.** 세션 결과는 날짜와 함께
[`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md)에 남긴다.
