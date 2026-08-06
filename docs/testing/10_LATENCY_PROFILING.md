# 10 — per-step 레이턴시 계측 (`HIL_LATENCY_PROFILE`)

> ## 이 문서가 존재하는 이유
>
> `serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md` **§8 P1**이 요구하는 계측이
> 이것이다. 지금까지 우리가 가진 숫자는 두 종류뿐이었다 —
> `_RoundTripStats`가 종료할 때 한 번 찍는 **Step 왕복의 count/mean/max**, 그리고 learner가
> 자기 스레드에 대해 남기는 `timing/*`. 둘 다 **per-step 표본이 디스크에 남지 않으므로
> percentile도 tail도 phase 귀속도 사후에 계산할 수 없다.**
>
> 그래서 **실측 1.95 Hz(주기 512 ms, 최대 854 ms)** 의 512 ms가 어디로 갔는지 — `env.step`인지
> Step RPC인지 카메라 디코드인지 learner 경합인지 — 아무도 모른다. 이 문서의 절차는 그 512 ms를
> phase 별 p50/p90/p99/max로 쪼갠다.
>
> 🛑 **이건 기본값이 아니다.** `HIL_LATENCY_PROFILE`을 주지 않으면 코드 경로는 **비트 단위로
> 동일**하고 파일도 생기지 않는다. 계측하려고 마음먹은 세션에서만 켠다.

---

## 1. 무엇을 재는가

두 호스트가 **각자 자기 파일에** 쓴다. 프로토콜은 한 글자도 안 바뀐다 — proto도 pb2도
`SCHEMA_VERSION`도 그대로다(⚠️ 이유는 §6).

### 1.1 actor 쪽 (laptop3) — 루프 1회당 1줄

| 필드 | 뜻 |
|---|---|
| `iter_interval_ms` | **루프 주기.** 연속한 iteration 시작 사이. §8 P1이 말하는 "실측 `env.step` 간격"이 이것이다. **다른 phase의 합이 아니라 그것들을 담는 그릇이다** |
| `env_step_ms` | `env.step()` 자체. ⚠️ **env의 100 ms 자체 페이싱을 포함하므로 바닥값이 있다** |
| `transition_build_ms` | `build_data` + `validate` |
| `sidecar_encode_ms` | 분류기 sidecar JPEG 인코드. **붙은 스텝에만 있다**(약 2 Hz) |
| `step_rpc_ms` | 블로킹 Step RPC 왕복. **`env.step` 밖이다** — 이 격차가 P1의 본체 |
| `server_inference_ms` | 📌 **서버가 잰 값을 응답에서 그대로 베낀 것.** actor가 잰 게 아니다 |
| ids | `run_id` · `episode_id` · `env_step` · `step_id` · `transition_id` · `request_id` |
| flags | `intervened` · `sidecar_attached` |
| 개입 | `intervention_saturation` · `intervention_saturated` · `intervention_follow_ticks` — `info[]`에서 **베껴** 온다(재계산하지 않는다) |

### 1.2 server 쪽 (junhyeong_ai) — Step RPC 1회당 1줄

| 필드 | 뜻 |
|---|---|
| `total_ms` | 핸들러 진입→탈출. **아래 phase들을 담는다** |
| `request_decode_ms` | proto → 배열/관측 |
| `service_step_ms` | 세션 서비스 본체(아래 phase들을 담는다) |
| `trunk_encode_ms` | frozen-trunk feature 인코드 |
| `classifier_ms` | sidecar 게이트 + 분류기 추론. 🔴 **분류기가 죽어 있으면 여기는 사실상 0이고 그 전이는 미채점으로 강등된다** — 루트 `CLAUDE.md`의 🟠 박스 |
| `reward_finalize_ms` | reward 확정 |
| `replay_insert_ms` | replay store insert |
| `response_build_ms` | 응답 조립 |
| `concurrent_rpcs` | **게이지.** 핸들러 진입 시점의 동시 Step 수(`--max-workers 4`). 진짜 queue 대기는 공통 시계 없이 못 재므로 이게 대용물이다 |
| ids | `transition_id` · `request_id` (+ 있으면 `run_id`/`episode_id`) |

`BeginEpisode`는 `rpc: "begin_episode"` 태그가 붙은 별도 줄로 `total_ms`만 남는다.

### 1.3 learner 쪽 — **새로 만들지 않는다**

learner는 이미 `logs/learner.jsonl`에 `learner_update`마다
`timing/sample_ms` · `timing/critic_update_ms` · `timing/full_update_ms` ·
`timing/learner_step_ms`를 남긴다. 분석기는 **그 파일을 그대로 읽는다.** 중복 계측 없음.

---

## 2. 켜는 법 — 3-CLI 워크플로우

평소와 같은 세 터미널이고, **환경변수 하나만 앞에 붙인다.**

```bash
cd /home/laptop3/gello_software/ros2_ur_ws

# Terminal 1 — 서버 learner + tunnel
HIL_LATENCY_PROFILE=1 ./run_hil_server.sh

# Terminal 2 — 하드웨어 (계측 대상이 아니다. 평소 그대로)
./run_hil_hardware.sh

# Terminal 3 — cameras + GUI + actor
HIL_LATENCY_PROFILE=1 ./run_hil_session.sh
```

> ## 🪤 **함정 1 — 재사용된 learner는 서버 쪽을 안 남긴다. 이게 제일 흔한 실패다.**
>
> `run_hil_server.sh`는 **살아 있는 learner를 재사용한다**(`HIL_SERVER_RESULT=reused`).
> 재사용된 프로세스는 **자기가 기동될 때의 환경변수**를 갖고 있으므로, 그 learner가
> `HIL_LATENCY_PROFILE` 없이 떴다면 **Terminal 1 앞에 몇 번을 붙이든 서버 파일은 생기지
> 않는다.** 환경변수는 프로세스 기동 시점에만 전달된다.
>
> **확인:** Terminal 1 출력의 `HIL_SERVER_RESULT`를 본다.
> - `started` → 서버 쪽도 남는다.
> - `reused` → **서버 쪽은 안 남는다.** actor 쪽만으로 §4의 1·3·5절은 나오고, 2절
>   (network+queue)과 4절(경합)은 "server data absent"로 강등된다.
>
> 서버 쪽까지 필요하면 learner **프로세스를 죽이고** 새로 띄워야 한다. 🛑 그 순간
> **그 프로세스 RAM에만 있던 online replay는 사라진다**(→ `09` §5.4). 계측 하나를 위해
> lineage를 버릴지는 **조작자가 판단한다** — 이 문서는 시키지 않는다.

> ## 🪤 함정 2 — T1과 T3는 서로 독립이다
>
> 두 곳의 `HIL_LATENCY_PROFILE`은 **다른 프로세스의 다른 스위치**다. T3에만 붙이면
> actor 파일만, T1에만 붙이면 server 파일만 생긴다. **join(§4 2절)은 둘 다 있어야 한다.**

### 2.1 출력 디렉터리를 옮기고 싶으면

`HIL_LATENCY_PROFILE_DIR`은 그 변수가 보이는 **프로세스의** 기본 경로를 덮는다(파일명은
자동 생성: `<UTC YYYYmmdd_HHMMSS>_<role>_<pid>.jsonl`).

🪤 **실질적으로 actor 전용이다.** `run_hil_server.sh`가 learner 호스트로 넘기는 것은
`HIL_LATENCY_PROFILE` **하나뿐**이므로, T1 앞에 `HIL_LATENCY_PROFILE_DIR`을 붙여도
서버 파일은 **여전히 `<run_root>/logs/latency_server.jsonl`에 떨어진다.** 서버 경로를
옮기려면 learner 프로세스 자신의 환경에 그 변수가 있어야 한다(= 런처 수정).

---

## 3. 파일이 어디에 떨어지나

| 쪽 | 경로 |
|---|---|
| **actor** (laptop3) | `/home/laptop3/gello_software/ros2_ur_ws/gello_logs/hil_latency/<UTC>_actor_<pid>.jsonl` |
| **server** (junhyeong_ai) | `<run_root>/logs/latency_server.jsonl` |
| **learner** (junhyeong_ai, 원래부터 있던 것) | `<run_root>/logs/learner.jsonl` |

`<run_root>`는 `./run_hil_server.sh --check`가 알려 준다. 📌 **run root는 스냅샷이므로
문서에 적힌 값을 복사하지 말고 매번 읽는다.**

분석은 laptop3에서 하므로 서버 쪽 두 파일을 먼저 끌어온다:

```bash
RUN=<--check 가 알려 준 run root>
scp junhyeong_ai:"$RUN/logs/latency_server.jsonl" /tmp/
scp junhyeong_ai:"$RUN/logs/learner.jsonl"        /tmp/
```

---

## 4. 분석기 — `analyze_hil_latency.py`

```bash
cd /home/laptop3/gello_software
PY=/home/laptop3/venvs/gello-hil-actor/bin/python

# (a) actor 파일만 — 1·3·5절이 나온다
$PY serl_ur_infra/scripts/analyze_hil_latency.py \
  --actor ros2_ur_ws/gello_logs/hil_latency/20260806_101500_actor_12345.jsonl

# (b) 전체 — 5개 절 전부 + markdown 저장
$PY serl_ur_infra/scripts/analyze_hil_latency.py \
  --actor   ros2_ur_ws/gello_logs/hil_latency/20260806_101500_actor_12345.jsonl \
  --server  /tmp/latency_server.jsonl \
  --learner /tmp/learner.jsonl \
  --out     /tmp/hil_latency_report.md
```

`--actor`만 필수다. `--server`/`--learner`는 없으면 **해당 절이 이유를 적고 강등**되며
**exit code는 0이다.** 없는 파일을 가리키면 stderr에 `warning:`이 한 줄 뜨고 나머지는 그대로
나온다. `--actor`가 없는 파일이면 그때만 **exit 2 + 한 줄 오류**다(traceback 없음).

### 4.1 보고서 5개 절

| 절 | 내용 | 필요한 파일 |
|---|---|---|
| **0. INPUTS** | 파일별 레코드 수, 깨진 줄 수, 잘린 마지막 줄, run_id/episode | actor |
| **1. PER-PHASE LATENCY** | actor phase 표 + server Step phase 표 + BeginEpisode 표 + `concurrent_rpcs` 게이지. 각각 count/mean/p50/p90/p99/max | actor (+server) |
| **2. CROSS-HOST JOIN** | `transition_id` inner join → **`network_plus_queue_ms = step_rpc_ms − total_ms`** 분포 + **join coverage**(matched / actor / server) | actor **+ server** |
| **3. LOOP BUDGET** | `iter_interval_ms` 분포 vs 10 Hz(100 ms) + **100/200/500 ms 초과 비율** + 환산 Hz | actor |
| **4. LEARNER CONTENTION** | server의 `t_epoch`로 Step RPC와 `learner_update` 창을 겹쳐 **"learner 갱신 중 vs 아님"으로 RPC 통계를 가른다.** learner 자신의 `timing/*` 표. **`utd_ratio`는 `learner.jsonl`에서 읽는다** | server + learner (learner만 있어도 `timing/*` 표는 나온다) |
| **5. INTERVENTION** | 개입 스텝 수, **포화 비율**, `intervention_saturation` 분포, sidecar 유무별 `step_rpc_ms` | actor |

**5절이 P1의 acceptance 항목 중 "포화 비율"을 그대로 답한다** — 창이 길어지면 개입 액션이
포화되고 **저장된 액션이 실제 이동을 과소진술한다**(`08_OPEN_GAPS.md` G33). 이건 조작자가
알아챌 수 없는 종류의 오염이라 계측이 유일한 탐지 수단이다.

### 4.2 phase 목록은 하드코딩이 아니다

분석기는 **`_ms`로 끝나는 키를 전부 phase로 취급**한다(단 `mark()`가 쓰는 `*_at_ms` 오프셋은
제외한다). 나중에 wiring 커밋이 phase를 하나 추가해도 **이 스크립트를 고치지 않아도 표에
나타난다.** 알려진 이름은 읽기 좋은 순서로 앞에 오고 나머지는 알파벳순으로 뒤에 붙는다.

---

## 5. 🛑 반드시 지킬 것 — 시계 규칙

> **두 호스트의 타임스탬프는 절대로 서로 빼지 않는다. 어디에서도.**

laptop3와 junhyeong_ai는 시계가 동기화돼 있지 않고 그 오차는 **bounded가 아니다.**

- 모든 `*_ms`는 **그 값을 쓴 프로세스 안에서만** 잰 `time.perf_counter()` 차이다.
- `t_epoch`(`time.time()`)는 **같은 호스트 안의 순서·상관용**이다. 4절의 겹침 판정은
  **server의 `t_epoch` ↔ server 호스트의 `learner.jsonl`** 로만 한다 — 같은 기계, 같은 시계.
- **network+queue는 재는 값이 아니라 유도하는 값이다:**

  ```
  network_plus_queue_ms = step_rpc_ms (actor가 잰 왕복) − total_ms (server가 잰 핸들러)
  ```

  **두 duration의 뺄셈이므로 공통 epoch이 필요 없다.** 이 값은 **선(wire) + SSH 터널 +
  양쪽 gRPC 직렬화 + 핸들러 큐 대기**를 전부 뭉뚱그린 **잔차**다. 특정 홉의 측정값이 **아니다.**

⚠️ 분석기가 이 값이 **음수**라고 보고하면 시계 오차가 아니다(애초에 시계를 안 썼다).
서버의 `total_ms`가 actor의 Step 호출보다 더 넓은 구간을 재고 있다는 뜻이고 — 즉 **응답을
gRPC에 넘긴 뒤에도 핸들러가 일을 더 하고 있다** — 그건 실제로 볼 가치가 있는 사실이다.

---

## 6. 왜 응답에 실어 보내지 않았나 (다시 시도하지 말 것)

"서버 타이밍을 Step 응답에 담아 보내면 join이 필요 없다"는 뻔한 설계는 **proto 변경**을
요구한다. 그리고 **protobuf는 모르는 필드를 조용히 버린다** — 새 actor ↔ 재사용된 옛 learner
(또는 그 반대)의 반쪽 업그레이드가 **에러 없이 그럴듯한 숫자**를 만들어 낸다. `08_OPEN_GAPS.md`
**G33**과 같은 실패 모드다.

그래서 **양쪽이 각자 로컬 JSONL을 쓰고 이미 존재하는 `transition_id`로 오프라인 join**한다.
proto 0줄, `SCHEMA_VERSION` 불변, 핸드셰이크 영향 0.

같은 이유로 서버 쪽 opt-in은 **CLI 플래그가 아니라 환경변수**다 — `run_hil_server.sh --check`의
`validate_process_contract`가 **argv를 문자열 동등성으로 비교**하므로 플래그를 하나라도 추가하면
기존 learner 재사용 검사가 깨진다.

---

## 7. 오버헤드

| 상태 | 비용 |
|---|---|
| `HIL_LATENCY_PROFILE` 미설정/`0` | **호출당 1 µs 미만.** 속성 읽기 1회 + 분기 1회. `perf_counter` 호출 없음, 할당 없음, **파일 생성 없음**, `mkdir` 없음 |
| `=1` | **스텝당 수십 µs 수준.** `perf_counter` 몇 번 + 작은 dict + `json.dumps` 한 줄. 쓰기는 버퍼링(50건 또는 5초) |

512 ms 주기에서 수십 µs는 **0.01 % 미만**이다. 그래도 기본값이 off인 이유는 성능이 아니라
**계약**이다 — 켜지 않은 세션의 제어 흐름이 비트 단위로 동일하다는 것 자체가 안전 속성이다.

🟢 **로그 sink가 죽어도 세션은 죽지 않는다.** 디스크가 차거나 경로가 읽기 전용이면
`LatencyProfiler`는 **경고를 한 번 찍고 스스로 disabled로 강등**한다(커밋 `c86dc54`가
learner metrics sink에 세운 규칙과 같다: 죽은 sink는 logger를 강등시키지 학습을 끝내지 않는다).
생성자는 **파일시스템을 건드리지 않는다** — 디렉터리 생성과 open은 첫 commit 때 lazy로 일어나므로
경로 오타가 세션 기동을 막지 못한다.

---

## 8. 이 파일들을 읽을 때 알아야 할 것

- **한 줄 = 한 JSON 객체.** 모든 줄에 `schema` · `role` · `t_epoch` · `seq`가 있다.
- **마지막 줄이 잘려 있는 것은 정상이다.** SIGKILL 당한 세션의 버퍼 잔여분이다.
  분석기는 그것을 세어서 보고하고 무시한다. 중간 줄이 깨진 것은 **다른 사실**이라 따로 센다.
- **같은 phase 이름이 한 레코드에 두 번 나오면 합산된다**(마지막 값이 아니라 총합).
  재시도 루프나 카메라 2대 디코드가 그 phase의 총 시간을 보고하도록 한 의도적 설계다.
- `seq`는 프로파일러별 단조 증가라 **한 파일 안의 유실 여부**를 볼 수 있다.

---

## 9. 상태

| 항목 | 상태 |
|---|---|
| 모듈 · wiring · 분석기 · 오프라인 단위 테스트 | **코드 통합** |
| **실기 세션에서 한 번이라도 켜고 돌린 적** | 🛑 **없다 — 미검증** |
| P1 acceptance(실기 루프 주기 · 포화 비율 · phase 귀속) | 🛑 **미측정.** 이 문서는 **재는 방법**이지 측정 결과가 아니다 |

📌 **여기에 실측치를 적지 말 것.** 세션 결과는 `README.md` §1 상태표와
`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md` §8 P1에 날짜와 함께 남긴다.
