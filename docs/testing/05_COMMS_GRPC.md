# 05 — 통신 (gRPC actor · 스키마 · 레이턴시 · Kanu 터널)

**상태: 2026-07-27 실기 세션에서 크게 진전.**

| 항목 | 상태 |
|---|---|
| gRPC 트랜스포트 단위/루프백 | **PASS** — 35 tests (venv python) |
| 스키마 해시 랩톱↔Kanu 일치 | **PASS** (2026-07-27 왕복에서 확인) |
| Kanu 왕복 (Stage A, fake-env, 100 스텝) | 📌 **PASS (2026-07-27)** — `replay_insert_count: 100` |
| 레이턴시 예산 | 📌 **측정 완료 (2026-07-27) — 예산이 사실상 소진 상태** (§5.3) |
| 시스템 python3의 grpcio | 🛑 **손상. 절대 쓰지 말 것** (§1). 📌 2026-07-29 재확인: 여전히 `1.30.2` |
| 실기 센서를 붙인 Stage B | **미검증** → `09_HIL_ACTOR_RUNBOOK.md` |
| Kanu 서버 가동 여부 | **지금은 안 떠 있다** — port 50053 미바인딩, GPU 유휴 (2026-07-29). 아래 절차는 서버를 **새로 띄우는 것부터** 시작한다 |

정본: [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md)(기동 절차·2026-07-27 실측),
`serl_ur_infra/REMOTE_ACTOR_GRPC.md`, `serl_ur_infra/RL_RECEIVE_SERVER.md`,
`serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md`.
이 문서는 **계약과 판정 기준**을, 09는 **한 줄 기동 절차**를 담는다.

```bash
export WT=/home/laptop3/gello_software     # 2026-07-29 머지(3f199d4) 이후 통합 checkout이 정본
export ACTOR_PY=/home/laptop3/venvs/gello-hil-actor/bin/python   # grpcio 1.74.0
```

---

## 1. 🛑 인터프리터 — 이게 1번인 이유 (2026-07-27 갱신)

### 1.0 시스템 `python3`로 gRPC를 부르면 **영구 정지한다**

apt의 `python3-grpcio 1.30.2`가 이 머신에서 손상돼 있다. 채널을 하나 만들기만 해도
**에러도 로그도 없이 CPU 100%로 무한 스핀**한다. 30초 뒤에도, 30분 뒤에도 안 돌아온다.

```bash
python3 -c "import grpc; print(grpc.__version__)"   # -> 1.30.2   ← 이 인터프리터는 금지
$ACTOR_PY -c "import grpc; print(grpc.__version__)" # -> 1.74.0   ← 이것만 쓴다
```

재현·관찰 방법과 "왜 실운용 Kanu 왕복은 멀쩡했는가"는 `00_SETUP_AND_SAFETY.md` §3.4에 있다.

**이것이 실제로 물었던 자리:**

- 2026-07-27 actor 기동 실패 4건 중 1건 (`09` §0의 #1)
- `serl_ur_infra` 테스트 4개 파일의 무한 hang (`00` §4.2)

**규칙:** `serl_ur_infra`의 어떤 것이든 gRPC를 만질 가능성이 있으면 `$ACTOR_PY`를
**절대경로로** 쓴다. `python3`을 손으로 치지 않는다 — 실기 기동은 `run_hil_actor.sh`가
이걸 강제한다 (`09` §1).

### 1.1 venv 격리 원칙

**ROS Humble의 시스템 grpc/protobuf를 절대 갈아엎지 마라.** 갈아엎으면 rclpy가 죽고,
로봇 랩톱 전체가 못 쓰게 된다. 그래서 **고치는 게 아니라 우회한다.**

현재 쓰는 venv:

| | |
|---|---|
| 경로 | `/home/laptop3/venvs/gello-hil-actor` |
| 종류 | `--system-site-packages` (rclpy 등을 시스템에서 상속) |
| grpcio | **1.74.0** |
| protobuf | **3.20.3** (체크인된 생성 모듈이 요구. 올리면 깨진다) |
| 기타 | numpy 2.2.6, gymnasium 1.2.0, opencv-python 4.13.0, scipy 1.15.3 |

📌 2026-07-29 실측(위 표 전체를 이 한 줄로 재확인했다):

```bash
$ACTOR_PY -c "import grpc,numpy,gymnasium,cv2,scipy,google.protobuf as p; \
print(grpc.__version__, numpy.__version__, gymnasium.__version__, cv2.__version__, scipy.__version__, p.__version__)"
# -> 1.74.0 2.2.6 1.2.0 4.13.0 1.15.3 3.20.3
```

> 🔧 **정정:** 이전 판은 `/tmp/gello-hil-grpc-venv`를 만들라고 했다. `/tmp`는 재부팅에
> 날아간다. 지금 정본은 위의 `~/venvs/gello-hil-actor`이고, `run_hil_actor.sh`의
> `ACTOR_VENV` 기본값도 이 경로다.

- `serl_ur_infra` 자체는 `pip install --user -e ... --no-deps`로 설치한다 (`--no-deps` 필수).
- 실행 시 `PYTHONPATH`는 **덮어쓰지 말고 이어붙인다**:
  `PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}"` → `00` §3.4(2).

### 1.2 Kanu(서버) 쪽 — 오버레이 venv

Kanu의 공유 conda env `il`을 **수정하지 말 것**. 시스템 사이트 패키지를 읽는 작은 오버레이를
만든다 (`RL_RECEIVE_SERVER.md`의 "Preferred Kanu runtime"):

```bash
/home/junhyeong/miniconda3/envs/il/bin/python -m venv \
  --system-site-packages /tmp/gello-hil-rl-receive-overlay-v2
/tmp/gello-hil-rl-receive-overlay-v2/bin/python -m pip install --no-deps \
  -r serl_ur_infra/requirements-rlpd-receive-overlay.txt
```

`--no-deps`가 **의도적**이다. pip이 `il`에서 상속받은 패키지를 교체하는 것을 막는다.
오버레이가 더하는 것은 agentlace(upstream replay store의 베이스 클래스), lz4, protobuf 3.20.3뿐.

> ### 🛑 이 오버레이를 **learner 서버에 재사용하지 말 것**
> 오버레이가 고정하는 **protobuf 3.20.3이 `wandb` import를 깨뜨린다.**
> receive server(`run_rlpd_receive_server.py`)는 wandb를 쓰지 않아 문제가 없지만,
> learner(`run_rlpd_learner_server.py`)는 쓴다. learner는 별도 환경에서 띄운다
> (`serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md`).
>
> 그리고 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 반드시 export한다 — 안 하면 JAX가
> GPU 메모리를 통째로 선점해 같은 카드의 다른 작업을 죽인다.
>
> ⚠️ Kanu 환경 드리프트(2026-07-29 확인): numpy 2.2.5(lock 1.26.4), orbax 0.11.12(0.11.5),
> grpcio 1.80.0(1.74.0). 런타임 fail-closed 검사 대상은 jax/flax/distrax/tfp/wandb뿐이라
> **이 세 개는 자동으로 안 걸린다.**

---

## 2. 루프백 스모크 (로봇 없이, 위험 0)

### 2.1 mock 서버 + 스모크 클라이언트

```bash
# 터미널 1 — mock 서버 (기본 127.0.0.1:50052)
PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}" \
$ACTOR_PY \
  $WT/serl_ur_infra/scripts/run_actor_mock_server.py --host 127.0.0.1 --port 50052
```

```bash
# 터미널 2 — 스모크 클라이언트 (128x128 RGB 2장 × 관측 2개, 일반 1건 + 종단 개입 1건)
PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}" \
$ACTOR_PY \
  $WT/serl_ur_infra/scripts/run_actor_smoke_client.py --host 127.0.0.1 --port 50052
```

포트 기본값 (헷갈리기 쉬움):

| 스크립트 | 기본 포트 | 근거 |
|---|---|---|
| `run_actor_mock_server.py` | **50052** | `:23` |
| `run_actor_smoke_client.py` | **50052** | `:22` |
| `run_rlpd_receive_server.py` | **50053** | `:42` |
| `run_rlpd_receive_smoke_client.py` | **50053** | `:25` |
| `run_remote_rlpd_actor.py` (config 미지정 시) | **50053** | `:111` |
| `run_hil_actor.sh` (`SERVER_PORT` 기본값) | **50153** ← 터널 로컬 입구 | `:72` |
| `REMOTE_ACTOR_GRPC.md`의 `NETWORK` 예제 | 50053 | 문서 예제 |

📌 2026-07-29 재확인: 위 다섯 개 코드 기본값 전부 일치.

> ⚠️ `REMOTE_ACTOR_GRPC.md`는 한 문서 안에서 50052(스모크)와 50053(터널 예제)을 섞어 쓴다.
> **혼동하지 말고 항상 명시적으로 `--port`를 넘겨라.**

### 2.2 mock 서버가 무엇이고 무엇이 아닌가

- 루프백 전용 **제로 액션** mock이다. 정책 추론도, 보상도, 실제 replay도 없다
  (`REMOTE_ACTOR_GRPC.md` 머리말).
- 기본 sink는 리스트당 최대 **256개**만 보관하고 그 뒤로는 **조용히 버리지 않고 Step을 거부**한다.
- 표준출력에는 ID/카운터/타임스탬프/개입 라벨/텐서 dtype·shape만 찍는다.
  **이미지·액션 값은 저장도 출력도 하지 않는다.**

---

## 3. 프로토콜 요약 (v2, 서버 권위)

reset 시 `BeginEpisode(O0)` → `A0`. step `t`에서:

```
Local  → Server :  Step(O(t+1), D(t), request_action=?)
Server → Local  :  ACK(D(t)) + TransitionOutcome + A(t+1)
```

- 이미지는 **raw lossless numpy 바이트**로 한 번만 보낸다. `O(t)`는 `D(t)` 안에 다시 넣지 않는다.
- 종단/절단 스텝은 `request_action=false`로 같은 `Step`을 보낸다.
- **Local은 ACK를 검증하기 전에 절대 reset하지 않는다.**
- 서버 분류기가 잠정 non-terminal transition을 terminal로 확정할 수 있다.

### 3.1 실패 규칙 (fail-fast, 폴백 없음)

| 상황 | 동작 | 근거 |
|---|---|---|
| 액션 identity / shape `(7,)` / 유한성 / `[-1,1]` / 신선도 / `policy_version` 비감소 | Local에서 검사 | `REMOTE_ACTOR_GRPC.md` "Failure rules" |
| 일시적 timeout·UNAVAILABLE | **똑같은 직렬화 요청을 1회 재시도.** 서버 응답 캐싱으로 중복 추론/삽입 방지 | `grpc_actor_transport.py:41`(`_TRANSIENT_CODES`), `:792-806` |
| Step 실패 | transition을 pending으로 유지, reset/새 액션 차단 | 동일 |
| 분류기/replay 삽입 실패 | ACK 없음, 서버 not-ready, **Local은 fail stopped** | 동일 |
| 애플리케이션 오류·malformed 액션 | 에피소드 abort. **폴백/랜덤 액션을 실행하지 않는다.** `random_steps == 0` 필수 | 동일 |
| `retry_count != 1` | **생성 시 즉시 거부** | `grpc_actor_transport.py:401` |

---

## 4. 스키마 fail-fast — **v2로 확정됨**

### 4.1 어떻게 강제되는가

`GetServerInfo`가 광고하는 `observation_schema_hash`를 클라이언트가 자기 값과 대조하고,
다르면 즉시 예외를 던진다 (`grpc_actor_transport.py:575-584`,
`rlpd_receive_smoke.py:53-54`). `schema_version` 불일치도 마찬가지다 (`:538-542`),
`model_id`/`reward_authority`/`reward_model_id`도 같은 블록에서 pin된다 (`:551-574`).

해시는 스키마 문서의 SHA256이고, 문서에는 **정렬된 상태 피처 이름 목록**이 들어간다.
`json.dumps(sort_keys=True)`는 dict **키**만 정렬하고 **리스트 원소**는 정렬하지 않으므로,
**순서를 바꾸면 해시가 바뀐다 — 그게 의도다** (`observation_schema.py`의 `_schema_document()` 주석).

### 4.2 canonical v2 계약 (확정)

`serl_ur_infra/ur_env/observation_schema.py`의 v2 계약은 hardware commit `6a0b127`과
learner/hardware merge `248255f`에 통합됐고, 2026-07-27 Kanu 왕복에서 **양쪽 해시 일치가
실제로 확인**됐다. 더 이상 "바뀌는 중"이 아니다:

| | 이전 (v1, 사용 금지) | 현재 canonical (v2) |
|---|---|---|
| `OBSERVATION_SCHEMA_ID` | `hil-serl-ur-canonical-observation-v1` | `...-v2` |
| state 순서 | pose6, vel6, force3, torque3, **gripper(마지막)** | **알파벳순**: `gripper_pose`, `tcp_force`, `tcp_pose`, `tcp_torque`, `tcp_vel` |
| 그리퍼 인덱스 | 18 (`-1`) | **0** |
| 해시 | (구값) | 바뀜 |

이유: 평탄화를 하는 것은 우리가 아니라 upstream `SERLObsWrapper`이고, 그것이 쓰는
`gym.spaces.Dict`가 **평범한 매핑을 알파벳순으로 재정렬**한다. `proprio_keys`는 *어떤* 그룹이
들어갈지만 정하고 *어떤 순서로* 놓일지는 정하지 못한다.

> ### 실행 규칙
> - **해시나 인덱스를 문서/코드에 하드코딩하지 말 것.** 항상 라이브로 출력한다.
> - `state[..., -1]`로 그리퍼를 읽는 코드는 **틀렸다.** 그건 TCP 각속도 z다.
>   `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`를 쓴다.
> - laptop/server 모두 live `CANONICAL_OBSERVATION_SCHEMA_HASH`를 사용한다.

라이브 확인 (📌 실측 결과가 아래 블록 다음에 있다. **이 명령은 시스템 `python3`로 돌려도
안전하다** — `observation_schema.py`는 gRPC를 import하지 않는 의존성 경량 모듈이다):

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH python3 - <<'PY'
from ur_env.observation_schema import (
    OBSERVATION_SCHEMA_ID, CANONICAL_OBSERVATION_SCHEMA_HASH,
    STATE_FEATURES, STATE_DIM, GRIPPER_POSITION_INDEX, CANONICAL_STATE_LAYOUT)
print("id    :", OBSERVATION_SCHEMA_ID)
print("hash  :", CANONICAL_OBSERVATION_SCHEMA_HASH)
print("dim   :", STATE_DIM)
print("grip@ :", GRIPPER_POSITION_INDEX)
for k, a, b in CANONICAL_STATE_LAYOUT:
    print(f"  {k:12s} [{a}:{b})")
PY
```

📌 랩톱 실측 (2026-07-29 재실행. 머지 전후로 값이 바뀌지 않았다):

```
id    : hil-serl-ur-canonical-observation-v2
hash  : 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
dim   : 19
grip@ : 0
  gripper_pose [0:1)
  tcp_force    [1:4)
  tcp_pose     [4:10)
  tcp_torque   [10:13)
  tcp_vel      [13:19)
```

2026-07-27에 **Kanu 서버가 광고한 값과 동일함이 확인됐다** (`09_HIL_ACTOR_RUNBOOK.md` §2.1).
그래도 **양쪽에서 이 명령을 돌려 대조한 뒤에** 원격 스모크를 시작한다 — 위 해시를
복사해서 비교하지 말고, 양쪽 출력을 나란히 본다. 해시가 다르면 어떤 통신도 시도하지 말 것.

> ⚠️ `run_hil_actor.sh`의 `OBS_SCHEMA_HASH` 기본값도 이 해시로 **하드코딩**돼 있다
> (`:77`). 스키마를 바꾸면 그 줄과 Kanu 쪽을 함께 올려야 한다.

레이아웃 회귀 테스트 (📌 2026-07-29: `test_state_layout_contract` **22 passed**,
`test_observation_schema` **5 passed**):

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH $ACTOR_PY -m pytest \
  tests/test_state_layout_contract.py tests/test_observation_schema.py -q -p no:anyio
```

---

## 5. 🛑 레이턴시 예산 — 스모크 2.0 s vs 실제 0.6 s 함정

**스모크 클라이언트의 기본 타임아웃은 프로덕션 기본값보다 3배 이상 느슨하다.**
스모크가 통과했다고 실제 예산 안에 든다는 뜻이 **전혀 아니다.**

| 설정 | 프로덕션 기본 | `run_actor_smoke_client.py` 기본 | `run_rlpd_receive_smoke_client.py` 기본 |
|---|---|---|---|
| `timeout_s` | **0.6** | **2.0** | **3.0** |
| `max_response_age_s` | **0.8** | 3.0 | 4.0 |
| `retry_count` | 1 (다른 값 금지) | 1 | 1 |
| 근거 | `grpc_actor_transport.py:384-386`(생성자), `:494-496`(config 로더), `run_remote_rlpd_actor.py:112-114` | `run_actor_smoke_client.py:24`, `:25` | `run_rlpd_receive_smoke_client.py:30`, `:31` |

### 5.1 올바른 확인 방법

스모크를 **프로덕션 값으로 다시** 돌린다:

```bash
PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}" \
$ACTOR_PY \
  $WT/serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50052 \
  --timeout-s 0.6 --max-response-age-s 0.8
```

- **여기서 실패하면 실제 루프에서도 실패한다.** 통과할 때까지 예산을 늘리지 말고
  네트워크/서버를 고친다.
- 예산을 정말 늘려야 한다면 `NETWORK` config의 `timeout_s`/`max_response_age_s`를
  명시적으로 바꾸고, **왜 늘렸는지 이 문서에 남긴다.**

### 5.2 예산의 의미

- `timeout_s` = 한 gRPC 호출의 데드라인. 넘으면 transient로 보고 **1회만** 재시도.
- `max_response_age_s` = 서버가 찍은 `created_ns` 기준 응답 나이 상한
  (`grpc_actor_transport.py:794-797`). 오래된 응답은 버린다.
- env 루프는 `HZ = 10.0` (`config.py:32`) → 스텝 주기 100 ms.
  **`timeout_s=0.6`은 이미 6스텝 분량이다.** 여기에 재시도까지 겹치면 1.2 s가 날아간다.
  즉 이 예산은 "여유"가 아니라 이미 상당히 관대한 값이다.

### 5.3 🛑 📌 실측 기록 (2026-07-27, Kanu 왕복 100 스텝) — **예산이 사실상 소진 상태다**

> **이 절 전체가 2026-07-27 한 세션의 기록이다.** 링크(WiFi)와 서버 부하가 그때와 다르면
> 값도 다르다. 아래 숫자를 "현재 상태"로 인용하지 말고, 옮긴 뒤에는 §5.1의 명령으로
> 다시 측정해 이 표 아래에 새 줄을 추가한다.

Stage A(fake-env) 100 스텝 acceptance가 통과했고, 그때 측정된 값:

| 지표 | 실측 |
|---|---|
| 왕복 RTT p50 | **58.6 ms** |
| 왕복 RTT p95 | **75.8 ms** |
| **왕복 RTT p99** | **97.1 ms** |
| 관측 1건 크기 | **96.1 KiB** |
| 10 Hz에서의 상행 대역폭 | **약 7.9 Mbit/s** |

동반 확인: 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`
(절차·전체 로그는 `09_HIL_ACTOR_RUNBOOK.md`).

> ### 무엇을 뜻하는가
> **10 Hz 스텝 예산은 100 ms다. p99가 97.1 ms다.** 즉 100스텝 중 한 번꼴로 통신 하나가
> 스텝 예산을 통째로 먹는다. `timeout_s = 0.6`은 아직 여유가 있지만, **"예산 안에 든다"와
> "루프가 10 Hz를 유지한다"는 다른 얘기**다. 이 값들은 통과 판정이 아니라 **경고**로 읽어야 한다.
>
> **병목은 서버 추론이 아니라 WiFi 대역폭이다.** 96.1 KiB × 10 Hz = 7.9 Mbit/s를 무선으로
> 계속 밀고 있고, RTT 분포의 꼬리는 그 링크에서 나온다.

**따라서 다음 사람이 할 일 (우선순위 순):**

1. **유선으로 옮긴다.** 가장 싸고 가장 크게 듣는 조치다. 옮긴 뒤 같은 100스텝을 다시
   돌려 p99를 이 표 아래에 기록한다.
2. 그 전까지는 **`timeout_s`/`max_response_age_s`를 늘리지 말 것.** 늘리면 증상만 감춘다.
   재시도는 `retry_count = 1`로 고정이고 다른 값은 생성 시 거부된다.
3. 관측 크기를 줄이는 것은 **최후 수단**이다 — 이미지 인코딩을 바꾸면 스키마 해시가 바뀌고
   체크포인트/데모와 어긋난다.
4. 실기(Stage B)에서는 여기에 **센서 파이프라인 지연이 더해진다.** 이 표의 숫자는
   fake-env 값이므로 **실기 상한이 아니라 하한**이다.

---

## 6. Kanu 터널

서버 gRPC는 **서버 loopback에만** 열어 두고 랩톱에서 포워딩한다.

```bash
# 랩톱 — 2026-07-27에 실제로 쓴 형태 (로컬 50153 -> 원격 50053)
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50153:127.0.0.1:50053 kanu
```

- `ExitOnForwardFailure=yes`가 **중요하다.** 없으면 포워딩이 실패해도 ssh가 살아 있어서
  "연결됐는데 왜 안 되지"로 시간을 버린다.
- 양쪽에서 **비어 있는 것이 확인된 포트**를 쓴다.
- 🔧 **로컬 쪽이 50053이 아니라 50153인 이유:** 2026-07-27에 로컬 `50053`이 다른
  프로세스에 잡혀 있었다. **터널의 로컬 쪽만 바꾸고 원격 쪽은 50053 그대로** 둔다.
  `run_hil_actor.sh`의 `SERVER_PORT` 기본값도 `50153`이다.

```bash
# 터널 확인 (로컬 쪽 포트를 본다)
ss -ltnp | grep 50153
```

서버 기동 (Kanu). **정본 절차와 함정은 [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) §2.2**
— 여기서는 형태만 보인다:

```bash
CUDA_VISIBLE_DEVICES=<nvidia-smi로 비어 있는 GPU> \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
<overlay venv>/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 --port 50053 \
  --checkpoint <classifier checkpoint 경로> \
  --expected-checkpoint-sha256 <그 체크포인트의 sha256> \
  --reward-model-id <서버가 광고할 id> \
  --require-jax-backend gpu
```

> ### 🔧 `--threshold`를 빼 놓은 이유 (2026-07-29 정정)
> 이전 판은 `--threshold 0.5`를 박아 뒀다. **코드 기본값은 이제 `0.2`다**
> (`DEFAULT_REWARD_THRESHOLD`, `rlpd_receive_server.py:73`; `0.85` → `0.5`(`53d5cf6`) →
> `0.2`(`1b02857`)). 문서에 리터럴을 두면 또 어긋나므로 **생략해서 코드 기본값을 쓰게 한다.**
> 명시해야 하는 경우는 하나뿐이다: **다른 threshold로 학습된 checkpoint를 resume할 때.**
> threshold는 learner fingerprint에 들어가고(`run_rlpd_learner_server.py:548-550`),
> 불일치는 fail-closed로 거부된다.
>
> `--replay-capacity` / `--intervention-capacity`도 뺐다 — 코드 기본값이 정확히
> 50000 / 10000이다 (`rlpd_receive_server.py:49-50`).

랩톱에서 수신 스모크 (**터널 로컬 포트**를 쓴다):

```bash
PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}" \
$ACTOR_PY \
  $WT/serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50153 --timeout-s 0.6 --max-response-age-s 0.8
```

수신 스모크는 시작하자마자 세 가지를 **fail-fast**로 검사한다 (`rlpd_receive_smoke.py:46-58`).
⚠️ 이 클라이언트는 **서버 버퍼에 synthetic transition을 실제로 삽입한다** — 프로덕션 수집
중에는 돌리지 말 것 (`09` §3.2):

1. `health()`의 `alive && ready`
2. `observation_schema_hash` 일치
3. `reward_authority == "server_classifier"`

> ### upstream submodule 확인
> 수신 서버는 upstream replay store를 쓰므로 `serl_launcher`가 필요하다.
> `git submodule status third_party/hil-serl`에 `-`가 붙으면 `00_SETUP_AND_SAFETY.md` §2.3에
> 따라 초기화한다. Kanu도 동일한 pinned revision을 사용한다.

---

## 7. 판정 체크리스트

- [x] **인터프리터가 `$ACTOR_PY`인지 확인** — `$ACTOR_PY -c "import grpc; print(grpc.__version__)"`
      가 `1.74.0`이어야 한다. `1.30.2`가 보이면 즉시 중단 (§1.0)
- [x] **시스템 python이 오염되지 않았다** (`python3 -c "import rclpy"`가 여전히 동작)
- [x] mock 서버 + 스모크 클라이언트가 루프백에서 통과 (기본값)
- [ ] **같은 스모크가 `--timeout-s 0.6 --max-response-age-s 0.8`에서도 통과** (§5.1)
- [x] 랩톱과 Kanu의 스키마 해시가 **동일** (§4.2) — 2026-07-27 확인
- [x] `test_state_layout_contract.py` 통과 (📌 2026-07-29: 22 passed)
- [x] SSH 터널이 `ExitOnForwardFailure`로 열리고 `ss -ltnp`에 보인다 (로컬 50153)
- [x] 수신 스모크의 3가지 fail-fast 검사 통과
- [ ] 서버 로그에 이미지/액션 값이 찍히지 않는다 (§2.2)
- [ ] **§5.3의 레이턴시를 유선에서 재측정하고 p99를 기록** ← 지금 가장 값싼 개선
- [ ] **서버 checkpoint가 폐기된 07-24 것이 아닌지 확인** — `run_rlpd_receive_server.py:34`의
      `DEFAULT_CHECKPOINT_SHA256`이 아직 그것을 pin하고 있다 → `08` G19

> `[x]`는 2026-07-27 Kanu Stage A(fake-env) 왕복에서 확인된 것이다 (📌 기록).
> **실기 센서를 붙인 Stage B는 아직 하나도 체크되지 않았다** → `09_HIL_ACTOR_RUNBOOK.md`.
> 지금은 Kanu에 서버가 떠 있지 않으므로, 다시 하려면 §6의 서버 기동부터 시작한다.
