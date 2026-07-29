# 05 — 통신 (gRPC actor · 스키마 · 레이턴시 · Kanu 터널)

**상태: 2026-07-27 실기 세션에서 크게 진전.**

| 항목 | 상태 |
|---|---|
| gRPC 트랜스포트 단위/루프백 | **PASS** — 35 tests (venv python) |
| 스키마 해시 랩톱↔Kanu 일치 | **PASS** (2026-07-27 왕복에서 확인) |
| Kanu 왕복 (Stage A, fake-env, 100 스텝) | 📌 **PASS (2026-07-27)** — `replay_insert_count: 100` |
| 레이턴시 예산 | 📌 **07-27과 07-29 두 세션이 6배 다르다.** 저장된 상수를 믿지 말고 **세션 시작마다 다시 재라** (§5.3) |
| 분류기 sidecar (무크롭 프레임 첨부) | **코드·단위테스트까지. 실기 미검증.** proto 변경 0, **스키마 해시 불변**, 대역폭 +2.8 % (§3.2, §5.4) |
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

### 3.2 분류기 sidecar — 관측에 얹혀 가는 **무크롭 JPEG** (2026-07-29 신규)

**무엇을 고쳤나.** reward classifier는 **무크롭** 1280×720 프레임을 128×128로 눌러
학습됐는데, actor는 **정책의 크롭된 관측**(`ur_experiments/cube_in_cup.py`의 `IMAGE_CROP` —
cam1 `img[20:670, 340:990]`, cam2 `img[0:720, 420:1140]`)을 그대로 분류기에 먹이고 있었다.
대가는 실측 **recall@0.85 100 % → 33.3 %**. 그것이 G15였다.

**해결책은 재학습이 아니라 분리다.** 정책은 실측 크롭을 그대로 유지하고,
분류기에게는 **자기 몫의 무크롭 128×128 JPEG**를 따로 붙여 보낸다.

| | |
|---|---|
| 예약 관측 키 | `classifier` (`ur_env/classifier_sidecar.py::CLASSIFIER_SIDECAR_KEY`) |
| 안에 든 것 | `cam1_jpeg` / `cam2_jpeg` — 각각 JPEG 파일 하나가 통째로 든 **1-D uint8** 배열 |
| 붙는 빈도 | **약 2 Hz** (`interval_steps = 5` @ 10 Hz) **이고 팔이 정지해 있을 때만** |
| 만드는 쪽 | 랩톱. `get_im()`이 이미 디코드해 둔 **크롭 전** 풀해상도 BGR(`ur7e_env.py`의 `self._last_camera_frames[key] = bgr`, 크롭은 바로 **다음 줄**)에 `cv2.resize(bgr,(128,128))` → JPEG 인코드 |
| 푸는 쪽 | 서버. `decode_classifier_frames()`가 **라이브 뷰어와 같은 레시피**(imdecode → 무크롭 → `resize(128,128)` → RGB → batch axis)로 푼다 |
| 프레임당 상한 | 64 KiB (`MAX_SIDECAR_JPEG_BYTES`). 넘으면 **거부** — 풀해상도 프레임이 새어 들어온 것 |

> ### 🟢 **proto 변경 0 · 스키마 해시 불변** — 그래서 기존 pin이 전부 살아 있다
> `proto/actor_transport.proto`는 이미 **일반 named-tensor 맵**이다
> (`Tensor{path,dtype,shape,data}` + `Observation{repeated Tensor tensors}`).
> 관측 키가 하나 늘어도 proto를 손댈 이유가 없다.
>
> 그리고 해시는 **wire payload가 아니라 `ur_env/observation_schema.py`의
> `CANONICAL_OBSERVATION_SPEC` 문서에서** 나온다. sidecar는 canonical 검증 **전에**
> `actor_network.py`의 `_split_classifier_sidecar()`가 벗겨 내므로
> `validate_canonical_observation()`은 예약 키를 아예 보지 못한다.
>
> **실측 확인 (2026-07-29, 이 문서 작성 중 직접 계산):**
> `CANONICAL_OBSERVATION_SCHEMA_HASH = 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903`
> — **§4.2의 값과 같다.** `run_hil_actor.sh`의 `OBS_SCHEMA_HASH`, Kanu 서버,
> §4.2의 라이브 대조 절차까지 **전부 그대로 유효하다.**

**대신 바뀐 것: `reward_model_id`가 입력 계약을 이름에 담는다.**
서버가 광고하는 값이 `cube-in-cup-checkpoint-150` → **`cube-in-cup-all3-ckpt150+sidecar-v1`**
로 바뀌었고 (`scripts/run_rlpd_{receive,learner}_server.py::DEFAULT_REWARD_MODEL_ID`,
이제 **기본값**이다), actor 쪽 짝인 `run_hil_actor.sh::EXPECTED_REWARD_MODEL_ID`도 같이 바뀌었다.
접미사 `+sidecar-v1`은 `CLASSIFIER_INPUT_ID = "fullframe-jpeg-passthrough-v1"`을 가리킨다.

> 🎯 **왜 일부러 id를 깨뜨렸나.** sidecar 이전 actor ↔ 이후 server(또는 반대)는
> **서로 다른 픽셀에서 reward를 계산한다.** 이 불일치는 아무 에러도 내지 않고 한 세션을
> 통째로 오염시키므로 **핸드셰이크에서 거부**한다 (`grpc_actor_transport.py:551-574`의
> 동일성 검사). 옛 값을 그대로 넘기면 **첫 inference 전에 죽는다 — 그게 의도다.**

**성긴 분류는 타협이 아니라 설계다.** 대부분의 transition은 sidecar 없이 도착하고,
그 스텝은 `rewards`가 **0.0으로 확정**되며 `classifier_evaluated=0`,
probability/threshold/success는 전부 0, `reward_model_id`는 빈 문자열이다
(`masks`/`dones`/`truncated`는 로컬 제안이 그대로 통과한다 —
`rlpd_receive_server.py::RewardTransitionFinalizer`). 이유 두 가지:
큐브를 놓은 뒤 **장면이 가라앉을 시간**을 주고, 10 Hz로 판정이 **깜빡이는 것**을 막는다.
대역폭 절감은 부수 효과다.

N-of-M 시간 평활(`--success-confirmations`)은 **기본값이 1 = 꺼짐**이므로,
서버가 보고하는 `classifier_probability`는 **시간 필터가 전혀 없는 순간 sigmoid**다 —
라이브 뷰어가 찍는 것과 **같은 종류의 수치**이고, 그래서 둘의 비교가 근사가 아니라
**등가성 점검**이 된다. 확인 절차는 `09` §4.4.

> ⚠️ **"비트 단위로 같다"까지는 아니다.** 평활 단계는 양쪽 다 없지만, sidecar는 랩톱에서
> 128×128로 재인코딩되므로 **픽셀이 JPEG 1세대만큼 다르다.** 실측 기대 차이는
> `|Δp| ≤ 0.010`(순수 JPEG 왕복 대조군) ~ `0.022–0.090`(판정 경계 근처, 합성 프레임) —
> **주변 센서 노이즈와 같은 수준**이다 (§5.4). 오프라인에서 **비트 단위**로 강제되는 것은
> **resize-only 경로**뿐이고, 그것을 `tests/test_classifier_sidecar.py`가 뷰어 출력과 대조한다.

📌 `DEFAULT_REWARD_THRESHOLD`는 **0.2 그대로**다. 바뀌지 않았다.

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

> ### 📌 2026-07-29 분류기 sidecar가 이 해시를 **바꾸지 않았다**
> wire에는 관측 키가 하나 늘었지만(§3.2) 해시는 `CANONICAL_OBSERVATION_SPEC` **문서**에서
> 나오고 sidecar는 canonical 검증 전에 벗겨진다. 위 실측 블록의 값은 sidecar 코드가
> 들어온 트리에서 다시 계산한 것이고 **머지 이전과 같다.**
> **즉 이 절의 대조 절차는 손댈 것이 없다.** 그래도 대조 자체는 계속 한다 —
> 이 절이 막는 것은 sidecar가 아니라 state 레이아웃 드리프트다.

레이아웃 회귀 테스트 (📌 2026-07-29: `test_state_layout_contract` **22 passed**,
`test_observation_schema` **5 passed**):

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH $ACTOR_PY -m pytest \
  tests/test_state_layout_contract.py tests/test_observation_schema.py -q -p no:anyio
```

---

## 5. 🛑 레이턴시·대역폭 예산 — 스모크 2.0 s vs 실제 0.6 s 함정

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

### 5.3 🛑 📌 레이턴시 실측 — **두 세션이 6배 다르다. 저장된 상수를 믿지 마라**

> ## 🛑 이 절에서 숫자를 베껴 쓰지 마라
> 같은 링크를 잰 값이 리포 안에 **셋 있고 전부 다르다** — 13 / 47.8 / 83 Mbit/s.
> **셋 다 진짜 관측일 가능성이 높다.** 결론은 "빨라졌다"가 아니라
> **"이 링크는 세션 간 약 6배 흔들린다"**이다. 그러니 **세션 시작마다 다시 재라 —
> 07-29 값도 상수가 아니다.**

#### (A) 📌 2026-07-27 기록 — Kanu 왕복 100 스텝, Stage A(fake-env)

| 지표 | 실측 |
|---|---|
| 왕복 RTT p50 | **58.6 ms** |
| 왕복 RTT p95 | **75.8 ms** |
| **왕복 RTT p99** | **97.1 ms** |
| 관측 1건 크기 | **96.1 KiB** |
| 10 Hz에서의 상행 대역폭 | **약 7.9 Mbit/s** |
| 당시 추정 링크 용량 | 약 **13 Mbit/s** |

동반 확인: 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`
(절차·전체 로그는 `09_HIL_ACTOR_RUNBOOK.md`).

이 기록 위에 세워진 결론이 *"p99가 100 ms 예산의 97 %다. 여유가 없다. 병목은 무선 링크다"*
였고, 이 문서와 `08`·handoff가 그것을 반복했다.

> ### 🔍 그 수치의 정체 — **추론 (INFERRED, 실측 아님)**
> 07-27 수치는 **거의 전부 전송 시간으로 설명된다**:
> **96.1 KiB ÷ 13 Mbit/s = 60.5 ms**인데 기록된 **p50이 58.6 ms**다. 거의 일치한다.
> 즉 그날의 **서버 계산 시간은 무시할 만했고**, 측정된 것은 사실상 링크뿐이었다.
> 이것은 두 저장값을 맞춰 본 산술이지 그날 별도로 잰 값이 아니다 — **추론으로 인용할 것.**

#### (B) 📌 2026-07-29 실측 — 같은 랩톱 → 같은 Kanu, **조건이 완전히 다르다**

| | 값 | 조건 |
|---|---|---|
| ssh-실효 처리량 | **중앙값 약 83 Mbit/s** (min 75.5 / max 98.5) | 32 MiB **비압축** 업로드 12회, **multiplexed SSH 연결 안에서** |
| ICMP RTT | p50 **1.75** / p95 **9.99** / p99 **24.4** / max **38.3 ms**, 손실 **0 %** | 600 샘플 |
| 링크 | WiFi **2.4 GHz ch.3**, SSID `iptime_709`, PHY 216–270 Mb/s, 신호 −42/−44 dBm | AP가 **5 GHz SSID를 아예 방송하지 않는다** |
| 경로 | **캠퍼스 4홉** `192.168.0.1 → 10.20.44.1 → 10.22.2.101 → kanu` | 주소는 공인 IP(`166.104.35.33`)지만 **트래픽은 인터넷을 타지 않는다** |
| 주변 간섭 | 2.4 GHz에 **AP 41개** 관측 | **공유·혼잡 대역** |
| 리그 상태 | **완전 유휴** — 카메라·actor·조작자 트래픽 **전무** | ⚠️ 이게 이 표의 가장 중요한 한 줄이다 |

**`ssh`-실효 값이 맞는 기준이다** — actor의 gRPC가 **SSH 터널 안으로** 흐르기 때문이다(§6).
암호화는 천장이 아니다(chacha20 ↔ aes128-gcm이 77 → 83으로만 움직였다).
**천장은 2.4 GHz 라디오다.**

> 📌 이 문서를 쓰면서 **독립적으로 재확인한 것**(전부 랩톱에서 직접 실행):
> `iptime_709`의 BSSID는 **하나뿐이고 2422 MHz(ch.3)** — 5 GHz 짝이 없다.
> `tracepath 166.104.35.33` → **4홉, 전 구간 2 ms 미만**.
> `ping -c 100` → **min 0.922 / avg 3.607 / max 25.6 ms, 손실 0 %** (위 600샘플과 같은 계열).
> `enx00e04c3600bd`는 **존재하고 `NO-CARRIER`** — 즉 **꽂혀만 있지 않다.**
> 단 **AP 개수는 스캔마다 흔들린다** — 같은 날 재스캔에서는 2.4 GHz에 20개였다.
> "41개"는 그 시점의 관측이고, 옮겨 적을 값이 아니라 **혼잡한 공유 대역이라는 뜻**으로 읽는다.

#### (C) 애플리케이션 burst 왕복 — **전송 바닥(floor)이지 스텝 시간이 아니다**

| 페이로드 | p50 | p95 |
|---|---|---|
| 96 KiB (= 관측 1건 크기) | **11.8 – 21.6 ms** | 28 – 36 ms |
| 200 KiB | 24.9 – 28.9 ms | 50 – 55 ms |

> ⚠️ **원격 쪽이 바이트를 그냥 버렸다.** 서버 추론이 포함된 end-to-end 스텝 시간을
> **예측하지 않는다.** 전송 하한만 묶어 준다.
>
> 🔍 **추론:** (A)의 6.4배를 적용하면 p50 **9–12 ms**가 예측되는데, 위 96 KiB 실측이
> **11.8–21.6 ms**로 그 예측을 감싼다 — 독립적인 두 줄기가 맞아떨어진다.

#### (D) 🛑 그래서 결론이 무엇인가

**"레이턴시 문제가 해결됐다"로 읽지 마라.** 위 측정은 **유휴 리그**에서, **변동이 큰
공유 2.4 GHz 대역**에서 나왔다. 카메라 2대 + actor + 조작자가 **동시에** 붙은 상태는
**아직 한 번도 측정된 적이 없다.** 13 Mbit/s는 오류가 아니라 **열화된 상태의 실제 관측**이었을
것이고 **다시 나타날 수 있다** — 그러면 p99 97.1 ms 세계가 그대로 돌아온다.

**따라서 다음 사람이 할 일 (우선순위 순):**

1. **세션 시작마다 §5.1을 다시 돌려 그날의 값을 만든다.** 이 문서의 (A)도 (B)도
   설정값이 아니라 기록이다. 새 세션 값은 이 절 아래에 **줄을 추가**해서 남긴다.
2. **유선으로 옮긴다 — 지금 물리적으로 가능하다.** USB 이더넷 NIC
   **`enx00e04c3600bd`가 랩톱에 존재하고, 꽂혀만 있지 않다.** 코드 변경 0으로 변동성을
   통째로 없앤다. 옮긴 뒤 같은 100스텝을 다시 돌려 p99를 기록한다.
3. 그 전까지는 **`timeout_s`/`max_response_age_s`를 늘리지 말 것.** 늘리면 증상만 감춘다.
   재시도는 `retry_count = 1`로 고정이고 다른 값은 생성 시 거부된다.
4. 관측 크기를 줄이는 것은 **최후 수단**이다 — 이미지 인코딩을 바꾸면 canonical 스키마가
   바뀌고 체크포인트/데모와 어긋난다. (분류기 sidecar는 이 함정을 피했다 — §5.4.)
5. 실기(Stage B)에서는 여기에 **센서 파이프라인 지연이 더해진다.** (A)·(C) 모두
   fake-env / 전송-only 값이므로 **실기 상한이 아니라 하한**이다.

### 5.4 📌 대역폭 회계 (2026-07-29 실측) — 관측 크기와 sidecar 비용

#### 관측 1건이 실제로 몇 바이트인가

| | 값 |
|---|---|
| numpy 텐서 합계 | **98,380 B = 96.07 KiB** (cam1 49,152 + cam2 49,152 + state 76) |
| `Observation` protobuf 프레이밍 | +126 B |
| `StepRequest` 봉투 포함 **wire 총량** | **98,888 B = 96.57 KiB / step** |
| 10 Hz 상행 | **7.911 Mbit/s** |

> 📌 numpy 합계 98,380 B는 이 문서를 쓰면서 `CANONICAL_OBSERVATION_SPEC`에서 직접
> 계산해 확인했다 — (A) 표의 "96.1 KiB"와 같은 값이다. 프레이밍/봉투 항목은
> 07-29 계측 세션의 값이다.

> ### 🔑 이미지는 **raw uint8로, 압축 없이** 간다
> 그래서 128×128짜리가 장당 **48 KiB**나 한다 (`128×128×3 = 49,152 B`).
> 그리고 이것이 **압축된** 128×128 sidecar가 자기 옆에 실려 가는 **raw 정책 이미지보다
> 30배 싼** 이유다. 관측 이미지를 압축하는 것은 canonical 스키마를 건드리는 일이라
> 별개의(그리고 훨씬 비싼) 결정이다 — §5.3 (D)-4.

#### 🔧 정정 두 개 — 이전 판이 두 번 틀렸다

> **🔧 정정 1.** 이전 판(및 handoff)은 분류기 이미지를 따로 보내면 대역폭이 **"약 2배"**가
> 된다고 적었다. 그 추정은 **매 스텝 10 Hz로 raw uint8**을 보낸다는 가정이었다.
> **둘 다 지금 설계에 해당하지 않는다** — 실제로는 **2 Hz**로, **JPEG**로 간다.
>
> **🔧 정정 2 (더 크다).** `REWARD_CLASSIFIER_LIVE_KO.md`의 성능 표(2026-07-29 기준
> "720p JPEG 디코드 + 리사이즈" 줄. **줄 번호는 밀린다 — `rg "68 KB"`로 찾을 것**) 등이 라이브 720p JPEG를
> **"약 68 KB"**로 적고 있다. **3배 틀렸다.** 카메라 노드는 튜닝 안 된 ROS
> `image_transport` 기본값 `jpeg_quality = 95`로 돌고, 실측(각 250샘플, 29.98/30.13 Hz,
> `1280x720`, `rgb8; jpeg compressed bgr8`)은 **cam1 206.30 KiB / cam2 193.59 KiB 중앙값**이다.
> 68 KB는 **q75 재인코딩** 값이었다(q75가 cam1 = 68,686 B를 재현한다).
> **바로 이것이 sidecar가 원본 프레임을 그대로 보내지 않고 128×128로 줄여 보내는 이유다.**

#### sidecar 실측 크기 (100 프레임/카메라, protobuf 프레이밍 +82 B 포함)

| JPEG quality | cam1 평균 | cam2 평균 | 쌍 합계 |
|---|---|---|---|
| **95 (출하 기본값)** | 7,463 B | 6,090 B | **13.32 KiB** |
| 90 | 5,325 B | 4,060 B | 9.24 KiB |
| 85 | 4,329 B | 3,106 B | 7.34 KiB |
| 75 | 3,342 B | 2,277 B | 5.57 KiB |

산포는 무시할 만하다 (q95 쌍 최댓값 13,663 B vs 평균 13,553 B).

#### 합산 스트림 (q95, 관측 10 Hz + sidecar 2 Hz)

| | 값 |
|---|---|
| 합계 | **8.129 Mbit/s** = 현재 7.911 대비 **+2.8 %** |
| 13 Mbit/s 링크 대비 | 62.5 % |
| 45.6 Mbit/s 링크 대비 | 17.8 % |
| **에스컬레이션 최악(10 Hz 부착)** | **9.00 Mbit/s — 두 링크 다 들어간다** |
| 부착 스텝 스파이크 | **+8.4 ms** @13 Mbit/s · **+2.4 ms** @45.6 Mbit/s |

(A)의 RTT 분포에 +8.4 ms를 얹으면 **p99 꼬리만** 예산을 넘는다 —
**열화된 링크에서 5.5 ms 초과(105.5 ms)**, 빠른 링크에서는 **99.5 ms로 예산 안**이다.

> ### ❌ 기각된 설계: 원본 720p JPEG passthrough
> 첫 설계는 카메라의 JPEG를 손대지 않고 그대로 실어 보내는 것이었다 — CPU 0, 새 아티팩트 0.
> **실측이 죽였다.** 쌍 400 KiB → 10 Hz 에스컬레이션 시 **40.68 Mbit/s = WiFi 링크의 313 %**,
> 부착 스텝 스파이크 **+252.0 ms**(빠른 링크에서도 +71.9 ms) — **100 ms 예산에서 즉사**다.
> 그래서 랩톱이 **서버가 어차피 했을 바로 그 `cv2.resize(bgr,(128,128))`**를 직접 돌리고
> JPEG로 인코딩한다. 서버는 어느 호스트가 resize했는지 알 필요가 없다.

#### 왜 q95인가 — **직관과 반대라서 적어 둔다**

싼 quality를 쓰면 될 것 같지만 **아니다.** 함께 측정된 대조군:

| | 판정 경계 근처 \|Δp\| | 경계 뒤집힘 (20건 중) |
|---|---|---|
| 720p → JPEG 왕복 → 720p (대조군) | ≤ 0.010 | **0** |
| 주변 센서 노이즈, 프레임 간 | 0.027 – 0.102 | — |
| **128×128 q95** | 0.022 – 0.090 | 0 – 7 |
| 128×128 q75 | 0.118 – 0.198 | **14 – 16** |

**JPEG 압축 자체는 거의 공짜다.** 교란은 **128×128에서 양자화**하는 데서 온다 —
그 해상도에서 8×8 DCT 블록 하나가 화면의 **1/16**을 덮는다.
q95는 **주변 센서 노이즈와 같은 수준**에 앉는다. 무손실 PNG도 재 봤다 —
**쌍당 55.3 KiB로 두 링크 모두 예산 초과**다.

> ### ⚠️ 이 표의 경계 수치는 **합성(SYNTHETIC)이다 — 반드시 같이 인용할 것**
> 큐브를 컵에 합성해 알파 블렌딩한 프레임에서 잰 값이다.
> **quality 사이의 순위는 믿어도 된다. 절대 뒤집힘 횟수는 실기 오류율 예측이 아니다.**
> 실제 프레임 100장은 전부 `p = 0.0033 – 0.0165`에 있었으므로,
> 그 100장이 보여주는 것은 **"확신하는 표본은 계속 확신한다"**뿐이다.

부착/미부착 왕복은 실기에서 **추정하지 말고** `ActorRunSummary`의
`sidecar_round_trip_ms_mean/max` vs `plain_round_trip_ms_mean/max`로 **따로** 본다.
두 계열을 분리해 둔 이유가 이것이다 (`ur_env/remote_actor.py`).

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
  --reward-model-id cube-in-cup-all3-ckpt150+sidecar-v1 \
  --require-jax-backend gpu
```

> ### 🔧 `--reward-model-id`는 2026-07-29부터 **기본값이 있다**
> `cube-in-cup-all3-ckpt150+sidecar-v1` (`run_rlpd_receive_server.py::DEFAULT_REWARD_MODEL_ID`).
> **이전 값 `cube-in-cup-checkpoint-150`을 그대로 넘기면 핸드셰이크에서 거부된다** —
> actor 쪽 짝인 `run_hil_actor.sh::EXPECTED_REWARD_MODEL_ID`도 같이 바뀌었기 때문이다.
> id가 체크포인트와 **입력 계약**을 둘 다 담는 이유는 §3.2에 있다. 위 명령에 명시해 둔 것은
> **서버와 actor가 같은 문자열을 쓰는지 눈으로 대조하라는 뜻**이지, 기본값이 없어서가 아니다.
>
> 같이 생긴 플래그: **`--success-confirmations`(기본 `1` = 평활 꺼짐)**.
> 기본값을 그대로 두면 서버의 `classifier_probability`가 라이브 뷰어와 **같은 숫자**다.
> 올리면 그 등가성이 깨지고, learner fingerprint도 바뀐다.
>
> 🛑 **체크포인트 기본 SHA도 07-29에 교체됐다.** 두 `DEFAULT_*_SHA256`가
> 폐기된 `e329986b…`(새 도메인 recall 0 %)에서
> **`512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d`**
> (= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`의 **디렉터리** digest, 정규 파일 14개)로
> 바뀌었고, `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임해
> orbax 디렉터리를 해시한다 → `08` G19.

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
- [ ] **세션 시작마다 §5.3의 레이턴시를 다시 재고 그날 값을 기록** ← 저장된 상수를 쓰지 않는다
- [ ] **유선(`enx00e04c3600bd`)으로 옮기고 같은 100스텝 재측정** ← 지금 가장 값싼 개선
- [x] **서버 checkpoint 기본 SHA가 폐기된 07-24 것이 아니다** — 두 `DEFAULT_*_SHA256`가
      `512b6575…`(orbax 디렉터리 digest)로 교체됐다 (2026-07-29, `08` G19).
      그래도 **기동 로그에서 어느 체크포인트가 로드됐는지 눈으로 확인한다** — 이 실패는 조용하다
- [ ] **서버와 actor의 `reward_model_id`가 같은 문자열이다** (`cube-in-cup-all3-ckpt150+sidecar-v1`).
      다르면 첫 inference 전에 죽는 것이 정상이다 (§3.2)
- [ ] **뷰어 `p(success)` == 서버 `classifier_probability`** (평활 꺼짐 기준) → `09` §4.4.
      크롭 불일치가 실제로 고쳐졌는지 확인하는 가장 값싼 증거다

> `[x]`는 2026-07-27 Kanu Stage A(fake-env) 왕복에서 확인된 것이다 (📌 기록).
> **실기 센서를 붙인 Stage B는 아직 하나도 체크되지 않았다** → `09_HIL_ACTOR_RUNBOOK.md`.
> 지금은 Kanu에 서버가 떠 있지 않으므로, 다시 하려면 §6의 서버 기동부터 시작한다.
