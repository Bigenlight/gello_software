# 05 — 통신 (gRPC actor · 스키마 · 레이턴시 · Kanu 터널)

**상태: 이 브랜치에서 미검증.** 절차와 계약은 코드/문서에 있고, 아래는 그것을 실행 가능한
형태로 정리한 것이다.

정본: `serl_ur_infra/REMOTE_ACTOR_GRPC.md`, `serl_ur_infra/RL_RECEIVE_SERVER.md`,
`serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md`.

```bash
export WT=/home/laptop3/gello_software
```

---

## 1. 🛑 venv 격리 원칙 — 이게 1번인 이유

**ROS Humble의 시스템 grpc/protobuf를 절대 갈아엎지 마라.** 갈아엎으면 rclpy가 죽고,
로봇 랩톱 전체가 못 쓰게 된다.

```bash
# 로봇 랩톱: gRPC 전용 격리 venv
python3 -m venv /tmp/gello-hil-grpc-venv
/tmp/gello-hil-grpc-venv/bin/python -m pip install -r $WT/serl_ur_infra/requirements-grpc.lock
```

`requirements-grpc.lock` 내용 (핀 고정):

```
grpcio==1.74.0
numpy==1.26.4
protobuf==3.20.3
```

- `protobuf==3.20.3`은 **체크인된 생성 gRPC 모듈이 요구하는 버전**이다. 올리면 깨진다.
- `serl_ur_infra` 자체는 `pip install --user -e ... --no-deps`로 설치한다 (`--no-deps` 필수).
- 실행 시엔 `PYTHONPATH=$WT/serl_ur_infra`를 앞에 붙인다.

### 1.1 Kanu(서버) 쪽 — 오버레이 venv

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

---

## 2. 루프백 스모크 (로봇 없이, 위험 0)

### 2.1 mock 서버 + 스모크 클라이언트

```bash
# 터미널 1 — mock 서버 (기본 127.0.0.1:50052)
PYTHONPATH=$WT/serl_ur_infra \
/tmp/gello-hil-grpc-venv/bin/python \
  $WT/serl_ur_infra/scripts/run_actor_mock_server.py --host 127.0.0.1 --port 50052
```

```bash
# 터미널 2 — 스모크 클라이언트 (128x128 RGB 2장 × 관측 2개, 일반 1건 + 종단 개입 1건)
PYTHONPATH=$WT/serl_ur_infra \
/tmp/gello-hil-grpc-venv/bin/python \
  $WT/serl_ur_infra/scripts/run_actor_smoke_client.py --host 127.0.0.1 --port 50052
```

포트 기본값 (헷갈리기 쉬움):

| 스크립트 | 기본 포트 | 근거 |
|---|---|---|
| `run_actor_mock_server.py` | **50052** | `:23` |
| `run_actor_smoke_client.py` | **50052** | `:22` |
| `run_rlpd_receive_server.py` | **50053** | `:42` |
| `run_rlpd_receive_smoke_client.py` | **50053** | `:25` |
| `REMOTE_ACTOR_GRPC.md`의 `NETWORK` 예제 | 50053 | 문서 예제 |

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
| 일시적 timeout·UNAVAILABLE | **똑같은 직렬화 요청을 1회 재시도.** 서버 응답 캐싱으로 중복 추론/삽입 방지 | `grpc_actor_transport.py:733-746` |
| Step 실패 | transition을 pending으로 유지, reset/새 액션 차단 | 동일 |
| 분류기/replay 삽입 실패 | ACK 없음, 서버 not-ready, **Local은 fail stopped** | 동일 |
| 애플리케이션 오류·malformed 액션 | 에피소드 abort. **폴백/랜덤 액션을 실행하지 않는다.** `random_steps == 0` 필수 | 동일 |
| `retry_count != 1` | **생성 시 즉시 거부** | `grpc_actor_transport.py:398` |

---

## 4. 🛑 스키마 fail-fast — 지금 계약이 바뀌는 중이다

### 4.1 어떻게 강제되는가

`GetServerInfo`가 광고하는 `observation_schema_hash`를 클라이언트가 자기 값과 대조하고,
다르면 즉시 예외를 던진다 (`grpc_actor_transport.py:513-521`,
`rlpd_receive_smoke.py:52-54`). `schema_version` 불일치도 마찬가지다 (`:501-503`).

해시는 스키마 문서의 SHA256이고, 문서에는 **정렬된 상태 피처 이름 목록**이 들어간다.
`json.dumps(sort_keys=True)`는 dict **키**만 정렬하고 **리스트 원소**는 정렬하지 않으므로,
**순서를 바꾸면 해시가 바뀐다 — 그게 의도다** (`observation_schema.py`의 `_schema_document()` 주석).

### 4.2 canonical v2 계약

`serl_ur_infra/ur_env/observation_schema.py`의 v2 계약은 hardware commit `6a0b127`과
learner/hardware merge `248255f`에 통합됐다:

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

라이브 확인:

```bash
cd $WT/serl_ur_infra
python3 - <<'PY'
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

**양쪽(랩톱/Kanu)에서 이 명령을 돌려 해시가 같은지 먼저 확인한 뒤에** 원격 스모크를 시작한다.
해시가 다르면 어떤 통신도 시도하지 말 것 — 이미 fail-fast가 막아주지만,
그 전에 원인을 알고 들어가는 편이 빠르다.

레이아웃 회귀 테스트:

```bash
cd $WT/serl_ur_infra
python3 -m pytest tests/test_state_layout_contract.py tests/test_observation_schema.py \
  -q -p no:anyio
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
| 근거 | `grpc_actor_transport.py:384-386`, `:463-465` | `:24`, `:25` | `:30`, `:31` |

### 5.1 올바른 확인 방법

스모크를 **프로덕션 값으로 다시** 돌린다:

```bash
PYTHONPATH=$WT/serl_ur_infra \
/tmp/gello-hil-grpc-venv/bin/python \
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
  (`grpc_actor_transport.py:735-737`). 오래된 응답은 버린다.
- env 루프는 `HZ = 10.0` (`config.py:32`) → 스텝 주기 100 ms.
  **`timeout_s=0.6`은 이미 6스텝 분량이다.** 여기에 재시도까지 겹치면 1.2 s가 날아간다.
  즉 이 예산은 "여유"가 아니라 이미 상당히 관대한 값이다.

---

## 6. Kanu 터널

서버 gRPC는 **서버 loopback에만** 열어 두고 랩톱에서 포워딩한다.

```bash
# 랩톱
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu
```

- `ExitOnForwardFailure=yes`가 **중요하다.** 없으면 포워딩이 실패해도 ssh가 살아 있어서
  "연결됐는데 왜 안 되지"로 시간을 버린다.
- 양쪽에서 **비어 있는 것이 확인된 포트**를 쓴다.

```bash
# 터널 확인
ss -ltnp | grep 50053
```

서버 기동 (Kanu, `RL_RECEIVE_SERVER.md`의 명령을 워크트리에 맞춘 것):

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/tmp/gello-hil-rl-receive-overlay-v2/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 --port 50053 \
  --checkpoint <classifier checkpoint 경로> \
  --expected-checkpoint-sha256 <sha256> \
  --threshold 0.85 \
  --replay-capacity 50000 --intervention-capacity 10000 \
  --require-jax-backend gpu
```

랩톱에서 수신 스모크:

```bash
PYTHONPATH=$WT/serl_ur_infra \
/tmp/gello-hil-grpc-venv/bin/python \
  $WT/serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50053 --timeout-s 0.6 --max-response-age-s 0.8
```

수신 스모크는 시작하자마자 세 가지를 **fail-fast**로 검사한다 (`rlpd_receive_smoke.py:46-58`):

1. `health()`의 `alive && ready`
2. `observation_schema_hash` 일치
3. `reward_authority == "server_classifier"`

> ### upstream submodule 확인
> 수신 서버는 upstream replay store를 쓰므로 `serl_launcher`가 필요하다.
> `git submodule status third_party/hil-serl`에 `-`가 붙으면 `00_SETUP_AND_SAFETY.md` §2.3에
> 따라 초기화한다. Kanu도 동일한 pinned revision을 사용한다.

---

## 7. 판정 체크리스트

- [ ] `/tmp/gello-hil-grpc-venv`가 만들어졌고 **시스템 python이 오염되지 않았다**
      (`python3 -c "import rclpy"`가 여전히 동작)
- [ ] mock 서버 + 스모크 클라이언트가 루프백에서 통과 (기본값)
- [ ] **같은 스모크가 `--timeout-s 0.6 --max-response-age-s 0.8`에서도 통과** (§5.1)
- [ ] 랩톱과 Kanu의 스키마 해시가 **동일** (§4.2 스크립트)
- [ ] `test_state_layout_contract.py` 통과
- [ ] SSH 터널이 `ExitOnForwardFailure`로 열리고 `ss -ltnp`에 보인다
- [ ] 수신 스모크의 3가지 fail-fast 검사 통과
- [ ] 서버 로그에 이미지/액션 값이 찍히지 않는다 (§2.2)
