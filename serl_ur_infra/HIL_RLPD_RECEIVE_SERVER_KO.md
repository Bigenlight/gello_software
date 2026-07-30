# HIL-SERL gRPC receive server 작업 정리

> **상태: 과거 milestone 기록.** 2026-07-29에 머지된 트리 기준으로 재검증하면서
> 아래 세 가지를 정정했다 — 브랜치/worktree 위치, 19-D `state` 순서(v1 순서로
> 적혀 있었고 그건 틀렸다), reward threshold(문서에 숫자를 적지 않는다).
> Kanu에는 2026-07-29 현재 이 서버가 떠 있지 않다(port 50053 미바인딩, GPU 유휴).

## 저장 위치 (2026-07-29 정정)

- 통합 브랜치: **`feat/gello-ur7e-humble-22.04`** (= `origin/HEAD`). 이 milestone의
  코드는 여기에 병합돼 있다.
- ~~`feat/hil-rl-receive-server` / `origin/feat/hil-rl-receive-server`~~ —
  **더 이상 존재하지 않는다.** 로컬·원격 어디에도 없다(2026-07-29 `git branch -a` 확인).
  이 이름으로 checkout하려 하면 실패한다.
- `/tmp/gello-hil-rl-receive-server` (laptop3), `/tmp/gello-hil-rl-receive-server-v2`
  (Kanu 전용 worktree): `/tmp`이므로 **남아 있다고 가정하지 않는다.** laptop3에서는
  이미 사라졌다(2026-07-29 확인). 필요하면 통합 브랜치에서 새로 만든다.

구현 내용은 통합 브랜치에 커밋되어 있으므로 `/tmp` 디렉터리에만 저장된 상태가 아니다.

## 이번 단계의 목표와 범위

로봇이 연결된 laptop에서 observation과 transition을 보내고, Kanu server가
다음을 수행하는 receive-only HIL-SERL 경로를 구현했다.

1. 실제 reward classifier 추론
2. server-authoritative reward 및 episode 종료 판정
3. 전체 transition을 replay buffer에 삽입
4. `intervened=1` transition을 intervention buffer에도 삽입
5. 다음 action과 방금 transition의 최종 outcome을 laptop에 반환

현재 action은 안전한 `float32[7]` zero action이다. 실제 RL learner, 학습된
policy inference, parameter update와 실제 로봇 제어는 이번 범위에 포함되지
않는다.

## 통신 흐름

### Episode 시작

```text
Laptop -> Server: BeginEpisode(O0)
Server -> Laptop: A0
```

### 매 step

```text
Laptop -> Server: Step(data{meta, transition}, O(t+1))
Server -> Laptop: ACK + TransitionOutcome + optional A(t+1)
```

Server는 session에 저장한 `O(t)`와 이번 요청의 `O(t+1)`을 사용해 replay용
`observations`와 `next_observations`를 만든다. `O(t+1)`은 classifier,
replay 삽입, 다음 policy action 계산에 함께 사용하므로 이미지를 같은 step에
두 번 보내지 않는다.

최종 transition이 `done` 또는 `truncated`이면 다음 action을 반환하지 않고
laptop이 reset한다. 요청 ID와 transition content fingerprint를 이용해 같은
요청의 재시도는 중복 삽입하지 않는다.

## 데이터 계약

개념적인 전송 단위는 다음과 같다.

```text
data
├── meta
│   ├── schema_version
│   ├── run_id / actor_id / session_id
│   ├── transition_id
│   ├── env_step
│   ├── timestamp_ns
│   ├── policy_version
│   ├── policy_action
│   └── intervened                 # 0 또는 1
└── transition
    ├── episode_id / step_id
    ├── observation_id / next_observation_id
    ├── actions                    # 실제 실행된 action
    ├── rewards / masks
    ├── dones / truncated
    └── grasp_penalty              # 존재할 때만
```

- `meta.policy_action`: 개입 전 server policy가 제안한 action
- `transition.actions`: 실제 로봇에 실행된 action
- `meta.intervened=1`: 두 action이 human intervention으로 구분되는 표식
- `meta.timestamp_ns`: policy state에 섞지 않는 environment 시각
- `meta.env_step`: 전체 run에서 증가하는 transition 번호
- `transition.episode_id`: reset될 때 증가하는 episode 번호

Server가 observation을 materialize하고 classifier 결과를 반영한 뒤 실제 RAM
buffer에 넣는다. 문자열 ID는 학습 tensor에 섞지 않고 bounded sidecar metadata로
보관한다.

## Observation 및 reward 계약

Canonical observation은 다음 세 tensor로 고정돼 있다.

```text
schema id:   hil-serl-ur-canonical-observation-v2
schema hash: 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

- `state`: `float32 (1, 19)`
- `cam1`: RGB `uint8 (1, 128, 128, 3)`
- `cam2`: RGB `uint8 (1, 128, 128, 3)`

Schema hash에는 shape/dtype뿐 아니라 19개 state feature의 **순서**도 들어간다.

> 🪤 **2026-07-29 정정.** 이 문서는 순서를 "TCP position XYZ, Euler XYZ,
> linear velocity XYZ, angular velocity XYZ, force XYZ, torque XYZ, gripper
> position"이라고 적고 있었다. **그건 v1 순서이고 지금 계약과 다르다.**
> flat layout은 upstream `SERLObsWrapper`가 만들고 `gym.spaces.Dict`가 proprio
> 그룹을 **알파벳 순으로 재정렬**하므로 실제 순서는 다음과 같다.

```text
[0]     gripper_pose   gripper_position          <- gripper는 index 0. -1이 아니다
[1:4]   tcp_force      x, y, z
[4:10]  tcp_pose       position x,y,z + euler x,y,z
[10:13] tcp_torque     x, y, z
[13:19] tcp_vel        linear x,y,z + angular x,y,z
```

`state[0, -1]`은 gripper가 아니라 TCP angular velocity z다. gripper는
`GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`로만 읽는다.
shape `(1,19)`가 같다고 v1 peer와 v2 peer가 호환되는 것이 아니다.

hash를 손으로 옮겨 적지 않는다. 코드에서 다시 뽑는다(2026-07-29에 위 값과
일치함을 확인):

```bash
PYTHONPATH=serl_ur_infra python3 -c \
  "from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH as h; print(h)"
```

Reward는 server classifier가 최종 권한을 가진다.

- `probability > threshold`: `reward=1`, `done=true`, `mask=0` (엄격한 초과 비교)
- 그 외: `reward=0`
- classifier 성공과 local time-limit truncation이 동시에 발생하면 성공 종료가
  우선한다.
- classifier 또는 buffer 삽입이 실패하면 ACK하지 않고 server가 fail-stop한다.

> ⚠️ **threshold 숫자를 이 문서에서 베끼지 마라.** 이틀 사이 0.85 → 0.5 → 0.2로
> 두 번 움직였다. 권위 있는 값은 `ur_env/rlpd_receive_server.py`의
> `DEFAULT_REWARD_THRESHOLD` 하나뿐이고, 그게 `--threshold`의 기본값이다.
> 그래서 `--threshold`를 아예 주지 않는 것이 코드와 어긋나지 않는 유일한 방법이다.
> 2026-07-29 확인 시점 값은 `0.2`였다. 근거는
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md).

## Buffer와 ACK 의미

- 모든 transition은 replay buffer로 간다.
- `intervened=1` transition은 replay와 intervention buffer 양쪽으로 간다.
- intervention buffer 삽입까지 성공해야 해당 transition을 ACK한다.
- ACK 기준은 현재 RAM 삽입이다. Disk journal은 아직 없다.
- server를 재시작하면 저장된 transition은 사라진다.

Upstream HIL-SERL의 `MemoryEfficientReplayBufferDataStore`를 실제로 사용한다.
설정한 capacity는 bootstrap frame slot이 아니라 sample 가능한 논리 transition
개수를 뜻하도록 보정돼 있다.

## Kanu 실행 환경

Docker는 사용하지 않았다. 공유 conda `il` 환경도 변경하지 않았다.

- base Python: `/home/junhyeong/miniconda3/envs/il/bin/python`
- overlay venv: `/tmp/gello-hil-rl-receive-overlay-v2`
- overlay 추가 용량: 약 33 MB
- overlay 전용 패키지: Agentlace, LZ4, protobuf 3.20.3

protobuf 3.20.3은 저장소에 체크인된 generated gRPC module과 Kanu의 protobuf
7.x가 호환되지 않아 overlay 안에서만 사용한다.

> ⚠️ **이 overlay를 production learner에 재사용하지 않는다.** protobuf 3.20.3 핀이
> `run_rlpd_learner_server.py`가 요구하는 `wandb` import를 깨뜨린다. learner는
> 별도의 CUDA 환경을 쓴다 —
> [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) 1.2절.

> ℹ️ Kanu `il` 환경은 lock에서 벗어나 있다(2026-07-29 팩트 시트 기준:
> numpy 2.2.5 / orbax 0.11.12 / grpcio 1.80.0 vs lock 1.26.4 / 0.11.5 / 1.74.0).
> 런타임 fail-closed 대상이 jax·flax·distrax·tfp·wandb뿐이라 이 셋은 자동으로
> 걸리지 않는다. overlay를 다시 만들 때 실제 값을 기록해 둔다.

정확한 생성·실행 명령은 [RL_RECEIVE_SERVER.md](./RL_RECEIVE_SERVER.md)에 있다.

## 파이프라인 스모크 결과 — 2026-07-27 기록 (분류 성능 미검증, 재입력 금지)

> 📌 **아래는 그날 나온 값의 기록이다.** GPU 번호·포트 점유·타이밍은 그때의 상황이지
> 고정값이 아니고, 지금 Kanu에서는 아무것도 돌고 있지 않다. **checkpoint 경로와 SHA를
> 여기서 복사해 새 run에 넣지 마라.**

그날 비어 있던 GPU(당시 index 7)와 당시 checkpoint를 사용했다. 아래는
**transport/classifier I/O/buffer 배선이 동작한다는 스모크 결과**이며,
classifier의 **분류 성능(recall·FPR)은 검증하지 않았다.** load 성공과 warm-up
시간은 artifact가 열렸다는 사실만 말한다.

> ⚠️ 여기서 쓴 checkpoint `e329986b...`는 이후 **폐기됐다.** 2026-07-28 Kanu
> 실측에서 0724 도메인 success recall이 `0.0%`였다(성공 1,123 프레임 중 0건,
> mean 확률 `0.007`). 아래 기록은 당시 실측 그대로 남기되, 이 checkpoint를
> 신규 run에 사용하지 않는다. 측정 근거는
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md),
> 새 정본 경로와 orbax 디렉터리 제약은
> [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) 1.3절에 있다.

- checkpoint:
  `/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150`
  (폐기)
- SHA-256:
  `e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997`
  (폐기)
- JAX backend: `gpu`
- classifier load 및 JIT warm-up: 성공, 약 1.65초
- laptop -> SSH tunnel -> Kanu gRPC synthetic steps: 100
- replay inserts: 100
- intervention inserts: 10
- replay와 intervention 실제 batch sampling: 성공
- 검증 batch image shape: `(8, 2, 128, 128, 3)`
- 검증 batch state shape: `(8, 1, 19)`
- 로컬 전체 test suite: 74 passed — **이 milestone 시점의 개수다.** 통합 브랜치의
  현재 기준선은 `serl_ur_infra/tests` 전체에서 **429 passed / 11 skipped**다
  (2026-07-30 laptop3 실행 확인, actor venv 기준). 74와 비교해 회귀 여부를 판단하지 않는다.
  이력: `333`(07-29) → `337`(`40b99f8`) → **`429`**(분류기 사이드카). 옛 판에 적힌 333을
  현재 기준선으로 인용하지 말 것.
  **🪤 인터프리터에 따라 수가 달라진다** — `/home/laptop3/venvs/hilserl/bin/python`으로
  돌리면 jax가 있어 **464 passed / 4 skipped**다(35개가 더 실행된다). 위 429는
  `gello-hil-actor` venv 기준이고, 어느 쪽이든 **skipped 수가 기준선과 다르면
  PYTHONPATH를 잘못 준 것**이다.

검증 후 server와 SSH tunnel은 정상 종료해 그날 쓰던 GPU와 port 50053을 반환했다.
당시 전용 worktree와 overlay를 재실행용으로 남겨두었지만 둘 다 `/tmp`이므로
지금 남아 있다고 가정하지 않는다.

## 주요 구현 파일

- `proto/actor_transport.proto`: gRPC source of truth (**경로 정정 2026-07-29** —
  `ur_env/proto/`에는 생성물 `actor_transport_pb2*.py`만 있다)
- `ur_env/actor_network.py`: transport-independent session, validation, dedupe
- `ur_env/grpc_actor_transport.py`: gRPC client/server adapter
- `ur_env/remote_actor.py`: laptop actor loop 및 transition 작성
- `ur_env/observation_schema.py`: canonical observation과 schema hash
- `ur_env/rlpd_receive_server.py`: classifier, replay/intervention ingress
- `scripts/run_rlpd_receive_server.py`: Kanu receive server entrypoint
- `scripts/run_rlpd_receive_smoke_client.py`: 100-step acceptance client

## 다음 작업 — 2026-07-29 갱신

이 절의 1~4번은 그 뒤에 **구현됐다.** `scripts/run_rlpd_learner_server.py`가
versioned JAX policy inference, replay/intervention 50:50 RLPD learner,
policy snapshot 원자 교체, learner step/policy version을 담은 checkpoint를 모두
갖고 있다. 그 절차는 이 문서가 아니라
[HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)에 있다.

~~1. fake zero action을 versioned JAX policy inference로 교체~~ (완료)
~~2. replay/intervention 50:50 sampling을 사용하는 RLPD learner 연결~~ (완료)
~~3. learner parameter를 inference snapshot에 원자적으로 교체~~ (완료)
~~4. checkpoint에 learner step과 policy version 저장~~ (완료)

남은 것:

5. ACK 전 disk append journal — **미구현.** replay/intervention은 여전히 RAM-only이고
   process를 재시작하면 내용이 사라진다.
6. 실제 laptop UR task config는 `ur_experiments/cube_in_cup.py`로 존재하지만,
   camera contract·reset/fault·workspace safety는 **실기 미검증**이다.
   `scripts/run_remote_rlpd_actor.py`는 실기에서 한 번도 실행된 적이 없다.
7. 정본 orbax classifier를 이 서버에 물릴 수 없다 — `checkpoint_sha256()`이
   `os.path.isfile()`을 강제한다. 디렉터리 digest 계약이 선행돼야 한다.

이 문서는 receive-only milestone의 기준점 기록으로 남긴다. 새 작업의 출발점은
통합 브랜치와 learner runbook이다.
