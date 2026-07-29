# HIL-SERL gRPC receive server 작업 정리

## 저장 위치

- Git 브랜치: `feat/hil-rl-receive-server`
- 원격 브랜치: `origin/feat/hil-rl-receive-server`
- 로컬 전용 worktree: `/tmp/gello-hil-rl-receive-server`
- Kanu 전용 worktree: `/tmp/gello-hil-rl-receive-server-v2`

worktree는 브랜치를 안전하게 checkout한 작업 디렉터리일 뿐이다. 구현 내용은
원격 브랜치에 커밋되어 있으므로 `/tmp` 디렉터리에만 저장된 상태가 아니다.

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

- `state`: `float32 (1, 19)`
- `cam1`: RGB `uint8 (1, 128, 128, 3)`
- `cam2`: RGB `uint8 (1, 128, 128, 3)`

Schema hash에는 shape/dtype뿐 아니라 19개 state feature의 순서도 들어간다.
순서는 TCP position XYZ, Euler XYZ, linear velocity XYZ, angular velocity XYZ,
force XYZ, torque XYZ, gripper position이다.

Reward는 server classifier가 최종 권한을 가진다.

- `probability > 0.5`: `reward=1`, `done=true`, `mask=0`
- 그 외: `reward=0`
- classifier 성공과 local time-limit truncation이 동시에 발생하면 성공 종료가
  우선한다.
- classifier 또는 buffer 삽입이 실패하면 ACK하지 않고 server가 fail-stop한다.

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

정확한 생성·실행 명령은 [RL_RECEIVE_SERVER.md](./RL_RECEIVE_SERVER.md)에 있다.

## 파이프라인 스모크 결과 (분류 성능 미검증)

Kanu GPU 7과 실제 checkpoint를 사용했다. 아래는 **transport/classifier I/O/buffer
배선이 동작한다는 스모크 결과**이며, classifier의 **분류 성능(recall·FPR)은
검증하지 않았다.** load 성공과 warm-up 시간은 artifact가 열렸다는 사실만 말한다.

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
- 로컬 전체 test suite: 74 passed

검증 후 server와 SSH tunnel은 정상 종료해 GPU 7과 port 50053을 반환했다.
전용 worktree와 overlay는 재실행을 위해 남겨두었다.

## 주요 구현 파일

- `ur_env/proto/actor_transport.proto`: gRPC source of truth
- `ur_env/actor_network.py`: transport-independent session, validation, dedupe
- `ur_env/grpc_actor_transport.py`: gRPC client/server adapter
- `ur_env/remote_actor.py`: laptop actor loop 및 transition 작성
- `ur_env/observation_schema.py`: canonical observation과 schema hash
- `ur_env/rlpd_receive_server.py`: classifier, replay/intervention ingress
- `scripts/run_rlpd_receive_server.py`: Kanu receive server entrypoint
- `scripts/run_rlpd_receive_smoke_client.py`: 100-step acceptance client

## 다음 작업

1. fake zero action을 versioned JAX policy inference로 교체
2. replay/intervention 50:50 sampling을 사용하는 RLPD learner 연결
3. learner parameter를 inference snapshot에 원자적으로 교체
4. checkpoint에 learner step과 policy version 저장
5. 필요하면 ACK 전에 disk append journal을 추가해 transition을 복구 가능하게 함
6. 실제 laptop UR task config, camera contract, reset/fault 및 workspace safety 검증

Learner를 붙이기 전까지 이 브랜치는 receive server milestone의 기준점으로
사용한다.
