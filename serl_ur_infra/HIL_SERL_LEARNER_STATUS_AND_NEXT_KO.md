# HIL-SERL production learner 현황과 다음 작업

> 기준일: 2026-07-27 KST
>
> 문서상 최종 운영 checkout: `/home/laptop3/gello_software`
>
> 문서상 최종 운영 branch: `feat/gello-ur7e-humble-22.04`
>
> learner/hardware 통합 merge: `248255f` (schema v2 검증 및 canonical branch 통합 완료)
>
> Kanu 실행 절차: [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)

## 한눈에 보기

- receive server, 실제 hybrid SAC learner, versioned policy, strict replay ingress, gripper penalty, checkpoint/resume, JSONL/W&B를 **하나의 production CLI**로 조립하는 구현은 learner/hardware 통합 merge `248255f` 계열에 모였다. schema v2 검증과 최종 운영 위치 `/home/laptop3/gello_software`의 `feat/gello-ur7e-humble-22.04` 통합을 완료했다.
- robot actor의 실제 실행 action을 기준으로 `grasp_penalty`를 생성하는 wrapper도 두 actor entrypoint에 배선됐다. learner ingress는 penalty 누락을 허용하지 않는다.
- 실제 `SACAgentHybridSingleArm`을 사용해 CTA update → publish → checkpoint → fresh agent restore → production composition 재조립 → action/RNG/counter 확인 → 추가 update/checkpoint까지 검증했다.
- 실제 reward classifier checkpoint는 로컬에서 SHA 검증, load, warm-up까지 성공했다. annotation-only TensorFlow shim 때문에 Flax가 잘못된 TensorFlow I/O backend를 고르던 문제는 infra-owned local-I/O 설정으로 수정했다.
- fake canonical demo generator가 추가됐다. 이 raw artifact는 construction `--dry-run` 또는 명시적으로 bounded된 `--synthetic-e2e` acceptance에만 허용된다. 일반 robot-data learner serving은 계속 거부한다.
- unified schema v2 기본 suite는 `253 passed, 4 skipped, 6 warnings`, UR/GELLO suite는 `436 passed`다. 실제 frozen-trunk agent `2 passed`, checkpoint/resume `1 passed`, local fake E2E `1 passed`도 다시 통과했다.
- 사용자가 현재 milestone 완료 조건으로 지정한 **fake data laptop→SSH tunnel→Kanu learning E2E**는 schema v2에서 exact 100 ingress→actual classifier→feature replay→CTA update→publish→checkpoint full-load roundtrip까지 통과했다. 새 Kanu process가 checkpoint를 `1/2/1`로 restore하고 policy version 1의 finite 7D action을 serving하는 것도 확인했다. 사용자 요청에 따라 v2 resume process의 불필요한 두 번째 SAC update는 생략했다.
- 아직 실제 robot/task/camera E2E, production 50-step publish/5,000-step checkpoint bounded run, 장시간 GPU contention/latency, 운영 heartbeat는 검증되지 않았다. 따라서 현재 fake-data milestone은 완료됐지만 “실기 운용 승인 완료” 상태는 아니다.
- external policy/classifier는 canonical raw `uint8 (1,128,128,3)` image를 계속 받지만, replay/demo에는 frozen ResNet-10의 `stop_gradient` 직후 camera당 `float32 (1,4,4,512)` map의 current/next만 저장한다. GAP은 적용하지 않고 augmentation은 `none`이다.
- `SpatialLearnedEmbeddings(8) -> Dropout(0.1) -> Dense(256) -> LayerNorm -> tanh`는 동결하지 않았다. learner가 feature batch를 꺼낼 때 현재 weight로 적용하므로 critic/grasp critic CTA update가 유지된다.
- checkpoint만 영속화되고 replay/intervention buffer는 RAM-only다. checkpoint는 덮어쓰기·삭제·자동 pruning을 하지 않는다.

---

## 1. 현재 판정

현재 milestone은 다음처럼 판정한다.

| 범위 | 판정 |
| --- | --- |
| learner 라이브러리 | 구현·자동 검증 |
| receive server + learner 단일 프로세스 composition | 구현·loopback 자동 검증 |
| 실제 agent checkpoint/resume/continue | opt-in 실제 agent 자동 검증 |
| actor gripper penalty wiring | 구현·자동 검증 |
| classifier 실제 checkpoint local restore | 로컬 실제 artifact 검증 및 회귀 테스트 |
| JSONL + 실제 W&B offline artifact | 자동 검증 |
| fake canonical demo | 생성기·strict loader·dry-run/synthetic-E2E scope gate 자동 검증 |
| Kanu GPU production dry-run | actual classifier/agent, feature demo conversion, 128/32 RAM preflight 통과 |
| Kanu GPU feature CTA smoke | 1 learner step/2 gradient step, raw/cached action, trunk invariant 통과 |
| laptop→Kanu fake learning E2E | unified schema v2 fresh actual update/checkpoint + fresh-process resume serving 통과 |
| Kanu GPU continuous learner | 미검증 |
| 실제 robot actor → Kanu learner E2E | 미검증 |
| frozen-trunk feature replay/demo | 구현·자동 검증; Kanu GPU dry-run/CTA smoke 통과 |

즉, 코드의 핵심 경계, 실제 agent state 복원, Kanu GPU construction/CTA, unified schema v2 bounded fake learning/checkpoint/resume serving까지 확인했다. fake-data milestone은 완료다.

## 2. 작업 위치와 branch

### 2.1 최종 운영 위치와 통합 기준점

| 용도 | 위치 | branch/commit | 상태 |
| --- | --- | --- | --- |
| canonical 운영 checkout | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` | 최종 server/learner/local-hardware 통합 위치 |
| learner/hardware 통합 기준점 | 위 canonical branch에 포함 | `248255f` | learner `8f242d8` 계열과 hardware `6a0b127`을 병합; schema v2 검증 완료 |

통합 lineage는 historical base `f0dd3e7` 위의 learner snapshot `8f242d8`, synthetic acceptance 문서 `5322119`, hardware contract fix `6a0b127`을 `248255f`에서 합쳤다. 최종 작업 branch는 `feat/gello-ur7e-humble-22.04` 하나다.

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

## 3. 현재 production data flow

현재 구현은 RAM replay store를 별도 프로세스에서 sampling할 RPC가 없기 때문에, 첫 production topology를 단일 프로세스로 고정한다.

```text
robot laptop
  run_remote_rlpd_actor.py
  task env -> GripperPenaltyWrapper -> episode stats -> timestamp adapter
          |
          | SSH local forwarding / gRPC
          v
Kanu loopback-only learner process
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

CLI는 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부한다. 외부 공개 port 대신 SSH tunnel을 사용한다. `--require-jax-backend`는 필수라 Kanu에서 GPU가 CPU로 조용히 fallback하는 것을 허용하지 않는다.

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

실제 agent checkpoint payload는 약 305 MiB였고 schema v1과 최종 v2 Kanu synthetic E2E에서 `320,100,609 B`를 관측했다. pruning이 없으므로 production 5,000-step checkpoint마다 이 정도가 누적된다고 가정하고 disk를 계획해야 한다. synthetic scope는 검증용으로 period 1이므로 target을 1..10으로 제한한다.

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

resume에서 document 또는 SHA가 다르면 즉시 실패한다. 이전 raw/random-crop checkpoint와 frozen-trunk/no-aug checkpoint는 fingerprint가 다르며 자동 migration하지 않는다. synthetic E2E와 production robot execution scope도 서로 resume하지 않는다. 아직 task identity, exact source commit/upstream revision, actor allowlist까지 모두 묶는 최종 format은 확정 전이다.

### 4.8 reward classifier와 Flax local-I/O 수정

실제 classifier artifact:

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

실제 local artifact의 load/warm-up이 성공했다. 별도 측정에서 전체 준비 약 7.89초, warm-up 약 674 ms였고 zero-image smoke probability는 약 `0.081304`였다. 이 수치는 GPU production latency 기준이 아니라 local acceptance 관측값이다.

### 4.9 logging, W&B, protobuf

- JSONL은 항상 local path에 structured event를 기록한다.
- W&B 기본 mode는 `offline`이다.
- W&B `0.26.0`을 pin한다.
- 실제 W&B offline run directory 생성과 checked-in protobuf module 동시 import를 자동 테스트한다.
- protobuf 7.34.1 process에서는 generated binding import 전에 pure-Python compatibility mode를 선택한다.
- W&B online login/upload는 사용자가 credential을 제공한 뒤 별도 1회 검증한다.

실제 W&B offline artifact test와 protobuf co-import test는 존재한다. 장시간 learner/classifier/gRPC 동시 부하에서의 logging 안정성은 아직 Kanu soak 범위다.

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

### 5.1 unified schema v2 최종 회귀

learner/hardware 통합 merge `248255f`에서 확인한 최종 결과는 다음과 같다.

```text
serl_ur_infra: 253 passed, 4 skipped, 6 warnings in 3.10s
ur_gello_bringup: 436 passed in 7.35s
```

skip은 JAX 비용이 큰 opt-in 실제 agent/E2E 경로다. 같은 unified tree에서 실제 frozen-feature agent `2 passed in 22.66s`, 실제 checkpoint `1 passed in 33.61s`, 실제 localhost fake E2E `1 passed in 32.77s`를 별도 실행했다. 실패는 0이었다. UR/GELLO suite는 system pytest/anyio plugin 충돌을 피하기 위해 문서대로 `-p no:anyio`를 사용했다.

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

처음 사용한 `kanu_junhyeong`은 등록되지 않은 이름이었고 실제 SSH alias는 `kanu`였다. read-only preflight에서 RAM 251 GiB(available 152 GiB), RTX A4000 16 GB 8장과 `/home/junhyeong/miniconda3/envs/il/bin/python`의 JAX/JAXLIB 0.5.3, Flax 0.10.5, backend `gpu`, device 8개를 확인했다.

Kanu의 기존 repository가 dirty detached 상태였으므로 그 workspace를 수정하지 않고 `/tmp/hil-feature-dryrun-BUJNWu`에 rsync/symlink로 일회성 검증 tree를 구성했다. `CUDA_VISIBLE_DEVICES=0`으로 실제 classifier + agent production dry-run(128/32 capacity)이 통과했고 당시 pre-execution-scope fingerprint는 `8465e464b3f4eb638513eaa4ab3daea85a9435a9ddf47bdbf841a2c2f2aacce9`였다. 별도 실제 GPU feature CTA smoke는 backend `gpu`, device 1개, feature `[1,4,4,512]`, `gradient_step=2`, `augmentation_function=None`(JSON `null`)을 확인했다. raw/cached deterministic action의 max absolute difference는 `0.00012614415027201176`였고 online/target trunk invariant도 통과했다. 5.5의 `defda67...`는 pre-hardware schema v1 interim이고, unified schema v2 authoritative fingerprint는 아래 `fa198537...`다. 어느 결과도 continuous server/robot E2E로 확대 해석하지 않는다.

### 5.5 laptop→Kanu 실제 fake-data learning E2E (schema v1 interim)

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

1. **production lifecycle bounded/continuous GPU acceptance**
   - JAX/JAXLIB 0.5.3 GPU production dry-run, 단일 feature CTA, laptop→Kanu synthetic E2E step 1/resume step 2는 통과했다.
   - production 기본 50-step publish + 5,000-step checkpoint는 아직 미검증이다.
   - 기본 ring camera map `7,864,320,000 B` + offline demo + reserve 할당 후 장시간 RSS/VMS/GPU memory/compile/contention을 계측한다.
   - CPU용 `requirements-learner.lock`을 공유 Kanu env에 그대로 설치하지 않는다.

2. **real serving용 canonical robot demo 부재**
   - 사용자가 지정한 현재 fake-data acceptance는 완료됐다.
   - fake는 bounded `--synthetic-e2e`에서만 live learner에 허용되고 production robot-data scope에는 의도적으로 차단된다.
   - 실제 robot serving 단계에는 canonical EEF/action/grasp-penalty demo가 필요하다.

3. **실제 robot/task/camera E2E**
   - task config의 `GRASP_PENALTY`, action convention, camera key/shape/timing을 확인한다.
   - robot laptop → SSH tunnel → Kanu → classifier/replay → policy response를 검증한다.

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

1. 실제 canonical robot demo가 준비되면 production-scope 5,000-step bounded learner run을 수행한다.
2. 장시간 RSS/VMS/GPU memory/compile/contention과 W&B offline disk 증가를 계측한다.
3. task/camera/safety review 후 실제 robot actor를 연결한다.
4. learner fault heartbeat와 shutdown escalation을 보강한 뒤 continuous mode를 승인한다.

## 9. 재현 명령

상세 Kanu 명령은 [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)에 있다.

로컬 빠른 suite의 권장 형태는 다음과 같다.

```bash
cd /home/laptop3/gello_software

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests
```

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
- Kanu에서는 GPU backend를 명시하고 CPU fallback을 허용하지 않는다.
- `third_party/hil-serl`, proto, generated binding을 직접 수정하지 않는다.
- checkpoint를 덮어쓰기·삭제·pruning하지 않는다.
- resume는 fingerprint/counter/RNG 검사를 우회하지 않는다.
- learner fault가 나도 last-known-good policy를 오염시키지 않는다.
- replay가 RAM-only라는 사실을 checkpoint resume와 혼동하지 않는다.
- replay/demo에 raw image, GAP512, 또는 trainable 256-D head 출력을 저장하지 않는다.
- verified frozen trunk 직후 `float32 (1,4,4,512)` current/next 계약과 augmentation `none`을 유지한다.
- online/target trunk invariant 검사와 target repin을 우회하지 않는다.
- raw/random-crop checkpoint를 feature/no-aug lineage에 resume하지 않는다.
