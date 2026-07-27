# HIL-SERL production learner 현황과 다음 작업

> 기준일: 2026-07-27 KST
>
> 현재 구현 worktree: `/home/laptop3/gello_worktrees/hil-production-learner`
>
> 현재 구현 branch: `feat/hil-production-learner`
>
> 분기 기준 commit: `f0dd3e7deb6f96a51889b6e9d9d41812d34b26b8`
>
> Kanu 실행 절차: [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)

## 한눈에 보기

- receive server, 실제 hybrid SAC learner, versioned policy, strict replay ingress, gripper penalty, checkpoint/resume, JSONL/W&B를 **하나의 production CLI**로 조립하는 코드는 현재 `feat/hil-production-learner` worktree에 구현돼 있다.
- robot actor의 실제 실행 action을 기준으로 `grasp_penalty`를 생성하는 wrapper도 두 actor entrypoint에 배선됐다. learner ingress는 penalty 누락을 허용하지 않는다.
- 실제 `SACAgentHybridSingleArm`을 사용해 CTA update → publish → checkpoint → fresh agent restore → production composition 재조립 → action/RNG/counter 확인 → 추가 update/checkpoint까지 검증했다.
- 실제 reward classifier checkpoint는 로컬에서 SHA 검증, load, warm-up까지 성공했다. annotation-only TensorFlow shim 때문에 Flax가 잘못된 TensorFlow I/O backend를 고르던 문제는 infra-owned local-I/O 설정으로 수정했다.
- fake canonical demo generator가 추가됐다. 이 artifact는 **acceptance `--dry-run` 전용**이며 production serving에는 사용할 수 없도록 CLI가 거부한다.
- current hardening 뒤 fresh process 기본 전체 suite는 `158 passed, 1 skipped, 6 warnings in 2.59s`다. skip 1개인 실제 agent checkpoint/resume test도 별도로 opt-in 실행해 통과했다.
- 아직 Kanu GPU 실제 learner run, 실제 robot/task/camera E2E, 장시간 GPU contention/latency, 운영 heartbeat가 검증되지 않았다. 따라서 “production-shaped 코드”는 맞지만 “실기 운용 승인 완료” 상태는 아니다.
- 현재 replay는 raw image와 random-crop augmentation을 사용한다. 사용자가 요청한 기본 no-augmentation feature buffer는 아직 구현되지 않았다. **GAP512 cut point와 전체 freeze 계약을 먼저 확정해야 한다.**
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
| fake canonical demo | 생성기·strict loader·dry-run-only gate 자동 검증 |
| Kanu GPU dry-run | 미검증 |
| Kanu GPU continuous learner | 미검증 |
| 실제 robot actor → Kanu learner E2E | 미검증 |
| feature/GAP512 replay | 설계 결정 대기, 미구현 |

즉, 코드의 핵심 경계와 실제 agent state 복원은 확인했지만 GPU 서버와 로봇을 포함한 acceptance gate는 아직 통과하지 않았다.

## 2. 작업 위치와 branch

### 2.1 현재 작업 위치

| 용도 | 위치 | branch | 상태 |
| --- | --- | --- | --- |
| production learner 구현 | `/home/laptop3/gello_worktrees/hil-production-learner` | `feat/hil-production-learner` | 이 문서와 현재 미커밋 구현의 작업 위치 |
| canonical 통합 workspace | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` | 현재 구현 검증 후 fast-forward/cherry-pick 대상. 다른 세션 변경을 임의로 덮어쓰면 안 됨 |
| hardware 통신 검증 | `/home/laptop3/gello_worktrees/hil-hardware-comms` | `test/hil-hardware-comms` | 별도 worktree. learner 문서 작업 대상 아님 |

세 branch는 현재 `f0dd3e7`에서 분기돼 있다. production learner 변경은 아직 별도 commit으로 고정되지 않았으므로, commit 전에는 worktree 경로 자체가 handoff 기준이다.

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
- canonical workspace나 다른 worktree의 dirty 변경을 learner 작업으로 흡수하지 않는다.

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
  RewardClassifierRuntime
          |
          v
  ActorSessionService
          |
          v
  FaultGatedReplayIngress
       |               |
       v               v
  online replay   online intervention pool
       |               |
       +-------+-------+
               |
  canonical offline demo pool
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
- 실제 `SACAgentHybridSingleArm` 생성
- fresh/resume checkpoint state 준비
- learner-mode `ReplayIngress`와 fault gate
- offline demo pool과 RLPD sampler
- versioned policy runtime과 actor service
- loopback-only gRPC server
- 정확히 한 개의 non-daemon learner worker
- JSONL과 W&B logger
- signal/shutdown 처리

CLI는 `127.0.0.1`, `localhost`, `::1` 외 bind를 거부한다. 외부 공개 port 대신 SSH tunnel을 사용한다. `--require-jax-backend`는 필수라 Kanu에서 GPU가 CPU로 조용히 fallback하는 것을 허용하지 않는다.

fresh run은 checkpoint root 아래에 기존 `checkpoint_*` entry가 있으면 거부한다. resume는 counters, CTA ratio, publish/checkpoint boundary, inference RNG, fingerprint를 조립 전에 검사한다. initial policy는 step 0/version 0부터 actor service와 learner가 동일한 parameter reference를 공유한다.

robot actor는 expected policy model ID, reward authority, reward model ID, observation schema hash를 pin할 수 있다. pin이 하나라도 설정되면 episode 시작마다 `GetServerInfo`를 새로 조회하고 mismatch를 첫 inference 전에 거부한다. 현재 production 값은 policy `hil-serl-hybrid-sac-resnet10`, reward authority `server_classifier`이며 reward model ID는 server CLI에 준 값이다.

### 4.2 학습 계약

- agent: upstream `SACAgentHybridSingleArm`
- observation: `state float32 (1,19)`, `cam1/cam2 uint8 (1,128,128,3)`
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

위의 external canonical observation shape와 replay sample의 packed layout은 구분한다. memory-efficient replay가 내놓는 raw batch에서 `observations.cam*`은 current/next frame을 묶은 `(B,2,128,128,3)`이고 `next_observations`에는 state만 있다. sampler가 upstream `pack_batch()`로 agent update용 current/next observation을 복원한다.

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

실제 agent checkpoint payload는 로컬 측정에서 약 305 MiB였다. pruning이 없으므로 5,000-step마다 이 정도가 누적된다고 가정하고 disk를 계획해야 한다.

### 4.7 fingerprint

현재 fingerprint는 observation schema, learner algorithm config, ResNet SHA에 더해 아래 production run contract를 포함한다.

- raw-pixel hybrid SAC contract revision
- augmentation mode
- action dtype/shape/range/gripper values
- reward classifier SHA/threshold/model ID
- offline demo artifact SHA 목록과 transition 수
- exact grasp-penalty allowed values `[0, configured penalty]`
- JAX/JAXLIB/Flax/Distrax/TFP version

resume에서 document 또는 SHA가 다르면 즉시 실패한다. 아직 task identity, exact source commit/upstream revision, actor allowlist까지 모두 묶는 최종 format은 확정 전이다.

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

`scripts/generate_fake_canonical_demo.py`는 deterministic canonical transition 두 개를 pickle로 만든다. strict loader를 즉시 다시 통과시키고 SHA, transition 수, synthetic marker를 JSON으로 출력한다.

모든 item은 `synthetic_acceptance_only=true` provenance를 가진다. 생성기는 기존 파일을 덮어쓰지 않는다. production CLI는 이 marker를 발견하면 `--dry-run`이 아닌 실행을 거부한다.

fake artifact로 확인할 수 있는 것:

- canonical loader/schema
- classifier/agent construction
- dependency/backend/asset preflight
- fingerprint 생성
- production composition construction
- JSONL/W&B offline initialization

fake `--dry-run`으로 확인할 수 없는 것:

- gRPC bind와 SSH transport
- replay 100개 도달
- learner CTA update
- 50-step publish
- 5,000-step checkpoint
- 실제 robot distribution의 reward/action 품질

이 문서는 “실제 demo artifact는 fake로 하자”는 사용자 결정을 construction acceptance용 fake artifact로 해석한다. fake data로 Kanu gRPC/CTA/publish/checkpoint까지 실행하려는 뜻이라면 현재 dry-run-only gate와 충돌하므로, production gate를 제거하지 말고 별도의 명시적 synthetic-training acceptance mode를 먼저 설계해야 한다.

## 5. 검증 현황

### 5.1 현재 확정 회귀 기준

current hardening 완료 뒤 기본 전체 suite 결과는 다음과 같다.

```text
158 passed, 1 skipped, 6 warnings in 2.59s
```

skip 1개는 opt-in 실제 hybrid SAC checkpoint test다. 기본 suite에는 production composition, gripper wiring, exact penalty contract, ingress fault gate, classifier local-I/O, actor server-identity pinning, 실제 W&B offline/protobuf smoke가 포함된다.

역사적으로 hardening 전 확정 기준은 `133 passed, 1 skipped`였다. fingerprint, classifier, exact penalty, actor identity 테스트가 추가되며 현재 숫자로 증가했다. 문서 검수 중 demo SHA preflight와 mock fixture가 잠시 불일치해 transient failure가 있었지만 fixture를 실제 파일 contract에 맞춘 뒤 위 최종 suite가 통과했다.

### 5.2 opt-in 실제 agent checkpoint/resume

`tests/test_actual_agent_checkpoint_integration.py`는 실제 upstream `SACAgentHybridSingleArm`을 생성해 다음을 확인한다.

1. synthetic raw-image batch로 CTA update
2. publish와 checkpoint 저장
3. fresh real-agent template 생성
4. checkpoint restore
5. `prepare_learner_state()`와 `compose_learner()`로 production 객체 재조립
6. train-state leaf, counters, agent/inference RNG 확인
7. deterministic/stochastic action exact 비교
8. resume learner에서 추가 CTA update/publish/checkpoint

비용을 줄이기 위해 test config의 publish/checkpoint period는 1이다. 따라서 boundary 구현을 검증하지만 default 50/5,000 장시간 run을 대체하지는 않는다.

opt-in 환경 변수는 `RUN_HIL_SERL_ACTUAL_CHECKPOINT=1`이다. current production-composition hardening 뒤 CPU JAX 0.5.3에서 다시 실행해 `1 passed, 92 warnings in 43.35s`를 확인했다.

### 5.3 production dry-run과 실제 classifier

실제 classifier checkpoint와 생성한 fake canonical demo를 사용한 local production CLI dry-run이 current hardening 뒤 다시 통과했다.

- event: `rlpd_learner_dry_run_passed`
- artifact root: `/tmp/hil-production-dryrun-mmXlf2`
- backend: CPU
- W&B mode: `disabled`
- fake demo SHA-256: `6907f4e458e87001bc26dc3d9d8e9b7a4c5ae2ac1ee637bd56ce0e82372c9177`
- ResNet cache SHA-256: `175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b`
- production fingerprint: `add5c7f0247d4efd7c95fd492f1826ce3770f5fc61f35d3b12805b4aa8c86009`
- ready state: learner 0, gradient 0, policy version 0, synthetic demo 2개

이전 pre-hardening dry-run에서는 wall time 약 15.03초와 peak RSS 약 1.51 GiB를 관측했다. 최신 hardening run에서는 이 timing/RSS를 다시 계측하지 않았으므로 역사적 참고값으로만 남긴다.

이 결과는 production construction의 실제 dependency/artifact smoke다. 최신 run은 W&B를 disabled로 실행했으므로 실제 W&B offline artifact 검증은 별도 자동 테스트가 근거다. server bind나 training을 실행하지 않으므로 Kanu/robot E2E 결과로 확대 해석하지 않는다.

처음 사용한 `kanu_junhyeong`은 등록되지 않은 이름이었고 실제 SSH alias는 `kanu`였다. 수정 후 read-only 접속에 성공해 RAM 251 GiB(available 152 GiB), RTX A4000 16 GB 8장과 `/home/junhyeong/miniconda3/envs/il/bin/python`의 JAX/JAXLIB 0.5.3, Flax 0.10.5, backend `gpu`, device 8개를 확인했다. Kanu filesystem이나 process는 변경하지 않았다. 실제 production learner GPU dry-run은 아직 미실행이다.

## 6. 현재 raw replay의 메모리와 visual encoder 사실관계

### 6.1 현재 기본값

현재 CLI 기본 capacity는 다음과 같다.

- replay: 50,000 logical transitions
- intervention: 10,000 logical transitions

upstream memory-efficient buffer의 frame/bootstrap storage까지 고려한 physical camera slots는 대략 replay 100,000 + intervention 20,000이다. 두 카메라 `128x128x3 uint8` raw arrays만 약 10.99 GiB다. Python object, state/action/metadata, JAX/XLA, classifier/learner model memory는 별도다.

따라서 raw mode 기본 capacity를 Kanu에서 그대로 쓰려면 host RAM과 실제 RSS를 먼저 계측해야 한다. RAM buffer는 process 종료 시 사라진다.

### 6.2 upstream encoder는 어디까지 frozen인가

“pretrained image encoder를 가져와 얼린다”는 방향은 맞지만, 현재 upstream의 **전체 visual path가 frozen인 것은 아니다.**

```text
uint8 image
  -> ImageNet normalization
  -> pretrained ResNet-10 backbone
  -> 4x4x512 map                     [stop_gradient; frozen]
  -> SpatialLearnedEmbeddings
  -> Dropout(0.1)
  -> Dense(256) -> LayerNorm -> tanh [critic/grasp critic에서 trainable]
```

backbone map까지는 frozen이지만 뒤의 spatial/bottleneck head는 학습된다. actor는 encoder output에 stop-gradient를 쓰지만 critic/grasp critic은 head를 갱신한다. 현재 raw learner는 padding 4 random crop을 update마다 수행한다.

### 6.3 pending GAP512 결정

메모리 절감을 위한 현재 우선 후보는 frozen `4x4x512` map에 deterministic global-average pooling을 적용한 camera당 `512 float32` feature다.

```text
raw image -> frozen ResNet-10 -> 4x4x512 -> GAP -> 512-D
```

두 camera의 GAP512는 observation당 4 KiB이므로 raw 두 camera 96 KiB보다 약 24배 작다. physical 120,000 slot의 feature arrays는 단순 계산으로 약 469 MiB다. 실제 buffer RSS는 구현 후 계측해야 한다.

이 선택은 단순 storage refactor가 아니다.

- pixel random crop을 제거하고 기본 no-augmentation으로 바꾼다.
- 기존 trainable spatial/Dense visual head를 제거하거나 별도 policy head로 재정의한다.
- offline demo는 같은 pinned encoder/backend로 한 번 encode해야 한다.
- online ingress는 classifier가 raw image를 먼저 사용한 뒤 feature만 replay에 commit해야 한다.
- encoder SHA/tree/schema/backend를 buffer와 checkpoint fingerprint에 넣어야 한다.
- raw checkpoint와 feature checkpoint를 자동 호환하지 않는다.

GAP512, frozen `4x4x512` map 저장, 또는 전체 256-D head까지 동결하는 방식 중 하나를 명시적으로 승인하기 전에는 feature implementation을 시작하지 않는다. 현재 논의의 우선안은 **no-aug + immutable GAP512**지만 최종 결정은 아직 아니다.

## 7. 남은 차단점

### P0 — 다음 구현 전 결정/검증 필수

1. **GAP512 feature boundary 승인**
   - 정확한 cut point, pooling, dtype, no-augmentation, offline migration, fingerprint를 확정해야 한다.
   - 이 결정이 raw replay schema, agent network, checkpoint format, memory budget을 모두 바꾼다.

2. **Kanu JAX 0.5.3 GPU environment**
   - SSH alias `kanu`와 기존 `il` environment에서 JAX/JAXLIB 0.5.3, Flax 0.10.5, GPU backend를 read-only 확인했다.
   - 실제 learner command에서도 `jax.default_backend() == "gpu"`를 fail-closed로 다시 확인한다.
   - CPU용 `requirements-learner.lock`을 공유 Kanu env에 그대로 설치하지 않는다.
   - classifier + policy inference + learner update의 GPU memory/compile/contention을 계측한다.

3. **real serving용 canonical robot demo 부재**
   - 사용자가 이번 acceptance artifact는 fake로 하기로 했다.
   - fake는 dry-run에 충분하지만 live learner serving에는 의도적으로 차단된다.
   - 실제 robot serving 단계에는 canonical EEF/action/grasp-penalty demo가 필요하다.

4. **실제 robot/task/camera E2E**
   - task config의 `GRASP_PENALTY`, action convention, camera key/shape/timing을 확인한다.
   - robot laptop → SSH tunnel → Kanu → classifier/replay → policy response를 검증한다.

5. **continuous learner degraded monitoring**
   - 현재 learner fault는 stdout/JSONL event로만 확인하며 gRPC health는 last-known-good serving 때문에 ready일 수 있다.
   - heartbeat/status/alert 또는 운영 supervisor 정책이 필요하다.

### P1 — production 장기 운용 전 보강

1. actor가 server model/reward/schema identity를 pin하는 기능은 구현됐다. 반대 방향의 actor ID/task/run allowlist 또는 별도 인증은 없으며 현재 보안 경계는 Kanu loopback + SSH access다.
2. run fingerprint에 exact source commit/upstream revision/task identity를 최종 포함해야 한다.
3. JAX update가 native backend에서 영구 hang하면 graceful shutdown이 worker join을 계속 기다릴 수 있다.
4. replay/intervention은 RAM-only라 restart 후 training distribution이 달라진다.
5. sampler NumPy RNG와 RAM buffer는 checkpoint에 없으므로 whole-process bitwise continuation은 아니다.
6. W&B online 인증/업로드와 장시간 offline disk 증가를 검증하지 않았다.
7. default 50-step publish/5,000-step checkpoint까지 실제 GPU bounded run이 필요하다.
8. checkpoint 약 305 MiB가 계속 누적되므로 run별 disk capacity/retention 운영 절차가 필요하다. 자동 pruning은 추가하지 않는다.
9. runtime dependency validator는 JAX/JAXLIB/Flax/Distrax/TFP와 W&B만 검사한다. NumPy/Optax/protobuf/grpcio/Orbax는 lock/known environment에는 고정돼 있지만 같은 fail-closed preflight에 아직 포함되지 않았다.

## 8. 권장 다음 순서

1. GAP512 ADR을 승인한다: feature key/schema, no-aug, encoder freeze, offline encoding, fingerprint.
2. current hardening과 feature 변경을 완료한 뒤 기본 suite와 opt-in real-agent test를 clean 재실행한다.
3. fake canonical demo로 Kanu `--dry-run`을 실행한다.
4. 실제 classifier + GPU backend + W&B offline artifact + RSS를 확인한다.
5. 실제 canonical robot demo가 준비되면 5,000-step bounded learner run을 수행한다.
6. SSH tunnel을 통해 fake-env actor loopback을 먼저 검증한다.
7. task/camera/safety review 후 실제 robot actor를 연결한다.
8. learner fault heartbeat와 shutdown escalation을 보강한 뒤 continuous mode를 승인한다.
9. 검증된 production learner commit을 canonical `feat/gello-ur7e-humble-22.04`에 통합하고 임시 branch/worktree를 정리한다.

## 9. 재현 명령

상세 Kanu 명령은 [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md)에 있다.

로컬 빠른 suite의 권장 형태는 다음과 같다.

```bash
cd /home/laptop3/gello_worktrees/hil-production-learner

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
PYTHONPATH=/home/laptop3/gello_worktrees/hil-production-learner/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests
```

실제 agent opt-in test:

```bash
cd /home/laptop3/gello_worktrees/hil-production-learner

RUN_HIL_SERL_ACTUAL_CHECKPOINT=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
MPLCONFIGDIR=/tmp/gello-hil-production-matplotlib \
PYTHONPATH=/home/laptop3/gello_worktrees/hil-production-learner/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q \
  serl_ur_infra/tests/test_actual_agent_checkpoint_integration.py
```

## 10. handoff 불변식

- fake demo는 acceptance `--dry-run` 전용이다.
- 실제 serving에서 fake marker를 제거하거나 우회하지 않는다.
- loopback bind와 SSH tunnel 경계를 유지한다.
- Kanu에서는 GPU backend를 명시하고 CPU fallback을 허용하지 않는다.
- `third_party/hil-serl`, proto, generated binding을 직접 수정하지 않는다.
- checkpoint를 덮어쓰기·삭제·pruning하지 않는다.
- resume는 fingerprint/counter/RNG 검사를 우회하지 않는다.
- learner fault가 나도 last-known-good policy를 오염시키지 않는다.
- replay가 RAM-only라는 사실을 checkpoint resume와 혼동하지 않는다.
- feature buffer는 GAP512/freeze/no-aug 결정 전까지 구현 완료라고 기록하지 않는다.
