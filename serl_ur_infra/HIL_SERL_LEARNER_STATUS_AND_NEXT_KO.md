# HIL-SERL 로컬 Learner 작업 현황과 다음 단계

> 기준 시각: 2026-07-27 KST
>
> 작업 브랜치: `feat/hil-rl-learner`
>
> 분기 기준 커밋: `e42dbf3848f16a97cffe9ce9ca8ccfdfc2b4265b`

## 한눈에 보기

- 실제 작업 위치는 `/home/laptop3/gello_worktrees/hil-rl-learner`, 브랜치는 `feat/hil-rl-learner`이다. `feat/hil-rl-receive-server`의 `e42dbf3`에서 분기했다.
- CPU JAX/JAXLIB `0.5.3` 환경에서 실제 `SACAgentHybridSingleArm` 생성, CTA update, policy publish, 전체 Flax train state checkpoint 저장·복원, 복원 후 추가 학습까지 통과했다.
- 전체 테스트는 기존 74개와 신규 19개를 합쳐 `93 passed`이다. 실제 agent checkpoint 검증은 성공했지만 아직 자동 테스트나 재현 스크립트로 저장소에 들어가지는 않았다.
- learner 라이브러리 경계는 구현했지만 실행 CLI, receive server와 learner의 production 조립, robot wrapper chain 배선, 실제 W&B offline artifact 검증은 아직 남아 있다. 따라서 현재 코드를 Kanu나 실제 robot에서 바로 실행하면 안 된다.
- 다음 큰 방향은 replay buffer에 raw image 대신 image encoder feature vector를 저장하는 것이다. 다만 현재 upstream은 ResNet backbone 뒤의 visual head가 학습되므로, 최종 256-D feature 저장은 단순 캐시가 아니라 encoder 동결과 augmentation 변경을 포함하는 새 학습 구성이다.
- 병렬 에이전트 3개가 작업 인벤토리, 코드·테스트, feature-buffer 설계를 각각 읽기 전용으로 검수했다. 검수에서 intervention 소수 표본 starvation, 불완전한 최신 checkpoint 선택, resume 객체 간 counter 불일치 가능성을 후속 수정 항목으로 확인했다.

---

## 1. 이 문서의 목적과 판정 기준

이 문서는 지금까지 실제로 구현하고 검증한 범위, 아직 라이브러리 형태로만 존재하는 범위, 다음 세션에서 이어서 해야 할 일을 한곳에 남기기 위한 handoff 문서다. 특히 “코드가 존재한다”와 “실제 agent로 실행했다”, “production 경로에 연결됐다”를 구분한다.

| 표기 | 의미 |
| --- | --- |
| 완료·자동 검증 | 저장소 코드와 자동 테스트가 모두 존재한다. |
| 완료·수동 실 agent 검증 | 실제 upstream agent로 실행해 성공했지만 자동 재현 테스트는 아직 없다. |
| 구현만 완료 | 라이브러리 코드는 있으나 운영 조립 또는 실제 외부 결합이 없다. |
| 유보 | 이번 로컬 milestone에서 의도적으로 하지 않았다. |
| 다음 우선 수정 | 병렬 검수에서 정확성 또는 운영 안정성 문제를 확인했다. |

현재 판정은 “로컬 learner foundation 및 실제 checkpoint/resume smoke 완료”다. “Kanu/robot에서 사용할 수 있는 learner service 완료”는 아니다.

## 2. 작업 위치, 브랜치, 기준점

### 2.1 Git worktree와 branch

| 용도 | 디렉터리 | 브랜치 | 기준 HEAD | 비고 |
| --- | --- | --- | --- | --- |
| 원래 주 workspace | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` | `dc25cbe403aec6b610a4634caede4687606a2c35` | learner 작업과 분리되어 있으며 이 worktree의 별도 dirty 파일은 이번 커밋 대상이 아니다. |
| 현재 learner 작업 | `/home/laptop3/gello_worktrees/hil-rl-learner` | `feat/hil-rl-learner` | 작업 시작 시 `e42dbf3848f16a97cffe9ce9ca8ccfdfc2b4265b` | 이 문서와 learner 변경을 커밋할 위치다. |
| receive server 기준 | `/tmp/gello-hil-rl-receive-server` | `feat/hil-rl-receive-server` | `e42dbf3848f16a97cffe9ce9ca8ccfdfc2b4265b` | learner branch의 실제 분기 기준이다. |
| actor adapter 참고 | `/tmp/gello-hil-actor-adapter` | `feat/hil-grpc-actor-transport` | `5709bb59d7354afdca9d0e62a298c60f1bc08376` | 기존 actor transport 확인용이다. |

learner worktree는 2026-07-27 KST에 `e42dbf3`에서 만들었다. 기준 커밋 메시지는 `docs(hil): add Korean receive server handoff`다.

### 2.2 third-party와 생성 코드

- `third_party/hil-serl` submodule은 `c32939bccb65f3b8c43a9f9add3d322d4ab0264a`이며 clean 상태다.
- `third_party/hil-serl` 내부 코드는 수정하지 않았다.
- `actor_transport_pb2.py`, `actor_transport_pb2_grpc.py`, `.proto` 파일은 수정하지 않았다.
- 다만 protobuf 런타임 호환 설정을 적용하기 위해 infra-owned `ur_env/proto/__init__.py`는 수정했다.
- 기존 receive-only server 파일은 learner-mode penalty 계약을 추가하기 위해 수정했다. 최초 계획의 “receive server를 수정하지 않는다”와 달라진 부분이며 아래 변경 목록에 명시한다.

### 2.3 로컬 검증 환경과 임시 artifact

| 용도 | 경로 | 상태 |
| --- | --- | --- |
| 최종 검증 virtualenv | `/tmp/gello-hil-rl-learner-venv` | Python 3.10.12, 약 1.2 GiB, 이번 결과의 기준 환경 |
| 탐색용 Kanu 호환 venv | `/tmp/gello-hil-rl-learner-kanu-venv` | 탐색용이며 최종 lock 환경이 아님 |
| 중간 lock 탐색 venv | `/tmp/gello-hil-rl-lock-venv` | 최종 검증에 사용하지 않음 |
| 검증된 ResNet cache | `/tmp/gello-hil-rl-learner-assets/resnet10_params.pkl` | repo asset과 같은 SHA-256 |
| 직접 checkpoint roundtrip | `/tmp/gello-hil-rl-actual-checkpoint-v1/checkpoint_000000000001` | 보존 중, 약 306 MiB |
| 독립 updated-agent roundtrip | `/tmp/actual-updated-checkpoint-r7qijh75/checkpoint_000000000001` | 보존 중, 약 306 MiB |
| learner resume 연속 검증 | `/tmp/actual-learner-resume-y_dfio4l/` | step 1과 step 2 checkpoint 모두 보존 중 |

`/tmp` 아래 artifact는 이번 작업에서 삭제하거나 pruning하지 않았다. 다만 `/tmp`는 장기 영속 저장소가 아니므로 중요한 실험 checkpoint의 최종 저장 위치로 사용하면 안 된다.

## 3. 이번 milestone의 학습 계약

### 3.1 Agent와 observation/action

- upstream `SACAgentHybridSingleArm`을 사용한다.
- observation은 `state: float32 (1,19)`, `cam1/cam2: uint8 (1,128,128,3)`의 canonical schema다.
- action은 `float32 (7,)`이며 EEF 6D 연속값과 gripper 한 축으로 구성한다.
- gripper 출력은 정확히 `{-1, 0, 1}` 중 하나여야 한다.
- seed는 `42`, discount는 `0.97`, encoder는 pretrained ResNet-10이다.
- batch size 기본값은 `256`이다.
- online replay 50%와 demo union 50%로 batch를 구성한다.
- CTA ratio는 2다. 한 learner step마다 `critic + grasp_critic` update를 한 번 수행한 뒤 전체 network update를 한 번 수행한다.
- online replay가 100개 이상이고 canonical offline demo가 비어 있지 않을 때만 학습을 시작한다.
- 초기 policy는 첫 observation부터 사용 가능하며 version 0이다.
- 기본 publish 주기는 learner step 50, checkpoint 주기는 publish 경계인 learner step 5,000이다.

### 3.2 Learner batch 계약

학습기로 전달하는 batch에는 다음 필드만 남긴다.

```text
observations
next_observations
actions
rewards
masks
grasp_penalty
```

agent update 직전에는 timestamp, episode ID, transition ID, intervention 여부, success label 등을 제거한다. receive ingress의 raw replay store에는 일부 numeric metadata array가 존재하지만 `sanitize_learner_batch()`가 위 6개 학습 필드만 남긴다. demo loader의 추가 logging/diagnostic metadata는 sidecar에 둔다. actor backup의 `{meta, transition}`와 canonical flat transition pickle을 지원한다. joint-space LeRobot 형식, 잘못된 shape/dtype, non-finite 값, 누락된 `grasp_penalty`는 fail-fast로 거부한다.

### 3.3 Gripper penalty 계약

- penalty 기본값은 upstream task 설정과 같은 `-0.02`다.
- robot-local wrapper가 실제 실행된 action을 기준으로 계산해야 한다.
- intervention이 있으면 policy proposal 대신 `info["intervene_action"]`을 사용한다.
- 이미 닫힌 gripper에 close를 반복하거나 이미 열린 gripper에 open을 반복할 때만 penalty를 준다.
- learner mode에서 신규 transition이나 demo에 `grasp_penalty`가 없으면 0으로 조용히 보정하지 않고 계약 오류로 거부한다.

## 4. 구현된 파일과 역할

### 4.1 신규 learner 모듈

| 파일 | 구현 내용 | 현재 성숙도 |
| --- | --- | --- |
| `serl_ur_infra/ur_env/learner/config.py` | seed, batch, 50:50, CTA, publish/checkpoint 주기, discount, encoder, W&B mode 검증 | 완료·자동 검증 |
| `serl_ur_infra/ur_env/learner/demo.py` | canonical flat/actor backup loader, 엄격한 schema/dtype/shape 검사, success 정규화, sidecar, canonical pool | 완료·자동 검증 |
| `serl_ur_infra/ur_env/learner/batches.py` | learner field 정리, online/demo 50:50, offline/online-intervention demo source 배분 | 구현·자동 검증이나 starvation 이슈 있음 |
| `serl_ur_infra/ur_env/learner/agent.py` | dependency pin 확인, ResNet SHA/cache 확인, 실제 hybrid SAC factory, augmentation | 완료·실 agent 검증 |
| `serl_ur_infra/ur_env/learner/policy.py` | `PolicyPublisher`, 논리적으로 immutable하게 취급하는 parameter reference의 atomic 교체, tree/dtype/finite/action smoke, version 증가 | 완료·자동 검증, 외부 mutation 방어 보강 필요 |
| `serl_ur_infra/ur_env/learner/checkpoint.py` | 전체 Flax train state msgpack, metadata/checksum/fingerprint/counters/inference RNG, overwrite 금지 | 완료·실 agent 검증, latest 선택 보강 필요 |
| `serl_ur_infra/ur_env/learner/logging.py` | 동기 JSONL과 W&B event mirror, tensor/log payload 제한 | fake W&B 자동 검증만 완료 |
| `serl_ur_infra/ur_env/learner/runtime.py` | CTA loop, publish/checkpoint 경계, learner-only fault state, last-known-good policy 유지 | 완료·실 agent 수동 검증, production 조립 미완료 |
| `serl_ur_infra/ur_env/learner/__init__.py` | 공개 API export | 완료 |

### 4.2 환경 및 호환성 파일

| 파일 | 구현 내용 |
| --- | --- |
| `serl_ur_infra/requirements-learner.lock` | CPU learner 검증 dependency pin |
| `serl_ur_infra/sitecustomize.py` | process 시작 시 protobuf 4+를 pure-Python implementation으로 선택 |
| `serl_ur_infra/ur_env/compat.py` | protobuf가 이미 잘못 import된 경우 조용히 진행하지 않고 명시적으로 실패 |
| `serl_ur_infra/tensorflow/__init__.py` | upstream type annotation용 `tf.Tensor` 및 최소 marker만 제공 |
| `serl_ur_infra/tensorflow/io/__init__.py` | TensorFlow I/O 접근을 명시적으로 거부 |
| `serl_ur_infra/tensorflow/errors.py` | optional backend import에 필요한 최소 exception name |

TensorFlow 전체 runtime은 설치하지 않았다. shim은 TensorFlow 기능의 대체 구현이 아니며, I/O나 실제 TensorFlow 연산을 요청하면 실패하도록 설계했다.

### 4.3 기존 파일 수정

| 파일 | 변경 내용 |
| --- | --- |
| `serl_ur_infra/ur_env/envs/config.py` | `GRASP_PENALTY=-0.02` 기본값 추가 |
| `serl_ur_infra/ur_env/envs/wrappers.py` | `GripperPenaltyWrapper` 추가 |
| `serl_ur_infra/ur_env/rlpd_receive_server.py` | `learner_mode`와 `require_grasp_penalty` strict contract 추가 |
| `serl_ur_infra/ur_env/proto/__init__.py` | generated binding import 전에 protobuf compatibility 적용 |
| `serl_ur_infra/tests/test_rlpd_receive_server.py` | learner mode의 penalty 누락 거부 검증 추가 |

중요하게도 `GripperPenaltyWrapper`와 `learner_mode=True`는 현재 production 실행 경로에 아직 배선되지 않았다. 클래스와 계약이 존재한다고 해서 실제 actor transition에 penalty가 항상 들어오는 상태는 아니다.

### 4.4 신규 테스트

| 파일 | 주요 범위 |
| --- | --- |
| `serl_ur_infra/tests/test_gripper_penalty.py` | 기본 penalty, 중복 open/close, intervention action 우선 |
| `serl_ur_infra/tests/test_learner_data.py` | demo schema, LeRobot 거부, batch field, 50:50, startup 조건 |
| `serl_ur_infra/tests/test_learner_policy_checkpoint.py` | atomic publish, bad snapshot 거부, learner fault 격리, fake-agent checkpoint, fake-W&B logging |

## 5. 현재 코드의 의도된 데이터 흐름

아래 흐름의 구성 요소는 구현됐지만, 하나의 production entrypoint로 아직 조립되지 않았다.

```text
robot actor
  -> canonical raw transition + grasp_penalty
  -> ReplayIngress learner-mode validation
  -> online replay / online intervention route

canonical offline demo ----------------------+
                                               |
online intervention pool ------------------+  |
                                            v  v
online replay --------------------------> RLPDBatchSampler
                                         50% replay
                                         50% demo union
                                              |
                                              v
                                  HILSERLLearner.train_once()
                                  1) critic/grasp update
                                  2) all-network update
                                              |
                             +----------------+----------------+
                             |                                 |
                      every 50 steps                    every 5000 steps
                             v                                 v
                  PolicyPublisher.publish()            CheckpointManager.save()
                             |
                             v
               VersionedPolicyRuntime last-known-good snapshot
```

Learner update, validation, publish, checkpoint, log 중 예외나 non-finite state가 나오면 learner 객체만 영구 fault 상태로 전환한다. update 또는 publish validation이 실패하면 이전 snapshot을 유지한다. publish가 성공한 뒤 checkpoint나 logging이 실패하면 방금 검증·publish된 snapshot이 active인 채 learner만 fault된다. 두 경우 모두 마지막으로 검증된 policy를 유지한다. 단 외부 코드가 내부 mutable dict를 직접 변경하는 경우까지 현재 runtime이 막지는 못하며, 이 위험은 후속 항목에 기록한다. 실제 inference/classifier/replay ingress의 기존 fail-stop 정책은 변경하지 않았다.

## 6. Dependency와 asset 고정 상태

최종 검증 venv의 주요 버전은 다음과 같다.

| package | version |
| --- | --- |
| Python | 3.10.12 |
| JAX / JAXLIB | 0.5.3 / 0.5.3 |
| Flax | 0.10.5 |
| Distrax | 0.1.5 |
| TensorFlow Probability | 0.25.0 |
| Optax | 0.2.4 |
| NumPy | 1.26.4 |
| W&B | 0.26.0 |
| protobuf | 7.34.1 |
| grpcio | 1.74.0 |
| Orbax | 0.11.5 |

ResNet-10 repo asset과 검증 cache는 둘 다 21,677,915 bytes이며 SHA-256은 아래와 같다.

```text
175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b
```

cache가 이미 존재하면서 SHA가 다르면 덮어쓰지 않고 실패한다. 이 milestone에서는 `/tmp/gello-hil-rl-learner-assets/resnet10_params.pkl`을 사용했다.

## 7. 검증 결과

### 7.1 자동 테스트

| 검증 | 결과 | 해석 |
| --- | --- | --- |
| 기존 receive/actor suite | 74 passed | learner 변경 전 기준 회귀 |
| 신규 learner/gripper/penalty suite | 19 passed | 이번 변경의 신규 자동 검증 |
| 최종 전체 suite | `93 passed in 1.22s` | 허용된 localhost loopback 환경에서 최종 확인 |
| `git diff --cached --check` | pass | 신규 파일까지 stage한 뒤 전체 변경의 whitespace 오류가 없음을 확인 |

restricted sandbox에서 처음 실행했을 때는 `88 passed, 5 failed`였다. 실패 5개는 모두 `127.0.0.1:0` bind가 차단되어 발생한 gRPC loopback 테스트였다. 같은 코드를 localhost bind 권한으로 재실행해 93개 전부 통과했으므로 코드 assertion 실패로 판정하지 않는다.

### 7.2 실제 hybrid SAC 생성과 update

CPU JAX/JAXLIB `0.5.3`에서 실제 `SACAgentHybridSingleArm`으로 다음을 확인했다.

- agent parameter tree: 81 leaves, 32,006,980 bytes, 약 30.524 MiB
- deterministic action: `float32 (7,)`, finite, policy version 0, gripper 3-state 만족
- stochastic action: `float32 (7,)`, finite, deterministic 결과와 다름, gripper 3-state 만족
- synthetic packed raw-image batch size 2로 critic/grasp update 후 `state.step=1`
- 이어서 all-network update 후 `state.step=2`
- 양 update info와 agent state의 scalar/tensor가 finite

이 smoke는 local CPU에서 약 23초 수준이었다. production 성능 수치가 아니라 import/JIT/update 계약 확인 수치다.

### 7.3 실제 agent checkpoint 직접 roundtrip

실제 updated agent의 전체 state를 저장한 뒤 fresh agent template에 복원해 다음을 확인했다.

| 항목 | 결과 |
| --- | --- |
| checkpoint payload | 320,100,609 bytes, 약 305.272 MiB |
| save / restore | 약 0.773초 / 0.464초였던 1차 측정 |
| learner / gradient / policy counters | 보존 |
| params | 동일 |
| target params | 동일 |
| optimizer state | 동일 |
| learner internal RNG | 동일 |
| inference RNG | 동일 |
| deterministic action | exact equal |
| 같은 RNG의 stochastic action | exact equal |
| checksum / fingerprint | 검증 통과 |

별도 독립 검수에서는 state leaf 829개 전체의 exact value equality도 확인했다. serialized msgpack byte identity까지 주장하는 결과는 아니다. fake agent가 아니라 실제 upstream `SACAgentHybridSingleArm`의 updated state다.

checkpoint payload는 `agent.state`를 저장한다. upstream agent의 non-pytree 구성, apply function, optimizer transform 객체 등은 checkpoint에서 직접 복원하는 것이 아니라 fingerprint와 같은 config로 fresh agent template을 재생성한 뒤 state를 주입한다. 따라서 “전체 process object 직렬화”가 아니라 “전체 Flax train state 복원”으로 이해해야 한다.

### 7.4 실제 learner publish/checkpoint/resume/continue E2E

checkpoint와 publish 주기를 1로 낮춘 smoke configuration으로 실제 learner orchestration을 다음 순서로 실행했다.

1. 실제 hybrid agent와 synthetic batch sampler를 생성했다.
2. `HILSERLLearner.train_once()`로 CTA update를 수행했다.
3. learner 1, gradient 2, policy 1이 된 시점에 publish와 `checkpoint_000000000001` 저장을 확인했다.
4. fresh real-agent template에 checkpoint를 복원했다.
5. 복원한 agent, policy runtime, learner를 counters와 inference RNG로 재구성했다.
6. deterministic action과 policy version이 복원 전과 동일함을 확인했다.
7. 복원 learner에서 다시 `train_once()`를 수행했다.
8. learner 2, gradient 4, policy 2, `checkpoint_000000000002`, fault 없음을 확인했다.

두 checkpoint는 `/tmp/actual-learner-resume-y_dfio4l/`에 보존돼 있다. 각각의 msgpack payload는 320,100,609 bytes다. metadata fingerprint는 이 smoke config 기준 `f687cbffb2ebcd9e704b38157b452e9ec46d1dade7218107bd4c4a1b81917d43`이다.

이 결과는 2026-07-27 로컬 ad-hoc 검증이다. 저장소 안의 자동 테스트나 스크립트로 아직 남아 있지 않다는 점이 가장 중요한 제한이다. 또한 실제 기본 주기인 50/5,000까지 장시간 돌린 검증이 아니라, 동일 경계를 빠르게 검증하기 위해 period를 1로 낮춘 smoke다.

resume 후 추가 학습 성공은 model/train-state가 이어진다는 뜻이다. checkpoint가 sampler의 NumPy RNG나 RAM replay/intervention 내용을 저장하지 않으므로 동일 mini-batch sequence를 재현하는 deterministic process continuation은 아니다.

### 7.5 아직 검증하지 않은 것

- Kanu GPU에서의 실제 학습
- 실제 robot/camera/task config 결합
- 실제 reward classifier와 learner의 GPU 동시 운용
- production receive server에서 learner까지 이어지는 end-to-end data flow
- W&B `offline` 실제 run directory/artifact 생성
- W&B online login과 업로드
- W&B와 protobuf/gRPC를 한 production process에서 장시간 함께 사용
- 기본 50-step publish 및 5,000-step checkpoint까지의 장시간 run
- replay/intervention buffer persistence
- sampler RNG와 RAM buffer까지 포함한 process 전체 deterministic resume

## 8. 현재 완성도 표

### 8.1 완료·검증된 foundation

- canonical demo/actor-backup strict loader와 sidecar
- bad shape/dtype, LeRobot, label 불일치, penalty 누락 거부
- learner batch 필드 제한
- online 50% + demo 50% batch 경계
- 실제 pretrained hybrid SAC 생성과 action/update
- gripper redundant action penalty 계산 로직
- atomic policy snapshot, monotonic version, bad snapshot 거부
- learner non-finite fault와 last-known-good policy 유지
- full Flax train state checkpoint, checksum, fingerprint, overwrite 금지
- 실제 agent save/load 및 resume 후 추가 update
- JSONL 및 W&B adapter의 fake-module 단위 검증
- protobuf 7.34.1 pure-Python compatibility와 TensorFlow annotation shim

### 8.2 구현됐지만 production에 연결되지 않은 항목

- `HILSERLLearner`, sampler, publisher, logger, checkpoint manager를 조립하는 실행 CLI
- receive ingress buffer를 learner sampler에 연결하는 장기 실행 process
- initial policy version 0을 actor inference에 실제 제공하는 wiring
- 50-step publish와 5,000-step checkpoint를 운영 경로에서 실행하는 scheduler
- `GripperPenaltyWrapper`를 실제 UR task wrapper chain에 삽입하는 구성
- receive server CLI에서 `ReplayIngress(..., learner_mode=True)`를 활성화하는 구성
- restore 결과를 policy runtime과 learner에 안전하게 함께 주입하는 단일 resume factory

### 8.3 의도적으로 유보한 항목

- learner와 inference를 같은 process에 둘지 별도 service로 나눌지 결정
- Kanu GPU deployment
- 실제 robot reward classifier/GPU contention 검증
- W&B online 인증
- replay/intervention buffer persistence
- checkpoint pruning 또는 삭제
- raw image 대신 feature-only replay를 구현하는 알고리즘 v2

## 9. 병렬 코드 검수에서 확인한 후속 수정 사항

### 9.1 P0: demo-union minority starvation

현재 offline demo와 online intervention의 batch 수를 매 batch마다 deterministic largest-remainder 방식으로 나눈다. 예를 들어 offline 1,000개, intervention 1개, demo draw 128개면 intervention 배정이 매번 0이 될 수 있다. 이것은 두 저장소의 합집합에서 uniform sampling하는 upstream 의미와 동등하지 않다.

다음 수정에서는 아래 중 하나를 구현하고 통계 테스트를 추가해야 한다.

- source를 전체 크기 비례 categorical로 매 draw stochastic 선택
- 누적 fractional remainder를 유지해 장기적으로 기대 비율을 맞춤
- 두 pool을 하나의 logical index space로 노출해 uniform index sampling

acceptance 기준은 작은 intervention pool도 충분한 반복 횟수에서 기대 비율로 선택되고, seed를 고정하면 실행이 재현되는 것이다.

### 9.2 P0: incomplete latest checkpoint 선택

현재 `latest_path()`는 이름이 가장 큰 checkpoint 디렉터리를 고른다. save 도중 실패한 디렉터리를 보존하는 정책과 결합하면, 다음 기본 resume가 불완전한 최신 디렉터리를 선택해 실패하고 이전 정상 checkpoint로 자동 fallback하지 못할 수 있다.

다음 수정의 권장 계약은 다음과 같다.

- payload와 metadata를 모두 기록하고 fsync한 뒤 immutable completion marker를 마지막에 생성
- `latest_path()`는 completion marker, checksum, metadata format을 통과한 checkpoint만 후보로 취급
- 불완전 checkpoint는 삭제하지 않고 fault event에 경로를 기록
- 가장 최신 valid checkpoint가 없으면 명확히 실패

### 9.3 P0: resume 객체 간 invariant

`HILSERLLearner.__init__`는 전달된 learner/policy counters와 publisher가 가진 counters가 일치하는지 검증하지 않는다. caller가 `RestoredCheckpoint` 값을 learner나 runtime 한쪽에 누락해도 초기화가 통과할 수 있다.

`RestoredCheckpoint -> VersionedPolicyRuntime + HILSERLLearner`를 만드는 단일 factory를 추가하고 아래를 검사해야 한다.

- `agent.state.step == gradient_step`
- `gradient_step == learner_step * cta_ratio` 또는 명시적인 resume offset 계약
- runtime과 learner의 `learner_step` 일치
- runtime과 learner의 `policy_version` 일치
- checkpoint inference RNG가 runtime에 정확히 전달됨
- resume 직후 publish version이 단조 증가함

### 9.4 P0: production 조립 부재

현재 `scripts/`에 learner entrypoint가 없다. 따라서 library API가 맞아도 실제 receive ingress, replay, demo load, logger, publisher, checkpoint가 함께 실행되는 경로는 없다. 다음 구현에서 명시적인 composition root와 dry-run CLI가 필요하다.

### 9.5 P1: penalty 생산/검사 배선

`GripperPenaltyWrapper`는 현재 test 외에서 인스턴스화되지 않는다. receive CLI도 learner mode를 활성화하지 않는다. 실제 robot path에 wrapper를 넣고, learner server에서는 penalty 누락을 반드시 거부하도록 해야 한다. wrapper 생성 시 자체 default에 의존하지 말고 task config의 `GRASP_PENALTY`를 명시적으로 전달해 task별 override가 실제 transition에 반영되는지도 acceptance test로 확인한다.

### 9.6 P1: counter 검증 강도

checkpoint save/load는 `state.step == gradient_step`은 검사하지만 learner step과 CTA ratio의 정확한 관계는 검사하지 않는다. runtime도 현재는 lower-bound 성격의 검사다. resume offset을 지원할지, 항상 정확히 2:1을 강제할지 결정한 뒤 fingerprint/metadata에 계약을 넣어야 한다.

### 9.7 P1: 실제 integration regression 부재

실 agent checkpoint와 resume 후 추가 update는 성공했지만 그 절차가 ad-hoc이다. 환경 비용을 고려해 기본 unit suite에는 넣지 않더라도, opt-in marker를 가진 integration test 또는 체크인된 smoke script로 남겨야 한다.

### 9.8 P1: logging과 dependency 검증 부족

- W&B 테스트는 `_FakeWandb`만 사용한다.
- 실제 offline run artifact 확인이 없다.
- W&B + checked-in protobuf + gRPC simultaneous import/loopback 자동 검증이 없다.
- 잘못된 ResNet cache와 dependency version mismatch의 자동 회귀 테스트가 부족하다.
- `requirements-learner.lock`은 이름과 달리 모든 transitive dependency의 hash까지 고정한 완전한 lockfile이 아니다.
- runtime validator도 JAX/JAXLIB, Flax, Distrax, TFP와 선택적 W&B만 확인하므로 Optax, NumPy, protobuf, grpcio, Orbax drift는 설치 단계 또는 별도 preflight에서 보강해야 한다.

### 9.9 P1: policy parameter reference의 외부 mutation

`VersionedPolicyRuntime`은 약 32 MiB parameter tree를 복사하거나 직렬화하지 않고 reference를 atomic swap한다. 이 성능 특성은 의도한 것이지만 실제 real-agent params가 내부적으로 mutable dict일 수 있어, publisher 밖의 코드가 같은 object를 변경하면 “immutable snapshot” 가정이 깨진다.

다음 수정에서는 publish candidate를 Flax `FrozenDict` 등 구조적으로 immutable한 tree로 정규화하거나, owner API를 제한하고 publish 후 tree leaf identity/content가 바뀌지 않는 검사를 추가해야 한다. 전체 device array deep copy는 하지 않는다는 원칙은 유지한다.

### 9.10 P1: fingerprint와 process-resume 범위

현재 checkpoint fingerprint는 observation schema, `LearnerConfig`, ResNet asset SHA를 포함한다. upstream HIL-SERL commit, JAX/Flax/Optax versions, hardcoded network architecture와 implementation revision, action schema까지 포함하지 않으므로 shape는 같지만 의미가 바뀐 코드에 state가 load될 여지가 있다.

다음 format에서는 위 항목을 fingerprint에 추가하고 fresh template을 만들기 전에 검증한다. sampler의 NumPy RNG와 RAM replay/intervention은 현재 checkpoint 범위 밖이므로, train-state resume와 whole-process deterministic resume를 API와 문서에서 별도 capability로 구분한다. RAM-only buffer 정책을 유지한다면 resume 후 sample sequence가 달라지는 것을 명시적으로 허용하고 기록한다.

### 9.11 운영 리스크: checkpoint 크기와 보존 정책

agent parameter 자체는 약 30.5 MiB지만 target params와 여러 optimizer state가 포함된 전체 checkpoint는 약 305 MiB다. 기본 5,000-step마다 전부 보존하고 pruning하지 않으므로 장기 run 전에 예상 step 수에 따른 disk budget, filesystem free-space preflight, write latency metric이 필요하다. 자동 삭제는 이번 정책과 맞지 않으므로 추가하지 않는다.

## 10. 다음 방향: raw image 대신 encoder feature 저장

### 10.1 가장 중요한 결론

> 최종 256-D feature-only buffer는 현 upstream과 동등한 캐시가 아니다. trainable visual head, target encoder, pixel augmentation을 포기하거나 재설계하는 새 학습 구성이다. 구현 전에 cut-point와 freeze policy를 승인하고, encoder fingerprint가 다른 feature는 절대 섞지 않는다.

사용자가 요청한 방향은 “transition마다 image를 저장하지 않고, ingress에서 image encoder를 한 번 실행해 feature vector를 저장하며, buffer sample마다 encoder를 다시 실행하지 않는다”이다. 이 방향의 잠재적 성능·메모리 이점은 크지만 아직 benchmark 전이며, 현재 upstream 구조를 그대로 둔 채 storage type만 바꿀 수는 없다.

### 10.2 현재 upstream visual encoder의 실제 구조

현재 `resnet-pretrained` 경로는 다음과 같다.

```text
raw uint8 image 128x128x3
  -> ImageNet mean/std preprocessing
  -> pretrained ResNet-10 convolution backbone
  -> pre-pooling spatial map 4x4x512      [canonical 128x128에서 여기까지 stop_gradient]
  -> SpatialLearnedEmbeddings
  -> Dropout(0.1)
  -> Dense(256) -> LayerNorm -> tanh       [현재 trainable visual head]
  -> camera당 256-D
  -> cam1 256 + cam2 256 + state latent 64
  -> actor/critic/grasp critic이 사용하는 dimensional layout 576
```

actor loss는 image encoder output에 stop-gradient를 적용하지만 critic과 grasp critic은 trainable visual head로 gradient를 흘린다. target tree에도 이 head가 Polyak update된다. 따라서 현재 head가 학습 중 바뀌는데 과거 final feature를 buffer에 저장하면 stale feature가 된다.

현재 update는 raw image를 unpack하고 padding 4 random crop을 매 update마다 새 RNG로 수행한 뒤 encoder를 호출한다. raw image를 버리면 이 pixel-space augmentation을 나중에 정확히 재현할 수 없다. feature-space shift는 같은 augmentation이 아니다.

### 10.3 저장 cut-point 비교

한 unique observation frame에 cam1과 cam2가 모두 있다고 가정한다.

| 방식 | camera당 표현 | 두 camera 저장량 | 장점 | 핵심 단점 |
| --- | --- | ---: | --- | --- |
| 현재 raw | `128x128x3 uint8` | 96 KiB | augmentation과 encoder 재학습 가능 | sample마다 ResNet 계산, 큰 RAM |
| frozen backbone map | `4x4x512 float32` | 64 KiB | conv backbone 재실행 제거, trainable head 유지 가능 | raw 대비 약 1/3만 절감, pixel crop 소실 |
| 최종 compact feature | `256 float32` | 2 KiB | raw 대비 약 48배 축소, learner visual encode 제거 | visual head 전체 freeze와 algorithm 변경 필요 |

현재 worst-case physical slot이 replay 100k와 intervention 20k라면 raw image array의 이론상 크기는 약 10.99 GiB다. 실제 RSS는 allocation/commit과 container 구현에 따라 따로 계측해야 한다. final feature를 observations와 next_observations에 단순 중복 저장해도 logical 60k transition 기준 약 234 MiB이며 observation-ID 공유 registry를 쓰면 더 줄일 수 있다.

### 10.4 권장 의사결정

다음 milestone의 목표가 정말 “작은 feature vector만 저장”이라면 아래를 하나의 명시적인 algorithm v2로 채택하는 것을 권장한다.

1. replay 수집 시작 전에 feature-producing encoder 전체를 확정하고 영구 immutable로 둔다.
2. 현재 random-initialized trainable head를 그대로 얼리는 것은 위험하므로, 별도 pretrained/fitted encoder를 사용할지 deterministic fixed pooling/projector를 사용할지 먼저 A/B gate로 결정한다.
3. 최종 선택한 encoder는 `train=False`, dropout 없음으로 고정한다. 동일 pinned backend/process에서는 같은 raw input의 bitwise-stable feature를 요구한다.
4. pixel random crop augmentation은 v2에서 `None`으로 명시하거나, 사전 계산한 K개 deterministic crop feature를 저장하는 별도 실험으로 분리한다.
5. policy version과 encoder version을 분리한다. v2의 fixed encoder는 run 동안 `encoder_version=0`으로 유지하고 policy core만 publish한다.
6. encoder 교체는 같은 buffer 안에서 하지 않는다. 새 encoder는 새 run, 새 fingerprint, 빈 buffer에서 시작한다.

현 upstream 의미를 더 많이 보존하는 보수적 대안은 frozen backbone의 `4x4x512` map을 저장하는 것이다. 이 경우 trainable spatial/bottleneck head는 learner에 남길 수 있지만 저장량 절감은 작고 augmentation 의미는 여전히 달라진다. 어느 cut-point를 선택해도 ADR과 baseline 비교가 필요하다.

### 10.5 권장 feature schema

기존 `cam1`, `cam2` key에 feature를 위장해서 넣지 않는다. tensor payload와 metadata의 위치도 분리한다. 예시는 다음과 같다.

```text
learner tensor payload:
  observations:
    state:          float32 [1, 19]
    cam1_feature:   float32 [1, F]
    cam2_feature:   float32 [1, F]
  next_observations:
    state:          float32 [1, 19]
    cam1_feature:   float32 [1, F]
    cam2_feature:   float32 [1, F]

buffer-global manifest 또는 logging sidecar:
  feature_schema_id: string
  encoder_version: uint64
  encoder_fingerprint: sha256
```

F는 final compact mode라면 우선 256을 권장하되 ADR로 확정한다. 초기 dtype은 정확성 검증이 쉬운 `float32`로 고정한다. `float16`, bfloat16, int8은 action/Q parity와 실제 task 성능을 확인하기 전에는 사용하지 않는다. 모든 feature는 contiguous, finite, exact shape/dtype를 검사한다.

`feature_schema_id`, `encoder_version`, `encoder_fingerprint` 같은 문자열/metadata를 JAX learner tensor batch에 그대로 넣으면 기존 six-key batch 계약과 tree shape 검사를 깨뜨릴 수 있다. 이 값은 buffer-global manifest 또는 sidecar에서 검증하고, agent update 직전 sanitizer는 계속 기존 6개 learner field와 그 안의 tensor observation만 남겨야 한다. per-transition metadata가 꼭 필요하다면 batch-shaped numeric ID로 저장한 뒤 sanitizer에서 제거한다.

`encoder_fingerprint`에는 최소한 다음을 포함한다.

- feature schema ID와 format version
- cut-point 이름
- camera key와 순서
- output shape와 dtype
- resize/crop/color order/ImageNet normalization 등 preprocessing
- ResNet asset SHA-256
- feature-producing 전체 parameter tree SHA-256
- freeze mode와 augmentation policy
- encoder implementation version
- JAX/JAXLIB version, CPU/GPU backend, precision과 XLA numeric policy

parameter tree SHA는 tree path, shape, dtype, byte order와 payload byte를 포함하는 canonical serialization 규칙으로 계산해야 한다. CPU에서 만든 offline feature와 GPU에서 만든 online feature는 같은 weights라도 convolution의 부동소수점 결과가 bitwise identical하지 않을 수 있다. 한 run에서는 encoding backend와 numeric policy를 하나로 고정하고, 불가피한 cross-backend 사전 검증은 bitwise equality 대신 사전에 승인한 tolerance의 feature/action/Q parity로 판정한다.

buffer 하나는 encoder fingerprint 하나만 허용한다. offline demo, online replay, intervention pool, checkpoint 중 하나라도 다르면 즉시 실패한다. 기존 raw-pixel checkpoint는 feature-mode checkpoint로 자동 migration하지 않고 format mismatch로 fail-fast한다.

fingerprint는 encoder parameter를 복원하지 못한다. fitted/pretrained visual head처럼 parameter가 있는 fixed encoder는 immutable encoder state를 checkpoint에 함께 넣거나, content-addressed 별도 artifact에 보존하고 checkpoint가 그 artifact SHA를 참조해야 한다. load 시 artifact payload의 SHA와 canonical parameter-tree SHA를 모두 확인한 뒤에만 buffer/checkpoint를 연다. ResNet asset SHA만으로는 별도 fitted head나 projector를 복원할 수 없다. 무파라미터 fixed pooling을 선택한 경우에만 implementation/preprocessing fingerprint로 복원이 충분할 수 있다.

### 10.6 권장 process 경계

infra-owned 경계를 다음처럼 분리한다. `third_party/hil-serl`은 수정하지 않는다.

```text
FixedImageEncoder.encode(raw canonical observation)
  -> EncodedObservation(state + cam1_feature + cam2_feature)
  -> FeatureSACCore.action/update(encoded observation)
```

`VersionedPolicyRuntime`은 raw image를 받은 뒤 fixed encoder를 한 번 호출하고 versioned policy-core params를 사용한다. `PolicyPublisher`는 core params만 atomic swap하되 snapshot, buffer, checkpoint의 encoder fingerprint가 모두 같은지 검사한다.

robot/actor 흐름은 다음 순서를 지켜야 한다.

```text
raw next observation
  -> canonical raw schema validation
  -> reward classifier가 raw image 소비
  -> fixed encoder 1회 실행 또는 cache hit
  -> feature schema/fingerprint validation
  -> feature-only replay commit
  -> commit 성공 후 ACK
  -> non-terminal이면 같은 feature로 action inference
  -> raw image 해제
```

classifier가 raw next image를 필요로 하므로 encoder만 실행했다고 바로 raw를 버리면 안 된다. classifier가 먼저 소비한 후 feature-only commit이 완료돼야 한다. encode나 buffer commit이 실패하면 ACK하지 않고 기존 actor transition pipeline처럼 fail-stop한다.

### 10.7 observation feature 재사용

다음 key의 bounded registry를 권장한다.

```text
(run_id, actor_id, session_id, episode_id, observation_id,
 encoder_fingerprint)
```

동일 ID에 다른 raw-byte hash가 오면 collision으로 거부한다. `run_id`와 `episode_id`를 포함하거나 동등한 run/session-scoped monotonic identity를 강제해 종료된 session의 ID 재사용과 충돌하지 않게 한다. 정상 episode에서는 `next_observation_id[t] == observation_id[t+1]`이므로 아래처럼 N-step episode에 fixed policy feature encoder 호출 N+1회를 목표로 한다. reward classifier가 자체 encoder를 사용한다면 그 호출 수는 별도다.

- Begin의 O0를 encode/cache하고 inference에 사용한다.
- Step에서 이전 O(t)는 cache hit한다.
- 새 O(t+1)는 classifier 후 한 번 encode한다.
- 같은 O(t+1)를 transition next feature와 다음 action inference에 재사용한다.
- terminal next observation은 replay를 위해 encode하지만 action inference는 하지 않는다.
- retry/dedup은 재-encode하지 않는다.

episode gap, truncation, intervention route, out-of-order retry에서는 인접성을 가정하지 않는다. replay와 intervention pool에 같은 transition이 필요하면 immutable feature blob/reference를 공유하고 ring overwrite 시 lifetime/refcount를 정확히 처리한다. NumPy host array는 `writeable=False`로 만들거나 defensive immutable wrapper를 사용하고, 양쪽 pool insert 중 한쪽만 실패하는 경우의 refcount rollback도 원자적으로 처리한다.

현재 actor session state는 다음 Step을 위해 raw `current_observation`을 보관한다. buffer에서 raw image만 없애는 수준을 넘어 process 전체의 장기 raw retention을 없애려면 session state도 feature, observation identity, raw hash 중심으로 refactor해야 한다. 단 reward classifier와 오류 진단이 끝나기 전까지 필요한 raw lifetime은 명시적으로 유지한다.

### 10.8 offline demo migration

- 기존 canonical raw demo와 actor backup은 load 시 한 번 fixed encoder로 변환하고 raw object를 해제한다.
- 미리 계산한 feature-demo artifact를 허용한다면 manifest, item count, checksums, encoder fingerprint가 필수다.
- raw demo와 feature demo를 한 run에서 섞을 때는 둘 다 같은 encoder fingerprint의 feature로 normalize된 뒤에만 pool에 들어갈 수 있다.
- joint-space LeRobot와 기존 잘못된 schema는 계속 거부한다.
- timestamp/episode/intervention label은 feature tensor와 분리된 sidecar에 유지한다.

### 10.9 feature-buffer 필수 테스트

- 동일 raw + immutable encoder가 동일 feature를 생성하는지 검사
- shape/dtype/finite/contiguous와 preprocessing/fingerprint mismatch 거부
- final compact cut을 선택했다면 CTA update 후 fixed encoder params가 optimizer/target tree에 없고 bitwise 불변인지 검사
- backbone-map cut을 선택했다면 frozen backbone만 밖에 있고 trainable spatial/bottleneck head는 online/target tree에서 update되는지 검사
- feature buffer sample/update 동안 encoder spy call count가 0인지 검사
- raw-to-feature-to-core action과 prepared-feature action parity
- deterministic/stochastic 7D action, gripper 3-state
- N-step episode의 fixed policy feature encoder 호출 횟수가 N+1인지 검사하고 classifier encoder 호출은 별도 계측
- terminal, retry, intervention gap, ID collision 처리
- replay/intervention immutable feature 공유, host mutation 거부, atomic insert/refcount rollback과 overwrite lifetime
- buffer 내부에 `uint8 [128,128,3]` leaf가 전혀 없는 memory audit
- 50:50 replay/demo와 수정된 minority-safe proportional sampling 유지
- `grasp_penalty` strict contract 유지
- feature batch의 실제 CTA update와 `_unpack` 미호출 확인
- classifier가 raw next를 먼저 소비하고 feature-only commit되는 loopback
- commit 실패 시 no ACK와 actor fail-stop
- checkpoint roundtrip action/counters/RNG 재현
- encoder/schema mismatch 및 old pixel checkpoint 거부
- augmentation disabled 계약
- K-crop 방식을 실험한다면 cam1/cam2는 같은 crop index, observations/next_observations는 독립 crop 선택을 쓰는 paired-camera 계약
- baseline 대비 throughput, encoder call count, RAM/RSS benchmark
- 최종적으로 실제 robot success/generalization A/B

## 11. 다음 작업 순서

이번 커밋 이후에는 아래 순서로 진행하는 것이 안전하다.

### Phase 0: 현재 learner correctness 보강

1. demo source sampling starvation을 수정한다.
2. valid checkpoint completion marker와 latest fallback을 구현한다.
3. resume composition factory와 cross-object invariant를 추가한다.
4. actual-agent checkpoint/resume/continue smoke를 opt-in integration test 또는 script로 저장소에 남긴다.
5. 실제 W&B offline artifact와 W&B/protobuf/gRPC 동시 import loopback을 검증한다.
6. published params를 구조적으로 immutable하게 만들거나 외부 mutation을 방지하는 ownership 검사를 추가한다.
7. checkpoint fingerprint에 upstream/code/architecture/action/dependency revision을 포함한다.
8. 완전한 lock 생성 또는 전체 dependency preflight 정책을 정한다.

### Phase 1: feature-mode ADR와 baseline 계측

1. final 256-D, backbone map, fixed pooling 후보의 cut-point를 문서로 결정한다.
2. 현재 CTA update 전후에 backbone은 불변이고 visual head는 변한다는 테스트를 남긴다.
3. encoder call count, update latency, replay memory를 baseline으로 측정한다.
4. augmentation 제거가 알고리즘 변경임을 실험 config/fingerprint에 반영한다.

### Phase 2: immutable encoder와 schema

1. `FixedImageEncoder`와 `EncodedObservation`을 infra-owned API로 추가한다.
2. deterministic preprocessing과 feature validation을 구현한다.
3. encoder fingerprint/version contract를 구현한다.
4. parameter가 있는 fixed encoder의 content-addressed artifact 저장/load/checksum 계약을 구현한다.
5. raw/feature 혼합과 mixed fingerprint를 fail-fast로 거부한다.

### Phase 3: feature-native replay와 demo

1. raw packed image와 `_unpack`에 의존하지 않는 pool/sampler를 추가한다.
2. observation-ID registry로 obs/next feature를 재사용한다.
3. offline demo one-time encoding과 optional artifact manifest를 구현한다.
4. classifier 소비 후 raw 해제, feature commit 후 ACK 순서를 연결한다.
5. actor session state의 raw `current_observation` 보관을 feature/identity/hash 중심으로 refactor한다.

### Phase 4: Feature SAC core와 policy/checkpoint

1. vendor 수정 없이 infra-owned feature input agent/core를 구성한다.
2. final compact cut이면 fixed encoder params를 optimizer/target tree에서 제거한다. backbone-map cut이면 trainable spatial/bottleneck head는 online/target tree에 유지한다.
3. policy runtime을 fixed encoder + versioned core로 분리한다.
4. checkpoint가 immutable encoder state 또는 content-addressed encoder artifact SHA를 보존하도록 하고, format/fingerprint를 feature mode로 올린 뒤 old format은 fail-fast한다.
5. 실제 feature batch CTA와 resume 후 continue를 검증한다.

### Phase 5: production learner composition

1. receive ingress, buffers, demo loader, learner, publisher, logger, checkpoint를 조립하는 CLI를 만든다.
2. robot wrapper chain에 `GripperPenaltyWrapper`를 연결하고 task config의 `GRASP_PENALTY`를 명시적으로 전달한다.
3. learner receive mode에서 penalty strict contract를 활성화한다.
4. bounded encoder queue, backpressure, latency/fault metrics를 추가한다.
5. dry-run과 localhost gRPC end-to-end를 통과시킨다.

### Phase 6: Kanu와 robot 검증

1. Kanu GPU 환경에서 dependency/asset fingerprint를 확인한다.
2. reward classifier와 fixed encoder/learner의 GPU resource contention을 측정한다.
3. W&B online 검증은 login 정보가 제공된 뒤 별도 한 번 수행한다.
4. 실제 task/camera config와 safety 조건을 검토한 뒤 robot A/B를 수행한다.

## 12. 재현 명령과 운영 메모

### 12.1 현재 worktree 확인

```bash
cd /home/laptop3/gello_worktrees/hil-rl-learner
git branch --show-current
git rev-parse HEAD
git status --short
git submodule status third_party/hil-serl
```

### 12.2 전체 자동 테스트

```bash
cd /home/laptop3/gello_worktrees/hil-rl-learner
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_ADDOPTS='-p no:cacheprovider' \
MPLCONFIGDIR=/tmp/gello-hil-rl-matplotlib \
PYTHONPATH=/home/laptop3/gello_worktrees/hil-rl-learner/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python -m pytest -q serl_ur_infra/tests
```

예상 결과는 `93 passed`다. loopback port를 금지하는 sandbox에서는 gRPC 5개가 bind 단계에서 실패할 수 있으므로 localhost bind 권한이 있는 환경에서 최종 판정한다.

### 12.3 asset 확인

```bash
cd /home/laptop3/gello_worktrees/hil-rl-learner
sha256sum third_party/hil-serl/examples/experiments/resnet10_params.pkl
sha256sum /tmp/gello-hil-rl-learner-assets/resnet10_params.pkl
```

두 결과는 모두 `175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b`여야 한다.

### 12.4 실제 checkpoint artifact 확인

```bash
find /tmp/actual-learner-resume-y_dfio4l -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
sed -n '1,220p' /tmp/actual-learner-resume-y_dfio4l/checkpoint_000000000001/metadata.json
sed -n '1,220p' /tmp/actual-learner-resume-y_dfio4l/checkpoint_000000000002/metadata.json
```

실제 agent 통합 검증을 실행한 one-off Python 본문은 저장소에 아직 없다. 따라서 위 artifact 검사는 결과 확인용이고 완전한 재현 명령은 아니다. Phase 0에서 opt-in integration script/test로 남기는 것이 첫 후속 작업이다.

## 13. 다음 작업자가 지켜야 할 경계

- 모든 변경은 `/home/laptop3/gello_worktrees/hil-rl-learner`와 `feat/hil-rl-learner`에서 이어간다.
- `/home/laptop3/gello_software` 주 workspace의 별도 dirty 변경을 이 branch 커밋에 섞지 않는다.
- `third_party/hil-serl`, `.proto`, generated protobuf 파일은 수정하지 않는다.
- checkpoint를 overwrite, 삭제, pruning하지 않는다.
- Kanu/robot에서 실행하기 전에 production wiring 미완료 항목을 먼저 해결한다.
- feature mode를 raw-pixel mode와 같은 checkpoint/buffer schema로 취급하지 않는다.
- encoder fingerprint가 다른 feature는 절대 한 buffer나 한 run에서 섞지 않는다.
- 실제 검증과 fake-agent/fake-W&B 단위 테스트를 문서와 로그에서 명확히 구분한다.

## 14. 병렬 검수 기록

이 문서는 다음 세 역할의 읽기 전용 병렬 검수 결과를 합쳤다.

1. 작업 인벤토리: worktree, branch, commit, 변경 파일, venv, asset, checkpoint artifact와 test count 확인
2. 코드·테스트 검수: 실제 updated agent checkpoint와 learner resume 후 continue 독립 검증, production wiring과 correctness gap 확인
3. feature-buffer 설계 검수: upstream encoder의 frozen/trainable 경계, augmentation, memory estimate, fingerprint와 migration 계약 확인

세 검수 모두 `third_party/hil-serl`과 작업 소스를 수정하지 않았다. 최종 전체 회귀 테스트는 별도로 localhost loopback 권한 환경에서 다시 실행해 `93 passed`를 확정했다.
