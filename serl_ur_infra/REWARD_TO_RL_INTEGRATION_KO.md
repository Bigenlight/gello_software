# reward classifier → 개입 → 학습 연결 계약

> 작성 2026-07-29 KST. **모든 내용은 코드에서 직접 추적해 확인했다** — 인용한 줄 번호는
> `feat/gello-ur7e-humble-22.04` 기준이며, 옮겨 적은 게 아니라 읽고 쓴 것이다.
>
> 대상 독자: **라이브 뷰어에서 검증된 분류기를 실제 RL 루프에 연결하는 사람.**
> 뷰어를 돌리는 방법은 [`REWARD_CLASSIFIER_LIVE_KO.md`](REWARD_CLASSIFIER_LIVE_KO.md),
> 전체 현황은 [`HANDOFF_NEXT_SESSION_KO.md`](HANDOFF_NEXT_SESSION_KO.md)에 있다.

---

## 🔴 먼저 — 지금 연결하면 reward가 틀린다

두 개가 먼저 고쳐져야 한다. 둘 다 [`HANDOFF_NEXT_SESSION_KO.md`](HANDOFF_NEXT_SESSION_KO.md) §7 A·B다.

| # | 문제 | 연결했을 때 벌어지는 일 |
| --- | --- | --- |
| **A** | **크롭 불일치** — 분류기는 무크롭으로 학습됐는데 액터 경로는 `IMAGE_CROP`을 적용한다 | 분포 밖 입력. recall@0.85가 **100% → 33%**. 성공을 거의 못 잡는다 |
| **B** | `checkpoint_sha256()`가 `os.path.isfile()`을 요구해 **orbax 디렉터리 체크포인트를 못 읽는다**. `DEFAULT_CLASSIFIER_CHECKPOINT_SHA256`은 아직 recall 0%짜리 은퇴 모델을 가리킨다 | 1만 고치고 2를 빼먹으면 **서버가 조용히 뜨고 reward가 영구 0**이 된다 |

**라이브 뷰어가 잘 나오는 것은 이 둘과 무관하다.** 뷰어와 gRPC 경로는 다른 그림을 본다
(`HANDOFF_NEXT_SESSION_KO.md` §6). 뷰어 화면을 근거로 "연결만 하면 된다"고 판단하지 마라.

---

## 1. 전체 사슬 — 한 스텝에 무슨 일이 일어나나

```
[랩톱]                                          [kanu]
 정책 액션 ──┐
             ├─→ GelloIntervention wrapper ─→ 실행 액션 ─→ UR7e
 사람 GELLO ─┘         (사람이 잡으면 치환)
                              │
                              ├─ info["intervened"]      = 0 또는 1
                              └─ info["intervene_action"] = 사람 액션 (개입 시에만)
                              │
                     remote_actor.py 가 transition 조립
                     (저장되는 action = 실행된 액션)
                              │
                         gRPC Step ──────────────────────→ RewardTransitionFinalizer
                                                                   │
                                              next_observations 로 분류 → rewards 덮어쓰기
                                                                   │
                                                            ReplayIngress
                                                              ├─→ replay 버퍼      (항상)
                                                              └─→ intervention 버퍼 (개입일 때 추가로)
                                                                   │
                                                         learner: RLPD 50:50 샘플링
```

각 단계의 정확한 계약은 아래에 있다.

---

## 2. 개입 — 사람이 잡았을 때 저장되는 것

### 2.1 wrapper가 붙이는 두 필드

`ur_env/envs/wrappers.py:354-356`:

```python
info["intervene_action"] = np.asarray(new_action).copy()   # 사람 액션, env 단위 [-1,1]^7
info["intervened"] = int(replaced)
```

`intervene_action`은 **개입했을 때만 존재한다.** `intervened`는 항상 0 또는 1로 들어간다.

### 2.2 저장되는 액션은 "실행된" 액션이다 — 정책이 제안한 것이 아니다

`ur_env/remote_actor.py:107-121`:

```python
has_intervention_action = "intervene_action" in info
intervened_value = info.get("intervened", int(has_intervention_action))
...
intervened = int(bool(intervened_value))
if bool(intervened) != has_intervention_action:
    raise ActorProtocolError("intervened label and intervene_action presence are inconsistent")
executed_action = validate_action(
    info["intervene_action"] if intervened else requested_action, ...
)
```

두 가지를 확인하라:

1. **버퍼에 들어가는 것은 `executed_action`이다.** 사람이 개입한 스텝에서는 사람의 액션이
   저장된다. 정책의 제안은 버려진다. 이게 HIL-SERL이 개입을 demo로 쓸 수 있는 이유다.
2. **라벨과 실체의 교차 검증이 강제된다.** `intervened=1`인데 `intervene_action`이 없거나,
   그 반대이면 `ActorProtocolError`로 즉시 죽는다. **조용히 틀리지 않는다** — 이 리포에서
   드문 좋은 성질이니 새 코드를 붙일 때 깨뜨리지 마라.

### 2.3 🔴 개입 transition도 분류기 reward를 **똑같이** 받는다

**특별 취급이 없다.** 사람이 실패를 수습해서 큐브를 컵에 넣어도, 분류기가 성공으로 판정하지
않으면 그 스텝의 reward는 **0**이다. 반대로 사람이 개입 중이어도 분류기가 성공을 때리면
reward 1이 붙고 에피소드가 끝난다.

즉 **개입은 "무엇을 했는가"(액션)를 가르치지 "그것이 좋았는가"(보상)를 가르치지 않는다.**
보상 신호는 전적으로 분류기 몫이다. 분류기 recall이 낮으면 사람이 아무리 잘 개입해도
그 성공은 보상으로 기록되지 않는다.

---

## 3. reward / termination 계약 — 서버가 최종 권위

`ur_env/rlpd_receive_server.py:556-620`, `RewardTransitionFinalizer`.

### 3.1 분류 대상은 `next_observations`다

```python
result = self.classifier.classify(transition["next_observations"])
```

**액션 이전이 아니라 이후 관측을 본다.** "이 액션을 한 결과 성공 상태가 되었는가"를 묻는다.
`next_observations`가 없으면 `ActorProtocolError`로 거부한다.

→ **그래서 크롭 불일치가 정확히 여기서 문다.** 액터가 `ur7e_env.get_im()`으로 만든
크롭된 관측이 그대로 분류기에 들어간다.

### 3.2 reward는 무조건 덮어써진다

```python
transition["rewards"] = 1.0 if result.success else 0.0
```

코드 주석이 의도를 못박아 뒀다:

> Reward is classifier-authoritative in both directions. Local may propose episode
> terminal/truncation semantics, but it cannot inject a positive reward when the
> server classifier is negative.

액터가 보낸 `reward` 값은 **읽히지만 버려진다.** 랩톱에서 reward를 만들어 넣으려는 시도는
전부 무의미하다. 사람이 수동으로 성공을 인정하고 싶다면 **분류기 경로를 거쳐야 한다** —
현재 그런 우회로는 구현돼 있지 않다(§6 참조).

### 3.3 성공이면 에피소드가 끝난다 — 그리고 terminal로 부트스트랩된다

```python
if result.success:
    transition["masks"] = 0.0
    transition["dones"] = True
    transition["truncated"] = False
```

세 줄 전부 중요하다:

| 필드 | 의미 |
| --- | --- |
| `dones = True` | 에피소드 종료 |
| `masks = 0.0` | **부트스트랩 차단** — 이 상태의 가치를 0으로 본다 (진짜 terminal) |
| `truncated = False` | 로컬이 같은 스텝에서 시간 제한 truncation을 걸었어도 **분류기 성공이 이긴다** |

**이것이 false positive가 치명적인 진짜 이유다.** 오탐은 "보상을 잘못 준다"에서 끝나지 않는다:

1. 실패 상태에 **reward 1**을 준다
2. **에피소드를 그 자리에서 끝낸다** — 로봇이 회복할 기회를 뺏는다
3. `masks=0.0`으로 **그 실패 상태를 terminal로 부트스트랩**한다 — 가치 함수에 "여기가 목표다"라고 새긴다
4. 그 transition은 **리플레이 버퍼에 영구히 남는다**

반대로 false negative는 에피소드가 계속되고 reward만 0일 뿐이다. threshold를 0.2로 내린
근거가 이 비대칭이다 → [`REWARD_CLASSIFIER_THRESHOLD_KO.md`](REWARD_CLASSIFIER_THRESHOLD_KO.md).

### 3.4 서버가 transition에 남기는 감사 필드

```python
transition["classifier_evaluated"]   = np.uint8(1)
transition["classifier_probability"] = float(result.probability)
transition["classifier_threshold"]   = float(result.threshold)
transition["classifier_success"]     = np.uint8(result.success)
transition["reward_model_id"]        = result.reward_model_id
```

**사후 분석에 이걸 쓰라.** 확률이 저장되므로 "threshold를 X로 했으면 어땠을까"를 학습을
다시 돌리지 않고 계산할 수 있다. `reward_model_id`가 있어 어느 체크포인트가 그 보상을
만들었는지도 남는다.

`ReplayIngress`는 삽입 시 `classifier_success != (probability > threshold)`이면 거부한다
(`:1283`). 확률과 판정이 어긋난 transition은 버퍼에 못 들어간다.

---

## 4. 버퍼 라우팅 — 개입은 **이중 기록**이다 (택일이 아니다)

`ur_env/rlpd_receive_server.py:890-930`. 가장 오해하기 쉬운 부분이다.

```python
if not route.replay_inserted:
    ...  # 모든 transition이 replay_store로
    route.replay_inserted = True

if intervened and not route.intervention_inserted:
    ...  # 개입 transition은 intervention_store에 '추가로'
    route.intervention_inserted = True

route.completed = route.replay_inserted and (route.intervention_inserted or not intervened)
if not route.completed:
    raise ReceiveRuntimeError("transition routing did not complete")
```

| transition | replay 버퍼 | intervention 버퍼 |
| --- | --- | --- |
| 정책 자율 | ✅ | ❌ |
| 사람 개입 | ✅ | ✅ **(같은 것이 양쪽에 들어간다)** |

**개입 transition은 두 버퍼 모두에 있다.** 따라서:

- `replay_size`는 총 스텝 수다. `intervention_size`는 그 부분집합이다.
- `SamplingMetrics.intervention_ratio = intervention_size / replay_size` — 개입 비율이
  그대로 나온다 (`learner/batches.py:41-45`).
- 용량이 다르다: replay `50,000`, intervention `10,000` (`DEFAULT_*_CAPACITY`).
  **개입 비율이 20%를 넘으면 intervention 버퍼가 먼저 덮어쓰기를 시작한다.**

---

## 5. 학습 배치 — RLPD 50:50과 그 안의 비례 분할

`ur_env/learner/batches.py:366-435`. 파일 첫 줄이 `"""RLPD 50:50 sampling..."""`이다.

```python
half = self.batch_size // 2
offline_count, intervention_count = proportional_sample_counts(
    half, (offline_size, intervention_size)
)
```

```
배치 (batch_size)
├── 1/2  online_replay            ← 정책이 실제로 겪은 것 전부
└── 1/2  "demo" 쪽
     ├── offline_demos        ┐
     └── online_interventions ┘  두 버퍼 크기에 비례해서 나눔
```

### 연결 담당자가 알아야 할 결과

1. **오프라인 demo가 최소 1개 없으면 학습이 시작조차 안 된다.**
   `ready`는 `len(online_replay) >= training_starts` **그리고** `len(offline_demos) > 0`을
   요구한다 (`:360-365`). demo가 0이면 `LearnerBatchError`.
   `composition.py:289`에도 "at least one canonical offline demonstration is required"가 있다.

2. **사람이 개입할수록 오프라인 demo가 희석된다.** demo 절반이 크기 비례로 나뉘므로,
   intervention 버퍼가 커질수록 offline demo가 차지하는 몫이 줄어든다. 초반엔 demo가
   지배하고 개입이 쌓이면 뒤집힌다. **의도된 동작이지만 개입 품질이 곧 학습 신호 품질이 된다.**

3. **`cta_ratio = 2`로 고정돼 있다** (`learner/config.py:57-58`, 2가 아니면 `ValueError`).
   `gradient_step == learner_step * cta_ratio`가 체크포인트에서도 검증된다.

---

## 6. 사람이 수동으로 성공을 주는 경로는 **아직 없다**

운영 판단으로 "분류기가 성공을 놓치면 사람이 수동으로 준다"고 정해져 있다
([`REWARD_CLASSIFIER_THRESHOLD_KO.md`](REWARD_CLASSIFIER_THRESHOLD_KO.md) 최종 결정 절).
**그 경로는 현재 코드에 구현돼 있지 않다.**

- §3.2대로 reward는 서버 분류기가 유일한 출처다. 액터가 보낸 reward는 버려진다.
- `/reward_classifier/status`(라이브 뷰어)를 구독하는 것은 **GUI 3개뿐**이다
  (`policy_run_gui_node.py`, `classifier_view_gui_node.py`, 그리고 노드 자신의 문서화).
  **RL 경로에는 소비자가 없다.** 뷰어를 학습에 연결하는 배선은 존재하지 않는다.

구현한다면 설계 선택지는 대략 이렇다 — **아직 결정된 바 없다**:

| 방식 | 장점 | 위험 |
| --- | --- | --- |
| 액터가 `meta`에 `operator_success` 플래그를 넣고 서버가 OR | 기존 계약 최소 변경 | 서버 권위 원칙(§3.2)에 구멍. 감사 필드에 출처 구분 필요 |
| 서버 쪽 수동 오버라이드 RPC | 권위가 서버에 남음 | 새 RPC + proto 재생성 |
| 사후 라벨링 (버퍼 재작성) | 실시간 부담 0 | 이미 학습에 쓰인 transition은 못 되돌림 |

어느 쪽이든 **`classifier_success`와 구분되는 필드로 남겨야 한다.** 지금 스키마는 성공의
출처가 분류기 하나뿐이라고 가정하고, `ReplayIngress`의 `:1283` 교차검증도 그 가정 위에 있다.

---

## 7. 연결 순서 — 이 순서대로

전제: §0의 A·B가 끝났고, 크롭 정합 체크포인트로 재학습 + threshold 스윕 재실행이 됐다.

```
1. kanu: receive server를 zero-action 목으로 띄운다
   → 랩톱에서 SSH 터널 50153 -> 50053
   → 확인: GetServerInfo / GetBufferStatus 가 응답하고 reward_model_id 가 새 체크포인트

2. 액터 preflight만 (액터 미기동, 완전 안전)
   → 확인: 9종 preflight 전부 통과

3. fake-env로 왕복 (로봇·센서 미사용)
   → 확인: replay_size 가 증가, classifier_evaluated=1, probability 가 0/1 극단이 아님

4. 실센서 + DRY_RUN (팔 안 움직임)
   → 확인: 라이브 뷰어와 서버 probability 를 나란히 본다.
     ⚠️ 크롭이 맞았다면 두 값이 비슷해야 한다. 크게 다르면 전처리가 아직 어긋난 것이다.
     이게 크롭 정합을 실기에서 검증하는 가장 싼 방법이다.

5. 개입만 검증 (--arm 없이)
   → 확인: GELLO 를 잡았다 놓으면 intervention_size 가 증가하고 replay_size 도 같이 증가
     (§4 이중 기록). 둘 중 하나만 늘면 라우팅이 깨진 것이다.

6. --arm 추가 (팔이 실제로 움직인다)
   → 확인: 성공 시 dones=True 로 에피소드가 끝나는가. 오탐이 한 번이라도 보이면 즉시 중단.
```

세부 절차와 실제 명령은 [`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)와
[`HANDOFF_NEXT_SESSION_KO.md`](HANDOFF_NEXT_SESSION_KO.md) §7 D에 있다.

**5번은 이 문서가 새로 넣는 단계다.** 기존 런북은 개입 라우팅을 따로 검증하지 않는다.

---

## 8. 연결하면서 지켜볼 것

| 신호 | 어디서 | 정상 | 이상하면 |
| --- | --- | --- | --- |
| `classifier_probability` 분포 | transition 감사 필드 | 성공 구간은 0.9+, 실패는 0.01 미만 | 전부 0.5 근처 → 체크포인트 미로드(§0-B) 또는 이미지 미도달 |
| `intervention_size / replay_size` | `GetBufferStatus` | 개입 비율. 20% 넘으면 intervention 버퍼 덮어쓰기 시작 | 지속적으로 높으면 정책이 아직 못 하는 것 |
| `replay_size` vs `intervention_size` 증가 패턴 | `GetBufferStatus` | 개입 시 **둘 다** 증가 | 하나만 증가 → §4 라우팅 손상 |
| `reward_model_id` | `GetServerInfo` | 새 크롭 정합 체크포인트 | 옛 값이면 §0-B의 SHA 갱신 누락 |
| `dones=True` 발생 시점 | 에피소드 로그 | 사람이 봐도 성공인 순간 | 실패 상태에서 나면 **즉시 중단**, threshold 0.5로 되돌림 |

---

## 9. 관련 문서

| 문서 | 무엇 |
| --- | --- |
| [`REWARD_CLASSIFIER_LIVE_KO.md`](REWARD_CLASSIFIER_LIVE_KO.md) | 라이브 뷰어 실행 (사람 확인용, RL 경로 아님) |
| [`REWARD_CLASSIFIER_THRESHOLD_KO.md`](REWARD_CLASSIFIER_THRESHOLD_KO.md) | threshold 0.2 근거, 측정 조건, 07-29 누출 감사 |
| [`HANDOFF_NEXT_SESSION_KO.md`](HANDOFF_NEXT_SESSION_KO.md) | 전체 현황. §6 두 경로 대조, §7 할 일 순서 |
| [`../docs/testing/08_OPEN_GAPS.md`](../docs/testing/08_OPEN_GAPS.md) | G15 크롭 불일치 전문 |
| [`../docs/testing/04_HIL_INTERVENTION.md`](../docs/testing/04_HIL_INTERVENTION.md) | 개입 루프·좌표계·메타데이터 |
| [`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md) | 액터 기동 절차 |
| [`REMOTE_ACTOR_GRPC.md`](REMOTE_ACTOR_GRPC.md) | gRPC 전송 계약 v2 (영문) |

### 코드 진입점

```
ur_env/envs/wrappers.py:354-356          개입 라벨 부착 (intervene_action / intervened)
ur_env/remote_actor.py:107-121           실행 액션 선택 + 라벨 교차검증
ur_env/rlpd_receive_server.py:556-620    RewardTransitionFinalizer — reward 권위
ur_env/rlpd_receive_server.py:890-930    버퍼 이중 라우팅
ur_env/rlpd_receive_server.py:1283       classifier_success ↔ probability 교차검증
ur_env/learner/batches.py:366-435        RLPD 50:50 + demo 비례 분할
ur_env/learner/config.py:57-58           cta_ratio 고정
```
