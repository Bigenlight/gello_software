# Offline RL 구조 — IFQL부터

이 패키지는 고정 데이터로 학습하는 별도 경로의 골격이다. 첫 알고리즘으로
IFQL의 실제 loss/update와 best-of-N 선택까지 구현한다. 온라인 RLPD의
`LearnerWorker`, replay ingress, robot/gRPC에는 연결하지 않는다.

## 구성

```text
ChunkDataset.sample_chunks(B, H)
                 │
             ChunkBatch
                 │
     OfflineTrainer.train_once()
                 │
          IFQL.update(batch)
            ├─ critic.update()  Q + V, target Q
            └─ actor.update()   flow matching

추론: actor.sample_candidates() → critic.score() → 행마다 argmax
```

| 파일 | 책임 |
| --- | --- |
| `batch.py` | chunk shape, reward 누적, bootstrap, padding/종료 경계 검증 |
| `interfaces.py` | Actor / Critic / Algorithm / ChunkDataset 계약 |
| `actor.py` | 주입된 flow network를 FM loss로 업데이트 |
| `aggregation.py` | 알고리즘이 공유하는 Q ensemble 집계 함수 |
| `vision.py` | 공통 frozen feature의 flatten 또는 spatial softmax 변환 |
| `models.py` | MLP policy/Q/V와 `create_feature_ifql` 초기화 함수 |
| `normalization.py` | 학습 통계 추정, 공통 정규화/역변환, 통계 저장·복원 |
| `ifql.py` | Q/V 업데이트 순서, expectile/TD loss, target 갱신, 후보 선택 |
| `trainer.py` | 고정 데이터에서 한 배치를 받아 알고리즘을 한 번 업데이트 |

`actor.update()` / `critic.update()`는 새 객체를 반환한다. JAX/Flax의 불변 상태
방식이며, 전달한 객체를 제자리 변경하지 않는다. RNG는 호출자가 매번 새 키를
전달하고 데이터 샘플러는 자체 RNG를 관리한다.

## 첫 구현: IFQL

`~/repos/fmrl/agents/ifql.py`의 알고리즘 구조를 참고했다.

- V: 집계한 target Q와 V의 차이에 대한 expectile loss. 기본 집계는 `mean - rho * std`다.
- Q: `return + bootstrap_discount * V(bootstrap_observation)`에 대한 MSE.
- Actor: 행동 데이터에 대한 FM loss. Q gradient/advantage weight를 적용하지 않는다.
- 추론: FM 후보 N개를 만들고 집계한 Q가 가장 높은 chunk를 선택한다. 기본 집계는 `min`이다.

### Critic 설정

```python
config = IFQLConfig(
    num_qs=10,
    kappa=0.9,
    rho=0.5,
    value_q_aggregation="mean_minus_std",
    selection_q_aggregation="min",
    target_tau=0.005,
    num_candidates=32,
)
```

| 설정 | 의미 |
| --- | --- |
| `num_qs` | Q ensemble 개수. 주입하는 Q network도 이 개수로 초기화해야 한다 |
| `kappa` | V 학습의 expectile 계수, `0 < kappa < 1` |
| `rho` | `mean_minus_std` 집계의 표준편차 계수, 0 이상 |
| `value_q_aggregation` | V 학습에 사용하는 target Q 집계 |
| `selection_q_aggregation` | 추론에서 후보 행동을 선택하는 online Q 집계 |

두 aggregation 설정은 각각 `min`, `mean`, `max`, `mean_minus_std`를 선택할 수 있다.
`rho`는 `mean_minus_std`에서만 사용하고, 표준편차는 `ddof=0`으로 계산한다.
`num_qs=1`이면 표준편차는 0이다. 여러 집계 방식을 실험할 수 있도록 제공하는 것이며,
각 설정의 실데이터 성능은 검증하지 않았다.

이전 골격의 `expectile`/`pessimism` 설정 이름은 원본 fmrl과 같은 `kappa`/`rho`로
바꿨다. IFQL의 Q target은 `V(s')`이므로 `target_q_aggregation` 설정을 별도로 두지 않는다.
향후 QAM/endcorr는 `aggregate_q`를 재사용하고 각 알고리즘 설정에 추가할 수 있다.

`num_qs`는 이미 만들어진 network의 크기를 바꾸는 명령이 아니다. 초기화 때 적용하고,
업데이트와 점수 계산에서 출력이 `(num_qs,B)`인지 검사한다. 설정과 network가 다르면
오류를 낸다. 개수를 바꿔 재학습하려면 Q network와 target, optimizer도 다시 생성해야 한다.

Q/V loss의 target은 둘 다 **업데이트 전 snapshot**으로 계산한다.
Q/V를 업데이트한 뒤 target Q를 Polyak 갱신하고 FM actor를 업데이트한다.
원본 fmrl의 `target_update`는 `self.network`의 이전 Q를 참조하지만, 여기서는
**새 Q 파라미터**를 추적한다. 이 차이는 의도적으로 명시하고 테스트한다.

아키텍처를 고정하지 않도록 Q/V는 Flax `TrainState`로 주입한다:

```python
# q_model(obs_tree, actions[B,H,A]) -> (ensemble, B)
# v_model(obs_tree) -> (B,)
# flow_model(obs_tree, noisy_actions[B,H,A], time[B]) -> (B,H,A)
actor = FlowActor(flow_state, sample_fn)
critic = IFQLCritic(q_state, v_state, target_q_params=q_state.params, config=config)
algorithm = IFQL(actor, critic)
trainer = OfflineTrainer(algorithm, batch_size=256, horizon=16)
trainer, metrics = trainer.train_once(dataset, rng=step_key)
```

위 코드는 직접 network를 주입하는 구성 예시다. `dataset` adapter는 아직 없다.
MLP head를 초기화하는 기본 경로는 아래 `create_feature_ifql`이다.
기존 `learner/flow_matching.py`의 actor를 별도로 주입하려면 다음 adapter로 사용할 수 있다:

```python
sample_fn = lambda state, obs, rng: sample_action_chunks(
    flow_model, state.params, obs, rng, discretize_gripper=True,
)
```

FM 모델의 params는 `flow_state`에 전달할 수 있다. Q/V와 optimizer를 포함하는
offline RL checkpoint는 기존 FM-init / SAC checkpoint와 별도 형식으로 설계해야 한다.

## 공통 vision 입력과 MLP head

기존 frozen ResNet10 feature는 pooling 전 `(B,1,4,4,512)` 출력이다.
spatial softmax는 적용되어 있지 않다. 새 offline 경로에서는 다음처럼 선택한다:

```python
from ur_env.offline_rl.models import MLPConfig, create_feature_ifql
from ur_env.offline_rl.ifql import IFQLConfig
from ur_env.offline_rl.vision import VisionConfig

algorithm = create_feature_ifql(
    example_observations,  # cam1/cam2: (B,1,4,4,512), state: (B,1,19)
    rng=init_key,
    vision=VisionConfig(pooling="spatial_softmax", spatial_softmax_temperature=1.0),
    mlp=MLPConfig(hidden_dims=(256, 256), horizon=16, integration_steps=8),
    ifql=IFQLConfig(num_qs=10),
)
```

기본 `pooling="flatten"`은 두 카메라와 state를 합쳐 16,403차원이다.
`pooling="spatial_softmax"`는 채널마다 공간 방향 softmax 후 기대 x/y 좌표를 계산해
카메라당 1,024차원, state 포함 2,067차원이 된다. 좌표 범위는 `[-1,1]`이며
temperature는 학습하지 않는 고정 설정이다. 공간 softmax는 feature 크기 정보를
좌표로 요약하므로 flatten과 정보가 같지는 않다.

`FeatureIFQL`은 현재/다음 관측을 각각 한 번 변환하고, 같은 벡터를 policy/Q/V에 전달한다.
MLP 입력은 policy `(vector, noisy chunk, time embedding)`, Q `(vector, chunk)`,
V `(vector)`다. 각 head와 Q ensemble의 각 MLP는 독립 파라미터를 가진다.
pretrained backbone은 optimizer에 포함되지 않는다. factory는 raw image를 처리하거나
pretrained 가중치를 내려받지 않으므로 기존 extractor의 cached feature를 공급해야 한다.
이 MLP policy는 기존 spatial learned embedding 기반 FM 모델과 구조가 달라
기존 FM-init 가중치를 그대로 복원할 수 없다.

### 입출력 normalization

| 대상 | 현재 처리 |
| --- | --- |
| Raw RGB → 기존 ResNet | `(pixel / 255 - ImageNet mean) / ImageNet std` |
| Cached vision feature | pooling 후 `none` / `mean_std` / `min_max` / `quantile` 선택 |
| 19차원 state | `none` / `mean_std` / `min_max` / `quantile` 선택 |
| MLP hidden layer | 기본 `layer_norm=True`, Dense 뒤 LayerNorm 적용 |
| 학습 action | canonical `[-1,1]` action에 선택한 통계 정규화를 추가 적용 |
| FM velocity | 범위를 제한하지 않음. action과 Gaussian noise 차이를 학습 |
| 샘플링 action | 적분 → 역변환 → canonical `[-1,1]` 제한 → gripper `-1/+1` 변환 |
| Reward | 선택한 `reward_scale * r + reward_bias` 적용 |
| Q/V 출력 | scalar 범위 제한 없음. 선택한 reward 변환의 단위로 학습 |

추가 정규화는 전부 기본 `none`이며 reward는 기본 scale=1, bias=0이다.
pretrained 이미지 전처리는 backbone 계약을 그대로 따른다. 이미 추출된 feature에
ImageNet 정규화를 다시 적용하거나 pretrained 전처리를 임의로 끄지 않는다.

```python
from ur_env.offline_rl.normalization import (
    Normalization, NormalizationConfig, fit_training_normalization,
)

norm_config = NormalizationConfig(
    state="mean_std",
    features="none",
    actions="mean_std",
    normalize_state_gripper=False,
    normalize_action_gripper=False,
    quantile_low=0.01,
    quantile_high=0.99,
    epsilon=1e-6,
    reward_scale=1.0,
    reward_bias=0.0,
)
vision = VisionConfig(pooling="flatten")
normalization = fit_training_normalization(
    train_observations, train_actions,  # 학습 split의 데이터만 전달
    config=norm_config, vision=vision,
    valid_mask=train_action_mask,      # padding된 action은 통계에서 제외
)
algorithm = create_feature_ifql(
    example_observations, rng=init_key, vision=vision,
    normalization=normalization,
)
normalization.save("normalization.json")  # 이미 있으면 덮어쓰지 않고 오류
restored = Normalization.load("normalization.json")
```

정규화를 끄려면 `create_feature_ifql`의 `normalization` 인자를 생략한다.
`MLPConfig(layer_norm=False)`로 MLP hidden LayerNorm도 독립적으로 끌 수 있다.

- `mean_std`: 차원별 `(x - mean) / std`.
- `min_max`: 차원별 min/max가 `[-1,1]`이 되도록 affine 변환.
- `quantile`: 지정한 하위/상위 분위수를 `[-1,1]`로 변환. 기본 1%/99%.
  이상치를 clip하지 않으므로 범위 밖 값도 역변환할 수 있다.
- 분산이나 범위가 epsilon보다 작으면 scale=1로 두어 미세한 잡음을 증폭하지 않는다.
- gripper는 기본적으로 제외하며 각각의 boolean 옵션으로 포함할 수 있다.
- state/feature 통계용 관측은 padding 없이 서로 다른 학습 관측을 전달한다.
  action은 `(N,7)` 또는 `(B,H,7)`이고 시간 위치 전체가 같은 7개 통계를 공유한다.
  겹치는 chunk로 통계를 내면 중복 action이 가중되므로 가능한 한 원본 transition을 사용한다.

현재/다음 관측과 actor/Q/V 및 추론에 같은 고정 통계를 사용한다. 업데이트 중 통계를
재추정하지 않으며, factory가 example batch에서 임의로 fit하지 않는다. 학습/검증 split
선택은 데이터 adapter의 책임이다. 통계에는 pooling 방식과 temperature도 저장하고,
모델 초기화 때 다른 vision 설정이 들어오면 오류를 낸다.

Action은 canonical 공간과 모델 공간을 구분한다. z-score 후에는 `[-1,1]`을 넘을 수
있으므로 모델 공간에서 clip하지 않는다. 후보를 역변환한 뒤 실행 범위와 gripper를
처리하고, 그 후보를 다시 모델 공간으로 바꿔 Q로 평가한다. 외부 반환값은 canonical
action이다. Padding은 정규화 후에도 0으로 유지한다.

Reward 변환은 chunk 누적과 일치하도록
`scale * return + bias * sum_i gamma^i`로 적용한다. `make_chunk_batch`가
`reward_discount_sum`을 기록하므로 terminal의 bootstrap이 0이어도 올바른 합을 쓴다.
이 필드 없이 reward bias를 켜면 오류를 낸다. Reward bias는 episode 길이에 따라
문제의 목적도 바꿀 수 있으므로 기본 0이며, Q/V의 출력 activation은 제한하지 않는다.

정규화 JSON은 모델 가중치와 함께 보관하고 복원 시 위처럼 전달해야 한다.
전체 optimizer/RNG/model checkpoint 작성은 여전히 후속 구현 항목이다.

기존 canonical demo 변환은 tool frame의 이동/회전 delta를 task의 position/rotation
scale로 나눈다. 현재 기본값은 각각 `0.0125 m`, `0.0625 rad`다. 그 뒤 translation과
rotation의 norm을 각각 제한한다. 새 offline 모듈은 이미 정규화된 action을 받아
다시 나누지 않는다. 모델 출력의 실제 로봇 단위 복원과 실행은 배포 계층의 책임이다.

LayerNorm은 데이터셋 통계로 입력을 정규화하는 것과 다르다. state에는 pose, force,
torque, velocity처럼 단위가 다른 값이 함께 있다. 통계 정규화를 사용할 때는
학습 split에서만 통계를 구하고, 해당 통계를 checkpoint와 함께 저장하여
추론에도 동일하게 적용해야 한다.

## Chunk 계약

`make_chunk_batch`는 **이미 에피소드 단위로 나눈** 시퀀스를 받는다.
파일이나 pickle을 읽거나 episode ID를 추측하는 작업은 데이터 adapter가 담당한다.

- action은 `(B,H,A)`, valid mask는 `(B,H)`다. 유효 부분은 처음부터 연속되며 비어 있지 않아야 한다.
- reward는 각 스텝의 값을 전달하고, `sum_i gamma^i * reward_i`를 계산한다.
- `bootstrap_observations`는 마지막 유효 action 실행 후, reset 전의 관측이다.
- true terminal이면 bootstrap discount를 0으로 설정한다.
- truncation이면 `gamma^k`로 bootstrap한다. `k`는 유효 스텝 수다.
- terminal/truncation 이후에 다른 유효 action이 이어지는 chunk는 거부한다.
- 초기 IFQL의 Q/V는 **H스텝 모두 유효한 행만** 학습한다.
  짧은 마지막 chunk는 FM의 유효 위치만 학습하고, Q/V 학습에서는 제외한다.
  배치 전체가 짧은 chunk라면 Q/V, target, optimizer 상태를 갱신하지 않는다.

이 제한은 padding된 행을 H개의 행동을 모두 실행한 Q로 학습하는 것을 방지한다.
모든 episode가 H보다 짧으면 critic이 전혀 학습되지 않으므로 데이터에 맞는 H가 필요하다.
고정 H의 Q 평가와 실제로 chunk의 앞부분 몇 스텝을 실행할지는 별도의 계약이다.
가변 길이 chunk의 Q/V와 receding-horizon 실행의 가치 평가는 후속 설계 항목이다.

## 후속 구현 항목

- 데이터 adapter: canonical demo의 연속성, episode 경계, 동일한 좌표계의 action,
  reward, terminal/truncation, bootstrap 관측을 확인하여 `ChunkDataset`을 구현한다.
- 기본 MLP 초기화는 구현되어 있다. 실제 pretrained feature 추출과 데이터 adapter를 연결한다.
- 장시간 학습 CLI, optimizer/RNG를 포함하는 checkpoint, 평가 및 로그를 추가한다.
- QAM/endcorr: 새로운 `Algorithm`을 구현하여 공통 trainer에서 호출한다.
  critic target과 actor loss를 모두 교체할 수 있다. `context`에 후보와 flow 궤적을
  전달하여 endcorr의 후보 풀을 재사용한다. 업데이트 순서는 알고리즘이 결정한다.

현재 QAM/endcorr와 실데이터 학습 CLI는 미구현이다.
실데이터 학습 품질, GPU 성능, 실기 동작은 미검증이다.

## CPU 검증

리포지토리 최상위 경로에서 실행한다:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=serl_ur_infra \
  conda run -n il python -m unittest discover \
  -s serl_ur_infra/tests -p 'test_offline*.py' -v
```

합성 배치 업데이트, TD/expectile 수치, target 갱신, terminal/truncation 처리,
짧은 마지막 chunk, 배치별 후보 선택, 기존 FM sampler 연결을 검증한다.

실제 MLP head로 합성 feature 고정 배치를 20회 학습하는 smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  conda run -n il python serl_ur_infra/scripts/smoke_offline_ifql.py --pooling flatten
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  conda run -n il python serl_ur_infra/scripts/smoke_offline_ifql.py --pooling spatial_softmax
```

정규화 옵션을 켜서 확인하려면:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  conda run -n il python serl_ur_infra/scripts/smoke_offline_ifql.py \
  --pooling flatten --state-normalization mean_std --action-normalization mean_std \
  --reward-scale 2 --reward-bias -1
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  conda run -n il python serl_ur_infra/scripts/smoke_offline_ifql.py \
  --pooling spatial_softmax --state-normalization min_max \
  --feature-normalization mean_std --action-normalization quantile
```

`il` 환경 CPU에서 단위 테스트 26개와 위 정규화 조합 각각 20회 업데이트를 통과했다.
policy/Q/V 파라미터 변경, 유한한 loss/파라미터, 추론 action shape/범위를 확인했다.
이는 학습 계산 경로의 검증이며, pretrained 추출이나 실데이터 학습 성능 검증은 아니다.
