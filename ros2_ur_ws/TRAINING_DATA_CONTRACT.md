# Visual policy training / inference contract

이 문서는 `gello_software_jazzy`에서 visual BC/IFQL/SVF를 학습하거나 ROS2 실물 inference를 실행할 때 먼저 확인해야 하는 공용 주의사항이다.

## 가장 중요한 문제: periodic joint angle branch

UR joint angle은 주기값이므로 같은 물리 자세가 `+pi`와 `-pi` 근처의 서로 다른 숫자로 표현될 수 있다.

```text
+3.12 rad ~= -3.16 rad
```

하지만 policy는 숫자와 normalized state를 입력으로 받으므로, dataset과 live ROS state의 branch가 다르면 같은 자세를 전혀 다른 state로 본다. 특히 q1/q6가 약 `2*pi`만큼 어긋나면 첫 action부터 반대 방향 또는 큰 joint motion이 나올 수 있다.

실제 carrot 사례에서는 학습 데이터가 q1/q6를 `+pi` branch로 저장했고, ROS live state는 `-pi` branch를 사용했다. 두 encoder가 모두 같은 방향으로 실패했지만 loss/checkpoint/server warm-up은 정상이었으므로 encoder보다 state-coordinate mismatch가 원인이었다.

## 학습/실행 전 필수 체크

- dataset과 ROS live state의 joint 순서가 같은가?
- q1/q6를 포함한 periodic joint의 branch가 같은가?
- train/val/inference의 `state_coordinate`가 같은가?
- proprio `mean/std`가 현재 dataset으로 fit됐는가?
- action `lo/hi`, horizon, stride 및 unwrap branch가 serving과 같은가?
- image encoder를 의심하기 전에 raw state와 norm 복원 state를 비교했는가?
- checkpoint마다 recorded JPEG + state 첫 action smoke를 했는가?

## canonicalization 규칙

현재 UR/GELLO carrot 기준은 `state_coordinate=ur_wrapped_v1`이다.

```text
live ROS state
  -> training branch로 canonicalize
  -> proprio normalization
  -> policy inference
  -> action denormalization
  -> live ROS branch 근처로 action unwrap
  -> robot
```

**action만 unwrap하면 안 된다.** policy 입력 state도 training coordinate로 변환해야 한다. 둘 중 하나만 처리하면 policy가 이미 잘못된 state에서 action을 계산한다.

## 진단 순서

두 encoder/policy가 같은 방향으로 동시에 실패하면 다음 순서로 확인한다.

1. raw state와 norm stats로 복원한 state 비교
2. live state와 training state의 periodic joint 차이 확인
3. proprio/action norm stats, horizon/stride, action branch 확인
4. recorded-frame server replay로 첫 action과 max joint deviation 확인
5. 그 뒤에 encoder artifact, feature shape, loss, checkpoint 조사

학습 loss가 정상적으로 내려가도 live state가 학습 분포와 같다는 보장은 없다. 따라서 finite loss만으로 real inference를 검증했다고 판단하지 않는다.

## 기록 규칙

새 학습/실험에는 dataset path, source revision, `state_coordinate`, norm stats, encoder artifact hash, augmentation recipe, checkpoint interval, recorded smoke 결과를 남긴다. 실제 robot task success를 측정하지 않았으면 `real validated`라고 쓰지 말고 `recorded-frame smoke` 또는 `serving contract check`라고 명시한다.
