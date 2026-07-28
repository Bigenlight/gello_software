# reward threshold를 0.85에서 0.5로 낮춘 근거

> 측정일: 2026-07-28 KST · 측정 위치: Kanu GPU · 인터프리터
> `~/workspace/youngwoong/hil-serl/.venv-train/bin/python` (JAX/JAXLIB 0.5.3, Flax 0.10.5)
>
> 코드 반영: `ur_env/rlpd_receive_server.py`의 `DEFAULT_REWARD_THRESHOLD`

## ⚠️ 먼저 읽을 것 — 이 변경은 breaking change다

**threshold는 learner fingerprint에 들어간다.** 따라서 0.85로 학습한 checkpoint는 기본값이 0.5인
상태에서 resume할 수 없다.

경로: `scripts/run_rlpd_learner_server.py:93`(`--reward-threshold` 기본값) →
`:519-522`(`RewardClassifierRuntime(threshold=...)`) → `:548-551`(`run_contract["reward_classifier"]["threshold"]`)
→ `:570-573`(`LearnerFingerprint.create`) → `ur_env/learner/checkpoint.py:81-94`(canonical JSON SHA-256에 포함)
→ `checkpoint.py:407-413`(로드 시 SHA·document 양쪽 비교, 불일치 시 `CheckpointFingerprintError`).

실패 시나리오: 0.85로 학습한 `checkpoint_50000`을 두고, 변경 후 운영자가 **같은 resume 명령**을
그대로 실행하면 fingerprint SHA가 달라져 `learner checkpoint fingerprint mismatch`로 즉시 죽는다.

이 동작 자체는 옳다 — reward function이 바뀌면 다른 MDP이므로 lineage를 섞으면 안 된다. 다만
**기존 lineage를 이어받으려면 `--reward-threshold 0.85`를 명시**해야 하고, 그러면 이 변경의 이득이
사라진다. 현재 존재하는 checkpoint는 synthetic acceptance용뿐이므로 실질 피해는 없지만, production
run이 시작된 뒤에는 threshold를 바꾸지 마라.

## 결론

0.85는 근거 없이 잡힌 값이었다. **실패 프레임의 최대 확률이 0.0086**이고 성공 프레임 median이
0.99이므로, `0.01`과 `0.85` 사이 구간은 통째로 비어 있다. 그 구간 어디에 threshold를 두어도
false-positive rate는 정확히 `0%`다. 따라서 0.85를 유지할 이유가 없고, 낮출수록 놓치는 성공만 줄어든다.

## 측정 설계 — 누출을 피한 방법

`cube_in_cup_all3` checkpoint로 `cube_in_cup_0724_success.pkl`을 채점하면 recall이 100%로 나오지만
이는 누출이다. 그 파일은 `take_03_20260724_213944_success.pkl`과 SHA-256이 동일하고
(`748159cc89c3a7ea...`) all3의 학습 데이터에 들어 있다.

그래서 threshold는 **leave-one-take-out CV fold**에서만 튜닝했다. 각 fold는 자기 held-out take를
학습에서 제외한 별도 checkpoint를 가진다 (`dataset/cube_in_cup_cv/fold_take_0N/classifier_ckpt`).

- success: 각 fold의 `val/classifier_data/held_success.pkl` (그 fold가 학습하지 않은 take)
- failure(구 도메인): `hil-serl/examples/experiments/cube_in_cup/dataset/test/classifier_data/cube_in_cup_failure.pkl`
  (n=284, 0720 test split — 세 fold 모두 학습에 쓰지 않음)
- failure(새 도메인): `dataset/cube_in_cup_combined/val_0724/classifier_data/cube_in_cup_0724_failure.pkl`
  (n=281). 학습에 쓰인 `train/.../cube_in_cup_0724_failure.pkl`과 **프레임 내용 해시 교집합 0건**,
  촬영 세션도 분리(train은 21:41 단일 세션, val은 15:26~15:27 10개 세션)임을 확인했다.

## 결과 — 구 도메인(0720 test) failure 기준

held-out success recall | failure FPR:

| thr | take_01 | take_02 | take_03 | pooled recall | FPR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 95.1% | 95.9% | 38.8% | 81.9% | 0.00% |
| **0.85** (구값) | 96.7% | 96.3% | 44.3% | 83.9% | 0.00% |
| 0.80 | 97.8% | 96.6% | 47.9% | 85.3% | 0.00% |
| 0.70 | 98.8% | 96.7% | 54.2% | 87.3% | 0.00% |
| 0.60 | 99.5% | 96.9% | 60.6% | 89.1% | 0.00% |
| **0.50** (신값) | 99.9% | 96.9% | 65.7% | **90.5%** | **0.00%** |
| 0.40 | 100.0% | 96.9% | 72.3% | 92.2% | 0.00% |
| 0.30 | 100.0% | 97.0% | 77.2% | 93.4% | 0.00% |
| 0.20 | 100.0% | 97.6% | 84.9% | 95.4% | 0.00% |
| 0.10 | 100.0% | 98.5% | 91.3% | 97.3% | 0.00% |
| 0.05 | 100.0% | 98.9% | 96.0% | 98.6% | 0.00% |

n: success 1784 / 1752 / 1123, failure 284 (fold마다 동일).

확률 분포:

```text
failure  take_01: max=0.0077  p99=0.0046  median=0.00060   >0.5: 0건
         take_02: max=0.0086  p99=0.0052  median=0.00061   >0.5: 0건
         take_03: max=0.0038  p99=0.0024  median=0.00057   >0.5: 0건

success  take_01: min=0.4238  p05=0.9029  median=0.9988
         take_02: min=0.0109  p05=0.9589  median=0.9993
         take_03: min=0.0052  p05=0.0574  median=0.7602
```

## 결과 — 새 도메인(0724) failure 기준

| thr | fold_take_01 | fold_take_02 | fold_take_03 | pooled |
| ---: | ---: | ---: | ---: | ---: |
| 0.85 ~ 0.05 전 구간 | 0.00% | 0.00% | 0.00% | 0.00% (0/843) |

failure 확률: pooled max `0.0024`, p99 `0.0022`, median `0.0005`. `>0.2` 건수 0.
축퇴 검증: 같은 checkpoint로 val_0724 **success**를 채점하면 mean `0.999` — 모든 입력에 같은 값을
뱉는 상태가 아님을 확인했다.

즉 **새 도메인 failure에서도 0.05까지 FPR이 0%**다. rule-of-three 단측 95% 상한은 pooled 기준
`3/843 ≈ 0.36%`.

## 왜 하필 0.5인가 — 데이터가 아니라 판단이다

**데이터는 0.05까지 안전하다고 말한다.** 0.5는 데이터가 강제한 값이 아니라 안전 마진 판단이다.

1. failure 표본이 0720 284개 + 0724 843개(281×3 fold)로, 관측된 도메인이 둘뿐이다. 조명·배경·카메라가
   또 달라지면 분포가 밀릴 수 있다.
2. 0.5는 관측된 failure 최대값 `0.0086` 대비 **약 58배 마진**이다. 0.2는 23배, 0.05는 6배.
3. classifier가 reward/done의 authority이므로 false positive 1건이 에피소드를 잘못 종료시킨다.
   false negative는 학습을 느리게 할 뿐 잘못된 성공을 만들지는 않는다 — 비대칭 비용이다.

**0.2로 내리는 것도 근거가 있다.** take_03이 65.7% → 84.9%로 올라가고 두 도메인 모두 FPR 0%다.
더 공격적으로 갈 준비가 되면 0.2가 다음 후보다.

## threshold로 해결되지 않는 것

### take_03 편차 — 원인은 카메라도 라벨링도 아니다

take_01/02는 96%대인데 take_03만 44%다. 조사 결과:

- **라벨링 규칙은 동일하다.** 0724 take는 전부 "큐브가 이미 컵에 들어간 상태로 로봇만 움직이며 녹화"
  방식이라 take 전체가 단일 라벨이다(`examples/export_0724.py`). 프레임 수 차이(1784/1752/1123)는
  녹화 길이 차이일 뿐이다(59.5s / 58.4s / 37.5s, 약 30fps).
- **cam2의 컵 크기는 통계적으로 다르지 않다.** HSV 블롭 면적 중앙값 31,755 / 26,250 / 26,417 px로
  겹친다. 화질·종횡비도 확률과 무상관(|r|<0.13).
- **진짜 원인은 팔이 카메라 프레임을 침범하거나 이탈하는 궤적이다.** take_03은 확률이 1~2초 주기로
  0.005~1.0을 톱니처럼 진동한다. cam1에서 로봇 팔이 화면 상단에서 내려오는 정도가 확률과 프레임 단위로
  역상관한다(t=8.40s 팔 위축 p=0.93 → t=9.21s 팔 하강 p=0.31). take_01도 같은 실패 모드를 겪지만
  짧고 얕다(최저 0.42).

**시사점**: threshold를 낮춰도 구제되지 않는다 — 0.0052까지 떨어지는 프레임이 존재한다. 실기에서
로봇이 성공한 뒤 리셋 동작으로 움직이는 동안 판정이 "실패"로 뒤집힐 수 있다. 완화책은 (a) 팔이
정지/근접한 자세에서만 classifier를 질의, (b) 단일 프레임 대신 N-of-M 다수결 또는 시간 평활,
(c) 광범위 스윙 궤적 데이터를 프레임 단위로 재검수해 학습에 포함.

### 전처리 불일치

classifier는 크롭 없이 128×128로 학습됐는데(`analysis/infer_new_takes.py` 헤더: "matches training and
deployment exactly: ... frames are resized to 128x128") `ur_env/envs/ur7e_env.py`의 canonical
observation은 `IMAGE_CROP`을 적용한다. **위 수치는 전부 크롭 없는 이미지에서 측정한 것이므로,
크롭된 입력을 주는 순간 이 표는 무효다.**

### 구 checkpoint는 사용 불가

`e329986b...`(7/24 msgpack)는 0724 도메인 recall이 `0.0%`다(성공 1123프레임 중 0건, mean 확률 0.007).
threshold를 아무리 낮춰도 살아나지 않는다. `run_rlpd_learner_server.py`의
`DEFAULT_CLASSIFIER_CHECKPOINT_SHA256` 기본값이 아직 이 값이므로,
**`--expected-classifier-sha256`를 반드시 명시**하라.

### actor/learner threshold 불일치를 막는 가드가 없다

`rlpd_receive_server.py:1261`의 이중 검증은 전송된 transition 내부 정합성만 본다
(`classifier_success == (probability > threshold)`). learner 설정값과 대조하지 않는다. actor와 learner가
서로 다른 threshold로 뜨면 **서로 다른 reward function으로 라벨링된 transition이 조용히 섞인다.**
fail-closed 가드 추가를 검토할 것.

## 재현

```bash
# threshold 스윕 (fold별 checkpoint + 0720 test failure)
CUDA_VISIBLE_DEVICES=<idle gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/home/junhyeong/workspace/youngwoong/hil-serl/serl_launcher \
/home/junhyeong/workspace/youngwoong/hil-serl/.venv-train/bin/python /tmp/sweep_thr.py

# checkpoint 간 비교 (0720 val/test, 0724 held-out)
cd ~/workspace/youngwoong/dataset/cube_in_cup_combined
CUDA_VISIBLE_DEVICES=<idle gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/home/junhyeong/workspace/youngwoong/hil-serl/serl_launcher \
/home/junhyeong/workspace/youngwoong/hil-serl/.venv-train/bin/python evaluate_all.py <ckpt_dir> ...
```

`nvidia-smi`로 유휴 GPU를 먼저 확인한다. Kanu의 `gello_software` checkout은 dirty detached 상태이므로
건드리지 않는다.
