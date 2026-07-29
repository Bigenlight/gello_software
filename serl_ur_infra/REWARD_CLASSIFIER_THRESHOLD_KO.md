# reward threshold를 0.85에서 0.5로 낮춘 근거

> 최초 측정일: 2026-07-28 KST · **교정 감사일: 2026-07-29 KST** · 측정 위치: Kanu GPU · 인터프리터
> `~/workspace/youngwoong/hil-serl/.venv-train/bin/python` (JAX/JAXLIB 0.5.3, Flax 0.10.5)
>
> 코드 반영: `ur_env/rlpd_receive_server.py`의 `DEFAULT_REWARD_THRESHOLD`

## 🔴 이 문서는 2026-07-29 감사로 핵심 수치가 교정됐다

**결론(threshold 0.5)은 유효하다. 그러나 07-28 최초 작성본의 근거 수치 상당수가 틀렸다.**
아래 표를 먼저 보고, 07-28 판본의 숫자를 인용해 둔 다른 문서·슬라이드·커밋 메시지가 있으면 같이 고쳐라.

| 07-28 최초 작성본의 주장 | 07-29 감사 결과 |
| --- | --- |
| 새 도메인(0724) failure 281프레임은 held-out이다 | **거짓 — 281프레임 전부가 학습 데이터다.** 누출 대조 파일을 잘못 골랐다 |
| 관측된 failure 최대 확률 = `0.0086` | **`0.1428`** (0720 **val** split을 평가에서 빠뜨렸다) |
| `0.01`~`0.85` 구간이 통째로 비어 있다 | 거짓. failure가 없는 구간은 약 **`0.15`~`0.83`**, 그마저도 성공 쪽은 비어 있지 않다 |
| 0.5는 58배 마진, 0.2는 23배, 0.05는 6배 | **0.5 = 3.5배, 0.2 = 1.4배, 0.05 = 0.35배**(이미 관측된 negative 아래) |
| "데이터는 0.05까지 안전하다" | 거짓. 0.05·0.1은 held-out에서 실제로 오탐한다 |
| rule-of-three 상한 `3/843 ≈ 0.36%` | 무효(비독립·학습데이터). 진짜 상한은 n=470 → 0.64%, hard n=24 → 12.5%, take n=6 → **50%** |
| "0.2가 다음 후보다" | **뒤집힘 — 0.2는 보류.** 마진이 1.4배뿐이다 |
| 전처리 크롭 불일치로 "위 표는 전부 무효" | 과장. **지금은 일치한다.** 크롭을 넣는 순간 무효가 된다 |

07-29 감사에서 **새로 추가된** 절: [난이도 분포](#난이도-분포--held-out-negative의-95는-경계를-시험하지-않는다),
[held-out success recall](#처음-측정된-held-out-success-recall-0720),
[경계 프레임 부재](#경계-프레임이-데이터에-없다),
[라벨 품질 검수](#라벨-품질은-문제-없다),
[후속 권고](#후속-권고).

교정 원칙: **틀린 숫자는 지우지 않고 남긴다.** 어떤 주장이 왜 틀렸는지(대조 파일 오선택, val split 누락)를
남겨야 같은 실수가 반복되지 않는다.

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

(이 절은 07-29 감사에서 변경 없음.)

## 결론

> **07-28 원문 (근거 무효):** "0.85는 근거 없이 잡힌 값이었다. **실패 프레임의 최대 확률이 0.0086**이고
> 성공 프레임 median이 0.99이므로, `0.01`과 `0.85` 사이 구간은 통째로 비어 있다. 그 구간 어디에
> threshold를 두어도 false-positive rate는 정확히 `0%`다. 따라서 0.85를 유지할 이유가 없고,
> 낮출수록 놓치는 성공만 줄어든다."

**[07-29 교정] 결론(0.5)은 유지한다. 근거는 아래로 바꾼다.**

1. 진짜 held-out negative는 0720 test 284 + **0720 val 186 = 470프레임(6 take)** 이다.
   여기서 관측된 **failure 최대 확률은 `0.1428`**(0.0086 아님).
2. 0.5에서 held-out negative FPR은 여전히 **0%**다(470/470). 즉 0.5는 안전하다.
3. 0.5는 0.85 대비 held-out success recall을 올린다(배포 ckpt `all3` 기준 **83.1% → 86.8%**).
   낮출수록 놓치는 성공이 준다는 방향성 자체는 맞다.
4. 다만 마진은 58배가 아니라 **3.5배**(`0.5 / 0.1428`)다. 그래서 **0.2(1.4배)는 보류**,
   **0.05(0.35배)는 금지**다 — 0.05는 이미 관측된 negative 확률보다 아래다.
5. "그 구간 어디에 두어도 0%"는 거짓이다. **0.1과 0.05에서는 실제로 오탐이 발생한다**
   (`take_11_20260720_205805` frame 195, onset-6).

한 줄 요약: **0.5는 옳다. 그러나 여유가 두 자릿수 배수라고 믿지 마라. 3.5배다.**

## 측정 설계 — 누출을 피하려 했으나 절반은 실패했다

`cube_in_cup_all3` checkpoint로 `cube_in_cup_0724_success.pkl`을 채점하면 recall이 100%로 나오지만
이는 누출이다. 그 파일은 `take_03_20260724_213944_success.pkl`과 SHA-256이 동일하고
(`748159cc89c3a7ea...`) all3의 학습 데이터에 들어 있다. **(이 진단은 07-29에도 유효하다.)**

그래서 threshold는 **leave-one-take-out CV fold**에서만 튜닝했다. 각 fold는 자기 held-out take를
학습에서 제외한 별도 checkpoint를 가진다 (`dataset/cube_in_cup_cv/fold_take_0N/classifier_ckpt`).

- success(0724): 각 fold의 `val/classifier_data/held_success.pkl` (그 fold가 학습하지 않은 take) — **유효**
- failure(구 도메인 0720): `hil-serl/examples/experiments/cube_in_cup/dataset/test/classifier_data/cube_in_cup_failure.pkl`
  (n=284, 0720 test split — 세 fold 모두 학습에 쓰지 않음) — **유효, 다만 불완전**
  (아래 [val split 누락](#교정-2--0720-val-split을-평가에서-빠뜨렸다) 참조)
- failure(새 도메인 0724): `dataset/cube_in_cup_combined/val_0724/classifier_data/cube_in_cup_0724_failure.pkl`
  (n=281) — **🔴 07-29 교정: 전부 학습 데이터. held-out 아니다.**

### 교정 1 — 0724 failure "held-out"은 누출이었다

> **07-28 원문 (거짓):** "학습에 쓰인 `train/.../cube_in_cup_0724_failure.pkl`과 **프레임 내용 해시
> 교집합 0건**, 촬영 세션도 분리(train은 21:41 단일 세션, val은 15:26~15:27 10개 세션)임을 확인했다."

**무엇이 틀렸나 — 대조 파일을 잘못 골랐다.**

`val_0724/classifier_data/cube_in_cup_0724_failure.pkl`의 **281프레임 전부**가

```text
hil-serl/examples/experiments/cube_in_cup/dataset/train/classifier_data/cube_in_cup_failure.pkl
  sha256 3aeab0f6…
  키: external_failure/take_NN_20260724_152NNN
```

안에 **바이트 동일하게** 들어 있다. `run_cv.sh`가 이 파일을 모든 fold에 복사하므로
**fold_take_01/02/03 + all3, 네 checkpoint 전부가 이 281프레임을 label 0으로 학습했다.**

07-28에는 이름이 비슷한 `train/.../cube_in_cup_0724_**0724**_failure.pkl`(21:41 단일 세션)하고만
비교했다. 정작 학습에 들어간 경로는 파일명이 다른 `cube_in_cup_failure.pkl`이었고, 그 안의
`external_failure/` 하위에 15:26 세션이 통째로 들어 있었다. **"해시 교집합 0건"은 잘못된 파일 쌍에
대해서는 참이었고, 실제 학습 집합에 대해서는 거짓이다.**

교훈: 누출 검증은 **파일명이 아니라 "그 fold가 실제로 로드한 학습 파일 목록"에서 출발**해야 한다.
`run_cv.sh`가 fold 디렉터리로 복사하는 파일 전부를 열거한 뒤 해시를 비교하라.

### 교정 2 — 0720 val split을 평가에서 빠뜨렸다

CV fold는 **train take만 학습**한다. 따라서 0720 **val** split도 test split과 똑같이 held-out이다.
07-28에는 test split(284프레임)만 채점하고 val split(failure 186프레임)을 통째로 빠뜨렸다.

포함해서 다시 채점한 failure 최대 확률:

| ckpt | 0720 test fail max (07-28에 본 것) | 0720 val fail max (07-28에 빠뜨린 것) |
| --- | ---: | ---: |
| fold_take_01 | 0.0077 | **0.0879** |
| fold_take_02 | 0.0086 | **0.1428** |
| fold_take_03 | 0.0038 | **0.1061** |
| all3 (배포용) | 0.0042 | **0.0856** |

→ **관측된 최대 failure 확률은 `0.0086`이 아니라 `0.1428`이다.** 한 자릿수 배수만큼 틀렸고,
이 하나가 문서의 마진 계산 전체를 무너뜨렸다.

→ **"`0.01`과 `0.85` 사이가 통째로 비어 있다"는 거짓이다.** held-out failure가 한 건도 없는 구간은
약 **`0.15`~`0.83`**이다. 게다가 그 구간도 *성공* 쪽에서는 비어 있지 않다 — 병리 take인
`take_21_20260720_210234`의 success 프레임들이 그 안에 몰려 있다
([아래](#처음-측정된-held-out-success-recall-0720) 참조). **"비어 있다"는 표현 자체를 쓰지 마라.**

## 결과 — 구 도메인(0720 test) failure 기준

held-out success recall | failure FPR:

| thr | take_01 | take_02 | take_03 | pooled recall | FPR (0720 **test**만) |
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

n: success 1784 / 1752 / 1123 (0724 held-out take), failure 284 (fold마다 동일).

> **[07-29 교정] 이 표의 recall 열은 유효하다. FPR 열은 "0720 test split만"이라는 조건부다.**
> val split 186프레임을 합치면(= 진짜 held-out 470프레임) **0.1과 0.05에서 FPR이 0%가 아니다.**
> 0.5와 0.2는 470프레임 전부에서 여전히 0%다.
> 교정된 FPR은 아래 [난이도 분포](#난이도-분포--held-out-negative의-95는-경계를-시험하지-않는다) 표를 쓸 것.

확률 분포 (0720 test failure / 0724 held-out success):

```text
failure  take_01: max=0.0077  p99=0.0046  median=0.00060   >0.5: 0건
         take_02: max=0.0086  p99=0.0052  median=0.00061   >0.5: 0건
         take_03: max=0.0038  p99=0.0024  median=0.00057   >0.5: 0건
         ^^ [07-29] test split만. val split 포함 시 max는 각각 0.0879 / 0.1428 / 0.1061

success  take_01: min=0.4238  p05=0.9029  median=0.9988
         take_02: min=0.0109  p05=0.9589  median=0.9993
         take_03: min=0.0052  p05=0.0574  median=0.7602
```

## 결과 — 새 도메인(0724) failure 기준 🔴 이 절 전체가 무효다

> **07-28 원문 (전부 무효 — 학습 데이터를 채점한 것이다):**
>
> | thr | fold_take_01 | fold_take_02 | fold_take_03 | pooled |
> | ---: | ---: | ---: | ---: | ---: |
> | 0.85 ~ 0.05 전 구간 | 0.00% | 0.00% | 0.00% | 0.00% (0/843) |
>
> "failure 확률: pooled max `0.0024`, p99 `0.0022`, median `0.0005`. `>0.2` 건수 0.
> 즉 **새 도메인 failure에서도 0.05까지 FPR이 0%**다. rule-of-three 단측 95% 상한은 pooled 기준
> `3/843 ≈ 0.36%`."

**[07-29 교정]** 위 281프레임은 네 checkpoint 전부의 **학습 집합**이다([교정 1](#교정-1--0724-failure-held-out은-누출이었다)).
학습한 negative에서 FPR 0%가 나오는 것은 당연하고 아무것도 증명하지 않는다.
따라서 **"새 도메인에서도 0.05까지 안전하다"는 주장은 근거가 사라졌다.**
그 결과 이 문서에는 **현재 새 도메인(0724) failure에 대한 held-out 증거가 한 건도 없다.**
0724 도메인에서의 FPR은 **미측정 상태**로 취급하라.

살아남는 것 하나: **축퇴 검증**. 같은 checkpoint로 val_0724 **success**를 채점하면 mean `0.999`이므로
모든 입력에 같은 값을 뱉는 상태는 아니다. (다만 이 성공 프레임들도 누출이라 recall 근거로는 못 쓴다.)

## 난이도 분포 — held-out negative의 95%는 경계를 시험하지 않는다

*(07-29 감사에서 신규 추가)*

FPR을 프레임 수로 나눈 값은 표본이 얼마나 어려운지를 감춘다. held-out negative 470프레임을
`success_onset` 기준 난이도로 쪼개면:

```text
                       n     max_p   FPR@0.5  FPR@0.2  FPR@0.1  FPR@0.05
hard (onset-15 이내)    24   0.1428     0.0%     0.0%     4.2%     4.2%
easy (그 이전)         446   0.0086     0.0%     0.0%     0.0%     0.0%
```

수치는 최대값을 낸 ckpt(`fold_take_02`) 기준이다.

- **446/470 = 95%가 "쉬운" 프레임**이다. 큐브가 컵 근처에도 안 간 구간이라 어떤 threshold에서도
  판정이 흔들리지 않는다. 즉 **"470프레임에서 FPR 0%"의 실질 표본은 24프레임짜리다.**
- 0.05의 **유일한** 오탐은 `take_11_20260720_205805` **frame 195**(onset-6, p ≈ 0.086~0.143 — ckpt마다 다름).
  0.1에서는 fold_take_02(0.1428)·fold_take_03(0.1061)이 오탐하고 fold_take_01(0.0879)·all3(0.0856)은 넘긴다.
  0.05에서는 네 ckpt 전부 오탐한다.
- 그 오탐이 뜻하는 것: **큐브 투하 0.2초 전에 성공을 선언 = 조기 에피소드 종료.**
  RL 입장에서는 "던지기 직전 자세"에 보상이 붙는 것이라 정책이 실제로 망가지는 종류의 오류다.

**이 표가 0.2 보류의 진짜 이유다.** 0.2는 관측 오탐이 0건이지만, 오탐이 시작되는 지점(0.1428)과의
거리가 1.4배뿐이고 그 판단을 24프레임이 떠받치고 있다.

## 처음 측정된 held-out success recall (0720)

*(07-29 감사에서 신규 추가. 07-28 문서는 0720 held-out success를 **한 번도 재지 않았다** —
`evaluate_all.py`에 이미 들어 있었는데 실행·인용을 누락했다.)*

0720 held-out positive **266프레임**(test 166 + val 100, **6 takes**):

| ckpt | @0.85 | @0.5 | @0.2 | @0.05 |
| --- | ---: | ---: | ---: | ---: |
| fold_take_01 | 87.2% | 89.5% | 92.9% | 97.4% |
| fold_take_02 | 86.5% | 89.9% | 93.6% | 97.0% |
| fold_take_03 | 85.3% | 86.5% | 89.1% | 95.5% |
| **all3 (배포용)** | **83.1%** | **86.8%** | 88.7% | 94.0% |

**0.85 → 0.5는 배포 ckpt에서 recall을 83.1% → 86.8%로 올린다. threshold 인하의 이득은 실재한다.**

### take 단위로 보면 훨씬 나쁘다 (all3 기준)

- `take_01` / `take_05` / `take_10` / `take_11` / `take_24` (n=228): @0.85에서 **89~100%** — 정상.
- **`take_21_20260720_210234` (n=38): @0.85 `0.0%`, @0.5 `7.9%`, @0.2 `21.1%`, @0.05 `57.9%`.**

즉 **배포 checkpoint가 held-out success take 6개 중 1개를 통째로 놓친다.** 그 take의 확률들은

```text
0.073  0.465  0.730  0.830  0.347  0.625  0.197  0.087  0.061  0.050  0.029
```

으로, 07-28에 "비어 있다"고 단언한 바로 그 구간(0.15~0.83)에 몰려 있다.

원인은 라벨이 아니라 **시야**다 — 팔이 컵 위에 머무르거나 컵이 cam2에서 사라진다.
[take_03에서 진단한 것과 같은 병리](#take_03-편차--원인은-카메라도-라벨링도-아니다)가
0720 held-out에도 존재한다는 뜻이다.

**시사점**: take_21은 threshold로 구제되지 않는다(0.05까지 내려도 57.9%). 그런데 0.05는 negative를
오탐한다. **즉 이 실패 모드를 threshold로 푸는 해는 존재하지 않는다.**
해법은 [시간 평활/게이팅](#후속-권고)뿐이다.

## 경계 프레임이 데이터에 없다

*(07-29 감사에서 신규 추가)*

두 데이터셋 모두 **결정 경계 근처를 담고 있지 않다.**

- **0720**: `success_onset` 기준 프레임 단위 라벨이고 `pre_success_uncertain_frames=3`으로 전환 구간을
  **의도적으로 drop**한다. 실측하면 마지막 failure 프레임과 첫 success 프레임 사이에 **4~6프레임 공백**이 있다.
- **0724**: "큐브가 이미 들어간 / 전혀 안 들어간 상태로 로봇만 움직이며 녹화"하는 방식이라
  **전환 자체가 구조적으로 발생하지 않는다.**

따라서 두 데이터셋은 **쉬운 양극단만** 담고 있고, **진짜 결정 경계(큐브가 컵 가장자리에 걸치는 순간,
들어갔다가 튀어나오는 순간)에서 classifier가 어떻게 행동하는지는 검증된 적이 없다.**

이것이 [난이도 분포](#난이도-분포--held-out-negative의-95는-경계를-시험하지-않는다)와 합쳐지면:
"0.15~0.83이 비어 있다"는 관측은 **분류기가 확신에 차 있다는 증거가 아니라, 애매한 프레임을
데이터에서 잘라냈다는 증거**일 수 있다. threshold를 더 내리기 전에 이 구간을 채우는 데이터부터 모아라.

## 라벨 품질은 문제 없다

*(07-29 감사에서 신규 추가 — 낮은 확률의 원인이 라벨 오염일 가능성을 배제한 기록)*

60+ 프레임을 원본 해상도로 육안 확인한 결과 **오염 0%**다. 특히:

- `take_03`의 min `0.0052` 프레임은 **라벨 오류가 아니다.** 원본 해상도에서 큐브가 컵 안에 명확히
  존재하고, 물체 상태가 고확률(0.99) 프레임과 동일하다. → **진짜 classifier 실패**다.
- 즉 take_03 / take_21의 저확률은 "데이터를 고치면 사라지는" 종류가 아니다.
  분류기가 팔 위치·시야 변화에 민감한 것이 원인이다.

## 왜 하필 0.5인가 — 데이터가 아니라 판단이다

> **07-28 원문 (거짓 전제):** "**데이터는 0.05까지 안전하다고 말한다.** … 0.5는 관측된 failure 최대값
> `0.0086` 대비 **약 58배 마진**이다. 0.2는 23배, 0.05는 6배."

**[07-29 교정] 데이터는 0.05가 안전하다고 말한 적이 없다.** 교정된 마진(기준: 관측 최대 failure 확률
`0.1428`, fold_take_02 / 0720 val):

| thr | 07-28 주장 마진 | **교정 마진** | 판정 |
| ---: | ---: | ---: | --- |
| 0.5 | 58배 | **3.5배** | **채택** — held-out 470프레임 FPR 0%, recall 상승 |
| 0.2 | 23배 | **1.4배** | **보류** — 오탐 0건이지만 여유가 없다 |
| 0.1 | — | 0.70배 | **금지** — 실제 오탐 발생(ckpt 2/4) |
| 0.05 | 6배 | **0.35배** | **금지** — 이미 관측된 negative보다 아래. 네 ckpt 전부 오탐 |

0.5를 고르는 이유는 그대로 유효하다:

1. failure 표본의 도메인이 사실상 **0720 하나뿐**이다(0724 negative는 전부 학습 데이터라 증거가 아니다).
   조명·배경·카메라가 달라지면 분포가 밀린다.
2. classifier가 reward/done의 authority이므로 false positive 1건이 에피소드를 잘못 종료시킨다.
   false negative는 학습을 느리게 할 뿐 잘못된 성공을 만들지는 않는다 — 비대칭 비용이다.
3. 교정된 마진 3.5배는 "넉넉하다"가 아니라 **"허용 가능한 최소한"**에 가깝다.

### rule-of-three 상한 — 07-28 계산은 무효

> **07-28 원문 (무효):** "rule-of-three 단측 95% 상한은 pooled 기준 `3/843 ≈ 0.36%`."

pooled n=843은 **281프레임 × 3모델**이라 서로 독립이 아니고, 그 281프레임은 **전부 학습 데이터**다.
독립 표본이 아닌 것을 n으로 쓰면 상한이 임의로 작아진다.

진짜 held-out 기준으로 다시 계산하면:

| 모집단 | n | 단측 95% 상한 (3/n) |
| --- | ---: | ---: |
| ~~0724 pooled (281×3, 학습 데이터)~~ | ~~843~~ | ~~0.36%~~ **무효** |
| 0720 held-out 프레임 (test 284 + val 186) | 470 | **0.64%** |
| 그중 hard 프레임만 | 24 | **12.5%** |
| **진짜 독립 단위 = take** | **6** | **50%** |

프레임은 서로 독립이 아니다(같은 take 안에서 30fps로 연속). **의사결정에는 take 단위 상한(50%)을
염두에 두라.** "FPR 상한 0.36%"라고 적힌 07-28 판본을 인용하지 마라.

### 0.2는 다음 후보가 아니다 — 보류다 🔴

> **07-28 원문 (뒤집힘):** "**0.2로 내리는 것도 근거가 있다.** take_03이 65.7% → 84.9%로 올라가고
> 두 도메인 모두 FPR 0%다. 더 공격적으로 갈 준비가 되면 0.2가 다음 후보다."

**[07-29 교정] 0.2는 보류한다.** 근거:

- "두 도메인 모두 FPR 0%"의 절반(0724)은 **학습 데이터 채점**이라 무효다.
- 남은 0720에서 마진은 **1.4배**(`0.2 / 0.1428`)뿐이다. ckpt 하나만 조금 밀려도 오탐으로 넘어간다.
- 그 판단을 떠받치는 hard negative는 **24프레임**, 독립 단위로는 **6 take**다.
- 0.2로 얻는 이득(all3 recall 86.8% → 88.7%, +1.9%p)은 **take_21을 구제하지 못한다**(7.9% → 21.1%).
  즉 위험은 실질적이고 이득은 명목상이다.

**0.2를 다시 검토해도 되는 조건**: (a) 0724 negative를 학습에 쓰지 않은 새 세션으로 재수집해 진짜
held-out FPR을 측정했고, (b) [경계 프레임](#경계-프레임이-데이터에-없다)을 포함한 데이터에서
hard negative가 최소 수백 프레임 규모로 확보됐을 때. 그전에는 0.5를 유지하라.

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
- **[07-29 추가] 라벨 오류가 아님을 육안 검수로 확인했다** — min 0.0052 프레임에도 큐브는 컵 안에 있다.
- **[07-29 추가] 이 병리는 0724 전용이 아니다.** 0720 held-out의
  [`take_21`](#take-단위로-보면-훨씬-나쁘다-all3-기준)이 같은 실패 모드이고 더 심하다(@0.85 recall 0.0%).

**시사점**: threshold를 낮춰도 구제되지 않는다 — 0.0052까지 떨어지는 프레임이 존재한다. 실기에서
로봇이 성공한 뒤 리셋 동작으로 움직이는 동안 판정이 "실패"로 뒤집힐 수 있다. 완화책은 (a) 팔이
정지/근접한 자세에서만 classifier를 질의, (b) 단일 프레임 대신 N-of-M 다수결 또는 시간 평활,
(c) 광범위 스윙 궤적 데이터를 프레임 단위로 재검수해 학습에 포함.
**[07-29] (a)와 (b)는 이제 "완화책"이 아니라 [배포 전 필수](#후속-권고)다.**

### 전처리 — 지금은 일치한다. 크롭을 넣는 순간 무효가 된다

> **07-28 원문 (과장):** "`ur_env/envs/ur7e_env.py`의 canonical observation은 `IMAGE_CROP`을 적용한다.
> **위 수치는 전부 크롭 없는 이미지에서 측정한 것이므로, 크롭된 입력을 주는 순간 이 표는 무효다.**"

**[07-29 교정] 지금은 불일치가 없다.** 확인한 사실:

- `ur_env/envs/config.py:26` — `IMAGE_CROP: Dict[str, Callable] = {}` (기본값이 빈 dict).
- `serl_ur_infra`에는 **cube_in_cup용 UR config가 아직 없다** — 즉 크롭을 채우는 subclass가 없다.
- `ur_env/envs/ur7e_env.py:471-478` — 키가 없으면 원본 `bgr`을 그대로 128×128로 resize한다.

```python
cropped = (
    self.config.IMAGE_CROP[key](bgr)
    if key in self.config.IMAGE_CROP
    else bgr                      # ← 현재는 항상 이쪽
)
resized = cv2.resize(cropped, ...)
```

따라서 현재 파이프라인은 측정 시 쓴 `preprocess_frame(frame, None)`과 **동일**하고, 위 표들은 유효하다.

**경고의 방향은 그대로 유지한다: 크롭을 넣지 마라.** cube_in_cup용 UR config를 만들면서
`IMAGE_CROP`에 항목을 추가하는 순간, 이 문서의 모든 확률·threshold 수치가 무효가 된다
(classifier는 크롭 없이 학습됐다 — `analysis/infer_new_takes.py` 헤더: "matches training and
deployment exactly: ... frames are resized to 128x128"). 크롭이 필요하면 **classifier를 같은 크롭으로
재학습하고 이 문서의 스윕을 전부 다시 돌려라.**

> 왜 문구를 고쳤나: 지금 사실이 아닌 경고를 남겨 두면 진짜로 크롭이 들어갔을 때 아무도 반응하지 않는다.
> 경고는 실제로 깨졌을 때만 울려야 한다.

**[07-29 추가] `state` 더미도 무해함이 확인됐다.** 측정 스크립트가 넘기는 `state=np.zeros((1,1))`은
결과에 영향이 없다 — `create_classifier()`가 `use_proprio=False`라 `observations["state"]`를 읽지 않는다.

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

## 후속 권고

*(07-29 감사에서 신규 추가. (c)는 배포 전 필수다.)*

**(a) 0724 negative를 새 세션으로 재수집하라.** 현재 0724 negative는 전량 학습 데이터라
새 도메인 FPR 증거가 0건이다. 학습에 한 프레임도 쓰이지 않은 별도 세션(다른 날짜/조명)으로
failure를 다시 찍고, 수집 직후 `run_cv.sh`가 복사하는 학습 파일 전부와 해시를 대조해
**교집합 0건임을 파일 목록 기준으로** 확인하라([교정 1](#교정-1--0724-failure-held-out은-누출이었다)의 교훈).
가능하면 [경계 프레임](#경계-프레임이-데이터에-없다)(큐브가 컵 가장자리에 걸치는 구간)을 의도적으로 포함하라.

**(b) `evaluate_all.py`를 threshold 스윕의 표준 진입점으로 고정하라.** 07-28의 두 오류(val split 누락,
0720 held-out success 미측정)는 둘 다 **일회용 `/tmp/sweep_thr.py`를 따로 짜서** 생긴 것이다.
필요한 채점은 `evaluate_all.py`에 이미 들어 있었다. 앞으로 threshold를 논할 때는

- 0720 **test + val** × **failure + success** 네 조합을 **항상 전부** 포함하고,
- 프레임 수와 함께 **take 수**를 같이 출력하고,
- 채점한 pkl의 sha256과 그 fold의 학습 파일 목록을 리포트 상단에 찍어라.

**(c) 배포 전 팔 정지 게이팅 또는 N-of-M 시간 평활을 반드시 넣어라.** 선택 사항이 아니다.
[take_21](#take-단위로-보면-훨씬-나쁘다-all3-기준)(@0.85 recall 0.0%)과
[take_03](#take_03-편차--원인은-카메라도-라벨링도-아니다)(min 0.0052)은 **어떤 threshold로도 구제되지 않는다**
— 이들을 살리는 낮은 threshold는 동시에 negative를 오탐한다. 단일 프레임 판정을 reward/done의
authority로 쓰는 한 이 실패 모드는 남는다. 최소 요건:

- classifier는 **팔이 정지/기준 자세일 때만** 질의(움직이는 동안의 판정은 버린다), 또는
- 연속 M 프레임 중 N개 이상이 threshold를 넘을 때만 success 확정(N-of-M), 또는 확률 시간 평활.

## 재현

```bash
# [권장] 표준 진입점 — 0720 val/test × success/failure 전부 포함
cd ~/workspace/youngwoong/dataset/cube_in_cup_combined
CUDA_VISIBLE_DEVICES=<idle gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/home/junhyeong/workspace/youngwoong/hil-serl/serl_launcher \
/home/junhyeong/workspace/youngwoong/hil-serl/.venv-train/bin/python evaluate_all.py <ckpt_dir> ...

# [비권장 · 기록용] 07-28에 쓴 일회용 스윕 — 0720 val split과 held-out success가 빠져 있다.
# 이 스크립트만 돌려서 나온 수치가 이 문서의 오류 원인이다. 다시 쓰지 마라.
CUDA_VISIBLE_DEVICES=<idle gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/home/junhyeong/workspace/youngwoong/hil-serl/serl_launcher \
/home/junhyeong/workspace/youngwoong/hil-serl/.venv-train/bin/python /tmp/sweep_thr.py

# 누출 검증 — 파일명이 아니라 fold가 실제로 로드하는 학습 파일 목록에서 출발할 것
grep -n "cp\|copy" ~/workspace/youngwoong/hil-serl/.../run_cv.sh   # fold로 복사되는 파일 열거
sha256sum <열거된 파일 전부> <채점하려는 파일>                       # 그 다음에 해시 비교
```

`nvidia-smi`로 유휴 GPU를 먼저 확인한다. Kanu의 `gello_software` checkout은 dirty detached 상태이므로
건드리지 않는다.
