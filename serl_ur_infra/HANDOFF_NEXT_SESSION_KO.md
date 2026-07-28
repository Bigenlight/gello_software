# 다음 세션 인수인계 — HIL-SERL 실기 투입

> 갱신: 2026-07-28 KST · HEAD `c069e79` (branch `test/hil-hardware-comms`, origin 푸시 완료)
> 이전판: 2026-07-27 `db01e64`
>
> **이 문서 하나만 읽고 바로 이어서 작업할 수 있게 쓴 것이다.** 배경이 더 필요하면 §10의 문서 지도를 따라간다.

---

## 0. 30초 요약

GELLO 리더 + UR7e + 개입(intervention) 시스템 + 원격 GPU 서버 양방향 통신을 실기에서 돌리는 것이 목표다. **실제 정책(SAC) 코드는 다른 분이 다른 브랜치에서 만들고 있다** — 우리는 그 자리에 mock을 넣고 하드웨어 경로를 완성한다.

**2026-07-28에 관문 두 개를 통과했다:**

1. **팔이 실제로 움직였다.** `run_real_hil.py --arm --scale 0.25`로 GELLO 개입 구동 성공. 운영자 확인: "gello가 의도한대로 잘 움직여".
2. **frame-map 측정 완료.** 좌표계 매핑 = **단위행렬**. 부호 뒤집힘도 축 교환도 없다(§4.1).

**그리고 reward classifier의 치명적 결함을 찾아 증명했다** — 학습은 크롭 없이, 추론은 크롭해서 먹인다. recall이 **100% → 33.3%**로 무너진다(§6). 다만 **지금 프로덕션은 안 망가져 있다**: 크롭을 켜는 task config가 Kanu 체크아웃 브랜치에 없다. **우리 액터 브랜치를 머지하는 순간 유입된다.**

**다음에 할 일은 §7에 순서대로 있다.**

---

## 1. 작업 좌표

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
cd $WT
git log --oneline -1     # c069e79 여야 한다
```

| | 값 |
| --- | --- |
| 작업 위치 | `/home/laptop3/gello_worktrees/hil-hardware-comms` (**워크트리**) |
| branch | `test/hil-hardware-comms` (origin 푸시됨) |
| 분기점 | `5fb716b` — 여기서 12 commit 앞 |
| 액터 python | `/home/laptop3/venvs/gello-hil-actor/bin/python` |
| 원격 GPU | Kanu `166.104.35.33`, 8× RTX A4000 |

**⚠️ `/home/laptop3/gello_software`(canonical checkout, branch `feat/gello-ur7e-humble-22.04`)에는 이 작업의 코드가 없다.** `serl_ur_infra/ur_experiments/` 디렉터리 자체가 존재하지 않는다. 같은 PC에서 다른 작업이 진행 중이라 일부러 분리했다 — **건드리지 말 것.**

텔레옵·그리퍼·GELLO·카메라(`ros2_ur_ws/**`) 절차는 양쪽이 사실상 같지만(차이는 신규 파일 2개뿐), `serl_ur_infra` 절차는 **워크트리에서만** 돈다.

---

## 2. 지금 물리적으로 어떤 상태인가

| 장비 | 상태 (2026-07-28 확인) |
| --- | --- |
| UR7e (192.168.10.11) | ✅ ping 0% loss, RTT 0.15 ms. **Remote 모드**로 운용 |
| GELLO 리더 | ✅ `/dev/ttyUSB0` |
| 2F-85 그리퍼 | ✅ 개폐·방향 실기 확인 완료 |
| 카메라 cam1/cam2 | ✅ **복구됨.** 허브 `4-4`에 둘 다 정상 (`4-4.1`, `4-4.3`) |
| Kanu | 🔴 **아무것도 안 떠 있다.** port 50053 미바인딩, GPU 5/6/7 전부 유휴 |

### 🔴 카메라 시리얼이 교체됐다

```
리포에 하드코딩돼 있던 것:  cam1 147122072740 (D435)   cam2 243222072700 (D435if)
실제 연결된 것:             cam1 151623020789 (D435)   cam2 322743060038 (D435if)
```

**옛 시리얼은 이 PC가 커널 로그상 한 번도 본 적 없는 하드웨어다**(2026-07-05까지 소급 확인). 없는 시리얼로 바인딩하면 **조용히 안 뜬다** — 오류가 아니라 "프레임 없음"으로 보인다.

`607e541`에서 3곳을 갱신했다: `launch_cameras.sh`, `run_recorder.sh`, `gello_recorder_gui.py`.
**아직 옛 시리얼이 남아 있는 곳**: `docs/ros2/GELLO_UR7E_{ACT,DIFFUSION,FM}_DEPLOY.md`, `docs/testing/06_SENSORS.md`, 그리고 **다른 저장소인 `gello_software_remote_classifier`**.

### cam2는 손목(wrist) 카메라다 — 확정

녹화된 데이터셋 영상에서 픽셀로 증명했다: cam2는 그리퍼 손가락이 **같은 픽셀에 고정**된 채 배경만 흐른다(정지 픽셀 비율 0.4–1.0%, cam1은 32–43%). 저분산 컬럼이 `x[476,536]`·`x[1008,1149]`에 몰려 `cube_in_cup.py:172`의 손가락 측정치를 독립 재현했다.

→ `IMAGE_CROP` 값 자체는 **정책 관점에서는 올바르다. 고치지 말 것.** `launch_cameras.sh:126`과 `ur_env/envs/config.py:24`의 "close-up/workspace" 주석이 낡은 것이다.

**미확정**: 연결된 두 대 중 **어느 개체가 손목에 달려 있는지**. cam1=D435 / cam2=D435if 배정은 하드웨어 교체를 가로지른 모델-클래스 추론이고, USB 포트 순서도 녹화 당시와 뒤바뀌었다. **팔을 한 번 흔들어 cam2 창에서 손가락이 고정되는지 보면 끝난다.**

---

## 3. 무엇이 검증됐나

### ✅ 실기에서 확인

- **팔 구동** — `--arm --scale 0.25`, 100 스텝 중 개입 64 스텝, `held=0`. 의도대로 움직임
- **frame-map** — 매핑 = 단위행렬 (§4.1)
- **개입 불변식 4종** — anchor-latch(변동 0), gain-latch(변동 0), **action-exec(저장 액션 == 실행 액션, 비율 1.000)**, held-rate 0%
- **2F-85 그리퍼** — Modbus RTU over UR tool-comm `:54321`
- **GELLO 리더** — 7개 모터 baud 57600
- **laptop → SSH 터널 → Kanu 100-step gRPC 왕복** — `replay_insert_count:100`, schema hash 양쪽 일치
- **지연/대역폭** — RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**, step당 96.1 KiB → 7.9 Mbit/s. 병목은 **WiFi(약 13 Mbit/s)**
- **reward classifier 성능** — held-out에서 측정 (§6.3)

### 🟡 코드·단위테스트는 통과, 실기 미검증

- `clip_safety_box` (workspace box) — **`run_real_hil.py`에서는 애초에 비활성**(§5)
- `go_to_reset` branch-cut 게이트 (`wrapped_nearest`)
- 액터 entrypoint 전 경로 (`run_remote_rlpd_actor.py`) — 카메라 포함 아직 안 돌려봄

### 🔴 아직 못 한 것

- **액터 entrypoint를 실기에서 돌리기** (지금까지는 `run_real_hil.py`만 돌렸다 — **다른 코드 경로다**)
- 카메라를 켠 상태의 canonical observation 전 경로
- Kanu 실제 정책 서빙
- **classifier 크롭 불일치 수정** (§6)

### 테스트 — 이 명령 그대로 쓸 것

```bash
cd $WT
set +u; source /opt/ros/humble/setup.bash; source $WT/ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WT/serl_ur_infra:$WT/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
  -p no:cacheprovider serl_ur_infra/tests
# -> 333 passed, 11 skipped
```

**🪤 `third_party/hil-serl/serl_launcher`를 PYTHONPATH에서 빼면 조용히 299 passed로 떨어진다.** 사라지는 것이 하필 핵심 테스트 2개(`test_cube_in_cup_config.py`, `test_frame_wrappers.py`)이고, skip 사유가 **"submodule is not checked out"이라고 거짓말을 한다**(서브모듈은 `c32939b`로 체크아웃돼 있다). **녹색이 아니라 passed 수를 볼 것.**

UR·GELLO suite는 별도로 **436 passed**.

---

## 4. 실기 측정 결과

### 4.1 frame-map — 매핑 = 단위행렬 (해결)

`--scale 1.0`, move-then-hold, 개입 267 스텝.

포화(스텝 델타가 `ACTION_SCALE` 노름 클립에 붙음) 51.3%를 **제외하고** 보면:

| | 전체 267 | **포화 제외 130** |
| --- | --- | --- |
| 단위행렬 잔차 | 0.203 | **0.093** ✅ (기준 0.15) |
| alpha (추종이득) | 0.914 | **0.984** |

자유 9-파라미터 행렬이 전체 표본에서 0.165밖에 못 내려간다 — **단위행렬(파라미터 0개)의 0.203과 거의 차이가 없다.** 진짜 회전이나 축 교환이 있었다면 자유 행렬이 압도적으로 잘 맞았어야 한다. M의 대각 평균 0.871, 비대각 최대 0.084.

**결론: 부호 뒤집힘 없음, 축 교환 없음, 매핑 = I.** 텔레옵이 잘 되는 것과 일치한다.

> **🪤 러너의 frame-map 판정은 아직 포화 표본을 걸러내지 않는다.** 그래서 FAIL로 뜨면서 "좌표계를 의심하라"고 안내한다 — 오진 유도다. 포화 구간에서는 명령 델타가 리더 변위와 무관하게 10 mm로 고정되므로 `robot_dp ≈ M @ leader_dp` 모델 자체가 성립하지 않는다. `held`(거버너 거부)만 세고 **ACTION_SCALE 노름 클립은 안 세는 것**이 사각지대다. 미수정.

### 4.2 속도 — 검증된 EEF 텔레옵에 정렬 (`a5f9890`)

| | 텔레옵 (검증됨) | HIL 이전 | **현재** |
| --- | --- | --- | --- |
| 조인트 슬루 `max_step_rad` | 0.0025 | 0.0020 | **0.0025** ← 동일 |
| 유효 병진속도 | 0.16 m/s | 0.10 | **0.125** |
| 유효 회전속도 | 1.0 rad/s | 0.50 | **0.625** |

**3층을 균일하게 1.25배** 했다. 균일한 것이 핵심 — 한 층만 올리면 다음 층이 조용히 잘라먹어 버퍼 정합성 불변식이 깨진다. 헤드룸 1.200x 유지.

1.25배를 고른 이유는 `max_step_rad`가 정확히 텔레옵 값 0.0025에 떨어지기 때문이다. **더 올리지 말 것**: 0.16 m/s까지 가려면 0.0032가 되는데, `ur7e_gello.yaml:56-63`이 250 Hz 업샘플 + 500 Hz 드라이버 사이클 조건에서 ~0.00314를 천장으로 못박아 뒀다. 더 빠르게 하려면 이 상수가 아니라 **업샘플러 레이트**를 먼저 올려야 한다.

출처: `ur7e_gello_eef.yaml:238,241`, `ur7e_gello.yaml:64`.

---

## 5. ⚠️ workspace box가 `run_real_hil.py`에서 꺼져 있다

```
[UR7eEnv] WARNING: ... Safety box DISABLED
```

버그가 아니라 구조다. `run_real_hil.py`는 `DefaultUR7eEnvConfig`를 쓰는데 거기 `ABS_POSE_LIMIT_* = zeros`이고(`config.py:67-68`), 측정된 실제 박스는 `cube_in_cup.py`(액터용 task config)에만 있다. 코드가 0-부피 박스를 감지하고 "한 점에 팔을 고정하느니 끄겠다"고 판단한다 — 올바른 처리다.

**하지만 실측 결과가 무섭다.** DRY RUN 300 스텝에서 명령 TCP가:

```
cube_in_cup 박스 밖으로 나간 스텝: 241/300 (80%)
  x [-0.075, +0.506]   박스 +0.375~+0.642   ← 45 cm 이탈
  y [+0.093, +0.572]   박스 -0.229~+0.272   ← 30 cm 이탈
시작점에서 최대 이탈: 73.9 cm
```

`--arm`이었다면 팔이 실제로 저기까지 갔다. **속도 제한은 얼마나 빨리 가는지만 막지, 어디로 가는지는 안 막는다.**

운영자 판단으로 박스 없이 진행했고 대신 `--max-steps`를 짧게 잡았다. **다음에 `run_real_hil.py --arm`을 쓸 때는 리더를 5~10 cm 이내로만 움직일 것.**

---

## 6. 🔴 reward classifier — 크롭 불일치 (핵심 이슈)

### 6.1 무슨 일인가

**학습**(kanu `~/workspace/youngwoong/hil-serl/examples/cube_classifier_pipeline.py:290-297`):

```python
def preprocess_frame(frame_bgr, crop):
    if crop:                       # <- export_0724.py:32 는 crop=None 을 넘긴다
        frame_bgr = frame_bgr[y0:y1, x0:x1]
    resized_bgr = cv2.resize(frame_bgr, (128,128), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized_bgr[..., ::-1], dtype=np.uint8)
```

→ **크롭 없이** 1280×720을 통째로 128×128로 찌그러뜨린다(종횡비 왜곡).

**추론**(우리 `ur7e_env.py::get_im()`): JPEG decode → **`IMAGE_CROP` 적용**(cam1 650×650, cam2 720×720) → 128×128 리사이즈.

→ **완전히 다른 그림을 먹인다.**

서버(`rlpd_receive_server.py::_classifier_observation`)는 이미지 변환을 **하나도** 안 한다 — 순수 통과다. 원인은 전적으로 **액터 쪽 `IMAGE_CROP`**이다.

> **원래 G15/B8 문구는 틀렸다.** `decode_classifier_image()`를 지목했는데, 그건 ZMQ GUI 뷰어 전용 모듈이고 gRPC 경로와 **호출 관계가 없다.** 고칠 위치가 다르다.

### 6.2 증명 — 3중 독립 확인

| 방법 | 결과 |
| --- | --- |
| 코드 판독 | `export_0724.py:32` → `preprocess_frame(frame, None)` |
| **픽셀 대조** | full-frame 리사이즈 가설 **MAE 0.00, 100% 비트 일치** / 우리 크롭 **MAE 21–35, 일치 픽셀 ~4%** |
| **실제 체크포인트 실행** | 성공 프레임 36개에서 recall@0.85 **100.0% → 33.3%**, 평균 P 0.9990 → 0.6027, 최저 P **0.0213** |

반증 시도 6종 전부 실패: "all3의 3개 소스 중 하나는 크롭됐을 것"(6개 pkl 전부 무크롭 확인), "프레임 정렬 오류"(100% 완전 일치는 정렬 오류 시 불가능), "종횡비 왜곡이 크롭과 비슷할 것"(상관 0.43–0.48, cam1은 1.97배 확대), "학습 증강이 흡수"(`padding=4`는 프레임의 3.1%, 크롭은 44–54%를 버림 — **1.6–2.4배 초과**).

### 6.3 그런데 classifier 자체는 좋다 — held-out 실측

정확도 1.0000(학습)이 take 암기를 시사해 직접 재봤다. `dataset_manifest.json`의 val/test 분할(학습에 **한 번도 안 들어간** take)과 사람이 프레임 단위로 찍은 정답 라벨 사용.

| 체크포인트 | split | thr | recall | **FPR** | acc |
| --- | --- | --- | --- | --- | --- |
| **Jul-27 all3** | test | 0.5 | **100.0%** | **0.0%** | **100.0%** |
| Jul-27 all3 | test | 0.85 | 95.8% | 0.0% | 98.4% |
| Jul-24 | test | 0.85 | 93.4% | 0.0% | 97.6% |

**FPR이 양쪽 체크포인트·양쪽 split 전부 0.0%다** — 한 번도 측정된 적 없던 수치다. val failure 최대 확률은 Jul-27이 **0.086**, Jul-24가 **0.488**(0.5 임계값에 위험할 만큼 근접).

에피소드별로도 **사람이 찍은 성공 프레임과 사실상 같은 프레임에서 발화**한다(Jul-27 6개 중 5개 정확, 1개는 2프레임 늦음).

**→ 채택: Jul-27 `cube_in_cup_all3` @ threshold 0.5.** Jul-24는 학습 데이터를 복구할 수 없어 held-out 수치를 신뢰할 수 없고, Jul-27이 음성 마진이 약 6배 넓다.

로드 검증: `LOAD OK 13.5s`, 워밍업 후 **1.03 ms/frame**.

### 6.4 체크포인트 출처 — 확정

```
repo:   github.com/YWhero/hil-serl      ← rail-berkeley 아님
경로:   kanu ~/workspace/youngwoong/hil-serl
branch: agent/cube-in-cup-classifier
commit: d753571
실행:   2026-07-27 11:14:38 UTC  (wandb run-20260727_111438-yufxxoum)
소요:   46초 (150 epoch, 7375 샘플)
```

wandb metadata에 program 경로·git SHA·전체 argv가 남아 있어 확정됐다. uncommitted 수정 2개도 복구했고 **둘 다 동작에 영향 없음**을 확인했다(wandb 로깅 플래그, franka import를 try/except로 감싼 것).

**`5fb716b`는 이걸 만들 수 없다.** 서브모듈을 `c32939b`(rail-berkeley 업스트림 = `d753571`의 **부모**)에 핀하고, 어느 gello_software checkout에도 `cube_in_cup` 실험 디렉터리가 없다.

### 6.5 함정

1. **`checkpoint_150`이 5개다.** 같은 12분 세션 산출물: `cube_in_cup_combined`(11:04), `cv/fold_take_01`(11:10), `fold_take_02`(11:11), `fold_take_03`(11:13), **`cube_in_cup_all3`(11:15) ← 이것만 우리 것**. 크기까지 비슷하다.
2. **서빙 코드는 세 번째 저장소에 있다** — `gello_software_remote_classifier @ a2733ee`. 그 서버의 `--checkpoint` **기본값이 Jul-24 옛 모델**을 가리킨다.
3. **`checkpoint_sha256()`이 `os.path.isfile`을 요구한다**(`rlpd_receive_server.py:126-132`). Jul-27 체크포인트는 **orbax 디렉터리**라 그대로 넣으면 즉시 `FileNotFoundError`. **최신 체크포인트를 쓰려면 이 코드를 먼저 고쳐야 한다.**
4. `serl_launcher`는 어디에도 pip 설치돼 있지 않다 — 순전히 `PYTHONPATH`로 해결된다.

### 6.6 수정 방향 — upstream이 이미 답을 갖고 있다

`usb_pickup_insertion`은 **같은 물리 카메라를 두 키로 두 번 등록**해 분류기에 전용 크롭을 준다:

```python
"side_policy":     {"serial_number": "130322274175", ...}
"side_classifier": {"serial_number": "130322274175", ...}   # 같은 카메라
IMAGE_CROP = {"side_policy": img[250:500,350:650], "side_classifier": img[270:398,500:628]}  # 정확히 128×128
image_keys = ["side_policy", ...];  classifier_keys = ["side_classifier"]
```

`ram_insertion`은 반대 선택지 — `classifier_keys == image_keys`, 분류기가 정책 크롭을 그대로 먹는다. **둘 다 upstream 계약 안이다.**

| | (a) 크롭으로 재학습 | (b) 분류기용 이미지 별도 전송 |
| --- | --- | --- |
| schema hash | 그대로 | **변경** (액터·서버 동시) |
| gRPC 대역폭 | 그대로 96 KiB/step | **약 2배** ← p99가 이미 97.1 ms / 100 ms 예산 |
| upstream 패턴 | `ram_insertion` | `usb_pickup_insertion` |
| 비용 | **46초** + 재export | 코드 변경 + 대역폭 |
| 위험 | 크롭 값이 체크포인트에 각인 | 없음 |

`cube_classifier_pipeline.py`에 **`--cam1-crop`/`--cam2-crop`이 이미 배선돼 있다**(경로만 안 썼을 뿐). 변환:

```
cam1  img[20:670, 340:990]  →  --cam1-crop 340,20,990,670
cam2  img[0:720, 420:1140]  →  --cam2-crop 420,0,1140,720
```

**(a)를 택하면 반드시 크롭 값을 체크포인트 옆에 기록하고 서버가 불일치 시 fail-closed 하게 할 것.** 안 그러면 나중에 `IMAGE_CROP`을 바꿨을 때 또 조용히 깨진다.

미적용 패치 초안이 scratchpad에 있다(`git apply --check` 미실시):
`01-actor-classifier-preprocessing-contract.diff`, `02-training-fork-crop-and-sidecar.diff`, `03-remote-classifier-repo-mirror.diff`

---

## 7. 다음에 할 일

### A. ZMQ classifier 뷰어로 실기 확인 ← **여기서 시작**

가장 안전하고(팔 안 움직임) 목표에 가장 직접적이다. held-out에서 6개 take 전부 사람이 찍은 프레임과 같은 프레임에 발화했으니, 실기에서도 그래야 정상이다.

3-터미널 절차는 `docs/rl/REMOTE_CUBE_CLASSIFIER_LAB_RUNBOOK.md`(다른 저장소 `~/youngwoong_ws/gello_software_remote_classifier`)를 따르되 **두 곳을 고쳐야 한다**:

- `REWARD_CLASSIFIER_CHECKPOINT` → Jul-27 all3 디렉터리
- `CAM1_SERIAL=151623020789 CAM2_SERIAL=322743060038` 를 `launch_cameras.sh`에 전달 (그 저장소는 아직 옛 시리얼)

텔레옵으로 큐브를 컵에 넣으면서 `p(success)`가 **그 순간에** 오르는지 본다.

### B. 액터 entrypoint를 실기에서 처음 돌리기

지금까지 실기에서 돈 것은 `run_real_hil.py`뿐이다. **액터는 다른 코드 경로다.**

```bash
cd $WT/ros2_ur_ws
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh \
  --deadman topic --mock-policy-noise 0.05
# 좋으면 --arm 추가
```

`EXPECTED_MODEL_ID` 오버라이드가 필요하다 — 래퍼 기본값은 learner용 ID인데 receive server는 `fake-zero-action-v0`를 광고해서 handshake가 거부된다. **단 Kanu에 지금 아무것도 안 떠 있으므로 receive server를 먼저 띄워야 한다.**

### C. classifier 크롭 불일치 수정 (§6.6)

A의 결과를 보고 (a)/(b) 중 택한다. **`checkpoint_sha256()`의 디렉터리 미지원(§6.5-3)도 같이 고쳐야 한다.**

### D. 남은 액터 결함

1. **전역 ESC 리스너** (`ur7e_env.py:197-208`) — 데드맨과 **별개**다. 아무 창에서 ESC를 누르면 에피소드가 끝난다. 미수정
2. `run_real_hil.py`의 frame-map 판정이 포화 표본을 안 거른다(§4.1) — 오진 유도. 미수정

### E. Kanu 정책 서빙 — §9

---

## 8. 미해결 위험

1. **초기 정책의 action 크기를 아직 모른다.** 학습 전 SAC가 full-scale action을 내면 팔이 튄다. learner를 붙이는 첫 시도는 `--mock-policy-noise` 또는 낮은 `--scale`로 먼저 관측할 것.
2. **환경 드리프트** — Kanu `il` env의 numpy가 **2.2.5**인데 lock은 1.26.4다(메이저 점프). orbax 0.11.12 vs 0.11.5, grpcio 1.80.0 vs 1.74.0. 런타임이 fail-closed로 강제하는 것은 jax/flax/distrax/tfp/wandb뿐이라 **이 3개는 자동으로 안 걸린다.**
3. **Kanu 디스크 95% 사용** (여유 90 G). checkpoint가 305 MiB × N이고 자동 pruning이 없다.
4. **take_21은 양쪽 체크포인트 모두 약하다** (Jul-27은 0.85에서 아예 발화 안 함).
5. **`cygrpc.so`의 기계어 원인 미확인.** 이전에 `__wrap_memcpy` 무한 루프라고 적었으나 **그 심볼이 바이너리에 없다.** 행동(100% CPU 무한 정지)은 100% 재현되므로 결론(venv를 써라)은 유효하다.

---

## 9. Kanu 현황

**아무것도 안 떠 있다.** port 50053 미바인딩, HIL 프로세스 없음, GPU 5/6/7 전부 유휴(2 MiB, 0%). 문서에 있던 "PID 1096786, GPU 7에 12.3 GiB"는 **낡은 정보다.**

### 경로가 문서와 다르다

`/home/laptop3/gello_software`는 **kanu에 존재하지 않는다.** 실제로는:

| 경로 | 상태 |
| --- | --- |
| `~/workspace/youngwoong/hil-serl` | **`agent/cube-in-cup-classifier` @ `d753571`** — classifier를 학습시킨 곳 |
| `~/workspace/youngwoong/gello_software` | detached `0c8a5a8`, **dirty** — 쓰지 말 것 |
| `~/workspace/youngwoong/gello_software_remote_classifier` | `feat/remote-cube-classifier-viewer` @ `a2733ee` — ZMQ 뷰어 + Jul-24 체크포인트 |
| `/tmp/gello-hil-rl-receive-server-v2` | `5fb716b`, clean (Kanu 전용 worktree) |

### 실제 정책 서빙의 하드 블로커: canonical robot demo가 없다

`run_rlpd_learner_server.py`의 `--demo-path`가 required이고 3중 검증된다. `--synthetic-e2e`는 서버가 run_id 화이트리스트를 강제하는데 액터가 `uuid4()`로 매번 새로 만들어 우회 불가.

생산 경로는 코드에 있다 — `remote_actor.py::_dump_data`가 `--checkpoint-path`를 받으면 `<ckpt>/actor_data/<run_id>/replay/data_<step>.pkl`을 남기고 `load_demo_pickles`가 그 형식을 받는다. **단 `buffer_period`가 0이면 아무것도 안 쓴다**(`cube_in_cup.py:245`) — CLI 플래그도 없다.

### Kanu 환경

| 항목 | 값 |
| --- | --- |
| python | **`/home/junhyeong/miniconda3/envs/il/bin/python`** (베이스 `il`) |
| jax·jaxlib·flax·distrax·tfp·wandb | `0.5.3 / 0.5.3 / 0.10.5 / 0.1.5 / 0.25.0 / 0.26.0` — lock 일치 |
| GPU | 8× RTX A4000 (16 GiB). **5/6 권장** |

**⚠️ overlay venv `/tmp/gello-hil-rl-receive-overlay-v2`를 재사용하지 말 것** — protobuf 3.20.3 핀이 wandb 0.26.0 import를 깨뜨린다.
**⚠️ `XLA_PYTHON_CLIENT_PREALLOCATE=false` 필수** — JAX 기본 75% preallocation.

액터가 pin해야 하는 값:

```text
--expected-model-id        hil-serl-hybrid-sac-resnet10-trunk-cache-v1
--expected-reward-authority server_classifier
--expected-reward-model-id  cube-in-cup-checkpoint-150
--observation-schema-hash  3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

---

## 10. 문서 지도

| 문서 | 용도 |
| --- | --- |
| **이 파일** | 다음 세션 시작점 |
| `HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` | 전체 상태. learner는 §1–10, **actor/하드웨어는 §11** |
| `HIL_SERL_KANU_RUNBOOK_KO.md` | Kanu 실행 절차 |
| `docs/testing/README.md` | 하드웨어·통신 검증 런북 인덱스 (00~09) |
| `docs/testing/08_OPEN_GAPS.md` | 미해결 갭 (G15 = 크롭 불일치) |
| `serl_ur_infra/tests/run_real_hil.py` | 실기 개입 러너 — **파일 상단 주석이 안전 설계를 전부 설명한다** |

핵심 코드:

```
serl_ur_infra/ur_experiments/cube_in_cup.py       task config (측정값 + KNOWN CONFLICT 주석 181-190)
serl_ur_infra/ur_env/envs/frame_wrappers.py       RelativeFrame + Quat2EulerWrapper
serl_ur_infra/ur_env/envs/ur7e_env.py             get_im, clip_safety_box, go_to_reset, _await_first_frames
serl_ur_infra/ur_env/rlpd_receive_server.py       RewardClassifierRuntime, _classifier_observation
serl_ur_infra/scripts/run_remote_rlpd_actor.py    액터 entrypoint (--arm/--deadman/--mock-policy-noise)
ros2_ur_ws/run_hil_actor.sh                       액터 실행 래퍼
ros2_ur_ws/run_hil_gui.sh                         데드맨/개입 GUI
```

---

## 11. 최근 commit

| commit | 내용 |
| --- | --- |
| `c069e79` | `--arm` 시 다른 퍼블리셔가 컨트롤러를 잡고 있으면 거부 |
| `607e541` | 액터를 실기에서 돌릴 수 있게 — `--deadman`/`--arm`/`--mock-policy-noise`, 카메라 첫 프레임 대기, 카메라 시리얼 갱신 |
| `a5f9890` | 속도 3층을 검증된 EEF 텔레옵 값에 정렬 |
| `30d21c4` | 인수인계 문서 신설 |
| `db01e64` | actor 쪽 하드웨어·통신 현황 기록 |
| `ee3240e` | z 바닥과 reset gate 정정 |

---

## 12. 함정 모음

| 함정 | 대응 |
| --- | --- |
| **시스템 grpcio 1.30.2 고장** — gRPC 사용 시 오류/로그 없이 100% CPU 무한 정지 | 반드시 `/home/laptop3/venvs/gello-hil-actor/bin/python`. **시스템 `python3`로 gRPC 절대 금지** |
| `PYTHONPATH=…`가 ROS 오버레이를 날림 | **덮어쓰지 말고 이어붙인다**: `:$PYTHONPATH` |
| **반대로** pytest는 ROS PYTHONPATH가 있으면 1개만 수집 | `-p no:launch_testing`. `--ignore=`로는 안 막힌다 |
| ROS `setup.bash`를 `set -u` 아래에서 source | `set +u` / `set -u`로 감쌀 것 |
| `ros2 launch`에 전부 숫자인 `key:=value` | 따옴표를 안에 넣는다: `"key:='val'"` |
| 그리퍼 `:54321`은 클라이언트 **하나만** | 반드시 `Ctrl-C`. **`kill -9` 금지** (FIN-WAIT-2가 재접속을 30–45초 굶긴다) |
| `/joint_states`의 name 순서가 canonical이 아님 | 실제: `[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan]`. **name으로 매핑할 것** |
| 없는 카메라 시리얼로 바인딩 | 조용히 안 뜬다. §2 참조 |

---

## 13. 절대 하지 말 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **Kanu는 읽기 전용.** 쓰기·설치·`git checkout/switch/stash/restore` 금지, 프로세스 kill 금지, **GPU 7 사용 금지**.
- **RViz에서 좌우/전후가 뒤집혀 보인다는 이유로 `wrappers.py`의 X/Y 부호를 뒤집지 말 것.** 카메라 방위각 artifact이며 `R_align = I`다. frame-map 측정으로 재확인됐다(§4.1).
- **프레임 통일 명목으로 wrench(`tcp_force`/`tcp_torque`)를 회전시키지 말 것.** upstream도 안 건드린다.
- **`RelativeFrame.step`의 두 시점을 하나로 합치지 말 것.** action은 step **이전**, observation은 **이후** 행렬로 변환한다. 의도적으로 한 제어 주기 떨어져 있다.
- **`TCP_POSE_SOURCE` / `TCP_OFFSET_XYZ_RPY` / `ABS_POSE_LIMIT`은 결합된 한 세트다.** 하나만 바꾸면 관측 pose가 **오류 없이** 17.4 cm 이동한다.
- **`wrapped_nearest`가 elbow(index 2)를 unwrap하지 않는 것은 의도다.** 없으면 reset이 wrist_3를 한 바퀴 돌려 tool-comm 케이블을 감는다.
- **`IMAGE_CROP` 값을 "분류기가 안 맞으니" 임의로 바꾸지 말 것.** 정책 관점에서는 측정에 근거한 올바른 값이다. §6.6의 두 선택지 중 하나를 의식적으로 고를 것.
- **classifier `image_keys` 순서(`["cam1","cam2"]`)를 바꾸지 말 것.** 파라미터 트리에 각인돼 있다.
- headless 세션에서 `ur_play`/`ur_load`/`ur_stop` 금지. `ur_resend`만.
- canonical checkout `/home/laptop3/gello_software`를 건드리지 말 것.

---

## 14. 진행 방식 (사용자 선호)

- 사용자는 실제 로봇이 처음이다. **한 번에 한 단계씩 가르치듯 설명하고, 명령은 사용자가 직접 실행한다.**
- 문제 해결 시 **검수·피드백 과정을 반드시 넣을 것.** 이 과정에서 값 2개가 뒤집혔고(z 바닥, `RESET_MAX_DIST_RAD`), classifier 크롭 불일치도 이렇게 잡혔다.
- upstream hil-serl 공식 코드가 하는 대로 따른다.
- 병렬 에이전트를 적극 사용해도 좋다. **단 kanu 접근 에이전트에게는 읽기 전용 제약을 반드시 명시할 것.**
