# 다음 세션 인수인계 — HIL-SERL 실기 투입

> 갱신: 2026-07-29 KST · branch `feat/gello-ur7e-humble-22.04` · 기준 머지 `3f199d4` 이후
> (문서 커밋이 계속 올라오므로 HEAD 해시는 고정하지 않는다)
>
> **이 판은 머지 후 통합본이다.** 이전 판은 하드웨어 브랜치(`test/hil-hardware-comms`)에서
> 하드웨어 쪽만 보고 쓴 것이라, learner·reward classifier 쪽 사실이 빠져 있거나 반대로 적혀 있었다.
> 두 갈래가 며칠간 같은 대상을 따로 기록했으므로, 아래는 **양쪽을 대조해 남은 것**이다.
>
> **이 문서 하나만 읽고 바로 이어서 작업할 수 있게 쓴 것이다.** 배경이 더 필요하면 §11 문서 지도.

---

## 0. 무엇을 만들고 있나

**HIL-SERL — 사람이 개입하는 온라인 RL — 을 실제 로봇에서 돌린다.**

- `laptop3`가 로봇을 잡는다: **GELLO 리더팔**(USB, EEF 공간에서 UR7e를 텔레옵), **RealSense 2대**(USB),
  **UR7e**(이더넷, ROS2 드라이버), **2F-85 그리퍼**.
- laptop3 GPU가 약하다. 그래서 **정책 추론·학습·reward classifier를 전부 `kanu`(GPU 서버)에서** 돌리고,
  랩톱과 서버가 매 스텝 실시간으로 주고받는다. reward의 권위는 서버에 있다
  (`reward_authority=server_classifier`) — 랩톱의 `UR7eEnv.compute_reward()`는 이 배치에서 호출되지 않는다.
- 사람은 GELLO를 잡고 언제든 개입한다. 개입 transition은 정책 transition과 함께 리플레이로 간다.

### 실시간 제약이 이 프로젝트의 중심 수치다

제어 루프가 **10 Hz**(`ur_env/envs/config.py:32`)이므로 **한 스텝의 예산은 100 ms**다.
그 안에 관측 인코딩 → 서버 왕복 → 액션 실행이 다 들어가야 한다. 실측(§3, 100-step gRPC 왕복):

| | 값 |
| --- | --- |
| 왕복 RTT p50 / p95 / **p99** | 58.6 / 75.8 / **97.1 ms** |
| 관측 1건 | **96.1 KiB** |
| 10 Hz 상행 대역폭 | 약 **7.9 Mbit/s** |
| 링크 실효 대역폭 | **WiFi 약 13 Mbit/s** ← 병목 |

**p99 97.1 ms는 예산 100 ms의 97%다.** 왕복의 약 1%가 97 ms를 넘고, 그 꼬리가 곧바로 예산에 닿는다 —
여유가 사실상 없다. 병목은 서버 추론이 아니라 **무선 링크**다. 그래서 아래 두 가지가 따라온다:

1. **관측을 키우는 설계는 대가가 크다.** 분류기용 이미지를 따로 보내면 96 KiB가 약 2배가 된다(§5.5).
2. **유선으로 바꾸는 것이 가장 값싼 개선이다.** 코드 변경이 필요 없다.

---

## 1. 작업 좌표

```bash
cd /home/laptop3/gello_software
git log --oneline | grep 3f199d4     # 머지 커밋이 히스토리에 있어야 한다
```

| | 값 |
| --- | --- |
| checkout | **`/home/laptop3/gello_software`** (canonical, 여기서만 작업한다) |
| branch | `feat/gello-ur7e-humble-22.04` |
| 머지 커밋 | `3f199d4` (부모 `3ff5f80` + `1a4f93d`) — `test/hil-hardware-comms` 를 통합 |
| 액터 python | `/home/laptop3/venvs/gello-hil-actor/bin/python` |
| classifier python (랩톱) | `/home/laptop3/venvs/hilserl/bin/python` (jax 0.5.3 / flax 0.10.5, **CPU 백엔드**) |
| 원격 GPU | Kanu `166.104.35.33`, 8× RTX A4000 (16 GiB) |

> **⚠️ 워크트리 `/home/laptop3/gello_worktrees/hil-hardware-comms`(`1a4f93d`)는 이제 쓰지 않는다.**
> 그 브랜치는 통째로 머지됐다. 옛 문서가 `export WT=...`로 시작하라고 하면 **무시하고 canonical
> checkout을 쓴다.** 워크트리에서 계속 작업하면 이미 머지된 코드를 두 번 만들게 된다.

---

## 2. 물리 리그 — 실제 값

| 장비 | 값 / 상태 |
| --- | --- |
| UR7e | `192.168.10.11`, ping RTT 0.15 ms, **Remote 모드**로 운용 |
| GELLO 리더 | `/dev/ttyUSB0` (by-id `usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0`), Dynamixel **7개**(조인트 6 + 그리퍼 id 7), baud **57600** |
| 2F-85 그리퍼 | UR tool-comm **TCP `:54321`**, Modbus RTU. **로봇 전원 ON 필요**, 클라이언트는 **하나만** |
| cam1 | **SCENE** — 삼각대 3인칭 고정. plain D435 |
| cam2 | **WRIST** — 그리퍼에 강체 장착. D435IF |
| Kanu | 🔴 **HIL 프로세스가 하나도 안 떠 있다** (port 50053 미바인딩, GPU 유휴) |

### 🔧 카메라 시리얼은 하드코딩하지 않는다 (`43ba314`, `fb48100`)

리포 여러 곳에 **D435 시리얼 두 쌍**이 등장하는데, **이 PC에 실제로 붙은 적이 있는 것은 한 쌍뿐이다.**

```
쌍 A: 147122072740 / 243222072700     ← 이 호스트에서 열거된 적 없음 (문서에만 존재)
쌍 B: 151623020789 / 322743060038     ← 실제로 붙어 있는 쌍
```

전체 영속 저널(97 부팅, 2025-07-28 ~ 현재)로 검증했다:

```bash
journalctl -k --boot=all | grep -c '151623020789\|322743060038'   # -> 650
journalctl -k --boot=all | grep -c '147122072740\|243222072700'   # -> 0
```

마지막 USB 열거는 2026-07-28 20:32이고 07-29에는 재열거 자체가 없었다. 즉 **"쌍이 뒤집혔다"는
관찰은 근거가 없다.** 제품 문자열도 일치한다: `4-4.1 = Depth Camera 435`(plain D435 → cam1),
`4-4.3 = Depth Camera 435if`(→ cam2 손목).

**없는 시리얼로 바인딩해도 오류가 나지 않는다.** `realsense2_camera` 노드는 정상 기동한 뒤
아무것도 발행하지 않고, 소비자는 "프레임 없음"만 본다 — 스택 어디에도 "빠진 카메라"와
"잘못 설정된 카메라"를 구별하는 지점이 없다. 그래서 시리얼을 **요구사항이 아니라 선호값**으로
다루고, 둘 다 안 붙어 있으면 **모델 클래스로 배정**한다
(plain D435 → cam1 SCENE, D435IF/i → cam2 WRIST). 같은 클래스 2대면 시리얼 정렬 순서로 떨어뜨리고
"패널을 눈으로 확인하라"고 경고한다. 2대 미만이면 **카메라 프로세스를 띄우기 전에 하드 에러**로 죽는다.

이 정책은 이제 **한 군데에 있고 세 경로가 공유한다**(`fb48100`):

| 구현 | 쓰는 곳 |
| --- | --- |
| `ros2_ur_ws/_resolve_camera_serials.sh` (source 전용, `resolve_serials`) | `launch_cameras.sh`, `run_recorder.sh` |
| `gello_recorder_gui.py::_resolve_camera_serials()` (같은 정책의 파이썬 판) | GUI 레코더 (카메라를 **자기가 직접 띄운다**) |

> **🚫 시리얼을 하드코딩하거나 `CAM1_SERIAL=`/`CAM2_SERIAL=`로 덮어쓰지 말 것.**
> 기본값은 이미 붙어 있는 쌍이고, 다른 쌍이 꽂히면 자동 해석이 알아서 잡는다.
> 손으로 넣은 값이 틀리면 **경고 없이 프레임이 안 나온다**(위 문단). 오버라이드는 스크립트가
> "모델 클래스가 애매하다"거나 "2대 미만"이라고 **먼저 경고했을 때만**, 그 경고문이 알려주는
> 시리얼을 그대로 넣는 용도다.

> **미확정 — 팔을 한 번 흔들면 끝난다.** cam1/cam2 배정은 **모델 클래스 추론**이고, 그것이 각 개체를
> 마운트에 묶는 유일한 근거다. **cam2가 손목이므로, 팔을 움직였을 때 그리퍼 손가락은 같은 픽셀에
> 고정된 채 배경만 쓸려야 한다.** (녹화 데이터셋에서 픽셀로 증명된 성질: 정지 픽셀 비율 cam2 0.4–1.0%,
> cam1 32–43%.)

`cube_in_cup`의 `IMAGE_CROP`은 이 사실 위에 세워져 있다 — cam2 크롭은 테이블이 아니라
**측정된 파지축 x=781**(손가락 x 512–563 / 1000–1117의 중앙)에 맞춰져 있다.
**정책 관점에서 이 값은 옳다. 분류기가 안 맞는다는 이유로 고치지 말 것**(§5).

---

## 3. 무엇이 검증됐고 무엇이 아닌가

### ✅ 실기에서 확인 (2026-07-28 · 07-29)

- **🆕 07-29 — reward classifier가 실기에서 라이브로 돈다.** `3ff5f80`이 ROS2 노드 + PyQt 모니터 GUI를
  넣었고, **랩톱 CPU에서 약 12 ms/frame**으로 카메라 토픽을 그대로 받아 `p(success)`를 띄운다.
  **GPU도 kanu도 SSH 터널도 필요 없다.** 절차는 `REWARD_CLASSIFIER_LIVE_KO.md`(§11)에 따로 있다 —
  여기서는 한 줄만:
  ```bash
  cd /home/laptop3/gello_software/ros2_ur_ws
  VIEW=false ./launch_cameras.sh          # 먼저 카메라 (뷰어 창은 안 띄운다)
  REWARD_CLASSIFIER_PYTHON=/home/laptop3/venvs/hilserl/bin/python ./run_classifier_viewer.sh
  ```
  **⚠️ 이것이 검증한 것은 "이 체크포인트가 이 조명·이 배경·이 카메라에서 동작한다"까지다.**
  뷰어는 **무크롭** 입력이라 §5.1의 크롭 불일치는 여전히 열려 있고, **RL 루프의 reward는 이걸로
  검증되지 않았다.**
- **팔 구동** — `run_real_hil.py --arm --scale 0.25`, 100 스텝 중 개입 64, `held=0`. 의도대로 움직임
- **frame-map = 단위행렬** — 부호 뒤집힘도 축 교환도 없다(§4.1)
- **개입 불변식 4종** — anchor-latch 변동 0, gain-latch 변동 0, **action-exec(저장 액션 == 실행 액션) 비율 1.000**, held-rate 0%
- **2F-85 그리퍼** — 개폐·방향 확인
- **GELLO 리더** — 7개 모터
- **laptop → SSH 터널 → Kanu 100-step gRPC 왕복** — `replay_insert_count:100`, schema hash 양쪽 일치.
  **단 상대는 zero-action 목 서버**(`model_id=fake-zero-action-v0`, `rlpd_receive_server.py:191`)였다
- **지연/대역폭** — §0 표. 같은 왕복에서 계측
- **reward classifier 성능** — kanu GPU에서 held-out 채점(§5.2). **크롭 없는 입력** 기준

### 🟡 코드·단위테스트는 통과, 실기 미검증

- `clip_safety_box` (`ur7e_env.py:334`) — 구현·배선은 됐으나 `run_real_hil.py` 경로에서는 **비활성**(§4.3)
- `go_to_reset` branch-cut 게이트 (`ur7e_env.py:520`, `ur_kin.wrapped_nearest`)
- 카메라 첫 프레임 대기 (`_await_first_frames`, `ur7e_env.py:475`)
- **kanu GPU 쪽 ZMQ 뷰어**(`run_remote_classifier_viewer.sh` + kanu의 `run_remote_reward_classifier_server.sh`) —
  코드는 이제 kanu에도 있지만(§9) **한 번도 안 돌렸다.** 랩톱 CPU 판(위 ✅)으로 충분하다

### 🔴 아직 안 된 것 — 여기를 솔직하게 읽을 것

- **액터 entrypoint(`scripts/run_remote_rlpd_actor.py`)를 실기에서 단 한 번도 돌린 적이 없다.**
  지금까지 팔을 움직인 것은 **전부 `serl_ur_infra/tests/run_real_hil.py`**다. 이 둘은 **다른 코드 경로**다 —
  run_real_hil.py는 개입 경로만 격리해 보는 러너이고 정책은 항상 zero, 서버도 안 붙는다.
  액터는 gRPC 세션·핸드셰이크·canonical observation·데드맨 배선을 전부 거친다.
  **"팔이 움직였다"를 "액터가 된다"로 읽지 말 것.**
- **카메라를 켠 상태의 canonical observation 전 경로가 실기에서 안 돌았다.**
- **Kanu에 실제 정책이 서빙된 적이 없다.** 지금껏 붙었던 것은 zero-action 목 서버뿐이다.
- **🔴 1순위 — classifier 크롭 불일치.** gRPC/50053 canonical observation은 **크롭돼 있고**
  분류기는 **무크롭**으로 학습됐다(비트 단위로 증명, §5.1). 실측 대가는 recall@0.85 **100% → 33%**.
  **즉 라이브 뷰어는 믿을 수 있지만 RL 루프의 reward는 못 믿는다.** 그 둘은 다른 그림을 본다(§6).
- **🔴 2순위 — receive server가 지금 체크포인트를 아예 못 읽는다.** `checkpoint_sha256()`이 파일만
  받는데 정본 체크포인트는 orbax **디렉터리**다 → `FileNotFoundError`. 게다가 코드 기본 SHA는
  **0724 도메인 recall 0.0%**인 폐기 체크포인트를 가리킨다 → 그대로 띄우면 **reward가 영구 0**이다(§5.3).
- **canonical robot demo가 없다** — learner의 하드 블로커(§9).

### 테스트 — 이 명령 그대로 (2026-07-29 canonical checkout에서 재실행, `fb48100` 이후 기준)

```bash
cd /home/laptop3/gello_software
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
  -p no:cacheprovider serl_ur_infra/tests
# -> 333 passed, 11 skipped  (3.35 s)
```

> **🪤 `third_party/hil-serl/serl_launcher`를 PYTHONPATH에서 빼면 조용히 `300 passed, 13 skipped`로 떨어진다.**
> 사라지는 것이 하필 핵심 테스트 2개(`test_cube_in_cup_config.py`, `test_frame_wrappers.py`)이고,
> skip 사유가 **"submodule is not checked out"이라고 거짓말을 한다.** 새 체크아웃에서 서브모듈이
> 정말 미초기화면 296/17이 된다. **녹색이 아니라 passed 수를 볼 것 — 333이 아니면 잘못 돌린 것이다.**

UR·GELLO suite(`ros2_ur_ws/src/*/test`, 11개 파일)는 별도다. 07-28 기록은 **436 passed**이나
이 판에서 재실행하지 않았다 — 인용 전에 다시 돌릴 것.

---

## 4. 실기 측정 결과

### 4.1 frame-map — 매핑 = 단위행렬 (해결)

`--scale 1.0`, move-then-hold, 개입 267 스텝. 포화(스텝 델타가 `ACTION_SCALE` 노름 클립에 붙음) 51.3%를
**제외하고** 보면:

| | 전체 267 | **포화 제외 130** |
| --- | --- | --- |
| 단위행렬 잔차 | 0.203 | **0.093** ✅ (기준 0.15) |
| alpha (추종이득) | 0.914 | **0.984** |

자유 9-파라미터 행렬은 전체 표본에서 0.165까지밖에 못 내려간다 — **파라미터 0개인 단위행렬의 0.203과
거의 차이가 없다.** 진짜 회전이나 축 교환이 있었다면 자유 행렬이 압도적으로 이겼어야 한다.
M의 대각 평균 0.871, 비대각 최대 0.084. **결론: 매핑 = I.** 텔레옵이 잘 되는 것과 일치한다.

> **🪤 `run_real_hil.py`의 frame-map 판정은 포화 표본을 안 거른다.** 그래서 FAIL을 띄우며
> "좌표계를 의심하라"고 안내한다 — **오진 유도다.** 포화 구간에서는 명령 델타가 리더 변위와 무관하게
> 클립값에 고정되므로 `robot_dp ≈ M @ leader_dp` 모델 자체가 성립하지 않는다. `held`(거버너 거부)만 세고
> **ACTION_SCALE 노름 클립은 안 세는 것**이 사각지대다. 미수정.

### 4.2 속도 3층 — 검증된 EEF 텔레옵에 정렬 (`a5f9890`)

세 층을 **균일하게 1.25배** 했다. 균일한 것이 핵심 — 한 층만 올리면 다음 층이 조용히 잘라먹어
버퍼 정합성 불변식(`ACTION_SCALE * HZ < GOVERNOR`)이 깨진다. 헤드룸 1.200x 양축 유지.

| 층 | 값 | 출처 |
| --- | --- | --- |
| `ACTION_SCALE` | `[0.0125, 0.0625, 1.0]` | `ur_env/envs/config.py:73` |
| `GOVERNOR` | v_max 0.15 / w_max 0.75 / dq_step_max 0.0625 | `ur_env/envs/config.py:86` |
| `UPSAMPLER` | hz 250.0, max_step_rad 0.0025 | `ur_env/envs/config.py:98` |

거버너·업샘플러의 조인트 레이트가 둘 다 0.625 rad/s로 맞아떨어진다.
텔레옵 검증값은 v_max **0.16**(`ur7e_gello_eef.yaml:238`), w_max **1.0**(`:241`),
max_step_rad **0.0025**(`ur7e_gello.yaml:64`) — 지금 스택은 그보다 빠르지 않다.

**더 올리지 말 것.** 0.16 m/s까지 가려면 `max_step_rad`가 0.0032가 되는데,
`ur7e_gello.yaml:56-63`이 250 Hz 업샘플 + 500 Hz 드라이버 사이클 조건에서 **~0.00314를 천장으로**
못박아 뒀다(두 publish가 한 드라이버 사이클로 합쳐질 수 있으므로 예산의 절반). 더 빠르게 하려면
이 상수가 아니라 **업샘플러 레이트(hz)를 먼저** 올려야 한다.

### 4.3 ⚠️ workspace box가 `run_real_hil.py`에서 꺼져 있다

```
[UR7eEnv] WARNING: ... Safety box DISABLED
```

버그가 아니라 구조다. `run_real_hil.py`는 `DefaultUR7eEnvConfig`를 쓰는데 거기
`ABS_POSE_LIMIT_* = zeros`이고(`config.py:76-77`), 측정된 실제 박스는 액터용 task config인
`ur_experiments/cube_in_cup.py:162-167`에만 있다. 코드가 0-부피 박스를 감지하고 "한 점에 팔을
고정하느니 끄겠다"고 판단한다 — 올바른 처리다.

**하지만 실측이 무섭다.** DRY RUN 300 스텝에서 명령 TCP가:

```
cube_in_cup 박스 밖으로 나간 스텝: 241/300 (80%)
  x [-0.075, +0.506]   박스 +0.375~+0.642   ← 45 cm 이탈
  y [+0.093, +0.572]   박스 -0.229~+0.272   ← 30 cm 이탈
시작점에서 최대 이탈: 73.9 cm
```

`--arm`이었다면 팔이 실제로 저기까지 갔다. **속도 제한은 얼마나 빨리 가는지만 막지, 어디로 가는지는
안 막는다.** 운영자 판단으로 박스 없이 진행하고 대신 `--max-steps`를 짧게 잡았다.
**다음에 `run_real_hil.py --arm`을 쓸 때는 리더를 5~10 cm 이내로만 움직일 것.**
(액터 경로는 `cube_in_cup` config를 쓰므로 박스가 살아 있다 — 단 그쪽은 실기 미검증이다.)

---

## 5. reward classifier — 양쪽 기록을 합친 현재 상태

두 갈래가 **서로 다른 것을 측정했고 둘 다 진짜였다.** 하드웨어 쪽은 "액터가 먹이는 그림"을,
learner 쪽은 "어떤 확률 분포에 threshold를 둘 것인가"를 쟀다. 아래는 각 수치가 **무엇 위에서
측정됐는지**를 붙여 정리한 것이다. 그 조건을 떼면 숫자가 서로 모순돼 보인다.

### 5.1 🔴 크롭 불일치 — 증명됨, 미해결

**학습**(kanu `~/workspace/youngwoong/hil-serl/examples/cube_classifier_pipeline.py`):
export 스크립트가 `preprocess_frame(frame, crop=None)`을 넘겨 **1280×720 전체를 그대로 128×128로**
찌그러뜨린다(종횡비 왜곡).

**추론**(우리 `ur_env/envs/ur7e_env.py::get_im()`, `:743`): JPEG decode → **`IMAGE_CROP` 적용**
(cam1 `img[20:670,340:990]` 650×650, cam2 `img[0:720,420:1140]` 720×720, `cube_in_cup.py:211-214`)
→ 128×128 리사이즈. **완전히 다른 그림을 먹인다.**

3중 독립 확인:

| 방법 | 결과 |
| --- | --- |
| 코드 판독 | export가 `crop=None` |
| 픽셀 대조 | 무크롭 가설 **MAE 0.00 / 100% 비트 일치** vs 우리 크롭 **MAE 21–35 / 일치 픽셀 ~4%** |
| 실제 체크포인트 실행 | 성공 프레임 36개에서 **recall@0.85 100.0% → 33.3%**, 평균 P 0.9990 → 0.6027, 최저 P 0.0213 |

반증 시도 6종 전부 실패(대표: "학습 증강이 흡수할 것" — `padding=4`는 프레임의 3.1%인데
크롭은 44–54%를 버린다, **1.6–2.4배 초과**).

**원인은 전적으로 액터 쪽 `IMAGE_CROP`이다.** 서버(`rlpd_receive_server.py::_classifier_observation`)는
이미지 변환을 **하나도** 하지 않는다 — 순수 통과다.

> **옛 문서(G15/B8 초판)가 지목한 `decode_classifier_image()`는 범인이 아니다.** 그 함수는
> ZMQ 뷰어 전용 모듈(`ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_runtime.py:81`)에
> 있고 gRPC 경로와 **호출 관계가 없다**(§6). 그쪽에서 크롭을 안 하는 것은 **그 용도에서는 옳다.**

**언제부터 문제인가:** 머지 이전 `feat/gello-ur7e-humble-22.04`에는 `IMAGE_CROP`을 채우는 task config가
아예 없어서(`config.py`의 기본값은 빈 dict) 파이프라인이 학습과 **일치**했다.
**`ur_experiments/cube_in_cup.py`가 머지로 들어오면서 불일치가 실재하게 됐다.**

### 5.2 체크포인트 2개 — 어느 쪽이 무엇을 증명했나

| | **Jul-24** `e329986b…` | **Jul-27 `cube_in_cup_all3`** |
| --- | --- | --- |
| 형식 | 단일 파일 flax msgpack, 87 MB | **orbax 디렉터리**, 43 MB (`checkpoint_150` 포함) |
| 0724 도메인 recall | **0.0%** (성공 1123프레임 중 0건) → **폐기 대상** | — |
| 0720 test @0.85 | 93.4% (FPR 0.0%) | 95.8% (FPR 0.0%) |
| 랩톱 스테이징 | — | `/home/laptop3/gello_software/classifier_ckpt/cube_in_cup_all3/` (gitignore, 43 MB) |
| 출처 | (학습 데이터 복구 불가) | YWhero/hil-serl `agent/cube-in-cup-classifier` @ `d753571`, 2026-07-27 11:14:38 UTC, 150 epoch / 7375 샘플 / **46초** |

**Jul-27 `all3`의 held-out 수치는 두 종류가 있고, 둘 다 맞다 — 분할이 다르다:**

| 채점 대상 | thr | recall | FPR |
| --- | ---: | ---: | ---: |
| **0720 도메인** test split (`dataset_manifest.json`, n=166, 학습에 안 들어간 take) | 0.5 | **100.0%** | **0.0%** |
| **0720 도메인** held-out positive 266프레임 = test 166 + val 100, **6 take** | 0.85 | 83.1% | — |
| 〃 | 0.5 | **86.8%** | — |
| 〃 | 0.2 | 88.7% | — |
| 0720 held-out negative **470프레임**(test 284 + val 186) | 0.85 / 0.5 / 0.2 | — | **전부 0.0%** |

**⚠️ 이 수치 전부가 "크롭 없는 입력"에서 측정됐다.** §5.1의 크롭이 활성인 경로에는 그대로 적용되지 않는다.
측정 자체는 유효하고, 붙는 조건이 "무크롭"일 뿐이다.

관측된 **failure 최대 확률은 0.1428**(fold_take_02 / 0720 val). `all3`만 보면 0.0856.
**take_21은 어떤 threshold로도 구제되지 않는다** — `all3` @0.85에서 recall **0.0%**, @0.05에서도 57.9%.
원인은 라벨이 아니라 **시야**(팔이 컵 위에 머무르거나 컵이 cam2에서 사라진다). 60+ 프레임 육안 검수로
라벨 오염 0% 확인됨.

로드 검증(kanu GPU): `LOAD OK 13.5 s`, 워밍업 후 **1.03 ms/frame**.
랩톱 CPU에서도 실기로 돈다 — 약 **12 ms/frame**(§3).

#### ⚠️ 분류기는 "해결됨"이 아니다 — 이 다섯 줄을 지우지 말 것

이 문서를 읽고 "분류기는 됐고 다음은 RL"이라고 결론내면 안 된다. 근거는
`REWARD_CLASSIFIER_THRESHOLD_KO.md`이고, 요약은 이렇다:

1. **pooled recall 86.8% @0.5, FPR 0%(0/470)** — 여기까지가 좋은 소식이다.
2. **take 단위로 보면 훨씬 나쁘다.** `take_21`은 @0.5에서 **7.9%**로 무너진다(@0.2에서도 21.1%).
   pooled 평균이 take별 붕괴를 가린다.
3. **"FPR 0%"의 통계적 강도가 약하다.** 독립 단위는 프레임이 아니라 take이고 held-out take는 **6개**뿐이라,
   rule-of-three 단측 95% 상한은 **50%**다. 470프레임 기준 0.64%는 프레임 비독립성을 무시한 값이다.
   게다가 470 중 446(95%)은 큐브가 컵 근처에도 안 간 "쉬운" 프레임이라 **실질 표본은 24프레임**이다.
4. **배포용 `all3`가 측정된 ckpt 중 가장 나쁘다.** 0720 held-out positive 266프레임에서
   fold_take_01/02/03이 각각 89.5 / 89.9 / 86.5% @0.5인데 `all3`는 **86.8%**로 **모든 threshold에서 최저**다.
   즉 **0724 데이터를 더한 것이 0720 도메인을 퇴행시켰다.** 그래도 `all3`를 쓰는 이유는 0724 도메인을
   유일하게 커버하기 때문이다.
5. 그러므로 **어떤 비교든 "재학습해서 좋아졌다"를 주장하기 전에 무엇과 비교하는지 명시할 것.**

> **🟡 미기록 수치 하나.** "동일한 데이터로 재학습만 해도 take별 recall이 최대 21%p 움직인다(시드 분산)"는
> 구두로 돌던 값인데 **리포 어느 문서에도 측정 기록이 없다.** 인용하지 말고, 필요하면
> `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 재현 절차로 직접 측정할 것.

> **🪤 `checkpoint_150`이라는 이름의 디렉터리가 5개다.** 같은 12분 세션 산출물이고 크기까지 비슷하다:
> `cube_in_cup_combined`(11:04), `cv/fold_take_01`(11:10), `fold_take_02`(11:11), `fold_take_03`(11:13),
> **`cube_in_cup_all3`(11:15) ← 이것만 우리 것.**

### 5.3 🔴 receive server는 지금 정본 체크포인트를 **못 읽는다** (블로커 2순위)

두 결함이 겹쳐 있다. 둘 다 고쳐야 gRPC 경로가 산다.

**(1) 디렉터리를 못 받는다.** `checkpoint_sha256()`(`ur_env/rlpd_receive_server.py:148-159`)이
`os.path.isfile`을 요구한다(`:151`). 정본 `cube_in_cup_all3/checkpoint_150`은 **orbax OCDBT 디렉터리**라
로드 시도조차 못 하고 `FileNotFoundError`로 죽는다.

> **🪤 `--expected-classifier-sha256`로 우회되지 않는다.** `RewardClassifierRuntime.__init__`은
> 기대 SHA를 줬든 말든 `checkpoint_sha256(self.checkpoint_path)`를 **무조건 먼저 호출한다**(`:289`).
> 옛 판 문서에 "그전까지는 SHA를 명시하면 된다"고 적혀 있었는데 **틀렸다.**
> **디렉터리 인식(재귀 해시)을 넣는 것 말고 우회로는 없다.**

**(2) 코드 기본 SHA가 폐기된 체크포인트를 가리킨다.**

```
scripts/run_rlpd_receive_server.py:34-36   DEFAULT_CHECKPOINT_SHA256            = e329986b…
scripts/run_rlpd_learner_server.py:72-74   DEFAULT_CLASSIFIER_CHECKPOINT_SHA256 = e329986b…
```

`e329986b…`는 Jul-24 msgpack이고 **0724 도메인 success recall이 0.0%**다
(성공 1123프레임 중 **0건**, 평균 확률 **0.007**). 즉 **(1)을 고치기만 하고 이 상수를 안 바꾸면
서버가 조용히 뜨고 reward가 영구 0이 된다** — 학습이 도는데 아무것도 안 배우는, 가장 알아채기 어려운 실패다.
**(1)과 (2)는 반드시 같은 커밋에서 고칠 것.**

**ZMQ 뷰어 경로는 왜 되는가:** SHA pin이 아예 없고 `os.path.exists()`만 본다
(`reward_classifier_runtime.py:56`) — 그래서 디렉터리 체크포인트가 그냥 로드된다. 즉 **뷰어가 되는 것이
gRPC 경로도 된다는 증거가 아니다.** 뷰어는 이미 Jul-27로 넘어가 있다.
`run_classifier_viewer.sh`, `run_reward_classifier_gui.sh`, `run_remote_reward_classifier_server.sh`,
`reward_classifier_runtime.py:DEFAULT_CHECKPOINT_PATH` 전부 `classifier_ckpt/cube_in_cup_all3`가 기본값이다.
**두 경로가 서로 다른 체크포인트를 보고 있다는 뜻이다. 뷰어에서 본 확률이 gRPC 경로의 reward가 아니다.**

### 5.4 threshold — 0.85 → 0.5 → **0.2**

현재 `DEFAULT_REWARD_THRESHOLD = 0.2` (`ur_env/rlpd_receive_server.py:73`, 커밋 `1b02857`).

경위: 0.85는 근거 없이 잡힌 값이었다 → 07-28 측정으로 0.5(`53d5cf6`) → **07-29 누출 감사(`921a155`)에서
07-28 근거 수치 상당수가 뒤집혔다**(0724 failure 281프레임이 전부 학습 데이터였고, 0720 val split을
평가에서 빠뜨려 관측 failure 최대값이 0.0086이 아니라 **0.1428**이었다). 감사 결론도 0.5였다 —
0.2는 마진이 1.4배뿐이라 **보류**였다.

**0.2로 내려간 것은 새 숫자 때문이 아니라 판단 기준을 바꿨기 때문이다.** 운영자 판단:

> "성공 데이터를 좀 놓치는 것은 괜찮아. 실패를 성공으로 감지하지 않는게 중요해."

| | 결과 | 복구 |
| --- | --- | --- |
| **false positive** | 정책이 실패 행동에 보상받음. MDP 오염 | **불가** — 리플레이에 남는다 |
| **false negative** | 에피소드 안 끝남, 보상 0 | **가능** — 사람이 개입으로 메꾼다 |

held-out negative 470프레임에서 FPR은 0.85 / 0.5 / 0.2가 **전부 0%**라 이 셋은 FPR로 구분되지 않는다.
구분되는 것은 recall뿐이다(83.1 → 86.8 → 88.7%). **0.2에서 지불하는 값은 "관측된 오탐"이 아니라
"관측되지 않은 오탐에 대한 여유"이고, 그 여유가 1.4배로 줄어드는 것을 알고 받아들인 결정이다.**

**붙는 조건 (전부 유효):**
1. 첫 실험은 **감시하에** 돌린다. 오탐이 한 번이라도 보이면 즉시 0.5로 되돌린다.
2. 새 footage가 들어오면 재측정한다. 마진 논증 전체가 **프레임 한 장**(`take_11_20260720_205805`
   frame 195, 큐브 투하 약 0.2초 전)에 걸려 있다.
3. **production run이 시작된 뒤에는 바꾸지 마라** — threshold는 learner fingerprint에 들어가
   (`run_rlpd_learner_server.py` → `LearnerFingerprint` → `ur_env/learner/checkpoint.py`),
   다른 값으로 학습한 checkpoint의 resume은 **fail-closed로 거부**된다. 이 거부는 옳다(reward가 바뀌면
   다른 MDP다). 구 lineage를 이으려면 `--reward-threshold`를 명시해야 하고 그러면 이득이 사라진다.
4. **0.05는 금지다** — 이미 관측된 negative(0.1428) 아래다. 0.1도 ckpt 2/4가 오탐한다.

> **🪤 actor/learner threshold 불일치를 막는 가드가 없다.** 서버의 이중 검증은 전송된 transition의
> 내부 정합성(`classifier_success == (probability > threshold)`)만 본다. 양쪽이 다른 값으로 뜨면
> **다른 reward function으로 라벨링된 transition이 조용히 섞인다.**

### 5.5 해결 방향 — 크롭을 빼는 것이 아니다

`IMAGE_CROP` 값은 **데이터셋에서 측정해 나온 값**이고(테이블 z 바닥, 손목 파지축 x=781) 1차 소비자는
**RL 정책**이다. upstream도 두 선택지를 다 갖고 있다 — `usb_pickup_insertion`은 같은 물리 카메라를
두 키(`side_policy` / `side_classifier`)로 등록해 분류기 전용 크롭을 주고, `ram_insertion`은
`classifier_keys == image_keys`로 정책 크롭을 그대로 먹인다.

| | **(a) 같은 크롭으로 재학습** ← 권장 | (b) 분류기용 이미지 별도 전송 |
| --- | --- | --- |
| schema hash | 그대로 | **변경** (액터·서버 동시) |
| gRPC 대역폭 | 그대로 96.1 KiB/step | **약 2배** ← p99가 이미 97.1/100 ms |
| upstream 패턴 | `ram_insertion` | `usb_pickup_insertion` |
| 비용 | **46초** + 재export + threshold 스윕 재실행 | 코드 변경 + 대역폭 |
| 위험 | 크롭 값이 체크포인트에 각인 | 없음 |

`cube_classifier_pipeline.py`에 인자가 **이미 배선돼 있다**(경로에서 안 썼을 뿐):

```
cam1  img[20:670, 340:990]  ->  --cam1-crop 340,20,990,670
cam2  img[0:720, 420:1140]  ->  --cam2-crop 420,0,1140,720
```

**⚠️ 이 리포는 `y0,y1,x0,x1`로 저장하고 파이프라인은 `x0,y0,x1,y1`을 받는다 — 순서가 바뀐다.**

**(a)를 택하면 반드시 크롭 값을 체크포인트 옆에 기록하고 서버가 불일치 시 fail-closed 하게 할 것.**
안 그러면 나중에 `IMAGE_CROP`을 바꿨을 때 또 조용히 깨진다. 그리고 재학습 후에는
`REWARD_CLASSIFIER_THRESHOLD_KO.md`의 스윕을 **다시 돌려야 한다** — §5.2 표는 전부 무크롭 기준이다.

> **📌 재학습은 threshold 변경과 묶어서 할 것.** 재학습은 체크포인트 SHA를 바꾸고, 그 SHA도
> threshold와 함께 learner fingerprint에 들어간다. 따로 하면 resume이 두 번 깨진다.

> **📌 배포 전 필수 (threshold로 해결되지 않는 것):** take_21(@0.85 recall 0.0%)과
> take_03(min 확률 0.0052)은 **어떤 threshold로도 구제되지 않고**, 이들을 살리는 낮은 값은 동시에
> negative를 오탐한다. 단일 프레임 판정을 reward/done의 권위로 쓰는 한 이 실패 모드는 남는다.
> 최소 요건: **팔이 정지/기준 자세일 때만 질의**, 또는 **N-of-M 시간 평활**.

---

## 6. 경로가 두 개다 — 섞지 말 것

이 둘은 **다른 파일·다른 프로세스·다른 전송·다른 전처리**다. 문서와 대화에서 반복해서 혼동됐다.

| | **gRPC 경로 (실제 RL)** | **ZMQ 뷰어 경로 (사람 확인용)** |
| --- | --- | --- |
| 용도 | 정책 서빙 + reward + 리플레이 | 화면에 p(success)를 띄워 사람이 눈으로 본다 |
| 포트 | Kanu `50053`, 랩톱 터널 입구 **`50153`** | `tcp://127.0.0.1:5594` |
| 입력 | **canonical observation**(19-D state + cam1/cam2 128×128) — **크롭됨** | **raw 카메라 토픽 직접 구독** — **크롭 없음** |
| 전처리 | `ur7e_env.get_im()` (`IMAGE_CROP` 적용) | `decode_classifier_image()` (리사이즈만) |
| 체크포인트 | 코드 기본값이 아직 **Jul-24** pin (§5.3) | **Jul-27 `cube_in_cup_all3`** |
| 엔트리 | `scripts/run_remote_rlpd_actor.py` ↔ `scripts/run_rlpd_{receive,learner}_server.py` | 아래 |
| 로봇 | 움직인다(`--arm` 시) | **안 움직인다** |

**ZMQ 뷰어가 크롭을 안 하는 것은 버그가 아니다.** 그 뷰어의 체크포인트가 무크롭으로 학습됐으므로
그쪽 용도에서는 **정확히 옳다.** 크롭 불일치의 원인은 gRPC 경로의 액터 쪽이다.

ZMQ 뷰어에는 두 가지 구성이 있다:

```bash
# (A) 랩톱 로컬 실행 — 분류기를 랩톱 CPU에서 돌린다. 터널 불필요. ✅ 07-29 실기 검증됨
#     체크포인트는 이미 스테이징돼 있다(classifier_ckpt/cube_in_cup_all3, 43 MB).
cd /home/laptop3/gello_software/ros2_ur_ws
VIEW=false ./launch_cameras.sh          # 먼저 카메라 (뷰어 창 없이)
REWARD_CLASSIFIER_PYTHON=/home/laptop3/venvs/hilserl/bin/python ./run_classifier_viewer.sh

# (B) kanu GPU 실행 — 스크립트가 SSH 터널까지 직접 연다. 🟡 코드는 있으나 미실행
#     kanu 쪽: serl_ur_infra/run_remote_reward_classifier_server.sh (REWARD_CLASSIFIER_PYTHON 지정 필요)
cd /home/laptop3/gello_software/ros2_ur_ws
./run_remote_classifier_viewer.sh
```

두 구성 모두 **cam1/cam2 ROS 토픽이 이미 떠 있어야 한다**(`VIEW=false ./launch_cameras.sh`).
threshold 기본값은 양쪽 다 0.2로 `DEFAULT_REWARD_THRESHOLD`와 맞춰져 있다.
**(A)는 랩톱 CPU에서 약 12 ms/frame으로 돈다 — (B)를 쓸 이유가 딱히 없다.**

> **🪤 (B)를 쓴다면 kanu에서 경로를 고를 것.** `run_remote_reward_classifier_server.sh`와
> `remote_reward_classifier_server.py`는 `3ff5f80`(07-29)에서 새로 생긴 파일이다.
> **새 체크아웃 `/home/junhyeong/gello_software_hil`에는 있다**(확인함). 반면
> kanu의 옛 워크트리 `/tmp/gello-hil-rl-receive-server-v2`는 `5fb716b`(머지 이전)에 고정돼 있어 **없고**,
> 별도 저장소 `~/workspace/youngwoong/gello_software_remote_classifier`에는 옛 뷰어와
> **Jul-24 체크포인트**가 있다 — 그쪽을 쓰면 recall 0%짜리 폐기 모델을 보게 된다(§5.3).

---

## 7. 다음에 할 일 — 순서대로

> **✅ 07-28~29 판의 "A. ZMQ 뷰어로 classifier를 실기에서 눈으로 확인"은 끝났다.**
> 랩톱 CPU에서 라이브로 돌고 절차는 `REWARD_CLASSIFIER_LIVE_KO.md`에 있다(§3, §6-(A)).
> 그래서 아래 목록은 **그 다음부터** 시작한다.

> 📘 **분류기를 개입·학습에 실제로 연결하는 사람은
> [`REWARD_TO_RL_INTEGRATION_KO.md`](REWARD_TO_RL_INTEGRATION_KO.md)를 함께 읽어라.**
> 아래 A·B가 *왜* 선행 조건인지, reward가 누구 권위로 어느 관측에서 만들어지는지,
> 개입 transition이 **두 버퍼에 이중 기록**된다는 것, RLPD 50:50 배치에서 개입이 오프라인
> demo를 희석한다는 것이 거기 있다. 이 §7은 "무엇을 할지"이고 그쪽은 "무엇에 연결되는지"다.

### A. 🔴 classifier 크롭 불일치를 고친다 ← **여기서 시작**

**이것이 최상위 블로커다.** 지금 상태에서 RL 루프를 돌리면 **reward가 틀린다**(§5.1: recall 100% → 33%).
라이브 뷰어가 잘 보인다고 이 문제가 사라지지 않는다 — 뷰어와 gRPC 경로는 **다른 그림을 본다**(§6).

권장은 §5.5의 **(a) 같은 크롭으로 재학습**이다. `IMAGE_CROP`을 지우는 것이 아니다(그 값은 정책
관점에서 측정에 근거해 옳다). 절차:

1. kanu에서 `cube_classifier_pipeline.py`에 크롭 인자를 주고 재학습 (**약 46초**)
   ```
   --cam1-crop 340,20,990,670      # 리포의 img[20:670, 340:990]
   --cam2-crop 420,0,1140,720      # 리포의 img[0:720, 420:1140]
   ```
   **⚠️ 리포는 `y0,y1,x0,x1`로 쓰고 파이프라인은 `x0,y0,x1,y1`을 받는다 — 순서가 바뀐다.**
2. `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 threshold 스윕을 **통째로 다시 돌린다.**
   §5.2의 수치는 전부 무크롭 기준이라 그대로 못 쓴다.
3. 크롭 값을 체크포인트 옆에 기록하고, 서버가 불일치 시 fail-closed 하게 한다.
4. **B와 묶어서 한 커밋으로 낸다** — 재학습은 체크포인트 SHA를 바꾸고, SHA와 threshold는 둘 다
   learner fingerprint에 들어간다. 따로 하면 checkpoint resume이 두 번 깨진다(§5.4-3).

### B. 🔴 receive server가 체크포인트를 읽을 수 있게 만든다 (§5.3)

A의 결과물을 배포하려면 이게 먼저다. **A와 같은 커밋.**

1. `checkpoint_sha256()`(`ur_env/rlpd_receive_server.py:148-159`)에 **디렉터리 재귀 해시**를 넣는다.
   현재는 `os.path.isfile`을 요구해 orbax 디렉터리에서 `FileNotFoundError`로 죽는다.
2. `DEFAULT_CHECKPOINT_SHA256`(`scripts/run_rlpd_receive_server.py:34`)과
   `DEFAULT_CLASSIFIER_CHECKPOINT_SHA256`(`scripts/run_rlpd_learner_server.py:72`)을
   **새 체크포인트 SHA로 갱신한다.** 지금 값 `e329986b…`는 recall 0%짜리 폐기 모델이다.
3. 1만 고치고 2를 빼먹으면 **서버가 조용히 뜨고 reward가 영구 0**이 된다.

### C. 새 데이터 수집 — **GUI 레코더로**

분류기 recall은 아직 충분하지 않다(§5.2의 다섯 줄). take 수를 늘리는 것이 계획된 다음 단계다.

```bash
cd /home/laptop3/gello_software
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
ros2 run gello_recorder gello_recorder_gui
```

- **GUI 레코더는 카메라를 자기가 직접 띄운다.** `launch_cameras.sh`를 같이 돌리지 말 것 —
  같은 USB 장치를 두 프로세스가 잡으려 든다.
- 출력: `ros2_ur_ws/gello_logs/take_NN_<YYYYmmdd_HHMMSS>/{cam1.mp4,cam2.mp4,vectors.h5}`
  (`RECORDER_OUTPUT_ROOT`로 바꿀 수 있다).
- 🪤 **헤드리스 `run_recorder.sh`를 쓰면 안 된다.** 그쪽은 `session_<timestamp>/`를 쓰는데
  라벨링 파이프라인의 `prepare`가 `raw_root.glob("take_*")`로 **직접 자식만** 훑어서 **한 개도 못 읽는다.**
  디렉터리 이름만 다르고 내용물(`cam1.mp4`/`cam2.mp4`/`vectors.h5`)은 같다.
- 📌 준비 자료(파이프라인 CLI 레퍼런스, 촬영 체크리스트, split 계획, `split_takes.py`/`retrain.sh`/`reeval.sh`)가
  이미 있다:
  `/tmp/claude-1000/-home-laptop3-gello-software/22de95e4-65f3-449b-b662-5ae4bd21dd7b/scratchpad/intake/`
  **⚠️ 세션 스크래치 경로다 — 계속 쓸 거면 리포 안으로 옮겨야 살아남는다.**

### D. 액터 entrypoint를 실기에서 **처음** 돌리기

지금까지 실기에서 돈 것은 `run_real_hil.py`뿐이다. **액터는 다른 코드 경로다**(§3).
Kanu에 지금 아무것도 안 떠 있으므로 **서버를 먼저 띄워야 한다.**

```bash
# 0) kanu: receive server (zero-action 목). 랩톱에서 SSH 터널 50153 -> 50053.
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:50153:127.0.0.1:50053 kanu

# 1) preflight만 (actor 미기동, 완전 안전)
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --dry-preflight

# 2) fake-env로 왕복 (로봇·센서 미사용)
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --fake-env

# 3) 실센서 + DRY_RUN (팔 안 움직임)
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --deadman topic

# 4) 좋으면 --arm 추가 (팔이 물리적으로 움직인다)
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --deadman topic --arm --mock-policy-noise 0.05
```

- `EXPECTED_MODEL_ID` 오버라이드가 필요하다 — 래퍼 기본값은 learner용
  `hil-serl-hybrid-sac-resnet10-trunk-cache-v1`인데 receive server는 `fake-zero-action-v0`를 광고해
  핸드셰이크가 거부된다.
- 래퍼가 9단계 읽기 전용 preflight를 돈다(venv / grpcio 버전 / cygrpc 생존 / 오버레이 / 모듈 해석 /
  터널 포트 / 토픽 hz / 컨트롤러 상태 / **이중 퍼블리셔**). 이중 퍼블리셔가 있으면 **거부하고 중단**한다.
- `--mock-policy-noise`는 zero-action 서버 상대로 로봇을 움직여 개입 경로를 실증한다.
  **전이마다 `meta.policy_actions_synthetic=true`가 박힌다 — demo로 쓰면 안 된다.**

### C. classifier 크롭 불일치 수정 (§5.5)

A의 결과를 본 뒤 (a)/(b) 중 의식적으로 고른다. **`checkpoint_sha256()`의 디렉터리 미지원(§5.3)도
같이 고쳐야 한다** — 안 고치면 재학습해도 gRPC 경로가 새 체크포인트를 pin하지 못한다.

### D. 남은 액터 결함 (전부 미수정)

1. **전역 ESC 리스너** (`ur_env/envs/ur7e_env.py:197-208`) — 데드맨과 **별개**다.
   아무 창에서 ESC를 누르면 `self.terminate`가 서고 에피소드가 끝난다.
2. `run_real_hil.py`의 frame-map 판정이 포화 표본을 안 거른다(§4.1) — 오진 유도.
3. actor/learner threshold 불일치 가드 없음(§5.4).
4. classifier 시간 평활 / 팔 정지 게이팅 — **배포 전 필수**(§5.5).

### E. Kanu 실제 정책 서빙 (§9)

---

## 8. 미해결 위험

1. **초기 SAC 정책의 action 크기를 아직 모른다.** 학습 전 정책이 full-scale action을 내면 팔이 튄다.
   learner를 붙이는 첫 시도는 `--mock-policy-noise` 또는 낮은 `--scale`로 먼저 관측할 것.
2. **`ACTION_SCALE`이 learner fingerprint에 없다.** 0.01로 녹화한 데이터를 0.0125에서 재개하면
   **경고 없이 같은 액션이 25% 더 멀리 간다.**
3. **workspace box가 `run_real_hil.py`에서 비활성**(§4.3). 액터 경로는 살아 있지만 실기 미검증.
4. **Kanu 환경 드리프트** — `il` env가 lock과 다르다: numpy **2.2.5**(lock 1.26.4, 메이저 점프),
   orbax 0.11.12(0.11.5), grpcio 1.80.0(1.74.0). 런타임 fail-closed 대상은 jax/flax/distrax/tfp/wandb뿐이라
   **이 3개는 자동으로 안 걸린다.**
5. **Kanu 디스크 95% 사용**(여유 90 G). checkpoint가 305 MiB × N이고 자동 pruning이 없다.
6. **`cygrpc.so`의 기계어 원인 미확인.** 이전에 `__wrap_memcpy` 무한 루프라고 적었으나
   **그 심볼이 바이너리에 없다.** 행동(100% CPU 무한 정지)은 100% 재현되므로 결론(venv를 써라)은 유효하다.

---

## 9. Kanu 현황

**지금 HIL 프로세스가 하나도 안 떠 있다.** port 50053 미바인딩, GPU 유휴.
옛 문서의 "PID 1096786, GPU 7에 12.3 GiB"는 **낡은 정보다.**

### 🔴 `/home/laptop3/gello_software`는 kanu에 존재하지 않는다

| kanu 경로 | 내용 |
| --- | --- |
| `~/workspace/youngwoong/hil-serl` | **classifier를 학습시킨 곳.** YWhero/hil-serl fork, `agent/cube-in-cup-classifier` @ `d753571` |
| `~/workspace/youngwoong/dataset/cube_in_cup_all3/` | 학습 데이터 + **Jul-27 체크포인트** |
| `~/workspace/youngwoong/gello_software_remote_classifier` | 옛 ZMQ 뷰어 + **Jul-24 체크포인트** (폐기 대상) |
| `~/workspace/youngwoong/gello_software` | detached, **dirty** — 쓰지 말 것 |
| `/tmp/gello-hil-rl-receive-server-v2` | `5fb716b`, clean. **머지 이전이라 §6의 새 스크립트가 없다** |

| 항목 | 값 |
| --- | --- |
| python | **`/home/junhyeong/miniconda3/envs/il/bin/python`** (베이스 `il`) |
| jax / jaxlib / flax / distrax / tfp / wandb | `0.5.3 / 0.5.3 / 0.10.5 / 0.1.5 / 0.25.0 / 0.26.0` — lock 일치 |
| GPU | 8× RTX A4000 (16 GiB). **5/6 권장** |

- **⚠️ `XLA_PYTHON_CLIENT_PREALLOCATE=false` 필수.** JAX 기본 75% preallocation이 16 GiB 카드에서
  learner + classifier 동거를 막는다.
- **⚠️ overlay venv `/tmp/gello-hil-rl-receive-overlay-v2`를 재사용하지 말 것** —
  protobuf 3.20.3 핀이 wandb 0.26.0 import를 깨뜨린다.

### 실제 정책 서빙의 하드 블로커: canonical robot demo가 없다

`run_rlpd_learner_server.py`의 `--demo-path`가 required이고 3중 검증된다.
`--synthetic-e2e`는 서버가 run_id 화이트리스트를 강제하는데 액터가 `uuid4()`로 매번 새로 만들어 우회 불가.

생산 경로는 코드에 있다 — `ur_env/remote_actor.py::_dump_data`(`:189`)가 `--checkpoint-path`를 받으면
`<ckpt>/actor_data/<run_id>/replay/data_<step>.pkl`을 남기고 `load_demo_pickles`가 그 형식을 받는다.
**단 `buffer_period`가 0이면 아무것도 안 쓴다**(`ur_experiments/cube_in_cup.py:265`) — **CLI 플래그도 없다.**
canonical demo 녹화의 선결 조건이 이것이다.

또한 학습 시작 게이트는 **online replay ≥ 100 AND offline demo ≥ 1**이다. 둘 다 필요하다.

액터가 pin해야 하는 값 (`run_hil_actor.sh` 기본값과 동일):

```text
--expected-model-id        hil-serl-hybrid-sac-resnet10-trunk-cache-v1
--expected-reward-authority server_classifier
--expected-reward-model-id  cube-in-cup-checkpoint-150
--observation-schema-hash  3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

canonical observation v2: `state (1,19) float32`, `cam1`/`cam2 (1,128,128,3) uint8`.
19-D state는 **알파벳 순**이다: `[0] gripper_pose | [1:4) tcp_force | [4:10) tcp_pose | [10:13) tcp_torque | [13:19) tcp_vel`.
**`state[..., -1]`은 그리퍼가 아니라 `tcp_angular_velocity_z`다** — `GRIPPER_POSITION_INDEX` /
`gripper_position_from_state()`(`ur_env/observation_schema.py:120`, `:358`)만 쓸 것.
이 해시는 리포 **9개 파일 총 12곳**에 pin돼 있고 전부 일치한다.

---

## 10. 함정 모음 · 절대 하지 말 것

### 🪤 함정

| 함정 | 대응 |
| --- | --- |
| **시스템 grpcio 1.30.2 고장** — gRPC 사용 시 오류/로그 없이 100% CPU 무한 정지 | 반드시 `/home/laptop3/venvs/gello-hil-actor/bin/python`. **시스템 `python3`로 gRPC·pytest 절대 금지** |
| `PYTHONPATH=…`가 ROS 오버레이를 날림 | **덮어쓰지 말고 이어붙인다**: `${PYTHONPATH:+:$PYTHONPATH}` |
| **반대로** pytest는 ROS PYTHONPATH가 있으면 1개만 수집 | `-p no:launch_testing`. `--ignore=`로는 안 막힌다 |
| ROS `setup.bash`를 `set -u` 아래에서 source | `set +u` / `set -u`로 감쌀 것 |
| `ros2 launch`에 전부 숫자인 `key:=value` | 따옴표를 안에 넣는다: `"serial_no:='151623020789'"` |
| 그리퍼 `:54321`은 클라이언트 **하나만** | 반드시 `Ctrl-C`. **`kill -9` 금지** (FIN-WAIT-2가 재접속을 30–45초 굶긴다) |
| `/joint_states`의 name 순서가 canonical이 아님 | 실제: `[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan]`. **name으로 매핑할 것** |
| 없는 카메라 시리얼로 바인딩 | 조용히 안 뜬다. 이제 자동 해석된다(§2) |
| 테스트가 "녹색"인데 통과 수가 적음 | serl_launcher 누락. **333이 아니면 잘못 돌린 것**(§3) |
| `checkpoint_150` 디렉터리가 5개 | `cube_in_cup_all3`만 우리 것(§5.2) |
| 뷰어의 확률과 learner의 reward가 다름 | 서로 다른 체크포인트·다른 전처리다(§5.3, §6) |

### 🚫 절대 하지 말 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **Kanu는 읽기 전용.** 쓰기·설치·`git checkout/switch/stash/restore` 금지, 프로세스 kill 금지,
  **GPU 7 사용 금지**.
- **RViz에서 좌우/전후가 뒤집혀 보인다는 이유로 X/Y 부호를 뒤집지 말 것.** 카메라 방위각 artifact이며
  `R_align = I`다. frame-map 실측으로 재확인됐다(§4.1).
- **프레임 통일 명목으로 wrench(`tcp_force`/`tcp_torque`)를 회전시키지 말 것.** upstream도 안 건드린다.
- **`RelativeFrame.step`의 두 시점을 하나로 합치지 말 것.** action은 step **이전**, observation은 **이후**
  행렬로 변환한다. 의도적으로 한 제어 주기 떨어져 있다.
- **`TCP_POSE_SOURCE` / `TCP_OFFSET_XYZ_RPY` / `ABS_POSE_LIMIT`은 결합된 한 세트다.**
  하나만 바꾸면 관측 pose가 **오류 없이** 17.4 cm 이동한다.
- **`wrapped_nearest`가 elbow(index 2)를 unwrap하지 않는 것은 의도다.** 없으면 reset이 wrist_3를
  한 바퀴 돌려 tool-comm 케이블을 감는다.
- **`IMAGE_CROP` 값을 "분류기가 안 맞으니" 임의로 바꾸지 말 것.** 정책 관점에서는 측정에 근거한
  올바른 값이다. §5.5의 두 선택지 중 하나를 의식적으로 고를 것.
- **classifier `image_keys` 순서(`("cam1","cam2")`)를 바꾸지 말 것.** 파라미터 트리에 각인돼 있다.
- **production run이 시작된 뒤 threshold를 바꾸지 말 것**(§5.4-3).
- **`--mock-policy-noise`로 만든 전이를 demo로 쓰지 말 것.** `policy_actions_synthetic=true`가 박힌다.
- headless 세션에서 `ur_play`/`ur_load`/`ur_stop` 금지. `ur_resend`만.

---

## 11. 문서 지도

### 지금 지침으로 읽을 것

| 문서 | 용도 |
| --- | --- |
| **이 파일** | 다음 세션 시작점. 현재 상태 · 다음 할 일 · 안전 규칙 |
| `HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` | 전체 상태 기록 — learner 구현 상세(CLI, 학습 계약, fingerprint, checkpoint/resume, feature replay)와 actor/하드웨어 현황, classifier 조사. 이 문서보다 깊다 |
| `HIL_SERL_KANU_RUNBOOK_KO.md` | Kanu learner 실행 절차 |
| `REWARD_CLASSIFIER_THRESHOLD_KO.md` | threshold 결정의 **전체 근거**와 07-29 누출 감사. §5.4의 원본. 틀린 07-28 수치를 지우지 않고 왜 틀렸는지와 함께 남겨 뒀다 — 인용할 때 07-28/07-29 판을 구별할 것 |
| `REMOTE_ACTOR_GRPC.md` | gRPC 액터 전송 계약 (영문) |
| `RVIZ_HIL_TEST_CLI.md` | mock(use_fake_hardware) 개입 테스트 절차. 실기 위험 0 |
| `docs/testing/README.md` + `00`~`09` | 하드웨어·통신 검증 런북 인덱스와 개별 절차 |
| `docs/testing/08_OPEN_GAPS.md` | 미해결 갭 목록 (G15 = 크롭 불일치) |
| `docs/testing/09_HIL_ACTOR_RUNBOOK.md` | actor 기동 런북 (Stage A fake-env / Stage B 실센서) |
| `serl_ur_infra/tests/run_real_hil.py` | **파일 상단 주석이 실기 안전 설계를 전부 설명한다.** 터미널 구성까지 들어 있다 |

> **⚠️ `docs/testing/**` 는 워크트리 시절에 쓰였다.** `export WT=/home/laptop3/gello_worktrees/...`
> 표기가 남아 있으면 **`/home/laptop3/gello_software`로 읽는다**(§1). 절차 자체는 유효하다.

### 역사적 기록 — 지침으로 따르지 말 것

| 문서 | 왜 |
| --- | --- |
| `HIL_SERL_STATUS_AND_NEXT.md` | 2026-07-24 서버 핸드오프. **전송 계층 서술이 현행과 다르다** — upstream agentlace(5588/5589) + `train_rlpd.py --learner/--actor` + jax 0.4.35 핀을 권고하는데, 실제 구현은 자체 gRPC(50053) + `run_rlpd_learner_server.py` / `run_remote_rlpd_actor.py`이고 런타임이 jax 0.5.3을 fail-closed로 강제한다. **이대로 서버를 준비하면 `LearnerDependencyError`로 즉사한다** |
| `RL_RECEIVE_SERVER.md` | receive-only 마일스톤 기록. 현행 learner는 frozen ResNet-10 trunk 맵을 저장하고 다른 model ID를 서빙한다 |
| `HIL_RLPD_RECEIVE_SERVER_KO.md` | `feat/hil-rl-receive-server` 브랜치 시절의 작업 정리. 브랜치·워크트리 경로가 낡았다 |
| `ACTOR_ADAPTER.md` | 로컬 어댑터 `scripts/train_rlpd_actor.py` 기준. **transition 계약 설명은 여전히 유효**하나, 실기 entrypoint는 `run_remote_rlpd_actor.py`다 |
| `README.md` (serl_ur_infra) | 설계 개요. 머리말의 "UNTESTED SKELETON / 실기 사용 금지"는 낡았다 |
| `docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md`, `docs/rl/GELLO_UR7E_SERL_ENV_STATUS.md` | 초기 설계와 env 스냅샷 |

### 핵심 코드

```
serl_ur_infra/ur_experiments/cube_in_cup.py       task config (측정 크롭·박스·NETWORK·buffer_period)
serl_ur_infra/ur_env/envs/config.py               ACTION_SCALE / GOVERNOR / UPSAMPLER / HZ / 기본 박스
serl_ur_infra/ur_env/envs/ur7e_env.py             get_im, clip_safety_box, go_to_reset, _await_first_frames, ESC 리스너
serl_ur_infra/ur_env/envs/frame_wrappers.py       RelativeFrame + Quat2EulerWrapper
serl_ur_infra/ur_env/rlpd_receive_server.py       RewardClassifierRuntime, _classifier_observation, checkpoint_sha256
serl_ur_infra/ur_env/remote_actor.py              액터 루프, _dump_data(demo pickle)
serl_ur_infra/scripts/run_remote_rlpd_actor.py    액터 entrypoint (--arm/--deadman/--mock-policy-noise)
serl_ur_infra/scripts/run_rlpd_learner_server.py  실제 learner + 정책 서빙
serl_ur_infra/tests/run_real_hil.py               실기 개입 러너 (정책 zero, 서버 없음)
ros2_ur_ws/run_hil_actor.sh                       액터 실행 래퍼 + 9단계 preflight
ros2_ur_ws/run_hil_gui.sh                         데드맨/개입 GUI (/hil/deadman 20 Hz)
ros2_ur_ws/launch_cameras.sh                      카메라 (시리얼 자동 해석)
ros2_ur_ws/run_classifier_viewer.sh               ZMQ 뷰어 (랩톱 로컬 분류)
ros2_ur_ws/run_remote_classifier_viewer.sh        ZMQ 뷰어 (kanu GPU + SSH 터널)
serl_ur_infra/run_remote_reward_classifier_server.sh  ZMQ 뷰어의 kanu 쪽 서버
```

---

## 12. 진행 방식 (사용자 선호)

- 사용자는 실제 로봇이 처음이다. **한 번에 한 단계씩 가르치듯 설명하고, 명령은 사용자가 직접 실행한다.**
- 문제 해결 시 **검수·피드백 과정을 반드시 넣을 것.** 이 과정에서 값 2개가 뒤집혔고(z 바닥,
  `RESET_MAX_DIST_RAD`), classifier 크롭 불일치가 잡혔고, threshold 근거 수치의 누출도 이렇게 드러났다.
- upstream hil-serl 공식 코드가 하는 대로 따른다.
- 병렬 에이전트를 적극 사용해도 좋다. **단 kanu 접근 에이전트에게는 읽기 전용 제약을 반드시 명시할 것.**
