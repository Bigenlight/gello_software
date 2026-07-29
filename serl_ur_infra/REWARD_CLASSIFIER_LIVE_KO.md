# 실시간 reward classifier 뷰어 런북 (laptop3 로컬 CPU)

> **상태: 동작함. 2026-07-29 실기 검증 완료(운영자 직접 실행).**
> 이 경로는 **모니터 전용**이다 — 로봇에 어떤 명령도 보내지 않고, reward·done·reset을
> 발행하지 않으며, policy server가 없어도 뜬다. 카메라 토픽과 체크포인트만 있으면 된다.
>
> 🔴 **그러나 뷰어가 보여주는 확률은 RL 학습 loop가 보게 될 확률이 아니다.**
> 이유와 크기는 [§3 크롭 불일치](#3--크롭-불일치--뷰어가-보는-것은-rl-loop가-보는-것이-아니다)에 있다.
> 이 절을 읽지 않고 뷰어 수치를 learner의 reward 품질 근거로 인용하지 마라.

---

## 0. 무엇이 검증됐고 무엇이 아닌가

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| 4-터미널 절차로 실기에서 뜨고 p(success)가 갱신됨 | ✅ 검증 (2026-07-29, 운영자) | 실기 실행 |
| laptop3 CPU 단독 추론 (GPU·서버 왕복 없음) | ✅ 검증 | `jax.default_backend() == "cpu"` 재확인 |
| forward pass latency, load/JIT 시간 | ✅ 재측정 | [§9](#9-성능--실측값) |
| observation 계약(무크롭 · RGB · uint8) | ✅ 코드 대조 + 학습 텐서 bit-exact 대조 | `reward_classifier_runtime.py:81-97` |
| 파라미터 수 7,267,649 / 백본 공유 | ✅ 재측정 | [§8](#8-모델--입출력-계약) |
| all-zeros 입력의 p = 0.5957 | ✅ 재측정 | [§12 트러블슈팅](#12-트러블슈팅) |
| 체크포인트가 kanu 원본과 동일 | 🟡 **미재현** — 운영자 rsync 시점 대조만 있음 | [§6](#6-체크포인트-스테이징과-무결성) |
| 이 checkpoint의 실기 분류 성능(recall/FPR) | ❌ **미측정** — held-out 데이터셋 채점값만 있다 | [§10](#10-알려진-행동적-한계--정직하게) |
| ~~크롭 적용 시 실기 성능~~ | **해당 없음이 됐다 (2026-07-29)** — RL 경로가 이제 **무크롭 sidecar**를 쓴다. 분류기는 크롭된 그림을 아예 안 본다 | [§3](#3--뷰어와-rl-loop는-이제-같은-그림을-본다-2026-07-29부터) |
| 뷰어 확률 == 서버 확률 (같은 장면) | 🟡 **코드상 그래야 함 · 실기 미검증** — 기본 설정에서 서버도 순간 sigmoid를 보고한다. **이것이 크롭 수정의 값싼 현장 점검이다** | [§3](#3--뷰어와-rl-loop는-이제-같은-그림을-본다-2026-07-29부터) |
| 팔 가림(occlusion) | ❌ **안 고쳐졌다** — `take_21` @0.85 recall 0%. 원인은 전처리가 아니라 **시야**. 정지 게이트는 완화일 뿐 | [§3](#3--뷰어와-rl-loop는-이제-같은-그림을-본다-2026-07-29부터) |

---

## 1. 실행 — 터미널 4개

카메라(T1)와 뷰어(T2)만 있으면 된다. T3/T4는 팔을 움직여 장면을 바꾸고 싶을 때만.

```bash
# T1 — 카메라 (뷰어 창은 띄우지 않는다; 뷰어 GUI가 같은 토픽을 그린다)
cd ~/gello_software/ros2_ur_ws && VIEW=false ./launch_cameras.sh

# T2 — classifier 노드 + 모니터 GUI
cd ~/gello_software/ros2_ur_ws
REWARD_CLASSIFIER_PYTHON=/home/laptop3/venvs/hilserl/bin/python ./run_classifier_viewer.sh

# T3 — (선택) 팔을 움직이려면. 이 뷰어는 팔을 안 움직인다.
cd ~/gello_software/ros2_ur_ws && HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef

# T4 — (선택) EEF 조작 GUI
cd ~/gello_software/ros2_ur_ws && ./run_eef_gui.sh
```

`Ctrl-C` 한 번으로 T2의 두 자식 프로세스(classifier 노드, GUI)가 함께 정리된다
(`run_classifier_viewer.sh:79-98`의 `stop_child`/`trap`).

**GUI에서 보는 것**

- 파란 배너 `MONITOR ONLY — no policy execution and no robot commands from this UI`
- 배지 `LOCAL` (로컬 경로) / `REMOTE ...` ([§11](#11-원격kanu-gpu-변형--존재하지만-비권장)의 원격 경로일 때만)
- 판정: `SUCCESS`(초록, `p > threshold`) / `FAILURE`(빨강) / `WAIT / INVALID`(주황) / `OFFLINE / STALE`(회색)
- 진단 문자열: `threshold=... | camera skew=... ms | inference=... ms | status age=... s`

`OFFLINE / STALE`은 status 토픽이 1.0초 넘게 안 왔다는 뜻이다
(`classifier_view_gui.py:91-99`). classifier 노드가 죽었거나 아직 로딩 중이다.

---

## 2. 구조 — 프로세스 2개, 토픽 1개

```text
realsense2_camera (T1)
   /cam1/cam1/color/image_raw/compressed  ─┐
   /cam2/cam2/color/image_raw/compressed  ─┤
                                           ├─> reward_classifier_node   (JAX/CPU, 10 Hz)
                                           │      └─> /reward_classifier/status  (std_msgs/String, JSON)
                                           │
                                           └─> classifier_view_gui      (PyQt5, 카메라 패널 + 상태 표시)
```

- 두 프로세스는 **분리돼 있다.** JAX 초기화·추론이 Qt 이벤트 루프를 절대 막지 않게 하려는
  의도적 설계다(`reward_classifier_node.py:1-19` 모듈 docstring).
- GUI는 `gello_recorder_gui.MainWindow`를 상속해 카메라/로봇 상태 패널을 재사용하되,
  teleop·녹화·policy 컨트롤을 전부 read-only 패널로 교체한다
  (`classifier_view_gui.py:35-87`). policy 서비스 클라이언트가 **아예 없다**
  (`classifier_view_gui_node.py:17-19`).

status JSON은 `status_json()`(`reward_classifier_runtime.py:118`)이 만드는 정렬된 compact JSON이다.

```json
{"cam_skew_ms":4.2,"inference_ms":7.1,"message":"ok","probability":0.912,
 "ready":true,"success":true,"threshold":0.2}
```

| 키 | 의미 |
| --- | --- |
| `ready` | 이번 tick에서 유효한 추론이 끝났는가 |
| `probability` | sigmoid(logit). `ready=false`면 `null` |
| `success` | `probability > threshold` (**엄격한 초과**, `reward_classifier_node.py:207`) |
| `threshold` | 노드가 실제로 쓰는 값 |
| `cam_skew_ms` | cam1/cam2 ROS 헤더 stamp 차이 |
| `inference_ms` | JPEG 디코드 + forward pass 합계 (`_infer_tick` 내부 구간) |
| `message` | `ok` 또는 거절 사유 문자열 |
| `remote` | 원격 경로에서만 `true`([§11](#11-원격kanu-gpu-변형--존재하지만-비권장)) |

---

## 3. ✅ 뷰어와 RL loop는 이제 **같은 그림**을 본다 (2026-07-29부터)

> ### 🔧 이 절의 이전 판은 정반대를 경고했다 — 보존한다
> 제목부터 *"🔴 크롭 불일치 — 뷰어가 보는 것은 RL loop가 보는 것이 아니다"*였고,
> 결론은 *"**뷰어에서 `SUCCESS`가 떠도, 같은 순간 gRPC 경로의 classifier는 실패로 판정할 수 있다.**
> 뷰어 관찰을 'learner가 reward를 잘 준다'의 근거로 쓰지 마라"*였다.
> 그리고 해결 방향으로 **"크롭 조건에서 재학습"**을 제시했다.
>
> **그 경고는 2026-07-29에 해소됐고, 해결 방법도 재학습이 아니었다.**
> 액터가 분류기에게 **무크롭 전용 이미지(sidecar)**를 따로 보낸다 — 정책 크롭은 그대로 둔 채로.
> 아래 표는 **현재** 상태다. 옛 경고를 근거로 판단하지 말 것.

| | **이 뷰어 (로컬/ZMQ 경로)** | **gRPC RL 경로 (port 50053)** |
| --- | --- | --- |
| 분류기 입력 | 카메라 토픽 직접 구독 | **`classifier` sidecar** (정책 관측 아님) |
| 크롭 | **없음** | **없음** ← 같아졌다 |
| 전처리 | `decode_classifier_image()` | `decode_classifier_frames()` — **비트 단위로 같아야 하고 테스트가 강제한다** |
| 체크포인트 | `cube_in_cup_all3` | `cube_in_cup_all3` ← 같아졌다 |
| 학습 전처리와 일치? | ✅ **일치한다** | ✅ **일치한다** |
| 보고 확률 | 프레임마다 순간 sigmoid | 기본 설정에서 **같은 순간 sigmoid** (평활 OFF) |
| 분류 빈도 | 프레임마다 | **약 2 Hz, 팔 정지 시** ← **유일하게 남은 차이** |
| 로봇 | 안 움직인다 | 움직인다 |

### 🎯 그래서 이 뷰어가 **크롭 수정을 검증하는 값싼 현장 점검 도구**가 됐다

기본 설정(`--success-confirmations 1`)에서 서버가 보고하는 확률은 **순간 sigmoid 그 자체**다.
즉 **뷰어와 서버가 같은 장면에서 같은 숫자를 내야 한다.** 안 맞으면 그 자체가 버그 신호다.
이것이 크롭 불일치가 실제로 고쳐졌는지 확인하는 **가장 싼 방법**이다.

> **🪤 단 하나 남은 차이는 "언제 묻느냐"다.** 뷰어는 프레임마다, RL은 약 2 Hz · 팔 정지 시에만
> 묻는다. 그러니 **뷰어가 잠깐 1.0을 튀겨도 RL이 그 프레임을 봤다는 보장은 없다** —
> 그리고 그것이 의도다(성긴 분류는 판정 흔들림을 줄이는 설계 속성이다).
> 반대로 **RL이 본 것은 뷰어에서 재현된다.**

- 뷰어가 크롭을 안 하는 것은 **버그가 아니라 정답이다.** 이 checkpoint는 무크롭으로 학습됐다
  (kanu `examples/cube_classifier_pipeline.py`의 export가 `crop=None`).
  따라서 **뷰어 수치는 이 checkpoint에 대해 유효하다.**
- 불일치는 머지 `3f199d4`가 `serl_ur_infra/ur_experiments/cube_in_cup.py`를 이 브랜치로
  가져오면서 생겼다. 그 파일 `:211-214`에 크롭이 **채워져 있다**:

  ```python
  IMAGE_CROP: Optional[dict] = {
      "cam1": lambda img: img[20:670, 340:990],   # 650x650
      "cam2": lambda img: img[0:720, 420:1140],   # 720x720
  }
  ```

  머지 이전에는 `ur_env/envs/config.py`의 기본값 빈 dict를 쓰는 경로뿐이라 학습과 **일치했다.**
- 측정된 대가: 같은 held-out split에서 **recall@0.85가 100% → 33.3%로 떨어진다**
  (2026-07-28 측정, 무크롭 가설과 bit-exact 픽셀 일치 확인 후).

> ✅ **이제 위 문단들은 "왜 그랬는지"의 기록이다.** 크롭은 여전히 활성이지만
> **정책 관측에만** 적용되고, 분류기는 같은 스텝의 무크롭 원본을 sidecar로 따로 받는다.
>
> **`IMAGE_CROP`을 지우는 것은 여전히 답이 아니다.** 그 박스는 실측으로 정한 값이고
> (cam2는 그리퍼 파지축 x=781 중심), 지우면 크롭이 해결하려던 문제가 되돌아온다.
> `cube_in_cup.py`의 `KNOWN CONFLICT (G15)` 주석이 같은 말을 코드 옆에 적어 두고 있다.
>
> **🔴 그리고 sidecar가 고치지 못한 것이 하나 있다: 팔 가림(occlusion).**
> `take_21`은 @0.85 recall 0%, @0.05에서도 57.9%다. 팔이 cam1 시야를 쓸고 갈 때 확률이
> 0.005↔1.0으로 진동한다. 원인은 전처리도 라벨도 아닌 **시야**이고, 진짜 해결은
> **팔이 가로지르지 않는 카메라 배치**다. 뷰어로 팔을 움직여 보면 이 현상을 직접 볼 수 있다.

상세 근거: [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)
「전처리 — 이 문서의 수치는 무크롭 조건부다」절, [HANDOFF_NEXT_SESSION_KO.md](./HANDOFF_NEXT_SESSION_KO.md) §5.1/§6.

---

## 4. 환경 변수 — `run_classifier_viewer.sh`

| 변수 | 기본값 | 효과 |
| --- | --- | --- |
| `REWARD_CLASSIFIER_PYTHON` | `python3` | classifier 노드를 실행할 인터프리터. **실기에서는 반드시 지정한다** ([§5](#5-인터프리터--system-site-packages-없이-만든-이유)) |
| `REWARD_CLASSIFIER_CHECKPOINT` | `<repo>/classifier_ckpt/cube_in_cup_all3` | orbax **디렉터리**. 없으면 스크립트가 exit 2 + 스테이징 명령 출력 |
| `CLASSIFIER_THRESHOLD` | `0.2` | `threshold` ROS 파라미터로 전달. `[0,1]` 밖이면 preflight에서 실패 |
| `HIL_SERL_ROOT` | `<repo>/third_party/hil-serl` | `serl_launcher`를 `sys.path`에 넣는 데 사용 |

스크립트는 이 넷만 읽는다. **GUI(`classifier_view_gui`)는 별도로 `CAM1_NAME`/`CAM2_NAME`
(각각 기본 `cam1`/`cam2`)를 읽는다** — `classifier_view_gui.py:161-164`.

> 🪤 **카메라 이름을 바꾸면 GUI와 classifier 노드가 서로 다른 토픽을 본다.**
> `CAM1_NAME=foo`는 GUI 패널만 `/foo/foo/...`로 옮기고, classifier 노드는
> `run_classifier_viewer.sh:100-103`이 `cam1_topic`/`cam2_topic`을 **전달하지 않으므로**
> 하드코딩 기본값 `/cam1/cam1/...`을 계속 구독한다. 그러면 GUI에는 그림이 보이는데
> p(success)는 `waiting for cam1/cam2`에서 멈춘다. 토픽을 바꾸려면 노드를 직접 띄워라
> ([§13](#13-노드를-직접-띄우기-파라미터-전부-지정)).

---

## 4.1 ROS 파라미터 — `reward_classifier_node`

`reward_classifier_node.py:55-66` 선언 그대로다.

| 파라미터 | 기본값 | 의미 |
| --- | --- | --- |
| `cam1_topic` | `/cam1/cam1/color/image_raw/compressed` | 씬(삼각대) 카메라 |
| `cam2_topic` | `/cam2/cam2/color/image_raw/compressed` | **손목(그리퍼 장착)** 카메라 |
| `checkpoint_path` | `""` → env → `<repo>/classifier_ckpt/cube_in_cup_all3` | `resolve_checkpoint_path()` 우선순위 |
| `hil_serl_root` | `""` → `HIL_SERL_ROOT` env | `serl_launcher` 경로 |
| `threshold` | `default_threshold()` = `CLASSIFIER_THRESHOLD` env, 없으면 `0.2` | `success` 판정 경계 |
| `inference_hz` | `10.0` | 추론 타이머. `max(hz, 0.1)`로 하한 클램프 |
| `max_camera_age_s` | `0.5` | 프레임 수신 후 경과가 이보다 크면 거절(`camera frame stale`) |
| `max_camera_skew_s` | `0.10` | cam1/cam2 stamp 차이가 이보다 크면 거절(`camera timestamp skew too large`) |

- `age`는 **ROS stamp가 아니라 수신 시각(`time.monotonic()`)** 기준이다
  (`reward_classifier_node.py:141`, `:167`). `skew`만 ROS stamp를 쓴다.
- 매 tick은 **가장 최근 프레임 1장씩**을 쓴다. 큐도 페어링도 없다(`_frames` dict 덮어쓰기).
  원격 노드 쪽은 다르다 — 거긴 12칸 deque에서 최근접 쌍을 고른다.

---

## 5. 인터프리터 — `--system-site-packages` **없이** 만든 이유

```text
/home/laptop3/venvs/hilserl/bin/python     Python 3.10.12
  include-system-site-packages = false     <- 의도적
```

| 패키지 | 버전 |
| --- | --- |
| jax / jaxlib | 0.5.3 / 0.5.3 |
| flax | 0.10.5 |
| orbax-checkpoint | 0.11.5 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0.86 |
| optax | 0.2.4 |
| distrax | 0.1.5 |
| tensorflow-probability | 0.25.0 |
| tensorflow | **설치 안 됨 (의도적)** |

> 🪤 **`--system-site-packages`를 켜면 뷰어가 깨진다. 이건 실제로 밟은 함정이다.**
>
> 그 플래그는 `~/.local/lib/python3.10/site-packages`까지 열어 준다. 그 안에는
> **numpy 2.2.6**과 **`easy-install.pth`** 가 있고, 후자의 내용은 딱 한 줄이다:
>
> ```text
> /home/laptop3/gello_software/serl_ur_infra
> ```
>
> 그러면 `serl_ur_infra/tensorflow/` (**annotation 전용 shim**)이 진짜 `tensorflow`로
> import되고, flax는 TF가 있다고 판단해 TF I/O 백엔드를 고른다. 결과는 `restore_checkpoint`
> 단계에서 다음 예외다:
>
> ```text
> RuntimeError: TensorFlow I/O is unavailable in the HIL-SERL annotation-only
> compatibility shim (requested tf.io.gfile.<...>)
> ```
>
> (`serl_ur_infra/tensorflow/io/__init__.py`의 `_UnavailableGFile`.)
> shim은 `serl_launcher.common.typing`의 `tf.Tensor` 어노테이션만 만족시키려고
> 존재한다. 없애면 안 되고, classifier venv에서 보이면 안 된다.

**그럼 rclpy는 어떻게 import되나 — site-packages가 아니라 `PYTHONPATH`다.**
`source /opt/ros/humble/setup.bash`가 `PYTHONPATH`에 ROS 경로를 넣으므로
`--system-site-packages` 없이도 그대로 잡힌다. 확인:

```bash
source /opt/ros/humble/setup.bash
/home/laptop3/venvs/hilserl/bin/python -c "import rclpy; print(rclpy.__file__)"
# /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py
# 같은 세션에서 numpy는 venv의 1.26.4가 잡혀야 한다.
```

venv 안 `.pth` 두 개가 나머지를 메운다(둘 다 의도된 것):

```text
serl_launcher.pth            -> /home/laptop3/gello_software/third_party/hil-serl/serl_launcher
zz-system-dist-packages.pth  -> /usr/lib/python3/dist-packages     # apt 패키지(~/.local 아님)
```

> ⚠️ **스크립트 주석은 아직 틀렸다.** `run_classifier_viewer.sh:38`과
> `run_reward_classifier_gui.sh:43`은 여전히
> "The classifier venv must therefore also be created with `--system-site-packages`"라고
> 적혀 있다. **실제 동작 중인 venv는 `include-system-site-packages = false`다.**
> 주석이 아니라 이 문서와 `pyvenv.cfg`를 믿어라. (코드 수정 범위 밖이라 이번엔 안 고쳤다.)

---

## 6. 체크포인트 스테이징과 무결성

정본은 **orbax OCDBT 디렉터리**다. flax msgpack 단일 파일이 아니다.

```text
classifier_ckpt/cube_in_cup_all3/
└── checkpoint_150/           <- 43 MB, 파일 14개 (디렉터리 제외)
    ├── _CHECKPOINT_METADATA
    ├── _METADATA
    ├── _sharding
    ├── manifest.ocdbt
    ├── array_metadatas/process_0
    ├── d/00fe021b5ce77da67639cdee5f4529e2
    └── ocdbt.process_0/{manifest.ocdbt, d/* (7개)}
```

`classifier_ckpt/.gitignore`가 `*`로 전부 무시한다 — **git에 없다. 매번 스테이징해야 한다.**

```bash
mkdir -p /home/laptop3/gello_software/classifier_ckpt/cube_in_cup_all3
rsync -a kanu:'~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150' \
      /home/laptop3/gello_software/classifier_ckpt/cube_in_cup_all3/
```

(스크립트 오류 메시지는 `.../classifier_ckpt/` 전체를 받는 변형을 안내한다. 둘 다
`cube_in_cup_all3/checkpoint_150/`으로 떨어지면 된다.)

**ResNet-10 사전학습 가중치도 필요하다.** 없으면 upstream `create_classifier()`가 실행 중에
GitHub에서 다운로드를 시도한다(`third_party/hil-serl/.../reward_classifier.py:91-107`).
`run_classifier_viewer.sh`의 preflight가 그걸 막으려고 존재 여부를 먼저 검사한다.

```bash
sha256sum ~/.serl/resnet10_params.pkl
# 175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b
```

### 디렉터리 digest

단일 파일이 아니라 `sha256sum` 한 줄로 끝나지 않는다. **digest를 인용할 때는 반드시
계산식을 같이 적는다** — 식이 다르면 값도 다르다.

```bash
cd /home/laptop3/gello_software/classifier_ckpt/cube_in_cup_all3
python3 - <<'PY'
import hashlib, os
root = "checkpoint_150"
ent = []
for dp, dn, fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        ent.append((os.path.relpath(p, root),
                    hashlib.sha256(open(p, "rb").read()).hexdigest()))
ent.sort()
outer = hashlib.sha256()
for rel, h in ent:            # relpath 와 per-file sha256 을 이어붙인다
    outer.update(rel.encode()); outer.update(h.encode())
print(len(ent), outer.hexdigest())
PY
# 14 4e9c81494eb0f13441882f92220811f8ae1c3d9d95a3b1cdf6fbc75cb8500668
```

> ⚠️ **다른 값(`6185796c…`)이 인수인계 메모에 돌아다닌다.** 위 식으로는 재현되지 않고,
> 구분자(개행/콜론/공백)·기준 경로(`checkpoint_150` 상대 vs 상위 상대)를 바꾼 6가지 변형에서도
> 재현되지 않았다(2026-07-29 확인). **어느 쪽이 맞는지가 아니라 "식 없는 digest는 쓸 수 없다"는
> 것이 결론이다.** 위 값은 이 트리에서 실제로 계산한 값이고, kanu 원본과의 대조는
> 이 문서에서 **재현하지 않았다**(운영자 rsync 시점 대조만 존재).

> ℹ️ gRPC 경로의 `checkpoint_sha256()`(`ur_env/rlpd_receive_server.py:148-159`)은
> `os.path.isfile()`을 강제하므로 이 디렉터리를 **아직 받지 못한다.** 뷰어 경로에는 SHA pin이
> 없어서 그냥 된다. 그래서 지금 두 경로는 서로 다른 체크포인트를 보고 있다
> ([HANDOFF_NEXT_SESSION_KO.md](./HANDOFF_NEXT_SESSION_KO.md) §5.2).

---

## 7. 🪤 조용한 실패 — `restore_checkpoint`는 없는 경로에 에러를 내지 않는다

flax `checkpoints.restore_checkpoint(path, target=classifier)`는 경로가 없거나 읽을 수
없으면 **예외 없이 `target`을 그대로 돌려준다.** 즉 **무작위 초기화된 네트워크**가
`load_classifier_func`의 반환값이 되고, 뷰어는 그럴듯한 숫자를 자신 있게 표시한다.
이건 조용히 틀린 reward를 보는 최악의 형태다.

그래서 `resolve_checkpoint_path()`(`reward_classifier_runtime.py:44-62`)가 **먼저 죽는다.**

```python
if not os.path.exists(path):
    raise RuntimeError(_MISSING_CHECKPOINT_HELP.format(...))
```

`reward_classifier_node.py:110-112`와 `remote_reward_classifier_server.py:31-34`에
"왜 이 검사가 먼저인지"가 주석으로 못 박혀 있다.

> 🔴 **이 검사를 제거하거나 우회하지 마라.** 제거하면 실패 모드가
> "안 뜬다"에서 "틀린 숫자를 자신 있게 보여준다"로 바뀐다.
> 존재 검사는 통과하되 내용이 깨진 디렉터리는 여전히 이 함정에 걸린다 —
> 그래서 [§12](#12-트러블슈팅)의 `0.5957` 신호와 [§6](#6-체크포인트-스테이징과-무결성)의
> digest가 필요하다.

---

## 8. 모델 / 입출력 계약

**입력** (`make_observation`, `reward_classifier_runtime.py:91-97`):

```python
{
  "state": np.zeros((1, 1), dtype=np.float32),      # 실제로 안 쓰이지만 shape은 계약이다
  "cam1":  uint8 (1, 128, 128, 3),  RGB, 0-255
  "cam2":  uint8 (1, 128, 128, 3),  RGB, 0-255
}
```

**전처리** (`decode_classifier_image`, `:81-88`):

```text
cv2.imdecode(..., IMREAD_COLOR)   -> BGR (720, 1280, 3)
크롭 없음                          <- 학습과 일치. §3
cv2.resize(bgr, (128, 128))       <- 1280x720을 1:1로 눌러 담는다(가로가 세로의 0.5625배로 압축)
[..., ::-1]                       -> RGB
[None, ...]                       -> (1, 128, 128, 3)
```

- **255로 미리 나누지 마라.** 모델이 내부에서 정규화한다. uint8 0-255를 그대로 넣는다.
- `image_keys=["cam1","cam2"]` **순서는 아키텍처의 일부다.** 임베딩이 그 순서로 이어붙어
  head Dense에 들어간다. 바꾸면 다른 모델이다.
- 출력은 **raw logit 스칼라**다. sigmoid는 호출자가 적용한다
  (`sigmoid_probability`, `:100-115`. 오버플로 안전한 분기 구현).

**아키텍처** (재측정, 2026-07-29):

```text
총 파라미터            7,267,649
ResNet-10 백본         5,418,792 (frozen, ImageNet-1K 사전학습)
출력 유닛              1  (binary)
백본 공유              예 — params 트리에서 'pretrained_encoder'는 encoder_cam1 아래에만 존재하고
                       encoder_cam2에는 없다. 같은 모듈 인스턴스를 두 인코더가 공유한다.
카메라별 개별 파라미터  SpatialLearnedEmbeddings_0 / Dense_0 / LayerNorm_0 (각 인코더에 1벌씩)
```

---

## 9. 성능 — 실측값

**laptop3 CPU에서 돈다. GPU도, 서버 왕복도 없다.**

| 항목 | 값 | 출처 |
| --- | --- | --- |
| forward pass (JIT 후) | **6.6 ms** (운영자) / 재측정 median **7.1 ms**, p90 8.8 ms | 30회 반복 |
| 720p JPEG 디코드 + 리사이즈 (카메라 1대) | 약 2.1 ms | 50회 반복, 68 KB JPEG |
| 디코드 포함 1회 추론 | **11.9 ms → 약 84 Hz 여력** | 운영자 실측 |
| 실제 동작 주파수 | **10 Hz** (`inference_hz` 기본값) | 여력의 1/8 |
| 체크포인트 로드 + JIT 워밍업 | 약 **5.6–5.9 초**, **1회성** | 재측정 5.92 s |
| JAX backend | `cpu` | `jax.default_backend()` |

노드는 구독 시작 **전에** 더미 입력으로 한 번 호출해 컴파일을 끝낸다
(`reward_classifier_node.py:134-137`). 그래서 첫 화면 숫자에 JIT 지연이 섞이지 않는다.
대신 T2를 띄우고 약 6초 동안은 GUI가 `OFFLINE / STALE`이다. 정상이다.

### 왜 kanu 원격을 버렸나

| | 로컬 CPU | kanu 원격 |
| --- | --- | --- |
| 추론 | 6.6 ms | GPU라 더 빠름 |
| 전송 | 0 | **관측 1개 ≈ 96.1 KiB** |
| 링크 | — | **5.7 MiB/s ≈ 47.8 Mbit/s** (그때의 실측) |
| 전송만으로 | 0 ms | **≈ 16 ms** (96.1 KiB ÷ 5.7 MiB/s) |
| 합계 | **11.9 ms** | 16 ms + 추론 + RTT + SSH 터널 |

**즉 순수 전선 시간만으로도 로컬 전체 경로보다 느리다.** GPU가 아무리 빨라도 이길 수 없는
구조다. 그래서 원격 경로는 [§11](#11-원격kanu-gpu-변형--존재하지만-비권장)로 강등했다.

> ### 🔧 정정 (2026-07-29) — 위 표의 "공용 IP 경유"는 **틀렸다**
> 이전 판은 링크를 *"공용 IP 경유 5.7 MiB/s"*로 적었다. **인터넷 전송 구간은 없다.**
> `tracepath`가 **캠퍼스 4홉**만 보여준다:
> `192.168.0.1 → 10.20.44.1 → 10.22.2.101 → kanu`.
> 대역폭 숫자 자체는 그날의 실측으로 유효하니 남기고, 경로 서술만 바로잡는다.

> ### 🚨 그리고 이 링크 수치를 상수로 쓰지 마라 — 세션 간 **약 6배** 흔들린다
> 리포에 저장된 값이 셋인데 전부 다르고, **셋 다 진짜 관측일 가능성이 높다:**
>
> | 출처 | 실효 대역폭 |
> | --- | --- |
> | 이 문서 (위 표) | 5.7 MiB/s ≈ **47.8 Mbit/s** |
> | `HANDOFF_NEXT_SESSION_KO.md` §0 (07-27 판) | 약 **13 Mbit/s** |
> | **2026-07-29 실측** | 약 **83 Mbit/s** (유휴 리그, 2.4 GHz, 주변 AP 41개) |
>
> 13 Mbit/s는 오류가 아니라 **열화된 상태의 실제 관측**이었을 것이고 **다시 나타날 수 있다.**
> **세션 시작할 때 다시 재는 것이 유일하게 옳은 절차다** — 07-29 값 포함이다.
> 자세한 조건은 `HANDOFF_NEXT_SESSION_KO.md` §0 / `docs/testing/08_OPEN_GAPS.md` G16.
>
> **어느 수치를 써도 이 절의 결론(로컬 CPU가 낫다)은 같다.**
> 🔌 변동성을 통째로 없애려면 USB 이더넷 NIC `enx00e04c3600bd`를 꽂으면 된다 — 존재하는데 안 꽂혀 있다.

---

## 10. 알려진 행동적 한계 — 정직하게

아래는 전부 **held-out 데이터셋 오프라인 채점**이다. 무크롭 입력, 배포 checkpoint `all3` 기준.
**실기에서의 recall/FPR은 측정된 적이 없다.**

| 항목 | 값 |
| --- | --- |
| pooled held-out recall @0.5 | **86.8%** (0720 held-out positive 266프레임 = test 166 + val 100) |
| held-out negative FPR @0.85 / 0.5 / 0.2 | **정확히 0%** (0/470 = test 284 + val 186, 6 take) |
| 최악 take: `take_21_20260720_210234` (n=38) | @0.85 **0.0%**, @0.5 **7.9%**, @0.2 21.1%, @0.05 57.9% |
| take_03 (0724) | @0.85 **44.3%** — 다른 take는 96%대 |
| 동일 데이터 재학습의 take별 recall 변동 | 최대 **21%p** (시드 분산) |

> ⚠️ **"470프레임 FPR 0%"를 그대로 믿지 마라.** 그중 446프레임(95%)은 큐브가 컵 근처에도
> 안 가는 "쉬운" 프레임이다. 경계를 실제로 시험하는 표본은 **24프레임**뿐이다.
> take 단위로 보면 n=6이고 rule-of-three 상한은 **50%**다.

**확률 진동의 원인은 팔이다.** take_03은 확률이 1~2초 주기로 0.005~1.0을 톱니처럼 오간다.
cam1에서 로봇 팔이 화면 상단에서 내려오는 정도와 프레임 단위로 역상관한다
(t=8.40s 팔 위축 p=0.93 → t=9.21s 팔 하강 p=0.31). take_21은 같은 병리가 더 심한 경우로,
**threshold를 아무리 낮춰도 구제되지 않는다**(0.05에서도 57.9%).

→ 뷰어를 보다가 큐브가 컵 안에 있는데 판정이 `FAILURE`로 뒤집히면, 그건 대체로 **모델이
틀린 게 아니라 팔이 프레임을 지나가고 있는 것**이다. 이 실패 모드의 해법은 threshold가 아니라
**팔이 정지한 자세에서만 질의하거나 N-of-M 시간 평활**이다.

---

## 10.1 threshold

기본값 **0.2**다. 근거는 통계가 아니라 **비용 비대칭**이다 — sparse binary reward에서
**false positive는 복구 불가능**(episode가 잘못 성공 종료되고 그 transition이 버퍼에 남는다)인 반면
**false negative는 사람이 개입해 메울 수 있다.**

수치·측정 설계·이력(0.85 → 0.5 → 0.2)은 여기서 되풀이하지 않는다.
권위 있는 문서는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)이고,
권위 있는 코드값은 `ur_env/rlpd_receive_server.py:73`의 `DEFAULT_REWARD_THRESHOLD` 하나다.
뷰어 쪽 기본값(`reward_classifier_runtime.py:18`, `run_classifier_viewer.sh:20`)은 **그것과
동기화하라고 주석으로 못 박혀 있다.** 한쪽만 바꾸면 뷰어 판정과 learner reward가 어긋난다.

---

## 11. 원격(kanu GPU) 변형 — 존재하지만 **비권장**

**결론부터: 쓰지 마라. [§9](#9-성능--실측값)에서 로컬 CPU가 이겼다.**
아래는 코드가 남아 있으니 무엇인지 알아두라는 기록이다.

| 파일 | 위치 | 역할 |
| --- | --- | --- |
| `serl_ur_infra/remote_reward_classifier_server.py` | kanu | ZMQ **REP**, 기본 `--bind tcp://127.0.0.1:5594`. ROS 의존 없음 |
| `serl_ur_infra/run_remote_reward_classifier_server.sh` | kanu | 위 서버 기동. `REQUIRE_JAX_GPU=true`가 기본이라 CPU면 거부 |
| `gello_recorder/remote_reward_classifier_node.py` | laptop3 | 카메라 구독 → ZMQ **REQ**. `ros2 run gello_recorder remote_reward_classifier` |
| `ros2_ur_ws/run_remote_classifier_viewer.sh` | laptop3 | **SSH 터널(`-L 5594`)을 스스로 연다** + 위 노드 + 같은 GUI |

- 프로토콜: `reward-classifier-v1`, 3-프레임 multipart `[JSON 헤더, cam1 JPEG, cam2 JPEG]`,
  응답은 JSON 1프레임. `request_id` 불일치 응답은 거부한다
  (`remote_classifier_runtime.py:23-63`).
- 백프레셔: in-flight 1건만 유지하고 나머지는 **최신으로 덮어쓴다**
  (`RemoteInferenceWorker.submit`, `:85-96`). 큐가 쌓이지 않는다.
- 프레임 페어링이 로컬 노드와 **다르다**: 12칸 deque에서 stamp 최근접 쌍을 고르고
  `max_camera_skew_s` 안에 들어올 때만 채택한다(`remote_reward_classifier_node.py:57-72`).
- GUI 배지가 `REMOTE CONNECTED`/`REMOTE STALE`로 바뀌고 진단에 `roundtrip_ms`,
  `capture_age_ms`가 추가로 붙는다.
- 추가 환경변수: `CLASSIFIER_SSH_HOST`(기본 `kanu`), `CLASSIFIER_LOCAL_PORT`/
  `CLASSIFIER_REMOTE_PORT`(기본 5594), `CLASSIFIER_TIMEOUT_S`(기본 2.0).

> 🪤 **쓰려고 해도 지금은 안 될 가능성이 높다.** `run_remote_reward_classifier_server.sh`와
> `remote_reward_classifier_server.py`는 최근 커밋에서 새로 생긴 파일이고, kanu의 worktree는
> 머지 이전 커밋에 고정돼 있어 **그 파일들이 없다.** kanu의 별도 저장소
> `~/workspace/youngwoong/gello_software_remote_classifier`에는 **폐기된 Jul-24 체크포인트**가
> 있으니 그쪽을 쓰면 recall 0%짜리 모델을 보게 된다.

---

## 12. 트러블슈팅

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| GUI 카메라 패널이 검고 판정이 `WAIT / INVALID`, message = `waiting for cam1/cam2` | T1이 안 떴거나 토픽 이름이 다름 | `ros2 topic hz /cam1/cam1/color/image_raw/compressed`로 확인. [§4](#4-환경-변수--run_classifier_viewersh)의 `CAM1_NAME` 함정도 확인 |
| T1이 `fewer than 2 connected devices` / 시리얼 경고 | RealSense 시리얼이 저장소 기본값과 다름 | 자동 해석이 이미 있다 — [§14](#14-카메라-시리얼은-자동-해석된다) |
| `Reward-classifier checkpoint not found: ...` (exit 2) | orbax 디렉터리가 스테이징 안 됨 | [§6](#6-체크포인트-스테이징과-무결성)의 rsync |
| `classifier checkpoint not found: ...` (RuntimeError, 노드 쪽) | 파라미터/env/기본값 전부 해석 실패 | 같은 조치. **이 에러는 기능이다** — [§7](#7--조용한-실패--restore_checkpoint는-없는-경로에-에러를-내지-않는다) |
| `RuntimeError: TensorFlow I/O is unavailable in the HIL-SERL annotation-only compatibility shim` | venv가 `~/.local`을 보고 있고 `easy-install.pth`가 `serl_ur_infra`를 열어 TF shim이 진짜 tensorflow로 잡힘 | `--system-site-packages` **없이** 만든 venv를 쓴다. [§5](#5-인터프리터--system-site-packages-없이-만든-이유) |
| **p(success)가 `0.596` 근처에 얼어붙음** | **`0.5957`은 전부 0인 더미 입력의 값이다**(재측정: logit 0.3877 → p 0.595731). 이미지가 실제로 안 들어오고 있다는 뜻 | 토픽 확인. GUI에 그림이 보이는데도 이 값이면 classifier 노드만 다른 토픽을 보고 있는 것 |
| 판정이 장면과 반대로 움직임 / 손목 뷰가 이상함 | cam1↔cam2 뒤바뀜. **cam1 = 씬(삼각대, D435), cam2 = 손목(그리퍼 장착, D435if)** | T1 로그의 `cam1 = SCENE ... cam2 = WRIST` 줄과 GUI 패널을 대조. 시리얼 수동 지정: `CAM1_SERIAL=... CAM2_SERIAL=... ./launch_cameras.sh` |
| message = `camera frame stale (0.7xx s)` | 수신 후 경과 > `max_camera_age_s`(0.5) | 카메라 노드가 죽었거나 USB 대역폭 부족. 720p/30fps 두 대가 같은 USB 컨트롤러에 물렸는지 확인 |
| message = `camera timestamp skew too large` | cam1/cam2 stamp 차이 > `max_camera_skew_s`(0.10) | 한쪽 카메라가 프레임을 흘리고 있다. 임계값을 올리기 전에 원인을 먼저 본다 |
| message = `inference error: ...` | JPEG 디코드 실패 등 | 노드는 죽지 않고 계속 진단을 발행한다. T2 로그의 `classifier inference failed:` 확인 |
| GUI가 계속 `OFFLINE / STALE` | status가 1초 넘게 안 옴. 기동 후 약 6초는 정상(로드+JIT) | 6초 넘게 지속되면 T2 로그를 본다. classifier 프로세스가 죽으면 스크립트가 GUI도 함께 닫는다 |
| 소스를 고쳤는데 반영 안 됨 | install space는 **심볼릭 링크가 아니라 복사본**이다 | `cd ~/gello_software/ros2_ur_ws && ./build_ur7e.sh` (또는 해당 패키지 colcon build) 후 재실행 |
| preflight가 `Reward-classifier preflight failed.`로 끝남 | `cv2/flax/jax/optax/rclpy/requests/sensor_msgs/std_msgs/tqdm`, `gello_recorder`, `serl_launcher` 중 하나가 import 실패 | 실패한 import를 직접 재현한다: `source /opt/ros/humble/setup.bash && source install/setup.bash && /home/laptop3/venvs/hilserl/bin/python -c "import jax, flax, cv2, rclpy"` |

---

## 13. 노드를 직접 띄우기 (파라미터 전부 지정)

토픽 변경, 주파수 변경, 임계값 실험 등 스크립트가 노출하지 않는 것을 만질 때.

```bash
cd ~/gello_software/ros2_ur_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export HIL_SERL_ROOT=~/gello_software/third_party/hil-serl
export PYTHONPATH="$HIL_SERL_ROOT/serl_launcher${PYTHONPATH:+:$PYTHONPATH}"

/home/laptop3/venvs/hilserl/bin/python -m gello_recorder.reward_classifier_node --ros-args \
  -p hil_serl_root:="$HIL_SERL_ROOT" \
  -p checkpoint_path:=/home/laptop3/gello_software/classifier_ckpt/cube_in_cup_all3 \
  -p threshold:=0.2 \
  -p inference_hz:=10.0 \
  -p max_camera_age_s:=0.5 \
  -p max_camera_skew_s:=0.10 \
  -p cam1_topic:=/cam1/cam1/color/image_raw/compressed \
  -p cam2_topic:=/cam2/cam2/color/image_raw/compressed

# 다른 터미널에서 GUI만
ros2 run gello_recorder classifier_view_gui
# 또는 GUI 없이 원시 상태만
ros2 topic echo /reward_classifier/status
```

`ros2 run gello_recorder reward_classifier`도 같은 노드를 띄우지만, 그러면 **시스템 python3**가
쓰여 jax/flax를 못 찾는다. `-m`으로 venv 인터프리터를 명시하는 이유가 그것이다.

---

## 14. 카메라 시리얼은 자동 해석된다

코드 기본 시리얼은 **`147122072740`(cam1, plain D435) / `243222072700`(cam2, D435IF)**이다
(`launch_cameras.sh:69-70`). `launch_cameras.sh`는 기동 전에 **살아 있는 USB 버스와 대조해
시리얼을 재배정한다**(공용 헬퍼 `ros2_ur_ws/_resolve_camera_serials.sh`, 커밋 `43ba314`·`fb48100`).

> ### 🔧 정정 — 이전 판이 적은 기본 시리얼은 **틀렸다**
> 이전 판은 *"저장소에 적힌 기본 시리얼(`151623020789` / `322743060038`)"*이라고 적었다.
> **그 둘은 ASIC 시리얼이고 `serial_no:=`로는 영원히 해석되지 않는다.**
>
> **카메라 쌍은 하나뿐이다. 두 쌍처럼 보인 것은 시리얼 *필드*가 둘이기 때문이다** —
> 2026-07-29 직접 측정, 같은 물리 USB 포트가 두 값을 동시에 보고한다:
>
> | 포트 | `serial_number` (← `serial_no:=`가 매칭) | `asic_serial_number` | 장치 |
> |---|---|---|---|
> | `4-4.1` | **`147122072740`** | `151623020789` | plain D435 → cam1 |
> | `4-4.3` | **`243222072700`** | `322743060038` | D435IF → cam2 |
>
> **🚫 이것을 저널 grep(`journalctl`, `/sys/bus/usb/devices/*/serial`)으로 다시 유도하지 말 것.**
> 커널은 **ASIC 시리얼**을 노출하므로 몇 번을 해도 "쌍이 바뀌었다"는 같은 오답이 나온다.
> 확인은 `rs-enumerate-devices` 또는 `camera_info`의 **`serial_number` 필드**로 한다.
> 정본 설명은 `ros2_ur_ws/launch_cameras.sh` 머리 주석에 있다.

요약만 적는다 — 전체 동작은 그 파일 상단 주석에 있고, 여기서 복제하지 않는다.

- 설정된 두 시리얼이 **둘 다** 꽂혀 있으면 조용히 통과.
- 아니면 **모델 클래스로 배정**한다(장치명에 `d435i` 포함 → cam2, 평범한 D435 → cam1). WARN 출력.
- 두 대가 같은 모델 클래스면 모호하므로 시리얼 정렬 순서로 떨어뜨리고 **강한 WARN**을 낸다 —
  이때는 **녹화/판정 전에 GUI 패널을 눈으로 확인하라.**
- 2대 미만이면 **하드 에러**로 스크립트를 종료한다.
- `pyrealsense2`가 없으면 SKIP(치명적 아님).

수동 우회: `CAM1_SERIAL=... CAM2_SERIAL=... VIEW=false ./launch_cameras.sh`

---

## 15. 관련 파일

| 경로 | 역할 |
| --- | --- |
| `ros2_ur_ws/run_classifier_viewer.sh` | **정본 진입점** (로컬 CPU) |
| `ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_node.py` | ROS 노드: 구독 → 추론 → status 발행 |
| `ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_runtime.py` | ROS 비의존 헬퍼: 경로 해석, 전처리, sigmoid, status JSON |
| `ros2_ur_ws/src/gello_recorder/gello_recorder/classifier_view_gui.py` | 모니터 GUI 창(읽기 전용 패널) |
| `ros2_ur_ws/src/gello_recorder/gello_recorder/classifier_view_gui_node.py` | GUI 뒤의 ROS 노드 (policy 클라이언트 없음) |
| `ros2_ur_ws/run_reward_classifier_gui.sh` | 같은 classifier 노드 + **policy_run_gui**. 정책 컨트롤이 붙으므로 모니터 전용이 아니다 |
| `ros2_ur_ws/launch_cameras.sh`, `ros2_ur_ws/_resolve_camera_serials.sh` | 카메라 기동 / 시리얼 자동 해석 |
| `serl_ur_infra/remote_reward_classifier_server.py` 외 3종 | 원격 변형([§11](#11-원격kanu-gpu-변형--존재하지만-비권장)) |
| `serl_ur_infra/ur_experiments/cube_in_cup.py:211-214` | **`IMAGE_CROP`** — 크롭 불일치의 출처([§3](#3--크롭-불일치--뷰어가-보는-것은-rl-loop가-보는-것이-아니다)) |
| `serl_ur_infra/ur_env/rlpd_receive_server.py:73` | `DEFAULT_REWARD_THRESHOLD` — threshold의 유일한 권위 |
| `serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md` | threshold 근거와 채점 결과 전문 |
| `serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md` §5–§6 | 두 경로(gRPC vs 뷰어) 비교, 체크포인트 pin 현황 |
