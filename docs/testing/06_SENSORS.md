# 06 — 센서 (RealSense · QoS · 19-D state 계약)

**상태: 부분 검증 (2026-07-27 실기 세션 반영).**

| 항목 | 상태 |
|---|---|
| cam1/cam2 역할 정의 | **정정됨** — cam2는 **손목** 카메라 (§1.1) |
| QoS 호환성 | **PASS** — RealSense는 RELIABLE/TRANSIENT_LOCAL (§3) |
| 19-D state 레이아웃 | **PASS** — 알파벳순, 그리퍼 index 0 (§5.2) |
| F/T wrench 프레임 | **정정됨** — tool0가 맞고 upstream과 일치 (§5.3) |
| RealSense 2대 동시 스트림 안정성 | **미검증** (§1.2, §2) |
| 7개 토픽 유량 루프 | **미검증** (§3.1) |

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## 1. RealSense 2대

### 1.1 계약 — 🔧 **cam2는 손목 카메라다 (정정 2026-07-27)**

> ### 🔧 시리얼 정정 (2026-07-28) — 카메라 개체가 교체됐다
> 이전 판의 `147122072740` / `243222072700`은 **이 PC가 커널 로그상 한 번도 본 적 없는
> 하드웨어**다(2026-07-05까지 소급 확인). 아래 표는 실제 연결된 개체로 갱신했다.
> **없는 시리얼로 바인딩하면 조용히 안 뜬다** — 오류가 아니라 "프레임 없음"으로 보인다.
> 아직 옛 시리얼이 남은 곳: `docs/ros2/GELLO_UR7E_{ACT,DIFFUSION,FM}_DEPLOY.md`,
> `ros2_ur_ws/src/gello_{policy,recorder}/README.md`, 별도 저장소 `gello_software_remote_classifier`.
>
> **미확정**: 연결된 두 대 중 어느 개체가 손목에 달렸는지. 아래 배정은 모델 클래스 추론이고
> USB 포트 순서도 녹화 당시와 뒤바뀌었다. 팔을 흔들어 cam2 창을 보면 끝난다.

| | cam1 | cam2 |
|---|---|---|
| 역할 | **SCENE** (삼각대, 3인칭, 고정) | 🔧 **WRIST** — 그리퍼에 **강체로 장착**. 팔과 함께 움직인다 |
| 시리얼 | `151623020789` (plain D435) | `322743060038` (D435IF) |
| 토픽 | `/cam1/cam1/color/image_raw/compressed` | `/cam2/cam2/color/image_raw/compressed` |
| 프로파일 | `1280x720x30` (둘 다 동일) | 동일 |
| `cube_in_cup` 크롭 | `img[20:670, 340:990]` (650×650) | `img[0:720, 420:1140]` (720×720) |

근거: `serl_ur_infra/ur_experiments/cube_in_cup.py`의 `IMAGE_CROP` 주석 —
> *"cam1 is the fixed tripod scene camera. cam2 is the WRIST camera, rigidly mounted to the
> gripper: its crop is centred on the measured grasp axis (x=781, midway between the
> fingertips at x 512-563 and x 1000-1117) rather than on a table region, because the
> fingers stay at fixed pixels while the background moves with the arm."*

> ### 🔧 폐기된 서술 (왜 틀렸는지가 정보다)
> | 이전 판 | 실제 |
> |---|---|
> | "cam2 = CLOSE-UP (책상 위 작업 근접 고정 카메라)" | **손목 카메라다.** 그리퍼와 함께 움직인다 |
> | "cam2가 작업 영역을 안 본다 / 조준이 틀렸다" | **폐기.** 손목 카메라는 항상 그리퍼 앞을 본다. 손가락이 **자세 불변으로 같은 픽셀**에 있고 배경이 팔 자세를 따라 바뀌는 것이 정상 동작이다 |
> | (크롭 기준을 "책상 영역"으로 잡으려 함) | 크롭 기준은 **측정된 grasp 축(x=781)**이다. 테이블 영역 기준으로 잡으면 안 된다 |
>
> ⚠️ **`ros2_ur_ws/launch_cameras.sh`의 머리말 주석은 아직 "cam2 = CLOSE-UP (workspace)"로
> 낡아 있다** (그 파일은 이 문서의 소유 범위 밖이다). 뷰어 육안 확인(§1.3)의 판정 문구는
> 그 주석이 아니라 이 표를 따른다.

**손목 카메라라서 달라지는 것 — 판정할 때 이걸 본다:**

| | 고정 카메라라면 | 손목 카메라이므로 (실제) |
|---|---|---|
| 팔을 움직였을 때 | 배경 고정, 팔만 움직임 | **배경 전체가 흐른다** |
| 그리퍼 손가락 위치 | 자세마다 다름 | **항상 같은 픽셀** (자세 불변) |
| "화면에 물체가 없다" | 카메라 조준 문제 | **팔이 다른 곳을 보고 있을 뿐** — 정상일 수 있다 |
| 크롭 좌표 | 책상 기준 | **그리퍼 기준** (grasp 축 중심) |

### 1.2 기동

```bash
cd $WT/ros2_ur_ws
./launch_cameras.sh              # 뷰어 포함 (기본)
VIEW=false ./launch_cameras.sh   # 뷰어 없이
```

- 스크립트가 **두 스트림이 실제로 ~25 Hz 이상 흐를 때까지 최대 30초 대기**하고,
  실패하면 명확히 죽는다 (`launch_cameras.sh`의 `wait_for_stream`).
- 로그는 `/tmp/launch_cameras_<timestamp>/cam{1,2}_launch.log`.
- **Ctrl-C 한 번으로 뷰어 + 카메라 2대가 깨끗이 정리된다.** 고아 프로세스가 남지 않도록
  `_kill_and_wait`(유예 후 SIGKILL) + 이름 기반 `pkill` 백스톱까지 들어 있다.

### 1.3 🛑 cam1/cam2 매핑 육안 확인은 안전 관련 절차다

뷰어가 **기본 ON**인 이유가 있다. 매핑이 뒤집히면 학습된 정책이 **조용히** 열화된다
(`launch_cameras.sh:12-16`). 판정 기준은 §1.1의 정정된 표를 쓴다:

- **좌측 창 = cam1 = 전체 장면.** 팔을 움직여도 **배경이 고정**돼 있다.
- **우측 창 = cam2 = 손목.** 팔을 움직이면 **화면 전체가 흐르고**, 그리퍼 손가락은
  **같은 자리에 남는다.** ← 이 한 가지 동작으로 두 카메라를 즉시 구별할 수 있다.

바뀌어 있으면 Ctrl-C하고 시리얼부터 다시 확인한다.

### 1.4 `serial_no`는 반드시 따옴표로 감싼다

```bash
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam1 camera_namespace:=cam1 "serial_no:='151623020789'" \
    "rgb_camera.color_profile:='1280x720x30'"
```

`ros2 launch`는 CLI 인자 타입을 내용으로 추론한다. **전부 숫자인 시리얼을 그냥 넘기면
정수로 강제 변환**되고, `serial_no`는 STRING 파라미터라 노드가 즉사한다
(`launch_cameras.sh:130-134`). 이건 이 리포에서 실제로 물린 함정이다.

---

## 2. 카메라 문제 해결

### 2.1 문서화된 근본 원인: `uvcvideo` 전역 quirks

**"Frame didn't arrive within 5000/8000"**의 확인된 원인은 USB 대역폭이 아니라
`/etc/modprobe.d/uvcvideo-actioncam.conf`의 `options uvcvideo quirks=0x12`였다.
`uvcvideo`는 **모든 UVC 카메라가 공유**하므로 전역으로 적용되어 RealSense까지 망가뜨린다.

```bash
cat /sys/module/uvcvideo/parameters/quirks     # 4294967295(=미설정)가 정상
grep -r quirks /etc/modprobe.d/
journalctl -k | grep -i "unknown video format"
```

수정: 해당 `.conf`를 `.disabled`로 옮기고 `sudo rmmod uvcvideo && sudo modprobe uvcvideo`.
수정 후 depth+color 파이프라인 5/5 연속 성공, 20초 연속 스트림 stall 0 (이전엔 ~2/3 실패).

전문: `docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`.

### 2.2 DFU / 복구 모드

`lsusb`에 `8086:0adb "RS400 Device"`가 보이면 (정상은 `8086:0b07 "RealSense D435"`)
복구 모드다. **물리적으로 USB를 뽑았다 꽂는다.** quirks 문제와는 무관하다.

```bash
lsusb | grep 8086
```

### 2.3 ⚠️ "2대가 같은 USB 허브에 있으면 문제" — **가설이지 확인된 사실이 아니다**

리포에서 이 주장을 뒷받침하는 근거는 **없다.** 유일하게 관련된 문자열은
`launch_cameras.sh`의 실패 힌트 문구 `"bad serial? camera unplugged? USB bandwidth?"`뿐이고,
`REALSENSE_D435_TROUBLESHOOTING.md`가 지목한 근본 원인은 §2.1의 quirks다.

그래도 확인은 싸니까, 두 대가 30 Hz를 못 채울 때 다음 순서로 본다:

```bash
lsusb -t                       # 두 카메라가 같은 버스/허브 아래인지
lsusb -t | grep -i "5000M"     # USB3(5 Gbps)로 잡혔는지 (480M이면 USB2로 폴백된 것)
```

- **USB2(480M)로 폴백**되어 있으면 그건 실제 원인이 될 수 있다 — 케이블/포트를 바꾼다.
- 별도 버스로 옮겨서 재현이 사라지면 그때 **이 문서에 사실로 승격**하고 근거를 남긴다.
  그 전까지는 가설로 둔다.

부수적으로, 이 머신은 PREEMPT_RT 커널(`6.8.2-rt11`)이라 RT 스케줄링이 USB isochronous
전송 타이밍에 영향을 줄 수 있다 — **기여 요인이지 주원인은 아니다**
(`REALSENSE_D435_TROUBLESHOOTING.md:30`).

---

## 3. QoS — 🔧 **이 리그에서는 해소됨 (2026-07-27)**

`URRosBackend`의 **모든** 구독은 기본(reliable) QoS를 쓴다. 코드의 경고는 그대로 남아 있다:

```python
# ros_backend.py:208
# VERIFY(hw): all subscriptions here use default (reliable) QoS. If a
# publisher is best-effort the subscription silently never fires —
# check every topic actually delivers on first bring-up (ros2 topic hz).
```

> ### 🔧 정정: `VERIFY(hw)` 우려는 해소됐다
> 2026-07-27 실기에서 퍼블리셔 QoS를 직접 확인한 결과, **RealSense 퍼블리셔는
> RELIABLE / TRANSIENT_LOCAL**이고 백엔드의 기본 reliable 구독과 **호환된다.**
> "best-effort 퍼블리셔 때문에 콜백이 한 번도 안 뜬다"는 시나리오는 **이 리그에서는
> 발생하지 않는다.**
>
> ⚠️ 단서 두 가지:
> 1. **코드 주석(`ros_backend.py:208`)은 아직 갱신되지 않았다.** 주석을 보고 "미해결"이라고
>    판단하지 말 것.
> 2. 이건 **이 리그의 이 퍼블리셔 조합**에 대한 판정이다. 카메라 드라이버 버전이 바뀌거나
>    다른 퍼블리셔(예: 별도 image_transport 릴레이)를 끼우면 다시 확인해야 한다.
>    §3.1의 확인 명령은 여전히 매 세션 돌린다 — QoS 때문이 아니라 **토픽이 실제로
>    흐르는지**(§3.2의 조용한 오염) 때문이다.

### 3.0 ⚠️ 남은 부작용: TRANSIENT_LOCAL은 **캐시된 마지막 프레임**을 준다

퍼블리셔가 TRANSIENT_LOCAL이므로, **늦게 붙은 구독자는 접속 즉시 마지막으로 발행된
프레임을 하나 받는다.** 결과:

- 카메라가 이미 멈춘 뒤에 env를 띄워도 **첫 프레임은 온다.** "카메라 살아 있음"으로 보인다.
- 신선도 검사(`IMAGE_STALE_S = 0.5`)가 이걸 잡아 주지만, 그건 **stamp 기준**이므로
  퍼블리셔가 stamp를 갱신하지 않고 죽었을 때만 잡힌다.
- 그러므로 "한 장 받았다"를 카메라 정상의 근거로 쓰지 말고, **항상 `ros2 topic hz`로
  지속 유량을 본다.**

카메라 구독만 depth 1이고(최신 프레임만 의미 있으므로), 나머지는 depth 10이다.

### 3.1 브링업 시 반드시 하는 확인

```bash
source /opt/ros/humble/setup.bash && source $WT/ros2_ur_ws/install/setup.bash

for t in \
  /joint_states \
  /gello/joint_states \
  /robotiq_gripper/position_percent \
  /force_torque_sensor_broadcaster/wrench \
  /tcp_pose_broadcaster/pose \
  /cam1/cam1/color/image_raw/compressed \
  /cam2/cam2/color/image_raw/compressed ; do
  echo "=== $t"
  timeout 5 ros2 topic hz "$t" --window 20 2>&1 | tail -2
done
```

QoS 프로파일 자체를 보려면:

```bash
ros2 topic info /cam1/cam1/color/image_raw/compressed --verbose
ros2 topic info /cam2/cam2/color/image_raw/compressed --verbose
```

2026-07-27 실측: 두 카메라 모두 **`Reliability: RELIABLE`, `Durability: TRANSIENT_LOCAL`**
→ 백엔드 구독과 호환. `Reliability: BEST_EFFORT`인 퍼블리셔가 나타나면 그때는 그 토픽이
**env에 도달하지 않는다** (에러 없이 조용히).

### 3.2 각 토픽이 없을 때의 증상

| 토픽 | env 설정 키 | 없으면 |
|---|---|---|
| `/joint_states` | `joint_states_topic` | `_update_currpos()`에서 `RuntimeError: no /joint_states received yet` (`ur7e_env.py:613`) |
| `/joint_states`가 0.2 s 이상 낡음 | `JOINT_STATE_STALE_S` | `RuntimeError: /joint_states stale` (`:615-617`) |
| `/tcp_pose_broadcaster/pose` | `tcp_pose_topic` | `TCP_POSE_SOURCE="driver"`일 때 `RuntimeError: no /tcp_pose_broadcaster/pose received` (`:624`). ⚠️ **`cube_in_cup`은 `"driver"`를 쓴다** — `"fk"`로 바꾸는 것은 단독 우회가 아니다 (§5.3 아래, `08` G1의 세트 3종) |
| `/force_torque_sensor_broadcaster/wrench` | `wrench_topic` | **조용히 0으로 채워진다** — 관측 키는 유효한 채로 남는다 (`:597-600`). mock에서 정상 |
| `/robotiq_gripper/position_percent` | `gripper_state_topic` | `pct is None` → `curr_gripper_pos = 0.0` = **"열림"으로 보임**. 조용한 오정보 |
| `/camX/.../compressed` (또는 0.5 s 이상 낡음) | `IMAGE_STALE_S` | `RuntimeError: camera 'camX' has no fresh frame` (`:705-717`) |

> ### 🛑 위 표에서 위험한 두 줄
> **wrench와 gripper는 없어도 예외가 안 난다.** 0으로 채워진 F/T와 "열림"으로 보이는
> 그리퍼가 그대로 관측/버퍼에 들어간다. 즉 **브로드캐스터를 빠뜨린 채로 데이터를 몇 시간
> 모을 수 있다.** §3.1의 `topic hz` 루프를 **매 세션 시작 시** 돌리는 이유가 이것이다.

---

## 4. 이미지 파이프라인

`ur7e_env.get_im()` (`ur7e_env.py:463-500`):

```
compressed JPEG → cv2.imdecode(BGR) → IMAGE_CROP[key](선택) → resize(128x128) → [..., ::-1](RGB)
```

- 관측은 **RGB**, 표시는 BGR (FrankaEnv 관례).
- `IMAGE_CROP`는 `DefaultUR7eEnvConfig`에서는 비어 있고, **태스크 config가 채운다.**
  `cube_in_cup`은 **정사각 크롭**을 쓴다 (128×128 리사이즈에서 종횡비 왜곡이 없도록):
  cam1 `img[20:670, 340:990]`, cam2 `img[0:720, 420:1140]`.
  크롭이 없으면 1280×720이 1:1로 눌려 **가로가 세로의 0.5625로 압축**된다.

> ### 🔴 알려진 충돌 (G15): 분류기는 크롭 없이 학습됐다 — **2026-07-28 증명됨**
> 학습은 크롭 없이 1280×720 full-frame을 128×128로 찌그러뜨렸다
> (kanu `hil-serl/examples/cube_classifier_pipeline.py::preprocess_frame`,
> `export_0724.py`가 `crop=None`을 넘긴다). 그런데 actor의 `ur7e_env.get_im()`은
> **`IMAGE_CROP` 적용 후** 리사이즈한다. 즉 크롭을 켜는 순간 **분류기 입력이 분포 밖으로 나간다.**
>
> 증명: 픽셀 대조 MAE **0.00**(무크롭 가설, 100% 비트 일치) vs **21–35**(우리 크롭),
> 실제 체크포인트 실행에서 recall@0.85 **100.0% → 33.3%**, 개별 최저 P 0.0213.
>
> **🔧 이 문서의 이전 판은 원인을 `reward_classifier_runtime.decode_classifier_image()`로
> 지목했는데 그것은 틀렸다.** 그 함수는 ZMQ GUI 뷰어(port 5594) 전용이고 gRPC 경로와
> **호출 관계가 없다.** 서버 `_classifier_observation()`은 이미지 변환을 하나도 하지 않는
> passthrough다. **고칠 위치는 actor의 `IMAGE_CROP`이다.**
>
> **현재 프로덕션은 안 망가져 있다** — 크롭을 켜는 task config가 kanu 브랜치에 없다.
> **actor 브랜치 머지 시 유입된다.** 수정 선택지(크롭으로 재학습 vs 전처리 분리)는
> `serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` §12.7.
>
> **`IMAGE_CROP` 값 자체는 정책 관점에서 올바르다 — 임의로 바꾸지 말 것.**
- `DISPLAY_IMAGE`가 기본 `True` (`config.py:28`) → OpenCV 창이 뜬다. 헤드리스 세션에서는 끈다.
- **RealSense를 두 번 열 수 없기 때문에** env는 pyrealsense2로 장치를 직접 열지 않고
  `launch_cameras.sh`의 토픽을 구독한다 (`config.py:17-21`). 즉 `launch_cameras.sh`가
  떠 있어야만 RL이 돈다. 반대로, 뷰어/레코더와 공존할 수 있다.

---

## 5. 19-D `state` 필드 순서 계약

### 5.1 무엇이 문제였나

env는 `state`를 **중첩 dict**로 낸다 (`ur7e_env.py:404-416`):

```python
{"tcp_pose": (7,), "tcp_vel": (6,), "gripper_pose": (1,), "tcp_force": (3,), "tcp_torque": (3,)}
```

이걸 19-D 평탄 벡터로 만드는 것은 우리 코드가 아니라 **upstream `SERLObsWrapper`**이고
(`Quat2EulerWrapper`가 먼저 pose 7 → 6으로 줄인다), 그 wrapper가 쓰는
`gym.spaces.Dict`는 **평범한 매핑을 알파벳순으로 재정렬한다.**
따라서 `proprio_keys`에 어떤 순서로 적든 평탄 레이아웃은 알파벳순이다.

### 5.2 현재 상태 — 커밋됨, **알파벳순이 정본** (2026-07-27 라이브 확인)

`serl_ur_infra/ur_env/observation_schema.py`가 이 현실을 계약으로 승격시켰다:

| 슬라이스 | 그룹 | 피처 |
|---|---|---|
| `[0:1)` | `gripper_pose` | `gripper_position` |
| `[1:4)` | `tcp_force` | force x,y,z |
| `[4:10)` | `tcp_pose` | position x,y,z + euler x,y,z |
| `[10:13)` | `tcp_torque` | torque x,y,z |
| `[13:19)` | `tcp_vel` | linear x,y,z + angular x,y,z |

> ## 🛑 그리퍼는 인덱스 **0**이다. `-1`이 아니다.
> `state[..., -1]`은 **`tcp_angular_velocity_z`**다 (`tcp_vel[13:19)`의 마지막 원소).
> 반드시 `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`를 쓴다.
> 숫자 인덱스를 call site에 다시 적지 않는다.

2026-07-27 라이브 출력 (§5.2.1의 스니펫 결과):

```
id   : hil-serl-ur-canonical-observation-v2
hash : 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
dim  : 19 grip@ 0
  gripper_pose [0:1)
  tcp_force    [1:4)
  tcp_pose     [4:10)
  tcp_torque   [10:13)
  tcp_vel      [13:19)
```

- 이 해시는 **Kanu 서버와 동일함이 확인됐다** (`09_HIL_ACTOR_RUNBOOK.md` §2.1).
  그래도 매번 양쪽에서 출력해서 대조한다 — 해시를 문서에서 복사하지 말 것.
- 회귀 테스트 `serl_ur_infra/tests/test_state_layout_contract.py`(**22 passed**)가
  **살아 있는 env + 살아 있는 gymnasium**에서 레이아웃을 다시 유도해서 대조한다.
  "알파벳순"을 하드코딩하지 않으므로, gymnasium이 바뀌면 테스트가 새 진실을 알려준다.

### 5.2.1 라이브 확인 스니펫

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH python3 - <<'PY'
from ur_env.observation_schema import (
    OBSERVATION_SCHEMA_ID, CANONICAL_OBSERVATION_SCHEMA_HASH,
    STATE_DIM, GRIPPER_POSITION_INDEX, CANONICAL_STATE_LAYOUT)
print("id   :", OBSERVATION_SCHEMA_ID)
print("hash :", CANONICAL_OBSERVATION_SCHEMA_HASH)
print("dim  :", STATE_DIM, "grip@", GRIPPER_POSITION_INDEX)
for k, a, b in CANONICAL_STATE_LAYOUT:
    print(f"  {k:12s} [{a}:{b})")
PY
```

### 5.3 🔧 F/T wrench 프레임 — **문제가 아니다 (정정)**

이전 판(및 주변 문서)은 "관측 안에 세 프레임이 섞여 있다"를 미해결 문제로 적었다.
**wrench에 관한 한 그것은 틀렸다.**

| | 우리 | upstream (Franka) |
|---|---|---|
| wrench 소스 | `/force_torque_sensor_broadcaster/wrench` = **`tool0` 프레임** | libfranka `K_F_ext_hat_K` = **stiffness(=EE) 프레임** |
| `RelativeFrame`이 손대는가 | **아니다** | **아니다** |
| 결과 | tool 프레임 wrench + tool 프레임 `tcp_vel` | 동일 |

즉 **양쪽 스택이 같은 모양**이고, wrench를 base 프레임으로 돌리는 것은 upstream과의
정합을 깨는 행위다. 코드가 이걸 명시한다 (`frame_wrappers.py:98-102`):

> *"Note `tcp_force` / `tcp_torque` are deliberately left alone — upstream does not touch
> them either. Our `/force_torque_sensor_broadcaster/wrench` publishes in `tool0`, and
> libfranka's `K_F_ext_hat_K` is likewise in the stiffness (end-effector) frame, so both
> stacks end up with wrench in the tool frame alongside a tool-frame `tcp_vel`."*

**하지 말 것:** "프레임을 통일한다"는 명목으로 `wrappers`/`frame_wrappers`에서 wrench를
회전시키는 것. 그러면 이미 기록된 데모/replay와 학습 체크포인트가 전부 어긋난다.

### 5.4 아직 남은 위험

- canonical v2 layout과 실제 upstream wrapper 경로는 통합 suite 및 Kanu fake-data E2E에서
  검증됐다. 실행 전 laptop/Kanu가 같은 schema hash를 광고하는지는 계속 확인한다.
- **F/T 브로드캐스터가 없으면 force/torque 6개가 전부 0으로 들어간다** (§3.2).
  레이아웃은 맞지만 값이 죽어 있는 것 — 스키마 해시로는 절대 잡히지 않는다.
  (이건 **프레임 문제가 아니라 유량 문제**다. §5.3과 혼동하지 말 것.)
- `tcp_pose` 프레임은 별개 사안이다 → `08_OPEN_GAPS.md` G7 / G1의 flange 커플링.

---

## 6. 판정 체크리스트

- [ ] `./launch_cameras.sh`가 30초 안에 두 스트림 ~30 Hz 도달
- [ ] 뷰어 좌 = cam1 SCENE(배경 고정) / 우 = **cam2 WRIST(팔을 움직이면 배경이 흐르고
      손가락은 제자리)** — §1.3의 "팔 한 번 움직여 보기"로 판정
- [ ] `lsusb -t`에서 두 카메라 모두 5000M(USB3)
- [ ] `cat /sys/module/uvcvideo/parameters/quirks` = `4294967295`
- [ ] §3.1의 `topic hz` 루프에서 **7개 토픽 전부** 값이 나온다
      (TRANSIENT_LOCAL 때문에 `echo --once`는 죽은 카메라에서도 성공한다 → §3.0.
      **반드시 `hz`로 판정**)
- [ ] `ros2 topic info --verbose`로 best-effort 퍼블리셔가 없음을 확인
      (2026-07-27 기준 RealSense는 RELIABLE/TRANSIENT_LOCAL)
- [ ] wrench가 **0이 아닌** 실제 값을 낸다 (팔을 살짝 밀어보면 변한다).
      값은 **tool0 프레임**이 정상이다 — base로 안 보인다고 고치지 말 것 (§5.3)
- [ ] 그리퍼 `position_percent`가 실제 상태를 반영한다 (0.0이 "그냥 없음"이 아님을 확인)
- [ ] 레이아웃 회귀 테스트 통과 (`00_SETUP_AND_SAFETY.md` §4.2의 명령 형식으로):
      `pytest tests/test_state_layout_contract.py -q -p no:anyio`
- [ ] 랩톱/Kanu 스키마 해시 일치 (§5.2.1, `05_COMMS_GRPC.md` §4.2)
