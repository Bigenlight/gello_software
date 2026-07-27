# 06 — 센서 (RealSense · QoS · 19-D state 계약)

**상태: 이 브랜치에서 미검증.**

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## 1. RealSense 2대

### 1.1 계약

| | cam1 | cam2 |
|---|---|---|
| 역할 | **SCENE** (삼각대, 3인칭) | **CLOSE-UP** (작업 근접) |
| 시리얼 | `147122072740` (plain D435) | `243222072700` (D435IF) |
| 토픽 | `/cam1/cam1/color/image_raw/compressed` | `/cam2/cam2/color/image_raw/compressed` |
| 프로파일 | `1280x720x30` (둘 다 동일) | 동일 |

근거: `ros2_ur_ws/launch_cameras.sh:38-43`, `:57-59`; env 쪽은
`serl_ur_infra/ur_env/envs/config.py:22-24`.

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

뷰어가 **기본 ON**인 이유가 있다. 좌측 창이 **전체 장면**, 우측 창이 **근접**이어야 한다.
바뀌어 있으면 Ctrl-C하고 시리얼부터 다시 확인한다 —
매핑이 뒤집히면 학습된 정책이 **조용히** 열화된다 (`launch_cameras.sh:12-16`).

### 1.4 `serial_no`는 반드시 따옴표로 감싼다

```bash
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam1 camera_namespace:=cam1 "serial_no:='147122072740'" \
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

## 3. 🛑 QoS 함정

`URRosBackend`의 **모든** 구독은 기본(reliable) QoS를 쓴다:

```python
# ros_backend.py:100-103
# VERIFY(hw): all subscriptions here use default (reliable) QoS. If a
# publisher is best-effort the subscription silently never fires —
# check every topic actually delivers on first bring-up (ros2 topic hz).
```

> **best-effort 퍼블리셔 + reliable 구독자 = 콜백이 한 번도 안 불린다. 에러도 경고도 없다.**
> 이건 "느려진다"가 아니라 "완전히 조용히 안 온다"다.

카메라 구독만 depth 1이고(최신 프레임만 의미 있으므로), 나머지는 depth 10이다
(`ros_backend.py:104-110`).

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
```

`Reliability: BEST_EFFORT`인 퍼블리셔가 하나라도 있으면 그 토픽은 **env에 도달하지 않는다.**

### 3.2 각 토픽이 없을 때의 증상

| 토픽 | env 설정 키 | 없으면 |
|---|---|---|
| `/joint_states` | `joint_states_topic` | `_update_currpos()`에서 `RuntimeError: no /joint_states received yet` (`ur7e_env.py:362-363`) |
| `/joint_states`가 0.2 s 이상 낡음 | `JOINT_STATE_STALE_S` | `RuntimeError: /joint_states stale` (`:365-367`, `config.py:121`) |
| `/tcp_pose_broadcaster/pose` | `tcp_pose_topic` | `TCP_POSE_SOURCE="driver"`일 때 `RuntimeError: no /tcp_pose_broadcaster/pose received` (`:371-379`). `TCP_POSE_SOURCE="fk"`로 우회 가능 |
| `/force_torque_sensor_broadcaster/wrench` | `wrench_topic` | **조용히 0으로 채워진다** — 관측 키는 유효한 채로 남는다 (`:398-402`). mock에서 정상 |
| `/robotiq_gripper/position_percent` | `gripper_state_topic` | `pct is None` → `curr_gripper_pos = 0.0` = **"열림"으로 보임**. 조용한 오정보 (`:394-395`) |
| `/camX/.../compressed` (또는 0.5 s 이상 낡음) | `IMAGE_STALE_S` | `RuntimeError: camera 'camX' has no fresh frame` (`:475-481`) |

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
- `IMAGE_CROP`는 기본 **빈 dict** (`config.py:25`) — 크롭 없음. 태스크별 config에서 채운다.
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

### 5.2 현재 상태 (2026-07-27, 워킹트리 / 커밋 안 됨)

`serl_ur_infra/ur_env/observation_schema.py`가 이 현실을 계약으로 승격시켰다:

| 슬라이스 | 그룹 | 피처 |
|---|---|---|
| `[0:1)` | `gripper_pose` | `gripper_position` |
| `[1:4)` | `tcp_force` | force x,y,z |
| `[4:10)` | `tcp_pose` | position x,y,z + euler x,y,z |
| `[10:13)` | `tcp_torque` | torque x,y,z |
| `[13:19)` | `tcp_vel` | linear x,y,z + angular x,y,z |

> ## 🛑 그리퍼는 인덱스 **0**이다. `-1`이 아니다.
> `state[..., -1]`은 **TCP 각속도 z**다. 반드시
> `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`를 쓴다.

- `OBSERVATION_SCHEMA_ID`가 `...-v1` → `...-v2`로 올라갔고 해시가 바뀌었다.
  **랩톱과 Kanu 양쪽을 같이 올려야 통신이 된다** → `05_COMMS_GRPC.md` §4.
- 회귀 테스트 `serl_ur_infra/tests/test_state_layout_contract.py`(신규, untracked)가
  **살아 있는 env + 살아 있는 gymnasium**에서 레이아웃을 다시 유도해서 대조한다.
  "알파벳순"을 하드코딩하지 않으므로, gymnasium이 바뀌면 테스트가 새 진실을 알려준다.

### 5.3 아직 남은 위험

- `serl_ur_infra/RL_RECEIVE_SERVER.md`는 **아직 옛 순서**를 적고 있다 (문서 낡음).
- 이 워크트리에 `third_party/hil-serl`이 없어서 `SERLObsWrapper`를 **실제로 통과시켜 본
  적이 없다.** 레이아웃 테스트는 upstream을 찾으면 쓰고 없으면 건너뛴다 —
  서브모듈이 있는 환경(메인 워크트리 / Kanu)에서 반드시 한 번 돌려야 한다.
- **F/T 브로드캐스터가 없으면 force/torque 6개가 전부 0으로 들어간다** (§3.2).
  레이아웃은 맞지만 값이 죽어 있는 것 — 스키마 해시로는 절대 잡히지 않는다.

---

## 6. 판정 체크리스트

- [ ] `./launch_cameras.sh`가 30초 안에 두 스트림 ~30 Hz 도달
- [ ] 뷰어 좌=SCENE / 우=CLOSE-UP **육안 확인**
- [ ] `lsusb -t`에서 두 카메라 모두 5000M(USB3)
- [ ] `cat /sys/module/uvcvideo/parameters/quirks` = `4294967295`
- [ ] §3.1의 `topic hz` 루프에서 **7개 토픽 전부** 값이 나온다
- [ ] `ros2 topic info --verbose`로 best-effort 퍼블리셔가 없음을 확인
- [ ] wrench가 **0이 아닌** 실제 값을 낸다 (팔을 살짝 밀어보면 변한다)
- [ ] 그리퍼 `position_percent`가 실제 상태를 반영한다 (0.0이 "그냥 없음"이 아님을 확인)
- [ ] `python3 -m pytest tests/test_state_layout_contract.py -q -p no:anyio` 통과
- [ ] 랩톱/Kanu 스키마 해시 일치 (`05_COMMS_GRPC.md` §4.2)
