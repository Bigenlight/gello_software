# gello_recorder

GELLO 리더팔로 UR7e를 텔레옵하는 동안 **로봇·GELLO·그리퍼 신호와 RealSense 2대의
영상**을 take 단위 폴더로 기록하는 ROS 2 (Jazzy) 패키지입니다. 기록 경로는 구독만
하는 **읽기 전용**이고, GUI에는 텔레옵 일시정지/재개 버튼과 "Delete last take" /
"Tag last take" 버튼이 함께 있습니다.

## 진입점 3개 — 언제 무엇을 쓰나

| 진입점 | 실행 | 언제 쓰나 |
| --- | --- | --- |
| 헤드리스 기록기 `gello_ur_recorder` | `./run_recorder.sh` | 화면 없이 한 세션을 통째로 기록(진단·로그). Ctrl-C로 종료 = 세션 1개 |
| 조인트 텔레옵 GUI `gello_recorder_gui` | `ros2 run gello_recorder gello_recorder_gui` | 조인트 모드 텔레옵으로 데모를 **take 여러 개**로 수집. 카메라 미리보기 + Start/Stop |
| EEF 수집 GUI `eef_recorder_gui` (ur_gello_bringup) | `./run_eef_collection_gui.sh` | EEF 모드 텔레옵 수집. 위 GUI의 기록 창과 EEF ENGAGE/START POSE 창을 한 창에 합친 것. 브링업을 재시작해도 카메라·기록 창이 살아 있음 |

세 진입점 모두 같은 `RecordingSession`으로 쓰므로 **파일 형식이 같습니다**(차이는
아래 표의 *헤드리스 전용* 표시뿐).

## 빠른 시작

```bash
cd ros2_ur_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select gello_recorder ur_gello_bringup
source install/setup.bash

# 터미널 1: 텔레옵
./run_ur7e_gello_real.sh                                   # 조인트 모드
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef   # EEF 모드

# 터미널 2: 기록기 하나를 고른다
ros2 run gello_recorder gello_recorder_gui   # 조인트 모드 수집
./run_eef_collection_gui.sh                  # EEF 모드 수집 (단축키 R/SPACE/G/H)
CAMS=true ./run_recorder.sh                  # 헤드리스, 카메라 포함
```

GUI는 카메라 노드 2개를 직접 띄우고, 두 카메라가 워밍업(`CAMERA_WARMUP_S`, 기본
4 s)을 마쳐야 **Start Recording**이 켜집니다. 헤드리스는 `CAMS=true`일 때만 카메라를
띄웁니다.

## 출력 형식

```
ros2_ur_ws/gello_logs/                        # GUI는 RECORDER_OUTPUT_ROOT 로 변경 가능
├── take_<NN>_<YYYYmmdd_HHMMSS>[_<tag>...]/   # GUI take (NN은 GUI를 켤 때마다 01부터)
└── session_<YYYYmmdd_HHMMSS>/                # 헤드리스 세션
    ├── vectors.h5      # 벡터 신호 테이블 9개
    ├── cam1.mp4        # 컬러 영상 (cam1 = 장면 카메라)
    ├── cam2.mp4        # 컬러 영상 (cam2 = 손목 카메라)
    ├── depth.h5        # ENABLE_DEPTH=1 일 때만
    └── metadata.json   # 헤드리스 전용
```

### `vectors.h5`

테이블마다 HDF5 **그룹 하나**, 그 안에 **컬럼마다 float64 1-D 데이터셋 하나**가
있습니다. 컬럼 순서는 그룹 속성 `columns`(JSON 리스트)에 있습니다. 발행되지 않은
토픽은 에러 없이 빈 테이블/NaN으로 남습니다.

| 그룹 | 컬럼 | 출처 |
| --- | --- | --- |
| `gello_joint_states` | `t_rel_s q1..q6 qd1..qd6 stamp_s` | `/gello/joint_states` (rad, rad/s) |
| `ur_joint_states` | `t_rel_s q1..q6 qd1..qd6 eff1..eff6 stamp_s` | `/joint_states`, UR 관절 순서로 재정렬 |
| `command` | `t_rel_s cmd1..cmd6` | `/forward_position_controller/commands` (rad) |
| `gripper` | `t_rel_s gello_grip grip_cmd grip_pos` | GELLO 방아쇠 · `/robotiq_gripper/command_percent` · `/robotiq_gripper/position_percent` |
| `wrench` | `t_rel_s fx fy fz tx ty tz stamp_s` | `/force_torque_sensor_broadcaster/wrench` (N, N·m) |
| `tcp_pose` | `t_rel_s x y z qx qy qz qw stamp_s` | `/tcp_pose_broadcaster/pose` (m, 쿼터니언) |
| `cam1_frames`, `cam2_frames` | `t_rel_s frame_idx stamp_s` | MP4 프레임 번호 ↔ 시각 (행 수 = MP4 프레임 수) |
| `synchronized` | `t_rel_s t_wall gello_q1..6 gello_qd1..6 gello_grip cmd1..6 ur_q1..6 ur_qd1..6 ur_eff1..6 grip_cmd grip_pos fx..tz tcp_x..tcp_qw cam1_frame_idx cam2_frame_idx` | *헤드리스 전용*: `RATE`(기본 100 Hz)마다 모든 최신값을 찍은 와이드 테이블. GUI take에서는 비어 있음 |

**시각 컬럼의 의미**

- `t_rel_s` — 이 take/세션을 연 순간부터의 초. 메시지가 **도착해 콜백이 실행된**
  시각입니다(카메라 프레임은 도착 시각을 쓰기 큐에 함께 실어 보내므로 쓰기 지연이
  섞이지 않음).
- `stamp_s` — 메시지 `header.stamp`(epoch 초). 드라이버가 값을 **캡처한** 시각이며
  없으면 NaN. 테이블끼리 맞출 때는 `stamp_s`가 더 정확합니다. `command`/`gripper`는
  메시지 타입에 헤더가 없어 이 컬럼이 없습니다.
- `t_wall` (`synchronized`만) — 그 행을 샘플링한 `time.time()`.

**그리퍼 규약:** `gello_grip` / `grip_cmd` / `grip_pos` 모두 **0.0 = 열림, 1.0 = 닫힘**.
`grip_pos`는 측정값이라 물체를 쥐면 1.0까지 가지 않습니다(잡음/헛잡음을 구분하는 정보).

### `cam1.mp4`, `cam2.mp4`

RealSense `.../color/image_raw/compressed`(JPEG)를 디코드해 `COLOR_PROFILE`
(기본 1280x720, 30 fps)로 다시 인코딩합니다. `camN_frames.frame_idx`가 MP4 프레임
번호입니다. 워밍업 전 프레임은 기록되지 않습니다.

### `depth.h5` (선택, `ENABLE_DEPTH=1`)

```
/<cam>/   (cam1, cam2)   attrs: encoding, unit, depth_scale_m, width, height, aligned_to_color, ...
    png        vlen uint8 (N,)   # 드라이버가 만든 16UC1 PNG(mm)를 그대로 저장
    t_rel_s    float64    (N,)   # vectors.h5 와 같은 시각 원점
    frame_idx  int64      (N,)
    stamp_s    float64    (N,)
/<cam>/camera_info/                attrs: width, height, distortion_model, frame_id, D, K, R, P
/<cam>/extrinsics_depth_to_color/  attrs: rotation(9, column-major), translation(3, m)
```

디코드는 `gello_recorder.depth_writer.read_depth_frame` / `iter_depth_frames`
(uint16 mm 배열)를 씁니다. 두 대 합쳐 디스크 약 6 MB/s에 기록기 CPU도 늘기 때문에
**기본은 꺼짐**입니다. `ALIGN_DEPTH=1`은 컬러에 정렬된 depth를 기록하지만 카메라 노드
CPU가 크게 늘고 두 스트림이 ~25 Hz로 떨어지므로, 보통은 기록된 내·외부 파라미터로
오프라인 정렬합니다.

### `metadata.json` (헤드리스 전용)

`start_wall`(ISO 시각, `t_rel_s`의 원점), `sample_rate_hz`, `record_depth`,
`duration_s`, `message_counts`(토픽별 수신 수), `finalized`, 그리고 기록 건강도
`ros_lag_s_max` · `spin_starvation_suspected` · `dropped_frames`. GUI는 같은 요약을
take 종료 시 상태줄과 로그로 보여 줍니다.

### take 읽기 예

```python
import json
import h5py

with h5py.File("take_01_20260101_120000/vectors.h5", "r") as f:
    g = f["ur_joint_states"]
    cols = json.loads(g.attrs["columns"])     # ['t_rel_s', 'q1', ..., 'stamp_s']
    ur = {c: g[c][:] for c in cols}           # 컬럼 이름 -> numpy 배열
    cam1_stamp = f["cam1_frames"]["stamp_s"][:]   # cam1.mp4 프레임 i 의 캡처 시각
print(len(ur["q1"]), len(cam1_stamp))
```

## 환경변수

| 이름 | 기본값 | 대상 | 의미 |
| --- | --- | --- | --- |
| `CAM1_SERIAL`, `CAM2_SERIAL` | `ros2_ur_ws/camera_serials.env` | 전부 | RealSense **device** serial (ASIC serial 아님) |
| `GELLO_CAMERA_SERIALS_FILE` | — | 전부 | 위 기본값 파일 경로를 바꿈 |
| `CAM1_NAME`, `CAM2_NAME` | `cam1`, `cam2` | 전부 | 카메라 namespace |
| `COLOR_PROFILE` | `1280x720x30` | 전부 | 컬러 해상도·fps |
| `ENABLE_DEPTH`, `ALIGN_DEPTH` | `0` | 전부 | depth 기록 / 노드 안 정렬 |
| `CAMERA_WARMUP_S` | `4.0` | GUI | Start 허용 전 카메라 워밍업 |
| `RECORDER_OUTPUT_ROOT` | `ros2_ur_ws/gello_logs` | GUI | take 저장 위치 |
| `RECORDER_TAG_PRESETS` | `failed,review` | GUI | "Tag last take" 콤보 프리셋(쉼표 구분). 직접 입력도 됨 |
| `GELLO_HOME_JOINTS` | 코드 기본값 | `home_move` | GO HOME 목표 관절각 6개(rad, 쉼표 구분) |
| `CAMS`, `BAG`, `RATE` | off, off, `100` | 헤드리스 | 카메라 기록 / `ros2 bag record -a` 동시 기록 / `synchronized` 주기 |
| `CAMERA_ALLOW_SERIAL_FALLBACK` | `0` | 헤드리스 | 설정한 serial 쌍이 없을 때 다른 쌍 허용(임시 리그용) |

헤드리스 노드의 ROS 파라미터(`session_dir`, `sample_rate_hz`, `record_depth`,
`cam*_depth_topic` 등)는 `gello_ur_recorder_node.py` 모듈 docstring에 있습니다.

**다른 리그에서:** `rs-enumerate-devices -s`의 "Serial Number"를
`ros2_ur_ws/camera_serials.env`에 적거나 환경변수로 넘깁니다(cam1 = 장면, cam2 = 손목).

## 문제 해결

| 증상 | 원인 / 조치 |
| --- | --- |
| 카메라 창이 "no frames", Start가 계속 Warming up | serial이 연결된 장치와 다름. `rs-enumerate-devices -s`로 확인 후 `CAM1_SERIAL`/`CAM2_SERIAL` 지정. journalctl에 보이는 ASIC serial은 매칭되지 않음 |
| 헤드리스가 `expected RealSense pair is not fully connected`로 카메라 실행 전에 종료 | 의도된 동작. 두 카메라를 USB 3에 다시 꽂거나 serial을 지정 |
| 헤드리스가 `image_transport compressed publisher plugin is unavailable` | `ros-jazzy-compressed-image-transport` 설치 |
| 상태 패널의 `ros lag`가 빨간색, take 요약에 `spin_starvation_suspected` | 기록기 spin 스레드가 밀려 로봇 행이 늦게 기록됨. depth를 끄거나 부하를 줄이고 다시 기록 |
| take 요약에 `dropped_frames` | 디스크/CPU가 쓰기를 못 따라감. 그 take는 그만큼 프레임이 빠져 있음 |
| 어떤 테이블이 비어 있음 | 해당 토픽이 발행되지 않음(sim, 그리퍼 없는 브링업 등). 헤드리스는 `message_counts`로 확인 |

## 테스트

```bash
cd ros2_ur_ws && source /opt/ros/jazzy/setup.bash && source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q src/gello_recorder/test
```

ROS 그래프·카메라·로봇 없이 돕니다(GUI 로직은 duck-typed 대역으로 검증).
