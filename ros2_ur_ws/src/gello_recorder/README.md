# gello_recorder

GELLO 리더암 + UR7e 로봇암 teleop 파이프라인의 진단 신호(HDF5)와 듀얼 RealSense 카메라 영상(MP4)을 기록하는 ROS2 (Humble, ament_python) 패키지. **실제 물리 로봇 하드웨어**(real UR7e + real GELLO + 2x real Intel RealSense)를 대상으로 하며, 완전히 read-only(구독만, 로봇/GELLO에 명령을 내리지 않음)이다. `ur_gello_bringup`(UR 전용 브링업 패키지)과 의도적으로 분리되어 있어 UR-specific 의존성이 전혀 없다.

두 가지 실행 방법이 있다:
- **헤드리스 CLI** (`gello_ur_recorder`, `run_recorder.sh`): launch부터 Ctrl-C까지 연속 기록.
- **인터랙티브 GUI** (`gello_recorder_gui`): 카메라를 한 번만 켜두고 Start/Stop 버튼으로 여러 번 "take"를 찍는 방식.

목차: [Requirements & Quick Start](#requirements--quick-start) · [Architecture & Code Walkthrough](#architecture--code-walkthrough) · [Troubleshooting / Debugging History](#troubleshooting--debugging-history)

---

## Requirements & Quick Start

> ⚠️ **Important — 실제 로봇 하드웨어입니다 (시뮬레이션 아님)**
>
> - 이 패키지는 **실제 UR7e + 실제 GELLO 리더암 + 실제 Intel RealSense 카메라 2대**를 대상으로 합니다. 시뮬레이션 도구가 아닙니다.
> - 기록기는 **완전히 read-only** 입니다. ROS 토픽을 *구독(subscribe)만* 하며, 로봇이나 GELLO에 명령을 내리지 않습니다. 켜고 끄는 것이 로봇 동작에 영향을 주지 않습니다.
> - 카메라 2대는 **둘 다 USB로 물리적으로 연결**되어 있어야 하고, 다른 `realsense2_camera_node` 프로세스가 **이미 점유하고 있으면 안 됩니다.** 이전 실행이 비정상 종료되면 좀비 프로세스가 USB 장치를 잠근 채 남아 카메라가 안 켜질 수 있습니다. 그럴 때는 우선 `pkill -9 -f realsense2_camera_node` 로 정리하세요. (자세한 디버깅은 [Troubleshooting](#troubleshooting--debugging-history) 섹션 참고.)
> - `ros2 launch` 의 CLI 인자 중 **숫자처럼 보이는 값**(예: 카메라 시리얼 번호)은 작은따옴표로 감싸지 않으면 조용히 정수로 형변환되어 노드가 즉시 죽습니다. 이 패키지는 내부적으로 `serial_no:='151623020789'` 처럼 값에 따옴표를 박아 처리하고 있으니, **launch 명령을 직접 수정할 때는 이 따옴표를 반드시 유지**하세요.

### 1. Prerequisites

- **ROS2 Humble** on **Ubuntu 22.04**.
- 빌드된 워크스페이스 (`ros2_ur_ws/` 에서 `colcon build --symlink-install`, 아래 2번 참고).
- Python 런타임 의존성 (package.xml 의 `exec_depend`):
  - `python3-h5py` (HDF5 벡터 로그)
  - `python3-opencv` (MP4 인코딩 / 프레임 디코딩)
  - `python3-pyqt5` (GUI)
- **카메라를 쓰려면** `realsense2_camera` ROS2 패키지 + librealsense2 SDK 가 설치되어 있어야 합니다. (헤드리스 기록기를 `CAMS` 없이 신호만 기록할 때는 카메라 스택이 필요 없습니다.)
- 실제 하드웨어(UR7e / GELLO / RealSense 2대)가 연결·전원 인가된 상태.

### 2. Build

```bash
source /opt/ros/humble/setup.bash
cd /home/laptop3/gello_software/ros2_ur_ws
colcon build --symlink-install
source install/setup.bash
```

> `--symlink-install` 을 쓰면 Python 소스를 고칠 때마다 다시 빌드하지 않아도 됩니다. 단, `package.xml` / `setup.py` / entry point 를 바꾸면 다시 `colcon build` 해야 합니다.

### 3. Quick Start — 헤드리스 기록기 (`run_recorder.sh`)

`run_recorder.sh` 는 실행 시점부터 Ctrl-C 까지 **연속으로** 기록하는 CLI 방식입니다. **teleop 이 이미 돌고 있는 상태에서, 두 번째 터미널**에서 실행해야 합니다.

```bash
# 첫 번째 터미널: teleop 구동 (기록기와 같은 ros2_ur_ws/ 에 있는 스크립트)
cd /home/laptop3/gello_software/ros2_ur_ws
HEADLESS=true ./run_ur7e_gello_real.sh
```

```bash
# 두 번째 터미널: 기록기
cd /home/laptop3/gello_software/ros2_ur_ws

./run_recorder.sh                 # 신호만 기록 (vectors.h5)
BAG=true ./run_recorder.sh        # 위 + 전체 토픽을 `ros2 bag -a` 로 캡처
CAMS=true ./run_recorder.sh       # 위 + RealSense 카메라 2대 실행 (-> cam1.mp4 / cam2.mp4)
RATE=200 ./run_recorder.sh        # synchronized 테이블 샘플링 레이트(Hz), 기본 100
```

환경변수는 조합할 수 있습니다 (예: `CAMS=true RATE=200 ./run_recorder.sh`). Ctrl-C 로 멈추면 버퍼를 flush 하고 `metadata.json` 을 마무리하며, bag 과 카메라도 함께 정리합니다. 세션은 `ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/` 아래에 저장됩니다.

`CAMS=true` 일 때만 쓰이는 카메라 환경변수(기본값은 아래 GUI 섹션과 동일): `CAM1_SERIAL`, `CAM2_SERIAL`, `CAM1_NAME`, `CAM2_NAME`, `COLOR_PROFILE`.

### 4. Quick Start — 인터랙티브 GUI (`gello_recorder_gui`)

GUI 는 카메라 스택을 매번 재실행하지 않고 Start/Stop 버튼으로 여러 "take" 를 찍는 방식입니다.

```bash
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash

ros2 run gello_recorder gello_recorder_gui
```

실행하면 **카메라 2대가 자동으로 launch** 되고, 창에는 **항상 라이브 프리뷰**가 표시됩니다. **Start 버튼은 `cameras_ready()` 가 될 때까지 비활성화**되어 있습니다 — 즉 **각 카메라의 첫 프레임 수신 시점부터 `CAMERA_WARMUP_S` 초가 지날 때까지** 눌리지 않습니다. 이는 RealSense 컬러 센서가 켜진 직후 자동 노출/화이트밸런스를 잡는 동안 나오는 초록빛 warm-up 틴트가 녹화에 들어가지 않게 하기 위함입니다. 그동안 버튼에는 "Warming up..." 이 표시됩니다.

다음 환경변수로 동작을 재정의할 수 있습니다 (값은 소스에서 검증한 실제 기본값):

> ### 🔧 카메라 시리얼 정정 (2026-07-28)
>
> 카메라 **개체가 물리적으로 교체됐다.** 이전 판의 `147122072740` / `243222072700`은
> 이 PC가 커널 로그상 한 번도 열거한 적 없는 하드웨어다(2026-07-05까지 소급 확인).
> 이 문서의 시리얼은 실제 연결된 개체(`151623020789` / `322743060038`)로 갱신했다.
> **없는 시리얼로 바인딩하면 조용히 안 뜬다** — 오류가 아니라 "프레임 없음"으로 보인다.
> 모델 클래스(D435 / D435if)와 cam1·cam2 배정은 그대로지만, **어느 개체가 손목에 달렸는지는
> 미확정**이다. 팔을 흔들어 cam2 화면에서 손가락이 고정되는지 확인할 것.
> 아래 "검증 완료" 류의 과거 기록은 **옛 개체로 수행된 것**이라 그대로 두었다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `CAM1_SERIAL` | `151623020789` | 카메라 #1 시리얼 (plain D435) |
| `CAM2_SERIAL` | `322743060038` | 카메라 #2 시리얼 (D435IF) |
| `CAM1_NAME` | `cam1` | 카메라 #1 이름/네임스페이스 |
| `CAM2_NAME` | `cam2` | 카메라 #2 이름/네임스페이스 |
| `COLOR_PROFILE` | `1280x720x30` | 컬러 스트림 `WxHxFPS` (두 카메라 공통) |
| `CAMERA_WARMUP_S` | `4.0` | 각 카메라 첫 프레임 후 이 초만큼 지나야 Start 활성화 (warm-up 프레임 폐기) |
| `RECORDER_OUTPUT_ROOT` | `$GELLO_REPO_ROOT/ros2_ur_ws/gello_logs` | take 폴더가 생성되는 루트 (`GELLO_REPO_ROOT` 미설정 시 `~/gello_software`) |

예: `CAM1_SERIAL=151623020789 CAMERA_WARMUP_S=6 ros2 run gello_recorder gello_recorder_gui`

> 참고: 헤드리스 노드의 ROS 파라미터 `camera_warmup_s` 기본값은 `3.0`으로, GUI 의 `4.0`과 **다릅니다** (둘은 독립적인 설정값).

창을 닫으면 진행 중인 take 를 먼저 마무리 저장한 뒤 카메라 프로세스를 정리합니다. take 는 `<RECORDER_OUTPUT_ROOT>/take_<NN>_<YYYYmmdd_HHMMSS>/` 아래에 저장됩니다 (헤드리스의 `session_<stamp>/` 와 이름 규칙이 다름에 유의).

#### 4-1. Teleop 바 — 텔레옵 일시정지 / 재개 (씬 리셋)

GUI 창의 녹화 컨트롤 **바로 위**에는 **Teleop 바**가 있습니다. 혼자 작업하는 오퍼레이터가 **두 손을 자유롭게 하여 씬을 리셋**(물체 재배치, 지그 이동, 재-그립)할 수 있도록, 리더→로봇 신호 경로를 일시정지/재개하는 조작 패널입니다. 이 블록은 그 외에는 100% read-only 인 이 패키지에서 **유일한 제어(control-path) 코드**이며, 로봇을 직접 명령하지 않고 두 브릿지의 `std_srvs/Trigger` 서비스만 호출합니다(호출은 전부 non-blocking `call_async` + done-callback — Qt 스레드에서 절대 spin/블록하지 않음).

구성 요소:

- **상태 라벨 2개** — `arm:` 와 `gripper:`. 각 브릿지의 `/state` 토픽(5 Hz)을 반영해 색으로 표시합니다: `PAUSED`/`STALE` = 빨강, `CHASING`/`RAMPING` = 주황, `FOLLOWING` = 초록, `WAITING`/미상 = 회색. state 토픽이 2초 넘게 갱신 안 되면 미상(회색) 처리.
  - 팔 상태 어휘: `PAUSED` · `WAITING` · `STALE` · `CHASING` · `FOLLOWING`.
  - 그리퍼 상태 어휘: `PAUSED` · `WAITING` · `RAMPING` · `FOLLOWING`.
  - **`gripper: n/a`** 로 보이면 그리퍼 브릿지/state가 아직 없다는 뜻입니다. 정상일 수 있음 — 그리퍼 브릿지는 **핸드셰이크 완료 후에 기동**하므로, 팔 텔레옵이 스트리밍을 시작한 뒤 채워집니다.
- **Pause Teleop 버튼** — **한 번 클릭**. pause는 무조건(unconditional) 서비스라 항상 성공하며, 팔과 그리퍼 브릿지를 함께 정지시킵니다. 로봇은 마지막 자세를 홀드.
- **Resume Teleop 버튼** — 실 로봇을 움직이므로 **두 번 클릭 확인**입니다. 첫 클릭에서 버튼이 주황색 *"Confirm Resume (robot will move!)"* 로 바뀌고, **3초 안에** 다시 누르면 실제 재개 요청이 나갑니다(시간 초과 시 원상복귀). 이 버튼은 팔이 `PAUSED` 를 보고할 때만 활성화됩니다. 재개는 팔의 `/gello_ur_bridge/resume_chase` 와 그리퍼의 `/gello_gripper_bridge/resume` 를 함께 호출합니다.

**거부(refusal)는 에러가 아닙니다.** 재개는 게이팅되어 있어 `success=false` 로 정당하게 거부될 수 있으며(예: 리더가 안 멈춤, gap이 너무 큼), 그때 로봇은 **움직이지 않고 브릿지는 PAUSED로 유지**됩니다. 거부 사유는 상태바에 ~6초간 표시되니, 읽고 그 한 가지(리더를 가만히 / 더 가깝게 등)를 고친 뒤 다시 재개하세요.

> 안전 게이트(fresh 리더, quasi-still, 관절별 gap ≤ 1.5 rad), zero-jump 재시딩, 소프트스타트 글라이드, 그리고 각 거부 메시지의 정확한 의미는 `ur_gello_bringup` 의 **README "Pause / Resume (scene reset with both hands free)"** 절과 `docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md` §6 에 정리되어 있습니다. Teleop 바는 그 브릿지 서비스들의 GUI 프런트엔드일 뿐입니다.

---

## 5. Output Layout — 세션/테이크 폴더 안에 생기는 것

```
session_<YYYYmmdd_HHMMSS>/       (GUI는 take_<NN>_<YYYYmmdd_HHMMSS>/)
├── vectors.h5        # 모든 벡터 신호 테이블 (HDF5, 아래 9개 테이블)
├── cam1.mp4          # RealSense #1 컬러 영상 (CAMS/카메라 사용 시)
├── cam2.mp4          # RealSense #2 컬러 영상
└── metadata.json     # 파라미터, 시작 시각, 종료 시 토픽별 메시지 카운트
```

`vectors.h5` 안의 9개 테이블:

| 테이블 | 내용 |
|---|---|
| `synchronized` | `sample_rate_hz`(기본 100Hz)로 시간 정렬된 **와이드 메인 분석 테이블** — 매 순간 모든 신호의 최신값 |
| `gello_joint_states` | GELLO 관절 위치 + 유한차분 속도 (native rate) |
| `ur_joint_states` | UR7e 실제 관절 position / velocity / effort |
| `command` | 브리지가 UR 로 내보낸 명령 |
| `gripper` | GELLO 그리퍼 width + Robotiq 명령/실제 퍼센트 |
| `wrench` | TCP 힘/토크 (fx fy fz tx ty tz) |
| `tcp_pose` | TCP Cartesian pose (x y z qx qy qz qw) |
| `cam1_frames` | cam1 프레임별 인덱스↔캡처 타임스탬프 매핑 |
| `cam2_frames` | cam2 프레임별 인덱스↔캡처 타임스탬프 매핑 |

**영상 프레임 ↔ 신호 행 교차 참조:** `synchronized` 테이블의 `cam1_frame_idx` / `cam2_frame_idx` 컬럼이 그 시점에 해당하는 MP4 프레임 번호입니다. HDF5 는 `pandas.read_hdf("vectors.h5", key="synchronized")` 로 열거나, 순수 `h5py` 로 직접 읽으면 됩니다.

> 신호를 발행하지 않은 토픽(예: 특정 브로드캐스터 미로드)은 에러 없이 **빈 컬럼(NaN)**으로 남습니다. 어떤 토픽이 실제로 들어왔는지는 `metadata.json` 의 `message_counts` 로 확인하세요. (`gripper`/`synchronized` 는 그 자체로는 카운트되지 않고, `gripper` 테이블에 쓰는 세 토픽 각각이 `gello_grip`/`grip_cmd`/`grip_pos` 키로 개별 카운트됩니다 — 아래 아키텍처 섹션 참고.)

---

## Architecture & Code Walkthrough

이 섹션은 `gello_recorder` 패키지의 코드 구조와 각 파일이 서로 어떻게 맞물리는지 설명한다. 나중에 이 코드를 수정할 엔지니어(또는 Claude Code 세션)를 위한 것이다. 모든 식별자(메서드명, 토픽명, 파일명)는 소스에서 그대로 가져왔다.

### 1. Module map (의존성 순서)

- **`hdf5_writer.py`** — 성장 가능한(growable) HDF5 테이블 writer. 하나의 "논리 테이블"이 하나의 HDF5 group이 되고, 각 컬럼은 자기 자신만의 resizable 1-D `float64` dataset이 되어 한 번에 한 행(row)씩 append된다. 예전 CSV 패턴(`_open_csv(name, header)` + `writer.writerow([...])`)을 거의 1:1로 대체한다. `h5py` + `numpy`에만 의존하며 **ROS/Qt import가 전혀 없다**. 핵심 클래스는 `Hdf5TableWriter`, 편의 생성자는 `open_h5_table(h5file, table_name, header)`. `python3 hdf5_writer.py`로 self-test 실행 가능.

- **`video_writer.py`** — 한 대의 카메라가 발행하는 `sensor_msgs/msg/CompressedImage`(JPEG 바이트)를 하나의 성장하는 MP4 파일로 쓰는 `Mp4FrameWriter`. `cv2` + `numpy`에만 의존, ROS import 없음. 카메라 해상도를 미리 알 필요 없이 첫 프레임을 디코드할 때 `(height, width)`를 자동 감지해 `cv2.VideoWriter`를 lazy하게 연다. 프레임 rate 매칭/중복/드롭은 하지 않고 decode → lazy open → write → frame index 반환만 한다. `python3 video_writer.py`로 self-test 실행 가능.

- **`recording_session.py`** — 위 두 순수 모듈을 조합하는 파일-I/O 코어 `RecordingSession`. "한 번의 녹화(session/take) 분량의 파일 전체"를 소유한다: 9개 테이블이 든 `vectors.h5` + `cam1.mp4` + `cam2.mp4`. **ROS/Qt/thread가 전혀 없다**. `python3 recording_session.py` self-test 있음.

- **`gello_ur_recorder_node.py`** & **`gello_gui_node.py`** — 두 개의 rclpy `Node` 진입점. 둘 다 `RecordingSession`을 실제 ROS 구독에 배선한다. 전자(`GelloUrRecorder`)는 헤드리스로 launch부터 Ctrl-C까지 무조건 녹화. 후자(`GelloRecorderGuiNode`)는 구독은 항상 켜두되 Start/Stop으로 디스크 쓰기를 게이팅(멀티 take).

- **`gello_recorder_gui.py`** — 오퍼레이터용 PyQt5 프런트엔드. RealSense 카메라 노드 2개를 subprocess로 띄우고, `GelloRecorderGuiNode`를 백그라운드 스레드에서 spin하며, Qt 이벤트 루프로 프리뷰/상태/버튼을 그린다.

**`setup.py`의 console_scripts 진입점** 2개:

```
gello_ur_recorder  = gello_recorder.gello_ur_recorder_node:main   # 헤드리스
gello_recorder_gui = gello_recorder.gello_recorder_gui:main       # GUI
```

### 2. RecordingSession (`recording_session.py`) — 공유 파일-I/O 코어

`RecordingSession`은 녹화 1회(session 또는 take)당 한 번 `RecordingSession(session_dir, camera_fps)`로 생성된다. 생성자는 `session_dir`을 만들고, `session_dir/vectors.h5`를 `'w'` 모드로 연 뒤 `open_h5_table`로 **9개 테이블 전부**를 생성하고, `session_dir/cam1.mp4` / `session_dir/cam2.mp4`에 대해 `Mp4FrameWriter` 두 개를 만든다. 이 세션 자신의 상대 시각 원점 `self.t0 = time.time()`을 기록하며, `t()`는 이 생성 시점 기준 상대 시각이다.

**9개 테이블의 정확한 헤더(컬럼명 그대로):**

1. `synchronized`: `["t_rel_s", "t_wall"]` + `gello_q1..gello_q6` + `gello_qd1..gello_qd6` + `["gello_grip"]` + `cmd1..cmd6` + `ur_q1..ur_q6` + `ur_qd1..ur_qd6` + `ur_eff1..ur_eff6` + `["grip_cmd", "grip_pos"]` + `["fx", "fy", "fz", "tx", "ty", "tz"]` + `["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]` + `["cam1_frame_idx", "cam2_frame_idx"]`
2. `gello_joint_states`: `["t_rel_s"]` + `q1..q6` + `qd1..qd6`
3. `ur_joint_states`: `["t_rel_s"]` + `q1..q6` + `qd1..qd6` + `eff1..eff6`
4. `command`: `["t_rel_s"]` + `cmd1..cmd6`
5. `gripper`: `["t_rel_s", "gello_grip", "grip_cmd", "grip_pos"]`
6. `wrench`: `["t_rel_s", "fx", "fy", "fz", "tx", "ty", "tz"]`
7. `tcp_pose`: `["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"]` (주의: `synchronized`의 `tcp_x..tcp_qw`와 이름이 다름 — 별개 네이밍)
8. `cam1_frames`: `["t_rel_s", "frame_idx"]`
9. `cam2_frames`: `["t_rel_s", "frame_idx"]`

(컬럼 블록 크기는 `_N = 6` 상수로, UR 관절 6개에 맞춰져 있다.)

**Public API 메서드 (각 행 레이아웃 / 정밀도):**

- `t() -> float` — `time.time() - self.t0`, 이 세션 생성 이후 경과 초.
- `bump(key)` — 공개 카운터 증가. 하나의 `write_*` 호출에 1:1 대응하지 않는 topic별 카운트를 위한 것(아래 counter-key 서브틀티 참조).
- `write_gello(pos, qd)` — `gello_joint_states` 행 `[t()] + pos + qd`. `pos`는 raw float, `qd`는 `None` 또는 `f"{v:.5f}"`. `"gello_joint_states"` 카운터를 bump.
- `write_gello_grip(gello_grip, grip_cmd, grip_pos)` — `gripper` 행 `[t(), gello_grip, grip_cmd, grip_pos]`, 각 값 `.4f` 또는 None. **어떤 카운터도 bump하지 않는다**(핵심, 아래 참조).
- `write_cmd(cmd)` — `command` 행 `[t()] + cmd`(raw). `"command"` bump.
- `write_ur(pos, vel, eff)` — `ur_joint_states` 행 `[t()] + pos + vel + eff`. pos/vel은 `.6f`, eff는 `.4f`, None은 NaN 통과. `"ur_joint_states"` bump.
- `write_wrench(wrench6)` — `wrench` 행 `[t()] + wrench6`, 6개 float 각 `.5f`. `"wrench"` bump.
- `write_tcp(tcp7)` — `tcp_pose` 행 `[t()] + tcp7`, 7개 float 각 `.6f`. `"tcp_pose"` bump.
- `write_cam1_frame(jpeg_bytes) -> int` — `cam1.mp4`에 프레임 쓰고, 성공 시(`idx >= 0`)에만 `cam1_frames`에 `[t(), idx]` 추가 + `"cam1_frames"` bump. decode 실패 시 `-1` 반환, 로깅도 카운트도 없음. **warm-up 로직 없음** — skip 판단은 호출자 몫.
- `write_cam2_frame(jpeg_bytes) -> int` — 카메라 2 / `cam2.mp4` / `cam2_frames`에 대해 동일.
- `write_sample(gello_q, gello_qd, gello_grip, cmd, ur_q, ur_qd, ur_eff, grip_cmd, grip_pos, wrench, tcp, cam1_frame_idx, cam2_frame_idx)` — `synchronized` 행. 컬럼 0/1은 `t_rel_s`(`.4f`) / `t_wall`(`time.time()` `.4f`); 이후 gello_q `.6f`, gello_qd `.5f`, gello_grip `.4f`, cmd `.6f`, ur_q `.6f`, ur_qd `.6f`, ur_eff `.4f`, grip_cmd/grip_pos `.4f`, wrench `.5f`, tcp `.6f`; 마지막에 cam1/cam2 frame index를 **raw(int 또는 None)로** append. **어떤 카운터도 bump하지 않는다.**
- `flush()` — 공유 `h5py.File`만 flush(비디오 writer는 flush 안 함).
- `close() -> dict` — `{"duration_s", "message_counts"}` 반환. 스냅샷을 파일 핸들을 건드리기 전에 먼저 뜨고, 두 비디오 writer를 close(idempotent) 후 HDF5 flush+close. 부분 생성 실패나 아무것도 안 쓴 경우에도 안전하게 유효한 dict를 반환한다.

**Counter-key 서브틀티 (반드시 이해할 것):** `write_gello_grip`은 **스스로 어떤 카운터도 bump하지 않는다**. 세 개의 서로 다른 ROS 토픽(GELLO grip, `grip_cmd`, `grip_pos`)이 모두 이 하나의 공유 `gripper` 테이블 행을 통해 쓰기 때문이다. 하지만 `metadata.json`의 message count는 세 토픽을 **각각 따로** 세야 한다. 그래서 호출자(노드)가 콜백마다 `session.bump("gello_grip")` / `session.bump("grip_cmd")` / `session.bump("grip_pos")` 중 자기 것을 별도로 호출한다. 이 규약을 어기고 `write_gello_grip` 안에 bump를 넣으면 세 토픽 카운트가 뭉개진다. (`message_counts`에는 `"gripper"`나 `"synchronized"` 라는 키 자체는 존재하지 않는다.)

`recording_session.py`의 self-test(`python3 recording_session.py`)는 9개 테이블 전부 + MP4 2개를 round-trip하고, `vectors.h5`를 read-only로 다시 열어 각 테이블 행 수와 spot value(첫 gello 행의 qd가 NaN, gripper 행의 None→NaN, `synchronized` row 1의 값들)를 assert한다. 손상 프레임이 카운터를 bump하지 않는지, `close()`가 idempotent한지, 미완성 객체(`__new__`)에서도 안전한지까지 검증한다.

### 3. Headless node (`gello_ur_recorder_node.py`)

`GelloUrRecorder(Node)`는 launch부터 Ctrl-C까지 **start/stop 게이팅 없이** 연속 녹화하는 헤드리스 노드다.

**선언하는 ROS 파라미터** (모두 `declare_parameter`):

- `output_root` (기본: `$GELLO_REPO_ROOT/ros2_ur_ws/gello_logs`)
- `session_dir` (빈 문자열이면 `session_<YYYYmmdd_HHMMSS>`를 `output_root` 아래 새로 만듦; 주어지면 그대로 사용 — run_recorder.sh가 rosbag을 같은 폴더에 떨구려고 씀)
- `sample_rate_hz` (기본 100.0)
- `flush_period_s` (기본 2.0)
- `cam1_topic` (기본 `/cam1/cam1/color/image_raw/compressed`)
- `cam2_topic` (기본 `/cam2/cam2/color/image_raw/compressed`)
- `camera_fps` (기본 30.0)
- `camera_warmup_s` (기본 **3.0** — GUI 쪽 기본값 4.0과 다름)

`__init__`에서 `RecordingSession`을 **정확히 한 번** 생성한다. start/stop 개념이 없으므로 세션은 노드 수명 전체와 같다.

**구독(전부 READ-ONLY)** — 토픽명 + 메시지 타입 그대로:

| 토픽 | 타입 | 콜백 |
|---|---|---|
| `/gello/joint_states` | `JointState` | `_on_gello` |
| `/gripper/gripper_client/target_gripper_width_percent` | `Float32` | `_on_gello_grip` |
| `/forward_position_controller/commands` | `Float64MultiArray` | `_on_cmd` |
| `/joint_states` | `JointState` | `_on_ur` |
| `/robotiq_gripper/command_percent` | `Float32` | `_on_grip_cmd` |
| `/robotiq_gripper/position_percent` | `Float32` | `_on_grip_pos` |
| `/force_torque_sensor_broadcaster/wrench` | `WrenchStamped` | `_on_wrench` |
| `/tcp_pose_broadcaster/pose` | `PoseStamped` | `_on_tcp` |
| `cam1_topic` (파라미터) | `CompressedImage` | `_on_cam1` |
| `cam2_topic` (파라미터) | `CompressedImage` | `_on_cam2` |

각 native-rate 콜백은 들어온 값을 파싱(`_reorder`로 `UR_JOINT_ORDER`에 정렬, GELLO 속도는 유한차분으로 계산)한 뒤 즉시 해당 `write_*`로 native-rate 테이블에 쓰고, 동시에 "최신 값 저장소"(`self._gello_q`, `self._ur_q` 등)를 갱신한다.

**시각 정렬(time-alignment) 전략:** `_sample_timer`가 `sample_rate_hz`(기본 100Hz)로 `_on_sample`을 호출한다. `_on_sample`은 그 순간의 **모든 신호의 최신 알려진 값**을 모아 `session.write_sample(...)`로 wide `synchronized` 행 하나를 쓴다. 즉 메시지별(per-message) 동기화가 아니라 **"이 tick 시점의 가장 최근 값"** 방식이다. 별도의 `_flush_timer`가 `flush_period_s`마다 `session.flush()`를 호출한다.

**카메라 warm-up 게이팅(인라인, per-frame):** Start 버튼이 없으므로 노드가 프레임마다 직접 게이팅한다. `_on_cam1`/`_on_cam2`는 각 카메라의 첫 프레임 시각을 `self._cam1_first_frame_t`/`self._cam2_first_frame_t`에 기록하고, `now - first_frame_t < self.camera_warmup_s`인 동안 프레임을 **드롭한다**(오토 익스포저 안정화 중의 녹색 틴트 프레임이 MP4에 안 들어가게). warm-up 경과 후에만 `session.write_cam1_frame(...)`을 호출한다.

`destroy_node()`는 `session.close()`를 호출해 통계를 받고, 최종 `metadata.json`을 쓴다.

### 4. GUI node (`gello_gui_node.py`)

`GelloRecorderGuiNode(Node)`는 평범한 rclpy Node지만, **구독은 항상 활성**(라이브 프리뷰 / 상태 패널용)인 반면 `start_recording()`이 호출되기 전에는 디스크에 아무것도 쓰지 않고 `stop_recording()`이 반환된 뒤에도 더 이상 쓰지 않는다. 한 프로세스 수명에서 여러 번의 start/stop "take"를 지원한다.

**두-락(two-lock) 설계 (정확히 이해할 것):**

- `_session_lock` — `RecordingSession` 참조(`self._session`)를 지키며, **모든 구독 콜백 안의 모든 write 호출 동안 잡혀 있다**. 그래서 콜백은 "살아있는 세션을 보고 그것을 통해 쓰거나" "None을 보고 no-op하거나" 둘 중 하나만 한다. 이것이 `stop_recording()`이 `self._session`을 None으로 바꾸는 스왑을 동시 실행되는 ROS 콜백과 race 없이 만드는 핵심이다. `stop_recording()`은 락 안에서 참조를 로컬로 빼고 `self._session = None`으로 바꾼 뒤, **실제 파일 I/O인 `session.close()`는 락 밖에서** 실행한다(이 시점엔 콜백이 이미 None을 보고 no-op하므로 `session`을 동시에 건드리는 게 없다).
- `_state_lock` — GUI 폴링 getter(`get_preview_frames()`, `get_state_snapshot()`)가 읽는 라이브 프리뷰 프레임 / 최신 값 필드를 별도로 지킨다. 녹화 상태와 분리돼 있어 **idle 상태에서도 라이브 프리뷰가 계속 동작한다.**

**카메라 warm-up (노드 수명당 1회):** `_cam1_first_frame_t`/`_cam2_first_frame_t`는 각 카메라의 **맨 첫 프레임에만** 세팅되고 take 사이에 **절대 리셋되지 않는다**. 그래서 여러 take가 warm-up 비용을 다시 지불하지 않는다. Start 버튼이 `cameras_ready()`가 True가 될 때까지(warm-up 한 번 경과) 그냥 비활성화되고, 그 후에는 활성 세션 중 들어오는 모든 프레임이 **무조건** write된다(헤드리스 노드와 달리 per-frame skip이 없음).

**Public API:**

- `cameras_ready() -> bool` — `warmup_seconds_remaining() <= 0.0`.
- `warmup_seconds_remaining() -> float` — 두 카메라 중 하나라도 첫 프레임이 없으면 `inf`.
- `is_recording() -> bool`
- `take_index() -> int`
- `start_recording() -> str` — warm-up 안 됐거나 이미 녹화 중이면 `RuntimeError`. take index 증가, `take_<NN>_<stamp>` 디렉터리에 새 `RecordingSession` 생성.
- `stop_recording() -> dict` — 녹화 중이 아니면 `RuntimeError`. `close()` 통계 + `session_dir` 반환.
- `get_preview_frames()` — `(cam1_frame, cam2_frame)` numpy 배열.
- `get_state_snapshot() -> dict` — 모든 최신 값 + `cam1_last_frame_age_s`/`cam2_last_frame_age_s`.

`destroy_node()`는 `_session_lock` 아래 세션을 빼서 None으로 만든 뒤 진행 중 take를 best-effort로 `close()`한다.

### 5. GUI front-end (`gello_recorder_gui.py`)

오퍼레이터용 PyQt5 프런트엔드. ROS는 `rclpy.init`/`spin`/`shutdown`과 노드의 thread-safe getter 외에는 직접 건드리지 않으며, Qt 위젯은 오직 Qt/메인 스레드에서만 만진다.

**카메라 subprocess 관리:** `_launch_realsense(camera_name, serial, color_profile)`가 각 RealSense 카메라 ROS2 노드를 `subprocess.Popen(argv, ..., start_new_session=True)`로 띄운다. `start_new_session=True`는 자기만의 프로세스 그룹을 주고, `_kill_process_group`이 `os.killpg(pgid, SIGTERM)`(안 죽으면 `SIGKILL`)로 그룹 전체를 확실히 죽인다 — 맨 `.terminate()`는 `ros2 launch`가 spawn한 `realsense2_camera_node` 손자 프로세스를 reap하지 못해 다음 실행 때 USB를 두고 싸우는 좀비를 남긴다. argv의 `serial_no:='...'` / `rgb_camera.color_profile:='...'`는 값 안에 임베디드 작은따옴표를 넣어 all-digit serial이 int로 coerce되는 것을 막는다.

**스레딩 모델:** 노드는 백그라운드 **daemon 스레드**에서 평범한 `rclpy.spin(node)`로 spin한다(명시적으로 single-threaded, `MultiThreadedExecutor` 아님). Qt 이벤트 루프는 메인 스레드가 소유한다. 세 개의 `QTimer`가 노드를 폴링한다:

- `_preview_timer` ~30Hz → `get_preview_frames()`
- `_state_timer` ~10Hz → `get_state_snapshot()`
- `_control_timer` ~5Hz → 버튼 enable/disable + status/take/elapsed

**종료:** `closeEvent()`는 창이 실제로 닫히기 전에 항상 (1) 진행 중 take가 있으면 `stop_recording()`으로 finalize, (2) 두 카메라 프로세스 그룹을 죽임, (3) `rclpy.shutdown()` 순으로 마무리한다.

**`main()` 설정(env-var 구동):** 모든 기본값은 `run_recorder.sh`의 env var 이름/값을 그대로 미러링한다. 또한 `import cv2`가 부작용으로 세팅한 `QT_QPA_PLATFORM_PLUGIN_PATH`를 `QApplication` 생성 전에 `os.environ.pop`으로 지워 시스템 PyQt5와 cv2 번들 Qt5 충돌을 피한다.

### 6. Design rationale callouts

- **`RecordingSession`이 ROS/Qt import를 하나도 안 하는 이유:** 독립적으로 단위 테스트가 가능하고, 두 진입점(헤드리스 노드 + GUI 노드) 사이에서 파일-I/O 로직을 중복 없이 재사용하기 위해서다.
- **헤드리스 노드와 GUI 노드의 warm-up 게이팅 방식이 다른 이유:** 헤드리스 노드는 게이팅할 버튼이 없어 프레임마다 게이팅해야 한다. GUI 노드는 Start 버튼이 `cameras_ready()`로 딱 한 번 게이팅하므로, 활성 세션 중에는 모든 프레임을 무조건 쓴다.
- **락을 하나가 아니라 두 개로 나눈 이유:** 녹화 start/stop churn(`_session_lock`)이 고빈도 프리뷰 폴링(`_state_lock`)을 절대 막거나 그것에 막혀선 안 되기 때문이다.

---

## Troubleshooting / Debugging History

이 섹션은 `gello_recorder` 패키지를 실제 하드웨어(real UR7e + real GELLO + 2x RealSense)에서 개발하며 실제로 겪고 해결한 문제들을 기록한다. 각 항목은 **Symptom** → **Root cause** → **Fix** 순서로 정리했다.

### 1. `libdiagnostic_updater.so` 누락 / undefined symbol

- **Symptom**: ROS2 launch 시점에 `libdiagnostic_updater.so`를 찾을 수 없다는 로딩 에러 또는 undefined symbol 에러가 발생한다.
- **Root cause**: 설치된 apt 패키지 `ros-humble-diagnostic-updater` 4.0.6 버전이 헤더와 Python 파일만 배포하고 정작 컴파일된 `.so` 공유 라이브러리를 포함하지 않았다.
- **Fix**:
  ```bash
  sudo apt-get install --only-upgrade -y ros-humble-diagnostic-updater
  ```
  (4.0.7로 업그레이드됨)

### 2. `librealsense2.so.2.58` 런타임 누락 (SDK ABI 불일치)

- **Symptom**: 실행 시 `librealsense2.so.2.58`를 찾을 수 없다는 에러가 발생한다.
- **Root cause**: `realsense2_camera` ROS2 노드는 RealSense SDK ABI 2.58에 대해 빌드/링크되어 있었지만, 실제 설치된 `librealsense2` 패키지는 2.56.4 버전뿐이었다.
- **Fix**:
  ```bash
  sudo apt-get install --only-upgrade -y ros-humble-librealsense2
  ```

### 3. `realsense2_camera_msgs` 버전 불일치 (`HardwareMonitorCommandSend`)

- **Symptom**: 카메라 노드 시작 시 `HardwareMonitorCommandSend`를 참조하는 undefined symbol 에러가 발생한다.
- **Root cause**: 컴파일된 카메라 노드와 실제 설치된 `realsense2_camera_msgs` 패키지 버전 간의 ABI 불일치.
- **Fix**:
  ```bash
  sudo apt-get install --only-upgrade -y ros-humble-realsense2-camera-msgs
  ```

### 4. RealSense USB 접근 에러 `RS2_USB_STATUS_ACCESS`

- **Symptom**: 카메라를 열 때 `RS2_USB_STATUS_ACCESS` 에러가 발생하며 장치를 claim하지 못한다.
- **Root cause**: 현재 사용자에게 RealSense USB 장치 접근 권한을 부여하는 udev 규칙이 설치되어 있지 않았다.
- **Fix**: (이름이 바뀐) `realsenseai` GitHub org에서 공식 `99-realsense-libusb.rules` udev 규칙 파일을 설치한 뒤:
  ```bash
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
  카메라를 다시 연결한다.

### 5. USB "disconnected" / "No such device" 에러 폭주 (하드웨어 고장으로 오인)

- **Symptom**: USB "disconnected" 및 "No such device" 에러가 대량으로 쏟아져 마치 하드웨어 고장처럼 보인다.
- **Root cause**: 실제 하드웨어 문제가 아니라, 이전에 크래시되거나 중단된 테스트 실행에서 남은 `realsense2_camera_node` 프로세스들(4개 이상)이 살아남아 동일한 2개의 물리 USB 카메라를 서로 차지하려고 싸우고 있었던 것이 원인.
- **Fix**: 재테스트 전마다 남은 프로세스를 확실히 죽이고, 아무것도 남지 않았는지 확인한 뒤 재실행한다.
  ```bash
  pkill -9 -f realsense2_camera_node
  pkill -9 -f "ros2 launch realsense2_camera"
  pgrep -fa realsense2_camera_node   # 아무것도 안 나와야 정상
  ```

### 6. 두 카메라 간 해상도 불일치

- **Symptom**: 동시 녹화 시 `cam1.mp4`와 `cam2.mp4`의 프레임 크기가 서로 다르다.
- **Root cause**: D435는 640x480 color가 기본, D435IF는 1280x720 color가 기본이라 동시 녹화 결과의 프레임 크기가 어긋난다.
- **Fix**: `rgb_camera.color_profile` launch 파라미터로 두 카메라를 같은 프로파일로 강제한다. `run_recorder.sh`와 `gello_recorder_gui.py` 양쪽에서 `COLOR_PROFILE` 환경 변수(기본값 `1280x720x30`)로 노출되어 있다.

### 7. `ros2 launch` CLI 숫자 인자 타입 강제변환 버그

- **Symptom**: 카메라 실행이 시작되자마자 크래시하며 다음과 비슷한 에러가 뜬다.
  ```
  parameter {serial_no} is of type {string}, setting it to {integer} is not allowed
  ```
- **Root cause**: `serial_no`는 `realsense2_camera` 노드에서 STRING 파라미터로 선언되어 있지만, 값이 전부 숫자(all-digits)라서 `ros2 launch`의 argv 파서가 이를 정수로 타입 추론해 버린다. 처음에는 카메라 launch의 stdout/stderr를 `/dev/null`로 리다이렉트하고 있어서 이 크래시가 한동안 보이지 않았다.
- **Fix**:
  - (a) 값을 argv 문자열 내부에서 embedded single quote로 감싼다 (셸 quoting이 아니라 인자 문자열 자체에 따옴표를 포함):
    ```python
    "serial_no:='{}'".format(serial)
    ```
    ```bash
    "serial_no:='${CAM1_SERIAL}'"
    ```
  - (b) 카메라 launch 로그를 `/dev/null`이 아니라 세션별 파일(`cam1_launch.log`, `cam2_launch.log`)로 리다이렉트해서 다음에는 이 부류의 실패가 눈에 보이게 한다.
  - 동일한 quoting 트릭은 `rgb_camera.color_profile`에도 적용한다.

### 8. 녹화 초반 3~4초 초록빛/색 바랜 화면

- **Symptom**: 모든 카메라 녹화의 처음 약 3~4초가 초록빛으로 물들거나 색이 바랜(washed-out) 상태로 나온다.
- **Root cause**: RealSense 컬러 센서는 스트림 시작 후 auto-exposure/auto-white-balance가 수 초간 안정화(settling)되는데, 이 구간이 종종 초록빛을 띤다.
- **Fix**: `camera_warmup_s` 파라미터 추가(headless 노드 ROS param 기본값 3.0s, GUI는 `CAMERA_WARMUP_S` 환경 변수 기본값 4.0s). 각 카메라 자신이 첫 프레임을 받은 시점부터 그 초 수가 지날 때까지 프레임을 버린다. headless 노드는 프레임마다 인라인으로 판정하고, GUI 노드는 `cameras_ready()`가 될 때까지 Start 버튼을 비활성화하는 방식으로 대신한다.

### 9. `ros2 run gello_recorder gello_ur_recorder` → "No executable found"

- **Symptom**: 실행 시 `No executable found` 에러.
- **Root cause**: `gello_recorder` 패키지의 `setup.py`에는 console_script 엔트리 포인트가 정의되어 있었지만, 그 스크립트를 어디에 설치할지 `colcon`/`setuptools`에 알려주는 `setup.cfg`가 빠져 있었다(`ur_gello_bringup`에는 있었으나 새로 만든 `gello_recorder`에는 처음에 없었음).
- **Fix**: 다음 내용의 `setup.cfg`를 추가하고 재빌드한다 (현재 이 패키지에는 이미 적용되어 있음):
  ```
  [develop]
  script_dir=$base/lib/gello_recorder
  [install]
  install_scripts=$base/lib/gello_recorder
  ```
  ```bash
  colcon build --symlink-install --packages-select gello_recorder
  ```

### 10. `RecordingSession.write_gello_grip`의 메시지 카운터 회귀 (출시 전 발견)

- **Symptom**: `metadata.json`에 기록되는 per-topic 메시지 카운트가 손상된다 — 서로 구분되어야 할 세 토픽의 카운트가 하나로 뭉개진다.
- **Root cause**: 리팩터 중 도입된 버전에서 `write_gello_grip`이 호출될 때마다 내부적으로 `self._bump("gello_grip")`를 호출했다. 그 결과 원래 별개여야 할 세 개의 per-topic 카운터(`gello_grip`, `grip_cmd`, `grip_pos`)가 단일 카운터로 뭉개졌다.
- **Fix**: `write_gello_grip` 내부의 bump 호출을 제거하고, 대신 public `bump(key: str)` 메서드를 추가했다. 모든 호출부(`gello_ur_recorder_node.py`의 `_on_gello_grip`/`_on_grip_cmd`/`_on_grip_pos`, `gello_gui_node.py`의 대응 콜백들)가 각자의 `write_gello_grip(...)` 호출 옆에서 명시적으로 `self._session.bump("gello_grip")` / `bump("grip_cmd")` / `bump("grip_pos")`를 호출하도록 수정했다.

### 11. PyQt5 GUI 실행 시 크래시: `Could not load the Qt platform plugin "xcb"`

- **Symptom**:
  ```
  qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
  ```
  + `QObject::moveToThread` 경고.
- **Root cause**: `opencv-python` 휠이 자체 Qt5 사본(플랫폼 플러그인 포함)을 번들하고 있으며, `import cv2`의 부작용으로 `QT_QPA_PLATFORM_PLUGIN_PATH` 환경 변수를 설정한다. `gello_gui_node.py`가 `cv2.imdecode`를 위해 `cv2`를 import하기 때문에, 이 import가 `main()`에서 `QApplication`이 생성되기 **전에** 전이적으로 일어난다.
- **Fix**: `gello_recorder_gui.py`의 `main()`에서 `GelloRecorderGuiNode`/`QApplication`을 생성하기 직전에 해당 환경 변수를 제거한다.
  ```python
  os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
  ```
  GUI를 다시 실행해 창이 정상적으로 렌더링되고 라이브 카메라 피드가 나오는 것을 확인해 수정 완료를 검증했다.

---

**마무리 참고**: 위의 11개 문제는 모두 시뮬레이션이나 mock이 아니라 **실제 물리 하드웨어**(real UR7e + real GELLO + 2개의 real RealSense 카메라)에 대해 테스트하면서 발견하고 고친 것이다. 따라서 이 목록의 어떤 항목이 재현되지 않는다면, 수정이 틀렸다고 가정하기 전에 먼저 환경/버전 차이를 의심하라.
