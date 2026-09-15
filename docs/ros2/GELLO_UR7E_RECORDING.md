# GELLO → UR7e 진단 레코더 (데이터 로깅)

GELLO 리더에서 뽑히는 값과 UR7e의 **실제 관절(위치·속도·토크)**, 브리지 커맨드, 그리퍼, TCP 힘/토크·포즈까지 한 번의 실행에서 CSV로 저장하는 도구입니다. 트래킹 오차·지연·진동을 오프라인으로 분석할 때 씁니다. **읽기 전용**(구독만) — 로봇/GELLO에 명령을 내리지 않습니다.

## 실행

텔레오퍼(sim 또는 실기)를 **먼저** 띄운 뒤, **두 번째 터미널**에서:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_recorder.sh                 # CSV만
BAG=true ./run_recorder.sh        # + 전체 토픽을 ros2 bag으로도 캡처
RATE=200 ./run_recorder.sh        # synchronized.csv 샘플레이트(Hz), 기본 100
```

> **`HEADLESS=true`는 레코더에 필요 없습니다.** 그 플래그는 텔레오퍼 실행(`run_ur7e_gello_real.sh`)이 Remote 모드로 드라이버를 띄우게 하는 것이고, 레코더는 읽기 전용(구독만)이라 드라이버를 시작하지 않습니다. 실기에서는 터미널 1에서 `HEADLESS=true ./run_ur7e_gello_real.sh`, 터미널 2에서 `./run_recorder.sh` — 같은 PC/같은 ROS 그래프(기본 `ROS_DOMAIN_ID=0`)라 그대로 토픽을 봅니다.

`Ctrl-C`로 종료 → flush + `metadata.json` 마무리(그리고 bag 정리). 직접 실행도 가능:

```bash
ros2 run ur_gello_bringup gello_ur_recorder --ros-args -p sample_rate_hz:=100.0
```

## 저장 위치 (하위 폴더)

```
ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/
├─ synchronized.csv        ← 핵심: 모든 신호를 100 Hz로 시간 정렬한 wide 테이블
├─ gello_joint_states.csv  ← GELLO 관절 위치 + 유한차분 속도 (native ~30 Hz)
├─ ur_joint_states.csv     ← UR7e 실제 위치/속도/토크 (native ~500 Hz)
├─ command.csv             ← 브리지가 보낸 커맨드 (/forward_position_controller/commands)
├─ gripper.csv             ← GELLO 그리퍼 / 그리퍼 커맨드 / 그리퍼 실제 위치
├─ wrench.csv              ← TCP 힘·토크 (force_torque_sensor_broadcaster)
├─ tcp_pose.csv            ← TCP 데카르트 포즈 (tcp_pose_broadcaster, 로드된 경우)
├─ metadata.json           ← 파라미터·시작시각·토픽별 수신 개수
└─ rosbag/                 ← BAG=true일 때만: 전체 토픽 원본
```

> `gello_logs/`는 `.gitignore` 처리되어 있습니다(데이터는 커밋 안 함, 코드만 커밋).

### 깊이(depth) 녹화 — **기본 OFF (RGB만), `ENABLE_DEPTH=1` 로 켠다**

```bash
CAMS=true ENABLE_DEPTH=1 ./run_recorder.sh                     # 헤드리스
ENABLE_DEPTH=1 ros2 run gello_recorder gello_recorder_gui      # GUI
ENABLE_DEPTH=1 ros2 run gello_recorder task_recorder_gui       # 태스크 GUI
```

기본 캡처는 **RGB만**이다. depth 는 세션마다 명시적으로 요청하는 opt-in 이고, 켜면
세션 폴더에 `depth.h5` 가 추가되어 프레임마다 `compressedDepth` 페이로드(PNG, `16UC1`,
단위 **mm**)가 한 행씩 저장된다. `t_rel_s`는 `vectors.h5`와 **같은 시간 원점**을 쓰므로
관절/색상 프레임과 바로 정렬되고, 각 카메라의 depth 내부 파라미터(`camera_info`)와
depth→color 외부 파라미터(`extrinsics/depth_to_color`)도 같이 저장된다. 색상 프레임을
버리는 워밍업 구간에서는 depth 프레임도 **정확히 같은 기준으로** 버린다(별도 시계 없음).

🔴 **왜 기본 OFF 인가 — 2026-09-14 하루의 기본 ON 이 54개 take 의 타임스탬프를 망가뜨렸다.**
depth 녹화는 카메라당 30 Hz 구독 2개와 약 6 MB/s HDF5 쓰기를 **모든 로봇 토픽을 처리하는
바로 그 단일 rclpy spin 스레드**에 얹는다. 그 결과 executor 라운드 주파수가 ~100 Hz →
60–69 Hz 로 떨어졌고, 그보다 빨리 발행되는 토픽은 전부 가득 찬 큐에서 가장 오래된 샘플을
받게 되어 **`ur_joint_states` +0.900 초, `tcp_pose`/`wrench` 약 +0.45 초** 낡은 채로
기록됐다. 상세와 수정은 아래 「타임스탬프 아티팩트」 절. **결함은 고쳤지만 비용은 그대로**라
depth 는 opt-in 으로 되돌렸다. 켜면 기동 시 한 줄이 찍힌다:

```
depth ON: ~6 MB/s disk, +2 subscriptions/cam, recorder CPU +~35 %; watch ros_lag_s
```

- 정렬(`ALIGN_DEPTH=1`, 기본 **0**): 카메라 노드 안에서 depth를 1280×720 색상 이미지에
  맞춰 재샘플링하고 `aligned_depth_to_color/*` 토픽을 녹화한다. **실측(2026-09-14) 노드
  CPU 9 % → 49 %, 색상·depth 모두 30 → ~25 Hz**라 기본에서 끈다. 저장된 내·외부 파라미터로
  오프라인에서 정렬하는 편이 싸다.
- 카메라 자체는 문제가 아니었다(자체 전원 Genesys USB3 허브, 2대 동시): 색상 30 Hz 유지,
  depth ~29 Hz, 프레임당 80–130 KB, 노드 CPU ~9 %, USB 오류 0. 병목은 **레코더 쪽**이었다.
  `launch_cameras.sh`(HIL 경로)는 원래부터 **기본 0**이다 — depth 를 읽는 소비자가 없다.
- `metadata.json` / GUI 정지 요약에는 **항상** `record_depth` 가 적힌다. RGB-only take 도
  "depth 가 꺼져 있었다"를 스스로 증언하므로, depth 파일이 없는 것과 구별된다.
- 읽기:

```python
from gello_recorder.depth_writer import read_depth_frame
# 자세한 파일 레이아웃은 ros2_ur_ws/src/gello_recorder/README.md의 출력 구성 절 참조
```

### 타임스탬프 아티팩트 (2026-09-14) — 로봇 행이 최대 0.9 초 낡게 기록됐다

**증상.** `command` 가 `ur_joint_states` 를 0.9 초 앞서고, 같은 RTDE 패킷에서 나오는
`tcp_pose` 와 `ur_joint_states` 가 0.495 초 어긋난다(물리적으로 불가능하다 — 둘은 같은
controller_manager 사이클에 발행된다). 파형은 온전하고 잔차는 0.5 mm 수준의 **순수 지연**,
NaN 0개, 파일 검증 전부 통과. **데이터 안에는 아무 표시도 없었다.**

**원인.** 레코더의 모든 구독 콜백이 **단일 rclpy spin 스레드**에서 돌고, 각 행의 `t_rel_s`
는 **콜백이 실행된 시각**이다(`RecordingSession.t()`, 헤더 스탬프는 저장하지 않았다).
`SingleThreadedExecutor` 는 한 라운드에 구독당 메시지 하나를 처리하므로 구독의 서비스
주파수 = 라운드 주파수이고, 그보다 빨리 발행되는 토픽은 KEEP_LAST 히스토리가 항상 가득 차
**가장 오래된 샘플**을 받는다. 그 나이는 정확히 `QoS depth ÷ 발행 주파수` 다:

| 토픽 | 당시 QoS depth | 발행 | 예상 = 실측 나이 |
|---|---|---|---|
| `/joint_states` | **100** | ~100 Hz | **0.900 s** |
| `/tcp_pose_broadcaster/pose` | **50** | ~100 Hz | **~0.45 s** |
| `/force_torque_sensor_broadcaster/wrench` | **50** | ~100 Hz | **~0.45 s** |
| `/forward_position_controller/commands` | 50 | 250 Hz | 0.02–0.20 s |
| 카메라 color/depth, GELLO, 그리퍼 (≤ 39 Hz) | 10 / 50 / 20 | 30–40 Hz | **0.022 s (신선)** |

라운드 주파수를 ~100 Hz 에서 60–69 Hz 로 끌어내린 것이 `c694d2c` 의 **depth 녹화**다
(카메라당 구독 2개 추가 + 약 5–6 MB/s HDF5 쓰기, 게다가 컬러 프레임은 프리뷰용으로 한 번,
MP4 writer 가 또 한 번 디코드하고 있었다 — 실측 1280x720 JPEG 디코드 9.3 ms + MPEG-4
인코드 6.9 ms, 카메라 2대 30 Hz 면 **초당 약 1.5 초어치의 일**). 임계값이 두 세션 사이에서
정확히 문제의 토픽들을 가로질렀다: 7월 세션(depth 없음)은 아무것도 굶지 않아 세 토픽이
98.9 Hz 로 기록됐다.

**영향받는 릴리스:**

| 릴리스 | 낡음 |
|---|---|
| `carrot_in_pot_raw` (2026-09-14, 54 take, EEF 모드) | `ur_joint_states` **+0.900 s** (54 take 전부 5 ms 그리드 최적 τ 동일), `tcp_pose` / `wrench` **≈ +0.45 s**. `command` · 카메라 · depth · GELLO · 그리퍼는 **신선** |
| 2026-07-24 GUI take 2개 | 0.83 s / 0.42 s (같은 결함의 약한 형태 — GUI 레코더 설계 문제였고 depth 부하가 그것을 상수로 크게 만들었다) |
| `cube_in_cup` / `banana_in_pot` (7월, joint 모드, depth 없음) | ≈ 0.05 s — 실질적으로 영향 없음 |

⚠️ **raw take 는 다시 녹화할 필요가 없다.** 형상이 보존된 순수 지연이므로 테이블별 상수
시프트로 완전히 복구된다. LeRobot 재변환은 `observation.state[0:6]`(ur_q)의 `t_rel_s` 를
**−0.900 s** 시프트해 다시 만들었고, 그 사실은 `meta/source_takes.json` 의
`ur_joint_states_lag_s` 에 적혀 있다. 🛑 **`grip_cmd` 같은 굶지 않은 채널에는 이 시프트를
적용하면 안 된다** — 없던 0.9 초 오정렬을 새로 만든다.

**수정** (전부 `ros2_ur_ws/src/gello_recorder/`, 새 모듈 `gello_recorder/spin_health.py` 에
진단 전문이 들어 있다):

1. **헤더 스탬프 저장.** stamped 메시지에서 온 모든 테이블에 마지막 컬럼 `stamp_s`
   (float64 초, 없으면 NaN): `gello_joint_states` · `ur_joint_states` · `tcp_pose` ·
   `wrench` · `cam1_frames` · `cam2_frames`. **기존 컬럼은 이름·순서 그대로**이고 뒤에
   덧붙기만 하므로 예전 리더가 그대로 동작한다. `command`(Float64MultiArray)와
   `gripper`(Float32 ×3)는 **메시지 타입에 헤더가 없어** 스탬프를 가질 수 없다 — 그래서
   2번이 먼저였다.
2. **고속 구독 큐 깊이 100/50 → 5.** 최악 나이가 100 Hz 에서 50 ms, 500 Hz 에서 10 ms 로
   묶인다. 이 레코더는 도착한 것을 그대로 쓰므로 깊은 큐는 아무 이득이 없고 **"행이 빠짐"을
   "행이 낡음"으로 바꿀 뿐**이다. 빠지는 편이 낫다 — 빠진 것은 보이고, 낡은 것은 안 보인다.
3. **프레임 I/O 를 spin 스레드 밖으로.** 컬러 디코드+MP4 인코드와 depth PNG HDF5 append 를
   백그라운드 writer 스레드 하나가 처리하고(`RecordingSession.submit_cam_frame` /
   `submit_cam_depth_frame`), 콜백은 바이트만 넘긴다. **행의 `t_rel_s` 는 제출(도착) 시점에
   찍어서 큐로 들려 보낸다** — 일을 옮기되 시각은 옮기지 않는다. `close()` 는 파일을
   마무리하기 전에 큐를 비우므로 프레임 수가 정확히 맞고, 큐가 가득 차 버린 프레임은
   `dropped_frames` 로 센다(정상값 0). GUI 프리뷰 디코드도 latest-wins 디코더 스레드로 뺐다.
4. **굶음 경보 2겹.** 실시간으로 `ros_lag_s`(= ROS now − `/joint_states` 헤더 스탬프)를
   GUI 상태줄에 상시 표시하고 0.15 s 초과 시 5초에 한 번 WARN. take 종료 시에는 네 native
   테이블의 **기록** 주파수가 0.5 % 이내로 같아졌는데 90 Hz 미만이면
   `native tables converged at X Hz — spin thread starved; robot rows may be stale` 를
   WARN 으로 찍고 `metadata.json` / GUI 정지 요약에 `spin_starvation_suspected` ·
   `ros_lag_s_max` · `native_rates_hz` · `dropped_frames` · `record_depth` 를 남긴다.

**수정 후 실측** (2026-09-14, laptop3, 카메라 1대 + 합성 퍼블리셔 `/joint_states` 500 Hz ·
tcp/wrench/commands 각 100 Hz, 28.5 s):

| | RGB만 (기본) | `ENABLE_DEPTH=1` |
|---|---|---|
| 레코더 프로세스 CPU | 131.6 % | 135.0 % (카메라 1대분) |
| 기록 주파수 `command`/`tcp_pose`/`wrench`/`ur_joint_states` | 94.6 / 94.5 / 94.5 / 184.0 Hz | 86.4 / 86.4 / 86.3 / 138.9 Hz |
| `stamp_s` 나이 중앙값 ur / tcp / wrench | **9.3 / 8.0 / 7.2 ms** | **25.6 / 12.9 / 12.4 ms** |
| 같은 값 최대 | 21.4 / 54.0 / 50.7 ms | 615.4 / 54.3 / 59.3 ms |
| `ros_lag_s_max` | 0.021 s | 0.613 s |
| `dropped_frames` | 0 | 0 |
| `spin_starvation_suspected` | false | false |

네 테이블의 기록 주파수가 **더 이상 수렴하지 않는다**는 것이 핵심 판정이다(굶었다면 넷이
소수점까지 같아진다). `ur_joint_states` 나이는 **900 ms → 9.3 ms**(RGB) / 25.6 ms(depth).

자세한 스키마·상태 필드·한계는
[`ros2_ur_ws/src/gello_recorder/README.md`](../../ros2_ur_ws/src/gello_recorder/README.md)
§5·5-1·5-2.

## `synchronized.csv` 열 (분석용 메인 파일)

| 열 | 의미 |
|---|---|
| `t_rel_s`, `t_wall` | 시작 기준 경과초 / 벽시계 유닉스시각 |
| `gello_q1..6` | GELLO 관절 각도(rad, UR 순서) |
| `gello_qd1..6` | GELLO 관절 속도(rad/s, 유한차분) |
| `gello_grip` | GELLO 그리퍼 폭 (0=열림..1=닫힘) |
| `cmd1..6` | 브리지가 로봇에 보낸 목표 위치(필터/클램프 후) |
| `ur_q1..6` | UR7e **실제** 관절 각도(rad) |
| `ur_qd1..6` | UR7e **실제** 관절 속도(rad/s) |
| `ur_eff1..6` | UR7e 관절 토크/전류(effort) |
| `grip_cmd`, `grip_pos` | 그리퍼 목표 % / 실제 % (0=열림..1=닫힘) |
| `fx..fz`, `tx..tz` | TCP 힘(N)·토크(Nm) |
| `tcp_x..tcp_qw` | TCP 위치(m) + 쿼터니언 |

비어 있는 열 = 해당 토픽이 이번 실행에 발행되지 않음(예: sim에는 wrench/tcp 없음, 팔만 돌리면 gripper 없음).

## 빠른 분석 예 (pandas)

```python
import pandas as pd
df = pd.read_csv("gello_logs/session_XXXX/synchronized.csv")
# 트래킹 오차: 브리지 커맨드 vs 로봇 실제
err = (df[[f"cmd{i}" for i in range(1,7)]].values
       - df[[f"ur_q{i}" for i in range(1,7)]].values)
# 진동: 정지 구간에서 ur_qd(실제 속도)의 표준편차
print(df[[f"ur_qd{i}" for i in range(1,7)]].std())
# 지연: gello_q 대비 ur_q의 상호상관 피크로 lag 추정
```

## 허깅페이스 업로드 (raw + LeRobot, depth 포함) — 2026-09-14 carrot 릴리스 절차

이 절차로 [`Bigenlight/carrot_in_pot_raw`](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_raw)와
[`Bigenlight/carrot_in_pot_lerobot_v3`](https://huggingface.co/datasets/Bigenlight/carrot_in_pot_lerobot_v3)를 올렸다
(54 take, 18,557 프레임, 619 s). 스크립트는 전부 [`scripts/dataset/`](../../scripts/dataset/)에 있다.

**환경.** LeRobot 변환·검증은 `/home/laptop3/youngwoong_ws/lr_env/bin/python`(lerobot 0.6.1, git 핀 `8a74e0a`)로만 돈다.
`ros2_ur_ws/act_venv`에는 설치하지 말 것(실기 배포 검증 환경). 이 lerobot에는 depth가 정식으로 들어 있다 —
feature `info: {"is_depth_map": true}` + `DepthEncoderConfig`. 데이터셋 양식은 여전히 **v3.0**(2026-09 기준 최신, v3.1/v4 없음).
HF 계정 `Bigenlight`, 토큰은 `hf auth whoami`로 확인.

1. **스테이징(하드링크).** `gello_logs/`는 모든 세션의 take가 섞인 풀이므로 한 태스크만 골라 묶는다. 복사가 아니라 하드링크라 디스크를 안 먹는다.
   ```bash
   S=/home/laptop3/youngwoong_ws/Put_carrot_in_pot; mkdir -p $S
   cd ros2_ur_ws/gello_logs; for d in take_*_20260914_*; do mkdir -p $S/$d; for f in vectors.h5 cam1.mp4 cam2.mp4 depth.h5; do ln -f $d/$f $S/$d/$f; done; done
   ```
   raw 릴리스는 이 폴더 그대로다: take 폴더들 + `README.md`(데이터셋 카드) + `DATA_DICTIONARY.md` + `dataset_stats.json` + `assets/`(세팅 사진, 첫 프레임, depth 샘플).
2. **raw 통계.** `python3 scripts/dataset/make_carrot_raw_stats.py --data $S` → `dataset_stats.json`(ffprobe `-count_frames`로 프레임 수 대조, NaN/Inf, 주기, 값 범위, depth 유효율, 리더/팔로워 차이). 카드의 숫자는 전부 여기서 나온다.
   📌 2026-09-16부터는 태스크 무관 판 `scripts/dataset/make_raw_stats.py --data $S --dataset Bigenlight/<repo> --task "<task string>"`를 쓴다(`--no-depth` 자동 감지, `stamp_s` 블록과 `timestamp_lag.recorder_artefact_expected` 추가). carrot 스크립트는 그 릴리스 재현용으로 남겨 둔다.
3. **LeRobot 변환.**
   ```bash
   cd /home/laptop3/youngwoong_ws && lr_env/bin/python /home/laptop3/gello_software/scripts/dataset/convert_carrot_to_lerobot.py \
     --data Put_carrot_in_pot --out carrot_in_pot_lerobot --procs 4 --threads 2   # [--exclude take_NN_...]
   ```
   레시피는 cube와 같다(마스터 클럭 `cam1_frames/t_rel_s` 30 fps, 나머지 스트림은 각자의 `t_rel_s`로 nearest-timestamp 리샘플, cam2는 카메라 타임스탬프로 정렬, `gello_*` 제외, `grip_cmd` ffill/bfill). 추가된 것:
   - `observation.images.cam1_depth` / `cam2_depth` = `video (480, 848, 1)`, `info.is_depth_map=true`. depth.h5의 자체 `t_rel_s`로 마스터 프레임마다 최근접 depth 프레임을 고른다(두 토픽이 위상 고정이 아니라 한 프레임 이내).
   - 인코더 `DepthEncoderConfig(depth_min=0.0, depth_max=10.0, shift=0.0, use_log=False)`: HEVC gray12le + x265 `lossless=1`, 12-bit **선형** 양자화(스텝 2.442 mm → ±1.25 mm, **무효 0 → 0 보존**, D435 범위 안 클리핑 없음). ⚠️ lerobot 기본값(log, `depth_min=0.01`)은 **0을 10 mm로 바꿔 무효 마스크를 잃는다** — 쓰지 말 것.
   - `meta/depth_cameras.json`(K/D/외부 파라미터, 양자화 파라미터, 무효값 설명)과 `meta/source_takes.json`(에피소드↔take)을 `finalize()` **뒤에** 쓴다. `info.json` 최상위에 임의 키를 넣으면 lerobot이 다음 rewrite에서 지워 버린다.
   - 54 take 변환 실측 약 20분(take당 ~20 s), 결과 약 3 GB(depth가 대부분, 프레임당 cam1 ~90 KB / cam2 ~55 KB).
4. **검증.** `lr_env/bin/python scripts/dataset/validate_carrot_conversion.py --raw $S --lerobot carrot_in_pot_lerobot --json validation.json`.
   변환기를 import하지 않고 리샘플링을 다시 구현해 비교한다. cube의 7개 검사(state/action `max|Δ|`=0, 프레임 수, NaN, task, AV1↔raw 상관, 타임스탬프)에 depth 6개를 더했다(info 블록, 프레임 수, 왕복 오차 median≤0.75/p99≤1.25/max≤1.25 mm, raw 0→0, 사이드카 K/D/외부 파라미터 전 take 일치, depth↔color Δt). 카드에 인용하는 검증 숫자는 `validation.json`에서 가져온다.
5. **업로드 + 태그.** `hf upload`는 폴더를 그대로 올리고 repo를 없으면 만든다. LeRobot 쪽은 `v3.0` 태그가 있어야 `LeRobotDataset("<repo>")`로 허브에서 바로 열린다(`hf upload`는 태그를 안 만든다).
   ```bash
   hf upload Bigenlight/carrot_in_pot_raw        $S                     --repo-type dataset
   hf upload Bigenlight/carrot_in_pot_lerobot_v3 carrot_in_pot_lerobot  --repo-type dataset
   python3 -c "from huggingface_hub import HfApi; HfApi().create_tag('Bigenlight/carrot_in_pot_lerobot_v3', tag='v3.0', repo_type='dataset')"
   lr_env/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d=D('Bigenlight/carrot_in_pot_lerobot_v3'); print(d.meta.total_episodes, d[0]['observation.images.cam1_depth'].shape)"
   ```

**이번 세션에서 알게 된 것.**
- 녹화가 **EEF 텔레옵 모드**(`control_mode:=eef`)였으므로 `gello_joint_states`는 리더 자체 관절이지 팔로워의 사본이 아니다(1번 관절은 반상관까지 난다). `command`가 IK 결과인 진짜 관절 타깃이다. cube/banana는 joint 모드였다. 카드에 모드를 반드시 적을 것.
- 손목 depth(cam2)는 D435 최소거리(848×480에서 약 0.2 m)에 걸려 파지 직전 유효 픽셀이 37~68 %다. 손목 depth를 제대로 쓰려면 다음 촬영부터 해상도를 낮추거나 카메라를 뒤로 물린다.
- LeRobot 허브 뷰어는 v3.0 데이터셋의 영상(RGB·depth 모두)을 어차피 안 보여준다. depth 렌더는 `lerobot-dataset-viz`(Rerun) 로컬.
- 허브에 depth를 넣은 공개 데이터셋 156개 중 79개는 8-bit 컬러맵 영상이라 metric depth가 아니다 — `shape[2]==1` + `is_depth_map`가 아니면 RGB 파이프라인을 탄다.
- 상위 lerobot의 ACT는 1채널 입력을 못 받는다(issue #4475). "depth를 올렸다" ≠ "depth로 학습된다".

### 2026-09-16 — `Bigenlight/orange_bowl_in_purple_bowl_raw` (수정된 레코더로 찍은 첫 릴리스)

태스크 "put the orange bowl into the purple bowl", **52 take**(take_01~54, 22·40은 조작자가 삭제), 2026-09-15 23:09~09-16 00:01,
EEF 모드, **RGB만**(depth OFF 기본값), 16,009 프레임 / 534 s / 343 MB. 스테이징 `/home/laptop3/youngwoong_ws/Orange_bowl_in_purple_bowl/`.
위 절차의 1·2·5단계만으로 올렸다(LeRobot 변환은 별도). 이 코퍼스가 **레코더 수정의 실기 증거**다: `stamp_s` 열이 있고,
command 대비 `ur_joint_states` 지연이 52 take 전부 0.140~0.150 s(= 실제 서보 추종 지연, 빼면 안 됨), tcp↔joints 상호상관 ≈ 0,
표 간 `stamp_s` 오프셋 < 40 ms, `detect_spin_starvation` 0/52. **타임스탬프 보정 불필요.** LeRobot 변환 시 τ 이동 없이
`stamp_s`로 정렬하면 된다. 세팅 사진은 카드 「Setup photo」 자리에 추가 예정.

## 관련 문서

- [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md) — 턴키 셋업 + 튜닝(§5, One-Euro 진동 필터)
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실기 런북
- [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) — 2F-85 그리퍼
