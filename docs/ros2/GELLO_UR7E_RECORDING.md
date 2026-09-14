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

### 깊이(depth) 녹화 — 2026-09-14부터 기본 ON

카메라를 함께 띄우는 경로(`CAMS=true ./run_recorder.sh`, `gello_recorder_gui`,
`task_recorder_gui`)는 이제 RealSense **depth 스트림도 기본으로 녹화**한다. 세션
폴더에 `depth.h5`가 추가되고, 프레임마다 `compressedDepth` 페이로드(PNG, `16UC1`,
단위 **mm**)가 한 행씩 저장된다. `t_rel_s`는 `vectors.h5`와 **같은 시간 원점**을 쓰므로
관절/색상 프레임과 바로 정렬되고, 각 카메라의 depth 내부 파라미터(`camera_info`)와
depth→color 외부 파라미터(`extrinsics/depth_to_color`)도 같이 저장된다. 색상 프레임을
버리는 워밍업 구간에서는 depth 프레임도 **정확히 같은 기준으로** 버린다(별도 시계 없음).

- 끄기: `ENABLE_DEPTH=0` (색상만, 이전과 동일한 카메라 인자).
- 정렬(`ALIGN_DEPTH=1`, 기본 **0**): 카메라 노드 안에서 depth를 1280×720 색상 이미지에
  맞춰 재샘플링하고 `aligned_depth_to_color/*` 토픽을 녹화한다. **실측(2026-09-14) 노드
  CPU 9 % → 49 %, 색상·depth 모두 30 → ~25 Hz**라 기본에서 끈다. 저장된 내·외부 파라미터로
  오프라인에서 정렬하는 편이 싸다.
- 실측 비용(자체 전원 Genesys USB3 허브, 2대 동시): 색상 30 Hz 유지, depth ~29 Hz,
  프레임당 80–130 KB, 노드 CPU ~9 %, USB 오류 0. 이 값이 기본 ON의 근거다.
  `launch_cameras.sh`(HIL 경로)는 depth를 읽는 소비자가 없으므로 **기본 0을 유지**한다.
- 읽기:

```python
from gello_recorder.depth_writer import read_depth_frame
# 자세한 파일 레이아웃은 ros2_ur_ws/src/gello_recorder/README.md의 출력 구성 절 참조
```

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

## 관련 문서

- [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md) — 턴키 셋업 + 튜닝(§5, One-Euro 진동 필터)
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실기 런북
- [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) — 2F-85 그리퍼
