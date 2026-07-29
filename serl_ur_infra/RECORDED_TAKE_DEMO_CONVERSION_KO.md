# recorder take → HIL-SERL offline demo 변환

## 한 줄 요약

`gello_recorder`의 `take_*/{vectors.h5,cam1.mp4,cam2.mp4}`는 아래 CLI로 learner가
직접 받는 canonical pickle로 변환할 수 있다. 종료 라벨은 추측하지 않으므로
`--outcome success|truncated`를 반드시 명시한다.

```bash
cd /home/laptop3/gello_software

python3 serl_ur_infra/scripts/convert_recorded_takes_to_demo.py \
  ros2_ur_ws/gello_logs/take_01_20260720_205207 \
  --output /tmp/cube_demo_take01.pkl \
  --outcome success
```

출력 pickle은 그대로 Kanu learner의 `--demo-path`에 전달한다. learner는 시작할 때
raw 128×128 이미지를 frozen ResNet-10으로 **한 번만** encode하고, 장기 demo/replay
buffer에는 `(1,4,4,512) float32` feature만 보관한다.

## 2026-07-20 묶음 변환

`take_23_20260720_210316`은 1.6초 동안 주차 상태였으므로 현재 분석에서는 제외한다.
나머지 take가 전부 성공 episode라는 **사람의 확인을 받은 뒤에만** 다음처럼 실행한다.

```bash
cd /home/laptop3/gello_software

mapfile -t CUBE_TAKES < <(
  find ros2_ur_ws/gello_logs -maxdepth 1 -type d \
    -name 'take_*_20260720_*' \
    ! -name 'take_23_20260720_210316' | sort
)

python3 serl_ur_infra/scripts/convert_recorded_takes_to_demo.py \
  "${CUBE_TAKES[@]}" \
  --output /원하는/경로/cube_in_cup_20260720_canonical.pkl \
  --outcome success
```

성공과 실패/중단 take가 섞여 있으면 한 번에 같은 라벨을 주지 않는다. 성공 묶음과
중단 묶음을 별도 pickle로 만들고 learner에 `--demo-path`를 여러 번 전달한다.

```bash
python serl_ur_infra/scripts/run_rlpd_learner_server.py \
  ... \
  --demo-path /path/success_demos.pkl \
  --demo-path /path/truncated_demos.pkl
```

## 변환 계약

### 동기화

- `synchronized` group이 비어 있어도 된다.
- native-rate table을 공통 유효 구간에서 10 Hz로 latest-at-or-before(as-of) join한다.
- 기본 최대 age는 signal 0.20초, camera 0.20초다. 초과하면 조용히 보간하지 않고
  실패한다.
- 카메라 frame index는 `cam1_frames`/`cam2_frames` timestamp로 MP4에 연결한다.

### observation

실제 actor와 동일한 canonical v2를 만든다.

```text
state: (1,19) float32
  [0]       gripper_pose
  [1:4]     tcp_force
  [4:10]    reset 기준 relative tcp_pose (xyz + Euler xyz)
  [10:13]   tcp_torque
  [13:19]   tool-frame tcp_vel (J(q) @ qd를 현재 TCP 회전으로 변환)

cam1/cam2: (1,128,128,3) uint8 RGB
  MP4 BGR → CubeInCupEnvConfig.IMAGE_CROP → resize 128×128 → RGB
```

### action

`command` table은 `/forward_position_controller/commands`의 UR joint target이다.
각 0.1초 endpoint를 실제 `ur_kin.fk()`로 pose로 바꾼 뒤 다음을 계산한다.

```text
base translation = p_cmd(t+1) - p_cmd(t)
base rotation    = log(R_cmd(t+1) @ R_cmd(t).T)
tool delta       = R_observation(t).T @ base delta
normalized       = tool delta / [0.0125 m, 0.0625 rad]
```

옛 teleop가 현재 10 Hz RL 한 스텝보다 빠른 구간은 표현 범위를 넘을 수 있다.
이때 기존 `GelloIntervention`과 동일하게 translation/rotation vector를 각각 norm 1로
비례 축소해 방향을 보존한다. CLI JSON의 다음 값을 반드시 확인한다.

```text
saturated_action_count
saturated_action_fraction
max_raw_action_group_norm
```

포화는 변환 오류가 아니라 과거 teleop 주기와 현재 RL 주기의 차이다. 비율을 숨기지
않지만, 지나치게 높은 take를 demo에 포함할지는 별도 데이터 품질 결정이다.

그리퍼 action은 기록된 GELLO trigger에 actor와 동일한 hysteresis를 적용한다.

```text
trigger >= 0.7 → -1 (close)
trigger <= 0.3 → +1 (open)
그 사이        → 이전 명령 유지
```

직전 실제 `grip_pos`가 이미 열린 상태에서 open, 이미 닫힌 상태에서 close를 반복하면
기본 `grasp_penalty=-0.02`를 부여한다.

### 종료 라벨

- `--outcome success`: 마지막 transition만 `reward=1, done=true, mask=0`
- `--outcome truncated`: 전부 `reward=0, done=false, mask=1`, 마지막에
  `truncated=true`

GUI가 success/failure를 파일에 기록하지 않았으므로 converter는 디렉터리 이름이나
녹화 종료를 성공으로 추측하지 않는다.

## 검증

CLI는 생성 전 각 transition을 production strict loader로 검사하고, 기존 출력 파일을
절대 덮어쓰지 않는다. 출력 후 learner 환경에서도 다시 확인할 수 있다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/tmp/gello-hil-rl-learner-venv/bin/python - <<'PY'
from ur_env.learner import load_demo_pickle

path = "/path/to/canonical.pkl"
demos = load_demo_pickle(path)
print("transitions:", len(demos))
print("last label:", demos.sidecars[-1].metadata)
PY
```

로컬 recorder Python은 NumPy 2.2, pinned learner는 NumPy 1.26이므로 ndarray pickle의
내부 모듈명이 다르다. strict loader는 오직 `numpy._core` → `numpy.core` 이름 변경만
호환 처리하며 나머지 pickle 오류는 그대로 실패시킨다. pickle은 기존 계약대로
**신뢰하는 로컬 artifact에만** 사용한다.

## 현재 실데이터 smoke 결과

- `take_23_20260720_210316`을 truncated로 변환: 15 transitions
- signal 최대 age 36.8 ms, camera 최대 age 45.1 ms
- 변환 pickle을 pinned NumPy 1.26 learner strict loader로 재로딩 성공
- 같은 15개를 실제 frozen ResNet-10으로 encode해 feature demo pool
  `(cam1/cam2: (N,1,4,4,512) float32, augmentation=none)` 생성 성공
- `take_23`을 제외한 2026-07-20 23개 take: 2,037 transitions 생성 확인
- 위 2,037개 중 norm clamp 516개(25.33%), 최대 raw group norm 2.755
- 전체 23개에서 signal 최대 age 44.6 ms, camera 최대 age 60.3 ms

마지막 묶음은 품질 검사 목적으로 메모리에서 `truncated`로 변환했을 뿐, 성공이라고
라벨한 영구 artifact는 아직 만들지 않았다.
