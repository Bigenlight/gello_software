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

## 왜 이게 차단점인가 — learner 시작 게이트 (G20)

**learner는 두 조건을 *동시에* 만족해야 학습을 시작한다** (`ur_env/learner/batches.py` 의
`RLPDBatchSampler.ready`, 2026-07-29 코드 확인):

```python
@property
def ready(self) -> bool:
    return (
        len(self.online_replay) >= self.training_starts   # 기본 100
        and len(self.offline_demos) > 0                   # ← 이게 0이면 영원히 False
    )
```

`training_starts` 기본값은 **100** (`ur_env/learner/config.py` 의 `training_starts: int = 100`).
즉 **offline demo가 하나도 없으면 online transition을 아무리 쌓아도 학습이 시작되지 않는다.**
못 미치면 `sample()`이 `LearnerBatchError`를 던지고, `ur_env/learner/runtime.py`가
`replay=<n>/<training_starts>`를 같이 찍는다.

> 🟢 **G20("canonical demo 없음")은 2026-07-29 닫혔다.** 사용자가 아래 23개 take를
> success로 승인했고 영구 artifact를 생성했다. 변환기가 outcome을 추측하지 않는 계약은
> 그대로이며, 이후 데이터에도 사람 라벨이 필요하다.

> **입력 레이아웃 주의 — 이 CLI 자체는 디렉터리 *이름*을 보지 않는다.**
> `convert_recorded_take()`가 요구하는 것은 디렉터리 안의 `vectors.h5` · `cam1.mp4` ·
> `cam2.mp4`와 h5 안의 7개 그룹(`command`, `ur_joint_states`, `gripper`, `wrench`,
> `tcp_pose`, `cam1_frames`, `cam2_frames`)뿐이다. GUI recorder의 `take_<NN>_<stamp>/`와
> headless recorder의 `session_<stamp>/`는 **둘 다 같은 `RecordingSession`이 쓰므로 파일
> 구성이 같고, 이 변환기는 둘 다 받는다.**
>
> **그런데도 `take_*/`로 녹화해야 한다.** kanu의 classifier 라벨링 파이프라인
> (`cube_classifier_pipeline.py prepare`)이 **`take_*/` 디렉터리만 읽기 때문이다.**
> headless `run_recorder.sh`가 만드는 `session_<stamp>/`는 그쪽에 **보이지 않고**, 촬영은
> 정상 종료되므로 **손실이 라벨링 시점에야 드러난다** — 그때는 이미 세션이 끝나 있다.
> 기존 학습 take는 전부 GUI에서 나왔다. 근거: `32a193b`,
> [`../ros2_ur_ws/src/gello_recorder/README.md`](../ros2_ur_ws/src/gello_recorder/README.md) `:153-159`.

> **sidecar와 무관하다.** 2026-07-29 reward classifier sidecar 변경은 **gRPC 실시간 경로**의
> 것이다. 여기서 만드는 offline demo는 `--outcome`으로 **명시 라벨**을 받고 분류기를 전혀
> 타지 않으므로, 이 문서의 계약은 그 변경에 영향받지 않는다. 아래 observation 절의
> `IMAGE_CROP` 적용도 **그대로 옳다** — offline demo는 **정책이 보는 이미지**여야 한다.

## 2026-07-20 묶음 변환

`take_23_20260720_210316`은 1.6초 동안 주차 상태였으므로 현재 분석에서는 제외한다.
나머지 take는 2026-07-29 사용자가 전부 성공 episode라고 확인했다. 재현 명령은 다음과 같다.

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

2026-07-20 승인 23개 take를 직접 측정한 결과 원본은 10 Hz가 아니었다. count/duration
기준으로 command·UR state는 take별 약 87–118 Hz, gripper는 약 35.8–37.7 Hz,
cam1/cam2는 약 29.9–30.1 Hz였다. 변환기는 원본 행 번호나 고정 decimation 비율을 쓰지
않고 timestamp 공통구간에 0.1초 격자를 만든다. 각 격자 시각에는 미래 보간 없이 직전 최신
샘플을 선택하고, action은 command의 `t → t+0.1초` FK 차이로 다시 계산한다. 따라서 이
데이터는 이미 올바른 10 Hz canonical pickle로 변환됐다.

원본 stream이 10 Hz보다 느리면 직전 값이 여러 격자에 유지될 수 있다. 다만 이전 샘플의
age가 기본 0.20초를 넘으면 변환을 거절한다. 이 경우 target Hz나 age 제한을 임의로 바꾸지
말고 끊긴 take를 제외하거나 센서 기록을 복구한다. 10 Hz는 현재 actor/learner action 계약이라
원본 native Hz에 맞춰 바꾸는 값이 아니다.

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

🪤 **`/tmp/gello-hil-rl-learner-venv`는 `/tmp`에 있다 — 리부트하면 사라진다.** 없으면
`HIL_SERL_KANU_RUNBOOK_KO.md` 절차로 다시 만든다. 변환 CLI 자체는 이 venv가 필요 없다
(`python3`로 충분하다 — 스크립트가 `sys.path`를 스스로 세운다). 이 venv는 **pinned NumPy
1.26에서의 재로딩 확인**에만 쓴다.

## 코드와 대조한 항목 (2026-07-29)

이 문서의 계약을 코드에서 직접 확인했다. **문서와 코드가 어긋나는 곳은 발견되지 않았다.**

| 이 문서의 서술 | 확인한 코드 |
| --- | --- |
| `--outcome` 필수, `success\|truncated` 두 값 | `scripts/convert_recorded_takes_to_demo.py::_parser` (`required=True`, `choices`) |
| 출력 파일을 덮어쓰지 않음 | 같은 파일 `--output` 도움말(`never overwritten`) + `write_recorded_demo_pickle` |
| `success` → 마지막만 `reward=1/done=true/mask=0` · `truncated` → 전부 `reward=0/mask=1` + 마지막 `truncated=true` | `ur_env/learner/recorded_demo.py` 의 `truncated = terminal and outcome == "truncated"` 및 `"rewards"/"masks"/"dones"/"truncated"` 조립부 |
| 그리퍼 hysteresis `≥0.7` 닫기 / `≤0.3` 열기 / 그 사이 유지 | 같은 파일 `if trigger >= 0.7:` / `elif trigger <= 0.3:` |
| JSON의 `saturated_action_count` · `saturated_action_fraction` · `max_raw_action_group_norm` | 같은 파일 `TakeConversionStats` 필드 + 생성부 |
| 19-D state 순서 (`gripper` / `force` / `rel pose` / `torque` / `vel`) | `ur_env/observation_schema.py::STATE_FEATURE_INDEX` 를 직접 출력해 대조 — 인덱스 0 / 1:4 / 4:10 / 10:13 / 13:19 일치 |
| `load_demo_pickle` + `len(demos)` + `demos.sidecars[-1].metadata` | `ur_env/learner/demo.py` 의 `LoadedDemos.__len__` · `DemoSidecar.metadata` · `load_demo_pickle` |
| 필수 h5 그룹 7종 | `ur_env/learner/recorded_demo.py::convert_recorded_take` 의 `required_groups` |
| `python3` 로 바로 실행 가능 | `scripts/convert_recorded_takes_to_demo.py` 머리의 `_REPO_ROOT`/`_INFRA_ROOT` `sys.path` 부트스트랩 |

*(줄 번호 대신 심볼로 적었다 — 이 트리는 동시 편집 중이라 줄 번호가 계속 밀린다.)*

**아래 「현재 실데이터 smoke 결과」의 수치는 `40b99f8` 작성자의 실행 기록이며 이 세션에서
재실행하지 않았다.** 인용할 때 그 조건을 함께 옮길 것.

## 현재 실데이터 smoke 결과

- `take_23_20260720_210316`을 truncated로 변환: 15 transitions
- signal 최대 age 36.8 ms, camera 최대 age 45.1 ms
- 변환 pickle을 pinned NumPy 1.26 learner strict loader로 재로딩 성공
- 같은 15개를 실제 frozen ResNet-10으로 encode해 feature demo pool
  `(cam1/cam2: (N,1,4,4,512) float32, augmentation=none)` 생성 성공
- `take_23`을 제외한 2026-07-20 23개 take: 2,037 transitions 생성 확인
- 위 2,037개 중 norm clamp 516개(25.33%), 최대 raw group norm 2.755
- 전체 23개에서 signal 최대 age 44.6 ms, camera 최대 age 60.3 ms

사용자 승인 후 `success`로 만든 영구 artifact는 다음과 같다.

- laptop3: `/home/laptop3/hil-serl-artifacts/demos/cube_in_cup_20260720_success_23takes.pkl`
- Kanu: `/home/junhyeong/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl`
- 크기: `203,573,172 B`, transitions: `2,037`
- SHA256: `f97185582401ce7570d44fddc33d1bd64b215d7e32d6384d5fe13e1b405032fa`
- laptop3와 Kanu learner strict loader 통과, 양쪽 digest 일치

> 🟢 **따라서 G20은 닫혔다.** production learner의 offline-demo 시작 조건을 만족한다.
>
> 🪤 그리고 **norm clamp 25.33%(2,037개 중 516개, 최대 raw group norm 2.755)** 를 그냥
> 넘기지 마라. 포화는 변환 버그가 아니라 **옛 teleop 주기와 현재 10 Hz RL 스텝의 차이**지만,
> 방향만 보존하고 크기를 깎은 액션이 demo의 1/4이라는 뜻이다. **어느 take를 demo에 넣을지는
> 별도의 데이터 품질 결정**이고, 라벨링과 같이 해야 한다.
