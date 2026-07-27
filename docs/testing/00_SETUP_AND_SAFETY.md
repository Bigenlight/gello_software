# 00 — 셋업 · 빌드 · 안전

대상 checkout: `/home/laptop3/gello_software` (브랜치 `feat/gello-ur7e-humble-22.04`)

```bash
export WT=/home/laptop3/gello_software
```

---

## 1. 세션 시작 시 무조건 먼저 하는 것

```bash
cd $WT
git status --short          # 다른 사람이 만든 커밋 안 된 수정 확인
git log --oneline -3
git worktree list
```

**커밋 안 된 수정이 보이면 지우지 말 것.** 위 hardware 변경은 commits `4171e7a`,
`6a0b127`과 learner/hardware merge `248255f`에 통합됐다. canonical checkout에는 별도의
사용자 산출물이 있을 수 있으므로 `git checkout .`, `git stash`, `git clean -fd`를 자동으로
실행하지 않는다. 이 디렉터리의 `파일:줄` 근거가 어긋나면 `rg`로 내용을 다시 찾는다.

---

## 2. 빌드

### 2.1 ROS2 워크스페이스 (`ur_gello_bringup`)

```bash
cd $WT/ros2_ur_ws
./build_ur7e.sh
```

- 이 스크립트는 `rosdep install ... --skip-keys dynamixel_sdk` 후
  `colcon build --packages-select ur_gello_bringup`을 돈다 (`build_ur7e.sh:28`, `:30`).
- `dynamixel_sdk`는 apt가 아니라 `pip install --user dynamixel-sdk`로 들어온다
  (`build_ur7e.sh:8-11`). 실기 GELLO를 쓸 때만 필요하다.
- **언제 다시 빌드해야 하나:** `ros2_ur_ws/src/**`의 파이썬 노드 / launch / entry-point /
  **config yaml**을 고쳤을 때. 반면 `serl_ur_infra`는 editable 설치라 재빌드가 필요 없다
  (`serl_ur_infra/RVIZ_HIL_TEST_CLI.md`의 "빌드 규칙" 참조).

빌드 결과 확인:

```bash
ls $WT/ros2_ur_ws/install/setup.bash    # 있으면 빌드됨
```

### 2.2 `serl_ur_infra` (editable)

```bash
python3 -m pip install --user pynput
python3 -m pip install --user -e $WT/serl_ur_infra --no-deps
```

> `--no-deps`가 중요하다. deps를 풀면 시스템 ROS Humble의 numpy/grpc/protobuf를
> 갈아엎을 수 있다. gRPC 계열은 **별도 venv**를 쓴다 → `05_COMMS_GRPC.md` §1.

### 2.3 (필요할 때만) upstream hil-serl 서브모듈

canonical checkout에서 다음으로 상태를 확인한다.

```bash
cd $WT && git submodule status
# 앞의 '-' = 미초기화
```

앞에 `-`가 붙었다면 `serl_launcher` import 전 다음으로 초기화한다.

```bash
git -C $WT submodule update --init third_party/hil-serl
```

---

## 3. 환경변수 함정

### 3.1 `GELLO_REPO_ROOT` — checkout을 바꿀 때 반드시 신경 쓸 것

`gello_publisher` 노드는 리포 루트의 파이썬 패키지 `gello/`를 import한다.
`ros2 run`으로 직접 띄우면 cwd가 `sys.path`에 안 올라가서 `No module named 'gello'`로 죽는다.

- 래퍼 스크립트들은 **스크립트 자신의 위치 기준**으로 자동 설정한다:
  `export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"`
  — `run_ur7e_gello_real.sh:100`, `run_eef_gui.sh:23`, `run_hil_gui.sh:24`,
  `run_operator_console.sh:20`, `remote_helpers.sh:49`.
- **함정:** `${GELLO_REPO_ROOT:-...}`는 *이미 설정된 값이 있으면 그것을 쓴다*.
  셸에 예전 `GELLO_REPO_ROOT=$HOME/gello_software`가 export되어 있으면,
  다른 checkout의 스크립트를 실행해도 **예전 경로의 `gello/` 코드가 로드될 수 있다.**

```bash
# 확인
echo "GELLO_REPO_ROOT=${GELLO_REPO_ROOT:-<unset>}"

# 세션에서는 명시적으로 canonical root를 지정한다
export GELLO_REPO_ROOT=$WT
```

`ros2 run`으로 직접 띄울 때는 앞에 붙인다:

```bash
GELLO_REPO_ROOT=$WT \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $WT/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

### 3.2 `ROBOT_IP`

기본값 `192.168.10.11`이 `run_ur7e_gello_real.sh:83`, `run_ur7e_gripper.sh:17`,
`remote_helpers.sh:50`, `ur7e_gripper_only.launch.py`의 `DeclareLaunchArgument`에 각각
따로 박혀 있다. 다른 IP를 쓸 거면 **전부** 넘겨야 한다.

### 3.3 `set -euo pipefail` vs ROS setup.bash (해결됨, 커밋 안 됨)

ROS Humble의 `setup.bash`는 `AMENT_TRACE_SETUP_FILES` 같은 변수를 **정의하기 전에 읽는다.**
`set -u` 아래에서는 그 시점에 스크립트가 죽는다.

| 스크립트 | 옵션 | 상태 |
|---|---|---|
| `run_ur7e_gripper.sh:14` | `set -euo pipefail` | **수정됨(커밋 안 됨)** — `:29-32`에서 `set +u` / source / `set -u` |
| `build_ur7e.sh:13` | `set -euo pipefail` | **수정됨(커밋 안 됨)** — `:24-26`에서 동일 처리 |
| `run_ur7e_gello_real.sh:81` | `set -e` | 원래부터 정상 |
| `run_eef_gui.sh:20` | `set -e` | 원래부터 정상 |
| `run_hil_gui.sh:21` | `set -e` | 원래부터 정상 |

> 새 스크립트를 쓸 때 `set -u`를 쓰고 싶으면, ROS/워크스페이스 setup을 source하는
> 구간만 `set +u` … `set -u`로 감싼다. 위 두 파일이 정본 패턴이다.

---

## 4. 오프라인 테스트 (로봇 불필요, 위험 0)

### 4.1 `ur_gello_bringup` — 436 tests

```bash
cd $WT/ros2_ur_ws/src/ur_gello_bringup
python3 -m pytest test/ -q -p no:anyio
```

- `-p no:anyio`가 **필수**다. 이 머신의 시스템 pytest 6.2.5 + 사용자 anyio 플러그인 조합이
  `ModuleNotFoundError: No module named '_pytest.scope'`로 죽는다.
- 기대: `436 tests collected` (2026-07-27 재확인).

### 4.2 `serl_ur_infra`

```bash
cd $WT/serl_ur_infra
python3 -m pytest tests -q -p no:anyio --ignore=tests/test_learner_policy_checkpoint.py
```

- `--ignore=`가 **필수**다. `tests/test_learner_policy_checkpoint.py:15`의 모듈 레벨
  `pytest.importorskip("jax")`가 pytest 6.2.5에서 **컬렉션 전체를 중단시킨다.**
  그냥 `pytest tests`를 돌리면 에러 없이 `no tests collected`가 뜨고,
  아무것도 안 돌았는데 통과한 것처럼 보인다. (2026-07-27 재현 확인.)
- 무시 후 기대: `109 tests collected` (문서 작성 시점. 다른 담당자가 테스트를 추가/삭제 중이라
  숫자는 변한다 — 숫자보다 **0이 아닌지**와 **실패 0인지**를 본다.)

### 4.3 JAX가 필요한 테스트

로봇 랩톱에는 jax가 없다. `test_learner_policy_checkpoint.py`,
`test_actual_agent_checkpoint_integration.py`는 canonical checkout
(`/home/laptop3/gello_software`) 또는 Kanu에서 돌린다.

---

## 5. 세션 전 체크리스트 (실기)

로봇을 켜고 뭔가 움직일 수 있는 세션이면 **전부** 통과해야 한다.

### 5.1 물리

- [ ] 작업 공간이 비어 있다. **팔꿈치·어깨가 지나갈 공간**까지 확보했다
      (EEF 모드에서는 TCP가 안 움직여도 팔꿈치가 크게 스윙한다 → `03_EEF_MODE.md` §2).
- [ ] **E-STOP이 손 닿는 곳**에 있고, 눌러본 적 있어서 위치를 안다.
- [ ] 그리퍼 손가락 사이에 손·물체가 없다 (연결 시 auto-calibrate로 open/close 스윕을 한다).
- [ ] GELLO 리더가 안정된 자세로 거치되어 있다.

### 5.2 로봇 상태 (dashboard, 포트 29999)

```bash
echo -e 'robotmode\n'    | nc 192.168.10.11 29999
echo -e 'safetystatus\n' | nc 192.168.10.11 29999
echo -e 'is in remote control\n' | nc 192.168.10.11 29999
```

기대(=검증된 정상 상태):

```
Robotmode: RUNNING
Safetystatus: NORMAL
true
```

- `POWER_OFF`면 그리퍼는 **죽어 있다** (tool 전압이 안 나온다).
  `echo -e 'power on\n' | nc ... 29999` → `echo -e 'brake release\n' | nc ... 29999`
  (`docs/ros2/GELLO_UR7E_GRIPPER.md:97-105`).
- `is in remote control`이 `false`면 HEADLESS(Method B)를 쓸 수 없다.
  펜던트에서 Remote로 바꿔야 한다. **Remote → Local 복귀는 펜던트에서만 가능하다**(안전 설계,
  `remote_helpers.sh:12-14`).

ROS2 쪽 래퍼도 있다:

```bash
source $WT/ros2_ur_ws/remote_helpers.sh
ur_mode        # get_robot_mode
ur_program     # program state
ur_help        # 전체 목록
```

### 5.3 Method A / B — 하나만 고른다

| | 방법 | 시작 | 리버스 인터페이스 끊김 복구 |
|---|---|---|---|
| **A** | 펜던트에 External Control URCap 프로그램 | `ur_load ur_caps.urp` → `ur_play` | `ur_play` |
| **B** ⭐ | Headless (URScript 직접) | `HEADLESS=true ./run_ur7e_gello_real.sh` | `ur_resend` |

**둘을 한 세션에 섞지 말 것** — 두 URScript 주입 경로가 서로 싸운다
(`remote_helpers.sh:31-33`).

### 5.4 소프트웨어

- [ ] `ros2 topic list`에 좀비 노드가 없다. 이전 세션이 남아 있으면 `:54321`(그리퍼)이
      점유되어 다음 세션이 붙지 못한다 → `01_GRIPPER.md` §5.
- [ ] `GELLO_REPO_ROOT`가 의도한 워크트리를 가리킨다 (§3.1).
- [ ] `git status`로 남의 작업물을 확인했다 (§1).

---

## 6. 🛑 비상 정지 — 우선순위

**위에서부터. 망설이면 아래 것을 쓰지 말고 위 것을 쓴다.**

| 순위 | 수단 | 무엇이 일어나는가 | 언제 |
|---|---|---|---|
| **1** | **물리 E-STOP** (펜던트/외부 버튼) | 로봇 전원 회로 차단, 브레이크 체결. 소프트웨어와 무관하게 항상 작동 | **충돌 임박·사람 위험·"뭔가 이상하다" 전부** |
| 2 | 펜던트 정지 버튼 / 프로그램 Stop (`ur_stop`) | URScript 실행 중단 → 팔 정지 | E-STOP보다 덜 급한 이상 |
| 3 | 실행 중인 터미널의 **Ctrl-C** | launch 트리 종료 → 명령 발행 중단 → `forward_position_controller`가 **마지막 명령 자세를 홀드** | 소프트웨어 오작동, 팔은 위험 위치가 아님 |
| 4 | (EEF 모드) GUI **DISENGAGE** 또는 `~/eef_disengage` | 브리지가 즉시 명령 생성 중단, 팔 홀드 | 텔레옵 중 통제권 회수 |

Ctrl-C에 대한 정확한 거동: 브리지가 죽으면 컨트롤러는 **마지막으로 받은 포즈를 그대로 홀드**한다.
감속 램프도 fault도 없다 (`docs/ros2/GELLO_UR7E_SETUP_CLI.md:667`).
즉 **Ctrl-C는 "그 자리에 세우는 것"이지 "안전한 곳으로 빼는 것"이 아니다.**

### 6.1 ⛔ 정지가 **아닌** 것들 — 절대 이걸 믿지 말 것

| 착각 | 실제로 일어나는 일 | 근거 |
|---|---|---|
| **RL 데드맨을 놓으면 팔이 선다** | **아니다.** 개입만 해제되고 **정책이 즉시 팔을 계속 움직인다.** `GelloIntervention.action()`은 미개입 시 정책 액션을 그대로 통과시킨다 | `serl_ur_infra/ur_env/envs/wrappers.py:287-290` |
| **HIL GUI DISENGAGE = 정지** | **아니다.** GUI는 `/hil/deadman`만 발행한다. 로봇도 브리지도 건드리지 않는다. DISENGAGE는 "정책으로 복귀"다 | `gello_hil_gui_node.py:6-10`, `:38-41` |
| **GUI가 죽으면 팔이 선다** | **아니다.** 0.5 s 하트비트 끊김 워치독은 `engaged=False`로 fail-safe할 뿐 → **정책이 이어받는다** | `wrappers.py:145` (`STALE_S = 0.5`), `:158-165` |
| **ESC를 누르면 정지한다** | **아니다.** `self.terminate=True` → 그 **에피소드가 끝나고**, 그 다음 `reset()`이 `go_to_reset()`으로 **팔을 RESET_JOINTS로 이동시킨다.** ESC는 "정지"가 아니라 "지금 에피소드 끝내고 리셋 자세로 가"다. 게다가 pynput 리스너라 **터미널 포커스**가 필요하고, 반영은 다음 step 경계에서다 | `ur7e_env.py:200-212`(리스너), `:260`(done), `:274-280`(reset→go_to_reset), `:284` |
| **펜던트 속도 슬라이더를 0%로 내리면 정지** | **아니다.** 속도 스케일일 뿐이며 명령 스트림은 계속 흐른다. 슬라이더를 올리는 순간 밀린 명령이 그대로 실행된다. 정지 수단으로 쓰지 말 것 | (UR PolyScope 동작. 리포 근거 없음 — 조작 원칙) |
| **`DRY_RUN=True`니까 안전하다** | 조건부로 맞다. `DRY_RUN`은 `URRosBackend(dry_run=...)`로 전달되어 명령 발행을 막지만, **`run_rviz_hil.py`는 `DRY_RUN=False`를 일부러 박아 놨다**(`tests/run_rviz_hil.py:56`). 그 스크립트를 **실기에 겨누지 말 것** | `config.py:131`, `run_rviz_hil.py:34-35`, `:56` |
| **`pos_scale:=0.0`이면 팔이 안 움직인다** | **아니다.** TCP 위치만 고정되고 회전 채널은 100% 살아 있어 팔이 크게 스윙한다 | `03_EEF_MODE.md` §2, `config/ur7e_gello_eef.yaml:66-73` |

### 6.2 protective stop 복구

```bash
source $WT/ros2_ur_ws/remote_helpers.sh
ur_unlock     # /dashboard_client/unlock_protective_stop  (원인을 먼저 제거한 뒤에!)
ur_resend     # Method B 복구 (또는 Method A면 ur_play)
```

**순서:** ① 원인 제거(장애물 치우기, 속도 낮추기) → ② `ur_unlock` → ③ `ur_resend`/`ur_play`.
원인을 안 없애고 unlock하면 바로 다시 걸린다.
