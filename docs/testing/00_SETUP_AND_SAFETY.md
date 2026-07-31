# 00 — 셋업 · 빌드 · 안전

대상 checkout: `/home/laptop3/gello_software` (브랜치 `feat/gello-ur7e-humble-22.04`)

```bash
export WT=/home/laptop3/gello_software
```

> 🔧 **정정 (2026-07-29):** 2026-07-27 판은 `WT=/home/laptop3/gello_worktrees/hil-hardware-comms`
> 였다. 머지 커밋 `3f199d4`가 그 브랜치를 통합 checkout으로 가져왔으므로 **이제는 통합
> checkout이 정본**이다. 워크트리는 `1a4f93d`에 멈춰 있어 threshold 0.2와 카메라 시리얼
> 자동 해석이 빠져 있다. 이유는 `README.md` §2.

---

## 1. 세션 시작 시 무조건 먼저 하는 것

```bash
cd $WT
git status --short          # 다른 사람이 만든 커밋 안 된 수정 확인
git log --oneline -3        # 3f199d4 머지 위에 있는지 확인
git submodule status        # 앞의 '-' = 미초기화 (§2.3)
```

**커밋 안 된 수정이 보이면 지우지 말 것.** 하드웨어 변경은 commits `4171e7a`,
`6a0b127`과 learner/hardware merge `248255f`에, HIL 안전 수정은 `d49d0f6`~`ee3240e`에,
actor 하드웨어 작업은 merge `3f199d4`에 통합됐다. checkout에는 별도의
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

다음으로 상태를 확인한다.

```bash
cd $WT && git submodule status
# 앞의 '-' = 미초기화
```

앞에 `-`가 붙었다면 `serl_launcher` import 전 다음으로 초기화한다.

```bash
git -C $WT submodule update --init third_party/hil-serl
```

📌 2026-07-29 실측: `third_party/hil-serl`은 초기화되어 있다(`-` 없음).
**미초기화 상태의 증상이 "에러"가 아니라 "조용한 skip"이라는 점이 중요하다** — §4.2 참조.

---

## 3. 환경변수 함정

### 3.1 `GELLO_REPO_ROOT` — checkout을 바꿀 때 반드시 신경 쓸 것

`gello_publisher` 노드는 리포 루트의 파이썬 패키지 `gello/`를 import한다.
`ros2 run`으로 직접 띄우면 cwd가 `sys.path`에 안 올라가서 `No module named 'gello'`로 죽는다.

- 래퍼 스크립트들은 **스크립트 자신의 위치 기준**으로 자동 설정한다:
  `export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"`
  — `run_ur7e_gello_real.sh:100`, `run_eef_gui.sh:23`, `run_hil_gui.sh:24`,
  `run_operator_console.sh:19`, `remote_helpers.sh:49`, `run_hil_actor.sh:255`.
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

### 3.3 `set -euo pipefail` vs ROS setup.bash (해결·커밋됨 `4171e7a`)

ROS Humble의 `setup.bash`는 `AMENT_TRACE_SETUP_FILES` 같은 변수를 **정의하기 전에 읽는다.**
`set -u` 아래에서는 그 시점에 스크립트가 죽는다.

| 스크립트 | 옵션 | 상태 |
|---|---|---|
| `run_ur7e_gripper.sh:14` | `set -euo pipefail` | **수정·커밋됨** — `:29-32`에서 `set +u` / source / `set -u` |
| `build_ur7e.sh:13` | `set -euo pipefail` | **수정·커밋됨** — `:24-26`에서 동일 처리 |
| `run_ur7e_gello_real.sh:81` | `set -e` | 원래부터 정상 |
| `run_eef_gui.sh:20` | `set -e` | 원래부터 정상 |
| `run_hil_gui.sh:21` | `set -e` | 원래부터 정상 |

> 새 스크립트를 쓸 때 `set -u`를 쓰고 싶으면, ROS/워크스페이스 setup을 source하는
> 구간만 `set +u` … `set -u`로 감싼다. 위 두 파일이 정본 패턴이다.

### 3.4 🛑 인터프리터와 `PYTHONPATH` — 2026-07-27 실기에서 4번 실패한 자리

2026-07-27 실기 세션에서 actor 기동이 **코드가 아니라 실행 환경 때문에만** 4번 실패했다.
아래 두 규칙이 그 전부다. (래퍼 `ros2_ur_ws/run_hil_actor.sh`가 이걸 강제한다 → `09` §0.)

#### (1) gRPC를 만지는 것은 전부 **venv python**으로

```bash
ACTOR_PY=/home/laptop3/venvs/gello-hil-actor/bin/python   # grpcio 1.74.0
```

> ## 🛑 시스템 `python3-grpcio 1.30.2`는 손상돼 있다
> **채널을 하나 만들기만 해도 에러도 로그도 없이 CPU 100%로 영구 정지한다.**
> 타임아웃도 안 걸린다 — 파이썬 레벨에 도달하지 못한다.
>
> 재현 (2026-07-27 실측, 로봇 불필요):
>
> ```bash
> cat > /tmp/g.py <<'PY'
> import grpc
> print("grpc", grpc.__version__, flush=True)
> ch = grpc.insecure_channel("127.0.0.1:59999")
> try:
>     grpc.channel_ready_future(ch).result(timeout=1.0)
> except Exception as e:
>     print("expected timeout:", type(e).__name__, flush=True)
> print("DONE", flush=True)
> PY
>
> timeout 25 python3 /tmp/g.py ; echo "rc=$?"                     # -> grpc 1.30.2 / rc=124 (hang)
> timeout 25 $ACTOR_PY /tmp/g.py ; echo "rc=$?"                   # -> DONE / rc=0
> ```
>
> 멈춘 프로세스를 관찰하면 **단일 스레드가 `R` 상태로 100% CPU**를 태우고 있다
> (`ps -L -o pid,tid,pcpu,stat,wchan -p <pid>`; `wchan`이 비어 있다 = 커널 대기가 아니라 스핀).
>
> **왜 실운용 Kanu 왕복은 멀쩡했나:** 그건 venv를 썼기 때문이다. 손으로 `python3`를 친
> 순간에만 걸린다. 그래서 "어제는 됐는데" 라는 증상으로 나타난다.
>
> 원인으로 보고된 것은 `cygrpc.so`의 `__wrap_memcpy`가 무한 루프로 컴파일된 것이다.
> ⚠️ 다만 이 `.so`에는 해당 심볼도, `endbr64; jmp $` 바이트 패턴도 **찾을 수 없었다**
> (`readelf -sW`, 바이트 스캔 모두 음성). 정확한 기계어 원인은 **미확인**이고,
> **행동 자체는 위 명령으로 100% 재현된다.** 판단 근거로는 재현 결과만 쓸 것.

##### 어느 인터프리터를 쓰는지는 "gRPC를 import하는가"로 정해진다 (2026-07-30 보강)

| 실행할 것 | 인터프리터 | 왜 |
|---|---|---|
| actor (`scripts/run_remote_rlpd_actor.py`, `run_hil_actor.sh`) | **`/home/laptop3/venvs/gello-hil-actor/bin/python`** | gRPC 채널을 연다 |
| receive server / learner (Kanu 쪽 포함) | **venv python** | 같은 이유 |
| `serl_ur_infra` pytest | **venv python** | gRPC 테스트 4파일이 포함된다 (§4.2) |
| **`serl_ur_infra/tests/run_real_hil.py`** | **시스템 `python3`가 맞다** | 이 러너는 **gRPC를 import하지 않는다**(정책이 zero 고정, learner·서버 불필요). 대신 **rclpy·cv2가 필요**하고 그건 시스템 python3에 있다 |
| `ur_gello_bringup` pytest | 시스템 `python3` | ROS `launch` 모듈이 필요 (§4.1) |

즉 **"무조건 venv"가 아니다.** 규칙은 "gRPC를 타면 venv, ROS/rclpy를 타면 시스템 python3"이고,
`run_real_hil.py`는 후자다.

#### (2) `PYTHONPATH`는 **이어붙인다**. 덮어쓰지 않는다

```bash
# ✅ 올바름
PYTHONPATH="$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH}" ...

# ❌ ModuleNotFoundError: No module named 'ur_gello_bringup'
PYTHONPATH="$WT/serl_ur_infra" ...
```

`ur_gello_bringup`(그리고 `ur_kin`)은 apt 패키지가 아니라 **ROS 오버레이**에서 온다:

```bash
ls $WT/ros2_ur_ws/install/ur_gello_bringup/lib/python3.10/site-packages   # 존재 확인
```

`source install/setup.bash`가 이 경로를 `PYTHONPATH`에 넣어 주는데, 덮어쓰면 사라진다.

> ### 🛑 2026-07-30 실기에서 이걸로 넘어졌다 — 덮어쓰면 **rclpy가 사라진다**
> pytest 명령의 `PYTHONPATH=...` 부분을 러너에 그대로 옮겨 붙인 것이 원인이다:
>
> ```bash
> # ❌ RuntimeError: rclpy not available — use fake_env=True
> PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
>   python serl_ur_infra/tests/run_real_hil.py
> ```
>
> `source /opt/ros/humble/setup.bash`가 넣어 준 경로가 날아가서 **rclpy 자체가 안 보인다.**
> 메시지가 `fake_env=True`를 권하지만 그건 **오진 유도**다 — 환경이 잘못된 것이고, fake_env로
> 바꾸면 로봇을 전혀 안 건드리는 다른 실험이 된다.
>
> **ROS 노드를 쓰는 러너(`run_real_hil.py`, `run_rviz_*.py`)는 `source` 두 줄만 하고
> `PYTHONPATH`를 손대지 않는다.** 꼭 추가해야 하면 반드시 끝에 `:$PYTHONPATH`를 붙인다:
>
> ```bash
> set +u; source /opt/ros/humble/setup.bash; source $WT/ros2_ur_ws/install/setup.bash; set -u
> cd $WT/serl_ur_infra
> python3 tests/run_real_hil.py                       # ✅ PYTHONPATH 미변경
> PYTHONPATH="$WT/serl_ur_infra:$PYTHONPATH" python3 tests/run_real_hil.py   # ✅ 이어붙임
> ```
>
> `run_hil_actor.sh` preflight **[4]단계**가 검사하는 항목이 정확히 이것이지만
> (**이어붙였는지 / 덮어썼는지**), 명령을 대화형으로 조립할 때는 아무도 안 잡아 준다.

#### (3) 그런데 pytest에는 ROS `PYTHONPATH`를 붙이면 **안 된다**

→ §4.2. **이 둘은 서로 반대다** — 헷갈리면 §4의 명령을 그대로 복사한다.

| | ROS `PYTHONPATH` | 이유 |
|---|---|---|
| `run_real_hil.py` 등 **ROS 러너** | **필요** (`source`가 넣어 준 것을 지우지 말 것) | rclpy·`ur_kin` |
| `serl_ur_infra` **pytest** | **금지** (`env -u PYTHONPATH` 후 재구성) | ROS pytest 플러그인이 컬렉션을 무너뜨려 `1 skipped`로 떨어진다 |

**그래서 §4.2의 정본 명령이 `PYTHONPATH`를 덮어쓰는 것은 실수가 아니라 의도다.**
그 명령줄을 러너에 재사용하는 것이 실수다.

### 3.5 🛑 컨트롤러 요구가 러너마다 다르다 — 틀리면 **조용히** 아무 일도 안 일어난다 (2026-07-30)

| 무엇을 띄우나 | active 컨트롤러 | 근거 |
|---|---|---|
| `ros2_ur_ws/run_hil_hardware.sh` (운영 3-CLI) | **STJC** (`scaled_joint_trajectory_controller`) | `run_hil_hardware.sh`의 `initial_joint_controller:=` |
| `serl_ur_infra/tests/run_real_hil.py` (개입 검증 러너) | **FPC** (`forward_position_controller`), **컨트롤러 전환 로직이 없다** | `tests/run_real_hil.py` 상단 주석 "T1 — 실기 드라이버 + forward_position_controller" |

**즉 3-CLI로 하드웨어를 띄운 뒤 `run_real_hil.py`를 그대로 돌리면 명령이 아무데도 안 간다.**
증상은 예외가 아니라 `!! 구독자가 없다 — forward_position_controller가 active가 아닐 수 있다`
**경고 한 줄**뿐이고 스텝은 계속 돈다 — 조용한 실패다. `run_real_hil.py`용 T1은 브리지 없이 드라이버를 직접 띄운다:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur7e robot_ip:=192.168.10.11 \
    initial_joint_controller:=forward_position_controller \
    headless_mode:=true launch_rviz:=false
# headless_mode:=true 는 펜던트가 REMOTE 모드여야 한다
ros2 control list_controllers | grep forward_position_controller   # -> active 확인
```

절차 전문·안전 설계는 [`04_HIL_INTERVENTION.md`](04_HIL_INTERVENTION.md) §4.5·§9와
`tests/run_real_hil.py` 상단 주석에 있다 — 여기서는 **함정만** 적는다.
(FPC에 퍼블리셔가 둘이 되는 경우의 위험은 `04` §4.)

---

## 4. 오프라인 테스트 (로봇 불필요, 위험 0)

> 두 스위트의 실행 조건이 **서로 반대**다. 표부터 본다.
>
> | | 인터프리터 | ROS `PYTHONPATH` |
> |---|---|---|
> | `ur_gello_bringup` (§4.1) | 시스템 `python3` | **필요** (`launch` 모듈을 쓴다) |
> | `serl_ur_infra` (§4.2) | **venv python** (gRPC 테스트 때문) | **금지** (수집이 죽는다) |

### 4.1 `ur_gello_bringup` — 436 tests

```bash
source /opt/ros/humble/setup.bash
cd $WT/ros2_ur_ws/src/ur_gello_bringup
python3 -m pytest test/ -q -p no:anyio
```

- `-p no:anyio`가 **필수**다. 이 머신의 시스템 pytest 6.2.5 + 사용자 anyio 플러그인 조합이
  `ModuleNotFoundError: No module named '_pytest.scope'`로 죽는다.
- **ROS 환경이 필요하다.** `env -u PYTHONPATH`로 돌리면
  `ImportError: cannot import name 'LaunchContext' from 'launch'`로 컬렉션이 중단된다
  (`test/test_launch_derivation.py`).
- 📌 실측: **`436 passed in 7.09s`** (2026-07-29, 통합 checkout).
  2026-07-27 워크트리 실측은 `436 passed in 10.14s`였다 — 개수는 같다.
- 이 스위트는 실행 중 **타이밍 baseline을 직접 찍는다** (`test/test_ur_kin.py`의 (k) 항목,
  `test_k_worst_case_tick_timing`). 즉 타이밍 근거를 요구받으면 이 명령이 곧 산출물이다.

  > ### ⚠️ 이 타이밍 숫자를 고정값으로 인용하지 말 것
  > 테스트는 350개 pose(generic 150 / near-singular 150 / unreachable 50)를 돌려
  > **그 실행에서 가장 느렸던 하나**를 찍는다. 값도, 붙는 pose 이름도 실행마다 바뀐다.
  > 📌 2026-07-29: `worst-case tick = 0.836 ms (on a generic pose; budget 4.0 ms @250Hz)`
  > 📌 2026-07-27: `worst-case tick = 1.314 ms (on a near-singular pose; ...)`
  > **판정 기준은 "예산 4.0 ms 미만"이지 특정 값이 아니다.** 예산을 넘으면 출력에
  > `WARNING: exceeds budget`이 붙는다 (`test_ur_kin.py:445-446`).

### 4.2 `serl_ur_infra` — 579 passed / 11 skipped (2026-07-30 `4197f5b`, actor venv)

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH \
PYTHONPATH="$WT/ros2_ur_ws/install/ur_gello_bringup/lib/python3.10/site-packages:$WT/third_party/hil-serl/serl_launcher" \
/home/laptop3/venvs/gello-hil-actor/bin/python -m pytest tests -q -p no:anyio
```

📌 실측(2026-07-30, `4197f5b`, `/home/laptop3/venvs/gello-hil-actor/bin/python`):
**`579 passed, 11 skipped in 8.98s`**.

계보 — 옛 문서에 남은 숫자를 현재 기준선으로 인용하지 말 것:

| 값 | 시점 |
|---|---|
| `332 passed, 11 skipped, 1 xfailed` | 2026-07-27 (구 워크트리) |
| `333` | 2026-07-29 오전 (머지 후, xfail 소멸) |
| `337` | `40b99f8` recorder take → learner demo 변환기 |
| `429` | classifier sidecar |
| `497` | 2026-07-30 오전 (`4197f5b` **이전**) |
| **`579`** | **`4197f5b`** — 신규 82 = `test_leader_stream` 28 / `test_governor_dt` 38 / `test_intervention_substeps` 16 |

> ### 🪤 인터프리터를 안 적은 "passed 개수"는 **무의미하다** (2026-07-30 실측)
> 위 579는 **actor venv**(gRPC용, jax 없음, numpy 2.2.6) 기준이다. 같은 명령을
> `/home/laptop3/venvs/hilserl/bin/python`(jax 0.5.3, numpy 1.26.4)으로 돌리면
> passed가 늘고 **skipped가 11 → 4로 줄어든다** (jax가 있어 §4.3의 skip들이 실제로
> 실행된다).
>
> **판정 기준선은 actor venv 값이다.** 다른 인터프리터의 수와 비교해 회귀를 판단하지 말 것.
>
> ### ✅ 해소된 함정 — 허용범위가 너무 타이트했다 (2026-07-30)
>
> 한때 hilserl 쪽에서 `1 failed`가 났다:
> `test_governor_dt.py::test_env_step_surfaces_governed_in_info`,
> `0.8485281248` vs `0.8485281374 ± 8.5e-10` (3회 재현, actor venv에서는 통과).
>
> **numpy 승격 차이가 원인이 아니었다.** 허용범위가 애초에 틀렸다 — 액션 dtype이
> 계약상 `float32`(`run_contract`의 `"action": {"dtype": "float32"}`)라
> `xi = action * ACTION_SCALE`에 **~1e-7 상대오차**가 실리는데 단언이 `rel=1e-9`였다.
> 산술이 낼 수 있는 정밀도보다 타이트했고, actor venv에서 통과한 것이 우연이다.
> `rel=1e-6`으로 고쳐 **두 인터프리터 모두 통과**한다.
>
> 🪤 **일반 교훈**: `env.step`을 통과하는 값에 `rel < 1e-7`을 걸지 마라. 액션은
> float32다. 컨트롤러를 직접 호출하는(float64 `xi`) 단언은 1e-9도 안전하다 —
> 같은 파일에 두 종류가 섞여 있고, 그 구분이 이유다.

**개수를 기대값으로 하드코딩하지 말고, 아래 세 함정 때문에 개수가 줄지 않았는지만 본다.**

> ### 🪤 그리고 총계에 **아예 안 나타나는** 파일이 하나 있다
> `tests/test_env_fake_backend.py`는 `def test_*`가 없어 pytest에서 **0개 수집**된다
> (skip도 error도 남기지 않는다). 10 Hz 스텝 페이싱·업샘플러 틱 예산·`ACTION_SCALE`↔governor
> 정합 단언이 거기 있는데 **실행되지 않는다** → [`08_OPEN_GAPS.md`](08_OPEN_GAPS.md) **G25**.
> 새 테스트 파일을 넣을 때 `--collect-only`로 수집 여부를 확인하는 습관이 이래서 필요하다.

동등한 대안(플러그인 자동로딩까지 끄고 리포 루트에서 실행):

```bash
cd $WT
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q -p no:cacheprovider serl_ur_infra/tests
```

📌 2026-07-30: 같은 **`579 passed, 11 skipped in 8.98s`** (2026-07-29에는 같은 형태로 `333`이었다).
이 형태가 `HANDOFF_NEXT_SESSION_KO.md` 「테스트 — 이 명령 그대로」 절의 정본 명령과
같은 것이다 (줄번호로 찾지 말 것 — 밀린다).
(`OVERLAY`를 하드코딩하지 않아 오버레이 레이아웃이 바뀌어도 버틴다.
`env -u PYTHONPATH`가 **`OVERLAY` 계산 뒤에** 와야 하는 점만 주의. `PYTHONPATH`를 이렇게
**덮어쓰는 것은 pytest에서만 옳다** — 러너에 재사용하면 rclpy가 사라진다, §3.4(2).)

세 부분이 전부 필요하고, 각각 빠뜨렸을 때의 증상이 다르다:

| 빠뜨린 것 | 증상 | 실측 |
|---|---|---|
| `env -u PYTHONPATH` (= ROS 경로가 남음) | **`collected 0 items / 1 skipped`** → `no tests collected`. **에러가 없어서 통과한 것처럼 보인다** | `PYTHONPATH=/opt/ros/humble/... pytest tests` → 0 collected |
| venv python 대신 `python3` | 4개 파일이 **영원히 멈춘다**(각각 무한 hang): `test_actor_grpc_transport.py`, `test_actor_identity_pinning.py`, `test_actor_smoke.py`, `test_rlpd_receive_smoke.py`. 나머지는 정상 통과하므로 "느린 테스트"로 착각하기 쉽다 | 45 s 타임아웃 4/4 발생. venv에서는 각각 21/11/2/1 = **35 passed** |
| `serl_launcher` 경로 | hang도 실패도 없이 **조용히 skip**된다: `test_cube_in_cup_config`, `test_frame_wrappers`, `test_reset_branch_cut` 등 | 📌 2026-07-29 실측: `333 passed, 11 skipped` → **`300 passed, 13 skipped`**. *(이 낙차는 333 시절 값이다 — 579 기준의 낙차는 재측정 안 했다. 판정은 "기준선보다 내려갔는가"로 한다.)* |

> ### 🪤 skip 사유가 거짓말을 한다
> serl_launcher가 없을 때 뜨는 skip 메시지는
> `third_party/hil-serl submodule is not checked out`이다. **서브모듈은 초기화돼 있는데도
> 그렇게 뜬다** — 실제 원인은 그 경로가 `PYTHONPATH`에 없는 것이다.
> 그러므로 **"초록색인가"가 아니라 "passed 개수가 유지되는가"로 판정한다.**
> 새로 만든 worktree에서 서브모듈이 정말 미초기화면 개수가 더 떨어진다.

> ### 🔧 정정: "`--ignore=tests/test_learner_policy_checkpoint.py`가 필수"는 틀렸다
> 이전 판은 `test_learner_policy_checkpoint.py:15`의 모듈 레벨 `pytest.importorskip("jax")`가
> 컬렉션을 중단시킨다고 적었다. 실제 범인은 **ROS의 pytest 플러그인**이다.
> ROS `PYTHONPATH`가 있으면 `launch_testing` / `launch_testing_ros` / `ament_*` 플러그인이
> 로드되고, 그 상태에서 모듈 레벨 `importorskip`이 걸리는 **첫 파일**(현재는 알파벳순으로
> `test_cube_in_cup_config.py`)이 세션 전체 컬렉션을 무너뜨린다.
>
> 확인:
>
> ```bash
> cd $WT/serl_ur_infra
> PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages python3 -m pytest tests -p no:anyio --co -rs
> #   collected 0 items / 1 skipped
> #   SKIPPED [1] tests/test_cube_in_cup_config.py:14: third_party/hil-serl submodule is not checked out
> ```
>
> `--ignore=`로는 못 막는다 (다음 파일이 같은 자리를 물려받는다). ROS `PYTHONPATH`를 빼는 것이
> 유일한 해법이다.

### 4.3 JAX가 필요한 테스트

**actor venv**(`venvs/gello-hil-actor`)에는 jax/flax가 없다. 아래는 그 인터프리터에서
`SKIPPED`로 나오는 것이 **정상**이고, 학습 자체는 Kanu에서 돌린다.

> 🔧 **정정 (2026-07-30):** 이전 판은 "**로봇 랩톱에** jax/flax가 없다"고 적었다. 랩톱 기준으로는
> 낡았다 — `/home/laptop3/venvs/hilserl/bin/python`에는 **jax 0.5.3이 있다**. 없는 것은
> **actor venv**다. 그 venv로 돌리면 아래 skip들이 실제로 실행되고 총계가 달라진다(§4.2의 함정 박스).

```
test_learner_composition.py            could not import 'jax'
test_learner_policy_checkpoint.py      could not import 'jax'
test_reward_classifier_local_io.py     could not import 'flax.io'
test_rlpd_receive_server.py [4개]      could not import 'jax'
```

추가로 아래는 **환경변수로 옵트인**하는 무거운 테스트다 (skip이 정상):

```
RUN_HIL_SERL_ACTUAL_CHECKPOINT=1     test_actual_agent_checkpoint_integration.py
RUN_HIL_SERL_FAKE_E2E=1              test_actual_fake_data_e2e_learning.py
RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1  test_actual_frozen_trunk_feature_agent.py
```

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
| **RL 데드맨을 놓으면 팔이 선다** | **아니다.** 개입만 해제되고 **정책이 즉시 팔을 계속 움직인다.** `GelloIntervention.action()`은 미개입 시 정책 액션을 그대로 통과시킨다 | `serl_ur_infra/ur_env/envs/wrappers.py:330-343` |
| **HIL GUI DISENGAGE = 정지** | **아니다.** fresh DISENGAGE heartbeat는 명시적으로 **정책 복귀를 승인**한다. GUI는 로봇/브리지에 관절 명령을 직접 보내지는 않지만 deadman-only도 아니다. `/hil/actor_status`를 표시하고 `/hil/scene_ready`, `/hil/set_auto_success`, `/hil/manual_success` 서비스로 HOME/장면 승인, MANUAL/AUTO 선택, one-shot 성공 표시를 전달한다 | `gello_hil_gui_node.py`, `operator_session.py::RosOperatorSession` |
| **GUI가 죽는 것은 E-STOP과 같다** | **아니다.** 첫 heartbeat 수신 뒤 0.5 s 단절은 이제 `DeadmanHeartbeatStaleError`로 actor를 fail-stop하고 새 정책 액션을 보내지 않아 FPC가 마지막 명령을 홀드한다. 하지만 이는 소프트웨어 정지일 뿐 전원 차단·브레이크 체결이 아니다 | `RosTopicDeadman.is_engaged()` (`STALE_S = 0.5`), actor CLI의 `finally` |
| **ESC를 누르면 즉시 정지하거나 HOME으로 간다** | **둘 다 아니다.** ESC는 다음 step 경계에서 로컬 terminal을 제안한다. 서버 ACK 뒤 armed topic actor는 terminal 자세를 홀드한 채 `WAIT_HOME_APPROVAL`에 들어가며, GUI의 **APPROVE HOME**을 눌러야만 `go_to_reset()`으로 HOME 이동한다. pynput 전역 리스너라 터미널 포커스는 필요 없다 | `ur7e_env.py`의 ESC listener, `remote_actor.py`의 terminal ACK → `wait_for_home_approval()` → approved reset, `operator_session.py::_on_scene_ready` |
| **펜던트 속도 슬라이더를 0%로 내리면 정지** | **아니다.** 속도 스케일일 뿐이며 명령 스트림은 계속 흐른다. 슬라이더를 올리는 순간 밀린 명령이 그대로 실행된다. 정지 수단으로 쓰지 말 것 | (UR PolyScope 동작. 리포 근거 없음 — 조작 원칙) |
| **`DRY_RUN=True`니까 안전하다** | 조건부로 맞다. `DRY_RUN`은 `URRosBackend(dry_run=...)`로 전달되어 명령 발행을 막지만, **`run_rviz_hil.py`는 `DRY_RUN=False`를 일부러 박아 놨다**(`tests/run_rviz_hil.py:55`). 또 `run_real_hil.py --arm`과 actor의 `--arm`은 **CLI로 `DRY_RUN`을 끈다.** 그 스크립트들을 무심코 실기에 겨누지 말 것 | `config.py:232`, `cube_in_cup.py:233`, `run_rviz_hil.py:34`, `:55` |
| **`pos_scale:=0.0`이면 팔이 안 움직인다** | **아니다.** TCP 위치만 고정되고 회전 채널은 100% 살아 있어 팔이 크게 스윙한다 | `03_EEF_MODE.md` §2, `config/ur7e_gello_eef.yaml:66-73` |

### 6.1.1 🔧 새로 확인된 위험: 리셋이 손목을 **한 바퀴 돌릴 수 있었다** (H3, 수정됨)

`forward_position_controller`에 보내는 관절값은 **각도가 아니라 그냥 숫자**다. 컨트롤러는
raw 관절 공간에서 선형 보간하며 2π를 모른다. `cube_in_cup`의 `RESET_JOINTS`는 정확히
±π 경계 위에 있다 (`wrist_3 = -3.1331`, `shoulder_pan = 3.1382`).

2026-07-27 실측 사례:

| | 값 |
|---|---|
| 파킹된 실제 자세 `wrist_3` | **+3.1795** |
| 목표 `wrist_3` | **−3.1331** |
| 물리적 차이 | **0.029 rad** |
| 예전 코드가 계산한 차이 | **6.3126 rad** |

두 가지가 동시에 터졌다.

1. `go_to_reset()`의 거리 가드가 6.31 rad를 보고 **배선 고장처럼 보이는 에러**를 냈다.
2. 가드를 통과시켰다면 명령도 **먼 길(한 바퀴)로** 나갔다 — 약 10초의 맹목 슬루와
   **Robotiq 2F-85 tool-comm 케이블이 손목에 한 바퀴 감김**. 그때 이게 실제로 일어나지
   않은 유일한 이유는 `DRY_RUN=True`였다는 것뿐이다.

**수정됨** (commit `d49d0f6`): `go_to_reset()`이 먼저 `ur_kin.wrapped_nearest(target, q)`로
목표를 **팔의 현재 회전수**로 옮긴 뒤 거리 가드·명령·도착 판정을 전부 그 값으로 한다
(`ur7e_env.py:520`의 `go_to_reset`, wrap은 `:567-591`).
회귀 테스트 `tests/test_reset_branch_cut.py` (📌 2026-07-29 재실행 **10 passed**).

> ⚠️ **팔꿈치(index 2)는 일부러 감싸지 않는다.** 팔꿈치 가동범위가 ±π라 "더 짧은" 래핑
> 목표가 도달 불가 영역에 놓이기 때문이다. 그래서 branch-safe 거리는 순진한 원형 거리
> 2.755가 아니라 **3.5281**이다.
>
> **하드웨어에서는 아직 확인되지 않았다.** 2026-07-28 실기 세션이 팔을 움직였지만
> `run_real_hil.py --reset-mode startup`은 **기동 시점의 실제 관절을 그대로 `RESET_JOINTS`로
> 잡으므로** ±π 경계를 건드리지 않는다 — 이 경로는 그때 실행되지 않았다.
> `cube_in_cup`의 `RESET_JOINTS`로 첫 실기 리셋을 할 때는 `DRY_RUN`으로 로그만 보고,
> 명령된 `wrist_3` 값이 현재 값 근처인지 눈으로 확인한 뒤에 arm한다.

### 6.2 protective stop 복구

```bash
source $WT/ros2_ur_ws/remote_helpers.sh
ur_unlock     # /dashboard_client/unlock_protective_stop  (원인을 먼저 제거한 뒤에!)
ur_resend     # Method B 복구 (또는 Method A면 ur_play)
```

**순서:** ① 원인 제거(장애물 치우기, 속도 낮추기) → ② `ur_unlock` → ③ `ur_resend`/`ur_play`.
원인을 안 없애고 unlock하면 바로 다시 걸린다.
