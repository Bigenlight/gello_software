# 04 — HIL 개입 (데드맨 · 앵커 · 좌표계 · 메타데이터)

**상태: 실기에서 한 번 검증됐다 (2026-07-28, `run_real_hil.py` 경로에 한함).**
mock + RViz 절차는 **여전히 미검증**이고 정본은 `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`다.
이 문서는 **판정 기준과 함정**에 집중한다.

> ### 🔧 상태 갱신 (2026-07-29) — "개입 루프는 실기 미검증"은 더 이상 맞지 않는다
> 2026-07-28 실기 세션에서 `serl_ur_infra/tests/run_real_hil.py`로 **팔을 실제로 구동**했다.
> 📌 그때의 기록:
>
> | 항목 | 값 |
> |---|---|
> | 실행 | `run_real_hil.py --arm --scale 0.25` |
> | 스텝 | 100 (그중 개입 64) |
> | `held` | **0** |
> | anchor-latch 위반 | 0 |
> | gain-latch 위반 | 0 |
> | 저장 액션 == 실행 액션 | **1.000** (§2의 불변식) |
> | held-rate | 0 % |
> | frame-map | **단위행렬** — 포화 표본 제외 잔차 0.093(기준 0.15), alpha 0.984. 자유 9-파라미터 행렬은 전체 표본에서 0.165로 단위행렬(0.203)보다 나을 게 없었다 → **회전 없음 확정** |
>
> 이것이 §5의 좌표계 3×3을 **수치로** 대체한다 — mock RViz에서 다시 할 필요는 없지만,
> mock 루프 자체(§3)는 여전히 한 번도 안 돌았다.
>
> **아직 아닌 것:** 그리퍼 개입(§6, 기본 비활성), actor entrypoint 경로(`09`),
> `cube_in_cup` config로의 실기 구동(`08` G1 — 그때 쓴 것은 `DefaultUR7eEnvConfig`라
> 워크스페이스 박스가 **꺼져 있었다**).

> ### 2026-07-27 세션이 이 문서에 미친 영향
> - **리셋 경로가 바뀌었다**: `go_to_reset()`이 branch-cut을 적용하게 됐고
>   (`00` §6.1.1), `cube_in_cup`의 `RESET_MAX_DIST_RAD`가 `0.5` → **`0.9`**로 올랐다
>   (`cube_in_cup.py:103`; 0.5는 23테이크 중 16개의 정상 종료 자세를 거부했다).
>   §4.5의 "리셋이 가드에 걸려 멈추는 것이 의도된 안전 실패"는 여전히 맞지만,
>   **이제 정상 에피소드는 통과한다.**
> - 개입 액션이 저장되는 프레임 관련 테스트가 크게 강화됐다
>   (`test_frame_wrappers.py`, commit `ee3240e`, 📌 2026-07-29 재실행 **20 passed**) —
>   이전 테스트들은 identity 자세 fixture라 "회전했는가" 단정이 전부 자명하게
>   통과하고 있었다.

```bash
export WT=/home/laptop3/gello_software     # 2026-07-29 머지(3f199d4) 이후 통합 checkout이 정본
```

---

## 1. 데드맨 — 두 가지 소스

`GelloIntervention`은 `DeadmanSource` 인터페이스로 engage 신호와 gain을 받는다
(`serl_ur_infra/ur_env/envs/wrappers.py:71-83`).

| | `SpacebarDeadman` | `RosTopicDeadman` (**권장 · 러너 기본값**) |
|---|---|---|
| 코드 | `wrappers.py:86` | `wrappers.py:130` |
| 신호원 | pynput 전역 키보드 리스너 | ROS 토픽 `/hil/deadman` |
| gain | **1.0 고정** (`wrappers.py:126-127`) | 슬라이더 0.10 ~ 1.00 |
| 워치독 | **없음** | 첫 수신 뒤 0.5 s: 액터 fail-stop (`STALE_S`) |
| mock 러너 | `python3 tests/run_rviz_hil.py` | `python3 tests/run_rviz_hil.py --deadman topic` |
| 실기 러너 | `--deadman spacebar` (비권장) | **기본값** (`run_real_hil.py:652`) |
| actor | `--deadman spacebar` (비권장) | **기본값** (`run_remote_rlpd_actor.py:52-62`) |

> ### 🔧 정정 (2026-07-29): "스페이스바가 기본값"은 더 이상 사실이 아니다
> `GelloIntervention(env, deadman=None)`을 **직접** 만들면 여전히 `SpacebarDeadman()`이
> 붙는다 (`wrappers.py:189`). 그러나 **실기에서 쓰는 두 entrypoint는 모두 `--deadman`을
> 명시적으로 전달하고 기본값이 `topic`이다.** 즉 `08_OPEN_GAPS.md` G12는 닫혔다.
> 남은 위험은 "새로 쓰는 코드가 `deadman=`을 안 넘기면 조용히 스페이스바로 떨어진다"뿐이다.

### 1.1 🛑 스페이스바 데드맨이 위험한 이유

권장하지 않는다. 네 가지 이유가 있고, 셋은 코드 근거가 있다.

1. **X11 키 오토리핏이 홀드를 "뗐다 눌렀다"로 만든다.**
   → 개입이 깜빡이고, 그때마다 `_disengage()` → `_engage()`가 돌아 **앵커가 재래치된다**
   (`wrappers.py:330-343`). 즉 조작 도중 기준점이 계속 리셋된다.
   (`RVIZ_HIL_TEST_CLI.md`의 "스페이스바 개입이 깜빡임" 항목)
2. **pynput 리스너는 전역이다.** 터미널 포커스와 무관하게 X 세션의 스페이스를 잡는다
   (`wrappers.py:100-120`). 다른 창에서 친 스페이스가 **로봇을 engage시킬 수 있다.**
3. **gain이 1.0으로 고정된다** (`wrappers.py:126-127`). 감도를 낮출 수단이 없다.
4. **워치독이 없다.** 토픽 데드맨은 첫 하트비트를 받은 뒤 0.5 s 끊기면
   `DeadmanHeartbeatStaleError`로 액터를 중단하지만, 스페이스바는 그런 안전망이 없다.

> ✅ 다만 한 가지 fail-safe는 있다: 디스플레이가 없어 pynput이 죽으면
> `except Exception`으로 잡아 **"영원히 engage 안 됨"** 상태로 떨어진다
> (`wrappers.py:113-120`). 조용히 engage된 채로 남지는 않는다.
>
> ⚠️ 같은 pynput 전역 리스너가 **env 본체에도** 하나 더 있다 (`ur7e_env.py:197-208`,
> ESC = 에피소드 종료). 그쪽은 데드맨과 무관하게 항상 살아 있고 데드맨 설정으로 못 끈다
> → `00_SETUP_AND_SAFETY.md` §6.1, `08_OPEN_GAPS.md` G17.

### 1.2 GUI 데드맨 (권장)

```bash
cd $WT/ros2_ur_ws && ./run_hil_gui.sh
```

**동결 계약 (env와 GUI가 공유. 변경 금지):**

| | 값 | 근거 |
|---|---|---|
| 토픽 | `/hil/deadman` | `gello_hil_gui_node.py:77`, `wrappers.py:135-137` |
| 타입 | `std_msgs/Float32MultiArray` | 동일 |
| `data` | `[engaged, gain]`, `engaged ∈ {0.0, 1.0}`, `gain ∈ [0.10, 1.00]` | `gello_hil_gui_node.py:79-80`, `:118-136` |
| 주기 | **20 Hz 하트비트** (변화 시가 아니라 **상시** 발행) | `gello_hil_gui_node.py:80-81` |
| QoS | 기본 reliable, depth 10 | `wrappers.py:120` |
| 스테일 | 첫 수신 뒤 0.5 s 넘으면 `DeadmanHeartbeatStaleError` | `wrappers.py`의 `RosTopicDeadman.is_engaged()` |

- ENGAGE(OFF→ON)는 **두 번 클릭 확인**, DISENGAGE(ON→OFF)는 한 번 클릭
  (`gello_hil_gui_node.py:42-44`).
- GUI는 **로봇도 GELLO도 브리지도 건드리지 않는다.** 퍼블리셔 하나만 소유한다
  (`gello_hil_gui_node.py:6-10`, `:29-31`).

> ### ⛔ 명시적 DISENGAGE와 하트비트 단절은 다르다
> 살아 있는 GUI가 `engaged=0`을 보내면 **정책이 즉시 이어받는다**. 그러나 첫 메시지
> 수신 뒤 0.5 s 동안 하트비트가 없으면 정책 복귀로 해석하지 않는다.
> `DeadmanHeartbeatStaleError`가 하위 `env.step()` 전에 발생하고 액터가 종료되며,
> 해당 틱의 정책 액션은 FPC로 전달되지 않는다. 엔트리포인트의 `finally`가 네트워크와
> env를 닫는 동안 FPC에는 마지막 명령만 남는다. GUI가 처음부터 없으면 별도 15 s
> 시작 가드가 액터 구동을 거부한다 (`CubeInCupConfig._resolve_deadman()`).

---

## 2. 앵커와 gain 래치

engage 순간(`_engage()`, `wrappers.py:242-249`)에 세 가지가 **한 번에 래치**된다:

```python
self.T_g_anchor = self._leader_T(q_lead)     # 리더 TCP (현재는 flange 기준)
self.T_r_anchor = self._robot_T_cmd()        # 로봇 '명령' TCP  (측정값 아님)
self._gain      = self.expert.gain()         # 감도 — 여기서 한 번만 읽는다
self._anchored  = True
```

핵심 성질:

| 성질 | 의미 | 근거 |
|---|---|---|
| **gain은 engage 에지에서만 읽는다** | 스트로크 도중 슬라이더를 움직여도 진행 중인 개입이 갑자기 재스케일되지 않는다 | `wrappers.py:242-249` |
| **gain은 병진에만 곱한다** | 회전은 항상 1:1 | `wrappers.py:256-299` |
| **앵커는 `controller.tcp_cmd()`(명령값) 기준** | 관측(`tcp_pose`)과 비교하지 않으므로 `TCP_POSE_SOURCE`가 driver든 fk든 앵커 수식은 자기일관적이다 | `wrappers.py:239-240` → `policy_delta_controller.py:229`; `ur7e_env.py:644` 주석 |
| **disengage 시 앵커 폐기** | 다음 engage는 완전히 새 기준 | `wrappers.py:251-254` |
| **리더가 0.3 s 이상 낡으면 개입 거부** | `LEADER_STALE_S = 0.3` | `wrappers.py:218`, `:333` |
| **anti-windup은 비례(norm) 클램프** | 축별 `np.clip`이 아니다. 대각선 이동의 **방향**이 왜곡되지 않는다 | `wrappers.py:256-299` |

> ### 저장 액션 불변식 — 📌 실기에서 1.000으로 확인됨 (2026-07-28)
> 개입 액션은 `÷ACTION_SCALE → clip`으로 만들어져 **실행값과 저장값이 같다.**
> gain은 그 나눗셈 **앞**에 적용되므로 불변식을 깨지 않는다 (`wrappers.py:256-299`).
> 단 `ACTION_SCALE * HZ`가 거버너 캡을 넘으면 **실행만 잘리고 저장은 안 잘려** 불변식이
> 깨진다. env 기동 시 위반하면 WARNING을 찍는다 (`ur7e_env.py:96-109`).
>
> 기본 config는 이 불변식을 만족한다: `ACTION_SCALE = [0.0125, 0.0625, 1.0]`,
> `HZ = 10` → 0.125 m/s · 0.625 rad/s vs `GOVERNOR v_max 0.15 / w_max 0.75`
> = **양축 1.200x 헤드룸** (`config.py:73`, `:86-90`).
>
> 🛑 **`ACTION_SCALE`은 learner fingerprint에 들어 있지 않다.** 0.01로 녹화한 데이터를
> 0.0125에서 재개해도 **아무 경고 없이 같은 액션이 25% 더 멀리 간다** → `08` G18.

---

## 3. mock + RViz 개입 루프 (실기 위험 0)

정본: `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`. 워크트리 경로로 옮긴 4터미널:

```bash
# 공통 (각 터미널)
source /opt/ros/humble/setup.bash
source $WT/ros2_ur_ws/install/setup.bash
```

```bash
# T1 — mock UR7e + RViz
cd $WT/ros2_ur_ws && ./run_mock_rviz.sh
```
> **직접 `ros2 launch`로 띄우지 말 것.** 이 랩톱의 Humble 드라이버는 Jazzy 이름
> `use_mock_hardware`를 **조용히 무시**하고 실제 로봇에 접속하려 한다.
> `run_mock_rviz.sh`가 `use_fake_hardware:=true`를 박아 넣는다.

```bash
# T2 — 가짜 카메라 (실행 파일명은 복수형, _node 없음)
ros2 run gello_policy fake_diffusion_observations
```
> T2가 없으면 `reset()`이 카메라 디코드에서 죽는다.

```bash
# T3 — 실제 GELLO 리더
GELLO_REPO_ROOT=$WT \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $WT/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

```bash
# T4 — HIL 러너
cd $WT/serl_ur_infra
python3 tests/run_rviz_hil.py --deadman topic       # GUI 개입 (권장)
```

```bash
# T5 — GUI
cd $WT/ros2_ur_ws && ./run_hil_gui.sh
```

기동 로그가 `joint_states ok → leader ok → cameras ok → OPERATOR INSTRUCTIONS` 순이면 정상.

유용한 플래그: `--policy {zero,scripted,random}` (기본 `zero` = 개입 안 할 때 팔 정지),
`--episodes N`(기본 3), `--max-steps N`, `--leader-timeout S`, `--deadman {topic,spacebar}`
(`run_rviz_hil.py:150-181`).

> ### ⚠️ `run_rviz_hil.py`를 실기에 겨누지 말 것
> `DRY_RUN = False`가 **일부러** 박혀 있고 (`run_rviz_hil.py:55`),
> mock 전용 완화값들이 들어 있다: `RESET_MAX_DIST_RAD = 7.0` (`:68`),
> `ACTION_SCALE = [0.03, 0.10, 1.0]` = **0.3 m/s** (`:80`),
> `GOVERNOR = {v_max: 0.36, w_max: 1.2, dq_step_max: 0.12}` (`:81`).
> 이 값들을 실기 config로 복사하면 안 된다 (`:82-85`에 그렇게 적혀 있다).
> 실기 기본값은 그 **1/2.4배**다 (`config.py:73`, `:86-90`).
> 실기에서 개입을 보고 싶으면 §4.5의 `run_real_hil.py`를 쓴다 — 그쪽은 기본이 DRY_RUN이다.

**판정:**

- ENGAGE 후 GELLO를 움직이면 RViz 팔이 따라온다. 놓으면 즉시 정책 복귀.
- **누르기만 하고 GELLO를 안 움직이면 정지가 정상** — 개입은 engage 이후의 **변화량**이다.
- HUD에 `held=True reject=STEP_LIMIT`가 **드물게** 뜨는 것은 특이점 근처 안전 HOLD다(정상).
- 팔이 10 Hz rate-limited chase로 따라오므로 실기 EEF(250 Hz)보다 덜 쫀쫀한 것이 정상.

---

## 4. 🛑 두 퍼블리셔 충돌 — HIL 세션에서 절대 하면 안 되는 것

`/forward_position_controller/commands`에 **두 개가 동시에 발행할 수 있다**:

| 퍼블리셔 | 언제 | 근거 |
|---|---|---|
| `gello_ur_bridge` (텔레옵) | `run_ur7e_gello_real.sh` | `ur7e_gello_real.launch.py:66`, `:779` |
| `URRosBackend` (RL env) | `run_rviz_hil.py` / actor | `ros_backend.py:12`, `:111-113`, `config.py:96` |

**동시에 띄우면 두 스트림이 컨트롤러를 놓고 싸운다.** ROS2는 이걸 막아주지 않는다.

> ### HIL 세션 규칙
> HIL 개입 루프에 필요한 것은 **`gello_publisher`(리더 관절 + 트리거 발행)뿐**이고,
> **`gello_ur_bridge`는 필요 없다.** 그래서 T3에서 `run_ur7e_gello_real.sh`가 아니라
> `ros2 run ur_gello_bringup gello_publisher`를 직접 띄운다.
> `run_ur7e_gello_real.sh`를 HIL 러너와 같이 띄우지 말 것.

확인 명령:

```bash
ros2 topic info /forward_position_controller/commands --verbose | grep -c "Node name"
# Publisher가 2 이상이면 즉시 중단
```

> ✅ 실기 러너 `serl_ur_infra/tests/run_real_hil.py`는 기동 시
> `node.count_publishers()`로 이걸 직접 세고, 자기 것 외에 하나라도 더 있으면
> **`--arm`을 거부한다** (`run_real_hil.py:34-38`, `:439`, `:455`).
>
> ✅ **actor entrypoint에도 같은 가드가 생겼다** (`run_remote_rlpd_actor.py:156-198`,
> commit `c069e79`). `--arm` 시 퍼블리셔가 있으면 거부하고, **아무도 구독하지 않아도**
> 거부한다("컨트롤러가 안 떠 있다"). 래퍼 `run_hil_actor.sh`의 preflight [9]는 `--arm`
> 여부와 무관하게 퍼블리셔가 1개라도 있으면 기동을 거부한다.
>
> ❌ mock 러너(`run_rviz_hil.py`)에는 이 가드가 **없다** — 수동으로 확인할 것.
> 사람이 손으로 두 스택을 띄우는 것도 여전히 막지 못한다.

---

## 4.5 실기 HIL 러너 (`run_real_hil.py`) — **2026-07-28 실기 PASS**

`run_rviz_hil.py`는 **mock 전용**이라("Never point this at a real robot") 실기에서
개입 경로를 검증할 러너가 없었다. `serl_ur_infra/tests/run_real_hil.py`가 그 자리를
채우고, **2026-07-28에 실제로 팔을 움직여 통과했다** (문서 머리말의 기록 표).

설계 원칙 세 가지 (`run_real_hil.py:14-24`):

1. **기본이 안전하다** — `DRY_RUN=True`가 기본이라 로봇 명령을 **한 줄도 발행하지 않는다.**
   팔을 움직이려면 `--arm`을 명시해야 한다 (`:637-641`).
2. **속도는 3층을 함께 올린다** — `--scale`이 `ACTION_SCALE` / `GOVERNOR` / `UPSAMPLER`를
   동시에 곱한다. (한 층만 올리면 다음 층이 조용히 먹어버린다. `build_config()`가
   기본 config에서 비율을 읽어 오므로 헤드룸이 자동 보존된다.)
3. **모든 스텝을 CSV로 남긴다** — 앵커 래치 / gain 래치 / 좌표계 매핑 /
   "저장 액션 == 실행 액션" 불변식을 **사후에 수치로** 검증하기 위해서.

정책은 **항상 zero**다. 학습 정책은 붙이지 않는다.

```bash
# ROS 오버레이가 필요하다 (ur_kin / ur_gello_bringup). 이 두 줄을 먼저.
source /opt/ros/humble/setup.bash
source $WT/ros2_ur_ws/install/setup.bash

cd $WT/serl_ur_infra
python3 tests/run_real_hil.py                       # DRY_RUN — 로봇 명령 0줄
python3 tests/run_real_hil.py --arm --scale 0.25 --yes   # ⚠️ 실제로 움직인다 (07-28 실행값)
```

> 여기서는 **시스템 `python3`가 맞다.** 이 러너는 gRPC를 전혀 import하지 않으므로
> `08_OPEN_GAPS.md` G14의 hang에 걸리지 않는다 (확인: `grep -rn "import grpc"
> ur_env/envs/ tests/run_real_hil.py` → 0 hit). 반대로 `ur_kin`이 필요하므로 ROS
> 오버레이는 **반드시** 있어야 한다 — actor(`09`)와 정확히 반대 조건이다.

전체 플래그 (`--help`로 재확인 가능):

| 플래그 | 기본 | 비고 |
|---|---|---|
| `--arm` | off | `DRY_RUN` 해제. 이중 퍼블리셔가 있으면 **거부**한다 (`:439`) |
| `--scale` | **0.5** | 3층 동시 배율. `1.0` = `config.py` 기본값 |
| `--allow-fast` | off | `--arm` + `--scale > 1.0` 조합의 잠금 해제. 없으면 `ap.error` (`:683-687`) |
| `--deadman {topic,spacebar}` | **topic** | `spacebar`는 워치독 없음 — 실기 비권장 |
| `--episodes` | 1 | |
| `--max-steps` | 300 | 10 Hz에서 30초 |
| `--reset-mode {startup,hold}` | startup | 아래 참조 |
| `--reset-max-dist` | **0.05** | `RESET_MAX_DIST_RAD` 덮어쓰기 |
| `--tcp-source {fk,driver}` | **fk** | 컨트롤러/앵커와 같은 프레임 |
| `--cameras` | off | 켜면 `launch_cameras.sh` 필요 |
| `--gripper` | off | 끄면 `ACTION_SCALE[2]=0.0` |
| `--csv` | `~/gello_hil_logs/real_hil_<타임스탬프>.csv` | |
| `--leader-timeout` | 15.0 s | 리더/데드맨/카메라 대기 |
| `--yes` | off | `--arm` 확인 프롬프트 생략 |

`--scale` 하드 상한은 3.0, `--arm` 소프트 상한은 1.0이다
(`ARM_SCALE_SOFT_MAX`/`SCALE_HARD_MAX`, `:262-263`). 범위를 벗어나면 argparse가 거부한다.
안내된 상승 경로는 **0.25 → 0.5 → 1.0**이다.

리셋 설계가 중요하다: `--reset-mode startup`은 기동 시점의 실제 관절값을 그대로
`RESET_JOINTS`로 잡고 `RESET_MAX_DIST_RAD=0.05`로 조인다 → **"리셋 = 지금 자세 유지"**.
개입으로 팔을 많이 옮긴 뒤 다음 에피소드를 리셋하면 가드에 걸려 **에러로 멈춘다 —
그게 의도된 안전 실패다** (먼 거리를 쓸고 오지 않는다). 그 경우 `--reset-mode hold`
또는 `--episodes 1`.

> ⚠️ 그래서 이 러너는 **`cube_in_cup`의 `RESET_JOINTS`(±π 경계) branch-cut 경로를
> 실행하지 않는다.** `08` G13은 이 세션으로 닫히지 않았다.

> 🛑 **이 러너는 `DefaultUR7eEnvConfig`를 base로 쓴다** (`:296`). 즉
> `ABS_POSE_LIMIT_*`가 0벡터라 **워크스페이스 박스가 꺼진 채로 돈다** — REFUSE-DON'T-CLAMP
> 설계상 경고만 찍고 비활성화된다. 07-28 DRY RUN 300스텝 중 **241스텝(80%)이
> `cube_in_cup` 박스 밖**이었고 최대 이탈은 **73.9 cm**였다. → `08` G1

> **먼저 `--arm` 없이** 돌려서 CSV·좌표계·앵커가 말이 되는지 확인한 다음에 arm한다.
> 아래 §5 좌표계 3×3과 §7 메타데이터 검증은 **DRY_RUN에서 전부 가능하다.**

---

## 5. 좌표계 3×3 검증

> ### ✅ 이 절은 2026-07-28에 **수치로 통과했다** (📌 기록)
> `run_real_hil.py --arm --scale 0.25` CSV를 회귀 분석한 결과 리더→로봇 프레임 맵은
> **단위행렬**이었다: 포화 표본을 제외한 잔차 0.093(기준 0.15), alpha 0.984.
> 자유 9-파라미터 행렬을 맞춰도 전체 표본 잔차가 0.165로, 단위행렬(0.203) 대비
> 유의미한 개선이 없었다 — **회전 성분 없음이 확정됐다.**
>
> 아래 절차는 (a) mock 루프에서 재현하거나, (b) 배선을 바꾼 뒤 재확인할 때 쓴다.
> 새 판정을 할 때도 **눈이 아니라 CSV/토픽 수치로** 한다.

**목적:** 리더 축 → 로봇 base 축 매핑이 identity인지, 축간 크로스토크가 없는지.

**전제:** mock + RViz (§3), 또는 §4.5의 `run_real_hil.py` DRY_RUN + CSV.
개입 중 `info["intervene_action"]`(CSV의 `ia0..ia6`)을 로깅한다.

절차: engage 후 리더를 **한 축씩** 이동시키고, 로봇 명령 델타의 어느 축이 반응하는지 본다.

|  | 로봇 +X | 로봇 +Y | 로봇 +Z |
|---|---|---|---|
| **리더 +X** | **큰 양수** | ~0 | ~0 |
| **리더 +Y** | ~0 | **큰 양수** | ~0 |
| **리더 +Z** | ~0 | ~0 | **큰 양수** |

- 대각선이 양수이고 비대각이 작으면 PASS.
- **PASS 기준을 눈(RViz 그림)으로 잡지 말 것.** `info["intervene_action"]` 또는
  `/forward_position_controller/commands`의 수치로 판정한다.

> ### 🛑 방향이 반대로 "보인다"고 코드에서 뒤집지 말 것
> 오프라인 실측에서 리더 base +X/+Y/+Z → 로봇 base +X/+Y/+Z (identity)이고,
> 기록되는 `intervene_action`도 참 base-frame이다.
> `wrappers.py`에서 X/Y를 부호 반전으로 "고치면" RViz 그림만 맞아 보이고
> **SERL 버퍼에 저장되는 base-frame 액션이 오염된다.**
> 반대로 보이는 실제 원인 두 가지:
> (1) 기본 `ur_description` 카메라가 반대편에서 본다,
> (2) mock 시작 자세 `RESET_JOINTS`가 실기 작업 자세와 반대쪽이다
> (`run_rviz_hil.py:63-67`에서 `shoulder_pan = pi`로 맞춰 놓은 이유).
> — `RVIZ_HIL_TEST_CLI.md`의 "방향/좌표 주의", 사용자 메모 "HIL X/Y flip is camera, not code"

---

## 6. 그리퍼 개입 — **결함이었고, 고쳐졌다 (여전히 하드웨어 미검증)**

> ### ⚠️ 실기 결과와 혼동하지 말 것 (2026-07-29 갱신)
> 실기에서 **PASS한 것**은 (a) 그리퍼 단독 경로(`01`), (b) GELLO 리더 발행과
> 트리거 스팬 0.000~1.000(`02` §2), (c) 2026-07-28 **팔** 개입 루프(§4.5)다.
> **"리더 트리거를 쥐면 개입 중 로봇 그리퍼가 움직인다"는 아직 한 번도 확인되지 않았다** —
> (c)의 러너는 기본값이 그리퍼 **비활성**(`ACTION_SCALE[2]=0.0`)이라 그 채널이 아예 안 돌았다.
> 셋을 합쳐 "그리퍼 개입 검증됨"으로 승격하지 말 것.

### 6.1 원래 결함 (확증됨)

```
gello_publisher  ─ position 길이 6 ─►  /gello/joint_states     (gello_publisher_node.py:190)
                 └ Float32 ──────────►  /gripper/gripper_client/target_gripper_width_percent  (:193-195)

URRosBackend     ── /gello/joint_states 만 구독 ──►  len(arr) == 6
GelloExpert.get_leader():  grip = arr[6] if len(arr) > 6 else None   # → 항상 None
GelloIntervention._expert_gripper(None):  return 0.0                 # → 항상 홀드
```

`GRIP_CLOSE_THR = 0.7` / `GRIP_OPEN_THR = 0.3` 히스테리시스가 **데드 코드**였고,
개입 중 사람이 트리거를 아무리 쥐어도 로봇 그리퍼가 움직이지 않았다.

### 6.2 현재 코드 (커밋 `6a0b127`) — 코드는 고쳐졌고 **하드웨어는 아직**

`URRosBackend`가 트리거 토픽을 **별도로 구독**하고 `merge_gello_state()`가 두 스트림을
7-요소로 합친다 (`ros_backend.py:90-155`, 호출은 `:360-364`). `get_leader()`는 NaN을
`None`으로 매핑해 "지금 트리거 없음"을 한 가지 표현으로 통일한다.

핵심 설계 결정 세 가지 — **판정할 때 이걸 본다**:

| 결정 | 왜 |
|---|---|
| 트리거 부재 = **`NaN`**, `0.0` 아님 | `0.0`은 "완전 열림"이라는 정당한 값이다. 센티널로 쓰면 토픽이 죽을 때마다 **그리퍼를 조용히 연다** (`ros_backend.py:102-109`) |
| 반환 age는 **관절 age만** | 트리거가 없다고 팔 텔레옵까지 막으면 안 된다 (`:110-115`) |
| 트리거 없음 → `_expert_gripper` **0.0 = HOLD**, 래치는 건드리지 않음 | `0.0`은 그리퍼를 움직일 수 없는 유일한 값이라, 죽은 토픽이 **잡은 것을 떨어뜨리지도, 뭔가를 물지도** 못한다. 래치를 재발행하면 이미 없는 신호에서 유래한 grasp를 계속 명령하게 된다 (`wrappers.py:301-328`) |

트리거 스테일 임계는 `GELLO_TRIGGER_STALE_S = 0.3`으로 `LEADER_STALE_S`와 맞춰 뒀다 —
두 스트림이 같은 30 Hz 타이머에서 나오므로, 한쪽만 조용해졌다는 건 트리거 읽기 자체가
실패했다는 뜻이다 (`ros_backend.py:84-87`).

### 6.3 판정 (미실행)

```bash
# 1) 유닛 (rclpy·시리얼 불필요) — 📌 2026-07-29 재실행: 23 passed
cd $WT/serl_ur_infra
env -u PYTHONPATH /home/laptop3/venvs/gello-hil-actor/bin/python \
  -m pytest tests/test_gello_gripper_wiring.py -q -p no:anyio

# 2) 라이브 배선 — 러너 없이 토픽만
ros2 topic hz  /gripper/gripper_client/target_gripper_width_percent   # ~30 Hz
ros2 topic echo /gripper/gripper_client/target_gripper_width_percent  # 0.000 ~ 1.000
```

3) mock/실기 개입 중 판정:

- 트리거를 **0.7 이상**까지 쥔다 → `intervene_action[6]`이 **-1.0**(닫기)
- 트리거를 **0.3 이하**로 푼다 → `+1.0`(열기)
- 그 사이에서는 **직전 값이 유지**된다(래치)
- 트리거 퍼블리셔를 죽인다(`pkill -f gello_publisher`) → 0.3 s 후 `intervene_action[6]`이
  **0.0(HOLD)**가 되고, **그리퍼가 저절로 열리지 않는다** ← 이게 핵심 회귀 판정이다

> ### ⚠️ 아직 하드웨어에서 확인되지 않았다
> `08_OPEN_GAPS.md` G3의 완화책은 실기 판정이 끝날 때까지 유효하다.
> 특히 `run_real_hil.py`는 기본적으로 그리퍼를 **비활성**(`ACTION_SCALE[2]=0.0`)으로 두고
> `--gripper`를 줘야 켜진다 — 그 모드에서는 CSV의 `ia6`가 기록은 되지만 실행되지 않으므로
> **그리퍼 채널에 한해 "저장 == 실행" 불변식이 깨진다.** 의도된 것이고 러너 docstring에
> 명시돼 있다.

---

## 7. 개입 메타데이터 계약

### 7.1 `step()`이 반환하는 `info`

`GelloIntervention.step()` (`wrappers.py:346-360`):

| 키 | 타입 | 의미 | 언제 |
|---|---|---|---|
| `policy_action` | `float32[7]` | **오버라이드 전** 정책 출력 (개입 중에는 counterfactual) | **항상** |
| `intervened` | `int` 0/1 | 사람 액션이 실행되었는가 | **항상** |
| `intervene_action` | `float32[7]` | **실제로 실행된** 사람 액션 (`[-1,1]^7`) | 개입 시**에만** |
| `left` / `right` | `bool` False | spacemouse 버튼 호환용 자리 | 항상 |

- `policy_action`은 wrapped env가 보기 **전에 복사**된다 — 호출자 버퍼나 실행 액션과
  절대 별칭(alias)이 되지 않는다 (`wrappers.py:347-350`).

### 7.2 `build_transition()`의 fail-fast

`serl_ur_infra/ur_env/rlpd_actor.py:34-93`. 다음 세 경우에 **예외를 던진다**:

| 조건 | 예외 |
|---|---|
| `intervened`가 0/1이 아님 | `ValueError: intervened must be 0 or 1` (`:63`) |
| `intervened`와 `intervene_action` 존재 여부가 불일치 | `ValueError: inconsistent intervention metadata` (`:65-68`) |
| 실행 액션과 정책 액션의 shape 불일치 | `ValueError: executed and policy action shapes differ` (`:76-79`) |

생성되는 transition 키 (`:80-93`):

```
observations, actions(=실행된 액션), policy_actions, intervened(uint8),
next_observations, rewards, masks(=1.0-done), dones,  [grasp_penalty]
```

라우팅 규칙 (`route_transition()`, `:96-`): 모든 transition은 replay에,
`intervened == 1`인 것은 **추가로** intervention buffer에 들어간다.

### 7.3 판정 명령

```bash
# 📌 2026-07-29 재실행: 2 passed + 8 passed
cd $WT/serl_ur_infra
env -u PYTHONPATH /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest \
  tests/test_intervention_metadata.py tests/test_rlpd_actor_adapter.py -q -p no:anyio
```

프레임 변환 회귀 (개입 액션이 어느 프레임으로 저장되는가):

```bash
cd $WT/serl_ur_infra
env -u PYTHONPATH \
PYTHONPATH="$WT/ros2_ur_ws/install/ur_gello_bringup/lib/python3.10/site-packages:$WT/third_party/hil-serl/serl_launcher" \
/home/laptop3/venvs/gello-hil-actor/bin/python -m pytest tests/test_frame_wrappers.py -q -p no:anyio
```

📌 2026-07-29 재실행: **20 passed**.

> ⚠️ 오버레이 `PYTHONPATH` 없이 돌리면 **`1 skipped`로 조용히 넘어간다**
> (`serl_launcher` 미발견). 통과했다고 착각하기 쉽다 → `00_SETUP_AND_SAFETY.md` §4.2.

### 7.4 reward / 분류 메타데이터 — **개입 스텝은 사실상 채점되지 않는다** (2026-07-29 신규)

위 §7.1–7.3은 **개입** 메타데이터 계약이고 **바뀌지 않았다.**
바뀐 것은 그 옆에 붙는 **reward 쪽 메타데이터**이고, 그 변화가 개입 루프에 직접 영향을 준다.

분류기는 이제 정책의 크롭된 관측이 아니라 actor가 따로 붙이는 **무크롭 sidecar**를 채점한다
(설계는 `05_COMMS_GRPC.md` §3.2). sidecar는 **모든 스텝에 붙지 않는다** —
`ur_env/classifier_sidecar.py::SidecarScheduler`가 두 가지로 게이트한다:

| 게이트 | 규칙 |
|---|---|
| **주기** | `interval_steps = 5` → 10 Hz 루프에서 **약 2 Hz** |
| **정지** | TCP 선속도가 `stationary_speed_max`(기본 `0.05 m/s`, **PLACEHOLDER 값이다**) 이하일 때만 |
| **에스컬레이션** | 확률이 `escalate_probability`(0.05) 이상이면 주기를 버리고 **매 스텝** |
| **예외 (게이트 무시)** | 로컬이 잠정 terminal이라고 판단한 스텝은 **정지 게이트를 무시하고 반드시 붙는다** — 성공을 한 스텝 늦게 잡으면 그건 실패로 기록된다 |

> ### 🎯 개입 루프에 대한 함의 — 이게 이 절의 요점이다
> **사람이 GELLO로 팔을 움직이는 동안에는 정지 게이트가 열리지 않는다.**
> 즉 **개입 스텝은 거의 전부 미분류(unclassified)로 replay에 들어간다.**
> 예외는 그 개입이 에피소드를 끝낸 스텝뿐이다(위 표 마지막 줄).

미분류 transition의 필드는 **정확히 이 모양**이어야 하고, 세 곳에서 독립적으로 강제된다
(`actor_network._validate_finalized_transition`, `grpc_actor_transport.outcome_from_proto`,
`ReplayIngress._convert`):

| 필드 | 값 | 왜 |
|---|---|---|
| `rewards` | **0.0으로 확정** | 서버가 reward 권위다. 분류가 없으면 성공의 증거가 없다 |
| `masks` / `dones` / `truncated` | **로컬 제안 그대로 통과** | `masks == 0.0 iff dones` 정합성이 특수 처리 없이 유지된다 |
| `classifier_evaluated` | `0` | |
| `classifier_probability` / `classifier_threshold` | `0.0` / `0.0` | 미평가일 때 0이 아니면 **거부된다** |
| `classifier_success` | `0` | |
| `reward_model_id` | `""` (빈 문자열) | 마찬가지로 미평가일 때 비어 있어야 한다 |

**학습 관점에서 미분류 transition은 평범한 zero-reward · non-terminal 표본이다.**
못 하는 것은 딱 하나 — **양의 reward로 에피소드를 끝내는 것**이고, 그건 아무도 채점하지 않은
스텝에서 일부러 뺏은 권한이다.

> 🛑 **"개입했는데 reward가 0이다"를 버그로 읽지 말 것.** 설계다.
> 성긴 채점은 대역폭 절감이 아니라 **판정 안정성**을 위한 것이다 — 큐브를 놓은 뒤 장면이
> 가라앉게 두고, 10 Hz로 성공 판정이 깜빡이는 것을 막는다.
>
> 반대로 **경고해야 할 신호**는 있다: 서버가 **연속 100건 미분류**이거나
> **한 세션이 단 한 건도 분류되지 않고** 끝나면 stderr에 경고를 찍는다
> (`rlpd_receive_server.py::RewardTransitionFinalizer`). 그게 뜨면 정지 게이트가 한 번도
> 안 열렸다는 뜻이므로 `--classifier-stationary-speed-max`를 의심한다 (`09` §1.2.1).

---

## 8. 판정 체크리스트

`[x]`는 2026-07-28 실기(`run_real_hil.py --arm --scale 0.25`)에서 확인된 것이다.
mock 루프(§3)에서는 **아무것도 확인되지 않았다** — 두 열을 섞지 말 것.

| | 항목 | 실기 (`run_real_hil.py`) | mock (§3) |
|---|---|---|---|
| 1 | `/forward_position_controller/commands` 퍼블리셔가 **1개**뿐 (§4) | [x] (러너가 자동 거부) | [ ] 수동 확인 |
| 2 | ENGAGE 시 팔이 리더를 추종, DISENGAGE 시 즉시 정책 복귀 | [x] | [ ] |
| 3 | engage 순간 zero-jump (누르기만 하고 안 움직이면 팔도 정지) | [x] anchor-latch 위반 0 | [ ] |
| 4 | 좌표계 3×3 대각 우세 (§5) — **수치로** 판정 | [x] 단위행렬 확정 | [ ] |
| 5 | gain 슬라이더를 스트로크 중에 움직여도 진행 중 개입이 안 튄다 (§2) | [x] gain-latch 위반 0 | [ ] |
| 6 | 저장 액션 == 실행 액션 (§2 불변식) | [x] 1.000 | [ ] |
| 7 | `held` / `reject_reason`이 폭주하지 않는다 | [x] held-rate 0 % | [ ] |
| 8 | GUI를 죽이면 첫 수신 후 0.5 s 안에 액터가 `DeadmanHeartbeatStaleError`로 종료되고, 단절 틱의 정책 액션이 전송되지 않으며 env/network가 닫힌다 | [ ] | [ ] |
| 9 | `info["intervened"]` / `intervene_action` 일관성 테스트 (§7.3) | [x] 오프라인 | — |
| 10 | **그리퍼 개입** (§6.3): 트리거 0.7↑ → `ia6 = -1.0`, 0.3↓ → `+1.0`, 그 사이 래치 유지 | [ ] `--gripper` 필요 | [ ] |
| 11 | **핵심 회귀:** 트리거 퍼블리셔를 죽이면 `ia6`가 0.0(HOLD)이 되고 **그리퍼가 저절로 열리지 않는다** | [ ] | [ ] |
| 12 | `cube_in_cup` config(워크스페이스 박스 활성)로 같은 루프 | [ ] → `08` G1 | — |
| 13 | **개입 스텝이 미분류로 들어온다** (§7.4): `classifier_evaluated=0`, `rewards=0.0`, `masks/dones`는 로컬 제안 그대로 | [ ] ⚠️ **판독구가 없다** — `09` §4.4 참조 | — |
| 14 | 정지 상태에서 sidecar가 **실제로 붙는다** (actor 종료 줄의 `sidecar n=`이 0이 아니다) | [ ] → `09` §4.4 | — |

항목 8의 코드 경로(ENGAGE/DISENGAGE 양쪽 stale, 정책 액션 미전달,
`run_remote_actor` 예외 전파와 CLI `finally` 정리)는
`tests/test_deadman_wiring.py`의 오프라인 회귀로 고정돼 있다. 위 표의 실기/mock 칸은
실제 GUI 프로세스를 죽여 본 판정만 기록하므로 아직 비워 둔다.

> 위 14개 항목은 **개입이 옳은가**를 본다. 개입이 **어떻게 느껴지는가**(빳빳함/덜덜거림)는
> 별개 문제이고 §9에 있다. §9는 오프라인 실측이며 **실기 판정 항목이 아니다.**

---

## 9. 개입 "빳빳함" — 실측된 원인과 🛑 손대면 안 되는 것 3개 (2026-07-30 신규)

조작자가 개입 중 팔이 **빳빳하고 덜덜거린다**고 보고했다. 이 절은 그 원인을 실측으로
지목하고, **되돌리면 더 나빠지는 세 가지**를 못박는다. 다음 사람이 "가속도 제한이 원인
같다"고 지우는 것을 막는 것이 이 절의 주 목적이다.

**측정 날짜: 2026-07-30. 로봇·rclpy·카메라 없이 순수 오프라인 측정이다.**
`AccelerationLimitedJointStream`(`serl_ur_infra/ur_env/envs/ros_backend.py:140-353`)은
순수 numpy이므로 `(target_q, now, target_time)`을 직접 먹여 명령 속도 프로파일을 그대로
꺼낼 수 있다. 조작자가 느끼는 것 전부가 그 객체 안에서 만들어진다.

```bash
# 재현 (5개 표를 한 번에 출력한다)
cd $WT/serl_ur_infra
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /home/laptop3/venvs/gello-hil-actor/bin/python \
  /home/laptop3/.claude/jobs/5dfbc13c/tmp/measure_intervention_smoothness.py
```

> ⚠️ **이 스크립트는 리포 밖(에이전트 작업 tmp)에 있다 — 사라질 수 있다.**
> 방법은 이 절에 전부 적혀 있으므로 없어지면 다시 만들 수 있다: 위 클래스를
> `UPSAMPLER` 기본값으로 만들고, 등속 리더(`target = v_lead * now`)를 **표에 적힌 주기마다
> 한 번씩** 갱신하며 250 Hz로 `advance()`를 돌리고, `stream.velocity`와 `stream.mode`를
> 모으면 된다. 각 표의 분석 창(run 길이 / 버리는 앞부분 / `soft_start_s`)이 표마다 다르고
> **창이 다르면 숫자가 안 맞으므로 반드시 같이 적어야 한다.**

> **📌 이 절은 원인 규명과 금지 사항만 담는다.** 코드 조치(현재 진행 중)의 상태·파일 목록·
> 검증 순서는 `08_OPEN_GAPS.md` **G24**에 있다. **여기의 어떤 숫자도 "고쳐졌다"는 뜻이
> 아니다** — 전부 오프라인 실측이고 실기·mock 검증은 0이다.

### 9.1 원인은 가속도 제한이 아니라 **타깃 갱신 주기**다

코드에서 확인한 사실 세 줄:

| | 근거 |
|---|---|
| `env.step()`은 100 ms 창에서 **타깃을 딱 한 번** 세팅하고 나머지를 잔다 | `ur7e_env.py:431-441` (`_apply_action` → `time.sleep(1/hz - dt)`) |
| `dt`는 `_apply_action` 시간만 뺀다 — **gRPC는 안 뺀다** | 같은 줄 |
| 그래서 gRPC 왕복이 `env.step` **밖**에 있다 | `remote_actor.py:538` (`env.step`) → `:609` (`network.step`) → 다음 루프의 `env.step` |

즉 250 Hz 스트리머가 보는 **실효 타깃 갱신 주기 = 100 ms + RPC 왕복**이고,
G16의 RTT 분포를 얹으면 실효 갱신율이 **약 5~9 Hz**로 떨어진다.

그 주기에서 스트리머가 하는 일은 정확히 다음이다 —
`_safe_step_for_distance()`(`ros_backend.py:280-297`)가 **목표를 넘지 않도록 제동거리를
남긴다.** 100 ms에 한 번만 갱신되는 타깃은 스트리머에게 "여기서 멈춰야 하는 점"이고,
가속 상한(8 rad/s²) 도달까지 **19.5틱 = 78 ms**가 걸린다. 그래서 매 주기가
**가속 → 제동 → 정지 → 새 타깃 대기**로 끝난다. 이것이 손에 전달되는 것이 "덜덜거림"이다.

**표 1 — 실제 세션이 만드는 주기** (리더 등속 0.15 rad/s, run 4.0 s의 뒤 60%, `soft_start_s = 0.7` = config):

| 상황 | 주기 | 관절 완전정지 | BRAKING | HOLD | 리플 | 리더속도 추종 |
| --- | --- | --- | --- | --- | --- | --- |
| 명목 10 Hz (RPC 0) | 0.100 s | 16.0 % | 0 % | 0 % | 2.27 | 100 % |
| + RPC p50 58 ms | 0.158 s | 32.5 % | 0 % | 0 % | 2.86 | 100 % |
| + RPC p99 97 ms | 0.197 s | 40.0 % | 0 % | 0 % | 3.20 | 100 % |
| + learner 경합 250 ms | 0.350 s | 17.5 % | 0 % | 14.8 % | 1.92 | 99 % |
| + 첫 publish 600 ms | 0.700 s | 51.5 % | 6.5 % | 51.5 % | 3.23 | **66 %** |

- **`관절 완전정지`** = 명령 속도가 정확히 0인 250 Hz 틱의 비율. **리플** = `(vmax−vmin)/mean`
  (0이면 등속, 2 이상은 완전한 stop-and-go).
- **RPC 0인 명목 10 Hz에서도 이미 16 %가 정지다.** RPC를 0으로 만들어도 이 증상은 안 없어진다.
- 0.158/0.197 s 값은 G16의 **열화된 링크**(13 Mbit/s) 기준이다. 07-29 실측 빠른 링크
  (p50 약 12~22 ms)에서는 주기가 약 0.112~0.122 s이므로 표 1의 첫 두 행 사이에 있다.

**표 2 — 주기를 100 ms로 고정하고 타깃 갱신 rate만 바꾼 것**
(run 3.0 s의 뒤 50 %, `soft_start_s = 0` — 정상상태만 본다):

| | 0.05 rad/s | 0.15 rad/s | 0.30 rad/s |
| --- | --- | --- | --- |
| **10 Hz 타깃 (현재)** | 정지 52.0 %, 리플 3.84 | 정지 16.0 %, 리플 2.27 | 정지 0 %, 리플 1.28 |
| **30 Hz 타깃** | 정지 22.4 %, 리플 2.25 | **정지 0 %, 리플 0.85** | 정지 0 %, 리플 0.43 |

**느리게 움직일수록 나쁘다** — 예산이 남아 목표에 더 빨리 도달하고 더 오래 기다린다.
조작자가 정밀 조작(느린 이동)에서 가장 심하게 느끼는 이유가 여기 있다.
그리고 100 ms 창 안에서 타깃을 **여러 번** 갱신하면(= 리더를 서브스텝) 정지가 사라진다.
액션은 여전히 10 Hz로 저장된다는 점이 핵심이다 — 이것은 **액추에이터 층** 변경이다.

### 9.2 "덜덜거림"과 "뒤처짐"은 다른 증상이고 해법도 다르다

타깃 외삽(velocity feed-forward, +1 주기)을 넣어 봤다. **리플을 전혀 개선하지 못한다.**

**표 4** (run 3.0 s의 뒤 50 %, `soft_start_s = 0`):

| 케이스 | 리더 0.05 | 리더 0.15 | 리더 0.30 |
| --- | --- | --- | --- |
| 현재 (10 Hz, 외삽 없음) | 리플 3.84, lag **−0.0035** | 리플 2.27, lag **−0.0131** | 리플 1.28, lag **−0.0288** |
| 10 Hz + 외삽 (+1 주기) | 리플 **3.84**(동일), lag **+0.0015** | 리플 **2.27**(동일), lag **+0.0019** | 리플 **1.28**(동일), lag **+0.0012** |
| 30 Hz 서브스텝 | 리플 2.25, lag −0.0014 | 리플 **0.85**, lag −0.0049 | 리플 0.43, lag −0.0123 |
| 30 Hz + 외삽 | 리플 2.25 | 리플 0.85 | 리플 0.43 |

- **외삽은 lag만 고친다** (0.15 rad/s에서 −0.0131 rad → +0.0019 rad, 즉 부호가 뒤집힐 만큼
  완전히 상쇄된다). 리플 숫자는 **소수점까지 동일**하다.
- **서브스텝은 리플을 고치고 lag를 부분적으로만 줄인다.**
- 그러므로 **조작자 보고를 먼저 분해해야 한다.** "덜덜거린다"면 rate, "뒤처진다/무겁다"면
  외삽이다. 둘을 한 덩어리로 취급하면 엉뚱한 쪽을 고치고 개선이 없다고 결론 내린다.

### 9.3 🛑 손대면 안 되는 것 3개 — 전부 되돌리면 **더 나빠진다**

이게 이 절의 가장 중요한 내용이다. 셋 다 "이게 원인 같다"고 지목되기 쉬운 것들이고,
실측은 **정반대**를 말한다.

**(a) 가속도 제한(`max_accel_rad_s2 = 8.0`)을 없애면 정지가 5배가 된다**
(리더 0.15, 주기 0.100, run 3.0 s 뒤 50 %, `soft_start_s = 0`):

| | 관절 완전정지 | 리플 | vmax |
| --- | --- | --- | --- |
| 8 rad/s² (현재) | 16.0 % | 2.27 | 0.341 rad/s |
| 제한 제거 (즉시 점프) | **76.0 %** | **4.17** | 0.625 rad/s |

이유: 가속 제한이 없으면 첫 틱에 최고속(0.625 rad/s)으로 튀어 **목표에 즉시 도달하고
남은 시간 전부를 정지로 보낸다.** 가속 제한은 원인이 아니라 **주기가 만든 stop-and-go를
완화하고 있던 것**이다. `ca19652`가 넣은 이 제한을 되돌리면 손맛이 더 나빠진다.

**(b) `target_stale_s`를 0.30에서 올리면 정지가 3배가 된다**
(리더 0.15, 주기 0.350, run 4.0 s 뒤 60 %, `soft_start_s = 0.7`):

| `target_stale_s` | 관절 완전정지 | 리플 | 추종 |
| --- | --- | --- | --- |
| **0.30 (현재)** | 17.5 % | 1.92 | 99 % |
| 0.50 | **54.5 %** | **4.09** | 102 % |
| 0.80 | 54.5 % | 4.09 | 102 % |

> **🔎 왜 0.50과 0.80이 완전히 같은가 — 여기에 메커니즘이 있다.**
> 주기가 0.350 s일 때 `target_stale_s = 0.30`은 **매 주기 stale을 발동시킨다.**
> stale → BRAKING → HOLD로 들어갔다가 새 타깃에 복귀할 때 `advance()`가
> **soft-start를 재무장한다**(`ros_backend.py:338-341`). 재무장된 soft-start는 스텝 상한을
> 15 %에서 다시 올리므로, 결과적으로 **팔이 목표를 향해 덜 세게 튀고 리플이 줄어든다.**
> 0.50 이상으로 올리면 stale이 아예 발동하지 않아 그 재무장이 사라지고, 아래 (c)의
> `soft_start_s = 0` 행과 **숫자가 완전히 같아진다**(54.5 % / 4.09). 즉 (b)와 (c)는
> **같은 메커니즘의 두 얼굴**이다.
>
> 추종이 102 %로 100 %를 넘는 것도 여기서 나온다 — 누적된 lag를 창 안에서 따라잡느라
> 평균 속도가 리더보다 잠깐 빨라진다. 좋은 신호가 아니라 **오버슈트 성향**의 표시다.
>
> ⚠️ 그러므로 표 1의 `0.350 s` 행이 `0.197 s` 행보다 **좋아 보이는 것은 착시가 아니라
> 실제**이지만, 좋은 이유가 "주기가 길어서"가 아니라 "stale HOLD + soft-start 재무장이
> 개입해서"다. **이걸 근거로 주기를 늘리려 하지 말 것** — 0.700 s 행이 그 끝을 보여준다.

**(c) `soft_start_s`(0.7 s)를 줄이면 리플이 2배가 된다**
(리더 0.15, 주기 0.350, run 4.0 s 뒤 60 %):

| `soft_start_s` | 관절 완전정지 | 리플 | 추종 |
| --- | --- | --- | --- |
| **0.7 (현재)** | 17.5 % | 1.92 | 99 % |
| 0.2 | 44.3 % | 3.18 | 98 % |
| 0.0 | **54.5 %** | **4.09** | 102 % |

"soft-start가 팔을 느리게 만든다"고 읽기 쉽지만, 추종률은 **98~102 %로 사실상 동일**하다.
soft-start가 하는 일은 속도를 깎는 것이 아니라 **stale 복귀마다 스텝 상한을 낮게 다시
시작해 튐을 없애는 것**이다. `ur7e_gello.yaml`이 실기 브리지에서 0.7 s로 검증한 값과 같고,
그것을 그대로 가져온 것이 맞았다.

### 9.4 리더 떨림 증폭 — rate 상향은 **One-Euro와 짝**이어야 한다

**개입 경로에는 리더 입력 필터가 없다.** 그래서 rate를 올리면 리더 떨림도 같이 증폭된다.

**표 5** (리더를 0 rad에 주차, 갱신마다 가우시안 노이즈, run 3.0 s 뒤 50 % = 1.5 s):

| 떨림 (1σ) | 10 Hz 타깃 | 30 Hz 타깃 |
| --- | --- | --- |
| 0.002 rad | 헛움직임 **23.7 mrad** / 1.5 s (최대속도 0.149 rad/s) | **95.0 mrad** (0.213 rad/s) |
| 0.004 rad | **47.5 mrad** (0.214 rad/s) | **161.3 mrad** (0.296 rad/s) |

**즉 rate 상향은 단독으로 넣으면 "덜덜거림"을 "지지직거림"으로 바꾼다.**
따라서 rate 상향은 **One-Euro 이식과 같이** 들어가야 한다. One-Euro는 정지 시 강하게 매끄럽게
하고 빠르게 움직일 때 cutoff를 열어 lag를 줄이는 필터로, **실기 텔레옵에서 검증된 손맛의
출처**다 (`ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/bridge_stages.py:42-106`).

> ### ⚠️ deadband는 **이식 대상이 아니다** — 원본에서 죽은 분기다
> `config/ur7e_gello.yaml:71`의 `deadband_rad: 0.004`는 매력적으로 보이지만,
> `filter_stage_joint()`가 `use_euro = euro is not None`으로 분기하고
> (`bridge_stages.py:134-143`) **One-Euro 경로에서는 deadband/EMA 줄에 도달하지 않는다.**
> 운용 설정이 `filter_type: "one_euro"`이므로 **검증된 손맛은 One-Euro 단독이 만든 것이고
> deadband는 한 번도 실행된 적이 없다.** deadband를 "원본에 있으니 같이 가져간다"고 넣으면
> 검증되지 않은 동작을 추가하는 것이 된다.
>
> One-Euro를 이식할 때 지켜야 하는 계약도 그 docstring에 있다
> (`bridge_stages.py:56-59`): **`update_input()`은 리더 샘플 cadence(약 30 Hz)로,
> `__call__()`은 출력 틱마다** 호출한다. 이 둘을 합치면 30 Hz 샘플 간격이
> 더 빠른 점프로 보여 cutoff가 **눈에 보이는 펄스로** 열린다.
>
> 30 Hz는 임의값이 아니다 — `gello_publisher.publish_rate_hz: 30.0`
> (`config/ur7e_gello.yaml:26`)이고, 백엔드 리더 캐시가 실제로 갱신되는 rate다.
> 그보다 빠르게 서브스텝하면 **같은 샘플을 다시 내는 것**이다.

### 9.5 남은 한계 — 예산이 소진되면 코드로는 못 고친다 → **G21**

표 1의 마지막 행이 그 경계다. 주기가 0.700 s가 되면:

- 관절이 시간의 **51.5 %를 HOLD**로 보낸다 (`target_stale_s = 0.30`을 넘겨 stale 판정).
- 리더 속도의 **66 %만** 추종한다 — 조작자가 미는 만큼 팔이 안 간다.
- 이 구간의 HOLD는 **의도된 안전 동작**이다(G4b). 필터·rate·외삽 어느 것도 이걸 못 없앤다.
  없애려면 `target_stale_s`를 올려야 하는데, 그건 §9.3(b)가 금지하는 것이고
  **낡은 목표를 계속 추종하는 안전 후퇴**다.

**그러므로 개입 부드러움의 나머지 절반은 코드가 아니라 RPC 지연에 달려 있다.**
0.700 s는 2026-07-29 실기에서 관측된 **첫 policy publish 5.474 s**와 learner 경합 중
learner step 중앙값 **약 1.12 s**에서 온 시나리오다 → `08_OPEN_GAPS.md` **G21**.
G21이 열려 있는 동안 위 세 표는 **좋은 날의 숫자**다.
