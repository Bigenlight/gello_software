# 04 — HIL 개입 (데드맨 · 앵커 · 좌표계 · 메타데이터)

**상태: 이 브랜치에서 미검증.** mock + RViz 절차는 존재하고 정본은
`serl_ur_infra/RVIZ_HIL_TEST_CLI.md`다. 이 문서는 **판정 기준과 함정**에 집중한다.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## 1. 데드맨 — 두 가지 소스

`GelloIntervention`은 `DeadmanSource` 인터페이스로 engage 신호와 gain을 받는다
(`serl_ur_infra/ur_env/envs/wrappers.py:54-64`).

| | `SpacebarDeadman` (기본) | `RosTopicDeadman` (**권장**) |
|---|---|---|
| 코드 | `wrappers.py:67` | `wrappers.py:112` |
| 신호원 | pynput 전역 키보드 리스너 | ROS 토픽 `/hil/deadman` |
| gain | **1.0 고정** (`wrappers.py:109-110`) | 슬라이더 0.10 ~ 1.00 |
| 워치독 | **없음** | 0.5 s (`STALE_S`, `wrappers.py:145`) |
| 선택 | `python3 tests/run_rviz_hil.py` | `python3 tests/run_rviz_hil.py --deadman topic` |

### 1.1 🛑 스페이스바 데드맨이 위험한 이유

권장하지 않는다. 네 가지 이유가 있고, 셋은 코드 근거가 있다.

1. **X11 키 오토리핏이 홀드를 "뗐다 눌렀다"로 만든다.**
   → 개입이 깜빡이고, 그때마다 `_disengage()` → `_engage()`가 돌아 **앵커가 재래치된다**
   (`wrappers.py:287-293`). 즉 조작 도중 기준점이 계속 리셋된다.
   (`RVIZ_HIL_TEST_CLI.md`의 "스페이스바 개입이 깜빡임" 항목)
2. **pynput 리스너는 전역이다.** 터미널 포커스와 무관하게 X 세션의 스페이스를 잡는다
   (`wrappers.py:82-97`). 다른 창에서 친 스페이스가 **로봇을 engage시킬 수 있다.**
3. **gain이 1.0으로 고정된다** (`wrappers.py:109-110`). 감도를 낮출 수단이 없다.
4. **워치독이 없다.** 토픽 데드맨은 하트비트가 0.5 s 끊기면 자동 해제되지만
   (`wrappers.py:158-165`), 스페이스바는 그런 안전망이 없다.

> ✅ 다만 한 가지 fail-safe는 있다: 디스플레이가 없어 pynput이 죽으면
> `except Exception`으로 잡아 **"영원히 engage 안 됨"** 상태로 떨어진다
> (`wrappers.py:98-103`). 조용히 engage된 채로 남지는 않는다.

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
| 스테일 | 0.5 s 넘으면 `engaged=False` | `wrappers.py:145`, `:158-165` |

- ENGAGE(OFF→ON)는 **두 번 클릭 확인**, DISENGAGE(ON→OFF)는 한 번 클릭
  (`gello_hil_gui_node.py:42-44`).
- GUI는 **로봇도 GELLO도 브리지도 건드리지 않는다.** 퍼블리셔 하나만 소유한다
  (`gello_hil_gui_node.py:6-10`, `:29-31`).

> ### ⛔ 다시 강조: DISENGAGE는 정지가 아니다
> 데드맨을 놓으면 **정책이 즉시 이어받아 팔을 계속 움직인다**
> (`wrappers.py:287-290`). GUI가 죽어서 워치독이 걸려도 결과는 같다 — **정책 복귀**다.
> 정지는 `00_SETUP_AND_SAFETY.md` §6의 수단으로만 한다.

---

## 2. 앵커와 gain 래치

engage 순간(`_engage()`, `wrappers.py:215-222`)에 세 가지가 **한 번에 래치**된다:

```python
self.T_g_anchor = self._leader_T(q_lead)     # 리더 TCP (현재는 flange 기준)
self.T_r_anchor = self._robot_T_cmd()        # 로봇 '명령' TCP  (측정값 아님)
self._gain      = self.expert.gain()         # 감도 — 여기서 한 번만 읽는다
self._anchored  = True
```

핵심 성질:

| 성질 | 의미 | 근거 |
|---|---|---|
| **gain은 engage 에지에서만 읽는다** | 스트로크 도중 슬라이더를 움직여도 진행 중인 개입이 갑자기 재스케일되지 않는다 | `wrappers.py:218-221` |
| **gain은 병진에만 곱한다** | 회전은 항상 1:1 | `wrappers.py:243-248` |
| **앵커는 `controller.tcp_cmd()`(명령값) 기준** | 관측(`tcp_pose`)과 비교하지 않으므로 `TCP_POSE_SOURCE`가 driver든 fk든 앵커 수식은 자기일관적이다 | `wrappers.py:212-213`, `ur7e_env.py:355-358` |
| **disengage 시 앵커 폐기** | 다음 engage는 완전히 새 기준 | `wrappers.py:224-227` |
| **리더가 0.3 s 이상 낡으면 개입 거부** | `LEADER_STALE_S = 0.3` | `wrappers.py:190`, `:287` |
| **anti-windup은 비례(norm) 클램프** | 축별 `np.clip`이 아니다. 대각선 이동의 **방향**이 왜곡되지 않는다 | `wrappers.py:254-267` |

> ### 저장 액션 불변식
> 개입 액션은 `÷ACTION_SCALE → clip`으로 만들어져 **실행값과 저장값이 같다.**
> gain은 그 나눗셈 **앞**에 적용되므로 불변식을 깨지 않는다 (`wrappers.py:243-248`).
> 단 `ACTION_SCALE * HZ`가 거버너 캡을 넘으면 **실행만 잘리고 저장은 안 잘려** 불변식이
> 깨진다. env 기동 시 위반하면 WARNING을 찍는다 (`ur7e_env.py:96-109`).

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
`--episodes N`, `--max-steps N`, `--leader-timeout S` (`run_rviz_hil.py:149-181`).

> ### ⚠️ `run_rviz_hil.py`를 실기에 겨누지 말 것
> `DRY_RUN = False`가 **일부러** 박혀 있고 (`run_rviz_hil.py:56`),
> mock 전용 완화값들이 들어 있다: `RESET_MAX_DIST_RAD = 7.0` (`:69`),
> `ACTION_SCALE = [0.03, 0.10, 1.0]` = **0.3 m/s** (`:80`),
> `GOVERNOR = {v_max: 0.36, w_max: 1.2, ...}` (`:81`).
> 이 값들을 실기 config로 복사하면 안 된다 (`:83-85`에 그렇게 적혀 있다).

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

> ✅ 신규 실기 러너 `serl_ur_infra/tests/run_real_hil.py`는 기동 시
> `node.count_publishers()`로 이걸 직접 세고, 자기 것 외에 하나라도 더 있으면
> **`--arm`을 거부한다** (`run_real_hil.py:34-38`, `:439`). mock 러너
> (`run_rviz_hil.py`)에는 이 가드가 없다 — 수동으로 확인할 것.

---

## 4.5 실기 HIL 러너 (`run_real_hil.py`) — 신규, 미검증

`run_rviz_hil.py`는 **mock 전용**이라("Never point this at a real robot") 실기에서
개입 경로를 검증할 러너가 없었다. `serl_ur_infra/tests/run_real_hil.py`(untracked)가
그 자리를 채운다.

설계 원칙 세 가지 (`run_real_hil.py:14-24`):

1. **기본이 안전하다** — `DRY_RUN=True`가 기본이라 로봇 명령을 **한 줄도 발행하지 않는다.**
   팔을 움직이려면 `--arm`을 명시해야 한다 (`:319`).
2. **속도는 3층을 함께 올린다** — `--scale`이 `ACTION_SCALE` / `GOVERNOR` / `UPSAMPLER`를
   동시에 곱한다. (한 층만 올리면 다음 층이 조용히 먹어버린다.)
3. **모든 스텝을 CSV로 남긴다** — 앵커 래치 / gain 래치 / 좌표계 매핑 /
   "저장 액션 == 실행 액션" 불변식을 **사후에 수치로** 검증하기 위해서.

정책은 **항상 zero**다. 학습 정책은 붙이지 않는다.

```bash
cd $WT/serl_ur_infra
python3 tests/run_real_hil.py                    # DRY_RUN — 로봇 명령 0줄
python3 tests/run_real_hil.py --arm --yes        # ⚠️ 실제로 움직인다
```

주요 플래그: `--arm`, `--scale`, `--episodes`(기본 1), `--max-steps`(기본 300),
`--reset-mode {startup,hold}`, `--reset-max-dist`(기본 0.05), `--tcp-source {fk,driver}`
(기본 **fk**), `--cameras`, `--gripper`, `--csv`, `--leader-timeout`, `--yes`.

리셋 설계가 중요하다: 기동 시점의 실제 관절값을 그대로 `RESET_JOINTS`로 잡고
`RESET_MAX_DIST_RAD=0.05`로 조인다 → **"리셋 = 지금 자세 유지"**. 개입으로 팔을 많이
옮긴 뒤 다음 에피소드를 리셋하면 가드에 걸려 **에러로 멈춘다 — 그게 의도된 안전 실패다**
(먼 거리를 쓸고 오지 않는다). 그 경우 `--reset-mode hold` 또는 `--episodes 1`.

> **먼저 `--arm` 없이** 돌려서 CSV·좌표계·앵커가 말이 되는지 확인한 다음에 arm한다.
> 아래 §5 좌표계 3×3과 §7 메타데이터 검증은 **DRY_RUN에서 전부 가능하다.**

---

## 5. 좌표계 3×3 검증

**목적:** 리더 축 → 로봇 base 축 매핑이 identity인지, 축간 크로스토크가 없는지.

**전제:** mock + RViz (§3). 개입 중 `info["intervene_action"]`을 로깅한다.

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

## 6. 그리퍼 개입 — **결함이었고, 방금 고쳐졌다 (하드웨어 미검증)**

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

### 6.2 현재 코드 (2026-07-27, 커밋 안 됨)

`URRosBackend`가 트리거 토픽을 **별도로 구독**하고 `merge_gello_state()`가 두 스트림을
7-요소로 합친다 (`ros_backend.py:64-124`, `:162-172`). `get_leader()`는 NaN을 `None`으로
매핑해 "지금 트리거 없음"을 한 가지 표현으로 통일한다 (`wrappers.py:179-196`).

핵심 설계 결정 세 가지 — **판정할 때 이걸 본다**:

| 결정 | 왜 |
|---|---|
| 트리거 부재 = **`NaN`**, `0.0` 아님 | `0.0`은 "완전 열림"이라는 정당한 값이다. 센티널로 쓰면 토픽이 죽을 때마다 **그리퍼를 조용히 연다** (`ros_backend.py:91-96`) |
| 반환 age는 **관절 age만** | 트리거가 없다고 팔 텔레옵까지 막으면 안 된다 (`:97-101`) |
| 트리거 없음 → `_expert_gripper` **0.0 = HOLD**, 래치는 건드리지 않음 | `0.0`은 그리퍼를 움직일 수 없는 유일한 값이라, 죽은 토픽이 **잡은 것을 떨어뜨리지도, 뭔가를 물지도** 못한다. 래치를 재발행하면 이미 없는 신호에서 유래한 grasp를 계속 명령하게 된다 (`wrappers.py:284-310`) |

트리거 스테일 임계는 `GELLO_TRIGGER_STALE_S = 0.3`으로 `LEADER_STALE_S`와 맞춰 뒀다 —
두 스트림이 같은 30 Hz 타이머에서 나오므로, 한쪽만 조용해졌다는 건 트리거 읽기 자체가
실패했다는 뜻이다 (`ros_backend.py:70-74`).

### 6.3 판정 (미실행)

```bash
# 1) 유닛 (rclpy·시리얼 불필요)
cd $WT/serl_ur_infra
python3 -m pytest tests/test_gello_gripper_wiring.py -q -p no:anyio

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

`GelloIntervention.step()` (`wrappers.py:300-314`):

| 키 | 타입 | 의미 | 언제 |
|---|---|---|---|
| `policy_action` | `float32[7]` | **오버라이드 전** 정책 출력 (개입 중에는 counterfactual) | **항상** |
| `intervened` | `int` 0/1 | 사람 액션이 실행되었는가 | **항상** |
| `intervene_action` | `float32[7]` | **실제로 실행된** 사람 액션 (`[-1,1]^7`) | 개입 시**에만** |
| `left` / `right` | `bool` False | spacemouse 버튼 호환용 자리 | 항상 |

- `policy_action`은 wrapped env가 보기 **전에 복사**된다 — 호출자 버퍼나 실행 액션과
  절대 별칭(alias)이 되지 않는다 (`wrappers.py:302-304`).

### 7.2 `build_transition()`의 fail-fast

`serl_ur_infra/ur_env/rlpd_actor.py:33-92`. 다음 세 경우에 **예외를 던진다**:

| 조건 | 예외 |
|---|---|
| `intervened`가 0/1이 아님 | `ValueError: intervened must be 0 or 1` (`:62-63`) |
| `intervened`와 `intervene_action` 존재 여부가 불일치 | `ValueError: inconsistent intervention metadata` (`:64-68`) |
| 실행 액션과 정책 액션의 shape 불일치 | `ValueError: executed and policy action shapes differ` (`:75-79`) |

생성되는 transition 키 (`:80-92`):

```
observations, actions(=실행된 액션), policy_actions, intervened(uint8),
next_observations, rewards, masks(=1.0-done), dones,  [grasp_penalty]
```

라우팅 규칙 (`:95-105`): 모든 transition은 replay에, `intervened == 1`인 것은
**추가로** intervention buffer에 들어간다.

### 7.3 판정 명령

```bash
cd $WT/serl_ur_infra
python3 -m pytest tests/test_intervention_metadata.py tests/test_rlpd_actor_adapter.py \
  -q -p no:anyio
```

---

## 8. 판정 체크리스트 (mock 단계)

- [ ] `/forward_position_controller/commands` 퍼블리셔가 **1개**뿐 (§4)
- [ ] GUI ENGAGE 시 RViz 팔이 리더를 추종, DISENGAGE 시 즉시 정책 복귀
- [ ] engage 순간 zero-jump (누르기만 하고 안 움직이면 팔도 정지)
- [ ] 좌표계 3×3 대각 우세 (§5) — **수치로** 판정
- [ ] gain 슬라이더를 스트로크 중에 움직여도 진행 중 개입이 안 튄다 (§2)
- [ ] GUI를 죽였을 때 0.5 s 안에 `intervened`가 0으로 떨어진다 (→ 정책 복귀, 정지 아님)
- [ ] `info["intervened"]` / `intervene_action` 일관성 테스트 통과 (§7.3)
- [ ] **그리퍼 개입이 안 되는 것을 재확인** (§6) — 이건 PASS가 아니라 **기록된 결함**이다
