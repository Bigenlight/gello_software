# 08 — 미해결 안전 갭과 임시 완화책

이 문서는 **아직 닫히지 않은 것**만 모은다. "곧 고칠 것" 같은 낙관적 표현을 쓰지 않는다.
각 항목은 코드 근거 → 왜 위험한가 → **지금 쓸 수 있는 완화책** 순이다.

> ## 🛑 게이트 선언
> **G1 ~ G4가 닫히기 전에는 RL 정책 경로로 실기 UR7e를 구동하지 않는다.**
> `DRY_RUN`은 기본값 `True`로 둔다 (`serl_ur_infra/ur_env/envs/config.py:130`).

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## G1 — `clip_safety_box`가 **구현되어 있지 않다** 🔴

### 사실

```python
# ur7e_env.py:117-127
# workspace safety box (same semantics as FrankaEnv.clip_safety_box)
self.xyz_bounding_box = gym.spaces.Box(...ABS_POSE_LIMIT_LOW[:3], ...HIGH[:3])
self.rpy_bounding_box = gym.spaces.Box(...ABS_POSE_LIMIT_LOW[3:], ...HIGH[3:])
```

**두 객체는 생성된 뒤 리포 어디에서도 다시 참조되지 않는다.** (전수 grep 확인:
`xyz_bounding_box` / `rpy_bounding_box`는 `ur7e_env.py:118`, `:123` 두 줄이 전부.)

`_apply_action()`의 주석은 클램프가 적용되는 것처럼 읽힌다:

```python
# ur7e_env.py:234-237
# workspace box: clamp is applied on the *commanded* TCP pose.
# TODO(together): fold clip_safety_box into the controller gates so
# a clamped target re-solves IK instead of holding.
self.backend.send_joint_command(q_cmd)
```

**주석만 있고 클램프 코드는 없다.** `serl_ur_infra/README.md`의 현황표도
"워크스페이스 박스 (`ABS_POSE_LIMIT`) … ❌ config만 존재, 미작동"이라고 적고 있다.

### 왜 위험한가

- 정책이 발산하거나 사람이 개입을 놓친 순간, TCP가 **어디로든** 갈 수 있다.
  남은 것은 관절 리밋(`JOINT_LIMIT` HOLD)과 스텝 게이트뿐 — **작업 공간 개념이 없다.**
- `keepout` 존도 RL 경로에는 없다(README 현황표). 즉 **테이블·고정구·사람과의 충돌 인지가 0**이다.
- EEF 텔레옵 쪽 `max_excursion_m = 0.5`는 워크스페이스 제한이 **아니다** — engage마다
  앵커가 새로 잡혀 예산이 0으로 리셋된다 (`03_EEF_MODE.md` §6).

### 임시 완화책

1. **물리적 제약을 우선한다.** 로봇 주변에 실제 장애물/펜스를 두고, 팔이 물리적으로 닿을 수 있는
   범위 자체를 좁힌다.
2. **펜던트의 UR 안전 평면(Safety Planes)을 설정한다.** 이건 소프트웨어와 무관하게 컨트롤러가
   강제하며, protective stop으로 나타난다. **현재 유일하게 신뢰할 수 있는 작업공간 경계다.**
   (설정 여부 미확인 — 실기 투입 전 반드시 확인할 것.)
3. `ACTION_SCALE`과 `GOVERNOR`를 낮춰 단위 시간당 이동량을 줄인다 (사고를 막지는 못하고
   충돌 에너지만 줄인다).
4. 사람이 항상 E-STOP 위에 손을 둔다.

---

## G2 — `PolicyDeltaController`는 `eef_delta` 후반부의 **단순화판**이다 🔴

### 사실

`serl_ur_infra/README.md`의 현황표를 코드로 재확인했다
(`serl_ur_infra/ur_env/envs/policy_delta_controller.py:1-20`, `:101-134`):

| 기능 | `eef_delta` 본체 (텔레옵) | `PolicyDeltaController` (RL) |
|---|---|---|
| 속도 거버너 v_max/w_max | ✅ | ✅ |
| 관절 리밋 게이트 | ✅ | ✅ (`:105`) |
| 스텝 게이트 → HOLD | ✅ | ✅ (`:134` `STEP_LIMIT`) |
| **IK** | **branch-lock 해석 IK** (8분기 고정, merge point 처리, 가중 최근접) | **`ik_numeric` seed 방식만** (`:101-104`) |
| **`sigma_min` 특이점 감속** | ✅ | ❌ |
| **keepout 존** | ✅ | ❌ |
| **anti-windup lag 클램프** | ✅ | ❌ |
| **해석적 line search** | ✅ | ❌ (수치 축소 line search로 대체, `:108-134`) |
| **워크스페이스 박스** | (keepout으로 대체) | ❌ (= G1) |

README의 결론: **"분기 튐 방지가 약함 — 실기 전 교체 필수"**, 그리고 TODO 목록에
`policy_delta_controller → eef_delta 후반부 재사용 리팩토링` — **DRY_RUN 해제 전 필수**.

### 왜 위험한가

- **IK 분기 튐**: seed 기반 수치 IK는 특이점 근처에서 다른 분기로 넘어갈 수 있다.
  분기가 바뀌면 TCP는 거의 같은 자리인데 **관절이 크게 재배치된다** — 팔꿈치가 예고 없이
  반대편으로 넘어가는 형태다. branch-lock IK는 이걸 막으려고 있는 것이다.
- **특이점 감속 없음**: 텔레옵 EEF는 `gamma`로 부드럽게 감속하지만 RL 경로는 그냥 HOLD하거나
  통과한다.

### 임시 완화책

1. RL 경로는 **mock 하드웨어에서만** 돌린다 (`04_HIL_INTERVENTION.md` §3).
2. 실기가 불가피하면 `ACTION_SCALE`/`GOVERNOR`를 config 기본값
   (`[0.01, 0.05, 1.0]`, `v_max 0.12 / w_max 0.60 / dq_step_max 0.05`) **이하로만** 쓴다.
3. `info["held"]` / `info["reject_reason"]` 빈도를 **매 세션 로깅**한다.
   `NO_IK`/`STEP_LIMIT`이 늘어나면 특이점에 접근 중이라는 신호다.
4. 시작 자세를 특이점에서 먼 곳으로 잡는다 (신전 자세 회피).

---

## G3 — 그리퍼 개입이 **항상 비활성** 🟠

### 사실

```python
# wrappers.py:185
grip = float(arr[6]) if len(arr) > 6 else None      # arr 길이는 항상 6 → 항상 None
# wrappers.py:274-276
def _expert_gripper(self, grip):
    if grip is None:
        return 0.0                                   # → 항상 홀드
```

`/gello/joint_states`의 `position` 길이가 6이기 때문이다 (`gello_publisher_node.py:190`).
트리거는 **버려지지 않고** 별도 토픽 `/gripper/gripper_client/target_gripper_width_percent`로
정상 발행되지만 (`:193-195`), `URRosBackend`가 그 토픽을 구독하지 않는다
(`ros_backend.py:87-89`).

부수 효과: `GRIP_CLOSE_THR = 0.7` / `GRIP_OPEN_THR = 0.3` 히스테리시스가 **데드 코드**다
(`wrappers.py:191-192`, `:277-281`).

### 왜 위험한가

**"데드맨을 잡고 있으면 그리퍼도 내가 통제한다"는 거짓이다.** 개입 중에도 그리퍼는
**정책의 통제 아래** 있다. 물체를 놓치거나 물어버리는 것을 사람이 막을 수 없다.

### 임시 완화책

1. **개입 세션에서는 부서지기 쉬운 물체·손가락을 그리퍼 근처에 두지 않는다.**
2. 그리퍼를 즉시 열어야 하면 **별도 터미널에서** 서비스를 부른다 (수 초 지연 감수):
   ```bash
   ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"
   ```
   > 단, 러너가 계속 `command_percent`를 쏘고 있으면 다시 덮어써진다.
   > 확실한 해제는 **러너 정지 또는 E-STOP**이다.
3. `GRASP_PENALTY`/`GripperPenaltyWrapper` 실험 시 이 갭을 반드시 명시한다 —
   "사람이 그리퍼를 시연했다"는 전제가 성립하지 않는다.

**고칠 곳:** publisher가 아니라 `ros_backend`(트리거 토픽 추가 구독) + `get_gello_state()`가
7번째 값을 합치는 것. 미구현.

---

## G4 — 두 퍼블리셔 충돌 + 명령 스트림 타임아웃 부재 🔴

### G4a — 두 퍼블리셔

`/forward_position_controller/commands`에 두 개가 발행할 수 있다:

| 퍼블리셔 | 근거 |
|---|---|
| `gello_ur_bridge` (텔레옵, 250 Hz) | `ur7e_gello_real.launch.py:66`, `:779` |
| `URRosBackend` (RL env, 250 Hz 업샘플러) | `ros_backend.py:12`, `:111-113` |

**ROS2는 이걸 막지 않는다.** 두 스트림이 섞이면 컨트롤러는 마지막에 도착한 값을 따라가고,
결과는 두 목표 사이를 오가는 채터링이다.

**완화책:** HIL 개입 루프에는 `gello_publisher`만 띄우고 `run_ur7e_gello_real.sh`는 띄우지 않는다.
매 세션 확인:

```bash
ros2 topic info /forward_position_controller/commands --verbose | grep -c "Node name"
# 2 이상이면 즉시 중단
```

### G4b — 업샘플러에 타깃 스테일 정책이 없다

```python
# ros_backend.py:246-250
# Once at the target it keeps publishing the held pose; if the env dies
# the robot simply holds position. TODO(together): target-staleness
# policy (stop publishing after N s without a fresh target?), to be
# decided with the other safe-stop cases.
```

- 프로세스가 죽으면 daemon 스레드도 죽으니 괜찮다.
- **프로세스는 살아 있는데 env 루프만 멈춘 경우**(카메라 디코드 hang, gRPC 대기 등)에는
  업샘플러가 **마지막 타깃을 무한히 재발행**한다. 지시하는 사람이 없는 명령이 계속 나간다.
- 시나리오 E3b (`07_FAILURE_INJECTION.md`)에서 관측하기로 되어 있으나 **미실행**이다.

**완화책:** 세션 중 `ros2 topic hz /forward_position_controller/commands`를 별도 창에 띄워
두고, 러너가 멈췄는데 250 Hz가 유지되면 즉시 Ctrl-C/E-STOP.

---

## G5 — 19-D `state` 순서 계약 (변경 **진행 중**) 🟠

### 사실

문서 작성 시점(2026-07-27 17:15~17:16) `serl_ur_infra/ur_env/observation_schema.py`가
**커밋되지 않은 채** 수정되어 있다:

- 스키마 ID `hil-serl-ur-canonical-observation-v1` → **`-v2`**
- 평탄 레이아웃이 **알파벳순**으로 재정의됨:
  `gripper_pose[0:1)`, `tcp_force[1:4)`, `tcp_pose[4:10)`, `tcp_torque[10:13)`, `tcp_vel[13:19)`
- **그리퍼 인덱스가 18 → 0**. `state[..., -1]`은 TCP 각속도 z다.
- 스키마 해시가 바뀜 → **랩톱/Kanu 양쪽을 같이 올려야 통신이 성립**한다 (fail-fast로 거부됨).

이유: 평탄화를 하는 것은 우리가 아니라 upstream `SERLObsWrapper`이고, 그것이 쓰는
`gym.spaces.Dict`가 매핑을 **알파벳순으로 재정렬**한다. `proprio_keys`는 순서를 정하지 못한다.

### 남은 위험

1. `serl_ur_infra/RL_RECEIVE_SERVER.md`는 **아직 옛 순서**("TCP pose 6, TCP velocity 6,
   TCP force 3, TCP torque 3, gripper 1")를 적고 있다. **문서가 코드와 어긋나 있다.**
2. 이 워크트리에는 `third_party/hil-serl`이 없어 **실제 `SERLObsWrapper`를 통과시켜 본 적이
   없다.** 서브모듈이 있는 환경에서 `tests/test_state_layout_contract.py`를 반드시 한 번 돌려야 한다.
3. **수치가 바뀌는 중이므로 어떤 문서·코드에도 해시나 인덱스를 하드코딩하지 말 것.**

### 완화책

- 항상 라이브로 출력해서 확인한다 (`05_COMMS_GRPC.md` §4.2의 스니펫).
- 그리퍼는 `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`로만 읽는다.
- 원격 통신 전에 양쪽 해시 일치를 **먼저** 확인한다.

---

## G6 — staleness 시 safe-stop이 아니라 **예외** 🟠

| 위치 | 동작 | 코드가 인정하는 것 |
|---|---|---|
| `/joint_states` 0.2 s 초과 | `RuntimeError` | `TODO(together): safe-stop policy (freeze + operator prompt) instead of raise` (`ur7e_env.py:365-367`) |
| 카메라 0.5 s 초과 / 부재 | `RuntimeError` | `TODO(together): safe-stop policy instead of raise` (`ur7e_env.py:464-466`) |
| tcp_pose 부재/낡음 | `RuntimeError` | `:371-379` |

즉 센서가 끊기면 **러너가 크래시**한다. 결과적으로 팔은 마지막 명령 자세에서 멈추지만
(G4b의 예외 상황 제외), 이건 **설계된 안전 정지가 아니라 부작용**이다.
`info`에 `held`/`reject_reason`을 노출하고 staleness 정책을 통일하는 것은
README TODO에 남아 있다 ("HOLD/reject_reason을 step info로 노출 + staleness safe-stop 정책
통일 + UR fault recovery").

**완화책:** 데이터 수집 세션에서는 크래시 = 에피소드 손실이므로,
`06_SENSORS.md` §3.1의 topic hz 루프를 **세션 시작 시 반드시** 돌려 사전에 걸러낸다.

---

## G7 — 명목 DH ↔ 실기 캘리브레이션, 그리고 두 좌표계의 공존 🟠

### 사실

- `ur_kin.py`의 DH 상수는 UR5e/UR7e **명목값**이지 이 개체의 공장 캘리브레이션 값이 아니다.
- `config/ur7e_dh.yaml`을 읽는 `ur_kin.load_dh()`는 **정보용일 뿐** `fk`/`ik`/`jacobian`에
  배선되어 있지 않다. 캘리브레이션 파일을 넣어도 자동 반영되지 않는다.
  (근거: `docs/ros2/GELLO_UR7E_EEF_MODE.md`의 P6 (d) 항목)
- engage 게이트 G7은 **우리 FK와 우리 IK만** 비교하므로 이 불일치를 못 잡는다.

### RL에서의 추가 문제

`TCP_POSE_SOURCE` 기본이 `"driver"` (`config.py:114`)다. 그러면:

- **관측** `tcp_pose` = 벤더 FK (`/tcp_pose_broadcaster/pose`)
- **명령/개입 앵커** = 우리 명목 DH (`ur_kin.fk` / `controller.tcp_cmd()`)

두 계가 다르면 관측과 명령이 서로 다른 좌표계 위에 있게 된다.
코드는 앵커 수식이 자기일관적이라 zero-jump는 유지된다고 적고 있지만
(`ur7e_env.py:355-358`), **관측-액션 정합**은 별개 문제다.

### 완화책

1. `03_EEF_MODE.md` §4.1의 3자세 대조(5 mm / 5 mrad)를 실기 투입 전에 실행하고 결과를 기록한다.
2. 오차가 크면 `TCP_POSE_SOURCE = "fk"`로 바꿔 **관측과 명령을 같은 kinematics로 통일**한다.
   (그러면 절대 정확도는 포기하지만 내부 정합성은 얻는다.)
3. 어느 쪽이든 **절대 좌표 정밀도를 요구하는 태스크를 설계하지 않는다.**

---

## G8 — 브로드캐스터 누락이 **조용히** 관측을 오염시킨다 🟠

| 누락 | 결과 | 근거 |
|---|---|---|
| F/T 브로드캐스터 | force/torque 6개가 **전부 0** — 예외 없음 | `ur7e_env.py:398-402` |
| 그리퍼 `position_percent` | `curr_gripper_pos = 0.0` = **"완전 열림"으로 보임** | `ur7e_env.py:394-395` |

**스키마 해시로는 절대 안 잡힌다** — dtype/shape는 완벽히 맞기 때문이다.
`validate_canonical_observation`도 dtype/shape/유한성만 본다.

**완화책:** `06_SENSORS.md` §3.1의 topic hz 루프 + "팔을 살짝 밀어 wrench가 변하는지",
"그리퍼를 실제로 닫아 percent가 변하는지"를 매 세션 눈으로 확인한다.

---

## G9 — 개입 앵커가 **flange 기준**이다 (TCP 오프셋 미배선) 🟡

```python
# wrappers.py:208-210
def _leader_T(self, q_lead):
    # TODO: @ T_tool_L (flange-only for the skeleton)
    return fk(q_lead)
```

env는 `TCP_OFFSET_XYZ_RPY`로 `self.T_tool`을 만들어 두었지만 (`ur7e_env.py:179-182`),
개입 앵커 계산에는 쓰이지 않는다. README TODO에도 남아 있다
("`GelloIntervention._leader_T`에 TCP_OFFSET 배선 (config는 있음, 현재 플랜지 기준)").

**영향:** 텔레옵 EEF 경로에서 `tool_l = tool_r`이 중요했던 것과 같은 이유로
(`03_EEF_MODE.md` §6), 리더가 **회전할 때** 개입 델타의 회전 중심이 어긋난다.
병진만 하면 차이가 없다.

**완화책:** 개입 시 회전보다 병진 위주로 시연한다. 회전 시연이 필요한 태스크면
이 갭을 먼저 닫는다.

---

## G10 — 이 워크트리에 `third_party/hil-serl`이 없다 🟡

```bash
cd $WT && git submodule status
# -c32939bcc... third_party/hil-serl     ← 미초기화
```

`serl_launcher`를 import하는 모든 것이 실패한다: `SERLObsWrapper`, `RelativeFrame`,
`Quat2EulerWrapper`, `ChunkingWrapper`, 수신 서버의 replay store.
따라서 **wrapper 체인 전체를 이 워크트리에서 통과시켜 본 적이 없다.**

**완화책:** `00_SETUP_AND_SAFETY.md` §2.3의 (a) 또는 (b).

---

## G11 — 스크립트 수정이 **커밋되지 않았다** 🟡

`run_ur7e_gripper.sh` / `build_ur7e.sh`의 `set -u` 버그 수정이 워킹트리에만 있다
(`00_SETUP_AND_SAFETY.md` §3.3).
`git checkout .` / `git stash` / `git clean -fd` 한 번이면 **그리퍼 브링업과 빌드가 다시 깨진다.**

**완화책:** 세션 시작 시 `git status`. 그리고 이 수정을 소유한 담당자가 커밋할 것.

---

## G12 — 스페이스바 데드맨이 기본값이다 🟡

`GelloIntervention(env, deadman=None)`이면 `SpacebarDeadman()`이 만들어진다
(`wrappers.py:171`). 그런데 그것은 **전역 pynput 리스너**라 다른 창에서 친 스페이스도 잡고
(`wrappers.py:82-97`), X11 오토리핏으로 깜빡이며(그때마다 앵커 재래치),
gain이 1.0으로 고정이고(`:109-110`), 하트비트 워치독이 없다.

**완화책:** 항상 `--deadman topic` + `run_hil_gui.sh`를 쓴다 (`04_HIL_INTERVENTION.md` §1).
실기 세션의 기본을 토픽 데드맨으로 바꾸는 것이 바람직하나 **미구현**이다.

---

## 부록 — 발견된 문서 불일치 (코드가 정답)

이 조사 중 상위 문서에서 발견한 낡은 서술. 고치는 것은 각 문서 소유자의 몫이다.

| 문서 | 낡은 서술 | 코드/실제 |
|---|---|---|
| `docs/ros2/GELLO_UR7E_EEF_MODE.md:380` | P9 "yaml 기본 `v_max=0.08`" | `config/ur7e_gello_eef.yaml:238` = **0.16** (같은 문서 `:7`도 0.16이라 자기모순) |
| `serl_ur_infra/RL_RECEIVE_SERVER.md` | state 순서 = pose6, vel6, force3, torque3, gripper1 | 현재 `observation_schema.py` = **알파벳순, gripper가 index 0** |
| `serl_ur_infra/README.md` TODO | "v_max(0.1 m/s) vs ACTION_SCALE 최대(0.2 m/s) 정합 — 거버너가 풀액션을 절반으로 자름" | `config.py:64` `ACTION_SCALE=[0.01,0.05,1.0]` × `HZ=10` = 0.1 m/s < `GOVERNOR v_max=0.12` (`:75`). **이미 정합되어 있고 경고도 안 뜬다** |
| `serl_ur_infra/README.md` 머리말 | "⚠️ UNTESTED SKELETON" | 여전히 맞다 (실기 RL 경로 미검증). 유지 |
