# GELLO → UR7e **JOINT_DELTA 모드**(시작 앵커 기준 관절 델타) 실행 런북 (operator runbook)

> **문서 상태 (2026-07-24)**: **무동작(no-motion) 기동 배선 완료**(eef와 동일 방식: 제자리 STRICT 전환 + `~/joint_delta_start` 앵커) + 단위테스트 통과. **실기 UR7e에서 `HEADLESS=true … control_mode:=joint_delta jd_gain:=1.0` 동작 확인(사용자, 2026-07-24)** — 기동 시 팔이 GELLO 자세를 쫓아가지(chase) 않고 **제자리에서 앵커**, 이후 GELLO **델타만** 추종. 이 문서가 조작자용 **정본**이다.
>
> **안전 불변**: GELLO는 항상 passive read-only. Dynamixel에 절대 토크 X.
>
> **기존 모드는 한 줄도 바뀌지 않았다.** `control_mode:=joint`(기본)과 `control_mode:=eef`의 동작은 이 기능 추가 전과 **동일**하다. joint 모드가 그렇다는 것은 `test/test_bridge_stages.py`의 bit-identity 테스트로 계속 증명된다.

---

## 0. JOINT_DELTA 모드가 하는 일 (한 줄)

**절대 미러링이 아니라 "움직인 만큼"을 따라간다.** engage(앵커) 순간 리더 관절과 로봇 명령 관절을 각각 스냅샷하고, 이후로는 **리더가 앵커 이후 이동한 누적량만** 로봇 앵커에 더한다.

```
engage:      q_lead_anchor, q_robot_anchor 래치, delta := 0
매 tick:     delta += wrap_to_pi(q_lead - q_lead_prev)      # 관절별
             q_cmd  = q_robot_anchor + jd_gain * delta
             q_cmd  = clamp(q_cmd, joint limits, excursion cage)   # anti-windup
             q_cmd  = q_cmd_prev + clip(q_cmd - q_cmd_prev, ±step) # slew
```

joint 모드(`q_cmd = q_leader`, 절대 미러링) 및 eef 모드와 공존하며 `control_mode` launch 파라미터로 고른다. 상류(FACTR/YAM)의 `auto_align`(`gello/factr/gravity_compensation.py:469-483`)과 대수적으로 같은 사상이다 — 다만 상류는 `map_signs`(±1)만, 여기는 연속 `jd_gain`을 쓴다.

**이 모드가 존재하는 이유는 두 가지다.**

1. **클러치(마우스 리프트).** `~/joint_delta_clutch` → 리더를 자유롭게 다시 잡고 → `~/joint_delta_reclutch`. 마우스를 들어서 다시 놓는 것과 정확히 같다. 로봇은 그동안 **능동적으로 제자리를 유지**한다(브리지는 pause되지 **않는다**).
2. **리더 드리프트 내성.** GELLO의 절대 원점이 조금 틀어져 있어도 상관없어진다 — 앵커 이후의 **변화량만** 쓰기 때문이다.

---

## CLI 빠른 참조 (요약)

> 아래는 요약이다. 실기 전에 반드시 §"시작 전 위험 고지"와 §"실기 절차"를 읽을 것.

**빌드 (yaml/launch/py 변경 시마다 필수):**
```bash
cd /home/laptop3/gello_software/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

**① mock 예행연습 (로봇 없음 — 여기부터):**
```bash
# gain 0 — fake 리더를 움직여도 팔이 안 움직여야 정상
PATTERN=hold JD_GAIN=0.0 ./run_ur7e_gello_joint_delta_mock.sh launch_rviz:=false
# gain 1 — 델타 추종 확인
PATTERN=sweep JD_GAIN=1.0 ./run_ur7e_gello_joint_delta_mock.sh
```

**② 실기 (REMOTE 모드, Play 불필요 — 반드시 감독 하 / E-STOP 손 닿는 거리):**
```bash
# 첫 구동은 반드시 gain 0 — 기동 시 팔이 제자리에 있고, GELLO를 흔들어도 미동 없어야 함
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=joint_delta jd_gain:=0.0
# 정지 확인되면 델타 추종 (사용자 2026-07-24 확인)
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=joint_delta jd_gain:=1.0
```
> 무동작 기동이라 예전처럼 팔이 GELLO 자세로 끌려가지 **않는다.** 기동 시 `forward_position_controller`로 **제자리 STRICT 전환** → 로봇 현재 자세에서 앵커 → 첫 명령 = 현재 자세(점프 0) → 이후 GELLO 델타만 추종. 별도 `joint_delta_engage` 수동 호출 **불필요**(launch가 `~/joint_delta_start`로 자동 진입).

**③ 델타 제어 서비스 (별도 터미널, 모두 `std_srvs/srv/Trigger`):**
```bash
ros2 topic echo /gello_ur_bridge/joint_delta/state                              # 상태·joint_gap 관찰
ros2 service call /gello_ur_bridge/joint_delta_clutch    std_srvs/srv/Trigger "{}"  # 얼림(리더 재배치용)
ros2 service call /gello_ur_bridge/joint_delta_reclutch  std_srvs/srv/Trigger "{}"  # 현 위치서 재앵커(마우스 리프트)
ros2 service call /gello_ur_bridge/joint_delta_to_joint  std_srvs/srv/Trigger "{}"  # 절대 모드로 반납/탈출
ros2 service call /gello_ur_bridge/joint_delta_disengage std_srvs/srv/Trigger "{}"  # 델타 해제
# joint_delta_engage / joint_delta_start 는 launch가 자동 처리 — 수동 복구용
```

**주요 인자 / 환경변수:**

| 인자·변수 | 뜻 | 기본값 |
|---|---|---|
| `control_mode:=joint_delta` | 모드 선택 (`joint`/`eef`/`joint_delta`) | `joint` |
| `jd_gain:=` | 델타 게인 (`0.0`=정지, `1.0`=1:1 델타) | `1.0` |
| `HEADLESS=true` | Method B (REMOTE, 펜던트 Play 불필요) | 없음(=Method A) |
| `ROBOT_IP=` | 로봇 IP | `192.168.10.11` |
| `PATTERN=` / `JD_GAIN=` (mock) | fake 리더 동작 / mock 게인 | — |

---

## ⚠️ 시작 전 위험 고지 — 반드시 읽을 것

### (H1) **리더의 자세가 더 이상 로봇의 자세를 알려주지 않는다** — 이 모드의 최대 잔여 위험

joint 모드에서는 "내가 GELLO를 든 모양 = 팔의 모양"이 **구조적으로 보장**된다. JOINT_DELTA에서는 그 대응이 **의도적으로 끊긴다.** clutch/reclutch를 몇 번 하고 나면 리더와 팔은 임의로 어긋나 있을 수 있고, 코드가 이것을 막아주지 않는다(막으면 기능 자체가 없어진다).

- **완화 수단은 가시화뿐이다.** `~/joint_delta/state`의 `joint_gap`(리더 vs 팔의 관절별 원거리)이 지금 얼마나 벌어졌는지를 10 Hz로 알려준다.
- **표준 대응**: 헷갈리면 즉시 `~/joint_delta_clutch` → 리더를 편한 자세로 → `~/joint_delta_reclutch`. 그래도 감이 안 오면 `~/joint_delta_to_joint`(절대 모드로 반납)로 나가서 정렬 후 `~/resume`.

### (H2) **관절 리밋 쪽으로 팔을 몰아갈 수 있다** — joint 모드에는 없던 위험

joint 모드에서 명령은 `q_cmd = q_leader`라서 **리더의 기구적 가동범위가 곧 명령의 한계**였다. `q_robot_anchor + gain*delta`에는 그런 암묵적 한계가 없다. 리더가 중립 근처에 있어도 팔은 리밋까지 갈 수 있다.

- **코드가 막는다**(이 항목은 완화가 아니라 차단이다): 관절별 **포화 clamp** = `ur_kin.JOINT_LIMITS ± jd_limit_margin_rad`, 그 안쪽에 다시 `jd_max_excursion_rad` **케이지**(앵커 기준 ±). 팔꿈치(index 2)는 **±π**이지 ±2π가 아니다(`ur_kin.py:63-76`, `test_h`/`test_p`).
- 포화는 **freeze가 아니라 saturate**다. 한 관절이 리밋에 닿아도 나머지는 계속 따라간다(6축 전체가 굳는 HOLD보다 훨씬 안전하고 예측 가능하다). 포화 중에는 `~/joint_delta/state`의 `limited[i]`가 true다.
- 되돌아올 때 **데드존이 없다**: 포화 시 accumulator를 back-project(anti-windup)하므로 리더를 반대로 0.01 rad 움직이면 **바로 다음 tick에** `gain*0.01`만큼 움직인다(`test_i`).

### (H2b) ⚠️ 포화(clamp)가 한 번이라도 걸리면 **사상(mapping)이 영구히 어긋난다**

(H2)의 back-project는 공짜가 아니다. 케이지/리밋을 넘어간 리더 이동량은 **버려진다.** 그래서 **한 번이라도 포화된 뒤**에는 리더를 앵커 자리로 **정확히** 되돌려도 명령은 앵커로 돌아오지 않는다.

- 실측 예 (`jd_gain: 1.0`, `jd_max_excursion_rad: 1.0`, `test_s`): 리더를 +1.5 rad 밀면 팔은 +1.0에서 포화(0.5 rad 폐기), 리더를 0.0으로 완전히 되돌리면 팔은 **−0.5 rad(−28.6°)** 에 머문다. **영구적이다.**
- **경계는 있다**: 변위는 언제나 케이지 안(`|q_cmd − anchor| <= jd_max_excursion_rad`)이다.
- **보이게 만들었다**: `~/joint_delta/state`의 `clamp_discarded[6]`(앵커 이후 폐기된 리더 이동량 누적, rad). **0이 아니면 (H1)의 어긋남이 그만큼 커진 것이다.**
- **대응은 (H1)과 같다**: `~/joint_delta_clutch` → 재배치 → `~/joint_delta_reclutch` (재앵커 시 `clamp_discarded`도 0으로 리셋된다, `test_v`).
- 대안이 더 나쁘기 때문에 이 절충을 택했다: back-project를 빼면 복귀 스트로크에 **무한정** 데드존이 생기고("텔레오퍼레이션이 고장난 느낌"), 포화 시 HOLD를 걸면 관절 하나 때문에 **6축 전체가 굳는다**. 자세한 논증은 `ur_gello_bringup/joint_delta.py` 모듈 docstring.
- ⚠️ 따라서 `jd_max_excursion_rad`를 **더 조인다고 더 안전해지지 않는다** — 포화 빈도가 올라가 대응 상실이 잦아진다.

### (H3) `~/resume`의 정렬 게이트는 **델타 스트리밍을 지켜주지 않는다**

`~/resume`/`~/resume_chase`는 `|리더 - 팔| <= resume_align_tol`을 요구한다. 델타 모드에서 드리프트가 쌓인 뒤에는 이 조건이 **만족 불가능**해서 영원히 거부되고, 반대로 앵커 직후에는 **정의상 항상 만족**되어 아무것도 지켜주지 않는다. 그래서

- `~/resume` / `~/resume_chase`는 **손대지 않았고**, 성공 시 항상 `JOINT_BOOTSTRAP`(절대 passthrough)으로 착지한다. 즉 그 게이트는 **진짜 절대 사상에 대해** 측정되므로 의미를 유지한다.
- 델타 진입은 **전용 경로**로만 한다: `~/joint_delta_start`(PAUSED 브리지를 chase 없이 팔 현재 자세에 앵커 — 무동작 기동에서 런치가 자동으로 부르는 경로이자 PAUSED 복구 경로), 또는 이미 스트리밍 중이면 `~/joint_delta_engage`(게이트 G0~G7, 재앵커). `~/joint_delta_start`의 `_has_streamed` 게이트와 그 무동작-기동 opt-in은 §4.1 참조.

### (H4) 이 모드로 녹화한 데이터셋은 기존 데이터셋과 **호환되지 않는다**

`(observation, action)`의 action은 여전히 로봇 관절 명령이지만, **리더 채널이 더 이상 로봇 채널과 같지 않다.** 기존 ACT/Diffusion/FM 체크포인트와 `cube_in_cup`/banana 데이터셋은 모두 절대 사상 전제다.

- **Phase 1 범위에서 JOINT_DELTA 모드 녹화는 범위 밖이며 권장하지 않는다.**
- 굳이 한다면 (i) 세션 메타데이터에 `control_mode`를 반드시 기록하고, (ii) 별도 데이터셋 디렉토리에 두고, (iii) **action-space 감사 없이 기존 데이터셋과 절대 병합하지 말 것**.
- `control_mode` 기본값은 `joint`이므로 기존 녹화/배포 런북은 **아무 영향을 받지 않는다**.

---

## 1. 설정 walkthrough — `config/ur7e_gello_joint_delta.yaml`

이 파일은 **단독으로 쓰이지 않는다.** `config/ur7e_gello.yaml` 위에 덮어 씌워지며(런치가 그렇게 계층화한다), joint 모드 파라미터(`filter_type`, `max_step_rad`, `soft_start_s`, `publish_rate_hz`, `staleness_timeout_s`, `resume_*` …)는 **하나도 건드리지 않는다**.

### 1.1 `jd_gain: 1.0` — 리더→로봇 관절 이득

**1.0이 기본인 이유**: 1:1이면 engage 이후의 **손맛이 joint 모드와 완전히 동일**하다. 바뀐 것은 **원점뿐**이다. 다른 값을 쓰면 조작자가 몸으로 익힌 손↔팔 비율이 달라지는데, 그건 앵커 자체보다 더 큰 놀라움이다.

- `0.0` = 팔이 **전혀** 움직이지 않는다 → 단계별 기동의 첫 스테이지.
- 생성자에서 `[0.0, 2.0]`을 **하드 검증**한다(`JointDeltaController.__init__`, `test_q`).
- CLI 오버라이드 가능: `jd_gain:=0.0` (rebuild 불필요).

### 1.2 `jd_limit_margin_rad: 0.05` — 관절 리밋 여유

`ur_kin.JOINT_LIMITS`를 양쪽으로 이만큼 좁힌 뒤 포화시킨다. eef 오버레이의 `limit_margin_rad`와 의미는 같지만 **별도 노브**라 두 모드를 독립적으로 튜닝할 수 있다. 컨트롤러는 관절별 배열을 그대로 들고 있으므로 **팔꿈치 ±π**가 정확히 반영된다.

### 1.3 `jd_max_excursion_rad: 1.0` — 앵커 기준 케이지 **(실기 튜닝 필요)**

(H2)에서 잃어버린 "리더가 기구적으로 유한하니 명령도 유한하다"는 성질의 **대체물**이다. 명령은 관절별로 `[anchor - 이 값, anchor + 이 값]`을 절대 벗어나지 못한다.

**보수적으로 시작해도 안전한 이유**: 케이지는 언제나 움직임을 **줄이기만** 하고, 걸리면 `limited[i]`로 **눈에 보인다**. 1.0 rad ≈ 57°.

### 1.4 `jd_leader_jump_max_rad: 0.35` — 단일 tick 리더 점프 상한 **(실기 튜닝 필요)**

한 tick에 이보다 큰 증분은 사람의 동작이 아니라 USB 글리치·샘플 뭉치 유실·엔코더 multi-turn 재해석이다. 이런 tick은 **흡수(absorb)** 한다 — 사상(mapping)이 이동하고 로봇은 움직이지 않는다.

**왜 latch가 아니라 absorb인가**: latch하면 다음 tick이 같은(혹은 더 큰) 간격을 다시 측정해서 **영원히 거부**하는 데드락이 된다. 흡수 횟수는 `jump_absorbed`로 계측된다.
30 Hz GELLO 기준 0.35 rad ≈ 10 rad/s로, 의도적 사람 동작보다 훨씬 위다. 너무 빡빡하면 실패 양상은 **"로봇이 뒤처진다"** 이지 **"로봇이 튄다"** 가 아니다 → 보수적으로 시작해도 된다.

### 1.5 `jd_delta_deadband_rad: 0.0` — **일부러 꺼 두었다. 켜지 말 것**

증분은 **텔레스코핑**된다(각 tick 증분의 합 = 총 이동량). 그래서 **누적 자체에는 드리프트가 원리적으로 없다** — 나갔다가 정확히 돌아오면 명령도 앵커로 정확히 돌아온다(`test_r`, 5000 tick 잡음 주행에서 오차 < 1e-9).

여기에 데드밴드를 걸면 **증분에 대한 비선형**이므로 편향이 **적분되어 드리프트를 만든다** — joint 모드에서 데드밴드가 **절대값**을 게이팅하던 것과 정반대다. 지터는 상류(리더 1-Euro, `one_euro_*`)와 하류(포화·slew clamp)에서 잡는다.

### 1.6 왜 증분을 앵커가 아니라 **직전 tick** 기준으로 wrap하는가 (가장 중요한 규약)

```python
d = wrap_to_pi(q_lead[i] - q_lead_prev[i])   # 맞음
delta[i] += d
# delta[i] = wrap_to_pi(q_lead[i] - q_lead_anchor[i])   # 틀림 — ±π에서 접힌다
```

**worked counter-example**: 리더 손목을 앵커에서 +200°(≈ +1.11π) 돌렸다고 하자.
- 올바른 형태: 매 tick의 작은 증분이 그대로 쌓여 `delta = +3.49 rad` → 팔도 +200° 돈다.
- 순진한 형태: `wrap_to_pi(+3.49) = -2.79 rad` → **팔이 반대로 −160° 돈다.**

`test_c_accumulation_beyond_pi_regression`이 이 두 값을 **각각 손으로 계산해서 발산함을 단언**한다. 또한 이 가드는 **매 tick 무상태(stateless)** 여야 한다 — 리더는 앵커를 잡은 **한참 뒤에** ±π 컷을 넘을 수 있기 때문이다(`test_d`, wrist_3 ≈ π 부근의 실제 배치).

### 1.7 재사용되는 공용 게이트 노브

`anchor_agree_tol`, `filter_settled_tol`, `hold_latch_s`, `tick_budget_us`, `tick_overrun_limit`는 eef 모드와 **이름을 공유**한다(한 프로세스에서 모드는 하나만 살아 있고 오버레이도 모드별이므로 중복 정의하지 않는다). 값은 eef 오버레이와 동일하게 맞춰 두었다.

---

## 2. 빌드 / 배포 — **변한 것이 없다**

- **새 런타임 의존성 없음.** `joint_delta.py`는 stdlib(`math`) + `angle_utils`만 쓴다. numpy도 `ur_kin`도 import하지 않는다. 그래서 `package.xml`은 **손대지 않았고**, eef의 `requirements-eef.txt` 같은 것도 필요 없다.
- **`setup.py`도 손대지 않았다.** `find_packages()`가 새 모듈을, `config/*.yaml` / `launch/*.py` glob이 새 오버레이와 새 런치를 자동으로 집는다(설치 확인 완료).
- **yaml/launch를 고쳤으면 반드시 rebuild.** colcon은 심볼릭 링크가 아니라 **복사**한다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
```

---

## 3. 상태 기계와 클러치 워크플로

```
                 joint_delta_engage  (G0..G7)   /  joint_delta_start (PAUSED에서)
 JOINT_BOOTSTRAP ───────────────────────────────────────────────►  ENGAGED
   (절대 passthrough)                                              │    ▲
        ▲                                        joint_delta_clutch│    │joint_delta_reclutch
        │ joint_delta_to_joint                                     ▼    │
        │ (+ ~/resume 로 복귀)                                   CLUTCHED
        │                                                          │
        └────────────  pause / disengage / 자동 fault ────────►  DISENGAGED (== _paused)
                                                                   │
                                          joint_delta_start ───────┘
```

- `HOLD`는 **상태가 아니다.** 한 tick의 거부 결과가 `info["state"]`에만 나타나는 값이며, `self.state`는 `ENGAGED` 그대로다.
- **`CLUTCHED`는 `_paused`를 건드리지 않는다.** 이것이 eef의 release 경로를 재사용할 수 없었던 이유다(`~/eef_disengage`는 pause하고, 돌아오는 길인 `~/resume`은 델타 모드가 일부러 깨는 정렬을 요구한다).
- `~/state`(operator UI)는 이 모드에서 `DELTA_FOLLOWING` / `DELTA_CLUTCHED`를 새로 낸다. **joint / eef의 기존 문자열은 하나도 바뀌지 않았다** (`check_pause_resume_sim.sh` 영향 없음).

### 마우스 리프트 레시피 (그대로 복사해서 쓸 것)

```bash
# 1) 델타 제어 시작 (팔은 움직이지 않는다 — delta = 0)
ros2 service call /gello_ur_bridge/joint_delta_engage   std_srvs/srv/Trigger "{}"

# 2) 리더를 다시 잡고 싶을 때: 클러치 (팔은 능동적으로 제자리 유지, pause 아님)
ros2 service call /gello_ur_bridge/joint_delta_clutch   std_srvs/srv/Trigger "{}"
#    ... GELLO를 편한 자세로 옮긴다. 로봇은 미동도 하지 않는다 ...
# 3) 다시 물기 (delta = 0 으로 재앵커, 점프 없음)
ros2 service call /gello_ur_bridge/joint_delta_reclutch std_srvs/srv/Trigger "{}"

# 즉시 정지 (앵커 폐기 + pause)
ros2 service call /gello_ur_bridge/joint_delta_disengage std_srvs/srv/Trigger "{}"
# PAUSED에서 복구 (chase 없이 현재 팔 자세에 앵커, delta = 0)
ros2 service call /gello_ur_bridge/joint_delta_start     std_srvs/srv/Trigger "{}"
# 절대 모드로 반납 (이후 리더 정렬 → ~/resume)
ros2 service call /gello_ur_bridge/joint_delta_to_joint  std_srvs/srv/Trigger "{}"
```

### engage 게이트 (G0~G7, 전부 fail-closed)

| Gate | 내용 | 적용 |
|---|---|---|
| G1 | `control_mode == joint_delta` | 전부 |
| G0 | 브리지가 **PAUSED가 아닐 것** (PAUSED면 `~/joint_delta_start`를 쓰라고 거부) | engage |
| G2 | 리더 샘플 신선도 `age <= staleness_timeout_s` | 전부 |
| G3 | 명령 기준선 `_last_published` 존재 | 전부 |
| G4 | `_actual_pose` 신선 + `max|cmd-actual| <= anchor_agree_tol` (**양쪽 다 로봇 쪽**) | engage |
| G5 | `leader_quasi_still(...)` — **동작 도중 앵커 금지** | 전부 |
| G6 | 리더 1-Euro 필터 수렴 `max|q_lead_f - raw| < filter_settled_tol` | 전부 |
| G7 | 앵커 자체가 마진 포함 관절 리밋 **안쪽**일 것 | 전부 |

eef의 G7(IK self-test)/G8(특이점)/G9(keepout)은 **의도적으로 뺐다** — 이 모드에는 Cartesian 사상이 없어서 그 게이트들은 연극이 된다. 그 자리를 G7(앵커 리밋)과 컨트롤러의 매 tick 포화 clamp가 대신한다.

**G5는 선택 사항이 아니다.** 절대 모드에는 아예 대응물이 없는 **새 위험**을 막는 유일한 장치다: 동작 중에 앵커를 잡으면 **움직이고 있던 손 위치가 조용히 원점으로 재정의된다.**

---

## 4. 기동(bring-up) — **무동작(no-motion), eef와 동일**

`control_mode:=joint_delta`의 기동 경로는 이제 **eef 모드와 완전히 동일한 무동작 경로**다(더 이상 joint 모드의 chase가 아니다). 팔은 **처음부터 자기 현재 자세에 그대로 머무르고**, 앵커 이후에는 **GELLO가 앵커 대비 움직인 만큼만** 움직인다.

- `start_mode`는 AUTO에서 `switch_only`로, `bridge_resume_service`는 `/gello_ur_bridge/joint_delta_start`로 파생된다(런치 파생 로직은 `_auto_start_mode`/`_auto_resume_service`, `ur7e_gello_real.launch.py`; `test/test_launch_derivation.py`가 joint/eef/joint_delta 세 값을 못 박는다).
- 드라이버는 `scaled_joint_trajectory_controller`로 올라오고, `gello_move_to_start`는 `forward_position_controller`로 **제자리 STRICT 전환**만 한다(궤적을 만들지도 보내지도 않으므로 팔은 움직이지 않는다). 그 뒤 `~/joint_delta_start`가 **팔의 실제 현재 자세**에 앵커하고 PAUSED 브리지를 `ENGAGED` 델타 스트리밍으로 푼다 — chase 없음.
- 첫 publish 명령은 팔의 실제 자세와 **허용오차가 아니라 대수적으로** 같다(`delta == 0`이므로 zero-jump). protective stop이 발생할 여지가 없다.
- 델타 제어를 시작하는 데 **수동 `~/joint_delta_engage` 호출은 필요 없다.** `~/joint_delta_engage`는 **이미 스트리밍 중인** 브리지를 다시 앵커하는 수동 경로로 남는다.

> **`~/joint_delta_start`가 chase 없이도 안전한 이유.** 절대 모드에서 chase가 필요한 이유는 첫 명령이 `q_leader`이고 그게 팔에서 임의로 멀 수 있기 때문이다. 여기서는 첫 명령이 `q_robot_anchor := 팔의 현재 자세`, `delta == 0`이므로 zero-jump가 대수적으로 성립한다. chase의 나머지 두 보호는 사라지는 게 아니라 **자리를 옮겼고, 둘 다 구현되어 있다**: "움직이는 리더에게 넘기지 말 것"은 게이트 (c) `leader_quasi_still`, "gross-mispose 백스톱"은 스트리밍 시점의 관절 리밋 + excursion 케이지다.

### 4.1 게이트 (a2) `_has_streamed`와 `jd_start_allow_unstreamed` opt-in

`_on_jd_start`에는 게이트 **(a2)** 가 있다 — 기본적으로 **브리지가 최소 한 번은 스트리밍한 적이 있어야** `~/joint_delta_start`를 받는다(`_has_streamed`). 이 게이트가 막으려는 것은 딱 하나다: 핸드셰이크 이전 PAUSED 구간에서 서비스를 부르면 chase **이전**의 팔 자세로 앵커한 뒤 **비활성** `forward_position_controller`에 250 Hz로 쏘게 되고, 나중에 그 컨트롤러가 활성화되는 순간 스냅이 발생한다. (`jd_gain:=0.0`도 이걸 막지 못한다 — gain 0은 명령을 그 낡은 앵커에 고정시키는데, 그 값이 바로 튀는 목표값이기 때문이다.)

무동작 `switch_only` 기동에서는 이 위험이 **구조적으로 제거된다**: `gello_move_to_start`가 `_resume_bridge_after_switch`(= `~/joint_delta_start` 호출)에 도달하는 것은 STRICT 전환이 **성공한 뒤에만**이므로, 서비스가 불릴 때 `forward_position_controller`는 이미 **ACTIVE**다. `_has_streamed`가 대리하려던 불변식("컨트롤러가 먼저 활성이어야 한다")이 호출 순서로 보장된다 — `~/eef_resume`가 게이트 없이 의존하는 바로 그 거래다.

그래서 런치는 joint_delta 브리지를 **`jd_start_allow_unstreamed:=True`** 로 띄운다(노드 파라미터, 기본값 `False`). 이 opt-in이 켜지면 (a2)가 우회되어 never-streamed 브리지도 무동작 기동에서 앵커된다. **기본값 `False`이므로** joint/eef 브리지나 손으로 브리지를 띄운 조작자에게는 (a2)가 그대로 남아 수동 오호출을 막는다. yaml에는 넣지 않는다(런치 범위의 안전 opt-in). 회귀 테스트는 `test/test_bridge_stages.py`의 `test_joint_delta_start_arms_never_streamed_bridge_when_opted_in`(수락) / `test_joint_delta_start_still_refuses_never_streamed_by_default`(기본 거부).

---

## 5. 단계별 브링업 표

| 단계 | 환경 | 명령 | 합격 기준 |
|---|---|---|---|
| **P1** | mock, 합성 리더 | `PATTERN=hold JD_GAIN=0.0 ./run_ur7e_gello_joint_delta_mock.sh launch_rviz:=false` | `~/joint_delta_engage`가 **수락**된다(G5 통과). 이후 `/forward_position_controller/commands`가 **완전히 고정**. RViz 모델 미동. |
| **P2** | mock, 합성 리더 | 같은 런치, `JD_GAIN=1.0`, `PATTERN=sweep` | 팔이 리더 **변화량**을 따라간다. `~/joint_delta/state`의 `slewed`가 가끔 true(정상), `limited`는 false. |
| **P3** | mock | `PATTERN=step` | `raw_jumps`가 증가하고 **명령은 튀지 않는다**. `reject_reason: "LEADER_RESYNC"` + `resyncing: true`가 보였다 사라진다. 창이 닫힌 뒤 명령이 **원래 자리 그대로**여야 한다(잔차 0 — `test_u`). `jump_absorbed`(백스톱)는 보통 0. |
| **P4** | mock | `PATTERN=full_rotation` | wrist_3가 **한 방향으로 계속** 돈다(중간에 반대로 접히면 §1.6 회귀). `delta[5]`가 단조 증가하다 `jd_max_excursion_rad`에서 `limited[5]=true`로 포화. |
| **P5** | mock | clutch/reclutch 왕복 20회 | reclutch 직후 명령이 **정확히 동일**(점프 0). `joint_gap`이 커지는 것을 눈으로 확인 → (H1) 체감. |
| **P6** | **실기**, `jd_gain:=0.0` | `./run_ur7e_gello_real.sh control_mode:=joint_delta jd_gain:=0.0` | (a) 드라이버가 `scaled_joint_trajectory_controller`로 올라오고 `forward_position_controller`로 **제자리 STRICT 전환**(팔 미동)이 끝난 뒤, 브리지가 팔 실제 자세에 앵커하며 **곧바로 `ENGAGED`** 로 진입한다(수동 `~/joint_delta_engage` 불필요). (b) 전환·앵커 순간 **팔이 전혀 움직이지 않는다**. (c) 이후 GELLO를 크게 움직여도 **팔은 미동도 하지 않는다**(gain 0). (d) `~/joint_delta_to_joint` → 리더 정렬 → `~/resume`로 절대 모드 복귀가 된다. |
| **P7** | 실기 | `jd_gain:=0.25` | 작은 리더 동작에 팔이 **1/4로** 따라간다. protective stop 없음. |
| **P8** | 실기 | `jd_gain:=0.5` → `1.0` | 1.0에서 joint 모드와 **손맛이 같다**. clutch/reclutch가 실사용 가능. |

**P6의 (c)(d)는 이 모드의 안전 판정 기준이다.** eef의 P6(`pos_scale=0.0`)와 달리 여기서는 `jd_gain=0.0`이 **모든 관절 채널**을 곱하므로 "팔이 전혀 움직이지 않아야 한다"가 **문자 그대로 참**이다(`test_n`).

---

## 6. 진단 / 디버깅

```bash
ros2 topic echo /gello_ur_bridge/joint_delta/state
ros2 topic echo /gello_ur_bridge/state          # DELTA_FOLLOWING / DELTA_CLUTCHED
```

`~/joint_delta/state`(JSON, 10 Hz) 필드:

| 필드 | 뜻 |
|---|---|
| `state` | 노드 상태 `JOINT_BOOTSTRAP` / `ENGAGED` / `CLUTCHED` / `DISENGAGED` (PAUSED 중 ENGAGED로 보이는 일이 없도록 방어됨) |
| `ctrl_state` | 컨트롤러의 tick 결과 — `ENGAGED` / `HOLD` / `CLUTCHED` / `DISENGAGED` |
| `reject_reason` | `BAD_INPUT` / `LEADER_JUMP` / `LEADER_RESYNC` / `null` (`ESTOP`은 컨트롤러 레벨 가드로, 브리지의 soft-start 램프가 항상 >0이라 실제로는 도달 불가) |
| `auto_reason` | 자동 disengage 사유 (`leader_stale`, `robot_joint_state_stale`, `hold_latched (...)`, `tick_budget ...`, `exception: ...`) |
| `gain` | 적용 중인 `jd_gain` |
| `delta[6]` | 앵커 이후 누적 리더 이동량(rad) |
| `limited[6]` | 이번 tick에 포화 clamp가 걸린 관절 |
| `slewed[6]` | 이번 tick에 slew clamp가 걸린 관절 |
| `excursion_rad` / `max_excursion_rad` | 앵커 대비 현재/허용 최대 |
| `max_joint_step` | 이번 tick의 최대 관절 변화 |
| `jump_absorbed` | 컨트롤러 내부 **백스톱** 가드가 흡수한 횟수(필터된 신호 기준 — 보통 0이어야 정상) |
| `raw_jumps` / `resyncing` | **1차** 리더 텔레포트 검출기(원시 GELLO 스트림, 소스 cadence)의 누적 검출 횟수 / 지금 hold-resync 창이 열려 있는지 |
| `clamp_discarded[6]` | 포화 clamp가 버린 리더 이동량 누적(rad). **0이 아니면 (H2b)** — 리더를 앵커로 되돌려도 팔은 그만큼 어긋난 채 남는다 |
| `joint_gap[6]` | **리더 vs 팔**의 관절별 원거리 → (H1) 가시화 |
| `tick_us`, `tick_soft_overruns` | 스테이지 소요 시간(성능 신호, 게이트 아님) |

### 증상 → 원인 순서

1. **팔이 전혀 안 움직인다** → `state`가 `ENGAGED`인가? (`JOINT_BOOTSTRAP`이면 engage가 거부된 것 — 서비스 응답 메시지에 `[gate_key] — detail`이 그대로 들어 있다.) → `gain`이 0.0인가? → `limited`가 전부 true인가(케이지/리밋 포화)?
2. **팔이 뒤처진다** → `slewed`가 계속 true → `max_step_rad`(joint 모드와 공유)가 병목. 리더를 천천히.
3. **팔이 굳었다(HOLD)** → `reject_reason` 확인. `LEADER_RESYNC`(정상 경로: 원시 스트림에서 텔레포트를 잡아 `jd_resync_hold_s` 동안 홀드 중)나 `LEADER_JUMP`(백스톱)가 반복되면 GELLO USB/드라이버 문제이거나 `jd_leader_jump_max_rad`가 과도하게 빡빡한 것. `raw_jumps` 증가 속도로 구분한다.
3b. **팔이 앵커에서 조금씩 어긋난 채로 남는다** → `clamp_discarded`를 본다. 0이 아니면 (H2b)다 — 버그가 아니라 문서화된 절충이며, clutch/reclutch로 리셋한다.
4. **갑자기 PAUSED가 됐다** → `auto_reason`. 그대로 `~/joint_delta_start`로 복구(리더 정렬 불필요).
5. **engage가 계속 거부된다** → 응답의 게이트 키를 본다: `leader_moving`(G5, 리더를 0.5초 가만히), `filter_not_settled`(G6, 잠시 대기), `chains_disagree`(G4), `bridge_paused`(G0 → `~/joint_delta_start`를 쓸 것), `anchor_at_limit`(G7).

---

## 7. 범위 밖 (의도적)

관절별 gain 벡터/부호 반전, 델타 모드에서의 데이터 녹화·정책 배포, Cartesian 관련 일체(FK/IK/pose 토픽 없음), `scripts/joint_delta_replay.py`(오프라인 HDF5 리플레이 — `eef_replay.py`의 형제, 유효하지만 보류), eef와 joint_delta 혼합, 런타임 `control_mode` 전환, 그리퍼 델타 사상(그리퍼는 **절대** 유지), chase 없는 기동을 자동화하는 시퀀서 노드(`gello_anchor_switch`, Phase 2 — **미구현·미검증**).
