# 03 — EEF 모드 단계 상승 (P6 → P7 → P8 → P9a → P9b)

**상태: 사용자가 실기에서 EEF 텔레옵을 이미 검증했다.**
이 문서는 그 단계 상승을 **재현/재검증**할 때의 절차와 판정 기준이다.

정본: `docs/ros2/GELLO_UR7E_EEF_MODE.md`. 여기서는 테스트 관점의 요약 + 이 브랜치의 실행 명령만 담는다.

> ### 2026-07-27 실기 세션의 영향: **없음 (의도적으로)**
> 오늘 진전된 것은 RL/HIL 쪽(`04`~`09`)이고 EEF 텔레옵 경로는 건드리지 않았다.
> 이 문서의 판정은 그대로 유효하다.
>
> 다만 **RL 경로와 혼동하지 말 것**: EEF 텔레옵의 `keepout`/`sigma_min` 감속/branch-lock IK는
> `PolicyDeltaController`(RL)에는 **없다** (`08_OPEN_GAPS.md` G2). 새로 생긴
> `clip_safety_box`는 **RL 경로 전용**이고 EEF 텔레옵에는 적용되지 않는다.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
cd $WT/ros2_ur_ws
```

---

## 0. 시작 전 (매 단계 공통)

- `00_SETUP_AND_SAFETY.md` §5 체크리스트 전부 통과.
- **`keepout_json`은 `"{}"`다 — 충돌 인지가 전무하다.** 모든 EEF 단계는 사람이 지켜본다.
- **손을 E-STOP 위에 둔다.** EEF 모드에서도 물리 E-STOP은 동일하게 작동한다
  (`GELLO_UR7E_EEF_MODE.md:356`).
- 두 번째 터미널에 상태 모니터를 띄워 둔다:

```bash
# 판정용 라이브 모니터 (별도 터미널)
source /opt/ros/humble/setup.bash && source $WT/ros2_ur_ws/install/setup.bash
ros2 topic echo /gello_ur_bridge/eef/state
```

- 세 번째 터미널에 GUI:

```bash
cd $WT/ros2_ur_ws && ./run_eef_gui.sh
```

> **시작 시 팔은 움직이지 않는다.** `control_mode:=eef`면 `run_ur7e_gello_real.sh`가
> `start_mode`를 자동으로 `switch_only`로 잡는다 (`run_ur7e_gello_real.sh:86-92`, `:135-140`).
> 컨트롤러 스위치는 제자리에서 일어나고 브리지가 팔을 그 자리에 홀드한다.
> 팔은 **`eef_engage`(GUI 큰 토글) 이후에만** 움직인다.

---

## 1. 🛑 이 문서에서 가장 중요한 한 가지

> # `pos_scale:=0.0`을 줘도 팔은 **크게 움직인다**.
>
> `pos_scale`은 **위치 항 하나만** 곱한다. 회전 채널에는 스케일이 **아예 없다**:
>
> ```
> R_des = R_delta @ R_r_anchor                              # ← pos_scale 없음
> p_des = p_r_anchor + pos_scale * R_align @ (p_g - p_g_anchor)
> ```
> (`config/ur7e_gello_eef.yaml:66-73`, `:87-90`; 수식 원본은 `eef_delta.py`의 `step()`)
>
> 따라서 `pos_scale=0.0`은 **TCP 위치를 앵커에 못박는 것**일 뿐이고,
> 로봇은 TCP를 제자리에 둔 채 공구를 회전시키며 **어깨·팔꿈치·손목이 실제로 크게 스윙한다.**
>
> 실기 실측 (`GELLO_UR7E_EEF_MODE.md:407-412`):
>
> | 리더 입력 | TCP 위치 변화 | TCP 자세 변화 | **최대 관절 변화** |
> |---|---|---|---|
> | `wrist_3` +40° | 2.6e-16 m | 40.00° | **0.698 rad (≈40°)** |
> | 팔 전체 흔들기 | 2.7e-16 m | 60.29° | **1.020 rad (≈58°)** |
>
> ## → **판정 대상은 "팔의 무동작"이 아니라 `excursion_m`이다.**
> 눈으로 팔을 보고 P6를 FAIL로 기록하지 말 것. 팔이 움직이는 것이 **정상**이다.
> `excursion_m`이 0이 아니면 그때가 앵커 로직 버그다.
>
> **그래서 P6는 "안전하게 아무것도 안 하는 단계"가 아니다.** 팔꿈치·어깨가 지나갈
> 물리적 클리어런스를 미리 확보하고, 흔드는 동작은 **천천히, 작게** 한다.

### 1.1 두 번째로 중요한 것: `rot_freeze` / `pos_freeze`는 **존재하지 않는다**

PLAN 문서의 P7 `rot_freeze:=true` / P8 `pos_freeze:=true`는 **리포 어디에도 구현되어 있지
않다** (`GELLO_UR7E_EEF_MODE.md:382-390`). 넘겨도 아무 효과가 없다.
채널 분리는 **조작자의 손동작으로만** 한다. P7 결과에는 P8의 회전 거동이 이미 섞여 들어온다.

---

## 2. 파라미터 — yaml 기본값과 오버라이드

### 2.1 yaml 기본값 (2026-07-24에 2배로 상향됨)

| 파라미터 | yaml 기본 | 근거 |
|---|---|---|
| `pos_scale` | **1.0** (결정 완료, 재논의 금지) | `config/ur7e_gello_eef.yaml:74` |
| `v_max` | **0.16 m/s** (플랜의 0.08에서 2배) | `:238` |
| `w_max` | **1.0 rad/s** (플랜의 0.5에서 2배) | `:241` |
| `max_excursion_m` | 0.5 | `gello_ur_bridge_node.py:305-306` |

> ⚠️ **정본 문서의 단계표(`GELLO_UR7E_EEF_MODE.md:380`)는 아직 "P9 … yaml 기본 `v_max=0.08`"이라고
> 적혀 있다. 그 줄은 낡았다.** 같은 문서 머리말(`:7`)과 실제 yaml(`:238`)이 0.16이다.
> P9는 **0.16 m/s / 1.0 rad/s**에서 도는 것이 현재 기본이다 — 플랜 문서 숫자보다 2배 빠르다.
> 처음 재현할 때는 이 점을 반드시 의식하고 들어갈 것.

### 2.2 launch 인자로 오버라이드 (재빌드 불필요)

`v_max` / `w_max` / `pos_scale` **세 개만** launch 인자다. 나머지(`tool_*`, `keepout_json`,
`r_align_rpy`, `sigma_*` …)는 yaml을 고치고 `./build_ur7e.sh`로 재빌드해야 한다
(`ur7e_gello_eef.yaml:120`, `GELLO_UR7E_EEF_MODE.md:288`, `:315`).

```bash
# 인자 존재 확인
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py --show-args \
  | grep -E "v_max|w_max|pos_scale|control_mode"
```

- 정수로 써도 안전하다 (`pos_scale:=0` == `pos_scale:=0.0`). launch가 `OpaqueFunction` 안에서
  직접 `float()`로 바꾼다 — 이 리포의 "전부 숫자인 CLI 값이 int로 강제 변환" 함정을 막아준다.
- **인자를 빼면 yaml 값으로 돌아온다.**

### 2.3 🛑 라이브 튜닝 — 파라미터 이름이 정확해야 한다

```bash
ros2 param set /gello_ur_bridge v_max 0.10
ros2 param set /gello_ur_bridge w_max 0.50
ros2 param set /gello_ur_bridge pos_scale 0.5
```

> **이름은 정확히 `v_max` / `w_max`여야 한다.** `eef_v_max` 같은 접두사를 붙이면
> **조용히 무시된다** (에러도 안 난다):
> `NOTE: param name MUST be v_max (maps 1:1 to EefDeltaController cfg key and the node's
> declare_parameter("v_max")); an eef_ prefix is silently dropped.`
> — `config/ur7e_gello_eef.yaml:231-233`

- 라이브로 바꾼 값은 **런치를 다시 띄우면 사라진다.** 영구 반영은 yaml + 재빌드.
- `pos_scale`은 ENGAGED 중에 바꾸면 진행 중 스트로크가 미끄러진다. GUI는 **disengage/engage
  순간에만** 커밋한다 (`ur7e_gello_eef.yaml:200` 근처 주석, GUI A안).
- 설정 후 반드시 확인:

```bash
ros2 param get /gello_ur_bridge v_max
ros2 topic echo --once /gello_ur_bridge/eef/state   # v_max/w_max가 읽기전용으로 표시됨
```

---

## 3. 단계표

> **P9a / P9b는 이 테스트 문서에서 도입한 분할이다.** 정본 문서에는 `P9` 하나뿐이다
> (`GELLO_UR7E_EEF_MODE.md:380`). yaml 기본이 2배로 올라간 뒤 "6-DoF"와 "전속 재클러치
> 워크플로"를 한 번에 하는 것이 부담스러워서 나눴다. 상위 문서와 대조할 때 유의할 것.

| 단계 | 명령 | 무엇을 보는가 | 판정 |
|---|---|---|---|
| **P6** | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef pos_scale:=0.0 v_max:=0.01 w_max:=0.05` | **TCP 위치** 무동작 (팔은 움직임) | §4 |
| **P7** | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.02` → `v_max:=0.05` | 병진 위주. `R_align` 검증 | §5 |
| **P8** | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.05` | 회전 위주. tool 오프셋 검증 | §6 |
| **P9a** | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.08 w_max:=0.5` | 6-DoF 자유 조작 (**플랜의 옛 기본 속도**) | §7 |
| **P9b** | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef` (yaml 기본 0.16 / 1.0) | 6-DoF + 재클러치 워크플로, **현재 기본 속도** | §7 |

> ⚠️ **P6 → P7로 넘어갈 때 `pos_scale:=0.0`을 빼는 것을 잊지 말 것.**
> 안 빼면 로봇이 계속 안 움직이는데 "고장난 줄" 알게 된다.

---

## 4. P6 — TCP 위치 고정

```bash
cd $WT/ros2_ur_ws
HEADLESS=true ./run_ur7e_gello_real.sh \
    control_mode:=eef pos_scale:=0.0 v_max:=0.01 w_max:=0.05
```

**절차:** GUI에서 ENGAGE(두 번 클릭) → GELLO를 **천천히, 작게** 움직인다 → DISENGAGE. 20회 반복.

**판정 (a)~(e) 전부 만족해야 P7):**

| | 항목 | 판정 방법 |
|---|---|---|
| (a) | engage 20회, 매번 zero-jump | `ros2 topic echo /forward_position_controller/commands` — engage 순간 스텝 0 |
| (b) | **TCP 위치 무동작** | `/gello_ur_bridge/eef/state`의 **`excursion_m`이 계속 0**. **팔을 눈으로 보는 것은 판정 기준이 아니다** |
| (c) | protective stop **0회** | 펜던트 / `ur_mode` |
| (d) | **실기 DH 대조** (이 단계에서만 가능) | 아래 §4.1 |
| (e) | 틱 예산 초과 disengage 0회 | `~/eef/state`의 `auto_reason`에 **`tick_budget SUSTAINED`가 없을 것**. SOFT WARN 로그 자체는 무시 (실기 baseline ~1000–1400 µs라 자주 뜨는 게 정상) |

> `excursion_m`이 0이 아니면 **그때가 앵커 로직 버그**다. 위치 수식은 mock P1/P4에서 이미 검증됐다.

### 4.1 (d) 실기 DH 대조

engage 게이트 G7은 **우리 FK와 우리 IK만** 비교하므로 실기 캘리브레이션 불일치를 **못 잡는다.**
그런데 잡아야 할 문제가 실제로 있다: `ur_kin.py`의 DH 상수는 **명목값**이지 이 개체의 공장
캘리브레이션 값이 아니고, `config/ur7e_dh.yaml`을 읽는 `ur_kin.load_dh()`는
**정보용일 뿐 `fk`/`ik`/`jacobian`에 배선되어 있지 않다.**

```bash
ros2 topic list | grep tcp_pose
ros2 topic echo --once /tcp_pose_broadcaster/pose             # 벤더 FK 기준 TCP
ros2 topic echo --once /gello_ur_bridge/eef/commanded_pose    # 우리 FK 기준 TCP
```

- **3개 이상의 서로 다른 자세**에서 비교. 일치 기준 **5 mm / 5 mrad**.
- 어긋나면: 상대 텔레옵에는 대체로 무해하지만(왕복 항등이 우리 DH 안에서 닫힘) **절대 좌표
  정밀도를 기대하면 안 된다.** mm를 크게 넘으면 P7 진행 전 에스컬레이션.
- 이것은 **HIL-SERL에 직접 영향을 준다**: RL env의 `TCP_POSE_SOURCE` 기본이 `"driver"`
  (`config.py:114`)이므로 관측은 **벤더 FK**로 오는데, 명령 경로와 개입 앵커는 **우리 DH**를
  쓴다. 두 계가 다르면 관측과 명령이 서로 다른 좌표계 위에 있게 된다. → `08_OPEN_GAPS.md`

---

## 5. P7 — 병진 위주 (`R_align` 검증)

```bash
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.02
# 문제 없으면 종료 후
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.05
```

- `pos_scale`은 **빼서** yaml 기본 1.0으로 돌아간다.
- **절차:** 리더의 자세(회전)를 최대한 유지한 채 **평행이동만** 준다. 축 하나씩.
- **판정:** TCP가 손과 **같은 축·같은 부호**로 가는가. 축간 크로스토크 없는가.
  기생 회전은 무시한다 (`rot_freeze`가 없으므로 불가피).

| 손 이동 | 기대 TCP 이동 |
|---|---|
| base +X | +X |
| base +Y | +Y |
| base +Z | +Z |

> **부호가 반대로 보인다고 코드에서 뒤집지 말 것.** 오프라인 실측에서 리더 base +X/+Y/+Z →
> 로봇 base +X/+Y/+Z (identity)로 확인되어 있다. 반대로 보이는 원인은 대개 **카메라 시점**이다
> (`serl_ur_infra/RVIZ_HIL_TEST_CLI.md`의 "방향/좌표 주의", 그리고 사용자 메모
> "HIL X/Y flip is camera, not code").

---

## 6. P8 — 회전 위주 (tool 오프셋 검증)

```bash
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.05
```

- **절차:** 리더 그립점을 **한 자리에 고정**한 채 손목만 비튼다.
  손 자체가 이동하면 그 병진은 정상 명령이라 판정이 오염된다 — **손을 고정하는 것이 이 단계의
  핵심 절차**다.
- **판정: 기생 병진 `excursion_m` < 1 cm** (`/gello_ur_bridge/eef/state`).
- 이것이 `tool_l = tool_r = 0.174` 설정의 실증이다. `tool_l`을 0으로 두면 리더 델타가
  **가상 flange 변위**가 되어 로봇 **TCP**에 적용되고, 그 어긋남이 하필 P8에서 처음 드러난다
  (`GELLO_UR7E_EEF_MODE.md:135`, `:148`).

> ### `max_excursion_m`은 **구간 상한**이지 총 이동거리 상한이 아니다
> `max_excursion_m = 0.5`는 `‖p_cmd − p_r_anchor‖`를 본다. engage/재클러치마다
> `p_r_anchor`가 새로 스냅샷되어 **예산이 0으로 초기화**된다
> (`GELLO_UR7E_EEF_MODE.md:68-70`). 즉 클러치를 반복하면 로봇은 얼마든지 멀리 갈 수 있다.
> 이것을 워크스페이스 제한으로 착각하지 말 것.

---

## 7. P9a / P9b — 6-DoF + 재클러치

```bash
# P9a — 플랜의 옛 기본 속도로 먼저
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef v_max:=0.08 w_max:=0.5

# P9b — 현재 yaml 기본 (0.16 / 1.0)
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef
```

**절차 (재클러치 워크플로):**

```
ENGAGE → 펜처럼 이동 → DISENGAGE → GELLO를 편한 자세로 재배치 → ENGAGE (새 앵커)
```

매 engage가 "지금 로봇 자세 + 지금 GELLO"를 새 앵커로 잡는다. 그래서 GUI에는 reclutch/재무장
버튼이 없다 — 토글 하나가 전부 흡수한다.

**판정:**

- 6축 전부 의도대로 따라오는가
- 재클러치 시 zero-jump (engage 순간 팔이 튀지 않는가)
- protective stop 0회
- `~/eef/state`의 `reject_reason` / `auto_reason` 확인:
  - `STEP_FLOOR` HOLD가 늘어나면 → `v_max`를 너무 올려서 **관절 슬루 상한
    `max_step_rad`(0.0025 rad/tick @250 Hz = 0.625 rad/s/joint)**가 binding limit이 된 것.
    더 빠르게 하려면 `publish_rate_hz`를 500으로 먼저 올려야 한다
    (`GELLO_UR7E_EEF_MODE.md:198`).
  - 특이점 근처의 자동 감속(`gamma`)은 **정상**이다.

---

## 8. 복구

```bash
source $WT/ros2_ur_ws/remote_helpers.sh
ur_unlock     # protective stop 해제 — 원인 제거 후에!
ur_resend     # Method B 리버스 인터페이스 재전송
```

EEF 브리지 자체의 fault 복구는 `~/eef_resume`(실제 자세에서 재시드하여 HOLD로,
정렬 게이트 없음) 후 `~/eef_engage`다 (`gello_ur_bridge_node.py:43-44`).
GUI의 큰 토글은 DISENGAGED에서 이 체인을 자동으로 연결해준다.
