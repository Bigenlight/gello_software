# GELLO → UR7e **EEF(카테시안) 델타 텔레오퍼레이션 모드** — 설계·구현 계획서 (PLAN)

> 📌 **문서 상태 — 2026-07-22 갱신. 이 문서는 이제 "계획"이 아니라 설계·근거 아카이브다.**
> 코드는 **이미 구현되어 리포에 들어와 있다**(커밋 `2905925` "feat(ur_gello_bringup): add EEF (Cartesian delta) teleop mode alongside joint mode"): `ur_gello_bringup/ur_kin.py`, `ur_gello_bringup/eef_delta.py`, `gello_ur_bridge_node.py`의 eef 분기 + 4종 Trigger 서비스, `config/ur7e_gello_eef.yaml`, `launch/ur7e_gello_eef_mock.launch.py`, `test/test_ur_kin.py` · `test/test_eef_delta.py`, `scripts/eef_replay.py`, 조작자 콘솔 메뉴 7~10번이 전부 존재한다. **mock 단계(P4/P5)까지 검증 완료, 실기 첫 투입은 아직**이다. 여기 적힌 수치·임계값은 여전히 대부분 **미검증 초기값**이며 실기 튜닝을 전제한다.
>
> ▶ **조작자는 이 문서가 아니라 실행 런북 [`GELLO_UR7E_EEF_MODE.md`](GELLO_UR7E_EEF_MODE.md)를 따른다.** 두 문서가 어긋나면 **런북이 최신이자 우선**이다. 이 문서에서 런북에 의해 대체된 서술은 본문에 `⛔ SUPERSEDED` 로 표시하고 원래 논거는 남겨 두었다(왜 그렇게 판단했는지가 나중에 다시 필요하기 때문).
>
> **안전 대전제 (변함없음, 절대 위반 불가)**: GELLO는 **항상 passive read-only** 입력 장치입니다. GELLO Dynamixel에는 **어떤 경우에도 토크를 인가하지 않습니다.** 이 계획서의 어떤 변경도 이 불변식을 건드리지 않습니다 — EEF 모드는 GELLO 관절각을 **읽는 방식**만 바꿉니다.
>
> **스코프 못박기.** 이 계획서의 범위는 **"기존 joint 미러링 모드 옆에 EEF 텔레오퍼 모드를 하나 추가한다"** 까지입니다. online RL 학습 루프, actor/learner, replay buffer, reward, 정책↔사람 자동 arbiter, intervention 자동 라벨링, `gello_policy` 통합은 **이번 범위 밖**이며 설계하지 않습니다(§12에 훅 위치만 한 문단 언급).
>
> ### 🔴 2026-07-22 프레이밍 전환 — **"GELLO = 3D 펜"** (이 문서 전반에 영향)
>
> 이 계획서는 원래 **"EEF 모드는 joint 모드 거동을 재현해야 한다"** 는 암묵 전제 위에 쓰였다(engage 시 `q_lead ≈ q_robot`, `R_align = I`, `tool_l = tool_r`이 전부 그 전제에서 따라왔다). **사용자 의도는 이제 명시적으로 다르다: 리더와 로봇은 영구히 다른 관절 자세로 있고, 오직 EEF 델타만 전달된다.**
>
> - **수식은 이미 준비되어 있었다.** 델타 사상의 **자세 무관성이 증명**되었다 — `test/test_eef_pose_independence.py`의 **22개 신규 테스트**(현재 전체 스위트 **342개**), 관절 불일치 **최대 5.62 rad**에서 최악 오차 **8.8e-13 m / 2.7e-13 rad**, 불일치 크기에 대한 **추세 없음**. 거부(reject)는 리더 불일치가 아니라 **로봇 쪽 조건수 `sigma_min`** 을 따라간다. 즉 **3D 펜 요구사항은 수학 차원에서 이미 충족**되어 있었고, 손봐야 했던 것은 **기동 경로**뿐이었다(§4.2 SUPERSEDED 박스).
> - **폐기되는 논거들**: engage 시 `q_lead ≈ q_robot`(§4.2 / MODE §1.1), `R_align = I`가 유일한 정답(§3.3 박스), P7/P8의 채널 분리(§8 P7 박스).
> - **아직 열려 있는 것**: `r_align_rpy`가 **측정된 적이 없다** — 3D 펜 프레이밍에서 이것은 잠재 버그이며 EEF 이동 **방향**의 정확도가 여기 걸려 있다([MODE.md §1.6](GELLO_UR7E_EEF_MODE.md)에 로봇 없이 하는 측정 절차).

## 요약 (TL;DR)

| 항목 | 결정 / 값 |
|---|---|
| 무엇을 만드나 | `gello_ur_bridge`에 `control_mode: "joint" \| "eef"` 파라미터 추가. **eef 모드에서만** 클러치-앵커 기반 EEF 델타 제어가 동작 |
| 새 노드 | **없음.** `/forward_position_controller/commands`의 단일 소유자는 계속 `gello_ur_bridge` 하나 (dual-writer 원천 차단) |
| 새 순수 모듈 | `ur_kin.py`(FK/Jacobian/IK), `eef_delta.py`(앵커·델타·거버너·게이트) — **rclpy 비의존**, pytest로 실기 없이 전량 검증 |
| 리더 EEF pose | GELLO 관절을 **UR7e DH에 그대로 넣어** FK. GELLO URDF/CAD가 리포에 없고, 필요도 없다 (§3.1) |
| 제어 방식 | **앵커(clutch) 대비 절대 pose 서보** + SE(3) 레퍼런스 거버너(Cartesian 속도 상한 × 특이점 감쇠) → 해석적 IK → 기존 slew clamp |
| clutch 트리거 | `~/eef_engage` / `~/eef_disengage` / `~/eef_reclutch` (`std_srvs/Trigger`), 조작자 콘솔 메뉴에서 호출 |
| zero-jump | engage 시 델타가 **항등적으로 0** → 첫 명령 = 직전 발행 명령 (수학적 보장, §4.4) |
| disengage | **기존 `~/pause`와 동일한 즉시 정지 경로 재사용** — 다음 250 Hz 틱에 발행 중단 |
| 안전 | bounding box 없음. 대신 (a) Cartesian 속도 상한, (b) 특이점 감쇠, (c) 브랜치 잠금 + 점프 거부, (d) 관절 리밋 마진, (e) keep-out 기하 게이트, (f) 기존 slew clamp 백스톱 |
| 최대 리스크 | 충돌 비인지. EEF 모드는 관절 형상 선택권을 IK에 넘기므로 **joint 모드보다 충돌 위험이 높다** (§6.5) |
| 환경 주의 | 이 개발 PC는 **Jazzy / Ubuntu 24.04 / Python 3.12** 이다(브랜치명 `...humble-22.04`와 불일치). 실기 PC의 distro·python은 **P-1에서 실측 확인 필수** |

### 목차

1. [목적과 배경](#1-목적과-배경)
2. [설계 개요와 아키텍처](#2-설계-개요와-아키텍처)
3. [핵심 수식](#3-핵심-수식)
4. [모드 선택 및 clutch 프로토콜](#4-모드-선택-및-clutch-프로토콜)
5. [IK 전략](#5-ik-전략)
6. [안전 설계](#6-안전-설계)
7. [변경/신규 파일 목록](#7-변경신규-파일-목록)
8. [단계별 구현 계획](#8-단계별-구현-계획)
9. [디버깅 · 관측 수단](#9-디버깅--관측-수단)
10. [미해결 질문 / 사용자 결정 필요 사항](#10-미해결-질문--사용자-결정-필요-사항)
11. [기각한 대안과 이유](#11-기각한-대안과-이유)
12. [향후 확장 지점 (구현하지 않음)](#12-향후-확장-지점-구현하지-않음)

---

## 1. 목적과 배경

### 1.1 현재 상태 — 절대 joint 미러링

지금 파이프라인은 GELLO 관절각을 UR7e 관절각으로 **절대값 1:1 미러링**한다. 코드상 이 매핑은 `gello_ur_bridge_node.py`의 단 한 줄이다:

```python
raw_target = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
```

즉 GELLO 관절 `q_g`가 그대로 로봇 관절 목표 `q_r`이 된다. 이 방식은 **GELLO와 로봇이 항상 같은 자세를 유지한다**는 전제 위에서만 성립하고, 실제로 그 전제를 지키기 위해 시스템 전체가 상당한 비용을 지불하고 있다:

- `gello_move_to_start`의 **수렴 게이트 chase 핸드셰이크** — 스트리밍 시작 전에 로봇을 살아있는 GELLO 자세로 데려간다.
- `~/resume`의 **정렬 게이트** (`resume_align_tol = 0.08 rad`) — pause 후 재개하려면 GELLO와 로봇이 이미 정렬되어 있어야 한다.
- `~/resume_chase`의 **완화 게이트** (`resume_chase_max_gap = 1.5 rad`) — 더 큰 갭은 리더가 quasi-still일 때만 rate-limited glide로 닫는다.

이 세 장치는 전부 **"두 자세가 어긋나면 위험하다"**는 같은 문제를 다른 각도에서 막고 있다.

### 1.2 왜 절대 미러링으로는 부족한가

GELLO는 **passive**다. 토크를 걸 수 없으므로 **로봇 자세로 되돌려 놓을 방법이 없다.** 이 사실이 만드는 실사용 제약:

1. **작업 범위 끝 문제.** GELLO 팔을 어느 방향으로 끝까지 뻗으면 더 못 간다. 로봇은 아직 여유가 있는데 리더가 한계다. joint 모드에서는 "리더를 되잡고 이어가기"가 구조적으로 불가능하다 — 리더를 놓는 순간 중력으로 처지고, 다시 잡으면 절대각이 달라져 로봇이 그리로 점프한다.
2. **자세 어긋남 = 즉시 위험.** 어떤 이유로든(pause 중 리더 이동, 스트림 끊김, 사람이 리더를 놓았다 잡음) 두 자세가 벌어지면, 다시 붙이는 유일한 수단이 "로봇을 리더 자세로 이동시키는 것"이다. 이건 `resume_chase`가 이미 하고 있지만, 최대 1.5 rad를 **충돌 인지 없이** 자율 활강하는 동작이라 본질적으로 조심스러운 조작이다.
3. **조작 좌표계가 관절이다.** "TCP를 앞으로 5 cm" 같은 카테시안 의도를 관절로 번역하는 일을 사람이 손으로 한다.

### 1.3 EEF 델타 모드가 푸는 것

**제어권을 잡는 순간(engage)을 앵커(clutch)로 삼아** GELLO EEF pose와 로봇 EEF pose를 각각 스냅샷하고, 이후로는 **GELLO EEF의 앵커 대비 델타를 로봇 EEF 앵커에 적용**한다.

- 사람은 GELLO를 **아무 자세로나** 잡아도 된다. engage 순간 델타가 정의상 0이므로 로봇은 **미동도 하지 않는다**.
- 리더가 가동범위 끝에 닿으면 **클러치를 떼고(disengage) → GELLO를 편한 자세로 되돌리고 → 다시 잡는다(re-clutch)**. 로봇은 그 동안 정지해 있고, 재클러치 후 이어서 조작한다. 이것이 UR7e 전체 가동범위를 리더 가동범위의 제약 없이 커버하는 방법이다.
- 조작 좌표계가 카테시안이 되어 "손을 앞으로 밀면 TCP가 앞으로" 라는 직관이 성립한다.

**중요 — 이 모드는 joint 모드를 대체하지 않는다.** joint 모드는 검증된 경로이고, EEF 모드는 그 옆에 추가되는 선택지다. §7에 joint 모드 무회귀를 어떻게 보장하는지 명시한다.

---

## 2. 설계 개요와 아키텍처

### 2.1 핵심 아키텍처 결정 3가지

**(D1) 새 퍼블리셔 노드를 만들지 않는다.** `/forward_position_controller/commands`의 소유자는 계속 `gello_ur_bridge` 하나다. 이유: 이 리포에는 이미 `/gello/joint_states`에 `gello_publisher_node`와 `policy_leader_node`가 동시에 붙을 수 있는 dual-writer 문제가 **코드 레벨 배타 장치 없이** 남아 있다(런치 구성으로만 배타). 명령 토픽에서 같은 실수를 반복하지 않는다.

**(D2) EEF 수학은 rclpy 비의존 순수 모듈로 분리한다.** `ur_kin.py`(기구학)와 `eef_delta.py`(앵커/델타/거버너)는 ROS를 모른다. 브리지는 이 모듈을 호출만 한다. 이렇게 하면 실기·ROS 없이 pytest로 수식 전체를 검증할 수 있고, 실기 디버깅 시 "수식 문제 / 배선 문제"를 단계적으로 분리할 수 있다(§8).

**(D3) 훅 지점은 `_on_timer`의 필터 뒤·slew clamp 앞 딱 한 곳이다.** 그 아래의 slew clamp / soft-start / staleness watchdog / seed 브랜치 / pause·resume은 그대로 상속된다.

> **주의 — "코드를 한 줄도 안 건드린다"는 주장은 하지 않는다.** 실제 코드에서 1-Euro 필터 적용과 slew clamp는 **같은 per-joint 루프 안에 인터리브**되어 있다(`_on_timer`). EEF stage는 6관절 벡터를 한꺼번에 받아야 하므로 이 루프를 **필터 루프 / 클램프 루프 두 개로 쪼개야 한다.** 관절별 독립 연산이라 joint 모드의 부동소수 결과는 동일하지만, 그건 **주장이 아니라 P3에서 비트 비교로 증명할 사항**이다.

### 2.2 노드/토픽/서비스 다이어그램

```
 [GELLO Dynamixel]   passive, torque OFF — 불변식
        │
   gello_publisher_node  (30 Hz)
        ├── /gello/joint_states                                   (JointState, 6 arm)
        └── /gripper/gripper_client/target_gripper_width_percent   (Float32, 0..1)
                                    │
                                    │   ※ 그리퍼 경로는 EEF 모드에서도 그대로 통과.
                                    │      gello_gripper_bridge → robotiq_* 체인 무변경.
                                    ▼
                              gello_gripper_bridge → /robotiq_gripper/...
        │
        ▼
┌───────────────────────── gello_ur_bridge_node ──────────────────────────────┐
│                                                                              │
│  _on_joint_state()  [30 Hz, GELLO 콜백]           ※ 무변경                    │
│    이름 매칭 재정렬 → wrapped_nearest 연속 unwrap                             │
│    → self._raw_target      (= 리더 관절, 두 모드 공통 유일 진실)              │
│    → self._gello_history   (quasi-still 게이트용)                             │
│                                                                              │
│  _on_actual_joint_state()  [로봇 실측]             ※ 무변경                   │
│    → self._actual_pose                                                       │
│                                                                              │
│  _on_timer()  [250 Hz]                                                       │
│    ├ paused? → return                        ※ 무변경 (disengage가 이 경로 재사용)│
│    ├ staleness watchdog → 재시드 예약          ※ +EEF 앵커 무효화 1줄           │
│    ├ seed 브랜치(actual pose 시드)             ※ 무변경                        │
│    │                                                                          │
│    ├─ [루프 A] 필터  ── control_mode 분기 ─────────────────────────────────    │
│    │    joint: filtered[i] = euro[i](raw_target[i])        (기존과 동일)       │
│    │    eef  : q_lead_f = euro_lead(raw_target)   ← EEF 전용 필터 인스턴스      │
│    │           filtered  = eef_stage(q_lead_f, step_eff)   ← 신규 (아래)       │
│    │                                                                          │
│    └─ [루프 B] slew clamp + soft_start 램프    ※ 로직 무변경, 루프만 분리       │
│         → self._last_published → _publish()                                   │
│                                                                              │
│  서비스 (기존):  ~/pause  ~/resume  ~/resume_chase                            │
│  서비스 (신규):  ~/eef_engage  ~/eef_disengage  ~/eef_reclutch  ~/eef_to_joint │
│  토픽  (기존):  ~/state                                                       │
│  토픽  (신규):  ~/eef/state (String/JSON, 10 Hz)                              │
│                 ~/eef/leader_pose  ~/eef/desired_pose  ~/eef/commanded_pose   │
│                                        (PoseStamped, 10~30 Hz 데시메이트)      │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
              /forward_position_controller/commands  (Float64MultiArray, 250 Hz)
                                   ▼
                      forward_position_controller → UR7e (500 Hz servo)
```

### 2.3 `eef_stage` 내부 (순수 함수, 30 Hz 상당 정보량이지만 250 Hz로 호출)

```
eef_stage(q_lead_f, step_eff):
  T_g   = FK(q_lead_f) · T_tool_L                       리더 EEF pose (§3.2)
  T_des = anchor_apply(T_g)                             앵커 대비 델타 적용 (§3.3)
  γ     = sing_gain(σ_min)  (비대칭: 탈출 방향은 감쇠 면제, §6.3)
  ξ     = clamp_se3(log(T_cmd⁻¹·T_des), γ·v_max·dt, γ·ω_max·dt)
  s     = analytic_scale(ξ, step_eff)                   해석적 배율 (§5.4)
  T_try = T_cmd · exp(s·ξ)
  q_try = IK_branchlock(T_try, seed=q_ik_prev)
  수용 테스트: 해 존재 ∧ 브랜치 유지 ∧ 관절리밋 마진 ∧ |Δq|≤step_eff ∧ keep-out
  수용 → T_cmd ← T_try ; q_ik_prev ← q_try ; return q_try
  거부 → HOLD: return q_ik_prev (T_cmd 동결) + reject_reason 발행 + lag anti-windup
```

### 2.4 두 모드의 공존 구조

| | joint 모드 | eef 모드 |
|---|---|---|
| `control_mode` | `"joint"` (기본값) | `"eef"` |
| 리더 → 명령 | 절대 관절 미러링 | 앵커 대비 EEF 델타 → IK |
| 시작 시퀀스 | `gello_move_to_start` 수렴 게이트 chase → STRICT 전환 → `~/resume` | **동일** (아래 참고) |
| 사람이 로봇을 멈추는 법 | `~/pause` | `~/eef_disengage` (내부적으로 `~/pause`와 동일 경로) |
| GELLO/로봇 자세 관계 | 항상 일치해야 함 | 임의로 어긋나도 됨 |

**부트스트랩은 두 모드가 공유한다.** `gello_move_to_start`는 `/gello/joint_states`를 그대로 구독하고 EEF 모드를 전혀 모른다. eef 모드에서도 **핸드셰이크 구간은 joint 모드처럼 동작**한다 — 브리지는 `control_mode:="eef"`로 떠 있어도 `_eef_engaged == False`인 동안은 **joint 패스스루로 동작**한다(즉 `filtered = euro(raw_target)`). 핸드셰이크가 끝나 로봇이 리더를 따라가고 있는 상태에서 조작자가 `~/eef_engage`를 부르면 그때 EEF 델타 제어로 전환된다. 이 순간 GELLO와 로봇은 이미 정렬돼 있으므로 앵커 델타 0이 자명하게 성립한다.

> **왜 이렇게 하나:** 상류 별도 노드(`gello_eef_mapper`)로 분리하는 안을 검토했으나, 그러면 `gello_move_to_start`가 EEF 노드의 출력을 기다리게 되어 **부트스트랩 데드락**이 생긴다(EEF 노드는 engage 전 침묵 → move_to_start가 첫 메시지를 영원히 대기 → 브리지가 PAUSED에서 못 나옴 → engage 게이트가 PAUSED를 이유로 거부). §11 참고.

---

## 3. 핵심 수식

표기: `T = (R, p)`, `R ∈ SO(3)`, `p ∈ R³`, 4×4 동차변환. 모든 pose는 **UR `base_link` 기준**.

> **회전은 3×3 행렬(또는 부호 연속성이 강제된 쿼터니언)로만 다룬다.** 쿼터니언 컨벤션 불일치(ROS는 scalar-last `xyzw`, MuJoCo/hil-rl은 scalar-first `wxyz`)와 axis-angle log map의 `θ∈[0,π]` 되접힘을 **제어 경로에서 원천 제거**하기 위함이다. 쿼터니언 변환은 오직 진단 토픽 발행에서만 한다.

### 3.1 리더 EEF pose — GELLO 기구학 모델을 만들지 않는다

리포 전체를 검색한 결과 **GELLO의 URDF / DH / CAD / 링크 길이 데이터가 존재하지 않는다.** 존재하는 URDF 3개는 전부 무관하다(mujoco_menagerie UR5e, factr yam, franka xacro).

그런데 **만들 필요가 없다.** GELLO 관절은 `DynamixelRobot.get_joint_state()`에서
```
q_g = (ticks_to_rad(raw) − joint_offsets) ⊙ joint_signs
```
로 보정되며, 이 보정의 **정의 자체가 "q_g를 UR 관절 규약에 1:1로 맞춘다"**이다. 현행 joint 모드가 실기에서 동작한다는 사실이 이 등가성의 실증이다.

따라서 리더 EEF를 다음과 같이 **정의**한다:

```
T_g(q_g) := FK_UR7e(q_g) · T_tool_L      "GELLO 자세를 미러링한 가상 UR7e의 그립점"
T_r(q_r) := FK_UR7e(q_r) · T_tool_R      로봇 TCP (동일 FK 함수)
```

**성질 1 — 리더·로봇에 같은 FK를 쓰므로 기구학 모델 오차가 앵커 델타에서 상쇄된다.** `T_des = T_r_anchor` (델타 0)일 때 `IK(FK(q_anchor)) = q_anchor`가 **항등적으로** 성립한다(같은 DH 사용). 벤더 FK(`/tcp_pose_broadcaster/pose`)를 앵커로 쓰면 우리 IK와 기구학 모델이 달라 왕복 잔차가 그대로 engage 스텝이 된다 — 그래서 앵커 소스로 쓰지 않는다.

**성질 2 — 리더 변위는 이미 "풀사이즈 UR7e 공간"의 변위다.** "GELLO는 UR7e의 균일 축소 모델"이라는 배경 전제가 참이라면, 사람 손의 **물리** 변위는 `Δp_phys = k · Δp_g`이다(`k` = 축소비, `Δp_g`는 위 정의의 가상-UR 공간 변위). 즉 `pos_scale = 1.0`이면 **손 물리 변위 → 로봇 변위 게인이 1이 아니라 1/k**(예 `k≈0.4`면 2.5배)가 된다.

> ⛔ **SUPERSEDED (2026-07-22).** 아래 원본 대응책 (a)(b)는 **폐기**한다. 현행 결정은 **`pos_scale = 1.0` 유지, `k` 실측 불필요**이며, 근거와 운영 지침은 [`GELLO_UR7E_EEF_MODE.md` §1.2](GELLO_UR7E_EEF_MODE.md)가 정본이다.
>
> **한 줄 이유:** 리더 EEF는 GELLO 관절각을 **풀사이즈 UR7e FK**에 넣어 정의된다(`T_g = fk(q_lead_f) @ T_tool_L`, `eef_delta.py` 모듈 docstring). 따라서 `Δp_g`는 처음부터 풀사이즈 UR7e 공간의 변위이고, `pos_scale=1.0`은 **joint 모드의 EEF 이동량을 그대로 재현**한다 — EEF 모드를 joint 모드와 등가로 두는 값이 곧 1.0이다.
>
> **"손 5 cm ↔ 로봇 12.5 cm"는 여전히 물리적으로 참이다.** 다만 그것은 버그가 아니라 **joint 모드에서 이미 매일 겪고 있는, 축소 리더의 정상적이고 익숙한 성질**이다(작은 GELLO를 조금 움직이면 큰 로봇이 그 비율만큼 움직인다). EEF 모드가 새로 만들어내는 위험이 아니므로 "안전 이슈"로 분류하지 않는다. `pos_scale`을 `k`로 낮추면 오히려 **joint 모드보다 둔해져** 조작 감각이 두 모드 사이에서 달라진다.
>
> **살아남는 대응책:** (c) `1.0` 이외 값은 "의도적 증폭/감쇠"로 취급하고 기동 시 로그에 남긴다. (d) engage 이후 누적 변위 `‖p_cmd − p_r_anchor‖`에 상한(`max_excursion_m`, 기본 0.5 m)을 걸어 게인 오설정이 **속도가 아니라 거리로도** 막히게 한다 — 이건 `pos_scale`을 어떤 값으로 두든 유효한 백스톱이라 그대로 유지한다.

<details><summary>원본 서술 (기록 보존용, 더 이상 따르지 말 것)</summary>

> ❗ **이건 안전 이슈다.** 조작자가 손을 5 cm 뻗었는데 로봇이 12.5 cm 가는 상황은 각 틱 증분이 `v_max` 이하라서 **어떤 거부/홀드도 발동하지 않는다.** 속도 제한은 게인 오설정을 막지 못한다.
>
> **대응:** (a) `k`를 P-1에서 **1회 실측**한다(§8). (b) `pos_scale` 기본값을 `k`로 둔다(= 손 물리 변위 : 로봇 변위 = 1:1). (c) `1.0` 이외 값은 "의도적 증폭"으로 기동 시 경고 로그. (d) engage 이후 누적 변위 `‖p_cmd − p_r_anchor‖`에 상한(`max_excursion_m`, 기본 0.5 m)을 걸어 게인 오설정이 **속도가 아니라 거리로도** 막히게 한다.

</details>

**성질 3 — tool 오프셋을 반드시 명시해야 한다.** 회전 중심 문제 때문이다. 사람이 그립점 `c`를 중심으로 `ΔR` 회전시키면 flange 변위는 `Δp = (I−ΔR)·d`, `d = c − p_flange`이다. 리더와 로봇의 `d`가 다르면(앵커 자세가 어긋나 있으면 반드시 다르다) **순수 회전 입력에 기생 병진**이 붙는다. 최악의 경우 손목만 30° 비틀었는데 TCP가 10 cm 이상 예상 밖 방향으로 이동할 수 있고, 증상이 `R_align` 오정합과 구분 불가하다.

`T_tool_L`과 `T_tool_R`을 명시적으로 도입하면 회전 중심이 각자의 tool 원점에 걸려 `d = d_r = 0`이 되고 기생항이 소멸한다.

> ⛔ **SUPERSEDED (2026-07-22) — `T_tool_L`의 해석.** 위 원문은 `T_tool_L`을 "GELLO 손잡이 그립점(가상-UR 스케일 = 물리 오프셋 / `k`)"으로 정의했다. **이 해석은 채택하지 않는다.** 정본은 [MODE.md §1.1](GELLO_UR7E_EEF_MODE.md).
>
> **이유:** 리더 EEF는 GELLO 관절각을 **로봇과 동일한 UR7e `fk()`** 에 넣어 만든다(`T_g = fk(q_lead_f) @ T_tool_L`, `eef_delta.py`의 `step()`). GELLO 자체의 링크 기하는 코드 어디에도 없다. 따라서 `tool_l_xyz_rpy`는 **물리 GELLO flange가 아니라 "가상 풀사이즈 UR7e flange" 기준** 오프셋이다. 사람의 물리적 그립점을 이 파라미터로 표현하려면 먼저 `k`로 나눠 가상-UR 공간으로 환산해야 하는데, **`k` 실측은 생략하기로 결정된 항목**이다(§3.1 성질 2 SUPERSEDED, P-1 5번). 즉 그립점 해석은 "타이핑만 하면 되는 다른 규약"이 아니라 **하지 않기로 한 측정을 전제로 하는 방식**이다.
>
> **현행 설정(실측 반영):** `tool_l_xyz_rpy = tool_r_xyz_rpy = [0, 0, 0.174, 0, 0, 0]` — Robotiq 2F-85 끝점 flange +Z **174 mm** 실측, 펜던트 TCP `TCP_2f85`와 동일. 의미는 **"가상 리더 로봇도 실제 로봇과 같은 그리퍼를 단다"** 이고, 그 결과 리더=팔로워가 같은 로봇이 되어 델타 사상이 항등으로 환원된다. engage 시 `q_lead ≈ q_robot`(BOOTSTRAP joint 패스스루 + G4 `anchor_agree_tol = 0.02 rad`)이므로 **EEF 모드가 joint 모드 거동을 그대로 재현**한다.
>
> **참고 (수식):** `T_tool_L`의 **회전 성분은 델타에서 항등적으로 상쇄**된다(`R_g @ R_g_anchor.T`에서 `R_toolL @ R_toolL.T = I`, `eef_delta.py`의 `step()`). 효과가 있는 것은 **병진 성분뿐이며, 그마저도 리더가 회전할 때만** 나타난다. → **P6/P7(무동작·순수 병진)은 tool 값과 무관하게 진행 가능**하고, 검증 단계는 P8이다.
>
> **함정:** `tool_l = 0`, `tool_r = 0.174`로 두면 리더 델타가 **가상 flange 변위**인데 그것을 **로봇 TCP**에 적용하게 되어, 회전이 들어가는 순간(P8) 거동이 joint 모드에서 벗어난다.

**DH 파라미터 소스.** 이 PC의 `/opt/ros/jazzy/share/ur_description/config/ur7e/default_kinematics.yaml`을 확인했고 UR5e와 **완전히 동일**하다(`d1=0.1625, a2=-0.425, a3=-0.3922, d4=0.1333, d5=0.0997, d6=0.0996`). 다만 실기 PC의 distro가 다를 수 있으므로 **이 값을 리포의 `config/ur7e_dh.yaml`에 상수로 커밋**하고 배포판 의존을 없앤다.

### 3.2 앵커(clutch) 스냅샷 — engage 시각 `t0`, 단 1회

```
q_anchor   := self._last_published            명령 체인 기준 (실측 아님, 아래 근거)
T_r_anchor := FK(q_anchor) · T_tool_R
T_g_anchor := FK(q_lead_f(t0)) · T_tool_L     EEF 전용 필터 출력 기준
T_cmd      ← T_r_anchor
q_ik_prev  ← q_anchor
branch0    ← branch_id(q_anchor)
```

**왜 `_last_published`(명령 체인)인가.** zero-jump의 실질적 정의는 "**다음에 발행할 명령이 직전에 발행한 명령과 같다**"이다. `forward_position_controller`는 보간 없이 명령을 즉시 적용하므로, 명령 체인이 연속이면 로봇은 움직이지 않는다. 실측(`_actual_pose`)을 앵커로 잡으면 추종 오차만큼 명령이 한 사이클에 역방향 계단을 만들 수 있다(추종오차 0.05 rad → 0.05/0.002s = 25 rad/s → protective stop).

**단, 두 체인이 벌어진 상태에서의 engage는 거부한다.** engage 게이트에 `max_i |_last_published[i] − _actual_pose[i]| ≤ anchor_agree_tol`(기본 0.02 rad)을 넣는다. 이미 벌어져 있으면 어느 쪽을 골라도 물리 점프가 생기기 때문이다(예: pause 중 조작자가 펜던트 freedrive로 팔을 옮긴 경우).

**왜 `q_lead_f`(EEF 전용 필터 출력)인가, 그리고 왜 `_filtered`를 쓰면 안 되는가.**

> ❗ **함정.** `_filtered`는 리더 추정기가 아니라 **명령 체인 상태**다. seed 브랜치가 `self._filtered = list(self._actual_pose)`와 `self._euro[i].seed(self._actual_pose[i])`로 이 변수를 **로봇 실측 자세**로 시드한다. joint 모드에서는 이게 정확히 맞는 설계(zero-jump 시드)지만, EEF 모드에서 `_filtered`를 리더 pose로 쓰면 seed 직후 구간의 "리더 pose"가 실제로는 **로봇 pose**가 된다. 1-Euro가 진짜 GELLO 값으로 수렴하는 데 정지 시 `τ ≈ 1/(2π·1.0Hz) ≈ 159 ms`, 실질 정착 0.5~1 s가 걸린다. 그 사이 `T_g`가 **사람이 아무것도 안 했는데 혼자 이동**하고, 거버너는 이를 "사람이 손을 움직였다"로 해석해 로봇을 수십 cm 스윕시킨다. `leader_quasi_still` 게이트는 `_gello_history`(원시 리더)만 보므로 이걸 **전혀 못 잡는다**.
>
> **해소:** EEF 모드는 `_raw_target`(리더 관절, 항상 GELLO에서 옴)을 입력으로 하는 **전용 1-Euro 인스턴스 `_euro_lead`**를 쓰고, 이 필터는 항상 `_raw_target`으로 시드한다. 추가로 engage 게이트에 **필터 수렴 검사** `max_i |q_lead_f[i] − _raw_target[i]| < filter_settled_tol`(기본 0.005 rad)을 넣어 수렴을 **양성 증명**한다.

### 3.3 델타 적용 — 곱 순서와 좌표계 명시

`R_align` = GELLO 베이스 → UR `base_link` 정렬 회전(순수 회전, 기본 `I`, yaml에 RPY로 노출).

> ### ⚠️ 추가 (2026-07-22) — 이 정의는 **여기서만 맞게 적혀 있었다**
>
> `config/ur7e_gello_eef.yaml`의 주석은 이것을 *"리더 **손목** 프레임 ↔ UR **툴** 프레임 오프셋"* 이라고 잘못 설명하고 있었다(정정 완료). 위 문장이 옳다: **베이스 대 베이스** 회전 `R_align = R_{UR base_link ← 가상 리더 base_link}` 이다.
>
> **그리고 기본값 `I`는 측정된 적이 없다.** `I`는 "GELLO 베이스 프레임 = UR base_link 프레임"이라는 **단언**이며, `scripts/gello_get_offset.py`가 관절 오프셋을 **π/2 배수로 스냅**하므로(`np.linspace(-8π, 8π, 33)`) 베이스 관절에 **0/90/180/270° 4중 모호성**이 남아 있다.
>
> **옛 프레이밍에서는 `I`가 유일하게 옳은 값이었다** — "EEF는 joint 모드를 재현해야 한다"는 전제 하에서는 양쪽 관절이 같으니 베이스도 같다는 결론이 자동이었다. **"GELLO = 3D 펜" 프레이밍은 그 논거를 폐기한다.** 이제 측정해야 한다.
>
> 수치 확인: `r_align_rpy`를 바꾸면 로봇 TCP 변위의 **방향만 회전하고 크기는 보존**되며, 이 성질은 **로봇 자세와 무관**하다. 증상이 고약한 이유가 여기 있다 — **10~20° 오차는 "팔이 좀 비딱하다"로 합리화되고, 90° 오차라야 명백해진다.** 요구 정밀도는 **≤ 5°**.
>
> **로봇 없이 할 수 있는 측정 절차의 정본은 [MODE.md §1.6](GELLO_UR7E_EEF_MODE.md)** 이다. 핵심 항등식: 두 팔이 **같은 관절 벡터**에 있을 때 `ψ = (GELLO 세계 방위각) − (UR 세계 방위각)`이고 **자세 의존 항이 정확히 상쇄**되므로, 어떤 자세를 골라도 결과가 같다.

```
─ 회전 (world / left multiply) ───────────────────────────────
  R_delta = R_align · ( R_g(t) · R_g_anchorᵀ ) · R_alignᵀ
  R_des   = R_delta · R_r_anchor

─ 위치 ───────────────────────────────────────────────────────
  p_des   = p_r_anchor + pos_scale · R_align · ( p_g(t) − p_g_anchor )

  T_des   = (R_des, p_des)
```

- `R_g(t) · R_g_anchorᵀ` 이지 `R_g_anchorᵀ · R_g(t)`가 **아니다.** 후자는 body-frame 델타이며 `R_r_anchor`에 **우측**으로 곱해야 짝이 맞는다. world/left를 택하는 이유: 사람은 GELLO를 "세상 축" 기준으로 비튼다고 지각한다.
- `R_align` 켤레변환(`R_align · R · R_alignᵀ`)이 맞다. 리더 프레임의 회전을 로봇 프레임으로 옮기는 연산이다.
- **회전 스케일(`rot_scale`)은 제공하지 않는다.** 회전은 무차원이라 1:1이 정답이고, 게인을 넣으려면 `exp(k·log(R))`이 필요한데 그 순간 log map의 `θ` 되접힘이 부활한다. 파라미터로 1.0 이외 값을 주면 **노드 기동 거부**.

> **정정 — ">π 회전은 표현 불가"는 사실이 아니다.** `R_delta`는 회전행렬이므로 어떤 각도에서도 well-defined하고 연속이다. log를 취하는 대상은 오직 **증분** `ξ = log(T_cmd⁻¹T_des)`이고, `T_cmd`가 `T_des`를 연속 추종하므로 매 틱 `|ξ_ω| ≤ ω_max·dt = 0.002 rad ≪ π`라 double-cover 문제가 발생하지 않는다. 오히려 `R_delta`를 rotvec으로 바꿔 `θ`를 `[0,π]`로 캡하는 코드를 넣으면 `θ`가 π를 지날 때 **축이 부호 반전해 인공 불연속이 주입**된다(로봇이 반대 방향으로 최대 속도 선회). **캡 코드를 넣지 않고, "0→360° 연속 회전 시 무결" 테스트를 P1에 넣는다.**

### 3.4 SE(3) 레퍼런스 거버너

`T_des`를 IK에 직접 넣지 않는다. 내부 상태 `T_cmd`를 두고 속도 제한 수렴시킨다.

```
ξ_raw = log_SE3( T_cmd⁻¹ · T_des ) = (v, ω)
ξ_lim = ξ_raw · min( 1, γ·v_max·dt/‖v‖, γ·ω_max·dt/‖ω‖ )     방향 보존 등방 스케일
T_cmd ← T_cmd · exp_SE3( s · ξ_lim )
```

성질:
1. **절대 서보다.** 사람이 GELLO를 들고 정지하면 `T_des`가 고정되고 `T_cmd`가 수렴해 **로봇이 그 위치에 도달해 머문다**(tick-to-tick 상대 델타 방식의 "들고 있으면 멈춤" 문제 없음). 동시에 engage 순간 오차 0이라 튀지 않는다.
2. **Cartesian 속도 상한이 관절 자코비안과 무관하게 보장된다.**
3. **등방 스케일이라 운동 방향이 보존된다.** 관절별 개별 클램프(하류 slew clamp의 성질)는 Cartesian 방향을 왜곡한다 — 그래서 slew clamp는 백스톱으로만 남긴다.
4. **박스가 없다.** 도달 불가 pose로 GELLO를 들고 있어도 `T_cmd`는 마지막 실현 가능 pose에 동결된다. 절대 목표를 박스로 clip하면 경계 밖 목표에서 오차가 영구히 남아 포화/채터링이 나는데, 여기서는 **증분을 거부**하므로 원리적으로 없다.

**Anti-windup (중요).** HOLD가 반복되면 사람이 계속 움직이는 한 `‖ξ_raw‖`가 단조 증가해 **자기강화 데드락**이 된다. 이를 막기 위해:
- `‖log(T_cmd⁻¹T_des)‖`(pose-level lag)에 상한 `lag_max_pose`(기본 위치 0.05 m / 회전 0.3 rad)를 걸어 초과분을 잘라낸다. 즉 `T_des`가 `T_cmd`에서 그 이상 멀어질 수 없다.
- 이렇게 하면 사람이 손을 되돌리는 즉시 로봇이 반응한다(늦게 따라오는 느낌 제거).

> **감시 대상 쌍을 정정한다.** 관절 공간의 `q_ik_prev` vs `_last_published`가 아니라 **pose 공간의 `T_cmd` vs `T_des`**가 실제로 폭주하는 쌍이다. 전자는 하류 slew clamp가 바인딩하지 않는 한 거의 같다(§3.5).

### 3.5 하류 slew clamp와의 관계 — 예산 정합

기존 slew clamp는 `step = max_step_rad · (0.15 + 0.85·frac)` (soft-start 램프)로 관절당 스텝을 자른다. 배포값 `max_step_rad = 0.0025 @ 250 Hz` → 지속 **0.625 rad/s**.

> ❗ **함정.** EEF stage의 수용 조건 상한을 상수 `max_step_rad`로 두면, soft-start 램프 구간(0.7 s)에서 하류가 최대 **6.7배** 더 잘라낸다. 그러면 `q_ik_prev`와 `_last_published`가 벌어져 "슬루 클램프 발동 = 버그"라는 진단 명제가 무너지고, lag 워치독이 매 engage마다 오발동한다.
>
> **해소:** EEF stage에 **하류가 실제로 적용할 `step_eff`(램프 반영값)를 그대로 넘겨** 수용 조건에 쓴다. 그러면 하류 클램프는 정의상 절대 바인딩하지 않고, `slew_saturated_joints`가 비어있지 않은 것 = 진짜 버그 신호가 된다.
>
> 추가로 **engage 시 `_seed_time`을 갱신하지 않는다.** zero-jump가 이미 수학적으로 보장되므로 soft-start 이중 보호는 불필요하고, 램프 재시작은 위 불일치를 만든다.

### 3.6 그리퍼

**EEF 델타 로직은 arm 6-DoF만 다룬다. 그리퍼 경로는 한 글자도 건드리지 않는다.**

- `gello_publisher`가 `/gripper/gripper_client/target_gripper_width_percent`(Float32)를 발행 → `gello_gripper_bridge`(deadband, rate limit, pause/resume, 0..1 클램프) → `robotiq_*`.
- 브리지는 이 토픽을 **구독하지도 발행하지도 않는다.** identity 매핑 · `invert: false` · threshold/binarize 금지(crush 방지 불변식)가 그대로 유지된다.
- 결과: **disengage 중에도 그리퍼는 GELLO 레버를 따른다.** 재클러치 중 손이 레버를 건드리면 그리퍼가 움직인다.

> 이건 의도적 트레이드오프다. "클러치 뗀 동안 그리퍼 동결"을 구현하려면 브리지가 그리퍼 토픽을 가로채 재발행해야 하는데, 그러면 `gello_gripper_bridge`와 **동일 토픽 이중 publisher**가 되거나 그 노드의 안전장치를 우회하게 된다. 대신 조작 절차로 다룬다: **재클러치 전에 조작자 콘솔의 그리퍼 pause를 쓴다**(이미 존재하는 기능).

---

## 4. 모드 선택 및 clutch 프로토콜

### 4.1 모드 지정

`control_mode`는 **런타임 서비스가 아니라 launch 파라미터**다.

```bash
# joint 모드 (기본값, 오늘과 완전히 동일)
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP>

# eef 모드
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP> control_mode:=eef
```

`control_mode:=eef`일 때만 `config/ur7e_gello_eef.yaml` 오버레이가 파라미터 목록 뒤에 추가된다.

**런타임 joint↔eef 전환을 지원하지 않는 이유:** 전환 시 앵커·필터·시드 상태를 어떻게 옮길지가 또 하나의 설계 문제가 되고, 실기 로그에서 "그때 무슨 모드였나"가 시간에 따라 변해 원인 추적이 어려워진다. **프로세스 수명 = 모드 수명**이면 `ros2 param get`이나 프로세스 목록만 봐도 자명하다. 조작자가 실제로 원하는 유연성(클러치 잡기/놓기)은 아래 engage/disengage로 100% 충족된다.

### 4.2 EEF 모드 내부 상태

```
  BOOTSTRAP ──(핸드셰이크 완료 + ~/eef_engage 게이트 통과)──▶ ENGAGED
      ▲                                                        │  │
      │                                                        │  └─(거부 지속/장애)─▶ HOLD ──▶ (탈출 or 자동 disengage)
      │                                                        │
      └────────────(~/eef_to_joint, 정렬 게이트 통과)───────────┤
                                                               ▼
                                             DISENGAGED (= _paused, 로봇 정지)
                                                               │
                                                    ~/eef_engage (재클러치)
                                                               ▼
                                                           ENGAGED
```

- ⛔ ~~**BOOTSTRAP**: `control_mode:="eef"`로 떴지만 아직 한 번도 engage하지 않은 상태. **joint 패스스루로 동작**한다(§2.4). `gello_move_to_start` 핸드셰이크가 이 상태에서 정상 수행된다.~~
  > **SUPERSEDED (2026-07-22) — 3D 펜 프레이밍으로 교체됨.** engage 전 상태가 joint 패스스루라는 것은, EEF 모드에서는 **첫 틱부터 팔이 리더 관절 형상을 향해 셀을 가로질러 스윙한다**는 뜻이었다. 3D 펜 운용(리더와 로봇이 영구히 다른 관절 자세)에서는 정반대로 틀린 동작이다.
  >
  > **현행 구현**: EEF 모드는 **`HOLD`로 부팅**한다(마지막 명령 포즈 유지, 리더 미러링 없음). joint 패스스루는 **`JOINT_BOOTSTRAP`** 이라는 별도 상태로 분리되었고, 거기로 들어가는 경로는 **정렬 게이트가 있는 3개 서비스뿐**(`~/resume`, `~/resume_chase`, `~/eef_to_joint`)이다. 기동은 `start_mode:=switch_only`(궤적 없음, 팔 무동작) + `~/eef_resume`(정렬 게이트 없이 `HOLD`로 재무장)로 이루어진다.
  >
  > 상태 4종·서비스·게이트의 정본은 **[MODE.md §3.5](GELLO_UR7E_EEF_MODE.md)** 다.
- **ENGAGED**: EEF 델타 제어. 유일하게 카테시안 조작이 되는 상태.
- **HOLD**: 수용 테스트가 거부되어 `T_cmd`가 동결된 상태. 로봇은 마지막 명령에 정지. 사람이 GELLO를 되돌리면 자동 복귀.
- **DISENGAGED**: `_paused = True`. **기존 pause 경로와 물리적으로 동일** — 다음 250 Hz 틱에 발행 중단, 로봇 즉시 정지.

### 4.3 ENGAGE — `~/eef_engage` (`std_srvs/Trigger`)

기존 `~/resume`의 fail-closed Trigger 골격을 그대로 본뜬다. 아래 게이트를 **순서대로** 통과해야 성공하고, 실패하면 **로봇은 1 mm도 움직이지 않는다.**

| # | 게이트 | 거부 메시지 키 |
|---|---|---|
| G1 | `control_mode == "eef"` | `not_eef_mode` |
| G2 | GELLO 샘플 신선 (`age ≤ staleness_timeout_s = 0.5 s`) | `leader_stale` |
| G3 | `_last_published` 존재 (명령 기준 있음) | `no_command_baseline` |
| G4 | `_actual_pose` 존재 **AND** `max_i \|_last_published − _actual_pose\| ≤ anchor_agree_tol` (0.02 rad) | `chains_disagree` |
| G5 | `leader_quasi_still(_gello_history, window, speed)` == True (기존 함수 재사용, fail-closed) | `leader_moving` |
| G6 | **EEF 필터 수렴**: `max_i \|q_lead_f − _raw_target\| < filter_settled_tol` (0.005 rad) | `filter_not_settled` |
| G7 | **내부 왕복 항등 검사**: `‖IK(FK(q_anchor)) − q_anchor‖∞ < 1e-6 rad`, `‖FK(IK(FK(q_anchor))) ⊖ FK(q_anchor)‖ < (1e-4 m, 1e-4 rad)` | `kinematics_selftest` |
| G8 | `σ_min(q_anchor) > σ_warn` — 특이 자세에서 클러치를 잡지 못하게 | `singular_anchor` |
| G9 | keep-out 기하 게이트 통과 (`q_anchor`에서) | `keepout` |

> ❗ **G7의 이름과 주장을 정직하게 조정했다.** 이 검사는 **우리 FK와 우리 IK만** 사용하므로 같은 DH를 공유하는 한 실기 로봇의 실제 DH·tool 오프셋·공장 캘리브레이션과 **무관하게 항상 통과**한다. 즉 "실기 DH 불일치를 잡는다"는 주장은 성립하지 않는다. 이건 **코딩 실수 / 솔버 백엔드 불일치 검출용 내부 무결성 검사**다.
>
> 실기 DH 대조는 **별도의 오프라인 절차(P-1/P6)**로 한다: `/tcp_pose_broadcaster/pose`가 활성이면 `‖FK_ours(q_actual) ⊖ T_vendor‖ < (5 mm, 5 mrad)`를 확인하고, 없으면 펜던트 TCP 표시와 3개 자세에서 수동 대조한다.

전부 통과 시 (원자적, 같은 틱 안에서):
```
앵커 3종 스냅샷 (§3.2) + T_cmd ← T_r_anchor + q_ik_prev ← q_anchor + branch0 설정
state ← ENGAGED
※ _seed_time 은 건드리지 않는다 (§3.5)
```
응답 메시지에 앵커 요약을 실어 반환한다:
```
ENGAGED — anchor p_r0=(0.412,-0.133,0.287) rpy=(179.2,-1.4,88.7)deg,
branch=elbow_up/shoulder_L, ik_residual=3.1e-16, sigma_min=0.142, k_eff=1.00
```

> **`resume_align_tol`(관절 절대 정렬) 게이트는 절대 재사용하지 않는다.** GELLO와 로봇이 어긋나 있어도 되는 것이 델타 모드의 **존재 이유**다. 재사용하면 클러치가 영구히 거부되거나, 반대로 무력화해서 기존 안전장치를 의미 없게 만든다. 재사용하는 것은 `leader_quasi_still` **유틸 함수** 하나뿐이다.

### 4.4 Zero-jump 논증

`t = t0`에서 `R_g(t0) = R_g_anchor`, `p_g(t0) = p_g_anchor` 이므로:

```
R_delta = R_align · I · R_alignᵀ = I     ⇒  R_des = R_r_anchor
p_des   = p_r_anchor + pos_scale · 0     ⇒  p_des = p_r_anchor
∴ T_des = T_r_anchor = T_cmd
⇒ ξ = log(T_cmd⁻¹ T_des) = 0  ⇒  T_try = T_cmd
⇒ q_try = IK(FK(q_anchor), seed=q_anchor) = q_anchor      (왕복 항등, G7이 사전 검증)
⇒ filtered = q_anchor = _last_published
⇒ slew clamp의 delta = 0
⇒ 발행 명령 = 직전 발행 명령
```

**즉 명령 체인이 비트 수준에서 연속은 아니지만(IK 왕복이 ~1e-15 rad 잔차를 남김), `max|Δq| < 1e-9 rad`이 보장된다.** 이는 `max_step_rad = 0.0025`의 약 백만분의 1이다.

> **검증 기준을 "비트 동일"이 아니라 수치 임계로 쓴다.** IK∘FK 왕복은 부동소수 잔차를 남기므로 비트 동일 테스트는 **항상 실패**하고, 그러면 진짜 버그가 났을 때 "원래 실패하니까"로 무시하게 된다. tolerance chain 최하단에 `1e-9 rad`를 명시한다.

### 4.5 DISENGAGE — `~/eef_disengage`

```
self._paused = True          ← 기존 ~/pause 와 완전히 동일한 경로
state ← DISENGAGED
앵커 폐기 (T_g_anchor, T_r_anchor, T_cmd, branch0 무효화)
```

**게이트 없음. 놓는 건 언제나 즉시 허용된다.**

> ❗ **"타깃만 동결하면 된다"는 접근을 기각했다.** `_raw_target`을 마지막 값에 얼려두는 방식은, 상류(EEF stage)가 하류(slew clamp)보다 빠를 수 있는 구조에서 **미실행 잔여 궤적**을 남긴다. 그 상태로 disengage하면 브리지가 0.625 rad/s로 수 초 동안 팔을 계속 밀어붙인다 — "멈춰"를 눌렀는데 로봇이 계속 오는 것은 이 설계에서 가장 위험한 시나리오다.
>
> `_paused = True`는 `_on_timer` 최상단에서 즉시 `return`하므로 **다음 틱(≤4 ms)에 발행이 완전히 멈춘다.** `forward_position_controller`는 마지막 setpoint를 유지하고 로봇은 그 자리에 선다. 이 경로는 이미 실기에서 매일 검증되고 있다.
>
> 추가로 §3.4의 pose-level anti-windup(`lag_max_pose`)이 `T_cmd`가 `T_des`보다 뒤처지는 양 자체에 상한을 걸어, 잔여 궤적이 애초에 크게 쌓이지 않게 한다.

**즉시성.** EEF 모드에는 진행 중인 궤적 개념이 없다(`move_to_start`의 catch-up 스윕 같은 비즉시 구간이 없다). 물리 E-STOP이 최종 백스톱인 원칙은 불변.

### 4.6 RE-CLUTCH — `~/eef_reclutch`

가동범위 끝, 회전 축적, 브랜치 막힘 후 탈출에 쓴다.

```
1. 즉시 T_cmd 동결 (증분 0)  — 로봇은 이 시점부터 정지
2. G2/G5/G6/G7/G8/G9 재검사 (사람이 손을 멈춰야 함)
3. T_g_anchor ← 현재 T_g,  T_r_anchor ← FK(_last_published),
   T_cmd ← T_r_anchor,  q_ik_prev ← _last_published,  branch0 재설정
4. 재개
```

**`T_r_anchor`를 현재 명령 pose로 다시 잡는 것이 핵심** — 그래야 재클러치 시에도 델타가 0에서 시작한다. 로봇은 이 과정 내내 정지한다.

`~/eef_disengage` → (GELLO 되잡기) → `~/eef_engage`도 동일한 결과를 준다. 차이는 `~/eef_reclutch`가 `_paused`를 거치지 않아 조작 흐름이 끊기지 않는다는 점뿐이다.

### 4.7 EEF → joint 모드 복귀 — `~/eef_to_joint`

> ❗ **이 경로가 없으면 EEF 세션에서 빠져나올 수 없다.** EEF 세션의 목적 자체가 두 자세를 decouple하는 것이므로, 몇 분만 조작해도(특히 재클러치를 몇 번 하면) 관절 갭이 여러 관절에서 `resume_chase_max_gap = 1.5 rad`를 쉽게 넘는다. 그러면 `~/resume`도 `~/resume_chase`도 전부 거부되어 **스택 재시작 외에 방법이 없다**(= External Control 재Play + move_to_start 재실행).

```
~/eef_to_joint:
  1. _paused = True 강제 (아직 아니면)
  2. state ← BOOTSTRAP (= joint 패스스루), 앵커 폐기
  3. 응답에 "이제 GELLO를 로봇 자세로 물리적으로 맞춘 뒤
     ~/resume 또는 ~/resume_chase 를 호출하세요" + 현재 per-joint 갭 표시
```

정렬 작업을 돕기 위해 `~/eef/state`에 **관절별 실시간 갭 배열**(`joint_gap`)을 발행하고 조작자 콘솔에 표시한다.

> **`resume_chase_max_gap`을 올리지 않는다.** 3.0 rad로 올리면 게이트는 통과하지만, 로봇이 0.625 rad/s로 6관절 동시에 큰 각도를 **충돌 인지 없이 자율 활강**한다. 조작자가 로봇 근처에 있는 상태로. 이게 이 시스템에서 사람이 다칠 가장 현실적인 시나리오다. 문서에 "한 세션 안에서 EEF→joint 복귀는 **수동 재정렬을 필요로 한다**"를 운영 제약으로 명시한다.

### 4.8 자동 disengage / HOLD (사람 개입 없이)

| 트리거 | 전이 | 이유 |
|---|---|---|
| GELLO staleness > 0.5 s | ENGAGED → DISENGAGED + 앵커 무효화 | stale 동안 사람이 GELLO를 옮겼을 수 있다. 이전 앵커 재사용은 큰 델타 누적을 의미 |
| `/joint_states` staleness > 0.5 s | → DISENGAGED | 앵커 유효성 근거 상실. External Control 드롭도 여기서 잡힌다 |
| 수용 테스트 거부 (`s < s_floor`) | ENGAGED → HOLD | 브랜치/리밋/특이점/keep-out |
| HOLD가 `hold_latch_s`(기본 2.0 s) 연속 | HOLD → DISENGAGED + 앵커 무효화 | 영구 포화 방지 |
| `‖p_cmd − p_r_anchor‖ > max_excursion_m` | → HOLD | 게인 오설정 / 폭주 백스톱 |
| EEF stage 실행시간 예산 초과 연속 N회 | → DISENGAGED (fail-closed) | 타이머 오버런 → 명령 지터 → protective stop 방지 (§6.6) |
| `~/pause` 호출 | ENGAGED → DISENGAGED | pause는 자동 disengage를 동반 |
| EEF stage 내 임의 예외 | → DISENGAGED (전체 `try/except`) | 콜백 예외로 노드가 죽는 것보다 명시적 정지가 낫다 |

**모든 자동 전이는 `WARN` 로그 + `~/eef/state`에 고유 사유 문자열로 남는다.** 실기에서 "왜 갑자기 멈췄나"를 로그 한 줄로 구분할 수 있어야 한다.

### 4.9 조작자 인터페이스

`gello_operator_console_node.py`에 메뉴 항목을 추가한다. 이 노드는 이미 phase-aware `call_first()` 라우팅을 갖고 있어 서비스가 없으면(joint 모드) 무해하게 안내 메시지를 낸다 — **joint 모드 콘솔 동작에 영향 없음.**

```
  7) EEF 클러치 잡기 (engage)
  8) EEF 클러치 놓기 (disengage)  — 로봇 즉시 정지
  9) EEF 재클러치 (reclutch)
 10) joint 모드로 복귀 (eef_to_joint)
```

콘솔 상단에 `~/eef/state` 요약을 표시: `state`, `σ_min`, `γ`, 마지막 `reject_reason`, `joint_gap` 최댓값.

**물리 데드맨 스위치(풋페달)는 이번 범위 밖이다.** 조사 결과 "손을 놓았는지"를 자세/힘으로 추론하는 확립된 방법이 없고(GELLO엔 힘센서도 그립센서도 없다), 명시적 토글이 표준 관행임이 확인됐다. 콘솔 토글로 시작하고, 실사용 후 불편하면 나중에 얹는다. 얹을 때는 `~/eef_engage`/`~/eef_disengage`를 부르는 얇은 클라이언트 노드 하나면 되고, 서비스 계약은 그대로다.

---

## 5. IK 전략

### 5.1 선택: 해석적 위치레벨 IK (백엔드 2개)

`ur_kin.py`가 두 백엔드를 노출한다:

| 백엔드 | 내용 | 상태 |
|---|---|---|
| `analytic` (선호) | `ur_analytic_ik` (PyPI, MIT, `Victorlouisdg/ur-analytic-ik`)의 UR7e 닫힌형 8-해 IK | ⚠️ **이 PC에 미설치** (`ModuleNotFoundError` 실측 확인). 소스 배포(sdist)만 존재하며 nanobind/scikit-build **소스 빌드 필요**. → **P-1에서 실기 타깃 머신 설치·검증 필수** |
| `numeric` (폴백) | 자체 FK + 감쇠 최소자승(DLS) 뉴턴 반복 (seed = `q_ik_prev`, 최대 20 iter, tol 1e-8), 순수 numpy | 의존성 없음. CI/유닛테스트 기본 백엔드이자 `analytic`의 **교차검증기** |

**FK와 자코비안은 자체 구현한다.** DH 소스는 리포에 커밋한 `config/ur7e_dh.yaml`(§3.1). 두 백엔드가 같은 pose에 대해 1e-6 이내 일치하는지를 단위 테스트로 고정한다.

> ❗ **`ur_analytic_ik`의 `inverse_kinematics_closest`는 쓰지 않는다.** 무가중 최근접 선택이라, 손목 특이점 근방에서 유효 해가 8→6개로 줄며 추적 중이던 브랜치가 사라지면 **구조적으로 다른 브랜치로 3.14 rad 점프**를 낸다(조사에서 합성 궤적으로 실측 재현됨). 전체 해집합(`inverse_kinematics`)을 받아 **자체 선택 레이어**를 건다.

### 5.2 왜 위치레벨인가 (속도레벨 J⁻¹ 적분 기각)

1. **드리프트가 조용하다.** 매 틱 `J⁻¹Δx`를 적분하면 `FK(q)`와 목표 pose 사이 오차가 아무 신호 없이 누적된다. "손을 앵커로 되돌렸는데 로봇이 제자리로 안 온다"는 형태로 나타나며 원인 규명이 매우 어렵다. 위치레벨은 매 틱 `T_cmd`에서 새로 풀므로 드리프트가 정의상 0이고, `‖FK(q_try) ⊖ T_try‖`를 매 틱 잔차로 발행해 감시한다.
2. **DLS는 특이점 근방에서 통제되지 않은 방향으로 태스크 오차를 만든다** — 로봇이 사람이 민 방향과 **다르게** 움직인다. 우리 방식은 그 상황에서 "안 움직인다"로 수렴한다. 텔레오퍼레이션에서 잘못된 방향으로 가는 것보다 안 가는 게 낫다.
3. 위치레벨의 유일한 약점(다중해 불연속)은 **관측 가능하고 거부 가능**하다. 드리프트는 그렇지 않다. **실패 모드를 "조용한 것"에서 "시끄러운 것"으로 바꾸는 것**이 전 workspace 안전의 핵심이다.

### 5.3 해 연속성 — 브랜치 선택

```python
S = ik_all(T_try)                                   # 최대 8해 (정규 구간)
S = [s for s in S if branch_id(s) == branch0]       # ★ 브랜치 필터를 먼저
S = [wrapped_nearest(s, q_ik_prev) for s in S]      # 그 다음 ±2π 재앵커
S = [s for s in S if within_joint_limits(s)]        # elbow는 언랩 금지
q_try = argmin_{s∈S} ‖s − q_ik_prev‖_W
accept ⟺ ‖q_try − q_ik_prev‖_W ≤ branch_tol         # 0.25 rad
```

> ❗ **순서가 중요하다.** `wrapped_nearest`는 관절별로 ±2π를 더하므로, 언랩 **후**에 `sign(q3)`/`sign(q5)` 같은 부호 기반 브랜치 라벨을 계산하면 물리적으로 동일한 구성이 반대 라벨을 받는다(예: `q5 = +0.08`이 `q_prev = −6.20` 기준으로 `−6.20`으로 재앵커되면 부호가 뒤집힘). 그러면 손목을 살짝 돌렸을 뿐인데 해집합이 공집합이 되어 로봇이 갑자기 멈춘다.
>
> **더 근본적으로, 부호 기반 라벨은 병합점(`q3≈0`, `q5≈0`)에서 원리적으로 불안정하다** — 그 지점은 두 브랜치가 만나는 곳이라 라벨을 엔코더 노이즈가 결정한다.
>
> **해소:** (a) 브랜치 라벨은 IK 함수가 반환하는 **해 인덱스**(UR 해석해 유도의 분기 선택자: `θ1`의 ± 근, `θ3`의 ± 근, `θ5`의 ± 근)를 1급 값으로 노출해 쓴다. 반환 순서가 결정론적임을 P1 테스트로 고정한다. (b) 부호 추론이 불가피한 경우 `|q3| < 0.05` 또는 `|q5| < 0.05` 구간에서는 브랜치 검사를 **유예**하고 `branch_tol`(가중 노름 거리)에만 의존한다. (c) `branch_tol`이 단독으로도 충분히 강하다: 손목 플립 해는 `(q4+π, −q5, q6+π)`라 가중 노름 ≈ 3.8 rad ≫ 0.25, 어깨 플립은 더 크다.

**가중치** `W = diag(2.0, 2.0, 1.5, 1.0, 1.0, 0.5)`. 근거 — 위험도는 "관절이 얼마나 도느냐"가 아니라 "**링크가 공간을 얼마나 쓸어내느냐**"다. `shoulder_pan`/`shoulder_lift`가 같은 각도로 돌면 팔 전체가 쓸리고, `wrist_3`는 툴만 돈다.

### 5.4 수용 테스트 — 해석적 배율 (이산 라인서치 기각)

```
q_try0 = IK_branchlock(T_cmd · exp(ξ_lim), seed=q_ik_prev)
s*     = min( 1,
              step_eff / max_i|q_try0[i] − q_ik_prev[i]|,
              (리밋까지 마진) / ... )
s      ← 0.9 · s*
q_try  = IK_branchlock(T_cmd · exp(s·ξ_lim), seed=q_ik_prev)   # 재검증
(필요 시 1회 더 반복, 최대 3회 IK 호출)
s < s_floor(0.02) 이면 HOLD
```

> ❗ **고정 집합 `s ∈ {1, ½, ¼, ⅛, 0}` 방식을 기각한 이유.** 필요한 배율이 1/8 미만이면 **곧바로 0(완전 정지)**으로 떨어진다. 저-manipulability 자세에서는 이게 흔하다: 유효 모멘트암 0.02 m인 자세에서 `v_max·dt = 0.32 mm`를 내려면 `Δq ≈ 0.016 rad`로 `step_eff`를 6배 초과해 s=1/8도 부족하다. 실현 가능한 초저속 이동(s=0.06)이 존재하는데 로봇이 "설명 없는 벽"을 만든다. 게다가 수용 판정량이 s에 대해 단조라는 보장이 없다(브랜치 선택이 s에 따라 바뀔 수 있음).
>
> 해석적 배율은 최대 3회 IK 호출(µs급)이므로 비용이 무시 가능하고, 필요한 만큼만 정확히 줄인다.

### 5.5 특이점 / 리밋 / wrist-flip 정리

| 현상 | 처리 |
|---|---|
| 손목 특이점 (`q5→0`) | γ 감쇠(§6.3) + `branch_tol` 거부 |
| 어깨 특이점 (손목중심이 pan 축 근처) | γ가 자코비안 `σ_min` 기반이므로 **자동으로 커버**. 추가로 해석해의 `√(px²+py²−d4²)` radicand를 `~/eef/diag`에 직접 실어 0 접근을 조기 경보 |
| 팔꿈치 신전 (`q3→0`) | γ 감쇠. 브랜치는 안 바뀌므로 브랜치 락은 도움 안 됨 — **γ가 유일한 방어** |
| 도달 한계 | IK 해 없음 → `NO_IK` → 배율 축소 → HOLD. 추가로 `‖p_des‖`가 도달반경(0.8172 m)의 0.95를 넘으면 반경 성분만 clip해 부드럽게 정지 |
| 관절 리밋 | 마진 0.05 rad 포함 수용 조건. **`elbow_joint`만 ±π** (`joint_limits.yaml` 실측: "shoulder_lift가 방해해서 물리적으로 불가능", ros-industrial/universal_robot#265), 나머지 5축 ±2π. **elbow는 ±2π 언랩 금지** |
| wrist flip | 브랜치 인덱스 유지 + `branch_tol` |

> ❗ **`μ = |sin q5|` 같은 손목 전용 지표를 쓰지 않는다.** UR 특이점은 셋(손목/어깨/팔꿈치)이고, 손목 지표만 보면 나머지 둘에서 γ가 1.0을 유지한 채 감속 없이 진입한다. 어깨 특이점에서는 수 mm 이동이 `θ1`에 ~π를 요구해 조작자 의도와 무관한 큰 스윙이 나온다. **자코비안 `σ_min` 하나로 셋을 통합**한다.

**자코비안 무차원화.** `J_geom`은 단위가 섞여 있어(m vs rad) 그대로 SVD하면 무의미하다. 특성 길이 `L`(기본 0.30 m)로 정규화:
```
J_w(q) = diag(1/L, 1/L, 1/L, 1, 1, 1) · J_geom(q)
σ_min  = min(svd(J_w))
```

---

## 6. 안전 설계

### 6.1 설계 원칙 — 왜 bounding box가 아닌가

요구사항이 **UR7e 전체 가동범위**다. HIL-SERL 계열의 task별 수 cm 절대 pose 박스(`clip_safety_box`)는 정면 충돌한다. 더 근본적으로, **절대 목표를 박스로 clip하면 경계 밖 목표를 홀드할 때 오차가 영구히 남아 명령이 포화/채터링한다.**

대신 **증분을 거부**한다. `T_cmd`가 그 자리에 동결되므로 채터링이 원리적으로 없고, 사람이 GELLO를 되돌리면 즉시 풀린다.

### 6.2 안전 레이어 (안쪽 → 바깥쪽)

```
L1  Cartesian 속도 상한   ‖v‖ ≤ γ·v_max,  ‖ω‖ ≤ γ·ω_max        (거버너)
L2  특이점 감쇠 γ         σ_min 기반, 비대칭                      (거버너)
L3  pose-level anti-windup ‖T_cmd ⊖ T_des‖ ≤ lag_max_pose        (거버너)
L4  브랜치 유지 + 점프 거부 ‖Δq‖_W ≤ branch_tol                   (IK 레이어)
L5  관절 리밋 마진        lo+m ≤ q ≤ hi−m                        (IK 레이어)
L6  keep-out 기하 게이트   링크 원점 vs 바닥/실린더/반평면          (IK 레이어)
L7  per-joint 스텝 상한   max_i|Δq| ≤ step_eff                   (IK 레이어)
L8  누적 변위 상한        ‖p_cmd − p_r_anchor‖ ≤ max_excursion_m  (게인 백스톱)
L9  slew clamp + soft-start                                      (기존, 최종 백스톱)
L10 staleness watchdog                                           (기존)
L11 물리 E-STOP                                                  (최종)
```

L1~L8은 전부 **거부 → HOLD**로 수렴한다. 어떤 레이어도 "목표를 조용히 바꿔서 실행"하지 않는다.

### 6.3 γ 락업 방지 (비대칭 감쇠)

```
γ(q) = clamp( (σ_min − σ_stop)/(σ_warn − σ_stop), γ_min, 1 )
       γ_min = 0.05  (0이 아님)
비대칭 규칙: σ_min(q_try) > σ_min(q_ik_prev) 인 증분(= 특이점에서 멀어지는 방향)에는
             γ를 적용하지 않고 통과시킨다.
```

> ❗ **등방 스칼라 γ를 하한 없이 쓰면 영구 락업이 생긴다.** `σ_min < σ_stop`이 되는 순간 `γ=0 → ξ=0 → T_cmd 동결 → q_ik_prev 동결 → 다음 틱도 동일` = 영구 정지. 조작자가 GELLO를 **원래 방향으로 되돌려도** 탈출이 불가능하다. 유일한 복구가 disengage → joint 복귀인데, 그 경로가 정렬을 요구하므로 실질적으로 세션이 죽는다.
>
> γ는 특정 방향(최소 특이 방향)만 퇴화하는 현상에 대한 스칼라 근사인데, 스칼라는 멀쩡한 5방향까지 죽인다. **비대칭 규칙 + `γ_min` 하한**으로 "들어가는 건 막고 나오는 건 허용"한다. 안전은 L4~L9가 담당하며, 그쪽은 상태 동결을 만들지 않는다.

추가로 `γ < 0.2`가 `hold_latch_s` 이상 지속되면 콘솔에 **"특이점 근접 — 후퇴 또는 재클러치"** 를 띄우고, **재클러치가 γ 락업을 실제로 푸는지**를 P5 테스트로 고정한다(재클러치는 `q_ik_prev`를 `_last_published`로 되돌리므로 풀려야 한다).

### 6.4 keep-out 기하 게이트 (L6)

FK를 이미 갖고 있으므로 추가 의존 없이 수십 µs면 된다. 후보 `q_try`에 대해 링크 원점(shoulder, elbow, wrist_1..3, TCP)을 FK로 뽑아:

- (a) 바닥 평면: 모든 원점의 `z > z_floor_margin`
- (b) yaml로 정의한 keep-out 실린더 / 반평면 (테이블 상판, 카메라 마운트 기둥, 조작자가 서는 쪽)
- (c) 베이스 실린더 반경

하나라도 위반이면 `GEOM_KEEPOUT` 사유로 배율 축소 → HOLD.

**완전한 충돌 검사가 아니다.** 그러나 실제 사고 모드의 상당 부분을 덮는다.

### 6.5 미해결 리스크 — 충돌 비인지 (완화만 가능)

> ⚠️ **EEF 모드는 joint 모드보다 충돌 위험이 높다.**
>
> joint 모드에서는 GELLO의 관절 형상 = 로봇의 관절 형상이므로, 조작자는 손 안의 축소 모델로 팔꿈치·어깨가 어디 있는지 **운동감각적으로** 안다. 리포 전체에 충돌 검사가 없음에도 실기 운용이 성립해온 이유가 이것이다. **EEF 모드는 형상 선택권을 IK에 넘기므로 이 채널이 끊긴다.**
>
> 브랜치 잠금은 이산 형상(elbow up/down)만 고정할 뿐, **같은 브랜치 안에서 팔꿈치가 쓸고 지나가는 공간은 전혀 제한하지 않는다.** TCP가 직선으로 10 cm 움직이는 동안 팔꿈치는 크게 스윙할 수 있고, 조작자에겐 예측 근거가 없다.
>
> **완화**: keep-out 게이트(§6.4), 낮은 `v_max`, 물리적 워크스페이스 클리어런스, 운영자 주시, E-STOP.
> **결과**: `elbow ±π` 근처 / 어깨 특이점 탐침(P10)은 **keep-out 게이트 구현 이후**로 순서를 옮기고, 실기가 아니라 **mock에서 먼저** 수행한다.

### 6.6 실행시간 예산 (250 Hz 단일 스레드)

`gello_ur_bridge_node`는 `rclpy.spin(node)` 즉 **SingleThreadedExecutor**에서 돈다. 이 성질 덕분에 engage 서비스 콜백과 `_on_timer`가 직렬화되어 **앵커 스냅샷의 원자성이 보장**된다.

> 📌 **주석으로 못박을 것**: 나중에 `MultiThreadedExecutor`로 바꾸면 이 보장이 즉시 깨진다.

같은 스레드에 EEF stage가 추가된다: FK 1회 + 자코비안 + 6×6 SVD + IK 최대 3회 + 해 선택 루프 + 진단 발행. 해석 IK가 µs급이라도 순수 Python 루프 오버헤드와 numpy 소형 연산 호출 비용은 무시할 수 없다.

**대응:**
- EEF stage에 **실행시간 계측**을 넣어 틱당 소요를 `~/eef/state`에 p50/p99로 발행. 예산(4 ms의 25% = **1.0 ms**) 초과가 연속 N회면 fail-closed DISENGAGED.
- `PoseStamped` 3종은 250 Hz가 아니라 **10~30 Hz로 데시메이트**(디버깅에 250 Hz는 불필요하고 rosbag도 감당 못 한다).
- 자코비안 SVD는 **N틱마다(기본 10틱 = 25 Hz)** 갱신하고 그 사이 γ는 홀드. `σ_min`은 `q`의 매끄러운 함수라 25 Hz면 충분하다.
- **P0 벤치에 "워스트 케이스(IK 3회 전량 소진) 틱 실행시간"을 필수 측정 항목으로 넣는다.** 초과 시 `publish_rate_hz`를 낮추거나 백엔드를 재검토.

### 6.7 Fail-safe 매트릭스

| 실패 | 감지 | 동작 | 관측 |
|---|---|---|---|
| GELLO 스트림 끊김 | `staleness_timeout_s` 0.5 s | 발행 중단(홀드) + **EEF 앵커 무효화**. 복구 후 재engage 필요 | `~/eef/state: auto_disengaged/leader_stale` |
| 로봇 `/joint_states` 끊김 | 동일 | DISENGAGED | `robot_state_stale` |
| External Control 드롭 | `/joint_states` 정지로 간접 감지 | DISENGAGED + 앵커 무효화. **EC 재Play만으로 복귀 금지** — 핸드셰이크 재실행 | `robot_state_stale` |
| IK 해 없음 | 수용 테스트 | 배율 축소 → HOLD → 2 s 후 DISENGAGED | `reject: NO_IK` |
| 브랜치 점프 | `‖Δq‖_W > branch_tol` | HOLD | `reject: BRANCH_JUMP` |
| 관절 리밋 근접 | 마진 위반 | HOLD (사전에 `NEAR_LIMIT` 경고) | `reject: JOINT_LIMIT` |
| keep-out 침범 | 링크 원점 검사 | HOLD | `reject: GEOM_KEEPOUT` |
| 특이점 접근 | `σ_min < σ_warn` | γ 감쇠 (탈출 방향 면제) | `gamma`, `sigma_min` |
| 게인 오설정 / 폭주 | `‖p_cmd − p_r_anchor‖` | HOLD | `reject: EXCURSION` |
| 틱 예산 초과 | 실행시간 계측 | DISENGAGED | `tick_us_p99`, `budget_overrun` |
| EEF stage 예외 | `try/except` | DISENGAGED | `reject: EXCEPTION` + 스택 로그 |
| 조작자 정지 | `~/eef_disengage` / `~/pause` | 다음 틱 발행 중단 | `DISENGAGED` |
| 그 무엇도 안 들을 때 | — | **물리 E-STOP** | — |

**모든 실패의 방향은 "정지"이지 "예상 못한 이동"이 아니다.**

### 6.8 speed scaling

`forward_position_controller`는 펜던트 speed-scaling(감속 슬라이더, 안전 감속)을 **존중하지 않는다** — 기존 문서에 명시된 알려진 취약점이다.

> ## ⛔ SUPERSEDED (2026-07-22) — **구독은 구현되지 않았다**
>
> 아래 원문은 `/speed_scaling_state_broadcaster`를 구독해 `v_max`·`ω_max`에 곱하겠다고 적었다. **구현되지 않았다** — 브리지 소스 전수 grep 결과 `speed_scaling` 문자열이 **한 곳도 없다**.
>
> **운영상 의미: 펜던트 속도 슬라이더는 EEF 명령 속도에 아무 영향이 없다.** 슬라이더를 50%로 내려도 우리가 내보내는 EEF 속도는 그대로다. 감속 수단은 **`v_max` / `w_max` launch 인자뿐**이다. → [MODE.md (H5)](GELLO_UR7E_EEF_MODE.md)
>
> 브로드캐스터 자체는 `ur_robot_driver`의 기본 로드 목록에 있어 **토픽은 존재하지만 아무도 구독하지 않는다.** 구현하려면 별도 작업이다.

<details><summary>원본 서술 (미구현, 참고용)</summary>

> EEF 모드에서는 `v_max`/`ω_max`가 우리 손에 있으므로 **`/speed_scaling_state_broadcaster`를 구독해 `scale < 1`이면 `v_max`·`ω_max`에 그대로 곱한다.** 전 workspace를 허용하는 대가로 이번 범위에 포함한다.
>
> 이 브로드캐스터는 `ur_robot_driver`의 `ur_controllers.yaml` 기본 로드 목록에 있어 활성일 가능성이 높지만, **P-1에서 `ros2 topic list`로 실측 확인**한다. 없으면 EEF 모드에서 `v_max`를 보수적으로(0.05 m/s) 고정하고 문서에 명시한다.

</details>

---

## 7. 변경/신규 파일 목록

| 경로 | 액션 | 내용 |
|---|---|---|
| `ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/ur_kin.py` | **신규** | UR7e FK(`T(q) = ∏A_i`), 기하 자코비안, 무차원화 `σ_min`, IK 래퍼(백엔드 `analytic`/`numeric`), `branch_id`, `within_joint_limits`, keep-out 기하, SE(3) `log`/`exp`, 회전행렬↔쿼터니언(**xyzw**). DH는 `config/ur7e_dh.yaml`에서 로드. **rclpy 임포트 금지** |
| `.../ur_gello_bringup/eef_delta.py` | **신규** | `EefDeltaController` 클래스: `engage/reclutch/disengage/step`. 앵커 스냅샷, 델타 적용, SE(3) 거버너, 비대칭 γ, 해석적 배율, 수용 테스트, anti-windup, reject reason. **rclpy 임포트 금지** |
| `.../ur_gello_bringup/gello_ur_bridge_node.py` | **수정** | ① `control_mode` 파라미터 ② `_on_timer` per-joint 루프를 **필터 루프 / 클램프 루프로 분리** ③ eef 분기에서 `_euro_lead` + `eef_stage` 호출 ④ `~/eef_engage`/`~/eef_disengage`/`~/eef_reclutch`/`~/eef_to_joint` Trigger 서비스(기존 `~/resume` 골격 복제) ⑤ staleness 브랜치에 앵커 무효화 1줄 ⑥ `~/eef/state` + PoseStamped 3종 발행 ⑦ `/speed_scaling_state_broadcaster` 구독. **lazy import**: `control_mode=="joint"`이면 `ur_kin`/`eef_delta`를 임포트조차 하지 않음 |
| `.../config/ur7e_dh.yaml` | **신규** | UR7e DH 상수를 리포에 커밋(배포판 의존 제거). Jazzy `ur_description`에서 확인한 값 + 출처 주석 |
| `.../config/ur7e_gello.yaml` | **수정** | `gello_ur_bridge` 섹션에 `control_mode: "joint"` **명시**(= 기존 실행 동작 100% 불변을 설정 파일 자체로 보증) |
| `.../config/ur7e_gello_eef.yaml` | **신규** | EEF 전용 오버레이(짧음): `pos_scale`(**실제 커밋값 `1.0`** — `k` 아님, §3.1 성질 2 SUPERSEDED), `r_align_rpy`, `tool_l_xyz_rpy`, `tool_r_xyz_rpy`, `v_max` 0.08, `w_max` 0.5, `sigma_warn` 0.10, `sigma_stop` 0.03, `gamma_min` 0.05, `char_length` 0.30, `branch_tol` 0.25, `branch_weights`, `limit_margin_rad` 0.05, `s_floor` 0.02, `lag_max_pose` [0.05, 0.3], `max_excursion_m` 0.5, `anchor_agree_tol` 0.02, `filter_settled_tol` 0.005, `hold_latch_s` 2.0, `tick_budget_us` 1000, `keepout` 블록, `ik_backend` |
| `.../launch/ur7e_gello_real.launch.py` | **수정** | `control_mode` LaunchArgument(기본 `"joint"`). `eef`일 때만 params 목록 뒤에 `ur7e_gello_eef.yaml` 추가. **분기는 `IfCondition` 두 개로 끝내고 joint 경로에 조건부 코드가 끼어들지 않게** |
| `.../launch/ur7e_gello_eef_mock.launch.py` | **신규** | mock ros2_control + `fake_gello` + 브리지(`control_mode:=eef`) + RViz. 실기 없이 zero-jump/거부 로직을 눈으로 확인하는 1차 경로 |
| `.../ur_gello_bringup/fake_gello_node.py` | **수정** | 재현 가능한 입력 패턴 추가(`pattern`: `hold` / `offset_hold`(로봇과 고의로 어긋난 정지) / `line_xyz` / `wrist_singularity` / `full_rotation`(0→360°) / `step`). 기본값은 기존 동작 유지 |
| `.../ur_gello_bringup/gello_operator_console_node.py` | **수정** | 메뉴 7~10번 추가 + `~/eef/state` 요약 표시. 기존 `call_first()` phase-aware 라우팅 덕에 joint 모드 영향 없음 |
| `.../test/test_ur_kin.py` | **신규** | FK 왕복 항등, 두 백엔드 교차검증, 자코비안 vs 유한차분, `branch_id` 안정성(연속 궤적에서 불변), 리밋 필터, SE(3) log/exp 왕복, xyzw 컨벤션 왕복, keep-out |
| `.../test/test_eef_delta.py` | **신규** | zero-jump, hold 수렴, 재클러치 zero-delta, disengage가 앵커를 전진시키지 않음, `R_align≠I`, **0→360° 연속 회전 무결(캡 없음)**, γ 비대칭 탈출, 해석적 배율, 거부 사유 6종, anti-windup |
| `.../scripts/eef_replay.py` | **신규** | `gello_recorder` HDF5(`gello_q`/`ur_q`/`cmd`)를 읽어 전 체인 오프라인 재생 → Cartesian 속도 / 관절 스텝 / `σ_min` / γ / 배율 / 거부사유 시계열을 CSV+플롯. **실기 0회로 실제 사람 궤적의 거부율·특이점 근접도를 측정** |
| `.../setup.py` | **수정** | (신규 console_script 없음 — 새 노드가 없으므로) `scripts/` 설치 항목만 |
| `.../package.xml` | **수정** | `python3-numpy` exec_depend. `ur_analytic_ik`는 rosdep 키가 없으므로 **여기 넣지 않고** `requirements-eef.txt` + 문서 절차로 분리 |
| `.../requirements-eef.txt` | **신규** | `ur_analytic_ik` 핀 버전. EEF 모드에만 필요한 pip 의존을 joint 모드와 파일 경계로 분리 |
| `docs/ros2/GELLO_UR7E_EEF_MODE.md` | **존재함 — 현행 운영 문서** | 실행 런북: 실기 전 설정값, 빌드/배포, 단계적 브링업(P4~P9), Remote(headless) 기동, clutch 조작 순서, 조작자 위험 항목, 상태 확인. **조작자는 이 문서를 따른다** — 본 PLAN과 어긋나면 런북이 우선 |

### 7.1 joint 모드 무회귀 보장 — 4중 방어

1. **기본값 방어.** `control_mode` 기본값 = `"joint"`, `ur7e_gello.yaml`에도 명시. 기존 실행 커맨드는 문자 그대로 오늘과 같다.
2. **lazy import 방어.** joint 모드에서는 `ur_kin`/`eef_delta`를 임포트하지 않는다. → `ur_analytic_ik` 미설치여도 **기존 실기 운영에 영향 0**. P3에서 패키지를 일부러 제거한 상태로 joint 모드 기동을 확인한다.
3. **경로 방어.** joint 분기는 기존과 **동일한 per-joint 연산**(`euro[i](raw_target[i])`)을 그대로 수행한다. 루프 분리는 관절별 독립 연산이라 부동소수 결과가 바뀌지 않는다.
4. **측정 방어 (주된 증거).** P3에서 동일 `fake_gello` 시드로 변경 전/후 `/forward_position_controller/commands`를 `ros2 bag`으로 기록해 **샘플 단위 비트 동일**을 diff로 확인한다. 깨지면 즉시 롤백하고 EEF 작업을 진행하지 않는다.

> **"코드를 안 건드렸다"를 무회귀의 근거로 삼지 않는다.** 실제로는 루프를 건드려야 하므로, 근거는 **측정**이다.

---

## 8. 단계별 구현 계획

각 단계는 **독립적으로 확인 가능**하고, 실패 시 **원인 후보가 좁혀지도록** 설계했다. 실기 단계는 반드시 mock 단계 통과 후에만 진행한다.

---

### P-1 — 환경 실증 및 물리 상수 측정 (실기 PC, 로봇 미동작) · 난이도 ★☆☆

> **이 단계를 건너뛰면 뒤의 모든 전제가 흔들린다.** 개발 PC(Jazzy/24.04/py3.12)와 실기 PC가 다를 수 있고, 브랜치명(`...humble-22.04`)이 환경을 오도하고 있다.

측정/확인 항목:
1. 실기 PC의 `ROS_DISTRO`, OS, Python 버전 — **문서 첫 줄에 실측값으로 기록**
2. `ls /opt/ros/$ROS_DISTRO/share/ur_description/config/ | grep ur7e` — 없으면 `config/ur7e_dh.yaml`의 커밋 상수를 그대로 사용(이미 그렇게 설계됨)
3. `pip install ur_analytic_ik` 빌드 성공/실패. 실패 시 `numeric` 백엔드로 진행하고 그 성능을 P0에서 실측
4. `ros2 topic list | grep -E "tcp_pose|speed_scaling"` — 브로드캐스터 활성 여부
5. ~~**GELLO 축소비 `k` 실측**~~ — ⛔ **SUPERSEDED (2026-07-22): `pos_scale`을 위해서는 필요 없다.** `pos_scale = 1.0` 고정으로 결정되었기 때문이다(§3.1 성질 2 SUPERSEDED 박스, 정본은 [MODE.md §1.2](GELLO_UR7E_EEF_MODE.md)). 측정 절차 자체는 **Q1(균일 축소인가?) 판정용으로만** 선택적으로 남긴다: 관절 하나(예 `shoulder_lift`)를 알려진 각도만큼 돌리고 그립점 이동 거리를 자로 측정 → `k = 실측 거리 / FK 예측 거리`, 최소 2개 관절로 교차 확인. **실기 브링업의 선행 조건이 아니다.**
6. ~~**`T_tool_L` 실측**: GELLO flange 원점 → 손잡이 그립점 오프셋 (물리 실측 후 `/k`로 가상-UR 스케일 변환)~~ → ⛔ **SUPERSEDED**: `tool_l_xyz_rpy = tool_r_xyz_rpy`로 확정(§3.1 성질 3 SUPERSEDED 박스, 정본 [MODE.md §1.1](GELLO_UR7E_EEF_MODE.md)). 별도 실측 없음
7. **`T_tool_R` 확인**: Robotiq 2F-85 TCP 오프셋 — ✅ **완료(2026-07-22)**: flange +Z **0.174 m** 실측, 펜던트 TCP `TCP_2f85`와 일치, yaml 반영 완료. **펜던트 TCP는 코드로 자동 전파되지 않으므로 손으로 동기화한다**

**완료 기준**: 위 항목 중 1~4, 6, 7이 전부 문서화된 수치로 존재(5번 `k`는 **선택**). `k`를 측정했다면 2개 관절에서 5% 이내 일치(= 균일 축소 가정 검증).
**실패 시 구분**: `k`가 관절마다 크게 다르면 **비균일 축소**이며, 위치 델타가 방향 왜곡된다 → §10-Q1로 에스컬레이션. (이 왜곡은 `pos_scale` 값과 무관한 별개 현상이다 — `pos_scale`은 등방 스칼라라 방향 왜곡을 만들지도, 고치지도 못한다.)

---

### P0 — 기구학 백엔드 확보 및 벤치 (ROS 없음, 로봇 없음) · 난이도 ★☆☆

`ur_kin.py` 없이 스크립트 하나로 백엔드만 확인.

**완료 기준**:
- (a) 무작위 `q` 10⁴개에 대해 `analytic` 백엔드가 `q`를 1e-9 rad 이내로 복원
- (b) `numeric` 백엔드도 동일 pose에서 1e-6 이내 일치
- (c) `FK(홈자세)` 위치가 UR7e 실제 홈 TCP와 cm 단위 일치
- (d) **워스트 케이스 틱 실행시간**(IK 3회 + SVD + 해 선택) < 1.0 ms

**실패 시 구분**: 여기서 깨지면 라이브러리/DH 문제이지 우리 로직 문제가 아니다.

---

### P1 — `ur_kin.py` + `eef_delta.py` 순수 모듈 (ROS 없음, 로봇 없음) · 난이도 ★★★

**완료 기준** — `pytest` 전량 통과:
- (a) FK 왕복 항등, 자코비안 vs 중앙차분 1e-6
- (b) **zero-jump**: 임의 `q_g`/`q_r` 조합 500쌍에 대해 engage 첫 출력이 `q_anchor`와 **1e-9 rad 이내**
- (c) **joint 등가성**: `T_g_anchor == T_r_anchor`인 앵커에서 임의 궤적 `q(t)`에 대해 출력 == `q(t)` (거버너 rate limit 무시 시). 이 테스트가 통과하면 FK/IK 배선 오류가 사실상 배제된다
- (d) **회전 곱 순서 회귀**: 순서를 뒤집으면 (c)가 실패함을 명시적으로 assert
- (e) **0→360° 연속 회전**: `T_des`가 전 구간 연속(틱간 회전각 < `ω_max·dt`), 거부 0회, 360°에서 앵커 자세로 정확히 복귀 (**캡 없음 확인**)
- (f) `R_align = RotZ(90°)`일 때 x축 리더 이동이 y축 로봇 이동으로 매핑
- (g) **γ 비대칭**: `σ_min < σ_stop` 상태에서 탈출 방향 증분이 통과함
- (h) **해석적 배율**: 저-manipulability 자세에서 `s`가 0이 아닌 유한값으로 수렴
- (i) 손목 특이점 스윕에서 무가중 최근접이 ≥3.0 rad 점프를 내는 것을 **재현**하고, 브랜치 잠금이 이를 거부
- (j) disengage 중 앵커가 전진하지 않음 / 재클러치가 fresh clutch
- (k) anti-windup: HOLD 지속 시 `‖T_cmd ⊖ T_des‖`가 상한을 넘지 않음

**실패 시 구분**: ROS도 로봇도 없으므로 원인 후보가 수식 하나뿐이다. (c) 실패 = FK/IK 또는 곱 순서, (i) 실패 = 브랜치 로직, (g)(h) 실패 = 거버너.

---

### P2 — 실주행 데이터 오프라인 재생 (실기 0회) · 난이도 ★★☆

`scripts/eef_replay.py`로 기존 `gello_recorder` HDF5의 `gello_q`(실제 사람 텔레오퍼 궤적)를 전 체인에 통과시킨다. `ur_q[0]`를 로봇 앵커, `gello_q[0]`를 GELLO 앵커로 engage.

**완료 기준**:
- (a) 앵커가 일치하는 이 설정에서 출력 궤적이 기록된 `gello_q`와 전 구간 1e-6 rad 이내 일치
- (b) **거부율 0%** — 정상 궤적에서 거부가 나면 `branch_tol`/`σ_warn`/`s_floor`를 **여기서** 조정(실기 아님)
- (c) 전 구간 `max|Δq/틱| ≤ step_eff` (= 하류 slew clamp가 **한 번도 발동하지 않음**)
- (d) Cartesian 속도가 `v_max` 이하
- (e) `‖FK(q_out) ⊖ T_cmd‖ < 1e-9` 전 구간 (위치레벨 IK의 무드리프트 확인)
- (f) `σ_min` 분포 히스토그램 → 실제 작업이 특이점에 얼마나 가까이 갔는지 정량화 → `σ_warn`/`σ_stop` 값을 **데이터 기반으로** 결정
- (g) 프레임 간 최대 관절 변화량 분포 → `branch_tol` 데이터 기반 결정

**이 단계가 실기 시간을 가장 많이 절약한다.** 사람 손 궤적 분포에서의 가드 오발동률을 로봇 없이 측정한다.

---

### P3 — joint 모드 무회귀 증명 (mock, 로봇 없음) · 난이도 ★☆☆

브리지에 훅을 실제로 넣은 뒤 `control_mode:=joint`(기본값)로 기존 mock 경로 실행.

**완료 기준**:
- (a) 변경 전/후 `/forward_position_controller/commands`를 `ros2 bag record`해 **샘플 단위 비트 동일**
- (b) 기존 `pytest` + `colcon test` 전량 통과
- (c) `~/pause`/`~/resume`/`~/resume_chase` 3종이 전과 동일 동작
- (d) `ur_analytic_ik`를 일부러 제거한 상태에서도 joint 모드 정상 기동 (lazy import 검증)
- (e) `ros2 topic info -v /gello/joint_states` 결과 publisher count == 1

**(a)가 깨지면 즉시 롤백하고 EEF 작업을 진행하지 않는다.**

---

### P4 — mock EEF 부트스트랩 + zero-jump (로봇 없음) · 난이도 ★★☆

`ur7e_gello_eef_mock.launch.py` + `fake_gello pattern:=offset_hold`(로봇과 고의로 크게 어긋난 정지 자세).

**완료 기준**:
- (a) **부트스트랩**: `control_mode:=eef`로 떠도 `gello_move_to_start` 핸드셰이크가 joint 모드와 동일하게 정상 완료(§2.4 검증)
- (b) engage 전에는 EEF 델타가 적용되지 않고 joint 패스스루로 동작
- (c) `~/eef_engage` 호출 직후 첫 명령이 직전 명령과 **1e-9 rad 이내** → RViz에서 로봇이 **미동도 안 함**
- (d) `~/eef/leader_pose` / `desired_pose` / `commanded_pose` 3 프레임이 engage 순간 `desired == commanded`로 겹침
- (e) engage 후 fake_gello를 움직이면 로봇이 따라옴

**실패 시 구분**: (c) 실패면 앵커 스냅샷 타이밍(engage 시점 값이 최신인가) 또는 필터 수렴 게이트 누락. P1이 통과한 상태이므로 수식은 용의선상에서 제외.

---

### P5 — mock 클러치 UX + 실패 주입 (로봇 없음) · 난이도 ★★☆

**완료 기준**:
- (a) `~/eef_disengage` 후 fake_gello를 크게 흔들어도 로봇 무반응, **disengage 응답이 다음 틱(≤4 ms)에 명령 중단으로 반영됨**(bag 타임스탬프 확인)
- (b) 빠른 조작 직후 disengage해도 로봇이 **계속 움직이지 않음**(§4.5의 잔여 궤적 시나리오 회귀 테스트)
- (c) 재클러치 50회 반복, 매번 zero-jump
- (d) fake_gello kill → `auto_disengaged/leader_stale` 보고, **재시작해도 engage 없이는 안 움직임**
- (e) 도달 불가 pose로 몰면 배율 축소 → HOLD → `hold_latch_s` 후 자동 disengage
- (f) `pattern:=wrist_singularity`로 γ 감쇠 → 매끄러운 감속 정지, **3.14 rad 점프 0회**
- (g) **γ 락업 탈출**: γ가 `γ_min`에 도달한 상태에서 (i) 되돌리는 방향 조작으로 자동 복귀, (ii) 재클러치로 복구 — 둘 다 성립
- (h) engage 거부 9종(G1~G9)이 각각 **서로 다른 사유 문자열**을 반환
- (i) `~/eef_to_joint` 후 갭이 커서 `~/resume`가 거부되고, `~/eef/state`의 `joint_gap`을 보며 fake_gello를 정렬하면 통과

---

### P6 — 실기: 게인 0, 움직이지 않는 것을 확인 · 난이도 ★★☆

실제 UR7e + 실제 GELLO. 기존 실기 절차(프리플라이트, 핸드셰이크, EC Play) 그대로 + `control_mode:=eef`. **`pos_scale = 0.0`, `v_max = 0.01`, `ω_max = 0.05`로 고정.**

> ## ⛔ SUPERSEDED (2026-07-22) — "이 단계에서 로봇은 원리상 한 번도 움직이지 않아야 한다"는 **거짓이었다**
>
> **`pos_scale`은 위치 항 하나만 곱한다**(`eef_delta.py`의 `step()`). 회전 채널 `R_des = R_delta @ R_r_anchor`(`step()`의 회전 항)에는 **스케일이 없다.** 따라서 `pos_scale = 0.0`은 **TCP 위치만 앵커에 고정**하고, 로봇은 공구를 제자리에서 회전시키며 어깨·팔꿈치·손목이 실제로 크게 스윙한다.
>
> 실기 컨트롤러 실측(`pos_scale = 0.0`):
>
> | 리더 입력 | TCP 위치 변화 | TCP 자세 변화 | 최대 관절 변화 |
> |---|---|---|---|
> | `wrist_3` +40° | 2.6e-16 m | **40.00°** | **0.698 rad** |
> | 팔 전체 흔들기 | 2.7e-16 m | **60.29°** | **1.020 rad** |
>
> **왜 이것이 위험한 오류였나**: P6는 **실기** 단계이고 `keepout_json = "{}"`(충돌 인지 전무, §6.5/Q7)다. 아래 원래 기준 (b)는 조작자에게 **60초간 리더를 마구 흔들라**고 지시하는데, 그동안 팔꿈치는 예측 없이 최대 1 rad 가까이 스윙한다. 게다가 그 움직임을 본 조작자는 **정상 시스템을 FAIL로 기록**하게 된다.
>
> **정정된 기준과 절차의 정본은 [MODE.md §3.2 "P6에서 `pos_scale=0.0`이 하는 일"](GELLO_UR7E_EEF_MODE.md)이다.** 요지: 판정 대상은 **TCP 위치**(`~/eef/state`의 `excursion_m`)이며, **팔이 움직이는 것 자체는 PASS**다. 흔드는 동작은 천천히·작게, 팔꿈치 클리어런스 확보, 손은 E-STOP 위.

**완료 기준** (아래 (a)(b)는 위 박스대로 정정됨):
- (a) engage/disengage 20회, 매번 engage 순간 `/forward_position_controller/commands` 스텝 = 0 (zero-jump)
- (b) ⛔ ~~engage 후 60초간 GELLO를 마구 흔들어도 실기 팔 무동작~~ → **정정: engage 후 GELLO를 천천히 움직여도 `excursion_m`이 계속 0**(TCP 위치 무동작). 팔의 회전 동작은 정상
- (c) protective stop 0회
- (d) **실기 DH 대조**(G7이 못 하는 것): `/tcp_pose_broadcaster/pose`(펜던트 TCP `TCP_2f85`가 설정되어 있어 발행됨)와 `~/eef/commanded_pose`를 **3개 이상 자세**에서 비교, `< (5 mm, 5 mrad)`. 없으면 펜던트 TCP 표시와 3개 자세 수동 대조.
  **이것이 잡아야 하는 실제 위험**: `ur_kin.py`의 DH는 **명목값**이며 이 개체의 공장 캘리브레이션이 아니다(`config/ur7e_dh.yaml` 주석). 게다가 `ur_kin.load_dh()`는 **정보용일 뿐 `fk`/`ik`/`jacobian`에 배선되어 있지 않다**(`ur_kin.load_dh()` docstring) — 캘리브레이션 yaml을 넣어도 자동 반영되지 않는다. 절차 상세는 [MODE.md §3.2 P6 (d)](GELLO_UR7E_EEF_MODE.md)
- (e) 틱 실행시간 p99 < 1.0 ms — **관측법(구현 실측)**: `~/eef/state`는 `tick_us_p99`를 발행하지 **않는다**(§9.1 주석). 대신 `tick_budget_us`(1000 µs) 워치독이 `tick_overrun_limit`(5) 연속 초과 시 fail-closed 자동 disengage하므로, **세션 내내 `auto_reason`에 `tick_budget exceeded`가 없고 브리지 WARN 로그가 0회**면 이 기준을 만족한 것으로 본다

**실패 시 구분**: (a)(b)에서 **TCP 위치**가 움직이면(=`excursion_m` ≠ 0) 앵커 로직 버그다(위치 수식은 P1이 검증). **팔의 회전 동작은 실패 신호가 아니다**(위 SUPERSEDED 박스). (d) 실패면 실기 캘리브레이션 반영 필요 → §10-Q3.

---

### P7 — 실기: 순수 병진, 저게인 · 난이도 ★★☆

⛔ ~~회전 델타 강제 무효화(`rot_freeze:=true`, `R_delta ≡ I`).~~ → **SUPERSEDED (2026-07-22): `rot_freeze`는 구현되어 있지 않다** (아래 박스). ~~`pos_scale`을 P-1에서 측정한 `k`로 설정.~~ → ⛔ **SUPERSEDED (2026-07-22): `pos_scale = 1.0`(yaml 기본값)으로 그대로 둔다.** 이유는 §3.1 성질 2의 SUPERSEDED 박스, 정본은 [MODE.md §1.2](GELLO_UR7E_EEF_MODE.md). `v_max` 0.02 → 0.05 m/s 단계 상승. 손은 E-STOP 위.

> ## ⛔ SUPERSEDED (2026-07-22) — `rot_freeze` / `pos_freeze`는 **존재하지 않는다**
>
> 리포 전수 grep 결과 이 두 문자열은 **이 PLAN 산문 두 줄(P7, P8)에만** 나온다. **어떤 소스 파일에도, launch 파일에도, yaml에도 없다** — launch 인자도, 노드 파라미터도, `eef_delta.py`의 cfg 키도 아니다.
>
> **결과: P7과 P8을 격리된 채널로 실행할 수 없다.** P7 실행에는 P8의 회전 거동이 이미 섞여 들어오므로, *"회전 버그를 여기서 완전히 배제한다"* 는 아래 문장은 **이 구현에서 성립하지 않는다.**
>
> 실제로 실행 가능한 P7/P8(조작 절차로만 채널을 분리)의 정본은 **[MODE.md §3.2의 P7/P8 박스](GELLO_UR7E_EEF_MODE.md)** 다. 요지: P7은 **리더 자세를 유지한 채 평행이동만**, P8은 **리더 그립점을 한 자리에 고정한 채 손목만** 비튼다.
>
> 두 플래그를 진짜로 원하면 **별도 구현 작업**이며 이번 범위 밖이다.

**완료 기준**:
- (a) 리더 그립점을 x/y/z 각 축으로 이동 → 로봇 TCP가 **같은 축·같은 부호로** 이동 (`R_align` 검증). `pos_scale=1.0`이므로 이동 **거리**는 손의 물리 이동 거리가 아니라 **joint 모드에서 같은 관절 입력을 줬을 때의 TCP 이동량**과 같아야 한다 — 그 비교가 이 항목의 판정 기준이다(손 5 cm에 TCP가 그보다 크게 가는 것은 정상)
- (b) **축간 크로스토크 없음** — x만 움직였는데 y/z가 따라오면 §10-Q1(비균일 축소) 또는 `R_align` 문제
- (c) 손을 멈추면 로봇도 그 자리에 정지(절대 서보 특성 — 상대/속도 매핑이었다면 드리프트)
- (d) `slew_saturated_joints`가 계속 비어 있음, IK 잔차 < 1e-9

⛔ ~~**회전 버그를 여기서 완전히 배제한다.**~~ → **성립하지 않는다** (`rot_freeze` 미구현, 위 박스). 조작 절차로 회전 입력을 **작게 만들 뿐**이다.

---

### P8 — 실기: 순수 회전, 저게인 · 난이도 ★★☆

⛔ ~~병진 델타 무효화(`pos_freeze:=true`).~~ → **SUPERSEDED (2026-07-22): `pos_freeze`는 구현되어 있지 않다**(P7의 박스 참조). 대신 **리더 그립점을 한 자리에 물리적으로 고정**한 채 손목만 비튼다 — 손이 이동하면 그 병진은 정상 명령이므로 (b) 판정이 오염된다. 앵커 대비 roll/pitch/yaw 각 ±45°.

**완료 기준**:
- (a) 로봇 TCP 자세 변화가 **월드축 기준으로** 리더와 일치 (좌측 곱 컨벤션 검증 — 어긋나면 body-frame 곱을 잘못 쓴 것)
- (b) **기생 병진 노름** `‖p_cmd − p_r_anchor‖`가 tool 오프셋으로 설명되는 수준 이하(< 1 cm). 크면 `T_tool_L`/`T_tool_R` 값이 틀린 것(§3.1 성질 3)
- (c) 앵커 대비 200° 회전 시 **잘림 없이 계속 회전**(§3.3 정정 확인)

> **P7과 P8을 분리하는 이유**: 병진 버그와 회전 버그는 증상이 섞이면 구분이 거의 불가능하다. 특히 tool 오프셋 오류로 인한 기생 병진은 `R_align` 오정합과 증상이 동일하다.

---

### P9 — 실기: 6-DoF 통합 + 재클러치 워크플로 · 난이도 ★★★

`v_max` 0.08 m/s. 실제 태스크 유사 동작(집기-옮기기). GELLO 가동범위 끝에 닿으면 disengage → 되잡기 → re-engage를 3회 이상.

**완료 기준**:
- (a) 매 re-engage에서 로봇 무동작
- (b) 전 세션 protective stop 0회
- (c) 거부율 < 1%, P2 오프라인 예측치와 대조(어긋나면 재생 모델이 실기를 대표하지 못한다는 뜻 → P2 스크립트 보정)
- (d) `~/eef_to_joint` → 수동 정렬 → `~/resume`로 joint 모드 복귀 성공
- (e) 15분 연속 세션에서 조작자가 "의도한 대로 움직인다"고 판단

---

### P10 — mock: 경계 탐침 (실기 아님) · 난이도 ★★★

> ⚠️ **의도적으로 특이점·리밋·keep-out 경계로 몰아넣는 실험은 실기가 아니라 mock에서 한다.** §6.5의 충돌 비인지 때문이다. `elbow ±π` 근처는 정의상 팔이 자기 자신에 접근하는 영역이다.

**완료 기준**: (a) 손목 특이점, (b) 어깨 특이점(TCP가 pan 축 위), (c) `elbow ±π` 근처, (d) 도달 한계, (e) keep-out 침범 — 각 케이스에서 γ 감쇠 또는 거부로 **점프 없이 정지**하고, 사유 코드가 예상과 일치하며, 물러나면 정상 추종이 복귀.

실기에서의 경계 접근은 **keep-out 게이트를 실측 워크스페이스로 튜닝한 이후**, 극저속(`v_max` 0.02)에서 감독 하에만 수행한다.

---

### P11 — 튜닝 + 문서화 · 난이도 ★☆☆

P2의 replay 도구로 P9 실주행 로그를 다시 돌려 `v_max`/`σ_warn`/`branch_tol`/1-Euro 파라미터를 오프라인 조정한 뒤 실기 재확인. `GELLO_UR7E_EEF_MODE.md` 런북 작성.

**완료 기준**: 제3자가 문서만 보고 EEF 모드를 처음부터 띄워 재클러치까지 성공.

---

## 9. 디버깅 · 관측 수단

### 9.1 관측 토픽

| 토픽 | 타입 | 레이트 | 용도 |
|---|---|---|---|
| `~/eef/state` | `std_msgs/String` (JSON) | 10 Hz | 아래 필드 전부 |
| `~/eef/leader_pose` | `PoseStamped` | 10~30 Hz | `T_g` (리더 EEF) |
| `~/eef/desired_pose` | `PoseStamped` | 10~30 Hz | `T_des` (앵커 델타 적용 결과) |
| `~/eef/commanded_pose` | `PoseStamped` | 10~30 Hz | `T_cmd` (거버너 출력) |

`~/eef/state` JSON 필드:
```json
{ "mode":"eef", "state":"ENGAGED", "sigma_min":0.142, "gamma":1.0,
  "ls_scale":1.0, "reject_reason":null, "reject_count_60s":0,
  "ik_residual_m":2.1e-16, "ik_residual_rad":3.3e-16,
  "lag_pos_m":0.003, "lag_rot_rad":0.008,
  "excursion_m":0.121, "branch_id":5, "n_ik_solutions":8,
  "slew_saturated_joints":[], "joint_gap":[0.02,1.31,...],
  "shoulder_radicand":0.083, "tick_us_p50":210, "tick_us_p99":640,
  "anchor_stamp":1234.567, "pos_scale":1.00 }
```

> **실제 구현과의 차이(2026-07-22 실측).** 위 JSON은 설계 시안이다. 현행 `gello_ur_bridge_node._on_eef_state_timer()`가 실제로 싣는 키는
> `mode, state, reject_reason, auto_reason, sigma_min, gamma, ls_scale, ik_residual, lag_pos, lag_rot, excursion_m, branch_id, n_ik_solutions, pos_scale, joint_gap` 이다.
> **`tick_us_p50` / `tick_us_p99` / `budget_overrun` / `reject_count_60s` / `shoulder_radicand` / `slew_saturated_joints` / `anchor_stamp` 는 발행되지 않는다.** 틱 예산은 발행 대신 **워치독**으로만 존재한다: `tick_budget_us`(기본 1000 µs)를 `tick_overrun_limit`(기본 5회) 연속 초과하면 fail-closed 자동 disengage되고, 그 사유가 `auto_reason`에 `tick_budget exceeded ...` 로 찍힌다. 따라서 "p99 < 1.0 ms"의 실무 판정은 **세션 중 해당 auto-disengage / WARN 로그가 0회인가**로 한다.

**3개 pose를 RViz에 동시에 띄우는 것이 1차 진단 도구다.** `leader` / `desired` / `commanded` 세 프레임을 한 화면에서 보면 "리더가 이상한가 / 거버너가 막았나 / IK가 틀렸나"를 눈으로 즉시 구분할 수 있다.

### 9.2 증상 → 의심 지점 표

| 증상 | 1순위 의심 | 확인 방법 |
|---|---|---|
| engage 했는데 로봇이 스르르 움직인다 (사람은 정지) | **EEF 리더 필터가 아직 수렴 안 됨** (§3.2 함정) 또는 앵커가 늦게 잡힘 | `~/eef/state`의 engage 시각 vs `leader_pose` 시계열. G6 게이트 로그 확인 |
| engage 순간 로봇이 튄다 | 앵커 소스 불일치 (`_last_published` vs `_actual_pose`) | G4 `chains_disagree` 로그, `ik_residual` |
| 로봇이 안 움직인다 (engage 성공했는데) | HOLD 중 | `reject_reason`. `NO_IK`/`BRANCH_JUMP`/`JOINT_LIMIT`/`GEOM_KEEPOUT` 구분 |
| 특이점 근처에서 영원히 안 풀린다 | γ 락업 (비대칭 규칙 미적용?) | `gamma`, `sigma_min`. 되돌리는 방향에서 `gamma`가 1로 돌아오는지 |
| 손을 멈췄는데 로봇이 계속 간다 | anti-windup 미작동 / `T_cmd`가 `T_des`보다 크게 뒤처짐 | `lag_pos_m`, `lag_rot_rad` |
| disengage 했는데 로봇이 계속 간다 | `_paused` 경로를 안 탐 | `~/state`가 `PAUSED`인지, 발행이 멈췄는지 `ros2 topic hz` |
| 축이 뒤바뀌거나 섞인다 (병진) | `R_align` 또는 비균일 축소 | **`R_align`을 먼저 배제한다** — 두 원인은 증상이 겹치는데 `R_align`은 로봇 없이 측정 가능하다([MODE.md §1.6](GELLO_UR7E_EEF_MODE.md)). 그 다음 P7 축별 독립 테스트, 그래도 크로스토크면 §10-Q1 |
| 이동 **거리는 맞는데 방향이 조금 비딱하다** ("게가 걷는 느낌") | **`R_align` 오정합 10~20°** — 가장 놓치기 쉬운 구간 | 크기 보존 + 방향만 회전이 `R_align`의 고유 지문이다(§3.3 박스). [MODE.md §1.6](GELLO_UR7E_EEF_MODE.md) 측정 절차. 10 cm 이동 시 가로 오차: 10°→1.74 cm, 20°→3.42 cm |
| **의도한 방향으로 이동량이 0** | `R_align` 오차 90° (π/2 스냅 모호성) | 같은 절차. 이쪽은 오히려 즉시 티가 나므로 안전한 실패다 |
| 특이점 근처에서 팔이 **아주 느리게 계속 기어간다** | **버그 아님 — 의도된 락업 방지.** `σ_min < σ_stop`이어도 `γ_min·v_max = 4 mm/s`로 움직이고, 탈출 방향은 감속조차 안 한다 | `eef_delta.py`의 `step()` γ 블록(γ 하한), `:414-419`(비대칭 면제). §6.3 |
| 2초쯤 멈춰 있다가 **세션이 끊긴다** | `hold_latch_s`는 **최소 유지시간이 아니라 최대 허용시간**(2.0 s)이다 | `auto_reason`에 `hold_latched`. 복귀는 `13) eef 재무장`(`~/eef_resume`) |
| **engage 전인데 팔이 리더 관절을 따라 움직인다** | `JOINT_BOOTSTRAP` 상태에 있다 (3D 펜 `HOLD`가 아님) | `~/eef/state`의 `hold_when_not_engaged`. `false`면 패스스루다 → `~/eef_resume`으로 `HOLD` 복귀([MODE.md §3.5](GELLO_UR7E_EEF_MODE.md)) |
| 펜던트 **속도 슬라이더를 내려도 안 느려진다** | **버그 아님 — 미구현.** 브리지는 `/speed_scaling_state_broadcaster`를 구독하지 않는다 | §6.8 SUPERSEDED 박스. `v_max:=` / `w_max:=`로 재기동 |
| 축이 뒤바뀐다 (회전) | 좌/우 곱 순서 | P1 (d) 회귀 테스트가 통과했는데 실기에서 틀리면 GELLO `joint_signs` 문제 |
| 손목만 돌렸는데 TCP가 크게 병진 | **tool 오프셋 미설정/오설정** (§3.1 성질 3) | P8 (b). `tool_l_xyz_rpy`와 `tool_r_xyz_rpy`가 **같은 값(`[0,0,0.174,0,0,0]`)인지 먼저 확인** — 한쪽만 0이면 이 증상이 정확히 나타난다. 그 다음 펜던트 TCP와 yaml이 동기화됐는지 |
| 손 5 cm에 로봇이 12 cm | **버그 아님 — 정상 동작.** 축소 리더의 당연한 성질이고 joint 모드와 동일하다(§3.1 성질 2 SUPERSEDED 박스, [MODE.md §1.2](GELLO_UR7E_EEF_MODE.md)) | `~/eef/state`의 `pos_scale`이 `1.0`인지만 확인. `1.0`이면 정상. `1.0`이 아닌데 아무도 바꾼 기억이 없다면 그때가 오설정이다 |
| 손을 조금 움직였는데 로봇이 **joint 모드보다도 훨씬 크게** 간다 | `pos_scale`이 `1.0`보다 크게 설정됨 (의도적 증폭 값이 남아있음) | `~/eef/state`의 `pos_scale`, 기동 로그의 `pos_scale=...` 줄, launch에 `pos_scale:=` 를 넘겼는지 |
| 정상 조작 중 갑자기 멈춤이 잦다 | `branch_tol`/`s_floor`/`σ_warn` 과대 | `reject_count_60s`. P2 replay로 오발동률 재측정 |
| 로봇이 굼뜨다 / 진동 | `slew_saturated_joints` 비어있지 않음 = **버그 신호**(§3.5) | `step_eff` 전달 누락 확인 |
| 가끔 명령이 끊긴다 | 틱 예산 초과 | `tick_us_p99`, `budget_overrun` |
| 팔꿈치가 예상 밖으로 스윙 | **충돌 비인지 — 근본 미해결**(§6.5) | keep-out 정의 확대. 물리 클리어런스 |

### 9.3 오프라인 재생

`scripts/eef_replay.py`는 실기 세션 없이 실주행 데이터로 전 체인을 재생한다. **파라미터 튜닝은 원칙적으로 여기서 먼저 하고, 실기는 확인만 한다.**

---

## 10. 미해결 질문 / 사용자 결정 필요 사항

### Q1. GELLO 축소 모델이 **균일**한가? — 부분 해소 (2026-07-22): **`k`는 측정하지 않는다**

> **결정 기록.** 축소비 `k` 자체는 **측정하지 않기로 확정**되었고(`pos_scale = 1.0` 유지, §3.1 성질 2 SUPERSEDED 박스), 사용자가 3D 펜 프레이밍 하에서 이를 **재확인**했다. 따라서 이 항목의 "P-1에서 `k`를 2개 관절로 교차 측정" 부분은 **수행하지 않는다.**
>
> **대신 정직하게 기록해 둘 것**: `pos_scale = 1.0`은 **가상 풀사이즈 UR7e 공간에서만 1:1**이고, **조작자의 손은 로봇 TCP보다 대략 `k`배 적게 움직인다**. 즉 이것은 **의도적으로 증폭된 펜**이며, **"내 손이 가는 자리에 공구 끝이 간다"는 이 설정이 제공하지 않는다.** 그 대가로 얻는 것은 joint 모드와 동일한 조작 감각(익숙한 기준선)이다. → [MODE.md §1.2](GELLO_UR7E_EEF_MODE.md)
>
> **남는 질문은 "균일성"뿐이다.** 비균일이면 위치 델타의 **방향**이 왜곡되고, 그것은 P7의 축간 크로스토크로 나타난다. 다만 그 증상은 **`R_align` 오정합과 구별이 어렵다**(§3.3 박스) — **`R_align`을 먼저 측정해 배제한 뒤**에 비균일성을 의심할 것.

(원문)
리포 어디에도 GELLO 링크 길이 데이터가 없다. "UR7e와 링크 길이 비율이 동일한 축소 모델"이라는 전제는 **리포의 어떤 파일로도 뒷받침되지 않는다.** 우리 설계는 회전이 스케일 불변이라는 성질에만 강하게 의존하므로 비율이 균일하면 `k` 하나로 끝나지만, **링크마다 비율이 다르면 위치 델타의 방향이 왜곡된다**(회전은 여전히 정확). P-1에서 2개 관절로 `k`를 교차 측정하고 P7에서 축간 크로스토크로 판정한다.
→ **비균일로 판명되면**: GELLO 실측 링크 길이로 별도 DH를 만들거나(하드웨어 작업), 폴백으로 "회전만 EEF 델타, 위치는 joint 미러링" 축소 모드를 검토한다.

### Q2. `T_tool_L` / `T_tool_R` 실측값 — ✅ **해결 (2026-07-22)**
`T_tool_R` = Robotiq 2F-85 TCP = flange **+Z 0.174 m** (실측, 펜던트 `TCP_2f85`와 일치).
`T_tool_L` = **`T_tool_R`과 동일값**. "사람이 GELLO를 어디를 잡느냐"라는 원래 질문은 **성립하지 않는다** — `tool_l`은 물리 GELLO가 아니라 **가상 풀사이즈 UR7e flange** 기준 오프셋이기 때문이다(§3.1 성질 3 SUPERSEDED 박스). 정본 [MODE.md §1.1](GELLO_UR7E_EEF_MODE.md).
→ 남은 확인은 **P8에서 기생 병진 < 1 cm**로 하는 실기 검증뿐이다.

> ⚠️ **운영 주의**: 펜던트에서 TCP를 다시 티칭하면 `config/ur7e_gello_eef.yaml`의 `tool_*_xyz_rpy`도 **손으로** 고치고 재빌드해야 한다. 코드는 드라이버/펜던트 TCP를 읽지 않는다.

### Q3. 실기 UR7e 기구학 캘리브레이션 반영 여부
zero-jump는 리더/로봇 FK와 IK가 **같은 DH**를 쓸 때 왕복 항등으로 보장된다. `config/ur7e_calibration.yaml`의 보정 DH를 로봇측에만 넣으면 해석 IK(명목 DH 가정)와 불일치해 engage 시 미세 스텝이 생긴다. **기본은 명목 DH 통일**이며, 이는 TCP 절대 위치에 1~2 mm 오차를 남긴다(상대 텔레오퍼에는 무해).

> **구현 실측(2026-07-22)**: 현재 코드는 **명목 DH 통일로 이미 고정되어 있다.** `ur_kin.load_dh()`는 `config/ur7e_dh.yaml`을 읽지만 **정보용일 뿐이며 `fk`/`ik`/`jacobian`은 모듈 상수(`D`, `A`, `ALPHA`, …)를 쓴다**(`ur_kin.load_dh()` docstring, "INFORMATIONAL ONLY … Wiring it into the kinematics … is intentionally not done"). 따라서 캘리브레이션 파일을 넣어도 **자동 반영되지 않는다.** 실제 오차 크기는 P6 (d)에서 `/tcp_pose_broadcaster/pose` vs `~/eef/commanded_pose` 대조로 **측정만** 하고, 반영 여부는 그 결과를 보고 결정한다.
→ **결정 필요**: 절대 좌표 기반 정밀 작업이 EEF 모드의 목표인가? 아니라면 명목 DH 통일로 확정한다.

### Q4. `ur_analytic_ik` 빌드 가능성 (P-1)
이 개발 PC에 **미설치이며 sdist만 존재**(소스 빌드 필요). 실기 PC에서 빌드 실패 시 `numeric` 백엔드로 진행하는데, 그 250 Hz 성능은 **미실측**이다. P0 (d)에서 판정한다.
→ 실패 시 폴백: UR 6R 구면손목 해석해는 공개 표준 유도이므로 `ur_kin.py`에 순수 numpy로 벤더링(~200줄)하는 것이 현실적 플랜 B.

### Q5. `σ_warn` / `σ_stop` / `char_length` / `branch_tol` 초기값
전부 **추정치**다. 무차원화 방식이 다르면 수치가 통째로 달라진다. P2 오프라인 재생에서 실제 사람 궤적의 `σ_min` 분포를 보고 정한다. 미튜닝 상태로 실기에 올리면 정상 자세에서 γ가 떨어져 로봇이 이유 없이 무거워지거나(과대), 특이점을 그냥 통과한다(과소).

### Q6. keep-out 기하 정의 (**사용자 입력 필요**)
§6.4의 keep-out 게이트는 실제 워크스페이스 형상을 모른다. 테이블 상판 높이, 카메라 마운트 위치·반경, 조작자가 서는 방향을 yaml로 받아야 한다.
→ **사용자 확인 필요**: 실기 셋업의 물리 장애물 좌표.

### Q7. 충돌 비인지 (**미해결로 남는다**)
EEF 모드는 joint 모드보다 충돌 위험이 **높다**(§6.5). keep-out 게이트는 완화일 뿐 해결이 아니다. MoveIt planning scene 기반 충돌 검사를 250 Hz 루프에 넣는 것은 이번 범위 밖이며, 넣더라도 planning scene에 실제 환경이 등록되어 있어야 한다.
→ 운영 제약으로 문서에 명시하고, 물리 클리어런스 + 운영자 주시 + E-STOP에 의존한다.

### Q8. `/speed_scaling_state_broadcaster` — ⛔ **모의(moot) 처리 (2026-07-22)**
질문은 "브로드캐스터가 활성인가"였는데, **활성 여부와 무관하게 브리지가 구독하지 않는다**(§6.8 SUPERSEDED 박스, 소스 전수 grep). 따라서 **"UR이 내부적으로 감속 중인데 우리는 원속도로 스트리밍하는 불일치"는 항상 존재한다.**
→ 대응은 원안 그대로: **`v_max`를 보수적으로 잡는다**(브링업 단계에서는 launch 인자로 0.01→0.02→0.05→0.08). 펜던트 슬라이더를 안전 수단으로 세지 말 것.

### Q9. 그리퍼 — 클러치 중 동결이 필요한가?
현재 설계는 **그리퍼 경로를 건드리지 않는다**(§3.6). 따라서 disengage 중 GELLO를 되잡을 때 손이 레버를 건드리면 그리퍼가 움직인다. 조작 절차(콘솔의 그리퍼 pause)로 대응하도록 했다.
→ **사용자 확인 필요**: 실사용에서 이게 불편하면, `gello_gripper_bridge`에 "EEF disengaged 동안 자동 pause" 연동을 별도 작업으로 추가한다(이번 범위 밖).

### Q10. 실기 PC 환경 (P-1)
브랜치명은 Humble/22.04를 가리키지만 이 개발 PC는 **Jazzy/24.04/py3.12**다. 조사에서 확인한 모든 패키지 경로·버전·`joint_limits.yaml` 값은 Jazzy 기준이다. **실기 PC에서 재확인 전까지는 추정으로 취급한다.**

---

## 11. 기각한 대안과 이유

| 대안 | 기각 이유 |
|---|---|
| **상류에 별도 노드(`gello_eef_mapper`)를 두고 `/gello/joint_states`에 재발행** | (1) **부트스트랩 데드락**: `gello_move_to_start`가 `/gello/joint_states`를 하드코딩 구독하고 첫 메시지를 무한 대기하는데, EEF 노드는 engage 전 침묵한다. 동시에 engage 게이트는 브리지가 PAUSED가 아닐 것을 요구하고, 그 pause를 푸는 유일한 주체가 move_to_start다 → 순환. (2) **windup**: 상류가 하류보다 빠르면 미실행 잔여 궤적이 쌓여 disengage가 로봇을 못 멈춘다. (3) 동일 토픽 dual-writer 문제를 하나 더 만든다. (4) `_actual_pose`/`_gello_history`/staleness 상태를 전부 중복 구현해야 한다 |
| **MoveIt Servo (`ur_servo.yaml`)** | 배선은 완벽히 맞다(`command_out_topic`이 이미 우리 토픽). 그럼에도: (a) **속도(Twist) 인터페이스**라 "앵커 대비 절대 pose 서보"와 패러다임이 다르고, 속도 적분으로 흉내내면 "들고 있으면 멈춤" 드리프트로 되돌아간다. (b) 같은 토픽에 쓰면 검증된 안전스택(1-Euro/slew/soft-start/staleness/pause·resume)을 **통째로 우회**해 두 모드가 서로 다른 안전 파이프라인을 타게 된다. (c) UR 프리셋의 특이점 임계값(100/200)이 패키지 기본(17/30)과 크게 달라 근거 불명 — 이해 못 하는 안전 파라미터에 사람 안전을 맡기지 않는다 |
| **속도레벨(자코비안 DLS) 자체 구현 + 적분** | 드리프트가 **조용히** 쌓이고 아무 신호도 안 낸다. 특이점 근방 damping이 사람이 민 방향과 **다른 방향**으로 태스크 오차를 만든다. 위치레벨의 약점(다중해 불연속)은 관측·거부 가능하므로 "조용한 실패"를 "시끄러운 실패"로 바꾸는 쪽을 택했다 |
| **KDL / TRAC-IK / pick_ik** | 수치 반복이라 250 Hz 루프에서 수렴 보장이 없고 특이점 근방 실패율이 높다. 간헐 실패 = 간헐 홀드 = 진동. 실기 디버깅에서 "가끔 안 움직인다"의 원인이 IK 타임아웃인지 배선인지 구분 불가 |
| **`ur_analytic_ik`의 `inverse_kinematics_closest` 직접 사용** | 무가중 최근접이라 해집합이 8→6으로 줄 때 3.14 rad 점프를 낸다(실측 재현됨). 브랜치 소실을 감지하지 못한다 |
| **부호 기반 브랜치 라벨(`sign(q3)`, `sign(q5)`)** | 병합점(`q3≈0`, `q5≈0`)에서 라벨을 엔코더 노이즈가 결정하고, `wrapped_nearest` 후에 계산하면 물리적 동일 구성이 반대 라벨을 받는다. 해 인덱스 + `branch_tol`로 대체 |
| **이산 라인서치 `s ∈ {1,½,¼,⅛,0}`** | 필요 배율이 1/8 미만이면 즉시 완전 정지 → 저-manipulability 자세에서 "설명 없는 벽". 해석적 배율로 대체(§5.4) |
| **`μ = \|sin q5\|` 손목 전용 특이점 지표** | UR 특이점은 셋인데 하나만 본다. 어깨/팔꿈치 특이점에서 γ=1.0을 유지한 채 감속 없이 진입한다. 자코비안 `σ_min`으로 통합 |
| **하한 없는 등방 γ** | `γ=0`이 되면 탈출 방향까지 죽어 **영구 락업**. `γ_min` + 비대칭 규칙으로 대체(§6.3) |
| **bounding box 클리핑(HIL-SERL `clip_safety_box`)** | task별 수 cm 절대 pose 박스가 "UR7e 전 가동범위"와 정면 충돌. 게다가 절대 목표 clip은 경계 밖 목표 홀드 시 오차가 영구히 남아 포화/채터링. 증분 거부로 대체 |
| **tick-to-tick 상대 델타 매핑 (hil-rl `gello_bridge/delta.py`)** | engage zero-delta는 보장하지만 매 틱 기준을 전진시키므로 GELLO를 들고 정지하면 로봇도 멈춘다("박스에 갇힘"). 절대 앵커 서보를 채택 |
| **`/tcp_pose_broadcaster/pose`를 로봇 앵커 소스로 사용** | 벤더 FK와 우리 IK 사이 왕복 잔차가 그대로 engage 스텝이 되어 zero-jump의 수학적 보장이 깨진다. 실기 teleop launch에서 활성인지도 미확인. **대조 검증용으로는 쓴다**(P6 (d)) |
| **MuJoCo(`sim_robot.py`의 `attachment_site` FK)를 리더 FK로** | ROS2 브리지 프로세스에 MuJoCo 런타임을 끌어들여야 하고, UR5e 메시 기반 근사이며, IK와 다른 기구학 소스라 왕복 항등이 깨진다. FK와 IK는 반드시 같은 파라미터를 공유해야 한다 |
| **쿼터니언(hil-rl `rotation.py`) 이식** | scalar-first(`wxyz`) ↔ ROS scalar-last(`xyzw`) 변환에서 부호/축이 조용히 틀어지는 것이 최대 함정이고, 실기에서 "팔이 이상하게 돈다"로만 드러나 디버깅이 지옥이다. 회전행렬만 쓰면 두 문제가 **존재하지 않는다** |
| **`R_delta`를 rotvec으로 캡(`θ∈[0,π]`)** | ">π는 표현 불가"는 **틀린 전제**다. 캡을 넣으면 π 통과 시 축이 부호 반전해 **인공 불연속**이 주입된다(§3.3) |
| **`resume_align_tol` 정렬 게이트를 engage에 재사용** | GELLO와 로봇이 달라도 되는 것이 델타 모드의 정의다. 재사용하면 클러치가 영구 거부되거나 안전장치가 무의미해진다 |
| **`resume_chase_max_gap`을 올려 EEF→joint 복귀** | 로봇이 6관절 동시에 큰 각도를 충돌 인지 없이 자율 활강한다. 이 시스템에서 사람이 다칠 가장 현실적인 시나리오. 수동 재정렬로 대체(§4.7) |
| **disengage 시 타깃만 동결(발행 계속)** | 미실행 잔여 궤적 때문에 로봇이 수 초 더 움직인다. `_paused` 경로 재사용으로 대체(§4.5) |
| **런타임 `~/set_control_mode` 서비스** | 전환 시 앵커·필터·시드 상태 이전이 또 하나의 설계 문제가 되고, 실기 로그에서 "그때 무슨 모드였나"가 시간에 따라 변해 원인 추적이 어려워진다 |
| **`rot_scale` 파라미터 제공** | 회전 스케일링은 log/exp map을 강제하고 그 순간 π 캡 문제가 부활한다 |
| **`ur_rtde` servoL/speedL 직결** | RTDE Control 인스턴스는 동시에 하나만 연결 가능해 현재 External Control URCap + ros2_control 경로와 **배타적**이다. 기존 안전스택 전체를 우회한다 |
| **FZI `cartesian_controllers`** | 미설치, 소스 빌드 필요, distro 지원 미검증, controller_manager에 새 컨트롤러 스폰이 필요해 `gello_move_to_start`의 STRICT 스위치 로직까지 손대야 한다 |
| **물리 데드맨 스위치(풋페달)를 1단계에 포함** | 리포에 없는 하드웨어를 새로 요구하고, 하트비트 타임아웃이라는 새 FAULT 원인을 추가한다. 콘솔 토글로 시작하고 나중에 얹는다(서비스 계약은 그대로) |

---

## 12. 향후 확장 지점 (구현하지 않음)

> 아래는 **이번 범위 밖**이며, 지금 코드를 한 줄도 쓰지 않는다. 나중에 online RL human intervention을 얹을 때 **어디에 훅이 생기는지**만 기록한다.

나중에 intervention을 얹을 때 손댈 곳은 정확히 세 군데다. **(1)** `~/eef_engage` / `~/eef_disengage` Trigger를 부르는 주체를 조작자 콘솔 대신 arbiter 노드로 바꾸면 된다 — 앵커·게이트·zero-jump 로직은 그대로 재사용되고 서비스 계약도 변하지 않는다. **(2)** `~/eef/state`의 `state` 필드(`ENGAGED` = 사람이 조종 중)가 그대로 intervention 라벨의 타임스탬프 소스가 된다 — 별도 라벨링 채널이 필요 없다. **(3)** 유일하게 새로 만들어야 하는 것은 브리지 **상류의 mux/arbiter**다: 현재 `policy_leader_node`(정책)와 `gello_publisher_node`(사람)가 `/gello/joint_states`에 대한 dual-writer이며 코드 레벨 배타 장치가 없고 런치 구성으로만 배타되어 있다. 정확히 하나만 발행하도록 강제하는 계층이 이번 설계 **밖의** 미해결 아키텍처 gap이다. 이번 설계는 이 세 훅이 자연스럽게 붙도록 상태기계와 토픽 계약을 그 모양으로 만들어 두었을 뿐, RL 관련 코드는 포함하지 않는다.
