# GELLO → UR7e **EEF 모드** 실행 런북 (operator runbook)

> 설계·근거는 [`GELLO_UR7E_EEF_TELEOP_PLAN.md`](GELLO_UR7E_EEF_TELEOP_PLAN.md) 참고. 이 문서는 **실기에서 EEF 모드를 실제로 띄우기 전에 설정·측정해야 하는 것**과 단계적 브링업 절차만 담는다.
>
> **안전 불변**: GELLO는 항상 passive read-only. Dynamixel에 절대 토크 X.

---

## 0. EEF 모드가 하는 일 (한 줄)

**상대(델타) EEF 텔레오퍼레이션.** engage(클러치) 순간 GELLO EEF와 로봇 EEF를 각각 앵커로 스냅샷하고, 이후로는 **GELLO EEF의 앵커 대비 변화량만** 로봇 EEF 앵커에 적용한다. GELLO와 로봇 자세가 **일치하지 않아도** 되고, engage 순간 로봇은 움직이지 않는다(zero-jump). joint 모드(절대 미러링)와 공존하며 `control_mode` launch 파라미터로 고른다.

---

## 1. ⚙️ 실기 전에 설정해야 하는 값 — `config/ur7e_gello_eef.yaml`

아래 값들은 현재 **미측정 플레이스홀더**다. joint 모드에는 전혀 영향이 없지만(이 파일은 `control_mode:=eef`일 때만 로드), EEF 모드를 실기에서 쓰려면 확인/설정해야 한다.

### 1.1 tool 오프셋 (`tool_r_xyz_rpy`, `tool_l_xyz_rpy`) — **회전 조작 전에 설정**

`[x, y, z, roll, pitch, yaw]` (m, rad). flange(tool0) 기준 오프셋으로, **EEF의 회전 중심**을 정의한다.

| 키 | 무엇 | 안 맞으면 |
|---|---|---|
| `tool_r_xyz_rpy` | 로봇 TCP 오프셋 (flange → 실제 작업점, 예: Robotiq 2F-85 손끝 ≈ +0.16 m in z) | 순수 회전 입력에 **기생 병진**이 붙어 TCP가 예상 밖으로 수 cm 이동 |
| `tool_l_xyz_rpy` | GELLO를 잡는 손의 기준점 (flange → 그립점) | 위와 동일. 리더/로봇 회전 중심 불일치가 기생항을 만듦 |

- **병진(P6/P7)에는 무관** — tool 오프셋은 회전 중심에만 영향을 준다. 그래서 첫 병진 브링업은 zeros(=flange 기준)로도 된다.
- **회전(P8) 전에 반드시 설정** — 최소한 `tool_r`을 실제 TCP로 맞춰야 손목만 비틀었을 때 TCP가 병진하지 않는다.
- Q2 결정(2026-07-20): "로봇이 주는 기본 TCP 기준으로." → `tool_r`을 UR 펜던트/드라이버가 쓰는 실제 TCP 오프셋과 일치시키면 된다(정밀 작업 아니면 대략값 OK). 커스텀 툴 오프셋이 필요해지면 여기서 수정.
- **측정법**: flange 원점에서 작업점까지 자로 재거나 데이터시트값 사용. `tool_l`은 "사람이 GELLO를 어디를 잡는가"에 의존하므로 조작자 합의 + 실측.

### 1.2 `pos_scale` — **`1.0` 그대로 두면 됨 (측정 불필요)**

리더 EEF는 GELLO 관절각을 **풀사이즈 UR7e FK**에 넣어 계산하므로(`T_g = fk(q_gello)·T_tool_L`), 위치 델타 `Δp_g`는 이미 풀사이즈 UR7e 공간의 변위다. 따라서 `pos_scale=1.0`이면 **로봇 EEF 이동량 = joint 모드의 EEF 이동량**과 일치한다 — "작은 GELLO 조금 → 큰 로봇 많이"는 joint 모드에서 이미 익숙한 정상·의도된 동작이다. 축소비 `k`를 따로 측정할 필요 없다. (`1.0` 이외 값은 "의도적 증폭"으로 취급.)

### 1.3 속도/보간은 우리 파이프라인이 처리 (UR 드라이버에 의존 X)

`forward_position_controller`는 **자체 속도 제한을 하지 않고**, 명령이 너무 빠르면 protective stop을 낸다. 그래서 rate limiting은 우리 쪽에서 한다 — 리더 전용 one-euro(`_euro_lead`) + SE(3) 카테시안 속도 거버너(`v_max`/`w_max`) + joint slew clamp. **손을 아무리 빨리 휘둘러도 로봇 EEF 속도는 `v_max`로 캡**된다.

### 1.4 keep-out (선택)

`keepout_json`은 현재 `"{}"`(미설정, G9 항상 통과). 워크셀 장애물 게이트가 필요하면 JSON 문자열로 채운다(예시는 yaml 주석 참고). Q6 결정: 당분간 펜던트에서 처리 → 비워둠.

### 1.5 `ik_backend`

`"analytic"` (ur_kin에 벤더링된 순수 numpy 폐형해, 8해 enumerate → branch-lock에 필요). `"numeric"`은 단순 폴백.

---

## 2. 빌드 & 배포

```bash
# 실기 PC (distro 확인: 이 코드는 Jazzy에서 검증. Humble이면 재확인 필요)
cd <ws>/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

> ⚠️ **config yaml을 수정하면 반드시 `colcon build`를 다시 돌려라.** colcon은 config를 install로 **복사**한다(symlink 아님). 수정 후 재빌드 안 하면 launch가 옛 값을 읽는다. (`--symlink-install`도 ament_python data_files는 복사한다.)

---

## 3. 단계적 브링업 (mock → 실기)

첫 실기 실행은 "연결하고 engage"가 아니라 **저게인 게이트드 브링업**이다. pos_scale 때문이 아니라 특이점 거동·IK·충돌 비인지(플랜 §6.5, Q7)가 실기 첫 확인이라서.

| 단계 | 환경 | 확인 |
|---|---|---|
| P4 | mock + RViz | `ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py pattern:=offset_hold` → 핸드셰이크 후 `~/eef_engage` → RViz에서 **로봇 미동(zero-jump)** 눈으로 |
| P5 | mock | 거부 사유들, γ 락업 탈출, 재클러치, disengage 즉시정지 |
| P6 | **실기** | `control_mode:=eef` + **`v_max:=0.01`** → engage/disengage 반복, 로봇 **무동작** + protective stop 0회 |
| P7 | 실기 | 순수 병진, `v_max` 0.02→0.05 단계 상승. 손 축이동 → TCP 동축 이동 |
| P8 | 실기 | 순수 회전. **여기 전에 §1.1 `tool_r` 설정** — 기생 병진 < 1cm 확인 |
| P9 | 실기 | 6-DoF + 재클러치 워크플로 |

실행 진입점:
```bash
# joint 모드 (기본, 오늘과 동일)
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP>
# eef 모드
ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP> control_mode:=eef
```

조작(조작자 콘솔 메뉴): 7) engage  8) disengage(즉시정지)  9) reclutch  10) joint 복귀.

---

## 4. 상태 확인 · 디버깅

- `ros2 topic echo /gello_ur_bridge/eef/state` — state(BOOTSTRAP/ENGAGED/DISENGAGED), sigma_min, gamma, reject_reason, joint_gap.
- RViz에 `~/eef/leader_pose` / `desired_pose` / `commanded_pose` 3프레임 → "리더가 이상한가 / 거버너가 막았나 / IK가 틀렸나" 눈으로 구분.
- 증상→원인 표는 플랜 §9.2 참고.
