# 08 — 미해결 안전 갭과 임시 완화책

이 문서는 **아직 닫히지 않은 것**만 모은다. "곧 고칠 것" 같은 낙관적 표현을 쓰지 않는다.
각 항목은 코드 근거 → 왜 위험한가 → **지금 쓸 수 있는 완화책** 순이다.

> ## 🛑 게이트 선언
> **G2 · G4 · G6 · G9가 닫히고, G1 · G3 · G13이 실기에서 확인되기 전에는
> RL 정책 경로로 실기 UR7e를 구동하지 않는다.**
> `DRY_RUN`은 기본값 `True`로 둔다 (`DefaultUR7eEnvConfig`, `CubeInCupEnvConfig.DRY_RUN`).

### 현황 요약 (2026-07-27 갱신)

| | 갭 | 이전 | 지금 |
|---|---|---|---|
| G1 | `clip_safety_box` | 🔴 미구현 | 🟢 **구현·단위검증** (실기 미검증) |
| G2 | `PolicyDeltaController` 단순화판 | 🔴 | 🔴 (변화 없음) |
| G3 | 그리퍼 개입 | 🟠 데드코드 | 🟡 **코드 수정됨**, 하드웨어 미확인 |
| G4 | 두 퍼블리셔 충돌 / 업샘플러 스테일 | 🔴 | 🟠 **자동 거부 가드 추가** (G4a), G4b는 그대로 |
| G5 | 19-D state 순서 | 🟢 | 🟢 + **Kanu 해시 일치 실증** |
| G6 | staleness 시 예외 | 🟠 | 🟠 |
| G7 | 명목 DH ↔ 캘리브레이션 | 🟠 | 🟠 (+ 플랜지 프레임 확정으로 일부 명확해짐) |
| G8 | 브로드캐스터 누락 = 조용한 오염 | 🟠 | 🟠 |
| G9 | 개입 앵커가 flange 기준 | 🟡 | 🟡 |
| G10 | pinned submodule | 🟡 | 🟡 |
| G11 | build/gripper shell | 🟢 | 🟢 |
| G12 | 스페이스바 데드맨 기본값 | 🟡 | 🟢 **해결 (07-28)** — `--deadman topic` 배선, 기본값 `topic` |
| **G13** | **리셋 branch-cut** | (미발견) | 🟡 **수정·단위검증**, 실기 미검증 |
| **G14** | **시스템 grpcio 1.30.2 손상** | (미발견) | 🟠 **완화만 됨**(venv 강제), 근본 미해결 |
| **G15** | **분류기 전처리 ↔ 크롭 불일치** | (미발견) | 🔴 **07-28 증명됨** — recall 100%→33.3%. 미수정. **아래 정정 참조** |
| **G16** | **10 Hz 레이턴시 예산 소진** | (미발견) | 🟠 유선 전환 필요 |

> ### 🔧 G15 정정 (2026-07-28) — 원인 지목이 틀렸다
>
> 이 문서와 `06_SENSORS.md`가 원인으로 지목한 `reward_classifier_runtime.decode_classifier_image()`는
> **ZMQ GUI 뷰어(port 5594) 전용 모듈**이고, gRPC 경로(port 50053)와 **호출 관계가 전혀 없다.**
> 서버 `rlpd_receive_server._classifier_observation()`은 이미지 변환을 **하나도** 하지 않는
> strict-validating passthrough다 (`grep -n "resize\|cv2\." rlpd_receive_server.py` → 0 hit).
>
> **진짜 원인은 actor 쪽 `IMAGE_CROP`이다.** 학습은 크롭 없이 full-frame을 128×128로 찌그러뜨렸는데
> (`cube_classifier_pipeline.py::preprocess_frame` + `export_0724.py`가 `crop=None`), actor의
> `ur7e_env.get_im()`은 크롭 후 리사이즈한다. 이 문구를 근거로 서버 코드를 고치면 **엉뚱한 모듈을
> 건드리게 된다.**
>
> 증명: 픽셀 대조 MAE **0.00**(무크롭 가설, 비트 일치) vs **21–35**(크롭), 실제 체크포인트 실행에서
> recall@0.85 **100.0% → 33.3%**. 반증 시도 6종 전부 실패.
>
> **단 현재 프로덕션은 안 망가져 있다** — `ur_experiments/cube_in_cup.py`가 kanu 체크아웃 브랜치에
> 없어 거기서는 `IMAGE_CROP={}`가 적용된다. **actor 브랜치를 머지하는 순간 유입된다.**
>
> 전체 내용과 수정 선택지는 `serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` §12.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## G1 — `clip_safety_box` 🟢 **구현·단위검증 완료 (실기 미검증)**

> ### 🔧 정정 (2026-07-27)
> 이전 판: *"두 gym Box는 생성된 뒤 리포 어디에서도 다시 참조되지 않는다. 주석만 있고
> 클램프 코드는 없다."* — **더 이상 사실이 아니다.** commit `d49d0f6` / `ee3240e`에서
> 구현·배선됐다. `serl_ur_infra/README.md`의 "❌ config만 존재, 미작동" 현황표도 낡았다.

### 지금의 사실

| | 위치 |
|---|---|
| 박스 구성·검증 | `UR7eEnv._build_safety_box()` (`ur7e_env.py:213-297`) |
| 공개 클립 (upstream 시그니처) | `UR7eEnv.clip_safety_box(pose7) -> pose7` (`:334`) |
| 컨트롤러 훅 (4×4 어댑터) | `UR7eEnv._clip_command_pose(T) -> (T, clipped)` (`:357`) |
| 배선 | `PolicyDeltaController(..., clip_pose=self._clip_command_pose)` (`:188`), 적용 지점 `policy_delta_controller.py:140-141` |
| 회귀 테스트 | `tests/test_clip_safety_box.py` — **26 passed** (2026-07-27) |

적용 시점은 upstream과 동일하게 **명령 포즈**다. 관측은 클립하지 않는다 — 관측을 클립하면
정책에게 팔 위치를 거짓말하게 된다.

**두 가지 안전 설계가 들어 있다. 이걸 "단순화"하지 말 것:**

1. **REFUSE-DON'T-CLAMP.** `ABS_POSE_LIMIT_*`가 기본값(0벡터)이거나 뒤집혀 있으면 박스를
   **끄고 크게 경고한다.** 0-부피 박스로 클램프하면 TCP를 (0,0,0)·rpy 0으로,
   즉 **로봇 베이스를 관통하는 방향**으로 명령하게 된다.
2. **인덱스 3은 `rx`가 아니라 `|rx|`다.** 툴이 바닥을 향하므로 `rx`는 ±π 근처에 살고
   `as_euler("xyz")`가 가까운 쪽 분기로 보고한다. cube_in_cup 샘플의 **12%가 음의 분기**에
   있다. 순진한 `np.clip(rx, 2.60, pi)`는 `rx=-3.14`를 `+2.60`으로 보내 **5.7 rad 손목
   플립**을 "안전 조치"로 명령한다. upstream처럼 **크기를 클립하고 부호를 복원**한다.

### 실측 박스 (`cube_in_cup`, 23테이크 19,802샘플) — ⚠️ 프레임에 주의

**전부 플랜지(`tool0`) 프레임이다.**

```python
ABS_POSE_LIMIT_LOW  = [0.375, -0.229, 0.185, 2.60, -0.30, 1.10]   # [3]은 |rx|
ABS_POSE_LIMIT_HIGH = [0.642,  0.272, 0.550, pi,    0.35, 2.20]
TCP_POSE_SOURCE     = "driver"
TCP_OFFSET_XYZ_RPY  = [0, 0, 0, 0, 0, 0]
```

> ### 🛑 이 셋은 **한 세트**다. 하나만 바꾸면 안전 바닥이 테이블 17.4 cm 아래로 간다
> 2F-85는 플랜지 z로 174 mm지만 **cube_in_cup을 녹화할 때 펜던트 TCP는 0이었다**
> (검증: 기록된 `ur_joint_states`를 FK로 재생하면 `/tcp_pose_broadcaster/pose`를
> **플랜지 해석에서 중앙값 0.6 mm**로 재현한다. 그리퍼 끝 가설에서는 174.2 mm 어긋난다).
> `PolicyDeltaController`도 플랜지(`fk(q)`, `T_tool` 없음)에서 적분한다.
>
> 진짜 TCP 의미론으로 옮기려면 **넷을 동시에** 바꿔야 한다:
> `TCP_POSE_SOURCE → "fk"`, `TCP_OFFSET_XYZ_RPY → [0,0,0.174,0,0,0]`,
> z 한계 −0.174 (0.011 … 0.376), `PolicyDeltaController`가 `fk(q) @ T_tool`에서 적분.
>
> `_build_safety_box()`가 `TCP_OFFSET`이 0이 아닌데 박스가 켜져 있으면 경고를 찍는다
> (`ur7e_env.py:285-297`). **경고일 뿐 막지는 않는다.**

> ### 🔧 z 바닥이 오늘 한 번 더 정정됐다 (`0.1785` → `0.185`)
> 처음에는 "관측 최솟값 = 테이블"이라고 보고 `0.1785`를 썼다. **틀렸다.** 그 최솟값의
> 40샘플이 전부 take_11 하나에서 나오는데, 거기서는 그리퍼가 빈손으로 닫힌 채
> (`grip_pos` 0.05–0.18) `fz`가 0.4초 동안 **−24 ~ −133 N**을 찍고 있다. 즉 테이블이
> 아니라 **실패한 그랩이 표면을 눌러 박은 깊이**다.
> 접촉 없는 샘플(`fz > −5 N`)만 남기면 실제 표면은 **0.1808**이고, 옛 바닥은 그보다
> **2.3 mm 아래**였다.
>
> 왜 중요한가: `PolicyDeltaController`는 **클립된 포즈를 자기 적분기에 되쓴다.** 아래로
> 미는 정책은 명령 플랜지를 정확히 "133 N이 필요했던 깊이"에 주차시키고, **이 루프에는
> 힘을 제한하는 것이 아무것도 없다.** `0.185`는 자유 표면을 확보하고 take_11의 73샘플
> (0.37%)만 버리며, 가장 낮은 성공 그랩(0.1941)보다 9 mm 아래에 있다.
> 테이블이나 베이스 마운트를 다시 앉히면 **0.190**을 쓴다 (이 박스 전체에 걸친 0.5°
> 기울기가 z로 5 mm다).

### 남은 위험

- **실기에서 한 번도 클립이 발동한 적이 없다.** 첫 실기는 반드시 `DRY_RUN`으로,
  `info["clipped"]` 빈도를 보면서 시작한다.
- 박스는 **정책/개입이 명령하는 포즈**만 막는다. 팔꿈치·어깨의 실제 스윕은 여전히
  자유다 (keepout 존은 RL 경로에 없다 — G2).
- EEF 텔레옵 쪽 `max_excursion_m = 0.5`는 워크스페이스 제한이 **아니다** — engage마다
  앵커가 새로 잡혀 예산이 0으로 리셋된다 (`03_EEF_MODE.md` §6).

### 여전히 유효한 완화책

1. **물리적 제약을 우선한다.** 로봇 주변에 실제 장애물/펜스를 두고, 팔이 물리적으로 닿을 수 있는
   범위 자체를 좁힌다.
2. **펜던트의 UR 안전 평면(Safety Planes)을 설정한다.** 이건 소프트웨어와 무관하게 컨트롤러가
   강제하며, protective stop으로 나타난다. **소프트웨어 박스와 독립적인 두 번째 방어선이다.**
   (설정 여부 **미확인** — 실기 투입 전 반드시 확인할 것.)
3. `ACTION_SCALE`과 `GOVERNOR`를 낮춰 단위 시간당 이동량을 줄인다.
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

## G3 — 그리퍼 개입 🟡 **코드는 고쳐졌고, 하드웨어에서 미확인**

> ### 🔧 정정
> 이전 판: *"`URRosBackend`가 트리거 토픽을 구독하지 않는다. 미구현."* — commit `6a0b127`에서
> 구현됐다. `GRIP_CLOSE_THR = 0.7` / `GRIP_OPEN_THR = 0.3`
> (`wrappers.py:219-220`, `:323-325`)도 더 이상 데드 코드가 아니다.

### 지금의 사실

`URRosBackend`가 `/gripper/gripper_client/target_gripper_width_percent`를 별도 구독하고
`merge_gello_state()`가 관절 6 + 트리거 1 = 7요소로 합친다
(`ros_backend.py:81`, `:90`, `:362`). 트리거 부재는 **`NaN`**으로 표현한다 —
`0.0`은 "완전 열림"이라는 정당한 값이라 센티널로 쓰면 토픽이 죽을 때마다 그리퍼를
조용히 열어버리기 때문이다. 회귀: `tests/test_gello_gripper_wiring.py` **23 passed**.

### 왜 아직 🟡인가

**하드웨어에서 한 번도 확인되지 않았다.** 2026-07-27 실기에서 통과한 것은
그리퍼 단독 경로와 GELLO 트리거 **발행**(0.000~1.000 전 구간)까지이고,
"개입 중 트리거 → 로봇 그리퍼 동작"은 미검증이다. 확인될 때까지는 아래 완화책을 유지한다.

### 완화책 (실기 판정 전까지 유효)

1. **개입 세션에서는 부서지기 쉬운 물체·손가락을 그리퍼 근처에 두지 않는다.**
2. 그리퍼를 즉시 열어야 하면 **별도 터미널에서** 서비스를 부른다 (수 초 지연 감수):
   ```bash
   ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: false}"
   ```
   > 단, 러너가 계속 `command_percent`를 쏘고 있으면 다시 덮어써진다.
   > 확실한 해제는 **러너 정지 또는 E-STOP**이다.
3. `GRASP_PENALTY`/`GripperPenaltyWrapper` 실험 시 이 갭을 명시한다 —
   "사람이 그리퍼를 시연했다"는 전제가 **아직** 성립하지 않는다.
   (`cube_in_cup`의 `GRASP_PENALTY = -0.02`는 upstream 태스크 값을 그대로 쓴 것이다.)

**판정 절차:** `04_HIL_INTERVENTION.md` §6.3. 핵심 회귀는
"트리거 퍼블리셔를 죽였을 때 그리퍼가 **저절로 열리지 않는다**".

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

> ### 🔧 2026-07-27: 자동 가드가 생겼다 (완전 해결은 아님)
> `ros2_ur_ws/run_hil_actor.sh`의 preflight [9]가 이 토픽의 **퍼블리셔 수를 직접 세고,
> 하나라도 있으면 기동을 거부한다** ("이 리그 최대 하자"). preflight [8]은
> `forward_position_controller`가 `active`면 경고한다.
> `run_real_hil.py`도 같은 가드를 갖는다. **`run_rviz_hil.py`(mock 러너)에는 없다.**
> 즉 가드는 "actor/실기 러너"에만 있고, 사람이 손으로 두 스택을 띄우는 것은 여전히 막지 못한다.

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

## G5 — 19-D `state` 순서 계약 (코드 통합 완료, runtime pin 유지) 🟢

### 사실

`serl_ur_infra/ur_env/observation_schema.py`의 다음 변경은 hardware commit `6a0b127`과
learner/hardware merge `248255f`에 통합됐다:

- 스키마 ID `hil-serl-ur-canonical-observation-v1` → **`-v2`**
- 평탄 레이아웃이 **알파벳순**으로 재정의됨:
  `gripper_pose[0:1)`, `tcp_force[1:4)`, `tcp_pose[4:10)`, `tcp_torque[10:13)`, `tcp_vel[13:19)`
- **그리퍼 인덱스가 18 → 0**. `state[..., -1]`은 TCP 각속도 z다.
- 스키마 해시가 바뀜 → **랩톱/Kanu 양쪽을 같이 올려야 통신이 성립**한다 (fail-fast로 거부됨).

이유: 평탄화를 하는 것은 우리가 아니라 upstream `SERLObsWrapper`이고, 그것이 쓰는
`gym.spaces.Dict`가 매핑을 **알파벳순으로 재정렬**한다. `proprio_keys`는 순서를 정하지 못한다.

### 2026-07-27 추가 확인

랩톱과 Kanu 서버가 **같은 해시를 광고함이 실제 왕복에서 확인**됐다:
`3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903`
(`09_HIL_ACTOR_RUNBOOK.md` §2.1). 100스텝 acceptance도 통과했고
서버가 본 `state_shape`는 `[8, 1, 19]`였다.

### 남은 위험

runtime dependency/gymnasium 변경으로 실제 flatten 순서가 달라질 수 있다. 그래서 live env
layout assertion과 laptop/server schema hash pin을 계속 유지하고, gripper index를 call site에
숫자로 재작성하지 않는다. **위 해시를 문서에서 복사해 비교하지 말고, 양쪽에서 출력해서
대조한다** (`05_COMMS_GRPC.md` §4.2).

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

`TCP_POSE_SOURCE`는 `cube_in_cup`에서도 `"driver"`다. 그러면:

- **관측** `tcp_pose` = 벤더 FK (`/tcp_pose_broadcaster/pose`)
- **명령/개입 앵커** = 우리 명목 DH (`ur_kin.fk` / `controller.tcp_cmd()`)

두 계가 다르면 관측과 명령이 서로 다른 좌표계 위에 있게 된다.
코드는 앵커 수식이 자기일관적이라 zero-jump는 유지된다고 적고 있지만,
**관측-액션 정합**은 별개 문제다.

> ### 🔧 2026-07-27: 크기는 실측됐다 (그리고 프레임은 확정됐다)
> `cube_in_cup` 23테이크의 `ur_joint_states`를 우리 FK로 재생해
> 기록된 `/tcp_pose_broadcaster/pose`와 대조한 결과, **중앙값 0.6 mm**로 일치한다
> — 단, **플랜지(tool0) 해석에서만**. 그리퍼 끝 가설에서는 174.2 mm 어긋난다.
>
> 두 가지 결론:
> 1. **명목 DH ↔ 벤더 FK의 불일치는 이 리그에서 0.6 mm 수준이다.** G7의 "두 계가 다르다"는
>    여전히 참이지만 크기는 작다. `TCP_POSE_SOURCE`를 바꿀 시급한 이유는 없다.
> 2. **관측·명령·워크스페이스 박스가 전부 플랜지 프레임으로 통일돼 있다.**
>    이건 우연이 아니라 유지해야 할 세트다 → G1의 "17.4 cm 함정".
>
> ⚠️ 그러므로 아래 완화책 2번(`TCP_POSE_SOURCE = "fk"`로 전환)을 **단독으로 실행하면 안 된다.**
> `TCP_OFFSET`·z 한계·컨트롤러 적분점까지 넷을 함께 바꿔야 한다 (G1 참조).

### 완화책

1. `03_EEF_MODE.md` §4.1의 3자세 대조(5 mm / 5 mrad)를 실기 투입 전에 실행하고 결과를 기록한다.
   (오프라인 데이터 대조는 이미 0.6 mm로 통과했다 — 위 박스.)
2. `TCP_POSE_SOURCE = "fk"`로의 전환은 **G1의 4종 세트와 함께만** 한다. 단독 변경 금지.
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

## G10 — pinned `third_party/hil-serl` 초기화 확인 🟡

```bash
cd $WT && git submodule status
# 앞의 '-'는 미초기화 상태
```

canonical/Kanu 검증에서는 pinned submodule을 사용했다. 새 checkout에서 앞에 `-`가 붙으면
`serl_launcher` import가 실패하므로 `00_SETUP_AND_SAFETY.md` §2.3에 따라 초기화한다.

---

## G11 — build/gripper shell 수정 통합 완료 🟢

`run_ur7e_gripper.sh` / `build_ur7e.sh`의 `set -u` 수정은 hardware snapshot `4171e7a`에
포함됐다. 세션 시작 시 `git status`로 별도 사용자 변경을 보존하는 원칙은 계속 적용한다.

---

## G12 — 스페이스바 데드맨이 기본값이다 🟡

`GelloIntervention(env, deadman=None)`이면 `SpacebarDeadman()`이 만들어진다
(`wrappers.py:171`). 그런데 그것은 **전역 pynput 리스너**라 다른 창에서 친 스페이스도 잡고
(`wrappers.py:82-97`), X11 오토리핏으로 깜빡이며(그때마다 앵커 재래치),
gain이 1.0으로 고정이고(`:109-110`), 하트비트 워치독이 없다.

**완화책:** 항상 `--deadman topic` + `run_hil_gui.sh`를 쓴다 (`04_HIL_INTERVENTION.md` §1).
실기 세션의 기본을 토픽 데드맨으로 바꾸는 것이 바람직하나 **미구현**이다.

---

## G13 — 리셋이 손목을 **한 바퀴 돌릴 수 있었다** 🟡 (수정됨, 실기 미검증)

### 사실

`forward_position_controller`는 raw 관절 공간에서 선형 보간하며 **2π를 모른다.**
`cube_in_cup`의 `RESET_JOINTS`는 정확히 ±π 경계 위에 있다
(`wrist_3 = -3.1331`, `shoulder_pan = 3.1382`). 실측된 파킹 자세는 `wrist_3 = +3.1795`였다.

| | |
|---|---|
| 물리적 차이 | **0.029 rad** |
| 예전 코드의 계산 | **6.3126 rad** |

옛 `go_to_reset()`은 `gap = max(abs(q - target))`만 봤으므로 (a) 거리 가드가 배선 고장처럼
보이는 에러를 냈고, (b) 가드를 통과시켰다면 **명령도 먼 길로** 나갔다:
약 10초의 맹목 슬루 + **2F-85 tool-comm 케이블이 손목에 한 바퀴 감김**(H3).
그때 실제로 일어나지 않은 유일한 이유는 `DRY_RUN=True`였다는 것뿐이다.

### 수정 (commit `d49d0f6`)

`ur_kin.wrapped_nearest(target, q)`로 목표를 **팔의 현재 회전수**로 옮긴 뒤, 거리 가드·명령·
도착 판정을 **전부 그 값으로** 한다 (`ur7e_env.py:507-540`).
팔꿈치(index 2)는 가동범위가 ±π라 일부러 감싸지 않는다 — 그래서 branch-safe 거리는
순진한 원형 거리 2.755가 아니라 **3.5281**이다.
회귀: `tests/test_reset_branch_cut.py` **10 passed**.

같이 바뀐 것: `RESET_MAX_DIST_RAD` `0.5` → **`0.9`**. 23테이크의 최종 프레임에서
`RESET_JOINTS`까지의 branch-safe 거리가 중앙값 0.615 / 최대 0.774라서, `0.5`는
**23개 중 16개의 정상 종료 자세를 거부**했다 (= 대부분의 에피소드 뒤에 `reset()`이 예외).

### 남은 위험

- **실기에서 리셋을 한 번도 실행하지 않았다.** 첫 실기 리셋은 `DRY_RUN`으로 로그만 보고,
  명령된 `wrist_3`가 현재 값 근처인지 눈으로 확인한 뒤 arm한다.
- 이 함정은 `go_to_reset()` 밖에도 있다. **`RESET_JOINTS`와 관절값을 비교하는 새 코드는
  전부 branch-cut safe여야 한다.** 순진한 차이는 동일 자세를 ~2π 떨어진 것으로 보고한다.

---

## G14 — 시스템 `python3-grpcio 1.30.2`가 손상돼 있다 🟠

### 사실

apt 패키지 `python3-grpcio 1.30.2-3build6`으로 gRPC 채널을 만들면 **에러도 로그도 없이
단일 스레드가 CPU 100%로 영구 스핀**한다. 재현·관찰 절차는 `00_SETUP_AND_SAFETY.md` §3.4.

이것이 물었던 자리:

- 2026-07-27 actor 기동 실패 4건 중 1건
- `serl_ur_infra` 테스트 4개 파일의 무한 hang
  (`test_actor_grpc_transport`, `test_actor_identity_pinning`, `test_actor_smoke`,
  `test_rlpd_receive_smoke` — venv에서는 **35 passed**)

### 왜 아직 🟠인가 (완화만 됐다)

- `run_hil_actor.sh`가 `$ACTOR_PY`를 절대경로로 `exec`하고, preflight [2]가 버전 1.30.2를
  거부하고, [3]이 `cygrpc.CompletionQueue()`를 **타임아웃 건 서브프로세스**로 실제 호출해
  코어 생존을 확인한다. → 래퍼를 통과하는 경로는 안전하다.
- **그러나 손으로 `python3`를 치는 경로는 아무것도 막지 못한다.** 그리고 증상이
  "그냥 멈춤"이라 원인 추적에 시간이 크게 든다.
- 시스템 패키지를 제거·업그레이드하는 것은 **rclpy를 깨뜨릴 수 있으므로 하지 않는다.**

### 완화책

1. gRPC를 만질 수 있는 모든 명령을 **`$ACTOR_PY` 절대경로**로 쓴다.
2. 실기 기동은 `run_hil_actor.sh`만 쓴다.
3. 새 스크립트/테스트를 추가할 때 **shebang을 `#!/usr/bin/env python3`로 두지 않는다.**
4. "멈췄다"를 만나면 **가장 먼저 인터프리터를 확인한다** —
   `ls -l /proc/<pid>/exe`, `ps -L -o pcpu,stat -p <pid>`(100% / `R`이면 이 버그 의심).

> ⚠️ 원인으로 보고된 `cygrpc.so`의 `__wrap_memcpy` 무한 루프는 **심볼·바이트 패턴 스캔으로
> 확인되지 않았다.** 기계어 수준 원인은 미확인이고, **행동은 100% 재현된다.**

---

## G15 — 분류기가 **크롭 없이 학습됐는데 액터는 크롭해서 먹인다** 🔴

> 이 절은 2026-07-28에 전면 개정됐다. 이전 판은 원인을 `decode_classifier_image()`로
> 지목했으나 **틀렸다** — 위 정정 박스 참조. 아래는 측정으로 확정된 내용이다.

### 사실

- **학습**은 크롭 없이 1280×720 full-frame을 128×128로 찌그러뜨렸다
  (kanu `hil-serl/examples/cube_classifier_pipeline.py::preprocess_frame`,
  `export_0724.py`가 `crop=None`을 넘긴다).
- **액터**는 `ur7e_env.get_im()`에서 `IMAGE_CROP` 적용 후 리사이즈한다
  (cam1 650×650, cam2 720×720 → 128×128).
- **서버는 이미지 변환을 하나도 하지 않는다.** `_classifier_observation()`은 canonical
  관측을 그대로 넘기는 passthrough다 (`grep "resize\|cv2\." rlpd_receive_server.py` → 0 hit).

즉 **분류기 입력이 분포 밖으로 나간다.** 그리고 분류기는 보상·종단의 **권위**다.

측정: 픽셀 대조 MAE **0.00**(무크롭 가설, 비트 일치) vs **21–35**(크롭),
실제 체크포인트에서 recall@0.85 **100.0% → 33.3%**.

### 왜 지금 당장 터지지 않는가

`ur_experiments/`가 대상 브랜치(`feat/gello-ur7e-humble-22.04`)에 **없어서** 거기서는
`IMAGE_CROP={}`가 적용된다 — 우연히 학습 조건과 일치한다. **액터 브랜치를 머지하는 순간
크롭이 활성화된다.** 그리고 `DRY_RUN=True`가 마지막 방어선이다.

### 해결 방향 — 크롭이 기준이다

**`IMAGE_CROP`을 되돌리지 말 것.** 크롭 값은 데이터셋 측정에서 나왔고(테이블 z 바닥,
손목 파지축 x=781) 1차 소비자는 RL 정책이다. **분류기를 그 크롭으로 재학습하는 것이
정답이다** — 현 체크포인트를 만든 학습이 150 epoch에 46초였고,
`cube_classifier_pipeline.py`에 `--cam1-crop`/`--cam2-crop`이 이미 배선돼 있다:

```
cam1  img[20:670, 340:990]  →  --cam1-crop 340,20,990,670
cam2  img[0:720, 420:1140]  →  --cam2-crop 420,0,1140,720
```

재학습 시 크롭 값을 체크포인트 옆에 sidecar로 기록해 서버가 불일치를 감지하게 할 것.

### 해결 방향 (둘 중 하나, 미결정)

1. 크롭된 이미지로 분류기를 재학습한다.
2. 정책 전처리와 분류기 전처리를 분리한다.

근거: `ur_experiments/cube_in_cup.py`의 `IMAGE_CROP` 주석 "KNOWN CONFLICT".

---

## G16 — 10 Hz 레이턴시 예산이 **사실상 소진 상태** 🟠

2026-07-27 Kanu 왕복 실측: RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**.
관측 96.1 KiB × 10 Hz = **7.9 Mbit/s**. **10 Hz 스텝 예산은 100 ms다.**

병목은 서버 추론이 아니라 **WiFi 대역폭**이다. 그리고 이 숫자는 fake-env 값이므로
실기(Stage B)에서는 센서 파이프라인 지연이 더해진다 — **상한이 아니라 하한**이다.

**완화책:** 유선으로 옮기고 같은 100스텝을 재측정한다. 그 전까지
`timeout_s`/`max_response_age_s`를 늘리지 않는다 (증상만 감춘다).
상세: `05_COMMS_GRPC.md` §5.3.

---

## 부록 — 발견된 문서 불일치 (코드가 정답)

이 조사 중 상위 문서에서 발견한 낡은 서술. 고치는 것은 각 문서 소유자의 몫이다.

| 문서 | 낡은 서술 | 코드/실제 |
|---|---|---|
| `docs/ros2/GELLO_UR7E_EEF_MODE.md:380` | P9 "yaml 기본 `v_max=0.08`" | `config/ur7e_gello_eef.yaml:238` = **0.16** (같은 문서 `:7`도 0.16이라 자기모순) |
| `serl_ur_infra/RL_RECEIVE_SERVER.md` | state 순서 = pose6, vel6, force3, torque3, gripper1 | 현재 `observation_schema.py` = **알파벳순, gripper가 index 0** |
| `serl_ur_infra/README.md` 현황표 | "워크스페이스 박스 (`ABS_POSE_LIMIT`) … ❌ config만 존재, 미작동" | **낡음.** 구현·배선됐다 → G1 |
| `serl_ur_infra/README.md` TODO | "v_max(0.1 m/s) vs ACTION_SCALE 최대(0.2 m/s) 정합 — 거버너가 풀액션을 절반으로 자름" | `ACTION_SCALE=[0.01,0.05,1.0]` × `HZ=10` = 0.1 m/s < `GOVERNOR v_max=0.12`. **이미 정합되어 있고 경고도 안 뜬다** |
| `serl_ur_infra/ur_env/envs/ros_backend.py:208` | `VERIFY(hw)` — "퍼블리셔가 best-effort면 구독이 조용히 안 뜬다, 첫 브링업에서 확인할 것" | **확인 완료.** RealSense는 RELIABLE/TRANSIENT_LOCAL → 호환. 주석만 안 지워졌다 → `06_SENSORS.md` §3 |
| `ros2_ur_ws/launch_cameras.sh:12-16` | "cam2 = CLOSE-UP (workspace)" | **cam2는 손목 카메라다** → `06_SENSORS.md` §1.1 |
| `ros2_ur_ws/src/gello_policy/config/act_deploy.yaml:35` (및 diffusion/FM 형제) | `RESET_JOINTS = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` — 주석은 그냥 "the dataset's start pose" | 그건 **banana-in-pot** 자세다. cube_in_cup과 어깨에서 0.29 rad 차이 = TCP가 13 cm 높고 10 cm 뒤. **HIL에 복사하지 말 것.** cube_in_cup 값은 `[3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331]` |
| `serl_ur_infra/README.md` 머리말 | "⚠️ UNTESTED SKELETON" | 여전히 맞다 (실기 RL 경로 미검증). 유지 |
