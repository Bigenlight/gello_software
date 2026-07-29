# 08 — 미해결 안전 갭과 임시 완화책

이 문서는 **아직 닫히지 않은 것**만 모은다. "곧 고칠 것" 같은 낙관적 표현을 쓰지 않는다.
각 항목은 코드 근거 → 왜 위험한가 → **지금 쓸 수 있는 완화책** 순이다.

> ## 🛑 게이트 선언
> **G2 · G4 · G6 · G9가 닫히고, G1 · G3 · G13이 실기에서 확인되기 전에는
> RL 정책 경로로 실기 UR7e를 구동하지 않는다.**
> `DRY_RUN`은 기본값 `True`로 둔다 (`config.py:153`, `cube_in_cup.py:227`).
>
> ⚠️ **2026-07-28에 팔이 실제로 움직였다. 그것은 이 게이트의 위반이 아니다** — 그때 돈 것은
> 정책이 아니라 **zero-policy + 사람 개입**(`run_real_hil.py --arm --scale 0.25`)이었다.
> 다만 그 세션은 **워크스페이스 박스가 꺼진 채로** 돌았다 (G1). 게이트는 그대로 유효하다.

### 현황 요약 (2026-07-29, 머지 `3f199d4` 이후)

| | 갭 | 07-27 | **지금 (07-29)** |
|---|---|---|---|
| G1 | `clip_safety_box` | 🟢 구현·단위검증 | 🟠 **구현됐지만 실기에서 돌린 경로에서는 꺼져 있었다** — 아래 참조 |
| G2 | `PolicyDeltaController` 단순화판 | 🔴 | 🔴 (변화 없음) |
| G3 | 그리퍼 개입 | 🟡 코드 수정됨 | 🟡 (변화 없음 — 07-28 세션은 그리퍼 채널을 껐다) |
| G4 | 두 퍼블리셔 충돌 / 업샘플러 스테일 | 🟠 | 🟠 **actor에도 가드 추가**(G4a 더 좁아짐), G4b는 그대로 |
| G5 | 19-D state 순서 | 🟢 | 🟢 (변화 없음) |
| G6 | staleness 시 예외 | 🟠 | 🟠 |
| G7 | 명목 DH ↔ 캘리브레이션 | 🟠 | 🟠 |
| G8 | 브로드캐스터 누락 = 조용한 오염 | 🟠 | 🟠 |
| G9 | 개입 앵커가 flange 기준 | 🟡 | 🟡 |
| G10 | pinned submodule | 🟡 | 🟡 (📌 07-29 현재 초기화돼 있음) |
| G11 | build/gripper shell | 🟢 | 🟢 |
| G12 | 스페이스바 데드맨 기본값 | 🟢 해결 | 🟢 **해결 확정** — 두 entrypoint 모두 `--deadman` 기본값 `topic` |
| G13 | 리셋 branch-cut | 🟡 수정·단위검증 | 🟡 (변화 없음 — 07-28 세션이 이 경로를 안 탔다) |
| G14 | 시스템 grpcio 1.30.2 손상 | 🟠 완화만 | 🟠 (📌 07-29 재확인: 여전히 1.30.2) |
| G15 | 분류기 전처리 ↔ 크롭 불일치 | 🔴 증명됨·미유입 | 🟡 **코드 해결(sidecar 분리) · 실기 미검증.** 잔여 결함은 크롭이 아니라 **가림** — 아래 G15 §잔여 |
| G16 | 10 Hz 레이턴시 예산 소진 | 🟠 | 🟠 유선 전환 필요. sidecar는 **+2.8% 상행**만 더한다(q95, 쌍 13.32 KiB 실측). 링크가 세션 간 **6배** 흔들리는 것이 진짜 문제 |
| **G17** | **전역 ESC 리스너** | (미기재) | 🟠 **신규 기재** — 데드맨과 별개, 아무 창의 ESC가 에피소드를 끝낸다 |
| **G18** | **`ACTION_SCALE`이 learner fingerprint에 없다** | (미발견) | 🟠 **신규** — 스케일이 바뀌어도 경고 없이 resume된다 |
| **G19** | **`checkpoint_sha256()`가 orbax 디렉터리를 pin 못 한다** | (미발견) | 🟢 **해결.** `directory_sha256()` 위임 + 두 기본 SHA 교체. G15과 **같은 변경**에서 처리됐다 |
| **G20** | **canonical demo가 없다** | (미발견) | 🟢 **해결(2026-07-29)** — 사용자가 `take_23` 제외 23개를 success로 승인했고 2,037-transition 영구 artifact를 laptop3/Kanu에서 검증 |

```bash
export WT=/home/laptop3/gello_software     # 2026-07-29 머지(3f199d4) 이후 통합 checkout이 정본
```

---

## G1 — `clip_safety_box` 🟠 **구현됐지만, 실기에서 돌린 경로에서는 꺼져 있었다**

> ### 🔧 정정 1 (2026-07-27)
> 이전 판: *"두 gym Box는 생성된 뒤 리포 어디에서도 다시 참조되지 않는다. 주석만 있고
> 클램프 코드는 없다."* — **더 이상 사실이 아니다.** commit `d49d0f6` / `ee3240e`에서
> 구현·배선됐다. `serl_ur_infra/README.md:59`의 "❌ config만 존재, 미작동" 현황표는
> 아직 낡은 채로 남아 있다 (부록 참조).

> ### 🛑 정정 2 (2026-07-29) — 상태를 🟢에서 🟠로 **내린다**
> 07-27 판은 "구현·단위검증, 실기 미검증"이라고 적었다. 그 뒤 실기 세션이 있었지만
> **박스를 켠 채로 돈 것이 아니다.**
>
> 2026-07-28에 팔을 움직인 것은 `serl_ur_infra/tests/run_real_hil.py`이고, 이 러너는
> `DefaultUR7eEnvConfig`를 base로 쓴다 (`:296`). 그 기본값은
> `ABS_POSE_LIMIT_LOW/HIGH = np.zeros((6,))` (`config.py:76-77`)이다.
> `_build_safety_box()`의 **REFUSE-DON'T-CLAMP** 설계상 0-부피 박스는 클램프하지 않고
> **박스를 끄고 경고만 찍는다.** 즉 그 세션 내내 워크스페이스 클리핑이 없었다.
>
> 📌 그 세션의 DRY RUN 300스텝을 사후 분석한 결과: **241스텝(80%)이 `cube_in_cup` 박스
> 밖**이었고 최대 이탈은 **73.9 cm**였다. 사람 개입은 박스를 존중하지 않으며,
> 박스를 켠 채로 같은 조작을 하면 클립이 상시 발동한다는 뜻이다.
>
> **그래서 다음 단계는 "박스를 켜고 DRY RUN을 다시 돌려 `info["clipped"]` 빈도를 보는 것"**
> 이다. 그 전에 arm하지 말 것.

### 지금의 사실

| | 위치 |
|---|---|
| 박스 구성·검증 | `UR7eEnv._build_safety_box()` (`ur7e_env.py:213-297`) |
| 공개 클립 (upstream 시그니처) | `UR7eEnv.clip_safety_box(pose7) -> pose7` (`:334`) |
| 컨트롤러 훅 (4×4 어댑터) | `UR7eEnv._clip_command_pose(T) -> (T, clipped)` (`:356`) |
| 배선 | `PolicyDeltaController(..., clip_pose=self._clip_command_pose)` (`:188`), 적용 지점 `policy_delta_controller.py:140-141` |
| 회귀 테스트 | `tests/test_clip_safety_box.py` — 📌 2026-07-29 재실행 **26 passed** |
| **박스 값이 있는 곳** | **`ur_experiments/cube_in_cup.py:162-167`뿐.** `DefaultUR7eEnvConfig`는 0벡터 = 비활성 |

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
> z 한계 −0.174 (**0.011 … 0.376**), `PolicyDeltaController`가 `fk(q) @ T_tool`에서 적분.
>
> ⚠️ `cube_in_cup.py:126`의 주석은 이 구간을 `0.0045 .. 0.376`이라고 적고 있는데
> **산술이 낡았다** — z 바닥이 `0.1785`에서 `0.185`로 올라갈 때 같이 갱신되지 않았다.
> `0.185 − 0.174 = 0.011`이 맞다. (코드 주석 수정은 이 문서 소유 범위 밖 → 부록)
>
> `_build_safety_box()`가 `TCP_OFFSET`이 0이 아닌데 박스가 켜져 있으면 경고를 찍는다
> (`ur7e_env.py:285-297`). **경고일 뿐 막지는 않는다.**

> ### 🔧 z 바닥이 2026-07-27에 한 번 더 정정됐다 (`0.1785` → `0.185`, commit `ee3240e`)
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

- **실기에서 한 번도 클립이 발동한 적이 없다.** 07-28 세션은 박스가 꺼진 채였다.
  첫 실기는 반드시 `DRY_RUN`으로, `info["clipped"]` 빈도를 보면서 시작한다.
- **07-28 데이터로 예고된 것:** 같은 조작을 박스 켜고 하면 클립이 **상시** 발동한다
  (DRY RUN 300스텝 중 241스텝이 박스 밖). 그건 고장이 아니라 박스가 일하는 것이지만,
  `PolicyDeltaController`가 **클립된 포즈를 자기 적분기에 되쓰기** 때문에 조작감이
  달라진다. 처음 켤 때 그 차이를 예상하고 시작할 것.
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

1. **정책** 경로는 mock 하드웨어에서만 돌린다 (`04_HIL_INTERVENTION.md` §3).
   (사람 개입만 있는 zero-policy 경로는 07-28에 실기에서 돌았다 — `04` §4.5.)
2. 실기가 불가피하면 `ACTION_SCALE`/`GOVERNOR`를 **config 기본값 이하로만** 쓴다.
   🔧 그 기본값은 2026-07-28에 검증된 EEF 텔레옵 값에 맞춰 **균일 1.25배로 재조정**됐다
   (commit `a5f9890`). 07-27 판이 적어 둔 `[0.01, 0.05, 1.0]` / `v_max 0.12 / w_max 0.60 /
   dq_step_max 0.05`는 **낡았다.** 지금 값:

   | 층 | 값 | 근거 |
   |---|---|---|
   | `ACTION_SCALE` | `[0.0125, 0.0625, 1.0]` → 10 Hz에서 0.125 m/s · 0.625 rad/s | `config.py:73` |
   | `GOVERNOR` | `v_max 0.15 / w_max 0.75 / dq_step_max 0.0625` (양축 1.200x 헤드룸) | `config.py:86-90` |
   | `UPSAMPLER` | `hz 250.0 / max_step_rad 0.0025` (= 0.625 rad/s) | `config.py:98-108` |

   **세 층의 비율을 유지한 채로만 움직인다.** `run_real_hil.py --scale`이 그렇게 한다.
   한 층만 올리면 다음 층이 조용히 잘라내고 "저장 == 실행" 불변식이 깨진다.
   더 올리지 말 것: 0.16 m/s를 내려면 `max_step_rad`가 0.0032여야 하는데
   `ur7e_gello.yaml:56-63`이 250 Hz에서 **~0.00314를 천장으로 못박아 뒀다.**
3. `info["held"]` / `info["reject_reason"]` 빈도를 **매 세션 로깅**한다.
   `NO_IK`/`STEP_LIMIT`이 늘어나면 특이점에 접근 중이라는 신호다.
   📌 baseline: 07-28 실기 100스텝에서 **held-rate 0 %**였다.
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
"개입 중 트리거 → 로봇 그리퍼 동작"은 미검증이다.

🔧 **2026-07-28 세션도 이 갭을 닫지 못했다.** `run_real_hil.py`는 기본값이 그리퍼
**비활성**(`ACTION_SCALE[2] = 0.0`)이라 그 채널이 아예 실행되지 않았다.
`--gripper`를 명시해야 켜지고, 그 모드에서는 CSV의 `ia6`가 기록되면서 실행도 된다.
확인될 때까지는 아래 완화책을 유지한다.

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

> ### 🔧 자동 가드 현황 (2026-07-29 갱신 — 세 겹이 됐다)
> | 어디 | 무엇을 하나 | 근거 |
> |---|---|---|
> | `run_hil_actor.sh` preflight [9] | 이 토픽의 **퍼블리셔 수를 직접 세고, 하나라도 있으면 기동 거부** ("이 리그 최대 하자"). `--arm` 여부와 무관 | `run_hil_actor.sh:480-503` |
> | 같은 스크립트 preflight [8] | `forward_position_controller`가 `active`면 **경고** | `:440-` |
> | `run_real_hil.py` | `--arm` 시 `count_publishers()`로 세고 거부 | `:439`, `:455` |
> | **`run_remote_rlpd_actor.py` (신규, `c069e79`)** | `--arm` 시 퍼블리셔가 있으면 거부, **구독자가 0이어도 거부**("컨트롤러가 안 떠 있다") | `:156-198` |
>
> **`run_rviz_hil.py`(mock 러너)에는 여전히 없다.** 그리고 사람이 손으로 두 스택을
> 띄우는 것은 어떤 가드도 막지 못한다 — 그래서 G4a는 🟠이지 🟢이 아니다.

### G4b — 업샘플러에 타깃 스테일 정책이 없다

```python
# ros_backend.py:417-421
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

## G6 — 센서 staleness 시 safe-stop이 아니라 **예외** 🟠

| 위치 | 동작 | 코드가 인정하는 것 |
|---|---|---|
| `/joint_states` 0.2 s 초과 | `RuntimeError` | `TODO(together): safe-stop policy (freeze + operator prompt) instead of raise` (`ur7e_env.py:652-655`) |
| 카메라 0.5 s 초과 / 부재 | `RuntimeError` | `TODO(together): safe-stop policy instead of raise` (`ur7e_env.py:751-757`) |
| tcp_pose 부재/낡음 | `RuntimeError` | `:660-668` |
| 첫 `/hil/deadman` 뒤 0.5 s 초과 | `DeadmanHeartbeatStaleError` | **의도된 fail-stop**: silence를 DISENGAGE로 간주하지 않고 정책 액션 전송 전에 액터 종료 |

즉 센서가 끊기면 **러너가 크래시**한다. 결과적으로 팔은 마지막 명령 자세에서 멈추지만
(G4b의 예외 상황 제외), 이건 **설계된 안전 정지가 아니라 부작용**이다.
`info`에 `held`/`reject_reason`을 노출하고 staleness 정책을 통일하는 것은
README TODO에 남아 있다 ("HOLD/reject_reason을 step info로 노출 + staleness safe-stop 정책
통일 + UR fault recovery").

단, `/hil/deadman` 행은 이 미해결 센서 정책과 구분한다. 정상적인 최신
`engaged=0`만 정책 복귀를 허용하고, 첫 수신 이후 heartbeat loss는 명시적으로 예외를
전파해 `run_remote_actor`를 빠져나온다. actor CLI의 `finally`가 network/env를 닫으며,
단절을 감지한 틱에는 하위 env/FPC로 정책 액션이 전달되지 않는다. 첫 메시지가 전혀 없는
경우도 `cube_in_cup`의 15 s 시작 가드가 거부한다.

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
| F/T 브로드캐스터 | force/torque 6개가 **전부 0** — 예외 없음 | `ur7e_env.py:685-690` |
| 그리퍼 `position_percent` | `curr_gripper_pos = 0.0` = **"완전 열림"으로 보임** | `ur7e_env.py:681-683` |
| (같은 성질) actor preflight [7b] | 그리퍼 토픽 부재는 **WARN이지 FAIL이 아니다** — 그리퍼 없이도 actor가 뜬다 | `run_hil_actor.sh:418-439` |

**스키마 해시로는 절대 안 잡힌다** — dtype/shape는 완벽히 맞기 때문이다.
`validate_canonical_observation`도 dtype/shape/유한성만 본다.

**완화책:** `06_SENSORS.md` §3.1의 topic hz 루프 + "팔을 살짝 밀어 wrench가 변하는지",
"그리퍼를 실제로 닫아 percent가 변하는지"를 매 세션 눈으로 확인한다.

---

## G9 — 개입 앵커가 **flange 기준**이다 (TCP 오프셋 미배선) 🟡

```python
# wrappers.py:235-237
def _leader_T(self, q_lead):
    # TODO: @ T_tool_L (flange-only for the skeleton)
    return fk(q_lead)
```

env는 `TCP_OFFSET_XYZ_RPY`로 `self.T_tool`을 만들어 두었지만 (`ur7e_env.py:179-182`),
개입 앵커 계산에는 쓰이지 않는다. `serl_ur_infra/README.md:80` TODO에도 남아 있다
("`GelloIntervention._leader_T`에 TCP_OFFSET 배선 (config는 있음, 현재 플랜지 기준)").

> 🔎 **07-28 실기 결과와의 관계:** 그 세션에서 frame-map이 단위행렬로 확인됐지만
> (`04` 머리말), 그건 **축 정렬**을 본 것이지 **회전 중심**을 본 것이 아니다.
> `cube_in_cup` / `run_real_hil.py` 모두 `TCP_OFFSET_XYZ_RPY = 0`이라 이 갭은 현재
> 수치적으로 발현되지 않는다 — 오프셋을 켜는 순간 발현된다. G1의 "네 개 한 세트" 참조.

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

## G12 — 스페이스바 데드맨이 기본값이었다 🟢 **닫힘**

> ### 🔧 정정 (2026-07-29)
> 07-27 판의 마지막 문장 *"실기 세션의 기본을 토픽 데드맨으로 바꾸는 것이 바람직하나
> **미구현**이다"*는 **더 이상 사실이 아니다.** 실기에서 쓰는 두 entrypoint가 모두
> `--deadman`을 명시적으로 전달하고 **기본값이 `topic`이다**:
>
> | entrypoint | 근거 |
> |---|---|
> | `scripts/run_remote_rlpd_actor.py` | `:51-62` (`default="topic"`), env로 전달은 `:220` |
> | `tests/run_real_hil.py` | `:651-655` (`default="topic"`) |
> | `tests/run_rviz_hil.py` (mock) | `--deadman topic`을 **줘야** 한다 — 여기만 여전히 opt-in |

### 남은 잔여 위험 (그래서 이 절을 지우지 않는다)

`GelloIntervention(env, deadman=None)`을 **직접** 만들면 여전히 `SpacebarDeadman()`이
붙는다 (`wrappers.py:189`). 그것은 **전역 pynput 리스너**라 다른 창에서 친 스페이스도 잡고
(`wrappers.py:100-120`), X11 오토리핏으로 깜빡이며(그때마다 앵커 재래치),
gain이 1.0으로 고정이고(`:126-127`), 하트비트 워치독이 없다.

**완화책:** 새 러너/스크립트를 쓸 때 `deadman=`을 **반드시 명시**한다.
운영에서는 `--deadman topic` + `run_hil_gui.sh`를 쓴다 (`04_HIL_INTERVENTION.md` §1).
토픽 소스는 첫 메시지를 15 s 안에 못 받으면 시작을 거부하고, 일단 받은 뒤 0.5 s
끊기면 정책으로 되돌아가지 않고 액터를 fail-stop한다.

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
도착 판정을 **전부 그 값으로** 한다 (`go_to_reset()` = `ur7e_env.py:520`, wrap = `:567-591`).
팔꿈치(index 2)는 가동범위가 ±π라 일부러 감싸지 않는다 — 그래서 branch-safe 거리는
순진한 원형 거리 2.755가 아니라 **3.5281**이다.
회귀: `tests/test_reset_branch_cut.py` 📌 2026-07-29 재실행 **10 passed**.

같이 바뀐 것: `cube_in_cup`의 `RESET_MAX_DIST_RAD` `0.5` → **`0.9`** (`cube_in_cup.py:103`;
`DefaultUR7eEnvConfig` 기본값은 `1.5`, `config.py:53`). 23테이크의 최종 프레임에서
`RESET_JOINTS`까지의 branch-safe 거리가 중앙값 0.615 / 최대 0.774라서, `0.5`는
**23개 중 16개의 정상 종료 자세를 거부**했다 (= 대부분의 에피소드 뒤에 `reset()`이 예외).

### 남은 위험

- **실기에서 이 리셋을 한 번도 실행하지 않았다.**
  🔧 2026-07-28에 팔이 움직였지만 `run_real_hil.py --reset-mode startup`은
  **기동 시점의 실제 관절을 그대로 `RESET_JOINTS`로 잡는다** — ±π 경계를 건드리지 않으므로
  이 경로가 실행되지 않았다. 이 갭은 그 세션으로 닫히지 **않았다.**
- 닫으려면 `cube_in_cup`의 `RESET_JOINTS`를 쓰는 경로(actor 또는 `--ur-config-module`)로
  가야 한다. 첫 실기 리셋은 `DRY_RUN`으로 로그만 보고,
  명령된 `wrist_3`가 현재 값 근처인지 눈으로 확인한 뒤 arm한다 (`07` E14).
- 이 함정은 `go_to_reset()` 밖에도 있다. **`RESET_JOINTS`와 관절값을 비교하는 새 코드는
  전부 branch-cut safe여야 한다.** 순진한 차이는 동일 자세를 ~2π 떨어진 것으로 보고한다.
  `cube_in_cup.py:82-89`가 그 이유를 데이터로 적어 뒀다 (시연 표본의 96 %가 `wrist_3 < -π`).

---

## G14 — 시스템 `python3-grpcio 1.30.2`가 손상돼 있다 🟠

### 사실

apt 패키지 `python3-grpcio 1.30.2-3build6`으로 gRPC 채널을 만들면 **에러도 로그도 없이
단일 스레드가 CPU 100%로 영구 스핀**한다. 재현·관찰 절차는 `00_SETUP_AND_SAFETY.md` §3.4.

📌 2026-07-29 재확인: `python3 -c "import grpc; print(grpc.__version__)"` → **여전히 `1.30.2`**.
(import 자체는 안전하다. 채널을 만드는 순간 걸린다.)

이것이 물었던 자리:

- 2026-07-27 actor 기동 실패 4건 중 1건
- `serl_ur_infra` 테스트 4개 파일의 무한 hang
  (`test_actor_grpc_transport`, `test_actor_identity_pinning`, `test_actor_smoke`,
  `test_rlpd_receive_smoke` — venv에서는 **35 passed**)

> 🔎 **어디에 걸리고 어디에 안 걸리는가** — 헷갈리기 쉬운 지점이다.
> | 스크립트 | 인터프리터 | 왜 |
> |---|---|---|
> | actor (`run_hil_actor.sh` 경유) | **venv 필수** | gRPC를 만진다. 래퍼가 강제한다 |
> | `serl_ur_infra` pytest | **venv 필수** | 위 4개 파일이 gRPC를 만진다 |
> | `tests/run_real_hil.py` | 시스템 `python3` **가능** | gRPC를 import하지 않는다 (`grep -rn "import grpc" ur_env/envs/ tests/run_real_hil.py` → 0 hit). 대신 ROS 오버레이가 필요하다 |
> | `observation_schema.py` 라이브 출력 | 시스템 `python3` **가능** | 의존성 경량 모듈 |
> | `ur_gello_bringup` pytest | 시스템 `python3` **권장** | `launch` 모듈이 필요하다 |

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

## G15 — 분류기가 **크롭 없이 학습됐는데 액터는 크롭해서 먹인다** 🟡 코드 해결 · 실기 미검증

> ### ✅ 2026-07-29 (후속): **닫혔다 — 분류기에게 무크롭 sidecar를 따로 준다**
> 액터가 분류기 전용 무크롭 원본 JPEG를 관측에 실어 보낸다(약 2 Hz, 팔 정지 시).
> 정책 크롭은 **안 건드린다**. proto 변경 없음, **schema hash 변경 없음**.
> 상세는 아래 §"✅ 채택된 해결". 같은 변경에서 **G19도 함께** 고쳤다 — 따로 하면
> 서버가 조용히 뜨고 reward가 영구 0이 된다.
>
> **🔴 두 가지를 같이 기억할 것:**
> 1. **실기 미검증.** 코드·단위테스트까지다.
> 2. **가림(occlusion)은 안 고쳐졌다** — 아래 §잔여. 그건 크롭이 아니라 카메라 배치 문제다.
>
> 아래 07-28/29 판의 기록은 **그대로 보존한다.** 이 갭이 왜 최상위 블로커였는지,
> 그리고 원인 지목이 한 번 틀렸었다는 사실이 다시 필요해질 수 있다.

> ### 🛑 2026-07-29 (선행 판): 이 갭은 **악화됐다** (유예가 끝났다)
> 07-28 판은 *"현재 프로덕션은 안 망가져 있다 — `ur_experiments/`가 대상 브랜치에 없어서
> `IMAGE_CROP={}`가 적용된다. **actor 브랜치를 머지하는 순간 유입된다**"* 라고 적었다.
>
> **머지가 일어났다.** `3f199d4`가 `serl_ur_infra/ur_experiments/`를
> `feat/gello-ur7e-humble-22.04`로 가져왔다. 확인:
>
> ```bash
> git -C $WT log --oneline -1 3f199d4
> ls $WT/serl_ur_infra/ur_experiments/cube_in_cup.py   # 존재한다
> ```
>
> 즉 `EXP_NAME=cube_in_cup`(= `run_hil_actor.sh`의 기본값)으로 actor를 띄우면
> **크롭이 활성이고 분류기 입력은 분포 밖이다.** 남은 방어선은 `DRY_RUN=True` 하나뿐이며,
> 그것은 **팔만 막고 보상/종단은 막지 않는다** — 서버 분류기는 그대로 돌아가고
> 잘못된 reward가 replay buffer에 들어간다.
>
> **그러므로 실센서 Stage B를 돌리기 전에 이것부터 처리한다.**

> ### 🔧 원인 지목 정정 (2026-07-28) — 이전 판은 틀린 모듈을 지목했다
> 이 문서와 `06_SENSORS.md`가 원인으로 지목한
> `reward_classifier_runtime.decode_classifier_image()`는 **ZMQ 뷰어 전용 모듈**이고
> (`serl_ur_infra/remote_reward_classifier_server.py`가 쓴다), gRPC 경로(port 50053)와
> **호출 관계가 전혀 없다.** 그 함수가 크롭 없이 전체 프레임을 리사이즈하는 것은
> **그쪽 용도에서는 올바르다** — 뷰어는 raw 카메라 토픽을 직접 구독해서 canonical
> observation을 아예 보지 않는다. **고칠 대상이 아니다.**
> 서버 `rlpd_receive_server._classifier_observation()`은 이미지 변환을 **하나도** 하지 않는
> strict-validating passthrough다 (`grep -n "resize\|cv2\." rlpd_receive_server.py` → 0 hit).
> 이 문구를 근거로 서버 코드를 고치면 **엉뚱한 모듈을 건드리게 된다.**
> 반증 시도 6종이 전부 실패했다. 전체 내용은
> `serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` §12.

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

### ~~왜 지금 당장 터지지 않는가~~ → ~~**지금 터진다**~~ → sidecar가 경로를 갈랐다

`ur_experiments/`는 이제 대상 브랜치에 **있다** (머지 `3f199d4`). 따라서
`cube_in_cup` config를 쓰는 모든 경로에서 `IMAGE_CROP`이 활성이다.
`DRY_RUN=True`는 **팔만** 막는다 — 관측 → 서버 분류기 → reward → replay 삽입은 그대로 돈다.

**위 문단은 sidecar 이전 기준이다.** 이제 `IMAGE_CROP`은 여전히 활성이지만 **정책 관측에만**
적용되고, 분류기는 같은 스텝의 무크롭 원본을 따로 받는다. 크롭이 활성이라는 사실 자체는
더 이상 reward 오염의 근거가 아니다.

### ✅ 채택된 해결 — **재학습이 아니라 분리(decoupling)다**

> **🔧 이전 판의 권고를 뒤집는다 (보존).** 07-28/29 판은 *"분류기를 그 크롭으로 재학습하는 것이
> 정답이다 — 150 epoch에 46초"* 라고 적었고, 아래 `--cam1-crop`/`--cam2-crop` 인자와
> 좌표 순서 경고를 근거로 달았다. **재학습은 채택되지 않았다.** 그 절차 설명 자체는 여전히
> 정확하므로 재학습을 다시 검토할 때를 위해 남겨 둔다:
>
> ```
> cam1  img[20:670, 340:990]  →  --cam1-crop 340,20,990,670
> cam2  img[0:720, 420:1140]  →  --cam2-crop 420,0,1140,720
> ```
>
> ⚠️ **좌표 순서가 뒤집힌다.** 이 리포는 `img[y0:y1, x0:x1]`로 저장하고, 파이프라인
> 인자는 `x0,y0,x1,y1`을 받는다 (`cube_in_cup.py:196-197`).

**액터가 분류기에게 무크롭 이미지를 따로 보낸다.** 정책은 측정된 `IMAGE_CROP`을
**그대로** 유지한다. 구현은 `serl_ur_infra/ur_env/classifier_sidecar.py`이고,
설계 근거 전문이 그 모듈 docstring에 있다.

| | 값 (코드에서 확인) |
| --- | --- |
| 관측 키 | `classifier` (`CLASSIFIER_SIDECAR_KEY`) — **중첩 맵**이라 정책 텐서 이름과 충돌 불가, `pop` 하나로 통째 제거 |
| 내용물 | `cam1_jpeg` / `cam2_jpeg` — **무크롭 전체 화각**을 랩톱에서 128×128로 resize 후 JPEG 인코딩한 것 (1-D uint8) |
| 입력 계약 id | `CLASSIFIER_INPUT_ID = "fullframe-jpeg-passthrough-v1"` — **이름은 역사적이다.** 최초 설계(원본 바이트 passthrough)에서 온 문자열인데, 이 id가 기록하려는 의미("전체 화각, 무크롭")는 그대로라서 유지한다 |
| 부착 주기 | 5스텝(HZ=10 → **약 2 Hz**), **팔이 정지**했을 때만. 종단 예정 스텝은 정지 게이트를 무시하고 무조건 부착 |
| 정지 판정 | TCP 선속도 ≤ **0.05 m/s** (`stationary_speed_max`) |
| 에스컬레이션 | 직전 확률 ≥ **0.05**이면 매 스텝 부착으로 전환(threshold 0.2보다 **낮게** 잡아 임계 교차 스텝을 놓치지 않는다) |
| 프레임당 상한 | **512 KiB** (`MAX_SIDECAR_JPEG_BYTES`) — 초과 시 loud fail |

**왜 이것이 맞는가:**

- **proto 변경이 없다.** `proto/actor_transport.proto`는 이미 일반 named-tensor 맵이다
  (`Tensor{path,dtype,shape,data}` + `Observation{repeated Tensor tensors}`). 새 관측 키에
  proto를 손댈 이유가 없다.
- **observation schema hash가 안 바뀐다.** 해시는 `ur_env/observation_schema.py`의
  `CANONICAL_OBSERVATION_SPEC` **문서**에서 나오지 wire payload에서 나오지 않는다.
  값은 그대로 `3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903`이다
  (이 문서 작성 중 `CANONICAL_OBSERVATION_SCHEMA_HASH`를 직접 계산해 대조함).
  sidecar는 canonical 검증 **전에** `actor_network.py`에서 벗겨진다 —
  `validate_canonical_observation()`은 여분 키를 거부하므로 벗기지 않으면 즉시 터진다.
  **따라서 기존 pin 자리는 전부 그대로 유효하다.**
- **무크롭 전체 화각을 보낸다.** `ur7e_env.get_im()`이 decode/crop 전에 이미
  `jpeg, age = self.backend.get_image(key)`로 원본을 들고 있다(ROS 토픽이
  `/camX/camX/color/image_raw/compressed`이므로). 크롭이 닿기 전의 그 화각이 sidecar로 간다.
- **서버가 뷰어와 같은 레시피로 푼다.** `decode_classifier_frames()`는
  `reward_classifier_runtime.decode_classifier_image()`와 비트 단위로 같아야 하고
  (imdecode → **무크롭** → `resize(128,128)` → RGB → batch axis),
  `tests/test_classifier_sidecar.py`가 실제 JPEG로 `np.array_equal`을 건다.
  그 뷰어 경로는 **2026-07-29 실기 검증됨**.
- **threshold 측정값이 살아남았다.** 분류기가 계속 무크롭을 먹으므로
  `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 스윕 전체가 **그대로 유효하다.**
  재학습이었다면 그 수치가 전부 무효가 되고 스윕을 다시 돌려야 했다 —
  분리를 택한 가장 큰 실익이다.

**여전히 아닌 것:** `IMAGE_CROP`을 지우는 것. 그러면 정책 입력이 1:1로 눌려
가로가 세로의 0.5625로 압축된다. 근거: `cube_in_cup.py:181-210`의 "KNOWN CONFLICT (G15)" 주석.

### 대역폭 — **측정이 설계를 두 번 바꿨다**

> **🔧 정정 1.** 이전 판(및 handoff §5.5 표)은 분류기 이미지를 따로 보내면 대역폭이
> **"약 2배"**가 된다고 적었다. 그 추정은 **매 스텝 10 Hz로 raw uint8**을 보낸다는 가정이었다.
>
> **🔧 정정 2 (더 중요).** 그 다음 판은 **원본 720p JPEG를 그대로 통과(passthrough)**시키고
> 그 크기를 **약 68 KiB/장(쌍 136 KiB)**으로 적었다. **실측이 그 설계를 죽였다.**
> 68 KiB는 q75 재인코딩에서 나온 값이었고, 카메라 노드는 `jpeg_quality = 95`
> (튜닝 안 된 ROS `image_transport` 기본값)로 돈다.

**실측된 원본 프레임 크기 — passthrough가 불가능한 이유:**

| | 값 (실측) |
| --- | --- |
| 라이브 720p 프레임 | **206 KiB (cam1) / 194 KiB (cam2)** → 쌍 **400 KiB** |
| 2 Hz 상행 | **+6.55 Mbit/s** → 합계 **14.46 Mbit/s** |
| 13 Mbit/s 링크 대비 | **링크 용량을 통째로 초과** |
| 부착 스텝 스파이크 | **+252 ms** @13 Mbit/s (+72 ms @45.6 Mbit/s) — 예산 **100 ms** |

**sidecar 하나가 제어 루프를 날려버린다.** 그래서 설계가 바뀌었다.

**채택된 것: 랩톱이 128×128 resize를 직접 하고 그것을 JPEG로 인코딩한다.**

- resize는 **서버가 어차피 했을 바로 그 결정적 `cv2.resize` 호출**이다. 어느 호스트가
  돌려도 비트 단위로 같고, `tests/test_classifier_sidecar.py`가 뷰어 출력과 대조해 강제한다.
- 유일하게 내주는 것은 **128×128에서의 JPEG 1세대 추가**다.
  > **🔧 즉 "새 압축 아티팩트 0"이라는 이전 판 문구는 더 이상 맞지 않는다.** 아티팩트는
  > 생긴다. 근거는 "없다"가 아니라 **"측정했더니 주변 센서 노이즈와 같은 수준"**이다(아래).

**실측 페이로드** (카메라당 100프레임, `cv2.resize(bgr,(128,128))` + `cv2.imencode('.jpg', 95)`,
protobuf 프레이밍 포함):

| quality | 쌍당 크기 | 기각된 720p passthrough 대비 |
| ---: | ---: | ---: |
| **95 (채택)** | **13.32 KiB** | **30.0배 작다** |
| 90 | 9.24 KiB | 43.3배 |
| 75 | 5.57 KiB | 71.8배 |

**대역폭** (관측 10 Hz + sidecar 2 Hz, q95): 합계 **8.129 Mbit/s** = 현재 7.911 대비 **+2.8%**.
13 Mbit/s 링크의 **62.5%**, 45.6 Mbit/s 링크의 17.8%.
**10 Hz 에스컬레이션 최악의 경우도 9.00 Mbit/s로 두 링크 모두에 들어간다** —
720p 설계에서는 그 경우가 WiFi 링크의 **313%**였다.

**부착 스텝 스파이크 (q95):** **+8.4 ms** @13 Mbit/s, +2.4 ms @45.6
(720p였다면 +252.0 / +71.9 ms). 옛 RTT 분포에 얹으면 **p99 꼬리만** 예산을 넘고, 그것도
열화된 링크에서 **5.5 ms**만큼(105.5 ms)이다. 빠른 링크에서는 p99가 99.5 ms로 **예산 안**이다.
- **하류는 아무것도 안 바뀐다.** wire 키는 그대로 `cam1_jpeg`/`cam2_jpeg`이고,
  `decode_classifier_frames()`는 여전히 뷰어의 imdecode → resize(128,128) → RGB → `[None]`
  레시피를 돈다 — 이미 128×128인 페이로드에서 resize가 no-op이 될 뿐이다.
  **서버는 어느 호스트가 resize했는지 알 필요가 없다.**
- **안 바뀐 것:** sidecar는 여전히 **무크롭 전체 화각**이다. 그게 핵심이고, 옮긴 것은
  전송 해상도뿐이다.

> **📌 크기를 "나쁜 링크 기준"으로 잡은 것이 핵심 판단이다.** §G16이 보여주듯 이 링크는
> 세션 간 약 6배 흔들린다. 좋은 날 기준으로 잡으면 나쁜 날 제어 루프가 죽는다.

#### 왜 q95이고 더 낮추지 않았나 — **직관과 반대다**

측정에 **대조군 둘**을 넣었고, 그 결과가 순진한 기대를 뒤집는다:

| | 판정 경계 근처 \|Δp\| | 경계 뒤집힘 (20개 중) |
| --- | --- | ---: |
| 720p → JPEG 왕복 → 720p **(대조군)** | ≤ 0.010 | **0** |
| **주변 센서 노이즈**, 프레임 간 | 0.027 – 0.102 | — |
| **128×128 q95 (채택)** | 0.022 – 0.090 | 0 – 7 |
| 128×128 q75 | 0.118 – 0.198 | **14 – 16** |

**JPEG 압축 자체는 사실상 공짜다** — 720p 왕복 대조군이 ≤0.010에 뒤집힘 0이다.
교란은 **128×128에서 양자화한다는 것**에서 온다. 그 해상도에서는 8×8 DCT 블록이
**프레임의 1/16**을 덮는다. 720p에서는 같은 블록이 다운샘플 후 서브픽셀이 되어 평균으로 사라진다.

**q95의 교란은 주변 센서 노이즈와 같은 수준이다** — 경계 근처에서 재인코딩이 확률을 움직이는
정도가 **그냥 다음 카메라 프레임을 읽는 것과 비슷하다.**

**q75로 내리면** 0.127 Mbit/s와 스파이크 4.9 ms를 아끼는 대신 교란이 **5–8배** 커진다.
**분류기가 reward와 termination의 권위인 이상 이것은 나쁜 거래다.**
무손실 PNG도 재 봤다: 쌍당 **55.3 KiB로 두 링크 모두 예산 초과**이고, 사는 것은
**이미 노이즈 수준인 교란**을 없애는 것뿐이다.

> ### ⚠️ 이 표의 경계 근처 수치는 **합성(SYNTHETIC)이다** — 지우지 말 것
> 큐브를 컵에 **합성해 넣고 알파 블렌딩**한 프레임이다. 뒤집힘 횟수를
> **품질 등급을 서로 비교하는 스트레스 테스트**로 읽어야지 **예측된 현장 오류율로 읽으면 안 된다.**
> **믿을 수 있는 부분은 순위다**(q95 ≈ 주변 노이즈 ≪ q75). **절대값은 아니다.**
>
> 실제 100프레임은 전부 **p = 0.0033–0.0165**에 있었다(큐브가 테이블 위, 팔은 주차 상태) —
> **"확신하던 것이 계속 확신한다"** 결과이고 증거로서는 약하다.
>
> **🔴 열린 항목:** 이것을 제대로 닫으려면 **큐브가 실제로 컵에 들어가 있는 실제 프레임**이
> 필요하고, 그건 **조작자가 큐브를 옮겨 줘야** 얻어진다. 아직 안 됐다.

`ActorRunSummary`가 부착/미부착 왕복 시계열을 **따로** 집계한다
(`sidecar_round_trip_ms_mean/max` vs `plain_round_trip_ms_mean/max`) —
실기에서 추정이 아니라 실측으로 확인하라고 만든 필드다.

### 성긴 분류는 타협이 아니라 **설계 속성**이다

매 스텝 분류하면 성공 판정이 흔들릴 기회가 그만큼 많아진다. 큐브를 놓은 뒤 장면이
잠깐 가라앉게 두면 판정이 더 안정적이다. **부착 빈도를 낮춘 것은 대역폭 때문만이 아니다.**

시간 평활은 **기본값이 꺼져 있다**(`--success-confirmations 1`,
`scripts/run_rlpd_receive_server.py`).

**구현 방식:** 보고되는 확률은 최근 `confirmations`개 순간 확률의 **`min()`**이다.
따라서 `min(window) > threshold`가 **정확히 N연속 확인**과 같아지고,
`success == (probability > threshold)` 불변식이 **두 모드 모두에서 구조적으로 성립한다.**
기본값 1에서는 창에 방금 분류한 프레임만 있으므로 **보고 확률 = 순간 sigmoid**이고
**라이브 뷰어 값과 비트 단위로 같다.** 근거는 코드 주석에 있다: 현 체크포인트는
threshold 0.2에서 잘 동작하고, 평활된 확률이 뷰어와 조용히 달라지면
*"뷰어는 0.9인데 서버는 왜 실패라고 하지"* 를 디버깅하는 비용이 더 크다.

### 🔴 잔여 — 이것은 **안 고쳐졌다**: 팔 가림(occlusion)

**sidecar는 전처리 불일치를 고쳤지 시야 문제를 고치지 못한다.** 정직하게 적는다:

- `take_21`은 @0.85에서 recall **0.0%**, @0.05에서도 **57.9%**다.
- 팔이 cam1 시야를 쓸고 지나가는 동안 확률이 **0.005 → 1.0**으로 진동한다.
- 원인은 **전처리도 라벨도 아니다** — 60+ 프레임 육안 검수로 라벨 오염 0% 확인됨.
  원인은 **시야/가림**이다.

정지 게이트가 이것을 **완화**한다(움직이는 동안은 아예 안 물어본다). 그러나 진짜 해결은
**팔이 가로지르지 않는 카메라 배치**다. 그때까지 이 실패 모드는 남아 있다.
`REWARD_CLASSIFIER_THRESHOLD_KO.md`가 반복해 말하듯 **어떤 threshold로도 구제되지 않는다.**

> 🪤 **"G15가 닫혔으니 reward를 믿어도 된다"로 읽지 말 것.** 닫힌 것은 크롭 불일치다.
> 그리고 **실기에서 아직 한 번도 안 돌렸다** — 코드와 단위테스트까지다.

---

## G16 — 10 Hz 레이턴시 예산이 **사실상 소진 상태** 🟠

📌 2026-07-27 Kanu 왕복 실측(그날의 기록이다 — 링크가 바뀌면 다시 재야 한다):
RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**.
관측 96.1 KiB × 10 Hz = **7.9 Mbit/s**. **10 Hz 스텝 예산은 100 ms다.**

병목은 서버 추론이 아니라 **WiFi 대역폭**이다. 그리고 이 숫자는 fake-env 값이므로
실기(Stage B)에서는 센서 파이프라인 지연이 더해진다 — **상한이 아니라 하한**이다.

### 🚨 2026-07-29 재측정 — **저장된 상수를 믿지 말고 세션마다 다시 재라**

이 링크는 세션 간 **약 6배** 흔들린다. 리포에 저장된 값이 셋인데 전부 다르다:

| 출처 | 실효 대역폭 |
| --- | --- |
| 이 문서 / handoff (07-27 판) | 약 **13 Mbit/s** |
| `REWARD_CLASSIFIER_LIVE_KO.md` §9 | 5.7 MiB/s ≈ **47.8 Mbit/s** |
| **07-29 실측** | 약 **83 Mbit/s** |

**셋 다 진짜 관측일 가능성이 높다.** 13 Mbit/s는 오류가 아니라 **열화된 상태의 실제 관측**이었을
것이고 **다시 나타날 수 있다.** 그러니 **어떤 숫자도 상수로 쓰지 마라 — 07-29 값 포함이다.**

**07-29 측정값 (조건 포함 — 조건 없는 숫자는 쓸모없다):**

| | 값 |
| --- | --- |
| 링크 | WiFi `wlp5s0` → SSID `iptime_709`, **2.4 GHz ch.3**, PHY 216–270 Mb/s, 신호 −42/−44 dBm |
| 처리량 | **약 83 Mbit/s 중앙값** (min 75.5 / max 98.5) — 32 MiB 비압축 업로드 12회, **multiplexed SSH 안에서** |
| ICMP RTT | p50 **1.75** / p95 9.99 / p99 24.4 / max 38.3 ms (600 샘플, 손실 0%) |
| 경로 | **캠퍼스 4홉** `192.168.0.1 → 10.20.44.1 → 10.22.2.101 → kanu`. **WAN 아니다** |
| 측정 조건 | **유휴 리그** — 카메라·액터·조작자 트래픽 **전무**. 2.4 GHz에 **AP 41개** |

`ssh`-실효 값이 맞는 기준이다 — 액터의 gRPC가 **SSH 터널 안으로** 흐른다.
암호화는 천장이 아니다(chacha20 ↔ aes128-gcm이 77 → 83으로만 움직였다).
**2.4 GHz 라디오가 천장이고, AP는 5 GHz SSID를 아예 방송하지 않는다.**

> **🔍 추론 (INFERRED — 실측 아님).** 옛 수치는 거의 전부 **전송 지연**으로 설명된다:
> 96.1 KiB ÷ 13 Mbit/s = **60.5 ms**인데 기록된 p50이 **58.6 ms**다. 즉 **서버 추론 시간은
> 무시할 만했다.** 07-29의 6.4배를 적용하면 p50 **9–12 ms**가 예측되고, 실제 96 KiB
> 애플리케이션 왕복이 **p50 11.8–21.6 ms**로 그 예측을 감싼다. **독립적인 두 줄기가 일치한다.**

> **⚠️ 측정된 burst 왕복은 "전송 바닥(floor)"이다.** 원격 쪽이 바이트를 **버렸다** —
> 서버 추론이 포함된 end-to-end 스텝 시간을 예측하지 않는다. 전송 하한만 묶어 준다.

**"예산 문제가 해결됐다"로 읽지 마라.** 2.4 GHz · 공유 AP · 주변 AP 41개는 변동이 큰
환경이고, 위 측정은 **유휴 리그**에서 나왔다. 카메라 2대 + 액터 + 조작자가 동시에 붙은
상태는 **측정된 적이 없다.** 13 Mbit/s로 되돌아가면 p99 97.1 ms 세계가 그대로 돌아온다.

> **🔌 변동성을 통째로 없애는 가장 값싼 방법이 지금 물리적으로 가능하다.**
> USB 이더넷 NIC **`enx00e04c3600bd`가 존재하고, 꽂혀만 있지 않다.** 코드 변경 0.

**완화책:** 유선으로 옮기고 같은 100스텝을 재측정한다. 그 전까지
`timeout_s`/`max_response_age_s`를 늘리지 않는다 (증상만 감춘다).
상세: `05_COMMS_GRPC.md` §5.3.

### 📌 classifier sidecar가 이 예산에 미치는 영향 (2026-07-29)

**이 갭이 sidecar(G15)의 페이로드 설계를 직접 결정했다.** 원본 720p JPEG passthrough는
실측 400 KiB/쌍이라 **13 Mbit/s 링크를 통째로 초과**하고 부착 스텝을 **+252 ms** 밀어냈다 —
100 ms 예산에서 즉사다. 그래서 랩톱이 128×128로 resize 후 인코딩해 **쌍당 약 10–15 KiB**로
보낸다. 상세와 대가(JPEG 1세대 추가, MAE 1.29–1.77)는 G15 §대역폭.

**크기를 좋은 링크가 아니라 나쁜 링크(13 Mbit/s) 기준으로 잡은 것이 핵심 판단이다** —
위 표가 보여주듯 링크가 6배 흔들리기 때문이다.

부착이 **팔이 멈춰 있을 때만** 일어난다는 것이 두 번째 보호막이다.
정지 게이트를 끄거나 `interval_steps`를 1로 내리면 **그 보호가 사라진다.**

실기에서는 추정하지 말고 `ActorRunSummary`의 `sidecar_round_trip_ms_mean/max`와
`plain_round_trip_ms_mean/max`를 **따로** 볼 것. 두 계열을 분리해 둔 이유가 이것이다.

---

## G17 — 전역 ESC 리스너가 **아무 창에서나** 에피소드를 끝낸다 🟠 (신규 기재)

이 갭은 새로 생긴 것이 아니라 **여태 갭으로 기재되지 않았을 뿐**이다.

```python
# ur7e_env.py:197-208
from pynput import keyboard
def on_press(key):
    if key == keyboard.Key.esc:
        self.terminate = True
self.listener = keyboard.Listener(on_press=on_press)
self.listener.start()
```

- **pynput 리스너는 전역이다.** 터미널 포커스와 무관하게 X 세션 전체의 ESC를 잡는다.
  브라우저에서, 편집기에서, 다른 GUI에서 누른 ESC가 **로봇 에피소드를 끝낸다.**
- 데드맨 설정(`--deadman topic`)과 **무관하게** 항상 살아 있다. 끌 수 있는 플래그가 없다.
- 그리고 `terminate=True`는 정지가 아니다. 그 에피소드가 끝나고, 다음 `reset()`이
  `go_to_reset()`으로 **팔을 `RESET_JOINTS`로 이동시킨다** (`:505-512`).
  즉 ESC의 실제 효과는 **"지금 에피소드 끝내고 리셋 자세로 팔을 옮겨라"**다.
- 유일한 fail-safe: 디스플레이가 없으면 `except Exception`으로 잡혀 리스너가 안 뜬다
  (`:207-208`). 즉 **헤드리스에서는 없고, GUI 세션에서는 항상 있다.**

**완화책:** 실기 세션 중에는 로봇 랩톱에서 다른 GUI 작업을 하지 않는다.
`00_SETUP_AND_SAFETY.md` §6.1의 "ESC는 정지가 아니다" 항목을 세션 전에 읽는다.

---

## G18 — `ACTION_SCALE`이 learner fingerprint에 **없다** 🟠 (신규)

`run_rlpd_learner_server.py`의 `run_contract`(`:528-`)는 schema hash, threshold,
classifier sha256, action dtype/shape/range, grasp penalty, 의존성 버전을 담는다.
**`ACTION_SCALE`은 없다.**

액션은 정규화된 `[-1,1]^7`이므로 fingerprint 상으로는 동일해 보이지만, **같은 액션이
물리적으로 얼마나 움직이는가는 `ACTION_SCALE`이 정한다.** 그리고 그 값은 2026-07-28에
`[0.01, 0.05, 1.0]` → `[0.0125, 0.0625, 1.0]`으로 **25 % 올랐다** (commit `a5f9890`).

**결과:** 옛 스케일로 녹화한 데모/replay로 새 스케일에서 resume하면 **아무 경고 없이
같은 액션이 25 % 더 멀리 간다.** threshold 불일치는 fail-closed로 거부되는데
이건 통과한다.

**완화책:**
1. 데이터셋과 체크포인트에 **어떤 `ACTION_SCALE`로 수집됐는지 손으로 기록**한다.
2. 스케일을 바꾸면 그 이전 데이터를 재사용하지 않는다.
3. `run_real_hil.py --scale`은 3층을 함께 곱하므로 **CSV 헤더에 남는 배율을 확인**한다.

---

## G19 — `checkpoint_sha256()`가 orbax 디렉터리를 pin 못 한다 🟢 **해결 (2026-07-29)**

> ### ✅ 해결됨 — G15과 **같은 변경**에서
> `checkpoint_sha256()`이 `ur_env/classifier_sidecar.py::directory_sha256()`에 위임한다.
> 파일이면 예전과 **완전히 같은 digest**를 내므로 기존 단일 파일 pin은 그대로 살아 있고,
> 디렉터리면 트리를 재귀 해시한다(정렬된 POSIX relpath + 크기 + 내용, 1 MiB 스트리밍 —
> 이름 변경과 동일 바이트 재분할까지 잡는다).
>
> 폐기 체크포인트를 가리키던 **두 상수도 함께 교체**했다:
>
> | 상수 | 이전 | 현재 |
> | --- | --- | --- |
> | `run_rlpd_receive_server.py::DEFAULT_CHECKPOINT_SHA256` | `e329986b…` | `512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d` |
> | `run_rlpd_learner_server.py::DEFAULT_CLASSIFIER_CHECKPOINT_SHA256` | `e329986b…` | `512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d` (동일 값) |
>
> 새 상수는 `classifier_ckpt/cube_in_cup_all3/checkpoint_150`의 directory digest다
> (재계산 절차가 `run_rlpd_receive_server.py` 주석에 있다).
> 서버는 이제 `DEFAULT_REWARD_MODEL_ID = "cube-in-cup-all3-ckpt150+sidecar-v1"`을 광고하고
> 액터도 같은 값을 pin한다(`run_hil_actor.sh::EXPECTED_REWARD_MODEL_ID`, 이전 값은
> `cube-in-cup-checkpoint-150`). id가 **체크포인트와 입력 계약을 둘 다** 이름에 담고 있어서,
> sidecar 이전 액터와 sidecar를 기대하는 서버가 **핸드셰이크에서 거부된다** —
> 한 세션을 통째로 잘못된 reward로 돌리는 것보다 낫다.
>
> **🔴 이 둘은 반드시 같은 커밋이어야 했다.** 하나만 고치면
> **서버가 정상으로 뜨고 reward가 영구 0**이 된다 — 학습은 잘 도는데 아무것도 안 배우는,
> 이 리그에서 가장 알아채기 어려운 실패다.
>
> **📌 learner fingerprint가 한 번 깨진다 — 의도된 것이다.** 체크포인트 SHA ·
> `reward_model_id` · 새 `run_contract` 필드가 전부 fingerprint에 들어가므로 **구 체크포인트
> resume은 fail-closed로 거부된다.** 잃는 것은 없다 — 구 lineage는 recall 0%짜리 폐기
> 체크포인트 위에 세워져 있었다. `DEFAULT_REWARD_THRESHOLD`는 **0.2 그대로**다.

아래는 해결 전 기록이다.

```python
# rlpd_receive_server.py:148-151 (해결 전)
def checkpoint_sha256(path: str) -> str:
    ...
    if not os.path.isfile(checkpoint):
        ...
```

`os.path.isfile`을 요구하므로 **디렉터리 형태의 orbax 체크포인트는 pin할 수 없다.**
그 결과 코드 기본값이 아직 단일 파일 형태의 **07-24 체크포인트**를 가리키고 있다:

```python
# scripts/run_rlpd_receive_server.py:34-36
DEFAULT_CHECKPOINT_SHA256 = (
    "e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997"
)
```

**그 체크포인트는 폐기 대상이다** — 새 도메인(07-24 이후 수집분)에서 recall 0.0 %로
측정됐다. 후속 07-27 체크포인트(`cube_in_cup_all3`, orbax 디렉터리)는
0720 test split(n=166)에서 100.0 % @0.5(FPR 0.0 %), 0720 held-out pool(test 166 + val 100
= 266프레임)에서 86.8 % @0.5 / 83.1 % @0.85로 훨씬 낫다.

> ⚠️ 두 수치가 다른 것은 **분할이 달라서**이고 둘 다 맞다 — 차이는 전부 취약 take인 `take_21`이
> val에 있기 때문이다. 보수적으로는 **266프레임 쪽**을 쓴다. 어느 쪽도 leave-one-take-out CV가
> 아니다(진짜 CV는 `fold_take_01/02/03` 별도 체크포인트, @0.5에서 89.5 / 89.9 / 86.5).
> 그리고 **두 수치 모두 크롭 없는 입력에서 측정됐다** — G15의 크롭 활성 경로에는
> 적용되지 않는다.

**즉 G19는 G15 해결의 선결 조건이었다.** ~~재학습 결과가 orbax 디렉터리로 나오면
지금 코드로는 그것을 pin할 수 없다.~~ → **둘 다 해결됐다**(위 박스).

> **🔧 위 문단의 "크롭 없는 입력에서 측정됐다 → G15의 크롭 활성 경로에는 적용되지 않는다"는
> 이제 반대로 읽어야 한다.** 분류기가 sidecar 덕분에 **계속 무크롭을 먹으므로**,
> 저 수치들은 gRPC RL 경로에 **그대로 적용된다.** 이것이 재학습 대신 분리를 택한 실익이다.

~~**완화책 (해결 전):** 서버를 띄울 때 `--checkpoint`와 `--expected-checkpoint-sha256`을
**항상 명시**하고, 기본값에 기대지 않는다.~~ 이제 기본값이 정본을 가리키므로 명시가
필수는 아니다. 그래도 **어느 체크포인트가 로드됐는지 서버 기동 로그에서 확인하는 습관은
유지할 것** — 이 실패는 조용하다.

---

## G20 — canonical demo가 없다 🟢 **해결 (2026-07-29)**

> ### 🔧 상태 갱신 — 변환 경로와 사람 승인 artifact가 모두 생겼다
> `40b99f8`("convert recorder takes to learner demos")이 recorder take를 learner가 직접 받는
> canonical pickle로 바꾸는 경로를 넣었다:
> `scripts/convert_recorded_takes_to_demo.py`, `ur_env/learner/recorded_demo.py`,
> `tests/test_recorded_demo_converter.py`, 그리고 절차 문서
> [`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](../../serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md).
> 출력 pickle은 learner의 `--demo-path`에 그대로 넘긴다.
>
> 사용자가 `take_23_20260720_210316`을 제외한 2026-07-20의 23개 take를 success로
> 승인했다. `--outcome success`로 만든 영구 artifact는 2,037 transitions이며 laptop3와
> Kanu strict loader가 모두 통과했고 양쪽 SHA256은
> `f97185582401ce7570d44fddc33d1bd64b215d7e32d6384d5fe13e1b405032fa`로 같다.
> Kanu 경로는
> `/home/junhyeong/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl`이다.
>
> learner의 시작 게이트는 online replay ≥ `training_starts` **그리고** offline demo ≥ 1을
> 요구하므로(`ur_env/learner/runtime.py:228`의 `LearnerNotReadyError` 메시지가 두 수치를 같이 찍는다),
> **성공 라벨 artifact가 없으면 learner는 학습을 시작하지 않는다.** 현재 artifact는 이
> 조건을 만족한다.

아래 actor 쪽 서술은 **여전히 유효하다** — 변환 경로는 learner 쪽이고, actor 쪽 배선은
그대로 없다.

HIL-SERL은 오프라인 데모로 replay를 시드하는 것을 전제로 한다. 이 선결 조건은 위
artifact로 충족됐다. 다만 actor 자체의 주기적 pickle 기록 배선은 여전히 꺼져 있다:

- `cube_in_cup`의 **`buffer_period = 0`**이다 (`cube_in_cup.py:265`).
  actor의 주기적 pickle 덤프는 `if step > 0 and config.buffer_period > 0`으로 게이트돼
  있으므로 (`ur_env/rlpd_actor.py:258-270`, 기록 함수는 `remote_actor.py:202-207`),
  **`--checkpoint-path`를 줘도 아무 pickle이 쓰이지 않는다.**
- 그리고 그 경로는 **쓰기용**이다. 기존 데모를 **읽어서 replay에 시드하는 경로는
  actor entrypoint에 없다** — 그건 learner 쪽 일이다
  (`serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md`).

**그래서 `--mock-policy-noise`로 만든 데이터를 데모로 쓰면 안 된다.**
그 모드의 전이에는 `meta.policy_actions_synthetic = true`가 박히고
(`run_remote_rlpd_actor.py:341-346`), actor가 종료 시 그 사실을 경고한다.
07-27 Kanu 왕복 100스텝도 **synthetic**이다 — 상대가 zero-action 서버였다.

**남은 주의:** 2,037개 중 action norm-clamp가 516개(25.33%)다. 현재 artifact의 출처와
사람 라벨은 확정됐지만, 이 포화 비율은 production run 기록에 남기고 정책 품질과 별도로 본다.

---

## 부록 — 다른 파일에서 발견된 낡은 서술

**이 표의 항목은 전부 `docs/testing/` 밖이다. 고치는 것은 각 파일 소유자의 몫이고,
여기서는 "이걸 근거로 쓰면 틀린다"는 경고로만 둔다.**
보통은 코드가 정답이지만, `cube_in_cup.py:126`과 `ros_backend.py:208`처럼
**코드 주석 쪽이 낡은** 경우도 있다.

| 파일 | 낡은 서술 | 실제 (2026-07-29 확인) |
|---|---|---|
| `docs/ros2/GELLO_UR7E_EEF_MODE.md:380` | P9 "yaml 기본 `v_max=0.08`" | `config/ur7e_gello_eef.yaml:238` = **0.16** (같은 문서 `:7`도 0.16이라 자기모순). **여전히 낡음** |
| `serl_ur_infra/RL_RECEIVE_SERVER.md:62-70` | state 순서 = "TCP pose 6, TCP velocity 6, TCP force 3, TCP torque 3, gripper 1" | 현재 `observation_schema.py` = **알파벳순, gripper가 index 0**. **여전히 낡음 — 이 문서를 근거로 인덱스를 쓰면 틀린다** |
| `serl_ur_infra/README.md:59` 현황표 | "워크스페이스 박스 (`ABS_POSE_LIMIT`) … ❌ config만 존재, 미작동" | **낡음.** 구현·배선됐다 → G1. (단 `DefaultUR7eEnvConfig`에서는 실제로 비활성이므로 "반쯤 맞다"는 점이 혼란의 근원이다) |
| ~~`serl_ur_infra/README.md` TODO "v_max vs ACTION_SCALE 정합"~~ | (07-27 판이 낡았다고 지적한 항목) | **해소됨.** README `:77`이 이제 `[x] 2026-07-28 해결 — 3층을 균일 1.25배로 맞춰 헤드룸 1.20x 유지`로 갱신돼 있다. 이 줄은 더 이상 불일치가 아니다 |
| `serl_ur_infra/ur_env/envs/ros_backend.py:208` | `VERIFY(hw)` — "퍼블리셔가 best-effort면 구독이 조용히 안 뜬다, 첫 브링업에서 확인할 것" | **확인 완료.** RealSense는 RELIABLE/TRANSIENT_LOCAL → 호환. 주석만 안 지워졌다 → `06_SENSORS.md` §3 |
| `ros2_ur_ws/launch_cameras.sh` 머리말 + 기동 배너 | "cam2 = CLOSE-UP (workspace)" | **cam2는 손목 카메라다** → `06_SENSORS.md` §1.1. 시리얼 자동 해석은 `43ba314`/`fb48100`에서 들어왔지만 **이 문구는 그대로 남았다** |
| `serl_ur_infra/ur_experiments/cube_in_cup.py:126` | true-TCP 전환 시 z 한계를 `0.0045 .. 0.376`이라고 적음 | **산술이 낡았다.** z 바닥이 `0.1785`→`0.185`로 올라갈 때 안 따라왔다. `0.185 − 0.174 = 0.011`이 맞다 → G1 |
| `ros2_ur_ws/src/gello_policy/config/act_deploy.yaml:35` (및 diffusion/FM 형제) | 키 이름은 `start_pose`이고 값은 `[3.106, -1.817, 1.653, -1.618, -1.628, -3.195]`, 주석은 그냥 "the dataset's start pose" | 그건 **banana-in-pot** 자세다. cube_in_cup과 어깨에서 0.29 rad 차이 = TCP가 13 cm 높고 10 cm 뒤. **HIL에 복사하지 말 것.** cube_in_cup 값은 `[3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331]` (`cube_in_cup.py:90-92`). (07-27 판은 이 키를 `RESET_JOINTS`라고 불렀는데 그건 이 리포 쪽 이름이다) |
| `docs/ros2/GELLO_UR7E_{ACT,DIFFUSION,FM}_DEPLOY.md`, `ros2_ur_ws/src/gello_{policy,recorder}/README.md` | 카메라 시리얼이 리터럴로 박혀 있음 | 시리얼은 이제 `_resolve_camera_serials.sh`가 live 버스에서 해석한다. 리터럴은 **둘 중 어느 쌍이든 동전 던지기** → `06_SENSORS.md` §1.1 |
| `serl_ur_infra/README.md` 머리말 | "⚠️ UNTESTED SKELETON" | **여전히 맞다** (실기 RL **정책** 경로 미검증). 유지. 단 zero-policy + 사람 개입 경로는 07-28에 실기에서 돌았다 |
