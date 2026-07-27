# HIL-SERL 실기 투입 — 통신·하드웨어 검증 런북 (인덱스)

이 디렉터리는 **HIL-SERL을 실제 UR7e에 올리기 전에 통신·하드웨어 경로를 사람이 직접
확인하는 절차**를 담는다. 학습 모델(learner/policy)은 이 문서의 범위가 **아니다**.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

> ### 🔧 정정 (2026-07-27) — `WT`가 바뀌었다
> 이 문서들은 원래 `WT=/home/laptop3/gello_software`(브랜치
> `feat/gello-ur7e-humble-22.04`)를 가리켰다. 그러나 오늘 들어간 HIL 신규 코드
> (`serl_ur_infra/ur_experiments/`, `clip_safety_box`, branch-cut `go_to_reset`,
> `ros2_ur_ws/run_hil_actor.sh`)는 **워크트리
> `/home/laptop3/gello_worktrees/hil-hardware-comms`(브랜치 `test/hil-hardware-comms`)에만
> 있다.** 확인:
>
> ```bash
> git -C /home/laptop3/gello_software worktree list
> ls /home/laptop3/gello_software/serl_ur_infra/ur_experiments   # -> No such file
> ```
>
> 텔레옵·그리퍼·GELLO·카메라(`ros2_ur_ws/**`)는 양쪽이 같으므로 00~03의 하드웨어 절차는
> 어느 checkout에서 돌려도 된다. **04~09의 `serl_ur_infra` 절차는 워크트리에서만 돈다.**
> 워크트리에도 자체 `ros2_ur_ws/install/`이 빌드되어 있다(확인함).

> ### 이 문서의 규칙
> - **검증됨(PASS)** = 실제 하드웨어에서 사람이 확인한 것. 날짜와 실측치를 남긴다.
> - **미검증(TODO)** = 절차만 적혀 있고 아직 실행되지 않은 것. 절대 PASS로 승격하지 말 것.
> - 중요한 주장에는 `파일:줄` 근거를 단다. 근거 없는 문장은 추측이며 그렇게 표시한다.
> - **줄 번호는 빠르게 낡는다.** 2026-07-27 하루에만 `ur7e_env.py`가 200줄 넘게 밀렸다.
>   `파일:줄`이 안 맞으면 틀린 게 아니라 밀린 것이다 — `rg`로 내용을 다시 찾는다.

---

## 0. 지금 당장 알아야 할 것 (2026-07-27 실기 세션 반영)

1. **🛑 시스템 `python3`로 gRPC를 쓰면 조용히 영구 정지한다.**
   `python3-grpcio 1.30.2`(apt)가 이 머신에서 손상돼 있다. 채널을 하나만 만들어도
   **에러도 로그도 없이 단일 스레드가 CPU 100%로 무한 회전**한다. 반드시
   `/home/laptop3/venvs/gello-hil-actor/bin/python`(grpcio 1.74.0)을 쓴다.
   → `00_SETUP_AND_SAFETY.md` §3.4, `05_COMMS_GRPC.md` §1

2. **✅ `clip_safety_box`는 구현됐다 (정정).**
   이전 판은 "박스는 객체로만 만들어지고 아무도 참조하지 않는다"였다. 지금은
   `UR7eEnv.clip_safety_box()` / `_clip_command_pose()`가 있고 후자가
   `PolicyDeltaController(clip_pose=...)`로 배선되어 **명령 포즈**에 적용된다
   (`ur7e_env.py:188`, `:334`, `:357`; `policy_delta_controller.py:140-141`).
   실측 박스는 `ur_experiments/cube_in_cup.py`에 있다. **단위 테스트만 통과했고
   실기 미검증이다.** → `08_OPEN_GAPS.md` G1

3. **19-D `state` 레이아웃은 알파벳순이 정본이다.**
   `[0] gripper_pose | [1:4) tcp_force | [4:10) tcp_pose | [10:13) tcp_torque | [13:19) tcp_vel`.
   `state[..., -1]`은 그리퍼가 **아니라 `tcp_angular_velocity_z`**다.
   `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`만 쓴다.
   → `05_COMMS_GRPC.md` §4, `06_SENSORS.md` §5

4. **cam2는 손목(wrist) 카메라다 (정정).**
   "책상 위 근접(close-up) 고정 카메라"는 **틀린 서술**이었다. cam2는 그리퍼에 강체로
   물려 있어 손가락이 항상 같은 픽셀에 있고 배경이 팔 자세를 따라 움직인다
   (`ur_experiments/cube_in_cup.py`의 `IMAGE_CROP` 주석). → `06_SENSORS.md` §1.1

5. **`PYTHONPATH`는 덮어쓰지 말고 이어붙인다.** 덮어쓰면 ROS 오버레이가 날아가
   `ModuleNotFoundError: ur_gello_bringup`이 난다. 반대로 **`serl_ur_infra`의 pytest에는
   ROS `PYTHONPATH`가 붙어 있으면 안 된다** — 수집이 통째로 죽고 `1 skipped`만 뜬다.
   → `00_SETUP_AND_SAFETY.md` §3.4, §4

> ### 통합 상태
> 하드웨어 계약 변경은 commit `6a0b127` / merge `248255f`에, 오늘의 HIL 변경은
> `50f65de`~`ee3240e`에 들어 있다. 작업 전 `git status`를 확인하고 기존 사용자 변경을
> 지우지 않는 원칙은 그대로다.

---

## 1. 현재 검증 상태표

| # | 항목 | 상태 | 근거 / 실측치 |
|---|---|---|---|
| 1 | UR7e 도달성·상태 | **PASS** | `Robotmode: RUNNING`, `Safetystatus: NORMAL`, remote control `true`, IP `192.168.10.11` |
| 2 | 그리퍼 경로 (Modbus over tool-comm :54321) | **PASS** | 열기 `position_percent`=0.0118, 빈손 완전닫힘=0.8980(=229/255), 액션 `position: 0.085`(=열림) → `reached_goal: true`, 피드백 5.000 Hz (std 1.3 ms) |
| 3 | 그리퍼 **방향** 육안 확인 | **PASS** | crush 게이트 해소. `_send_gripper_command`의 `VERIFY(hw)` 주석(`ur7e_env.py:437`) 조건 충족 |
| 4 | GELLO 리더 (Dynamixel) | **PASS** | baud 57600, ID1~6 = model 1200, ID7 = model 1190 전부 응답 |
| 5 | GELLO 발행 안정성 | **PASS** | 30.004 Hz (std 0.15 ms), 30초 901샘플, 드롭 0, `comm failed` 0회, 트리거 0.000~1.000 전 구간 |
| 6 | EEF 텔레옵 (실기) | **PASS** (사용자 직접 검증) | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef` + `./run_eef_gui.sh` |
| 7 | 오프라인 단위 테스트 `ur_gello_bringup` | **PASS** | **436 passed in 10.14s** (2026-07-27 워크트리에서 재실행). §4에 재현 명령 |
| 7b | 오프라인 단위 테스트 `serl_ur_infra` | **PASS** | **332 passed, 11 skipped, 1 xfailed in 3.46s** (2026-07-27 재실행). **venv python + ROS PYTHONPATH 제거**가 조건 → `00` §4.2 |
| 8 | 타이밍 baseline | **재측정됨** | `test_ur_kin.py`(k)가 매 실행마다 찍는다: **worst-case tick = 1.314 ms** (예산 4.0 ms @250 Hz). 이전 판의 "0.644 ms / 산출물 없음"은 **폐기** — §7 명령이 곧 산출물이다 |
| 9 | HIL 개입 루프 (mock + RViz) | **미검증(이 브랜치에서)** | 절차는 `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`에 존재. → `04_HIL_INTERVENTION.md` |
| 9b | HIL 개입 루프 (**실기**, DRY_RUN) | **코드 통합, 실기 미검증** | `serl_ur_infra/tests/run_real_hil.py`. 기본 `DRY_RUN=True`, `--arm` 없이는 명령 미발행 |
| 9c | 리더 트리거 → 개입 그리퍼 배선 | **코드 통합·커밋됨, 하드웨어 미검증** | `ros_backend.py:81-140`, `wrappers.py`, `tests/test_gello_gripper_wiring.py`(23 passed); commit `6a0b127` |
| 10 | gRPC actor 루프백 스모크 | **PASS (오프라인)** | `test_actor_grpc_transport/identity_pinning/smoke/rlpd_receive_smoke` = **35 passed** (venv python). 같은 4개 파일이 **시스템 python3에서는 무한 hang** → §0-1 |
| 10b | **Kanu 왕복 (Stage A, fake-env)** | **PASS (2026-07-27)** | 100스텝 acceptance 통과. 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`. 절차·수치 정본은 [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) |
| 10c | 레이턴시 실측 (Kanu 왕복) | **측정됨 — 예산 소진 상태** | RTT p50 **58.6** / p95 **75.8** / p99 **97.1 ms**. 관측 96.1 KiB → 10 Hz에 **7.9 Mbit/s**. **병목은 WiFi 대역폭**이고 p99가 10 Hz 예산(100 ms)을 거의 다 쓴다 → `05` §5.3 |
| 11 | RealSense 2대 동시 스트림 | **미검증(이 브랜치에서)** | → `06_SENSORS.md` |
| 11b | RealSense QoS 호환성 | **PASS (해소됨)** | 퍼블리셔가 RELIABLE/TRANSIENT_LOCAL → 백엔드의 기본 reliable 구독과 호환. 이전의 "best-effort면 콜백이 안 뜬다" 우려는 **이 리그에서는 해소**. 단 TRANSIENT_LOCAL 부작용 있음 → `06` §3 |
| 11c | cam1/cam2 역할 | **정정됨** | cam1 = 삼각대 SCENE, **cam2 = 손목(wrist) 카메라**. "cam2가 작업 영역을 안 본다"는 판정은 **폐기** → `06` §1.1 |
| 12 | `clip_safety_box` (워크스페이스 박스) | **구현·단위검증, 실기 미검증** | `tests/test_clip_safety_box.py` **26 passed**. 실측 박스는 `ur_experiments/cube_in_cup.py` → `08` G1 |
| 12b | `go_to_reset` branch-cut | **수정·단위검증, 실기 미검증** | `tests/test_reset_branch_cut.py` **10 passed**. 실측 케이스: wrist_3 +3.1795 → 목표 −3.1331은 물리적으로 0.029 rad인데 예전 코드는 6.31 rad로 계산·명령했다 → `08` G13 |
| 13 | 장애 주입 매트릭스 | **미검증 (전 항목)** | → `07_FAILURE_INJECTION.md` |
| 14 | RL 정책 경로로 실기 팔 구동 | **금지 / 미검증** | `DRY_RUN=True`가 기본(`config.py`, `cube_in_cup.py:DRY_RUN`). `08_OPEN_GAPS.md`의 갭이 닫히기 전에는 해제 금지 |

---

## 2. checkout 지도 (2026-07-27 갱신)

| 역할 | 경로 | 브랜치 | 무엇이 여기에만 있나 |
|---|---|---|---|
| 통합 checkout | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` | (오늘 이전의 통합 상태) |
| **HIL 하드웨어·통신 작업본 ⭐** | `/home/laptop3/gello_worktrees/hil-hardware-comms` | `test/hil-hardware-comms` | `ur_experiments/`, `clip_safety_box`, branch-cut reset, `run_hil_actor.sh`, `run_hil_preposition.sh`, 09 런북 |
| 보상 오버라이드 작업본 | `/home/laptop3/gello_worktrees/human-reward-override` | `feat/human-reward-override` | (범위 밖) |

```bash
git -C /home/laptop3/gello_software worktree list   # 세 개가 다 보인다
```

세션 시작마다 `git status`와 `git submodule status`를 확인하고, checkout에 이미 있던
사용자 변경이나 submodule 내 산출물을 `reset`, `clean`, `stash`로 지우지 않는다.
`third_party/hil-serl`은 pinned submodule이며 직접 수정하지 않는다.

> ⚠️ **여러 에이전트/사람이 같은 워크트리를 동시에 고치고 있다.** 파일이 몇 분 사이에
> 바뀔 수 있다. 숫자를 인용하기 전에 `git log --oneline -5`로 최신 커밋을 확인할 것 —
> 실제로 오늘 `ABS_POSE_LIMIT_LOW[2]`가 `0.1785` → `0.185`로, `RESET_MAX_DIST_RAD`가
> `0.5` → `0.9`로 바뀌었다 (commit `ee3240e`).

---

## 3. 문서 목록

| 파일 | 내용 |
|---|---|
| [`00_SETUP_AND_SAFETY.md`](00_SETUP_AND_SAFETY.md) | 워크트리 셋업·빌드·환경변수 함정, 세션 전 체크리스트, **비상 정지 우선순위와 "정지가 아닌 것들"** |
| [`01_GRIPPER.md`](01_GRIPPER.md) | Robotiq 2F-85 단독/통합 검증 (**완료**) |
| [`02_GELLO_LEADER.md`](02_GELLO_LEADER.md) | GELLO 리더 검증 + Dynamixel 진단 스캔 (**완료**) |
| [`03_EEF_MODE.md`](03_EEF_MODE.md) | EEF 단계 상승 P6 → P7 → P8 → P9a → P9b |
| [`04_HIL_INTERVENTION.md`](04_HIL_INTERVENTION.md) | 데드맨, 앵커/gain 래치, 좌표계 3×3 검증, 개입 메타데이터 계약 |
| [`05_COMMS_GRPC.md`](05_COMMS_GRPC.md) | venv 격리, 루프백 스모크, schema fail-fast, 레이턴시 예산, Kanu 터널 |
| [`06_SENSORS.md`](06_SENSORS.md) | RealSense 2대, QoS 함정, 19-D state 계약 |
| [`07_FAILURE_INJECTION.md`](07_FAILURE_INJECTION.md) | 장애 주입 매트릭스 (유발·기대·확인·PASS·복구) |
| [`08_OPEN_GAPS.md`](08_OPEN_GAPS.md) | 미해결 안전 갭과 임시 완화책 |
| [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) | **HIL actor 기동 런북** (Stage A fake-env / Stage B 실센서 DRY_RUN, `run_hil_actor.sh`, Kanu 왕복 결과) |

관련 기존 문서(이 디렉터리 밖, 읽기 전용 참조):

- `docs/ros2/GELLO_UR7E_EEF_MODE.md` — EEF 모드 정본. `03_EEF_MODE.md`는 여기서 파생.
- `docs/ros2/GELLO_UR7E_GRIPPER.md` — 그리퍼 경로 정본.
- `docs/ros2/GELLO_UR7E_SETUP_CLI.md` — 실기 텔레옵 세션 체크리스트 정본.
- `serl_ur_infra/RVIZ_HIL_TEST_CLI.md` — mock HIL 4터미널 절차 정본.
- `serl_ur_infra/REMOTE_ACTOR_GRPC.md`, `serl_ur_infra/RL_RECEIVE_SERVER.md` — gRPC/수신서버 정본.
- `serl_ur_infra/README.md` — env 설계 요점 + `PolicyDeltaController` 현황표.

---

## 4. 전체 실행 순서 (권장)

각 단계는 **앞 단계가 PASS일 때만** 진행한다. 굵은 글씨는 로봇이 실제로 움직이는 단계다.

```
[A] 오프라인 (로봇 불필요, 위험 0)
 A1  워크트리 빌드                     -> 00 §2
 A2  ur_gello_bringup 단위테스트 436개  -> 00 §4.1  [PASS 2026-07-27]
 A3  serl_ur_infra 단위테스트 332개     -> 00 §4.2  [PASS 2026-07-27]
 A4  gRPC 루프백 스모크 (mock 서버)     -> 05 §2    [PASS 오프라인]
        ↓
[B] 하드웨어 단독 (팔 미동작)
 B1  로봇 도달성 / dashboard 상태       -> 00 §5   [PASS]
 B2  그리퍼 단독                        -> 01      [PASS]
 B3  GELLO 리더 단독                    -> 02      [PASS]
 B4  RealSense 2대                      -> 06 §1
        ↓
[C] mock 하드웨어 + RViz (실기 위험 0)
 C1  mock RViz HIL 개입 루프            -> 04 §3
 C2  좌표계 3x3 검증                    -> 04 §5
        ↓
[D] 실기 EEF 텔레옵 (팔 움직임)  ** 사람이 E-STOP 위에 손 **
 D1  P6  pos_scale:=0.0                 -> 03 §2   [사용자 검증 완료]
 D2  P7  병진 위주                      -> 03 §3   [사용자 검증 완료]
 D3  P8  회전 위주                      -> 03 §4   [사용자 검증 완료]
 D4  P9a / P9b 6-DoF + 재클러치         -> 03 §5   [사용자 검증 완료]
        ↓
[E] 장애 주입 (팔 움직임 포함)
 E1  통신/프로세스 계열 (E1~E5)         -> 07
 E2  로봇 안전 계열 (E6~E11)            -> 07
        ↓
[F] 원격 통신 (Kanu)
 F1  SSH 터널 + 스키마 핸드셰이크       -> 05 §6, 09 §2   [PASS 2026-07-27]
 F2  레이턴시 예산 실측                  -> 05 §5.3       [측정 완료 — 예산 소진]
 F3  Stage A actor (fake-env) 왕복       -> 09            [PASS 2026-07-27]
 F4  Stage B actor (실센서, DRY_RUN)     -> 09            [미검증]
        ↓
[G] RL 정책 경로 실기 투입  <-- 08_OPEN_GAPS.md의 갭이 닫히기 전에는 금지
```

**현재 위치: [A]·[B]·[D] 완료, [F1]~[F3] 완료. [C]/[E]/[F4] 미착수.**

> [F]가 [C]/[E]보다 먼저 끝난 것은 순서를 어긴 게 아니라, [F]가 **로봇을 전혀 움직이지 않는
> fake-env 경로**이기 때문이다. 팔이 움직이는 단계는 여전히 [D]까지만 검증돼 있다.
