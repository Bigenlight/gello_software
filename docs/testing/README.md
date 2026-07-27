# HIL-SERL 실기 투입 — 통신·하드웨어 검증 런북 (인덱스)

이 디렉터리는 **HIL-SERL을 실제 UR7e에 올리기 전에 통신·하드웨어 경로를 사람이 직접
확인하는 절차**를 담는다. 학습 모델(learner/policy)은 이 문서의 범위가 **아니다**.

- 통합 checkout: `/home/laptop3/gello_software` (브랜치 `feat/gello-ur7e-humble-22.04`)
- 이 문서들의 모든 명령은 위 경로 기준이다. 아래처럼 셸 변수를 잡아두고 복사해서 쓰면 된다.

```bash
export WT=/home/laptop3/gello_software
```

> ### 이 문서의 규칙
> - **검증됨(PASS)** = 실제 하드웨어에서 사람이 확인한 것. 날짜와 실측치를 남긴다.
> - **미검증(TODO)** = 절차만 적혀 있고 아직 실행되지 않은 것. 절대 PASS로 승격하지 말 것.
> - 중요한 주장에는 `파일:줄` 근거를 단다. 근거 없는 문장은 추측이며 그렇게 표시한다.

---

## 0. 지금 당장 알아야 할 것 3가지

1. **`clip_safety_box`는 구현되어 있지 않다.** 워크스페이스 박스는 객체로 만들어지기만 하고
   (`serl_ur_infra/ur_env/envs/ur7e_env.py:118`, `:123`) 코드 어디에서도 다시 참조되지 않는다.
   즉 RL 정책/개입이 명령하는 TCP 위치에 **소프트웨어 경계가 없다.** → `08_OPEN_GAPS.md`
2. **19-D `state` 레이아웃 계약은 v2로 확정·통합됐다.**
   `serl_ur_infra/ur_env/observation_schema.py`의 그리퍼 스칼라 인덱스는
   **-1이 아니라 0**이 되었다. 값을 문서에 박아 넣지 말고 항상 라이브로 출력해서 확인할 것.
   → `05_COMMS_GRPC.md` §4, `08_OPEN_GAPS.md`
3. **그리퍼 개입 데드코드는 수정·통합됐지만 실기 검증은 아직이다.**
   `/gello/joint_states`의 `position` 길이는 여전히 6이지만
   (`gello_publisher_node.py:190`), `URRosBackend`가 트리거 토픽을 **추가로 구독**해서
   7-요소로 합쳐 주게 바뀌었다 (`ros_backend.py:64-124`, `:162-172`).
   → `04_HIL_INTERVENTION.md` §6, `08_OPEN_GAPS.md` G3

> ### 통합 상태
> 위 변경은 hardware commit `6a0b127`과 learner/hardware merge `248255f`에 포함됐다.
> 작업 전 `git status`를 확인하고 기존 사용자 변경을 지우지 않는 원칙은 그대로다.
> 파일 줄 번호는 이후 commit에 따라 어긋날 수 있으므로 내용은 `rg`로 재확인한다.

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
| 7 | 오프라인 단위 테스트 `ur_gello_bringup` | **PASS** | 436 tests collected / passed (§4에 재현 명령) |
| 8 | 타이밍 baseline | **측정됨(참고치)** | `step()` p99 = 94 µs(정지) / 538 µs(이동), worst-case tick 0.644 ms. **리포에 산출물이 커밋되어 있지 않다** — 재측정 가능한 스크립트 없음 |
| 9 | HIL 개입 루프 (mock + RViz) | **미검증(이 브랜치에서)** | 절차는 `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`에 존재. → `04_HIL_INTERVENTION.md` |
| 9b | HIL 개입 루프 (**실기**, DRY_RUN) | **코드 통합, 실기 미검증** | `serl_ur_infra/tests/run_real_hil.py`. 기본 `DRY_RUN=True`, `--arm` 없이는 명령 미발행 |
| 9c | 리더 트리거 → 개입 그리퍼 배선 | **코드 통합, 하드웨어 미검증** | `ros_backend.py`, `wrappers.py`, `tests/test_gello_gripper_wiring.py`; hardware commit `6a0b127` |
| 10 | gRPC actor 루프백 스모크 | **미검증(이 브랜치에서)** | 절차 존재. → `05_COMMS_GRPC.md` |
| 11 | RealSense 2대 동시 스트림 | **미검증(이 브랜치에서)** | → `06_SENSORS.md` |
| 12 | 장애 주입 매트릭스 | **미검증 (전 항목)** | → `07_FAILURE_INJECTION.md` |
| 13 | RL 정책 경로로 실기 팔 구동 | **금지 / 미검증** | `DRY_RUN=True`가 기본(`config.py:131`). `08_OPEN_GAPS.md`의 갭이 닫히기 전에는 해제 금지 |

> **8번 주의:** p99 수치는 이전 세션의 측정 보고이며 리포에 로그·스크립트가 남아 있지 않다.
> 근거를 요구받으면 "재측정 필요"라고 답할 것. 사실로 인용하려면 재측정 후 여기에 산출물 경로를 남긴다.

---

## 2. 통합 checkout

| 역할 | 경로 | 브랜치 |
|---|---|---|
| **통신·하드웨어·학습 통합** | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` |

임시 learner/hardware worktree의 코드는 이 branch에 통합됐다. 세션 시작마다 `git status`와
`git submodule status`를 확인하고, canonical checkout에 이미 있던 사용자 변경이나 submodule
내 산출물을 `reset`, `clean`, `stash`로 지우지 않는다. `third_party/hil-serl`은 pinned submodule이며
직접 수정하지 않는다.

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
 A2  ur_gello_bringup 단위테스트 436개  -> 00 §4  [PASS]
 A3  serl_ur_infra 단위테스트           -> 00 §4
 A4  gRPC 루프백 스모크 (mock 서버)     -> 05 §2
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
 F1  SSH 터널 + 스키마 핸드셰이크       -> 05 §5
 F2  레이턴시 예산 실측                  -> 05 §6
        ↓
[G] RL 정책 경로 실기 투입  <-- 08_OPEN_GAPS.md의 G1~G4가 닫히기 전에는 금지
```

**현재 위치: [D] 완료, [C]/[E]/[F] 미착수.**
