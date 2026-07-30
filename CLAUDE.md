# CLAUDE.md

문서 색인이다. 상태 보고서가 아니다 — 숫자와 근거는 링크된 문서에 있다.

## 이 리포에서 진행 중인 작업

**HIL-SERL(사람 개입 온라인 RL)을 실기 UR7e에서 돌린다.**

```
laptop3                                        kanu (GPU 서버)
  GELLO 리더팔 (USB, EEF teleop)                 정책 추론 (SAC)
  RealSense ×2 (USB)          ──gRPC :50053──►   온라인 학습 (RLPD)
  UR7e (이더넷, ROS2 Humble)  ◄──── 액션 ─────    reward classifier
```

laptop3의 GPU가 약해 **정책·학습·reward classifier를 전부 kanu에서** 돌리고 gRPC로 실시간
통신한다. reward 권위는 서버에 있다. 명목 제어 루프는 10 Hz다. 2026-07-30 startup에서 정상
reply가 832.3 ms에 도착해 옛 0.6/0.8 s 경계를 넘었으므로 현재 RPC timeout/response-age는
bounded `1.5/2.0 s`로 완화했다.
분류기는 정책 관측이 아니라 **자기 전용 무크롭 이미지(sidecar)를 약 2 Hz로** 따로 받는다.

**branch `feat/gello-ur7e-humble-22.04`**. 2026-07-29 머지 `3f199d4`가 로봇/하드웨어
작업을 이 브랜치로 가져왔다. **워크트리 분리는 끝났다** — 로봇 코드와 learner 코드가 **다른
checkout에 있다**고 적힌 문서는 전부 낡은 것이다(아직 여러 개 남아 있다). 작업은
`/home/laptop3/gello_software` 한 곳에서만 한다.

## 지금 상태 한 줄

**2026-07-30 현재 HIL-SERL 원형은 실물 UR7e에서 구동됐다.** Kanu policy action으로 팔이
움직이고, GUI `ENGAGE` 중에는 GELLO 개입, 해제 뒤에는 policy 제어로 돌아가며, online
transition과 learner update까지 관측했다. 현재 정상 운용은 아래 세 terminal뿐이다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh    # Terminal 1: Kanu learner 재사용/기동 + tunnel
./run_hil_hardware.sh  # Terminal 2: UR7e + Robotiq + passive GELLO
./run_hil_session.sh   # Terminal 3: cameras + GUI + actor
```

**현재 episode 운영 계약:** 성공 판정은 기본 `MANUAL`이다. Kanu classifier는 MANUAL에서도
계속 평가·표시·replay 기록되지만 terminal 권한은 GUI `MARK SUCCESS`에 있다. `AUTO`로
전환하면 strict `p(success) > 0.5`가 성공 권한을 가진다. 성공 또는 episode limit 뒤에는
`WAIT_HOME_APPROVAL`에서 로봇을 hold하고, GUI `APPROVE HOME` 뒤 HOME, 장면을 사람이
재배치한 뒤 `START / NEXT ITERATION`을 눌러 다음 policy episode를 연다. classifier headline은
16 pt의 compact GUI로 표시된다.

**시작 조작 간소화:** `run_hil_preposition.sh`의 예전 대문자 `GO` 입력은 기본 경로에서
없어졌다. RESET 0.10 rad 밖이면 체크리스트를 출력한 뒤 기본값은 곧바로 JTC 이동을 시작한다
(`PREPOSITION_DELAY_S=N`으로 취소 가능한 카운트다운, `PREPOSITION_CONFIRM=1`로 옛 GO
프롬프트를 opt-in할 수 있다). `run_hil_session.sh`의 별도 Enter 프롬프트도 없어졌고 GUI가
ENGAGED가 될 때까지 폴링한다. 이것은 **타이핑 제거**이지 controller/pose proof나 fresh
ENGAGED heartbeat 검사를 없앤 것이 아니다.

**Kanu 현재 계약:** actor transport는 protocol 2 / schema 3, reward threshold는 0.5다.
2026-07-30 새 learner는 GPU 5에서 canonical offline demo 2,037개를 로드해 health-ready가
됐다. 최신 읽기 전용 스냅샷은 online replay 400 / intervention 225, learner 301 /
gradient 602 / policy version 6으로 schema-3 실물 전이가 다시 유입되고 학습되는 중이다.
이 수치는 계속 변한다. 이전 learner RAM에만 있던 테스트 replay 257 / intervention 107 /
learner step 158은
checkpoint가 없고 의미 없는 시험값이라는 사용자 판단에 따라 폐기했다. offline demo pickle은
그대로 보존했다. PID와 run root는 스냅샷이므로 매번 `./run_hil_server.sh --check`로 읽는다.

**2026-07-30 개입 손맛 수정이 실기에서 PASS (`4197f5b`).** 개입 중 팔이 "빳빳"했던 원인은
10 Hz 자체가 아니라 **타깃 갱신 방식**이었다 — `env.step`이 100 ms 창에서 관절 타깃을 한 번만
세우고 그 사이 리더를 다시 읽지 않아, 250 Hz 업샘플러가 매 창마다 가속→도달→제동→**정지**를
반복했다(gRPC 왕복이 `env.step` 밖이라 실효 갱신은 6~10 Hz). 해결은 검증된 EEF teleop과 같은
방식이다: 개입 중 **리더를 30 Hz로 재샘플링** + **One-Euro**(`bridge_stages.py`에서 비트 동일
이식) + `InterventionBudget`이 창당 총 변위를 `ACTION_SCALE`로 묶는다
(신규 `ur_env/envs/leader_stream.py`, `ros_backend.py`는 무변경).
실기 3 run 중 뒤 두 run 전체 PASS (2026-07-30, 실제 UR7e, `tests/run_real_hil.py`).
첫 run(DRY `--scale 0.5`, 개입 144)은 고친 판정으로 **SKIP**이다 — 포화 표본을 빼면
축별 여기가 2 cm 게이트에 못 미친다. 포화를 포함하면 세 run **전부** FAIL로 나온다
(잔차 0.776 / 0.154 / 0.358) — 그래서 포화 제외는 첫 run을 구제하는 사후 변명이 아니다:

| run | 결과 |
| --- | --- |
| DRY RUN `--scale 1.0` (300스텝, 개입 272) | frame-map 잔차 **0.016** / alpha **1.005** / 표본 141(포화 131 제외) · action-exec dp_ratio 중앙값 **1.000** · held **0 %** |
| ARMED `--scale 1.0 --max-steps 150` (개입 120) | 잔차 **0.130** / alpha **0.983** / 표본 51(포화 69 제외) · dp_ratio **1.000** · held **0 %** · **조작자 손맛 확인 양호** |

모든 개입 스텝에 `substeps=2`(창당 타깃 3회 갱신 = 첫 타깃 + 서브스텝 2)가 기록됐고
`governed=0`이다 — 예산이 governor보다 타이트해 **먼저 묶는** 설계대로의 동작이다. ⚠️ 단 이 세 run은 **첫 타깃만 집계하던 코드**로 측정됐고, shipped config에서는 `_paced_request`가 요청을 `ACTION_SCALE/3`(0.00417 m)로 깎아 서브스텝 governor 캡(`v_max/substep_hz`=0.0050 m)에 **닿을 수 없다** — 즉 `governed=0`은 "창 전체에서 절삭 없음"이 아니라 애초에 절삭이 불가능했다는 뜻이다.
원인 규명·금지사항·5개 실측표는 [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §9,
요약은 [`docs/testing/08_OPEN_GAPS.md`](docs/testing/08_OPEN_GAPS.md) G24.

**2026-07-29 첫 실물 production-model E2E smoke 성공.** 실제 UR7e에서
`ENGAGE=GELLO`, `DISENGAGE=policy`를 확인했고, Kanu에 online transition **201개**
(intervention **153개**)가 들어가 learner **102 step / gradient 204 / policy version 2**까지
진행했다. actor의 RPC deadline 예외 뒤 controller도 FPC에서 STJC로 자동 복귀했다.

**크롭 불일치(G15)는 재학습이 아니라 분리(decoupling)로 해결됐고 실물 actor 경로에도
들어갔다.** 액터가 분류기에게 **무크롭 원본 JPEG를 sidecar로 따로** 보낸다
(`ur_env/classifier_sidecar.py`). 정책은 측정된 `IMAGE_CROP`을 그대로 유지한다. 같은 변경에서
checkpoint 디렉터리 해시(G19)도 고쳤다. GUI에서 실제 episode의 마지막 classifier
확률/threshold/verdict가 표시되는 것까지 관측했다. MANUAL에서도 이 telemetry는 계속 돈다.
장시간 사후 감사를 위한 별도 영구 verdict 로그 정리는 여전히 남아 있다.

> **이전 판 문구(보존):** *"안 되는 것 — RL 루프의 reward. 뷰어는 믿어도 되고 RL reward는
> 믿으면 안 된다."* 이 경고는 sidecar 이전 기준이다. 이제 뷰어와 RL 경로는 **같은 그림**을 본다
> (같은 무크롭 JPEG, 같은 `decode_classifier_image()` 레시피).

**아직 남은 것** — 첫 E2E에서 0.6/0.8 s 경계가 정상 832.3 ms reply를 stale로 잘못 거부한
문제는 현재 1.5/2.0 s bounded 값으로 완화했다. 하지만 장시간 run에서 learner/GPU contention과
RPC tail latency가 어떻게 변하는지는 계속 계측해야 한다. classifier는 현재 정확도가 충분하지
않아 MANUAL이 기본이고, AUTO를 production 기본으로 되돌리려면 새 데이터로 재학습·재검증해야
한다. sidecar가 고치지 못하는 cam1 **가림(occlusion)**과 `08_OPEN_GAPS.md`의 G27/G28
(실행 액션 기록 정합성)도 남아 있다. 새 schema-3 Kanu lineage의 transition/학습은 실제로
진행 중이며, MANUAL `MARK SUCCESS` 버튼으로 끝낸 episode의 one-shot provenance만 별도로
한 번 확인하면 된다.

**개입 쪽에서 남은 것 하나** — 위 2 run은 zero-policy + 사람 개입이라 창 주기가 짧았다.
연속 운용에서 창이 늘어져(첫 publish 5.47 s, learner step 중앙 1.12 s) **예산이 소진되면 남은
시간은 HOLD**이고, 그건 필터·rate·외삽으로 못 고친다 — **G21(RPC 지연) 종속**이다
(`08_OPEN_GAPS.md` G21, `04_HIL_INTERVENTION.md` §9.5).

## 읽는 순서 — 이 셋만 읽고 멈춰라

1. **이 파일** — 무엇을 만들고 있고 어디를 봐야 하는지. (2분)
2. [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) —
   **현재 진입점.** 첫 실물 E2E의 증거, 정확한 PASS/미완료 경계, HIL 제어권 의미, 다음 구현
   우선순위와 재현 CLI. (15분)
3. [`docs/testing/08_OPEN_GAPS.md`](docs/testing/08_OPEN_GAPS.md) — 지금 무엇이 깨져 있는지.
   **G15(크롭 불일치)가 어떻게 sidecar로 닫혔고 무엇이 안 닫혔는지(가림), 워크스페이스 박스가
   꺼져 있던 이유(G1)가 여기 있다.**

그다음은 **하려는 일 하나만** 아래에서 골라 읽는다. 나머지는 필요할 때 펼친다 —
순서대로 다 읽지 마라.

## 문서 지도

### A. 상태를 더 깊이 이해하려면

| 문서 | 무엇이 들어 있나 |
| --- | --- |
| [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) | **최신 정본.** 실물 E2E, operator episode 상태기계, 3-CLI와 다음 방향 |
| [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md) | 첫 E2E 이전의 상세 리그 조사 기록. 최신 상태 지침은 위 문서가 대체 |
| [`serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 기록. learner 구현 §1–10 / actor·하드웨어 §11 / classifier 조사 §12 |
| [`serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md`](serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md) | 0.85 → 0.5 → 0.2의 **역사적** 근거와 07-29 누출 감사. 현재 runtime 기본값은 사용자 결정으로 다시 **0.5**이며 코드 상수가 권위다 |
| [`docs/testing/README.md`](docs/testing/README.md) | 하드웨어·통신 검증 런북 인덱스(00~09) + 항목별 PASS/미검증 상태표 |
| [`serl_ur_infra/README.md`](serl_ur_infra/README.md) | env 설계 규약(좌표계·액션 계약·컨트롤러 격차) + `serl_ur_infra/` 문서 인덱스 |
| [`serl_ur_infra/REMOTE_ACTOR_GRPC.md`](serl_ur_infra/REMOTE_ACTOR_GRPC.md) | gRPC 전송 계약 v2 — 서버 entrypoint 3종의 차이 (영문) |

### B. 무언가를 실행하려면 (런북)

| 하려는 일 | 문서 |
| --- | --- |
| 셋업 · 빌드 · 인터프리터 함정 · 비상 정지 | [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) |
| 실물 HIL 세션 기동 (운영용 3-CLI, preflight, actor) | [`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](docs/testing/09_HIL_ACTOR_RUNBOOK.md) — 정상 운용은 `run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh` 세 terminal |
| kanu에서 learner 띄우기 | [`serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md`](serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md) |
| 녹화 take를 learner용 offline demo로 변환 | [`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md) (`40b99f8`) — **`--outcome success\|truncated`는 사람이 명시한다.** 변환기는 성공을 추측하지 않는다. learner는 offline demo가 0이면 학습을 시작하지 않는다 |
| 라이브 reward classifier 뷰어 보기 | [`serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md`](serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md) — **2026-07-29 실기 검증 완료.** 터미널 4개 절차·인터프리터 함정·크롭 주의·트러블슈팅 |
| mock RViz로 개입 경로 확인 (실기 위험 0) | [`serl_ur_infra/RVIZ_HIL_TEST_CLI.md`](serl_ur_infra/RVIZ_HIL_TEST_CLI.md) |
| GELLO 개입 손맛·좌표계 **실기** 검증 | [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §4.5·§9 — `serl_ur_infra/tests/run_real_hil.py`. 기본 `DRY_RUN`이고 `--arm`을 줄 때만 움직인다. **정책이 zero 고정이라 learner도 gRPC 서버도 필요 없다** — 3-CLI 운영 워크플로우와 혼동하지 말 것(실제로 혼동이 있었다). 컨트롤러 요구도 다르다(FPC) → [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) §3.5 |
| GELLO로 실기 팔 텔레옵 (HIL 개입이 이 경로 위에 있다) | [`docs/ros2/GELLO_UR7E_EEF_MODE.md`](docs/ros2/GELLO_UR7E_EEF_MODE.md) · 조인트 모드는 [`GELLO_UR7E_REAL_ROBOT.md`](docs/ros2/GELLO_UR7E_REAL_ROBOT.md) |
| 처음부터 환경 세팅 / 세션 전 프리플라이트 | [`docs/ros2/GELLO_UR7E_SETUP_CLI.md`](docs/ros2/GELLO_UR7E_SETUP_CLI.md) |
| 그리퍼만 단독으로 | [`docs/ros2/GELLO_UR7E_GRIPPER.md`](docs/ros2/GELLO_UR7E_GRIPPER.md) |
| 카메라가 안 뜰 때 | [`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md) · [`docs/testing/06_SENSORS.md`](docs/testing/06_SENSORS.md) |

### C. 특정 진행 중 작업을 이어받으려면

| 작업 | 시작점 |
| --- | --- |
| **분류기를 개입·학습에 연결** | [`serl_ur_infra/REWARD_TO_RL_INTEGRATION_KO.md`](serl_ur_infra/REWARD_TO_RL_INTEGRATION_KO.md) — reward/termination 계약, 개입 이중 라우팅, RLPD 50:50, 연결 순서. **코드에서 직접 추적해 쓴 문서다.** 아래 두 블로커가 선행 조건 |
| ~~크롭 불일치 해소~~ → **sidecar verdict 가시화** | `08_OPEN_GAPS.md` G15 → `ur_env/classifier_sidecar.py` (모듈 docstring이 설계 근거 전부). production actor 배선은 첫 실물 E2E에서 사용됐다. 남은 것은 per-transition probability/verdict GUI·영구 로그 검증이다. **`IMAGE_CROP`을 지우는 건 여전히 해결이 아니다** |
| ~~checkpoint 로딩~~ → **해결됨(G19)** | `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임해 orbax **디렉터리**를 해시한다. 두 `DEFAULT_*_SHA256` 상수도 은퇴 체크포인트(`e329986b…`)에서 교체됐다 |
| **canonical demo artifact** (learner 시작 조건) | ✅ 2026-07-29 사람 승인·생성 완료. `take_23` 제외 23 take → 2,037 transition, SHA256 `f9718558…032fa`; laptop3와 Kanu strict-load 통과. 경로·품질 주의는 [`serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md`](serl_ur_infra/RECORDED_TAKE_DEMO_CONVERSION_KO.md) + `08_OPEN_GAPS.md` G20 |
| actor entrypoint 실기 재실행·timeout 계측 | [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) + [`docs/testing/09_HIL_ACTOR_RUNBOOK.md`](docs/testing/09_HIL_ACTOR_RUNBOOK.md) |
| 개입 루프·좌표계·메타데이터 | [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) |
| 장애 주입 매트릭스 (거의 미검증) | [`docs/testing/07_FAILURE_INJECTION.md`](docs/testing/07_FAILURE_INJECTION.md) |

### D. 🗄️ 낡음 / 대체됨 — 찾더라도 따르지 말 것

| 문서 | 왜 |
| --- | --- |
| [`serl_ur_infra/HIL_SERL_STATUS_AND_NEXT.md`](serl_ur_infra/HIL_SERL_STATUS_AND_NEXT.md) | 2026-07-24 서버 핸드오프 아카이브. agentlace 5588/5589 · `train_rlpd.py` · jax 0.4.35 핀을 권고한다 — **그대로 준비하면 learner가 즉사한다.** 문서 머리에 대조표가 있다 |
| [`serl_ur_infra/RL_RECEIVE_SERVER.md`](serl_ur_infra/RL_RECEIVE_SERVER.md) · [`HIL_RLPD_RECEIVE_SERVER_KO.md`](serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md) | receive-only 마일스톤 기록. 브랜치·체크포인트가 은퇴했고 **19-D `state` 순서를 v1(틀린 순서)로 적은 곳이 남아 있다** |
| [`serl_ur_infra/ACTOR_ADAPTER.md`](serl_ur_infra/ACTOR_ADAPTER.md) | agentlace 로컬 어댑터(`scripts/train_rlpd_actor.py`) 기준. **실기 경로가 아니다** — transition 계약 설명만 유효 |
| [`docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md`](docs/rl/GELLO_UR7E_HIL_SERL_PLAN.md) · [`GELLO_UR7E_SERL_ENV_STATUS.md`](docs/rl/GELLO_UR7E_SERL_ENV_STATUS.md) | 2026-07-24 채택 계획과 첫 커밋 스냅샷. upstream 구조 설명만 참고 가치 |
| [`docs/ros2/GELLO_UR_ROS2_PLAN.md`](docs/ros2/GELLO_UR_ROS2_PLAN.md) · [`GELLO_UR_ROS2_BRINGUP.md`](docs/ros2/GELLO_UR_ROS2_BRINGUP.md) | UR5e + Jazzy 시절. 이 브랜치는 UR7e + Humble이고 실기 경로가 이미 있다 |
| [`docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md`](docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md) | EEF 설계·근거 아카이브. 대체된 부분에 `⛔ SUPERSEDED` 표시가 붙어 있고, 조작자 정본은 `GELLO_UR7E_EEF_MODE.md`다 |
| [`README.md`](README.md) (루트) | 2026-07-20. Panda/UR5e sim 소개 중심이라 HIL-SERL이 없다. 데이터셋·HF 절만 유효 |
| [`docs/sim/`](docs/sim) · [`docs/ros2/GELLO_FRANKA_RVIZ.md`](docs/ros2/GELLO_FRANKA_RVIZ.md) · [`GELLO_ROS2_CONTROL_REFERENCE.md`](docs/ros2/GELLO_ROS2_CONTROL_REFERENCE.md) · [`ros2/`](ros2) · [`docs/reference/`](docs/reference) | Panda/Jazzy 계보이거나 upstream 원본. **이 작업 범위 밖** |

### E. 모방학습(ACT/Diffusion/FM) 시대 문서 — 유효하지만 HIL과 다른 스택

같은 로봇·같은 브리지를 쓰지만 **RL 경로와 코드가 다르다.** 섞지 말 것.
[`docs/ros2/GELLO_UR7E_ACT_DEPLOY.md`](docs/ros2/GELLO_UR7E_ACT_DEPLOY.md)(유일하게 실기 검증) ·
[`GELLO_UR7E_DIFFUSION_DEPLOY.md`](docs/ros2/GELLO_UR7E_DIFFUSION_DEPLOY.md) ·
[`GELLO_UR7E_FM_DEPLOY.md`](docs/ros2/GELLO_UR7E_FM_DEPLOY.md)(뒤 둘은 실기 미검증) ·
[`ros2_ur_ws/src/gello_policy/README.md`](ros2_ur_ws/src/gello_policy/README.md) ·
[`ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md`](ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md).
데이터 기록은 [`ros2_ur_ws/src/gello_recorder/README.md`](ros2_ur_ws/src/gello_recorder/README.md) ·
[`docs/ros2/GELLO_UR7E_RECORDING.md`](docs/ros2/GELLO_UR7E_RECORDING.md).
단위 확인은 [`docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md`](docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md).
종결된 조사 기록: [`ros2_ur_ws/gello_logs/experiments/README.md`](ros2_ur_ws/gello_logs/experiments/README.md)(startup-snap,
수정 반영 완료) · [`docs/ros2/GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md`](docs/ros2/GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md)(결과 부정적, 기능 default-OFF).

## 코드 지도

```
serl_ur_infra/
  ur_env/envs/ur7e_env.py          get_im, clip_safety_box, go_to_reset, 관측 조립
  ur_env/envs/config.py            속도 3층(ACTION_SCALE/GOVERNOR/UPSAMPLER), 카메라 토픽
  ur_env/envs/wrappers.py          GelloIntervention, 데드맨, 그리퍼 페널티
  ur_env/envs/leader_stream.py     개입용 리더 30 Hz 재샘플링 — One-Euro 이식(bridge_stages.py
                                   비트 동일) + InterventionBudget(창당 변위 예산). **설계 근거가
                                   모듈 docstring에 전부 있다**
  ur_env/envs/frame_wrappers.py    RelativeFrame, Quat2EulerWrapper
  ur_env/envs/ros_backend.py       rclpy 백엔드, 250 Hz 업샘플러
  ur_env/remote_actor.py           actor 루프, 전이 생성·전송, sidecar 부착 계측
  ur_env/operator_session.py       GUI 상태/서비스, MANUAL/AUTO·HOME/scene-ready operator gate
  ur_env/classifier_sidecar.py     분류기 sidecar 계약 — build/validate/decode, 정지·2 Hz 게이트,
                                   directory_sha256. **설계 근거가 모듈 docstring에 전부 있다**
  ur_env/rlpd_receive_server.py    서버 ingress + RewardClassifierRuntime + checkpoint_sha256
  ur_env/learner/                  RLPD learner, checkpoint, fingerprint
  ur_experiments/cube_in_cup.py    태스크 config (측정값 전부 여기, IMAGE_CROP 포함)
  scripts/run_remote_rlpd_actor.py actor entrypoint
  tests/run_real_hil.py            실기 개입 러너 (파일 상단 주석이 안전 설계를 설명)
ros2_ur_ws/
  run_hil_server.sh                Terminal 1: Kanu learner 검증/재사용·기동 + SSH tunnel
  run_hil_hardware.sh              Terminal 2: UR7e + Robotiq + passive GELLO supervisor
  run_hil_session.sh               Terminal 3: cameras + GUI + preposition/preflight + actor
  run_hil_actor.sh                 actor 실행 래퍼 (11단계 preflight + controller cleanup)
  run_hil_preposition.sh           RESET pose 이동 + controller handoff proof 생성
  run_hil_gui.sh                   데드맨/개입 GUI
  launch_cameras.sh                RealSense 2대 (시리얼 자동 해석)
  run_classifier_viewer.sh         라이브 분류기 뷰어 (랩톱 CPU)
  run_remote_classifier_viewer.sh  라이브 분류기 뷰어 (kanu GPU + SSH 터널)
```

## 반드시 지킬 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **gRPC 코드는 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로만.** 시스템 `python3`의
  grpcio 1.30.2가 손상돼 오류 없이 100% CPU로 무한 정지한다.
  **단 "무조건 venv"는 아니다** — `tests/run_real_hil.py`는 gRPC를 안 쓰고 rclpy를 쓰므로
  **시스템 `python3`가 맞다.** 그리고 **ROS 러너에서 `PYTHONPATH`를 덮어쓰면 rclpy가 사라진다**
  (`RuntimeError: rclpy not available` — 2026-07-30 실기에서 발생). pytest 정본 명령은 반대로
  덮어쓰는 게 맞다. 두 규칙의 대비는 [`docs/testing/00_SETUP_AND_SAFETY.md`](docs/testing/00_SETUP_AND_SAFETY.md) §3.4에 표로 있다.
- **`IMAGE_CROP`을 "분류기가 안 맞으니" 지우지 말 것.** 데이터셋 측정값이고 정책이 1차 소비자다.
  해결은 **분류기에게 무크롭 sidecar를 따로 주는 것**이다(`ur_env/classifier_sidecar.py`).
  *(이전 판은 "해결은 classifier 재학습이다"라고 적었다 — 재학습은 채택되지 않았다. 분리를
  택한 덕분에 `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 측정값이 전부 살아남았다.)*
- **개입이 빳빳하다는 이유로 가속도 제한 · `target_stale_s` · `soft_start_s`를 되돌리지 말 것.**
  실측은 셋 다 **반대로 간다**: `UPSAMPLER.max_accel_rad_s2`(8.0) 제거 → 관절 완전정지
  16 % → **76 %**(리플 2.27 → 4.17), `target_stale_s` 0.30 → 0.50 → 정지 17.5 % → **54.5 %**,
  `soft_start_s` 0.7 → 0 → 리플 1.92 → **4.09**. 셋은 원인이 아니라 낮은 갱신 주기가 만든
  stop-and-go를 **완화하고 있던 것**이다. 근거(창 조건·메커니즘 포함)는
  [`docs/testing/04_HIL_INTERVENTION.md`](docs/testing/04_HIL_INTERVENTION.md) §9(특히 §9.3),
  요약은 `08_OPEN_GAPS.md` G24.
- **`ACTION_SCALE`을 "개입이 느리다"는 이유로 올리지 말 것.** 그건 안전 한계가 아니라
  **정책의 액션 의미 자체**다 — 0.0125 m/step × 10 Hz = **12.5 cm/s가 정책의 최고속**이고,
  개입 변위 예산도 여기서 직접 파생된다(`ur_env/envs/config.py`의 `ACTION_SCALE` INVARIANT 주석).
  올리면 사람이 **정책이 실행할 수 없는 시범**을 보이게 되고, canonical demo 2,037 transition과
  액션의 의미가 갈리는데 learner fingerprint에 `ACTION_SCALE`이 **없어서**(G18) 조용히 통과한다.
  손맛 확인용으로는 `serl_ur_infra/tests/run_real_hil.py --scale`을 쓴다 — 3층
  (`ACTION_SCALE`/`GOVERNOR`/`UPSAMPLER`)을 **함께** 곱하고 그 배율을 CSV 헤더에 남긴다.
- **테스트는 passed 수를 볼 것 — 그리고 어느 인터프리터인지 같이 적을 것.** PYTHONPATH에서
  `serl_launcher`가 빠지면 조용히 떨어지고 skip 사유가 거짓말을 한다. 2026-07-30 `4197f5b`
  기준선은 **579 passed / 11 skipped** (8.98 s,
  `/home/laptop3/venvs/gello-hil-actor/bin/python`). 재현 명령 정본은
  `serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md:254-264`:

  ```bash
  cd /home/laptop3/gello_software
  set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
  OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
    /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
    -p no:cacheprovider serl_ur_infra/tests
  ```

  계보: 333 → 337(`40b99f8`) → 429(classifier sidecar) → **497**(07-30 오전, `4197f5b` 이전)
  → **579**(`4197f5b`; 신규 82 = `test_leader_stream` 28 / `test_governor_dt` 38 /
  `test_intervention_substeps` 16). 옛 문서에 남은 333·337·429·497은 전부 이전 값이다.
  🪤 **인터프리터를 안 적은 "passed 개수"는 무의미하다.** 같은 명령을
  `/home/laptop3/venvs/hilserl/bin/python`(jax 0.5.3 있음, numpy 1.26.4)으로 돌리면
  jax 테스트가 더 돌아 passed가 늘고 skipped가 11 → 4로 줄어든다. gRPC·actor 경로의
  정본 인터프리터는 actor venv이므로 **기준선은 actor venv 값**이다.

  ✅ **해소됨** — 한때 hilserl에서 `test_governor_dt.py::test_env_step_surfaces_governed_in_info`
  1건이 떨어졌다(`0.8485281248` vs `0.8485281374 ± 8.5e-10`). 원인은 numpy 승격 차이가
  아니라 **허용범위가 애초에 잘못됐던 것**이다: 액션 dtype이 계약상 `float32`라
  `xi = action * ACTION_SCALE`에 ~1e-7 상대오차가 실리는데 `rel=1e-9`로 잡혀 있었다.
  actor venv에서 통과한 건 우연이다. `rel=1e-6`으로 고쳐 **두 인터프리터 모두 통과**한다.
  잡으려는 회귀(캡이 안 물림 1.0, 3배 오차)는 1e-6에서 수십만 배 떨어져 있다.
  그리고 `serl_ur_infra/tests/test_env_fake_backend.py`는 pytest에서 **0개 수집**되므로 이
  총계에 **아무 흔적도 남기지 않는다** → `08_OPEN_GAPS.md` G25.
- **메인 브랜치는 여러 사람이 공유한다.** 머지·리베이스 전에 상대 checkout이 깨끗한지 확인할 것.
