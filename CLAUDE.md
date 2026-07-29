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
통신한다. reward 권위는 서버에 있다. 명목 제어 루프는 10 Hz지만 현재 RPC timeout은 0.6 s이고,
첫 실물 online-learning run에서 동시 학습/추론 contention 때문에 그 deadline을 넘었다.
분류기는 정책 관측이 아니라 **자기 전용 무크롭 이미지(sidecar)를 약 2 Hz로** 따로 받는다.

**branch `feat/gello-ur7e-humble-22.04`** (origin/HEAD). 2026-07-29 머지 `3f199d4`가 로봇/하드웨어
작업을 이 브랜치로 가져왔다. **워크트리 분리는 끝났다** — 로봇 코드와 learner 코드가 **다른
checkout에 있다**고 적힌 문서는 전부 낡은 것이다(아직 여러 개 남아 있다). 작업은
`/home/laptop3/gello_software` 한 곳에서만 한다.

## 지금 상태 한 줄

**2026-07-29 첫 실물 production-model E2E smoke 성공.** 실제 UR7e에서
`ENGAGE=GELLO`, `DISENGAGE=policy`를 확인했고, Kanu에 online transition **201개**
(intervention **153개**)가 들어가 learner **102 step / gradient 204 / policy version 2**까지
진행했다. actor의 RPC deadline 예외 뒤 controller도 FPC에서 STJC로 자동 복귀했다.

**크롭 불일치(G15)는 재학습이 아니라 분리(decoupling)로 해결됐고 실물 actor 경로에도
들어갔다.** 액터가 분류기에게 **무크롭 원본 JPEG를 sidecar로 따로** 보낸다
(`ur_env/classifier_sidecar.py`). 정책은 측정된 `IMAGE_CROP`을 그대로 유지한다. 같은 변경에서
checkpoint 디렉터리 해시(G19)도 고쳤다. 단 이번 run은 per-transition classifier 확률을 GUI나
영구 로그로 관측하지 못했으므로 **online verdict 정합 검증은 아직 남아 있다.**

> **이전 판 문구(보존):** *"안 되는 것 — RL 루프의 reward. 뷰어는 믿어도 되고 RL reward는
> 믿으면 안 된다."* 이 경고는 sidecar 이전 기준이다. 이제 뷰어와 RL 경로는 **같은 그림**을 본다
> (같은 무크롭 JPEG, 같은 `decode_classifier_image()` 레시피).

**아직 안 되는 것** — 연속 운용 중 첫 policy publish 경계에서 `Step RPC`가 0.6초를 넘어 actor가
종료됐다. startup cold-JIT는 해결됐지만, 실제 actor와 함께 돌 때 learner step 중앙값이 약
1.12초였고 첫 publish는 5.47초였다. 또한 launcher는 commissioning용 `ENGAGED` 시작을 강제하고,
classifier verdict GUI와 `SUCCESS -> HOME -> scene reset WAIT -> operator RESUME` 상태 기계가 없다.
sidecar가 고치지 못하는 cam1 **가림(occlusion)**도 남아 있다.

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
| [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) | **최신 정본.** 2026-07-29 실물 E2E 결과와 다음 방향 |
| [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md) | 첫 E2E 이전의 상세 리그 조사 기록. 최신 상태 지침은 위 문서가 대체 |
| [`serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md`](serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 기록. learner 구현 §1–10 / actor·하드웨어 §11 / classifier 조사 §12 |
| [`serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md`](serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md) | threshold를 0.85 → 0.2로 내린 근거 + 07-29 누출 감사. 07-28 수치와 07-29 수치를 구별해서 인용할 것 |
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
  ur_env/envs/frame_wrappers.py    RelativeFrame, Quat2EulerWrapper
  ur_env/envs/ros_backend.py       rclpy 백엔드, 250 Hz 업샘플러
  ur_env/remote_actor.py           actor 루프, 전이 생성·전송, sidecar 부착 계측
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
- **`IMAGE_CROP`을 "분류기가 안 맞으니" 지우지 말 것.** 데이터셋 측정값이고 정책이 1차 소비자다.
  해결은 **분류기에게 무크롭 sidecar를 따로 주는 것**이다(`ur_env/classifier_sidecar.py`).
  *(이전 판은 "해결은 classifier 재학습이다"라고 적었다 — 재학습은 채택되지 않았다. 분리를
  택한 덕분에 `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 측정값이 전부 살아남았다.)*
- **테스트는 passed 수를 볼 것.** PYTHONPATH에서 `serl_launcher`가 빠지면 조용히 떨어지고
  skip 사유가 거짓말을 한다. 2026-07-29 실기 준비 기준선은 **497 passed / 11 skipped**다.
  *(HEAD `40b99f8`에서는 337이었고, 옛 문서의 333/429는 그보다 이전 값이다.)*
- **메인 브랜치는 여러 사람이 공유한다.** 머지·리베이스 전에 상대 checkout이 깨끗한지 확인할 것.
