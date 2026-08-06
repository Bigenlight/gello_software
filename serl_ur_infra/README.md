# serl_ur_infra

`third_party/hil-serl`의 `serl_robot_infra`(Franka 전용)에 대응하는 **UR7e + GELLO** robot infra.
`FrankaEnv`의 관측/액션 계약을 그대로 복제해서 hil-serl의 wrapper 체인·actor 루프가 무수정으로 돌게 한다.

> 🚩 **이 리포에 처음 왔다면** 루트 [`CLAUDE.md`](../CLAUDE.md) →
> [`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md)
> 순서로 읽어라. 이 README는 **env 설계 규약**과 **이 디렉터리 문서 색인**이지 현재 상태 문서가 아니다.

> ⚠️ **정책 경로는 여전히 `config.DRY_RUN=True`(명령 미발행)가 기본이다.**
> 실제 명령은 검증된 `run_hil_actor.sh --arm`만 이 값을 해제한다. 2026-07-29 첫
> production-model 실기에서 policy 48 transition, GELLO intervention 153 transition,
> Kanu learner 102 step/policy version 2와 controller 자동 복귀까지 확인했다. 따라서 예전
> "UNTESTED SKELETON / 정책 실기 미검증" 머리말은 낡았다. 다만 첫 publish 경계의 RPC timeout으로
> continuous run은 PARTIAL이며, learner-side v1/v2 publish가 actor/robot에 전달됐다는 증거는 없다.

> 🆕 **(2026-07-29) reward classifier는 이제 정책과 다른 이미지를 본다 — G15 해소.**
> 분류기는 무크롭 프레임으로 학습됐는데 actor가 **정책의 크롭된** 관측을 먹이고 있었다
> (recall@0.85 100% → 33.3%). **재학습이 아니라 분리로 고쳤다**: actor가 관측에
> **무크롭 128×128 JPEG sidecar**를 ~2 Hz로, **팔이 정지해 있을 때만** 덧붙이고
> 서버는 그것만 분류한다. 정책의 `IMAGE_CROP`은 **불변**이고, **관측 schema hash도 불변**이다
> (`3459098d…` — hash는 `CANONICAL_OBSERVATION_SPEC` 문서에서 나오지 wire payload에서 나오지 않는다).
> 계약은 `ur_env/classifier_sidecar.py`, 전모는
> [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) §12.1.
> production actor 배선은 첫 실물 E2E에서 사용됐다. 이제 actor status/GUI가 마지막으로
> 평가된 probability와 threshold를 표시하고 서버 replay provenance에도 verdict를 남긴다.
> 장시간 online 분포 정합은 더 확인해야 한다. **팔 가림 병리(`take_21`)도 안 고쳐졌다**;
> 정지 게이트가 완화할 뿐 진짜 해법은 카메라 배치다.

> 🔴 **(2026-07-31) GPU 서버가 `kanu` → `junhyeong_ai`로 바뀌었다.** learner는 이제
> **`junhyeong_ai` GPU 0**(RTX 5070 Ti ×1, sm_120)에서 돌고, 데이터·모델은 전부
> **`~/hil-serl-data/`** 한 뿌리에 있으며, Terminal 1은 **환경변수 0개**로 `./run_hil_server.sh`다.
> **3-CLI 절차 자체는 안 바뀌었다** — `run_hil_hardware.sh`에는 서버 참조가 0개이고
> `run_hil_session.sh` / `run_hil_actor.sh`는 터널의 로컬 끝 `127.0.0.1:50153`만 본다.
> 실기 세션도 **PASS**했다(2026-07-31, 조작자 확인: replay 316 / intervention 210).
> 🗄️ **이 README와 다른 문서에 남은 kanu 시절 숫자·PID·경로는 kanu에서 실제로 측정된
> 것이므로 그대로 둔다 — 호스트만 바꿔 읽지 말 것.** 환경변수는 `HIL_KANU_REPO` →
> `HIL_REMOTE_REPO`, `HIL_KANU_PYTHON` → `HIL_REMOTE_PYTHON`으로 바뀌었고 **옛 이름은 alias로
> 계속 동작한다**; 신설 `HIL_REMOTE_DATA_ROOT`. ⚠️ **`run_hil_server.sh`로 kanu를 몰 수는 없다**
> (kanu의 classifier가 `hil-serl-data` 밖에 있어 단일 data root로 설명되지 않는다) — kanu는
> 읽기 전용 ssh로만 본다. 정본:
> [DATA_AND_MODELS_JUNHYEONG_AI_KO.md](DATA_AND_MODELS_JUNHYEONG_AI_KO.md) ·
> [SERVER_MIGRATION_E2E_JUNHYEONG_AI.md](SERVER_MIGRATION_E2E_JUNHYEONG_AI.md) ·
> 현재 진입점 [HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) **§3D**.

> 🆕 **(2026-07-30) 성공 판정은 protocol 2 / transition schema 3이며 GUI 기본은 MANUAL이다.**
> canonical observation schema는 여전히 v2/hash `3459098d…`라 서로 혼동하지 않는다.
> MANUAL에서도 classifier sidecar는 계속 서버에서 평가되고 probability/threshold/verdict가
> GUI의 **LAST CLASSIFIER**에 표시되며 현재 process의 RAM replay provenance에 저장된다.
> 다만 자동 종료 권한만 없다. GUI **MARK SUCCESS**는 현재 episode에 one-shot
> `operator_success`를 찍고, 명시적으로
> AUTO로 바꾸면 `auto_success AND classifier_success`가 success가 된다. 최종 식은
> `operator_success OR (auto_success AND classifier_success)`이고 두 mode 신호의 동시 true는
> 거부된다. 현행 threshold는 **0.5**다.
>
> 같은 날 읽기 전용 확인에서 Kanu production learner는 PID `1112465` / GPU 5 / health
> `ready`, run `cube_in_cup_manual_schema3_thr05_20260730_1715`, replay/intervention `400/225`,
> learner/gradient/policy `301/602/6`이었다. 옛 process의 RAM replay는 복원하지 않았고,
> 사람 승인 offline demo **2,037개**는 보존·재로드했다. 정상 3-CLI의 server terminal은
> `cd /home/laptop3/gello_software/ros2_ur_ws && ./run_hil_server.sh`; 읽기 전용 상태 확인은
> `./run_hil_server.sh --check`다. 변동 가능한 최신 counter는 상태 문서 §11.5를 재확인한다.
>
> 🗄️ **(2026-07-31) 위 문단의 PID·GPU·run·카운터는 `kanu` 스냅샷이며 그대로 보존한다.**
> 그 lineage는 kanu의 RAM에만 있었고 **새 서버로 넘어오지 않았다**(kanu에는 저장된 checkpoint가
> **하나도** 없었다 — run root 8개 전부 빈 `checkpoints/`). 두 명령
> (`./run_hil_server.sh`, `--check`)은 **그대로 유효하지만 이제 `junhyeong_ai`를 본다.**
> 계약(protocol 2 / schema 3 / threshold 0.5)과 demo 2,037개는 호스트와 무관하게 그대로다.

> 🆕 **(2026-07-30 오후, `edbb3f5` + `d6965a9`) 개입 추종이 RL 창 밖으로 나갔고 개입 예산이
> 사라졌다.** ENGAGED 동안 팔은 `UR7eEnv`의 **데몬 스레드**가 30 Hz로 몰고, `env.step`은
> 10 Hz **관찰자**가 되어 transition만 뽑는다(`config.INTERVENTION["follow_mode"]` 기본값이
> `"background"`). 이 변경이 이 README의 세 절을 바꿨으니 **셋을 같이 읽어라**:
> **「저장 액션 불변식」**(개입 경로에 **명시적 예외**가 생겼다) ·
> **「norm 비례 축소」**(축별 `np.clip`이 실기를 죽였다) ·
> **「속도 3층」**(개입 경로에서 **governor가 유일한 속도 상한**, **박스가 유일한 위치 상한**).
> 실기 판정은 **조작자 보고뿐이고 계측이 없다** —
> [HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) **§3C**.

최종 운영 checkout은 `/home/laptop3/gello_software`, branch는 `feat/gello-ur7e-humble-22.04` 하나다.
2026-07-29 머지 `3f199d4`로 로봇/하드웨어 작업이 이 브랜치에 들어왔다 — **워크트리 분리 시절 서술은
전부 낡았다.** 통합 상태·checkpoint·Kanu 검증·frozen-trunk feature replay·bounded fake-data
learning E2E 계약은 [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)를 기준으로 한다.

`scripts/run_fake_e2e_actor.py`는 robot를 제어하는 actor가 아니라 `--synthetic-e2e` learner에 canonical raw fake observation 100개를 보내 gRPC→classifier→feature replay→CTA→publish→checkpoint→fresh-process resume를 검증하는 acceptance tool이다. server는 exact actor/run ID, exact 100 inserts, bounded timeout을 강제하고 synthetic-only model ID를 advertise한다. cleanup 후 full checkpoint roundtrip/trunk invariant까지 통과해야 pass한다. synthetic checkpoint는 fingerprint/model scope가 다르므로 production robot lineage에 사용할 수 없다.

📌 이 도구는 **2026-07-31 새 서버 `junhyeong_ai`에서 다시 돌아 PASS**했다(fresh + resume 2 run,
transition 200개) → [SERVER_MIGRATION_E2E_JUNHYEONG_AI.md](SERVER_MIGRATION_E2E_JUNHYEONG_AI.md).
단 그 시험은 sidecar를 붙이지 않고 `intervened=1`도 0건이라 **classifier 추론과 intervention
ingress는 증명하지 못한다** — 그 둘은 같은 날 실기 세션이 닫았다(상태 문서 §3D).

## 이 디렉터리의 문서 (`serl_ur_infra/*.md`)

### 지금 유효 — 지침으로 읽어도 되는 것

| 문서 | 무엇인가 |
| --- | --- |
| [HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) | **현재 진입점.** 첫 실물 E2E 증거(§3), 개입 손맛 오전 `in_window` 검증(§3A), **오후 `background` 추종·예산 제거(§3C)**, **새 서버 `junhyeong_ai` 첫 실기 세션 PASS(§3D·§6.4)**, PASS/미완료 경계(§6·§6.1·§6.3), 3-CLI(§9)와 다음 우선순위(§8) |
| [DATA_AND_MODELS_JUNHYEONG_AI_KO.md](DATA_AND_MODELS_JUNHYEONG_AI_KO.md) 🆕 | **서버 이전 정본 (2026-07-31).** `junhyeong_ai`의 접속·GPU·checkout(`gello_software_runtime`)·`~/hil-serl-data/` 배치·demo/classifier SHA·conda `il` env·**코드 기본값(환경변수 0개)**. 🔴 **kanu에는 저장된 policy checkpoint가 하나도 없었다**는 실측도 여기 |
| [SERVER_MIGRATION_E2E_JUNHYEONG_AI.md](SERVER_MIGRATION_E2E_JUNHYEONG_AI.md) 🆕 | **로봇 없는 실통신 수락 시험 PASS (2026-07-31).** transition 200개로 gRPC→finalize→feature replay→CTA→publish→checkpoint→resume 전 구간, **RPC 실측**(`BeginEpisode` 57.7/82.2 ms, `Step` 156.1/211.0)과 kanu 대비, 그리고 **이 시험이 증명하지 못한 것**(classifier 추론·`intervened=1`·실기 루프 주기) |
| [HANDOFF_NEXT_SESSION_KO.md](HANDOFF_NEXT_SESSION_KO.md) | 첫 E2E 이전의 하드웨어/classifier 상세 조사 기록. 최신 상태 지침으로 쓰지 않는다 |
| [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 상태 기록. learner 구현 §1–10 / actor·하드웨어 §11 / reward classifier 조사 §12. 위 문서보다 깊다 |
| [HIL_SERL_KANU_RUNBOOK_KO.md](HIL_SERL_KANU_RUNBOOK_KO.md) | learner 기동의 **옵션·게이트 의미** 정본 (dry-run → bounded run). ⚠️ **2026-07-31부터 🗄️ 기록 문서다** — 파일명도 본문도 kanu 기준이라 **본문 명령을 그대로 치지 않는다**(예외는 그 안의 `run_hil_server.sh` 절). 호스트·GPU index·경로는 위 `DATA_AND_MODELS_JUNHYEONG_AI_KO.md`가 정본 |
| [REWARD_CLASSIFIER_THRESHOLD_KO.md](REWARD_CLASSIFIER_THRESHOLD_KO.md) | 0.85 → 0.5 → 0.2 결정의 측정·역사. **현행 실행값은 코드 상수 0.5**이므로 옛 문서 숫자를 CLI에 복사하지 않는다 |
| [REMOTE_ACTOR_GRPC.md](REMOTE_ACTOR_GRPC.md) | gRPC protocol v2 / transition schema 3 — 같은 전송을 쓰는 **서버 entrypoint 3종의 차이**. canonical observation schema v2와 구별할 것 |
| [RVIZ_HIL_TEST_CLI.md](RVIZ_HIL_TEST_CLI.md) | mock(`use_fake_hardware`) 4터미널 개입 테스트 절차. 실기 위험 0 |
| [REWARD_CLASSIFIER_LIVE_KO.md](REWARD_CLASSIFIER_LIVE_KO.md) | 라이브 reward classifier 뷰어 런북 (랩톱 CPU, 터미널 4개). **2026-07-29 실기 검증됨.** 인터프리터 함정 · 조용한 실패 · 트러블슈팅 |
| [REWARD_TO_RL_INTEGRATION_KO.md](REWARD_TO_RL_INTEGRATION_KO.md) | **분류기를 개입·학습에 연결하는 사람이 읽을 것.** reward/termination 계약(서버 권위, `next_observations` 기준) · 개입의 **버퍼 이중 기록** · RLPD 50:50 배치 · 연결 순서 6단계 · 감시 지표. 코드에서 직접 추적해 작성 |
| [HIL_LOCAL_INFERENCE_KO.md](HIL_LOCAL_INFERENCE_KO.md) 🆕 | **로컬 추론 모드 (`HIL_POLICY_MODE=local`) 설계 정본.** 정책 추론만 laptop3로 옮겨 블로킹 Step RPC를 10 Hz 루프에서 빼는 opt-in 경로 — 서로 분리된 세 경로(액션/전이 forward/파라미터 pull), 조작자 CLI, **MANUAL 전용 · AUTO fail-closed**, 그리고 정직한 한계(서버는 forward된 전이당 여전히 ~156 ms를 쓰므로 **에피소드 중 백로그가 자라고 대기 화면에서 빠진다**, 파라미터는 한 전송분 뒤처진다, divergence 로깅이 parity 알람이다). proto·`SCHEMA_VERSION`·`remote_actor.py` **무변경**이 설계 조건. 🛑 **실기 미검증** |
| [RECORDED_TAKE_DEMO_CONVERSION_KO.md](RECORDED_TAKE_DEMO_CONVERSION_KO.md) | `gello_recorder`의 `take_*/{vectors.h5,cam1.mp4,cam2.mp4}` 를 strict canonical offline demo pickle로 바꾸는 CLI·동기화·action 복원·라벨 계약 (`40b99f8`). **learner 시작 게이트의 절반이 여기 달려 있다** — online replay ≥ `training_starts`(기본 100) **그리고** offline demo ≥ 1 이어야 학습이 시작된다(`ur_env/learner/batches.py::RLPDBatchSampler.ready`). ⚠️ **headless recorder의 `session_<stamp>/` 는 입력이 아니다** — GUI recorder의 `take_*/` 레이아웃이 필요하다 |

### 🗄️ 기록물 — 사료로만. 여기 적힌 명령을 실행하지 말 것

| 문서 | 왜 |
| --- | --- |
| [HIL_SERL_STATUS_AND_NEXT.md](HIL_SERL_STATUS_AND_NEXT.md) | 2026-07-24 서버 핸드오프. 전송을 agentlace 5588/5589로, 진입점을 `train_rlpd.py`로, jax를 0.4.35로 적는다 — 셋 다 현행과 다르고 **그대로 준비하면 learner가 즉사한다.** 문서 머리에 대조표가 있다 |
| [RL_RECEIVE_SERVER.md](RL_RECEIVE_SERVER.md) | receive-only 마일스톤(영문). 인용된 classifier checkpoint는 은퇴했고 **19-D `state` 순서를 v1(틀린 순서)로 적은 곳이 남아 있다** |
| [HIL_RLPD_RECEIVE_SERVER_KO.md](HIL_RLPD_RECEIVE_SERVER_KO.md) | 은퇴한 `feat/hil-rl-receive-server` 브랜치 시절 작업 정리. 브랜치·워크트리 경로가 낡았다 |
| [ACTOR_ADAPTER.md](ACTOR_ADAPTER.md) | agentlace 로컬 어댑터 `scripts/train_rlpd_actor.py` 기준(영문). **실기 경로가 아니다** — `create_actor_network()`가 `agentlace`를 `NotImplementedError`로 거부한다. transition 계약 설명만 유효 |

디렉터리 밖 관련 문서: [`../docs/testing/README.md`](../docs/testing/README.md)(하드웨어·통신 검증 런북 00~09) ·
[`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)(actor 기동) ·
[`../docs/testing/08_OPEN_GAPS.md`](../docs/testing/08_OPEN_GAPS.md)(미해결 갭) ·
[`../docs/testing/04_HIL_INTERVENTION.md`](../docs/testing/04_HIL_INTERVENTION.md)
(개입 루프 정본. **§4.5 실기 러너 `tests/run_real_hil.py`**, **§9 손맛 실측과 🛑 되돌리면 안 되는 것 3개**) ·
[`../docs/testing/00_SETUP_AND_SAFETY.md`](../docs/testing/00_SETUP_AND_SAFETY.md)
(인터프리터·`PYTHONPATH`·컨트롤러 함정 정본).

## 구조

| 파일 | 대응하는 hil-serl 코드 | 역할 |
| --- | --- | --- |
| `ur_env/envs/ur7e_env.py` | `franka_env/envs/franka_env.py` | gym env 본체 (step/reset, 관측, 카메라, 그리퍼) **+ 개입 추종 데몬 스레드** 🆕 |
| `ur_env/envs/config.py` | `DefaultEnvConfig` | 태스크별 config 베이스 |
| `ur_env/envs/ros_backend.py` | Flask 로봇 서버 (HTTP) | rclpy 백그라운드 노드 — 토픽 I/O |
| `ur_env/envs/policy_delta_controller.py` | (Franka 임피던스 컨트롤러가 하던 일) | 정책 델타 → 거버너 → IK → 게이트 → 조인트 명령. **(2026-07-30, `edbb3f5`) 스레드 안전** — 자체 `RLock`, `_commit()` 한 곳에서 `(T_cmd, q_cmd)` 원자적 갱신, `_hold()`는 `_last_issued` 반환. 그 전에는 두 스레드에서 쌍 불일치 11.1 %, HOLD 틱이 최대 0.2047 rad를 발행, `dq_step_max` 게이트 6.08 % 누출 |
| `ur_env/envs/wrappers.py` | `SpacemouseIntervention` | `GelloIntervention` — 데드맨 + 앵커 클러치 개입. `substep()`=창 안 서브스텝(`in_window`) / **`follow_xi()`=배경 추종 틱**(`background`, **예산·`_paced_request` 없음**) / `_update_follow_arming()`=**유일한 승격 지점**(RL 스레드의 step 경계에서만) |
| `ur_env/envs/leader_stream.py` 🆕 | (upstream에 대응물 **없다** — `SpacemouseIntervention`에는 입력 저역통과나 창 예산이 없고, `filtered_expert_a`는 축 마스킹일 뿐이다: `franka_env/envs/wrappers.py:207-247`) | **개입 손맛 계약** (`4197f5b`). `OneEuro`(`bridge_stages.py:42-106` 비트 동일 이식) · `LeaderFilter`(`note_sample`=리더 cadence / `filtered`=출력 틱, **API가 두 cadence 분리를 강제한다**) · `InterventionBudget`(창당 `ACTION_SCALE` 변위 예산, 경로 길이 회계 — **2026-07-30 오후 `edbb3f5` 이후 `follow_mode="in_window"` 전용이다. 기본 `background` 개입 경로에는 예산이 없다**). 순수 numpy/stdlib — rclpy·gym·config import 없음. **설계 근거가 모듈 docstring에 전부 있다** |
| `ur_env/classifier_sidecar.py` 🆕 | (upstream `classifier_keys` 별도 카메라 등록에 대응) | **reward classifier 전용 무크롭 이미지 계약.** `build_sidecar`(랩톱: full-res BGR → 128×128 → JPEG) · `decode_classifier_frames`(서버: 라이브 뷰어와 같은 레시피) · `validate_sidecar`(구조 검증, numpy만) · `SidecarScheduler`(~2 Hz + 정지 게이트 + 성공 근처 에스컬레이션) · `directory_sha256`(orbax **디렉터리** 체크포인트 핑거프린트, G19) |

### 🆕 개입 추종 스레드 심볼 지도 (2026-07-30 오후, `edbb3f5`)

전부 `ur_env/envs/ur7e_env.py`다. *(줄 번호는 적지 않는다 — 이 파일은 오늘 크게 밀렸다.)*

| 심볼 | 역할 |
| --- | --- |
| `UR7eEnv._follow_loop` / `_follow_tick(now)` | 데몬 스레드 본체와 한 틱. **`_follow_tick`은 시계를 인자로 받고 스스로 읽지 않는다** — 그래서 스레드 없이 가상 시계로 테스트된다 (`PolicyDeltaController.step`·`AccelerationLimitedJointStream.advance`와 같은 규약) |
| `UR7eEnv._emit_arm_command(xi, dt, owner)` | 🛑 **관절 명령의 유일한 출구.** 락 안 **첫 줄**에서 owner를 확인한다. governor·워크스페이스 박스·업샘플러가 전부 이 아래에 있으므로, 추종이 다른 경로로 명령을 만들면 **상한 셋이 통째로 사라진다** |
| `UR7eEnv.arm_intervention_follow` / `disarm_intervention_follow` | 승격/강등. 승격은 `GelloIntervention._update_follow_arming`에서만(앵커·gain 래치·데드맨 읽기가 전부 RL 스레드 상태라서), 강등은 어디서나 |
| `UR7eEnv.await_follower_quiescent()` | disarm의 **ACK**. `reset`이 쓴다 — 20 Hz reset과 30 Hz 추종이 싸우면 `RESET_TOLERANCE_RAD`로 **수렴하지 않는다**. 유계이고 실패는 `RuntimeError`로 보고한다(세션을 wedge하지 않는다) |
| `UR7eEnv.suspend_follower()` | context manager. 🔴 **아직 호출부가 없다** — `remote_actor`의 `WAIT_SCENE_READY` / `WAIT_HOME_APPROVAL`용. 재-arm은 호출자 책임(승격은 step 경계에서만) |
| `UR7eEnv._harvest_follow_window(ctrl_info)` | 창이 **커밋한** 변위 → 이 transition의 액션. **창은 `obs_k → obs_{k+1}` 경계 사이 전부**이고 `env.step` 길이가 아니다 — actor가 gRPC에 쓰는 ~412 ms 동안의 움직임도 이 transition의 것이다. **설계 근거·대가·두 근본 해결이 이 docstring에 전부 있다** |
| `UR7eEnv.command_owner()` / `intervention_follow_armed()` | 소유권·arm 상태 조회 |
| `UR7eEnv.open_gripper_for_reset()` | 에피소드 경계에서 그리퍼 개방. 모든 offline demo가 **열린 그리퍼**에서 시작하므로 닫힌 채 시작하면 정책이 한 번도 행동하기 전에 이미 OOD다(`gripper_position`은 `state[0]`) |

테스트는 `tests/test_intervention_follower.py`(36개, 뮤테이션 15/15 사살)다 — ROS·로봇 없이 돈다.
스레드가 **있어야만** 존재하는 성질(소유권 mux, teardown 순서, `step()` 원자성)만 실제 스레드와
`Event`/`Barrier`로 최악 인터리빙을 강제한다.

**sidecar가 관측 계약을 깨지 않는 이유** (헷갈리기 쉬운 지점이라 여기 적는다):

- 전송이 **generic named-tensor map**(`proto/actor_transport.proto`)이라 **proto 변경이 없다.**
- **관측 schema hash가 불변**이다 — `CANONICAL_OBSERVATION_SPEC` **문서**에서 파생되지 wire
  payload에서 파생되지 않는다. 게다가 `ur_env/actor_network.py::ActorSessionService.step` 이
  canonical 검증 **이전에** 예약 키를 벗겨내므로, 아래 어느 것도 완화할 필요가 없었다:
  `validate_canonical_observation` 의 exact-key 검사, `ur_env/learner/policy.py`, `ReplayIngress`.
- 옛 서버는 남는 텐서를 그냥 무시하므로 **구/신 peer가 공존**한다. 다만 서빙
  `reward_model_id`(`cube-in-cup-all3-ckpt150+sidecar-v1`)가 **입력 계약까지 이름에 담고 있어서**,
  실제로는 짝이 안 맞는 조합이 **핸드셰이크에서 거부**된다 — 서로 다른 픽셀로 reward를
  계산한 채 세션이 끝까지 가는 것보다 낫다.

## 설계 요점

- **액션**: `[-1,1]^7` (6D EEF 델타 + 그리퍼), `ACTION_SCALE`로 실단위 변환 — FrankaEnv와 동일.
- **정책 경로**: 델타를 `T_cmd`에 적분 → 거버너 rate cap → seed 기반 IK → 조인트 스텝 게이트 → 의심스러우면 HOLD. (`eef_delta.py` 후반부의 단순화판; 추후 본체 재사용으로 교체 예정)
- **개입 경로**: 데드맨(GUI 토픽 기본, 추후 풋스위치)을 누르는 순간 (GELLO, 로봇 명령 자세) 앵커 래치 → 앵커 델타를 per-step env 액션으로 재표현(클립이 자연스러운 추격 속도 제한이 됨) → `info["intervene_action"]` 보고.
  - **(2026-07-30 오후, `edbb3f5`) 기본값: 추종은 `env.step` *밖*의 데몬 스레드다**
    (`config.INTERVENTION["follow_mode"] = "background"`). ENGAGED 동안 팔은 `substep_hz`
    (30 Hz)로 리더를 따라가고 **RL 루프는 관찰자**가 되어 10 Hz로 transition만 뽑는다.
    앵커/gain 래치 의미는 그대로고 **"1 스텝 = 1 transition"과 10 Hz 저장 주기도 불변**이다.
    바뀐 것은 **누가 언제 타깃을 발행하는가**다.
    - **왜 창 밖으로 나가야 했나.** 창은 명목 100 ms인데 production actor의 실측 스텝
      주기는 **512 ms(1.95 Hz)** 다 — 나머지 ~412 ms는 블로킹 gRPC `Step` RPC와 카메라
      디코드이고 **둘 다 `env.step` 밖**이다. 창 안 서브스텝은 512 ms 중 앞 66.7 ms만
      타깃을 갱신했다(주기의 **87 %** 공백).
    - **예산이 없다.** `GelloIntervention.follow_xi`에는 `InterventionBudget`도
      `_paced_request`도 없다 → 「저장 액션 불변식」의 **개입 경로 예외**와 「속도 3층」 절.
    - `follow_mode="in_window"`가 아래 이전 동작을 그대로 보존한다.
  - **(2026-07-30 오전, `4197f5b`) — 이제 `follow_mode="in_window"` 전용:** 창 안에서 리더를
    30 Hz로 재샘플링한다. 100 ms 창의 sleep이 서브스텝 페이싱으로 바뀌어
    (`ur7e_env.py::_drive_intervention_substeps`) 타깃이 창당 3회 갱신되고, 리더 입력은
    One-Euro를 지나며, 창 총 변위는 `InterventionBudget`이 `ACTION_SCALE`로 묶는다.
    정책 경로는 `driver is None`으로
    갈라져 예전 그대로 sleep 한 번이다. `config.INTERVENTION["substep_hz"] <= HZ`거나
    `INTERVENTION` 블록이 없는 config는 기능이 꺼지고(그리고 `in_window`로 강제되고)
    변경 이전 동작으로 퇴화한다.
- **카메라**: franka_env처럼 pyrealsense2로 장치를 직접 열지 않고, `launch_cameras.sh`가 띄우는 realsense2_camera 드라이버의 `/camX/.../compressed` 토픽을 구독 (RealSense는 이중 오픈 불가 + 기존 viewer/recorder 생태계와 공존). JPEG 디코드→크롭→128×128 리사이즈→RGB는 FrankaEnv와 동일. `DISPLAY_IMAGE=True`면 정책 시점 이미지를 OpenCV 창으로 실시간 표시 (ImageDisplayer 포팅).
  - **(2026-07-29) 디코드된 full-res BGR은 크롭 직전에 보관된다** (`ur7e_env.py` 의 `self._last_camera_frames[key] = bgr`, ≈`:1064` — 바로 다음 줄이 크롭이다). `last_camera_frames()` 로 꺼내며 **참조지 복사가 아니다 — read-only로 다룰 것.** reward classifier sidecar의 원본이고, 이 덕분에 추가 디코드 비용이 0이다. fake env는 항상 비어 있다.
  - 🪤 **RealSense 시리얼: 카메라는 한 쌍뿐이고 필드가 두 개다.** `serial_no:=` 가 매칭하는 **모듈 시리얼**은 `147122072740`(cam1, plain D435) / `243222072700`(cam2, D435IF)이고, 커널 USB 디스크립터(`journalctl`, `/sys/.../serial`)가 노출하는 **ASIC 시리얼**은 `151623020789` / `322743060038` 이다. **저널 grep으로 "어느 카메라가 붙어 있나"를 판정하지 마라** — 두 세션이 그렇게 해서 각각 정반대의 틀린 결론에 도달했다. 정본은 [`../docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](../docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md).
- **fake_env 모드**: ROS/카메라 없이 space 정의와 zero 관측만 제공 — learner 노드용.

## 좌표계 규약 (중요 — 한 번 틀렸던 것)

**모든 env 액션 델타는 base(world) 프레임이다.** FrankaEnv 계약: 위치는 base
축 평행이동, 회전은 world rotvec을 자세에만 왼쪽 곱 (TCP 점 중심 회전, 위치 불변).
정책이 EEF 프레임에서 행동하는 건 `RelativeFrame` wrapper의 몫이지 env가 아니다.

실제로 겪은 버그: 컨트롤러가 `T_cmd @ exp(xi)`(오른쪽 곱 = **툴 프레임** 증분)로
구현되어 있어서 +x 액션이 리셋 자세에서 base +y로 움직였다. fake backend 테스트
(`tests/test_env_fake_backend.py`)의 "+x 액션 → base +x 이동" 검증으로 발견.
수정: `T_des[:3,3] = p+v; T_des[:3,:3] = so3_exp(w) @ R` 로 위치/회전 분리 조립
(SE(3) 왼쪽 곱도 답이 아님 — base 원점 중심 회전이 되어 위치가 휩쓸림).
`GelloIntervention._expert_delta_action`도 같은 규약 (base 프레임 오차 출력).

🪤 **프레임이 하나 더 있다 — 정책 체인의 `RelativeFrame`.** env가 내보내는 base-frame 액션은
`transform_action_inv`(= `blockdiag(R, R)`)를 통과해 정책 프레임으로 간다. **회전은 norm은
보존하지만 축별 최댓값은 보존하지 않으므로**, `[-1,1]` 검증을 통과시키려면 기록 액션을
**norm으로 비례 축소**해야 한다. 2026-07-30에 이걸 축별 `np.clip`으로 했다가 실기 actor가
즉사했다 → 아래 「norm 비례 축소」 절.

## PolicyDeltaController 현황 vs eef_delta 본체 (이어서 작업할 사람용)

`policy_delta_controller.py`는 `eef_delta.EefDeltaController.step()`의 **후반부
단순화판**이다. 무엇이 있고 없는지:

| 기능 | eef_delta 본체 | 현재 simplified | 비고 |
| --- | --- | --- | --- |
| 속도 거버너 (v_max/w_max rate cap) | ✅ | ✅ | 동일 개념, 단순 구현 |
| 조인트 리밋 게이트 | ✅ | ✅ | `within_joint_limits` |
| 스텝 게이트 → HOLD | ✅ | ✅ (`dq_step_max`) | |
| IK | **branch-lock 해석 IK** (8분기 고정, merge point 처리, 가중 최근접) | `ik_numeric` seed 방식만 | 분기 튐 방지가 약함 — 실기 전 교체 필수 |
| sigma_min 특이점 감속 | ✅ (+탈출 방향은 비대칭으로 통과) | ❌ | |
| keepout 존 | ✅ | ❌ | |
| anti-windup lag 클램프 | ✅ | ❌ (리더 폭주용이라 정책 경로엔 덜 급함) | |
| 해석적 line search | ✅ | ❌ | |
| 워크스페이스 박스 (`ABS_POSE_LIMIT`) | (keepout으로 대체) | 🟠 **구현·배선 완료**(`clip_safety_box`), **단 기본 config에서는 비활성** | 구현 `ur7e_env.py::UR7eEnv._build_safety_box`(≈`:272`), 클립 `::_clip_xyz_euler`(≈`:357`)·`::clip_safety_box`(≈`:393`). *(줄 번호는 동시 편집으로 밀린다 — 심볼 이름으로 찾을 것.)* `DefaultUR7eEnvConfig`의 `ABS_POSE_LIMIT_LOW/HIGH`가 **영벡터**라 `_safety_box_active=False` 로 떨어진다 — **의도된 refuse-don't-clamp**(0 부피 박스로 클램프하면 TCP를 base 원점으로 몰아 팔을 자기 베이스에 박는다). `run_real_hil.py`가 그 config를 쓰므로 실기에서 박스가 발동한 적이 없다. 측정 박스는 `cube_in_cup.py`에만 있다. 🔴 **(2026-07-30 오후) 이 칸의 위험도가 올라갔다** — `edbb3f5`가 개입 예산을 없애면서 **박스가 개입 경로의 유일한 위치 상한**이 됐다(governor는 속도만 막는다). 박스가 꺼진 config로 개입을 arm하면 위치 상한이 아예 없다 → 아래 「개입 경로에서 governor가 유일한 속도 상한」 절 |

> 🪤 **이 줄에 대한 낡은 포인터 주의.** `../docs/testing/08_OPEN_GAPS.md`(G1 및 부록)는
> 이 항목을 **`serl_ur_infra/README.md:59` 의 "❌ config만 존재, 미작동"** 으로 인용한다.
> 그 서술은 이미 고쳐졌고 줄 번호도 옮겨졌다(현재 이 표). `❌ 미작동` 이라고 적힌
> README 줄을 찾으려 하지 마라 — 없다. 다만 **"반쯤 맞다"는 지적 자체는 유효하다**:
> 코드는 있는데 기본 config에서 실제로 꺼져 있다는 것이 혼란의 근원이었고, 위 칸이
> 그 둘을 분리해 적은 것이다.

**교체 계획**: `eef_delta`의 step()은 "리더→T_des 앵커 매핑(전반부)" +
"거버너→IK→게이트(후반부)"로 나뉜다. 후반부를 `step_task_target(T_des)` 같은
진입점으로 분리 리팩토링하면 — (a) 정책 경로는 base-frame 델타로 T_des를 만들어
후반부만 호출, (b) GELLO 개입은 전반부+후반부 그대로, (c) 텔레옵 브리지도 무변경 —
세 경로가 게이트 스택 하나를 공유하게 된다. `eef_delta`는 rclpy 없는 순수 numpy라
이 리팩토링은 ROS 없는 dev 머신에서 테스트 가능.

## 남은 TODO

- [x] ~~`go_to_reset()`~~ — RESET_JOINTS 고정 init pose로 업샘플러 스트리밍 추격 (원거리 거부)
- [x] ~~관측 경로~~ — 카메라(compressed 토픽), robot state(driver/fk 선택), F/T, tcp_vel
- [x] ~~250Hz 업샘플러~~ — slew-limited, EMA 없음
- [x] ~~fake backend 테스트~~ — `tests/test_env_fake_backend.py`
- [ ] `policy_delta_controller` → `eef_delta` 후반부 재사용 리팩토링 (위 표 참고) — **DRY_RUN 해제 전 필수**
- [ ] 워크스페이스 박스 클램프를 컨트롤러 게이트에 통합
- [ ] HOLD/reject_reason을 step info로 노출 + staleness safe-stop 정책 통일 + UR fault recovery
- [x] v_max vs ACTION_SCALE 정합 — 2026-07-28 해결. 3층을 균일 1.25배로 맞춰 헤드룸 1.20x 유지
      (`ACTION_SCALE` 0.0125/0.0625, `GOVERNOR` 0.15/0.75/0.0625, `UPSAMPLER` 0.0025)
- [ ] 데드맨 하드웨어 (풋스위치) + 성공/실패 라벨링 키
      — *(이전 판 문구 "현재 스페이스바"는 낡았다: 두 entrypoint 모두 `--deadman` 기본값이
      `topic`(GUI)이다 — `tests/run_real_hil.py`·`scripts/run_remote_rlpd_actor.py`의
      `--deadman` 인자, G12. 스페이스바는 명시 옵션이고 `04` §1.1이 위험을 설명한다.)*
- [x] **개입 손맛 2차 — 2026-07-30 오후, 추종을 RL 창 밖으로** (`edbb3f5` + `d6965a9`).
      `follow_mode="background"`(기본)에서 데몬 스레드가 30 Hz로 리더를 따라가고 `env.step`은
      관찰자다. 개입 최고속 2.4 → 최대 12.5 cm/s(계산값), 창 밖 HOLD 27.7 % → 없음,
      slew 천장 46 % → 100 %, 데드맨 release 512 → 33 ms — **전부 설계/계산값이고 실기
      계측이 아니다.** 실기 확인은 조작자 보고("개입 속도는 좀 고쳐졌어")뿐이다.
      테스트 **701 / 11 / 1**(actor venv, numpy 2.2.6) · **741 / 4 / 1**(`hilserl`).
      대가는 「저장 액션 불변식」의 개입 경로 예외 → 그 절.
- [ ] 🔴 **`suspend_follower()` 호출부 배선** — `ur_env/remote_actor.py`의 `WAIT_SCENE_READY` /
      `WAIT_HOME_APPROVAL`은 RL 스레드를 무한 블록하고 **후자는 데드맨을 보지 않는다.**
      배선 전에는 그 화면에서 GELLO를 잡으면 팔이 따라온다. context manager는 이미 있다.
      **다음 armed 세션의 선행 조건**으로 취급할 것.
- [ ] 🔴 **포화 transition의 서버측 제외** — proto 신규 필드 + `SCHEMA_VERSION` bump + 양끝
      동시 업그레이드. protobuf가 unknown field를 조용히 버리므로 **반쪽 업그레이드는 무증상
      오염**이다. **먼저 포화 비율을 실기에서 재고**, 그 숫자로 이것/G21 중 무엇을 할지 정한다.
- [ ] 🟠 **포화 계측을 actor 요약/GUI에 배선** — `intervention_saturation` /
      `intervention_saturated` / `intervention_follow_ticks`는 `info`에 있지만 조작자도 서버도
      영구 로그도 못 본다. 지금은 `run_real_hil.py` CSV에만 있다.
- [ ] 🟠 **`run_real_hil.py --arm`이 production보다 덜 안전해졌다** — 그 러너는
      `DefaultUR7eEnvConfig`라 박스가 꺼져 있고(G1), 예산도 없으니 **위치 상한이 없다.**
      측정 박스를 쓰는 config를 주거나 G1을 닫는다.
- [x] **개입 손맛 1차(빳빳함) — 2026-07-30 오전 해결, 실기 PASS** (`4197f5b`). 원인은 10 Hz가 아니라
      창당 타깃 1회 갱신. 30 Hz 재샘플링 + One-Euro 이식 + `InterventionBudget`.
      DRY RUN 300스텝(개입 272)·ARMED 150스텝(개입 120) 둘 다 전체 PASS,
      dp_ratio 중앙값 1.000 / held 0% / `substeps=2`, 조작자 확인 양호.
      (같은 날 세 번째 run — DRY `--scale 0.5`, 개입 144 — 은 고친 판정으로 **SKIP**이다:
      비포화 58표본이 개수 게이트 20은 넘지만 축 여기가 1.4/3.6/1.4 cm로 2 cm 게이트 미달.
      `governed`는 3 run 전부 0이었으나 **첫 타깃만 집계한 값**이다 → `04` §9.9.)
      금지사항 3개(가속도 제한·`target_stale_s`·`soft_start_s`)는 `../docs/testing/04_HIL_INTERVENTION.md` §9.3.
      📌 **이 판정은 이제 `follow_mode="in_window"` 경로의 판정이다** — 같은 날 오후
      `edbb3f5`가 기본값을 `background`로 바꿨다(바로 위 항목).
- [ ] 🟠 **대각 이동의 governor 상시 절삭** — `ACTION_SCALE` 헤드룸이 축별로만 성립해서
      2축 0.849배 / 3축 0.693배로 이미 잘리고 있었다. `4197f5b`가 만든 게 아니고
      **관측 수단(`governed_scale`)만 생겼다.** 저장 액션 과대기록 경로다 → 위 "저장 액션 불변식" 절.
- [x] ~~개입 예산 소진 후 HOLD (창이 늘어질 때) — **코드로 못 고친다, G21 종속**~~
      — 2026-07-30 오후 `edbb3f5`가 **예산 자체를 개입 경로에서 제거**해 해소. 단
      **G21이 사라진 게 아니라 비용의 형태가 바뀌었다**: HOLD 대신 **포화(저장 액션
      과소보고)** 로 나타난다. `follow_mode="in_window"`에서는 이 항목이 그대로 유효하다.
- [ ] `GelloIntervention._leader_T`에 TCP_OFFSET 배선 (config는 있음, 현재 플랜지 기준)
- [ ] 로봇 노트북에서 mock 하드웨어 검증 (QoS `VERIFY(hw)` 주석 참고, 그리퍼 방향 육안 확인)
- [ ] per-task config 예제 (`examples/experiments/<task>/config.py` 형식)
- [x] **★ Kanu bounded synthetic learning acceptance — final schema v2**
      — unified `248255f`, 실제 SSH alias `kanu`, JAX/JAXLIB 0.5.3 GPU actual classifier/agent, laptop3 SSH tunnel에서 exact 100 transition → step 1/gradient 2/policy 1/checkpoint full-load roundtrip을 통과했다. fresh process resume가 1/2/1과 policy version 1 finite 7D action을 serving했다. fingerprint는 `fa1985378ad2729f466783e4f112d54022e14090430374a6531e4fb715440fcd`다.
      🗄️ **위 fingerprint·커밋·호스트는 kanu 실측이라 그대로 둔다.**
- [x] **★ `junhyeong_ai` bounded synthetic acceptance — 2026-07-31 재실행 PASS**
      — 같은 도구로 새 서버에서 fresh + resume 2 run, transition 200개, 전 metric finite,
      checkpoint round-trip verified. jax 0.5.3이 sm_120에서 **네이티브**로 돌아 핀 bump가
      필요 없었다. 실측 전문은 [SERVER_MIGRATION_E2E_JUNHYEONG_AI.md](SERVER_MIGRATION_E2E_JUNHYEONG_AI.md).
- [ ] **production robot/continuous acceptance (이제 `junhyeong_ai`)**
      — 남은 범위는 기본 50-step publish/5,000-step checkpoint, 장시간 memory/contention,
      **장시간 연속 robot E2E**다. real canonical demo(2,037개)는 이미 이 서버에 있고
      2026-07-31 실기 세션이 **PASS**했다(replay 316 / intervention 210 — 상태 문서 §3D).
      **checkpoint는 여전히 0개다**: kanu의 run root 8개 전부 빈 `checkpoints/`였고
      (`checkpoint_period` 5,000 vs 최고 도달 step 301) 새 서버는 clean lineage로 시작한다.
      정확한 명령과 feature RAM gate는 `HIL_SERL_KANU_RUNBOOK_KO.md`를 따르되,
      **호스트·GPU index·경로는 `DATA_AND_MODELS_JUNHYEONG_AI_KO.md`가 우선**한다
      (⚠️ 새 호스트는 GPU **1장**, 여유 RAM 약 **37 GB**다 — kanu의 8장/140 GB가 아니다).
- [x] **reward classifier 입력 정합 (G15)** — 2026-07-29 해결. **재학습이 아니라 sidecar 분리다.**
      `IMAGE_CROP`은 불변, 관측 schema hash도 불변(`3459098d…`), proto 무변경.
- [x] **orbax 디렉터리 checkpoint pin (G19)** — 2026-07-29 해결. `checkpoint_sha256()`이
      재귀 `directory_sha256()`에 위임하고, 기본 SHA 2개가 `512b6575…`(`checkpoint_150`, 14파일)로 교체됐다.
      단일 파일 digest는 예전과 동일해 기존 pin이 살아 있다.
- [ ] 🚧 **`CLASSIFIER_SIDECAR["stationary_speed_max"]` 실측** — 현재 `0.05 m/s`는
      **코드가 스스로 `PLACEHOLDER`라고 표시한 값**이다(`ur_experiments/cube_in_cup.py`).
      의도는 "팔이 가라앉았다"이고, 옳은 값은 릴리스 후 TCP가 실제로 머무는 속도다 —
      녹화 take에서 잰다. 낮으면 게이트가 안 열리고 높으면 모션 블러를 분류한다.
- [ ] **sidecar 경로 실기 첫 투입** — 위 두 개는 코드·자동 테스트 수준까지만 검증됐다.
- [ ] 🔴 **팔 가림(occlusion)에 강한 카메라 배치** — sidecar가 **못 고치는** 항목이다.
      `take_21`은 @0.85 recall 0.0%, @0.05로 내려도 57.9%이고 팔이 cam1을 쓸 때 확률이 `0.005↔1.0`으로
      진동한다(출처: `REWARD_CLASSIFIER_THRESHOLD_KO.md`). 원인은 시야/가림이지 전처리도 라벨도 아니다.
- [ ] **최초 1회 checkpoint 계보 단절 처리** — reward 계약(`input_contract`,
      `success_confirmations`, classifier SHA, `reward_model_id`)이 learner fingerprint에 들어가
      옛 checkpoint resume이 fail-closed로 거부된다. **의도된 1회성**이며 새 `--checkpoint-root`로
      한 번 시작하면 끝난다(옛 계보는 recall 0% 체크포인트가 매긴 reward로 학습됐다).

## RViz fake RL 테스트 (로봇 노트북, 실기 리스크 0)

mock ros2_control에 대고 진짜 env·backend·업샘플러를 돌려 RViz로 확인한다.
자세한 커맨드는 `tests/run_rviz_fake_rl.py` docstring 참고. 요약:

```
T1: ros2 launch gello_policy ur_control_fake_safe.launch.py ... use_mock_hardware:=true launch_rviz:=true
T2: ros2 run gello_policy fake_diffusion_observation_node     # 가짜 카메라
T3: python3 tests/run_rviz_fake_rl.py                          # 사인 궤적 / --random
```

확인 항목: RViz에서 부드러운 원 궤적(점프·덜컹 없음 = 업샘플러 동작), reset 시
home 복귀, `VERIFY(hw)` QoS 주석 항목(`ros2 topic hz`로 각 토픽 수신 확인),
터미널의 tcp 로그가 RViz 자세와 일치. 이 스크립트의 `RESET_MAX_DIST_RAD=7.0`은
mock 전용 완화값이므로 실기 config에 복사 금지.

## 저장 액션 불변식 (buffer correctness)

GELLO 개입이든 정책이든, buffer에 저장되는 액션은 **실행된 액션과 동일한
[-1,1] 값**이어야 한다. `GelloIntervention`은 앵커 델타를 `÷ACTION_SCALE →
clip`으로 만들어 실행과 저장에 같은 값을 쓰므로 구조적으로 보장된다. 단
`ACTION_SCALE*HZ`가 거버너 캡을 넘으면 실행만 잘리고 저장은 그대로라 이
불변식이 깨진다 — 그래서 거버너 기본값은 스케일 최대의 ~120%로 잡았고
(순수 안전망), env 기동 시 위반하면 WARNING을 찍는다.

### 🛑 (2026-07-30 오후, `edbb3f5`) **개입 경로의 명시적 예외** — 불변식은 유효하고, 이 경로만 예외다

**불변식을 지우지 마라.** 정책 경로에서는 그대로 요구된다. 다만
`follow_mode="background"`(현재 기본값)의 **개입 경로에서는 의도적으로 깨진다.**

**무엇이 깨지나.** 추종 스레드에 예산이 없으므로 창이 길면 사람이 `ACTION_SCALE` 여러
스텝을 실제로 이동하는데, 기록되는 액션은 창 순변위를 `÷ACTION_SCALE` 한 뒤 `[-1,1]`로
축소한 값이라 **최대 `1.0`밖에 말할 수 없다.** 즉 transition이 실제 움직임을
**체계적으로 과소보고**하고, learner는 "액션 1.0이 한 `ACTION_SCALE` 스텝을 움직였다"고
읽어 **critic이 동역학을 낙관 학습**한다. 방향은 정책 경로의 알려진 **과대**기록(대각 절삭,
G26)과 정확히 반대이며 원인은 다르다.

**왜 그럼에도 채택했나 (2026-07-30 조작자 결정).** 예산을 두면 개입 최고속이
`ACTION_SCALE / T` = 12.5 mm / 512 ms = **2.4 cm/s**다(production actor 실측 스텝 주기
512 ms 기준). 그 속도로는 **아무도 팔을 못 몬다** → 시연 데이터가 **0개**다.
**편향된 표본이 빈 표본보다 낫다.** 조작자 요구 그대로: *"intervention 시는 그냥 teleop이
서버 통신 시간과 상관없이 쭉 되는 거고, 정보만 그때그때 주는 것."*

**대가를 가정하지 않고 잰다.** `UR7eEnv._harvest_follow_window`가 남긴다:

| 키 | 뜻 |
| --- | --- |
| `info["intervention_saturation"]` | **클립 전** 비율. `4.94`면 이 transition이 실제 이동을 **4.94배 과소보고**한다 |
| `info["intervention_saturated"]` | 위 값 > 1.0 |
| `info["intervention_follow_ticks"]` | 이 창에서 추종이 실제로 발행한 타깃 수. **0이면 추종이 안 돈 것** |

`tests/run_real_hil.py`가 CSV 컬럼 `saturation` / `saturated` / `follow_ticks`와 요약의
"포화 창 N/M (x %)" 줄로 뽑는다. ⚠️ **wire로는 보내지 않는다** — 서버는 이 값을 모르므로
**learner 로그만으로는 사후 판별이 불가능하다.**

**복구 경로는 둘, 그리고 순서가 있다.**

1. **포화 transition을 서버측에서 제외.** proto **신규 필드 + `SCHEMA_VERSION` bump**가
   필요하고 **양끝을 같이 올려야 한다.** 🛑 protobuf가 unknown field를 **조용히 버리므로
   반쪽 업그레이드는 무증상 오염**이다 — 안 올린 쪽 learner가 과소보고된 액션으로 학습하는데
   어떤 경고도 나지 않는다(호스트가 kanu에서 `junhyeong_ai`로 바뀌어도 이 위험은 그대로다).
2. **창 주기 자체를 줄인다** — `../docs/testing/08_OPEN_GAPS.md` **G21**
   (= 상태 문서 §8 **P1**). 포화는 창 길이에 비례하므로 이쪽이 근본이다.

**먼저 할 일은 둘 다 아니고 "재는 것"이다.** 포화 비율이 1과 2 중 무엇을 할지의 근거이고,
그 숫자 없이 proto를 건드리지 않는다. 근거 전문은
`UR7eEnv._harvest_follow_window` docstring과
[HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) §3C.6.

### 🛑 norm 비례 축소 — 개입 액션에 **축별 `np.clip`을 쓰지 마라** (근거가 둘이다)

개입 액션을 `[-1,1]`에 맞출 때는 **각 3-벡터를 자기 norm으로 나눠 비례 축소**한다. 축별
`np.clip`은 **금지**다. 이 규칙은 `GelloIntervention._expert_delta_xi`에 이미 있었고
("PROPORTIONAL (norm) clamp, NOT the old per-axis `np.clip`"),
`UR7eEnv._harvest_follow_window`에도 같은 규칙으로 들어갔다.

| # | 근거 | 어디서 나왔나 |
| --- | --- | --- |
| (a) | **방향 왜곡.** 포화된 대각의 축별 clip은 경로를 꺾는다(`[2.0, 0.5] → [1.0, 0.5]`). 조작자는 이걸 "엉뚱한 방향/배율로 반응한다"로 느낀다 | `_expert_delta_xi`의 원래 근거 |
| (b) | 🔴 **`RelativeFrame`이 이 벡터를 회전시킨다.** `transform_action_inv`가 `blockdiag(R, R)`을 곱하는데, **회전은 각 3-벡터의 norm은 보존하지만 축별 최댓값은 보존하지 않는다.** 축별 clip은 norm을 `sqrt(3) = 1.73`까지 남기므로 `[1.0, 1.0, 0]`이 회전 뒤 `[1.41, 0, 0]`으로 나와 액션 공간을 벗어난다 | **2026-07-30 실기에서 actor가 즉사했다** (`d6965a9`) |

(b)의 실제 예외:

```text
ActorProtocolError: executed_action must be within [-1, 1]
  remote_actor.build_data -> actor_network.validate_action
```

**왜 norm 축소가 옳은가:** `norm <= 1`인 벡터는 **어떤 회전을 해도** 모든 성분이 `[-1,1]`
안에 있다. 즉 기록 액션이 "측정된 프레임에서만"이 아니라 **모든 프레임에서** 합법이 된다.

📌 **옛 예산 경로가 이 버그를 안 겪은 이유도 같다** — `InterventionBudget`의 clamp가 norm
기준이라 프레임 변환을 공짜로 통과했다. **예산을 없애면서 그 성질이 같이 사라졌고**, 그래서
같은 실수가 새 경로에서 재발했다. 포화 보고도 축별 최댓값이 아니라 `max(n_p, n_w)`로 맞췄다.
회귀 테스트는 포화된 대각을 harvest한 뒤 **네 각도로 회전시켜** 모든 성분을 확인한다 —
축별 clip으로 되돌리면 **FAIL한다(확인함)**.

### 🗄️ (2026-07-30 오전, `4197f5b`) `in_window` 경로에는 `InterventionBudget`이 그 불변식을 지킨다

> **이 절은 이제 `follow_mode="in_window"`에만 적용된다.** 기본 `background` 경로에는 예산이
> 없다 — 바로 위 「개입 경로의 명시적 예외」가 그 경로의 정본이다. 아래를 지우지 않는 이유는
> `in_window`가 아직 선택 가능하고, 무엇보다 **왜 예산이 존재했는지**를 여기서만 읽을 수
> 있기 때문이다.

개입 중 리더를 **창 안에서 30 Hz로 재샘플링**하게 되면서(손맛 근거는
[`../docs/testing/04_HIL_INTERVENTION.md`](../docs/testing/04_HIL_INTERVENTION.md) §9),
"창당 한 번 앵커 델타 → clip"만으로는 이 불변식이 **보장되지 않는다.** 이유는 창의 실제 길이다:

- 명목 창은 100 ms지만 gRPC 왕복이 `env.step` **밖**에 있어 실효 스텝 주기가 늘어난다
  (`04` §9.1). 숫자는 출처를 구분해서 인용할 것:
  - **101 ms** — 2026-07-30 실기 3 run의 실측 스텝 주기 중앙값. **단 이 run들은
    `run_real_hil.py`이고 gRPC가 없다** (`tests/run_real_hil.py` docstring).
  - **112~122 ms / 158 ms / 197 ms** — actor 경로의 *산출* 주기. 07-29 빠른 링크
    (RPC p50 12~22 ms)면 앞의 값, **열화 링크**(13 Mbit/s, RPC p50 58 ms / p99 97 ms)면
    뒤의 두 값이다 (`04` §9.1 표 1 + G16).
  - **0.700 s** — 오프라인 시나리오 행. 2026-07-29 관측된 첫 policy publish **5.474 s**와
    learner 경합 중 **learner step** 중앙값 약 **1.12 s**에서 유도한 것이며,
    **1.12 s는 env 스텝 주기의 실측이 아니다** (`04` §9.5 / `08` G21).
- 사람이 그 창을 **무제한 추종**하면 1 s 창에서 `ACTION_SCALE` 여러 스텝 분량을 이동하는데,
  기록되는 액션은 `[-1,1]`에서 클립되므로 최대 `1.0`밖에 못 말한다. 즉 transition이 실제로
  일어난 움직임을 **체계적으로 과소기록**한다 (정책 경로의 과대기록과 부호만 반대인 같은 결함).

그래서 `ur_env/envs/leader_stream.py::InterventionBudget`이 **창당 총 변위를 정확히
`ACTION_SCALE` 1스텝**(cube_in_cup 기준 0.0125 m / 0.0625 rad)으로 묶고, 그 창에서 실제로
소비된 양을 `consumed_action()`으로 되돌려 buffer에 넣는다. 즉 **예산이 불변식을 지키는
장치**다. 회계는 net 변위가 아니라 **경로 길이**(서브스텝 크기의 합)로 하므로 `exhausted`가
래치되고(되돌아와도 False로 돌아가지 않는다), net norm ≤ 경로 길이이므로 보고되는 액션도
같이 묶인다. 예산 소진 후 남은 시간은 HOLD이고 그건 코드가 아니라
**G21(RPC 지연) 비용**이다 → `../docs/testing/08_OPEN_GAPS.md` G21.

### 🆕 (2026-07-30 오후, `edbb3f5`) 개입 경로에서 **governor가 유일한 속도 상한**이 됐다

예산이 개입 제어 경로에서 빠지면서, `follow_mode="background"`의 개입 중 팔을 묶는 것은
**정확히 다음 셋뿐이다** (`GelloIntervention.follow_xi` docstring이 정본):

| 층 | 무엇을 막나 | 값 |
| --- | --- | --- |
| **GOVERNOR** (틱당, `controller.step` 안) | **속도만** | `v_max` 0.15 m/s · `w_max` 0.75 rad/s를 `dt = 1/substep_hz`로 스케일 + `dq_step_max` 조인트 게이트와 line search |
| **워크스페이스 박스** (`clip_pose`, `controller.step` 안) | **위치** | 결과를 적분기에 되써서 windup이 아니라 **하드 벽** |
| **250 Hz 업샘플러** | 가속 | slew 제한 |

이 절의 두 문장이 핵심이다.

- 🛑 **개입 경로에서 `GOVERNOR`가 유일한 속도 상한이다.** 예전에는 예산(창당 0.0125 m)이
  governor 캡(0.0150 m)보다 타이트해서 **예산이 먼저** 묶었다. 지금은 그 위층이 없다.
- 🛑 **박스가 유일한 위치 상한이다 — governor는 속도만 막지 거리를 막지 않는다.** 참고
  크기: `v_max` 0.15 m/s만으로는 5초 gRPC stall에서 약 **75 cm**의 자유 주행이 허용되고,
  이는 측정된 박스 x 범위의 약 **2.8배**다.

따라서 **박스가 꺼진 config로 개입을 arm하면 위치 상한이 아예 없다.** 그게 정확히
`tests/run_real_hil.py`의 상태다(`DefaultUR7eEnvConfig` → `ABS_POSE_LIMIT_*`이 영벡터 →
`_safety_box_active=False`, **G1**). 측정 박스는 `ur_experiments/cube_in_cup.py`에만 있다.
→ 위의 `PolicyDeltaController` 현황 표 「워크스페이스 박스」 칸.

세 층이 **전부 `PolicyDeltaController.step` 아래**에 있다는 것이 왜 중요한가: 추종 스레드가
관절 명령을 다른 경로로 계산하면 **상한 셋이 통째로 사라진다.** 그래서
`UR7eEnv._emit_arm_command`가 유일한 출구다.

### 속도 3층과 개입 예산 — 예산은 4번째 층이 **아니다** (`in_window` 경로)

> **아래는 `follow_mode="in_window"` 기준이다.** 기본 `background`에는 예산이 없다(바로 위 절).
> 3층 비율 계약 자체는 두 경로 공통이다.

3층(`ACTION_SCALE` / `GOVERNOR` / `UPSAMPLER` — `ur_env/envs/config.py`의 세 블록, 각 상단
주석이 근거를 담고 있다)은 2026-07-28에
균일 1.25배로 정렬해 헤드룸 1.20x를 유지한다. 개입 예산은 그 위에 얹은 **새 층이 아니라
`ACTION_SCALE`에서 직접 파생된 같은 수**다(`InterventionBudget.__init__`가 `action_scale`을
그대로 받는다). 그래서:

- 예산은 3층 비율을 **바꾸지 않는다.** 다만 개입 창에서는 실효 상한이 governor가 아니라
  **예산**이 된다 — 0.0125 m(예산) < 0.0150 m(`v_max`/HZ)이므로 예산이 먼저 묶는다.
  2026-07-30 실기 **3 run**(개입 536표본)에서 `governed=0`이었다.
  🛑 **이것을 "governor 절삭이 없음을 확인했다"로 읽지 말 것 — 개입 경로에서는 절삭이
  구조적으로 불가능하다.** `_paced_request`가 요청을 `ACTION_SCALE/3 = 0.00417 m`로 깎고
  서브스텝 governor cap은 `v_max/substep_hz = 0.0050 m`이므로 **요청이 cap에 절대 닿지
  않는다.** `governed=0`은 관측 결과가 아니라 **산술의 필연**이고 **재측정해도 0이다**
  → `../docs/testing/08_OPEN_GAPS.md` **G26**. 실기 관측을 가능하게 하려면 `GOVERNOR`를
  `ACTION_SCALE` 대비 조여야 한다.
  *(별개 사실: 그 3 run은 창의 **첫 타깃만** 집계하던 코드로 측정됐고, 그 뒤 `governed`는 창
  전체 **OR** / `governed_scale`은 창 전체 **최솟값**으로 고쳐졌다 — `tests/run_real_hil.py`
  docstring (e). 신호로서는 개선이지만 위 부등식 때문에 값은 안 바뀐다.)*
- **예산을 끄거나 완화해도 3층 비율 계약 자체는 깨지지 않는다.** 커밋된 코드는 창의 **첫
  타깃도 `dt = 1/substep_hz`로** 과금하므로(`ur7e_env.py::UR7eEnv._apply_action`) 창 총
  governor 허용량은 `3 × v_max/30 = 0.0150 m` = `v_max/HZ`와 정확히 같다.
  > **이전 판(보존):** *"예산을 끄거나 완화하면 서브스텝이 만든 governor 창 총량(첫 타깃
  > `1/HZ` + 서브스텝 `1/30` ×2 = 0.0250 m, **1.67배**)이 그대로 드러나 헤드룸 계약이 개입
  > 경로에서 깨진다."* — 이 관찰은 **커밋 전 워킹 트리** 기준이었고 `4197f5b`가 해소했다.
  > 📌 2026-07-30 커밋된 코드에서 직접 확인. 전모는 `../docs/testing/08_OPEN_GAPS.md` G24의
  > "✅ 정정" 절.
- `ACTION_SCALE`을 올리면 이 여유(0.0125 → 0.0150)가 **먼저** 소진된다 → G18.

**`governed` / `governed_scale`로 governor 절삭이 처음 관측 가능해졌다**
(`ur_env/envs/policy_delta_controller.py::PolicyDeltaController.step`이 반환하는 `info`.
*줄 번호는 적지 않는다 — 이 파일은 동시 편집으로 계속 밀린다*). `run_real_hil.py` CSV에도
`substeps`와 함께 컬럼으로 나온다. 이전에는 절삭이 일어나도 어디에도 안 남았다.
🛑 **계측 범위가 07-30 실기 이후 바뀌었다:** 지금 `governed`는 창 전체 **OR**,
`governed_scale`은 창 전체 **최솟값**이고, `ur7e_env.py::UR7eEnv.step`의 기본 `info`가 두 키를
**항상** 담는다(이전에는 `held` 창에만 없어서 스키마가 분기에 따라 달라졌다). 07-30 3 run은
**첫 타깃만** 집계하던 코드로 측정됐으므로 옛 CSV와 새 CSV의 같은 이름 컬럼을 **비교하면
안 된다.**

🔴 **`governed`가 못 보는 것 — 저장 액션 불변식의 실제 구멍은 여기다.**
`InterventionBudget.take`는 **요청**을 과금하는데, 예산 **아래**에 게이트가 하나 더 있다:
`PolicyDeltaController`의 IK **line search**. 그것이 물리면 명령은 더 깎이는데 기록은 요청
그대로이고, **`governed`는 governor가 아니라 joint gate가 물렸으므로 `False`로 남는다** —
관측 수단이 아예 없다. 📌 실측 과대 진술: `dq_step_max` 0.0625(shipped) 1.000x /
0.01 **5.66x** / 0.002 **28.4x**. `strict=True` xfail로 못 박혀 있다
(`tests/test_intervention_substeps.py::test_a_line_search_shrink_must_not_be_left_out_of_the_recorded_action`)
→ `../docs/testing/08_OPEN_GAPS.md` **G27**.
반대 방향 오염도 있다: 리더가 창 중간에 죽은 창은 기록이 `zeros(7)`인데 첫 타깃은 이미
**0.004167 m**(정규화 0.333)를 명령했다 → **G28**. 둘 다 `4197f5b`가 만든 것이 아니다.

> ### 🐛 기존 결함이 이때 드러났다 — `ACTION_SCALE` 헤드룸은 **축별로만** 성립한다
> 헤드룸 1.20x는 **한 축 기준** 계산이다(`0.0150 / 0.0125`). 액션이 여러 축에 동시에 걸리면
> 요청 변위의 크기가 `√n`배로 커지는데 governor 캡은 벡터 norm에 걸리므로:
>
> | 전 스케일 액션 | 요청 변위 | governor 캡(`v_max`/HZ) | `governed_scale` |
> | --- | --- | --- | --- |
> | 1축 | 0.0125 m | 0.0150 m | 1.000 (절삭 없음) |
> | 2축 대각 | 0.0177 m | 0.0150 m | **0.849** |
> | 3축 대각 | 0.0217 m | 0.0150 m | **0.693** |
>
> **즉 대각 이동은 이미 상시 절삭되고 있었다.** 이것은 `4197f5b`가 만든 문제가 **아니고**
> (3층 정렬 이후 계속 그랬다) 이번 변경이 만든 것은 **관측 수단**뿐이다. 저장 액션 불변식의
> 관점에서는 정책 경로의 알려진 과대기록 경로이며, 아직 **고쳐지지 않았다** → G2/G18.

## 설치 (예정)

```bash
conda activate hilserl
pip install -e third_party/hil-serl/serl_launcher
pip install -e serl_ur_infra
source ros2_ur_ws/install/setup.bash    # ur_gello_bringup (ur_kin, eef_delta)
# 카메라는 별도 터미널에서: ./ros2_ur_ws/launch_cameras.sh
```
