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
> production actor 배선은 첫 실물 E2E에서 사용됐다. 다만 GUI/영구 JSONL에 per-transition
> probability가 없어 online verdict 정합은 아직 미확인이다. **팔 가림 병리(`take_21`)도
> 안 고쳐졌다**; 정지 게이트가 완화할 뿐 진짜 해법은 카메라 배치다.

최종 운영 checkout은 `/home/laptop3/gello_software`, branch는 `feat/gello-ur7e-humble-22.04` 하나다.
2026-07-29 머지 `3f199d4`로 로봇/하드웨어 작업이 이 브랜치에 들어왔다 — **워크트리 분리 시절 서술은
전부 낡았다.** 통합 상태·checkpoint·Kanu 검증·frozen-trunk feature replay·bounded fake-data
learning E2E 계약은 [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)를 기준으로 한다.

`scripts/run_fake_e2e_actor.py`는 robot를 제어하는 actor가 아니라 `--synthetic-e2e` Kanu learner에 canonical raw fake observation 100개를 보내 gRPC→classifier→feature replay→CTA→publish→checkpoint→fresh-process resume를 검증하는 acceptance tool이다. server는 exact actor/run ID, exact 100 inserts, bounded timeout을 강제하고 synthetic-only model ID를 advertise한다. cleanup 후 full checkpoint roundtrip/trunk invariant까지 통과해야 pass한다. synthetic checkpoint는 fingerprint/model scope가 다르므로 production robot lineage에 사용할 수 없다.

## 이 디렉터리의 문서 (`serl_ur_infra/*.md`)

### 지금 유효 — 지침으로 읽어도 되는 것

| 문서 | 무엇인가 |
| --- | --- |
| [HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md](HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) | **현재 진입점.** 첫 실물 E2E 증거, PASS/미완료 경계, 3-CLI와 다음 우선순위 |
| [HANDOFF_NEXT_SESSION_KO.md](HANDOFF_NEXT_SESSION_KO.md) | 첫 E2E 이전의 하드웨어/classifier 상세 조사 기록. 최신 상태 지침으로 쓰지 않는다 |
| [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 상태 기록. learner 구현 §1–10 / actor·하드웨어 §11 / reward classifier 조사 §12. 위 문서보다 깊다 |
| [HIL_SERL_KANU_RUNBOOK_KO.md](HIL_SERL_KANU_RUNBOOK_KO.md) | Kanu에서 learner를 띄우는 절차 (dry-run → bounded run) |
| [REWARD_CLASSIFIER_THRESHOLD_KO.md](REWARD_CLASSIFIER_THRESHOLD_KO.md) | reward threshold를 0.85 → 0.2로 내린 근거 + 2026-07-29 누출 감사. 07-28 수치와 07-29 수치를 구별해 인용할 것 |
| [REMOTE_ACTOR_GRPC.md](REMOTE_ACTOR_GRPC.md) | gRPC 액터 전송 계약 v2 — 같은 전송을 쓰는 **서버 entrypoint 3종의 차이**. 영문, 머지 트리 기준 재검증 |
| [RVIZ_HIL_TEST_CLI.md](RVIZ_HIL_TEST_CLI.md) | mock(`use_fake_hardware`) 4터미널 개입 테스트 절차. 실기 위험 0 |
| [REWARD_CLASSIFIER_LIVE_KO.md](REWARD_CLASSIFIER_LIVE_KO.md) | 라이브 reward classifier 뷰어 런북 (랩톱 CPU, 터미널 4개). **2026-07-29 실기 검증됨.** 인터프리터 함정 · 조용한 실패 · 트러블슈팅 |
| [REWARD_TO_RL_INTEGRATION_KO.md](REWARD_TO_RL_INTEGRATION_KO.md) | **분류기를 개입·학습에 연결하는 사람이 읽을 것.** reward/termination 계약(서버 권위, `next_observations` 기준) · 개입의 **버퍼 이중 기록** · RLPD 50:50 배치 · 연결 순서 6단계 · 감시 지표. 코드에서 직접 추적해 작성 |
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
| `ur_env/envs/ur7e_env.py` | `franka_env/envs/franka_env.py` | gym env 본체 (step/reset, 관측, 카메라, 그리퍼) |
| `ur_env/envs/config.py` | `DefaultEnvConfig` | 태스크별 config 베이스 |
| `ur_env/envs/ros_backend.py` | Flask 로봇 서버 (HTTP) | rclpy 백그라운드 노드 — 토픽 I/O |
| `ur_env/envs/policy_delta_controller.py` | (Franka 임피던스 컨트롤러가 하던 일) | 정책 델타 → 거버너 → IK → 게이트 → 조인트 명령 |
| `ur_env/envs/wrappers.py` | `SpacemouseIntervention` | `GelloIntervention` — 데드맨 + 앵커 클러치 개입 + 창 안 30 Hz 서브스텝 드라이버 |
| `ur_env/envs/leader_stream.py` 🆕 | (upstream에 대응물 **없다** — `SpacemouseIntervention`에는 입력 저역통과나 창 예산이 없고, `filtered_expert_a`는 축 마스킹일 뿐이다: `franka_env/envs/wrappers.py:207-247`) | **개입 손맛 계약** (`4197f5b`). `OneEuro`(`bridge_stages.py:42-106` 비트 동일 이식) · `LeaderFilter`(`note_sample`=리더 cadence / `filtered`=출력 틱, **API가 두 cadence 분리를 강제한다**) · `InterventionBudget`(창당 `ACTION_SCALE` 변위 예산, 경로 길이 회계). 순수 numpy/stdlib — rclpy·gym·config import 없음. **설계 근거가 모듈 docstring에 전부 있다** |
| `ur_env/classifier_sidecar.py` 🆕 | (upstream `classifier_keys` 별도 카메라 등록에 대응) | **reward classifier 전용 무크롭 이미지 계약.** `build_sidecar`(랩톱: full-res BGR → 128×128 → JPEG) · `decode_classifier_frames`(서버: 라이브 뷰어와 같은 레시피) · `validate_sidecar`(구조 검증, numpy만) · `SidecarScheduler`(~2 Hz + 정지 게이트 + 성공 근처 에스컬레이션) · `directory_sha256`(orbax **디렉터리** 체크포인트 핑거프린트, G19) |

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
  - **(2026-07-30, `4197f5b`) 창 안에서 리더를 30 Hz로 재샘플링한다.** 앵커/gain 래치 의미는
    그대로고, 바뀐 것은 **액추에이터 층**이다: 100 ms 창의 sleep이 서브스텝 페이싱으로 바뀌어
    (`ur7e_env.py::_drive_intervention_substeps`) 타깃이 창당 3회 갱신되고, 리더 입력은
    One-Euro를 지나며, 창 총 변위는 `InterventionBudget`이 `ACTION_SCALE`로 묶는다.
    **"1 스텝 = 1 transition"과 10 Hz 저장 주기는 불변이다.** 정책 경로는 `driver is None`으로
    갈라져 예전 그대로 sleep 한 번이다. `config.INTERVENTION["substep_hz"] <= HZ`거나
    `INTERVENTION` 블록이 없는 config는 기능이 꺼지고 변경 이전 동작으로 퇴화한다.
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
| 워크스페이스 박스 (`ABS_POSE_LIMIT`) | (keepout으로 대체) | 🟠 **구현·배선 완료**(`clip_safety_box`), **단 기본 config에서는 비활성** | 구현 `ur7e_env.py::UR7eEnv._build_safety_box`(≈`:272`), 클립 `::_clip_xyz_euler`(≈`:357`)·`::clip_safety_box`(≈`:393`). *(줄 번호는 동시 편집으로 밀린다 — 심볼 이름으로 찾을 것.)* `DefaultUR7eEnvConfig`의 `ABS_POSE_LIMIT_LOW/HIGH`가 **영벡터**라 `_safety_box_active=False` 로 떨어진다 — **의도된 refuse-don't-clamp**(0 부피 박스로 클램프하면 TCP를 base 원점으로 몰아 팔을 자기 베이스에 박는다). `run_real_hil.py`가 그 config를 쓰므로 실기에서 박스가 발동한 적이 없다. 측정 박스는 `cube_in_cup.py`에만 있다 |

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
- [x] **개입 손맛(빳빳함) — 2026-07-30 해결, 실기 PASS** (`4197f5b`). 원인은 10 Hz가 아니라
      창당 타깃 1회 갱신. 30 Hz 재샘플링 + One-Euro 이식 + `InterventionBudget`.
      DRY RUN 300스텝(개입 272)·ARMED 150스텝(개입 120) 둘 다 전체 PASS,
      dp_ratio 중앙값 1.000 / held 0% / `substeps=2`, 조작자 확인 양호.
      (같은 날 세 번째 run — DRY `--scale 0.5`, 개입 144 — 은 고친 판정으로 **SKIP**이다:
      비포화 58표본이 개수 게이트 20은 넘지만 축 여기가 1.4/3.6/1.4 cm로 2 cm 게이트 미달.
      `governed`는 3 run 전부 0이었으나 **첫 타깃만 집계한 값**이다 → `04` §9.9.)
      금지사항 3개(가속도 제한·`target_stale_s`·`soft_start_s`)는 `../docs/testing/04_HIL_INTERVENTION.md` §9.3.
- [ ] 🟠 **대각 이동의 governor 상시 절삭** — `ACTION_SCALE` 헤드룸이 축별로만 성립해서
      2축 0.849배 / 3축 0.693배로 이미 잘리고 있었다. `4197f5b`가 만든 게 아니고
      **관측 수단(`governed_scale`)만 생겼다.** 저장 액션 과대기록 경로다 → 위 "저장 액션 불변식" 절.
- [ ] 개입 예산 소진 후 HOLD (창이 늘어질 때) — **코드로 못 고친다, G21 종속**
- [ ] `GelloIntervention._leader_T`에 TCP_OFFSET 배선 (config는 있음, 현재 플랜지 기준)
- [ ] 로봇 노트북에서 mock 하드웨어 검증 (QoS `VERIFY(hw)` 주석 참고, 그리퍼 방향 육안 확인)
- [ ] per-task config 예제 (`examples/experiments/<task>/config.py` 형식)
- [x] **★ Kanu bounded synthetic learning acceptance — final schema v2**
      — unified `248255f`, 실제 SSH alias `kanu`, JAX/JAXLIB 0.5.3 GPU actual classifier/agent, laptop3 SSH tunnel에서 exact 100 transition → step 1/gradient 2/policy 1/checkpoint full-load roundtrip을 통과했다. fresh process resume가 1/2/1과 policy version 1 finite 7D action을 serving했다. fingerprint는 `fa1985378ad2729f466783e4f112d54022e14090430374a6531e4fb715440fcd`다.
- [ ] **Kanu production robot/continuous acceptance**
      — 남은 범위는 real canonical demo, 기본 50-step publish/5,000-step checkpoint, 장시간 memory/contention, robot E2E다.
      정확한 명령과 feature RAM gate는 `HIL_SERL_KANU_RUNBOOK_KO.md`를 따른다.
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

### 🆕 (2026-07-30, `4197f5b`) 개입 경로에는 `InterventionBudget`이 그 불변식을 지킨다

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

### 속도 3층과 개입 예산 — 예산은 4번째 층이 **아니다**

3층(`ACTION_SCALE` / `GOVERNOR` / `UPSAMPLER` — `ur_env/envs/config.py`의 세 블록, 각 상단
주석이 근거를 담고 있다)은 2026-07-28에
균일 1.25배로 정렬해 헤드룸 1.20x를 유지한다. 개입 예산은 그 위에 얹은 **새 층이 아니라
`ACTION_SCALE`에서 직접 파생된 같은 수**다(`InterventionBudget.__init__`가 `action_scale`을
그대로 받는다). 그래서:

- 예산은 3층 비율을 **바꾸지 않는다.** 다만 개입 창에서는 실효 상한이 governor가 아니라
  **예산**이 된다 — 0.0125 m(예산) < 0.0150 m(`v_max`/HZ)이므로 예산이 먼저 묶는다.
  2026-07-30 실기 **3 run**(개입 536표본)에서 `governed=0`으로 관측된 것이 이 설계대로의
  결과다. ⚠️ **단 그 값은 창의 첫 타깃만 집계하던 코드로 측정됐다** — "첫 타깃에서 절삭
  없음"만 증명하고, 창 전체(서브스텝 2회 포함) 절삭 여부는 아직 **미관측**이다.
  지금 `governed`는 창 전체 OR / `governed_scale`은 창 전체 최솟값이다
  (`tests/run_real_hil.py` docstring (e)).
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
`governed_scale`은 창 전체 **최솟값**이다. 07-30 3 run은 **첫 타깃만** 집계하던 코드로
측정됐으므로 옛 CSV와 새 CSV의 같은 이름 컬럼을 **비교하면 안 된다.**

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
