# serl_ur_infra

`third_party/hil-serl`의 `serl_robot_infra`(Franka 전용)에 대응하는 **UR7e + GELLO** robot infra.
`FrankaEnv`의 관측/액션 계약을 그대로 복제해서 hil-serl의 wrapper 체인·actor 루프가 무수정으로 돌게 한다.

> 🚩 **이 리포에 처음 왔다면** 루트 [`CLAUDE.md`](../CLAUDE.md) → [`HANDOFF_NEXT_SESSION_KO.md`](HANDOFF_NEXT_SESSION_KO.md)
> 순서로 읽어라. 이 README는 **env 설계 규약**과 **이 디렉터리 문서 색인**이지 현재 상태 문서가 아니다.

> ⚠️ **정책 경로는 여전히 `config.DRY_RUN=True`(명령 미발행)가 기본이다.**
> (2026-07-29 정정: 예전 머리말의 "UNTESTED SKELETON / 실기 사용 금지"는 낡았다. 사람 개입 경로는
> 2026-07-28에 실기에서 팔을 구동해 검증했다 — `tests/run_real_hil.py`, `docs/testing/04_HIL_INTERVENTION.md` §4.5.
> 다만 **정책이 팔을 움직인 적은 없고**, actor entrypoint `scripts/run_remote_rlpd_actor.py`도
> 실기에서 돈 적이 없다.)

최종 운영 checkout은 `/home/laptop3/gello_software`, branch는 `feat/gello-ur7e-humble-22.04` 하나다.
2026-07-29 머지 `3f199d4`로 로봇/하드웨어 작업이 이 브랜치에 들어왔다 — **워크트리 분리 시절 서술은
전부 낡았다.** 통합 상태·checkpoint·Kanu 검증·frozen-trunk feature replay·bounded fake-data
learning E2E 계약은 [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)를 기준으로 한다.

`scripts/run_fake_e2e_actor.py`는 robot를 제어하는 actor가 아니라 `--synthetic-e2e` Kanu learner에 canonical raw fake observation 100개를 보내 gRPC→classifier→feature replay→CTA→publish→checkpoint→fresh-process resume를 검증하는 acceptance tool이다. server는 exact actor/run ID, exact 100 inserts, bounded timeout을 강제하고 synthetic-only model ID를 advertise한다. cleanup 후 full checkpoint roundtrip/trunk invariant까지 통과해야 pass한다. synthetic checkpoint는 fingerprint/model scope가 다르므로 production robot lineage에 사용할 수 없다.

## 이 디렉터리의 문서 (`serl_ur_infra/*.md`)

### 지금 유효 — 지침으로 읽어도 되는 것

| 문서 | 무엇인가 |
| --- | --- |
| [HANDOFF_NEXT_SESSION_KO.md](HANDOFF_NEXT_SESSION_KO.md) | **진입점.** 현재 상태 · 리그 실측값 · 다음 할 일 · 함정. 이 하나로 작업을 시작할 수 있게 쓰여 있다 |
| [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md) | 전체 상태 기록. learner 구현 §1–10 / actor·하드웨어 §11 / reward classifier 조사 §12. 위 문서보다 깊다 |
| [HIL_SERL_KANU_RUNBOOK_KO.md](HIL_SERL_KANU_RUNBOOK_KO.md) | Kanu에서 learner를 띄우는 절차 (dry-run → bounded run) |
| [REWARD_CLASSIFIER_THRESHOLD_KO.md](REWARD_CLASSIFIER_THRESHOLD_KO.md) | reward threshold를 0.85 → 0.2로 내린 근거 + 2026-07-29 누출 감사. 07-28 수치와 07-29 수치를 구별해 인용할 것 |
| [REMOTE_ACTOR_GRPC.md](REMOTE_ACTOR_GRPC.md) | gRPC 액터 전송 계약 v2 — 같은 전송을 쓰는 **서버 entrypoint 3종의 차이**. 영문, 머지 트리 기준 재검증 |
| [RVIZ_HIL_TEST_CLI.md](RVIZ_HIL_TEST_CLI.md) | mock(`use_fake_hardware`) 4터미널 개입 테스트 절차. 실기 위험 0 |
| [REWARD_CLASSIFIER_LIVE_KO.md](REWARD_CLASSIFIER_LIVE_KO.md) | 라이브 reward classifier 뷰어 런북 (랩톱 CPU, 터미널 4개). **2026-07-29 실기 검증됨.** 인터프리터 함정 · 조용한 실패 · 트러블슈팅 |
| [REWARD_TO_RL_INTEGRATION_KO.md](REWARD_TO_RL_INTEGRATION_KO.md) | **분류기를 개입·학습에 연결하는 사람이 읽을 것.** reward/termination 계약(서버 권위, `next_observations` 기준) · 개입의 **버퍼 이중 기록** · RLPD 50:50 배치 · 연결 순서 6단계 · 감시 지표. 코드에서 직접 추적해 작성 |

### 🗄️ 기록물 — 사료로만. 여기 적힌 명령을 실행하지 말 것

| 문서 | 왜 |
| --- | --- |
| [HIL_SERL_STATUS_AND_NEXT.md](HIL_SERL_STATUS_AND_NEXT.md) | 2026-07-24 서버 핸드오프. 전송을 agentlace 5588/5589로, 진입점을 `train_rlpd.py`로, jax를 0.4.35로 적는다 — 셋 다 현행과 다르고 **그대로 준비하면 learner가 즉사한다.** 문서 머리에 대조표가 있다 |
| [RL_RECEIVE_SERVER.md](RL_RECEIVE_SERVER.md) | receive-only 마일스톤(영문). 인용된 classifier checkpoint는 은퇴했고 **19-D `state` 순서를 v1(틀린 순서)로 적은 곳이 남아 있다** |
| [HIL_RLPD_RECEIVE_SERVER_KO.md](HIL_RLPD_RECEIVE_SERVER_KO.md) | 은퇴한 `feat/hil-rl-receive-server` 브랜치 시절 작업 정리. 브랜치·워크트리 경로가 낡았다 |
| [ACTOR_ADAPTER.md](ACTOR_ADAPTER.md) | agentlace 로컬 어댑터 `scripts/train_rlpd_actor.py` 기준(영문). **실기 경로가 아니다** — `create_actor_network()`가 `agentlace`를 `NotImplementedError`로 거부한다. transition 계약 설명만 유효 |

디렉터리 밖 관련 문서: [`../docs/testing/README.md`](../docs/testing/README.md)(하드웨어·통신 검증 런북 00~09) ·
[`../docs/testing/09_HIL_ACTOR_RUNBOOK.md`](../docs/testing/09_HIL_ACTOR_RUNBOOK.md)(actor 기동) ·
[`../docs/testing/08_OPEN_GAPS.md`](../docs/testing/08_OPEN_GAPS.md)(미해결 갭).

## 구조

| 파일 | 대응하는 hil-serl 코드 | 역할 |
| --- | --- | --- |
| `ur_env/envs/ur7e_env.py` | `franka_env/envs/franka_env.py` | gym env 본체 (step/reset, 관측, 카메라, 그리퍼) |
| `ur_env/envs/config.py` | `DefaultEnvConfig` | 태스크별 config 베이스 |
| `ur_env/envs/ros_backend.py` | Flask 로봇 서버 (HTTP) | rclpy 백그라운드 노드 — 토픽 I/O |
| `ur_env/envs/policy_delta_controller.py` | (Franka 임피던스 컨트롤러가 하던 일) | 정책 델타 → 거버너 → IK → 게이트 → 조인트 명령 |
| `ur_env/envs/wrappers.py` | `SpacemouseIntervention` | `GelloIntervention` — 데드맨 + 앵커 클러치 개입 |

## 설계 요점

- **액션**: `[-1,1]^7` (6D EEF 델타 + 그리퍼), `ACTION_SCALE`로 실단위 변환 — FrankaEnv와 동일.
- **정책 경로**: 델타를 `T_cmd`에 적분 → 거버너 rate cap → seed 기반 IK → 조인트 스텝 게이트 → 의심스러우면 HOLD. (`eef_delta.py` 후반부의 단순화판; 추후 본체 재사용으로 교체 예정)
- **개입 경로**: 데드맨(스페이스바 홀드, 추후 풋스위치)을 누르는 순간 (GELLO, 로봇 명령 자세) 앵커 래치 → 앵커 델타를 per-step env 액션으로 재표현(클립이 자연스러운 추격 속도 제한이 됨) → `info["intervene_action"]` 보고.
- **카메라**: franka_env처럼 pyrealsense2로 장치를 직접 열지 않고, `launch_cameras.sh`가 띄우는 realsense2_camera 드라이버의 `/camX/.../compressed` 토픽을 구독 (RealSense는 이중 오픈 불가 + 기존 viewer/recorder 생태계와 공존). JPEG 디코드→크롭→128×128 리사이즈→RGB는 FrankaEnv와 동일. `DISPLAY_IMAGE=True`면 정책 시점 이미지를 OpenCV 창으로 실시간 표시 (ImageDisplayer 포팅).
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
| 워크스페이스 박스 (`ABS_POSE_LIMIT`) | (keepout으로 대체) | 🟠 구현됨(`clip_safety_box`), 단 실기 미발동 | `run_real_hil.py`는 `DefaultUR7eEnvConfig`(=0)를 쓰므로 박스가 꺼진다. 측정 박스는 `cube_in_cup.py`에만 있다 |

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
- [ ] 데드맨 하드웨어 (풋스위치, 현재 스페이스바) + 성공/실패 라벨링 키
- [ ] `GelloIntervention._leader_T`에 TCP_OFFSET 배선 (config는 있음, 현재 플랜지 기준)
- [ ] 로봇 노트북에서 mock 하드웨어 검증 (QoS `VERIFY(hw)` 주석 참고, 그리퍼 방향 육안 확인)
- [ ] per-task config 예제 (`examples/experiments/<task>/config.py` 형식)
- [x] **★ Kanu bounded synthetic learning acceptance — final schema v2**
      — unified `248255f`, 실제 SSH alias `kanu`, JAX/JAXLIB 0.5.3 GPU actual classifier/agent, laptop3 SSH tunnel에서 exact 100 transition → step 1/gradient 2/policy 1/checkpoint full-load roundtrip을 통과했다. fresh process resume가 1/2/1과 policy version 1 finite 7D action을 serving했다. fingerprint는 `fa1985378ad2729f466783e4f112d54022e14090430374a6531e4fb715440fcd`다.
- [ ] **Kanu production robot/continuous acceptance**
      — 남은 범위는 real canonical demo, 기본 50-step publish/5,000-step checkpoint, 장시간 memory/contention, robot E2E다.
      정확한 명령과 feature RAM gate는 `HIL_SERL_KANU_RUNBOOK_KO.md`를 따른다.

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

## 설치 (예정)

```bash
conda activate hilserl
pip install -e third_party/hil-serl/serl_launcher
pip install -e serl_ur_infra
source ros2_ur_ws/install/setup.bash    # ur_gello_bringup (ur_kin, eef_delta)
# 카메라는 별도 터미널에서: ./ros2_ur_ws/launch_cameras.sh
```
