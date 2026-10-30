# UR7e + GELLO 위에서 HIL-SERL 구현 계획

> 참고 코드: [`rail-berkeley/hil-serl`](https://github.com/rail-berkeley/hil-serl) (Apache-2.0)
> 대상 리포: `gello_software` (`feat/gello-ur7e-humble-22.04` 브랜치)
> 팀 규모: 3인

## 0. 배경

현재 리포는 GELLO 수동 리더암 → UR7e 텔레옵 → (ACT / Diffusion / FM) **모방학습** 스택이며, 강화학습 관련 코드는 없다. HIL-SERL은 여기에 새로 얹는 레이어다.

`rail-berkeley/hil-serl`을 검토한 결과:

- **`serl_launcher`** (SAC/BC 에이전트, replay buffer, vision, reward classifier, gym wrapper) → **그대로 재사용**. JAX 기반.
- **`serl_robot_infra`** (로봇별 Flask 서버 + gym env) → **Franka 전용**이라 재사용 불가. `serl_franka_controllers`(임피던스 컨트롤러)에 강결합. UR7e용은 신규 작성 필요.
- **actor–learner 통신**: `agentlace` 라이브러리로 이미 구현돼 있음. 우리 gRPC 파이프라인을 개조하지 말고 `agentlace`를 그대로 채택.

## 1. 재사용 가능한 기존 자산

| 자산 | 위치 | 용도 |
| --- | --- | --- |
| gRPC 원격 추론 골격 (참고용 패턴) | `ros2_ur_ws/src/gello_policy/policy_server`, `.../remote_diffusion_client.py` | actor/learner 분리 설계 시 참고 (실제 통신은 agentlace로 대체) |
| 안전 클램프 / 워치독 | policy_leader_node의 safety check, stale-input 정지 로직 | UR7e env wrapper의 안전 레이어 베이스 |
| GELLO 리더암 | `gello/agents`, `gello/dynamixel` | RL 정책 실행 중 사람 개입(intervention) 채널 |
| UR7e 턴키 브링업 / fake-hardware 안전 launch | `ros2_ur_ws/src/ur_gello_bringup`, `ur_control_fake_safe.launch.py` | 실기 없이 ROS 그래프 검증 |
| MuJoCo 시뮬레이션 | `gello/dm_control_tasks` | 알고리즘/파이프라인 초기 검증 (실기 리스크 없이) |
| 카메라 파이프라인 | `experiments/launch_camera_*.py` | 관측 이미지 소스 |
| 데이터 레코더 노하우 | `ros2_ur_ws/src/gello_recorder` | 온라인 buffer/데모 수집 설계 참고 |

## 2. 신규로 만들어야 하는 것

1. **`ur_env` (gym 형 env wrapper)** — Franka용 `franka_env`에 대응. 고정 control-rate(예: 10Hz) step/reset 인터페이스로 UR7e + Robotiq 2F-85 + 카메라를 감싼다.
2. **UR7e 로봇 서버** — `serl_robot_infra`의 Flask 서버에 대응하는 ROS2 브리지. 기존 `ur_gello_bringup` 컨트롤러 그래프 위에 얹는다.
3. **사람 개입(intervention) 중재 로직** — GELLO 입력이 들어오면 정책 액션을 override하고, 매 스텝 액션 출처(policy vs human)를 태깅. 현재 텔레옵 코드엔 이 개념이 없음.
4. **리워드 분류기 학습 데이터 파이프라인** — 성공/실패 이미지 라벨링 → `serl_launcher`의 reward classifier 학습에 투입.
5. **자율 탐색용 안전 강화** — 워크스페이스 바운딩 박스, 조인트/카르테시안 속도 리밋, E-stop 연동을 텔레옵 때보다 훨씬 보수적으로 재설계 (RL은 사람 없이도 무작위 행동을 시도하므로).
6. **에피소드 리셋 루틴** — 스크립트 리셋 또는 사람 개입 리셋 절차 (태스크별로 설계).
7. **데모 부트스트랩** — 초기 replay buffer를 seed할 20~30개 데모를 `ur_env` 포맷으로 신규 수집 (기존 LeRobot 데이터셋은 포맷이 달라 그대로 못 씀).
8. **모니터링/실험 로깅** — 장시간 학습 중 실시간 상태 확인 (`run_operator_console.sh` 확장) + wandb 등 실험 추적.

## 3. 태스크 선정

- 1차 태스크는 **`banana_in_pot`처럼 복잡한 것 말고, `cube_in_cup` 수준 이하의 단순 삽입/누르기**로 시작해서 파이프라인 전체를 먼저 검증한다.
- RL은 모방학습보다 실기 소모 시간(사람 개입 포함)이 훨씬 크다는 점을 일정에 반영한다.

## 4. 3인 분업

| 담당 | 범위 |
| --- | --- |
| **A — 로봇/ROS2 + 안전 + 개입 하드웨어** | `ur_env`/로봇 서버, 리셋 루틴, 안전 리밋·워치독 강화, GELLO 개입 채널 및 액션 출처 태깅 |
| **B — RL 알고리즘 + 분산 인프라** | `serl_launcher`(SAC/BC) 통합, `agentlace` 기반 actor-learner 연결, 온라인 replay buffer, 데모 부트스트랩 |
| **C — 리워드/데이터/실험 운영** | 성공 분류기 데이터 수집·라벨링·학습, 모니터링/로깅, 태스크 선정 및 평가 프로토콜 |

## 5. 초반 결정 사항

1. ~~공식 코드 통합 vs 자체 재구현~~ → **통합으로 확정** (`serl_launcher` + `agentlace` 채택, Apache-2.0 라이선스 확인 완료).
2. JAX(SAC) ↔ PyTorch(ACT/Diffusion/FM) 프레임워크 공존을 전제로 진행 (프로세스 분리이므로 문제 없음, 통일 기대는 하지 않음).
3. 1차 태스크는 최소 난이도로 선정 (§3).
4. 학습 중 사람 개입에 필요한 인력/시간을 일정에 명시적으로 반영.

## 6. 리스크

- 자율 탐색 중 충돌 리스크 — 안전 레이어를 텔레옵 대비 강화 필요.
- 사람 개입 인력이 3인 팀의 개발 시간과 경합.
- UR7e `robot_infra`가 공식 예제 없이 신규 작성이라 디버깅 비용이 예상보다 클 수 있음.
