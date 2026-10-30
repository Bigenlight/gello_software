# serl_ur_infra 준비 상태 점검 (2026-07-24)

`serl_ur_infra/`(UR7e + GELLO용 hil-serl robot infra) 첫 커밋 시점의 상태 스냅샷.
계획 문서는 [GELLO_UR7E_HIL_SERL_PLAN.md](GELLO_UR7E_HIL_SERL_PLAN.md), 설계
세부(좌표계 규약, 컨트롤러 비교표, 런북)는 `serl_ur_infra/README.md` 참고.

## 컴포넌트별 상태

| 컴포넌트 | 상태 | 검증 수준 |
| --- | --- | --- |
| 관측 경로 — 카메라(compressed 토픽→JPEG 디코드→crop→resize→RGB), robot state(driver/fk 선택), UR 내장 F/T, tcp_vel(J·dq) | 구현 완료 | 오프라인 테스트 통과 |
| 액션 경로 — [-1,1]→ACTION_SCALE→거버너→IK→게이트→조인트/그리퍼 명령 | 구현 완료 (컨트롤러는 단순화판) | 오프라인 테스트 통과 |
| 250 Hz 명령 업샘플러 (slew-limited, EMA 없음) + `go_to_reset` (RESET_JOINTS 추격, 원거리 거부) | 구현 완료 | slew 수학 검증됨, 실 ROS 미검증 |
| GELLO 개입 wrapper — 데드맨(스페이스바) + 앵커 클러치 + [-1,1] 저장 | 구현 완료 | **미테스트** |
| fake_env 모드 (learner 프로세스용 공간 정의) | 완결 | 통과 |
| 저장 액션 불변식 — ACTION_SCALE×HZ < 거버너 캡, 기동 시 검사 | 반영 완료 | 통과 |

## 오프라인 테스트가 잡은 버그 (수정 완료)

1. **델타 좌표계**: 컨트롤러가 `T_cmd @ exp(xi)`(툴 프레임)로 구현돼 +x 액션이
   base +y로 이동. base 프레임 분리 조립(`p+v`, `so3_exp(w)@R`)으로 수정.
   FrankaEnv/RelativeFrame 계약은 base 프레임이 정답.
2. **그리퍼 스케일**: Robotiq 노드는 0..1인데 0..100으로 명령/판독. 수정.

## 테스트 자산 (재현 방법)

- `serl_ur_infra/tests/test_env_fake_backend.py` — ROS 없는 머신에서 전체
  제어/관측 경로 실행. `conda run -n lerobot python tests/test_env_fake_backend.py`
- `serl_ur_infra/tests/run_rviz_fake_rl.py` — 로봇 노트북에서 mock 하드웨어 +
  RViz로 RL 루프 육안 검증 (README 런북 참고)
- 기존 `ur_gello_bringup/test/test_eef_delta.py` + `test_ur_kin.py` 56개 —
  컨트롤러 본체 교체 리팩토링의 안전망. 순수 numpy라 dev 머신에서 통과 확인
  (worst tick 0.815 ms — 250 Hz 예산 4 ms 안이므로 본체를 업샘플러 루프에서
  돌려도 됨)

## 남은 작업 (단계별)

### A. 실기 DRY_RUN 해제 전 필수
1. `PolicyDeltaController` → `eef_delta` 후반부 재사용 (branch-lock 해석 IK,
   sigma_min 특이점 감속, keepout) — `serl_ur_infra/README.md` 비교표 참고
2. 워크스페이스 박스(`ABS_POSE_LIMIT`)를 컨트롤러 게이트에 연결 (현재 config만 존재)
3. 로봇 노트북 mock 검증 — `run_rviz_fake_rl.py`, QoS `VERIFY(hw)` 주석 항목,
   그리퍼 방향 육안 확인
4. staleness safe-stop 정책 통일 + UR fault recovery

### B. 학습 시작 전 필수
5. `GelloIntervention` 오프라인 테스트 + 데드맨 풋스위치 + 성공/실패 라벨 키
6. 첫 태스크 config 1개 (RESET_JOINTS·ACTION_SCALE·바운딩 박스·카메라 크롭 실측)
7. **★ GPU 서버(ssh alias `kanu_junhyeong`)에 RLPD learner 구축** — JAX(cuda12)
   + serl_launcher + agentlace, SSH 터널을 agentlace 포트 2개(5588/5589)로.
   기존 remote-diffusion의 터널/프리플라이트/Docker 패턴 재사용 (gRPC 코드 제외)
8. 리워드 분류기 데이터 수집·라벨링·학습, 데모 20~30개 (`record_demos.py` —
   GelloIntervention이 곧 수집 도구)

### C. 운영 품질
9. 모니터링(터널 상태 operator console 노출, wandb), 에피소드 mp4 저장 포팅,
   `RANDOM_RESET`

## 핵심 설계 결정 기록

- **`serl_launcher`/`agentlace`는 그대로 사용, `serl_robot_infra`(Franka 전용)만
  자체 작성** — env는 FrankaEnv 관측/액션 계약을 복제해 wrapper 체인·actor 루프
  무수정 재사용
- **정책 액션 = base 프레임 EEF 델타 10 Hz**, 부드러움은 250 Hz 업샘플러가 합성
  (Franka 임피던스의 소프트웨어 대체; UR은 토크 인터페이스가 없어 임피던스 불가)
- **개입 = GELLO 앵커 클러치 + 데드맨** (절대자세 장치라 스페이스마우스식
  "출력=0" 감지 불가), 저장 액션은 실행 액션과 동일한 [-1,1] 값
- **그리퍼는 3-상태 이산** (hil-serl 하이브리드 에이전트의 이산 grasp critic에 맞춤)
- 카메라·명령·상태 전부 기존 UR7e 배포 라인과 같은 토픽 소비 (RealSense 이중
  오픈 불가, 기존 viewer/recorder 생태계와 공존)
