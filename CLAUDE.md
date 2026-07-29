# CLAUDE.md

## 이 리포에서 진행 중인 작업

**HIL-SERL(사람 개입 온라인 RL)을 실기 UR7e에서 돌린다.**

```
laptop3                                        kanu (GPU 서버)
  GELLO 리더팔 (USB, EEF teleop)                 정책 추론 (SAC)
  RealSense ×2 (USB)          ──gRPC :50053──►   온라인 학습 (RLPD)
  UR7e (이더넷, ROS2 Humble)  ◄──── 액션 ─────    reward classifier
```

laptop3의 GPU가 약해 **정책·학습·reward classifier를 전부 kanu에서** 돌리고 gRPC로 실시간
통신한다. reward 권위는 서버에 있다. 10 Hz 루프라 스텝 예산이 100 ms인데 현재 RTT p99가 97.1 ms다.

**branch `feat/gello-ur7e-humble-22.04`** (origin/HEAD). 2026-07-29에 `test/hil-hardware-comms`
(로봇/하드웨어)를 머지해서 로봇 쪽과 learner/classifier 쪽이 **하나로 합쳐졌다**. 이전 문서들이
말하던 "워크트리 분리"·"canonical checkout에는 이 코드가 없다"는 **더 이상 사실이 아니다.**

**→ 이어서 작업하려면 [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md)를 먼저 읽어라.**

## 지금 상태 한 줄

하드웨어 경로와 통신은 뚫렸다(팔 구동·개입·frame-map·gRPC 왕복 실기 검증 완료).
남은 관문은 **(a) actor entrypoint를 실기에 처음 올리는 것**과
**(b) reward classifier 전처리를 맞추는 것** 둘이다. 지금은 **reward를 믿으면 안 된다.**

## 문서 지도

| 문서 | 용도 |
| --- | --- |
| `serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md` | **시작점** — 통합 현황, 다음 할 일, 함정 |
| `serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` | 전체 기록. learner §1–10 / actor·하드웨어 §11 / classifier 조사 §12 |
| `serl_ur_infra/REWARD_CLASSIFIER_THRESHOLD_KO.md` | threshold 근거 + leakage 감사 |
| `serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md` | kanu 실행 절차 |
| `docs/testing/README.md` | 하드웨어·통신 검증 런북 (00~09) |
| `docs/testing/08_OPEN_GAPS.md` | 미해결 갭 목록 |
| `serl_ur_infra/HIL_SERL_STATUS_AND_NEXT.md` | 🗄️ 2026-07-24 기록물 — 현재와 다르다. 역사 참고용 |

## 코드 지도

```
serl_ur_infra/
  ur_env/envs/ur7e_env.py          get_im, clip_safety_box, go_to_reset, 관측 조립
  ur_env/envs/config.py            속도 3층(ACTION_SCALE/GOVERNOR/UPSAMPLER), 카메라 토픽
  ur_env/envs/wrappers.py          GelloIntervention, 데드맨, 그리퍼 페널티
  ur_env/envs/frame_wrappers.py    RelativeFrame, Quat2EulerWrapper
  ur_env/envs/ros_backend.py       rclpy 백엔드, 250 Hz 업샘플러
  ur_env/remote_actor.py           actor 루프, 전이 생성·전송
  ur_env/rlpd_receive_server.py    서버 ingress + RewardClassifierRuntime
  ur_env/learner/                  RLPD learner, checkpoint, fingerprint
  ur_experiments/cube_in_cup.py    태스크 config (측정값 전부 여기, IMAGE_CROP 포함)
  scripts/run_remote_rlpd_actor.py actor entrypoint
  tests/run_real_hil.py            실기 개입 러너 (파일 상단 주석이 안전 설계를 설명)
ros2_ur_ws/
  run_hil_actor.sh                 actor 실행 래퍼 (preflight 9종)
  run_hil_gui.sh                   데드맨/개입 GUI
  launch_cameras.sh                RealSense 2대
```

## 반드시 지킬 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **gRPC 코드는 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로만.** 시스템 `python3`의
  grpcio 1.30.2가 손상돼 오류 없이 100% CPU로 무한 정지한다.
- **`IMAGE_CROP`을 "분류기가 안 맞으니" 지우지 말 것.** 데이터셋 측정값이고 정책이 1차 소비자다.
  해결은 classifier 재학습이다(handoff §4).
- **테스트는 passed 수를 볼 것.** PYTHONPATH에서 `serl_launcher`가 빠지면 조용히 333 → 299가 되고
  skip 사유가 거짓말을 한다.
- **메인 브랜치는 여러 사람이 공유한다.** 머지·리베이스 전에 상대 checkout이 깨끗한지 확인할 것.
