# CLAUDE.md

## 이 워크트리에서 진행 중인 작업

branch `test/hil-hardware-comms` — HIL-SERL을 실기 UR7e에 올리는 작업(GELLO 리더 + UR7e + 개입 시스템 + 원격 GPU 서버 양방향 통신). 학습 모델과 reward classifier는 다른 담당자의 범위다.

**→ 작업을 이어받으려면 [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md)를 먼저 읽어라.** 현재 상태, 다음 할 일(복붙 가능한 CLI 포함), 안전 규칙, 반복되는 함정이 전부 거기 있다.

## 문서 지도

| 문서 | 용도 |
| --- | --- |
| `serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md` | **시작점** — 지금 무엇을 하면 되는가 |
| `serl_ur_infra/HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` | 전체 상태 기록. learner는 §1–10, actor/하드웨어는 §11 |
| `serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md` | 원격 GPU 서버(Kanu) 실행 절차 |
| `docs/testing/README.md` | 하드웨어·통신 검증 런북 인덱스 (00~09) |

## 이 워크트리에 대해

- 작업 위치는 `/home/laptop3/gello_worktrees/hil-hardware-comms`다. canonical checkout `/home/laptop3/gello_software`(branch `feat/gello-ur7e-humble-22.04`)에는 이 작업의 코드가 **없다** — 같은 PC에서 다른 작업이 진행 중이라 일부러 분리했다. **canonical checkout을 건드리지 말 것.**
- gRPC 코드는 반드시 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로 실행한다. 시스템 `python3`의 grpcio 1.30.2가 손상돼 있어 오류 없이 100% CPU로 무한 정지한다.
- 실기 안전 규칙은 handoff §10에 있다. 특히 **GELLO Dynamixel에 토크를 걸지 말 것**(수동 read-only 리더다).
