# 하드웨어 검증 런북 (텔레옵 리그)

GELLO → UR7e 텔레옵 리그의 부품별 검증 절차와 판정 기준이다. 각 문서는 **재현 절차 + 실측
기록 + 완료 판정 체크리스트**로 되어 있다. 실측값은 대부분 2026-07 Humble 리그에서 잰 것이며,
해당 경로(그리퍼 Modbus, Dynamixel 리더, EEF 브리지)는 Jazzy에서도 코드가 같다.

| 문서 | 대상 | 로봇 이동 |
| --- | --- | --- |
| [`01_GRIPPER.md`](01_GRIPPER.md) | Robotiq 2F-85 단독 (Modbus over tool-comm `:54321`) | 그리퍼만 |
| [`02_GELLO_LEADER.md`](02_GELLO_LEADER.md) | GELLO 리더 Dynamixel 스캔·발행 주기·트리거 스팬·개체 확인(probe) | 없음 |
| [`03_EEF_MODE.md`](03_EEF_MODE.md) | EEF 텔레옵 단계 상승 P6 → P9b | **있음** |

카메라 검증(cam1/cam2 역할, 손목 확인, `uvcvideo` quirks)은
[`docs/hardware/REALSENSE_D435_TROUBLESHOOTING.md`](../hardware/REALSENSE_D435_TROUBLESHOOTING.md)에 있다.

HIL-SERL용 런북(00 안전/셋업, 04 개입, 05 gRPC, 06 센서·19-D state, 07 장애 주입, 08 갭 목록,
09 actor, 10 지연, 11 sim 리허설)은 `research/hil-serl` 브랜치의 `docs/testing/`에 있다.
