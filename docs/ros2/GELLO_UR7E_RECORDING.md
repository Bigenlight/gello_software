# GELLO → UR7e 진단 레코더 (데이터 로깅)

GELLO 리더에서 뽑히는 값과 UR7e의 **실제 관절(위치·속도·토크)**, 브리지 커맨드, 그리퍼, TCP 힘/토크·포즈까지 한 번의 실행에서 CSV로 저장하는 도구입니다. 트래킹 오차·지연·진동을 오프라인으로 분석할 때 씁니다. **읽기 전용**(구독만) — 로봇/GELLO에 명령을 내리지 않습니다.

## 실행

텔레오퍼(sim 또는 실기)를 **먼저** 띄운 뒤, **두 번째 터미널**에서:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_recorder.sh                 # CSV만
BAG=true ./run_recorder.sh        # + 전체 토픽을 ros2 bag으로도 캡처
RATE=200 ./run_recorder.sh        # synchronized.csv 샘플레이트(Hz), 기본 100
```

`Ctrl-C`로 종료 → flush + `metadata.json` 마무리(그리고 bag 정리). 직접 실행도 가능:

```bash
ros2 run ur_gello_bringup gello_ur_recorder --ros-args -p sample_rate_hz:=100.0
```

## 저장 위치 (하위 폴더)

```
ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/
├─ synchronized.csv        ← 핵심: 모든 신호를 100 Hz로 시간 정렬한 wide 테이블
├─ gello_joint_states.csv  ← GELLO 관절 위치 + 유한차분 속도 (native ~30 Hz)
├─ ur_joint_states.csv     ← UR7e 실제 위치/속도/토크 (native ~500 Hz)
├─ command.csv             ← 브리지가 보낸 커맨드 (/forward_position_controller/commands)
├─ gripper.csv             ← GELLO 그리퍼 / 그리퍼 커맨드 / 그리퍼 실제 위치
├─ wrench.csv              ← TCP 힘·토크 (force_torque_sensor_broadcaster)
├─ tcp_pose.csv            ← TCP 데카르트 포즈 (tcp_pose_broadcaster, 로드된 경우)
├─ metadata.json           ← 파라미터·시작시각·토픽별 수신 개수
└─ rosbag/                 ← BAG=true일 때만: 전체 토픽 원본
```

> `gello_logs/`는 `.gitignore` 처리되어 있습니다(데이터는 커밋 안 함, 코드만 커밋).

## `synchronized.csv` 열 (분석용 메인 파일)

| 열 | 의미 |
|---|---|
| `t_rel_s`, `t_wall` | 시작 기준 경과초 / 벽시계 유닉스시각 |
| `gello_q1..6` | GELLO 관절 각도(rad, UR 순서) |
| `gello_qd1..6` | GELLO 관절 속도(rad/s, 유한차분) |
| `gello_grip` | GELLO 그리퍼 폭 (0=열림..1=닫힘) |
| `cmd1..6` | 브리지가 로봇에 보낸 목표 위치(필터/클램프 후) |
| `ur_q1..6` | UR7e **실제** 관절 각도(rad) |
| `ur_qd1..6` | UR7e **실제** 관절 속도(rad/s) |
| `ur_eff1..6` | UR7e 관절 토크/전류(effort) |
| `grip_cmd`, `grip_pos` | 그리퍼 목표 % / 실제 % (0=열림..1=닫힘) |
| `fx..fz`, `tx..tz` | TCP 힘(N)·토크(Nm) |
| `tcp_x..tcp_qw` | TCP 위치(m) + 쿼터니언 |

비어 있는 열 = 해당 토픽이 이번 실행에 발행되지 않음(예: sim에는 wrench/tcp 없음, 팔만 돌리면 gripper 없음).

## 빠른 분석 예 (pandas)

```python
import pandas as pd
df = pd.read_csv("gello_logs/session_XXXX/synchronized.csv")
# 트래킹 오차: 브리지 커맨드 vs 로봇 실제
err = (df[[f"cmd{i}" for i in range(1,7)]].values
       - df[[f"ur_q{i}" for i in range(1,7)]].values)
# 진동: 정지 구간에서 ur_qd(실제 속도)의 표준편차
print(df[[f"ur_qd{i}" for i in range(1,7)]].std())
# 지연: gello_q 대비 ur_q의 상호상관 피크로 lag 추정
```

## 관련 문서

- [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md) — 턴키 셋업 + 튜닝(§5, One-Euro 진동 필터)
- [`GELLO_UR7E_REAL_ROBOT.md`](./GELLO_UR7E_REAL_ROBOT.md) — 실기 런북
- [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) — 2F-85 그리퍼
