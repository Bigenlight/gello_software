# UR GELLO 두 대 — 같은 캘리브레이션으로 바꿔 쓰기 (2026-10-30)

GELLO 리더팔이 두 대 있다. **두 팔은 모터 방향과 캘리브레이션이 같다.** 그래서 어느 팔을 꽂든
`ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml`을 그대로 쓴다.
재캘리브레이션도 모터 방향 변경도 필요 없다.

## 기준값

UR 캘리브레이션 포즈 `[0, −90°, 90°, −90°, −90°, 0]`(`start_joints`)에 둔 GELLO에서 기대되는
모터 원시 각도다. `raw = q × sign + offset (mod 2π)`, 틱은 4096/rev로 계산했다.

| 모터 | 관절 | offset | sign | 기대 raw | 기준 팔 실측 | 2번 팔 실측 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | shoulder_pan | 0.0 | +1 | 0° | 354.4° | 356.4° |
| 2 | shoulder_lift | 1.571 | +1 | 0° | 359.1° | 0.4° |
| 3 | elbow | 4.712 | −1 | 180° | 182.9° | 181.6° |
| 4 | wrist_1 | 4.712 | +1 | 180° | 181.7° | 180.0° |
| 5 | wrist_2 | 4.712 | +1 | 180° | 174.0° | 178.5° |
| 6 | wrist_3 | 3.142 | +1 | 180° | 176.4° | 175.4° |
| 7 | 그리퍼 (M077) | open 210.65° / close 168.85° | | 열림 ≈ 210.65° | 215.3° | 209.7° |

- 두 팔 모두 XL330-M288 ×6과 그리퍼 XL330-M077로 구성되고, drive_mode는 0, 토크는 OFF다.
- 실측값을 π/2 단위로 반올림한 offset은 두 팔 모두 `[0, 1.571, 4.712, 4.712, 4.712, 3.142]`로,
  yaml의 값과 같다. 실측 오차는 손으로 자세를 잡은 오차이고 모두 ±6° 안이다
  (시작 정렬 허용치 0.2 rad = 11.5°).
- 2π 단위 차이는 무관하다. 부팅할 때 `gello/robots/dynamixel.py`의 `start_joints` 오토랩이
  offset을 가장 가까운 2π 배수로 정규화한다.

## ⚠️ 정지 자세만으로는 방향을 증명할 수 없다

기대 raw 값이 0°와 180°뿐인데, 모터가 반대로 장착됐을 때의 값 `360° − raw`도 그 두 각도에서는
**똑같이 나온다.** 그래서 방향은 관절을 움직여 봐야 확인할 수 있다. 이번에는 MuJoCo sim에서
두 팔을 번갈아 끼워 움직였고, 관절 6개와 그리퍼가 같은 방향으로 도는 것을 조작자가 확인했다.

## 재현 절차

1. GELLO를 캘리브레이션 포즈로 들고 읽기 전용 프로브를 실행한다(아무것도 쓰지 않는다):
   ```bash
   python scripts/gello_probe_readonly.py --port /dev/serial/by-id/<U2D2 by-id>
   ```
   `matches calibration : True`가 나오면 정지 자세 기준으로는 맞는 것이다. 포트가 `busy`라고 나오면
   DYNAMIXEL Wizard를 Disconnect한다.
2. 방향은 sim에서 확인한다. `configs/ur7e_gello_sim.yaml`은 실기와 같은 offset을 쓴다
   (`rwh_ur.yaml`과는 pan offset이 다르다):
   ```bash
   DISPLAY=:1 PYTHONPATH=$PWD python experiments/launch_yaml.py \
     --left-config-path configs/ur7e_gello_sim.yaml
   ```
   팔을 바꿀 때는 **프로세스를 종료하고 → 바꿔 끼우고 → 재실행**한다. 오토랩은 기동 시점에만
   계산되고, 실행 중에 뽑으면 `comm failed`가 난다.
   `third_party/mujoco_menagerie` 서브모듈이 필요하다. 비어 있으면 다른 checkout의 경로를 쓴다.
3. 어떤 관절이 반대로 돌면 그 모터의 Drive Mode bit0(reverse)을 켜서 `joint_signs`를 공유하게
   만들고, 혼 위치가 90°/180° 어긋나 있으면 혼을 재장착하거나 Homing Offset으로 맞춘다.
   (둘 다 EEPROM 쓰기다. 이번에는 필요 없었다.)

## U2D2 어댑터

시리얼 번호는 **팔이 아니라 U2D2에 붙어 있다.** 어댑터를 바꾸면 `/dev/serial/by-id/...` 경로만
바뀌므로 yaml의 `port`를 고치면 된다. 모터 값은 어댑터와 무관하다.
현재 어댑터는 `FTBEO6QK`(yaml 기본값)와 `FTBEO6QM` 두 개다. 새 어댑터 `FTBEO6QM`으로는 모터 7개가
모두 응답하는 것까지 확인했다. 그 어댑터에서 캘리브레이션 포즈로 다시 읽어 보는 확인은 아직 하지 않았다.

## 함께 알아둘 것

`ur7e_gello.yaml:15`의 주석은 "w3 오프셋을 3.142→0.0으로 바꿨다"라고 되어 있다. 그러나 git 이력
(`0ad407b` → `d038beb`)을 보면 원본 `[3.142, …, 3.142]`와 지금 값이 다른 곳은 **J1(pan)**이다.
w3는 `d038beb`에서 3.142로 되돌아갔다. 같은 UR 포즈에서 `ur_gello.yaml`/`rwh_ur.yaml`을 쓰면
모터 1의 raw가 180°로 나온다.
