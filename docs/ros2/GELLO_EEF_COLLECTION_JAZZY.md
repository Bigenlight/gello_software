# GELLO EEF 텔레옵 데이터 수집 (Jazzy)

EEF(Cartesian delta) 텔레옵으로 실기 데모 take를 모으는 운영 절차다. 레코더 통합 GUI 하나가 카메라
2대와 녹화 세션을 소유하고, 텔레옵 프로세스는 충돌 뒤에 따로 내렸다 올릴 수 있다.

- EEF 모드 자체(앵커·재클러치·속도 상한·게이트): [`GELLO_UR7E_EEF_MODE.md`](./GELLO_UR7E_EEF_MODE.md)
- 레코더 출력 스키마·GUI 상세: [`ros2_ur_ws/src/gello_recorder/README.md`](../../ros2_ur_ws/src/gello_recorder/README.md)
- MuJoCo 시뮬레이션에서 같은 형식의 take를 렌더링해 모으는 `sim_collect`는 `research/sim-collect` 브랜치에 있다.

래퍼는 자기 checkout 기준으로 경로를 풀므로 어느 디렉터리에서 실행해도 되고, ROS Jazzy·시스템
Python 3.12를 auto-activate된 conda보다 앞에 둔다. 다른 배포판은 `GELLO_ROS_DISTRO`로 바꾼다.

## 실기 전

- 로봇 전용 이더넷에 carrier가 있고(`setup_jazzy/setup_ur7e_ethernet.sh --check`) RealSense 2대가 꽂혀
  있을 것(`./launch_cameras.sh --check`).
- 펜던트 E-STOP을 손 닿는 곳에. External Control 또는 headless/Remote 설정은
  [`GELLO_UR7E_SETUP_CLI.md`](./GELLO_UR7E_SETUP_CLI.md) §3–4.
- 빌드: `cd ~/gello_software/ros2_ur_ws && ./build_ur7e.sh`

## 2-터미널 수집

```bash
# Terminal 1 — 실기 UR7e + 물리 GELLO, EEF 텔레옵 (충돌 뒤 내렸다 다시 올려도 된다)
cd ~/gello_software/ros2_ur_ws
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef gripper_mode:=continuous
```

```bash
# Terminal 2 — 통합 수집 GUI (카메라 + 녹화 + EEF 조작). Terminal 1을 재시작해도 계속 띄워 둔다
cd ~/gello_software/ros2_ur_ws
./run_eef_collection_gui.sh
```

창 어디서나 쓰는 키:

| 키 | 동작 |
|---|---|
| `R` | 녹화 START / STOP (카메라 워밍업 중에는 무시) |
| `SPACE` | EEF ENGAGE / DISENGAGE (ENGAGE는 두 번 눌러 확인) |
| `G` | 그리퍼 PAUSE / RESUME (RESUME은 두 번 눌러 확인) |

GUI는 연결된 카메라 시리얼을 설정값과 대조하고(불일치면 실패 — [`REALSENSE_D435_TROUBLESHOOTING.md`](../hardware/REALSENSE_D435_TROUBLESHOOTING.md)),
워밍업을 기다린 뒤 `take_<NN>_<timestamp>/` 폴더를 레코더 출력 루트 아래에 쓴다. `CAM1_SERIAL`, `CAM2_SERIAL`,
`COLOR_PROFILE`, `CAMERA_WARMUP_S`, `START_POSE_CONFIG`는 필요할 때만 덮어쓴다.

> 마우스 GUI만 따로 쓰고 싶으면 Terminal 2에서 `./run_eef_gui.sh`(EEF 조작만) +
> 별도 터미널에서 `ros2 run gello_recorder gello_recorder_gui`(녹화만)로 나눌 수 있다.

## 동작과 자세 기본값

`control_mode:=eef`는 `start_mode:=switch_only`를 유도한다 — 기동은 컨트롤러를 제자리에서 전환할 뿐
**팔을 움직이지 않는다.** EEF 동작은 GUI가 ENGAGE한 뒤에만 시작한다. 그 외에 팔을 움직이는 것은
레코더의 **GO HOME**과 EEF GUI의 **GO TO START POSE** 두 버튼뿐이다. 녹화 START는 take만 연다.

**GO TO START POSE**는 `START_POSE_CONFIG`(기본 `src/ur_gello_bringup/config/start_pose.yaml`)의
`policy_leader_node.ros__parameters.start_pose`/`start_gripper`를 읽는다. 기본 파일은 carrot_in_pot 시작 자세다:

```text
start_pose    = [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]
start_gripper = 0.0
```

다른 태스크는 그 태스크의 자세를 담은 yaml을 가리킨다(같은 키 경로라 정책 deploy yaml을 그대로 줘도 된다).
"비슷해 보이는 값"으로 고치지 말 것 — banana와 carrot 시작 자세는 pan 값이 약 2π 차이로 비슷해 보여도
다른 관절이 최대 0.33 rad 다르다.

레코더 **GO HOME**은 기본적으로 자기 고정 목표 `HOME_JOINTS`
(`3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331`)로 가고 그리퍼를 연다. 두 목표는 **의도적으로 다르다** —
GO HOME을 `START_POSE_CONFIG` 목표로 취급하지 말 것.

GO HOME 뒤에는 두 브리지가 설계대로 `PAUSED`로 남는다. 레코더의 일반 **Resume Teleop** 버튼은 조인트 모드
재개 경로라 쓰지 않는다. GELLO 트리거를 연 채로 EEF 쪽 **ENGAGE**를 누르면 `eef_resume` → `pos_scale` 커밋 →
`eef_engage`가 이어져 현재 자세에 다시 앵커한다. 그리퍼는 별개다 — 준비되면 그리퍼 RESUME을 따로 한다.
팔 ENGAGE만으로는 멈춘 그리퍼가 풀리지 않는다.

## 그리퍼 규약

실기 래퍼 기본은 `gripper_mode:=continuous`다. 명령은 텔레옵·레코더 전 구간에서 **0.0 = 열림 / 1.0 = 닫힘**으로
유지된다. continuous는 트리거 값을 그 범위로 그대로 흘리고, `gripper_mode:=discrete`는 히스테리시스
(`>=0.7` 닫힘, `<=0.3` 열림, 사이는 유지)로 양 끝에 스냅하므로 녹화된 `grip_cmd`가 이진이 된다.
**한 데이터셋 안에서는 모드를 고정**하고, GO HOME 뒤 재개 전에는 트리거를 연 상태로 둔다.
상세: [`GELLO_UR7E_GRIPPER.md`](./GELLO_UR7E_GRIPPER.md) §8.

## 안전한 래퍼 검증 (로봇·GELLO·카메라 미접속)

```bash
cd ~/gello_software/ros2_ur_ws
bash -n run_eef_gui.sh run_eef_collection_gui.sh run_recorder.sh \
  run_ur7e_gello_real.sh run_ur7e_gello_mock.sh run_ur7e_gello_sim.sh
DRY_RUN=1 ./run_ur7e_gello_mock.sh --print-args control_mode:=eef launch_rviz:=false
```

mock 스크립트는 `robot_ip:=127.0.0.1`과 `use_fake_hardware:=true`를 고정하므로 실기로 돌릴 수 없다.

EEF 서비스 mock 스모크(GenericSystem, 쓰지 않는 로컬 ROS 도메인에서):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ROS_DOMAIN_ID=176 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  /usr/bin/python3 setup_jazzy/eef_mock_smoke.py
```

점유된 도메인은 거부하고, GenericSystem임을 확인한 뒤에만 mock 위치 명령을 보낸다. 드라이버 기본 자세가
특이점으로 거부되는 것을 확인한 다음 비특이 자세를 시드한다. 검증 범위: STRICT 핸드오프 → HOLD, 특이점 앵커
거부, 리더/로봇 자세가 다를 때 zero-jump engage, 재클러치, 델타 추종, disengage 후 hold, zero-jump 재engage
(2026-09-17 PASS). 실제 UR 추종, 카메라 시리얼 배정, 그리퍼 동작, 실기 HOME 경로는 검증하지 않는다.
