# 시뮬레이션 퀵스타트 (예제 2개)

기본 동작만 확인하는 작은 예제 두 개다. 대규모 MuJoCo 데이터 수집 sim(`sim_collect`)은
`research/sim-collect` 브랜치로 옮겨졌다. 긴 가이드가 필요하면
[`GELLO_UR_SIM_TELEOP.md`](GELLO_UR_SIM_TELEOP.md)를 본다(이 문서와 겹치는 상세판).

> GELLO는 수동(passive) 리더다. **Dynamixel 토크를 절대 켜지 않는다.** 아래 어떤 예제도 실물 로봇에 연결하지 않는다.

| | A. ROS mock + RViz | B. 업스트림 MuJoCo |
| --- | --- | --- |
| 로봇 모델 | **실제 UR7e** 기구학 (물리 없음) | UR5e + Robotiq 2F-85 (물리 있음, **UR7e 아님**) |
| 리더 | `source:=fake` 합성 리더 **(GELLO 불필요)** / `source:=gello` 실물 | 실물 GELLO / 가짜 리더 스크립트 **(GELLO 불필요)** |
| 환경 | ROS 2 Jazzy, Ubuntu 24.04 | conda `gello-sim` |

## A. ROS mock + RViz

사전 준비 (한 번):

```bash
cd ros2_ur_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select ur_gello_bringup gello_recorder
```

**GELLO 없이** (fake 리더가 관절을 흔든다):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake            # RViz 창
ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false   # 헤드리스
```

`source:=fake`가 기본값이다. **실물 GELLO로 구동**하려면 `./run_ur7e_gello_sim.sh`
(내부적으로 `source:=gello`, `/dev/ttyUSB0` 읽기 전용). `run_ur7e_gello_mock.sh`,
`run_ur7e_gello_joint_delta_mock.sh`도 모두 **물리 GELLO가 필요**하다(실로봇 기동 순서
리허설용이며 `robot_ip` 루프백·mock이 강제된다).

확인 (다른 터미널):

```bash
ros2 topic hz /joint_states                       # ~490 Hz
ros2 topic echo --once /joint_states --field position   # 두 번 찍으면 값이 다르다
```

보이는 것: RViz에서 UR7e가 움직인다. 로그에 `[robotiq_urcap] [no gripper] would move to POS=…`가 반복된다(그리퍼 mock).

## B. 업스트림 MuJoCo (UR5e)

사전 준비:

```bash
git submodule update --init third_party/mujoco_menagerie   # UR5e / 2F-85 모델
conda env list | grep gello-sim                           # python 3.11, mujoco 3.10
```

**GELLO 없이** (사인파 가짜 리더, 실제 ZMQ 경로 사용):

```bash
conda run -n gello-sim --no-capture-output env PYTHONPATH=$PWD python scripts/sim_ur5e_fake_leader.py          # MuJoCo 뷰어 창 (DISPLAY 필요)
MUJOCO_GL=egl conda run -n gello-sim --no-capture-output env PYTHONPATH=$PWD python scripts/sim_ur5e_fake_leader.py --headless   # 창 없이
```

출력 `joint[1] moved -1.570 -> -1.4xx rad` 와 `PASS: arm moved`. 뷰어 모드에서 `Timeout in ZMQLeaderServer serve`가 반복되는 것은 정상(리더 대기 루프).

**실물 GELLO로**:

```bash
conda run -n gello-sim --no-capture-output env PYTHONPATH=$PWD \
  python experiments/launch_yaml.py --left-config-path configs/ur7e_gello_sim.yaml   # 또는 configs/rwh_ur.yaml
```

`ur7e_gello_sim.yaml`은 실로봇 캘리브레이션, `rwh_ur.yaml`은 pan offset만 다르다(3.142). 시리얼 포트는 yaml의 `agent.port`.

## 알려진 한계

- MuJoCo 모델은 **UR5e**다(UR7e 아님). 링크 길이·토크가 다르므로 동작 확인용일 뿐 정량 비교 금지.
- Mock(A)은 **물리가 없다**. 충돌·접촉·그리퍼 동작은 검증되지 않는다.
- `MujocoRobotServer`는 실행 위치에 `arena.xml`을 쓴다(gitignore됨). 리포 루트에서 실행할 것(`xml_path`가 상대경로).

## 문제 해결

1. **`ROS setup not found` / `Workspace is not built`** — 위 colcon build를 먼저. zsh가 아니라 bash로 source할 것.
2. **`/joint_states`가 안 보임** — 다른 ROS 세션과 섞인 경우. `export ROS_DOMAIN_ID=77`처럼 분리하고 launch와 echo 양쪽에 같게 준다. 종료는 Ctrl-C 후 `pgrep -fa ros2_control_node`로 잔여 확인.
3. **MuJoCo 모델 파일 없음** — `third_party/mujoco_menagerie`가 비어 있다. `git submodule update --init third_party/mujoco_menagerie`.
4. **뷰어가 안 뜸 / GLFW 오류** — `DISPLAY` 확인, 없으면 `--headless` + `MUJOCO_GL=egl`.
5. **`ModuleNotFoundError: gello`** — 리포 루트에서 `PYTHONPATH=$PWD`를 붙이거나 `pip install -e .`.
