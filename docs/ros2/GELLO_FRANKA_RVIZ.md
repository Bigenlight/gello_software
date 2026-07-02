# GELLO로 Franka를 RViz2에서 보기

GELLO(passive leader)를 흔들면 RViz2 안의 Franka 모델이 1:1로 따라오게 만드는 방법을 정리한 문서입니다. 실제 로봇 없이도 캘리브레이션과 텔레오퍼레이션 파이프라인을 눈으로 검증할 수 있습니다.

> **안전 (변함없음):** GELLO는 항상 **passive(읽기 전용)**입니다. GELLO Dynamixel(XL330)에 토크/파워를 절대 인가하지 않습니다. 여기서 소개하는 모든 ROS2 스택은 `dynamixel_torque_enable: [0]*8`로 토크를 비활성화하고, 종료 시에도 `disable_torque()`를 호출합니다. GELLO는 오직 관절 각도를 **읽기만** 합니다.

---

## 어느 Franka를 돌리는가

용어를 반드시 구분합니다. "Franka"라고만 쓰지 않고 항상 어떤 기체·어떤 경로인지 명시합니다.

| 기체 / 경로 | 무엇인가 | 언제 쓰나 |
|---|---|---|
| **Panda (sim)** | MuJoCo 시뮬레이션의 Panda. `configs/rwh_panda.yaml` + `experiments/launch_yaml.py` (ZMQ, ROS2 아님) | 로컬에서 가장 빠르게 텔레오퍼레이션을 보고 싶을 때. → [docs/sim/GELLO_PANDA_SIM_TELEOP.md](../sim/GELLO_PANDA_SIM_TELEOP.md) |
| **FR3 (repo `ros2/`, upstream reference)** | 이 리포 `ros2/` 폴더의 공식 `franka_ros2` 스택. **FR3 전용** (joint 이름 `fr3_joint1..7` 하드코딩, FR3 임피던스 게인). RViz에는 **FR3 모델**이 뜸 | 실제 FR3를 붙이거나 upstream 구조를 참고할 때만. **우리 랩 Panda와는 다른 로봇** |
| **Panda (minje227 docker, RViz)** | 조민제 도커 이미지 `minje227/gello:RWH_Gello_PandaV1.0`의 `multipanda_ros2` 스택. `rwh_gello_rviz.launch.py` | **RViz에 실제 Panda를 띄우는 유일한 턴키 경로.** 이 문서의 1순위 | 

> 우리 랩 기체는 **Panda**입니다 (`panda.xml`, `configs/rwh_panda.yaml`, `panda_arm.urdf.xacro`). 따라서 RViz에서 **우리 기체를 그대로** 보려면 아래 **PRIMARY 경로(도커)**를 씁니다. 리포 `ros2/`(FR3)는 다른 로봇 모델이라는 점을 항상 기억하세요.

---

## PRIMARY 경로 — 조민제 도커 (RViz에 실제 Panda)

**이것이 권장 경로입니다.** 이유: RViz 안에 뜨는 모델이 **실제 우리 Panda**이고, GELLO를 손으로 흔들면 그 Panda가 1:1로 실시간 미러링합니다. 캘리브레이션(`rwh_panda_ros2.yaml`)도 우리 값 그대로라 재측정이 필요 없습니다. 자세한 배경·비교는 [GELLO_ROS2_CONTROL_REFERENCE.md](./GELLO_ROS2_CONTROL_REFERENCE.md) §4~5를 참고하세요.

### 전제 조건 / 주의 (반드시 읽기)

- **이 이미지는 "현재 경로"일 뿐, 최종본이 아닙니다.** 사용자 명시 사항: `minje227/gello:RWH_Gello_PandaV1.0`은 이후 **수정 + 새로 rebuild/재배포가 필요**합니다. 지금 당장 RViz로 Panda를 보는 데는 충분하지만, 확정된 배포본으로 취급하지 마세요.
- **오프라인·완전 로컬이 아닙니다.** 이 경로가 돌아가려면 (1) private 도커 이미지 `minje227/gello:RWH_Gello_PandaV1.0`에 접근 가능해야 하고, (2) **물리 GELLO가 실제로 꽂혀 있어야** 합니다(publisher가 시리얼로 GELLO를 읽음). 이미지가 없거나 GELLO가 없으면 이 경로는 쓸 수 없습니다.
- **GELLO는 여전히 passive.** 도커 스택도 토크 비활성 3중 안전(config `[0]*8` + 코드 기본값 + 종료 훅)을 유지합니다.

### 실행 (docker run + launch)

아래 명령은 [GELLO_ROS2_CONTROL_REFERENCE.md](./GELLO_ROS2_CONTROL_REFERENCE.md) §5-A에서 검증된 형태를 그대로 옮긴 것입니다. `DISPLAY`는 본인 세션의 X 디스플레이로 바꾸세요(일반 모니터면 보통 `:0`; 이 헤드리스 빌드 PC 예시는 `:1`).

```bash
xhost +local:docker
docker run --rm -it --privileged \
  --device=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0 \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  minje227/gello:RWH_Gello_PandaV1.0 bash -c "
    source /workspace/panda_ros2_ws/install/setup.bash && \
    source /gello/ros2/install/setup.bash && \
    ros2 launch franka_bringup rwh_gello_rviz.launch.py \
      gello_config_file:=rwh_panda_ros2.yaml \
      gello_com_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO75A-if00-port0 \
      launch_gello:=true"
```

- `--device=...`의 `by-id` 경로는 replug/재부팅에도 안정적입니다. 다른 어댑터를 쓰면 `ls -l /dev/serial/by-id/`로 본인 경로를 찾아 두 군데(`--device`, `gello_com_port`) 모두 바꾸세요. Dynamixel baud는 57600.
- 내부적으로 `use_fake_hardware:=true`, `robot_ip:=dummy`로 동작하므로 **실제 Panda가 없어도** RViz만으로 확인됩니다.
- 컨트롤러까지(게인·move-to-start·staleness) 파이프라인 전체를 로봇 없이 점검하려면 `rwh_gello_realtime.launch.py`에 `use_fake_hardware:=true fake_sensor_commands:=true use_rviz:=true`를 주는 full-stack fake-hardware 모드가 있습니다 (control-reference §5-B).

### 무엇이 보여야 하는가

- RViz2 창이 열리고 **Panda** 모델이 표시됩니다.
- GELLO를 손으로 움직이면 RViz 속 Panda 팔 7축이 **1:1로 즉시 추종**합니다.
- GELLO 그리퍼 레버를 여닫으면 Panda 핸드가 열리고 닫힙니다.

> docker run에 들어가는 세부 인자·게인·대시보드 스크립트(`rwh_gello_realtime_dashboard.sh`) 등 이 문서에 없는 항목은 [GELLO_ROS2_CONTROL_REFERENCE.md](./GELLO_ROS2_CONTROL_REFERENCE.md)를 보세요.

---

## SECONDARY / 참고 경로 — 리포 `ros2/` FR3 스택 (devcontainer)

> **경고 — 이건 다른 로봇입니다.** 리포 `ros2/` 폴더는 upstream 공식 `franka_ros2` 스택이고 **FR3 전용**입니다. joint 이름이 `fr3_joint1..7`로 하드코딩되어 있고 FR3 임피던스 게인(J1–7 `[240,240,240,240,100,60,20]`)을 씁니다. 여기서 RViz를 띄우면 **FR3 모델**이 뜹니다 — **우리 랩 Panda가 아닌 다른 로봇**입니다. 실제 FR3를 붙이거나 upstream 구조를 참고할 때만 쓰세요. FR3 게인을 Panda에 섞으면 위험합니다.

### 왜 도커(devcontainer)인가

이 경로는 **docker(devcontainer) 사용을 권장**합니다. 이유: `franka_ros2 v2.1.0` / `libfranka 0.18.2` / `franka_description 1.3.0`이 **ROS2 Humble / Ubuntu 22.04에 하드-락**되어 있기 때문입니다. 우리 빌드 PC는 Ubuntu 24.04(ROS2 Jazzy)라 네이티브로는 버전이 맞지 않습니다. `ros2/` 서브폴더를 VS Code에서 열어 **"Reopen in Container"**를 선택하면 됩니다(자세한 절차는 `ros2/README.md`의 Setup Environment 참고).

> 참고로 **ROS2 UR 경로는 도커를 쓰지 않습니다** — UR 드라이버는 최신 Ubuntu에서 잘 동작해 네이티브 Jazzy로 갑니다([GELLO_UR_ROS2_BRINGUP.md](./GELLO_UR_ROS2_BRINGUP.md)). 도커가 필요한 건 이 FR3 스택처럼 옛 버전에 락된 경우뿐입니다.

### fake-hardware로 FR3를 RViz에 (로봇 없이)

컨테이너 안에서 workspace를 빌드/소스한 뒤 fake-hardware로 돌리면 **실제 FR3 없이** RViz에서 FR3 모델을 확인할 수 있습니다. 단, `franka_fr3_arm_controllers.launch.py`는 CLI 인자로 **`robot_config_file` 하나만** 받습니다(파일 line 110–114). `use_fake_hardware`와 `use_rviz`는 CLI가 아니라 **config YAML 안의 키**로 읽히므로(launch 파일이 각각 line 60·85에서 config에서 꺼내 씀), `:=`로 넘기면 무시됩니다. 따라서 YAML을 편집해서 켜야 합니다:

```bash
# ros2/ devcontainer 안에서, colcon build && source install/setup.bash 이후

# 1) config YAML을 열어 두 키를 "true"로 편집
#    ros2/src/franka_fr3_arm_controllers/config/example_fr3_config.yaml
#      use_fake_hardware: "true"
#      use_rviz: "true"
#    (원본을 보존하려면 example_fr3_config.yaml을 fake 전용 파일로 복사해 편집한 뒤
#     아래 robot_config_file:= 에 그 파일 이름을 넘겨도 됩니다)

# 2) 편집한 config로 launch (robot_config_file 만 CLI 인자로 존재)
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_config.yaml
```

> ⚠️ `franka_fr3_arm_controllers.launch.py`에 `use_fake_hardware:=true use_rviz:=true`를 **직접 주면 동작하지 않습니다.** 이 두 인자는 이 launch 파일에 **선언되어 있지 않아** `ros2 launch`가 경고만 내고 무시하며, 기본 config(`example_fr3_config.yaml`, `use_fake_hardware:"false"`·`use_rviz:"false"`)를 그대로 로드해 **실제 로봇 IP로 접속을 시도하고 RViz도 뜨지 않습니다.** 반드시 위처럼 YAML을 편집하세요. 이는 [GELLO_ROS2_CONTROL_REFERENCE.md](./GELLO_ROS2_CONTROL_REFERENCE.md)의 "YAML에서 `use_fake_hardware:=true` 편집" 설명과 동일한 방식입니다.
>
> `use_fake_hardware`/`use_rviz`를 **CLI `:=` 인자로 받는 것은 형제 파일 `franka.launch.py`뿐**입니다(위 상위 launch가 내부적으로 include하는 파일). 여기서는 그 파일을 직접 호출하지 않으므로 config YAML 편집이 유일한 경로입니다.

### 실제 GELLO → FR3 텔레오퍼레이션 (참고)

실제 FR3 로봇에 GELLO를 붙이는 3-노드 루틴(publisher → 임피던스 컨트롤러 → gripper manager)은 `ros2/README.md`의 "Getting Started" / "Detailed Launch Routine"에 있습니다. 요지만:

```bash
# ros2/ 컨테이너 안, colcon build && source install/setup.bash 이후
ros2 launch franka_gello_state_publisher main.launch.py config_file:=franka_gello_single.yaml
ros2 launch franka_gripper_manager franka_gripper_client.launch.py config_file:=example_fr3_config_franka_hand.yaml
# GELLO를 편한 자세로 잡은 뒤 (로봇이 그 자세로 부드럽게 이동)
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:=example_fr3_config.yaml
```

이 경로는 실제 FR3(FCI 활성, `robot_ip` 설정)와 GELLO가 모두 있어야 하고, **FR3용 캘리브레이션·게인**을 씁니다. 우리 Panda 값과 섞지 마세요.

---

## 요약: 어느 경로를 고를까

- **RViz에서 우리 실제 Panda를 GELLO로 보고 싶다** → **PRIMARY(도커, minje227)**. 단 private 이미지 + 물리 GELLO 필요, 이미지는 추후 수정/재배포 예정.
- **가장 빠르게 텔레오퍼레이션만 보고 싶다(ROS2 불필요)** → **Panda (sim)**, [GELLO_PANDA_SIM_TELEOP.md](../sim/GELLO_PANDA_SIM_TELEOP.md).
- **실제 FR3를 붙이거나 upstream 구조 참고** → **SECONDARY(리포 `ros2/`, FR3)**. RViz에 뜨는 건 FR3 모델(우리 Panda 아님).

두 ROS2 스택의 상세 비교·데이터 흐름·마이그레이션 로드맵은 [GELLO_ROS2_CONTROL_REFERENCE.md](./GELLO_ROS2_CONTROL_REFERENCE.md)에 있습니다.
