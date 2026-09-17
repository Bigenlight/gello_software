# UR driver 3.8.0 — Jazzy 포팅 검토 (2026-09-16)

대상은 이 PC에 설치된 `ur_robot_driver` / `ur_controllers` 3.8.0,
`forward_command_controller` / `position_controllers` 4.42.1이다.
실물 로봇의 운용 검증과 소스 검토·mock 검증은 구분한다.

## 캘리브레이션 인자: 기존 핸드오프의 유실 주장은 철회

`ur_control.launch.py`가 `kinematics_params_file`을 직접 선언하지 않는 것은 맞다.
그러나 ROS launch의 `IncludeLaunchDescription`은 선언되지 않은 인자도
`SetLaunchConfiguration`으로 설정한다. 하위 `ur_rsp.launch.py`는 같은 context를
상속하고, `DeclareLaunchArgument`의 기본값은 이미 설정된 값을 덮어쓰지 않는다.
따라서 별도 description 래퍼는 필요하지 않다.

`gello_policy/test/test_jazzy_description_contract.py`는 실제 설치된 launch를
로드하고, 별도 calibration YAML의 고유 hash가 xacro로 생성한 최종 URDF에
남는지 검사한다. policy 런치와 GELLO 런치 모두 통과했다.
노드나 로봇 프로세스는 실행하지 않는 테스트다.

근거: 설치된
`/opt/ros/jazzy/lib/python3.12/site-packages/launch/actions/include_launch_description.py`,
`/opt/ros/jazzy/share/ur_robot_driver/launch/{ur_control,ur_rsp}.launch.py`.

## 활성 컨트롤러와 STRICT 전환

설치된 3.8.0의 기본 활성 목록에는 `gravity_update_controller`가 없다.
추가 활성 컨트롤러는 `friction_model_controller`다. 이 컨트롤러는
`friction_model/viscous_0..5`, `friction_model/coulomb_0..5`,
`friction_model/async_success`를 사용하며 `<joint>/position`을 점유하지 않는다.
따라서 이 이유로 SJTC→FPC 전환의 deactivate 목록을 늘릴 필요는 없다.

프로젝트 `gello_move_to_start_node.py`는 Jazzy에서도 존재하는
`activate_controllers`, `deactivate_controllers`, `strictness=2`,
`activate_asap`, `timeout` 필드를 사용한다. `home_move_ros.py`의 기존 변경은
주석뿐이며 동작 변경이 없다.

근거: 설치된 `ur_control.launch.py`, `ur_controllers.yaml`,
[3.8.0 FrictionModelController](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/3.8.0/ur_controllers/src/friction_model_controller.cpp).

## SJTC→FPC 명령 재시딩

UR hardware interface의 `perform_command_mode_switch`는 position 모드를
멈출 때와 시작할 때 모두 현재 측정 관절각을 명령 버퍼와 이전 명령 버퍼로 복사한다.
이 동작은 2.13.2와 3.8.0 양쪽에 있다.

FPC 4.42.1은 activation 때 내부 reference 메시지를 NaN으로 초기화하며,
새 메시지가 없어 모든 reference가 NaN이면 `update()`가 명령 인터페이스에
쓰지 않고 반환한다. 따라서 소스상 전환 직후 첫 명령 전에는 UR 드라이버가
재시딩한 측정 위치가 유지된다. 예전 2.x 메모의 `nullptr` 구현 설명을
4.42.1에 그대로 적용하지 않는다.

근거:
- [UR 2.13.2 hardware interface](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/2.13.2/ur_robot_driver/src/hardware_interface.cpp)
- [UR 3.8.0 hardware interface](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/3.8.0/ur_robot_driver/src/hardware_interface.cpp)
- [FPC 4.42.1](https://github.com/ros-controls/ros2_controllers/blob/4.42.1/forward_command_controller/src/forward_controllers_base.cpp)

소스 검토는 실제 측정 지연, External Control 연결, 첫 bridge 명령의 연속성을
증명하지 않는다. mock 시스템은 실제 UR hardware interface를 사용하지 않으므로
mock 전환 성공도 이 실기 재시딩의 직접 측정으로 간주하지 않는다.

## mock 런치

2026-09-16 isolated ROS domain 173에서 `mock_components/GenericSystem`을
기동했다. SJTC active 상태에서 FPC activate / SJTC deactivate,
`strictness=2`, `activate_asap=true`, timeout 5초 요청은 `ok=True`였다.
후속 목록은 FPC active / SJTC inactive / friction controller active였다.
실험 프로세스는 종료했고 로그는 `ros2_ur_ws/log/jazzy_port_20260916/mock_switch.log`에 보존했다.

`run_mock_rviz.sh`는 Jazzy에서 `use_mock_hardware`, Humble opt-in에서
`use_fake_hardware`를 선택한다. `ur_control_fake_safe.launch.py`의 기존
URScript 필터는 Humble 호환을 위해 유지했다. Jazzy upstream은 이미
`UnlessCondition(use_mock_hardware)`로 URScript 노드를 비활성화한다.

실기 전 남는 확인은 실제 펜던트 버전/External Control 설정, 두 카메라 연결,
로봇 네트워크, 실제 핸드셰이크와 첫 명령 연속성이다.
