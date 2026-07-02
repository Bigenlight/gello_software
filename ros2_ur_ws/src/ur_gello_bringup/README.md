# ur_gello_bringup

ROS2 (Jazzy) `ament_python` package that teleoperates a Universal Robots UR5e
from a GELLO leader arm, with RViz visualization.

## Nodes

- **gello_publisher** (`gello_publisher_node:main`) — reads the GELLO Dynamixel
  leader arm over serial and publishes `sensor_msgs/JointState` on
  `/gello/joint_states` (6 arm joints in UR order) plus the gripper width on
  `/gripper/gripper_client/target_gripper_width_percent` (`std_msgs/Float32`, 0..1).
- **gello_ur_bridge** (`gello_ur_bridge_node:main`) — subscribes to
  `/gello/joint_states`, applies EMA smoothing / step limiting / staleness
  guarding, and publishes `/forward_position_controller/commands`
  (`std_msgs/Float64MultiArray`, 6 doubles, UR order).
- **robotiq_urcap** (`robotiq_urcap_node:main`) — drives the Robotiq gripper
  via the UR URCap from the gripper width topic.
- **fake_gello** (`fake_gello_node:main`) — publishes synthetic
  `/gello/joint_states` for testing the pipeline in RViz without hardware.

### Topics

| Topic | Type | Notes |
| --- | --- | --- |
| `/gello/joint_states` | `sensor_msgs/JointState` | 6 arm joints, UR order, radians |
| `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` | 0..1 |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 6 doubles, UR order |

UR joint order:
`[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`

## Build

```bash
cd /home/theo_lab/gello_software/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

## Run

### Fake GELLO (RViz test, no hardware)

Publishes synthetic joint states so you can verify the pipeline and see the
UR5e move in RViz without a real leader arm or robot.

```bash
ros2 launch ur_gello_bringup ur_gello_rviz.launch.py
```

### Real GELLO -> UR5e

Reads the physical GELLO leader arm and commands the real UR5e. Config
(serial port, joint offsets/signs, gripper) is in `config/ur_gello.yaml`.

```bash
ros2 launch ur_gello_bringup ur_gello_rviz.launch.py source:=gello
```

> Note: real-robot usage runs on the UR control PC (the machine connected to
> the UR5e and the GELLO serial adapter), not necessarily this development
> machine.
