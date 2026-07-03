# ur_gello_bringup

ROS2 (Humble/Jazzy) `ament_python` package that teleoperates a Universal
Robots arm (ur5e/ur7e) from a GELLO leader arm, with RViz visualization.

## Nodes

- **gello_publisher** (`gello_publisher_node:main`) — reads the GELLO Dynamixel
  leader arm over serial and publishes `sensor_msgs/JointState` on
  `/gello/joint_states` (6 arm joints in UR order) plus the gripper width on
  `/gripper/gripper_client/target_gripper_width_percent` (`std_msgs/Float32`, 0..1).
- **gello_ur_bridge** (`gello_ur_bridge_node:main`) — subscribes to
  `/gello/joint_states`, applies EMA smoothing / step limiting / staleness
  guarding / a `deadband_rad` noise gate (suppresses at-rest Dynamixel/hand
  tremor), seeds its first command from the robot's **actual `/joint_states`**
  (not the GELLO pose) to avoid a start-up snap (this fixed a real "External
  Control speed limit" protective stop), and publishes
  `/forward_position_controller/commands`
  (`std_msgs/Float64MultiArray`, 6 doubles, UR order).
- **robotiq_urcap** (`robotiq_urcap_node:main`) — drives the Robotiq gripper
  via the UR URCap from the gripper width topic.
- **fake_gello** (`fake_gello_node:main`) — publishes synthetic
  `/gello/joint_states` for testing the pipeline in RViz without hardware.
- **gello_move_to_start** (`gello_move_to_start_node:main`) — one-shot
  handshake used on the real robot: drives the arm to the **current live GELLO
  pose** (read from `/gello/joint_states`, reordered by name) via the
  `scaled_joint_trajectory_controller`'s `FollowJointTrajectory` action, then
  STRICT-switches control to `forward_position_controller` for streaming
  teleop. (Target is the live GELLO pose, NOT `start_joints`.) Not needed on
  the fake/mock path.

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
cd /home/laptop3/gello_software/ros2_ur_ws
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

### Real GELLO -> UR5e mock (RViz)

Reads the physical GELLO leader arm and drives **mock** hardware in RViz.
Config (serial port, joint offsets/signs, gripper) is in `config/ur_gello.yaml`.

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
ros2 launch ur_gello_bringup ur_gello_rviz.launch.py source:=gello
```

> Note: `ur_gello_rviz.launch.py` defaults to `use_fake_hardware:=true` (and a
> dummy `robot_ip:=192.168.56.101`), so the command above drives mock/RViz
> hardware from the real GELLO — it does **not** move a physical UR5e. To
> command a real UR5e, override the args, e.g.
> `... source:=gello use_fake_hardware:=false robot_ip:=<real-UR5e-ip>`, and run
> on the UR control PC (the machine connected to the robot and the GELLO serial
> adapter).

## UR7e (Humble)

UR7e support reuses the same nodes unchanged (ur5e and ur7e share identical
joint limits) via a dedicated `ur7e_gello_rviz.launch.py` (fake/mock) and
`ur7e_gello_real.launch.py` (real robot) launch pair, plus the new
`gello_move_to_start` node for a safe handshake before streaming teleop.

> Note: with `ros-humble-ur` 2.8.1, ur7e visuals in RViz currently render the
> ur5e mesh (`config/ur7e` visual_parameters point at `meshes/ur5e`, not yet
> updated upstream) and a missing `ur7e_update_rate.yaml` produces a harmless
> warning; joint-space teleop itself is unaffected. Upgrade to
> `ros-humble-ur` >= 2.13.2 for accurate ur7e visuals/kinematics.

**Safety invariant:** GELLO is always a passive, read-only input device —
its Dynamixel torque is never enabled, and it never receives commands.

1. **Fake verify** (mock hardware, no GELLO/robot attached — this is the
   path used to validate the bringup on a dev machine):

   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake
   ```

2. **Real GELLO -> ur7e mock** (physical GELLO leader arm driving mock
   hardware/RViz, no real robot):

   ```bash
   export GELLO_REPO_ROOT=/home/laptop3/gello_software
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello
   ```

3. **Real robot** (physical ur7e; runs on the UR control PC):

   ```bash
   export GELLO_REPO_ROOT=/home/laptop3/gello_software
   ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP>
   ```

See `../../../docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md` for the full pipeline
diagram, node-role/topic-contract tables, and troubleshooting.
