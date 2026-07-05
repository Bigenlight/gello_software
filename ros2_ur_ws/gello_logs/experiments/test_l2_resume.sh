#!/usr/bin/env bash
# End-to-end test of the L2 paused-bridge -> resume handoff (the reviewer's B1 gap):
# bridge pre-spawned start_paused:=true must publish NOTHING until move_to_start
# (resume_bridge:=true) resumes it AFTER the switch. Real ros2_control mock stack.
set -o pipefail
export ROS_DOMAIN_ID=62
source /opt/ros/humble/setup.bash
source /home/theo/gello_software/ros2_ur_ws/install/setup.bash
EXP=/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments
L=$EXP/l2_resume

cleanup(){ pkill -f gello_ur_bridge 2>/dev/null; pkill -f gello_move_to_start 2>/dev/null
  pkill -f gello_pub.py 2>/dev/null; pkill -f ur_control.launch 2>/dev/null
  pkill -9 -f ros2_control_node 2>/dev/null; sleep 2; }
trap cleanup EXIT
cleanup

echo "### 1. bring up mock stack"
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur7e robot_ip:=127.0.0.1 \
  use_fake_hardware:=true launch_rviz:=false headless_mode:=true \
  initial_joint_controller:=scaled_joint_trajectory_controller > ${L}_stack.log 2>&1 &
for i in $(seq 1 40); do
  ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers 2>/dev/null \
    | grep -q "joint_state_broadcaster', state='active'" && break; sleep 1; done
echo "### 2. activate joint_trajectory_controller"
ros2 run controller_manager spawner joint_trajectory_controller -c /controller_manager --controller-manager-timeout 15 >>${L}_stack.log 2>&1

echo "### 3. start bridge PAUSED (start_paused:=true)"
ros2 run ur_gello_bringup gello_ur_bridge --ros-args \
  -p start_paused:=true -p filter_type:=one_euro -p max_step_rad:=0.0025 \
  -p publish_rate_hz:=250.0 -p soft_start_s:=0.7 -p resume_align_tol:=0.05 > ${L}_bridge.log 2>&1 &
sleep 2
echo "### 4. start gello leader (quiet)"
/usr/bin/python3 $EXP/gello_pub.py --scenario quiet > ${L}_gello.log 2>&1 &
sleep 1

echo "### 5. ASSERT bridge SILENT while paused (2s, expect 0 msgs)"
PAUSED_MSGS=$(/usr/bin/python3 $EXP/count_topic.py 2.0 2>/dev/null | grep "^COUNT" | awk '{print $2}')
echo "PAUSED_CMD_COUNT=$PAUSED_MSGS  (expect 0)"

echo "### 6. run move_to_start with resume_bridge:=true"
timeout 40 ros2 run ur_gello_bringup gello_move_to_start --ros-args \
  -p start_mode:=gello -p source_controller:=joint_trajectory_controller \
  -p target_controller:=forward_position_controller \
  -p resume_bridge:=true -p bridge_resume_service:=/gello_ur_bridge/resume \
  -p chase_timeout_s:=20.0 -p chase_tol:=0.03 -p chase_dwell_s:=0.4 \
  -p chase_v_budget:=0.5 -p min_traj_duration:=0.5 -p activation_timeout:=15.0 > ${L}_mts.log 2>&1
MTS_EXIT=$?
echo "MTS_EXIT=$MTS_EXIT  (expect 0)"

echo "### 7. ASSERT bridge NOW streaming (3s, expect >0 msgs)"
STREAM_OUT=$(/usr/bin/python3 $EXP/count_topic.py 3.0 2>/dev/null)
STREAM_MSGS=$(echo "$STREAM_OUT" | grep "^COUNT" | awk '{print $2}')
echo "STREAM_CMD_COUNT=$STREAM_MSGS  (expect >0); $(echo "$STREAM_OUT" | grep '^LAST')"

echo "### 8. key log lines"
grep -E "Converged|switch OK|Bridge resumed|did NOT resume|not available" ${L}_mts.log | tail -4
grep -E "RESUMED|refus|REFUS|Following" ${L}_bridge.log | tail -3

echo "### VERDICT"
if [ "$PAUSED_MSGS" -eq 0 ] && [ "$MTS_EXIT" -eq 0 ] && [ "$STREAM_MSGS" -gt 0 ]; then
  echo "L2_RESUME_PASS: paused-silent -> handshake -> resume -> streaming"
else
  echo "L2_RESUME_FAIL: paused=$PAUSED_MSGS mts_exit=$MTS_EXIT stream=$STREAM_MSGS"
fi
