# Remote Diffusion runbook

The GPU server container must already report `running healthy`. This runner does
not manage that container; it only creates a loopback SSH tunnel and starts the
existing ROS2 UR7e launch with `inference_transport:=grpc`.

On the robot laptop:

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
./run_ur7e_diffusion_remote.sh
```

Common overrides:

```bash
SSH_HOST=kanu ROBOT_IP=192.168.10.11 \
LOCAL_GRPC_PORT=50051 REMOTE_GRPC_PORT=50051 \
CALIB=/path/to/ur7e_calibration.yaml START_MODE=gello \
./run_ur7e_diffusion_remote.sh launch_rviz:=false
```

Use `HEADLESS=true` only when the real robot is in Remote mode. Otherwise play
the External Control program on the pendant. The arm parks at the held start
pose; autonomous execution still requires the explicit service call:

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

Ctrl-C stops the ROS launch and its SSH tunnel. It does not stop the GPU server.
