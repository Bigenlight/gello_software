# Remote Diffusion runbook

The GPU server container must already report `running healthy`. This runner does
not manage that container; it only creates a loopback SSH tunnel and starts the
existing ROS2 UR7e launch with `inference_transport:=grpc`.

On the robot laptop:

```bash
cd /home/laptop3/youngwoong_ws/gello_software/ros2_ur_ws
./setup_remote_client_venv.sh
./run_ur7e_diffusion_remote.sh
```

Run the setup script once (and again only when its lock file changes). It creates
`.venv-remote-client` with `--system-site-packages`, so ROS2 continues to use the
Ubuntu system Python packages while the remote runner puts the verified
`grpcio==1.74.0` ahead of Ubuntu 22.04's `python3-grpcio` 1.30.2. The latter was
observed accepting the forwarded TCP connection but timing out during the gRPC
handshake. Do not replace the system ROS Python or activate this venv globally.

Before launching ROS, the runner prints the loaded grpcio version and requires a
successful channel-ready handshake through the SSH tunnel. A failure stops before
any robot launch. To rebuild the isolated environment:

```bash
rm -rf .venv-remote-client
./setup_remote_client_venv.sh
```

Verify only the SSH tunnel and gRPC handshake without starting ROS or touching the
robot:

```bash
PREFLIGHT_ONLY=1 ./run_ur7e_diffusion_remote.sh
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
