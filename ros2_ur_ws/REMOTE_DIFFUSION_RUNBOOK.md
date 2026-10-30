# Remote GPU Diffusion inference: architecture and runbook

This document is the hand-off reference for the remote-inference work in
`gello_software`. Its scope is deliberately narrow: the robot laptop still owns
the existing GELLO/UR pipeline, while only Diffusion policy inference moves to a
Docker container on the GPU server.

## 1. Goal and current status

The target data flow is:

```text
robot laptop                                                GPU server
┌───────────────────────────────────────────┐               ┌─────────────────────────┐
│ cameras + UR/gripper feedback             │               │ Docker container        │
│       │                                   │  SSH tunnel   │                         │
│ policy_leader_node                        │  + gRPC       │ Diffusion model on CUDA │
│       │ observation (2 JPEGs + 7-D state) ├──────────────►│                         │
│       │                                   │◄──────────────┤ 7-D action              │
│ safety checks/clamps → bridge → controller│               │ (inference only)        │
└───────────────────────────────────────────┘               └─────────────────────────┘
```

As of 2026-07-16, the following gates have passed:

- The server image builds and imports the remote service.
- PyTorch `2.11.0+cu128` initializes CUDA with the server's unchanged NVIDIA
  550.144.03 driver and performs a tensor operation on an RTX A4000.
- The model loads, warms up, and the container reports healthy.
- An in-container synthetic inference returns one 7-D action.
- The laptop builds `ur_gello_bringup` and `gello_policy`, and imports the
  production gRPC client.
- An SSH loopback tunnel and a gRPC/HTTP2 handshake from the laptop pass.
- The production laptop client sends two synthetic black JPEGs plus the held
  7-D start state through the tunnel, the GPU executes Diffusion inference, and
  the laptop receives one 7-D action.

The verified end-to-end synthetic result included:

```text
model: Bigenlight/diffusion_banana_in_pot_joint
checkpoint: sha256:d4722b60caee5d76d004a37e16b2d7adecc1668f79703259ea7260fc9c723c57
scheduler / steps: DDIM / 10
server inference: 114.0 ms
server total: 128.4 ms
laptop round trip (including setup/contract/reset): 395.9 ms
result: PASS, one finite 7-D action
```

This proves the remote inference boundary, not physical robot motion. The ROS
fake-hardware launch has additionally reached the held start pose, completed its
strict controller hand-off, entered ROS-path ARMING then EXECUTE, and sustained
remote inference without recurrence of `RESOURCE_EXHAUSTED` or FAULT after the
laptop worker was changed to close each one-request gRPC stream at EOF before
opening the next. Returned 7-D actions were observed on `/gello/joint_states`
at approximately 9.8--10 Hz and reached
`/forward_position_controller/commands`. The node timer remains 30 Hz; in gRPC
EXECUTE it publishes only when a new inference result is ready, so the observed
topic rate follows the remote request/response cadence rather than the timer.
The controlled SSH-tunnel disconnect test also passed: terminating only the
runner-owned forwarding process closed the laptop loopback port, the next RPC
failed with gRPC `UNAVAILABLE` (`Socket closed`), the leader entered FAULT and
stopped publishing, and the bridge reported stale input about 0.703 seconds
later. A 5-second `/gello/joint_states` rate check produced no samples and timed
out with exit code 124. The forward-position controller remaining active is
expected; the safety boundary is that no new commands are published after the
leader and bridge fail silent.
Do not treat a synthetic action as safe to publish to physical hardware.

The no-device path uses
`src/gello_policy/launch/ur_control_fake_safe.launch.py`, a ROS 2 Humble-compatible
wrapper around the installed UR launch. It retains mock ros2_control and the
normal controller graph while filtering only `urscript_interface`, which is a
real-robot-only process that must not run in fake mode. The physical-hardware
path continues to include the official UR launch directly and is unchanged.

## 2. Terms for readers new to the stack

- **Client**: the laptop-side program that requests inference. It packages the
  latest cameras and robot state, sends them, validates the reply, and gives the
  action to the existing policy leader.
- **Server**: the GPU-side program that loads the checkpoint and computes an
  action. It does not connect to cameras or the robot.
- **gRPC**: the application protocol between client and server. The `.proto`
  file defines the exact request/reply fields and generates matching Python
  message/stub code. gRPC runs over HTTP/2; merely opening a TCP connection does
  not prove that a gRPC handshake works.
- **SSH tunnel**: a secure local port forward. The laptop connects to
  `127.0.0.1:50051`; SSH carries that connection to the server's own
  `127.0.0.1:50051`. The inference port is therefore not exposed publicly.
- **Docker**: isolates the large CUDA/PyTorch/LeRobot/Diffusers dependency set on
  the GPU server. ROS and hardware drivers remain native on the laptop.
- **Observation**: two JPEG camera frames and seven state values: UR joints
  `q1..q6` in radians followed by gripper position in `[0, 1]`.
- **Action**: seven returned values: six UR joint targets followed by the
  gripper command.

## 3. What changed in the repository

### Protocol and GPU service

- `src/gello_policy/proto/remote_diffusion.proto` defines protocol v1:
  `Health`, `GetServerInfo`, `ResetEpisode`, and `StreamActions`.
- `src/gello_policy/policy_server/remote_diffusion_server.py` exposes the
  existing Diffusion inference engine over gRPC and enforces session/request
  sequencing.
- `src/gello_policy/deploy/remote_diffusion/` contains the Dockerfile, Compose
  service, health check, server smoke client, environment example, and focused
  server documentation.
- The server image installs LeRobot 0.6.1 from the exact upstream commit
  `8a74e0ac6d01706d67fddfed682a09d694d9c8c0`; 0.6.1 was not available from
  PyPI when built. `pyzmq` remains installed because the reused inference engine
  imports it even though this service communicates through gRPC.

### Laptop client and ROS integration

- `src/gello_policy/gello_policy/remote_diffusion_client.py` is a ROS-free
  worker. ROS callbacks submit immutable observation snapshots while its worker
  thread owns network calls, so the ROS callback path does not wait on GPU
  inference.
- Generated protocol stubs are committed as `remote_diffusion_pb2.py` and
  `remote_diffusion_pb2_grpc.py`; regenerate them with
  `src/gello_policy/scripts/generate_remote_diffusion_stubs.sh` whenever the
  protocol changes.
- `policy_leader_node.py` accepts `inference_transport:=zmq|grpc`. The default
  remains `zmq`, preserving the previous local path. The remote runner explicitly
  selects `grpc`.
- `src/gello_policy/config/diffusion_deploy.yaml` contains the expected model
  contract and remote limits. The leader checks that the server reports exactly
  the intended model, checkpoint hash, scheduler, inference/action steps, resize
  dimensions, state/action dimensions, protocol version, and CUDA device before
  an episode is armed.
- `setup_remote_client_venv.sh` creates a small laptop-only compatibility
  environment. `run_ur7e_diffusion_remote.sh` owns the SSH tunnel, preflight,
  smoke modes, and the existing ROS launch.

### Client failure/staleness behavior

The client permits at most one request in flight and one replaceable pending
observation. Newer pending data replaces older pending data rather than forming
an unbounded queue. It validates:

- protocol, session ID, monotonically assigned request ID, dimensions, finite
  action values, and successful server status;
- camera timestamps, cross-camera skew, encoding and JPEG size;
- RPC deadline, response age, and configured model/runtime contract.

`ResetEpisode` must succeed before submission. HOLD/FAULT disarms the worker and
invalidates pending/late results. A network/server error clears the session and
is surfaced to the policy leader; it is not converted to a held action or CPU
fallback. Existing laptop-side joint limits, deviation clamps, bridge watchdog,
controller, and operator execution gate remain in authority.

UR joint positions are periodic. Before the start-pose gate, model observation,
or action-deviation clamp uses live joint feedback, the laptop expresses each
joint in the `2*pi`-equivalent branch nearest the configured `start_pose`. This
keeps the state in the checkpoint's dataset convention and prevents a wrapped
value such as wrist `+3.088` from appearing `6.283 rad` away from its equivalent
dataset value `-3.195`. A real shortest-angle deviation still fails the gate or
engages the configured clamp normally.

## 4. Server layout and configuration

The server account may be shared. Define a project-owned workspace and keep all
runtime data under it:

```text
$GPU_WORKSPACE/
├── gello_software/   # server checkout
├── models/           # checkpoints (not committed)
├── logs/             # build/runtime logs
└── runtime/          # transfer bundles or local runtime files
```

`SSH_HOST` is site-specific and has no default: export it to your own hostname
or `~/.ssh/config` alias (for example `my-gpu-box`) before running the laptop
runner. Set `GPU_WORKSPACE` to the project directory owned by the server
account. The validated server class is Ubuntu 24.04 x86_64, Docker 27.4.1, Compose 2.32.1,
NVIDIA Container Toolkit 1.17.3, eight RTX A4000 16-GB GPUs, and NVIDIA driver
550.144.03. Do not upgrade a shared host driver for this work.

The server runtime directory is:

```bash
export GPU_WORKSPACE=/path/to/server/project-workspace
cd "$GPU_WORKSPACE/gello_software/ros2_ur_ws/src/gello_policy/deploy/remote_diffusion"
```

Create `.env` from `.env.example` and keep it uncommitted. The validated values
are conceptually:

```dotenv
CHECKPOINT_DIR=/path/to/server/project-workspace/models/diffusion_banana_in_pot_joint
GPU_DEVICE=7                         # example only: choose after checking usage
INFERENCE_BIND_IP=127.0.0.1
INFERENCE_PORT=50051
IMAGE_TAG=dev
MODEL_ID=Bigenlight/diffusion_banana_in_pot_joint
CHECKPOINT_REVISION=sha256:d4722b60caee5d76d004a37e16b2d7adecc1668f79703259ea7260fc9c723c57
DIFFUSION_SCHEDULER=DDIM
DIFFUSION_NUM_INFERENCE_STEPS=10
DIFFUSION_N_ACTION_STEPS=32
```

The checkpoint directory must include `config.json`, `model.safetensors`,
`policy_preprocessor.json`, and `policy_postprocessor.json`. The observed model
directory size was about 1.1 GB. The model and `.env` are runtime data and must
not be committed.

### Build and start

Always inspect GPU occupancy and choose a physical device for that run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
GPU_DEVICE=7 docker compose --env-file .env build diffusion-server
GPU_DEVICE=7 docker compose --env-file .env up -d --no-build --force-recreate diffusion-server
```

The chosen physical GPU appears as `cuda:0` inside the container. Compose has
`restart: "no"`: after a host/daemon restart, an operator must recheck shared GPU
occupancy and deliberately select a device. Never prune Docker storage or stop
containers belonging to other users.

Verify readiness and loopback-only publication:

```bash
docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' gello-remote-diffusion
docker port gello-remote-diffusion 50051/tcp
docker compose --env-file .env logs --tail=100 diffusion-server
```

Expected essentials are `running healthy`, `127.0.0.1:50051`, a successful model
warm-up, and `ready at 0.0.0.0:50051` inside the container. Binding inside the
container to `0.0.0.0` is not public exposure; the host publish remains
loopback-only. Do not set `INFERENCE_BIND_IP=0.0.0.0` without an authenticated
private network/firewall design.

The runner intentionally does not start, stop, rebuild, or restart this server
container. Server GPU selection and lifecycle remain server-operator actions.

## 5. Laptop setup

The validated laptop is Ubuntu 22.04/ROS Humble/Python 3.10. Build the package and
create the narrow gRPC environment:

```bash
export LAPTOP_WS=/path/to/laptop/gello_software/ros2_ur_ws
cd "$LAPTOP_WS"
source /opt/ros/humble/setup.bash
colcon build --packages-up-to gello_policy
sudo apt install -y python3.10-venv        # once, only if venv support is missing
./setup_remote_client_venv.sh
```

Use `--packages-up-to`, not `--packages-select`, because `gello_policy` depends on
`ur_gello_bringup`. The isolated environment uses `--system-site-packages` so ROS
continues to use Ubuntu's packages, while the runner prepends verified
`grpcio==1.74.0`. Do not activate this venv globally and do not replace the ROS
system Python.

Ubuntu's `python3-grpcio` 1.30.2 was observed to accept the forwarded TCP socket
but time out during the gRPC handshake. An A/B test with 1.74.0 passed, which is
why the version is pinned in `requirements-remote-client.lock`. If the lock file
changes or the environment is corrupt, recreate only it:

```bash
rm -rf .venv-remote-client
./setup_remote_client_venv.sh
```

## 6. Verification ladder

Run these gates in order. A later gate does not replace an earlier diagnosis.

### Gate A: server CUDA and in-container inference

With the image built, verify CUDA on the selected GPU:

```bash
GPU_DEVICE=7 docker compose --env-file .env run --rm -T --no-deps \
  --entrypoint python diffusion-server \
  -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.ones(1,device="cuda"); print(x)'
```

Then, with the healthy service running:

```bash
docker exec gello-remote-diffusion python /app/smoke_client.py
```

This sends synthetic inputs inside the server container. It verifies model
inference but not SSH or the laptop client.

### Gate B: laptop tunnel and gRPC handshake only

```bash
cd "$LAPTOP_WS"
PREFLIGHT_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

The script may request the configured SSH account password and allows up to 60
seconds for entry. Expected output ends with:

```text
### gRPC preflight: grpcio=1.74.0, target=127.0.0.1:50051
### gRPC preflight PASS
### PREFLIGHT_ONLY=1: tunnel/gRPC verified; skipping ROS launch.
```

This mode neither starts ROS nor touches the robot.

### Gate C: laptop-to-GPU synthetic round trip

```bash
ROUNDTRIP_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

This uses the production `RemoteDiffusionWorker`, checks the exact server
contract, resets a temporary session, sends two 640x360 black JPEGs and the held
start state, performs real GPU inference, validates one 7-D response, skips ROS,
and closes the tunnel. Expected output contains both:

```text
### gRPC preflight PASS
### ROUNDTRIP_ONLY PASS: received one 7-D action; skipping ROS launch.
```

The JSON action is diagnostic only and must not be manually sent to the robot.

### Gate D: ROS fake hardware, with no robot or cameras

The repository now includes an opt-in `fake_diffusion_observations` node. The UR
driver's fake hardware publishes `/joint_states`; this node supplies only the
otherwise-missing inputs: synchronized valid black JPEGs on both configured
camera topics and fixed open-gripper feedback. It has no command publisher. The
launch starts it only when both `use_fake_hardware:=true` and
`fake_observations:=true` are explicitly set, so the physical-hardware path is
unchanged.

Build after pulling this change, then start the no-device launch:

```bash
cd "$LAPTOP_WS"
source /opt/ros/humble/setup.bash
colcon build --packages-up-to gello_policy
ROBOT_IP=127.0.0.1 ./run_ur7e_diffusion_remote.sh \
  use_fake_hardware:=true fake_observations:=true launch_rviz:=false
```

The verified Gate D launch passed the no-device bring-up through the operator
gate:

- the SSH tunnel and gRPC 1.74 preflight passed;
- the mock UR7e hardware initialized and published joint state;
- both opt-in synthetic camera/gripper observations started;
- `policy_leader_node` started in `HOLD` with the gRPC endpoint
  `127.0.0.1:50051` and did not request inference;
- the move-to-start trajectory reached the held start pose;
- the controller switched strictly from
  `scaled_joint_trajectory_controller` to `forward_position_controller`; and
- `gello_ur_bridge` resumed after alignment and the handshake exited cleanly.
- both synthetic camera topics were observed at approximately 10 Hz, gripper
  feedback was `0.0`, and all six mock UR joint positions were present.

No further output after that handshake is expected: the launch is intentionally
waiting in `HOLD` for the operator's `~/start_execution` service call.

After the move-to-start handshake succeeds and the bridge is resumed, use a
second ROS-sourced terminal to confirm inputs before arming:

```bash
source /opt/ros/humble/setup.bash
source "$LAPTOP_WS/install/setup.bash"
ros2 topic hz /cam1/cam1/color/image_raw/compressed
ros2 topic hz /cam2/cam2/color/image_raw/compressed
ros2 topic echo /robotiq_gripper/position_percent --once
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

This launch still retains the normal explicit arming gate. The synthetic images
are suitable only for transport/state-machine validation; their resulting action
has no task-performance meaning. Do not use `fake_observations:=true` with a
physical robot.

The helper node has passed package build, executable installation, JPEG decoding
checks, and the full fake-hardware handshake through the explicit HOLD gate. The
operator service call then completed ARMING -> EXECUTE, and returned actions were
observed continuously on `/gello/joint_states` at about 9.8--10 Hz and on
`/forward_position_controller/commands`. A target near `-4.12 rad` appeared at
the controller as its shortest equivalent near `+2.06 rad`; this is the expected
bridge handling across the +/-pi representation boundary, not a different
physical target. Repeated `SAFETY CLAMP` warnings are expected for black synthetic
images because their policy output has no task meaning; they confirm the normal
maximum-deviation safety path remains active. No `RESOURCE_EXHAUSTED`, stream
collision, or FAULT recurred during this sustained run.

Gate D is complete. In the controlled disconnect test, only the SSH process that
owned `127.0.0.1:50051` was terminated. The listener disappeared, the active RPC
failed with `StatusCode.UNAVAILABLE: Socket closed`, and the policy leader entered
FAULT and stopped `/gello/joint_states`. The bridge detected stale input about
0.703 seconds later and stopped publishing commands. A 5-second topic-rate probe
received no leader samples and exited 124. `forward_position_controller` remained
active, which is expected: controller lifecycle state is distinct from receiving
fresh commands, and the verified fail-safe behavior is that the leader and bridge
send no new commands.

Only after the now-complete fake-hardware gate should an operator plan
a real-hardware test with the normal UR safety procedure, held start pose, clear
workspace, pendant/e-stop access, and explicit `start_execution` service call.

## 7. Normal runner behavior (after all safety gates)

The eventual real launch command is:

```bash
cd "$LAPTOP_WS"
SSH_HOST=my-gpu-box ./run_ur7e_diffusion_remote.sh
```

`SSH_HOST` is required; the script exits immediately with an error if it is unset.

It opens the tunnel, verifies gRPC 1.74.0, sources ROS Humble and the built
workspace, then launches the existing `ur7e_diffusion_real.launch.py` with
`inference_transport:=grpc`, `act_host:=127.0.0.1`, and the selected local port.
Common overrides, alongside the required `SSH_HOST`, are:

```bash
SSH_HOST=my-gpu-box ROBOT_IP=192.168.10.11 \
LOCAL_GRPC_PORT=50051 REMOTE_GRPC_PORT=50051 \
CALIB=/path/to/ur7e_calibration.yaml START_MODE=gello \
./run_ur7e_diffusion_remote.sh launch_rviz:=false
```

`Ctrl-C` stops the ROS launch and its SSH tunnel; it does not stop the GPU
container. `HEADLESS=true` is valid only when the real robot is in Remote mode;
otherwise the External Control program must be played on the pendant. Autonomous
execution remains operator-gated:

```bash
ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
```

Do not issue this service call merely because preflight or round-trip smoke
passed. Those modes intentionally stop before ROS and robot control.

## 8. Focused troubleshooting

### TCP listener exists but gRPC times out

Check the runner's printed version. If it is 1.30.2, recreate the remote client
environment and use the runner; do not test with bare system Python:

```bash
./setup_remote_client_venv.sh
PREFLIGHT_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

### Local port is already in use

Stop the old tunnel you own, or select a different matching local port:

```bash
ss -ltnp | grep ':50051'
LOCAL_GRPC_PORT=50052 PREFLIGHT_ONLY=1 ./run_ur7e_diffusion_remote.sh
```

The remote port remains 50051. Never kill an unidentified process on the shared
server.

### Preflight cannot reach the service

On the server, verify container health, host loopback publication, logs, and the
selected GPU. On the laptop, use `ssh -v "$SSH_HOST"` for authentication diagnosis.
`ExitOnForwardFailure=yes` detects local-forward setup failure, while gRPC
preflight separately proves the application handshake.

### Model contract mismatch

Do not bypass the check. Confirm `.env`, the mounted checkpoint directory, its
SHA-256, and `diffusion_deploy.yaml`. A mismatch means the laptop and server are
not agreeing on the policy that will control the robot.

### Inference timeout or server error

The expected behavior is FAULT/disarm, not retrying stale actions, widening
timeouts blindly, holding the last target, silently changing models, or falling
back to CPU. Diagnose server logs, GPU memory/occupancy, tunnel stability, and
observation sizes before changing timing.

## 9. Git and ownership boundaries

All work is isolated on `feat/remote-diffusion-gpu-server`; modifying this branch
does not modify `main` or another branch unless it is explicitly merged. Server
copies have been transferred through Git bundles because the server checkout is
separate. Before sending another bundle, compare commit IDs and include only the
intended branch history.

Do not commit `.env`, SSH credentials, model weights, `.venv-remote-client`, logs,
or server runtime bundles. Do not refactor unrelated GELLO/UR mechanisms. Changes
to an existing mechanism are in scope only when necessary for the remote GPU
Diffusion boundary.

## 10. Handoff checklist

- [x] Remote protocol and generated Python stubs
- [x] CUDA Docker image with pinned model dependencies
- [x] Explicit per-run GPU selection and loopback-only endpoint
- [x] Model checksum/contract enforcement
- [x] Nonblocking, latest-only laptop worker with deadlines/stale validation
- [x] Optional ROS transport preserving legacy ZMQ default
- [x] Isolated compatible laptop gRPC runtime
- [x] Server CUDA and in-container model smoke
- [x] Laptop SSH/gRPC preflight
- [x] Laptop-to-server production-client synthetic inference
- [x] Opt-in fake camera/gripper observation helper builds and installs
- [x] ROS fake-hardware bring-up, held-pose handshake, and bridge resume
- [x] ROS fake-hardware HOLD-to-ARMING-to-EXECUTE transition
- [x] Sustained fake-hardware EXECUTE and returned-action topic test
- [x] SSH-tunnel disconnect-to-FAULT and bridge-staleness test under ROS
- [ ] Operator-reviewed real-hardware test plan and execution

When updating this document, record only commands and behavior confirmed by code
or observed output. Keep failed exploratory commands out of the normal procedure;
retain only the resulting diagnosis when it helps future troubleshooting.
