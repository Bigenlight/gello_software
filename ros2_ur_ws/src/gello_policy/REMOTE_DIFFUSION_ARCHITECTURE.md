# Remote Diffusion deployment architecture

## Ownership boundary

The robot laptop owns every safety-critical component: RealSense capture, UR and
Robotiq feedback, observation freshness checks, operator HOLD/EXECUTE/FAULT state,
action validation and clamps, the synthetic GELLO publishers, `gello_ur_bridge`,
the move-to-start handshake, and the ROS controllers. The GPU server cannot publish
ROS commands and has no direct route to the robot controller.

The GPU server runs only the trained JOINT Diffusion policy. It receives two
compressed JPEG frames plus the seven-element robot state and returns a seven-element
joint/gripper action.

## Process layout

```text
robot laptop (native ROS2 Humble)        GPU server (NVIDIA Docker)

RealSense + UR/Robotiq feedback
              |
              v
latest observation snapshot
              |
     asynchronous gRPC worker  ------->  RemoteDiffusion service
              ^                         JPEG decode + preprocessing
              |                         DDIM policy inference
         validated action       <-------
              |
              v
policy leader clamps -> existing GELLO/UR safety stack -> physical robot
```

The UR driver may report a periodic joint on a different `2*pi` branch from the
training dataset. The laptop maps live joint feedback to the equivalent angle
nearest `start_pose` before the arming gate, model state construction, and
action-deviation clamp. Published policy targets remain in the model/dataset
convention; this conversion does not relax genuine shortest-angle deviations.

## Transport contract

The canonical schema is `proto/remote_diffusion.proto`. Protocol v1 defines a
bidirectional `StreamActions` RPC, but the current laptop worker opens one short-lived
stream per request and permits at most one request in flight. After receiving the
single expected reply, it drains that stream to EOF before starting the next call;
this releases the server's single-stream guard deterministically. Images stay as the
existing ROS `CompressedImage` JPEG payload; raw RGB or float tensors are not sent
over the network. See `ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md` for the verified
deployment state and operating procedure.

Each request carries a monotonically increasing request ID, session ID, laptop
monotonic timestamp, camera ROS timestamps, state, and both JPEGs. A reply is usable
only when its protocol, session, and request identifiers match and the complete
round-trip age is within the laptop's action freshness limit.

Queued stale observations are forbidden. If observations 102--104 arrive while 101
is in flight, the next request uses 104. The worker does not replay 102 and 103.

## Failure behavior

The laptop enters FAULT and stops publishing the synthetic leader when any of these
occurs during EXECUTE:

- RPC deadline or connection failure;
- stale/missing robot, gripper, or camera observation;
- excessive cross-camera timestamp skew;
- protocol, session, or request ID mismatch;
- stale response;
- action with the wrong dimension or a non-finite value;
- inference worker failure.

Network recovery never resumes robot motion automatically. The server must be ready,
the policy queues must reset successfully, and the operator must explicitly arm
EXECUTE again.

## Container boundary

Only the GPU inference service is containerized initially. Its image pins Python,
PyTorch/CUDA, LeRobot, Diffusers, protobuf, and gRPC. The checkpoint is mounted
read-only and is not baked into the image. The service binds to the server loopback
interface when reached through an SSH tunnel, or to a restricted private interface
when protected by a firewall/VPN.

The native laptop ROS stack remains outside Docker so USB cameras, robot networking,
ROS controller lifecycle, RViz, and the existing safety behavior do not acquire a new
container/device/network failure boundary.

## No-device integration boundary

For no-device ROS integration, `use_fake_hardware:=true` selects a local wrapper
around the installed Humble UR launch. That installed launch creates
`urscript_interface` even in fake mode; the wrapper removes only this
real-robot-only action so it cannot retry `robot_ip:30002`. With fake hardware
disabled, the project continues to include the official launch file directly.
The wrapper uses the ROS 2 Humble launch API and handles both resolved strings
and substitution objects when identifying the executable. It was checked
against the installed package: it loads the official `ur_control.launch.py`,
generates its description, retains ordinary launch actions, and removes only
the node whose executable is `urscript_interface`.

For the pre-hardware gate, `use_fake_hardware:=true` makes the UR driver provide
mock arm state but does not provide camera or gripper observations. The optional
`fake_diffusion_observations` node fills exactly those missing inputs with two
synchronized black JPEG messages and a fixed gripper position. The launch permits
that node to run only when `fake_observations:=true` is also explicit. It does not
publish actions or replace the policy leader, bridge, controller, gRPC worker, or
server, so the inference/control path under test remains the production path.

Synthetic observations demonstrate transport and state-machine behavior, not task
quality. They must never be enabled for physical-arm operation.

The observed no-device launch has passed mock UR initialization, synthetic
observation startup, held-pose move-to-start, strict controller handover, bridge
resume, operator-gated ARMING, and EXECUTE. Returned actions were observed on
`/gello/joint_states` at about 9.8--10 Hz and at the forward-position controller.
Although the policy timer runs at 30 Hz, the gRPC path publishes only when its
nonblocking worker has a new inference result, so action topic frequency follows
the remote round-trip cadence. The bridge mapped a wrapped joint target to its
shortest physically equivalent controller angle as designed. Synthetic black
images repeatedly engaged the maximum-deviation clamp, which is expected and
confirms the production safety path is still active. After draining every
one-request response iterator to EOF, sustained execution showed no repeat of
the server's single-stream `RESOURCE_EXHAUSTED` rejection or FAULT.

The controlled disconnect gate also passed. Only the runner-owned SSH forwarding
process was terminated; the laptop loopback listener closed, the active RPC
failed with gRPC `UNAVAILABLE` (`Socket closed`), and the policy leader entered
FAULT and stopped its synthetic leader publication. The bridge observed stale
input about 0.703 seconds later and stopped publishing commands. A 5-second
`/gello/joint_states` rate probe received no samples and timed out. The forward
position controller remaining active is expected and does not imply command
flow: fail-safe control here is achieved by the leader and bridge going silent.
This completes the no-device fake-hardware gate; physical hardware remains
unverified.

## Implementation gates

1. Generate and test compatible Python stubs for Humble Python 3.10 and server
   Python 3.12.
2. Bring up the CUDA Docker service and pass Health/GetServerInfo after warm-up.
3. Validate dummy and recorded JPEG round trips without ROS or a robot.
4. Move inference I/O out of the rclpy timer into a latest-only worker.
5. Exercise timeouts, malformed actions, camera loss, and server loss.
6. Pass the full path using ROS fake hardware before any physical-arm run.

## Accepted shared-path changes from this merge

Remote diffusion is a work-in-progress, opt-in feature. It is selected explicitly
(`inference_transport: grpc` plus the runner and tunnel described in
`ros2_ur_ws/REMOTE_DIFFUSION_RUNBOOK.md`); the LOCAL ACT, LOCAL Flow-Matching, and
LOCAL Diffusion paths (`run_ur7e_act_real.sh`, `run_ur7e_fm_real.sh`,
`run_ur7e_diffusion_real.sh`, all of which speak ZMQ to a policy server on this
laptop) continue to run without a GPU server. Everything the merge added is gated
behind that explicit selection, with two deliberate exceptions recorded here.

Both exceptions were reviewed and knowingly ACCEPTED as un-gated rather than hidden
behind `inference_transport == 'grpc'`. They therefore change SHARED production code
and apply to every transport. They are documented here so that a future maintainer
debugging an arming refusal or a latency regression has a paper trail instead of only
an inline code comment.

### 1. Wrap-aware joint comparison in `policy_leader_node.py` (all transports, including ZMQ)

Two call sites changed:

- The `~/start_execution` arming gate (around line 381) now computes
  `angular_deviations(self._live_q, self._start_pose)` instead of the previous raw
  `abs(self._live_q[i] - self._start_pose[i])`.
- `_tick_execute` (around line 476) now computes
  `live_q = positions_near_reference(self._live_q, self._start_pose)` instead of
  `live_q = list(self._live_q)`. That wrapped vector is BOTH the state sent to the
  LOCAL ZMQ ACT/FM/Diffusion server AND the reference for the per-joint
  `max_dev_rad` action clamp.

The helpers live in `gello_policy/joint_angles.py`, and reduce to
`nearest_equivalent(angle, reference) = reference + math.remainder(angle - reference, math.tau)`.

Why it is safe: `math.remainder(d, tau)` reproduces `d` with exactly zero
floating-point error whenever `|d| < pi`, so for any joint already within `+/-pi` of
its `start_pose` reference the new code is bit-identical to the old. It differs ONLY
at a wrap boundary.

Why it was kept rather than gated behind gRPC: the shipped default `start_pose` is
`[3.106, -1.817, 1.653, -1.618, -1.628, -3.195]`, in which shoulder_pan sits 0.036 rad
from `+pi` and wrist_3 sits 0.053 rad past `-pi`. A `/joint_states` publisher reporting
wrist_3 as its positive equivalent (about +3.088 rad) previously registered a spurious
~`2*pi` deviation, which falsely REFUSED arming.

More importantly, the rest of this workspace had ALREADY committed to branch-cut-aware
comparison. `ur_gello_bringup/angle_utils.py` states the rule outright ("Every angular
COMPARISON must therefore be circular"), `gello_move_to_start_node.py` parks the arm via
`wrapped_nearest(target, actual_pose)` — i.e. on whichever revolution is nearest the
arm — and `gello_ur_bridge_node.py:451` re-anchors every outgoing target onto the
actual-pose branch before publishing to `/forward_position_controller/commands`.
Leaving `policy_leader_node` comparing raw values made it the ONLY node in the chain
using a non-circular comparison, against a `start_pose` that its own upstream node may
legitimately park on the far branch.

Note what this does NOT do: normalizing `live_q` cannot cause a full-revolution spin.
The clamped action is an absolute joint target, and `gello_ur_bridge_node` re-anchors it
to the arm's actual pose (`wrapped_nearest`, line 451) before it reaches the controller,
so the `2*pi` is stripped downstream. Sending the `start_pose`-branch representation is
also what the policy expects, since the checkpoint was trained on a dataset recorded in
that branch; feeding the raw far-branch value would be out-of-distribution.

Residual risk to check: the arming gate is now permissive in a case where it used to
be conservative. This is worth one deliberate arming test at the shipped `start_pose`.

### 2. Timing instrumentation in `policy_server/diffusion_server.py` (LOCAL diffusion hot loop)

`DiffusionInferenceEngine.act()` gained `t_start`/`t_preprocessed`/`t_inferred`/
`t_finished` timestamps, an explicit `if self.device == "cuda": torch.cuda.synchronize()`
between `select_action()` and the postprocess step, and a `self.last_act_metadata`
dict that only the remote gRPC wrapper consumes.

Accepted un-gated because the immediately following `action.squeeze(0).cpu().numpy()`
already forced the identical device synchronization: the explicit `synchronize()`
moves WHERE the wait happens by two statements, not how long it is. On the ZMQ path
`last_act_metadata` is write-only, and the return type of `act()` is unchanged, so
the validated LOCAL ZMQ deploy stays backward compatible.

Residual risk to check: during the next LOCAL Diffusion session, confirm that
per-tick latency is unchanged within noise and that `act_timeout_s: 0.6`
(`config/diffusion_deploy.yaml`) is still comfortably met. Note that this file runs
in the separate Python 3.12 torch/LeRobot venv, so it is covered by neither
`colcon build` nor the package's pytest suite; only an empirical run exercises it.

### Build note

`gello_policy/package.xml` gained two `exec_depend` entries, `python3-grpcio` and
`python3-protobuf`, for the opt-in remote path. Both are already present on the current
robot PC (protobuf 3.12.4, grpcio 1.30.2), so nothing breaks today; the declaration
matters for a fresh machine.

`ros2_ur_ws/build_ur7e.sh` runs `rosdep install --from-paths src --ignore-src -r -y
--skip-keys dynamixel_sdk`, which covers EVERY package under `src/` and therefore does
resolve the two new keys. But its `colcon build --packages-select ur_gello_bringup`
builds only that one package — it does NOT build `gello_policy`. On a fresh machine run
both steps:

```bash
./build_ur7e.sh                                   # rosdep for all packages + ur_gello_bringup
colcon build --packages-select gello_policy       # then gello_policy itself
```
