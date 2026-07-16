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

## Transport contract

The canonical schema is `proto/remote_diffusion.proto`. Protocol v1 uses one
persistent bidirectional gRPC stream with at most one request in flight. Images stay
as the existing ROS `CompressedImage` JPEG payload; raw RGB or float tensors are not
sent over the network.

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

## Implementation gates

1. Generate and test compatible Python stubs for Humble Python 3.10 and server
   Python 3.12.
2. Bring up the CUDA Docker service and pass Health/GetServerInfo after warm-up.
3. Validate dummy and recorded JPEG round trips without ROS or a robot.
4. Move inference I/O out of the rclpy timer into a latest-only worker.
5. Exercise timeouts, malformed actions, camera loss, and server loss.
6. Pass the full path using ROS fake hardware before any physical-arm run.
