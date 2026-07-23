# Remote common LeRobot policy interface

The robot laptop owns cameras, ROS2, safety checks, start/hold/fault state, and
UR7e/gripper control. Kanu owns the GPU checkpoint and performs all LeRobot
preprocessing and inference. Only observations and actions cross the SSH/gRPC
boundary.

```text
Robot laptop                            Kanu GPU container
-------------                           ------------------
joint/gripper/cam1/cam2  -- gRPC/SSH -> LeRobotPolicyWrapper
policy_leader_node                      config -> policy class
safety clamps            <- action[7]-- pre -> select_action -> post
UR7e + Robotiq
```

## Components

- `policy_server/lerobot_policy_wrapper.py`: standard LeRobot checkpoint
  lifecycle using `PreTrainedConfig`, `get_policy_class`,
  `from_pretrained`, and saved pre/postprocessors.
- `policy_server/remote_lerobot_server.py`: policy-neutral Kanu gRPC server.
- `gello_policy/remote_policy_client.py`: policy-neutral local import names for
  the validated latest-observation-only worker.
- `deploy/remote_diffusion/compose.yaml`: the existing verified Diffusion
  service plus an opt-in `policy-server` profile.

The protobuf service retains its historical `RemoteDiffusion` v1 name so the
already generated Python stubs and deployed laptop remain wire-compatible. Its
session ownership, request ordering, freshness metadata, and 7-D observation /
action payload are not Diffusion-specific.

## Compatibility boundary

Policy loading is independent of ACT, Diffusion, or Flow Matching. The current
robot protocol remains intentionally strict:

```text
observation.state          shape (7,)
observation.images.cam1    visual
observation.images.cam2    visual
action                     shape (7,)
```

An incompatible checkpoint fails while the Kanu service starts. It is never
adapted by guessing joint order, camera order, or action semantics.

## Kanu configuration

On Kanu, copy `.env.example` to an untracked `.env` and set:

```dotenv
CHECKPOINT_DIR=/absolute/path/to/pretrained_model
GPU_DEVICE=0
MODEL_ID=Bigenlight/example_model
CHECKPOINT_REVISION=sha256:...
POLICY_INFERENCE_PORT=50052
POLICY_TASK=put the cube in the cup
POLICY_WARMUP_STATE=[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]
POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
EXTERNAL_IMAGE_SIZE=native
```

Examples of policy-specific config, still using the same wrapper:

```text
ACT:
  POLICY_CONFIG_OVERRIDES={"n_action_steps":30}
  EXTERNAL_IMAGE_SIZE=360x640

Diffusion:
  POLICY_CONFIG_OVERRIDES={"noise_scheduler_type":"DDIM","num_inference_steps":10,"n_action_steps":32}
  EXTERNAL_IMAGE_SIZE=360x640

MultiTaskDiT / Flow Matching:
  POLICY_CONFIG_OVERRIDES={"num_integration_steps":10,"n_action_steps":24}
  EXTERNAL_IMAGE_SIZE=native
  POLICY_TASK=<exact training task string>
```

Start only the generic service:

```bash
cd ros2_ur_ws/src/gello_policy/deploy/remote_diffusion
GPU_DEVICE=0 docker compose --env-file .env --profile generic \
  up --build policy-server
```

Before opening the gRPC port as ready, the service runs two synthetic inference
ticks at `POLICY_WARMUP_STATE` to initialize CUDA kernels and then resets all
policy/processor queues. This prevents the first operator-authorized observation
from paying one-time initialization latency.

The published endpoint remains loopback-only. The laptop should reach Kanu with
the same SSH tunnel pattern as `run_ur7e_diffusion_remote.sh`, using remote port
`50052` (or the configured `POLICY_INFERENCE_PORT`).

## Server contract mapping in protobuf v1

For compatibility, `ServerInfoReply` keeps its original field names:

- `scheduler`: `<policy.type>` or `<policy.type>:<objective>`
- `num_inference_steps`: effective sampling/integration steps, or 1
- `n_action_steps`: checkpoint action-chunk execution length
- `resize_height/width`: external resize, or `0/0` when the policy resizes internally

Configure the robot-side expected contract to these effective values before
arming. The existing local worker rejects a mismatched model, revision, sampling
contract, dimensions, device, or resize contract.

## Validation requirement

Before connecting a real robot:

1. Run the container smoke request with held state and synthetic images.
2. Replay recorded observations through both the policy-specific reference
   server and this generic server.
3. Compare actions and refill timing.
4. Measure p99 Kanu round-trip latency through the SSH tunnel.
5. Only then use the ROS start service with the existing safety stack.

Interface unit tests do not establish model parity or real-robot safety.
