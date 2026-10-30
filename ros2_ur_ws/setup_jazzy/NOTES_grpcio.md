# grpcio on Jazzy/noble — investigation note (gello_policy)

> Follow-up, 2026-09-16: the initial install omitted python3-grpcio/protobuf/h5py/zmq.
> `install_jazzy.sh` now includes all four. Existing installations need
> `sudo apt-get install -y python3-grpcio python3-protobuf python3-h5py python3-zmq`.
> The first combined suite stopped during collection on missing protobuf and h5py.
> Local IFQL uses ZMQ; its smoke test does not establish gRPC interoperability.
> Resolved later the same day: the user installed all four apt packages.
> Full ROS-package pytest: 928 passed, no skips. Runtime imports succeeded with
> grpc 1.51.1, protobuf 4.21.12, h5py 3.10.0 and zmq 24.0.1 on system Python 3.12.3.
> Actual remote-server gRPC interoperability remains a separate, unrun check.

## Where grpc is actually imported (scope: files owned by the Jazzy-port task
for `gello_policy/gello_policy/**`)

- `gello_policy/gello_policy/remote_diffusion_pb2_grpc.py` — `import grpc` at
  module top level, unconditional. This module is the handwritten gRPC stub
  for the `RemoteDiffusion` service (Health/GetServerInfo/Predict/...).
- `gello_policy/gello_policy/remote_diffusion_client.py` — `import grpc` is
  **lazy**, done inside `create_worker()` only (and `from . import
  remote_diffusion_pb2 as pb` is lazily imported inside `get_server_info()`
  too). The module docstring says why: "the local ZMQ deploy paths (ACT / FM
  / local Diffusion) must never load google.protobuf." So `grpc` is only
  imported when the **remote Diffusion** deploy path is actually selected
  (`inference_transport` / remote diffusion launch), not for ACT/FM/local
  Diffusion, which stay pure-ZMQ and never touch grpc or protobuf.
- Nothing else under `gello_policy/gello_policy/` or `gello_recorder/` imports
  `grpc`.

## The known Jammy/Humble problem (for context, not this task's fix)

`ros2_ur_ws/requirements-remote-client.lock` (top-level, not owned by this
port) pins `grpcio==1.74.0` for the robot laptop because Ubuntu 22.04
(Jammy)'s `python3-grpcio` apt package is 1.30.2, and that version cannot
complete the HTTP/2 handshake to the remote policy server (it opens the
SSH-forwarded TCP socket fine, then hangs/fails at the protocol level). That
lock file's own comment documents this.

## Jazzy/noble situation

Ubuntu 24.04 (Noble)'s apt `python3-grpcio` is **1.51.1** — twenty-one minor
versions newer than the broken 1.30.2 on Jammy. This has NOT been tested
against the actual remote Diffusion server from this worktree (no live
server reachable, no ROS environment installed yet at investigation time), so
the following is a documented risk assessment, not a verified result:

- Version numbers alone do not establish HTTP/2 interoperability. Test the
  actual remote server with the selected ROS client interpreter.
- If a pinned alternative is needed, use a separate system-Python 3.12
  environment with ROS packages available, following the existing remote
  client setup and `requirements-remote-client.lock`. Do not install into
  shared conda `il`, or assume `pip install --user` works on Noble's
  externally managed system Python.

## What this task did NOT do

- Did not install grpcio (apt or pip) on this machine — ROS/apt changes are
  out of scope for this worktree per the task's constraints.
- Did not modify `requirements-remote-client.lock` — that file lives at
  `ros2_ur_ws/` top level and is not part of `gello_recorder/` or
  `gello_policy/gello_policy/**`.
- Did not change `remote_diffusion_pb2_grpc.py`'s `import grpc` to be lazy —
  it is generated/handwritten to mirror a real `grpc.Stub` class shape and is
  only ever imported (module-level `import grpc` and all) from inside
  `create_worker()` in `remote_diffusion_client.py`, i.e. it is already behind
  the same lazy gate at the point it's actually loaded. Its own top-level
  `import grpc` only matters once something imports
  `remote_diffusion_pb2_grpc` directly — which no code in this repo's ACT/FM/
  local-Diffusion paths does.

## Recommendation for whoever does the live Jazzy verification

1. Bring up the remote Diffusion server and a Jazzy robot-laptop client with
   stock apt `python3-grpcio` 1.51.1 first (no pip override).
2. If `create_worker()` → `stub.GetServerInfo(...)` succeeds, apt's grpcio is
   sufficient on noble and `requirements-remote-client.lock`'s pin can stay
   Jammy-specific (or become distro-conditional).
3. If the remote handshake fails, capture the error and test the separately
   pinned client environment before changing dependencies. Record the actual
   interpreter and versions alongside the result.
