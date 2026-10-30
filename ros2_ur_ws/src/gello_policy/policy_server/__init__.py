"""py3.12 ACT inference server for the UR7e banana-in-pot deploy.

This subpackage runs in the project `lr_env` (python3.12, lerobot 0.6.1 + torch),
SEPARATE from the py3.10 rclpy nodes. It exposes the trained ACT policy over a
localhost ZMQ REQ/REP link. See BUILD_SPEC.md §3/§4.

Modules:
  zmq_protocol      -- shared frame keys / command names / endpoint defaults (stdlib only)
  image_preprocess  -- JPEG -> RGB 360x640 float CHW [0,1] (training-parity resize)
  act_server        -- policy load + ZMQ REP inference loop
"""
