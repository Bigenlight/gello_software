"""Policy-neutral names for the validated remote gRPC inference client.

The underlying protobuf service keeps its v1 ``RemoteDiffusion`` name for wire
compatibility, but its observation/session/action transport is policy agnostic.
"""

from .remote_diffusion_client import (  # noqa: F401
    ACTION_DIM,
    PROTOCOL_VERSION,
    STATE_DIM,
    ActionResult,
    ImageSnapshot,
    ObservationSnapshot,
    RemoteDiffusionWorker as RemotePolicyWorker,
    ServerContract,
    create_worker,
)

__all__ = [
    "ACTION_DIM", "PROTOCOL_VERSION", "STATE_DIM", "ActionResult",
    "ImageSnapshot", "ObservationSnapshot", "RemotePolicyWorker",
    "ServerContract", "create_worker",
]
