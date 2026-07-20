"""Minimal client-side gRPC bindings for remote_diffusion.proto.

This stays handwritten so ROS Humble's grpcio 1.30 remains supported. Regenerate
``remote_diffusion_pb2.py`` with ``scripts/generate_remote_diffusion_stubs.sh``
when the schema changes, then update these four method bindings if needed.
"""

import grpc

from . import remote_diffusion_pb2 as remote__diffusion__pb2


class RemoteDiffusionStub:
    def __init__(self, channel):
        self.Health = channel.unary_unary(
            "/gello.remote_diffusion.v1.RemoteDiffusion/Health",
            request_serializer=remote__diffusion__pb2.HealthRequest.SerializeToString,
            response_deserializer=remote__diffusion__pb2.HealthReply.FromString,
        )
        self.GetServerInfo = channel.unary_unary(
            "/gello.remote_diffusion.v1.RemoteDiffusion/GetServerInfo",
            request_serializer=remote__diffusion__pb2.ServerInfoRequest.SerializeToString,
            response_deserializer=remote__diffusion__pb2.ServerInfoReply.FromString,
        )
        self.ResetEpisode = channel.unary_unary(
            "/gello.remote_diffusion.v1.RemoteDiffusion/ResetEpisode",
            request_serializer=remote__diffusion__pb2.ResetEpisodeRequest.SerializeToString,
            response_deserializer=remote__diffusion__pb2.ResetEpisodeReply.FromString,
        )
        self.StreamActions = channel.stream_stream(
            "/gello.remote_diffusion.v1.RemoteDiffusion/StreamActions",
            request_serializer=remote__diffusion__pb2.ObservationRequest.SerializeToString,
            response_deserializer=remote__diffusion__pb2.ActionReply.FromString,
        )
