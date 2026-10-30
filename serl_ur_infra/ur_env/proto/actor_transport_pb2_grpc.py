"""grpcio bindings for actor_transport.proto.

Kept handwritten so the committed protobuf module remains usable with the
Ubuntu/ROS protobuf toolchain as well as the isolated grpcio 1.74 runtime.
"""

import grpc

from . import actor_transport_pb2 as actor__transport__pb2


class ActorTransportStub:
    def __init__(self, channel):
        self.Health = channel.unary_unary(
            "/gello.hil_serl.v1.ActorTransport/Health",
            request_serializer=actor__transport__pb2.HealthRequest.SerializeToString,
            response_deserializer=actor__transport__pb2.HealthReply.FromString,
        )
        self.GetServerInfo = channel.unary_unary(
            "/gello.hil_serl.v1.ActorTransport/GetServerInfo",
            request_serializer=actor__transport__pb2.ServerInfoRequest.SerializeToString,
            response_deserializer=actor__transport__pb2.ServerInfoReply.FromString,
        )
        self.GetBufferStatus = channel.unary_unary(
            "/gello.hil_serl.v1.ActorTransport/GetBufferStatus",
            request_serializer=actor__transport__pb2.BufferStatusRequest.SerializeToString,
            response_deserializer=actor__transport__pb2.BufferStatusReply.FromString,
        )
        self.BeginEpisode = channel.unary_unary(
            "/gello.hil_serl.v1.ActorTransport/BeginEpisode",
            request_serializer=actor__transport__pb2.BeginEpisodeRequest.SerializeToString,
            response_deserializer=actor__transport__pb2.ActionReply.FromString,
        )
        self.Step = channel.unary_unary(
            "/gello.hil_serl.v1.ActorTransport/Step",
            request_serializer=actor__transport__pb2.StepRequest.SerializeToString,
            response_deserializer=actor__transport__pb2.StepReply.FromString,
        )


class ActorTransportServicer:
    def Health(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Health is not implemented")
        raise NotImplementedError("Health is not implemented")

    def GetServerInfo(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GetServerInfo is not implemented")
        raise NotImplementedError("GetServerInfo is not implemented")

    def GetBufferStatus(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GetBufferStatus is not implemented")
        raise NotImplementedError("GetBufferStatus is not implemented")

    def BeginEpisode(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("BeginEpisode is not implemented")
        raise NotImplementedError("BeginEpisode is not implemented")

    def Step(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Step is not implemented")
        raise NotImplementedError("Step is not implemented")


def add_ActorTransportServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=actor__transport__pb2.HealthRequest.FromString,
            response_serializer=actor__transport__pb2.HealthReply.SerializeToString,
        ),
        "GetServerInfo": grpc.unary_unary_rpc_method_handler(
            servicer.GetServerInfo,
            request_deserializer=actor__transport__pb2.ServerInfoRequest.FromString,
            response_serializer=actor__transport__pb2.ServerInfoReply.SerializeToString,
        ),
        "GetBufferStatus": grpc.unary_unary_rpc_method_handler(
            servicer.GetBufferStatus,
            request_deserializer=actor__transport__pb2.BufferStatusRequest.FromString,
            response_serializer=actor__transport__pb2.BufferStatusReply.SerializeToString,
        ),
        "BeginEpisode": grpc.unary_unary_rpc_method_handler(
            servicer.BeginEpisode,
            request_deserializer=actor__transport__pb2.BeginEpisodeRequest.FromString,
            response_serializer=actor__transport__pb2.ActionReply.SerializeToString,
        ),
        "Step": grpc.unary_unary_rpc_method_handler(
            servicer.Step,
            request_deserializer=actor__transport__pb2.StepRequest.FromString,
            response_serializer=actor__transport__pb2.StepReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "gello.hil_serl.v1.ActorTransport", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
