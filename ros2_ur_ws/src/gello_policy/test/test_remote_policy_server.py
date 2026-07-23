import numpy as np
import sys
import types

from gello_policy import remote_diffusion_pb2 as pb

# Docker generates the same stubs inside policy_server. The source checkout keeps
# laptop stubs in gello_policy, so alias them for server-side unit tests.
sys.modules["policy_server.remote_diffusion_pb2"] = pb
sys.modules["policy_server.remote_diffusion_pb2_grpc"] = types.SimpleNamespace(
    RemoteDiffusionServicer=object,
)

from policy_server.remote_diffusion_server import (
    RemoteDiffusionService,
    RequestValidationError,
)
from policy_server.remote_lerobot_server import _sampling_contract


class FakeEngine:
    device = "cuda"
    last_act_metadata = {
        "preprocess_ms": 1.0,
        "inference_ms": 2.0,
        "total_server_ms": 3.0,
        "chunk_refill": True,
        "remaining_chunk_actions": 23,
    }

    def reset(self):
        pass

    def act(self, state, cam1, cam2):
        return np.asarray(state, dtype=np.float64)


class AbortCalled(RuntimeError):
    pass


class FakeContext:
    def abort(self, code, detail):
        raise AbortCalled(f"{code}: {detail}")


def _request(request_id=1):
    frame = pb.ImageFrame(encoding="jpeg", data=b"jpeg")
    return pb.ObservationRequest(
        protocol_version="1", client_id="client", session_id="session",
        request_id=request_id, state=[0.0] * 7, cam1=frame, cam2=frame,
    )


def _service(engine):
    service = RemoteDiffusionService(
        engine, model_id="model", checkpoint_revision="sha", scheduler="policy",
        num_inference_steps=1, n_action_steps=1, max_jpeg_bytes=1024,
    )
    service._active_client_id = "client"
    service._active_session_id = "session"
    return service


class FakeConfig:
    type = "multi_task_dit"
    objective = "flow_matching"
    num_integration_steps = 10


def test_generic_sampling_contract_uses_policy_type_and_objective():
    assert _sampling_contract(FakeConfig()) == ("multi_task_dit:flow_matching", 10)


def test_server_info_reports_generic_resize_and_sampling_contract():
    service = RemoteDiffusionService(
        FakeEngine(), model_id="cube-fm", checkpoint_revision="sha",
        scheduler="multi_task_dit:flow_matching", num_inference_steps=10,
        n_action_steps=24, max_jpeg_bytes=1024,
        resize_height=0, resize_width=0,
    )
    info = service.GetServerInfo(pb.ServerInfoRequest(), None)
    assert info.model_id == "cube-fm"
    assert info.scheduler == "multi_task_dit:flow_matching"
    assert info.num_inference_steps == 10
    assert info.n_action_steps == 24
    assert (info.resize_height, info.resize_width) == (0, 0)


def test_protocol_error_is_recoverable_without_advancing_request():
    engine = FakeEngine()
    service = _service(engine)
    replies = list(service.StreamActions(iter([_request(2), _request(1)]), FakeContext()))
    assert not replies[0].ok
    assert "request_id must be 1" in replies[0].error
    assert replies[1].ok
    assert service._last_request_id == 1
    assert service._ready


def test_engine_value_error_marks_stateful_server_restart_required():
    class BrokenEngine(FakeEngine):
        def act(self, state, cam1, cam2):
            raise ValueError("postprocessor failed after queue mutation")

    service = _service(BrokenEngine())
    try:
        list(service.StreamActions(iter([_request()]), FakeContext()))
    except AbortCalled as exc:
        assert "server restart required: ValueError" in str(exc)
    else:
        raise AssertionError("inference failure must abort the stream")
    assert not service._ready
    assert not service._active_client_id


def test_pre_policy_input_error_is_recoverable_with_same_request_id():
    class InputCheckingEngine(FakeEngine):
        calls = 0

        def act(self, state, cam1, cam2):
            self.calls += 1
            if self.calls == 1:
                raise RequestValidationError("cv2.imdecode failed")
            return super().act(state, cam1, cam2)

    service = _service(InputCheckingEngine())
    replies = list(service.StreamActions(iter([_request(1), _request(1)]), FakeContext()))
    assert not replies[0].ok
    assert "imdecode" in replies[0].error
    assert replies[1].ok
    assert service._last_request_id == 1
    assert service._ready


def test_reset_failure_clears_session_and_requires_restart():
    class ResetBrokenEngine(FakeEngine):
        def reset(self):
            raise RuntimeError("reset failed")

    service = _service(ResetBrokenEngine())
    request = pb.ResetEpisodeRequest(client_id="next", session_id="episode")
    try:
        service.ResetEpisode(request, FakeContext())
    except AbortCalled as exc:
        assert "server restart required: RuntimeError" in str(exc)
    else:
        raise AssertionError("reset failure must abort")
    assert not service._ready
    assert not service._active_client_id
    assert not service._active_session_id
    assert service._last_request_id == 0
