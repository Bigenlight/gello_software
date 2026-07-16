import threading
import time

from gello_policy import remote_diffusion_pb2 as pb
from gello_policy.remote_diffusion_client import (
    ImageSnapshot,
    ObservationSnapshot,
    RemoteDiffusionWorker,
    ServerContract,
)


def _observation(marker, created_ns=None):
    image = ImageSnapshot(ros_stamp_ns=marker, width=640, height=360, jpeg=b"jpeg")
    return ObservationSnapshot(
        created_monotonic_ns=created_ns or time.monotonic_ns(),
        state=(float(marker), 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        cam1=image,
        cam2=image,
    )


def _reply(request, **overrides):
    values = dict(
        protocol_version="1",
        session_id=request.session_id,
        request_id=request.request_id,
        action=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
        ok=True,
    )
    values.update(overrides)
    return pb.ActionReply(**values)


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


class FakeStub:
    def __init__(self):
        self.requests = []
        self.timeouts = []
        self.block_first = None
        self.reply_transform = lambda request: _reply(request)

    def GetServerInfo(self, request, timeout):
        return pb.ServerInfoReply(
            ready=True, protocol_version="1", state_dim=7, action_dim=7,
            device="cuda", model_id="model", checkpoint_revision="sha",
            scheduler="DDIM", num_inference_steps=10, n_action_steps=32,
            resize_height=360, resize_width=640,
        )

    def ResetEpisode(self, request, timeout):
        return pb.ResetEpisodeReply(ok=True)

    def StreamActions(self, requests, timeout):
        request = next(requests)
        self.requests.append(request)
        self.timeouts.append(timeout)
        if len(self.requests) == 1 and self.block_first is not None:
            self.block_first.wait(1.0)
        return iter((self.reply_transform(request),))


def test_server_info_reset_and_action_round_trip():
    stub = FakeStub()
    worker = RemoteDiffusionWorker(stub, client_id="laptop", rpc_deadline_s=0.25)
    assert worker.get_server_info().ready
    assert worker.reset_episode("episode-1") == "episode-1"
    worker.start()
    worker.submit(_observation(1))
    _wait_until(worker.has_result)
    result = worker.take_result(0.1)
    worker.close()

    assert result.request_id == 1
    assert result.action[-1] == 0.5
    assert stub.requests[0].client_id == "laptop"
    assert stub.requests[0].session_id == "episode-1"
    assert stub.timeouts == [0.25]


def test_server_contract_is_strictly_checked():
    worker = RemoteDiffusionWorker(FakeStub(), client_id="laptop")
    expected = ServerContract("model", "sha", "DDIM", 10, 32, 360, 640)
    assert worker.get_server_info(expected).model_id == "model"
    try:
        worker.get_server_info(ServerContract("wrong", "sha", "DDIM", 10, 32, 360, 640))
        raise AssertionError("contract mismatch should fail")
    except RuntimeError as exc:
        assert "contract mismatch" in str(exc)


def test_latest_only_and_one_in_flight():
    stub = FakeStub()
    stub.block_first = threading.Event()
    worker = RemoteDiffusionWorker(stub, client_id="laptop")
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1))
    _wait_until(lambda: len(stub.requests) == 1)
    worker.submit(_observation(2))
    worker.submit(_observation(3))
    assert len(stub.requests) == 1
    stub.block_first.set()
    _wait_until(lambda: len(stub.requests) == 2)
    worker.close()

    assert [request.state[0] for request in stub.requests] == [1.0, 3.0]


def test_mismatched_request_id_is_rejected():
    stub = FakeStub()
    stub.reply_transform = lambda request: _reply(request, request_id=999)
    worker = RemoteDiffusionWorker(stub, client_id="laptop")
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1))
    _wait_until(lambda: worker.error() is not None)
    worker.close()

    assert "session/request ID mismatch" in worker.error()
    assert not worker.has_result()


def test_stale_response_is_rejected():
    stub = FakeStub()
    now = time.monotonic_ns()
    worker = RemoteDiffusionWorker(
        stub, client_id="laptop", max_response_age_s=0.1, clock_ns=lambda: now
    )
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1, created_ns=now - 200_000_000))
    _wait_until(lambda: worker.error() is not None)
    worker.close()

    assert "stale response" in worker.error()
    assert not worker.has_result()


def test_transport_deadline_failure_does_not_publish_action():
    stub = FakeStub()

    def timeout(_request):
        raise TimeoutError("deadline exceeded")

    stub.reply_transform = timeout
    worker = RemoteDiffusionWorker(stub, client_id="laptop", rpc_deadline_s=0.2)
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1))
    _wait_until(lambda: worker.error() is not None)
    worker.close()

    assert "deadline exceeded" in worker.error()
    assert not worker.has_result()
    assert stub.timeouts == [0.2]
    try:
        worker.submit(_observation(2))
        raise AssertionError("a failed RPC must require a new ResetEpisode")
    except RuntimeError as exc:
        assert "ResetEpisode" in str(exc)


def test_failure_clears_a_prior_success_result():
    stub = FakeStub()
    worker = RemoteDiffusionWorker(stub, client_id="laptop")
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1))
    _wait_until(worker.has_result)
    stub.reply_transform = lambda request: (_ for _ in ()).throw(TimeoutError("down"))
    worker.submit(_observation(2))
    _wait_until(lambda: worker.error() is not None)
    worker.close()
    assert not worker.has_result()


def test_reset_is_rejected_while_in_flight():
    stub = FakeStub()
    stub.block_first = threading.Event()
    worker = RemoteDiffusionWorker(stub, client_id="laptop")
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1))
    _wait_until(lambda: len(stub.requests) == 1)
    try:
        worker.reset_episode("episode-2")
        raise AssertionError("reset must be rejected while inference is in flight")
    except RuntimeError as exc:
        assert "in flight" in str(exc)
    stub.block_first.set()
    _wait_until(worker.has_result)
    worker.close()


def test_take_result_is_one_shot_and_checks_poll_age():
    now = time.monotonic_ns()
    clock = [now]
    stub = FakeStub()
    worker = RemoteDiffusionWorker(stub, client_id="laptop", clock_ns=lambda: clock[0])
    worker.reset_episode("episode-1")
    worker.start()
    worker.submit(_observation(1, created_ns=now))
    _wait_until(worker.has_result)
    assert worker.take_result(0.1) is not None
    assert worker.take_result(0.1) is None
    worker.submit(_observation(2, created_ns=now))
    _wait_until(worker.has_result)
    clock[0] += 200_000_000
    try:
        worker.take_result(0.1)
        raise AssertionError("stale polled action should fail")
    except RuntimeError as exc:
        assert "polled action is stale" in str(exc)
    worker.close()


def test_camera_metadata_size_and_skew_are_validated():
    worker = RemoteDiffusionWorker(FakeStub(), client_id="laptop", max_jpeg_bytes=3)
    worker.reset_episode("episode-1")
    obs = _observation(1)
    try:
        worker.submit(obs)
        raise AssertionError("oversize JPEG should fail")
    except ValueError as exc:
        assert "size limit" in str(exc)
    worker = RemoteDiffusionWorker(FakeStub(), client_id="laptop", max_camera_skew_s=0.0)
    worker.reset_episode("episode-1")
    cam2 = ImageSnapshot(2, 640, 360, b"jpg")
    skewed = ObservationSnapshot(obs.created_monotonic_ns, obs.state, obs.cam1, cam2)
    try:
        worker.submit(skewed)
        raise AssertionError("camera skew should fail")
    except ValueError as exc:
        assert "skew" in str(exc)


def test_submit_validates_input_and_requires_reset():
    worker = RemoteDiffusionWorker(FakeStub(), client_id="laptop")
    try:
        worker.submit(_observation(1))
        raise AssertionError("submit should require ResetEpisode")
    except RuntimeError as exc:
        assert "ResetEpisode" in str(exc)

    bad = _observation(1)
    bad = ObservationSnapshot(bad.created_monotonic_ns, (1.0,), bad.cam1, bad.cam2)
    try:
        worker.submit(bad)
        raise AssertionError("invalid state dimension should fail")
    except ValueError as exc:
        assert "7 values" in str(exc)
