import threading
import time
from types import SimpleNamespace

import zmq

from gello_policy import obs_assembler
from gello_policy.policy_leader_node import EXECUTE, HOLD, PolicyLeaderNode


class _Logger:
    def __init__(self):
        self.info_rows = []
        self.error_rows = []

    def info(self, value):
        self.info_rows.append(str(value))

    def error(self, value):
        self.error_rows.append(str(value))


def _bare_node():
    node = object.__new__(PolicyLeaderNode)
    node._episode_finalize_lock = threading.Lock()
    node._episode_finalize_inflight = False
    node._episode_finalize_generation = 0
    node._episode_finalize_thread = None
    node._episode_finalize_timeout_s = 2.0
    node._transport = "zmq"
    node._arming_generation = 0
    node._arming_result = None
    node._grpc_worker = None
    node._auto_start_fired = False
    node._goto_active = False
    node._live_q = [0.0] * 6
    node._start_pose = [0.0] * 6
    node._last_grip_cmd = 0.25
    node._state = EXECUTE
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    return node


def test_hold_stops_immediately_and_starts_background_finalize():
    node = _bare_node()
    seen = []

    def begin():
        seen.append(node._state)
        return True

    node._begin_episode_finalize = begin
    response = SimpleNamespace(success=False, message="")
    node._srv_hold(None, response)

    assert node._state == HOLD
    assert seen == [HOLD]
    assert response.success
    assert "episode finalizing in background" in response.message


def test_start_is_refused_while_previous_episode_finalizes():
    node = _bare_node()
    node._state = HOLD
    node._episode_finalize_inflight = True
    assert node._try_start_execution() == (
        False,
        "refused: previous episode is still finalizing",
    )


def test_background_finalize_sends_reset_and_releases_gate():
    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    got = []

    def serve_once():
        request = server.recv_multipart()
        got.append(obs_assembler.parse_reply(request))
        server.send_multipart([b'{"ok":true}'])

    server_thread = threading.Thread(target=serve_once)
    server_thread.start()

    node = _bare_node()
    node._zmq_ctx = context
    node._act_host = "127.0.0.1"
    node._act_port = port
    assert node._begin_episode_finalize()
    node._episode_finalize_thread.join(timeout=3.0)
    server_thread.join(timeout=3.0)

    assert not node._episode_finalize_thread.is_alive()
    assert not node._episode_finalize_inflight
    assert got == [{"cmd": "reset"}]
    assert any("finalize complete" in row for row in node._logger.info_rows)
    assert not node._logger.error_rows

    server.close(0)
    context.term()
