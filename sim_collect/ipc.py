"""ZMQ helpers shared by sim_main (A), capture (B) and gui (C).

Messages are pickled dicts (same convention as gello/zmq_core). Endpoints default to
ipc:// sockets under /tmp/sim_collect_<user>/ so several users on one machine do not
collide; set SIM_COLLECT_IPC=tcp to fall back to 127.0.0.1 ports (6701 sim REP,
6702 capture REP, 6711 state PUB, 6712 preview PUB).
"""
from __future__ import annotations

import getpass
import os
import pickle
import time
from typing import Any, Dict, Optional

import zmq

_TCP_PORTS = {"sim_rep": 6701, "capture_rep": 6702, "state_pub": 6711, "preview_pub": 6712}


def endpoint(name: str) -> str:
    """Return the ZMQ endpoint for one of: sim_rep, capture_rep, state_pub, preview_pub."""
    if name not in _TCP_PORTS:
        raise KeyError(f"unknown endpoint {name!r}; expected one of {sorted(_TCP_PORTS)}")
    if os.environ.get("SIM_COLLECT_IPC", "ipc").lower() == "tcp":
        return f"tcp://127.0.0.1:{_TCP_PORTS[name]}"
    d = f"/tmp/sim_collect_{getpass.getuser()}"
    os.makedirs(d, exist_ok=True)
    return f"ipc://{d}/{name}.sock"


_ctx: Optional[zmq.Context] = None


def context() -> zmq.Context:
    global _ctx
    if _ctx is None:
        _ctx = zmq.Context.instance()
    return _ctx


class Publisher:
    """PUB socket. `send(topic, payload_dict)`; payload is pickled."""

    def __init__(self, name: str, hwm: int = 4):
        self.sock = context().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.bind(endpoint(name))

    def send(self, topic: str, payload: Dict[str, Any]) -> None:
        self.sock.send_multipart([topic.encode(), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)], zmq.NOBLOCK)

    def close(self) -> None:
        self.sock.close(linger=0)


class Subscriber:
    """SUB socket with CONFLATE-like behaviour done in software: `latest()` drains the
    queue and returns the newest message (or None). `recv(timeout_ms)` blocks for one."""

    def __init__(self, name: str, topic: str = "", hwm: int = 4):
        self.sock = context().socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, hwm)
        self.sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.sock.connect(endpoint(name))

    def recv(self, timeout_ms: int = 1000) -> Optional[Dict[str, Any]]:
        if self.sock.poll(timeout_ms) == 0:
            return None
        _topic, raw = self.sock.recv_multipart()
        return pickle.loads(raw)

    def latest(self) -> Optional[Dict[str, Any]]:
        msg = None
        while self.sock.poll(0):
            _topic, raw = self.sock.recv_multipart()
            msg = pickle.loads(raw)
        return msg

    def close(self) -> None:
        self.sock.close(linger=0)


class Server:
    """REP socket. Call `poll(handler, timeout_ms)` from the owner's loop; handler gets the
    request dict and returns a reply dict (must contain "ok": bool)."""

    def __init__(self, name: str):
        self.sock = context().socket(zmq.REP)
        self.sock.bind(endpoint(name))

    def poll(self, handler, timeout_ms: int = 0) -> bool:
        if self.sock.poll(timeout_ms) == 0:
            return False
        req = pickle.loads(self.sock.recv())
        try:
            rep = handler(req)
            if not isinstance(rep, dict) or "ok" not in rep:
                rep = {"ok": False, "msg": f"handler returned {type(rep).__name__} without 'ok'"}
        except Exception as e:  # never let a bad request kill the loop
            rep = {"ok": False, "msg": f"{type(e).__name__}: {e}"}
        self.sock.send(pickle.dumps(rep, protocol=pickle.HIGHEST_PROTOCOL))
        return True

    def close(self) -> None:
        self.sock.close(linger=0)


class Client:
    """REQ client with a timeout. A timed-out socket is recreated (REQ state machine)."""

    def __init__(self, name: str, timeout_ms: int = 2000):
        self.name = name
        self.timeout_ms = timeout_ms
        self.sock = None
        self._connect()

    def _connect(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = context().socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint(self.name))

    def call(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        req = {"cmd": cmd, **kwargs}
        try:
            self.sock.send(pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL))
            if self.sock.poll(self.timeout_ms) == 0:
                self._connect()
                return {"ok": False, "msg": f"{self.name}: timeout after {self.timeout_ms} ms", "timeout": True}
            return pickle.loads(self.sock.recv())
        except zmq.ZMQError as e:
            self._connect()
            return {"ok": False, "msg": f"{self.name}: {e}"}

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)


def wait_for(client: Client, timeout_s: float = 10.0, cmd: str = "get_status") -> bool:
    """Block until the server behind `client` answers `cmd` or timeout_s elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client.call(cmd).get("ok"):
            return True
        time.sleep(0.2)
    return False
