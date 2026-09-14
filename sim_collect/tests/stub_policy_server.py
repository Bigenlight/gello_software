"""ZMQ REP stub speaking the REAL remote-policy protocol (policy_server/zmq_protocol.py).

    RESET: [b'{"cmd":"reset"}']                   -> [b'{"ok":true, ...v2 fields}']
    ACT:   [b'{"cmd":"act","state":[7]}', jpg, jpg] -> [b'{"ok":true,"action":[7]}']

Action modes: "hold" (echo the observed q + grip 0 — the arm stays put), "constant"
(a fixed 7-vector), or a callable `(state, cam1_jpeg, cam2_jpeg, n) -> action7`.
`delay_s` sleeps before every reply (to provoke the client's timeout); `reset_extra` is
merged into the RESET reply (e.g. {"state_dim": 16} to test the EEF refusal).

In-process (tests):   with StubPolicyServer(port=0) as s: ZmqPolicy(s.endpoint)
CLI (O1 / manual):    .venv/bin/python -m sim_collect.tests.stub_policy_server --port 5599 [--delay 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import zmq

CMD_RESET, CMD_ACT = "reset", "act"


class StubPolicyServer:
    def __init__(self, port: int = 0, host: str = "127.0.0.1", action: Any = "hold", delay_s: float = 0.0,
                 reset_extra: Optional[Dict[str, Any]] = None, verbose: bool = False) -> None:
        self.host, self.port = host, int(port)
        self.action = action
        self.delay_s = float(delay_s)
        self.reset_extra = dict(reset_extra or {"policy_type": "stub", "checkpoint": "stub://none",
                                                "state_dim": 7, "action_dim": 7})
        self.verbose = verbose
        self.calls: List[Dict[str, Any]] = []
        self.n_reset = 0
        self.n_act = 0
        self._stop = threading.Event()
        self._bound = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.endpoint = ""

    # -- lifecycle ----------------------------------------------------------------
    def start(self) -> "StubPolicyServer":
        self._thread = threading.Thread(target=self._run, name="stub_policy_server", daemon=True)
        self._thread.start()
        if not self._bound.wait(5.0):
            raise RuntimeError("stub policy server did not bind")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)

    def __enter__(self) -> "StubPolicyServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- protocol -----------------------------------------------------------------
    def _action_for(self, state: Sequence[float], cam1: bytes, cam2: bytes) -> List[float]:
        if callable(self.action):
            return [float(v) for v in self.action(state, cam1, cam2, self.n_act)]
        if isinstance(self.action, str) and self.action == "hold":
            return [float(v) for v in state[:6]] + [0.0]
        return [float(v) for v in self.action]

    def handle(self, frames: List[bytes]) -> bytes:
        try:
            req = json.loads(bytes(frames[0]).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "err": f"bad request: {exc}"}).encode()
        cmd = req.get("cmd")
        if cmd == CMD_RESET:
            self.n_reset += 1
            self.calls.append({"cmd": cmd})
            return json.dumps({"ok": True, **self.reset_extra}).encode()
        if cmd == CMD_ACT:
            if len(frames) != 3:
                return json.dumps({"ok": False, "err": f"ACT needs 3 frames, got {len(frames)}"}).encode()
            state = req.get("state")
            if not isinstance(state, list) or len(state) != 7:
                return json.dumps({"ok": False, "err": "state must be 7 floats"}).encode()
            cam1, cam2 = bytes(frames[1]), bytes(frames[2])
            if not (cam1[:2] == b"\xff\xd8" and cam2[:2] == b"\xff\xd8"):
                return json.dumps({"ok": False, "err": "frames 1/2 must be JPEG"}).encode()
            self.n_act += 1
            self.calls.append({"cmd": cmd, "state": state, "cam1_bytes": len(cam1), "cam2_bytes": len(cam2)})
            return json.dumps({"ok": True, "action": self._action_for(state, cam1, cam2)}).encode()
        return json.dumps({"ok": False, "err": f"unknown cmd {cmd!r}"}).encode()

    def _run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        if self.port == 0:
            self.port = sock.bind_to_random_port(f"tcp://{self.host}")
        else:
            sock.bind(f"tcp://{self.host}:{self.port}")
        self.endpoint = f"zmq://{self.host}:{self.port}"
        self._bound.set()
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(50):
                    continue
                frames = sock.recv_multipart()
                if self.delay_s > 0:
                    time.sleep(self.delay_s)
                reply = self.handle(frames)
                if self.verbose:
                    print(f"[stub] {frames[0][:60]!r} -> {reply[:80]!r}", flush=True)
                sock.send(reply)
        finally:
            sock.close(0)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--mode", choices=["hold", "constant"], default="hold")
    ap.add_argument("--constant", type=float, nargs=7, default=None, help="7 floats for --mode constant")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep before every reply")
    ap.add_argument("--state-dim", type=int, default=7)
    ap.add_argument("--action-dim", type=int, default=7)
    args = ap.parse_args(argv)
    action: Any = "hold" if args.mode == "hold" else (args.constant or [0.0] * 7)
    srv = StubPolicyServer(args.port, args.host, action=action, delay_s=args.delay, verbose=True,
                           reset_extra={"policy_type": "stub", "checkpoint": "stub://none",
                                        "state_dim": args.state_dim, "action_dim": args.action_dim})
    srv.start()
    print(f"[stub] serving {srv.endpoint} mode={args.mode} delay={args.delay}s", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
