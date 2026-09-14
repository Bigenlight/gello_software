import threading, time
from sim_collect import ipc


def test_req_rep_and_pub_sub_roundtrip():
    srv = ipc.Server("sim_rep"); pub = ipc.Publisher("state_pub")
    stop = False
    def loop():
        while not stop:
            srv.poll(lambda req: {"ok": True, "echo": req}, timeout_ms=50)
            pub.send("state", {"t": time.time()})
    th = threading.Thread(target=loop, daemon=True); th.start()
    cli = ipc.Client("sim_rep", timeout_ms=2000); sub = ipc.Subscriber("state_pub", "state")
    assert ipc.wait_for(cli, 5.0)
    rep = cli.call("engage", value=3)
    assert rep["ok"] and rep["echo"] == {"cmd": "engage", "value": 3}
    assert sub.recv(2000) is not None
    time.sleep(0.05); assert sub.latest() is not None
    stop = True; th.join(2); cli.close(); sub.close(); srv.close(); pub.close()


def test_client_timeout_recovers():
    cli = ipc.Client("capture_rep", timeout_ms=100)
    rep = cli.call("get_status")
    assert rep["ok"] is False and rep.get("timeout")
    cli.close()
