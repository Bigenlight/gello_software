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


def test_conflate_subscriber_returns_newest_only():
    pub = ipc.Publisher("preview_pub")
    fresh = ipc.Subscriber("preview_pub", "state", conflate=True)
    every = ipc.Subscriber("preview_pub", "state", hwm=1000)
    time.sleep(0.3)   # slow-joiner
    for i in range(300):
        pub.send("state", {"i": i}); time.sleep(0.0005)
    time.sleep(0.2)
    assert fresh.recv(500)["i"] == 299          # conflate keeps only the newest
    assert fresh.latest() is None
    got = [every.recv(50) for _ in range(5)]
    assert [g["i"] for g in got if g] == [0, 1, 2, 3, 4]   # non-conflate keeps order
    pub.close(); fresh.close(); every.close()


def test_topic_filter_on_single_part_frames():
    pub = ipc.Publisher("state_pub")
    sub = ipc.Subscriber("state_pub", "preview")
    time.sleep(0.3)
    pub.send("state", {"x": 1}); pub.send("preview", {"x": 2}); time.sleep(0.1)
    assert sub.recv(300) == {"x": 2} and sub.recv(100) is None
    pub.close(); sub.close()
