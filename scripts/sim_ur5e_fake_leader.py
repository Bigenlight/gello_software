#!/usr/bin/env python3
"""No-GELLO smoke test for the upstream MuJoCo UR5e sim.

Builds the same MujocoRobotServer as ``experiments/launch_yaml.py`` (robot
section of a config), but replaces the physical GELLO with a sine-wave fake
leader that talks to the server through the real ZMQ client. No serial port is
opened, so nothing can touch GELLO torque.

  conda run -n gello-sim python scripts/sim_ur5e_fake_leader.py            # viewer window
  MUJOCO_GL=egl conda run -n gello-sim python scripts/sim_ur5e_fake_leader.py --headless
"""
import argparse
import threading
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf

from gello.robots.sim_robot import MujocoRobotServer
from gello.zmq_core.robot_node import ZMQClientRobot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ur7e_gello_sim.yaml")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--amp", type=float, default=0.3, help="sine amplitude [rad]")
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    rcfg = {k: v for k, v in cfg["robot"].items() if k != "_target_"}
    start = np.array(cfg["agent"]["start_joints"], dtype=float)
    server = MujocoRobotServer(**rcfg)
    server.reset_joint_state(start)
    stop = threading.Event()

    def fake_leader() -> None:
        client = ZMQClientRobot(port=rcfg["port"], host=rcfg["host"])
        t0, first, last = time.time(), None, None
        while not stop.is_set() and time.time() - t0 < args.seconds:
            cmd = start.copy()
            cmd[1] += args.amp * np.sin(2 * np.pi * 0.5 * (time.time() - t0))
            cmd[2] += args.amp * np.sin(2 * np.pi * 0.5 * (time.time() - t0) + 1.0)
            client.command_joint_state(cmd)
            last = client.get_joint_state()
            first = last if first is None else first
            time.sleep(1 / 30)
        print(f"joint[1] moved {first[1]:+.3f} -> {last[1]:+.3f} rad (commanded {start[1]:+.3f}+-{args.amp})")
        print("PASS: arm moved" if abs(last[1] - first[1]) > 1e-2 else "FAIL: arm did not move")
        stop.set()

    if args.headless:
        server._zmq_server_thread.start()  # serve() does this itself in viewer mode
        threading.Thread(target=fake_leader, daemon=True).start()
        m, d = server._model, server._data
        while not stop.is_set():
            server._apply_pending_reset()
            d.ctrl[:] = server._joint_cmd
            mujoco.mj_step(m, d)
            server._joint_state = d.qpos.copy()[: server._num_joints]
            time.sleep(m.opt.timestep)
    else:
        threading.Timer(1.5, lambda: threading.Thread(target=fake_leader, daemon=True).start()).start()
        server.serve()  # runs until you close the window (Ctrl-C also works)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
