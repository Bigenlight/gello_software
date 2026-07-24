"""Offline end-to-end test of UR7eEnv with a ROS-free fake backend.

Runs the REAL control path — _action_to_xi -> PolicyDeltaController (governor,
ik_numeric, gates) -> backend commands -> fk-based observations — with only the
ROS I/O layer replaced. ur_kin is pure numpy, so this works on a machine with
no ROS installed (the dev box; the robot laptop is the 22.04/Humble one).

The fake robot is ideal: it teleports to each commanded q instantly. That
isolates OUR math from robot dynamics — tcp_pose in obs must track the
commanded deltas almost exactly.

Run:
    conda run -n lerobot python tests/test_env_fake_backend.py
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
# ur_gello_bringup (ur_kin/eef_delta are numpy-only — importable without ROS)
sys.path.insert(
    0, os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
)

import cv2  # noqa: E402

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402


class FakeBackend:
    """URRosBackend-compatible fake: an ideal robot + synthetic cameras.

    send_joint_command teleports the robot (no upsampler — that thread is
    ROS-side; its slew math has its own convergence check below).
    """

    dry_run = False

    def __init__(self, q0: np.ndarray, cam_names):
        self.q = np.asarray(q0, dtype=float).copy()
        self.gripper = 0.0  # 0 open .. 1 closed
        self.cam_names = list(cam_names)
        self.sent_gripper = []

    # ---- state getters (data, age) ---- #
    def get_joint_state(self):
        return self.q.copy(), np.zeros(6), 0.0

    def get_gripper_percent(self):
        return self.gripper, 0.0

    def get_gello_state(self):
        return None, float("inf")

    def get_wrench(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")  # test uses TCP_POSE_SOURCE="fk"

    def get_image(self, cam_name):
        img = np.full((480, 640, 3), 90, np.uint8)
        cv2.putText(img, f"{cam_name} {time.time():.1f}", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        ok, jpeg = cv2.imencode(".jpg", img)
        assert ok
        return jpeg.tobytes(), 0.0

    # ---- commands ---- #
    def send_joint_command(self, q_cmd):
        self.q = np.asarray(q_cmd, dtype=float).reshape(6).copy()

    def send_gripper_percent(self, fraction):
        self.gripper = float(np.clip(fraction, 0.0, 1.0))
        self.sent_gripper.append(self.gripper)

    def reset_command_stream(self):
        pass

    def close(self):
        pass


class TestConfig(DefaultUR7eEnvConfig):
    DISPLAY_IMAGE = False        # headless
    TCP_POSE_SOURCE = "fk"       # no driver topic in the fake
    RESET_JOINTS = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    MAX_EPISODE_LENGTH = 20
    GRIPPER_SLEEP = 0.0          # no debounce wait in tests


def main():
    cfg = TestConfig()
    backend = FakeBackend(q0=cfg.RESET_JOINTS + 0.05, cam_names=cfg.CAMERAS)
    env = UR7eEnv(fake_env=False, config=cfg, backend=backend)

    # ---- reset: fake robot teleports, arrival check must pass ---- #
    obs, info = env.reset()
    assert np.allclose(backend.q, cfg.RESET_JOINTS, atol=cfg.RESET_TOLERANCE_RAD)
    print("reset ok  q ->", np.round(backend.q, 3))

    # ---- obs sanity ---- #
    assert set(obs["state"].keys()) == {
        "tcp_pose", "tcp_vel", "gripper_pose", "tcp_force", "tcp_torque"
    }
    for k in cfg.CAMERAS:
        img = obs["images"][k]
        assert img.shape == (*cfg.IMAGE_OBS_SIZE, 3) and img.dtype == np.uint8
        assert img.std() > 0, "camera frame decoded to a constant image?"
    print("obs ok    tcp:", np.round(obs["state"]["tcp_pose"][:3], 4))

    # ---- +x delta actions must move tcp_pose in +x ---- #
    x0 = obs["state"]["tcp_pose"][0]
    holds = 0
    t0 = time.time()
    for _ in range(10):
        a = np.zeros(7, dtype=np.float32)
        a[0] = 1.0  # full-scale +x
        obs, r, done, trunc, info = env.step(a)
    elapsed = time.time() - t0
    dx = obs["state"]["tcp_pose"][0] - x0
    # governor caps at v_max (0.1 m/s) * 1 s of stepping
    expected = min(cfg.ACTION_SCALE[0], cfg.GOVERNOR["v_max"] / cfg.HZ) * 10
    print(f"moved dx={dx:+.4f} m over 10 steps (expected ~{expected:.4f}), "
          f"rate {10 / elapsed:.1f} Hz")
    assert abs(dx - expected) < 5e-3, "tcp did not track commanded +x deltas"
    assert 8.0 < 10 / elapsed < 12.0, "step pacing far from 10 Hz"

    # ---- gripper: -1 closes, +1 opens (0..1 scale, 1=closed) ---- #
    a = np.zeros(7, dtype=np.float32); a[6] = -1.0
    env.step(a)
    assert backend.sent_gripper[-1] == 1.0, "close must send 1.0 (closed)"
    env._update_currpos()
    a[6] = +1.0
    env.step(a)
    assert backend.sent_gripper[-1] == 0.0, "open must send 0.0 (open)"
    print("gripper ok (close->1.0, open->0.0)")

    # ---- upsampler slew math (unit-level; thread itself is ROS-side) ---- #
    step = cfg.UPSAMPLER["max_step_rad"]
    target = np.array([0.05, -0.03, 0.0, 0.01, 0.0, -0.002])
    stream = np.zeros(6)
    for i in range(1000):
        stream = stream + np.clip(target - stream, -step, step)
        if np.allclose(stream, target):
            break
    per_env_step = cfg.UPSAMPLER["hz"] / cfg.HZ
    print(f"upsampler converges in {i + 1} ticks "
          f"(budget {per_env_step:.0f} ticks per env step)")
    assert i + 1 <= per_env_step, "slew too slow to keep up with 10 Hz targets"

    print("\nALL PASS — env loop runs end-to-end on the fake backend")


if __name__ == "__main__":
    main()
