"""RViz fake-hardware RL loop test — run on the ROBOT LAPTOP (22.04/Humble).

Drives the real UR7eEnv (real URRosBackend, real controller, real upsampler)
against MOCK ros2_control hardware, so you can watch scripted "RL" actions move
the robot in RViz with zero physical risk. This validates, on the real ROS
graph: QoS matching (VERIFY(hw) notes), topic wiring, /joint_states feedback,
the 250 Hz upsampler stream, go_to_reset arrival, and camera decode if the
fake camera node is up.

Terminals (see serl_ur_infra/README.md "RViz fake RL 테스트"):

  T1  mock UR + RViz:
      ros2 launch gello_policy ur_control_fake_safe.launch.py \
          ur_type:=ur7e robot_ip:=0.0.0.0 use_mock_hardware:=true \
          initial_joint_controller:=forward_position_controller launch_rviz:=true
  T2  fake cameras (env defaults subscribe to these topics):
      ros2 run gello_policy fake_diffusion_observation_node
  T3  this script:
      python3 tests/run_rviz_fake_rl.py            # scripted sine actions
      python3 tests/run_rviz_fake_rl.py --random   # random policy instead

DRY_RUN is False here ON PURPOSE — commands must actually flow so RViz moves.
That is safe against mock hardware only. Never point this at a real robot.
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402


class RvizFakeConfig(DefaultUR7eEnvConfig):
    DRY_RUN = False              # mock hardware — publishing is the point
    TCP_POSE_SOURCE = "fk"       # avoid depending on tcp_pose_broadcaster here
    DISPLAY_IMAGE = False        # RViz is the display; flip on to see fake cams
    IMAGE_STALE_S = 2.0          # fake camera node may publish slowly
    RESET_JOINTS = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    # Mock hardware boots at all-zeros, farther than the real-robot guard
    # allows. Mock-only relaxation — do NOT copy into a real-robot config.
    RESET_MAX_DIST_RAD = 7.0
    RESET_TIMEOUT_S = 30.0       # zeros -> home at the 0.5 rad/s slew takes ~20 s
    MAX_EPISODE_LENGTH = 100


def scripted_action(t: float) -> np.ndarray:
    """Slow base-frame circle in x/y + gentle z — easy to eyeball in RViz."""
    a = np.zeros(7, dtype=np.float32)
    a[0] = 0.6 * np.sin(0.4 * t)
    a[1] = 0.6 * np.cos(0.4 * t)
    a[2] = 0.3 * np.sin(0.15 * t)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true", help="random actions")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = RvizFakeConfig()
    env = UR7eEnv(fake_env=False, config=cfg)
    print("waiting for /joint_states from the mock stack...")
    for _ in range(50):
        q, _, age = env.backend.get_joint_state()
        if q is not None:
            break
        time.sleep(0.2)
    else:
        sys.exit("no /joint_states — is the T1 mock launch running?")
    print("joint_states ok:", np.round(q, 2))

    for ep in range(args.episodes):
        print(f"\n=== episode {ep}: go_to_reset (watch RViz) ===")
        obs, _ = env.reset()
        t0 = time.time()
        holds = 0
        while True:
            t = time.time() - t0
            a = (
                env.action_space.sample() * 0.5
                if args.random
                else scripted_action(t)
            )
            obs, rew, done, trunc, info = env.step(a)
            if env.curr_path_length % 20 == 0:
                tcp = obs["state"]["tcp_pose"][:3]
                print(
                    f"  step {env.curr_path_length:3d}  "
                    f"tcp=[{tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}]"
                )
            if done:
                break
        print(f"episode {ep} done ({env.curr_path_length} steps)")

    env.close()
    print("\nPASS — mock RL loop ran; check RViz motion was smooth (no jumps)")


if __name__ == "__main__":
    main()
