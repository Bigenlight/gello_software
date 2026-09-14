import atexit
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    use_save_interface: bool = False
    """Enable saving data with keyboard interface."""


def initialize_sim_from_agent(robot, env, agent, timeout_s: float = 2.0) -> None:
    """Teleport the simulated robot to the agent's (e.g. GELLO's) current pose.

    Reads the leader once, hands the pose to the sim's physics thread via
    ``reset_joint_state`` and waits until the observed joints match. Prints the
    pose in a form that can be pasted into ``agent.start_joints`` if the user
    wants to pin it as the episode reset pose later.
    """
    import numpy as np

    pose = np.asarray(agent.act(env.get_obs()), dtype=float)
    if pose.shape[0] != env.get_obs()["joint_positions"].shape[0]:
        print(
            f"Warning: agent dim {pose.shape[0]} != robot dim "
            f"{env.get_obs()['joint_positions'].shape[0]}; skipping sim init."
        )
        return
    print("Initializing sim robot at the leader's current pose (teleport, no motion):")
    print("  start_joints: [" + ", ".join(f"{v:.4f}" for v in pose) + "]")
    robot.reset_joint_state(pose)

    # The teleport itself is exact; what remains is gravity sag of the position
    # actuators (measured 0.011 rad at the UR calibration pose, wrist kp=500),
    # so accept anything under 0.05 rad. A residual far above that means the
    # pose collides with the scene (table / cube) or the arm itself.
    n_arm = pose.shape[0] - 1  # gripper settles physically; only check the arm
    tol = 0.05
    deadline = time.time() + timeout_s
    err = float("inf")
    while time.time() < deadline:
        obs = env.get_obs()["joint_positions"]
        err = float(np.abs(obs[:n_arm] - pose[:n_arm]).max())
        if err < tol:
            print(f"  sim robot placed (residual {err:.4f} rad).")
            return
        time.sleep(0.02)
    print(
        f"Warning: sim robot is {err:.3f} rad from the leader pose after {timeout_s:.0f}s "
        "(collision with the scene?); continuing."
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Initial pose. Two paths:
    #  - Simulation (robot has reset_joint_state): read the agent ONCE and
    #    teleport the sim arm to exactly that pose, so the loop starts with zero
    #    leader/follower error. No motion at all. Set `init_from_agent: false`
    #    in the yaml to get the legacy start_joints move instead.
    #  - Hardware (no reset_joint_state): legacy gradual move to start_joints.
    #    Hardware is never teleported.
    from gello.utils.launch_utils import move_to_start_position

    init_from_agent = bool(cfg.get("init_from_agent", True))
    if not bimanual and init_from_agent and hasattr(robot, "reset_joint_state"):
        initialize_sim_from_agent(robot, env, agent)
    elif bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=Path(args.left_config_path).parents[1] / "data",
            agent_name=agent.__class__.__name__,
            expand_user=True,
        )

    # Run main control loop
    run_control_loop(env, agent, save_interface)


if __name__ == "__main__":
    main()
