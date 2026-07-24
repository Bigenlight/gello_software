"""RViz human-in-the-loop (HIL) teleop-intervention test — ROBOT LAPTOP (22.04/Humble).

Human-in-the-loop sibling of tests/run_rviz_fake_rl.py. Same idea — drive the
REAL UR7eEnv (real URRosBackend, real PolicyDeltaController, real 250 Hz
upsampler) against MOCK ros2_control so motion shows in RViz with zero physical
risk — but here the env is wrapped in GelloIntervention, so a human holding the
deadman (SPACEBAR) and moving the PHYSICAL GELLO leader arm injects expert
interventions that the mock robot follows in RViz. This is the "H" in HIL.

While the deadman is NOT held, a benign "policy" action drives the arm (zero by
default, or a scripted circle / random). The instant the deadman is held and the
leader is fresh, GelloIntervention.step() replaces the policy action with the
anchored leader delta and reports it in info["intervene_action"] ([-1,1]^7, the
exact value executed — buffer-correctness invariant, see README). Release hands
control back to the policy instantly.

Terminals (see serl_ur_infra/README.md "RViz fake RL 테스트"):

  T1  mock UR + RViz:
      ros2 launch gello_policy ur_control_fake_safe.launch.py \
          ur_type:=ur7e robot_ip:=0.0.0.0 use_fake_hardware:=true \
          initial_joint_controller:=forward_position_controller launch_rviz:=true
  T2  fake cameras (env defaults subscribe to these topics):
      ros2 run gello_policy fake_diffusion_observations
  T3  REAL GELLO leader publisher (publishes /gello/joint_states from the
      physical leader arm — NO robot bridge, this only publishes leader joints):
      ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
          <ws>/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
  T4  this script:
      python3 tests/run_rviz_hil.py                    # zero policy, human takes over
      python3 tests/run_rviz_hil.py --policy scripted  # autonomous circle + human
      python3 tests/run_rviz_hil.py --policy random

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
from ur_env.envs.wrappers import (  # noqa: E402
    GelloIntervention,
    RosTopicDeadman,
)


class RvizHilConfig(DefaultUR7eEnvConfig):
    DRY_RUN = False              # mock hardware — publishing is the point
    TCP_POSE_SOURCE = "fk"       # avoid depending on tcp_pose_broadcaster here
    DISPLAY_IMAGE = False        # RViz is the display; flip on to see fake cams
    IMAGE_STALE_S = 2.0          # fake camera node may publish slowly
    RESET_JOINTS = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    # Mock hardware boots at all-zeros, farther than the real-robot guard
    # allows. Mock-only relaxation — do NOT copy into a real-robot config.
    RESET_MAX_DIST_RAD = 7.0
    RESET_TIMEOUT_S = 30.0
    MAX_EPISODE_LENGTH = 200     # override with --max-steps

    # ---- HIL feel: raise the intervention speed ceiling (mock-only) ---- #
    # Intervention chase speed is hard-capped at ACTION_SCALE*HZ — the human
    # cannot exceed the policy's own speed limit (executed==stored invariant).
    # The base default (0.1 m/s) feels crippled next to the 1:1 teleop bridge,
    # which is what "GELLO barely moves the RViz arm" reports are. The caps
    # come in three coupled layers; raise them TOGETHER or the next layer
    # silently eats the speed:
    #   ACTION_SCALE*HZ  ->  governor (~120% of scale)  ->  upsampler slew
    ACTION_SCALE = np.array([0.03, 0.10, 1.0])   # 0.3 m/s, 1.0 rad/s @ 10 Hz
    GOVERNOR = {"v_max": 0.36, "w_max": 1.2, "dq_step_max": 0.12}
    UPSAMPLER = {"hz": 250.0, "max_step_rad": 0.0048}  # 1.2 rad/s joint slew
    # Real-task values must be tuned deliberately (start from the config.py
    # defaults, not these) — faster caps mean harder physical crashes.


def scripted_action(t: float) -> np.ndarray:
    """Slow base-frame circle in x/y + gentle z — easy to eyeball in RViz.

    Identical to run_rviz_fake_rl.py so autonomous motion looks the same; the
    human can override any of it by holding the deadman.
    """
    a = np.zeros(7, dtype=np.float32)
    a[0] = 0.6 * np.sin(0.4 * t)
    a[1] = 0.6 * np.cos(0.4 * t)
    a[2] = 0.3 * np.sin(0.15 * t)
    return a


def policy_action(mode: str, env, t: float) -> np.ndarray:
    """The benign default the human overrides.

    zero     -> still arm; the human takes over entirely (HIL default).
    scripted -> the gentle base-frame circle (see autonomous motion too).
    random   -> small random actions.
    """
    if mode == "scripted":
        return scripted_action(t)
    if mode == "random":
        return env.action_space.sample() * 0.5
    return np.zeros(7, dtype=np.float32)  # zero


def _leader_arrived(env, timeout_s: float) -> bool:
    """Poll /gello/joint_states via the backend until a leader reading lands."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        arr, age = env.unwrapped.backend.get_gello_state()
        if arr is not None:
            return True
        time.sleep(0.1)
    return False


def _cameras_arrived(env, cfg, timeout_s: float):
    """Poll the fake-camera topics until every configured camera has a FRESH
    frame. reset() -> _get_obs() -> get_im() raises on a stale/absent camera,
    so without this preflight a missing T2 aborts at the first reset() with a
    raw traceback AFTER the runner already said everything was ready. Returns
    the first camera key still missing, or None once all are fresh.
    """
    deadline = time.time() + timeout_s
    missing = list(cfg.CAMERAS)
    while time.time() < deadline:
        missing = [
            k
            for k in cfg.CAMERAS
            if (lambda j, a: j is None or a > cfg.IMAGE_STALE_S)(
                *env.unwrapped.backend.get_image(k)
            )
        ]
        if not missing:
            return None
        time.sleep(0.1)
    return missing[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy",
        choices=["zero", "scripted", "random"],
        default="zero",
        help="benign action when the human is NOT intervening (default: zero)",
    )
    ap.add_argument(
        "--deadman",
        choices=["spacebar", "topic"],
        default="spacebar",
        help="deadman source: 'spacebar' (default, HOLD SPACEBAR) or 'topic' "
        "(/hil/deadman from the HIL GUI: ENGAGE/DISENGAGE + sensitivity slider)",
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override MAX_EPISODE_LENGTH (steps per episode)",
    )
    ap.add_argument(
        "--leader-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the GELLO leader before running without it",
    )
    args = ap.parse_args()

    cfg = RvizHilConfig()
    if args.max_steps is not None:
        cfg.MAX_EPISODE_LENGTH = args.max_steps

    env = UR7eEnv(fake_env=False, config=cfg)
    # Deadman source: spacebar (default, unchanged) or the /hil/deadman ROS
    # topic from the GUI. RosTopicDeadman subscribes on the backend's rclpy node
    # (spun in its bg thread), so no extra executor is needed.
    if args.deadman == "topic":
        deadman = RosTopicDeadman(env.unwrapped.backend._node)
    else:
        deadman = None  # -> GelloIntervention constructs SpacebarDeadman()
    env = GelloIntervention(env, deadman=deadman)  # needs backend + .controller

    # ---- wait for the mock robot feedback (same as run_rviz_fake_rl.py) ---- #
    print("waiting for /joint_states from the mock stack...")
    for _ in range(50):
        q, _, age = env.unwrapped.backend.get_joint_state()
        if q is not None:
            break
        time.sleep(0.2)
    else:
        sys.exit("no /joint_states — is the T1 mock launch running?")
    print("joint_states ok:", np.round(q, 2))

    # ---- additionally require the GELLO leader for the human to intervene ---- #
    print(f"waiting for /gello/joint_states (leader) up to {args.leader_timeout:.0f}s...")
    if _leader_arrived(env, args.leader_timeout):
        arr, age = env.unwrapped.backend.get_gello_state()
        print(f"leader ok: q={np.round(np.asarray(arr[:6]), 2)}  age={age:.2f}s")
    else:
        print(
            "!! no /gello/joint_states — T3 (gello_publisher) is NOT running.\n"
            "   Running WITHOUT a leader: the policy drives, but holding the\n"
            "   deadman will do nothing (no leader = no intervention). Start T3:\n"
            "     ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \\\n"
            "       <ws>/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml"
        )

    # ---- require fresh fake-camera frames (reset() decodes them or raises) ---- #
    print(f"waiting for fake cameras {list(cfg.CAMERAS)} up to {args.leader_timeout:.0f}s...")
    missing = _cameras_arrived(env, cfg, args.leader_timeout)
    if missing is not None:
        env.close()
        sys.exit(
            f"no fresh frame on camera '{missing}' ({cfg.CAMERAS[missing]}).\n"
            "   Start T2 first:  ros2 run gello_policy fake_diffusion_observations\n"
            "   (reset() decodes every camera, so the run cannot start without it.)"
        )
    print("cameras ok")

    if args.deadman == "topic":
        engage_help = (
            "  Use the HIL GUI ENGAGE/DISENGAGE button + sensitivity slider\n"
            "  (run ros2_ur_ws/run_hil_gui.sh) to take over, move the GELLO,\n"
            "  DISENGAGE to hand control back to the policy. (ESC terminates.)"
        )
    else:
        engage_help = (
            "  HOLD SPACEBAR to engage, move the GELLO, RELEASE to hand\n"
            "  control back to the policy. (ESC terminates the episode.)"
        )
    print(
        "\n================= OPERATOR INSTRUCTIONS =================\n"
        f"  deadman: {args.deadman}\n"
        f"{engage_help}\n"
        f"  policy-when-idle: {args.policy}   episodes: {args.episodes}\n"
        "========================================================\n"
    )

    for ep in range(args.episodes):
        print(f"\n=== episode {ep}: go_to_reset (watch RViz home) ===")
        obs, _ = env.reset()
        t0 = time.time()
        prev_intervening = None
        while True:
            t = time.time() - t0
            a = policy_action(args.policy, env, t)
            obs, rew, done, trunc, info = env.step(a)

            intervening = "intervene_action" in info
            step_n = env.unwrapped.curr_path_length
            # HUD: throttle to ~every 10 steps, but always print on a
            # POLICY<->INTERVENING transition so the handoff is visible.
            if intervening != prev_intervening or step_n % 10 == 0:
                tcp = obs["state"]["tcp_pose"][:3]
                held = info.get("held")
                reason = info.get("reject_reason")
                tag = "INTERVENING" if intervening else "POLICY     "
                line = (
                    f"  step {step_n:3d}  {tag}  "
                    f"tcp=[{tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}]  "
                    f"held={held} reject={reason}"
                )
                if intervening:
                    _, age = env.unwrapped.backend.get_gello_state()
                    ia = info["intervene_action"]
                    line += (
                        f"\n             leader_age={age:.2f}s  "
                        f"intervene_action={np.round(ia, 2)}"
                    )
                print(line)
                prev_intervening = intervening
            if done:
                break
        print(f"episode {ep} done ({env.unwrapped.curr_path_length} steps)")

    env.close()
    print("\nPASS — mock HIL loop ran; check RViz motion + spacebar handoffs")


if __name__ == "__main__":
    main()
