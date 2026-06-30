"""Scripted MuJoCo demo of the Franka Emika Panda (no GELLO hardware).

This proves the Panda moves in MuJoCo using a purely scripted joint trajectory.
It does NOT touch the GELLO Dynamixel / serial port at all -- pure simulation.

The script:
  * loads ``panda.xml`` directly,
  * reads actuator control ranges and joint ranges from the model and clamps
    all commands to stay within safe limits,
  * drives a smooth sine-sweep over the 7 arm actuators plus an open/close
    cycle on the gripper actuator,
  * offscreen-renders frames and writes an animated GIF + a few PNG keyframes,
  * prints model info and sampled qpos so motion can be confirmed.

Run with the project venv::

    .venv/bin/python scripts/sim_panda_scripted_demo.py
"""

import os

import numpy as np

# Offscreen rendering needs a headless GL backend. Pick one before importing
# mujoco's renderer; we try egl first and fall back to osmesa.
_OUT_DIR = (
    "/tmp/claude-1002/-home-theo-lab-gello-software/"
    "d97069ca-6f4c-4ccc-9c54-092f7cd02ca0/scratchpad"
)
_XML_PATH = "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
_GIF_PATH = os.path.join(_OUT_DIR, "panda_scripted_demo.gif")


def _make_renderer(model, height=480, width=640):
    """Create a mujoco.Renderer, trying egl then osmesa GL backends."""
    import mujoco

    last_err = None
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            print(f"[render] offscreen backend OK: MUJOCO_GL={backend}")
            return renderer
        except Exception as exc:  # noqa: BLE001 - report and try fallback
            print(f"[render] backend {backend!r} failed: {exc}")
            last_err = exc
    raise RuntimeError(f"No working offscreen GL backend: {last_err}")


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)

    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(_XML_PATH)
    data = mujoco.MjData(model)

    n_arm = 7  # actuator1..7 drive the 7 arm joints
    grip_act = model.nu - 1  # last actuator is the coupled gripper/hand

    ctrlrange = model.actuator_ctrlrange.copy()
    print(f"[model] nu={model.nu} nq={model.nq} nv={model.nv} "
          f"timestep={model.opt.timestep}")
    print("[model] actuator ctrlranges:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  act{i} {name:>10}: [{ctrlrange[i, 0]:.4f}, {ctrlrange[i, 1]:.4f}]")

    # Start from the model's documented "home" pose for the arm.
    if model.nkey > 0:
        home_ctrl = model.key_ctrl[0].copy()
    else:
        home_ctrl = np.zeros(model.nu)
    arm_home = home_ctrl[:n_arm].copy()

    # Conservative per-joint oscillation amplitudes (rad), kept small.
    amplitudes = np.array([0.5, 0.4, 0.5, 0.4, 0.5, 0.5, 0.5])
    # Slightly different frequencies so the motion looks like a sweep.
    freqs = np.array([0.6, 0.5, 0.7, 0.55, 0.8, 0.65, 0.9])

    # Reset to home keyframe so qpos starts at the home pose.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    renderer = _make_renderer(model)

    duration = 7.0  # seconds of sim time
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    frame_interval = 0.03  # capture a frame every ~30 ms
    steps_per_frame = max(1, int(frame_interval / dt))

    frames = []
    qpos_samples = []  # (sim_time, qpos copy) for evidence of motion
    sample_every = max(1, n_steps // 8)

    for step in range(n_steps):
        t = step * dt

        # Smooth sine sweep around home for the arm.
        arm_cmd = arm_home + amplitudes * np.sin(2.0 * np.pi * freqs * t)
        # Clamp each arm command to its actuator ctrlrange (safe limits).
        arm_cmd = np.clip(arm_cmd, ctrlrange[:n_arm, 0], ctrlrange[:n_arm, 1])
        data.ctrl[:n_arm] = arm_cmd

        # Gripper open/close cycle within its ctrlrange (0..255 here).
        g_lo, g_hi = ctrlrange[grip_act]
        g_mid = 0.5 * (g_lo + g_hi)
        g_amp = 0.5 * (g_hi - g_lo)
        grip_cmd = g_mid + g_amp * np.sin(2.0 * np.pi * 0.3 * t)
        data.ctrl[grip_act] = np.clip(grip_cmd, g_lo, g_hi)

        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"Non-finite qpos at step {step} (instability)")

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

        if step % sample_every == 0:
            qpos_samples.append((t, data.qpos[:n_arm].copy()))

    # Final sample.
    qpos_samples.append((n_steps * dt, data.qpos[:n_arm].copy()))

    print(f"[sim] stepped {n_steps} steps ({n_steps * dt:.2f}s), "
          f"captured {len(frames)} frames")
    print("[sim] sampled arm qpos over time (rad):")
    for t, q in qpos_samples:
        print(f"  t={t:5.2f}s  " + " ".join(f"{v:+.3f}" for v in q))

    first = qpos_samples[0][1]
    last = qpos_samples[-1][1]
    max_delta = np.max(np.abs(last - first))
    print(f"[sim] max |qpos[end]-qpos[start]| over arm joints = {max_delta:.4f} rad")
    if max_delta < 1e-3:
        print("[sim] WARNING: joints barely moved!")
    else:
        print("[sim] motion confirmed: joints changed position.")

    # Write the GIF (~33 fps -> matches the 30ms capture interval).
    imageio.mimsave(_GIF_PATH, frames, duration=frame_interval, loop=0)
    print(f"[out] wrote GIF: {_GIF_PATH} ({len(frames)} frames)")

    # Save a few PNG keyframes.
    keyframe_idx = np.linspace(0, len(frames) - 1, 4).astype(int)
    for n, idx in enumerate(keyframe_idx):
        png_path = os.path.join(_OUT_DIR, f"panda_scripted_demo_key{n}.png")
        imageio.imwrite(png_path, frames[idx])
        print(f"[out] wrote PNG keyframe: {png_path}")

    renderer.close()


if __name__ == "__main__":
    main()
