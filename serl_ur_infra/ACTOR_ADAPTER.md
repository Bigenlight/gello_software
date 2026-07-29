# UR7e HIL-SERL local actor adapter

> ⚠️ **Scope: this is NOT the deployed path.** This page documents
> `scripts/train_rlpd_actor.py`, which runs policy inference **locally** and
> ships transitions over **agentlace**. The deployed HIL-SERL path on this rig
> runs inference on Kanu and speaks gRPC:
> `scripts/run_remote_rlpd_actor.py` → `REMOTE_ACTOR_GRPC.md` →
> `HIL_SERL_KANU_RUNBOOK_KO.md`.
>
> The two are not interchangeable. `create_actor_network()` raises
> `NotImplementedError` for `network.type='agentlace'`, so nothing in the gRPC
> stack can fall back to the transport described here. Verified 2026-07-29.

`scripts/train_rlpd_actor.py` runs on the robot laptop.  It reuses upstream
HIL-SERL for task/agent initialization but owns the actor transition boundary,
so `third_party/hil-serl` remains unmodified.

## Runtime data flow

```text
policy action ──► GelloIntervention/env ──► next observation
                        │
                        ├─ no intervention: execute policy action
                        └─ intervention:    execute GELLO action

every transition ─────────────► actor_env       ─► online replay buffer
intervention transitions only ► actor_env_intvn ─► demo/intervention buffer
```

An intervention transition is intentionally sent to both datastores, matching
upstream HIL-SERL's RLPD sampling behavior.

## Transition contract

The upstream learner keys remain unchanged:

```python
{
    "observations": ...,
    "actions": ...,              # action actually executed by the robot
    "next_observations": ...,
    "rewards": ...,
    "masks": ...,
    "dones": ...,
}
```

The local actor adds:

```python
{
    "policy_actions": ...,       # outermost policy-frame action before override
    "intervened": np.uint8(...), # exactly 0 or 1
}
```

`grasp_penalty` is also copied when supplied by the environment.

The current upstream replay buffer ignores unknown keys, so SAC training
remains compatible.  The enriched transitions are retained in actor pickle
files **only when the task config has `buffer_period > 0`**.  Retaining the
metadata inside the learner's in-memory replay buffer requires a separate
server-side schema extension.

> 🪤 **The shipped UR task config sets `buffer_period = 0`**
> (`ur_experiments/cube_in_cup.py`, verified 2026-07-29), and there is no CLI
> flag to override it on either actor entry point. So **no pickle is ever
> written**, even if you pass `--checkpoint_path` / `--checkpoint-path`. The
> guard is `if buffer_period and ...` in `ur_env/rlpd_actor.py` and
> `ur_env/remote_actor.py`; a zero value short-circuits it silently.
>
> This is the direct reason there is still **no canonical robot demo**, which is
> the precondition for the production learner run. To produce one, raise
> `buffer_period` in the task config *and* supply a checkpoint path — the code
> raises `checkpoint_path is required when buffer_period > 0` if you do only the
> first.

## Launch

The task configuration must produce the same observation/action spaces as the
learner.  A local mapping module can be supplied without modifying upstream:

Source the ROS overlay first — this entry point builds `UR7eEnv`, which needs
`rclpy` / `ur_gello_bringup` / `cv2` / `pyrealsense2` — and use the actor venv
rather than the system `python3`:

```bash
set +u
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
set -u

/home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/train_rlpd_actor.py \
  --actor \
  --exp_name cube_in_cup \
  --checkpoint_path <path> \
  --ip 127.0.0.1 \
  --ur_config_module ur_experiments.mappings
```

Unlike the gRPC actor, this path additionally needs upstream's JAX/agentlace
stack (it constructs the SAC agent locally), which the robot laptop deliberately
does not carry — inference lives on the GPU server. Treat a missing-`jax`
ImportError here as expected, not as a broken install.

`ur_experiments/mappings.py` exports `CONFIG_MAPPING = {"cube_in_cup":
CubeInCupConfig}`; the entry point merges it over upstream's registry. The flags
are absl (`--exp_name`, `--checkpoint_path`, `--ur_config_module`, underscores),
unlike the gRPC actor's argparse flags (`--exp-name`, `--ur-config-module`,
dashes). They are not interchangeable.

`--actor` is required and `--learner` must be absent; the entry point raises
otherwise.

Policy inference runs locally.  Agentlace sends queued transitions to the
learner at episode boundaries and receives updated policy parameters on its
broadcast channel.  This is the legacy transport — see the scope banner.

## Tests

Do **not** run these with the system `python3`: its grpcio is `1.30.2` and the
suite hangs at 100% CPU with no error. Use the actor venv with the ROS overlay
on `PYTHONPATH` (this exact invocation produced `10 passed` on 2026-07-29):

```bash
set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q -p no:cacheprovider \
  serl_ur_infra/tests/test_intervention_metadata.py \
  serl_ur_infra/tests/test_rlpd_actor_adapter.py
```

The full `serl_ur_infra/tests` baseline with the same environment is **333
passed / 11 skipped**. Judge by the passed count, not by "green": dropping
`third_party/hil-serl/serl_launcher` from `PYTHONPATH` silently reduces it to
300 passed / 13 skipped, and an uninitialised submodule gives 296 / 17 — neither reports a failure.
