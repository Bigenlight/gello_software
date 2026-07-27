# UR7e HIL-SERL remote-inference actor adapter

`scripts/train_rlpd_actor.py` runs on the robot laptop without a local JAX
policy.  Policy inference and learning run as separate server processes, while
the robot laptop owns the environment, intervention selection, and transition
boundary.  `third_party/hil-serl` remains unmodified.

## Runtime data flow

```text
server policy action ──► GelloIntervention/env ──► next observation
                        │
                        ├─ no intervention: execute policy action
                        └─ intervention:    execute GELLO action

every transition ─────────────► actor_env       ─► online replay buffer
intervention transitions only ► actor_env_intvn ─► demo/intervention buffer

learner parameter broadcast ──► server inference process (never the laptop)
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
    "timestamp_ns": np.int64(...), # env observation-finalization Unix epoch
}
```

`grasp_penalty` is also copied when supplied by the environment.

`UR7eEnv` creates `timestamp_ns` when it finishes assembling each observation
and returns it through Gym's `info` mapping.  The actor copies the step result's
timestamp unchanged into the transition.  It is transition metadata, not part of
`observations["state"]`.  Keeping it outside the observation prevents SERL's
state flattener from treating wall-clock time as a learned policy feature.

The current upstream replay buffer ignores unknown keys, so SAC training
remains compatible.  The enriched transitions are retained in actor pickle
files when the task config has `buffer_period > 0`.  Retaining the metadata
inside the learner's in-memory replay buffer requires the metadata-aware
learner entry point:

```bash
python serl_ur_infra/scripts/train_rlpd_learner.py \
  --learner \
  --exp_name <task> \
  --checkpoint_path <path> \
  --demo_path <demos.pkl>
```

That entry point allocates `policy_actions`, `intervened`, `timestamp_ns`, and
`policy_version` columns in both server replay buffers.  Legacy demonstrations
without these keys remain loadable and receive `timestamp_ns == -1`.

## Launch

The task configuration must produce the same observation/action spaces as the
learner.  A local mapping module can be supplied without modifying upstream:

```bash
python serl_ur_infra/scripts/train_rlpd_actor.py \
  --exp-name <task> \
  --checkpoint-path <path> \
  --learner-ip 127.0.0.1 \
  --inference-ip 127.0.0.1 \
  --ur-config-module <python.module.with.CONFIG_MAPPING>
```

The actor uses two independent Agentlace connections:

- inference request `observation + env timestamp -> policy action + version`
- asynchronous transition batches to the learner datastores

The transition uploader retries independently and defaults to a one-second
flush interval.  An inference timeout does not silently fall back to an old
action.

On the GPU server, start the metadata learner and the inference process:

```bash
python serl_ur_infra/scripts/train_rlpd_learner.py \
  --learner --exp_name <task> --checkpoint_path <path> \
  --demo_path <demos.pkl> \
  --ur_config_module <python.module.with.CONFIG_MAPPING>

python serl_ur_infra/scripts/run_rlpd_inference.py \
  --exp-name <task> \
  --learner-ip 127.0.0.1 \
  --ur-config-module <python.module.with.CONFIG_MAPPING>
```

The learner remains on ports 5588/5589.  The inference request server defaults
to 5590.  Parameter broadcast 5589 is server-local; it is not forwarded to the
robot laptop.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  serl_ur_infra/tests/test_intervention_metadata.py \
  serl_ur_infra/tests/test_rlpd_actor_adapter.py
```
