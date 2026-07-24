# UR7e HIL-SERL local actor adapter

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
files when the task config has `buffer_period > 0`.  Retaining the metadata
inside the learner's in-memory replay buffer requires a separate server-side
schema extension.

## Launch

The task configuration must produce the same observation/action spaces as the
learner.  A local mapping module can be supplied without modifying upstream:

```bash
python serl_ur_infra/scripts/train_rlpd_actor.py \
  --actor \
  --exp_name <task> \
  --checkpoint_path <path> \
  --ip 127.0.0.1 \
  --ur_config_module <python.module.with.CONFIG_MAPPING>
```

Policy inference runs locally.  Agentlace sends queued transitions to the
learner at episode boundaries and receives updated policy parameters on its
broadcast channel.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  serl_ur_infra/tests/test_intervention_metadata.py \
  serl_ur_infra/tests/test_rlpd_actor_adapter.py
```
