# DSRL real-evaluation gate

현재 5070에는 carrot/orange real-demo DSRL 후보가 `setup_jazzy/staged_models/` 아래 stage되어 있음. 둘 다 non-privileged actor/critic이고 아직 **실물 rollout 성능은 미검증**임. 직접 경로를 조립하기보다 [[REAL_EVAL_QUICKSTART]]의 profile을 사용함.

```bash
./run_real_policy.sh dsrl-carrot --dry-run
./run_real_policy.sh dsrl-orange --dry-run
```

`run_ur7e_dsrl_real.sh` 자체는 계속 default checkpoint를 추측하지 않으며, explicit `real_serve_meta.json` gate를 요구함.

```bash
DSRL_DRY_RUN=1 \
DSRL_RUN_DIR=/absolute/path/to/staged-dsrl \
DSRL_SERVER_PY=/absolute/path/to/vision_carrot/dsrl/dsrl_server.py \
DSRL_PY=/absolute/path/to/python \
./run_ur7e_dsrl_real.sh
```

`DSRL_DRY_RUN=1`은 manifest만 검증하고 해석된 checkpoint/base/norm hash를 출력한 뒤 끝남. ROS source, ZMQ bind, robot process 전부 안 뜸. 실제 실행도 같은 preflight가 먼저 돌고, `dsrl_server.py`가 bind 전에 manifest를 한 번 더 검증함.

## `real_serve_meta.json` contract

Manifest는 `DSRL_RUN_DIR/real_serve_meta.json`에 있어야 하고, 최소한 아래 값이 **명시적으로** 들어가야 함.

```json
{
  "schema": "dsrl-real-serve/v1",
  "status": "staged",
  "placeholder": false,
  "policy_family": "dsrl",
  "domain": "real",
  "nonprivileged_actor": true,
  "use_critic_obs": false,
  "actor_use_critic_obs": false,
  "task": "<real-task-id>",
  "sampler": "dsrl_det",
  "runtime": {"action_dim": 7, "chunk_dim": 56, "chunk_horizon": 8, "n_action_steps": 24, "port": 5596},
  "bindings": {
    "checkpoint_sha256": "<same SHA256 as artifacts.checkpoint>",
    "base_checkpoint_sha256": "<same SHA256 as artifacts.base_checkpoint>",
    "norm_stats_sha256": "<same SHA256 as artifacts.norm_stats>",
    "flags_sha256": "<same SHA256 as artifacts.flags>"
  },
  "artifacts": {
    "flags": {"path": "flags.json", "sha256": "<sha256>"},
    "checkpoint": {"path": "params_<step>.pkl", "step": 1, "sha256": "<sha256>"},
    "base_checkpoint": {"path": "/absolute/path/params_<base-step>.pkl", "step": 1, "sha256": "<sha256>"},
    "norm_stats": {"path": "norm_stats_*.json", "sha256": "<sha256>"},
    "deploy_yaml": {"path": "ifql_deploy.yaml", "task": "<real-task-id>", "sha256": "<sha256>"}
  },
  "start_pose": [0, 0, 0, 0, 0, 0]
}
```

`flags.json`과 `flags.agent`도 `nonprivileged_actor: true`, `use_critic_obs: false`, `actor_use_critic_obs: false`를 모두 명시해야 함. `flags.domain`과 norm stats의 `domain`도 `real`이어야 함. checkpoint/base/norm은 manifest artifact hash와 `bindings` hash가 전부 동일해야 하고, flags의 `base_run_dir`, `base_epoch`, `norm_stats`, `real_serve_meta` 포인터도 같은 artifact를 가리켜야 함.

## Recorder trace

DSRL은 canonical episode NPZ와 `refill_stats.jsonl`을 남김. 각 refill row에는 `trace_schema: dsrl-refill-trace/v1`, exact float32 `latent_z_f32`, base sampler의 lossless PCG64 state bytes 또는 non-base sampler의 JAX key/subkey/next-key를 기록함. Critic input이 actor observation과 같은 nonprivileged contract일 때는 `q_action_heads`와 `q_latent_heads` 전체도 기록함. critic observation을 실물에서 만들 수 없으면 `q_evaluated: false`와 이유만 기록하며, diagnostics가 action selection을 바꾸지는 않음.

공통 `real_eval/v1` recorder도 기본 활성화됨. 따라서 manifest 검증을 통과한 실물 DSRL eval은 canonical NPZ/JSONL과 함께 request JPEG, state/output, exact latent/PRNG, critic heads를 담은 HDF5 및 diagnostic MP4를 남김.
