# Remote policy fake-UR7e validation

This validation assumes the generic gRPC policy server is already running. It
does not start, stop, or otherwise own that server.

Run from the source workspace after building and sourcing the ROS overlay:

```bash
ros2_ur_ws/src/gello_policy/scripts/run_remote_policy_fake_ur7e_validation.sh \
  /absolute/path/to/policy_validation_params.yaml 127.0.0.1 50051
```

The parameter file must contain the expected gRPC model contract for the server
being tested. The runner first checks TCP reachability and performs a real
synthetic-observation inference roundtrip. Only then does it launch the mock UR7e.

The ROS validator checks public interfaces only: stable initial HOLD controller
commands, successful `start_execution`, finite post-arm commands, wrap-aware
command-to-mock-joint tracking in UR joint order, and stable continuous command
publication after `hold`. A policy is allowed to return the start pose, so measured
motion is logged as a diagnostic and is not a pass gate. RViz is optional via
`LAUNCH_RVIZ=true`; the default is headless.

Observation-fault injection is intentionally deferred. The existing fake
observation publisher is unchanged, and adding a second publisher on the same
camera topics would not provide deterministic fault ownership. A future test
should use a validation-only publisher with explicit pause/corruption services
and assert the leader's fail-silent transition separately.
