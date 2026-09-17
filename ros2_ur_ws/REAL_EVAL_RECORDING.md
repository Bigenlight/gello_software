# IFQL/SVF real-eval recording contract

`run_ur7e_ifql_real.sh` creates one atomic directory for every launch:

```text
log/ifql/real_YYYYMMDD/<task>_<sampler>/run_YYYYMMDDTHHMMSS.NNNNNNNNNZ_pid<PID>/
├── launch_manifest.json
├── server_stdout.log
├── hdf5/                  # Team 1 server HDF5 output
└── policy_render.mp4      # Team 1 renderer output
```

The timestamp is UTC and includes nanoseconds. `mkdir` must create this directory
exactly once; an already-present path is refused, never reused. The manifest begins
as `launching`, contains resolved server/ROS commands, host, policy metadata, input
artifact hashes, and planned output paths. It is updated to `finalized` after the
launcher has stopped its own ROS child and then its own server child; emitted HDF5 and
MP4 files receive SHA-256 hashes when present.

Recording is on by default. It can be disabled only by an explicit
`REAL_EVAL_RECORDING=0`; this is recorded in the manifest. `IFQL_LOG_DIR` remains a
compatibility alias for the per-task log root. Prefer `IFQL_LOG_ROOT` when choosing a
root and reserve `REAL_EVAL_RUN_DIR` for a deliberate, empty one-run path.

The launcher calls Team 1's server integration with this CLI contract:

```bash
--hdf5-log-dir "$REAL_EVAL_HDF5_LOG_DIR" \
--renderer-hook "$REAL_EVAL_RENDERER_HOOK" \
--renderer-output-path "$REAL_EVAL_MP4_PATH"
```

It also exports `REAL_EVAL_FFMPEG`, `REAL_EVAL_RUN_DIR`, and the three artifact-path
variables to the server. The default hook is `<directory containing ifql_server.py>/real_eval_renderer.py`.
Team 1 owns that hook and must make it produce the requested MP4 atomically (for
example, write `.part` then rename). This wrapper deliberately has no ROS/camera
recorder implementation and does not alter start poses, joint envelopes, timeouts,
or the ZMQ protocol.

Before any process starts, `setup_jazzy/real_eval_recording_preflight.py` verifies the
inference Python can import `h5py`, the hook exists, the chosen output root is
writable, free disk meets `REAL_EVAL_MIN_FREE_GIB` (default `10`), and no HDF5/MP4
partial conflict exists. The launcher runs this complete check before creating the run
directory, then repeats it after `mkdir`; a missing dependency therefore never leaves
an empty run directory. It resolves ffmpeg in this order:

1. explicit `REAL_EVAL_FFMPEG`;
2. `ffmpeg` on `PATH`;
3. `imageio_ffmpeg.get_ffmpeg_exe()` in `IFQL_PY` / `SVF_PY`.

The resolved executable must advertise `libx264`. `ffprobe` is intentionally not a
hard prerequisite: the 5070 inference Python's static `imageio_ffmpeg` binary has no
separate ffprobe binary. `IFQL_DRY_RUN=1` performs these checks, writes only the
launch manifest/run directory, prints both resolved commands, and starts neither
server nor ROS/camera process.

`run_ur7e_svf_real.sh` sets `policy_type=svf` and passes the staged run, qflow runtime,
server, real dataset, and norm-stat paths into the shared manifest before delegating to
the IFQL launcher. Its existing staged-SVF provenance gate remains first.
