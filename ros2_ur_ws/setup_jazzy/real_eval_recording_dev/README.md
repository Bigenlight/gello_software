# SVF real evaluation recorder

Enabled by default for `run_ur7e_svf_carrot_ifqlparity.sh` on the next launch.
Running processes are not patched. Restart cameras/viewer and policy launcher at
an operator-chosen safe point to load the changes.

Terminal 1: `./launch_cameras.sh` — existing camera display now includes
SUCCESS / FAILURE / UNKNOWN for the latest accepted recorded episode. Its ID is
displayed. Buttons are disabled while that episode is writing/rendering.

Terminal 2: `HEADLESS=true ./run_ur7e_svf_carrot_ifqlparity.sh`.
Existing START EXECUTION and HOLD buttons or ROS services control the interval.
FAULT, go-to-start, or shutdown also end it. Rejected starts carry an explicit
termination reason and are excluded from the tagging panel and trial counts.

Default destination:
`ros2_ur_ws/log/svf_ifqlparity/run_<UTC>_pid<PID>/episodes/ep_<UUID>__unlabeled/`

- recording.h5: timestamped numeric JSON streams and exact JPEG bytes.
- episode.json: episode ID/state, timestamps, flags/checkpoint/norm/source hashes,
  encoder identity, reset seed/counter and existing server NPZ path.
- outcome.json: operator label and correction history, authoritative over suffix.
- cam1.mp4, cam2.mp4: ACT input views, 30 FPS, timestamp-resampled and letterboxed.
- diagnostic.mp4: paired images, input state, policy action and clamp delta.
- video_frames.json: video frame to original ACT image timestamp/request ID.
- summary.json: duration, command/clamp counts and roundtrip latency percentiles.
- render_status.json / render.log: independent video rendering status/errors.

HDF5 records ACT request/reply, native camera samples, live joints/velocity/effort,
gripper state/command, leader post-clamp command and controller commands. Source
camera stamps and observation arrival times are retained separately. Host monotonic
timestamps and ROS header timestamps must not be directly subtracted.
Existing server NPZ (at run root) retains chunks/queue IDs/refill timings and is
linked by RESET counter. It finalizes independently of HDF5. A link is not a claim
that the NPZ is already durably written.

No changes to policy inputs, random seeds, action queue or safety limits. Disk/queue
errors invalidate the recording and request existing HOLD on the next timer tick.
Task outcome is operator-labelled; HOLD is not an automatic failure label.

Disable: `SVF_EPISODE_RECORDING=0 HEADLESS=true ...`.
The parity wrapper keeps legacy REAL_EVAL_RECORDING=0 because the staged SVF adapter
does not support its HDF5 CLI. The new recorder runs on the ROS side.
Custom destination: match IFQL_LOG_ROOT in policy terminal with SVF_EVAL_LOG_ROOT
in camera terminal. Root paths must be absolute.

CLI label correction (also useful for older episodes):
`conda run -n il python recording.py /absolute/episode/path failure --note 'reason'`
UUID and contents remain; suffix changes. Label history is retained. A stale tag
lock after a killed renderer requires manual inspection; do not remove a live lock.

Validation: unit tests, MP4 decode, viewer target identity, disk errors/overflow,
actual Jazzy class START/clamp/HOLD in domain 231 with in-memory inference socket,
and a 3-second synthetic 200 Hz state/controller + 30 Hz image throughput check.
These are not hardware validation. Do a supervised episode after restart.

Not included yet: online Q/V/gradient/latent diagnostics, full flow noise keys and
features, F/T/TCP/safety-mode telemetry, per-stage inference timings, automatic crash
recovery of HDF5/locks, older-episode browser. Native topics are captured at callback
receipt, not guaranteed lossless ROS transport. Writer uses a bounded thread;
rendering uses a separate process and may overlap the next trial. Long-duration
hardware load and actual inference latency still require measurement.
