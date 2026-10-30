# Real evaluation diagnostic video

`render_eval_video.py` can render one self-contained real robot recording
without starting a policy or contacting the robot:

```bash
PYTHONPATH=gello python3 -m sim_collect.eval.render_eval_video \
  --real-h5 /path/to/ep_0042.h5 \
  --out /path/to/ep_0042_diagnostic.mp4
```

`--out` is the final MP4 path. The renderer writes
`ep_0042_diagnostic.partial.mp4`, verifies it, and only then atomically
renames it to the requested path. It produces a 1280x720, H.264/yuv420p,
30 fps video with exactly one frame per recorded request.

## Accepted `real_eval/v1` HDF5 contract

The file must have `complete=true` and `schema="real_eval/v1"` as attributes
on the root (or `/meta` or `/metadata`). `policy`, `checkpoint`, `sampler`,
`finalize_reason`, and `outcome` are optional attributes in those locations.
If `outcome` is absent, the dashboard says `UNKNOWN`; it never derives an
outcome from the video.

Request datasets may be at the root, under `/requests`, or under `/frames`:

| Required field | Meaning |
| --- | --- |
| `request_idx` | Non-empty, strictly increasing integer IDs for output frames. They do not need to start at zero or be contiguous. |
| `cam1_jpeg`, `cam2_jpeg` | One JPEG byte array per request row. Each decoded image is resized to 640x360 only if needed. |

Optional request fields are `state`, `output`, `gripper`, `chunk_step`, and
`t_rel_s`. The logger may additionally preserve request-rate
`model_input`/`input`, `model_output`/`action`, and `noise` values. They must
have exactly one row per request. Absent fields are displayed as `missing`;
the renderer does not derive a gripper, elapsed time, model input, output, or
noise value from another field.

Every recording also needs a `/decisions` group (the legacy `/decision` alias
is accepted):

| Required field | Meaning |
| --- | --- |
| `decisions/request_idx` | Strictly increasing integer request IDs. Every ID must occur exactly in `request_idx`. |
| `decisions/decision_idx` | Strictly increasing integer decision IDs, one per decision row. |

All other datasets in `/decisions` are values recorded for that decision and
must have one row per decision. A decision is joined only to the matching
`request_idx`; its values are then displayed as a zero-order hold until the
next decision. The renderer never uses nearest timestamps, offsets, index
compaction, or a frame-number-derived elapsed time. It rejects incomplete
files, duplicate/reordered IDs, missing camera rows, and a decision that
cannot be joined exactly.

## Dashboard fields

The top row is the stored `cam1` and `cam2` JPEGs. The bottom row includes the
request/decision IDs, chunk step, recorded input/output/gripper, elapsed time,
policy/checkpoint/sampler/schema, finalization reason, and outcome. Every
unrecorded item is labelled `missing`.

The graph uses only critic fields present in the matching decision rows:
`q1`/`Q1`, `q2`/`Q2`, or raw `q_heads` with `q_argmax`/`chosen_idx`, plus one
of `q_chosen`, `q_aggregate`, `q_selected`, or `aggregate_q`. For raw
`q_heads`, it shows exactly head 1 and head 2 at the logged selected candidate;
it never averages them or chooses a candidate itself. Missing series remain
unavailable rather than being estimated.

- `policy=IFQL` shows both selected critic heads (when logged), aggregate
  selected Q, `K`/`k`, selected candidate index, logged mean/std/min/max/Q
  spread, candidate/noise norms, and recorded latency.
- `policy=SVF` shows PRNG, recorded noise/noise norm, and chunk norm. It
  explicitly does not invent a critic panel when Q was not recorded.
- `policy=DSRL` shows recorded latent z, z norm, bound fraction, maximum z
  magnitude, sampler, noise scale, and latency. Q is graphed only if the
  decision rows include it.

## Encoder availability and verification

The encoder resolves `ffmpeg` from `PATH` first. On inference servers without
system ffmpeg/ffprobe it falls back to `imageio_ffmpeg.get_ffmpeg_exe()`.
The produced partial file is decoded with ffmpeg and inspected through OpenCV
before publication. The check requires H.264, yuv420p, 30 fps, and the exact
recorded request-frame count; a failed check leaves no final MP4.
