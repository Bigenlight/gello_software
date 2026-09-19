# Cube stack dataset — 2026-09-19

Task: `stack the red cube on the blue cube`.

- Raw: https://huggingface.co/datasets/Bigenlight/cube_stack_raw
- LeRobot v3: https://huggingface.co/datasets/Bigenlight/cube_stack_lerobot_v3
- Source: `ros2_ur_ws/gello_logs/cube_stack/take_*/`.
- 60 episodes; 20,177 cam1 master frames, 20,175 raw cam2 frames; about 673 seconds.
- RGB 1280×720 at 30 FPS, cam1 scene / cam2 wrist, no depth.
- 7-D state = measured UR joints + gripper position. 7-D action = commanded UR joints + gripper command. Joint units are radians; gripper 0=open, 1=closed.

## Collection tags

Original take names are retained. Two takes have number 01 but distinct timestamps; use the full name or episode index as identity.
Exact folder suffixes are stored as `folder_tag` and `folder_labels`; `recovery_label="recover"` groups recovery variants. Counts: 48 untagged, 9 recover, 3 recover_many. The full mapping is in raw `source_takes.json` and LeRobot `meta/source_takes.json`. No tagged take is dropped. Collection tags do not certify success/failure, and conversion validation does not establish robot policy success.

## Conversion and reproducibility

Use `scripts/dataset/convert_takes_to_lerobot.py` in a separate LeRobot 0.6.1 environment. The fixed-recorder recipe aligns camera/joint `stamp_s` and command/gripper `t_rel_s` on the cam1 capture clock, preserving all cam1 frames without subtracting servo tracking lag. Raw copies are checked against SHA-256 in `checksums.json`. `dataset_stats.json` and `DATA_DICTIONARY.md` describe raw signals.

The independent `scripts/dataset/validate_takes_conversion.py` re-derives all state/action rows and compares six decoded video samples per episode per camera. Result: 43 PASS, 0 FAIL, 0 WARN, 0 SKIP; maximum state/action deviation 0.0. For 360 video samples per camera, minimum correlation was 0.99756 (cam1) / 0.99955 (cam2), with no failing sample. The release report is stored as `meta/validation_2026-09-19.json`.

LeRobot's default train range exposes the full corpus; no paper evaluation split or outcome labels are inferred. Define episode-level splits before training and reuse them across methods.

Dataset cards contain first/middle/final cam1 frames from `take_01_20260919_212809` as a preview, explicitly distinguished from a separate setup photograph.

## Publication verification

Both datasets are public. Every release file was compared to the uploaded commit using size and SHA-256 for LFS files, or downloaded bytes for other files.

- Raw commit: `a79935cf37139735d3cf62fcad4351ea1ffeb526` (186 release files verified).
- LeRobot commit: `6b1b2353556f2dd571198ef3b5a89f64ae65e80d` (11 release files verified).
- LeRobot `v3.0` tag resolves to the exact LeRobot commit above.
