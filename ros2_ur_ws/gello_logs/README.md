# gello_logs — recorded data

This folder is where `gello_recorder` (headless `run_recorder.sh` and the
`gello_recorder_gui`) writes its output: `session_<stamp>/` or
`take_<NN>_<stamp>/` directories, each containing `vectors.h5`, `cam1.mp4`,
`cam2.mp4`, and `metadata.json`. See
`../src/gello_recorder/README.md` for the exact format.

**The recorded data itself is no longer tracked in this git repo** (mp4/h5
files are large binary data and were bloating the repo). Everything recorded
so far is archived here instead:

https://drive.google.com/drive/folders/1Qn03uv1cMv3jbu4S_B-SS_koVjO876sn?usp=sharing

`.gitignore` ignores everything under `gello_logs/` except this README (and
the pre-existing `experiments/`/`replays/` reports+scripts, which are small
and still tracked normally). New recordings will accumulate here locally as
before; upload them to the Drive folder above and clean up locally whenever
disk space matters — nothing under `take_*/`/`session_*/` needs to be
committed to git going forward.
