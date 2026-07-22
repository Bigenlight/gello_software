# Offline Ensemble Study: Is Across-Sample Variance a Usable Uncertainty Signal?

> **STATUS (2026-07-20): study COMPLETE, result is LARGELY NEGATIVE, feature stays default-OFF.**
> The offline harness, the analysis library, and the three design fixes are all implemented and
> exercised on real data (739 in-distribution refills + 23 perturbation conditions x 30 paired
> observations, K=64, on this build PC's RTX 3060 12GB). The headline finding is that across-sample
> variance of this diffusion policy is a **global-photometric-corruption detector**, not an
> uncertainty estimate: it separates corrupted images from clean ones perfectly, and it is at the
> sampling-noise floor for the two OOD types that actually matter operationally (wrong-looking target
> object, novel arm pose). `DIFFUSION_ENSEMBLE_K` remains `0` and nothing on the real-time control
> path was touched. **No robot was run and no failure labels exist anywhere in this study** — every
> number below is input-OOD separability, which is a strictly weaker claim than failure prediction.
> This doc also carries a **correction notice** (§6) retracting a fabricated literature claim made
> earlier in this project.

## TL;DR

| | |
|---|---|
| Research question | Is across-sample variance of a diffusion policy a usable uncertainty signal, and can it tell "several valid options exist" apart from "the model has no idea"? |
| Short answer | **Partly, in a much narrower form than hoped.** It reliably detects global image corruption. It is blind to semantic OOD (recoloured target object) and to proprioceptive OOD (displaced joints). It cannot answer the multimodality-vs-ignorance question from second moments alone, and the shape probes that could are uncalibrated here. |
| Three design fixes shipped | horizon slicing (`v(h)` profile, exec-vs-tail), logger provenance + embedded normalization stats, arm/gripper separation. **Two of the three were vindicated by the data; horizon slicing was not** (§4). |
| Why offline first | No real-time deadline, so K can go far above 16 and no GPU is shared with robot control. This build PC has a **12GB** RTX 3060; the FAIL benchmark in [GELLO_DIFFUSION_ENSEMBLE_RECORDER.md](GELLO_DIFFUSION_ENSEMBLE_RECORDER.md) §6 refers to a **different, 6GB laptop** machine (§5). |
| Correction | An earlier literature review in this project attributed a "Diffusion Output Variance" baseline with specific TPR/TNR numbers to FAIL-Detect (arXiv:2503.08558). **No such baseline and no such numbers exist in that paper.** See §6. |
| Cost finding | **K=4 already matches K=64** on every effect this signal can detect. The K=16 default was ~4x more conservative than necessary. |
| Recommendation | Keep `DIFFUSION_ENSEMBLE_K=0`. If a monitor is ever built, scope it honestly as a camera-integrity tripwire, and benchmark it against a plain image-statistics check first — which would very likely catch the same faults for a fraction of the compute. |
| How to reproduce | See **[How to run the code](#how-to-run-the-code)** for the four scripts and copy-paste commands (`offline_ensemble.py`, `ood_sweep.py`, `ood_perturbations.py`, `ensemble_analysis.py`). |

## 1. The research question

The recorder doc ([GELLO_DIFFUSION_ENSEMBLE_RECORDER.md](GELLO_DIFFUSION_ENSEMBLE_RECORDER.md) §1) states the
motivating hypothesis: *more divergence across noise-seeded samples from the same observation = less
confident policy*. Its §8.5 already flagged the obvious confound, and that confound is the actual
research question:

> **Multi-modality confound**: high sample-divergence can mean "two equally valid grasps exist," not
> "the policy is unconfident in a bad way."

So there are two questions, and they are not the same:

1. **Is across-sample variance a usable uncertainty signal at all?** Does it rise when the
   observation leaves the training distribution, by enough to threshold on?
2. **Can it distinguish "multiple valid options exist" from "the model does not know"?** These are
   both high-variance states, and a robot should respond to them very differently (the first is fine;
   the second should stop the arm).

Question 2 has a discouraging structural answer that predates any experiment: **variance is a second
moment, and a second moment provably cannot tell two narrow modes from one wide unimodal blob.**
`ensemble_analysis.py`'s self-checks demonstrate this constructively rather than asserting it — they
build a 2-mode cloud and a 1-blob cloud with second moments matched to 1e-9, and show
`mean_pairwise_l2` agrees between them to within 0.7% while shape probes (dip test, ΔBIC) separate
them cleanly (ΔBIC +80.2 vs +3.4). The same self-checks verify the identity
`mean squared pairwise L2 == 2·trace(Cov)` to 1e-10, making the redundancy of the whole family of
second-moment metrics explicit.

That is why the analysis library carries separate shape probes (PCA structure, dip test, 1-vs-2
Gaussian ΔBIC) alongside the variance metrics. **The honest outcome for question 2 in this study is
"could not answer it":** the optional `diptest` package was absent, so `dip_test` fell back to the
uncalibrated bimodality-coefficient heuristic and returned **NaN for every p-value across all 739
refills**. The resulting 16.6% arm-only multimodality flag rate rests entirely on `BC > 5/9` and
`ΔBIC > 6`, neither of which is a calibrated test. Per the module's own docstring, `flagged=False` is
near-uninformative, so **83.4% unflagged is not evidence of unimodality**. Question 2 remains open.

Question 1 got a real, decisive answer, and most of this doc is about it.

## 2. Why offline first

Three independent reasons, in descending order of importance.

**(a) The online ensemble contends with robot control for the same GPU.** This is the finding that
stopped the online path: on the robot PC, even `K=1` failed the safety benchmark (contended p99
545ms against a 600ms `act_timeout_s`), because a single DDIM-10 refill already saturates that GPU's
compute and separate CUDA streams cannot conjure extra silicon. Offline there is no ZMQ round trip,
no `act_timeout_s`, and no robot. The GPU is free to be saturated.

**(b) No real-time deadline means K is unconstrained.** Online, K is bounded by whatever fits in the
leftover ~600–800ms per refill cycle, which is what pinned the design at K=16. Offline, K is bounded
only by patience. This study ran **K=64** throughout, and measured VRAM at k=128 to be only 149 MiB
above k=8 — the cost is time, not memory. That mattered, because K is exactly the axis you need to
sweep in order to find out whether K=16 was the right call. (It was not; see §7.5.)

**(c) The offline path can construct the ID/OOD contrast that the online path cannot.** All 51 demos
in `banana_in_pot_lerobot_v3` are teleoperated successes. The failure label is a **constant**. There
is no variance in the dependent variable, so no AUROC, correlation, or detection threshold for
"variance predicts failure" is computable — the question is not hard here, it is *undefined*. What
*is* computable is whether variance rises when the observation leaves the training distribution,
because a controlled perturbation manufactures a known ID/OOD pair with a dialable severity that
substitutes for the missing label. This is falsification-only by design: a flat severity response
would kill the mechanism outright, while a rising one is necessary but not sufficient for the
failure-prediction claim. That asymmetry is why the perturbation registry deliberately contains both
global photometric ops *and* a localized semantic distractor — the interesting question was never
"does variance rise" but "does it rise for the perturbations that would actually break the task."

### GPU discrepancy found (correcting the recorder doc)

The recorder doc's §6 benchmark table reports FAIL at every K on "RTX 3060 Laptop, 6GB". **That is a
different machine.** This build PC reports:

```
NVIDIA GeForce RTX 3060, 11889 MiB   (~12GB, desktop part)
```

The 6GB FAIL is a property of the robot PC (`/home/laptop3/...`) and is **not binding here**. Peak
usage during the K=64 sweep was **1244 MiB allocated / 1664 MiB reserved** — roughly 14% of this
card. VRAM was never remotely a constraint, and K was never reduced for memory reasons. Anyone
reading the recorder doc's FAIL table should not carry it over to offline work on this machine.

## 3. Why the three fixes were made — and what the data said about each

This is the part worth reading, because **the data agreed with two of the three rationales and
substantially undercut the third.** Documenting that honestly is the point.

### 3.1 Horizon slicing — `v(h)` profile, `v_exec` vs `v_tail`

**The rationale.** The policy predicts `horizon=64` steps but executes only `n_action_steps=32` of
them (exec slice `[1:33)`, verified against `modeling_diffusion.py:316-318` rather than assumed). The
far-future half is **never executed** and is naturally wider, because the far future is
under-determined by the current observation regardless of how confident the policy is about the
present. Collapsing one scalar over all 64 steps therefore mixes two phenomena: structural fan-out
with horizon distance (a property of the *task*) and situational uncertainty (the thing we want).
Keeping `v(h)` as a vector lets a caller normalize per-`h`, restrict to the executed prefix, or
compare the same `h` across refills — none of which a scalar permits.

**What the data said: directionally correct, quantitatively minor, and the strong version of the
claim is not supported.**

Fan-out is real but weak. Over the entire 64-step horizon, mean `v_all` rises only **1.80x**
min-to-max and `v_arm` **3.10x**. Spearman(h, mean v) = 0.878 across the ensemble average, but it is
**not monotonic** (only 41 of 63 consecutive steps increase; `v_arm` actually *decreases* from h=0 to
h=4), and it barely holds per-refill: median per-refill Spearman is **0.120**, with only 57.4% of
refills positive. So `v(h)` growth is an ensemble-average property; on any individual refill the
profile is close to flat noise. Anyone planning to normalize a live refill against a reference `v(h)`
curve should know that.

The exec/tail bias is real and small:

| dims | `v_exec` | `v_tail` | tail/exec |
|---|---|---|---|
| all | 0.004787 | 0.005590 | **1.168** |
| arm | 0.000138 | 0.000236 | **1.704** |
| grip | 0.032679 | 0.037717 | 1.154 |

Collapsing over all 64 steps instead of 32 inflates the number by ~17% (all dims) / ~70% (arm) — the
predicted direction, nowhere near an order of magnitude. And on 41% of refills the tail is *less*
variable than the executed part.

**The decisive test is whether slicing improves a detector, and it does not:**

| condition | `v_exec` AUC | `v_full` AUC | `v_tail` AUC | exec − full |
|---|---|---|---|---|
| gaussian_noise sev1 | 1.000 | 1.000 | 1.000 | 0.000 |
| blur sev1 | 1.000 | 0.998 | 0.993 | +0.002 |
| brightness_down sev1 | 1.000 | 1.000 | 0.999 | 0.000 |
| occlusion sev1 | 0.821 | 0.793 | 0.781 | +0.028 |
| shift_x sev1 | 0.707 | 0.706 | 0.681 | +0.001 |
| color_shift_region sev1 | 0.516 | 0.546 | 0.530 | **−0.030** |
| state_offset_random sev1 | 0.538 | 0.533 | 0.540 | +0.005 |

**Median benefit +0.002 AUC, max +0.028, and negative on one condition.** Even `v_tail` alone
separates almost as well. The exec/tail value bias is a near-constant multiplicative offset that
cancels out of any rank statistic.

> **Honest verdict:** `variance_profile`'s docstring claim that structural fan-out *dominates* a
> collapsed scalar is **not supported by this checkpoint's data**. Keep `executed_only=True` as free
> hygiene — measuring variance in a slice with no causal path to the robot's behaviour is still
> conceptually wrong — but stop describing it as fixing a large measured problem. It fixes a small
> one.

There is one unexplained artifact worth recording: median `v_all` roughly **triples between h=48 and
h=63**, a jump larger than all growth over h=0..48 combined, concentrated in the last few indices. It
sits entirely in the never-executed tail. It looks like a U-Net horizon-boundary artifact, but that
was not determined. It should not be read as evidence of genuine predictive fan-out, and it inflates
any all-64-step scalar.

### 3.2 Logger provenance + embedded normalization stats

**The rationale.** Every number in `trajectories`, `committed_chunk`, and `obs_state` lives in
lerobot's **normalized** space — the recorder doc says so in §4 and then leaves the reader to go find
the checkpoint. A checkpoint hash is an *identity*, not a *decoder*: it lets a future analyst confirm
"this was the same artifact" but not answer "what joint angle is 0.83?". If the checkpoint is
deleted, retrained over, or simply lives on a different machine than the analysis, a hash alone
leaves the HDF5 **permanently undecodable** — you would know exactly which artifact you needed and
still not have it.

So the fix embeds both. `/ensemble_trajectories` gains provenance attrs (checkpoint path, hash,
lerobot/torch versions, `num_inference_steps`, `noise_scheduler_type`), and a `norm_stats` subgroup
holds the raw stat arrays per feature plus a JSON blob giving feature types, the `FeatureType ->
NormalizationMode` map, and the **already-resolved** `mode_by_key`. That last part is what makes the
arrays usable rather than merely present: `MEAN_STD`, `MIN_MAX` and `QUANTILES` imply different
inverse transforms, and lerobot resolves the mode per `FeatureType`, not per key. The file now states
which formula applies to which feature instead of making the reader re-derive lerobot's resolution
rule.

The cost asymmetry is extreme — a handful of 7-element vectors and a few `(3,1,1)` image arrays,
written once per process, against files holding `(N, K, 64, 7)` float32 ensembles. And embedding
removes a *dependency*, not just a file: unnormalizing from a checkpoint means importing lerobot and
rebuilding the processor pipeline, dragging in a specific lerobot version plus torch plus CUDA. With
the arrays in the file, an analysis script needs only `h5py` and `numpy`.

**What the data said: strongly vindicated, and the specific failure it guards against actually
occurred during this very study.**

Three separate version-mismatch incidents hit this project in one session:

- The checkpoint was trained with **lerobot 0.6.1**. `/home/theo_lab/lerobot` is **0.5.2**, one minor
  version *behind* (inverting the risk noted in the original brief). Loading unmodified raises
  `draccus.utils.DecodingError: The fields 'pretrained_revision' are not valid for DiffusionConfig`.
- **lerobot 0.6.1 does not exist.** PyPI's latest is 0.6.0 and `git ls-remote --tags` shows only
  `v0.6.0`. The pinned "0.6.1" in `requirements-diffusion.lock` was a post-0.6.0 main-branch dev
  build from someone's environment — not a reproducible release. The new `act_venv` was built on
  0.6.0.
- The two venvs on this machine disagree: `act_venv` has lerobot 0.6.0, `/home/theo_lab/lerobot/.venv`
  has 0.5.2. Two agents in this study reported different lerobot versions for the same task and both
  were correct.

Any of those would have blocked a hash-only-provenance reader from decoding an ensemble file. The
stats-in-file design sidesteps all three.

Also relevant to provenance, and discovered the hard way: **the stats are not where the old code
expected them.** They are not inside `model.safetensors` (all 398 keys checked — zero normalization
keys), there is no `dataset_stats.pth`, and `hasattr(policy, "normalize_inputs")` is `False`. This
checkpoint uses the post-refactor processor pipeline, so the classic
`policy.normalize_inputs.buffer_observation_state.mean` access path **does not exist**. The stats live
in standalone safetensors files described by JSON manifests, readable with flat dotted keys:

```python
from safetensors.torch import load_file
st = load_file(f"{CKPT}/policy_preprocessor_step_3_normalizer_processor.safetensors")
a_min, a_max = st["action.min"], st["action.max"]            # (7,) float32
s_min, s_max = st["observation.state.min"], st["observation.state.max"]
```

STATE and ACTION both use `MIN_MAX`, so min/max are the decode-critical arrays. Recorded values are
`float32` in normalized space; `action.min[6] = 0.0` / `action.max[6] = 0.9998` is the gripper.

**Caveat, reported honestly:** `collect_normalization()` was exercised against a hand-built
duck-typed stand-in, not a real loaded preprocessor, because `checkpoints/` did not exist at the time
that agent ran. Field names and str-Enum access were confirmed against installed lerobot source, but
the end-to-end extraction path from a real `make_pre_post_processors` output is **untested**. The
later offline runs did write real `norm_stats` groups successfully, which is indirect evidence it
works, but it was never unit-tested against a real processor.

The hash was deliberately kept cheap — sha256 over `config.json` bytes plus `(relpath, size, mtime)`
per safetensors, not a content hash of hundreds of MB — because it runs during server startup on the
robot PC and the strong guarantee is not worth seconds of disk I/O when the decode-critical
information is embedded anyway. Both writers swallow their own exceptions, so malformed stats degrade
to a missing attr rather than preventing the server from coming up. Both params default to `None`, so
the file layout is byte-identical for any caller that passes nothing — verified by constructing with
no new args and confirming the attr/member lists are unchanged.

### 3.3 Arm/gripper separation

**The rationale.** Dataset dim 6 is a near-binary open/close step: **95.44%** of all 21,524 frames
sit within 0.05 of a rail (68.36% near 0.0, 27.08% near 0.9998), with only 4.56% in between forming a
thin, flat bridge — the signature of fast linear ramps, not a continuous control distribution.
Transitions take a median of **3 frames** (~0.1s at 30fps), and an average episode spends ~1% of its
length in transit.

Consequently, two samples that agree perfectly on *what* to do but disagree by a few timesteps on
*when* to close produce a variance spike of order `(open−closed)²/4`. That encodes **task phase**
(proximity to a grasp) rather than model uncertainty. `ensemble_analysis.py` demonstrates the failure
mode constructively: a synthetic ensemble with **identical** arm trajectories across all samples
(true arm spread max 6.9e-34) but ±3-timestep gripper jitter yields a pooled 7-dim variance of
**0.038** — uncertainty manufactured out of nothing by one dimension.

Note also an asymmetry that matters: `observation.state[6]` (measured 2F-85 position) is **not**
bimodal (std 0.240, only 0.59% near max — the physical finger caps around 0.90 and commonly rests
near 0.55 while gripping). Commanded and measured gripper do not share a distribution or a
normalization.

**What the data said: strongly vindicated, and understated if anything.**

Executed-slice variance by dimension, mean over 739 ID refills:

| dim | exec variance | % of pooled |
|---|---|---|
| cmd1 | 0.000134 | 0.40 |
| cmd2 | 0.000132 | 0.39 |
| cmd3 | 0.000167 | 0.50 |
| cmd4 | 0.000122 | 0.36 |
| cmd5 | 0.000216 | 0.64 |
| cmd6 | 0.000059 | 0.18 |
| **grip** | **0.032679** | **97.52** |

The gripper is **236x** the arm mean and **97.5% of pooled executed variance**. Among the six arm
joints there is no dominator at all — cmd5 to cmd6 is only a 3.7x spread. (Pleasingly, cmd6 has the
*lowest* ensemble variance despite the *widest* data range at 3.08 rad; normalization is doing its
job.) Any unweighted pooled metric on this policy is ~97.5% a gripper-timing measurement, and the arm
signal is invisible unless the dims are split.

The detector-level cost is the number that settles it:

| condition | arm 0–5 AUC | all 7 dims AUC | delta |
|---|---|---|---|
| gaussian_noise sev1 | 1.000 | 0.954 | −0.046 |
| blur sev1 | 1.000 | 0.803 | **−0.197** |
| brightness_down sev1 | 1.000 | 0.833 | **−0.167** |
| occlusion sev1 | 0.821 | 0.645 | **−0.176** |
| shift_x sev1 | 0.707 | 0.661 | −0.046 |
| color_shift_region sev1 | 0.516 | 0.575 | +0.059 |
| state_offset_random sev1 | 0.538 | 0.551 | +0.013 |
| **control_b (null)** | 0.519 | 0.546 | +0.027 |

**Including the gripper destroys 0.17–0.20 AUC on exactly the conditions the signal can detect.** The
apparent small gains on the semantic conditions are worthless: the null also rises by +0.027, so
that is noise-floor drift, not signal. Mechanism confirmed — pooled all-dim ID `v_exec` has CV
**2.52** vs arm-only **1.36**; pooling nearly doubles the noise floor's relative width.

**One genuine nuance that partially qualifies the rationale, and should be recorded.** The gripper is
*not purely* a phase artifact. Gripper variance does spike near grasp transitions (Spearman(distance
to transition, `v_grip`) = **−0.660**, p=1.2e-93; 2.6–3.8x higher near transitions) — but **arm
variance correlates too** (rho = **−0.230**, p=2.6e-10; 1.9–2.1x higher near transitions). Grasp
proximity is a genuine, arm-visible uncertainty driver, not solely a step-function shape effect. What
kills pooling is not that dim 6 is meaningless; it is that its amplification factor (2.6–3.8x) is
*comparable* to the arm's (1.9–2.1x) while its absolute scale is 236x larger. The 236x gap is shape,
not information.

And under perturbation the gripper is actively harmful: `v_grip_exec` **decreases** for 13 of 21
perturbed conditions and rises only for gaussian_noise, and then only 1.2–1.65x while the arm moves
250x. It would have diluted the signal, not merely distorted it.

> **Honest verdict:** split arm and gripper. The split is structural in `ensemble_analysis.py` rather
> than a default flag, specifically so the analysis cannot silently do the wrong thing; `combined=True`
> still exposes the pooled profile so pooled-vs-split stays directly comparable.

## 4. Fix scorecard

| Fix | Rationale | What the data said |
|---|---|---|
| Horizon slicing | Structural fan-out dominates a 64-step scalar; must keep `v(h)` and restrict to exec | **Partly contradicted.** Fan-out is only 1.8x/3.1x, non-monotonic, near-absent per-refill. Exec/tail bias 1.17x/1.70x is real but cancels out of rank statistics: median AUC gain **+0.002**. Keep as hygiene, drop the strong claim. |
| Provenance + embedded stats | A hash is an identity, not a decoder; normalized values are undecodable without the checkpoint | **Vindicated.** Three separate lerobot version incidents in this session (0.5.2 vs 0.6.0 vs a nonexistent 0.6.1), plus the stats not being where the classic access path expects. Any of them would have blocked a hash-only reader. |
| Arm/gripper separation | Dim 6 is 95.4% rail-saturated; timing jitter manufactures variance encoding task phase, not uncertainty | **Vindicated, understated.** Gripper is 97.5% of pooled variance; pooling costs **0.17–0.20 AUC** on every detectable condition. Nuance: grasp proximity also raises arm variance (rho −0.230), so dim 6 is an amplified real signal, not pure artifact — but pooling still loses. |

## How to run the code

Everything below runs from `ros2_ur_ws/` with the study venv (`act_venv`, see §10). No
robot, no ROS, GPU-only. The checkpoint ships gitignored under the package; the dataset
re-downloads from the hub on first use.

**One-time: fetch the checkpoint (~1.1 GB, gitignored).**
```bash
cd ros2_ur_ws
ACT_VENV=act_venv src/gello_policy/scripts/download_diffusion_checkpoint.sh
# -> src/gello_policy/checkpoints/diffusion_banana_in_pot_joint
# NOTE: under huggingface_hub 1.x the script's huggingface-cli probe can exit 0 without
# downloading (see §10). If checkpoints/ ends up empty, pull directly instead:
#   act_venv/bin/hf download Bigenlight/diffusion_banana_in_pot_joint \
#       --local-dir src/gello_policy/checkpoints/diffusion_banana_in_pot_joint
```

**The four scripts and what each is for:**

| Script | Role | Entry |
|---|---|---|
| `scripts/offline_ensemble.py` | Sweep K samples/refill over a LeRobotDataset → one `ensemble_*.h5` (same schema as the online logger) | CLI, `--help` |
| `scripts/ood_sweep.py` | Paired ID-vs-OOD contrast: same observations under each perturbation × severity → one `ensemble_*.h5` per condition + `manifest.json` | CLI, `--help` |
| `scripts/ood_perturbations.py` | Perturbation registry imported by `ood_sweep`; `__main__` renders a before/after contact sheet | library / self-check |
| `policy_server/ensemble_analysis.py` | Dependency-light (numpy+h5py+scipy) reader + all metrics. Import it, or run on a file for a text report | `--self-check` \| `<file.h5>` |

**(a) In-distribution baseline** — the §5.1 `id_k64.h5`:
```bash
act_venv/bin/python src/gello_policy/scripts/offline_ensemble.py \
    --checkpoint src/gello_policy/checkpoints/diffusion_banana_in_pot_joint \
    --dataset Bigenlight/banana_in_pot_lerobot_v3 \
    --k 64 --episodes 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50 \
    --stride 16 --seed 0 \
    --out /path/to/offline_runs   # writes ensemble_<timestamp>.h5 inside
```
Defaults mirror the deploy sampler (DDIM/10, `n_action_steps=32`); `--k 8` is enough in
practice (§5.5). `--out` is a directory; the file is auto-named like the online logger.

**(b) OOD contrast sweep** — the §5.1/§5.4 `ood/` corpus. Probe timing first, then run:
```bash
# tiny timing-only run (1 episode, 1 perturbation, 1 severity):
act_venv/bin/python src/gello_policy/scripts/ood_sweep.py \
    --checkpoint src/gello_policy/checkpoints/diffusion_banana_in_pot_joint \
    --dataset Bigenlight/banana_in_pot_lerobot_v3 \
    --out /path/to/offline_runs/ood --probe

# full sweep (7 perturbations × 3 severities + 2 controls, ~12 min on the 3060):
act_venv/bin/python src/gello_policy/scripts/ood_sweep.py \
    --checkpoint src/gello_policy/checkpoints/diffusion_banana_in_pot_joint \
    --dataset Bigenlight/banana_in_pot_lerobot_v3 \
    --k 64 --out /path/to/offline_runs/ood
```
`--dataset` also accepts a local LeRobotDataset root (then pass `--repo-id`). Eyeball the
perturbations first with `python src/gello_policy/scripts/ood_perturbations.py` (writes a
contact sheet; needs `OOD_SCRATCH` pointing at a dir with the dataset, and matplotlib).

**(c) Read / analyze a file** — one-liner report, or the library:
```bash
# human-readable report on any ensemble file (online OR offline):
act_venv/bin/python src/gello_policy/policy_server/ensemble_analysis.py /path/to/ensemble_XXXX.h5
# self-test the metrics (synthetic 2-mode vs 1-blob, executed-slice indexing, ...):
act_venv/bin/python src/gello_policy/policy_server/ensemble_analysis.py --self-check
```
```python
# programmatic: the analysis surface used throughout §5
import sys; sys.path.insert(0, "src/gello_policy/policy_server")
import ensemble_analysis as EA
ef = EA.load("ensemble_XXXX.h5")            # tolerates pre-provenance files
prof = EA.variance_profiles(ef.trajectories, ef.exec_slice)  # v(h), exec/tail, arm/grip
s = EA.refill_scores(ef, dims=EA.ARM_DIMS, executed_only=True, statistic="mean_var")
band = EA.conformal_calibrate(cal_scores, alpha=0.05)        # success-only calibration
# norm_stats are embedded (§3.2): ef.norm_stats["action"]["min"/"max"] to unnormalize.
```

> **ANALYSIS.md, plots, and the exact per-condition h5 files from this study live in the
> session scratchpad (`.../scratchpad/offline_runs/`), which is ephemeral.** The runnable
> scripts above reproduce them; the analysis wrappers used for the figures
> (`analyze_id.py`, `synth.py`, `stac_ood.py`) were session-local and are not committed —
> `ood_sweep.py` + `ensemble_analysis.py` are the reusable, committed surface.

## 5. What was actually run

Two corpora, both K=64, both on the **deploy sampler** (DDIM, 10 inference steps, `n_action_steps=32`)
rather than the checkpoint's own DDPM/100. That choice characterizes the policy as the robot would
actually run it, not as it was trained; a DDPM/100 baseline was never run and is a known gap.

**In-distribution baseline.** 26 episodes (even indices 0–50), stride 16, seed 0 → **739 refills**,
0 dropped, 861s wall (1.17 s/sample). Peak GPU 1244 MiB. Per-sample: obs build + warmup 174ms,
refill 171ms, K=64 ensemble 818ms.

**OOD sweep.** 7 perturbations x 3 severities + 2 controls = **23 conditions**, 30 paired
observations each (6 episodes x 5 within-episode phase fractions), 12.5 min total. Perturbations are
injected *after* the production image pipeline's resize/BGR→RGB/[0,1] and *before* the checkpoint's
own normalizer — the exact tensor the policy sees. `control_b` is identical observations with a
different ensemble sampling seed: a proper null.

Two correctness checks worth recording, both unprompted:

- **The K-sample mean over the executed slice matches the policy's own committed chunk to
  maxabs ≤ 0.017 normalized at every K** — independent evidence that the offline harness conditions
  the ensemble identically to the real deploy path.
- A **real bug was found and fixed** in the harness: the saved lerobot processor injects an `action`
  key (None at inference) into every transition, so `populate_queues` filled the action deque with 32
  `None`s, the sampled frame never refilled, no `global_cond` was produced, and `select_action`
  returned `None`. Fixed by replicating `select_action:146-147`'s `batch.pop(ACTION)`. Missing
  `global_cond` is now a hard `RuntimeError`, not a silent log.

### 5.1 Does variance separate ID from OOD?

Score = `v_exec` over arm dims 0–5. AUC against the broad 739-refill ID corpus:

| condition | mean `v_exec` | ratio | AUC | TPR@5%FPR | TPR@1%FPR |
|---|---|---|---|---|---|
| gaussian_noise sev1 | 2.669e-02 | 196.9x | **1.000** | **1.000** | **1.000** |
| brightness_down sev1 | 7.936e-03 | 58.6x | **1.000** | **1.000** | **1.000** |
| blur sev1 | 3.082e-03 | 22.7x | **1.000** | **1.000** | **1.000** |
| occlusion sev1 | 3.728e-04 | 2.8x | 0.821 | 0.167 | 0.067 |
| shift_x sev1 | 1.913e-04 | 1.4x | 0.707 | 0.033 | 0.000 |
| state_offset_random sev1 | 1.386e-04 | 1.02x | 0.538 | 0.033 | 0.000 |
| color_shift_region sev1 | 1.408e-04 | 1.04x | 0.516 | 0.033 | 0.000 |
| **control_b (NULL)** | 1.440e-04 | 1.06x | **0.519** | 0.067 | 0.000 |
| control | 1.355e-04 | 1.00x | — | 0.033 | 0.000 |

**The AUC-versus-deployability gap is the most important structural result here.** Occlusion's AUC of
0.821 sounds usable and catches **16.7%** at 5% FPR and **6.7%** at 1%. shift_x at 0.707 catches
**3.3%** — identical to the control's own false-alarm rate, i.e. nothing. The cause is that ID
`v_exec` itself has CV **1.364**, spanning 1.30e-05 to 1.97e-03, a **151x range** driven purely by
which frame you happen to be on. A 1.4–2.8x mean shift disappears inside that. AUC rewards a small
consistent shift; a deployable detector has to clear the ID *tail*. Both things can be true, and here
they are: those effects are statistically real in the paired sense and useless in the deployable one.

Severity monotonicity is also weak — only 3 of 7 perturbations are monotone, and two of those
(brightness_down, blur) are sharply convex (nothing happens until a threshold, then variance
explodes). **gaussian_noise is non-monotone: sev1.0 gives *lower* variance than sev0.5.** That is
unexplained. A plausible mechanism is clamp-induced saturation toward a uniform grey field, but it
was not tested and is not being asserted.

### 5.2 The signal saturates and decorrelates where it is strongest

At high severity, Spearman(control, perturbed) collapses from +0.98 at mild severity to **−0.12**
(gaussian sev1), **−0.04** (blur sev1), **−0.07** (brightness sev1), while the coefficient of
variation drops from 1.05 (control) to **0.099**. Under severe corruption the policy does not become
"more uncertain in proportion to difficulty" — it falls into a uniformly high-variance regime
carrying no per-observation information. **Even where the signal works, it works as a binary
tripwire, not a graded confidence score.** Any downstream design assuming `v_exec` is a continuous
confidence measure in the high-variance regime is wrong.

### 5.3 Metric choice is a non-issue

Mean AUC by OOD family across nine metrics: the top six (`pairwise_l2_exec`, `v_exec`,
`endpoint_full`, `v_full`, `endpoint_exec`, `v_exec_max`) tie within **0.03 AUC** — unsurprising,
since they are all functionals of the same second moment. `pca_top_eig_exec` is the clear loser and
is the only metric whose null (0.551) exceeds its semantic signal (0.541). `pca_partic_ratio_exec`
runs *below* 0.5 on both the null (0.388) and semantic OOD (0.443) — a genuinely inverted signal,
unusable one-sided.

Recommend `v_exec(arm)` — not because it won, but because it tied while being cheapest, is already
the `refill_scores` default, and has documented units and failure modes.

### 5.4 Conformal calibration — the closest thing to a real detector

Score = `refill_scores(dims=ARM, executed_only=True, statistic="mean_var")`. 13 calibration episodes,
13 held-out ID episodes, alpha=0.10, time-varying band by normalized phase.

| condition | alarm rate |
|---|---|
| gaussian_noise sev0.25 / 0.5 / 1.0 | **1.000 / 1.000 / 1.000** |
| blur sev1.0 · brightness_down sev1.0 | **1.000 / 1.000** |
| blur sev0.5 | 0.767 |
| brightness_down sev0.5 · occlusion sev0.5 | 0.167 / 0.167 |
| **occlusion sev1.0** | **0.133** |
| shift_x sev0.25 · **control_b (null)** | 0.133 / **0.133** |
| blur sev0.25 | 0.100 |
| **control (null)** · color_shift sev1.0 · shift_x sev1.0 | **0.067** / 0.067 / 0.067 |
| state_offset_random sev0.25 / 0.5 / 1.0 | 0.033 / 0.033 / **0.033** |

This is the cleanest single statement of the whole study. A calibrated, label-free detector fires on
**100%** of photometric corruption *including the mildest gaussian noise at severity 0.25*; fires on
**13.3%** of *severe* occlusion, **exactly the 13.3% it fires on `control_b`** — severe occlusion is
indistinguishable from resampling the same clean observation with a new seed; and fires on **6.7%**
of severe target-recolouring and **3.3%** of severe joint displacement, both **at or below the
control's own 6.7% false-alarm rate**. At alpha=0.05 the picture is unchanged.

**Calibration caveat:** the time-varying band over-covers (held-out ID exceedance 0.046 against a
0.10 target), because at n=13 the conformal rank is `ceil(14x0.9)=13` — it takes the *maximum* across
episodes per bin, making it a max-envelope rather than a 90% quantile. Honest at n=13, not fixable by
analysis; it needs more episodes. The global threshold is marginally well-calibrated (0.112) but
blind in 6 of 10 phase bins, which is why the band exists at all.

### 5.5 K stability — K=16 was ~4x too conservative

CV of `v_exec` falls as ~K^(−1/2) with **no knee** — nothing "stabilizes" at any K, so the
decision-relevant curve is AUC-versus-K:

| condition | K=4 | K=8 | K=16 | K=32 | K=64 |
|---|---|---|---|---|---|
| gaussian_noise sev1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| blur sev1 | 0.9996 ± 0.0009 | 1.0000 ± 0.0002 | 1.0000 | 1.0000 | 1.0000 |
| occlusion sev1 | 0.7958 ± 0.0338 | 0.7926 ± 0.0269 | 0.7908 ± 0.0196 | 0.7934 ± 0.0107 | 0.7867 |

**K=4 suffices for every effect this signal can detect** (blur's worst case over 40 subset draws is
0.9967). Occlusion's AUC is *flat* from K=4 to K=64, because the binding constraint is ID width, not
estimator noise — more samples buy score precision and zero separability. Mean `v_exec` is unbiased
in K (1.342e-04 at K=4 vs 1.355e-04 at K=64), confirming `ddof=1` is correct.

Use K=8 for margin. K=16 only if you want score *values* stable to ±19% for plotting.

### 5.6 Unprompted supplementary: STAC temporal MMD sees what variance cannot

This is the one positive result on the OOD family that matters most operationally, and it deserves a
follow-up. Only **12 usable pairs** (the OOD sweep sampled observations 50–88 frames apart, so half
the adjacencies exceed `horizon=64` and have no overlap at all). Gamma fixed from control. Paired
Wilcoxon:

| condition | mean MMD² | ratio | Wilcoxon p | paired d |
|---|---|---|---|---|
| control_b (NULL) | 0.4506 | 1.006 | 0.910 | +0.04 |
| state_offset_random sev0.25 | 0.6201 | 1.384 | 0.034 | +0.56 |
| **state_offset_random sev0.5** | 0.7522 | **1.679** | **0.0024** | **+1.12** |
| **state_offset_random sev1.0** | 0.7740 | **1.728** | **0.0034** | **+1.03** |
| color_shift_region sev0.5 / 1.0 | 0.5554 / 0.5861 | 1.240 / 1.308 | 0.064 / 0.052 | +0.51 / +0.58 |

**`state_offset_random` — which every variance metric scored at exactly chance (AUC 0.516–0.552,
conformal alarm 0.033) — is detected at p=0.0024, d=1.12, with a monotone dose-response.**
`color_shift_region` trends the same way at marginal significance. The null sits at 1.006, p=0.91.

**Severe caveats, and one thing that must not be repeated as a finding:** n=12 pairs is suggestive,
not established; the lag is 50–54, not the deploy cadence of 32, overlapping only 10–14 timesteps;
and the photometric conditions' MMD→0 is a **kernel-bandwidth artifact, not a result** — with gamma
fixed from control, the inflated corrupted clouds drive every kernel evaluation to ~0. It "separates
perfectly" for a reason that would not survive adaptive bandwidth. **Do not report "STAC detects
gaussian noise."**

## 6. CORRECTION NOTICE — a fabricated literature claim, retracted

An earlier literature review in this project stated that **FAIL-Detect (arXiv:2503.08558)** evaluated
a **"Diffusion Output Variance"** baseline and reported specific TPR/TNR figures for it.

**This is false. Four independent readings of the actual paper found no such baseline and no such
numbers.** The claim, and any figure attributed to it, must not be reused. It is recorded here
specifically so nobody re-derives it from the same plausible-sounding premise.

What the paper actually contains, and how it differs:

- The nearest method in FAIL-Detect's comparison set is **STAC (Agia et al., CoRL 2024)**, which
  measures **MMD between the temporally overlapping regions of two *consecutive* predictions**. That
  is a fundamentally different quantity from across-sample variance: STAC compares predictions made
  at two *different* timesteps from two *different* observations, whereas this study's `v_exec`
  measures spread across K samples drawn from **one fixed observation**. Temporal consistency and
  sampling dispersion are not the same signal, and §5.6 above shows empirically that they detect
  different things on this checkpoint.
- **STAC was omitted entirely from that paper's real-robot experiments**, because it costs 256 samples
  per timestep. Its negative result there is therefore **simulation-only** and should not be cited as
  evidence about real-robot performance.

The practical consequence for this project: there is **no published baseline number** that this
study's results can be compared against, and none should be invented. Everything in §5 stands on its
own measurements or not at all.

## 7. Verdict

**Across-sample variance is worth keeping only as a cheap camera-integrity tripwire. It is not worth
pursuing as a general uncertainty or failure signal, and the evidence for that negative is strong.**

The supported positive claim, stated precisely: *`v_exec` over arm dims with a conformal band detects
100% of global photometric image corruption at ~5–10% false-alarm rate, at K=4, with no failure
labels required.* That is a real capability — dirty lens, exposure fault, defocus, dying sensor — and
nearly free. It should be shipped at that scope and described at that scope, if at all.

The supported negatives are the more valuable output:

1. **It is a photometric-corruption detector, not an uncertainty estimate.** Recolouring the
   task-named target object (AUC 0.516, conformal alarm 0.067) and displacing joints (0.538, 0.033)
   both sit at the sampling-noise floor (`control_b`: 0.519, 0.133). The failure modes you would most
   want to catch on a robot are invisible to it; a dirty lens lights it up 200x. Two independent
   methods agree on this.
2. **High AUC does not survive contact with an operating point.** Occlusion 0.821 → 16.7% @ 5% FPR →
   13.3% under conformal, identical to the null. ID's own 151x frame-to-frame range is the binding
   constraint and no amount of K fixes it.
3. **Metric choice is a non-issue** — the top six tie within 0.03 AUC.
4. **Of the two design worries carried into this study, the data supports one and largely dismisses
   the other:** gripper separation was understated (−0.17 to −0.20 AUC if ignored), horizon slicing
   was overstated (+0.002 AUC).
5. **K=16 was ~4x too conservative.** K=4 matches K=64 everywhere it matters.

**Concrete recommendation.** Keep `DIFFUSION_ENSEMBLE_K=0`. If a monitor is ever built: `v_exec` over
arm dims 0–5, executed slice, K=8, conformal at alpha=0.05 calibrated on **≥30** success episodes
(not 13). At K=8 the online cost is ~8x a refill's denoise — which, on the robot PC that already
fails the K=1 benchmark, is not obviously affordable at all. And the only thing it buys is
camera-fault detection, which **a cheaper image-statistics check on the raw frames would very likely
catch just as well and should be benchmarked against before any of this ships.**

## 8. Limitations

Stated plainly, because several of them materially bound the conclusions above.

- **No failure labels anywhere.** All 51 demos are teleoperated successes. This is input-OOD
  separability only. "Variance predicts failure" was never tested and remains untested.
- **Deploy sampler only (DDIM/10), never the trained DDPM/100.** Rankings, the arm/gripper result,
  and the K conclusions are likely robust to this; the absolute 20–200x photometric ratios are not
  verified under DDPM/100.
- **n=30 observations per condition, 6 of 51 episodes, single perturbation seed (0).** Within a
  condition every observation receives the *same* noise field / occluder position / offset direction.
  The large effects are safe; the marginal ones (occlusion, shift_x, d≈0.5) are not generalized over
  perturbation realizations.
- **`state_offset_random` sev1.0 displaces each joint by only ~0.8σ**, mostly staying inside the
  training MIN_MAX box. The correct conclusion is "offsets up to ~0.8σ/joint are undetectable by
  across-sample variance", **not** "state OOD is undetectable" — and §5.6 suggests another metric
  does see them.
- **`shift_x` and `occlusion` are lower bounds on a real camera move** — no parallax, no lighting
  change, no occlusion-geometry change, as `ood_perturbations.py`'s own docstring states. The weak
  shift_x result must not be read as "the policy is robust to camera movement".
- **13 calibration episodes is too few**; the conformal band demonstrably over-covers (0.046 vs 0.10).
- **The ID corpus mixes train and held-out episodes** (even indices 0–50 include eval episodes
  46/48/50). Episode index is retained in `id_summary.npz` if a strict train-only baseline is wanted;
  no re-run needed.
- **The dip test never produced a p-value** (`diptest` absent), so the multimodality question (§1,
  question 2) is genuinely unanswered rather than answered negatively.
- **The gaussian_noise non-monotonicity (sev1.0 < sev0.5) is unexplained.**
- **`collect_normalization()` was never exercised against a real loaded preprocessor**, only a
  duck-typed stand-in.
- **The `h=63` variance spike is unexplained** — real U-Net boundary effect or sampler/logger artifact
  was not determined.
- **Bit-exact parity between lerobot 0.5.2 and the training version was never verified.** The forward
  pass produces plausible in-range joint values, but silent numerical drift cannot be fully ruled out.

## 9. Next steps, in priority order

1. **The STAC temporal-MMD result on `state_offset_random`** (§5.6) — with adaptive bandwidth,
   deploy-cadence lag 32, and ≥100 pairs. This is the highest-value follow-up because it is the only
   positive result on the OOD family that operationally matters, and because it directly probes the
   signal that variance is blind to.
2. **A run with actual rollout failure labels.** Without this, none of the above connects to the
   question the project actually cares about. This requires robot time and a deliberate mix of
   successes and failures.
3. **Benchmark a plain image-statistics baseline** (per-frame mean/variance/gradient energy on raw
   frames) against `v_exec` on the same 23 conditions. If it matches the photometric detection at a
   fraction of the compute — which is likely — then even the surviving positive claim does not justify
   an ensemble.
4. **Re-run `state_offset` at `sigma_scale` 5–10** to establish where in the offset range, if
   anywhere, variance starts to respond.
5. **A DDPM/100 baseline** to check whether the deploy sampler's aggressive step reduction is itself
   suppressing informative variance structure.
6. **Fold `ood_sweep.py`'s paired-observation caching into `offline_ensemble.py`** if the two runners
   start to diverge. The OOD runner is now committed at `scripts/ood_sweep.py` (parameterized with
   `--checkpoint`/`--dataset`/`--out`, no hardcoded scratch paths) and imports `offline_ensemble`
   read-only, so the experiment is reproducible from the tree — this item is now a consolidation nicety,
   not a reproducibility gap.

## 10. Environment and reproduction notes

- **New venv:** `ros2_ur_ws/act_venv` (python 3.12.3, built with `uv`, 7.2GB). Not reused from
  `/home/theo_lab/lerobot/.venv`, which lacked `diffusers`, `h5py`, `scipy`, `sklearn`, and
  `matplotlib`, and whose mutation would have affected an unrelated source checkout.
- **`act_venv` (7.2GB) is ignored only incidentally.** No repo `.gitignore` rule covers it; it is
  excluded solely by the `act_venv/.gitignore` file containing `*` that `uv venv` auto-generates
  inside it (`git check-ignore -v` confirms `ros2_ur_ws/act_venv/.gitignore:1:*`). Delete or
  regenerate that directory without uv and 7.2GB becomes stageable. A real `act_venv/` entry in
  `ros2_ur_ws/.gitignore` would be worth adding. The checkpoint directory, by contrast, is properly
  ignored by `ros2_ur_ws/src/gello_policy/.gitignore:6:checkpoints/`.
- The lock file pins `lerobot==0.6.1`, **which does not exist**; 0.6.0 was installed. See §3.2.
- `pillow` resolved to 12.2.0 rather than the pinned 12.3.0 (torchvision constraint). Decode/resize
  only, low risk.
- `act_venv` was mutated during the harness build: `uv pip install "datasets>=4.7.0,<5.0.0"
  "av>=15.0.0,<16.0.0"` (lerobot 0.6.0's own pins). Note **`av` 18 is incompatible with lerobot**
  (`AttributeError: module 'av' has no attribute 'option'`). The resolver also downgraded `fsspec`
  2026.4.0 → 2026.2.0. `torch` untouched.
- Large `uv` installs need `UV_HTTP_TIMEOUT=600` (a `pypi.nvidia.com` timeout on
  `nvidia-cusparselt-cu12` killed the first attempt).
- **Bug in `scripts/download_diffusion_checkpoint.sh` (not fixed, not owned by this work):** its CLI
  probe picks `huggingface-cli` first, but under `huggingface_hub` 1.x that binary is a deprecation
  stub that prints a notice and **exits 0 without downloading**. `set -euo pipefail` cannot catch a
  zero exit, so the script reports success on an empty directory. Fix is to prefer `hf` over
  `huggingface-cli` in the candidate order.
- Running `ensemble_analysis.py` at scale needs `OMP/OPENBLAS/MKL_NUM_THREADS` pinned — unbounded BLAS
  threads during per-refill SVD caused a 12x CPU oversubscription and an 11-minute hang. Also,
  `mean_pairwise_l2` on the 739-refill file needs chunking to avoid a 4.6GB intermediate.
- No robot, ROS2 node, or hardware was run at any point. No real-time control-path file was modified.
  No git operations were performed.
