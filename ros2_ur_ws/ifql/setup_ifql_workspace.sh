#!/usr/bin/env bash
# Reproduce the IFQL serving workspace (~/carrot_ifql) on a fresh PC.
#
#   ./ros2_ur_ws/ifql/setup_ifql_workspace.sh            # -> $HOME/carrot_ifql (carrot task)
#   IFQL_ROOT=/data/carrot_ifql ./ros2_ur_ws/ifql/setup_ifql_workspace.sh
#   IFQL_TASK=orange ./ros2_ur_ws/ifql/setup_ifql_workspace.sh   # -> $IFQL_ROOT/hf_orange/
#
# What it builds (the layout run_ur7e_ifql_real.sh expects, see
# docs/ros2/GELLO_UR7E_IFQL_DEPLOY.md §2 and GELLO_UR7E_IFQL_ORANGE_HANDOFF.md):
#   carrot (IFQL_TASK=carrot, default):
#     $IFQL_ROOT/hf/            HF snapshot: recommended run + lead0 control + cards/code/hf_release
#   orange (IFQL_TASK=orange):
#     $IFQL_ROOT/hf_orange/     HF snapshot: 3 candidate runs + deploy/ (yaml, real_limits.json,
#                                norm_stats) + cards/code/hf_release — see run_ur7e_ifql_real.sh's
#                                IFQL_TASK=orange profile for how the 3 candidates are selected
#                                (IFQL_RUN_DIR alone).
#   $IFQL_ROOT/code/<snap>/   the code tarball, sha256-verified, D_c-null patch applied
#   $IFQL_ROOT/.venv-svf/     uv venv, Python 3.11, CUDA wheels (requirements-svf-infer.txt)
#   $IFQL_ROOT/.cache/torch/  TORCH_HOME with ResNet18 IMAGENET1K_V1 + dinov2 hub cache
#   $IFQL_ROOT/eval_runs/     server --log-dir root
#
# Needs: `uv` on PATH, `hf auth login` done with an account that can read the private
# Bigenlight/* repos, internet (HF + PyPI + torch.hub for dinov2). ~8 GB disk.
# Idempotent: re-running skips what is already there. Nothing here touches ROS or the robot.
set -euo pipefail

IFQL_ROOT="${IFQL_ROOT:-$HOME/carrot_ifql}"
IFQL_TASK="${IFQL_TASK:-carrot}"
STEP="${IFQL_STEP:-100000}"
PY_VER="${IFQL_PY_VER:-3.11}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Task profile: mirrors run_ur7e_ifql_real.sh's IFQL_TASK switch (carrot|orange). An
# unknown IFQL_TASK is refused rather than silently falling back to carrot.
case "${IFQL_TASK}" in
    carrot)
        HF_REPO="${HF_REPO:-Bigenlight/carrot-in-pot-ifql}"
        _HF_SUBDIR="hf"
        RUN="${IFQL_RUN:-ifql_real_lead6_k0.9_s0}"
        RUN_LEAD0="${IFQL_RUN_LEAD0:-ifql_real_k0.9_s0}"
        _EXTRA_RUNS=()
        _WANT_DEPLOY_DIR=""
        ;;
    orange)
        HF_REPO="${HF_REPO:-Bigenlight/orange-bowl-in-purple-bowl-ifql}"
        _HF_SUBDIR="hf_orange"
        # 3 candidates (docs/ros2/GELLO_UR7E_IFQL_ORANGE_HANDOFF.md §1); RUN is the
        # aug8 candidate (val actor loss 0.151, best of 19 — try this one first).
        RUN="${IFQL_RUN:-ifql_orange_k0.9_lead2_aug8_p0.5_s0}"
        RUN_LEAD0="${IFQL_RUN_LEAD0:-ifql_orange_k0.9_lead2_s0}"
        _EXTRA_RUNS=("px_orange_impala_lead2_s0")
        # deploy/ carries ifql_deploy_orange.yaml provenance (orange_real_limits.json)
        # and every norm_stats variant the 3 candidates need.
        _WANT_DEPLOY_DIR="deploy/*"
        ;;
    *)
        echo "ERROR: IFQL_TASK must be carrot|orange (got '${IFQL_TASK}')." >&2
        exit 1 ;;
esac

echo "### IFQL workspace -> ${IFQL_ROOT} (task=${IFQL_TASK}, hf subdir=${_HF_SUBDIR})"
command -v uv >/dev/null || { echo "ERROR: uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2; exit 1; }
mkdir -p "${IFQL_ROOT}"/{"${_HF_SUBDIR}",code,eval_runs,.cache/torch}

# --- 1) HF snapshot (selective: the server reads params_<step>.pkl, NOT .infer.pkl) ------
echo "### [1/5] snapshot_download ${HF_REPO}"
python3 - "$IFQL_ROOT/${_HF_SUBDIR}" "$HF_REPO" "$STEP" "$_WANT_DEPLOY_DIR" "$RUN" "$RUN_LEAD0" "${_EXTRA_RUNS[@]}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
root, repo, step, deploy_pat, *runs = sys.argv[1:]
pats = ["cards/*", "code/*", "hf_release/*", "models.json", "README.md"]
if deploy_pat:
    pats.append(deploy_pat)
for r in runs:
    pats += [f"{r}/flags.json", f"{r}/norm_stats_*.json", f"{r}/serve_meta.json",
             f"{r}/train.csv", f"{r}/RUN.md", f"{r}/params_{step}.pkl"]
snapshot_download(repo_id=repo, repo_type="model", local_dir=root, allow_patterns=pats)
print("snapshot ok")
EOF
[ -f "${IFQL_ROOT}/${_HF_SUBDIR}/${RUN}/params_${STEP}.pkl" ] || { echo "ERROR: params_${STEP}.pkl missing for ${RUN}" >&2; exit 1; }

# --- 2) code tarball: sha256 verify, extract, apply the D_c-null patch -----------------
echo "### [2/5] code tarball"
TARBALL="$(ls "${IFQL_ROOT}/${_HF_SUBDIR}"/code/*.tar.gz | head -1)"
SNAP="$(basename "${TARBALL}" .tar.gz)"
if [ -f "${TARBALL}.sha256" ]; then
    exp="$(awk '{print $1}' "${TARBALL}.sha256")"; got="$(sha256sum "${TARBALL}" | awk '{print $1}')"
    [ "${exp}" = "${got}" ] || { echo "ERROR: tarball sha256 mismatch (${got} != ${exp})" >&2; exit 1; }
    echo "    sha256 ok ${got:0:12}"
fi
# The tarball carries its own top-level dir (<snap>/vision_carrot, <snap>/qflow_svf_merged, ...).
[ -d "${IFQL_ROOT}/code/${SNAP}/vision_carrot" ] || tar -xzf "${TARBALL}" -C "${IFQL_ROOT}/code"
SERVER="${IFQL_ROOT}/code/${SNAP}/vision_carrot/ifql_server.py"
[ -f "${SERVER}" ] || { echo "ERROR: ${SERVER} not found after extract" >&2; exit 1; }
[ -d "${IFQL_ROOT}/code/${SNAP}/qflow_svf_merged/agents" ] || { echo "ERROR: qflow_svf_merged/agents missing" >&2; exit 1; }
if grep -q 'd.get("D_c") or' "${SERVER}"; then
    echo "    D_c-null patch already present"
else
    cp "${SERVER}" "${SERVER}.orig"
    (cd "${IFQL_ROOT}/code/${SNAP}" && patch -p1 < "${HERE}/ifql_server_Dc_null.patch")
fi

# --- 3) venv ------------------------------------------------------------------------
echo "### [3/5] uv venv (${PY_VER}) + CUDA wheels"
[ -x "${IFQL_ROOT}/.venv-svf/bin/python" ] || uv venv "${IFQL_ROOT}/.venv-svf" --python "${PY_VER}"
uv pip install --python "${IFQL_ROOT}/.venv-svf/bin/python" --no-cache \
    --index-strategy unsafe-best-match -r "${HERE}/requirements-svf-infer.txt"
# Import check only. A missing/broken GPU is reported, not fatal: the launcher falls back
# to --device cpu, and CUDA can be unavailable here for reasons unrelated to the venv
# (no driver yet, or a laptop that suspended/resumed without the nvidia-suspend services).
XLA_PYTHON_CLIENT_PREALLOCATE=false "${IFQL_ROOT}/.venv-svf/bin/python" - <<'EOF' 2>/dev/null
import jax, torch, torchvision, zmq, cv2
try:
    dev = jax.devices()
except Exception as e:  # cuda plugin present but cuInit failed -> jax has no backend at all
    dev = f"UNAVAILABLE ({type(e).__name__})"
print(f"    jax {jax.__version__} devices={dev}  torch {torch.__version__} cuda={torch.cuda.is_available()}")
EOF

# --- 4) TORCH_HOME: ResNet18 + dinov2 (FeatureHeads loads BOTH unconditionally) --------
echo "### [4/5] torch.hub cache"
TORCH_HOME="${IFQL_ROOT}/.cache/torch" "${IFQL_ROOT}/.venv-svf/bin/python" - <<'EOF'
import torch, torchvision
torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
print("    resnet18 + dinov2_vits14 cached in", torch.hub.get_dir())
EOF

# --- 5) server --help smoke (no bind) ---------------------------------------------------
echo "### [5/5] ifql_server.py --help"
(cd "$(dirname "${SERVER}")" && QFLOW_DIR="${IFQL_ROOT}/code/${SNAP}/qflow_svf_merged" \
    "${IFQL_ROOT}/.venv-svf/bin/python" ifql_server.py --help >/dev/null) && echo "    ok"

if [ "${IFQL_TASK}" = "orange" ]; then
    _NEXT_ENV="IFQL_ROOT=${IFQL_ROOT} IFQL_TASK=orange IFQL_SAMPLER=bc HEADLESS=true"
else
    _NEXT_ENV="IFQL_ROOT=${IFQL_ROOT} IFQL_SAMPLER=bc HEADLESS=true"
fi
cat <<EOF

### DONE (task=${IFQL_TASK}). Next (robot PC, ROS Humble, this repo built):
###   cd ros2_ur_ws && ./launch_cameras.sh                                   # T1
###   ${_NEXT_ENV} ./run_ur7e_ifql_real.sh   # T2
### (IFQL_ROOT is only needed if it is not \$HOME/carrot_ifql; IFQL_TASK is only needed
###  for orange — carrot stays the default.)
EOF
