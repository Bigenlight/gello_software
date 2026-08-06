#!/usr/bin/env bash
# setup_local_policy_venv.sh — build /home/laptop3/venvs/gello-local-policy (CUDA jax).
#
# WHY THIS EXISTS
#   BC/FM policy serving has so far run on the junhyeong_ai GPU server. To measure
#   inference / step latency on laptop3's own RTX 3060 (6 GB, sm_86, driver 595.84)
#   we need a venv that is package-identical to ~/venvs/hilserl except that jax is
#   the CUDA build instead of the CPU build. Same jax version (0.5.3) as the server's
#   `il` env, so local-GPU numbers are comparable to server-GPU numbers.
#
# CONTRACT
#   - Idempotent: re-running a good venv re-verifies and exits 0 ("already ready").
#   - Touches NOTHING else. The two existing venvs (hilserl, gello-hil-actor) are
#     read-only inputs; the only write outside the new venv is this script's log line.
#   - Fails loudly. There is no "probably fine" exit path: either jax.devices() shows
#     a CUDA device and a real GPU matmul comes back, or the script exits non-zero.
#
# 🪤 THE FREEZE TRAP — read before "fixing" the requirements step.
#   `hilserl/bin/pip freeze` (no flags) prints 247 lines, not 65. That venv has
#   `include-system-site-packages = false` but ships a
#   `zz-system-dist-packages.pth` that appends /usr/lib/python3/dist-packages, plus a
#   `serl_launcher.pth`. So a plain freeze sweeps in apt-managed and ROS packages —
#   apturl, python-apt==2.4.0+ubuntu4.1, devscripts===2.22.1ubuntu1, rclpy,
#   h5py.-debian-h5py-serial, xkit==0.0.0 — most of which do not exist on PyPI at all.
#   Installing that list into a fresh venv cannot succeed. We therefore use
#   `pip freeze --local`, which is exactly the 65 packages the hilserl venv itself
#   owns, and which is what "mirror hilserl's package set" actually means.
#   (Verified 2026-08-06: --local == a PYTHONPATH-scrubbed freeze minus the .pth paths.)
#
#   Deliberately NOT reproduced in the new venv: `zz-system-dist-packages.pth` and
#   `serl_launcher.pth`. The bench and the policy server scripts insert
#   serl_ur_infra + third_party/hil-serl/serl_launcher into sys.path themselves, and
#   leaving /usr/lib/python3/dist-packages off sys.path keeps the new venv
#   self-contained (pip resolves six/decorator/gast as real dependencies instead of
#   silently borrowing the system copies, which is how hilserl gets them today).
#
# USAGE
#   ros2_ur_ws/setup_local_policy_venv.sh
#
# ENV OVERRIDES
#   LOCAL_POLICY_VENV       target venv path      (default /home/laptop3/venvs/gello-local-policy)
#   LOCAL_POLICY_SRC_VENV   package-set source    (default /home/laptop3/venvs/hilserl, read-only)
#   LOCAL_POLICY_JAX_SPEC   jax requirement line  (default jax[cuda12]==0.5.3)
#   LOCAL_POLICY_MIN_FREE_G minimum free GB on /  (default 6)
#   LOCAL_POLICY_RECREATE=1 delete and rebuild the venv instead of installing on top

set -euo pipefail

VENV_DIR="${LOCAL_POLICY_VENV:-/home/laptop3/venvs/gello-local-policy}"
SRC_VENV="${LOCAL_POLICY_SRC_VENV:-/home/laptop3/venvs/hilserl}"
JAX_SPEC="${LOCAL_POLICY_JAX_SPEC:-jax[cuda12]==0.5.3}"
MIN_FREE_G="${LOCAL_POLICY_MIN_FREE_G:-6}"
RECREATE="${LOCAL_POLICY_RECREATE:-0}"

VENV_PARENT="$(dirname "$VENV_DIR")"
REQ_FILE="$VENV_DIR/requirements.local-policy.txt"

# Pins the brief says must survive the freeze filter. If the source venv drifts and
# one of these disappears, that is a real change we must not paper over.
REQUIRED_PINS=(
  "numpy==1.26.4"
  "flax==0.10.5"
  "grpcio==1.74.0"
)

log()  { printf '[setup-local-policy-venv] %s\n' "$*"; }
warn() { printf '[setup-local-policy-venv] WARN: %s\n' "$*" >&2; }
die()  { printf '[setup-local-policy-venv] FATAL: %s\n' "$*" >&2; exit 1; }

snapshot() {
  # $1 = label
  log "df -h / ($1):"
  df -h / | sed 's/^/    /'
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi memory.used ($1):"
    nvidia-smi --query-gpu=memory.used --format=csv 2>&1 | sed 's/^/    /'
  else
    warn "nvidia-smi not on PATH — cannot snapshot GPU memory ($1)"
  fi
}

# ---------------------------------------------------------------------------
# Verification — this is BOTH the final gate and the idempotency check.
# Runs with PYTHONPATH scrubbed so we are testing the venv itself, not whatever
# ROS overlay happens to be sourced in the caller's shell.
# ---------------------------------------------------------------------------
VERIFY_PY='
import sys

failures = []

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - reported, not raised
    print("VERIFY: import jax FAILED: %r" % (exc,))
    sys.exit(1)

devs = jax.devices()
print("VERIFY: jax.__version__ =", jax.__version__)
print("VERIFY: jax.devices() =", devs)

gpu_devs = [d for d in devs if getattr(d, "platform", "") in ("gpu", "cuda")]
if not gpu_devs:
    failures.append(
        "no CUDA device in jax.devices() -> %r (jax default backend=%s)"
        % (devs, jax.default_backend())
    )
else:
    print("VERIFY: cuda device kind =", getattr(gpu_devs[0], "device_kind", "?"))

if gpu_devs:
    try:
        import numpy as _np
        dev = gpu_devs[0]
        a = jnp.asarray(_np.arange(256 * 256, dtype=_np.float32).reshape(256, 256) / 65536.0)
        a = jax.device_put(a, dev)
        out = (a @ a.T).block_until_ready()
        where = list(out.devices())
        print("VERIFY: matmul out.shape =", out.shape, "devices =", where)
        if not any(getattr(d, "platform", "") in ("gpu", "cuda") for d in where):
            failures.append("matmul result did not land on a CUDA device: %r" % (where,))
        val = float(out[0, 0])
        if not (val == val and abs(val) < 1e30):  # NaN / inf guard
            failures.append("matmul produced a non-finite value: %r" % (val,))
        else:
            print("VERIFY: matmul out[0,0] =", val)
    except Exception as exc:
        failures.append("GPU matmul FAILED: %r" % (exc,))

for mod in ("grpc", "flax", "distrax", "optax", "chex"):
    try:
        __import__(mod)
        print("VERIFY: import %s OK" % mod)
    except Exception as exc:
        failures.append("import %s FAILED: %r" % (mod, exc))

try:
    import numpy
    print("VERIFY: numpy.__version__ =", numpy.__version__)
except Exception as exc:
    failures.append("import numpy FAILED: %r" % (exc,))

if failures:
    print("VERIFY: FAIL")
    for f in failures:
        print("VERIFY:   - " + f)
    sys.exit(1)

print("VERIFY: PASS")
sys.exit(0)
'

run_verification() {
  local py="$VENV_DIR/bin/python"
  [[ -x "$py" ]] || return 1
  # The JAX_* / CUDA_VISIBLE_DEVICES scrub is load-bearing, not tidiness.  This
  # same function is the IDEMPOTENCY check, so anything that makes it fail
  # spuriously triggers a full multi-gigabyte reinstall AND an error message
  # that blames the venv.  A `JAX_PLATFORMS=cpu` left exported in the caller's
  # shell -- the single most likely export on a laptop that also runs the CPU
  # fallback -- forces jax.devices() to CPU, so a perfectly good CUDA venv
  # reports "no CUDA device in jax.devices()".  Measured.  We are testing the
  # venv, so the venv's own defaults are the only ones allowed to speak.
  env -u PYTHONPATH \
      -u JAX_PLATFORMS \
      -u JAX_PLATFORM_NAME \
      -u CUDA_VISIBLE_DEVICES \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$py" -c "$VERIFY_PY"
}

# ---------------------------------------------------------------------------
# [1/6] pre-checks
# ---------------------------------------------------------------------------
log "[1/6] pre-checks"

avail_g="$(df -BG --output=avail / | tail -n1 | tr -dc '0-9')"
[[ -n "$avail_g" ]] || die "could not parse free space from df on /"
if (( avail_g < MIN_FREE_G )); then
  die "only ${avail_g}G free on / — need >${MIN_FREE_G}G for the CUDA wheels (~2-3G download, ~4-5G installed). Free space and re-run."
fi
log "free space on /: ${avail_g}G (need >${MIN_FREE_G}G) OK"

command -v python3 >/dev/null 2>&1 || die "system python3 not found on PATH"
PY_VER="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
PY_MM="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$PY_MM" == "3.10" ]] || die "system python3 is ${PY_VER}; this venv must be 3.10 to match ~/venvs/hilserl and the jax 0.5.3 wheels"
python3 -c 'import venv' >/dev/null 2>&1 || die "python3-venv module missing (apt install python3.10-venv)"
log "system python3: ${PY_VER} OK"

[[ -d "$VENV_PARENT" ]] || die "$VENV_PARENT does not exist"
[[ -w "$VENV_PARENT" ]] || die "$VENV_PARENT is not writable"
log "$VENV_PARENT writable OK"

# --- forbidden targets ------------------------------------------------------
# LOCAL_POLICY_VENV is an env override, and with LOCAL_POLICY_RECREATE=1 the
# very next section runs `rm -rf "$VENV_DIR"`.  Point it at ~/venvs/hilserl --
# the obvious typo, since that is also the value of LOCAL_POLICY_SRC_VENV --
# and this script deletes the interpreter that the whole HIL stack, the CPU
# fallback, and this script's OWN package-set source depend on.  Resolved with
# readlink -f so a symlink cannot smuggle it past a string compare.
RESOLVED_VENV="$(readlink -f -- "$VENV_DIR" 2>/dev/null || printf '%s' "$VENV_DIR")"
for _forbidden in "$SRC_VENV" /home/laptop3/venvs/hilserl /home/laptop3/venvs/gello-hil-actor; do
  _resolved_forbidden="$(readlink -f -- "$_forbidden" 2>/dev/null || printf '%s' "$_forbidden")"
  [[ "$RESOLVED_VENV" != "$_resolved_forbidden" ]] || die \
    "LOCAL_POLICY_VENV resolves to $_resolved_forbidden, which this script must never write to or delete.
   ($VENV_DIR -> $RESOLVED_VENV)
   That venv is a read-only input here: it is either the package-set source, the
   documented CPU fallback interpreter, or the actor's gRPC interpreter.  With
   LOCAL_POLICY_RECREATE=1 the next step would have removed it outright.
   Pick a fresh path, e.g. LOCAL_POLICY_VENV=/home/laptop3/venvs/gello-local-policy"
done
log "target venv is not a protected venv: $RESOLVED_VENV OK"

# --- one builder at a time --------------------------------------------------
# Two concurrent runs install into the SAME site-packages: pip has no
# cross-process locking, so the second one unpacks wheels over files the first
# is still writing and the loser is a venv that imports halfway.  The lock is
# held for the whole script (fd 9 stays open until it exits), which also covers
# the rm -rf in the recreate path.
command -v flock >/dev/null 2>&1 || die "flock not found (util-linux); refusing to build without the concurrency lock"
LOCK_FILE="$VENV_PARENT/.setup_local_policy_venv.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || die \
  "another setup_local_policy_venv.sh already holds $LOCK_FILE.
   A second concurrent build would pip-install into the same site-packages as the
   running one, mid-unpack.  See the build that is running:
       fuser -v $LOCK_FILE
       ps -o pid,etime,cmd -p \$(fuser $LOCK_FILE 2>/dev/null)
   Wait for it to finish (a cold CUDA install is ~10 min), then re-run.  Nothing
   was created or modified."
log "build lock acquired: $LOCK_FILE"

[[ -x "$SRC_VENV/bin/pip" ]] || die "source venv pip not found: $SRC_VENV/bin/pip"
log "package-set source (read-only): $SRC_VENV"

snapshot "before"

# ---------------------------------------------------------------------------
# [2/6] idempotency — an already-good venv is left completely alone
# ---------------------------------------------------------------------------
log "[2/6] idempotency check"
if [[ "$RECREATE" == "1" ]]; then
  if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
    log "LOCAL_POLICY_RECREATE=1 — removing existing venv $VENV_DIR"
    rm -rf "$VENV_DIR"
  else
    log "LOCAL_POLICY_RECREATE=1 but no venv at $VENV_DIR — nothing to remove"
  fi
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  log "venv exists — running verification before deciding to install"
  if verify_out="$(run_verification 2>&1)"; then
    printf '%s\n' "$verify_out" | sed 's/^/    /'
    log "already ready — $VENV_DIR verifies clean, nothing to do"
    snapshot "after (no-op)"
    exit 0
  fi
  printf '%s\n' "$verify_out" | sed 's/^/    /'
  warn "existing venv did not verify — will install on top of it (set LOCAL_POLICY_RECREATE=1 to rebuild from scratch)"
else
  log "no venv at $VENV_DIR — will create it"
fi

# ---------------------------------------------------------------------------
# [3/6] create venv + upgrade pip
# ---------------------------------------------------------------------------
log "[3/6] create venv + upgrade pip"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
  log "created $VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
env -u PYTHONPATH "$VENV_PY" -m pip install --upgrade --progress-bar off pip
env -u PYTHONPATH "$VENV_PY" -m pip --version

# ---------------------------------------------------------------------------
# [4/6] build the requirements file from the source venv's OWN packages
# ---------------------------------------------------------------------------
log "[4/6] build requirements from ${SRC_VENV} (pip freeze --local)"
raw_freeze="$(env -u PYTHONPATH "$SRC_VENV/bin/pip" freeze --local)"
raw_count="$(printf '%s\n' "$raw_freeze" | grep -c . || true)"
log "source venv owns ${raw_count} packages"

# Filter per plan ADDENDUM item 3:
#   - serl_launcher    : .pth injection artifact, not on PyPI
#   - agentlace @ git+ : git direct reference, not needed for local serving
#   - jax / jaxlib     : replaced by the single CUDA spec below
filtered="$(printf '%s\n' "$raw_freeze" \
  | grep -v -E '^[[:space:]]*$' \
  | grep -v -i -E '^serl[-_]launcher[ =@]' \
  | grep -v -i -E '^agentlace[ =@]' \
  | grep -v -i -E '^jax(\[[^]]*\])?[ =@]' \
  | grep -v -i -E '^jaxlib[ =@]' || true)"
[[ -n "$filtered" ]] || die "the freeze filter removed every line — refusing to build an empty venv"

# Any remaining direct reference (`name @ url`) would be an unreviewed install source.
if leftovers="$(printf '%s\n' "$filtered" | grep -E ' @ ' || true)"; [[ -n "$leftovers" ]]; then
  warn "requirements still contain direct references (not in the reviewed drop-list):"
  printf '%s\n' "$leftovers" | sed 's/^/    /' >&2
fi

{
  printf '# generated by ros2_ur_ws/setup_local_policy_venv.sh on %s\n' "$(date -Is)"
  printf '# source: %s/bin/pip freeze --local  (%s packages)\n' "$SRC_VENV" "$raw_count"
  printf '# dropped: serl_launcher (.pth artifact), agentlace (git ref), jax, jaxlib\n'
  printf '# added:   %s\n' "$JAX_SPEC"
  printf '%s\n' "$filtered"
  printf '%s\n' "$JAX_SPEC"
} > "$REQ_FILE"

# -F is load-bearing: the jax spec contains [cuda12], which a regex grep would read
# as a character class and "match" against jaxc==0.5.3.
for pin in "${REQUIRED_PINS[@]}"; do
  grep -q -i -F -x -- "$pin" "$REQ_FILE" \
    || die "expected pin '${pin}' is missing from the generated requirements — ${SRC_VENV} has drifted; review $REQ_FILE before proceeding"
done
grep -q -i -E '^protobuf==' "$REQ_FILE" \
  || die "no protobuf pin in the generated requirements — review $REQ_FILE"
grep -q -i -F -x -- "$JAX_SPEC" "$REQ_FILE" \
  || die "jax spec '${JAX_SPEC}' did not make it into $REQ_FILE"

log "requirements written: $REQ_FILE ($(grep -c -v '^#' "$REQ_FILE") requirement lines)"
log "pins kept: $(grep -i -E '^(numpy|flax|grpcio|protobuf)==' "$REQ_FILE" | tr '\n' ' ')"
log "jax spec:  $JAX_SPEC"

# ---------------------------------------------------------------------------
# [5/6] install
# ---------------------------------------------------------------------------
log "[5/6] pip install (downloads ~2-3G of nvidia CUDA wheels on a cold cache)"
env -u PYTHONPATH "$VENV_PY" -m pip install --progress-bar off -r "$REQ_FILE"

# ---------------------------------------------------------------------------
# [6/6] verify
# ---------------------------------------------------------------------------
log "[6/6] verification (XLA_PYTHON_CLIENT_PREALLOCATE=false, PYTHONPATH scrubbed)"
if ! run_verification; then
  snapshot "after (FAILED)"
  die "verification FAILED — the venv exists but is not usable for local GPU serving. See the VERIFY: lines above."
fi

snapshot "after"
log "venv size: $(du -sh "$VENV_DIR" | cut -f1)"
log "PASS — $VENV_DIR is ready for local GPU policy serving"
log "use it as: $VENV_PY"
