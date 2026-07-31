#!/usr/bin/env bash
# =============================================================================
# run_hil_server.sh -- start/reuse the remote HIL-SERL learner and hold its tunnel
# =============================================================================
#
# Normal use (this terminal remains the tunnel owner):
#
#   ./run_hil_server.sh
#
# Read-only status, with no learner start and no tunnel:
#
#   ./run_hil_server.sh --check
#
# Select the physical GPU on the learner host and the fresh run name used only
# if no healthy learner already exists:
#
#   ./run_hil_server.sh --gpu 0 --run-id cube_in_cup_real_20260729_220000
#
# Demand a new lineage instead of reusing a healthy learner.  This never stops
# the old learner: if one exists, the command refuses and asks the operator to
# stop it explicitly first.
#
#   ./run_hil_server.sh --new-lineage --gpu 0 --run-id my_fresh_run
#
# Ownership is intentionally narrow:
#
#   * A production learner is server-owned.  It is detached on the learner
#     host and keeps running when this script exits.
#   * This script owns only the SSH process it creates for
#       laptop 127.0.0.1:50153 -> learner host 127.0.0.1:50053.
#   * Ctrl-C, TERM, tunnel failure, and local gRPC-probe failure clean up only
#     that SSH child.  They never signal a reused or newly started learner.
#
# The learner host moved kanu -> junhyeong_ai on 2026-07-31 and the defaults
# below moved with it.  This script can no longer drive kanu at all, and that
# is fail-closed rather than an oversight (measured 2026-07-31, not inferred):
# kanu's artifacts were never under one root -- its classifier sits inside an
# unrelated stack's dataset directory -- so no HIL_REMOTE_DATA_ROOT satisfies
# it, and its still-running learner was launched with --run-name
# "<run>-kanu-5000" and the old classifier path, both of which
# validate_process_contract compares by exact string equality.  Inspect kanu
# read-only instead, and never signal anything there:
#
#   ssh kanu 'ps -p <pid> -o pid,etime,cmd'
#   ssh kanu PYTHONPATH=<kanu repo>/serl_ur_infra <kanu python> -c \
#       'GrpcActorNetwork("127.0.0.1:50053", ...).health()'
#
# Reverting this single commit restores the kanu defaults wholesale.
#
# Configuration overrides:
#
#   HIL_SSH_HOST             default junhyeong_ai
#   HIL_GPU_INDEX            default 0 (new starts only; reuse reports actual)
#   HIL_RUN_ID               default cube_in_cup_real_<UTC timestamp>
#   HIL_START_TIMEOUT_S      default 300
#   HIL_LOCAL_PORT           default 50153
#   HIL_REMOTE_PORT          fixed production default 50053
#   HIL_REMOTE_REPO          default /home/junhyeong/gello_software_runtime
#                            (deprecated alias: HIL_KANU_REPO)
#   HIL_REMOTE_PYTHON        default /home/junhyeong/miniconda3/envs/il/bin/python
#                            (deprecated alias: HIL_KANU_PYTHON)
#   HIL_REMOTE_DATA_ROOT     default /home/junhyeong/hil-serl-data -- the single
#                            root holding runs/, demos/, classifier_ckpt/ and
#                            the launcher lock
#   ACTOR_VENV               default /home/laptop3/venvs/gello-hil-actor
#   HIL_ACCEPT_HEAD_MISMATCH default 0; 1 downgrades the fresh-lineage refusal
#                            on a learner-host/laptop3 HEAD mismatch to a
#                            warning (offline emergency only -- see "code
#                            identity")
#
# The artifact pins, feature-ring sizes, RAM reserve, reward contract and
# learner options below are the production command from
# serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md; that runbook is still correct
# about the learner options and is kanu-worded about hosts and paths, which
# serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md supersedes.  Do not turn
# this into a generic remote-process runner: strict reuse is what prevents an
# old/synthetic learner from silently receiving real robot transitions.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SSH_HOST="${HIL_SSH_HOST:-junhyeong_ai}"
GPU_INDEX="${HIL_GPU_INDEX:-0}"
RUN_ID="${HIL_RUN_ID:-}"
START_TIMEOUT_S="${HIL_START_TIMEOUT_S:-300}"
LOCAL_PORT="${HIL_LOCAL_PORT:-50153}"
REMOTE_PORT="${HIL_REMOTE_PORT:-50053}"
# HIL_KANU_* survive as fallback aliases -- same shape as
# DEADMAN_WAIT_S/ENGAGE_WAIT_S -- so an operator shell or a runbook that still
# exports the old names keeps working across the host move.
REMOTE_REPO="${HIL_REMOTE_REPO:-${HIL_KANU_REPO:-/home/junhyeong/gello_software_runtime}}"
REMOTE_PYTHON="${HIL_REMOTE_PYTHON:-${HIL_KANU_PYTHON:-/home/junhyeong/miniconda3/envs/il/bin/python}}"
# One root for every server-side artifact this launcher touches.  Until
# 2026-07-31 the run base, the demo pickle, the classifier checkpoint and the
# launcher lock were four independent hardcoded absolute paths -- one of them
# buried inside an unrelated stack's dataset directory -- so moving the learner
# to another machine meant finding them one at a time and missing one silently.
# The SHA pins below are unchanged: identity is content, not location, which is
# what lets an artifact move and still prove it arrived intact.
REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"
REMOTE_RUN_BASE="$REMOTE_DATA_ROOT/runs"
ACTOR_VENV="${ACTOR_VENV:-/home/laptop3/venvs/gello-hil-actor}"
ACTOR_PY="$ACTOR_VENV/bin/python"

MODE="start"
NEW_LINEAGE=0
RUN_ID_EXPLICIT=0
[[ -n "$RUN_ID" ]] && RUN_ID_EXPLICIT=1

usage() {
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" |
        sed '$d; s/^# \{0,1\}//'
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check|--status)
            MODE="check"
            shift
            ;;
        --new-lineage)
            NEW_LINEAGE=1
            shift
            ;;
        --gpu)
            need_value "$1" "${2:-}"
            GPU_INDEX="$2"
            shift 2
            ;;
        --gpu=*)
            GPU_INDEX="${1#*=}"
            shift
            ;;
        --run-id)
            need_value "$1" "${2:-}"
            RUN_ID="$2"
            RUN_ID_EXPLICIT=1
            shift 2
            ;;
        --run-id=*)
            RUN_ID="${1#*=}"
            RUN_ID_EXPLICIT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ "$START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || \
    die "HIL_START_TIMEOUT_S must be a positive integer"
(( START_TIMEOUT_S <= 1800 )) || die "HIL_START_TIMEOUT_S must be <= 1800"
for port_name in LOCAL_PORT REMOTE_PORT; do
    port_value="${!port_name}"
    [[ "$port_value" =~ ^[0-9]+$ ]] || die "$port_name must be numeric"
    (( port_value >= 1 && port_value <= 65535 )) || \
        die "$port_name must be in [1, 65535]"
done
[[ "$REMOTE_PORT" == "50053" ]] || \
    die "production learner port is fixed at 50053 (got $REMOTE_PORT)"
[[ -n "$SSH_HOST" ]] || die "HIL_SSH_HOST cannot be empty"
# The data root becomes four remote paths and is marshalled as a bare
# positional, so keep it boring: absolute, no trailing slash, no whitespace or
# shell metacharacters.  A rejected override here is far cheaper than a remote
# path that expands into something plausible but wrong.
[[ -n "$REMOTE_DATA_ROOT" ]] || die "HIL_REMOTE_DATA_ROOT cannot be empty"
[[ "$REMOTE_DATA_ROOT" =~ ^/[A-Za-z0-9._/-]*[A-Za-z0-9._-]$ ]] || \
    die "HIL_REMOTE_DATA_ROOT must be an absolute path of [A-Za-z0-9._/-] with no trailing slash (got: $REMOTE_DATA_ROOT)"
[[ "$REMOTE_DATA_ROOT" != *".."* ]] || \
    die "HIL_REMOTE_DATA_ROOT must not contain '..'"

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="cube_in_cup_real_$(date -u +%Y%m%d_%H%M%S)"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "run ID may contain only A-Z, a-z, 0-9, dot, underscore and dash"
(( ${#RUN_ID} <= 128 )) || die "run ID must be at most 128 characters"
if [[ "$MODE" == "check" && "$NEW_LINEAGE" == "1" ]]; then
    die "--new-lineage cannot be combined with --check"
fi

command -v ssh >/dev/null 2>&1 || die "ssh is not installed"
if [[ "$MODE" == "start" && ! -x "$ACTOR_PY" ]]; then
    die "actor venv python is missing: $ACTOR_PY"
fi

local_port_is_free() {
    /usr/bin/python3 - "$LOCAL_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f"local port {sys.argv[1]} is unavailable: {exc}")
finally:
    sock.close()
PY
}

# Check before any remote mutation.  A second check after the learner is ready
# closes the unavoidable race where another local process binds during the
# learner's startup warm-up; even in that race the learner remains
# server-owned.
if [[ "$MODE" == "start" ]] && ! local_port_is_free; then
    die "local port $LOCAL_PORT is occupied; no remote learner operation was attempted"
fi

# ---------------------------------------------------------------------------
# Cross-host code identity.
#
# The defect this exists for (kanu, 2026-07-30, back when kanu was the learner
# host): its HIL checkout had `origin` pointing at a LOCAL PATH instead of
# GitHub.  `git fetch origin` there returned rc=0 and fetched nothing, so the
# learner host sat a full day behind laptop3 while every existing check still
# passed.  Nothing in the stack ever compared the learner host's HEAD to
# laptop3's, so the staleness was invisible until someone nearly reset the live
# learner's checkout to that stale FETCH_HEAD.
#
# The fix compares the two hosts DIRECTLY over the ssh channel that already
# connects them, rather than asking GitHub whether either one is current.  The
# failure mode was the learner host diverging from laptop3, so laptop3 is the
# reference and network reachability becomes irrelevant to the guard.  The
# GitHub tip is only a best-effort third opinion (remote-side
# diagnose_github_tip).
#
# ACCEPTED LIMITATION, stated so nobody oversells it: this compares committed
# HEADs only.  laptop3's working tree is routinely dirty -- the entire
# 2026-07-30 evening operator batch ran uncommitted -- so "HEADs match" does NOT
# mean the learner host holds the exact bytes the actor is executing.  Shipping
# the working tree is out of scope by design.
LAPTOP_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
LAPTOP_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ ! "$LAPTOP_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
    LAPTOP_HEAD="unknown"
    echo "WARNING: cannot read laptop3 HEAD from $REPO_ROOT; the learner-host/laptop3 code identity check will report relation=laptop_head_unknown." >&2
fi
[[ -n "$LAPTOP_BRANCH" ]] || LAPTOP_BRANCH="unknown"
ACCEPT_HEAD_MISMATCH="${HIL_ACCEPT_HEAD_MISMATCH:-0}"

SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=60
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
)

# The remote program is passed on stdin.  In check mode it performs no mkdir,
# lock-file open, process start, signal, or other external mutation.
set +e
REMOTE_OUTPUT="$(
    ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" bash -s -- \
        "$MODE" "$NEW_LINEAGE" "$GPU_INDEX" "$RUN_ID" \
        "$START_TIMEOUT_S" "$REMOTE_REPO" "$REMOTE_PYTHON" \
        "$REMOTE_PORT" "$REMOTE_RUN_BASE" \
        "$LAPTOP_HEAD" "$LAPTOP_BRANCH" "$ACCEPT_HEAD_MISMATCH" \
        "$REMOTE_DATA_ROOT" <<'REMOTE_SCRIPT'
set -euo pipefail

MODE="$1"
NEW_LINEAGE="$2"
GPU_INDEX="$3"
RUN_ID="$4"
START_TIMEOUT_S="$5"
# Canonicalize a stable deployment symlink before building the exact process
# contract; both fresh starts and reuse checks then compare the same real path.
REMOTE_REPO="$(readlink -f "$6")"
REMOTE_PYTHON="$7"
REMOTE_PORT="$8"
REMOTE_RUN_BASE="$9"
# laptop3's committed code identity, measured on the laptop and carried over the
# same ssh channel; see the laptop-side "Cross-host code identity" comment.
LAPTOP_HEAD="${10}"
LAPTOP_BRANCH="${11}"
ACCEPT_HEAD_MISMATCH="${12}"
# The single root every server-side artifact hangs off.  The names below are
# fixed; only the root is configurable, because the SHA pins -- not the paths --
# are what identify these files.
REMOTE_DATA_ROOT="${13}"

CLASSIFIER="$REMOTE_DATA_ROOT/classifier_ckpt/checkpoint_150"
CLASSIFIER_SHA256="512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d"
REAL_DEMO="$REMOTE_DATA_ROOT/demos/cube_in_cup_20260720_success_23takes.pkl"
REAL_DEMO_SHA256="f97185582401ce7570d44fddc33d1bd64b215d7e32d6384d5fe13e1b405032fa"
RESNET_SOURCE="$REMOTE_REPO/third_party/hil-serl/examples/experiments/resnet10_params.pkl"
RESNET_SHA256="175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
REWARD_THRESHOLD="0.5"
REWARD_MODEL_ID="cube-in-cup-all3-ckpt150+sidecar-v1"
POLICY_MODEL_ID="hil-serl-hybrid-sac-resnet10-trunk-cache-v1"
OBSERVATION_SCHEMA_HASH="3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903"
RUN_LOCK="$REMOTE_DATA_ROOT/.run_hil_server.lock"

remote_die() {
    echo "REMOTE ERROR: $*" >&2
    exit 1
}

# REMOTE_RUN_BASE arrives as its own positional because the laptop validates the
# reported run root against it.  Assert the two agree, so a mis-ordered
# positional dies here instead of quietly pointing a fresh lineage at a
# plausible-looking wrong directory.
[[ "$REMOTE_RUN_BASE" == "$REMOTE_DATA_ROOT/runs" ]] || \
    remote_die "run base '$REMOTE_RUN_BASE' does not derive from data root '$REMOTE_DATA_ROOT'; the launcher arguments are inconsistent"

validate_static_contract() {
    [[ -d "$REMOTE_REPO" ]] || remote_die "learner-host repo is missing: $REMOTE_REPO"
    [[ -x "$REMOTE_PYTHON" ]] || remote_die "learner-host Python is missing: $REMOTE_PYTHON"
    [[ -f "$REMOTE_REPO/serl_ur_infra/scripts/run_rlpd_learner_server.py" ]] || \
        remote_die "learner entrypoint is missing from $REMOTE_REPO"

    # --- repository topology (network-free) ---------------------------------
    # A linked git worktree chains its object store and its refs to ANOTHER
    # local checkout, so what this repo reports can be advanced or rewound by
    # work done elsewhere on the learner host that nothing here can see.  That
    # is exactly the defect class this guard exists to eliminate, so refuse the
    # shape rather than try to audit the checkout it is chained to.
    local git_dir git_common_dir origin_url origin_pattern
    git_dir="$(git -C "$REMOTE_REPO" rev-parse --git-dir 2>/dev/null)" || \
        remote_die "learner-host checkout is not a git repository: $REMOTE_REPO"
    git_common_dir="$(git -C "$REMOTE_REPO" rev-parse --git-common-dir 2>/dev/null)" || \
        remote_die "learner-host checkout is not a git repository: $REMOTE_REPO"
    [[ "$git_dir" == "$git_common_dir" ]] || \
        remote_die "learner-host checkout is a linked git worktree chained to another local checkout (git-dir=$git_dir, git-common-dir=$git_common_dir): $REMOTE_REPO"

    # `origin` pointing at a LOCAL PATH is the precise 2026-07-30 defect: fetch
    # returned rc=0, transferred nothing, and left this checkout a full day
    # stale while looking healthy.  Pin origin to the canonical GitHub remote so
    # a silent no-op fetch cannot happen again.
    origin_url="$(git -C "$REMOTE_REPO" remote get-url origin 2>/dev/null || true)"
    origin_pattern='^(https://|ssh://git@|git@)github\.com[:/]Bigenlight/gello_software(\.git)?$'
    [[ "$origin_url" =~ $origin_pattern ]] || \
        remote_die "learner-host 'origin' is not the canonical GitHub remote (got: ${origin_url:-<none>}); a local-path origin makes 'git fetch origin' a silent no-op"

    git -C "$REMOTE_REPO" merge-base --is-ancestor ca19652 HEAD || \
        remote_die "learner-host checkout does not contain required learner baseline ca19652"

    submodule_state="$(git -C "$REMOTE_REPO" submodule status third_party/hil-serl)"
    [[ "${submodule_state:0:1}" == " " ]] || \
        remote_die "third_party/hil-serl is missing or not at the recorded gitlink: $submodule_state"

    printf '%s  %s\n' "$REAL_DEMO_SHA256" "$REAL_DEMO" |
        sha256sum --check --strict --status || \
        remote_die "production demo SHA mismatch: $REAL_DEMO"
    printf '%s  %s\n' "$RESNET_SHA256" "$RESNET_SOURCE" |
        sha256sum --check --strict --status || \
        remote_die "ResNet-10 SHA mismatch: $RESNET_SOURCE"

    actual_classifier_sha="$(
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$REMOTE_REPO/serl_ur_infra" \
        "$REMOTE_PYTHON" - "$CLASSIFIER" <<'PY'
import sys
from ur_env.classifier_sidecar import directory_sha256
print(directory_sha256(sys.argv[1]))
PY
    )"
    [[ "$actual_classifier_sha" == "$CLASSIFIER_SHA256" ]] || \
        remote_die "classifier directory SHA mismatch: $actual_classifier_sha"

    defaults="$(
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$REMOTE_REPO/serl_ur_infra" \
        "$REMOTE_PYTHON" - <<'PY'
from ur_env.classifier_sidecar import CLASSIFIER_INPUT_ID
from ur_env.rlpd_receive_server import (
    DEFAULT_CLASSIFIER_CONFIRMATIONS,
    DEFAULT_REWARD_THRESHOLD,
)
print(DEFAULT_REWARD_THRESHOLD)
print(DEFAULT_CLASSIFIER_CONFIRMATIONS)
print(CLASSIFIER_INPUT_ID)
PY
    )"
    [[ "$defaults" == $'0.5\n1\nfullframe-jpeg-passthrough-v1' ]] || \
        remote_die "remote reward defaults no longer match the pinned production contract: $defaults"
}

# ---------------------------------------------------------------------------
# Cross-host code identity.  HEAD_RELATION is one of:
#   match | remote_behind | remote_ahead | diverged | remote_never_fetched |
#   laptop_head_unknown | unchecked
# It is only ever FATAL at the fresh-lineage gate further down; every other
# caller (reuse, --check) gets the banner and proceeds.  See that gate for the
# rationale of the split.
# ---------------------------------------------------------------------------
HEAD_RELATION="unchecked"
REMOTE_HEAD=""
REMOTE_BRANCH=""

# Best-effort third opinion, never fatal.  This deliberately uses ls-remote and
# NEVER `git fetch`: fetch writes refs into the very checkout being audited and
# can block indefinitely on an unreachable remote.  ls-remote only reads, and is
# bounded by `timeout`.
diagnose_github_tip() {
    local ref line github_head
    if [[ "$LAPTOP_BRANCH" == "unknown" || "$LAPTOP_BRANCH" == "HEAD" ]]; then
        echo "GitHub tip UNVERIFIED: laptop3 is not on a named branch, so there is nothing to look up. The laptop3<->learner-host comparison above is authoritative." >&2
        return 0
    fi
    ref="refs/heads/$LAPTOP_BRANCH"
    if ! line="$(timeout 8 git -C "$REMOTE_REPO" ls-remote origin "$ref" 2>/dev/null)"; then
        echo "GitHub tip UNVERIFIED: 'git ls-remote origin $ref' failed or timed out on the learner host (no fetch was attempted). The laptop3<->learner-host comparison above is authoritative." >&2
        return 0
    fi
    github_head="${line%%[[:space:]]*}"
    if [[ -z "$github_head" ]]; then
        echo "GitHub has no $ref: neither laptop3 nor the learner host has pushed this branch. The laptop3<->learner-host comparison above is authoritative." >&2
        return 0
    fi
    echo "HIL_SERVER_GITHUB_TIP=branch=$LAPTOP_BRANCH head=$github_head"
    {
        if [[ "$github_head" == "$LAPTOP_HEAD" && "$github_head" == "$REMOTE_HEAD" ]]; then
            echo "Three-way: laptop3, the learner host and GitHub $ref all agree ($github_head)."
        elif [[ "$LAPTOP_HEAD" == "$REMOTE_HEAD" ]]; then
            echo "Three-way: laptop3 and the learner host agree ($LAPTOP_HEAD) but GitHub $ref is $github_head -- the agreeing pair is UNPUSHED (or GitHub carries work neither host has)."
        elif [[ "$github_head" == "$LAPTOP_HEAD" ]]; then
            echo "Three-way: laptop3 matches GitHub $ref; THE LEARNER HOST is the stale one."
        elif [[ "$github_head" == "$REMOTE_HEAD" ]]; then
            echo "Three-way: the learner host matches GitHub $ref; LAPTOP3 is the odd one out (unpushed local commits, or laptop3 is behind)."
        else
            echo "Three-way: laptop3, the learner host and GitHub $ref are ALL different."
        fi
    } >&2
}

compare_heads() {
    # A missing or non-git REMOTE_REPO is reported far more precisely by
    # validate_static_contract, which runs immediately after this.  Stay silent
    # rather than pre-empt it with a raw git error.
    [[ -d "$REMOTE_REPO" ]] || return 0
    git -C "$REMOTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || return 0
    REMOTE_HEAD="$(git -C "$REMOTE_REPO" rev-parse HEAD 2>/dev/null || true)"
    REMOTE_BRANCH="$(git -C "$REMOTE_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    [[ -n "$REMOTE_HEAD" ]] || return 0

    if [[ ! "$LAPTOP_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
        HEAD_RELATION="laptop_head_unknown"
    elif [[ "$LAPTOP_HEAD" == "$REMOTE_HEAD" ]]; then
        HEAD_RELATION="match"
    elif ! git -C "$REMOTE_REPO" cat-file -e "$LAPTOP_HEAD" 2>/dev/null; then
        # The learner host's object store has never even seen laptop3's commit.
        # This is the signature of the incident: a fetch that reports success
        # while transferring nothing.
        HEAD_RELATION="remote_never_fetched"
    elif git -C "$REMOTE_REPO" merge-base --is-ancestor "$REMOTE_HEAD" "$LAPTOP_HEAD" 2>/dev/null; then
        HEAD_RELATION="remote_behind"
    elif git -C "$REMOTE_REPO" merge-base --is-ancestor "$LAPTOP_HEAD" "$REMOTE_HEAD" 2>/dev/null; then
        HEAD_RELATION="remote_ahead"
    else
        HEAD_RELATION="diverged"
    fi

    if [[ "$HEAD_RELATION" == "match" ]]; then
        echo "HIL_SERVER_HEAD_MATCH=laptop=$LAPTOP_HEAD remote=$REMOTE_HEAD relation=match"
        diagnose_github_tip
        return 0
    fi

    echo "HIL_SERVER_HEAD_MISMATCH=laptop=$LAPTOP_HEAD remote=$REMOTE_HEAD relation=$HEAD_RELATION"
    {
        echo "=================================================================="
        echo "  !!  LEARNER HOST / LAPTOP3 CODE IDENTITY MISMATCH  !!"
        echo "=================================================================="
        echo "  laptop3 : $LAPTOP_HEAD  (${LAPTOP_BRANCH})"
        echo "  remote  : $REMOTE_HEAD  (${REMOTE_BRANCH:-unknown})"
        echo "  remote repo: $REMOTE_REPO"
        echo "  relation: $HEAD_RELATION"
        echo "------------------------------------------------------------------"
        case "$HEAD_RELATION" in
            remote_never_fetched)
                echo "  The learner host's object store does not contain laptop3's"
                echo "  commit AT ALL.  That is the signature of a fetch that reported"
                echo "  success and transferred nothing (for example an 'origin' that"
                echo "  points at a local path).  It has never seen this work."
                ;;
            remote_behind)
                echo "  The learner host is BEHIND laptop3: it has the commit but has"
                echo "  not checked it out.  The learner is running older code than the"
                echo "  actor."
                ;;
            remote_ahead)
                echo "  The learner host is AHEAD of laptop3: laptop3 is running older"
                echo "  code than the learner, or laptop3 was rewound."
                ;;
            diverged)
                echo "  The checkouts have DIVERGED: neither HEAD contains the other."
                ;;
            laptop_head_unknown)
                echo "  laptop3's HEAD could not be read, so no comparison was made."
                ;;
        esac
        echo "------------------------------------------------------------------"
        echo "  This is a REPORT, not a repair.  Nothing here fetches, checks"
        echo "  out, merges or resets either side.  Align them by hand."
        echo "=================================================================="
    } >&2
    diagnose_github_tip
}

list_learner_pids() {
    "$REMOTE_PYTHON" - <<'PY'
from pathlib import Path

result = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = [
            part.decode("utf-8", "surrogateescape")
            for part in (entry / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(arg.endswith("/run_rlpd_learner_server.py") for arg in argv):
        result.append(int(entry.name))
for pid in sorted(result):
    print(pid)
PY
}

validate_process_contract() {
    local pid="$1"
    "$REMOTE_PYTHON" - "$pid" "$REMOTE_REPO" "$REMOTE_PYTHON" \
        "$REMOTE_PORT" "$CLASSIFIER" "$CLASSIFIER_SHA256" \
        "$REWARD_THRESHOLD" "$REWARD_MODEL_ID" "$REAL_DEMO" \
        "$RESNET_SOURCE" <<'PY'
import os
from pathlib import Path
import sys

(
    pid_text,
    repo,
    expected_python,
    port,
    classifier,
    classifier_sha,
    threshold,
    reward_model_id,
    demo,
    resnet_source,
) = sys.argv[1:]
pid = int(pid_text)
proc = Path("/proc") / str(pid)

try:
    argv = [
        part.decode("utf-8", "surrogateescape")
        for part in (proc / "cmdline").read_bytes().split(b"\0")
        if part
    ]
except FileNotFoundError:
    raise SystemExit(f"learner PID {pid} disappeared")

def values(flag):
    found = []
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 >= len(argv):
                raise SystemExit(f"{flag} has no value in PID {pid}")
            found.append(argv[index + 1])
    return found

def one(flag):
    found = values(flag)
    if len(found) != 1:
        raise SystemExit(f"PID {pid}: expected exactly one {flag}, got {found}")
    return found[0]

def exact(flag, expected):
    actual = one(flag)
    if actual != expected:
        raise SystemExit(
            f"PID {pid}: {flag}={actual!r}, expected production value {expected!r}"
        )

def number(flag, expected):
    actual = one(flag)
    try:
        matches = float(actual) == float(expected)
    except ValueError:
        matches = False
    if not matches:
        raise SystemExit(
            f"PID {pid}: {flag}={actual!r}, expected numeric value {expected!r}"
        )

if not any(arg.endswith("/run_rlpd_learner_server.py") for arg in argv):
    raise SystemExit(f"PID {pid} is not the learner entrypoint")
if os.path.realpath(argv[0]) != os.path.realpath(expected_python):
    raise SystemExit(f"PID {pid} uses unexpected Python: {argv[0]}")
if os.path.realpath(proc / "cwd") != os.path.realpath(repo):
    raise SystemExit(f"PID {pid} uses unexpected cwd: {os.path.realpath(proc / 'cwd')}")

for forbidden in (
    "--dry-run",
    "--synthetic-e2e",
    "--resume-path",
    "--resume-latest",
):
    if forbidden in argv:
        raise SystemExit(f"PID {pid} is not a fresh production learner: {forbidden}")

exact("--host", "127.0.0.1")
exact("--port", port)
exact("--classifier-checkpoint", classifier)
exact("--expected-classifier-sha256", classifier_sha)
number("--reward-threshold", threshold)
exact("--reward-model-id", reward_model_id)
number("--success-confirmations", "1")
if values("--demo-path") != [demo]:
    raise SystemExit(f"PID {pid}: production demo path mismatch: {values('--demo-path')}")
number("--checkpoint-reserve-gib", "2")
exact("--wandb-mode", "offline")
exact("--wandb-project", "hil-serl")
exact("--hil-serl-root", f"{repo}/third_party/hil-serl")
exact("--resnet-source", resnet_source)
number("--replay-capacity", "50000")
number("--intervention-capacity", "10000")
number("--feature-memory-reserve-gib", "2")
number("--demo-extraction-batch-size", "64")
number("--grasp-penalty", "-0.02")
number("--utd-ratio", "1")
number("--max-workers", "4")
number("--max-message-bytes", "16777216")
exact("--require-jax-backend", "gpu")
number("--target-learner-step", "5000")
number("--poll-interval", "0.1")

checkpoint_root = Path(one("--checkpoint-root"))
run_root = checkpoint_root.parent
expected_paths = {
    "--jsonl-path": run_root / "logs" / "learner.jsonl",
    "--memory-preflight-path": run_root / "logs" / "memory-preflight.jsonl",
    "--wandb-dir": run_root / "wandb",
    "--resnet-cache": run_root / "assets" / "resnet10_params.pkl",
}
if checkpoint_root.name != "checkpoints":
    raise SystemExit(f"PID {pid}: checkpoint root is not RUN_ROOT/checkpoints")
for flag, expected in expected_paths.items():
    if Path(one(flag)) != expected:
        raise SystemExit(f"PID {pid}: {flag} is outside the learner run root")
exact("--run-name", f"{run_root.name}-hil-5000")

environment = {}
for item in (proc / "environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "replace")] = value.decode(
            "utf-8", "replace"
        )
gpu = environment.get("CUDA_VISIBLE_DEVICES", "")
if not gpu.isdigit():
    raise SystemExit(f"PID {pid}: CUDA_VISIBLE_DEVICES is not one physical GPU: {gpu!r}")
if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
    raise SystemExit(f"PID {pid}: XLA preallocation must be disabled")
expected_pythonpath = f"{repo}/serl_ur_infra:{repo}/third_party/hil-serl/serl_launcher"
if environment.get("PYTHONPATH") != expected_pythonpath:
    raise SystemExit(
        f"PID {pid}: PYTHONPATH does not match the production learner contract"
    )
if environment.get("WANDB_SILENT") != "true":
    raise SystemExit(f"PID {pid}: WANDB_SILENT must be true")
if environment.get("WANDB_DISABLE_CODE") != "true":
    raise SystemExit(f"PID {pid}: WANDB_DISABLE_CODE must be true")

print(f"HIL_SERVER_PID={pid}")
print(f"HIL_SERVER_RUN_ROOT={run_root}")
print(f"HIL_SERVER_GPU={gpu}")
PY
}

probe_server() {
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$REMOTE_REPO/serl_ur_infra" \
    timeout 12 "$REMOTE_PYTHON" - "$REMOTE_PORT" \
        "$POLICY_MODEL_ID" "$REWARD_MODEL_ID" "$OBSERVATION_SCHEMA_HASH" <<'PY'
from dataclasses import asdict
import json
import sys

from ur_env.grpc_actor_transport import GrpcActorNetwork

port, model_id, reward_model_id, schema_hash = sys.argv[1:]
network = GrpcActorNetwork(
    f"127.0.0.1:{port}",
    actor_id="run-hil-server-readonly-probe",
    action_shape=(7,),
    timeout_s=2.0,
    max_response_age_s=3.0,
    expected_observation_schema_hash=schema_hash,
    expected_model_id=model_id,
    expected_reward_authority="server_classifier",
    expected_reward_model_id=reward_model_id,
)
try:
    alive, ready, detail = network.health()
    if not alive or not ready:
        raise SystemExit(f"learner health is not ready: {alive=} {ready=} {detail=}")
    info = network.get_server_info()
    status = network.get_buffer_status()
    print(json.dumps({
        "health": detail,
        "server": asdict(info),
        "buffer": asdict(status),
    }, sort_keys=True, separators=(",", ":")))
finally:
    network.close()
PY
}

validate_ready_evidence() {
    local run_root="$1"
    "$REMOTE_PYTHON" - "$run_root" <<'PY'
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
jsonl_path = run_root / "logs" / "learner.jsonl"
if not jsonl_path.is_file():
    raise SystemExit(f"ready evidence is missing: {jsonl_path}")

raw = jsonl_path.read_bytes()
lines = raw.splitlines()
records = []
for index, line in enumerate(lines, start=1):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        # An active JSONL writer can be observed between write(2) calls.  Only
        # an unterminated final fragment is retryable; earlier damage is not.
        if index == len(lines) and not raw.endswith(b"\n"):
            continue
        raise SystemExit(f"invalid learner JSONL at line {index}: {exc}")
    if not isinstance(record, dict):
        raise SystemExit(f"learner JSONL line {index} is not an object")
    records.append((index, record))

warmups = [item for item in records if item[1].get("event") == "learner_update_warmup_complete"]
ready = [item for item in records if item[1].get("event") == "learner_process_ready"]
if len(warmups) != 1 or len(ready) != 1:
    raise SystemExit(
        "fresh-lineage ready evidence must contain exactly one warmup and one "
        f"process-ready event (got warmup={len(warmups)}, ready={len(ready)})"
    )

warmup_line, warmup = warmups[0]
ready_line, process_ready = ready[0]
if warmup_line >= ready_line:
    raise SystemExit("learner process-ready event precedes its update warmup")
expected_warmup = {
    "warmup_outer_steps": 3,
    "warmup_gradient_updates": 6,
    "learner_step": 0,
    "gradient_step": 0,
    "policy_version": 0,
    "production_state_advanced": False,
}
for key, expected in expected_warmup.items():
    if warmup.get(key) != expected:
        raise SystemExit(
            f"warmup evidence {key}={warmup.get(key)!r}, expected {expected!r}"
        )
expected_ready = {
    "jax_backend": "gpu",
    "demo_count": 2037,
    "synthetic_acceptance_demo_count": 0,
    "restored_checkpoint": None,
    "learner_step": 0,
    "gradient_step": 0,
    "policy_version": 0,
}
for key, expected in expected_ready.items():
    if process_ready.get(key) != expected:
        raise SystemExit(
            f"process-ready evidence {key}={process_ready.get(key)!r}, "
            f"expected {expected!r}"
        )

print("HIL_SERVER_READY_EVIDENCE=" + json.dumps({
    "demo_count": process_ready["demo_count"],
    "jax_backend": process_ready["jax_backend"],
    "warmup_gradient_updates": warmup["warmup_gradient_updates"],
    "warmup_outer_steps": warmup["warmup_outer_steps"],
}, sort_keys=True, separators=(",", ":")))
PY
}

port_is_open() {
    "$REMOTE_PYTHON" - "$REMOTE_PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.4):
        pass
except OSError:
    raise SystemExit(1)
PY
}

inspect_existing() {
    local pids pid contract probe run_root
    mapfile -t pids < <(list_learner_pids)
    if (( ${#pids[@]} == 0 )); then
        if port_is_open; then
            remote_die "port $REMOTE_PORT is occupied by a non-learner process"
        fi
        return 2
    fi
    if (( ${#pids[@]} != 1 )); then
        remote_die "refusing duplicate learners; found PIDs: ${pids[*]}"
    fi
    pid="${pids[0]}"
    contract="$(validate_process_contract "$pid")" || return 1
    printf '%s\n' "$contract"
    if probe="$(probe_server 2>/dev/null)"; then
        run_root="$(printf '%s\n' "$contract" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p')"
        [[ -n "$run_root" ]] || remote_die "learner contract omitted its run root"
        validate_ready_evidence "$run_root" || return 1
        printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
        return 0
    fi
    return 3
}

# compare_heads runs BEFORE validate_static_contract even though the spec for
# this feature said "right after", and the deviation is deliberate: the
# comparison is never fatal at this point (the only fatal use is the
# fresh-lineage gate below), while a checkout wrong enough to fail the static
# contract -- missing learner entrypoint, missing baseline commit ca19652 -- is
# usually wrong precisely BECAUSE it is a different lineage.  Reporting the HEAD
# relation first turns "learner entrypoint is missing" into "...and here is why".
# Nothing is weakened: validate_static_contract runs immediately after with
# every one of its checks intact and still aborts the run on failure.
compare_heads
validate_static_contract

if [[ "$MODE" == "check" ]]; then
    set +e
    inspect_existing
    status=$?
    set -e
    case "$status" in
        0)
            echo "HIL_SERVER_RESULT=healthy"
            exit 0
            ;;
        2)
            echo "HIL_SERVER_RESULT=absent"
            exit 3
            ;;
        3)
            echo "HIL_SERVER_RESULT=present_but_not_ready"
            exit 4
            ;;
        *)
            exit 1
            ;;
    esac
fi

# Start/reuse is serialized before checking for a process.  The learner itself
# also owns a per-lineage writer lock, but it binds gRPC only after expensive
# JAX warm-up; this outer lock prevents two launchers from doing that warm-up
# concurrently during the pre-bind window.
command -v flock >/dev/null 2>&1 || remote_die "flock is required on the learner host"
exec 9>"$RUN_LOCK"
flock -w 10 9 || remote_die "another run_hil_server launcher holds $RUN_LOCK"

set +e
existing_output="$(inspect_existing)"
existing_status=$?
set -e
if [[ "$existing_status" -eq 0 ]]; then
    printf '%s\n' "$existing_output"
    if [[ "$NEW_LINEAGE" == "1" ]]; then
        remote_die "--new-lineage refuses to reuse the active learner; stop it explicitly first"
    fi
    echo "HIL_SERVER_RESULT=reused"
    exit 0
fi
if [[ "$existing_status" -eq 1 ]]; then
    printf '%s\n' "$existing_output"
    remote_die "the existing learner does not match the production contract"
fi
if [[ "$existing_status" -eq 3 ]]; then
    printf '%s\n' "$existing_output"
    if [[ "$NEW_LINEAGE" == "1" ]]; then
        remote_die "--new-lineage refuses while a matching learner is initializing; stop it explicitly first"
    fi
    # A matching process may simply be in its 80-150 second startup warm-up.
    # Never launch a competitor; wait for this one under the launcher lock.
    initializing_pid="$(printf '%s\n' "$existing_output" | sed -n 's/^HIL_SERVER_PID=//p')"
    initializing_run_root="$(printf '%s\n' "$existing_output" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p')"
    echo "learner PID $initializing_pid exists and is initializing; waiting for ready..." >&2
    deadline=$(( $(date +%s) + START_TIMEOUT_S ))
    while kill -0 "$initializing_pid" 2>/dev/null; do
        if probe="$(probe_server 2>/dev/null)"; then
            validate_ready_evidence "$initializing_run_root" || \
                remote_die "existing learner became healthy without valid fresh-lineage ready evidence"
            printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
            echo "HIL_SERVER_RESULT=reused"
            exit 0
        fi
        if (( $(date +%s) >= deadline )); then
            remote_die "existing learner PID $initializing_pid did not become ready within ${START_TIMEOUT_S}s; it was not stopped"
        fi
        sleep 2
    done
    remote_die "existing learner PID $initializing_pid exited during initialization"
fi
[[ "$existing_status" -eq 2 ]] || remote_die "unexpected learner inspection status $existing_status"

# A fresh lineage is permanent: every transition it will ever learn from is
# produced by the actor running laptop3's code, and a lineage born from a
# different learner-host commit cannot be repaired afterwards.  This is
# therefore the one place where a HEAD mismatch is fatal.
#
# Reuse and --check deliberately do NOT die on a mismatch.  Hard-failing reuse
# would force one of two worse outcomes: advancing the learner host's checkout
# underneath a live learner -- which imports modules lazily, so it would then be
# running a mixture of two versions -- or freezing laptop3 development until
# the lineage ends.  Cross-version safety for a running learner is enforced
# by the observation-schema hash, the policy/reward model IDs and
# validate_process_contract.  The defect being cured here is the SILENCE, not
# the mismatch itself.
if [[ "$HEAD_RELATION" != "match" ]]; then
    if [[ "$ACCEPT_HEAD_MISMATCH" == "1" ]]; then
        echo "HIL_ACCEPT_HEAD_MISMATCH=1: starting a fresh lineage despite code identity relation '$HEAD_RELATION'." >&2
    else
        remote_die "refusing to start a fresh lineage while the learner host and laptop3 are not on the same commit (relation=$HEAD_RELATION, laptop=$LAPTOP_HEAD, remote=${REMOTE_HEAD:-unknown}); align the checkouts, or set HIL_ACCEPT_HEAD_MISMATCH=1 to accept it deliberately"
    fi
fi

dirty_checkout="$(git -C "$REMOTE_REPO" status --porcelain --untracked-files=normal)"
[[ -z "$dirty_checkout" ]] || \
    remote_die "learner-host checkout is dirty; refusing a new production learner: $dirty_checkout"

gpu_name="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader 2>/dev/null)" || \
    remote_die "physical GPU $GPU_INDEX does not exist"
gpu_pids="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if printf '%s\n' "$gpu_pids" | grep -Eq '[0-9]'; then
    remote_die "physical GPU $GPU_INDEX is already used by PID(s): $(printf '%s' "$gpu_pids" | tr '\n' ' ')"
fi

memory_values="$(
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$REMOTE_REPO/serl_ur_infra" \
    "$REMOTE_PYTHON" - <<'PY'
from ur_env.learner import (
    estimate_feature_demo_memory,
    estimate_feature_replay_memory,
    system_available_memory_bytes,
)
replay = estimate_feature_replay_memory(
    replay_capacity=50_000,
    intervention_capacity=10_000,
).fixed_tensor_bytes
demo = estimate_feature_demo_memory(2_037).total_bytes
reserve = 2 * 1024**3
print(system_available_memory_bytes(), replay + demo + reserve)
PY
)"
read -r available_bytes required_bytes <<<"$memory_values"
(( available_bytes >= required_bytes )) || \
    remote_die "RAM preflight rejected before launch: available=$available_bytes required=$required_bytes"

run_root="$REMOTE_RUN_BASE/$RUN_ID"
[[ ! -e "$run_root" ]] || \
    remote_die "fresh lineage root already exists: $run_root (choose a new --run-id)"
mkdir -p "$REMOTE_RUN_BASE"
umask 077
mkdir "$run_root"
mkdir -p "$run_root/logs" "$run_root/wandb" "$run_root/assets"

checkpoint_root="$run_root/checkpoints"
jsonl_path="$run_root/logs/learner.jsonl"
memory_path="$run_root/logs/memory-preflight.jsonl"
wandb_dir="$run_root/wandb"
resnet_cache="$run_root/assets/resnet10_params.pkl"
stdout_path="$run_root/logs/stdout.log"

echo "Starting production learner on physical GPU $GPU_INDEX ($gpu_name)" >&2
echo "Fresh lineage: $run_root" >&2
cd "$REMOTE_REPO"
nohup env \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_SILENT=true \
    WANDB_DISABLE_CODE=true \
    PYTHONPATH="$REMOTE_REPO/serl_ur_infra:$REMOTE_REPO/third_party/hil-serl/serl_launcher" \
    "$REMOTE_PYTHON" \
    serl_ur_infra/scripts/run_rlpd_learner_server.py \
    --host 127.0.0.1 \
    --port "$REMOTE_PORT" \
    --classifier-checkpoint "$CLASSIFIER" \
    --expected-classifier-sha256 "$CLASSIFIER_SHA256" \
    --reward-threshold "$REWARD_THRESHOLD" \
    --reward-model-id "$REWARD_MODEL_ID" \
    --success-confirmations 1 \
    --demo-path "$REAL_DEMO" \
    --checkpoint-root "$checkpoint_root" \
    --checkpoint-reserve-gib 2 \
    --jsonl-path "$jsonl_path" \
    --memory-preflight-path "$memory_path" \
    --wandb-dir "$wandb_dir" \
    --wandb-mode offline \
    --wandb-project hil-serl \
    --run-name "$RUN_ID-hil-5000" \
    --hil-serl-root "$REMOTE_REPO/third_party/hil-serl" \
    --resnet-source "$RESNET_SOURCE" \
    --resnet-cache "$resnet_cache" \
    --replay-capacity 50000 \
    --intervention-capacity 10000 \
    --feature-memory-reserve-gib 2 \
    --demo-extraction-batch-size 64 \
    --grasp-penalty -0.02 \
    --utd-ratio 1 \
    --max-workers 4 \
    --max-message-bytes 16777216 \
    --require-jax-backend gpu \
    --target-learner-step 5000 \
    --poll-interval 0.1 \
    >"$stdout_path" 2>&1 </dev/null 9>&- &
learner_pid=$!

kill -0 "$learner_pid" 2>/dev/null || \
    remote_die "learner exited immediately; inspect $stdout_path"
contract=""
for _ in $(seq 1 50); do
    if contract="$(validate_process_contract "$learner_pid" 2>/dev/null)"; then
        break
    fi
    kill -0 "$learner_pid" 2>/dev/null || \
        remote_die "learner exited immediately; inspect $stdout_path"
    sleep 0.1
done
[[ -n "$contract" ]] || \
    remote_die "new learner PID $learner_pid did not expose its production command contract"
printf '%s\n' "$contract"
echo "HIL_SERVER_MEMORY_AVAILABLE_BYTES=$available_bytes"
echo "HIL_SERVER_MEMORY_REQUIRED_BYTES=$required_bytes"

deadline=$(( $(date +%s) + START_TIMEOUT_S ))
while kill -0 "$learner_pid" 2>/dev/null; do
    if probe="$(probe_server 2>/dev/null)"; then
        validate_ready_evidence "$run_root" || \
            remote_die "new learner became healthy without valid fresh-lineage ready evidence; it was not stopped"
        printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
        echo "HIL_SERVER_RESULT=started"
        exit 0
    fi
    if (( $(date +%s) >= deadline )); then
        echo "Learner startup log tail:" >&2
        tail -n 30 "$stdout_path" >&2 || true
        remote_die "learner PID $learner_pid did not become ready within ${START_TIMEOUT_S}s; it remains server-owned and was not stopped"
    fi
    sleep 2
done

echo "Learner startup log tail:" >&2
tail -n 50 "$stdout_path" >&2 || true
if [[ -f "$memory_path" ]]; then
    echo "Learner RAM decisions:" >&2
    tail -n 2 "$memory_path" >&2 || true
fi
remote_die "learner PID $learner_pid exited before ready; evidence is preserved at $run_root"
REMOTE_SCRIPT
)"
REMOTE_RC=$?
set -e

printf '%s\n' "$REMOTE_OUTPUT"
if [[ "$REMOTE_RC" -ne 0 ]]; then
    if [[ "$MODE" == "check" && "$REMOTE_RC" -eq 3 ]]; then
        echo "No production learner is running on $SSH_HOST."
    fi
    exit "$REMOTE_RC"
fi

if [[ "$MODE" == "check" ]]; then
    echo "CHECK PASS: one exact production learner is healthy; no tunnel was opened."
    exit 0
fi

SERVER_RESULT="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_RESULT=//p' | tail -n1)"
SERVER_PID="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_PID=//p' | tail -n1)"
SERVER_RUN_ROOT="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p' | tail -n1)"
SERVER_GPU="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_GPU=//p' | tail -n1)"
[[ "$SERVER_RESULT" == "started" || "$SERVER_RESULT" == "reused" ]] || \
    die "remote launcher returned an invalid result: $SERVER_RESULT"
[[ "$SERVER_PID" =~ ^[0-9]+$ ]] || die "remote launcher did not report a learner PID"
[[ "$SERVER_GPU" =~ ^[0-9]+$ ]] || die "remote launcher did not report a physical GPU"
[[ "$SERVER_RUN_ROOT" == "$REMOTE_RUN_BASE/"* ]] || \
    die "remote launcher reported an unexpected run root: $SERVER_RUN_ROOT"
if [[ "$SERVER_RESULT" == "reused" ]]; then
    if [[ "$SERVER_GPU" != "$GPU_INDEX" ]]; then
        echo "INFO: healthy learner already uses physical GPU $SERVER_GPU; requested GPU $GPU_INDEX applies only to a new start." >&2
    fi
    if [[ "$RUN_ID_EXPLICIT" == "1" && "${SERVER_RUN_ROOT##*/}" != "$RUN_ID" ]]; then
        echo "INFO: healthy learner lineage ${SERVER_RUN_ROOT##*/} was reused; requested run ID $RUN_ID applies only to a new start." >&2
    fi
fi

# Refuse if another local process claimed the actor endpoint while the remote
# learner was being prepared.
# We never inspect-and-kill an old tunnel: ownership cannot be inferred safely
# from a port number alone.
if ! local_port_is_free; then
    die "local port $LOCAL_PORT is occupied; no process was stopped and learner PID $SERVER_PID remains running"
fi

TUNNEL_PID=""
cleanup_tunnel() {
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""
    fi
}
on_signal() {
    exit 130
}
trap cleanup_tunnel EXIT
trap on_signal INT TERM HUP

echo "Opening tunnel: 127.0.0.1:$LOCAL_PORT -> $SSH_HOST:127.0.0.1:$REMOTE_PORT"
ssh -N -T "${SSH_OPTIONS[@]}" \
    -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    "$SSH_HOST" &
TUNNEL_PID=$!

tunnel_ready=0
for _ in $(seq 1 40); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        wait "$TUNNEL_PID" || true
        die "SSH tunnel exited during startup; learner PID $SERVER_PID was not stopped"
    fi
    if /usr/bin/python3 - "$LOCAL_PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
        pass
except OSError:
    raise SystemExit(1)
PY
    then
        tunnel_ready=1
        break
    fi
    sleep 0.25
done
[[ "$tunnel_ready" -eq 1 ]] || \
    die "tunnel did not accept TCP within 10s; learner PID $SERVER_PID was not stopped"

# Verify the actual gRPC contract through the exact endpoint run_hil_actor.sh
# uses.  This is GetServerInfo/GetBufferStatus only: it does not begin an
# episode, consume policy RNG, insert replay, or command the robot.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO_ROOT/serl_ur_infra" \
timeout 15 "$ACTOR_PY" - "$LOCAL_PORT" <<'PY'
from dataclasses import asdict
import json
import sys

from ur_env.grpc_actor_transport import GrpcActorNetwork

network = GrpcActorNetwork(
    f"127.0.0.1:{sys.argv[1]}",
    actor_id="run-hil-server-local-readonly-probe",
    action_shape=(7,),
    timeout_s=2.0,
    max_response_age_s=3.0,
    expected_observation_schema_hash=(
        "3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903"
    ),
    expected_model_id="hil-serl-hybrid-sac-resnet10-trunk-cache-v1",
    expected_reward_authority="server_classifier",
    expected_reward_model_id="cube-in-cup-all3-ckpt150+sidecar-v1",
)
try:
    alive, ready, detail = network.health()
    if not alive or not ready:
        raise SystemExit(f"remote learner is not ready: {alive=} {ready=} {detail=}")
    info = network.get_server_info()
    status = network.get_buffer_status()
    print("Tunnel gRPC check:", json.dumps({
        "server": asdict(info),
        "buffer": asdict(status),
    }, sort_keys=True, separators=(",", ":")))
finally:
    network.close()
PY

echo "============================================================"
echo "HIL server ready"
echo "  learner : $SERVER_RESULT PID $SERVER_PID (server-owned)"
echo "  GPU     : physical $SERVER_GPU"
echo "  run root: $SERVER_RUN_ROOT"
echo "  actor   : grpc://127.0.0.1:$LOCAL_PORT"
echo "  tunnel  : PID $TUNNEL_PID (owned by this launcher)"
echo "============================================================"
echo "Keep this terminal open. Ctrl-C closes only the tunnel."
echo "The learner on $SSH_HOST keeps running until you stop it explicitly there."

set +e
wait "$TUNNEL_PID"
TUNNEL_RC=$?
set -e
TUNNEL_PID=""
if [[ "$TUNNEL_RC" -ne 0 ]]; then
    echo "SSH tunnel exited with status $TUNNEL_RC; learner PID $SERVER_PID is still running." >&2
fi
exit "$TUNNEL_RC"
