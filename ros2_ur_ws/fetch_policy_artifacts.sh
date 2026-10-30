#!/usr/bin/env bash
# =============================================================================
# fetch_policy_artifacts.sh -- copy the BC/FM policy artifacts to laptop3 and
#                              verify every digest they declare  (gate G2)
# =============================================================================
#
#   ./fetch_policy_artifacts.sh
#
# One-time (and idempotent) setup step for LOCAL policy evaluation: the BC and
# FM serving scripts need their artifact directory on THIS machine, because the
# server scripts' _DEFAULT_ARTIFACT_DIR points at /home/junhyeong/... .  This
# mirrors the server layout so nothing downstream has to learn a second path:
#
#   junhyeong_ai:~/hil-serl-data/diagnostics/<name>  ->  ~/hil-serl-data/diagnostics/<name>
#
# WHAT THIS SCRIPT IS FOR -- the copy is the easy half.  The point is the
# verification: an artifact that arrives truncated, or that is silently the
# 50-epoch run instead of the 20-epoch one, would still load and still serve a
# robot.  So every digest the artifacts declare about themselves is recomputed
# here with sha256sum, and for BC the repo's own loader gates are run too.
#
# OWNERSHIP -- read before editing:
#
#   * The server is touched READ-ONLY: one `ls -ld` per artifact and `scp -r`.
#     No remote write, no remote compute, no remote process.
#   * /home/junhyeong/gello_software (no _runtime suffix) is SOMEONE ELSE'S work
#     tree.  Every remote path is refused below if it names gello_software.
#   * Nothing local is deleted automatically.  A digest mismatch on an artifact
#     that is already here is reported and the script exits non-zero; disposing
#     of a bad copy is an operator decision, because the same symptom is
#     produced by "wrong artifact fetched" and by "someone is mid-experiment".
#     The one exception is this script's OWN staging directory (.staging.*),
#     which it created moments earlier and which never becomes the real path
#     until the copy finished -- that is what keeps a half-copy from being
#     mistaken for a verified artifact on the next run.  An EXIT/INT/TERM trap
#     removes it on any interrupted run, and preflight sweeps .staging.* older
#     than 60 minutes (only SIGKILL can leave one behind).
#
# Overrides (defaults shown):  HIL_SSH_HOST=junhyeong_ai ·
# HIL_REMOTE_DATA_ROOT=/home/junhyeong/hil-serl-data ·
# HIL_LOCAL_DATA_ROOT=/home/laptop3/hil-serl-data ·
# BC_ARTIFACT_NAME=<20epoch .bc-init> · FM_ARTIFACT_NAME=<200epoch .fm-init> ·
# HIL_ACTOR_PYTHON=/home/laptop3/venvs/gello-hil-actor/bin/python
#
# Exit codes: 0 = every artifact present and verified · 1 = usage/preflight ·
# 2 = copy failed · 3 = verification failed (nothing deleted).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SSH_HOST="${HIL_SSH_HOST:-junhyeong_ai}"
REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"
LOCAL_DATA_ROOT="${HIL_LOCAL_DATA_ROOT:-/home/laptop3/hil-serl-data}"
ACTOR_PYTHON="${HIL_ACTOR_PYTHON:-/home/laptop3/venvs/gello-hil-actor/bin/python}"

# The artifacts this evaluation is pinned to.  Names are the identity: the
# diagnostics directory also holds a 50-epoch BC run and a 50-epoch FM run.
DEFAULT_BC_ARTIFACT_NAME="bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init"
DEFAULT_FM_ARTIFACT_NAME="jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init"
BC_ARTIFACT_NAME="${BC_ARTIFACT_NAME:-$DEFAULT_BC_ARTIFACT_NAME}"
FM_ARTIFACT_NAME="${FM_ARTIFACT_NAME:-$DEFAULT_FM_ARTIFACT_NAME}"

# The frozen trunk both policies were trained against.  Local already -- the
# repo carries it -- so this is verification only, never a copy.  ~/.serl holds
# a cache copy; the source of truth for the digest is the repo path.
RESNET_PATH="$REPO_ROOT/third_party/hil-serl/examples/experiments/resnet10_params.pkl"

# Plan pins.  The manifests are the authority (they are what the loaders check),
# these are the second opinion that catches "verified fine -- but it is the
# wrong run".  Applied only when the artifact name was not overridden.
PIN_RESNET_SHA256="175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
PIN_BC_PARAM_SHA256="8ffcfac59c40a427c9359d824f5bfe765d6929991e0e506fed851e5ae0d4eadf"
PIN_FM_PARAM_SHA256="2df39048a5a775aeda47af56a67c226a5bd17b7899affce29dabdca419ebf396"

EXIT_PREFLIGHT=1
EXIT_COPY=2
EXIT_VERIFY=3

SUMMARY_ROWS=()

# The staging directory currently in flight, or empty.  Global on purpose: the
# only way to clean it up on a Ctrl-C in the middle of an scp is from a trap,
# and a trap cannot see fetch_dir's locals.  A leaked .staging.* is not
# cosmetic on a disk sitting at 96% -- it is a hidden multi-megabyte directory
# that nothing ever looks at again.
STAGING=""

# --- small helpers ----------------------------------------------------------

say()  { printf '%s\n' "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }
die()  { local code="$1"; shift; printf '\nFATAL: %s\n' "$*" >&2; exit "$code"; }

sha256_of() {
    # sha256sum, not a python digest: the value printed here must be one an
    # operator can reproduce by hand from the shell.
    sha256sum -- "$1" | cut -d' ' -f1
}

json_get() {
    # json_get <file> <key> [<key> ...]  -- stdlib only, no jax anywhere near it
    "$ACTOR_PYTHON" -c '
import json, sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2:]:
    obj = obj[key]
print(obj)
' "$@"
}

human_size() { du -sh -- "$1" | cut -f1; }

add_row() { SUMMARY_ROWS+=("$1|$2|$3|$4"); }

cleanup_staging() {
    # A function rather than an inline `rm -rf -- "${STAGING:-}"` trap because
    # the same cleanup is wanted at four points -- scp failure, "scp produced
    # nothing", success, and the trap -- and because clearing STAGING afterwards
    # makes it idempotent: once the artifact has been mv'd into place the trap
    # must not fire again on a path that no longer belongs to this run.
    # (Measured: the inline form is exit-code-safe too -- `rm -rf -- ""` returns
    # 0, the -f swallows the empty operand -- so this is not fixing a bug in it.)
    [[ -n "${STAGING:-}" ]] || return 0
    rm -rf -- "$STAGING" 2>/dev/null || true
    STAGING=""
}
trap cleanup_staging EXIT
trap 'cleanup_staging; exit 130' INT TERM

assert_artifact_name_safe() {
    # assert_artifact_name_safe <var-name> <value>
    #
    # An artifact NAME is a single directory component and nothing else.  Two
    # demonstrated escapes are why this is a hard refusal and not a warning:
    #
    #   1. rm -rf traversal.  fetch_dir builds `.staging.$$.<name>` and then
    #      `rm -rf -- "$staging"`.  A name of `x/../../victim` makes that
    #      `rm -rf .../diagnostics/.staging.<pid>.x/../../victim`, which resolves
    #      OUT of the diagnostics tree.
    #   2. Remote command injection.  The remote probe is
    #      `ssh HOST "ls -ld -- '<path>'"`.  Single quotes protect nothing
    #      against a name that CONTAINS a single quote, and the remote shell
    #      re-parses the whole string.  (OpenSSH 8.9's `scp -r` re-parses
    #      remotely too, so the copy is a second injection point.)
    #
    # The class below has no `/`, no quote, no `$`, no whitespace, and cannot
    # start with `.`, so `..` is not expressible.
    local var="$1" value="$2"
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "$EXIT_PREFLIGHT" \
"$var is not a plain artifact directory name and is refused.
  got     : $value
  allowed : ^[A-Za-z0-9][A-Za-z0-9._-]*\$  (one path component: letters, digits, dot, dash, underscore)
  This is a name, not a path -- the remote and local directories are built from
  it.  Nothing was copied and nothing remote was contacted."
}

assert_remote_path_safe() {
    # /home/junhyeong/gello_software* is another person's work tree.  A typo'd
    # HIL_REMOTE_DATA_ROOT is the only way this script could reach it, so the
    # refusal lives on the path itself rather than in a comment.
    case "$1" in
        *gello_software*)
            die "$EXIT_PREFLIGHT" \
                "remote path names gello_software and is refused (that tree is not ours): $1"
            ;;
    esac
}

# --- preflight --------------------------------------------------------------

preflight() {
    step "preflight"
    [[ -x "$ACTOR_PYTHON" ]] || die "$EXIT_PREFLIGHT" \
        "actor venv python is missing or not executable: $ACTOR_PYTHON"
    say "interpreter        : $ACTOR_PYTHON ($("$ACTOR_PYTHON" -V 2>&1))"
    say "remote             : $SSH_HOST:$REMOTE_DATA_ROOT/diagnostics"
    say "local              : $LOCAL_DATA_ROOT/diagnostics"
    assert_remote_path_safe "$REMOTE_DATA_ROOT"
    # Before ssh, before scp, before any path is built out of them.  Both names
    # are attacker-controllable in the only sense that matters here: they are
    # environment overrides, and a typo'd one is indistinguishable from a
    # malicious one to everything downstream.
    assert_artifact_name_safe BC_ARTIFACT_NAME "$BC_ARTIFACT_NAME"
    assert_artifact_name_safe FM_ARTIFACT_NAME "$FM_ARTIFACT_NAME"
    say "artifact names     : validated (single path component, no shell metacharacters)"
    mkdir -p "$LOCAL_DATA_ROOT/diagnostics"
    # Sweep staging directories older than an hour.  These can only come from a
    # fetch that was killed hard enough to skip the trap (SIGKILL, power loss);
    # they are hidden, so nobody trips over them, and they are megabytes each on
    # a disk that has none to spare.  An hour is far longer than a ~56M scp, so
    # this can never race a fetch running in another terminal.
    local stale
    stale="$(find "$LOCAL_DATA_ROOT/diagnostics" -maxdepth 1 -name '.staging.*' -mmin +60 2>/dev/null | wc -l)"
    if (( stale > 0 )); then
        say "stale staging      : $stale abandoned .staging.* dir(s) older than 60 min -- removing"
        find "$LOCAL_DATA_ROOT/diagnostics" -maxdepth 1 -name '.staging.*' -mmin +60 \
            -exec rm -rf -- {} + 2>/dev/null || true
    fi
    local avail
    avail="$(df -h --output=avail "$LOCAL_DATA_ROOT" | tail -1 | tr -d ' ')"
    say "free space         : $avail (the two artifacts are ~56M together)"
}

# --- copy -------------------------------------------------------------------

fetch_dir() {
    # fetch_dir <artifact-name>.  Copies into a staging directory first so an
    # interrupted transfer can never be mistaken for a verified artifact.
    local name="$1"
    local remote_dir="$REMOTE_DATA_ROOT/diagnostics/$name"
    local dest="$LOCAL_DATA_ROOT/diagnostics/$name"
    local staging="$LOCAL_DATA_ROOT/diagnostics/.staging.$$.$name"

    assert_remote_path_safe "$remote_dir"
    # Belt and braces: preflight already validated the name, but fetch_dir is
    # the function that builds an rm -rf target and a remote command string out
    # of it, so it re-checks rather than trusting a caller.
    assert_artifact_name_safe "artifact name" "$name"

    say "remote check       : ssh $SSH_HOST ls -ld $remote_dir"
    ssh "$SSH_HOST" "ls -ld -- '$remote_dir'" \
        || die "$EXIT_COPY" "remote artifact directory is unreachable or missing: $remote_dir"

    rm -rf -- "$staging"
    # Published to the trap BEFORE the directory exists, not after: the window
    # this closes is a Ctrl-C during the scp, and mkdir is already inside it.
    STAGING="$staging"
    mkdir -p "$staging"
    say "copying            : scp -r (this is the only remote write-free transfer)"
    if ! scp -r -- "$SSH_HOST:$remote_dir" "$staging/"; then
        cleanup_staging
        die "$EXIT_COPY" "scp failed for $remote_dir"
    fi
    [[ -d "$staging/$name" ]] || {
        cleanup_staging
        die "$EXIT_COPY" "scp produced no directory named $name under $staging"
    }
    mv -- "$staging/$name" "$dest"
    cleanup_staging
    say "copied             : $dest"
}

# --- verification -----------------------------------------------------------

verify_bc() {
    # Returns 0 on success, 1 on any mismatch.  Never deletes.
    local dir="$1"
    local manifest="$dir/manifest.json"
    local ok=0

    [[ -f "$manifest" ]] || { say "  MISSING manifest.json in $dir"; return 1; }

    local param_file declared actual
    param_file="$(json_get "$manifest" parameter_file)" || return 1
    declared="$(json_get "$manifest" parameter_sha256)" || return 1
    [[ -f "$dir/$param_file" ]] || { say "  MISSING parameter file: $dir/$param_file"; return 1; }
    actual="$(sha256_of "$dir/$param_file")"

    say "  parameter file   : $param_file"
    say "  manifest sha256  : $declared"
    say "  sha256sum        : $actual"
    if [[ "$declared" == "$actual" ]]; then
        say "  digest           : MATCH"
    else
        say "  digest           : MISMATCH"
        ok=1
    fi

    if [[ "$BC_ARTIFACT_NAME" == "$DEFAULT_BC_ARTIFACT_NAME" ]]; then
        if [[ "$actual" == "$PIN_BC_PARAM_SHA256" ]]; then
            say "  plan pin         : MATCH (${PIN_BC_PARAM_SHA256:0:8}...)"
        else
            say "  plan pin         : MISMATCH -- expected ${PIN_BC_PARAM_SHA256:0:8}..., got ${actual:0:8}..."
            ok=1
        fi
    else
        say "  plan pin         : skipped (BC_ARTIFACT_NAME overridden)"
    fi

    # Bonus gate: the repo's own loader.  load_bc_init_manifest() re-checks the
    # format identity, the subtree list, parameter_bytes, the digest, and the
    # completion.json cross-checks (which prove the writer finished AND that
    # this manifest is the one it finished against); verify_resnet_asset()
    # requires our local trunk to be the trunk BC trained against.  Stdlib-only
    # by design, so the jax-free actor venv can run it.
    if PYTHONPATH="$REPO_ROOT/serl_ur_infra" "$ACTOR_PYTHON" - "$dir" "$RESNET_PATH" <<'PY'
import sys
from ur_env.learner.bc_init import load_bc_init_manifest, verify_resnet_asset

manifest = load_bc_init_manifest(sys.argv[1])
verify_resnet_asset(manifest, sys.argv[2])
print("  repo loader      : PASS (manifest identity, subtrees, size, digest, "
      "completion cross-check, resnet pin)")
PY
    then
        :
    else
        say "  repo loader      : FAIL (see the refusal above)"
        ok=1
    fi

    return "$ok"
}

verify_fm() {
    # The FM loader (learner/flow_matching.load_flow_artifact) imports jax, so
    # it cannot run in the actor venv.  These are exactly the checks it performs
    # before it touches an array, recomputed here with sha256sum.
    local dir="$1"
    local manifest="$dir/manifest.json"
    local completion="$dir/completion.json"
    local ok=0

    [[ -f "$manifest" ]]   || { say "  MISSING manifest.json in $dir"; return 1; }
    [[ -f "$completion" ]] || { say "  MISSING completion.json in $dir"; return 1; }

    local fmt
    fmt="$(json_get "$manifest" format)" || return 1
    if [[ "$fmt" == "hil-serl-jax-flow-matching" ]]; then
        say "  format           : $fmt"
    else
        say "  format           : UNEXPECTED ($fmt)"
        ok=1
    fi

    local complete
    complete="$(json_get "$completion" complete)" || return 1
    if [[ "$complete" == "True" ]]; then
        say "  completion       : complete=true"
    else
        say "  completion       : NOT COMPLETE (complete=$complete)"
        ok=1
    fi

    local declared_manifest_sha actual_manifest_sha
    declared_manifest_sha="$(json_get "$completion" manifest_sha256)" || return 1
    actual_manifest_sha="$(sha256_of "$manifest")"
    say "  manifest sha256  : declared=$declared_manifest_sha"
    say "                     sha256sum=$actual_manifest_sha"
    if [[ "$declared_manifest_sha" == "$actual_manifest_sha" ]]; then
        say "  manifest digest  : MATCH"
    else
        say "  manifest digest  : MISMATCH"
        ok=1
    fi

    local which path declared actual
    for which in best final; do
        path="$(json_get "$manifest" parameter_files "$which" path)" || return 1
        declared="$(json_get "$manifest" parameter_files "$which" sha256)" || return 1
        if [[ ! -f "$dir/$path" ]]; then
            say "  MISSING $which parameter file: $dir/$path"
            ok=1
            continue
        fi
        actual="$(sha256_of "$dir/$path")"
        printf '  %-16s : %s\n' "$which file" "$path"
        say "    manifest sha256: $declared"
        say "    sha256sum      : $actual"
        if [[ "$declared" == "$actual" ]]; then
            say "    digest         : MATCH"
        else
            say "    digest         : MISMATCH"
            ok=1
        fi
        if [[ "$FM_ARTIFACT_NAME" == "$DEFAULT_FM_ARTIFACT_NAME" ]]; then
            if [[ "$actual" == "$PIN_FM_PARAM_SHA256" ]]; then
                say "    plan pin       : MATCH (${PIN_FM_PARAM_SHA256:0:8}...)"
            else
                say "    plan pin       : MISMATCH -- expected ${PIN_FM_PARAM_SHA256:0:8}..., got ${actual:0:8}..."
                ok=1
            fi
        fi
    done

    # The trunk pin the FM manifest declares must be the trunk we hold, for the
    # same reason as BC: the served features come from OUR resnet, not theirs.
    local declared_resnet actual_resnet
    declared_resnet="$(json_get "$manifest" resnet_sha256)" || return 1
    actual_resnet="$(sha256_of "$RESNET_PATH")"
    if [[ "$declared_resnet" == "$actual_resnet" ]]; then
        say "  resnet pin       : MATCH (${actual_resnet:0:8}...)"
    else
        say "  resnet pin       : MISMATCH -- manifest=$declared_resnet local=$actual_resnet"
        ok=1
    fi

    return "$ok"
}

verify_resnet() {
    step "resnet10_params.pkl (local only -- verified, never copied)"
    say "  path             : $RESNET_PATH"
    if [[ ! -f "$RESNET_PATH" ]]; then
        say "  MISSING"
        add_row "resnet10_params.pkl" "-" "-" "MISSING"
        return 1
    fi
    local actual
    actual="$(sha256_of "$RESNET_PATH")"
    say "  expected sha256  : $PIN_RESNET_SHA256"
    say "  sha256sum        : $actual"
    if [[ "$actual" == "$PIN_RESNET_SHA256" ]]; then
        say "  digest           : MATCH"
        add_row "resnet10_params.pkl" "$(du -h -- "$RESNET_PATH" | cut -f1)" \
                "${actual:0:16}" "verified (local)"
        return 0
    fi
    say "  digest           : MISMATCH"
    add_row "resnet10_params.pkl" "$(du -h -- "$RESNET_PATH" | cut -f1)" \
            "${actual:0:16}" "MISMATCH"
    return 1
}

# --- per-artifact driver ----------------------------------------------------

ensure_artifact() {
    # ensure_artifact <label> <artifact-name> <verify-fn> <sha-field-fn>
    local label="$1" name="$2" verify_fn="$3" sha_fn="$4"
    local dest="$LOCAL_DATA_ROOT/diagnostics/$name"
    local status

    step "$label -- $name"

    if [[ -d "$dest" ]]; then
        say "local copy         : present"
        say "verifying          :"
        if "$verify_fn" "$dest"; then
            say "$label: already verified -- skipping copy"
            status="already verified"
        else
            die "$EXIT_VERIFY" \
"$label is present at $dest but FAILED verification (see above).
Nothing was deleted.  Inspect it, then either remove that directory and re-run
this script, or point BC_ARTIFACT_NAME/FM_ARTIFACT_NAME at the right artifact."
        fi
    else
        say "local copy         : absent -- fetching"
        fetch_dir "$name"
        say "verifying          :"
        if "$verify_fn" "$dest"; then
            status="verified (copied)"
        else
            die "$EXIT_VERIFY" \
"$label was copied to $dest but FAILED verification (see above).
Nothing was deleted -- the bad copy is left in place for inspection.  Remove it
by hand before re-running."
        fi
    fi

    add_row "$label" "$(human_size "$dest")" "$("$sha_fn" "$dest")" "$status"
}

bc_headline_sha() {
    local dir="$1" pf
    pf="$(json_get "$dir/manifest.json" parameter_file)"
    sha256_of "$dir/$pf" | cut -c1-16
}

fm_headline_sha() {
    local dir="$1" pf
    pf="$(json_get "$dir/manifest.json" parameter_files best path)"
    sha256_of "$dir/$pf" | cut -c1-16
}

# --- summary ----------------------------------------------------------------

print_summary() {
    step "summary"
    printf '%-22s %-7s %-18s %s\n' "ARTIFACT" "SIZE" "SHA256[0:16]" "STATUS"
    printf '%-22s %-7s %-18s %s\n' "----------------------" "-------" \
           "------------------" "------"
    local row
    for row in "${SUMMARY_ROWS[@]}"; do
        IFS='|' read -r a b c d <<<"$row"
        printf '%-22s %-7s %-18s %s\n' "$a" "$b" "$c" "$d"
    done
    say ""
    say "local root: $LOCAL_DATA_ROOT/diagnostics"
}

# --- main -------------------------------------------------------------------

main() {
    say "fetch_policy_artifacts.sh -- BC/FM artifact fetch + digest gate (G2)"
    preflight
    ensure_artifact "BC bc-init" "$BC_ARTIFACT_NAME" verify_bc bc_headline_sha
    ensure_artifact "FM fm-init" "$FM_ARTIFACT_NAME" verify_fm fm_headline_sha
    local resnet_ok=0
    verify_resnet || resnet_ok=1
    print_summary
    if [[ "$resnet_ok" -ne 0 ]]; then
        die "$EXIT_VERIFY" \
"the local ResNet-10 asset is not the one both manifests pin.
Nothing was deleted.  Every served feature comes from THIS file, so a
mismatch means the trained heads would be fed features they never saw."
    fi
    say ""
    say "G2 PASS: every declared digest verified."
}

main "$@"
