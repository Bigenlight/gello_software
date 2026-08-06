#!/usr/bin/env bash
# =============================================================================
# run_hil_local_policy.sh — laptop3 로컬 policy proxy 실행 래퍼
# =============================================================================
#
# ── 무엇을 띄우나 ────────────────────────────────────────────────────────────
#   serl_ur_infra/scripts/run_local_policy_proxy.py 하나.  127.0.0.1:50253에서
#   actor의 ActorTransport 프로토콜을 그대로 말하며, action은 laptop3 GPU에서
#   직접 뽑고(≈2 ms), 받은 요청 바이트는 배경 큐로 진짜 학습 서버
#   (터널의 로컬 끝 127.0.0.1:50153)에 순서대로 재생한다.
#   설계는 serl_ur_infra/HIL_LOCAL_INFERENCE_KO.md, 코드는
#   serl_ur_infra/ur_env/local_policy/proxy.py.
#
# ── 이 래퍼가 존재하는 이유 (전부 실패 경험에서 나온 것) ─────────────────────
#   1) **인터프리터.** 이 프로세스는 jax와 grpc를 한 인터프리터에서 같이 쓴다.
#      시스템 python3의 grpcio 1.30.2는 에러 없이 CPU 100%로 영구 정지하고,
#      actor venv에는 jax가 없다. 여기서는 항상
#      $LOCAL_POLICY_VENV/bin/python (기본 /home/laptop3/venvs/gello-local-policy)
#      절대경로로 실행하고, 그 안에서 jax가 실제로 import되는지까지 확인한다.
#   2) **터널이 먼저다.** proxy는 기동 중에 진짜 서버와 핸드셰이크하고
#      (protocol/schema/observation-hash + reward 필드 미러링) 그 뒤로 전이를
#      계속 보낸다. 터널이 없으면 몇십 초 뒤 핸드셰이크에서 죽는데, 그때쯤이면
#      조작자는 이미 다음 터미널로 넘어가 있다. 여기서 먼저 TCP로 확인한다.
#   3) **파라미터 export 디렉터리.** proxy는 <run_root>/params_live/LATEST.json을
#      폴링한다. 그 run_root는 **서버에서 지금 돌고 있는 learner의 것**이어야 하고,
#      그것을 아는 권위 소스는 run_hil_server.sh --check다 (아래 참조).
#   4) **RAM.** laptop3의 MemAvailable은 낮다(실측 ~1.3 GiB). 3 GiB 미만이면
#      경고한다 — 치명적으로 만들지는 않는다. jax 힙은 GPU에 있고, 부족하면
#      OOM으로 명확하게 죽지 조용히 틀리지 않는다.
#
# ── params 디렉터리 해석 순서 (먼저 맞는 것 하나만 쓴다) ─────────────────────
#   a) $HIL_PARAMS_REMOTE_DIR        — 조작자가 직접 지정. ssh를 아예 안 한다.
#   b) $HIL_PARAMS_RUN_ROOT          — 그 밑의 params_live.  역시 ssh 안 함.
#   c) `run_hil_server.sh --check`   — **권위 소스.** 그 출력의
#      HIL_SERVER_RUN_ROOT= 는 서버에서 **실제로 돌고 있는 learner 프로세스의
#      argv**(validate_process_contract)에서 나온 값이다. 추측이 아니다.
#   d) ssh ls fallback               — c)가 run root를 못 내놨을 때만.
#      `ls -td $HIL_REMOTE_DATA_ROOT/runs/*/params_live | head -1`.
#      **최신 디렉터리일 뿐 "지금 돌고 있는 learner의 것"이라는 보장이 없다** —
#      그래서 마지막이고, 쓰면 그렇다고 출력한다.
#   어느 경로를 썼는지는 항상 한 줄로 찍는다. 조용히 다른 lineage의 파라미터를
#   먹는 것이 이 설계에서 가장 알아채기 어려운 고장이다.
#
# ── 중단 ─────────────────────────────────────────────────────────────────────
#   Ctrl-C 한 번: proxy가 listener를 닫고 **업로드 큐를 비운다**(최대
#   --drain-timeout-s, 기본 60 s). 그때까지 기다려라 — 중간에 죽이면 아직
#   서버에 못 간 전이가 그대로 사라진다. 정말 버리려면 Ctrl-C를 한 번 더.
#   종료 코드는 proxy의 것을 그대로 전파한다 (0=정상, 3=기동 실패,
#   4=drain 미완료 = 전이 유실).
#
# ── 사용법 ───────────────────────────────────────────────────────────────────
#   ./run_hil_local_policy.sh                     # 단독 실행
#   ./run_hil_local_policy.sh --device cpu        # 인자는 그대로 proxy로 통과
#   HIL_POLICY_MODE=local ./run_hil_session.sh    # 정상 운용 (세션이 이걸 띄운다)
#
#   환경변수:
#     LOCAL_POLICY_VENV                venv 경로 (setup_local_policy_venv.sh와 같은 이름)
#     HIL_LOCAL_POLICY_PORT            proxy 포트 (기본 50253)
#     HIL_LOCAL_POLICY_REMOTE_TARGET   진짜 서버 host:port (기본 127.0.0.1:50153)
#     HIL_LOCAL_POLICY_DEVICE          auto|cpu|gpu — proxy가 직접 읽는다
#     HIL_PARAMS_REMOTE_DIR / HIL_PARAMS_RUN_ROOT   위 a)/b)
#     HIL_PARAMS_FETCH                 ssh|local (기본 ssh)
#     HIL_PARAMS_POLL_S                LATEST.json 폴링 주기 — proxy가 직접 읽는다
#     HIL_SSH_HOST                     기본 junhyeong_ai (run_hil_server.sh와 같은 기본값)
#     HIL_REMOTE_DATA_ROOT             기본 /home/junhyeong/hil-serl-data (동일)
#     HIL_LATENCY_PROFILE              설정돼 있으면 proxy/uploader/paramsync 세 role 계측
#     HIL_SERVER_SCRIPT                run_hil_server.sh 경로 (다른 checkout/테스트용)
#     HIL_LOCAL_POLICY_MIN_MEM_KB      RAM 경고 임계값 (기본 3145728 = 3 GiB)
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCAL_POLICY_VENV="${LOCAL_POLICY_VENV:-/home/laptop3/venvs/gello-local-policy}"
LOCAL_POLICY_PY="$LOCAL_POLICY_VENV/bin/python"
PROXY_SCRIPT="$REPO_ROOT/serl_ur_infra/scripts/run_local_policy_proxy.py"
SETUP_SCRIPT="$SCRIPT_DIR/setup_local_policy_venv.sh"
SERVER_SCRIPT="${HIL_SERVER_SCRIPT:-$SCRIPT_DIR/run_hil_server.sh}"

PROXY_HOST="127.0.0.1"
PROXY_PORT="${HIL_LOCAL_POLICY_PORT:-50253}"
REMOTE_TARGET="${HIL_LOCAL_POLICY_REMOTE_TARGET:-127.0.0.1:50153}"
SSH_HOST="${HIL_SSH_HOST:-junhyeong_ai}"
REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"
PARAMS_FETCH="${HIL_PARAMS_FETCH:-ssh}"
MIN_AVAILABLE_KB="${HIL_LOCAL_POLICY_MIN_MEM_KB:-3145728}"

PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' \
                "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            PASSTHRU+=("$arg")
            ;;
    esac
done

die() {
    echo "FATAL: $*" >&2
    exit 2
}

say() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# 설정값 검증 (fail-closed).  오타를 조용히 기본값으로 해석하지 않는다.
# ---------------------------------------------------------------------------
[[ "$PROXY_PORT" =~ ^[0-9]+$ ]] && (( PROXY_PORT >= 1 && PROXY_PORT <= 65535 )) || \
    die "HIL_LOCAL_POLICY_PORT must be a TCP port in [1, 65535] (got '$PROXY_PORT')"
case "$PARAMS_FETCH" in
    ssh|local) ;;
    *) die "HIL_PARAMS_FETCH must be 'ssh' or 'local' (got '$PARAMS_FETCH')" ;;
esac
REMOTE_HOST="${REMOTE_TARGET%:*}"
REMOTE_PORT="${REMOTE_TARGET##*:}"
[[ -n "$REMOTE_HOST" && "$REMOTE_PORT" =~ ^[0-9]+$ ]] || \
    die "HIL_LOCAL_POLICY_REMOTE_TARGET must be host:port (got '$REMOTE_TARGET')"
[[ "$MIN_AVAILABLE_KB" =~ ^[0-9]+$ ]] || \
    die "HIL_LOCAL_POLICY_MIN_MEM_KB must be a non-negative integer (got '$MIN_AVAILABLE_KB')"
[[ -f "$PROXY_SCRIPT" ]] || die "proxy entrypoint missing: $PROXY_SCRIPT"

say "============================================================"
say " HIL-SERL local policy proxy"
say "   repo root : $REPO_ROOT"
say "   venv      : $LOCAL_POLICY_VENV"
say "   listen    : $PROXY_HOST:$PROXY_PORT"
say "   forward to: $REMOTE_TARGET (진짜 학습 서버 — 터널의 로컬 끝)"
say "   latency   : ${HIL_LATENCY_PROFILE:-off}"
say "============================================================"

# ---------------------------------------------------------------------------
# [1] venv — 인터프리터가 있고 그 안에서 jax가 실제로 import되는가.
#     "설치돼 있다"가 아니라 "import된다"를 본다: CUDA plugin이 깨진 venv는
#     파일은 다 있고 import에서만 죽는다.
# ---------------------------------------------------------------------------
say ""
say "[1] local policy venv"
if [[ ! -x "$LOCAL_POLICY_PY" ]]; then
    echo "FATAL: local policy venv python이 없다: $LOCAL_POLICY_PY" >&2
    echo "       만들려면: $SETUP_SCRIPT" >&2
    echo "       (시스템 python3는 쓸 수 없다 — grpcio 1.30.2가 조용히 정지한다)" >&2
    exit 2
fi
set +e
JAX_PROBE="$(JAX_PLATFORMS=cpu timeout --signal=KILL 120 \
    "$LOCAL_POLICY_PY" -c 'import jax; print(jax.__version__)' 2>&1)"
JAX_RC=$?
set -e
if [[ "$JAX_RC" -ne 0 ]]; then
    echo "FATAL: $LOCAL_POLICY_PY 에서 jax를 import하지 못했다 (rc=$JAX_RC)" >&2
    printf '       %s\n' "${JAX_PROBE:-<no output>}" >&2
    echo "       고치려면: $SETUP_SCRIPT" >&2
    exit 2
fi
say "  [ OK ] $LOCAL_POLICY_PY (jax $(printf '%s' "$JAX_PROBE" | tail -n1))"

# ---------------------------------------------------------------------------
# [2] RAM — 경고만 한다 (치명적이지 않다).
# ---------------------------------------------------------------------------
say ""
say "[2] MemAvailable"
MEM_AVAILABLE_KB="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null || true)"
if [[ ! "$MEM_AVAILABLE_KB" =~ ^[0-9]+$ ]]; then
    say "  [WARN] /proc/meminfo에서 MemAvailable을 읽지 못했다 — 확인 없이 진행한다."
elif (( MEM_AVAILABLE_KB < MIN_AVAILABLE_KB )); then
    say "  [WARN] MemAvailable $(( MEM_AVAILABLE_KB / 1024 )) MiB < $(( MIN_AVAILABLE_KB / 1024 )) MiB"
    say "         proxy는 파라미터 트리(32 MB)를 여러 벌 들고 있고 gRPC 버퍼도 쓴다."
    say "         브라우저/뷰어를 닫으면 여유가 는다. 진행은 한다 — 부족하면 OOM으로"
    say "         명확하게 죽지 조용히 틀리지 않는다."
else
    say "  [ OK ] MemAvailable $(( MEM_AVAILABLE_KB / 1024 )) MiB"
fi

# ---------------------------------------------------------------------------
# TCP 도달성 probe (bash 내장 /dev/tcp).  loopback이라 연결은 즉시 성공하거나
# 즉시 거절되므로 별도 timeout이 필요 없다. python을 쓰지 않는 이유: 이 검사는
# venv가 무엇이든 성립해야 하고, 여기서 인터프리터를 한 번 더 띄우면 첫 실행에서
# 수 초가 그냥 사라진다.
# ---------------------------------------------------------------------------
tcp_open() {  # $1=host $2=port
    (exec 3<>"/dev/tcp/$1/$2") >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# [3] 학습 서버 터널 (Terminal 1이 연다)
# ---------------------------------------------------------------------------
say ""
say "[3] 학습 서버 터널 $REMOTE_TARGET"
if ! tcp_open "$REMOTE_HOST" "$REMOTE_PORT"; then
    echo "FATAL: $REMOTE_TARGET 에 연결할 수 없다 — 학습 서버 터널이 없다." >&2
    echo "       Terminal 1에서 ./run_hil_server.sh 를 먼저 실행하라." >&2
    echo "       (proxy는 기동 중에 그 서버와 핸드셰이크하고, 그 뒤 모든 전이를" >&2
    echo "        그 연결로 재생한다. 터널 없이 뜨면 아무것도 서버에 도달하지 않는다.)" >&2
    exit 2
fi
say "  [ OK ] TCP $REMOTE_TARGET 연결 성공"

# ---------------------------------------------------------------------------
# [4] proxy 포트가 비어 있는가.  이미 누가 듣고 있으면 어느 프로세스가 actor에게
#     action을 주는지 모호해진다 — 그 모호함은 절대 허용하지 않는다.
# ---------------------------------------------------------------------------
say ""
say "[4] proxy 포트 $PROXY_HOST:$PROXY_PORT"
if tcp_open "$PROXY_HOST" "$PROXY_PORT"; then
    echo "FATAL: $PROXY_HOST:$PROXY_PORT 에 이미 누군가 리스닝 중이다." >&2
    echo "       옛 proxy가 살아 있는지 확인하고 (pgrep -af run_local_policy_proxy.py)" >&2
    echo "       명시적으로 멈춘 뒤 다시 실행하라. 어느 프로세스가 actor에게 action을" >&2
    echo "       주는지 모호한 상태로는 시작하지 않는다." >&2
    exit 2
fi
say "  [ OK ] 비어 있다"

# ---------------------------------------------------------------------------
# [5] 파라미터 export 디렉터리 (파일 상단 "params 디렉터리 해석 순서")
# ---------------------------------------------------------------------------
say ""
say "[5] 파라미터 export 디렉터리"
PARAMS_DIR=""
PARAMS_SOURCE=""
if [[ -n "${HIL_PARAMS_REMOTE_DIR:-}" ]]; then
    PARAMS_DIR="${HIL_PARAMS_REMOTE_DIR%/}"
    PARAMS_SOURCE="\$HIL_PARAMS_REMOTE_DIR (ssh 조회 없음)"
elif [[ -n "${HIL_PARAMS_RUN_ROOT:-}" ]]; then
    PARAMS_DIR="${HIL_PARAMS_RUN_ROOT%/}/params_live"
    PARAMS_SOURCE="\$HIL_PARAMS_RUN_ROOT (ssh 조회 없음)"
elif [[ "$PARAMS_FETCH" == "local" ]]; then
    die "HIL_PARAMS_FETCH=local 에서는 HIL_PARAMS_REMOTE_DIR 또는 HIL_PARAMS_RUN_ROOT가 필요하다"
else
    if [[ -x "$SERVER_SCRIPT" ]]; then
        say "  .. run_hil_server.sh --check 로 돌고 있는 learner의 run root를 읽는다 (권위 소스)"
        set +e
        CHECK_OUT="$("$SERVER_SCRIPT" --check 2>&1)"
        CHECK_RC=$?
        set -e
        SERVER_RUN_ROOT="$(printf '%s\n' "$CHECK_OUT" \
            | sed -n 's/^HIL_SERVER_RUN_ROOT=//p' | tail -n1)"
        if [[ -n "$SERVER_RUN_ROOT" ]]; then
            PARAMS_DIR="${SERVER_RUN_ROOT%/}/params_live"
            PARAMS_SOURCE="run_hil_server.sh --check (rc=$CHECK_RC, 살아 있는 learner의 argv)"
        else
            say "  !! --check가 run root를 내놓지 않았다 (rc=$CHECK_RC). 마지막 몇 줄:"
            printf '%s\n' "$CHECK_OUT" | tail -n 5 | sed 's/^/     /'
        fi
    else
        say "  !! $SERVER_SCRIPT 를 실행할 수 없다 — 권위 소스를 건너뛴다."
    fi
    if [[ -z "$PARAMS_DIR" ]]; then
        say "  .. fallback: $SSH_HOST 에서 가장 최근 params_live 디렉터리를 찾는다"
        set +e
        LS_OUT="$(ssh -o BatchMode=yes "$SSH_HOST" \
            "ls -td '$REMOTE_DATA_ROOT'/runs/*/params_live 2>/dev/null | head -1" 2>/dev/null)"
        set -e
        PARAMS_DIR="$(printf '%s\n' "$LS_OUT" | tail -n1 | tr -d '\r')"
        PARAMS_DIR="${PARAMS_DIR%/}"
        PARAMS_SOURCE="ssh ls fallback — ⚠️ 최신 디렉터리일 뿐, 지금 돌고 있는 learner의 것이라는 보장은 없다"
    fi
fi
if [[ -z "$PARAMS_DIR" ]]; then
    echo "FATAL: 파라미터 export 디렉터리를 찾지 못했다." >&2
    echo "       T1을 HIL_PARAMS_EXPORT=1 로 띄웠는지 확인하고 (재사용된 learner에는" >&2
    echo "       환경변수가 안 먹는다 — HIL_SERVER_RESULT를 볼 것), 그래도 안 되면" >&2
    echo "       HIL_PARAMS_RUN_ROOT=<서버의 run root> 를 직접 지정하라." >&2
    exit 2
fi
say "  [ OK ] $PARAMS_DIR"
say "         출처: $PARAMS_SOURCE"

# ---------------------------------------------------------------------------
# 기동.  exec하지 않고 부모로 남는다: 신호를 한 번만 전달하고, drain이 끝날 때까지
# 기다렸다가 proxy의 종료 코드를 그대로 전파하기 위해서다.
#
# setsid는 **일부러 쓰지 않는다.** run_hil_session.sh는 이 래퍼를 자기 process
# group의 리더로 띄우고 정리할 때 그 group 전체에 신호를 보낸다. python 자식을
# 다른 session으로 빼면 그 정리가 자식에게 닿지 않아 proxy만 살아남는다.
# ---------------------------------------------------------------------------
PROXY_CMD=("$LOCAL_POLICY_PY" "$PROXY_SCRIPT"
    --host "$PROXY_HOST"
    --port "$PROXY_PORT"
    --remote-target "$REMOTE_TARGET"
    --params-remote-dir "$PARAMS_DIR"
    --params-fetch "$PARAMS_FETCH")
if [[ -n "${HIL_PARAMS_POLL_S:-}" ]]; then
    PROXY_CMD+=(--poll-interval-s "$HIL_PARAMS_POLL_S")
fi
if [[ -n "${HIL_LOCAL_POLICY_DEVICE:-}" ]]; then
    PROXY_CMD+=(--device "$HIL_LOCAL_POLICY_DEVICE")
fi
PROXY_CMD+=("${PASSTHRU[@]}")

say ""
say "proxy 기동 (Ctrl-C로 중단 — 큐를 비우는 동안 기다려라):"
printf '  '
printf '%q ' "${PROXY_CMD[@]}"
printf '\n'
say ""

PROXY_PID=""
SIGNALLED=0
forward_proxy_signal() {
    local signal_name="$1"
    if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        if (( SIGNALLED == 0 )); then
            SIGNALLED=1
            echo ""
            echo "[local-policy] $signal_name 전달 — 업로드 큐를 비우는 중이다."
            echo "               지금 한 번 더 중단하면 아직 서버에 못 간 전이가 사라진다."
        fi
        kill -s "$signal_name" "$PROXY_PID" 2>/dev/null || true
    fi
}
trap 'forward_proxy_signal INT' INT
trap 'forward_proxy_signal TERM' TERM
trap 'forward_proxy_signal HUP' HUP

"${PROXY_CMD[@]}" &
PROXY_PID=$!
set +e
# 신호에 걸린 wait는 자식이 실제로 끝나기 전에 돌아온다. 정말 사라질 때까지 다시 기다린다.
while true; do
    wait "$PROXY_PID"
    PROXY_RC=$?
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        break
    fi
done
set -e
PROXY_PID=""
trap - INT TERM HUP

say ""
case "$PROXY_RC" in
    0)  say "[local-policy] proxy 정상 종료 — 큐를 전부 비웠다." ;;
    3)  say "[local-policy] proxy가 기동에 실패했다 (rc=3). 위 로그를 볼 것." ;;
    4)  say "[local-policy] ⚠️ proxy가 큐를 다 비우지 못했다 (rc=4) — 일부 전이가 서버에 도달하지 못했다." ;;
    *)  say "[local-policy] proxy 종료 (rc=$PROXY_RC)" ;;
esac
exit "$PROXY_RC"
