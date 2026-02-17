#!/usr/bin/env bash
set -euo pipefail

# ===== Cleanup on interrupt/termination =====
cleanup() {
  local _status=$?
  local _main_py
  if [[ "${_status}" -ne 0 ]]; then
    echo "[run_sweep] Caught signal, terminating child processes..." >&2
  fi
  _main_py="$(basename "$MAIN_PY")"
  if [[ -n "${ETA_PID:-}" ]]; then
    kill "${ETA_PID}" 2>/dev/null || true
  fi
  # Kill entire process group if possible
  if command -v pkill >/dev/null 2>&1; then
    pkill -P $$ 2>/dev/null || true
    pkill -f "python .*${_main_py}" 2>/dev/null || true
    pkill -f "chrome|chromium|kaleido" 2>/dev/null || true
  fi
  # Hard kill leftover python/chrome if still alive
  if command -v pkill >/dev/null 2>&1; then
    pkill -9 -P $$ 2>/dev/null || true
    pkill -9 -f "python .*${_main_py}" 2>/dev/null || true
    pkill -9 -f "chrome|chromium|kaleido" 2>/dev/null || true
  fi
  return "${_status}"
}
trap cleanup INT TERM EXIT
# ===========================================

# =========================
# Defaults (override via flags/env)
# =========================
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="${MAIN_PY:-app/main.py}"

IN_DIR="${IN_DIR:-app/input_sweep_r}"
OUT_DIR="${OUT_DIR:-results/input_sweep_r}"

MODE="${MODE:-seq}"          # seq | par
JOBS="${JOBS:-4}"            # parallel workers when MODE=par
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"  # 0 = no timeout (requires coreutils timeout)

LOG_ROOT="${LOG_ROOT:-results/_logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/run-${STAMP}}"

OK_LIST="${OK_LIST:-${RUN_DIR}/ok.list}"
FAIL_LIST="${FAIL_LIST:-${RUN_DIR}/fail.list}"
SUMMARY_TXT="${SUMMARY_TXT:-${RUN_DIR}/summary.txt}"

# ETA monitor
ETA="${ETA:-1}"                    # 1=enable, 0=disable
ETA_INTERVAL_SEC="${ETA_INTERVAL_SEC:-30}"  # refresh interval (seconds)

# =========================
# Helpers
# =========================
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --mode seq|par        Execution mode (default: ${MODE})
  -j, --jobs N          Parallel jobs when --mode par (default: ${JOBS})
  --in-dir DIR          Input directory containing *.ini (default: ${IN_DIR})
  --out-dir DIR         Output directory passed to --outdir (default: ${OUT_DIR})
  --python PATH         Python executable (default: ${PYTHON_BIN})
  --main PATH           main.py path (default: ${MAIN_PY})
  --timeout SEC         Per-case timeout seconds (0 = disabled, default: ${TIMEOUT_SEC})
  --resume              Only run cases NOT in ok.list from the latest run directory under <out-dir>/_logs
  --resume-from-out DIR  Resume using latest run under DIR/_logs (DIR is a previous --out-dir)
  --resume-run-dir DIR   Resume using this explicit run directory (e.g., .../_logs/run-YYYYMMDD-HHMMSS)
  --resume-ok-list FILE  Resume using this explicit ok.list path
                         Precedence: --resume-ok-list > --resume-run-dir > --resume-from-out > --out-dir
  --eta 0|1             Print live progress + ETA (default: ${ETA})
  --eta-interval SEC    ETA refresh interval seconds (default: ${ETA_INTERVAL_SEC})
  -h, --help            Show this help

Env overrides:
  PYTHON_BIN, MAIN_PY, IN_DIR, OUT_DIR, MODE, JOBS, TIMEOUT_SEC, LOG_ROOT, ETA, ETA_INTERVAL_SEC
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

format_hms() {
  # args: seconds (int)
  local t="${1:-0}"
  local h=$(( t / 3600 ))
  local m=$(( (t % 3600) / 60 ))
  local s=$(( t % 60 ))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

eta_monitor() {
  # args: total_cases start_ts
  local total="$1"
  local start_ts="$2"
  local last_done=-1

  while true; do
    # counts (files always exist but guard anyway)
    local ok=0 fail=0 done=0
    [[ -f "$OK_LIST" ]] && ok="$(wc -l < "$OK_LIST" | tr -d ' ')"
    [[ -f "$FAIL_LIST" ]] && fail="$(wc -l < "$FAIL_LIST" | tr -d ' ')"
    done=$(( ok + fail ))

    local now elapsed rate rem eta_sec eta_hms el_hms fin_ts fin_str
    now="$(date +%s)"
    elapsed=$(( now - start_ts ))
    rem=$(( total - done ))

    # Only print if progress changed or every interval (simpler: always print)
    if [[ "$elapsed" -gt 0 && "$done" -gt 0 ]]; then
      # rate: cases per second (floating via awk)
      rate="$(awk -v d="$done" -v e="$elapsed" 'BEGIN{printf "%.6f", d/e}')"
      eta_sec="$(awk -v r="$rate" -v rem="$rem" 'BEGIN{ if (r<=0) print -1; else printf "%d", rem/r }')"
    else
      rate="0"
      eta_sec="-1"
    fi

    el_hms="$(format_hms "$elapsed")"
    if [[ "$eta_sec" -ge 0 ]]; then
      eta_hms="$(format_hms "$eta_sec")"
      fin_ts=$(( now + eta_sec ))
      fin_str="$(date -d "@$fin_ts" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date "+%Y-%m-%d %H:%M:%S")"
    else
      eta_hms="--:--:--"
      fin_str="N/A"
    fi

    # Progress line (stderr). Use carriage return for in-place update.
    printf "\r[ETA] done %d/%d (OK %d, FAIL %d) | elapsed %s | ETA %s | finish %s" \
      "$done" "$total" "$ok" "$fail" "$el_hms" "$eta_hms" "$fin_str" >&2

    if [[ "$done" -ge "$total" ]]; then
      echo >&2
      break
    fi

    sleep "$ETA_INTERVAL_SEC"
  done
}
# Per-case runner: expects one argument = ini path
run_one() {
  local f="$1"
  local base_raw base_case log rc
  base_raw="$(basename "$f" .ini)"      # input_RIMP9_260213-1350
  base_case="${base_raw#input_}"        # RIMP9_260213-1350 (prefix removed)
  log="${RUN_DIR}/${base_raw}.log"

  # Success artifact
  local artifact="${OUT_DIR}/${base_case}/ConvergedData.txt"

  mkdir -p "${OUT_DIR}" "${RUN_DIR}"


  # Run the command (capture rc but decide success by artifact)
  set +e

  # Force single-threaded numerics inside each worker (avoid oversubscription)
  local -a PYENV=(
    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=1
    OMP_DYNAMIC=FALSE
    OMP_MAX_ACTIVE_LEVELS=1
    MKL_NUM_THREADS=1
    MKL_DYNAMIC=FALSE
    NUMEXPR_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1
    BLIS_NUM_THREADS=1
  )

  if [[ "$TIMEOUT_SEC" -gt 0 ]]; then
    has_cmd timeout || die "timeout not found but --timeout was set."
    timeout --preserve-status "${TIMEOUT_SEC}" \
      env "${PYENV[@]}" \
      "${PYTHON_BIN}" "${MAIN_PY}" -i "${f}" --outdir "${OUT_DIR}" >"${log}" 2>&1
  else
    env "${PYENV[@]}" \
      "${PYTHON_BIN}" "${MAIN_PY}" -i "${f}" --outdir "${OUT_DIR}" >"${log}" 2>&1
  fi

  rc=$?
  set -e



  # Success criterion: artifact exists (non-empty). Some runs can finish sizing,
  # write ConvergedData.txt, then fail later in plotting/GUI cleanup.
  if [[ -s "${artifact}" ]]; then
    echo "$(basename "$f")" >> "${OK_LIST}"
    if [[ "$rc" -ne 0 ]]; then
      {
        echo
        echo "[run_sweep] NOTE: command exited with ${rc}, but artifact exists. Treating as OK."
        echo "[run_sweep] artifact: ${artifact}"
      } >> "${log}"
    fi
    return 0
  else
    echo "$(basename "$f")" >> "${FAIL_LIST}"
    {
      echo
      echo "[run_sweep] FAIL criteria: artifact not found or empty"
      if [[ "$rc" -ne 0 ]]; then
        echo "[run_sweep] command exit: ${rc}"
      fi
      echo "[run_sweep] expected artifact: ${artifact}"
      echo "[run_sweep] exit code: ${rc}"
    } >> "${log}"
    return 1
  fi
}


# =========================
# Parse args
# =========================
RESUME=0
RESUME_FROM_OUT=""
RESUME_RUN_DIR=""
RESUME_OK_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    -j|--jobs) JOBS="$2"; shift 2;;
    --in-dir) IN_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --main) MAIN_PY="$2"; shift 2;;
    --timeout) TIMEOUT_SEC="$2"; shift 2;;
    --resume) RESUME=1; shift 1;;
    --resume-from-out) RESUME=1; RESUME_FROM_OUT="$2"; shift 2;;
    --resume-run-dir) RESUME=1; RESUME_RUN_DIR="$2"; shift 2;;
    --resume-ok-list) RESUME=1; RESUME_OK_LIST="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done

[[ "$JOBS" =~ ^[0-9]+$ ]] || die "Invalid jobs value: $JOBS"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "Invalid timeout value: $TIMEOUT_SEC"
[[ "$ETA" =~ ^[01]$ ]] || die "Invalid eta value: $ETA"
[[ "$ETA_INTERVAL_SEC" =~ ^[0-9]+$ ]] || die "Invalid eta interval value: $ETA_INTERVAL_SEC"
[[ "$JOBS" -gt 0 ]] || die "Jobs must be >= 1"
[[ "$ETA_INTERVAL_SEC" -gt 0 ]] || die "ETA interval must be >= 1"

[[ -f "$MAIN_PY" ]] || die "main.py not found: $MAIN_PY"
[[ -d "$IN_DIR" ]] || die "input dir not found: $IN_DIR"
mkdir -p "$OUT_DIR"

if [[ $RESUME -eq 0 ]]; then
  mkdir -p "$RUN_DIR"
  : > "$OK_LIST"
  : > "$FAIL_LIST"
fi


# Collect inputs (stable order)
shopt -s nullglob
inputs=( "${IN_DIR}"/*.ini )
shopt -u nullglob
[[ ${#inputs[@]} -gt 0 ]] || die "No .ini files found in: $IN_DIR"

# Resume logic (optional): skip already OK from latest run (flexible sources)
if [[ $RESUME -eq 1 ]]; then

  # ----- Determine reference ok.list only -----
  if [[ -n "${RESUME_OK_LIST}" ]]; then
      RESUME_OK_REF="${RESUME_OK_LIST}"
  elif [[ -n "${RESUME_RUN_DIR}" ]]; then
      RESUME_OK_REF="${RESUME_RUN_DIR}/ok.list"
  else
      RESUME_LOG_ROOT="${LOG_ROOT}"
      if [[ -n "${RESUME_FROM_OUT}" ]]; then
          RESUME_LOG_ROOT="${RESUME_FROM_OUT}/_logs"
      fi
      latest="$(find "${RESUME_LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'run-*' -print | sort -r | head -n 1 || true)"
      [[ -n "${latest:-}" ]] || die "--resume requested but no previous run-* found"
      RESUME_OK_REF="${latest}/ok.list"
  fi

  [[ -f "$RESUME_OK_REF" ]] || die "Previous ok.list not found: $RESUME_OK_REF"

  # ----- Always create NEW run directory -----
  mkdir -p "$RUN_DIR"
  : > "$OK_LIST"
  : > "$FAIL_LIST"

  declare -A okset
  while IFS= read -r line; do
      b="$(basename "$line")"
      okset["$b"]=1
  done < "$RESUME_OK_REF"

  filtered=()
  for f in "${inputs[@]}"; do
      b="$(basename "$f")"
      if [[ -z "${okset[$b]+x}" ]]; then
          filtered+=("$f")
      fi
  done

  inputs=("${filtered[@]}")
  echo "[RESUME] Reference OK list: $RESUME_OK_REF"
  echo "[RESUME] Remaining cases: ${#inputs[@]}"
  [[ ${#inputs[@]} -gt 0 ]] || { echo "Nothing to run."; exit 0; }

fi


echo "MODE      : $MODE"
echo "JOBS      : $JOBS"
echo "IN_DIR    : $IN_DIR"
echo "OUT_DIR   : $OUT_DIR"
echo "MAIN      : $MAIN_PY"
echo "PYTHON    : $PYTHON_BIN"
echo "TIMEOUT   : $TIMEOUT_SEC"
echo "RUN_DIR   : $RUN_DIR"
echo "N_CASES   : ${#inputs[@]}"
echo

start_ts="$(date +%s)"

# Start ETA monitor (stderr)
TOTAL_CASES="${#inputs[@]}"
ETA_PID=""
if [[ "${ETA}" == "1" ]]; then
  eta_monitor "${TOTAL_CASES}" "${start_ts}" &
  ETA_PID=$!
fi

# =========================
# Execute
# =========================
if [[ "$MODE" == "seq" ]]; then
  for f in "${inputs[@]}"; do
    echo "[RUN] $(basename "$f")"
    if run_one "$f"; then
      echo "  -> OK"
    else
      echo "  -> FAIL (see log in $RUN_DIR)" >&2
    fi
  done

elif [[ "$MODE" == "par" ]]; then
  has_cmd parallel || die "GNU parallel not found."
  # Export everything needed by subshell
  export PYTHON_BIN MAIN_PY IN_DIR OUT_DIR RUN_DIR OK_LIST FAIL_LIST TIMEOUT_SEC
  export -f run_one die has_cmd


  printf '%s\0' "${inputs[@]}" \
    | parallel -0 -j "$JOBS" --linebuffer '
        f={}
        echo "[RUN] $(basename "$f")"
        run_one "$f" && echo "  -> OK" || { echo "  -> FAIL (see log in '"$RUN_DIR"')" >&2; exit 0; }
      '
else
  die "Invalid MODE: $MODE (use seq|par)"
fi

end_ts="$(date +%s)"
elapsed=$(( end_ts - start_ts ))

ok_n="$(wc -l < "$OK_LIST" | tr -d ' ')"
fail_n="$(wc -l < "$FAIL_LIST" | tr -d ' ')"

{
  echo "Run directory : $RUN_DIR"
  echo "Input dir     : $IN_DIR"
  echo "Output dir    : $OUT_DIR"
  echo "Mode          : $MODE"
  echo "Jobs          : $JOBS"
  echo "Timeout(s)    : $TIMEOUT_SEC"
  echo "Total cases   : $(( ok_n + fail_n ))"
  echo "OK            : $ok_n"
  echo "FAIL          : $fail_n"
  echo "Elapsed(s)    : $elapsed"
  echo
  echo "OK list   : $OK_LIST"
  echo "FAIL list : $FAIL_LIST"
} | tee "$SUMMARY_TXT"

if [[ "$fail_n" -gt 0 ]]; then
  echo
  echo "Some cases failed. Inspect logs in: $RUN_DIR"
  exit 3
fi
