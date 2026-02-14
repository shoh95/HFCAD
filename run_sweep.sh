#!/usr/bin/env bash
set -euo pipefail

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

LOG_ROOT="${LOG_ROOT:-${OUT_DIR}/_logs}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/run-${STAMP}}"

OK_LIST="${OK_LIST:-${RUN_DIR}/ok.list}"
FAIL_LIST="${FAIL_LIST:-${RUN_DIR}/fail.list}"
SUMMARY_TXT="${SUMMARY_TXT:-${RUN_DIR}/summary.txt}"

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
  --resume              Only run cases NOT in ok.list from the latest run directory
  -h, --help            Show this help

Env overrides:
  PYTHON_BIN, MAIN_PY, IN_DIR, OUT_DIR, MODE, JOBS, TIMEOUT_SEC, LOG_ROOT
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

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
  if [[ "$TIMEOUT_SEC" -gt 0 ]]; then
    has_cmd timeout || die "timeout not found but --timeout was set."
    timeout --preserve-status "${TIMEOUT_SEC}" \
      "${PYTHON_BIN}" "${MAIN_PY}" -i "${f}" --outdir "${OUT_DIR}" >"${log}" 2>&1
  else
    "${PYTHON_BIN}" "${MAIN_PY}" -i "${f}" --outdir "${OUT_DIR}" >"${log}" 2>&1
  fi
  rc=$?
  set -e

  # Success criterion: artifact existence (non-empty recommended)
  if [[ -s "${artifact}" ]]; then
    echo "${f}" >> "${OK_LIST}"
    return 0
  else
    echo "${f}" >> "${FAIL_LIST}"
    {
      echo
      echo "[run_sweep] FAIL criteria: artifact not found or empty"
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
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done

[[ -f "$MAIN_PY" ]] || die "main.py not found: $MAIN_PY"
[[ -d "$IN_DIR" ]] || die "input dir not found: $IN_DIR"
mkdir -p "$OUT_DIR" "$RUN_DIR"
: > "$OK_LIST"
: > "$FAIL_LIST"

# Collect inputs (stable order)
shopt -s nullglob
mapfile -t inputs < <(ls -1 "${IN_DIR}"/*.ini 2>/dev/null || true)
[[ ${#inputs[@]} -gt 0 ]] || die "No .ini files found in: $IN_DIR"

# Resume logic (optional): skip already OK from latest run
if [[ $RESUME -eq 1 ]]; then
  # Find latest run directory under LOG_ROOT
  latest="$(ls -1dt "${LOG_ROOT}"/run-* 2>/dev/null | head -n 1 || true)"
  [[ -n "${latest:-}" ]] || die "--resume requested but no previous run-* directory found under ${LOG_ROOT}"
  ok_prev="${latest}/ok.list"
  [[ -f "$ok_prev" ]] || die "Previous ok.list not found: $ok_prev"

  # Build skip set
  declare -A okset
  while IFS= read -r line; do
    okset["$line"]=1
  done < "$ok_prev"

  # Filter inputs
  filtered=()
  for f in "${inputs[@]}"; do
    if [[ -z "${okset[$f]+x}" ]]; then
      filtered+=("$f")
    fi
  done
  inputs=("${filtered[@]}")
  echo "[RESUME] Skipping cases already OK in: $ok_prev"
  echo "[RESUME] Remaining cases: ${#inputs[@]}"
  [[ ${#inputs[@]} -gt 0 ]] || { echo "Nothing to run. Exiting."; exit 0; }
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
  has_cmd xargs || die "xargs not found."
  # Export everything needed by subshell
  export PYTHON_BIN MAIN_PY IN_DIR OUT_DIR RUN_DIR OK_LIST FAIL_LIST TIMEOUT_SEC
  export -f run_one die has_cmd

  printf '%s\0' "${inputs[@]}" \
    | xargs -0 -n 1 -P "$JOBS" bash -lc '
        f="$1"
        echo "[RUN] $(basename "$f")"
        run_one "$f" && echo "  -> OK" || { echo "  -> FAIL (see log in '"$RUN_DIR"')" >&2; exit 0; }
      ' _

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

