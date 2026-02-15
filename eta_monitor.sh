#!/usr/bin/env bash
set -euo pipefail

# ====== 사용자 설정 ======
IN_DIR="app/input_sweep_r1"
LOG_ROOT="results/input_sweep_r/_logs"
WINDOW_N="${WINDOW_N:-8}"     # 최근 N개 완료 기준 rolling ETA (환경변수로 조정 가능)
# =========================

RUN_DIR=$(ls -1dt "${LOG_ROOT}"/run-* 2>/dev/null | head -n 1)
[[ -n "${RUN_DIR:-}" ]] || { echo "No run directory found under: ${LOG_ROOT}"; exit 1; }

TOTAL=$(ls "${IN_DIR}"/*.ini 2>/dev/null | wc -l | tr -d ' ')
OK=$(wc -l < "${RUN_DIR}/ok.list" 2>/dev/null | tr -d ' ' || echo 0)
FAIL=$(wc -l < "${RUN_DIR}/fail.list" 2>/dev/null | tr -d ' ' || echo 0)
DONE=$((OK + FAIL))

# Parse run-YYYYMMDD-HHMMSS
RAW=$(basename "${RUN_DIR}" | sed 's/run-//')
YEAR=${RAW:0:4}; MONTH=${RAW:4:2}; DAY=${RAW:6:2}
HOUR=${RAW:9:2}; MIN=${RAW:11:2}; SEC=${RAW:13:2}
START_EPOCH=$(date -d "${YEAR}-${MONTH}-${DAY} ${HOUR}:${MIN}:${SEC}" +%s)

NOW=$(date +%s)
ELAPSED=$((NOW - START_EPOCH))

echo "--------------------------------------"
echo "Run directory : ${RUN_DIR}"
echo "LOG_ROOT      : ${LOG_ROOT}"
echo "IN_DIR        : ${IN_DIR}"
echo "Progress      : ${DONE} / ${TOTAL} (OK ${OK}, FAIL ${FAIL})"
echo "Elapsed       : $((ELAPSED/60)) min"

# Guard
if [[ "${TOTAL}" -le 0 ]]; then
  echo "ETA           : N/A (TOTAL=0, IN_DIR wrong?)"
  echo "--------------------------------------"
  exit 2
fi
if [[ "${DONE}" -gt "${TOTAL}" ]]; then
  echo "ETA           : N/A (DONE > TOTAL, wrong IN_DIR for this run)"
  echo "--------------------------------------"
  exit 2
fi

# ---------- AVG ETA (global average) ----------
avg_eta_line="AVG ETA       : N/A"
avg_finish_line="AVG Finish    : N/A"

if [[ "${DONE}" -gt 0 && "${ELAPSED}" -gt 0 && "${DONE}" -lt "${TOTAL}" ]]; then
  RATE=$(echo "${DONE} / ${ELAPSED}" | bc -l)                # cases per sec
  REMAIN=$((TOTAL - DONE))
  ETA_SEC=$(echo "${REMAIN} / ${RATE}" | bc -l | cut -d. -f1)
  [[ "${ETA_SEC}" -lt 0 ]] && ETA_SEC=0

  avg_eta_line="AVG Remaining  : $((ETA_SEC/60)) min"
  avg_finish_line="AVG Finish    : $(date -d "+${ETA_SEC} seconds")"
elif [[ "${DONE}" -ge "${TOTAL}" ]]; then
  avg_eta_line="AVG Remaining  : 0 min"
  avg_finish_line="AVG Finish    : already completed"
fi

# ---------- ROLL ETA (recent WINDOW_N completions) ----------
roll_eta_line="ROLL Remaining : N/A"
roll_finish_line="ROLL Finish   : N/A"

if [[ "${DONE}" -ge 2 && "${DONE}" -lt "${TOTAL}" ]]; then
  # Read last WINDOW_N items from ok.list; each line is an ini path
  # Map: input_XXX.ini -> case dir XXX (strip input_) -> OUT_DIR/XXX/ConvergedData.txt
  # OUT_DIR is not guaranteed to be derivable from LOG_ROOT; infer from summary if possible,
  # else assume results dir is sibling of LOG_ROOT: results/input_sweep_r
  #
  # Practical approach: parse OUT_DIR from RUN_DIR/summary.txt when present.
  OUT_DIR=$(awk -F': ' '/^Output dir/{print $2}' "${RUN_DIR}/summary.txt" 2>/dev/null || true)
  if [[ -z "${OUT_DIR:-}" ]]; then
    # Fallback (common in your setup): LOG_ROOT = <OUT_DIR>/_logs
    OUT_DIR="${LOG_ROOT%/_logs}"
  fi

  # Collect mtime epochs for last WINDOW_N completed cases
  mapfile -t last_ini < <(tail -n "${WINDOW_N}" "${RUN_DIR}/ok.list" 2>/dev/null || true)

  times=()
  for ini in "${last_ini[@]}"; do
    braw="$(basename "${ini}" .ini)"          # input_RIMP9_...
    bcase="${braw#input_}"                    # RIMP9_...
    artifact="${OUT_DIR}/${bcase}/ConvergedData.txt"
    if [[ -f "${artifact}" ]]; then
      times+=( "$(stat -c %Y "${artifact}" 2>/dev/null || true)" )
    fi
  done

  # Need at least 2 timestamps to compute a rate
  if [[ "${#times[@]}" -ge 2 ]]; then
    # Sort epochs (just in case)
    IFS=$'\n' times_sorted=($(printf "%s\n" "${times[@]}" | sort -n)); unset IFS
    t_first="${times_sorted[0]}"
    t_last="${times_sorted[-1]}"
    dn="${#times_sorted[@]}"
    dt_s=$((t_last - t_first))

    if [[ "${dt_s}" -gt 0 ]]; then
      # rolling rate = (dn-1) completions over dt_s
      roll_rate=$(echo "(${dn}-1) / ${dt_s}" | bc -l)   # cases per sec
      remain=$((TOTAL - DONE))
      roll_eta_sec=$(echo "${remain} / ${roll_rate}" | bc -l | cut -d. -f1)
      [[ "${roll_eta_sec}" -lt 0 ]] && roll_eta_sec=0

      roll_eta_line="ROLL Remaining : $((roll_eta_sec/60)) min (window=${dn})"
      roll_finish_line="ROLL Finish   : $(date -d "+${roll_eta_sec} seconds")"
    fi
  fi
fi

echo "${avg_eta_line}"
echo "${avg_finish_line}"
echo "${roll_eta_line}"
echo "${roll_finish_line}"
echo "--------------------------------------"

