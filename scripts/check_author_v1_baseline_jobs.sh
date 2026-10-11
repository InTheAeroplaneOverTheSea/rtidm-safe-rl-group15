#!/bin/bash

# Read-only scheduler and result check; safe to run on a login node.
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/rtidm-safe-rl-group15}"
RECORD="${PROJECT_DIR}/author_v1_all_baselines_evaluation/submitted_jobs.tsv"

test -f "${RECORD}"
mapfile -t JOB_IDS < <(tail -n +2 "${RECORD}" | cut -f6 | sort -u)
if (( ${#JOB_IDS[@]} == 0 )); then
    echo "No submitted jobs recorded in ${RECORD}"
    exit 0
fi

job_csv=$(IFS=,; echo "${JOB_IDS[*]}")
echo "Checked at: $(date)"
echo
echo "=== Recorded submissions ==="
column -t -s $'\t' "${RECORD}"
echo
echo "=== Queue ==="
squeue -j "${job_csv}" -o '%.18i %.24j %.2t %.10M %.6D %R' || true
echo
echo "=== Accounting ==="
sacct -X -j "${job_csv}" \
    --format=JobID,JobName,State,Elapsed,ExitCode \
    --noheader
echo
echo "=== Completed result counts ==="
for algorithm in CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag; do
    baseline_count=$(find "${PROJECT_DIR}/author_v1_all_baselines_evaluation/no_shield/t1d/${algorithm}" \
        -type f -name summary_results.csv 2>/dev/null | wc -l | tr -d ' ')
    shield_count=$(find "${PROJECT_DIR}/author_v1_all_baselines_evaluation/predictive_penalty10/t1d/${algorithm}" \
        -type f -name summary_results.csv 2>/dev/null | wc -l | tr -d ' ')
    printf '%-8s baseline=%2d/81 shield=%2d/81\n' \
        "${algorithm}" "${baseline_count}" "${shield_count}"
done
