#!/usr/bin/env bash
# Seed-parallel policy sweep for the corrected-weight arms (2026-09-05).
# Wraps udwm/scripts/run_policy_2x2_split_seeds.py: one worker per GPU/seed,
# each running every requested variant for its seed with the exact per-seed
# teacher pairing (CUDA_VISIBLE_DEVICES pinning handled by the driver).
#
# The corrected arms (see research/RESULTS-CORRUPTION-2026-09-05.md):
#   identified_eq     - equal-weight identified (recovers g/u-rank at probe)
#   identified_wonly  - w-only EMA (collapse control)
#   identified_hybrid - EMA-both (historical config; collapse control)
# Use ordinary/hybrid/lagged_hybrid as the non-identified baselines.
#
# Register first: any N-study verdict needs its endpoints/seeds/bar in
# research/ before reading rows (per repo discipline). This script is the
# runner; the DMC out-of-sample study is pre-registered in
# research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md.
#
# Usage (GPU): SEEDS="0 1 2 3" GPU_IDS="0 1" ./policy_sweep_gpu.sh
# Usage (CPU, parallel): SEEDS="0 .. 9" GPU_IDS="" JOBS=10 ./policy_sweep_gpu.sh
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
VARIANTS="${VARIANTS:-ordinary hybrid lagged_hybrid identified_hybrid identified_eq identified_wonly}"
STEPS="${STEPS:-1800}"
GPU_IDS="${GPU_IDS:-}"
JOBS="${JOBS:-}"
THREADS="${THREADS:-2}"
OUT="${OUT:-runs/policy_corrected_weights.json}"

args=(--seeds $SEEDS --variants $VARIANTS --steps "$STEPS" --out "$OUT")
if [ -n "$GPU_IDS" ]; then
  read -ra GPUS <<< "${GPU_IDS//,/ }"
  njobs="${#GPUS[@]}"
  args+=(--gpu-ids "$GPU_IDS" --jobs "$njobs")
  echo "[policy-sweep] GPU mode: ${njobs} worker(s) on ${GPU_IDS}, seeds: ${SEEDS}"
else
  if [ -z "$JOBS" ]; then JOBS=4; fi
  args+=(--jobs "$JOBS" --threads "$THREADS")
  echo "[policy-sweep] CPU mode: ${JOBS} parallel worker(s), seeds: ${SEEDS}"
fi
echo "[policy-sweep] variants: ${VARIANTS} | steps: ${STEPS} | out: ${OUT}"
python -m udwm.scripts.run_policy_2x2_split_seeds "${args[@]}"
echo "[policy-sweep] done: ${OUT}"
