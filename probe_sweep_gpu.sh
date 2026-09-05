#!/usr/bin/env bash
# Seed-parallel GPU probe sweep (one worker per GPU).
# Env: VARIANT (default identified_hybrid), SEEDS, STEPS, GPU_IDS, plus optional
# SET (comma-separated --set model overrides) and STUDENT_LR.
# Usage: VARIANT=identified_hybrid SEEDS="0 1 2 3" GPU_IDS="0 1" ./probe_sweep_gpu.sh
set -euo pipefail

VARIANT="${VARIANT:-identified_hybrid}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-1800}"
GPU_IDS="${GPU_IDS:-0}"
SET="${SET:-}"
STUDENT_LR="${STUDENT_LR:-}"

read -ra GPUS <<< "${GPU_IDS//,/ }"
njobs="${#GPUS[@]}"
[ "$njobs" -ge 1 ] || { echo "GPU_IDS empty"; exit 1; }

run_arm() {
  local seed="$1" gpu="$2" out="runs/probe_u_gpu_${VARIANT}_s${seed}.json"
  local args=(--variant "$VARIANT" --seed "$seed" --steps "$STEPS" --device cuda --out "$out")
  [ -n "$SET" ] && args+=(--set "$SET")
  [ -n "$STUDENT_LR" ] && args+=(--student-lr "$STUDENT_LR")
  echo "[sweep] seed $seed -> cuda:$gpu -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python -m udwm.scripts.probe_u_collapse "${args[@]}"
}

pids=()
for seed in $SEEDS; do
  gpu="${GPUS[$(( seed % njobs ))]}"
  run_arm "$seed" "$gpu" &
  pids+=("$!")
  while [ "${#pids[@]}" -ge "$njobs" ]; do
    wait -n || true
    pids=($(jobs -pr))
  done
done
wait
echo "[sweep] done: runs/probe_u_gpu_${VARIANT}_s*.json"