#!/usr/bin/env bash
# Staged DMC/GPU runner for the registered DMC gate + payoff study
# (research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md, incl. Amendment 1).
#
# Stages (run in order; stage2 refuses to start until the gate is recorded):
#   stage0 : env + trainer smoke on the GPU host (dm_control via shimmy)
#   stage1 : diagnostic gate - ordinary-arm pilot on GATE_SEEDS, then
#            udwm.scripts.dmc_gate_ratio prints teacher g*/w* + regime
#   stage2 : main comparison (seeds, five arms) seed-parallel over GPUs
#   stage3 : adjudication summary on CPU (summarize_corrected_policy)
#
# Env knobs:
#   TASK       shimmy DMC id, default dm_control/hopper-hop-v0
#   CONFIG     config for the task (create from delayed_bimodal_distill.yaml:
#              env.id = $TASK, max_episode_steps >= 500, gamma ~0.99)
#   STEPS      env steps per cell, default 3600 (payoff protocol)
#   GATE_SEEDS pilot seeds for the gate, default "0 1"
#   SEEDS      main-study seeds, default "0 1 2 ... 29"
#   GPU_IDS    comma list of CUDA devices, default "0,1"
#   OUT        merged output path
#   VARIANTS   main-study arms (registered set, default all five)
#
# Register first: the preregistration (and any amendment) is committed before
# compute. This script does not change endpoints/seeds/arms.
set -euo pipefail

TASK="${TASK:-dm_control/hopper-hop-v0}"
CONFIG="${CONFIG:-configs/dmc_hopper_distill.yaml}"
STEPS="${STEPS:-3600}"
GATE_SEEDS="${GATE_SEEDS:-0 1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29}"
GPU_IDS="${GPU_IDS:-0,1}"
OUT="${OUT:-runs/dmc_payoff_30seed_gpu.json}"
VARIANTS="${VARIANTS:-ordinary hybrid lagged_hybrid identified_hybrid identified_eq}"
PILOT="${PILOT:-runs/dmc_gate_pilot.json}"

stage0() {
  echo "[dmc:stage0] smoke: env space + one MBPO step on $TASK (device cuda)"
  python -c "
import torch, udwm.envs.registry as reg
assert torch.cuda.is_available(), 'CUDA unavailable'
e = reg.make_env('$TASK')
print('space:', reg.space_info(e))
print('smoke ok: env + spaces')
"
  echo "[dmc:stage0] next: create $CONFIG from configs/delayed_bimodal_distill.yaml"
  echo "  (env.id: $TASK, max_episode_steps >= 500, gamma ~0.99), then re-run stage1."
}

stage1() {
  echo "[dmc:stage1] gate pilot: ordinary arm only on seeds: ${GATE_SEEDS}"
  python -m udwm.scripts.run_policy_2x2_split_seeds \
    --config "$CONFIG" --seeds $GATE_SEEDS --variants ordinary \
    --steps "$STEPS" --jobs 2 --threads 2 --device cuda \
    --out "$PILOT"
  echo "[dmc:stage1] gate ratio (teacher g*/w* under the live critic):"
  python -m udwm.scripts.dmc_gate_ratio --data "$PILOT"
}

stage2() {
  if [ ! -f "$PILOT" ]; then
    echo "[dmc:stage2] refusing: gate pilot not found ($PILOT). Run stage1 first." >&2
    exit 1
  fi
  read -ra GPUS <<< "${GPU_IDS//,/ }"
  njobs="${#GPUS[@]}"
  echo "[dmc:stage2] main comparison: ${njobs} worker(s) on ${GPU_IDS}, seeds: ${SEEDS}"
  echo "  variants: ${VARIANTS} | steps: ${STEPS} | out: ${OUT}"
  python -m udwm.scripts.run_policy_2x2_split_seeds \
    --config "$CONFIG" --seeds $SEEDS --variants $VARIANTS \
    --steps "$STEPS" --jobs "$njobs" --threads 2 --gpu-ids "$GPU_IDS" \
    --out "$OUT"
  echo "[dmc:stage2] done: ${OUT}"
}

stage3() {
  echo "[dmc:stage3] adjudication summary (CPU):"
  python -m udwm.scripts.summarize_corrected_policy --data "$OUT"
}

case "${1:-all}" in
  stage0) stage0 ;;
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  all) stage0; stage1; stage2; stage3 ;;
  *) echo "usage: $0 [stage0|stage1|stage2|stage3|all]"; exit 1 ;;
esac
