#!/usr/bin/env bash
# Staged DMC/GPU runner for the registered DMC gate + payoff study
# (research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md, incl. Amendment 1).
#
# Stages (run in order; stage2 refuses to start until the gate is recorded):
#   stage0 : env + CUDA smoke on the GPU host (dm_control via shimmy)
#   stage1 : diagnostic gate - ordinary-arm pilot on GATE_SEEDS, then
#            udwm.scripts.dmc_gate_ratio prints teacher g*/w* + regime
#   stage2 : main comparison (SEEDS x VARIANTS) seed-parallel over GPU_IDS
#   stage3 : adjudication summary on the merged JSON
#
# Env knobs (defaults match the registered study + committed config):
#   TASK       shimmy DMC id, default dm_control/hopper-hop-v0
#   CONFIG     task config, default configs/dmc_hopper_distill.yaml (committed)
#   STEPS      env steps per seed, default 3600 (payoff protocol)
#   GATE_SEEDS gate pilot seeds (ordinary arm), default "0 1"
#   SEEDS      main-study seeds, default "0 1 ... 29"
#   GPU_IDS    comma list of CUDA devices, default "0,1" (2x T4 session);
#              single-GPU sessions MUST set GPU_IDS=0
#   OUT        merged output path, default runs/dmc_payoff_30seed_gpu.json
#   VARIANTS   main-study arms (registered set), default all five
#   PILOT      gate pilot output path, default runs/dmc_gate_pilot.json
#
# Resume semantics: per-seed partial files are written atomically per arm;
# re-running stage1/stage2 skips completed seeds, so a dropped Kaggle session
# is recoverable by re-running the same stage.
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

n_gpus() {
  local n=0 g
  for g in ${GPU_IDS//,/ }; do n=$((n + 1)); done
  echo "$n"
}

stage0() {
  echo "[dmc:stage0] smoke: env spaces on $TASK (CUDA)"
  if [ ! -f "$CONFIG" ]; then
    echo "[dmc:stage0] missing config: $CONFIG (committed config expected; clone fresh)." >&2
    exit 1
  fi
  python -c "
import torch, udwm.envs.registry as reg
assert torch.cuda.is_available(), 'CUDA unavailable (GPU session off?)'
e = reg.make_env('$TASK')
print('space:', reg.space_info(e))
print('smoke ok: env + spaces | cuda devices:', torch.cuda.device_count())
"
  echo "[dmc:stage0] ok. Next: stage1 (diagnostic gate - mandatory before stage2)."
}

stage1() {
  ng=$(n_gpus)
  nseeds=$(echo "$GATE_SEEDS" | wc -w)
  njobs=$(( ng < nseeds ? ng : nseeds ))
  if [ "$njobs" -lt 1 ]; then njobs=1; fi
  echo "[dmc:stage1] gate pilot: ordinary arm only, seeds: ${GATE_SEEDS} | jobs=${njobs} | gpus=${GPU_IDS}"
  python -m udwm.scripts.run_policy_2x2_split_seeds \
    --config "$CONFIG" --seeds $GATE_SEEDS --variants ordinary \
    --steps "$STEPS" --jobs "$njobs" --threads 2 --gpu-ids "$GPU_IDS" \
    --out "$PILOT"
  echo "[dmc:stage1] gate ratio (teacher g*/w* under the live critic):"
  python -m udwm.scripts.dmc_gate_ratio --data "$PILOT"
}

stage2() {
  if [ ! -f "$PILOT" ]; then
    echo "[dmc:stage2] refusing: gate pilot not found ($PILOT). Run stage1 first." >&2
    exit 1
  fi
  njobs=$(n_gpus)
  echo "[dmc:stage2] main comparison: ${njobs} worker(s) on ${GPU_IDS}, seeds: ${SEEDS}"
  echo "  variants: ${VARIANTS} | steps: ${STEPS} | out: ${OUT}"
  python -m udwm.scripts.run_policy_2x2_split_seeds \
    --config "$CONFIG" --seeds $SEEDS --variants $VARIANTS \
    --steps "$STEPS" --jobs "$njobs" --threads 2 --gpu-ids "$GPU_IDS" \
    --out "$OUT"
  echo "[dmc:stage2] done: ${OUT}"
}

stage3() {
  echo "[dmc:stage3] adjudication summary (CPU aggregation of the GPU rows):"
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