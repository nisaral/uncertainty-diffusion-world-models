#!/usr/bin/env bash
# Kaggle one-shot runner: MACURA head-to-head (DMC @15k) + Walker2d gate/probe.
# Registrations: research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md
# and research/WALKER2D-PAYOFF-PREREGISTRATION-2026-09-09.md.
# Env knobs: FULL (1=also run the 30-seed MACURA adjudication, default),
# SEEDS, STEPS, GPU_IDS, JOBS. Resume-safe: re-running skips finished seeds.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

HOPPER=configs/dmc_hopper_probe.yaml
WALKER=configs/dmc_walker2d_distill.yaml
STEPS="${STEPS:-15000}"
FULL="${FULL:-1}"
GPU_IDS="${GPU_IDS:-0,1}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29}"
MACURA_PILOT=runs/dmc_macura_gate_pilot.json
MACURA_PROBE=runs/dmc_macura_probe_2seed_15k_gpu.json
MACURA_OUT=runs/dmc_macura_30seed_15k_gpu.json
WALKER_PILOT=runs/walker_gate_pilot.json
WALKER_PROBE=runs/walker2d_budget_probe_gpu.json

echo "[kaggle] removing stale wrong-budget files from earlier broken pastes"
rm -f runs/policy_identifiability_2x2_10seed.json runs/dmc_gate_pilot.json

echo "== [1/6] MACURA gate pilot (ordinary, seeds 0-1, ${STEPS}) =="
CONFIG=$HOPPER STEPS=$STEPS GATE_SEEDS="0 1" PILOT=$MACURA_PILOT GPU_IDS=$GPU_IDS bash dmc_payoff.sh stage1

echo "== [2/6] MACURA budget probe (ordinary identified_eq macura_gate, seeds 0-1) =="
python -m udwm.scripts.run_policy_2x2_split_seeds --config $HOPPER --seeds 0 1 --variants ordinary identified_eq macura_gate --steps $STEPS --jobs $JOBS --threads 2 --gpu-ids $GPU_IDS --out $MACURA_PROBE
python -m udwm.scripts.summarize_dmc_payoff --data $MACURA_PROBE --no-ctrl

echo "== [3/6] Walker2d gate pilot (ordinary, seeds 0-1) =="
CONFIG=$WALKER STEPS=$STEPS GATE_SEEDS="0 1" PILOT=$WALKER_PILOT GPU_IDS=$GPU_IDS bash dmc_payoff.sh stage1

echo "== [4/6] Walker2d budget probe (ordinary lagged_hybrid identified_eq) =="
python -m udwm.scripts.run_policy_2x2_split_seeds --config $WALKER --seeds 0 1 --variants ordinary lagged_hybrid identified_eq --steps $STEPS --jobs $JOBS --threads 2 --gpu-ids $GPU_IDS --out $WALKER_PROBE
python -m udwm.scripts.summarize_walker2d_payoff --data $WALKER_PROBE

if [ "$FULL" = "1" ]; then
  echo "== [5/6] MACURA 30-seed adjudication (LONG ~9-15h on 2xT4; re-run cell to resume) =="
  python -m udwm.scripts.run_policy_2x2_split_seeds --config $HOPPER --seeds $SEEDS --variants ordinary identified_eq macura_gate --steps $STEPS --jobs $JOBS --threads 2 --gpu-ids $GPU_IDS --out $MACURA_OUT
  echo "== [6/6] MACURA adjudication readout =="
  python -m udwm.scripts.summarize_dmc_payoff --data $MACURA_OUT --no-ctrl
else
  echo "== [5/6] skipped (FULL=0). Re-run with FULL=1 for the 30-seed run. =="
fi
echo "ALL STAGES DONE"