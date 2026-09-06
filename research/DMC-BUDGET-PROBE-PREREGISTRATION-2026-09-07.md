# DMC budget-calibration probe registration (2026-09-07)

**Status: registered before any probe row runs; diagnostic only - this probe
NEVER adjudicates the DMC payoff bars.** It exists to resolve the operating-
point confound left open by the 10-seed sanity
(`research/RESULTS-DMC-SANITY-2026-09-07.md`) before the 30-seed adjudication
is launched. Data: `runs/dmc_budget_probe_gpu.json` (to be written on the
Kaggle host). Runner: `udwm/scripts/run_policy_2x2_split_seeds` with
`configs/dmc_hopper_probe.yaml` (identical to the DMC config except
`eval_freq: 3000` to cap eval wall-time; final-eval metrics unchanged).

**Why this probe exists.** On DelayedBimodal every arm reaches u_rank
0.62-0.96; on the DMC 10-seed sanity every arm loses ~half its level
(ordinary 0.51, lagged_hybrid 0.57, eq 0.43, EMA 0.01). One reading: the
equal-weight identifiability benefit genuinely does not transfer to DMC.
Another: 3,600 env steps = 3.6 episodes of a 1,000-step task (vs 15-30
episodes for the DelayedBimodal studies), so the live critic and policy are
undertrained, returns are flat (~0.02-0.10 for every arm), and u_rank under
that critic is floor-compressed for *all* arms - making the eq-vs-hybrid
comparison noise-on-a-broken-baseline. The probe discriminates: if baseline
arms climb out of the ~0.5 band when the budget reaches episode-parity with
the DelayedBimodal studies (~15 episodes), the 3,600-step budget is the
confound and the registered 30-seed protocol must be amended (Amendment 2)
before it is run.

**Protocol (diagnostic).** Arms `ordinary`, `lagged_hybrid`, `identified_eq`
(the floor, the best ranker, and the candidate); seeds {0, 1} (same seeds as
the gate and sanity rows, so 3,600-step baselines are read from
`runs/dmc_payoff_10seed_gpu.json` without re-running them); budget 15,000 env
steps = 15 episodes, matching the DelayedBimodal corrected-weight study's
15-episode experience at 1,800 steps. Runner per seed, on 2 GPUs:

```bash
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --config configs/dmc_hopper_probe.yaml \
  --seeds 0 1 \
  --variants ordinary lagged_hybrid identified_eq \
  --steps 15000 --jobs 2 --threads 2 --gpu-ids 0,1 \
  --out runs/dmc_budget_probe_gpu.json
```

Rows written after 2026-09-07 carry `eval_history` (per-eval step,
return_mean, u_rank_corr, u_rmse, next_state_mse); reader
`udwm/scripts/print_eval_history.py` prints the curves.

**Endpoints (diagnostic, N=2 seeds - never quoted as adjudication).**
Per arm: `u_rank_corr`, `u_rmse`, `next_state_mse`, `final_return` at 15,000
steps (final eval, 10 episodes) and their in-training curves; paired with the
same-seed 3,600-step sanity rows where useful.

**Pre-committed decision rule (decided before rows exist).**
- If mean(seeds 0,1) u_rank at 15k is >= 0.70 for BOTH `ordinary` and
  `lagged_hybrid` (vs 0.51/0.57 at 3.6k) AND/OR paired final_return
  (15k - 3.6k) rises clearly on both seeds -> **budget confound supported**:
  amend the DMC protocol (Amendment 2, higher budget ~15k-30k, re-run gate +
  sanity + 30-seed under the amended protocol) before any 30-seed
  adjudication at 3,600 steps.
- If ordinary/lagged u_rank at 15k stays <= 0.60 and returns stay flat ->
  the 3.6k operating point is not primarily a budget artifact; run the
  registered 30-seed adjudication at 3,600 steps as written.
- Branch eq-only: if `identified_eq` climbs to >= 0.70 while baselines do
  not, report budget-dependent transfer of the identified arm (informative,
  not adjudicative at N=2).