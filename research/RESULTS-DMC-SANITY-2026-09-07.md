# DMC 10-seed sanity + gate record (hopper-hop, 2026-09-07)

**Status: sanity subset, NOT adjudication.** Registration:
`research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md` (incl. Amendment 1);
adjudication is reserved for the 30-seed run, which is *held* pending the
registered budget probe (`research/DMC-BUDGET-PROBE-PREREGISTRATION-
2026-09-07.md`) because this sanity pattern leaves an unresolved operating-
point confound (below). **Data:** `runs/dmc_payoff_10seed_gpu.json` (50
rows, 10 seeds x 5 arms, exact per-seed teacher checksum pairing, gap 0.0,
GPU/2xT4, Kaggle). **Protocol:** `configs/dmc_hopper_distill.yaml`,
dm_control/hopper-hop-v0 (1000-step episodes), MBPO/SAC, 3,600 env steps,
live critic, M=2 latents for identified arms, seeds 0-9.

## Gate record (mandatory diagnostic, Amendment 1)

Ordinary-arm pilot, seeds {0, 1}, live-critic eval states, reader
`udwm/scripts/dmc_gate_ratio.py`:

| seed | teacher_g | teacher_w | g*/w* |
|---|---:|---:|---:|
| 0 | 3.935 | 0.00044 | 9028.60 |
| 1 | 1.534 | 0.00021 | 7419.26 |

Median g*/w* = 8223.93 -> **aleatoric-dominated** (pre-committed branch:
expect the DelayedBimodal pattern - EMA collapse, equal-weight partial
transfer; the payoff question open at long horizon). Gate written to
`runs/dmc_gate_pilot.json` before stage2 was allowed to start.

## Per-arm means (n=10)

| arm | u_rank_corr | w_rank_corr | u_rmse | w_rmse | next_state_mse | final_return |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 0.5112 | 0.1500 | 2.4567 | 0.0423 | 1.5228 | 0.0233 |
| lagged_hybrid | 0.5727 | 0.1555 | 1.0791 | 0.0110 | 1.4778 | 0.1013 |
| hybrid | 0.4376 | 0.1673 | 1.8951 | 0.0070 | 1.1419 | 0.0655 |
| identified_eq | 0.4294 | 0.1113 | 1.7264 | 0.0159 | 1.2193 | 0.0428 |
| identified_hybrid | 0.0101 | 0.1211 | 6.1838 | 0.0009 | 0.9528 | 0.0677 |

## Paired contrasts of note (mean delta, wins/N, bootstrap 95%; N=10, sanity)

`identified_eq - identified_hybrid` (bar 1, mechanism):
- u_rank +0.419 (10/10, CI [+0.339, +0.487]) -> **replicates: eq >> EMA**;
- u_rmse -4.457 (0/10) eq better; next_state_mse +0.267 (10/10) eq worse.

`identified_eq - ordinary`:
- u_rank -0.082 (2/10, CI [-0.143, -0.013]) -> **confirmed below ordinary**
  (DelayedBimodal eq-ordinary was -0.104, 0/10: same direction, same size);
- next_state_mse -0.304 (0/10) eq better; w_rmse -0.027 (1/10) eq better.

`identified_eq - hybrid` (the DelayedBimodal headline contrast):
- u_rank -0.008 (5/10) -> **inconclusive**. On DelayedBimodal this contrast
  was +0.214, 10/10. This is the single cleanest non-replication.
- next_state_mse +0.077 (8/10) eq worse; w_rmse +0.009 (9/10) eq worse.

`identified_eq - lagged_hybrid`:
- u_rank -0.143 (0/10, CI [-0.165, -0.123]) eq worse (confirmed);
- u_rmse +0.647 (8/10) eq worse; next_state_mse -0.259 (0/10) eq better.

`hybrid - ordinary`:
- u_rank -0.074 (2/10, CI [-0.129, -0.016]) -> hybrid dip reproduces
  directionally (DelayedBimodal: -0.318, 0/10), smaller magnitude here;
- next_state_mse -0.381 (0/10) and w_rmse -0.035 (0/10) hybrid better.

final_return: no contrast confirms (all CIs include 0; per-arm means
0.023-0.101). selective_rank_corr / selective_recall_bad: no eq contrast
confirms; hybrid - ordinary selective_recall_bad +0.031 (7/10) confirms.

## Pre-registered bars (sanity check, verdicts at N=30)

- Bar 1 (eq - identified_hybrid u_rank, wins >= 7/10 AND CI excludes 0):
  MET at N=10 (+0.419, 10/10).
- Bar 2 (eq u_rank >= 0.70 on >= 7/10): **NOT met - 0/10**, mean 0.429.
- MSE bound (eq next_state_mse <= 2x ordinary): met (1.219 <= 3.046).

## Reading

1. **The EMA-collapse mechanism generalizes across environments.** eq > EMA
   on u-rank by +0.419 with a tight CI and 10/10 wins - the 2026-09-05
   attribution (EMA per-term reweighting annihilates the aleatoric channel;
   the identified target itself is not the pathological object) is not a
   DelayedBimodal artifact.
2. **The within-study arm ordering mostly replicates, absolute levels do
   not.** DelayedBimodal u_rank: lagged 0.955 > ordinary 0.939 > eq 0.835 >
   hybrid 0.621 > EMA 0.11. DMC: lagged 0.573 > ordinary 0.511 > hybrid
   0.438 ~= eq 0.429 > EMA 0.01. Every arm loses ~half its level on DMC;
   even the pure state-MSE baseline (ordinary) reaches only 0.51 here vs
   0.94 on DelayedBimodal. The eq/hybrid pair is the only inversion.
3. **The eq - hybrid reversal is the one true DMC-specific signal.** On
   DelayedBimodal the M=2 equal-weight arm beat the M=1 conflated arm on
   every seed; here they are indistinguishable and both sit ~0.08 below
   ordinary on u_rank (eq - ordinary and hybrid - ordinary are both
   confirmed-below). The identifiability correction only pays when the
   decision term carries signal; on DMC at this budget it appears to add
   noise without signal.
4. **Secondary costs do not replicate either way.** The DelayedBimodal w hole
   (eq w_rmse ~2x ordinary) is absent: on DMC eq w_rmse is *better* than
   ordinary (0.016 vs 0.042, 9/10). eq and hybrid improve u_rmse and
   next_state_mse vs ordinary while hurting u_rank - the fixed-map
   "reallocation" pattern, now under a live critic.
5. **Unresolved operating-point confound (why the 30-seed run is held).**
   3,600 env steps on a 1,000-step-episode task is 3.6 episodes, vs 15-30
   episodes for the DelayedBimodal studies (1,800/3,600 steps at 120-step
   episodes). Returns are ~0.02-0.10 and indistinguishable across arms; the
   live critic and policy are plausibly undertrained, so u_rank measured
   under that critic may be floor-compressed for every arm. This sanity
   cannot distinguish "identified_eq does not transfer to DMC" from "the
   whole u_rank measurement is degenerate at a 3.6-episode budget". The
   pre-committed aleatoric-branch consequence (equal-weight recovers u-rank)
   was therefore NOT reproduced at sanity scale - recorded here before any
   further compute. The registered budget probe discriminates the two
   readings; do not quote any sanity verdict as adjudicated.

## Files / doc updates

- This record; data on the Kaggle host (`runs/dmc_payoff_10seed_gpu.json`).
- `DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`: dated addendum (sanity exists;
  no protocol change until the probe lands).
- `DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`: decision rule for the
  30-seed run (budget confound vs genuine non-transfer).
- `RESULTS.md` / `PAPER-NARRATIVE.md`: dated addenda.