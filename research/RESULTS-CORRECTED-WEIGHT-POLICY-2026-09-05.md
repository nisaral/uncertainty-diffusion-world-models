# Corrected-weight policy N-study: equal-weight identified transfers to the live critic (2026-09-05)

**Registration:** `research/CORRECTED-WEIGHT-POLICY-PREREGISTRATION-2026-09-05.md`
(written and committed before any row ran). Adjudication summarizer
(`udwm/scripts/summarize_corrected_policy.py`) was also committed before
results. **Data:** `runs/policy_corrected_weights_10seed.json` (60 rows,
10 seeds x 6 arms, exact per-seed teacher checksum pairing, gap 0.0).
**Protocol:** configs/delayed_bimodal_distill.yaml, MBPO/SAC, 1,800 env
steps, live SAC critic, M=2 latents for identified arms, CPU, seeds 0-9.

## Why this run exists

The N=30 policy 2x2 verdict ("identified_hybrid does not recover u-rank,
0/30") was measured on the EMA-reweighted identified arm - the variant the
2026-09-05 mechanism correction showed annihilates the aleatoric channel even
in easy regimes (probe: u-rank eq - ema +0.712, 10/10; toy: EMA kills g even
at w*/g* parity). Whether the identified target itself (equal weight, no EMA)
transfers to policy was never run. This study is that run.

## Per-seed u_rank (primary endpoint)

| seed | ordinary | hybrid | lagged_hybrid | identified_hybrid | identified_eq | identified_wonly |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.960 | 0.746 | 0.962 | -0.016 | 0.879 | 0.014 |
| 1 | 0.974 | 0.698 | 0.980 | 0.070 | 0.833 | 0.212 |
| 2 | 0.936 | 0.533 | 0.980 | 0.046 | 0.898 | 0.041 |
| 3 | 0.970 | 0.693 | 0.963 | 0.103 | 0.834 | 0.110 |
| 4 | 0.963 | 0.572 | 0.945 | -0.007 | 0.853 | -0.044 |
| 5 | 0.934 | 0.632 | 0.968 | 0.107 | 0.893 | 0.050 |
| 6 | 0.951 | 0.658 | 0.959 | 0.105 | 0.836 | 0.274 |
| 7 | 0.876 | 0.436 | 0.940 | 0.226 | 0.651 | 0.083 |
| 8 | 0.935 | 0.637 | 0.905 | 0.330 | 0.890 | 0.333 |
| 9 | 0.889 | 0.602 | 0.949 | 0.180 | 0.780 | 0.003 |
| mean | 0.939 | 0.621 | 0.955 | 0.114 | **0.835** | 0.108 |

## Adjudication (pre-registered bars)

- **Bar 1 (eq - identified_hybrid u_rank): MET.** +0.720, 10/10 seeds,
  5,000-draw bootstrap 95% CI [+0.627, +0.802].
- **Bar 2 (eq u_rank >= 0.70 on >= 7/10 seeds): MET.** 9/10 seeds (seed 7 =
  0.651, the same weak seed as the probe's seed 7), mean 0.835.
- **MSE bound (eq next_state_mse <= 2x ordinary): MET.** eq 0.244 vs
  ordinary 0.276 (2x bound 0.552); eq is in fact *better* than ordinary on
  next-state MSE (-0.032, 0/10 seeds worse, CI [-0.044, -0.021]).

**Branch A (both bars met): equal-weight identified transfers to the live
critic at policy scale.** Pre-committed consequence: the N=30 "identified
does not matter in policy" verdict was the EMA variant, not the identified
target; the central policy claim flips.

## Paired contrasts of note (mean delta, wins/N, bootstrap 95%)

- identified_eq - identified_hybrid: u_rank +0.720 (10/10); u_rmse -109.9
  (better, 0/10 worse); w_rmse +0.98 worse (10/10); next_state_mse +0.081
  worse (10/10).
- identified_eq - hybrid: u_rank +0.214 (10/10, CI [+0.172, +0.262]) - the
  M=2 equal-weight identified arm beats the M=1 conflated hybrid arm on
  u-rank under the live critic on every seed. u_rmse -134.7 better (0/10
  worse). w_rmse +0.92 worse (10/10); next_state_mse +0.051 worse (10/10).
- identified_eq - ordinary: u_rank -0.104 (0/10, CI [-0.139, -0.073]) -
  confirmed below ordinary: transfer is real but does not reach parity.
- identified_eq - lagged_hybrid: u_rank -0.120 (0/10) - lagged_hybrid remains
  the best ranker (0.955) at this N; eq is second.
- identified_wonly - identified_eq: u_rank -0.727 (0/10) - w-only EMA
  collapses identically to EMA-both; the epistemic up-weight is the driver
  (reproduces the probe conclusion at policy level).
- hybrid - ordinary: u_rank -0.318 (0/10) - reproduces the N=30 hybrid dip
  (-0.310) almost exactly.
- final_return: no contrast confirms (all CIs include 0); identified arms'
  earlier "best return" does not replicate (eq -99.1, ordinary -82.4, EMA
  arms -74.7/-83.3, lagged -129.9). Return does not adjudicate (secondary,
  per registration).

## Reading

The story now has a measured policy-level spine, not just a probe:

1. EMA-both and w-only-EMA annihilate u-rank under the live critic (0.11,
   noise level), reproducing the N=30 collapse and confirming the 2026-09-05
   attribution at policy scale: the pathological object was the per-term EMA
   reweighting (epistemic up-weight ~1e5-1e6), not the identified loss.
2. Equal-weight identified (identified_eq) recovers u-rank at policy scale:
   0.835 mean, 9/10 >= 0.70, and beats the M=1 hybrid arm on every seed
   (+0.214, 10/10). The identifiability correction (M>=2 split) now has a
   measured policy benefit over the conflated single-latent target.
3. The transfer is partial, not parity: eq sits below ordinary (0.939) and
   lagged_hybrid (0.955) by ~0.10-0.12 (confirmed, 0/10). Lagging the value
   map remains the strongest single correction for u-rank.
4. Costs are stable across all identified arms: w_rmse is worse (the equal-
   weight hole on w persists at policy scale; w_rmse 0.995 vs ordinary
   0.497), and equal-weight identified's next-state MSE is worse than the
   EMA arms (0.24 vs 0.16) but better than ordinary (0.28) at this N.

Implication for the registered DMC gate (research/DMC-PAYOFF-PREREGISTRATION-
2026-09-05.md): the pre-registration stands as written - identified_eq is the
uncertainty-preserving candidate (now confirmed to transfer at DelayedBimodal
policy scale), identified_hybrid is the expected-collapse control. The
regime-conditional predictions (aleatoric-dominated -> collapse under EMA,
transfer under equal weight) no longer rest on probe data alone.

## Files / doc updates

- `CORRECTED-WEIGHT-POLICY-PREREGISTRATION-2026-09-05.md`: registration.
- `RESULTS.md` / `PAPER-NARRATIVE.md`: dated addenda (the N=30 "identified
  does not matter in policy" line is superseded for the equal-weight arm).
- `RESULTS-POLICY-2X2-30SEED-2026-09-03.md`: addendum note (its identified
  rows were the EMA variant; see this doc).
- `RESULTS-CORRUPTION-2026-09-05.md`: addendum note (probe conclusion now
  confirmed at policy scale).