# Policy 2x2 — first pass, 5 seeds (preliminary, previously unreported) (2026-09-01)

**Data:** `runs/policy_identifiability_2x2.json` (written 2026-08-29 18:31, never
summarized). 5 seeds x 5 arms x 1800 env steps, exact teacher checksum pairing
(5/5 arms share the teacher per seed), DelayedBimodal-v0, CPU.
**Runner:** `udwm/scripts/run_delayed_bimodal_policy_ablation.py`.
**Registered predictions:** `research/RESULTS-IDENTIFIABILITY-2026-08-29.md` §4.

This is a **preliminary** table, not a verdict. It exists to (a) report data
that was already collected, and (b) state where the registered experiment
stands before more seeds are spent.

## Headline numbers (mean over 5 seeds)

| arm | final_return | w_rank_corr | u_rank_corr | w_rmse | u_rmse | next_state_mse |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | **−71.3** | 0.107 | 0.954 | 0.526 | 28.9 | 0.274 |
| hybrid | −124.0 | 0.136 | 0.653 | **0.070** | 166.6 | **0.188** |
| lagged_hybrid | −103.8 | 0.137 | 0.964 | 0.241 | **25.5** | 0.269 |
| identified_hybrid | −97.2 | 0.157 | 0.829 | 0.788 | 41.0 | 0.234 |
| lagged_identified | −110.2 | **0.170** | 0.946 | 0.348 | 30.2 | 0.272 |

Paired deltas (5 seeds):

| contrast | w_rank | u_rank | w_rmse | u_rmse | return | mse |
|---|---:|---:|---:|---:|---:|---:|
| hybrid − ordinary | +0.028 (3/5) | **−0.301 (0/5)** | −0.456 | +137.7 (5/5) | −52.7 (0/5) | −0.086 |
| identified − hybrid | +0.022 (3/5) | +0.176 (4/5) | +0.719 (5/5) | −125.6 | +26.8 (5/5) | +0.046 |
| lagged_hybrid − hybrid | +0.001 (3/5) | **+0.311 (5/5)** | +0.172 | −141.0 | +20.1 (4/5) | +0.081 |
| lagged_identified − identified | +0.012 (3/5) | +0.117 (5/5) | −0.440 | −10.8 | −13.0 (1/5) | +0.038 |

## Reproduction

The hybrid u-rank collapse reproduces the published 10-seed study almost
exactly: mean delta −0.301 (0/5 seeds better) vs −0.2968 (worse 10/10) in
`research/RESULTS-POLICY-SCALE-2026-08-22.md`. The same loss, same failure.

## Registered predictions vs this data

1. **`lagged_hybrid` "reduces but does not eliminate" the collapse.** At 5
   seeds it *eliminates* it on u_rank (0.964 vs ordinary 0.954; +0.311, 5/5)
   and on u_rmse (25.5 vs 28.9). Direction is consistent with "nonstationarity
   is the dominant measured driver on this benchmark," but 5 seeds cannot
   distinguish "reduced but present" from "removed."
2. **`identified_hybrid` (live critic) recovers rank correlation.** Partial:
   u_rank 0.653 → 0.829 (+0.176, 4/5), still below ordinary; w_rank is the best
   non-lagged arm (0.157) but w_rmse is the worst (0.788). **Confounded:** this
   arm ran with the known equal-weighting hole
   (`RESULTS-IDENTIFIABILITY-2026-08-29.md` §5) — g is ~orders of magnitude
   larger than w, so the epistemic term was numerically negligible. No verdict.
3. **`lagged_identified` best or tied.** On uncertainty endpoints yes: best
   w_rank (0.170), u_rank ≈ ordinary (0.946), u_rmse 30.2. On return, no
   decision arm beats ordinary at this horizon.

Return: ordinary remains best; every decision arm costs return here
(consistent with the 10-seed study's 3/10). next_state_mse: hybrid best.

## Honest bottom line

- The data leans **nonstationarity as the measured cause of the online u-rank
  collapse**; the identifiability mechanism's *policy* consequence is not yet
  isolated because the identified arm ran with the reweighting hole.
- The ground-truth toy (`theory/ground_truth_w_g.py`) quantifies that hole
  (w_rank 0.40 when g* >> w*) and shows the reweighting fix is partial.
- **Before more seeds:** the weighting is now settled on the toy (EMA per-term
  reweighting, `distill_reweight_ema: true` on both identified arms) and the
  10-seed rerun is pre-registered with a written decision tree
  (`research/DECISION-TREE-2X2-PREREGISTRATION.md`). Until that rerun,
  "identified distillation is confirmed in policy" must not be claimed.
