# Policy 2x2 decision tree — pre-registered (2026-09-01)

**Status:** registered BEFORE the 10-seed rerun. The 5-seed preliminary
(`research/RESULTS-POLICY-2X2-2026-09-01.md`) is excluded from adjudication:
its identified arms ran with the known equal-weighting hole and are confounded.
The rerun uses the EMA per-term reweighting fix
(`udwm.models.consistency.TermScaleEMA`, enabled via `distill_reweight_ema: true`
on both identified arms in `run_delayed_bimodal_policy_ablation.py`).

## Protocol (fixed in advance)

- Arms (all 5, same batch, 10 seeds each, 1,800 env steps, exact teacher
  checksum pairing per seed): `ordinary`, `hybrid`, `lagged_hybrid`,
  `identified_hybrid`, `lagged_identified`.
- Run:
  ```
  python -m udwm.scripts.run_delayed_bimodal_policy_ablation \
    --variants ordinary hybrid lagged_hybrid identified_hybrid lagged_identified \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out runs/policy_identifiability_2x2_10seed.json
  ```
  (Fresh output file: the old `policy_identifiability_2x2.json` holds the
  confounded equal-weight 5-seed rows and must not be mixed in.)
- **Primary endpoint:** `u_rank_corr` (teacher-student rank correlation of the
  uncertainty signal on the eval set), analysed as paired per-seed deltas.
- **Secondary endpoints:** `w_rank_corr`, `u_rmse`, `w_rmse`, `next_state_mse`,
  `final_return`, `selective_rank_corr`, `selective_recall_bad`.
- **Adjudication statistic** (matches repo convention): mean paired delta,
  wins/10, and 5,000-seed percentile bootstrap 95% CI.
  - "**confirmed**" = CI excludes 0 AND >= 7/10 seeds in the predicted
    direction.
  - "**absent**" = CI excludes 0 in the opposite direction (or >= 7/10 seeds
    against).
  - "**inconclusive**" = anything else (too few seeds / noise). Inconclusive is
    a possible outcome; it will be reported as such, not re-run until it
    cooperates.
- Comparisons used for adjudication:
  - `lagged_hybrid` vs `hybrid` (does lagging fix the u-rank collapse?),
  - `identified_hybrid` vs `hybrid` (does the identified target fix it under
    the LIVE critic?),
  - `lagged_identified` vs `hybrid` (combined),
  - `lagged_identified` vs `identified_hybrid` (does lag add anything once the
    target is identified?).

## The tree

| Outcome (10-seed verdicts) | What it means for the paper | Paper framing |
|---|---|---|
| **Row 1:** `lagged_hybrid` confirmed to fix u-rank (to ~ordinary) AND `identified_hybrid` not confirmed under the live critic | Nonstationarity is the dominant measured mechanism; identifiability exists in principle but is not the measured policy failure | Identifiability = theoretical contribution (proof + toy); **nonstationarity = headline empirical mechanism**. Strong, honest result: "the online failure mode is dominated by target nonstationarity, not the identifiability pathology we proved exists in principle." Payoff test (#4) should test **lagging alone**, not the identified machinery. |
| **Row 2:** `identified_hybrid` confirmed to fix u-rank under the LIVE critic | Identifiability is the measured mechanism; the identified loss is the fix | Thesis becomes a calibration result that transfers to any value map. Lagging is secondary. |
| **Row 3:** both confirmed | Both mechanisms matter; combined (`lagged_identified`) should be best or tied | Report both fixes; if lag adds nothing once identified (`lagged_identified` ~ `identified_hybrid`), say so explicitly. |
| **Row 4:** neither confirmed | The mechanisms do not transfer to policy on this benchmark (identifiability stands as theory; lagging alone insufficient) | Publishable negative about the mechanism's reach. Report honestly; do not force the story. |

**Anti-p-hacking rules (fixed):**
1. The primary endpoint and adjudication thresholds above are not adjusted
   after seeing the data.
2. If a verdict lands in "inconclusive", it is reported as inconclusive with
   the CI shown. Extending seeds is a NEW registration, not a continuation.
3. Return is NOT the primary endpoint; uncertainty-fidelity endpoints decide
   the mechanism question. Return is reported for completeness and to inform
   the payoff test, not to adjudicate the tree.
4. All five arms run in one batch with exact teacher pairing; no arm is
   dropped post-hoc.

## What the toy says the rerun is testing

`theory/ground_truth_w_g.py` P3-P4 (2026-09-01): equal weights fail on w when
g* >> w* (w rank 0.40); the EMA normaliser recovers w; rank robustness of w to
the weighting is the open question P4 answers. If w *ranking* is robust to the
weighting, the identified arm's downstream effect (rank-based gating) is
bounded even if magnitudes remain imperfect -- that reframing is stated in the
write-up either way.
