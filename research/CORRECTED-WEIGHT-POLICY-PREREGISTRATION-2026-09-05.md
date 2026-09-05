# Corrected-weight policy preregistration: does equal-weight identified (identified_eq) transfer to policy under the live critic? (2026-09-05)

**Status: registered before any arm of this study runs.** Data file
`runs/policy_corrected_weights_10seed.json` does not exist yet; no row of this
study has been read.

**Why this study exists (the load-bearing experiment).** The N=30 policy 2x2
adjudicated "identified_hybrid does not recover u-rank (0/30)" under the live
SAC critic (`RESULTS-POLICY-2X2-30SEED-2026-09-03.md`). The 2026-09-05
mechanism correction (`RESULTS-CORRUPTION-2026-09-05.md`,
`CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md`) showed that arm ran with the
per-term EMA reweighting on, and that EMA-both / w-only-EMA annihilate the
aleatoric channel at probe scale (student g ~ 0.01) even while equal-weight
identified recovers it (N=10: u-rank eq - ema +0.712, 10/10, CI [+0.589,
+0.819]). The EMA normaliser is additionally pathological on its own terms: in
the toy it annihilates g even at w*/g* parity (g_rank 0.403 vs 0.901 equal
weight). Therefore the N=30 policy verdict was measured on the variant this
thread has shown to be broken in easy regimes; whether the *identified target
itself* (equal weight, no EMA) transfers to policy was never run. This study
is that run.

**Hypothesis (one sentence).** If equal-weight identified preserves the
decision object under the live critic at policy scale (u-rank at/above the
hybrid arm's level, approaching ordinary/lagged), the central policy claim
flips from "nonstationarity dominates, identifiability does not matter in
practice" to "identifiability matters; only the naive EMA-based enforcement
was broken - the simple fix works".

## Protocol (mirrors the N=30 policy 2x2)

- Config `configs/delayed_bimodal_distill.yaml`, MBPO/SAC, 1,800 env steps,
  exact per-seed teacher pairing (checksum), M=2 latents for the identified
  arms, CPU (adjudicating rows are CPU; CUDA is not bit-identical in this
  repo and the RTX 4050 is slower than CPU at this scale).
- Arms (6): `ordinary`, `hybrid`, `lagged_hybrid` (unchanged baselines from
  the N=30 table); `identified_hybrid` (EMA-both; expected-collapse
  reproduction control), `identified_wonly` (w-only EMA; collapse control),
  `identified_eq` (test arm: equal weight, no EMA, live critic - the
  corrected-weight candidate).
- Seeds 0-9 (N=10). Runner:
  `python -m udwm.scripts.run_policy_2x2_split_seeds --seeds 0 1 ... 9
  --variants ordinary hybrid lagged_hybrid identified_hybrid identified_eq
  identified_wonly --steps 1800 --jobs 10 --threads 2
  --out runs/policy_corrected_weights_10seed.json`

## Endpoints and adjudication (decided before running)

Report every endpoint for every arm, no post-hoc arm addition, no joint
multiple-comparison correction (statement in `PAPER-NARRATIVE.md`): u_rank,
w_rank, u_rmse, w_rmse, next_state_mse, final_return, selective_rank,
selective_recall (same list the 2x2 summarizer reports). Primary endpoint:
`u_rank_corr` under the live critic (same as the N=30 adjudication; historical
levels: ordinary ~0.99, lagged_hybrid ~0.99, hybrid ~0.68, identified_hybrid
~0.2-0.3).

Success bar (pre-committed, mirrors the DMC balanced-regime bar):
1. Paired `u_rank_corr` identified_eq - identified_hybrid: positive on >= 7/10
   seeds AND 5,000-draw bootstrap 95% CI excludes 0 (mechanism transfer: eq
   must beat the EMA arm, as at probe scale), AND
2. identified_eq `u_rank_corr` >= 0.70 on >= 7/10 seeds (informative level,
   at/above the historical hybrid arm).

## Pre-committed branch interpretations

- **Branch A (both bars met): identified_eq transfers.** Central policy claim
  flips: the N=30 "identified does not matter in policy" verdict was the EMA
  variant, not the identified target. This becomes the headline result; the
  DMC pre-registration stands as written (identified_eq = uncertainty
  candidate, identified_hybrid = expected-collapse control) and the outreach
  framing settles on "identifiability matters; the naive EMA enforcement was
  the failure mode".
- **Branch B (bar 1 met, bar 2 not):** EMA is not the whole story; eq recovers
  relative to the EMA arm but sits below an informative rank under the live
  critic - a second, nonstationarity-specific obstruction remains. Revise the
  DMC pre-registration before Kaggle spend; the paper is "probe transfer
  without policy transfer, mechanism partially resolved".
- **Branch C (bar 1 not met):** the probe-scale equal-weight recovery does not
  survive the full MBPO loop; EMA was necessary-but-not-sufficient, and the
  corrected-weight story is probe-limited. DMC predictions (regime-conditional
  transfer) must be revised before spending GPU time.

## Files

Registration: this file. Data: `runs/policy_corrected_weights_10seed.json`
(+ per-seed partials). Results doc:
`research/RESULTS-CORRECTED-WEIGHT-POLICY-*.md` (written after adjudication,
same discipline as every prior doc).