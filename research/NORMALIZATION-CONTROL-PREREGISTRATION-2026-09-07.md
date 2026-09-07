# G7 pre-registration: normalization-controlled w-channel contrast (2026-09-07)

**Status: registered before any G7 row runs.** No G7 row has run; this file
fixes arms, endpoints, and bars before execution. Novel-gap G7 of
`research/NOVEL-GAPS-2026-09-07.md`; closes the Theorem 7 caveat
(`research/proofs/identifiability-frontier.md`, open item 1(ii)): the
absolute w-RMSE scale of lagged vs live arms is confounded because the
lagged arms train with the registered `distill_normalize_values: true` knob
and the live arms with `false`, so "the combined arm closes the w hole
(0.92 -> 0.28)" is not yet a controlled mechanism claim.

## Why this run exists

The combined-fix table (N=10, DelayedBimodal) quotes w_rmse 0.922
(identified_eq) vs 0.283 (lagged_identified_eq). Two knobs differ across
that contrast: the lag axis (target-critic map) and value-target
normalization (`normalize_values`). The discriminator holds normalization
fixed within each lag cell and holds the lag axis fixed within each
normalization cell, so the w-gap can be attributed to lagging or to
normalization - or shown to be neither (in which case the two knobs
interact and the w claim stays rank-only).

## Arms (2x2, all DelayedBimodal, M=2 identified, equal weights, no EMA)

| cell | variant name | lag axis | normalize_values | guard / warmup |
|---|---|---|---|---|
| A | `identified_eq` (existing) | live critic | false | false / 0 |
| B | `identified_eq_norm` (new) | live critic | true | false / 0 |
| C | `lagged_identified_eq` (existing) | target critic | true | true / 50 |
| D | `lagged_identified_eq_nonorm` (new) | target critic | false | true / 50 |

New variant rows in `udwm/scripts/run_delayed_bimodal_policy_ablation.py`
copy the existing arm exactly and toggle only `distill_normalize_values`.
Everything else (teacher pairing, seeds, steps, eval endpoints) matches the
combined-fix protocol: DelayedBimodal-v0, 1,800 env steps, seeds 0-9, exact
per-seed teacher checksum pairing, CPU or GPU rows never mixed.

## Endpoints and bars (decided before reading)

- Primary: paired `w_rmse` contrast `lagged - live` **within each
  normalization cell**: (C - A) with both unnormalized... (correction,
  read as cells): the cell-pair contrasts are D - A (both unnormalized) and
  C - B (both normalized). If BOTH exclude 0 in the "lagged better"
  direction with >= 7/10 wins: lagging reduces w_rmse independent of
  normalization -> the w claim is a controlled mechanism claim.
- If NEITHER excludes 0: normalization (not lagging) drove the 0.92 -> 0.28
  gap; the w claim is dropped to rank endpoints only.
- If exactly one excludes 0: report the interaction; w claim conditional.
- Secondary (never adjudicative): normalization effect within lag cell
  (B - A, D - C), and `u_rank_corr` for all four cells (the u-rank bars of
  the combined-fix registration must not move; this study does not
  re-adjudicate them).
- Stats: paired deltas over shared seeds, 20,000-draw percentile bootstrap
  95% CI with fixed RNG seed, wins/N, same convention as every
  adjudicated table in this repo. N=10.

## Files

- Pre-registration: this file. Runner:
  `udwm/scripts/run_delayed_bimodal_policy_ablation.py` with
  `configs/delayed_bimodal_distill.yaml`. Data:
  `runs/normalization_control_10seed.json`. ALL FOUR cells (A-D) run in one
  file on one machine - existing A/C rows from `runs/policy_combined_fix_10seed.json`
  are NOT reused for adjudication (CPU rows across machines are not
  bit-comparable; within-file pairing is the controlled unit).
- Results doc: `research/RESULTS-NORMALIZATION-CONTROL-2026-09-07.md` (to be
  written after rows complete).
