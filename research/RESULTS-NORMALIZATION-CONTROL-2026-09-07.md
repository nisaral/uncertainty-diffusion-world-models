# G7 normalization-controlled w-channel contrast: result (2026-09-07)

**Registration:** `research/NORMALIZATION-CONTROL-PREREGISTRATION-2026-09-07.md`
(written before any G7 row ran). Novel-gap G7 of
`research/NOVEL-GAPS-2026-09-07.md`; closes the Theorem 7 caveat
(`research/proofs/identifiability-frontier.md`, open item 1): the absolute
w-RMSE scale of lagged vs live arms was confounded because the registered
lagged arms train with `distill_normalize_values: true` and the live arms
with `false`.

**Data:** `runs/normalization_control_10seed.json` (40 rows, 10 seeds x 4
cells A-D, exact per-seed teacher checksum pairing, gap 0.0 for all 10
seeds). **Protocol:** DelayedBimodal-v0, `configs/delayed_bimodal_distill.yaml`,
1,800 env steps, live MBPO/SAC, M=2 latents, equal weights, no EMA, CPU,
seeds 0-9. All four cells ran in one file on one machine; A/C rows from
`runs/policy_combined_fix_10seed.json` were NOT reused (within-file pairing
is the controlled unit). Adjudication summarizer
(`udwm/scripts/summarize_normalization_control.py`) written before the rows
completed; stats are paired deltas, 20,000-draw percentile bootstrap 95% CI,
fixed RNG seed 20260907, wins/N.

## Why this run exists

The combined-fix table (N=10) quoted w_rmse 0.922 (identified_eq) vs 0.283
(lagged_identified_eq). Two knobs differ across that contrast: the lag axis
(live critic vs slow target critic) and value-target standardization
(`normalize_values`). The discriminator holds normalization fixed within each
lag cell, so the w-gap can be attributed to lagging or to normalization - or
shown to be neither.

## Cells (registered 2x2)

| cell | variant | lag axis | normalize_values | guard / warmup |
|---|---|---|---|---|
| A | `identified_eq` | live critic | false | false / 0 |
| B | `identified_eq_norm` | live critic | true | false / 0 |
| C | `lagged_identified_eq` | target critic | true | true / 50 |
| D | `lagged_identified_eq_nonorm` | target critic | false | true / 50 |

B and D copy the existing arms and toggle ONLY `distill_normalize_values`.

## Per-arm means (n=10)

| cell | arm | u_rank | w_rank | u_rmse | w_rmse | next_state_mse | final_return |
|---|---|---:|---:|---:|---:|---:|---:|
| A | identified_eq | 0.8408 | 0.2575 | 46.74 | 1.2703 | 0.2522 | -95.2 |
| B | identified_eq_norm | 0.9524 | 0.1398 | 34.31 | 0.3646 | 0.2760 | -140.5 |
| C | lagged_identified_eq | 0.9517 | 0.1776 | 31.71 | 0.3394 | 0.2730 | -98.0 |
| D | lagged_identified_eq_nonorm | 0.8397 | 0.1842 | 47.10 | 1.1220 | 0.2539 | -77.0 |

Direction-level cross-check vs the combined-fix file (separate process run;
never mixed for adjudication): A was 0.877 u_rank / 0.922 w_rmse there vs
0.841 / 1.270 here; C was 0.948 / 0.283 there vs 0.952 / 0.339 here.
Ordering is preserved on both arms; absolute values drift between runs.

## Adjudication (pre-registered primary bars, N=10)

Primary: paired w_rmse contrast "lagged - live" within each normalization
cell (negative = lagged better):

| contrast | mean | lagged-better | 95% CI | bar |
|---|---:|---:|---|---|
| D - A (both unnormalized) | -0.1483 | 7/10 | [-0.360, +0.065] | NOT met (CI includes 0) |
| C - B (both normalized) | -0.0252 | 7/10 | [-0.076, +0.042] | NOT met (CI includes 0) |

**Pre-committed verdict: NEITHER contrast excludes 0 -> normalization, not
lagging, drove the 0.92 -> 0.28 w_rmse gap. The absolute w-scale claim is
dropped to rank endpoints only** (the u-rank bars, scale-free under monotone
score transforms per Theorem 6, carry the adjudicated story).

## Descriptive within-cell reads (never adjudicative on their own)

Within-cell lag contrasts on u_rank_corr (normalization held fixed):

| contrast | u_rank mean | up | 95% CI |
|---|---:|---:|---|
| D - A (unnormalized cell) | -0.0011 | 6/10 | [-0.033, +0.026] |
| C - B (normalized cell) | -0.0007 | 6/10 | [-0.013, +0.010] |

The lag axis adds ~0 to u_rank at N=10 within either normalization cell.

Normalization effect (within a lag cell):

| contrast | u_rank mean | w_rmse mean | reads |
|---|---:|---:|---|
| B - A (live cell, norm - nonorm) | +0.1115 (10/10, CI [+0.078, +0.145]) | -0.906 (10/10 lower) | standardization lifts u_rank and collapses w_rmse scale |
| C - D (lagged cell, norm - nonorm) | +0.1120 (10/10, CI [+0.084, +0.145]) | -0.783 (10/10 lower) | same sign, same size |

## What this changes in the paper record

1. **The w_rmse channel is a normalization-scale artifact, not a lagging
   result.** The combined-fix doc's "w hole largely closed by the combined
   arm (-0.639 vs identified_eq)" is re-read: that contrast changed
   normalization AND the lag axis; G7 shows the scale drop is carried by
   standardization (10/10 in both lag cells) and the lag axis adds ~0
   within-cell (CIs include 0). w-scale language is not admissible; the w
   claim is rank endpoints only. (Secondary: w_rank_corr is weak everywhere
   in this file, 0.14-0.26, consistent with the aleatoric-dominated regime -
   the u_rank endpoints are the ones that move.)
2. **The combined-fix Bar-2 attribution (lagged_identified_eq - identified_eq
   u_rank +0.0707, 8/10) is re-opened.** That contrast also changed
   normalization. G7's within-cell lag contrasts are null (-0.0011, -0.0007),
   while standardization alone moves u_rank +0.11 (10/10, both cells). The
   registered lagged configuration's u-rank advantage over the live
   configuration is therefore carried by `normalize_values` at N=10 on
   DelayedBimodal; the empirical arm-to-term mapping in Theorem 7's measured
   table must be re-read accordingly. Theorem 7 C7.2 itself (lagging bounds
   the drift term - a deterministic identity on the frozen-map decomposition)
   is theory and is unaffected; what changes is the empirical claim that the
   lag axis is what closed the eq gap.
3. **A simpler arm reaches the same u-rank level.** identified_eq_norm (live
   critic + standardization) matches lagged_identified_eq on u_rank (0.9524
   vs 0.9517, C - B = -0.0007) and w_rmse (0.365 vs 0.339) in this file. The
   "two fixes compose" sentence becomes: equal-weight M>=2 identified plus
   value-target standardization recovers u-rank to the measured ceiling on
   the LIVE critic; the slow-map arm adds nothing measurable here. A
   within-file re-adjudication that also carries ordinary/hybrid/lagged_hybrid
   baselines is registered separately (G9) so the corrected arm-to-term
   mapping can be computed inside one file.
4. **What is NOT affected.** The mechanism claims at fixed configs stand:
   M=1 single-latent non-identifiability (fibre/sign flip), EMA-both
   aleatoric annihilation (identified_hybrid u-rank ~ 0.11), and equal-weight
   M>=2 recovery are all contrasts that hold normalize_values and the lag
   axis fixed. The combined-fix u-rank bars as measured (A 0.877, C 0.948 in
   that file) are not re-adjudicated here; their causal attribution is.

## Files

- Pre-registration: `research/NORMALIZATION-CONTROL-PREREGISTRATION-2026-09-07.md`.
- Data: `runs/normalization_control_10seed.json`.
- Summarizer: `udwm/scripts/summarize_normalization_control.py`.
- Follow-up: G9 normalization-controlled re-adjudication with in-file
  baselines (`research/NORMALIZATION-CONTROL-READJUDICATION-PREREGISTRATION-2026-09-07.md`).
