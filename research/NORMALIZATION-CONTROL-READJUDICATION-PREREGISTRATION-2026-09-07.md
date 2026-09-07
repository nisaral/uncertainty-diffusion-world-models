# G9 pre-registration: normalization-controlled re-adjudication of the
# combined-fix arm-to-term mapping, with in-file baselines (2026-09-07)

**Status: registered before any G9 row runs.** No G9 row has run. Triggered
by the G7 verdict (`research/RESULTS-NORMALIZATION-CONTROL-2026-09-07.md`):
G7 adjudicated the normalization-controlled w-channel contrast (both primary
bars failed; within-cell lag effects null; standardization alone lifts u_rank
+0.11, 10/10, in both cells) but its file carries NO baselines, so the
corrected arm-to-term mapping cannot be computed inside one file and
"identified_eq_norm reaches the measured ceiling / parity with ordinary" is
not yet adjudicated. This run supplies the missing cells in the same file.

## Why this run exists

The combined-fix record's G7 addendum withdraws the lag-attribution of Bar 2
(+0.0707 vs identified_eq was measured across a contrast that also changed
`normalize_values`) and the w-hole closing (scale artifact). The remaining
questions that decide the paper's arm-to-term sentences, all of which need
the baselines in the SAME file as the four normalization cells:

1. Does `identified_eq_norm` (live critic + standardization, a NEW arm never
   measured against baselines) reach the combined-fix u-rank bar (>= 0.70 on
   >= 7/10 seeds) and what is its within-file contrast vs `ordinary` and
   `lagged_hybrid` (ceiling-bound parity read)?
2. Does the within-cell lag null of G7 replicate in a larger-arm file
   (C - B and D - A on u_rank)?
3. Does the file reproduce the fixed-config mechanism contrasts that are NOT
   in dispute (identified_eq vs hybrid under live/nonorm; C vs lagged_hybrid
   under lagged/norm at the ceiling)?

## Arms (all DelayedBimodal, 1,800 env steps, seeds 0-9, M=2 identified,
equal weights, no EMA, CPU, one file on one machine)

| arm | role | lag axis | normalize_values | guard / warmup |
|---|---|---|---|---|
| ordinary | baseline | n/a | false | false / 0 |
| hybrid | M=1 baseline (live) | live | false | false / 0 |
| lagged_hybrid | M=1 baseline (lagged) | target | true | true / 50 |
| identified_eq | eq, live, nonorm (G7 cell A) | live | false | false / 0 |
| identified_eq_norm | eq, live, norm (G7 cell B) | live | true | false / 0 |
| lagged_identified_eq_nonorm | eq, lagged, nonorm (G7 cell D) | target | false | true / 50 |
| lagged_identified_eq | eq, lagged, norm (G7 cell C) | target | true | true / 50 |

Rows for the four G7 cells will be re-run in this file (not reused from
`runs/normalization_control_10seed.json`): within-file pairing is the
controlled unit and cross-file rows are never mixed for adjudication.

## Endpoints and bars (decided before reading; N=10, repo conventions:
wins >= 7/10 AND 20,000-draw percentile bootstrap 95% CI excludes 0, fixed
RNG seed 20260907)

Primary:
- Bar 1 (lag-null replication): within-cell u_rank contrasts D - A and C - B.
  If EITHER excludes 0 with >= 7/10 wins, the G7 null was a power artifact
  and the lag-attribution re-opens; if both stay null, the standardization-
  carried reading is confirmed at N=10 with baselines in the same file.
- Bar 2 (standardization lifts eq on the live critic): identified_eq_norm -
  identified_eq u_rank >= 0 (confirmed direction), >= 7/10 wins, CI excludes
  0 (G7 measured +0.1115, 10/10; this is the replication inside the bigger
  file).
- Bar 3 (eq_norm reaches the registered bar): identified_eq_norm u_rank >=
  0.70 on >= 7/10 seeds (combined-fix convention).
- Bar 4 (file-level mechanism sanity): identified_eq - hybrid u_rank >= 0,
  >= 7/10 wins, CI excludes 0 (fixed live/nonorm config; the M>=2 vs M=1
  mechanism claim must reproduce inside this file).

Reported without a bar (ceiling-bound or noisy): identified_eq_norm vs
ordinary / vs lagged_hybrid u_rank (parity read), C vs lagged_hybrid
(ceiling), next_state_mse, u_rmse, final_return (secondary, never
adjudicative). w_rmse is NOT adjudicated anywhere in this study (G7 verdict:
rank-only); it is printed for record only.

## Files

- Pre-registration: this file. Runner:
  `udwm/scripts/run_delayed_bimodal_policy_ablation.py` with
  `configs/delayed_bimodal_distill.yaml`. Data:
  `runs/normalization_readjudication_10seed.json`. Summarizer:
  `udwm/scripts/summarize_normalization_readjudication.py` (written before
  rows complete). Results doc:
  `research/RESULTS-NORMALIZATION-READJUDICATION-2026-09-07.md` (to be
  written after rows complete).
