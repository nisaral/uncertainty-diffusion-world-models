# G9 normalization-controlled re-adjudication: corrected arm-to-term mapping
# with in-file baselines (rows completed 2026-09-08 00:18 IST)

**Registration:** `research/NORMALIZATION-CONTROL-READJUDICATION-PREREGISTRATION-2026-09-07.md`
(written before any G9 row ran; the G7 verdict triggered this run). **Data:**
`runs/normalization_readjudication_10seed.json` (70 rows, 10 seeds x 7 arms,
exact per-seed teacher checksum pairing, gap 0.0 for all 10 seeds). **Protocol:**
DelayedBimodal-v0, `configs/delayed_bimodal_distill.yaml`, 1,800 env steps,
live MBPO/SAC, M=2 latents for identified arms, equal weights, no EMA, CPU,
seeds 0-9, one file on one machine. Summarizer
(`udwm/scripts/summarize_normalization_readjudication.py`) written before
rows completed; paired deltas, 20,000-draw bootstrap 95% CI, fixed RNG seed
20260907, wins/N.

## Why this run exists

G7 (`research/RESULTS-NORMALIZATION-CONTROL-2026-09-07.md`) adjudicated the
normalization-controlled w-channel contrast but its file carried NO
baselines, so the corrected arm-to-term mapping could not be computed inside
one file and "identified_eq_norm reaches the measured ceiling / parity with
ordinary" was not yet adjudicated. G9 supplies ordinary / hybrid /
lagged_hybrid alongside the four G7 cells in one file.

## Per-arm means (n=10)

| arm | u_rank | w_rank | u_rmse | w_rmse | next_state_mse | final_return |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 0.9232 | 0.2375 | 39.61 | 0.3927 | 0.2709 | -71.8 |
| hybrid (M=1, live) | 0.6157 | 0.2573 | 158.82 | 0.0796 | 0.1863 | -109.4 |
| lagged_hybrid (M=1, lagged) | 0.9265 | 0.2259 | 37.85 | 0.2561 | 0.2673 | -181.1 |
| identified_eq (M>=2, live, nonorm) | 0.8302 | 0.1883 | 46.33 | 1.0923 | 0.2464 | -89.4 |
| identified_eq_norm (M>=2, live, norm) | **0.9481** | 0.1721 | 34.60 | 0.3495 | 0.2737 | -107.2 |
| lagged_identified_eq_nonorm (M>=2, lagged, nonorm) | 0.8384 | 0.1850 | 48.35 | 1.0036 | 0.2510 | -82.1 |
| lagged_identified_eq (M>=2, lagged, norm) | **0.9521** | 0.2220 | 36.49 | 0.3153 | 0.2708 | -122.5 |

## Adjudication (pre-registered bars, N=10)

- **Bar 1 (within-cell lag null replication): MET (null replicated).**
  D - A u_rank +0.0082 (5/10, CI [-0.016, +0.038]); C - B +0.0039 (5/10,
  CI [-0.009, +0.017]). With normalization held fixed the lag axis adds ~0,
  exactly as in G7.
- **Bar 2 (standardization lifts eq on the live critic): MET.**
  identified_eq_norm - identified_eq u_rank +0.1179, 10/10, CI [+0.088,
  +0.147]. G7's +0.1115 replicates inside the bigger file.
- **Bar 3 (eq_norm reaches the registered bar): MET.** 10/10 seeds >= 0.70,
  mean 0.9481 (min seed value above 0.70).
- **Bar 4 (file-level mechanism sanity): MET.** identified_eq - hybrid u_rank
  +0.2145, 10/10, CI [+0.145, +0.281]. The M>=2-vs-M=1 mechanism claim
  reproduces inside this file at the fixed live/nonorm config.

## Corrected arm-to-term mapping (all contrasts now inside one file)

| reading | arms | u_rank | evidence |
|---|---|---:|---|
| mechanism floor | hybrid (M=1, live, nonorm) | 0.616 | eq beats it +0.214 (10/10) |
| partial transfer, live, nonorm | identified_eq | 0.830 | above hybrid, below ordinary (-0.093, 0/10, CI [-0.127, -0.058]) - the registered eq deficit reproduces |
| standardization closes it on the live critic | identified_eq_norm | 0.948 | +0.118 over eq (10/10); vs ordinary +0.025 (6/10, CI [+0.001, +0.051]); vs lagged_hybrid +0.022 (7/10, CI [+0.001, +0.048]) |
| lag axis, normalization held fixed | D - A, C - B | ~0 | both null (Bar 1) - G7 conclusion confirmed with baselines in-file |
| top single arm | lagged_identified_eq | 0.952 | vs eq_norm +0.004 (null); vs lagged_hybrid +0.026 (8/10, CI [+0.003, +0.052]) - small but confirmed above the M=1 lagged arm |

## Headline findings

1. **The empirical driver of the combined-fix advantage is value-target
   standardization, and it acts on the LIVE critic.** `identified_eq_norm`
   (0.948) matches `lagged_identified_eq` (0.952; C - B null) and sits at or
   above `ordinary` (0.923) and `lagged_hybrid` (0.927) with CIs that just
   exclude 0 (+0.025, 6/10; +0.022, 7/10 - small effects near the ceiling,
   reported as parity-plus, never as large margins).
2. **The lag axis is not separable from zero at N=10 (DelayedBimodal).** Both
   within-cell lag contrasts are null in G7 and replicate null here. The
   slow target critic is theoretically motivated (Theorem 7 C7.2 removes the
   drift term) but on this benchmark - where the live critic's drift over a
   1,800-step window is evidently small enough not to matter - it adds no
   measurable u-rank. Whether the lag axis matters where the map actually
   drifts is precisely what the DMC verdict (long episodes, fast policy
   improvement) is designed to test.
3. **The mechanism claims at fixed configs are intact and now in-file.**
   M=1 hybrid 0.616 < M>=2 eq 0.830 (+0.214, 10/10); the eq deficit vs
   ordinary at live/nonorm (-0.093, 0/10) is the registered partial-transfer
   magnitude; standardization removes that deficit without the slow map.
4. **w_rmse is NOT adjudicated** (G7 verdict: rank-only; the column is the
   same normalization scale artifact - identified_eq 1.092 vs
   lagged_identified_eq 0.315 tracks the knob, not the arm). w_rank_corr is
   weak everywhere (0.17-0.26), consistent with the aleatoric-dominated
   regime; u_rank is the decision endpoint.

## Paper-language consequence (registered re-read)

The sentence "both known failure modes addressed (equal-weight M>=2 + lagged
critic) closes partial transfer to parity" becomes: "the equal-weight M>=2
identified loss preserves the decision-statistic rank at the measured ceiling
when value targets are standardized - a training-side knob that acts on the
live critic (identified_eq_norm 0.948, 10/10 >= 0.70, at/above ordinary);
the slow target critic adds nothing measurable on DelayedBimodal at N=10 and
its contribution is deferred to the DMC verdict." The mechanism headline
(M=1 single-latent non-identifiability; EMA annihilation; equal-weight M>=2
recovery) is unchanged and now carries an in-file reproduction of the
critical contrast (Bar 4).

## Files

- Pre-registration: `research/NORMALIZATION-CONTROL-READJUDICATION-PREREGISTRATION-2026-09-07.md`.
- Data: `runs/normalization_readjudication_10seed.json`.
- Summarizer: `udwm/scripts/summarize_normalization_readjudication.py`.
