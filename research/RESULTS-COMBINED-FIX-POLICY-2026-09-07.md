# Combined-fix policy N-study: lagged critic + equal-weight identified (2026-09-07)

**Registration:** `research/COMBINED-FIX-PREREGISTRATION-2026-09-07.md`
(written before any row ran; Amendment 1 recorded the bit-exactness-gate
outcome and switched the design to a fresh full 7-arm table). Hypothesis H1
of `upgrade-plan-hypotheses-experiments-publication.md`: the {lagged critic}
x {equal-weight identified} cell of the 2x2 had never been run. Adjudication
summarizer (`udwm/scripts/summarize_combined_fix.py`) written before the
rows completed. **Data:** `runs/policy_combined_fix_10seed.json` (70 rows,
10 seeds x 7 arms, exact per-seed teacher checksum pairing, gap 0.0 for all
10 seeds). **Protocol:** DelayedBimodal-v0, `configs/delayed_bimodal_distill.yaml`,
1,800 env steps, live MBPO/SAC, M=2 latents for identified arms, CPU, seeds
0-9. The 6 previously-run arms double as an internal reproduction of the
2026-09-05 corrected-weight N-study.

## Why this run exists

Two fixes were confirmed separately: lagging the critic fixes nonstationarity
(`lagged_hybrid` u-rank ~ ordinary, 30/30 vs hybrid at N=30); equal-weight
identified fixes identifiability under the live critic (`identified_eq`
u-rank 0.835, 9/10 >= 0.70, beats `hybrid` 10/10). The only historical
lag+identified arm (`lagged_identified`) combined lag with the EMA
reweighting whose aleatoric starvation was only diagnosed afterwards. The
{lag} x {equal weight} cell - the intended method - had never run.

## Per-arm means (n=10) and reproduction cross-check

| arm | u_rank mean | u_rank sd | w_rmse | next_state_mse | final_return |
|---|---:|---:|---:|---:|---:|
| lagged_identified_eq | **0.9482** | 0.037 | 0.283 | 0.270 | -85.6 |
| lagged_hybrid | 0.9456 | 0.018 | 0.237 | 0.269 | -134.0 |
| ordinary | 0.9340 | 0.039 | 0.418 | 0.269 | -88.1 |
| identified_eq | 0.8775 | 0.035 | 0.922 | 0.247 | -67.4 |
| hybrid | 0.6356 | 0.081 | 0.064 | 0.187 | -123.2 |
| identified_hybrid (EMA) | 0.1137 | 0.106 | 0.015 | 0.162 | -89.9 |

Reproduction of the 09-05 rows (fresh process runs; means drift but ordering
and adjudications do not): ordinary 0.934 (vs 0.939), hybrid 0.636 (vs
0.621), lagged_hybrid 0.946 (vs 0.955), identified_eq 0.877 (vs 0.835; seed 7
is 0.873 here vs 0.651 there - the single weak seed moves between process
runs), EMA 0.114 (vs 0.114). Exact row-level reproduction is not expected
(the amendment records the gate failure); direction-level reproduction holds
on all six arms.

## Adjudication (pre-registered bars, N=10)

- **Bar 1 (lagged_identified_eq - lagged_hybrid u_rank >=, confirmed): NOT
  met.** +0.0026, 5/10, CI [-0.019, +0.023]. The two arms are at parity at
  the top of the scale (0.948 vs 0.946): lagged_hybrid was already at the
  measured ceiling (~0.95-0.98 max per seed), so this contrast is a ceiling
  test, not a headroom test.
- **Bar 2 (lagged_identified_eq - identified_eq u_rank >=, confirmed): MET.**
  +0.0707, 8/10, CI [+0.036, +0.103].
- **Bar 3 (lagged_identified_eq u_rank >= 0.70 on >= 7/10): MET.** 10/10
  (min seed value 0.870), mean 0.948 - the top of the table.
- **Branch: B + parity-at-top.** Lagging adds a confirmed +0.07 on top of the
  equal-weight correction (bar 2); the equal-weight correction adds nothing
  measurable on top of lagging because lagging had already reached the
  ceiling (bar 1 at parity). The combined arm is the best single
  configuration measured.

## Headline findings

**The strongest non-ceiling-bound effects are on the w and u-RMSE channels,
and both are confirmed with paired CIs.** The equal-weight w hole
(identified_eq w_rmse 0.922, 10/10 worse than ordinary, CI [+0.330, +0.689])
is largely closed by the combined arm: lagged_identified_eq w_rmse 0.283 -
confirmed better than identified_eq (-0.639, 0/10 worse, CI [-0.797,
-0.472]) and than ordinary (-0.135, 2/10 worse, CI [-0.234, -0.041]), at
parity with lagged_hybrid (0.237). Caveat: w_rmse alone is a trap metric -
the EMA arm is "best" on w_rmse (0.015) precisely because it annihilates the
aleatoric channel (its u-rank is 0.114); the u-rank table carries the
validity reading, and the lagged arms train with normalized value targets
(registered knob), so the absolute w_rmse scale of lagged vs live arms needs
a normalization-controlled check before being quoted as a mechanism. The
u_rmse improvement over identified_eq (-14.0, 3/10 worse, CI [-25.0, -1.9])
is also confirmed.

**On u-rank the combined arm is top-of-table at 0.948 (10/10 >= 0.70), but
every top-arm contrast is parity, not a confirmed exceedance - the ceiling
is real.** Paired contrasts on the fresh table:

| contrast | endpoint | mean delta | wins/N | bootstrap 95% |
|---|---|---|---:|---|
| lid - ordinary | u_rank | +0.014 | 6/10 | [-0.013, +0.047] |
| lid - lagged_hybrid | u_rank | +0.003 | 5/10 | [-0.019, +0.022] |
| lid - identified_eq | u_rank | +0.071 | 8/10 | [+0.035, +0.104] |
| identified_eq - ordinary | u_rank | -0.057 | 1/10 | [-0.086, -0.022] |
| lid - hybrid | u_rank | +0.313 | 10/10 | [+0.255, +0.375] |

identified_eq's u-rank deficit vs ordinary is **confirmed** on this fresh
table (-0.057; the 09-05 file measured -0.104), and lagged_identified_eq
sits at parity with ordinary and lagged_hybrid with no deficit in either
direction. So: "partial transfer -> full transfer once both known failure
modes (nonstationarity + identifiability) are addressed" is supported at
N=10 on DelayedBimodal - as confirmed-closure of eq's deficits (bars 2-3)
plus parity-at-the-top for the two single-fix baselines. The stronger H1
reading (combined *exceeding* lagged_hybrid) is not adjudicable on this
benchmark: lagged_hybrid is already at the measured ceiling (~0.95), so the
bar-1 contrast (+0.003) is a ceiling test, not a headroom test. That
question moves to DMC, where the 30-seed adjudication (held pending the
budget probe) has actual room between arms.

## Secondary endpoints

- next_state_mse: 0.270, parity with ordinary (0.269) and lagged_hybrid
  (0.269); the identified-arm cost vs hybrid persists (+0.083 vs hybrid,
  10/10, CI excludes 0) but the combined arm is no worse than ordinary -
  the naive arm's MSE advantage is not lost relative to the pure baseline.
- final_return: -85.6; better than lagged_hybrid (-134.0), 8/10, CI
  [+6.6, +94.7], but returns are noisy on this benchmark and do not
  adjudicate (secondary per registration); identified_eq (-67.4) remains the
  best point estimate.
- selective_recall_bad / selective_rank_corr: no contrast confirms; see
  `RESULTS-H2H4-REANALYSIS-2026-09-07.md` for why these columns do not
  adjudicate tail preservation.## Files / doc updates

- This record; addenda in `RESULTS.md` and `PAPER-NARRATIVE.md`.
- DMC: the combined arm should ride along in the registered DMC adjudication
  (30-seed, held pending the budget probe) - this run is DelayedBimodal-only.

## Addendum (2026-09-07): parity contrasts re-verified with one method;
combined-vs-ordinary CI recorded explicitly

The review pass on this table asked for wins/N and a CI for the
combined-vs-ordinary u-rank contrast (the one headline-sounding number that
was not yet sitting in the adjudication block with the other contrasts). It
was quoted in the table above and in `PAPER-NARRATIVE.md`, but the bootstrap
was not yet recorded as reproducible. Re-run with the same convention as the
adjudication summarizer - paired per-seed deltas over the 10 shared seeds,
20,000-draw percentile bootstrap, fixed RNG seed, wins = seeds where the
delta is positive:

| contrast | endpoint | mean delta | wins/N | bootstrap 95% | P(delta > 0) |
|---|---|---|---:|---:|---:|
| lagged_identified_eq - ordinary | u_rank | +0.0142 | 6/10 | [-0.0132, +0.0464] | 0.821 |
| lagged_identified_eq - lagged_hybrid | u_rank | +0.0026 | 5/10 | [-0.0192, +0.0224] | 0.612 |
| lagged_identified_eq - identified_eq | u_rank | +0.0707 | 8/10 | [+0.0354, +0.1032] | 1.0000 |
| identified_eq - ordinary | u_rank | -0.0565 | 1/10 | [-0.0862, -0.0215] | 0.0014 |

Reading (unchanged from the adjudication above, now with every contrast under
the identical method): the combined arm is at **parity** with ordinary and
lagged_hybrid (CIs straddle 0, wins at chance), not a confirmed exceedance -
correct phrasing everywhere is "met parity and then some on a ceiling-bound
benchmark", never "beats". The confirmed, non-ceiling effects remain Bar 2
(vs identified_eq, +0.071, 8/10) and the w-hole closing. Method:
`numpy.random.default_rng(0)`, 20,000 draws.
