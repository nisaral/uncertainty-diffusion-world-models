# Payoff test #4 - lagging alone at 2x horizon (2026-09-03)

**Data:** `runs/payoff_lagged_10seed_3600.json` (10 seeds x 3 arms x 3,600 env
steps, exact teacher checksum pairing, gap 0 for all seeds).
**Registration:** `research/PAYOFF-LAGGED-PREREGISTRATION.md` (2026-09-03),
written before the run.
**Arms:** `ordinary` / `hybrid` / `lagged_hybrid` (identified arms excluded per
the tree's Row 1 instruction to test lagging alone).

## Verdicts (fixed before the run)

| Prediction | Result | Status |
|---|---|---|
| P1: mechanism survives 2x horizon (`lagged_hybrid` u-rank ~ ordinary, beats `hybrid`) | u-rank 0.9891 vs ordinary 0.9865 (mean delta +0.0026, 7/10 within -0.02 harm); vs hybrid +0.2852, 10/10, CI [+0.238, +0.331] | **confirmed** |
| P2: return payoff of `lagged_hybrid` vs `ordinary` (no direction pre-judged) | mean delta -24.17, 4/10 seeds better, 95% CI [-148.5, +80.3] | **not demonstrated** (null at this scale) |
| P3: `hybrid` worst on uncertainty endpoints | u-rank 0.7039, u-rmse 147.0 vs 24.3-27.7 | **confirmed** |

## Per-arm means (n=10, 3,600 steps)

| arm | final_return | u_rank_corr | w_rank_corr | u_rmse | w_rmse | next_state_mse | selective_rank_corr | selective_recall_bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | -293.51 | 0.9865 | 0.2365 | 27.69 | 0.1521 | 0.3061 | -0.1291 | 0.2746 |
| hybrid | -293.82 | 0.7039 | 0.1786 | 147.01 | 0.0646 | **0.2113** | -0.0624 | 0.3376 |
| lagged_hybrid | -317.69 | **0.9891** | 0.2324 | **24.33** | 0.1393 | 0.3022 | -0.0821 | 0.3156 |

## Reading

- **The fix transfers to the longer horizon.** At 3,600 steps the lagged arm
  keeps teacher-student uncertainty rank at ordinary level (0.989) while the
  live-critic hybrid still collapses it (0.704, -0.283 vs ordinary, 0/10, CI
  excludes zero). This is the same Row-1 mechanism as the N=30 adjudication,
  measured again at 2x the horizon.
- **No return payoff is demonstrated on this benchmark.** Return deltas are
  null-to-negative for both decision arms and the bootstrap intervals are wide
  (+/-100-150). At this scale (small MBPO/SAC lab, 1-NFE student) preserving
  uncertainty does not convert into policy return within 3,600 steps.
- **Absolute returns drift down with horizon** (ordinary -104 at 1,800 steps in
  the N=30 study vs -293 at 3,600 here): the MBPO stack itself degrades on
  DelayedBimodal past ~1,800 steps, so "more steps" did not give the better
  world models more room to pay off. State this caveat plainly; a payoff claim
  would need a benchmark and protocol where longer horizons help (a
  different-registration question, e.g. the deferred modality choice).

## Registration hygiene

- Protocol, arms, seeds, and adjudication rule were fixed before the run
  (`research/PAYOFF-LAGGED-PREREGISTRATION.md`).
- Exact per-seed teacher pairing (gap 0); all arms in one merged table; no arm
  dropped post-hoc. Return is evaluated at 10 eval episodes/seed as recorded by
  the runner.


---

## N=30 extension (2026-09-03), pooled adjudication

**Data:** `runs/payoff_lagged_30seed_3600.json` = seeds 0..9 (the N=10 file
above) merged with seeds 10..29 (identical code/config, run 2026-09-03 via
`udwm.scripts.run_policy_2x2_split_seeds`; all 20 extension seeds report
`exact_teacher_match: true`, `max_teacher_checksum_gap: 0.0`).
**Registration:** N=30 amendment appended to
`research/PAYOFF-LAGGED-PREREGISTRATION.md` before seeds 10..29 were run.

## Pooled verdicts (n=30, same pre-registered rules, seed bar >= 21/30)

| Prediction | Result at N=30 | Status |
|---|---|---|
| P1: mechanism survives 2x horizon (`lagged_hybrid` u-rank ~ ordinary, beats `hybrid`) | u-rank 0.9876 vs ordinary 0.9859 (mean delta +0.0017; 29/30 within -0.02 harm); vs hybrid +0.2731, 30/30, CI [+0.2445, +0.3030] | **confirmed** (unchanged from N=10) |
| P2: return payoff of `lagged_hybrid` vs `ordinary` | mean delta -44.63, 15/30 better, 95% CI [-102.6, +10.2] | **not demonstrated** (inconclusive; point estimate null-to-negative, CI now excludes large positive payoff) |
| P3: `hybrid` worst on uncertainty endpoints | u-rank 0.7145 (vs 0.986-0.988), u-rmse 165.6 (vs 23.6-23.8) | **confirmed** |

## Per-arm means (n=30, 3,600 steps)

| arm | final_return | u_rank_corr | w_rank_corr | u_rmse | w_rmse | next_state_mse | selective_rank_corr | selective_recall_bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | -238.90 | 0.9859 | 0.1816 | 23.63 | 0.1667 | 0.3062 | -0.1661 | 0.2667 |
| hybrid | -255.37 | 0.7145 | 0.2044 | 165.59 | 0.0657 | **0.2112** | -0.0951 | 0.3163 |
| lagged_hybrid | -300.00 | **0.9876** | 0.2115 | **23.77** | 0.1959 | 0.3035 | -0.1291 | 0.2875 |

## Reading (pooled)

- The mechanism claim is unchanged and stronger: at N=30 and 2x horizon the
  lagged arm keeps teacher-student u-rank at ordinary level (30/30 vs hybrid,
  CI excludes zero) while the live-critic hybrid still collapses it.
- The return question stays open in the same direction as N=10: the pooled
  point estimate is negative (-44.6), 15/30 seeds favor lagging, and the 95%
  CI [-102.6, +10.2] is about half the N=10 width. A return payoff claim would
  need a benchmark where longer horizons reward better world models (the
  deferred modality choice), not more seeds on DelayedBimodal.

## Registration hygiene (extension)

- Amendment registered (2026-09-03) before seeds 10..29 were launched.
- Seeds 0..9 rows are identical to the published N=10 file; no arm or seed
  dropped post-hoc; endpoints unchanged; pooled set adjudicates.
