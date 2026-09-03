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
