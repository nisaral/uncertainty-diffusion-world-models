# Payoff test (#4) — does lagging alone pay off? Pre-registration (2026-09-03)

**Status:** registered BEFORE running. Follows the policy 2×2 N=30 verdict
(decision tree **Row 1**: `lagged_hybrid` restores u-rank to ~ordinary 30/30;
the identified loss does not transfer). The tree says the payoff test should
test **lagging alone**, not the identified machinery, so the identified arms are
excluded from this registration.

## Question

At a horizon long enough for the policy to exploit the distilled world model,
does the mechanism the tree confirmed (target-critic lagging, which preserves
teacher-student uncertainty rank) produce measurable downstream payoff on this
benchmark, compared with ordinary member distillation?

## Protocol (fixed in advance)

- Environment: `DelayedBimodal-v0`, **3,600 env steps** (2× the N=30
  adjudication horizon; 1,800 steps produced null/negative return deltas for
  every decision arm, so the payoff question needs more room).
- Arms (exact per-seed teacher pairing, gap 0): `ordinary` (baseline),
  `hybrid` (the falsified live-critic coupling), `lagged_hybrid` (the fix).
- Seeds: 0..9 (10). Same code/config as the N=30 run
  (`configs/delayed_bimodal_distill.yaml`), executed per-seed and merged
  (driver: `udwm.scripts.run_policy_2x2_split_seeds`).
- Endpoints (all recorded per arm by the runner): `final_return` (primary
  payoff), `u_rank_corr` (mechanism survival check at the longer horizon),
  `w_rank_corr`, `u_rmse`, `w_rmse`, `next_state_mse`,
  `selective_rank_corr`, `selective_recall_bad`.

## Predictions

1. **P1 (mechanism survives the longer horizon):** `lagged_hybrid` u-rank stays
   at ~ordinary (mean delta `lagged_hybrid - ordinary` not worse than -0.02)
   and beats `hybrid` (delta > +0.1 with CI excluding 0 preferred).
2. **P2 (payoff, primary):** `final_return` of `lagged_hybrid` vs `ordinary`.
   No direction is pre-judged; this is the measurement.
3. **P3 (control):** `hybrid` remains worst on the uncertainty endpoints.

## Adjudication (fixed in advance)

- Stats: mean paired delta vs `ordinary`, wins/10, 5,000-draw percentile
  bootstrap 95% CI (same conventions as the 2×2).
- P1 confirmed: CI on (`lagged_hybrid - ordinary`) u-rank excludes a -0.02
  harm AND >= 7/10 seeds not worse than -0.02.
- P2: "payoff confirmed" = mean return delta > 0, CI excludes 0, >= 7/10 seeds
  better; "no payoff on this benchmark" = opposite confirmed; else
  inconclusive, reported as such.
- Return is evaluated at 10 eval episodes/seed as recorded by the runner; no
  arm is dropped post-hoc.

## Cost note

CPU-only (small MLPs + 4-NFE diffusion teacher). ~3 arms x 10 seeds x 3,600
steps; per-seed subprocesses at 2 torch threads each, 10-way parallel on a
20-core box: ~20 min wall. No GPU required; the script is the same on any
Linux box with the requirements installed.

## Run commands

Serial equivalent (one batch, exact per-seed teacher pairing):
```bash
python -m udwm.scripts.run_delayed_bimodal_policy_ablation \
  --variants ordinary hybrid lagged_hybrid \
  --seeds 0 1 2 3 4 5 6 7 8 9 --steps 3600 \
  --out runs/payoff_lagged_10seed_3600.json
```
Split-by-seed (recommended on multi-core):
```bash
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --variants ordinary hybrid lagged_hybrid \
  --seeds 0 1 2 3 4 5 6 7 8 9 --steps 3600 \
  --out runs/payoff_lagged_10seed_3600.json --jobs 10 --threads 2
```


---

## Extension registration (2026-09-03): N=10 -> N=30, precision on the pooled set

**Registered before running seeds 10..29.** The N=10 slice is already reported
(`research/RESULTS-PAYOFF-LAGGED-2026-09-03.md`); this extension exists to
sharpen the two open quantities, not to re-adjudicate them.

- Same arms (`ordinary` / `hybrid` / `lagged_hybrid`), same config
  (`configs/delayed_bimodal_distill.yaml`), 3,600 env steps, exact teacher
  checksum pairing per seed (gap 0).
- **N = 30 seeds total (0..29).** Seeds 0..9 are the published 10-seed file
  (`runs/payoff_lagged_10seed_3600.json`); seeds 10..29 run with the identical
  code and protocol via `udwm.scripts.run_policy_2x2_split_seeds` and are
  merged into a pooled file (`runs/payoff_lagged_30seed_3600.json`).
- **Adjudication on the pooled 30-seed set**, same endpoints as the N=10
  registration (P1 mechanism: `u_rank_corr`; P2 payoff: `final_return`; P3
  control), same 5,000-draw percentile bootstrap, seed-count rule generalised:
  - P1 confirmed: mean delta (`lagged_hybrid` - `ordinary`) u-rank not worse
    than -0.02 with >= 21/30 seeds not worse than -0.02, AND
    `lagged_hybrid` beats `hybrid` with CI excluding 0 (preferred).
  - P2: "payoff confirmed" = mean return delta > 0, CI excludes 0, >= 21/30
    better; "no payoff on this benchmark" = opposite confirmed (>= 21/30
    worse or CI excludes positive); else inconclusive, reported as such.
  - P3 confirmed: `hybrid` worst on the uncertainty endpoints.
- Both pooled and extension-only (seeds 10..29) summaries are reported; the
  pooled set adjudicates. No arm is dropped post-hoc; no endpoint added.
