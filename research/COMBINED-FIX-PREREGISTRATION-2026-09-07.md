# Combined-fix preregistration: lagged critic + equal-weight identified loss (2026-09-07)

**Status: registered BEFORE any row of this study runs.** Hypothesis H1 from
`upgrade-plan-hypotheses-experiments-publication.md`. Data file written:
`runs/policy_combined_fix_10seed.json` (fresh full 7-arm table per
Amendment 1; the original merge-with-existing design was abandoned after the
registered bit-exactness gate failed).

## Why this run exists

Two fixes are independently confirmed: (a) lagging the value-map critic fixes
the nonstationarity failure of decision-term distillation
(`lagged_hybrid` u-rank 0.955 ~ ordinary 0.939, N=30, 30/30 vs hybrid);
(b) equal-weight identified (M=2 split, no EMA) fixes the identifiability
failure that the EMA-reweighted variant introduced (`identified_eq` u-rank
0.835, 9/10 >= 0.70, beats `hybrid` 10/10 on DelayedBimodal). Every past run
tested them separately, or combined lag with the BROKEN EMA reweighting
(`lagged_identified`, run before EMA starvation was diagnosed). The 2x2 cell
{lagged critic} x {equal-weight identified} has never been run. This study
runs it: `lagged_identified_eq`.

## Protocol

- Environment/config: DelayedBimodal-v0, `configs/delayed_bimodal_distill.yaml`
  (identical to the corrected-weight N-study; 1,800 env steps, live MBPO/SAC).
- Seeds 0-9; CPU; split-by-seed driver `run_policy_2x2_split_seeds`;
  new rows merged with `runs/policy_corrected_weights_10seed.json` so every
  seed has its 6 prior arms plus the new arm, all under one shared teacher
  (exact per-seed teacher checksum pairing, gap 0.0 required).
- Variant `lagged_identified_eq` = `identified_eq` fields (M=2, aleatoric
  weight 1.0, `distill_reweight_ema: false`) + the lagged fields of
  `lagged_hybrid`/`lagged_identified` (`distill_use_target_critic: true`,
  `distill_normalize_values: true`, `distill_guard_enabled: true`,
  `distill_value_warmup_updates: 50`).
- Bit-exactness gate: before the new-arm rows run, one existing row (seed 0,
  `ordinary`) is re-run under the same driver/env; if its
  `teacher_final_checksum` does not byte-match the existing row, the run is
  aborted (pairing would not be exact) and a fresh full 7-arm table is
  required instead.

## Pre-registered prediction (upgrade plan H1, written before rows exist)

`lagged_identified_eq` u-rank >= `lagged_hybrid` AND >= `identified_eq`,
ideally approaching or exceeding `ordinary`. If the two mechanisms are
independent (nonstationarity and identifiability are separate failure modes),
the combined arm closes the remaining eq-vs-parity gap (~0.10-0.12 on
u-rank) and "partial transfer" becomes "full transfer once both known failure
modes are addressed".

## Bars (N=10, repo conventions: wins >= 7/10 AND 5,000-draw bootstrap 95% CI excludes 0)

Primary endpoint: `u_rank_corr`.
1. `lagged_identified_eq` - `lagged_hybrid` u_rank confirmed >= (does the
   identifiability correction add anything on top of lagging?).
2. `lagged_identified_eq` - `identified_eq` u_rank confirmed >= (does lagging
   add anything on top of the equal-weight correction?).
3. `lagged_identified_eq` u_rank >= 0.70 on >= 7/10 seeds (level bar, same
   as the corrected-weight study's bar 2).
Reference contrasts (reported, not bars): vs `ordinary`, vs `hybrid`, vs
`identified_hybrid` (EMA control).
Secondary endpoints: `u_rmse`, `w_rmse`, `next_state_mse`, `final_return`,
`selective_recall_bad`, `selective_rank_corr`.

## Branch readings

- A: bars 1 AND 2 met -> fixes compose; combined reaches parity territory.
- B: bar 2 only -> lagging is the binding fix; eq adds nothing once
  nonstationarity is fixed (identifiability correction redundant in policy).
- C: bar 1 only -> eq is the binding fix; lag adds nothing to equal-weight
  identified (unexpected, given lag fixes the naive loss).
- D: neither bar 1 nor 2 -> composition fails; the two fixes interact
  negatively.

Any verdict is at N=10 (CPU); the DMC 30-seed adjudication remains the
cross-environment verdict, per the registered DMC protocol.
## Amendment 1 (2026-09-07, BEFORE the adjudication rows ran)

Bit-exactness gate executed as registered: seed 0 `ordinary` re-run under the
same driver/env. `teacher_initial_checksum` byte-matches the 2026-09-05 row
(4859.354153081775) but `teacher_final_checksum` does not (new 4886.055... vs
existing 4890.686...); the arm-phase teacher walk is reproducible within a
batch but not across process runs on this host. Per the registered rule, the
merge-with-existing design is ABANDONED: the study runs a **fresh full 7-arm
table** (`ordinary`, `hybrid`, `lagged_hybrid`, `identified_hybrid`,
`identified_eq`, `identified_wonly`, `lagged_identified_eq`) x seeds 0-9 in
one driver batch, so every seed's arms share one prepared teacher and one
process (internal exact pairing, gap 0 required). Output:
`runs/policy_combined_fix_10seed.json` (70 rows). Cross-check against the
2026-09-05 file is by seed-paired means, not by merged-row pairing. Bars and
branch readings above are unchanged; the 6 existing arms now double as an
internal reproduction check of the corrected-weight N-study.