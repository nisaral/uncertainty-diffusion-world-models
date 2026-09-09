# Compute-normalized re-analysis - methods note (2026-09-09)

**Status: methods note written before the re-analysis script runs on the
adjudicated files.** This is a pure re-analysis of already-adjudicated rows
(`runs/dmc_payoff_30seed_15k_gpu.json`, `runs/policy_combined_fix_10seed.json`);
no new training, no new compute spent. Per project discipline it fixes the
question and the normalization rule first.

## 1. Question

The DMC 30-seed verdict compared arms at a FIXED env-step budget (15k). The
identified family (`M=2` paired latents) and the hybrid family (`M=1`, all
members) spend more teacher-sample compute per training step than `ordinary`
(which distills a single random member). The re-analysis question: does the
arm ordering change when arms are compared at fixed *teacher-sample compute*
instead of fixed env steps - specifically, does `identified_eq`'s apparent
underperformance narrow or persist when its extra per-step cost is
accounted for?

## 2. Normalization rule (fixed before looking at results)

Teacher-sample-equivalent units = ONE ordinary model-train epoch's teacher
cost (one teacher-member + one student-member diffusion forward on one
minibatch of `model_batch_size` states). Cumulative units at step s =
(model-train calls up to s) x (`num_model_epochs` = 12) x per-epoch multiplier.

Multipliers are read from the loss implementation in
`udwm/models/consistency.py`, not assumed:

| family | loss fn | teacher-member fwds / epoch | value_fn calls / epoch |
|---|---|---|---|
| ordinary | `distill_loss` (1 random member) | 1 | 0 |
| hybrid / lagged_hybrid | `decision_preserving_distill_loss` (all N=5 members, M=1) | 5 | 2 x (N x B states) |
| identified_eq / lagged_identified_eq / identified_hybrid (+norm/nonorm cells) | `identified_*_loss`, schedule corruption (all N members, M=2 latents) | 10 | 4 x (N x B states) |

The user-level assumption "identified uses ~2x ordinary/hybrid's teacher cost"
is only half right against the code: identified is 2x the *hybrid family*
(all-member, M=1) but 10x `ordinary`, because `ordinary` distills one random
member per epoch while every decision arm runs all N members. The M=2 factor
is relative to the all-member M=1 pass, not to ordinary.

Critic value_fn calls are NOT part of the unit (they are small critic/actor
MLP state evaluations with no teacher-sample interpretation); their raw counts
are printed by the script so the basis is auditable.

Fresh-eval milestone extraction follows the registered forward-fill
convention: the LAST `eval_history` record at each duplicated step is the
fresh eval (the trainer appends `log(step)` then `eval(step)` at eval_freq
multiples); row-level final metrics form the terminal point when no history
record exists at the final step.

## 3. Pre-committed reading rules

- Any fixed-compute comparison must be read on the COMMON teacher-sample
  domain where every arm has measured eval data; interpolation outside the
  observed domain is extrapolation and is NOT reported.
- If the common domain is empty at this protocol (cheap arms exhaust their
  budget before the expensive arms' first eval), the fixed-compute head-to-head
  is reported as NOT READABLE at 15k, and the gap is quantified on the
  efficiency axis instead (u_rank per 1000 teacher-sample units, measured
  points only). Closing the gap is then staged for the conditional 30k
  extension with an early eval checkpoint - it is not patched by
  extrapolating.
- This re-analysis NEVER changes the 2026-09-08 DMC verdict or any
  registered bar; it only adds the compute-normalized view.
