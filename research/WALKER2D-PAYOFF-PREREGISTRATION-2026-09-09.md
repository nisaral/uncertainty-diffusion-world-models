# Walker2d payoff attempt (Hypothesis B) - staging registration (2026-09-09)

**Status: STAGED - config + this registration only. Nothing runs until the
Hypothesis-A (CRN-bias) probe resolves, because the arm list depends on the
fix (if any) that comes out of it.** Hypothesis B asks whether the
mechanistically-general identified fix pays off in return when the
environment does not share hopper-hop's structure.

## 1. Hypothesis-B motivation

hopper-hop-v0's reward is dominated by survival: most seeds sit at ~0 return
and falls terminate without reset, so the return axis on Hopper is
floor-bound at the 15k budget (measured: all per-arm return medians 0.013-
0.075 at 15k; means driven by a few non-floor seeds). Walker2d
(`dm_control/walker-walk-v0`) does not share that zero-on-fall-without-reset
structure, giving the return-payoff question a fairer test. This is the
registered conditional 30k-return-thread environment extension, staged as its
own config so it never mixes rows with the Hopper files.

## 2. Config

`configs/dmc_walker2d_distill.yaml` - cloned from the registered Hopper
probe config (`configs/dmc_hopper_probe.yaml`) with ONLY the task id swapped
(`dm_control/walker-walk-v0`, `max_episode_steps: 1000`) and its own
log/checkpoint paths. All other arms/knobs are byte-identical to the Hopper
probe config. No new shell script: `dmc_payoff.sh`'s staged structure is
reused directly with the task/config swapped.

## 3. Arms (provisional, fixed at launch by this queue)

At minimum: `ordinary`, `identified_eq`, and whichever arm Hypothesis A's
fix produces if it is ready by launch time (e.g. `eq_crn` after the CRN-bias
probe confirms and the Wu-style correction is registered). The final arm list
is locked in an addendum to THIS document at launch, before any row runs.

## 4. Staged budget discipline (do not skip stages)

1. Gate check at the registered protocol budget (re-measure g*/w* and the
   u-rank floor; Hopper's numbers are NOT transferable - Walker2d's
   convergence rate differs).
2. Budget sanity probe (seeds 0-1) exactly as the Hopper Amendment-2 path:
   the fixed-step budget is only registered for adjudication if baseline
   u_rank at that budget clears the same ~0.70 operating-point bar Hopper
   required at 15k; otherwise the budget is amended the same way (probe
   first, amendment second, never the reverse).
3. Full adjudication at the registered budget with the registered arms.

## 5. Endpoints (registered at launch, pre-committed shape)

The same endpoint set as the DMC payoff study: u_rank_corr/w_rmse/g-ratio
mechanism endpoints plus final_return on the return thread. Return is
adjudicated HERE (unlike Hopper at 15k) only if the Walker2d return
distribution is not floor-bound at the registered budget - the pre-committed
floor rule from the DMC study (return medians ~0, means driven by a few
seeds -> defer) applies unchanged.

## 6. Sequencing note

This item is AFTER the CRN-bias probe in the queue because its arm list
depends on the fix Hypothesis A produces. Registering the config + staging
doc now is deliberate: it is the only zero-compute preparation that must
happen before the probe resolves.
