# Corruption-distribution preregistration: does eval-equivalent pure-noise corruption un-starve the aleatoric channel? (2026-09-05)

**Status: registered before any arm of this experiment ran.**

**Question.** The leverage-fix experiment
(`RESULTS-LEVERAGE-FIX-2026-09-05.md`) closed the step-size axis: raising the
student learning rate does not lift `g`; it drives it to the `g = 0`
non-negativity boundary. The remaining lever named there is the **objective's
corruption distribution**: the identified loss evaluates the w/g value
statistics at schedule corruptions whose injected-noise std never exceeds
`sqrt(1 - alpha_bar) ~= 0.28` of the data scale (8-step schedule), where both
the student's and the teacher's per-latent spread is nearly invisible, while
the eval-time object (decompose_ube) is measured at **pure-noise latents
x_T ~ N(0,I)** where the teacher's within-member value spread is ~80 (g/w ~ 1e4
on this map). The hypothesis: if the decision terms are evaluated at the same
pure-noise latents the eval decomposition uses, the aleatoric term's gradient
acts directly on the object that decides the endpoint, and student `g` moves
off the ~0.006-0.016 boundary.

**One-sentence hypothesis (pre-registered).** If moving the w/g terms to
eval-equivalent pure-noise corruptions raises the student's eval-time `g` (and
latent-to-state spread) above the schedule-corruption control while the
student stays stable, the corruption distribution is the binding constraint on
the aleatoric channel and the identified loss can be repaired by evaluating
its decision terms at the eval object's corruption level. If `g` stays at the
boundary, the corruption distribution is not the (only) binding constraint.

**Correction registered with this experiment.** The probe's `--set` overrides
were previously applied to the base config *before* `make_cfg`, and the
`VARIANTS` dictionary re-applies `distill_reweight_ema=True` (and
`distill_aleatoric_weight=1.0`) afterwards, so every historical lever-fix arm
labeled "equal weight" or "lambda_g = 1e4" actually executed with the EMA
reweighting on and the variant-default aleatoric weight (verified in the
saved scale logs: `w_scale ~ 1e12`, `g_scale ~ 2.3e8` at step 300 of B2).
`udwm/scripts/probe_u_collapse.py` is fixed so `--set` is applied to the final
`cfg` after `make_cfg` and therefore wins over the variant defaults. The
leverage-fix doc is amended accordingly (see
`RESULTS-LEVERAGE-FIX-2026-09-05.md` addendum). This experiment's control and
arms are therefore run under the corrected semantics: equal-weight identified
(`distill_reweight_ema=false`, `distill_aleatoric_weight=1.0`, clip default
10, student lr 1e-3), so the only varied factor is the corruption
distribution of the decision terms.

**Protocol.** Exact policy protocol of the probe
(`udwm/scripts/probe_u_collapse.py`, 1,800 env steps, 180 student updates,
exact matched teacher per seed, CPU, default threads). Arm configuration:
`--variant identified_hybrid` + `--set` with
`distill_reweight_ema=false,distill_aleatoric_weight=1.0` and the corruption
mode below. New loss knob `distill_corruption` (default `schedule`,
bit-identical behavior when unset):
- `schedule` (control): decision terms at random schedule corruptions
  (historical behavior);
- `pure`: decision terms at eval-equivalent pure-noise latents x_T ~ N(0,I)
  shared across members; student one-step (t_scaled=1), frozen teacher
  multi-step DDIM decode of the same latent; member/state anchors unchanged;
- `maxt`: decision terms at the maximum schedule timestep (t = T-1, ~28%
  injected noise) with the one-step analytic teacher target.

Arms: `schedule`, `pure`, `maxt` x seeds {0, 1}. Teacher checksum pairing is
per seed and shared across arms (same prepared buffer + teacher as the
existing probe data for that seed).

**Endpoints (decided before running).** Primary: eval-time student `g`
(decomp summary `g.mean`) and the student latent-to-state spread
(`delta_spread_mean`). "Moved off the boundary" = student `g` > 3x the
schedule control's student `g` on the same seed AND `delta_spread_mean` >=
0.10 (teacher ~0.36-0.38). Secondary: student w (not blown up, `w_rmse` <=
0.1), `u_rank_corr` (informative at N=1-2), `next_state_mse` (stability:
must not exceed 2x the ordinary arm's ~0.16-0.17 value), and the per-log-step
student g trajectory (does g rise during the 180 updates, or only at the
end?).

**Adjudication (decided before running).** Diagnostic, N=2 seeds per arm, no
seed-bar statistics (this is a mechanism probe, not a policy claim).
Outcome 1: `pure` moves student g off the boundary while stable -> corruption
distribution confirmed as the binding lever; the identified loss is repaired
by evaluating decision terms at the eval object's corruption level, and the
DMC pre-registration carries the `pure` decision-term corruption as the
identified-loss arm. Outcome 2: `pure` leaves g at/below the schedule
control -> the corruption distribution is not the binding constraint; the
starvation must be explained by something else (e.g. the member anchor or the
optimizer geometry), a further finding. Outcome 3: `pure` moves g only by
destabilizing the student (next_state_mse blow-up) -> trade-off recorded.
The `maxt` arm is a diagnostic ladder point: under the corruption lever,
student g should order `schedule < maxt < pure` if the lever is monotone in
the corruption level seen by the decision terms.

**Files.** Code: `distill_corruption` knob in `udwm/models/consistency.py`
(default `schedule`, schedule path bit-identical). Data:
`runs/probe_u_corrupt_{schedule,pure,maxt}_s{0,1}.json`. Results appended to
this file or a `RESULTS-CORRUPTION-*.md`.
