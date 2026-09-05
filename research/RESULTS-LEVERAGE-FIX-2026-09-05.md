# Leverage-fix experiment: is the aleatoric starvation a step-size artifact? (2026-09-05)

**Registration:** `research/LEVERAGE-FIX-PREREGISTRATION-2026-09-05.md`
(written before any arm ran; A-arm results and the B-arm plan were appended
before B ran).
**Data:** `runs/probe_u_lrscale_*.json` (same 1,800-step policy protocol,
exact teacher pairing, CPU, seeds 0-1).
**Question.** The U-collapse mechanism doc measured the aleatoric term's
per-update leverage on the student's g at ~2e-6, ~1e6-1e7 updates short of the
gap in a 180-update run. Does fixing the leverage directly - larger effective
step size for the aleatoric channel, replacing the EMA reweighting - resolve
the collapse?

## Arms and results (seed 0; key arms replicated on seed 1)

All arms: identified loss, M=2 latents (B3: M=8), live critic, no value
normalization, `distill_reweight_ema=false`. Control = equal-weight identified
(lambda_g = 1). A arms scale the aleatoric loss by lambda_g at the default
gradient clip; B arms scale the student learning rate with an
aleatoric-dominated gradient (lambda_g = 1e4) and a raised clip so the
gradient may move. `s_ds` = student latent-to-state spread over 8 paired
pure-noise latents (teacher ~0.36-0.38 everywhere).

| arm | seed | lambda_g | student_lr | clip | M | student_u_mean | teacher_u_mean | student g | s_ds | u_rank | next_state_mse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 1 | 1e-3 | 10 | 2 | -0.0049 | -86.1 | 0.0133 | 0.0060 | +0.23 | 0.166 |
| control | 1 | 1 | 1e-3 | 10 | 2 | -0.0090 | -81.9 | 0.0143 | 0.0062 | +0.17 | 0.158 |
| A1 | 0 | 1e2 | 1e-3 | 10 | 2 | -0.0005 | -97.2 | 0.0140 | 0.0057 | -0.11 | 0.153 |
| A2 | 0 | 1e4 | 1e-3 | 10 | 2 | -0.0056 | -81.6 | 0.0121 | 0.0060 | +0.15 | 0.149 |
| A3 | 0 | 1e4 | 1e-3 | 1e3 | 2 | -0.0026 | -76.1 | 0.0129 | 0.0061 | +0.07 | 0.157 |
| B1 | 0 | 1e4 | 1e-2 | 1e3 | 2 | +0.0088 | -87.7 | 0.0029 | 0.0014 | -0.05 | 0.167 |
| B1 | 1 | 1e4 | 1e-2 | 1e3 | 2 | +0.0088 | -67.9 | 0.0017 | 0.0009 | -0.15 | 0.167 |
| B2 | 0 | 1e4 | 1e-1 | 1e3 | 2 | +0.0240 | -74.4 | 0.0000 | 0.0000 | -0.23 | 0.173 |
| B3 | 0 | 1e4 | 1e-2 | 1e3 | 8 | +0.0162 | -116.0 | 0.0020 | 0.0015 | -0.27 | 0.157 |

## Verdict (pre-registered rule, outcome 2)

**No arm closes the gap. Leverage starvation is not solely a step-size
artifact.** Every configuration leaves student_u ~ 0 (or slightly positive)
versus teacher -68..-116, and student g ~ 1e-2 or *smaller*.

Two clean negative results, both replicated on seed 1 where run:
- A1-A3: scaling the aleatoric loss by lambda_g in {1e2, 1e4} and raising the
  gradient clip to 1e3 does not move g (student g 0.012-0.014, delta-spread
  ~0.006). Under Adam's per-coordinate normalization, a common loss scale is
  (to first order) inert - the EMA reweighting used in the policy study is
  therefore not the operative dial either.
- B1-B3: giving the aleatoric-dominated gradient a 10x (B1) or 100x (B2)
  larger student learning rate drives g **down**, not up: g falls to 0.003,
  then to exactly 0.0 with delta-spread 0.0000 (the student output is exactly
  latent-independent) at lr=0.1. M=8 latents (B3) does not help either.

## Reading

The aleatoric channel sits at its non-negativity boundary (g ~ 0). Near that
boundary the first-order aleatoric signal is real but tiny (single-batch step
response +2.3e-6 at the end state; see the mechanism doc), while the 2-latent
training-time estimator noise is ~100x larger. Gradient descent therefore
accumulates mostly noise; the noise pushes the student to the g=0 boundary,
where the within-member variance is exactly flat and the gradient vanishes.
Larger steps make the noise-driven walk stronger, so the student parks exactly
at g=0 (B2). Two structural constraints, not a step size, bind:

1. The aleatoric statistic is estimated at training-time corruptions whose
   noise magnitude is <= ~28% of the data scale (8-step schedule), where the
   student's two draws are nearly identical: low signal, high relative noise.
2. The quantity that matters at eval is the pure-noise (t=1) latent spread,
   which training-corruption gradients do not directly grow.

## What this changes

- The thread is closed as a diagnosis + a confirmed dead end for the naive
  fix: the identified loss's aleatoric failure in policy is not curable by
  reweighting, clip, learning-rate, or M scaling within this objective; the
  student must be given the pure-noise spread as the training target (or the
  schedule/parametrization changed) for the aleatoric channel to have
  leverage. Suggested next probe (if pursued): evaluate the w/g terms at
  t_scaled = 1 pure-noise corruptions so the aleatoric gradient acts on the
  object that decides the eval endpoint.
- The DMC-scale pre-registration (future work) should **not** carry the
  lambda_g/lr lever as a promising arm; if the identified loss is run there,
  the value map and the w/g ratio should be checked first (this failure needs
  g >> w, which the SAC critic produces on DelayedBimodal).
- This is a same-scale negative that completes the causal chain from the
  mechanism doc: identifiable-in-theory -> ineffective-in-policy -> why
  (aleatoric starvation) -> naive leverage fixes do not help (measured, 2
  seeds) -> the remaining lever is the objective's corruption distribution.## Addendum (2026-09-05, post-hoc): config-override bug invalidates the arm labels

**Correction (see `RESULTS-CORRUPTION-2026-09-05.md`).** The probe applied
`--set` overrides before `make_cfg`, and `VARIANTS["identified_hybrid"]`
re-applied `distill_reweight_ema=True` and `distill_aleatoric_weight=1.0`
afterwards. Every arm in this document therefore ran **EMA-both reweighted
with the variant-default aleatoric weight**, regardless of the `--set`
labels: "control = equal-weight" was actually EMA-on; "lambda_g = 1e4" was
aleatoric weight 1.0; "M = 8" (B3) was M = 2 (`distill_m_latents` is also in
`VARIANTS`). Only `distill_grad_clip` and `--student-lr` (set directly on the
optimizer) varied as documented. The probe is fixed (overrides applied after
`make_cfg`); fresh corrected runs in `RESULTS-CORRUPTION-2026-09-05.md` show:

- Equal-weight identified (schedule corruptions) **recovers g** (u_rank
  0.885/0.920, student g within ~1.2x teacher) on the live critic: the
  aleatoric starvation measured here was the EMA-down-weighted channel.
- EMA-both and w-only-EMA both collapse g to ~0.01 (the epistemic up-weight
  `w_scale ~ 1e5-1e6` is the driver, not the aleatoric down-weight).
- The B-arm endpoint (student LR 0.1 + clip 1e3 -> g exactly 0) replicates
  across seeds under the corrected attribution (EMA-on config); see the N=10
  table in `RESULTS-CORRUPTION-2026-09-05.md`.

The verdict of this document ("not solely a step-size artifact; the remaining
lever is the corruption distribution") is superseded: under equal weights the
corruption distribution is not the binding constraint either (g recovers at
schedule/maxt/pure alike). The binding constraint found in this whole thread
is the **per-term EMA reweighting ratio** (~(g*/w*)^2) that starves the
aleatoric channel when the map is aleatoric-dominated, and the equal-weight
hole on w when it is not - i.e. a balance-window problem, consistent with the
toy (`theory/identified_balance_window.py`).
