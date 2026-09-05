# Leverage-fix preregistration: can the aleatoric channel be un-starved? (2026-09-05)

**Status: registered before any arm of this experiment ran.**

**Question.** The U-collapse mechanism doc
(`U-COLLAPSE-MECHANISM-2026-09-05.md`) attributes the identified-arm collapse
to gradient starvation of the aleatoric channel: the measured per-update
leverage of the aleatoric term on the student's g is ~2e-6 (matched-draw step
response at lr=1e-3), ~100x below its own 2-latent estimator noise and
~1e6-1e7 updates short of closing the g-gap, in a run that performs 180
student updates. Diagnosis: the aleatoric loss term is effectively dead even
at weight 1 (g-only cell). Natural next question (reviewer thread): **if the
aleatoric gradient is given a directly larger effective step size - per-term
learning-rate/scale scaling proportional to inverse leverage, replacing the
EMA reweighting - does the collapse resolve?**

**One-sentence hypothesis (as proposed):** if per-term learning-rate scaling
proportional to inverse leverage does not close the g gap within the 180-update
run, leverage starvation is not solely a step-size artifact.

**Protocol.** Reuse the exact policy protocol of the probe
(`udwm/scripts/probe_u_collapse.py`, 1,800 env steps, 15 model-train calls x 12
epochs = 180 student updates, exact matched teacher per seed, CPU, default
threads). Arm configuration: `identified_hybrid` base (M=2 latents, live
critic, no value normalization) with `distill_reweight_ema=false`
(equal-weight baseline) and the aleatoric term scaled by lambda_g
(`distill_aleatoric_weight`). A new trainer knob `distill_grad_clip` (default
10.0, unchanged from current behavior) allows raising the gradient clip so the
aleatoric-dominated gradient is not capped at the current 10-norm limit, which
alone would cap per-step parameter movement at lr*10 = 1e-2 and make closing
the gap impossible regardless of lambda_g.

**Arms (seed 0 first; seed 1 for any arm that passes the resolution bar or
shows movement):**
- control: lambda_g = 1, clip = 10 (equal-weight identified; already measured
  to collapse, re-run for a same-run teacher)
- A1: lambda_g = 1e2, clip = 10
- A2: lambda_g = 1e4, clip = 10
- A3: lambda_g = 1e4, clip = 1e3
- A4: lambda_g = 1e6, clip = 1e3
- A5: lambda_g = 1e4, clip = 1e3, m_latents = 8 (extra per-batch signal axis)
Exploratory, decided before running: if A3/A4/A5 destabilize the student
(next_state_mse > 2x ordinary's 0.28 or non-finite), the clip/lambda ceiling
is recorded as part of the finding, not dropped.

**Endpoints (decided before running).** Primary: eval-time student_u_mean
relative to teacher on the same run - "gap closed" if
student_u_mean/teacher_u_mean in [0.5, 1.5] (both negative) AND
u_rank_corr >= 0.7. Secondary: student g mean within 2x teacher g; student
latent-to-state spread (delta_spread) >= 0.15; student w_rmse does not blow up
(<= 0.1); next_state_mse not degraded beyond 2x ordinary's arm value.

**Adjudication.** Diagnostic, N=1-2 seeds, no seed-bar statistics. Outcomes:
(1) an arm closes the gap -> leverage starvation is (at least partly) a
step-size/scale artifact, and the DMC pre-registration should carry that
variant as an arm; (2) no arm closes the gap while the student stays stable ->
starvation is not solely a step-size artifact (e.g. the 2-latent estimate
signal-to-noise or the member-anchor cost dominates), a further finding;
(3) arms close the gap only by destabilizing the student -> the trade-off is
recorded as the finding. Either way, the thread is complete after this
experiment and the write-up is updated.

**Files.** Data: `runs/probe_u_lrscale_*.json`. Write-up appended to
`U-COLLAPSE-MECHANISM-2026-09-05.md` (new section) or a follow-up doc.
## Amendment (registered before the B arms ran)

A1-A3 (seed 0) ran and all collapsed identically to the control
(student_u ~ -0.001..-0.006, student g ~ 0.012-0.014, delta-spread ~ 0.006):
lambda_g in {1e2, 1e4} and clip in {10, 1e3} do not move g. Under Adam the
per-coordinate update normalizes by the gradient RMS, so loss-scale lambda and
the global clip are (to first order) inert; the operative levers are the
student learning rate and the per-batch estimator signal-to-noise (M). The
lambda/clip arms are therefore recorded as a negative control and the B arms
below test the real levers, decided before running them:
- B1: lambda_g = 1e4 (aleatoric-dominated gradient direction), student_lr =
  1e-2, clip = 1e3, M = 2
- B2: lambda_g = 1e4, student_lr = 1e-1, clip = 1e3, M = 2 (stability
  boundary probe)
- B3: lambda_g = 1e4, student_lr = 1e-2, clip = 1e3, M = 8 (signal-to-noise
  axis)
Same endpoints and resolution bar as the A arms. If B1-B3 do not close the
gap while the student stays stable, leverage starvation is not solely a
step-size artifact: the binding constraint is the coherent-signal accumulation
rate (2-latent estimator noise ~100x the per-step signal over a 180-update
run).
