# Does distillation preserve decision-relevant uncertainty in diffusion world models?

**Working draft — not submission-ready.**  
**Code:** https://github.com/nisaral/uncertainty-diffusion-world-models

## Claim (conditional)

Decision-aware distillation of a diffusion world-model ensemble can preserve
downstream epistemic disagreement **when the value map is fixed or separately
validated**. The same loss, attached to an **online** SAC critic, can keep
next-state accuracy and destroy teacher–student uncertainty. Reliable use
therefore needs a lagged/target critic, target normalization, and an explicit
uncertainty-preservation guard.

## Evidence

See `research/RESULTS-STRESS-LARGE-2026-08-21.md` (20-seed controlled, hybrid
helps RMSE 20/20) and `research/RESULTS-POLICY-SCALE-2026-08-22.md` (10-seed
policy, hybrid hurts uncertainty 10/10 while helping MSE 10/10). Teacher
checksum gaps were exactly zero in both.

## Method sketch

Teacher: \(K\)-step diffusion ensemble. Student: 1-NFE. Local UBE \(u,w\)
estimated by Monte Carlo because members are implicit samplers.

Ordinary loss: member-wise \(x_0\) match. Hybrid: plus state geometry and
value-map disagreement. Lagged correction: value map is `critic_target` with
stop-gradient actions, z-scored on teacher batch statistics; decision terms
are zeroed when a stop-grad fidelity guard fires.

## What is not claimed

A new UBE. Conformal coverage. First gated imagination. Policy SOTA.
