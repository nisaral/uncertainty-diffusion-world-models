# Research overview

**Keyush Nisar** · code: [uncertainty-diffusion-world-models](https://github.com/nisaral/uncertainty-diffusion-world-models)

## Question

Can diffusion world-model distillation preserve **decision-relevant epistemic
uncertainty**, rather than only average transition accuracy, under a **fixed
inference budget**?

## What the evidence currently says

**Yes, under a fixed value map.** On a 20-seed controlled stress study (exact
teacher pairing, five members, 1,024-state grid), hybrid decision-aware
distillation beat ordinary member matching on uncertainty RMSE on 20/20 seeds
(mean Δ −0.00374, bootstrap 95% [−0.00474, −0.00289]) and on rank correlation
on 13/20 (mean Δ +0.0317, 95% [+0.00117, +0.0632]). Top-decile recall was
inconclusive.

**No, if the value map is the online critic.** On a 10-seed, 1,800-step policy
study, the same hybrid loss improved next-state MSE on 10/10 seeds and
**worsened** teacher–student uncertainty rank and RMSE on 10/10 (mean rank
Δ −0.2968, mean RMSE Δ +180.4). Return improved on only 3/10.

So the publishable claim is conditional:

> Decision-aware distillation preserves downstream uncertainty under a fixed
> or separately validated value map. Online critic-derived targets are
> nonstationary and unsafe.

## What we are building next

Lagged **target-critic** distillation with normalized value targets, a critic
warmup, and a minibatch **guard** that drops decision terms when student
uncertainty is anti-aligned or scale-exploded. That method is implemented
(`configs/lagged_target_distill.yaml`). It does not yet have a 10-seed table.

## What this is not

Not a new UBE, not conformal coverage, not first gated imagination (MACURA
and earlier work), not a large-scale control result.

## Limits

Luis Assumptions 1–2 fail as in Luis’s own deep-RL instantiation.
\(\sqrt{U}\) is a decision score, not a calibrated interval.
