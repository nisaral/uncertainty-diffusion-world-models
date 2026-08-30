# Research overview

**Keyush Nisar** · code: [uncertainty-diffusion-world-models](https://github.com/nisaral/uncertainty-diffusion-world-models)

## Question

Can diffusion world-model distillation preserve **decision-relevant epistemic
uncertainty**, rather than only average transition accuracy, under a **fixed
inference budget**?

## What the evidence currently says

The hybrid loss matches teacher–student value disagreement at **one shared
latent**. That statistic is \(w^\star+(g^\star-\bar\Sigma)\): epistemic
disagreement plus a latent-conditional remainder. One equation, two unknowns.
A student with **zero** ensemble disagreement can still match it.

That is why two tables look opposite and are not:

- **Fixed value map, 20 seeds.** Hybrid beats ordinary on uncertainty RMSE
  (20/20, mean Δ −0.00374) and rank (13/20, mean Δ +0.0317). Recall
  inconclusive. Diagnostics: epistemic `w_deb` improves while aleatoric `g`
  degrades — the split is being moved, not preserved.
- **Live SAC critic, 10 seeds × 1,800 steps.** Same loss: next-state MSE
  better 10/10, uncertainty rank and RMSE **worse 10/10**. Return 3/10.

The sign of the epistemic move is a property of the value map. The freedom to
move it is a property of the loss. A lagged target critic addresses
nonstationarity. It does not identify the target. Related shrinkage result:
Voelcker et al., arXiv:2505.22772.

On the same fixed-map study, an identified loss (\(M\ge2\), separate
\(\hat w,\hat g\)) is a **wash** vs hybrid. Theory said a stationary map would
not discriminate. The sharp test is the policy 2×2, not yet run.

## What is implemented next

Identified distillation + optional lagged critic, registered as
`identified_hybrid` / `lagged_identified`. Equal weighting of \(\hat w\) and
\(\hat g\) is a known numerical hole (`g` dominates).

## What this is not

Not a new UBE, not conformal coverage, not first gated imagination, not
policy SOTA.

## Limits

Luis Assumptions 1–2 fail as in Luis’s deep-RL instantiation. \(\sqrt{U}\) is
a score, not a calibrated interval. DelayedBimodal / toy scale only.
