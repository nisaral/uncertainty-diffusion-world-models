# Research overview

**Keyush Nisar** · code: [uncertainty-diffusion-world-models](https://github.com/nisaral/uncertainty-diffusion-world-models)

## Question

I am testing whether **diffusion world-model distillation** can preserve
**decision-relevant epistemic uncertainty**, rather than only average
transition accuracy, under a **fixed inference budget**.

## Why it is not obvious

A multi-step diffusion ensemble can generate imagined transitions, but online
control wants **one or few** network evaluations. Distillation is the usual
fix for that latency.

Uncertainty-aware control does not only need the mean next state. Local UBE
rewards, pessimism, and “may this imagined transition train the policy” all
depend on **ensemble disagreement after a value map**. Ordinary member-wise
MSE can keep next-state error while changing that disagreement. If it does,
the cheap student is not a substitute for the teacher’s uncertainty.

## Method (what is actually implemented)

1. Train a small **diffusion teacher ensemble** on real transitions.
2. Distill a **one-step student ensemble**, either
   - **ordinary:** member-wise teacher matching, or
   - **geometry:** same matching plus ensemble mean, centered member
     deviations, and pairwise squared distances.
3. Evaluate **teacher vs student** Monte Carlo UBE local rewards \(u\) and
   disagreement \(w\) on identical states, with a **fixed** critic, policy,
   and sample budget \(M\).

Primary numbers: rank correlation and MAE / RMSE of \(u\) and \(w\), next-state
MSE, and NFE (teacher multi-step vs student 1).

A common-mode student error can move every member together and leave
disagreement almost unchanged. A member-differential error corrupts \(w\)
directly. That is why the extra loss targets **centered geometry**, not only
mean MSE.

## What would count as a result

- **Supports the hypothesis:** at matched next-state error and matched NFE,
  geometry distillation keeps teacher–student \(u,w\) rankings higher (and
  RMSE lower) than ordinary distillation.
- **Falsifies it:** geometry does not help, or it only helps by sacrificing
  transition accuracy. That is still worth reporting.

## What this is not

- Not a new world-model architecture.
- Not a new Uncertainty Bellman Equation (Luis et al., 2023).
- Not conformal coverage, and not a guarantee.
- Not “first uncertainty-gated imagination” (MACURA and earlier work already
  stop or shorten rollouts when the model looks uncertain).
- Not a large-scale control result. Current experiments are **Pendulum-v1**.

## Status

The distillation losses, freeze-teacher two-stage runner, and teacher–student
uncertainty evaluator are implemented and unit-tested. The matched ordinary
vs geometry study is the next empirical object. The MBPO / SAC loop and
U-gated rollouts exist as a downstream user of the score, not as the claim.

## Honest limits

Luis Assumptions 1–2 fail in this deep-RL setting the same way they fail in
Luis’s own neural instantiation. The ensemble is treated as a discrete
posterior, which is standard and contestable. \(\sqrt{U}\) is a **decision
score**, not a calibrated interval.
