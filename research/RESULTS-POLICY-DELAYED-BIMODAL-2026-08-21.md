# Delayed-Bimodal Policy Ablation (2026-08-21)

## Purpose

This experiment tests whether uncertainty-preserving distillation remains useful
when the distilled world model is used inside an MBPO-style policy learner. It
compares ordinary member-wise distillation, state-space geometry preservation,
and the hybrid loss that also matches downstream value disagreement and value
variance.

## Protocol

- Environment: `DelayedBimodal-v0`, a 5D Markov continuous-control task with
  action-dependent bimodal impulses delayed by four steps.
- Arms: `ordinary`, `state_geometry`, `hybrid`.
- Seeds: 0, 1, 2.
- Same architecture, replay collection seed, teacher pretraining budget,
  student budget, SAC hyperparameters, rollout schedule, and evaluation seeds.
- One teacher is pretrained per seed and its exact state dictionary is loaded
  into all three arms. The output JSON records the checksum gap.
- Teacher: 4 diffusion sampling steps. Student: one consistency step.

## Results

The run is stored in
`runs/delayed_bimodal_policy_paired_3seed_360.json`.

| Arm | Final return mean +/- SD | Next-state MSE | Selective rank correlation | Selective recall of bad transitions |
|---|---:|---:|---:|---:|
| Ordinary | -180.84 +/- 17.24 | 0.1701 | -0.5482 | 0.0909 |
| State geometry | -175.08 +/- 25.20 | 0.1656 | -0.5341 | 0.1087 |
| Hybrid | -181.53 +/- 21.93 | 0.1669 | -0.5559 | 0.0600 |

Teacher checksum gaps were exactly zero across arms for each seed. The
empirical perturbed-variance inequality had 100% satisfaction on all evaluated
states; this validates the algebra and implementation of the bound, not the
assumptions needed for a formal theorem.

## Interpretation

This policy-level run does **not** establish a broad performance improvement.
State geometry slightly improves mean return and one-step MSE, while the hybrid
arm is not better on the selected-policy uncertainty metrics. The result is
therefore inconclusive/negative for the claim that the hybrid loss automatically
improves downstream control under this short training budget.

The controlled multimodal stress benchmark remains the main positive result:
hybrid distillation reduced uncertainty RMSE on every one of ten confirmatory
seeds and improved top-decile recall on most seeds. The policy experiment shows
that this effect does not transfer automatically to a learned MBPO policy.

## Important metric limitation

The debiased MC-UBE estimator used by the current diagnostic can produce
negative `u` values. Consequently, raw UBE calibration and selective metrics
should not be presented as final evidence until the estimator is evaluated with
an explicitly non-negative uncertainty quantity (for example `max(u, 0)` or a
variance parameterization whose non-negativity is guaranteed). The theorem
diagnostics use member-value variance and remain well-defined.

## Current claim

The defensible claim is narrow: on a controlled multimodal prediction stress
test, hybrid decision-aware distillation improves uncertainty magnitude fidelity;
on a delayed-bimodal MBPO task, the benefit is not yet demonstrated and may be
metric- or optimization-dependent.
