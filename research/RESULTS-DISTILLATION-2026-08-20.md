# Distillation preservation results (2026-08-20)

## Setup

- Environment: Pendulum-v1
- Teacher: diffusion ensemble, hidden dimensions `[128,128]`, 3 members
- Student: `[32,32]` capacity bottleneck
- Data: 1,000 collected transitions per seed
- Teacher training: 10 epochs
- Student training: 20 epochs
- Evaluation: 1,024 held-out state-action evaluations per row, `M=8`
- Seeds: 0, 1, 2
- Paired teacher/student latents during uncertainty evaluation

Raw outputs:

- `runs/distillation_ablation_credible.json` (full-capacity student)
- `runs/distillation_ablation_bottleneck.json` (capacity-bottleneck student)
- `runs/distillation_ablation_bottleneck_exact_teacher.json` (clean paired-teacher rerun)

## Full-capacity student

Both ordinary and geometry students achieved approximately `0.988-0.992`
teacher/student local-U rank correlation. Geometry did not improve uncertainty
RMSE or rank correlation consistently. The runner verdict was
`inconclusive_requires_more_seeds_or_better_metric`.

## Capacity-bottleneck student

Per-seed `u_rank_corr`:

| Seed | Ordinary | Geometry | Geometry - Ordinary |
|---:|---:|---:|---:|
| 0 | 0.9703 | 0.9745 | +0.0042 |
| 1 | 0.9847 | 0.9665 | -0.0182 |
| 2 | 0.9908 | 0.9920 | +0.0012 |

Mean delta: `-0.00425`, so geometry was slightly worse on average.

Per-seed `u_rmse` deltas (geometry minus ordinary):

| Seed | Delta |
|---:|---:|
| 0 | -0.00030 |
| 1 | +0.00137 |
| 2 | +0.00011 |

The runner again returned `inconclusive_requires_more_seeds_or_better_metric`.

## Clean paired-teacher rerun

The previous harness trained the teacher independently inside each arm. The
corrected harness trains one teacher per seed, copies the same replay buffer and
teacher state into both arms, then trains only the students. The resulting
teacher checksum gap is exactly `0.0` for every seed.

| Seed | Ordinary U rank | Geometry U rank | Δ rank | Ordinary U RMSE | Geometry U RMSE | Δ RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.993553 | 0.993339 | -0.000214 | 0.003547 | 0.003650 | +0.000104 |
| 1 | 0.994072 | 0.993729 | -0.000342 | 0.004988 | 0.005145 | +0.000156 |
| 2 | 0.993017 | 0.992979 | -0.000038 | 0.005233 | 0.005243 | +0.000010 |

Across the three paired seeds, geometry-minus-ordinary mean Δ rank is
`-0.000198` and mean Δ RMSE is `+0.000090`. These are negligible in absolute
terms and consistently favor ordinary distillation on this benchmark. The
result is therefore still `inconclusive_requires_more_seeds_or_better_metric`,
but the clean pairing makes the negative direction more credible: there is no
measurable benefit from adding state-space geometry terms to member-wise
distillation on Pendulum-v1 under this protocol.

The paired differences have sample standard deviations of approximately
`0.000153` (rank) and `0.000075` (RMSE), with standard errors `0.000088` and
`0.000043`, respectively (`n=3`). These intervals are descriptive only; three
seeds are not enough for a confirmatory statistical claim.

## Current conclusion

These results do **not** establish that state-space mean/centered/pairwise
geometry preservation improves uncertainty preservation. On this Pendulum
benchmark, ordinary member-wise distillation is already enough to preserve the
teacher's local-U ranking very strongly. The current geometry loss therefore
does not yet justify a paper claim.

This is a useful falsification result: the benchmark is too easy, or the loss is
not targeting the actual failure mode. The next valid pivot is direct
value-disagreement distillation and a stress environment with multimodal,
delayed, or action-sensitive consequences. Do not claim the current method as
positive evidence until it beats the ordinary baseline on those tests.

## Controlled multimodal stress benchmark

Pendulum-v1 was a ceiling benchmark: ordinary distillation already preserved
local-U ranking at approximately 0.993, leaving little room for a geometry loss
to help. The new stress protocol uses a two-dimensional state, action-dependent
bimodal transitions, sparse/OOD evaluation states, independently bootstrapped
teacher members, and a nonlinear downstream value map. All student arms use the
same teacher state, data, initialization, update count, and evaluation latents.

### Exploratory comparison (seeds 0-4)

The direct decision-value objective reduced uncertainty RMSE from `0.00863` to
`0.00410` on average, but did not improve rank correlation. Combining state
geometry and decision-value disagreement (the hybrid objective) improved mean
rank correlation by `+0.01999` and reduced RMSE by approximately `0.00482`.
Because this hybrid was selected after inspecting these seeds, these results are
exploratory rather than confirmatory.

### Held-out confirmation (seeds 5-9)

The pre-specified ordinary-versus-hybrid comparison gave:

| Metric | Hybrid minus ordinary | Wins |
|---|---:|---:|
| Value-disagreement rank correlation | `+0.04636 +/- 0.03653` | 3/5 |
| Value-disagreement RMSE | `-0.00924 +/- 0.00335` | 4/5 |
| Top-decile uncertainty recall | `+0.08312 +/- 0.04547` | 3/5 |

### Additional held-out seeds (10-19)

A second untouched block gave:

| Metric | Hybrid minus ordinary | Wins |
|---|---:|---:|
| Value-disagreement rank correlation | `+0.03645 +/- 0.03534` | 6/10 |
| Value-disagreement RMSE | `-0.00541 +/- 0.00103` | 10/10 |
| Top-decile uncertainty recall | `+0.07013 +/- 0.03545` | 8/10 |

The rank result remains noisy and should not be presented as a definitive
ranking improvement. The RMSE and selective-recall effects are substantially
more stable. For the RMSE paired differences, a two-sided 95% Student-t
interval is approximately `[-0.00774, -0.00308]`, excluding zero. The analogous
rank and top-decile intervals include zero. The appropriate claim is therefore:

> On a controlled multimodal stress benchmark, a hybrid state-geometry plus
decision-value disagreement objective consistently improves the magnitude and
selective detection of distilled uncertainty relative to ordinary member-wise
matching, while ranking fidelity remains variable.

This does not prove literature novelty. It is empirical support for a narrower,
decision-relevant method claim. A final paper should add a real environment,
matched wall-clock/NFE measurements, confidence intervals over evaluation
latents, and an ablation of each hybrid term.

## Current thesis status

The Pendulum result weakens the original claim that state-space geometry alone
is generally necessary or beneficial. It does **not** refute the broader thesis:
ordinary distillation can preserve uncertainty on easy deterministic tasks, but
can lose decision-relevant uncertainty under multimodal, sparse, or nonlinear
consequences. The implemented hybrid objective is the current lead method.
