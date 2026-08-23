# Large Controlled Stress Study (2026-08-21)

## Protocol

Twenty paired seeds (0-19), four matched arms, 3,000 training transitions,
160 teacher updates, 240 student updates, five teacher members, eight paired
latent samples, and a 1,024-state held-out evaluation grid.

The teacher state was restored exactly before each arm. The maximum teacher
checksum gap was 0.0 for every seed.

## Paired results versus ordinary distillation

| Arm | Uncertainty RMSE delta | Bootstrap 95% interval | Rank-correlation delta | Bootstrap 95% interval | Top-decile recall delta |
|---|---:|---:|---:|---:|---:|
| State geometry | +0.000163 | [-0.000028, 0.000405] | +0.00633 | [-0.00163, 0.01513] | +0.00146 |
| Decision geometry | -0.003133 | [-0.004148, -0.002255] | +0.02985 | [0.00228, 0.05583] | +0.02524 |
| Decision hybrid | -0.003741 | [-0.004743, -0.002889] | +0.03167 | [0.00117, 0.06321] | +0.01748 |

For the hybrid arm, RMSE improved on 20/20 seeds and rank correlation improved
on 13/20 seeds. Top-decile recall improved on 9/20 seeds; its interval included
zero, so this endpoint is inconclusive.

The hybrid also reduced member-value RMSE (mean delta -0.00662, bootstrap
interval [-0.00995, -0.00356]) and paired state MSE (mean delta -0.000975,
interval [-0.00141, -0.00055]).

## Decision

This strongly supports the narrow empirical claim that decision-aware
distillation improves downstream uncertainty magnitude fidelity under this
controlled multimodal benchmark. It does **not** prove universal preservation,
and it does not establish reliable high-uncertainty retrieval or policy-return
improvements.

The state-only geometry arm was effectively null: both its RMSE and rank gains
had intervals crossing zero. This is evidence that the decision-value terms,
not generic state geometry, carry the measured benefit in this benchmark.

Raw data: `runs/decision_distillation_stress_large_20seed.json`.
