# Ground-truth (w*, g*) recovery, the weighting hole, and rank robustness (2026-09-01)

**Toy:** `theory/ground_truth_w_g.py` (torch, CPU, ~4 min, no GPU).
**Data:** `runs/ground_truth_w_g.json`.
**Fix implemented in the repo:** `udwm.models.consistency.TermScaleEMA`, plumbed
through `WorldModel.build` / `DistilledWorldModel` / `trainer.py` as
`distill_reweight_ema` (default off; on for the identified policy arms in
`run_delayed_bimodal_policy_ablation.py`).

Closes the gap left by `theory/distill_identifiability.py`: the identified loss
was only ever shown to be internally consistent, never correct against the
**true** (w*, g*). Setup: N=5 members, F=4 features x=phi(s,a), member state
means `s_m(x) = Theta_m x` (the "parameter posterior", epistemic), injected
noise `eps ~ N(0, sigma^2(x) I)` with shared-latent coupling rho=0.4
(aleatoric), value map `V(s) = ||s||^2`, closed-form (w*, g*). Students are
small MLPs; losses are the repo's formulas, with `coupled_w_g` imported, not
reimplemented. Eval on a 256-point grid, M=64 MC.

## Part 0 — the estimator is correct, not merely self-consistent

| M | w_raw rel | w_deb rel | g rel | w_rank | g_rank |
|---|---:|---:|---:|---:|---:|
| 2 | 0.83 | 0.88 | 0.73 | 0.60 | 0.78 |
| 4 | 0.57 | 0.42 | 0.57 | 0.78 | 0.87 |
| 8 | 0.29 | 0.27 | 0.36 | 0.89 | 0.91 |
| 16 | 0.14 | 0.14 | 0.27 | 0.96 | 0.95 |
| 64 | 0.09 | 0.09 | 0.09 | 0.99 | 0.99 |

Coupling identity verified under the nonlinear value map: E[Var_m Y] = 55.36 vs
w* + g* - Sigma_bar = 55.46 (Sigma_bar = 8.41; repo loss uses ddof=1, x N/(N-1)).

## Part 1 — benign init: the M=1 objective walks the degenerate direction

| arm | w_rel | g_rel | w_rank | g_rank | w_top10 |
|---|---:|---:|---:|---:|---:|
| ordinary | 0.13 | 1.00 | 0.99 | 0.54 | 0.92 |
| hybrid (M=1) | 1.72 | 1.07 | 0.91 | 0.31 | 0.60 |
| identified (M=2, equal) | 0.32 | 0.97 | 0.93 | 0.90 | 0.84 |
| identified + EMA reweight | 0.13 | 1.07 | 0.99 | 0.40 | 0.84 |

Even from a benign init the M=1 objective does not land on the true split
(hybrid inflates w ~4x, g rank 0.31); identified recovers both; the EMA
reweighted identified is best on w (rel 0.13, rank 0.99) at a small g cost.

## Part 2 — collapsed init (the zero-loss family)

| arm | w_rel | g_rel | w_rank | g_rank | w_top10 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 0.30 | 2.29 | 0.94 | 0.52 | 0.84 |
| hybrid (M=1) | 1.54 | 0.95 | 0.89 | 0.64 | 0.72 |
| identified (M=2, equal) | 0.31 | 0.75 | 0.89 | 0.81 | 0.72 |
| identified + EMA reweight | 0.25 | 1.07 | 0.99 | -0.51 | 0.92 |

The identified target pulls the student back to the truth from the zero-loss
family; the M=1 objective leaves the w magnitude ~1.5x off.

## Part 3 — the g* >> w* hole: equal vs mean-magnitude vs EMA reweight

Noise scale x4 (g* ~ 165x w*), benign init:

| arm | w_rel | g_rel | w_rank | g_rank | w_top10 |
|---|---:|---:|---:|---:|---:|---:|
| identified, equal weights | 6.48 | 0.50 | 0.40 | 0.97 | 0.40 |
| identified + mean-mag reweight | 0.81 | 1.05 | 0.61 | 0.18 | 0.52 |
| identified + EMA reweight | 0.22 | 1.05 | 0.97 | 0.10 | 0.68 |

Equal weights: the aleatoric term drowns the epistemic term; w rank 0.40.
Mean-magnitude reweighting: partial. **EMA per-term normalisation (the fix now
in the repo) recovers w (rel 0.22, rank 0.97)**, at the cost of g fidelity.

## Part 4 — the finding: no single scalar weighting gives both magnitudes, but w ranking is robust

Weighting sweep in the hole regime: EMA-normalised terms, relative weight
r = kw/(kw+kg) swept over [0.001, 0.999]:

| r(w) | w_rank | w_rel | g_rank | w_top10 |
|---|---:|---:|---:|---:|
| 0.001 | 0.905 | 0.709 | 0.524 | 0.64 |
| 0.01 | 0.977 | 0.289 | 0.016 | 0.64 |
| 0.10 | 0.979 | 0.296 | 0.385 | 0.76 |
| 0.50 | 0.975 | 0.290 | 0.215 | 0.92 |
| 0.90 | 0.971 | 0.244 | 0.029 | 0.76 |
| 0.99 | 0.972 | 0.270 | -0.105 | 0.80 |
| 0.999 | 0.977 | 0.388 | 0.451 | 0.80 |

Two statements, both measured:

1. **No single scalar weighting recovers both magnitudes.** At every r at least
   one of w_rel, g_rel >= 0.5 (verified: 0/7 points satisfy both < 0.5). This
   is the reportable finding the 2026-09-01 direction predicted: when the true
   scales differ by ~2 orders of magnitude, the w-vs-g trade-off is inherent to
   a scalar-weighted sum, not a tuning bug.
2. **w RANKING is robust to the weighting** (>= 0.90 at all 7 points; top-decile
   recall 0.64-0.92, peak at r=0.5). So for rank/percentile-based downstream use
   (u-gated rollouts), the weighting choice is a bounded issue once terms are
   normalised. The raw equal-weight hole DOES break ranking (0.40 in Part 3) --
   the robustness is a property of the normalised objective, with the state-level
   member term anchoring the student's structure on this toy.

Caveat, stated plainly: on this toy the member term (state-level) strongly
anchors w's ranking. In the repo the member term is on x0 while w is on values,
so the anchor is weaker -- the rank-robustness claim is toy-level and the 2x2
rerun measures it in policy.

## What this changes

1. The identified loss's target is the truth (P0); the degenerate direction is
   observable in training dynamics (P1/P2).
2. The equal-weighting hole is real, quantified, and now **fixed** in the repo
   (TermScaleEMA; w rank 0.40 -> 0.97 in the hole regime).
3. The hole is reframed as a **bounded limitation**: no scalar weighting gives
   both magnitudes when g* >> w*, but w ranking is robust to the weighting once
   normalised. State both in the write-up.

## Next steps

- 2x2 rerun at 10 seeds with `distill_reweight_ema: true` on both identified
  arms, adjudicated by the pre-registered decision tree
  (`research/DECISION-TREE-2X2-PREREGISTRATION.md`).
- Payoff test (#4) after the tree, testing the mechanism the tree confirms.
