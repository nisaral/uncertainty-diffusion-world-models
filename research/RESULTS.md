# Result index

Read **identifiability** first. The two earlier tables are the same mechanism
under two value maps, not two competing theses.

| Study | What it shows | File |
|---|---|---|
| Identifiability of the distillation target | Single-latent hybrid matches \(w+(g-\bar\Sigma)\); zero-\(w^\star\) students get ~0 loss; \(M\ge2\) + split \((\hat w,\hat g)\) identifies. Fixed-map 20-seed (confirmed at 30): identified vs hybrid is a **wash**. Policy 2×2 adjudicated at N=30: lagging (not identifiability) is the measured fix -- [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md). | [RESULTS-IDENTIFIABILITY-2026-08-29.md](RESULTS-IDENTIFIABILITY-2026-08-29.md) |
| Large controlled stress (20 seeds, **fixed** map) | Hybrid RMSE 20/20; rank 13/20; recall inconclusive. `w_deb` up, `g` down: variance is reallocated. | [RESULTS-STRESS-LARGE-2026-08-21.md](RESULTS-STRESS-LARGE-2026-08-21.md) |
| Fixed-map stress, N=30 extension | 20-seed rows bit-identical to the 08-29 study (strict superset). Hybrid magnitude 29/30; w-rank gain now confirmed (22/30, CI excludes 0); `g` worse (reallocation); top-decile recall still inconclusive. Identified vs hybrid: ranking wash, slightly worse magnitude at 2x cost (unchanged at 30 seeds). Policy is the discriminating experiment (see N=30 policy row). | [RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md](RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md) |
| Policy-scale (10 seeds, **live** critic) | Hybrid MSE 10/10; uncertainty rank/RMSE **0/10**; return 3/10. | [RESULTS-POLICY-SCALE-2026-08-22.md](RESULTS-POLICY-SCALE-2026-08-22.md) |
| Earlier policy probe | Inconclusive / negative transfer | [RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md](RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md) |
| Earlier distillation table | Superseded by the 20-seed study | [RESULTS-DISTILLATION-2026-08-20.md](RESULTS-DISTILLATION-2026-08-20.md) |

| Ground-truth (w*, g*) recovery | First check of the identified loss against ANALYTIC (w*, g*): estimator unbiased; M=1 objective walks the degenerate direction from both inits; equal-weighting hole (w rank 0.40 when g* >> w*) and the partial reweighting fix quantified. | [RESULTS-GROUND-TRUTH-W-G-2026-09-01.md](RESULTS-GROUND-TRUTH-W-G-2026-09-01.md) |
| Policy 2×2, 5-seed preliminary | Hybrid u-rank collapse reproduces (-0.301, 0/5); lagged critic recovers u-rank at 5 seeds (leans nonstationarity-dominant); identified arm confounded by reweighting hole. Not a verdict. | [RESULTS-POLICY-2X2-2026-09-01.md](RESULTS-POLICY-2X2-2026-09-01.md) |
| Policy 2×2, N=30 adjudicated rerun | Pre-registered extension to 30 seeds (EMA-reweighted identified arms). **Row 1 of the decision tree**: lagging fixes the u-rank collapse (30/30, to ~ordinary); the identified loss does not (0/30, U collapses to ~0). Nonstationarity is the measured mechanism; identifiability stands as theory. Return: identified arms best, but return does not adjudicate. | [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md) |
| Payoff test #4 (lagging alone, 3,600 steps) | Mechanism survival confirmed at 2x horizon (lagged u-rank 0.989 ~ ordinary 0.987; hybrid still collapses 0.704). No return payoff demonstrated (return deltas null-to-negative, wide CIs; MBPO returns drift down past ~1,800 steps on this benchmark). | [RESULTS-PAYOFF-LAGGED-2026-09-03.md](RESULTS-PAYOFF-LAGGED-2026-09-03.md) |

Estimator notes (not policy tables):
[ESTIMATOR-BIAS-FINDING.md](ESTIMATOR-BIAS-FINDING.md) (first moment, distribution-free),
[ESTIMATOR-VARIANCE-FINDING.md](ESTIMATOR-VARIANCE-FINDING.md) (second moment, kurtosis; why the model class can matter).

**Payoff test #4 (2026-09-03):** run at N=10 (3,600 steps, lagging alone):
mechanism survives, no return payoff demonstrated at this scale
([RESULTS-PAYOFF-LAGGED-2026-09-03.md](RESULTS-PAYOFF-LAGGED-2026-09-03.md)).
N=30 precision extension registered 2026-09-03 before running seeds 10..29
(pooled adjudication on `runs/payoff_lagged_30seed_3600.json`). The open
question for a payoff claim is the benchmark/protocol (the deferred modality
choice), not more steps on DelayedBimodal: MBPO returns drift down past ~1,800
env steps here.
