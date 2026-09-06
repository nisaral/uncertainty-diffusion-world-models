# Result index

Read **identifiability** first. The two earlier tables are the same mechanism
under two value maps, not two competing theses.

| Study | What it shows | File |
|---|---|---|
| Identifiability of the distillation target | Single-latent hybrid matches \(w+(g-\bar\Sigma)\); zero-\(w^\star\) students get ~0 loss; \(M\ge2\) + split \((\hat w,\hat g)\) identifies. Fixed-map 20-seed (confirmed at 30): identified vs hybrid is a **wash**. Policy 2×2 adjudicated at N=30: lagging (not identifiability) is the measured fix -- [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md). | [RESULTS-IDENTIFIABILITY-2026-08-29.md](RESULTS-IDENTIFIABILITY-2026-08-29.md) |
| Large controlled stress (20 seeds, **fixed** map) | Hybrid RMSE 20/20; rank 13/20; recall inconclusive. `w_deb` up, `g` down: variance is reallocated. | [RESULTS-STRESS-LARGE-2026-08-21.md](RESULTS-STRESS-LARGE-2026-08-21.md) |
| Fixed-map stress, N=30 extension | 20-seed rows bit-identical to the 08-29 study (strict superset). Hybrid magnitude 29/30; w-rank gain now confirmed (22/30, CI excludes 0); `g` worse (reallocation); top-decile recall still inconclusive. Identified vs hybrid: ranking wash, slightly worse magnitude at 2x cost. Policy is the discriminating experiment (see N=30 policy row). | [RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md](RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md) |
| Fixed-map stress, N=50 precision extension | Registered before seeds 30..49 ran (seeds 0..29 byte-identical, strict superset). Magnitude 49/50; w-rank 38/50 (CI excludes 0); `g` reallocation stable. Top-decile recall: mean CI [+0.0012, +0.0456] just excludes 0 but only 27/50 seeds positive -- inconclusive by the pre-registered 70% seed bar. Identified-vs-hybrid wash unchanged at 50. | [RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md](RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md) |
| Policy-scale (10 seeds, **live** critic) | Hybrid MSE 10/10; uncertainty rank/RMSE **0/10**; return 3/10. | [RESULTS-POLICY-SCALE-2026-08-22.md](RESULTS-POLICY-SCALE-2026-08-22.md) |
| Earlier policy probe | Inconclusive / negative transfer | [RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md](RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md) |
| Earlier distillation table | Superseded by the 20-seed study | [RESULTS-DISTILLATION-2026-08-20.md](RESULTS-DISTILLATION-2026-08-20.md) |

| Ground-truth (w*, g*) recovery | First check of the identified loss against ANALYTIC (w*, g*): estimator unbiased; M=1 objective walks the degenerate direction from both inits; equal-weighting hole (w rank 0.40 when g* >> w*) and the partial reweighting fix quantified. | [RESULTS-GROUND-TRUTH-W-G-2026-09-01.md](RESULTS-GROUND-TRUTH-W-G-2026-09-01.md) |
| Policy 2×2, 5-seed preliminary | Hybrid u-rank collapse reproduces (-0.301, 0/5); lagged critic recovers u-rank at 5 seeds (leans nonstationarity-dominant); identified arm confounded by reweighting hole. Not a verdict. | [RESULTS-POLICY-2X2-2026-09-01.md](RESULTS-POLICY-2X2-2026-09-01.md) |
| Policy 2×2, N=30 adjudicated rerun | Pre-registered extension to 30 seeds (EMA-reweighted identified arms). **Row 1 of the decision tree**: lagging fixes the u-rank collapse (30/30, to ~ordinary); the identified loss does not (0/30, U collapses to ~0). Nonstationarity is the measured mechanism (2026-09-05 addendum: the identified rows were the EMA-reweighted variant - see the corrected-weight N-study row below; identifiability now has a measured policy benefit for the equal-weight arm). Return: identified arms best, but return does not adjudicate. | [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md) |
| Payoff test #4 (lagging alone, 3,600 steps), N=30 | Mechanism survival confirmed (lagged u-rank 0.988 ~ ordinary 0.986 at N=30; vs hybrid +0.273, 30/30). Return payoff still not demonstrated: pooled delta -44.6, 15/30, CI [-102.6, +10.2] (MBPO returns drift down past ~1,800 steps on this benchmark). | [RESULTS-PAYOFF-LAGGED-2026-09-03.md](RESULTS-PAYOFF-LAGGED-2026-09-03.md) |
| U-collapse mechanism (thread #1) | Why the identified arms zero the U object in policy (2026-09-05 addendum: the leverage/estimator-noise numbers describe the EMA-down-weighted channel; corrected attribution in the Corruption-probe row). The local object is aleatoric-dominated (teacher g/w ~ 1e4 under the SAC critic); identified arms match w but leave g at initialization (student latent-to-state spread ~0.006 vs teacher ~0.37). Corrects the "u_min=0 floor" claim in the N=30 policy doc. | [U-COLLAPSE-MECHANISM-2026-09-05.md](U-COLLAPSE-MECHANISM-2026-09-05.md) |
| Leverage-fix experiment | Registered test of whether the aleatoric starvation is a step-size artifact (2026-09-05 addendum: a config-override bug made every arm run EMA-on; the verdict and the "corruption is the remaining lever" claim are superseded by the Corruption-probe row). Loss-scale scaling is inert under Adam; 10x-100x larger student LR drives g to exactly 0 at the flat g=0 boundary. | [RESULTS-LEVERAGE-FIX-2026-09-05.md](RESULTS-LEVERAGE-FIX-2026-09-05.md) |
| Corruption probe + weighting correction (2026-09-05) | Registered corruption-distribution test with the probe's config-override bug fixed. Verdict: corruption is NOT the binding constraint - equal-weight identified recovers g on the live critic (u-rank 0.885/0.920, 2 seeds; N=10 replication in the doc) while EMA-both and w-only EMA annihilate g to ~0.01 (driver: the epistemic up-weight ~1e5-1e6, not the aleatoric down-weight). `pure` decision corruptions additionally degrade next-state MSE. Supersedes the leverage-fix/mechanism rows above. | [RESULTS-CORRUPTION-2026-09-05.md](RESULTS-CORRUPTION-2026-09-05.md) |
| Corrected-weight policy N-study (N=10, 2026-09-05) | Equal-weight identified (identified_eq) transfers to the live critic at policy scale: u-rank mean 0.835, 9/10 >= 0.70 (pre-registered bars A met); beats hybrid +0.214 (10/10) and the EMA arms +0.720 (10/10); below ordinary -0.104 (0/10) and lagged_hybrid -0.120 (0/10). The N=30 "identified does not matter in policy" verdict is attributed to the EMA variant, not the identified target; the w_rmse hole persists (0.995 vs ordinary 0.497). | [RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md](RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md) |
| Estimator-noise scaling + balance window | Analytic fit std(g_hat)/g* ~ sqrt(2/(N(M-1))) and std(w_hat)/w* ~ 1.35@M=2 -> 0.22@M=32 set the window where the split (w_hat, g_hat) is learnable. Toy re-adjudication: equal-weight identified recovers g (g_rank 0.81-0.97) but leaves w at the hole (w_rank 0.40) when g* >> w*; EMA-both fixes w (0.97-0.99) and annihilates g (g_hat ~ 0.03-0.4 vs g* 46-3966). No single scalar weight recovers both at g* >> w*. | [theory/identified_balance_window.py](../theory/identified_balance_window.py) |
| Paper narrative assembly | Single-arc ordering of the result docs (identifiability -> fixed map N=50 -> live-critic falsification -> policy 2x2 -> payoff -> mechanism/correction -> balance window -> limitations), plus the statistics statement (no joint multiple-comparisons correction; per-endpoint pre-registration as the mitigation) and the future-work position (long-horizon low-dimensional benchmark before pixels). | [PAPER-NARRATIVE.md](PAPER-NARRATIVE.md) |

Estimator notes (not policy tables):
[ESTIMATOR-BIAS-FINDING.md](ESTIMATOR-BIAS-FINDING.md) (first moment, distribution-free),
[ESTIMATOR-VARIANCE-FINDING.md](ESTIMATOR-VARIANCE-FINDING.md) (second moment, kurtosis; why the model class can matter).

**Payoff test #4 (2026-09-03):** adjudicated at N=30 (3,600 steps, lagging
alone): mechanism survives (lagged u-rank 0.988 ~ ordinary 0.986; +0.273 vs
hybrid, 30/30); return payoff not demonstrated (delta -44.6, 15/30, CI
[-102.6, +10.2]). The open question for a payoff claim is the
benchmark/protocol (the deferred modality choice), not more steps on
DelayedBimodal: MBPO returns drift down past ~1,800 env steps here.

---

## Addendum 2026-09-07: DMC 10-seed sanity + budget probe

DMC/hopper-hop 10-seed sanity (GPU rows, N=10, NOT adjudication): gate regime
aleatoric-dominated (median g*/w* 8,224). The EMA-collapse mechanism
replicates cross-environment (eq - identified_hybrid u_rank +0.419, 10/10);
the arm ordering replicates (lagged_hybrid 0.573 > ordinary 0.511 > hybrid
0.438 ~= eq 0.429 > EMA 0.01); absolute levels do not (ordinary 0.51 here vs
0.94 on DelayedBimodal). The DelayedBimodal headline (eq > hybrid, +0.214,
10/10) is a wash on DMC (-0.008, 5/10); eq - ordinary and hybrid - ordinary
are confirmed-below on u_rank; the eq w hole does not replicate (eq w_rmse
better than ordinary, 9/10). Because 3,600 steps = 3.6 1,000-step episodes
and returns are flat everywhere, the sanity cannot distinguish "identified_eq
does not transfer" from "the u_rank measurement is degenerate at this
budget"; the 30-seed adjudication is held pending the registered budget probe
(15k = 15 episodes, seeds 0-1).
[RESULTS-DMC-SANITY-2026-09-07.md](RESULTS-DMC-SANITY-2026-09-07.md)