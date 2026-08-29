# Identifiability of the decision-distillation uncertainty target (2026-08-29)

**Theory:** `theory/distill_identifiability.py` (exact, verified).
**Experiment:** `runs/identified_stress_fair_20seed.json`, 20 paired seeds, exact
teacher checksum pairing (one checksum value per seed), same protocol as
[`RESULTS-STRESS-LARGE-2026-08-21.md`](RESULTS-STRESS-LARGE-2026-08-21.md).
**Bootstrap:** 20,000 percentile resamples over seeds
(`udwm/scripts/summarize_identified_stress.py`).

---

## 1. The mechanism

`decision_preserving_distill_loss` matches teacher and student cross-member value
variance at **one shared diffusion latent** per batch element (a single `t_idx` /
`noise` draw, reused for every member). Writing member values as
\(Y_i=\mu_i+\varepsilon_i\) with the \(\varepsilon_i\) correlated through the shared
latent, the repo's own coupling identity
(`MCUBELocalRewards.combine_coupled`) gives

\[
\mathbb E\!\left[\operatorname{Var}_i(Y_i)\right]
= w^\star + \big(g^\star-\bar\Sigma\big),
\qquad
g^\star-\bar\Sigma=\frac{1}{2N^2}\sum_{ij}\operatorname{Var}(Y_i-Y_j)\;\ge 0 .
\]

The matched statistic is a **sum** of an epistemic term and a latent-conditional
term: one scalar equation, two unknowns. The objective is unidentified, and the
student can satisfy it by trading one against the other at zero loss.

`theory/distill_identifiability.py` Part 1 constructs the zero-loss family
explicitly. Four students, all reproducing the teacher's single-latent statistic
to \(\sim\!10^{-6}\):

| student \(w^\star\) | student \(g^\star\) | single-latent statistic | loss (M=1) |
|---:|---:|---:|---:|
| 0.1480 (truth) | 0.3600 | 0.4370 | 2.9e-08 |
| 0.1480 | 0.2520 | 0.4365 | 2.6e-07 |
| 0.0725 | 0.3463 | 0.4358 | 1.4e-06 |
| 0.0237 | 0.4074 | 0.4364 | 3.6e-07 |
| **0.0000** | 0.4370 | 0.4384 | 1.8e-06 |

A student with **zero epistemic disagreement** is indistinguishable from a
faithful one. Drawing \(M\ge2\) latents and matching the debiased \(\hat w\) and the
aleatoric \(\hat g\) as two separate terms separates them: loss 0.0000 for the
truth vs 0.0279 for the collapse, at both \(M=2\) and \(M=8\).

**Nothing in that construction involves a critic.** The value map is a fixed
deterministic function throughout. So the degeneracy is not caused by
nonstationarity, and a lagged target critic addresses a different problem.

### Relation to prior art

This is the ensemble/epistemic analogue of the variance-shrinkage pathology
proved for value-aware model losses under sampled models by Voelcker et al.,
arXiv:2505.22772, Lemmata 1–3: an uncalibrated value-matching loss **prefers
low-variance models**, because a zero-variance \(q\) with
\((\mathbb E_q[f]-\mathbb E_p[f])^2<\tfrac1k\operatorname{Var}_{p^\star}[f]\) beats every
correct-in-expectation \(p^\star\). Their collapsing variance is one stochastic
model's aleatoric spread; ours is disagreement across ensemble members. Their fix
(subtract the estimated model variance, requires \(k\ge2\)) has the same shape as
the fix here.

That paper must be cited. It also means the framing "online critic is unsafe" is
weaker than the framing available: the real statement is a **calibration /
identifiability** statement, which transfers to any value map.

---

## 2. What the 20-seed fixed-map study says

Arms, all with exact teacher pairing, `decision_value` fixed throughout:

- `ordinary` — member-wise \(x_0\) matching only (baseline).
- `decision_hybrid` — the published arm: single-latent value geometry + value
  variance + state geometry + state pairwise.
- `decision_identified` — \(M{=}2\) latents, separate \((\hat w_{\mathrm{deb}},\hat g)\)
  terms, **same** state-geometry weights as `decision_hybrid`.
- `decision_identified_nostate` — same but with state-geometry weights zeroed.

### Reproduction check

`decision_hybrid` reproduces the published table exactly: `w_rank_corr`
+0.031668 (published +0.03167), `w_rmse` −0.003741 (published −0.003741),
`top_decile_recall` +0.017476 (published +0.01748). The added diagnostics do not
perturb the arm.

### Versus `ordinary`

| endpoint | `decision_hybrid` | `decision_identified` |
|---|---|---|
| `w_rank_corr` | +0.0317, 13/20, [+0.0015, +0.0633] | **+0.0344, 14/20, [+0.0019, +0.0671]** |
| `w_deb_rank_corr` | +0.0372, 15/20, [+0.0052, +0.0696] | **+0.0427, 16/20, [+0.0086, +0.0764]** |
| `top_decile_recall` | +0.0175, 9/20, [−0.0214, +0.0573] | +0.0282, 12/20, [−0.0102, +0.0660] |
| `w_rmse` | **−0.00374, 20/20** | −0.00337, 20/20 |
| `w_deb_rmse` | **−0.00262, 20/20** | −0.00237, 20/20 |
| `g_rmse` | +0.00559, 5/20, [+0.0032, +0.0081] | +0.00585, 4/20, [+0.0035, +0.0083] |
| `g_rank_corr` | −0.00319, 6/20, [−0.0052, −0.0014] | −0.00390, 4/20, [−0.0058, −0.0020] |
| `paired_state_mse` | −0.000975, 16/20 | −0.000572, 14/20 |

`decision_identified_nostate` is inconclusive on every ranking endpoint
(`w_deb_rank_corr` +0.0167, [−0.0246, +0.0517]), which confirms the
state-geometry terms carry a real share of the published arm's gain. An earlier
version of this comparison omitted them and lost for that reason; the table above
is the corrected, matched comparison.

### Head to head, `decision_identified` − `decision_hybrid`

| endpoint | delta | wins | 95% | verdict |
|---|---:|---:|---|---|
| `w_rank_corr` | +0.00269 | 11/20 | [−0.0200, +0.0213] | inconclusive |
| `w_deb_rank_corr` | +0.00547 | 13/20 | [−0.0153, +0.0234] | inconclusive |
| `top_decile_recall` | +0.01068 | 12/20 | [−0.0301, +0.0427] | inconclusive |
| `w_rmse` | +0.00037 | 3/20 | [+0.00012, +0.00065] | worse |
| `w_deb_rmse` | +0.00024 | 4/20 | [+0.00006, +0.00044] | worse |
| `paired_state_mse` | +0.00040 | 4/20 | [+0.00019, +0.00062] | worse |

**On this benchmark the identified objective is a wash on ranking and slightly
worse on magnitude, at 2× the sampling cost.** State that plainly. It is also
exactly what the theory predicts: Part 3 of `distill_identifiability.py` says the
degenerate direction exists under a fixed map but that a benign, stationary
`decision_value` gives gradient descent no reason to walk it. A fixed-map study
therefore **cannot discriminate** between the two objectives. The discriminating
experiment is the policy setting.

---

## 3. The new finding: the loss *reallocates* variance, and that unifies both results

The diagnostics added to the stress evaluator report the coupling-aware split
\((\hat w_{\mathrm{deb}},\hat g)\) rather than only \(\hat w\). Across all decision arms:

- epistemic fidelity **improves** (`w_deb_rank_corr` and `w_deb_rmse`, intervals
  excluding zero),
- aleatoric fidelity **degrades** (`g_rmse` worse 5/20 and 4/20, `g_rank_corr`
  worse 6/20 and 4/20, all intervals excluding zero),
- `split_distortion` — the ratio \((\hat g_S/\hat g_T)/(\hat w_S/\hat w_T)\), where 1 is a
  faithful split — moves from 0.187 (`ordinary`) to 0.487 (`hybrid`) and 0.382
  (`identified`), 20/20 seeds in both cases.

So the loss does not preserve the two components; it **moves variance between
them**. That is the degenerate direction, observed being walked on real models.

This is the useful reframing, because it explains **both** of the repo's two
headline results with one mechanism:

> Matching \(w+(g-\bar\Sigma)\) leaves the split free, so the optimiser reallocates
> between epistemic and aleatoric variance. Under a benign fixed value map the
> reallocation happens to move the epistemic component toward the teacher
> (`w_deb_rmse` better 20/20). Under an online critic it runs the other way and
> destroys the epistemic ranking (10/10 in
> [`RESULTS-POLICY-SCALE-2026-08-22.md`](RESULTS-POLICY-SCALE-2026-08-22.md)).
> The **sign** is a property of the value map; the **freedom** is a property of the loss.

The current two-result thesis presents these as a positive and a negative finding
held in tension. They are one finding measured twice.

---

## 4. Predictions, stated before the policy run

Registered here so the policy 2×2 (`runs/policy_identifiability_2x2.json`,
arms `ordinary` / `hybrid` / `lagged_hybrid` / `identified_hybrid` /
`lagged_identified`) is a test rather than a search.

1. **`lagged_hybrid` reduces but does not eliminate the uncertainty-rank
   collapse.** It fixes nonstationarity and leaves identifiability untouched. If
   rank correlation still loses on most seeds, that is this prediction confirming,
   not the correction failing.
2. **`identified_hybrid` — the identified target with the *live* critic, i.e.
   deliberately the falsified condition — recovers rank correlation.** If it does,
   the collapse was identifiability, not nonstationarity, and the thesis becomes a
   calibration result that transfers to any value map.
3. `lagged_identified` should be best or tied with `identified_hybrid`. If the
   lagged critic adds nothing once the target is identified, say so.

If prediction 2 fails while 1 holds, the honest conclusion is that
nonstationarity dominates on this benchmark and the identifiability result stands
as theory with no measured policy consequence. That is a publishable negative
about the mechanism's reach, and it should be reported as such.

---

## 5. Known weakness in the current implementation

`identified_decision_distill_loss` weights the epistemic and aleatoric terms
equally, but on this benchmark \(\hat g\approx 0.15\) and \(\hat w\approx 3\times10^{-4}\).
`F.mse_loss(s_g, t_g)` is therefore ~3 orders of magnitude larger than
`F.mse_loss(s_w, t_w)`, so the epistemic term is numerically negligible without
reweighting. That is the most likely reason the head-to-head in §2 is a wash
rather than a win, and it is fixable: standardise each term by the teacher's own
spread of that statistic across the minibatch before summing. Not done.
