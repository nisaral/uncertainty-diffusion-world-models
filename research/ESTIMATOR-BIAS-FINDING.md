# The naive MC-UBE local reward is biased at O(1/M) — derivation and correction

**Status:** Derived (§2, closed form); implemented (`udwm/uncertainty/mc_ube.py`).
**Numerical verification: RUN 2026-08-29.** `theory/estimator_bias.py` Part 1
reproduces the boxed constants to Monte Carlo precision for **both** the Gaussian
and the bimodal law (at \(M=8\): measured \(+0.0916\) / \(+0.0914\) against predicted
\(+0.0900\); the debiased column is \(\sim\!0\) at every \(M\ge2\)). The bias claim and
the distribution-free claim are confirmed; this document is citable.
**Three claims in it are NOT confirmed by that run** — see the box in §0 below.
**Closes:** gap #2 in `research/proofs/theorem-A1-mc-concentration-sketch.md`
("biased vs unbiased variance formula consistency between \(w^\star\) and \(\hat w\)")
**Why it matters:** this is not bookkeeping. At the repo's default budgets the bias
is the same order as the estimand.
**Companion:** [`ESTIMATOR-VARIANCE-FINDING.md`](ESTIMATOR-VARIANCE-FINDING.md)
covers the second moment, which is *not* distribution-free, and revises §7's
conclusion about what the model class buys.

---

## 0. Corrections after running the verification

The bias algebra survived. Three downstream claims did not.

| Claim in this document | What the run shows |
|---|---|
| §5: adaptive-\(M\) is distorted by a heterogeneous \(1/M\) bias, and "under the debiased estimator this particular differential vanishes", implying debiasing rescues novelty B | Debiasing is **not sufficient**. Adaptive-\(M\) still loses to uniform-\(M\) at matched cost: Spearman 0.821 vs 0.909, MAE 0.328 vs 0.221. The residual cause is the *variance* coefficient \(A\), not the bias — see `ESTIMATOR-VARIANCE-FINDING.md` §4. The load-bearing defect is the \(m_{\mathrm{probe}}=2\) floor. |
| §5: selecting on the noisy probe is a winner's curse that biases unrefined states **low**, so oracle selection should rank better | The measured gap has the **opposite sign**: probe-\(w\) selection 0.821, true-\(w\) selection 0.788, gap \(-0.033\). Probe selection accidentally refines high-\(g\) states (a noisy probe is likelier to come out high when \(g\) is large), which is closer to the Neyman-optimal rule than oracle-\(w\) selection is. The §5 narrative is inverted. |
| §5: the sample-splitting fix is "not yet implemented" | It **is** — `AdaptiveMCUBELocalRewards.sample_split=True` by default since 2026-08-19, and `ASSUMPTIONS-LOG.md` A8 records it. **Drift closed 2026-09-03:** `theory/estimator_bias.py` Part 3 now simulates BOTH paths with corrected cost accounting (samples actually drawn, including the selection probe that refined states discard): reuse-probe ≈ 8/state, sample-split ≈ 9/state at the stated settings. Measured (debiased, 2026-09-03): uniform M=9 Spearman 0.941 vs adaptive sample-split 0.826 vs reuse-probe 0.821. Splitting removes the winner’s-curse conditioning on unrefined scores, but the dominant residual versus uniform at matched cost is the A/M variance term, not the bias. |

None of these touch §1–§4 or §6.

---

## 1. The claim


The estimator in `MCUBELocalRewards.estimate` is, per state-action:

\[
\hat\mu_i=\tfrac1M\textstyle\sum_m y_{i,m},\quad
\hat v_i=\tfrac1M\textstyle\sum_m (y_{i,m}-\hat\mu_i)^2,\quad
\hat w=\tfrac1N\textstyle\sum_i(\hat\mu_i-\hat{\bar\mu})^2,\quad
\hat g=\tfrac1N\textstyle\sum_i\hat v_i,
\]

with \(y_{i,m}=\bar Q(s'_{i,m},a'_{i,m})\), \(s'_{i,m}\sim p_{\theta_i}(\cdot\mid s,a)\), and \(\hat u=\hat w-\hat g\).

Write \(\mu_i^\star=\mathbb E[y_{i,1}]\), \(\sigma_i^2=\mathrm{Var}[y_{i,1}]\), and the population targets
\(w^\star=\frac1N\sum_i(\mu_i^\star-\bar\mu^\star)^2\), \(g^\star=\frac1N\sum_i\sigma_i^2\), \(u^\star=w^\star-g^\star\).

**Result.**

\[
\boxed{\;
\mathbb E[\hat w]=w^\star+\frac{N-1}{N}\cdot\frac{g^\star}{M},
\qquad
\mathbb E[\hat g]=g^\star\cdot\frac{M-1}{M},
\qquad
\mathbb E[\hat u]=u^\star+g^\star\cdot\frac{2N-1}{NM}.
\;}
\]

The two errors **compound rather than cancel**: \(\hat w\) is inflated and \(\hat g\)
is deflated, and \(\hat u=\hat w-\hat g\) collects both.

---

## 2. Derivation

Write \(\hat\mu_i=\mu_i^\star+e_i\) with \(\mathbb E[e_i]=0\), \(\mathrm{Var}(e_i)=\tau_i^2=\sigma_i^2/M\),
and \(e_1,\dots,e_N\) independent (independent sampler restarts per member — true in
the code, since each member is sampled in its own call).

**(a) \(\hat w\).** With \(\bar e=\frac1N\sum_i e_i\),

\[
\mathbb E[\hat w]
=\tfrac1N\textstyle\sum_i \mathbb E\big[(\mu_i^\star-\bar\mu^\star+e_i-\bar e)^2\big]
=w^\star+\tfrac1N\textstyle\sum_i\mathbb E[(e_i-\bar e)^2],
\]

the cross terms vanishing because \(\mathbb E[e_i]=0\). Independence gives
\(\mathbb E[(e_i-\bar e)^2]=\tau_i^2-\frac{2}{N}\tau_i^2+\frac1{N^2}\sum_j\tau_j^2\), so

\[
\tfrac1N\textstyle\sum_i\mathbb E[(e_i-\bar e)^2]
=\Big(1-\tfrac1N\Big)\cdot\tfrac1N\textstyle\sum_i\tau_i^2
=\frac{N-1}{N}\cdot\frac{g^\star}{M}.
\]

**(b) \(\hat g\).** Standard: the \(\mathrm{ddof}=0\) sample variance satisfies
\(\mathbb E[\hat v_i]=\sigma_i^2(M-1)/M\), hence \(\mathbb E[\hat g]=g^\star(M-1)/M\).

**(c) \(\hat u\).** Subtract:

\[
\mathbb E[\hat u]=w^\star+\frac{N-1}{N}\frac{g^\star}{M}-g^\star+\frac{g^\star}{M}
=u^\star+\frac{g^\star}{M}\Big(\frac{N-1}{N}+1\Big)
=u^\star+g^\star\frac{2N-1}{NM}. \qquad\square
\]

**Distribution-free.** Only \(\sigma_i^2\) enters. The shape of the next-value law is
irrelevant, so multimodality — the reason to use diffusion at all — provides no
protection *against the bias*. `theory/estimator_bias.py` tests this by drawing a
Gaussian and a bimodal mixture with *matched* \(\mu_i^\star,\sigma_i^2\) (half the
variance from mode separation, half within-mode) and checking both reproduce the
same bias. **Run 2026-08-29: they do**, to Monte Carlo precision at every
\(M\in\{2,4,8,16,32,64\}\).

> **But this is a statement about the first moment only.** The *variance* of
> \(\hat u\) carries the fourth moment and is **not** distribution-free:
> \(\operatorname{Var}(\hat u_{\mathrm{deb}})=A/M\) with
> \(A\ni\frac{1}{N^2}\sum_i(\kappa_i-1)\sigma_i^4\). A rare mode of probability \(p\)
> has \(\kappa\approx1/p\), giving \(M=\Omega(1/(p\varepsilon^2))\) — measured 14.6×
> Gaussian at \(p=0.02\). The bimodal control used here is *symmetric*, hence
> \(\kappa=2.5<3\), i.e. **below** Gaussian: it is the benign multimodal case and
> structurally cannot detect the effect. See
> [`ESTIMATOR-VARIANCE-FINDING.md`](ESTIMATOR-VARIANCE-FINDING.md).


---

## 3. Why this is a first-order problem, not a rounding error

\(u^\star=w^\star-g^\star\) is a **difference of two comparable quantities**. The bias is
a fraction of \(g^\star\), i.e. of one of the two terms being differenced — so it is
naturally the same size as the result.

| \(M\) | bias\(/g^\star\) at \(N=5\) | where this \(M\) is used |
|---:|---:|---|
| 1 | 1.80 | `adaptive_mc` floor — \(g\) unidentifiable, no correction possible |
| 2 | 0.90 | `m_probe` default → **most of the batch** |
| 8 | 0.225 | `ube.m_samples` default |
| 12 | 0.15 | `m_max` → refined states only |

At \(M=8\) the estimator adds 22.5% of the aleatoric term to the epistemic one. On the
`theory/toy_mc_ube_estimator.py` ensemble (\(w^\star=0.290\), \(g^\star=0.400\), \(u^\star=-0.110\))
the bias at \(M=8\) is \(+0.090\) — **82% of \(|u^\star|\)**, and enough to flip the sign of a
quantity that is genuinely negative. The `u_min` clamp then hides the failure:
a negative \(u^\star\) that should clamp to 0 instead reports a confident positive value.

This is the same failure mode Luis et al.'s \(u=w-g\) correction exists to fix.
Naive MC silently reintroduces a Zhou-style overestimate at scale \(g^\star/M\)
instead of \(g^\star\) — smaller, but the same sign and the same character.

### Relation to Theorem A1

Theorem A1's \(O(1/\sqrt M)\) Hoeffding bound is *not wrong* — an \(O(1/M)\) bias is
dominated by an \(O(1/\sqrt M)\) envelope, so the bound holds. But it is **too loose to
see the problem**: it bounds \(|\hat w-w^\star|\) without distinguishing bias from
variance. Variance shrinks under averaging across training batches; bias does not.
Since the U-network is fit by regression on many \(\hat u\) targets, the variance
largely averages out and the bias is precisely what survives into \(U_\varphi\).
The concentration statement should therefore be split into a bias term and a
variance term rather than bounding the total.

---

## 4. The correction

Bessel-correct the inner variance, then subtract the MC noise of the means:

\[
\hat g_{\mathrm{ub}}=\frac1N\sum_i \hat v_i\cdot\frac{M}{M-1},
\qquad
\hat w_{\mathrm{deb}}=\hat w-\frac{N-1}{N}\cdot\frac{\hat g_{\mathrm{ub}}}{M},
\qquad
\hat u_{\mathrm{deb}}=\hat w_{\mathrm{deb}}-\hat g_{\mathrm{ub}}.
\]

By linearity of expectation this is **exactly unbiased** for \(u^\star\) at every \(M\ge2\),
with no asymptotic argument and no extra samples — the correction is free, computed
from statistics already needed for \(\hat g\). Cost: a modest variance increase, since
\(\hat g_{\mathrm{ub}}\) now enters \(\hat w_{\mathrm{deb}}\) as well.

At \(M=1\) the within-member variance is unidentifiable, so neither \(g\) nor the
correction exists. `adaptive_mc`'s `m_min=1` floor should be raised to 2.

Implemented as `MCUBELocalRewards.combine(..., debias=True)` (default on).
The Gaussian branch passes `means_are_mc=False`: its member means are exact
closed-form plug-ins carrying no MC noise, so only the Bessel correction applies.

---

## 5. Consequence for adaptive MC budgets (novelty B)

`AdaptiveMCUBELocalRewards` spends \(M=\) `m_probe` on most states and `m_max` on the
top-\(w\) fraction. Because the bias scales as \(1/M\), **refined states receive a
smaller inflation than the states that were skipped**:

\[
\underbrace{g^\star\frac{2N-1}{N\,m_{\mathrm{probe}}}}_{\text{unrefined, large}}
\;>\;
\underbrace{g^\star\frac{2N-1}{N\,m_{\max}}}_{\text{refined, small}}
\]

With `m_probe=2`, `m_max=12`, \(N=5\): the differential is \(0.75\,g^\star\), applied in
exactly the direction that pushes high-uncertainty states *down* relative to
low-uncertainty ones. The scheme compresses the ranking it was built to sharpen,
and the distortion is strongest where it spends the most compute.

Two consequences worth stating plainly in the paper:

1. Any measured benefit of adaptive-\(M\) under the naive estimator is confounded by
   this artifact — a spurious state-dependent offset, not sharper uncertainty.
2. Under the debiased estimator this particular differential vanishes.
   **Debiasing is a precondition for novelty B to mean anything**, which makes the
   correction load-bearing rather than cosmetic.

### A second, independent defect: selection on a noisy probe

Debiasing is necessary but **not sufficient**. `AdaptiveMCUBELocalRewards.estimate`
ranks states by the *probe* \(\hat w\) at \(M=\)`m_probe`, then keeps that same probe
estimate as the reported value for every state it does not refine. Selecting on a
statistic and then reporting it conditions on the draw: unrefined states are
precisely those whose \(\hat w\) came out low, so their reported \(\hat u\) is biased
low *beyond* the mean-field \(1/M\) term. Refined states get a fresh independent
draw, so they escape it.

This is a winner's curse, not a moment bias, and the closed-form correction cannot
touch it — that correction fixes \(\mathbb E[\hat u]\) at a *given* \(M\), whereas here the
conditioning is on the realized sample. The two effects also act in opposite
directions on the ranking (the \(1/M\) differential compresses it, selection noise
stretches it), so they do not simply add and the net sign is an empirical question.

The clean fix is sample splitting: probe with \(2m\) samples, select on the first
half, report from the second. Cost is one extra probe pass on the cheap budget.
Not yet implemented — recorded here so the ablation is not read as validating a
scheme that still has a known confound.

`theory/estimator_bias.py` Part 3 measures rank correlation \(\rho(\hat u,u^\star)\) across
a bank of synthetic states for uniform vs adaptive \(M\), naive vs debiased, and
isolates the selection effect by comparing probe-\(w\) against oracle true-\(w\)
selection under the debiased estimator. It simulates the reuse-probe path the
implementation actually takes.

---

## 6. Downstream: propagation into \(U\)

The bias is same-signed at every state, so it does not cancel in the Bellman
recursion. Corollary A2 with a uniform local error \(\varepsilon\) gives
\(\lVert\tilde U-U^\star\rVert_\infty\le\frac{\gamma^2}{1-\gamma^2}\varepsilon\); at \(\gamma=0.99\) the
amplification is \(\approx49\times\). For a constant offset this mostly rescales \(U\),
which is harmless for ranking but not for the \(\bar Q\pm\lambda\sqrt U\) objective:
the effective exploration/pessimism strength becomes a function of \(M\), so
**re-tuning \(\lambda\) after changing the sample budget is partly just chasing this bias.**
That is a testable prediction — with the naive estimator the best \(\lambda\) should shift
with \(M\); with the debiased one it should not.

---

## 7. Effect on the paper's positioning

This strengthens the contribution rather than weakening it. The original claim (A)
was "write down the obvious MC estimator and prove it concentrates" — true but thin,
and a reviewer can reasonably call the estimator immediate. The sharper claim:

> The obvious sample-based estimator of the UBE local reward is **biased at
> \(O(1/M)\) by a multiple of the aleatoric term it is supposed to subtract**, with an
> exact, distribution-free constant; the bias is removable in closed form at no
> sampling cost; and it interacts adversarially with per-state adaptive budgets.

That is a genuine estimator-theory result specific to implicit generative dynamics:
it does not arise for Gaussian ensembles, because there the member means are exact
and \(\tau_i^2=0\). **The bias exists precisely because diffusion forces you to
estimate the means by sampling.** The gap is created by the model class, which is
exactly the story the paper wants.

---

## 8. Open items

0. ~~**Run the verification.**~~ **Done 2026-08-29.** Part 1 reproduces the boxed
   constants for both laws. See the correction box in §0 for the three §5 claims
   the same run contradicts.
1. **Selection bias in adaptive \(M\)** (§5). The sample-splitting probe is
   implemented (`sample_split=True`); this document and
   `theory/estimator_bias.py` Part 3 both still describe the pre-2026-08-19 reuse
   path and need rewriting. The measured selection gap also has the opposite sign
   to the one predicted here.
2. **Dependent samples.** Derived and implemented — `combine_coupled`, with the
   bias equal to the mean pairwise disagreement variance over \(M\). Part 5
   confirms numerically that the *independent* correction over-subtracts under a
   coupling (bias \(-0.0357\) at \(\rho=0.99\)). **`udwm/eval/metrics.py` was applying
   exactly that wrong correction to paired latents** and has been fixed to use
   `combine_coupled`; the policy-scale numbers computed with the old path need
   re-running before they are cited. Note also that FLARE (arXiv:2602.09170)
   conditions on a shared reverse path for the same reason, so the coupling
   *device* is prior art — claim the identity, not the trick.
3. **Distillation interaction (claim B).** Now sharper than "report them
   separately": the distillation loss's single-latent variance target is
   *unidentified*, matching \(w^\star+(g^\star-\bar\Sigma)\) rather than \(w^\star\). See
   `theory/distill_identifiability.py` and
   [`RESULTS-IDENTIFIABILITY-2026-08-29.md`](RESULTS-IDENTIFIABILITY-2026-08-29.md).
4. **Variance of the debiased estimator.** Done: \(A/M\) in closed form, verified.
   The shrinkage-estimator question (better MSE than either extreme) is still open.
5. **Empirical.** The `debias` on/off ablation and the \(\lambda\)-vs-\(M\) prediction
   in §6 remain unrun.
6. **Convention mismatch.** `combine` uses `unbiased=False` (ddof=0) for \(\hat w\),
   matching Luis's finite-population \(\frac1N\sum_i(\mu_i-\bar\mu)^2\); the
   distillation loss and every metric use `unbiased=True` (ddof=1). At \(N=5\) that
   is a fixed \(1.25\times\) factor between the quantity the U-network regresses on
   and the quantity the student is trained and scored against. Constant, so it
   cannot flip a rank correlation, but it inflates every reported `w_rmse` by 25%.
7. **The Gaussian branch is not bias-free.** §7 claims the bias "does not arise
   for Gaussian ensembles". Its member mean is an exact plug-in, but
   `mc_ube.estimate` computes \(Q(\mathbb E[s'])\), not \(\mathbb E[Q(s')]\) — a
   deterministic Jensen/curvature gap, not removable by sampling. The honest
   asymmetry is *stochastic and fixable* versus *deterministic and not*, not
   *present* versus *absent*.

