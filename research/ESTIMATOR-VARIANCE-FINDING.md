# The MC-UBE variance is not distribution-free — and it is what argues for diffusion

**Status:** Derived and **numerically verified** (`theory/estimator_variance.py`, run 2026-08-29).
**Companion to:** [`ESTIMATOR-BIAS-FINDING.md`](ESTIMATOR-BIAS-FINDING.md), which covers the
first moment. This file covers the second, and reverses one of its conclusions.
**Relation to prior art:** the correction *shape* is anticipated by
Voelcker et al., *Calibrated Value-Aware Model Learning with Stochastic
Environment Models* (arXiv:2505.22772). The kurtosis result below is not.

---

## 1. Why this document exists

`ESTIMATOR-BIAS-FINDING.md` §2 establishes that the finite-\(M\) bias of
\(\hat u=\hat w-\hat g\) is \(g^\star(2N-1)/(NM)\) and that it is **distribution-free**
— it touches the next-value law only through \(\sigma_i^2\). It then concludes:

> multimodality — the reason to use diffusion at all — provides no protection

That is correct, and it is also a problem for the paper, because it removes the
only argument that the *model class* matters. §7 tries to recover one ("the bias
exists precisely because diffusion forces you to estimate the means by
sampling"), but that is an argument about *sampling*, not about *multimodality*,
and it is weakened by the fact that the Gaussian branch has its own
\(Q(\mathbb E[s'])\neq\mathbb E[Q(s')]\) plug-in bias (see §6 below).

The bias is also removable in closed form at zero sampling cost. So after the
correction, the only thing left is **variance** — and the variance is a
fourth-moment object.

---

## 2. The result

Write \(d_i=\mu_i^\star-\bar\mu^\star\) and let \(\sigma_i^2,\mu_{3i},\mu_{4i}\) be the
central moments of \(Y_i=\bar Q(s',a')\) under member \(i\). For the debiased
estimator \(\hat u_{\mathrm{deb}}\),

\[
\boxed{\;
\operatorname{Var}(\hat u_{\mathrm{deb}})=\frac{A}{M}+O(M^{-2}),\qquad
A=\underbrace{\frac{4}{N^2}\sum_i d_i^2\sigma_i^2}_{\text{epistemic}\times\text{aleatoric}}
+\underbrace{\frac{1}{N^2}\sum_i(\mu_{4i}-\sigma_i^4)}_{\text{fourth moment}}
-\underbrace{\frac{4}{N^2}\sum_i d_i\mu_{3i}}_{\text{skew cross term}}
\;}
\]

Substituting \(\mu_{4i}=\kappa_i\sigma_i^4\), the middle term is
\(\frac{1}{N^2}\sum_i(\kappa_i-1)\sigma_i^4\).

**So the bias depends on the law only through \(\sigma^2\); the variance depends on
it through \(\kappa\) and \(\mu_3\).** These are different moments, and the
"distribution-free" conclusion applies to exactly one of them.

### Verification

Four laws with **identical** \(\mu_i\) and \(\sigma_i^2\) — so identical \(w^\star,g^\star,u^\star\)
and identical bias — at \(N=5\), 60,000 trials:

| law | \(\kappa\) | \(A\) predicted | \(A\) measured (\(M{=}16\)) | \(A\) measured (\(M{=}64\)) | vs Gaussian |
|---|---:|---:|---:|---:|---:|
| gaussian | 3.00 | 0.1814 | 0.2006 | 0.1871 | 1.00× |
| bimodal, symmetric | 2.50 | 0.1597 | 0.1800 | 0.1652 | 0.88× |
| rare mode, \(p=0.10\) | 8.11 | 0.6015 | 0.6194 | 0.6041 | 3.32× |
| rare mode, \(p=0.02\) | 48.02 | 2.6452 | 2.6539 | 2.6574 | **14.58×** |

Predicted and measured agree to ~3% at \(M=64\) (the residual gap at \(M=16\) is the
\(O(M^{-2})\) term, and it shrinks in the right direction).

---

## 3. This is the "why the model class matters" bridge

A next-value law with one mode of probability \(p\) far from the bulk — the
archetypal thing a Gaussian ensemble **cannot represent** and a diffusion model
can — has \(\kappa\approx 1/p\). Hence

\[
M=\Omega\!\left(\frac{\kappa}{\varepsilon^2}\right)=\Omega\!\left(\frac{1}{p\,\varepsilon^2}\right).
\]

For \(\mathrm{sd}(\hat u)\le 0.10\,w^\star\) on the reference ensemble:

| law | \(\kappa\) | \(M\) required |
|---|---:|---:|
| gaussian | 3.00 | 215 |
| rare mode \(p=0.10\) | 8.11 | 713 |
| rare mode \(p=0.05\) | 18.05 | 1,354 |
| rare mode \(p=0.02\) | 48.02 | 3,137 |
| rare mode \(p=0.01\) | 98.01 | 5,973 |

Against a repo default of `ube.m_samples: 8`.

The statement to make in the paper is therefore **not** "diffusion is more
expensive to score". It is:

> The laws that require an implicit multimodal model class are exactly the laws
> whose epistemic score is most expensive to estimate, and the cost is the
> inverse rare-mode probability. The gap is created by the estimand, not by the
> sampler.

That is a claim about the *interaction* of model class and estimator, which is
the story `DETAILED-NOTES-AND-GAP-ANALYSIS.md` Direction A wants and the bias
result cannot supply.

### Correction to `ASSUMPTIONS-LOG.md`, edge case 5

That entry originally read "highly multi-modal \(p(s'\mid s,a)\) with small \(M\) →
high variance of \(\hat u\); Bernstein may beat Hoeffding", and was later annotated
with a **Correction** stating the dominant error is a distribution-free bias, not
variance. The annotation over-corrected. Both are true and they are different
moments: the bias is distribution-free, the variance is not. The original
instinct about multimodality was right, and Bernstein-type bounds — which carry
the variance explicitly — are the right tool precisely because of \(A\).

### The repo's own bimodal control cannot see this

`theory/estimator_bias.py::draw_bimodal` splits variance evenly between mode
separation and within-mode spread, which gives \(\kappa=2.5\) — **below** Gaussian.
It is the *benign* multimodal case. That is why the bias check passes cleanly and
why this effect never surfaced. Any future distributional control should include
an **asymmetric, low-probability** mode, not a symmetric one.

---

## 4. Consequence for adaptive budgets (novelty B)

`ESTIMATOR-BIAS-FINDING.md` §5 argues the adaptive scheme is distorted by a
heterogeneous \(1/M\) bias, and that debiasing removes it. Running Part 3 shows
debiasing is **not sufficient**: adaptive-\(M\) still loses to uniform-\(M\) at
matched cost (Spearman 0.821 vs 0.909; MAE 0.328 vs 0.221). The reason is in \(A\).

Minimising \(\sum_s A_s/M_s\) subject to \(\sum_s M_s=B\) is a Lagrange problem whose
solution is the **Neyman allocation**

\[
M_s\propto\sqrt{A_s}.
\]

\(A_s\) is driven by \(\sigma^4\) and kurtosis, i.e. by the **aleatoric** level.
Measured rank correlations over a 600-state bank: \(\rho(A,g)=+0.78\),
\(\rho(A,w)=+0.61\), \(\rho(w,g)=+0.05\). At a matched budget of 4,200 samples/member
(mean \(M=7\)), all rows debiased:

| scheme | mean \(M\) | Spearman\((\hat u,u^\star)\) | MAE | RMSE |
|---|---:|---:|---:|---:|
| uniform \(M=7\) | 7.0 | 0.8989 | 0.2206 | 0.3080 |
| **top-\(w\) refine (2 / 12) — the repo rule** | 7.0 | **0.7885** | **0.3357** | **0.5196** |
| Neyman \(M_s\propto\sqrt{A_s}\) (oracle) | 7.0 | 0.8959 | 0.2272 | 0.2978 |
| \(M_s\propto\sqrt{g_s}\) (implementable proxy) | 7.0 | 0.9089 | 0.2175 | 0.3001 |
| \(M_s\propto\sqrt{w_s}\) (wrong statistic) | 7.0 | 0.8574 | 0.2639 | 0.3780 |

Two things follow, and the second is the useful one:

1. Refining the top-\(w\) states is refining the wrong states. \(\rho(A,w)\) is not
   zero, so it is not *maximally* wrong, but \(\sqrt{g}\) beats it on every column.
2. **The damage is the \(m_{\mathrm{probe}}=2\) floor, not adaptivity.** A hard
   2/12 split leaves half the batch at a budget where \(\operatorname{Var}(\hat u)=A/2\)
   dominates the whole comparison, and a rank statistic over a mixed population is
   set by its noisiest half. A smooth proportional rule never assigns \(M=2\) to a
   state with meaningful \(A\), and recovers or beats uniform. Novelty B is
   salvageable as *Neyman allocation for MC-UBE*, which is a closed-form
   estimator-theory statement rather than a heuristic schedule — but not in its
   current top-\(k\) form.

---

## 5. What the coupling identity does to \(A\)

`MCUBELocalRewards.combine_coupled` already handles a shared-latent coupling for
the bias, where the constant becomes \((g^\star-\bar\Sigma)/M\), the mean pairwise
disagreement variance. The measured `sd(w_deb)` column in
`theory/estimator_bias.py` Part 5 falls monotonically in \(\rho\)
(0.114 → 0.055 from \(\rho=0\) to \(\rho=0.99\)), i.e. coupling shrinks the variance
too, by roughly \(2\times\) at high \(\rho\). Deriving the coupled analogue of \(A\) in
closed form is the obvious next step and is **not done here**; the numerical
column is the only evidence so far.

Note also that FLARE (arXiv:2602.09170) conditions on a single shared reverse
path for the same reason, independently. The shared-latent device is prior art;
the transferable contribution is the identity and the estimator that stays
unbiased at any coupling.

---

## 6. Loose end: the Gaussian branch is not bias-free either

`ESTIMATOR-BIAS-FINDING.md` §7 says the bias "does not arise for Gaussian
ensembles, because there the member means are exact and \(\tau_i^2=0\)". The member
means are exact **as plug-ins**, but `mc_ube.estimate`'s Gaussian branch computes
`mu = q_fn(next_mean, policy_fn(next_mean))`, i.e. \(Q(\mathbb E[s'])\), whereas the
estimand is \(\mathbb E[Q(s')]\). The gap is
\(\tfrac12\operatorname{tr}(\nabla^2 Q\,\Sigma)+\dots\) — a curvature bias, deterministic
rather than stochastic, and not removable by sampling.

So the honest asymmetry is not "diffusion has a bias, Gaussian does not". It is:

| | member mean | bias in \(w\) | removable? |
|---|---|---|---|
| diffusion / implicit | \(M\)-sample MC | stochastic, \(O(1/M)\), \(\propto g^\star\) | yes, closed form, free |
| Gaussian ensemble | closed-form plug-in | deterministic Jensen gap \(\propto\nabla^2 Q\cdot\Sigma\) | no |

That is a *better* story than the current one — the diffusion bias is the one you
can actually fix — but it must be stated correctly. `adaptive_mc`'s
`gaussian_mean_blend` already exposes the knob that interpolates between the two
estimands; nothing measures the Jensen gap yet.

---

## 7. Open items

1. Closed-form \(A\) under a coupling (\(\S5\)). Currently numerical only.
2. Implementable Neyman rule: estimate \(A_s\) from the probe's own \(\hat g_s\) and
   fourth-moment sample, allocate \(M_s\propto\sqrt{\hat A_s}\) with a floor, and
   compare against uniform and top-\(w\) on real models. Synthetic evidence only so far.
3. Measure the Gaussian Jensen gap (\(\S6\)) so the model-class comparison is not
   confounded by two different biases.
4. Bernstein-type concentration for \(\hat u\) carrying \(A\), replacing the
   Hoeffding envelope in Theorem A1, which is too loose to see either moment.
5. The \(\lambda\)-vs-\(M\) prediction in `ESTIMATOR-BIAS-FINDING.md` §6 is still unrun.
