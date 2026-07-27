# Luis Theorem 1 — Hand-reproduction checklist

**Goal:** Confirm that Appendix A.1 (Lemmas 1–3 → Theorem 1) never uses Gaussian structure.  
**Paper:** Luis et al., AISTATS 2023, arXiv:2302.12526  
**Status:** Checklist for manual derivation (do on paper; tick when done)

---

## Setup (copy to paper)

- MDP \(\mathcal{M} = \{\mathcal{S},\mathcal{A},p,\rho,r,\gamma\}\), finite \(\mathcal{S},\mathcal{A}\) in theory section.
- \(V^{\pi,p}(s) = \mathbb{E}_\tau[\sum_h \gamma^h r(s_h,a_h) \mid s_0=s]\).
- Posterior \(\Phi_t\) over transition kernels \(p\); \(\bar p_t = \mathbb{E}_{p\sim\Phi_t}[p]\); \(\bar V_t^\pi = \mathbb{E}_{p\sim\Phi_t}[V^{\pi,p}]\).
- Assump. 1: independent transitions at distinct states.
- Assump. 2: acyclic (DAG) episodes.

---

## Lemma 1 (typical content to verify)

**Claim pattern:** Under Assump. 2 (and independence), \(V^{\pi,p}(s')\) is independent of (or uncorrelated with) the one-step kernel \(p(\cdot\mid s,a)\) for \(s'\) reachable in one step without cycles.

- [ ] Write the statement exactly as in the paper.
- [ ] Proof uses only: graph structure / no revisit, not Gaussian.
- [ ] Note: still true if \(p(\cdot\mid s,a)\) is a diffusion-induced measure?

**Diffusion note:** Independence is about **which random variables** (kernel at \(s\) vs value at descendant \(s'\)), not about the **shape** of \(p(\cdot\mid s,a)\). If two states do not share parameters in \(\Phi_t\), Assump. 1 is analogous; with shared score-network weights, Assump. 1 fails the same way Gaussian ensembles with shared architecture also only *approximately* satisfy it.

---

## Lemma 2–3

- [ ] Reproduce: uncorrelatedness of \(V^{\pi,p}(s')\) and \(p(s'\mid s,a)\).
- [ ] Mark every use of \(\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y]\) or law of total variance.
- [ ] Confirm no step requires \(\mathrm{Var}(s')=\Sigma\) closed form.

---

## Theorem 1 main recursion

Start from Bellman:

\[
V^{\pi,p}(s) = \sum_a \pi(a\mid s)\Big(r(s,a) + \gamma \sum_{s'} p(s'\mid s,a) V^{\pi,p}(s')\Big).
\]

- [ ] Take \(\mathrm{Var}_{p\sim\Phi_t}\) of both sides.
- [ ] Expand using law of total variance / uncorrelatedness lemmas.
- [ ] Arrive at \(U(s) = \gamma^2 u(s) + \gamma^2 \mathbb{E}_{a,s'\sim\pi,\bar p}[U(s')]\).
- [ ] Identify \(u(s)\) as total var of mean next-values minus expected aleatoric var of true next-values.

**Checkpoint:** If every line is measure-theoretic (expectations under \(p\) and \(\Phi_t\)) without densities, **diffusion is in scope** for the theorem.

---

## Theorem 2 (gap)

- [ ] Show \(u = w - g\), \(g\ge 0\).
- [ ] Recover Zhou bound as over-generous local reward.

---

## Section 4 / App D.1 (estimator only)

- [ ] Quote: ensemble of Gaussian NNs is a **choice of \(\Gamma_t\)**, not a theorem hypothesis.
- [ ] Quote: \(s'\) taken as **mean of Gaussian** for \(w_t\) finite-sample estimate.
- [ ] List every place code would need a sample from \(p_i\) instead of \(\mu_i\).

---

## Outcome of this checklist

When complete, you should be able to say in a paper:

> “Luis et al. establish that posterior value variance obeys an exact UBE under Assumps. 1–2 for general posteriors over transition kernels. Our contribution is orthogonal: we replace their Gaussian closed-form local-reward estimator with a Monte Carlo estimator valid for implicit generative dynamics, and prove finite-sample accuracy of that estimator.”

Date completed: ________  
Notes / surprises: ________
