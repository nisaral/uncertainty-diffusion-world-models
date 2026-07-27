# Theorem A1 — MC concentration for local UBE rewards (proof sketch)

**Status:** Sketch / first attempt — fill constants carefully before claiming in a paper.  
**Depends on:** Assumptions A1–A4 in `research/ASSUMPTIONS-LOG.md`.

---

## Statement (working)

**Setting.** Fix \((s,a)\), ensemble size \(N\), and member kernels \(p_{\theta_1},\ldots,p_{\theta_N}\).  
Let \(\bar Q:\mathcal{S}\times\mathcal{A}\to\mathbb{R}\) with \(\lvert\bar Q\rvert\le Q_{\max}\).  
For each member \(i\), draw i.i.d. pairs \((s'_{i,m}, a'_{i,m})\) with

\[
s'_{i,m}\sim p_{\theta_i}(\cdot\mid s,a),\qquad
a'_{i,m}\sim\pi(\cdot\mid s'_{i,m}),
\qquad m=1,\ldots,M,
\]

and set \(y_{i,m}=\bar Q(s'_{i,m},a'_{i,m})\).

Define population member means and the epistemic local reward:

\[
\mu_i^\star
=
\mathbb{E}[y_{i,1}],
\qquad
w^\star
=
\frac{1}{N}\sum_{i=1}^N (\mu_i^\star - \bar\mu^\star)^2,
\quad
\bar\mu^\star=\frac1N\sum_i\mu_i^\star
\]

(or the unbiased \(1/(N-1)\) form — pick one and stick to it).

Estimator:

\[
\hat\mu_i=\frac1M\sum_{m=1}^M y_{i,m},
\qquad
\hat w
=
\frac1N\sum_{i=1}^N (\hat\mu_i - \hat{\bar\mu})^2.
\]

**Theorem A1.** For any \(\varepsilon\in(0,1)\), \(\delta\in(0,1)\), there exists

\[
M_0
=
O\!\Big(
\frac{Q_{\max}^2}{\varepsilon^2}
\log\frac{N}{\delta}
\Big)
\]

such that if \(M\ge M_0\), then \(\mathbb{P}(\lvert\hat w - w^\star\rvert > \varepsilon) \le \delta\).

---

## Proof outline

### Step 1 — Concentrate each \(\hat\mu_i\)

\(y_{i,m}\in[-Q_{\max},Q_{\max}]\). Hoeffding:

\[
\mathbb{P}\big(\lvert\hat\mu_i-\mu_i^\star\rvert > t\big)
\le
2\exp\!\Big(-\frac{M t^2}{2Q_{\max}^2}\Big).
\]

Union bound over \(i=1..N\): with prob. \(\ge 1-\delta/2\),

\[
\max_i\lvert\hat\mu_i-\mu_i^\star\rvert \le t
\quad\text{for}\quad
t = Q_{\max}\sqrt{\frac{2}{M}\log\frac{4N}{\delta}}.
\]

### Step 2 — Lipschitz of sample variance map

Consider \(f:\mathbb{R}^N\to\mathbb{R}\), \(f(\mu)=\frac1N\sum_i(\mu_i-\bar\mu)^2\).  
If \(\lVert\mu-\mu'\rVert_\infty\le t\) and \(\lvert\mu_i\rvert\le Q_{\max}\), then

\[
\lvert f(\mu)-f(\mu')\rvert
\le
L(N,Q_{\max})\, t
\]

with \(L = O(Q_{\max})\) (differentiate or bound:

\[
\lvert(\mu_i-\bar\mu)^2 - (\mu_i'-\bar\mu')^2\rvert
\le
2Q_{\max}\cdot(2t)
\]

etc.). Thus \(\lvert\hat w - w^\star\rvert\le L t\).

### Step 3 — Choose \(M\)

Set \(L t \le \varepsilon\) ⇒ \(M \ge C Q_{\max}^2 L^2 \varepsilon^{-2}\log(N/\delta)\).

### Step 4 — Same for \(\hat g\) and \(\hat u\)

Per-member sample variance \(\hat v_i\) concentrates about \(v_i^\star=\mathrm{Var}(y_{i,1})\) (use sample-variance concentration for bounded r.v., or Bernstein).  
\(\hat g = N^{-1}\sum_i\hat v_i\), \(\hat u=\hat w-\hat g\).  
Union bound absorbs extra \(\delta\) terms.

### Step 5 — Corollary A2 (UBE perturbation)

UBE operator \(T_u U = \gamma^2 u + \gamma^2 P^{\pi,\bar p} U\) is a \(\gamma^2\)-contraction in \(\lVert\cdot\rVert_\infty\).  
If \(\lVert\hat u - u\rVert_\infty\le\varepsilon\), then fixed points satisfy

\[
\lVert\tilde U - U^\star\rVert_\infty
\le
\frac{\gamma^2}{1-\gamma^2}\varepsilon.
\]

---

## Gaps to close before submission

1. **Exact Lipschitz constant** \(L(N,Q_{\max})\) written cleanly.
2. **Biased vs unbiased** variance formula consistency between \(w^\star\) and \(\hat w\).
3. **Action sampling:** if \(J>1\) or \(J=1\), state the estimand clearly (Luis uses \(J=1\) for \(w\)).
4. **Dependent samples** from a single reverse chain (MCMC-style) — prefer independent restarts of the sampler.
5. **Distillation:** A1 assumes exact \(p_{\theta_i}\); B1 handles \(K\)-step law mismatch as **bias**, not variance.

---

## Toy corollary you can check in code

In `theory/toy_mc_ube_estimator.py`, log-log plot of mean \(\lvert\hat w - w^\star\rvert\) vs \(M\) should show slope \(\approx -1/2\) for fixed \(N\), matching Hoeffding-type rates.
