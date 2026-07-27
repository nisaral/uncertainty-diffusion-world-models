# Sample-Based UBE Estimation for Diffusion / Consistency World Models

**Status:** Active research target (revised 2026-07-18)  
**Claim type:** Estimator validity + concentration (not a new UBE recursion)  
**Primary application:** DIAMOND-style image-space diffusion world models; continuous-control diffusion dynamics  

---

## 0. One-sentence pitch

Luis et al. (AISTATS 2023) already give a **distribution-agnostic** Uncertainty Bellman Equation (UBE) for posterior value variance; what is missing is a **tractable, bias-characterized Monte Carlo estimator** of the local uncertainty \(u_t(s,a)\) when ensemble members are **implicit generative models** (diffusion / consistency) rather than Gaussian MLPs.

---

## 1. Motivation (live application)

**DIAMOND** (Alonso et al., NeurIPS 2024 Spotlight) trains an RL agent entirely inside a diffusion world model and reports a mean human-normalized score of **1.46** on Atari 100k — the best result at publication time for agents trained fully in a world model. That makes diffusion dynamics a *deployed*, not hypothetical, world-model class. Any Diffusion-UBE result is directly relevant to that stack (and to continuous-control diffusion dynamics / distilled few-step samplers used for imagination speed).

**Why UBE still matters for that stack**

| Use case | Role of multi-step epistemic \(U_t\) |
|---|---|
| Exploration in imagination | OFU-style \( \bar Q + \lambda\sqrt{U} \) (Luis online) |
| Offline / risk-averse | Pessimism \( \bar Q - \lambda\sqrt{U} \) (QU-SAC / MOPO-style) |
| Data reweighting | Prefer synthetic transitions with calibrated uncertainty |

One-step ensemble variance of next-state means **does not** propagate epistemic value uncertainty over the imagination horizon the way a UBE does.

---

## 2. What Luis et al. actually prove (and do not)

### 2.1 Theorem 1 is distribution-agnostic

**Source:** Luis et al., *Model-Based Uncertainty in Value Functions*, AISTATS 2023 (arXiv:2302.12526), Theorem 1 + Appendix A.1.

**Assumptions used in the proof (Lemmas 1–3):**

1. **Assumption 1 (Independent transitions):** \(p(s'\mid x,a)\) and \(p(s'\mid y,a)\) are independent if \(x \neq y\).
2. **Assumption 2 (Acyclic MDP):** states are not revisited within an episode (finite-horizon / time-augmented reduction OK).

**Theorem 1 (informal).** Under Assumps. 1–2, for any policy \(\pi\),

\[
U_t^\pi(s)
=
\gamma^2 u_t(s)
+
\gamma^2
\sum_{a,s'}
\pi(a\mid s)\,
\bar p_t(s'\mid s,a)\,
U_t^\pi(s'),
\]

where \(U_t^\pi(s) = \mathrm{Var}_{p\sim\Phi_t}\!\big[V^{\pi,p}(s)\big]\) and the **local uncertainty** is

\[
u_t(s)
=
\mathrm{Var}_{a,s'\sim\pi,\bar p_t}\!\big[\bar V_t^\pi(s')\big]
-
\mathbb{E}_{p\sim\Phi_t}\!
\Big[
\mathrm{Var}_{a,s'\sim\pi,p}\!\big[V^{\pi,p}(s')\big]
\Big].
\]

**Critical observation (hand-check priority):** Nowhere in Appendix A.1 does the proof assume Gaussian transition noise, Gaussian posteriors, or finite support of \(\Phi_t\). The recursion holds for **any** posterior \(\Phi_t\) over MDPs satisfying Assumps. 1–2. The objects \(u_t\) and \(\bar p_t\) are defined via expectations/variances under that posterior — they do not require closed form.

**Implication for this project:** You do **not** need to “extend Theorem 1 to diffusion” in the mathematical sense. Diffusion world models already induce a posterior (or ensemble-as-posterior) over transition kernels; Theorem 1 applies at that level of abstraction.

### 2.2 Where Gaussians actually enter (implementation only)

**Section 4 (Deep RL implementation)** chooses a practical posterior:

> “we consider \(\Gamma_t\) to be a discrete uniform distribution of \(N\) probabilistic neural networks … that output the mean and covariance of a **Gaussian** distribution over next states and rewards.”

This is a **computational** choice so that local terms \(u_t(s,a)\) / \(w_t(s,a)\) have cheap estimators.

**Appendix D.1 (Uncertainty reward estimation)** — what closed form buys them:

| Quantity | Gaussian ensemble practice (Luis App. D.1) | Why it is easy |
|---|---|---|
| Next-state point estimate for \(w_t\) | Take \(s' = \mu_i(s,a)\) (mean of ensemble member \(i\)'s Gaussian) | One forward pass, no sampling of \(s'\) |
| Inner expectation under \(\pi\) | Single sample \(a'\sim\pi(\cdot\mid s')\) at that mean | Cheap |
| Gap / aleatoric term in \(u_t\) | Finite-sample: ~10 actions from \(\pi\); Gaussian structure for \(s'\) | Variance under Gaussian is closed-form or low-MC |
| Ensemble epistemic part of \(w_t\) | Variance **across** \(N\) member means of next-value | \(O(N)\) critic evals |

So the Gaussian assumption is **not** a hypothesis of Theorem 1; it is an **estimator design** that makes \(u_t,w_t\) nearly free. Diffusion removes that free lunch.

### 2.3 Theorem 2 gap (still useful language)

\[
u_t(s) = w_t(s) - g_t(s),\qquad g_t(s)\ge 0,
\]

where \(g_t\) is average aleatoric next-value noise. Zhou-style \(w_t\) alone overestimates epistemic local uncertainty. For diffusion, **both** \(w_t\) and \(g_t\) become sample-based — the same estimator story.

---

## 3. Revised research gap (precise statement)

### What is **not** the gap

- Re-deriving a new Bellman recursion for diffusion (Theorem 1 already covers general \(\Phi_t\)).
- Claiming “UBE does not apply to multi-modal dynamics” without qualification (it applies; estimation is hard).
- Pure empirical “diffusion is better than Gaussian for world models” (DIAMOND already).

### What **is** the gap

> **Gap (estimator):** Given an ensemble of diffusion / consistency world models \(\{p_{\theta_i}\}_{i=1}^N\) that define **implicit** one-step kernels \(p_{\theta_i}(s'\mid s,a)\) only through a (possibly distilled) generative sampler, construct estimators \(\hat u_t(s,a)\) and \(\hat w_t(s,a)\) of the local UBE rewards, and prove finite-sample guarantees relating \(\hat u_t\) to the population \(u_t\) (bias, variance, sample complexity). Optionally characterize **additional** bias from using a \(K\)-step distilled sampler instead of the full reverse process.

This is smaller, cleaner, and more reviewer-honest than “we extend the UBE to diffusion.”

---

## 4. Revised ranking of contributions

| ID | Contribution | Phase | Novelty type | Risk |
|---|---|---|---|---|
| **(A)** | Sample-based UBE local-reward estimator for diffusion/consistency ensembles + MC concentration | **Primary** | Estimator theory | Medium — tractable |
| **(B)** | Distillation-aware bias/variance of (A) under 1–4 step samplers vs full diffusion | Phase 2 on (A) | Structured error term | Medium |
| **(C)** | Plug validated \(\hat U_t\) into offline pessimism (MOPO / QU-SAC-style) | Systems + theory | Production story | Lower theory risk |
| **(D)** | Empirical calibration only | Fallback / workshop | Empirical | Low |

**(A) is the strongest, most tractable target** now that Theorem 1 is known to be distribution-agnostic.

---

## 5. Formal objects (write before any large codebase)

### 5.1 Posterior / ensemble

Treat the diffusion ensemble as a discrete Bayesian model average (same modeling choice as Luis Section 4, non-Gaussian members):

\[
\Phi_t = \mathrm{Unif}\{\theta_1,\ldots,\theta_N\},
\qquad
p\sim\Phi_t \;\Rightarrow\; p = p_\theta.
\]

Each \(p_\theta(\cdot\mid s,a)\) is the **pushforward** of the reverse diffusion (or consistency map) conditioned on \((s,a)\). We never need density evaluation — only samples.

Mean dynamics (mixture over ensemble + aleatoric samples):

\[
\bar p_t(s'\mid s,a)
=
\frac{1}{N}\sum_{i=1}^N p_{\theta_i}(s'\mid s,a).
\]

### 5.2 State-action local rewards (practical form)

Following Luis Appendix B.3 (Q-form used in deep RL):

\[
\begin{aligned}
w_t(s,a)
&=
\mathrm{Var}_{p\sim\Phi_t}
\Big[
\mathbb{E}_{s'\sim p(\cdot\mid s,a)}\,
\mathbb{E}_{a'\sim\pi(\cdot\mid s')}\,
\bar Q_t^\pi(s',a')
\Big],
\\[0.5em]
u_t(s,a)
&=
w_t(s,a) - g_t(s,a),
\end{aligned}
\]

with \(g_t\) the expected aleatoric variance of next Q under each member (exact formulas as in Luis B.3). For exposition we write the MC estimators below for \(w_t\); \(u_t\) adds one more nested variance estimate.

### 5.3 Diffusion ensemble Monte Carlo estimator (core of claim A)

**Inputs:** ensemble \(\{\theta_i\}_{i=1}^N\), policy \(\pi\), mean critic \(\bar Q\), state-action \((s,a)\),  
sample budgets: \(M\) next-state samples per member, \(J\) action samples (or 1 as in Luis).

**Algorithm `MC-UBE-Local`:**

```
for i = 1..N:
    for m = 1..M:
        s'_{i,m}  ~  SampleDiffusion(θ_i, s, a; K steps)   # K=full or distilled
        a'_{i,m}  ~  π(· | s'_{i,m})                         # J=1 default
        y_{i,m}   =  Q̄(s'_{i,m}, a'_{i,m})
    μ_i  =  mean_m(y_{i,m})          # E_{s',a'|θ_i}[Q̄]
    v_i  =  var_m(y_{i,m})           # aleatoric proxy under member i
ŵ(s,a) = var_i(μ_i)                  # epistemic: disagreement of means
ĝ(s,a) = mean_i(v_i)                 # average aleatoric
û(s,a) = ŵ(s,a) - ĝ(s,a)             # or clip max(u_min, û)
```

**Comparison to Luis Gaussian estimator:**

| Step | Luis (Gaussian MLP) | Diffusion MC |
|---|---|---|
| \(s'\) | Closed-form mean \(\mu_i(s,a)\) | \(M\) reverse-process samples |
| \(\mathrm{Var}_{s'}[\cdot]\) | Closed-form or 1-sample at mean | Sample variance over \(M\) |
| Cost per \((s,a)\) | \(O(N)\) forwards | \(O(N\cdot M\cdot K)\) denoising steps |

### 5.4 UBE solve (unchanged)

Train / iterate \(U_\varphi\) with targets

\[
z = \gamma^2 \hat u(s,a) + \gamma^2 U_{\bar\varphi}(s',a'),
\]

exactly as Luis deep RL — only the **local reward estimator** changes.

---

## 6. First theorem attempt (estimator, not recursion)

### 6.1 Setup

Assume bounded next-values: \(\lvert \bar Q(s',a')\rvert \le Q_{\max}\) almost surely (true if rewards bounded and \(\gamma<1\)).  
Let \(w^\star(s,a)\) be the population local reward under the **true** ensemble kernels \(\{p_{\theta_i}\}\) (infinite samples, exact sampler).  
Let \(\hat w_{N,M}\) be the MC estimator of Section 5.3 (finite \(M\); \(N\) fixed as part of \(\Phi_t\)).

### 6.2 Theorem A1 (Hoeffding-style sample complexity for \(\hat w\)) — *conjecture to prove*

**Claim.** Fix ensemble \(\{\theta_i\}\) and \((s,a)\). Suppose each next-value sample \(y_{i,m}\) is independent, \(\lvert y_{i,m}\rvert\le Q_{\max}\), and the diffusion sampler is **exact** for \(p_{\theta_i}\) (no distillation bias). Then for any \(\varepsilon>0\), \(\delta\in(0,1)\), if

\[
M
\;\ge\;
\frac{2 Q_{\max}^2}{\varepsilon^2}
\log\!\Big(\frac{2N}{\delta}\Big)
\]

(or a similar constant from Hoeffding / Bernstein), then with probability at least \(1-\delta\),

\[
\big\lvert \hat w_{N,M}(s,a) - w^\star(s,a)\big\rvert
\;\le\;
\varepsilon
\]

after accounting for the continuous mapping from member-means \(\{\mu_i\}\) to sample variance (use bounded differences / McDiarmid on the vector of \(\mu_i\)).

**Proof strategy (Track 1 + Track 4):**

1. For each \(i\), Hoeffding ⇒ \(\mu_i\) concentrates around \(\mathbb{E}[y\mid\theta_i]\).
2. Sample variance of bounded vectors \(\{\mu_i\}\) is Lipschitz in \(\ell_\infty\) (or use McDiarmid on all \(NM\) samples jointly).
3. Union bound over \(N\) members.
4. Same argument for \(\hat g\) (mean of per-member sample variances) ⇒ \(\hat u = \hat w - \hat g\).

**What this theorem is *not*:** It does not re-prove UBE. It says: *if you plug \(\hat u\) into the existing UBE operator, the local reward is \(\varepsilon\)-accurate w.h.p., hence the fixed point of the approximate UBE is close to true \(U_t\) by \(\gamma^2\)-contraction* (standard Bellman perturbation: \(\lVert U - \tilde U\rVert_\infty \le \frac{\gamma^2}{1-\gamma^2}\varepsilon\)).

### 6.3 Corollary A2 (UBE fixed-point perturbation)

If \(\lvert \hat u(s,a) - u(s,a)\rvert \le \varepsilon\) for all \((s,a)\), and \(T_u\), \(T_{\hat u}\) are the UBE Bellman operators, then

\[
\lVert U^\star - \tilde U\rVert_\infty
\le
\frac{\gamma^2}{1-\gamma^2}\,\varepsilon.
\]

This is the “guarantee preservation” story for reviewers: **naive MC preserves Theorem 1 up to explicit sample complexity**, under exact sampling.

### 6.4 Theorem B1 (distillation bias) — Phase 2

Let \(p_{\theta}^{(K)}\) be the law of a \(K\)-step sampler and \(p_{\theta}^{(\infty)}\) the ideal reverse process. Define total variation (or Wasserstein) error \(\Delta_K(s,a)\). Then

\[
\big\lvert w^{(K)}(s,a) - w^{(\infty)}(s,a)\big\rvert
\le
f(Q_{\max}, \Delta_K, N),
\]

and \(\hat w\) under distilled sampling estimates \(w^{(K)}\), not \(w^{(\infty)}\).  
**Practical message:** report calibration of \(\hat U\) **at the sampler used for imagination** (the object the agent actually rolls out).

---

## 7. Assumption dependencies (chart)

```
Assump. 1 (indep. transitions) ──┐
Assump. 2 (acyclic MDP) ─────────┼──► Theorem 1: U_t = UBE(u_t)   [exact, any Φ_t]
                                 │
                                 └──► Theorem 2: u = w − g ≤ w

Φ_t = ensemble of Gaussians ─────► closed-form / 1-mean estimator (Luis D.1)
Φ_t = ensemble of diffusion ─────► need MC estimator (THIS WORK, A)
     + distilled K-step ─────────► extra bias Δ_K (THIS WORK, B)

Function approx (U-net, Q-net) ──► inherited practical gap (Luis also has this)
Violations of 1–2 in deep RL ───► same caveats as Luis; clip u_min ≥ 0
```

**Honest limitations to keep in the paper**

- Ensemble ≠ true Bayesian posterior (same as PETS / Luis deep RL).
- Image-space diffusion: \(s'\) is high-dimensional; variance of \(\bar Q(s',a')\) is still 1-D (critic reduces dimension) — estimators stay on scalar next-values, not full pixel covariance.
- Nested MC cost: mitigate with common random numbers, shared noise across ensemble, few-step DDIM/consistency, and caching.

---

## 8. Literature positioning (delta chain)

| Paper | What it established | Remaining assumption / limit |
|---|---|---|
| O'Donoghue et al. 2018 | UBE upper-bounds posterior var | Loose local rewards |
| Zhou et al. 2020 | Tighter local \(w_t\) | Ignores aleatoric next-value noise |
| Luis et al. 2023 | Exact \(u_t\); characterizes Zhou gap \(g_t\) | Deep RL estimator uses **Gaussian** ensembles |
| Luis et al. 2024 (QU-SAC) | Offline + online with UBE | Same Gaussian/ensemble dynamics |
| DIAMOND 2024 | Diffusion WM SOTA Atari-in-imagination | No Luis-style multi-step epistemic UBE |
| **This work** | **MC estimators + concentration for \(u_t\) under diffusion/consistency ensembles** | — |

**Literature check (re-run every 3–4 weeks):**  
`("uncertainty Bellman" OR UBE) + diffusion`, `epistemic uncertainty + diffusion world model + Bellman`, `consistency distillation + value variance`. As of this writeup, no mainstream UBE-line paper closes the **sample-based local reward** gap for diffusion dynamics.

---

## 9. Implementation plan (theory-aligned)

### Phase T0 — Hand proofs (this week)

- [ ] Reproduce Appendix A.1 Lemmas 1–3 on paper; mark every step that would break for non-Gaussian \(p\).
- [ ] Write explicit \(u_t,w_t\) formulas for Q-values (Luis B.3) side-by-side with MC estimators.
- [ ] Prove toy version of Theorem A1 in 1D (complete; see `theory/toy_mc_ube_estimator.py`).

### Phase T1 — Toy numerical validation

- [ ] Finite acyclic MRP / 1D continuous state where ground-truth \(U_t\) is computable by enumerating ensemble MDPs.
- [ ] Compare: (i) closed-form Gaussian \(u_t\), (ii) MC with exact Gaussian samples, (iii) MC with multi-step denoising toy diffusion that matches the same marginals.
- [ ] Plot \(\lvert\hat u - u^\star\rvert\) vs \(M\); check \(1/\sqrt{M}\) rate and Corollary A2 on \(\lVert\hat U - U\rVert\).

### Phase T2 — Distillation stress test (claim B)

- [ ] Same toy: full reverse vs 1–4 step consistency/DDIM student; measure bias in \(\hat u\) and downstream policy regret / pessimism error.

### Phase T3 — Systems integration

- [ ] Swap PETS Gaussian ensemble in MBPO+UBE for diffusion dynamics (state-space first; pixels later).
- [ ] Offline: D4RL with \(\bar Q - \lambda\sqrt{\hat U}_{\mathrm{diff}}\).
- [ ] Calibration: predicted \(\sqrt{U}\) vs empirical squared Bellman residuals.

### Phase T4 — Optional pixels / DIAMOND-scale

- Only after T1–T3. Ensemble of full DIAMOND models is expensive; start with latent diffusion or multi-checkpoint SWA as epistemic proxy, stated honestly as approximation.

---

## 10. Paper narrative (revised Step 5 positioning)

```text
1. Epistemic value uncertainty drives exploration and offline pessimism.
2. O'Donoghue → Zhou → Luis: UBE becomes exact for posterior Var(V)
   under Assumps. 1–2 — with no Gaussian requirement in the theorem.
3. Practical UBE systems (Luis deep RL, QU-SAC) estimate local rewards
   with Gaussian ensemble MLPs (closed-form means / easy variances).
4. Parallel track: diffusion world models (DIAMOND) are now competitive
   for imagination-based RL, but define p(s'|s,a) only implicitly.
5. Gap: not a missing recursion — a missing, validated sample-based
   estimator of u_t / w_t for generative dynamics, with finite-sample
   guarantees and distillation error characterization.
6. This paper: MC-UBE-Local + concentration (A), distillation bias (B),
   and QU-SAC-style use of Û_diff (C).
```

**Title sketches**

- *Monte Carlo Uncertainty Bellman Rewards for Diffusion World Models*
- *Estimating Local Epistemic Uncertainty under Implicit Generative Dynamics*
- *Sample-Efficient UBE Local Rewards when the World Model is a Diffusion Ensemble*

---

## 11. What to write next (ordered)

1. Hand proof reproduction notes → `research/proofs/luis-theorem1-reproduction.md`
2. Fully expanded Theorem A1 proof → `research/proofs/theorem-A1-mc-concentration.md`
3. Assumptions log (living) → `research/ASSUMPTIONS-LOG.md`
4. Keep this file as the north-star claim document; update after each arXiv re-check.

---

## 12. Key references

- Luis et al., 2023. *Model-Based Uncertainty in Value Functions.* AISTATS. arXiv:2302.12526  
- O'Donoghue et al., 2018. *The Uncertainty Bellman Equation and Exploration.* ICML.  
- Zhou et al., 2020. (UBE upper bound; local \(w_t\))  
- Alonso et al., 2024. *Diffusion for World Modeling: Visual Details Matter in Atari.* NeurIPS (DIAMOND). arXiv:2405.12399  
- Song et al., 2023. Consistency Models.  
- Janner et al., 2019. MBPO.  
- Code lineage: `boschresearch/ube-mbrl`, `eloialonso/diamond`
