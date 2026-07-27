# Detailed Learning Notes + Gap Analysis
## RL + World Models + Diffusion + Theoretical UQ

**Purpose of this document**
1. Self-contained study notes for every topic in `full-learning-roadmap-theory-gaps.md` — simple intuition *and* technical depth, with math, theory, practical code patterns, and production notes.
2. Explicit reconstruction of the **gap-finding process** from that roadmap.
3. A researched **novelty analysis**: which gaps are still real (as of mid-2026), which are already partially closed, and which paper-worthy directions are most legitimate.

**Depth tags (same as the roadmap)**
- **[USE]** — implement / apply; no need to re-derive
- **[DERIVE]** — should be able to prove by hand
- **[EXTEND]** — novelty lives here

**Primary paper spine (memorize this chain)**
```
O'Donoghue et al. 2018 (UBE upper bound, exploration)
        ↓ (loose bound in practice — Markou & Rasmussen 2019)
Zhou et al. 2020 (tighter UBE upper bound; ignores aleatoric in local uncertainty)
        ↓ (gap = ignored aleatoric term)
Luis et al. 2023 AISTATS (exact UBE = posterior variance; characterizes Zhou gap)
        ↓ (same theory → algorithm)
Luis et al. 2024 (QU-SAC: risk-seeking online / risk-averse offline)
        ↓  ← YOUR LIKELY NOVELTY LIVES HERE
Diffusion / consistency world models as the posterior (or as a new noise source)
```

---

# PART I — DETAILED NOTES BY TRACK

---

## TRACK 1 — Math Foundations

### 1.1 Linear Algebra (eigendecomposition, SVD) — [USE]

**Simple explanation**
Matrices stretch and rotate space. Eigenvectors are directions that only get scaled (not rotated). SVD is the universal “stretch + rotate + stretch” factorization that works for *any* matrix, not just square ones.

**Technical**
- Eigendecomposition: \(A = Q\Lambda Q^{-1}\) for diagonalizable \(A\).
- SVD: \(A = U\Sigma V^\top\) with orthogonal \(U,V\), singular values on \(\Sigma\).
- In deep learning: PCA / whitening of latents, low-rank adapters, analyzing Jacobians of dynamics models, spectral norms for Lipschitz bounds (useful if you ever prove contraction of a learned Bellman operator).

**Practical**
```python
import torch
# Condition number of a Jacobian ≈ ratio of largest/smallest singular values
J = torch.randn(64, 32)
U, S, Vh = torch.linalg.svd(J, full_matrices=False)
cond = S[0] / S[-1]
```
You need this to *read* architecture papers, not to prove UBE theorems.

---

### 1.2 Multivariate calculus, gradients, Jacobians — [USE]

**Simple**
Gradient = “which way does the scalar go up fastest.” Jacobian = all partial derivatives of a vector-valued map (e.g. dynamics \(s' = f_\theta(s,a)\)).

**Technical**
- \(\nabla_\theta L \in \mathbb{R}^d\)
- Jacobian \(J_f(x)_{ij} = \partial f_i / \partial x_j\)
- Chain rule is backprop: \(\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} J_f(x)\)
- Hessian \(H_{ij} = \partial^2 L / \partial \theta_i \partial \theta_j\) — shows up in Laplace approximations for epistemic UQ

**Production note**
Use `torch.func.jacrev` / `vmap` for vectorized Jacobians of ensemble members rather than finite differences (slow, unstable).

---

### 1.3 Probability fluency — [DERIVE]

**Must-know objects**
| Object | Notation | Meaning |
|---|---|---|
| Random variable | \(X\) | Map from outcomes to numbers |
| Joint | \(p(x,y)\) | Simultaneous behavior |
| Marginal | \(p(x)=\int p(x,y)\,dy\) | “forget y” |
| Conditional | \(p(y\|x)=p(x,y)/p(x)\) | “given we know x” |
| Expectation | \(\mathbb{E}[X]=\int x\,p(x)\,dx\) | average under p |
| Variance | \(\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2]\) | spread |

**Derive by hand (no notes):** Bayes rule
\[
p(\theta\mid D)=\frac{p(D\mid\theta)\,p(\theta)}{p(D)},\qquad p(D)=\int p(D\mid\theta)p(\theta)\,d\theta
\]
Posterior predictive:
\[
p(x^*\mid D)=\int p(x^*\mid\theta)\,p(\theta\mid D)\,d\theta
\]
This integral *is* Bayesian model averaging — ensembles approximate it with a discrete sum over MAP-ish modes.

---

### 1.4 Law of total expectation / total variance — [DERIVE] ⭐ CRITICAL

**Simple (the UQ split)**
- **Aleatoric** = noise that remains even if you know the true model (dice, wind, sensor noise).
- **Epistemic** = uncertainty because you don't know which model is true (reducible with more data).

**Technical (memorize forever)**
\[
\underbrace{\mathrm{Var}(Y)}_{\text{total}}
=
\underbrace{\mathbb{E}\!\big[\mathrm{Var}(Y\mid\theta)\big]}_{\text{aleatoric (expected noise)}}
+
\underbrace{\mathrm{Var}\!\big(\mathbb{E}[Y\mid\theta]\big)}_{\text{epistemic (spread of means)}}
\]

**Proof sketch (derive fully)**
Start from \(\mathrm{Var}(Y)=\mathbb{E}[Y^2]-(\mathbb{E}Y)^2\). Condition on \(\theta\):
\[
\mathbb{E}[Y^2]=\mathbb{E}\big[\mathbb{E}[Y^2\mid\theta]\big]
=\mathbb{E}\big[\mathrm{Var}(Y\mid\theta)+(\mathbb{E}[Y\mid\theta])^2\big]
\]
and \((\mathbb{E}Y)^2=(\mathbb{E}[\mathbb{E}[Y\mid\theta]])^2\). Rearrange → law of total variance.

**RL instantiation**
Let \(Y=V^{\pi,p}(s)\), \(\theta=p\) (transition function drawn from posterior \(\Phi_t\)). Then:
- Epistemic term = \(\mathrm{Var}_{p\sim\Phi_t}(V^{\pi,p}(s))\) ← **UBE's object**
- Aleatoric term = expected return variance under a *fixed* MDP

**Luis et al. (2023) insight in one line**
Zhou's local uncertainty \(w_t(s)\) mixes total one-step variance; the *exact* local uncertainty \(u_t(s)\) subtracts the expected aleatoric piece so that the fixed point of the UBE equals pure epistemic variance.

---

### 1.5 Bayesian inference — [DERIVE]

**Simple**
Prior = what you believe before data. Likelihood = how well each hypothesis explains data. Posterior = updated belief. Predictive = what you expect next, averaging over remaining doubt.

**UBE-as-Bayesian-RL**
- Prior \(\Phi_0\) over transition kernels \(p\)
- Data \(D_t\) → posterior \(\Phi_t = p(p\mid D_t)\)
- Induced value random variable \(V^{\pi,p}\) for \(p\sim\Phi_t\)
- Object of study: \(\mathrm{Var}_{p\sim\Phi_t}(V^{\pi,p}(s))\)

**Practical posterior approximations in MBRL**
| Method | How it approximates \(\Phi_t\) | Cost |
|---|---|---|
| Deep ensembles (PETS/MBPO style) | \(N\) independently trained probabilistic NNs | \(N\times\) train & forward |
| Bootstrap ensembles | Resample data, train \(N\) models | same |
| MC Dropout | Bernoulli mask ≈ variational posterior | cheap train, multi-forward test |
| Gaussian process (PILCO) | Exact GP posterior on dynamics | \(O(n^3)\), small data only |
| Laplace / SWAG | Local Gaussian around MAP | moderate |

**Contested point (write this in related work carefully)**
Deep ensembles are *not* a true Bayesian posterior; they often behave *like* a good approximation of the posterior *predictive* (Wilson & Izmailov; Lakshminarayanan et al. 2017). Luis et al. treat the ensemble as a discrete uniform \(\Phi_t=\{p_i\}_{i=1}^N\). Your theory should state this assumption explicitly.

---

### 1.6 Concentration inequalities — [USE], target [DERIVE] for Hoeffding

**Simple**
Concentration inequalities turn “my average is roughly the true mean” into a *probability of large error*.

**Ladder of strength**
1. **Markov**: \(P(X\ge a)\le \mathbb{E}X/a\) for \(X\ge 0\)
2. **Chebyshev**: \(P(|X-\mu|\ge t)\le \mathrm{Var}(X)/t^2\)
3. **Hoeffding**: for independent bounded \(X_i\in[a_i,b_i]\),
\[
P\!\left(\bar X - \mathbb{E}\bar X \ge t\right)
\le \exp\!\left(\frac{-2n^2 t^2}{\sum_i (b_i-a_i)^2}\right)
\]

**Derive Hoeffding outline**
Markov on \(e^{sS}\) → optimize Chernoff bound → use Hoeffding's lemma (\(\mathbb{E}e^{s(X-\mathbb{E}X)}\le e^{s^2(b-a)^2/8}\)) → product over independent vars → optimize \(s\).

**Why you need this**
- Sample complexity of tabular RL / UCB bandits
- Turning finite-ensemble variance estimates into high-probability bounds
- Calibrating confidence intervals that feed pessimistic offline RL

---

### 1.7 Stochastic processes: Markov chains, martingales — [USE], [DERIVE] Markov property

**Markov property**
\[
P(s_{t+1}\mid s_t,s_{t-1},\ldots)=P(s_{t+1}\mid s_t)
\]
This is the structural assumption that makes MDPs tractable and lets Bellman equations close.

**Martingale (intuition)**
A process \(M_t\) with \(\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=M_t\) — “fair game.” Value estimation errors under true dynamics often form (near) martingales; some UBE / regret proofs use martingale concentration (Azuma–Hoeffding).

---

### 1.8 Measure-theoretic probability — skip unless forced

Most RL theory papers stay in finite/countable spaces or assume continuous densities with mild regularity. Don't front-load measure theory.

---

### 1.9 SDEs and Itô calculus basics — [DERIVE] (score-based diffusion parts)

**Simple**
An SDE is a differential equation with a random kick each instant:
\[
dx = f(x,t)\,dt + g(x,t)\,dW_t
\]
- \(f\) = drift (deterministic push)
- \(g\) = diffusion coefficient (noise scale)
- \(W_t\) = Wiener process (Brownian motion)

**Itô's lemma (chain rule for SDEs)** — the extra \(\frac12 g^2 \partial_{xx}\) term is the key difference from ordinary calculus.

**Score-based view (Song et al. 2021)**
Forward SDE destroys data into noise. Reverse-time SDE regenerates data if you know the **score** \(\nabla_x \log p_t(x)\):
\[
dx = \big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]dt + g(t)\,d\bar W_t
\]
(time runs backwards). This is the continuous-time language in which you can define:
> “Aleatoric uncertainty injected by the diffusion process at noise level \(\sigma(t)\).”

**Fokker–Planck** — [USE]
PDE for the density \(p_t(x)\) induced by the SDE. Know the statement; don't derive from scratch unless your paper needs it.

**Minimal code mental model**
```python
# Euler–Maruyama step for reverse SDE (schematic)
score = score_net(x, t)
drift = f(x, t) - g(t)**2 * score
x = x + drift * dt + g(t) * sqrt(dt) * torch.randn_like(x)
```

---

## TRACK 2 — RL Theory

### 2.1 MDP formalism — [DERIVE]

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma,\rho_0)
\]
- \(P(s'|s,a)\): transition kernel
- \(R(s,a)\) or \(R(s,a,s')\): reward (often assumed bounded \(|R|\le R_{\max}\))
- \(\gamma\in[0,1)\): discount
- \(\rho_0\): start-state distribution

Policy \(\pi(a|s)\). Trajectory \(\tau=(s_0,a_0,r_0,s_1,\ldots)\).

Return:
\[
G_t=\sum_{k=0}^\infty \gamma^k r_{t+k}
\]

---

### 2.2 Bellman expectation & optimality — [DERIVE]

**Expectation (fixed policy)**
\[
V^\pi(s)=\mathbb{E}_{a\sim\pi}\!\big[R(s,a)+\gamma\,\mathbb{E}_{s'}V^\pi(s')\big]
\]
\[
Q^\pi(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s'}\mathbb{E}_{a'\sim\pi}Q^\pi(s',a')
\]

**Optimality**
\[
V^*(s)=\max_a Q^*(s,a),\qquad
Q^*(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s'}\max_{a'}Q^*(s',a')
\]
Optimality = expectation form with \(\max\) outside (greedy choice). Soft/entropy-regularized versions put a soft-max (log-sum-exp) instead — see SAC.

---

### 2.3 Bellman operator, contraction, Banach fixed point — [DERIVE] ⭐

**Operator**
\[
(T^\pi Q)(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s',a'\sim\pi}Q(s',a')
\]
\[
(T^* Q)(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s'}\max_{a'}Q(s',a')
\]

**γ-contraction under sup-norm**
\[
\|T^\pi Q_1 - T^\pi Q_2\|_\infty \le \gamma\|Q_1-Q_2\|_\infty
\]
**Proof idea:** the max difference of expectations is ≤ max difference of the functions; the \(\gamma\) factor comes from the discounted next-state term.

**Banach fixed-point theorem (applied form)**
A contraction on a complete metric space has a unique fixed point, and iteration from any start converges to it. \(\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}\) with \(\|\cdot\|_\infty\) is complete → value iteration converges.

**Why this is the UBE template**
Luis/O'Donoghue define an *uncertainty* Bellman operator on U-values with discount \(\gamma^2\) (because variances scale with squares of discounted rewards) and prove it is a \(\gamma^2\)-contraction → unique fixed point = (bound on) posterior variance.

---

### 2.4 Policy gradient theorem — [DERIVE]

\[
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\,G_t\right]
\]
or with baseline / advantage \(A^\pi\). REINFORCE is the Monte Carlo form; actor-critic replaces \(G_t\) with a learned critic.

---

### 2.5 Actor-critic & SAC soft Bellman — [DERIVE]

**SAC objective (entropy-regularized)**
\[
J(\pi)=\mathbb{E}\!\left[\sum_t r_t + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\right]
\]

**Soft Bellman**
\[
Q^{\ soft}(s,a)=r+\gamma\,\mathbb{E}_{s'}\!\big[V^{\ soft}(s')\big]
\]
\[
V^{\ soft}(s)=\mathbb{E}_{a\sim\pi}\big[Q(s,a)-\alpha\log\pi(a|s)\big]
\]
Soft policy improvement → \(\pi\propto \exp(Q/\alpha)\). Temperature \(\alpha\) trades return vs exploration/entropy.

**QU-SAC (Luis 2024) modification**
Replace / augment the soft objective with an uncertainty term:
\[
\pi \leftarrow \arg\max_\pi \;\bar Q + \lambda \cdot \mathrm{sign}\cdot \sqrt{\hat U}
\]
- \(\lambda>0\), risk-seeking (online exploration): add \(\sqrt{\hat U}\)
- \(\lambda<0\), risk-averse (offline): subtract \(\sqrt{\hat U}\) (pessimism)

---

### 2.6 Model-based RL & compounding error (MBPO) — [DERIVE] ⭐

**Simple**
A 1% one-step model error, chained for 100 steps, can completely destroy the trajectory distribution. Long imagined rollouts are dangerous.

**MBPO core idea (Janner et al. 2019)**
1. Learn an *ensemble* of probabilistic dynamics models.
2. Branch **short** \(k\)-step rollouts from real replay-buffer states (not from \(s_0\) for full horizon).
3. Train a model-free optimizer (SAC) on the mixed real + synthetic data.
4. Optionally schedule \(k\) upward as the model improves.

**Monotonic improvement bound (schematic)**
True return vs model return gap scales roughly like:
\[
|J(\pi)-J_{\hat P}(\pi)| \;\lesssim\;
\frac{\gamma}{(1-\gamma)^2}\cdot \epsilon_{\mathrm{model}}
+ \text{(TV / distribution-shift terms)}
\]
So error grows with effective horizon \(1/(1-\gamma)\) — often **quadratically** in the bound. Short \(k\) keeps the coefficient small.

**Production pattern**
```python
# Pseudocode: MBPO-style branched rollout
for _ in range(n_rollouts):
    s = sample_state_from_replay(D_real)
    model = random.choice(ensemble)          # bootstrap epistemic
    for t in range(k):                       # short horizon!
        a = policy.sample(s)
        s, r = model.sample(s, a)            # probabilistic next state
        D_model.add(s, a, r, s_next)
# then SAC updates on D_model (and maybe D_real)
```

**Libraries**
- `mbrl-lib` (Facebook/Meta) — what Luis et al. build on
- `mbpo` original, `PETS` (Chua et al.)

---

### 2.7 Regret / sample complexity — [USE]

**Bandit UCB (sketch)**
Pull arm maximizing \(\hat\mu_i + \sqrt{2\log t / n_i}\). Regret \(\tilde O(\sqrt{K T \log T})\).

**Tabular RL**
Optimism in the Face of Uncertainty (OFU): act optimally in an optimistic MDP inside a confidence set. UBE-style methods are a practical continuous-control cousin of OFU: use \(\bar Q + \lambda\sqrt{U}\) instead of solving a full optimistic MDP.

Don't chase full gap-dependent regret theory unless your theorem needs it.

---

### 2.8 Offline RL: distributional shift & pessimism — [DERIVE]

**Problem**
Policy can query \((s,a)\) never seen in the offline dataset. Model/Q overestimates OOD actions → catastrophic exploitation of model error.

**Pessimism principle**
Optimize a *lower* confidence bound on value:
\[
\pi = \arg\max_\pi \;\mathbb{E}_{s\sim d^\pi}\big[Q(s,a)-\beta\cdot u(s,a)\big]
\]
or regularize Q downward on OOD actions (CQL).

**Landscape (skim, don't derive all)**

| Method | Idea | Uncertainty? |
|---|---|---|
| **CQL** (Kumar 2020) | Push Q down on OOD actions via logsumexp regularizer | No explicit UQ |
| **COMBO** (Yu 2021) | CQL-style conservatism on model rollouts + real data | Uncertainty-free |
| **MOPO** (Yu 2020) | Penalize reward by ensemble aleatoric std | Yes (dynamics) |
| **MOReL** | Unknown-state detector via ensemble disagreement | Yes |
| **CBOP** (Jeong 2023) | Bootstrap Q uncertainty for pessimism | Yes (value-level) |
| **RAMBO** (Rigter 2022) | Adversarially robust model, no explicit UQ | No |
| **QU-SAC** (Luis 2024) | UBE-estimated epistemic var as risk-averse term | Yes (UBE) |

**Your Phase-6 connection**
Any novel UBE for diffusion models should plug into this pessimism slot: \(\hat U_{\mathrm{diff}}\) becomes the penalty.

---

## TRACK 3 — Generative Models / Diffusion

### 3.1 DDPM — [DERIVE]

**Simple**
- **Forward:** gradually add Gaussian noise for \(T\) steps until data looks like pure noise.
- **Reverse:** train a network to denoise step by step.
- **Trick:** predicting the noise \(\epsilon\) is equivalent (and stabler) than predicting the clean \(x_0\) or the mean of the reverse Gaussian.

**Forward process**
\[
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
\]
Closed form:
\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)
\]

**ELBO → training objective (derive)**
Variational lower bound on \(\log p_\theta(x_0)\) decomposes into KL terms between true posterior \(q(x_{t-1}|x_t,x_0)\) and model reverse \(p_\theta(x_{t-1}|x_t)\). With fixed reverse variance, minimizing those KLs ≡ minimizing
\[
\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{t,x_0,\epsilon}\big\|\epsilon-\epsilon_\theta(x_t,t)\big\|^2
\]
(Ho et al. 2020). **You should derive this once fully** — it is a non-negotiable.

**Practical training loop**
```python
x0 = batch
t = torch.randint(0, T, (B,))
eps = torch.randn_like(x0)
xt = sqrt_alpha_bar[t]*x0 + sqrt_one_minus[t]*eps
loss = F.mse_loss(eps_theta(xt, t), eps)
```

---

### 3.2 DDIM — [USE]

Deterministic, non-Markovian sampling that reuses the same trained \(\epsilon_\theta\) but allows fewer steps. Useful for faster world-model rollouts. Derive the non-Markovian reformulation if time allows.

---

### 3.3 Score-based models / SDE view — [DERIVE] ⭐

**Score** \(s(x)=\nabla_x\log p(x)\). Denoising score matching trains \(s_\theta(x_t,t)\approx\nabla\log p_t(x_t)\).

**Connection to DDPM**
Predicting noise is predicting score (up to scaling):
\[
\nabla_{x_t}\log p_t(x_t) \approx -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
\]

**Why this matters for your theory**
You can define the instantaneous “diffusion-injected” uncertainty via the SDE noise coefficient \(g(t)\) and the residual score error. That object is the continuous-time analogue of “aleatoric uncertainty of the generative transition model.”

**Resources**
- Song et al., ICLR 2021 (SDE paper) — arxiv 2011.13456
- Song blog: https://yang-song.net/blog/2021/score/
- Code: `yang-song/score_sde_pytorch`

---

### 3.4 Consistency models — [DERIVE]

**Simple**
Instead of many denoising steps, train a model \(f_\theta(x_t,t)\) that maps *any* noisy point on a trajectory to the same clean origin. Then one (or few) evaluation(s) generate a sample.

**Self-consistency**
\[
f_\theta(x_t,t)=f_\theta(x_{t'},t')\quad\text{for points on the same PF-ODE trajectory}
\]
plus boundary \(f_\theta(x_\epsilon,\epsilon)=x_\epsilon\) (almost clean). Distillation or consistency training enforces this.

**Why it matters for world models**
Diffusion world models are slow (many NFE per transition). Consistency / distilled students make imagination rollouts cheap enough for MBRL. **But distillation introduces error** — a new uncertainty source your theory can bound.

---

### 3.5 Mode-covering vs mode-seeking — [DERIVE]

| Divergence | Behavior | World-model implication |
|---|---|---|
| Forward KL \(KL(p\|q)\) | mode-covering | model spreads mass over all data modes; safer but blurrier |
| Reverse KL \(KL(q\|p)\) | mode-seeking | model collapses to one mode; sharp but misses alternatives |
| MLE with Gaussian MLP | often averages modes → blur | bad multi-modal dynamics |
| Diffusion / score matching | closer to mode-covering | captures multi-modal next-state distributions |

**Why diffusion beats Gaussian MLPs for world models (theory-level reason)**
Real environment transitions are often multi-modal (contact, branching futures). A unimodal Gaussian mean-averages modes → systematic model bias that compounds (Track 2). Diffusion can represent multi-modal \(p(s'|s,a)\).

**DIAMOND (Alonso et al., NeurIPS 2024)**
First strong demonstration that diffusion world models work for online RL in imagination (Atari 100k SOTA among pure world-model agents). Visual detail preservation is the empirical selling point; multi-modality is the theoretical one.

---

### 3.6 Diffusion world models — practical stack

**Patterns in the wild**
- **DIAMOND**: EDM-style diffusion over image observations; policy trained in imagination.
- **PolyGRAD / diffusion dynamics**: diffusion as \(p(s'|s,a)\) in continuous control (state-space, not pixels).
- **Hybrid**: VAE/tokenizer latent + diffusion in latent space (faster).

**Production constraints**
| Issue | Mitigation |
|---|---|
| Slow sampling (50–1000 NFE) | DDIM, consistency distillation, few-step samplers |
| Training cost | Latent diffusion, smaller U-Nets, progressive distillation |
| Rollout compounding | Short k like MBPO; ensemble of diffusion models (expensive!) |
| Calibration of uncertainty | Often *not* done — open research |
| Serving latency | Batch rollouts on GPU; compile with `torch.compile` / TensorRT |

**Sketch: one-step dynamics via diffusion**
```python
def sample_next_state(s, a, net, steps=10):
    # condition diffusion on (s,a); sample s'
    x = torch.randn_like(s)
    for t in reversed(range(steps)):
        x = denoise_step(net, x, t, cond=(s, a))
    return x
```

---

## TRACK 4 — Uncertainty Quantification Theory ⭐ GAP HOME

### 4.1 Aleatoric vs epistemic — formal — [DERIVE]

Already covered via law of total variance. Additional precision:

- **Distributional RL** (Bellemare et al.): models *aleatoric* return distribution for a fixed MDP/policy.
- **Bayesian RL / UBE**: models *epistemic* distribution over values induced by posterior over MDPs.
- They are complementary; Eriksson, Moskovitz et al. combine them for risk measures.

---

### 4.2 Deep ensembles as approximate Bayesian posteriors — [USE] + [DERIVE] BMA link

**BMA**
\[
p(y|x,D)=\int p(y|x,\theta)\,p(\theta|D)\,d\theta
\approx \frac1N\sum_{i=1}^N p(y|x,\theta_i)
\]
Deep ensembles: \(\theta_i\) from independent trainings (different init + data order ≈ multi-basin exploration).

**What ensemble variance captures**
- Disagreement of means → proxy for epistemic
- Average of predictive variances → proxy for aleatoric (if each member is probabilistic)

**Caveats for papers**
- Ensemble size \(N=5\)–\(10\) common; not a full posterior
- Can be overconfident under distribution shift
- Contested theoretical status — cite Wilson & Izmailov; Rahaman & Thiery

---

### 4.3 Calibration — [DERIVE]

**Definition**
A predictive CI is calibrated if
\[
P\big(Y \in CI_p(X)\big)=p \quad\forall p\in(0,1)
\]
(or for classification: among all predictions with confidence \(p\), fraction correct is \(p\)).

**ECE (Expected Calibration Error)**
Bin predictions by confidence; average \(|\mathrm{acc}(b)-\mathrm{conf}(b)|\). Approximation error depends on binning.

**Why it matters for your project**
A theoretical UBE bound that is valid but *wildly miscalibrated* in practice is less useful for risk-aware control. Phase-3 empirical calibration is how you validate a bound's tightness.

---

### 4.4 Uncertainty Bellman Equation — full story — [EXTEND]

#### Lineage (detail)

**O'Donoghue et al. 2018** — “The Uncertainty Bellman Equation and Exploration”
- Define a UBE whose fixed point **upper-bounds** posterior variance of Q-values.
- Use for directed exploration (Atari with DQN).
- Bound can be loose (Markou & Rasmussen 2019).

**Zhou et al. 2020**
- Tighter upper bound via a different local uncertainty
\[
w_t(s)=\mathrm{Var}_{p\sim\Phi_t}\!\Big(\sum_{a,s'}\pi(a|s)p(s'|s,a)\bar V^\pi(s')\Big)
\]
- UBE:
\[
W^\pi(s)=\gamma^2 w_t(s)+\gamma^2\sum_{a,s'}\pi(a|s)\bar p_t(s'|s,a)W^\pi(s')
\]
- Guarantees \(W^\pi \ge \mathrm{Var}_p(V^{\pi,p})\) under Assumptions 1–2 (independent transitions across states; acyclic / episodic structure).
- **Looseness source:** \(w_t\) does **not** subtract aleatoric uncertainty about next-state values.

**Luis et al. 2023 (AISTATS)** — “Model-Based Uncertainty in Value Functions”
Exact local uncertainty:
\[
u_t(s)=\mathrm{Var}_{a,s'\sim\pi,\bar p_t}\!\big(\bar V^\pi(s')\big)
-\mathbb{E}_{p\sim\Phi_t}\!\Big[\mathrm{Var}_{a,s'\sim\pi,p}\big(V^{\pi,p}(s')\big)\Big]
\]
UBE:
\[
U^\pi(s)=\gamma^2 u_t(s)+\gamma^2\sum_{a,s'}\pi(a|s)\bar p_t(s'|s,a)\,U^\pi(s')
\]
**Theorem:** under same assumptions, \(U^\pi(s)=\mathrm{Var}_{p\sim\Phi_t}(V^{\pi,p}(s))\) exactly.

**Gap characterization (Zhou − Luis)**
\[
u_t(s)=w_t(s)-g_t(s),\quad g_t(s)\ge 0
\]
\(g_t\) = expected aleatoric variance of next values. Zhou overestimates by ignoring \(g_t\). Gap vanishes if MDP+policy are deterministic or if epistemic uncertainty → 0.

**Important practical notes from Luis**
- \(u_t(s)\) can be **negative** (subtract too much aleatoric). They clip: \(\hat u=\max(u_{\min},u_t)\) with \(u_{\min}\le 0\).
- Assumptions fail under NN function approximation and cyclic MDPs — they still get empirical gains.
- Implementation: ensemble of probabilistic NNs + U-network trained with uncertainty Bellman residual (like a critic for variance).
- Code: https://github.com/boschresearch/ube-mbrl

**Luis et al. 2024** — QU-SAC extension
- Risk-seeking (online) and risk-averse (offline) with one hyperparameter sign flip.
- First UBE-based algorithm used for offline RL (claimed).
- Compares against MOPO, MOReL, CBOP, COMBO-style baselines.

#### Assumptions log (copy this into your paper's limitations later)
1. Independent transitions across distinct states (Assump. 1)
2. Acyclic / episodic structure (Assump. 2) — or time-augmented state
3. Posterior represented as finite ensemble
4. Known or separately learned rewards
5. Uncorrelated value functions and transitions (used in Zhou lineage)
6. Function approximation error ignored in theory

---

### 4.5 Risk-aware / pessimistic offline landscape — [USE]

Already tabulated in §2.8. Positioning rule for papers:
> “Prior UBE work assumes ensemble/Gaussian epistemic sources. We extend the exact UBE of Luis et al. to diffusion-parameterized transition models / distilled consistency dynamics, and use the resulting \(\hat U\) for risk-aware SAC.”

---

# PART II — PRACTICAL & PRODUCTION CHEATSHEET

## Reference implementations to study

| Component | Where to look |
|---|---|
| MBPO + probabilistic ensembles | `facebookresearch/mbrl-lib`, original MBPO |
| UBE + QU-SAC | `boschresearch/ube-mbrl` |
| DDPM / score SDE | `hojonathanho/diffusion`, `yang-song/score_sde_pytorch` |
| Consistency models | OpenAI consistency models repo |
| DIAMOND world model | `eloialonso/diamond` |
| Offline RL baselines | `Farama-Foundation/d3rlpy`, `rail-berkeley/rlkit` |
| SAC | stable-baselines3, cleanrl |

## Minimal “research stack” architecture

```
Real env ──► Replay D_real
                │
                ▼
     Dynamics model(s):
       - Ensemble of Gaussians  (baseline Luis)
       - Diffusion / consistency (your extension)
                │
                ▼
     Short k-step imagination ──► D_model
                │
                ▼
     Critics: Q_ensemble + U-network (UBE residual)
                │
                ▼
     Actor: SAC / QU-SAC  ( ± λ √U )
```

## Production-ish checklist (if this ever ships)
1. **Latency budget** for one imagination step (ms) → forces consistency distillation or latent diffusion.
2. **Calibration monitoring**: online ECE / reliability diagrams on held-out transitions.
3. **Safe defaults**: risk-averse \(\lambda<0\) for deployment; risk-seeking only in sim.
4. **Ensemble cost**: if diffusion ensemble is too heavy, use MC-dropout *inside* the score network or multi-checkpoint SWA as cheaper epistemic proxy — but state this as approximation in the theory.
5. **Logging**: track \(\mathbb{E}[u_t]\), fraction of clipped negative \(u_t\), rollout length \(k\), model one-step MSE, policy return — these diagnose bound tightness.

## Toy derivation sandbox (do this before general theory)

```text
2-state or 3-state chain MDP
  - Posterior: 2 discrete transition matrices with probs (0.5, 0.5)
  - Or continuous: beta/Dirichlet posterior
  - Compute exact Var_p(V^π) by enumeration
  - Compute Zhou W and Luis U by solving small linear systems
  - Verify U == exact variance, W >= U
  - Replace Gaussian transition with a 1D mixture / tiny diffusion
  - See which terms break and what new residual appears
```
If the toy case won't close cleanly in a week, the general paper will struggle for months.

---

# PART III — GAP-FINDING PROCESS (FROM THE ROADMAP) + LIVE APPLICATION

## The method (repeatable)

### Step 1 — Find the “delta” in 2–3 related papers
For each theoretical claim, write:
1. What assumption did the *previous* paper make that caused looseness/error?
2. What did the *new* paper change?
3. What assumption does the *new* paper still make that looks suspicious?

### Step 2 — Check the gap is real (not already solved)
- arXiv/Scholar keyword combos every 3–4 weeks
- Workshop papers (ICLR/NeurIPS workshops lead main track)
- If something is close: differentiate on assumption, tightness, distillation, empirics

### Step 3 — Tractability toy test
Hand-derive 2–3 step MDP + 1D generative transition before general case.

### Step 4 — Iterate
State conjecture → try to break at edge cases → compare to empirical calibration → tighten / relax / downgrade claim.

### Step 5 — Position as a chain
Related work = Zhou → Luis fix → remaining assumption → your fix.

---

## Live application (researched, mid-2026)

### Delta chain (filled in)

| Paper | Assumption / limitation | What next paper fixed |
|---|---|---|
| O'Donoghue 2018 | UBE gives **upper bound**, can be loose | — |
| Zhou 2020 | Local uncertainty ignores **aleatoric** next-value noise | Tighter than O'Donoghue but still ≥ true var |
| Luis 2023 | Subtracts aleatoric → **exact** posterior variance under Assump. 1–2 | Characterizes Zhou gap \(g_t\) |
| Luis 2024 (QU-SAC) | Same UBE; neural approx issues; **ensemble/Gaussian dynamics** as \(\Phi_t\) | Practical online+offline algorithm |
| **Open (revised)** | Theorem 1 already holds for general \(\Phi_t\); what is missing is a **tractable, bias-characterized MC estimator** of \(u_t/w_t\) when ensemble members are **diffusion/consistency** (implicit samplers), not Gaussian MLPs | **Your territory** — see `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` |

### Revised precision (2026-07-18): theorem vs estimator

**Theorem 1 is distribution-agnostic.** Appendix A.1 Lemmas 1–3 use only Assumptions 1 (independent transitions) and 2 (acyclic MDP). The recursion \(U_t = \mathrm{UBE}(u_t)\) holds for any posterior over MDPs satisfying those assumptions — **not** only Gaussians.

**Gaussians enter only in Section 4 / Appendix D.1 (implementation):**
- \(\Gamma_t\) chosen as uniform over \(N\) probabilistic NNs that output Gaussian mean/covariance.
- For \(w_t(s,a)\): take \(s' =\) **mean of each member Gaussian**, one action from \(\pi\).
- For the gap term in \(u_t\): finite-sample (~10 actions); Gaussian structure makes \(\mathrm{Var}_{s'}\) cheap.

So the real gap is **not** “extend Theorem 1 to diffusion.” It is:

> Does Monte Carlo estimation of \(u_t(s,a)/w_t(s,a)\) from diffusion-ensemble samples preserve the theorem’s guarantees, and how many samples (and what distillation bias) are needed?

### What is still hard for diffusion world models (estimator layer)

1. **Implicit kernels.** \(p_i(s'|s,a)\) has no closed-form mean/variance; only samples from a reverse process (or consistency map). Nested MC replaces Luis’s one-mean trick.

2. **Cost.** \(O(N\cdot M\cdot K)\) denoising steps per local-reward estimate vs \(O(N)\) Gaussian forwards.

3. **Generative-model-specific error (Phase B):**
   - Truncation / few-step sampler error (DDIM \(K\ll T\))
   - Distillation bias (consistency student ≠ teacher)
   - Score residual \(\|s_\theta - \nabla\log p_t\|\)
   - Mode-coverage failure under limited offline data

4. **Function approximation gap**: theory exact for tabular/acyclic; neural U-net is heuristic (same as Luis).

5. **Independence / acyclicity** — inherited; same limitations.

### Is the gap already solved? (literature check summary)

| Query theme | Finding |
|---|---|
| “Uncertainty Bellman” + diffusion | No established exact-UBE extension to diffusion world models found in mainstream UBE lineage (O'Donoghue–Zhou–Luis). Adjacent work exists on uncertainty in generative models and on diffusion world models (DIAMOND etc.) **separately**. |
| Diffusion world models + epistemic UQ for RL | Empirical / heuristic (ensembles of diffusions, MC samples) — not a Luis-style exact recursion. |
| Distillation + epistemic uncertainty bounds | Consistency/distillation literature studies sample quality & NFE, not Bellman-propagated value variance for RL. |
| QU-SAC / UBE offline | Exists for ensemble dynamics; not for diffusion dynamics. |

**Conclusion:** The gap flagged in the roadmap remains **legitimate and paper-worthy** as of this research pass. Re-check arXiv every few weeks.

---

## Legitimate novel directions (ranked, revised 2026-07-18)

> **North-star writeup:** `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md`  
> **Toy estimator code:** `theory/toy_mc_ube_estimator.py`  
> **Assumptions log:** `research/ASSUMPTIONS-LOG.md`

### Direction A (revised) — **Primary: Sample-based UBE estimation for diffusion/consistency ensembles**
**Title sketch:** *Monte Carlo Uncertainty Bellman Rewards for Diffusion World Models*

**Claim form (honest, smaller, stronger)**
Theorem 1 already applies to general posteriors over transition kernels. We give estimators \(\hat u_t,\hat w_t\) when each ensemble member is an **implicit** diffusion/consistency sampler, and prove (or bound) that Monte Carlo estimates concentrate about the population local rewards — so the UBE fixed point remains \(\varepsilon\)-close to true posterior value variance (via \(\gamma^2\)-contraction). Compare to closed-form Gaussian estimators of Luis Appendix D.1.

**Why novel**
- Not a new recursion — a **validated estimator** for an existing theorem under the dynamics class that DIAMOND made practical.
- Clean reviewer story: “we prove the estimator, not a new UBE.”
- Directly plug into existing U-net / QU-SAC machinery.

**Theory plan**
1. Reproduce Luis Appendix A.1 by hand (checklist in `research/proofs/luis-theorem1-checklist.md`).
2. Write explicit MC-UBE-Local algorithm (research note §5.3).
3. **Theorem A1:** Hoeffding/McDiarmid sample complexity for \(\hat w,\hat u\) under exact sampling + bounded \(\bar Q\).
4. **Corollary A2:** UBE fixed-point perturbation \(\lVert\hat U-U^\star\rVert_\infty \le \frac{\gamma^2}{1-\gamma^2}\varepsilon\).
5. Toy numerical: error vs \(M\) slope \(\sim 1/\sqrt{M}\) (`theory/toy_mc_ube_estimator.py`).

**Empirics**
- Toy 1D / small acyclic MRP with enumerable ensemble ground truth
- Continuous control: diffusion dynamics + U-net vs Gaussian PETS+UBE
- Calibration: predicted \(\sqrt{U}\) vs residual error

**Risks**
- Nested MC cost — CRN, few-step samplers, cache local rewards
- Deep RL still violates Assumps. 1–2 (same as Luis; clip \(u_{\min}\))

---

### Direction B — **Distillation-aware extension (Phase 2 on A)**
**Title sketch:** *Bounding Local UBE Error under Consistency Distillation of World Models*

**Claim form**
Once (A)’s estimator exists, characterize **additional** bias/variance when samples come from a distilled \(K\in\{1,2,4\}\)-step sampler instead of the full reverse process. Prefer bounds on \(\lvert w^{(K)}-w^{(\infty)}\rvert\) and honesty that \(\hat U\) is calibrated to the **imagination sampler actually used**.

**Why novel**
- Fast rollouts are why consistency models exist; UBE line ignores distillation as structured error.
- Sequential on (A), not competing.

**Tractability**
TV/Wasserstein sampler error + Lipschitz \(\bar Q\) + contraction — same toolkit as (A).

---

### Direction C — **Offline pessimism with this estimator (later phase)**
**Title sketch:** *Pessimistic Offline RL via Sample-Based UBE under Generative Dynamics*

**Claim form**
Plug validated \(\hat U_{\mathrm{diff}}\) into \(\bar Q - \lambda\sqrt{U}\) (MOPO/QU-SAC-style). Systems + theory + production story once (A) is solid.

---

### Direction D — **Empirical-only calibration (fallback)**
Calibrated aleatoric/epistemic decomposition for diffusion WMs without a concentration theorem — workshop-viable if (A) does not close cleanly.

---

## What is **not** a good gap (avoid)

1. “Diffusion world models work better empirically” — already DIAMOND et al.
2. Re-deriving Luis exact UBE without a new assumption change — incremental at best.
3. Pure regret bounds for tabular UBE — deep rabbit hole, crowded, not your comparative advantage if your stack is deep MBRL + diffusion.
4. Full measure-theoretic SDE theory without an RL payoff — wrong venue fit.

---

## Recommended paper narrative (Step 5 positioning, revised)

```text
Related work / intro arc:

1. Epistemic value uncertainty matters for exploration & offline pessimism.
2. O'Donoghue introduced UBE upper bounds; Zhou tightened local uncertainty.
3. Luis et al. gave an exact UBE for posterior Var(V) under Assumps. 1–2 —
   the theorem is distribution-agnostic; Gaussian ensembles appear only in
   their deep RL estimator (closed-form means / easy variances).
4. Parallel development: diffusion world models (DIAMOND, …) define p(s'|s,a)
   only implicitly via multi-step (or distilled) generative sampling.
5. Gap: not a missing recursion — a missing, finite-sample-validated estimator
   of local rewards u_t / w_t for generative dynamics (+ distillation bias).
6. This paper: MC-UBE-Local + concentration (A), distillation error (B),
   and QU-SAC-style use of Û_diff (C).
```

---

## Assumptions log template (keep running while you derive)

| # | Assumption | Where used | Can we relax? | Failure mode if false |
|---|---|---|---|---|
| A1 | Independent transitions across states | Luis Thm 1 | Hard | Covariance terms appear |
| A2 | Acyclic / finite horizon effective | Contraction / uncorr. | Time-augment state | Cycles → correlations |
| A3 | Posterior = finite ensemble of models | Practical \(\Phi_t\) | Laplace / VI | Underestimated epistemic |
| A4 | Bounded rewards | Variance finite | Soft | Heavy tails break bounds |
| A5 | Diffusion reverse process well-specified | New | Score error term | Biased \(u_t\) |
| A6 | Distillation error ≤ ε in TV or W₂ | Dir. B | Measure carefully | Invalid upper bound |
| A7 | Function approx. error = 0 | Theory only | Bellman residual bounds | U-net ≠ true U |

---

## 12-week theory attack plan (aligned with roadmap)

| Weeks | Focus |
|---|---|
| 1–2 | Derive law of total variance, Bayes, Hoeffding; re-derive Bellman contraction |
| 3–4 | Reproduce O'Donoghue + Zhou + Luis UBE **on paper**, line by line |
| 5 | Toy 2–3 step MDP with Gaussian ensemble: exact U = Luis U |
| 6 | Replace transitions with 1D diffusion; identify broken terms |
| 7–8 | Conjecture Diffusion-UBE or Distillation-UBE; try prove bound |
| 9 | Edge cases: deterministic env, ε→0 distillation, infinite ensemble |
| 10 | Match against empirical calibration from a tiny implementation |
| 11 | Related-work narrative + assumptions log → paper outline |
| 12 | Decide: equality theorem vs upper bound vs empirical conjecture; freeze scope |

Parallel: re-search arXiv weekly for “uncertainty Bellman”, “diffusion world model”, “epistemic uncertainty distillation RL”.

---

# PART IV — CURATED RESOURCE LIST

## Track 1
- Blitzstein & Hwang, *Introduction to Probability* (total variance, conditioning)
- Wainwright, *High-Dimensional Statistics* Ch. 2 (concentration) — optional depth
- Oksendal, *Stochastic Differential Equations* Ch. 1–5 (Itô basics) — skim
- Song score-SDE blog (intuition for reverse SDE)

## Track 2
- Sutton & Barto, *RL: An Introduction* (MDP, Bellman) — free online
- Agarwal, Jiang, Kakade, Sun — *Reinforcement Learning: Theory and Algorithms* (draft)
- Janner et al. 2019, MBPO — arxiv 1906.08253 (+ BAIR blog)
- Haarnoja et al. 2018, SAC
- Kumar et al. 2020, CQL; Yu et al. 2021, COMBO; Yu et al. 2020, MOPO
- Levine et al. offline RL tutorial

## Track 3
- Ho et al. 2020, DDPM — arxiv 2006.11239
- Song et al. 2021, Score SDE — arxiv 2011.13456
- Song et al. consistency models — arxiv 2303.01469
- Alonso et al. 2024, DIAMOND — arxiv 2405.12399 (+ github.com/eloialonso/diamond)
- Lilian Weng blogs: diffusion, scoring models (excellent pedagogy)

## Track 4
- O'Donoghue et al. 2018, UBE — arxiv 1709.05380 / ICML 2018
- Zhou et al. 2020 (model-based UBE + PPO)
- Luis et al. 2023, Model-Based Uncertainty in Value Functions — arxiv 2302.12526 / AISTATS 2023
- Luis et al. 2024, Model-Based Epistemic Variance… QU-SAC — arxiv 2312.04386
- Lakshminarayanan et al. 2017, Deep Ensembles
- Chua et al. 2018, PETS
- Code: github.com/boschresearch/ube-mbrl

## Gap monitoring queries (save these)
```
"uncertainty Bellman equation" diffusion
"uncertainty Bellman" "world model"
UBE epistemic diffusion OR consistency
"value variance" diffusion dynamics RL
QU-SAC OR "exact-ube" distillation
```

---

# PART V — EXECUTIVE SUMMARY

### What you must master
1. Law of total variance as the epistemic/aleatoric split  
2. Bellman operator γ-contraction (template for UBE)  
3. MBPO compounding-error bound (why short rollouts)  
4. DDPM ELBO → ε-prediction and score-SDE reverse process  
5. **Full Luis 2023 UBE derivation + Zhou gap \(g_t\)**  
6. Offline pessimism principle  

### Where novelty lives
**Luis et al. made UBE exact for ensemble/Gaussian MDP posteriors by removing aleatoric pollution from Zhou's local uncertainty. They still assume that style of posterior/dynamics.**  
Modern world models are **diffusion / consistency** generative processes. Extending the exact (or tightly bounded) UBE to those generative dynamics — including sampler noise, score error, and distillation error — is a **legitimate, still-open, paper-scale gap**.

### Best first paper bet
**Direction A (Diffusion-UBE)** with **Direction B (distillation term)** as a natural theorem corollary or second paper. Algorithmic vehicle: QU-SAC-style actor with diffusion world model, evaluated on exploration + D4RL-style offline.

### Process discipline
- Derive on paper 2–3×/week  
- Toy MDP before general proof  
- Assumptions log always on  
- arXiv sweep every 3–4 weeks  
- Related work written as the delta chain  

---

*Document generated from the learning roadmap + primary sources (O'Donoghue 2018, Zhou 2020, Luis 2023/2024, MBPO, SAC, DDPM/Score-SDE, DIAMOND, COMBO/MOPO/CQL landscape). Re-validate novelty claims with a fresh arXiv search before submitting.*
