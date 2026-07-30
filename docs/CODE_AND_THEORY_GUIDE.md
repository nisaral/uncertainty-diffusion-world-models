# Complete Understanding Guide: Theory, Math, Code, Literature & Differentiation

**Project:** Uncertainty-Aware Diffusion World Models (UDWM)  
**Repo:** https://github.com/nisaral/uncertainty-diffusion-world-models  
**How to use this doc:** Read top-to-bottom once, then use the table of contents as a reference.  
Every major theory idea is written twice:

- **Simple** — intuition, analogies, no prerequisites  
- **Technical** — equations, assumptions, what the paper would say  

---

# Table of contents

1. [Big picture](#1-big-picture)  
2. [What problem we solve](#2-what-problem-we-solve)  
3. [Math foundations](#3-math-foundations)  
4. [MDPs, Bellman equations, value functions](#4-mdps-bellman-equations-value-functions)  
5. [Aleatoric vs epistemic uncertainty](#5-aleatoric-vs-epistemic-uncertainty)  
6. [Model-based RL & imagination (MBPO)](#6-model-based-rl--imagination-mbpo)  
7. [The UBE story (O’Donoghue → Zhou → Luis)](#7-the-ube-story)  
8. [Luis exact UBE — full math](#8-luis-exact-ube--full-math)  
9. [Why Gaussians are only an implementation trick](#9-why-gaussians-are-only-an-implementation-trick)  
10. [Diffusion models for dynamics](#10-diffusion-models-for-dynamics)  
11. [Our contribution: MC-UBE](#11-our-contribution-mc-ube)  
12. [Concentration & fixed-point guarantees](#12-concentration--fixed-point-guarantees)  
13. [Distillation / consistency (speed)](#13-distillation--consistency-speed)  
14. [Joint reward diffusion (Gap 1)](#14-joint-reward-diffusion-gap-1)  
15. [Policy: SAC + optimism/pessimism](#15-policy-sac--optimismpessimism)  
16. [Extensive literature survey](#16-extensive-literature-survey)  
17. [What we do differently (comparison matrix)](#17-what-we-do-differently)  
18. [Codebase architecture](#18-codebase-architecture)  
19. [File-by-file code guide](#19-file-by-file-code-guide)  
20. [Math ↔ code map](#20-math--code-map)  
21. [Training loop step-by-step](#21-training-loop-step-by-step)  
22. [Toy experiments (theory validation)](#22-toy-experiments-theory-validation)  
23. [How to run everything](#23-how-to-run-everything)  
24. [Three research gaps on one stack](#24-three-research-gaps-on-one-stack)  
25. [Limitations & honest caveats](#25-limitations--honest-caveats)  
26. [Glossary](#26-glossary)  
27. [Study plan](#27-study-plan)  

---

# 1. Big picture

### Simple
Imagine teaching a robot by letting it practice in a **video-game version of the world** that it learned from a little real experience. That game is the **world model**.  

Problems:
1. The game can be **wrong** in weird ways (especially if the real world has multiple possible next outcomes).  
2. If the robot **trusts the game too much**, it learns a policy that looks great in the game and fails in reality.  

We use:
- **Diffusion models** as a powerful game engine (good at multiple possible futures),  
- An **Uncertainty Bellman Equation** so the robot tracks *how unsure it is about long-term value*, not just “next frame noise,”  
- **Fast sampling** (few steps / one-step student) so practice is cheap,  
- A policy that explores or stays safe using \(\sqrt{U}\).

### Technical
We study **model-based RL** where the transition model \(p(s'\mid s,a)\) is an **ensemble of implicit generative models** (diffusion / consistency). We estimate multi-step **epistemic** uncertainty  
\[
U^\pi(s)=\mathrm{Var}_{p\sim\Phi}\big[V^{\pi,p}(s)\big]
\]  
via the **exact UBE** of Luis et al. (2023), using **Monte Carlo local rewards** when closed-form moments do not exist. The policy is SAC with optional OFU/pessimism \(Q\pm\lambda\sqrt{U}\).

**One-sentence research claim:**  
*Theorem 1 of Luis et al. already applies beyond Gaussians; we provide sample-based estimators of the UBE local rewards for diffusion ensembles, with systems support for continuous-control imagination, distillation, and joint reward modeling.*

---

# 2. What problem we solve

### Simple
**Industry problem:**  
“World models are hot (DIAMOND, Dreamer, WIMLE). Uncertainty is hot (ensembles, MOPO, UBE). Nobody cleanly connects **diffusion dynamics** with **multi-step value uncertainty** that theorists already understand (UBE), in a way that is **fast enough** for online RL.”

### Technical
| Challenge | Why hard | Our approach |
|---|---|---|
| Multi-modal \(p(s'\mid s,a)\) | Gaussians average modes | Diffusion ensemble |
| Multi-step epistemic \(U\) | One-step var does not compound correctly | UBE recursion + U-net |
| Implicit kernels | No closed-form \(\mu,\Sigma\) | MC-UBE-Local |
| Online latency | Multi-step denoising | DDIM \(K\)-step + consistency student |
| Reward modeling | DIAMOND uses separate \(R_\psi\) | Optional joint \([\Delta s,r]\) diffusion |

---

# 3. Math foundations

## 3.1 Expectation and variance

### Simple
- **Average** of a random number: expectation \(\mathbb{E}[X]\).  
- **Spread** around the average: variance \(\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2]\).

### Technical
For a random variable \(X\) with finite second moment:
\[
\mathbb{E}[X]=\int x\,dP(x),\qquad
\mathrm{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2.
\]
Linearity: \(\mathbb{E}[aX+bY]=a\mathbb{E}X+b\mathbb{E}Y\).  
Scaling: \(\mathrm{Var}(aX)=a^2\mathrm{Var}(X)\) — **this is why UBE has \(\gamma^2\)**, not \(\gamma\).

## 3.2 Law of total expectation / variance

### Simple
If you first pick a “mode” \(\theta\), then sample \(Y\):
- overall average = average of the averages of each mode,  
- overall spread = average within-mode spread **plus** spread of the mode-averages.

### Technical
\[
\mathbb{E}[Y]=\mathbb{E}\big[\mathbb{E}[Y\mid\theta]\big],
\]
\[
\mathrm{Var}(Y)
=
\underbrace{\mathbb{E}\big[\mathrm{Var}(Y\mid\theta)\big]}_{\text{aleatoric (avg)}}
+
\underbrace{\mathrm{Var}\big(\mathbb{E}[Y\mid\theta]\big)}_{\text{epistemic}}.
\]
This identity is the **entire philosophical backbone** of UBE local rewards \(w\) vs \(g\).

## 3.3 Contractions and fixed points

### Simple
If a process always “shrinks disagreements” by a factor \(\alpha<1\), it converges to a unique answer — like repeatedly averaging with a fixed recipe.

### Technical
An operator \(T\) on bounded functions is an \(\alpha\)-contraction in \(\|\cdot\|_\infty\) if
\[
\|Tf-Tg\|_\infty\le \alpha\|f-g\|_\infty.
\]
Banach fixed-point theorem ⇒ unique fixed point \(f^\star=Tf^\star\), and value iteration converges.  

**Bellman operator** is a \(\gamma\)-contraction.  
**UBE operator** is a \(\gamma^2\)-contraction.  
**Corollary A2:** if local rewards differ by \(\varepsilon\) everywhere, fixed points of UBE differ by at most \(\frac{\gamma^2}{1-\gamma^2}\varepsilon\).

## 3.4 Concentration (Hoeffding)

### Simple
If you average many independent noisy measurements of a bounded quantity, the average gets close to the truth with high probability. More samples ⇒ tighter guarantee.

### Technical
If \(X_i\in[a,b]\) i.i.d., \(\bar X=\frac1M\sum_i X_i\), then
\[
\mathbb{P}\big(\lvert\bar X-\mathbb{E}X\rvert>\varepsilon\big)
\le
2\exp\!\Big(-\frac{2M\varepsilon^2}{(b-a)^2}\Big).
\]
This is the tool for **Theorem A1**: how large \(M\) (next-state samples) must be for \(\hat w\approx w^\star\).

---

# 4. MDPs, Bellman equations, value functions

## 4.1 MDP

### Simple
A game with states, actions, rewards, and random next states. You want a policy that maximizes long-term score.

### Technical
MDP \(\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)\) (sometimes \(\rho\) for start state).  
Transition kernel \(P(s'\mid s,a)\). Reward \(r(s,a)\) (or random \(R\)). Discount \(\gamma\in[0,1)\).

## 4.2 Policy and return

### Simple
A policy is the agent’s strategy. Return is the discounted sum of rewards along a trajectory.

### Technical
Policy \(\pi(a\mid s)\). Trajectory \(\tau=(s_0,a_0,r_0,s_1,\ldots)\).  
Return \(G_t=\sum_{k=0}^\infty \gamma^k r_{t+k}\).  
State value:
\[
V^{\pi,P}(s)=\mathbb{E}\big[G_0\mid s_0=s,\,\pi,P\big].
\]
Action value \(Q^{\pi,P}(s,a)\) similarly.

## 4.3 Bellman equation

### Simple
“Your value today = reward now + discounted value of tomorrow (on average).”

### Technical
**Expectation Bellman equation:**
\[
V^{\pi,P}(s)
=
\sum_a\pi(a\mid s)
\Big(
r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{\pi,P}(s')
\Big).
\]
Operator form \(V=T^{\pi,P}V\). \(T^{\pi,P}\) is a \(\gamma\)-contraction ⇒ unique fixed point.

## 4.4 Soft Bellman / SAC (preview)

### Simple
SAC also encourages **randomness** in the policy (exploration / robustness) via entropy.

### Technical
Soft value uses entropy-regularized objective; soft Bellman backup includes \(-\alpha\log\pi(a'\mid s')\). Implemented in `udwm/rl/sac.py`.

---

# 5. Aleatoric vs epistemic uncertainty

### Simple
| Type | Analogy | Example |
|---|---|---|
| **Aleatoric** | Dice are random even if you know the dice | Wind noise on a drone |
| **Epistemic** | You don’t know *which* physics is true yet | “Have I seen this terrain enough?” |

You explore to reduce **epistemic** uncertainty. You cannot remove pure aleatoric noise by collecting more data of the same kind (only by changing sensors, etc.).

### Technical
Given posterior \(\Phi\) over transition models \(p\):
\[
\underbrace{\mathrm{Var}_{p,s'}\big[V(s')\big]}_{\text{total}}
=
\underbrace{\mathbb{E}_p\big[\mathrm{Var}_{s'\sim p}V\big]}_{\text{aleatoric}}
+
\underbrace{\mathrm{Var}_p\big(\mathbb{E}_{s'\sim p}V\big)}_{\text{epistemic}}.
\]
UBE’s local \(u\) isolates a **relative epistemic local signal** so multi-step \(U\) tracks posterior variance of **values**, not raw next-state noise.

**Why this matters for RL:**  
If you explore high total variance regions, you may chase irreducible noise. If you explore high **epistemic value** uncertainty, you visit places where learning the model still changes long-term return estimates.

---

# 6. Model-based RL & imagination (MBPO)

## 6.1 Dyna / imagination idea

### Simple
Learn a simulator from data. Practice inside the simulator. Occasionally check reality. Repeat.

### Technical
Sutton’s Dyna: real transitions update model; simulated transitions update value/policy.  
Modern deep MBRL: neural \(p_\theta(s',r\mid s,a)\), imagination rollouts of length \(k\), model-free learner on synthetic data.

## 6.2 MBPO (Janner et al., 2019)

### Simple
Don’t roll out the model for huge horizons (errors explode). Use **short** imagined branches from real states, dump them in a buffer, train SAC.

### Technical
- Dynamics: ensemble of probabilistic nets.  
- Branched rollouts length \(k\) small.  
- Agent: SAC on mixture of real and model data with real ratio \(\eta\).  
- Theoretical motivation: bound policy improvement under model error (model bias compounds with horizon).

**Our code:** outer loop in `udwm/rl/trainer.py` is MBPO-shaped; dynamics can be Gaussian **or** diffusion.

## 6.3 Why model error compounds

### Simple
A 1% mistake each step becomes huge after 100 steps — like a slightly wrong map for a long road trip.

### Technical
If one-step TV (or Wasserstein) error is \(\varepsilon\), \(k\)-step trajectory error scales roughly \(O(k\varepsilon)\) (or worse under compounding). Hence MBPO’s short \(k\) and our interest in **uncertainty** along rollouts.

---

# 7. The UBE story

## 7.1 Why not just ensemble variance of Q?

### Simple
Training 5 critics and taking their disagreement is a rough uncertainty. But it doesn’t systematically say how **one-step model doubt** becomes **long-horizon value doubt**.

### Technical
Model-free ensemble variance of \(Q\) mixes many error sources and does not use the model’s \(\Phi_t\) explicitly. Model-based UBE uses the **posterior over MDPs** and a recursion aligned with Bellman structure.

## 7.2 Historical chain

| Paper | Result | Limit |
|---|---|---|
| **O’Donoghue et al. 2018** | UBE fixed point **upper-bounds** \(\mathrm{Var}(Q)\) | Can be loose |
| **Zhou et al. 2020** | Tighter local \(w_t\) | Still ignores aleatoric next-value noise |
| **Luis et al. 2023** | Exact \(U=\mathrm{Var}(V)\) under Assumps. 1–2; characterizes Zhou gap \(g_t\) | Deep RL estimator uses **Gaussian** ensembles |
| **Luis et al. 2024 QU-SAC** | Practical actor-critic with UBE | Same Gaussian dynamics assumption in practice |
| **This project** | MC local rewards for **diffusion/consistency** ensembles | Estimator theory + systems |

### Simple narrative you can tell a professor
“First people had a loose uncertainty Bellman bound. Then a tighter local term. Then Luis made it exact by subtracting aleatoric noise. They implemented it with Gaussian ensembles. We keep their exact equation and change the **estimator** so it works when the world model is a diffusion model.”

---

# 8. Luis exact UBE — full math

## 8.1 Object of study

### Technical
Posterior \(\Phi_t\) over transition functions \(p\) (rewards known in pure theory; unknown OK with extension).  
\[
\bar p_t=\mathbb{E}_{p\sim\Phi_t}[p],\qquad
\bar V_t^\pi=\mathbb{E}_{p\sim\Phi_t}[V^{\pi,p}],\qquad
U_t^\pi(s)=\mathrm{Var}_{p\sim\Phi_t}[V^{\pi,p}(s)].
\]

## 8.2 Assumptions

1. **Independent transitions (Assump. 1):** \(p(\cdot\mid x,a)\) independent of \(p(\cdot\mid y,a)\) for \(x\neq y\).  
   - Simple: learning dynamics at one state doesn’t correlate with another (tabular-ish).  
   - Technical: fails with shared neural weights — same practical caveat as Luis deep RL.

2. **Acyclic MDP (Assump. 2):** no state revisits in an episode (or time-augment states).  
   - Simple: no loops, or add “time” to the state.  
   - Enables uncorrelatedness lemmas between \(V(s')\) and \(p(\cdot\mid s,a)\).

## 8.3 Theorem 1 (exact UBE)

### Simple
Long-term value uncertainty = local uncertainty “reward” + discounted expected next uncertainty under average dynamics. Same *shape* as a value function, but for variances, with \(\gamma^2\).

### Technical
Under Assumps. 1–2, for any \(\pi\):
\[
U_t^\pi(s)
=
\gamma^2 u_t(s)
+
\gamma^2
\sum_{a,s'}
\pi(a\mid s)\,\bar p_t(s'\mid s,a)\,U_t^\pi(s'),
\]
with local uncertainty
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

**Critical:** No Gaussian assumption appears in the proof (Appendix A.1, Lemmas 1–3). Distribution-agnostic over \(\Phi_t\).

## 8.4 Theorem 2 (Zhou gap)

### Technical
\[
u_t(s)=w_t(s)-g_t(s),\qquad g_t(s)\ge 0,
\]
where
\[
w_t(s)
=
\mathrm{Var}_{p\sim\Phi_t}
\Big[
\sum_{a,s'}\pi(a\mid s)p(s'\mid s,a)\bar V_t^\pi(s')
\Big]
\]
is Zhou’s local term, and \(g_t\) is the expected extra aleatoric next-value noise.  
Hence Zhou’s UBE with reward \(w\) **overestimates** true epistemic variance.

### Simple
Zhou counted some “dice noise” as if it were “I don’t know the model.” Luis subtracts that dice noise so \(U\) means true model doubt about values.

## 8.5 Decomposition (Eq. 8 style)

\[
\underbrace{\mathrm{Var}_{a,s'\sim\pi,\bar p}\big[\bar V(s')\big]}_{\text{total under mean kernel}}
=
\underbrace{w}_{\text{epistemic}}
+
\underbrace{\mathbb{E}_p\big[\mathrm{Var}_{a,s'\sim\pi,p}\bar V\big]}_{\text{aleatoric on mean values}}.
\]

## 8.6 Q-function form (what we implement)

### Technical
Deep RL uses state-action \(U(s,a)\) and local \(u(s,a),w(s,a)\) (Luis Appendix B).  
Practical deep targets:
\[
z=\gamma^2 \hat u(s,a)+\gamma^2(1-d)\,U(s',a'),\qquad a'\sim\pi(\cdot\mid s').
\]
Clipping \(u\leftarrow\max(u_{\min},u)\) with \(u_{\min}=0\) keeps \(U\) non-negative (practical upper bound between exact \(U\) and Zhou \(W\) when theory holds).

---

# 9. Why Gaussians are only an implementation trick

### Simple
Luis’s *math* is general. Their *code* assumed each model outputs a bell-curve next state so averages and spreads are free. Diffusion doesn’t give free averages — only samples after denoising.

### Technical
Section 4 of Luis: \(\Gamma_t=\mathrm{Unif}\{p_{\theta_i}\}\) where each \(p_{\theta_i}\) is Gaussian.  
Appendix D.1:
- For \(w\): take \(s'=\mu_i(s,a)\) (Gaussian mean), one \(a'\sim\pi\).  
- For gap in \(u\): ~10 actions; Gaussian structure eases variance.  

**Cost comparison**

| | Gaussian ensemble | Diffusion ensemble |
|---|---|---|
| Get \(s'\) | \(O(1)\) forward mean | \(O(K)\) denoising steps × samples |
| \(\mathrm{Var}_{s'}[Q]\) | closed form / cheap | MC over \(M\) samples |
| Cost for local \(u\) | \(O(N)\) | \(O(N\cdot M\cdot K)\) |

This is exactly why Claim A is about **estimators**, not new recursion.

---

# 10. Diffusion models for dynamics

## 10.1 DDPM idea

### Simple
To learn a complicated distribution:  
1. Gradually add noise to real data until pure noise (forward).  
2. Train a network to remove a bit of noise (reverse).  
3. At test time, start from pure noise and repeatedly “denoise” into a sample.

### Technical
Forward (variance-preserving discrete):
\[
q(x_t\mid x_0)=\mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big).
\]
Train \(\epsilon_\theta(x_t,t)\) to predict noise; simplified loss
\[
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}\big\|\epsilon-\epsilon_\theta(x_t,t)\big\|^2.
\]
Sampling: reverse chain / DDIM acceleration.

## 10.2 Conditional dynamics

### Simple
We don’t generate random images from nothing — we generate **next state given current state and action**.

### Technical
Condition denoiser on \((s,a)\):
\[
\epsilon_\theta(x_t,t,s,a),\qquad x_0\approx \Delta s = s'-s
\]
(normalized in code for stability).  
Optional joint:
\[
x_0=\big[\Delta s_{\mathrm{norm}},\, r_{\mathrm{norm}}\big].
\]

## 10.3 Why diffusion for world models?

### Simple
Real environments often have **several plausible next outcomes** (contact, branching events). A single Gaussian averages them into a blur. Diffusion can represent multi-modal clouds of futures.

### Technical
Mode-covering generative training (likelihood / score matching) vs mode-seeking reverse-KL. Empirically DIAMOND shows visual detail matters for Atari imagination. Continuous control: same multi-modality argument (WIMLE uses IMLE for the same reason, but avoids diffusion for speed).

## 10.4 Our diffusion code shape

File: `udwm/models/diffusion_dynamics.py`

1. Normalize \(\Delta s\) (running mean/std).  
2. Sample \(t\), noise, form \(x_t\).  
3. Predict noise with MLP conditioned on \((s,a,t)\).  
4. MSE loss.  
5. Sample with DDIM-like loop for `sample_steps` steps.  
6. Ensemble: \(N\) independent denoisers.

---

# 11. Our contribution: MC-UBE

## 11.1 Population quantities

### Technical
For fixed ensemble \(\{\theta_i\}_{i=1}^N\) treated as \(\Phi=\mathrm{Unif}\):
\[
\mu_i(s,a)
=
\mathbb{E}_{s'\sim p_{\theta_i}(\cdot\mid s,a),\,a'\sim\pi(\cdot\mid s')}
\big[\bar Q(s',a')\big],
\]
\[
w(s,a)=\mathrm{Var}_{i\sim\mathrm{Unif}[N]}(\mu_i),
\]
\[
v_i=\mathrm{Var}_{s',a'}\big[\bar Q\mid\theta_i\big],\qquad
g=\mathbb{E}_i[v_i],
\]
and practical \(u=\max(u_{\min}, w-g)\) (or Luis’s exact \(u\) form when computable).

## 11.2 Monte Carlo estimator (MC-UBE-Local)

### Simple
For each of \(N\) world models:
- Imagine the next state many times (\(M\) samples),  
- Ask the critic “how good is that next state?”,  
- Average and spread within that model,  
Then see how much the \(N\) models **disagree** (epistemic) vs how noisy each is (aleatoric).

### Technical
```
for i in 1..N:
  for m in 1..M:
    s'_{i,m} ~ SampleDynamics(θ_i, s, a; K steps)
    a'_{i,m} ~ π(·|s'_{i,m})
    y_{i,m}  = Q̄(s'_{i,m}, a'_{i,m})
  μ̂_i = mean_m y
  v̂_i = var_m y
ŵ = var_i(μ̂_i)
ĝ = mean_i(v̂_i)
û = max(u_min, ŵ - ĝ)
```

### Code
`udwm/uncertainty/mc_ube.py` → `MCUBELocalRewards.estimate`  
- Gaussian branch: can use deterministic mean + optional stochastic samples.  
- Diffusion branch: `sample_next_multi`.

## 11.3 Learning multi-step \(U\)

### Simple
Local \(\hat u\) is only “this step’s uncertainty reward.” A small network learns the **accumulated** uncertainty the same way a critic learns value — by Bellman backups.

### Technical
\[
z=\gamma^2\hat u(s,a)+\gamma^2(1-d)\,U_\varphi(s',a'),
\quad
\mathcal{L}(\varphi)=\big(U_\varphi(s,a)-z\big)^2+\text{reg}.
\]
Network: `UNetwork` with softplus for non-negativity.

## 11.4 What is NOT claimed

- Not a new Bellman recursion for diffusion.  
- Not “first diffusion world model” (DIAMOND etc.).  
- Not exact Bayesian posterior (ensemble approximation, same as PETS/Luis).

---

# 12. Concentration & fixed-point guarantees

## 12.1 Theorem A1 (sketch)

### Simple
If next-values are bounded and samples are independent, take enough samples and your estimated disagreement \(\hat w\) is close to the true disagreement \(w\), with high probability.

### Technical
Assume \(\lvert\bar Q\rvert\le Q_{\max}\). For each member, Hoeffding ⇒ \(\hat\mu_i\) concentrates.  
Sample variance map \(f(\mu)=\frac1N\sum_i(\mu_i-\bar\mu)^2\) is Lipschitz on bounded domain ⇒  
\[
\mathbb{P}\big(\lvert\hat w-w^\star\rvert>\varepsilon\big)\le\delta
\quad\text{for}\quad
M=\tilde O\!\Big(\frac{Q_{\max}^2}{\varepsilon^2}\log\frac{N}{\delta}\Big).
\]
Similar for \(\hat g,\hat u\).  
Details: `research/proofs/theorem-A1-mc-concentration-sketch.md`.

## 12.2 Corollary A2

### Technical
UBE Bellman operator \(T_u U=\gamma^2 u+\gamma^2 P^{\pi,\bar p}U\) is \(\gamma^2\)-contraction in \(\|\cdot\|_\infty\).  
If \(\|\hat u-u\|_\infty\le\varepsilon\), fixed points satisfy
\[
\|\tilde U-U^\star\|_\infty\le\frac{\gamma^2}{1-\gamma^2}\varepsilon.
\]

### Simple
If each step’s uncertainty reward is slightly wrong, the long-horizon uncertainty map is still controlled — it can’t blow up forever because of contraction.

## 12.3 Toy validation

Run `python theory/toy_ube_mdp.py`:
- Exact DAG ensemble MRP.  
- UBE recovers \(\mathrm{Var}(V)\) to \(\sim 10^{-15}\).  
- MC error on \(w\) falls as \(M\) grows \(\sim 1/\sqrt{M}\).

---

# 13. Distillation / consistency (speed)

## 13.1 The WIMLE critique

### Simple
WIMLE (ICLR 2026 continuous control + uncertainty) basically says: diffusion is cool but **too slow** for online rollouts, so they use one-step IMLE instead.

### Technical
Online MBRL needs thousands of model steps per env step. Multi-step reverse diffusion multiplies cost by NFE \(K\).

## 13.2 Our answer

### Simple
Train a **student** that jumps from noise to next state in **one** network call, mimicking the teacher diffusion model.

### Technical
`udwm/models/consistency.py`:
- Teacher: multi-step diffusion ensemble (noise prediction).  
- Student: predicts clean \(x_0\) from \((x_t,t,s,a)\).  
- Loss: \(\|x_0^{\mathrm{student}}-x_0^{\mathrm{teacher}}\|^2\) with teacher \(x_0\) from analytic reconstruction of noise prediction.  
- Rollouts use student (**1 NFE**).  
- Config: `model.use_consistency_distill: true` → `configs/consistency_distill.yaml`.

## 13.3 Phase B (future theory)

Distillation changes the **estimand**: \(\hat U\) may track student-induced uncertainty, not full teacher reverse process. Need bias bound
\[
\big\lvert w^{(K)}-w^{(\infty)}\big\rvert\le f(\Delta_K,Q_{\max},N).
\]

---

# 14. Joint reward diffusion (Gap 1)

## 14.1 DIAMOND’s open problem

### Simple
DIAMOND authors literally wrote: we left combining reward/termination into the diffusion model for future work — it’s nontrivial.

### Technical
DIAMOND Limitations (NeurIPS 2024): separate CNN–LSTM \(R_\psi\) for reward/termination; joint objective + representation extraction cited as hard.

## 14.2 Our path

### Technical
When `joint_with_diffusion: true`:
\[
x_0=\big[\Delta s_{\mathrm{norm}},\, r_{\mathrm{norm}}\big],
\]
single denoising process samples \((s',r)\).  
Termination: auxiliary BCE head (binary discrete variable).  

### Code
`DiffusionDynamicsEnsemble(joint_reward=True)`, configs `joint_reward.yaml`.

---

# 15. Policy: SAC + optimism/pessimism

## 15.1 SAC

### Simple
Learn a soft, stochastic policy that maximizes reward + entropy (stays exploratory).

### Technical
Twin Q-networks, squashed Gaussian actor, automatic temperature \(\alpha\).  
File: `udwm/rl/sac.py`.

## 15.2 Uncertainty-aware objectives

### Simple
- **Optimism** (explore): act as if uncertain places might be great → \(Q+\lambda\sqrt{U}\).  
- **Pessimism** (offline/safety): assume uncertain places are bad → \(Q-\lambda\sqrt{U}\).

### Technical
\[
\pi \approx \arg\max_\pi\;
\mathbb{E}\big[\bar Q(s,a)+\lambda\sqrt{U(s,a)}-\alpha\log\pi(a\mid s)\big].
\]
Config: `agent.optimism_lambda` (positive explore, negative pessimism), `agent.use_ube`.

---

# 16. Extensive literature survey

## 16.1 Model-based RL & world models

| Work | Contribution | Assumptions / limits | Relation to us |
|---|---|---|---|
| **Sutton 1991 Dyna** | Real + imagined updates | Tabular classic | Conceptual ancestor |
| **Deisenroth & Rasmussen 2011 PILCO** | GP dynamics, analytic policy gradients | Scales poorly | Early Bayesian MBRL |
| **Chua et al. 2018 PETS** | Probabilistic ensembles + trajectory sampling | Gaussian members | Our `GaussianEnsemble` baseline |
| **Janner et al. 2019 MBPO** | Short branched rollouts + SAC | Ensemble PNN | **Our outer loop** |
| **Kaiser et al. 2019 SimPLe** | Video prediction Atari WM | Discrete | Pre-DIAMOND pixel WM |
| **Hafner et al. PlaNet/Dreamer/V3** | Latent RSSM, imagination in latent | Latent not diffusion | Complementary baseline class |
| **Yu et al. 2020 MOPO** | Offline: penalize one-step model uncertainty | One-step penalty | Contrast multi-step \(U\) |
| **Kidambi et al. MOReL / COMBO / RAMBO** | Offline conservatism variants | Various | Future offline chapter |

## 16.2 Uncertainty in RL

| Work | Contribution | Relation |
|---|---|---|
| **Osband et al.** randomized value functions / deep exploration | Model-free epistemic | Different lineage |
| **Depeweg et al. 2018** aleatoric/epistemic decomposition | Conceptual | Same split we use |
| **Chen et al. 2017** ensemble Q variance | Simple baseline | We compare conceptually to UBE |
| **O’Donoghue et al. 2018 UBE** | Upper bound recursion | Start of UBE line |
| **Zhou et al. 2020** tighter local \(w\) | Still ≥ true var | Gap characterized by Luis |
| **Luis et al. 2023 exact UBE** | \(U=\mathrm{Var}(V)\) under Assumps. 1–2 | **Our theorem foundation** |
| **Luis et al. 2024 QU-SAC** | Practical UBE actor-critic | Algorithmic template |
| **Sun et al. 2023** conformal UQ for diffusion dynamics | Planning-focused CP | Different UQ tool than UBE |

## 16.3 Diffusion & generative modeling

| Work | Contribution | Relation |
|---|---|---|
| **Ho et al. 2020 DDPM** | Denoising diffusion | Training backbone |
| **Song et al. 2021 Score-SDE** | Continuous reverse SDEs | Theory language for continuous time |
| **Song et al. 2023 Consistency Models** | One/few-step generation | Motivates our student |
| **Salimans & Ho progressive distillation** | Fewer NFEs | Same systems goal |
| **Karras et al. EDM** | Improved diffusion design space | DIAMOND uses EDM-style |
| **Chi et al. Diffusion Policy** | Diffuse **actions** | Different object (policy not dynamics) |

## 16.4 Diffusion world models & continuous control UQ

| Work | Contribution | Relation |
|---|---|---|
| **Alonso et al. 2024 DIAMOND** | Diffusion WM agent; Atari HNS 1.46; NeurIPS spotlight | Landmark; cites open continuous + joint R/T |
| **Diffusion World Model variants** | Multistep state/reward prediction | Adjacent; we focus UBE multi-step epistemic |
| **PolyGRAD / diffusion dynamics planning** | Continuous control diffusion dynamics | Dynamics yes; not Luis-style UBE |
| **Aghabozorgi et al. 2026 WIMLE** | IMLE multi-modal WM + ensembles; continuous control; **avoids diffusion** for speed | Direct competitor *setting*; we keep diffusion + distill + UBE |
| **Cheng et al. 2026 MBDPO** | Diffusion **policy** optimization inside WMs | Diffuses policy, not dynamics UQ |
| **Valdi (2026) etc.** | Value/latent diffusion WMs | Monitor overlap |

## 16.5 Multi-agent PO (Gap 2)

| Work | Contribution | Relation |
|---|---|---|
| **Oliehoek et al.** Dec-POMDP foundations | Formalism | Future |
| **Wang et al. AAMAS 2025** | Diffusion history→state fixed points; Jacobian deviation bounds; composite flow | **State** UQ done; we propose **value** lift |
| **Xu et al.** diffusion in Dec-POMDPs without full disagreement resolution | Prior | Wang et al. go further on state |
| **QMIX/VDN/MAPPO** | Multi-agent value learning | Credit assignment baselines later |

## 16.6 Generative modeling divergences (why not only Gaussians)

### Simple
Some losses make models **cover all modes** of the data; some make them **pick one mode** and look sharp but miss alternatives.

### Technical
- Forward KL / likelihood / many score-based objectives → mode covering (good for multi-modal dynamics).  
- Reverse KL → mode seeking.  
- IMLE: implicit likelihood-style matching used by WIMLE.  
- Diffusion: highly successful multi-modal generative family.

---

# 17. What we do differently

## 17.1 One paragraph (simple)

Most papers pick **one** of: better world models (DIAMOND), faster multi-modal models (WIMLE), or multi-step uncertainty theory (Luis UBE) with Gaussian nets. We connect them: **diffusion (and distilled) dynamics + Monte Carlo UBE local rewards + MBPO/SAC**, with optional joint reward, and a clear claim that we validate an **estimator**, not rewrite UBE from scratch.

## 17.2 Comparison matrix

| Capability | PETS/MBPO | Dreamer | DIAMOND | WIMLE | Luis UBE | **UDWM (us)** |
|---|---|---|---|---|---|---|
| Continuous control focus | ✓ | ✓ | △ (open) | ✓ | ✓ | ✓ |
| Multi-modal generative dynamics | △ Gaussian | latent | ✓ diffusion | ✓ IMLE | △ Gaussian | ✓ diffusion |
| Fast online rollouts | ✓ | ✓ | △ NFE | ✓ 1-step | ✓ | ✓ few-step + **distill** |
| Multi-step epistemic \(U\) (UBE) | △ ad hoc | △ | ✗ | △ ensemble | ✓ Gaussian est. | ✓ **MC for diffusion** |
| Exact UBE theory link | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ inherit + estimator |
| Joint reward in generative WM | often | yes | ✗ separate | yes | often | ✓ optional joint |
| Multi-agent value lift | ✗ | ✗ | ✗ | ✗ | ✗ | planned Gap 2 |
| Open code stack | some | yes | yes | — | ube-mbrl | **this repo** |

## 17.3 Differentiation bullets for a paper intro

1. **vs Luis / QU-SAC:** same recursion; new **sample-based local rewards** for implicit generative members.  
2. **vs DIAMOND:** continuous-control + multi-step \(U\) + joint reward path; not Atari pixel SOTA chase in v1.  
3. **vs WIMLE:** same continuous UQ-MBRL problem; we use **diffusion + distillation + UBE**, not IMLE.  
4. **vs MBDPO / Diffusion Policy:** they diffuse **policies/trajectories**; we diffuse **dynamics** and quantify **value epistemic uncertainty**.  
5. **vs naive ensemble-var of Q:** UBE propagates local model uncertainty through mean dynamics with \(\gamma^2\).

## 17.4 What would make a reviewer say “incremental”?

Avoid claiming:
- “We invent UBE for diffusion” (overclaim).  
- “First continuous diffusion WM” (likely false / contested).  

Prefer:
- “We prove/validate MC estimators of UBE local rewards under diffusion ensembles and demonstrate a full continuous-control stack with distillation.”

---

# 18. Codebase architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Real environment                         │
│                  (Gymnasium Box envs)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ transitions
                            ▼
                   ┌─────────────────┐
                   │  real_buffer    │
                   └────────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
   ┌────────────────┐ ┌────────────┐ ┌─────────────┐
   │ World model    │ │ Reward/T   │ │ (optional)  │
   │ Gaussian or    │ │ separate   │ │ Consistency │
   │ Diffusion ens. │ │ or joint   │ │ student     │
   └───────┬────────┘ └────────────┘ └──────┬──────┘
           │ imagination                    │ 1-NFE
           ▼                                │
   ┌─────────────────┐                      │
   │  model_buffer   │◄─────────────────────┘
   └────────┬────────┘
            │ mixed batches
            ▼
   ┌──────────────────────────────────────────┐
   │ SAC critics/actor  +  MC-UBE → U-network │
   │         Q ± λ √U                          │
   └──────────────────────────────────────────┘
```

**Package root:** `udwm/`  
**Configs:** `configs/*.yaml`  
**Theory toys:** `theory/`  
**Writing:** `papers/`, `docs/`, `research/`

---

# 19. File-by-file code guide

## 19.1 Config & utils

| File | Purpose |
|---|---|
| `configs/default.yaml` | Full default experiment |
| `configs/smoke_train.yaml` | Short CPU train |
| `configs/pendulum_long.yaml` | Longer Gap 3 run |
| `configs/joint_reward.yaml` | Gap 1 |
| `configs/consistency_distill.yaml` | 1-step student |
| `configs/ablation_fast.yaml` | Ablation suite |
| `udwm/utils/config.py` | YAML load + seeding |
| `udwm/utils/torch_utils.py` | `mlp`, soft target update, device |

**Learn configs first** — every flag maps to a design decision above.

## 19.2 Data

`udwm/data/replay_buffer.py`  
Circular buffer of \((s,a,r,s',d)\). Two instances in trainer: real vs model.

## 19.3 Models

### `gaussian_ensemble.py`
- \(N\) MLPs → mean/log-var of \(\Delta s\) and \(r\).  
- Loss: Gaussian NLL.  
- `sample_next`, `member_means` for UBE-friendly means.

### `diffusion_dynamics.py`
- `NoiseSchedule`: \(\beta_t,\bar\alpha_t\).  
- `ConditionalDenoiser`: \(\epsilon_\theta(x_t,t,s,a)\).  
- `DiffusionDynamicsEnsemble`: train + DDIM sample + joint reward pack/unpack + term head.  
- `sample_next_multi`: critical for MC-UBE.

### `reward_term.py`
Separate \(r\) and done heads (DIAMOND-style).

### `consistency.py`
- `ConsistencyStudent`, `distill_loss`, `DistilledWorldModel`.  
- Adapter so MC-UBE can call `sample_next_multi` on student.

### `world_model.py`
Factory `WorldModel.build(...)` selecting Gaussian / diffusion / joint / distilled.  
Unified `train_loss` + `rollout`.

## 19.4 Uncertainty

### `mc_ube.py`
- `MCUBELocalRewards` — implements §11 algorithm.  
- `UNetwork` — softplus \(U(s,a)\).  
- `ube_loss` — TD + regularization.

### `calibration.py`
Correlation of \(\sqrt{U}\) with \(\lvert\delta_{\mathrm{TD}}\rvert\).

## 19.5 RL

### `sac.py`
Squashed Gaussian actor, twin critics, auto-α, optional \(\lambda\sqrt{U}\).

### `trainer.py` (`MBPOTrainer`)
Full loop: env step → WM train → imagine → SAC → UBE → eval metrics.

## 19.6 Eval & scripts

| Script | Role |
|---|---|
| `smoke_test.py` | Sanity all modules |
| `train_mbpo.py` | CLI training |
| `run_ablations.py` | Variant grid → CSV/JSON |
| `benchmark_throughput.py` | Samples/sec |
| `plot_results.py` | Curves/bars |
| `evaluate.py` | Checkpoint metrics |
| `eval/metrics.py` | Return, MSE, calibration, throughput |
| `envs/registry.py` | Multi-env + dm_control note |

## 19.7 Theory toys

| File | Role |
|---|---|
| `theory/toy_ube_mdp.py` | Exact UBE = Var(V) on DAG MRP |
| `theory/toy_mc_ube_estimator.py` | MC vs closed-form Gaussian local rewards |

## 19.8 Research writing

| File | Role |
|---|---|
| `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` | Claim A north-star |
| `research/ASSUMPTIONS-LOG.md` | Living assumptions |
| `research/proofs/*` | Checklists / A1 sketch |
| `papers/LITERATURE_SURVEY.md` | Shorter survey (this doc §16 is expanded) |
| `papers/PAPER_DRAFT.md` | Paper skeleton |
| `papers/references.bib` | BibTeX |
| `docs/PROFESSOR_BRIEFING.md` | Talk script |

---

# 20. Math ↔ code map

| Math | Code |
|---|---|
| Ensemble \(\{\theta_i\}_{i=1}^N\) | `ensemble_size` members |
| \(p_{\theta_i}(s'\mid s,a)\) | `dynamics.sample_next(..., member=i)` |
| \(\Delta s\) diffusion target | `next_obs - obs` normalized |
| Joint \(x=[\Delta s,r]\) | `joint_reward=True` |
| \(\bar Q(s,a)\) | `SACAgent.q_min` |
| \(\pi(a\mid s)\) | `SACAgent.policy_tensor` / actor |
| \(\hat\mu_i,\hat w,\hat g,\hat u\) | `MCUBELocalRewards.estimate` |
| \(U(s,a)\) | `UNetwork` |
| UBE target \(z\) | `ube_targets` |
| \(\gamma^2\) | `gamma * gamma` in targets |
| \(Q+\lambda\sqrt{U}\) | `optimism_lambda` in SAC |
| Teacher NFE \(K\) | `sample_steps` |
| Student 1-NFE | `ConsistencyStudent` / distill config |
| Real vs model data | `real_buffer`, `model_buffer`, `real_ratio` |
| Calibration | `reliability_summary` |

---

# 21. Training loop step-by-step

### Simple timeline
1. Wiggle in the real world a bit; save memories.  
2. Update the world model on those memories.  
3. Dream short dreams; save dream memories.  
4. Improve actor/critic on mix of real + dream.  
5. Update uncertainty network with MC-UBE.  
6. Sometimes evaluate and log metrics.

### Technical (matches `MBPOTrainer.train`)
```
for t = 1..T:
  a_t ← random if t < warmup else π_θ(s_t)
  step env → real_buffer
  if t ≥ warmup and t % model_train_freq == 0:
     for epoch: minimize WM train_loss on real batches
     for many starts: rollout WM with π for horizon k → model_buffer
  if t ≥ warmup:
     for G gradient steps:
        batch ← mix(real, model; real_ratio)
        SAC critic + actor update (with optional √U)
        (once per cycle) MC-UBE estimate → U-net TD update
  if t % eval_freq == 0:
     log return, next-state MSE, reward MSE, U–TD correlation
```

---

# 22. Toy experiments (theory validation)

## 22.1 Exact MRP (`theory/toy_ube_mdp.py`)

**Setup:**  
\(s_0\) transitions to \(s_1\) w.p. \(p_i\) else \(s_2\); then terminal. Ensemble over \(p_i\).  
\(V(s_0)=\gamma(p R_1+(1-p)R_2)\).  

**Result:**  
\(U_{\mathrm{UBE}}(s_0)=\gamma^2 w = \mathrm{Var}_p V(s_0)\) exactly.  
MC \(\hat w\) RMSE decreases with \(M\).

## 22.2 Gaussian MC toy (`theory/toy_mc_ube_estimator.py`)

Compares closed-form local rewards vs MC vs biased distilled-style sampler — shows bias **does not** vanish with more samples (motivation for Phase B).

---

# 23. How to run everything

```bash
# Install
pip install -r requirements.txt

# Sanity
python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py -q
python theory/toy_ube_mdp.py

# Train
python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml
python -m udwm.scripts.train_mbpo --config configs/pendulum_long.yaml --steps 25000
python -m udwm.scripts.train_mbpo --config configs/joint_reward.yaml
python -m udwm.scripts.train_mbpo --config configs/consistency_distill.yaml

# Ablations + plots
python -m udwm.scripts.run_ablations --config configs/ablation_fast.yaml --steps 2000
python -m udwm.scripts.benchmark_throughput
python -m udwm.scripts.plot_results

# Flags
#   --model gaussian|diffusion
#   --joint-reward / --separate-reward
#   --no-ube
#   --device cpu|cuda
```

---

# 24. Three research gaps on one stack

| Gap | Question | Simple pitch | Technical pitch | Code |
|---|---|---|---|---|
| **3** | Continuous diffusion WM + multi-step \(U\) | Make DIAMOND-style models work for continuous robots *with* trust signals | MC-UBE + distill + MBPO/SAC on DMC-like envs | default / pendulum_long / consistency |
| **1** | Joint reward in diffusion | Stop bolting reward on the side | Joint generative \(p(s',r\mid s,a)\) | `joint_with_diffusion` |
| **2** | Multi-agent PO value uncertainty | Agents disagree about the world; how does that hit credit assignment? | Lift Wang et al. state deviation \(D\) to \(\mathrm{Var}(V)\) / multi-agent UBE | future `udwm/envs` |

**Build order:** Core → Gap 3 → Gap 1 → Gap 2.

---

# 25. Limitations & honest caveats

1. **Assumps. 1–2** fail under shared NN weights and cyclic envs (same as Luis deep RL). We clip \(u_{\min}\).  
2. **Ensemble ≠ true Bayesian posterior.**  
3. **Nested MC cost** \(O(N M K)\); distillation helps \(K\) but adds bias.  
4. **Function approximation:** U-net/Q-net are heuristic relative to tabular theory.  
5. **Pixels / Atari scale** not the current code path (state-based continuous first).  
6. **Joint termination** still aux head, not fully discrete diffusion.  
7. **Paper tables** need multi-seed strong benchmarks before claiming SOTA.  
8. **Gap 2** is aspirational until single-agent \(U\) is solid.

---

# 26. Glossary

| Term | Simple | Technical |
|---|---|---|
| World model | Learned simulator | \(p_\theta(s',r\mid s,a)\) |
| Imagination | Practice in simulator | Model rollouts into replay |
| Ensemble | Committee of models | \(\{p_{\theta_i}\}\) as approx \(\Phi\) |
| Epistemic | “I don’t know” | \(\mathrm{Var}_\theta \mathbb{E}[Y\mid\theta]\) |
| Aleatoric | “It’s random” | \(\mathbb{E}_\theta\mathrm{Var}[Y\mid\theta]\) |
| UBE | Bellman equation for uncertainty | Recursion for \(\mathrm{Var}_p V^{\pi,p}\) |
| Local \(u\) | One-step uncertainty reward | \(u=w-g\) (Luis) |
| \(U(s,a)\) | Multi-step uncertainty | Fixed point of UBE |
| NFE | How many NN calls to sample | Denoising steps |
| DDIM | Faster diffusion sampler | Non-Markovian / few-step |
| Consistency student | One-step generator | Distilled \(x_0\) map |
| SAC | Modern continuous RL agent | Max entropy actor-critic |
| MBPO | Short model rollouts + SAC | Branched imagination MBRL |
| OFU | Try uncertain options | Optimism in face of uncertainty |
| Pessimism | Distrust uncertain options | Offline safe policy learning |

---

# 27. Study plan

### Weekend crash course
| Block | Do this |
|---|---|
| 2h | Read §1–§8 (theory through Luis) carefully |
| 1h | Run `toy_ube_mdp.py`; re-derive why \(U=\gamma^2 w\) on that MRP |
| 2h | Read §10–§12 (diffusion + MC-UBE + concentration) |
| 2h | Read §18–§21; open `trainer.py` + `mc_ube.py` side by side |
| 1h | `smoke_test` + short `train_mbpo` |
| 1h | Skim §16–§17 (literature + differentiation) |
| 1h | Read `papers/PAPER_DRAFT.md` abstract + intro |

### Before talking to a professor
1. This doc §1, §7, §11, §17  
2. `docs/PROFESSOR_BRIEFING.md` Part A + C  
3. Be ready to say: *“We don’t re-derive UBE; we make UBE local rewards computable for diffusion ensembles and build the MBRL stack.”*

### Before writing experiments section
1. Multi-seed ablations via `run_ablations`  
2. Throughput figure via `benchmark_throughput` + `plot_results`  
3. Calibration numbers from `evaluate_full`  
4. Toy MRP figure from theory scripts  

---

# Appendix A — Core equations cheat sheet

**Value Bellman**
\[
V(s)=\mathbb{E}_{a,s'}\big[r+\gamma V(s')\big]
\]

**Posterior value variance**
\[
U(s)=\mathrm{Var}_{p\sim\Phi}\big[V^{\pi,p}(s)\big]
\]

**Exact UBE**
\[
U(s)=\gamma^2 u(s)+\gamma^2\mathbb{E}_{a,s'\sim\pi,\bar p}[U(s')]
\]

**Local (conceptual)**
\[
u = \underbrace{\mathrm{Var}_{\bar p}[\bar V]}_{\text{total under mean kernel}}
-
\underbrace{\mathbb{E}_p[\mathrm{Var}_p V]}_{\text{avg aleatoric}}
\quad(\text{related to }w-g)
\]

**MC estimates**
\[
\hat\mu_i=\frac1M\sum_m \bar Q(s'_{i,m},a'_{i,m}),\quad
\hat w=\mathrm{Var}_i(\hat\mu_i),\quad
\hat u=\max(u_{\min},\hat w-\hat g)
\]

**Policy**
\[
\max_\pi\;\mathbb{E}\big[\bar Q+\lambda\sqrt{U}-\alpha\log\pi\big]
\]

**Concentration (schematic)**
\[
M \gtrsim \frac{Q_{\max}^2}{\varepsilon^2}\log\frac{N}{\delta}
\;\Rightarrow\;
\lvert\hat w-w^\star\rvert\le\varepsilon \text{ w.h.p.}
\]

**Fixed-point perturbation**
\[
\|\tilde U-U^\star\|_\infty\le\frac{\gamma^2}{1-\gamma^2}\varepsilon
\]

---

# Appendix B — Key citations (memorize)

1. Janner et al. 2019 — MBPO  
2. Chua et al. 2018 — PETS  
3. O’Donoghue et al. 2018 — UBE upper bounds  
4. Luis et al. 2023 — Exact UBE (AISTATS)  
5. Alonso et al. 2024 — DIAMOND (NeurIPS)  
6. Song et al. 2023 — Consistency models  
7. Aghabozorgi et al. 2026 — WIMLE  
8. Wang et al. 2025 — Multi-agent diffusion PO bounds  
9. Ho et al. 2020 — DDPM  
10. Haarnoja et al. 2018 — SAC  

Full bib: `papers/references.bib`. Living short survey: `papers/LITERATURE_SURVEY.md`.

---

*This is the master study document. When code or claims change, update §11, §17, §19, and Appendix A first.*
