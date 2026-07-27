# Project Briefing: Uncertainty-Aware Diffusion World Models

**Who this is for:** You — when explaining the project to a professor, advisor, or lab mate.  
**Two layers:** *Simple language* first, then *technical* detail you can go deeper on if they ask.

---

# Part A — Simple language (2–3 minute pitch)

## What problem are we solving?

Modern AI agents (robots, game AIs, controllers) often learn by **trial and error in the real world**. That is slow, expensive, and sometimes dangerous.

A popular fix is a **world model**: a neural network that *simulates* what happens next if the agent takes an action — like a learned video-game engine of the environment. The agent can then practice **inside its imagination** thousands of times without touching the real world.

**The catch:** the simulator is imperfect. If the agent trusts a wrong prediction, it can learn a policy that looks great in imagination but fails in reality — or it can be overconfident in dangerous regions.

So we need two things together:

1. A **strong simulator** (good at complex, multi-modal “what happens next?”).
2. An honest sense of **uncertainty** — “how much should we trust this simulated future?”

## What is the usual approach in the field?

| Trend | What people do | Weakness |
|---|---|---|
| **Gaussian ensembles (PETS / MBPO)** | Ensemble of simple “next state ~ Normal(μ,σ)” networks | Average over modes; blur multi-modal physics (contact, branching outcomes) |
| **Latent world models (Dreamer, etc.)** | Compress pixels to latent space, predict there | Great sample efficiency; less explicit image-space fidelity / different UQ story |
| **Diffusion world models (DIAMOND, 2024)** | Use diffusion (like Stable Diffusion) to generate next frames | Strong visuals / multi-modality; **slow** sampling; reward/done often a **separate** network; little multi-step **value** uncertainty |
| **Fast generative alternatives (WIMLE / IMLE)** | One-step generative models + ensembles for continuous control | Explicitly **avoid diffusion** because iterative sampling is too slow for online RL |
| **Diffusion policies (not world models)** | Diffuse over *actions*, not environment dynamics | Different problem — “what action to take,” not “what the world does” |

**Bottom line industry/academia trend:**  
Generative world models are hot (especially diffusion after DIAMOND). Uncertainty-aware continuous control is also hot (WIMLE). **Connecting generative diffusion dynamics with multi-step epistemic uncertainty for RL value functions is still under-developed.**

## What are *we* doing differently?

We build a **shared stack** that treats uncertainty as first-class:

1. **Diffusion (or Gaussian) ensemble** as the world model of next state.
2. **Fast few-step sampling** (DDIM-style) so imagination is cheap enough for online RL — answering WIMLE’s “diffusion is too slow” critique.
3. **MC-UBE**: estimate *local* uncertainty from ensemble samples, then propagate it through an **Uncertainty Bellman Equation** so uncertainty compounds over imagined horizons — not just “next-step disagreement.”
4. Use that multi-step uncertainty \(U\) for **exploration** (optimism) or **safety/offline** (pessimism): \(Q \pm \lambda\sqrt{U}\).
5. Path to **joint reward inside diffusion** (DIAMOND’s own stated open problem) and later multi-agent partial observability.

**One sentence:**  
*We make diffusion world models usable inside model-based RL by combining fast sampling with multi-step epistemic uncertainty (UBE), instead of treating generative models as black-box simulators with only next-step variance.*

## What have we built so far? (demoable)

A working Python package `udwm`:

- Train world models (Gaussian ensemble **or** diffusion ensemble) on Gymnasium continuous control (Pendulum).
- Separate reward/termination head **or** **joint reward+state diffusion**.
- SAC agent trained with MBPO-style imagination.
- Sample-based UBE local rewards + U-network.
- Evaluation: return, next-state/reward MSE, uncertainty calibration, throughput vs sampling steps.

```bash
python -m udwm.scripts.smoke_test
python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml
python -m udwm.scripts.benchmark_throughput
```

## Three research “gaps” we can publish toward

1. **Gap 3 (primary):** Continuous control with **uncertainty-quantified distilled diffusion world models** + UBE.  
2. **Gap 1:** Put **reward (and termination)** into the diffusion process itself (DIAMOND authors left this open).  
3. **Gap 2 (stretch):** Take multi-agent diffusion **state-reconstruction error bounds** (Wang et al., AAMAS 2025) and lift them to **value uncertainty**.

We do **not** build three separate systems — one shared core, then branch.

---

# Part B — Technical language (for deeper discussion)

## Formal setup

Infinite-horizon discounted MDP \(\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)\).  
Agent maintains approximate posterior / ensemble \(\Phi_t\) over transition models.  
Value under model \(p\): \(V^{\pi,p}\). Epistemic object of interest:

\[
U_t^\pi(s) \;=\; \mathrm{Var}_{p\sim\Phi_t}\big[V^{\pi,p}(s)\big].
\]

**Luis et al. (AISTATS 2023)** show under independent transitions + acyclic (time-augmented) MDPs that \(U_t^\pi\) satisfies an **exact Uncertainty Bellman Equation**:

\[
U(s) = \gamma^2 u(s) + \gamma^2 \mathbb{E}_{a,s'\sim\pi,\bar p}[U(s')],
\]

with local reward \(u(s)\) that **subtracts average aleatoric next-value noise** from total predictive variance of mean values (Zhou’s \(w\) overestimates pure epistemic).

**Critical observation (our positioning):** Theorem 1 is **distribution-agnostic** — it does not require Gaussian \(p(s'|s,a)\). Gaussians appear only in Luis’s **implementation** (closed-form means/variances). Diffusion defines \(p\) only via sampling ⇒ we need **Monte Carlo estimators** of \(u,w\) and concentration guarantees (Claim A).

## Architecture we implemented

```
Real env → D_real
    → train ensemble dynamics {p_θi}   # Gaussian NLL or DDPM noise-pred on Δs
    → optional joint channel r in diffusion (Gap 1)
    → MBPO rollouts into D_model (few-step DDIM)
    → SAC on mix(D_real, D_model)
    → MC-UBE-Local: samples → ŵ, ĝ, û → train U_φ
    → policy objective uses Q̄ ± λ √U
```

### Dynamics

- **Gaussian ensemble:** each member predicts \((\mu_{\Delta s},\sigma_{\Delta s},\mu_r,\sigma_r)\); NLL training.  
- **Diffusion ensemble:** DDPM on normalized \(\Delta s = s'-s\); optional joint \(x=[\Delta s_{\mathrm{norm}}, r_{\mathrm{norm}}]\); DDIM-style \(K\)-step sampling.  
- **Termination:** light BCE head (binary is awkward in pure Gaussian noise space even in joint mode).

### MC-UBE-Local (core novelty interface)

For members \(i=1..N\), sample \(m\) next states from \(p_{\theta_i}\), evaluate \(\bar Q(s',a')\):

\[
\hat\mu_i = \mathrm{mean}_m Q,\quad
\hat v_i = \mathrm{var}_m Q,\quad
\hat w = \mathrm{var}_i(\hat\mu_i),\quad
\hat g = \mathrm{mean}_i(\hat v_i),\quad
\hat u = \max(u_{\min},\hat w-\hat g).
\]

UBE TD target for \(U_\varphi\): \(z = \gamma^2\hat u + \gamma^2(1-d)\,U_{\bar\varphi}(s',a')\).

### Theory targets (not all code-proved yet)

| ID | Claim | Status |
|---|---|---|
| A1 | MC \(\hat w,\hat u\) concentrate; \(M = O(Q_{\max}^2\varepsilon^{-2}\log(N/\delta))\) | Sketch + toy numerics |
| A2 | \(\|\tilde U - U^\star\|_\infty \le \frac{\gamma^2}{1-\gamma^2}\varepsilon\) | Standard contraction argument |
| B1 | Distillation bias \(\|w^{(K)}-w^{(\infty)}\|\) | Phase 2 |
| Gap1 | Joint generative model for \((s',r)\) improves imagination return / reward calibration | Implemented path; empirics pending |
| Gap2 | \(\|D\|_{\mathrm{state}} \Rightarrow\) bound on value error under Lip\(V\) | Future |

## Relation to landmark papers

| Paper | They contribute | We add / differ |
|---|---|---|
| **MBPO** (Janner et al.) | Branched imagination + model-based policy | Same outer loop; richer dynamics + UBE |
| **PETS** (Chua et al.) | Ensemble Gaussian uncertainty | Baseline; multi-modal generative alternative |
| **Luis UBE** | Exact recursion for \(\mathrm{Var}(V)\) | Keep recursion; **new estimator class** for implicit models |
| **DIAMOND** | Diffusion WM SOTA Atari-in-imagination | Continuous control path; joint R; multi-step \(U\) not in DIAMOND |
| **WIMLE** | Uncertainty-aware continuous control with **IMLE** (not diffusion) | Same problem setting with **diffusion/consistency** + UBE |
| **MBDPO** | Diffusion *policies* in world models | We diffuse *dynamics*, not the policy |
| **Wang et al. AAMAS’25** | Diffusion PO state reconstruction bounds | Lift to value-level uncertainty (Gap 2) |

## Honest limitations (say this unprompted — builds trust)

1. **Assumptions 1–2** of UBE (independence, acyclicity) are violated by shared NN weights and cyclic envs — same caveat as Luis deep RL; we clip \(u_{\min}\ge 0\).  
2. Ensemble ≠ true Bayesian posterior.  
3. CPU Pendulum demos ≠ paper-ready DMC/Humanoid numbers yet.  
4. Nested MC for UBE is expensive; need few-step samplers + caching.  
5. Joint termination fully inside diffusion is still incomplete (binary done via aux head).  
6. Gap 2 not started in code beyond stubs.

## What success looks like (for a thesis / paper)

**Minimum viable paper (Gap 3):**  
On continuous control, show (i) distilled diffusion WM matches/beats Gaussian ensemble MBPO on return or multi-modal tasks, (ii) multi-step \(U\) improves exploration or offline pessimism vs one-step ensemble variance, (iii) calibration plots and throughput tables addressing WIMLE’s latency critique.

**Strong follow-up (Gap 1):**  
Cite DIAMOND Limitations §3; show joint \((s',r)\) diffusion matches/beats separate \(R_\psi\) on reward MSE and agent return.

**Stretch (Gap 2):**  
Theorem + SMACv2-style empirics connecting reconstruction \(D\) to \(\mathrm{Var}(V)\).

---

# Part C — Conversation cheat sheet

### If they ask: “Is this novel?”

> “The UBE recursion already exists and is general. What’s missing — and what we focus on — is a **validated sample-based estimator** of local UBE rewards when the world model is a **diffusion ensemble**, plus making that stack work for **continuous-control MBRL** with **fast sampling**. DIAMOND proved diffusion WMs; WIMLE proved uncertainty-aware continuous control without diffusion; we connect those threads.”

### If they ask: “Why not just Dreamer?”

> “Dreamer is an excellent latent RSSM baseline. Our contribution is specifically about **generative diffusion dynamics** and **Bellman-propagated epistemic uncertainty**. We’re complementary, not a Dreamer replacement in v1.”

### If they ask: “Why diffusion over Gaussians?”

> “Unimodal Gaussians average modes. Contact-rich or multi-modal transitions need mode-covering generative models. Diffusion is currently the strongest practical tool for that; the open systems issue is speed + uncertainty for RL — which is exactly our engineering and theory target.”

### If they ask: “What’s the mathematical contribution?”

> “Estimator concentration for Monte Carlo UBE local rewards under implicit kernels, fixed-point perturbation of the UBE operator, and (later) distillation bias and joint generative objectives for reward. We explicitly do **not** claim a new UBE recursion.”

### If they ask: “What can you show me today?”

> “Runnable MBPO+SAC with Gaussian or diffusion ensembles, MC-UBE, joint-reward diffusion mode, throughput benchmarks vs sampling steps, and evaluation metrics (return, s′/r MSE, U–TD calibration).”

### If they ask: “Timeline?”

> “Shared core: done. Gap 3 paper experiments: next. Gap 1 joint R empirics: in parallel on same code. Gap 2: after single-agent \(U\) is solid.”

---

# Part D — Map of the repository (for a quick screen-share)

| Path | What to say |
|---|---|
| `udwm/models/diffusion_dynamics.py` | “Diffusion ensemble; joint reward channel is Gap 1.” |
| `udwm/uncertainty/mc_ube.py` | “Sample-based local UBE rewards + U-network.” |
| `udwm/rl/trainer.py` | “MBPO loop wiring everything.” |
| `udwm/eval/metrics.py` | “How we measure success: return, model error, calibration, throughput.” |
| `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` | “Theory north-star.” |
| `docs/PROFESSOR_BRIEFING.md` | “This document.” |
| `configs/*.yaml` | “Experiment knobs without code changes.” |

---

# Part E — Suggested 5-slide structure for a lab meeting

1. **Motivation:** Imagination RL needs simulators *and* trust.  
2. **Landscape:** PETS → DIAMOND → WIMLE; the hole we fill.  
3. **Method:** Diffusion ensemble + few-step sample + MC-UBE + SAC.  
4. **Theory claim:** Estimator validity for existing UBE, not a new recursion.  
5. **Status + plan:** Working core; Gap 3 → 1 → 2; demos.

---

# Part F — Key citations (memorize 5)

1. Janner et al., 2019 — MBPO  
2. Luis et al., 2023 — Exact UBE (AISTATS)  
3. Alonso et al., 2024 — DIAMOND (NeurIPS)  
4. Aghabozorgi et al., 2026 — WIMLE (ICLR; continuous control + UQ, not diffusion)  
5. Wang et al., 2025 — Multi-agent diffusion PO bounds (AAMAS)  

Optional: O'Donoghue 2018 (UBE origin), Song 2023 (consistency models), Chua 2018 (PETS).

---

*Last updated: 2026-07 — matches `udwm` package with joint reward path + eval suite.*
