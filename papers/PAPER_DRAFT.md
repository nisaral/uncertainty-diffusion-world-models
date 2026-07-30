# Monte Carlo Uncertainty Bellman Rewards for Diffusion World Models

**Working draft — not submission-ready**  
**Status:** Outline + written introduction/method/experiments skeleton  
**Code:** https://github.com/nisaral/uncertainty-diffusion-world-models  

---

## Abstract *(draft)*

Model-based reinforcement learning (MBRL) can train policies inside learned world models, but long-horizon decisions require multi-step epistemic uncertainty—not only one-step predictive variance. The Uncertainty Bellman Equation (UBE) of Luis et al. (2023) gives an exact recursion for posterior value variance under mild assumptions, yet practical estimators assume Gaussian ensemble dynamics. Diffusion world models (e.g., DIAMOND) define transitions only implicitly through denoising, making closed-form local UBE rewards unavailable. We develop **MC-UBE**, sample-based estimators of local uncertainty rewards for ensembles of diffusion (and consistency-distilled) dynamics, with concentration arguments and fixed-point perturbation guarantees relative to the population UBE. We instantiate a full MBPO+SAC stack with Gaussian baselines, multi-step DDIM and one-step distilled rollouts, optional joint reward–state diffusion, and uncertainty-aware policy objectives \(Q\pm\lambda\sqrt{U}\). Preliminary continuous-control experiments and exact toy MDPs support estimator validity and systems feasibility.

**Keywords:** model-based RL, diffusion world models, epistemic uncertainty, Uncertainty Bellman Equation, consistency distillation

---

## 1. Introduction

Training agents purely from real environment interaction is sample-expensive. World models enable *imagination*: the agent optimizes a policy against a learned simulator [janner2019mbpo, hafner2023dreamerv3]. Two pressures shape modern world models:

1. **Expressivity.** Unimodal Gaussian dynamics (PETS-style ensembles [chua2018pets]) underfit multi-modal transitions. Diffusion models provide strong multi-modal generative dynamics and recently powered Atari-scale imagination agents (DIAMOND [alonso2024diamond]).
2. **Trust.** Policies that exploit model errors fail on the true environment. Multi-step *epistemic* value uncertainty is the right object for exploration and offline pessimism [odonoghue2018ube, luis2023ube, yu2020mopo].

**Problem.** Luis et al. show that posterior value variance obeys an exact UBE under distribution-agnostic assumptions—but deep RL instantiations compute local rewards using **Gaussian** means/variances. Diffusion members expose only **samples**. Separately, online MBRL requires **fast** sampling; iterative diffusion is costly [aghabozorgi2026wimle]. DIAMOND further leaves joint reward/termination modeling inside diffusion as future work.

**Contributions.**

1. **Estimator theory.** We formalize Monte Carlo estimators \(\hat u,\hat w\) of UBE local rewards for diffusion/consistency ensembles and sketch finite-sample concentration + UBE fixed-point stability (Claims A1–A2).
2. **Systems.** An open MBPO+SAC implementation with Gaussian and diffusion ensembles, few-step DDIM, optional **consistency distillation** (one NFE), optional **joint reward diffusion**, and U-network training.
3. **Empirics.** Exact DAG-MRP verification that UBE recovers \(\mathrm{Var}(V)\); continuous-control ablations (Gaussian vs diffusion vs joint reward vs no-UBE); throughput vs sampling steps.

---

## 2. Related work

*(Expand from `papers/LITERATURE_SURVEY.md`.)*

**MBRL.** Dyna, MBPO, PETS, Dreamer-family.  
**UBE.** O'Donoghue → Zhou → Luis exact UBE; QU-SAC.  
**Diffusion WM.** DIAMOND; concurrent diffusion dynamics/planning work.  
**Fast generative RL.** Consistency models [song2023consistency]; WIMLE/IMLE [aghabozorgi2026wimle].  
**Offline pessimism.** MOPO-style one-step penalties vs multi-step \(U\).

**Gap we fill:** multi-step UBE-compatible uncertainty for *implicit generative* dynamics under online latency constraints.

---

## 3. Background

### 3.1 MDPs and posterior value variance

Standard discounted MDP. Posterior \(\Phi_t\) over transition kernels induces \(U_t^\pi(s)=\mathrm{Var}_{p\sim\Phi_t}[V^{\pi,p}(s)]\).

### 3.2 Exact UBE (Luis et al., 2023)

Under independent transitions and acyclic structure,

\[
U(s)=\gamma^2 u(s)+\gamma^2\mathbb{E}_{a,s'\sim\pi,\bar p}[U(s')],
\]

with local \(u=w-g\) subtracting average aleatoric next-value noise from Zhou-style \(w\). **No Gaussian assumption in the theorem.**

### 3.3 Diffusion dynamics

Conditional DDPM on \(\Delta s=s'-s\) (optional joint channel for reward). Sampling via \(K\)-step DDIM or one-step consistency student.

---

## 4. Method

### 4.1 Ensemble as approximate posterior

\(\Phi_t=\mathrm{Unif}\{\theta_i\}_{i=1}^N\) over dynamics members (Gaussian NLL or diffusion noise-prediction).

### 4.2 MC-UBE-Local

For each member \(i\), draw \(M\) next states (closed-form mean if Gaussian; DDIM/student if diffusion). Evaluate \(\bar Q(s',a')\):

\[
\hat\mu_i=\mathrm{mean}_m Q,\quad
\hat v_i=\mathrm{var}_m Q,\quad
\hat w=\mathrm{var}_i\hat\mu_i,\quad
\hat g=\mathrm{mean}_i\hat v_i,\quad
\hat u=\max(u_{\min},\hat w-\hat g).
\]

Train \(U_\varphi\) with targets \(z=\gamma^2\hat u+\gamma^2(1-d)U_{\bar\varphi}(s',a')\).

### 4.3 Concentration (sketch)

Bounded \(Q\Rightarrow\) Hoeffding on \(\hat\mu_i\); Lipschitz sample-variance map \(\Rightarrow\) \(\lvert\hat w-w^\star\rvert\le\varepsilon\) w.h.p. for \(M=\tilde O(Q_{\max}^2/\varepsilon^2\log N)\). UBE is a \(\gamma^2\)-contraction \(\Rightarrow\) \(\lVert\tilde U-U^\star\rVert_\infty\le\frac{\gamma^2}{1-\gamma^2}\varepsilon\).

### 4.4 Distillation-aware sampling

Teacher: multi-step diffusion. Student: one-step \(x_0\) predictor distilled by matching teacher’s analytic \(x_0\) targets. Imagination uses student (1 NFE). Distillation bias deferred to Phase B bound.

### 4.5 Joint reward diffusion (Gap 1 path)

Denoise \(x=[\Delta s_{\mathrm{norm}}, r_{\mathrm{norm}}]\) jointly; termination via light BCE head.

### 4.6 Policy optimization

SAC [haarnoja2018sac] on mixed real/model batches (MBPO). Optional \(Q+\lambda\sqrt{U}\) (exploration) or \(Q-\lambda\sqrt{U}\) (pessimism).

---

## 5. Experiments

### 5.1 Exact toy MRP
Acyclic ensemble Bernoulli MRP: UBE recovers \(\mathrm{Var}(V)\) to numerical precision; MC \(w\) RMSE decreases with \(M\).

### 5.2 Throughput
Samples/sec vs DDIM steps and Gaussian baseline (WIMLE latency narrative).

### 5.3 Continuous control ablations (Pendulum / DMC)
| Variant | Return | \(s'\) MSE | Reward MSE | \(U\)–TD corr |
|---|---|---|---|---|
| Gaussian+UBE | *TBD* | | | |
| Diffusion+UBE | *TBD* | | | |
| Joint-R diffusion+UBE | *TBD* | | | |
| Diffusion, no UBE | *TBD* | | | |
| Distilled 1-NFE+UBE | *TBD* | | | |

### 5.4 Calibration
Reliability of \(\sqrt{U}\) vs TD residual magnitude.

---

## 6. Discussion and limitations

- UBE Assumps. 1–2 violated by function approximation (shared caveat with Luis).  
- Ensemble ≠ Bayesian posterior.  
- Nested MC cost; distillation introduces bias not yet bounded tightly.  
- Pixel/DIAMOND-scale not yet attempted.  
- Multi-agent Gap 2 future work [wang2025diffusionpo].

---

## 7. Conclusion

We connect exact UBE theory to diffusion world models via Monte Carlo local-reward estimators, and provide a full open-source MBRL stack with distillation and joint-reward paths toward continuous-control uncertainty-aware imagination.

---

## Checklist for coauthors / advisor

- [ ] Tighten Theorem A1 constants; full proof appendix  
- [ ] Multi-seed DMC results (≥3 seeds)  
- [ ] Compare against WIMLE numbers if reimplementation feasible  
- [ ] Ethics / compute statement  
- [ ] Expand related work to 1.5 pages  
- [ ] Figures: architecture, toy concentration, learning curves, throughput  

---

## Appendix pointers (repo)

| Appendix topic | Repo path |
|---|---|
| MC estimator details | `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` |
| Assumptions log | `research/ASSUMPTIONS-LOG.md` |
| Toy MRP | `theory/toy_ube_mdp.py` |
| Code guide | `docs/CODE_AND_THEORY_GUIDE.md` |
| Literature | `papers/LITERATURE_SURVEY.md` |
