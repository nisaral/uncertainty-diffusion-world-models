# Code + Theory Guide (read this to understand the whole project)

**Audience:** You — when you want one document that connects *math*, *research story*, and *every major file in the repo*.  
**Companion docs:** `docs/PROFESSOR_BRIEFING.md` (talking points), `papers/PAPER_DRAFT.md` (paper), `papers/LITERATURE_SURVEY.md` (citations).

---

# 0. The project in one page

## Goal
Train RL agents **inside a learned simulator** (world model), using:

1. **Diffusion** (or Gaussian ensembles) to predict “what happens next,”
2. **Uncertainty** about those predictions, propagated over many imagined steps via the **Uncertainty Bellman Equation (UBE)**,
3. A **policy** that uses that uncertainty for exploration or caution: \(Q \pm \lambda\sqrt{U}\).

## Why it matters
- Simulators that are wrong but **overconfident** produce bad policies.
- Diffusion is great at multi-modal physics/images but **slow** and **hard to plug into classic UBE formulas** (no closed-form mean/variance).
- We fix the “plug in” problem with **Monte Carlo UBE**, and the speed problem with **few-step / consistency distillation**.

## Three research gaps on one codebase

| Gap | Research question | Code switch |
|---|---|---|
| **3 (primary)** | Continuous control + diffusion WM + multi-step \(U\) | `model.type=diffusion`, `use_ube=true` |
| **1** | Put reward inside diffusion (DIAMOND open problem) | `reward_term.joint_with_diffusion=true` |
| **2 (later)** | Multi-agent state-error → value uncertainty | `udwm/envs` stubs only |

---

# 1. Theory you need (minimal, in order)

## 1.1 MDP and value
- State \(s\), action \(a\), reward \(r\), discount \(\gamma\in[0,1)\).
- Policy \(\pi(a\mid s)\).
- Value \(V^{\pi,p}(s)\): expected discounted return under transitions \(p\).

**Bellman expectation:**
\[
V(s)=\mathbb{E}_{a\sim\pi,\,s'\sim p}\big[r(s,a)+\gamma V(s')\big].
\]

## 1.2 Epistemic vs aleatoric
- **Aleatoric:** randomness of the env even if you know \(p\) (dice).
- **Epistemic:** uncertainty about *which* \(p\) is true given limited data.

Law of total variance:
\[
\mathrm{Var}(Y)=\mathbb{E}[\mathrm{Var}(Y\mid\theta)]+\mathrm{Var}(\mathbb{E}[Y\mid\theta]).
\]
Second term = epistemic if \(\theta\) is model parameters.

## 1.3 What UBE is
You care about:
\[
U^\pi(s)=\mathrm{Var}_{p\sim\Phi}\big[V^{\pi,p}(s)\big]
\]
where \(\Phi\) is a posterior (we approximate with an **ensemble**).

**Luis et al. 2023:** under Assumps. 1–2 (independent transitions across states; acyclic episodes), \(U\) satisfies a Bellman-like recursion with a **local uncertainty reward** \(u(s)\):

\[
U(s)=\gamma^2 u(s)+\gamma^2\mathbb{E}_{a,s'\sim\pi,\bar p}[U(s')].
\]

Intuition: uncertainty about values is like “rewards” of uncertainty, discounted by \(\gamma^2\), flowing through the **mean** dynamics \(\bar p\).

Important:
- The **theorem does not need Gaussians**.
- Gaussians only make \(u\) easy to compute in code (means/vars closed form).

## 1.4 Local rewards \(w\), \(g\), \(u\)
- \(w\): how much ensemble members **disagree** on the expected next value (epistemic-ish).
- \(g\): average **within-member** variance of next value (aleatoric-ish).
- \(u = w - g\) (with clipping in practice): local signal that UBE integrates into multi-step \(U\).

## 1.5 Our Claim A (the actual novelty framing)
When each ensemble member is a **diffusion model**, you only have **samples** of \(s'\).  
So:

1. Estimate \(\hat u,\hat w\) by Monte Carlo.
2. Prove/bound that \(\hat u\approx u\) when you take enough samples (Hoeffding-style).
3. Because UBE is a \(\gamma^2\)-contraction, small local error ⇒ small error in \(U\).

You are **not** inventing a new recursion; you invent a **valid estimator** for an existing recursion under diffusion.

## 1.6 Distillation (latency)
Full diffusion needs many denoising steps.  
**Student** network learns to predict the clean next-state (and maybe reward) in **1 step**, matching the teacher.  
That answers: “Isn’t diffusion too slow for online RL?” (WIMLE’s critique).

## 1.7 Joint reward (Gap 1)
DIAMOND predicts frames with diffusion and **bolts on** a separate reward/termination net.  
We can denoise \([\Delta s, r]\) **together** so reward is part of the generative process.

---

# 2. Algorithm end-to-end (what training does)

```
repeat:
  1. Act in real env a bit → store (s,a,r,s',done) in real_buffer
  2. Every model_train_freq steps:
       train world model on real_buffer
       roll out policy in world model → model_buffer
  3. Update SAC actor/critics on mix(real_buffer, model_buffer)
  4. Estimate MC-UBE local û on a batch → train U-network
  5. Policy uses Q ± λ√U (if enabled)
```

This is **MBPO-style** model-based RL with an extra U-network.

---

# 3. Repository map (where is what?)

```
uncertainty-diffusion-world-models/
├── udwm/                    # Python package (the implementation)
│   ├── models/              # World models
│   ├── uncertainty/         # MC-UBE + U-net
│   ├── rl/                  # SAC + MBPO trainer
│   ├── eval/                # Metrics
│   ├── envs/                # Env registry (multi-env)
│   ├── data/                # Replay buffer
│   ├── scripts/             # CLI entry points
│   └── utils/               # Config, seed, MLP helpers
├── configs/                 # YAML experiment configs
├── theory/                  # Pure math toys (no neural nets)
├── research/                # Theory notes & proof sketches
├── papers/                  # Literature survey + paper draft
├── docs/                    # This guide + professor briefing
└── tests/                   # Unit tests
```

---

# 4. Code walkthrough (file by file)

## 4.1 Config & utils

| File | Role |
|---|---|
| `configs/*.yaml` | All knobs: env, model type, UBE, MBPO steps |
| `udwm/utils/config.py` | `load_config`, `set_seed` |
| `udwm/utils/torch_utils.py` | `mlp`, `soft_update`, device |

**Read first:** `configs/default.yaml` — if you understand every key, you understand the system.

### Important config keys
- `model.type`: `gaussian` | `diffusion`
- `model.sample_steps`: DDIM steps at inference
- `model.use_consistency_distill`: train 1-step student
- `reward_term.joint_with_diffusion`: Gap 1 joint reward
- `agent.use_ube` / `optimism_lambda`: multi-step uncertainty in policy
- `ube.m_samples`: MC samples per ensemble member for local \(u\)
- `mbpo.*`: how much real vs imagined data, rollout length, etc.

---

## 4.2 Data

### `udwm/data/replay_buffer.py`
Stores transitions; `sample(batch_size)` returns torch tensors on device.

Two buffers in the trainer:
- **real_buffer** — true environment
- **model_buffer** — imagination

---

## 4.3 Models (the world simulator)

### `udwm/models/gaussian_ensemble.py` — baseline
- \(N\) MLPs, each predicts mean/log-var of \(\Delta s\) and reward.
- Trained with Gaussian NLL.
- `sample_next` draws from member Gaussians.
- Used for: PETS-style baseline + closed-form-friendly UBE path.

**Theory link:** Luis App. D.1 style “take the mean of the Gaussian.”

### `udwm/models/diffusion_dynamics.py` — main generative model
- Each ensemble member is a **conditional denoiser** \(\epsilon_\theta(x_t,t,s,a)\).
- Trains DDPM noise prediction on normalized \(\Delta s\) (and optional reward channel).
- Sampling: **DDIM-like** loop with `sample_steps` (few-step).
- `joint_reward=True`: \(x=[\Delta s_{\mathrm{norm}}, r_{\mathrm{norm}}]\).
- Termination: small BCE head `term_head` (binary done is awkward in pure continuous noise).

**Key methods**
- `nll_loss` — training
- `sample_next` / `sample_next_with_reward` — imagination
- `sample_next_multi` — for MC-UBE (M samples)

### `udwm/models/reward_term.py`
Separate reward + done heads (DIAMOND-style baseline when joint is off).

### `udwm/models/consistency.py` — speed path
- `ConsistencyStudent`: one-step \(x_0\) predictor.
- `distill_loss`: student matches teacher’s reconstructed \(x_0\).
- `DistilledWorldModel`: trains teacher + student; **rollouts use student (1 NFE)**.
- Adapter makes student look like diffusion dynamics for MC-UBE.

### `udwm/models/world_model.py` — unified API
`WorldModel.build(...)` constructs:
- Gaussian, or
- Diffusion + separate R/T, or
- Diffusion joint R, or
- DistilledWorldModel if `use_consistency_distill`.

Methods:
- `train_loss(obs,act,next,rew,done)`
- `rollout(obs, policy_fn, horizon)` → imagined trajectories

---

## 4.4 Uncertainty (the research heart)

### `udwm/uncertainty/mc_ube.py`

**`MCUBELocalRewards.estimate(world_model, q_fn, policy_fn, obs, actions)`**
For each ensemble member:
1. Sample \(M\) next states from that member’s dynamics.
2. Sample next actions from policy.
3. Evaluate min-Q (or Q) → mean & var over samples.
4. Across members: \(\hat w=\mathrm{Var}_i(\mu_i)\), \(\hat g=\mathrm{mean}_i(v_i)\), \(\hat u=\mathrm{clip}(\hat w-\hat g)\).

**`UNetwork`**
Neural net \(U(s,a)\ge 0\) (softplus). Learns multi-step integrated uncertainty via TD:
\[
z=\gamma^2 \hat u + \gamma^2 (1-d)\,U(s',a').
\]

**`ube_loss`**
MSE to targets + small regularizer on raw pre-softplus.

### `udwm/uncertainty/calibration.py`
Compares \(\sqrt{U}\) to absolute TD residuals (rough calibration diagnostic).

**Theory toys (no nets)**
- `theory/toy_ube_mdp.py` — exact UBE = Var(V) on a tiny DAG.
- `theory/toy_mc_ube_estimator.py` — MC vs closed form for Gaussian ensembles.

---

## 4.5 RL agent

### `udwm/rl/sac.py` — Soft Actor-Critic
- Squashed Gaussian actor.
- Twin critics + targets.
- Automatic entropy temperature \(\alpha\).
- If `use_ube` and \(\lambda\neq 0\):  
  target / actor objective uses \(Q + \lambda\sqrt{U}\).

### `udwm/rl/trainer.py` — MBPOTrainer
The main loop wiring everything:
1. Env step → real buffer  
2. Train WM → imagine → model buffer  
3. SAC updates on mixed batches  
4. MC-UBE update  
5. Periodic `evaluate_full` (return, s/r MSE, U calibration)

---

## 4.6 Eval & scripts

| Script | What it does |
|---|---|
| `python -m udwm.scripts.smoke_test` | Fast check all modules |
| `python -m udwm.scripts.train_mbpo --config ...` | Train |
| `python -m udwm.scripts.run_ablations` | Compare variants → CSV/JSON |
| `python -m udwm.scripts.benchmark_throughput` | Samples/sec vs steps |
| `python -m udwm.scripts.plot_results` | Curves/bars from JSON |
| `python -m udwm.scripts.evaluate --checkpoint ...` | Load & score |
| `python theory/toy_ube_mdp.py` | Theory identity check |

### `udwm/eval/metrics.py`
- Policy return  
- One-step world-model MSE  
- U vs TD residual correlation  
- Throughput benchmark  

### `udwm/envs/registry.py`
- `make_env(id)`, `space_info`  
- Notes for Pendulum / MountainCarContinuous / optional dm_control via shimmy  

---

# 5. How theory maps to code (cheat table)

| Math object | Code |
|---|---|
| Ensemble \(\{\theta_i\}\) | `ensemble_size` members in Gaussian/Diffusion |
| \(p_{\theta_i}(s'\mid s,a)\) | `sample_next(..., member=i)` |
| \(\bar Q\) | `SACAgent.q_min` |
| \(\hat w,\hat g,\hat u\) | `MCUBELocalRewards.estimate` |
| \(U(s,a)\) | `UNetwork` |
| UBE backup | `ube_targets` + `ube_loss` |
| \(\gamma^2\)-contraction intuition | smaller local error → better \(U\) (Cor A2) |
| Distilled sampler | `ConsistencyStudent` / `DistilledWorldModel` |
| Joint \(p(s',r\mid s,a)\) | `joint_reward=True` in diffusion |
| Policy optimism/pessimism | `optimism_lambda` in SAC |

---

# 6. Suggested reading order (1–2 days)

### Day 1 — intuition + math
1. This file §0–§1  
2. `docs/PROFESSOR_BRIEFING.md` Part A  
3. Run `python theory/toy_ube_mdp.py` and stare at the numbers  
4. Skim `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` §1–§6  

### Day 2 — code
1. `configs/default.yaml`  
2. `world_model.py` + `gaussian_ensemble.py`  
3. `diffusion_dynamics.py` (train + sample)  
4. `mc_ube.py`  
5. `sac.py` + `trainer.py` loop  
6. Run `python -m udwm.scripts.smoke_test`  
7. Run short train: `python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml`  

### Day 3 — research narrative
1. `papers/LITERATURE_SURVEY.md`  
2. `papers/PAPER_DRAFT.md`  
3. Hand-check `research/proofs/luis-theorem1-checklist.md`  

---

# 7. Common confusions

**Q: Is \(U\) the same as ensemble variance of Q?**  
A: Not exactly. Naive ensemble-var of Q is a baseline (`ensemble-var` in Luis). UBE *propagates* local uncertainty through the mean dynamics so multi-step epistemic effects accumulate.

**Q: Why \(\gamma^2\) not \(\gamma\)?**  
A: Variance of a discounted sum scales with \(\gamma^2\) (roughly: \(\mathrm{Var}(\gamma X)=\gamma^2\mathrm{Var}(X)\)).

**Q: Why can \(u\) be negative in theory?**  
A: Luis allows it; practice clips \(u_{\min}\ge 0\) so \(U\) stays interpretable as variance-like.

**Q: Why joint reward still has a separate done head?**  
A: Done is binary; pure Gaussian noise diffusion on a binary bit is awkward. Research can later use discrete diffusion / classifier heads.

**Q: Does consistency student destroy uncertainty?**  
A: It can add **bias** (estimand becomes “student-induced \(w\)”, not teacher \(w\)). That’s Phase B theory. Empirically we still keep an ensemble of students.

---

# 8. How to extend the code (recipes)

### Add a new env
```yaml
env:
  id: MountainCarContinuous-v0
```
Must be 1D Box obs/act. For dm_control: install `shimmy[dm-control]`.

### Ablation without UBE
```bash
python -m udwm.scripts.train_mbpo --no-ube
```

### Joint reward
```bash
python -m udwm.scripts.train_mbpo --config configs/joint_reward.yaml
```

### Consistency distill
```bash
python -m udwm.scripts.train_mbpo --config configs/consistency_distill.yaml
```

### Plot ablations
```bash
python -m udwm.scripts.run_ablations --steps 2000
python -m udwm.scripts.plot_results --ablation runs/ablations/ablation_results.json
```

### Gap 2 (future)
1. History-conditioned denoiser inputs \((\tau_i, y)\) like Wang et al.  
2. Bound \(\lvert V(s)-V(\hat s)\rvert\le L\|s-\hat s\|\).  
3. Feed reconstruction disagreement into local UBE terms.

---

# 9. What “done” looks like for a paper

**Theory**
- [x] Framing: estimator not new recursion  
- [x] Toy exact UBE identity  
- [ ] Fully written Theorem A1 proof with constants  
- [ ] Distillation bias theorem B1  

**Systems**
- [x] Gaussian + diffusion + joint R + distill hooks  
- [x] MBPO+SAC+UBE  
- [x] Ablations + throughput + plots  
- [ ] Strong multi-seed continuous-control tables (DMC)  

**Writing**
- [x] Literature survey  
- [x] Paper draft skeleton  
- [x] This guide + professor briefing  
- [ ] Polished related work + figures for submission  

---

# 10. Glossary

| Term | Meaning |
|---|---|
| WM | World model (learned simulator) |
| MBPO | Model-Based Policy Optimization |
| SAC | Soft Actor-Critic |
| UBE | Uncertainty Bellman Equation |
| NFE | Neural function evaluations (denoising steps) |
| DDIM | Fast deterministic-ish diffusion sampler |
| Epistemic | Uncertainty due to limited knowledge |
| Aleatoric | Inherent randomness |
| Ensemble | Set of models; disagreement ≈ epistemic |
| Imagination | Rolling out the WM to generate synthetic experience |

---

*If you only maintain one personal study doc, maintain this one. Update §9 checkboxes as you finish items.*
