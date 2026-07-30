# Literature Survey: Diffusion World Models, Uncertainty, and Model-Based RL

**Project:** Uncertainty-Aware Diffusion World Models (UDWM)  
**Purpose:** Living survey for paper writing and gap justification  
**Last updated:** 2026-07  

---

## 1. How to read this survey

Clusters:

1. Model-based RL & imagination  
2. Uncertainty quantification in RL / UBE lineage  
3. Generative / diffusion world models  
4. Fast generative alternatives (distillation, IMLE, consistency)  
5. Offline / risk-sensitive MBRL  
6. Multi-agent partial observability (stretch)  

For each paper: **what it did**, **what it assumed**, **how we relate**.

---

## 2. Model-based RL and imagination

### Sutton (1991) — Dyna
- **Idea:** Learn a model; interleave real experience with simulated updates.  
- **Relation:** Ancestor of all imagination-based methods.

### Janner et al. (2019) — MBPO (ICLR)
- **Idea:** Short branched model rollouts into a model-free agent (SAC); ensemble dynamics.  
- **Assumes:** Probabilistic NN ensembles (often Gaussian).  
- **Relation:** **Our outer loop.** We swap/extend dynamics and add multi-step UBE.

### Chua et al. (2018) — PETS
- **Idea:** Probabilistic ensembles + trajectory sampling for planning.  
- **Relation:** Gaussian ensemble baseline in our code (`GaussianEnsemble`).

### Hafner et al. (2019–2023) — PlaNet / Dreamer / DreamerV3
- **Idea:** Latent RSSM world models; learn policy entirely in latent imagination.  
- **Relation:** Strong baseline class; orthogonal to image-space/state-space diffusion. We position as complementary, not a Dreamer reimplementation.

### Kaiser et al. (2019) — SimPLe
- **Idea:** Video prediction world models for Atari sample efficiency.  
- **Relation:** Early discrete WM; DIAMOND continues this line with diffusion.

---

## 3. Uncertainty Bellman Equation (UBE) lineage

### O'Donoghue et al. (2018) — UBE (ICML)
- **Idea:** Bellman recursion whose fixed point **upper-bounds** posterior variance of Q-values; used for exploration.  
- **Limit:** Can be loose.

### Zhou et al. (2020)
- **Idea:** Tighter local uncertainty \(w_t\).  
- **Limit:** Still ignores aleatoric next-value noise → overestimate.

### Luis et al. (2023) — Exact UBE (AISTATS) [arXiv:2302.12526]
- **Idea:** Under Assumps. 1–2 (independent transitions, acyclic MDP),  
  \(U_t^\pi(s)=\mathrm{Var}_{p\sim\Phi_t}[V^{\pi,p}(s)]\) obeys an **exact** UBE with local reward \(u_t=w_t-g_t\).  
- **Deep RL practice:** Ensemble of **Gaussian** NNs; App. D.1 uses mean of Gaussian as \(s'\).  
- **Relation:** **Core theory we inherit.** We do **not** re-prove the recursion; we give **MC estimators** when members are diffusion/consistency models.

### Luis et al. (2024) — QU-SAC
- **Idea:** Practical online/offline actor-critic using UBE epistemic variance.  
- **Relation:** Algorithmic template for \(Q\pm\lambda\sqrt{U}\); we target generative dynamics.

### Related UQ in RL
- Osband et al. — randomized value functions / deep exploration  
- Depeweg et al. (2018) — aleatoric vs epistemic decomposition  
- Chen et al. (2017) — ensemble variance of Q  

---

## 4. Diffusion and generative world models

### Ho et al. (2020) — DDPM
- **Idea:** Denoising diffusion probabilistic models.  
- **Relation:** Training backbone of our dynamics ensemble.

### Song et al. (2021) — Score-SDE
- **Idea:** Continuous-time reverse SDEs; score \(\nabla\log p_t\).  
- **Relation:** Language for continuous-time aleatoric analysis (Track 3 theory).

### Alonso et al. (2024) — DIAMOND (NeurIPS Spotlight) [arXiv:2405.12399]
- **Idea:** RL agent trained **entirely** in a diffusion world model (EDM-style next-frame); Atari 100k mean HNS **1.46**.  
- **Limitations (authors’ own):**  
  1. Mainly discrete control → continuous open  
  2. Frame-stack memory  
  3. **Reward/termination not inside diffusion** (separate CNN–LSTM)  
- **Relation:** Landmark application; Gap 1 cites Limitation 3; Gap 3 cites continuous extension.

### Other diffusion WMs / dynamics
- **Diffusion World Model (DWM)** variants — multistep state/reward prediction (e.g. concurrent prediction papers).  
- **PolyGRAD / diffusion dynamics** for continuous control planning.  
- **Valdi (2026)** — value diffusion world models (latent diffusion + uncertainty-aware prediction) — monitor for overlap.  
- **Conformal prediction for diffusion dynamics** (Sun et al., NeurIPS 2023) — UQ via CP, not UBE.

### Chi et al. — Diffusion Policy
- **Idea:** Diffuse over **actions**, not transitions.  
- **Relation:** Different object; MBDPO-style methods also optimize policies as diffusion.

### Cheng et al. (2026) — MBDPO
- **Idea:** Diffusion **policy optimization** inside world models (search + policy as diffusion over trajectories).  
- **Relation:** Complementary: they diffuse policies; we diffuse **dynamics** and quantify multi-step epistemic value uncertainty.

---

## 5. Fast generative models for online RL

### Song et al. (2023) — Consistency Models
- **Idea:** Self-consistency → few/one-step sampling.  
- **Relation:** Motivates our `ConsistencyStudent` distillation path.

### Progressive distillation (Salimans & Ho, etc.)
- **Idea:** Student matches teacher with fewer NFEs.  
- **Relation:** Same systems goal as our distill loss.

### Aghabozorgi et al. (2026) — WIMLE (ICLR)
- **Idea:** IMLE multi-modal world models + ensembles for continuous control; **explicitly avoids diffusion** due to iterative sampling cost.  
- **Relation:** Direct competitor setting. Our claim: diffusion **can** be viable with distillation + UBE multi-step uncertainty.

---

## 6. Offline / risk-sensitive MBRL

| Method | Uncertainty use |
|---|---|
| MOPO (Yu et al. 2020) | One-step model uncertainty penalty |
| MOReL / COMBO / RAMBO | Pessimism / conservative objectives |
| QU-SAC | Multi-step UBE epistemic term |

**Our angle:** Replace one-step penalties with \(\sqrt{U}\) from diffusion-MC-UBE when dynamics are generative.

---

## 7. Multi-agent PO (Gap 2 stretch)

### Wang et al. (AAMAS 2025) [arXiv:2410.13953]
- **Idea:** Diffusion models map local histories → global states as fixed points; Jacobian-rank deviation bounds; composite multi-agent flow.  
- **Stops at:** State reconstruction.  
- **Our Gap 2:** Lift \(\|D\|\) to value/credit uncertainty via Lipschitz / multi-agent UBE-style arguments.

### Xu et al. and earlier Dec-POMDP deep methods
- History encoders, communication, value factorization (QMIX, VDN) — context for later.

---

## 8. Gap matrix (survey synthesis)

| Capability | PETS/MBPO | Dreamer | DIAMOND | WIMLE | Luis UBE | **UDWM (us)** |
|---|---|---|---|---|---|---|
| Continuous control WM | ✓ | ✓ | △ open | ✓ | ✓ | ✓ target |
| Multi-modal generative dynamics | △ weak | latent | ✓ diffusion | ✓ IMLE | △ Gaussian | ✓ diffusion |
| Fast online rollouts | ✓ | ✓ | △ NFE cost | ✓ one-step | ✓ | ✓ distill + few-step |
| Multi-step epistemic \(U\) (UBE) | △ ad hoc | △ | ✗ | △ ensemble | ✓ Gaussian | ✓ **MC for diffusion** |
| Joint reward in generative model | often joint Gauss | yes | ✗ separate | yes | often joint | ✓ joint path |
| Multi-agent PO value lift | ✗ | ✗ | ✗ | ✗ | ✗ | future Gap 2 |

---

## 9. Search queries (re-run every 3–4 weeks)

```
uncertainty Bellman equation diffusion
diffusion world model continuous control
consistency model world model RL
WIMLE world model
DIAMOND reward termination diffusion
Dec-POMDP diffusion fixed points value uncertainty
```

---

## 10. Bib skeleton (copy into paper)

See `papers/references.bib` for BibTeX entries used in the draft.

---

## 11. Overlap risk watchlist

1. Any paper titled “UBE for diffusion world models”  
2. Joint reward-diffusion WM with strong Atari/DMC results  
3. Valdi / value-diffusion WMs with multi-step epistemic claims  

If found: differentiate on **estimator theory (concentration)** vs pure systems, or **joint R + UBE continuous** combo.
