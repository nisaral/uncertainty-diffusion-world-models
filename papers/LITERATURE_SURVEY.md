# Literature Survey: Diffusion World Models, Uncertainty, and Model-Based RL

**Project:** Uncertainty-preserving distillation for diffusion world models  
**Purpose:** Living survey. The active question is whether distillation preserves
decision-relevant epistemic uncertainty, not only transition MSE.  
**Last updated:** 2026-08-19  

> **Active claim:** test whether a one-step student keeps teacher ensemble
> disagreement (UBE \(u,w\)) under a fixed inference budget.
> Gated imagination and adaptive MC are supporting machinery.
>
> **Search status:** several sections were written without a fresh database
> pass. Re-check arXiv / Semantic Scholar before any submission. No novelty
> claim in this project is treated as search-verified.

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

## 7.5 Variance reduction, coupling, and CRN — the prior-art collision

**Status: written from memory on 2026-08-07 with WebSearch unavailable. Nothing in
this section has been checked against a database. Treat every citation as a lead to
verify, not as a verified fact, and re-read this section before writing any novelty
claim into a paper.**

This cluster exists because `research/COUPLED-MC-UBE-PROPOSAL.md` proposes sharing a
DDIM latent \(x_T\) across ensemble members. That mechanism is *old*, and the survey
must say so plainly before the proposal claims anything.

### The mechanism is common random numbers, and CRN is ~70 years old
- **Kahn & Marshall (1953); Glasserman & Yao (1992), "Some guidelines and guarantees
  for common random numbers"** — reuse the same random stream across configurations
  so that a *difference* of estimates has lower variance. Conditions under which CRN
  provably helps are known (monotone/synchronised structure).
- **Antithetic and control variates; Multilevel Monte Carlo (Giles 2008)** — MLMC in
  particular gets its entire speedup from *coupling* a fine and a coarse simulation.
  The "coupling reduces the variance of a difference" idea is textbook.
- **Reading:** the coupling *device* in the proposal is not novel and must never be
  presented as such. Any reviewer with a simulation background will name CRN in one
  line.

### CRN already exists in RL, by name
- **Ng & Jordan (2000) — PEGASUS.** Fix the simulator's random seeds to convert a
  stochastic MDP into a deterministic one, so policy value becomes a deterministic
  function of the policy parameters. This is exactly "share the latent, compare the
  models", with policies in the role our ensemble members play.
- **Reparameterisation trick (Kingma & Welling 2014; Rezende et al. 2014) and
  Stochastic Value Gradients (Heess et al. 2015).** Holding \(\epsilon\) fixed while
  varying parameters *is* coupling across models. A shared \(x_T\) across ensemble
  members is the same construction, one level up.
- **Reading:** the strongest objection to the proposal is "this is the
  reparameterisation trick applied across ensemble members." That objection is
  correct about the mechanism. The proposal survives only if the *result* — the bias
  identity — is what is new, not the sampling scheme.

### Coupled / shared-noise ensembles in UQ
- **Osband et al. (2018) — randomized prior functions**; **Wen et al. — BatchEnsemble**;
  **hyper-deep ensembles**. These share *structure* across members; I do not recall
  one that shares the *sampling noise* to reduce the bias of an ensemble-variance
  estimator, but I have low confidence here and this is the highest-risk gap in this
  section.
- **Fixed-mask MC-dropout** (consistent dropout masks across a batch) is the closest
  thing I can recall in deep UQ to deliberately correlating members' stochasticity.
- **To check first:** does any UQ paper state the bias of an ensemble-variance
  estimator under *dependent* member samples? That is the precise claim.

### Latent alignment across diffusion models
- **Song et al. (2021) — DDIM**: the reverse update is deterministic given \(x_T\),
  which is the structural fact the whole proposal leans on. Not in dispute.
- **DDIM inversion and latent editing transfer**; **Kwon et al. — h-space semantics**;
  work on the semantic structure of diffusion latent space. The *empirical* claim the
  proposal needs is stronger than any of these establish: that two independently
  trained members map the same \(x_T\) to *pointwise* similar next states. Nobody has
  shown that for small conditional dynamics MLPs, and it may simply be false here.
  `udwm/scripts/measure_coupling.py` exists to settle it on this codebase.

### What, if anything, is left
Not the coupling. The candidate contributions, in decreasing confidence:

1. **The bias identity for the UBE local reward.**
   \(\mathbb{E}[\hat w] = w^* + (g^* - \bar\Sigma)/M\) with
   \(g^* - \bar\Sigma = \frac{1}{2N^2}\sum_{ij}\mathrm{Var}(Y_i - Y_j)\): the finite-\(M\)
   bias *is* the mean pairwise disagreement variance over \(M\). I have not seen this
   stated for UBE, and it has a useful consequence — the bias is estimable from the
   same samples, so the correction needs no knowledge of the coupling.
2. **That the \(O(1/M)\) bias is proportional to \(g\)**, the exact quantity
   \(u = w - g\) exists to subtract. This is the sharper, more quotable form and does
   not need coupling at all.
3. **A matched-compute study** of whether generative dynamics buy anything over
   Gaussian ensembles once NFE is held fixed. Unglamorous, unrun, and probably the
   most useful thing in the project.

Points 1–2 are estimator-theory notes, not a paper on their own. Paired with 3 they
could be a solid workshop or a methods section. Calling any of it "groundbreaking"
before the search runs would be dishonest.

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

### Priority queries for the coupling / estimator-bias claim (§7.5)

Run these **before** any novelty claim reaches a draft. The first three decide
whether the bias identity is new; the rest map how close the nearest neighbour is.

```
bias of ensemble variance estimator finite Monte Carlo samples
unbiased estimator variance of means correlated samples ensemble
uncertainty Bellman equation local reward estimator bias
common random numbers ensemble uncertainty quantification deep learning
shared noise ensemble members variance reduction epistemic uncertainty
coupled sampling diffusion models compare two models same latent
PEGASUS common random numbers policy evaluation
epistemic aleatoric decomposition Monte Carlo bias finite samples
DDIM same initial noise different checkpoints similar output
multilevel Monte Carlo coupling reinforcement learning value estimation
```

---

## 10. Bib skeleton (copy into paper)

See `papers/references.bib` for BibTeX entries used in the draft.

---

## 11. Overlap risk watchlist

1. Any paper titled “UBE for diffusion world models”  
2. Joint reward-diffusion WM with strong Atari/DMC results  
3. Valdi / value-diffusion WMs with multi-step epistemic claims  
4. **Any statement of the finite-sample bias of an ensemble-variance estimator** —
   in UQ, simulation, or statistics. This is the closest thing to our claimed
   contribution and the search has not been run (see §7.5). A hit here does the most
   damage of anything on this list.
5. **Shared-noise or CRN ensembles for uncertainty estimation** in deep learning.
6. Any empirical demonstration that independently trained diffusion models map a
   shared \(x_T\) to pointwise-similar outputs — this would be prior art for the
   enabling assumption, and would also be *good news* for the proposal's validity.

If found: differentiate on **estimator theory (concentration)** vs pure systems, or **joint R + UBE continuous** combo.

If item 4 is found stated in general form, the honest move is to cite it and demote
our contribution to "instantiating a known bias result for the UBE local reward, plus
the disagreement-variance identity and the matched-compute study" — not to hunt for a
narrower framing in which we are still first.
