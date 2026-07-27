# Full Learning Roadmap: RL + World Models + Diffusion + Theoretical UQ

Depth key used throughout:
- **[USE]** — know how to use/implement it, don't need to derive
- **[DERIVE]** — should be able to derive/prove it yourself by hand, not just cite it
- **[EXTEND]** — this is the level where novel contributions live; you should be able to modify the derivation for a new assumption

---

## TRACK 1 — Math foundations

| Topic | Depth | Notes |
|---|---|---|
| Linear algebra (eigendecomposition, SVD) | [USE] | needed for reading architectures, not for the theory track |
| Multivariate calculus, gradients, Jacobians | [USE] | backprop-level, you likely have this |
| Probability: random variables, joint/conditional/marginal | [DERIVE] | do NOT skip — everything below depends on being fluent here |
| Law of total expectation / total variance | [DERIVE] | this is literally the epistemic/aleatoric split, know it cold |
| Bayesian inference: priors, likelihoods, posteriors, posterior predictive | [DERIVE] | UBE theory = Bayesian posterior over MDPs |
| Concentration inequalities (Markov, Chebyshev, Hoeffding) | [USE], target [DERIVE] for Hoeffding | needed to turn estimators into bounds |
| Stochastic processes: Markov chains, martingales | [USE], [DERIVE] for Markov property proofs | martingale concept resurfaces in some UBE proofs |
| Measure-theoretic probability | skip unless a specific proof forces it | most RL theory papers avoid heavy measure theory; don't over-invest here upfront |
| Stochastic differential equations (SDEs), Itô calculus basics | [DERIVE] (the parts used in score-based diffusion) | needed for Track 3's continuous-time diffusion view |

**How deep, concretely:** you should be able to sit down with no reference and derive the law of total variance, Bayes' rule, and prove Hoeffding's inequality from Markov's inequality. This is maybe 3-4 weeks of focused work if you're already comfortable with basic probability, longer if not.

---

## TRACK 2 — RL theory

| Topic | Depth | Notes |
|---|---|---|
| MDP formalism (S, A, P, R, γ) | [DERIVE] | |
| Bellman expectation & optimality equations | [DERIVE] | derive both, understand why optimality equation is a max over the expectation one |
| Bellman operator, contraction mapping property | [DERIVE] | prove it's a γ-contraction under sup-norm; this underpins convergence of value iteration AND is the template for how UBE proves its own recursion converges |
| Banach fixed-point theorem | [DERIVE] (statement + why it applies here) | don't need the general topology proof, just the application |
| Policy gradient theorem | [DERIVE] | |
| Actor-critic, SAC's entropy-regularized objective | [DERIVE] | derive the soft Bellman equation, not just implement SAC |
| Model-based RL: why model error compounds over horizon | [DERIVE] | this is literally MBPO's core lemma — work through their monotonic improvement bound derivation by hand |
| Regret bounds / sample complexity basics | [USE], skim [DERIVE] for one simple case (e.g., UCB regret bound for bandits first, then tabular MDP) | full gap-dependent regret theory (the arxiv papers I found) is a deep rabbit hole — don't chase full rigor here unless your novel contribution ends up needing it |
| Offline RL theory: distributional shift, pessimism principle | [DERIVE] | know why pessimistic value estimates (lower-confidence-bound style) are the standard fix for offline model exploitation — this connects directly to your Phase 6 |

**How deep:** MBPO's bound and the Bellman contraction proof are your two must-derive items — everything else in this track supports understanding those two well enough to extend them.

---

## TRACK 3 — Generative models / diffusion

| Topic | Depth | Notes |
|---|---|---|
| DDPM: forward/reverse process, ELBO derivation | [DERIVE] | derive the training objective from the variational lower bound, don't just take the "predict the noise" recipe on faith |
| DDIM: deterministic sampling | [USE], [DERIVE] the non-Markovian reformulation if time allows | |
| Score-based generative models (Song et al., SDE formulation) | [DERIVE] | this is the continuous-time view — score function = ∇ log p(x_t), reverse-time SDE. This is the mathematical language you need to formalize "aleatoric uncertainty of a diffusion process" rigorously |
| Fokker-Planck equation (basic statement, not full proof) | [USE] | shows up if you go deep into the SDE view; know what it says, don't need to derive from scratch |
| Consistency models: self-consistency property, distillation objective | [DERIVE] | prove to yourself why self-consistency + boundary condition ⇒ single-step sampling works |
| Multi-modality / mode-covering vs mode-seeking divergences (KL vs reverse-KL, IMLE's objective) | [DERIVE] | this is *why* diffusion/IMLE beat Gaussian MLPs for world models — know the divergence-level reason, not just "diffusion is better empirically" |

**How deep:** DDPM's ELBO derivation and the score-SDE formulation are the two non-negotiables. The SDE view specifically is what lets you write a clean mathematical definition of "the aleatoric uncertainty injected by the diffusion process at noise level σ(t)" — which is the object your theoretical contribution needs.

---

## TRACK 4 — Uncertainty quantification theory (this is where your gap lives)

| Topic | Depth | Notes |
|---|---|---|
| Aleatoric vs epistemic decomposition (formal, not just intuitive) | [DERIVE] | via law of total variance: Var(Y) = E[Var(Y|θ)] + Var(E[Y|θ]) — aleatoric is the first term, epistemic the second, θ = model/parameter uncertainty |
| Deep ensembles as approximate Bayesian posteriors | [USE], [DERIVE] the connection to Bayesian model averaging | know the (contested) justification for treating ensemble variance as epistemic uncertainty |
| Calibration: formal definition (not just ECE as a metric) | [DERIVE] | a model is calibrated if P(Y ∈ CI_p(x)) = p for all p — derive how ECE approximates this via binning, understand the approximation error this introduces |
| **Uncertainty Bellman Equation (UBE)** — O'Donoghue et al. 2018, Luis et al. 2023 | **[EXTEND] via estimator, not recursion** | Central object. **Revised claim (2026-07):** Theorem 1 is already distribution-agnostic (Assumps. 1–2 only). Reproduce Appendix A.1 by hand to confirm no Gaussian step. Your [EXTEND] target is a **sample-based estimator** of \(u_t/w_t\) for diffusion ensembles + concentration (Theorem A1), not a new UBE. See `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` |
| Where prior bounds (Zhou et al. 2020) were loose, and how Luis et al. tightened them | [DERIVE] | Template for *characterizing gaps* (Luis gap \(g_t\)). Your Phase B does the same move for **distillation/sampler** error on top of Luis |
| Monte Carlo / Hoeffding bounds for nested sampling of local rewards | **[DERIVE]** | First real theorem attempt: sample complexity of \(\hat u_t\) under implicit kernels; then UBE fixed-point perturbation via \(\gamma^2\)-contraction |
| Risk-aware RL / pessimism-based offline RL (QU-SAC, CBOP, COMBO, RAMBO — skim, don't derive all) | [USE] | know what they do and how they use uncertainty for pessimistic value adjustment; you don't need to derive every one of these, just understand the landscape enough to correctly position your work against them in a related-work section |

**How deep:** the UBE derivation is the single most important thing in this entire roadmap. Budget real time here — this is not a "read once" paper, it's a "reproduce every step on paper, then try changing one assumption and see what breaks" paper.

---

## Total time estimate (studying, not implementing — implementation roadmap already covers building)

- Track 1: 3-4 weeks
- Track 2: 4-5 weeks (can overlap with Phase 0 implementation)
- Track 3: 4-5 weeks (overlap with Phase 1-2 implementation)
- Track 4: 4-6 weeks, with the UBE derivation itself deserving 2+ weeks alone
- Total: roughly 15-20 weeks of study, much of it parallelizable with implementation phases from the earlier plan

---

## HOW TO FIND AND VALIDATE GAPS (the actual process, not just "read more papers")

This is a repeatable method, not a one-time search. Use it now for your current gap, and again later if you want to iterate further.

### Step 1: Find the "delta" in 2-3 recent related papers
For any paper making a theoretical claim (like Luis et al. tightening Zhou et al.'s bound), explicitly write down, in your own words:
- What assumption did the *previous* paper make that turned out to be the source of looseness/error?
- What did the *new* paper change to fix it?
- Is there a *further* assumption in the new paper that looks similarly suspicious?

**Refined application of this method (2026-07):** The first pass said “Luis assumes Gaussian ensembles.” Hand-checking Appendix A.1 showed that **Theorem 1 does not** — Gaussians are only in the **Section 4 estimator**. The gap sharpened from “extend UBE to diffusion” to “prove MC local-reward estimators for implicit generative dynamics.” Same gap-finding method; sharper claim after reading the proof, not just the abstract.

### Step 2: Check if the gap is "real" (not already quietly solved)
Before committing:
- Search arXiv/Google Scholar for combinations of your exact keywords ("uncertainty Bellman equation" + "diffusion", "distillation" + "epistemic uncertainty bound", etc.) — do this every 3-4 weeks throughout the project, not just once, since this is a live area and someone could publish the same idea while you're working
- Check recent workshop papers (ICLR/NeurIPS workshops move faster than main track and often signal where a gap is about to be filled)
- If you find something close, don't panic-abandon — read it carefully and see if your version differs in a meaningful way (different assumption, tighter bound, different distillation method, empirical vs theoretical emphasis)

### Step 3: Sanity-check the gap is theoretically tractable before committing months to it
Try a simplified toy version of your extension first — e.g., work the whole derivation through for a 2-step or 3-step toy MDP with a simple 1-dimensional diffusion process, by hand, before attempting the general case. If you can't get the toy case to work cleanly, that's a strong signal the general proof will be very hard or the bound won't be tight — better to know this in week 2 than week 12.

### Step 4: Iterative refinement once you have a first attempt
- State your conjectured bound/theorem explicitly, even before fully proving it
- Try to break it: construct adversarial or edge-case scenarios (deterministic environment, extremely high distillation error, degenerate ensembles) and check if your bound still makes sense in those limits — real theorems should degrade gracefully, not produce nonsense at edge cases
- Validate against Phase 3's empirical calibration data — if your bound predicts more/less miscalibration than you observe, figure out which of your assumptions is responsible
- Iterate: tighten the bound, relax an assumption, or honestly downgrade the claim to an empirical conjecture if the proof won't close

### Step 5: Positioning once you have something
Write the related-work section as the "gap-finding chain" itself: Zhou et al. → Luis et al.'s fix → your identified remaining assumption → your fix. This narrative structure is standard in theory papers and makes the contribution's novelty legible to a reviewer immediately, rather than making them hunt for what's actually new.

---

## Suggested weekly rhythm once you're in the theory-heavy phase
- 2-3 sessions/week dedicated purely to derivation-on-paper (no code) — Track 4 especially demands this
- 1 session/week re-searching arXiv for anything new in your exact niche (Step 2 above, done repeatedly)
- Keep a running "assumptions log" — every time you use an assumption in your derivation (Gaussian noise, bounded reward, finite horizon, etc.), write it down. This becomes both your limitations section later and your map of "which assumption could I relax next" for further iteration.
