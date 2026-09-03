# Assumptions Log (living document)

Every assumption used in derivations or implementations. Update when proofs or code change.

Last updated: 2026-09-03

Active claim: distillation vs ordinary member matching, measured by teacher–student
\(u,w\) rank correlation and RMSE at fixed \(M\) and NFE.

---

## From Luis et al. 2023 (inherited)

| ID | Assumption | Used for | Status for diffusion project |
|---|---|---|---|
| L1 | Independent transitions across distinct states | Theorem 1 proof (uncorrelated \(V(s')\) and \(p(\cdot\mid s,a)\)) | Inherited; violated by shared NN parameters — same caveat as Luis deep RL |
| L2 | Acyclic MDP (or time-augmented) | Theorem 1; no revisit correlations | Inherited; practical envs may cycle — Luis clips \(u_{\min}\) |
| L3 | Known bounded rewards (theory); unknown OK with extension B.1 | Value definition | OK |
| L4 | Discount \(\gamma \in [0,1)\) | Contraction of UBE operator | OK |
| L5 | \(\Phi_t\) is a posterior (or discrete ensemble treated as one) | Definition of \(U_t = \mathrm{Var}_{p\sim\Phi_t}[V^{\pi,p}]\) | We use ensemble-as-posterior (standard, contested Bayesian status) |

**Not assumed by Theorem 1:** Gaussian transitions, Gaussian \(V\), finite \(|\mathcal{S}|\) for the equality statement beyond the tabular setup, density evaluation of \(p\).

---

## Introduced by this work (estimator theory)

| ID | Assumption | Used for | Risk if violated |
|---|---|---|---|
| A1 | Bounded next-values \(\lvert \bar Q\rvert \le Q_{\max}\) | Hoeffding / McDiarmid for MC \(\hat w,\hat u\) | Use reward bounds + \(\gamma\) to enforce |
| A2 | i.i.d. samples from each \(p_{\theta_i}\) given \((s,a)\) | Concentration of \(\mu_i, v_i\); **and the \(O(1/M)\) bias constant** | Shared RNG across members correlates \(e_i\) and *changes the bias constant* — do not adopt common random numbers without re-deriving (see `ESTIMATOR-BIAS-FINDING.md` §8) |
| A3 | Exact sampler: law of `SampleDiffusion(θ_i; K)` equals \(p_{\theta_i}\) | Theorem A1 unbiasedness for \(w^\star\) under true kernels | Fails under distillation → Theorem B1 |
| A4 | Lipschitz / continuous mapping from member means to sample variance | Transfer mean concentration → \(\hat w\) concentration | True for finite \(N\), bounded \(Q\) |
| A5 | Discrete uniform \(\Phi_t = \mathrm{Unif}\{\theta_i\}\) | Matches Luis deep RL modeling | Continuous posterior over scores left as future work |
| A6 | **Debias applied:** \(\hat u\) uses the closed-form \(O(1/M)\) correction | Unbiasedness of \(\hat u\) for \(u^\star\) at finite \(M\) | Without it \(\hat u\) is inflated by \(g^\star(2N-1)/(NM)\) — at default \(M=8,N=5\) that is 22.5% of the aleatoric term, routinely comparable to \(u^\star\) itself. Requires \(M\ge2\). |
| A7 | Member means carry MC noise \(c^2\sigma_i^2/M\) with **known** \(c\) | Sizing the debias correction (\(c^2\) enters linearly) | \(c=1\) for diffusion (pure MC mean), \(c=0\) for the Gaussian closed-form plug-in. A \((1-\beta)\)-plug-in/\(\beta\)-MC blend has \(c=\beta\): treating this as a boolean over- or under-subtracts by \(1-\beta^2\). `adaptive_mc.gaussian_mean_blend` defaults to 0 so the Gaussian arm matches `mc_ube.estimate`. |
| A8 | Refinement selection is **independent** of the reported estimate | Adaptive-\(M\) ranking claims (novelty B) | **Satisfied by default since 2026-08-19.** `adaptive_mc.sample_split=true` reserves the probe for selection and reports unrefined states on fresh samples. Disabling it is an explicitly biased low-cost mode and must not be used for calibration claims. |

---

## Distillation extension (Phase B)

| ID | Assumption | Used for |
|---|---|---|
| B1 | Sampler error measurable in TV or Wasserstein-\(p\) | Bound \(\lvert w^{(K)} - w^{(\infty)}\rvert\) |
| B2 | \(\bar Q\) Lipschitz in state (or bounded differences) | Convert state-law error → value error |
| B3 | Same ensemble weights under teacher and student | Compare \(w\) under paired models |

---

## Implementation assumptions (not theory)

| ID | Choice | Note |
|---|---|---|
| I1 | Clip \(\hat u \leftarrow \max(u_{\min}, \hat u)\) with \(u_{\min}=0\) | Luis practical bound; yields \(\tilde U\) between exact \(U\) and Zhou \(W\) when theory holds |
| I2 | \(J=1\) action sample for inner \(\pi\) expectation | Same as Luis D.1 for \(w_t\) |
| I3 | U-net + TD on model rollouts | Same practical gap as Luis (no UBE existence proof with FA) |
| I4 | Image observations: apply \(\bar Q\) in latent or pixel space consistently with policy | Dimensionality handled by critic, not full pixel covariance |

---

## Edge cases to break-test theorems

1. Deterministic env + deterministic policy → \(g_t \to 0\), \(u_t = w_t\).
2. Zero epistemic uncertainty (\(N=1\) fixed model) → \(w_t \to 0\), \(u_t \to 0\).
3. Extreme distillation error (\(K=1\) collapsed sampler) → large B1 bias; \(\hat U\) should not be trusted as teacher-UBE.
4. Negative population \(u_t\) (Luis toy MRP) → clipping introduces known overestimation. **Compounded by the \(O(1/M)\) MC bias, which can push a genuinely negative \(u^\star\) positive so the clamp never fires and the failure is silent.**
5. Highly multi-modal \(p(s'\mid s,a)\) with small \(M\) → high variance of \(\hat u\); Bernstein may beat Hoeffding. **Correction: the dominant small-\(M\) error is a *bias*, not variance, and it is distribution-free — driven by the magnitude of \(\sigma_i^2\), not by multimodality. The bimodal check in `theory/estimator_bias.py` Part 1 HAS been run (2026-08-29, re-run 2026-09-03): the Gaussian and the symmetric-bimodal law reproduce the same \(O(1/M)\) bias constants to Monte Carlo precision, so "distribution-free" is confirmed for the first moment. The second moment is NOT distribution-free — see `ESTIMATOR-VARIANCE-FINDING.md` (kurtosis; the symmetric-bimodal control with \(\kappa=2.5\) is the benign case).**
