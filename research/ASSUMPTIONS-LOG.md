# Assumptions Log (living document)

Every assumption used in derivations or implementations. Update when proofs or code change.

Last updated: 2026-07-18

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
| A2 | i.i.d. samples from each \(p_{\theta_i}\) given \((s,a)\) | Concentration of \(\mu_i, v_i\) | Shared RNG across members OK if accounted for (reduces variance) |
| A3 | Exact sampler: law of `SampleDiffusion(θ_i; K)` equals \(p_{\theta_i}\) | Theorem A1 unbiasedness for \(w^\star\) under true kernels | Fails under distillation → Theorem B1 |
| A4 | Lipschitz / continuous mapping from member means to sample variance | Transfer mean concentration → \(\hat w\) concentration | True for finite \(N\), bounded \(Q\) |
| A5 | Discrete uniform \(\Phi_t = \mathrm{Unif}\{\theta_i\}\) | Matches Luis deep RL modeling | Continuous posterior over scores left as future work |

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
4. Negative population \(u_t\) (Luis toy MRP) → clipping introduces known overestimation.
5. Highly multi-modal \(p(s'\mid s,a)\) with small \(M\) → high variance of \(\hat u\); Bernstein may beat Hoeffding.
