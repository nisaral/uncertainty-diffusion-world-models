# Novelty landscape — what already exists vs what is still ours

**Date:** 2026-08. Re-check arXiv every 3–4 weeks.  
**Purpose:** Do not overclaim. Faculty will know MACURA.

---

## Pattern that looks like us (does **not** refute, but shrinks A)

Several papers already **stop or shorten model rollouts when the model looks uncertain**:

| Paper | What they use to gate rollouts | Not the same as us |
|---|---|---|
| **MACURA** (Frauenknecht et al., 2024, arXiv:2405.19014) — *Trust the Model Where It Trusts Itself* | One-step / inherent **model** uncertainty → adapt rollout length | Score is **not** Luis UBE multi-step \(\mathrm{Var}(V)\); not diffusion MC estimators; no selective over-rejection / audit |
| **Nguyen et al.** adaptive rollout | Compound **dynamics error** vs value-function error | Heuristic error estimate, not UBE \(U\) |
| **Kalweit & Boedecker 2017** | Use imagined data only where uncertainty is low | Early, not generative WM, not UBE |
| **MOPO** (Yu et al. 2020) | Penalize *one-step* ensemble variance in the reward | Penalty on \(r\), not stop/weight of imagination + not Bellman \(U\) |

**Implication:** “We stop imagination when uncertainty is high” is **not** a new idea. Do **not** pitch A as “first uncertainty-aware rollout.”

**What is still a real difference (if we stay honest):**

1. The **score** is **Bellman-propagated epistemic value variance** (UBE \(U\)), estimated by **MC** because members are **implicit diffusion** models — not one-step \(\|\Sigma\|\) of a Gaussian ensemble.  
2. We treat \(\sqrt{U}\) as a **decision score** and **measure** ranking vs threshold (over-rejection), in the same language as DIO / Panov selective prediction. We do **not** claim conformal coverage.  
3. **Adaptive MC budget** (B) — spend \(M,K\) only where ensemble disagreement is high — is closer to an estimator-systems contribution than MACURA’s horizon schedule.  
4. **Audit log** of refusals (replay why a dream was not allowed to train) is the Lukas/oversight angle, not in MACURA.

---

## What would actually refute us

| If this exists | Effect |
|---|---|
| Paper that does **exact/MC UBE local rewards for diffusion ensembles** + uses \(U\) to gate imagination | Direct collision — read it and pivot (estimator theory or selective metrics only) |
| Proof that **ensemble-of-diffusions is not a valid \(\Phi_t\)** so UBE equality is meaningless | Weakens Claim A; we already treat ensemble as approximation (same as Luis deep RL) |
| Result that **few-step / distilled samples make \(\hat u\) so biased** that \(U\) ranks worse than random | Threatens A+B empirically — we must measure rank-corr; if it fails, say so |
| MACURA-style method **strictly dominates** U-gate when both use the same diffusion ensemble | Then A is only a reimplementation; keep B + selective + diffusion-MC-UBE |

**As of this search:** no mainstream paper titled or abstract-matched “UBE + diffusion world model + MC local rewards.” That hole still looks open. Adaptive *rollout length* is crowded; adaptive *MC sample budget for UBE* is not.

---

## How to write the related-work paragraph

> Uncertainty-aware rollouts are established: Kalweit & Boedecker restrict imagined data by uncertainty; Nguyen et al. and MACURA (Frauenknecht et al., 2024) adapt horizon using one-step model disagreement or error estimates. We do not claim to invent gated imagination. We change the **object that is gated**: multi-step posterior value variance under Luis’s UBE, with Monte Carlo local rewards because the dynamics are implicit (diffusion / distilled) rather than Gaussian. We additionally report selective-prediction diagnostics (risk–coverage, over-rejection) when \(\sqrt{U}\) is treated as a decision threshold, and an audit trail of refusals.

---

## Next empirical test this implies

If MACURA-style **one-step ensemble variance** already gates as well as UBE \(\sqrt{U}\) on Pendulum, A is weak. So the ablation must include, later:

- `gate_on_onestep_ensemble_var` vs `gate_on_UBE_U`

That comparison is more important than another env. Not implemented yet — scheduled after the A×B factorial.
