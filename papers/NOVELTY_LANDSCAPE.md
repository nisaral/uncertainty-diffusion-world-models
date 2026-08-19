# Novelty landscape — what already exists vs what is still a question

**Date:** 2026-08-19. Re-check arXiv every 3–4 weeks.  
**Purpose:** Do not overclaim. Faculty will know MACURA and distillation.

---

## Active question (the only thing we pitch)

Does **distillation** of a diffusion world-model ensemble preserve **decision-relevant
epistemic disagreement** (local UBE \(u,w\)), or only average next-state accuracy,
at matched NFE?

This is an empirical / estimator question. It is **not** “we invented gated
imagination” and **not** “we invented UBE.”

Related work that *touches* this without answering it:

| Paper / line | What it does | What it does not measure |
|---|---|---|
| Consistency models / progressive distillation | Match samples or scores in few steps | Teacher vs student **ensemble disagreement after a value map** |
| WIMLE | Avoid diffusion because of latency | Uncertainty preservation of a distilled student |
| DIAMOND | Strong diffusion WM for imagination | Multi-step epistemic value variance under distillation |
| Luis UBE / QU-SAC | Exact recursion + Gaussian deep-RL estimators | Implicit diffusion members; distilled students |
| Knowledge distillation + uncertainty (classification) | Sometimes match predictive entropy | Not MBRL imagination; not UBE local rewards |

**What would refute the question as a contribution:** a paper that already
compares teacher vs student UBE (or equivalent decision scores) for diffusion
WM distillation under matched compute. As of the last search, that comparison
was not the stated object of the WM distillation papers we read.

---

## Pattern that looks like “A” (does **not** refute, but shrinks gated imagination)

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

## What would actually refute the distillation claim

| If this exists | Effect |
|---|---|
| Paper that already reports teacher vs student UBE / ensemble-disagreement rankings for distilled diffusion world models at matched NFE | Direct collision — cite it and shrink to a tighter estimator or control comparison |
| Result that **ordinary** distillation already preserves \(u,w\) rankings | Geometry loss is unnecessary; report the negative |
| Result that geometry preserves rankings only by destroying next-state accuracy | Hypothesis fails as a *useful* method |
| Proof that an ensemble of independently distilled members is not a valid \(\Phi_t\) | Same caveat as Luis deep RL; state it, do not hide it |

Gated-imagination collision (MACURA + \(U\)) would refute a *different* pitch. We are not making that pitch.

---

## How to write the related-work paragraph

> Distillation of generative world models is used to cut sampling cost. We ask a different question: after distillation, does the student still rank the same states as epistemically uncertain as the teacher, as measured by Monte Carlo local UBE rewards? Sample-matching losses need not preserve ensemble geometry after a value map. Uncertainty-aware rollouts (Kalweit & Boedecker; Nguyen et al.; MACURA) are related as *users* of a score, not as the contribution. We do not claim conformal coverage.

---

## Next empirical test this implies

If MACURA-style **one-step ensemble variance** already gates as well as UBE \(\sqrt{U}\) on Pendulum, A is weak. So the ablation must include, later:

- `gate_on_onestep_ensemble_var` vs `gate_on_UBE_U`

That comparison is more important than another env. Not implemented yet — scheduled after the A×B factorial.

Update 2026-08-19: the one-step state-disagreement score is now implemented as
`udwm.uncertainty.baselines.one_step_state_disagreement` and can be selected with
`u_gate.score: one_step_state`. The comparison is still unrun.
