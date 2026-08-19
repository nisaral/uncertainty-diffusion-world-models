# Uncertainty-preserving distillation for diffusion world models

Open research code for a **falsifiable** question:

> Can diffusion world-model distillation preserve **decision-relevant epistemic
> uncertainty**, rather than only average transition accuracy, under a **fixed
> inference budget**?

The stack is a small MBPO / SAC laboratory on Pendulum: a multi-step diffusion
teacher ensemble, a one-step consistency-style student, Monte Carlo UBE local
rewards, and teacher–student uncertainty metrics. It is **not** a new world
model, **not** a conformal coverage method, and **not** a claim of SOTA control.

**Repo:** https://github.com/nisaral/uncertainty-diffusion-world-models

---

## Why this question

Ordinary distillation matches next-state samples. Policies that use ensemble
disagreement — local UBE rewards, pessimism, or “do not train on this dream” —
depend on a different object: **how members disagree after a value map**.

A student can keep next-state MSE and still scramble that disagreement. If it
does, cheap 1-NFE rollouts are not a drop-in replacement for the teacher’s
uncertainty. The experiment asks whether an extra **ensemble-geometry** loss
reduces that scramble.

One-page statement: [`papers/RESEARCH_OVERVIEW.md`](papers/RESEARCH_OVERVIEW.md).

---

## Install and checks

```bash
pip install -r requirements.txt

python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py -q
python theory/toy_ube_mdp.py
```

---

## Primary experiment

Matched compute. Same data, teacher budget, student budget, architecture, and
estimator \(M\). The only changed factor is whether the student is trained with
ordinary member matching or with extra mean / centered-geometry / pairwise
disagreement terms.

```bash
# Ordinary student (member matching only)
python -m udwm.scripts.train_mbpo --config configs/consistency_distill_baseline.yaml

# Geometry-preserving student
python -m udwm.scripts.train_mbpo --config configs/consistency_distill.yaml

# Teacher vs student UBE quantities on a checkpoint
python -m udwm.scripts.evaluate_distillation_uncertainty --config configs/consistency_distill.yaml

# Preferred: two-stage freeze-teacher ablation + honest summary
python -m udwm.scripts.run_distillation_ablation --seeds 0 1 2
```

**Success (supports the hypothesis):** higher rank correlation of teacher vs
student local \(u\) / \(w\), and lower RMSE, at matched next-state error and
matched NFE.

**Failure (also a result):** geometry terms do not preserve rankings, or they
preserve rankings only by harming transition accuracy. Report that.

---

## What the code implements

| Piece | Location |
|---|---|
| Geometry-preserving distill loss | `udwm/models/consistency.py` |
| Distilled world model + teacher freeze | `udwm/models/world_model.py` |
| Teacher–student \(u,w\) metrics | `udwm/eval/metrics.py` |
| Ordinary vs geometry runner | `udwm/scripts/run_distillation_ablation.py` |
| MC-UBE local rewards | `udwm/uncertainty/mc_ube.py` |
| Diffusion / Gaussian ensembles | `udwm/models/diffusion_dynamics.py`, `gaussian_ensemble.py` |
| MBPO + SAC | `udwm/rl/trainer.py`, `udwm/rl/sac.py` |
| Exact UBE toy MRP | `theory/toy_ube_mdp.py` |

The student is a **practical one-step distillation** of a frozen (or jointly
trained) diffusion teacher, not a full Song et al. continuous-time consistency
model.

---

## Secondary machinery (not the claim)

U-gated imagination and adaptive Monte Carlo live in the same repo because
distilled uncertainty is only useful if something *uses* it. They are **not**
pitched as a new algorithm.

Uncertainty-aware rollouts already exist (Kalweit & Boedecker; Nguyen et al.;
MACURA). This project does not claim to invent gated imagination. The gate
here uses Bellman-propagated value uncertainty estimated from implicit
samplers. That comparison is a control, not the headline.

```bash
python -m udwm.scripts.train_mbpo --config configs/ab_novel.yaml
python -m udwm.scripts.eval_selective --config configs/ab_novel.yaml
```

---

## Current scale and limits

- Environment: **Pendulum-v1**. No MuJoCo / DMC results yet.
- Luis UBE Assumptions 1–2 (independent transitions, acyclic MDP) fail here
  the same way they fail in Luis’s deep-RL instantiation.
- \(\sqrt{U}\) is treated as a **score**. There is **no** conformal coverage
  claim.
- Ensemble-as-posterior is the same modelling choice as PETS / Luis, not a
  new Bayesian justification.

---

## Layout

```
udwm/       package
configs/    YAML experiments
theory/     math toys (no neural nets)
papers/     overview, literature notes, working draft
research/   assumptions and proof sketches
tests/      unit tests
```

Personal study notes stay on the author’s machine (`private_notes/`, gitignored).

---

## Lineage

MBPO · PETS · Luis exact UBE · DIAMOND · consistency models / progressive
distillation · WIMLE (latency critique) · MACURA (do not overclaim gated
imagination)

See [`papers/LITERATURE_SURVEY.md`](papers/LITERATURE_SURVEY.md) and
[`papers/NOVELTY_LANDSCAPE.md`](papers/NOVELTY_LANDSCAPE.md).
