# Uncertainty-preserving distillation for diffusion world models

Open research code for a **conditional, falsifiable** claim:

> Decision-aware distillation can preserve downstream epistemic uncertainty
> under a **fixed or separately validated** value map. Naively coupling the
> same loss to an **online learned critic** is unsafe and can systematically
> destroy that uncertainty, even while improving next-state accuracy.

The stack is a small MBPO / SAC laboratory: a multi-step diffusion teacher
ensemble, a one-step student, Monte Carlo UBE local rewards, and
teacher–student uncertainty metrics. It is **not** a new world model, **not**
conformal coverage, and **not** a SOTA control claim.

**Repo:** https://github.com/nisaral/uncertainty-diffusion-world-models

---

## Results (current)

Two matched studies, both with **exact teacher checksum pairing** (gap 0).

**1. Controlled prediction study** — 20 seeds, five-member ensembles, eight
paired latents, 1,024-state held-out grid
([`research/RESULTS-STRESS-LARGE-2026-08-21.md`](research/RESULTS-STRESS-LARGE-2026-08-21.md)).

Hybrid decision-aware vs ordinary distillation, **fixed** value map:

| Endpoint | Hybrid vs ordinary |
|---|---|
| Uncertainty RMSE | better on **20/20** seeds; mean Δ −0.00374; bootstrap 95% [−0.00474, −0.00289] |
| Rank correlation | better on **13/20**; mean Δ +0.0317; 95% [+0.00117, +0.0632] |
| Top-decile recall | **inconclusive** (9/20; interval includes 0) |

This supports the **narrow** claim: with a correctly specified, stationary
decision map, hybrid distillation improves uncertainty-**magnitude** fidelity.

**2. Online policy study** — 10 seeds × 1,800 env steps
([`research/RESULTS-POLICY-SCALE-2026-08-22.md`](research/RESULTS-POLICY-SCALE-2026-08-22.md)).

Same hybrid loss, but the value map is the **live SAC critic**:

| Endpoint | Hybrid vs ordinary |
|---|---|
| Next-state MSE | better on **10/10** |
| Uncertainty rank correlation | **worse on 10/10**; mean Δ −0.2968 |
| Uncertainty RMSE | **worse on 10/10**; mean Δ +180.4 |
| Return | better on only 3/10 |

The broader transfer claim is **falsified**: coupling uncertainty-preserving
distillation to an online critic can collapse teacher–student uncertainty
while helping state prediction.

That negative is the sharper thesis, not a lab failure.

---

## What comes next (implemented, not yet the 10-seed table)

The correction is **lagged target-critic distillation**:

1. Stop-gradient **SAC target critic** as the value map (not the online Q).
2. **Normalize** teacher value targets (location/scale from the teacher batch).
3. **Guard:** drop decision terms on a minibatch if student uncertainty is
   anti-aligned or scale-exploded relative to the teacher.
4. **Warmup:** do not use Q as a distillation target until the critic has
   been updated.

```bash
python -m udwm.scripts.train_mbpo --config configs/lagged_target_distill.yaml

# Matched policy arms, including the live-Q control and the lagged correction
python -m udwm.scripts.run_delayed_bimodal_policy_ablation \
  --variants ordinary hybrid lagged_hybrid --seeds 0 1 2
```

The lagged arm is the next empirical object. Do not treat it as confirmed.

---

## Install and checks

```bash
pip install -r requirements.txt

python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py -q
python theory/toy_ube_mdp.py
```

---

## What the code implements

| Piece | Location |
|---|---|
| Geometry + decision distill losses, fidelity guard | `udwm/models/consistency.py` |
| Distilled world model + teacher freeze | `udwm/models/world_model.py` |
| Lagged target-critic value map | `udwm/rl/trainer.py` (`_make_distill_value_fn`) |
| Teacher–student \(u,w\) metrics | `udwm/eval/metrics.py` |
| Controlled stress runner | `udwm/scripts/run_decision_distillation_stress.py` |
| Policy-scale runner | `udwm/scripts/run_delayed_bimodal_policy_ablation.py` |
| MC-UBE local rewards | `udwm/uncertainty/mc_ube.py` |

One-page statement: [`papers/RESEARCH_OVERVIEW.md`](papers/RESEARCH_OVERVIEW.md).

---

## Secondary machinery (not the claim)

U-gated imagination and adaptive Monte Carlo exist because distilled
uncertainty is only useful if something *uses* it. Uncertainty-aware rollouts
already exist (MACURA and earlier). This repo does not claim to invent them.

---

## Limits

- Benchmarks are **DelayedBimodal** (decision-aware stress) and small
  continuous-control toys. No MuJoCo / DMC table.
- Luis UBE Assumptions 1–2 fail here the same way they fail in Luis’s deep-RL
  instantiation.
- \(\sqrt{U}\) is a **score**, not a conformal interval.
- Top-decile retrieval of high-uncertainty states is **not** established.

---

## Lineage

MBPO · PETS · Luis exact UBE · DIAMOND · consistency / progressive
distillation · WIMLE (latency) · MACURA (do not overclaim gated imagination)

See [`papers/LITERATURE_SURVEY.md`](papers/LITERATURE_SURVEY.md) and
[`papers/NOVELTY_LANDSCAPE.md`](papers/NOVELTY_LANDSCAPE.md).
