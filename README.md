# Uncertainty-preserving distillation for diffusion world models

Open research code for a **falsifiable** claim about the *loss*, not only the critic:

> Matching teacher–student value disagreement at a **single shared latent**
> does not identify epistemic uncertainty. The matched statistic is
> \(w^\star+(g^\star-\bar\Sigma)\): one scalar, two unknowns. The student can
> keep that number by trading ensemble disagreement for aleatoric noise.
> Whether that trade *helps* or *destroys* \(w^\star\) is a property of the
> value map. The **freedom** is a property of the loss.

The two headline tables (fixed map vs live critic) are **one mechanism
measured twice**, not a positive result held in tension with a negative one.

The stack is a small MBPO / SAC laboratory: diffusion teacher ensemble,
one-step student, Monte Carlo UBE local rewards. It is **not** a new world
model, **not** conformal coverage, and **not** a SOTA control claim.

**Repo:** https://github.com/nisaral/uncertainty-diffusion-world-models

Theory: [`theory/distill_identifiability.py`](theory/distill_identifiability.py).  
Write-up: [`research/RESULTS-IDENTIFIABILITY-2026-08-29.md`](research/RESULTS-IDENTIFIABILITY-2026-08-29.md).  
Index: [`research/RESULTS.md`](research/RESULTS.md).

---

## Results

All paired studies below use **exact teacher checksums** (gap 0).

### 1. Fixed value map — 20 seeds

Protocol: five members, eight paired latents, 1,024-state grid.
[`research/RESULTS-STRESS-LARGE-2026-08-21.md`](research/RESULTS-STRESS-LARGE-2026-08-21.md),
reproduced in the identifiability rerun.

Hybrid (single-latent decision loss) vs ordinary member matching:

| Endpoint | Hybrid vs ordinary |
|---|---|
| Uncertainty RMSE | better **20/20**; mean Δ −0.00374; 95% [−0.00474, −0.00289] |
| Rank correlation | better **13/20**; mean Δ +0.0317; 95% [+0.00117, +0.0632] |
| Top-decile recall | **inconclusive** (9/20; interval includes 0) |

Diagnostics on the same run: **epistemic** fidelity (`w_deb`) improves, **aleatoric**
fidelity (`g`) degrades, on the decision arms. The loss is reallocating variance
between the two terms, not preserving the split.

### 2. Live SAC critic — 10 seeds × 1,800 steps

[`research/RESULTS-POLICY-SCALE-2026-08-22.md`](research/RESULTS-POLICY-SCALE-2026-08-22.md).
Same hybrid loss, online Q as the value map:

| Endpoint | Hybrid vs ordinary |
|---|---|
| Next-state MSE | better **10/10** |
| Uncertainty rank correlation | worse **10/10**; mean Δ −0.2968 |
| Uncertainty RMSE | worse **10/10**; mean Δ +180.4 |
| Return | better 3/10 |

The transfer claim (“drop this loss into MBPO”) is **falsified**.

### 3. Identifiability — why both tables happen

[`research/RESULTS-IDENTIFIABILITY-2026-08-29.md`](research/RESULTS-IDENTIFIABILITY-2026-08-29.md).

Exact construction: four students, including one with **zero** epistemic
disagreement, all match the teacher’s single-latent statistic to ~10⁻⁶.
Drawing \(M\ge 2\) latents and matching debiased \(\hat w\) and \(\hat g\)
**separately** breaks that family.

An ensemble analogue of the variance-shrinkage pathology in Voelcker et al.,
arXiv:2505.22772 (their collapsing variance is one model’s aleatoric spread;
ours is cross-member disagreement). Cite that paper. Nothing in the
construction needs a changing critic, so a lagged target critic is a
**different** fix (nonstationarity), not this one.

**Identified vs hybrid on the same fixed-map 20 seeds:** ranking wash
(intervals include 0); RMSE slightly *worse*; 2× sampling cost. That is what
the theory predicted: a benign stationary map does not force collapse, so this
study cannot tell the two losses apart. The discriminating experiment is the
policy 2×2, **registered, not yet reported**.

Known hole: identified loss currently weights \(\hat w\) and \(\hat g\) equally,
while \(\hat g\) is ~orders of magnitude larger, so the epistemic term is
numerically weak. Not yet reweighted.

---

## What comes next

Registered policy arms (`ordinary` / `hybrid` / `lagged_hybrid` /
`identified_hybrid` / `lagged_identified`):

1. `lagged_hybrid` should **reduce but not kill** the rank collapse (fixes
   nonstationarity, leaves identifiability).
2. `identified_hybrid` on the **live** critic is the sharp test: if rank
   survives, the collapse was identifiability, not nonstationarity.
3. `lagged_identified` should be best or tied. If lag adds nothing once the
   target is identified, say so.

```bash
python -m udwm.scripts.run_delayed_bimodal_policy_ablation \
  --variants ordinary hybrid lagged_hybrid identified_hybrid lagged_identified \
  --seeds 0 1 2
```

Until that table exists, do not treat identified distillation as confirmed in
policy.

---

## Install and checks

```bash
pip install -r requirements.txt

python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py -q
python theory/toy_ube_mdp.py
python theory/distill_identifiability.py
python theory/estimator_variance.py
```

---

## What the code implements

| Piece | Location |
|---|---|
| Ordinary / geometry / hybrid / **identified** distill losses, guard | `udwm/models/consistency.py` |
| Distilled world model + teacher freeze | `udwm/models/world_model.py` |
| Lagged target-critic value map | `udwm/rl/trainer.py` |
| Teacher–student \(u,w,w_{\mathrm{deb}},g\) metrics | `udwm/eval/metrics.py` |
| Controlled stress runner | `udwm/scripts/run_decision_distillation_stress.py` |
| Policy 2×2 runner | `udwm/scripts/run_delayed_bimodal_policy_ablation.py` |
| Identifiability toy (zero-loss family) | `theory/distill_identifiability.py` |
| MC-UBE bias / variance | `theory/estimator_bias.py`, `theory/estimator_variance.py` |

One-page statement: [`papers/RESEARCH_OVERVIEW.md`](papers/RESEARCH_OVERVIEW.md).

---

## Secondary machinery (not the claim)

U-gated imagination and adaptive Monte Carlo exist because distilled
uncertainty is only useful if something *uses* it. Uncertainty-aware rollouts
already exist (MACURA and earlier). This repo does not claim to invent them.

---

## Limits

- DelayedBimodal / small continuous-control toys. No MuJoCo / DMC table.
- Luis UBE Assumptions 1–2 fail here as in Luis’s deep-RL instantiation.
- \(\sqrt{U}\) is a **score**, not a conformal interval.
- Top-decile retrieval of high-uncertainty states is **not** established.
- Policy 2×2 for identified vs hybrid is **not** in yet.

---

## Lineage

MBPO · PETS · Luis exact UBE · DIAMOND · consistency / progressive
distillation · WIMLE (latency) · MACURA (do not overclaim gated imagination)
· Voelcker et al. 2025 (calibrated value-aware model learning)

See [`papers/LITERATURE_SURVEY.md`](papers/LITERATURE_SURVEY.md) and
[`papers/NOVELTY_LANDSCAPE.md`](papers/NOVELTY_LANDSCAPE.md).
