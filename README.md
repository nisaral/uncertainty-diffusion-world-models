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
Mechanism of the identified-arm policy collapse: [`research/U-COLLAPSE-MECHANISM-2026-09-05.md`](research/U-COLLAPSE-MECHANISM-2026-09-05.md).
Leverage-fix negative (naive fix does not help): [`research/RESULTS-LEVERAGE-FIX-2026-09-05.md`](research/RESULTS-LEVERAGE-FIX-2026-09-05.md).
Paper-shaped assembly: [`research/PAPER-NARRATIVE.md`](research/PAPER-NARRATIVE.md).
Reproduction: [`REPRODUCE.md`](REPRODUCE.md).  GPU: [`RUN_ON_GPU.md`](RUN_ON_GPU.md).

---

## Results

All paired studies below use **exact teacher checksums** (gap 0).

### 1. Fixed value map — 20 seeds, extended to 50

Protocol: five members, eight paired latents, 1,024-state grid.
[`research/RESULTS-STRESS-LARGE-2026-08-21.md`](research/RESULTS-STRESS-LARGE-2026-08-21.md),
reproduced in the identifiability rerun and extended to 50 seeds
([`research/RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md`](research/RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md)
and
[`research/RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md`](research/RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md);
seeds 0–29 byte-identical to the 09-03 file).

Hybrid (single-latent decision loss) vs ordinary member matching:

| Endpoint | Hybrid vs ordinary (20 seeds) | at 30 seeds | at 50 seeds |
|---|---|---|---|
| Uncertainty RMSE (`w_rmse`) | better 20/20; Δ −0.00374 | better 29/30; Δ −0.00340 | better **49/50**; Δ −0.00283; 95% [−0.00347, −0.00223] |
| Rank correlation (`w_deb_rank_corr`) | better 14/20; Δ +0.0167; CI includes 0 | better 24/30; Δ +0.0555 | better **38/50**; Δ +0.0449; 95% [+0.0231, +0.0665] |
| Top-decile recall | **inconclusive** (9/20) | **inconclusive** (16/30) | **inconclusive** by seed-count bar (27/50 < 35/50); mean CI [+0.0012, +0.0456] just excludes 0 |

Diagnostics on the same run: **epistemic** fidelity (`w_deb`) improves, **aleatoric**
fidelity (`g`) degrades, on the decision arms. The loss is reallocating variance
between the two terms, not preserving the split. Identified vs hybrid on the
fixed map stays a ranking wash at 50 seeds (slightly worse magnitude at 2×
cost; see the 50-seed doc); the policy 2×2 is the discriminating experiment.

### 2. Live SAC critic — 10 seeds × 1,800 steps

[`research/RESULTS-POLICY-SCALE-2026-08-22.md`](research/RESULTS-POLICY-SCALE-2026-08-22.md).
Same hybrid loss, online Q as the value map:

| Endpoint | Hybrid vs ordinary |
|---|---|
| Next-state MSE | better **10/10** |
| Uncertainty rank correlation | worse **10/10**; mean Δ −0.2968 |
| Uncertainty RMSE | worse **10/10**; mean Δ +180.4 |
| Return | better 3/10 |

Confirmed at N=30 inside the policy 2×2
([`research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md`](research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md)):
u-rank 0/30 (Δ −0.310), u-RMSE worse 30/30, next-state MSE better 30/30
(Δ −0.0817), return inconclusive (12/30). The falsification is now 30/30,
not 10/10.

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

**Identified vs hybrid on the fixed map (20 seeds, confirmed at 30):** ranking wash
(intervals include 0); RMSE slightly *worse*; 2× sampling cost. That is what
the theory predicted: a benign stationary map does not force collapse, so this
study cannot tell the two losses apart. The discriminating experiment is the
policy 2×2, adjudicated at N=30 (2026-09-03):
[`research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md`](research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md).

Known hole, reweighted and re-measured: the identified loss’s equal
\((w,g)\) weighting made the epistemic term numerically negligible (g is ~orders
of magnitude larger). `TermScaleEMA` (`distill_reweight_ema: true`) fixes the
toy-scale hole (w rank 0.40 → 0.97) but at N=30 in policy the reweighted
identified arms collapse the propagated U to ~0 (student_u_mean ≈ −0.003 vs
teacher −50…−105): local u = w - g is not clamped in the eval object; the measured mechanism (2026-09-05, [`research/U-COLLAPSE-MECHANISM-2026-09-05.md`](research/U-COLLAPSE-MECHANISM-2026-09-05.md)) is that the local object here is aleatoric-dominated (teacher g/w ~ 1e4 under the SAC critic), the identified loss matches w but leaves g at initialization (the aleatoric gradient's measured leverage on g is ~2e-6 per update, ~1e6-1e7 steps short of the gap in a 180-update run), so u ~ w ~ 0. See the ground-truth write-up and the N=30 results doc for the bounded-limitation framing.

---

## What comes next

**Policy 2×2, adjudicated at N=30 (2026-09-03):**
[`research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md`](research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md).
Pre-registered extension to 30 seeds x 5 arms (exact teacher pairing, gap 0).
Decision tree **Row 1** is confirmed: `lagged_hybrid` restores teacher-student
u-rank to ~ordinary (30/30, +0.331), `identified_hybrid` does not (0/30,
-0.464; its reweighted local (w,g) matching collapses the propagated U to ~0).
Nonstationarity is the dominant measured policy mechanism; identifiability
stands as a theoretical result (proof + toy). The 10-seed slice and the
confounded 5-seed preliminary are kept for continuity but do not adjudicate.
Data: [`runs/policy_identifiability_2x2_30seed.json`](runs/policy_identifiability_2x2_30seed.json).
[`research/DECISION-TREE-2X2-PREREGISTRATION.md`](research/DECISION-TREE-2X2-PREREGISTRATION.md).

Ground-truth toy (2026-09-01): [`theory/ground_truth_w_g.py`](theory/ground_truth_w_g.py)
checks the identified loss's (w, g) against ANALYTIC (w*, g*) for the first
time - the estimator is unbiased, hybrid walks the degenerate direction from
both inits, and the equal-weighting hole (w rank 0.40 when g* >> w*) is
quantified. The EMA per-term reweighting fix
(`udwm.models.consistency.TermScaleEMA`, `distill_reweight_ema: true`) recovers
w (rank 0.97) in the hole regime. Part 4's finding: no single scalar weighting
gives both w and g good relative magnitude when g* >> w*, but w RANKING is
robust to the weighting once normalised - the hole is a bounded limitation for
rank-based downstream gating.
[`research/RESULTS-GROUND-TRUTH-W-G-2026-09-01.md`](research/RESULTS-GROUND-TRUTH-W-G-2026-09-01.md).

**N=30 verdict (2026-09-03), registered predictions vs data:**
1. `lagged_hybrid` “reduces but does not kill” the rank collapse → **confirmed**,
   and then some: it removes it (u-rank +0.331, 30/30, to ~ordinary).
2. `identified_hybrid` (live critic) recovers rank → **not confirmed**: 0/30,
   interval excludes zero; the EMA-reweighted arm collapses U to ~0.
3. `lagged_identified` best or tied → **not confirmed** on u-rank; lagging
   directionally helps within identified (20/30, CI excludes 0, below the
   70% bar). Identified distillation is NOT confirmed in policy.

Reproduce the adjudicated table (split by seed, then merged):
```bash
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --out runs/policy_identifiability_2x2_30seed.json --jobs 10 --threads 2
```
Summarize: `python -m udwm.scripts.summarize_policy_2x2 --data runs/policy_identifiability_2x2_30seed.json`.

**Payoff test #4 (2026-09-03), done at N=30:** lagging alone at 2x horizon
([`research/RESULTS-PAYOFF-LAGGED-2026-09-03.md`](research/RESULTS-PAYOFF-LAGGED-2026-09-03.md),
protocol [`research/PAYOFF-LAGGED-PREREGISTRATION.md`](research/PAYOFF-LAGGED-PREREGISTRATION.md)).
Mechanism survival confirmed at N=30 (lagged u-rank 0.988 ~ ordinary 0.986,
30/30 vs hybrid); no return payoff demonstrated (pooled delta -44.6, 15/30,
CI [-102.6, +10.2]).

---

## Running on GPU

Published rows are CPU-verified (checksum-exact). The runners accept `--device cuda` and the seed-split driver pins workers to GPUs with `--gpu-ids`; CUDA rows are not bit-identical to CPU rows, so keep them in separate `_cuda` files. Full guide: [`RUN_ON_GPU.md`](RUN_ON_GPU.md).

## Install and checks

```bash
pip install -r requirements.txt

python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py tests/test_ground_truth_w_g.py -q
python theory/toy_ube_mdp.py
python theory/distill_identifiability.py
python theory/estimator_variance.py
python theory/ground_truth_w_g.py
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
- Policy 2×2 for identified vs hybrid: adjudicated at N=30 — lagging (not identifiability) is the measured fix; the identified loss does not transfer in policy on this benchmark.

---

## Lineage

MBPO · PETS · Luis exact UBE · DIAMOND · consistency / progressive
distillation · WIMLE (latency) · MACURA (do not overclaim gated imagination)
· Voelcker et al. 2025 (calibrated value-aware model learning)

See [`papers/LITERATURE_SURVEY.md`](papers/LITERATURE_SURVEY.md) and
[`papers/NOVELTY_LANDSCAPE.md`](papers/NOVELTY_LANDSCAPE.md).
