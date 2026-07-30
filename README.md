# Uncertainty-Aware Diffusion World Models for RL

Open research stack: **diffusion / Gaussian world models** + **MC-UBE multi-step epistemic uncertainty** + MBPO/SAC, with paths to joint reward diffusion and consistency distillation.

| Doc | When to read |
|---|---|
| **`docs/CODE_AND_THEORY_GUIDE.md`** | **Start here** — full math (simple+technical), code, literature, differentiation |
| `docs/PROFESSOR_BRIEFING.md` | Short talk script for advisors |
| `papers/LITERATURE_SURVEY.md` | Compact related-work notes |
| `papers/PAPER_DRAFT.md` | Working research paper draft |
| `research/SAMPLE-BASED-UBE-FOR-DIFFUSION.md` | Theory north-star (Claim A) |

**GitHub:** https://github.com/nisaral/uncertainty-diffusion-world-models

---

## Quick start

```bash
pip install -r requirements.txt
# optional plots: pip install matplotlib

python -m udwm.scripts.smoke_test
python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml
python theory/toy_ube_mdp.py
python -m pytest tests/test_core.py -q
```

### Experiments

```bash
# Ablations (Gaussian / diffusion / joint-R / no-UBE)
python -m udwm.scripts.run_ablations --config configs/ablation_fast.yaml --steps 2000

# Consistency-distilled 1-step student
python -m udwm.scripts.train_mbpo --config configs/consistency_distill.yaml

# Joint reward in diffusion (Gap 1)
python -m udwm.scripts.train_mbpo --config configs/joint_reward.yaml

# Throughput + plots
python -m udwm.scripts.benchmark_throughput
python -m udwm.scripts.plot_results
```

---

## Research map

| Gap | Question | Status |
|---|---|---|
| Core | Shared WM + UBE + SAC stack | Done |
| Gap 3 | Continuous control + diffusion UQ + distill | Code + draft paper |
| Gap 1 | Joint reward in diffusion | Implemented path |
| Gap 2 | Multi-agent state→value uncertainty | Future |

---

## Package layout

```
udwm/models/       Gaussian, diffusion, joint-R, consistency distill
udwm/uncertainty/  MC-UBE local rewards + U-network
udwm/rl/           SAC + MBPO trainer
udwm/eval/         return, MSE, calibration, throughput
udwm/scripts/      train, ablate, plot, evaluate, smoke
theory/            exact UBE MRP + MC estimator toys
papers/            literature survey + paper draft + bib
docs/              understanding guide + professor briefing
```

---

## Citation lineage (short)

MBPO · PETS · Luis exact UBE · DIAMOND · Consistency models · WIMLE · Wang et al. multi-agent diffusion PO  

See `papers/LITERATURE_SURVEY.md` and `papers/references.bib`.
