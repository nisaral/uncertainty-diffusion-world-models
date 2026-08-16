# Uncertainty-Aware Diffusion World Models for RL

Open research code: **diffusion / Gaussian world models** + **MC-UBE multi-step epistemic uncertainty** + MBPO/SAC, with joint reward and consistency-distillation paths.

**Repo:** https://github.com/nisaral/uncertainty-diffusion-world-models

---

## Install & quick start

```bash
pip install -r requirements.txt

python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py -q
python theory/toy_ube_mdp.py

python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml
```

### More experiments

```bash
# Multi-seed aggregate
python -m udwm.scripts.run_multiseed --config configs/smoke_train.yaml --seeds 0 1 2 --steps 2000

# Ablations
python -m udwm.scripts.run_ablations --config configs/ablation_fast.yaml --steps 2000

# Consistency distill (1-step student)
python -m udwm.scripts.train_mbpo --config configs/consistency_distill.yaml

# Joint reward diffusion
python -m udwm.scripts.train_mbpo --config configs/joint_reward.yaml

# Longer Pendulum (horizon curriculum 1→3)
python -m udwm.scripts.train_mbpo --config configs/pendulum_long.yaml --steps 25000

# Throughput + model diagnostics
python -m udwm.scripts.benchmark_throughput
python -m udwm.scripts.diagnose_model --config configs/smoke_train.yaml
python -m udwm.scripts.plot_results
```

### Flags

```text
--model gaussian|diffusion
--joint-reward | --separate-reward
--no-ube
--steps N
--device cpu|cuda
```

---

## What this code implements

| Piece | Location |
|---|---|
| Gaussian ensemble dynamics | `udwm/models/gaussian_ensemble.py` |
| Diffusion ensemble (+ joint reward) | `udwm/models/diffusion_dynamics.py` |
| Consistency distillation | `udwm/models/consistency.py` |
| MC-UBE local rewards + U-net | `udwm/uncertainty/mc_ube.py` |
| SAC + optional \(Q\pm\lambda\sqrt{U}\) | `udwm/rl/sac.py` |
| MBPO trainer + horizon curriculum | `udwm/rl/trainer.py` |
| Exact UBE toy MRP | `theory/toy_ube_mdp.py` |
| Paper draft / short lit notes | `papers/` |

**Research claim (short):** Luis et al. (2023) give a distribution-agnostic Uncertainty Bellman Equation; we implement **Monte Carlo estimators** of local UBE rewards for **diffusion / consistency** world-model ensembles and a full continuous-control MBRL stack.

### Architectural novelty (A+B)

| Module | Idea | Config |
|---|---|---|
| **A — U-gated imagination** | Stop / down-weight imagined transitions when \(\sqrt{U}\) is high | `u_gate.mode: stop\|weight\|both` |
| **B — Adaptive MC-UBE** | Probe with few samples; refine high-\(w\) states with more \(M\) (and optional NFE) | `ube.adaptive_mc.enabled: true` |

```bash
python -m udwm.scripts.train_mbpo --config configs/ab_novel.yaml
# Selective prediction diagnostics (√U as a *score*, not a guaranteed threshold)
python -m udwm.scripts.eval_selective --config configs/ab_novel.yaml
```

Gate decisions can be replayed from `runs/gate_audit.jsonl` (`decision`, `sqrt_u`, `tau`, `reason`).  
`eval_selective` reports risk–coverage and **over-rejection** if √U is treated as an absolute threshold. These are measurements, not conformal coverage.

---

## Layout

```
udwm/          Python package
configs/       YAML experiments
theory/        Math toys (no neural nets)
papers/        Paper draft + bib + short literature notes
tests/         Unit tests
```

Personal study notes live only on your machine under `private_notes/` (gitignored).

---

## Citation lineage

MBPO · PETS · Luis exact UBE · DIAMOND · Consistency models · WIMLE  

See `papers/LITERATURE_SURVEY.md` and `papers/PAPER_DRAFT.md`.
