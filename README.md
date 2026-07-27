# Uncertainty-Aware Diffusion World Models for RL

Shared research stack for **continuous-control world models** with **multi-step epistemic uncertainty** (UBE) and paths to joint reward diffusion (DIAMOND open problem) and multi-agent PO (stretch).

## Talk to a professor?

Read **`docs/PROFESSOR_BRIEFING.md`** — simple pitch + technical depth + Q&A cheat sheet.

---

## What this repo is

| Layer | Contents |
|---|---|
| **Code (`udwm/`)** | Gaussian or diffusion ensemble WM, joint/separate reward, SAC+MBPO, MC-UBE, eval metrics |
| **Theory (`research/`)** | Sample-based UBE claim, assumptions log, proof checklists |
| **Docs (`docs/`)** | Professor briefing, project narrative |

### Research map

| Gap | Question | Code status |
|---|---|---|
| **Core** | One stack for all gaps | Done |
| **Gap 3** | Continuous control + distilled diffusion + UBE | Trainer + eval + long config |
| **Gap 1** | Joint reward in diffusion (DIAMOND limitation) | `joint_with_diffusion: true` implemented |
| **Gap 2** | Multi-agent state-error → value U | Stub only |

---

## Install & run

```bash
cd uncertainty-diffusion-world-models
pip install -r requirements.txt

# Component smoke test (Gaussian + diffusion + joint R + UBE)
python -m udwm.scripts.smoke_test

# Short train (Pendulum)
python -m udwm.scripts.train_mbpo --config configs/smoke_train.yaml

# Joint reward diffusion (Gap 1 path)
python -m udwm.scripts.train_mbpo --config configs/joint_reward.yaml

# Longer Gap 3-style run
python -m udwm.scripts.train_mbpo --config configs/pendulum_long.yaml --steps 25000

# Throughput: Gaussian vs diffusion sample steps (answers "is diffusion too slow?")
python -m udwm.scripts.benchmark_throughput

# Evaluate a checkpoint
python -m udwm.scripts.evaluate --checkpoint checkpoints/mbpo_diffusion_sepR_seed0.pt --config configs/smoke_train.yaml

# Unit tests
python -m pytest tests/test_core.py -q

# Gap-3 ablations (Gaussian vs diffusion vs joint R vs no-UBE)
python -m udwm.scripts.run_ablations --config configs/ablation_fast.yaml --steps 2000

# Theory toy: exact UBE vs MC on a small DAG MRP
python theory/toy_ube_mdp.py
python theory/toy_mc_ube_estimator.py
```

### Useful flags

```bash
--model gaussian|diffusion
--joint-reward          # Gap 1
--separate-reward
--no-ube
--steps N
--device cpu|cuda
```

---

## Package layout

```
udwm/
  models/          gaussian_ensemble, diffusion_dynamics (joint R), reward_term, world_model
  uncertainty/     mc_ube (MC-UBE-Local + U-net), calibration
  rl/              sac, trainer (MBPO + full eval metrics)
  eval/            return, WM accuracy, U calibration, throughput
  scripts/         smoke_test, train_mbpo, evaluate, benchmark_throughput
configs/           default, smoke_train, pendulum_long, joint_reward
docs/              PROFESSOR_BRIEFING.md
research/          theory north-star
theory/            pure-math toy MC-UBE
```

---

## Method in one diagram

```
Env ──► real buffer ──► train ensemble WM (Gaussian | diffusion | joint-R diffusion)
                              │
                              ▼ few-step DDIM imagination
                        model buffer ──► SAC  +  MC-UBE ──► U(s,a)
                              │
                              └─ policy: Q ± λ√U
```

---

## References

- Luis et al. 2023 — Exact UBE  
- Alonso et al. 2024 — DIAMOND  
- Janner et al. 2019 — MBPO  
- WIMLE 2026 — uncertainty-aware continuous control (IMLE)  
- Wang et al. 2025 — multi-agent diffusion PO bounds  
