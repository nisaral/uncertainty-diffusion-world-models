# Uncertainty-Aware Diffusion World Models for RL

**Core Idea**: Build fast, uncertainty-aware diffusion-based world models that can be used safely and efficiently inside model-based reinforcement learning loops (policy learning, planning, and offline settings).

We combine:
- Distilled / consistency diffusion models (fast enough for online rollouts)
- Ensemble + sampling-based uncertainty estimation (epistemic + aleatoric)
- Uncertainty-weighted synthetic data (WIMLE-style reweighting)

**Reference Papers**
- MBPO — Janner et al. (2019)
- WIMLE — Aghabozorgi et al. (ICLR 2026)
- Consistency Models — Song et al. (2023)
- SAC — Haarnoja et al. (2018)
- Diffusion Policy — Chi et al.

## Project Status
**Current Phase**: Phase 0 — Reproducing MBPO baseline

## Key Results (will be updated)
- [ ] Phase 0: MBPO reproduction on DeepMind Control Suite
- [ ] Phase 1: Diffusion dynamics model (prediction accuracy)
- [ ] Phase 2: Consistency distillation + throughput benchmark
- [ ] Phase 3: Uncertainty calibration plots
- [ ] Phase 4: Sample-efficiency curves with uncertainty weighting

## Repository Structure
