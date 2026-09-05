# GPU validation on the RTX 4050 (2026-09-05)

**Machine:** NVIDIA GeForce RTX 4050 Laptop (6 GB), driver 591.74 (WDDM),
torch 2.4.1 built with CUDA 11.8, `torch.cuda.is_available() = True`,
1 device. All runs below are informal lab checks, not paper tables.

## 1. Throughput: CPU vs CUDA (batch 256, ms/batch)

`python -m udwm.scripts.benchmark_throughput --device cpu|cuda --batch-size 256`

| model | steps | cpu ms/b | cuda ms/b | speedup |
|---|---:|---:|---:|---:|
| gaussian | 1 | 1.12 | 2.92 | 0.38x |
| diffusion_separate_r | 1 | 2.00 | 4.60 | 0.44x |
| diffusion_separate_r | 2 | 3.20 | 19.83 | 0.16x |
| diffusion_separate_r | 4 | 6.04 | 39.81 | 0.15x |
| diffusion_separate_r | 8 | 18.61 | 69.34 | 0.27x |
| diffusion_separate_r | 10 | 44.08 | 85.36 | 0.52x |
| diffusion_joint_r | 1 | 6.03 | 14.57 | 0.41x |
| diffusion_joint_r | 2 | 10.73 | 22.37 | 0.48x |
| diffusion_joint_r | 4 | 17.52 | 40.45 | 0.43x |
| diffusion_joint_r | 8 | 35.34 | 73.26 | 0.48x |
| diffusion_joint_r | 10 | 45.00 | 89.07 | 0.51x |

**Verdict:** on this 20-core box the tiny-MLP lab is 2-6x SLOWER on the 4050
than on CPU (kernel-launch and WDDM overhead dominate; batch 256 is too small
to amortise). This is not a bug -- it is the reason the published tables run
on CPU. A Kaggle T4/A100 would not change that for DelayedBimodal: the policy
lab is env-step-bound (gymnasium stepping is CPU). GPU only pays off when the
diffusion model itself dominates wall time, i.e. an image-scale / DIAMOND-style
modality -- and then a 6 GB laptop GPU is still modest (see `RUN_ON_GPU.md`).

## 2. End-to-end runner plumbing on cuda

While validating, one real device bug was found and fixed:

- `udwm/models/diffusion_dynamics.py`: the DDPM noise schedule lived on plain
  attributes of `DiffusionDynamicsEnsemble`, so `nn.Module.to(cuda)` on a
  PARENT module (the world model) never moved it; indexing the schedule with a
  cuda timestep tensor raised "indices should be on cpu or the same device".
  Fix: `_apply` override keeps the schedule tensors in sync on every device
  transform (including nested `.to()`); CPU numerics unchanged (verified by
  the unchanged test suite).

After the fix, both policy-runner paths complete on `cuda` with exit 0:

- `run_delayed_bimodal_policy_ablation --device cuda` (seed 0, 120 and 1,000
  steps; the 1,000-step `hybrid` arm exercises teacher pretraining, live-critic
  distillation, UBE, and gated rollouts, ~115 s wall).
- Exact per-seed teacher pairing is preserved on GPU (`max_teacher_checksum_gap
  0.0`), but GPU rows are not byte-identical to CPU rows (cuDNN kernels) --
  keep `_cuda` outputs separate and CPU-verify anything decisive.

## Files

- Runner changes: `--device` on `run_decision_distillation_stress.py`,
  `run_delayed_bimodal_policy_ablation.py`; `--device` + `--gpu-ids` on
  `run_policy_2x2_split_seeds.py` (see `RUN_ON_GPU.md`).
