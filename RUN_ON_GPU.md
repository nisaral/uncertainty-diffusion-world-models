# Running on GPU

**Read this first.** The published tables in `research/` are CPU-verified with
exact per-seed teacher checksums. CUDA is *not* bit-identical to CPU (different
kernel and reduction orderings), so a GPU row will never byte-match a published
CPU row. Use GPU for speed and exploration; if a result is going to adjudicate
anything, re-run the decisive rows on CPU with the `REPRODUCE.md` commands.

## When GPU actually helps this repo

- The policy lab (`DelayedBimodal`) is **env-step-bound**: the gymnasium
  stepping loop runs on the CPU no matter what, so a single GPU buys little for
  the current benchmark. Multi-GPU only helps by running several *seeds* in
  parallel.
- GPU pays off where the diffusion model dominates the wall time: the
  fixed-map stress grid evaluation (`1024` states x `M` latents x `N` members),
  and any future image-scale / DIAMOND-style modality.
- Check whether a GPU matters on this box before scheduling anything:
  `python -m udwm.scripts.benchmark_throughput --device cuda` reports
  teacher-vs-student sampling seconds per batch on the device you name.

## Measured on this machine (RTX 4050 Laptop, 6 GB, 2026-09-05)

`benchmark_throughput --batch-size 256` reports the diffusion sampler is
**2-6x slower on CUDA than on this machine's CPU** (tiny MLPs: kernel
launch + WDDM overhead dominate at batch 256; see
`research/RESULTS-GPU-VALIDATION-2026-09-05.md`). Consequence: run the
current DelayedBimodal lab on CPU; treat the GPU path as validated-but-not-faster until a workload (image-scale modality) makes the diffusion
model dominate wall time.

## Prerequisites

```bash
# torch must be a CUDA build (not the CPU wheel). Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# sanity: prints version, availability, and device count
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

All configs ship as `device: cpu`. You do **not** need to edit them: every
runner now accepts `--device cpu|cuda` and fails loudly if you ask for `cuda`
and it is unavailable. Runners that read a config (`MBPOTrainer` and friends)
already move every model, buffer, and tensor through the config/CLI device, so
`--device cuda` is sufficient end to end.

## Fixed-map stress (one GPU)

```bash
python -m udwm.scripts.run_decision_distillation_stress \
  --seeds 0 1 2 ... 29 --device cuda \
  --data-size 3000 --teacher-updates 160 --student-updates 240 \
  --ensemble-size 5 --m-samples 8 --grid-size 1024 --m-latents 2 \
  --variants ordinary decision_hybrid decision_identified decision_identified_nostate \
  --out runs/identified_stress_30seed_cuda.json
```

`--device` is recorded in the output's `protocol` (via `vars(args)`), so a
`_cuda` file is self-describing and must never be merged into the CPU file.

## Policy lab (one GPU, seeds serial)

The exact per-seed teacher pairing is preserved by construction on any device
(each seed's teacher state is restored before every arm), so seed-parallel is
safe:

```bash
python -m udwm.scripts.run_delayed_bimodal_policy_ablation \
  --variants ordinary hybrid lagged_hybrid --seeds 0 1 2 ... 9 --steps 3600 \
  --device cuda --out runs/payoff_10seed_3600_cuda.json
```

The `--device` value lands in `protocol` via `vars(args)`.

## Policy lab (multi-GPU, seed-parallel)

`run_policy_2x2_split_seeds.py` pins each seed worker to one GPU with
`CUDA_VISIBLE_DEVICES`. Pass the physical ids you want to cycle and keep
`--jobs <=` the number of GPUs (one seed per GPU):

```bash
# 2 GPUs, 30 seeds: seeds alternate cuda:0 / cuda:1
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --variants ordinary hybrid lagged_hybrid --seeds 0 ... 29 --steps 1800 \
  --out runs/policy_30seed_cuda.json --jobs 2 --gpu-ids 0,1
```

- `--gpu-ids 0,1` implies `--device cuda` for every worker.
- The driver prints each assignment (`[split] seed N -> cuda:X`); the merged
  `protocol` records `device` and `gpu_ids`.
- More workers than GPUs is allowed but several seeds will contend for one
  device; the driver prints a note when that happens.
- Monitor with `nvidia-smi`; each worker is one `python` process on its own
  GPU.

## Determinism and the pairing bar

- Do not mix GPU and CPU rows in one `runs/*.json`: the repo's quality bar is
  checksum-exact, byte-identical reproduction (e.g., the fixed-map 20-seed
  superset check). CUDA rows satisfy the within-seed checksum pairing but not
  byte-identity with the CPU files, so keep `_cuda` outputs separate and label
  them.
- If a GPU experiment looks decisive, confirm on CPU (`--device cpu` or the
  stock config) before it enters a results doc.
- cuDNN is allowed to be non-deterministic here; nothing in the repo sets
  `torch.backends.cudnn.deterministic`, and we do not promise GPU
  reproducibility across driver versions.

## Memory

All models in this lab are small MLPs (hidden 16-256) with batch sizes
64-1024. Any modern GPU fits everything; no memory tuning is needed. If you
later add an image encoder/decoder (the deferred modality step), that is where
VRAM budgeting starts.

## Mechanism probe and leverage sweep (2026-09-05 tooling)

`udwm/scripts/probe_u_collapse.py` re-runs one policy arm and logs the
identified-loss trajectory plus an eval-time (w, g, u, coupling, latent-spread)
decomposition. It accepts the same `--device` and `--set` (model-config
overrides) plus `--student-lr` and `--save`:

```bash
# single arm on GPU (slower than CPU at this lab's scale; exploration only)
python -m udwm.scripts.probe_u_collapse --variant identified_hybrid --seed 0 \
  --steps 1800 --device cuda --out runs/probe_u_x.json

# leverage-sweep cell (the A/B arms of RESULTS-LEVERAGE-FIX-2026-09-05.md)
python -m udwm.scripts.probe_u_collapse --variant identified_hybrid --seed 0 \
  --set distill_reweight_ema=false,distill_aleatoric_weight=1e4,distill_grad_clip=1000.0 \
  --student-lr 0.01 --steps 1800 --out runs/probe_u_lrscale_B1_s0.json
```

Seed-parallel GPU sweep: use the committed `probe_sweep_gpu.sh`
(one worker per GPU, trimmed with `jobs -pr` so the worker count stays at the
GPU count over any number of seeds):

```bash
VARIANT=identified_hybrid SEEDS="0 1 2 3" GPU_IDS="0 1" ./probe_sweep_gpu.sh
# optional: SET="distill_reweight_ema=false,distill_aleatoric_weight=1e4,distill_grad_clip=1000.0" STUDENT_LR=0.01
```

## 2026-09-05 addendum: corrected-weight arms and corruption mode (GPU/CPU commands)

The mechanism correction (see `research/RESULTS-CORRUPTION-2026-09-05.md` and
`research/CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md`) adds two policy
variants to every runner and a `distill_corruption` knob to the identified
loss.  The corrected arms are what the next policy N-study should run:

```bash
# policy-level N-study with the corrected arms (local CPU: parallel jobs)
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --variants ordinary hybrid lagged_hybrid identified_hybrid identified_eq identified_wonly \
  --jobs 10 --threads 2 \
  --out runs/policy_corrected_weights_10seed.json

# same, spread over GPUs (each seed worker pinned to a device; env-step-bound
# lab, so GPU helps only via seed parallelism, and CUDA rows are not
# bit-identical to CPU - adjudicate on CPU)
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --seeds 0 .. 29 --variants ... --jobs 8 --threads 2 --gpu-ids 0,1 \
  --out runs/policy_corrected_weights_30seed_gpu.json
```

Probe-level corrected-arm cells (fast, ~2 min/seed/arm on CPU):

```bash
# equal-weight identified (recovers g; the corrected control)
python -m udwm.scripts.probe_u_collapse --variant identified_eq --seed 0 \
  --out runs/probe_u_eq_s0.json

# EMA-both identified (historical config; collapses g -> the control cell)
python -m udwm.scripts.probe_u_collapse --variant identified_hybrid --seed 0 \
  --out runs/probe_u_ema_s0.json

# w-only EMA (also collapses g; identifies the epistemic up-weight as driver)
python -m udwm.scripts.probe_u_collapse --variant identified_wonly --seed 0 \
  --out runs/probe_u_wonly_s0.json
```

Corruption-mode probes (the registered corruption-distribution test;
`schedule` is the bit-identical default):

```bash
python -m udwm.scripts.probe_u_collapse --variant identified_eq --seed 0 \
  --set distill_corruption=pure  --out runs/probe_u_pure_s0.json
python -m udwm.scripts.probe_u_collapse --variant identified_eq --seed 0 \
  --set distill_corruption=maxt  --out runs/probe_u_maxt_s0.json
```

Verdict at probe scale (2 seeds, 2026-09-05): the corruption distribution is
NOT the binding constraint - equal-weight identified recovers `g` (u-rank
0.88-0.92) at every corruption level, while EMA-both and w-only collapse it.
GPU use for these cells is pointless (CPU is faster at this lab's scale);
use GPU only for seed-parallel policy batches or future image-scale work.

## Kaggle / cloud GPU quickstart (future DMC-scale work)

The registered next benchmark is a long-horizon low-dimensional task (a
DMC/MuJoCo-scale env), pre-registered in
`research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`; that file is the binding
contract (diagnostic gate first: measure g*/w* under the DMC critic, then run
the comparison with the pre-committed arms). Kaggle images ship
`dm_control`/`mujoco`, which this machine does not have (`gymnasium` only).

1. Register before running: the DMC pre-registration is committed; any change
   to arms/endpoints/seeds needs a new registration before compute is spent.
2. Repo does not have `setup.py`; on the Kaggle host export PYTHONPATH and
   install extras:
   ```bash
   git clone https://github.com/nisaral/uncertainty-diffusion-world-models
   cd uncertainty-diffusion-world-models
   pip install -q gymnasium numpy torch dm_control mujoco pyyaml tqdm
   export PYTHONPATH=$PWD
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```
3. Add the DMC env adapter under `udwm/envs/registry.py` (`dmc.<task>` id) -
   the only missing piece - then run the policy runners with `--device cuda`
   and `--gpu-ids` for seed parallelism. Adjudicate decisive rows on CPU
   afterwards (CUDA is not bit-identical to CPU in this repo; see top of file).
4. If the gate says aleatoric-dominated, run the `identified_eq` arm (equal
   weight) as the uncertainty-preserving candidate and keep
   `identified_hybrid` (EMA) as the expected-collapse control; do NOT carry
   lambda/LR scaling as a promising arm (that thread closed 2026-09-05).
