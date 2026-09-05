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
