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

# same, spread over GPUs - use the committed launcher (root-level
# policy_sweep_gpu.sh; one worker per seed/GPU, CUDA_VISIBLE_DEVICES pinning
# handled by the driver). CUDA rows are not bit-identical to CPU - adjudicate
# on CPU:
SEEDS="0 1 2 ... 29" GPU_IDS="0,1" OUT=runs/policy_corrected_weights_30seed_gpu.json \
  ./policy_sweep_gpu.sh
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

## Kaggle / cloud GPU quickstart (DMC payoff study)

The registered next benchmark is a long-horizon low-dimensional DMC task,
pre-registered in `research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`
(binding contract, incl. Amendment 1: diagnostic gate first - measure the
teacher `g*/w*` under the live critic - then the pre-committed arms). The
task config (`configs/dmc_hopper_distill.yaml`) and the staged launcher
(`dmc_payoff.sh`) are committed; nothing needs to be hand-edited on the host.

Target host: Kaggle notebook, 2x T4. The repo's GPU-scaling note applies:
GPU pays off as seed-parallel policy batches (`jobs = #GPUs`), which is what
`stage1`/`stage2` do. A single-GPU session works too - set `GPU_IDS=0` and
the launcher falls back to one worker.

### Notebook cells (copy each block into its own cell)

Use `%%bash` cells, NOT `!cmd` lines: pasting from a chat/markdown renderer
often adds leading whitespace, and an indented `!cmd` becomes an
`IndentationError` in a Python cell. A `%%bash` body tolerates indentation
(bash ignores leading spaces), and each cell below is self-contained
(`cd`/`export` repeat per cell because shell state does not persist between
cells). The launcher defaults already match the registered study, so only
`GPU_IDS` is ever overridden (`0,1` for 2x T4; set `0` on a single-GPU
session).

```bash
%%bash
# Cell 0 - clone/update the repo
cd /kaggle/working
[ -d uncertainty-diffusion-world-models ] || git clone https://github.com/nisaral/uncertainty-diffusion-world-models
cd uncertainty-diffusion-world-models
git pull --ff-only
git log --oneline -1
```

```bash
%%bash
# Cell 1 - deps + CUDA sanity (pip no-ops on packages Kaggle already ships;
# shimmy bridges dm_control -> gymnasium for udwm.envs.registry)
pip install -q gymnasium numpy dm_control mujoco shimmy pyyaml tqdm
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
python -c "import torch; print('cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count())"
```

```bash
%%bash
# Cell 2 - stage0 smoke: env spaces + CUDA (fix installs here if it fails)
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
GPU_IDS=0,1 bash dmc_payoff.sh stage0
```

```bash
%%bash
# Cell 3 - stage1 diagnostic gate (mandatory; prints teacher g*/w* + regime)
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
GPU_IDS=0,1 bash dmc_payoff.sh stage1
# Record the regime in research/RESULTS-DMC-PAYOFF-*.md BEFORE stage2.
```

```bash
%%bash
# Cell 4 - stage2 sanity subset (10 seeds) before the full 30
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
GPU_IDS=0,1 SEEDS="0 1 2 3 4 5 6 7 8 9" OUT=runs/dmc_payoff_10seed_gpu.json bash dmc_payoff.sh stage2
```

```bash
%%bash
# Cell 5 - stage2 full run, 30 seeds (resume-safe: re-running this cell after
# a dropped session skips completed per-seed partials)
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
GPU_IDS=0,1 bash dmc_payoff.sh stage2
```

```bash
%%bash
# Cell 6 - stage3 adjudication summary (CPU aggregation of the GPU rows)
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
bash dmc_payoff.sh stage3
```

```python
# Cell 7 - download the merged json(s) for runs/ + the results doc
# (keep this python cell at column 0 - no leading spaces)
from IPython.display import FileLink
FileLink("runs/dmc_payoff_30seed_gpu.json")
```

### Single-cell quick start (setup -> 10-seed sanity)

Prefer one cell over cells 0-7 above? This runs clone, install, stage0,
stage1 gate, stage2 on a 10-seed subset, and stage3 in a single cell.
`set -e` stops the cell at the first failure (a smoke failure must not
cascade into later stages). The full 30-seed run stays a separate cell
because it runs for hours and the gate regime must be recorded first.

```bash
%%bash
set -euo pipefail
cd /kaggle/working
[ -d uncertainty-diffusion-world-models ] || git clone https://github.com/nisaral/uncertainty-diffusion-world-models
cd uncertainty-diffusion-world-models
git pull --ff-only
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
pip install -q gymnasium numpy dm_control mujoco shimmy pyyaml tqdm
echo "=== [1/4] stage0: env smoke ==="
GPU_IDS=0,1 bash dmc_payoff.sh stage0
echo "=== [2/4] stage1: diagnostic gate (2 seeds, ordinary) ==="
GPU_IDS=0,1 bash dmc_payoff.sh stage1
echo "=== [3/4] stage2: 10-seed sanity subset ==="
GPU_IDS=0,1 SEEDS="0 1 2 3 4 5 6 7 8 9" OUT=runs/dmc_payoff_10seed_gpu.json bash dmc_payoff.sh stage2
echo "=== [4/4] stage3: sanity summary ==="
OUT=runs/dmc_payoff_10seed_gpu.json bash dmc_payoff.sh stage3
echo "sanity done - READ the gate regime in [2/4], record it in"
echo "research/RESULTS-DMC-PAYOFF-*.md, then run the 30-seed cell below."
```

```bash
%%bash
# Full 30-seed run + adjudication (run AFTER recording the gate regime)
set -euo pipefail
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
git pull --ff-only
GPU_IDS=0,1 bash dmc_payoff.sh stage2   # defaults: seeds 0..29 -> runs/dmc_payoff_30seed_gpu.json
GPU_IDS=0,1 bash dmc_payoff.sh stage3
```

### Registered budget probe (2026-09-07, diagnostic)

The 10-seed sanity left an operating-point confound (every arm ~0.5 u_rank at
3,600 steps = 3.6 episodes of a 1,000-step task; returns flat). Per
`research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`, run this probe
BEFORE the 30-seed cell below - it decides whether the 30-seed adjudication
runs at 3,600 steps or is amended to a higher budget. Diagnostic only
(ordinary / lagged_hybrid / identified_eq at 15,000 steps = 15 episodes,
seeds 0-1; ~1-2 h on 2x T4):

```bash
%%bash
set -euo pipefail
cd /kaggle/working/uncertainty-diffusion-world-models
export PYTHONPATH=/kaggle/working/uncertainty-diffusion-world-models
git pull --ff-only
python -m udwm.scripts.run_policy_2x2_split_seeds \
  --config configs/dmc_hopper_probe.yaml \
  --seeds 0 1 \
  --variants ordinary lagged_hybrid identified_eq \
  --steps 15000 --jobs 2 --threads 2 --gpu-ids 0,1 \
  --out runs/dmc_budget_probe_gpu.json
python -m udwm.scripts.print_eval_history --data runs/dmc_budget_probe_gpu.json
```

Compare the 15k u_rank/return against the same seeds' 3.6k rows in
`runs/dmc_payoff_10seed_gpu.json`; the decision rule is pre-committed in the
probe registration.

Terminal equivalent (one shell):

```bash
git clone https://github.com/nisaral/uncertainty-diffusion-world-models && cd uncertainty-diffusion-world-models
pip install -q gymnasium numpy dm_control mujoco shimmy pyyaml tqdm
export PYTHONPATH=$PWD
export GPU_IDS=0,1        # single-GPU session: GPU_IDS=0
bash dmc_payoff.sh stage0
bash dmc_payoff.sh stage1
bash dmc_payoff.sh stage2 # defaults: 30 seeds -> runs/dmc_payoff_30seed_gpu.json
bash dmc_payoff.sh stage3
```

Notes:
- dm_control tasks expose Dict observations (`position`/`velocity`/`touch`,
  etc.); `udwm.envs.registry.make_env` flattens them to 1-D Box with
  gymnasium's `FlattenObservation`, so any registered locomotion task works.
- Registered arms run unchanged: `ordinary hybrid lagged_hybrid
  identified_hybrid identified_eq` (Amendment 1: EMA-collapse control +
  equal-weight partial-transfer candidate), 3,600 env steps per seed. Do NOT
  change arms/endpoints/seeds without a new registration.
- Resume: per-seed partial files are written atomically per arm
  (`runs/*_seedN.partial.json`); re-running stage1/stage2 resumes finished
  seeds after a session drop.
- Install conflicts (Kaggle images drift): `pip list | grep -iE
  "gym|shimmy|dm-control|mujoco|torch"` and pin the versions that match the
  image; requirements are gymnasium >= 0.28, shimmy, dm_control, mujoco,
  and a CUDA torch.
- DMC rows are GPU rows. Within-seed teacher checksum pairing holds, but CUDA
  is not bit-identical to CPU and this machine has no dm_control, so DMC
  verdicts are adjudicated on the GPU json (stage3). Label the results doc
  `_gpu` and record `protocol.device`/`gpu_ids`; never merge into a CPU file.