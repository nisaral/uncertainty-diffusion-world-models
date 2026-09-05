# Reproduce every table

All runners default to CPU (`device: cpu` in every config) and every
published number was produced on CPU. All runners also accept `--device
cuda`, but CUDA is not bit-identical to the CPU runs below, so published
rows should be reproduced on CPU.

GPU variant of any command below: add `--device cuda` (stress and policy runners) or, for seed-parallel policy runs on several GPUs, `--jobs <#gpus> --gpu-ids 0,1,...` on the split driver. CUDA rows are not bit-identical to CPU rows, so GPU runs are for speed/exploration only -- published numbers come from CPU. See `RUN_ON_GPU.md`.

## Environment and quick checks

```bash
pip install -r requirements.txt          # python >= 3.10, torch >= 2.0
python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py tests/test_ground_truth_w_g.py -q
```

Theory scripts (seconds each):
```bash
python theory/toy_ube_mdp.py
python theory/distill_identifiability.py
python theory/estimator_bias.py
python theory/estimator_variance.py
python theory/ground_truth_w_g.py        # ~1 min; writes runs/ground_truth_w_g.json
```

## Quality bar for every paired study

Exact per-seed teacher checksum pairing (gap 0): each seed's teacher is
restored to the same state for every arm before the student trains. The
summarizers print `exact=True gap=0.0` per seed -- a run without that line on
every seed does not satisfy the repo's pairing bar and is not comparable to
the tables below.

## 1. Fixed-map stress (identifiability rerun), N=30

Protocol: 3,000 transitions, 160 teacher updates, 240 student updates,
5 teacher members, 8 paired latents, 1,024-state eval grid, arms
`ordinary / decision_hybrid / decision_identified / decision_identified_nostate`
(`m_latents=2`).

```bash
python -m udwm.scripts.run_decision_distillation_stress   --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29   --data-size 3000 --teacher-updates 160 --student-updates 240   --ensemble-size 5 --m-samples 8 --grid-size 1024 --m-latents 2   --variants ordinary decision_hybrid decision_identified decision_identified_nostate   --out runs/identified_stress_30seed.json
python -m udwm.scripts.summarize_identified_stress --path runs/identified_stress_30seed.json
```

Rows for seeds 0..19 are byte-identical to the published 20-seed file
(`runs/identified_stress_fair_20seed.json`); a strict-superset check is the
reproduction test.

## 2. Policy 2x2 (fixed vs live critic x M=1 vs M>=2), N=30

5 arms x 30 seeds x 1,800 env steps, DelayedBimodal-v0
(`ordinary / hybrid / lagged_hybrid / identified_hybrid / lagged_identified`).

```bash
python -m udwm.scripts.run_policy_2x2_split_seeds   --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29   --out runs/policy_identifiability_2x2_30seed.json --jobs 10 --threads 2
python -m udwm.scripts.summarize_policy_2x2 --data runs/policy_identifiability_2x2_30seed.json
```

`--jobs` is the number of parallel seed subprocesses; each seed runs all five
arms against one prepared teacher. On a 20-core box use `--jobs 10 --threads 2`
(~1-2 h wall for 30 seeds at 1,800 steps).

## 3. Payoff test #4 (lagging alone, 2x horizon), N=30

3 arms x 30 seeds x 3,600 env steps (`ordinary / hybrid / lagged_hybrid`).
Extension registration: `research/PAYOFF-LAGGED-PREREGISTRATION.md`.

```bash
# serial equivalent (one batch, exact per-seed teacher pairing)
python -m udwm.scripts.run_delayed_bimodal_policy_ablation   --variants ordinary hybrid lagged_hybrid   --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29   --steps 3600 --out runs/payoff_lagged_30seed_3600.json

# split-by-seed (recommended on multi-core): seeds 0..9 then 10..29, merge rows
python -m udwm.scripts.run_policy_2x2_split_seeds   --variants ordinary hybrid lagged_hybrid   --seeds 0 1 2 3 4 5 6 7 8 9 --steps 3600   --out runs/payoff_lagged_10seed_3600.json --jobs 10 --threads 2
python -m udwm.scripts.run_policy_2x2_split_seeds   --variants ordinary hybrid lagged_hybrid   --seeds 10 ... 29 --steps 3600   --out runs/payoff_lagged_ext20_3600.json --jobs 10 --threads 2
# merge the two row sets into runs/payoff_lagged_30seed_3600.json

python -m udwm.scripts.summarize_policy_2x2 --data runs/payoff_lagged_30seed_3600.json
```

## Where each table is written up

| Table | Results doc |
|---|---|
| Identifiability toy (zero-loss family, M>=2 fix) | `research/RESULTS-IDENTIFIABILITY-2026-08-29.md` |
| Fixed-map stress N=30 | `research/RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md` |
| Policy 2x2 N=30 adjudication | `research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md` |
| Payoff #4 (lagging alone) | `research/RESULTS-PAYOFF-LAGGED-2026-09-03.md` |
| Ground-truth (w*, g*) recovery | `research/RESULTS-GROUND-TRUTH-W-G-2026-09-01.md` |
| Estimator bias / variance findings | `research/ESTIMATOR-BIAS-FINDING.md`, `research/ESTIMATOR-VARIANCE-FINDING.md` |
