# Fixed-map stress, N=30 extension (2026-09-03)

**Data:** `runs/identified_stress_30seed.json` (30 paired seeds x 4 arms).
**Protocol:** same as the 08-29 identified-fair study -- 3,000 transitions, 160
teacher updates, 240 student updates, 5 teacher members, 8 paired latent
samples, 1,024-state eval grid, `decision_value` fixed map, arms `ordinary` /
`decision_hybrid` / `decision_identified` / `decision_identified_nostate`
(`m_latents=2`). Teacher restored exactly per seed; every seed reports one
checksum value (gap 0).
**Reproduction:** the 80 rows for seeds 0-19 are byte-identical to
`runs/identified_stress_fair_20seed.json`; 20/20 teacher checksums bit-identical
to the 2026-08-29 run. The N=30 file is a strict superset, so every 20-seed
conclusion below carries over unchanged and is sharpened by 10 new seeds.
**Tool:** `udwm/scripts/summarize_identified_stress.py` (20,000-draw percentile
bootstrap over seeds).

## 1. `decision_hybrid` vs `ordinary` (extends the fixed-map headline)

| endpoint | mean delta | wins/30 | bootstrap 95% | verdict |
|---|---:|---:|---|---|
| `w_rmse` | -0.003397 | 29/30 | [-0.004209, -0.002662] | BETTER |
| `w_deb_rmse` | -0.002367 | 29/30 | [-0.002987, -0.001831] | BETTER |
| `w_rank_corr` | +0.046524 | 22/30 | [+0.019270, +0.075348] | BETTER |
| `w_deb_rank_corr` | +0.055529 | 24/30 | [+0.028537, +0.083364] | BETTER |
| `top_decile_recall` | +0.029126 | 16/30 | [-0.001626, +0.061489] | inconclusive |
| `g_rank_corr` | -0.002329 | 11/30 | [-0.004396, -0.000368] | WORSE |
| `g_rmse` | +0.004520 | 9/30 | [+0.002101, +0.007049] | WORSE |
| `conflated_rank_corr` | -0.002543 | 10/30 | [-0.004641, -0.000563] | WORSE |
| `paired_state_mse` | -0.000978 | 23/30 | [-0.001327, -0.000630] | BETTER |
| `member_value_rmse` | -0.006448 | 24/30 | [-0.009027, -0.003955] | BETTER |
| `split_distortion` | +0.241296 | 29/30 | [+0.153252, +0.352040] | moves toward 1 |

At 30 seeds the fixed-map magnitude claim is 29/30 and the w-rank gain is now
confirmed (22/30, interval excludes zero) where the 20-seed table's rank line
was borderline. The aleatoric side degrades (`g_*` worse) and the
single-latent conflated statistic worsens: the loss is reallocating the split
(epistemic up, aleatoric down), exactly as Section 3 of the identifiability
write-up predicts. Top-decile recall remains inconclusive (16/30; the interval
just touches zero).

## 2. `decision_identified` vs `ordinary`, and vs `hybrid`

vs `ordinary` (30 seeds): `w_rmse` -0.002878 (29/30), `w_deb_rmse` -0.002020
(29/30), `w_rank_corr` +0.037948 (21/30, CI [+0.0093, +0.0679]), `w_deb_rank_corr`
+0.050025 (24/30); `g_*` worse; `top_decile_recall` inconclusive.

Head-to-head `decision_identified` - `decision_hybrid` (30 seeds):

| endpoint | mean delta | wins/30 | bootstrap 95% | verdict |
|---|---:|---:|---|---|
| `w_rank_corr` | -0.008577 | 15/30 | [-0.033476, +0.012746] | inconclusive (wash) |
| `w_rmse` | +0.000519 | 3/30 | [+0.000296, +0.000762] | WORSE |
| `w_deb_rmse` | +0.000347 | 5/30 | [+0.000189, +0.000517] | WORSE |
| `member_value_rmse` | +0.002792 | 7/30 | [+0.001629, +0.003981] | WORSE |
| `split_distortion` | -0.092934 | 5/30 | [-0.163710, -0.043607] | WORSE |

**On the fixed map the identified objective remains a ranking wash and slightly
worse on magnitude, at 2x the sampling cost** -- the same conclusion as the
20-seed table, now at 30 seeds. That is what the theory predicts: a benign,
stationary `decision_value` gives gradient descent no reason to walk the
degenerate direction, so the fixed map cannot discriminate the two objectives.
The discriminating experiment is the policy setting, which is adjudicated at
N=30 in `research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md` (Row 1: lagging is
the measured fix; the identified loss does not transfer).

`decision_identified_nostate` vs `decision_hybrid` is worse on the w endpoints
(`w_rank_corr` -0.034, 13/30, CI excludes 0; `w_rmse` +0.0013, 3/30),
reconfirming that the state-geometry terms carry a real share of the gain.

## Registration hygiene

- Same code and protocol as the 08-29 fair study (bit-identical rows for seeds
  0-19); seeds 20-29 added in one batch, per-seed subprocesses merged; no arm
  dropped post-hoc; bootstrap conventions unchanged.
