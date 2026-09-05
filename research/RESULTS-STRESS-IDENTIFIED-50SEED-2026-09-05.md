# Fixed-map stress, N=50 precision extension (2026-09-05)

**Status: registered before seeds 30..49 ran. Results appended below after the
run; verdicts were not read before registration.**

**Data (after merge):** `runs/identified_stress_50seed.json` = seeds 0..29 from
`runs/identified_stress_30seed.json` (rows byte-identical, strict superset) +
seeds 30..49 run on CPU with the identical protocol.

**Protocol:** unchanged from the N=30 study -- 3,000 transitions, 160 teacher
updates, 240 student updates, 5 teacher members, 8 paired latent samples,
1,024-state eval grid, `decision_value` fixed map, arms `ordinary` /
`decision_hybrid` / `decision_identified` / `decision_identified_nostate`
(`m_latents=2`). Teacher restored exactly per seed; every seed reports one
checksum value (gap 0 within seed, exact pairing by construction).

**Why N=50:** the only unresolved endpoint at N=30 is `top_decile_recall`
(16/30 for hybrid vs ordinary; bootstrap interval just touches zero). This
extension is a precision run on that single endpoint. All other N=30 verdicts
(magnitude 29/30, w-rank 22-24/30, `g` degradation, identified-vs-hybrid wash,
`nostate` worse) are expected to be stable; they are re-reported, not
re-adjudicated.

**Adjudication rules (fixed before the run):**
- Stats: mean paired delta vs `ordinary`, wins/N, 20,000-draw percentile
  bootstrap 95% CI (same conventions as `summarize_identified_stress`).
- `top_decile_recall` "resolved": CI excludes 0 AND >= 35/50 seeds in the
  direction; else reported inconclusive at N=50 (a further extension would
  need a new registration).
- All other endpoints: reported with the N=30 verdicts unchanged if the
  direction and CI agree; any sign flip is called out explicitly.

**Registration hygiene:** protocol, endpoint, and thresholds fixed in this file
before seeds 30..49 were launched; no arm or seed dropped post-hoc; seeds
0..29 rows are the published 30-seed file untouched.

---

## Results (filled after the run)

**Data:** `runs/identified_stress_50seed.json` (200 rows = 50 seeds x 4 arms).
Seeds 0..29 are byte-identical to `runs/identified_stress_30seed.json`
(strict-superset check passed); seeds 30..49 ran 2026-09-05 on CPU with the
identical protocol (per-seed subprocesses, exact teacher pairing, gap 0).

## Hybrid vs ordinary at N=50 (paired per seed, vs `ordinary`)

| endpoint | mean delta | wins/50 | bootstrap 95% | verdict |
|---|---:|---:|---|---|
| `w_rmse` | -0.002827 | 49/50 | [-0.003468, -0.002228] | BETTER |
| `w_deb_rmse` | -0.001959 | 49/50 | [-0.002428, -0.001532] | BETTER |
| `w_rank_corr` | +0.032099 | 33/50 | [+0.011774, +0.053159] | BETTER |
| `w_deb_rank_corr` | +0.044870 | 38/50 | [+0.023135, +0.066495] | BETTER |
| `top_decile_recall` | +0.023107 | 27/50 | [+0.001165, +0.045631] | inconclusive (see below) |
| `g_rank_corr` | -0.001114 | 24/50 | [-0.002873, +0.000627] | inconclusive |
| `g_rmse` | +0.002825 | 20/50 | [+0.000745, +0.004905] | WORSE |
| `conflated_rank_corr` | -0.001301 | 23/50 | [-0.003079, +0.000460] | inconclusive |
| `paired_state_mse` | -0.000995 | 38/50 | [-0.001364, -0.000632] | BETTER |
| `member_value_rmse` | -0.005953 | 39/50 | [-0.008413, -0.003496] | BETTER |
| `split_distortion` | +0.229875 | 47/50 | [+0.165719, +0.303348] | moves toward 1 |

## The pre-registered question: does `top_decile_recall` resolve at N=50?

No -- not by the rule fixed before the run. The mean paired delta is now
distinguishable from zero (CI [+0.0012, +0.0456] excludes 0, +0.023 on a
~0.40 base), but only 27/50 seeds are positive, far below the >= 35/50
seed-count bar, so the endpoint is reported **inconclusive at N=50**. The
effect, if real, is small and driven by a minority of seeds. State it as: the
recall coordinate is not something this loss reliably moves seed-by-seed; it
stays a "not established" endpoint, with a weak positive mean that only just
clears zero at 50 seeds. A further extension would need a new registration.

## Stability vs N=30 (all other endpoints)

- Magnitude claim strengthens: `w_rmse` 49/50 (was 29/30).
- w-rank gain confirmed: `w_deb_rank_corr` 38/50, CI excludes 0 (was 24/30).
- Aleatoric reallocation stable: `g_rmse` worse; `g_rank_corr`/conflated
  inconclusive at 50 (was worse at 30) -- direction unchanged, size shrinking.
- `split_distortion` moves toward 1 on 47/50: the loss reallocates the split,
  exactly as the identifiability write-up predicts.
- Identified vs hybrid head-to-head at N=50 (vs `decision_hybrid`):
  `w_rank_corr` -0.0051, 27/50, CI includes 0 -> ranking **wash**;
  `w_rmse` +0.0004, 8/50 -> slightly WORSE magnitude at 2x cost;
  `member_value_rmse`, `paired_state_mse`, `split_distortion` also worse.
  The stationary fixed map still cannot discriminate the two objectives --
  the policy 2x2 remains the discriminator (Row 1: lagging, not
  identifiability).
- `decision_identified_nostate` vs `decision_hybrid` remains worse on the w
  endpoints (state-geometry terms carry the gain), unchanged from N=30.

## Registration hygiene

- Protocol and thresholds were written into this file and committed
  (`4027dde`) before seeds 30..49 were launched; verdicts were read after.
- Seeds 0..29 rows are the published 30-seed file untouched (byte-identical
  superset check passed at merge time); no arm or seed dropped post-hoc.
