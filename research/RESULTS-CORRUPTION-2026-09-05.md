# Corruption-probe results + weighting correction: the identified-loss policy collapse was an EMA artifact (2026-09-05)

**Registration:** `research/CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md`
(written before any arm ran).
**Data:** `runs/probe_u_corrupt_{schedule,maxt,pure}_s{0,1}.json`,
`runs/probe_u_corrupt_{wonly,ema}_s{0,1}.json`,
`runs/probe_u_n10_{eq,ema,emalr1}_s{0..9}.json` (N=10 batch; eq/ema seeds 0-1
are the probe rows above; emalr1 seeds 0-1 were filled with the identical
batch config after the batch, before analysis). Summarizer:
`udwm/scripts/summarize_corruption_probe.py`.
**Probe:** `udwm/scripts/probe_u_collapse.py`, 1,800 env steps, 180 student
updates, exact matched teacher per seed, CPU.

## Config-override bug (registered with this experiment)

The probe's `--set` overrides were applied to the base config **before**
`make_cfg`, and `VARIANTS[variant]` re-applies `distill_reweight_ema=True`
(and `distill_aleatoric_weight=1.0`) afterwards. Every historical
"identified_*" mechanism/leverage arm that was labelled "equal weight",
"w-only", "g-only", "lambda_g = 1e4" or "M = 8" therefore executed with the
**EMA reweighting on** and the variant-default weights; only the flags not
present in `VARIANTS` (`distill_grad_clip`, `--student-lr`) actually varied.
Saved scale logs confirm: `w_scale ~ 1e12`, `g_scale ~ 2.3e8` at step 300 of
the B2 file. The probe is fixed (overrides applied after `make_cfg`) and the
historical claims below are re-read under the corrected semantics.

## Fresh contrast (equal weight vs EMA-both vs w-only; schedule corruption)

Same seed-0/1 teacher pairing, live SAC critic, M=2 latents; only the loss
reweighting differs (`schedule` decision corruptions in all three):

| arm (equal-weight identified, s0/s1) | s_u | t_u | u_rank | s_g | t_g | s_w | t_w | s_delta-spread | next_state_mse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| schedule s0 | -76.2 | -90.6 | 0.885 | 76.6 | 90.6 | 0.40 | 0.007 | 0.261 | 0.243 |
| schedule s1 | -90.9 | -95.2 | 0.920 | 91.2 | 95.2 | 0.31 | 0.008 | 0.284 | 0.270 |
| EMA-both s0 (historical config) | ~0.00 | -92.7 | -0.094 | 0.01 | 92.7 | 0.014 | 0.008 | 0.007 | 0.168 |
| EMA-both s1 | ~0.00 | -49.4 | 0.004 | 0.01 | 49.4 | 0.008 | 0.005 | 0.006 | 0.167 |
| w-only EMA s0 | 0.00 | -82.7 | -0.132 | 0.01 | 82.7 | 0.013 | 0.006 | 0.006 | 0.165 |
| w-only EMA s1 | -0.00 | -57.0 | 0.184 | 0.01 | 57.0 | 0.009 | 0.003 | 0.006 | 0.154 |

Rows: `s_u/t_u` = eval-time student/teacher local-u mean; `s_g/t_g` =
student/teacher within-member value spread; `s_delta-spread` = student
latent-to-state spread over 8 paired pure-noise latents (teacher 0.36-0.38).

**Equal-weight identified recovers the aleatoric channel and the decision
object on the live critic** (u_rank 0.885/0.920; student g within ~1.2x of
teacher; student u within ~0.85-0.95x of teacher). **EMA-both and w-only EMA
annihilate g** (student g ~0.01, delta-spread ~0.006, u-rank ~ noise) exactly
reproducing the N=30 policy collapse. The driver is the **epistemic
up-weight** (`w_scale ~ 1e5-1e6`), present in both EMA variants, not the
aleatoric down-weight: w-only EMA (aleatoric at weight 1) collapses just the
same. The equal-weight recovery has the known counterpart hole: student w is
~50x inflated (0.31-0.40 vs teacher ~0.007), the g >> w weighting hole, and
next_state_mse degrades mildly (0.24-0.27 vs EMA's 0.15-0.17).

## Corruption distribution is not the binding constraint (registered outcome 2)

Equal-weight identified at every decision-corruption level (2 seeds):

| arm | seed | u_rank | s_u | t_u | s_g | t_g | s_delta | next_state_mse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| schedule | 0 | 0.885 | -76.2 | -90.6 | 76.6 | 90.6 | 0.261 | 0.243 |
| schedule | 1 | 0.920 | -90.9 | -95.2 | 91.2 | 95.2 | 0.284 | 0.270 |
| maxt | 0 | 0.899 | -145.0 | -155.2 | 145.4 | 155.3 | 0.252 | 0.254 |
| maxt | 1 | 0.631 | -40.1 | -56.1 | 40.5 | 56.1 | 0.242 | 0.249 |
| pure | 0 | 0.729 | -109.0 | -135.4 | 110.2 | 135.4 | 0.260 | 0.384 |
| pure | 1 | 0.769 | -61.2 | -69.4 | 62.1 | 69.4 | 0.250 | 0.332 |

Note: teacher stats differ across arms within a seed because the real buffer
is re-collected under each arm's policy (evaluation states differ); the
student-vs-teacher ratio on the same states is the comparable quantity.
Verdict: g leaves the boundary at **every** corruption level, so the
"remaining lever is the corruption distribution" claim from the leverage-fix
doc (which was inferred from EMA-on runs) is refuted: the schedule-corruption
aleatoric target was never too weak to lift g once the EMA down-weighting was
removed. `pure` additionally degrades next_state_mse (0.33-0.38): evaluating
the decision terms at full-noise latents pushes the student's spread beyond
what keeps its one-step dynamics accurate. No corruption-mode arm is worth
carrying to policy.

## What the mechanism/leverage threads now say

- The U-collapse mechanism doc measured per-update aleatoric "leverage"
  ~1e-6 and estimator noise ~100x signal on runs that all had the EMA on;
  those numbers describe the EMA-down-weighted channel, not the raw aleatoric
  term. Under equal weights the aleatoric term lifts g to ~teacher within the
  180-update run (student g tracks teacher from ~step 800, `schedule_s0` log).
- The leverage-fix A/B arms (lambda/clip/LR) likewise ran EMA-on with the
  variant-default aleatoric weight; the `lambda_g` and "equal-weight" labels
  in `RESULTS-LEVERAGE-FIX-2026-09-05.md` did not execute as documented. The
  empirical endpoint (student LR 0.1 + clip 1e3 drives g to exactly 0) is
  replicated at N=10 below under the corrected attribution (EMA-on config).
- The balance picture is reproduced in the repo's own ground-truth toy
  (`runs/ground_truth_w_g.json` re-adjudicated in
  `theory/identified_balance_window.py`): equal-weight identified recovers g
  (g_rank 0.81-0.97) but leaves w at the hole (w_rank 0.40 when g* >> w*);
  EMA-both fixes w (0.97-0.99) and annihilates g (g_rank ~0.1-0.4, g_hat
  ~0.03-0.4 vs g* 46-3966). The earlier ground-truth write-up adjudicated
  only w and called EMA "the fix"; that was the same blind spot.

## N=10 replication (paired seeds 0-9; batch seeds 2-9 appended)

Same probe protocol as the 2-seed contrast (1,800 env steps, 180 student
updates, exact matched teacher per seed, M=2 latents, schedule decision
corruptions). Seeds 2-9 were appended with the same arms and the same
registered endpoints (nothing added after reading rows) to tighten the two
headline claims; seeds 0-1 are the rows above.

| arm | seed | u_rank | s_g | t_g | s_w | t_w | s_delta | s_mse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eq | 0 | 0.885 | 76.60 | 90.60 | 0.399 | 0.007 | 0.261 | 0.243 |
| eq | 1 | 0.920 | 91.16 | 95.24 | 0.309 | 0.008 | 0.284 | 0.270 |
| eq | 2 | 0.757 | 42.29 | 54.28 | 0.237 | 0.007 | 0.253 | 0.253 |
| eq | 3 | 0.900 | 44.89 | 61.68 | 0.166 | 0.003 | 0.263 | 0.235 |
| eq | 4 | 0.772 | 52.64 | 67.06 | 0.330 | 0.012 | 0.231 | 0.223 |
| eq | 5 | 0.870 | 64.85 | 77.15 | 0.203 | 0.005 | 0.260 | 0.243 |
| eq | 6 | 0.815 | 54.51 | 62.05 | 0.262 | 0.005 | 0.261 | 0.263 |
| eq | 7 | 0.635 | 45.14 | 88.68 | 0.444 | 0.006 | 0.213 | 0.236 |
| eq | 8 | 0.858 | 30.63 | 41.97 | 0.246 | 0.005 | 0.274 | 0.244 |
| eq | 9 | 0.865 | 67.06 | 75.60 | 0.231 | 0.006 | 0.253 | 0.244 |
| ema | 0 | -0.094 | 0.01 | 92.74 | 0.014 | 0.008 | 0.007 | 0.168 |
| ema | 1 | 0.004 | 0.01 | 49.38 | 0.008 | 0.005 | 0.006 | 0.167 |
| ema | 2 | 0.173 | 0.01 | 67.08 | 0.011 | 0.006 | 0.006 | 0.174 |
| ema | 3 | 0.281 | 0.02 | 52.15 | 0.007 | 0.006 | 0.006 | 0.164 |
| ema | 4 | -0.047 | 0.01 | 89.81 | 0.025 | 0.020 | 0.006 | 0.165 |
| ema | 5 | 0.098 | 0.02 | 77.33 | 0.009 | 0.006 | 0.006 | 0.173 |
| ema | 6 | 0.150 | 0.01 | 61.81 | 0.005 | 0.003 | 0.006 | 0.161 |
| ema | 7 | 0.352 | 0.01 | 85.27 | 0.005 | 0.006 | 0.006 | 0.170 |
| ema | 8 | 0.063 | 0.02 | 69.44 | 0.008 | 0.006 | 0.006 | 0.166 |
| ema | 9 | 0.181 | 0.01 | 51.67 | 0.010 | 0.003 | 0.006 | 0.161 |
| emalr1 | 0 | -0.044 | 0.00 | 336.2 | 0.025 | 0.036 | 0.000 | 0.178 |
| emalr1 | 1 | -0.385 | 0.00 | 102.7 | 0.011 | 0.006 | 0.000 | 0.168 |
| emalr1 | 2 | -0.266 | 0.00 | 163.5 | 0.026 | 0.020 | 0.000 | 0.169 |
| emalr1 | 3 | -0.181 | 0.00 | 46.4 | 0.002 | 0.007 | 0.000 | 0.166 |
| emalr1 | 4 | -0.085 | 0.00 | 151.0 | 0.007 | 0.020 | 0.000 | 0.166 |
| emalr1 | 5 | 0.028 | 0.00 | 115.2 | 0.011 | 0.010 | 0.000 | 0.156 |
| emalr1 | 6 | -0.309 | 0.00 | 351.5 | 0.287 | 0.040 | 0.000 | 0.163 |
| emalr1 | 7 | -0.609 | 0.00 | 872.8 | 0.500 | 0.099 | 0.000 | 0.185 |
| emalr1 | 8 | 0.047 | 0.00 | 134.5 | 0.011 | 0.012 | 0.000 | 0.172 |
| emalr1 | 9 | -0.395 | 0.00 | 301.1 | 0.130 | 0.028 | 0.000 | 0.171 |

Teacher stats differ across arms within a seed because the real buffer is
re-collected under each arm's policy (evaluation states differ); the
student-vs-teacher comparison on the same states is the valid contrast.

**Paired eq - ema (n=10; mean delta, wins/10, 5,000-draw bootstrap 95%):**

- u_rank: +0.712, 10/10, CI [+0.589, +0.819] - equal weight recovers the
  decision object's ranking vs EMA-both on every seed (eq u_rank 0.635-0.920;
  ema -0.094..0.352, noise).
- student g: +57.0, 10/10, CI [+46.8, +67.8]; student g is 0.51-0.96x teacher
  (median 0.81; seed 7 is the weak seed at 0.51x), i.e. the aleatoric channel
  recovers to ~teacher scale, not to machine precision.
- next-state MSE: +0.079 worse (eq 0.22-0.27 vs ema 0.16-0.17), 10/10, CI
  [+0.071, +0.088] - the documented counterpart cost of equal weighting.
- student w: eq/teacher ratio 28-69x (mean 47x) - the equal-weight hole on w
  is stable at N=10, not a 2-seed accident.

**Headline B2 replication (corrected attribution):** emalr1 (EMA-on, grad
clip 1e3, student lr 0.1) pins student g at exactly 0.0 on 10/10 seeds (max
0.0) with u-rank noise (-0.61..+0.05; student u ~ 0-0.5 vs teacher -46..-873).
The "student lr 0.1 drives g to exactly 0" endpoint of
`RESULTS-LEVERAGE-FIX-2026-09-05.md` holds at N=10 under the EMA-on
attribution. ema (lr 1e-3) sits just above the boundary (g ~0.01, CI of the
difference ema - emalr1 excludes 0).

Verdict at N=10: the 2-seed contrast is confirmed on the registered endpoints
- EMA-both annihilates the aleatoric channel (u-rank ~ noise) while
equal-weight identified recovers it (10/10 positive u-rank, CI excludes 0),
with the stable counterpart holes (w 47x inflated; next-state MSE +0.08).
This is a probe-scale mechanism claim (no policy-return claim; the N=10
extension tightens the headline, it does not change the registered
N=2 diagnostic's status).

## Files / doc updates

- `RESULTS-LEVERAGE-FIX-2026-09-05.md`: addendum (this correction).
- `U-COLLAPSE-MECHANISM-2026-09-05.md`: addendum (collapse attribution).
- `RESULTS-GROUND-TRUTH-W-G-2026-09-01.md`: addendum (re-adjudication).
- `PAPER-NARRATIVE.md` / `RESULTS.md`: corrected storyline.
