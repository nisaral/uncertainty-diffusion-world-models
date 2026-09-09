# Summary: what this project is, and where it stands

A plain-language walkthrough. For the precise tables and every adjudicated
number, follow the links in the [README](README.md) and the result index
[`research/RESULTS.md`](research/RESULTS.md). This file exists so a first-time
reader can get the story before the statistics.

**Visual entry points:** interactive verdict explorer
[`visuals/v4-explorer.html`](visuals/v4-explorer.html) - 90-second story, EMA-collapse curves,
ordering flip, per-arm tables (self-contained, works offline / on GitHub Pages). Static figures
for the repo main page: [`visuals/svg/v1-identifiability-skip.svg`](visuals/svg/v1-identifiability-skip.svg),
[`visuals/svg/v3-cross-env-ordering.svg`](visuals/svg/v3-cross-env-ordering.svg).

## The problem in one paragraph

Diffusion world models are good simulators of an environment, but slow: one
prediction costs many denoising steps. To use one inside an RL training loop
you distill it into a one-step "student". That is fine for *mean* prediction,
but the open question is whether the student still knows when it is unsure.
RL algorithms that use imagined rollouts (MBPO-style) decide how much to
trust the model based on exactly that uncertainty signal, so a student that
*silently* loses it - while still predicting well on average - is a trap: it
feeds confident-sounding but wrong imagined data into policy training.

## The claim

> Matching teacher-student value disagreement at a single shared latent does
> not identify epistemic uncertainty. The matched statistic is one number
> that mixes two different kinds of uncertainty: epistemic ("I haven't seen
> enough") and aleatoric ("this is inherently random"). A student can match
> the mixed number while completely losing the part a decision needs.

The repo is a falsifiable test of that claim about the *loss*, not a claim
about making RL faster or better on its own. It is a laboratory for one
mechanism, measured twice.

## The story, in order (what to read, and why)

1. **The mathematical trap** - `theory/distill_identifiability.py` +
   [`research/RESULTS-IDENTIFIABILITY-2026-08-29.md`](research/RESULTS-IDENTIFIABILITY-2026-08-29.md).
   The usual "decision-aware" distillation term matches one scalar,
   `w* + (g* - Sigma_bar)`: one equation, two unknowns. Four exact students,
   including one with *zero* epistemic uncertainty, all match the teacher to
   ~1e-6. Drawing M>=2 latents and matching the two components separately
   fixes the identifiability of the *target*.

2. **Fixed value map: the loss reallocates uncertainty, it does not
   preserve it** - N=50 on a fixed benign value map. The decision-aware loss
   improves the epistemic magnitude (49/50 seeds) at the cost of the
   aleatoric component. The loss is moving variance between the two terms,
   not preserving the split.

3. **Live critic: the transfer claim is falsified** - N=30 policy 2x2.
   Dropping the same loss into an online MBPO/SAC loop with the live critic
   as the value map: uncertainty rank correlation collapses (0/30 seeds
   better than the naive baseline), next-state MSE improves 30/30. "Drop
   this loss into MBPO and keep uncertainty" is falsified at 30/30.

4. **Two candidate fixes, one real bug found on the way** - Lagging the
   value-map critic fixes the nonstationarity failure (30/30). Separating w
   and g with an EMA-style per-term reweighting *looked* like it fixed the
   equal-weighting hole - until a probe showed it quietly annihilates the
   aleatoric channel (student aleatoric spread pinned near zero; u-rank at
   noise level). That was an implementation attribution bug in our own
   pipeline, found by mechanism tracing, with the config bug fixed and the
   earlier rows re-adjudicated.

5. **The corrected weight: partial transfer** - Equal-weight identified
   (no EMA) transfers to the live critic at policy scale (u-rank 0.835,
   9/10 seeds >= 0.70, beats the conflated arm 10/10), but sits below the
   plain baselines - partial, not parity.

6. **The combined fix: full transfer at parity (DelayedBimodal)** - The
   never-run cell {lagged critic} x {equal-weight identified}:
   `lagged_identified_eq` (2026-09-07, N=10) is top of the table at u-rank
   0.948 (10/10 >= 0.70), closes the equal-weight arm's confirmed deficit vs
   the naive baseline entirely, and closes most of its w-magnitude hole.
   Every top arm is near the measurement ceiling on this benchmark, so the
   remaining "does it *beat* the single fixes" question can only be answered
   where there is headroom - the DMC environment.

7. **Cross-environment check (in progress)** - DMC/hopper-hop 10-seed sanity
   (2026-09-07): the EMA-collapse mechanism replicates across environments
   (equal-weight beats EMA by +0.42 on u-rank, 10/10), and the arm ordering
   replicates, but every arm loses ~half its u-rank level, and 3,600 steps is
   only 3.6 of the 1,000-step episodes. The 30-seed adjudication is held
   pending a registered budget probe that discriminates "does not transfer"
   from "the measurement is degenerate at this budget". The DMC verdict is
   not yet in; do not quote the sanity as adjudicated.

8. **Zero-compute re-analyses (2026-09-07)** - Two hypotheses from the data
   were checked against existing rows. The logged decision-relevant "tail"
   columns do not rescue the story (no arm preserves tail risk flagging at
   these budgets; the EMA pathology is *muted* on tail metrics, so tail
   metrics alone would understate it). And the equal-weight benefit is
   operating-point-bound, not tied to naive-loss instability across
   environments.


9. **Cross-environment verdict (2026-09-08)** - The 30-seed x six-arm
   adjudication at the amended 15k budget completed. Controls pass 30/30:
   the EMA collapse reproduces cross-environment (mean -0.076, 0/30 >= 0.70)
   and equal-weight identified clears it (+0.788, 30/30) - the mechanism is
   general. The practical ordering is not: both registered eq bars fail
   (15/30 >= 0.70; eq - hybrid is a wash, 13/30), the combined-fix arm lands
   below eq on DMC (-0.073, 7/30, confirming the n=2 reversal at N=30), and
   the plain self-gated ordinary arm tops the table (0.957, 30/30). Gate-off
   control is return-neutral at 15k; returns stay deferred. Full record:
   research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md

## What the numbers are not

- Not a new world model. Not conformal coverage (`sqrt(U)` is a score, not a
  calibrated interval). Not a policy/SOTA claim. Not a claim that gated
  imagination is new (it is not - see MACURA).
- Small benchmarks: toy/DelayedBimodal fully adjudicated; DMC/hopper-hop
  adjudicated at 15k (2026-09-08: mechanism controls 30/30; transfer-ordering
  bars not met). No pixels.

## How to reproduce

```bash
pip install -r requirements.txt
python -m udwm.scripts.smoke_test
python -m pytest tests/test_core.py tests/test_ground_truth_w_g.py -q
python theory/distill_identifiability.py   # the identifiability construction
python theory/ground_truth_w_g.py          # analytic (w*, g*) check
```

Every adjudicated policy table is reproducible with the exact runner
commands recorded in the result docs and `REPRODUCE.md`. Rows use exact
per-seed teacher checksum pairing (same teacher, same buffer for every arm
of a seed); GPU rows are kept separate from CPU rows because they are not
bit-identical.

## Repo map (10-second version)

| Piece | Where |
|---|---|
| Losses: ordinary / hybrid / identified / combined + guard | `udwm/models/consistency.py` |
| Distilled world model, teacher freeze | `udwm/models/world_model.py` |
| Lagged target-critic value map, MBPO/SAC trainer | `udwm/rl/trainer.py` |
| u / w / g teacher-student metrics | `udwm/eval/metrics.py` |
| Policy runners and adjudication summarizers | `udwm/scripts/` |
| Theory (identifiability, estimator bias/variance, balance window) | `theory/` |
| Every result + its pre-registration | `research/` (index: `research/RESULTS.md`) |

## Status board (2026-09-08)

- Identifiability theorem: proven, with exact-construction verification.
- Fixed-map stress N=50: adjudicated. Falsified live-critic transfer: N=30,
  adjudicated. EMA attribution bug: found, fixed, re-adjudicated.
- Equal-weight identified transfer: N=10 bars met on DelayedBimodal
  (partial, below parity); DMC 30-seed verdict (2026-09-08): eq clears the
  EMA collapse control 30/30 but fails both position bars (eq - hybrid wash;
  -0.246 below ordinary).
- Combined fix: N=10, top of table at parity (DelayedBimodal); on DMC the
  combined arm is below eq (-0.073, 7/30) - a DelayedBimodal-scale result;
  ordinary is the top u-rank arm on hopper-hop at 15k (0.957, 30/30).
- Paper: not yet a submission. GitHub: this repo. Hugging Face: diagnostic
  checkpoint artifact released (teacher + students incl. the broken and fixed
  arms; DelayedBimodal seed 0):
  https://huggingface.co/nisaralll/udwm-identifiability-diagnostic

Pre-registration discipline is the reason to trust the tables: every
adjudicated endpoint, bar, and wins/N rule was written down before the rows
ran, including the contingency rule that was actually triggered when a
bit-exactness gate failed. That is a feature of the process, not a
workaround.