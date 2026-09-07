# Identifiability-frontier formalization: theorem statements verified (2026-09-07)

**What this is.** The formal upgrade of the identifiability claim
(`theory/distill_identifiability.py`, PAPER-NARRATIVE Section 2) from a
verified identity into a theorem set with proofs
(`research/proofs/identifiability-frontier.md`) and a deterministic numeric
verification of every statement (`theory/identifiability_frontier.py`, all
checks PASS, seeded). This is **mathematical verification, not an empirical
hypothesis test**: no seeds-based adjudication, no bootstrap bars; the
statements are population-level and the script confirms them to MC tolerance.

## Statements verified (and what each adds to the paper)

1. **Theorem 1 - the fibre and the missing sign.** For any feasible teacher
   (`S* > 0`, coupling `rho < 1`), the M = 1 zero-loss set contains students
   realising every `u` in `[-S* N/((N-1)(1-rho)), S*)`; the decision statistic
   is never sign-identified. Verified on four regime teachers, including the
   measured DelayedBimodal map (`w* ~ 1.5e-7`, `g* = 80.1`, `rho = 0.9998`):
   fibre `u_s` range `[-80.10, +0.005]` vs teacher `u* = -80.10` - the
   collapse student (`g_s ~ 0`, `u_s > 0`) and the teacher's decision object
   have **opposite signs** at zero M = 1 loss.
2. **Theorem 2 - coupling erosion.** `dS/dg = (1-rho)(N-1)/N`; at the measured
   coupling the M = 1 statistic is `S*/|u*| = 2.4e-4`, i.e. the matched
   quantity is ~4000x below the decision object on exactly the maps where
   `u ~ -g` determines the decision. The "decision-aware" term is blind to
   the channel that determines `u` where it matters most.
3. **Proposition 3 - the sticky boundary.** The separated aleatoric mismatch
   has gradient `O(g_s)` at `g_s = 0` (zero exactly at the boundary, any
   smooth parameterisation); per-state `std(g_hat)/g* = 0.76` at `M = 2`
   (0.43 at M = 4, 0.28 at M = 8). No multiplicative dial can create an
   escape force: at `lambda_g = 1e4` (the largest actually tried in the
   leverage-fix A arms) the escape force at `g_s ~ 0` is `~1e-3` of the noise
   floor. Explains why every A/B leverage arm stayed pinned at `g ~ 0` and
   why recovery needs a coupled channel whose aleatoric gradient does not
   vanish (the equal-weight arm's member-MSE anchor).
4. **Theorem 4 - sufficiency and the M rule.** M >= 2 with the coupling-aware
   debias has a unique population minimiser at the truth (`(w*, g*)`, hence
   `u*` identified); the M = 2 loss of the `w = 0` fibre collapse is 0.092
   while its M = 1 loss is ~1e-33. Teacher-noise floor gives the design rule
   `M >~ (A_w + A_g)/(B (Delta_w^2 + Delta_g^2))`; the repo's `M = 2` is a
   floor, and kurtotic value laws push it up.
5. **Theorem 5 - reweighting cannot rescue the fibre.** Any strictly positive
   multiplicative weighting leaves the fibre (and the sticky boundary)
   intact; the inverse-variance EMA ratio is `(w*/g*)^2`, which on the
   measured map down-weights the aleatoric term by `7.0e-9` - the
   "balanced" loss is the w-only loss to eight orders of magnitude exactly
   where `u ~ -g`.
6. **Theorem 6 - scale-robust gating.** Percentile thresholds are invariant
   to strictly increasing score transforms (verified: agreement 0.9999 on the
   refusal set under `1e-4` scale collapse); absolute thresholds stop nothing
   under the same collapse (agreement 0.85, all refusals lost). Fixes the
   payoff protocol: percentile gating only, held fixed across arms.
7. **Theorem 7 - moving-target composition (added same day, after the
   combined-fix N=10 table).** The student's residual against the eval-time
   map decomposes into an optimization gap at the map frozen over the
   distillation window plus the map drift over the window; the equal-weight
   M>=2 loss removes the first term at any fixed map (flow converges from any
   start: u endpoints -2.0000/-2.0000 vs u* = -2) and the lagged map removes
   the second (residual = drift to 1.4e-15 median on 200 drift draws). The
   M=1 hybrid flow on the same frozen map lands on the S-fibre at zero loss
   with an initialization-dependent decision statistic (u endpoints +0.059 /
   -5.676) - fibre memory, so lagged-hybrid parity (0.946, N=10) is anchor-
   and-init carried, not a loss-level guarantee. Measured arm-to-term mapping
   and the open moving-map-coupling conjecture are in the proof doc.

## Protocol audit note (separate from the theory)

`git log` on `configs/delayed_bimodal_distill.yaml` shows `u_gate.mode: both`
with `enable_after_steps: 900` has been in the base config since creation
(2026-08-23), and the policy/payoff runners do not strip it: every arm in
every N-study on this config trained with percentile stop+weight gating
(training-side UBE net) from step 900 onward (50% of a 1,800-step run, 75% of
a 3,600-step run). Arm comparisons are unaffected (all arms gated
identically), and all uncertainty endpoints are measured at eval time on
real-buffer states, not gated. But any prose that describes these runs as
"ungated MBPO/SAC" is inaccurate, and the "no payoff" reading is a statement
about *self-gated* training. No table is re-adjudicated here; this note
exists so the paper describes the protocol exactly as run and so the DMC
payoff design makes the gate an explicit arm axis.

## How to reproduce

    python theory/identifiability_frontier.py

Deterministic (fixed seed), numpy only, ~5 s (Parts 1-7). All checks PASS
(2026-09-07; full re-run after the Theorem 7 / Part 7 addition).

## Addendum (2026-09-07): Proposition 8 verified (Part 8 added)

Formal statement in `research/proofs/identifiability-frontier.md`
(Proposition 8): the M=1 population risk depends on the student pair
`(w_s, g_s)` only through the single-latent statistic `S_s = w_s + c g_s`, so
its sublevel sets are S-strips and the decision-statistic uncertainty does
not shrink as the population risk goes to zero; the M>=2 equal-weight loss
identifies with rate `sqrt(eps)` and tight coefficient `sqrt(2)`.

Part 8 of the verifier (deterministic, fixed seed, run 2026-09-07 after the
rows of the G7 control existed but independent of them):

- P8a PASS: M=1 zero-loss u-uncertainty spans >= 2S* with a sign flip and
  worst-case decision error >= S* for every regime teacher (measured
  DelayedBimodal map: S* = 0.0128, fibre span 80.12, worst case 80.12).
- P8b PASS: M>=2 sublevel disk bound tight: at eps = 1e-6 the sampled max
  |u - u*| = 0.001414 = sqrt(2 eps) exactly.
- P8c PASS: at matched eps = 1.6e-14 the M=1 worst-case error stays ~ the
  fibre span (80.1) while M>=2 is <= 1.8e-7; ratio >= 4.4e8. The gap is
  unbounded as eps -> 0.

Runtime ~8 s for Parts 1-8. Full re-run: all checks PASS.
