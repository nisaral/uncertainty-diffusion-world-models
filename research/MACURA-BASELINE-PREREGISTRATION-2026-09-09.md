# MACURA head-to-head baseline - registration (2026-09-09)

**Status: registered 2026-09-09, before any macura_gate row runs and before
the implementation is adjudicated. Mechanism section corrected 2026-09-09
(paper audit, arXiv:2405.19014 Sections 4-6): the earlier draft mislabelled
the MACURA signal as Jensen-Shannon divergence; the paper defines member-KL
uncertainty u_KL (Eq. 4) and implements a closed-form geometric-JS surrogate
u_GJS (Eqs. 15-20) for its Gaussian PNN ensemble. The repo estimator and this
registration now state the Eq. 4 quantity directly (see Section 1).**
Field-baseline item of the master plan (Part 3.5 / Part 4 item 3). This study
NEVER re-adjudicates the DMC payoff bars or the DelayedBimodal transfer bars;
it adds the missing direct field-baseline comparison for the paper's
discussion section.

## 1. Mechanism being implemented (exact)

MACURA, per Frauenknecht, Eisele, Subhasish, Solowjow & Trimpe, ICML 2024,
PMLR 235:13973-14005 (arXiv:2405.19014), adapts the model-rollout length of
an ensemble dynamics model per state from the ensemble's **epistemic
uncertainty**:

- Eq. (4): u_KL(s, a) = sum_e D_KL( p~_{theta_e}(. | s, a) || p~_PE(. | s, a) )
  with p~_PE(. | s, a) = (1/E) sum_e p~_{theta_e}(. | s, a) - each member's
  predictive distribution against the equal-weight ensemble mixture. MACURA
  keeps rolling where members agree (u small) and stops where they disagree.
- The paper's implemented estimator, u_GJS (Eqs. 15-20), averages a pairwise
  *geometric* Jensen-Shannon divergence between the members' predictive
  Gaussians N(mu_e(s,a), Sigma_e(s,a)); it exists only because MACURA's
  members are Gaussian PNNs with closed-form densities, so the KL-to-mixture
  sum of Eq. (4) is replaced by a closed-form surrogate.
- Admissible set (Eq. 20): E = { s : u(s, a) < kappa, a ~ pi(. | s) }, the
  states where model rollouts are trusted.
- Rollout rule (Algorithm 2): a branched transition (s_t, a_t, r, s'_{t+1}) is
  added to D_mod iff u(s_t, a_t) < kappa; otherwise the rollout breaks. A
  maximum horizon T_max caps error accumulation.
- Threshold self-tuning (Eq. 21): after the first prediction step of each
  round of M parallel rollouts, the base uncertainty u_hat_k is the zeta-
  quantile (default zeta = 95%) of that round's M first-step uncertainties,
  and kappa = (xi / K) sum_{k<=K} u_hat_k over rounds, with xi a single tunable
  scaling hyperparameter (default 1). Paper defaults: T_max = 10, zeta = 95%.

No w/g/u decomposition is produced - MACURA is a gating signal, not an
uncertainty estimator.

This repo's implementation (`udwm/uncertainty/macura_baseline.py`) computes
the **Eq. (4) quantity itself** - u_KL = sum_e D_KL(p~_e || p~_PE) - for the
repo's implicit diffusion members, which expose no closed-form Gaussian
density (so the paper's u_GJS closed form is inapplicable). Each member's
predictive next-state distribution is represented by its conditional Monte
Carlo draws, estimated on the pooled support of all members by a per-state
Gaussian-kernel density (bandwidth = mean within-member median pairwise
distance) and normalized to a categorical; u_KL is the member-KL sum over
those categoricals. The plugin identity u_KL = E * JSD(p_1, ..., p_E) (the
multi-distribution JSD of the same categoricals) is what the earlier draft
labelled "JSD"; the returned value is the Eq. (4) sum, the quantity the
admissible set (u < kappa) is defined on. (Plugin identity independently
verified 2026-09-09: a hand-worked 2-member categorical example, an E=3
kernel-plugin density check, and a shipped-function-vs-replication check
all pass in tests/test_macura_ukl_identity.py.) The signal plugs into the
trainer's
existing `score_fn` slot in `u_gated_rollout`
(`udwm/rl/u_gated_imagination.py`), the same stop/weight machinery the UBE
u-gate uses - a new gating rule on the existing ensemble outputs, not a new
model.

### 1.1 Registered deviations from Algorithm 2 (deliberate, fixed before
results)

The repo does not reproduce MACURA's pipeline end-to-end; it answers a
narrower question: does the MACURA *signal* (Eq. 4 u_KL) beat the learned
UBE u when both drive the repo's registered gate? Every deviation below is
deliberate, is applied identically to `macura_gate` and to the internal
comparison arms (`ordinary`, `identified_eq`, ...), and sits outside the
adjudicated endpoints (paired internal contrasts only). The deviations set
the shared data-mix context; none can manufacture or suppress a
`macura_gate`-vs-internal difference, which is the only quantity this study
reports. Reproducing MACURA's absolute numbers is explicitly out of scope:
no comparison against the MACURA paper's reported returns is made anywhere.

- **Per-round kappa only (Eq. 21 without the cross-round running average;
  xi = 1).** The paper averages the per-round base uncertainty over rounds
  "to stabilize kappa over iterations". The repo trainer gates each
  imagination round with that round's first-step quantile. Because
  `ordinary`/`identified_eq`/`macura_gate` share the identical rule, the
  stabilization device cannot affect an arm-vs-arm contrast; it only smooths
  the absolute kappa trajectory, which this study never reports.
- **Threshold percentile inherited from the registered environment
  calibration (`stop_percentile` = 0.85), not MACURA's zeta = 95%.** The
  self-gated percentile is the registered shared rule for every arm in the
  DMC study. Importing zeta = 95% for `macura_gate` alone would confound
  "signal quality" with "stop aggressiveness": a looser percentile retains
  more of the first-step distribution regardless of which signal is used.
  Same rule, only the signal differs, is the cleanest possible head-to-head.
  MACURA's zeta = 95% is a default tuned in its own pipeline; importing
  pipeline defaults that break arm symmetry is deliberately avoided.
- **Score is sqrt(u) before thresholding (shared `use_sqrt` default).** Not
  a behavioral deviation: the threshold is a quantile of the batch scores,
  and quantiles commute with strictly monotone transforms, so
  `sqrt(u) > q_p(sqrt(u))` iff `u > q_p(u)`. The stop rule is ordinally
  identical to comparing raw u against the same percentile of raw u.
- **The terminating (u >= kappa) transition is stored as a terminal
  (done=1) instead of being discarded (Algorithm 2 breaks before adding).**
  This is a property of the shared `u_gated_rollout` machinery used by every
  arm, including the UBE arms: the step that crosses the threshold is kept
  with done=1 and every later step carries weight 0 (never enters the model
  buffer). The effect is one arm-symmetric terminal per stopped rollout and
  cannot differentially favor either signal.
- **Soft weights coexist with the hard stop (mode "both", weight =
  exp(-beta * score) clamped to [0.05, 1]), whereas Algorithm 2 is hard
  accept/reject.** Stop + weight is the registered gating design for every
  arm; MACURA's own pipeline has no soft-weight term, so importing
  hard-only for this one arm would break same-machinery comparability. If
  soft weighting were suspected of absorbing the signal difference, the
  registered gate diagnostics (`imagine_stopped_frac`,
  `imagine_mean_weight`) and the stop-vs-weight endpoints separate the two.
- **Horizon cap is the registered shared imagination horizon, not MACURA's
  T_max = 10.** Identical for every arm; it only bounds error accumulation
  and cannot change which signal stops earlier.

## 2. Components reused (explicitly)

- Same frozen teacher ensemble, same consistency student, same SAC/UBE
  training loop as every other arm - `macura_gate` differs from `ordinary`
  in exactly the gate-signal knob: the rollout-gate score is `ukl` (Eq. 4
  member-KL plugin, `ukl_m_samples=8`) instead of the learned UBE `u`.
  Distillation is identical to `ordinary` (plain member matching).
- Runner: no new runner. `macura_gate` is registered in
  `udwm/scripts/run_delayed_bimodal_policy_ablation.py` VARIANTS (the shared
  arm table used by both the DelayedBimodal and DMC drivers), so it runs with
  the exact same configs/seeds/budget protocol as the registered arms.
- Adjudication: the existing `udwm/scripts/summarize_dmc_payoff.py` paired
  bootstrap machinery; macura contrasts print only when the run file carries
  `macura_gate` rows.

## 3. Endpoints and fair-comparison metric (fixed before running)

MACURA produces no w/g split, so the primary comparison metric is
**rollout-gating quality expressed in the endpoints the repo already
adjudicates**, per environment:

- `final_return` (paired vs `ordinary` and vs `identified_eq`) - the payoff
  question MACURA is a field baseline for.
- `u_rank_corr` vs `ordinary` is a DISTILLATION SANITY read only: `macura_gate`
  and `ordinary` share identical distillation, so u-rank parity is expected
  and a large gap would indicate an implementation bug, not a finding.
- Gate diagnostics (`imagine_stopped_frac`, `imagine_mean_weight`) from the
  same eval records, descriptive.

## 4. Pass bar

Registered as a comparison, not a win requirement (the discussion section is
strengthened by a clean honest comparison in either direction):

- Bar M1 (informative): the paired `macura_gate - ordinary` final_return
  bootstrap 95% CI excludes 0 on the DMC environment at the registered
  budget, OR the paired contrast vs `identified_eq` excludes 0 - i.e. the
  field baseline separates from at least one internal arm on the payoff axis.
- If both CIs include 0: MACURA is return-indistinguishable from the internal
  arms at this budget, and the discussion reports that with the u-rank
  fidelity context (ordinary distillation parity).
- Any u_rank_corr gap > 0.05 between `macura_gate` and `ordinary` on the same
  seeds is an implementation-consistency check failure, reported as such.

## 5. Protocol (identical to the registered arms)

Same staged budget discipline as the DMC study: gate check + budget sanity
probe first (seeds 0-1), then the registered adjudication budget. Seeds and
budget are inherited from the environment's registered protocol
(DelayedBimodal 1,800 steps; DMC 15k per Amendment 2), never re-derived
here. Env-specific budgets reuse `configs/delayed_bimodal_distill.yaml` and
`configs/dmc_hopper_probe.yaml`.

## 6. Caveats carried into the record

- The student ensemble's member disagreement is only as meaningful as the
  distillation preserved it; `ordinary` distillation does not optimize
  uncertainty preservation. If `macura_gate`'s u_KL signal is dominated by
  distillation artifacts, the comparison still stands as the honest field
  baseline for THIS stack (same limitation any external gating rule faces on
  a non-uncertainty-preserving student); a teacher-ensemble-u_KL variant can
  be registered separately if this one is unreadable.
- The kernel-plugin u_KL is a directional plugin estimate (each member's
  density evaluated on the pooled support incl. its own draws; bandwidth per
  state estimated from within-member spread). It is consistent for the Eq. (4)
  quantity in the m_samples -> inf limit, with variance controlled by
  `ukl_m_samples`; it is NOT the paper's closed-form u_GJS, which is
  undefined for implicit samplers. Scale differences vs the paper's u_GJS are
  expected and immaterial: the repo gate is percentile-thresholded, so the
  comparison is signal-vs-signal, not absolute-scale.
- MDE/power: the paired-delta SD reference is the DMC u-rank/return
  variance already measured (see the CRN-bias registration section 6 for the
  planning numbers); this baseline is a comparison study, and any null is
  reported with its MDE, not as a proof of equivalence.
- Multiple comparisons: per-endpoint pre-registration is the stated
  mitigation (same named methodological choice as every other study here).

## 7. Citations

- Frauenknecht, B., Eisele, A., Subhasish, D., Solowjow, F., Trimpe, S.,
  "Trust the Model Where It Trusts Itself - Model-Based Actor-Critic with
  Uncertainty-Aware Rollout Adaptation", ICML 2024, PMLR 235:13973-14005,
  arXiv:2405.19014. (Author list verified 2026-09-09; the repo bib files
  previously carried a wrong third author and were corrected.) Mechanism
  references: Eq. (4) (u_KL), Eqs. (15)-(20) (u_GJS and E), Eq. (21)
  (kappa self-tuning), Algorithm 2 (accept-iff-u<kappa rollout rule), and
  Section 6.1 text (defaults T_max = 10, zeta = 95%, xi = 1).