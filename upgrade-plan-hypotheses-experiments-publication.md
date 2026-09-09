# Upgrade Plan: New Hypotheses, Experiments, and Publication Strategy

## Part 1 — New hypotheses your existing data is already hinting at, not yet formally tested

### H1 (highest priority, cheapest, most obviously missing): the combined fix has never been run
You have two independently-confirmed fixes: (a) lagging the critic (fixes nonstationarity), (b) equal-weight identified loss (fixes identifiability). Every 2x2 so far tested them **separately**, or combined lag with the *broken* EMA-weighted version (`lagged_identified`, run before you knew EMA was the pathology). **Nobody has run "lagged critic + equal-weight identified loss" together.**

**Why this matters:** if the two mechanisms are independent (nonstationarity and identifiability are genuinely separate failure modes, which your mechanism-tracing work suggests), combining both fixes should close the remaining gap between `identified_eq`'s partial transfer and full parity with `ordinary`/`lagged_hybrid` — possibly exceeding both, since you'd be fixing two real problems at once instead of one.

**Experiment:** add `lagged_identified_eq` as a new arm to whatever adjudication run comes next (DelayedBimodal re-run is cheap on CPU; include it in the DMC main run too once budget is settled). Pre-register the prediction now: *lagged_identified_eq u-rank ≥ lagged_hybrid AND ≥ identified_eq, ideally approaching or exceeding `ordinary`.*

**This is your single best "prove it more" move.** It's cheap, it's an obvious gap, and if it lands as predicted it turns "partial transfer" into "full transfer once both known failure modes are fixed" — a much stronger headline result.

### H2: the eq-vs-hybrid win is conditional on naive-loss instability, not eq's own quality
Your own re-diagnosis of the DMC sanity run found eq's win over hybrid on DelayedBimodal was conditional — hybrid happened to be unusually bad there. This suggests a testable, general hypothesis: **the identified loss's measured benefit over the naive loss scales with how unstable/variable the naive loss is across seeds on a given environment**, rather than being a fixed advantage.

**Experiment:** across your existing runs (DB fixed-map, DB policy, DMC sanity), compute the naive `hybrid` loss's **cross-seed variance** in u-rank, and regress that against the `eq − hybrid` gap. If this correlation holds even loosely across your existing data (no new compute needed — this is a re-analysis), you have a predictive relationship: *identifiability-correction helps most exactly where naive matching is most unstable.* This is a genuinely elegant, checkable claim and costs zero GPU time.

### H3: is EMA-style starvation a general pathology, not a bug specific to your EMA scheme?
Your mechanism work diagnosed *this specific* EMA reweighting as pathological. You haven't tested whether **any** slow-adapting per-term normalization scheme under a moving target would fail the same way (this would be a much bigger claim — a general principle about adaptive loss-balancing under nonstationarity, not a one-off implementation bug).

**Experiment:** swap in one alternative adaptive scheme with different adaptation dynamics (e.g., a shorter EMA half-life, or a GradNorm-style per-step reweighting instead of a slow running average) and check whether it also starves `g`. If multiple different "adaptive" schemes all fail while equal-weight doesn't, you can claim something more general: *adaptive per-term reweighting is fundamentally risky under a nonstationary value target, independent of the specific scheme.* If only your specific EMA fails, the claim stays narrower but still valid. Either outcome is informative and cheap to obtain (reuses your probe infra directly).

### H4: mine the `selective_rank_corr` / `selective_recall_bad` columns you're already logging but haven't discussed
Your DMC output includes selective metrics that haven't factored into any verdict yet. Hypothesis: **even where overall u-rank fails, the ranking might still be preserved specifically among high-uncertainty states** — which is arguably the only regime that matters for a real gating decision (you don't care about ranking quality among low-uncertainty states, you care about correctly flagging the risky ones).

**Experiment:** re-read your existing DMC sanity data through this lens (no new runs needed) — does `identified_eq`'s `selective_recall_bad` tell a different, more favorable story than the aggregate `u_rank_corr`? If so, this reframes the "partial transfer, not parity" finding into something sharper: *aggregate ranking degrades, but the decision-relevant tail behavior is better preserved than the headline number suggests* — genuinely useful for the "so what" section of a paper, since gating decisions care about the tail, not the average.

### H5: the analytic budget-prediction bound (from the master plan, elevated here)
Combine the measured leverage rate (~1e-6/update) and your balance-window formula (`std(ĝ)/g* ≈ sqrt(2/(N(M−1)))`) into an explicit predicted minimum training budget for `g` to recover, as a function of `g*/w*`. Check this prediction against whatever the DMC budget probe shows. If it holds even approximately, this is your strongest theoretical upgrade — a predictive formula, not just a post-hoc description.

---

## Part 2 — Experiment priority list (ranked, with compute cost)

| # | Experiment | Confirms/refutes | Compute | Priority |
|---|---|---|---|---|
| 1 | `lagged_identified_eq` new arm (H1) | Do the two fixes compose? | Low (CPU for DB, folds into planned DMC run) | **Do first** |
| 2 | Cross-seed variance regression (H2) | Is eq's benefit conditional on naive-loss instability? | Zero (re-analysis of existing data) | **Do immediately, free** |
| 3 | Selective-metric re-read (H4) | Does tail-ranking survive even when aggregate doesn't? | Zero (re-analysis) | **Do immediately, free** |
| 4 | Budget probe + curve-shape read (already planned) | Is DMC's low u-rank a budget artifact? | Medium (Kaggle GPU, already scheduled) | In progress |
| 5 | Analytic budget-prediction bound (H5) | Turns mechanism into a predictive theory | Zero new compute, needs #4's data | After #4 |
| 6 | Alternative adaptive-scheme test (H3) | Is EMA-starvation general or scheme-specific? | Low-medium | After core DMC verdict lands |
| 7 | Second DMC task (Walker2d/Cheetah), from earlier plan | Environment-generality | Medium-high | After Hopper fully resolved |
| 8 | Conformal calibration wrapper | Turns √U into a real coverage guarantee | Zero GPU | Whenever there's a lull |

**Immediate next session, concretely:** run #2 and #3 today — they need no new compute, just re-reading data you already have, and either could materially change how you frame the DMC sanity result before you've spent more Kaggle hours. Register #1 alongside whatever the budget probe resolves to.

---

## Part 3 — Publication strategy across three surfaces

### Paper
Given everything landed so far, this is a legitimate **workshop paper at minimum** (ICLR/NeurIPS workshops on RL theory, model-based RL, or uncertainty), with a realistic shot at main-track/strong-venue consideration **if** H1 (combined fix reaching parity) and a resolved, honest DMC budget story both land — main-track reviewers respond well to exactly this shape: a clean identifiability theorem, a self-corrected empirical mistake reported honestly, and a generalization check across two environments with a resolved confound. Structure: identifiability proof → fixed-map study → naive-loss policy falsification → mechanism diagnosis (EMA starvation) → corrected-weight confirmation (partial transfer) → DMC cross-environment check with the budget-confound story told straight → (if H1 lands) combined-fix near-parity result as the closer. Don't skip narrating the confound-catching — it's a feature, reviewers trust papers that show their own mistakes more than ones that don't.

### GitHub
Already in excellent shape structurally (pre-registration discipline, dated addenda, clean test suite). Before wider release:
- Add a single top-level `SUMMARY.md` or expand the README's opening to walk a first-time reader through the *narrative* (identifiability → mechanism → correction → cross-environment check) in plain language before the tables — right now the README is precise but assumes the reader already knows the story.
- Tag a release (`v1.0-identifiability` or similar) once the DMC verdict lands, so the repo has a citable, stable snapshot rather than only a moving `main`.
- Add a `CITATION.cff` file so it's trivially citable once/if it accompanies a paper or preprint.

### HuggingFace
This is a genuinely good, slightly unusual fit — not a generative content model, but a **diagnostic reference artifact**. Concretely, release:
- The trained teacher ensemble + both distilled students (naive `hybrid`, broken `identified_hybrid` (EMA), corrected `identified_eq`) as checkpoints, for DelayedBimodal and (once resolved) Hopper.
- A model card that frames it explicitly as: *"a minimal, reproducible example of an identifiability failure in uncertainty-preserving distillation, plus a working fix — use these checkpoints to reproduce the epistemic/aleatoric collapse pathology without re-running training."* This is a legitimately useful thing for other researchers working on calibrated model-based RL to have on hand, and it's a distinctive, non-generic use of a HF model card (most RL checkpoints on HF are just "here's a trained agent," not "here's a documented failure mode you can inspect").
- Optionally, a small HF Space (Gradio) letting someone pick an arm and an environment and see the `u`/`w`/`g` decomposition plotted live — turns the driest part of the project (calibration tables) into something an interviewer or prof can interact with in 30 seconds, which is disproportionately valuable for outreach compared to the effort of building it (a simple Gradio wrapper around your existing eval/metrics code).

**Sequencing across all three:** finish H1 and the DMC budget resolution first — release everything (paper draft, tagged GitHub release, HF checkpoints) together once the story is complete, rather than staggering partial releases that would need retroactive correction if the DMC verdict shifts the narrative again.
