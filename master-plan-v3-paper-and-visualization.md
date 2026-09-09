# Master Plan v3 — Post-S2 Reframing, Paper Structure, and Visualization Plan
(2026-09-08, supersedes `revised-master-plan-post-s2.md` for framing; that file's experiment table still stands)

---

## PART 1 — Why this reframing is stronger than what came before

### 1.1 The core POV shift, argued in full

**Old story (pre-S2):** naive distillation is non-identifiable → EMA reweighting is a broken fix → equal-weight correction fixes it → does the fix transfer to policy? DelayedBimodal said "partially yes." DMC was the pending test.

**New story (post-S2):** the fix's *mechanism* transfers perfectly (EMA-collapse control: 30/30 cross-environment). The fix's *practical benefit* does not (both position bars fail; `ordinary` tops the DMC table). This is not a weaker paper — it's a paper with a second, sharper question: **why does a mechanistically-general fix stop paying off under a harder, drifting-critic regime, and is that failure itself explicable?**

This matters because "here's a fix, it partially works" is a much weaker paper than "here's a fix, we proved exactly when and why it stops working, and that failure is itself a specific, citable instance of a known-but-previously-unexplained phenomenon in the field." The second is what Hypothesis A gives you, if it holds.

### 1.2 Why Hypothesis A (Osband/head-collapse framing) is the load-bearing idea now

Three reasons this is better than every prior framing attempt (EMA-as-bug, nonstationarity-as-cause, decoupled-predictor):

1. **It explains the one result nothing else in the project explains:** the lag-family asymmetry (`lagged_hybrid` improves, `lagged_eq` degrades). Every prior explanation in this project (identifiability, EMA-starvation, nonstationarity-alone) predicts these two arms should respond *the same way* to lagging, since lagging is a property of the critic, not the loss family. They don't respond the same way. That's the anomaly Hypothesis A directly targets.

2. **It's grounded in the foundational paper for exactly this technique, not an adjacent one.** Osband et al. (2016), the paper that established ensemble-disagreement as an epistemic-uncertainty proxy under a live bootstrapped target — structurally the closest ancestor to what your `w` estimator does — explicitly names **head-diversity collapse** as a known failure mode under excessive representation-sharing or insufficient stochasticity. Your EMA-collapse finding is a new, mechanistically-detailed instance of exactly this named-but-underexplained phenomenon. Citing Osband directly, rather than the more tangential Double-Q/TD3 lineage, puts your contribution in direct dialogue with the paper whose gap you're filling — this is a much stronger "related work" positioning than a general analogy.

3. **A current paper (EEDQN, Ensemble Elastic DQN; arXiv:2506.05716, 2025) explicitly states that how ensemble-based uncertainty interacts with recursive value-learning dynamics "remains less well understood."** That's a live, citable admission from the field that this exact angle is open — you're not inferring a gap, you're pointing at a stated one.

### 1.2.5 The formal mathematical grounding for Hypothesis A — this is the real strengthening

The M≥2 identified loss draws paired latent samples and computes a debiased statistic from them — structurally, this is a **coupled/common-random-number (CRN) comparison**, a well-established variance-reduction technique in simulation methodology (Bratley et al. 1986; Glasserman & Yao 1992). CRN's entire validity rests on one explicit precondition, stated plainly in the methodology literature: *the paired draws must be evaluated under similar/identical conditions* — the technique's purpose is to attribute observed differences to the quantity being compared, not to fluctuation in the underlying experimental conditions between the paired draws. When that precondition holds, CRN provably reduces variance (and under specific monotonicity/continuity conditions, is provably optimal, per Glasserman & Yao). When it doesn't hold — when the "conditions" (here, the value target) drift between the two paired evaluations — the coupling that gives CRN its guarantees breaks down.

**This is not just an analogy — there is a direct, formal treatment of exactly this breakdown.** Wu, Zheng, Zhang, Zhang & Wang, "Nonstationary A/B Tests: Optimal Variance Reduction, Bias Correction, and Valid Inference" (Management Science, 2024) formalize precisely the case where paired/CRN-style comparisons are run against a system whose underlying state is nonstationary between the paired observations — showing that naive paired estimators become biased under nonstationarity, deriving the bias term explicitly, and proposing a correction.

**Reframe Hypothesis A around this, precisely:** your M≥2 debiased `(ŵ, ĝ)` estimator is a coupled-sample estimator whose paired latent draws are, in the live-critic setting, implicitly evaluated against a value target that has moved between them (the critic is being updated in the same training loop). Per CRN theory, this is exactly the condition under which the coupling's variance-reduction guarantee is void and a specific, characterizable bias is introduced instead. The conflated single-draw statistic (`hybrid`) never needs a *paired* comparison — it only evaluates once — so it structurally cannot suffer this particular failure mode, which is precisely why it doesn't respond to lagging the same way `eq` does. **This upgrades Hypothesis A from "a plausible empirical pattern, evidenced by an analogy to Double-Q/Osband" to "a predicted consequence of a formally understood statistical phenomenon (CRN bias under nonstationarity), for which a correction method already exists in the literature to adapt."**

**Concrete implication for the fix:** rather than inventing a new correction from scratch, adapt the bias-correction approach from Wu et al.'s nonstationary-A/B-test framework to your paired-latent debiasing step — this is a much stronger "solution" section for a paper than an ad hoc dedicated-value-reference fix, because it inherits a proof from an established, rigorous source rather than being a novel unproven patch.

### 1.3 Why Hypothesis B (Hopper reward-sparsity) matters even though it's the smaller finding

It's not meant to explain the u-rank results — it's meant to correctly scope the return-payoff null. Without it, a reviewer could read "return stayed floor-bound at every scale you tried" as evidence the whole line of work has no practical consequence. With it, you can correctly say: *this specific task variant has a documented structural property (zero-reward-on-fall without episode reset) that makes any payoff signal hard to detect regardless of method* — which is both honest and protects the u-rank findings from being unfairly discounted by an unrelated benchmark quirk.

---

## PART 2 — Full paper structure

### Title (working)
*"When Does Fixing an Identifiability Failure Matter? Uncertainty-Preserving Distillation for Diffusion World Models Under Drifting Value Targets"*

### Structure

1. **Introduction.** Frame via the confounding/identifiability literature broadly (nuclear IUQ, epidemiology — established cross-disciplinary pattern), narrowing to: decision-aware distillation for diffusion world models has never had this pattern rigorously characterized. State the two-part contribution up front: (a) a proof + corrected loss, (b) a characterization of exactly when the correction does and doesn't pay off, with a mechanistic explanation for the boundary.

2. **Background.** MBPO/SAC backbone, UBE (note: theorem is distribution-agnostic, Gaussian assumption is only in Luis et al.'s practical instantiation — this is what licenses your diffusion extension), diffusion/consistency distillation, DIAMOND/WIMLE/MACURA positioning, Osband et al. and the ensemble-disagreement-uncertainty lineage (new, elevated section given Part 1.2).

3. **Identifiability failure.** The `w* + (g* − Σ̄)` non-identifiability proof, the constructive counterexample, Proposition 8's level-set/S-strip tightening. This section is unchanged and is your theoretical anchor.

4. **The EMA-collapse mechanism.** Full elimination chain (ruled out LR, clip, M, corruption distribution), the measured leverage rate (~1e-6/update), framed explicitly as a new, mechanistically-detailed instance of Osband's head-collapse phenomenon.

5. **DelayedBimodal: corrected-weight transfer.** The N=10 policy result, partial transfer, combined-fix near-parity result. State plainly this is the "easy regime" result.

6. **DMC: cross-environment adjudication.** The N=30 verdict in full. Mechanism generalizes (30/30 EMA control), practical benefit does not (both position bars fail cleanly). This is the paper's turning point, not a disappointment — write it that way.

7. **Why the benefit doesn't transfer: target-staleness asymmetry.** Hypothesis A's probe and result (pending). If confirmed: the paired-latent value-reference-consistency mechanism, positioned against Osband/ensemble-uncertainty-under-bootstrapped-targets literature, plus whatever fix follows from the diagnosis.

8. **The return-payoff thread.** Honest treatment of the null across every scale tried, the Hopper-specific reward-sparsity caveat, and the Walker2d result if it's been attempted by submission time.

9. **Limitations.** Toy + two-benchmark scope, Luis UBE Assumptions 1-2 caveat (already stated throughout the project), `√U` is a score not a conformal interval, no full parity even in the best regime.

10. **Discussion.** The meta-point: calibration-correctness and downstream RL performance can come apart, and this project is a worked, mechanistically-explained example of exactly that — a caution for the field, not just a result about one loss function.

---

## PART 3 — GitHub visualization plan

Four visualizations, ranked by how much they'd help a reader (or a prof skimming the repo) understand the project in under a minute, before reading any tables.

### V1 — The identifiability geometry (conceptual, not data-dependent — can build now, doesn't need final numbers)
A diagram showing the **S-strip vs. point** distinction from Proposition 8: at M=1, the sublevel set of the matched statistic is a strip (a whole line of (w,g) pairs all giving the same loss) — visually show a 2D (w,g) plane, the true (w*,g*) point, and the degenerate strip of students that all achieve near-zero loss along it. At M≥2, show the sublevel set collapsing to a small ball around the true point. This is the single most explanatory image for the entire project's theoretical contribution — it makes the identifiability failure visually obvious in a way the algebra doesn't. Static SVG, put at the top of the README.

### V2 — EMA-collapse trajectory (data-dependent, use your actual logged w/g curves)
A training-curve plot: `ŵ`, `ĝ` over training steps, for `identified_eq` vs `identified_hybrid` (EMA), showing eq's `g` climbing toward the teacher's true value while EMA's `g` flatlines near zero almost immediately. This is your most visually dramatic real result — the numbers (student `g` ≈ 0.01 vs teacher ~80) are striking even before any statistics are applied.

### V3 — Cross-environment arm-ordering comparison (data-dependent)
Side-by-side bar charts: DelayedBimodal arm ordering vs. DMC arm ordering, same arms, same color coding, explicitly highlighting where the ordering reverses (`lagged_identified_eq`'s position). This is the clearest way to show the paper's central empirical turn (mechanism transfers, practical ordering doesn't) at a glance.

### V4 — Interactive HF Space (bigger lift, do last, highest payoff for outreach specifically)
A small Gradio app: pick an environment (DelayedBimodal / Hopper) and an arm, see the live `u`/`w`/`g` decomposition plotted against the teacher's ground truth, plus the checksum/exact-pairing metadata surfaced for credibility. This is the thing you'd actually open on a call with a prof or in an interview — much more effective than asking someone to read a markdown table cold.

**Sequencing:** build V1 now (no dependency on pending experiments), V2/V3 once the Hypothesis-A probe data and final write-up numbers are locked, V4 last as the outreach-facing capstone.

---

## PART 3.5 — Strengthening the experimental design itself

### Formalize the CRN-bias probe (replaces the vaguer "value-reference consistency" description)
Register a precise diagnostic: measure the wall-clock/step gap between the two paired latent draws' value evaluations under `eq` vs `lagged_eq`, and — following Wu et al.'s framework — estimate the induced bias term as a function of that gap and the critic's local rate of change (e.g., `|Q_t − Q_{t-k}|` over the relevant window). **Pre-registered prediction:** the estimated CRN-bias term should scale with the critic's drift rate, and should be small/negligible for `hybrid` (no pairing) but non-negligible for `eq`/`lagged_eq` — directly explaining the asymmetric lag response. This is a sharper, more falsifiable registration than "check if something about pairing matters."

### Add the missing direct field-baseline: MACURA head-to-head
This was flagged early in the project and never executed — worth doing now specifically because the paper's discussion section otherwise only compares your method against your own internal ablations (ordinary/hybrid/EMA variants), which a reviewer will flag as insufficient field-baseline comparison. Run MACURA's ensemble-JS-divergence rollout gating on the same DMC task, same budget, same seed protocol, as an additional arm. This doesn't need to "win" — even a clean, honest "our u-rank fidelity is higher/lower/comparable to MACURA's gating signal, here's why" materially strengthens the related-work section from citation to actual comparison.

### Statistical rigor additions
- **Report a minimum-detectable-effect (power) justification for N=30**, not just the bootstrap CIs after the fact — state, before running the CRN-bias probe and MACURA comparison, what effect size you could reliably detect at the planned seed count, so the eventual null/positive verdict is defensible against a "underpowered" critique.
- **Formalize the multiple-comparisons statement** you already drafted in an earlier thread (Thread #2) into an explicit paragraph in the manuscript's methods section: per-endpoint pre-registration as the stated mitigation for not jointly correcting across the many endpoints reported per study — make this a named, citable methodological choice, not an implicit practice.
- **Report effect sizes (standardized, e.g. Cohen's d equivalent for paired designs), not just raw deltas and CIs**, for every headline contrast in the final manuscript — raw deltas on differently-scaled endpoints (u-rank vs w-rmse vs return) aren't directly comparable in magnitude, and a reviewer will want a normalized sense of how large each effect actually is.

### Ablation to add: does the CRN-bias fix work when applied to `hybrid` too (as a placebo check)?
If the fix is really about pairing-under-drift specifically, applying the same bias-correction machinery to `hybrid` (which doesn't pair draws) should do approximately nothing — a useful, cheap negative control that strengthens the causal story if it comes back null as predicted.

## PART 4 — Updated priority list (supersedes the table in the prior revision doc only in ordering, not content)

1. Hypothesis A probe (value-reference consistency under paired M≥2 sampling + lag) — reframed around Osband, as above
2. Compute-normalized re-analysis (free, still outstanding)
3. V1 diagram (free, can build in parallel with #1)
4. Fix implementation, if #1 confirms the mechanism
5. Walker2d payoff attempt (Hypothesis B)
6. V2/V3 diagrams once final numbers are in
7. Manuscript draft, following Part 2's structure
8. V4 HF Space, GitHub release tagging, HF checkpoint upload — final capstone step, all together per the earlier sequencing note (don't stagger partial releases before the DMC/Hypothesis-A story is fully settled)
