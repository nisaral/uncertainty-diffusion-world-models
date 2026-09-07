# Literature re-validation (2026-09-07) - collision check per claim

**Scope.** Scheduled re-check of the novelty landscape
(`NOVELTY_LANDSCAPE.md`, last pass 2026-08-19) against arXiv/venue output up
to 2026-09-07. Method: targeted search per claim below; each collision is
assessed against the *specific* claim, not the topic. This file is the
working record the paper's related-work section must be consistent with.

## Summary: the central claim still stands, with one positioning constraint

**No paper found that reports teacher-vs-student decision-relevant
uncertainty (UBE local `u = w - g` / ensemble disagreement after a value map)
for distilled one-step diffusion world models at matched NFE** - the refutation
condition named in `NOVELTY_LANDSCAPE.md` remains unmet. The closest value-map
result, Voelcker et al., is now **published (ICML 2025)** and must be cited as
such; its setting is a sampled stochastic model with one aleatoric variance,
not a distilled ensemble's epistemic/aleatoric split. One 2026 paper
(Valdi, arXiv:2607.00917, "Value Diffusion World Models") explicitly flags
distillation from multi-step teachers to single-step students as **future
work**, which supports rather than refutes the gap.

**Positioning constraint (payoff claims).** Uncertainty-gated imagination is a
crowded idea in 2026 - it is the mechanism in MACURA (2024), ELVIS
(RSS 2026; arXiv:2605.04709), GIRL (arXiv:2604.07426), and an AAAI 2026 paper
("Perceiving the Knowledge Boundary", Liu et al., AAAI 40(28):23990-23998).
The repo must never headline "we stop imagination when the model is
uncertain"; the headline stays the distillation-identifiability mechanism and
the payoff result is supporting evidence for why preserving the object
matters. The distinguishing features to keep in every framing: (a) the score
is a **Bellman-propagated, MC-estimated local UBE object** evaluated on a
**distilled 1-NFE student**; (b) the claim is about what the *loss* preserves,
with a proven identifiability failure and a corrected loss; (c) gating is
percentile-calibrated and auditable (Theorem 6 of
`research/proofs/identifiability-frontier.md`).

## Collision table (per claim)

| Repo claim | Closest prior work | Collision level | What remains distinct |
|---|---|---|---|
| C1: M=1 decision-aware distillation loss cannot identify the epistemic decision statistic (one equation, two unknowns); student can match the mixed statistic at zero loss | Voelcker et al., ICML 2025 (arXiv:2505.22772): sampled VAML losses prefer low-variance models; CVAML variance correction | None found on the ensemble-after-value-map/distillation object; their collapse is one model's aleatoric variance, ours is cross-member epistemic disagreement under a value map | The two-unknowns fibre + sign-unidentifiability theorem (T1), coupling erosion (T2), M>=2 sufficiency (T4) |
| C2: the correction (EMA reweighting) annihilates the aleatoric channel; equal weights recover it | DEGU (ICLR 2025): distill ensemble mean+variance into one net; evidential distillation lines (2507.18366, 2505.11731) | None: those distill ensemble variance into a *single* net on classifiers; the repo's student is a 5-member ensemble of member-distilled 1-NFE models and the object is a value-map statistic | The sticky-boundary mechanism (P3), the (w*/g*)^2 EMA inversion (T5), live-critic policy transfer |
| C3: gated imagination on the *preserved* object improves/measures policy behaviour | MACURA (2024); Liu et al., AAAI 2026 (dynamic uncertainty filter on imagined rollouts, ensemble of lightweight predictors); ELVIS (RSS 2026, program paper 182; arXiv:2605.04709 confirmed via ADS record 2026arXiv260504709D - uncertainty-aware lambda-return gating in visual MPC); GIRL (arXiv:2604.07426, hallucination control) | MEDIUM on the *usage*; LOW on the claim "a distilled student that preserved u gates like the teacher" | Their scores are one-step disagreement / drift proxies, not a distilled teacher-student UBE comparison; none measures the *loss-induced* preservation failure. Payoff prose must cite all four |
| C4: MC-UBE local estimator has O(1/M) bias, kurtosis-driven variance, Neyman allocation on A not w | Luis et al. UBE; QU-SAC (arXiv:2312.04386) | None: QU-SAC propagates UBE with analytic/Gaussian estimators; the MC-over-implicit-diffusion-ensemble estimator with finite-M bias correction is not in that line | Bias closed form; rare-mode kurtosis cost; the adaptive-allocation negative and the Neyman-on-A candidate (G1 below) |
| C5: selective/tail evaluation of uncertainty scores | Selective prediction literature (El-Yaniv-Wiener); "Epistemic Reject Option" (2511.04855); "Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning" (arXiv:2607.14817, 2026-07; optimal selector is a thresholded convex combination of ground-truth aleatoric and epistemic uncertainties) | LOW: they study classifiers/estimators, not distilled world-model students | The repo's u = w - g is exactly a *combination* object, which that 2026 result independently motivates; the repo's H4 negative (tail metrics understate the EMA collapse) is new |

## What changed since 2026-08-19 (action items for the write-up)

1. **Voelcker et al. citation upgrade.** Cite the ICML 2025 proceedings
   version (pp. of PMLR 267; arXiv:2505.22772 v2). Use their term
   "Corrected VAML (CVAML)" for the variance correction and state the
   analogue precisely: their correction targets aleatoric variance of a
   sampled model; the repo's M>=2 debias targets the cross-member split under
   a value map and is needed even at zero coupling.
2. **Gate-payoff related work grows to four.** MACURA, AAAI'26 knowledge
   boundary (Liu et al.), ELVIS, GIRL must be cited whenever the payoff /
   gating story is told; ELVIS additionally uses an ensemble-of-critics UCB
   that is conceptually adjacent to the repo's UBE net - say what differs
   (propagation object, distilled 1-NFE student, loss-preservation claim).
3. **Benchmark norm check.** The AAAI/ICML/ICLR sample in this area reports
   DMC (and sometimes Atari/CARLA); the repo's DelayedBimodal-only results
   are below venue norm until the DMC verdict lands. The registered DMC path
   (budget probe -> 30-seed adjudication) is therefore load-bearing for venue
   placement, not optional. DelayedBimodal stays as the controlled laboratory
   for the mechanism tables.
4. **Protocol alignment.** No literature conflict with the repo's eval
   conventions (paired teacher checksums, matched NFE, rank endpoints). The
   selective-prediction literature supports reporting risk-coverage curves
   for gate claims (the repo logs `selective_rank`/`selective_recall_bad`;
   upgrade to full risk-coverage curves for the payoff study - G6 below).

## Searches run (2026-09-07) and nearest misses

Queries: diffusion world model distillation one-step uncertainty; distillation
preserving ensemble disagreement/epistemic uncertainty; uncertainty-gated
imagination MBRL; value-aware/decision-aware model learning; UBE local
rewards; selective prediction for RL; consistency distillation in RL; MACURA
follow-ups. Nearest misses (checked, no collision): RACTD
(arXiv:2506.07822, reward-aware consistency distillation of diffusion
*planners*, not world models); diffusion weather-ensemble distillation
(arXiv:2608.27728, one-step marginal matching, no decision object); WIMLE
(ICLR 2026, IMLE one-step world models, no ensemble UBE object);
IPD (arXiv:2603.04289, offline imaginary planning distillation, uncertainty
not the object of study).

## Unverified leads (do not cite until a primary source is confirmed)

- **Normalized ensemble Q-learning (OpenAI).** A DMC result reporting that
  *normalizing* the ensemble-difference statistic before gating/exploration
  materially changes behaviour is highly relevant to the repo's open G4
  (normalized vs unnormalized UBE gating) but could not be verified this pass
  (search-layer results expired; no arXiv ID or venue confirmed). Do not cite
  it, and do not assume the direction of its result, until a primary source is
  located and read.
- Citation discipline for this file: arXiv IDs from the search layer are
  treated as suspect until confirmed against a second record (ADS, venue
  program, HF papers). ELVIS is the only ID confirmed twice this pass
  (arXiv:2605.04709 + RSS 2026 program entry); 2607.14817 and 2604.07426 are
  confirmed against venue/secondary mirrors only.

## Corrections + metadata re-verification (second pass, 2026-09-07)

Re-checked the four payoff citations and the sibling negative against
primary/secondary records before writing `papers/references.bib` entries:

1. **MACURA venue: ICML 2024, not RLDM.** icml.cc virtual poster 34216
   ("Trust the Model Where It Trusts Itself...", 2024-05-03, ICML 2024).
   `papers/references.bib` carries the ICML `@inproceedings` form;
   `references_2026_sweep.bib` notes the correction. Authors confirmed:
   Frauenknecht, Ehlgen, Trimpe.
2. **Biased Dreams exact title: "...in Latent Space Models".** The arXiv
   listing (2604.25416) says "Latent Space Models", not "Latent Dynamics
   Models" as the sweep bib had it. Title fixed in both bib files. Authors:
   Julia Berger, Bernd Frauenknecht, Sebastian Trimpe, Bastian Leibe.
3. **GIRL venue unconfirmed.** arXiv:2604.07426 (2026-04-08), single author
   (Prakul Sunil Hiremath per Semantic Scholar / arXiv mirrors). No venue
   confirmed as of 2026-09-07 - cite as arXiv preprint; the C3 row above
   does not claim a venue, and neither should the paper.
4. **AAAI-Liu metadata confirmed.** "Perceiving the Knowledge Boundary:
   Uncertainty-Guided Exploration and Imagination for World Models",
   Zhenxian Liu, Peixi Peng, Yangru Huang, Yonghong Tian, AAAI 40(28):
   23990-23998, DOI 10.1609/aaai.v40i28.39576 (AAAI OJS + dblp).
5. **ELVIS metadata confirmed.** "ELVIS: Ensemble-Calibrated Latent
   Imagination for Long-Horizon Visual MPC", Yurui Du, Pinhao Song, Yutong
   Hu, Renaud Detry, RSS 2026, arXiv:2605.04709 (ADS record
   2026arXiv260504709D + RSS program + dblp).

Nothing in the collision table changes: these are metadata corrections, not
claim reassessments. The positioning constraint stands - headline is the
distillation-identifiability mechanism; MACURA/AAAI-Liu/ELVIS/GIRL are cited
whenever the payoff story is told, and Biased Dreams is the sibling negative
that motivates why the mechanism-level claim is the durable one.

## Third pass (2026-09-07, while the DMC budget probe runs): no collision,
one new adjacent item, two lead updates

Searches rerun for anything new through early September 2026 on: diffusion
world-model distillation that preserves uncertainty (one-step students),
value-aware model learning, and uncertainty-gated imagination in MBRL.

1. **Refutation condition still unmet.** No paper reports teacher-vs-student
   decision-relevant uncertainty (UBE local u = w - g after a value map) for
   distilled one-step diffusion world models at matched NFE. Nearest misses
   this pass, none colliding: OPTD (arXiv:2608.02942, on-policy transition
   distillation for few-step diffusion *language* models - no RL
   uncertainty object); ForgeWM (arXiv:2608.14022, few-step video world
   models - no decision statistic); Teacher-Feature Drifting
   (arXiv:2605.07327v2, one-step distillation in feature space - no
   uncertainty claim); Diffusion Distillation for Efficient Weather Ensembles
   (arXiv:2608.27728, already logged: energy-distance one-step marginal
   matching, no decision object).
2. **New adjacent item (classifier side), cite only as "distillation of
   ensemble uncertainty is active outside RL".** Credal Ensemble Distillation
   (AAAI 2026, DOI 10.1609/aaai.v40i31.39837) distills an ensemble into a
   single credal-set classifier (interval probabilities). No value map, no
   w/g split, no RL - same family as the memo's GROUP 5 lines; supports the
   claim that the classifier-side distillation-of-uncertainty line is active
   while the post-value-map RL side is not.
3. **Valdi (arXiv:2607.00917) still v1 as of 2026-09-07** (paper.dou.ac
   shows 2607.00917v1, 2026-07-01; code released at
   github.com/Kit115/ValueDiffusionWorldModels). The memo's claim that the
   paper lists distillation from multi-step teachers to one-step students as
   future work stands; re-check before submission.
4. **RACTD (arXiv:2506.07822) updated to v2 (2025-12-25)**; review-note
   aggregations place it at ICLR 2026, which is NOT authoritative - cite as
   arXiv with the v2 date, and verify the venue before submission. Still a
   nearest-miss (reward-aware consistency distillation of diffusion
   *planners*, not a world model; no uncertainty object).
5. **Unverified lead unchanged after a second attempt.** "Normalized
   ensemble Q-learning (OpenAI)" still has no locatable primary source
   (arXiv/OpenAI pages/aggregators: zero hits on 2026-09-07). Stays in the
   do-not-cite section; G4's design must not assume its direction.
