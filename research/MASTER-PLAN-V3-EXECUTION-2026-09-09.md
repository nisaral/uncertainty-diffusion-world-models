# Master Plan v3 - execution record: visualization layer + verified grounding

Status: implemented 2026-09-09 (visuals + repo main page). Companion to
`master-plan-v3-paper-and-visualization.md` (the plan). This file records what
was built, which numbers went in, where they came from, and the verified
literature grounding for Hypothesis A (the paper's load-bearing framing).

---

## 1. Anchor verdict the visuals encode (post-S2)

One line, from the 2026-09-08 DMC 30-seed adjudication:

> The fix's *mechanism* transfers perfectly (EMA-collapse control 30/30,
> cross-environment). The fix's *practical ordering* does not (both eq
> position bars fail on DMC; `lagged_eq` reverses below `eq`, 7/30; plain
> self-gated `ordinary` tops the DMC table at 0.957, 30/30).

That is the paper's sharper question: **why does a mechanistically-general fix
stop paying off under a harder, drifting-critic regime, and is that failure
itself explicable?**

## 2. Deliverables (V1-V4 of the plan, all committed paths under `visuals/`)

| Item | File | Content |
|---|---|---|
| V1 | `visuals/svg/v1-identifiability-skip.svg` | Conceptual M=1 strip vs M>=2 ball geometry (Proposition 8), with the measured DelayedBimodal annotation (teacher g* ~ 73 vs w* ~ 0.007) |
| V2 | `visuals/data/curves.json` + curves panels inside the explorer | Real S1 (n=10, diagnostic) checkpoint trajectories at 3k/6k/9k/12k/15k: u_rank means and g ratio (student/teacher), eq vs EMA |
| V3 | `visuals/svg/v3-cross-env-ordering.svg` | DelayedBimodal (N=10) vs DMC (N=30) u_rank bars, same arms/colors, reversal on `lagged_eq` highlighted |
| V4 | `visuals/v4-explorer.html` | Self-contained interactive tour (hero verdict cards, 5-slide story, env toggle, ranking bars, metric table, glossary, caveats, provenance). Works offline and on GitHub Pages; fetches `visuals/data/*.json` when served, falls back to the embedded payload otherwise |
| Repo main page | `README.md` | "Visual tour (start here)" section: explorer link + inline V1/V3 figures |
| Index | `SUMMARY.md` | Visual entry-points pointer |
| Docs | `visuals/README.md` | Layout, provenance rules, rebuild instructions |

Numbers in the figures/page are recomputed from the raw run files by
`visuals/scripts/build_payload.js` (edit `visuals/templates/` + the script,
never the generated page) and verified against the adjudication docs; the
builder exits non-zero if means drift.

## 3. Data provenance and verification

Sources (all `exact_teacher_match: true`, max checksum gap 0.0):

- `runs/dmc_payoff_30seed_15k_gpu.json` - 180 rows, 30 seeds x 6 arms.
- `runs/dmc_payoff_30seed_15k_gpu_ctrl.json` - 30 rows, `ordinary_gate_off`.
- `runs/policy_combined_fix_10seed.json` - 70 rows (6 arms used).
- `runs/dmc_sanity_15k_gpu.json` - 60 rows, 10 seeds (curve extraction).

Extraction rules: final-eval endpoints use the row-level fields exactly as
`udwm/scripts/summarize_dmc_payoff.py` does. Curve checkpoints use the LAST
eval record at each 3k-multiple step (each milestone stores a forward-filled
copy of the previous eval as its first record; the digest's "6k duplicates 3k"
artifact is exactly that copy). Rebuilt means, verified in the builder:

- DMC u_rank means: ordinary 0.957, lagged_hybrid 0.837, hybrid 0.731,
  identified_hybrid (EMA) -0.076, identified_eq 0.711, lagged_eq 0.638;
  n>=0.70: 30/27/21/0/15/8. Matches the adjudication doc.
- DB combined-fix u_rank means: ordinary 0.934, hybrid 0.636, lagged_hybrid
  0.946, EMA 0.114, eq 0.877, lagged_eq 0.948; matches RESULTS-COMBINED-FIX.
- Gate-off control u_rank mean 0.975 (descriptive), return paired vs ordinary
  +0.025, CI includes 0 (reading rule ii: return-neutral at 15k).

Curve means (10 sanity seeds) reproduce the digest trajectories, e.g.
ordinary u_rank 0.55 -> 0.60 -> 0.85 -> 0.93 -> 0.97 at 3k/6k/9k/12k/15k;
EMA flat near zero; eq g ratio climbs 0.38 -> 0.70 while EMA g ratio sits at
~1e-5 (aleatoric starvation).

Committed payload snapshots (`visuals/data/*.json`) carry source file, row
count, and a sha256 prefix of the raw file so a payload can be re-checked
against its provenance. The 76 MB raw DMC file stays gitignored.

## 4. Verified literature grounding (Hypothesis A framing)

Verified against primary/secondary records on 2026-09-09 (arXiv pages, icml.cc
poster record, journal DOI pages):

1. **Osband et al. 2016** - Deep Exploration via Bootstrapped DQN
   (arXiv:1602.04621). Real paper wording, the load-bearing head-collapse
   quote for Hypothesis A:
   > "The shared network learns a joint feature representation across all the
   > data, which can provide significant computational advantages at the cost
   > of lower diversity between heads."
   The project's EMA-collapse finding is positioned as a new,
   mechanistically-detailed instance of this named failure mode.

2. **EEDQN (Ensemble Elastic DQN), arXiv:2506.05716** - year is 2025 (not
   2026). Verified abstract phrase:
   > "Ensemble-based methods and multi-step methods have each been used to
   > improve the stability and sample efficiency of value-based reinforcement
   > learning, but their interaction remains less well understood."
   Phrase the gap as ensemble x multi-step interaction, not ensemble-vs-
   recursive-value-learning broadly. Root `master-plan-v3-paper-and-
   visualization.md` section 1.2 was updated to the correct year + id.

3. **CRN / paired-sample variance reduction under nonstationarity** -
   Glasserman & Yao 1992, "Some Guidelines and Guarantees for Common Random
   Numbers", Management Science 38(6):884-908, DOI 10.1287/mnsc.38.6.884
   (exact pages verified). Wu, Zheng, Zhang, Zhang & Wang, "Nonstationary A/B
   Tests: Optimal Variance Reduction, Bias Correction, and Valid Inference",
   Management Science, DOI 10.1287/mnsc.2022.01205 (published online
   2024-09-18) formalizes the biased-paired-estimator case under drift. The
   M>=2 debiased (u, w-hat) estimator is a coupled/common-random-number
   comparison, so CRN's validity precondition (paired draws evaluated under
   similar conditions) is exactly what the drifting critic violates.
   *Flag:* "Bratley et al. 1986" as cited in the plan needs a bibliographic
   re-check before manuscript use (2nd edition of Bratley, Fox & Schrage, A
   Guide to Simulation is Springer 1987) - verify the edition/year you cite.

4. **MACURA - author list corrected.** Correct record: Frauenknecht, Bernd and
   Eisele, Artur and Subhasish, Devdutt and Solowjow, Friedrich and Trimpe,
   Sebastian; ICML 2024; PMLR 235:13973-14005; arXiv:2405.19014; icml.cc
   virtual poster 34216. The repo bib files previously listed a third author
   "Ehlgen, Tobias" (wrong); `papers/references.bib` and
   `papers/references_2026_sweep.bib` were corrected on 2026-09-09, with PMLR
   pages added. Any prose built on the "Ehlgen"-authored MACURA (or on a
   "2026" EEDQN) must be re-checked before it reaches the manuscript.

## 5. Open items (from the plan's Part 4, still standing)

1. Hypothesis A probe (value-reference consistency under paired M>=2 sampling
   + lag), preregistered as the CRN-bias diagnostic.
2. MACURA head-to-head field baseline on DMC (discussion-section gap).
3. Compute-normalized re-analysis; effect sizes (paired Cohen's d equivalent);
   power/MDE justification for N=30; named multiple-comparisons statement.
4. Conditional 30k return extension (DMC), with `lagged_eq - ordinary` return
   +0.101 [+0.016, +0.199] as the flagged descriptive-only hypothesis.
5. GitHub Pages enable (branch `main`, root) before the next push, so
   `visuals/v4-explorer.html` is live at
   https://nisaral.github.io/uncertainty-diffusion-world-models/visuals/v4-explorer.html
