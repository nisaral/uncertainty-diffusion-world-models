# Does distillation preserve decision-relevant uncertainty in diffusion world models?

**Working draft — not submission-ready.**  
**Code:** https://github.com/nisaral/uncertainty-diffusion-world-models

## Claim

A single-latent decision-distillation loss matches
\(w^\star+(g^\star-\bar\Sigma)\) and is **unidentified** for epistemic
uncertainty. The student can reallocate variance between ensemble disagreement
and aleatoric noise. Under a fixed value map that reallocation can *look like*
preservation (`w_deb` improves). Under an online critic it can destroy
teacher–student rankings while improving next-state MSE. Identifying the
target requires \(M\ge2\) latents and separate \((\hat w,\hat g)\) terms.
Lagging the critic is a different intervention (nonstationarity).

Cite Voelcker et al., arXiv:2505.22772, as the aleatoric analogue.

## Evidence

- Identifiability construction and 20-seed identified-vs-hybrid table:
  `research/RESULTS-IDENTIFIABILITY-2026-08-29.md`
- Fixed-map hybrid vs ordinary: `research/RESULTS-STRESS-LARGE-2026-08-21.md`
- Live-critic hybrid vs ordinary: `research/RESULTS-POLICY-SCALE-2026-08-22.md`

Teacher checksum gaps were zero. Policy 2×2 for identified loss is registered,
not reported.

## What is not claimed

A new UBE. Conformal coverage. First gated imagination. Policy SOTA.
Identified distillation as a confirmed policy method.
