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

*Pending -- run launched 2026-09-05.*
