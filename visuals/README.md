# Visuals (generated)

The repo main page (README) shows two static figures and links to one
self-contained interactive page. Everything under this directory is
**generated from the local run files** by

```
node visuals/scripts/build_payload.js
```

so the numbers in the figures, the page, and the payloads can never drift from
each other. The builder recomputes every endpoint from the raw rows and FAILS
if the means drift from the adjudication docs
(`research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md`,
`research/RESULTS-COMBINED-FIX-POLICY-2026-09-07.md`).

## Layout

| Path | What it is |
|---|---|
| `data/dmc_30seed_final.json` | DMC hopper-hop 30-seed x 6-arm endpoints (u_rank mean/median, n>=0.70, w_rmse median, ns_mse, return, g ratio, per-seed vectors) + registered-verdict block |
| `data/dmc_ctrl_30seed_final.json` | DMC `ordinary_gate_off` control (30 seeds) + Addendum-4 reading rule |
| `data/db_10seed_final.json` | DelayedBimodal combined-fix 10-seed endpoints |
| `data/curves.json` | DMC sanity (S1, n=10) checkpoint trajectories at 3k/6k/9k/12k/15k |
| `svg/v1-identifiability-skip.svg` | Conceptual M=1 strip vs M>=2 ball geometry (V1) |
| `svg/v3-cross-env-ordering.svg` | DelayedBimodal vs DMC ordering flip (V3) |
| `v4-explorer.html` | Self-contained interactive tour (V4): embedded payload + embedded figures, works offline or on GitHub Pages with no server |
| `scripts/build_payload.js` | The generator + verifier |
| `templates/v4-explorer.template.html` | Hand-authored page shell; markers `__V1_SVG__`, `__V3_SVG__`, `/*__PAYLOAD_JSON__*/` are injected at build time |

## Provenance rules

- Sources are the gitignored files under `runs/` (76 MB DMC 30-seed file stays
  out of git); each payload records `source_file`, row count, and a sha256
  prefix so a committed payload can be checked against the raw file it came
  from.
- Final-eval numbers use the row-level fields, exactly like
  `udwm/scripts/summarize_dmc_payoff.py` (the adjudicator).
- Curve extraction rule: eval-side fields are logged at 3k-multiples; each
  milestone also stores a forward-filled copy of the previous eval as its
  first record. The builder plots the LAST record at each milestone (the fresh
  eval). See `data/curves.json` meta.
- Edit the template, never the generated page: rebuild with
  `node visuals/scripts/build_payload.js` from the repo root.
