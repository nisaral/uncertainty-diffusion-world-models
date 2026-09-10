---
title: UDWM Distill Lab
emoji: 🧪
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: A laboratory for a distillation loss, not a SOTA agent.
---

# UDWM distill lab

**A laboratory for a loss, not a SOTA agent.** It does not win control.

Press **Train all four**. Four students share one 2-D `(w, g)` field
(`g* ≫ w*`, the measured DelayedBimodal / hopper regime):

- **ordinary** — member-mean matching; ranking usually holds
- **hybrid** — one mixed scalar (M=1 fibre); `w` can collapse
- **identified_eq** — equal-weight `(w, g)`; ranking recovers
- **identified_ema** — inverse-variance EMA; aleatoric channel dies; u-rank → noise

The gate panel is the Infoprop / MACURA question in miniature: would you trust
this student to stop imagined rollouts?

Paper (not this toy): [github.com/nisaral/uncertainty-diffusion-world-models](https://github.com/nisaral/uncertainty-diffusion-world-models)
