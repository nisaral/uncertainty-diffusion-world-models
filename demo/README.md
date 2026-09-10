# Distill lab — a laboratory for a loss, not a SOTA agent

Interactive toy of the four distillation losses. It does **not** win control,
is **not** a robot or chatbot, and is **not** evidence. The adjudicated tables
live under `research/`. This page is for intuition: teacher vs student `u = w − g`,
rank correlation, and a percentile gate (the Infoprop / MACURA composition in miniature).

## Run locally

```bash
# static playground (open in a browser)
# demo/index.html

# numpy backend
python -m demo.lab

# Gradio (optional)
pip install -r demo/requirements.txt
python -m demo.app
```

From repo root, `python -m demo.app` needs the package on `PYTHONPATH` (repo root).

## Hugging Face Space

https://huggingface.co/spaces/nisaralll/udwm-distill-lab

Static Space serving `index.html`. No GPU. Press **Train all four**.

## What the four arms are

| arm | loss | what you should see |
|---|---|---|
| ordinary | member-mean matching | ranking holds |
| hybrid (M=1) | mixed scalar \(S = w+(1-\rho)g\) | fibre walk; `w` can collapse |
| identified_eq | equal-weight \((\hat w,\hat g)\) | ranking recovers |
| identified_ema | inverse-variance EMA weights | `g` starves; u-rank → noise |

The field is aleatoric-dominated (`g* ≫ w*`), the regime measured on DelayedBimodal and hopper-hop. **Rank-only** — do not read heatmap color as calibrated `w` magnitude (G7/G9).
