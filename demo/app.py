"""Gradio wrapper around the distill lab.

The interactive playground is demo/index.html (also the Hugging Face Space).
This file adds a numpy backend so `python demo/app.py` works offline.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

try:
    from demo.lab import train
except ImportError:
    from lab import train  # python demo/app.py from this folder

ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def run_backend() -> str:
    out = train()
    lines = [
        "Toy (w, g) field — not DelayedBimodal, not a control score.",
        "Rank-only. A laboratory for a loss, not a SOTA agent.",
        "",
    ]
    for key, arm in out["arms"].items():
        g = arm["gate"]
        lines.append(
            f"{key:16s}  u-rank={arm['u_rank_final']:+.3f}  "
            f"g_ratio={arm['g_ratio_final']:.3f}  "
            f"over-keep={g['over_keep_frac']:.2f}  under-keep={g['under_keep_frac']:.2f}"
        )
    return "\n".join(lines)


def build() -> gr.Blocks:
    with gr.Blocks(title="UDWM distill lab") as demo:
        gr.Markdown(
            "# UDWM distill lab\n"
            "**A laboratory for a loss, not a SOTA agent.** "
            "It does not win control. Press **Train all four** in the playground, "
            "or run the numpy backend below."
        )
        gr.HTML(HTML)
        with gr.Accordion("Numpy backend (same four losses)", open=False):
            btn = gr.Button("Train via numpy")
            box = gr.Textbox(label="u-rank / g-ratio / gate disagreement", lines=8)
            btn.click(run_backend, outputs=box)
        gr.Markdown(
            "Paper tables: [DMC 30-seed](https://github.com/nisaral/uncertainty-diffusion-world-models/blob/main/research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md) · "
            "[MACURA 30-seed](https://github.com/nisaral/uncertainty-diffusion-world-models/blob/main/research/RESULTS-MACURA-DMC-30SEED-2026-09-10.md) · "
            "[repo](https://github.com/nisaral/uncertainty-diffusion-world-models)"
        )
    return demo


if __name__ == "__main__":
    build().launch()
