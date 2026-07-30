"""Plot ablation learning curves and throughput bars from JSON outputs.

Usage::

    python -m udwm.scripts.plot_results --ablation runs/ablations/ablation_results.json
    python -m udwm.scripts.plot_results --throughput runs/throughput_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_ablation(path: Path, out: Path) -> None:
    with path.open(encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    plt = _try_matplotlib()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Always write a markdown summary table
    md = out.with_suffix(".md")
    lines = [
        "# Ablation summary",
        "",
        "| variant | seed | return | s_mse | r_mse | u_corr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('name')} | {r.get('seed')} | {r.get('final_return')} | "
            f"{r.get('next_state_mse')} | {r.get('reward_mse')} | {r.get('u_corr')} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", md)

    if plt is None:
        print("matplotlib not installed — skipped PNG curves (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        curve = r.get("curve") or []
        if not curve:
            continue
        xs = [c["step"] for c in curve if c.get("step") is not None]
        ys = [c["return_mean"] for c in curve if c.get("return_mean") is not None]
        if xs and ys and len(xs) == len(ys):
            ax.plot(xs, ys, marker="o", label=f"{r['name']} (s{r['seed']})")
    ax.set_xlabel("env steps")
    ax.set_ylabel("eval return")
    ax.set_title("Ablation learning curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("Wrote", png)

    # bar of final returns
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"{r['name']}\ns{r['seed']}" for r in rows]
    vals = [float(r.get("final_return") or 0) for r in rows]
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("final return")
    ax.set_title("Ablation final returns")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar = out.with_name(out.stem + "_bars.png")
    fig.savefig(bar, dpi=150)
    plt.close(fig)
    print("Wrote", bar)


def plot_throughput(path: Path, out: Path) -> None:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    plt = _try_matplotlib()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Throughput benchmark", ""]
    for model, blob in data.items():
        lines.append(f"## {model}")
        if "gaussian" in blob and "samples_per_sec" in blob.get("gaussian", {}):
            lines.append(f"- gaussian: {blob['gaussian']['samples_per_sec']:.1f} samples/s")
        for k, v in blob.items():
            if k.startswith("steps_") and isinstance(v, dict):
                lines.append(
                    f"- {k}: {v.get('samples_per_sec', 0):.1f} samples/s "
                    f"({v.get('ms_per_batch', 0):.2f} ms/batch)"
                )
        lines.append("")
    md = out.with_suffix(".md")
    md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", md)

    if plt is None:
        return

    labels, values = [], []
    for model, blob in data.items():
        if model == "gaussian" or ("gaussian" in blob and "samples_per_sec" in blob.get("gaussian", {})):
            g = blob.get("gaussian", blob)
            if isinstance(g, dict) and "samples_per_sec" in g:
                labels.append(f"{model}\ngaussian")
                values.append(g["samples_per_sec"])
        for k, v in blob.items():
            if k.startswith("steps_") and isinstance(v, dict):
                labels.append(f"{model}\n{k}")
                values.append(v["samples_per_sec"])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("samples / sec")
    ax.set_title("Imagination throughput")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("Wrote", png)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", type=str, default=None)
    p.add_argument("--throughput", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "plots"))
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.ablation:
        plot_ablation(Path(args.ablation), out_dir / "ablation")
    if args.throughput:
        plot_throughput(Path(args.throughput), out_dir / "throughput")
    if not args.ablation and not args.throughput:
        # defaults if present
        ab = ROOT / "runs" / "ablations" / "ablation_results.json"
        th = ROOT / "runs" / "throughput_benchmark.json"
        if ab.exists():
            plot_ablation(ab, out_dir / "ablation")
        if th.exists():
            plot_throughput(th, out_dir / "throughput")
        if not ab.exists() and not th.exists():
            print("No input JSON found. Pass --ablation and/or --throughput.")


if __name__ == "__main__":
    main()
