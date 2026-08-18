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


def plot_selective(path: Path, out: Path) -> None:
    """Risk–coverage + over-rejection plots from eval_selective JSON."""
    with path.open(encoding="utf-8") as f:
        report = json.load(f)
    plt = _try_matplotlib()
    out.parent.mkdir(parents=True, exist_ok=True)

    rc = report.get("risk_coverage") or []
    sw = report.get("threshold_sweep") or []
    md_lines = [
        "# Selective prediction (√U as a score, not a guarantee)",
        "",
        report.get("disclaimer", ""),
        "",
        f"rank_corr = {report.get('rank_corr_score_vs_error')}",
        "",
    ]
    if rc:
        md_lines += ["| coverage | abstention | risk |", "|---:|---:|---:|"]
        for row in rc:
            md_lines.append(
                f"| {row['coverage']:.2f} | {row['abstention_rate']:.2f} | {row['risk_mean_abs_td']:.4f} |"
            )
    (out.with_suffix(".md")).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Wrote", out.with_suffix(".md"))

    if plt is None:
        print("matplotlib missing — skipped PNG")
        return

    if rc:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["coverage"] for r in rc], [r["risk_mean_abs_td"] for r in rc], marker="o")
        ax.set_xlabel("coverage (fraction kept, lowest √U)")
        ax.set_ylabel("risk (mean |TD| on kept)")
        ax.set_title("Risk–coverage")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.with_name(out.stem + "_risk_coverage.png"), dpi=150)
        plt.close(fig)
        print("Wrote", out.with_name(out.stem + "_risk_coverage.png"))

    if sw:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["tau"] for r in sw], [r["over_rejection"] for r in sw], marker="o", label="over-rejection")
        ax.plot([r["tau"] for r in sw], [r["recall_bad"] for r in sw], marker="s", label="recall of bad")
        ax.set_xlabel("threshold τ on √U")
        ax.set_ylabel("rate")
        ax.set_title("If √U is treated as an absolute threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.with_name(out.stem + "_over_rejection.png"), dpi=150)
        plt.close(fig)
        print("Wrote", out.with_name(out.stem + "_over_rejection.png"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", type=str, default=None)
    p.add_argument("--throughput", type=str, default=None)
    p.add_argument("--selective", type=str, default=None, help="JSON from eval_selective")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "plots"))
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.ablation:
        plot_ablation(Path(args.ablation), out_dir / "ablation")
    if args.throughput:
        plot_throughput(Path(args.throughput), out_dir / "throughput")
    if args.selective:
        plot_selective(Path(args.selective), out_dir / "selective")
    if not args.ablation and not args.throughput and not args.selective:
        # defaults if present
        ab = ROOT / "runs" / "ablations" / "ablation_results.json"
        th = ROOT / "runs" / "throughput_benchmark.json"
        if ab.exists():
            plot_ablation(ab, out_dir / "ablation")
        if th.exists():
            plot_throughput(th, out_dir / "throughput")
        sel = ROOT / "runs" / "selective_report.json"
        if sel.exists():
            plot_selective(sel, out_dir / "selective")
        if not ab.exists() and not th.exists() and not sel.exists():
            print("No input JSON found. Pass --ablation, --throughput, and/or --selective.")


if __name__ == "__main__":
    main()
