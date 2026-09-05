"""Split-by-seed driver + merger for the registered 10-seed policy 2x2.

Runs the pre-registered 5-arm protocol
(ordinary / hybrid / lagged_hybrid / identified_hybrid / lagged_identified,
10 seeds x 1,800 env steps) as one subprocess per seed so the table finishes
in ~1/N_jobs of serial wall time on a multi-core machine, then merges the
per-seed payloads into the canonical output file using the same pairing and
summary logic as ``run_delayed_bimodal_policy_ablation.main``.

Semantics are identical to a single serial batch: each seed's five arms share
one prepared teacher (exact checksum pairing, gap 0) and each arm calls
``set_seed(seed)`` before training. Splitting by seed is only a process-
topology choice; no arm is added, dropped, or re-weighted post-hoc.

Usage:
    python -m udwm.scripts.run_policy_2x2_split_seeds \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out runs/policy_identifiability_2x2_10seed.json \
        --jobs 10 --threads 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from udwm.scripts.run_delayed_bimodal_policy_ablation import (
    VARIANTS,
    build_pairing,
    summarize,
)

VARIANT_ORDER = ["ordinary", "hybrid", "lagged_hybrid", "identified_hybrid", "lagged_identified"]


def seed_file(out: Path, seed: int) -> Path:
    return out.with_name(f"{out.stem}_seed{seed}.partial.json")


def seed_complete(path: Path, variants) -> bool:
    if not path.exists():
        return False
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    return {r["variant"] for r in rows} >= set(variants)


def run_seed(seed: int, args, out: Path, gpu_id=None) -> int:
    cmd = [
        sys.executable,
        "-m",
        "udwm.scripts.run_delayed_bimodal_policy_ablation",
        "--config",
        args.config,
        "--variants",
        *args.variants,
        "--seeds",
        str(seed),
        "--steps",
        str(args.steps),
        "--out",
        str(seed_file(out, seed)),
    ]
    if args.device is not None:
        cmd += ["--device", args.device]
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = str(max(1, int(args.threads)))
    env["MKL_NUM_THREADS"] = str(max(1, int(args.threads)))
    proc = subprocess.run(cmd, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return proc.returncode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delayed_bimodal_distill.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ORDER),
                        help="arm subset (default: all five 2x2 arms)")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--out", default="runs/policy_identifiability_2x2_10seed.json")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default=None,
                        help="device override passed to each seed (cpu/cuda); default: config value")
    parser.add_argument("--gpu-ids", default=None,
                        help="comma list of CUDA device ids cycled across seed workers; implies --device cuda")
    parser.add_argument("--keep-partials", action="store_true")
    parser.add_argument("--existing", default=None,
                        help="canonical JSON whose rows are merged for seeds without partial files")
    args = parser.parse_args(argv)

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
        if not gpu_ids:
            raise SystemExit("--gpu-ids must be a comma list, e.g. 0,1")
        if args.device is None:
            args.device = "cuda"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if args.existing:
        existing_rows = json.loads(Path(args.existing).read_text(encoding="utf-8")).get("rows", [])
    existing_seeds = {
        int(r["seed"]) for r in existing_rows if r["variant"] in set(args.variants)
    }
    covered = {s: seed_complete(seed_file(out, s), args.variants) or s in existing_seeds for s in args.seeds}
    pending = [s for s in args.seeds if not covered[s]]
    print(f"[split] {len(args.seeds) - len(pending)} seeds already available; {len(pending)} pending")

    if pending:
        def worker(seed: int, gpu_id) -> int:
            if gpu_id is not None:
                print(f"[split] seed {seed} -> cuda:{gpu_id}")
            rc = run_seed(seed, args, out, gpu_id)
            if rc != 0:
                raise SystemExit(f"[split] seed {seed} failed with exit {rc}")
            print(f"[split] seed {seed} done")
            return seed

        n_jobs = max(1, min(int(args.jobs), len(pending)))
        print(f"[split] launching {len(pending)} seeds across {n_jobs} workers "
              f"({args.threads} torch threads each)")
        if gpu_ids and n_jobs > len(gpu_ids):
            print(f"[split] note: {n_jobs} workers > {len(gpu_ids)} GPUs; several seeds share a device, use --jobs <= #gpus to avoid contention")
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(worker, s, gpu_ids[i % len(gpu_ids)] if gpu_ids else None)
                for i, s in enumerate(pending)
            ]
            for fut in as_completed(futures):
                fut.result()

    rows: list[dict] = []
    for s in args.seeds:
        path = seed_file(out, s)
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8")).get("rows", []))
            continue
        want = set(args.variants)
        src = [r for r in existing_rows if int(r["seed"]) == s and r["variant"] in want]
        rows.extend(src)
        if len(src) < len(want):
            raise SystemExit(f"[split] no rows available for seed {s}: {len(src)}/{len(want)} arms")
    order = {v: i for i, v in enumerate(args.variants)}
    rows.sort(key=lambda r: (int(r["seed"]), order.get(r["variant"], 99)))
    payload = {
        "protocol": {
            "config": args.config,
            "seeds": list(args.seeds),
            "steps": args.steps,
            "variants": list(args.variants),
            "out": str(out),
            "resume": False,
            "driver": "run_policy_2x2_split_seeds.py (per-seed batches, merged)",
            "device": args.device if args.device else "config",
            "gpu_ids": gpu_ids,
        },
        "rows": rows,
        "teacher_pairing": build_pairing(rows),
        "summary": summarize(rows),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[split] merged {len(rows)} rows -> {out}")
    for s in sorted({int(r["seed"]) for r in rows}):
        print(f"[split] seed {s}: {json.dumps(payload['teacher_pairing'].get(str(s), {}))}")
    if not args.keep_partials:
        for s in args.seeds:
            seed_file(out, s).unlink(missing_ok=True)
        print("[split] removed partial files (--keep-partials to retain)")


if __name__ == "__main__":
    main()
