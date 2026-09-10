# -*- coding: utf-8 -*-
"""One-command status for a (possibly still-running) multi-worker study.

    python -m udwm.scripts.run_status --out runs/dmc_macura_30seed_15k_gpu.json \
        --expect-seeds 30 --arms ordinary identified_eq macura_gate

Reads the canonical ``--out`` (if it exists yet) plus every per-seed / per-arm
partial next to it, and prints:

  * which ``(seed, arm)`` rows are already on disk,
  * the expected total and the exact missing list,
  * a merge-readiness verdict (COMPLETE / INCOMPLETE) - the ``seeds x arms``
    row-count sanity check that a short file would otherwise slip past,
  * a rough ETA from the observed arm completion rate, and a stale-write flag,
  * the summarizer command to run once the study completes.

Pure inspection: reads JSON, writes nothing, touches no GPU.  Handles both
partial shapes in the repo - the split driver's per-seed payload
(``{stem}_seed{N}.partial.json`` with a ``rows`` list) and the probes' single
row per ``(seed, arm)`` (``{stem}_s{N}_{arm}.partial.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

_SARM_RE = re.compile(r"_s(?P<seed>\d+)_(?P<arm>.+)\.partial\.json$")


def _rows_of(path: Path):
    """Rows in ``path`` (a payload with ``rows``, or a single row dict)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None
    if isinstance(payload, dict) and "seed" in payload and "variant" in payload:
        return [payload], {}
    if isinstance(payload, dict):
        return payload.get("rows", []), payload.get("protocol", {})
    return None, None


def _next_command(out: Path) -> str:
    name = out.stem
    if "walker" in name:
        return "python -m udwm.scripts.summarize_walker2d_payoff --data {}".format(out)
    if "crn_bias" in name:
        return "python -m udwm.scripts.summarize_crn_bias --data {}".format(out)
    if "gradient_interference" in name or name.startswith("gi_"):
        return "python -m udwm.scripts.summarize_gradient_interference --data {}".format(out)
    return ("python -m udwm.scripts.summarize_dmc_payoff --data {} --no-ctrl --no-bars"
            .format(out))


def _collect(out: Path):
    """{(seed, arm): source} from the canonical file and its partials."""
    found: dict[tuple[int, str], str] = {}
    canonical_rows, protocol = (None, {})
    if out.exists():
        canonical_rows, protocol = _rows_of(out)
        for r in canonical_rows or []:
            found[(int(r["seed"]), str(r["variant"]))] = "canonical"

    first_mtime = last_mtime = None
    for p in sorted(out.parent.glob(out.stem + "*.partial.json")):
        rows, _ = _rows_of(p)
        if rows is None:
            m = _SARM_RE.search(p.name)
            if m:
                found.setdefault((int(m.group("seed")), m.group("arm")), "partial(broken)")
            continue
        for r in rows:
            found[(int(r["seed"]), str(r["variant"]))] = "partial"
        mt = p.stat().st_mtime
        first_mtime = mt if first_mtime is None else min(first_mtime, mt)
        last_mtime = mt if last_mtime is None else max(last_mtime, mt)
    return found, canonical_rows, protocol, first_mtime, last_mtime


def _pace_line(out: Path, remaining: int, round_gap: float = 300.0) -> str:
    """Rough ETA from the most recent write round.

    Deliberately does *not* use "rows on disk / time since the first write":
    a resumed or partially-downloaded study inherits rows that were not
    produced during this window, which makes that ratio wildly optimistic.
    Instead it looks at the last two distinct write times: the arms written at
    the latest one are assumed to be one parallel round, so
    throughput = arms_in_last_round / (t_last - t_prev).
    """
    paths = sorted(out.parent.glob(out.stem + "*.partial.json"))
    if not paths:
        return "no partial files"
    stamps = sorted({round(p.stat().st_mtime, 1) for p in paths})
    # Cluster nearby writes into "rounds": parallel workers finish within
    # seconds of each other, so the raw last-two-distinct-times would report a
    # sub-minute round.  A gap larger than round_gap starts a new round.
    rounds: list[list[float]] = []
    for s in stamps:
        if rounds and s - rounds[-1][-1] <= round_gap:
            rounds[-1].append(s)
        else:
            rounds.append([s])
    if len(rounds) < 2:
        return "not enough recent completions to estimate (one write round so far)"
    t_prev, t_last = rounds[-2][-1], rounds[-1][-1]
    arms_last = max(1, len(rounds[-1]))
    round_min = (t_last - t_prev) / 60.0
    if round_min <= 0 or remaining == 0:
        return "not enough recent completions to estimate"
    eta_min = remaining * round_min / arms_last
    return ("{} arms per {:.1f} min -> ~{:.0f} min ({:.1f} h) remaining "
            "(rough; last write round only)".format(arms_last, round_min, eta_min, eta_min / 60.0))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="canonical output path of the study")
    ap.add_argument("--expect-seeds", type=int, default=None,
                    help="expected seed count (seeds 0..N-1); default: protocol, else inferred")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="expected arm list; default: protocol, else inferred")
    ap.add_argument("--now", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--round-gap", type=float, default=300.0,
                    help="seconds of quiet that separate two write rounds for the "
                         "ETA estimate (default 300); raise for slow arms")
    args = ap.parse_args(argv)

    out = Path(args.out)
    found, canonical_rows, protocol, first_mtime, last_mtime = _collect(out)

    arms = list(args.arms or protocol.get("variants") or [])
    if not arms:
        arms = sorted({a for _, a in found})
    n_expected = args.expect_seeds
    if n_expected is None and protocol.get("seeds"):
        n_expected = len(protocol["seeds"])
    inferred = False
    if n_expected is None:
        seeds = sorted({s for s, _ in found})
        n_expected = (max(seeds) + 1) if seeds else 0
        inferred = True
    expected = {(s, a) for s in range(n_expected) for a in arms}

    print("study      : {}".format(out))
    print("canonical  : {}".format(
        "absent (study still running)" if canonical_rows is None
        else "{} rows".format(len(canonical_rows))))
    if not arms and not n_expected:
        print("on disk    : nothing found; pass --expect-seeds/--arms "
              "(or run from the directory holding the output)")
        return 0
    print("expected   : {} seeds x {} arms = {} rows{}".format(
        n_expected, len(arms), len(expected),
        "  (seeds inferred - pass --expect-seeds if the study has more)" if inferred else ""))
    print("on disk    : {} rows".format(len(found)))

    per_seed: dict[int, set] = {}
    for (s, a) in found:
        per_seed.setdefault(s, set()).add(a)
    complete = sorted(s for s, a in per_seed.items() if set(arms) <= a)
    print("seeds done : {}/{}".format(len(complete), n_expected))

    missing = sorted(expected - set(found))
    if not missing:
        print("verdict    : COMPLETE - every (seed, arm) is present")
    else:
        shown = ", ".join("s{}:{}".format(s, a) for s, a in missing[:10])
        print("verdict    : INCOMPLETE - {} missing: {}{}".format(
            len(missing), shown, " ..." if len(missing) > 10 else ""))

    if last_mtime:
        now = args.now if args.now is not None else time.time()
        remaining = max(0, len(expected) - len(found))
        print("pace       : {}".format(_pace_line(out, remaining, args.round_gap)))
        idle = (now - last_mtime) / 60.0
        print("last write : {:.1f} min ago{}".format(
            idle, "  <-- check the workers" if idle > 40 else ""))

    print("next       : {}".format(_next_command(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
