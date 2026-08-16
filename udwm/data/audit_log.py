"""Reconstructable log of imagination-gate decisions.

SOP / Lukas angle: a refusal is only useful if a human can replay *why*.
Each row is one imagined (s, a) and the gate action. JSONL on disk.

This is an audit trail, not a security proof.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class GateAuditLog:
    def __init__(self, path: Optional[str] = None, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path) if path else None
        self.rows: List[Dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_batch(
        self,
        step: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        sqrt_u: torch.Tensor,
        weights: torch.Tensor,
        dones: torch.Tensor,
        stop_threshold: float,
        mode: str,
        max_rows: int = 64,
    ) -> None:
        if not self.enabled:
            return
        b = min(int(obs.shape[0]), max_rows)
        h = int(sqrt_u.shape[1]) if sqrt_u.ndim == 3 else 1
        su = sqrt_u.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()
        d = dones.detach().cpu().numpy()
        o = obs.detach().cpu().numpy()
        a = actions.detach().cpu().numpy()
        ts = datetime.now(timezone.utc).isoformat()
        for i in range(b):
            for t in range(h):
                score = float(su[i, t, 0]) if su.ndim == 3 else float(su[i])
                wt = float(w[i, t, 0]) if w.ndim == 3 else float(w[i])
                done = float(d[i, t, 0]) if d.ndim == 3 else float(d[i])
                if stop_threshold > 0 and score > stop_threshold:
                    decision = "abstain_stop"
                    reason = f"sqrt_u={score:.6f} > tau={stop_threshold:.6f}"
                elif wt < 0.99:
                    decision = "accept_downweighted"
                    reason = f"weight={wt:.4f} from sqrt_u={score:.6f}"
                else:
                    decision = "accept"
                    reason = "below_gate"
                row = {
                    "ts": ts,
                    "train_step": int(step),
                    "batch_i": i,
                    "horizon_t": t,
                    "sqrt_u": score,
                    "weight": wt,
                    "done": done,
                    "tau": float(stop_threshold),
                    "mode": mode,
                    "decision": decision,
                    "reason": reason,
                    "obs": o[i, t].tolist() if o.ndim == 3 else o[i].tolist(),
                    "action": a[i, t].tolist() if a.ndim == 3 else a[i].tolist(),
                }
                self.rows.append(row)
                if self.path is not None:
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")

    def summary(self) -> Dict[str, float]:
        if not self.rows:
            return {"n_logged": 0.0}
        dec = [r["decision"] for r in self.rows]
        n = len(dec)
        return {
            "n_logged": float(n),
            "frac_abstain_stop": float(sum(d == "abstain_stop" for d in dec) / n),
            "frac_downweighted": float(sum(d == "accept_downweighted" for d in dec) / n),
            "frac_accept": float(sum(d == "accept" for d in dec) / n),
        }
