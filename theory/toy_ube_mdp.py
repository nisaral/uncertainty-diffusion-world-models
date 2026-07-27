"""
Toy acyclic MRP: exact posterior Var(V) vs UBE fixed point vs MC local rewards.

Graph (DAG — Assump. 2)::

    s0  --p-->  s1  --1-->  sT   (absorbing)
        \\
         (1-p)--> s2 --1--> sT

Epistemic: p ~ Unif({p_i}) over ensemble members.
Rewards at leaves: r(s1)=R1, r(s2)=R2 (known). V(sT)=0.
V(s1)=R1, V(s2)=R2 have **no** epistemic uncertainty (only s0 does).

Then V^{p}(s0) = γ (p R1 + (1-p) R2), so
    Var_p V(s0) = γ² Var_p(p R1+(1-p)R2) = γ² w(s0).

With U(leaves)=0, UBE gives U(s0)=γ² u(s0). For this MRP, u=w (gap g_t=0
because next-state values have zero epistemic noise). Hence UBE recovers Var exactly.

MC-UBE estimates w by sampling s' from each member and taking Var of member-mean
next-values (here next-value = R(s')).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class EnsembleMRP:
    ps: List[float]
    r1: float = 10.0
    r2: float = 0.0
    gamma: float = 0.9

    @property
    def n(self) -> int:
        return len(self.ps)

    def value_member(self, i: int) -> float:
        p = self.ps[i]
        return self.gamma * (p * self.r1 + (1.0 - p) * self.r2)

    def posterior_mean_var(self) -> Tuple[float, float, List[float]]:
        vals = [self.value_member(i) for i in range(self.n)]
        mean = sum(vals) / self.n
        var = sum((v - mean) ** 2 for v in vals) / self.n
        return mean, var, vals

    def closed_form_w(self) -> float:
        """w = Var_i[ E_{s'|θ_i} V(s') ] with V(s1)=r1, V(s2)=r2."""
        next_means = [p * self.r1 + (1 - p) * self.r2 for p in self.ps]
        m = sum(next_means) / self.n
        return sum((x - m) ** 2 for x in next_means) / self.n

    def closed_form_g(self) -> float:
        """Mean aleatoric Var_{s'|θ_i}[V(s')] = E_i[ p(1-p)(r1-r2)^2 ]."""
        return sum(p * (1 - p) * (self.r1 - self.r2) ** 2 for p in self.ps) / self.n

    def mc_w(self, m_samples: int, rng: random.Random) -> float:
        """MC estimate of w: sample next values under each member."""
        member_means = []
        for p in self.ps:
            ys = [self.r1 if rng.random() < p else self.r2 for _ in range(m_samples)]
            member_means.append(sum(ys) / m_samples)
        bar = sum(member_means) / self.n
        return sum((m - bar) ** 2 for m in member_means) / self.n


def main() -> None:
    mrp = EnsembleMRP(ps=[0.2, 0.5, 0.8, 0.9, 0.1], r1=10.0, r2=0.0, gamma=0.9)
    mean_v, var_v, vals = mrp.posterior_mean_var()
    w = mrp.closed_form_w()
    g = mrp.closed_form_g()
    # u = w here (next-state values have no epistemic uncertainty)
    u = w
    ube_u = (mrp.gamma**2) * u

    print("=" * 64)
    print("Toy acyclic MRP — exact Var(V) vs UBE vs MC-w")
    print("=" * 64)
    print(f"Ensemble p(s1|s0) = {mrp.ps}")
    print(f"Member V(s0)      = {[round(v, 4) for v in vals]}")
    print(f"E[V(s0)]          = {mean_v:.6f}")
    print(f"Var_p[V(s0)]      = {var_v:.6f}   <-- ground-truth U(s0)")
    print()
    print(f"closed-form w (epistemic local) = {w:.6f}")
    print(f"closed-form g (mean aleatoric)  = {g:.6f}")
    print(f"u = w (gap zero at leaves)      = {u:.6f}")
    print(f"UBE U(s0) = γ² u                = {ube_u:.6f}")
    print(f"|UBE U − Var(V)|                = {abs(ube_u - var_v):.3e}")
    print()
    print("MC estimate of w → Û = γ² ŵ  (50 seeds)")
    print(f"{'M':>6}  {'rmse_w':>12}  {'rmse_U':>12}")
    for m in (4, 8, 16, 32, 64, 128, 256):
        se_w = se_U = 0.0
        for seed in range(50):
            rng = random.Random(seed)
            w_hat = mrp.mc_w(m, rng)
            u_hat = (mrp.gamma**2) * w_hat
            se_w += (w_hat - w) ** 2
            se_U += (u_hat - var_v) ** 2
        print(f"{m:6d}  {math.sqrt(se_w/50):12.6f}  {math.sqrt(se_U/50):12.6f}")
    print()
    print("Takeaways:")
    print("  1. Exact UBE recovers posterior Var(V) on this DAG MRP.")
    print("  2. Only need local w here; MC-w error falls ~1/√M (Claim A toy).")
    print("  3. Diffusion case: replace Bernoulli draws by denoising samples.")
    print("=" * 64)


if __name__ == "__main__":
    main()
