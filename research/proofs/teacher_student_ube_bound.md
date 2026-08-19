# Teacher--student uncertainty bound

## Setup

Fix `(s,a)` and an ensemble of transition kernels. Let `P_i` be the teacher
kernel and `Q_i` the student kernel. Define

`mu_i(P) = E_{x~P_i} f(x)`,

where `f(s') = E_{a'~pi(.|s')} Qbar(s',a')`. Assume `f` is `L_f`-Lipschitz
under the metric used by `W_1`, and let `epsilon_i = W_1(P_i,Q_i)`.

By Kantorovich--Rubinstein duality,

`|mu_i(P)-mu_i(Q)| <= L_f epsilon_i`.

Write `d_i = mu_i(Q)-mu_i(P)`, `w(P)=N^{-1} sum_i (mu_i(P)-mu_bar(P))^2`,
and likewise for `w(Q)`. Expanding the squares gives

`w(Q)-w(P) = 2 N^{-1} sum_i (mu_i(P)-mu_bar(P))(d_i-d_bar)
             + N^{-1} sum_i (d_i-d_bar)^2`.

Therefore, with `A = sqrt(w(P))` and
`D = [N^{-1} sum_i (d_i-d_bar)^2]^{1/2}`,

`|w(Q)-w(P)| <= 2 A D + D^2`.

Since `|d_i| <= L_f epsilon_i`,

`D <= L_f [N^{-1} sum_i (epsilon_i-epsilon_bar)^2]^{1/2}`.

The looser but easier-to-report bound is

`|w(Q)-w(P)| <= 2 L_f sqrt(w(P)) eps_rms + L_f^2 eps_rms^2`,

where `eps_rms^2 = N^{-1} sum_i epsilon_i^2`.

This is the key reason to preserve *centered ensemble geometry*, rather than
only minimizing average next-state MSE. A common-mode student error can move all
members together while leaving epistemic disagreement nearly unchanged; a
member-differential error directly corrupts `w`.

## Local UBE reward

Let `v_i(P)=Var_{x~P_i} f(x)` and `g(P)=N^{-1}sum_i v_i(P)`. The same argument
does not hold from Wasserstein alone for second moments unless `f` is bounded or
has controlled growth. Under `|f|<=B`, a coupling `(X_i,Y_i)` with
`E|X_i-Y_i|<=epsilon_i` and an additional bounded-difference condition gives

`|v_i(P)-v_i(Q)| <= 4 B L_f epsilon_i`.

Thus

`|u(Q)-u(P)| <= |w(Q)-w(P)| + 4 B L_f eps_avg`.

This is a deliberately conservative bound. A paper should state the exact
regularity assumption used for the chosen state metric; without it, a small
Wasserstein distance does not control unbounded critic second moments.

## Propagated UBE error

For an exact finite-horizon UBE operator `T_u` and an approximate operator
`T_v`, the transition expectation is non-expansive in the sup norm and the
discount factor is `gamma^2`:

`||T_u F - T_u G||_inf <= gamma^2 ||F-G||_inf`.

If `||u-v||_inf <= eta`, and `U` and `V` are the respective fixed points,

`||U-V||_inf <= gamma^2 eta + gamma^2 ||U-V||_inf`,

so

`||U-V||_inf <= gamma^2 eta / (1-gamma^2)`.

Combining this with the local bound gives a teacher--student guarantee for
decision-relevant uncertainty. It does not claim that the student reproduces
the teacher transition law everywhere; it claims that controlled transport and
critic regularity limit the uncertainty error.

## What the implementation optimizes

`udwm.models.consistency.uncertainty_preserving_distill_loss` uses:

1. member-wise teacher/student clean-state matching;
2. ensemble-mean matching;
3. centered member-deviation matching;
4. pairwise squared-distance matching.

Terms (3) and (4) are the empirical proxies for the centered variance and its
geometry in the expansion above. They should be evaluated against a baseline
student trained with member-wise MSE only.

## Failure cases

- A critic with sharp discontinuities violates the Lipschitz assumption.
- Matching state-space geometry may not preserve value geometry if the critic is
  highly nonlinear; add a direct Q-disagreement loss in later experiments.
- The theorem controls the student relative to the *teacher sampler*; teacher
  approximation to the real environment remains a separate error.
- A finite ensemble is a posterior approximation, not a Bayesian guarantee.
