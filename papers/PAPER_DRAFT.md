# Does distillation preserve decision-relevant uncertainty in diffusion world models?

**Working draft — not submission-ready.**  
**Claim to test:** whether a one-step student can preserve the teacher ensemble’s
epistemic disagreement, not only next-state accuracy, at fixed NFE.  
**Code:** https://github.com/nisaral/uncertainty-diffusion-world-models

This note replaces an earlier outline that treated Monte Carlo UBE for
diffusion as the main contribution. MC-UBE is the **estimator** used to
measure the object. The paper-shaped question is **preservation under
distillation**.

---

## 1. Setup

Teacher: ensemble of conditional diffusion dynamics \(p_{\theta_i}(s'\mid s,a)\),
sampled with \(K\)-step DDIM.

Student: ensemble of one-step maps \(q_{\psi_i}(s'\mid s,a)\), one network
evaluation.

Decision object: Luis local rewards. For a value map \(f(s')=\bar Q(s',a')\),

\[
w = N^{-1}\sum_i\bigl(\mu_i-\bar\mu\bigr)^2,\qquad
u = w - g,
\]

with \(\mu_i=\mathbb{E}_{q_i}[f]\) and \(g\) the mean within-member variance of
\(f\). In this repo these are estimated by Monte Carlo (`MCUBELocalRewards`)
because diffusion members do not expose closed-form means.

## 2. Distillation objectives

**Ordinary.** Member-wise MSE between student \(x_0\) and teacher reconstruction
on the same noisy input.

**Geometry.** The same term, plus MSE on the ensemble mean, on centered member
deviations, and on pairwise squared distances. These are the empirical
objects that determine one-step ensemble variance and, after a Lipschitz
value map, the local disagreement \(w\).

A short perturbation argument is in
`research/proofs/teacher_student_ube_bound.md`:

\[
\lvert w(Q)-w(P)\rvert \le 2\sqrt{w(P)}\,D + D^2,
\]

where \(D\) is the RMS of **centered** member-mean errors. Common-mode error
does not appear in \(D\).

## 3. Evaluation

On a shared replay batch, report:

| Metric | Role |
|---|---|
| next-state MSE | average transition accuracy |
| \(u,w\) rank correlation (teacher vs student) | does the student rank the same states as uncertain? |
| \(u,w\) MAE / RMSE | magnitude of the scramble |
| NFE | inference budget (teacher \(K\) vs student \(1\)) |

The critic and policy used inside \(f\) are **held fixed** so the gap is about
the sampler, not a changed RL training run.

## 4. What is not claimed

- A new UBE.
- Conformal coverage of \(\sqrt{U}\).
- First gated imagination (see MACURA / Nguyen / Kalweit).
- Results beyond Pendulum-scale until those experiments exist.

## 5. Status

Implemented: both losses, teacher freeze, evaluator, matched ablation runner.
Not yet: multi-seed Pendulum table that would support or kill the hypothesis.
