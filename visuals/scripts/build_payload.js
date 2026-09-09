// build_payload.js - regenerate the committed visualization payloads from the
// local (gitignored) run files, then rebuild the static SVGs and the
// self-contained v4 explorer page.
//
//   Usage:  node visuals/scripts/build_payload.js   (from repo root)
//
// Sources (all exact_teacher_match, checksum gap 0):
//   runs/dmc_payoff_30seed_15k_gpu.json          DMC hopper-hop, 30 seeds x 6 arms
//   runs/dmc_payoff_30seed_15k_gpu_ctrl.json     DMC ordinary_gate_off, 30 seeds
//   runs/policy_combined_fix_10seed.json         DelayedBimodal combined fix, 10 seeds
//   runs/dmc_sanity_15k_gpu.json                 DMC sanity, 10 seeds (curves)
// Adjudicated reference values below come from
// research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md (byte-verified run)
// and research/RESULTS-COMBINED-FIX-POLICY-2026-09-07.md; the script FAILS if
// the numbers it recomputes drift from them.
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ROOT = path.resolve(__dirname, "../..");
const SRC = (f) => path.join(ROOT, "runs", f);
const OUT = (f) => path.join(ROOT, "visuals", f);

// ---------------------------------------------------------------------------
// Arm registry: raw variant key -> stable display info (color + one-line gloss).
// ---------------------------------------------------------------------------
const ARMS = [
  { key: "ordinary",            display: "ordinary",   color: "#1d4ed8",
    gloss: "one-step distillation of the frozen teacher mean; rollout gating uses the student U score (self-gated). No matched-uncertainty term." },
  { key: "lagged_hybrid",       display: "lagged_hybrid", color: "#c2410c",
    gloss: "M=1 conflated objective (single shared latent, matched scalar) evaluated against the lagged target critic with value normalization + guard." },
  { key: "hybrid",              display: "hybrid",     color: "#f59e0b",
    gloss: "M=1 conflated objective: matches teacher vs student value disagreement at ONE shared latent against the live critic. The non-identifiable baseline." },
  { key: "identified_hybrid",   display: "EMA",        color: "#dc2626",
    gloss: "identified_hybrid: the M>=2 equal-weight fix WITH the per-term EMA reweighting on. The aleatoric channel is starved -> U collapses. This is the collapse arm." },
  { key: "identified_eq",       display: "eq",         color: "#16a34a",
    gloss: "identified_eq: M>=2 equal-weight identified loss (paired latents estimate w and g separately), live critic. The mechanism-transfer arm." },
  { key: "lagged_identified_eq",display: "lagged eq",  color: "#0f766e",
    gloss: "lagged_identified_eq: M>=2 equal-weight + lagged target critic + normalization + guard. The combined-fix arm (tops DelayedBimodal, not DMC)." },
];
const ARM_BY_KEY = Object.fromEntries(ARMS.map((a) => [a.key, a]));

// ---------------------------------------------------------------------------
// Small stats helpers (no numpy needed; endpoints only).
// ---------------------------------------------------------------------------
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
function median(xs) {
  const m = xs.slice().sort((a, b) => a - b);
  const n = m.length;
  return n % 2 ? m[(n - 1) / 2] : (m[n / 2 - 1] + m[n / 2]) / 2;
}
const sha16 = (buf) => crypto.createHash("sha256").update(buf).digest("hex").slice(0, 16);

function load(file) {
  const raw = fs.readFileSync(file);
  const j = JSON.parse(raw.toString("utf8"));
  return { rows: j.rows, pairing: j.teacher_pairing || {}, summary: j.summary || {},
           sha: sha16(raw), bytes: raw.length };
}
function pairingOk(pairing, nRows) {
  const entries = Object.values(pairing);
  if (!entries.length) return null;
  const allExact = entries.every((p) => p.exact_teacher_match === true);
  const maxGap = Math.max(...entries.map((p) => Number(p.max_teacher_checksum_gap || 0)));
  return { allExact, maxGap, arms: entries[0] ? entries[0].n_arms : null };
}

// Row-level final-eval fields are the adjudicated fields (see summarize_dmc_payoff.py).
function aggArm(rows, key) {
  const rs = rows.filter((r) => r.variant === key);
  if (!rs.length) return null;
  const u = rs.map((r) => r.u_rank_corr);
  const w = rs.map((r) => r.w_rmse);
  const ns = rs.map((r) => r.next_state_mse);
  const ret = rs.map((r) => r.final_return);
  const gr = rs.map((r) => r.teacher_g_mean > 0 ? r.student_g_mean / r.teacher_g_mean : NaN);
  return {
    key,
    n: rs.length,
    u_mean: mean(u),
    u_med: median(u),
    n_ge_070: u.filter((x) => x >= 0.7).length,
    w_rmse_med: median(w),
    w_rmse_mean: mean(w),
    ns_mse_mean: mean(ns),
    ret_mean: mean(ret),
    ret_med: median(ret),
    g_ratio_mean: mean(gr.filter((x) => !Number.isNaN(x))),
    per_seed_u: u,
    per_seed_ret: ret,
    seeds: rs.map((r) => r.seed).sort((a, b) => a - b),
  };
}

// Last eval record at each milestone step (the log appends a forward-filled copy
// of the previous milestone as the FIRST record of the next one).
function lastEvalAt(row, step) {
  const ev = row.eval_history.filter((e) => e.step === step && e.u_rank_corr !== undefined);
  return ev.length ? ev[ev.length - 1] : null;
}

// ---------------------------------------------------------------------------
// Payload construction.
// ---------------------------------------------------------------------------
const NOW = new Date().toISOString().slice(0, 10);

function dmc30Payload() {
  const d = load(SRC("dmc_payoff_30seed_15k_gpu.json"));
  const arms = ARMS.map((a) => aggArm(d.rows, a.key));
  const pair = pairingOk(d.pairing, d.rows.length);
  const byKey = Object.fromEntries(arms.map((a) => [a.key, a]));
  return {
    meta: {
      env: "DMC hopper-hop (dm_control/hopper-hop-v0)",
      budget: "15,000 env steps (Amendment 2 adjudication budget)",
      n_seeds: 30,
      arms: ARMS.map((a) => a.key),
      source_file: "runs/dmc_payoff_30seed_15k_gpu.json",
      source_rows: d.rows.length,
      source_sha16: d.sha,
      source_bytes: d.bytes,
      exact_teacher_match: pair ? pair.allExact : null,
      max_checksum_gap: pair ? pair.maxGap : null,
      adjudication_doc: "research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md",
      adjudicator: "udwm/scripts/summarize_dmc_payoff.py",
      generated: NOW,
      generated_by: "visuals/scripts/build_payload.js",
    },
    arms,
    verdict: {
      bar1_eq_ge_070: { need: 21, got: byKey.identified_eq.n_ge_070, met: byKey.identified_eq.n_ge_070 >= 21,
        note: "eq u_rank >= 0.70 on >= 21/30 seeds" },
      bar2_eq_vs_hybrid: { delta: -0.020, ci: [-0.067, 0.030], wins: 13, need: 21, met: false,
        note: "paired eq - hybrid, bootstrap 95% CI, 1e5 draws, rng seed 0" },
      mechanism_eq_vs_ema: { delta: 0.788, ci: [0.736, 0.840], wins: 30, met: true,
        note: "paired eq - EMA (identified_hybrid) u_rank" },
      collapse_control: { ema_mean: byKey.identified_hybrid.u_mean, ema_ge_070: byKey.identified_hybrid.n_ge_070,
        ordinary_minus_ema: { delta: 1.034, ci: [0.988, 1.079], wins: 30, met: true } },
      lagged_eq_reversal: { delta: -0.073, ci: [-0.111, -0.036], wins: 7, met: false,
        note: "paired lagged_eq - eq on DMC (DB had lagged_eq above eq)" },
      eq_vs_ordinary: { delta: -0.246, ci: [-0.278, -0.213], wins: 0 },
      hybrid_vs_ordinary: { delta: -0.226, ci: [-0.274, -0.183], wins: 0 },
      lagged_hybrid_vs_hybrid: { delta: 0.106, ci: [0.059, 0.153], wins: 22 },
    },
  };
}

function ctrlPayload() {
  const d = load(SRC("dmc_payoff_30seed_15k_gpu_ctrl.json"));
  const key = "ordinary_gate_off";
  const arm = aggArm(d.rows, key);
  arm.display = "ordinary, gate off";
  arm.gloss = "ordinary distillation with the U-gate disabled (Addendum 4 control): separates 'preserved U helps gating' from 'gating helps at all'.";
  const pair = pairingOk(d.pairing, d.rows.length);
  return {
    meta: {
      env: "DMC hopper-hop",
      budget: "15,000 env steps",
      n_seeds: 30,
      source_file: "runs/dmc_payoff_30seed_15k_gpu_ctrl.json",
      source_rows: d.rows.length,
      source_sha16: d.sha,
      source_bytes: d.bytes,
      exact_teacher_match: pair ? pair.allExact : null,
      max_checksum_gap: pair ? pair.maxGap : null,
      doc: "research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md (Addendum 4 reading rule)",
      generated: NOW,
      generated_by: "visuals/scripts/build_payload.js",
    },
    arms: [arm],
    reading: {
      rule: "registered reading rule (ii): if the paired return CI includes 0, gating is return-neutral at this budget",
      gate_off_minus_ordinary_return: { delta: 0.025, ci: [-0.057, 0.109], wins: 15, includes_zero: true },
      descriptive: { gate_off_u_mean: arm.u_mean, ordinary_u_mean: 0.9573 },
    },
  };
}

function dbPayload() {
  const d = load(SRC("policy_combined_fix_10seed.json"));
  const arms = ARMS.map((a) => aggArm(d.rows, a.key));
  const pair = pairingOk(d.pairing, d.rows.length);
  return {
    meta: {
      env: "DelayedBimodal (toy continuous control)",
      budget: "1,800 env steps",
      n_seeds: 10,
      source_file: "runs/policy_combined_fix_10seed.json",
      source_rows: d.rows.length,
      source_sha16: d.sha,
      source_bytes: d.bytes,
      exact_teacher_match: pair ? pair.allExact : null,
      max_checksum_gap: pair ? pair.maxGap : null,
      adjudication_doc: "research/RESULTS-COMBINED-FIX-POLICY-2026-09-07.md",
      adjudicator: "udwm/scripts/summarize_combined_fix.py",
      generated: NOW,
      generated_by: "visuals/scripts/build_payload.js",
    },
    arms,
    headline: {
      lagged_eq_u_mean: arms.find((a) => a.key === "lagged_identified_eq").u_mean,
      lagged_eq_ge_070: arms.find((a) => a.key === "lagged_identified_eq").n_ge_070,
      ordinary_u_mean: arms.find((a) => a.key === "ordinary").u_mean,
      note: "on DelayedBimodal the combined-fix arm (lagged eq) tops the table 10/10; on DMC it drops below eq (7/30 reversal).",
    },
  };
}

function curvesPayload() {
  const d = load(SRC("dmc_sanity_15k_gpu.json"));
  const steps = [3000, 6000, 9000, 12000, 15000];
  const arms = ARMS.map((a) => {
    const rows = d.rows.filter((r) => r.variant === a.key);
    const at = (step, pick) => {
      const vals = rows.map((r) => {
        const e = lastEvalAt(r, step);
        return e ? pick(e) : NaN;
      }).filter((x) => !Number.isNaN(x));
      return vals.length ? mean(vals) : NaN;
    };
    const finalEvals = rows.map((r) => lastEvalAt(r, 15000)).filter(Boolean);
    const uRank = steps.map((s) => at(s, (e) => e.u_rank_corr));
    const gRatio = steps.map((s) => at(s, (e) => (e.teacher_g_mean > 0 ? e.student_g_mean / e.teacher_g_mean : NaN)));
    const studG = steps.map((s) => at(s, (e) => e.student_g_mean));
    const teachG = steps.map((s) => at(s, (e) => e.teacher_g_mean));
    return {
      key: a.key,
      n: rows.length,
      u_rank_mean: uRank,
      g_ratio_mean: gRatio,
      student_g_mean: studG,
      teacher_g_mean: teachG,
      final_u_rank_mean: mean(finalEvals.map((e) => e.u_rank_corr)),
    };
  });
  const pair = pairingOk(d.pairing, d.rows.length);
  return {
    meta: {
      env: "DMC hopper-hop sanity (S1, diagnostic only - never adjudicative)",
      budget: "15,000 env steps; eval-side fields at 3k/6k/9k/12k/15k",
      n_seeds: 10,
      source_file: "runs/dmc_sanity_15k_gpu.json",
      source_rows: d.rows.length,
      source_sha16: d.sha,
      source_bytes: d.bytes,
      exact_teacher_match: pair ? pair.allExact : null,
      max_checksum_gap: pair ? pair.maxGap : null,
      note: "each milestone step stores a forward-filled copy of the previous eval as its first record; we plot the LAST record at each milestone (the fresh eval). Teacher g decays over training (SAC reward scaling), so use g_ratio_mean for the collapse signal.",
      generated: NOW,
      generated_by: "visuals/scripts/build_payload.js",
    },
    steps,
    arms,
  };
}

// ---------------------------------------------------------------------------
// Verification against the adjudicated tables (fail loudly on drift).
// ---------------------------------------------------------------------------
function verifyDmc(p) {
  const ref = { ordinary: 0.9573, lagged_hybrid: 0.8367, hybrid: 0.7310,
                identified_hybrid: -0.0763, identified_eq: 0.7112, lagged_identified_eq: 0.6378 };
  const refGe = { ordinary: 30, lagged_hybrid: 27, hybrid: 21, identified_hybrid: 0,
                  identified_eq: 15, lagged_identified_eq: 8 };
  const tol = 0.0025;
  const bad = [];
  for (const a of p.arms) {
    const want = ref[a.key];
    if (Math.abs(a.u_mean - want) > tol) bad.push(a.key + " u_mean " + a.u_mean.toFixed(4) + " vs " + want);
    if (a.n_ge_070 !== refGe[a.key]) bad.push(a.key + " n_ge_070 " + a.n_ge_070 + " vs " + refGe[a.key]);
  }
  return bad;
}
function verifyDb(p) {
  const ref = { ordinary: 0.9340, hybrid: 0.6356, lagged_hybrid: 0.9456,
                identified_hybrid: 0.1137, identified_eq: 0.8775, lagged_identified_eq: 0.9482 };
  const tol = 0.0025;
  const bad = [];
  for (const a of p.arms) {
    const want = ref[a.key];
    if (Math.abs(a.u_mean - want) > tol) bad.push(a.key + " u_mean " + a.u_mean.toFixed(4) + " vs " + want);
  }
  return bad;
}

// ---------------------------------------------------------------------------
// SVG V1 - identifiability geometry (conceptual; measured-axis annotation).
// ---------------------------------------------------------------------------
function svgV1() {
  const W = 940, H = 660;
  const esc = (s) => s; // content is authored, no user input
  const P = [];
  const push = (s) => P.push(s);
  push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Segoe UI, Arial, sans-serif">`);
  push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`);
  push(`<text x="40" y="38" font-size="25" font-weight="700" fill="#0f172a">Why matching the teacher's uncertainty is not enough</text>`);
  push(`<text x="40" y="66" font-size="15" fill="#475569">The teacher's value-disagreement signal is one scalar, ${esc("w* + g* - Sigma-bar")}: two unknowns. Students can keep that number</text>`);
  push(`<text x="40" y="86" font-size="15" fill="#475569">by trading ensemble disagreement w for aleatoric noise g. Whether the trade helps or destroys the ranking is a property of the value map.</text>`);

  // --- left panel: M = 1 ---
  const px0 = 120, px1 = 460, py0 = 250, py1 = 545; // plot box (data: w,g in [0,100])
  const X = (w) => px0 + (w / 100) * (px1 - px0);
  const Y = (g) => py1 - (g / 100) * (py1 - py0);
  const panel = (x, title, sub) => {
    push(`<rect x="${x}" y="150" width="420" height="425" rx="10" fill="#f8fafc" stroke="#e2e8f0"/>`);
    push(`<text x="${x + 20}" y="180" font-size="17" font-weight="700" fill="#0f172a">${title}</text>`);
    push(`<text x="${x + 20}" y="200" font-size="12.5" fill="#64748b">${sub}</text>`);
  };
  const axes = (x) => {
    push(`<line x1="${x + 60}" y1="${py1}" x2="${x + 380}" y2="${py1}" stroke="#94a3b8" stroke-width="1.4"/>`);
    push(`<line x1="${x + 60}" y1="${py0}" x2="${x + 60}" y2="${py1}" stroke="#94a3b8" stroke-width="1.4"/>`);
    push(`<text x="${x + 390}" y="${py1 + 16}" font-size="13" fill="#334155" font-style="italic">w (ensemble disagreement)</text>`);
    push(`<text x="${x + 46}" y="${py0 - 10}" font-size="13" fill="#334155" font-style="italic" text-anchor="end">g (aleatoric noise)</text>`);
    push(`<text x="${x + 48}" y="${py1 + 26}" font-size="11.5" fill="#94a3b8">0</text>`);
    push(`<text x="${x + 388}" y="${py1 + 26}" font-size="11.5" fill="#94a3b8">100</text>`);
  };

  // M=1: strip 74 <= w+g <= 86 through (w*,g*) = (10,70) (schematic)
  panel(40, "M = 1  -  one shared latent", "loss = matched scalar; level sets are lines of slope -1");
  const strip = [[0, 86], [86, 0], [74, 0], [0, 74]];
  const poly = strip.map(([w, g]) => `${X(w).toFixed(1)},${Y(g).toFixed(1)}`).join(" ");
  push(`<polygon points="${poly}" fill="#dbeafe" opacity="0.65"/>`);
  push(`<line x1="${X(0)}" y1="${Y(86)}" x2="${X(86)}" y2="${Y(0)}" stroke="#3b82f6" stroke-width="1.6" stroke-dasharray="5 4"/>`);
  push(`<line x1="${X(0)}" y1="${Y(74)}" x2="${X(74)}" y2="${Y(0)}" stroke="#3b82f6" stroke-width="1.6" stroke-dasharray="5 4"/>`);
  push(`<text x="${X(78)}" y="${Y(80) - 6}" font-size="12" fill="#1d4ed8">sublevel set = a strip</text>`);
  push(`<text x="${X(74)}" y="${Y(70) - 24}" font-size="12" fill="#1d4ed8">(whole line of zero-loss students)</text>`);
  // degenerate direction arrow
  const ax = X(30), ay = Y(100) - 26; // hmm inside panel: place arrow along band upper
  // true point
  const tw = 10, tg = 70;
  push(`<circle cx="${X(tw)}" cy="${Y(tg)}" r="6" fill="#f59e0b" stroke="#b45309" stroke-width="1.6"/>`);
  push(`<text x="${X(tw) + 10}" y="${Y(tg) - 10}" font-size="13" font-weight="700" fill="#b45309">(w*, g*) teacher</text>`);
  const students = [[2, 78], [6, 74], [14, 66], [18, 62], [26, 54]];
  students.forEach(([w, g]) => push(`<circle cx="${X(w)}" cy="${Y(g)}" r="5" fill="#475569"/>`));
  push(`<text x="${X(20)}" y="${Y(76) - 12}" font-size="12.5" fill="#0f172a" font-weight="600">students: same loss,</text>`);
  push(`<text x="${X(20)}" y="${Y(72) - 12}" font-size="12.5" fill="#0f172a" font-weight="600">wrong (w, g)</text>`);
  push(`<text x="${X(0)}" y="${Y(98) + 4}" font-size="12" fill="#475569">freedom along the strip:</text>`);
  push(`<text x="${X(0)}" y="${Y(93) + 4}" font-size="12" fill="#475569">w + g = const can hide w-collapse</text>`);

  // M>=2: ball around (w*,g*)
  panel(480, "M >= 2  -  paired latents", "two draws per state estimate w and g separately (identified)");
  push(`<ellipse cx="${X(10)}" cy="${Y(70)}" rx="26" ry="24" fill="#dcfce7" opacity="0.8" stroke="#16a34a" stroke-width="1.6"/>`);
  push(`<text x="${X(10) - 62}" y="${Y(62) + 6}" font-size="12" fill="#15803d">sublevel set collapses</text>`);
  push(`<text x="${X(10) - 62}" y="${Y(57) + 6}" font-size="12" fill="#15803d">to a small ball (rate ~ sqrt eps)</text>`);
  push(`<line x1="${X(1)}" y1="${Y(79)}" x2="${X(19)}" y2="${Y(61)}" stroke="#cbd5e1" stroke-width="1.6" stroke-dasharray="3 5"/>`);
  push(`<text x="${X(14)}" y="${Y(84)}" font-size="11.5" fill="#64748b" font-style="italic">M=1 degeneracy removed</text>`);
  push(`<circle cx="${X(10)}" cy="${Y(70)}" r="6" fill="#f59e0b" stroke="#b45309" stroke-width="1.6"/>`);
  push(`<text x="${X(10) + 10}" y="${Y(70) - 10}" font-size="13" font-weight="700" fill="#b45309">(w*, g*)</text>`);
  const cluster = [[8.8, 70.6], [10.6, 69.4], [9.6, 69.8], [11.2, 70.8], [9.2, 71.2]];
  cluster.forEach(([w, g]) => push(`<circle cx="${X(w)}" cy="${Y(g)}" r="4" fill="#16a34a"/>`));
  push(`<text x="${X(20)}" y="${Y(52)}" font-size="12.5" font-weight="600" fill="#0f172a">students cluster on</text>`);
  push(`<text x="${X(20)}" y="${Y(47)}" font-size="12.5" font-weight="600" fill="#0f172a">the true point</text>`);

  // bottom: measured-axis + mechanism annotation
  push(`<rect x="40" y="600" width="860" height="46" rx="8" fill="#f1f5f9"/>`);
  push(`<text x="56" y="620" font-size="13" font-weight="700" fill="#0f172a">Measured axes (DelayedBimodal policy runs): teacher g* ~ 73 vs teacher w* ~ 0.007 - the g &gt;&gt; w hole that EMA reweighting turns into a U collapse.</text>`);
  push(`<text x="56" y="638" font-size="13" fill="#334155">Equal-weight M&gt;=2 recovers both components; the EMA-reweighted arm (identified_hybrid) starves g and U falls to ~0 (DMC 30/30 collapse control). Schematic - axes not to scale.</text>`);
  push(`</svg>`);
  return P.join("\n");
}

// Corrected V1: X/Y map relative to each panel origin (left ox=40, right ox=480).
function svgV1() {
  const W = 940, H = 660;
  const P = [];
  const push = (s) => P.push(s);
  const X = (ox, w) => ox + 60 + (w / 100) * 320;   // data w in [0,100]
  const Y = (g) => 545 - (g / 100) * 295;           // data g in [0,100]
  const dot = (cx, cy, r, fill, stroke) => push(`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}" fill="${fill}"${stroke ? ` stroke="${stroke}" stroke-width="1.6"` : ""}/>`);
  const txt = (x, y, s, size, fill, extra) => push(`<text x="${x}" y="${y}" font-size="${size}" fill="${fill}"${extra || ""}>${s}</text>`);
  push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Segoe UI, Arial, sans-serif">`);
  push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`);
  txt(40, 38, "Why matching the teacher's uncertainty is not enough", 25, "#0f172a", ' font-weight="700"');
  txt(40, 64, "The teacher's value-disagreement signal is ONE scalar,  w* + g* - Sigma-bar:  two unknowns.", 14.5, "#475569");
  txt(40, 84, "A student can keep that number by trading ensemble disagreement w for aleatoric noise g.", 14.5, "#475569");
  txt(40, 104, "Whether the trade helps or destroys the ranking is a property of the value map.", 14.5, "#475569");

  // Panels
  const panelRect = (ox, title, sub) => {
    push(`<rect x="${ox}" y="145" width="420" height="425" rx="10" fill="#f8fafc" stroke="#e2e8f0"/>`);
    txt(ox + 20, 172, title, 17, "#0f172a", ' font-weight="700"');
    txt(ox + 20, 192, sub, 12.5, "#64748b");
  };
  const axes = (ox) => {
    push(`<line x1="${ox + 60}" y1="545" x2="${ox + 380}" y2="545" stroke="#94a3b8" stroke-width="1.4"/>`);
    push(`<line x1="${ox + 60}" y1="250" x2="${ox + 60}" y2="545" stroke="#94a3b8" stroke-width="1.4"/>`);
    txt(ox + 398, 561, "w  (ensemble disagreement)", 13, "#334155", ' font-style="italic"');
    txt(ox + 46, 244, "g  (aleatoric noise)", 13, "#334155", ' font-style="italic" text-anchor="end"');
    txt(ox + 48, 569, "0", 11.5, "#94a3b8");
    txt(ox + 388, 569, "100", 11.5, "#94a3b8");
    // faint grid ticks
    for (const v of [25, 50, 75]) {
      push(`<line x1="${ox + 60 + (v / 100) * 320}" y1="250" x2="${ox + 60 + (v / 100) * 320}" y2="545" stroke="#e2e8f0" stroke-width="1"/>`);
      push(`<line x1="${ox + 60}" y1="${545 - (v / 100) * 295}" x2="${ox + 380}" y2="${545 - (v / 100) * 295}" stroke="#e2e8f0" stroke-width="1"/>`);
    }
  };

  // ---- Left: M = 1 ----
  panelRect(40, "M = 1  -  one shared latent", "loss = matched scalar; level sets are lines of slope -1");
  axes(40);
  const strip = [[0, 86], [86, 0], [74, 0], [0, 74]];
  const poly = strip.map(([w, g]) => `${X(40, w).toFixed(1)},${Y(g).toFixed(1)}`).join(" ");
  push(`<polygon points="${poly}" fill="#dbeafe" opacity="0.7"/>`);
  push(`<line x1="${X(40, 0).toFixed(1)}" y1="${Y(86).toFixed(1)}" x2="${X(40, 86).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5 4"/>`);
  push(`<line x1="${X(40, 0).toFixed(1)}" y1="${Y(74).toFixed(1)}" x2="${X(40, 74).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5 4"/>`);
  txt(176, 288, "sublevel set = a strip:", 12, "#1d4ed8", ' font-weight="600"');
  txt(176, 304, "every (w,g) on this line", 12, "#1d4ed8");
  txt(176, 320, "has the SAME matched loss", 12, "#1d4ed8");
  // double-headed trade arrow along the band
  const a1 = [X(40, 4), Y(80)], a2 = [X(40, 30), Y(54)];
  push(`<line x1="${a1[0].toFixed(1)}" y1="${a1[1].toFixed(1)}" x2="${a2[0].toFixed(1)}" y2="${a2[1].toFixed(1)}" stroke="#0f172a" stroke-width="1.8"/>`);
  push(`<path d="M ${a1[0].toFixed(1)} ${a1[1].toFixed(1)} l 7 -3 m -7 3 l 3 7" stroke="#0f172a" stroke-width="1.8" fill="none"/>`);
  push(`<path d="M ${a2[0].toFixed(1)} ${a2[1].toFixed(1)} l -7 3 m 7 -3 l -3 -7" stroke="#0f172a" stroke-width="1.8" fill="none"/>`);
  txt(116, 414, "1:1 trade direction", 11.5, "#0f172a", ' font-weight="600"');
  // true point + students
  dot(X(40, 10), Y(70), 6.5, "#f59e0b", "#b45309");
  txt(196, 236, "(w*, g*)  teacher", 13, "#b45309", ' font-weight="700"');
  [[2, 78], [6, 74], [14, 66], [18, 62], [26, 54]].forEach(([w, g]) => dot(X(40, w), Y(g), 5, "#475569"));
  txt(210, 486, "students: same loss, wrong (w,g)", 12.5, "#0f172a", ' font-weight="600"');

  // ---- Right: M >= 2 ----
  panelRect(480, "M >= 2  -  paired latents", "two latent draws per state estimate w and g separately");
  axes(480);
  push(`<ellipse cx="${X(480, 10).toFixed(1)}" cy="${Y(70).toFixed(1)}" rx="24" ry="22" fill="#dcfce7" opacity="0.85" stroke="#16a34a" stroke-width="1.6"/>`);
  txt(616, 228, "sublevel set collapses", 12, "#15803d", ' font-weight="600"');
  txt(616, 244, "to a small ball around the", 12, "#15803d");
  txt(616, 260, "true point  (rate ~ sqrt eps)", 12, "#15803d");
  push(`<line x1="${X(480, 1).toFixed(1)}" y1="${Y(79).toFixed(1)}" x2="${X(480, 19).toFixed(1)}" y2="${Y(61).toFixed(1)}" stroke="#cbd5e1" stroke-width="1.6" stroke-dasharray="3 5"/>`);
  txt(646, 456, "M=1 degeneracy removed", 11.5, "#64748b", ' font-style="italic"');
  dot(X(480, 10), Y(70), 6.5, "#f59e0b", "#b45309");
  txt(660, 236, "(w*, g*)", 13, "#b45309", ' font-weight="700"');
  [[8.8, 70.8], [10.8, 69.2], [9.4, 69.6], [11.4, 70.6], [9.2, 71.4]].forEach(([w, g]) => dot(X(480, w), Y(g), 4, "#16a34a"));
  txt(560, 520, "students cluster on the true point", 12.5, "#0f172a", ' font-weight="600"');

  // Bottom measured-axis annotation
  push(`<rect x="40" y="596" width="860" height="52" rx="8" fill="#f1f5f9"/>`);
  txt(56, 616, "Measured axes (DelayedBimodal policy runs): teacher g* ~ 73 vs teacher w* ~ 0.007  -  the g >> w hole that EMA reweighting converts into a U collapse.", 13, "#0f172a", ' font-weight="700"');
  txt(56, 636, "Equal-weight M>=2 recovers both components; the EMA-reweighted arm (identified_hybrid) starves g and U falls to ~0 (DMC 30/30 collapse control).  Schematic: axes not to scale.", 12.5, "#334155");
  push(`</svg>`);
  return P.join("\n");
}

// ---------------------------------------------------------------------------
// SVG V3 - cross-environment ordering (DelayedBimodal N=10 vs DMC N=30).
// ---------------------------------------------------------------------------
function svgV3(dmcArms, dbArms) {
  const W = 1080, H = 600;
  const P = [];
  const push = (s) => P.push(s);
  const byKey = (arr) => Object.fromEntries(arr.map((a) => [a.key, a]));
  const dmc = byKey(dmcArms), db = byKey(dbArms);
  const order = ["lagged_identified_eq", "lagged_hybrid", "ordinary", "identified_eq", "hybrid", "identified_hybrid"];
  const rowH = 64, y0 = 130;
  const x0 = 250, x1 = 930;          // value axis span
  const vmin = -0.15, vmax = 1.0;
  const X = (v) => x0 + ((v - vmin) / (vmax - vmin)) * (x1 - x0);
  const zeroX = X(0);
  const bar = (y, val, wpx, color, op) => {
    if (val >= 0) push(`<rect x="${zeroX.toFixed(1)}" y="${y}" width="${wpx.toFixed(1)}" height="16" rx="2" fill="${color}" opacity="${op}"/>`);
    else push(`<rect x="${(zeroX + wpx).toFixed(1)}" y="${y}" width="${(-wpx).toFixed(1)}" height="16" rx="2" fill="${color}" opacity="${op}"/>`);
  };
  push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Segoe UI, Arial, sans-serif">`);
  push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`);
  push(`<text x="36" y="38" font-size="24" font-weight="700" fill="#0f172a">The transfer-ordering flip: same six arms, two environments</text>`);
  push(`<text x="36" y="64" font-size="14.5" fill="#475569">Mean u_rank_corr (rank correlation of teacher U vs student U). DelayedBimodal N=10 (1,800 steps) vs DMC hopper-hop N=30 (15,000 steps). u-rank is the only</text>`);
  push(`<text x="36" y="84" font-size="14.5" fill="#475569">scale-free endpoint here - w_rmse and return are NOT comparable across environments (different units). Arms are ordered by the DelayedBimodal result.</text>`);

  // threshold line
  const barX = X(0.70);
  push(`<line x1="${barX.toFixed(1)}" y1="${y0 - 18}" x2="${barX.toFixed(1)}" y2="${y0 + rowH * 6 - 6}" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6 4"/>`);
  push(`<text x="${barX + 6}" y="${y0 - 24}" font-size="12.5" font-weight="700" fill="#64748b">registered bar 0.70</text>`);
  // zero line
  push(`<line x1="${zeroX.toFixed(1)}" y1="${y0 - 18}" x2="${zeroX.toFixed(1)}" y2="${y0 + rowH * 6 - 6}" stroke="#cbd5e1" stroke-width="1"/>`);
  push(`<text x="${zeroX - 4}" y="${y0 + rowH * 6 + 18}" font-size="12" fill="#94a3b8" text-anchor="end">0</text>`);
  push(`<text x="${x1}" y="${y0 + rowH * 6 + 18}" font-size="12" fill="#94a3b8" text-anchor="end">1.0</text>`);

  order.forEach((key, i) => {
    const y = y0 + i * rowH;
    const a = ARM_BY_KEY[key];
    const dv = db[key].u_mean, mv = dmc[key].u_mean;
    const label = a.display === "identified_hybrid" ? "EMA (identified_hybrid)" : a.display;
    push(`<text x="238" y="${y + 22}" font-size="14.5" font-weight="600" fill="#0f172a" text-anchor="end">${label}</text>`);
    // DB light bar
    const dbW = (dv / (vmax - vmin)) * (x1 - x0);
    bar(y - 6, dv, dbW, a.color, 0.35);
    // DMC solid bar
    const dmcW = (mv / (vmax - vmin)) * (x1 - x0);
    bar(y + 12, mv, dmcW, a.color, 0.95);
    push(`<text x="${(X(dv) + 6).toFixed(1)}" y="${y + 4}" font-size="12" fill="#64748b">${dv.toFixed(3)}</text>`);
    push(`<text x="${(X(mv) + 6).toFixed(1)}" y="${y + 23}" font-size="12.5" font-weight="700" fill="#0f172a">${mv.toFixed(3)}</text>`);
    // seed counts
    push(`<text x="${x1 + 6}" y="${y + 4}" font-size="11" fill="#94a3b8">DB ${db[key].n_ge_070}/10 >= 0.70</text>`);
    push(`<text x="${x1 + 6}" y="${y + 23}" font-size="11" fill="#64748b">DMC ${dmc[key].n_ge_070}/30 >= 0.70</text>`);
    // reversal / headline chips
    if (key === "lagged_identified_eq") {
      push(`<rect x="36" y="${y - 16}" width="176" height="22" rx="11" fill="#fee2e2"/>`);
      push(`<text x="124" y="${y - 1}" font-size="12.5" font-weight="700" fill="#b91c1c" text-anchor="middle">REVERSAL: 0.948 -> 0.638</text>`);
    }
    if (key === "ordinary") {
      push(`<rect x="36" y="${y - 16}" width="166" height="22" rx="11" fill="#dbeafe"/>`);
      push(`<text x="119" y="${y - 1}" font-size="12.5" font-weight="700" fill="#1d4ed8" text-anchor="middle">tops the DMC table (30/30)</text>`);
    }
    if (key === "identified_eq") {
      push(`<rect x="36" y="${y - 16}" width="150" height="22" rx="11" fill="#dcfce7"/>`);
      push(`<text x="111" y="${y - 1}" font-size="12.5" font-weight="700" fill="#15803d" text-anchor="middle">mechanism transfer (30/30)</text>`);
    }
  });

  // legend
  push(`<rect x="36" y="${y0 + rowH * 6 + 28}" width="40" height="12" rx="2" fill="#64748b" opacity="0.35"/>`);
  push(`<text x="84" y="${y0 + rowH * 6 + 39}" font-size="12.5" fill="#334155">DelayedBimodal (N=10, easier regime)</text>`);
  push(`<rect x="356" y="${y0 + rowH * 6 + 28}" width="40" height="12" rx="2" fill="#64748b"/>`);
  push(`<text x="404" y="${y0 + rowH * 6 + 39}" font-size="12.5" fill="#334155">DMC hopper-hop (N=30, harder regime)</text>`);
  push(`<text x="610" y="${y0 + rowH * 6 + 39}" font-size="12.5" fill="#334155">EMA mean is negative on DMC (-0.076), drawn left of zero</text>`);
  push(`</svg>`);
  return P.join("\n");
}

// ---------------------------------------------------------------------------
// Build entry point.
// ---------------------------------------------------------------------------
function main() {
  fs.mkdirSync(OUT("data"), { recursive: true });
  fs.mkdirSync(OUT("svg"), { recursive: true });
  fs.mkdirSync(path.join(ROOT, "visuals", "templates"), { recursive: true });

  const dmc30 = dmc30Payload();
  const ctrl = ctrlPayload();
  const db10 = dbPayload();
  const curves = curvesPayload();

  // --- verification (fail loudly on drift from the adjudicated tables) ----
  const bad = [...verifyDmc(dmc30), ...verifyDb(db10)];
  const ctrlRef = 0.9746;
  if (Math.abs(ctrl.arms[0].u_mean - ctrlRef) > 0.003) bad.push("ctrl u_mean " + ctrl.arms[0].u_mean.toFixed(4) + " vs " + ctrlRef);
  console.log("== verification ==");
  console.log("DMC u_rank means:", dmc30.arms.map((a) => a.key + "=" + a.u_mean.toFixed(3)).join("  "));
  console.log("DMC n>=0.70    :", dmc30.arms.map((a) => a.n_ge_070).join("/"));
  console.log("DB  u_rank means:", db10.arms.map((a) => a.key + "=" + a.u_mean.toFixed(3)).join("  "));
  console.log("ctrl u_mean    :", ctrl.arms[0].u_mean.toFixed(4), "(gate off; ordinary 0.9573)");
  if (bad.length) {
    console.error("VERIFICATION FAILED:\n - " + bad.join("\n - "));
    process.exit(1);
  }
  console.log("verification: OK (means match the adjudication docs within tolerance)");

  // --- write data payloads ---
  const files = [
    ["data/dmc_30seed_final.json", dmc30],
    ["data/dmc_ctrl_30seed_final.json", ctrl],
    ["data/db_10seed_final.json", db10],
    ["data/curves.json", curves],
  ];
  for (const [rel, obj] of files) fs.writeFileSync(OUT(rel), JSON.stringify(obj, null, 1) + "\n");
  console.log("wrote", files.map((f) => "visuals/" + f[0]).join(", "));

  // --- write static SVGs ---
  const v1 = svgV1();
  const v3 = svgV3(dmc30.arms, db10.arms);
  fs.writeFileSync(OUT("svg/v1-identifiability-skip.svg"), v1);
  fs.writeFileSync(OUT("svg/v3-cross-env-ordering.svg"), v3);
  console.log("wrote visuals/svg/v1-identifiability-skip.svg, visuals/svg/v3-cross-env-ordering.svg");

  // --- assemble the self-contained explorer page ---
  const tplPath = path.join(ROOT, "visuals", "templates", "v4-explorer.template.html");
  let tpl = fs.readFileSync(tplPath, "utf8");
  const payloadJson = JSON.stringify({ dmc30, dmcCtrl: ctrl, db10, curves }, null, 1);
  if (!tpl.includes("/*__PAYLOAD_JSON__*/")) throw new Error("template marker __PAYLOAD_JSON__ not found");
  if (!tpl.includes("__V1_SVG__") || !tpl.includes("__V3_SVG__")) throw new Error("template svg markers not found");
  tpl = tpl.split("/*__PAYLOAD_JSON__*/").join(payloadJson);
  tpl = tpl.split("__V1_SVG__").join(v1);
  tpl = tpl.split("__V3_SVG__").join(v3);
  fs.writeFileSync(OUT("v4-explorer.html"), tpl);
  console.log("wrote visuals/v4-explorer.html (" + tpl.length + " bytes)");
}

main();
