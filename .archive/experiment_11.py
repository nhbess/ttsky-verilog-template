#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 11 — **local ε acceptance** vs global score baseline, plus dynamics / proxies.

**Variants**

0. **Global baseline:** credit-weighted gate pick, global ``new_score`` vs ``old_score`` COMPARE
   (experiment 9-style).
1. **Local ε only:** uniform gate pick; accept if output-line wrong-fraction ``ε_m`` falls
   (same plateau tie rule when ``ε_m`` unchanged).
2. **Hybrid:** credit-weighted pick; local ε COMPARE.

**Metrics (experiment 5 style, extended):** per COMPARE trial we log global
``Δ = new_score - old_score``, local ``Δε`` (nan for variant 0), accept bit, gate index.
From these: autocorrelation of ``Δ`` / ``Δε``, cascade run lengths of consecutive accepts,
distributions of accepted updates per gate, and a simple **information proxy** ``I_i`` at
the final snapshot (fraction of minterms where some alternate LUT at gate ``i`` flips
correctness).

Scenarios match experiments 8–10 (parity 3×1, random 4×2).

Run from repo root:
  python src/experiments/experiment_11.py

Outputs:
  src/results/experiment_11_main.png
  src/results/experiment_11_dynamics.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_11_worker as e11w  # noqa: E402
import tt_chain_learner_exp72 as e72  # noqa: E402
import tt_chain_learner_spec as chain  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_MAIN = RESULTS / "experiment_11_main.png"
OUT_DYN = RESULTS / "experiment_11_dynamics.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

VAR_LABELS = ["global + credit $u$", r"local $\varepsilon$, uniform $u$", r"local $\varepsilon$ + credit $u$"]
VAR_SHORT = ["global", "local uni", "local cr"]


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(max_lag + 1, np.nan)
    if n < 4:
        out[0] = 1.0
        return out
    x = x - x.mean()
    var = float(np.dot(x, x)) / n
    if var < 1e-12:
        out[0] = 1.0
        return out
    for lag in range(0, min(max_lag + 1, n)):
        out[lag] = float(np.dot(x[: n - lag], x[lag:]) / (n * var))
    return out


def _run_lengths_consecutive_true(mask: np.ndarray) -> List[int]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    out: List[int] = []
    i, n = 0, len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        out.append(j - i)
        i = j
    return out


def probe_from_result_row(d: Dict[str, Any]) -> e72.TTChainLearnerExp72:
    tgt = d["target"]
    p = e72.TTChainLearnerExp72(
        n_in=d["n_in"],
        n_out=d["n_out"],
        target=tgt,
        plateau_escape=True,
        plateau_mask=3,
        extra_gates=d["extra_gates"],
        topology=d["topology"],
    )
    p.gate = list(d["final_gates"])
    return p


def gate_c_bad_counts(probe: e72.TTChainLearnerExp72) -> Tuple[List[int], List[int]]:
    n_gates = len(probe.gate)
    c_good = [0] * n_gates
    c_bad = [0] * n_gates
    gpo = probe.gpo
    for idx in range(probe.nrows):
        xs = chain.row_bits(idx, probe.n_in)
        ys = probe.forward_all(xs)
        for m in range(probe.n_out):
            ok = ys[m] == probe.target[m][idx]
            base = m * gpo
            for off in range(gpo):
                gi = base + off
                if ok:
                    c_good[gi] += 1
                else:
                    c_bad[gi] += 1
    return c_good, c_bad


def information_proxy_I(probe: e72.TTChainLearnerExp72, gi: int) -> float:
    gpo = probe.gpo
    m = gi // gpo
    base_g = probe.gate[gi]
    n = probe.nrows
    flipped = 0
    for idx in range(n):
        xs = chain.row_bits(idx, probe.n_in)
        y0 = probe.forward_all(xs)[m]
        ok0 = y0 == probe.target[m][idx]
        row_sens = False
        for alt in range(16):
            if alt == base_g:
                continue
            probe._trial_overrides = {gi: alt}
            y1 = probe._forward_trial_all(xs)[m]
            probe._trial_overrides = None
            ok1 = y1 == probe.target[m][idx]
            if ok1 != ok0:
                row_sens = True
                break
        if row_sens:
            flipped += 1
    return flipped / n


def run_scenario_jobs(
    lab: str,
    n_in: int,
    n_out: int,
    kind: str,
    ts_fix: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    k: int,
    topology: str,
    credit_mode: str,
) -> List[Dict[str, Any]]:
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str, str]] = []
    for vi in range(3):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((vi, n_in, n_out, kind, s, max_ticks, ts_fix, k, topology, credit_mode))
    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        return pool.map(e11w.run_one, jobs)


def _default_workers() -> int:
    for key in ("EXPERIMENT_11_WORKERS", "EXPERIMENT_9_WORKERS", "EXPERIMENT_72_WORKERS", "EXPERIMENT_7_WORKERS"):
        env = os.environ.get(key, "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
    return max(1, os.cpu_count() or 1)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Experiment 11: local epsilon vs global acceptance")
    ap.add_argument("--topology", choices=("A", "B"), default="A")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--credit-mode", choices=("bad", "delta"), default="bad")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dyn-max-lag", type=int, default=40, help="autocorr max lag for dynamics fig")
    args = ap.parse_args()
    workers = max(1, args.workers if args.workers is not None else _default_workers())
    if args.k < 0:
        print("invalid --k", file=sys.stderr)
        return 1

    n_var = 3
    all_sr: List[np.ndarray] = []
    all_mt: List[np.ndarray] = []
    all_rows_by_scen: List[List[Dict[str, Any]]] = []

    for slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        print(
            f"{slab}  topology={args.topology}  K={args.k}  "
            f"({n_seeds} seeds × {n_var} variants, {workers} workers)..."
        )
        rows = run_scenario_jobs(
            slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, args.k, args.topology, args.credit_mode
        )
        all_rows_by_scen.append(rows)
        sr = np.zeros(n_var)
        mt = np.full(n_var, np.nan)
        ticks_by: List[List[int]] = [[] for _ in range(n_var)]
        ok_ct = [0] * n_var
        for r in rows:
            vi = r["variant"]
            if r["ok"]:
                ok_ct[vi] += 1
                ticks_by[vi].append(int(r["ticks"]))
        for vi in range(n_var):
            sr[vi] = ok_ct[vi] / n_seeds
            if ticks_by[vi]:
                mt[vi] = float(mean(ticks_by[vi]))
        all_sr.append(sr)
        all_mt.append(mt)

    x = np.arange(n_var)
    width = 0.55
    colors = ["steelblue", "darkorange", "seagreen"]
    fig_m, axes_m = plt.subplots(
        len(SCENARIOS), 2, figsize=(10, 3.6 * len(SCENARIOS)), constrained_layout=True
    )
    if len(SCENARIOS) == 1:
        axes_m = np.array([axes_m])
    fig_m.suptitle(
        rf"Experiment 11 — global vs local $\varepsilon$ accept ($K={args.k}$, topology {args.topology})",
        fontsize=11,
    )
    for si, row in enumerate(axes_m):
        ax0, ax1 = row[0], row[1]
        sr, mt = all_sr[si], all_mt[si]
        slab = SCENARIOS[si][0]
        ax0.bar(x, sr, width, color=colors, edgecolor="black", linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(VAR_SHORT, fontsize=8, rotation=15, ha="right")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(slab)
        ax0.grid(True, axis="y", alpha=0.3)
        for j in range(n_var):
            if np.isfinite(mt[j]):
                ax1.bar(j, mt[j], width, color=colors[j], edgecolor="black", linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(VAR_SHORT, fontsize=8, rotation=15, ha="right")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.grid(True, axis="y", alpha=0.3)
        if np.any(np.isfinite(mt)):
            ymax = float(np.nanmax(mt))
            ax1.set_ylim(0, ymax * 1.12 + 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig_m.savefig(OUT_MAIN, dpi=150)
    plt.close(fig_m)
    print(f"Wrote {OUT_MAIN}")

    parity_rows = all_rows_by_scen[0]
    rows_by_v = {0: [], 1: [], 2: []}
    for r in parity_rows:
        rows_by_v[int(r["variant"])].append(r)

    max_lag = max(1, args.dyn_max_lag)
    lags = np.arange(max_lag + 1)
    fig_d, axes_d = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig_d.suptitle(
        "Experiment 11 — dynamics (parity 3×1, all seeds pooled per variant)", fontsize=11
    )

    for vi in range(3):
        dlist = [np.array(r["trial_global_delta"], dtype=float) for r in rows_by_v[vi]]
        if dlist and any(len(a) > 0 for a in dlist):
            pooled = np.concatenate([a for a in dlist if len(a)])
            axes_d[0, 0].hist(
                pooled,
                bins=min(80, max(20, int(np.sqrt(len(pooled))))),
                alpha=0.45,
                label=VAR_SHORT[vi],
                density=True,
            )
    axes_d[0, 0].set_title(r"Pooled global trial $\Delta = S_{\mathrm{new}}-S_{\mathrm{old}}$")
    axes_d[0, 0].set_xlabel(r"$\Delta$")
    axes_d[0, 0].legend(fontsize=8)
    axes_d[0, 0].grid(True, alpha=0.3)

    for vi in (1, 2):
        dlist = [
            np.array([x for x in r["trial_eps_delta"] if np.isfinite(x)], dtype=float)
            for r in rows_by_v[vi]
        ]
        parts = [a for a in dlist if len(a)]
        if parts:
            pooled = np.concatenate(parts)
            axes_d[0, 1].hist(
                pooled,
                bins=min(80, max(20, int(np.sqrt(len(pooled))))),
                alpha=0.45,
                label=VAR_SHORT[vi],
                density=True,
            )
    axes_d[0, 1].set_title(r"Pooled local $\Delta\varepsilon$ (local variants only)")
    axes_d[0, 1].set_xlabel(r"$\Delta\varepsilon$")
    axes_d[0, 1].legend(fontsize=8)
    axes_d[0, 1].grid(True, alpha=0.3)

    for vi in range(3):
        acs = []
        for r in rows_by_v[vi]:
            d = np.array(r["trial_global_delta"], dtype=float)
            if len(d) >= 8:
                acs.append(_autocorr(d, max_lag))
        if acs:
            stacked = np.vstack(acs)
            m = np.nanmean(stacked, axis=0)
            axes_d[1, 0].plot(lags, m, "-o", ms=3, label=VAR_SHORT[vi])
    axes_d[1, 0].set_title(r"Mean autocorr of global $\Delta$ (per seed)")
    axes_d[1, 0].set_xlabel("lag")
    axes_d[1, 0].legend(fontsize=8)
    axes_d[1, 0].grid(True, alpha=0.3)
    axes_d[1, 0].set_ylim(-0.2, 1.05)

    cascade_lens_all: List[Tuple[str, List[int]]] = []
    for vi in range(3):
        lens: List[int] = []
        for r in rows_by_v[vi]:
            acc = np.array(r["trial_accepted"], dtype=bool)
            lens.extend(_run_lengths_consecutive_true(acc))
        if lens:
            cascade_lens_all.append((VAR_SHORT[vi], lens))
    if cascade_lens_all:
        ax = axes_d[1, 1]
        for lab_c, lens in cascade_lens_all:
            ax.hist(
                lens,
                bins=min(40, max(5, int(np.sqrt(len(lens))))),
                alpha=0.4,
                label=lab_c,
                density=True,
            )
        ax.set_title("Cascade lengths (consecutive accepts)")
        ax.set_xlabel("length")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig_d.savefig(OUT_DYN, dpi=150)
    plt.close(fig_d)
    print(f"Wrote {OUT_DYN}")

    I_vals: List[float] = []
    acc_vals: List[float] = []
    cb_vals: List[float] = []
    var_tags: List[int] = []
    for si, rows in enumerate(all_rows_by_scen):
        for r in rows:
            probe = probe_from_result_row(r)
            vi = int(r["variant"])
            _, c_bad = gate_c_bad_counts(probe)
            n_g = r["n_gates"]
            acc_dense = np.zeros(n_g, dtype=float)
            for g, c in r["accepts_per_gate"].items():
                acc_dense[int(g)] = float(c)
            for gi in range(n_g):
                I_vals.append(information_proxy_I(probe, gi))
                acc_vals.append(acc_dense[gi])
                cb_vals.append(float(c_bad[gi]))
                var_tags.append(vi)
    if I_vals:
        print(
            f"Information proxy I_i: grand mean={float(np.mean(I_vals)):.4f}  "
            f"(pooled over gates×seeds×scenarios, n={len(I_vals)})"
        )
        for vi in range(3):
            m = np.array(var_tags) == vi
            if np.any(m):
                ri = np.corrcoef(np.array(I_vals)[m], np.array(acc_vals)[m])
                r_ic = ri[0, 1] if ri.shape == (2, 2) else float("nan")
                ri2 = np.corrcoef(np.array(I_vals)[m], np.array(cb_vals)[m])
                r_ib = ri2[0, 1] if ri2.shape == (2, 2) else float("nan")
                print(
                    f"  variant {vi} ({VAR_SHORT[vi]}): corr(I_i, accepts)={r_ic:.3f}  corr(I_i, c_bad)={r_ib:.3f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
