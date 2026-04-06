#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 12 — **adaptive connectivity** vs fixed topology A, plus **σ_eff** and activity.

**Architecture (variant 1):** ``TTAdaptiveDAGLearner`` — per output subnet, ``G`` gates in
layer order; each gate’s two inputs are indices into the pool
``{x_0,…,x_{n-1}, wire_0,…,wire_{i-1}}`` (inputs plus earlier subnet wires). LUTs mutate as in the chain
learner; trials can **rewire** one input. Low **influence** (few rows where an XOR LUT
tweak changes the subnet output) raises the probability of a rewire trial.

**Baseline (variant 0):** ``Exp12CreditBaseline`` — same ``G = gpo_topology_A(n_in, K)`` as
experiment 8, topology A, credit-weighted gate pick.

**Metrics**

* success rate, mean ticks (successes)
* mean **σ_eff**: after each accepted COMPARE, distinct gates in the next 20 trials,
  normalized by window length
* **distinct** packed truth-table keys at end of run (across seeds)
* **gate activity:** histogram of per-gate accept counts (pooled over seeds, last scenario)

Run from repo root:
  python src/experiments/experiment_12.py

Output: src/results/experiment_12_connectivity.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_12_worker as e12w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_12_connectivity.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

VAR_SHORT = ["Exp72 A", "DAG"]


def collect_scenario(
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    k: int,
    credit_mode: str,
) -> Dict[str, Any]:
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str, str]] = []
    for vi in range(2):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((vi, n_in, n_out, kind, s, max_ticks, table_seed_fixed, k, credit_mode, "A"))

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        rows: List[Dict[str, Any]] = pool.map(e12w.run_one, jobs)

    sr = np.zeros(2)
    mt = np.full(2, np.nan)
    sig_lists: List[List[float]] = [[], []]
    ticks_by: List[List[int]] = [[], []]
    ok_ct = [0, 0]
    keys0: Set[tuple] = set()
    keys1: Set[tuple] = set()
    act0: List[int] = []
    act1: List[int] = []

    for r in rows:
        vi = int(r["variant"])
        if r["ok"]:
            ok_ct[vi] += 1
            ticks_by[vi].append(int(r["ticks"]))
        se = float(r["sigma_eff"])
        if np.isfinite(se):
            sig_lists[vi].append(se)
        if vi == 0:
            keys0.add(r["truth_key"])
        else:
            keys1.add(r["truth_key"])
        for _g, c in r["accepts_per_gate"].items():
            if vi == 0:
                act0.append(int(c))
            else:
                act1.append(int(c))

    for vi in range(2):
        sr[vi] = ok_ct[vi] / n_seeds
        if ticks_by[vi]:
            mt[vi] = float(mean(ticks_by[vi]))

    sig_mean = [float(mean(sig_lists[vi])) if sig_lists[vi] else float("nan") for vi in range(2)]

    return {
        "sr": sr,
        "mt": mt,
        "sig_mean": sig_mean,
        "keys0": keys0,
        "keys1": keys1,
        "act0": act0,
        "act1": act1,
    }


def _default_workers() -> int:
    for key in ("EXPERIMENT_12_WORKERS", "EXPERIMENT_11_WORKERS", "EXPERIMENT_72_WORKERS"):
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

    ap = argparse.ArgumentParser(description="Experiment 12: adaptive DAG vs Exp72 A")
    ap.add_argument("--k", type=int, default=4, help="extra gates K (sets G for both arms)")
    ap.add_argument("--credit-mode", choices=("bad", "delta"), default="bad")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    workers = max(1, args.workers if args.workers is not None else _default_workers())
    if args.k < 0:
        print("invalid --k", file=sys.stderr)
        return 1

    n_var = 2
    x = np.arange(n_var)
    width = 0.42
    colors = ["steelblue", "seagreen"]

    per_scen: List[Dict[str, Any]] = []
    for slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        print(f"{slab}  K={args.k}  ({n_seeds} seeds × 2 variants, {workers} workers)...")
        d = collect_scenario(
            n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, args.k, args.credit_mode
        )
        d["label"] = slab
        per_scen.append(d)
        print(
            f"  distinct truth keys — Exp72 {len(d['keys0'])}, DAG {len(d['keys1'])}  "
            f"mean σ_eff — {d['sig_mean'][0]:.3f} vs {d['sig_mean'][1]:.3f}"
        )

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05])
    fig.suptitle(
        rf"Experiment 12 — fixed topology A vs adaptive DAG ($K={args.k}$, matched $G$ per output)",
        fontsize=11,
    )

    for si, d in enumerate(per_scen):
        sr, mt = d["sr"], d["mt"]
        slab = d["label"]
        ax0 = fig.add_subplot(gs[si, 0])
        ax0.bar(x, sr, width, color=colors, edgecolor="black", linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(VAR_SHORT, fontsize=9)
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(f"{slab} — success")
        ax0.grid(True, axis="y", alpha=0.3)

        ax1 = fig.add_subplot(gs[si, 1])
        for j in range(n_var):
            if np.isfinite(mt[j]):
                ax1.bar(j, mt[j], width, color=colors[j], edgecolor="black", linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(VAR_SHORT, fontsize=9)
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.grid(True, axis="y", alpha=0.3)
        if np.any(np.isfinite(mt)):
            ymax = float(np.nanmax(mt))
            ax1.set_ylim(0, ymax * 1.12 + 1)

    ax_sig = fig.add_subplot(gs[2, 0])
    x2 = np.arange(len(SCENARIOS))
    w2 = 0.35
    for vi in range(2):
        offset = (vi - 0.5) * w2
        ys = [per_scen[s]["sig_mean"][vi] for s in range(len(SCENARIOS))]
        ax_sig.bar(
            x2 + offset,
            ys,
            w2,
            label=VAR_SHORT[vi],
            color=colors[vi],
            edgecolor="black",
            linewidth=0.4,
        )
    ax_sig.set_xticks(x2)
    ax_sig.set_xticklabels([s[0] for s in SCENARIOS], fontsize=9)
    ax_sig.set_ylabel(r"mean $\sigma_{\mathrm{eff}}$")
    ax_sig.legend(fontsize=8)
    ax_sig.grid(True, axis="y", alpha=0.3)
    ax_sig.set_title(r"$\sigma_{\mathrm{eff}}$ (distinct gates / next-20-trials window)")

    ax_hist = fig.add_subplot(gs[2, 1])
    last = per_scen[-1]
    act0, act1 = last["act0"], last["act1"]
    if act0 or act1:
        nb = min(30, max(8, int(np.sqrt(max(len(act0), len(act1), 1)))))
        ax_hist.hist(
            [act0, act1],
            bins=nb,
            label=VAR_SHORT,
            alpha=0.55,
            density=True,
            color=colors,
        )
    ax_hist.set_xlabel("accepted updates per gate (pooled seeds)")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title(f"Gate activity — {last['label']}")
    ax_hist.grid(True, alpha=0.3)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
