#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 13 — **local information only** vs supervised adaptive DAG (external task check).

**Variant 1** (``TTAdaptiveDAGLocalUnsupervised``): COMPARE uses a rolling sample of minterms
and a purely local statistic — absolute correlation between a gate’s output and the first
downstream consumer in the same subnet, or a balance score on the last gate. The supervised
target never enters acceptance.

**Variant 0** (``Exp12AdaptiveDAG``): same graph family and mutation rules, global score
acceptance (experiment 12).

Both runs log **external** ``score_current_gates()`` vs time (ticks) without using that
signal in variant 1’s learning rule. This separates *local self-organization* from *task
performance*.

Run from repo root:
  python src/experiments/experiment_13.py

Output: src/results/experiment_13_local_vs_supervised.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_13_worker as e13w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_13_local_vs_supervised.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

VAR_SHORT = ["supervised DAG", "local-only DAG"]


def collect_scenario(
    n_in: int,
    n_out: int,
    kind: str,
    ts_fix: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    k: int,
    log_stride: int,
) -> Dict[str, Any]:
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, int]] = []
    for vi in range(2):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((vi, n_in, n_out, kind, s, max_ticks, ts_fix, k, log_stride))

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        rows = pool.map(e13w.run_one, jobs)

    sr = np.zeros(2)
    fin = np.zeros(2)
    ok_ct = [0, 0]
    n_per = [0, 0]
    for r in rows:
        vi = int(r["variant"])
        n_per[vi] += 1
        if r["ok_task"]:
            ok_ct[vi] += 1
        fin[vi] += r["final_external_score"] / max(1, r["max_score"])
    for vi in range(2):
        sr[vi] = ok_ct[vi] / max(1, n_per[vi])
        fin[vi] = fin[vi] / max(1, n_per[vi])
    fin_mean = [float(fin[0]), float(fin[1])]

    curves0 = [r for r in rows if r["variant"] == 0]
    curves1 = [r for r in rows if r["variant"] == 1]

    return {
        "sr": sr,
        "fin_mean": fin_mean,
        "rows": rows,
        "curves0": curves0,
        "curves1": curves1,
    }


def _default_workers() -> int:
    for key in ("EXPERIMENT_13_WORKERS", "EXPERIMENT_12_WORKERS", "EXPERIMENT_72_WORKERS"):
        env = os.environ.get(key, "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
    return max(1, os.cpu_count() or 1)


def _mean_curve(
    curve_list: List[dict], max_ticks: int, log_stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    stride = max(1, log_stride)
    ts_template = list(range(0, max_ticks + 1, stride))
    if not ts_template or ts_template[-1] != max_ticks:
        ts_template.append(max_ticks)
    t_arr = np.array(ts_template, dtype=float)
    M = np.full((len(curve_list), len(ts_template)), np.nan, dtype=float)
    for i, r in enumerate(curve_list):
        tx = np.array(r["curve_ticks"], dtype=int)
        sc = np.array(r["curve_score"], dtype=float)
        if tx.size == 0:
            continue
        for j, t in enumerate(ts_template):
            idx = int(np.searchsorted(tx, t, side="right") - 1)
            if idx >= 0:
                M[i, j] = sc[idx]
    return t_arr, np.nanmean(M, axis=0)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Experiment 13: local-only vs supervised DAG")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--log-stride", type=int, default=400, help="external score log interval (ticks)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    workers = max(1, args.workers if args.workers is not None else _default_workers())
    if args.k < 0:
        return 1

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11, 4.2 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    fig.suptitle(
        rf"Experiment 13 — supervised vs local-only adaptive DAG ($K={args.k}$); external score only for local",
        fontsize=11,
    )

    for si, (slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds) in enumerate(SCENARIOS):
        print(f"{slab}  K={args.k}  log_stride={args.log_stride}  ({n_seeds} seeds × 2, {workers} workers)...")
        d = collect_scenario(
            n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, args.k, args.log_stride
        )
        ax0, ax1 = axes[si, 0], axes[si, 1]
        xb = np.arange(2)
        w = 0.4
        ax0.bar(xb, d["sr"], w, color=["steelblue", "darkorange"], edgecolor="black", lw=0.4)
        ax0.set_xticks(xb)
        ax0.set_xticklabels(VAR_SHORT, fontsize=8, rotation=12, ha="right")
        ax0.set_ylabel("task success (external)")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(f"{slab} — success")
        ax0.grid(True, axis="y", alpha=0.3)

        ax1.bar(xb, d["fin_mean"], w, color=["steelblue", "darkorange"], edgecolor="black", lw=0.4)
        ax1.set_xticks(xb)
        ax1.set_xticklabels(VAR_SHORT, fontsize=8, rotation=12, ha="right")
        ax1.set_ylabel("mean final external score / max")
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"{slab} — mean final normalized score")
        ax1.grid(True, axis="y", alpha=0.3)

        print(
            f"  {slab}: success {VAR_SHORT[0]}={d['sr'][0]:.2f}  {VAR_SHORT[1]}={d['sr'][1]:.2f}  "
            f"mean final norm {d['fin_mean'][0]:.3f} vs {d['fin_mean'][1]:.3f}"
        )

        fig2, axc = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        t0, m0 = _mean_curve(d["curves0"], max_ticks, args.log_stride)
        t1, m1 = _mean_curve(d["curves1"], max_ticks, args.log_stride)
        axc.plot(t0, m0, "-", color="steelblue", label=VAR_SHORT[0], lw=1.5)
        axc.plot(t1, m1, "-", color="darkorange", label=VAR_SHORT[1], lw=1.5)
        axc.set_xlabel("training tick")
        axc.set_ylabel("external score (mean over seeds)")
        axc.set_title(f"{slab} — external task score vs time")
        axc.legend(fontsize=8)
        axc.grid(True, alpha=0.3)
        ymax = float(n_out * (1 << n_in))
        axc.set_ylim(0, ymax * 1.02 + 0.01)
        axc.axhline(ymax, color="gray", ls="--", lw=0.8, alpha=0.6)
        out_name = RESULTS / f"experiment_13_curve_{si}.png"
        RESULTS.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_name, dpi=150)
        plt.close(fig2)
        print(f"  wrote {out_name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
