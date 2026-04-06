#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 16 — clamped I/O relaxation with local propagated error.

Three variants:
  0 — supervised hillclimber (global external score), same as Exp12.
  1 — clamp (x, y*), measure E_local = sum of propagated child mismatches, accept if it drops;
      LUT-only (topology fixed at init).
  2 — same local objective with adaptive DAG rewiring + LUT.

External ``score_current_gates()`` is used only for evaluation and success curves.
Internal metrics: E_local, backward error depth, relaxation sweeps to settle.

Run from repo root:
  python src/experiments/experiment_16.py

Outputs:
  src/results/experiment_16_clamp_alignment.png
  src/results/experiment_16_curve_<scenario_idx>.png
  src/results/experiment_16_Elocal_<scenario_idx>.png
  src/results/experiment_16_depth_<scenario_idx>.png
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

import experiment_16_worker as e16w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_16_clamp_alignment.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

VAR_SHORT = ["supervised", "clamp (LUT)", "clamp + DAG"]


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
    for vi in range(3):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((vi, n_in, n_out, kind, s, max_ticks, ts_fix, k, log_stride))

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        rows: List[Dict[str, Any]] = pool.map(e16w.run_one, jobs)

    sr = np.zeros(3)
    fin = np.zeros(3)
    fin_E = np.zeros(3)
    fin_depth = np.zeros(3)
    ok_ct = [0, 0, 0]
    n_per = [0, 0, 0]
    for r in rows:
        vi = int(r["variant"])
        n_per[vi] += 1
        if r["ok_task"]:
            ok_ct[vi] += 1
        fin[vi] += r["final_external_score"] / max(1, r["max_score"])
        fin_E[vi] += float(r["final_E_local"])
        fin_depth[vi] += float(r["final_max_depth"])
    for vi in range(3):
        sr[vi] = ok_ct[vi] / max(1, n_per[vi])
        fin[vi] = fin[vi] / max(1, n_per[vi])
        fin_E[vi] = fin_E[vi] / max(1, n_per[vi])
        fin_depth[vi] = fin_depth[vi] / max(1, n_per[vi])
    fin_mean = [float(fin[0]), float(fin[1]), float(fin[2])]

    return {
        "sr": sr,
        "fin_mean": fin_mean,
        "fin_E_local_mean": [float(fin_E[0]), float(fin_E[1]), float(fin_E[2])],
        "fin_depth_mean": [float(fin_depth[0]), float(fin_depth[1]), float(fin_depth[2])],
        "rows": rows,
        "curves0": [r for r in rows if r["variant"] == 0],
        "curves1": [r for r in rows if r["variant"] == 1],
        "curves2": [r for r in rows if r["variant"] == 2],
    }


def _default_workers() -> int:
    for key in ("EXPERIMENT_16_WORKERS", "EXPERIMENT_14_WORKERS", "EXPERIMENT_12_WORKERS"):
        env = os.environ.get(key, "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
    return max(1, os.cpu_count() or 1)


def _mean_curve(
    curve_list: List[dict], max_ticks: int, log_stride: int, key: str
) -> Tuple[np.ndarray, np.ndarray]:
    stride = max(1, log_stride)
    ts_template = list(range(0, max_ticks + 1, stride))
    if not ts_template or ts_template[-1] != max_ticks:
        ts_template.append(max_ticks)
    t_arr = np.array(ts_template, dtype=float)
    M = np.full((len(curve_list), len(ts_template)), np.nan, dtype=float)
    for i, r in enumerate(curve_list):
        tx = np.array(r["curve_ticks"], dtype=int)
        sc = np.array(r[key], dtype=float)
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

    ap = argparse.ArgumentParser(description="Experiment 16: clamped relaxation + local error")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--log-stride", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    workers = max(1, args.workers if args.workers is not None else _default_workers())
    if args.k < 0:
        return 1

    n_var = 3
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(len(SCENARIOS), 3, figsize=(13, 4.0 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    fig.suptitle(
        rf"Experiment 16 — clamp + local $E$ ($K={args.k}$); external eval + final internal error",
        fontsize=10,
    )

    for si, (slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds) in enumerate(SCENARIOS):
        print(f"{slab}  K={args.k}  ({n_seeds} seeds × 3 variants, {workers} workers)...")
        d = collect_scenario(
            n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, args.k, args.log_stride
        )
        ax0, ax1, ax2 = axes[si, 0], axes[si, 1], axes[si, 2]
        xb = np.arange(n_var)
        w = 0.24
        ax0.bar(xb, d["sr"], w, color=colors, edgecolor="black", linewidth=0.35)
        ax0.set_xticks(xb)
        ax0.set_xticklabels(VAR_SHORT, fontsize=7, rotation=14, ha="right")
        ax0.set_ylabel("task success (external)")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(f"{slab} — success")
        ax0.grid(True, axis="y", alpha=0.3)

        ax1.bar(xb, d["fin_mean"], w, color=colors, edgecolor="black", linewidth=0.35)
        ax1.set_xticks(xb)
        ax1.set_xticklabels(VAR_SHORT, fontsize=7, rotation=14, ha="right")
        ax1.set_ylabel("mean final score / max")
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"{slab} — mean final normalized score")
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(xb, d["fin_E_local_mean"], w, color=colors, edgecolor="black", linewidth=0.35)
        ax2.set_xticks(xb)
        ax2.set_xticklabels(VAR_SHORT, fontsize=7, rotation=14, ha="right")
        ax2.set_ylabel(r"mean final $E_{\mathrm{local}}$ (snapshot)")
        ax2.set_title(f"{slab} — internal error (last measured)")
        ax2.grid(True, axis="y", alpha=0.3)

        print(
            f"  success: sup={d['sr'][0]:.2f}  clamp={d['sr'][1]:.2f}  clamp+DAG={d['sr'][2]:.2f}  "
            f"final norm: {d['fin_mean'][0]:.3f} / {d['fin_mean'][1]:.3f} / {d['fin_mean'][2]:.3f}"
        )

        ymax = float(n_out * (1 << n_in))
        fig2, axc = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
        for vi, c_lab, c in zip(range(3), VAR_SHORT, colors):
            cl = [r for r in d["rows"] if r["variant"] == vi]
            t_arr, m_arr = _mean_curve(cl, max_ticks, args.log_stride, "curve_score")
            axc.plot(t_arr, m_arr, "-", color=c, label=c_lab, lw=1.4)
        axc.set_xlabel("FSM tick")
        axc.set_ylabel("external score (mean over seeds)")
        axc.set_title(f"{slab} — external task score vs time")
        axc.legend(fontsize=7)
        axc.grid(True, alpha=0.3)
        axc.set_ylim(0, ymax * 1.02 + 0.01)
        axc.axhline(ymax, color="gray", ls="--", lw=0.7, alpha=0.55)
        out_c = RESULTS / f"experiment_16_curve_{si}.png"
        RESULTS.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_c, dpi=150)
        plt.close(fig2)
        print(f"  wrote {out_c}")

        fig3, axe = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
        for vi, c_lab, c in zip(range(3), VAR_SHORT, colors):
            cl = [r for r in d["rows"] if r["variant"] == vi]
            if vi == 0:
                continue
            t_arr, m_arr = _mean_curve(cl, max_ticks, args.log_stride, "curve_E_local")
            axe.plot(t_arr, m_arr, "-", color=c, label=c_lab, lw=1.4)
        axe.set_xlabel("FSM tick")
        axe.set_ylabel(r"mean $E_{\mathrm{local}}$ (logged at OLD_CLEAR)")
        axe.set_title(f"{slab} — internal propagated error vs time (variants 1–2)")
        axe.legend(fontsize=7)
        axe.grid(True, alpha=0.3)
        out_e = RESULTS / f"experiment_16_Elocal_{si}.png"
        fig3.savefig(out_e, dpi=150)
        plt.close(fig3)
        print(f"  wrote {out_e}")

        fig4, axd = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
        for vi, c_lab, c in zip(range(3), VAR_SHORT, colors):
            cl = [r for r in d["rows"] if r["variant"] == vi]
            if vi == 0:
                continue
            t_arr, m_arr = _mean_curve(cl, max_ticks, args.log_stride, "curve_depth")
            axd.plot(t_arr, m_arr, "-", color=c, label=c_lab, lw=1.4)
        axd.set_xlabel("FSM tick")
        axd.set_ylabel("mean backward depth (gates with $e>0$)")
        axd.set_title(f"{slab} — error propagation depth vs time (variants 1–2)")
        axd.legend(fontsize=7)
        axd.grid(True, alpha=0.3)
        out_d = RESULTS / f"experiment_16_depth_{si}.png"
        fig4.savefig(out_d, dpi=150)
        plt.close(fig4)
        print(f"  wrote {out_d}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
