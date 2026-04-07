#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7 — systematic heterogeneous p_i vs fixed baseline.

Building on experiment 6: global schedules and scalar sigma-feedback did not beat a
simple fixed p ~ 1/4; plastic-mapped per-gate p_i helped on speed. Here we compare
**local** rules explicitly:

1. **Fixed** ``plateau_mask=3`` (~1/4 tie accept) — strong baseline.
2. **Plastic-mapped p_i:** ``plastic[u]`` maps to masks 15,7,3,1 (unstable gate → higher p).
3. **Recent-success exploit:** gate ``u`` uses low p (mask 15) if it had an **accepted**
   strictly improving trial within the last ``exploit_window`` global COMPAREs; else mask 3.
4. **Mixed:** ``max(plastic_mask(u), success_mask(u))`` — favor exploitation if *either*
   plastic is low or the gate recently improved (larger mask = lower p).

Two scenarios (push toward experiment-4-style hardness):

* **Easy:** N=3 M=1 parity (same as exp 6).
* **Hard:** N=4 M=2 **random** targets (table seed per trajectory seed).

Runs are parallelized over (policy, seed) jobs via ``multiprocessing`` (spawn-safe worker
in ``ref/experiment_7_worker.py``). Override worker count with env ``EXPERIMENT_7_WORKERS``
or pass ``--workers N`` (default: CPU count).

Run from repo root:
  python src/experiments/experiment_7.py

Output: src/results/experiment_7_heterogeneous_p.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_7_worker as e7w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_7_heterogeneous_p.png"

BATCH_SEED_START = 1

# (label, n_in, n_out, target_kind, table_seed or None for random-per-seed, max_ticks, n_seeds)
SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

POLICY_LABELS = [
    r"fixed $p\!\sim\!1/4$",
    r"plastic $p_i$",
    r"success-exploit $p_i$",
    r"mixed plastic+success",
]


def run_scenario(
    _scen_label: str,
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns success_rate[policy], mean_ticks_ok[policy] (nan if no successes)."""
    n_pol = len(POLICY_LABELS)
    fixed_ts = table_seed_fixed
    jobs: List[tuple[int, int, int, str, int, int, int | None]] = []
    for pi in range(n_pol):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((pi, n_in, n_out, kind, s, max_ticks, fixed_ts))

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        results = pool.map(e7w.run_one, jobs)

    sr = np.zeros(n_pol)
    mt = np.full(n_pol, np.nan)
    ticks_by_pi: List[List[int]] = [[] for _ in range(n_pol)]
    ok_ct = [0] * n_pol
    for policy_idx, ok, ticks in results:
        if ok:
            ok_ct[policy_idx] += 1
            ticks_by_pi[policy_idx].append(ticks)
    for pi in range(n_pol):
        sr[pi] = ok_ct[pi] / n_seeds
        if ticks_by_pi[pi]:
            mt[pi] = float(mean(ticks_by_pi[pi]))
    return sr, mt


def _default_workers() -> int:
    env = os.environ.get("EXPERIMENT_7_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Experiment 7: heterogeneous p_i (parallel)")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="process pool size (default: EXPERIMENT_7_WORKERS or CPU count)",
    )
    args = ap.parse_args()
    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)

    n_pol = len(POLICY_LABELS)
    policy_short = ["fixed", "plastic", "success", "mixed"]

    all_sr: List[np.ndarray] = []
    all_mt: List[np.ndarray] = []

    for lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        n_jobs = n_pol * n_seeds
        print(
            f"Running scenario: {lab} ({n_seeds} seeds x {n_pol} policies = {n_jobs} jobs, "
            f"{workers} workers, {max_ticks} tick cap)..."
        )
        sr, mt = run_scenario(lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers)
        all_sr.append(sr)
        all_mt.append(mt)

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11, 3.8 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Experiment 7 — heterogeneous $p_i$ vs fixed baseline (two scenarios)",
        fontsize=12,
    )

    x = np.arange(n_pol)
    w = 0.55
    for si, (row, (slab, *_)) in enumerate(zip(axes, SCENARIOS)):
        ax0, ax1 = row[0], row[1]
        sr_row = all_sr[si]
        mt_row = all_mt[si]
        ax0.bar(x, sr_row, w, color="steelblue", edgecolor="white")
        ax0.set_xticks(x)
        ax0.set_xticklabels(policy_short, rotation=15, ha="right")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(f"{slab}")
        ax0.grid(True, axis="y", alpha=0.3)
        for i in range(n_pol):
            ax0.text(i, sr_row[i] + 0.03, f"{100 * sr_row[i]:.0f}%", ha="center", fontsize=8)

        ax1.set_xticks(x)
        ax1.set_xticklabels(policy_short, rotation=15, ha="right")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.grid(True, axis="y", alpha=0.3)
        ymax = float(np.nanmax(mt_row)) if np.any(np.isfinite(mt_row)) else 1.0
        ax1.set_ylim(0, ymax * 1.15 + 1)
        for i in range(n_pol):
            v = mt_row[i]
            if v == v:
                ax1.bar(i, v, w, color="coral", edgecolor="white")
                ax1.text(i, v + ymax * 0.02, f"{v:.0f}", ha="center", fontsize=8)
            else:
                ax1.text(i, ymax * 0.05, "n/a", ha="center", fontsize=8, color="gray")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)

    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
