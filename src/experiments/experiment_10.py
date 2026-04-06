#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 10 — **stronger local rule**: credit-weighted gate pick plus credit-modulated
local plasticity and plateau tie-accept.

Building on experiment 9 (credit-weighted ``u`` only), this adds:

* **INIT:** relative ``c_bad`` maps to how often the chosen gate is allowed to start a trial
  (1–4 slots out of four ``rnd2`` outcomes); flat ``c_bad`` falls back to ``plastic[u]+1``.
* **COMPARE:** relative ``c_bad`` maps to plateau mask (high bad → smaller mask → more tie
  accepts on plateaus; low bad → larger mask → fewer).

Variant **0** = uniform ``u`` (baseline ``TTChainLearnerExp72``). Variant **1** = experiment 9
(``TTChainLearnerExp72CreditWeighted``). Variant **2** = full local rule
(``TTChainLearnerExp72CreditPlastic``).

Same scenarios and defaults as experiments 8–9 (topology A, ``K`` extra gates).

Run from repo root:
  python src/experiments/experiment_10.py
  python src/experiments/experiment_10.py --k 4 --credit-mode bad

Output: src/results/experiment_10_credit_local_plastic.png
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

import experiment_10_worker as e10w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_10_credit_local_plastic.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

VAR_SHORT = ["uniform", "credit $u$", "credit + local"]

TAKEAWAY_LINES = """
----------------------------------------------------------------------
Experiment 10 takeaway (typical interpretation)
----------------------------------------------------------------------
- Compares experiment 9 (credit selection only) to the escalated rule that also modulates
  per-trial update rate and tie-accept from the same c_bad snapshot.
- If variant 2 beats variant 1 on easy tasks, the extra local structure is doing work; if
  hard targets move, local plasticity was part of the gap (if not, capability limits remain).
----------------------------------------------------------------------
""".strip()


def run_scenario(
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    extra_gates: int,
    topology: str,
    credit_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_var = 3
    fixed_ts = table_seed_fixed
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str, str]] = []
    for vi in range(n_var):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append(
                (vi, n_in, n_out, kind, s, max_ticks, fixed_ts, extra_gates, topology, credit_mode)
            )

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        results = pool.map(e10w.run_one, jobs)

    sr = np.zeros(n_var)
    mt = np.full(n_var, np.nan)
    ticks_by: List[List[int]] = [[] for _ in range(n_var)]
    ok_ct = [0] * n_var
    for variant, ok, ticks in results:
        if ok:
            ok_ct[variant] += 1
            ticks_by[variant].append(ticks)
    for vi in range(n_var):
        sr[vi] = ok_ct[vi] / n_seeds
        if ticks_by[vi]:
            mt[vi] = float(mean(ticks_by[vi]))
    return sr, mt


def _default_workers() -> int:
    env = os.environ.get("EXPERIMENT_10_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    env9 = os.environ.get("EXPERIMENT_9_WORKERS", "").strip()
    if env9.isdigit() and int(env9) > 0:
        return int(env9)
    env72 = os.environ.get("EXPERIMENT_72_WORKERS", "").strip()
    if env72.isdigit() and int(env72) > 0:
        return int(env72)
    env7 = os.environ.get("EXPERIMENT_7_WORKERS", "").strip()
    if env7.isdigit() and int(env7) > 0:
        return int(env7)
    return max(1, os.cpu_count() or 1)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser(description="Experiment 10: credit + local plasticity / plateau")
    ap.add_argument("--topology", choices=("A", "B"), default="A")
    ap.add_argument("--k", type=int, default=4, help="extra gates K per output (default 4)")
    ap.add_argument(
        "--credit-mode",
        choices=("bad", "delta"),
        default="bad",
        help="credit weights for gate pick (same as experiment 9)",
    )
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.k < 0:
        print("invalid --k", file=sys.stderr)
        return 1
    workers = max(1, args.workers if args.workers is not None else _default_workers())

    n_var = 3
    x = np.arange(n_var)
    width = 0.55
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(9, 3.6 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    mode_lbl = "bad (c_bad+1)" if args.credit_mode == "bad" else "delta"
    fig.suptitle(
        rf"Experiment 10 — uniform vs credit $u$ vs credit + local plastic/plateau ($K={args.k}$, "
        rf"topology {args.topology}, weights={mode_lbl})",
        fontsize=10,
    )

    for si, row in enumerate(axes):
        ax0, ax1 = row[0], row[1]
        slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds = SCENARIOS[si]
        n_jobs = n_var * n_seeds
        print(
            f"{slab}  topology={args.topology}  K={args.k}  credit_mode={args.credit_mode}  "
            f"({n_seeds} seeds × {n_var} variants = {n_jobs} jobs, {workers} workers)..."
        )
        sr, mt = run_scenario(
            n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, args.k, args.topology, args.credit_mode
        )

        ax0.bar(x, sr, width, color=colors, edgecolor="black", linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(VAR_SHORT, fontsize=8, rotation=12, ha="right")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(slab)
        ax0.grid(True, axis="y", alpha=0.3)

        for j in range(n_var):
            if np.isfinite(mt[j]):
                ax1.bar(j, mt[j], width, color=colors[j], edgecolor="black", linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(VAR_SHORT, fontsize=8, rotation=12, ha="right")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.grid(True, axis="y", alpha=0.3)
        if np.any(np.isfinite(mt)):
            ymax = float(np.nanmax(mt))
            ax1.set_ylim(0, ymax * 1.12 + 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    print()
    print(TAKEAWAY_LINES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
