#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 9 — **local credit assignment for gate selection** (global accept/reject unchanged).

Baseline: sample which gate to mutate uniformly (LFSR), as in experiments 7.2 / 8 with ``B=1``.

Treatment: over the full truth table, each gate accumulates counts of rows where its output
subnet is used and the corresponding output line is **correct** vs **wrong**; sample ``u``
with probability proportional to ``c_bad + 1`` (``--credit-mode bad``) or
``max(1, c_bad - c_good + 1)`` (``--credit-mode delta``). Proposal and COMPARE are unchanged.

Same tasks and default topology / capacity as experiment 8 (fixed ``K``, topology A).

Run from repo root:
  python src/experiments/experiment_9.py
  python src/experiments/experiment_9.py --topology A --k 4 --credit-mode bad

Output: src/results/experiment_9_credit_selection.png
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

import experiment_9_worker as e9w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_9_credit_selection.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

SEL_SHORT = ["uniform", "credit"]

TAKEAWAY_LINES = """
----------------------------------------------------------------------
Experiment 9 takeaway (typical interpretation)
----------------------------------------------------------------------
- If credit-weighted selection raises success on hard targets vs uniform, the bottleneck
  includes **where** updates fire, not only global score shape or raw capacity.
- If curves coincide, either the task does not benefit from this coarse local signal or
  the weighting is too weak / too noisy (try --credit-mode delta).
- Mutation and acceptance are intentionally identical so this run isolates gate selection.
----------------------------------------------------------------------
""".strip()


def run_scenario_sel(
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
    n_sel = 2
    fixed_ts = table_seed_fixed
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str, str]] = []
    for si in range(n_sel):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append(
                (si, n_in, n_out, kind, s, max_ticks, fixed_ts, extra_gates, topology, credit_mode)
            )

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        results = pool.map(e9w.run_one, jobs)

    sr = np.zeros(n_sel)
    mt = np.full(n_sel, np.nan)
    ticks_by: List[List[int]] = [[] for _ in range(n_sel)]
    ok_ct = [0] * n_sel
    for sel_idx, ok, ticks in results:
        if ok:
            ok_ct[sel_idx] += 1
            ticks_by[sel_idx].append(ticks)
    for si in range(n_sel):
        sr[si] = ok_ct[si] / n_seeds
        if ticks_by[si]:
            mt[si] = float(mean(ticks_by[si]))
    return sr, mt


def _default_workers() -> int:
    env = os.environ.get("EXPERIMENT_9_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
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

    ap = argparse.ArgumentParser(description="Experiment 9: credit-weighted gate selection")
    ap.add_argument(
        "--topology",
        choices=("A", "B"),
        default="A",
        help="same as experiment 7.2 / 8 (default A)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=4,
        help="fixed extra gates K per output (default 4)",
    )
    ap.add_argument(
        "--credit-mode",
        choices=("bad", "delta"),
        default="bad",
        help="bad: weight c_bad+1; delta: max(1, c_bad-c_good+1)",
    )
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: CPU count)")
    args = ap.parse_args()
    if args.k < 0:
        print("invalid --k", file=sys.stderr)
        return 1
    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)
    topology = args.topology
    K = args.k
    credit_mode = args.credit_mode

    n_sel = 2
    x = np.arange(n_sel)
    width = 0.55

    mode_note = (
        r"$w_i \propto c_i^{\mathrm{bad}}+1$"
        if credit_mode == "bad"
        else r"$w_i \propto \max(1,\, c_i^{\mathrm{bad}}-c_i^{\mathrm{good}}+1)$"
    )

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(8, 3.6 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    fig.suptitle(
        rf"Experiment 9 — uniform vs credit-weighted gate pick ($K={K}$, topology {topology}); {mode_note}",
        fontsize=11,
    )

    for si, row in enumerate(axes):
        ax0, ax1 = row[0], row[1]
        slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds = SCENARIOS[si]
        n_jobs = n_sel * n_seeds
        print(
            f"{slab}  topology={topology}  K={K}  credit_mode={credit_mode}  "
            f"({n_seeds} seeds × {n_sel} selectors = {n_jobs} jobs, {workers} workers)..."
        )
        sr, mt = run_scenario_sel(
            n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, K, topology, credit_mode
        )

        ax0.bar(x, sr, width, color=["steelblue", "darkorange"], edgecolor="black", linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(SEL_SHORT, fontsize=9)
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(slab)
        ax0.grid(True, axis="y", alpha=0.3)

        colors = ["steelblue", "darkorange"]
        for j in range(n_sel):
            if np.isfinite(mt[j]):
                ax1.bar(j, mt[j], width, color=colors[j], edgecolor="black", linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(SEL_SHORT, fontsize=9)
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
