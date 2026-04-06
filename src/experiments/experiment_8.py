#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 8 — **multi-gate mutation** (parallel proposals, one global accept/reject).

Motivation: the circuit evaluates all gates in parallel, but the default learner mutates
one gate per trial. Here we propose **B** distinct gate truth-table nibbles simultaneously,
re-score the full target once, and accept or reject the **joint** move (same plateau tie
rule as experiment 7.2). ``B=1`` uses the parent ``tick`` and matches serial updates.

Fixed setup (same tasks and policies as experiment 7.2): sweep **B** = ``mutation_batch``
with fixed extra gates **K** and topology, and plot success / mean ticks vs B.

Run from repo root:
  python src/experiments/experiment_8.py
  python src/experiments/experiment_8.py --topology A --k 4 --b-list 1,2,4,8

Output: src/results/experiment_8_mutation_batch_B.png
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

import experiment_8_worker as e8w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_8_mutation_batch_B.png"

BATCH_SEED_START = 1

SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]

POLICY_LABELS = [
    r"fixed $p\!\sim\!1/4$",
    r"plastic $p_i$",
    r"success-exploit $p_i$",
    r"mixed",
]

POLICY_SHORT = ["fixed", "plastic", "success", "mixed"]

TAKEAWAY_LINES = """
----------------------------------------------------------------------
Experiment 8 takeaway (typical interpretation)
----------------------------------------------------------------------
- Larger B lets the learner explore coordinated moves across gates in one global score step;
  that can help on hard targets (e.g. random 4×2) but may also accept noisier joint moves
  or dilute plasticity checks, so success vs B is not monotone.
- B=1 matches the serial single-site learner (same tick path as experiment 7.2 for the base
  class); compare curves at B=1 to experiment 7.2 at the same K and topology.
- If gains appear only at B>1, the bottleneck is partly **sequential** local search rather
  than capacity alone.
----------------------------------------------------------------------
""".strip()


def run_scenario_b(
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    extra_gates: int,
    topology: str,
    mutation_batch: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_pol = len(POLICY_LABELS)
    fixed_ts = table_seed_fixed
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str, int]] = []
    for pi in range(n_pol):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append(
                (pi, n_in, n_out, kind, s, max_ticks, fixed_ts, extra_gates, topology, mutation_batch)
            )

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        results = pool.map(e8w.run_one, jobs)

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
    env = os.environ.get("EXPERIMENT_8_WORKERS", "").strip()
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

    ap = argparse.ArgumentParser(description="Experiment 8: multi-gate mutation batch B")
    ap.add_argument(
        "--topology",
        choices=("A", "B"),
        default="A",
        help="same as experiment 7.2 (default A)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=4,
        help="fixed extra gates K per output (default 4)",
    )
    ap.add_argument(
        "--b-list",
        type=str,
        default="1,2,4,8",
        help="comma-separated mutation batch sizes B (default 1,2,4,8)",
    )
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: CPU count)")
    args = ap.parse_args()
    b_values = [int(x.strip()) for x in args.b_list.split(",") if x.strip()]
    if not b_values or any(b < 1 for b in b_values):
        print("invalid --b-list (need integers >= 1)", file=sys.stderr)
        return 1
    if args.k < 0:
        print("invalid --k", file=sys.stderr)
        return 1
    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)
    topology = args.topology
    K = args.k

    n_pol = len(POLICY_LABELS)
    n_B = len(b_values)
    all_sr: List[np.ndarray] = []
    all_mt: List[np.ndarray] = []

    for lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        sr_mat = np.zeros((n_B, n_pol))
        mt_mat = np.full((n_B, n_pol), np.nan)
        for bi, B in enumerate(b_values):
            n_jobs = n_pol * n_seeds
            print(
                f"{lab}  topology={topology}  K={K}  B={B}  "
                f"({n_seeds} seeds × {n_pol} policies = {n_jobs} jobs, {workers} workers)..."
            )
            sr, mt = run_scenario_b(
                n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, K, topology, B
            )
            sr_mat[bi] = sr
            mt_mat[bi] = mt
        all_sr.append(sr_mat)
        all_mt.append(mt_mat)

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(10, 3.8 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_pol))
    fig.suptitle(
        rf"Experiment 8 — success vs mutation batch $B$ ($K={K}$, topology {topology})",
        fontsize=12,
    )

    for si, (row, (slab, *_)) in enumerate(zip(axes, SCENARIOS)):
        ax0, ax1 = row[0], row[1]
        sr_mat = all_sr[si]
        mt_mat = all_mt[si]
        for pi in range(n_pol):
            ax0.plot(b_values, sr_mat[:, pi], "o-", color=colors[pi], label=POLICY_SHORT[pi], markersize=5)
            y_mt = mt_mat[:, pi]
            mask = np.isfinite(y_mt)
            if np.any(mask):
                ax1.plot(
                    np.array(b_values)[mask],
                    y_mt[mask],
                    "o-",
                    color=colors[pi],
                    label=POLICY_SHORT[pi],
                    markersize=5,
                )
        ax0.set_xlabel(r"mutation batch $B$ (gates per trial)")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(slab)
        ax0.legend(loc="best", fontsize=7)
        ax0.grid(True, alpha=0.3)
        ax0.set_xticks(b_values)

        ax1.set_xlabel(r"mutation batch $B$")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.legend(loc="best", fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(b_values)
        if np.any(np.isfinite(mt_mat)):
            ymax = float(np.nanmax(mt_mat))
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
