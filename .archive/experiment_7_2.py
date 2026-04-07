#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7.2 — **capacity phase transition** vs extra internal gates (after fixing topology).

Motivation (7.1): the legacy chain forward ignored x3.. for n_in>3, capping realizability.
Here we use ``TTChainLearnerExp72``:

* **Topology A (default):** r <- g0(x0,x1); r <- gj(r, x_{j+1}); then K gates r <- g(r,r).
  Gates per output = (n_in - 1) + K.
* **Topology B:** legacy slice + K gates r <- g(r,r) (same 3 effective inputs for n_in>2).

Same policies, seeds, and tasks as experiment 7; sweep **K** (``extra_gates``) and plot success /
mean ticks vs K.

**Typical reading of results**

* **Parity 3x1:** extra gates often do *not* help; large K can *hurt* (bigger search space, same
  realizability class for easy targets) -- classic over-parameterization under a weak local
  optimizer.
* **Random 4x2:** if success stays at 0% for all K tried, either capacity is still too small *or*
  (increasingly likely) the hill-climbing / single-gate dynamics are too weak for that regime.
* **Next check:** run ``experiment_7_1_exp72a.py`` for topology A and the same N, K list: if the
  *exact* realizable set grows with K but learning does not, the bottleneck is optimization, not
  wiring-limited capacity.

One-line summary: with topology A, extra gates can increase search difficulty faster than they
increase usable learnability under the current learning rule.

Run from repo root:
  python src/experiments/experiment_7_2.py
  python src/experiments/experiment_7_2.py --topology B --k-list 0,2,4

Output: src/results/experiment_7_2_capacity_K.png
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

import experiment_72_worker as e72w  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_7_2_capacity_K.png"

BATCH_SEED_START = 1

# (label, n_in, n_out, target_kind, table_seed or None, max_ticks, n_seeds)
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
Experiment 7.2 takeaway (typical interpretation)
----------------------------------------------------------------------
- More gates is not automatically better learning: extra capacity can enlarge the search space
  and slow or destabilize the same local rule (especially on easy structured tasks like parity).
- If random 4x2 stays at 0% success for all K, capacity may still be short -- but if exact
  realizability vs K (see experiment_7_1_exp72a.py) grows while success does not, the blocker
  is likely the optimizer / signal, not topology-limited expressivity.
- Next step: python src/experiments/experiment_7_1_exp72a.py --n-in 4 --k-list 0,2,4,8
  (raise --max-configs if needed for larger K).
----------------------------------------------------------------------
""".strip()


def run_scenario_k(
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    extra_gates: int,
    topology: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_pol = len(POLICY_LABELS)
    fixed_ts = table_seed_fixed
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, str]] = []
    for pi in range(n_pol):
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            jobs.append((pi, n_in, n_out, kind, s, max_ticks, fixed_ts, extra_gates, topology))

    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        results = pool.map(e72w.run_one, jobs)

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
    env = os.environ.get("EXPERIMENT_72_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
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

    ap = argparse.ArgumentParser(description="Experiment 7.2: capacity sweep over K extra gates")
    ap.add_argument(
        "--topology",
        choices=("A", "B"),
        default="A",
        help="A = full-input chain + K×(r,r); B = legacy chain + K×(r,r)",
    )
    ap.add_argument(
        "--k-list",
        type=str,
        default="0,2,4,8",
        help="comma-separated extra gate counts K (default 0,2,4,8)",
    )
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: CPU count)")
    args = ap.parse_args()
    k_values = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not k_values or any(k < 0 for k in k_values):
        print("invalid --k-list", file=sys.stderr)
        return 1
    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, workers)
    topology = args.topology

    n_pol = len(POLICY_LABELS)
    n_K = len(k_values)
    # all_sr[scenario][k_idx, policy]
    all_sr: List[np.ndarray] = []
    all_mt: List[np.ndarray] = []

    for lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        sr_mat = np.zeros((n_K, n_pol))
        mt_mat = np.full((n_K, n_pol), np.nan)
        for ki, K in enumerate(k_values):
            n_jobs = n_pol * n_seeds
            print(
                f"{lab}  topology={topology}  K={K}  "
                f"({n_seeds} seeds x {n_pol} policies = {n_jobs} jobs, {workers} workers)..."
            )
            sr, mt = run_scenario_k(
                n_in, n_out, kind, ts_fix, max_ticks, n_seeds, workers, K, topology
            )
            sr_mat[ki] = sr
            mt_mat[ki] = mt
        all_sr.append(sr_mat)
        all_mt.append(mt_mat)

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(10, 3.8 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_pol))
    fig.suptitle(
        rf"Experiment 7.2 — success vs extra gates $K$ (topology {topology})",
        fontsize=12,
    )

    for si, (row, (slab, *_)) in enumerate(zip(axes, SCENARIOS)):
        ax0, ax1 = row[0], row[1]
        sr_mat = all_sr[si]
        mt_mat = all_mt[si]
        for pi in range(n_pol):
            ax0.plot(k_values, sr_mat[:, pi], "o-", color=colors[pi], label=POLICY_SHORT[pi], markersize=5)
            y_mt = mt_mat[:, pi]
            mask = np.isfinite(y_mt)
            if np.any(mask):
                ax1.plot(
                    np.array(k_values)[mask],
                    y_mt[mask],
                    "o-",
                    color=colors[pi],
                    label=POLICY_SHORT[pi],
                    markersize=5,
                )
        ax0.set_xlabel(r"extra gates $K$ per output")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(slab)
        ax0.legend(loc="best", fontsize=7)
        ax0.grid(True, alpha=0.3)
        ax0.set_xticks(k_values)

        ax1.set_xlabel(r"extra gates $K$ per output")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.legend(loc="best", fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
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
