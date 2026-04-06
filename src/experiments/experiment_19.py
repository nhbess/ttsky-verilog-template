#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 19 — **capacity sweep** over ``extra_gates`` (K) / total gates per output G.

Falsifies whether plateaus are representation limits vs optimization: if supervised keeps
gaining with larger G while local rules stay flat, learning is the story; if supervised
also saturates at modest G, the DAG family may be underparameterized for the task.

Three arms (same as ``experiment_19_worker``):
  supervised (Exp12), Exp16 clamp fixed-DAG, Exp18 local LUT stats (argmax).

Run from repo root:
  python src/experiments/experiment_19.py
  python src/experiments/experiment_19.py --variants 0,1              # skip Exp18 (much faster)
  python src/experiments/experiment_19.py --scenarios parity --variants 0,2
  # Heavier run (less noisy):  --seeds 12 --max-ticks 88000 --k-list 0,2,4,8,12,16,24

Requires ``tqdm`` for progress (``pip install tqdm``).

**Speed:** Variant **2** (Exp18) dominates wall time at large $K$ (heavy per tick). Use
``--variants 0,1`` for a supervised vs clamp-only sweep. The pool uses a tuned ``chunksize``
to cut IPC overhead on Windows.

Outputs:
  src/results/experiment_19_capacity.png

**Interpretation (short budgets, few seeds).** The top row is **exact** success (hit full
truth table within ``max_ticks``). With few seeds the success curve is quantized (0/4, 1/4, …).
**Defaults are tuned for exploration** (fewer K values, seeds, and ticks than a publication
run). The **bottom row** (mean final score / max) is usually the clearer signal on a tight
budget. Scale up with ``--seeds``, ``--max-ticks``, and a denser ``--k-list`` when you need
firmer conclusions.
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

import experiment_19_worker as e19w  # noqa: E402
from tt_chain_learner_exp72 import gpo_topology_a, integration_gates_topology_a  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_19_capacity.png"

BATCH_SEED_START = 1

# (label, n_in, n_out, kind, fixed_ts, max_ticks, n_seeds) — light defaults for exploration
SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("random 4×2", 4, 2, "random", None, 20_000, 4),
    ("parity 3×1", 3, 1, "parity", 0, 12_000, 4),
]

VAR_LABELS = [
    "supervised (Exp12)",
    "clamp $E_{local}$ (Exp16 fixed)",
    "local LUT fit (Exp18)",
]


def _parse_k_list(s: str) -> List[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out or any(k < 0 for k in out):
        raise ValueError("k-list must be non-empty non-negative integers")
    return sorted(set(out))


def _parse_variants(s: str) -> List[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out or any(v not in (0, 1, 2) for v in out):
        raise ValueError("variants must be a subset of 0,1,2 (supervised, clamp, Exp18)")
    return sorted(set(out))


def collect_capacity(
    n_in: int,
    n_out: int,
    kind: str,
    ts_fix: int | None,
    max_ticks: int,
    n_seeds: int,
    workers: int,
    k_list: List[int],
    log_stride: int,
    variants: List[int],
    *,
    progress_desc: str,
) -> Dict[str, Any]:
    jobs: List[tuple[int, int, int, str, int, int, int | None, int, int]] = []
    for k in k_list:
        for vi in variants:
            for i in range(n_seeds):
                s = BATCH_SEED_START + i
                jobs.append((vi, n_in, n_out, kind, s, max_ticks, ts_fix, k, log_stride))

    from multiprocessing import Pool

    from tqdm import tqdm

    chunksize = max(1, len(jobs) // (workers * 4)) if jobs else 1

    with Pool(processes=workers) as pool:
        rows = list(
            tqdm(
                pool.imap_unordered(e19w.run_one, jobs, chunksize=chunksize),
                total=len(jobs),
                desc=progress_desc,
                unit="job",
            )
        )

    g_vals = [gpo_topology_a(n_in, k) for k in k_list]
    nk = len(k_list)
    sr = np.full((3, nk), np.nan)
    fin = np.full((3, nk), np.nan)
    n_ct = np.zeros((3, nk), dtype=int)
    sr_acc = np.zeros((3, nk))
    fin_acc = np.zeros((3, nk))
    for r in rows:
        ki = k_list.index(int(r["extra_gates"]))
        vi = int(r["variant"])
        n_ct[vi, ki] += 1
        if r["ok_task"]:
            sr_acc[vi, ki] += 1
        fin_acc[vi, ki] += r["final_external_score"] / max(1, r["max_score"])
    for vi in range(3):
        for j in range(nk):
            n = int(n_ct[vi, j])
            if n == 0:
                continue
            sr[vi, j] = sr_acc[vi, j] / n
            fin[vi, j] = fin_acc[vi, j] / n
    return {
        "k_list": k_list,
        "g_vals": g_vals,
        "ig": integration_gates_topology_a(n_in),
        "sr": sr,
        "fin_mean": fin,
        "rows": rows,
        "variants": variants,
    }


def _default_workers() -> int:
    for key in ("EXPERIMENT_19_WORKERS", "EXPERIMENT_18_WORKERS", "EXPERIMENT_12_WORKERS"):
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

    ap = argparse.ArgumentParser(description="Experiment 19: capacity sweep vs extra_gates K")
    ap.add_argument(
        "--k-list",
        type=str,
        default="0,4,8,12,16",
        help="comma-separated extra_gates K (topology A: G = integration + K)",
    )
    ap.add_argument(
        "--log-stride",
        type=int,
        default=1200,
        help="curve logging interval in ticks (larger = slightly faster, sparser curves)",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override per-scenario n_seeds")
    ap.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="override max_ticks for every scenario (defaults: 20k random, 12k parity)",
    )
    ap.add_argument(
        "--variants",
        type=str,
        default="0,1,2",
        help="comma-separated subset of 0,1,2 = supervised, clamp, Exp18 (omit 2 to save time)",
    )
    ap.add_argument(
        "--scenarios",
        type=str,
        default="all",
        choices=("all", "parity", "random"),
        help="run only parity 3×1, only random 4×2, or both (default: all)",
    )
    args = ap.parse_args()
    try:
        k_list = _parse_k_list(args.k_list)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        variants = _parse_variants(args.variants)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    workers = max(1, args.workers if args.workers is not None else _default_workers())

    if args.scenarios == "all":
        scenario_rows = list(SCENARIOS)
    else:
        kind_want = args.scenarios
        scenario_rows = [row for row in SCENARIOS if row[3] == kind_want]
    if not scenario_rows:
        print("No scenarios match --scenarios filter.", file=sys.stderr)
        return 1

    n_sc = len(scenario_rows)
    if n_sc == 1:
        fig, ax_col = plt.subplots(2, 1, figsize=(5.2, 6.0), constrained_layout=True)
        axes = np.array([[ax_col[0]], [ax_col[1]]])
    else:
        fig, axes = plt.subplots(2, n_sc, figsize=(5.2 * n_sc, 6.0), constrained_layout=True)

    colors = ["steelblue", "darkorange", "seagreen"]

    for si, (slab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds) in enumerate(scenario_rows):
        if args.max_ticks is not None:
            max_ticks = max(1, args.max_ticks)
        if args.seeds is not None:
            n_seeds = max(1, args.seeds)
        nv = len(variants)
        print(
            f"{slab}: K in {k_list}  ({n_seeds} seeds × {nv} variants × {len(k_list)} K, {workers} workers)..."
        )
        d = collect_capacity(
            n_in,
            n_out,
            kind,
            ts_fix,
            max_ticks,
            n_seeds,
            workers,
            k_list,
            args.log_stride,
            variants,
            progress_desc=f"exp19 | {slab}",
        )
        gx = np.array(d["g_vals"], dtype=float)
        ax0, ax1 = axes[0, si], axes[1, si]
        for vi in variants:
            ax0.plot(gx, d["sr"][vi], "o-", color=colors[vi], label=VAR_LABELS[vi], lw=1.3, ms=4)
            ax1.plot(gx, d["fin_mean"][vi], "o-", color=colors[vi], label=VAR_LABELS[vi], lw=1.3, ms=4)
        ax0.set_ylabel("P(exact solve)")
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(f"{slab} — exact success (all minterms); harsh + coarse at few seeds")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=6, loc="best")

        ax1.set_xlabel(r"gates per output $G$ (topology A: $G = {} + K$)".format(d["ig"]))
        ax1.set_ylabel("mean final score / max (1 − gap)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(f"{slab} — partial performance (main signal if top row is all zeros)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6, loc="best")

        for ax in (ax0, ax1):
            ax.set_xticks(gx)
            ax.set_xticklabels([str(int(g)) for g in gx], fontsize=7)

    fig.suptitle(
        "Experiment 19 — capacity vs $G$: exact success is binary; mean score shows partial learning",
        fontsize=11,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
