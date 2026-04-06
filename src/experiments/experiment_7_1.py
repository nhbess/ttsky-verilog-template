#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7.1 — exact **topology-constrained capacity** of the chain learner.

This is not generic Karnaugh / minimal-gate count for an unconstrained circuit; it is the
size of the family

  R_{N,M}^{(chain)} = { truth tables realized by some assignment of 2-input LUT gates }

for the fixed wiring in ``tt_chain_learner_spec`` (disjoint M subnets, each an N-input chain).

**Architecture note (important):** for ``n_in >= 3``, ``_forward_one_out_slice`` only ever reads
``x0, x1, x2``; inputs ``x3 .. x_{n-1}`` are not wired into the gate DAG. Larger ``n_in`` adds
gates in depth along ``(h1, r)`` but does not increase the number of input bits observed. So
``|S_N|`` plateaus once ``N >= 3`` for this spec, and targets like 4-input parity that depend on
``x3`` are unrealizable regardless of training.

Method: brute force all gate tuples in {0..15}^{n_gates} for small (N, M), hash the
resulting multi-output truth table, count distinct tables.

For independent subnets, |R_{N,M}| = |S_N|^M where S_N is the set of single-output tables
at that N; the script checks this for a tiny case.

Run from repo root:
  python src/experiments/experiment_7_1.py

Optional plot: src/results/experiment_7_1_chain_capacity.png
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_chain_learner_spec as chain  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_7_1_chain_capacity.png"


def forward_one_row(g_slice: Sequence[int], xs: Sequence[int], n_in: int) -> int:
    """Same semantics as ``TTChainLearner._forward_one_out_slice`` (base offset 0)."""
    g = g_slice
    n = n_in
    base = 0
    if n <= 2:
        return chain.gate_eval(g[base], xs[0], xs[1])
    h1 = chain.gate_eval(g[base], xs[0], xs[1])
    r = xs[2]
    for k in range(1, n - 1):
        r = chain.gate_eval(g[base + k], h1, r)
    return chain.gate_eval(g[base + n - 1], h1, r)


def truth_table_int_subnet(g_slice: Tuple[int, ...], n_in: int) -> int:
    """Pack outputs for rows 0..2^n-1 into an integer (row i -> bit i)."""
    bits = 0
    for idx in range(1 << n_in):
        xs = chain.row_bits(idx, n_in)
        y = forward_one_row(g_slice, xs, n_in)
        bits |= y << idx
    return bits


def truth_table_key_full(gates: Tuple[int, ...], n_in: int, n_out: int) -> Tuple[int, ...]:
    """One int per output line (disjoint gate slices)."""
    gpo = chain.gates_per_output(n_in)
    key: List[int] = []
    for m in range(n_out):
        base = m * gpo
        sub = gates[base : base + gpo]
        key.append(truth_table_int_subnet(sub, n_in))
    return tuple(key)


def enumerate_realizable(
    n_in: int, n_out: int, *, show_progress: bool = False
) -> FrozenSet[Tuple[int, ...]]:
    gpo = chain.gates_per_output(n_in)
    ng = gpo * n_out
    seen: set[Tuple[int, ...]] = set()
    total = 16**ng
    step = max(1, total // 50)
    for i, tup in enumerate(itertools.product(range(16), repeat=ng)):
        if show_progress and i % step == 0:
            print(f"    {i}/{total} configs...", flush=True)
        seen.add(truth_table_key_full(tuple(tup), n_in, n_out))
    return frozenset(seen)


def universe_size(n_in: int, n_out: int) -> int:
    """Number of arbitrary M-column truth tables: (2^(2^n))^M."""
    return 1 << (n_out * (1 << n_in))


def main() -> int:
    ap = argparse.ArgumentParser(description="Chain topology exact capacity (small brute force)")
    ap.add_argument(
        "--n-max",
        type=int,
        default=5,
        help="largest N (inputs) for single-output sweep (default 5; N=6 is 16^6 configs)",
    )
    ap.add_argument(
        "--include-n3-m2",
        action="store_true",
        help="optional empirical brute-force N=3 M=2 (16^6 configs; slow, minutes on some machines)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help=f"write log2(|S_N|) plot to {OUT_FIG}",
    )
    ap.add_argument("--progress", action="store_true", help="progress for large enumerations")
    args = ap.parse_args()

    print(
        "Experiment 7.1 - chain realizability (exact brute force, topology from tt_chain_learner_spec)\n"
    )
    print(
        "Note: for n_in>=3 the reference forward() only uses bits x0,x1,x2; "
        "higher-index inputs are ignored. So |S_N| is constant for N>=3 here.\n"
    )

    # Single-output capacity |S_N|
    rows: List[Tuple[int, int, int, int, float]] = []
    s_cache: Dict[int, FrozenSet[Tuple[int, ...]]] = {}
    for n_in in range(2, args.n_max + 1):
        gpo = chain.gates_per_output(n_in)
        n_conf = 16 ** (gpo * 1)
        print(f"N={n_in}  M=1  gates_per_output={gpo}  configs={n_conf:,}", flush=True)
        s_set = enumerate_realizable(n_in, 1, show_progress=args.progress and n_conf >= 100_000)
        s_cache[n_in] = s_set
        s_n = len(s_set)
        u = universe_size(n_in, 1)
        frac = s_n / u
        lg2 = math.log2(s_n) if s_n > 0 else float("-inf")
        print(f"  |S_N| = {s_n} distinct single-output truth tables", flush=True)
        print(
            f"  log2(|S_N|) = {lg2:.4f}  (universe log2 = {math.log2(u):.4f}, fraction = {frac:.6e})",
            flush=True,
        )
        print(flush=True)
        rows.append((n_in, gpo, n_conf, s_n, frac))

    # Multi-output: M disjoint subnets => |R_{N,M}| = |S_N|^M exactly (no enumeration needed).
    if s_cache:
        n0 = max(s_cache.keys())
        s_n0 = len(s_cache[n0])
        print(
            "Multi-output capacity: subnets use disjoint gate RAMs, so any M-tuple of "
            f"functions in S_N is realizable and all keys are M-tuples in S_N^M. "
            f"Hence |R_{{N,M}}| = |S_N|^M (e.g. N={n0}, M=2 -> {s_n0}^2 = {s_n0 * s_n0}).\n"
        )

    if 3 in s_cache and args.include_n3_m2:
        s3 = len(s_cache[3])
        pred = s3 * s3
        print("N=3 M=2  (optional empirical brute-force)...", flush=True)
        r32 = enumerate_realizable(3, 2, show_progress=args.progress)
        print(f"  |R_{{3,2}}| = {len(r32)}  |S_3|^2 = {pred}  match: {len(r32) == pred}\n", flush=True)

    # Parity in S_N for N covered by the sweep
    print("Sanity: is standard parity column in S_N?")
    for n_in in range(2, args.n_max + 1):
        tgt = chain.make_truth_tables("parity", n_in, 1, table_seed=0)[0]
        packed = sum(tgt[i] << i for i in range(1 << n_in))
        ok = (packed,) in s_cache[n_in]
        print(f"  N={n_in} parity realizability: {ok}", flush=True)
    print(flush=True)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for --plot", file=sys.stderr)
            return 1
        ns = [r[0] for r in rows]
        logs = [math.log2(r[3]) for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(ns, logs, "o-", color="steelblue")
        ax.set_xlabel("N (inputs)")
        ax.set_ylabel(r"$\log_2 |S_N|$ (distinct single-output tables)")
        ax.set_title("Chain topology: exact single-output capacity")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ns)
        RESULTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FIG, dpi=150)
        plt.close(fig)
        print(f"Wrote {OUT_FIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
