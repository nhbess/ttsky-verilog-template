#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7.1 (topology A + K) — exact realizability vs **extra gates**, matching experiment 7.2.

Brute-force count distinct single-output truth tables for ``forward_topology_a`` with
``gpo = (n_in - 1) + K`` (for n_in >= 2). Use this after 7.2 to separate:

* if |S_{N,K}| grows strongly with K but learning stays at 0% -> optimizer bottleneck
* if |S_{N,K}| barely grows -> capacity still tight

Large ``16^gpo`` is skipped unless ``--max-configs`` allows it.

Run from repo root:
  python src/experiments/experiment_7_1_exp72a.py --n-in 4 --k-list 0,2
  python src/experiments/experiment_7_1_exp72a.py --n-in 4 --k-list 0,2,4 --max-configs 500000000
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import FrozenSet, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_chain_learner_spec as chain  # noqa: E402
import tt_chain_learner_exp72 as e72  # noqa: E402


def truth_table_int_topo_a(g_slice: Tuple[int, ...], n_in: int) -> int:
    bits = 0
    for idx in range(1 << n_in):
        xs = chain.row_bits(idx, n_in)
        y = e72.forward_topology_a(g_slice, xs, n_in)
        bits |= y << idx
    return bits


def truth_table_key_full(
    gates: Tuple[int, ...], n_in: int, n_out: int, gpo: int
) -> Tuple[int, ...]:
    key: List[int] = []
    for m in range(n_out):
        base = m * gpo
        sub = gates[base : base + gpo]
        key.append(truth_table_int_topo_a(sub, n_in))
    return tuple(key)


def enumerate_realizable(
    n_in: int, n_out: int, k_extra: int, *, show_progress: bool
) -> FrozenSet[Tuple[int, ...]]:
    gpo = e72.gpo_topology_a(n_in, k_extra)
    ng = gpo * n_out
    seen: set[Tuple[int, ...]] = set()
    total = 16**ng
    step = max(1, total // 50)
    for i, tup in enumerate(itertools.product(range(16), repeat=ng)):
        if show_progress and i % step == 0:
            print(f"    {i}/{total} configs...", flush=True)
        seen.add(truth_table_key_full(tuple(tup), n_in, n_out, gpo))
    return frozenset(seen)


def universe_size(n_in: int, n_out: int) -> int:
    return 1 << (n_out * (1 << n_in))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exact realizability for experiment 7.2 topology A vs K"
    )
    ap.add_argument("--n-in", type=int, default=4, help="number of inputs N (default 4)")
    ap.add_argument("--m-out", type=int, default=1, help="outputs M for full enumeration (default 1)")
    ap.add_argument(
        "--k-list",
        type=str,
        default="0,2,4",
        help="comma-separated K (extra r<-g(r,r) gates per output)",
    )
    ap.add_argument(
        "--max-configs",
        type=int,
        default=8_000_000,
        help="skip (K,N,M) combos whose 16^(gpo*M) exceeds this (default 8e6)",
    )
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if any(k < 0 for k in k_list):
        print("invalid --k-list", file=sys.stderr)
        return 1

    n_in = args.n_in
    n_out = args.m_out

    print(
        "Experiment 7.1-exp72a - exact |S| for topology A (all inputs wired) + K extra gates\n",
        flush=True,
    )

    for K in k_list:
        gpo = e72.gpo_topology_a(n_in, K)
        ng = gpo * n_out
        n_conf = 16**ng
        print(
            f"N={n_in}  M={n_out}  K={K}  gpo={gpo}  total gates={ng}  configs={n_conf:,}",
            flush=True,
        )
        if n_conf > args.max_configs:
            print(
                f"  SKIP: configs > --max-configs ({args.max_configs}). "
                f"Raise --max-configs or drop K / M.",
                flush=True,
            )
            print(flush=True)
            continue
        s_set = enumerate_realizable(
            n_in, n_out, K, show_progress=args.progress and n_conf >= 200_000
        )
        s_sz = len(s_set)
        u = universe_size(n_in, n_out)
        frac = s_sz / u
        print(f"  |R| distinct truth tables = {s_sz}", flush=True)
        lg_r = math.log2(s_sz) if s_sz else float("-inf")
        print(
            f"  log2(|R|) = {lg_r:.4f}  "
            f"(universe log2 = {math.log2(u):.4f}, fraction = {frac:.6e})",
            flush=True,
        )
        if n_out == 1:
            print(
                f"  Predicted |R_{{N,2}}| = |S|^2 = {s_sz}^2 = {s_sz * s_sz} (disjoint subnets).",
                flush=True,
            )
        print(flush=True)

    print(
        "Compare to 7.2: if |S| or |R| rises with K but success stays 0% on random tasks, "
        "the blocker is likely the learning rule / search, not topology-limited realizability.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
