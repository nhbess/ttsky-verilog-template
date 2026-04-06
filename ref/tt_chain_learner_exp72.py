# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7.2 chain learner: **capacity** via extra gates after fixing (or keeping) connectivity.

Topology **A** (recommended): integrate all inputs along one register
  r <- g0(x0,x1);  r <- g_j(r, x_{j+1}) for j=1..n_in-2;  then K times r <- g(r,r).

Topology **B**: legacy ``TTChainLearner`` slice (only x0,x1,x2 affect the DAG for n_in>2),
then K times r <- g(r,r).

Gates per output:  (n_in - 1) + K  (A, n_in>=2)  or  n_in + K  (B, n_in>2), with n_in<=2 -> 1+K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tt_chain_learner_spec as chain


def integration_gates_topology_a(n_in: int) -> int:
    """Gates that consume inputs (before K copies of r <- g(r,r))."""
    return 1 if n_in <= 2 else n_in - 1


def gpo_topology_a(n_in: int, k_extra: int) -> int:
    """Gates per output for topology A."""
    return integration_gates_topology_a(n_in) + k_extra


def forward_topology_a(g_slice: Sequence[int], xs: Sequence[int], n_in: int) -> int:
    """
    Evaluate one output slice for topology A. ``len(g_slice)`` must be
    ``integration_gates_topology_a(n_in) + k_extra`` for some ``k_extra >= 0``.
    """
    n = n_in
    ig = integration_gates_topology_a(n)
    if len(g_slice) < ig:
        raise ValueError(f"g_slice len {len(g_slice)} < integration gates {ig}")
    k_extra = len(g_slice) - ig
    if n <= 2:
        r = chain.gate_eval(g_slice[0], xs[0], xs[1])
    else:
        r = chain.gate_eval(g_slice[0], xs[0], xs[1])
        for j in range(1, n - 1):
            r = chain.gate_eval(g_slice[j], r, xs[j + 1])
    for k in range(k_extra):
        r = chain.gate_eval(g_slice[ig + k], r, r)
    return r


@dataclass
class TTChainLearnerExp72(chain.TTChainLearner):
    """Same FSM as ``TTChainLearner``; different ``gpo`` and forward."""

    extra_gates: int = 0
    topology: str = "A"  # "A" = full-input chain + K×(r,r); "B" = legacy + K×(r,r)

    @property
    def gpo(self) -> int:  # type: ignore[override]
        n = self.n_in
        if self.topology == "B":
            if n <= 2:
                base = 1
            else:
                base = n
            return base + self.extra_gates
        # topology A
        return gpo_topology_a(n, self.extra_gates)

    def __post_init__(self) -> None:
        if not self.target:
            self.target = chain.make_truth_tables("parity", self.n_in, self.n_out, table_seed=0)
        ng = self.n_out * self.gpo
        if len(self.gate) != ng:
            self.gate = [0] * ng
        if len(self.plastic) != ng:
            self.plastic = [3] * ng

    def reset(self, seed: int = 0xACE1) -> None:
        ng = self.n_out * self.gpo
        self.gate = [0] * ng
        self.plastic = [3] * ng
        self.unit_sel = 0
        self.sample_idx = 0
        self.old_score = 0
        self.new_score = 0
        self.old_gate = 0
        self.trial_gate = 0
        self.lfsr = seed & 0xFFFF or 0xACE1
        self.fsm = chain.FsmState.IDLE
        for _ in range(ng):
            self.lfsr = chain.lfsr16_step(self.lfsr)
            self.gate[_] = self.lfsr & 0xF

    def _forward_one_out_slice(self, xs: Sequence[int], base: int) -> int:
        g = self.gate
        n = self.n_in
        if self.topology == "B":
            if n <= 2:
                r = chain.gate_eval(g[base], xs[0], xs[1])
                ig = 1
            else:
                h1 = chain.gate_eval(g[base], xs[0], xs[1])
                r = xs[2]
                for k in range(1, n - 1):
                    r = chain.gate_eval(g[base + k], h1, r)
                r = chain.gate_eval(g[base + n - 1], h1, r)
                ig = n
            for k in range(self.extra_gates):
                r = chain.gate_eval(g[base + ig + k], r, r)
            return r

        sub = tuple(g[base + i] for i in range(self.gpo))
        return forward_topology_a(sub, xs, n)

    def _forward_one_out_trial(self, xs: Sequence[int], base: int) -> int:
        g = self.gate
        us = self.unit_sel
        tg = self.trial_gate
        n = self.n_in
        tro = getattr(self, "_trial_overrides", None)

        def gv(off: int) -> int:
            gi = base + off
            if tro is not None:
                if gi in tro:
                    return tro[gi]
                return g[gi]
            return tg if gi == us else g[gi]

        if self.topology == "B":
            if n <= 2:
                r = chain.gate_eval(gv(0), xs[0], xs[1])
                ig = 1
            else:
                h1 = chain.gate_eval(gv(0), xs[0], xs[1])
                r = xs[2]
                for k in range(1, n - 1):
                    r = chain.gate_eval(gv(k), h1, r)
                r = chain.gate_eval(gv(n - 1), h1, r)
                ig = n
            for k in range(self.extra_gates):
                r = chain.gate_eval(gv(ig + k), r, r)
            return r

        sub = tuple(gv(i) for i in range(self.gpo))
        return forward_topology_a(sub, xs, n)
