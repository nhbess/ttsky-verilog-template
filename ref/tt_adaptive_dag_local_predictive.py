# SPDX-License-Identifier: Apache-2.0
"""
Experiment 14 — **local predictive coding** on the adaptive DAG (no global COMPARE).

Only gates with an **immediate downstream** gate in the same subnet participate. On a
random minterm sample, build empirical ``P(y_j | y_i)`` from 2×2 counts with **leave-one-out**
prediction: ``ε`` = fraction of samples where ``y_j`` disagrees with the MAP predictor from
the other samples. Accept if ``ε`` decreases (plateau tie rule unchanged).

No subnet-output fallback (avoids task-aligned leakage). Rewire vs LUT: 50% coin flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import tt_chain_learner_spec as chain

import tt_adaptive_dag_learner as ad
import tt_adaptive_dag_local_unsup as adlu


def _prediction_error_loo(yi: List[int], yj: List[int]) -> float:
    S = len(yi)
    if S < 4:
        return 1.0
    wrong = 0
    for t in range(S):
        c0 = [0, 0]
        c1 = [0, 0]
        for s in range(S):
            if s == t:
                continue
            if yi[s] == 0:
                c0[yj[s]] += 1
            else:
                c1[yj[s]] += 1
        a = yi[t]
        bag = c0 if a == 0 else c1
        ntot = bag[0] + bag[1]
        if ntot == 0:
            pred = 0
        else:
            pred = 1 if bag[1] > bag[0] else 0
        if yj[t] != pred:
            wrong += 1
    return wrong / S


@dataclass
class TTAdaptiveDAGLocalPredictive(adlu.TTAdaptiveDAGLocalUnsupervised):
    """Local LOO prediction error vs immediate child only."""

    _old_eps: float = field(default=0.0, repr=False, compare=False)
    _new_eps: float = field(default=0.0, repr=False, compare=False)

    def _yi_yj_immediate_child(
        self,
        m: int,
        li: int,
        xs: Sequence[int],
        *,
        tri_flat: int = -1,
        tri_tt: int | None = None,
        tri_pin: int | None = None,
        tri_new_src: int | None = None,
    ) -> Tuple[int, int]:
        sig = self._subnet_signals(m, xs, tri_flat, tri_tt, tri_pin, tri_new_src)
        yi = sig[self.n_in + li]
        cj = self._child_gate_index(m, li)
        if cj is None:
            raise RuntimeError("predictive learner requires a downstream gate")
        return yi, sig[self.n_in + cj]

    def _compare_accept(self) -> bool:
        if self.plateau_escape:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return self._new_eps < self._old_eps or (
                self._new_eps == self._old_eps and rare
            )
        return self._new_eps < self._old_eps

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        self._gates_mutated_last_tick = False
        n_gates = self.n_gates_flat
        s = self.fsm

        if s == chain.FsmState.IDLE:
            self.fsm = chain.FsmState.INIT_UNIT

        elif s == chain.FsmState.INIT_UNIT:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            self.unit_sel = (self.lfsr & 0xFFFF) % n_gates
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rnd2 = self.lfsr & 0x3
            u = self.unit_sel
            if not (rnd2 < self.plastic[u] + 1):
                self.fsm = chain.FsmState.IDLE
            else:
                m, li = divmod(u, self.G)
                if self._child_gate_index(m, li) is None:
                    self.fsm = chain.FsmState.IDLE
                else:
                    self._opp_is_rewire = self._choose_rewire(u)
                    self.old_gate = self.gate[u]
                    if self._opp_is_rewire:
                        self.lfsr = chain.lfsr16_step(self.lfsr)
                        self._rewire_pin = self.lfsr & 1
                        if self._rewire_pin == 0:
                            self.old_src = self.ina[m][li]
                        else:
                            self.old_src = self.inb[m][li]
                    self.fsm = chain.FsmState.OLD_CLEAR

        elif s == chain.FsmState.OLD_CLEAR:
            S = min(self.n_local_samples, max(4, self.nrows))
            self._local_rows = []
            for _ in range(S):
                self.lfsr = chain.lfsr16_step(self.lfsr)
                self._local_rows.append((self.lfsr & 0xFFFF) % self.nrows)
            self._yi_old = []
            self._yc_old = []
            self.sample_idx = 0
            self.old_score = 0
            self.fsm = chain.FsmState.OLD_ACC

        elif s == chain.FsmState.OLD_ACC:
            r = self._local_rows[self.sample_idx]
            xs = chain.row_bits(r, self.n_in)
            m, li = divmod(self.unit_sel, self.G)
            yi, yc = self._yi_yj_immediate_child(m, li, xs)
            self._yi_old.append(yi)
            self._yc_old.append(yc)
            self.sample_idx += 1
            if self.sample_idx >= len(self._local_rows):
                self._old_eps = _prediction_error_loo(self._yi_old, self._yc_old)
                self.fsm = chain.FsmState.PROPOSE

        elif s == chain.FsmState.PROPOSE:
            u = self.unit_sel
            m, li = divmod(u, self.G)
            vmax = ad._valid_source_max(self.n_in, li)
            if self._opp_is_rewire:
                self.lfsr = chain.lfsr16_step(self.lfsr)
                ns = (self.lfsr & 0xFFFF) % (vmax + 1)
                if ns == self.old_src and vmax > 0:
                    ns = (ns + 1) % (vmax + 1)
                if (
                    self.require_distinct_inputs
                    and vmax > 0
                    and self._rewire_pin == 0
                    and ns == self.inb[m][li]
                ):
                    ns = (ns + 1) % (vmax + 1)
                    if ns == self.old_src:
                        ns = (ns + 1) % (vmax + 1)
                if (
                    self.require_distinct_inputs
                    and vmax > 0
                    and self._rewire_pin == 1
                    and ns == self.ina[m][li]
                ):
                    ns = (ns + 1) % (vmax + 1)
                    if ns == self.old_src:
                        ns = (ns + 1) % (vmax + 1)
                self.trial_src = ns
            else:
                self.trial_gate = self._propose_lut_nibble(u)
            self.fsm = chain.FsmState.NEW_CLEAR

        elif s == chain.FsmState.NEW_CLEAR:
            self.sample_idx = 0
            self._yi_new = []
            self._yc_new = []
            self.new_score = 0
            self.fsm = chain.FsmState.NEW_ACC

        elif s == chain.FsmState.NEW_ACC:
            r = self._local_rows[self.sample_idx]
            xs = chain.row_bits(r, self.n_in)
            m, li = divmod(self.unit_sel, self.G)
            u = self.unit_sel
            if self._opp_is_rewire:
                yi, yc = self._yi_yj_immediate_child(
                    m,
                    li,
                    xs,
                    tri_flat=u,
                    tri_tt=None,
                    tri_pin=self._rewire_pin,
                    tri_new_src=self.trial_src,
                )
            else:
                yi, yc = self._yi_yj_immediate_child(
                    m,
                    li,
                    xs,
                    tri_flat=u,
                    tri_tt=self.trial_gate,
                )
            self._yi_new.append(yi)
            self._yc_new.append(yc)
            self.sample_idx += 1
            if self.sample_idx >= len(self._local_rows):
                self._new_eps = _prediction_error_loo(self._yi_new, self._yc_new)
                self.fsm = chain.FsmState.COMPARE

        elif s == chain.FsmState.COMPARE:
            u = self.unit_sel
            accept = self._compare_accept()
            if accept:
                m, li = divmod(u, self.G)
                if self._opp_is_rewire:
                    if self._rewire_pin == 0:
                        self.ina[m][li] = self.trial_src
                    else:
                        self.inb[m][li] = self.trial_src
                else:
                    self.gate[u] = self.trial_gate
                self.plastic[u] = max(0, self.plastic[u] - 1)
                self._gates_mutated_last_tick = True
            else:
                self.plastic[u] = min(3, self.plastic[u] + 1)
            if not self._opp_is_rewire:
                self._after_lut_compare(u, accept)
            self._after_compare_hook()
            self.fsm = chain.FsmState.IDLE

        else:
            self.fsm = chain.FsmState.IDLE
