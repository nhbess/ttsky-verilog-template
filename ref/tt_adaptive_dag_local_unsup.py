# SPDX-License-Identifier: Apache-2.0
"""
Experiment 13 — **unsupervised local objective** on the adaptive DAG.

Accept/reject uses only statistics of a gate’s own output ``y_i`` and a **downstream**
signal: the first later gate in the same subnet that consumes wire ``(n_in+i)``, or the
subnet output if none. Over a fixed list of randomly chosen minterms (resampled each
trial), we maximize:

* **|corr(y_i, y_down)|** when a distinct downstream gate exists;
* otherwise **4 p(1-p)** with ``p = mean(y_i)`` (balance / marginal entropy proxy on the
  last gate).

No target bits enter COMPARE. ``target`` is still stored for **external**
``score_current_gates()`` only.

Rewire vs LUT trials use a simple coin flip (no global influence probe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import tt_chain_learner_spec as chain

import tt_adaptive_dag_learner as ad


def _pearson_abs_bits(yi: List[int], yc: List[int]) -> float:
    n = len(yi)
    if n < 4:
        return 0.0
    mx = sum(yi) / n
    my = sum(yc) / n
    vx = sum((float(yi[i]) - mx) ** 2 for i in range(n)) / n
    vy = sum((float(yc[i]) - my) ** 2 for i in range(n)) / n
    if vx < 1e-12 or vy < 1e-12:
        return 0.0
    cov = sum((float(yi[i]) - mx) * (float(yc[i]) - my) for i in range(n)) / n
    r = cov / (vx**0.5 * vy**0.5)
    return abs(r)


def _balance_score(yi: List[int]) -> float:
    if not yi:
        return 0.0
    p = sum(yi) / len(yi)
    return 4.0 * p * (1.0 - p)


@dataclass
class TTAdaptiveDAGLocalUnsupervised(ad.TTAdaptiveDAGLearner):
    """Same wiring rules as ``TTAdaptiveDAGLearner``; local-only COMPARE."""

    n_local_samples: int = 24

    _local_rows: List[int] = field(default_factory=list, repr=False)
    _yi_old: List[int] = field(default_factory=list, repr=False)
    _yc_old: List[int] = field(default_factory=list, repr=False)
    _yi_new: List[int] = field(default_factory=list, repr=False)
    _yc_new: List[int] = field(default_factory=list, repr=False)
    _use_corr: bool = field(default=False, repr=False, compare=False)
    _old_local_L: float = field(default=0.0, repr=False, compare=False)
    _new_local_L: float = field(default=0.0, repr=False, compare=False)

    def _child_gate_index(self, m: int, li: int) -> int | None:
        s = self.n_in + li
        for j in range(li + 1, self.G):
            if self.ina[m][j] == s or self.inb[m][j] == s:
                return j
        return None

    def _subnet_signals(
        self,
        m: int,
        xs: Sequence[int],
        tri_flat: int = -1,
        tri_tt: int | None = None,
        tri_pin: int | None = None,
        tri_new_src: int | None = None,
    ) -> List[int]:
        G = self.G
        sig: List[int] = [0] * (self.n_in + G)
        for j in range(self.n_in):
            sig[j] = xs[j]
        m_t, i_t = divmod(tri_flat, self.G) if tri_flat >= 0 else (-1, -1)
        for i in range(G):
            ia = self.ina[m][i]
            ib = self.inb[m][i]
            tt = self.gate[m * self.G + i]
            if m == m_t and i == i_t:
                if tri_tt is not None:
                    tt = tri_tt
                if tri_pin is not None and tri_new_src is not None:
                    if tri_pin == 0:
                        ia = tri_new_src
                    else:
                        ib = tri_new_src
            sig[self.n_in + i] = chain.gate_eval(tt, sig[ia], sig[ib])
        return sig

    def _yi_yc_pair(
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
        if cj is not None:
            yc = sig[self.n_in + cj]
        else:
            yc = sig[self.n_in + self.G - 1]
        return yi, yc

    def _local_L(self, yi: List[int], yc: List[int], use_corr: bool) -> float:
        if use_corr:
            return _pearson_abs_bits(yi, yc)
        return _balance_score(yi)

    def _choose_rewire(self, u: int) -> bool:
        self.lfsr = chain.lfsr16_step(self.lfsr)
        return (self.lfsr & 1) == 0

    def _compare_accept(self) -> bool:
        if self.plateau_escape:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return self._new_local_L > self._old_local_L or (
                self._new_local_L == self._old_local_L and rare
            )
        return self._new_local_L > self._old_local_L

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self._local_rows.clear()
        self._yi_old.clear()
        self._yc_old.clear()
        self._yi_new.clear()
        self._yc_new.clear()

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
                self._opp_is_rewire = self._choose_rewire(u)
                self.old_gate = self.gate[u]
                m, li = divmod(u, self.G)
                cj = self._child_gate_index(m, li)
                self._use_corr = cj is not None
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
            yi, yc = self._yi_yc_pair(m, li, xs)
            self._yi_old.append(yi)
            self._yc_old.append(yc)
            self.sample_idx += 1
            if self.sample_idx >= len(self._local_rows):
                self._old_local_L = self._local_L(self._yi_old, self._yc_old, self._use_corr)
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
                yi, yc = self._yi_yc_pair(
                    m,
                    li,
                    xs,
                    tri_flat=u,
                    tri_tt=None,
                    tri_pin=self._rewire_pin,
                    tri_new_src=self.trial_src,
                )
            else:
                yi, yc = self._yi_yc_pair(
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
                self._new_local_L = self._local_L(self._yi_new, self._yc_new, self._use_corr)
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
