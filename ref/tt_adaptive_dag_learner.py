# SPDX-License-Identifier: Apache-2.0
"""
Experiment 12 — **adaptive DAG connectivity** per output subnet.

Each output line uses ``G`` gates in fixed topological order. Gate ``i`` reads two signals
from indices ``0 .. n_in + i - 1`` (primary inputs and earlier gates in the same subnet).
``ina[m][i]`` and ``inb[m][i]`` are learnable; LUT nibbles live in a flat ``gate`` RAM like
``TTChainLearner``.

Training follows the same FSM as ``TTChainLearner``; each trial either proposes a new LUT
nibble **or** rewires one input pin. Rewire probability increases when a coarse **influence**
proxy for that gate is low (few rows where an XOR tweak to the LUT changes the subnet
output).

Global score acceptance / plateau ties match the default chain learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import tt_chain_learner_spec as chain


def _valid_source_max(n_in: int, local_i: int) -> int:
    return n_in + local_i - 1


@dataclass
class TTAdaptiveDAGLearner:
    n_in: int = 4
    n_out: int = 2
    G: int = 7
    target: List[List[int]] = field(default_factory=list)
    gate: List[int] = field(default_factory=list)
    ina: List[List[int]] = field(default_factory=list)
    inb: List[List[int]] = field(default_factory=list)
    plastic: List[int] = field(default_factory=list)
    unit_sel: int = 0
    sample_idx: int = 0
    old_score: int = 0
    new_score: int = 0
    old_gate: int = 0
    trial_gate: int = 0
    old_src: int = 0
    trial_src: int = 0
    lfsr: int = 0xACE1
    _gates_mutated_last_tick: bool = field(default=False, repr=False, compare=False)
    fsm: chain.FsmState = chain.FsmState.IDLE
    plateau_escape: bool = True
    plateau_mask: int = 3
    require_distinct_inputs: bool = True
    rewire_prob_floor: float = 0.22
    rewire_prob_ceil: float = 0.88

    _opp_is_rewire: bool = field(default=False, repr=False, compare=False)
    _rewire_pin: int = field(default=0, repr=False, compare=False)

    @property
    def nrows(self) -> int:
        return 1 << self.n_in

    @property
    def max_score(self) -> int:
        return self.nrows * self.n_out

    @property
    def n_gates_flat(self) -> int:
        return self.n_out * self.G

    def __post_init__(self) -> None:
        if not self.target:
            self.target = chain.make_truth_tables("parity", self.n_in, self.n_out, table_seed=0)
        ng = self.n_gates_flat
        if len(self.gate) != ng:
            self.gate = [0] * ng
        if len(self.plastic) != ng:
            self.plastic = [3] * ng
        if not self.ina or not self.inb:
            self.ina = [[0] * self.G for _ in range(self.n_out)]
            self.inb = [[0] * self.G for _ in range(self.n_out)]

    def reset(self, seed: int = 0xACE1) -> None:
        self.lfsr = seed & 0xFFFF or 0xACE1
        ng = self.n_gates_flat
        self.gate = [0] * ng
        self.plastic = [3] * ng
        self.ina = [[0] * self.G for _ in range(self.n_out)]
        self.inb = [[0] * self.G for _ in range(self.n_out)]
        for m in range(self.n_out):
            for i in range(self.G):
                vmax = _valid_source_max(self.n_in, i)
                self.lfsr = chain.lfsr16_step(self.lfsr)
                self.ina[m][i] = (self.lfsr & 0xFFFF) % (vmax + 1)
                self.lfsr = chain.lfsr16_step(self.lfsr)
                self.inb[m][i] = (self.lfsr & 0xFFFF) % (vmax + 1)
                if self.require_distinct_inputs and vmax > 0:
                    guard = 0
                    while self.ina[m][i] == self.inb[m][i] and guard < vmax + 3:
                        self.inb[m][i] = (self.inb[m][i] + 1) % (vmax + 1)
                        guard += 1
        for u in range(ng):
            self.lfsr = chain.lfsr16_step(self.lfsr)
            self.gate[u] = self.lfsr & 0xF
        self.unit_sel = 0
        self.sample_idx = 0
        self.fsm = chain.FsmState.IDLE
        self._opp_is_rewire = False

    def _subnet_output(
        self,
        m: int,
        xs: Sequence[int],
        *,
        tri_flat: int = -1,
        tri_tt: int | None = None,
        tri_pin: int | None = None,
        tri_new_src: int | None = None,
    ) -> int:
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
        return sig[self.n_in + G - 1]

    def forward_all(self, xs: Sequence[int]) -> Tuple[int, ...]:
        return tuple(self._subnet_output(m, xs) for m in range(self.n_out))

    def _forward_trial_all(self, xs: Sequence[int]) -> Tuple[int, ...]:
        u = self.unit_sel
        if self._opp_is_rewire:
            return tuple(
                self._subnet_output(
                    m,
                    xs,
                    tri_flat=u,
                    tri_pin=self._rewire_pin,
                    tri_new_src=self.trial_src,
                )
                for m in range(self.n_out)
            )
        return tuple(
            self._subnet_output(m, xs, tri_flat=u, tri_tt=self.trial_gate) for m in range(self.n_out)
        )

    def score_current_gates(self) -> int:
        s = 0
        for idx in range(self.nrows):
            xs = chain.row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            for m in range(self.n_out):
                if ys[m] == self.target[m][idx]:
                    s += 1
        return s

    def _rough_influence_flat(self, u: int) -> float:
        m, li = divmod(u, self.G)
        base_tt = self.gate[u]
        hits = 0
        for idx in range(self.nrows):
            xs = chain.row_bits(idx, self.n_in)
            y0 = self._subnet_output(m, xs)
            touched = False
            for alt in (1, 2, 4, 8, 3, 5):
                nt = base_tt ^ alt
                if nt == base_tt:
                    continue
                y1 = self._subnet_output(m, xs, tri_flat=u, tri_tt=nt)
                if y1 != y0:
                    touched = True
                    break
            if touched:
                hits += 1
        return hits / self.nrows

    def _choose_rewire(self, u: int) -> bool:
        infl = self._rough_influence_flat(u)
        span = self.rewire_prob_ceil - self.rewire_prob_floor
        p_rw = self.rewire_prob_floor + span * max(0.0, min(1.0, 1.0 - infl))
        self.lfsr = chain.lfsr16_step(self.lfsr)
        thresh = int(65536 * p_rw) & 0xFFFF
        return (self.lfsr & 0xFFFF) < thresh

    def _plateau_mask_for_compare(self) -> int:
        return self.plateau_mask

    def _after_compare_hook(self) -> None:
        pass

    def _propose_lut_nibble(self, u: int) -> int:
        """Proposed 4-bit LUT index for a non-rewire trial (override e.g. softmax in Exp15)."""
        self.lfsr = chain.lfsr16_step(self.lfsr)
        cand = self.lfsr & 0xF
        if cand == self.old_gate:
            cand ^= 0x1
        return cand

    def _after_lut_compare(self, u: int, accepted: bool) -> None:
        """Hook after COMPARE for LUT trials (override for Exp15 logits update)."""
        pass

    def _compare_accept(self) -> bool:
        if self.plateau_escape:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return self.new_score > self.old_score or (
                self.new_score == self.old_score and rare
            )
        return self.new_score > self.old_score

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
                if self._opp_is_rewire:
                    self.lfsr = chain.lfsr16_step(self.lfsr)
                    self._rewire_pin = self.lfsr & 1
                    if self._rewire_pin == 0:
                        self.old_src = self.ina[m][li]
                    else:
                        self.old_src = self.inb[m][li]
                self.fsm = chain.FsmState.OLD_CLEAR

        elif s == chain.FsmState.OLD_CLEAR:
            self.sample_idx = 0
            self.old_score = 0
            self.fsm = chain.FsmState.OLD_ACC

        elif s == chain.FsmState.OLD_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            mo = self.sample_idx // self.nrows
            xs = chain.row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            if ys[mo] == self.target[mo][idx]:
                self.old_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
                self.fsm = chain.FsmState.PROPOSE

        elif s == chain.FsmState.PROPOSE:
            u = self.unit_sel
            m, li = divmod(u, self.G)
            vmax = _valid_source_max(self.n_in, li)
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
            self.new_score = 0
            self.fsm = chain.FsmState.NEW_ACC

        elif s == chain.FsmState.NEW_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            mo = self.sample_idx // self.nrows
            xs = chain.row_bits(idx, self.n_in)
            ys = self._forward_trial_all(xs)
            if ys[mo] == self.target[mo][idx]:
                self.new_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
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

    def run_until_perfect(self, max_ticks: int) -> Tuple[bool, int]:
        if self.score_current_gates() == self.max_score:
            return True, 0
        for t in range(max_ticks):
            self.tick()
            if self._gates_mutated_last_tick and self.score_current_gates() == self.max_score:
                return True, t + 1
        return self.score_current_gates() == self.max_score, max_ticks
