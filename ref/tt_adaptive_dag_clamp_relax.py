# SPDX-License-Identifier: Apache-2.0
"""
Clamped I/O relaxation with local propagated error (Experiment 16).

For a sampled row (x, y*): forward each subnet, clamp the subnet output to y*, then
measure per-gate mismatch m_i = 1[rel_i != f_i(parent rel)] and local propagated error
e_i = sum_{j in children(i)} m_j (output gate: e = m_out). Accept mutations when the
scalar E_local = sum_i e_i decreases (training never uses global truth-table score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import tt_adaptive_dag_learner as ad
import tt_chain_learner_spec as chain


@dataclass
class TTAdaptiveDAGClampRelax(ad.TTAdaptiveDAGLearner):
    """Adaptive-DAG topology with local E_local acceptance and optional rewires."""

    relax_steps: int = 8
    allow_rewire: bool = True
    _e_local_old: float = field(default=0.0, repr=False, compare=False)
    _e_local_new: float = field(default=0.0, repr=False, compare=False)
    _clamp_idx: int = field(default=0, repr=False, compare=False)
    _clamp_xs: Tuple[int, ...] = field(default_factory=tuple, repr=False, compare=False)
    _clamp_y: Tuple[int, ...] = field(default_factory=tuple, repr=False, compare=False)
    exp16_last_E_local: float = field(default=0.0, repr=False, compare=False)
    exp16_last_max_depth: int = field(default=0, repr=False, compare=False)
    exp16_last_settle_steps: int = field(default=0, repr=False, compare=False)

    def _choose_rewire(self, u: int) -> bool:
        if not self.allow_rewire:
            return False
        return super()._choose_rewire(u)

    def _relax_subnet_rel(
        self,
        m: int,
        xs: Sequence[int],
        y_star_m: int,
        tri_flat: int,
        tri_tt: int | None,
        tri_pin: int | None,
        tri_src: int | None,
    ) -> List[int]:
        G = self.G
        n_in = self.n_in
        sig: List[int] = [0] * (n_in + G)
        for j in range(n_in):
            sig[j] = xs[j]
        m_t, i_t = divmod(tri_flat, self.G) if tri_flat >= 0 else (-1, -1)
        for i in range(G):
            ia, ib = self.ina[m][i], self.inb[m][i]
            tt = self.gate[m * self.G + i]
            if m == m_t and i == i_t:
                if tri_tt is not None:
                    tt = tri_tt
                if tri_pin is not None and tri_src is not None:
                    if tri_pin == 0:
                        ia = tri_src
                    else:
                        ib = tri_src
            sig[n_in + i] = chain.gate_eval(tt, sig[ia], sig[ib])
        sig[n_in + G - 1] = y_star_m
        return sig[n_in : n_in + G]

    def _relax_all(
        self,
        xs: Sequence[int],
        y_star: Tuple[int, ...],
        tri_flat: int = -1,
        tri_tt: int | None = None,
        tri_pin: int | None = None,
        tri_src: int | None = None,
    ) -> Tuple[List[List[int]], int]:
        rel_prev: List[List[int]] | None = None
        settle = self.relax_steps
        for r in range(self.relax_steps):
            rel: List[List[int]] = []
            for m in range(self.n_out):
                rel.append(
                    self._relax_subnet_rel(
                        m, xs, y_star[m], tri_flat, tri_tt, tri_pin, tri_src
                    )
                )
            if rel_prev is not None and rel_prev == rel:
                settle = r + 1
                break
            rel_prev = rel
        assert rel_prev is not None
        return rel_prev, settle

    def _dist_from_output(self, m: int) -> List[int]:
        G = self.G
        dist = [10**9] * G
        dist[G - 1] = 0
        for _ in range(G + 2):
            changed = False
            for li in range(G - 2, -1, -1):
                best = 10**9
                for cj in range(li + 1, G):
                    s = self.n_in + li
                    if self.ina[m][cj] == s or self.inb[m][cj] == s:
                        best = min(best, 1 + dist[cj])
                if best < dist[li]:
                    dist[li] = best
                    changed = True
            if not changed:
                break
        return dist

    def _local_propagated_metrics(
        self,
        rel: List[List[int]],
        xs: Sequence[int],
        tri_flat: int,
        tri_tt: int | None,
        tri_pin: int | None,
        tri_src: int | None,
    ) -> Tuple[float, int]:
        G = self.G
        n_in = self.n_in
        m_mismatch = [0] * self.n_gates_flat
        for m in range(self.n_out):
            sig: List[int] = [0] * (n_in + G)
            for j in range(n_in):
                sig[j] = xs[j]
            for i in range(G):
                sig[n_in + i] = rel[m][i]
            m_t, i_t = divmod(tri_flat, self.G) if tri_flat >= 0 else (-1, -1)
            for i in range(G):
                u = m * self.G + i
                ia, ib = self.ina[m][i], self.inb[m][i]
                tt = self.gate[u]
                if m == m_t and i == i_t:
                    if tri_tt is not None:
                        tt = tri_tt
                    if tri_pin is not None and tri_src is not None:
                        if tri_pin == 0:
                            ia = tri_src
                        else:
                            ib = tri_src
                calc = chain.gate_eval(tt, sig[ia], sig[ib])
                m_mismatch[u] = 1 if rel[m][i] != calc else 0

        d_out = [self._dist_from_output(m) for m in range(self.n_out)]
        e_sum = 0.0
        max_depth = 0
        for m in range(self.n_out):
            for li in range(G):
                u = m * self.G + li
                if li == G - 1:
                    e_u = float(m_mismatch[u])
                else:
                    ch_sum = 0
                    s_lin = n_in + li
                    for j in range(li + 1, G):
                        if self.ina[m][j] == s_lin or self.inb[m][j] == s_lin:
                            ch_sum += m_mismatch[m * self.G + j]
                    e_u = float(ch_sum)
                e_sum += e_u
                if e_u > 0:
                    max_depth = max(max_depth, d_out[m][li])
        return e_sum, max_depth

    def _relax_E(
        self,
        xs: Sequence[int],
        y_star: Tuple[int, ...],
        tri_flat: int = -1,
        tri_tt: int | None = None,
        tri_pin: int | None = None,
        tri_src: int | None = None,
    ) -> Tuple[float, int, int]:
        rel, settle = self._relax_all(xs, y_star, tri_flat, tri_tt, tri_pin, tri_src)
        e_loc, mx = self._local_propagated_metrics(rel, xs, tri_flat, tri_tt, tri_pin, tri_src)
        return e_loc, mx, settle

    def _clamp_compare_accept(self) -> bool:
        if self._e_local_new < self._e_local_old:
            return True
        if self.plateau_escape and self._e_local_new == self._e_local_old:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return rare
        return False

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
            self.lfsr = chain.lfsr16_step(self.lfsr)
            self._clamp_idx = (self.lfsr & 0xFFFF) % self.nrows
            self._clamp_xs = tuple(chain.row_bits(self._clamp_idx, self.n_in))
            self._clamp_y = tuple(self.target[mo][self._clamp_idx] for mo in range(self.n_out))
            self._e_local_old, d0, s0 = self._relax_E(self._clamp_xs, self._clamp_y)
            self.exp16_last_E_local = self._e_local_old
            self.exp16_last_max_depth = d0
            self.exp16_last_settle_steps = s0
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
            if self._opp_is_rewire:
                self._e_local_new, _, _ = self._relax_E(
                    self._clamp_xs,
                    self._clamp_y,
                    tri_flat=self.unit_sel,
                    tri_tt=None,
                    tri_pin=self._rewire_pin,
                    tri_src=self.trial_src,
                )
            else:
                self._e_local_new, _, _ = self._relax_E(
                    self._clamp_xs,
                    self._clamp_y,
                    tri_flat=self.unit_sel,
                    tri_tt=self.trial_gate,
                    tri_pin=None,
                    tri_src=None,
                )
            self.fsm = chain.FsmState.COMPARE

        elif s == chain.FsmState.COMPARE:
            u = self.unit_sel
            accept = self._clamp_compare_accept()
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


@dataclass
class TTAdaptiveDAGClampRelaxFixed(TTAdaptiveDAGClampRelax):
    """Local clamp relaxation with topology frozen (LUT-only mutations)."""

    allow_rewire: bool = False
