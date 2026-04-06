# SPDX-License-Identifier: Apache-2.0
"""
Experiment 17 — same clamp + local E_local acceptance as Exp16, but gate selection is
error-biased: sample u with P(u) ∝ (e_u + eps)^alpha instead of uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tt_adaptive_dag_clamp_relax as adcr
import tt_adaptive_dag_learner as ad
import tt_chain_learner_spec as chain


@dataclass
class TTAdaptiveDAGClampRelaxErrPick(adcr.TTAdaptiveDAGClampRelax):
    """Exp16 objective; INIT samples clamp first, then u ~ Categorical((e_u + eps)^alpha)."""

    err_pick_alpha: float = 1.0
    err_pick_eps: float = 1e-6

    def _sample_unit_from_e(self, e_list: List[float]) -> int:
        ng = len(e_list)
        w = [(e_list[i] + self.err_pick_eps) ** self.err_pick_alpha for i in range(ng)]
        s = sum(w)
        if s <= 0:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            return (self.lfsr & 0xFFFF) % ng
        self.lfsr = chain.lfsr16_step(self.lfsr)
        r = (self.lfsr & 0xFFFF) / 65536.0 * s
        c = 0.0
        for i, wi in enumerate(w):
            c += wi
            if r < c:
                return i
        return ng - 1

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        self._gates_mutated_last_tick = False
        n_gates = self.n_gates_flat
        s = self.fsm

        if s == chain.FsmState.IDLE:
            self.fsm = chain.FsmState.INIT_UNIT

        elif s == chain.FsmState.INIT_UNIT:
            e_list = self._baseline_after_clamp_pick()
            self.unit_sel = self._sample_unit_from_e(e_list)
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
                self.fsm = chain.FsmState.PROPOSE

        elif s == chain.FsmState.OLD_CLEAR:
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
class TTAdaptiveDAGClampRelaxFixedErrPick(TTAdaptiveDAGClampRelaxErrPick):
    """Error-picked gate; LUT-only (no rewires)."""

    allow_rewire: bool = False
