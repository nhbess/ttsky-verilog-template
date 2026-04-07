# SPDX-License-Identifier: Apache-2.0
"""
Experiment 20 — reinforce **current** LUT type in proportion to $I(Z;Y)$ at each gate.

After clamp-relax, $Z$ is the gate’s relaxed output and $Y$ is the clamped subnet output
(``y_star`` for that row). Counts build a 2×2 joint; mutual information scores how predictive
$Z$ is of $Y$. ``p_gate`` decays then adds mass only on the **currently installed** LUT id,
weighted by that score. Refit step: with probability ``1/16``, pick a **uniform random**
LUT id for the chosen slot; otherwise ``argmax(p_gate)`` (or weighted sample). No pseudo-targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np

import tt_adaptive_dag_clamp_relax as adcr
import tt_chain_learner_spec as chain


@dataclass
class TTAdaptiveDAGLocalInfo(adcr.TTAdaptiveDAGClampRelaxFixed):
    """Fixed DAG; MI-weighted reinforcement of current gate type; refit with rare uniform exploration."""

    stats_ema_decay: float = 0.99
    lut_pick_eps: float = 1e-6
    lut_pick_alpha: float = 1.0
    argmax_lut: bool = True
    p_gate: Any = field(init=False, repr=False, compare=False)
    count_z: Any = field(init=False, repr=False, compare=False)
    count_zy: Any = field(init=False, repr=False, compare=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        ng = self.n_gates_flat
        self.p_gate = np.ones((ng, 16), dtype=np.float64) / 16.0
        self.count_z = np.zeros((ng, 2), dtype=np.float64)
        self.count_zy = np.zeros((ng, 2, 2), dtype=np.float64)
        self.fsm = chain.FsmState.IDLE

    def _info_score(self, u: int) -> float:
        p = self.count_zy[u].astype(np.float64) + 1e-6
        s = float(p.sum())
        if s <= 0:
            return 0.0
        p = p / s
        pz = p.sum(axis=1, keepdims=True)
        py = p.sum(axis=0, keepdims=True)
        prod = pz @ py
        prod = np.maximum(prod, 1e-12)
        p_safe = np.maximum(p, 1e-12)
        mi = float(np.sum(p * (np.log(p_safe) - np.log(prod))))
        return max(0.0, mi)

    def _update_stats_and_p_gate(
        self,
        rel: List[List[int]],
        y_star: Tuple[int, ...],
    ) -> None:
        G = self.G
        d = float(self.stats_ema_decay)
        inc0 = 1.0 - d
        for m in range(self.n_out):
            yf = int(y_star[m]) & 1
            for i in range(G):
                u = m * self.G + i
                z = int(rel[m][i]) & 1
                self.count_z[u, z] += 1.0
                self.count_zy[u, z, yf] += 1.0
                sc = self._info_score(u)
                gcur = int(self.gate[u])
                pg = self.p_gate[u]
                pg *= d
                pg[gcur] += inc0 * sc
                np.maximum(pg, 1e-6, out=pg)
                pg /= float(pg.sum())

    def _sample_lut_from_stats(self, u: int) -> int:
        row = self.p_gate[u]
        if self.argmax_lut:
            return int(np.argmax(row))
        w = np.power(row + self.lut_pick_eps, self.lut_pick_alpha)
        s = float(w.sum())
        if s <= 0:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            return (self.lfsr & 0xFFFF) % 16
        self.lfsr = chain.lfsr16_step(self.lfsr)
        r = (self.lfsr & 0xFFFF) / 65536.0 * s
        c = 0.0
        for g in range(16):
            c += float(w[g])
            if r < c:
                return g
        return 15

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        self._gates_mutated_last_tick = False
        self.lfsr = chain.lfsr16_step(self.lfsr)
        idx = (self.lfsr & 0xFFFF) % self.nrows
        xs = tuple(chain.row_bits(idx, self.n_in))
        y_star = tuple(self.target[mo][idx] for mo in range(self.n_out))
        rel, settle = self._relax_all(xs, y_star, -1, None, None, None)
        self._update_stats_and_p_gate(rel, y_star)

        self.lfsr = chain.lfsr16_step(self.lfsr)
        u = (self.lfsr & 0xFFFF) % self.n_gates_flat

        self.lfsr = chain.lfsr16_step(self.lfsr)
        if (self.lfsr & 0xF) == 0:
            g_new = (self.lfsr >> 4) & 0xF
        else:
            g_new = self._sample_lut_from_stats(u)
        old = self.gate[u]
        if g_new != old:
            self.gate[u] = g_new
            self._gates_mutated_last_tick = True

        e_list = self._per_gate_propagated_e(rel, xs, -1, None, None, None)
        self.exp16_last_E_local = float(sum(e_list))
        self.exp16_last_max_depth = self._max_depth_from_e_list(e_list)
        self.exp16_last_settle_steps = settle
        self.fsm = chain.FsmState.IDLE
