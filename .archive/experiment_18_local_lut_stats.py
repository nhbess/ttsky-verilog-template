# SPDX-License-Identifier: Apache-2.0
"""
Experiment 18 — local fitting with a **distribution over 16 gate types** per physical gate.

Each slot ``u`` maintains ``p_gate[u]`` (simplex over LUT ids). For a few random gates per
tick, ``y_hat`` comes from a **flip test** (lower ``E_{local}`` vs baseline relaxed ``rel``);
other gates still use counterfactual ``y_u^*``. Pattern ``i_pat = (x_a<<1)|x_b``; EMA update reinforces gate types consistent
with that row: decay then add ``(1-decay)`` mass to all ``g`` with ``GATE_TABLE[g,i_pat]==y_u^*``
(no explicit penalty on mismatches). After each step, probabilities are floored at ``1e-6``
then renormalized so noisy ``y_u^*`` cannot permanently zero out hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple

import numpy as np

import tt_adaptive_dag_clamp_relax as adcr
import tt_chain_learner_spec as chain


def _build_gate_table() -> np.ndarray:
    """``GATE_TABLE[g, i_pat]`` = output of LUT id ``g`` for minterm index ``i_pat`` (same as ``gate_eval``)."""
    t = np.zeros((16, 4), dtype=np.uint8)
    for g in range(16):
        for i_pat in range(4):
            xa = (i_pat >> 1) & 1
            xb = i_pat & 1
            t[g, i_pat] = chain.gate_eval(g, xa, xb)
    return t


GATE_TABLE = _build_gate_table()


@dataclass
class TTAdaptiveDAGLocalLutStats(adcr.TTAdaptiveDAGClampRelaxFixed):
    """Fixed DAG; ``p_gate`` over 16 LUT types; mixed flip-test / counterfactual ``y_hat``; one refit/tick."""

    stats_ema_decay: float = 0.99
    lut_pick_eps: float = 1e-6
    lut_pick_alpha: float = 1.0
    argmax_lut: bool = True
    stats_flip_test_max: int = 3
    p_gate: Any = field(init=False, repr=False, compare=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        ng = self.n_gates_flat
        self.p_gate = np.ones((ng, 16), dtype=np.float64) / 16.0
        self.fsm = chain.FsmState.IDLE

    def _counterfactual_y_star(
        self,
        m: int,
        li: int,
        xs: Tuple[int, ...],
        rel_row: List[int],
        y_star_m: int,
    ) -> int:
        """Best hidden bit for downstream match to clamped output; last gate uses y_star_m."""
        G = self.G
        n_in = self.n_in
        if li == G - 1:
            return y_star_m
        best_err = 2
        best_bs: List[int] = []
        for b in (0, 1):
            sig: List[int] = [0] * (n_in + G)
            for j in range(n_in):
                sig[j] = xs[j]
            for k in range(li):
                sig[n_in + k] = rel_row[k]
            sig[n_in + li] = b
            for k in range(li + 1, G):
                uf = m * self.G + k
                ia, ib = self.ina[m][k], self.inb[m][k]
                tt = self.gate[uf]
                sig[n_in + k] = chain.gate_eval(tt, sig[ia], sig[ib])
            err = 1 if sig[n_in + G - 1] != y_star_m else 0
            if err < best_err:
                best_err = err
                best_bs = [b]
            elif err == best_err:
                best_bs.append(b)
        if len(best_bs) == 1:
            return best_bs[0]
        if rel_row[li] in best_bs:
            return rel_row[li]
        self.lfsr = chain.lfsr16_step(self.lfsr)
        return best_bs[(self.lfsr & 1) % len(best_bs)]

    def _pick_flip_test_subset(self) -> Set[Tuple[int, int]]:
        """1..stats_flip_test_max distinct (subnet, gate_index) pairs for flip-target stats."""
        n_g = self.n_gates_flat
        if self.stats_flip_test_max <= 0:
            return set()
        cap = min(self.stats_flip_test_max, n_g)
        self.lfsr = chain.lfsr16_step(self.lfsr)
        n_pick = 1 + (self.lfsr % cap)
        n_pick = min(n_pick, n_g)
        out: Set[Tuple[int, int]] = set()
        while len(out) < n_pick:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            u = (self.lfsr & 0xFFFF) % n_g
            m, i = divmod(u, self.G)
            out.add((m, i))
        return out

    def _update_lut_stats_from_rel(
        self,
        xs: Tuple[int, ...],
        rel: List[List[int]],
        y_star: Tuple[int, ...],
    ) -> None:
        G = self.G
        n_in = self.n_in
        d = float(self.stats_ema_decay)
        inc = 1.0 - d
        gt = GATE_TABLE
        e_base = float(
            sum(self._per_gate_propagated_e(rel, xs, -1, None, None, None))
        )
        flip_subset = self._pick_flip_test_subset()
        for m in range(self.n_out):
            sig: List[int] = [0] * (n_in + G)
            for j in range(n_in):
                sig[j] = xs[j]
            for j in range(G):
                sig[n_in + j] = rel[m][j]
            ysm = y_star[m]
            for i in range(G):
                u = m * self.G + i
                xa = sig[self.ina[m][i]] & 1
                xb = sig[self.inb[m][i]] & 1
                i_pat = (xa << 1) | xb
                if (m, i) in flip_subset:
                    y_cur = rel[m][i] & 1
                    rel_flip = [list(row) for row in rel]
                    rel_flip[m][i] = 1 - y_cur
                    e_flip = float(
                        sum(
                            self._per_gate_propagated_e(
                                rel_flip, xs, -1, None, None, None
                            )
                        )
                    )
                    if e_flip < e_base:
                        y_hat = 1 - y_cur
                    elif e_flip > e_base:
                        y_hat = y_cur
                    else:
                        continue
                else:
                    y_hat = self._counterfactual_y_star(m, i, xs, rel[m], ysm)
                pg = self.p_gate[u]
                pg *= d
                yh = int(y_hat)
                for g in range(16):
                    if int(gt[g, i_pat]) == yh:
                        pg[g] += inc
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
        self._update_lut_stats_from_rel(xs, rel, y_star)

        self.lfsr = chain.lfsr16_step(self.lfsr)
        u = (self.lfsr & 0xFFFF) % self.n_gates_flat
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
