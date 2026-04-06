# SPDX-License-Identifier: Apache-2.0
"""
Experiment 8 — **multi-gate mutation** on top of ``TTChainLearnerExp72``.

Each training cycle proposes new truth-table nibble values for **B** distinct gates at once,
accumulates old/new global scores over the full truth table, then accepts or rejects the
**joint** move (same plateau tie rule as the base learner).

``mutation_batch <= 1`` delegates to the parent ``tick`` so behavior matches single-site
updates bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import tt_chain_learner_spec as chain

import tt_chain_learner_exp72 as e72


@dataclass
class TTChainLearnerExp72Multi(e72.TTChainLearnerExp72):
    """Same topology and scoring as ``TTChainLearnerExp72``; batched gate proposals."""

    mutation_batch: int = 2

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self._trial_overrides = None
        self._multi_units: list[int] = []

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        if self.mutation_batch <= 1:
            super().tick(train_enable)
            return

        self._gates_mutated_last_tick = False
        n_gates = len(self.gate)
        s = self.fsm
        B = min(self.mutation_batch, n_gates)

        if s == chain.FsmState.IDLE:
            self.fsm = chain.FsmState.INIT_UNIT

        elif s == chain.FsmState.INIT_UNIT:
            order = list(range(n_gates))
            for i in range(n_gates - 1, 0, -1):
                self.lfsr = chain.lfsr16_step(self.lfsr)
                j = (self.lfsr & 0xFFFF) % (i + 1)
                order[i], order[j] = order[j], order[i]
            cand = order[:B]
            blocked = False
            for u in cand:
                self.lfsr = chain.lfsr16_step(self.lfsr)
                rnd2 = self.lfsr & 0x3
                if not (rnd2 < self.plastic[u] + 1):
                    blocked = True
                    break
            if blocked:
                self.fsm = chain.FsmState.IDLE
            else:
                self._multi_units = list(cand)
                self.unit_sel = self._multi_units[0]
                self._trial_overrides = None
                self.fsm = chain.FsmState.OLD_CLEAR

        elif s == chain.FsmState.OLD_CLEAR:
            self.sample_idx = 0
            self.old_score = 0
            self.fsm = chain.FsmState.OLD_ACC

        elif s == chain.FsmState.OLD_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            m = self.sample_idx // self.nrows
            xs = chain.row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            if ys[m] == self.target[m][idx]:
                self.old_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
                self.fsm = chain.FsmState.PROPOSE

        elif s == chain.FsmState.PROPOSE:
            self._trial_overrides = {}
            for u in self._multi_units:
                self.lfsr = chain.lfsr16_step(self.lfsr)
                cand = self.lfsr & 0xF
                og = self.gate[u]
                if cand == og:
                    cand ^= 0x1
                self._trial_overrides[u] = cand
            self.fsm = chain.FsmState.NEW_CLEAR

        elif s == chain.FsmState.NEW_CLEAR:
            self.sample_idx = 0
            self.new_score = 0
            self.fsm = chain.FsmState.NEW_ACC

        elif s == chain.FsmState.NEW_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            m = self.sample_idx // self.nrows
            xs = chain.row_bits(idx, self.n_in)
            ys = self._forward_trial_all(xs)
            if ys[m] == self.target[m][idx]:
                self.new_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
                self.fsm = chain.FsmState.COMPARE

        elif s == chain.FsmState.COMPARE:
            self.unit_sel = self._multi_units[0]
            if self.plateau_escape:
                self.lfsr = chain.lfsr16_step(self.lfsr)
                rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
                accept = self.new_score > self.old_score or (
                    self.new_score == self.old_score and rare
                )
            else:
                accept = self.new_score > self.old_score
            if accept:
                assert self._trial_overrides is not None
                for u in self._multi_units:
                    self.gate[u] = self._trial_overrides[u]
                    self.plastic[u] = max(0, self.plastic[u] - 1)
                self._gates_mutated_last_tick = True
            else:
                for u in self._multi_units:
                    self.plastic[u] = min(3, self.plastic[u] + 1)
            self._after_compare_hook()
            self._trial_overrides = None
            self.fsm = chain.FsmState.IDLE

        else:
            self.fsm = chain.FsmState.IDLE
