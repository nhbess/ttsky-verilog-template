# SPDX-License-Identifier: Apache-2.0
"""
Experiment 9 — **local credit for gate selection** on ``TTChainLearnerExp72``.

Over a full truth-table pass, each gate accumulates how often it lies on an output slice
for a **correct** vs **incorrect** row (same row minterm for that output line). Selection
samples the gate index ``u`` with probability proportional to a weight derived from those
counts; mutation and global accept/reject are unchanged from the base learner.

* ``credit_mode="bad"``: weight ``c_bad + 1`` (always at least uniform-like mass).
* ``credit_mode="delta"``: weight ``max(1, c_bad - c_good + 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import tt_chain_learner_spec as chain

import tt_chain_learner_exp72 as e72


@dataclass
class TTChainLearnerExp72CreditWeighted(e72.TTChainLearnerExp72):
    """Same topology, mutation, and global COMPARE as ``TTChainLearnerExp72``; weighted ``u``."""

    credit_mode: str = "bad"  # "bad" | "delta"

    def _gate_good_bad_counts(self) -> Tuple[List[int], List[int]]:
        n_gates = len(self.gate)
        c_good = [0] * n_gates
        c_bad = [0] * n_gates
        gpo = self.gpo
        for idx in range(self.nrows):
            xs = chain.row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            for m in range(self.n_out):
                ok = ys[m] == self.target[m][idx]
                base = m * gpo
                for off in range(gpo):
                    gi = base + off
                    if ok:
                        c_good[gi] += 1
                    else:
                        c_bad[gi] += 1
        return c_good, c_bad

    def _credit_weights_for_trial(self, n_gates: int) -> List[int]:
        """Refresh ``_credit_c_{good,bad}`` and return selection weights."""
        self._credit_c_good, self._credit_c_bad = self._gate_good_bad_counts()
        cg, cb = self._credit_c_good, self._credit_c_bad
        if self.credit_mode == "bad":
            return [cb[i] + 1 for i in range(n_gates)]
        if self.credit_mode == "delta":
            return [max(1, cb[i] - cg[i] + 1) for i in range(n_gates)]
        raise ValueError(f"unknown credit_mode {self.credit_mode!r}")

    def _sample_unit_for_trial(self, n_gates: int) -> int:
        w = self._credit_weights_for_trial(n_gates)
        total = sum(w)
        if total <= 0:
            return super()._sample_unit_for_trial(n_gates)
        self.lfsr = chain.lfsr16_step(self.lfsr)
        r = (self.lfsr & 0xFFFF) % total
        acc = 0
        for i in range(n_gates):
            acc += w[i]
            if r < acc:
                return i
        return n_gates - 1
