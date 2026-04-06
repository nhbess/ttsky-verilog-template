# SPDX-License-Identifier: Apache-2.0
"""
Experiment 11 — **local output-line error** for COMPARE (topology ``TTChainLearnerExp72``).

Accept/reject uses only the subnet that contains the mutated gate: let ``m = u // gpo`` and

  ε_m = (1 / 2^n) ∑_x  1[y_m(x) ≠ target_m(x)]

A trial is accepted if ``ε_m`` **decreases** (strict), or on a plateau tie with the same
rare-event rule as the global learner (``plateau_escape`` + ``_plateau_mask_for_compare``).

Global OLD_ACC / NEW_ACC still accumulate full multi-output scores for logging continuity;
``_compare_accept`` ignores them when using local rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import tt_chain_learner_spec as chain

import tt_chain_learner_exp72 as e72
import tt_chain_learner_exp72_credit as e72c


class TTChainLearnerExp72LocalEpsilonMixin:
    """Shared local ε acceptance + helpers (multiple inheritance with Exp72 / CreditWeighted)."""

    def _wrong_fraction_output_line(
        self, m: int, trial_u: int | None, trial_tt: int | None
    ) -> float:
        wrong = 0
        for idx in range(self.nrows):
            xs = chain.row_bits(idx, self.n_in)
            if trial_u is None:
                ys = self.forward_all(xs)
            else:
                self._trial_overrides = {trial_u: trial_tt}
                ys = self._forward_trial_all(xs)
                self._trial_overrides = None
            if ys[m] != self.target[m][idx]:
                wrong += 1
        return wrong / self.nrows

    def _compare_accept(self) -> bool:
        u = self.unit_sel
        m = u // self.gpo
        e_old = self._wrong_fraction_output_line(m, None, None)
        e_new = self._wrong_fraction_output_line(m, u, self.trial_gate)
        self._last_eps_old = e_old
        self._last_eps_new = e_new
        self._last_eps_delta = e_new - e_old
        if self.plateau_escape:
            self.lfsr = chain.lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return e_new < e_old or (e_new == e_old and rare)
        return e_new < e_old


@dataclass
class TTChainLearnerExp72LocalUniform(e72.TTChainLearnerExp72, TTChainLearnerExp72LocalEpsilonMixin):
    """Uniform gate pick; local ε acceptance."""


@dataclass
class TTChainLearnerExp72LocalCredit(
    e72c.TTChainLearnerExp72CreditWeighted, TTChainLearnerExp72LocalEpsilonMixin
):
    """Credit-weighted gate pick (experiment 9); local ε acceptance."""
