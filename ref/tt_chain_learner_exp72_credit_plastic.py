# SPDX-License-Identifier: Apache-2.0
"""
Experiment 10 — **credit-weighted selection + local plasticity / plateau** (``Exp72``).

Extends ``TTChainLearnerExp72CreditWeighted``: the same full-table ``c_good`` / ``c_bad``
counts are cached when ``u`` is chosen.

* **INIT:** probability of starting a trial is credit-graded: normalized ``c_bad`` maps to
  ``slots`` in ``{1,2,3,4}`` with ``P(start) = slots/4`` (low relative badness ≈ frozen).
  If all gates tie on ``c_bad``, falls back to ``plastic[u]+1`` like the base learner.

* **COMPARE (plateau ties):** per-gate mask from the same snapshot — high relative
  ``c_bad`` → small mask (more tie accepts / exploration); low → large mask (rarer ties).
"""

from __future__ import annotations

from dataclasses import dataclass

import tt_chain_learner_exp72_credit as e72c


@dataclass
class TTChainLearnerExp72CreditPlastic(e72c.TTChainLearnerExp72CreditWeighted):
    """Credit-weighted ``u`` plus credit-modulated INIT allow and plateau mask."""

    def _credit_init_slots(self, u: int) -> int:
        """How many values of ``rnd2`` in ``0..3`` allow a trial (1..4)."""
        cb = self._credit_c_bad[u]
        mx = max(self._credit_c_bad)
        mn = min(self._credit_c_bad)
        if mx == mn:
            return min(4, self.plastic[u] + 1)
        t = (cb - mn) / (mx - mn)
        return 1 + int(3 * t + 1e-12)

    def _plastic_trial_allowed(self, u: int, rnd2: int) -> bool:
        return rnd2 < self._credit_init_slots(u)

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        if len(getattr(self, "_credit_c_bad", ())) != len(self.gate):
            return self.plateau_mask
        mx = max(self._credit_c_bad)
        mn = min(self._credit_c_bad)
        if mx == mn:
            return self.plateau_mask
        t = (self._credit_c_bad[u] - mn) / (mx - mn)
        if t > 2.0 / 3.0:
            return 1
        if t > 1.0 / 3.0:
            return 3
        return 15
