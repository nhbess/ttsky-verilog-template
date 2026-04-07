# SPDX-License-Identifier: Apache-2.0
"""Chain learner variants for experiment 7 (importable from worker + script)."""

from __future__ import annotations

from typing import Any, List, Tuple

import tt_chain_learner_spec as chain


class PlasticMappedPlateauChain(chain.TTChainLearner):
    """Higher plasticity → smaller mask → higher p (more tie exploration)."""

    _PLASTIC_MASK: Tuple[int, int, int, int] = (15, 7, 3, 1)

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        pi = min(3, max(0, int(self.plastic[u])))
        return self._PLASTIC_MASK[pi]


class RecentSuccessExploitPlateauChain(chain.TTChainLearner):
    """
    Gates that recently produced an accepted strict improvement use low p (mask 15);
    others use mask 3. Counts global COMPARE index since last such event on that gate.
    """

    def __init__(self, *, exploit_window: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exploit_window = exploit_window
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve: List[int] = [-1_000_000] * n

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def _after_compare_hook(self) -> None:
        self._compare_n += 1
        u = self.unit_sel
        if self._gates_mutated_last_tick and self.new_score > self.old_score:
            self._last_accept_improve[u] = self._compare_n

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        if self._compare_n - self._last_accept_improve[u] <= self.exploit_window:
            return 15
        return 3


class MixedPlasticSuccessPlateauChain(chain.TTChainLearner):
    """max(plastic mask, success mask): exploit if plastic is calm OR recent win on gate."""

    _PLASTIC_MASK: Tuple[int, int, int, int] = (15, 7, 3, 1)

    def __init__(self, *, exploit_window: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exploit_window = exploit_window
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def _after_compare_hook(self) -> None:
        self._compare_n += 1
        u = self.unit_sel
        if self._gates_mutated_last_tick and self.new_score > self.old_score:
            self._last_accept_improve[u] = self._compare_n

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        pi = min(3, max(0, int(self.plastic[u])))
        mp = self._PLASTIC_MASK[pi]
        ms = 15 if self._compare_n - self._last_accept_improve[u] <= self.exploit_window else 3
        return max(mp, ms)
