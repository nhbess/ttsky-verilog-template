# SPDX-License-Identifier: Apache-2.0
"""Experiment 12 instrumented learners (COMPARE event log for σ_eff and activity)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import tt_chain_learner_exp72_credit as e72c
import tt_adaptive_dag_learner as ad


@dataclass
class Exp12CreditBaseline(e72c.TTChainLearnerExp72CreditWeighted):
    """Topology A Exp72 + credit-weighted ``u``; logs each COMPARE."""

    exp12_log: List[Tuple[int, bool, bool]] = field(default_factory=list, repr=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.exp12_log.clear()

    def _after_compare_hook(self) -> None:
        self.exp12_log.append((self.unit_sel, self._gates_mutated_last_tick, False))


@dataclass
class Exp12AdaptiveDAG(ad.TTAdaptiveDAGLearner):
    exp12_log: List[Tuple[int, bool, bool]] = field(default_factory=list, repr=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.exp12_log.clear()

    def _after_compare_hook(self) -> None:
        self.exp12_log.append((self.unit_sel, self._gates_mutated_last_tick, self._opp_is_rewire))
