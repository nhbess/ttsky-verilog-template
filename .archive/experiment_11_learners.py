# SPDX-License-Identifier: Apache-2.0
"""Instrumented Exp72 learners for experiment 11 (log each COMPARE trial)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import tt_chain_learner_spec as chain

import tt_chain_learner_exp72_credit as e72c
import tt_chain_learner_exp72_local as e72loc


@dataclass
class Exp11InstrumentedGlobal(e72c.TTChainLearnerExp72CreditWeighted):
    """Global SCORE accept; credit-weighted ``u`` (experiment 9-style baseline)."""

    compare_log_global_delta: List[int] = field(default_factory=list, repr=False)
    compare_log_eps_delta: List[float] = field(default_factory=list, repr=False)
    compare_log_accepted: List[bool] = field(default_factory=list, repr=False)
    compare_log_unit_sel: List[int] = field(default_factory=list, repr=False)

    def tick(self, train_enable: bool = True) -> None:
        entering = train_enable and self.fsm == chain.FsmState.COMPARE
        super().tick(train_enable)
        if entering:
            self.compare_log_global_delta.append(self.new_score - self.old_score)
            self.compare_log_accepted.append(self._gates_mutated_last_tick)
            self.compare_log_unit_sel.append(self.unit_sel)
            self.compare_log_eps_delta.append(float("nan"))

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.compare_log_global_delta.clear()
        self.compare_log_eps_delta.clear()
        self.compare_log_accepted.clear()
        self.compare_log_unit_sel.clear()


@dataclass
class Exp11InstrumentedLocalUniform(
    e72loc.TTChainLearnerExp72LocalUniform,
):
    compare_log_global_delta: List[int] = field(default_factory=list, repr=False)
    compare_log_eps_delta: List[float] = field(default_factory=list, repr=False)
    compare_log_accepted: List[bool] = field(default_factory=list, repr=False)
    compare_log_unit_sel: List[int] = field(default_factory=list, repr=False)

    def tick(self, train_enable: bool = True) -> None:
        entering = train_enable and self.fsm == chain.FsmState.COMPARE
        super().tick(train_enable)
        if entering:
            self.compare_log_global_delta.append(self.new_score - self.old_score)
            self.compare_log_accepted.append(self._gates_mutated_last_tick)
            self.compare_log_unit_sel.append(self.unit_sel)
            ed = getattr(self, "_last_eps_delta", None)
            self.compare_log_eps_delta.append(float(ed) if ed is not None else float("nan"))

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.compare_log_global_delta.clear()
        self.compare_log_eps_delta.clear()
        self.compare_log_accepted.clear()
        self.compare_log_unit_sel.clear()


@dataclass
class Exp11InstrumentedLocalCredit(e72loc.TTChainLearnerExp72LocalCredit):
    compare_log_global_delta: List[int] = field(default_factory=list, repr=False)
    compare_log_eps_delta: List[float] = field(default_factory=list, repr=False)
    compare_log_accepted: List[bool] = field(default_factory=list, repr=False)
    compare_log_unit_sel: List[int] = field(default_factory=list, repr=False)

    def tick(self, train_enable: bool = True) -> None:
        entering = train_enable and self.fsm == chain.FsmState.COMPARE
        super().tick(train_enable)
        if entering:
            self.compare_log_global_delta.append(self.new_score - self.old_score)
            self.compare_log_accepted.append(self._gates_mutated_last_tick)
            self.compare_log_unit_sel.append(self.unit_sel)
            ed = getattr(self, "_last_eps_delta", None)
            self.compare_log_eps_delta.append(float(ed) if ed is not None else float("nan"))

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.compare_log_global_delta.clear()
        self.compare_log_eps_delta.clear()
        self.compare_log_accepted.clear()
        self.compare_log_unit_sel.clear()
