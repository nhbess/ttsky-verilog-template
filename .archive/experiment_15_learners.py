# SPDX-License-Identifier: Apache-2.0
"""Experiment 15 — per-gate softmax over 16 LUT ops; sample proposals + accept/reject logits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import tt_chain_learner_spec as chain

import experiment_12_learners as e12l
import tt_adaptive_dag_local_predictive as adlp
import tt_adaptive_dag_local_unsup as adlu


def softmax_probs(w: List[float]) -> List[float]:
    m = max(w)
    exps = [math.exp(x - m) for x in w]
    s = sum(exps)
    return [e / s for e in exps]


def entropy_bits(probs: List[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * math.log2(p)
    return h


def sample_categorical_from_probs(probs: List[float], learner) -> int:
    learner.lfsr = chain.lfsr16_step(learner.lfsr)
    r = (learner.lfsr & 0xFFFF) / 65536.0
    c = 0.0
    for i in range(16):
        c += probs[i]
        if r < c:
            return i
    return 15


def per_gate_entropies_bits(logits: List[List[float]]) -> List[float]:
    return [entropy_bits(softmax_probs(w)) for w in logits]


@dataclass
class Exp15SupervisedDAG(e12l.Exp12AdaptiveDAG):
    eta: float = 1.0
    lambda_pen: float = 0.0
    logits: List[List[float]] = field(default_factory=list)
    op_choice_history: List[int] = field(default_factory=list, repr=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.logits = [[0.0] * 16 for _ in range(self.n_gates_flat)]
        self.op_choice_history.clear()

    def _propose_lut_nibble(self, u: int) -> int:
        probs = softmax_probs(self.logits[u])
        idx = sample_categorical_from_probs(probs, self)
        self.op_choice_history.append(idx)
        return idx

    def _after_lut_compare(self, u: int, accepted: bool) -> None:
        idx = self.trial_gate
        if accepted:
            self.logits[u][idx] += self.eta
        else:
            self.logits[u][idx] -= self.lambda_pen


@dataclass
class Exp15LocalUnsupDAG(adlu.TTAdaptiveDAGLocalUnsupervised):
    eta: float = 1.0
    lambda_pen: float = 0.0
    logits: List[List[float]] = field(default_factory=list)
    op_choice_history: List[int] = field(default_factory=list, repr=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.logits = [[0.0] * 16 for _ in range(self.n_gates_flat)]
        self.op_choice_history.clear()

    def _propose_lut_nibble(self, u: int) -> int:
        probs = softmax_probs(self.logits[u])
        idx = sample_categorical_from_probs(probs, self)
        self.op_choice_history.append(idx)
        return idx

    def _after_lut_compare(self, u: int, accepted: bool) -> None:
        idx = self.trial_gate
        if accepted:
            self.logits[u][idx] += self.eta
        else:
            self.logits[u][idx] -= self.lambda_pen


@dataclass
class Exp15LocalPredictiveDAG(adlp.TTAdaptiveDAGLocalPredictive):
    eta: float = 1.0
    lambda_pen: float = 0.0
    logits: List[List[float]] = field(default_factory=list)
    op_choice_history: List[int] = field(default_factory=list, repr=False)

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self.logits = [[0.0] * 16 for _ in range(self.n_gates_flat)]
        self.op_choice_history.clear()

    def _propose_lut_nibble(self, u: int) -> int:
        probs = softmax_probs(self.logits[u])
        idx = sample_categorical_from_probs(probs, self)
        self.op_choice_history.append(idx)
        return idx

    def _after_lut_compare(self, u: int, accepted: bool) -> None:
        idx = self.trial_gate
        if accepted:
            self.logits[u][idx] += self.eta
        else:
            self.logits[u][idx] -= self.lambda_pen
