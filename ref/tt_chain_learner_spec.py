"""
Fixed-topology multi-output chain learner (behavioral reference).

Generalizes the parity3 wiring: for N >= 3,
  h1 = g0(x0, x1),  r = x2,  r = gk(h1, r) for k = 1..N-2,  y = g_{N-1}(h1, r).
For N == 2: y = g0(x0, x1) (one gate per output).

M outputs = M disjoint subnets (same topology), independent gate RAMs. One global
score counts correct bits over all (input row, output line) pairs.

Training FSM matches tt_parity3_spec (plateau compare, plasticity, LFSR).
unit_sel is uniform over all gates via (lfsr % n_gates) — unlike the 3-gate RTL mux.

Subclasses may override ``_plateau_mask_for_compare`` (time-varying or per-gate p) and
``_after_compare_hook`` (e.g. feedback on trial Δ). Default behavior matches fixed ``plateau_mask``.

Use for scaling experiments: not all truth tables are reachable; success rate vs
(N, M) and target distribution measures learnability under this generator.

Run: python ref/tt_chain_learner_spec.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Sequence, Tuple

import random


def row_bits(idx: int, n_in: int) -> Tuple[int, ...]:
    """idx 0..2^n_in-1; bit n_in-1 is x0 (MSB), bit 0 is x_{n-1} (parity3 order)."""
    return tuple((idx >> (n_in - 1 - j)) & 1 for j in range(n_in))


def gate_eval(tt4: int, x: int, y: int) -> int:
    idx = (x & 1) << 1 | (y & 1)
    return (tt4 >> idx) & 1


def gates_per_output(n_in: int) -> int:
    return 1 if n_in <= 2 else n_in


def total_gate_count(n_in: int, n_out: int) -> int:
    return n_out * gates_per_output(n_in)


def parity_of_row_idx(idx: int, n_in: int) -> int:
    p = 0
    v = idx
    for _ in range(n_in):
        p ^= v & 1
        v >>= 1
    return p


def make_truth_tables(
    kind: str, n_in: int, n_out: int, *, table_seed: int
) -> List[List[int]]:
    """Each inner list length 2**n_in (target bit per minterm)."""
    nrows = 1 << n_in
    rng = random.Random(table_seed)
    if kind == "parity":
        col = [parity_of_row_idx(i, n_in) for i in range(nrows)]
        return [col[:] for _ in range(n_out)]
    if kind == "zero":
        return [[0] * nrows for _ in range(n_out)]
    if kind == "random":
        return [[rng.randint(0, 1) for _ in range(nrows)] for _ in range(n_out)]
    raise ValueError(f"unknown truth table kind: {kind!r}")


class FsmState(IntEnum):
    IDLE = auto()
    INIT_UNIT = auto()
    OLD_CLEAR = auto()
    OLD_ACC = auto()
    PROPOSE = auto()
    NEW_CLEAR = auto()
    NEW_ACC = auto()
    COMPARE = auto()


def lfsr16_step(state: int) -> int:
    state &= 0xFFFF
    if state == 0:
        state = 0xACE1
    lsb = state & 1
    state >>= 1
    if lsb:
        state ^= 0xB400
    return state & 0xFFFF


def unit_sel_mod(lfsr: int, n_units: int) -> int:
    if n_units <= 1:
        return 0
    return (lfsr & 0xFFFF) % n_units


@dataclass
class TTChainLearner:
    n_in: int = 3
    n_out: int = 1
    target: List[List[int]] = field(default_factory=list)
    gate: List[int] = field(default_factory=list)
    plastic: List[int] = field(default_factory=list)
    unit_sel: int = 0
    sample_idx: int = 0
    old_score: int = 0
    new_score: int = 0
    old_gate: int = 0
    trial_gate: int = 0
    lfsr: int = 0xACE1
    # Set True in tick() only when COMPARE accepted a gate write (score can change).
    _gates_mutated_last_tick: bool = field(default=False, repr=False, compare=False)
    fsm: FsmState = FsmState.IDLE
    plateau_escape: bool = True
    plateau_mask: int = 3

    @property
    def nrows(self) -> int:
        return 1 << self.n_in

    @property
    def max_score(self) -> int:
        return self.nrows * self.n_out

    @property
    def gpo(self) -> int:
        return gates_per_output(self.n_in)

    def __post_init__(self) -> None:
        if not self.target:
            self.target = make_truth_tables("parity", self.n_in, self.n_out, table_seed=0)
        ng = total_gate_count(self.n_in, self.n_out)
        if len(self.gate) != ng:
            self.gate = [0] * ng
        if len(self.plastic) != ng:
            self.plastic = [3] * ng

    def reset(self, seed: int = 0xACE1) -> None:
        ng = total_gate_count(self.n_in, self.n_out)
        self.gate = [0] * ng
        self.plastic = [3] * ng
        self.unit_sel = 0
        self.sample_idx = 0
        self.old_score = 0
        self.new_score = 0
        self.old_gate = 0
        self.trial_gate = 0
        self.lfsr = seed & 0xFFFF or 0xACE1
        self.fsm = FsmState.IDLE
        for _ in range(ng):
            self.lfsr = lfsr16_step(self.lfsr)
            self.gate[_] = self.lfsr & 0xF

    def _forward_one_out_slice(self, xs: Sequence[int], base: int) -> int:
        g = self.gate
        n = self.n_in
        if n <= 2:
            return gate_eval(g[base], xs[0], xs[1])
        h1 = gate_eval(g[base], xs[0], xs[1])
        r = xs[2]
        for k in range(1, n - 1):
            r = gate_eval(g[base + k], h1, r)
        return gate_eval(g[base + n - 1], h1, r)

    def _forward_one_out_trial(self, xs: Sequence[int], base: int) -> int:
        """Like _forward_one_out_slice but gate at ``unit_sel`` reads ``trial_gate``."""
        g = self.gate
        us = self.unit_sel
        tg = self.trial_gate
        n = self.n_in
        tro = getattr(self, "_trial_overrides", None)

        def gv(off: int) -> int:
            gi = base + off
            if tro is not None:
                if gi in tro:
                    return tro[gi]
                return g[gi]
            return tg if gi == us else g[gi]

        if n <= 2:
            return gate_eval(gv(0), xs[0], xs[1])
        h1 = gate_eval(gv(0), xs[0], xs[1])
        r = xs[2]
        for k in range(1, n - 1):
            r = gate_eval(gv(k), h1, r)
        return gate_eval(gv(n - 1), h1, r)

    def forward_all(self, xs: Sequence[int]) -> Tuple[int, ...]:
        gpo = self.gpo
        return tuple(self._forward_one_out_slice(xs, m * gpo) for m in range(self.n_out))

    def _forward_trial_all(self, xs: Sequence[int]) -> Tuple[int, ...]:
        gpo = self.gpo
        return tuple(self._forward_one_out_trial(xs, m * gpo) for m in range(self.n_out))

    def score_current_gates(self) -> int:
        s = 0
        for idx in range(self.nrows):
            xs = row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            for m in range(self.n_out):
                if ys[m] == self.target[m][idx]:
                    s += 1
        return s

    def _plateau_mask_for_compare(self) -> int:
        """Effective tie mask in COMPARE; override for time-varying or per-gate p."""
        return self.plateau_mask

    def _after_compare_hook(self) -> None:
        """Called at end of COMPARE (after accept/reject); override for feedback control."""
        pass

    def _sample_unit_for_trial(self, n_gates: int) -> int:
        """Which gate index to mutate this cycle (after IDLE → INIT). Default: uniform via LFSR."""
        self.lfsr = lfsr16_step(self.lfsr)
        return unit_sel_mod(self.lfsr, n_gates)

    def _plastic_trial_allowed(self, u: int, rnd2: int) -> bool:
        """INIT: ``rnd2`` in 0..3; return whether this gate may start a mutation trial."""
        return rnd2 < self.plastic[u] + 1

    def _compare_accept(self) -> bool:
        """COMPARE branch: accept proposed mutation (override, e.g. local error-only rules)."""
        if self.plateau_escape:
            self.lfsr = lfsr16_step(self.lfsr)
            rare = (self.lfsr & self._plateau_mask_for_compare()) == 0
            return self.new_score > self.old_score or (
                self.new_score == self.old_score and rare
            )
        return self.new_score > self.old_score

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        self._gates_mutated_last_tick = False
        n_gates = len(self.gate)
        s = self.fsm

        if s == FsmState.IDLE:
            self.fsm = FsmState.INIT_UNIT

        elif s == FsmState.INIT_UNIT:
            self.unit_sel = self._sample_unit_for_trial(n_gates)
            self.lfsr = lfsr16_step(self.lfsr)
            rnd2 = self.lfsr & 0x3
            u = self.unit_sel
            allow = self._plastic_trial_allowed(u, rnd2)
            if not allow:
                self.fsm = FsmState.IDLE
            else:
                self.old_gate = self.gate[u]
                self.fsm = FsmState.OLD_CLEAR

        elif s == FsmState.OLD_CLEAR:
            self.sample_idx = 0
            self.old_score = 0
            self.fsm = FsmState.OLD_ACC

        elif s == FsmState.OLD_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            m = self.sample_idx // self.nrows
            xs = row_bits(idx, self.n_in)
            ys = self.forward_all(xs)
            if ys[m] == self.target[m][idx]:
                self.old_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
                self.fsm = FsmState.PROPOSE

        elif s == FsmState.PROPOSE:
            self.lfsr = lfsr16_step(self.lfsr)
            cand = self.lfsr & 0xF
            if cand == self.old_gate:
                cand ^= 0x1
            self.trial_gate = cand
            self.fsm = FsmState.NEW_CLEAR

        elif s == FsmState.NEW_CLEAR:
            self.sample_idx = 0
            self.new_score = 0
            self.fsm = FsmState.NEW_ACC

        elif s == FsmState.NEW_ACC:
            npts = self.max_score
            idx = self.sample_idx % self.nrows
            m = self.sample_idx // self.nrows
            xs = row_bits(idx, self.n_in)
            ys = self._forward_trial_all(xs)
            if ys[m] == self.target[m][idx]:
                self.new_score += 1
            self.sample_idx += 1
            if self.sample_idx >= npts:
                self.fsm = FsmState.COMPARE

        elif s == FsmState.COMPARE:
            u = self.unit_sel
            accept = self._compare_accept()
            if accept:
                self.gate[u] = self.trial_gate
                self.plastic[u] = max(0, self.plastic[u] - 1)
                self._gates_mutated_last_tick = True
            else:
                self.plastic[u] = min(3, self.plastic[u] + 1)
            self._after_compare_hook()
            self.fsm = FsmState.IDLE

    def run_until_perfect(self, max_ticks: int) -> Tuple[bool, int]:
        """Same outcome as full score-every-tick loop: gates only change on COMPARE accept."""
        if self.score_current_gates() == self.max_score:
            return True, 0
        for t in range(max_ticks):
            self.tick()
            if self._gates_mutated_last_tick and self.score_current_gates() == self.max_score:
                return True, t + 1
        return self.score_current_gates() == self.max_score, max_ticks


def main() -> None:
    tgt = make_truth_tables("parity", 3, 1, table_seed=0)
    m = TTChainLearner(n_in=3, n_out=1, target=tgt, plateau_escape=True, plateau_mask=3)
    m.reset()
    print("chain N=3 M=1 parity", "score", m.score_current_gates(), "/", m.max_score)
    ok, cyc = m.run_until_perfect(80_000)
    print("learned", ok, "ticks", cyc)


if __name__ == "__main__":
    main()
