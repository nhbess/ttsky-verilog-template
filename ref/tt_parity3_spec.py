"""
3-bit parity learner (behavioral reference) — independent from XOR spec.

Topology: h1=g0(a,b), t=g1(h1,c), y=g2(h1,t). This 2-level chain can realize
a^b^c (e.g. g0,g1 XOR + g2 copying t). The wiring h1=f(a,b), h2=f(b,c), y=f(h1,h2)
cannot implement 3-input XOR (truth-table conflict on f).

Training: 8 rows, score 0..8. Plateau uses plateau_mask=3 (~1/4 tie rate after
LFSR step); XOR learner uses /8. Tunable on the dataclass for experiments.

Pin semantics (RTL): ui_in[0]=a, ui_in[1]=b, ui_in[2]=c, ui_in[3]=train_enable.
Matches: src/models/tt_um_parity3_learner.v (PLATEAU_AND_MASK default 3).

Run: python ref/tt_parity3_spec.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Tuple


def row_abc(idx: int) -> Tuple[int, int, int]:
    """idx 0..7 -> (a,b,c) with a=msb of 3-bit index (Verilog sample_idx order)."""
    i = idx & 7
    return (i >> 2) & 1, (i >> 1) & 1, i & 1


def gate_eval(tt4: int, x: int, y: int) -> int:
    idx = (x & 1) << 1 | (y & 1)
    return (tt4 >> idx) & 1


def target_parity3(a: int, b: int, c: int) -> int:
    return (a & 1) ^ (b & 1) ^ (c & 1)


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


def unit_sel_from_lfsr_low2(lfsr: int) -> int:
    u = lfsr & 3
    return 0 if u == 3 else u


@dataclass
class TTParity3Learner:
    gate: List[int] = field(default_factory=lambda: [0, 0, 0])
    plastic: List[int] = field(default_factory=lambda: [3, 3, 3])
    unit_sel: int = 0
    sample_idx: int = 0
    old_score: int = 0
    new_score: int = 0
    old_gate: int = 0
    trial_gate: int = 0
    lfsr: int = 0xACE1
    fsm: FsmState = FsmState.IDLE
    plateau_escape: bool = True
    # 3 for 1/4 tie rate (parity search space is larger than XOR); 7 for 1/8 like XOR.
    plateau_mask: int = 3

    def reset(self, seed: int = 0xACE1) -> None:
        self.gate = [0, 0, 0]
        self.plastic = [3, 3, 3]
        self.unit_sel = 0
        self.sample_idx = 0
        self.old_score = 0
        self.new_score = 0
        self.old_gate = 0
        self.trial_gate = 0
        self.lfsr = seed & 0xFFFF or 0xACE1
        self.fsm = FsmState.IDLE
        for i in range(3):
            self.lfsr = lfsr16_step(self.lfsr)
            self.gate[i] = self.lfsr & 0xF

    def forward(self, a: int, b: int, c: int) -> int:
        h1 = gate_eval(self.gate[0], a, b)
        t = gate_eval(self.gate[1], h1, c)
        return gate_eval(self.gate[2], h1, t)

    def _forward_trial(self, a: int, b: int, c: int) -> int:
        u = self.unit_sel
        g = list(self.gate)
        g[u] = self.trial_gate
        h1 = gate_eval(g[0], a, b)
        t = gate_eval(g[1], h1, c)
        return gate_eval(g[2], h1, t)

    def score_current_gates(self) -> int:
        s = 0
        for i in range(8):
            a, b, c = row_abc(i)
            if self.forward(a, b, c) == target_parity3(a, b, c):
                s += 1
        return s

    def tick(self, train_enable: bool = True) -> None:
        if not train_enable:
            return
        s = self.fsm

        if s == FsmState.IDLE:
            self.fsm = FsmState.INIT_UNIT

        elif s == FsmState.INIT_UNIT:
            self.lfsr = lfsr16_step(self.lfsr)
            self.unit_sel = unit_sel_from_lfsr_low2(self.lfsr)
            self.lfsr = lfsr16_step(self.lfsr)
            rnd2 = self.lfsr & 0x3
            u = self.unit_sel
            allow = rnd2 < self.plastic[u] + 1
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
            a, b, c = row_abc(self.sample_idx)
            if self.forward(a, b, c) == target_parity3(a, b, c):
                self.old_score += 1
            self.sample_idx += 1
            if self.sample_idx >= 8:
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
            a, b, c = row_abc(self.sample_idx)
            if self._forward_trial(a, b, c) == target_parity3(a, b, c):
                self.new_score += 1
            self.sample_idx += 1
            if self.sample_idx >= 8:
                self.fsm = FsmState.COMPARE

        elif s == FsmState.COMPARE:
            u = self.unit_sel
            if self.plateau_escape:
                self.lfsr = lfsr16_step(self.lfsr)
                rare = (self.lfsr & self.plateau_mask) == 0
                accept = self.new_score > self.old_score or (
                    self.new_score == self.old_score and rare
                )
            else:
                accept = self.new_score > self.old_score
            if accept:
                self.gate[u] = self.trial_gate
                self.plastic[u] = max(0, self.plastic[u] - 1)
            else:
                self.plastic[u] = min(3, self.plastic[u] + 1)
            self.fsm = FsmState.IDLE

    def run_until_parity(self, max_ticks: int = 200_000) -> Tuple[bool, int]:
        for t in range(max_ticks):
            if self.score_current_gates() == 8:
                return True, t
            self.tick()
        return self.score_current_gates() == 8, max_ticks


def main() -> None:
    m = TTParity3Learner(plateau_escape=True)
    m.reset()
    print("3-bit parity learner (plateau)")
    print("initial score (0..8):", m.score_current_gates())
    ok, cyc = m.run_until_parity(50_000)
    print("learned:", ok, "ticks:", cyc, "final score:", m.score_current_gates())


if __name__ == "__main__":
    main()
