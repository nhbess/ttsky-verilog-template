"""
Tiny Tapeout–style minimal XOR learner (behavioral reference).

Golden spec for: src/tt_um_xor_learner.v (keep in sync).

Run: python ref/tt_xor_learner_spec.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Tuple

# --- XOR training set (2-bit sample_idx indexes these) ---
XOR_ROWS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
)


def gate_eval(tt4: int, x: int, y: int) -> int:
    """4-bit truth table tt4[3:0]; index is {x,y} as bits (y LSb)."""
    idx = (x & 1) << 1 | (y & 1)
    return (tt4 >> idx) & 1


def target_xor(a: int, b: int) -> int:
    return (a & 1) ^ (b & 1)


class FsmState(IntEnum):
    """Sequential trainer FSM (one transition per simulated clock)."""

    IDLE = auto()
    INIT_UNIT = auto()
    OLD_CLEAR = auto()
    OLD_ACC = auto()
    PROPOSE = auto()
    NEW_CLEAR = auto()
    NEW_ACC = auto()
    COMPARE = auto()


def lfsr16_step(state: int) -> int:
    """16-bit Fibonacci LFSR, period 2^16-1; poly taps at 16,14,13,11."""
    state &= 0xFFFF
    if state == 0:
        state = 0xACE1
    lsb = state & 1
    state >>= 1
    if lsb:
        state ^= 0xB400
    return state & 0xFFFF


def unit_sel_from_lfsr_low2(lfsr: int) -> int:
    """Map lfsr[1:0] to 0..2 with 3 remapped to 0 (cheap biased mux in RTL)."""
    u = lfsr & 3
    return 0 if u == 3 else u


@dataclass
class TTXorLearner:
    """
    Register block (TT-minimal):

      gate1[3:0], gate2[3:0], gate3[3:0]
      plastic1[1:0], plastic2[1:0], plastic3[1:0]
                        # threshold vs LFSR: allow if rnd2 < plastic[u]+1

      unit_sel[1:0]     # 0..2
      sample_idx[1:0]   # 0..3 during scoring passes
      old_score[2:0]    # 0..4 fits in 3 bits
      new_score[2:0]

      old_gate[3:0]
      trial_gate[3:0]

      lfsr[15:0]        # random unit, trial nibble, tie-break
      state             # FSM (frozen when train_enable=0 in RTL)
    """

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
        # Randomize initial gates from LFSR (cold start = bad score, high plasticity).
        for i in range(3):
            self.lfsr = lfsr16_step(self.lfsr)
            self.gate[i] = self.lfsr & 0xF

    def forward(self, a: int, b: int) -> int:
        h1 = gate_eval(self.gate[0], a, b)
        h2 = gate_eval(self.gate[1], a, b)
        return gate_eval(self.gate[2], h1, h2)

    def _forward_trial(self, a: int, b: int) -> int:
        """Evaluate with trial_gate swapped into unit_sel only."""
        u = self.unit_sel
        g = list(self.gate)
        g[u] = self.trial_gate
        h1 = gate_eval(g[0], a, b)
        h2 = gate_eval(g[1], a, b)
        return gate_eval(g[2], h1, h2)

    def score_current_gates(self) -> int:
        s = 0
        for a, b, _ in XOR_ROWS:
            if self.forward(a, b) == target_xor(a, b):
                s += 1
        return s

    def tick(self, train_enable: bool = True) -> None:
        """
        One clock edge. When train_enable is True (ui_in[2] in RTL), advance the
        training FSM and LFSR as specified. When False, hold all trainer state;
        inference remains forward(a,b) from pins, unchanged.
        """
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
            a, b, _ = XOR_ROWS[self.sample_idx]
            if self.forward(a, b) == target_xor(a, b):
                self.old_score += 1
            self.sample_idx += 1
            if self.sample_idx >= 4:
                self.fsm = FsmState.PROPOSE
            # else stay OLD_ACC

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
            a, b, _ = XOR_ROWS[self.sample_idx]
            if self._forward_trial(a, b) == target_xor(a, b):
                self.new_score += 1
            self.sample_idx += 1
            if self.sample_idx >= 4:
                self.fsm = FsmState.COMPARE

        elif s == FsmState.COMPARE:
            u = self.unit_sel
            if self.new_score > self.old_score:
                self.gate[u] = self.trial_gate
                self.plastic[u] = max(0, self.plastic[u] - 1)
            else:
                # Trial never committed to live gates; only plasticity rises.
                self.plastic[u] = min(3, self.plastic[u] + 1)
            self.fsm = FsmState.IDLE

    def run_until_xor(self, max_ticks: int = 50_000) -> Tuple[bool, int]:
        """Run FSM cycles until XOR is perfect or max_ticks exhausted."""
        for t in range(max_ticks):
            if self.score_current_gates() == 4:
                return True, t
            self.tick()
        return self.score_current_gates() == 4, max_ticks


def main() -> None:
    m = TTXorLearner()
    m.reset()
    print("TT XOR learner - register-level Python model")
    print("initial gates:", [f"{g:04b}" for g in m.gate], "plastic:", m.plastic)
    print("initial score (0..4):", m.score_current_gates())
    ok, cycles = m.run_until_xor()
    print("learned XOR:", ok, "fsm cycles (approx clk edges):", cycles)
    print("final gates:", [f"{g:04b}" for g in m.gate], "plastic:", m.plastic)
    print("sanity:", [m.forward(a, b) for a, b, _ in XOR_ROWS], "want", [t for _, _, t in XOR_ROWS])


if __name__ == "__main__":
    main()
