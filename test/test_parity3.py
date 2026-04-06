# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

TRAIN_BIT = 3


def _set_abc(dut, a: int, b: int, c: int) -> None:
    v = int(dut.ui_in.value)
    v = (v & ~0x7) | ((a & 1) << 0) | ((b & 1) << 1) | ((c & 1) << 2)
    dut.ui_in.value = v


@cocotb.test()
async def test_parity3_learner_converges(dut):
    clock = Clock(dut.clk, 40, unit="ns")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    dut.ui_in.value = 1 << TRAIN_BIT
    max_cycles = 400_000
    for _ in range(max_cycles):
        await ClockCycles(dut.clk, 1)
        core = dut.user_project.u_core
        g0 = int(core.g0.value)
        g1 = int(core.g1.value)
        g2 = int(core.g2.value)

        def gate_out(gn, x, y):
            idx = (x & 1) << 1 | (y & 1)
            return (gn >> idx) & 1

        def forward(a, b, c):
            h1 = gate_out(g0, a, b)
            t = gate_out(g1, h1, c)
            return gate_out(g2, h1, t)

        ok = True
        for i in range(8):
            a = (i >> 2) & 1
            b = (i >> 1) & 1
            c = i & 1
            exp = a ^ b ^ c
            if forward(a, b, c) != exp:
                ok = False
                break
        if ok:
            break
    else:
        assert False, f"did not learn 3-bit parity within {max_cycles} cycles"

    dut.ui_in.value = 0
    await ClockCycles(dut.clk, 2)

    for i in range(8):
        a = (i >> 2) & 1
        b = (i >> 1) & 1
        c = i & 1
        exp = a ^ b ^ c
        _set_abc(dut, a, b, c)
        await ClockCycles(dut.clk, 1)
        y = int(dut.uo_out.value) & 1
        assert y == exp, f"pattern {i}: got {y} expected {exp}"
