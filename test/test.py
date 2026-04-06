# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def _set_ab(dut, a: int, b: int) -> None:
    v = int(dut.ui_in.value)
    v = (v & ~0x3) | ((a & 1) << 0) | ((b & 1) << 1)
    dut.ui_in.value = v


@cocotb.test()
async def test_xor_learner_converges(dut):
    dut._log.info("XOR learner cocotb start")
    clock = Clock(dut.clk, 40, unit="ns")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Train: ui_in[2] = train_enable
    dut.ui_in.value = 0x04
    max_cycles = 8000
    for _ in range(max_cycles):
        await ClockCycles(dut.clk, 1)
        core = dut.user_project.u_core.core
        g0 = int(core.g0.value)
        g1 = int(core.g1.value)
        g2 = int(core.g2.value)

        def gate_out(gn, a, b):
            idx = (a & 1) << 1 | (b & 1)
            return (gn >> idx) & 1

        def forward(a, b):
            h1 = gate_out(g0, a, b)
            h2 = gate_out(g1, a, b)
            return gate_out(g2, h1, h2)

        ok = True
        for a, b, t in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            if forward(a, b) != t:
                ok = False
                break
        if ok:
            dut._log.info("Learned XOR after training cycles")
            break
    else:
        assert False, f"did not learn XOR within {max_cycles} cycles"

    # Hold trainer; check combinational inference on outputs
    dut.ui_in.value = 0x00
    await ClockCycles(dut.clk, 2)

    for a, b, exp in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        _set_ab(dut, a, b)
        await ClockCycles(dut.clk, 1)
        y = int(dut.uo_out.value) & 1
        assert y == exp, f"y mismatch for a,b={a},{b}: got {y} expected {exp}"
