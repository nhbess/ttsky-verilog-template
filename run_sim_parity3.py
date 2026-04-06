#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cocotb + Icarus for 3-bit parity learner (separate from XOR run_sim.py)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _maybe_reexec_with_venv() -> None:
    root = Path(__file__).resolve().parent
    venv_python = (
        root / ".venv" / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else root / ".venv" / "bin" / "python"
    )
    if not venv_python.is_file():
        return
    try:
        if Path(sys.executable).resolve().samefile(venv_python.resolve()):
            return
    except (FileNotFoundError, OSError):
        pass
    script = Path(__file__).resolve()
    rc = subprocess.call(
        [str(venv_python), "-u", str(script), *sys.argv[1:]],
        cwd=str(root),
    )
    raise SystemExit(rc)


def _run() -> int:
    from cocotb_tools.check_results import get_results
    from cocotb_tools.runner import get_runner

    root = Path(__file__).resolve().parent
    test_dir = root / "test"
    src_dir = root / "src"
    sim = os.environ.get("SIM", "icarus").lower()
    waves = os.environ.get("WAVES", "0") == "1"

    if sim == "icarus" and shutil.which("iverilog") is None:
        print("ERROR: iverilog is not on PATH.", file=sys.stderr)
        return 1

    build_dir = test_dir / "sim_build" / "parity3"
    runner = get_runner(sim)
    runner.build(
        sources=[
            src_dir / "models" / "tt_um_parity3_learner.v",
            src_dir / "models" / "project_parity3.v",
            test_dir / "tb_parity3.v",
        ],
        includes=[src_dir / "models"],
        defines={"SIM": 1},
        hdl_toplevel="tb_parity3",
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
        waves=waves,
    )
    results_path = runner.test(
        hdl_toplevel="tb_parity3",
        test_module="test_parity3",
        build_dir=build_dir,
        test_dir=test_dir,
        results_xml="results_parity3.xml",
        waves=waves,
    )
    try:
        _, num_failed = get_results(Path(results_path))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    if num_failed:
        print(f"ERROR: {num_failed} test(s) failed (see {results_path}).", file=sys.stderr)
        return 1
    print("Parity3 simulation: all tests passed.")
    return 0


def main() -> int:
    _maybe_reexec_with_venv()
    return _run()


if __name__ == "__main__":
    sys.exit(main())
