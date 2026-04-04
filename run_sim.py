#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run cocotb + Icarus for this project (same sources as test/Makefile).

Use this from the repo root — with or without activating .venv; the script
re-invokes itself with .venv’s Python when that environment exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _maybe_reexec_with_venv() -> None:
    # If a repo-local .venv exists, restart this script with that interpreter so
    # cocotb / cocotb_tools are found without the user manually activating venv.
    root = Path(__file__).resolve().parent
    venv_python = (
        root / ".venv" / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else root / ".venv" / "bin" / "python"
    )
    if not venv_python.is_file():
        return
    try:
        # Already running under the venv — avoid infinite re-exec.
        if Path(sys.executable).resolve().samefile(venv_python.resolve()):
            return
    except (FileNotFoundError, OSError):
        pass
    script = Path(__file__).resolve()
    # -u: unbuffered stdout/stderr so cocotb output appears promptly.
    rc = subprocess.call(
        [str(venv_python), "-u", str(script), *sys.argv[1:]],
        cwd=str(root),
    )
    raise SystemExit(rc)


def _run_simulation() -> int:
    # Deferred imports: only load cocotb after optional venv re-exec above.
    from cocotb_tools.check_results import get_results
    from cocotb_tools.runner import get_runner

    root = Path(__file__).resolve().parent
    test_dir = root / "test"
    src_dir = root / "src"
    # SIM selects the simulator backend (default Icarus Verilog).
    sim = os.environ.get("SIM", "icarus").lower()
    # WAVES=1 enables waveform dumping if the runner/simulator supports it.
    waves = os.environ.get("WAVES", "0") == "1"

    if sim == "icarus" and shutil.which("iverilog") is None:
        print("ERROR: iverilog is not on PATH.", file=sys.stderr)
        return 1

    # Output directory for compiled RTL and cocotb artifacts (mirrors typical Makefile layout).
    build_dir = test_dir / "sim_build" / "rtl"

    runner = get_runner(sim)
    # Compile Verilog: DUT (project.v), testbench wrapper (tb.v), search paths for `include.
    runner.build(
        sources=[
            src_dir / "hvsync_generator.v",
            src_dir / "project.v",
            test_dir / "tb.v",
        ],
        includes=[src_dir],
        defines={"SIM": 1},
        hdl_toplevel="tb",
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
        waves=waves,
    )
    # Run Python tests in test/test.py (cocotb test module name "test") against top-level "tb".
    results_path = runner.test(
        hdl_toplevel="tb",
        test_module="test",
        build_dir=build_dir,
        test_dir=test_dir,
        results_xml="results.xml",
        waves=waves,
    )

    try:
        # Parse JUnit-style results written by the runner; num_failed drives exit status.
        _, num_failed = get_results(Path(results_path))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    if num_failed:
        print(f"ERROR: {num_failed} test(s) failed (see {results_path}).", file=sys.stderr)
        return 1
    print("All tests passed.")
    return 0


def main() -> int:
    _maybe_reexec_with_venv()
    return _run_simulation()


if __name__ == "__main__":
    sys.exit(main())
