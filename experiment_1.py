#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 1 — characterize the XOR learner (Python reference == RTL spec).

Inspired by test/test.py: reset, train every tick, measure convergence; adds
time-series traces and multi-seed statistics + plots.

Uses ref/tt_xor_learner_spec.py (same FSM as src/tt_um_xor_learner.v). RTL is
still validated with: python run_sim.py

Requires: matplotlib (see test/requirements.txt)

Run from repo root:
  python experiment_1.py
Output:
  viz/experiment_1.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref"))

import tt_xor_learner_spec as spec  # noqa: E402

# --- experiment parameters (edit freely) ---
TRAJECTORY_SEEDS = [0xACE1, 0x5A5A, 0x1234, 0xBEEF, 0x0001, 0xFFFF, 0x3C3C, 0x7777]
TRAJECTORY_MAX_STEPS = 600
BATCH_SEED_START = 1
BATCH_SEED_COUNT = 120
BATCH_MAX_TICKS = 8000
OUT_PATH = ROOT / "viz" / "experiment_1.png"


def trajectory(seed: int, max_steps: int) -> tuple[list[int], list[int], bool]:
    """Return (global_score trace, FSM id trace, converged_within_limit)."""
    m = spec.TTXorLearner()
    m.reset(seed=seed)
    scores: list[int] = [m.score_current_gates()]
    fsm_ids: list[int] = [int(m.fsm)]
    converged = scores[0] == 4
    for _ in range(max_steps):
        if scores[-1] == 4:
            converged = True
            break
        m.tick(train_enable=True)
        scores.append(m.score_current_gates())
        fsm_ids.append(int(m.fsm))
    return scores, fsm_ids, converged


def batch_run(
    seed_start: int, count: int, max_ticks: int
) -> tuple[list[tuple[int, bool, int]], int]:
    """
    For each seed in [seed_start, seed_start+count), run until XOR or cap.
    Returns list of (seed, ok, ticks_used) and count of successes.
    """
    rows: list[tuple[int, bool, int]] = []
    ok_n = 0
    for s in range(seed_start, seed_start + count):
        m = spec.TTXorLearner()
        m.reset(seed=s & 0xFFFF or 0xACE1)
        ok, ticks = m.run_until_xor(max_ticks=max_ticks)
        rows.append((s, ok, ticks))
        if ok:
            ok_n += 1
    return rows, ok_n


def main() -> int:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install matplotlib and numpy, e.g.: pip install matplotlib numpy", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        "Experiment 1 — XOR learner (Python reference / RTL spec)\n"
        "global score vs training tick; multi-seed convergence",
        fontsize=12,
    )

    # (0,0) Score trajectories
    ax0 = axes[0, 0]
    for seed in TRAJECTORY_SEEDS:
        scores, _, conv = trajectory(seed, TRAJECTORY_MAX_STEPS)
        x = list(range(len(scores)))
        label = f"0x{seed:04X}" + (" ✓" if conv and scores[-1] == 4 else "")
        ax0.plot(x, scores, alpha=0.85, linewidth=1.2, label=label)
    ax0.set_xlabel("Training tick (FSM step index)")
    ax0.set_ylabel("Global XOR score (0–4)")
    ax0.set_ylim(-0.1, 4.2)
    ax0.set_title("Score vs time (several LFSR init seeds)")
    ax0.legend(loc="lower right", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)

    # (0,1) FSM activity for one seed (step index vs state id)
    ax1 = axes[0, 1]
    demo_seed = TRAJECTORY_SEEDS[0]
    _, fsm_ids, _ = trajectory(demo_seed, min(TRAJECTORY_MAX_STEPS, 400))
    ax1.plot(range(len(fsm_ids)), fsm_ids, color="C0", linewidth=0.8)
    ax1.set_xlabel("Training tick")
    ax1.set_ylabel("FSM state (IntEnum value)")
    ax1.set_title(f"FSM trace (seed 0x{demo_seed:04X}, first {len(fsm_ids)} ticks)")
    ax1.grid(True, alpha=0.3)

    # (1,0) Histogram of ticks to converge
    ax2 = axes[1, 0]
    rows, ok_n = batch_run(BATCH_SEED_START, BATCH_SEED_COUNT, BATCH_MAX_TICKS)
    success_ticks = [t for _, ok, t in rows if ok]
    fail_n = BATCH_SEED_COUNT - ok_n
    if success_ticks:
        ax2.hist(success_ticks, bins=min(25, max(8, len(success_ticks) // 4)), color="steelblue", edgecolor="white")
        ax2.axvline(float(np.mean(success_ticks)), color="orange", linestyle="--", label=f"mean={np.mean(success_ticks):.0f}")
        ax2.legend()
    ax2.set_xlabel("Ticks until XOR (score 4)")
    ax2.set_ylabel("Count (seeds)")
    ax2.set_title(
        f"Convergence time (seeds {BATCH_SEED_START}..{BATCH_SEED_START + BATCH_SEED_COUNT - 1}, cap={BATCH_MAX_TICKS})"
    )
    ax2.grid(True, alpha=0.3)

    # (1,1) Summary text + stacked bar (success vs fail) + strip of first N seeds
    ax3 = axes[1, 1]
    ax3.set_axis_off()
    rate = ok_n / BATCH_SEED_COUNT if BATCH_SEED_COUNT else 0.0
    lines = [
        f"Batch: {BATCH_SEED_COUNT} seeds, max {BATCH_MAX_TICKS} ticks each",
        f"Success: {ok_n} ({100.0 * rate:.1f}%)",
        f"Failed (no XOR in time): {fail_n}",
    ]
    if success_ticks:
        lines.append(f"Ticks — min: {min(success_ticks)}, max: {max(success_ticks)}")
        lines.append(f"Ticks — median: {float(np.median(success_ticks)):.0f}")
    ax3.text(0.02, 0.72, "\n".join(lines), transform=ax3.transAxes, va="top", fontsize=9, family="monospace")

    ax3_bar = ax3.inset_axes([0.02, 0.38, 0.92, 0.12])
    ax3_bar.barh(0, ok_n, height=0.5, color="seagreen", label="converged")
    ax3_bar.barh(0, fail_n, left=ok_n, height=0.5, color="indianred", label="not in time")
    ax3_bar.set_xlim(0, BATCH_SEED_COUNT)
    ax3_bar.set_yticks([])
    ax3_bar.set_xlabel("seed outcomes (count)")
    ax3_bar.legend(loc="upper right", fontsize=7)

    strip_n = min(BATCH_SEED_COUNT, 80)
    mat = np.zeros((1, strip_n))
    for i in range(strip_n):
        mat[0, i] = 1.0 if rows[i][1] else 0.0
    ax3_strip = ax3.inset_axes([0.02, 0.08, 0.92, 0.18])
    ax3_strip.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax3_strip.set_xticks([0, strip_n - 1])
    ax3_strip.set_xticklabels([str(BATCH_SEED_START), str(BATCH_SEED_START + strip_n - 1)], fontsize=7)
    ax3_strip.set_yticks([])
    ax3_strip.set_title(f"First {strip_n} seeds (green=OK, red=fail)", fontsize=8)

    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
