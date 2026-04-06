#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 1 — compare strict vs plateau (sidewalk) XOR learner rules.

Uses ref/tt_xor_learner_spec.py (matches tt_um_xor_learner #(.PLATEAU_ESCAPE)).
RTL default build: tt_um_xor_learner_plateau via project.v — verify with:
  python run_sim.py

Outputs (under viz/):
  experiment_1_strict.png   — strict only (new > old)
  experiment_1_plateau.png  — plateau / sidewalk (tie ~1/8)
  experiment_1_compare.png  — both overlaid for A/B

Requires: matplotlib, numpy (test/requirements.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref"))

import tt_xor_learner_spec as spec  # noqa: E402

# --- experiment parameters ---
TRAJECTORY_SEEDS = [0xACE1, 0x5A5A, 0x1234, 0xBEEF, 0x0001, 0xFFFF, 0x3C3C, 0x7777]
TRAJECTORY_MAX_STEPS = 600
OVERLAY_SEEDS = TRAJECTORY_SEEDS[:4]
BATCH_SEED_START = 1
BATCH_SEED_COUNT = 120
BATCH_MAX_TICKS = 8000

VIZ = ROOT / "viz"
OUT_STRICT = VIZ / "experiment_1_strict.png"
OUT_PLATEAU = VIZ / "experiment_1_plateau.png"
OUT_COMPARE = VIZ / "experiment_1_compare.png"


class BatchStats(NamedTuple):
    rows: list[tuple[int, bool, int]]
    ok_n: int
    success_ticks: list[int]


def trajectory(
    seed: int, max_steps: int, plateau_escape: bool
) -> tuple[list[int], list[int], bool]:
    m = spec.TTXorLearner(plateau_escape=plateau_escape)
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
    seed_start: int, count: int, max_ticks: int, plateau_escape: bool
) -> BatchStats:
    rows: list[tuple[int, bool, int]] = []
    ok_n = 0
    for s in range(seed_start, seed_start + count):
        m = spec.TTXorLearner(plateau_escape=plateau_escape)
        m.reset(seed=s & 0xFFFF or 0xACE1)
        ok, ticks = m.run_until_xor(max_ticks=max_ticks)
        rows.append((s, ok, ticks))
        if ok:
            ok_n += 1
    success_ticks = [t for _, o, t in rows if o]
    return BatchStats(rows, ok_n, success_ticks)


def _single_figure(
    plateau_escape: bool,
    tag: str,
    out_path: Path,
    np,
    plt,
) -> BatchStats:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    rule = "plateau (sidewalk, ~1/8 tie)" if plateau_escape else "strict (new > old only)"
    fig.suptitle(
        f"Experiment 1 — {tag}\n{rule}; {BATCH_SEED_COUNT} seeds × {BATCH_MAX_TICKS} ticks",
        fontsize=11,
    )

    ax0 = axes[0, 0]
    for seed in TRAJECTORY_SEEDS:
        scores, _, conv = trajectory(seed, TRAJECTORY_MAX_STEPS, plateau_escape)
        x = list(range(len(scores)))
        label = f"0x{seed:04X}" + (" ✓" if conv and scores[-1] == 4 else "")
        ax0.plot(x, scores, alpha=0.85, linewidth=1.2, label=label)
    ax0.set_xlabel("Training tick (FSM step index)")
    ax0.set_ylabel("Global XOR score (0–4)")
    ax0.set_ylim(-0.1, 4.2)
    ax0.set_title("Score vs time (several LFSR init seeds)")
    ax0.legend(loc="lower right", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[0, 1]
    demo_seed = TRAJECTORY_SEEDS[0]
    _, fsm_ids, _ = trajectory(demo_seed, min(TRAJECTORY_MAX_STEPS, 400), plateau_escape)
    ax1.plot(range(len(fsm_ids)), fsm_ids, color="C0", linewidth=0.8)
    ax1.set_xlabel("Training tick")
    ax1.set_ylabel("FSM state (IntEnum value)")
    ax1.set_title(f"FSM trace (seed 0x{demo_seed:04X}, first {len(fsm_ids)} ticks)")
    ax1.grid(True, alpha=0.3)

    stats = batch_run(BATCH_SEED_START, BATCH_SEED_COUNT, BATCH_MAX_TICKS, plateau_escape)
    rows, ok_n, success_ticks = stats
    fail_n = BATCH_SEED_COUNT - ok_n

    ax2 = axes[1, 0]
    if success_ticks:
        ax2.hist(
            success_ticks,
            bins=min(25, max(8, len(success_ticks) // 4)),
            color="steelblue",
            edgecolor="white",
        )
        ax2.axvline(
            float(np.mean(success_ticks)),
            color="orange",
            linestyle="--",
            label=f"mean={np.mean(success_ticks):.0f}",
        )
        ax2.legend()
    ax2.set_xlabel("Ticks until XOR (score 4)")
    ax2.set_ylabel("Count (seeds)")
    ax2.set_title(
        f"Convergence time (seeds {BATCH_SEED_START}..{BATCH_SEED_START + BATCH_SEED_COUNT - 1})"
    )
    ax2.grid(True, alpha=0.3)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stats


def _compare_figure(
    st_strict: BatchStats,
    st_plateau: BatchStats,
    np,
    plt,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        "Experiment 1 — strict vs plateau (sidewalk)\n"
        f"{BATCH_SEED_COUNT} seeds × {BATCH_MAX_TICKS} ticks; Python ref = RTL PLATEAU_ESCAPE",
        fontsize=11,
    )

    # (0,0) overlaid histograms
    axh = axes[0, 0]
    bins = 20
    r_max = max(
        (max(st_strict.success_ticks) if st_strict.success_ticks else 0),
        (max(st_plateau.success_ticks) if st_plateau.success_ticks else 0),
        1,
    )
    hi = min(r_max * 1.05, BATCH_MAX_TICKS)
    if st_strict.success_ticks:
        axh.hist(
            st_strict.success_ticks,
            bins=bins,
            range=(0, hi),
            alpha=0.55,
            color="crimson",
            label=f"strict (n={len(st_strict.success_ticks)})",
            edgecolor="white",
        )
    if st_plateau.success_ticks:
        axh.hist(
            st_plateau.success_ticks,
            bins=bins,
            range=(0, hi),
            alpha=0.55,
            color="steelblue",
            label=f"plateau (n={len(st_plateau.success_ticks)})",
            edgecolor="white",
        )
    axh.set_xlabel("Ticks until XOR")
    axh.set_ylabel("Count")
    axh.set_title("Convergence time (shared bins)")
    axh.legend()
    axh.grid(True, alpha=0.3)

    # (0,1) success / fail bars
    axb = axes[0, 1]
    x = np.arange(2)
    w = 0.35
    axb.bar(x - w / 2, [st_strict.ok_n, st_plateau.ok_n], w, label="converged", color="seagreen")
    axb.bar(
        x + w / 2,
        [BATCH_SEED_COUNT - st_strict.ok_n, BATCH_SEED_COUNT - st_plateau.ok_n],
        w,
        label="not in time",
        color="indianred",
    )
    axb.set_xticks(x)
    axb.set_xticklabels(["strict", "plateau"])
    axb.set_ylabel("Seeds")
    axb.set_title("Outcome counts")
    axb.legend()
    axb.grid(True, axis="y", alpha=0.3)

    # (1,0) overlay score trajectories (same seeds)
    axt = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(OVERLAY_SEEDS)))
    for i, seed in enumerate(OVERLAY_SEEDS):
        s_st, _, _ = trajectory(seed, TRAJECTORY_MAX_STEPS, False)
        s_pl, _, _ = trajectory(seed, TRAJECTORY_MAX_STEPS, True)
        c = colors[i]
        axt.plot(s_st, color=c, linestyle="--", linewidth=1.1, alpha=0.9)
        axt.plot(s_pl, color=c, linestyle="-", linewidth=1.1, alpha=0.9, label=f"0x{seed:04X}")
    axt.set_xlabel("Training tick")
    axt.set_ylabel("Global score")
    axt.set_ylim(-0.1, 4.2)
    axt.set_title("Score vs time (solid=plateau, dashed=strict; color=seed)")
    axt.legend(loc="lower right", fontsize=7, ncol=2)
    axt.grid(True, alpha=0.3)

    # (1,1) metrics table
    axtbl = axes[1, 1]
    axtbl.set_axis_off()

    def fmt_ticks(ticks: list[int], np) -> str:
        if not ticks:
            return "—"
        return f"mean {np.mean(ticks):.0f}  med {np.median(ticks):.0f}  max {max(ticks)}"

    tbl = [
        ["", "strict", "plateau"],
        [
            "success",
            f"{st_strict.ok_n}/{BATCH_SEED_COUNT}",
            f"{st_plateau.ok_n}/{BATCH_SEED_COUNT}",
        ],
        [
            "ticks (ok)",
            fmt_ticks(st_strict.success_ticks, np),
            fmt_ticks(st_plateau.success_ticks, np),
        ],
    ]
    table = axtbl.table(
        cellText=tbl,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.39, 0.39],
    )
    table.scale(1.2, 2.0)
    axtbl.set_title("Summary", pad=12)

    OUT_COMPARE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COMPARE, dpi=150)
    plt.close(fig)


def main() -> int:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install matplotlib and numpy, e.g.: pip install matplotlib numpy", file=sys.stderr)
        return 1

    st_strict = _single_figure(False, "strict", OUT_STRICT, np, plt)
    print(f"Wrote {OUT_STRICT}")
    st_plateau = _single_figure(True, "plateau (sidewalk)", OUT_PLATEAU, np, plt)
    print(f"Wrote {OUT_PLATEAU}")
    _compare_figure(st_strict, st_plateau, np, plt)
    print(f"Wrote {OUT_COMPARE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
