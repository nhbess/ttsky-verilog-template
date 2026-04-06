#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 3 — parity3: sweep tie-accept probability p (exploration vs exploitation).

Uses ref/tt_parity3_spec.py. After one LFSR step in COMPARE, accept equal score if
  (lfsr & mask) == 0  →  P(accept | tie) ≈ 1/(mask+1) for mask = 2^k - 1.

Sweep:
  p = 0     strict (new > old only; no LFSR step in COMPARE)
  p = 1/16  mask 15
  p = 1/8   mask 7
  p = 1/4   mask 3
  p = 1/2   mask 1

RTL alignment: tt_um_parity3_learner uses PLATEAU_AND_MASK; strict = PLATEAU_ESCAPE 0.

Run from repo root:
  python src/experiments/experiment_3.py

Outputs: src/results/experiment_3_sweep.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_parity3_spec as spec  # noqa: E402

# --- knobs ---
BATCH_SEED_START = 1
BATCH_SEED_COUNT = 24
BATCH_MAX_TICKS = 35_000
MAX_SCORE = 8

# (label, p_theory, plateau_escape, plateau_mask when escape else ignored)
P_LEVELS: List[Tuple[str, float, bool, int]] = [
    ("strict\n($p=0$)", 0.0, False, 0),
    (r"$p=\frac{1}{16}$", 1 / 16, True, 15),
    (r"$p=\frac{1}{8}$", 1 / 8, True, 7),
    (r"$p=\frac{1}{4}$", 1 / 4, True, 3),
    (r"$p=\frac{1}{2}$", 1 / 2, True, 1),
]

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_3_sweep.png"

TRAJECTORY_SEED = 0xACE1
TRAJ_STEPS = 2500


def make_learner(plateau_escape: bool, plateau_mask: int) -> spec.TTParity3Learner:
    return spec.TTParity3Learner(
        plateau_escape=plateau_escape,
        plateau_mask=plateau_mask,
    )


def batch_for_level(
    plateau_escape: bool, plateau_mask: int
) -> Tuple[int, List[int], List[Tuple[int, bool, int]]]:
    rows: List[Tuple[int, bool, int]] = []
    ok_n = 0
    ticks_ok: List[int] = []
    for s in range(BATCH_SEED_START, BATCH_SEED_START + BATCH_SEED_COUNT):
        m = make_learner(plateau_escape, plateau_mask)
        m.reset(seed=s & 0xFFFF or 0xACE1)
        ok, ticks = m.run_until_parity(max_ticks=BATCH_MAX_TICKS)
        rows.append((s, ok, ticks))
        if ok:
            ok_n += 1
            ticks_ok.append(ticks)
    return ok_n, ticks_ok, rows


def trajectory_scores(
    plateau_escape: bool, plateau_mask: int, seed: int, max_steps: int
) -> List[int]:
    m = make_learner(plateau_escape, plateau_mask)
    m.reset(seed=seed)
    scores = [m.score_current_gates()]
    for _ in range(max_steps):
        if scores[-1] == MAX_SCORE:
            break
        m.tick(train_enable=True)
        scores.append(m.score_current_gates())
    return scores


def main() -> int:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy", file=sys.stderr)
        return 1

    n_levels = len(P_LEVELS)
    success_rates: List[float] = []
    mean_ticks: List[float] = []
    med_ticks: List[float] = []
    q25: List[float] = []
    q75: List[float] = []
    ticks_lists: List[List[int]] = []
    all_rows: List[List[Tuple[int, bool, int]]] = []

    for _label, _p, esc, mask in P_LEVELS:
        ok_n, ticks_ok, rows = batch_for_level(esc, mask)
        success_rates.append(ok_n / BATCH_SEED_COUNT)
        ticks_lists.append(ticks_ok)
        all_rows.append(rows)
        if ticks_ok:
            mean_ticks.append(float(np.mean(ticks_ok)))
            med_ticks.append(float(np.median(ticks_ok)))
            q25.append(float(np.percentile(ticks_ok, 25)))
            q75.append(float(np.percentile(ticks_ok, 75)))
        else:
            mean_ticks.append(float("nan"))
            med_ticks.append(float("nan"))
            q25.append(float("nan"))
            q75.append(float("nan"))

    x = np.arange(n_levels)
    p_numeric = [cfg[1] for cfg in P_LEVELS]

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.0])

    fig.suptitle(
        "Experiment 3 — 3-bit parity: tie-accept probability sweep\n"
        f"{BATCH_SEED_COUNT} seeds × {BATCH_MAX_TICKS} ticks per setting (Python ref = RTL policy)",
        fontsize=12,
    )

    # (0,0) Success rate
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(x, success_rates, color="steelblue", edgecolor="white")
    ax0.set_xticks(x)
    ax0.set_xticklabels([c[0] for c in P_LEVELS], fontsize=9)
    ax0.set_ylabel("Success rate")
    ax0.set_ylim(0, 1.05)
    ax0.grid(True, axis="y", alpha=0.3)
    for i, b in enumerate(bars):
        ax0.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{100 * success_rates[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # (0,1) Mean / median ticks (successes only)
    ax1 = fig.add_subplot(gs[0, 1])
    w = 0.35
    ax1.bar(x - w / 2, mean_ticks, w, label="mean ticks", color="coral", alpha=0.9)
    ax1.bar(x + w / 2, med_ticks, w, label="median ticks", color="seagreen", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in P_LEVELS], fontsize=9)
    ax1.set_ylabel("Ticks until score 8 (converged seeds)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Convergence speed when solved")

    # (1,:) Jitter scatter of ticks per level (handles 0 successes cleanly)
    ax2 = fig.add_subplot(gs[1, :])
    rng = np.random.default_rng(42)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_levels))
    for j, (ticks_j, c) in enumerate(zip(ticks_lists, colors)):
        if ticks_j:
            xp = (j + 1) * np.ones(len(ticks_j)) + rng.normal(0, 0.06, size=len(ticks_j))
            ax2.scatter(xp, ticks_j, alpha=0.75, s=22, c=[c], edgecolors="white", linewidths=0.3)
    ymax = max((max(t) for t in ticks_lists if t), default=1)
    ax2.set_ylim(0, min(BATCH_MAX_TICKS * 1.02, ymax * 1.15 + 1))
    for j, (ticks_j, c) in enumerate(zip(ticks_lists, colors)):
        if ticks_j and mean_ticks[j] == mean_ticks[j]:
            ax2.plot([j + 0.72, j + 1.28], [mean_ticks[j], mean_ticks[j]], color=c, linewidth=2, alpha=0.9)
        elif not ticks_j:
            ax2.text(
                j + 1,
                BATCH_MAX_TICKS * 0.45,
                f"0/{BATCH_SEED_COUNT}",
                ha="center",
                fontsize=9,
                color="crimson",
            )
    ax2.set_xticks(x + 1)
    ax2.set_xticklabels([c[0] for c in P_LEVELS], fontsize=9)
    ax2.set_xlim(0.4, n_levels + 0.6)
    ax2.set_ylabel("Ticks (successful runs)")
    ax2.set_title("Per-seed convergence time (dots) and mean (bar segment)")
    ax2.grid(True, axis="y", alpha=0.3)

    # (2,0) Heatmap: seed × p → success
    ax3 = fig.add_subplot(gs[2, 0])
    mat = np.zeros((BATCH_SEED_COUNT, n_levels))
    for j, rows in enumerate(all_rows):
        for i, (_s, ok, _t) in enumerate(rows):
            mat[i, j] = 1.0 if ok else 0.0
    im = ax3.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax3.set_yticks([0, BATCH_SEED_COUNT - 1])
    ax3.set_yticklabels([str(BATCH_SEED_START), str(BATCH_SEED_START + BATCH_SEED_COUNT - 1)])
    ax3.set_ylabel("seed index")
    ax3.set_xticks(range(n_levels))
    ax3.set_xticklabels([f"p={p_numeric[k]:.3g}" if p_numeric[k] else "0" for k in range(n_levels)], rotation=25, ha="right")
    ax3.set_title("Success matrix (green=yes)")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # (2,1) Score trajectories for one seed, all p
    ax4 = fig.add_subplot(gs[2, 1])
    cmap = plt.cm.plasma(np.linspace(0.15, 0.9, n_levels))
    for k, ((_label, _p, esc, mask), c) in enumerate(zip(P_LEVELS, cmap)):
        sc = trajectory_scores(esc, mask, TRAJECTORY_SEED, TRAJ_STEPS)
        ax4.plot(sc, color=c, linewidth=1.1, alpha=0.9, label=_label.replace("\n", " "))
    ax4.set_xlabel("Training tick")
    ax4.set_ylabel("Global score")
    ax4.set_ylim(-0.3, MAX_SCORE + 0.5)
    ax4.legend(loc="lower right", fontsize=6)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f"Single seed 0x{TRAJECTORY_SEED:04X}: score vs time")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
