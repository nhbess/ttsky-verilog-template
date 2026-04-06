#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 4 — scaling: fixed chain topology, vary inputs N and outputs M.

Search space per output line is truth tables in {0,1}^{2^N}; the generator is the
skew chain from ref/tt_chain_learner_spec.py (parity3 topology when N=3).

Compare structured targets (parity) vs random truth tables. Plateau policy matches
experiment 3 sweet spot: plateau_mask=3 (~1/4 tie acceptance).

Metrics: success rate and mean ticks (successful runs only) over a batch of seeds.
Each random target uses a deterministic table_seed derived from the run seed so
runs are reproducible. Tick caps grow slightly with problem size; skipped (N,M)
pairs (e.g. N=4, M=3) appear as grey in heatmaps to keep batch runtime reasonable.

Run from repo root:
  python src/experiments/experiment_4.py

Output: src/results/experiment_4_scaling.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_chain_learner_spec as spec  # noqa: E402

# --- knobs ---
N_LIST = [2, 3, 4]
# M=3 with N=4 is 48 score cells and very slow in pure Python; add M=3 only for smaller N.
M_BY_N = {2: [1, 2, 3], 3: [1, 2, 3], 4: [1, 2]}
BATCH_SEED_START = 1
# Failed runs burn the full tick budget — keep small so the sweep finishes quickly.
BATCH_SEED_COUNT = 8


def max_ticks_for(n_in: int, n_out: int) -> int:
    cells = (1 << n_in) * n_out
    # Flat modest cap: cost per trial scales with `cells` (two full passes per compare).
    return min(52_000, int(26_000 + 650 * cells))


def table_seed_for(kind: str, n_in: int, n_out: int, learner_seed: int) -> int:
    if kind != "random":
        return 0
    return (learner_seed * 1_000_003 + n_in * 97 + n_out * 1_009) & 0x7FFFFFFF


def run_batch(
    n_in: int, n_out: int, kind: str
) -> Tuple[float, List[int], int]:
    cap = max_ticks_for(n_in, n_out)
    ok_list: List[int] = []
    for s in range(BATCH_SEED_START, BATCH_SEED_START + BATCH_SEED_COUNT):
        ts = table_seed_for(kind, n_in, n_out, s)
        tgt = spec.make_truth_tables(kind, n_in, n_out, table_seed=ts)
        m = spec.TTChainLearner(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )
        m.reset(seed=s & 0xFFFF or 0xACE1)
        ok, ticks = m.run_until_perfect(cap)
        if ok:
            ok_list.append(ticks)
    rate = len(ok_list) / BATCH_SEED_COUNT
    return rate, ok_list, cap


def main() -> int:
    try:
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
    except ImportError:
        print("pip install matplotlib numpy", file=sys.stderr)
        return 1

    RESULTS = SRC_ROOT / "results"
    OUT = RESULTS / "experiment_4_scaling.png"

    m_cols = [1, 2, 3]
    n_n, n_m = len(N_LIST), len(m_cols)
    rate_p = np.full((n_n, n_m), np.nan)
    rate_r = np.full((n_n, n_m), np.nan)
    mean_ticks_p = np.full((n_n, n_m), np.nan)
    mean_ticks_r = np.full((n_n, n_m), np.nan)
    caps = np.full((n_n, n_m), np.nan)

    for i, n_in in enumerate(N_LIST):
        for j, n_out in enumerate(m_cols):
            if n_out not in M_BY_N[n_in]:
                continue
            cap = max_ticks_for(n_in, n_out)
            caps[i, j] = cap
            rp, ticks_p, _ = run_batch(n_in, n_out, "parity")
            rr, ticks_r, _ = run_batch(n_in, n_out, "random")
            rate_p[i, j] = rp
            rate_r[i, j] = rr
            if ticks_p:
                mean_ticks_p[i, j] = float(np.mean(ticks_p))
            if ticks_r:
                mean_ticks_r[i, j] = float(np.mean(ticks_r))

    cap_max = int(np.nanmax(caps))

    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    fig.suptitle(
        "Experiment 4 — fixed chain topology: scale N inputs × M outputs\n"
        f"{BATCH_SEED_COUNT} seeds/cell; plateau p≈1/4; tick cap ≤ {cap_max:,} (grey = cell skipped)",
        fontsize=11,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])

    def stamp_pct(ax, i: int, j: int, data: np.ndarray) -> None:
        t = ax.text(
            j,
            i,
            f"{100 * data[i, j]:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    def annotate_rate_heatmap(ax, data: np.ndarray, title: str, cmap_name: str) -> None:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#d0d0d0")
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_m))
        ax.set_yticks(range(n_n))
        ax.set_xticklabels([str(m) for m in m_cols])
        ax.set_yticklabels([str(n) for n in N_LIST])
        ax.set_xlabel("Outputs M")
        ax.set_ylabel("Inputs N")
        ax.set_title(title)
        for i in range(n_n):
            for j in range(n_m):
                if not np.isfinite(data[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="0.35")
                else:
                    stamp_pct(ax, i, j, data)
        fig.colorbar(im, ax=ax, fraction=0.046, label="success rate")

    def annotate_ticks_heatmap(ax, data: np.ndarray, title: str, cmap_name: str) -> None:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#d0d0d0")
        masked = np.ma.masked_invalid(data)
        vmax_tick = float(np.nanmax(caps))
        dmin = float(np.nanmin(data)) if np.any(np.isfinite(data)) else 100.0
        vmin_tick = max(20.0, dmin * 0.4)
        if vmin_tick >= vmax_tick:
            vmax_tick = vmin_tick * 1.5
        im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=LogNorm(vmin=vmin_tick, vmax=vmax_tick))
        ax.set_xticks(range(n_m))
        ax.set_yticks(range(n_n))
        ax.set_xticklabels([str(m) for m in m_cols])
        ax.set_yticklabels([str(n) for n in N_LIST])
        ax.set_xlabel("Outputs M")
        ax.set_ylabel("Inputs N")
        ax.set_title(title)
        for i in range(n_n):
            for j in range(n_m):
                v = data[i, j]
                if v == v:
                    t = ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white")
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="0.35")
        fig.colorbar(im, ax=ax, fraction=0.046, label="mean ticks (successes)")

    ax0 = fig.add_subplot(gs[0, 0])
    annotate_rate_heatmap(ax0, rate_p, "Structured: N-input parity (all M lines)", "viridis")

    ax1 = fig.add_subplot(gs[0, 1])
    annotate_rate_heatmap(ax1, rate_r, "Random: independent truth tables / line", "magma")

    ax_mt0 = fig.add_subplot(gs[1, 0])
    annotate_ticks_heatmap(
        ax_mt0,
        mean_ticks_p,
        "Mean convergence ticks (parity targets, successes only)",
        cmap_name="cividis",
    )

    ax_mt1 = fig.add_subplot(gs[1, 1])
    annotate_ticks_heatmap(
        ax_mt1,
        mean_ticks_r,
        "Mean convergence ticks (random targets, successes only)",
        cmap_name="inferno",
    )

    ax2 = fig.add_subplot(gs[2, 0])
    mid_m = m_cols.index(2)
    ax2.plot(N_LIST, rate_p[:, mid_m], "o-", label="parity (M=2)", color="seagreen", linewidth=2)
    ax2.plot(N_LIST, rate_r[:, mid_m], "s--", label="random (M=2)", color="darkorange", linewidth=2)
    ax2.set_xlabel("Inputs N")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(-0.05, 1.08)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Slice M=2: structured vs random")

    ax3 = fig.add_subplot(gs[2, 1])
    n_idx = N_LIST.index(3)
    ax3.plot(m_cols, rate_p[n_idx, :], "o-", label="parity (N=3)", color="steelblue", linewidth=2)
    ax3.plot(m_cols, rate_r[n_idx, :], "s--", label="random (N=3)", color="crimson", linewidth=2)
    ax3.set_xlabel("Outputs M")
    ax3.set_ylabel("Success rate")
    ax3.set_ylim(-0.05, 1.08)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("Slice N=3: more output lines")

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
