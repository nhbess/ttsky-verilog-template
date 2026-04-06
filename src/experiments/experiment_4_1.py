#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 4.1 — mutation landscape: P(Δ>0), P(Δ=0), P(Δ<0) vs score regime.

Uses the same chain topology as experiment 4 (ref/tt_chain_learner_spec.py).
Training runs with plateau policy (mask 3) until the global score first reaches
~50%, ~75%, or ~90% of max; then learning is frozen and we draw random mutations
(one gate, random 4-bit candidate with the same tie-break as PROPOSE when equal).

Each mutation yields Δ = score(θ') - score(θ); we tally up / flat / down over
many samples without applying any accept step.

Outputs: src/results/experiment_4_1_landscape.png

Run from repo root:
  python src/experiments/experiment_4_1.py

Optional: ``pip install tqdm`` for a progress bar during the sweep.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TypeVar

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_4_worker as e4w  # noqa: E402
import tt_chain_learner_spec as spec  # noqa: E402

N_LIST = [2, 3, 4]
M_BY_N = {2: [1, 2, 3], 3: [1, 2, 3], 4: [1, 2]}
SCORE_FRACS = (0.50, 0.75, 0.90)
MUT_SAMPLES = 120
TRAJECTORY_SEEDS = 6
TRAJ_SEED_START = 1
BURN_CAP = 55_000


def burn_in_max_ticks(n_in: int, n_out: int) -> int:
    cells = (1 << n_in) * n_out
    return min(90_000, int(BURN_CAP + 800 * cells))


def threshold_score(max_score: int, frac: float) -> int:
    """Minimum integer score to count as having reached ``frac`` of optimum."""
    t = int(round(frac * max_score))
    return max(1, min(max_score - 1, t))


def score_with_trial(m: spec.TTChainLearner) -> int:
    s = 0
    for idx in range(m.nrows):
        xs = spec.row_bits(idx, m.n_in)
        ys = m._forward_trial_all(xs)
        for om in range(m.n_out):
            if ys[om] == m.target[om][idx]:
                s += 1
    return s


def mutation_outcome_counts(m: spec.TTChainLearner, rng: random.Random, n_mut: int) -> Tuple[int, int, int]:
    """Freeze gates; count up / flat / down over ``n_mut`` random proposals."""
    up = flat = down = 0
    n_g = len(m.gate)
    old_score = m.score_current_gates()
    for _ in range(n_mut):
        u = rng.randrange(n_g)
        cand = rng.randint(0, 15)
        if cand == m.gate[u]:
            cand ^= 0x1
        m.unit_sel = u
        m.trial_gate = cand
        new = score_with_trial(m)
        d = new - old_score
        if d > 0:
            up += 1
        elif d == 0:
            flat += 1
        else:
            down += 1
    return up, flat, down


def burn_in_then_sample(
    n_in: int,
    n_out: int,
    kind: str,
    traj_seed: int,
    frac: float,
) -> Tuple[bool, int, int, int, int, int]:
    """
    Returns (ok, up, flat, down, final_score, max_score).
    ok False if burn-in never reaches threshold (no sample taken → zeros).
    """
    ts = e4w.table_seed_for(kind, n_in, n_out, traj_seed)
    tgt = spec.make_truth_tables(kind, n_in, n_out, table_seed=ts)
    m = spec.TTChainLearner(
        n_in=n_in,
        n_out=n_out,
        target=tgt,
        plateau_escape=True,
        plateau_mask=3,
    )
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    mx = m.max_score
    thr = threshold_score(mx, frac)
    cap = burn_in_max_ticks(n_in, n_out)

    reached = False
    if m.score_current_gates() >= thr:
        reached = True
    else:
        for _ in range(cap):
            m.tick()
            if m.score_current_gates() >= thr:
                reached = True
                break

    if not reached:
        return False, 0, 0, 0, m.score_current_gates(), mx

    mut_rng = random.Random((traj_seed << 16) ^ int(100 * frac) ^ n_in ^ (n_out << 8))
    u, f, d = mutation_outcome_counts(m, mut_rng, MUT_SAMPLES)
    return True, u, f, d, m.score_current_gates(), mx


def _tqdm(it: Iterable[T], **kw: Any) -> Iterable[T]:
    try:
        from tqdm import tqdm as real_tqdm
    except ImportError:
        return it
    return real_tqdm(it, **kw)


def valid_cells() -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for n in N_LIST:
        for m in M_BY_N[n]:
            cells.append((n, m))
    return cells


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    RESULTS = SRC_ROOT / "results"
    OUT = RESULTS / "experiment_4_1_landscape.png"

    kinds = ("parity", "random")
    # kind -> (n,m) -> frac -> {'up','flat','down','ok'}
    agg: Dict[str, Dict[Tuple[int, int], Dict[float, Dict[str, int]]]] = {
        k: {(n, m): {f: {"up": 0, "flat": 0, "down": 0, "ok": 0} for f in SCORE_FRACS} for n, m in valid_cells()}
        for k in kinds
    }

    cells = valid_cells()
    jobs = itertools.product(
        cells,
        kinds,
        SCORE_FRACS,
        range(TRAJ_SEED_START, TRAJ_SEED_START + TRAJECTORY_SEEDS),
    )
    for (n_in, n_out), kind, frac, s in _tqdm(
        jobs,
        desc="experiment 4.1",
        unit="job",
        total=len(cells) * len(kinds) * len(SCORE_FRACS) * TRAJECTORY_SEEDS,
    ):
        ok, up, flat, down, _fs, _mx = burn_in_then_sample(
            n_in, n_out, kind, s, frac
        )
        slot = agg[kind][(n_in, n_out)][frac]
        if ok:
            slot["ok"] += 1
            slot["up"] += up
            slot["flat"] += flat
            slot["down"] += down

    n_cells = len(cells)
    fig, axes = plt.subplots(
        len(kinds),
        n_cells,
        figsize=(2.4 * n_cells, 5.2),
        squeeze=False,
        constrained_layout=True,
    )

    frac_labels = [f"{int(100 * f)}%" for f in SCORE_FRACS]
    colors = ("#2ca02c", "#ffbb78", "#d62728")

    for row, kind in enumerate(kinds):
        for col, (n_in, n_out) in enumerate(cells):
            ax = axes[row][col]
            for fi, frac in enumerate(SCORE_FRACS):
                slot = agg[kind][(n_in, n_out)][frac]
                denom = slot["up"] + slot["flat"] + slot["down"]
                if denom == 0:
                    ax.text(fi, 0.5, "—", ha="center", fontsize=10, color="0.45")
                    continue
                pu = slot["up"] / denom
                pf = slot["flat"] / denom
                pd = slot["down"] / denom
                b = 0.0
                ax.bar(
                    fi,
                    pu,
                    bottom=b,
                    width=0.65,
                    color=colors[0],
                    edgecolor="white",
                    linewidth=0.4,
                )
                b += pu
                ax.bar(
                    fi,
                    pf,
                    bottom=b,
                    width=0.65,
                    color=colors[1],
                    edgecolor="white",
                    linewidth=0.4,
                )
                b += pf
                ax.bar(
                    fi,
                    pd,
                    bottom=b,
                    width=0.65,
                    color=colors[2],
                    edgecolor="white",
                    linewidth=0.4,
                )

            ax.set_xticks(range(len(SCORE_FRACS)))
            ax.set_xticklabels(frac_labels, fontsize=8)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("score target (burn-in)", fontsize=8)
            hits = [agg[kind][(n_in, n_out)][f]["ok"] for f in SCORE_FRACS]
            ax.set_title(
                f"N={n_in} M={n_out} — {kind}\nhits {hits[0]}/{TRAJECTORY_SEEDS}, "
                f"{hits[1]}/{TRAJECTORY_SEEDS}, {hits[2]}/{TRAJECTORY_SEEDS} "
                f"(50/75/90%)",
                fontsize=7,
            )
            ax.grid(True, axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel("probability", fontsize=9)
            else:
                ax.set_ylabel("")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[0], label=r"$P_{\mathrm{up}}$"),
        plt.Rectangle((0, 0), 1, 1, fc=colors[1], label=r"$P_{\mathrm{flat}}$"),
        plt.Rectangle((0, 0), 1, 1, fc=colors[2], label=r"$P_{\mathrm{down}}$"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9, frameon=True)
    fig.suptitle(
        "Experiment 4.1 — random mutation landscape (Δ = new_score − old_score)\n"
        f"{MUT_SAMPLES} mutations/state after plateau burn-in first hits score ≥ bucket; "
        f"max {BURN_CAP}+ burn ticks/cell",
        fontsize=11,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
