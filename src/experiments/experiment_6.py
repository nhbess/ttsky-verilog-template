#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 6 — dynamic tie-accept: time-varying p(t), per-gate p_i, sigma feedback.

Baseline: N=3 M=1 parity on ``ref/tt_chain_learner_spec.TTChainLearner`` with fixed
``plateau_mask`` (experiment 3 style).

Policies (Python ref only; hooks ``_plateau_mask_for_compare`` / ``_after_compare_hook``):

1. **Fixed** p ~ 1/16 and ~ 1/4 (masks 15, 3).
2. **Scheduled p(t):** mask 15 until global score >= 50% of max, then mask 3
   (explore early, exploit later).
3. **Plastic-mapped p_i:** effective mask from ``plastic[unit_sel]`` (0→15 … 3→1):
   unstable gates see higher tie-accept rate.
4. **Sigma feedback:** rolling window over trial Δ = new_score−old_score; target
   σ = P(Δ>0)/P(Δ<0) ≈ 1 by nudging among masks {15,7,3,1}.

Run from repo root:
  python src/experiments/experiment_6.py

Output: src/results/experiment_6_dynamic_p.png
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_chain_learner_spec as chain  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_6_dynamic_p.png"

N_IN = 3
N_OUT = 1
BATCH_SEED_START = 1
BATCH_SEED_COUNT = 24
MAX_TICKS = 38_000


def parity_target() -> List[List[int]]:
    return chain.make_truth_tables("parity", N_IN, N_OUT, table_seed=0)


@dataclass
class ScheduledMaskChain(chain.TTChainLearner):
    """p(t): low tie-accept until score reaches ``switch_frac`` of max, then high."""

    schedule_mask_early: int = 15
    schedule_mask_late: int = 3
    schedule_switch_frac: float = 0.5

    def _plateau_mask_for_compare(self) -> int:
        thr = max(
            1,
            min(self.max_score - 1, int(round(self.schedule_switch_frac * self.max_score))),
        )
        if self.score_current_gates() >= thr:
            return self.schedule_mask_late
        return self.schedule_mask_early


class PlasticMappedPlateauChain(chain.TTChainLearner):
    """p_i from plasticity: plastic 0..3 → masks 15,7,3,1 (more plastic → higher p)."""

    _PLASTIC_MASK: Tuple[int, int, int, int] = (15, 7, 3, 1)

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        pi = min(3, max(0, int(self.plastic[u])))
        return self._PLASTIC_MASK[pi]


class SigmaFeedbackPlateauChain(chain.TTChainLearner):
    """Adjust ``plateau_mask`` among {15,7,3,1} from rolling σ = n_up/n_down on Δ."""

    def __init__(
        self,
        *,
        sigma_window: int = 48,
        target_sigma: float = 1.0,
        start_level_idx: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma_window = sigma_window
        self.target_sigma = target_sigma
        self._mask_levels: Tuple[int, int, int, int] = (15, 7, 3, 1)
        self._start_level_idx = start_level_idx
        self._level_idx = start_level_idx
        self._buf: deque[int] = deque()
        self.plateau_mask = self._mask_levels[self._level_idx]

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        self._buf.clear()
        self._level_idx = self._start_level_idx
        self.plateau_mask = self._mask_levels[self._level_idx]

    def _after_compare_hook(self) -> None:
        d = self.new_score - self.old_score
        self._buf.append(d)
        if len(self._buf) < self.sigma_window:
            return
        n_up = sum(1 for x in self._buf if x > 0)
        n_down = sum(1 for x in self._buf if x < 0)
        self._buf.clear()
        if n_down == 0:
            if n_up > 0:
                self._level_idx = max(0, self._level_idx - 1)
        else:
            sig = n_up / n_down
            if sig < self.target_sigma:
                self._level_idx = min(len(self._mask_levels) - 1, self._level_idx + 1)
            elif sig > self.target_sigma:
                self._level_idx = max(0, self._level_idx - 1)
        self.plateau_mask = self._mask_levels[self._level_idx]


PolicySpec = Tuple[str, Callable[[int], chain.TTChainLearner]]


def _make_fixed(mask: int) -> Callable[[int], chain.TTChainLearner]:
    def _f(seed: int) -> chain.TTChainLearner:
        return chain.TTChainLearner(
            n_in=N_IN,
            n_out=N_OUT,
            target=parity_target(),
            plateau_escape=True,
            plateau_mask=mask,
        )

    return _f


def _make_scheduled() -> Callable[[int], chain.TTChainLearner]:
    def _f(seed: int) -> ScheduledMaskChain:
        return ScheduledMaskChain(
            n_in=N_IN,
            n_out=N_OUT,
            target=parity_target(),
            plateau_escape=True,
            plateau_mask=15,
            schedule_mask_early=15,
            schedule_mask_late=3,
            schedule_switch_frac=0.5,
        )

    return _f


def _make_plastic_mapped() -> Callable[[int], chain.TTChainLearner]:
    def _f(seed: int) -> PlasticMappedPlateauChain:
        return PlasticMappedPlateauChain(
            n_in=N_IN,
            n_out=N_OUT,
            target=parity_target(),
            plateau_escape=True,
            plateau_mask=3,
        )

    return _f


def _make_sigma_feedback() -> Callable[[int], chain.TTChainLearner]:
    def _f(seed: int) -> SigmaFeedbackPlateauChain:
        return SigmaFeedbackPlateauChain(
            n_in=N_IN,
            n_out=N_OUT,
            target=parity_target(),
            plateau_escape=True,
            plateau_mask=3,
            sigma_window=48,
            target_sigma=1.0,
            start_level_idx=2,
        )

    return _f


POLICIES: List[PolicySpec] = [
    (r"fixed $p\approx\frac{1}{16}$", _make_fixed(15)),
    (r"fixed $p\approx\frac{1}{4}$", _make_fixed(3)),
    (r"scheduled: $\frac{1}{16}\!\to\!\frac{1}{4}$ @ 50\% score", _make_scheduled()),
    (r"per-gate $p_i$ from plasticity", _make_plastic_mapped()),
    (r"$\sigma$-feedback $\approx 1$", _make_sigma_feedback()),
]


def run_batch(factory: Callable[[int], chain.TTChainLearner]) -> Tuple[int, List[int], List[Tuple[int, bool, int]]]:
    ok_n = 0
    ticks_ok: List[int] = []
    rows: List[Tuple[int, bool, int]] = []
    for s in range(BATCH_SEED_START, BATCH_SEED_START + BATCH_SEED_COUNT):
        m = factory(s)
        m.reset(seed=s & 0xFFFF or 0xACE1)
        ok, ticks = m.run_until_perfect(MAX_TICKS)
        rows.append((s, ok, ticks))
        if ok:
            ok_n += 1
            ticks_ok.append(ticks)
    return ok_n, ticks_ok, rows


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    n_pol = len(POLICIES)
    success_rates: List[float] = []
    mean_ticks: List[float] = []
    med_ticks: List[float] = []
    ticks_lists: List[List[int]] = []

    for _label, factory in POLICIES:
        ok_n, ticks_ok, _rows = run_batch(factory)
        success_rates.append(ok_n / BATCH_SEED_COUNT)
        ticks_lists.append(ticks_ok)
        if ticks_ok:
            mean_ticks.append(float(mean(ticks_ok)))
            med_ticks.append(float(median(ticks_ok)))
        else:
            mean_ticks.append(float("nan"))
            med_ticks.append(float("nan"))

    x = np.arange(n_pol)
    labels = [p[0] for p in POLICIES]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        "Experiment 6 — dynamic tie-accept vs fixed p (N=3 M=1 parity, chain ref)\n"
        f"{BATCH_SEED_COUNT} seeds × {MAX_TICKS:,} ticks; scheduled = mask 15 until score ≥ 50%, then 3",
        fontsize=10,
    )

    ax0 = axes[0, 0]
    bars = ax0.bar(x, success_rates, color="steelblue", edgecolor="white")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax0.set_ylabel("success rate")
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

    ax1 = axes[0, 1]
    w = 0.35
    ax1.bar(x - w / 2, mean_ticks, w, label="mean ticks", color="coral")
    ax1.bar(x + w / 2, med_ticks, w, label="median ticks", color="seagreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("ticks (converged runs)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1, 0]
    rng = np.random.default_rng(42)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_pol))
    ymax = max((max(t) for t in ticks_lists if t), default=1)
    for j, (ticks_j, c) in enumerate(zip(ticks_lists, colors)):
        if ticks_j:
            xp = (j + 1) * np.ones(len(ticks_j)) + rng.normal(0, 0.07, size=len(ticks_j))
            ax2.scatter(xp, ticks_j, alpha=0.75, s=20, c=[c], edgecolors="white", linewidths=0.3)
        if ticks_j and mean_ticks[j] == mean_ticks[j]:
            ax2.plot([j + 0.75, j + 1.25], [mean_ticks[j], mean_ticks[j]], color=c, linewidth=2)
        elif not ticks_j:
            ax2.text(
                j + 1,
                MAX_TICKS * 0.45,
                f"0/{BATCH_SEED_COUNT}",
                ha="center",
                fontsize=9,
                color="crimson",
            )
    ax2.set_xlim(0.4, n_pol + 0.6)
    ax2.set_xticks(x + 1)
    ax2.set_xticklabels([str(i + 1) for i in range(n_pol)], fontsize=9)
    ax2.set_xlabel("policy index")
    ax2.set_ylabel("ticks")
    ax2.set_ylim(0, min(MAX_TICKS * 1.02, ymax * 1.12 + 1))
    ax2.set_title("Per-seed convergence (dots) + mean segment")
    ax2.grid(True, axis="y", alpha=0.3)

    ax3 = axes[1, 1]
    ax3.axis("off")
    ax3.text(
        0.02,
        0.98,
        "Policies (see docstring):\n"
        "  0  fixed low p (mask 15)\n"
        "  1  fixed mid p (mask 3)\n"
        "  2  scheduled p(t): 15 then 3 after 50% score\n"
        "  3  per-gate mask from plasticity counter\n"
        "  4  sigma-feedback on rolling Δ window",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)

    print(f"Wrote {OUT_FIG}")
    for i, ((lab, _), sr, mt) in enumerate(zip(POLICIES, success_rates, mean_ticks)):
        mt_s = f"{mt:.0f}" if mt == mt else "n/a"
        print(f"  [{i}] success={100 * sr:.0f}%  mean_ticks={mt_s}  {lab[:50]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
