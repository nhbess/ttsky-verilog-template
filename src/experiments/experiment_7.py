#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 7 — systematic heterogeneous p_i vs fixed baseline.

Building on experiment 6: global schedules and scalar sigma-feedback did not beat a
simple fixed p ~ 1/4; plastic-mapped per-gate p_i helped on speed. Here we compare
**local** rules explicitly:

1. **Fixed** ``plateau_mask=3`` (~1/4 tie accept) — strong baseline.
2. **Plastic-mapped p_i:** ``plastic[u]`` maps to masks 15,7,3,1 (unstable gate → higher p).
3. **Recent-success exploit:** gate ``u`` uses low p (mask 15) if it had an **accepted**
   strictly improving trial within the last ``exploit_window`` global COMPAREs; else mask 3.
4. **Mixed:** ``max(plastic_mask(u), success_mask(u))`` — favor exploitation if *either*
   plastic is low or the gate recently improved (larger mask = lower p).

Two scenarios (push toward experiment-4-style hardness):

* **Easy:** N=3 M=1 parity (same as exp 6).
* **Hard:** N=4 M=2 **random** targets (table seed per trajectory seed).

Run from repo root:
  python src/experiments/experiment_7.py

Output: src/results/experiment_7_heterogeneous_p.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import experiment_4_worker as e4w  # noqa: E402
import tt_chain_learner_spec as chain  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_7_heterogeneous_p.png"

BATCH_SEED_START = 1

# (label, n_in, n_out, target_kind, table_seed or None for random-per-seed, max_ticks, n_seeds)
SCENARIOS: List[Tuple[str, int, int, str, int | None, int, int]] = [
    ("parity 3×1", 3, 1, "parity", 0, 42_000, 24),
    ("random 4×2", 4, 2, "random", None, 88_000, 12),
]


class PlasticMappedPlateauChain(chain.TTChainLearner):
    """Higher plasticity → smaller mask → higher p (more tie exploration)."""

    _PLASTIC_MASK: Tuple[int, int, int, int] = (15, 7, 3, 1)

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        pi = min(3, max(0, int(self.plastic[u])))
        return self._PLASTIC_MASK[pi]


class RecentSuccessExploitPlateauChain(chain.TTChainLearner):
    """
    Gates that recently produced an accepted strict improvement use low p (mask 15);
    others use mask 3. Counts global COMPARE index since last such event on that gate.
    """

    def __init__(self, *, exploit_window: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exploit_window = exploit_window
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve: List[int] = [-1_000_000] * n

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def _after_compare_hook(self) -> None:
        self._compare_n += 1
        u = self.unit_sel
        if self._gates_mutated_last_tick and self.new_score > self.old_score:
            self._last_accept_improve[u] = self._compare_n

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        if self._compare_n - self._last_accept_improve[u] <= self.exploit_window:
            return 15
        return 3


class MixedPlasticSuccessPlateauChain(chain.TTChainLearner):
    """max(plastic mask, success mask): exploit if plastic is calm OR recent win on gate."""

    _PLASTIC_MASK: Tuple[int, int, int, int] = (15, 7, 3, 1)

    def __init__(self, *, exploit_window: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exploit_window = exploit_window
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def reset(self, seed: int = 0xACE1) -> None:
        super().reset(seed)
        n = len(self.gate)
        self._compare_n = 0
        self._last_accept_improve = [-1_000_000] * n

    def _after_compare_hook(self) -> None:
        self._compare_n += 1
        u = self.unit_sel
        if self._gates_mutated_last_tick and self.new_score > self.old_score:
            self._last_accept_improve[u] = self._compare_n

    def _plateau_mask_for_compare(self) -> int:
        u = self.unit_sel
        pi = min(3, max(0, int(self.plastic[u])))
        mp = self._PLASTIC_MASK[pi]
        ms = 15 if self._compare_n - self._last_accept_improve[u] <= self.exploit_window else 3
        return max(mp, ms)


PolicyFactory = Callable[[int, int, str, int, int], Callable[[int], chain.TTChainLearner]]


def _target(n_in: int, n_out: int, kind: str, table_seed: int) -> List[List[int]]:
    return chain.make_truth_tables(kind, n_in, n_out, table_seed=table_seed)


def make_fixed_baseline(
    n_in: int, n_out: int, kind: str, table_seed: int
) -> Callable[[int], chain.TTChainLearner]:
    tgt = _target(n_in, n_out, kind, table_seed)

    def _f(seed: int) -> chain.TTChainLearner:
        return chain.TTChainLearner(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )

    return _f


def make_plastic_mapped(
    n_in: int, n_out: int, kind: str, table_seed: int
) -> Callable[[int], chain.TTChainLearner]:
    tgt = _target(n_in, n_out, kind, table_seed)

    def _f(seed: int) -> PlasticMappedPlateauChain:
        return PlasticMappedPlateauChain(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )

    return _f


def make_success_exploit(
    n_in: int, n_out: int, kind: str, table_seed: int
) -> Callable[[int], chain.TTChainLearner]:
    tgt = _target(n_in, n_out, kind, table_seed)

    def _f(seed: int) -> RecentSuccessExploitPlateauChain:
        return RecentSuccessExploitPlateauChain(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
            exploit_window=8,
        )

    return _f


def make_mixed(
    n_in: int, n_out: int, kind: str, table_seed: int
) -> Callable[[int], chain.TTChainLearner]:
    tgt = _target(n_in, n_out, kind, table_seed)

    def _f(seed: int) -> MixedPlasticSuccessPlateauChain:
        return MixedPlasticSuccessPlateauChain(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
            exploit_window=8,
        )

    return _f


POLICIES: List[Tuple[str, PolicyFactory]] = [
    (r"fixed $p\!\sim\!1/4$", lambda n, m, k, ts: make_fixed_baseline(n, m, k, ts)),
    (r"plastic $p_i$", lambda n, m, k, ts: make_plastic_mapped(n, m, k, ts)),
    (r"success-exploit $p_i$", lambda n, m, k, ts: make_success_exploit(n, m, k, ts)),
    (r"mixed plastic+success", lambda n, m, k, ts: make_mixed(n, m, k, ts)),
]


def _table_seed(kind: str, n_in: int, n_out: int, traj_seed: int, fixed: int | None) -> int:
    if fixed is not None:
        return fixed
    return e4w.table_seed_for(kind, n_in, n_out, traj_seed)


def run_scenario(
    scen_label: str,
    n_in: int,
    n_out: int,
    kind: str,
    table_seed_fixed: int | None,
    max_ticks: int,
    n_seeds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns success_rate[policy], mean_ticks_ok[policy] (nan if no successes)."""
    n_pol = len(POLICIES)
    sr = np.zeros(n_pol)
    mt = np.full(n_pol, np.nan)
    for pi, (_pname, pfactory) in enumerate(POLICIES):
        ts0 = _table_seed(kind, n_in, n_out, BATCH_SEED_START, table_seed_fixed)
        factory = pfactory(n_in, n_out, kind, ts0)
        ticks_ok: List[int] = []
        ok_ct = 0
        for i in range(n_seeds):
            s = BATCH_SEED_START + i
            ts = _table_seed(kind, n_in, n_out, s, table_seed_fixed)
            if ts != ts0:
                factory = pfactory(n_in, n_out, kind, ts)
                ts0 = ts
            m = factory(s)
            m.reset(seed=s & 0xFFFF or 0xACE1)
            ok, ticks = m.run_until_perfect(max_ticks)
            if ok:
                ok_ct += 1
                ticks_ok.append(ticks)
        sr[pi] = ok_ct / n_seeds
        if ticks_ok:
            mt[pi] = float(mean(ticks_ok))
    return sr, mt


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    n_pol = len(POLICIES)
    policy_short = ["fixed", "plastic", "success", "mixed"]

    all_sr: List[np.ndarray] = []
    all_mt: List[np.ndarray] = []

    for lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds in SCENARIOS:
        print(f"Running scenario: {lab} ({n_seeds} seeds, {max_ticks} ticks)...")
        sr, mt = run_scenario(lab, n_in, n_out, kind, ts_fix, max_ticks, n_seeds)
        all_sr.append(sr)
        all_mt.append(mt)

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11, 3.8 * len(SCENARIOS)), constrained_layout=True)
    if len(SCENARIOS) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Experiment 7 — heterogeneous $p_i$ vs fixed baseline (two scenarios)",
        fontsize=12,
    )

    x = np.arange(n_pol)
    w = 0.55
    for si, (row, (slab, *_)) in enumerate(zip(axes, SCENARIOS)):
        ax0, ax1 = row[0], row[1]
        ax0.bar(x, all_sr[si], w, color="steelblue", edgecolor="white")
        ax0.set_xticks(x)
        ax0.set_xticklabels(policy_short, rotation=15, ha="right")
        ax0.set_ylabel("success rate")
        ax0.set_ylim(0, 1.05)
        ax0.set_title(f"{slab}")
        ax0.grid(True, axis="y", alpha=0.3)
        for i in range(n_pol):
            ax0.text(i, all_sr[si, i] + 0.03, f"{100 * all_sr[si, i]:.0f}%", ha="center", fontsize=8)

        ax1.bar(x, np.nan_to_num(all_mt[si], nan=0.0), w, color="coral", edgecolor="white")
        ax1.set_xticks(x)
        ax1.set_xticklabels(policy_short, rotation=15, ha="right")
        ax1.set_ylabel("mean ticks (successes)")
        ax1.set_title(f"{slab} — speed when solved")
        ax1.grid(True, axis="y", alpha=0.3)
        ymax = float(np.nanmax(all_mt[si])) if np.any(np.isfinite(all_mt[si])) else 1.0
        ax1.set_ylim(0, ymax * 1.15 + 1)
        for i in range(n_pol):
            v = all_mt[si, i]
            txt = f"{v:.0f}" if v == v else "—"
            ax1.text(i, (v if v == v else ymax * 0.4) + ymax * 0.02, txt, ha="center", fontsize=8)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)

    print(f"Wrote {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
