#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Experiment 5 — training-time statistics vs tie probability p (parity3 / chain).

Uses the same p levels as experiment 3 on N=3 M=1 parity (chain ref). During
training we record, at every COMPARE (one proposed mutation evaluated):

  * trial delta  Δ_t = new_score - old_score  (score on all minterms, same as FSM)
  * global score after the trial  S_t = score_current_gates()

**Figure 1 (criticality on the trial process):**

  * P(Δ), mean, Var(Δ)=χ, skew(Δ), I = E[max(Δ,0)]
  * P(Δ>0) and P(Δ<0) vs p (trial-level fractions; avoids σ=P(up)/P(down) blowups)

**Figure 2 (dynamics on non-monotone observables):**

Global S(t) is almost monotone under accept/reject, so autocorrelation and
“cascades until S drops” on S are misleading. Instead:

  * Autocorrelation of the **centered trial series** Δ_t (same length as trials).
  * **Cascade A:** run lengths of consecutive **Δ_t > 0** (improving proposals).
  * **Cascade B:** run lengths of consecutive **δS_t > 0** with
    δS_t = S_{t+1} - S_t (accepted global score steps).

Complements ``study_sim.probe_landscape`` (frozen θ) — this is **online** during
training.

Run from repo root:
  python src/experiments/experiment_5.py

Outputs:
  src/results/experiment_5_criticality.png
  src/results/experiment_5_trial_dynamics.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ref"))

import tt_chain_learner_spec as chain  # noqa: E402

RESULTS = SRC_ROOT / "results"
OUT_FIG = RESULTS / "experiment_5_criticality.png"
OUT_TRIAL = RESULTS / "experiment_5_trial_dynamics.png"

# Match experiment 3 p sweep (parity3 / chain plateau policy)
P_LEVELS: List[Tuple[str, float, bool, int]] = [
    ("strict", 0.0, False, 0),
    (r"$p=\frac{1}{16}$", 1 / 16, True, 15),
    (r"$p=\frac{1}{8}$", 1 / 8, True, 7),
    (r"$p=\frac{1}{4}$", 1 / 4, True, 3),
    (r"$p=\frac{1}{2}$", 1 / 2, True, 1),
]

TRAJ_SEED = 0xACE1
MAX_TICKS = 45_000
N_IN = 3
N_OUT = 1


class InstrumentedChainLearner(chain.TTChainLearner):
    """Records one Δ and global score after each COMPARE (one mutation trial)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.trial_deltas: List[int] = []
        self.global_scores: List[int] = []
        self.accepted: List[bool] = []

    def tick(self, train_enable: bool = True) -> None:
        entering_compare = train_enable and self.fsm == chain.FsmState.COMPARE
        super().tick(train_enable)
        if entering_compare:
            d = self.new_score - self.old_score
            self.trial_deltas.append(d)
            self.accepted.append(self._gates_mutated_last_tick)
            self.global_scores.append(self.score_current_gates())


def _skew(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return float("nan")
    m = a.mean()
    s = a.std(ddof=0)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((a - m) / s) ** 3))


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of centered x; length max_lag+1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(max_lag + 1, np.nan)
    if n < 4:
        out[0] = 1.0
        return out
    x = x - x.mean()
    var = float(np.dot(x, x)) / n
    if var < 1e-12:
        out[0] = 1.0
        return out
    for lag in range(0, min(max_lag + 1, n)):
        out[lag] = float(np.dot(x[: n - lag], x[lag:]) / (n * var))
    return out


def _run_lengths_consecutive_true(mask: np.ndarray) -> List[int]:
    """Maximal contiguous True run lengths (1D)."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    out: List[int] = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        out.append(j - i)
        i = j
    return out


def cascade_delta_positive(d: np.ndarray) -> List[int]:
    """Run lengths of consecutive trials with Δ > 0."""
    return _run_lengths_consecutive_true(d > 0)


def cascade_dS_positive(scores: List[int]) -> List[int]:
    """Run lengths of consecutive trials with S_{t+1} > S_t."""
    if len(scores) < 2:
        return []
    s = np.array(scores, dtype=int)
    ds = np.diff(s)
    return _run_lengths_consecutive_true(ds > 0)


def run_trajectory(plateau_escape: bool, plateau_mask: int, seed: int) -> InstrumentedChainLearner:
    tgt = chain.make_truth_tables("parity", N_IN, N_OUT, table_seed=0)
    m = InstrumentedChainLearner(
        n_in=N_IN,
        n_out=N_OUT,
        target=tgt,
        plateau_escape=plateau_escape,
        plateau_mask=plateau_mask,
    )
    m.reset(seed=seed & 0xFFFF or 0xACE1)
    for _ in range(MAX_TICKS):
        if m.score_current_gates() == m.max_score:
            break
        m.tick()
    return m


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    n_p = len(P_LEVELS)
    p_numeric = [cfg[1] for cfg in P_LEVELS]
    labels = [cfg[0] for cfg in P_LEVELS]

    variances: List[float] = []
    skews: List[float] = []
    info_gains: List[float] = []
    p_up: List[float] = []
    p_down: List[float] = []
    all_deltas: List[np.ndarray] = []

    max_lag = 80
    ac_lags = np.arange(0, max_lag + 1)
    ac_delta_curves: List[np.ndarray] = []
    cascades_dpos: List[List[int]] = []
    cascades_ds: List[List[int]] = []

    for _lab, _p, esc, mask in P_LEVELS:
        m = run_trajectory(esc, mask, TRAJ_SEED)
        d = np.array(m.trial_deltas, dtype=int)
        all_deltas.append(d)
        n = int(d.size)
        variances.append(float(d.var()) if n else float("nan"))
        skews.append(_skew(d))
        info_gains.append(float(np.maximum(d, 0).mean()) if n else float("nan"))
        if n:
            p_up.append(float((d > 0).sum()) / n)
            p_down.append(float((d < 0).sum()) / n)
        else:
            p_up.append(float("nan"))
            p_down.append(float("nan"))
        ac_delta_curves.append(_autocorr(d.astype(float), max_lag))
        cascades_dpos.append(cascade_delta_positive(d))
        cascades_ds.append(cascade_dS_positive(m.global_scores))

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.0])

    fig.suptitle(
        "Experiment 5 — training statistics vs tie-accept p (N=3 M=1 parity, chain ref)\n"
        f"seed=0x{TRAJ_SEED:04X}, max_ticks={MAX_TICKS:,}; Δ = trial new_score - old_score",
        fontsize=11,
    )

    x = np.arange(n_p)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x, variances, "o-", color="C0", label=r"$\chi=\mathrm{Var}(\Delta)$")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{p:.3g}" if p else "0" for p in p_numeric], rotation=20, ha="right")
    ax0.set_xlabel(r"nominal $p$")
    ax0.set_ylabel(r"$\mathrm{Var}(\Delta)$")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax0b = fig.add_subplot(gs[0, 1])
    ax0b.plot(x, info_gains, "s-", color="C1", label=r"$I=\mathbb{E}[\max(\Delta,0)]$")
    ax0b.set_xticks(x)
    ax0b.set_xticklabels([f"{p:.3g}" if p else "0" for p in p_numeric], rotation=20, ha="right")
    ax0b.set_xlabel(r"nominal $p$")
    ax0b.set_ylabel("info gain proxy")
    ax0b.grid(True, alpha=0.3)
    ax0b.legend()

    ax1 = fig.add_subplot(gs[1, :])
    d_min = min(int(d.min()) for d in all_deltas if d.size) if any(d.size for d in all_deltas) else -4
    d_max = max(int(d.max()) for d in all_deltas if d.size) if any(d.size for d in all_deltas) else 4
    bins = np.arange(d_min, d_max + 2) - 0.5
    for j, d in enumerate(all_deltas):
        if d.size == 0:
            continue
        ax1.hist(
            d,
            bins=bins,
            alpha=0.45,
            label=labels[j].replace("\n", " "),
            density=True,
        )
    ax1.set_xlabel(r"$\Delta$ (trial score change)")
    ax1.set_ylabel("density")
    ax1.set_title(r"$P(\Delta)$ during training (overlaid by $p$)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(x, skews, "^-", color="C2", label="skew($\\Delta$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p:.3g}" if p else "0" for p in p_numeric], rotation=20, ha="right")
    ax2.set_xlabel(r"nominal $p$")
    ax2.set_ylabel("skew")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax2b = fig.add_subplot(gs[2, 1])
    ax2b.plot(x, p_up, "d-", color="C3", label=r"$P(\Delta>0)$")
    ax2b.plot(x, p_down, "v--", color="C4", label=r"$P(\Delta<0)$")
    ax2b.set_xticks(x)
    ax2b.set_xticklabels([f"{p:.3g}" if p else "0" for p in p_numeric], rotation=20, ha="right")
    ax2b.set_xlabel(r"nominal $p$")
    ax2b.set_ylabel("fraction of trials")
    ax2b.set_ylim(-0.05, 1.05)
    ax2b.grid(True, alpha=0.3)
    ax2b.legend()

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)

    # Trial-based dynamics (not raw S(t) — avoids monotone-drift artifact)
    fig2, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    fig2.suptitle(
        "Experiment 5 — trial process: autocorr($\\Delta_t$) and cascade run lengths",
        fontsize=11,
    )

    axa, axb, axc = axes
    for j, ac in enumerate(ac_delta_curves):
        axa.plot(ac_lags, ac, label=labels[j].replace("\n", " "), alpha=0.85)
    axa.set_xlabel(r"lag $\tau$ (trials)")
    axa.set_ylabel(r"normalized $C(\tau)$")
    axa.set_title(r"Autocorrelation of $\Delta_t$ (centered)")
    axa.legend(loc="upper right", fontsize=6)
    axa.grid(True, alpha=0.3)

    for j, casc in enumerate(cascades_dpos):
        if not casc:
            continue
        axb.hist(
            casc,
            bins=min(25, max(4, len(set(casc)))),
            alpha=0.4,
            label=labels[j].replace("\n", " "),
        )
    axb.set_xlabel("run length (consecutive $\\Delta_t > 0$)")
    axb.set_ylabel("count")
    axb.set_title("Cascades: improving trial proposals")
    axb.legend(loc="upper right", fontsize=6)
    axb.grid(True, alpha=0.3)

    for j, casc in enumerate(cascades_ds):
        if not casc:
            continue
        axc.hist(
            casc,
            bins=min(25, max(4, len(set(casc)))),
            alpha=0.4,
            label=labels[j].replace("\n", " "),
        )
    axc.set_xlabel("run length (consecutive $S_{t+1}>S_t$)")
    axc.set_ylabel("count")
    axc.set_title("Cascades: global score increases")
    axc.legend(loc="upper right", fontsize=6)
    axc.grid(True, alpha=0.3)

    fig2.savefig(OUT_TRIAL, dpi=150)
    plt.close(fig2)

    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_TRIAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
