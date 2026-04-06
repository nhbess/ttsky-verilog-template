#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Python-first study hub (behavioral refs in ``ref/``).

**Primary:** quick local studies, parameter sweeps, and ``probe_landscape`` (mutation
Δ histograms) — edit ``if __name__ == "__main__"`` section **D** first.

**Secondary (confirmation / silicon realism):** optional experiment subprocesses and
``run_model_backend`` to fire the same *logical* task on cocotb RTL
(``run_sim.py`` / ``run_sim_parity3.py``).

RTL mapping:
  xor      ↔  ``run_sim.py``
  parity3  ↔  ``run_sim_parity3.py``
  General chain (N, M, random targets) has no bundled RTL here.

Run from repo root: ``python src/study_sim.py``
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Literal, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
_REF = REPO_ROOT / "ref"
sys.path.insert(0, str(_REF))

import experiment_4_worker as e4w  # noqa: E402
import tt_chain_learner_spec as chain  # noqa: E402
import tt_parity3_spec as parity3  # noqa: E402
import tt_xor_learner_spec as xor  # noqa: E402

# Short name → path under repo root (optional batch plots).
EXPERIMENTS: Dict[str, str] = {
    "1": "src/experiments/experiment_1.py",
    "2": "src/experiments/experiment_2.py",
    "3": "src/experiments/experiment_3.py",
    "4": "src/experiments/experiment_4.py",
    "4_1": "src/experiments/experiment_4_1.py",
    "5": "src/experiments/experiment_5.py",
    "6": "src/experiments/experiment_6.py",
    "7": "src/experiments/experiment_7.py",
    "7_1": "src/experiments/experiment_7_1.py",
}

StudyModel = Literal["xor", "parity3", "chain"]
StudyBackend = Literal["python", "verilog"]


def _success_tick_summary(rows: List[Tuple[int, bool, int, int]]) -> None:
    ticks_ok = [t for _, ok, t, _ in rows if ok]
    if not ticks_ok:
        print("  aggregate: (no converged runs - mean/median ticks n/a)")
        return
    print(
        f"  aggregate: converged mean_ticks={mean(ticks_ok):.1f}  "
        f"median_ticks={median(ticks_ok):.1f}  (n={len(ticks_ok)})"
    )


def effective_chain_table_seed(
    traj_seed: int,
    n_in: int,
    n_out: int,
    target: Literal["parity", "random", "zero"],
    table_seed_override: int | None,
) -> int | None:
    """RNG seed used for ``make_truth_tables``; ``None`` if not applicable (parity/zero)."""
    if target != "random":
        return None
    if table_seed_override is not None:
        return table_seed_override
    return (traj_seed * 1_000_003 + n_in * 97 + n_out * 1_009) & 0x7FFFFFFF


def run_repo_script(relative_path: str) -> int:
    """Run a Python file from repo root; return its exit code."""
    script = REPO_ROOT / relative_path
    if not script.is_file():
        print(f"ERROR: missing script {script}", file=sys.stderr)
        return 1
    print(f"\n--- subprocess: {relative_path} ---\n")
    return subprocess.call([sys.executable, "-u", str(script)], cwd=str(REPO_ROOT))


def run_experiment(name: str) -> int:
    """Run a numbered experiment (plots / sweeps). See ``EXPERIMENTS`` keys."""
    path = EXPERIMENTS.get(name)
    if not path:
        print(f"ERROR: unknown experiment {name!r}; keys: {sorted(EXPERIMENTS)}", file=sys.stderr)
        return 1
    return run_repo_script(path)


def run_model_backend(
    model: StudyModel,
    backend: StudyBackend,
    *,
    xor_kw: Dict[str, Any] | None = None,
    parity3_kw: Dict[str, Any] | None = None,
    chain_kw: Dict[str, Any] | None = None,
) -> int:
    """
    Run one logical model on the Python ref or on cocotb Verilog.

    For ``backend="python"``, pass keyword dicts for ``run_xor`` / ``run_parity3`` /
    ``run_chain``. For ``verilog``, those dicts are ignored (TB defines stimulus).
    """
    print(f"\n{'=' * 64}\nmodel={model!r}  backend={backend!r}\n{'=' * 64}\n")

    if backend == "verilog":
        if model == "xor":
            return run_repo_script("run_sim.py")
        if model == "parity3":
            return run_repo_script("run_sim_parity3.py")
        if model == "chain":
            print(
                "NOTE: no generic chain RTL in this repo. "
                "For 3×1 parity use model='parity3' + verilog.\n",
                file=sys.stderr,
            )
            return 0

    if model == "xor":
        run_xor(**(xor_kw or {}))
        return 0
    if model == "parity3":
        run_parity3(**(parity3_kw or {}))
        return 0
    if model == "chain":
        run_chain(**(chain_kw or {}))
        return 0

    print(f"ERROR: unknown model {model!r}", file=sys.stderr)
    return 1


def run_xor(
    seed_start: int,
    num_seeds: int,
    max_ticks: int,
    *,
    plateau_escape: bool = True,
    verbose: bool = False,
) -> None:
    rows: List[Tuple[int, bool, int, int]] = []
    for i in range(num_seeds):
        seed = (seed_start + i) & 0xFFFF or 0xACE1
        m = xor.TTXorLearner(plateau_escape=plateau_escape)
        m.reset(seed=seed)
        s0 = m.score_current_gates()
        ok, ticks = m.run_until_xor(max_ticks)
        fin = m.score_current_gates()
        rows.append((seed, ok, ticks, fin))
        if verbose:
            print(f"  seed 0x{seed:04X}  start_score={s0}  ok={ok}  ticks={ticks}  final={fin}/4")

    oks = sum(1 for _, o, _, _ in rows if o)
    print(f"xor  plateau_escape={plateau_escape}  success {oks}/{num_seeds}  max_ticks={max_ticks}")
    _success_tick_summary(rows)
    if not verbose:
        for seed, ok, ticks, fin in rows:
            print(f"  0x{seed:04X}  ok={ok}  ticks={ticks}  score={fin}/4")


def run_parity3(
    seed_start: int,
    num_seeds: int,
    max_ticks: int,
    plateau_mask: int,
    *,
    plateau_escape: bool = True,
    verbose: bool = False,
) -> None:
    rows: List[Tuple[int, bool, int, int]] = []
    for i in range(num_seeds):
        seed = (seed_start + i) & 0xFFFF or 0xACE1
        m = parity3.TTParity3Learner(
            plateau_escape=plateau_escape,
            plateau_mask=plateau_mask,
        )
        m.reset(seed=seed)
        s0 = m.score_current_gates()
        ok, ticks = m.run_until_parity(max_ticks)
        fin = m.score_current_gates()
        rows.append((seed, ok, ticks, fin))
        if verbose:
            print(
                f"  seed 0x{seed:04X}  start_score={s0}  ok={ok}  ticks={ticks}  final={fin}/8"
            )

    oks = sum(1 for _, o, _, _ in rows if o)
    print(
        f"parity3  plateau_escape={plateau_escape}  mask={plateau_mask}  "
        f"success {oks}/{num_seeds}  max_ticks={max_ticks}"
    )
    _success_tick_summary(rows)
    if not verbose:
        for seed, ok, ticks, fin in rows:
            print(f"  0x{seed:04X}  ok={ok}  ticks={ticks}  score={fin}/8")


def run_chain(
    n_in: int,
    n_out: int,
    target: Literal["parity", "random", "zero"],
    seed_start: int,
    num_seeds: int,
    max_ticks: int,
    plateau_mask: int,
    *,
    plateau_escape: bool = True,
    table_seed_override: int | None = None,
    verbose: bool = False,
) -> None:
    rows: List[Tuple[int, bool, int, int]] = []
    for i in range(num_seeds):
        traj_seed = (seed_start + i) & 0xFFFF or 0xACE1
        ts_eff = effective_chain_table_seed(traj_seed, n_in, n_out, target, table_seed_override)
        if ts_eff is None:
            ts = 0
        else:
            ts = ts_eff
        tgt = chain.make_truth_tables(target, n_in, n_out, table_seed=ts)
        m = chain.TTChainLearner(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=plateau_escape,
            plateau_mask=plateau_mask,
        )
        m.reset(seed=traj_seed)
        s0 = m.score_current_gates()
        ok, ticks = m.run_until_perfect(max_ticks)
        fin = m.score_current_gates()
        rows.append((traj_seed, ok, ticks, fin))
        if verbose:
            extra = ""
            if target == "random":
                extra = f"  table_seed={ts_eff}"
            print(
                f"  seed 0x{traj_seed:04X}  start_score={s0}/{m.max_score}  "
                f"ok={ok}  ticks={ticks}  final={fin}/{m.max_score}{extra}"
            )

    oks = sum(1 for _, o, _, _ in rows if o)
    mx = (1 << n_in) * n_out
    print(
        f"chain  N={n_in} M={n_out} target={target}  plateau_escape={plateau_escape}  "
        f"mask={plateau_mask}  success {oks}/{num_seeds}  max_ticks={max_ticks}"
    )
    _success_tick_summary(rows)
    if not verbose:
        for seed, ok, ticks, fin in rows:
            print(f"  0x{seed:04X}  ok={ok}  ticks={ticks}  score={fin}/{mx}")


def sweep_chain(
    dimension: Literal["plateau_mask", "n_in", "n_out", "target"],
    values: Sequence[Any],
    fixed: Dict[str, Any],
) -> None:
    """
    Repeatedly call ``run_chain``, varying one keyword. ``fixed`` must supply every
    other ``run_chain`` argument (by name), and must **not** contain ``dimension``.
    """
    if dimension in fixed:
        raise ValueError(f"remove {dimension!r} from fixed; it is the swept axis")
    print(f"\n>>> sweep_chain  vary={dimension!r}  values={list(values)!r}\n")
    for v in values:
        kw = dict(fixed)
        kw[dimension] = v
        run_chain(**kw)
        print()


@dataclass(frozen=True)
class LandscapeProbeResult:
    up: int
    flat: int
    down: int
    mut_samples: int
    burn_in_reached: bool
    final_score: int
    max_score: int
    score_frac_target: float
    effective_table_seed: int | None

    @property
    def total(self) -> int:
        return self.up + self.flat + self.down


def _landscape_burn_max_ticks(n_in: int, n_out: int, burn_cap: int = 55_000) -> int:
    cells = (1 << n_in) * n_out
    return min(90_000, int(burn_cap + 800 * cells))


def _landscape_threshold_score(max_score: int, frac: float) -> int:
    t = int(round(frac * max_score))
    return max(1, min(max_score - 1, t))


def _score_with_trial(m: chain.TTChainLearner) -> int:
    s = 0
    for idx in range(m.nrows):
        xs = chain.row_bits(idx, m.n_in)
        ys = m._forward_trial_all(xs)
        for om in range(m.n_out):
            if ys[om] == m.target[om][idx]:
                s += 1
    return s


def _mutation_outcome_counts(m: chain.TTChainLearner, rng: random.Random, n_mut: int) -> Tuple[int, int, int]:
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
        new = _score_with_trial(m)
        d = new - old_score
        if d > 0:
            up += 1
        elif d == 0:
            flat += 1
        else:
            down += 1
    return up, flat, down


def probe_landscape(
    n_in: int,
    n_out: int,
    target: Literal["parity", "random", "zero"],
    *,
    score_frac: float = 0.75,
    traj_seed: int = 1,
    mut_samples: int = 120,
    plateau_mask: int = 3,
    plateau_escape: bool = True,
    burn_cap: int = 55_000,
    table_seed_override: int | None = None,
    verbose: bool = False,
) -> LandscapeProbeResult:
    """
    Burn in until score ≥ threshold(score_frac), then tally random-mutation Δ
    (up / flat / down) without accept. Same semantics as experiment 4.1 core loop.
    """
    ts = e4w.table_seed_for(target, n_in, n_out, traj_seed)
    if target == "random" and table_seed_override is not None:
        ts = table_seed_override
    ts_eff: int | None = None if target != "random" else ts

    tgt = chain.make_truth_tables(target, n_in, n_out, table_seed=ts)
    m = chain.TTChainLearner(
        n_in=n_in,
        n_out=n_out,
        target=tgt,
        plateau_escape=plateau_escape,
        plateau_mask=plateau_mask,
    )
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    mx = m.max_score
    thr = _landscape_threshold_score(mx, score_frac)
    cap = _landscape_burn_max_ticks(n_in, n_out, burn_cap)

    reached = m.score_current_gates() >= thr
    if not reached:
        for _ in range(cap):
            m.tick()
            if m.score_current_gates() >= thr:
                reached = True
                break

    if not reached:
        if verbose:
            print(
                f"probe_landscape: burn-in did not reach {score_frac:.0%} "
                f"(need score>={thr}/{mx}); no mutation sample"
            )
        return LandscapeProbeResult(
            up=0,
            flat=0,
            down=0,
            mut_samples=mut_samples,
            burn_in_reached=False,
            final_score=m.score_current_gates(),
            max_score=mx,
            score_frac_target=score_frac,
            effective_table_seed=ts_eff,
        )

    mut_rng = random.Random((traj_seed << 16) ^ int(100 * score_frac) ^ n_in ^ (n_out << 8))
    u, f, d = _mutation_outcome_counts(m, mut_rng, mut_samples)
    fin = m.score_current_gates()
    if verbose:
        tot = u + f + d
        pu, pf, pd = (100.0 * u / tot, 100.0 * f / tot, 100.0 * d / tot) if tot else (0.0, 0.0, 0.0)
        ts_note = f" table_seed={ts_eff}" if ts_eff is not None else ""
        print(
            f"probe_landscape: N={n_in} M={n_out} target={target}{ts_note}  "
            f"burn->score {fin}/{mx} (target>={thr})  "
            f"P_up={pu:.1f}% P_flat={pf:.1f}% P_down={pd:.1f}%  (n={mut_samples})"
        )
    return LandscapeProbeResult(
        up=u,
        flat=f,
        down=d,
        mut_samples=mut_samples,
        burn_in_reached=True,
        final_score=fin,
        max_score=mx,
        score_frac_target=score_frac,
        effective_table_seed=ts_eff,
    )


# Backwards-compatible alias
run_physical = run_model_backend


if __name__ == "__main__":
    exit_codes: List[int] = []

    # ---------------------------------------------------------------------------
    # D) PRIMARY — quick Python workbench (edit here first)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STUDY HUB - Python refs (default workbench)")
    print("=" * 72)

    # Example sweeps (comment out when not needed)
    # sweep_chain(
    #     "plateau_mask",
    #     [1, 3, 7, 15],
    #     fixed=dict(
    #         n_in=3,
    #         n_out=1,
    #         target="parity",
    #         seed_start=1,
    #         num_seeds=4,
    #         max_ticks=35_000,
    #         plateau_escape=True,
    #         table_seed_override=None,
    #         verbose=False,
    #     ),
    # )

    # Landscape probes (no matplotlib)
    probe_landscape(3, 1, "parity", score_frac=0.75, traj_seed=0xACE1, verbose=True)
    probe_landscape(4, 2, "random", score_frac=0.50, traj_seed=3, verbose=True)

    print("\n--- baseline runs ---\n")
    run_xor(
        seed_start=0x0001,
        num_seeds=8,
        max_ticks=15_000,
        plateau_escape=True,
        verbose=False,
    )
    print()

    run_xor(
        seed_start=0xACE1,
        num_seeds=1,
        max_ticks=20_000,
        plateau_escape=False,
        verbose=True,
    )
    print()

    run_parity3(
        seed_start=0x0001,
        num_seeds=8,
        max_ticks=40_000,
        plateau_mask=3,
        plateau_escape=True,
        verbose=False,
    )
    print()

    run_parity3(
        seed_start=0x1234,
        num_seeds=4,
        max_ticks=35_000,
        plateau_mask=7,
        plateau_escape=True,
        verbose=True,
    )
    print()

    run_chain(
        n_in=3,
        n_out=1,
        target="parity",
        seed_start=0x0001,
        num_seeds=6,
        max_ticks=50_000,
        plateau_mask=3,
        plateau_escape=True,
        table_seed_override=None,
        verbose=False,
    )
    print()

    run_chain(
        n_in=4,
        n_out=2,
        target="random",
        seed_start=0x0001,
        num_seeds=4,
        max_ticks=80_000,
        plateau_mask=3,
        plateau_escape=True,
        table_seed_override=99,
        verbose=True,
    )

    # ---------------------------------------------------------------------------
    # E) OPTIONAL — batch experiment scripts (long, plots)
    # ---------------------------------------------------------------------------
    EXPERIMENTS_TO_RUN: Tuple[str, ...] = (
        # "3",
        # "4_1",
    )
    for key in EXPERIMENTS_TO_RUN:
        exit_codes.append(run_experiment(key))

    # ---------------------------------------------------------------------------
    # F) OPTIONAL — RTL confirmation (cocotb); same logical model, Verilog TB
    # ---------------------------------------------------------------------------
    XOR_STUDY_KW: Dict[str, Any] = dict(
        seed_start=0x0001,
        num_seeds=4,
        max_ticks=8_000,
        plateau_escape=True,
        verbose=False,
    )
    PARITY3_STUDY_KW: Dict[str, Any] = dict(
        seed_start=0x0001,
        num_seeds=4,
        max_ticks=25_000,
        plateau_mask=3,
        plateau_escape=True,
        verbose=False,
    )
    CHAIN_STUDY_KW: Dict[str, Any] = dict(
        n_in=3,
        n_out=1,
        target="parity",
        seed_start=0x0001,
        num_seeds=4,
        max_ticks=40_000,
        plateau_mask=3,
        plateau_escape=True,
        table_seed_override=None,
        verbose=False,
    )

    MODEL_BACKEND_RUNS: List[Tuple[StudyModel, StudyBackend]] = [
        # ("xor", "python"),
        # ("xor", "verilog"),
        # ("parity3", "python"),
        # ("parity3", "verilog"),
    ]
    for mod, be in MODEL_BACKEND_RUNS:
        exit_codes.append(
            run_model_backend(
                mod,
                be,
                xor_kw=XOR_STUDY_KW,
                parity3_kw=PARITY3_STUDY_KW,
                chain_kw=CHAIN_STUDY_KW,
            )
        )

    raise SystemExit(0 if all(c == 0 for c in exit_codes) else 1)
