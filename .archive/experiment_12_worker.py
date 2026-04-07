"""
Picklable worker for experiment 12 (fixed Exp72 credit vs adaptive DAG connectivity).

Job:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed, extra_gates,
   credit_mode, topology)

variant 0 = ``Exp12CreditBaseline`` (topology A Exp72, matched gate count)
variant 1 = ``Exp12AdaptiveDAG`` (``G = gpo_topology_a(n_in, extra_gates)`` per output)
"""

from __future__ import annotations

from collections import Counter

import experiment_4_worker as e4w
import experiment_12_learners as e12l
import tt_chain_learner_spec as chain
from tt_chain_learner_exp72 import gpo_topology_a


def packed_truth_key(m: object) -> tuple[int, ...]:
    n_in = m.n_in
    n_out = m.n_out
    nrows = 1 << n_in
    key: list[int] = []
    for mo in range(n_out):
        bits = 0
        for idx in range(nrows):
            xs = chain.row_bits(idx, n_in)
            ys = m.forward_all(xs)
            bits |= ys[mo] << idx
        key.append(bits)
    return tuple(key)


def sigma_eff_from_log(
    log: list[tuple[int, bool, bool]], window: int = 20
) -> tuple[float, float]:
    """
    After each accepted COMPARE, count distinct gates proposed in the next ``window``
    trials (normalized by window). Returns (mean σ_eff, fraction of accepts that had a window).
    """
    if len(log) < 2:
        return float("nan"), float("nan")
    gates = [e[0] for e in log]
    acc = [e[1] for e in log]
    T = len(log)
    vals: list[float] = []
    for t in range(T - 1):
        if not acc[t]:
            continue
        end = min(T, t + 1 + window)
        nnext = end - (t + 1)
        if nnext <= 0:
            continue
        distinct = len(set(gates[t + 1 : end]))
        vals.append(distinct / nnext)
    if not vals:
        return float("nan"), float("nan")
    return sum(vals) / len(vals), len(vals) / max(1, sum(1 for x in acc if x))


def run_one(
    job: tuple[int, int, int, str, int, int, int | None, int, str, str],
) -> dict:
    (
        variant,
        n_in,
        n_out,
        kind,
        traj_seed,
        max_ticks,
        fixed_ts,
        extra_gates,
        credit_mode,
        topology,
    ) = job
    if topology != "A":
        raise ValueError("experiment 12 baseline uses topology A only")
    if fixed_ts is not None:
        ts = fixed_ts
    else:
        ts = e4w.table_seed_for(kind, n_in, n_out, traj_seed)
    tgt = chain.make_truth_tables(kind, n_in, n_out, table_seed=ts)
    G = gpo_topology_a(n_in, extra_gates)
    seed = traj_seed & 0xFFFF or 0xACE1

    if variant == 0:
        m = e12l.Exp12CreditBaseline(
            n_in=n_in,
            n_out=n_out,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
            extra_gates=extra_gates,
            topology="A",
            credit_mode=credit_mode,
        )
    elif variant == 1:
        m = e12l.Exp12AdaptiveDAG(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )
    else:
        raise ValueError(f"unknown variant {variant}")

    m.reset(seed=seed)
    ok = False
    ticks = max_ticks
    if m.score_current_gates() == m.max_score:
        ok, ticks = True, 0
    else:
        for t in range(max_ticks):
            m.tick()
            if m._gates_mutated_last_tick and m.score_current_gates() == m.max_score:
                ok, ticks = True, t + 1
                break
        else:
            ok = m.score_current_gates() == m.max_score

    log = m.exp12_log
    sig, win_frac = sigma_eff_from_log(log, window=20)
    acc_gates: Counter[int] = Counter()
    n_rewire_trials = sum(1 for _u, _a, rw in log if rw)
    n_accept_rewire = sum(1 for u, a, rw in log if a and rw)
    for u, a, _rw in log:
        if a:
            acc_gates[u] += 1

    return {
        "variant": variant,
        "ok": ok,
        "ticks": ticks,
        "sigma_eff": sig,
        "sigma_window_frac": win_frac,
        "truth_key": packed_truth_key(m),
        "n_compares": len(log),
        "n_rewire_trials": n_rewire_trials,
        "n_accept_rewire": n_accept_rewire,
        "accepts_per_gate": dict(acc_gates),
        "n_gates": len(m.gate) if hasattr(m, "gate") else m.n_gates_flat,
        "final_score": m.score_current_gates(),
        "max_score": m.max_score,
    }
