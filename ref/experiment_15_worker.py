"""
Picklable worker for experiment 15 (Exp14 setup + softmax per-gate LUT proposals).

Job:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed, extra_gates, log_stride)

variant 0 = Exp15SupervisedDAG, 1 = Exp15LocalUnsupDAG, 2 = Exp15LocalPredictiveDAG
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_15_learners as e15l
import tt_chain_learner_spec as chain
from tt_chain_learner_exp72 import gpo_topology_a


def _entropy_summary(m) -> tuple[float, list[float]]:
    ent = e15l.per_gate_entropies_bits(m.logits)
    if not ent:
        return 0.0, []
    return float(sum(ent) / len(ent)), ent


def _op_hist(choices: list[int]) -> list[int]:
    h = [0] * 16
    for c in choices:
        h[c] += 1
    return h


def run_one(
    job: tuple[int, int, int, str, int, int, int | None, int, int],
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
        log_stride,
    ) = job
    if fixed_ts is not None:
        ts = fixed_ts
    else:
        ts = e4w.table_seed_for(kind, n_in, n_out, traj_seed)
    tgt = chain.make_truth_tables(kind, n_in, n_out, table_seed=ts)
    G = gpo_topology_a(n_in, extra_gates)
    seed = traj_seed & 0xFFFF or 0xACE1

    if G <= 0:
        raise ValueError(f"invalid topology G={G}")

    if variant == 0:
        m = e15l.Exp15SupervisedDAG(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )
    elif variant == 1:
        m = e15l.Exp15LocalUnsupDAG(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )
    elif variant == 2:
        m = e15l.Exp15LocalPredictiveDAG(
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
    curve_ticks: list[int] = []
    curve_score: list[int] = []
    stride = max(1, log_stride)

    def pack(
        ok_task: bool,
        ticks_ran: int,
        final_external_score: int,
    ) -> dict:
        mh, pge = _entropy_summary(m)
        return {
            "variant": variant,
            "ok_task": ok_task,
            "ticks_ran": ticks_ran,
            "final_external_score": final_external_score,
            "max_score": m.max_score,
            "curve_ticks": curve_ticks,
            "curve_score": curve_score,
            "op_hist": _op_hist(m.op_choice_history),
            "mean_gate_entropy_bits": mh,
            "per_gate_entropy_bits": pge,
        }

    if m.score_current_gates() == m.max_score:
        curve_ticks.append(0)
        curve_score.append(m.max_score)
        return pack(True, 0, m.max_score)

    for t in range(max_ticks):
        if t % stride == 0:
            curve_ticks.append(t)
            curve_score.append(m.score_current_gates())
        m.tick()
        if variant == 0 and m._gates_mutated_last_tick and m.score_current_gates() == m.max_score:
            curve_ticks.append(t + 1)
            curve_score.append(m.max_score)
            return pack(True, t + 1, m.max_score)

    curve_ticks.append(max_ticks)
    curve_score.append(m.score_current_gates())
    fs = m.score_current_gates()
    return pack(fs == m.max_score, max_ticks, fs)
