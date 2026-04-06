"""
Picklable worker for experiment 13 (supervised adaptive DAG vs local-unsupervised DAG).

Job:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed, extra_gates, log_stride)

variant 0 = ``Exp12AdaptiveDAG`` (global target score in COMPARE)
variant 1 = ``TTAdaptiveDAGLocalUnsupervised`` (local |corr| or balance only; target unused in COMPARE)

Logs external ``score_current_gates()`` every ``log_stride`` **ticks** (wall-clock FSM steps).
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_12_learners as e12l
import tt_adaptive_dag_local_unsup as adlu
import tt_chain_learner_spec as chain
from tt_chain_learner_exp72 import gpo_topology_a


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

    if variant == 0:
        m = e12l.Exp12AdaptiveDAG(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
        )
    elif variant == 1:
        m = adlu.TTAdaptiveDAGLocalUnsupervised(
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

    if m.score_current_gates() == m.max_score:
        curve_ticks.append(0)
        curve_score.append(m.max_score)
        return {
            "variant": variant,
            "ok_task": True,
            "ticks_ran": 0,
            "final_external_score": m.max_score,
            "max_score": m.max_score,
            "curve_ticks": curve_ticks,
            "curve_score": curve_score,
        }

    for t in range(max_ticks):
        if t % stride == 0:
            curve_ticks.append(t)
            curve_score.append(m.score_current_gates())
        m.tick()
        if variant == 0 and m._gates_mutated_last_tick and m.score_current_gates() == m.max_score:
            curve_ticks.append(t + 1)
            curve_score.append(m.max_score)
            return {
                "variant": variant,
                "ok_task": True,
                "ticks_ran": t + 1,
                "final_external_score": m.max_score,
                "max_score": m.max_score,
                "curve_ticks": curve_ticks,
                "curve_score": curve_score,
            }

    curve_ticks.append(max_ticks)
    curve_score.append(m.score_current_gates())
    fs = m.score_current_gates()
    return {
        "variant": variant,
        "ok_task": fs == m.max_score,
        "ticks_ran": max_ticks,
        "final_external_score": fs,
        "max_score": m.max_score,
        "curve_ticks": curve_ticks,
        "curve_score": curve_score,
    }
