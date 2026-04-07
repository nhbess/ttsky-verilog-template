"""
Experiment 17: Exp16 setup + error-biased gate selection P(u) ∝ (e_u + eps)^alpha.

Job tuple matches experiment 16:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed, extra_gates, log_stride)
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_12_learners as e12l
import experiment_17_clamp_relax as e17cr
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
        m = e17cr.TTAdaptiveDAGClampRelaxFixedErrPick(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
            relax_steps=8,
            err_pick_alpha=1.0,
            err_pick_eps=1e-6,
        )
    elif variant == 2:
        m = e17cr.TTAdaptiveDAGClampRelaxErrPick(
            n_in=n_in,
            n_out=n_out,
            G=G,
            target=tgt,
            plateau_escape=True,
            plateau_mask=3,
            relax_steps=8,
            allow_rewire=True,
            err_pick_alpha=1.0,
            err_pick_eps=1e-6,
        )
    else:
        raise ValueError(f"unknown variant {variant}")

    m.reset(seed=seed)
    curve_ticks: list[int] = []
    curve_score: list[int] = []
    curve_E_local: list[float] = []
    curve_depth: list[int] = []
    curve_settle: list[int] = []
    stride = max(1, log_stride)

    def pack(
        ok_task: bool,
        ticks_ran: int,
        final_external_score: int,
    ) -> dict:
        return {
            "variant": variant,
            "ok_task": ok_task,
            "ticks_ran": ticks_ran,
            "final_external_score": final_external_score,
            "max_score": m.max_score,
            "curve_ticks": curve_ticks,
            "curve_score": curve_score,
            "curve_E_local": curve_E_local,
            "curve_depth": curve_depth,
            "curve_settle": curve_settle,
            "final_E_local": float(getattr(m, "exp16_last_E_local", 0.0)),
            "final_max_depth": int(getattr(m, "exp16_last_max_depth", 0)),
            "final_settle_steps": int(getattr(m, "exp16_last_settle_steps", 0)),
        }

    if m.score_current_gates() == m.max_score:
        curve_ticks.append(0)
        curve_score.append(m.max_score)
        curve_E_local.append(0.0)
        curve_depth.append(0)
        curve_settle.append(0)
        return pack(True, 0, m.max_score)

    for t in range(max_ticks):
        if t % stride == 0:
            curve_ticks.append(t)
            curve_score.append(m.score_current_gates())
            if variant == 0:
                curve_E_local.append(0.0)
                curve_depth.append(0)
                curve_settle.append(0)
            else:
                curve_E_local.append(float(m.exp16_last_E_local))
                curve_depth.append(int(m.exp16_last_max_depth))
                curve_settle.append(int(m.exp16_last_settle_steps))
        m.tick()
        if m._gates_mutated_last_tick and m.score_current_gates() == m.max_score:
            curve_ticks.append(t + 1)
            curve_score.append(m.max_score)
            if variant == 0:
                curve_E_local.append(0.0)
                curve_depth.append(0)
                curve_settle.append(0)
            else:
                curve_E_local.append(float(m.exp16_last_E_local))
                curve_depth.append(int(m.exp16_last_max_depth))
                curve_settle.append(int(m.exp16_last_settle_steps))
            return pack(True, t + 1, m.max_score)

    curve_ticks.append(max_ticks)
    curve_score.append(m.score_current_gates())
    if variant == 0:
        curve_E_local.append(0.0)
        curve_depth.append(0)
        curve_settle.append(0)
    else:
        curve_E_local.append(float(m.exp16_last_E_local))
        curve_depth.append(int(m.exp16_last_max_depth))
        curve_settle.append(int(m.exp16_last_settle_steps))
    fs = m.score_current_gates()
    return pack(fs == m.max_score, max_ticks, fs)
