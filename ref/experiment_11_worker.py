"""
Picklable worker for experiment 11 (global vs local ε acceptance + trial logs).

Job:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed,
   extra_gates, topology, credit_mode)

variant 0 = global score COMPARE + credit ``u`` (``Exp11InstrumentedGlobal``)
variant 1 = local ε + uniform ``u``
variant 2 = local ε + credit ``u``
"""

from __future__ import annotations

from collections import Counter

import experiment_4_worker as e4w
import experiment_11_learners as e11l
import tt_chain_learner_spec as chain


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
        topology,
        credit_mode,
    ) = job
    if fixed_ts is not None:
        ts = fixed_ts
    else:
        ts = e4w.table_seed_for(kind, n_in, n_out, traj_seed)
    tgt = chain.make_truth_tables(kind, n_in, n_out, table_seed=ts)
    common = dict(
        n_in=n_in,
        n_out=n_out,
        target=tgt,
        plateau_escape=True,
        plateau_mask=3,
        extra_gates=extra_gates,
        topology=topology,
    )
    if variant == 0:
        m = e11l.Exp11InstrumentedGlobal(**common, credit_mode=credit_mode)
    elif variant == 1:
        m = e11l.Exp11InstrumentedLocalUniform(**common)
    elif variant == 2:
        m = e11l.Exp11InstrumentedLocalCredit(**common, credit_mode=credit_mode)
    else:
        raise ValueError(f"unknown variant {variant}")
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
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

    acc_gates: Counter[int] = Counter()
    for g, acc in zip(m.compare_log_unit_sel, m.compare_log_accepted):
        if acc:
            acc_gates[g] += 1

    return {
        "variant": variant,
        "ok": ok,
        "ticks": ticks,
        "trial_global_delta": m.compare_log_global_delta,
        "trial_eps_delta": m.compare_log_eps_delta,
        "trial_accepted": m.compare_log_accepted,
        "trial_gate": m.compare_log_unit_sel,
        "accepts_per_gate": dict(acc_gates),
        "n_gates": len(m.gate),
        "final_score": m.score_current_gates(),
        "max_score": m.max_score,
        "final_gates": list(m.gate),
        "n_in": n_in,
        "n_out": n_out,
        "extra_gates": extra_gates,
        "topology": topology,
        "target": [list(row) for row in tgt],
    }
