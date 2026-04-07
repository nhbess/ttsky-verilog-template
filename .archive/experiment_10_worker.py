"""
Picklable worker for experiment 10 (uniform vs credit pick vs credit + local plasticity).

Job:
  (variant, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed,
   extra_gates, topology, credit_mode)

variant 0 = ``TTChainLearnerExp72`` (uniform ``u``, default plastic / plateau)
variant 1 = ``TTChainLearnerExp72CreditWeighted`` (credit ``u`` only)
variant 2 = ``TTChainLearnerExp72CreditPlastic`` (credit ``u`` + credit INIT + credit plateau)
"""

from __future__ import annotations

import experiment_4_worker as e4w
import tt_chain_learner_exp72 as e72
import tt_chain_learner_exp72_credit as e72c
import tt_chain_learner_exp72_credit_plastic as e72cp
import tt_chain_learner_spec as chain


def run_one(
    job: tuple[int, int, int, str, int, int, int | None, int, str, str],
) -> tuple[int, bool, int]:
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
        m: e72.TTChainLearnerExp72 = e72.TTChainLearnerExp72(**common)
    elif variant == 1:
        m = e72c.TTChainLearnerExp72CreditWeighted(**common, credit_mode=credit_mode)
    elif variant == 2:
        m = e72cp.TTChainLearnerExp72CreditPlastic(**common, credit_mode=credit_mode)
    else:
        raise ValueError(f"unknown variant {variant}")
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    ok, ticks = m.run_until_perfect(max_ticks)
    return variant, ok, ticks
