"""
Picklable worker for experiment 7 multiprocessing (Windows spawn-safe).

Job tuple:
  (policy_idx, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed)
``fixed_table_seed`` is ``None`` when the table seed must be derived per trajectory
(``random`` targets); otherwise it is the int passed to ``make_truth_tables``.
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_7_learners as e7l
import tt_chain_learner_spec as chain


def run_one(
    job: tuple[int, int, int, str, int, int, int | None],
) -> tuple[int, bool, int]:
    policy_idx, n_in, n_out, kind, traj_seed, max_ticks, fixed_ts = job
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
    )
    if policy_idx == 0:
        m: chain.TTChainLearner = chain.TTChainLearner(**common)
    elif policy_idx == 1:
        m = e7l.PlasticMappedPlateauChain(**common)
    elif policy_idx == 2:
        m = e7l.RecentSuccessExploitPlateauChain(**common, exploit_window=8)
    elif policy_idx == 3:
        m = e7l.MixedPlasticSuccessPlateauChain(**common, exploit_window=8)
    else:
        raise ValueError(f"unknown policy_idx {policy_idx}")
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    ok, ticks = m.run_until_perfect(max_ticks)
    return policy_idx, ok, ticks
