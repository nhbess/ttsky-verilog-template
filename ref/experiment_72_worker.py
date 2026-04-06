"""
Picklable worker for experiment 7.2 (extra gates K, topology A/B, Windows spawn-safe).

Job:
  (policy_idx, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed, extra_gates, topology)
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_72_learners as e72l
import tt_chain_learner_exp72 as e72
import tt_chain_learner_spec as chain


def run_one(
    job: tuple[int, int, int, str, int, int, int | None, int, str],
) -> tuple[int, bool, int]:
    (
        policy_idx,
        n_in,
        n_out,
        kind,
        traj_seed,
        max_ticks,
        fixed_ts,
        extra_gates,
        topology,
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
    if policy_idx == 0:
        m: e72.TTChainLearnerExp72 = e72.TTChainLearnerExp72(**common)
    elif policy_idx == 1:
        m = e72l.PlasticMappedPlateauChain(**common)
    elif policy_idx == 2:
        m = e72l.RecentSuccessExploitPlateauChain(**common, exploit_window=8)
    elif policy_idx == 3:
        m = e72l.MixedPlasticSuccessPlateauChain(**common, exploit_window=8)
    else:
        raise ValueError(f"unknown policy_idx {policy_idx}")
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    ok, ticks = m.run_until_perfect(max_ticks)
    return policy_idx, ok, ticks
