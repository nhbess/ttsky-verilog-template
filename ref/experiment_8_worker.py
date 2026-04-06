"""
Picklable worker for experiment 8 (multi-gate mutation batch B, topology A/B, Windows spawn-safe).

Job:
  (policy_idx, n_in, n_out, kind, traj_seed, max_ticks, fixed_table_seed,
   extra_gates, topology, mutation_batch)
"""

from __future__ import annotations

import experiment_4_worker as e4w
import experiment_8_learners as e8l
import tt_chain_learner_exp72_multi as e72m
import tt_chain_learner_spec as chain


def run_one(
    job: tuple[int, int, int, str, int, int, int | None, int, str, int],
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
        mutation_batch,
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
        mutation_batch=mutation_batch,
    )
    if policy_idx == 0:
        m: e72m.TTChainLearnerExp72Multi = e72m.TTChainLearnerExp72Multi(**common)
    elif policy_idx == 1:
        m = e8l.PlasticMappedPlateauChainMulti(**common)
    elif policy_idx == 2:
        m = e8l.RecentSuccessExploitPlateauChainMulti(**common, exploit_window=8)
    elif policy_idx == 3:
        m = e8l.MixedPlasticSuccessPlateauChainMulti(**common, exploit_window=8)
    else:
        raise ValueError(f"unknown policy_idx {policy_idx}")
    m.reset(seed=traj_seed & 0xFFFF or 0xACE1)
    ok, ticks = m.run_until_perfect(max_ticks)
    return policy_idx, ok, ticks
