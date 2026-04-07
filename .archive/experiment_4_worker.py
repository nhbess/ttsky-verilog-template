"""
Picklable entry point for experiment 4 multiprocessing.

``multiprocessing`` spawn (Windows) re-imports the main module; workers must live
in a real module, not ``__main__``, when the experiment is run as a script.
"""

from __future__ import annotations

import tt_chain_learner_spec as spec


def table_seed_for(kind: str, n_in: int, n_out: int, learner_seed: int) -> int:
    if kind != "random":
        return 0
    return (learner_seed * 1_000_003 + n_in * 97 + n_out * 1_009) & 0x7FFFFFFF


def run_one_seed(
    payload: tuple[int, int, int, int, str, int, int],
) -> tuple[int, int, str, bool, int]:
    i, j, n_in, n_out, kind, s, cap = payload
    ts = table_seed_for(kind, n_in, n_out, s)
    tgt = spec.make_truth_tables(kind, n_in, n_out, table_seed=ts)
    m = spec.TTChainLearner(
        n_in=n_in,
        n_out=n_out,
        target=tgt,
        plateau_escape=True,
        plateau_mask=3,
    )
    m.reset(seed=s & 0xFFFF or 0xACE1)
    ok, ticks = m.run_until_perfect(cap)
    return i, j, kind, ok, ticks
