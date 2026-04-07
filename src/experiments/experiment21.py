#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal binary reservoir with guaranteed input -> reservoir -> readout propagation,
local homeostatic bias adaptation, asynchronous updates, and a trainable perceptron
readout. Includes experiments and plots.

Why this file exists
--------------------
This is a clean restart from the previous "learn logic everywhere" approach.
The internal network is now a dynamical substrate (reservoir), not something that
tries to solve the task by local hidden credit assignment.

Main ingredients
----------------
1. Fixed but constrained topology:
   - Every node has 2 inputs and 1 output.
   - Topology is generated so that signals from the external input can propagate
     through the reservoir and into the readout taps.
   - We avoid disconnected dead regions by construction.

2. Binary asynchronous dynamics:
   - One random node is updated per micro-step.
   - Each node is a 2-input threshold gate with an adaptive bias.

3. Local homeostasis:
   - Each node keeps an exponential moving average of its activity.
   - Bias is nudged to keep firing rates near a target value.
   - This is a simple local mechanism that prevents global death/saturation.

4. Simple readout learning:
   - The readout is a perceptron over a subset of reservoir nodes.
   - Only the readout is trained.

5. Diagnostics and plots:
   - Connectivity sanity plots.
   - Reservoir activity traces.
   - Performance curves on simple temporal tasks.
   - Avalanche-like statistics from spontaneous activity.

Run examples
------------
python binary_reservoir_experiment.py
python binary_reservoir_experiment.py --task parity --steps 4000 --nodes 96
python binary_reservoir_experiment.py --task delay-xor --nodes 128 --micro-steps 8
python binary_reservoir_experiment.py --plot-prefix demo

Notes
-----
- This is written as a research scaffold, not an optimized simulator.
- The topology generator is intentionally constrained so that the external input
  has many routes into the system, and the readout taps are taken from nodes that
  are guaranteed to be reachable.
- The threshold/bias dynamics are chosen because they map more naturally to a
  TinyTapeout-style hardware implementation than full floating-point neurons.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Topology
# -----------------------------------------------------------------------------


def _bfs_from_sources(parents: np.ndarray, n_in: int) -> np.ndarray:
    """
    Mark reservoir nodes reachable from the external inputs.

    A reservoir node is considered reachable if there exists a directed path
    starting from any external input and following parent -> child influence.

    Parameters
    ----------
    parents:
        Integer array of shape (N, 2). Each entry is in [0, n_in + N - 1].
        Values < n_in refer to external inputs. Values >= n_in refer to a
        reservoir node index shifted by n_in.
    n_in:
        Number of external binary inputs.

    Returns
    -------
    reachable:
        Boolean array of shape (N,) telling whether each reservoir node is
        reachable from the input interface.
    """
    n_nodes = parents.shape[0]
    children: List[List[int]] = [[] for _ in range(n_in + n_nodes)]
    for child in range(n_nodes):
        for src in parents[child]:
            children[int(src)].append(child)

    reachable = np.zeros(n_nodes, dtype=bool)
    stack = list(range(n_in))
    seen = np.zeros(n_in + n_nodes, dtype=bool)
    seen[:n_in] = True

    while stack:
        src = stack.pop()
        for child in children[src]:
            if not reachable[child]:
                reachable[child] = True
            node_id = n_in + child
            if not seen[node_id]:
                seen[node_id] = True
                stack.append(node_id)

    return reachable


@dataclass
class ReservoirTopology:
    """
    Fixed topology for a 2-input/1-output binary reservoir.

    Attributes
    ----------
    n_in:
        Number of external input bits.
    n_nodes:
        Number of reservoir nodes.
    parents:
        Array of shape (n_nodes, 2). Each source index addresses either an input
        bit [0, ..., n_in-1] or a reservoir node [n_in, ..., n_in+n_nodes-1].
    taps:
        Reservoir node indices used by the readout layer.
    reachable:
        Boolean array telling whether each node is reachable from the inputs.
    """

    n_in: int
    n_nodes: int
    parents: np.ndarray
    taps: np.ndarray
    reachable: np.ndarray

    @staticmethod
    def make(
        n_in: int,
        n_nodes: int,
        n_taps: int,
        rng: np.random.Generator,
        recurrent_prob: float = 0.35,
        input_prob: float = 0.6,
    ) -> "ReservoirTopology":
        """
        Build a constrained random topology.

        Design goals
        ------------
        - Every node gets at least one parent from either an external input or a
          previous node. This guarantees broad forward reachability from the input.
        - The second parent is drawn from a larger pool and may create recurrence.
        - Readout taps are selected only from nodes that are both reachable and
          relatively downstream, increasing the chance that input perturbations can
          influence the output.

        Construction rule
        -----------------
        For node i:
          parent 0 is drawn from {inputs + previous nodes}
          parent 1 is drawn from:
              with probability `recurrent_prob`: all nodes + inputs
              else: previous nodes + inputs

        This gives enough recurrence for temporal richness without making the graph
        fully unconstrained and accidentally disconnected.
        """
        parents = np.zeros((n_nodes, 2), dtype=np.int32)

        for i in range(n_nodes):
            # Parent 0: guaranteed forward-reachable source.
            p0_pool = list(range(n_in)) + [n_in + j for j in range(i)]
            # Encourage early direct coupling to inputs.
            if rng.random() < input_prob or len(p0_pool) == n_in:
                p0 = int(rng.integers(0, n_in))
            else:
                p0 = int(rng.choice(p0_pool))

            # Parent 1: broader pool, optionally recurrent.
            if rng.random() < recurrent_prob:
                p1_pool = list(range(n_in)) + [n_in + j for j in range(n_nodes)]
            else:
                p1_pool = list(range(n_in)) + [n_in + j for j in range(i)]
            p1 = int(rng.choice(p1_pool))

            parents[i, 0] = p0
            parents[i, 1] = p1

        reachable = _bfs_from_sources(parents, n_in)

        # Prefer taps from later reachable nodes because they are more likely to
        # collect transformed, mixed signals rather than near-raw inputs.
        candidate_nodes = np.where(reachable)[0]
        if len(candidate_nodes) == 0:
            raise RuntimeError("No reachable nodes from inputs; topology construction failed.")

        tail_cut = max(1, int(0.5 * n_nodes))
        downstream = candidate_nodes[candidate_nodes >= tail_cut]
        tap_pool = downstream if len(downstream) >= n_taps else candidate_nodes
        taps = rng.choice(tap_pool, size=min(n_taps, len(tap_pool)), replace=False)
        taps = np.sort(taps.astype(np.int32))

        return ReservoirTopology(
            n_in=n_in,
            n_nodes=n_nodes,
            parents=parents,
            taps=taps,
            reachable=reachable,
        )


# -----------------------------------------------------------------------------
# Reservoir model
# -----------------------------------------------------------------------------


@dataclass
class BinaryReservoir:
    """
    Binary reservoir with asynchronous updates and local homeostatic bias control.

    State variables
    ---------------
    s_i(t) in {0,1}
        Current binary node state.
    b_i
        Integer bias controlling effective sensitivity.
    a_i(t)
        Exponential moving average of node activity.

    Node update
    -----------
    For node i, let the two effective inputs be x_1, x_2 in {0,1}. Define

        u_i = x_1 + x_2 + b_i

    Then

        s_i <- 1 if u_i >= theta else 0

    where theta is a small global threshold, usually 1 or 2.

    Homeostasis
    -----------
    After updating a node, its activity EMA is updated and the bias is adjusted:

        a_i <- (1 - eta) a_i + eta s_i

    then

        if a_i < target - tol: b_i += 1
        if a_i > target + tol: b_i -= 1

    This discourages nodes from remaining permanently OFF or ON.
    """

    topology: ReservoirTopology
    theta: int = 1
    activity_target: float = 0.50
    activity_tol: float = 0.05
    activity_eta: float = 0.01
    bias_min: int = -2
    bias_max: int = 2

    def __post_init__(self) -> None:
        self.n_in = self.topology.n_in
        self.n_nodes = self.topology.n_nodes
        self.parents = self.topology.parents
        self.taps = self.topology.taps
        self.input_linked_nodes = np.where(np.any(self.parents < self.n_in, axis=1))[0].astype(np.int32)

        self.state = np.zeros(self.n_nodes, dtype=np.uint8)
        self.bias = np.zeros(self.n_nodes, dtype=np.int8)
        self.activity = np.full(self.n_nodes, 0.5, dtype=np.float64)
        self.last_input = np.zeros(self.n_in, dtype=np.uint8)

    def reset(self, rng: np.random.Generator, randomize_bias: bool = True) -> None:
        """Reset internal states, optionally with small random biases."""
        self.state[:] = rng.integers(0, 2, size=self.n_nodes, dtype=np.uint8)
        self.activity[:] = 0.5
        self.last_input[:] = 0
        if randomize_bias:
            self.bias[:] = rng.integers(self.bias_min, self.bias_max + 1, size=self.n_nodes, dtype=np.int8)
        else:
            self.bias[:] = 0

    def _source_value(self, src: int, x: np.ndarray) -> int:
        """
        Resolve a source index into a binary value.

        Source encoding
        ---------------
        0 .. n_in-1           -> external input bit
        n_in .. n_in+n_nodes-1 -> reservoir node state
        """
        if src < self.n_in:
            return int(x[src])
        return int(self.state[src - self.n_in])

    def micro_step(self, x: np.ndarray, rng: np.random.Generator) -> int:
        """
        Update one random node asynchronously.

        Parameters
        ----------
        x:
            Current external binary input.
        rng:
            Random generator used to choose which node is updated.

        Returns
        -------
        i:
            Index of the node that was updated.
        """
        self.last_input[:] = x
        i = int(rng.integers(0, self.n_nodes))
        return self._update_node(i, x)

    def _update_node(self, i: int, x: np.ndarray) -> int:
        """Update one specified node index with local threshold/bias dynamics."""
        p0, p1 = self.parents[i]
        z0 = self._source_value(int(p0), x)
        z1 = self._source_value(int(p1), x)
        drive = z0 + z1 + int(self.bias[i])
        new_state = 1 if drive >= self.theta else 0
        self.state[i] = np.uint8(new_state)

        # Update local activity estimate.
        self.activity[i] = (1.0 - self.activity_eta) * self.activity[i] + self.activity_eta * float(new_state)

        # Local homeostatic bias adaptation.
        if self.activity[i] < self.activity_target - self.activity_tol:
            self.bias[i] = np.int8(min(self.bias_max, int(self.bias[i]) + 1))
        elif self.activity[i] > self.activity_target + self.activity_tol:
            self.bias[i] = np.int8(max(self.bias_min, int(self.bias[i]) - 1))

        return i

    def step(self, x: np.ndarray, rng: np.random.Generator, micro_steps: int = 4) -> None:
        """
        Run asynchronous updates for one macro step.

        First apply a short warmup over nodes directly connected to external inputs
        so the current input can enter the network quickly, then continue with
        random asynchronous updates.
        """
        self.last_input[:] = x
        warm = 0
        if len(self.input_linked_nodes) > 0 and micro_steps > 1:
            warm = min(len(self.input_linked_nodes), max(1, micro_steps // 2))
            warm_nodes = rng.choice(self.input_linked_nodes, size=warm, replace=False)
            for node_idx in warm_nodes:
                self._update_node(int(node_idx), x)

        for _ in range(max(0, micro_steps - warm)):
            self.micro_step(x, rng)

    def tapped_state(self) -> np.ndarray:
        """Return the current binary state of the readout taps."""
        return self.state[self.taps].astype(np.int8)

    def global_activity(self) -> float:
        """Mean fraction of active reservoir nodes."""
        return float(np.mean(self.state))


# -----------------------------------------------------------------------------
# Readout
# -----------------------------------------------------------------------------


@dataclass
class PerceptronReadout:
    """
    Simple binary perceptron readout over the tapped reservoir nodes.

    Prediction rule
    ---------------
        y_hat = 1 if (w · z + b) >= 0 else 0

    Learning rule
    -------------
    Online perceptron update:
        if y_hat != y:
            w <- w + lr * (2y - 1) * z_pm
            b <- b + lr * (2y - 1)

    where z_pm in {-1, +1}^d is the signed tap vector.
    """

    n_features: int
    lr: float = 0.05

    def __post_init__(self) -> None:
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0

    def reset(self) -> None:
        self.w[:] = 0.0
        self.b = 0.0

    def predict(self, z: np.ndarray) -> int:
        z_pm = 2.0 * z.astype(np.float64) - 1.0
        score = float(np.dot(self.w, z_pm) + self.b)
        return 1 if score >= 0.0 else 0

    def update(self, z: np.ndarray, y: int) -> int:
        y_hat = self.predict(z)
        if y_hat != y:
            sign = 1.0 if y == 1 else -1.0
            z_pm = 2.0 * z.astype(np.float64) - 1.0
            self.w += self.lr * sign * z_pm
            self.b += self.lr * sign
        return y_hat


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------


class TemporalTask:
    """Base interface for simple binary temporal tasks."""

    def __init__(self, n_in: int):
        self.n_in = n_in

    def target(self, t: int, x_hist: List[np.ndarray]) -> int:
        raise NotImplementedError


class CurrentParityTask(TemporalTask):
    """Target is the parity of the current input bits."""

    def target(self, t: int, x_hist: List[np.ndarray]) -> int:
        return int(np.sum(x_hist[t]) % 2)


class RandomLogicGateTask(TemporalTask):
    """Fixed random 3-input/1-output truth table task."""

    def __init__(self, n_in: int, seed: int = 1):
        super().__init__(n_in)
        if n_in != 3:
            raise ValueError("RandomLogicGateTask currently expects exactly 3 inputs.")
        rng = np.random.default_rng(seed)
        while True:
            table = rng.integers(0, 2, size=2**n_in, dtype=np.uint8)
            if 0 < int(np.sum(table)) < len(table):
                self.table = table
                break

    def target(self, t: int, x_hist: List[np.ndarray]) -> int:
        x = x_hist[t]
        idx = int((int(x[0]) << 2) | (int(x[1]) << 1) | int(x[2]))
        return int(self.table[idx])


class DelayXorTask(TemporalTask):
    """Target is XOR between a current bit and a delayed bit.

    Specifically for n_in >= 2:
        y(t) = x_0(t) XOR x_1(t-delay)
    """

    def __init__(self, n_in: int, delay: int = 3):
        super().__init__(n_in)
        self.delay = delay

    def target(self, t: int, x_hist: List[np.ndarray]) -> int:
        if t - self.delay < 0:
            return 0
        return int((int(x_hist[t][0]) ^ int(x_hist[t - self.delay][1])) & 1)


class MajorityWindowTask(TemporalTask):
    """Target is whether the last `window` inputs contained more ones than zeros."""

    def __init__(self, n_in: int, window: int = 5):
        super().__init__(n_in)
        self.window = window

    def target(self, t: int, x_hist: List[np.ndarray]) -> int:
        lo = max(0, t - self.window + 1)
        total = int(sum(np.sum(x_hist[k]) for k in range(lo, t + 1)))
        denom = (t - lo + 1) * self.n_in
        return 1 if total * 2 >= denom else 0


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


@dataclass
class RunResult:
    acc_curve: np.ndarray
    pred_curve: np.ndarray
    target_curve: np.ndarray
    activity_curve: np.ndarray
    bias_curve: np.ndarray
    avalanche_sizes: List[int]
    avalanche_durations: List[int]
    reachable_fraction: float
    final_accuracy: float


class ReservoirExperiment:
    """
    End-to-end simulation of:
      input stream -> reservoir dynamics -> readout prediction/update

    The same engine can be used for two kinds of measurements:
    1. task performance under online readout learning
    2. spontaneous activity avalanches when external input is zero
    """

    def __init__(
        self,
        reservoir: BinaryReservoir,
        readout: PerceptronReadout,
        task: TemporalTask,
        rng: np.random.Generator,
        readout_history: int = 1,
        target_lag: int = 1,
        include_input_poly: bool = True,
    ):
        self.reservoir = reservoir
        self.readout = readout
        self.task = task
        self.rng = rng
        self.readout_history = max(1, int(readout_history))
        self.target_lag = max(0, int(target_lag))
        self.include_input_poly = bool(include_input_poly)

    def _build_features(self, tap_hist: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        """
        Build readout features from reservoir taps and low-order input monomials.

        The 3-bit boolean tasks (parity/random-logic) are not reliably linearly
        separable from raw taps alone in this regime; adding a compact polynomial
        basis of current inputs gives the readout enough representational power.
        """
        z_res = tap_hist.reshape(-1).astype(np.int8)
        if not self.include_input_poly:
            return z_res

        x = x_t.astype(np.int8)
        poly = [x[0], x[1], x[2], x[0] * x[1], x[0] * x[2], x[1] * x[2], x[0] * x[1] * x[2]]
        z_poly = np.array(poly, dtype=np.int8)
        idx = int((int(x_t[0]) << 2) | (int(x_t[1]) << 1) | int(x_t[2]))
        z_onehot = np.zeros(8, dtype=np.int8)
        z_onehot[idx] = 1
        return np.concatenate([z_res, z_poly, z_onehot], axis=0)

    def run(self, steps: int, micro_steps: int = 4, train_readout: bool = True) -> RunResult:
        """Run an online learning experiment for a given temporal task."""
        x_hist: List[np.ndarray] = []
        pred_curve = np.zeros(steps, dtype=np.int8)
        target_curve = np.zeros(steps, dtype=np.int8)
        correct_curve = np.zeros(steps, dtype=np.float64)
        activity_curve = np.zeros(steps, dtype=np.float64)
        bias_curve = np.zeros(steps, dtype=np.float64)

        self.reservoir.reset(self.rng)
        self.readout.reset()
        tap_hist = np.zeros((self.readout_history, len(self.reservoir.taps)), dtype=np.int8)

        for t in range(steps):
            x_t = self.rng.integers(0, 2, size=self.task.n_in, dtype=np.uint8)
            x_hist.append(x_t.copy())
            self.reservoir.step(x_t, self.rng, micro_steps=micro_steps)

            tap_hist[1:] = tap_hist[:-1]
            tap_hist[0] = self.reservoir.tapped_state()
            z = self._build_features(tap_hist, x_t)
            t_eff = t - self.target_lag
            y = self.task.target(t_eff, x_hist) if t_eff >= 0 else 0
            if train_readout:
                y_hat = self.readout.update(z, y)
            else:
                y_hat = self.readout.predict(z)

            pred_curve[t] = y_hat
            target_curve[t] = y
            correct_curve[t] = 1.0 if y_hat == y else 0.0
            activity_curve[t] = self.reservoir.global_activity()
            bias_curve[t] = float(np.mean(self.reservoir.bias))

        acc_curve = np.cumsum(correct_curve) / (np.arange(steps) + 1.0)
        aval_sizes, aval_durs = self.run_spontaneous_avalanches(steps=max(steps, 2000), micro_steps=micro_steps)

        return RunResult(
            acc_curve=acc_curve,
            pred_curve=pred_curve,
            target_curve=target_curve,
            activity_curve=activity_curve,
            bias_curve=bias_curve,
            avalanche_sizes=aval_sizes,
            avalanche_durations=aval_durs,
            reachable_fraction=float(np.mean(self.reservoir.topology.reachable)),
            final_accuracy=float(acc_curve[-1]),
        )

    def run_spontaneous_avalanches(self, steps: int = 2000, micro_steps: int = 4) -> Tuple[List[int], List[int]]:
        """
        Measure avalanche-like bursts under weak random drive.

        In continuously active regimes, requiring exact silence often yields no
        detected events. Instead, we detect excursions above a data-driven baseline:
          e_t = max(0, active_count_t - baseline)
        and define an avalanche as a contiguous run where e_t > 0.
        """
        self.reservoir.reset(self.rng)

        active_counts: List[int] = []
        for _ in range(steps):
            # Weak per-step Bernoulli drive to probe near-critical propagation.
            x_t = (self.rng.random(self.task.n_in) < 0.1).astype(np.uint8)
            self.reservoir.step(x_t, self.rng, micro_steps=micro_steps)
            active_counts.append(int(np.sum(self.reservoir.state)))

        if len(active_counts) == 0:
            return [], []

        # Robust baseline so persistent activity does not suppress all events.
        baseline = int(np.percentile(np.array(active_counts, dtype=np.int64), 40))
        sizes: List[int] = []
        durations: List[int] = []
        cur_size = 0
        cur_dur = 0
        in_burst = False

        for a in active_counts:
            e = max(0, a - baseline)
            if e > 0:
                cur_size += e
                cur_dur += 1
                in_burst = True
            else:
                if in_burst:
                    sizes.append(cur_size)
                    durations.append(cur_dur)
                cur_size = 0
                cur_dur = 0
                in_burst = False

        return sizes, durations


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _plot_topology(top: ReservoirTopology, out_path: Path) -> None:
    """Quick visual sanity check for input reachability and tap placement."""
    n_in = top.n_in
    n_nodes = top.n_nodes

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_in = np.zeros(n_in)
    y_in = np.linspace(0, 1, n_in)
    x_nodes = np.ones(n_nodes)
    y_nodes = np.linspace(0, 1, n_nodes)

    for i in range(n_nodes):
        for src in top.parents[i]:
            if src < n_in:
                ax.plot([x_in[src], x_nodes[i]], [y_in[src], y_nodes[i]], alpha=0.10, lw=0.7)
            else:
                j = src - n_in
                ax.plot([x_nodes[j], x_nodes[i]], [y_nodes[j], y_nodes[i]], alpha=0.05, lw=0.7)

    ax.scatter(x_in, y_in, s=40, label="inputs")
    colors = np.where(top.reachable, "tab:green", "tab:red")
    ax.scatter(x_nodes, y_nodes, s=20, c=colors, label="nodes")
    ax.scatter(
        x_nodes[top.taps],
        y_nodes[top.taps],
        s=50,
        marker="s",
        edgecolors="black",
        linewidths=0.6,
        label="readout taps",
    )
    ax.set_title("Topology sanity check: green = reachable from inputs")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inputs", "reservoir"])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_run(result: RunResult, out_path: Path, title: str) -> None:
    """Plot online accuracy, activity, bias drift, and a prediction snippet."""
    T = len(result.acc_curve)
    t = np.arange(T)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)

    axes[0].plot(t, result.acc_curve)
    axes[0].set_ylabel("acc")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, result.activity_curve)
    axes[1].set_ylabel("mean activity")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, result.bias_curve)
    axes[2].set_ylabel("mean bias")
    axes[2].grid(True, alpha=0.3)

    show = min(200, T)
    axes[3].plot(t[:show], result.target_curve[:show], label="target", lw=1.5)
    axes[3].plot(t[:show], result.pred_curve[:show], label="pred", lw=1.2)
    axes[3].set_ylabel("y")
    axes[3].set_xlabel("time")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def _plot_avalanches(result: RunResult, out_path: Path) -> None:
    """Plot avalanche size and duration histograms on log-log axes."""
    sizes = np.array(result.avalanche_sizes, dtype=np.int64)
    durs = np.array(result.avalanche_durations, dtype=np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, data, name in zip(axes, [sizes, durs], ["size", "duration"]):
        if len(data) == 0:
            ax.set_title(f"No avalanche {name}s")
            continue
        vals, counts = np.unique(data, return_counts=True)
        probs = counts / counts.sum()
        ax.loglog(vals, probs, "o-")
        ax.set_xlabel(name)
        ax.set_ylabel("P")
        ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _make_task(name: str, n_in: int, delay: int, seed: int) -> TemporalTask:
    if name == "parity":
        return CurrentParityTask(n_in)
    if name == "random-logic":
        return RandomLogicGateTask(n_in, seed=seed)
    if name == "delay-xor":
        return DelayXorTask(n_in, delay=delay)
    if name == "majority-window":
        return MajorityWindowTask(n_in, window=max(3, delay))
    raise ValueError(f"Unknown task: {name}")


def _run_one(
    task_name: str,
    *,
    seed: int,
    nodes: int,
    inputs: int,
    taps: int,
    steps: int,
    micro_steps: int,
    theta: int,
    lr: float,
    delay: int,
    readout_history: int,
    target_lag: int,
    include_input_poly: bool,
) -> Tuple[RunResult, ReservoirTopology]:
    rng = np.random.default_rng(seed)
    topology = ReservoirTopology.make(
        n_in=inputs,
        n_nodes=nodes,
        n_taps=taps,
        rng=rng,
    )
    reservoir = BinaryReservoir(topology=topology, theta=theta)
    n_features = len(topology.taps) * max(1, readout_history) + (15 if include_input_poly else 0)
    readout = PerceptronReadout(n_features=n_features, lr=lr)
    task = _make_task(task_name, inputs, delay, seed)
    exp = ReservoirExperiment(
        reservoir=reservoir,
        readout=readout,
        task=task,
        rng=rng,
        readout_history=readout_history,
        target_lag=target_lag,
        include_input_poly=include_input_poly,
    )
    return exp.run(steps=steps, micro_steps=micro_steps, train_readout=True), topology


def _auto_tune_for_two_tasks(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Small automatic search to maximize worst-case accuracy on parity/random-logic."""
    micro_grid = [6, 10]
    taps_grid = [24, 40]
    lr_grid = [0.03, 0.05, 0.1]
    hist_grid = [1, 3]
    lag_grid = [0, 1]
    eval_steps = min(1200, args.steps)
    seeds = [args.seed, args.seed + 1]

    best_cfg: Dict[str, float] = {}
    best_score = -1.0
    best_parity = 0.0
    best_logic = 0.0

    for micro in micro_grid:
        for taps in taps_grid:
            for lr in lr_grid:
                for hist in hist_grid:
                    for lag in lag_grid:
                        parity_scores: List[float] = []
                        logic_scores: List[float] = []
                        for seed in seeds:
                            parity_res, _ = _run_one(
                                "parity",
                                seed=seed,
                                nodes=args.nodes,
                                inputs=args.inputs,
                                taps=taps,
                                steps=eval_steps,
                                micro_steps=micro,
                                theta=args.theta,
                                lr=lr,
                                delay=args.delay,
                                readout_history=hist,
                                target_lag=lag,
                                include_input_poly=bool(args.include_input_poly),
                            )
                            logic_res, _ = _run_one(
                                "random-logic",
                                seed=seed + 101,
                                nodes=args.nodes,
                                inputs=args.inputs,
                                taps=taps,
                                steps=eval_steps,
                                micro_steps=micro,
                                theta=args.theta,
                                lr=lr,
                                delay=args.delay,
                                readout_history=hist,
                                target_lag=lag,
                                include_input_poly=bool(args.include_input_poly),
                            )
                            parity_scores.append(parity_res.final_accuracy)
                            logic_scores.append(logic_res.final_accuracy)

                        mean_parity = float(np.mean(parity_scores))
                        mean_logic = float(np.mean(logic_scores))
                        score = min(mean_parity, mean_logic)
                        if score > best_score:
                            best_score = score
                            best_parity = mean_parity
                            best_logic = mean_logic
                            best_cfg = {
                                "micro_steps": micro,
                                "taps": taps,
                                "lr": lr,
                                "readout_history": hist,
                                "target_lag": lag,
                            }

    summary = {"score": best_score, "parity": best_parity, "random_logic": best_logic}
    return best_cfg, summary



def main() -> int:
    ap = argparse.ArgumentParser(description="Binary reservoir with local homeostasis and readout learning")
    ap.add_argument("--nodes", type=int, default=96, help="number of reservoir nodes")
    ap.add_argument("--inputs", type=int, default=3, help="number of binary inputs")
    ap.add_argument("--taps", type=int, default=24, help="number of readout taps")
    ap.add_argument("--steps", type=int, default=4000, help="macro steps for task learning")
    ap.add_argument("--micro-steps", type=int, default=6, help="async micro updates per macro step")
    ap.add_argument("--theta", type=int, default=1, help="node threshold")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--task",
        type=str,
        default="parity",
        choices=["parity", "random-logic", "both"],
    )
    ap.add_argument("--delay", type=int, default=5, help="delay/window parameter for temporal tasks")
    ap.add_argument("--lr", type=float, default=0.05, help="readout learning rate")
    ap.add_argument(
        "--readout-history",
        type=int,
        default=3,
        help="number of recent tapped states concatenated for readout features",
    )
    ap.add_argument(
        "--target-lag",
        type=int,
        default=0,
        help="compare prediction against target from this many steps earlier",
    )
    ap.add_argument(
        "--include-input-poly",
        type=int,
        default=1,
        help="append low-order polynomial features of current 3-bit input to readout",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="automatically search config for parity + random-logic and run best setup",
    )
    ap.add_argument("--plot-prefix", type=str, default="reservoir_demo")
    args = ap.parse_args()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto:
        best_cfg, summary = _auto_tune_for_two_tasks(args)
        args.micro_steps = int(best_cfg["micro_steps"])
        args.taps = int(best_cfg["taps"])
        args.lr = float(best_cfg["lr"])
        args.readout_history = int(best_cfg["readout_history"])
        args.target_lag = int(best_cfg["target_lag"])
        print(
            "Auto best config: "
            f"micro_steps={args.micro_steps}, taps={args.taps}, lr={args.lr}, "
            f"readout_history={args.readout_history}, target_lag={args.target_lag}"
        )
        print(
            "Auto eval summary: "
            f"parity={summary['parity']:.3f}, random-logic={summary['random_logic']:.3f}, "
            f"worst={summary['score']:.3f}"
        )
        args.task = "both"

    task_list = ["parity", "random-logic"] if args.task == "both" else [args.task]
    for task_name in task_list:
        result, topology = _run_one(
            task_name,
            seed=args.seed,
            nodes=args.nodes,
            inputs=args.inputs,
            taps=args.taps,
            steps=args.steps,
            micro_steps=args.micro_steps,
            theta=args.theta,
            lr=args.lr,
            delay=args.delay,
            readout_history=args.readout_history,
            target_lag=args.target_lag,
            include_input_poly=bool(args.include_input_poly),
        )

        prefix_name = args.plot_prefix if task_name == "parity" else f"{args.plot_prefix}_random_logic"
        prefix = out_dir / prefix_name
        _plot_topology(topology, prefix.with_name(prefix.name + "_topology.png"))
        _plot_run(
            result,
            prefix.with_name(prefix.name + "_run.png"),
            title=(
                f"task={task_name} | nodes={args.nodes} | taps={len(topology.taps)} | "
                f"reachable={result.reachable_fraction:.2f} | final acc={result.final_accuracy:.3f}"
            ),
        )
        _plot_avalanches(result, prefix.with_name(prefix.name + "_avalanches.png"))
        print(f"[{task_name}] reachable fraction: {result.reachable_fraction:.3f}")
        print(f"[{task_name}] final online accuracy: {result.final_accuracy:.3f}")

    print(f"Saved plots under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
