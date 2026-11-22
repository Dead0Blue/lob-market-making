"""Calibrate the inputs of the Avellaneda-Stoikov model from the market itself,
as a practitioner would:

- sigma: volatility of the mid, in ticks per sqrt(second);
- A, k : fill intensity lambda(delta) = A exp(-k delta). We quote a probe at a
         fixed distance delta from the mid on both sides, count fills per
         second, and fit log lambda = log A - k delta by least squares.
"""
from __future__ import annotations

import numpy as np

from .backtest import run_episode
from .market import MarketParams, MarketSim
from .strategies import FixedOffset


def estimate_sigma(params: MarketParams | None = None, seeds=range(3), horizon=3600) -> float:
    changes = []
    for s in seeds:
        sim = MarketSim(params or MarketParams(), seed=10_000 + s)
        mids = []
        for _ in range(horizon):
            sim.step()
            mids.append(sim.observe()["mid"])
        # 30-second changes, rescaled to 1 second: avoids the bid-ask bounce in 1s changes
        m = np.array(mids)
        changes.append(np.diff(m[::30]) / np.sqrt(30))
    return float(np.concatenate(changes).std())


def estimate_intensity(params: MarketParams | None = None, distances=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                       seeds=range(3), horizon=1800):
    """Return (A, k, distances, measured fill rates per side per second)."""
    rates = []
    for d in distances:
        probe = FixedOffset(offset=d, max_inventory=10**9)
        fills = sum(run_episode(probe, params, 20_000 + s, horizon).n_fills for s in seeds)
        rates.append(fills / (2 * horizon * len(seeds)))  # per side per second
    rates = np.array(rates)
    ok = rates > 0
    slope, intercept = np.polyfit(np.array(distances)[ok], np.log(rates[ok]), 1)
    return float(np.exp(intercept)), float(-slope), np.array(distances), rates
