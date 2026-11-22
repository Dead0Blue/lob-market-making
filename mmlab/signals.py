"""Short-horizon order-book signals and how much they predict.

- imbalance  I = (Qbid - Qask) / (Qbid + Qask) at the best prices, in [-1, 1]
- microprice = (Pbid * Qask + Pask * Qbid) / (Qbid + Qask): the mid tilted
               towards the side with less quantity (that side is more likely
               to be eaten first)

We record the book every second and measure, at several horizons, the
correlation between the signal and the future change of the mid
(information coefficient, IC) and the average future move per signal bucket.
"""
from __future__ import annotations

import numpy as np

from .market import MarketParams, MarketSim


def record_book(seconds=3600, seed=0, params=None):
    sim = MarketSim(params or MarketParams(), seed=seed)
    rows = []
    for _ in range(seconds):
        o = sim.observe()
        rows.append((o["mid"], o["imbalance"], o["microprice"], sim.value))
        sim.step()
    a = np.array(rows)
    return dict(mid=a[:, 0], imbalance=a[:, 1], microprice=a[:, 2], value=a[:, 3])


def predictive_power(data, horizons=(1, 5, 10, 30)):
    mid, imb, micro = data["mid"], data["imbalance"], data["microprice"]
    out = []
    for h in horizons:
        fut = mid[h:] - mid[:-h]
        out.append(dict(
            horizon=h,
            ic_imbalance=float(np.corrcoef(imb[:-h], fut)[0, 1]),
            ic_micro_minus_mid=float(np.corrcoef((micro - mid)[:-h], fut)[0, 1]),
        ))
    return out


def bucket_curve(data, horizon=5, n_buckets=10):
    """Average future mid move per imbalance decile."""
    mid, imb = data["mid"], data["imbalance"]
    fut = mid[horizon:] - mid[:-horizon]
    x = imb[:-horizon]
    edges = np.quantile(x, np.linspace(0, 1, n_buckets + 1))
    centres, means, errs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() < 10:
            continue
        centres.append(x[m].mean())
        means.append(fut[m].mean())
        errs.append(fut[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(centres), np.array(means), np.array(errs)


def error_vs_fair_value(data):
    """How far mid and microprice are from the hidden fair value (RMSE, ticks)."""
    v = data["value"]
    return (float(np.sqrt(np.mean((data["mid"] - v) ** 2))),
            float(np.sqrt(np.mean((data["microprice"] - v) ** 2))))
