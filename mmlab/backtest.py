"""Run a strategy in the simulated market and attribute its P&L.

P&L attribution (exact, in ticks):
    total P&L = spread capture + inventory P&L - fees - liquidation cost
- spread capture : for each fill, how far from the mid we traded, in our favour
                   (buying 1 below the mid earns +1 of edge)
- inventory P&L  : gains/losses from holding inventory while the mid moves
                   (the residual of the mark-to-market P&L)
- adverse selection, measured separately as the 10-second markout: how much
                   the mid moved against each fill right after it
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import MarketParams, MarketSim
from .strategies import Strategy


@dataclass
class EpisodeResult:
    pnl: float                 # final P&L after liquidation, ticks
    spread_capture: float
    inventory_pnl: float
    fees: float
    liquidation_cost: float
    markout_10s: float         # sum over fills of side * (mid_{t+10} - mid_fill) * qty (negative = adverse)
    n_fills: int
    volume: int
    mean_abs_inventory: float
    mtm: np.ndarray            # mark-to-market P&L path
    inventory: np.ndarray
    mid: np.ndarray
    quotes: np.ndarray         # (T, 2) our bid/ask (nan when not quoting)


def run_episode(strategy: Strategy, params: MarketParams | None = None, seed: int = 0,
                horizon: int = 3600) -> EpisodeResult:
    sim = MarketSim(params or MarketParams(), seed=seed)
    strategy.reset(seed)
    mids, invs, mtm, quotes = [], [], [], []
    obs = sim.observe()
    prev = 0.0
    for _ in range(horizon):
        bid, ask = strategy.quote(obs)
        sim.set_quotes(bid, ask)
        sim.step()
        obs = sim.observe()
        value = sim.cash + sim.inventory * obs["mid"]
        strategy.on_step_end(obs, value - prev)
        prev = value
        mids.append(obs["mid"]); invs.append(sim.inventory); mtm.append(value)
        quotes.append((np.nan if bid is None else bid, np.nan if ask is None else ask))

    mids = np.array(mids)
    spread_end = sim.book.spread() or 1
    liquidation = abs(sim.inventory) * spread_end / 2
    spread_capture = sum(f.side * (f.mid - f.price) * f.qty for f in sim.fills)
    markout = 0.0
    for f in sim.fills:
        later = mids[min(f.t + 10, horizon - 1)]
        markout += f.side * (later - f.mid) * f.qty
    final_mtm = mtm[-1]
    inventory_pnl = final_mtm + sim.fees - spread_capture
    return EpisodeResult(
        pnl=final_mtm - liquidation, spread_capture=spread_capture, inventory_pnl=inventory_pnl,
        fees=sim.fees, liquidation_cost=liquidation, markout_10s=markout,
        n_fills=len(sim.fills), volume=sum(f.qty for f in sim.fills),
        mean_abs_inventory=float(np.mean(np.abs(invs))), mtm=np.array(mtm),
        inventory=np.array(invs), mid=mids, quotes=np.array(quotes))


def evaluate(strategy: Strategy, seeds, params=None, horizon=3600) -> dict:
    """Run one episode per seed and summarise. Same seeds for every strategy."""
    res = [run_episode(strategy, params, s, horizon) for s in seeds]
    pnl = np.array([r.pnl for r in res])
    out = dict(
        strategy=strategy.name,
        mean_pnl=pnl.mean(), std_pnl=pnl.std(ddof=1),
        sharpe_per_episode=pnl.mean() / pnl.std(ddof=1),
        t_stat=pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))),
        spread_capture=np.mean([r.spread_capture for r in res]),
        inventory_pnl=np.mean([r.inventory_pnl for r in res]),
        markout_10s=np.mean([r.markout_10s for r in res]),
        liquidation=np.mean([r.liquidation_cost for r in res]),
        fees=np.mean([r.fees for r in res]),
        fills=np.mean([r.n_fills for r in res]),
        mean_abs_inventory=np.mean([r.mean_abs_inventory for r in res]),
        episodes=res,
    )
    return out
