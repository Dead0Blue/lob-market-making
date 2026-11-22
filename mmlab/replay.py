"""Backtest a market maker on recorded order-book data (no simulation).

Input: two CSV files, as written by scripts/record_binance.py
  snapshots.csv : ts, bid_px_1..N, bid_qty_1..N, ask_px_1..N, ask_qty_1..N
  trades.csv    : ts, price, qty, side   (side +1 = buyer-initiated, -1 = seller-initiated)

Fill model (conservative queue approximation)
- When we post at a price that already has displayed quantity, we join the
  back of the queue: queue_ahead = displayed quantity at that price.
- If we post inside the spread, nothing is ahead of us.
- A trade at a price strictly through our quote fills us completely.
- A trade exactly at our price first consumes the queue ahead of us; we are
  filled only by what is left over.
- Displayed quantity at our level that disappears between snapshots is treated
  as cancellations in front of us (queue_ahead can only shrink).
Our own orders do not move the market in a replay (no market impact): this is
the main limitation versus the live simulator, and it flatters the results.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayData:
    ts: np.ndarray          # snapshot times (seconds)
    bid_px: np.ndarray      # (n, levels) in ticks
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray
    trade_ts: np.ndarray
    trade_px: np.ndarray    # ticks
    trade_qty: np.ndarray
    trade_side: np.ndarray
    tick_size: float


def load(snapshots_csv: str, trades_csv: str, tick_size: float) -> ReplayData:
    with open(snapshots_csv) as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], np.array(rows[1:], dtype=float)
    n = (len(header) - 1) // 4
    to_ticks = lambda x: np.rint(x / tick_size).astype(np.int64)
    ts = rows[:, 0]
    bid_px, bid_qty = to_ticks(rows[:, 1:1 + n]), rows[:, 1 + n:1 + 2 * n]
    ask_px, ask_qty = to_ticks(rows[:, 1 + 2 * n:1 + 3 * n]), rows[:, 1 + 3 * n:1 + 4 * n]
    with open(trades_csv) as f:
        tr = np.array(list(csv.reader(f))[1:], dtype=float)
    if tr.size == 0:
        tr = np.zeros((0, 4))
    order = np.argsort(tr[:, 0], kind="stable")
    tr = tr[order]
    return ReplayData(ts, bid_px, bid_qty, ask_px, ask_qty, tr[:, 0], to_ticks(tr[:, 1]),
                      tr[:, 2], tr[:, 3].astype(int), tick_size)


def _obs(d: ReplayData, i: int, inventory: int) -> dict:
    bb, ba = int(d.bid_px[i, 0]), int(d.ask_px[i, 0])
    bq, aq = float(d.bid_qty[i, 0]), float(d.ask_qty[i, 0])
    mid = 0.5 * (bb + ba)
    return dict(t=i, mid=mid, best_bid=bb, best_ask=ba, bid_qty=bq, ask_qty=aq,
                imbalance=(bq - aq) / (bq + aq) if bq + aq > 0 else 0.0,
                microprice=(bb * aq + ba * bq) / (bq + aq) if bq + aq > 0 else mid,
                inventory=inventory)


def _displayed(d: ReplayData, i: int, side: int, price: int) -> float:
    px, qty = (d.bid_px[i], d.bid_qty[i]) if side == 1 else (d.ask_px[i], d.ask_qty[i])
    hit = np.nonzero(px == price)[0]
    return float(qty[hit[0]]) if hit.size else 0.0


@dataclass
class ReplayResult:
    pnl: float
    spread_capture: float
    n_fills: int
    mean_abs_inventory: float
    mtm: np.ndarray
    inventory: np.ndarray


def run_replay(strategy, d: ReplayData, order_size: float = 1.0) -> ReplayResult:
    """Replay the data, quoting `order_size` per side at each snapshot."""
    strategy.reset()
    inv, cash, spread_capture, n_fills = 0, 0.0, 0.0, 0
    quotes = {1: None, -1: None}   # side -> [price, queue_ahead]
    j = 0                          # next trade index
    mtm, invs = [], []
    prev_value = 0.0
    for i in range(len(d.ts)):
        obs = _obs(d, i, inv)
        bid, ask = strategy.quote(obs)
        for side, price in ((1, bid), (-1, ask)):
            if price is not None:  # never cross
                price = min(price, obs["best_ask"] - 1) if side == 1 else max(price, obs["best_bid"] + 1)
            q = quotes[side]
            shown = _displayed(d, i, side, price) if price is not None else 0.0
            if price is None:
                quotes[side] = None
            elif q is None or q[0] != price:
                quotes[side] = [price, shown]          # join the back of the queue
            else:
                q[1] = min(q[1], shown)                # cancellations ahead of us
        t_end = d.ts[i + 1] if i + 1 < len(d.ts) else np.inf
        while j < len(d.trade_ts) and d.trade_ts[j] < t_end:
            tp, tq, ts = d.trade_px[j], d.trade_qty[j], d.trade_side[j]
            our_side = -ts                     # a seller-initiated trade hits bids
            q = quotes[our_side]
            if q is not None:
                price, ahead = q
                through = tp < price if our_side == 1 else tp > price
                at = tp == price
                filled = 0.0
                if through:
                    filled = order_size
                elif at:
                    left = tq - ahead
                    q[1] = max(ahead - tq, 0.0)
                    filled = min(order_size, max(left, 0.0))
                if filled >= order_size - 1e-12:
                    inv += our_side
                    cash -= our_side * price
                    spread_capture += our_side * (obs["mid"] - price)
                    n_fills += 1
                    quotes[our_side] = None
            j += 1
        value = cash + inv * obs["mid"]
        strategy.on_step_end(obs, value - prev_value)
        prev_value = value
        mtm.append(value)
        invs.append(inv)
    last = _obs(d, len(d.ts) - 1, inv)
    liquidation = abs(inv) * (last["best_ask"] - last["best_bid"]) / 2
    return ReplayResult(pnl=mtm[-1] - liquidation, spread_capture=spread_capture, n_fills=n_fills,
                        mean_abs_inventory=float(np.mean(np.abs(invs))), mtm=np.array(mtm),
                        inventory=np.array(invs))


def calibrate(d: ReplayData, distances=(0.5, 1, 1.5, 2, 3, 4), step_seconds=None):
    """sigma (ticks per sqrt(second)) and fill intensity A exp(-k delta) from the data.

    A probe at distance delta from the mid counts as filled when a trade prints
    strictly through it before the next snapshot.
    """
    mid = 0.5 * (d.bid_px[:, 0] + d.ask_px[:, 0])
    dt = np.median(np.diff(d.ts)) if step_seconds is None else step_seconds
    lag = max(1, int(round(30 / dt)))
    sigma = float(np.std(np.diff(mid[::lag])) / math.sqrt(lag * dt))
    idx = np.searchsorted(d.ts, d.trade_ts, side="right") - 1
    valid = idx >= 0
    rates = []
    for delta in distances:
        hits = 0
        m = mid[idx[valid]]
        px, sd = d.trade_px[valid], d.trade_side[valid]
        hits += np.sum((sd == -1) & (px < m - delta))   # through our bid
        hits += np.sum((sd == 1) & (px > m + delta))    # through our ask
        rates.append(hits / (2 * (d.ts[-1] - d.ts[0])))
    rates = np.array(rates)
    ok = rates > 0
    slope, intercept = np.polyfit(np.array(distances)[ok], np.log(rates[ok]), 1)
    return sigma, float(np.exp(intercept)), float(-slope)


def export_synthetic(path_prefix: str, seconds: int = 3600, seed: int = 0, levels: int = 5,
                     tick_size: float = 0.01):
    """Write simulator output in the recorder's CSV format (to test the replay path)."""
    from .market import MarketSim
    from .orderbook import ASK, BID
    sim = MarketSim(seed=seed)
    trades = []
    orig_market = sim.book.market

    def logging_market(side, qty):
        fills = orig_market(side, qty)
        for f in fills:
            trades.append((sim.t + 0.5, f.price * tick_size, f.qty, side))
        return fills
    sim.book.market = logging_market
    with open(f"{path_prefix}_snapshots.csv", "w", newline="") as fs:
        w = csv.writer(fs)
        w.writerow(["ts"] + [f"bid_px_{k}" for k in range(1, levels + 1)] + [f"bid_qty_{k}" for k in range(1, levels + 1)]
                   + [f"ask_px_{k}" for k in range(1, levels + 1)] + [f"ask_qty_{k}" for k in range(1, levels + 1)])
        for _ in range(seconds):
            bids = sim.book.depth(BID, levels)
            asks = sim.book.depth(ASK, levels)
            bids += [(bids[-1][0] - 1 - k, 0) for k in range(levels - len(bids))]
            asks += [(asks[-1][0] + 1 + k, 0) for k in range(levels - len(asks))]
            w.writerow([sim.t] + [round(p * tick_size, 8) for p, _ in bids] + [q for _, q in bids]
                       + [round(p * tick_size, 8) for p, _ in asks] + [q for _, q in asks])
            sim.step()
    with open(f"{path_prefix}_trades.csv", "w", newline="") as ft:
        w = csv.writer(ft)
        w.writerow(["ts", "price", "qty", "side"])
        w.writerows((t, round(p, 8), q, s) for t, p, q, s in trades)
