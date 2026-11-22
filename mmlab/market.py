"""Synthetic market: a limit order book driven by three kinds of traders.

The book is real (see orderbook.py); the traders are simulated. Each step is
one second.

1. A hidden fair value V moves as a random walk (Brownian motion).
2. Liquidity providers post limit orders around their own estimate of value,
   which only partly reflects V (they are slow), and cancel orders at random.
3. Noise traders send market orders in random directions.
4. Informed traders know V: when the mid price is wrong by more than half a
   tick, they send market orders in the direction of V.

Point 4 creates adverse selection: a market maker who quotes passively gets
filled by informed traders exactly when its quotes are stale. It also makes
order-book imbalance informative, because informed traders eat one side of
the book first. These are the two effects a real market maker must manage.

Our market maker interacts through `set_quotes`; its orders sit in the same
book, keep their queue position while unchanged, and are filled by the same
market orders as everyone else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .orderbook import ASK, BID, OrderBook


@dataclass
class MarketParams:
    start_price: int = 10_000          # ticks (100.00 with a 0.01 tick)
    sigma: float = 0.35                # fair-value volatility, ticks per sqrt(second)
    lam_limit: float = 3.0             # background limit orders per second
    lam_noise: float = 0.5             # noise market orders per second
    lam_informed: float = 0.8          # informed market orders per second (when mispriced)
    informed_threshold: float = 0.5    # ticks of mispricing before informed traders act
    lp_alpha: float = 0.5              # weight liquidity providers put on V vs the mid
    depth_p: float = 0.35              # geometric parameter for posting distance
    cancel_prob: float = 0.08          # per-order cancel probability per second
    max_distance: int = 15             # background orders further than this are cancelled
    maker_fee: float = 0.0             # fee per unit filled for our passive orders (ticks; negative = rebate)
    latency: int = 0                   # steps between observing the book and quotes going live


@dataclass
class FillRecord:
    t: int
    side: int        # +1 we bought, -1 we sold
    price: int
    qty: int
    mid: float       # mid just before the fill


@dataclass
class MarketSim:
    params: MarketParams = field(default_factory=MarketParams)
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.book = OrderBook()
        p = self.params
        self.t = 0
        self.value = float(p.start_price)
        self.inventory = 0
        self.cash = 0.0
        self.fees = 0.0
        self.fills: list[FillRecord] = []
        self._mm_bid = None   # our resting order ids
        self._mm_ask = None
        self._pending = []    # quotes waiting for latency
        for _ in range(400):  # warm up the book
            self._post_background(n=1)
        self._seed_book()

    # ----- observation ---------------------------------------------------
    def observe(self) -> dict:
        b = self.book
        bq = b.level_qty(BID, b.best_bid) if b.best_bid is not None else 0
        aq = b.level_qty(ASK, b.best_ask) if b.best_ask is not None else 0
        mid = b.mid()
        micro = mid
        imb = 0.0
        if bq + aq > 0 and mid is not None:
            imb = (bq - aq) / (bq + aq)
            micro = (b.best_bid * aq + b.best_ask * bq) / (bq + aq)
        return dict(t=self.t, mid=mid, best_bid=b.best_bid, best_ask=b.best_ask,
                    bid_qty=bq, ask_qty=aq, imbalance=imb, microprice=micro,
                    inventory=self.inventory, cash=self.cash)

    # ----- our market maker ----------------------------------------------
    def set_quotes(self, bid: int | None, ask: int | None, size: int = 1):
        """Replace our quotes. Unchanged prices keep their queue position."""
        if self.params.latency > 0:
            self._pending.append((self.t + self.params.latency, bid, ask, size))
            return
        self._apply_quotes(bid, ask, size)

    def _apply_quotes(self, bid, ask, size):
        b = self.book
        # never cross the book with a passive quote
        if bid is not None and b.best_ask is not None and bid >= b.best_ask:
            bid = b.best_ask - 1
        if ask is not None and b.best_bid is not None and ask <= b.best_bid:
            ask = b.best_bid + 1
        if bid is not None and ask is not None and bid >= ask:
            bid, ask = ask - 1, ask
        self._mm_bid = self._replace(self._mm_bid, BID, bid, size)
        self._mm_ask = self._replace(self._mm_ask, ASK, ask, size)

    def _replace(self, oid, side, price, size):
        b = self.book
        live = oid is not None and oid in b.orders
        if live and price is not None and b.orders[oid].price == price:
            return oid  # keep queue priority
        if live:
            b.cancel(oid)
        if price is None:
            return None
        new_oid, fills = b.limit(side, price, size, owner="mm")
        self._record(fills, b.mid())
        return new_oid

    # ----- simulation step ------------------------------------------------
    def step(self):
        p, rng, b = self.params, self.rng, self.book
        while self._pending and self._pending[0][0] <= self.t:
            _, bid, ask, size = self._pending.pop(0)
            self._apply_quotes(bid, ask, size)

        self.value += p.sigma * rng.standard_normal()
        self._cancel_background()

        events = (["L"] * rng.poisson(p.lam_limit) + ["N"] * rng.poisson(p.lam_noise))
        mid = b.mid()
        if mid is not None and abs(self.value - mid) > p.informed_threshold:
            events += ["I"] * rng.poisson(p.lam_informed)
        rng.shuffle(events)
        for e in events:
            if e == "L":
                self._post_background(1)
            else:
                mid = b.mid()
                if e == "N":
                    side = BID if rng.random() < 0.5 else ASK
                else:
                    if mid is None or abs(self.value - mid) <= p.informed_threshold:
                        continue
                    side = BID if self.value > mid else ASK
                qty = 1 + rng.geometric(0.6) - 1
                self._record(b.market(side, qty), mid)
        if b.best_bid is None or b.best_ask is None:
            self._seed_book()
        self.t += 1

    # ----- internals ------------------------------------------------------
    def _record(self, fills, mid):
        for f in fills:
            if f.maker_owner != "mm":
                continue
            side = f.maker_side  # our bid filled -> we bought
            self.inventory += side * f.qty
            self.cash -= side * f.price * f.qty
            fee = self.params.maker_fee * f.qty
            self.cash -= fee
            self.fees += fee
            self.fills.append(FillRecord(self.t, side, f.price, f.qty, mid if mid is not None else f.price))

    def _reference(self):
        mid = self.book.mid()
        if mid is None:
            return self.value
        return mid + self.params.lp_alpha * (self.value - mid)

    def _post_background(self, n):
        p, rng, b = self.params, self.rng, self.book
        for _ in range(n):
            ref = self._reference()
            d = rng.geometric(p.depth_p) - 1
            qty = int(rng.geometric(0.5))
            if rng.random() < 0.5:
                price = int(np.floor(ref)) - d
                if b.best_ask is not None:
                    price = min(price, b.best_ask - 1)
                b.limit(BID, price, qty)
            else:
                price = int(np.ceil(ref)) + d
                if b.best_bid is not None:
                    price = max(price, b.best_bid + 1)
                b.limit(ASK, price, qty)

    def _cancel_background(self):
        p, rng, b = self.params, self.rng, self.book
        bg = [o for o in b.orders.values() if o.owner == "bg"]
        if not bg:
            return
        mid = b.mid()
        n = rng.binomial(len(bg), p.cancel_prob)
        idx = rng.choice(len(bg), size=n, replace=False) if n else []
        doomed = {bg[i].oid for i in idx}
        if mid is not None:
            doomed |= {o.oid for o in bg if abs(o.price - mid) > p.max_distance}
        for oid in doomed:
            b.cancel(oid)

    def _seed_book(self):
        """Refill an empty side (rare): a few orders around the value."""
        b = self.book
        v = int(round(self.value))
        if b.best_bid is None:
            for d in range(1, 4):
                b.limit(BID, (b.best_ask if b.best_ask is not None else v + 1) - d, 2)
        if b.best_ask is None:
            for d in range(1, 4):
                b.limit(ASK, (b.best_bid if b.best_bid is not None else v - 1) + d, 2)
