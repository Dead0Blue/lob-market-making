"""Limit order book with price-time priority.

Prices are integer ticks. Each price level is a FIFO queue of orders; a
dictionary maps order ids to their order for O(1) cancels. Matching follows
the rules of a real exchange:

- a market order (or a marketable limit order) trades against the best
  opposite price first, oldest order first at each price;
- partial fills leave the remainder resting (limit) or are dropped (market);
- cancelling removes the order and frees its place in the queue.

Every trade is reported as a Fill with the resting order's id, so the owner
of a resting order (e.g. our market maker) knows exactly when it was filled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

BID, ASK = 1, -1


@dataclass
class Order:
    oid: int
    side: int          # BID (+1) or ASK (-1)
    price: int         # ticks
    qty: int
    owner: str = "bg"  # "bg" background liquidity, "mm" our market maker


@dataclass
class Fill:
    maker_oid: int
    maker_owner: str
    maker_side: int    # side of the resting order
    price: int
    qty: int


@dataclass
class OrderBook:
    bids: dict = field(default_factory=dict)   # price -> deque[Order]
    asks: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict) # oid -> Order
    best_bid: int | None = None
    best_ask: int | None = None
    _ids: count = field(default_factory=lambda: count(1))

    # ----- queries -------------------------------------------------------
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    def spread(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def level_qty(self, side: int, price: int) -> int:
        book = self.bids if side == BID else self.asks
        q = book.get(price)
        return sum(o.qty for o in q) if q else 0

    def depth(self, side: int, levels: int = 5):
        """[(price, total qty)] for the best `levels` prices of one side."""
        book = self.bids if side == BID else self.asks
        prices = sorted(book, reverse=(side == BID))[:levels]
        return [(p, sum(o.qty for o in book[p])) for p in prices]

    def queue_ahead(self, oid: int) -> int:
        """Quantity resting in front of order `oid` at its price level."""
        o = self.orders[oid]
        book = self.bids if o.side == BID else self.asks
        ahead = 0
        for other in book[o.price]:
            if other.oid == oid:
                return ahead
            ahead += other.qty
        raise KeyError(oid)

    # ----- order entry ---------------------------------------------------
    def limit(self, side: int, price: int, qty: int, owner: str = "bg"):
        """Submit a limit order. Returns (oid or None if fully filled, fills)."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        fills = self._match(side, qty, limit_price=price)
        remaining = qty - sum(f.qty for f in fills)
        if remaining == 0:
            return None, fills
        oid = next(self._ids)
        order = Order(oid, side, price, remaining, owner)
        book = self.bids if side == BID else self.asks
        book.setdefault(price, deque()).append(order)
        self.orders[oid] = order
        if side == BID and (self.best_bid is None or price > self.best_bid):
            self.best_bid = price
        if side == ASK and (self.best_ask is None or price < self.best_ask):
            self.best_ask = price
        return oid, fills

    def market(self, side: int, qty: int):
        """Market order: trades immediately against the opposite side."""
        return self._match(side, qty, limit_price=None)

    def cancel(self, oid: int) -> bool:
        order = self.orders.pop(oid, None)
        if order is None:
            return False
        book = self.bids if order.side == BID else self.asks
        level = book[order.price]
        level.remove(order)
        if not level:
            del book[order.price]
            self._refresh_best(order.side)
        return True

    # ----- internals -----------------------------------------------------
    def _match(self, side: int, qty: int, limit_price: int | None):
        fills = []
        book = self.asks if side == BID else self.bids
        while qty > 0:
            best = self.best_ask if side == BID else self.best_bid
            if best is None:
                break
            if limit_price is not None and (best > limit_price if side == BID else best < limit_price):
                break
            level = book[best]
            while qty > 0 and level:
                maker = level[0]
                traded = min(qty, maker.qty)
                fills.append(Fill(maker.oid, maker.owner, maker.side, best, traded))
                maker.qty -= traded
                qty -= traded
                if maker.qty == 0:
                    level.popleft()
                    del self.orders[maker.oid]
            if not level:
                del book[best]
                self._refresh_best(-side)
        return fills

    def _refresh_best(self, side: int):
        if side == BID:
            self.best_bid = max(self.bids) if self.bids else None
        else:
            self.best_ask = min(self.asks) if self.asks else None
