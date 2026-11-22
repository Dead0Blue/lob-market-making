"""Market-making strategies. Each returns the bid and ask prices (in ticks) to
quote given what it observes; returning None for a side means "don't quote".

All strategies share the same hard inventory limit: at +max_inventory we stop
bidding, at -max_inventory we stop offering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _clip_inventory(bid, ask, inventory, max_inventory):
    if inventory >= max_inventory:
        bid = None
    if inventory <= -max_inventory:
        ask = None
    return bid, ask


class Strategy:
    name = "base"
    max_inventory = 10

    def reset(self, seed=None):
        pass

    def quote(self, obs) -> tuple[int | None, int | None]:
        raise NotImplementedError

    def on_step_end(self, obs_next, pnl_change):  # hook for learning agents
        pass


@dataclass
class FixedOffset(Strategy):
    """Quote at a fixed distance from the mid, both sides. Also used as a probe
    to measure how fill rates fall with distance (calibration)."""
    offset: float = 1.0
    max_inventory: int = 10
    name: str = "Naive (fixed spread)"

    def quote(self, obs):
        mid = obs["mid"]
        bid = int(math.floor(mid - self.offset + 1e-9))
        ask = int(math.ceil(mid + self.offset - 1e-9))
        return _clip_inventory(bid, ask, obs["inventory"], self.max_inventory)


@dataclass
class GLFT(Strategy):
    """Avellaneda-Stoikov market maker in the closed form of Gueant, Lehalle and
    Fernandez-Tapia (2013), which removes the dependence on the end time.

    Fills arrive at rate A exp(-k delta) at distance delta from the reference
    price. With risk aversion gamma and volatility sigma, the optimal distances
    for inventory q are
        delta_bid = c1 + (2q + 1)/2 * c2
        delta_ask = c1 - (2q - 1)/2 * c2
    with c1 = ln(1 + gamma/k)/gamma and
         c2 = sqrt(sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma)).
    c1 is the half-spread that pays for adverse selection and fills; c2 is the
    skew per unit of inventory: a long market maker lowers both quotes to sell
    more and buy less.

    use_microprice=True centres quotes on the microprice instead of the mid,
    i.e. uses the order-book imbalance signal.
    """
    A: float = 1.0
    k: float = 1.0
    sigma: float = 0.4
    gamma: float = 0.1
    max_inventory: int = 10
    use_microprice: bool = False
    name: str = "Avellaneda-Stoikov (GLFT)"

    def __post_init__(self):
        g, k = self.gamma, self.k
        self.c1 = math.log1p(g / k) / g
        self.c2 = math.sqrt(self.sigma**2 * g / (2 * k * self.A) * (1 + g / k) ** (1 + k / g))

    def quote(self, obs):
        ref = obs["microprice"] if self.use_microprice else obs["mid"]
        q = obs["inventory"]
        d_bid = self.c1 + (2 * q + 1) / 2 * self.c2
        d_ask = self.c1 - (2 * q - 1) / 2 * self.c2
        bid = int(math.floor(ref - d_bid + 0.5))
        ask = int(math.ceil(ref + d_ask - 0.5))
        if bid >= ask:
            bid = ask - 1
        return _clip_inventory(bid, ask, q, self.max_inventory)


class QLearningMM(Strategy):
    """Tabular Q-learning market maker.

    State  : inventory bucket (7) x order-book imbalance bucket (3)
    Action : bid distance in {1, 2, 3} ticks x ask distance in {1, 2, 3} ticks
             from the microprice (asymmetric distances = skew)
    Reward : change in mark-to-market P&L over the decision interval minus an
             inventory penalty eta * q^2 (the same risk term as Avellaneda-Stoikov)
    Decisions are taken every `interval` seconds; between decisions the quotes
    are re-centred on the moving microprice with the same distances.
    """
    INV_EDGES = np.array([-5.5, -2.5, -0.5, 0.5, 2.5, 5.5])
    IMB_EDGES = np.array([-1 / 3, 1 / 3])
    DISTANCES = (1, 2, 3)

    def __init__(self, alpha=0.05, discount=0.95, epsilon=0.1, eta=0.02, interval=5,
                 max_inventory=10, seed=0):
        self.name = "Q-learning agent"
        self.alpha, self.discount, self.epsilon, self.eta = alpha, discount, epsilon, eta
        self.interval, self.max_inventory = interval, max_inventory
        self.actions = [(b, a) for b in self.DISTANCES for a in self.DISTANCES]
        self.Q = np.zeros((len(self.INV_EDGES) + 1, len(self.IMB_EDGES) + 1, len(self.actions)))
        self.visits = np.zeros_like(self.Q)
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.reset()

    def state(self, obs):
        return (int(np.searchsorted(self.INV_EDGES, obs["inventory"])),
                int(np.searchsorted(self.IMB_EDGES, obs["imbalance"])))

    def reset(self, seed=None):
        self._s = self._a = None
        self._reward = 0.0
        self._clock = 0

    def _choose(self, s):
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.Q[s]))

    def quote(self, obs):
        if self._a is None or self._clock % self.interval == 0:
            s = self.state(obs)
            if self._a is not None and self.training:
                target = self._reward + self.discount * self.Q[s].max()
                self.Q[self._s][self._a] += self.alpha * (target - self.Q[self._s][self._a])
                self.visits[self._s][self._a] += 1
            self._s, self._a, self._reward = s, self._choose(s), 0.0
        self._clock += 1
        db, da = self.actions[self._a]
        ref = obs["microprice"]
        bid = int(math.floor(ref - db + 0.5))
        ask = int(math.ceil(ref + da - 0.5))
        if bid >= ask:
            bid = ask - 1
        return _clip_inventory(bid, ask, obs["inventory"], self.max_inventory)

    def on_step_end(self, obs_next, pnl_change):
        self._reward += pnl_change - self.eta * obs_next["inventory"] ** 2

    def greedy_policy(self):
        """Best action per state, as (bid distance, ask distance)."""
        return [[self.actions[int(np.argmax(self.Q[i, j]))] for j in range(self.Q.shape[1])]
                for i in range(self.Q.shape[0])]
