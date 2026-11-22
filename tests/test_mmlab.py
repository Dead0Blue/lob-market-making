import numpy as np
import pytest

from mmlab.backtest import evaluate, run_episode
from mmlab.market import MarketParams, MarketSim
from mmlab.orderbook import ASK, BID, OrderBook
from mmlab.replay import calibrate, export_synthetic, load, run_replay
from mmlab.signals import predictive_power, record_book
from mmlab.strategies import GLFT, FixedOffset, QLearningMM


# ---------------- order book ----------------
def test_price_time_priority_and_partial_fills():
    b = OrderBook()
    o1, _ = b.limit(BID, 99, 5)
    o2, _ = b.limit(BID, 99, 3)
    b.limit(BID, 98, 10)
    fills = b.market(ASK, 6)
    assert [(f.maker_oid, f.qty) for f in fills] == [(o1, 5), (o2, 1)]
    assert b.level_qty(BID, 99) == 2 and b.best_bid == 99


def test_walk_the_book_and_marketable_limit():
    b = OrderBook()
    b.limit(ASK, 101, 2)
    b.limit(ASK, 102, 3)
    oid, fills = b.limit(BID, 102, 6)
    assert [(f.price, f.qty) for f in fills] == [(101, 2), (102, 3)]
    assert oid is not None and b.best_bid == 102 and b.best_ask is None


def test_cancel_and_queue_ahead():
    b = OrderBook()
    a, _ = b.limit(BID, 50, 4)
    c, _ = b.limit(BID, 50, 2)
    assert b.queue_ahead(c) == 4
    assert b.cancel(a) and not b.cancel(a)
    assert b.queue_ahead(c) == 0
    assert b.cancel(c) and b.best_bid is None


# ---------------- simulator ----------------
def test_book_is_never_crossed_and_mid_tracks_value():
    sim = MarketSim(seed=3)
    gaps = []
    for _ in range(2000):
        sim.step()
        b = sim.book
        assert b.best_bid < b.best_ask
        gaps.append(sim.value - b.mid())
    assert np.sqrt(np.mean(np.square(gaps))) < 3.0


def test_simulation_is_reproducible():
    a = run_episode(FixedOffset(1.0), seed=5, horizon=600)
    b = run_episode(FixedOffset(1.0), seed=5, horizon=600)
    assert a.pnl == b.pnl and a.n_fills == b.n_fills


def test_pnl_attribution_is_exact():
    r = run_episode(FixedOffset(1.0), seed=1, horizon=1200)
    assert r.mtm[-1] == pytest.approx(r.spread_capture + r.inventory_pnl - r.fees)
    assert r.pnl == pytest.approx(r.mtm[-1] - r.liquidation_cost)


def test_inventory_limit_respected():
    r = run_episode(FixedOffset(0.5, max_inventory=3), seed=2, horizon=1800)
    assert np.abs(r.inventory).max() <= 3 + 1  # at most one in-flight fill beyond the limit


def test_maker_fee_reduces_pnl_by_exactly_the_fees():
    free = run_episode(FixedOffset(1.0), seed=4, horizon=900)
    paid = run_episode(FixedOffset(1.0), MarketParams(maker_fee=0.1), seed=4, horizon=900)
    assert paid.fees == pytest.approx(0.1 * paid.volume)
    assert paid.pnl == pytest.approx(free.pnl - paid.fees)  # fees don't change the market path


# ---------------- signals ----------------
def test_imbalance_predicts_short_term_moves_and_decays():
    res = predictive_power(record_book(3600, seed=0), horizons=(1, 60))
    assert res[0]["ic_imbalance"] > 0.1
    assert res[1]["ic_imbalance"] < res[0]["ic_imbalance"] / 2


# ---------------- strategies ----------------
def test_glft_skews_quotes_against_inventory():
    s = GLFT(A=0.5, k=1.5, sigma=0.35, gamma=0.5)
    obs = dict(mid=100.5, microprice=100.5, inventory=0)
    flat = s.quote(obs)
    long = s.quote({**obs, "inventory": 4})
    short = s.quote({**obs, "inventory": -4})
    assert long[0] < flat[0] and long[1] <= flat[1]      # long: lower both quotes
    assert short[0] >= flat[0] and short[1] > flat[1]    # short: raise both quotes


def test_glft_beats_naive_on_risk_adjusted_basis():
    seeds = range(500, 510)
    naive = evaluate(FixedOffset(1.0), seeds)
    glft = evaluate(GLFT(A=0.5, k=1.6, sigma=0.35, gamma=0.5), seeds)
    assert glft["sharpe_per_episode"] > naive["sharpe_per_episode"]
    assert glft["mean_abs_inventory"] < naive["mean_abs_inventory"]


def test_q_learning_learns_inventory_skew():
    ag = QLearningMM(seed=0)
    for ep in range(120):
        ag.epsilon = max(0.05, 0.3 * (1 - ep / 80))
        run_episode(ag, seed=ep, horizon=1200)
    policy = ag.greedy_policy()
    long_bid, long_ask = policy[5][1]    # inventory 3..5, neutral imbalance
    short_bid, short_ask = policy[1][1]  # inventory -5..-3
    assert long_ask <= long_bid          # when long, the ask is closer than the bid
    assert short_bid <= short_ask        # when short, the bid is closer


# ---------------- replay ----------------
def test_replay_pipeline(tmp_path):
    prefix = str(tmp_path / "s")
    export_synthetic(prefix, seconds=900, seed=1)
    d = load(prefix + "_snapshots.csv", prefix + "_trades.csv", tick_size=0.01)
    assert len(d.ts) == 900 and len(d.trade_ts) > 100
    sigma, A, k = calibrate(d)
    assert 0.1 < sigma < 1.0 and A > 0 and k > 0
    r = run_replay(GLFT(A=A, k=k, sigma=sigma, gamma=0.5), d)
    assert r.n_fills > 10
    assert r.mtm.shape == (900,)
