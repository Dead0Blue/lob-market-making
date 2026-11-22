"""Reproduce every number and figure in the README.

    python examples/run_experiments.py

Protocol
- Calibration and parameter choices use training seeds 0-299 only.
- The final comparison uses 50 test episodes (seeds 1000-1049, one hour each)
  never used for any choice. Every strategy sees the same seeds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmlab.backtest import evaluate, run_episode
from mmlab.calibration import estimate_intensity, estimate_sigma
from mmlab.market import MarketParams
from mmlab.replay import calibrate, load, run_replay
from mmlab.signals import bucket_curve, predictive_power, record_book
from mmlab.strategies import GLFT, FixedOffset, QLearningMM

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})
TEST_SEEDS = range(1000, 1050)
GAMMA = 0.5
lines = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)


# 1. Calibration ------------------------------------------------------------------
sigma = estimate_sigma()
A, k, dist, rates = estimate_intensity()
lines += ["## Calibration (training seeds)", "",
          f"- Mid volatility sigma = {sigma:.3f} ticks per sqrt(second)",
          f"- Fill intensity lambda(delta) = {A:.3f} x exp(-{k:.3f} delta) fills per second per side", ""]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.semilogy(dist, rates, "o", color=C[0], ms=6, label="measured (probe quotes)")
dd = np.linspace(dist.min(), dist.max(), 50)
ax.semilogy(dd, A * np.exp(-k * dd), color=MUTED, ls="--", lw=1.2, label=fr"fit $A e^{{-k\delta}}$, k = {k:.2f}")
ax.set(title="Fill rate falls exponentially with distance", xlabel="quote distance from mid (ticks)",
       ylabel="fills per second per side")
ax.legend()
save(fig, "calibration.png")

# 2. Signals ----------------------------------------------------------------------
books = [record_book(3600, seed=s) for s in range(5)]
data = {key: np.concatenate([b[key] for b in books]) for key in books[0]}
pp = predictive_power(data, horizons=(1, 5, 10, 30, 60))
lines += ["## Order-book signals (5 hours of simulated data)", "", "| Horizon (s) | IC of imbalance | IC of microprice - mid |", "|---:|---:|---:|"]
lines += [f"| {r['horizon']} | {r['ic_imbalance']:.3f} | {r['ic_micro_minus_mid']:.3f} |" for r in pp] + [""]
x, m, e = bucket_curve(data, horizon=5)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].errorbar(x, m, yerr=1.96 * e, fmt="o-", color=C[0], ms=5, capsize=3)
ax[0].axhline(0, color=MUTED, lw=1)
ax[0].set(title="Imbalance predicts the next move", xlabel="order-book imbalance (decile mean)",
          ylabel="mid change over next 5 s (ticks)")
hs = [r["horizon"] for r in pp]
ax[1].plot(hs, [r["ic_imbalance"] for r in pp], "o-", color=C[0], ms=5, label="imbalance")
ax[1].plot(hs, [r["ic_micro_minus_mid"] for r in pp], "o-", color=C[1], ms=5, label="microprice - mid")
ax[1].set(title="Information coefficient decays within a minute", xlabel="horizon (s)", ylabel="correlation with future move")
ax[1].legend()
save(fig, "signals.png")

# 3. Train the Q-learning agent ---------------------------------------------------
agent = QLearningMM(seed=0)
curve = []
for ep in range(600):
    agent.epsilon = max(0.02, 0.3 * (1 - ep / 400))
    curve.append(run_episode(agent, seed=ep, horizon=1800).pnl)
agent.training = False
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
roll = np.convolve(curve, np.ones(25) / 25, mode="valid")
ax[0].plot(curve, color=C[0], alpha=0.25, lw=1)
ax[0].plot(np.arange(24, len(curve)), roll, color=C[0])
ax[0].set(title="Q-learning training (30-min episodes)", xlabel="episode", ylabel="episode P&L (ticks)")
pol = agent.greedy_policy()
skew = np.array([[a - b for (b, a) in row] for row in pol])  # ask distance - bid distance
im = ax[1].imshow(skew.T, cmap="RdBu", vmin=-2, vmax=2, aspect="auto", origin="lower")
ax[1].set(title="Learned skew: ask distance - bid distance", xlabel="inventory", ylabel="imbalance")
ax[1].set_xticks(range(7), ["<=-6", "-5..-3", "-2..-1", "0", "1..2", "3..5", ">=6"], fontsize=8)
ax[1].set_yticks(range(3), ["sell-heavy", "neutral", "buy-heavy"], fontsize=8)
ax[1].grid(False)
fig.colorbar(im, ax=ax[1], fraction=0.046, label="ticks (<0: keen to sell)")
save(fig, "q_learning.png")

# 4. Out-of-sample comparison -----------------------------------------------------
strategies = [
    FixedOffset(1.0),
    GLFT(A=A, k=k, sigma=sigma, gamma=GAMMA, name="Avellaneda-Stoikov (mid)"),
    GLFT(A=A, k=k, sigma=sigma, gamma=GAMMA, use_microprice=True, name="Avellaneda-Stoikov + microprice"),
    agent,
]
results = [evaluate(s, TEST_SEEDS) for s in strategies]
lines += ["## Out-of-sample comparison (50 one-hour episodes, test seeds 1000-1049)", "",
          "| Strategy | Mean P&L | Std | Sharpe (per hour) | t-stat | Fills | Spread capture | Inventory P&L | 10 s markout | Mean abs inventory |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in results:
    lines.append(f"| {r['strategy']} | {r['mean_pnl']:.1f} | {r['std_pnl']:.1f} | {r['sharpe_per_episode']:.2f} | "
                 f"{r['t_stat']:.1f} | {r['fills']:.0f} | {r['spread_capture']:.1f} | {r['inventory_pnl']:.1f} | "
                 f"{r['markout_10s']:.1f} | {r['mean_abs_inventory']:.2f} |")
lines += ["", "P&L in ticks per hour, one unit per quote, after liquidating the final inventory at the half-spread.", ""]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for r, col in zip(results, C):
    paths = np.array([e.mtm for e in r["episodes"]])
    ax[0].plot(np.arange(paths.shape[1]) / 60, paths.mean(axis=0), color=col, label=r["strategy"])
ax[0].set(title="Average P&L path, 50 test hours", xlabel="minutes", ylabel="mark-to-market P&L (ticks)")
ax[0].legend(fontsize=8)
names = [r["strategy"].replace(" + ", "\n+ ").replace(" (", "\n(") for r in results]
xpos = np.arange(len(results)); w = 0.26
ax[1].bar(xpos - w, [r["spread_capture"] for r in results], w, color=C[0], label="spread capture")
ax[1].bar(xpos, [r["inventory_pnl"] for r in results], w, color=C[1], label="inventory P&L")
ax[1].bar(xpos + w, [r["mean_pnl"] for r in results], w, color=C[2], label="total P&L")
ax[1].axhline(0, color=MUTED, lw=1)
ax[1].set_xticks(xpos, names, fontsize=7.5)
ax[1].set(title="Where the P&L comes from", ylabel="ticks per hour")
ax[1].legend(fontsize=8)
save(fig, "comparison.png")

fig, ax = plt.subplots(figsize=(6, 3.6))
for r, col in zip(results, C):
    inv = np.concatenate([e.inventory for e in r["episodes"]])
    vals, counts = np.unique(inv, return_counts=True)
    ax.plot(vals, counts / counts.sum(), "o-", ms=3, color=col, label=r["strategy"])
ax.set(title="Inventory distribution", xlabel="inventory (units)", ylabel="share of time")
ax.legend(fontsize=8)
save(fig, "inventory.png")

ep = results[2]["episodes"][0]
sl = slice(600, 900)
t = np.arange(len(ep.mid))[sl] / 60
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(t, ep.mid[sl], color=INK, lw=1.2, label="mid")
ax.step(t, ep.quotes[sl, 0], where="post", color=C[0], lw=1, label="our bid")
ax.step(t, ep.quotes[sl, 1], where="post", color=C[1], lw=1, label="our ask")
ax2 = ax.twinx()
ax2.fill_between(t, ep.inventory[sl], step="post", color=C[2], alpha=0.2, label="inventory")
ax2.set_ylabel("inventory", color=MUTED); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax.set(title="Five minutes of Avellaneda-Stoikov + microprice quoting", xlabel="minutes", ylabel="price (ticks)")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
save(fig, "quotes_zoom.png")

# 5. Sensitivity to fees and latency (Avellaneda-Stoikov + microprice) ------------
sens_seeds = range(1000, 1020)
fees = [-0.1, 0.0, 0.1, 0.2]
lats = [0, 1, 2, 5]
fee_pnl = [evaluate(GLFT(A=A, k=k, sigma=sigma, gamma=GAMMA, use_microprice=True), sens_seeds,
                    MarketParams(maker_fee=f))["mean_pnl"] for f in fees]
lat_pnl = [evaluate(GLFT(A=A, k=k, sigma=sigma, gamma=GAMMA, use_microprice=True), sens_seeds,
                    MarketParams(latency=l))["mean_pnl"] for l in lats]
lines += ["## Sensitivity (Avellaneda-Stoikov + microprice, 20 test hours)", "",
          "| Maker fee (ticks per unit) | " + " | ".join(f"{f:+.1f}" for f in fees) + " |",
          "|---|" + "---:|" * len(fees),
          "| Mean P&L | " + " | ".join(f"{p:.1f}" for p in fee_pnl) + " |", "",
          "| Latency (seconds) | " + " | ".join(str(l) for l in lats) + " |",
          "|---|" + "---:|" * len(lats),
          "| Mean P&L | " + " | ".join(f"{p:.1f}" for p in lat_pnl) + " |", ""]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(fees, fee_pnl, "o-", color=C[0], ms=5); ax[0].axhline(0, color=MUTED, lw=1)
ax[0].set(title="P&L vs exchange maker fee", xlabel="fee per unit (ticks; negative = rebate)", ylabel="P&L per hour (ticks)")
ax[1].plot(lats, lat_pnl, "o-", color=C[1], ms=5); ax[1].axhline(0, color=MUTED, lw=1)
ax[1].set(title="P&L vs quote latency", xlabel="latency (seconds)", ylabel="P&L per hour (ticks)")
save(fig, "sensitivity.png")

# 6. Replay on recorded-format data ------------------------------------------------
d = load("data/sample_snapshots.csv", "data/sample_trades.csv", tick_size=0.01)
rs, rA, rk = calibrate(d)
rep = [(s.name, run_replay(s, d)) for s in (FixedOffset(1.0),
       GLFT(A=rA, k=rk, sigma=rs, gamma=GAMMA, use_microprice=True, name="Avellaneda-Stoikov + microprice"))]
lines += ["## Replay backtest (data/sample_*.csv: simulator output in the recorder's format)", "",
          f"Calibrated from the file: sigma = {rs:.3f}, A = {rA:.3f}, k = {rk:.3f}.", "",
          "| Strategy | P&L | Fills | Mean abs inventory |", "|---|---:|---:|---:|"]
lines += [f"| {n} | {r.pnl:.1f} | {r.n_fills} | {r.mean_abs_inventory:.2f} |" for n, r in rep] + [""]

Path("docs/results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
