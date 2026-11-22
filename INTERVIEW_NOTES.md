# Interview notes: how to present and defend this project

Read this, run everything, and change something before you put it on your CV. Market makers (Citadel Securities, Optiver) and bank electronic-trading teams will go deep on each design choice.

## 60-second pitch
"I built a limit order book simulator with informed and uninformed traders, and compared four market-making strategies on it. A naive fixed-spread quoter makes money but carries huge inventory risk. The Avellaneda-Stoikov model, which I calibrated from fill rates, gets five times the Sharpe ratio by skewing quotes against inventory. Centring quotes on the microprice, an order-book signal, adds 26% more profit by cutting adverse selection. A Q-learning agent learned inventory skewing by itself but stayed too conservative. I also showed that two seconds of latency wipes out the edge. There's a replay backtester for real Binance data and a C++ matching engine doing about 6 million messages a second."

## The concepts, simply

**Market maker.** Quotes a price to buy (bid) and a price to sell (ask) all day. Earns the gap (spread) when both sides trade. Risks: holding inventory when the price moves, and being picked off by traders who know more.

**Order book and priority.** Orders at the same price form a queue. First come, first served. Changing your price sends you to the back of the queue, which is why my simulator keeps an order in place when its price doesn't change.

**Adverse selection.** You get filled most when your price is wrong. Informed traders buy from you just before the price rises. Measured by the **markout**: where the mid is 10 seconds after each fill, relative to the fill.

**P&L attribution.** Total P&L = spread capture + inventory P&L − fees − liquidation. Spread capture is how far from the mid you traded, in your favour. Inventory P&L is gains or losses from holding a position while the price moves. In my results, adverse selection eats most of the spread. That's normal in real markets too.

**Avellaneda-Stoikov.** Two ideas:
1. **The spread** must pay for fill probability and risk. The constant c₁ = ln(1 + γ/k)/γ is the base half-spread.
2. **Inventory skew:** when you are long, lower both quotes, so you sell more easily and buy less. Each unit of inventory moves the quotes by c₂.

k is how fast fills disappear as you quote further away; I measured it with probe quotes. γ is risk aversion; I chose 0.5 on training seeds only.

**Microprice.** If 900 units sit on the bid and 100 on the ask, the ask is about to be eaten, so the "true" price is closer to the ask than the mid says. Microprice = (P_bid·Q_ask + P_ask·Q_bid)/(Q_bid + Q_ask).

**Q-learning.** The agent sees a state (inventory bucket, imbalance bucket) and picks an action (how far to quote on each side). After each 5-second interval it gets a reward: P&L change minus a penalty for holding inventory. It updates a table of action values: Q(s,a) ← Q(s,a) + α (reward + discount × best Q(next state) − Q(s,a)). It explores randomly at first (ε-greedy), less as it learns. Link this to your RL thesis.

**Sharpe here.** Mean hourly P&L divided by its standard deviation across 50 test hours. The naive strategy has the highest mean but terrible Sharpe, which is a good point to make in an interview.

## Questions you'll get, and answers

1. **"Why does the naive strategy earn more but you say it's worse?"** Its P&L comes with 8× the volatility and ±10 units of inventory, so per unit of risk it is 5 times worse. Scaled to the same risk, Avellaneda-Stoikov would trade larger size and earn much more.
2. **"How did you avoid overfitting?"** Calibration and γ come from training seeds; the comparison uses 50 test seeds never touched. Every strategy sees the same seeds.
3. **"Is the simulator realistic?"** It reproduces the key facts: 1–3 tick spreads, queues, mean-reverting order flow at 1 s, informative imbalance (IC 0.21 at 1 s, decaying within a minute), and adverse selection. Limitations: no self-exciting order flow (Hawkes), one asset, no hidden orders. Next step: real data via the recorder.
4. **"What does the replay assume?"** You join the back of the displayed queue; trades through your price fill you; trades at your price fill you only after the queue ahead is used up. It ignores your own impact, so replay P&L is optimistic.
5. **"Why does latency hurt so much?"** Stale quotes are exactly what informed traders hit. At 2 s, our quotes reflect an old book, so we're adversely selected on most fills.
6. **"Why didn't RL beat Avellaneda-Stoikov?"** Small action set (distances of at least 1 tick from the microprice), tabular state, and a sparse, noisy reward. It found the right skew but chose wide spreads when flat. Improvements: finer actions, policy-gradient methods, a reward based on risk-adjusted P&L, more training.
7. **"How does your C++ order book work?"** Each side is a `std::map` from price to a FIFO list; a hash map from id to list iterator gives O(1) cancel. Best bid is `rbegin()`, best ask `begin()`. Faster designs use flat arrays indexed by price tick and pre-allocated order pools to avoid allocation.
8. **"Explain the markout."** For each fill, side × (mid 10 s later − mid at fill). Negative means the price moved against you after you traded, which is adverse selection.

## CV line (only once you can defend everything above)
> **Market making on a limit order book** (Python, C++17): built a price-time-priority order book simulator with informed flow; calibrated Avellaneda-Stoikov (GLFT) quotes from fill intensities, reaching 5× the Sharpe of a naive quoter out of sample; microprice signal (IC 0.19 at 5 s) added 26% P&L by reducing adverse selection; Q-learning agent learned inventory skewing; replay backtester and recorder for live Binance data; C++ matching engine at 5.8M messages/s.

## Before you publish
- Run `pytest`, `examples/run_experiments.py` and the C++ build yourself.
- Record at least a few hours of real BTC-USDT data, run the replay, and add those results to the README (honestly, even if they're worse).
- Commit your own improvements step by step.
