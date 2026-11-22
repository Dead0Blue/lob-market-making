## Calibration (training seeds)

- Mid volatility sigma = 0.345 ticks per sqrt(second)
- Fill intensity lambda(delta) = 0.258 x exp(-1.697 delta) fills per second per side

## Order-book signals (5 hours of simulated data)

| Horizon (s) | IC of imbalance | IC of microprice - mid |
|---:|---:|---:|
| 1 | 0.213 | 0.186 |
| 5 | 0.191 | 0.158 |
| 10 | 0.126 | 0.098 |
| 30 | 0.038 | 0.025 |
| 60 | 0.013 | 0.010 |

## Out-of-sample comparison (50 one-hour episodes, test seeds 1000-1049)

| Strategy | Mean P&L | Std | Sharpe (per hour) | t-stat | Fills | Spread capture | Inventory P&L | 10 s markout | Mean abs inventory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive (fixed spread) | 153.4 | 139.5 | 1.10 | 7.8 | 297 | 326.8 | -169.4 | -154.2 | 4.97 |
| Avellaneda-Stoikov (mid) | 97.4 | 17.5 | 5.57 | 39.4 | 897 | 507.9 | -410.1 | -403.9 | 0.52 |
| Avellaneda-Stoikov + microprice | 123.0 | 21.4 | 5.76 | 40.7 | 823 | 488.8 | -365.4 | -348.3 | 0.59 |
| Q-learning agent | 49.9 | 18.9 | 2.63 | 18.6 | 100 | 115.7 | -65.6 | -63.4 | 0.26 |

P&L in ticks per hour, one unit per quote, after liquidating the final inventory at the half-spread.

## Sensitivity (Avellaneda-Stoikov + microprice, 20 test hours)

| Maker fee (ticks per unit) | -0.1 | +0.0 | +0.1 | +0.2 |
|---|---:|---:|---:|---:|
| Mean P&L | 206.9 | 124.1 | 41.3 | -41.5 |

| Latency (seconds) | 0 | 1 | 2 | 5 |
|---|---:|---:|---:|---:|
| Mean P&L | 124.1 | 63.4 | 4.3 | -161.8 |

## Replay backtest (data/sample_*.csv: simulator output in the recorder's format)

Calibrated from the file: sigma = 0.332, A = 0.476, k = 1.586.

| Strategy | P&L | Fills | Mean abs inventory |
|---|---:|---:|---:|
| Naive (fixed spread) | 97.0 | 277 | 5.04 |
| Avellaneda-Stoikov + microprice | 101.0 | 822 | 0.77 |

