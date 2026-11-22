"""Record live order-book snapshots and trades from Binance (public data, no account).

Usage (needs internet):
    pip install websockets
    python scripts/record_binance.py BTCUSDT 3600     # symbol, seconds to record

Writes data/<symbol>_snapshots.csv and data/<symbol>_trades.csv in the format
read by mmlab.replay. Snapshots: top 5 levels, one row per second (the stream
updates every 100 ms; we keep the latest book each second). Trades: every
aggregated trade with its aggressor side.

Timestamps: trades carry the exchange time; the partial-depth stream has no
timestamp, so snapshots use the local clock. Keep your PC clock synced; the
small offset (tens of ms) does not matter at a 1-second step.
"""
import asyncio
import csv
import json
import sys
import time

import websockets

LEVELS = 5


async def record(symbol: str = "BTCUSDT", seconds: int = 3600):
    s = symbol.lower()
    url = f"wss://stream.binance.com:9443/stream?streams={s}@depth{LEVELS}@100ms/{s}@aggTrade"
    snap_path, trade_path = f"data/{symbol}_snapshots.csv", f"data/{symbol}_trades.csv"
    latest, last_written, n_trades = None, 0, 0
    end = time.time() + seconds
    with open(snap_path, "w", newline="") as fs, open(trade_path, "w", newline="") as ft:
        ws_s, ws_t = csv.writer(fs), csv.writer(ft)
        ws_s.writerow(["ts"] + [f"bid_px_{k}" for k in range(1, LEVELS + 1)] + [f"bid_qty_{k}" for k in range(1, LEVELS + 1)]
                      + [f"ask_px_{k}" for k in range(1, LEVELS + 1)] + [f"ask_qty_{k}" for k in range(1, LEVELS + 1)])
        ws_t.writerow(["ts", "price", "qty", "side"])
        async with websockets.connect(url, ping_interval=20) as ws:
            while time.time() < end:
                msg = json.loads(await ws.recv())
                data = msg["data"]
                if msg["stream"].endswith("aggTrade"):
                    # m = True: the buyer was the maker, so the aggressor sold
                    side = -1 if data["m"] else 1
                    ws_t.writerow([data["T"] / 1000, data["p"], data["q"], side])
                    n_trades += 1
                else:
                    latest = data
                now = time.time()
                if latest and int(now) > last_written:
                    last_written = int(now)
                    bids, asks = latest["bids"][:LEVELS], latest["asks"][:LEVELS]
                    ws_s.writerow([round(now, 3)] + [b[0] for b in bids] + [b[1] for b in bids]
                                  + [a[0] for a in asks] + [a[1] for a in asks])
                    if last_written % 60 == 0:
                        print(f"{int(end - now)} s left, {n_trades} trades recorded", flush=True)
    print(f"done: {snap_path}, {trade_path}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    asyncio.run(record(sym, secs))
