"""Paper-LIVE runner for #2 (forward-test harness). Runs the live decision path
against REAL Binance public data with a DryRunLiveBroker (simulated fills, no keys).

  one_tick()     - pull one live snapshot, build the market-neutral book, simulate
                   the rebalance orders, print the book + tickets + risk flags.
  forward_test() - one schedulable 8h tick with state persistence + CSV logging.

Promotion to capital = swap DryRunLiveBroker -> LiveBroker(enable_live=True) AFTER a
>=4-6 week forward test tracks the replay. Exercise the order path on the futures
testnet first (testnet.py).
"""
from __future__ import annotations

import os

from .broker import DryRunLiveBroker
from .config import TrendConfig
from .engine import Engine
from .feed import LiveFeed
from .risk import RiskMonitor
from .signals import build_target

_FWD_DIR = "user_data/backtest_results"
STATE_PATH = os.path.join(_FWD_DIR, "trend_momentum_forward_state.json")
LOG_PATH = os.path.join(_FWD_DIR, "trend_momentum_forward.csv")


def one_tick():
    cfg = TrendConfig()
    feed = LiveFeed(cfg)
    broker = DryRunLiveBroker(cfg)
    risk = RiskMonitor(cfg)
    print("=" * 80)
    print("PAPER-LIVE #2 one tick against REAL Binance public data (dry-run, no keys)")
    print("=" * 80)
    print(f"Pulling live perp candles + funding for {len(feed.pairs)} pairs ...")
    step = feed.snapshot(is_rebalance=True)
    print(f"  signal for {step['n_pairs']} pairs at {step['t']:%Y-%m-%d %H:%M}Z")

    broker.accrue(step["accrual"])
    flags = risk.check(step, broker)
    target = build_target(step["sig"], step["vol"], broker.equity(), cfg)
    broker.set_book(target, step["prices"])

    longs = {p: n for p, n in target.items() if n > 0}
    shorts = {p: n for p, n in target.items() if n < 0}
    gross = sum(abs(n) for n in target.values())
    net = sum(target.values())
    print(f"\nMarket-neutral book: {len(target)} names "
          f"({len(longs)} long / {len(shorts)} short)")
    print(f"  gross ${gross:,.0f} ({gross/broker.equity():.1f}x)   "
          f"net ${net:+,.0f} ({net/broker.equity():+.1%} of equity)")
    top = sorted(target.items(), key=lambda kv: -abs(kv[1]))[:10]
    for p, n in top:
        print(f"    {p:6s} {'LONG ' if n > 0 else 'SHORT':5s} ${abs(n):7,.0f}"
              f"   momentum z {step['sig'].get(p, float('nan')):+.2f}")
    print(f"\nSimulated maker orders this tick: {len(broker.orders)} "
          f"(single-leg perp, postOnly, clientOrderId)")
    print(f"Risk flags: {[(e.level, e.kind) for e in flags] or 'none'}")
    print("\nNOTE: dry-run. Real orders require LiveBroker(enable_live=True) + keys,")
    print("      after a >=4-6 week forward test. Testnet-validate first (testnet.py).")


def forward_test(state_path=STATE_PATH, log_path=LOG_PATH, is_rebalance=True):
    """One schedulable 8h tick: restore state, snapshot, rebalance, persist, log a
    realized-vs-model row. Schedule externally; never busy-wait."""
    import csv

    from .state import load_state, save_state

    cfg = TrendConfig()
    feed, broker, risk = LiveFeed(cfg), DryRunLiveBroker(cfg), RiskMonitor(cfg)
    warm = load_state(broker, risk, state_path)

    step = feed.snapshot(is_rebalance=is_rebalance)
    broker.accrue(step["accrual"])
    flags = risk.check(step, broker)
    if step["is_rebalance"] and not risk.halted:
        target = build_target(step["sig"], step["vol"], broker.equity(), cfg)
        broker.set_book(target, step["prices"])
    save_state(broker, risk, state_path)

    pos = broker.positions()
    row = dict(time=f"{step['t']:%Y-%m-%d %H:%M}", cold_start=not warm,
               n_pairs=step["n_pairs"], n_positions=len(pos),
               equity=round(broker.equity(), 2),
               gross=round(sum(abs(n) for n in pos.values()), 2),
               net=round(sum(pos.values()), 2),
               price_pnl=round(broker.price_pnl, 2), funding=round(broker.funding, 2),
               costs=round(broker.costs, 2), orders=len(broker.orders),
               risk=";".join(f"{e.level}:{e.kind}" for e in flags) or "none")
    new = not os.path.exists(log_path)
    os.makedirs(_FWD_DIR, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"#2 forward-test tick appended -> {log_path}\n  {row}")


if __name__ == "__main__":
    one_tick()
