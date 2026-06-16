"""Paper-LIVE runner - the forward-test harness (Step E).

Runs the EXACT live decision/execution path against REAL Binance public data
(funding, mark, spot) with a DryRunLiveBroker that simulates the two-leg maker
order lifecycle and never sends a real order. No API keys required.

  * one_tick()        - pull one live snapshot, build the target book, simulate the
                        orders, print the book + tickets + risk flags. Proves the
                        live path works end-to-end right now.
  * forward_test()    - the loop you actually run for >=4-6 weeks: snapshot on the
                        8h settlement cadence, log decisions + simulated fills +
                        live funding to CSV so realized-vs-model can be compared
                        before any capital. (Scaffold; schedule externally.)

Promotion to real capital = swap DryRunLiveBroker -> LiveBroker(enable_live=True)
ONLY after the forward test tracks the replay within tolerance.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from .config import StrategyConfig  # noqa: E402
from .broker import DryRunLiveBroker  # noqa: E402
from .engine import Engine  # noqa: E402
from .feed import LiveFeed  # noqa: E402
from .risk import RiskMonitor  # noqa: E402
from .signals import build_target  # noqa: E402


def _live_cfg() -> StrategyConfig:
    cfg = StrategyConfig()
    br = json.load(open(cfg.borrow_file))
    cfg.borrow_rate = {c: v["ann"] for c, v in br.items()}
    cfg.borrow_limit = {c: v["limit_usd"] for c, v in br.items() if v.get("limit_usd")}
    cfg.perp_leverage = 3.0
    cfg.thresh_ann = 0.05
    return cfg


def one_tick():
    cfg = _live_cfg()
    feed = LiveFeed(cfg)
    broker = DryRunLiveBroker(cfg)
    risk = RiskMonitor(cfg)
    eng = Engine(cfg, feed, broker, risk)

    print("=" * 80)
    print("PAPER-LIVE  one tick against REAL Binance public data (dry-run, no keys)")
    print("=" * 80)
    print(f"Pulling live funding/mark/spot for {len(feed.pairs)} pairs ...")
    step = feed.snapshot(is_rebalance=True)
    print(f"  got live data for {step['n_pairs']} pairs at {step['t']:%Y-%m-%d %H:%M}Z")

    broker.accrue(step["accrual"])
    flags = risk.check(step, broker)
    desired = build_target(step["sig"], step["vol"], broker.positions(),
                           broker.equity(), cfg)
    eng._execute(desired, step["prices"])

    # show the would-be book
    longs = [(p, t) for p, t in desired.items() if t.side > 0]
    shorts = [(p, t) for p, t in desired.items() if t.side < 0]
    print(f"\nTarget book: {len(desired)} delta-neutral pairs "
          f"({len(longs)} +carry short-perp, {len(shorts)} -carry long-perp)")
    print("  Top by size (live annualised funding / position):")
    top = sorted(desired.items(), key=lambda kv: -kv[1].notional)[:12]
    for p, t in top:
        f = step["sig"].get(p, float("nan"))
        leg = "short-perp/long-spot" if t.side > 0 else "long-perp/short-spot"
        print(f"    {p:6s} {leg:21s} ${t.notional:8,.0f}  funding {f:+6.1%}/yr")

    print(f"\nSimulated maker orders this tick: {len(broker.orders)} tickets "
          f"(both legs postOnly, clientOrderId-keyed)")
    for o in broker.orders[:6]:
        print(f"    [{o.coid}] {o.action:6s} {o.pair:6s} "
              f"spot={o.leg_spot:4s} perp={o.leg_perp:4s} ${o.notional:8,.0f} DRY-RUN")
    if len(broker.orders) > 6:
        print(f"    ... +{len(broker.orders)-6} more")

    print(f"\nRisk flags this tick: {[(e.level, e.kind) for e in flags] or 'none'}")
    print(f"Deployed: ${sum(t.notional for t in desired.values()):,.0f} of "
          f"${broker.equity():,.0f} equity  |  borrow side capped by real VIP0 limits")
    print("\nNOTE: dry-run. Real orders require LiveBroker(enable_live=True) + keys,")
    print("      and only after a >=4-6 week forward test tracks the replay.")


def forward_test(log_path="user_data/backtest_results/funding_harvest_forward.csv"):
    """Scaffold: call snapshot() once per 8h settlement, persist a comparison row.
    Schedule this externally (cron / freqtrade loop); do NOT busy-wait here."""
    import csv
    cfg = _live_cfg()
    feed = LiveFeed(cfg)
    broker = DryRunLiveBroker(cfg)          # persist/restore between runs in real use
    risk = RiskMonitor(cfg)
    eng = Engine(cfg, feed, broker, risk)

    step = feed.snapshot(is_rebalance=True)
    broker.accrue(step["accrual"])
    risk.check(step, broker)
    desired = build_target(step["sig"], step["vol"], broker.positions(),
                           broker.equity(), cfg)
    eng._execute(desired, step["prices"])
    row = dict(time=step["t"], n_pairs=step["n_pairs"], n_positions=len(desired),
               equity=broker.equity(), gross_funding=broker.gross_funding,
               costs=broker.costs, orders=len(broker.orders))
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"forward-test row appended -> {log_path}: {row}")


if __name__ == "__main__":
    one_tick()
