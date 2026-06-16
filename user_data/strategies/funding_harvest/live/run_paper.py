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

from .broker import DryRunLiveBroker
from .config import StrategyConfig
from .engine import Engine
from .feed import LiveFeed
from .risk import RiskMonitor
from .signals import build_target


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


_FWD_DIR = "user_data/backtest_results"
STATE_PATH = os.path.join(_FWD_DIR, "funding_harvest_forward_state.json")
LOG_PATH = os.path.join(_FWD_DIR, "funding_harvest_forward.csv")


def forward_test(state_path: str = STATE_PATH, log_path: str = LOG_PATH,
                 is_rebalance: bool = True):
    """ONE forward-test tick. Restores the prior book, pulls a live snapshot,
    accrues/rebalances on the dry-run broker, persists state, and appends a row
    for the realized-vs-model comparison. Schedule this once per 8h settlement
    EXTERNALLY (cron / freqtrade loop / Claude /loop) - it must NOT busy-wait.

    Each row is what you check before trusting capital: does the live-data book
    track the replay's funding/cost/equity path?"""
    import csv

    from .state import load_state, save_state

    cfg = _live_cfg()
    feed = LiveFeed(cfg)
    broker = DryRunLiveBroker(cfg)
    risk = RiskMonitor(cfg)
    eng = Engine(cfg, feed, broker, risk)

    warm = load_state(broker, risk, state_path)     # continue the same book

    step = feed.snapshot(is_rebalance=is_rebalance)
    broker.accrue(step["accrual"])
    flags = risk.check(step, broker)
    if step["is_rebalance"] and not risk.halted:
        desired = build_target(step["sig"], step["vol"], broker.positions(),
                               broker.equity(), cfg)
        eng._execute(desired, step["prices"])

    save_state(broker, risk, state_path)

    row = dict(
        time=f"{step['t']:%Y-%m-%d %H:%M}", cold_start=not warm,
        n_pairs=step["n_pairs"], n_positions=len(broker.positions()),
        equity=round(broker.equity(), 2), gross_funding=round(broker.gross_funding, 2),
        borrow=round(broker.borrow, 2), costs=round(broker.costs, 2),
        orders_this_tick=len(broker.orders),
        risk=";".join(f"{e.level}:{e.kind}" for e in flags) or "none",
    )
    new = not os.path.exists(log_path)
    os.makedirs(_FWD_DIR, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"forward-test tick appended -> {log_path}\n  {row}")


if __name__ == "__main__":
    one_tick()
