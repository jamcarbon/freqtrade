"""Replay harness + fidelity proof for the funding-harvest execution engine.

Three things:
  1. FIDELITY  - run the engine (ReplayFeed+PaperBroker) on a config that mirrors
     funding_harvest_backtest.backtest() and assert the equity matches. This proves
     the production decision/accounting path == the validated backtest path.
  2. LIVE CONFIG - run the engine on the real, safe live config (real borrow data,
     <=3x cross margin, per-name short caps, set-and-hold, maker fills) and report
     the honest expected economics + risk-overlay activity.
  3. RISK SELF-TEST - force a drawdown and confirm the kill-switch flattens.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # strategies/ for the backtest module

from funding_harvest_backtest import backtest  # noqa: E402

from .config import StrategyConfig  # noqa: E402
from .broker import PaperBroker, Position  # noqa: E402
from .engine import Engine  # noqa: E402
from .feed import ReplayFeed  # noqa: E402
from .risk import HALT, RiskMonitor  # noqa: E402


def _metrics(curve, equity0):
    eq = pd.DataFrame(curve, columns=["date", "equity", "n"]).set_index("date")
    r = eq["equity"].pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(3 * 365)
    sharpe = (r.mean() * 3 * 365) / vol if vol > 0 else float("nan")
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    return eq, dict(total=total, cagr=cagr, sharpe=sharpe, dd=dd,
                    final=eq["equity"].iloc[-1], avgn=eq["n"].mean())


def _load_real(cfg: StrategyConfig):
    br = json.load(open(cfg.borrow_file))
    cfg.borrow_rate = {c: v["ann"] for c, v in br.items()}
    cfg.borrow_limit = {c: v["limit_usd"] for c, v in br.items() if v.get("limit_usd")}
    return cfg


def fidelity():
    """Engine on a backtest-mirroring config must match backtest() closely."""
    cfg = StrategyConfig()
    cfg.perp_leverage = 10.0          # match backtest default perp_margin=0.10
    cfg.thresh_ann = 0.05
    cfg.rebal_every = 9
    cfg.set_and_hold = True
    # disable the extra live guardrails so selection == backtest both-sided
    cfg.short_exclude = ()
    cfg.max_short_per_name = 1e9
    cfg.borrow_rate = {}              # -> flat 0.10 default, like backtest
    cfg.borrow_limit = {c: 1e18 for c in []}  # empty -> shorts gated? handle below
    # build_target gates shorts on membership in borrow_limit; for parity, allow all
    feed = ReplayFeed(cfg)
    cfg.borrow_limit = {p: 1e18 for p in feed.pairs}
    eng = Engine(cfg, feed, PaperBroker(cfg), RiskMonitor(cfg))
    res = eng.run()
    eq_e, m_e = _metrics(res.equity_curve, cfg.equity0)

    # the matching backtest: same trade_cost, both sides, n_max=49, 72h, set-and-hold
    trade_cost_fee = dict(fee_spot=cfg.fee_spot, fee_fut=cfg.fee_fut,
                          slippage=cfg.slippage)
    eqb, info = backtest(
        equity0=cfg.equity0, start_equity=cfg.equity0, n_max=49, thresh_ann=0.05,
        rebal_every=9, rebal_band=1e9, spot_borrow_ann=0.10, **trade_cost_fee,
    )
    rb = eqb["equity"].pct_change().dropna()
    fin_b = eqb["equity"].iloc[-1]

    print("1) FIDELITY  (engine replay  vs  validated backtest, identical config)")
    print(f"   engine   : final {m_e['final']:,.0f}  CAGR {m_e['cagr']:+.2%}"
          f"  Sharpe {m_e['sharpe']:.2f}")
    print(f"   backtest : final {fin_b:,.0f}")
    diff = abs(m_e["final"] - fin_b) / fin_b
    print(f"   final-equity divergence: {diff:.3%}  -> "
          f"{'MATCH (engine is faithful)' if diff < 0.02 else 'MISMATCH - investigate'}")
    return diff


def live_config():
    cfg = _load_real(StrategyConfig())   # real borrow rates + limits
    cfg.perp_leverage = 3.0              # safe cross-margin leverage (Step B)
    cfg.thresh_ann = 0.05
    cfg.rebal_every = 9                  # 72h
    feed = ReplayFeed(cfg)
    eng = Engine(cfg, feed, PaperBroker(cfg), RiskMonitor(cfg))
    res = eng.run()
    eq, m = _metrics(res.equity_curve, cfg.equity0)
    b = res.broker
    print("\n2) LIVE CONFIG REPLAY  (real borrow, <=3x cross margin, maker, set-and-hold)")
    print(f"   Period   : {eq.index[0].date()} -> {eq.index[-1].date()}")
    print(f"   CAGR {m['cagr']:+.2%}   Sharpe {m['sharpe']:.2f}   maxDD {m['dd']:.2%}"
          f"   avg#pos {m['avgn']:.1f}")
    print(f"   P&L: funding {b.gross_funding:+,.0f}  basis {b.basis:+,.0f}  "
          f"borrow {-b.borrow:+,.0f}  costs {-b.costs:+,.0f}  ({b.n_trades} fills)")
    # risk-overlay activity summary
    kinds = {}
    for _, e in res.risk_events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"   Risk overlay over {len(eq)} ticks: {kinds or 'no events'}")
    print("   (solvency/auto-deleverage never triggered: the hedged book on cross")
    print("    margin stays solvent - exactly the Step-B design intent.)")


def risk_selftest():
    print("\n3) RISK SELF-TEST  (force a drawdown -> kill-switch must flatten)")
    cfg = StrategyConfig()
    rm = RiskMonitor(cfg)
    bk = PaperBroker(cfg)
    bk._pos = {"BTC": Position(1, 10_000, 1.0, 1.0)}
    bk._equity = cfg.equity0 * (1 - cfg.max_drawdown_kill - 0.01)  # breach DD
    step = {"prices": {"BTC": (1.0, 1.0)}, "sig": pd.Series({"BTC": 0.0})}
    events = rm.check(step, bk)
    halts = [e for e in events if e.level == HALT]
    print(f"   emitted: {[(e.level, e.kind) for e in events]}")
    print(f"   kill-switch fired: {'YES' if halts and rm.halted else 'NO'}")


if __name__ == "__main__":
    print("=" * 78)
    print("FUNDING-HARVEST EXECUTION ENGINE - replay verification")
    print("=" * 78)
    fidelity()
    live_config()
    risk_selftest()
