"""Replay verification for the #2 engine: prove it reproduces the research backtest
(one engine, two feeds), then report the risk-overlay activity and self-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trend_momentum.research.backtest import backtest

from .broker import PaperBroker
from .config import TrendConfig
from .engine import Engine
from .feed import ReplayFeed
from .risk import HALT, RiskMonitor


def _metrics(curve, equity0):
    eq = pd.DataFrame(curve, columns=["date", "equity", "n", "gross", "net"]).set_index("date")
    r = eq["equity"].pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    v = r.std() * np.sqrt(3 * 365)
    sharpe = (r.mean() * 3 * 365) / v if v > 0 else float("nan")
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    return eq, dict(total=total, cagr=cagr, sharpe=sharpe, dd=dd,
                    final=eq["equity"].iloc[-1], gross=eq["gross"].mean(),
                    net=eq["net"].mean(), n=eq["n"].mean())


def fidelity():
    cfg = TrendConfig()
    eng = Engine(cfg, ReplayFeed(cfg), PaperBroker(cfg), RiskMonitor(cfg))
    res = eng.run()
    eq_e, m = _metrics(res.equity_curve, cfg.equity0)
    eqb, _ = backtest(equity0=cfg.equity0)
    fin_b = eqb["equity"].iloc[-1]
    print("1) FIDELITY  (engine replay vs research backtest, identical config)")
    print(f"   engine   : final {m['final']:,.0f}  CAGR {m['cagr']:+.2%}"
          f"  Sharpe {m['sharpe']:.2f}  net {m['net']:+.0%}  gross {m['gross']:.1f}x")
    print(f"   backtest : final {fin_b:,.0f}")
    diff = abs(m["final"] - fin_b) / fin_b
    print(f"   divergence: {diff:.3%}  -> "
          f"{'MATCH (engine is faithful)' if diff < 0.02 else 'MISMATCH - investigate'}")
    print(f"\n2) RISK OVERLAY over {len(eq_e)} ticks: "
          f"{eng.risk.counts or 'no events'}")
    return diff


def risk_selftest():
    print("\n3) RISK SELF-TEST (force a drawdown -> kill-switch flattens)")
    cfg = TrendConfig()
    rm, bk = RiskMonitor(cfg), PaperBroker(cfg)
    bk._pos = {"BTC": 5000.0, "ETH": -5000.0}
    bk._equity = cfg.equity0 * (1 - cfg.max_drawdown_kill - 0.01)
    ev = rm.check({"prices": {}, "sig": pd.Series(dtype=float)}, bk)
    print(f"   emitted {[(e.level, e.kind) for e in ev]}; halted={rm.halted}")


if __name__ == "__main__":
    print("=" * 74)
    print("STRATEGY #2 (MOMENTUM) - replay verification")
    print("=" * 74)
    fidelity()
    risk_selftest()
