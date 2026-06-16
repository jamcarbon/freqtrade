"""
Robustness for strategy #2 (market-neutral momentum). Mirrors the #3 discipline:

  1. leak test        - base vs signal_lag=1 (should barely move)
  2. cost/cadence     - turnover is the binding constraint, as in #3
  3. lookback grid    - is the edge a knife-edge fit or broad?
  4. walk-forward OOS  - fixed split + rolling, re-tuned each window

Run: PYTHONPATH=user_data/strategies python -m trend_momentum.research.analysis
"""
import numpy as np
import pandas as pd

from trend_momentum.research.backtest import FUND_PER_YEAR, backtest, load_panels, metrics


def _sharpe(eq):
    r = eq["equity"].pct_change().dropna()
    v = r.std() * np.sqrt(FUND_PER_YEAR)
    return (r.mean() * FUND_PER_YEAR) / v if v > 0 else np.nan


def run():
    print("=" * 84)
    print("STRATEGY #2 - MOMENTUM ROBUSTNESS  (market-neutral L/S, 2022-2026, $10k)")
    print("=" * 84)

    print("\n1) LEAK TEST (act on same-bar signal vs delay one full step)")
    for lag in (0, 1):
        eq, info = backtest(signal_lag=lag)
        print(f"  signal_lag={lag}: ", end="")
        metrics(eq, info)

    print("\n2) COST / CADENCE LADDER (realistic taker 5bps + 5bps slip)")
    print("   rebalance cadence vs net result - turnover is the cost driver")
    for re in (3, 9, 21, 42):
        eq, info = backtest(rebal_every=re)
        print(f"  every {re*8:>3}h: ", end="")
        metrics(eq, info)

    print("\n3) LOOKBACK SENSITIVITY (trailing-return horizon; vol_k fixed)")
    for lb in (21, 42, 63, 126, 189):
        eq, info = backtest(ts_lookback=lb, rebal_every=9)
        print(f"  ts_lookback {lb:>3} ({lb*8//24:>2}d): ", end="")
        metrics(eq, info)

    print("\n4) WALK-FORWARD / OUT-OF-SAMPLE")
    split = "2024-06-01"
    grid = [(lb, re) for lb in (42, 63, 126) for re in (9, 21)]
    best, best_sh = None, -1e9
    for lb, re in grid:
        eq, _ = backtest(ts_lookback=lb, rebal_every=re)
        sh = _sharpe(eq[eq.index < pd.Timestamp(split, tz="UTC")])
        if sh > best_sh:
            best_sh, best = sh, (lb, re)
    lb, re = best
    eq_oos, info_oos = backtest(start=split, ts_lookback=lb, rebal_every=re)
    print(f"  fixed split: train<{split} picks ts_lookback={lb}, every {re*8}h"
          f"  (IS Sharpe {best_sh:.2f})")
    print(f"    OOS {split}->: ", end="")
    metrics(eq_oos, info_oos)

    # rolling walk-forward: re-tune each 6-month window, stitch OOS equity
    close, _, _ = load_panels()
    t0 = pd.Timestamp("2022-02-01", tz="UTC")
    tend = close.index.max()
    seg = t0 + pd.Timedelta(days=int(365 * 1.5))
    equity, stitched, segs = 10_000.0, [], []
    while seg < tend:
        tr0 = seg - pd.Timedelta(days=int(365 * 1.5))
        te1 = seg + pd.Timedelta(days=182)
        b, bsh = None, -1e9
        for lb, re in grid:
            e, _ = backtest(start=str(tr0.date()), ts_lookback=lb, rebal_every=re)
            sh = _sharpe(e[e.index < seg])
            if sh > bsh:
                bsh, b = sh, (lb, re)
        lb, re = b
        e, _ = backtest(start=str(seg.date()), ts_lookback=lb, rebal_every=re,
                        start_equity=equity)
        e = e[(e.index >= seg) & (e.index < te1)]
        if len(e):
            stitched.append(e)
            equity = e["equity"].iloc[-1]
            segs.append((seg.date(), te1.date(), lb, re, equity))
        seg = te1
    full = pd.concat(stitched)
    full = full[~full.index.duplicated(keep="first")]
    print("\n  rolling walk-forward (1.5y train / 0.5y test, re-tuned each step):")
    for s0, s1, lb, re, eqend in segs:
        print(f"    {str(s0)} -> {str(s1)}  lb={lb:<3} every {re*8}h   {eqend:>10,.0f}")
    print("    STITCHED OOS: ", end="")
    metrics(full, None)


if __name__ == "__main__":
    run()
