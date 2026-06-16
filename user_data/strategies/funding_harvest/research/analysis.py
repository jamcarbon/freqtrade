"""
Extended robustness analysis for strategy #3 (funding_harvest_backtest.py):

  1. Walk-forward / out-of-sample   - is the edge real out-of-sample, or tuned in?
  2. Realistic frictions            - rebalance slippage + per-alt borrow rates &
                                       availability tiered by exchange liquidity (OI).
  3. Capacity curve                 - cap each pair's book at a fraction of its real
                                       open interest, then sweep AUM to find where
                                       the strategy stops scaling.

All of this leans on the same leak-free vectorised engine in
funding_harvest_backtest.backtest(). Run with the mltrade python.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from funding_harvest.research.backtest import (  # noqa: E402
    DATA,
    FUND_PER_YEAR,
    backtest,
    load_grid,
    metrics,
)

OI_FILE = os.path.join(DATA, "oi_snapshot.json")


# ----------------------------------------------------------------------------- #
# Liquidity model: tier per-alt borrow rate & availability from current OI.
# (Binance only serves ~30d of historical OI publicly, so we use a present-day
#  snapshot as the structural liquidity ranking - a documented approximation.)
# ----------------------------------------------------------------------------- #
def liquidity_tiers():
    oi = json.load(open(OI_FILE))
    oi = {k: v for k, v in oi.items() if v}
    borrow, no_borrow = {}, set()
    for p, usd in oi.items():
        if usd > 200e6:
            borrow[p] = 0.05          # deep, cheap to borrow
        elif usd > 50e6:
            borrow[p] = 0.12
        elif usd > 10e6:
            borrow[p] = 0.25          # thin, expensive
        else:
            no_borrow.add(p)          # too illiquid to reliably spot-short
    return oi, borrow, no_borrow


# ----------------------------------------------------------------------------- #
# 1. WALK-FORWARD / OUT-OF-SAMPLE
# ----------------------------------------------------------------------------- #
def _sharpe(eq):
    r = eq["equity"].pct_change().dropna()
    v = r.std() * np.sqrt(FUND_PER_YEAR)
    return (r.mean() * FUND_PER_YEAR) / v if v > 0 else np.nan


def fixed_split(**common):
    """Tune the few knobs on 2022-02..2024-06, then evaluate untouched on
    2024-06..2026-06. Reports in-sample-selected config and its OOS result."""
    split = "2024-06-01"
    grid_thresh = [0.0, 0.03, 0.05, 0.08, 0.12]
    grid_nmax = [8, 12, 20, 49]
    best, best_sh = None, -1e9
    for th in grid_thresh:
        for nm in grid_nmax:
            eq, _ = backtest(start="2022-02-01", thresh_ann=th, n_max=nm, **common)
            eq_is = eq[eq.index < pd.Timestamp(split, tz="UTC")]
            sh = _sharpe(eq_is)
            if sh > best_sh:
                best_sh, best = sh, (th, nm)
    th, nm = best
    # OOS: run only on the test window, fresh capital, with the IS-chosen config
    eq_oos, info_oos = backtest(start=split, thresh_ann=th, n_max=nm, **common)
    eq_is_full, _ = backtest(start="2022-02-01", thresh_ann=th, n_max=nm, **common)
    eq_is_full = eq_is_full[eq_is_full.index < pd.Timestamp(split, tz="UTC")]
    return best, _sharpe(eq_is_full), metrics(eq_oos, info_oos, show=False)


def rolling_wf(train_yrs=1.5, test_yrs=0.5, **common):
    """Anchored-then-rolling walk-forward: repeatedly tune threshold/n_max on a
    trailing train window, trade the next test window OOS, stitch the OOS equity."""
    perp, _, _, _ = load_grid()
    t0 = pd.Timestamp("2022-02-01", tz="UTC")
    tend = perp.index.max()
    grid_thresh = [0.0, 0.05, 0.08, 0.12]
    grid_nmax = [8, 12, 20, 49]
    seg_start = t0 + pd.Timedelta(days=int(365 * train_yrs))
    equity = 10_000.0
    stitched = []
    segments = []
    while seg_start < tend:
        tr0 = seg_start - pd.Timedelta(days=int(365 * train_yrs))
        te1 = seg_start + pd.Timedelta(days=int(365 * test_yrs))
        # tune on [tr0, seg_start)
        best, best_sh = None, -1e9
        for th in grid_thresh:
            for nm in grid_nmax:
                eq, _ = backtest(start=str(tr0.date()), thresh_ann=th, n_max=nm, **common)
                eq = eq[eq.index < seg_start]
                sh = _sharpe(eq)
                if sh > best_sh:
                    best_sh, best = sh, (th, nm)
        th, nm = best
        # trade [seg_start, te1) OOS with stitched equity
        eq, _ = backtest(start=str(seg_start.date()), thresh_ann=th, n_max=nm,
                         start_equity=equity, **common)
        eq = eq[(eq.index >= seg_start) & (eq.index < te1)]
        if len(eq):
            stitched.append(eq)
            equity = eq["equity"].iloc[-1]
            segments.append((seg_start.date(), te1.date(), th, nm,
                             eq["equity"].iloc[-1]))
        seg_start = te1
    full = pd.concat(stitched)
    full = full[~full.index.duplicated(keep="first")]
    return full, segments


# ----------------------------------------------------------------------------- #
# 2 + 3. REALISTIC FRICTIONS  &  CAPACITY CURVE
# ----------------------------------------------------------------------------- #
def capacity_caps(oi, oi_frac, aum):
    """Per-pair USD notional cap = oi_frac * current OI. Returns dict only where
    the cap actually binds relative to an equal slice of AUM (keeps it sparse)."""
    return {p: oi_frac * usd for p, usd in oi.items()}


def run():
    oi, borrow, no_borrow = liquidity_tiers()
    real = dict(slippage=0.0005, borrow_map=borrow, no_borrow=no_borrow)

    print("=" * 80)
    print("STRATEGY #3 - EXTENDED ROBUSTNESS  (walk-forward / frictions / capacity)")
    print("=" * 80)

    print("\nBorrow/availability tiers from current OI:")
    print(f"  cheap 5%  : {sorted(p for p,r in borrow.items() if r==0.05)}")
    print(f"  mid  12%  : {sorted(p for p,r in borrow.items() if r==0.12)}")
    print(f"  thin 25%  : {sorted(p for p,r in borrow.items() if r==0.25)}")
    print(f"  NO-BORROW : {sorted(no_borrow)}  (negative side disabled)")

    # ---- 1. WALK-FORWARD ----------------------------------------------------
    print("\n" + "-" * 80)
    print("1) OUT-OF-SAMPLE VALIDATION")
    print("-" * 80)
    (th, nm), is_sh, oos = fixed_split(**real)
    print(f"  Fixed split (train 2022-02->2024-06, test 2024-06->2026-06), realistic frictions:")
    print(f"    IS-selected config : thresh={th*100:.0f}%  n_max={nm}   (IS Sharpe {is_sh:.2f})")
    print(f"    OOS result         : ret={oos['total']:+.1%}  CAGR={oos['cagr']:+.1%}"
          f"  Sharpe={oos['sharpe']:.2f}  maxDD={oos['dd']:.1%}")

    full, segs = rolling_wf(**real)
    print(f"\n  Rolling walk-forward (1.5y train / 0.5y test, re-tuned each step,"
          f" frictions on):")
    print(f"    {'test window':25s} {'cfg':>12s} {'end equity':>12s}")
    for s0, s1, th, nm, eqend in segs:
        print(f"    {str(s0)+' -> '+str(s1):25s} th={th*100:>3.0f}% n={nm:<3d} {eqend:>12,.0f}")
    m = metrics(full, {"gross_funding": 0, "hedge_residual": 0, "borrow": 0,
                       "costs": 0, "rebalances": 0}, show=False)
    print(f"    STITCHED OOS       : ret={m['total']:+.1%}  CAGR={m['cagr']:+.1%}"
          f"  Sharpe={m['sharpe']:.2f}  maxDD={m['dd']:.1%}")

    # ---- 2. REALISTIC FRICTIONS (full sample, AUM=10k) ----------------------
    print("\n" + "-" * 80)
    print("2) FRICTION LADDER  (full sample, AUM 10k, stacking realistic costs)")
    print("-" * 80)
    def fr(label, **kw):
        eq, info = backtest(**kw)
        m = metrics(eq, info, show=False)
        g, c, b = info["gross_funding"], info["costs"], info["borrow"]
        print(f"  {label:34s} ret={m['total']:+7.1%} CAGR={m['cagr']:+6.1%}"
              f" Sharpe={m['sharpe']:5.2f} maxDD={m['dd']:6.1%}"
              f"  | fund={g:+7.0f} cost={-c:+7.0f} borrow={-b:+6.0f}")
    print("  [concentrated book: n_max=12]")
    fr("base (5bps fee, flat 10% borrow)")
    fr("+ 5bps rebalance slippage", slippage=0.0005)
    fr("+ per-alt borrow tiers", borrow_map=borrow, no_borrow=no_borrow)
    fr("+ both (full realistic)", slippage=0.0005, borrow_map=borrow,
       no_borrow=no_borrow)
    fr("+ both, 10bps slippage (stress)", slippage=0.0010, borrow_map=borrow,
       no_borrow=no_borrow)
    print("  [diversified, low-turnover book: n_max=49, thresh=0% - the WF pick]")
    fr("base", n_max=49, thresh_ann=0.0)
    fr("+ 5bps rebalance slippage", n_max=49, thresh_ann=0.0, slippage=0.0005)
    fr("+ both (full realistic)", n_max=49, thresh_ann=0.0, slippage=0.0005,
       borrow_map=borrow, no_borrow=no_borrow)
    fr("+ both, 10bps slippage (stress)", n_max=49, thresh_ann=0.0,
       slippage=0.0010, borrow_map=borrow, no_borrow=no_borrow)

    # ---- 3. CAPACITY CURVE --------------------------------------------------
    print("\n" + "-" * 80)
    print("3) CAPACITY CURVE  (viable low-turnover config n_max=49/thresh=0% + full")
    print("   realistic frictions; cap each pair's book at 1% of its current OI;")
    print("   sweep AUM. CAGR is on TOTAL equity incl. idle/un-deployable cash.)")
    print("-" * 80)
    oi_frac = 0.01
    print(f"  {'AUM (USD)':>12s}  {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s}"
          f" {'grossYield':>10s} {'$funding/yr':>12s}")
    for aum in (1e5, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9):
        caps = capacity_caps(oi, oi_frac, aum)
        eq, info = backtest(equity0=aum, start_equity=aum, notional_cap=caps,
                            n_max=49, thresh_ann=0.0, slippage=0.0005,
                            borrow_map=borrow, no_borrow=no_borrow)
        m = metrics(eq, info, equity0=aum, show=False)
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        # gross funding yield on TOTAL equity/yr: falls as OI caps idle the capital
        gross_yield = info["gross_funding"] / yrs / aum
        print(f"  {aum:>12,.0f}  {m['cagr']:+7.1%} {m['sharpe']:7.2f} {m['dd']:7.1%}"
              f" {gross_yield*100:9.1f}% {info['gross_funding']/yrs:>12,.0f}")
    print("\n  (As AUM rises the 1%-of-OI cap on the thin alts forces capital idle:")
    print("   $ funding/yr plateaus and CAGR on total equity decays - the capacity wall.)")


if __name__ == "__main__":
    run()
