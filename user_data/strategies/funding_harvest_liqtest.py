"""
STEP B - liquidation / tail stress test for the perp leg.

The vectorised sim shows ~0 hedge residual because spot and perp close-to-close
cancel. That conceals the real danger of a delta-neutral funding harvest: the perp
leg is held on ISOLATED margin and can be liquidated by an intra-candle spike even
while the (cash/cross) spot leg is fine. This script measures, from real perp
high/low, how often each leverage would have been liquidated, 2022-2026.

Adverse direction:
  * positive-carry side = SHORT perp -> hurt by UP spikes  (high/open - 1)
  * negative-carry side = LONG  perp -> hurt by DOWN spikes (1 - low/open)
We take the worse of the two per candle (the book runs both sides), so this is the
hostile-case excursion the margin buffer must survive.

Liquidation threshold per isolated leverage L (approx):
    liq_move ~= 1/L - maintenance_margin_rate   (mmr ~0.5% for these tiers)
e.g. 10x ~= 9.5%, 5x ~= 19.5%, 3x ~= 32.8%, 2x ~= 49.5%.
"""
import glob
import os

import numpy as np
import pandas as pd

DATA = "user_data/data/binance/futures"
MMR = 0.005


def load_perp_ohlc():
    out = {}
    for f in sorted(glob.glob(f"{DATA}/*-8h-futures.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        df = pd.read_feather(f).set_index("date")[["open", "high", "low", "close"]]
        out[name] = df
    return out


def run(start="2022-02-01"):
    perp = load_perp_ohlc()
    print("=" * 88)
    print("STEP B - PERP-LEG LIQUIDATION STRESS TEST  (real high/low, 2022-2026)")
    print("=" * 88)

    # per-candle hostile excursion across all pairs
    up, dn = [], []
    worst_up = {}   # pair -> (date, move)
    for name, df in perp.items():
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        u = (df["high"] / df["open"] - 1.0)         # short-perp adverse
        d = (1.0 - df["low"] / df["open"])          # long-perp adverse
        up.append(u); dn.append(d)
        iu = u.idxmax()
        worst_up[name] = (iu, u.max())
    up = pd.concat(up); dn = pd.concat(dn)
    hostile = pd.concat([up, dn], axis=0)           # both-side hostile excursions
    n = len(up)                                     # candle-pairs (per side)

    print(f"\nSingle-8h-candle hostile excursion distribution "
          f"({n:,} candle-observations/side):")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  {q*100:6.1f}th pct : up {up.quantile(q)*100:6.2f}%   "
              f"down {dn.quantile(q)*100:6.2f}%")

    print("\nIsolated-margin liquidation events (a single candle pierces the buffer):")
    print(f"  {'lever':>6s} {'liq move':>9s} {'short-leg hits':>15s} {'long-leg hits':>14s}"
          f" {'~ per year':>11s}")
    yrs = (up.index.max() - up.index.min()).days / 365.25 if hasattr(up.index, "max") else 4.4
    for L in (10, 5, 3, 2):
        thr = 1.0 / L - MMR
        su = int((up > thr).sum())
        sd = int((dn > thr).sum())
        print(f"  {L:>5d}x {thr*100:>8.1f}% {su:>15,d} {sd:>14,d}"
              f" {(su+sd)/4.37:>10.1f}")

    print("\nSustained-squeeze risk: worst cumulative UP-run over a 3-day (9-candle)")
    print("hold window per pair (short-perp leg), top 8 most dangerous names:")
    runs = []
    for name, df in perp.items():
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        # max forward 9-candle up-move from any open to a later high
        roll_high = df["high"].rolling(9).max().shift(-8)
        run_up = (roll_high / df["open"] - 1.0)
        runs.append((name, run_up.max()))
    for name, mx in sorted(runs, key=lambda x: -x[1])[:8]:
        d, _ = worst_up[name]
        print(f"  {name:6s} worst 3-day up-run {mx*100:6.1f}%   "
              f"(worst single candle {worst_up[name][1]*100:.1f}% on {d.date()})")

    print("\n" + "-" * 88)
    print("READOUT")
    print("-" * 88)
    thr10 = (up > (0.1 - MMR)).sum() + (dn > (0.1 - MMR)).sum()
    thr5 = (up > (0.2 - MMR)).sum() + (dn > (0.2 - MMR)).sum()
    thr3 = (up > (1/3 - MMR)).sum() + (dn > (1/3 - MMR)).sum()
    print(f"  10x isolated  : {thr10} liquidation-trigger candles  -> UNSAFE")
    print(f"   5x isolated  : {thr5} liquidation-trigger candles")
    print(f"   3x isolated  : {thr3} liquidation-trigger candles")
    print("  Note: these are SINGLE-candle pierces; sustained squeezes (above) are")
    print("  worse. A delta-neutral book should run LOW perp leverage (3x or less isolated)")
    print("  with a fat buffer, OR use cross/portfolio margin so the spot leg's gain")
    print("  offsets the perp loss before liquidation. Auto-deleverage on margin")
    print("  distance is mandatory. The funding edge does NOT need leverage - the")
    print("  perp leg only needs enough margin to stay solvent through the tail.")


if __name__ == "__main__":
    run()
