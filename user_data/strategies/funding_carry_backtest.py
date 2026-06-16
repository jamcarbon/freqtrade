"""
Strategy #1 - Perp-Spot Funding/Basis Carry (delta-neutral).

Freqtrade's engine trades ONE instrument per position and cannot hold a spot
hedge against a perp short, so a freqtrade-native backtest of this strategy would
measure an *unhedged* short's price P&L - which is not the strategy. The correct,
honest way to backtest a delta-neutral cash-and-carry is a dedicated vectorised
simulation on funding + spot + perp data. That is what this file is.

Position (same-venue, positive-carry side only):
    long  X USDT of SPOT   +   short X USDT of PERP   ->   delta-neutral
You collect funding on the short perp whenever funding is positive (the
structural retail-leverage premium). Price legs nearly cancel; the residual is
the change in basis (spot-perp) over the hold.

Point-in-time discipline (no lookahead):
  * The rebalance decision at time t uses only funding prints with timestamp <= t
    (already settled / observable).
  * A position opened at t earns funding only for prints in (t, t+R] - the future
    cashflows you actually receive by holding. No overlap, no leak.

Accounting (transparent, conservative):
  * Capital per pair = spot notional + perp initial margin (perp short at 10x ->
    10% margin). Sum of spot notionals == deployed equity.
  * Funding income/step  = sum_i  perp_notional_i * funding_i(next print)
  * Hedge residual /step = sum_i  notional_i * (spot_ret_i - perp_ret_i)
  * Costs: 2 legs at entry + 2 legs at exit, taker fee each -> 4*fee*notional
    round-trip per pair. Hysteresis keeps turnover (and thus cost) low.
"""

import glob
import os

import numpy as np
import pandas as pd

DATA = "user_data/data/binance"
FUND_PER_YEAR = 3 * 365  # funding settles every 8h


def load_grid():
    """Build aligned 8h panels: perp price, spot price, funding rate."""
    perp, spot, fund = {}, {}, {}
    for f in sorted(glob.glob(f"{DATA}/futures/*-8h-futures.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        perp[name] = pd.read_feather(f).set_index("date")["close"]
    for f in sorted(glob.glob(f"{DATA}/futures/*-1h-funding_rate.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        s = pd.read_feather(f).set_index("date")["open"]
        fund[name] = s[s.index.hour % 8 == 0]  # keep the 8h settlement prints
    for f in sorted(glob.glob(f"{DATA}/*-8h.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT", "")
        spot[name] = pd.read_feather(f).set_index("date")["close"]

    pairs = sorted(set(perp) & set(spot) & set(fund))
    perp = pd.DataFrame({p: perp[p] for p in pairs}).sort_index()
    spot = pd.DataFrame({p: spot[p] for p in pairs}).sort_index()
    fund = pd.DataFrame({p: fund[p] for p in pairs}).sort_index()
    idx = perp.index.intersection(spot.index).intersection(fund.index)
    return perp.loc[idx], spot.loc[idx], fund.loc[idx], pairs


def backtest(
    start="2022-02-01",
    fee=0.0005,          # taker fee per leg per side
    perp_margin=0.10,    # 10x on the perp short leg
    trail_k=9,           # funding prints (3 days) for the trailing-funding signal
    thresh_ann=0.05,     # enter when trailing annualised funding > 5%
    exit_ann=0.0,        # drop a held pair when its funding turns non-positive
    n_max=8,             # max simultaneous carry pairs
    rebal_every=3,       # rebalance cadence in funding steps (3 -> once a day)
    equity0=10_000.0,
):
    perp, spot, fund, pairs = load_grid()
    mask = perp.index >= pd.Timestamp(start, tz="UTC")
    perp, spot, fund = perp[mask], spot[mask], fund[mask]
    times = perp.index

    spot_ret = spot.pct_change().fillna(0.0)
    perp_ret = perp.pct_change().fillna(0.0)
    # Trailing annualised funding, observable at each timestamp (no lookahead).
    trail = fund.rolling(trail_k).mean() * FUND_PER_YEAR

    equity = equity0
    held: dict[str, float] = {}       # pair -> notional currently deployed
    eq_curve, gross_f, costs_c, resid_c = [], 0.0, 0.0, 0.0
    n_rebal = 0

    for i, t in enumerate(times):
        # --- accrue this step's P&L on the book held coming into t -----------
        if held:
            step_f = sum(n * fund.at[t, p] for p, n in held.items())
            step_r = sum(
                n * (spot_ret.at[t, p] - perp_ret.at[t, p]) for p, n in held.items()
            )
            equity += step_f + step_r
            gross_f += step_f
            resid_c += step_r

        # --- rebalance on cadence -------------------------------------------
        if i % rebal_every == 0 and i >= trail_k:
            sig = trail.loc[t].dropna()
            # candidates that clear the entry threshold, strongest funding first
            ranked = sig[sig > thresh_ann].sort_values(ascending=False)
            target = list(ranked.index[:n_max])
            # hysteresis: keep an existing pair while funding stays > exit level
            keep = [p for p in held if sig.get(p, -1) > exit_ann]
            new_set = list(dict.fromkeys(keep + target))[:n_max]

            opening = [p for p in new_set if p not in held]
            closing = [p for p in held if p not in new_set]

            # pay exit costs (2 legs) on closing pairs
            for p in closing:
                c = 2 * fee * held[p]
                equity -= c
                costs_c += c
            for p in closing:
                del held[p]

            if new_set:
                # equal-weight spot notionals; capital = spot + perp margin
                per_cap = equity / (len(new_set) * (1 + perp_margin))
                notional = per_cap  # spot notional == perp notional (neutral)
                # pay entry costs (2 legs) on opening pairs, then set book
                for p in opening:
                    c = 2 * fee * notional
                    equity -= c
                    costs_c += c
                held = {p: notional for p in new_set}
            n_rebal += 1

        eq_curve.append((t, equity, len(held)))

    eq = pd.DataFrame(eq_curve, columns=["date", "equity", "n_pos"]).set_index("date")
    return eq, dict(
        gross_funding=gross_f, costs=costs_c, hedge_residual=resid_c,
        rebalances=n_rebal, pairs=pairs,
    )


def metrics(eq, info, equity0=10_000.0):
    r = eq["equity"].pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    ann_vol = r.std() * np.sqrt(FUND_PER_YEAR)
    sharpe = (r.mean() * FUND_PER_YEAR) / ann_vol if ann_vol > 0 else np.nan
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    print(f"  Period            : {eq.index[0].date()} -> {eq.index[-1].date()}  ({yrs:.2f}y)")
    print(f"  Final equity      : {eq['equity'].iloc[-1]:,.0f} USDT")
    print(f"  Total return      : {total*100:+.2f}%")
    print(f"  CAGR              : {cagr*100:+.2f}%")
    print(f"  Annualised vol    : {ann_vol*100:.2f}%")
    print(f"  Sharpe (8h)       : {sharpe:.2f}")
    print(f"  Max drawdown      : {dd*100:.2f}%")
    print(f"  Avg # positions   : {eq['n_pos'].mean():.1f}")
    print(f"  Rebalances        : {info['rebalances']}")
    print("  --- P&L decomposition (USDT) ---")
    print(f"  Gross funding     : {info['gross_funding']:+,.0f}")
    print(f"  Hedge residual    : {info['hedge_residual']:+,.0f}")
    print(f"  Transaction costs : {-info['costs']:+,.0f}")
    return dict(total=total, cagr=cagr, vol=ann_vol, sharpe=sharpe, dd=dd)


if __name__ == "__main__":
    print("=" * 70)
    print("PERP-SPOT FUNDING/BASIS CARRY  (delta-neutral, positive-carry side)")
    print("=" * 70)
    eq, info = backtest()
    print(f"Universe ({len(info['pairs'])}): {', '.join(info['pairs'])}\n")
    metrics(eq, info)
    eq.to_csv("user_data/backtest_results/funding_carry_equity.csv")

    print("\n--- Sensitivity: entry threshold (annualised funding) ---")
    for th in (0.0, 0.03, 0.05, 0.08, 0.12):
        e, inf = backtest(thresh_ann=th)
        r = e["equity"].pct_change().dropna()
        vol = r.std() * np.sqrt(FUND_PER_YEAR)
        sh = (r.mean() * FUND_PER_YEAR) / vol if vol > 0 else float("nan")
        dd = (e["equity"] / e["equity"].cummax() - 1).min()
        print(f"  thresh={th*100:4.0f}%  ret={e['equity'].iloc[-1]/10000-1:+7.1%}"
              f"  CAGR={((e['equity'].iloc[-1]/10000)**(365.25/((e.index[-1]-e.index[0]).days))-1):+6.1%}"
              f"  vol={vol:5.1%}  Sharpe={sh:4.2f}  maxDD={dd:6.1%}  avg#pos={e['n_pos'].mean():.1f}")
