"""
Strategy #2 - Cross-Sectional + Time-Series Momentum (market-neutral, perp).

A directional-factor book that is the natural diversifier to the #3 carry book:
where carry bleeds in violent trends/squeezes, a momentum book is long the movers
and short the laggards and tends to *make* money there (crisis-alpha). We run it
MARKET-NEUTRAL (net ~ 0) so it isolates the momentum factor and adds little BTC-beta.

Edge (PROVEN-IN-LITERATURE, replicated in crypto): returns auto-correlate -
recent winners keep winning, losers keep losing - from behavioural under-reaction,
flows and reflexive leverage. We harvest the *cross-section* of that, vol-targeted.

Why a dedicated vectorised sim (same reasoning as #3)
-----------------------------------------------------
A faithful momentum book needs portfolio-level construction (cross-sectional rank,
vol-targeting, net/gross caps, funding-aware perp P&L) that a per-instrument
freqtrade backtest does not express. We simulate on the same 8h Binance perp panels.

Leak-free
---------
  * The weight at rebalance time t uses only closes/returns with timestamp <= t
    (a trailing-return signal and trailing realised vol).
  * A book formed at t earns the perp returns and pays the funding of (t, t+1].
    Pass signal_lag=1 to delay acting one more step; results barely move.

Accounting (USDT, signed positions: long > 0, short < 0)
--------------------------------------------------------
  price P&L/step   = sum_i  pos_i * perp_ret_i(this step)
  funding/step     = sum_i  -pos_i * funding_i(this step)   (a LONG perp pays funding>0)
  cost/rebalance   = (fee + slippage) * sum_i |pos_i^new - pos_i^old|   (turnover)
"""
import glob
import os

import numpy as np
import pandas as pd


def _find_repo(start: str) -> str:
    p = os.path.abspath(start)
    while not os.path.isdir(os.path.join(p, "user_data", "data")):
        nxt = os.path.dirname(p)
        if nxt == p:
            raise RuntimeError("could not locate repo root (user_data/data)")
        p = nxt
    return p


_REPO = _find_repo(__file__)
DATA = os.path.join(_REPO, "user_data", "data", "binance")
FUND_PER_YEAR = 3 * 365  # 8h funding/return periods per year


def load_panels(start="2022-02-01"):
    """Aligned 8h panels: perp close, perp $volume, and 8h funding, across the
    same 49-pair Binance universe used by #3 (ragged-listed pairs carry NaN)."""
    close, dvol, fund = {}, {}, {}
    for f in sorted(glob.glob(f"{DATA}/futures/*-8h-futures.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        df = pd.read_feather(f).set_index("date")
        close[name] = df["close"]
        dvol[name] = df["close"] * df["volume"]          # USD volume proxy
    for f in sorted(glob.glob(f"{DATA}/futures/*-1h-funding_rate.feather")):
        name = os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        s = pd.read_feather(f).set_index("date")["open"]
        fund[name] = s[s.index.hour % 8 == 0]            # keep 8h settlement prints
    close = pd.DataFrame(close).sort_index()
    dvol = pd.DataFrame(dvol).reindex_like(close)
    fund = pd.DataFrame(fund).reindex(close.index).reindex(columns=close.columns)
    mask = close.index >= pd.Timestamp(start, tz="UTC")
    return close[mask], dvol[mask], fund[mask].fillna(0.0)


def backtest(
    start="2022-02-01",
    ts_lookback=(42, 63, 126),  # blended trailing-return horizons (8h steps) - a
                                # multi-horizon blend is far less overfit than one
    vol_k=45,              # trailing realised-vol lookback (45 ~ 15 days)
    market_neutral=True,   # demean weights each rebalance -> net ~ 0
    target_vol=0.20,       # annualised portfolio vol target
    max_weight=0.15,       # per-name gross weight cap
    gross_cap=3.0,         # max gross leverage after vol-targeting
    rebal_every=21,        # 8h steps between rebalances (21 -> weekly; cost-aware)
    fee=0.0005,            # taker fee per side
    slippage=0.0005,       # slippage per side on traded notional
    adv_floor=2_000_000.0, # require trailing $volume/8h above this to trade a name
    adv_k=63,              # lookback for the liquidity filter
    signal_lag=0,
    equity0=10_000.0,
    start_equity=None,
):
    close, dvol, fund = load_panels(start)
    ret = close.pct_change()
    times = close.index

    vol = ret.rolling(vol_k).std() * np.sqrt(FUND_PER_YEAR)
    lbs = (ts_lookback,) if np.isscalar(ts_lookback) else tuple(ts_lookback)
    # blend horizons: cross-sectionally z-score each horizon's risk-adjusted
    # momentum (so each horizon gets equal say), then average.
    parts = []
    for lb in lbs:
        s = (close / close.shift(lb) - 1.0) / vol
        z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
        parts.append(z)
    sig = sum(parts) / len(parts)
    if signal_lag:
        sig = sig.shift(signal_lag)
    liquid = dvol.rolling(adv_k).mean() > adv_floor   # tradeable-universe mask

    equity = equity0 if start_equity is None else start_equity
    held: dict[str, float] = {}                       # pair -> signed notional
    eq_curve = []
    price_pnl = fund_pnl = cost_c = 0.0
    n_rebal = 0
    warm = max(max(lbs), vol_k, adv_k)

    for i, t in enumerate(times):
        # --- accrue this step's P&L on the book held coming into t -----------
        if held:
            pr = ret.loc[t]
            fr = fund.loc[t]
            step_p = sum(n * pr.get(p, 0.0) for p, n in held.items()
                         if np.isfinite(pr.get(p, np.nan)))
            step_f = sum(-n * fr.get(p, 0.0) for p, n in held.items()
                         if np.isfinite(fr.get(p, np.nan)))
            equity += step_p + step_f
            price_pnl += step_p
            fund_pnl += step_f

        # --- rebalance on cadence -------------------------------------------
        if i % rebal_every == 0 and i >= warm + signal_lag and equity > 0:
            s = sig.loc[t].copy()
            v = vol.loc[t]
            ok = liquid.loc[t]
            s = s[ok.reindex(s.index).fillna(False) & np.isfinite(s) & (v > 0)]
            if len(s) >= 6:                            # need a cross-section
                if market_neutral:
                    s = s - s.mean()                   # net ~ 0 (long/short)
                w = s / s.abs().sum()                  # gross 1 baseline
                w = w.clip(-max_weight, max_weight)
                if w.abs().sum() > 0:
                    w = w / w.abs().sum()
                # diagonal vol-target, clamp gross
                port_vol = np.sqrt(((w * v.reindex(w.index)) ** 2).sum())
                scale = min(target_vol / port_vol, gross_cap) if port_vol > 0 else 0.0
                target = {p: scale * w[p] * equity for p in w.index}
                # trade toward target; charge turnover
                pairs = set(target) | set(held)
                for p in pairs:
                    new = target.get(p, 0.0)
                    old = held.get(p, 0.0)
                    d = abs(new - old)
                    if d > 0:
                        c = (fee + slippage) * d
                        equity -= c
                        cost_c += c
                    if abs(new) > 1e-9:
                        held[p] = new
                    elif p in held:
                        del held[p]
                n_rebal += 1

        eq_curve.append((t, equity, len(held),
                         sum(abs(n) for n in held.values()) / max(equity, 1e-9),
                         sum(held.values()) / max(equity, 1e-9)))

    eq = pd.DataFrame(eq_curve,
                      columns=["date", "equity", "n_pos", "gross", "net"]
                      ).set_index("date")
    return eq, dict(price_pnl=price_pnl, funding=fund_pnl, costs=cost_c,
                    rebalances=n_rebal, pairs=list(close.columns))


def metrics(eq, info=None, equity0=10_000.0, show=True):
    r = eq["equity"].pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    v = r.std() * np.sqrt(FUND_PER_YEAR)
    sharpe = (r.mean() * FUND_PER_YEAR) / v if v > 0 else float("nan")
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    out = dict(total=total, cagr=cagr, sharpe=sharpe, dd=dd, vol=v,
               gross=eq["gross"].mean(), net=eq["net"].mean(), n=eq["n_pos"].mean())
    if show:
        print(f"  total {total:+.1%}  CAGR {cagr:+.1%}  Sharpe {sharpe:.2f}"
              f"  maxDD {dd:.1%}  vol {v:.0%}  gross {out['gross']:.1f}x"
              f"  net {out['net']:+.0%}  avg#pos {out['n']:.0f}")
        if info:
            print(f"  price P&L {info['price_pnl']:+,.0f}  funding "
                  f"{info['funding']:+,.0f}  costs {-info['costs']:+,.0f}"
                  f"  rebalances {info['rebalances']}")
    return out


if __name__ == "__main__":
    print("Strategy #2 - market-neutral momentum, base config:")
    eq, info = backtest()
    metrics(eq, info)
