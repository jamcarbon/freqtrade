"""
Strategy #3 - Funding-Rate Harvesting (delta-neutral, diversified yield portfolio).

This is the generalised, productised version of strategy #1 (funding_carry_backtest.py).
Where #1 harvested only the *positive*-carry side on a handful of majors, #3:

  * harvests BOTH sides of funding
        - funding > 0  ->  SHORT perp + LONG  spot   (longs pay us)
        - funding < 0  ->  LONG  perp + SHORT spot   (shorts pay us; spot leg borrowed)
  * runs across a wide universe (49 same-venue Binance pairs) and treats the book
    as a *yield portfolio*: rank by net-of-cost annualised funding, size by
    carry / basis-volatility with hard per-pair caps.

Why this is a vectorised sim and not a freqtrade-native backtest
----------------------------------------------------------------
Freqtrade trades ONE instrument per position; it cannot hold a spot leg as a
delta hedge against a perp short inside a single trade. A freqtrade-native run
would therefore measure an *unhedged* directional P&L - which is not this
strategy. The honest way to backtest a delta-neutral funding harvest is a
dedicated vectorised simulation on funding + spot + perp panels. That is this
file (same approach and accounting conventions as #1, extended to both sides,
many pairs, ragged listing dates, carry/vol sizing, and spot-borrow costs).

Point-in-time discipline (no lookahead)
---------------------------------------
  * The side/size decision at the rebalance time t uses only funding prints with
    timestamp <= t (already settled / observable) via a trailing mean.
  * A position opened at t earns the funding prints in (t, t+hold] - the future
    cashflows you actually receive by holding. No overlap, no leak. Pass
    `signal_lag=1` to additionally delay acting on the signal by one full step;
    results barely move, which is the expected signature of a leak-free design.

Accounting (transparent, conservative; all P&L in USDT)
-------------------------------------------------------
  side s_p   = +1 if we are short-perp/long-spot (positive carry)
               -1 if we are long-perp/short-spot (negative carry)
  Funding/step  = sum_i  s_i * notional_i * funding_i(this print)
                  (short perp receives +funding; long perp receives -funding)
  Hedge resid/step = sum_i s_i * notional_i * (spot_ret_i - perp_ret_i)
                  (delta-neutral: the two legs nearly cancel; residual is basis drift)
  Borrow cost/step = sum_{i: s_i=-1} notional_i * borrow_ann / FUND_PER_YEAR
                  (only the negative-carry side borrows the coin to short spot)
  Costs        = 2 legs at entry + 2 legs at exit, taker fee each; a side flip is
                 a close + reopen. Hysteresis keeps turnover (and cost) low.
  Capital/pair = spot notional + perp initial margin (10x -> 10%); sum of spot
                 notionals == deployed equity.
"""

import glob
import os

import numpy as np
import pandas as pd

def _find_repo(start: str) -> str:
    """Walk up until we find the repo root (the dir that contains user_data/data),
    so this works regardless of how deep the module is nested or the cwd."""
    p = os.path.abspath(start)
    while not os.path.isdir(os.path.join(p, "user_data", "data")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("could not locate repo root (user_data/data)")
        p = parent
    return p


_REPO = _find_repo(__file__)
DATA = os.path.join(_REPO, "user_data", "data", "binance")
FUND_PER_YEAR = 3 * 365  # funding settles every 8h on Binance


def load_grid():
    """Build aligned 8h panels (perp close, spot close, 8h funding) on a shared,
    ragged time grid - pairs that listed later simply carry NaN until they exist."""
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
    # OUTER align on the union 8h grid so late-listing pairs are kept (ragged),
    # not truncated to the shortest common history.
    perp = pd.DataFrame({p: perp[p] for p in pairs}).sort_index()
    spot = pd.DataFrame({p: spot[p] for p in pairs}).sort_index()
    fund = pd.DataFrame({p: fund[p] for p in pairs}).sort_index()
    grid = perp.index.union(spot.index).union(fund.index)
    grid = grid[grid.hour % 8 == 0]  # clean 8h cadence
    perp = perp.reindex(grid)
    spot = spot.reindex(grid)
    fund = fund.reindex(grid)
    return perp, spot, fund, pairs


def backtest(
    start="2022-02-01",
    fee=0.0005,           # symmetric taker fee per leg (fallback for the two below)
    fee_spot=None,        # spot-leg taker fee (Binance VIP0 ~0.0010); None -> fee
    fee_fut=None,         # perp-leg taker fee (Binance VIP0 ~0.0005); None -> fee
    slippage=0.0,         # extra cost per leg per side on traded notional (bps as frac)
    borrow_limit=None,    # {pair: max USD short-spot notional} (VIP borrow cap)
    perp_margin=0.10,     # 10x on the perp leg
    spot_borrow_ann=0.10, # default annualised cost to borrow the coin for short-spot
    borrow_map=None,      # {pair: annual borrow rate}; overrides scalar per pair
    no_borrow=None,       # set of pairs that CANNOT be spot-shorted (neg side disabled)
    trail_k=9,            # funding prints (3 days) for the trailing-funding signal
    thresh_ann=0.05,      # enter when |trailing annualised funding| clears this
    exit_ann=0.0,         # drop a held pair when |funding| falls to/under this
    n_max=12,             # max simultaneous neutral pairs (diversified book)
    max_weight=0.20,      # hard per-pair cap on portfolio weight
    rebal_every=3,        # rebalance cadence in funding steps (3 -> once a day)
    rebal_band=0.0,       # only re-size a held pair when its target weight drifts
                          # >this frac from current (0 -> resize every rebalance)
    weighting="carryvol", # "carryvol" (carry / basis-vol) or "equal"
    vol_k=30,             # lookback (prints) for trailing basis volatility
    both_sides=True,      # harvest negative funding too; False -> positive-only (#1-like)
    notional_cap=None,    # {pair: max USD spot notional}; caps deployment (capacity)
    signal_lag=0,         # extra steps to delay acting on the signal (leak stress test)
    equity0=10_000.0,
    start_equity=None,    # override equity0 for walk-forward stitching
):
    perp, spot, fund, pairs = load_grid()
    mask = perp.index >= pd.Timestamp(start, tz="UTC")
    perp, spot, fund = perp[mask], spot[mask], fund[mask]
    times = perp.index

    spot_ret = spot.pct_change().fillna(0.0)
    perp_ret = perp.pct_change().fillna(0.0)
    basis_ret = spot_ret - perp_ret  # P&L driver of a delta-neutral leg
    # Trailing annualised funding (signed), observable at each timestamp.
    trail = fund.rolling(trail_k).mean() * FUND_PER_YEAR
    if signal_lag:
        trail = trail.shift(signal_lag)
    # Trailing annualised volatility of the hedged (basis) spread -> the real risk.
    basis_vol = basis_ret.rolling(vol_k).std() * np.sqrt(FUND_PER_YEAR)

    borrow_map = borrow_map or {}
    no_borrow = set(no_borrow or ())
    notional_cap = notional_cap or {}
    borrow_limit = borrow_limit or {}
    spot_fee = fee if fee_spot is None else fee_spot
    fut_fee = fee if fee_fut is None else fee_fut
    # Cost to open (or to close) a pair = both legs' fee + slippage on both legs.
    trade_cost = spot_fee + fut_fee + 2 * slippage

    def borrow_step(p):
        return borrow_map.get(p, spot_borrow_ann) / FUND_PER_YEAR

    equity = equity0 if start_equity is None else start_equity
    held: dict[str, dict] = {}   # pair -> {"notional":, "side":}
    eq_curve = []
    gross_f = costs_c = resid_c = borrow_c = 0.0
    n_rebal = 0

    for i, t in enumerate(times):
        # --- accrue this step's P&L on the book held coming into t -------------
        if held:
            step_f = step_r = step_b = 0.0
            for p, pos in held.items():
                n, s = pos["notional"], pos["side"]
                fr = fund.at[t, p]
                br = basis_ret.at[t, p]
                if np.isfinite(fr):
                    step_f += s * n * fr
                if np.isfinite(br):
                    step_r += s * n * br
                if s < 0:  # short-spot leg pays borrow
                    step_b += n * borrow_step(p)
            equity += step_f + step_r - step_b
            gross_f += step_f
            resid_c += step_r
            borrow_c += step_b

        # --- rebalance on cadence ---------------------------------------------
        if i % rebal_every == 0 and i >= max(trail_k, vol_k) + signal_lag:
            sig = trail.loc[t]
            vol = basis_vol.loc[t]
            # candidate sides: positive carry (short perp), and (if enabled) negative
            side = pd.Series(0, index=sig.index)
            side[sig > thresh_ann] = 1
            if both_sides:
                side[sig < -thresh_ann] = -1
                # pairs we cannot spot-short can't harvest the negative side
                for p in no_borrow:
                    if p in side.index and side[p] < 0:
                        side[p] = 0
            cand = side[side != 0].index
            # net-of-cost annualised carry: gross |funding| minus (per-pair) borrow
            net = sig.abs().copy()
            for p in side.index[side < 0]:
                net[p] -= borrow_map.get(p, spot_borrow_ann)
            cand = [p for p in cand if np.isfinite(net.get(p, np.nan)) and net[p] > 0
                    and np.isfinite(vol.get(p, np.nan)) and vol[p] > 0]
            # rank by carry / basis-vol (Sharpe-like), strongest first
            score = {p: net[p] / vol[p] for p in cand}
            ranked = sorted(cand, key=lambda p: score[p], reverse=True)
            target = ranked[:n_max]
            # hysteresis: keep a held pair only while its funding still supports
            # the SAME side we hold and |funding| clears the exit level. (A pair
            # whose funding flipped sign is dropped here and closed below.)
            keep = [p for p in held
                    if np.isfinite(sig.get(p, np.nan))
                    and abs(sig[p]) > exit_ann
                    and np.sign(sig[p]) == held[p]["side"]]
            new_set = list(dict.fromkeys(target + keep))[:n_max]

            # a pair whose carry side flipped must be fully closed + reopened
            def cur_side(p):
                return 1 if sig.get(p, 0.0) > 0 else -1
            closing = [p for p in held if p not in new_set
                       or held[p]["side"] != cur_side(p)]
            opening = [p for p in new_set if p not in held
                       or held[p]["side"] != cur_side(p)]

            for p in closing:                      # exit costs (both legs)
                c = trade_cost * held[p]["notional"]
                equity -= c
                costs_c += c
                del held[p]

            if new_set:
                if weighting == "equal":
                    w = {p: 1.0 / len(new_set) for p in new_set}
                else:  # carry / basis-vol weights, capped, renormalised
                    raw = {p: max(score.get(p, 0.0), 1e-9) for p in new_set}
                    tot = sum(raw.values())
                    w = {p: raw[p] / tot for p in new_set}
                    w = {p: min(v, max_weight) for p, v in w.items()}
                    tot = sum(w.values())
                    w = {p: v / tot for p, v in w.items()}
                # capital = spot notional + perp margin; sum(spot notional)=equity
                budget = equity / (1 + perp_margin)
                for p in new_set:
                    notional = w[p] * budget
                    s = cur_side(p)
                    # capacity: a pair's book can't exceed its liquidity cap; the
                    # un-deployable remainder just sits idle as cash (earns nothing).
                    cap = notional_cap.get(p)
                    if cap is not None:
                        notional = min(notional, cap)
                    # the short-spot (negative-carry) leg is bounded by the borrow cap
                    if s < 0 and p in borrow_limit:
                        notional = min(notional, borrow_limit[p])
                    if p in opening:               # new/flipped: full open (both legs)
                        c = trade_cost * notional
                        equity -= c
                        costs_c += c
                        held[p] = {"notional": notional, "side": s}
                    else:                          # held-through: only re-size on drift
                        cur = held[p]["notional"]
                        if cur <= 0 or abs(notional - cur) > rebal_band * cur:
                            c = trade_cost * abs(notional - cur)  # cost on traded delta
                            equity -= c
                            costs_c += c
                            held[p]["notional"] = notional
                        # else: leave the position untouched (no trade, no cost)
            n_rebal += 1

        eq_curve.append((t, equity, len(held)))

    eq = pd.DataFrame(eq_curve, columns=["date", "equity", "n_pos"]).set_index("date")
    return eq, dict(
        gross_funding=gross_f, costs=costs_c, hedge_residual=resid_c,
        borrow=borrow_c, rebalances=n_rebal, pairs=pairs,
    )


def metrics(eq, info, equity0=10_000.0, show=True):
    r = eq["equity"].pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    ann_vol = r.std() * np.sqrt(FUND_PER_YEAR)
    sharpe = (r.mean() * FUND_PER_YEAR) / ann_vol if ann_vol > 0 else np.nan
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    if show:
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
        print(f"  Borrow cost       : {-info['borrow']:+,.0f}")
        print(f"  Transaction costs : {-info['costs']:+,.0f}")
    return dict(total=total, cagr=cagr, vol=ann_vol, sharpe=sharpe, dd=dd,
                avgpos=eq["n_pos"].mean())


def _line(label, **kw):
    e, inf = backtest(**kw)
    m = metrics(e, inf, show=False)
    print(f"  {label:28s} ret={m['total']:+7.1%}  CAGR={m['cagr']:+6.1%}"
          f"  vol={m['vol']:5.1%}  Sharpe={m['sharpe']:5.2f}"
          f"  maxDD={m['dd']:6.1%}  avg#pos={m['avgpos']:4.1f}")
    return e


if __name__ == "__main__":
    print("=" * 78)
    print("FUNDING-RATE HARVESTING #3  (delta-neutral, both-sided, diversified)")
    print("=" * 78)
    eq, info = backtest()
    print(f"Universe ({len(info['pairs'])}): {', '.join(info['pairs'])}\n")
    metrics(eq, info)
    eq.to_csv("user_data/backtest_results/funding_harvest_equity.csv")

    print("\n--- A) Both-sided vs positive-only (does the negative side add value?) ---")
    _line("both-sided (full #3)", both_sides=True)
    _line("positive-only (#1-like)", both_sides=False)

    print("\n--- B) Sizing: carry/vol vs equal-weight ---")
    _line("carry/vol weighted", weighting="carryvol")
    _line("equal weight", weighting="equal")

    print("\n--- C) Sensitivity: entry threshold (annualised |funding|) ---")
    for th in (0.0, 0.03, 0.05, 0.08, 0.12):
        _line(f"thresh={th*100:.0f}%", thresh_ann=th)

    print("\n--- D) Sensitivity: spot-borrow rate on the short-spot leg ---")
    for b in (0.0, 0.05, 0.10, 0.20, 0.40):
        _line(f"borrow={b*100:.0f}%/yr", spot_borrow_ann=b)

    print("\n--- E) Sensitivity: max simultaneous pairs (diversification) ---")
    for n in (4, 8, 12, 20, 49):
        _line(f"n_max={n}", n_max=n)

    print("\n--- F) Leakage stress test: delay acting on signal by 1 full step ---")
    _line("signal_lag=0 (base)", signal_lag=0)
    _line("signal_lag=1 (lagged)", signal_lag=1)
