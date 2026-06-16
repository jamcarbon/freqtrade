"""DataFeed for #2. ReplayFeed reproduces the research signal panels exactly so a
replay matches the backtest; LiveFeed pulls the same signal from public ccxt data.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from trend_momentum.research.backtest import FUND_PER_YEAR, load_panels

from .config import TrendConfig


def _blended_signal(close, vol, lbs):
    """Cross-sectionally z-scored, multi-horizon risk-adjusted momentum (the
    research construction). Returns a (time x pair) signal panel."""
    parts = []
    for lb in lbs:
        s = (close / close.shift(lb) - 1.0) / vol
        z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
        parts.append(z)
    return sum(parts) / len(parts)


class DataFeed(ABC):
    @abstractmethod
    def __iter__(self): ...


class ReplayFeed(DataFeed):
    def __init__(self, cfg: TrendConfig):
        self.cfg = cfg
        close, dvol, fund = load_panels(cfg.start)
        self.close, self.fund = close, fund
        self.ret = close.pct_change()
        self.vol = self.ret.rolling(cfg.vol_k).std() * np.sqrt(FUND_PER_YEAR)
        sig = _blended_signal(close, self.vol, cfg.ts_lookbacks)
        liquid = dvol.rolling(cfg.adv_k).mean() > cfg.adv_floor
        self.sig = sig.where(liquid)            # NaN out illiquid names
        self.times = close.index
        self.pairs = list(close.columns)
        self._warm = max(max(cfg.ts_lookbacks), cfg.vol_k, cfg.adv_k)

    def __iter__(self):
        for i, t in enumerate(self.times):
            r, fr = self.ret.loc[t], self.fund.loc[t]
            accrual = {p: {"ret": r[p], "funding": fr[p]}
                       for p in self.pairs
                       if np.isfinite(r.get(p, np.nan))}
            prices = {p: self.close.at[t, p] for p in self.pairs}
            yield dict(
                t=t, i=i, accrual=accrual,
                sig=self.sig.loc[t], vol=self.vol.loc[t], prices=prices,
                is_rebalance=(i % self.cfg.rebal_every == 0 and i >= self._warm),
                live=False,
            )


class LiveFeed(DataFeed):
    """Real-time momentum signal from Binance PUBLIC perp candles + funding."""

    def __init__(self, cfg: TrendConfig, exchange=None, pairs=None):
        import glob
        import time as _t

        import ccxt
        self.cfg = cfg
        self.time = _t
        self.ex = exchange or ccxt.binance({
            "enableRateLimit": True, "options": {"defaultType": "future"}})
        self.pairs = pairs or sorted(
            os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
            for f in glob.glob(f"{cfg.data_dir}/futures/*-8h-futures.feather"))

    def snapshot(self, is_rebalance: bool = True) -> dict:
        n = max(max(self.cfg.ts_lookbacks), self.cfg.vol_k, self.cfg.adv_k) + 5
        closes, dvols, frs, prices = {}, {}, {}, {}
        recv = self.time.time()
        for c in self.pairs:
            psym = f"{c}/USDT:USDT"
            try:
                o = self.ex.fetch_ohlcv(psym, "8h", limit=n)
                fh = self.ex.fetch_funding_rate_history(psym, limit=3)
            except Exception:
                continue
            if len(o) < n - 2:
                continue
            closes[c] = pd.Series([r[4] for r in o])
            dvols[c] = pd.Series([r[4] * r[5] for r in o])
            frs[c] = float(fh[-1]["fundingRate"]) if fh else 0.0
            prices[c] = float(o[-1][4])
        close = pd.DataFrame(closes)
        vol = close.pct_change().rolling(self.cfg.vol_k).std() * np.sqrt(FUND_PER_YEAR)
        sig = _blended_signal(close, vol, self.cfg.ts_lookbacks)
        liquid = pd.DataFrame(dvols).rolling(self.cfg.adv_k).mean().iloc[-1] > self.cfg.adv_floor
        sig_row = sig.iloc[-1].where(liquid)
        accrual = {c: {"ret": float(close[c].pct_change().iloc[-1]), "funding": frs[c]}
                   for c in close.columns}
        return dict(t=pd.Timestamp.utcnow(), i=0, accrual=accrual,
                    sig=sig_row, vol=vol.iloc[-1], prices=prices,
                    is_rebalance=is_rebalance, live=True, recv_time=recv,
                    n_pairs=int(sig_row.notna().sum()))

    def __iter__(self):
        yield self.snapshot()
