"""DataFeed abstraction. The feed is the ONLY other thing (besides the broker)
that differs between replay and live. It yields one 8h step at a time, each
carrying exactly the observable information the engine is allowed to act on.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# import the validated data loader from the sibling backtest module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funding_harvest_backtest import FUND_PER_YEAR, load_grid  # noqa: E402

from .config import StrategyConfig


class DataFeed(ABC):
    @abstractmethod
    def __iter__(self): ...


class ReplayFeed(DataFeed):
    """Replays history from the feather panels, computing the SAME trailing,
    point-in-time signal the backtest uses. Each yielded step is leak-free: the
    accrual is this tick's settled cashflow on the prior book; the signal uses
    only funding observable up to and including this tick."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        perp, spot, fund, pairs = load_grid()
        mask = perp.index >= pd.Timestamp(cfg.start, tz="UTC")
        self.perp, self.spot, self.fund = perp[mask], spot[mask], fund[mask]
        self.pairs = pairs
        self.spot_ret = self.spot.pct_change().fillna(0.0)
        self.perp_ret = self.perp.pct_change().fillna(0.0)
        self.basis_ret = self.spot_ret - self.perp_ret
        self.trail = self.fund.rolling(cfg.trail_k).mean() * FUND_PER_YEAR
        self.basis_vol = self.basis_ret.rolling(cfg.vol_k).std() * np.sqrt(FUND_PER_YEAR)
        self.times = self.perp.index
        self._warm = max(cfg.trail_k, cfg.vol_k)

    def __iter__(self):
        for i, t in enumerate(self.times):
            accrual = {}
            for p in self.pairs:
                fr = self.fund.at[t, p]
                br = self.basis_ret.at[t, p]
                if not np.isfinite(fr):
                    continue
                accrual[p] = {
                    "funding": fr,
                    "basis_ret": br if np.isfinite(br) else 0.0,
                    "borrow_step": self.cfg.borrow_rate.get(p, 0.10) / FUND_PER_YEAR,
                }
            prices = {p: (self.spot.at[t, p], self.perp.at[t, p]) for p in self.pairs}
            yield dict(
                t=t,
                i=i,
                accrual=accrual,
                sig=self.trail.loc[t],
                vol=self.basis_vol.loc[t],
                prices=prices,
                is_rebalance=(i % self.cfg.rebal_every == 0 and i >= self._warm),
                live=False,
            )


def _universe_from_disk(cfg: StrategyConfig) -> list[str]:
    import glob
    names = {
        os.path.basename(f).split("-")[0].replace("_USDT_USDT", "")
        for f in glob.glob(f"{cfg.data_dir}/futures/*-8h-futures.feather")
    }
    return sorted(names)


class LiveFeed(DataFeed):
    """Real-time feed from Binance PUBLIC endpoints (no keys needed): recent
    funding history + 8h spot/perp candles -> the SAME trailing, point-in-time
    signal the backtest/replay use. snapshot() returns one step identical in shape
    to ReplayFeed's, so the very same Engine consumes it unchanged.

    Each tick is stamped with receive-time so the stale-data kill-switch can fire
    if the live data goes quiet.
    """

    def __init__(self, cfg: StrategyConfig, exchange=None, pairs: list[str] | None = None):
        import time

        import ccxt
        self.cfg = cfg
        self.time = time
        # public handle is enough for mark/funding/ohlcv; keys only needed to trade
        self.ex = exchange or ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.spot_ex = ccxt.binance({"enableRateLimit": True})  # spot market data
        self.pairs = pairs or _universe_from_disk(cfg)

    def _pair_signal(self, c: str):
        """Trailing annualised funding, basis vol and current prices for one pair,
        from public history. Returns (sig_ann, vol_ann, spot_px, perp_px, last_fr,
        last_basis_ret) or None on failure / insufficient data."""
        psym, ssym = f"{c}/USDT:USDT", f"{c}/USDT"
        try:
            fh = self.ex.fetch_funding_rate_history(psym, limit=self.cfg.trail_k + 3)
            n8 = self.cfg.vol_k + 3
            po = self.ex.fetch_ohlcv(psym, "8h", limit=n8)
            so = self.spot_ex.fetch_ohlcv(ssym, "8h", limit=n8)
        except Exception:
            return None
        if len(fh) < self.cfg.trail_k or len(po) < self.cfg.vol_k + 1 or len(so) < self.cfg.vol_k + 1:
            return None
        fr = np.array([x["fundingRate"] for x in fh], dtype=float)
        trail = fr[-self.cfg.trail_k:].mean() * FUND_PER_YEAR
        pc = pd.Series([r[4] for r in po], dtype=float)
        sc = pd.Series([r[4] for r in so], dtype=float)
        n = min(len(pc), len(sc))
        basis_ret = sc[-n:].pct_change() - pc[-n:].pct_change()
        vol = basis_ret.tail(self.cfg.vol_k).std() * np.sqrt(FUND_PER_YEAR)
        return (trail, float(vol), float(sc.iloc[-1]), float(pc.iloc[-1]),
                float(fr[-1]), float(basis_ret.iloc[-1]) if np.isfinite(basis_ret.iloc[-1]) else 0.0)

    def snapshot(self, is_rebalance: bool = True) -> dict:
        """Pull one live step across the universe."""
        sig, vol, prices, accrual = {}, {}, {}, {}
        t_recv = self.time.time()
        for c in self.pairs:
            r = self._pair_signal(c)
            if r is None:
                continue
            trail, v, spx, ppx, last_fr, last_br = r
            sig[c] = trail
            vol[c] = v
            prices[c] = (spx, ppx)
            accrual[c] = {
                "funding": last_fr,
                "basis_ret": last_br,
                "borrow_step": self.cfg.borrow_rate.get(c, 0.10) / FUND_PER_YEAR,
            }
        return dict(
            t=pd.Timestamp.utcnow(),
            i=0,
            accrual=accrual,
            sig=pd.Series(sig),
            vol=pd.Series(vol),
            prices=prices,
            is_rebalance=is_rebalance,
            live=True,
            recv_time=t_recv,
            n_pairs=len(sig),
        )

    def __iter__(self):
        # In a real forward test the runner schedules snapshot() on the 8h
        # settlement cadence; iterating here would just yield one snapshot.
        yield self.snapshot()
