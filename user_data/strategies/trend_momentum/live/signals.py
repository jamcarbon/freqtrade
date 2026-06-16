"""Target-book construction for #2 (point-in-time, leak-free).

build_target() takes the currently observable blended momentum signal + realised
vol and returns the desired SIGNED notional per name (long > 0, short < 0). It
reproduces the research backtest's construction exactly, so replay matches:
demean (market-neutral) -> per-name cap -> gross-normalise -> diagonal vol-target
-> clamp gross. The engine diffs desired-vs-held and trades the difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TrendConfig


def build_target(sig: pd.Series, vol: pd.Series, equity: float,
                 cfg: TrendConfig) -> dict[str, float]:
    s = sig[np.isfinite(sig)]
    v = vol.reindex(s.index)
    s = s[(v > 0) & np.isfinite(v)]
    if len(s) < 6 or equity <= 0:        # need a cross-section to be neutral
        return {}
    if cfg.market_neutral:
        s = s - s.mean()                 # net ~ 0
    w = s / s.abs().sum()                # gross 1 baseline
    w = w.clip(-cfg.max_weight, cfg.max_weight)
    g = w.abs().sum()
    if g <= 0:
        return {}
    w = w / g
    port_vol = float(np.sqrt(((w * v.reindex(w.index)) ** 2).sum()))
    scale = min(cfg.target_vol / port_vol, cfg.gross_cap) if port_vol > 0 else 0.0
    return {p: scale * w[p] * equity for p in w.index
            if abs(scale * w[p] * equity) > 1e-9}
