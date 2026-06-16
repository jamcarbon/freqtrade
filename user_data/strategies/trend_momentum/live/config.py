"""Locked-in parameters + hard risk limits for strategy #2 (momentum).

Validated in research/: market-neutral, multi-horizon momentum blend, weekly
rebalance, vol-targeted. Risk limits are guardrails enforced by RiskMonitor, not
tuning knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from trend_momentum.research.backtest import DATA as _DATA

FUND_PER_YEAR = 3 * 365


@dataclass
class TrendConfig:
    # --- data / universe ----------------------------------------------------
    data_dir: str = _DATA
    start: str = "2022-02-01"

    # --- signal (validated multi-horizon blend) -----------------------------
    ts_lookbacks: tuple = (42, 63, 126)   # 8h steps (~14/21/42 days)
    vol_k: int = 45                        # realised-vol lookback
    market_neutral: bool = True            # demean -> net ~ 0
    adv_floor: float = 2_000_000.0         # liquidity floor ($vol/8h)
    adv_k: int = 63

    # --- sizing / cadence ---------------------------------------------------
    target_vol: float = 0.20               # annualised portfolio vol target
    max_weight: float = 0.15               # per-name gross weight cap
    gross_cap: float = 3.0                 # max gross leverage after vol-target
    rebal_every: int = 21                  # 8h steps (21 -> weekly)

    # --- execution costs (maker-preferred perp) -----------------------------
    fee: float = 0.0005                    # per-side; research used taker 5bps
    slippage: float = 0.0005
    maker_timeout_s: int = 20
    poll_interval_s: int = 2
    max_reprice: int = 3

    # --- HARD RISK LIMITS (RiskMonitor) -------------------------------------
    max_drawdown_kill: float = 0.25        # momentum draws deeper than carry
    net_band: float = 0.10                 # |net| must stay within this frac of equity
    gross_kill: float = 4.0                # halt if gross blows past this
    stale_data_secs: int = 600

    # --- account ------------------------------------------------------------
    equity0: float = 10_000.0
    fund_per_year: int = FUND_PER_YEAR
