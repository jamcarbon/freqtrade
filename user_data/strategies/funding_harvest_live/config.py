"""Locked-in strategy parameters and hard risk limits for funding harvesting #3.

Every number here is either (a) validated in the backtest/robustness work or
(b) a hard safety limit enforced OUTSIDE the signal logic. Risk limits are not
tuning knobs - they are the guardrails the strategy is never allowed to cross.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FUND_PER_YEAR = 3 * 365  # Binance funding settles every 8h


@dataclass
class StrategyConfig:
    # --- data / universe -----------------------------------------------------
    data_dir: str = "user_data/data/binance"
    borrow_file: str = "user_data/data/binance/borrow_rates.json"
    oi_file: str = "user_data/data/binance/oi_snapshot.json"
    start: str = "2022-02-01"

    # --- signal (point-in-time, validated) -----------------------------------
    trail_k: int = 9            # trailing funding prints (3 days) for the signal
    vol_k: int = 30             # trailing basis-vol lookback (prints)
    thresh_ann: float = 0.05    # enter when |annualised funding| clears this
    exit_ann: float = 0.0       # drop a held pair when |funding| falls under this
    both_sides: bool = True     # harvest negative funding too (long-perp/short-spot)
    n_max: int = 49             # max simultaneous neutral pairs
    max_weight: float = 0.20    # per-pair portfolio weight cap

    # --- sizing / cadence (Step C: set-and-hold + slow cadence) --------------
    rebal_every: int = 9        # rebalance cadence in 8h steps (9 -> 72h)
    set_and_hold: bool = True   # size at entry; never daily-reweight a held pair

    # --- execution costs (maker-preferred, VIP0 + BNB discount) --------------
    fee_spot: float = 0.00075   # spot maker leg
    fee_fut: float = 0.00018    # perp maker leg
    slippage: float = 0.0002    # per-leg slack vs mid for a passive fill

    # --- margin / leverage (Step B: low leverage on CROSS margin) ------------
    margin_mode: str = "cross"  # 'cross'/'portfolio' so the spot leg collateralises
    perp_leverage: float = 3.0  # <=3x effective on the perp leg
    maint_margin: float = 0.005 # maintenance-margin rate (tier-dependent)

    @property
    def perp_margin(self) -> float:
        """Initial margin fraction tied up on the perp leg = 1 / leverage."""
        return 1.0 / self.perp_leverage

    # --- HARD RISK LIMITS (enforced by RiskMonitor, not by the signal) -------
    # Auto-deleverage the whole book if combined margin distance falls under this.
    auto_dlv_margin_distance: float = 0.08
    # Net delta must stay within this fraction of equity, else hedge/halt.
    delta_band: float = 0.02
    # Squeeze-prone names (Step B worst 3-day up-runs) - never short the spot leg.
    short_exclude: tuple = ("GMT", "APE", "ENJ", "FIL", "SNX", "XLM", "SAND")
    # Cap any single short-spot position at this fraction of equity.
    max_short_per_name: float = 0.08
    # Portfolio drawdown that trips the kill-switch (de-size / halt new entries).
    max_drawdown_kill: float = 0.08
    # An 8h funding magnitude this large is anomalous -> flag for review
    # (0.75%/8h ~= 82%/yr; normal Binance funding is ~0.01-0.05%/8h).
    funding_spike_flag: float = 0.0075
    # Stale-data kill (live only): no fresh tick within this many seconds.
    stale_data_secs: int = 600

    # --- account ------------------------------------------------------------
    equity0: float = 100_000.0
    fund_per_year: int = FUND_PER_YEAR

    # Per-pair USD borrow limits / OI caps are loaded at runtime from the json
    # files above; kept out of the dataclass so they stay data, not config.
    borrow_limit: dict = field(default_factory=dict)
    borrow_rate: dict = field(default_factory=dict)
    oi_cap: dict = field(default_factory=dict)
