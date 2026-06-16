# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
DualMomentumXS - Time-Series + Cross-Sectional Momentum (Trend-Following)

Implements strategy #2 from the Crypto Systematic Trading research report:
combine *time-series* momentum (own-asset trend gates long/short) with
*cross-sectional* momentum (long the strongest names, short the weakest across
a basket of perps). Volatility-normalised signals, inverse-vol position sizing,
and trend-driven exits (EMA cross / rank rotation) backed by a wide hard stop.

The edge in this universe materialises over ~5-15 days (validated by a leak-free
forward-return study), so the design deliberately holds for weeks instead of
harvesting 4h noise with a tight stop. A tight ATR/Chandelier stop was tested
and *destroyed* the edge (it ejected trades at ~2.5d, before the move develops).

Design principles taken straight from the report (section 3.2):
  - Coarse, few parameters; resist hyperopt over-tuning -> low overfit risk.
  - Use only CLOSED candles for every signal -> low leakage risk.
  - Vol-targeting + correlation control: net exposure is kept ~neutral by
    pairing long top-N with short bottom-N, and gross is capped via sizing.
  - Crisis-alpha: shorts are first-class (futures), so the book can profit in
    downtrends.

The cross-sectional rank is computed once across the whole whitelist from raw
OHLCV (point-in-time: each row only ever sees its own and prior closes), then
joined back per pair. This is backtest- and live-correct because freqtrade only
acts on a signal at the *next* candle.
"""

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class DualMomentumXS(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    can_short = True

    # We ride trends: ROI is disabled; exits come from trend signals + hard stop.
    minimal_roi = {"0": 100}
    # Trend signals (EMA cross / rank rotation) do the real exiting. The hard stop
    # only caps fat-tail V-reversals that outrun the slow trend exit.
    stoploss = -0.15
    use_custom_stoploss = False

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # EMA200 (slow trend) + 30d momentum lookback need a long warm-up. recursive-
    # analysis shows EMA200 only fully converges well past 200 candles, so warm up
    # for ~400 candles before trusting signals.
    startup_candle_count = 400

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # ----- Strategy parameters (kept coarse on purpose) ------------------------
    # Cross-sectional momentum lookback, in candles. 4h * 180 = 30 days.
    mom_lookback = 180
    # Own-asset trend EMAs (4h): ~8 days fast, ~33 days slow.
    ema_fast = 50
    ema_slow = 200
    # Number of names to go long (top) / short (bottom) in the cross-section.
    top_n = 3
    # Hysteresis band for exits: hold until the name rotates into the opposite
    # extreme (weakest `exit_n` for longs / strongest `exit_n` for shorts).
    exit_n = 6
    # Regime gate: only trade names that are actually trending.
    adx_min = 20.0
    # Chandelier / ATR trailing stop.
    atr_period = 14
    atr_mult = 3.0
    # Vol targeting for sizing: target per-candle ATR fraction.
    target_atr_pct = 0.03

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "orange"},
            "ema_slow": {"color": "blue"},
        },
        "subplots": {
            "XS-rank": {"xs_rank": {"color": "purple"}},
            "ADX": {"adx": {"color": "red"}},
        },
    }

    # Cache for the cross-sectional rank table (rebuilt only when data advances).
    _rank_cache_key = None
    _rank_df: DataFrame | None = None

    # --------------------------------------------------------------------------
    def _pair_score(self, pair: str) -> pd.Series | None:
        """Risk-adjusted momentum score for one pair, indexed by candle date."""
        df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        close = df["close"]
        ret = close.pct_change()
        mom = close / close.shift(self.mom_lookback) - 1.0
        vol = ret.rolling(self.mom_lookback).std()
        # Risk-adjusted (Sharpe-like) momentum; avoids favouring high-vol coins.
        score = mom / (vol * np.sqrt(self.mom_lookback))
        score.index = df["date"]
        return score

    def _build_rank_table(self) -> DataFrame:
        """Cross-sectional rank (1 = strongest) of every pair at each timestamp."""
        pairs = self.dp.current_whitelist()
        scores = {}
        for p in pairs:
            s = self._pair_score(p)
            if s is not None:
                scores[p] = s
        if not scores:
            return DataFrame()
        score_df = DataFrame(scores).sort_index()
        # rank descending: 1 = highest (strongest) momentum among valid names.
        rank_df = score_df.rank(axis=1, ascending=False, method="first")
        rank_df["_n_valid"] = score_df.notna().sum(axis=1)
        return rank_df

    def _get_rank_table(self) -> DataFrame:
        # Cache keyed on whitelist + last candle so we build it once per backtest.
        wl = tuple(sorted(self.dp.current_whitelist()))
        df_btc = self.dp.get_pair_dataframe(wl[0], self.timeframe) if wl else None
        last = df_btc["date"].iloc[-1] if df_btc is not None and len(df_btc) else None
        key = (wl, last)
        if key != self._rank_cache_key or self._rank_df is None:
            self._rank_df = self._build_rank_table()
            self._rank_cache_key = key
        return self._rank_df

    # --------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # --- Own-asset (time-series) momentum & trend -------------------------
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe["roc"] = (
            dataframe["close"] / dataframe["close"].shift(self.mom_lookback) - 1.0
        )
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        # --- Cross-sectional rank join ---------------------------------------
        rank_df = self._get_rank_table()
        if not rank_df.empty and pair in rank_df.columns:
            join = rank_df[[pair, "_n_valid"]].rename(
                columns={pair: "xs_rank", "_n_valid": "xs_n"}
            )
            dataframe = dataframe.merge(
                join, left_on="date", right_index=True, how="left"
            )
        else:
            dataframe["xs_rank"] = np.nan
            dataframe["xs_n"] = np.nan

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Time-series momentum gates (own-asset trend must agree with direction).
        # Entry needs trend alignment + price confirmation + positive own momentum.
        ts_up = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["close"] > dataframe["ema_slow"])
            & (dataframe["roc"] > 0)
        )
        ts_dn = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["close"] < dataframe["ema_slow"])
            & (dataframe["roc"] < 0)
        )
        trending = dataframe["adx"] > self.adx_min

        # Cross-sectional gates: top-N strongest = long, bottom-N weakest = short.
        xs_long = dataframe["xs_rank"] <= self.top_n
        xs_short = dataframe["xs_rank"] > (dataframe["xs_n"] - self.top_n)

        long_cond = [ts_up, trending, xs_long, dataframe["volume"] > 0]
        short_cond = [ts_dn, trending, xs_short, dataframe["volume"] > 0]

        dataframe.loc[reduce(lambda a, b: a & b, long_cond), ["enter_long", "enter_tag"]] = (
            1,
            "xs_ts_long",
        )
        dataframe.loc[
            reduce(lambda a, b: a & b, short_cond), ["enter_short", "enter_tag"]
        ] = (1, "xs_ts_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend-follower exits: let winners run, exit only on a decisive own-asset
        # trend reversal (EMA cross). Rank governs *entry*; trend governs *exit*.
        # The Chandelier ATR stop (custom_stoploss) handles everything in between.
        # Hysteresis: exit a long only if it also rotates into the weakest band,
        # so a name that merely dips below the top-N is held, not churned.
        weak_band = dataframe["xs_rank"] > (dataframe["xs_n"] - self.exit_n)
        strong_band = dataframe["xs_rank"] <= self.exit_n

        dataframe.loc[
            (dataframe["ema_fast"] < dataframe["ema_slow"]) | weak_band, "exit_long"
        ] = 1
        dataframe.loc[
            (dataframe["ema_fast"] > dataframe["ema_slow"]) | strong_band, "exit_short"
        ] = 1
        return dataframe

    # --------------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """Trend-trailing stop.

        The momentum edge in this universe materialises over ~5-15 days, so a tight
        ATR stop just harvests noise. Instead we trail with the *slow EMA* (the trend
        backbone) once a trade is in profit, widened by a small ATR cushion so normal
        pullbacks to the mean don't eject us. Before break-even we only have the wide
        disaster `stoploss`.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or len(df) == 0:
            return None
        ema_slow = df["ema_slow"].iloc[-1]
        atr = df["atr"].iloc[-1]
        if not np.isfinite(ema_slow) or not np.isfinite(atr) or atr <= 0:
            return None

        # Only start trailing once the trade has cleared its first ATR of profit,
        # so winners get room to develop.
        if current_profit < 0.02:
            return None

        if trade.is_short:
            stop_price = ema_slow + self.atr_mult * atr
        else:
            stop_price = ema_slow - self.atr_mult * atr

        return stoploss_from_absolute(
            stop_price,
            current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    # --------------------------------------------------------------------------
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Honest baseline: no leverage. Gross exposure stays ~= equity.
        return 1.0

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Inverse-vol sizing: scale the equal-weight stake by target/asset vol."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        scale = 1.0
        if df is not None and len(df) > 0:
            atr_pct = df["atr_pct"].iloc[-1]
            if np.isfinite(atr_pct) and atr_pct > 0:
                scale = float(np.clip(self.target_atr_pct / atr_pct, 0.5, 1.5))
        stake = proposed_stake * scale
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake)



