"""Strategy #2 live engine - same ONE ENGINE, TWO FEEDS design as funding_harvest.

The Engine (decision + accounting) runs in three modes by swapping feed + broker:
  * replay     -> ReplayFeed + PaperBroker        (validate vs research backtest)
  * paper-live -> LiveFeed   + DryRunLiveBroker    (live data, simulated fills, no keys)
  * live       -> LiveFeed   + LiveBroker          (real single-leg perp account, gated)

Single-leg (directional perp) book, so execution is simpler than #3's two legs:
each rebalance trades each name toward its signed target notional, postOnly maker.

Modules: config, signals, broker, feed, risk, engine, state, testnet,
run_replay, run_paper.
"""
