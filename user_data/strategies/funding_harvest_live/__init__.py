"""
Funding-Rate Harvesting (#3) - production-grade delta-neutral execution engine.

Design principle: ONE ENGINE, TWO FEEDS.
The same decision + accounting code (engine.Engine) runs in two modes:
  * replay/paper  -> ReplayFeed + PaperBroker   (validate against the backtest)
  * live          -> LiveFeed   + LiveBroker     (ccxt, key-gated, forward-test first)
so the live path can never silently diverge from what was validated offline.

Validated config (see funding_harvest_exec.py): both-sided, n_max~49, carry/vol
sizing applied AT ENTRY (set-and-hold), maker-preferred fills, ~72h rebalance,
entry threshold ~5-8%, on CROSS/portfolio margin at <=3x with auto-deleverage.

Modules:
  config   - StrategyConfig (the locked-in parameters + risk limits)
  signals  - trailing funding signal + target-book builder (point-in-time, no leak)
  broker   - Broker ABC; PaperBroker (faithful sim) + LiveBroker (ccxt skeleton)
  feed     - DataFeed ABC; ReplayFeed (feathers) + LiveFeed (ccxt skeleton)
  risk     - RiskMonitor (solvency, auto-deleverage, delta band, kill-switches)
  engine   - orchestration loop + two-leg executor
"""
