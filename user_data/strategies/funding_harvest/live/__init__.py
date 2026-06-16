"""
Funding-Rate Harvesting (#3) - production-grade delta-neutral execution engine.

Design principle: ONE ENGINE, TWO FEEDS.
The same decision + accounting code (engine.Engine) runs in three modes, swapping
only feed + broker:
  * replay        -> ReplayFeed + PaperBroker        (validate vs research backtest)
  * paper-live    -> LiveFeed   + DryRunLiveBroker    (live data, simulated fills, no keys)
  * live          -> LiveFeed   + LiveBroker          (real two-leg account, gated)
so the live path can never silently diverge from what was validated offline.

Validated config (see ../research/exec_costs.py): both-sided, n_max~49, carry/vol
sizing applied AT ENTRY (set-and-hold), maker-preferred fills, ~72h rebalance,
entry threshold ~5-8%, on CROSS/portfolio margin at <=3x with auto-deleverage.

Modules:
  config   - StrategyConfig (locked-in parameters + hard risk limits)
  signals  - trailing funding signal + set-and-hold target-book builder (no leak)
  broker   - Broker ABC; PaperBroker, DryRunLiveBroker, LiveBroker (real, gated)
  feed     - DataFeed ABC; ReplayFeed (feathers) + LiveFeed (public ccxt)
  risk     - RiskMonitor (solvency, auto-deleverage, delta band, kill-switches)
  engine   - orchestration loop + two-leg executor
  state    - persist/restore the paper book between scheduled forward-test runs
  testnet  - Binance-sandbox wiring for the LiveBroker order path (guarded)
  run_replay / run_paper - verification + paper-live / forward-test entry points
"""
