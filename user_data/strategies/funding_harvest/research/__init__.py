"""Offline research & robustness for strategy #3 (run with the mltrade python):

  python -m funding_harvest.research.backtest     # base backtest + sensitivities
  python -m funding_harvest.research.analysis     # walk-forward / capacity
  python -m funding_harvest.research.realdata     # real Binance borrow/fee constraints
  python -m funding_harvest.research.exec_costs   # honest execution-cost model
  python -m funding_harvest.research.liqtest      # perp-leg liquidation stress
"""
