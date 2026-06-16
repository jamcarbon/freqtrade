"""
STEP C - low-cost execution model.

Two changes that attack the binding constraint (turnover/cost):

  1. HONEST re-weight cost. The earlier sim resized held positions to their new
     carry/vol target every rebalance for FREE. The engine now charges
     trade_cost * |delta_notional| on every re-size - so daily re-weighting of a
     49-name book is no longer free. This makes the baseline strictly more honest
     (and a bit worse).

  2. LEVERS to claw it back:
       - rebal_band : only re-size a held pair when its target drifts past a band
                      (a dead-band that suppresses tiny daily churn).
       - rebal_every: trade less often (cadence).
       - maker fills: lower per-leg fee (spot 0.075% BNB / perp 0.018%) + 2bps slip
                      vs taker (spot 0.10% / perp 0.05% + 5bps slip).

All runs: real VIP0 borrow rates+limits, both-sided, n_max=49, $100k.
"""
import json
import os

from funding_harvest.research.backtest import DATA, backtest, metrics  # noqa: E402

BR = json.load(open(os.path.join(DATA, "borrow_rates.json")))
borrow_map = {c: v["ann"] for c, v in BR.items()}
borrow_limit = {c: v["limit_usd"] for c, v in BR.items() if v.get("limit_usd")}

TAKER = dict(fee_spot=0.0010, fee_fut=0.0005, slippage=0.0005)
MAKER = dict(fee_spot=0.00075, fee_fut=0.00018, slippage=0.0002)
REAL = dict(n_max=49, thresh_ann=0.0, borrow_map=borrow_map,
            borrow_limit=borrow_limit)


def line(label, **kw):
    eq, info = backtest(equity0=1e5, start_equity=1e5, **kw)
    m = metrics(eq, info, equity0=1e5, show=False)
    print(f"  {label:40s} CAGR={m['cagr']:+6.1%} Sharpe={m['sharpe']:5.2f}"
          f" maxDD={m['dd']:6.1%} | fund={info['gross_funding']:+8.0f}"
          f" cost={-info['costs']:+8.0f}")


def run():
    print("=" * 90)
    print("STEP C - LOW-COST EXECUTION  (real constraints, both-sided, n_max=49, $100k)")
    print("=" * 90)

    print("\n--- 1) The honest re-weight cost (band=0, daily): taker vs maker ---")
    line("taker, daily, no band", **REAL, **TAKER)
    line("maker, daily, no band", **REAL, **MAKER)

    print("\n--- 2) Rebalance dead-band (maker fills, daily cadence) ---")
    for band in (0.0, 0.10, 0.25, 0.50, 1.00):
        line(f"maker, band={band:.0%}", **REAL, **MAKER, rebal_band=band)

    print("\n--- 3) Rebalance cadence (maker fills, band=25%) ---")
    for re in (3, 9, 21, 42):
        line(f"maker, rebal every {re*8}h", **REAL, **MAKER, rebal_band=0.25,
             rebal_every=re)

    print("\n--- 4) Best-practice execution vs naive, side by side ---")
    line("NAIVE  (taker, daily, no band)", **REAL, **TAKER)
    line("TUNED  (maker, band=25%, every 72h)", **REAL, **MAKER, rebal_band=0.25,
         rebal_every=9)
    print("\n  (The funding edge is fixed; execution discipline is what you keep of it.)")


if __name__ == "__main__":
    run()
