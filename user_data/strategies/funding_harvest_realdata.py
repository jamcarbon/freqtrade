"""
STEP A - re-run strategy #3 under REAL Binance constraints (no proxies):
  * real per-coin cross-margin borrow rates       (VIP0, public margin spec)
  * real per-coin borrow LIMITS as short-spot caps (VIP0)  -> this is the binding one
  * real asymmetric fees: spot leg 0.10% taker, perp leg 0.05% taker (VIP0)
  * 5bps rebalance slippage

The question: at $100k / VIP0, with the negative side throttled by borrow limits &
expensive borrow, is this still a viable book - and what does it actually look like?
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from funding_harvest_backtest import backtest, metrics  # noqa: E402

BR = json.load(open("user_data/data/binance/borrow_rates.json"))
borrow_map = {c: v["ann"] for c, v in BR.items()}
borrow_limit = {c: v["limit_usd"] for c, v in BR.items() if v.get("limit_usd")}

# VIP0 fee structures
TAKER = dict(fee_spot=0.0010, fee_fut=0.0005)            # plain VIP0 taker
TAKER_BNB = dict(fee_spot=0.00075, fee_fut=0.00045)      # 25% BNB discount
MAKER = dict(fee_spot=0.00075, fee_fut=0.00018)          # maker-preferred (BNB)

CFG = dict(n_max=49, thresh_ann=0.0)   # the walk-forward-selected low-turnover config


def line(label, **kw):
    eq, info = backtest(equity0=1e5, start_equity=1e5, **kw)
    m = metrics(eq, info, equity0=1e5, show=False)
    g, c, b = info["gross_funding"], info["costs"], info["borrow"]
    print(f"  {label:42s} CAGR={m['cagr']:+6.1%} Sharpe={m['sharpe']:5.2f}"
          f" maxDD={m['dd']:6.1%} | fund={g:+8.0f} cost={-c:+8.0f} borrow={-b:+7.0f}")


def run():
    print("=" * 92)
    print("STRATEGY #3 UNDER REAL BINANCE CONSTRAINTS  (AUM $100k, VIP0)")
    print("=" * 92)
    nlim = sum(1 for c in BR if BR[c].get("limit_usd") and BR[c]["limit_usd"] < 100_000)
    print(f"Real VIP0 borrow: rate range {min(borrow_map.values())*100:.1f}%-"
          f"{max(borrow_map.values())*100:.1f}%/yr; {nlim}/49 coins have a borrow "
          f"limit < $100k (short side throttled).\n")

    print("--- Build up the realism (low-turnover n_max=49 config throughout) ---")
    line("sim baseline (flat 5bps, flat 10% borrow)", **CFG, slippage=0.0005,
         spot_borrow_ann=0.10)
    line("+ real per-coin borrow RATES", **CFG, slippage=0.0005,
         borrow_map=borrow_map)
    line("+ real borrow LIMITS (short side capped)", **CFG, slippage=0.0005,
         borrow_map=borrow_map, borrow_limit=borrow_limit)
    line("+ real asymmetric fees (VIP0 taker)", **CFG, **TAKER, slippage=0.0005,
         borrow_map=borrow_map, borrow_limit=borrow_limit)
    print("  ^ this row = the honest VIP0 taker reality\n")

    print("--- Levers you actually control at VIP0 ---")
    line("VIP0 taker + BNB fee discount", **CFG, **TAKER_BNB, slippage=0.0005,
         borrow_map=borrow_map, borrow_limit=borrow_limit)
    line("maker-preferred fills (2bps slip)", **CFG, **MAKER, slippage=0.0002,
         borrow_map=borrow_map, borrow_limit=borrow_limit)
    print()

    print("--- Does the negative (short-spot) side still earn its keep? ---")
    line("BOTH sides, full real constraints", **CFG, **TAKER, slippage=0.0005,
         borrow_map=borrow_map, borrow_limit=borrow_limit, both_sides=True)
    line("POSITIVE-ONLY (no borrow needed)", **CFG, **TAKER, slippage=0.0005,
         both_sides=False)
    print("  (positive-only needs NO spot borrow -> immune to borrow rate/limit/availability)")

    print("\n--- Turnover lever on positive-only: rebalance cadence ---")
    for re in (3, 9, 21):
        line(f"positive-only, rebal every {re} steps ({re*8}h)", **CFG, **TAKER,
             slippage=0.0005, both_sides=False, rebal_every=re)


if __name__ == "__main__":
    run()
