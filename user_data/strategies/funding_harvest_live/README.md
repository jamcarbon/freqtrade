# Strategy #3 — Funding-Rate Harvesting (delta-neutral), end to end

This documents the full research → backtest → robustness → real-data → tail-risk →
execution-model → live-engine → forward-test work for strategy #3 from
*Crypto Systematic Trading — Strategy Research Report.md*.

**The edge:** crypto perpetual funding is structurally positive (leveraged longs pay
shorts), and on a subset of alts deeply negative (shorts pay longs). Harvest it
delta-neutral — long spot / short perp on positive funding, short spot / long perp
on negative funding — collecting funding while price legs cancel.

Freqtrade can't backtest this natively (it trades one instrument per position and
can't hold a spot hedge against a perp short), so the whole thing is a dedicated
vectorised simulation plus a production execution engine, all leak-free and
point-in-time correct.

---

## Honest expected economics (≤$100k, the target size)

| Setting | CAGR | Sharpe | maxDD |
|---|---:|---:|---:|
| Sim, 10x isolated, optimistic costs (NOT live-realistic) | ~+9–14% | ~5 | ~−2.5% |
| **Real Binance constraints, ≤3x cross margin, maker, disciplined** | **~+5%** | **~4.5** | **~−1.8%** |
| Naive (taker, daily re-weighting) | **negative** | <0 | −40%+ |

A real, market-neutral ~5% yield at safe leverage — modest, and **entirely
dependent on execution discipline and tail-safe margining.** The naive
implementation loses money; the discipline is the strategy.

---

## What was learned (each step changed the design)

1. **Backtest (49 Binance pairs, 2022–2026):** carry edge is real; both-sided +42%
   total / Sharpe 3.3 on a concentrated book (optimistic costs).
2. **Robustness:** walk-forward OOS positive (Sharpe ~3.9–5.7, 6/6 windows); the
   optimizer independently selects a **low-turnover** book every window. Capacity
   wall ~$10–30M — far above the target size.
3. **Real exchange data:** survives real VIP0 borrow rates (0.4–58%/yr), real
   borrow limits, and real asymmetric fees (spot 0.10% / perp 0.05% taker). At
   $100k the borrow limits don't bind, so the negative side still ~doubles funding.
4. **Liquidation stress (real high/low):** isolated 10x = 7,186 liquidation candles;
   sustained squeezes +150–207%. **Isolated margin is fatal → cross/portfolio
   margin + ≤3x + auto-deleverage is mandatory.**
5. **Execution cost (honest re-weight accounting):** naive daily re-weighting is
   negative. **Set-and-hold sizing + maker fills + ~72h cadence** recovers it.
6. **Live engine:** one-engine-two-feeds; replay reproduces the backtest to 0.012%.
7. **Forward-test bridge:** runs against real public data, dry-run safe.

---

## Locked-in live configuration

- Universe: 49 same-venue Binance USDT pairs (both spot + perp + funding).
- Both-sided; `n_max ≈ 49` (hold the whole qualifying universe = low turnover).
- Carry/vol sizing **applied at entry, then held** (set-and-hold; no daily reweight).
- Entry threshold ~5–8% annualised |funding|, exit at lower (hysteresis deadband).
- Rebalance every ~72h. Maker-preferred fills (postOnly), BNB fee discount.
- **Cross / portfolio margin at ≤3x**, auto-deleverage on margin distance.
- Per-name short caps (≤8% equity) and exclusion of squeeze-prone names
  (GMT/APE/ENJ/FIL/SNX/XLM/SAND) on the short-spot side.

---

## File map

Research / vectorised sims (`user_data/strategies/`):
- `funding_harvest_backtest.py` — core both-sided delta-neutral sim + sensitivities.
- `funding_harvest_analysis.py` — walk-forward/OOS, friction ladder, capacity curve.
- `funding_harvest_realdata.py` — re-run under REAL borrow rates/limits/fees.
- `funding_harvest_liqtest.py` — perp-leg liquidation stress test by leverage.
- `funding_harvest_exec.py` — honest execution-cost model (rebalance bands, maker).
- `funding_carry_backtest.py` (#1), `DualMomentumXS.py` (#2) — sibling strategies.

Production engine (`funding_harvest_live/`):
- `config.py` — StrategyConfig + hard risk limits.
- `signals.py` — point-in-time funding signal + set-and-hold target-book builder.
- `broker.py` — `Broker` ABC; `PaperBroker`, `DryRunLiveBroker`, gated `LiveBroker`.
- `feed.py` — `DataFeed` ABC; `ReplayFeed` (feathers), `LiveFeed` (public ccxt).
- `risk.py` — `RiskMonitor` (drawdown kill, funding-spike, delta band, solvency).
- `engine.py` — orchestration loop + two-leg executor (leg-risk unwind).
- `run_replay.py` — fidelity proof + safe-leverage replay + risk self-test.
- `run_paper.py` — paper-LIVE one tick + forward-test CSV scaffold.

Data snapshots (`user_data/data/binance/`):
- `borrow_rates.json` — real VIP0 cross-margin borrow rates + limits (per coin).
- `oi_snapshot.json` — current open interest per coin (capacity ceiling).

---

## How to run

Interpreter: the `mltrade` conda env. From the repo root:

```bash
PY="C:/Users/<you>/miniconda3/envs/mltrade/python.exe"

# vectorised research sims
"$PY" user_data/strategies/funding_harvest_backtest.py
"$PY" user_data/strategies/funding_harvest_analysis.py
"$PY" user_data/strategies/funding_harvest_realdata.py
"$PY" user_data/strategies/funding_harvest_liqtest.py
"$PY" user_data/strategies/funding_harvest_exec.py

# production engine: replay fidelity + risk self-test
PYTHONPATH=user_data/strategies "$PY" -m funding_harvest_live.run_replay

# paper-LIVE one tick against real Binance public data (dry-run, no keys)
PYTHONPATH=user_data/strategies "$PY" -m funding_harvest_live.run_paper
```

Price data is downloaded via `freqtrade download-data` (spot 8h + futures 8h +
funding_rate + mark for the 49-pair universe, from 2022-01-01).

---

## Remaining before real capital

1. Implement `LiveBroker._place_two_legs` against **Binance testnet**: postOnly per
   leg, fill monitoring, **partial-leg unwind** (never sit directionally exposed),
   reconciliation against exchange truth.
2. Persist `DryRunLiveBroker` state between runs; schedule `run_paper.forward_test()`
   on the 8h settlement cadence.
3. Run the **forward test for ≥4–6 weeks**, comparing realized vs modeled
   funding / borrow / slippage / fills.
4. Promote only if live tracks the replay: swap `DryRunLiveBroker` →
   `LiveBroker(enable_live=True)`.

**Caveats that remain:** OI is a present-day snapshot (Binance serves only ~30d of
historical OI); borrow rates are a current snapshot and spike exactly during the
squeezes you're short into; the sim's near-zero hedge residual omits leg-fill
timing and intra-rebalance slippage. The forward test exists to measure these.
