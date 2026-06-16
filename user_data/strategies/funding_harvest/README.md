# Strategy #3 — Funding-Rate Harvesting (delta-neutral carry)

A market-neutral yield strategy. For each perpetual-swap pair we hold a **spot leg
hedged against a perp leg** so the position has ~no price exposure, and we collect
the **funding** the perp pays:

- funding **> 0** → **short perp + long spot** (longs pay us)
- funding **< 0** → **long perp + short spot** (shorts pay us; spot leg is borrowed)

Run across a wide Binance universe (49 same-venue pairs), the book is a *yield
portfolio*: rank by net-of-cost annualised funding, size by carry / basis-volatility
with hard per-pair caps, and hold delta-neutral.

> **Bottom line:** real, structural edge; capacity-limited to small/mid AUM; **the
> entire risk is in the tails and in execution cost**, both of which are now modelled
> and engineered for. Honest expectation at $100k on safe (≤3× cross) leverage:
> **~5% CAGR, Sharpe ~4.5, ~2% drawdown** in normal regimes.

---

## Folder layout

```
funding_harvest/
├── README.md                 ← this file
├── research/                 ← offline backtest + robustness (the "why we believe it")
│   ├── backtest.py             core leak-free vectorised sim + sensitivities
│   ├── analysis.py             walk-forward / OOS + capacity curve
│   ├── realdata.py             real Binance borrow rates/limits + asymmetric fees
│   ├── exec_costs.py           honest execution-cost model (rebalance bands, maker)
│   └── liqtest.py              perp-leg liquidation / tail stress test
└── live/                     ← production engine (the "how we run it")
    ├── config.py               StrategyConfig + HARD risk limits
    ├── signals.py              point-in-time signal + set-and-hold target book
    ├── broker.py               Broker ABC; PaperBroker, DryRunLiveBroker, LiveBroker
    ├── feed.py                 DataFeed ABC; ReplayFeed (feathers), LiveFeed (ccxt)
    ├── risk.py                 RiskMonitor (solvency, auto-deleverage, kill-switches)
    ├── engine.py               orchestration loop + two-leg executor
    ├── state.py                persist/restore the paper book between runs
    ├── testnet.py              Binance-sandbox wiring for the LiveBroker order path
    ├── run_replay.py           fidelity proof + safe-leverage replay
    └── run_paper.py            one live tick + schedulable forward test
```

Shared market data lives in `user_data/data/binance/` (8h spot+perp feathers,
`borrow_rates.json`, `oi_snapshot.json`). All paths resolve to the repo root, so
commands work from any cwd.

### Architecture principle — ONE ENGINE, TWO FEEDS
The same `engine.Engine` (decision + accounting) runs in two modes by swapping only
the **feed** and **broker** behind clean interfaces:

| Mode | Feed | Broker | Purpose |
|------|------|--------|---------|
| replay / backtest | `ReplayFeed` (feathers) | `PaperBroker` | validate vs the research backtest |
| paper-live / forward test | `LiveFeed` (public ccxt) | `DryRunLiveBroker` | live data, simulated fills, no keys |
| live | `LiveFeed` | `LiveBroker` (gated) | real two-leg account |

This makes it structurally impossible for the live path to silently diverge from
what was validated offline.

---

## How to run (mltrade python, from repo root)

```bash
PYTHONPATH=user_data/strategies python -m funding_harvest.research.backtest    # base + sensitivities
PYTHONPATH=user_data/strategies python -m funding_harvest.research.analysis     # walk-forward + capacity
PYTHONPATH=user_data/strategies python -m funding_harvest.research.realdata     # real-constraint economics
PYTHONPATH=user_data/strategies python -m funding_harvest.research.exec_costs   # honest execution costs
PYTHONPATH=user_data/strategies python -m funding_harvest.research.liqtest      # liquidation stress

PYTHONPATH=user_data/strategies python -m funding_harvest.live.run_replay       # fidelity proof + safe-lev replay
PYTHONPATH=user_data/strategies python -m funding_harvest.live.run_paper        # one live tick (dry-run, public data)
```

---

## What's been done & what we learned

The work was a deliberate march from "looks great in a naive sim" to "honest enough
to risk money on". Each step changed the picture.

### 1. Base backtest (`research/backtest.py`)
Leak-free vectorised sim on funding + spot + perp panels (freqtrade can't hold a
two-leg neutral position, so a native backtest would measure the wrong thing).
- Base: **+42% total / Sharpe 3.26 / maxDD −3.6%** (2022-02 → 2026-06, $10k, 10× perp).
- `signal_lag=1` barely moves results — the signature of no look-ahead.

### 2. Robustness (`research/analysis.py`)
- **Walk-forward OOS:** rolling 1.5y-train/0.5y-test, re-tuned each window → **+23%
  stitched OOS, Sharpe 5.73, 6/6 windows positive**; the optimiser independently
  re-selects the low-turnover config every window.
- **Capacity wall** (cap each pair at 1% of its open interest): clean to ~$1M, fading
  by $10–30M, dead by $100M — set by the thin-OI alts. **A small/mid-capital strategy.**

### 3. Real Binance constraints (`research/realdata.py`)
Pulled **real VIP0 cross-margin borrow rates + limits** (public margin spec) and the
true asymmetric fees (spot 0.10% / perp 0.05% taker).
- Under full real constraints at $100k: **CAGR +8.2% / Sharpe 6.5**; with maker fills
  + BNB discount **+10.4% / Sharpe 8.5**.
- Borrow *limits* (tiny at VIP0) bind only at $1M+, so at $100k the **negative side
  still earns its keep** (~doubles gross funding vs positive-only).

### 4. Liquidation / tail stress (`research/liqtest.py`)
The sim's ~0 hedge residual hides the real danger. From real perp high/low:
- Isolated **10× → 7,186** liquidation-trigger candles (the 99th-pct single candle,
  11.3%, already pierces the 9.5% buffer); **5× → 937; 3× → 144**.
- Sustained 3-day squeezes hit **+150–207%** (GMT/FIL/ENJ/APE) — beyond *any* fixed
  leverage. Those are exactly the high-funding names the book wants to short.
- **Verdict:** isolated margin is fatal. Mandatory → **cross/portfolio margin** (spot
  gain offsets perp loss in one collateral pool), **≤3× leverage**, auto-deleverage,
  and **exclude the worst squeeze names from the short side**.

### 5. Honest execution cost (`research/exec_costs.py`)
Charging held-position re-weighting properly (it was free before) exposed that
**naive daily re-weighting of a 49-name book is a money-loser** (taker −11.5%). The
discipline recovers it:
- maker fills (biggest single lever) + **set-and-hold** sizing + ~72h cadence + 5–8%
  entry threshold → **CAGR +9 to +14% / Sharpe ~5**, stable across a wide parameter
  region. *The edge is fixed; how much you keep is an execution problem.*

### 6. Production engine (`live/`)
- **Fidelity proven:** engine replay vs research backtest diverge by **0.012%** of
  final equity — the live decision/accounting path *is* the validated backtest.
- **Safe-leverage live number** (real borrow, ≤3× cross, maker, set-and-hold, 72h):
  **CAGR +5.09% / Sharpe 4.52 / maxDD −1.81%** (the cost of safe leverage vs the 10×
  research figure).
- **Risk overlay active:** 12 real funding-spike warnings over the replay; solvency /
  auto-deleverage never fired (hedged cross-margin book stays solvent — by design);
  drawdown kill-switch self-test fires on cue.
- **Live path validated end-to-end on real public data** (`run_paper.one_tick`): pulls
  all 49 pairs, builds a delta-neutral book, emits correct two-leg postOnly maker
  tickets, per-name short caps bind at 8% of equity.

### 7. Testnet + forward-test code (`live/testnet.py`, `live/run_paper.forward_test`)
- `LiveBroker` fully implements the two-leg lifecycle: postOnly maker on both legs,
  clientOrderId idempotency, fill-monitor with re-pricing, **partial-leg unwind**
  (a pair fills on both legs or neither — never naked), `reconcile()` vs exchange.
- `testnet.py` wires ccxt sandbox exchanges and a **guarded** one-cycle self-test.
- `forward_test()` is a single schedulable tick with **state persistence** (`state.py`)
  and CSV logging for the realized-vs-model comparison.

---

## Tests run (all green)

| Test | Result |
|------|--------|
| Base backtest reproducibility | +42.0% / Sharpe 3.26 (stable across refactors) |
| No-leak check (`signal_lag=1`) | results barely move ✓ |
| Walk-forward OOS | +23.3% stitched, Sharpe 5.73, 6/6 windows positive |
| Real-constraint economics | +8.2% (taker) … +10.4% (maker) at $100k |
| Liquidation stress | quantified; drove the cross-margin / ≤3× decision |
| Engine ↔ backtest fidelity | **0.012% divergence — MATCH** |
| Risk kill-switch self-test | HALT fires & flattens ✓ |
| Live tick on real public data | 49 pairs, 16-pair book, correct maker tickets ✓ |
| LiveBroker fail-closed gate | raises without exchanges + enable_live ✓ |
| Import / reorg smoke test | all 13 modules import ✓ |
| Trend-filter improvement test | see below — off by default at safe leverage |

### Cheap-improvement experiment: dynamic squeeze (trend) filter
We added an optional momentum-based filter (reusing strategy #2's signal) that skips a
carry leg when its perp side faces an extreme adverse trend (`trend_filter_z`, default
**OFF**). Honest result:
- At **high leverage (10×)** it helps: Sharpe ~3.6 → ~4.2.
- At the **deployed safe 3× config** it is **redundant with the static `short_exclude`
  blocklist and slightly hurts** (Sharpe 5.3 static-only → 4.7 with filter).
- **Conclusion:** the static squeeze-name blocklist (already in the live config) is the
  better cheap guard at safe leverage. The dynamic filter stays as a documented,
  toggle-able option (`trend_filter_z=2.0`), wired end-to-end and fidelity-verified, but
  **off by default**.

---

## Current status

**Code-complete through the forward-test harness; nothing live has been started.**
The research is done, the engine is proven faithful, the real-order path is written
and the testnet/forward-test scaffolding is in place but **dormant** (no keys, all
live entry points fail closed or are guarded).

Locked-in live configuration:
> both-sided · n_max≈49 · carry/vol sizing **at entry (set-and-hold)** · maker-preferred
> · ~72h rebalance · entry threshold ~5–8% · **cross/portfolio margin at ≤3×** with
> auto-deleverage · squeeze-prone names excluded from the short side (static blocklist)
> · per-name short cap 8% of equity · dynamic trend filter available but off.

> **Runs alongside strategy #2 (momentum)** as a low-correlation (+0.05) diversifier —
> see [`../portfolio/README.md`](../portfolio/README.md). Carry is the low-vol/high-Sharpe
> stabiliser; momentum is the return driver.

---

## What's next (the path to capital)

1. **Testnet validation.** Set `BINANCE_TESTNET_*` env keys, then
   `FH_RUN_TESTNET=1 python -m funding_harvest.live.testnet` to exercise the positive
   side end-to-end (open → fills → reconcile → close). *Caveat:* the spot testnet has
   no margin/borrow, so the short-spot/borrow leg must be validated with a tiny **real**
   mainnet position under supervision.
2. **Forward test ≥ 4–6 weeks.** Schedule `run_paper.forward_test()` once per 8h
   settlement (cron / freqtrade loop / Claude `/loop`); it persists state and logs a
   row each tick. Gate: does the live-data book track the replay's funding/cost/equity?
3. **Go-live gate.** Only if the forward test tracks: wire `LiveBroker` with real keys
   (least-privilege, **no withdrawal**), start at **minimum size**, watch margin and
   reconciliation, scale slowly toward the $100k target.

### Known limitations / honesty notes
- Sim omits intra-candle leg-fill timing and exact funding-settlement microstructure;
  the forward test exists to measure exactly that gap.
- OI capacity uses a present-day snapshot (Binance serves only ~30d of historical OI).
- Borrow rates are a current VIP0 snapshot; real borrow cost moves and spikes during
  the squeezes you most want to short — the per-name short exclusions mitigate this.
- `LiveBroker.resize()` is intentionally unimplemented (not needed under set-and-hold).
