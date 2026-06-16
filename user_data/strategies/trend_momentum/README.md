# Strategy #2 — Cross-Sectional + Time-Series Momentum (market-neutral)

A directional-factor book that is the **diversifier and tail-hedge** to the #3 carry
book. Where carry bleeds in violent trends/squeezes, momentum is long the movers and
short the laggards and tends to *make* money there. We run it **market-neutral**
(net ≈ 0) so it isolates the momentum factor and adds little BTC-beta.

- funding **edge:** returns auto-correlate — recent winners keep winning, losers keep
  losing (behavioural under-reaction + flows + reflexive leverage). PROVEN-IN-LITERATURE,
  replicated in crypto with fatter tails.
- **construction:** multi-horizon, risk-adjusted, cross-sectionally z-scored momentum
  → demean to net ≈ 0 → per-name cap → gross-normalise → vol-target → clamp gross.
- **universe:** the same 49 Binance perp pairs as #3 (shared 8h data).

> **Bottom line:** a genuine but *lower-Sharpe, higher-drawdown* book on its own
> (Sharpe ~0.8 full-sample, ~1.0 OOS, maxDD ~20%). Its value is the **near-zero
> correlation to carry** — see `../portfolio/`.

---

## Folder layout
```
trend_momentum/
├── README.md
├── research/
│   ├── backtest.py     leak-free market-neutral momentum sim (funding-aware costs)
│   └── analysis.py     leak test · cost/cadence ladder · lookback grid · walk-forward
└── live/               one-engine-two-feeds (mirrors funding_harvest)
    ├── config.py  signals.py  broker.py  feed.py  risk.py  engine.py
    ├── state.py   testnet.py  run_replay.py  run_paper.py
```

## Run (mltrade python, from repo root)
```bash
PYTHONPATH=user_data/strategies python -m trend_momentum.research.backtest
PYTHONPATH=user_data/strategies python -m trend_momentum.research.analysis
PYTHONPATH=user_data/strategies python -m trend_momentum.live.run_replay   # fidelity proof
PYTHONPATH=user_data/strategies python -m trend_momentum.live.run_paper    # one live tick (dry-run)
```

---

## What's been done & what we learned

### 1. Backtest (`research/backtest.py`)
Leak-free, vol-targeted, market-neutral L/S on 8h perp panels; signed positions,
price + funding P&L, turnover cost. `signal_lag=1` barely moves results (no leak).

### 2. Robustness (`research/analysis.py`) — the decisive iteration
- **Single-lookback momentum is fragile:** Sharpe peaks sharply at one horizon
  (~14 days) and the **rolling walk-forward OOS was weak (Sharpe 0.16)**.
- **Fix = multi-horizon blend + don't re-tune.** Blending horizons (14/21/42 days)
  and committing to a *fixed* config lifted **rolling OOS Sharpe 0.16 → 1.00**
  (CAGR +17.5%, maxDD −22%). The overfitting lesson made concrete: a robust blend
  with no per-window tuning beats chasing the best single lookback.
- **Cost/cadence:** weekly rebalance keeps Sharpe (~0.9) at a fraction of daily's
  turnover cost → weekly is the default.

### 3. Live engine (`live/`)
Same **one-engine-two-feeds** design as #3 (single-leg, so simpler execution).
- **Fidelity proven:** engine replay reproduces the research backtest to **0.000%**
  of final equity — the live decision/accounting path *is* the backtest.
- **Risk overlay:** drawdown kill-switch (25%, momentum draws deeper than carry),
  net-exposure band (keep ≈ neutral), gross-cap breach; kill-switch self-test passes.
- **Live path validated** on real public data (`run_paper.one_tick`): builds the
  market-neutral book, emits correct single-leg postOnly tickets, net ≈ 0.

### 4. Testnet + forward test (`live/testnet.py`, `run_paper.forward_test`)
- `LiveBroker` = single-leg perp, postOnly maker, clientOrderId, reduce-only on
  de-risking trades, `reconcile()`; triple-gated.
- **#2 is the cleaner first live validation than #3:** a single-leg perp book is
  *fully* testable on the Binance **futures testnet** (no untestable spot-borrow leg).
- `forward_test()` = one schedulable 8h tick with state persistence + CSV logging.

---

## Numbers (market-neutral L/S, $10k, 2022-02 → 2026-06)

| View | CAGR | Sharpe | maxDD | net |
|------|:---:|:---:|:---:|:---:|
| Full-sample (blended, weekly) | +13.1% | 0.76 | −19.3% | ≈0% |
| Rolling walk-forward OOS (fixed blend) | +17.5% | **1.00** | −22.3% | ≈0% |
| Fixed-split OOS (2024-06→) | +20.5% | 1.07 | −26.9% | ≈0% |

Funding is net-positive (short legs collect it). Standalone Sharpe is modest — the
point is the **+0.05 correlation to carry** (see portfolio).

---

## Tests run (all green)

| Test | Result |
|------|--------|
| Leak test (`signal_lag=1`) | results barely move ✓ |
| Multi-horizon blend vs single | rolling OOS Sharpe 0.16 → 1.00 ✓ |
| Cost/cadence ladder | weekly ≈ daily Sharpe at ¼ the cost ✓ |
| Engine ↔ backtest fidelity | **0.000% divergence — MATCH** ✓ |
| Risk kill-switch self-test | HALT fires ✓ |
| Live tick on real public data | market-neutral book, correct tickets ✓ |
| LiveBroker fail-closed gate | raises without exchange + enable_live ✓ |

---

## Current status

**Code-complete through the forward-test harness; nothing live started.** Locked-in
config:
> market-neutral L/S · multi-horizon momentum blend (14/21/42d) · vol-targeted 20%
> · per-name cap 15% · gross ≤ 3× · weekly rebalance · maker-preferred · $10k.

## What's next (path to capital)
1. **Testnet validation** (easy here — single-leg perp on the futures testnet):
   `FH_RUN_TESTNET=1 python -m trend_momentum.live.testnet`.
2. **Forward test ≥ 4–6 weeks:** schedule `run_paper.forward_test()` per 8h; compare
   the live-data book to the replay.
3. **Run alongside #3** via `../portfolio/` (two $10k books, combined risk overlay).
4. **Go-live gate:** only if the forward test tracks — wire `LiveBroker` with
   least-privilege keys (no withdrawal), start at minimum size.

### Honesty notes
- Standalone momentum is a **modest** edge (Sharpe ~1 OOS) with real ~20% drawdowns;
  it earns its place as a *diversifier*, not a star.
- Universe is today's 49 listed pairs (mild survivorship bias; no delisted names).
- Diagonal vol-target ignores cross-correlation (crypto is high-beta); the net-band
  and gross cap are the backstops. Risk-weighting vs #3 is a portfolio choice (see
  `../portfolio/README.md`).
