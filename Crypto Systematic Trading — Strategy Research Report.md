# Crypto Systematic Trading — Strategy Research Report

**Prepared for:** Technical founders, quant developers, and strategy research teams
**Platform context:** Freqtrade-class orchestration & execution engine (rule-based strategies, FreqAI ML pipeline — LightGBM/XGBoost regressors & classifiers, PyTorch MLP/Transformer, stable-baselines3 reinforcement learning — futures/leverage support, orderflow & order book data, hyperopt, and built-in lookahead/recursive bias detection).
**Author role:** Quantitative researcher / systematic crypto strategist
**Status of claims:** Every "edge" below is labeled **HYPOTHESIS** unless I explicitly mark it **PROVEN-IN-LITERATURE**. No backtest numbers are asserted — none were provided. Profitability/risk ratings are *priors* to guide research sequencing, not results.

---

## ⚠️ Read This First (Epistemic Health Warning)

This report ranks **research bets**, not money-printers. In crypto specifically:

- The majority of strategies that look profitable in a naive backtest die on **fees + slippage + funding + latency**. Most of this report's value is in telling you *which bets survive contact with an exchange*.
- Crypto data is dirty: wash trading (especially on low-tier exchanges), spoofed order books, exchange-reported volume inflation, gappy historical funding/OI data, and survivorship bias (delisted/depegged tokens vanish from datasets).
- **Backtest profitability and live profitability are different random variables.** Treat any in-sample Sharpe with extreme suspicion. The platform's `lookahead-analysis` and `recursive-analysis` tools are not optional — they are the difference between a real edge and a leaked one.
- Capacity is the silent killer. A strategy that prints on $10k often evaporates at $1M because you *are* the order book.

---

## 1. Executive Summary

The strongest **risk-adjusted, automatable, platform-compatible** opportunities in crypto today cluster around **structural and microstructure edges that are hard to arbitrage away and have reliable data**, rather than around exotic ML. My core recommendations:

1. **Lead with carry/structure, not prediction.** Funding-rate and basis-driven strategies (perp-spot carry, funding harvesting) exploit a *structural* feature of crypto perpetual markets — leveraged longs persistently pay shorts. This is the most robust, data-reliable, and capacity-respecting family. It is closer to **PROVEN-IN-LITERATURE** than anything else here.

2. **Trend-following is the honest workhorse.** Cross-sectional and time-series momentum is the most robust *directional* edge across asset classes and survives crypto's fat-tailed, trending regimes. Low overfitting risk, excellent automation fit, but suffers in chop and requires regime awareness.

3. **Use ML as a filter/sizer, not an oracle.** ML as the *primary* decision engine on raw price prediction is the most overfit-prone, lowest-robustness category in crypto. ML earns its keep in **hybrid** roles: regime classification, signal ranking, meta-labeling (trade/skip), and adaptive position sizing. This is where the platform's FreqAI module is genuinely differentiated.

4. **Treat single-venue ML price prediction as a research liability until proven.** High capacity for self-deception, high infra cost, weak live robustness. Allowed in the roadmap, but sandboxed and last.

**Build order:** Funding/basis carry → Trend-following (TS + cross-sectional momentum) → Mean-reversion (range/stat-arb pairs) → Hybrid meta-labeling & regime filters → ML sizing → (sandbox) ML directional prediction.

---

## 2. Strategy Ranking Table

Ranked by **overall research priority** = blend of realistic profitability, robustness, data reliability, execution feasibility, scalability, automation fit, and improvability. Ratings 1–10 (10 = best for profitability/priority; for **Risk**, 10 = *most* risky).

| # | Strategy | Category | Profit (1-10) | Risk (1-10) | Priority (1-10) | Edge confidence |
|---|----------|----------|:---:|:---:|:---:|---|
| 1 | Perp-Spot Funding/Basis Carry | Rule-Based | 7 | 4 | 9 | Strong (structural) |
| 2 | Time-Series + Cross-Sectional Momentum (Trend) | Rule-Based | 7 | 5 | 9 | Strong |
| 3 | Funding-Rate Harvesting (delta-neutral) | Rule-Based | 6 | 4 | 8 | Strong (structural) |
| 4 | Meta-Labeling Filter on Rule Signals | Hybrid | 6 | 5 | 8 | Moderate |
| 5 | Regime-Switching Allocator (ML regime → rule book) | Hybrid | 6 | 5 | 7 | Moderate |
| 6 | Statistical Pairs / Cointegration Mean-Reversion | Rule-Based | 6 | 6 | 7 | Moderate |
| 7 | Cross-Exchange / Triangular Arbitrage | Rule-Based | 6 | 5 | 6 | Strong but infra-gated |
| 8 | Volatility Breakout (Donchian/ATR) | Rule-Based | 5 | 6 | 6 | Moderate |
| 9 | ML Position Sizing / Risk Overlay | Hybrid | 5 | 4 | 6 | Moderate |
| 10 | Intraday Mean-Reversion (Bollinger/RSI) | Rule-Based | 5 | 6 | 5 | Weak-Moderate |
| 11 | Order-Flow / Order-Book Imbalance | Rule-Based/Hybrid | 6 | 7 | 5 | Moderate but infra-gated |
| 12 | Gradient-Boosted Return Classifier (FreqAI) | ML | 5 | 7 | 5 | Weak (overfit-prone) |
| 13 | Sequence Model (Transformer/LSTM) Forecast | ML | 4 | 8 | 4 | Weak |
| 14 | Reinforcement Learning Execution/Trading Agent | ML | 4 | 9 | 3 | Speculative |
| 15 | Anomaly/Clustering Regime Detection (standalone) | ML | 3 | 6 | 3 | Weak as primary |

> **Interpretation:** The top of the table is dominated by **rule-based structural and trend edges** and **hybrid filters**. Pure ML clusters at the bottom — not because ML is useless, but because as a *primary* engine on crypto price data it has the worst overfit/robustness profile. Its value is realized in hybrid roles (#4, #5, #9).

---

## 3. Rule-Based Strategy Candidates

### 3.1 Perp-Spot Funding/Basis Carry  — *Priority 9*

- **Category:** Rule-Based
- **Core idea:** Capture the persistent premium of perpetual futures over spot. When perp trades above spot (positive basis / positive funding), go **long spot + short perp** to collect funding while being delta-neutral; reverse when basis is negative.
- **Edge exploited:** *Structural.* Crypto perps have no expiry; the funding mechanism forces price convergence by paying the side that stabilizes the peg. Retail leverage demand keeps funding persistently positive on average. **PROVEN-IN-LITERATURE** as a carry premium; the edge is a risk premium for providing liquidity to leveraged longs, not a pricing mistake.
- **Required data inputs:** Spot mark price, perp mark/index price, funding rate history & schedule, borrow/lending rates (if margin on spot leg), fees.
- **Recommended timeframe:** Funding-period-aligned (1h/8h depending on venue); positions held hours→weeks. Decision cadence: hourly.
- **Suitable markets:** Spot + perpetual futures on the *same* venue (lowest execution risk) or cross-venue (higher yield, higher risk). Portfolio-level across many pairs.
- **Entry logic:** Enter delta-neutral when annualized funding/basis exceeds a threshold that comfortably clears round-trip fees + expected funding decay (e.g., annualized carry > X% with X set above all-in cost). Size by funding magnitude and liquidity.
- **Exit logic:** Exit when carry compresses below cost threshold, flips sign, or when basis mean-reverts to target. Hard exit on funding regime break.
- **Risk management:** Delta-neutral by construction, but watch (a) **liquidation risk on the short perp leg** during spot squeezes — keep margin buffers large; (b) funding flip; (c) venue/counterparty risk (de-peg, withdrawal halts). Cap per-venue exposure.
- **Position sizing:** Risk-parity across pairs weighted by carry/vol and liquidity; cap by order-book depth and margin headroom (keep maintenance margin distance ≥ a large multiple of expected adverse move).
- **Execution considerations:** Both legs must fill near-simultaneously to avoid directional exposure (leg risk). Rebalance delta as spot moves. Funding is paid at fixed times — be in position before the snapshot. Beware fee tiers eating the carry.
- **Backtesting requirements:** Accurate historical funding *and* basis time series (often the weakest data link), realistic fees, margin/liquidation modeling. The platform supports futures; you must supply reliable funding history.
- **Key features/indicators:** Annualized funding, basis %, funding z-score, OI, liquidation proximity.
- **Strengths:** Market-neutral, low drawdown in normal regimes, high data reliability, strong automation fit, scales reasonably.
- **Weaknesses:** Yields compress as more capital chases them; capital-intensive (margin on both legs); tail risk concentrated in stress events.
- **Failure modes:** Liquidation cascade spikes funding *and* gaps the short leg; venue insolvency/withdrawal freeze; stablecoin de-peg on the spot leg; funding goes deeply negative unexpectedly.
- **Overfitting risk:** Low (few parameters, structural edge).
- **Data leakage risk:** Low, *if* funding is applied at the correct historical timestamp (a common backtest bug — applying funding you couldn't yet observe). Validate with lookahead-analysis.
- **Infrastructure:** Reliable multi-leg execution, margin monitoring, funding-snapshot timing. Moderate.
- **Difficulty:** Moderate (delta management + margin safety is the hard part).
- **Scalability/capacity:** Good — among the highest-capacity strategies, bounded by perp OI/liquidity.
- **Live robustness:** High in normal regimes; the entire risk is in the tails.
- **Profit 7 / Risk 4 / Priority 9.**

### 3.2 Time-Series + Cross-Sectional Momentum (Trend-Following) — *Priority 9*

- **Category:** Rule-Based
- **Core idea:** Crypto trends. Combine **time-series momentum** (own-asset return sign over a lookback drives long/short) with **cross-sectional momentum** (long top-quantile performers, short bottom-quantile across a basket).
- **Edge exploited:** Behavioral under-reaction + flows + reflexive leverage create persistent autocorrelation in returns. **PROVEN-IN-LITERATURE** across asset classes and replicated in crypto, though with higher turnover and fatter tails.
- **Required data inputs:** OHLCV across a liquid universe; funding (for futures carry-aware sizing); volume/liquidity for universe selection.
- **Recommended timeframe:** Daily or 4h–12h signals; multi-day to multi-week holds. Higher frequencies decay faster and pay more fees.
- **Suitable markets:** Perpetual futures (best — can short, leverage) and spot. Portfolio-level basket.
- **Entry logic:** Long when fast/slow trend filter aligns up and breakout confirms; cross-sectional: rank universe by risk-adjusted momentum, long top decile / short bottom decile. Volatility-normalize signals.
- **Exit logic:** Trend reversal, trailing stop (ATR/Chandelier), or rank dropping out of band. Time-based exit if signal stagnates.
- **Risk management:** Volatility targeting at portfolio and per-position level; correlation caps (crypto is one big beta to BTC — net exposure must be controlled); max gross/net leverage limits; circuit-breaker on realized vol spikes.
- **Position sizing:** Inverse-vol / vol-targeting; cap single-name and sector weight; reduce gross in high-correlation regimes.
- **Execution considerations:** Lower turnover = friendlier to fees. Rebalance on a schedule, not tick-by-tick. Watch crowding around obvious lookbacks.
- **Backtesting requirements:** Survivorship-bias-free universe (include delisted coins!), realistic fees/slippage, point-in-time universe membership. The platform handles multi-pair; you must curate the universe honestly.
- **Key features/indicators:** Multi-horizon returns, EMA crossovers, Donchian channels, ADX/regime filter, realized vol, cross-sectional rank.
- **Strengths:** Robust, few parameters, crisis-alpha (often profits in crashes via shorts), excellent automation fit, improvable.
- **Weaknesses:** Whipsaws in range-bound regimes; correlated drawdowns; "everything is BTC beta" reduces cross-sectional breadth.
- **Failure modes:** Sharp V-reversals; prolonged chop; funding bleed on held shorts; sudden correlation→1 events.
- **Overfitting risk:** Low–moderate (keep lookbacks coarse and few; resist hyperopt over-tuning).
- **Data leakage risk:** Low if signals use only closed candles. Validate.
- **Infrastructure:** Modest. Scheduled rebalancing + multi-pair data.
- **Difficulty:** Low–moderate.
- **Scalability/capacity:** Good on majors; limited by liquidity on small caps in the cross-section.
- **Live robustness:** High, with regime awareness.
- **Profit 7 / Risk 5 / Priority 9.**

### 3.3 Funding-Rate Harvesting (Delta-Neutral, generalized) — *Priority 8*

- **Category:** Rule-Based
- **Core idea:** A focused variant of 3.1 — systematically harvest funding across *all* pairs where annualized funding clears costs, while neutralizing delta. Treat it as a yield portfolio.
- **Edge exploited:** Same structural funding premium; diversification across many pairs smooths the carry. **PROVEN-IN-LITERATURE**.
- **Data inputs:** Funding history, OI, spot/perp prices, fees, borrow rates.
- **Timeframe:** Funding-period cadence; days→weeks holds.
- **Markets:** Spot+perp, cross-venue optional, portfolio-level.
- **Entry:** Rank pairs by net-of-cost annualized funding; allocate to top names with sufficient liquidity and acceptable de-peg/venue risk.
- **Exit:** Carry compression, funding flip, liquidity deterioration, risk-limit breach.
- **Risk management:** Per-pair and per-venue caps; margin buffers; exclude illiquid/manipulated names; de-peg monitoring on collateral.
- **Position sizing:** Carry/vol-weighted with hard liquidity caps.
- **Execution:** Simultaneous two-leg fills; periodic delta rebalancing; funding-snapshot timing.
- **Backtesting:** Same as 3.1; funding-data fidelity is the bottleneck.
- **Strengths/weaknesses/failure modes:** As 3.1, with added diversification benefit but more operational surface area.
- **Overfitting risk:** Low. **Leakage risk:** Low with correct funding timestamps.
- **Infra:** Moderate–high (many simultaneous neutral pairs). **Difficulty:** Moderate.
- **Capacity:** Good. **Live robustness:** High ex-tails.
- **Profit 6 / Risk 4 / Priority 8.**

### 3.4 Statistical Pairs / Cointegration Mean-Reversion — *Priority 7*

- **Category:** Rule-Based (stat-arb)
- **Core idea:** Find cointegrated crypto pairs/baskets; trade the spread's mean reversion (long the cheap leg, short the rich leg) when the z-score deviates.
- **Edge exploited:** Temporary dislocations between economically linked assets (e.g., L1s, or a token vs. a basket). **HYPOTHESIS** in crypto — cointegration relationships are *unstable* and regime-dependent, far less reliable than in equities.
- **Data inputs:** Synchronized OHLCV across candidate universe; funding for the short leg.
- **Timeframe:** Hours→days.
- **Markets:** Perp/perp (easiest to short both ways) or spot/perp.
- **Entry:** Spread z-score beyond threshold with a *recently re-validated* cointegration relationship.
- **Exit:** Spread reverts to mean, stop-out on z-score blowout (relationship broke), time stop.
- **Risk management:** Hard stop when the relationship breaks (the classic stat-arb killer); cap per-spread risk; monitor rolling cointegration p-value.
- **Position sizing:** Hedge-ratio (beta) weighted; inverse-vol on the spread.
- **Execution:** Two-leg fills; funding cost on shorts; rebalance hedge ratio.
- **Backtesting:** Rolling re-estimation of hedge ratios with **strict point-in-time** discipline (huge leakage trap — using full-sample cointegration is the #1 stat-arb backtest lie). Use recursive-analysis to catch it.
- **Key features:** Spread, z-score, rolling beta, half-life of mean reversion, ADF/Johansen tests.
- **Strengths:** Market-neutral, uncorrelated to trend book (good diversifier).
- **Weaknesses:** Relationships decay; crowded; crypto fundamentals shift fast (forks, narrative rotation).
- **Failure modes:** Permanent regime break (one token re-rates), funding bleed while waiting for reversion that never comes.
- **Overfitting risk:** High (universe mining for "pairs that worked" is rampant). **Leakage risk:** High (look-ahead cointegration). Both must be actively controlled.
- **Infra:** Moderate. **Difficulty:** Moderate–high.
- **Capacity:** Moderate. **Live robustness:** Moderate.
- **Profit 6 / Risk 6 / Priority 7.**

### 3.5 Cross-Exchange / Triangular Arbitrage — *Priority 6*

- **Category:** Rule-Based
- **Core idea:** Exploit transient price differences for the same asset across venues, or triangular inconsistencies within one venue.
- **Edge exploited:** Fragmented liquidity and latency. **PROVEN** but **infrastructure-gated** — the edge is real but accrues to the fastest, lowest-fee, best-capitalized participants.
- **Data inputs:** Real-time order books across venues, fees, withdrawal/transfer times & limits, latency.
- **Timeframe:** Sub-second to seconds.
- **Markets:** Cross-exchange spot; intra-venue triangular.
- **Entry/Exit:** Execute when cross-venue spread > all-in cost (fees + slippage + transfer risk); close instantly.
- **Risk management:** Inventory limits per venue; pre-funded balances (can't rely on real-time transfers); kill-switch on stale data.
- **Position sizing:** Bounded by per-venue inventory and book depth.
- **Execution:** Latency is the whole game; co-location/region selection; robust failover. This is more an *execution engineering* problem than a research problem.
- **Backtesting:** Requires high-res multi-venue book + latency simulation — most retail backtesters (this platform included) are *not* built for this.
- **Strengths:** Near market-neutral, high Sharpe *if* you can compete. **Weaknesses:** Capital fragmented across venues; transfer/withdrawal risk; brutal competition; thin margins.
- **Failure modes:** Latency loss (adverse fill), withdrawal halt stranding inventory, exchange outage mid-arb.
- **Overfitting risk:** Low. **Leakage risk:** Low conceptually, but backtests massively overstate fills.
- **Infra:** Very high. **Difficulty:** High (infra), low (logic).
- **Capacity:** Moderate, fragmented. **Live robustness:** Moderate, infra-dependent.
- **Profit 6 / Risk 5 / Priority 6** — *deprioritized for the core build because the infra demands sit largely outside a Freqtrade-class platform. Becomes viable with colocation — see the latency roadmap in §14.*

### 3.6 Volatility Breakout (Donchian / ATR / Keltner) — *Priority 6*

- **Category:** Rule-Based
- **Core idea:** Enter on confirmed breakouts of a volatility-defined channel; ride expansion.
- **Edge exploited:** Volatility clustering and breakout follow-through. **HYPOTHESIS / weak-moderate** — closely related to trend; standalone it whipsaws.
- **Data inputs:** OHLCV; ATR/realized vol.
- **Timeframe:** 1h–1d.
- **Markets:** Perp/spot.
- **Entry:** Close beyond N-period Donchian high/low with vol/volume confirmation.
- **Exit:** Trailing ATR stop, opposite channel, time stop.
- **Risk management:** ATR-based stops and sizing; avoid trading through known low-liquidity windows.
- **Position sizing:** Risk-per-trade fixed fraction via ATR.
- **Execution:** False breakouts and stop-hunts are endemic in crypto — require confirmation.
- **Backtesting:** Watch fee drag from frequent false signals; realistic slippage on breakout candles (you fill *worse* than the breakout level).
- **Strengths:** Simple, captures big moves. **Weaknesses:** Many small losses, regime-dependent.
- **Failure modes:** Chop, fakeouts, liquidity-vacuum wicks.
- **Overfitting risk:** Moderate (channel length tuning). **Leakage risk:** Low.
- **Infra:** Low. **Difficulty:** Low. **Capacity:** Good on majors. **Live robustness:** Moderate.
- **Profit 5 / Risk 6 / Priority 6.**

### 3.7 Intraday Mean-Reversion (Bollinger / RSI bands) — *Priority 5*

- **Category:** Rule-Based
- **Core idea:** Fade short-term extremes back to a moving average.
- **Edge exploited:** Short-horizon overreaction. **WEAK–HYPOTHESIS** — works in ranges, gets run over in trends; very crowded and fee-sensitive.
- **Data inputs:** OHLCV.
- **Timeframe:** 5m–1h.
- **Markets:** Liquid spot/perp only.
- **Entry:** Price beyond Bollinger band + RSI extreme, *with a trend filter forbidding counter-trend entries*.
- **Exit:** Reversion to mean / opposite band / tight stop / time stop.
- **Risk management:** Hard stops mandatory (mean reversion without stops = picking up pennies in front of a truck); regime filter to disable in strong trends.
- **Position sizing:** Small, fixed-fraction; reduce in high-vol.
- **Execution:** High turnover → fees dominate; needs maker fills or tight spreads to be viable.
- **Backtesting:** Fee/slippage realism is make-or-break; beware fills at the band that wouldn't happen live.
- **Strengths:** High hit rate in ranges. **Weaknesses:** Negative skew, trend risk, fee drag.
- **Failure modes:** Strong directional trend, vol expansion, gap through stop.
- **Overfitting risk:** High. **Leakage risk:** Low–moderate.
- **Infra:** Low. **Difficulty:** Low. **Capacity:** Moderate (turnover-limited). **Live robustness:** Low–moderate.
- **Profit 5 / Risk 6 / Priority 5.**

### 3.8 Order-Flow / Order-Book Imbalance — *Priority 5*

- **Category:** Rule-Based (or Hybrid with ML on features)
- **Core idea:** Use order-book imbalance, trade aggression (CVD), and footprint to predict very-short-horizon direction. The platform exposes orderflow/order-book data (see advanced-orderflow).
- **Edge exploited:** Microstructure pressure precedes short moves. **HYPOTHESIS / moderate** — real at HFT horizons, decays fast, and is heavily polluted by spoofing/wash trades.
- **Data inputs:** L2 order book, tick/trade data, CVD, footprint — high volume, storage-heavy.
- **Timeframe:** Seconds–minutes.
- **Markets:** Most liquid perps/spot.
- **Entry/Exit:** Imbalance/CVD thresholds; very tight, fast exits.
- **Risk management:** Hard latency-aware stops; disable on data staleness; ignore obvious spoof patterns.
- **Position sizing:** Small, inventory-bounded.
- **Execution:** Latency-sensitive; maker/taker mix matters; spoofing makes naive book signals dangerous.
- **Backtesting:** Requires high-fidelity tick + book replay — **hard**; most backtests here are unreliable. Forward-test heavily.
- **Strengths:** Genuinely different edge, diversifying. **Weaknesses:** Data-heavy, manipulation-prone, fast decay.
- **Failure modes:** Spoofing-induced false signals, latency loss, data gaps.
- **Overfitting risk:** High. **Leakage risk:** High (book snapshot timing).
- **Infra:** High (data + latency). **Difficulty:** High. **Capacity:** Low–moderate. **Live robustness:** Moderate, fragile.
- **Profit 6 / Risk 7 / Priority 5.**

---

## 4. Machine Learning Strategy Candidates

> **Framing:** As a *primary* decision engine on crypto price data, ML's main product is overconfidence. The platform's FreqAI supports LightGBM/XGBoost (regressor & classifier, multi-target), PyTorch MLP/Transformer, and RL — but the binding constraint is never the model; it's **labeling, leakage control, and non-stationarity.** Rate ML candidates harshly until they prove out.

### 4.1 Gradient-Boosted Return/Direction Classifier (FreqAI LightGBM/XGBoost) — *Priority 5*

- **Category:** ML
- **Core idea:** Engineer features (multi-horizon returns, vol, RSI/MACD, funding, OI, volume, regime flags) and train a boosted classifier/regressor to predict next-horizon return sign or magnitude; trade the prediction. Directly supported by FreqAI's sliding-window retrain pipeline.
- **Edge exploited:** Nonlinear interactions among features. **WEAK-HYPOTHESIS** — boosted trees readily fit noise; crypto's regime shifts break learned relationships.
- **Data inputs:** Engineered feature matrix from OHLCV + funding/OI + derived indicators.
- **Timeframe:** 1h–1d (lower frequencies = better signal/noise for this).
- **Markets:** Liquid perps/spot.
- **Entry/Exit:** Probability/threshold-based; combine with cost-aware filter so it only trades high-conviction calls.
- **Risk management:** Fixed-fraction risk with model-confidence scaling; mandatory stops; **regular retraining with purged, embargoed CV**.
- **Position sizing:** Confidence-weighted, capped.
- **Execution:** Don't trade marginal predictions (fees eat low-edge calls); avoid acting on near-0.5 probabilities.
- **Backtesting:** Walk-forward with **purged k-fold + embargo** (de Prado) to prevent leakage; FreqAI's sliding retrain helps but you must label without lookahead. Run lookahead-analysis & recursive-analysis. **Triple-barrier labeling** strongly recommended over naive fixed-horizon labels.
- **Key features:** Multi-horizon momentum, realized vol, funding/OI, volume, time-of-day, regime tags.
- **Strengths:** Captures nonlinearity; fast to train; interpretable via feature importance.
- **Weaknesses:** Overfits; non-stationarity; feature importance ≠ stable edge.
- **Failure modes:** Regime change invalidates the model silently; label leakage; degenerate "predict the trend you already see."
- **Overfitting risk:** **High.** **Leakage risk:** **High** (feature/label timing, normalization across the split).
- **Infra:** Moderate (retrain pipeline, feature store). **Difficulty:** Moderate. **Capacity:** Inherits underlying liquidity. **Live robustness:** Low–moderate.
- **Profit 5 / Risk 7 / Priority 5.**

### 4.2 Sequence Models — Transformer / LSTM Forecast (FreqAI PyTorch) — *Priority 4*

- **Category:** ML
- **Core idea:** Sequence model ingests windows of features to forecast returns/vol/direction. FreqAI ships a PyTorch Transformer regressor and MLP.
- **Edge exploited:** Temporal dependencies. **WEAK-HYPOTHESIS** — deep nets are data-hungry and crypto's stationary signal is thin; they overfit spectacularly and are hard to diagnose.
- **Data inputs:** Long, clean, multi-feature sequences.
- **Timeframe:** 1h–1d.
- **Markets:** Most-liquid majors only (data hunger).
- **Entry/Exit:** Threshold on forecast with cost filter.
- **Risk management:** As 4.1, plus strict early-stopping and out-of-sample gating; ensemble/uncertainty estimates to avoid acting on garbage.
- **Position sizing:** Uncertainty-aware, capped.
- **Backtesting:** Same leakage discipline; *also* watch normalization leakage across the window and train/val split. Forward-test before any capital.
- **Strengths:** Can model regime/vol dynamics if data suffices. **Weaknesses:** Data-hungry, opaque, compute-heavy, fragile.
- **Failure modes:** Overfit to a past regime; silent decay; training instability.
- **Overfitting risk:** **Very high.** **Leakage risk:** **High.**
- **Infra:** High (GPU, retrain, monitoring). **Difficulty:** High. **Capacity:** Inherits liquidity. **Live robustness:** Low.
- **Profit 4 / Risk 8 / Priority 4** — *research-only until it beats the boosted-tree baseline out-of-sample, which it usually won't.*

### 4.3 Reinforcement Learning Trading Agent (FreqAI RL) — *Priority 3*

- **Category:** ML
- **Core idea:** An RL agent learns a policy (enter/exit/hold, possibly sizing) to maximize a reward (risk-adjusted PnL). Platform ships 3/4/5-action RL envs on stable-baselines3.
- **Edge exploited:** Learned sequential decision-making. **SPECULATIVE** — RL is brittle, reward-hacking-prone, and extremely sensitive to reward design and non-stationarity. Genuinely useful RL successes in live crypto trading are rare and hard to reproduce.
- **Data inputs:** Feature/state representation, well-shaped reward (must penalize fees, drawdown, turnover).
- **Timeframe:** 1h–1d.
- **Markets:** Liquid majors.
- **Entry/Exit:** Policy-driven.
- **Risk management:** Hard external risk overlay (do NOT trust the policy's risk control); position/leverage caps outside the agent.
- **Position sizing:** Policy or external; cap externally.
- **Backtesting:** Train/validation/test split with regime diversity; beware the agent memorizing the training path. Reward shaping is where most projects fail.
- **Strengths:** Can in principle learn execution + timing jointly. **Weaknesses:** Reward hacking, instability, non-reproducibility, opaque failure.
- **Failure modes:** Policy collapses out-of-regime; overfits training path; degenerate hold-forever or churn policies.
- **Overfitting risk:** **Very high.** **Leakage risk:** **High** (env/reward can leak future info trivially).
- **Infra:** Very high. **Difficulty:** Very high. **Capacity:** Inherits liquidity. **Live robustness:** Very low (today).
- **Profit 4 / Risk 9 / Priority 3** — *sandbox/R&D only; not a near-term capital allocation.*

### 4.4 Anomaly Detection / Clustering Regime Detection (standalone) — *Priority 3*

- **Category:** ML
- **Core idea:** Unsupervised models (clustering, HMM, autoencoders) detect regimes or anomalies; trade off regime labels directly.
- **Edge exploited:** Regime structure. **WEAK as a *primary* strategy** — regime labels don't tell you *what to do*; they're an input, not a strategy.
- **Verdict:** **This belongs in Hybrid (§5.2), not as a standalone money-maker.** As a primary engine: Profit 3 / Risk 6 / Priority 3. Its real value is feeding rule-based books in a hybrid.

---

## 5. Hybrid Strategy Candidates

> Hybrids are where ML *should* live in crypto: bounded, supervised by deterministic logic, with clear fallbacks. These dominate the realistic risk-adjusted opportunity set after the structural rule-based plays.

### 5.1 Meta-Labeling Filter on Rule Signals — *Priority 8*

- **Category:** Hybrid (de Prado meta-labeling)
- **Core idea:** A robust rule-based strategy (e.g., trend or breakout) generates the *primary side* (long/short). A secondary ML model predicts **P(this specific signal is profitable)** and decides *whether to take it and how big* — it never decides direction.
- **Edge exploited:** ML improves *precision* (filters low-quality signals) while the rule provides the robust, interpretable directional edge. **MODERATE-HYPOTHESIS**, but the structure is sound and de-risks ML by construction.
- **Data inputs:** Primary-signal events + features at signal time (vol, regime, funding, volume, recent performance).
- **Timeframe:** Inherits the primary (4h–1d typical).
- **Markets:** Perp/spot, portfolio.
- **Entry:** Take the rule's trade only if meta-model P(win) > cost-aware threshold; size by P(win).
- **Exit:** Primary strategy's exit governs; meta-model can also trigger early skip-next.
- **Risk management:** Falls back to the *raw rule* if the model degrades — a built-in safety net. Standard stops/vol-targeting on top.
- **Position sizing:** Confidence-scaled within rule limits.
- **Execution:** Lower turnover than raw rule (filters bad trades) → fee-friendly.
- **Backtesting:** Label = did the primary trade hit its profit barrier (triple-barrier)? **Purged/embargoed CV mandatory.** Compare against the raw-rule baseline — the meta-model must beat *that*, not a buy-and-hold.
- **Key features:** Regime, vol, funding, signal-context features.
- **Strengths:** Bounds ML risk; interpretable; graceful degradation; improvable.
- **Weaknesses:** Only as good as the primary; can over-filter and starve the strategy.
- **Failure modes:** Meta-model overfits and filters out the *good* trades; regime shift breaks both layers.
- **Overfitting risk:** Moderate (lower than standalone ML). **Leakage risk:** Moderate (label/feature timing) — control rigorously.
- **Infra:** Moderate. **Difficulty:** Moderate. **Capacity:** Inherits primary. **Live robustness:** Moderate–high (fallback).
- **Profit 6 / Risk 5 / Priority 8** — *the best ML-in-the-loop bet on this platform.*

### 5.2 Regime-Switching Allocator (ML regime → rule book) — *Priority 7*

- **Category:** Hybrid
- **Core idea:** An ML/statistical regime classifier (trend vs. range vs. high-vol crisis — HMM, clustering, or supervised on vol/breadth/funding features) **routes capital** to the rule-based sub-strategy best suited to that regime (trend book in trends, mean-reversion in ranges, flat/carry in crisis).
- **Edge exploited:** Different edges work in different regimes; matching strategy to regime raises risk-adjusted return. **MODERATE-HYPOTHESIS** — regime detection is noisy and *lags*, but coarse regimes (calm vs. crisis) are detectable enough to be useful for *risk*, even if not for precise timing.
- **Data inputs:** Vol, correlation/breadth, funding, OI, trend strength; per-pair OHLCV.
- **Timeframe:** Regime updates daily/4h; sub-strategies on their own cadence.
- **Markets:** Portfolio-level across perp/spot.
- **Entry/Exit:** Delegated to the active sub-strategy; allocator sets weights/gross exposure by regime.
- **Risk management:** Use regime primarily to **cut gross exposure in detected crisis/high-vol** — the most reliable use of regime models. Hysteresis to avoid whipsawing between regimes.
- **Position sizing:** Regime-scaled gross + per-strategy vol-targeting.
- **Backtesting:** Point-in-time regime labels only (no future info). Validate the *router* adds value over a static blend. Recursive-analysis to catch label lookahead.
- **Strengths:** Smooths equity curve; reduces tail exposure; combines diversifying books. **Weaknesses:** Regime lag; misclassification at turning points (the worst moments).
- **Failure modes:** Whipsaw between regimes; late crisis detection; overfit regime boundaries.
- **Overfitting risk:** Moderate–high (boundary tuning). **Leakage risk:** Moderate.
- **Infra:** Moderate. **Difficulty:** Moderate. **Capacity:** Good (portfolio). **Live robustness:** Moderate–high if used for *risk* not *timing*.
- **Profit 6 / Risk 5 / Priority 7.**

### 5.3 ML Position Sizing / Risk Overlay — *Priority 6*

- **Category:** Hybrid
- **Core idea:** Keep deterministic entry/exit logic; let ML predict **forward volatility / drawdown probability / signal reliability** and modulate **size and gross exposure** accordingly. ML never picks direction or entries.
- **Edge exploited:** Better risk forecasting → better risk-adjusted returns. **MODERATE-HYPOTHESIS** — vol is more forecastable than direction (vol clusters), so this is among the safer ML uses.
- **Data inputs:** Realized/implied vol proxies, funding/OI, regime features, recent strategy performance.
- **Timeframe:** Matches host strategy.
- **Markets:** Any; portfolio-level.
- **Risk management:** ML output is *bounded* (size multiplier within hard caps); deterministic floors/ceilings always enforced.
- **Position sizing:** Vol-forecast-driven inverse-vol scaling, clamped.
- **Backtesting:** Compare risk-adjusted metrics vs. fixed/inverse-vol sizing baseline. Leakage control on vol features.
- **Strengths:** Safe ML use (vol is predictable-ish); improves Sharpe without betting on direction. **Weaknesses:** Marginal if naive inverse-vol already captures most of it.
- **Failure modes:** Vol-forecast breaks in regime jumps (exactly when you need it).
- **Overfitting risk:** Moderate. **Leakage risk:** Moderate.
- **Infra:** Moderate. **Difficulty:** Moderate. **Capacity:** Good. **Live robustness:** Moderate–high.
- **Profit 5 / Risk 4 / Priority 6.**

---

## 6. Top 5 Most Promising Strategies

1. **Perp-Spot Funding/Basis Carry (Rule-Based).** Structural edge, reliable data, market-neutral, high capacity, strong automation fit. The whole risk is in the tails — engineer margin safety obsessively. *Build first.*

2. **Time-Series + Cross-Sectional Momentum (Rule-Based).** The most robust directional edge; crisis-alpha; low overfit risk; great platform fit. *Build in parallel with #1 as the directional core.*

3. **Funding-Rate Harvesting, diversified (Rule-Based).** Productized #1 across many pairs; turns carry into a yield portfolio. *Build as the scaled version of #1.*

4. **Meta-Labeling Filter on Rule Signals (Hybrid).** The single best way to introduce ML without betting the firm on it — bounds ML to a precision-improving filter over #2 with a rule-based fallback. *Build once #1–#3 are live and generating clean signal logs to label.*

5. **Regime-Switching Allocator (Hybrid).** Ties the book together; primarily a *risk* tool (cut gross in crisis) that also routes between trend/mean-reversion/carry. *Build as the portfolio layer over the above.*

> **Notably absent from the top 5:** every pure-ML candidate. That is the central message — in crypto, ML's realistic, robust value is as a *filter, sizer, and regime-router*, not as a price oracle. Earn the right to do standalone ML prediction *after* the hybrids prove your data, labeling, and leakage discipline.

---

## 7. Recommended Research Roadmap

**Phase 0 — Foundations (weeks 0–4).**
- Build a *single source of truth* data layer: survivorship-bias-free universe, point-in-time funding/OI, fees per tier, realistic slippage model. **This is the highest-leverage work in the entire program.**
- Stand up the validation harness: walk-forward, purged+embargoed CV, and wire in the platform's `lookahead-analysis` and `recursive-analysis` on *every* strategy by default.
- Define cost model (maker/taker, funding, borrow, slippage curve vs. order size).

**Phase 1 — Structural rule-based core (weeks 4–10).**
- Implement & paper-trade #1 Funding/Basis Carry and #2 Momentum.
- Forward-test for ≥4–6 weeks before any capital; compare live vs. backtest slippage/fills.

**Phase 2 — Diversify the rule book (weeks 10–16).**
- Add #3 diversified funding harvesting, #4 pairs/cointegration (with strict rolling re-estimation), #6 breakout. Measure cross-strategy correlation.

**Phase 3 — Hybrid ML layer (weeks 16–26).**
- Meta-labeling (#5.1) on the now-clean signal logs, then ML sizing (#5.3), then regime allocator (#5.2). Each must beat its rule-only baseline out-of-sample.

**Phase 4 — Sandbox ML prediction (ongoing, ring-fenced).**
- FreqAI boosted-tree classifier (#4.1) as a research baseline; sequence models (#4.2) and RL (#4.3) strictly in sandbox, never sized until they beat the baseline live.

**Gate between every phase:** out-of-sample + forward-test performance, leakage audit passed, and live-vs-backtest slippage within tolerance. No promotion to capital without all three.

**Parallel track (documented, not yet built) — Latency & venue expansion.** Available infra: a London VPS (and budget for more). London is co-located with **Deribit** (options) and well-placed for EU/US-East venues, but ~230ms from Binance/Bybit/OKX (Asia) — those would need a Tokyo/Singapore box. Treat the latency family in **§14** as a researched, costed future option, not current work. Decision rule to *start* this track: only after the §7 core is live and profitable, and only once a specific latency edge has been validated in Python forward-testing. See §14.

---

## 8. Data Requirements

| Data | Used by | Reliability concern |
|------|---------|---------------------|
| OHLCV (multi-timeframe, multi-pair) | All | Gaps, wash-traded volume on low-tier venues |
| **Point-in-time funding rate history** | Carry, harvesting, features | *Weakest link* — sparse/incorrect history is common; apply at correct timestamp |
| Basis / index vs. mark | Carry | Reconstruction from spot+perp needs care |
| Open interest | Carry, regime, features | Venue-reported, sometimes inconsistent |
| Fee tiers / borrow rates | All (cost model) | Account-specific; model your *actual* tier |
| L2 order book / tick / CVD | Order-flow, arb | Heavy storage; spoofing/wash pollution |
| Cross-venue books | Arbitrage | Latency-stamped; transfer/withdrawal constraints |
| Delisted/depegged asset history | Universe construction | **Survivorship bias** if missing |

**Non-negotiables:** point-in-time correctness, realistic fees/slippage, survivorship-bias-free universe, correct funding timestamps. Garbage in → confidently wrong out.

---

## 9. Backtesting Framework Requirements

- **Walk-forward** evaluation as default; no single-split in-sample reporting.
- **Purged k-fold CV + embargo** (de Prado) for all ML/hybrid labeling to kill leakage from overlapping labels.
- **Triple-barrier labeling** for any supervised return/direction target.
- **Realistic cost model:** maker/taker fees at *your* tier, funding paid at correct snapshots, slippage as a function of order size vs. book depth, partial fills.
- **Futures realism:** margin, maintenance-margin liquidation, funding accrual.
- **Mandatory leakage audits:** run the platform's `lookahead-analysis` and `recursive-analysis` on every strategy before promotion; recursive-analysis specifically catches indicators whose past values change as new data arrives (a classic silent leak).
- **Baselines:** every strategy must beat (a) buy-and-hold BTC, (b) its own rule-only version (for hybrids), and (c) a random-entry/same-exit control.
- **Multiple-testing discipline:** track how many configs you tried; deflate Sharpe accordingly. Reserve a final hold-out you touch *once*.
- **Forward (paper) test** of ≥4–6 weeks before capital, comparing realized vs. modeled slippage/fills.

---

## 10. Live Execution Requirements

- **Order management:** limit/maker-preference where viable to cut fees; smart re-pricing; partial-fill handling; multi-leg (carry/pairs) near-simultaneous execution to minimize leg risk.
- **Latency & connectivity:** resilient websocket + REST failover; reconnect/replay logic; clock-sync.
- **Margin & liquidation monitoring:** real-time maintenance-margin distance, auto-deleverage on threshold (critical for carry/short legs).
- **Kill-switches:** halt on stale data, abnormal slippage, drawdown limit, API error storms, or detected exchange outage.
- **Idempotency & reconciliation:** every order tied to a client ID; periodic state reconciliation against exchange (prevents double-fills after disconnects).
- **Rate-limit & API-key hygiene:** respect limits; least-privilege keys (no withdrawal perms); secrets management.
- **Crypto-specific guards:** funding-spike handling, de-peg detection on stablecoin collateral, withdrawal-halt awareness, exchange-maintenance calendars.

---

## 11. Risk Management Framework

- **Volatility targeting** at position and portfolio level; **inverse-vol** sizing as the default prior.
- **Correlation control:** crypto is largely one BTC-beta factor — cap *net* exposure and gross during high-correlation regimes.
- **Hard limits:** per-trade risk fraction, per-pair, per-venue, per-strategy, and total gross/net leverage caps — enforced *outside* any ML model.
- **Drawdown governance:** strategy-level and portfolio-level drawdown circuit breakers that de-size or halt.
- **Tail protection:** stress scenarios for liquidation cascades, funding spikes, de-pegs, and venue outages; pre-decide responses.
- **Counterparty/venue diversification:** cap exposure per exchange; assume any single venue can freeze withdrawals.
- **No naked mean-reversion:** stops mandatory on negatively-skewed strategies.
- **Fee/funding budget:** monitor cumulative cost drag vs. gross alpha per strategy.

---

## 12. Monitoring and Model Governance

- **Live-vs-backtest tracking:** continuously compare realized slippage, fill rates, and per-strategy PnL against modeled expectations; alert on divergence (your edge-decay early warning).
- **Signal & feature drift:** monitor input feature distributions and prediction distributions for drift; alert when out-of-distribution.
- **Model registry & versioning:** every deployed model versioned with its training window, features, CV scores, and the data snapshot; reproducible retrains.
- **Scheduled retraining with guardrails:** retrain on a cadence (FreqAI sliding window), but **gate promotion** on out-of-sample performance — never auto-promote a worse model.
- **Champion/challenger:** new models shadow-trade before taking capital.
- **Performance attribution:** decompose PnL by strategy, regime, and cost; know *why* you made/lost money.
- **Kill criteria, pre-committed:** define in advance the drawdown / Sharpe-decay / divergence thresholds that retire a strategy — decide before you're emotionally invested.
- **Audit trail:** log every signal, order, fill, and parameter change for post-mortems and compliance.

---

## 13. Final Recommendation

**Build structure before prediction.**

1. **Allocate first capital to funding/basis carry and trend-following.** These are the only families here with a credible claim to a *structural or well-replicated* edge, reliable data, high capacity, and strong automation fit. They will fund the rest of the program.

2. **Make data and leakage control the actual product of Phase 0.** Point-in-time funding, survivorship-bias-free universe, realistic costs, and default `lookahead-analysis`/`recursive-analysis` on everything. Most crypto strategy failures are data/leakage failures wearing a model's clothes.

3. **Introduce ML only in hybrid, bounded roles** — meta-labeling, sizing, regime-routing — each required to beat its rule-only baseline out-of-sample with a rule-based fallback always available.

4. **Ring-fence pure-ML prediction (boosted trees → sequence models → RL) as sandbox R&D.** Do not size it until it beats the rule/hybrid baselines in live forward-testing. Expect most of it not to.

5. **Treat every backtest as guilty until proven innocent.** Forward-test, deflate for multiple testing, and pre-commit kill criteria.

**The blunt version:** In crypto, the reliable money is in *carry and trend with disciplined risk*, and ML's honest job is to *filter and size*, not to predict. A team that nails data integrity, execution, and risk on three rule-based strategies will outperform one chasing a transformer that looked brilliant in-sample. Build the boring, robust core first; earn the right to get fancy.

---

*Ratings are research priors to sequence work, not performance claims. No backtest results are asserted in this document. Validate every hypothesis on your own clean, point-in-time data with leakage audits and forward tests before committing capital.*

---

## 14. Infrastructure-Enabled Latency Strategies (Documented Roadmap — Not Yet Built)

**Status:** Parallel future track. The core (§§3–6) remains the priority and is built first. This section exists so the latency option is *researched and costed*, not so it is built now. **Decision rule to start this track:** core book live and profitable **and** a specific latency edge validated in Python forward-testing.

### 14.1 Three reality checks that govern this entire track

1. **Location beats language, by 100×.** Latency is dominated by physical distance to the matching engine, then kernel/network tuning, then exchange rate limits, and only *last* by language. A colocated Python bot beats a Rust bot in the wrong region. **Fix VPS placement before anything else.**

2. **Venue geography (verify before committing — cloud regions change):**

   | Venue | Matching engine | London VPS fit |
   |---|---|---|
   | **Deribit** (options/futures) | AWS London (eu-west) | ✅ co-located, sub-ms achievable |
   | Coinbase | AWS us-east-1 | 🟡 ~80ms |
   | Bitstamp / some EU | EU | ✅ good |
   | Binance (spot/perp) | AWS Tokyo (ap-northeast-1) | ❌ ~230ms — needs a Tokyo VPS |
   | Bybit | AWS Singapore | ❌ needs SG/Tokyo VPS |
   | OKX | HK / Singapore | ❌ needs SG/HK VPS |

   → The London box's natural advantage is **Deribit** and EU/US-East venues. Asia-venue latency requires buying a Tokyo/Singapore box (budgeted, acceptable per stakeholder).

3. **Architecture: two stacks, one data layer — do NOT rewrite freqtrade in Rust.**
   - **Python / freqtrade = the brain:** research, backtesting, hyperopt, FreqAI, leakage tooling, and the entire latency-*insensitive* book (carry, trend, pairs, hybrids).
   - **Separate execution microservice = the fast hands:** only for latency strategies. **Python-first** (asyncio/uvloop, websocket feeds) to validate the edge; **port only the proven hot path to Rust** if microseconds are demonstrably decisive. Rust is a last-5% optimization, never the starting point.
   - Note: freqtrade's `TradingMode` is **spot/margin/futures only — no options.** Any Deribit options strategy needs its own stack regardless.

### 14.2 Conditionally-viable latency candidates (priority *contingent on starting the track*)

- **Volatility Risk Premium harvesting on Deribit (sell options + delta-hedge)** — *the strongest new addition.* Structural edge analogous to funding carry: option buyers persistently overpay for protection; you collect the premium and hedge delta. London-native (low latency to Deribit). **Strengths:** structural, high-capacity, diversifying. **Weaknesses/failure modes:** short-gamma tail risk — a vol spike can wipe months of premium; requires disciplined delta/vega hedging and hard tail limits. **Needs:** options pricing/greeks skill, a non-freqtrade options stack. *Edge confidence: strong-structural, comparable to carry.*

- **Passive market making (spread capture + maker rebates)** — the canonical latency strategy. Viable on mid-tier pairs/venues where HFT pros don't crowd. **Failure modes:** adverse selection, inventory blow-out in trends. Needs colocation + eventual Rust for queue position.

- **Cross-exchange / triangular arbitrage** (upgrade of §3.5) — viable *if* colocated at the execution venue, listening to the price-leading venue (often Binance). Mid-frequency, thin margins, inventory pre-funding, transfer/withdrawal risk.

- **Liquidation-cascade fading** — fade forced-seller overshoot on perps; latency helps capture the wick and exit. High risk (catching a falling knife); needs a liquidation feed and hard stops. *Partly expressible in freqtrade for the slower variant.*

- **Funding-settlement scalping / basis-convergence** — predictable flows around funding snapshots; faster execution captures convergence. Mid-frequency; perp-based.

### 14.3 Honest caveat

True microsecond HFT (queue-position market making, latency arb on majors) means competing with Wintermute/GSR/Jump on their turf — realistic small-team edge lives in **mid-frequency (seconds) plays and less-contested venues/pairs**, plus the **structural Deribit VRP** edge, not microsecond wars on BTC. Budget the colocation and a Tokyo box as the real prerequisites; treat Rust as an optimization you earn into.
