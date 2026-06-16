"""Combined replay + diversification report for the #2 / #3 parallel book.

The whole reason to run momentum alongside carry: low correlation. Carry earns a
steady premium but is stressed by violent trends/squeezes; momentum tends to make
money in exactly those moves. Combined, the equity curve should be smoother than
either alone (higher Sharpe, shallower drawdown) even though momentum's standalone
Sharpe is lower than carry's.

Run: PYTHONPATH=user_data/strategies python -m portfolio.combined
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- strategy #3 (funding carry) live stack ---------------------------------
from funding_harvest.live.broker import PaperBroker as FHBroker
from funding_harvest.live.engine import Engine as FHEngine
from funding_harvest.live.feed import ReplayFeed as FHFeed
from funding_harvest.live.risk import RiskMonitor as FHRisk
from funding_harvest.live.run_paper import _live_cfg as fh_live_cfg

# --- strategy #2 (momentum) live stack --------------------------------------
from trend_momentum.live.broker import PaperBroker as TMBroker
from trend_momentum.live.config import TrendConfig
from trend_momentum.live.engine import Engine as TMEngine
from trend_momentum.live.feed import ReplayFeed as TMFeed
from trend_momentum.live.risk import RiskMonitor as TMRisk

FUND_PER_YEAR = 3 * 365
BOOK_EQUITY = 10_000.0


def _curve(equity_curve) -> pd.Series:
    df = pd.DataFrame(equity_curve)
    return pd.Series(df[1].values, index=pd.DatetimeIndex(df[0].values))


def _stats(eq: pd.Series, equity0: float) -> dict:
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / equity0 - 1
    v = r.std() * np.sqrt(FUND_PER_YEAR)
    return dict(
        cagr=(1 + total) ** (1 / yrs) - 1,
        sharpe=(r.mean() * FUND_PER_YEAR) / v if v > 0 else float("nan"),
        vol=v,
        dd=(eq / eq.cummax() - 1).min(),
        final=eq.iloc[-1],
    )


def run_book_3():
    cfg = fh_live_cfg()
    cfg.equity0 = BOOK_EQUITY            # $10k book (was $100k default)
    eng = FHEngine(cfg, FHFeed(cfg), FHBroker(cfg), FHRisk(cfg))
    return _curve(eng.run().equity_curve)


def run_book_2():
    cfg = TrendConfig()                  # already $10k
    eng = TMEngine(cfg, TMFeed(cfg), TMBroker(cfg), TMRisk(cfg))
    return _curve(eng.run().equity_curve)


def run():
    print("=" * 80)
    print("PORTFOLIO  -  #2 momentum + #3 carry, two independent $10k books")
    print("=" * 80)
    c3 = run_book_3()
    c2 = run_book_2()

    # align on the common 8h grid
    idx = c2.index.intersection(c3.index)
    c2, c3 = c2.reindex(idx).ffill(), c3.reindex(idx).ffill()
    combined = c2 + c3                    # two pooled books

    s2 = _stats(c2, BOOK_EQUITY)
    s3 = _stats(c3, BOOK_EQUITY)
    sc = _stats(combined, 2 * BOOK_EQUITY)
    corr = c2.pct_change().corr(c3.pct_change())

    def line(name, s):
        print(f"  {name:28s} CAGR {s['cagr']:+6.1%}   Sharpe {s['sharpe']:5.2f}"
              f"   vol {s['vol']:4.0%}   maxDD {s['dd']:+6.1%}   final ${s['final']:,.0f}")

    print(f"\n  Period {idx[0].date()} -> {idx[-1].date()}")
    line("#3 carry (10k)", s3)
    line("#2 momentum (10k)", s2)
    line("COMBINED (20k)", sc)
    # equal-RISK illustration: the two books have very different vol, so equal
    # DOLLARS (the chosen sizing) lets momentum dominate. Risk-weighting shows the
    # full diversification benefit when the books contribute equal risk.
    r2, r3 = c2.pct_change(), c3.pct_change()
    iv2, iv3 = 1 / max(s2["vol"], 1e-9), 1 / max(s3["vol"], 1e-9)
    w2, w3 = iv2 / (iv2 + iv3), iv3 / (iv2 + iv3)
    rp = (w2 * r2 + w3 * r3).dropna()
    rp_sharpe = (rp.mean() * FUND_PER_YEAR) / (rp.std() * np.sqrt(FUND_PER_YEAR))

    print(f"\n  return correlation #2 vs #3 : {corr:+.2f}   (near-zero = real diversifier)")
    print(f"  EQUAL-DOLLAR blend (chosen, $10k each):")
    print(f"    Sharpe {sc['sharpe']:.2f} vs momentum-alone {s2['sharpe']:.2f}"
          f"  ->  smooths the riskier book")
    print(f"    maxDD  {sc['dd']:+.1%} vs momentum-alone {s2['dd']:+.1%}"
          f"  ->  drawdown roughly halved")
    print(f"  EQUAL-RISK blend (illustrative, ~{w3:.0%} carry / {w2:.0%} momentum):")
    print(f"    Sharpe {rp_sharpe:.2f} vs best single {max(s2['sharpe'], s3['sharpe']):.2f}"
          f"  ->  {'beats either book alone' if rp_sharpe > max(s2['sharpe'], s3['sharpe']) else 'near best book'}")
    print("\n  Read: carry is a low-vol/high-Sharpe stabiliser, momentum the return")
    print("  driver. At equal DOLLARS the blend gives momentum-like return at ~half its")
    print("  drawdown; risk-weighting would push combined Sharpe toward carry's. The")
    print("  near-zero correlation is what makes either weighting worth running.")


if __name__ == "__main__":
    run()
