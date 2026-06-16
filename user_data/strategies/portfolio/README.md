# Portfolio — running #2 (momentum) and #3 (carry) in parallel

Two **independent $10k books** (combined $20k), each its own validated
one-engine-two-feeds stack. This layer adds the combined view and the
portfolio-level risk overlay. It launches nothing — both books' live brokers stay
gated.

```
portfolio/
├── README.md
├── combined.py     parallel replay + diversification report + equal/risk-weighted view
└── run_paper.py    orchestrates both books' 8h forward-test ticks + combined-DD kill
```

## Run
```bash
PYTHONPATH=user_data/strategies python -m portfolio.combined     # diversification report
# forward test is a scheduled action; one combined tick:
PYTHONPATH=user_data/strategies python -c "from portfolio.run_paper import forward_test; forward_test()"
```

## Why run both — the diversification result

Replay, $10k each, 2022-02 → 2026-06:

| Book | CAGR | Sharpe | vol | maxDD | final |
|------|:---:|:---:|:---:|:---:|:---:|
| #3 carry | +5.2% | **4.53** | 1% | −1.8% | $12.5k |
| #2 momentum | +13.1% | 0.76 | 18% | −19.3% | $17.1k |
| **Combined (20k)** | **+9.4%** | 0.97 | 10% | **−9.0%** | **$29.6k** |

- **Return correlation #2 vs #3 = +0.05** — a genuine diversifier.
- **Equal-dollar blend (the chosen sizing):** combined Sharpe 0.97 > momentum's 0.76,
  and **drawdown roughly halves** (−19% → −9%). You get momentum-like return at far
  less pain, with carry as a low-vol stabiliser.
- **Equal-risk blend (illustrative, ~94% carry / 6% momentum):** Sharpe ~3.65 — risk-
  weighting pushes the blend toward carry's high Sharpe. The near-zero correlation is
  what makes *either* weighting worth running.

> Read: **carry is the low-vol/high-Sharpe stabiliser; momentum is the return driver.**
> At equal dollars you buy momentum's upside at ~half its drawdown. If you later want
> maximum risk-adjusted return, shift dollars toward carry (risk-weighting) — a sizing
> choice, not a code change.

## Portfolio risk overlay
Both books are individually market/delta-neutral, so the portfolio's **net BTC-beta
is structurally ~0**. `run_paper.forward_test()` rolls up both books each 8h and
enforces a **combined-drawdown kill** (15% on pooled equity) on top of each book's own
RiskMonitor (carry: solvency/auto-deleverage; momentum: drawdown/net/gross). It also
flags if either book has silently halted.

## Current status & next steps
- **Ready, not launched.** Both stacks are code-complete and verified in replay; live
  brokers are gated.
- **Next:** testnet-validate each book (momentum first — it's fully testnet-able),
  then schedule `portfolio.run_paper.forward_test()` on the 8h cadence for ≥4–6 weeks,
  compare each book's live-data path to its replay, and only then wire real keys
  (least-privilege, no withdrawal) and start at minimum size.
- **Sizing decision to revisit:** equal-dollar ($10k each) is the current choice;
  risk-weighting toward carry would raise combined Sharpe if that's the goal.

See [`../funding_harvest/README.md`](../funding_harvest/README.md) and
[`../trend_momentum/README.md`](../trend_momentum/README.md) for each book's full story.
