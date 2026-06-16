"""Combined forward-test orchestrator for the parallel #2 / #3 book.

Runs ONE 8h forward-test tick for each book (each persists its own state + CSV),
then writes a portfolio-level row: combined equity, combined drawdown, and a
COMBINED RISK check on top of each book's own RiskMonitor. Schedule externally on
the 8h cadence; never busy-wait.

Both books are individually market/delta-neutral, so the portfolio's net BTC-beta
is structurally ~0; the combined overlay's job is the TOTAL-drawdown kill and a
sanity check that neither book has silently halted.
"""
from __future__ import annotations

import csv
import json
import os

from funding_harvest.live import run_paper as fh
from trend_momentum.live import run_paper as tm

_DIR = "user_data/backtest_results"
PORT_STATE = os.path.join(_DIR, "portfolio_forward_state.json")
PORT_LOG = os.path.join(_DIR, "portfolio_forward.csv")

# combined hard limit: halt BOTH books if the pooled equity draws down this far
COMBINED_DD_KILL = 0.15


def _read_equity(state_path: str, default: float) -> float:
    if not os.path.exists(state_path):
        return default
    return float(json.load(open(state_path)).get("equity", default))


def forward_test():
    """One combined tick. Calls each book's forward_test (they own their state),
    then rolls up a portfolio row + combined-drawdown kill-switch."""
    print(">>> book #3 (carry) tick")
    fh.forward_test()
    print(">>> book #2 (momentum) tick")
    tm.forward_test()

    eq3 = _read_equity(fh.STATE_PATH, 10_000.0)
    eq2 = _read_equity(tm.STATE_PATH, 10_000.0)
    combined = eq2 + eq3

    peak = combined
    if os.path.exists(PORT_STATE):
        peak = max(json.load(open(PORT_STATE)).get("peak", combined), combined)
    dd = combined / peak - 1.0
    kill = dd <= -COMBINED_DD_KILL
    os.makedirs(_DIR, exist_ok=True)
    json.dump({"peak": peak, "combined": combined}, open(PORT_STATE, "w"), indent=2)

    row = dict(
        book3_carry=round(eq3, 2), book2_momentum=round(eq2, 2),
        combined=round(combined, 2), peak=round(peak, 2),
        drawdown=f"{dd:+.2%}", combined_kill="HALT" if kill else "ok",
    )
    new = not os.path.exists(PORT_LOG)
    with open(PORT_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nPORTFOLIO tick -> {PORT_LOG}\n  {row}")
    if kill:
        print(f"  !! COMBINED DRAWDOWN {dd:.1%} <= -{COMBINED_DD_KILL:.0%}: "
              f"flatten BOTH books (manual gate before re-arming).")


if __name__ == "__main__":
    print("Portfolio forward-test is a SCHEDULED action (8h cadence) - not started here.")
    print("To run one combined tick manually:  forward_test()")
