"""Persist / restore the paper-live book between scheduled forward-test runs.

A forward test runs as a short process once per 8h funding settlement (cron, a
freqtrade loop, etc.). Each run must continue the SAME book, so the broker's
positions + P&L trackers and the RiskMonitor's peak/flags are snapshotted to JSON
and reloaded next time. This is the dry-run analogue of the exchange being the
source of truth in live (where LiveBroker.reconcile() plays this role instead).
"""
from __future__ import annotations

import json
import os

from .broker import DryRunLiveBroker, Position
from .risk import RiskMonitor


def save_state(broker: DryRunLiveBroker, risk: RiskMonitor, path: str):
    state = {
        "equity": broker._equity,
        "positions": {p: vars(pos) for p, pos in broker._pos.items()},
        "trackers": {
            "gross_funding": broker.gross_funding, "basis": broker.basis,
            "borrow": broker.borrow, "costs": broker.costs,
            "n_trades": broker.n_trades, "seq": broker._seq,
        },
        "risk": {
            "peak_equity": risk.peak_equity, "halted": risk.halted,
            "block_entries": risk.block_entries, "counts": risk.counts,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)            # atomic write - never leave a half-file


def load_state(broker: DryRunLiveBroker, risk: RiskMonitor, path: str) -> bool:
    """Restore in place. Returns False if there is no prior state (cold start)."""
    if not os.path.exists(path):
        return False
    s = json.load(open(path))
    broker._equity = s["equity"]
    broker._pos = {p: Position(**v) for p, v in s["positions"].items()}
    t = s["trackers"]
    broker.gross_funding, broker.basis = t["gross_funding"], t["basis"]
    broker.borrow, broker.costs = t["borrow"], t["costs"]
    broker.n_trades, broker._seq = t["n_trades"], t["seq"]
    r = s["risk"]
    risk.peak_equity, risk.halted = r["peak_equity"], r["halted"]
    risk.block_entries, risk.counts = r["block_entries"], r["counts"]
    return True
