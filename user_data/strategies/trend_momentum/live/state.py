"""Persist / restore the #2 paper book between scheduled forward-test runs."""
from __future__ import annotations

import json
import os

from .broker import DryRunLiveBroker
from .risk import RiskMonitor


def save_state(broker: DryRunLiveBroker, risk: RiskMonitor, path: str):
    state = {
        "equity": broker._equity,
        "positions": broker._pos,
        "trackers": {"price_pnl": broker.price_pnl, "funding": broker.funding,
                     "costs": broker.costs, "turnover": broker.turnover,
                     "seq": broker._seq},
        "risk": {"peak_equity": risk.peak_equity, "halted": risk.halted,
                 "block_entries": risk.block_entries, "counts": risk.counts},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_state(broker: DryRunLiveBroker, risk: RiskMonitor, path: str) -> bool:
    if not os.path.exists(path):
        return False
    s = json.load(open(path))
    broker._equity = s["equity"]
    broker._pos = {p: float(n) for p, n in s["positions"].items()}
    t = s["trackers"]
    broker.price_pnl, broker.funding = t["price_pnl"], t["funding"]
    broker.costs, broker.turnover, broker._seq = t["costs"], t["turnover"], t["seq"]
    r = s["risk"]
    risk.peak_equity, risk.halted = r["peak_equity"], r["halted"]
    risk.block_entries, risk.counts = r["block_entries"], r["counts"]
    return True
