"""Engine for #2 - the single orchestration loop shared by replay and live.

Per 8h step: accrue P&L on the held book, run the RiskMonitor (obey HALT), and on
a rebalance tick build the target book and trade the whole book toward it. Single
source of truth for decisions, identical in replay and live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .broker import Broker
from .config import TrendConfig
from .feed import DataFeed
from .risk import HALT, RiskMonitor
from .signals import build_target


@dataclass
class RunResult:
    equity_curve: list = field(default_factory=list)
    risk_events: list = field(default_factory=list)
    broker: Broker = None


class Engine:
    def __init__(self, cfg: TrendConfig, feed: DataFeed, broker: Broker,
                 risk: RiskMonitor):
        self.cfg, self.feed, self.broker, self.risk = cfg, feed, broker, risk

    def run(self) -> RunResult:
        res = RunResult(broker=self.broker)
        for step in self.feed:
            self.broker.accrue(step["accrual"])

            for e in self.risk.check(step, self.broker):
                res.risk_events.append((step["t"], e))
                if e.level == HALT and e.kind == "drawdown":
                    self.broker.set_book({}, step["prices"])     # flatten

            if step["is_rebalance"] and not self.risk.halted:
                target = build_target(step["sig"], step["vol"],
                                      self.broker.equity(), self.cfg)
                if self.risk.block_entries:                      # de-risk only
                    target = {p: n for p, n in target.items()
                              if abs(n) <= abs(self.broker.positions().get(p, 0.0))}
                self.broker.set_book(target, step["prices"])

            pos = self.broker.positions()
            res.equity_curve.append((
                step["t"], self.broker.equity(), len(pos),
                sum(abs(n) for n in pos.values()) / max(self.broker.equity(), 1e-9),
                sum(pos.values()) / max(self.broker.equity(), 1e-9),
            ))
        return res
