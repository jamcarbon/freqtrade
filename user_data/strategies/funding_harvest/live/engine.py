"""Engine - the single orchestration loop shared by replay and live.

Per 8h step:
  1. accrue the settled cashflows on the book held into the tick (broker).
  2. run the RiskMonitor; obey HALT (flatten / block entries).
  3. on a rebalance tick, build the target book and route desired-vs-held diffs
     to the two-leg executor.

The executor encodes the leg-risk discipline: open/close BOTH legs together; in
live, if one leg fills and the other doesn't, the position is unwound rather than
left directionally exposed (the cardinal sin of a delta-neutral book). In paper
the fill is atomic, so this reduces to the cost-accounting the backtest validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .broker import Broker, Position
from .config import StrategyConfig
from .feed import DataFeed
from .risk import HALT, RiskMonitor
from .signals import TargetPos, build_target


@dataclass
class RunResult:
    equity_curve: list = field(default_factory=list)   # (t, equity, n_pos)
    risk_events: list = field(default_factory=list)    # (t, RiskEvent)
    broker: Broker = None


class Engine:
    def __init__(self, cfg: StrategyConfig, feed: DataFeed, broker: Broker,
                 risk: RiskMonitor):
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.risk = risk

    # ---- two-leg executor --------------------------------------------------
    def _execute(self, desired: dict[str, TargetPos], prices: dict):
        held = self.broker.positions()
        # close positions no longer desired, or whose side flipped
        for p in list(held):
            d = desired.get(p)
            if d is None or d.side != held[p].side:
                self.broker.close(p, prices.get(p, (0.0, 0.0)))
        # open new / re-opened-on-flip; resize held-through only if size changed
        for p, d in desired.items():
            cur = self.broker.positions().get(p)
            spot, perp = prices.get(p, (0.0, 0.0))
            if cur is None:
                if not self.risk.block_entries:        # risk overlay gates entries
                    self.broker.open(p, d.side, d.notional, spot, perp)
            elif abs(d.notional - cur.notional) > 1e-9:
                self.broker.resize(p, d.notional)

    # ---- main loop ---------------------------------------------------------
    def run(self) -> RunResult:
        res = RunResult(broker=self.broker)
        for step in self.feed:
            self.broker.accrue(step["accrual"])

            for e in self.risk.check(step, self.broker):
                res.risk_events.append((step["t"], e))
                if e.level == HALT and e.kind == "drawdown":
                    # hard kill: flatten the book
                    for p in list(self.broker.positions()):
                        self.broker.close(p, step["prices"].get(p, (0.0, 0.0)))

            if step["is_rebalance"] and not self.risk.halted:
                desired = build_target(
                    step["sig"], step["vol"], self.broker.positions(),
                    self.broker.equity(), self.cfg,
                )
                self._execute(desired, step["prices"])

            res.equity_curve.append(
                (step["t"], self.broker.equity(), len(self.broker.positions()))
            )
        return res
