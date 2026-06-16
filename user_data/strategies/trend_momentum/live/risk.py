"""RiskMonitor for #2 - guardrails outside the signal.

Momentum's risk is different from carry's: no liquidation-of-a-hedge concern, but
trend reversals create deep drawdowns and the book must stay market-neutral (net
~ 0) and within a gross cap. Checks: drawdown kill, net-exposure band, gross-cap
breach, (live) stale data.
"""
from __future__ import annotations

from dataclasses import dataclass

from .broker import Broker
from .config import TrendConfig

INFO, WARN, HALT = "INFO", "WARN", "HALT"


@dataclass
class RiskEvent:
    level: str
    kind: str
    detail: str


class RiskMonitor:
    def __init__(self, cfg: TrendConfig):
        self.cfg = cfg
        self.peak_equity = cfg.equity0
        self.halted = False
        self.block_entries = False
        self.counts: dict[str, int] = {}

    def _bump(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1

    def check(self, step: dict, broker: Broker) -> list[RiskEvent]:
        cfg = self.cfg
        eq = broker.equity()
        pos = broker.positions()
        ev: list[RiskEvent] = []
        self.peak_equity = max(self.peak_equity, eq)

        dd = eq / self.peak_equity - 1.0
        if dd <= -cfg.max_drawdown_kill:
            self.halted = True
            ev.append(RiskEvent(HALT, "drawdown",
                                f"dd {dd:.1%} <= -{cfg.max_drawdown_kill:.0%}; flatten"))

        gross = sum(abs(n) for n in pos.values())
        net = sum(pos.values())
        if eq > 0 and gross > 0:
            if abs(net) / eq > cfg.net_band:
                self._bump("net_breach")
                ev.append(RiskEvent(WARN, "net",
                                    f"net {net/eq:+.1%} of equity > band; re-neutralise"))
            if gross / eq > cfg.gross_kill:
                self.block_entries = True
                ev.append(RiskEvent(HALT, "gross",
                                    f"gross {gross/eq:.1f}x > {cfg.gross_kill:.0f}x; de-risk"))
        return ev
