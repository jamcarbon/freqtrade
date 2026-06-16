"""RiskMonitor - the guardrails that live OUTSIDE the signal (Step B).

Runs every 8h tick (continuously in live). It can do three things, escalating:
  INFO  - log something noteworthy (e.g. a funding spike).
  WARN  - de-risk: rehedge delta, or block NEW entries.
  HALT  - kill-switch: stop trading and flatten / auto-deleverage.

The central design point from Step B: a delta-neutral book on CROSS/portfolio
margin stays solvent through huge price moves because the spot leg's gain offsets
the perp leg's loss in one collateral pool - so the auto-deleverage trigger fires
only if the HEDGE breaks (delta drift) or equity actually draws down, never just
because a name spiked. That is exactly why cross margin + low leverage is required.
"""
from __future__ import annotations

from dataclasses import dataclass

from .broker import Broker
from .config import StrategyConfig

INFO, WARN, HALT = "INFO", "WARN", "HALT"


@dataclass
class RiskEvent:
    level: str
    kind: str
    detail: str


class RiskMonitor:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.peak_equity = cfg.equity0
        self.halted = False
        self.block_entries = False
        self.counts: dict[str, int] = {}

    def _bump(self, kind: str):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def check(self, step: dict, broker: Broker) -> list[RiskEvent]:
        cfg = self.cfg
        eq = broker.equity()
        pos = broker.positions()
        prices = step["prices"]
        ev: list[RiskEvent] = []
        self.peak_equity = max(self.peak_equity, eq)

        # 1) portfolio drawdown kill-switch
        dd = eq / self.peak_equity - 1.0
        if dd <= -cfg.max_drawdown_kill:
            self.halted = True
            ev.append(RiskEvent(HALT, "drawdown",
                                f"dd {dd:.1%} <= -{cfg.max_drawdown_kill:.0%}; flatten"))

        # 2) funding-spike flag (observable, real squeeze early-warning)
        sig = step.get("sig")
        if sig is not None:
            for p, pos_ in pos.items():
                f8 = abs(sig.get(p, 0.0)) / cfg.fund_per_year  # back to per-8h
                if f8 > cfg.funding_spike_flag:
                    self._bump("funding_spike")
                    ev.append(RiskEvent(INFO, "funding_spike",
                                        f"{p} 8h funding {f8:.2%} on held position"))

        # 3) net-delta band: a hedged book must stay ~delta-neutral
        gross = sum(pp.notional for pp in pos.values())
        net_delta = 0.0
        for p, pp in pos.items():
            if p not in prices or pp.entry_spot <= 0 or pp.entry_perp <= 0:
                continue
            spot, perp = prices[p]
            # long-spot+short-perp drift (or the mirror); residual dollar delta
            leg = pp.notional * (spot / pp.entry_spot - perp / pp.entry_perp)
            net_delta += pp.side * leg
        if gross > 0 and abs(net_delta) / eq > cfg.delta_band:
            self._bump("delta_breach")
            ev.append(RiskEvent(WARN, "delta",
                                f"net delta {net_delta/eq:+.2%} of equity > band; rehedge"))

        # 4) cross-margin solvency / auto-deleverage
        maint_req = gross * cfg.maint_margin
        if gross > 0:
            margin_distance = (eq - maint_req) / gross
            if margin_distance < cfg.auto_dlv_margin_distance:
                self.block_entries = True
                ev.append(RiskEvent(HALT, "solvency",
                                    f"margin distance {margin_distance:.1%} < "
                                    f"{cfg.auto_dlv_margin_distance:.0%}; deleverage"))
        return ev
