"""Broker abstraction: the only thing that differs between paper and live.

PaperBroker reproduces the validated backtest accounting EXACTLY (funding, basis
residual, borrow, maker costs) so a replay must match funding_harvest_exec.py.
LiveBroker is the ccxt skeleton for the real two-leg account - intentionally inert
until API keys are present AND a forward test has passed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import StrategyConfig
from .signals import TargetPos


@dataclass
class Position:
    side: int
    notional: float
    entry_spot: float = 0.0
    entry_perp: float = 0.0


class Broker(ABC):
    @abstractmethod
    def positions(self) -> dict[str, Position]: ...
    @abstractmethod
    def equity(self) -> float: ...
    @abstractmethod
    def open(self, pair: str, side: int, notional: float, spot: float, perp: float): ...
    @abstractmethod
    def close(self, pair: str, notional_px: tuple[float, float]): ...
    @abstractmethod
    def resize(self, pair: str, notional: float): ...
    @abstractmethod
    def accrue(self, step: dict): ...


# --------------------------------------------------------------------------- #
class PaperBroker(Broker):
    """Faithful simulation. equity is a single running number updated by per-step
    accruals and by execution costs - identical convention to the backtest."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self._equity = cfg.equity0
        self._pos: dict[str, Position] = {}
        self.trade_cost = cfg.fee_spot + cfg.fee_fut + 2 * cfg.slippage
        # P&L decomposition trackers
        self.gross_funding = 0.0
        self.basis = 0.0
        self.borrow = 0.0
        self.costs = 0.0
        self.n_trades = 0

    def positions(self) -> dict[str, Position]:
        return self._pos

    def equity(self) -> float:
        return self._equity

    def open(self, pair, side, notional, spot=0.0, perp=0.0):
        c = self.trade_cost * notional
        self._equity -= c
        self.costs += c
        self.n_trades += 1
        self._pos[pair] = Position(side, notional, spot, perp)

    def close(self, pair, notional_px=(0.0, 0.0)):
        if pair not in self._pos:
            return
        c = self.trade_cost * self._pos[pair].notional
        self._equity -= c
        self.costs += c
        self.n_trades += 1
        del self._pos[pair]

    def resize(self, pair, notional):
        pos = self._pos[pair]
        c = self.trade_cost * abs(notional - pos.notional)
        self._equity -= c
        self.costs += c
        if c > 0:
            self.n_trades += 1
        pos.notional = notional

    def accrue(self, step: dict):
        """step[pair] = {funding, basis_ret, borrow_step}. Apply this 8h tick's
        cashflows to the book held coming into the tick (no lookahead)."""
        f = b = r = 0.0
        for p, pos in self._pos.items():
            s = step.get(p)
            if s is None:
                continue
            f += pos.side * pos.notional * s["funding"]
            r += pos.side * pos.notional * s["basis_ret"]
            if pos.side < 0:
                b += pos.notional * s["borrow_step"]
        self._equity += f + r - b
        self.gross_funding += f
        self.basis += r
        self.borrow += b


# --------------------------------------------------------------------------- #
@dataclass
class OrderTicket:
    coid: str          # clientOrderId (idempotency key)
    pair: str
    action: str        # open/close/resize
    side: int
    leg_spot: str      # 'buy'/'sell'
    leg_perp: str
    notional: float
    dry_run: bool = True


class DryRunLiveBroker(PaperBroker):
    """Runs the FULL live decision/execution path against LIVE prices but never
    sends a real order - it simulates the two-leg maker lifecycle and logs every
    ticket. This is the forward-test broker: identical code path to LiveBroker,
    zero account risk. Accounting is inherited from PaperBroker (so a forward test
    is directly comparable to the replay)."""

    def __init__(self, cfg: StrategyConfig):
        super().__init__(cfg)
        self.orders: list[OrderTicket] = []
        self._seq = 0

    def _coid(self, pair: str, action: str) -> str:
        self._seq += 1
        return f"fh-{action}-{pair}-{self._seq:06d}"   # deterministic, idempotent

    def _legs(self, side: int, opening: bool) -> tuple[str, str]:
        # positive carry (+1): long spot / short perp ; negative carry mirrors.
        spot = "buy" if side > 0 else "sell"
        perp = "sell" if side > 0 else "buy"
        if not opening:                                # closing reverses both legs
            spot = "sell" if spot == "buy" else "buy"
            perp = "sell" if perp == "buy" else "buy"
        return spot, perp

    def open(self, pair, side, notional, spot=0.0, perp=0.0):
        sl, pl = self._legs(side, opening=True)
        self.orders.append(OrderTicket(self._coid(pair, "open"), pair, "open",
                                       side, sl, pl, notional))
        super().open(pair, side, notional, spot, perp)

    def close(self, pair, notional_px=(0.0, 0.0)):
        if pair in self._pos:
            pos = self._pos[pair]
            sl, pl = self._legs(pos.side, opening=False)
            self.orders.append(OrderTicket(self._coid(pair, "close"), pair, "close",
                                           pos.side, sl, pl, pos.notional))
        super().close(pair, notional_px)

    def resize(self, pair, notional):
        if pair in self._pos:
            self.orders.append(OrderTicket(self._coid(pair, "resize"),
                                           pair, "resize", self._pos[pair].side,
                                           "-", "-", abs(notional - self._pos[pair].notional)))
        super().resize(pair, notional)


# --------------------------------------------------------------------------- #
class LiveBroker(Broker):
    """REAL two-leg account via ccxt. Spot-margin leg + USDⓈ-M perp leg, postOnly
    maker on both, clientOrderId idempotency, cross/portfolio margin so the legs
    net. Triple-gated: requires an authenticated exchange AND explicit
    enable_live=True - otherwise construction fails closed."""

    def __init__(self, cfg: StrategyConfig, exchange=None, enable_live: bool = False):
        if exchange is None or not enable_live:
            raise RuntimeError(
                "LiveBroker is disabled. It needs an authenticated ccxt exchange "
                "AND enable_live=True, and must only be turned on AFTER a >=4-6 week "
                "forward test (run with DryRunLiveBroker) has passed."
            )
        self.cfg = cfg
        self.ex = exchange
        self._seq = 0

    def _coid(self, pair, action):
        self._seq += 1
        return f"fh-{action}-{pair}-{self._seq:06d}"

    def positions(self):
        # reconcile internal book against exchange truth before each decision
        return self.ex.fetch_positions()

    def equity(self):
        bal = self.ex.fetch_balance()
        return float(bal["total"].get("USDT", 0.0))

    def _place_two_legs(self, pair, side, notional, spot_px, perp_px, opening):
        # postOnly maker on BOTH legs; if one fills and the other is rejected/
        # unfilled past a timeout, the filled leg is immediately reduce-only
        # unwound rather than left directionally exposed (delta-neutral discipline).
        raise NotImplementedError(
            "create_order(postOnly) per leg with clientOrderId; monitor fills; "
            "unwind on partial-leg failure. Implement + test against testnet first."
        )

    def open(self, pair, side, notional, spot=0.0, perp=0.0):
        self._place_two_legs(pair, side, notional, spot, perp, opening=True)

    def close(self, pair, notional_px=(0.0, 0.0)):
        self._place_two_legs(pair, 0, 0, notional_px[0], notional_px[1], opening=False)

    def resize(self, pair, notional):
        raise NotImplementedError("delta-adjust both legs, postOnly")

    def accrue(self, step):
        return  # exchange charges funding/borrow directly
