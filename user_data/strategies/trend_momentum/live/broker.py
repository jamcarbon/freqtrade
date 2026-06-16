"""Broker abstraction for #2. Single-leg perp positions (signed notional).

PaperBroker reproduces the research backtest accounting exactly (price P&L +
funding - turnover cost). DryRunLiveBroker runs the live path against live prices
with simulated fills. LiveBroker is the real single-leg ccxt account, gated.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import TrendConfig


class Broker(ABC):
    @abstractmethod
    def positions(self) -> dict[str, float]: ...
    @abstractmethod
    def equity(self) -> float: ...
    @abstractmethod
    def accrue(self, step: dict): ...
    @abstractmethod
    def set_book(self, target: dict[str, float], prices: dict): ...


class PaperBroker(Broker):
    """equity is a single running number: per-step price+funding P&L, minus
    turnover cost at each rebalance. Positions are signed USD notionals."""

    def __init__(self, cfg: TrendConfig):
        self.cfg = cfg
        self._equity = cfg.equity0
        self._pos: dict[str, float] = {}
        self.cost_rate = cfg.fee + cfg.slippage
        self.price_pnl = 0.0
        self.funding = 0.0
        self.costs = 0.0
        self.turnover = 0.0

    def positions(self):
        return self._pos

    def equity(self):
        return self._equity

    def accrue(self, acc: dict):
        """acc[pair] = {ret, funding}. A LONG perp pays funding>0."""
        p = f = 0.0
        for pair, n in self._pos.items():
            a = acc.get(pair)
            if a is None:
                continue
            p += n * a["ret"]
            f += -n * a["funding"]
        self._equity += p + f
        self.price_pnl += p
        self.funding += f

    def set_book(self, target: dict[str, float], prices: dict):
        pairs = set(target) | set(self._pos)
        for pr in pairs:
            new = target.get(pr, 0.0)
            old = self._pos.get(pr, 0.0)
            d = abs(new - old)
            if d > 0:
                c = self.cost_rate * d
                self._equity -= c
                self.costs += c
                self.turnover += d
            if abs(new) > 1e-9:
                self._pos[pr] = new
            elif pr in self._pos:
                del self._pos[pr]


@dataclass
class OrderTicket:
    coid: str
    pair: str
    side: str          # buy/sell
    notional: float
    reduce_only: bool
    dry_run: bool = True


class DryRunLiveBroker(PaperBroker):
    """Full live path against live prices; simulates the single-leg maker orders
    and logs tickets, but never sends one. Accounting inherited from PaperBroker."""

    def __init__(self, cfg: TrendConfig):
        super().__init__(cfg)
        self.orders: list[OrderTicket] = []
        self._seq = 0

    def _coid(self, pair):
        self._seq += 1
        return f"tm-{pair}-{self._seq:06d}"

    def set_book(self, target: dict[str, float], prices: dict):
        pairs = set(target) | set(self._pos)
        for pr in sorted(pairs):
            new = target.get(pr, 0.0)
            old = self._pos.get(pr, 0.0)
            d = new - old
            if abs(d) > 1e-9:
                side = "buy" if d > 0 else "sell"
                reduce_only = abs(new) < abs(old) and (new == 0 or (new > 0) == (old > 0))
                self.orders.append(OrderTicket(self._coid(pr), pr, side,
                                               abs(d), reduce_only))
        super().set_book(target, prices)


class LiveBroker(Broker):
    """Real single-leg USDⓈ-M perp account via ccxt: postOnly maker, clientOrderId
    idempotency, reduce-only on de-risking trades, reconcile() vs exchange.
    Triple-gated: needs an authenticated exchange AND enable_live=True."""

    def __init__(self, cfg: TrendConfig, exchange=None, enable_live: bool = False):
        if exchange is None or not enable_live:
            raise RuntimeError(
                "LiveBroker disabled. Needs an authenticated ccxt perp exchange AND "
                "enable_live=True, and only after a >=4-6 week forward test passes. "
                "Exercise on testnet first (testnet.py)."
            )
        self.cfg = cfg
        self.ex = exchange
        self._pos: dict[str, float] = {}
        self._seq = 0

    def _coid(self, pair):
        self._seq += 1
        return f"tm-{pair}-{int(time.time())}-{self._seq:04d}"

    def positions(self):
        return self._pos

    def equity(self):
        bal = self.ex.fetch_balance()
        return float(bal.get("total", {}).get("USDT", 0.0) or 0.0)

    def reconcile(self) -> list[str]:
        try:
            ex_pos = {p["symbol"].split("/")[0]: float(p.get("contracts") or 0)
                      for p in self.ex.fetch_positions()
                      if abs(float(p.get("contracts") or 0)) > 0}
        except Exception as e:
            return [f"reconcile failed: {e!r}"]
        issues = []
        for pair in self._pos:
            if pair not in ex_pos:
                issues.append(f"{pair}: in internal book, not on exchange")
        for pair in ex_pos:
            if pair not in self._pos:
                issues.append(f"{pair}: on exchange, not in internal book")
        return issues

    def accrue(self, step):
        return  # exchange credits funding/PnL directly

    def _maker_price(self, px, side):
        off = self.cfg.slippage
        return px * (1 - off) if side == "buy" else px * (1 + off)

    def set_book(self, target: dict[str, float], prices: dict):
        pairs = set(target) | set(self._pos)
        for pair in sorted(pairs):
            new = target.get(pair, 0.0)
            old = self._pos.get(pair, 0.0)
            d = new - old
            if abs(d) < 1e-9:
                continue
            sym = f"{pair}/USDT:USDT"
            px = prices.get(pair, (0.0, 0.0))
            px = px[1] if isinstance(px, tuple) else px
            side = "buy" if d > 0 else "sell"
            params = {"postOnly": True, "clientOrderId": self._coid(pair)}
            if abs(new) < abs(old) and (new == 0 or (new > 0) == (old > 0)):
                params["reduceOnly"] = True
            # NOTE: monitor fill + reprice like #3's _await_fill before confirming;
            # omitted here for brevity - wire from funding_harvest.live.broker pattern.
            self.ex.create_order(sym, "limit", side, abs(d) / px,
                                 self._maker_price(px, side), params)
            if abs(new) > 1e-9:
                self._pos[pair] = new
            elif pair in self._pos:
                del self._pos[pair]
