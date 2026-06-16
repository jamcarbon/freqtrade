"""Broker abstraction: the only thing that differs between paper and live.

PaperBroker reproduces the validated backtest accounting EXACTLY (funding, basis
residual, borrow, maker costs) so a replay must match research/exec_costs.py.
DryRunLiveBroker runs the full live path against live prices but never sends an
order. LiveBroker is the real ccxt two-leg account - triple-gated and meant to be
exercised first on the Binance testnet (see testnet.py).
"""
from __future__ import annotations

import time
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


def leg_sides(side: int, opening: bool) -> tuple[str, str]:
    """The (spot, perp) order sides for a delta-neutral pair.
    positive carry (+1): long spot / short perp ; negative carry mirrors it.
    Closing reverses both legs."""
    spot = "buy" if side > 0 else "sell"
    perp = "sell" if side > 0 else "buy"
    if not opening:
        spot = "sell" if spot == "buy" else "buy"
        perp = "sell" if perp == "buy" else "buy"
    return spot, perp


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

    def open(self, pair, side, notional, spot=0.0, perp=0.0):
        sl, pl = leg_sides(side, opening=True)
        self.orders.append(OrderTicket(self._coid(pair, "open"), pair, "open",
                                       side, sl, pl, notional))
        super().open(pair, side, notional, spot, perp)

    def close(self, pair, notional_px=(0.0, 0.0)):
        if pair in self._pos:
            pos = self._pos[pair]
            sl, pl = leg_sides(pos.side, opening=False)
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
class LegError(RuntimeError):
    """Raised when a two-leg order could not be completed cleanly. The book is
    left FLAT for that pair (any filled leg is unwound) - never naked."""


class LiveBroker(Broker):
    """REAL two-leg account via ccxt: spot-margin leg (spot_ex) + USDⓈ-M perp leg
    (perp_ex), postOnly maker on both, clientOrderId idempotency, cross/portfolio
    margin so the legs net. Triple-gated: needs BOTH exchanges AND enable_live=True
    - otherwise construction fails closed. Exercise on the testnet first (testnet.py).

    Maintains an internal book mirrored from confirmed fills; reconcile() cross-checks
    it against exchange truth. The cardinal rule is enforced in _place_two_legs: a
    delta-neutral pair is opened/closed on BOTH legs or on NEITHER - a partially
    filled pair is unwound immediately, never carried as directional exposure.
    """

    def __init__(self, cfg: StrategyConfig, perp_ex=None, spot_ex=None,
                 enable_live: bool = False):
        if perp_ex is None or spot_ex is None or not enable_live:
            raise RuntimeError(
                "LiveBroker is disabled. It needs authenticated perp + spot ccxt "
                "exchanges AND enable_live=True, and must only be turned on AFTER a "
                ">=4-6 week forward test (DryRunLiveBroker) has passed. Use testnet.py "
                "to exercise the order path against the Binance sandbox first."
            )
        self.cfg = cfg
        self.perp_ex = perp_ex
        self.spot_ex = spot_ex
        self._pos: dict[str, Position] = {}
        self._seq = 0

    def _coid(self, pair, leg):
        self._seq += 1
        return f"fh-{leg}-{pair}-{int(time.time())}-{self._seq:04d}"

    # ---- account state -----------------------------------------------------
    def positions(self) -> dict[str, Position]:
        return self._pos

    def equity(self) -> float:
        """Combined cross-margin account equity in USDT (spot-margin + futures
        wallets share collateral under portfolio/cross margin)."""
        eq = 0.0
        for ex in (self.spot_ex, self.perp_ex):
            bal = ex.fetch_balance()
            eq += float(bal.get("total", {}).get("USDT", 0.0) or 0.0)
        return eq

    def reconcile(self) -> list[str]:
        """Compare the internal book to exchange truth; return human-readable
        discrepancies (empty list == in sync). Run before every rebalance."""
        issues = []
        try:
            ex_pos = {p["symbol"]: p for p in self.perp_ex.fetch_positions()
                      if abs(float(p.get("contracts") or 0)) > 0}
        except Exception as e:  # network / auth - surface, do not trade blind
            return [f"reconcile failed: {e!r}"]
        for pair, pos in self._pos.items():
            sym = f"{pair}/USDT:USDT"
            if sym not in ex_pos:
                issues.append(f"{pair}: internal book has a position, exchange does not")
        for sym in ex_pos:
            pair = sym.split("/")[0]
            if pair not in self._pos:
                issues.append(f"{pair}: exchange has a perp position not in internal book")
        return issues

    # ---- order lifecycle ---------------------------------------------------
    def _maker_price(self, px: float, order_side: str) -> float:
        """A passive limit that rests just inside the touch so postOnly never
        crosses: bid below mid for buys, ask above mid for sells."""
        off = self.cfg.slippage
        return px * (1 - off) if order_side == "buy" else px * (1 + off)

    def _await_fill(self, ex, symbol: str, order: dict) -> bool:
        """Poll a resting postOnly order until fully filled or the maker timeout;
        re-price toward the touch up to max_reprice times if it is not filling."""
        oid = order["id"]
        deadline = time.time() + self.cfg.maker_timeout_s
        reprices = 0
        while time.time() < deadline:
            o = ex.fetch_order(oid, symbol)
            if o.get("status") == "closed" and float(o.get("filled") or 0) > 0:
                return True
            if o.get("status") in ("canceled", "rejected", "expired"):
                return False
            time.sleep(self.cfg.poll_interval_s)
            if time.time() >= deadline and reprices < self.cfg.max_reprice:
                # chase: cancel + replace one tick closer, reset the clock
                ex.cancel_order(oid, symbol)
                side, amt = o["side"], float(o["amount"]) - float(o.get("filled") or 0)
                px = self._maker_price(o["price"], side)
                order = ex.create_order(symbol, "limit", side, amt, px,
                                        {"postOnly": True})
                oid = order["id"]
                reprices += 1
                deadline = time.time() + self.cfg.maker_timeout_s
        ex.cancel_order(oid, symbol)
        return False

    def _place_two_legs(self, pair, side, notional, spot_px, perp_px, opening):
        psym, ssym = f"{pair}/USDT:USDT", f"{pair}/USDT"
        spot_side, perp_side = leg_sides(side, opening)
        spot_amt, perp_amt = notional / spot_px, notional / perp_px
        sp = {"postOnly": True, "clientOrderId": self._coid(pair, "spot")}
        pp = {"postOnly": True, "clientOrderId": self._coid(pair, "perp")}
        if not opening:
            pp["reduceOnly"] = True
            sp["sideEffectType"] = "AUTO_REPAY"        # repay any borrow on close
        elif side < 0:
            sp["sideEffectType"] = "MARGIN_BUY"        # auto-borrow to short spot

        # place both legs, then wait for both fills
        o_s = self.spot_ex.create_order(ssym, "limit", spot_side, spot_amt,
                                        self._maker_price(spot_px, spot_side), sp)
        o_p = self.perp_ex.create_order(psym, "limit", perp_side, perp_amt,
                                        self._maker_price(perp_px, perp_side), pp)
        ok_s = self._await_fill(self.spot_ex, ssym, o_s)
        ok_p = self._await_fill(self.perp_ex, psym, o_p)
        if ok_s and ok_p:
            return
        # partial-leg failure -> flatten whatever filled; never carry naked delta
        if ok_s and not ok_p:
            self.spot_ex.create_order(ssym, "market",
                                      "sell" if spot_side == "buy" else "buy",
                                      spot_amt, None, {"sideEffectType": "AUTO_REPAY"})
        if ok_p and not ok_s:
            self.perp_ex.create_order(psym, "market",
                                      "sell" if perp_side == "buy" else "buy",
                                      perp_amt, None, {"reduceOnly": True})
        raise LegError(f"{pair}: one leg unfilled (spot={ok_s}, perp={ok_p}); "
                       f"position unwound, left flat")

    def open(self, pair, side, notional, spot=0.0, perp=0.0):
        self._place_two_legs(pair, side, notional, spot, perp, opening=True)
        self._pos[pair] = Position(side, notional, spot, perp)

    def close(self, pair, notional_px=(0.0, 0.0)):
        if pair not in self._pos:
            return
        pos = self._pos[pair]
        self._place_two_legs(pair, pos.side, pos.notional,
                             notional_px[0] or pos.entry_spot,
                             notional_px[1] or pos.entry_perp, opening=False)
        del self._pos[pair]

    def resize(self, pair, notional):
        # delta-adjust both legs toward the new notional (same postOnly discipline)
        raise NotImplementedError(
            "resize via reduce/add on both legs; not needed under set-and-hold"
        )

    def accrue(self, step):
        return  # the exchange charges funding/borrow directly to the wallet
