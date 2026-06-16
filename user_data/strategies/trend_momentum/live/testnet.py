"""Binance testnet wiring for the #2 LiveBroker.

Unlike #3 (whose short-spot/borrow leg can't be exercised on the spot testnet),
#2 is a single-leg USDⓈ-M perp book and is FULLY testable on the futures testnet
(testnet.binancefuture.com). That makes #2 the cleaner first live-path validation.

Keys from env:  BINANCE_TESTNET_FUT_KEY / BINANCE_TESTNET_FUT_SECRET
Nothing runs unless FH_RUN_TESTNET=1, so it can never start by accident.
"""
from __future__ import annotations

import os

from .broker import LiveBroker
from .config import TrendConfig


def make_testnet_exchange():
    import ccxt
    k, s = os.getenv("BINANCE_TESTNET_FUT_KEY"), os.getenv("BINANCE_TESTNET_FUT_SECRET")
    if not (k and s):
        raise RuntimeError("Set BINANCE_TESTNET_FUT_KEY / _SECRET in the environment.")
    ex = ccxt.binance({"apiKey": k, "secret": s, "enableRateLimit": True,
                       "options": {"defaultType": "future"}})
    ex.set_sandbox_mode(True)
    return ex


def make_testnet_broker(cfg: TrendConfig | None = None) -> LiveBroker:
    cfg = cfg or TrendConfig()
    return LiveBroker(cfg, exchange=make_testnet_exchange(), enable_live=True)


def selftest(pair: str = "BTC", usd: float = 200.0):
    """Open a small long then flatten it on the futures testnet to prove the
    single-leg order path. Guarded: only runs with FH_RUN_TESTNET=1."""
    if os.getenv("FH_RUN_TESTNET") != "1":
        print("Refusing to run: set FH_RUN_TESTNET=1 to place sandbox orders.")
        return
    b = make_testnet_broker()
    print(f"[testnet] reconcile: {b.reconcile() or 'in sync'}")
    px = b.ex.fetch_ticker(f"{pair}/USDT:USDT")["last"]
    print(f"[testnet] OPEN long {pair} ${usd:.0f}")
    b.set_book({pair: usd}, {pair: px})
    print(f"  positions: {b.positions()}")
    print(f"[testnet] FLATTEN {pair}")
    b.set_book({}, {pair: px})
    print(f"  positions: {b.positions()}  reconcile: {b.reconcile() or 'in sync'}")


if __name__ == "__main__":
    selftest()
