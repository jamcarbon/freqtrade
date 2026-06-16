"""Binance testnet wiring for the LiveBroker order path.

Lets you exercise the REAL two-leg order lifecycle (postOnly placement, fill
monitoring, partial-leg unwind, reconciliation) against Binance's sandbox with
play money, BEFORE risking a cent on mainnet.

Keys come from the environment (never hard-code):
    BINANCE_TESTNET_FUT_KEY / BINANCE_TESTNET_FUT_SECRET   (testnet.binancefuture.com)
    BINANCE_TESTNET_SPOT_KEY / BINANCE_TESTNET_SPOT_SECRET (testnet.binance.vision)

IMPORTANT caveats (be honest about what the testnet can and cannot prove):
  * The USDⓈ-M FUTURES testnet is faithful - the perp leg path is fully testable.
  * The SPOT testnet has NO cross-margin / borrow, so the negative-carry side's
    short-spot+AUTO_BORROW path cannot be validated there. Test the positive side
    end-to-end on testnet; validate the short-spot/borrow leg with a *tiny* real
    mainnet position under supervision before enabling it at size.

Nothing here runs on import or on a plain `python -m ...testnet` invocation. The
self-test only fires when you explicitly set FH_RUN_TESTNET=1, so it can never be
started by accident.
"""
from __future__ import annotations

import os

from .broker import LiveBroker
from .config import StrategyConfig


def make_testnet_exchanges():
    """Return (perp_ex, spot_ex) ccxt handles in sandbox mode, or raise with a
    clear message if keys are missing."""
    import ccxt

    fk, fs = os.getenv("BINANCE_TESTNET_FUT_KEY"), os.getenv("BINANCE_TESTNET_FUT_SECRET")
    sk, ss = os.getenv("BINANCE_TESTNET_SPOT_KEY"), os.getenv("BINANCE_TESTNET_SPOT_SECRET")
    if not (fk and fs and sk and ss):
        raise RuntimeError(
            "Testnet keys missing. Set BINANCE_TESTNET_FUT_KEY/SECRET and "
            "BINANCE_TESTNET_SPOT_KEY/SECRET in the environment first."
        )
    perp_ex = ccxt.binance({
        "apiKey": fk, "secret": fs, "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    spot_ex = ccxt.binance({
        "apiKey": sk, "secret": ss, "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    perp_ex.set_sandbox_mode(True)
    spot_ex.set_sandbox_mode(True)
    return perp_ex, spot_ex


def make_testnet_broker(cfg: StrategyConfig | None = None) -> LiveBroker:
    """A LiveBroker wired to the sandbox. enable_live=True is safe here because the
    exchanges are testnet (play money)."""
    cfg = cfg or StrategyConfig()
    perp_ex, spot_ex = make_testnet_exchanges()
    return LiveBroker(cfg, perp_ex=perp_ex, spot_ex=spot_ex, enable_live=True)


def selftest(pair: str = "BTC", usd: float = 200.0):
    """Open then close ONE tiny positive-carry pair on testnet to prove the order
    path. Positive carry only (long spot / short perp) so it needs no borrow.
    Guarded: only runs when FH_RUN_TESTNET=1."""
    if os.getenv("FH_RUN_TESTNET") != "1":
        print("Refusing to run: set FH_RUN_TESTNET=1 to exercise the testnet path.")
        print("This will place real (play-money) orders on the Binance sandbox.")
        return
    broker = make_testnet_broker()
    print(f"[testnet] reconcile before: {broker.reconcile() or 'in sync'}")
    perp_px = broker.perp_ex.fetch_ticker(f"{pair}/USDT:USDT")["last"]
    spot_px = broker.spot_ex.fetch_ticker(f"{pair}/USDT")["last"]
    print(f"[testnet] OPEN  {pair} +carry ${usd:.0f} (long spot / short perp) ...")
    broker.open(pair, side=1, notional=usd, spot=spot_px, perp=perp_px)
    print(f"[testnet]   opened: {broker.positions()}")
    print(f"[testnet] CLOSE {pair} ...")
    broker.close(pair, (spot_px, perp_px))
    print(f"[testnet]   closed. positions now: {broker.positions()}")
    print(f"[testnet] reconcile after: {broker.reconcile() or 'in sync'}")


if __name__ == "__main__":
    selftest()
