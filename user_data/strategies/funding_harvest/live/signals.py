"""Funding signal + target-book construction (point-in-time, leak-free).

build_target() is the single source of truth for "what should the book look like
right now". It is pure: given the currently OBSERVABLE signal/vol and the current
holdings, it returns the desired set of delta-neutral positions. The engine then
diffs desired-vs-held and routes the difference to the executor. Identical in
replay and live - that is what keeps live faithful to the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class TargetPos:
    side: int          # +1 short-perp/long-spot ; -1 long-perp/short-spot
    notional: float    # USD spot notional (== perp notional, delta-neutral)


def build_target(
    sig: pd.Series,          # observable annualised funding per pair (signed)
    vol: pd.Series,          # observable annualised basis-vol per pair
    held: dict[str, TargetPos],
    equity: float,
    cfg: StrategyConfig,
    trend_z: pd.Series | None = None,   # momentum z for the squeeze-tail filter
) -> dict[str, TargetPos]:
    """Return the desired book. Held positions keep their entry notional
    (set-and-hold); only genuinely new positions are sized."""
    # ---- 1. admissible side per pair ---------------------------------------
    side = pd.Series(0, index=sig.index, dtype=int)
    side[sig > cfg.thresh_ann] = 1
    if cfg.both_sides:
        side[sig < -cfg.thresh_ann] = -1
        # never short-spot a squeeze-prone or borrow-restricted name
        for p in cfg.short_exclude:
            if p in side.index and side[p] < 0:
                side[p] = 0
        for p in side.index[side < 0]:
            if p not in cfg.borrow_limit:        # no borrow availability -> no short
                side[p] = 0
    # squeeze-tail filter: drop a leg whose perp side faces an extreme adverse trend
    if trend_z is not None and cfg.trend_filter_z:
        for p in side.index[side != 0]:
            z = trend_z.get(p, np.nan)
            if np.isfinite(z) and ((side[p] > 0 and z > cfg.trend_filter_z)
                                   or (side[p] < 0 and z < -cfg.trend_filter_z)):
                side[p] = 0

    # ---- 2. net-of-cost carry & rank ---------------------------------------
    net = sig.abs().copy()
    for p in side.index[side < 0]:
        net[p] -= cfg.borrow_rate.get(p, 0.10)   # subtract the real borrow cost
    cand = [
        p for p in side.index[side != 0]
        if np.isfinite(net.get(p, np.nan)) and net[p] > 0
        and np.isfinite(vol.get(p, np.nan)) and vol[p] > 0
    ]
    score = {p: net[p] / vol[p] for p in cand}            # carry / basis-vol
    ranked = sorted(cand, key=lambda p: score[p], reverse=True)
    target = ranked[: cfg.n_max]

    # hysteresis: keep a held pair while its funding still backs the SAME side
    # (but drop it if its trend has turned adversely extreme)
    def trend_adverse(p):
        if trend_z is None or not cfg.trend_filter_z:
            return False
        z = trend_z.get(p, np.nan)
        s = held[p].side
        return np.isfinite(z) and ((s > 0 and z > cfg.trend_filter_z)
                                   or (s < 0 and z < -cfg.trend_filter_z))
    keep = [
        p for p in held
        if np.isfinite(sig.get(p, np.nan))
        and abs(sig[p]) > cfg.exit_ann
        and np.sign(sig[p]) == held[p].side
        and not trend_adverse(p)
    ]
    desired_set = list(dict.fromkeys(target + keep))[: cfg.n_max]
    if not desired_set:
        return {}

    # ---- 3. carry/vol weights for NEW entries (capped, renormalised) -------
    raw = {p: max(score.get(p, 0.0), 1e-9) for p in desired_set}
    tot = sum(raw.values())
    w = {p: raw[p] / tot for p in desired_set}
    w = {p: min(v, cfg.max_weight) for p, v in w.items()}
    tot = sum(w.values())
    w = {p: v / tot for p, v in w.items()}
    budget = equity / (1.0 + cfg.perp_margin)

    def desired_side(p: str) -> int:
        return 1 if sig.get(p, 0.0) > 0 else -1

    out: dict[str, TargetPos] = {}
    for p in desired_set:
        s = desired_side(p)
        held_same = p in held and held[p].side == s
        if held_same and cfg.set_and_hold:
            out[p] = TargetPos(s, held[p].notional)   # keep entry size
            continue
        notional = w[p] * budget
        if p in cfg.oi_cap:                            # capacity cap
            notional = min(notional, cfg.oi_cap[p])
        if s < 0:                                      # short-spot constraints
            if p in cfg.borrow_limit:
                notional = min(notional, cfg.borrow_limit[p])
            notional = min(notional, cfg.max_short_per_name * equity)
        out[p] = TargetPos(s, notional)
    return out
