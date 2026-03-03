from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
import numpy as np

from .orderbook import Order, OrderBook, Side

Action = Union[None, Order, Tuple[Literal["cancel"], int]]


@dataclass
class BaseAgent:
    trader_id: int
    rng: np.random.Generator
    _next_order_id: int = 1

    def new_oid(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        # make ids globally unique-ish by namespacing with trader_id
        return int(self.trader_id * 1_000_000 + oid)

    def act(self, t: int, book: OrderBook) -> Action:
        return None


class NoiseTrader(BaseAgent):
    """
    Minimal placeholder:
    - 50/50 buy/sell
    - alternates between market and limit randomly
    """

    def act(self, t: int, book: OrderBook) -> Action:
        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(1, 5))

        if self.rng.random() < 0.5:
            # market
            return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)

        # limit: place ONE TICK away from current best (so we don't pin the best)
        bb, ba = book.best_bid(), book.best_ask()

        if side == "buy":
            if bb > 0:
                px = bb - book.tick
            else:
                px = 100.0 - book.tick
        else:
            if np.isfinite(ba):
                px = ba + book.tick
            else:
                px = 101.0 + book.tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)


class MarketMaker(BaseAgent):
    """
    Minimal placeholder:
    - posts one limit order each step
    - quotes ONE TICK away from best to avoid infinite stacking at best
    """

    def act(self, t: int, book: OrderBook) -> Action:
        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(1, 10))
        bb, ba = book.best_bid(), book.best_ask()

        if side == "buy":
            px = (bb - book.tick) if bb > 0 else (100.0 - book.tick)
        else:
            px = (ba + book.tick) if np.isfinite(ba) else (101.0 + book.tick)

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)

class InstitutionalTrader(BaseAgent):
    """
    Minimal placeholder:
    - rarely acts
    - when acts, sends a larger market order (exercises matching)
    """

    def act(self, t: int, book: OrderBook) -> Action:
        if self.rng.random() > 0.05:
            return None

        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(10, 50))
        return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)