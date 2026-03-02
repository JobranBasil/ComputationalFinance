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
    def __init__(
        self,
        trader_id: int,
        rng: np.random.Generator,
        participation_rate: float = 0.3,
        market_prob: float = 0.3,
        sign_persistence: float = 0.7,
        max_depth_ticks: int = 3,
    ):
        super().__init__(trader_id, rng)
        self.participation_rate = participation_rate
        self.market_prob = market_prob
        self.sign_persistence = sign_persistence
        self.max_depth_ticks = max_depth_ticks
        self.last_side: Optional[Side] = None

    def act(self, t: int, book: OrderBook):

        # participation decision
        if self.rng.random() > self.participation_rate:
            return None

        # persistent order sign
        if self.last_side is None or self.rng.random() > self.sign_persistence:
            side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        else:
            side = self.last_side

        self.last_side = side

        qty = int(self.rng.integers(1, 5))

        # market order
        if self.rng.random() < self.market_prob:
            return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)

        # limit order around mid
        mid = book.mid_price()
        if not np.isfinite(mid):
            mid = 100.0

        tick = book.tick
        offset_ticks = int(self.rng.integers(0, self.max_depth_ticks + 1))

        if side == "buy":
            px = mid - offset_ticks * tick
        else:
            px = mid + offset_ticks * tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)


class MarketMaker(BaseAgent):
    """
    Minimal placeholder:
    - posts one bid and one ask at the current bests (degenerate, but exercises add_limit)
    - occasionally cancels a random existing order from the book
    """

    def act(self, t: int, book: OrderBook) -> Action:
        r = self.rng.random()

        side: Side = "buy" if r < 0.55 else "sell"
        qty = int(self.rng.integers(1, 10))
        bb, ba = book.best_bid(), book.best_ask()

        px = bb if side == "buy" else ba
        if side == "buy" and bb <= 0:
            px = 100.0
        if side == "sell" and not np.isfinite(ba):
            px = 101.0

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