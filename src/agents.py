# src/agents.py
from __future__ import annotations

from typing import List, Optional, Tuple, Union, Literal
import numpy as np

from .orderbook import OrderBook, Order, Side

CancelAction = Tuple[Literal["cancel"], int, Optional[int]]
Action = Union[Order, CancelAction]


class Agent:
    """
    Base agent. Teammates implement step().
    """
    def __init__(self, agent_id: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.rng = rng
        self.new_order_id = None  # injected by simulator

    def step(self, t: int, book: OrderBook) -> List[Action]:
        return []


class NoiseTrader(Agent):
    """
    Placeholder:
      - small market orders, random direction
    """
    def __init__(self, agent_id: int, rng: np.random.Generator, p_act: float = 0.3, max_qty: int = 5):
        super().__init__(agent_id, rng)
        self.p_act = p_act
        self.max_qty = max_qty

    def step(self, t: int, book: OrderBook) -> List[Action]:
        if self.rng.random() > self.p_act:
            return []
        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(1, self.max_qty + 1))
        oid = self.new_order_id()
        return [Order(order_id=oid, trader_id=self.agent_id, side=side, qty=qty, price=None, ts=t)]


class MarketMaker(Agent):
    """
    Placeholder:
      - posts 1+ levels on both sides near best bid/ask
    """
    def __init__(self, agent_id: int, rng: np.random.Generator, quote_size: int = 5, levels: int = 1):
        super().__init__(agent_id, rng)
        self.quote_size = quote_size
        self.levels = levels

    def step(self, t: int, book: OrderBook) -> List[Action]:
        bb, ba = book.best_bid(), book.best_ask()
        if not np.isfinite(ba) or bb <= 0:
            return []

        acts: List[Action] = []
        for k in range(self.levels):
            bid_px = book.snap(bb - k * book.tick)
            ask_px = book.snap(ba + k * book.tick)
            acts.append(Order(self.new_order_id(), self.agent_id, "buy", self.quote_size, price=bid_px, ts=t))
            acts.append(Order(self.new_order_id(), self.agent_id, "sell", self.quote_size, price=ask_px, ts=t))
        return acts


class Institutional(Agent):
    """
    Placeholder iceberg parent order.
    NOTE: this submits iceberg slices; trade attribution can be added later.
    """
    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        side: Side = "buy",
        parent_qty: int = 500,
        display_qty: int = 10,
        refresh_size: int = 10,
    ):
        super().__init__(agent_id, rng)
        self.side = side
        self.remaining = parent_qty
        self.display_qty = display_qty
        self.refresh_size = refresh_size

    def step(self, t: int, book: OrderBook) -> List[Action]:
        if self.remaining <= 0:
            return []

        bb, ba = book.best_bid(), book.best_ask()
        if not np.isfinite(ba) or bb <= 0:
            return []

        if self.side == "buy":
            px = book.snap(bb)
            px = min(px, book.snap(ba - book.tick))  # never cross
        else:
            px = book.snap(ba)
            px = max(px, book.snap(bb + book.tick))

        visible = min(self.display_qty, self.remaining)
        hidden = max(0, self.remaining - visible)

        oid = self.new_order_id()
        o = Order(
            order_id=oid,
            trader_id=self.agent_id,
            side=self.side,
            qty=visible,
            price=px,
            ts=t,
            display_qty=self.display_qty,
            hidden_qty=hidden,
            refresh_size=self.refresh_size,
        )

        # Placeholder: decrement parent on submission.
        # Later: decrement using executed trades attributed to this agent’s child order_ids.
        self.remaining -= visible

        return [o]