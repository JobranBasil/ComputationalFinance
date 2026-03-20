import numpy as np
import logging

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Dict, Optional, Literal, Set
from .orderbook import Order as LitOrder

Side = Literal['buy', 'sell']

@dataclass
class Order:
    """
    Represents an order in the dark pool.
    """

    order_id: int
    trader_id: int
    side: Side
    qty: int
    ts: int = 0

@dataclass
class Trade:
    """
    Represents a trade in the dark pool.
    """

    price: float
    qty: int
    timestamp: int


class DarkPool:
    """
    Represents a dark pool for trading.
    """

    def __init__(self, lit_order_book, max_resting_ticks: int = 10, routing_delay: int = 10, tape_delay: int = 5):
        self.lit_order_book = lit_order_book

        self.bids: Deque[Order] = deque()
        self.asks: Deque[Order] = deque()

        self.trade_tape: List[Trade] = []

        self.max_resting_ticks: int = max_resting_ticks
        self.routing_delay: int = routing_delay
        self.tape_delay: int = tape_delay
        self.pending_lit_routes: List[tuple[int, LitOrder]] = []

        self.order_index: Dict[int, Order] = {}
        self.cancelled_ids: Set[int] = set()

    def compute_mid_price(self):
        # todo: implement
        pass

    def match_order(self, order: Order, lit_order: LitOrder):
        # todo: implement
        pass

    def submit_order(self, order: Order) -> List[Trade]:
        # todo: implement
        pass

    def tick(self, t: int):
        # todo: implement
        pass

    def cancel_order(self, order_id: int) -> bool:
        # todo: implement
        pass

    def has_order(self) -> bool:
        # todo: implement
        pass

    def queue_depth(self) -> tuple[int, int]:
        # todo: implement
        pass

    def _process_pending_routes(self, current_ts: int):
        # todo: implement
        pass

    def _expire_stale_orders(self, current_ts: int) -> List[LitOrder]:
        # todo: implement
        pass


