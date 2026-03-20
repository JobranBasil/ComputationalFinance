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

    # todo: implement
    pass

@dataclass
class Trade:
    """
    Represents a trade in the dark pool.
    """


    # todo: implement
    pass


class DarkPool:
    """
    Represents a dark pool for trading.
    """

    def __init__(self, lit_order_book, max_resting_ticks: int = 10, routing_delay: int = 10, tape_delay: int = 5):
        # todo: implement
        pass

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

    def queue_depth(self) -> [tuple, tuple]:
        # todo: implement
        pass

    def _process_pending_routes(self, current_ts: int):
        # todo: implement
        pass

    def _expire_stale_orders(self, current_ts: int) -> List[LitOrder]:
        # todo: implement
        pass


