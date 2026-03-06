from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Literal
import logging



plt.style.use('ggplot')

Side = Literal["buy", "sell"]

@dataclass
class Order:
    # TODO: review
    order_id: int
    trader_id: int
    side: Side
    qty: int
    ts: int = 0

@dataclass

class Trade:
    # TODO: complete
    price: float
    qty: int
    buyer_id: int
    seller_id: int
    timestamp: int



class DarkPool:
    """
    Dark pool implementation for institutional traders.
    """

    def __init__(self, lit_orderbook):
        """
        Initialize the dark pool with empty order list and trade history.
        """
        # TODO: complete

        # initialize the dark pool with a reference to the lit order book for price discovery only
        self.lit_orderbook = lit_orderbook

        # FIFO queues for bid and ask orders (we do not need to store the price level)
        self.asks: Dict[float, Deque[Order]] = {}
        self.bids: Dict[float, Deque[Order]] = {}

        # Store the bid and ask prices
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []

    def get_mid_price(self) -> float:
        """
        :return mid_price: mid price of the dark pool based on the lit order book.
        """

        if not self.lit_orderbook.bids or not self.lit_orderbook.asks:
            # check that the lit order book has bids and asks and has been initialized before traders can submit orders
            return np.nan

        else:
            # get the best bid and ask prices from the lit order book
            best_bid = self.lit_orderbook.best_bid()
            best_ask = self.lit_orderbook.best_ask()

            if best_bid is None or best_ask is None:
                # check that the best bid and ask are not None in the lit order book
                return np.nan

            if best_bid <= 0 or best_ask <= 0:
                # check that the best bid and ask are positive values.
                return np.nan

            if best_bid >= best_ask:
                # TODO: check if we want to return none or the best bid in the case of a cross market.
                return np.nan

            if best_bid == best_ask:
                # check for a locked market.
                return best_bid

            # If all checks pass, return the mid price
            return (best_bid + best_ask) / 2


    def submit_order(self, order: Order) -> List[Trade]:
        # TODO: complete

        """
        Function to submit an order to the dark pool.
        :param order: Order object containing order details.
        :return trades: List of trades executed by the dark pool.
        """

        if order.qty <= 0:
            # TODO: check how errors are handled, raising errors or returning an empty list
            raise ValueError("Order quantity must be positive.")

        if order.order_id in self.bids or order.order_id in self.asks:
            raise ValueError("Order ID already exists.")

        if order.side not in ["buy", "sell"]:
            raise ValueError("Order side must be 'buy (bid)' or 'sell (ask)'.")

        if order.side == "buy":
            self.bids[order.order_id] = order
        else:
            self.asks[order.order_id] = order

        # match orders in the dark pool
        # TODO: implement matching logic (price discovery via lit order book, then match orders in the dark pool)
        trades = []
        # trades = self.match_orders()

        # log the order
        # logging.info(f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")
        print(f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")

        return trades






