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
        self.asks: Deque[Order] = deque()
        self.bids: Deque[Order] = deque()

    def get_mid_price(self) -> float:
        """
        :return mid_price: mid price of the dark pool based on the lit order book.
        """

        if not self.lit_orderbook.bids or not self.lit_orderbook.asks:
            # check that the lit order book has bids and asks and has been initialized before traders can submit orders
            # raise ValueError("Lit order book is empty. Traders cannot submit orders.")
            return np.nan

        # get the best bid and ask prices from the lit order book
        best_bid = self.lit_orderbook.best_bid()
        best_ask = self.lit_orderbook.best_ask()

        if any(
                [
                    # check that the best bid and ask are not None in the lit order book
                    best_bid is None,
                    best_ask is None,

                    # check that the best bid and ask are positive values.
                    best_bid <= 0,
                    best_ask <= 0,

                    # check that the best bid and ask are finite values.
                    not np.isfinite(best_bid),
                    not np.isfinite(best_ask),

                    # TODO: check if we want to return none or the best bid in the case of a cross market.
                    best_bid >= best_ask
                ]
        ):
            return np.nan


        else:
            # If all checks pass, return the mid price
            return (best_bid + best_ask) / 2

    def submit_order(self, order: Order) -> List[Trade]:
        # TODO: complete

        """
        Function to submit an order to the dark pool.
        :param order: Order object containing order details.
        :return trades: List of trades executed by the dark pool.
        """

        trades: List[Trade] = []

        if order.qty <= 0:
            # TODO: check how errors are handled, raising errors or returning an empty list
            raise ValueError("Order quantity must be positive.")

        if any(o.order_id == order.order_id for o in self.bids) or any(o.order_id == order.order_id for o in self.asks):
            raise ValueError("Order ID already exists.")

        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

        logging.info(
            f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")

        mid_price = self.get_mid_price()

        if np.isnan(mid_price):
            # If the mid price is not available, we cannot execute the order.
            logging.warning("Mid price is not available. Order submitted but cannot be executed.")
            return trades

        # TODO: check logic of order matching

        while self.bids and self.asks:
            # get the best bid and ask prices from the lit order book
            bid = self.bids.popleft()
            ask = self.asks.pop()

            # determine the trade quantity based on the available quantity in the dark pool
            trade_qty = min(bid.qty, ask.qty)

            # check if the trade quantity is positive
            if trade_qty < 0:
                raise ValueError("Trade quantity must be positive.")

            # create a trade object
            trade = Trade(mid_price, trade_qty, bid.trader_id, ask.trader_id, order.ts)

            # add the trade to the trade history
            trades.append(trade)

            # update the quantity in the dark pool
            bid.qty -= trade_qty
            ask.qty -= trade_qty

            # check if the quantity in the dark pool becomes zero and add it back to the queue
            if bid.qty == 0:
                self.bids.appendleft(bid)
            if ask.qty == 0:
                self.asks.append(ask)

            # if the trade quantity is zero, we cannot execute the trade and we should break the loop
            if not self.bids and not self.asks:
                break

        return trades

    def cancel_order(self, order_id: int) -> bool:
        pass

    def has_order(self):
        pass
