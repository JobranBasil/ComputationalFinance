from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Literal
import logging
from src.orderbook import Order as LitOrder

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

    def __init__(self, lit_orderbook, max_resting_ticks: int = 10, routing_delay: int = 10):
        """
        Initialize the dark pool with empty order list and trade history.

        :param lit_orderbook: reference to the lit OrderBook for price discovery.
        :param max_resting_ticks: maximum number of timesteps an order can rest
                                  in the dark pool before being routed to the
                                  lit order book as a market order.
        :param routing_delay: number of timesteps to wait before a stale order
                              is actually submitted to the lit order book,
                              mimicking real-world routing latency.
        """

        # initialize the dark pool with a reference to the lit order book for price discovery only
        self.lit_orderbook = lit_orderbook

        # FIFO queues for bid and ask orders (we do not need to store the price level)
        self.asks: Deque[Order] = deque()
        self.bids: Deque[Order] = deque()

        # public trade tape so that every executed trade is published here so all
        # traders can observe dark-pool activity after the fact.
        self.trade_tape: List[Trade] = []

        # maximum number of timesteps an order may rest before expiry
        self.max_resting_ticks: int = max_resting_ticks

        # number of timesteps to delay before routing a stale order to the lit book
        self.routing_delay: int = routing_delay

        # orders pending lit-book routing: list of (execute_at_ts, LitOrder)
        self.pending_lit_routes: List[tuple[int, LitOrder]] = []

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
        # TODO: review

        """
        Function to submit an order to the dark pool.
        :param order: Order object containing order details.
        :return trades: List of trades executed by the dark pool.
        """

        trades: List[Trade] = []

        if order.qty <= 0:
            # TODO: check how errors are handled, raising errors or returning an empty list
            raise ValueError("Order quantity must be positive.")

        if (
            any(o.order_id == order.order_id for o in self.bids)
            or any(o.order_id == order.order_id for o in self.asks)
            or any(o.order_id == order.order_id for _, o in self.pending_lit_routes)
        ):
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

        trades = self._match_orders(mid_price, order.ts)

        return trades

    def _match_orders(self, mid_price: float, timestamp: int) -> List[Trade]:
        """
        FIFO matching engine for the dark pool.

        Matches the oldest buy against the oldest sell at the current lit mid price.
        Partially filled orders are placed back at the front of their respective queue
        so they retain time priority for the next matching round.

        After matching, stale orders (those exceeding max_resting_ticks) are expired
        and routed to the lit order book as market orders.

        Every executed trade is published to self.trade_tape so that all traders
        can observe dark-pool activity after the fact.

        :param mid_price: execution price (lit book mid).
        :param timestamp: timestamp attached to generated trades.
        :return trades: list of Trade objects produced by this matching round.
        """

        trades: List[Trade] = []

        # submit any previously scheduled lit-book routes whose delay has elapsed
        # todo: implement pending order processing logic
        # self._process_pending_routes(timestamp)

        while self.bids and self.asks:
            # peek at the oldest bid and ask
            bid = self.bids[0]
            ask = self.asks[0]

            # determine the fillable quantity
            trade_qty = min(bid.qty, ask.qty)

            if trade_qty <= 0:
                # safety check – should not happen with valid orders
                break

            # create a new trade
            trade = Trade(
                price=mid_price,
                qty=trade_qty,
                buyer_id=bid.trader_id,
                seller_id=ask.trader_id,
                timestamp=timestamp,
            )
            trades.append(trade)

            # publish the trade to the public tape
            self.trade_tape.append(trade)

            # log the trade
            logging.info(
                f"--- TRADE ---: buyer: {bid.trader_id}, seller: {ask.trader_id}, "
                f"price: {mid_price}, qty: {trade_qty}, timestamp: {timestamp}"
            )

            # update remaining quantities
            bid.qty -= trade_qty
            ask.qty -= trade_qty

            # remove fully filled orders from the front of each queue
            if bid.qty == 0:
                self.bids.popleft()
            if ask.qty == 0:
                self.asks.popleft()

        # expire stale orders that have been resting too long
        self._expire_stale_orders(timestamp)

        return trades

    def cancel_order(self, order_id: int) -> bool:
        """
        :param order_id: the ID of the order to cancel.
        :return: True if the order was found and cancelled, False otherwise.
        """

        for queue in (self.bids, self.asks):
            for i, order in enumerate(queue):
                if order.order_id == order_id:
                    del queue[i]
                    logging.info(f"--- ORDER CANCELLED ---: order_id: {order_id}")
                    return True
        return False

    def has_order(self) -> bool:
        """
        :return: True if there is at least one order in either queue, False otherwise.
        """

        return len(self.bids) > 0 or len(self.asks) > 0
