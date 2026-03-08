from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Literal
import logging

from .orderbook import OrderBook, Order as LitOrder, Trade as LitTrade

plt.style.use('ggplot')

Side = Literal["buy", "sell"]


@dataclass
class Order:
    order_id: int
    trader_id: int
    side: Side
    qty: int
    ts: int = 0  # timestamp when the order was submitted to the dark pool


@dataclass
class Trade:
    price: float
    qty: int
    buyer_id: int
    seller_id: int
    timestamp: int


@dataclass
class TapeEntry:
    """
    A single published record on the dark pool trade tape.
    Post-trade transparency: price, qty, and timestamp are published;
    trader identities remain anonymous.
    """
    price: float
    qty: int
    timestamp: int
    venue: str = "dark_pool"


class DarkPool:
    """
    Dark pool implementation for institutional traders.

    Key features:
    - FIFO matching at lit-book mid-price.
    - Order timeout: unfilled orders that have been resting for more than
      `max_resting_time` steps are expired and rerouted to the lit order book
      as market orders.
    - Trade tape: every dark-pool fill is published to `self.tape` for
      post-trade transparency (anonymous — no trader IDs).
    """

    def __init__(self, lit_orderbook: OrderBook, max_resting_time: int = 10):
        """
        :param lit_orderbook: reference to the lit OrderBook (used for mid-price
                              discovery and as a fallback venue for expired orders).
        :param max_resting_time: maximum number of time steps an order can rest
                                 in the dark pool before being rerouted to the
                                 lit book as a market order.
        """
        # initialize the dark pool with a reference to the lit order book for price discovery only
        self.lit_orderbook = lit_orderbook
        self.max_resting_time = max_resting_time

        # FIFO queues for bid and ask orders (we do not need to store the price level)
        self.asks: Deque[Order] = deque()
        self.bids: Deque[Order] = deque()

        # post-trade transparency tape (anonymous)
        self.tape: List[TapeEntry] = []

        # ledger of all orders that were rerouted to the lit book on expiry
        self.rerouted_orders: List[LitOrder] = []

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
        """
        Submit an order to the dark pool and attempt to match it.

        Dark pool matching rules:
        - All trades execute at the lit order book mid-price (no price improvement negotiation).
        - Orders are matched FIFO: earliest arriving buy is matched against earliest arriving sell.
        - Partial fills are supported: residual quantity stays at the front of its queue.
        - If no mid-price is available from the lit book, the order is queued but not matched.

        :param order: Order object containing order details.
        :return trades: List of trades executed during this submission.
        """

        trades: List[Trade] = []

        if order.qty <= 0:
            raise ValueError("Order quantity must be positive.")

        if any(o.order_id == order.order_id for o in self.bids) or \
                any(o.order_id == order.order_id for o in self.asks):
            raise ValueError("Order ID already exists.")

        # insert the incoming order on the appropriate side
        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

        logging.info(
            f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, "
            f"side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")

        # attempt to match after every submission
        trades = self._match_orders(order.ts)

        return trades

    def _match_orders(self, current_ts: int) -> List[Trade]:
        """
        FIFO matching of queued bids against queued asks at the lit mid-price.

        Matching continues as long as both sides have resting orders and a valid
        mid-price can be derived from the lit order book.

        Every fill is published to the anonymous trade tape for post-trade
        transparency.

        :param current_ts: timestamp to stamp on resulting trades.
        :return trades: list of Trade objects produced by this matching pass.
        """

        trades: List[Trade] = []

        while self.bids and self.asks:
            mid_price = self.get_mid_price()

            if np.isnan(mid_price):
                # cannot determine execution price — stop matching
                logging.warning("Mid price unavailable; pausing matching.")
                break

            # peek at the front of each queue (FIFO)
            bid = self.bids[0]
            ask = self.asks[0]

            # determine fill quantity
            trade_qty = min(bid.qty, ask.qty)
            assert trade_qty > 0, "Queued order with non-positive qty detected."

            # record the trade
            trade = Trade(
                price=mid_price,
                qty=trade_qty,
                buyer_id=bid.trader_id,
                seller_id=ask.trader_id,
                timestamp=current_ts,
            )
            trades.append(trade)

            logging.info(
                f"--- TRADE ---: price={mid_price}, qty={trade_qty}, "
                f"buyer={bid.trader_id} (oid {bid.order_id}), "
                f"seller={ask.trader_id} (oid {ask.order_id}), ts={current_ts}")

            # publish to the anonymous post-trade tape
            self.tape.append(TapeEntry(
                price=mid_price,
                qty=trade_qty,
                timestamp=current_ts,
            ))

            # decrement quantities
            bid.qty -= trade_qty
            ask.qty -= trade_qty

            # remove fully filled orders from the front of the queue
            if bid.qty == 0:
                self.bids.popleft()
            if ask.qty == 0:
                self.asks.popleft()

        return trades

    # -------- order expiry / rerouting --------

    def expire_orders(self, current_ts: int) -> List[LitTrade]:
        """
        Expire dark-pool orders that have been resting longer than
        `self.max_resting_time` time steps. Expired orders are removed from the
        dark pool and sent to the lit order book as **market orders** so that
        the trader still gets a fill (at the best available lit price).

        This should be called once per simulation time step (before or after
        agent actions, depending on desired semantics).

        :param current_ts: the current simulation timestamp.
        :return lit_trades: list of lit-book Trade objects generated by the
                           rerouted market orders.
        """
        lit_trades: List[LitTrade] = []
        expired: List[Order] = []

        # scan both queues and collect expired orders
        for q in (self.bids, self.asks):
            remaining: Deque[Order] = deque()
            while q:
                order = q.popleft()
                age = current_ts - order.ts
                if age >= self.max_resting_time:
                    expired.append(order)
                else:
                    remaining.append(order)
            q.extend(remaining)

        # reroute each expired order to the lit book as a market order
        for dp_order in expired:
            lit_order = LitOrder(
                order_id=dp_order.order_id,
                trader_id=dp_order.trader_id,
                side=dp_order.side,
                qty=dp_order.qty,
                price=None,   # market order — no price
                ts=current_ts,
            )

            logging.info(
                f"--- EXPIRY ---: dark pool order {dp_order.order_id} "
                f"(trader {dp_order.trader_id}, side={dp_order.side}, qty={dp_order.qty}) "
                f"expired after {current_ts - dp_order.ts} steps -> rerouted to lit book "
                f"as market order at ts={current_ts}")

            trades = self.lit_orderbook.execute_market(lit_order)
            lit_trades.extend(trades)
            self.rerouted_orders.append(lit_order)

        return lit_trades

    # -------- tape helpers --------

    def get_tape(self) -> List[TapeEntry]:
        """Return the full anonymous post-trade tape."""
        return list(self.tape)

    def get_tape_dataframe(self) -> pd.DataFrame:
        """Return the trade tape as a pandas DataFrame for analysis."""
        if not self.tape:
            return pd.DataFrame(columns=["price", "qty", "timestamp", "venue"])
        return pd.DataFrame([
            {"price": e.price, "qty": e.qty, "timestamp": e.timestamp, "venue": e.venue}
            for e in self.tape
        ])

    # -------- cancellation & query --------

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a resting order by its order_id.

        :param order_id: the id of the order to cancel.
        :return: True if the order was found and removed, False otherwise.
        """
        for q in (self.bids, self.asks):
            for i, o in enumerate(q):
                if o.order_id == order_id:
                    del q[i]  # deque supports deletion by index
                    logging.info(f"--- CANCEL ---: order {order_id} removed.")
                    return True
        return False

    def has_order(self, order_id: int) -> bool:
        """
        Check whether an order with the given id is resting in the dark pool.

        :param order_id: the id to look up.
        :return: True if found on either side, False otherwise.
        """
        return any(o.order_id == order_id for o in self.bids) or \
            any(o.order_id == order_id for o in self.asks)
