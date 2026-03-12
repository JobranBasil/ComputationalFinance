from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Literal, Set
import logging
from src.orderbook import Order as LitOrder

plt.style.use('ggplot')

Side = Literal["buy", "sell"]


@dataclass
class Order:
    order_id: int
    trader_id: int
    side: Side
    qty: int
    ts: int = 0


@dataclass
class Trade:
    price: float
    qty: int
    buyer_id: int
    seller_id: int
    timestamp: int


class DarkPool:
    def __init__(self, lit_orderbook, max_resting_ticks: int = 10, routing_delay: int = 10):
        """
        Initialize the dark pool with empty order list and trade history.

        :param lit_orderbook: reference to the lit OrderBook for price discovery.
        :param max_resting_ticks: maximum number of timesteps an order can rest in the dark pool before being routed to the lit order book as a market order.
        :param routing_delay: number of timesteps to wait before a stale order is actually submitted to the lit order book, mimicking real-world routing latency.
        """


        # initialize the dark pool with a reference to the lit order book only for price discovery
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

        # Order lookup dict for constant time order lookup
        self._order_index: Dict[int, Order] = {}

        # lazy-deletion set where cancelled order IDs skipped during matching/expiry
        self._cancelled_ids: Set[int] = set()

    def mid_price(self) -> float:
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
        Function to submit an order to the dark pool.
        :param order: Order object containing order details.
        :return trades: List of trades executed by the dark pool.
        """


        # TODO: review
        trades: List[Trade] = []

        if order.qty <= 0:
            raise ValueError("Order quantity must be positive.")

        if order.side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side '{order.side}'. Must be 'buy' or 'sell'.")

        if order.ts < 0:
            raise ValueError("Order timestamp must be non-negative.")

        # duplicate check via index; also covers pending lit routes
        if order.order_id in self._order_index or any(o.order_id == order.order_id for _, o in self.pending_lit_routes):
            raise ValueError("Order ID already exists.")

        # Prevent self-trading: reject if this trader already has an active order on the opposite side
        opposite_side = "sell" if order.side == "buy" else "buy"
        if any(o.trader_id == order.trader_id and o.side == opposite_side for o in self._order_index.values()):
            raise ValueError(
                f"Self-trading rejected: trader {order.trader_id} already has an active {opposite_side} order."
            )

        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

        self._order_index[order.order_id] = order

        logging.info(
            f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")

        # TODO: consider executing against the mid price immediately upon submission, or should we wait until the next time step? depends on behavior as the order is queued but is mid price in nan then the trades are none but order is still sitting in the dark pool.
        mid_price = self.mid_price()

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

        After matching, stale orders are expired and scheduled for lit-book routing.
        Pending routes are NOT processed here — the simulation loop must call
        _process_pending_routes(t) each tick so routing fires on actual time steps,
        not only when new dark pool orders arrive.

        Every executed trade is published to self.trade_tape so that all traders
        can observe dark-pool activity.

        :param mid_price: execution price (lit book mid).
        :param timestamp: timestamp attached to generated trades.
        :return trades: list of Trade objects produced by this matching round.
        """


        trades: List[Trade] = []

        while self.bids and self.asks:
            # drain any lazily-cancelled orders from the front of each queue
            while self.bids and self.bids[0].order_id in self._cancelled_ids:
                self._cancelled_ids.discard(self.bids.popleft().order_id)

            while self.asks and self.asks[0].order_id in self._cancelled_ids:
                self._cancelled_ids.discard(self.asks.popleft().order_id)

            # check if there are any orders left to match
            if not self.bids or not self.asks:
                break

            # peek at the oldest bid and ask
            bid = self.bids[0]
            ask = self.asks[0]

            # determine the fillable quantity as we can have partial fills.
            trade_qty = min(bid.qty, ask.qty)

            if trade_qty <= 0:
                # safety check and should not happen with valid orders
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
                f"--- (DARK POOL) TRADE ---: buyer: {bid.trader_id}, seller: {ask.trader_id}, "
                f"price: {mid_price}, qty: {trade_qty}, timestamp: {timestamp}"
            )

            # update remaining quantities
            bid.qty -= trade_qty
            ask.qty -= trade_qty

            # remove fully filled orders from the front of each queue
            if bid.qty == 0:
                self._order_index.pop(self.bids.popleft().order_id, None)
            if ask.qty == 0:
                self._order_index.pop(self.asks.popleft().order_id, None)

        # expire stale orders that have been resting too long and route them to the lit book as market orders
        self._expire_stale_orders(timestamp)

        # return the list of trades filled by this matching round
        return trades

    def _process_pending_routes(self, current_ts: int) -> None:
        """
        Submit scheduled lit-book routes whose routing delay has elapsed.

        :param current_ts: the current simulation timestamp.
        """


        due = [(ts, order) for ts, order in self.pending_lit_routes if current_ts >= ts]

        self.pending_lit_routes = [(ts, order) for ts, order in self.pending_lit_routes if current_ts < ts]

        for execute_at, lit_order in due:
            self.lit_orderbook.execute_market(lit_order)
            logging.info(
                f"--- PENDING ROUTE EXECUTED ---: "
                f"order_id: {lit_order.order_id}, trader_id: {lit_order.trader_id}, "
                f"side: {lit_order.side}, qty: {lit_order.qty}, "
                f"scheduled_at: {execute_at}, executed_at: {current_ts}"
            )

    def _expire_stale_orders(self, current_ts: int) -> List[LitOrder]:
        """
        Remove orders that have been resting in the dark pool longer than
        max_resting_ticks and route them to the lit order book as market orders.

        :param current_ts: the current simulation timestamp.
        :return routed: list of LitOrder objects that were sent to the lit book.
        """

        routed: List[LitOrder] = []

        for queue in (self.bids, self.asks):
            remaining: Deque[Order] = deque()
            while queue:
                order = queue.popleft()

                # discard canceled orders encountered during expiry sweep
                if order.order_id in self._cancelled_ids:
                    self._cancelled_ids.discard(order.order_id)
                    continue

                order_age = current_ts - order.ts

                if order_age >= self.max_resting_ticks:
                    # convert to a lit market order (price=None → market order)
                    lit_order = LitOrder(
                        order_id=order.order_id,
                        trader_id=order.trader_id,
                        side=order.side,
                        qty=order.qty,
                        price=None,
                        ts=current_ts,
                    )
                    execute_at = current_ts + self.routing_delay
                    self.pending_lit_routes.append((execute_at, lit_order))
                    self._order_index.pop(order.order_id, None)

                    logging.info(
                        f"--- EXPIRED ORDER SCHEDULED FOR LIT BOOK ---: "
                        f"order_id: {order.order_id}, trader_id: {order.trader_id}, "
                        f"side: {order.side}, qty: {order.qty}, "
                        f"age: {order_age} ticks, execute_at: {execute_at}"
                    )
                    routed.append(lit_order)
                else:
                    remaining.append(order)

            # replace the queue contents with the non-expired orders
            queue.extend(remaining)

        return routed

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel via lazy deletion. The order is marked as canceled and is "silently" dropped the next time it reaches
        the front of its queue.

        :param order_id: the ID of the order to cancel.
        :return: True if the order was found and marked canceled, False otherwise.
        """

        if order_id not in self._order_index:
            return False

        self._cancelled_ids.add(order_id)
        del self._order_index[order_id]
        logging.info(f"--- (DARK POOL) ORDER CANCELLED ---: order_id: {order_id}")
        return True

    def has_order(self) -> bool:
        """
        :return: True if there is at least one active (non-canceled) order in either queue, False otherwise.
        """

        return len(self._order_index) > 0

    def queue_depth(self):
        """
        Function to return the current queue depth of the dark pool.
        :return: bid_qty, ask_qty
        """


        bid_qty = sum(o.qty for o in self._order_index.values() if o.side == "buy")
        ask_qty = sum(o.qty for o in self._order_index.values() if o.side == "sell")

        # edge case checks
        if bid_qty < 0 or ask_qty < 0:
            raise ValueError("Negative queue depth detected.")

        return bid_qty, ask_qty
