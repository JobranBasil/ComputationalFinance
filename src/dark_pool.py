from __future__ import annotations

import numpy as np
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Literal, Set
import logging
from .orderbook import Order as LitOrder

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
    """
    Anonymous dark pool trade. Counterparty IDs are intentionally omitted —
    dark pools do not publish who traded, only that a trade occurred.
    """
    price: float
    qty: int
    timestamp: int


class DarkPool:
    def __init__(self, lit_orderbook, max_resting_ticks: int = 10, routing_delay: int = 10, tape_delay: int = 5, route_qty_cap: int = 50):
        """
        Initialize the dark pool with empty order list and trade history.

        :param lit_orderbook: reference to the lit OrderBook for price discovery.
        :param max_resting_ticks: maximum number of timesteps an order can rest in the dark pool
        :param routing_delay: number of timesteps to wait before a stale order is actually submitted to the lit order book.
        :param tape_delay: number of timesteps before a trade appears on the public tape (information revelation lag).
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

        # number of timesteps before a trade is visible on the public tape
        self.tape_delay: int = tape_delay

        # max units routed to lit per expiry event; remainder re-queues in dark pool
        self.route_qty_cap: int = route_qty_cap

        # parallel timestamp list for O(log n) bisect lookup in recent_volume
        self._tape_timestamps: List[int] = []

        # orders pending lit-book routing: list of (execute_at_ts, LitOrder)
        self.pending_lit_routes: List[tuple[int, LitOrder]] = []

        # Order lookup dict for order lookup
        self._order_index: Dict[int, Order] = {}

        # lazy-deletion set where cancelled order IDs skipped during matching/expiry
        self._cancelled_ids: Set[int] = set()

    def mid_price(self) -> float:
        """
        :return mid_price: mid price of the dark pool based on the lit order book.
        """

        if not self.lit_orderbook.bids or not self.lit_orderbook.asks:
            return np.nan

        best_bid = self.lit_orderbook.best_bid()
        best_ask = self.lit_orderbook.best_ask()

        if any(
                [
                    best_bid is None,
                    best_ask is None,
                    best_bid <= 0,
                    best_ask <= 0,
                    not np.isfinite(best_bid),
                    not np.isfinite(best_ask),
                    best_bid >= best_ask
                ]
        ):
            return np.nan

        return (best_bid + best_ask) / 2

    def submit_order(self, order: Order) -> List[Trade]:
        """
        Submit an order to the dark pool and attempt to match immediately.

        :param order: Order object containing order details.
        :return trades: List of trades executed by the dark pool.
        """

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

        if order.side == "buy":
            self.bids.append(order)
        else:
            self.asks.append(order)

        self._order_index[order.order_id] = order

        logging.info(
            f"--- ORDER SUBMISSION ---: trader: {order.trader_id}, order: {order.order_id}, "
            f"side: {order.side}, qty: {order.qty}, timestamp: {order.ts}"
        )

        mid_price = self.mid_price()

        if np.isnan(mid_price):
            logging.warning("Mid price is not available. Order submitted but cannot be executed.")
            return trades

        trades = self._match_orders(mid_price, order.ts)

        return trades

    def _match_orders(self, mid_price: float, timestamp: int) -> List[Trade]:
        """
        Size-priority matching for the dark pool.

        Selects the largest resting bid and largest resting ask each round,
        matching them at the current lit mid price. Mirrors real dark pool
        behaviour where block orders are given preference over smaller orders.

        Every executed trade is published to self.trade_tape.

        :param mid_price: execution price (lit book mid).
        :param timestamp: timestamp attached to generated trades.
        :return trades: list of Trade objects produced by this matching round.
        """

        trades: List[Trade] = []

        # Purge cancelled orders upfront so size comparisons are clean.
        self.bids = deque(o for o in self.bids if o.order_id not in self._cancelled_ids)
        self.asks = deque(o for o in self.asks if o.order_id not in self._cancelled_ids)
        self._cancelled_ids.clear()

        while self.bids and self.asks:
            # Size priority: largest qty wins on each side.
            bid = max(self.bids, key=lambda o: o.qty)
            ask = max(self.asks, key=lambda o: o.qty)

            # Self-trade prevention: find largest ask from a different trader.
            if bid.trader_id == ask.trader_id:
                other_asks = [o for o in self.asks if o.trader_id != bid.trader_id]
                if not other_asks:
                    break
                ask = max(other_asks, key=lambda o: o.qty)

            trade_qty = min(bid.qty, ask.qty)

            if trade_qty <= 0:
                break

            trade = Trade(
                price=mid_price,
                qty=trade_qty,
                timestamp=timestamp,
            )
            trades.append(trade)
            self.trade_tape.append(trade)
            self._tape_timestamps.append(timestamp)

            logging.info(
                f"--- (DARK POOL) TRADE ---: buyer: {bid.trader_id}, seller: {ask.trader_id}, "
                f"price: {mid_price}, qty: {trade_qty}, timestamp: {timestamp}"
            )

            bid.qty -= trade_qty
            ask.qty -= trade_qty

            if bid.qty == 0:
                self.bids.remove(bid)
                self._order_index.pop(bid.order_id, None)
            if ask.qty == 0:
                self.asks.remove(ask)
                self._order_index.pop(ask.order_id, None)

        return trades

    def _process_pending_routes(self, current_ts: int) -> None:
        """
        Submit scheduled lit-book routes whose routing delay has elapsed.

        :param current_ts: the current simulation timestamp.
        """

        due = [(ts, order) for ts, order in self.pending_lit_routes if current_ts >= ts]
        self.pending_lit_routes = [(ts, order) for ts, order in self.pending_lit_routes if current_ts < ts]

        for execute_at, lit_order in due:
            # bug fix as the calcuation was happening after the lit order submission
            original_qty = lit_order.qty
            lit_trades = self.lit_orderbook.execute_market(lit_order)
            filled_qty = sum(tr.qty for tr in lit_trades)
            unfilled_qty = original_qty - filled_qty
            logging.info(
                f"---------- PENDING ROUTE EXECUTED ----------: "
                f"order_id: {lit_order.order_id}, trader_id: {lit_order.trader_id}, "
                f"side: {lit_order.side}, qty: {original_qty}, "
                f"filled: {filled_qty}, unfilled: {unfilled_qty}, "
                f"scheduled_at: {execute_at}, executed_at: {current_ts}"
            )
            if unfilled_qty > 0:
                logging.warning(
                    f"----- PENDING ROUTE PARTIALLY FILLED ----- "
                    f"order_id: {lit_order.order_id}, unfilled_qty: {unfilled_qty} (lit book too thin)"
                )

    def _expire_stale_orders(self, current_ts: int) -> List[LitOrder]:
        """
        Remove orders that have been resting longer than max_resting_ticks and
        route them to the lit order book as market orders.

        :param current_ts: the current simulation timestamp.
        :return routed: list of LitOrder objects sent to the lit book.
        """

        routed: List[LitOrder] = []

        for queue in (self.bids, self.asks):
            remaining: Deque[Order] = deque()
            while queue:
                order = queue.popleft()

                if order.order_id in self._cancelled_ids:
                    self._cancelled_ids.discard(order.order_id)
                    continue

                order_age = current_ts - order.ts

                if order_age >= self.max_resting_ticks:
                    route_qty = min(order.qty, self.route_qty_cap)
                    remainder_qty = order.qty - route_qty

                    lit_order = LitOrder(
                        order_id=order.order_id,
                        trader_id=order.trader_id,
                        side=order.side,
                        qty=route_qty,
                        price=None,
                        ts=current_ts,
                    )
                    execute_at = current_ts + self.routing_delay
                    self.pending_lit_routes.append((execute_at, lit_order))
                    self._order_index.pop(order.order_id, None)

                    logging.info(
                        f"---------- EXPIRED ORDER SCHEDULED FOR LIT BOOK ----------: "
                        f"order_id: {order.order_id}, trader_id: {order.trader_id}, "
                        f"side: {order.side}, route_qty: {route_qty}, remainder: {remainder_qty}, "
                        f"age: {order_age} ticks, execute_at: {execute_at}"
                    )
                    routed.append(lit_order)

                    # Re-queue remainder with a fresh timestamp so it rests again
                    if remainder_qty > 0:
                        refreshed = Order(
                            order_id=order.order_id,
                            trader_id=order.trader_id,
                            side=order.side,
                            qty=remainder_qty,
                            ts=current_ts,
                        )
                        remaining.append(refreshed)
                        self._order_index[refreshed.order_id] = refreshed
                else:
                    remaining.append(order)

            queue.extend(remaining)

        return routed

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel via lazy deletion. The order is marked as canceled and is silently
        dropped the next time it reaches the front of its queue.

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

    def queue_depth(self) -> tuple[int, int]:
        """
        Return the total resting quantity on each side.

        :return: (bid_qty, ask_qty)
        """

        bid_qty = sum(o.qty for o in self._order_index.values() if o.side == "buy")
        ask_qty = sum(o.qty for o in self._order_index.values() if o.side == "sell")

        if bid_qty < 0 or ask_qty < 0:
            raise ValueError("Negative queue depth detected.")

        return bid_qty, ask_qty

    def recent_volume(self, current_ts: int, lookback: int) -> int:
        """
        Sum of traded quantity visible on the public tape within the window
        [current_ts - lookback - tape_delay, current_ts - tape_delay].

        :param current_ts: current simulation timestamp.
        :param lookback: number of ticks to look back from the publication frontier.
        :return: total traded quantity visible to the market in this window.
        """


        visible_until = current_ts - self.tape_delay
        window_start = visible_until - lookback
        lo = bisect_left(self._tape_timestamps, window_start)
        hi = bisect_right(self._tape_timestamps, visible_until)
        return int(sum(self.trade_tape[i].qty for i in range(lo, hi)))

    def tick(self, t: int) -> None:
        """
        Advance the dark pool clock by one simulation timestep.

        Must be called by the simulation loop every tick, regardless of whether
        any orders were submitted. This guarantees that stale orders are expired
        on actual time ticks and that pending lit-book routes whose routing delay
        has elapsed are executed.

        :param t: current simulation timestep.
        """

        self._expire_stale_orders(t)
        self._process_pending_routes(t)

        mid = self.mid_price()
        if not np.isnan(mid):
            self._match_orders(mid, t)

        logging.info(
            f"---------- DARK POOL TICK ----------: t={t}, active_orders={len(self._order_index)}, "
            f"pending_routes={len(self.pending_lit_routes)}"
        )