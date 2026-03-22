import numpy as np
import logging

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Dict, Literal, Set, Any

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

    def __init__(self, lit_orderbook, max_resting_ticks: int = 10, routing_delay: int = 10, tape_delay: int = 5):
        self.lit_orderbook = lit_orderbook

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
        if not self.lit_orderbook.bids or not self.lit_orderbook.asks:
            logging.warning("-----WARNING-----: No orders in lit orderbook")
            return np.nan

        best_bid = self.lit_orderbook.best_bid()
        best_ask = self.lit_orderbook.best_ask()

        if best_bid is None or best_ask is None:
            logging.warning("-----WARNING-----: Best bid or ask is None")
            return np.nan

        if best_bid <= 0 or best_ask <= 0:
            logging.warning("-----WARNING-----: Best bid or ask is <= 0")
            return np.nan

        if not np.isfinite(best_bid) or not np.isfinite(best_ask):
            logging.warning("-----WARNING-----: Best bid or ask is not finite")
            return np.nan

        if best_bid >= best_ask:
            logging.warning("-----WARNING-----: Best bid is >= best ask")
            return np.nan

        return (best_bid + best_ask) / 2

    def match_orders(self, mid_price: float, timestamp: int) -> list[Any] | None:
        trades: List[Trade] = []

        while self.bids and self.asks:

            while self.bids and self.bids[0].order_id in self.cancelled_ids:
                # Remove cancelled orders from the front of the bids queue
                logging.info(f"-----CANCELLED ORDER REMOVED FROM DP BID QUEUE-----: Removing cancelled order from bids queue: {self.bids[0].order_id}")
                self.cancelled_ids.discard(self.bids.popleft().order_id)

            while self.asks and self.asks[0].order_id in self.cancelled_ids:
                # Remove cancelled orders from the front of the asks queue
                logging.info(f"-----CANCELLED ORDER REMOVED FROM DP ASK QUEUE-----: Removing cancelled order from asks queue: {self.asks[0].order_id}")
                self.cancelled_ids.discard(self.asks.popleft().order_id)

            if not self.bids or not self.asks:
                # No more orders to match
                logging.warning(f"-----WARNING-----: No more orders to match: bids: {len(self.bids)}, asks: {len(self.asks)}")
                break

            # Get the best bid and ask
            bid = self.bids[0]
            ask = self.asks[0]

            if bid.trader_id == ask.trader_id:
                top_ask_id = ask.order_id
                self.asks.rotate(-1)
                while self.asks[0].order_id != top_ask_id and self.asks[0].trader_id == bid.trader_id:
                    self.asks.rotate(-1)
                if self.asks[0].order_id == top_ask_id:
                    # No more matching asks for this bid
                    logging.warning(f"-----WARNING-----: No more matching asks for this bid: bid: {bid.trader_id}, ask: {ask.trader_id}")
                    break
                logging.warning(f"-----WARNING-----: self.asks[0].trader_id == bid.trader_id: bid: {bid.trader_id}, ask: {ask.trader_id}")
                continue
            trade_qty = min(bid.qty, ask.qty)

            if trade_qty <= 0:
                logging.warning(f"-----WARNING-----: trade_qty <= 0: bid: {bid.trader_id}, ask: {ask.trader_id}, qty: {trade_qty}")
                break

            # create a trade at the mid price with the matched quantity
            trade = Trade(
                price = mid_price,
                qty = trade_qty,
                timestamp = timestamp,
            )

            # record the trade in the tape and the list of trades for this tick
            trades.append(trade)
            self.trade_tape.append(trade)

            logging.info(f"-----DARK POOL TRADE -----: buyer: {bid.trader_id}, seller: {ask.trader_id}, price: {mid_price}, qty: {trade_qty}, timestamp: {timestamp}")

            # reduce the qty of the matched orders to account for partial fills
            bid.qty -= trade_qty
            ask.qty -= trade_qty

            # check if the matched orders are now empty
            if bid.qty == 0:
                self.order_index.pop(self.bids.popleft().order_id, None)
            if ask.qty == 0:
                self.order_index.pop(self.asks.popleft().order_id, None)

        return trades

    def submit_order(self, order: Order) -> list[Trade] | None:

        if order.qty <= 0:
            logging.warning(f"-----WARNING: INVALID ORDER QUANTITY-----: Order qty <= 0: order_id: {order.order_id}, qty: {order.qty}")
            return []

        if order.side not in ["buy", "sell"]:
            logging.warning(f"-----WARNING: INVALID ORDER SIDE-----: Invalid order side: order_id: {order.order_id}, side: {order.side}")
            return []

        if order.ts < 0:
            logging.warning(f"-----WARNING: INVALID ORDER TIMESTAMP-----: Order timestamp < 0: order_id: {order.order_id}, ts: {order.ts}")
            return []

        if order.order_id in self.order_index or any(o.order_id == order.order_id for _, o in self.pending_lit_routes):
            logging.warning(f"-----WARNING: DUPLICATE ORDER ID-----: Order ID already exists: order_id: {order.order_id}")
            return []

        if order.side == "buy":
            self.bids.append(order)

        if order.side == "sell":
            self.asks.append(order)

        self.order_index[order.order_id] = order

        logging.info(f"-----ORDER SUBMISSION-----: order_id: {order.order_id}, side: {order.side}, qty: {order.qty}, timestamp: {order.ts}")

        mid_price = self.compute_mid_price()

        if np.isnan(mid_price):
            logging.warning(f"-----WARNING: MID PRICE IS NOT AVAILABLE-----: Order submitted in queue but cannot be executed: order_id: {order.order_id}")
            return []

        return self.match_orders(mid_price, order.ts)

    def tick(self, t: int):
        self._process_pending_routes(t)
        self._expire_stale_orders(t)

        mid_price = self.compute_mid_price()

        if np.isnan(mid_price):
            logging.warning(f"-----WARNING: MID PRICE IS NOT AVAILABLE-----: Cannot execute orders in queue: bids: {len(self.bids)}, asks: {len(self.asks)}")
            return
        self.match_orders(mid_price, t)
        logging.info(f"-----DARK POOL TICK-----: t={t}, active_orders={len(self.order_index)}, pending_routes={len(self.pending_lit_routes)}")

    def cancel_order(self, order_id: int) -> bool:
        if order_id not in self.order_index:
            logging.warning(f"-----WARNING: ORDER NOT FOUND-----: Order not found in order index: order_id: {order_id}")
            return False

        self.cancelled_ids.add(order_id)

        del self.order_index[order_id]
        logging.info(f"-----ORDER CANCELLED-----: order_id: {order_id}")
        return True

    def has_order(self) -> bool:
        return len(self.order_index) > 0

    def queue_depth(self) -> tuple[int, int]:
        bid_qty = sum(
            order.qty for order in self.order_index.values() if order.side == "buy"
        )
        ask_qty = sum(
            order.qty for order in self.order_index.values() if order.side == "sell"
        )

        if bid_qty < 0:
            logging.warning(f"-----WARNING: INVALID BID QTY-----: Negative queue depth detected: bid_qty: {bid_qty}")
            return 0, 0
        if ask_qty < 0:
            logging.warning(f"-----WARNING: INVALID ASK QTY-----: Negative queue depth detected: ask_qty: {ask_qty}")
            return 0, 0

        return bid_qty, ask_qty

    def recent_volume(self, current_ts: int, lookback: int) -> int:
        lookback_start = current_ts - self.tape_delay
        window_start = lookback_start - lookback

        if lookback < 0:
            logging.warning(f"-----WARNING: INVALID LOOKBACK PERIOD-----: Lookback period cannot be negative: lookback: {lookback}")
            return 0

        if lookback_start < 0:
            logging.warning(f"-----WARNING: INVALID LOOKBACK PERIOD-----: Lookback period cannot be negative: lookback_start: {lookback_start}")
            return 0

        if window_start < 0:
            logging.warning(f"-----WARNING: INVALID LOOKBACK PERIOD-----: Lookback period cannot be negative: window_start: {window_start}")
            return 0

        return  int(
            sum(
                trade.qty for trade in self.trade_tape if window_start <= trade.timestamp <= lookback_start
            )
        )

    def _process_pending_routes(self, current_ts: int):
        pending_orders = [
            (ts,order) for ts, order in self.pending_lit_routes if current_ts >= ts
        ]
        self.pending_lit_routes = [
            (ts, order) for ts, order in self.pending_lit_routes if current_ts < ts
        ]

        for execute_at, lit_order in pending_orders:
            original_qty = lit_order.qty
            lit_trades = self.lit_orderbook.execute_market(lit_order)
            filled_qty = sum(trade.qty for trade in lit_trades)
            unfilled_qty = original_qty - filled_qty
            logging.info(
                f"-----PENDING ROUTE EXECUTED-----: order_id: {lit_order.order_id}, trader_id: {lit_order.trader_id}, side: {lit_order.side}, qty: {original_qty}, filled: {filled_qty}, unfilled: {unfilled_qty}, scheduled_at: {execute_at}, executed_at: {current_ts}"
            )
            if unfilled_qty > 0:
                logging.warning(
                    f"-----PENDING ROUTE PARTIALLY FILLED-----: order_id: {lit_order.order_id}, unfilled_qty: {unfilled_qty} (lit book too thin)"
                )

    def _expire_stale_orders(self, current_ts: int) -> List[LitOrder]:
        routed_orders: List[LitOrder] = []

        for queue in (self.bids, self.asks):
            remaining: Deque[Order] = deque()
            while queue:
                order = queue.popleft()

                if order.order_id in self.cancelled_ids:
                    self.cancelled_ids.discard(order.order_id)
                    continue
                order_age = current_ts - order.ts

                if order_age >= self.max_resting_ticks:
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
                    self.order_index.pop(order.order_id, None)

                    logging.info(
                        f"-----STALE DARK POOL ORDER SCHEDULED FOR LIT BOOK-----: order_id: {order.order_id}, trader_id: {order.trader_id}, side: {order.side}, qty: {order.qty}, age: {order_age} ticks, execute_at: {execute_at}"
                    )
                    routed_orders.append(lit_order)
                else:
                    remaining.append(order)
            queue.extend(remaining)

        return routed_orders


