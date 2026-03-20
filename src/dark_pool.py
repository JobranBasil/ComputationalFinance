import numpy as np
import logging

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Dict, Optional, Literal, Set, Any
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
        if not self.lit_order_book.bids or not self.lit_order_book.asks:
            logging.warning("-----WARNING-----: No orders in lit orderbook")
            return np.nan

        best_bid = self.lit_order_book.best_bid()
        best_ask = self.lit_order_book.best_ask()

        if best_bid is None or best_ask is None:
            logging.warning("-----WARNING-----: Best bid or ask is None")
            return np.nan

        if best_bid <= 0 or best_ask <= 0:
            logging.warning("-----WARNING-----: Best bid or ask is <= 0")
            return np.nan

        if not np.isfinite(best_bid) or not np.isfinite(best_ask):
            logging.warning("-----WARNING-----: Best bid or ask is not finite")

        if best_bid >= best_ask:
            logging.warning("-----WARNING-----: Best bid is >= best ask")
            return np.nan

        return (best_bid + best_ask) / 2

    def match_orders(self, mid_price: float, timestamp: int) -> list[Any] | None:
        # add string literal comment
        trades = List[Trade] = []

        while self.bids and self.asks:

            while self.bids and self.bids[0].order_id in self.cancelled_ids:
                # Remove cancelled orders from the front of the bids queue
                logging.info(f"-----CANCELLED ORDER REMOVED FROM DP BID QUEUE-----: Removing cancelled order from bids queue: {self.bids[0].order_id}")
                self._cancelled_ids.discard(self.bids.popleft().order_id)

            while self.asks and self.asks[0].order_id in self.cancelled_ids:
                # Remove cancelled orders from the front of the asks queue
                logging.info(f"-----CANCELLED ORDER REMOVED FROM DP ASK QUEUE-----: Removing cancelled order from asks queue: {self.asks[0].order_id}")
                self._cancelled_ids.discard(self.asks.popleft().order_id)

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


